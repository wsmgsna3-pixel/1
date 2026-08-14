# -*- coding: utf-8 -*-
"""科技股周线SKDJ持续分离买点、退出与Top3审计 V4.4。

本版只验证一条可执行买入规则：
1. 周线SKDJ在25以下首次形成独立底部金叉，进入观察状态；
2. 不设等待上限，K首次从25下方上穿25仅为预确认；
3. 再观察一个完整周：K、D同时上升、K>D、K保持在25上方且K-D差值扩大；
4. 持续分离周结束后的下一市场交易日开盘买入。

历史前三个已完成金叉波段冲击65/70/75的次数只用于透明排序，不作为
硬条件。所有退出策略从持续分离后的真实次周开盘价重新计算。
"""
from __future__ import annotations

import io
import math
import os
import pickle
import shutil
import time
import zipfile
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts
TITLE = "科技股周线SKDJ持续分离买点与Top3审计 V4.4"
VERSION = "V4.4-WEEKLY-SKDJ-PERSISTENT-SEPARATION-ENTRY"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")

SKDJ_N = 9
SKDJ_M = 3
SKDJ_BOTTOM = 25.0
INDICATOR_WARMUP_WEEKS = 40
HOLD_20D = 20
HOLD_40D = 40
AUDIT_WEEKS = 8
HISTORY_PEAK_LEVELS = (65.0, 70.0, 75.0)
FIRST_HIT_PROFIT_LEVELS = (10.0, 15.0, 20.0)
STOP_LOSS_PCT = 10.0
TAKE_PROFITS = (10.0, 15.0, 20.0)
ACTIVATED_TRAILS = ((10.0, 10.0), (15.0, 10.0))
TOP_K = 3

CORE_TECH_L1 = {"电子", "计算机", "通信", "国防军工"}
EXTENDED_TECH_L1 = {"机械设备", "电力设备", "医药生物", "汽车", "基础化工", "有色金属", "建筑材料"}
TECH_INDUSTRY_KEYWORDS = {
    "半导体", "电子元件", "元件", "光学光电子", "消费电子", "电子化学品",
    "计算机设备", "软件开发", "IT服务", "通信设备", "军工电子", "航空装备",
    "航天装备", "自动化设备", "机器人", "激光设备", "工控设备", "仪器仪表",
    "电池", "光伏设备", "风电设备", "电网设备", "电机", "医疗器械",
    "生物制品", "汽车电子", "金属新材料", "非金属材料", "膜材料", "碳纤维",
}
BOARDS = ("主板", "创业板", "科创板")

pro = None
API_ERRORS: list[str] = []


def normalize_date(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip().replace("-", "").replace("/", "")
    if text.endswith(".0"):
        text = text[:-2]
    return text[:8] if len(text) >= 8 and text[:8].isdigit() else default


def finite_num(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def record_error(message: str) -> None:
    if len(API_ERRORS) < 300:
        API_ERRORS.append(message)


def validate_dates(signal_start: date, signal_end: date, market_end: date) -> str:
    if signal_start >= signal_end:
        return "信号开始日期必须早于信号截止日期"
    if market_end <= signal_end:
        return "行情观察截止日期必须晚于信号截止日期"
    return ""


def safe_get(func_name: str, retries: int = 3, required: bool = False, **kwargs) -> pd.DataFrame:
    global pro
    if pro is None:
        if required:
            raise RuntimeError("Tushare尚未初始化")
        return pd.DataFrame()
    try:
        func = getattr(pro, func_name)
    except AttributeError as exc:
        if required:
            raise RuntimeError(f"当前Tushare SDK不支持{func_name}") from exc
        record_error(f"缺少接口 {func_name}")
        return pd.DataFrame()
    last_error = None
    for attempt in range(retries):
        try:
            result = func(**kwargs)
            return pd.DataFrame() if result is None else result
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    message = f"{func_name}失败: {last_error}"
    record_error(message)
    if required:
        raise RuntimeError(message)
    return pd.DataFrame()


def atomic_pickle(payload: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = f"{path}.tmp"
    with open(temp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(files: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in files.items():
            archive.writestr(name, csv_bytes(frame))
    return buffer.getvalue()


@st.cache_data(ttl=24 * 3600)
def load_stock_basic() -> pd.DataFrame:
    frames = []
    fields = "ts_code,symbol,name,market,exchange,list_status,list_date,delist_date"
    for status in ("L", "P", "D"):
        frame = safe_get("stock_basic", list_status=status, fields=fields)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("stock_basic加载失败")
    result = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
    result = result[
        result["market"].isin(BOARDS)
        & result["exchange"].ne("BSE")
        & ~result["ts_code"].astype(str).str.endswith(".BJ", na=False)
        & ~result["name"].astype(str).str.contains("ST|退", na=False)
    ].copy()
    result["list_date"] = result["list_date"].map(lambda x: normalize_date(x, "19000101"))
    result["delist_date"] = result["delist_date"].map(lambda x: normalize_date(x, "99991231"))
    return result


def is_tech_industry(row: pd.Series) -> bool:
    l1, l2, l3 = str(row.get("l1_name", "")), str(row.get("l2_name", "")), str(row.get("l3_name", ""))
    if l1 in CORE_TECH_L1:
        return True
    return l1 in EXTENDED_TECH_L1 and any(word in f"{l2}|{l3}" for word in TECH_INDUSTRY_KEYWORDS)


@st.cache_data(ttl=7 * 24 * 3600)
def load_tech_memberships(api_pause: float) -> pd.DataFrame:
    levels = safe_get("index_classify", required=True, level="L1", src="SW2021")
    targets = levels[levels["industry_name"].isin(CORE_TECH_L1 | EXTENDED_TECH_L1)]
    if targets.empty:
        raise RuntimeError("未找到申万2021目标行业")
    frames = []
    jobs = [(str(row.index_code), str(row.industry_name), flag)
            for row in targets.itertuples(index=False) for flag in ("Y", "N")]
    progress = st.progress(0.0, text="构建申万历史科技池...")
    for number, (code, name, flag) in enumerate(jobs, start=1):
        frame = safe_get("index_member_all", l1_code=code, is_new=flag)
        if not frame.empty:
            if "ts_code" not in frame.columns and "con_code" in frame.columns:
                frame = frame.rename(columns={"con_code": "ts_code"})
            frames.append(frame)
        progress.progress(number / max(len(jobs), 1), text=f"行业池：{name} {flag}")
        time.sleep(api_pause)
    progress.empty()
    if not frames:
        raise RuntimeError("index_member_all未返回数据，请检查权限与SDK版本")
    result = pd.concat(frames, ignore_index=True)
    for column in ("ts_code", "l1_name", "l2_name", "l3_name", "in_date", "out_date"):
        if column not in result.columns:
            result[column] = ""
    result = result[result.apply(is_tech_industry, axis=1)].copy()
    result["in_date"] = result["in_date"].map(lambda x: normalize_date(x, "19000101"))
    result["out_date"] = result["out_date"].map(lambda x: normalize_date(x, "99991231"))
    return result.drop_duplicates(["ts_code", "l1_name", "l2_name", "l3_name", "in_date", "out_date"])


def build_period_index(memberships: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in memberships.itertuples(index=False):
        result.setdefault(str(row.ts_code), []).append({
            "in_date": str(row.in_date), "out_date": str(row.out_date),
            "l1": str(row.l1_name), "l2": str(row.l2_name), "l3": str(row.l3_name),
        })
    return result


def membership_on_date(periods: list[dict[str, str]], trade_date: str) -> dict[str, str] | None:
    for period in periods:
        if period["in_date"] <= trade_date < period["out_date"]:
            return period
    return None


def sample_board(row: pd.Series) -> str:
    market = str(row.get("market", ""))
    if market in BOARDS:
        return market
    code = str(row.get("ts_code", ""))
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith(("688", "689")):
        return "科创板"
    return "主板"


@st.cache_data(ttl=24 * 3600)
def load_trade_calendar(start_date: str, end_date: str) -> list[str]:
    frame = safe_get("trade_cal", required=True, exchange="SSE", start_date=start_date, end_date=end_date)
    if frame.empty:
        raise RuntimeError("交易日历为空")
    return sorted(frame.loc[frame["is_open"].eq(1), "cal_date"].astype(str).tolist())


@st.cache_data(ttl=24 * 3600)
def stock_cache_path(ts_code: str, start_date: str, end_date: str) -> str:
    return os.path.join(CACHE_DIR, f"{ts_code.replace('.', '_')}_{start_date}_{end_date}.pkl")


def fetch_pro_bar(ts_code: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    last_error = None
    for attempt in range(retries):
        try:
            frame = ts.pro_bar(api=pro, ts_code=ts_code, start_date=start_date, end_date=end_date,
                               adj="qfq", freq="D", factors=["tor"])
            return pd.DataFrame() if frame is None else frame
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    record_error(f"pro_bar {ts_code}失败: {last_error}")
    return pd.DataFrame()


def fetch_stock_history(ts_code: str, start_date: str, end_date: str,
                        use_cache: bool, api_pause: float) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    path = stock_cache_path(ts_code, start_date, end_date)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            return payload.get("daily", pd.DataFrame()), payload.get("basic", pd.DataFrame()), True
        except Exception as exc:
            record_error(f"缓存损坏 {ts_code}: {exc}")
    daily = fetch_pro_bar(ts_code, start_date, end_date)
    time.sleep(api_pause)
    basic = safe_get("daily_basic", ts_code=ts_code, start_date=start_date, end_date=end_date,
                     fields="ts_code,trade_date,close,circ_mv,turnover_rate")
    time.sleep(api_pause)
    if not daily.empty:
        for column in ("open", "high", "low", "close", "vol"):
            daily[column] = pd.to_numeric(daily.get(column), errors="coerce")
        daily["trade_date"] = daily["trade_date"].astype(str)
        daily = daily.dropna(subset=["trade_date", "open", "high", "low", "close"])
        daily = daily.drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)
    if not basic.empty:
        basic["trade_date"] = basic["trade_date"].astype(str)
        for column in ("close", "circ_mv", "turnover_rate"):
            basic[column] = pd.to_numeric(basic.get(column), errors="coerce")
        basic = basic.drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)
    if use_cache and not daily.empty:
        atomic_pickle({"daily": daily, "basic": basic}, path)
    return daily, basic, False


def complete_week_last_dates(open_dates: list[str]) -> dict[pd.Timestamp, str]:
    frame = pd.DataFrame({"trade_date": open_dates})
    frame["dt"] = pd.to_datetime(frame["trade_date"])
    frame["week_label"] = frame["dt"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    return frame.groupby("week_label")["trade_date"].max().to_dict()


def add_skdj(frame: pd.DataFrame, n: int = SKDJ_N, m: int = SKDJ_M) -> pd.DataFrame:
    work = frame.copy()
    lowv = work["low"].rolling(int(n), min_periods=int(n)).min()
    highv = work["high"].rolling(int(n), min_periods=int(n)).max()
    raw = (work["close"] - lowv) / (highv - lowv).replace(0, np.nan) * 100.0
    rsv = raw.ewm(span=int(m), adjust=False, min_periods=1).mean()
    work["SKDJ_K"] = rsv.ewm(span=int(m), adjust=False, min_periods=1).mean()
    work["SKDJ_D"] = work["SKDJ_K"].rolling(int(m), min_periods=int(m)).mean()
    work["SKDJ_Golden_Cross"] = work["SKDJ_K"].gt(work["SKDJ_D"]) & work["SKDJ_K"].shift(1).le(work["SKDJ_D"].shift(1))
    work["SKDJ_Death_Cross"] = work["SKDJ_K"].lt(work["SKDJ_D"]) & work["SKDJ_K"].shift(1).ge(work["SKDJ_D"].shift(1))
    work["SKDJ_Level"] = (work["SKDJ_K"] + work["SKDJ_D"]) / 2.0
    work["SKDJ_KD_Spread"] = work["SKDJ_K"] - work["SKDJ_D"]
    return work


def aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["dt"] = pd.to_datetime(work["trade_date"])
    return work.set_index("dt").resample("W-FRI").agg({
        "trade_date": "last", "open": "first", "high": "max", "low": "min",
        "close": "last", "vol": "sum",
    }).dropna(subset=["close"]).reset_index().rename(columns={"dt": "week_label"})


def build_complete_weekly(daily: pd.DataFrame, week_last_map: dict[pd.Timestamp, str]) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    weekly = aggregate_weekly(daily)
    weekly["calendar_week_last"] = weekly["week_label"].map(week_last_map)
    weekly = weekly[
        weekly["calendar_week_last"].notna()
        & weekly["trade_date"].astype(str).eq(weekly["calendar_week_last"].astype(str))
    ].copy().reset_index(drop=True)
    return add_skdj(weekly) if not weekly.empty else weekly


def add_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    work = add_skdj(daily.copy().sort_values("trade_date").reset_index(drop=True)).rename(columns={
        "SKDJ_K": "D_SKDJ_K", "SKDJ_D": "D_SKDJ_D",
        "SKDJ_Golden_Cross": "D_SKDJ_Golden_Cross",
    })
    work["D_MA60_Bias_pct"] = (work["close"] / work["close"].rolling(60).mean() - 1.0) * 100.0
    work["D_SKDJ_Level"] = (work["D_SKDJ_K"] + work["D_SKDJ_D"]) / 2.0
    work["D_SKDJ_Death_Cross"] = (
        work["D_SKDJ_K"].lt(work["D_SKDJ_D"])
        & work["D_SKDJ_K"].shift(1).ge(work["D_SKDJ_D"].shift(1))
    )
    return work


def market_snapshot(basic: pd.DataFrame, signal_date: str) -> dict[str, float]:
    row = basic[basic["trade_date"].astype(str).eq(signal_date)] if not basic.empty else pd.DataFrame()
    if row.empty:
        return {"Raw_Close": np.nan, "Circ_MV_Billion": np.nan, "Turnover_Rate": np.nan}
    item = row.iloc[-1]
    return {
        "Raw_Close": finite_num(item.get("close")),
        "Circ_MV_Billion": finite_num(item.get("circ_mv")) / 10000.0,
        "Turnover_Rate": finite_num(item.get("turnover_rate")),
    }


def signal_filter(snapshot: dict[str, float], min_price: float, min_mv: float) -> tuple[bool, str]:
    if not math.isfinite(snapshot["Raw_Close"]):
        return False, "缺少信号日原始收盘价"
    if snapshot["Raw_Close"] < min_price:
        return False, "低于最低股价"
    if not math.isfinite(snapshot["Circ_MV_Billion"]):
        return False, "缺少历史流通市值"
    if snapshot["Circ_MV_Billion"] < min_mv:
        return False, "低于最低流通市值"
    return True, ""


def is_main_board(ts_code: str) -> bool:
    return not str(ts_code).startswith(("300", "301", "688", "689"))


def market_week_sequence(open_dates: list[str]) -> list[tuple[pd.Period, str]]:
    frame = pd.DataFrame({"trade_date": open_dates})
    frame["period"] = pd.to_datetime(frame["trade_date"]).dt.to_period("W-FRI")
    return [(period, str(group["trade_date"].max())) for period, group in frame.groupby("period", sort=True)]


def prefix_keys(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def first_hit_label(path: pd.DataFrame, raw_entry: float, profit_pct: float) -> tuple[str, str]:
    upper, lower = raw_entry * (1.0 + profit_pct / 100.0), raw_entry * 0.90
    for row in path.itertuples(index=False):
        hit_up = finite_num(getattr(row, "high", np.nan)) >= upper
        hit_down = finite_num(getattr(row, "low", np.nan)) <= lower
        if hit_up and hit_down:
            return f"同日同时触发_保守按-10%先", str(getattr(row, "trade_date", ""))
        if hit_down:
            return "先到-10%", str(getattr(row, "trade_date", ""))
        if hit_up:
            return f"先到+{int(profit_pct)}%", str(getattr(row, "trade_date", ""))
    return f"W{AUDIT_WEEKS}内均未触发", ""


def entry_outcomes(daily: pd.DataFrame, signal_date: str, ts_code: str,
                   open_dates: list[str], open_pos: dict[str, int],
                   market_weeks: list[tuple[pd.Period, str]],
                   config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "Tradable": False, "Reason": "", "Entry_Date": "", "Raw_Entry_Open": np.nan,
        "Net_Entry_Price": np.nan, "Has_20D": False, "Has_40D": False,
        "Return_20D_Net_pct": np.nan, "Return_40D_Net_pct": np.nan,
        "MFE_20D_Raw_pct": np.nan, "MAE_20D_Raw_pct": np.nan,
        "MFE_40D_Raw_pct": np.nan, "MAE_40D_Raw_pct": np.nan,
        "First_Hit_10_vs_Minus10_W8": "", "First_Hit_10_Date_W8": "",
        "First_Hit_15_vs_Minus10_W8": "", "First_Hit_15_Date_W8": "",
        "First_Hit_20_vs_Minus10_W8": "", "First_Hit_20_Date_W8": "",
    }
    for week in range(1, AUDIT_WEEKS + 1):
        out.update({
            f"Has_W{week}": False, f"W{week}_End_Date": "",
            f"W{week}_Cum_Max_High_Raw_pct": np.nan,
            f"W{week}_Cum_MFE_Net_pct": np.nan,
            f"W{week}_Cum_MAE_Raw_pct": np.nan,
            f"W{week}_Close_Return_Net_pct": np.nan,
        })
    if signal_date not in open_pos or open_pos[signal_date] + 1 >= len(open_dates):
        out["Reason"] = "未来市场交易日不足"
        return out
    entry_market_pos = open_pos[signal_date] + 1
    entry_date = open_dates[entry_market_pos]
    out["Entry_Date"] = entry_date
    rows = daily[daily["trade_date"].astype(str).eq(entry_date)]
    if rows.empty:
        out["Reason"] = "下一市场交易日停牌或无行情"
        return out
    first = rows.iloc[-1]
    if is_main_board(ts_code) and float(first["open"]) == float(first["high"]) == float(first["low"]):
        out["Reason"] = "主板下一交易日一字板"
        return out
    raw_entry = float(first["open"])
    if not math.isfinite(raw_entry) or raw_entry <= 0:
        out["Reason"] = "开盘价无效"
        return out
    buy_cost = (config["commission_pct"] + config["transfer_fee_pct"]) / 100.0
    sell_cost = (config["commission_pct"] + config["transfer_fee_pct"] + config["stamp_duty_pct"]) / 100.0
    buy_factor = (1 + config["buy_slippage_pct"] / 100.0) * (1 + buy_cost)
    sell_factor = (1 - config["sell_slippage_pct"] / 100.0) * (1 - sell_cost)
    net_entry = raw_entry * buy_factor
    out.update({"Tradable": True, "Raw_Entry_Open": raw_entry, "Net_Entry_Price": net_entry})

    entry_period = pd.Timestamp(entry_date).to_period("W-FRI")
    future_weeks = [(period, end_date) for period, end_date in market_weeks if period >= entry_period]
    for week in range(1, AUDIT_WEEKS + 1):
        if len(future_weeks) < week:
            continue
        end_date = future_weeks[week - 1][1]
        path = daily[daily["trade_date"].astype(str).between(entry_date, end_date)].sort_values("trade_date")
        if path.empty:
            continue
        high, low, close = float(path["high"].max()), float(path["low"].min()), float(path.iloc[-1]["close"])
        out.update({
            f"Has_W{week}": True, f"W{week}_End_Date": end_date,
            f"W{week}_Cum_Max_High_Raw_pct": (high / raw_entry - 1.0) * 100.0,
            f"W{week}_Cum_MFE_Net_pct": (high * sell_factor / net_entry - 1.0) * 100.0,
            f"W{week}_Cum_MAE_Raw_pct": (low / raw_entry - 1.0) * 100.0,
            f"W{week}_Close_Return_Net_pct": (close * sell_factor / net_entry - 1.0) * 100.0,
        })
        if week == AUDIT_WEEKS:
            for profit_pct in FIRST_HIT_PROFIT_LEVELS:
                label, hit_date = first_hit_label(path, raw_entry, profit_pct)
                key = int(profit_pct)
                out[f"First_Hit_{key}_vs_Minus10_W8"] = label
                out[f"First_Hit_{key}_Date_W8"] = hit_date

    for days in (HOLD_20D, HOLD_40D):
        end_pos = entry_market_pos + days - 1
        if end_pos >= len(open_dates):
            continue
        end_date = open_dates[end_pos]
        path = daily[daily["trade_date"].astype(str).between(entry_date, end_date)].sort_values("trade_date")
        if path.empty:
            continue
        out[f"Has_{days}D"] = True
        out[f"Return_{days}D_Net_pct"] = (float(path.iloc[-1]["close"]) * sell_factor / net_entry - 1.0) * 100.0
        out[f"MFE_{days}D_Raw_pct"] = (float(path["high"].max()) / raw_entry - 1.0) * 100.0
        out[f"MAE_{days}D_Raw_pct"] = (float(path["low"].min()) / raw_entry - 1.0) * 100.0
    if not out[f"Has_W{AUDIT_WEEKS}"]:
        out["Reason"] = f"可买但未来不足{AUDIT_WEEKS}个完整市场周"
    return out


def trade_factors(config: dict[str, Any]) -> tuple[float, float]:
    """Return buy/sell multipliers including slippage and explicit fees."""
    buy_cost = (config["commission_pct"] + config["transfer_fee_pct"]) / 100.0
    sell_cost = (
        config["commission_pct"] + config["transfer_fee_pct"]
        + config["stamp_duty_pct"]
    ) / 100.0
    return (
        (1.0 + config["buy_slippage_pct"] / 100.0) * (1.0 + buy_cost),
        (1.0 - config["sell_slippage_pct"] / 100.0) * (1.0 - sell_cost),
    )


def exit_fields(path: pd.DataFrame, raw_entry: float, raw_exit: float,
                exit_date: str, trigger: str, config: dict[str, Any]) -> dict[str, Any]:
    buy_factor, sell_factor = trade_factors(config)
    net_return = (raw_exit * sell_factor / (raw_entry * buy_factor) - 1.0) * 100.0
    holding_days = 0
    if not path.empty:
        dates = path["trade_date"].astype(str).tolist()
        holding_days = dates.index(exit_date) + 1 if exit_date in dates else len(dates)
    return {
        "Available": True, "Exit_Date": exit_date, "Raw_Exit_Price": raw_exit,
        "Trigger": trigger, "Holding_Trading_Days": holding_days,
        "Net_Return_pct": net_return,
    }


def unavailable_exit(reason: str) -> dict[str, Any]:
    return {
        "Available": False, "Exit_Date": "", "Raw_Exit_Price": np.nan,
        "Trigger": reason, "Holding_Trading_Days": np.nan,
        "Net_Return_pct": np.nan,
    }


def simulate_bracket(path: pd.DataFrame, raw_entry: float, take_profit: float,
                     config: dict[str, Any]) -> dict[str, Any]:
    """Fixed -10% stop and take-profit; same-day double hit is conservatively a stop."""
    if path.empty:
        return unavailable_exit("无W8路径")
    stop_price = raw_entry * (1.0 - STOP_LOSS_PCT / 100.0)
    target_price = raw_entry * (1.0 + take_profit / 100.0)
    for row in path.itertuples(index=False):
        trade_date = str(row.trade_date)
        day_open, day_low, day_high = float(row.open), float(row.low), float(row.high)
        stop_hit, target_hit = day_low <= stop_price, day_high >= target_price
        if stop_hit:
            raw_exit = day_open if day_open < stop_price else stop_price
            label = (
                f"同日双触发_保守止损-{int(STOP_LOSS_PCT)}%"
                if target_hit else f"止损-{int(STOP_LOSS_PCT)}%"
            )
            return exit_fields(path, raw_entry, raw_exit, trade_date, label, config)
        if target_hit:
            raw_exit = day_open if day_open > target_price else target_price
            return exit_fields(
                path, raw_entry, raw_exit, trade_date,
                f"止盈+{int(take_profit)}%", config)
    last = path.iloc[-1]
    return exit_fields(
        path, raw_entry, float(last["close"]), str(last["trade_date"]),
        f"W{AUDIT_WEEKS}期末", config)


def simulate_activated_trail(path: pd.DataFrame, raw_entry: float, activation: float,
                             trail: float, config: dict[str, Any]) -> dict[str, Any]:
    """Use a fixed -10% stop until activated, then trail from prior-day peak only."""
    if path.empty:
        return unavailable_exit("无W8路径")
    fixed_stop = raw_entry * (1.0 - STOP_LOSS_PCT / 100.0)
    activation_price = raw_entry * (1.0 + activation / 100.0)
    prior_peak = raw_entry
    armed = False
    for row in path.itertuples(index=False):
        trade_date = str(row.trade_date)
        day_open, day_low, day_high = float(row.open), float(row.low), float(row.high)
        if armed:
            stop_price = max(raw_entry, prior_peak * (1.0 - trail / 100.0))
        else:
            stop_price = fixed_stop
        if day_low <= stop_price:
            raw_exit = day_open if day_open < stop_price else stop_price
            trigger = (
                f"激活+{int(activation)}%后回撤{int(trail)}%"
                if armed else f"激活前止损-{int(STOP_LOSS_PCT)}%"
            )
            return exit_fields(path, raw_entry, raw_exit, trade_date, trigger, config)
        prior_peak = max(prior_peak, day_high)
        if prior_peak >= activation_price:
            armed = True
    last = path.iloc[-1]
    return exit_fields(
        path, raw_entry, float(last["close"]), str(last["trade_date"]),
        f"W{AUDIT_WEEKS}期末", config)


def simulate_exit_policies(daily: pd.DataFrame, outcome: dict[str, Any],
                           config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not to_bool(outcome.get("Tradable")):
        for policy in (
            "Fixed_W5", "Fixed_W8", "SL10_TP10", "SL10_TP15", "SL10_TP20",
            "Activate10_Trail10", "Activate15_Trail10",
        ):
            result.update(prefix_keys(unavailable_exit("不可交易"), policy))
        return result
    raw_entry = finite_num(outcome.get("Raw_Entry_Open"))
    entry_date = str(outcome.get("Entry_Date", ""))
    for week in (5, AUDIT_WEEKS):
        policy = f"Fixed_W{week}"
        if not to_bool(outcome.get(f"Has_W{week}")):
            result.update(prefix_keys(unavailable_exit(f"未来不足W{week}"), policy))
            continue
        end_date = str(outcome[f"W{week}_End_Date"])
        path = daily[daily["trade_date"].astype(str).between(entry_date, end_date)].sort_values("trade_date")
        last = path.iloc[-1]
        result.update(prefix_keys(exit_fields(
            path, raw_entry, float(last["close"]), str(last["trade_date"]),
            f"固定W{week}期末", config), policy))
    if not to_bool(outcome.get(f"Has_W{AUDIT_WEEKS}")):
        for policy in (
            "SL10_TP10", "SL10_TP15", "SL10_TP20",
            "Activate10_Trail10", "Activate15_Trail10",
        ):
            result.update(prefix_keys(unavailable_exit(f"未来不足W{AUDIT_WEEKS}"), policy))
        return result
    end_date = str(outcome[f"W{AUDIT_WEEKS}_End_Date"])
    path = daily[daily["trade_date"].astype(str).between(entry_date, end_date)].sort_values("trade_date")
    for take_profit in TAKE_PROFITS:
        policy = f"SL10_TP{int(take_profit)}"
        result.update(prefix_keys(
            simulate_bracket(path, raw_entry, take_profit, config), policy))
    for activation, trail in ACTIVATED_TRAILS:
        policy = f"Activate{int(activation)}_Trail{int(trail)}"
        result.update(prefix_keys(
            simulate_activated_trail(path, raw_entry, activation, trail, config), policy))
    return result


def completed_golden_swings(weekly: pd.DataFrame) -> list[dict[str, Any]]:
    """Return completed ordinary golden-cross swings using only cross-to-death data."""
    swings: list[dict[str, Any]] = []
    active_position: int | None = None
    for position in range(INDICATOR_WARMUP_WEEKS, len(weekly)):
        row = weekly.iloc[position]
        if active_position is None and to_bool(row.get("SKDJ_Golden_Cross")):
            active_position = position
            continue
        if active_position is None or not to_bool(row.get("SKDJ_Death_Cross")):
            continue
        segment = weekly.iloc[active_position:position + 1]
        peak_index = int(pd.to_numeric(segment["SKDJ_K"], errors="coerce").idxmax())
        swings.append({
            "Golden_Position": active_position,
            "Golden_Date": str(weekly.iloc[active_position]["trade_date"]),
            "Death_Position": position,
            "Death_Date": str(row["trade_date"]),
            "Peak_K": finite_num(segment["SKDJ_K"].max()),
            "Peak_D": finite_num(segment["SKDJ_D"].max()),
            "Peak_K_Date": str(weekly.loc[peak_index, "trade_date"]),
            "Swing_Weeks": int(position - active_position),
        })
        active_position = None
    return swings


def build_bottom_cycles(weekly: pd.DataFrame) -> list[dict[str, Any]]:
    """Collapse repeated crosses below 25 into one observable bottom cycle."""
    cycles: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for position in range(INDICATOR_WARMUP_WEEKS, len(weekly)):
        row = weekly.iloc[position]
        k, d = finite_num(row.get("SKDJ_K")), finite_num(row.get("SKDJ_D"))
        if not (math.isfinite(k) and math.isfinite(d)):
            continue
        low_cross = to_bool(row.get("SKDJ_Golden_Cross")) and k <= SKDJ_BOTTOM and d <= SKDJ_BOTTOM
        if active is None:
            if not low_cross:
                continue
            active = {
                "Anchor_Position": position,
                "Trigger_Position": None,
                "Bottom_Min_K": k,
                "Bottom_Min_D": d,
                "Bottom_Min_Level": (k + d) / 2.0,
                "Weeks_Both_Below25": 1,
                "Bottom_Golden_Cross_Count": 1,
            }
            continue

        active["Bottom_Min_K"] = min(float(active["Bottom_Min_K"]), k)
        active["Bottom_Min_D"] = min(float(active["Bottom_Min_D"]), d)
        active["Bottom_Min_Level"] = min(float(active["Bottom_Min_Level"]), (k + d) / 2.0)
        if k <= SKDJ_BOTTOM and d <= SKDJ_BOTTOM:
            active["Weeks_Both_Below25"] = int(active["Weeks_Both_Below25"]) + 1
        if low_cross:
            active["Bottom_Golden_Cross_Count"] = int(active["Bottom_Golden_Cross_Count"]) + 1

        previous = weekly.iloc[position - 1]
        previous_k = finite_num(previous.get("SKDJ_K"))
        crossed_25 = math.isfinite(previous_k) and previous_k < SKDJ_BOTTOM <= k
        confirmed = k > d and k > previous_k
        if crossed_25 and confirmed:
            active["Trigger_Position"] = position
            cycles.append(active)
            active = None

    if active is not None:
        cycles.append(active)
    return cycles


def prior_swing_features(swings: list[dict[str, Any]], anchor_position: int) -> dict[str, Any]:
    known = [swing for swing in swings if int(swing["Death_Position"]) < anchor_position]
    recent = list(reversed(known[-3:]))
    result: dict[str, Any] = {"Prior_Swings_Available": len(recent)}
    peaks: list[float] = []
    for number in range(1, 4):
        if number <= len(recent):
            swing = recent[number - 1]
            peak = finite_num(swing["Peak_K"])
            peaks.append(peak)
            result.update({
                f"Prev_Swing{number}_Golden_Date": swing["Golden_Date"],
                f"Prev_Swing{number}_Death_Date": swing["Death_Date"],
                f"Prev_Swing{number}_Peak_K": peak,
                f"Prev_Swing{number}_Peak_K_Date": swing["Peak_K_Date"],
                f"Prev_Swing{number}_Weeks": swing["Swing_Weeks"],
            })
        else:
            result.update({
                f"Prev_Swing{number}_Golden_Date": "",
                f"Prev_Swing{number}_Death_Date": "",
                f"Prev_Swing{number}_Peak_K": np.nan,
                f"Prev_Swing{number}_Peak_K_Date": "",
                f"Prev_Swing{number}_Weeks": np.nan,
            })
    valid_peaks = [peak for peak in peaks if math.isfinite(peak)]
    result["Prior_3_Peak_K_Mean"] = float(np.mean(valid_peaks)) if valid_peaks else np.nan
    result["Prior_3_Peak_K_Min"] = float(np.min(valid_peaks)) if valid_peaks else np.nan
    result["Prior_3_Peak_K_Max"] = float(np.max(valid_peaks)) if valid_peaks else np.nan
    for level in HISTORY_PEAK_LEVELS:
        result[f"Prior_3_Count_Peak_GE{int(level)}"] = sum(peak >= level for peak in valid_peaks)
        result[f"Prev_Swing1_Peak_GE{int(level)}"] = (
            bool(valid_peaks and valid_peaks[0] >= level) if recent else False)
    return result


def post_confirmation_features(weekly: pd.DataFrame, trigger_position: int) -> dict[str, Any]:
    """W1 is the delayed entry confirmation; W2 remains a future diagnostic only."""
    result: dict[str, Any] = {}
    previous_position = trigger_position
    for offset in (1, 2):
        prefix = f"Post_Confirm_W{offset}"
        position = trigger_position + offset
        if position >= len(weekly):
            result.update({
                f"{prefix}_Available": False, f"{prefix}_Date": "",
                f"{prefix}_K": np.nan, f"{prefix}_D": np.nan,
                f"{prefix}_KD_Spread": np.nan, f"{prefix}_Spread_Change": np.nan,
                f"{prefix}_K_Change": np.nan, f"{prefix}_D_Change": np.nan,
                f"{prefix}_K_Above25": False, f"{prefix}_K_Above_D": False,
                f"{prefix}_Both_Rising": False, f"{prefix}_Spread_Widening": False,
                f"{prefix}_Strong_Separation": False,
            })
            continue
        current, previous = weekly.iloc[position], weekly.iloc[previous_position]
        k, d = finite_num(current["SKDJ_K"]), finite_num(current["SKDJ_D"])
        prior_k, prior_d = finite_num(previous["SKDJ_K"]), finite_num(previous["SKDJ_D"])
        spread, prior_spread = k - d, prior_k - prior_d
        both_rising = k > prior_k and d > prior_d
        widening = spread > prior_spread
        result.update({
            f"{prefix}_Available": True, f"{prefix}_Date": str(current["trade_date"]),
            f"{prefix}_K": k, f"{prefix}_D": d,
            f"{prefix}_KD_Spread": spread, f"{prefix}_Spread_Change": spread - prior_spread,
            f"{prefix}_K_Change": k - prior_k, f"{prefix}_D_Change": d - prior_d,
            f"{prefix}_K_Above25": k >= SKDJ_BOTTOM, f"{prefix}_K_Above_D": k > d,
            f"{prefix}_Both_Rising": both_rising, f"{prefix}_Spread_Widening": widening,
            f"{prefix}_Strong_Separation": bool(k >= SKDJ_BOTTOM and k > d and both_rising and widening),
        })
        previous_position = position
    return result


def stock_trend(weekly: pd.DataFrame, position: int, daily: pd.DataFrame, signal_date: str) -> dict[str, Any]:
    close = weekly["close"]
    ma20 = close.rolling(20).mean()
    bias = finite_num((close / ma20 - 1.0).iloc[position] * 100.0)
    slope = finite_num((ma20 / ma20.shift(4) - 1.0).iloc[position] * 100.0)
    ret12 = finite_num(close.pct_change(12, fill_method=None).iloc[position] * 100.0)
    history = daily[daily["trade_date"].astype(str).le(signal_date)]
    daily_bias = finite_num(history.iloc[-1].get("D_MA60_Bias_pct")) if not history.empty else np.nan
    values = [bias, slope, ret12, daily_bias]
    if all(math.isfinite(value) and value > 0 for value in values):
        state = "上涨"
    elif all(math.isfinite(value) and value < 0 for value in values):
        state = "下跌"
    else:
        state = "震荡/过渡"
    return {
        "Individual_Trend": state, "Weekly_MA20_Bias_pct": bias,
        "Weekly_MA20_Slope_4W_pct": slope, "Weekly_Return_12W_pct": ret12,
        "Daily_MA60_Bias_pct": daily_bias,
    }


def analyze_stock(stock: pd.Series, periods: list[dict[str, str]], daily_raw: pd.DataFrame,
                  daily_basic: pd.DataFrame, week_last_map: dict[pd.Timestamp, str],
                  open_dates: list[str], open_pos: dict[str, int],
                  market_weeks: list[tuple[pd.Period, str]], config: dict[str, Any]
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    weekly = build_complete_weekly(daily_raw, week_last_map)
    if len(weekly) < INDICATOR_WARMUP_WEEKS:
        config["rejects"]["周线不足"] = config["rejects"].get("周线不足", 0) + 1
        return [], [], []
    daily = add_daily_features(daily_raw)
    cycles = build_bottom_cycles(weekly)
    swings = completed_golden_swings(weekly)
    cycle_rows: list[dict[str, Any]] = []
    preliminary_rows: list[dict[str, Any]] = []
    entry_events: list[dict[str, Any]] = []
    code, board = str(stock["ts_code"]), sample_board(stock)

    for cycle_number, cycle in enumerate(cycles, start=1):
        anchor_position = int(cycle["Anchor_Position"])
        trigger_position = cycle.get("Trigger_Position")
        anchor = weekly.iloc[anchor_position]
        anchor_date = str(anchor["trade_date"])
        trigger_date = (
            str(weekly.iloc[int(trigger_position)]["trade_date"])
            if trigger_position is not None else "")
        wait_weeks = (
            int(trigger_position) - anchor_position
            if trigger_position is not None else np.nan)
        anchor_snapshot = market_snapshot(daily_basic, anchor_date)
        anchor_membership = membership_on_date(periods, anchor_date)
        anchor_passed, anchor_reason = signal_filter(
            anchor_snapshot, config["min_price"], config["min_mv"])
        anchor_listed = str(stock["list_date"]) <= anchor_date < str(stock["delist_date"])
        anchor_eligible = bool(anchor_membership is not None and anchor_listed and anchor_passed)
        reject_reason = ""
        if not anchor_eligible:
            reject_reason = anchor_reason or (
                "当时不在历史科技池" if anchor_membership is None else "当时未上市或已退市")
        history_features = prior_swing_features(swings, anchor_position)
        low_level = (float(anchor["SKDJ_K"]) + float(anchor["SKDJ_D"])) / 2.0
        post = (
            post_confirmation_features(weekly, int(trigger_position))
            if trigger_position is not None else {})
        separation_date = str(post.get("Post_Confirm_W1_Date", ""))
        include_cycle = (
            config["signal_start"] <= anchor_date <= config["signal_end"]
            or bool(trigger_date and config["signal_start"] <= trigger_date <= config["signal_end"])
            or bool(separation_date and config["signal_start"] <= separation_date <= config["signal_end"]))
        if include_cycle:
            cycle_rows.append({
                "ts_code": code, "name": str(stock["name"]), "Sample_Board": board,
                "Bottom_Cycle_Number": cycle_number,
                "Low_Cross_Date": anchor_date, "Low_Cross_K": float(anchor["SKDJ_K"]),
                "Low_Cross_D": float(anchor["SKDJ_D"]), "Low_Cross_Level": low_level,
                "Low_Cross_Gap_To25": SKDJ_BOTTOM - low_level,
                "Bottom_Min_K": cycle["Bottom_Min_K"], "Bottom_Min_D": cycle["Bottom_Min_D"],
                "Bottom_Min_Level": cycle["Bottom_Min_Level"],
                "Bottom_Max_Depth_From25": SKDJ_BOTTOM - float(cycle["Bottom_Min_Level"]),
                "Weeks_Both_Below25": cycle["Weeks_Both_Below25"],
                "Bottom_Golden_Cross_Count": cycle["Bottom_Golden_Cross_Count"],
                "Preconfirmed": trigger_position is not None,
                "Preconfirm_Date": trigger_date, "Wait_Weeks_To_Preconfirm": wait_weeks,
                "Separation_Check_Date": separation_date,
                "W1_Strong_Separation": to_bool(post.get("Post_Confirm_W1_Strong_Separation")),
                "Observation_Ended_At": str(weekly.iloc[-1]["trade_date"]),
                "Eligible_Low_Cross_Pool": anchor_eligible,
                "Low_Cross_Filter_Reason": reject_reason,
                **anchor_snapshot, **history_features,
            })

        if trigger_position is None or not separation_date:
            continue
        if not (config["signal_start"] <= separation_date <= config["signal_end"]):
            continue
        trigger = weekly.iloc[int(trigger_position)]
        previous = weekly.iloc[int(trigger_position) - 1]
        trigger_snapshot = market_snapshot(daily_basic, trigger_date)
        trigger_membership = membership_on_date(periods, trigger_date)
        trigger_passed, trigger_reason = signal_filter(
            trigger_snapshot, config["min_price"], config["min_mv"])
        preliminary_eligible = bool(
            anchor_eligible
            and trigger_membership is not None
            and str(stock["list_date"]) <= trigger_date < str(stock["delist_date"])
            and trigger_passed)
        prelim_reason = ""
        if not preliminary_eligible:
            prelim_reason = reject_reason or trigger_reason or (
                "预确认日不在历史科技池" if trigger_membership is None else "预确认日上市状态无效")

        preliminary = {
            "ts_code": code, "name": str(stock["name"]), "Sample_Board": board,
            "Low_Cross_Date": anchor_date, "Preconfirm_Date": trigger_date,
            "Separation_Check_Date": separation_date,
            "Wait_Weeks_To_Preconfirm": wait_weeks,
            "Low_Cross_K": float(anchor["SKDJ_K"]), "Low_Cross_D": float(anchor["SKDJ_D"]),
            "Low_Cross_Gap_To25": SKDJ_BOTTOM - low_level,
            "Preconfirm_K": float(trigger["SKDJ_K"]), "Preconfirm_D": float(trigger["SKDJ_D"]),
            "Preconfirm_KD_Spread": float(trigger["SKDJ_KD_Spread"]),
            "Preconfirm_K_Change_1W": float(trigger["SKDJ_K"] - previous["SKDJ_K"]),
            "Preconfirm_D_Change_1W": float(trigger["SKDJ_D"] - previous["SKDJ_D"]),
            "W1_Strong_Separation": to_bool(post.get("Post_Confirm_W1_Strong_Separation")),
            "W1_K": post.get("Post_Confirm_W1_K", np.nan),
            "W1_D": post.get("Post_Confirm_W1_D", np.nan),
            "W1_KD_Spread": post.get("Post_Confirm_W1_KD_Spread", np.nan),
            "W1_K_Change": post.get("Post_Confirm_W1_K_Change", np.nan),
            "W1_D_Change": post.get("Post_Confirm_W1_D_Change", np.nan),
            "W1_Spread_Change": post.get("Post_Confirm_W1_Spread_Change", np.nan),
            "Eligible_Preliminary": preliminary_eligible,
            "Preliminary_Filter_Reason": prelim_reason,
            **history_features,
        }
        preliminary_rows.append(preliminary)
        if not preliminary_eligible:
            key = f"预确认:{prelim_reason}"
            config["rejects"][key] = config["rejects"].get(key, 0) + 1
            continue
        if not to_bool(post.get("Post_Confirm_W1_Strong_Separation")):
            continue

        separation_position = int(trigger_position) + 1
        if not (str(stock["list_date"]) <= separation_date < str(stock["delist_date"])):
            config["rejects"]["持续分离日未上市或已退市"] = (
                config["rejects"].get("持续分离日未上市或已退市", 0) + 1)
            continue
        separation_membership = membership_on_date(periods, separation_date)
        if separation_membership is None:
            config["rejects"]["持续分离日不在历史科技池"] = (
                config["rejects"].get("持续分离日不在历史科技池", 0) + 1)
            continue
        separation_snapshot = market_snapshot(daily_basic, separation_date)
        separation_passed, separation_reason = signal_filter(
            separation_snapshot, config["min_price"], config["min_mv"])
        if not separation_passed:
            key = f"持续分离日:{separation_reason}"
            config["rejects"][key] = config["rejects"].get(key, 0) + 1
            continue

        count65 = int(history_features.get("Prior_3_Count_Peak_GE65", 0))
        count70 = int(history_features.get("Prior_3_Count_Peak_GE70", 0))
        count75 = int(history_features.get("Prior_3_Count_Peak_GE75", 0))
        entry = {
            "Rule": "上穿25后W1持续分离，次周开盘买",
            "ts_code": code, "name": str(stock["name"]), "Sample_Board": board,
            "SW_L1": separation_membership["l1"], "SW_L2": separation_membership["l2"],
            "SW_L3": separation_membership["l3"], "Signal_Date": separation_date,
            "Low_Cross_Date": anchor_date, "Preconfirm_Date": trigger_date,
            "Separation_Confirm_Date": separation_date,
            "Wait_Weeks_To_Preconfirm": wait_weeks,
            "Total_Wait_Weeks": int(wait_weeks) + 1,
            "Low_Cross_K": float(anchor["SKDJ_K"]), "Low_Cross_D": float(anchor["SKDJ_D"]),
            "Low_Cross_Level": low_level, "Low_Cross_Gap_To25": SKDJ_BOTTOM - low_level,
            "Bottom_Min_K": cycle["Bottom_Min_K"], "Bottom_Min_D": cycle["Bottom_Min_D"],
            "Bottom_Min_Level": cycle["Bottom_Min_Level"],
            "Weeks_Both_Below25": cycle["Weeks_Both_Below25"],
            "Bottom_Golden_Cross_Count": cycle["Bottom_Golden_Cross_Count"],
            "Preconfirm_K": float(trigger["SKDJ_K"]), "Preconfirm_D": float(trigger["SKDJ_D"]),
            "Preconfirm_KD_Spread": float(trigger["SKDJ_KD_Spread"]),
            "Separation_K": post["Post_Confirm_W1_K"],
            "Separation_D": post["Post_Confirm_W1_D"],
            "Separation_KD_Spread": post["Post_Confirm_W1_KD_Spread"],
            "Separation_K_Change": post["Post_Confirm_W1_K_Change"],
            "Separation_D_Change": post["Post_Confirm_W1_D_Change"],
            "Separation_Spread_Change": post["Post_Confirm_W1_Spread_Change"],
            "Low_Cross_Raw_Close": anchor_snapshot["Raw_Close"],
            "Preconfirm_Raw_Close": trigger_snapshot["Raw_Close"],
            "Signal_Raw_Close": separation_snapshot["Raw_Close"],
            "Circ_MV_Billion": separation_snapshot["Circ_MV_Billion"],
            "Turnover_Rate": separation_snapshot["Turnover_Rate"],
            "Price_Change_Cross_to_Entry_Signal_pct": (
                (separation_snapshot["Raw_Close"] / anchor_snapshot["Raw_Close"] - 1.0) * 100.0
                if anchor_snapshot["Raw_Close"] > 0 else np.nan),
            "History_Rank_Score": count75 * 100 + count70 * 10 + count65,
            "Period_Group": (
                "2025-06以后" if separation_date >= config["split_date"] else "2025-06以前"),
            **history_features,
            **stock_trend(weekly, separation_position, daily, separation_date),
        }
        outcome = entry_outcomes(
            daily, separation_date, code, open_dates, open_pos, market_weeks, config)
        entry.update(prefix_keys(outcome, "Entry"))
        entry.update(prefix_keys(simulate_exit_policies(daily, outcome, config), "Entry"))
        entry_events.append(entry)
    return cycle_rows, preliminary_rows, entry_events

def max_empty_run(counts: pd.Series) -> int:
    longest = current = 0
    for value in counts.tolist():
        current = current + 1 if int(value) == 0 else 0
        longest = max(longest, current)
    return longest


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def mature_entries(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events[
        events["Entry_Tradable"].map(to_bool)
        & events[f"Entry_Has_W{AUDIT_WEEKS}"].map(to_bool)
    ].copy()


def rank_same_week(events: pd.DataFrame) -> pd.DataFrame:
    """Rank with final-signal-date observable fields only."""
    if events.empty:
        return events.copy()
    work = events.copy()
    order = [
        "Signal_Date", "Prior_3_Count_Peak_GE75", "Prior_3_Count_Peak_GE70",
        "Prior_3_Count_Peak_GE65", "Separation_Spread_Change",
        "Separation_K_Change", "ts_code",
    ]
    work = work.sort_values(
        order, ascending=[True, False, False, False, False, False, True],
        kind="mergesort")
    work["Same_Week_Rank"] = work.groupby("Signal_Date", sort=False).cumcount() + 1
    work["Selected_Top3"] = work["Same_Week_Rank"].le(TOP_K)
    return work.sort_values(["Signal_Date", "Same_Week_Rank", "ts_code"]).reset_index(drop=True)


def signal_week_calendar(open_dates: list[str], start: str, end: str,
                         events: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"trade_date": [day for day in open_dates if start <= day <= end]})
    frame["period"] = pd.to_datetime(frame["trade_date"]).dt.to_period("W-FRI")
    calendar = frame.groupby("period", as_index=False)["trade_date"].max().rename(
        columns={"trade_date": "Week_Last_Trade_Date"})
    all_counts = events.groupby("Signal_Date").size() if not events.empty else pd.Series(dtype=int)
    top3 = events[events["Selected_Top3"].map(to_bool)] if not events.empty else events
    top_counts = top3.groupby("Signal_Date").size() if not top3.empty else pd.Series(dtype=int)
    calendar["Final_Entry_Signals"] = (
        calendar["Week_Last_Trade_Date"].map(all_counts).fillna(0).astype(int))
    calendar["Same_Week_Top3"] = (
        calendar["Week_Last_Trade_Date"].map(top_counts).fillna(0).astype(int))
    return calendar


def outcome_summary(events: pd.DataFrame) -> pd.DataFrame:
    mature = mature_entries(events)
    if mature.empty:
        return pd.DataFrame()
    row: dict[str, Any] = {
        "买入规则": "首次上穿25后再等1周持续分离，次周开盘买",
        "最终信号事件": len(events),
        "可交易事件": int(events["Entry_Tradable"].map(to_bool).sum()),
        "W8成熟事件": len(mature), "不同股票": mature["ts_code"].nunique(),
        "信号周": mature["Signal_Date"].nunique(),
        "总等待周数中位数": numeric(mature, "Total_Wait_Weeks").median(),
    }
    for week in (5, 8):
        mfe = numeric(mature, f"Entry_W{week}_Cum_MFE_Net_pct")
        mae = numeric(mature, f"Entry_W{week}_Cum_MAE_Raw_pct")
        close_ret = numeric(mature, f"Entry_W{week}_Close_Return_Net_pct")
        row.update({
            f"W{week}最大浮盈均值%": mfe.mean(),
            f"W{week}最大浮盈中位数%": mfe.median(),
            f"W{week}达到10%比例%": mfe.ge(10).mean() * 100,
            f"W{week}达到20%比例%": mfe.ge(20).mean() * 100,
            f"W{week}最大回撤中位数%": mae.median(),
            f"W{week}期末平均净收益%": close_ret.mean(),
            f"W{week}期末中位净收益%": close_ret.median(),
            f"W{week}期末胜率%": close_ret.gt(0).mean() * 100,
        })
    return pd.DataFrame([row])


def weekly_path_audit(events: pd.DataFrame) -> pd.DataFrame:
    mature = mature_entries(events)
    rows: list[dict[str, Any]] = []
    for week in range(1, AUDIT_WEEKS + 1):
        available = mature[mature[f"Entry_Has_W{week}"].map(to_bool)].copy()
        mfe = numeric(available, f"Entry_W{week}_Cum_MFE_Net_pct")
        mae = numeric(available, f"Entry_W{week}_Cum_MAE_Raw_pct")
        close_ret = numeric(available, f"Entry_W{week}_Close_Return_Net_pct")
        rows.append({
            "持有周": f"W{week}", "成熟事件": len(available),
            "最大浮盈均值%": mfe.mean(), "最大浮盈中位数%": mfe.median(),
            "达到10%比例%": mfe.ge(10).mean() * 100 if len(mfe) else np.nan,
            "达到15%比例%": mfe.ge(15).mean() * 100 if len(mfe) else np.nan,
            "达到20%比例%": mfe.ge(20).mean() * 100 if len(mfe) else np.nan,
            "最大回撤中位数%": mae.median(),
            "期末平均净收益%": close_ret.mean(), "期末中位净收益%": close_ret.median(),
            "期末胜率%": close_ret.gt(0).mean() * 100 if len(close_ret) else np.nan,
        })
    return pd.DataFrame(rows)


def grouped_outcome_audit(events: pd.DataFrame, columns: str | list[str]) -> pd.DataFrame:
    mature = mature_entries(events)
    if mature.empty:
        return pd.DataFrame()
    group_columns = [columns] if isinstance(columns, str) else list(columns)
    grouper: Any = group_columns[0] if len(group_columns) == 1 else group_columns
    rows: list[dict[str, Any]] = []
    for keys, group in mature.groupby(grouper, dropna=False, observed=False, sort=True):
        values = (keys,) if len(group_columns) == 1 else tuple(keys)
        row = {column: value for column, value in zip(group_columns, values)}
        mfe8 = numeric(group, "Entry_W8_Cum_MFE_Net_pct")
        mae8 = numeric(group, "Entry_W8_Cum_MAE_Raw_pct")
        fixed5 = numeric(group, "Entry_Fixed_W5_Net_Return_pct")
        fixed8 = numeric(group, "Entry_Fixed_W8_Net_Return_pct")
        labels = group["Entry_First_Hit_10_vs_Minus10_W8"].astype(str)
        row.update({
            "事件数": len(group), "不同股票": group["ts_code"].nunique(),
            "信号周": group["Signal_Date"].nunique(),
            "W8最大浮盈均值%": mfe8.mean(), "W8最大浮盈中位数%": mfe8.median(),
            "W8达到10%比例%": mfe8.ge(10).mean() * 100,
            "W8达到20%比例%": mfe8.ge(20).mean() * 100,
            "W8最大回撤中位数%": mae8.median(),
            "固定W5平均净收益%": fixed5.mean(), "固定W5中位净收益%": fixed5.median(),
            "固定W5胜率%": fixed5.gt(0).mean() * 100,
            "固定W8平均净收益%": fixed8.mean(), "固定W8中位净收益%": fixed8.median(),
            "固定W8胜率%": fixed8.gt(0).mean() * 100,
            "W8先到+10比例%": labels.str.contains("先到+10%", regex=False).mean() * 100,
            "W8先到-10比例%": labels.str.contains("-10%", regex=False).mean() * 100,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def first_hit_audit(events: pd.DataFrame) -> pd.DataFrame:
    mature = mature_entries(events)
    rows: list[dict[str, Any]] = []
    for threshold in FIRST_HIT_PROFIT_LEVELS:
        key = int(threshold)
        labels = mature[f"Entry_First_Hit_{key}_vs_Minus10_W8"].astype(str)
        rows.append({
            "盈利阈值": f"+{key}%", "W8成熟事件": len(mature),
            "先到盈利阈值": labels.str.contains(f"先到+{key}%", regex=False).sum(),
            "先到-10%": labels.str.contains("先到-10%", regex=False).sum(),
            "同日双触发保守止损": labels.str.contains("同日同时触发", regex=False).sum(),
            "W8均未触发": labels.str.contains("均未触发", regex=False).sum(),
            "先到盈利阈值比例%": labels.str.contains(f"先到+{key}%", regex=False).mean() * 100,
            "先到-10%比例%": labels.str.contains("先到-10%", regex=False).mean() * 100,
        })
    return pd.DataFrame(rows)


EXIT_POLICIES = {
    "Fixed_W5": "固定持有5周", "Fixed_W8": "固定持有8周",
    "SL10_TP10": "止损10%+止盈10%", "SL10_TP15": "止损10%+止盈15%",
    "SL10_TP20": "止损10%+止盈20%",
    "Activate10_Trail10": "浮盈10%后回撤10%",
    "Activate15_Trail10": "浮盈15%后回撤10%",
}


def exit_policy_audit(events: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    mature = mature_entries(events)
    if mature.empty:
        return pd.DataFrame()
    groups = [("全部", mature)] if group_column is None else list(
        mature.groupby(group_column, dropna=False, observed=False, sort=True))
    rows: list[dict[str, Any]] = []
    for group_value, group in groups:
        for policy, label in EXIT_POLICIES.items():
            available = group[group[f"Entry_{policy}_Available"].map(to_bool)].copy()
            returns = numeric(available, f"Entry_{policy}_Net_Return_pct")
            triggers = available[f"Entry_{policy}_Trigger"].astype(str)
            row = {
                "分组": group_value, "退出规则": label, "样本": len(available),
                "平均净收益%": returns.mean(), "中位净收益%": returns.median(),
                "胜率%": returns.gt(0).mean() * 100 if len(returns) else np.nan,
                "亏损10%以上比例%": returns.le(-10).mean() * 100 if len(returns) else np.nan,
                "平均持有交易日": numeric(available, f"Entry_{policy}_Holding_Trading_Days").mean(),
                "止损触发比例%": triggers.str.contains("止损", regex=False).mean() * 100 if len(triggers) else np.nan,
                "止盈或回撤退出比例%": (
                    triggers.str.contains("止盈|回撤", regex=True).mean() * 100 if len(triggers) else np.nan),
            }
            if group_column is not None:
                row["分组字段"] = group_column
            rows.append(row)
    return pd.DataFrame(rows)


def selection_metrics(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    returns = numeric(frame, "Entry_Fixed_W5_Net_Return_pct")
    weekly = frame.assign(_return=returns).groupby("Signal_Date", sort=True)["_return"].mean()
    return {
        "方案": label, "事件数": len(frame), "信号周": frame["Signal_Date"].nunique(),
        "事件平均净收益%": returns.mean(), "事件中位净收益%": returns.median(),
        "事件胜率%": returns.gt(0).mean() * 100,
        "等权周平均净收益%": weekly.mean(), "等权周中位净收益%": weekly.median(),
        "盈利周比例%": weekly.gt(0).mean() * 100,
    }


def top3_random_audit(events: pd.DataFrame, trials: int = 500,
                      seed: int = 20260814) -> tuple[pd.DataFrame, pd.DataFrame]:
    mature = mature_entries(events)
    top3 = mature[mature["Selected_Top3"].map(to_bool)].copy()
    comparison = [selection_metrics(mature, "全部成熟事件"), selection_metrics(top3, "历史股性同周Top3")]
    rng = np.random.default_rng(seed)
    # Random selection is also made from the signal-date candidate set first;
    # maturity/tradability is applied afterwards, matching the real Top3 audit.
    grouped = [group.index.to_numpy() for _, group in events.groupby("Signal_Date", sort=True)]
    trial_rows: list[dict[str, Any]] = []
    for trial in range(1, trials + 1):
        chosen: list[Any] = []
        for indices in grouped:
            size = min(TOP_K, len(indices))
            chosen.extend(rng.choice(indices, size=size, replace=False).tolist())
        selected = mature_entries(events.loc[chosen].copy())
        metrics = selection_metrics(selected, f"随机Top3_{trial}")
        metrics["试验编号"] = trial
        trial_rows.append(metrics)
    trials_frame = pd.DataFrame(trial_rows)
    if not trials_frame.empty:
        for quantile, label in ((0.05, "随机Top3_P05"), (0.50, "随机Top3_中位"), (0.95, "随机Top3_P95")):
            row: dict[str, Any] = {"方案": label}
            for column in trials_frame.select_dtypes(include=[np.number]).columns:
                if column != "试验编号":
                    row[column] = trials_frame[column].quantile(quantile)
            comparison.append(row)
    return pd.DataFrame(comparison), trials_frame


def add_groups(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["Total_Wait_Group"] = pd.cut(
        numeric(work, "Total_Wait_Weeks"), bins=[0, 3, 7, 13, 19, 27, 53, np.inf],
        labels=["2–3周", "4–7周", "8–13周", "14–19周", "20–27周", "28–53周", "超过53周"],
        include_lowest=True)
    work["History75_Count_Group"] = numeric(work, "Prior_3_Count_Peak_GE75").fillna(0).astype(int)
    return work


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption("首次上穿25只是预确认；再等一周持续分离后才生成可买信号，并从下一交易日开盘重新判卷。")
    with st.expander("V4.4验证规则", expanded=True):
        st.markdown(f"""
- **观察池**：完整周线K、D均≤25时首次金叉，进入底部观察状态；25以下反复交叉合并为一个周期。
- **预确认**：K首次从25下方上穿25且K>D、K上升；此时不买。
- **最终买点**：再观察一个完整周，要求K、D同时上升、K>D、K≥25、K-D差值扩大；该周结束后的下一市场交易日开盘买。
- **历史排序**：同周先按前3个已完成波段中触及75、70、65的次数排序，再按确认周差值扩大和K上升幅度排序，只取Top{TOP_K}做独立审计。
- **退出对照**：固定持有5/8周；止损10%搭配止盈10/15/20%；浮盈10%或15%激活、从此前最高价回撤10%。所有动态退出最长观察W{AUDIT_WEEKS}。
- **执行保守性**：同日止盈止损双触发按止损；跳空越过止损按开盘价；移动止损只使用前一日已知最高价。
- **股票池**：申万历史科技行业，主板/创业板/科创板；低位金叉、预确认、最终确认均检查股价≥10元、流通市值≥100亿元。
- **边界**：同周Top3是排名能力审计，不等同于“最多持有3只”的跨周组合回测；最终退出规则确定后再做持仓占位模拟。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("最终信号开始", date(2023, 6, 5), key="v44_start")
        signal_end_date = st.date_input("最终信号截止", date(2026, 6, 5), key="v44_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v44_market_end")
        split_date_value = st.date_input("近期行情分界", date(2025, 6, 1), key="v44_split")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v44_pause")
        use_cache = st.checkbox("复用逐股票缓存", True, key="v44_cache")
        st.divider()
        commission_pct = st.number_input("佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f", key="v44_commission")
        stamp_duty_pct = st.number_input("卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f", key="v44_stamp")
        transfer_fee_pct = st.number_input("过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f", key="v44_transfer")
        if st.button("清除本程序行情缓存", key="v44_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("缓存已清除")

    token = st.text_input("Tushare Token", type="password", key="v44_token")
    session_key = "weekly_skdj_persistent_separation_v44_zip"
    result_name = "weekly_skdj_persistent_separation_entry_v4_4_all_results.zip"
    if not token:
        st.info("请输入Tushare Token；日期范围一致时可复用旧版逐股票行情缓存。")
        return
    if not st.button("开始V4.4持续分离买点审计", type="primary", key="v44_run"):
        if session_key in st.session_state:
            st.download_button("下载上一次结果ZIP", st.session_state[session_key],
                               file_name=result_name, mime="application/zip", on_click="ignore")
        return

    error = validate_dates(signal_start_date, signal_end_date, market_end_date)
    if error:
        st.error(error)
        return
    if (market_end_date - signal_end_date).days < 70:
        st.warning("观察截止日距离最终信号截止日不足70天，末端事件可能没有完整W8；程序会单独标记成熟样本。")

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    signal_start = signal_start_date.strftime("%Y%m%d")
    signal_end = signal_end_date.strftime("%Y%m%d")
    market_end = market_end_date.strftime("%Y%m%d")
    preload = (signal_start_date - timedelta(days=3 * 365)).strftime("%Y%m%d")
    rejects: dict[str, int] = {}
    config = {
        "signal_start": signal_start, "signal_end": signal_end, "market_end": market_end,
        "split_date": split_date_value.strftime("%Y%m%d"), "min_price": 10.0, "min_mv": 100.0,
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct), "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct), "rejects": rejects,
    }
    try:
        with st.spinner("加载交易日历与申万历史科技池..."):
            open_dates = load_trade_calendar(preload, market_end)
            extended_end = (market_end_date + timedelta(days=7)).strftime("%Y%m%d")
            week_last_map = complete_week_last_dates(load_trade_calendar(preload, extended_end))
            market_weeks = market_week_sequence(open_dates)
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return

    period_index = build_period_index(memberships)
    codes = sorted(set(period_index) & set(stock_basic["ts_code"].astype(str)))
    stocks = stock_basic[stock_basic["ts_code"].isin(codes)].copy()
    stocks = stocks[~stocks["list_date"].gt(signal_end) & ~stocks["delist_date"].lt(preload)].copy()
    stocks["Sample_Board"] = stocks.apply(sample_board, axis=1)
    stocks = stocks.sort_values("ts_code").reset_index(drop=True)
    population = stocks.groupby("Sample_Board").size().reindex(
        BOARDS, fill_value=0).rename("股票数").reset_index()
    open_pos = {day: position for position, day in enumerate(open_dates)}

    cycle_rows: list[dict[str, Any]] = []
    preliminary_rows: list[dict[str, Any]] = []
    entry_rows: list[dict[str, Any]] = []
    cache_hits = data_failures = 0
    progress, status = st.progress(0.0), st.empty()
    for number, stock in stocks.iterrows():
        code = str(stock["ts_code"])
        progress.progress((number + 1) / max(len(stocks), 1), text=f"{number + 1}/{len(stocks)} {code}")
        status.caption(
            f"底部周期 {len(cycle_rows)}；预确认 {len(preliminary_rows)}；最终信号 {len(entry_rows)}；"
            f"缓存 {cache_hits}；失败 {data_failures}")
        daily, daily_basic, cache_hit = fetch_stock_history(
            code, preload, market_end, bool(use_cache), float(pause))
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        cycle_part, preliminary_part, entry_part = analyze_stock(
            stock, period_index.get(code, []), daily, daily_basic, week_last_map,
            open_dates, open_pos, market_weeks, config)
        cycle_rows.extend(cycle_part)
        preliminary_rows.extend(preliminary_part)
        entry_rows.extend(entry_part)
    progress.empty()
    status.empty()

    cycles = pd.DataFrame(cycle_rows)
    preliminary = pd.DataFrame(preliminary_rows)
    entries = pd.DataFrame(entry_rows)
    if entries.empty:
        st.error("研究区间没有生成符合股票池且通过W1持续分离的最终买入信号。")
        return
    dt = pd.to_datetime(entries["Signal_Date"].astype(str), format="%Y%m%d", errors="coerce")
    entries["Signal_Year"] = entries["Signal_Date"].astype(str).str[:4]
    entries["Signal_Half_Year"] = entries["Signal_Year"] + "H" + np.where(dt.dt.month.le(6), "1", "2")
    entries = rank_same_week(add_groups(entries))

    quality = outcome_summary(entries)
    path = weekly_path_audit(entries)
    first_hit = first_hit_audit(entries)
    exits_all = exit_policy_audit(entries)
    exits_year = exit_policy_audit(entries, "Signal_Year")
    exits_top3 = exit_policy_audit(entries[entries["Selected_Top3"].map(to_bool)])
    history75 = grouped_outcome_audit(entries, "History75_Count_Group")
    year_audit = grouped_outcome_audit(entries, "Signal_Year")
    half_year_audit = grouped_outcome_audit(entries, "Signal_Half_Year")
    trend_audit = grouped_outcome_audit(entries, "Individual_Trend")
    wait_audit = grouped_outcome_audit(entries, "Total_Wait_Group")
    rank_compare, random_trials = top3_random_audit(entries)
    calendar = signal_week_calendar(open_dates, signal_start, signal_end, entries)
    counts = calendar["Final_Entry_Signals"]
    mature = mature_entries(entries)
    prelim_eligible = (
        preliminary[preliminary["Eligible_Preliminary"].map(to_bool)].copy()
        if not preliminary.empty else pd.DataFrame())
    prelim_summary = pd.DataFrame([{
        "全部预确认": len(preliminary), "股票池合格预确认": len(prelim_eligible),
        "W1持续分离": int(prelim_eligible["W1_Strong_Separation"].map(to_bool).sum()) if not prelim_eligible.empty else 0,
        "W1未持续分离": int((~prelim_eligible["W1_Strong_Separation"].map(to_bool)).sum()) if not prelim_eligible.empty else 0,
        "持续分离通过率%": (
            prelim_eligible["W1_Strong_Separation"].map(to_bool).mean() * 100 if not prelim_eligible.empty else np.nan),
    }])

    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "最终信号开始": signal_start,
        "最终信号截止": signal_end, "观察截止": market_end,
        "底部周期": len(cycles), "预确认事件": len(preliminary),
        "最终买入信号": len(entries), "不同股票": entries["ts_code"].nunique(),
        "W8成熟事件": len(mature), "自然周": len(calendar),
        "有信号周": int(counts.gt(0).sum()), "空窗周": int(counts.eq(0).sum()),
        "最长连续空窗周": max_empty_run(counts), "每周信号均值": counts.mean(),
        "每周信号中位数": counts.median(), "单周最多": counts.max(),
        "同周Top3事件": int(entries["Selected_Top3"].map(to_bool).sum()),
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])

    metadata = pd.DataFrame([
        ("预确认", "K首次从25下方上穿25且K>D、K上升；此周不买"),
        ("最终买点", "预确认后再等一个完整周；K、D均上升、K>D、K≥25且差值扩大；下一市场交易日开盘买"),
        ("排名", "同周依次按过去3个已完成波段触及75/70/65次数、确认周差值扩大、K上升排序"),
        ("历史信息边界", "只有最终信号日前已经完成的金叉—死叉波段进入历史评分"),
        ("Top3边界", "同周Top3只验证排序，不处理前一周持仓尚未退出造成的仓位占用"),
        ("动态退出上限", f"止盈止损和移动回撤最长观察W{AUDIT_WEEKS}，未触发则W8收盘退出"),
        ("移动止损", "先用固定-10%止损；达到激活浮盈后，从前一交易日已知最高价回撤10%，且保护价不低于成本原价"),
        ("同日歧义", "止盈与止损同日触发时保守按止损；跳空跌破保护价按开盘价"),
        ("成本", "买卖均计0.2%滑点、佣金和过户费，卖出另计印花税"),
        ("股票池", "申万历史科技行业；主板/创业板/科创板；三个信号节点均检查股价≥10元、流通市值≥100亿元"),
        ("未使用", "月线、分钟线、未来W2信息、最高价卖出、全样本自动调参"),
    ], columns=["项目", "说明"])

    files = {
        "01_run_summary_v4_4.csv": run_summary,
        "02_final_entry_rule_quality_v4_4.csv": quality,
        "03_entry_w1_w8_path_v4_4.csv": path,
        "04_first_hit_profit_vs_stop_w8_v4_4.csv": first_hit,
        "05_exit_policy_comparison_all_v4_4.csv": exits_all,
        "06_exit_policy_by_year_v4_4.csv": exits_year,
        "07_exit_policy_same_week_top3_v4_4.csv": exits_top3,
        "08_history75_count_group_v4_4.csv": history75,
        "09_top3_vs_random_top3_v4_4.csv": rank_compare,
        "10_random_top3_500_trials_v4_4.csv": random_trials,
        "11_year_stability_v4_4.csv": year_audit,
        "12_half_year_stability_v4_4.csv": half_year_audit,
        "13_individual_trend_v4_4.csv": trend_audit,
        "14_total_wait_group_v4_4.csv": wait_audit,
        "15_weekly_signal_calendar_v4_4.csv": calendar,
        "16_preconfirmation_to_separation_summary_v4_4.csv": prelim_summary,
        "17_all_final_entry_events_v4_4.csv": entries,
        "18_same_week_top3_events_v4_4.csv": entries[entries["Selected_Top3"].map(to_bool)].copy(),
        "19_all_preconfirmation_events_v4_4.csv": preliminary,
        "20_all_bottom_cycles_v4_4.csv": cycles,
        "21_full_tech_universe_v4_4.csv": stocks,
        "22_board_population_v4_4.csv": population,
        "23_rejection_audit_v4_4.csv": pd.DataFrame(
            [{"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]),
        "24_api_errors_v4_4.csv": pd.DataFrame({"错误": API_ERRORS}),
        "25_metadata_v4_4.csv": metadata,
    }
    result_zip = make_zip(files)
    st.session_state[session_key] = result_zip
    st.success(
        f"完成：最终信号{len(entries)}个，W8成熟{len(mature)}个；"
        f"有信号周{int(counts.gt(0).sum())}，空窗{int(counts.eq(0).sum())}周。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最终买入信号", len(entries))
    c2.metric("W8成熟事件", len(mature))
    c3.metric("有信号周", int(counts.gt(0).sum()))
    c4.metric("空窗周", int(counts.eq(0).sum()))
    st.subheader("最终买点整体表现")
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.subheader("退出规则对照")
    st.dataframe(exits_all, use_container_width=True, hide_index=True)
    st.subheader("历史股性同周Top3与随机Top3")
    st.dataframe(rank_compare, use_container_width=True, hide_index=True)
    st.subheader("过去3个波段触及75的次数")
    st.dataframe(history75, use_container_width=True, hide_index=True)
    st.download_button("下载V4.4全部结果ZIP", result_zip, file_name=result_name,
                       mime="application/zip", type="primary", key="v44_download", on_click="ignore")
    st.info("优先看05判断退出方式，09判断历史股性Top3是否稳定优于随机，15检查信号覆盖与扎堆，17可逐笔核对真实买入价。")


if __name__ == "__main__":
    main()
