# -*- coding: utf-8 -*-
"""科技股周线SKDJ底部结构、上穿25确认与历史股性审计 V4.3。

本版只验证一条可执行买入规则：
1. 周线SKDJ在25以下首次形成独立底部金叉，进入观察状态；
2. 不设置6周失效期，等待K首次从25下方上穿25，且K>D、K继续上升；
3. 完整确认周结束后的下一市场交易日开盘买入。

买入时已知特征与买入后的持仓确认严格分开。底部深度、等待时间、
确认周K/D分离度、此前1~3次已完成金叉波段的最高K值均只使用信号当时
已经发生的数据；确认后第1~2周K/D是否继续分离只作为事后持仓诊断。
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
TITLE = "科技股周线SKDJ确认结构与历史股性审计 V4.3"
VERSION = "V4.3-WEEKLY-SKDJ-CONFIRMATION-STRUCTURE-AUDIT"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")

SKDJ_N = 9
SKDJ_M = 3
SKDJ_BOTTOM = 25.0
INDICATOR_WARMUP_WEEKS = 40
HOLD_20D = 20
HOLD_40D = 40
AUDIT_WEEKS = 12
HISTORY_PEAK_LEVELS = (65.0, 70.0, 75.0)
FIRST_HIT_PROFIT_LEVELS = (10.0, 15.0, 20.0)

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
        "First_Hit_10_vs_Minus10_W12": "", "First_Hit_10_Date_W12": "",
        "First_Hit_15_vs_Minus10_W12": "", "First_Hit_15_Date_W12": "",
        "First_Hit_20_vs_Minus10_W12": "", "First_Hit_20_Date_W12": "",
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
                out[f"First_Hit_{key}_vs_Minus10_W12"] = label
                out[f"First_Hit_{key}_Date_W12"] = hit_date

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
    """Future W1/W2 fields are diagnostics only and must never enter buy-time filters."""
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
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weekly = build_complete_weekly(daily_raw, week_last_map)
    if len(weekly) < INDICATOR_WARMUP_WEEKS:
        config["rejects"]["周线不足"] = config["rejects"].get("周线不足", 0) + 1
        return [], []
    daily = add_daily_features(daily_raw)
    cycles = build_bottom_cycles(weekly)
    swings = completed_golden_swings(weekly)
    cycle_rows: list[dict[str, Any]] = []
    confirmed_events: list[dict[str, Any]] = []
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

        include_cycle = (
            config["signal_start"] <= anchor_date <= config["signal_end"]
            or bool(trigger_date and config["signal_start"] <= trigger_date <= config["signal_end"]))
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
                "Confirmed_Event": trigger_position is not None,
                "Confirm_Date": trigger_date, "Wait_Weeks": wait_weeks,
                "Observation_Ended_At": str(weekly.iloc[-1]["trade_date"]),
                "Eligible_Low_Cross_Pool": anchor_eligible,
                "Low_Cross_Filter_Reason": reject_reason,
                **anchor_snapshot, **history_features,
            })

        if trigger_position is None or not (config["signal_start"] <= trigger_date <= config["signal_end"]):
            continue
        if not anchor_eligible:
            key = f"确认规则的低位金叉:{reject_reason}"
            config["rejects"][key] = config["rejects"].get(key, 0) + 1
            continue
        if not (str(stock["list_date"]) <= trigger_date < str(stock["delist_date"])):
            config["rejects"]["确认日未上市或已退市"] = config["rejects"].get("确认日未上市或已退市", 0) + 1
            continue
        trigger_membership = membership_on_date(periods, trigger_date)
        if trigger_membership is None:
            config["rejects"]["确认日不在历史科技池"] = config["rejects"].get("确认日不在历史科技池", 0) + 1
            continue
        trigger_snapshot = market_snapshot(daily_basic, trigger_date)
        trigger_passed, trigger_reason = signal_filter(
            trigger_snapshot, config["min_price"], config["min_mv"])
        if not trigger_passed:
            key = f"确认日:{trigger_reason}"
            config["rejects"][key] = config["rejects"].get(key, 0) + 1
            continue

        trigger = weekly.iloc[int(trigger_position)]
        previous = weekly.iloc[int(trigger_position) - 1]
        confirmed = {
            "Rule": "独立低位金叉后不限周数首次上穿25确认买",
            "ts_code": code, "name": str(stock["name"]), "Sample_Board": board,
            "SW_L1": trigger_membership["l1"], "SW_L2": trigger_membership["l2"],
            "SW_L3": trigger_membership["l3"], "Signal_Date": trigger_date,
            "Low_Cross_Date": anchor_date, "Confirm_Date": trigger_date,
            "Wait_Weeks": wait_weeks,
            "Low_Cross_K": float(anchor["SKDJ_K"]), "Low_Cross_D": float(anchor["SKDJ_D"]),
            "Low_Cross_Level": low_level, "Low_Cross_Gap_To25": SKDJ_BOTTOM - low_level,
            "Bottom_Min_K": cycle["Bottom_Min_K"], "Bottom_Min_D": cycle["Bottom_Min_D"],
            "Bottom_Min_Level": cycle["Bottom_Min_Level"],
            "Bottom_Max_Depth_From25": SKDJ_BOTTOM - float(cycle["Bottom_Min_Level"]),
            "Weeks_Both_Below25": cycle["Weeks_Both_Below25"],
            "Bottom_Golden_Cross_Count": cycle["Bottom_Golden_Cross_Count"],
            "Confirm_K": float(trigger["SKDJ_K"]), "Confirm_D": float(trigger["SKDJ_D"]),
            "Confirm_Level": float(trigger["SKDJ_Level"]),
            "Confirm_KD_Spread": float(trigger["SKDJ_KD_Spread"]),
            "Confirm_K_Change_1W": float(trigger["SKDJ_K"] - previous["SKDJ_K"]),
            "Confirm_D_Change_1W": float(trigger["SKDJ_D"] - previous["SKDJ_D"]),
            "Confirm_Spread_Change_1W": float(
                trigger["SKDJ_KD_Spread"] - previous["SKDJ_KD_Spread"]),
            "Confirm_Both_Rising": bool(
                trigger["SKDJ_K"] > previous["SKDJ_K"]
                and trigger["SKDJ_D"] > previous["SKDJ_D"]),
            "Low_Cross_Raw_Close": anchor_snapshot["Raw_Close"],
            "Confirm_Raw_Close": trigger_snapshot["Raw_Close"],
            "Circ_MV_Billion": trigger_snapshot["Circ_MV_Billion"],
            "Turnover_Rate": trigger_snapshot["Turnover_Rate"],
            "Price_Change_Cross_to_Confirm_pct": (
                (trigger_snapshot["Raw_Close"] / anchor_snapshot["Raw_Close"] - 1.0) * 100.0
                if anchor_snapshot["Raw_Close"] > 0 else np.nan),
            "Period_Group": (
                "2025-06以后" if trigger_date >= config["split_date"] else "2025-06以前"),
            **history_features,
            **stock_trend(weekly, int(trigger_position), daily, trigger_date),
            **post_confirmation_features(weekly, int(trigger_position)),
        }
        outcome = entry_outcomes(
            daily, trigger_date, code, open_dates, open_pos, market_weeks, config)
        confirmed.update(prefix_keys(outcome, "Confirmed"))
        confirmed_events.append(confirmed)
    return cycle_rows, confirmed_events

def max_empty_run(counts: pd.Series) -> int:
    longest = current = 0
    for value in counts.tolist():
        current = current + 1 if int(value) == 0 else 0
        longest = max(longest, current)
    return longest


def mature_confirmed(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    return events[
        events["Confirmed_Tradable"].map(to_bool)
        & events[f"Confirmed_Has_W{AUDIT_WEEKS}"].map(to_bool)
    ].copy()


def signal_week_calendar(open_dates: list[str], start: str, end: str,
                         events: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"trade_date": [day for day in open_dates if start <= day <= end]})
    frame["period"] = pd.to_datetime(frame["trade_date"]).dt.to_period("W-FRI")
    calendar = frame.groupby("period", as_index=False)["trade_date"].max().rename(
        columns={"trade_date": "Week_Last_Trade_Date"})
    counts = events.groupby("Signal_Date").size() if not events.empty else pd.Series(dtype=int)
    mature = mature_confirmed(events)
    mature_counts = mature.groupby("Signal_Date").size() if not mature.empty else pd.Series(dtype=int)
    calendar["Confirmed_Signals"] = calendar["Week_Last_Trade_Date"].map(counts).fillna(0).astype(int)
    calendar["Confirmed_W12_Mature"] = (
        calendar["Week_Last_Trade_Date"].map(mature_counts).fillna(0).astype(int))
    return calendar


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def outcome_summary(events: pd.DataFrame) -> pd.DataFrame:
    mature = mature_confirmed(events)
    if mature.empty:
        return pd.DataFrame()
    row: dict[str, Any] = {
        "买入规则": "独立低位金叉后不限周数首次上穿25，次周开盘买",
        "全部确认事件": len(events),
        "可交易事件": int(events["Confirmed_Tradable"].map(to_bool).sum()),
        "W12成熟事件": len(mature),
        "不同股票": mature["ts_code"].nunique(),
        "信号周": mature["Signal_Date"].nunique(),
        "等待周数中位数": numeric(mature, "Wait_Weeks").median(),
        "等待周数最大值": numeric(mature, "Wait_Weeks").max(),
    }
    for week in (5, 8, 12):
        mfe = numeric(mature, f"Confirmed_W{week}_Cum_MFE_Net_pct")
        mae = numeric(mature, f"Confirmed_W{week}_Cum_MAE_Raw_pct")
        close_ret = numeric(mature, f"Confirmed_W{week}_Close_Return_Net_pct")
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
    mature = mature_confirmed(events)
    rows = []
    for week in range(1, AUDIT_WEEKS + 1):
        available = mature[mature[f"Confirmed_Has_W{week}"].map(to_bool)].copy()
        mfe = numeric(available, f"Confirmed_W{week}_Cum_MFE_Net_pct")
        mae = numeric(available, f"Confirmed_W{week}_Cum_MAE_Raw_pct")
        close_ret = numeric(available, f"Confirmed_W{week}_Close_Return_Net_pct")
        rows.append({
            "持有周": f"W{week}", "成熟事件": len(available),
            "最大浮盈均值%": mfe.mean(), "最大浮盈中位数%": mfe.median(),
            "达到5%比例%": mfe.ge(5).mean() * 100 if len(mfe) else np.nan,
            "达到10%比例%": mfe.ge(10).mean() * 100 if len(mfe) else np.nan,
            "达到15%比例%": mfe.ge(15).mean() * 100 if len(mfe) else np.nan,
            "达到20%比例%": mfe.ge(20).mean() * 100 if len(mfe) else np.nan,
            "最大回撤中位数%": mae.median(),
            "期末平均净收益%": close_ret.mean(), "期末中位净收益%": close_ret.median(),
            "期末胜率%": close_ret.gt(0).mean() * 100 if len(close_ret) else np.nan,
        })
    return pd.DataFrame(rows)


def grouped_outcome_audit(events: pd.DataFrame, columns: str | list[str]) -> pd.DataFrame:
    mature = mature_confirmed(events)
    if mature.empty:
        return pd.DataFrame()
    group_columns = [columns] if isinstance(columns, str) else list(columns)
    rows: list[dict[str, Any]] = []
    grouper: Any = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in mature.groupby(grouper, dropna=False, observed=False, sort=True):
        values = (keys,) if len(group_columns) == 1 else tuple(keys)
        row = {column: value for column, value in zip(group_columns, values)}
        mfe12 = numeric(group, "Confirmed_W12_Cum_MFE_Net_pct")
        mae12 = numeric(group, "Confirmed_W12_Cum_MAE_Raw_pct")
        close5 = numeric(group, "Confirmed_W5_Close_Return_Net_pct")
        close8 = numeric(group, "Confirmed_W8_Close_Return_Net_pct")
        close12 = numeric(group, "Confirmed_W12_Close_Return_Net_pct")
        hit10 = group["Confirmed_First_Hit_10_vs_Minus10_W12"].astype(str)
        row.update({
            "事件数": len(group), "不同股票": group["ts_code"].nunique(),
            "信号周": group["Signal_Date"].nunique(),
            "W12最大浮盈均值%": mfe12.mean(), "W12最大浮盈中位数%": mfe12.median(),
            "W12达到5%比例%": mfe12.ge(5).mean() * 100,
            "W12达到10%比例%": mfe12.ge(10).mean() * 100,
            "W12达到20%比例%": mfe12.ge(20).mean() * 100,
            "W12最大回撤中位数%": mae12.median(),
            "W5期末平均收益%": close5.mean(), "W5期末中位收益%": close5.median(),
            "W8期末平均收益%": close8.mean(), "W8期末中位收益%": close8.median(),
            "W12期末平均收益%": close12.mean(), "W12期末中位收益%": close12.median(),
            "W12期末胜率%": close12.gt(0).mean() * 100,
            "W12先到+10比例%": hit10.str.contains(r"先到\+10", regex=True).mean() * 100,
            "W12先到-10比例%": hit10.str.contains("先到-10", regex=False).mean() * 100,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def first_hit_audit(events: pd.DataFrame) -> pd.DataFrame:
    mature = mature_confirmed(events)
    rows = []
    for threshold in FIRST_HIT_PROFIT_LEVELS:
        key = int(threshold)
        labels = mature[f"Confirmed_First_Hit_{key}_vs_Minus10_W12"].astype(str)
        rows.append({
            "盈利阈值": f"+{key}%", "成熟事件": len(mature),
            "先到盈利阈值": labels.str.contains(f"先到+{key}%", regex=False).sum(),
            "先到-10%": labels.str.contains("先到-10%", regex=False).sum(),
            "同日双触发保守按止损": labels.str.contains("同日同时触发", regex=False).sum(),
            "W12均未触发": labels.str.contains("均未触发", regex=False).sum(),
            "先到盈利阈值比例%": labels.str.contains(f"先到+{key}%", regex=False).mean() * 100,
            "先到-10%比例%": labels.str.contains("先到-10%", regex=False).mean() * 100,
        })
    return pd.DataFrame(rows)


def add_audit_groups(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["Wait_Group"] = pd.cut(
        numeric(work, "Wait_Weeks"),
        bins=[0, 2, 6, 12, 18, 26, 52, np.inf],
        labels=["1–2周", "3–6周", "7–12周", "13–18周", "19–26周", "27–52周", "超过52周"],
        include_lowest=True)
    work["Low_Cross_Gap_Group"] = pd.cut(
        numeric(work, "Low_Cross_Gap_To25"),
        bins=[-np.inf, 2.5, 5, 10, 15, np.inf],
        labels=["≤2.5", "2.5–5", "5–10", "10–15", ">15"])
    work["Bottom_Depth_Group"] = pd.cut(
        numeric(work, "Bottom_Max_Depth_From25"),
        bins=[-np.inf, 5, 10, 15, 20, np.inf],
        labels=["≤5", "5–10", "10–15", "15–20", ">20"])
    work["Bottom_Weeks_Group"] = pd.cut(
        numeric(work, "Weeks_Both_Below25"),
        bins=[0, 2, 6, 12, 18, 26, np.inf],
        labels=["1–2周", "3–6周", "7–12周", "13–18周", "19–26周", "超过26周"],
        include_lowest=True)
    work["Confirm_Spread_Group"] = pd.cut(
        numeric(work, "Confirm_KD_Spread"),
        bins=[-np.inf, 2, 5, 10, np.inf],
        labels=["≤2", "2–5", "5–10", ">10"])
    work["Prev_Swing1_Peak_Group"] = pd.cut(
        numeric(work, "Prev_Swing1_Peak_K"),
        bins=[-np.inf, 50, 65, 70, 75, np.inf],
        labels=["<50", "50–65", "65–70", "70–75", "≥75"])
    work["Prior_History_Completeness"] = np.select(
        [
            numeric(work, "Prior_Swings_Available").ge(3),
            numeric(work, "Prior_Swings_Available").eq(2),
            numeric(work, "Prior_Swings_Available").eq(1),
        ],
        ["有3次历史波段", "有2次历史波段", "仅1次历史波段"],
        default="无已完成历史波段")
    return work


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title="周线SKDJ确认结构审计 V4.3", layout="wide")
    st.title(TITLE)
    st.caption("低位金叉只建立观察状态；不限6周等待首次上穿25；验证底部结构、历史股性和确认后K/D分离。")
    with st.expander("V4.3验证规则", expanded=True):
        st.markdown(f"""
- **唯一买入规则**：完整周线K、D均≤25时首次低位金叉，进入观察状态；25以下反复金叉死叉合并为同一个底部周期。
- **不限等待周数**：从底部周期开始，等待K首次由25下方上穿25，同时K>D且K继续上升；确认周后下一市场交易日开盘买入。
- **买入前特征**：低位金叉距25的距离、等待周数、25以下停留周数、底部最低K/D、确认周K-D差值及变化速度。
- **历史股性**：只使用当前底部金叉之前已经完成的普通金叉—死叉波段，记录最近1～3次最高K值及达到65/70/75的次数。
- **持仓确认诊断**：确认后的第1、2周K、D是否同时上升且差值继续扩大。这是未来诊断字段，不参与当周买入筛选。
- **收益判卷**：确认后次周开盘买入，输出W1～W12累计最大浮盈、最大回撤和期末净收益，并比较先到+10/+15/+20还是先到-10%。
- **股票池**：申万历史科技行业，主板/创业板/科创板；低位金叉日和确认日均要求原始股价≥10元、流通市值≥100亿元。
- **严禁未来泄漏**：本版不根据未来是否继续分离筛选买入事件，不自动挑选最优阈值，也不把最高价当作实际卖出价。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("确认信号开始", date(2023, 6, 5), key="v43_start")
        signal_end_date = st.date_input("确认信号截止", date(2026, 6, 5), key="v43_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v43_market_end")
        split_date_value = st.date_input("近期行情分界", date(2025, 6, 1), key="v43_split")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v43_pause")
        use_cache = st.checkbox("复用逐股票缓存", True, key="v43_cache")
        st.divider()
        commission_pct = st.number_input(
            "佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f", key="v43_commission")
        stamp_duty_pct = st.number_input(
            "卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f", key="v43_stamp")
        transfer_fee_pct = st.number_input(
            "过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f", key="v43_transfer")
        if st.button("清除本程序行情缓存", key="v43_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("缓存已清除")

    token = st.text_input("Tushare Token", type="password", key="v43_token")
    session_key = "weekly_skdj_confirmation_structure_v43_zip"
    if not token:
        st.info("请输入Tushare Token；日期范围一致时可复用V4.2逐股票缓存。")
        return
    if not st.button("开始V4.3确认结构审计", type="primary", key="v43_run"):
        if session_key in st.session_state:
            st.download_button(
                "下载上一次结果ZIP", st.session_state[session_key],
                file_name="weekly_skdj_confirmation_structure_audit_v4_3_all_results.zip",
                mime="application/zip", on_click="ignore")
        return

    error = validate_dates(signal_start_date, signal_end_date, market_end_date)
    if error:
        st.error(error)
        return
    if (market_end_date - signal_end_date).days < 100:
        st.warning("观察截止日距离确认信号截止日不足100天，末端事件可能没有完整W12；程序仍会继续并单独标记成熟样本。")

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
            full_open_dates = load_trade_calendar(preload, extended_end)
            week_last_map = complete_week_last_dates(full_open_dates)
            market_weeks = market_week_sequence(open_dates)
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return

    period_index = build_period_index(memberships)
    codes = sorted(set(period_index) & set(stock_basic["ts_code"].astype(str)))
    stocks = stock_basic[stock_basic["ts_code"].isin(codes)].copy()
    stocks = stocks[
        ~stocks["list_date"].gt(signal_end) & ~stocks["delist_date"].lt(preload)].copy()
    stocks["Sample_Board"] = stocks.apply(sample_board, axis=1)
    stocks = stocks.sort_values("ts_code").reset_index(drop=True)
    population = stocks.groupby("Sample_Board").size().reindex(
        BOARDS, fill_value=0).rename("股票数").reset_index()
    open_pos = {day: position for position, day in enumerate(open_dates)}

    cycle_rows: list[dict[str, Any]] = []
    confirmed_rows: list[dict[str, Any]] = []
    cache_hits = data_failures = 0
    progress, status = st.progress(0.0), st.empty()
    for number, stock in stocks.iterrows():
        code = str(stock["ts_code"])
        progress.progress(
            (number + 1) / max(len(stocks), 1), text=f"{number + 1}/{len(stocks)} {code}")
        status.caption(
            f"底部周期 {len(cycle_rows)}；上穿25确认 {len(confirmed_rows)}；"
            f"缓存 {cache_hits}；失败 {data_failures}")
        daily, daily_basic, cache_hit = fetch_stock_history(
            code, preload, market_end, bool(use_cache), float(pause))
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        cycle_part, confirmed_part = analyze_stock(
            stock, period_index.get(code, []), daily, daily_basic, week_last_map,
            open_dates, open_pos, market_weeks, config)
        cycle_rows.extend(cycle_part)
        confirmed_rows.extend(confirmed_part)
    progress.empty()
    status.empty()

    cycles = pd.DataFrame(cycle_rows)
    confirmed = pd.DataFrame(confirmed_rows)
    if confirmed.empty:
        st.error("研究区间没有生成符合股票池的上穿25确认事件。")
        return
    dt = pd.to_datetime(confirmed["Signal_Date"].astype(str), format="%Y%m%d", errors="coerce")
    confirmed["Signal_Year"] = confirmed["Signal_Date"].astype(str).str[:4]
    confirmed["Signal_Half_Year"] = (
        confirmed["Signal_Year"] + "H" + np.where(dt.dt.month.le(6), "1", "2"))
    confirmed = add_audit_groups(confirmed)

    quality = outcome_summary(confirmed)
    path = weekly_path_audit(confirmed)
    first_hit = first_hit_audit(confirmed)
    wait_audit = grouped_outcome_audit(confirmed, "Wait_Group")
    gap_audit = grouped_outcome_audit(confirmed, "Low_Cross_Gap_Group")
    gap_wait_audit = grouped_outcome_audit(
        confirmed, ["Low_Cross_Gap_Group", "Wait_Group"])
    bottom_depth_audit = grouped_outcome_audit(confirmed, "Bottom_Depth_Group")
    bottom_weeks_audit = grouped_outcome_audit(confirmed, "Bottom_Weeks_Group")
    prior1_audit = grouped_outcome_audit(confirmed, "Prev_Swing1_Peak_Group")
    history_completeness = grouped_outcome_audit(confirmed, "Prior_History_Completeness")
    prior_threshold_parts = []
    for level in HISTORY_PEAK_LEVELS:
        column = f"Prior_3_Count_Peak_GE{int(level)}"
        part = grouped_outcome_audit(confirmed, column)
        if not part.empty:
            part.insert(0, "历史峰值门槛", int(level))
            prior_threshold_parts.append(part)
    prior_threshold_audit = (
        pd.concat(prior_threshold_parts, ignore_index=True)
        if prior_threshold_parts else pd.DataFrame())
    confirm_spread_audit = grouped_outcome_audit(confirmed, "Confirm_Spread_Group")
    confirm_rising_audit = grouped_outcome_audit(confirmed, "Confirm_Both_Rising")
    post_w1_audit = grouped_outcome_audit(confirmed, "Post_Confirm_W1_Strong_Separation")
    post_w2_audit = grouped_outcome_audit(confirmed, "Post_Confirm_W2_Strong_Separation")
    year_audit = grouped_outcome_audit(confirmed, "Signal_Year")
    half_year_audit = grouped_outcome_audit(confirmed, "Signal_Half_Year")
    trend_audit = grouped_outcome_audit(confirmed, "Individual_Trend")
    calendar = signal_week_calendar(open_dates, signal_start, signal_end, confirmed)
    counts = calendar["Confirmed_Signals"]
    mature = mature_confirmed(confirmed)
    unresolved = (
        cycles[
            cycles["Eligible_Low_Cross_Pool"].map(to_bool)
            & ~cycles["Confirmed_Event"].map(to_bool)
        ].copy() if not cycles.empty else pd.DataFrame())

    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "确认信号开始": signal_start,
        "确认信号截止": signal_end, "观察截止": market_end,
        "底部周期明细": len(cycles), "未确认观察状态": len(unresolved),
        "确认买事件": len(confirmed), "确认买不同股票": confirmed["ts_code"].nunique(),
        "W12成熟事件": len(mature), "自然周": len(calendar),
        "有确认信号周": int(counts.gt(0).sum()), "空窗周": int(counts.eq(0).sum()),
        "最长空窗周": max_empty_run(counts), "每周确认数均值": counts.mean(),
        "每周确认数中位数": counts.median(), "单周最多": counts.max(),
        "等待周数中位数": numeric(confirmed, "Wait_Weeks").median(),
        "等待周数最大值": numeric(confirmed, "Wait_Weeks").max(),
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])

    metadata = pd.DataFrame([
        ("底部周期", "K、D均≤25时首次低位金叉启动观察；上穿25前的反复金叉死叉合并，不重复生成买入事件"),
        ("确认买入", "不设6周失效期；K首次从25下方上穿25、K>D且K继续上升；完整确认周后下一市场交易日开盘买入"),
        ("历史波段", "普通SKDJ金叉至随后死叉为一个已完成波段；当前底部金叉之前已经完成的最近1～3个波段才可用于买入时特征"),
        ("历史峰值阈值", "同时报告65、70、75三个门槛，不按本次全样本结果自动挑选"),
        ("确认周分离", "Confirm_KD_Spread及K/D一周变化在买入前已知，可用于特征分析"),
        ("确认后分离", "Post_Confirm_W1/W2是买入后的未来持仓诊断，严禁回填到买入筛选或排序"),
        ("W1至W12", "W1为买入所在市场周；累计最大浮盈、最大回撤和期末收益均从次周开盘成交后计算"),
        ("最高价", "最大浮盈仅描述机会，不视为可实现卖出价；实际退出规则留待结构验证后单独审计"),
        ("成本", "买卖均计0.2%滑点、佣金和过户费，卖出另计印花税"),
        ("股票池", "申万历史科技行业；主板/创业板/科创板；低位金叉日及确认日股价≥10元、流通市值≥100亿元"),
        ("观察上限", "没有人为等待周数上限；尚未确认事件只表示截至行情观察日仍在观察，不能称为永久失败"),
        ("未使用", "月线、分钟线、未来分离筛选、自动评分、事后最优参数、最高价卖出"),
    ], columns=["项目", "说明"])

    files = {
        "01_run_summary_v4_3.csv": run_summary,
        "02_confirmed_buy_rule_quality_v4_3.csv": quality,
        "03_confirmed_w1_w12_path_v4_3.csv": path,
        "04_first_hit_profit_vs_stop_v4_3.csv": first_hit,
        "05_wait_week_group_v4_3.csv": wait_audit,
        "06_low_cross_gap_group_v4_3.csv": gap_audit,
        "07_gap_x_wait_interaction_v4_3.csv": gap_wait_audit,
        "08_bottom_depth_group_v4_3.csv": bottom_depth_audit,
        "09_bottom_duration_group_v4_3.csv": bottom_weeks_audit,
        "10_previous_swing1_peak_group_v4_3.csv": prior1_audit,
        "11_prior_history_completeness_v4_3.csv": history_completeness,
        "12_prior_3_swing_threshold_counts_v4_3.csv": prior_threshold_audit,
        "13_confirm_week_spread_group_v4_3.csv": confirm_spread_audit,
        "14_confirm_week_both_rising_v4_3.csv": confirm_rising_audit,
        "15_post_confirm_w1_separation_diagnostic_v4_3.csv": post_w1_audit,
        "16_post_confirm_w2_separation_diagnostic_v4_3.csv": post_w2_audit,
        "17_year_stability_v4_3.csv": year_audit,
        "18_half_year_stability_v4_3.csv": half_year_audit,
        "19_individual_trend_v4_3.csv": trend_audit,
        "20_weekly_signal_calendar_v4_3.csv": calendar,
        "21_all_confirmed_events_v4_3.csv": confirmed,
        "22_all_bottom_cycles_v4_3.csv": cycles,
        "23_unresolved_bottom_cycles_v4_3.csv": unresolved,
        "24_full_tech_universe_v4_3.csv": stocks,
        "25_board_population_v4_3.csv": population,
        "26_rejection_audit_v4_3.csv": pd.DataFrame(
            [{"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]),
        "27_api_errors_v4_3.csv": pd.DataFrame({"错误": API_ERRORS}),
        "28_metadata_v4_3.csv": metadata,
    }
    result_zip = make_zip(files)
    st.session_state[session_key] = result_zip
    st.success(
        f"完成：上穿25确认{len(confirmed)}个，W12成熟{len(mature)}个；"
        f"有信号周{int(counts.gt(0).sum())}，空窗{int(counts.eq(0).sum())}周。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("确认买事件", len(confirmed))
    c2.metric("W12成熟事件", len(mature))
    c3.metric("有信号周", int(counts.gt(0).sum()))
    c4.metric("空窗周", int(counts.eq(0).sum()))
    st.subheader("唯一可执行买入规则")
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.subheader("等待周数分组")
    st.dataframe(wait_audit, use_container_width=True, hide_index=True)
    st.subheader("金叉深度×等待时间")
    st.dataframe(gap_wait_audit, use_container_width=True, hide_index=True)
    st.subheader("前1～3次历史波段冲击65/70/75")
    st.dataframe(prior_threshold_audit, use_container_width=True, hide_index=True)
    st.subheader("确认后第1周K/D继续分离（仅持仓诊断）")
    st.dataframe(post_w1_audit, use_container_width=True, hide_index=True)
    st.download_button(
        "下载V4.3全部结果ZIP", result_zip,
        file_name="weekly_skdj_confirmation_structure_audit_v4_3_all_results.zip",
        mime="application/zip", type="primary", key="v43_download", on_click="ignore")
    st.info("优先查看07验证“金叉深度×底部等待时间”，查看12验证历史1～3次波段是否冲到65/70/75，查看15～16判断买入后K/D持续分离能否作为持仓确认。")


if __name__ == "__main__":
    main()
