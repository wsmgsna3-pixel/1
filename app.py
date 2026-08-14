# -*- coding: utf-8 -*-
"""科技股周线SKDJ全量低位金叉、确认买入与退出策略审计 V4.2。

两套可独立执行的买入规则：
1. 所有符合股票池的周线SKDJ低位金叉，下一市场交易日开盘买入；
2. 低位金叉后1~6周首次上穿25确认，下一市场交易日开盘买入。

所有低位金叉都判卷，绝不使用未来是否确认来筛选立即买入样本。
固定期限是基准；止损、止盈、回撤退出和日线SKDJ高位死叉均为预注册并列审计。
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
TITLE = "科技股周线SKDJ全量低位金叉与退出策略审计 V4.2"
VERSION = "V4.2-WEEKLY-SKDJ-ALL-LOW-CROSS-CONFIRMATION-EXIT-AUDIT"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")

SKDJ_N = 9
SKDJ_M = 3
SKDJ_BOTTOM = 25.0
INDICATOR_WARMUP_WEEKS = 40
HOLD_20D = 20
HOLD_40D = 40
WAIT_MIN_WEEKS = 1
WAIT_MAX_WEEKS = 6
AUDIT_WEEKS = 5
STOP_LOSS_PCT = 10.0
TAKE_PROFITS = (10.0, 15.0, 20.0)
TRAILING_STOPS = (10.0, 15.0)
SKDJ_EXIT_LEVELS = (75.0, 80.0)

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


def first_hit_label(path: pd.DataFrame, raw_entry: float) -> tuple[str, str]:
    upper, lower = raw_entry * 1.10, raw_entry * 0.90
    for row in path.itertuples(index=False):
        hit_up = finite_num(getattr(row, "high", np.nan)) >= upper
        hit_down = finite_num(getattr(row, "low", np.nan)) <= lower
        if hit_up and hit_down:
            return "同日同时触发_保守按止损先", str(getattr(row, "trade_date", ""))
        if hit_down:
            return "先到-10%", str(getattr(row, "trade_date", ""))
        if hit_up:
            return "先到+10%", str(getattr(row, "trade_date", ""))
    return "W5内均未触发", ""


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
        "First_Hit_10_vs_Minus10_W5": "", "First_Hit_Date_W5": "",
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
            label, hit_date = first_hit_label(path, raw_entry)
            out["First_Hit_10_vs_Minus10_W5"] = label
            out["First_Hit_Date_W5"] = hit_date

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
        out["Reason"] = "可买但未来不足5个完整市场周"
    return out


EXIT_POLICY_NAMES = (
    "Fixed_W5", "SL10_TP10", "SL10_TP15", "SL10_TP20",
    "SL10_Trail10", "SL10_Trail15", "SL10_SKDJ75", "SL10_SKDJ80",
)


def trade_factors(config: dict[str, Any]) -> tuple[float, float]:
    buy_cost = (config["commission_pct"] + config["transfer_fee_pct"]) / 100.0
    sell_cost = (config["commission_pct"] + config["transfer_fee_pct"] + config["stamp_duty_pct"]) / 100.0
    buy_factor = (1 + config["buy_slippage_pct"] / 100.0) * (1 + buy_cost)
    sell_factor = (1 - config["sell_slippage_pct"] / 100.0) * (1 - sell_cost)
    return buy_factor, sell_factor


def exit_result(entry_raw: float, exit_raw: float, exit_date: str, reason: str,
                holding_days: int, config: dict[str, Any]) -> dict[str, Any]:
    buy_factor, sell_factor = trade_factors(config)
    net_return = (exit_raw * sell_factor / (entry_raw * buy_factor) - 1.0) * 100.0
    return {
        "Return_Net_pct": net_return, "Exit_Date": exit_date, "Exit_Raw_Price": exit_raw,
        "Exit_Reason": reason, "Holding_Market_Days": holding_days,
    }


def simulate_bracket(path: pd.DataFrame, entry_raw: float, take_profit_pct: float,
                     config: dict[str, Any]) -> dict[str, Any]:
    stop_price = entry_raw * (1.0 - STOP_LOSS_PCT / 100.0)
    target_price = entry_raw * (1.0 + take_profit_pct / 100.0)
    for number, row in enumerate(path.itertuples(index=False), start=1):
        day, opn = str(row.trade_date), float(row.open)
        high, low = float(row.high), float(row.low)
        if opn <= stop_price:
            return exit_result(entry_raw, opn, day, "跳空止损", number, config)
        if opn >= target_price:
            return exit_result(entry_raw, opn, day, "跳空止盈", number, config)
        hit_stop, hit_target = low <= stop_price, high >= target_price
        if hit_stop and hit_target:
            return exit_result(entry_raw, stop_price, day, "同日双触发_保守止损", number, config)
        if hit_stop:
            return exit_result(entry_raw, stop_price, day, "固定止损", number, config)
        if hit_target:
            return exit_result(entry_raw, target_price, day, "固定止盈", number, config)
    last = path.iloc[-1]
    return exit_result(entry_raw, float(last["close"]), str(last["trade_date"]), "W5到期", len(path), config)


def simulate_trailing(path: pd.DataFrame, entry_raw: float, trailing_pct: float,
                      config: dict[str, Any]) -> dict[str, Any]:
    initial_stop = entry_raw * (1.0 - STOP_LOSS_PCT / 100.0)
    prior_peak = entry_raw
    for number, row in enumerate(path.itertuples(index=False), start=1):
        day, opn = str(row.trade_date), float(row.open)
        high, low = float(row.high), float(row.low)
        stop_price = max(initial_stop, prior_peak * (1.0 - trailing_pct / 100.0))
        if opn <= stop_price:
            return exit_result(entry_raw, opn, day, "跳空回撤/止损", number, config)
        if low <= stop_price:
            return exit_result(entry_raw, stop_price, day, "最高价回撤退出", number, config)
        # 当天最高价只从下一交易日起抬高回撤线，避免不知道日内高低先后。
        prior_peak = max(prior_peak, high)
    last = path.iloc[-1]
    return exit_result(entry_raw, float(last["close"]), str(last["trade_date"]), "W5到期", len(path), config)


def simulate_skdj_exit(path: pd.DataFrame, entry_raw: float, high_level: float,
                       config: dict[str, Any]) -> dict[str, Any]:
    stop_price = entry_raw * (1.0 - STOP_LOSS_PCT / 100.0)
    armed = pending_exit = False
    for number, row in enumerate(path.itertuples(index=False), start=1):
        day, opn = str(row.trade_date), float(row.open)
        high, low = float(row.high), float(row.low)
        if pending_exit:
            return exit_result(entry_raw, opn, day, f"日线SKDJ高位死叉{int(high_level)}", number, config)
        if opn <= stop_price:
            return exit_result(entry_raw, opn, day, "跳空止损", number, config)
        if low <= stop_price:
            return exit_result(entry_raw, stop_price, day, "固定止损", number, config)
        level = finite_num(getattr(row, "D_SKDJ_Level", np.nan))
        if math.isfinite(level) and level >= high_level:
            armed = True
        death = to_bool(getattr(row, "D_SKDJ_Death_Cross", False))
        if armed and death:
            pending_exit = True
    last = path.iloc[-1]
    return exit_result(entry_raw, float(last["close"]), str(last["trade_date"]), "W5到期", len(path), config)


def simulate_exit_policies(daily: pd.DataFrame, outcome: dict[str, Any],
                           config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not to_bool(outcome.get("Tradable")) or not to_bool(outcome.get(f"Has_W{AUDIT_WEEKS}")):
        for policy in EXIT_POLICY_NAMES:
            result.update({
                f"Exit_{policy}_Return_Net_pct": np.nan, f"Exit_{policy}_Exit_Date": "",
                f"Exit_{policy}_Exit_Raw_Price": np.nan, f"Exit_{policy}_Exit_Reason": "",
                f"Exit_{policy}_Holding_Market_Days": np.nan,
            })
        return result
    entry_date = str(outcome["Entry_Date"])
    end_date = str(outcome[f"W{AUDIT_WEEKS}_End_Date"])
    entry_raw = float(outcome["Raw_Entry_Open"])
    path = daily[daily["trade_date"].astype(str).between(entry_date, end_date)].sort_values("trade_date")
    if path.empty:
        return simulate_exit_policies(daily, {"Tradable": False}, config)
    fixed = exit_result(entry_raw, float(path.iloc[-1]["close"]), end_date, "W5到期", len(path), config)
    policies: dict[str, dict[str, Any]] = {"Fixed_W5": fixed}
    for target in TAKE_PROFITS:
        policies[f"SL10_TP{int(target)}"] = simulate_bracket(path, entry_raw, target, config)
    for trailing in TRAILING_STOPS:
        policies[f"SL10_Trail{int(trailing)}"] = simulate_trailing(path, entry_raw, trailing, config)
    for level in SKDJ_EXIT_LEVELS:
        policies[f"SL10_SKDJ{int(level)}"] = simulate_skdj_exit(path, entry_raw, level, config)
    for policy, values in policies.items():
        for key, value in values.items():
            result[f"Exit_{policy}_{key}"] = value
    return result


def find_threshold_trigger(weekly: pd.DataFrame, anchor_position: int) -> int | None:
    last = min(anchor_position + WAIT_MAX_WEEKS, len(weekly) - 1)
    for position in range(anchor_position + WAIT_MIN_WEEKS, last + 1):
        previous, current = weekly.iloc[position - 1], weekly.iloc[position]
        crossed_25 = float(previous["SKDJ_K"]) < SKDJ_BOTTOM <= float(current["SKDJ_K"])
        confirmed = float(current["SKDJ_K"]) > float(current["SKDJ_D"])
        rising = float(current["SKDJ_K"]) > float(previous["SKDJ_K"])
        if crossed_25 and confirmed and rising:
            return position
    return None


def stock_trend(weekly: pd.DataFrame, position: int, daily: pd.DataFrame, signal_date: str) -> dict[str, Any]:
    close = weekly["close"]
    ma20 = close.rolling(20).mean()
    bias = finite_num((close / ma20 - 1.0).iloc[position] * 100.0)
    slope = finite_num((ma20 / ma20.shift(4) - 1.0).iloc[position] * 100.0)
    ret12 = finite_num(close.pct_change(12, fill_method=None).iloc[position] * 100.0)
    history = daily[daily["trade_date"].astype(str).le(signal_date)]
    if history.empty:
        daily_bias = np.nan
    else:
        idx = int(history.index[-1])
        daily_bias = finite_num(history.iloc[-1].get("D_MA60_Bias_pct"))
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
    anchors_all: list[tuple[int, int | None]] = []
    for position in range(INDICATOR_WARMUP_WEEKS, len(weekly)):
        row = weekly.iloc[position]
        low_cross = (
            to_bool(row.get("SKDJ_Golden_Cross"))
            and float(row["SKDJ_K"]) <= SKDJ_BOTTOM
            and float(row["SKDJ_D"]) <= SKDJ_BOTTOM
        )
        if low_cross:
            anchors_all.append((position, find_threshold_trigger(weekly, position)))

    latest_anchor_by_trigger: dict[int, int] = {}
    for anchor_position, trigger_position in anchors_all:
        if trigger_position is not None:
            latest_anchor_by_trigger[trigger_position] = max(
                latest_anchor_by_trigger.get(trigger_position, anchor_position), anchor_position)

    anchor_rows: list[dict[str, Any]] = []
    immediate_events: list[dict[str, Any]] = []
    confirmed_events: list[dict[str, Any]] = []
    code, board = str(stock["ts_code"]), sample_board(stock)
    for anchor_position, trigger_position in anchors_all:
        anchor = weekly.iloc[anchor_position]
        anchor_date = str(anchor["trade_date"])
        trigger_date = str(weekly.iloc[trigger_position]["trade_date"]) if trigger_position is not None else ""
        wait_weeks = int(trigger_position - anchor_position) if trigger_position is not None else np.nan
        selected_anchor = trigger_position is None or latest_anchor_by_trigger.get(trigger_position) == anchor_position
        full_6w = anchor_position + WAIT_MAX_WEEKS < len(weekly)
        anchor_snapshot = market_snapshot(daily_basic, anchor_date)
        anchor_membership = membership_on_date(periods, anchor_date)
        anchor_passed, anchor_reason = signal_filter(anchor_snapshot, config["min_price"], config["min_mv"])
        anchor_listed = str(stock["list_date"]) <= anchor_date < str(stock["delist_date"])
        anchor_eligible = bool(anchor_membership is not None and anchor_listed and anchor_passed)
        reject_reason = ""
        if not anchor_eligible:
            reject_reason = anchor_reason or (
                "当时不在历史科技池" if anchor_membership is None else "当时未上市或已退市")

        if config["signal_start"] <= anchor_date <= config["signal_end"]:
            if reject_reason:
                key = f"低位金叉:{reject_reason}"
                config["rejects"][key] = config["rejects"].get(key, 0) + 1
            anchor_rows.append({
                "ts_code": code, "name": str(stock["name"]), "Sample_Board": board,
                "Low_Cross_Date": anchor_date, "Low_Cross_K": float(anchor["SKDJ_K"]),
                "Low_Cross_D": float(anchor["SKDJ_D"]),
                "Low_Cross_Level": (float(anchor["SKDJ_K"]) + float(anchor["SKDJ_D"])) / 2.0,
                "Has_Full_6W_Observation": full_6w, "Confirmed_Within_6W": trigger_position is not None,
                "Selected_As_Latest_Anchor": selected_anchor, "Wait_Weeks": wait_weeks,
                "Confirm_Date": trigger_date, "Eligible_Low_Cross_Pool": anchor_eligible,
                "Low_Cross_Filter_Reason": reject_reason, **anchor_snapshot,
            })

            # 立即买规则只依赖当时低位金叉及当时股票池，不查看未来是否确认。
            if anchor_eligible and anchor_membership is not None:
                immediate = {
                    "Rule": "全量低位金叉立即买", "ts_code": code, "name": str(stock["name"]),
                    "Sample_Board": board, "SW_L1": anchor_membership["l1"],
                    "SW_L2": anchor_membership["l2"], "SW_L3": anchor_membership["l3"],
                    "Signal_Date": anchor_date, "Low_Cross_Date": anchor_date,
                    "Low_Cross_K": float(anchor["SKDJ_K"]), "Low_Cross_D": float(anchor["SKDJ_D"]),
                    "Low_Cross_Level": (float(anchor["SKDJ_K"]) + float(anchor["SKDJ_D"])) / 2.0,
                    "Future_Confirmed_Within_6W": trigger_position is not None,
                    "Future_Confirm_Date": trigger_date, "Future_Wait_Weeks": wait_weeks,
                    "Selected_As_Latest_Anchor": selected_anchor,
                    "Raw_Close": anchor_snapshot["Raw_Close"],
                    "Circ_MV_Billion": anchor_snapshot["Circ_MV_Billion"],
                    "Turnover_Rate": anchor_snapshot["Turnover_Rate"],
                    "Period_Group": "2025-06以后" if anchor_date >= config["split_date"] else "2025-06以前",
                    **stock_trend(weekly, anchor_position, daily, anchor_date),
                }
                outcome = entry_outcomes(
                    daily, anchor_date, code, open_dates, open_pos, market_weeks, config)
                immediate.update(prefix_keys(outcome, "Immediate"))
                immediate.update(prefix_keys(simulate_exit_policies(daily, outcome, config), "Immediate"))
                immediate_events.append(immediate)

        if trigger_position is None or not selected_anchor:
            continue
        if not (config["signal_start"] <= trigger_date <= config["signal_end"]):
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
        trigger_passed, trigger_reason = signal_filter(trigger_snapshot, config["min_price"], config["min_mv"])
        if not trigger_passed:
            key = f"确认日:{trigger_reason}"
            config["rejects"][key] = config["rejects"].get(key, 0) + 1
            continue
        trigger = weekly.iloc[trigger_position]
        confirmed = {
            "Rule": "低位金叉后上穿25确认买", "ts_code": code, "name": str(stock["name"]),
            "Sample_Board": board, "SW_L1": trigger_membership["l1"],
            "SW_L2": trigger_membership["l2"], "SW_L3": trigger_membership["l3"],
            "Signal_Date": trigger_date, "Low_Cross_Date": anchor_date,
            "Confirm_Date": trigger_date, "Wait_Weeks": wait_weeks,
            "Low_Cross_K": float(anchor["SKDJ_K"]), "Low_Cross_D": float(anchor["SKDJ_D"]),
            "Confirm_K": float(trigger["SKDJ_K"]), "Confirm_D": float(trigger["SKDJ_D"]),
            "Confirm_K_Change_1W": float(trigger["SKDJ_K"] - weekly.iloc[trigger_position - 1]["SKDJ_K"]),
            "Low_Cross_Raw_Close": anchor_snapshot["Raw_Close"],
            "Confirm_Raw_Close": trigger_snapshot["Raw_Close"],
            "Circ_MV_Billion": trigger_snapshot["Circ_MV_Billion"],
            "Turnover_Rate": trigger_snapshot["Turnover_Rate"],
            "Price_Change_Cross_to_Confirm_pct": (
                (trigger_snapshot["Raw_Close"] / anchor_snapshot["Raw_Close"] - 1.0) * 100.0
                if anchor_snapshot["Raw_Close"] > 0 else np.nan),
            "Period_Group": "2025-06以后" if trigger_date >= config["split_date"] else "2025-06以前",
            **stock_trend(weekly, trigger_position, daily, trigger_date),
        }
        outcome = entry_outcomes(
            daily, trigger_date, code, open_dates, open_pos, market_weeks, config)
        confirmed.update(prefix_keys(outcome, "Confirmed"))
        confirmed.update(prefix_keys(simulate_exit_policies(daily, outcome, config), "Confirmed"))
        confirmed_events.append(confirmed)
    return anchor_rows, immediate_events, confirmed_events


def max_empty_run(counts: pd.Series) -> int:
    longest = current = 0
    for value in counts.tolist():
        current = current + 1 if int(value) == 0 else 0
        longest = max(longest, current)
    return longest


def signal_week_calendar(open_dates: list[str], start: str, end: str,
                         immediate: pd.DataFrame, confirmed: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"trade_date": [day for day in open_dates if start <= day <= end]})
    frame["period"] = pd.to_datetime(frame["trade_date"]).dt.to_period("W-FRI")
    calendar = frame.groupby("period", as_index=False)["trade_date"].max().rename(columns={"trade_date": "Week_Last_Trade_Date"})
    i_counts = immediate.groupby("Signal_Date").size() if not immediate.empty else pd.Series(dtype=int)
    c_counts = confirmed.groupby("Signal_Date").size() if not confirmed.empty else pd.Series(dtype=int)
    i_mature = immediate[
        immediate["Immediate_Tradable"].map(to_bool) & immediate[f"Immediate_Has_W{AUDIT_WEEKS}"].map(to_bool)
    ] if not immediate.empty else immediate
    c_mature = confirmed[
        confirmed["Confirmed_Tradable"].map(to_bool) & confirmed[f"Confirmed_Has_W{AUDIT_WEEKS}"].map(to_bool)
    ] if not confirmed.empty else confirmed
    calendar["Immediate_Signals"] = calendar["Week_Last_Trade_Date"].map(i_counts).fillna(0).astype(int)
    calendar["Confirmed_Signals"] = calendar["Week_Last_Trade_Date"].map(c_counts).fillna(0).astype(int)
    i_mature_counts = i_mature.groupby("Signal_Date").size() if not i_mature.empty else pd.Series(dtype=int)
    c_mature_counts = c_mature.groupby("Signal_Date").size() if not c_mature.empty else pd.Series(dtype=int)
    calendar["Immediate_Mature"] = calendar["Week_Last_Trade_Date"].map(i_mature_counts).fillna(0).astype(int)
    calendar["Confirmed_Mature"] = calendar["Week_Last_Trade_Date"].map(c_mature_counts).fillna(0).astype(int)
    return calendar.drop(columns="period")


def outcome_summary(frame: pd.DataFrame, prefix: str, label: str) -> dict[str, Any]:
    tradable = frame[frame[f"{prefix}_Tradable"].map(to_bool)].copy() if not frame.empty else frame
    mature = tradable[tradable[f"{prefix}_Has_W{AUDIT_WEEKS}"].map(to_bool)].copy() if not tradable.empty else tradable
    mfe = pd.to_numeric(mature.get(f"{prefix}_W{AUDIT_WEEKS}_Cum_Max_High_Raw_pct"), errors="coerce")
    close_ret = pd.to_numeric(mature.get(f"{prefix}_W{AUDIT_WEEKS}_Close_Return_Net_pct"), errors="coerce")
    mae = pd.to_numeric(mature.get(f"{prefix}_W{AUDIT_WEEKS}_Cum_MAE_Raw_pct"), errors="coerce")
    first_hit = mature.get(f"{prefix}_First_Hit_10_vs_Minus10_W5", pd.Series(dtype=str)).astype(str)
    return {
        "组别": label, "事件": len(frame), "可买": len(tradable), "W5成熟": len(mature),
        "不同股票": mature["ts_code"].nunique() if not mature.empty else 0,
        "W5最高涨幅均值%": mfe.mean(), "W5最高涨幅中位数%": mfe.median(),
        "W5达到10%比例%": mfe.ge(10).mean() * 100 if len(mfe) else np.nan,
        "W5达到20%比例%": mfe.ge(20).mean() * 100 if len(mfe) else np.nan,
        "W5达到30%比例%": mfe.ge(30).mean() * 100 if len(mfe) else np.nan,
        "W5期末净收益均值%": close_ret.mean(), "W5期末净收益中位数%": close_ret.median(),
        "W5期末胜率%": close_ret.gt(0).mean() * 100 if len(close_ret) else np.nan,
        "W5最大回撤中位数%": mae.median(),
        "先到+10%比例%": first_hit.eq("先到+10%").mean() * 100 if len(first_hit) else np.nan,
        "先到-10%比例%": first_hit.str.contains("-10%|止损先", regex=True).mean() * 100 if len(first_hit) else np.nan,
        "20日净收益均值%": pd.to_numeric(tradable.get(f"{prefix}_Return_20D_Net_pct"), errors="coerce").mean(),
        "40日净收益均值%": pd.to_numeric(tradable.get(f"{prefix}_Return_40D_Net_pct"), errors="coerce").mean(),
        "40日净收益中位数%": pd.to_numeric(tradable.get(f"{prefix}_Return_40D_Net_pct"), errors="coerce").median(),
    }


def grouped_outcome_audit(events: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    rows = []
    for value, group in events.groupby(column, dropna=False, sort=True):
        rows.append(outcome_summary(group, prefix, str(value)))
    return pd.DataFrame(rows)


def weekly_path_audit(events: pd.DataFrame, prefix: str) -> pd.DataFrame:
    mature = events[events[f"{prefix}_Tradable"].map(to_bool)].copy()
    rows = []
    for week in range(1, AUDIT_WEEKS + 1):
        sample = mature[mature[f"{prefix}_Has_W{week}"].map(to_bool)]
        mfe = pd.to_numeric(sample[f"{prefix}_W{week}_Cum_Max_High_Raw_pct"], errors="coerce")
        close_ret = pd.to_numeric(sample[f"{prefix}_W{week}_Close_Return_Net_pct"], errors="coerce")
        mae = pd.to_numeric(sample[f"{prefix}_W{week}_Cum_MAE_Raw_pct"], errors="coerce")
        rows.append({
            "持有到": f"W{week}", "成熟事件": len(sample), "最高涨幅均值%": mfe.mean(),
            "最高涨幅中位数%": mfe.median(), "达到10%比例%": mfe.ge(10).mean() * 100 if len(mfe) else np.nan,
            "达到20%比例%": mfe.ge(20).mean() * 100 if len(mfe) else np.nan,
            "达到30%比例%": mfe.ge(30).mean() * 100 if len(mfe) else np.nan,
            "期末净收益均值%": close_ret.mean(), "期末净收益中位数%": close_ret.median(),
            "期末胜率%": close_ret.gt(0).mean() * 100 if len(close_ret) else np.nan,
            "最大回撤中位数%": mae.median(),
        })
    return pd.DataFrame(rows)


def exit_policy_summary(events: pd.DataFrame, prefix: str, group_label: str) -> pd.DataFrame:
    mature = events[
        events[f"{prefix}_Tradable"].map(to_bool)
        & events[f"{prefix}_Has_W{AUDIT_WEEKS}"].map(to_bool)
    ].copy() if not events.empty else events
    rows = []
    for policy in EXIT_POLICY_NAMES:
        return_col = f"{prefix}_Exit_{policy}_Return_Net_pct"
        reason_col = f"{prefix}_Exit_{policy}_Exit_Reason"
        days_col = f"{prefix}_Exit_{policy}_Holding_Market_Days"
        values = pd.to_numeric(mature.get(return_col), errors="coerce").dropna()
        reasons = mature.loc[values.index, reason_col].astype(str) if len(values) else pd.Series(dtype=str)
        days = pd.to_numeric(mature.loc[values.index, days_col], errors="coerce") if len(values) else pd.Series(dtype=float)
        sorted_values = values.sort_values(ascending=False)
        after_top10 = sorted_values.iloc[min(10, len(sorted_values)):]
        rows.append({
            "买入规则": group_label, "退出策略": policy, "成熟事件": len(values),
            "不同股票": mature.loc[values.index, "ts_code"].nunique() if len(values) else 0,
            "平均净收益%": values.mean(), "中位净收益%": values.median(),
            "胜率%": values.gt(0).mean() * 100 if len(values) else np.nan,
            "收益≥10%比例%": values.ge(10).mean() * 100 if len(values) else np.nan,
            "亏损≤-10%比例%": values.le(-10).mean() * 100 if len(values) else np.nan,
            "平均持仓交易日": days.mean(), "持仓中位数交易日": days.median(),
            "止盈退出比例%": reasons.str.contains("止盈").mean() * 100 if len(reasons) else np.nan,
            "止损/回撤退出比例%": reasons.str.contains("止损|回撤").mean() * 100 if len(reasons) else np.nan,
            "SKDJ退出比例%": reasons.str.contains("SKDJ").mean() * 100 if len(reasons) else np.nan,
            "到期退出比例%": reasons.eq("W5到期").mean() * 100 if len(reasons) else np.nan,
            "剔除最赚钱10笔后均值%": after_top10.mean() if len(after_top10) else np.nan,
        })
    return pd.DataFrame(rows)


def exit_stability(events: pd.DataFrame, prefix: str, group_column: str,
                   rule_label: str) -> pd.DataFrame:
    rows = []
    for value, group in events.groupby(group_column, dropna=False, sort=True):
        audit = exit_policy_summary(group, prefix, rule_label)
        audit.insert(1, "分组", str(value))
        rows.append(audit)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def confirmation_audit(anchors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = anchors[anchors["Eligible_Low_Cross_Pool"].map(to_bool) & anchors["Has_Full_6W_Observation"].map(to_bool)].copy()
    unique = eligible[eligible["Selected_As_Latest_Anchor"].map(to_bool)].copy()
    rows = []
    for label, frame in (("全部低位金叉", eligible), ("去除同一确认周的重复锚点", unique)):
        confirmed = frame[frame["Confirmed_Within_6W"].map(to_bool)]
        rows.append({
            "口径": label, "完整观察低位金叉": len(frame), "六周内确认": len(confirmed),
            "六周确认率%": len(confirmed) / len(frame) * 100 if len(frame) else np.nan,
            "不同股票": frame["ts_code"].nunique(),
        })
    lag = unique[unique["Confirmed_Within_6W"].map(to_bool)].groupby("Wait_Weeks").agg(
        确认事件=("ts_code", "size"), 不同股票=("ts_code", "nunique"), 确认周=("Confirm_Date", "nunique")
    ).reset_index()
    return pd.DataFrame(rows), lag


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title="周线SKDJ全量低位金叉退出审计 V4.2", layout="wide")
    st.title(TITLE)
    st.caption("无未来信息比较全量低位金叉立即买与上穿25确认买，并列审计预先规定的退出方法。")
    with st.expander("V4.2验证规则", expanded=True):
        st.markdown(f"""
- **立即买规则**：完整周线K上穿D，且当周K、D均≤25；所有符合股票池的事件下一市场交易日开盘买入，包括未来六周不确认的失败信号。
- **确认买规则**：低位金叉后第1～6周，K首次从25下方上穿25，同时K>D且K继续上升；确认周后下一市场交易日开盘买入。
- **基础股票池**：申万历史科技行业，主板/创业板/科创板；信号日原始股价≥10元、流通市值≥100亿元。
- **固定基准**：W1～W5累计最高涨幅、期末净收益、最大回撤，以及固定20/40市场交易日收益。
- **止盈止损**：统一初始止损-10%，分别搭配固定止盈+10%、+15%、+20%；同日上下同时触发时保守按止损先。
- **回撤退出**：统一初始止损-10%，分别从历史最高价回撤10%或15%；当天最高价只从下一交易日起抬高回撤线。
- **SKDJ退出**：统一初始止损-10%；日线SKDJ曾达到75或80后发生死叉，下一市场交易日开盘卖出；最长持有到W5。
- **成本**：买卖均计滑点、佣金和过户费，卖出另计印花税；所有策略按同一成本口径比较。
- **限制**：退出参数是预先规定的审计组，不按本次结果挑选最佳参数；固定期限始终保留为基准。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("信号开始", date(2023, 6, 5), key="v42_start")
        signal_end_date = st.date_input("信号截止", date(2026, 6, 5), key="v42_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v42_market_end")
        split_date_value = st.date_input("近期行情分界", date(2025, 6, 1), key="v42_split")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v42_pause")
        use_cache = st.checkbox("复用逐股票缓存", True, key="v42_cache")
        st.divider()
        commission_pct = st.number_input("佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f", key="v42_commission")
        stamp_duty_pct = st.number_input("卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f", key="v42_stamp")
        transfer_fee_pct = st.number_input("过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f", key="v42_transfer")
        if st.button("清除本程序行情缓存", key="v42_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("缓存已清除")

    token = st.text_input("Tushare Token", type="password", key="v42_token")
    session_key = "weekly_skdj_all_low_cross_exit_v42_zip"
    if not token:
        st.info("请输入Tushare Token；日期范围一致时可直接复用V3.4～V4.1逐股票缓存。")
        return
    if not st.button("开始V4.2全量低位金叉与退出审计", type="primary", key="v42_run"):
        if session_key in st.session_state:
            st.download_button(
                "下载上一次结果ZIP", st.session_state[session_key],
                file_name="weekly_skdj_all_low_cross_exit_audit_v4_2_all_results.zip",
                mime="application/zip", on_click="ignore")
        return
    error = validate_dates(signal_start_date, signal_end_date, market_end_date)
    if error:
        st.error(error)
        return
    if (market_end_date - signal_end_date).days < 45:
        st.warning("观察截止日距离信号截止日不足45天，末端部分事件不会有完整W5结果，但程序仍会继续。")

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
    stocks = stocks[~stocks["list_date"].gt(signal_end) & ~stocks["delist_date"].lt(preload)]
    stocks["Sample_Board"] = stocks.apply(sample_board, axis=1)
    stocks = stocks.sort_values("ts_code").reset_index(drop=True)
    population = stocks.groupby("Sample_Board").size().reindex(BOARDS, fill_value=0).rename("股票数").reset_index()
    open_pos = {day: position for position, day in enumerate(open_dates)}

    anchor_rows: list[dict[str, Any]] = []
    immediate_rows: list[dict[str, Any]] = []
    confirmed_rows: list[dict[str, Any]] = []
    cache_hits = data_failures = 0
    progress, status = st.progress(0.0), st.empty()
    for number, stock in stocks.iterrows():
        code = str(stock["ts_code"])
        progress.progress((number + 1) / max(len(stocks), 1), text=f"{number + 1}/{len(stocks)} {code}")
        status.caption(
            f"符合池低位金叉 {len(immediate_rows)}；上穿25确认 {len(confirmed_rows)}；"
            f"缓存 {cache_hits}；失败 {data_failures}")
        daily, daily_basic, cache_hit = fetch_stock_history(
            code, preload, market_end, bool(use_cache), float(pause))
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        anchors, immediate_part, confirmed_part = analyze_stock(
            stock, period_index.get(code, []), daily, daily_basic, week_last_map,
            open_dates, open_pos, market_weeks, config)
        anchor_rows.extend(anchors)
        immediate_rows.extend(immediate_part)
        confirmed_rows.extend(confirmed_part)
    progress.empty()
    status.empty()
    if not immediate_rows:
        st.error("研究区间没有生成符合股票池的周线SKDJ低位金叉立即买事件。")
        return

    anchors = pd.DataFrame(anchor_rows)
    immediate = pd.DataFrame(immediate_rows)
    confirmed = pd.DataFrame(confirmed_rows)
    for frame in (immediate, confirmed):
        if not frame.empty:
            dt = pd.to_datetime(frame["Signal_Date"].astype(str), format="%Y%m%d", errors="coerce")
            frame["Signal_Year"] = frame["Signal_Date"].astype(str).str[:4]
            frame["Signal_Half_Year"] = frame["Signal_Year"] + "H" + np.where(dt.dt.month.le(6), "1", "2")

    confirmation_summary, confirmation_lag = confirmation_audit(anchors)
    rule_quality_rows = [outcome_summary(immediate, "Immediate", "全量低位金叉立即买")]
    if not confirmed.empty:
        rule_quality_rows.append(outcome_summary(confirmed, "Confirmed", "低位金叉后上穿25确认买"))
    rule_quality = pd.DataFrame(rule_quality_rows)
    immediate_path = weekly_path_audit(immediate, "Immediate")
    confirmed_path = weekly_path_audit(confirmed, "Confirmed") if not confirmed.empty else pd.DataFrame()
    future_confirmation_diagnostic = grouped_outcome_audit(
        immediate, "Future_Confirmed_Within_6W", "Immediate")
    immediate_trend = grouped_outcome_audit(immediate, "Individual_Trend", "Immediate")
    confirmed_trend = grouped_outcome_audit(
        confirmed, "Individual_Trend", "Confirmed") if not confirmed.empty else pd.DataFrame()
    confirmed_wait = grouped_outcome_audit(
        confirmed, "Wait_Weeks", "Confirmed") if not confirmed.empty else pd.DataFrame()

    immediate_exit = exit_policy_summary(immediate, "Immediate", "全量低位金叉立即买")
    confirmed_exit = exit_policy_summary(
        confirmed, "Confirmed", "上穿25确认买") if not confirmed.empty else pd.DataFrame()
    exit_comparison = pd.concat([immediate_exit, confirmed_exit], ignore_index=True)
    immediate_exit_year = exit_stability(
        immediate, "Immediate", "Signal_Year", "全量低位金叉立即买")
    confirmed_exit_year = exit_stability(
        confirmed, "Confirmed", "Signal_Year", "上穿25确认买") if not confirmed.empty else pd.DataFrame()
    exit_year = pd.concat([immediate_exit_year, confirmed_exit_year], ignore_index=True)
    immediate_exit_half = exit_stability(
        immediate, "Immediate", "Signal_Half_Year", "全量低位金叉立即买")
    confirmed_exit_half = exit_stability(
        confirmed, "Confirmed", "Signal_Half_Year", "上穿25确认买") if not confirmed.empty else pd.DataFrame()
    exit_half = pd.concat([immediate_exit_half, confirmed_exit_half], ignore_index=True)

    calendar = signal_week_calendar(open_dates, signal_start, signal_end, immediate, confirmed)
    i_counts, c_counts = calendar["Immediate_Signals"], calendar["Confirmed_Signals"]
    i_mature = immediate[
        immediate["Immediate_Tradable"].map(to_bool)
        & immediate[f"Immediate_Has_W{AUDIT_WEEKS}"].map(to_bool)]
    c_mature = confirmed[
        confirmed["Confirmed_Tradable"].map(to_bool)
        & confirmed[f"Confirmed_Has_W{AUDIT_WEEKS}"].map(to_bool)
    ] if not confirmed.empty else confirmed

    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "信号开始": signal_start, "信号截止": signal_end,
        "观察截止": market_end, "全量低位金叉事件": len(immediate),
        "低位金叉不同股票": immediate["ts_code"].nunique(), "低位金叉W5成熟可买": len(i_mature),
        "确认买事件": len(confirmed), "确认买不同股票": confirmed["ts_code"].nunique() if not confirmed.empty else 0,
        "确认买W5成熟可买": len(c_mature), "自然周": len(calendar),
        "低位金叉有信号周": int(i_counts.gt(0).sum()), "低位金叉空窗周": int(i_counts.eq(0).sum()),
        "低位金叉最长空窗": max_empty_run(i_counts), "低位金叉每周中位数": i_counts.median(),
        "低位金叉单周最多": i_counts.max(), "确认买有信号周": int(c_counts.gt(0).sum()),
        "确认买空窗周": int(c_counts.eq(0).sum()), "确认买最长空窗": max_empty_run(c_counts),
        "确认买每周中位数": c_counts.median(), "确认买单周最多": c_counts.max(),
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    unconfirmed = anchors[
        anchors["Eligible_Low_Cross_Pool"].map(to_bool)
        & anchors["Has_Full_6W_Observation"].map(to_bool)
        & ~anchors["Confirmed_Within_6W"].map(to_bool)
    ].copy()
    exit_dictionary = pd.DataFrame([
        ("Fixed_W5", "不设中途退出，W5最后一个可用收盘价卖出"),
        ("SL10_TP10", "-10%固定止损，+10%固定止盈，最长W5"),
        ("SL10_TP15", "-10%固定止损，+15%固定止盈，最长W5"),
        ("SL10_TP20", "-10%固定止损，+20%固定止盈，最长W5"),
        ("SL10_Trail10", "-10%初始止损，历史最高价回撤10%退出，最长W5"),
        ("SL10_Trail15", "-10%初始止损，历史最高价回撤15%退出，最长W5"),
        ("SL10_SKDJ75", "-10%初始止损，日线SKDJ达到75后死叉，下一交易日开盘退出，最长W5"),
        ("SL10_SKDJ80", "-10%初始止损，日线SKDJ达到80后死叉，下一交易日开盘退出，最长W5"),
    ], columns=["退出策略", "定义"])
    metadata = pd.DataFrame([
        ("立即买", f"完整周线SKDJ(N={SKDJ_N},M={SKDJ_M}) K上穿D且K、D均≤25；所有符合池事件均纳入，未来是否确认不参与筛选"),
        ("确认买", "低位金叉后第1至6周，K首次从25下方上穿25、K>D且K继续上升；同一确认周只保留最近锚点"),
        ("股票池", "申万历史科技行业；主板/创业板/科创板；信号日股价≥10元、流通市值≥100亿元"),
        ("买入", "信号周结束后下一市场交易日开盘；主板一字板不买"),
        ("未来确认字段", "立即买明细中的Future_Confirmed字段只用于事后诊断，绝不用于全量立即买规则筛选或排序"),
        ("W1至W5", "W1是买入所在市场周，累计最高价与回撤截至相应周；期末收益计完整交易成本"),
        ("止损止盈日内顺序", "先检查跳空开盘；盘中止损与止盈同日都达到时保守按止损先"),
        ("回撤线", "只用前一交易日及以前已形成的最高价计算当天回撤线，避免用当天最高价判断当天更早发生的最低价"),
        ("SKDJ退出", "高位死叉在当日收盘确认，下一市场交易日开盘执行；盘中固定止损仍优先保护"),
        ("固定对照", "固定W5、20日、40日始终保留；退出策略不得只凭本次全样本最好成绩定稿"),
        ("未使用", "分钟线、月线、评分模型、未来行情筛选、事后自动挑参数"),
    ], columns=["项目", "说明"])

    files = {
        "01_run_summary_v4_2.csv": run_summary,
        "02_executable_buy_rule_quality_v4_2.csv": rule_quality,
        "03_low_cross_confirmation_rate_v4_2.csv": confirmation_summary,
        "04_confirmation_count_by_wait_week_v4_2.csv": confirmation_lag,
        "05_immediate_w1_w5_path_v4_2.csv": immediate_path,
        "06_confirmed_w1_w5_path_v4_2.csv": confirmed_path,
        "07_immediate_future_confirmation_diagnostic_v4_2.csv": future_confirmation_diagnostic,
        "08_exit_policy_comparison_v4_2.csv": exit_comparison,
        "09_exit_policy_year_stability_v4_2.csv": exit_year,
        "10_exit_policy_half_year_stability_v4_2.csv": exit_half,
        "11_immediate_individual_trend_v4_2.csv": immediate_trend,
        "12_confirmed_individual_trend_v4_2.csv": confirmed_trend,
        "13_confirmed_wait_week_quality_v4_2.csv": confirmed_wait,
        "14_weekly_signal_calendar_v4_2.csv": calendar,
        "15_all_immediate_low_cross_events_v4_2.csv": immediate,
        "16_all_confirmed_events_v4_2.csv": confirmed,
        "17_all_low_cross_anchor_detail_v4_2.csv": anchors,
        "18_unconfirmed_after_6w_v4_2.csv": unconfirmed,
        "19_exit_policy_dictionary_v4_2.csv": exit_dictionary,
        "20_full_tech_universe_v4_2.csv": stocks,
        "21_board_population_v4_2.csv": population,
        "22_rejection_audit_v4_2.csv": pd.DataFrame(
            [{"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]),
        "23_api_errors_v4_2.csv": pd.DataFrame({"错误": API_ERRORS}),
        "24_metadata_v4_2.csv": metadata,
    }
    result_zip = make_zip(files)
    st.session_state[session_key] = result_zip
    st.success(
        f"完成：全量低位金叉{len(immediate)}个，W5成熟可买{len(i_mature)}个；"
        f"确认买{len(confirmed)}个，W5成熟可买{len(c_mature)}个。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("全量低位金叉", len(immediate))
    c2.metric("确认买事件", len(confirmed))
    c3.metric("低位金叉空窗周", int(i_counts.eq(0).sum()))
    c4.metric("确认买空窗周", int(c_counts.eq(0).sum()))
    st.subheader("两套可执行买入规则的统一判卷")
    st.dataframe(rule_quality, use_container_width=True, hide_index=True)
    st.subheader("全部低位金叉：未来是否确认的事后诊断")
    st.dataframe(future_confirmation_diagnostic, use_container_width=True, hide_index=True)
    st.subheader("退出策略总体比较")
    st.dataframe(exit_comparison, use_container_width=True, hide_index=True)
    st.subheader("退出策略分年稳定性")
    st.dataframe(exit_year, use_container_width=True, hide_index=True)
    st.download_button(
        "下载V4.2全部结果ZIP", result_zip,
        file_name="weekly_skdj_all_low_cross_exit_audit_v4_2_all_results.zip",
        mime="application/zip", type="primary", key="v42_download", on_click="ignore")
    st.info("先看02判断全量低位金叉是否有真实优势；再看07确认未来不上穿25的失败样本造成多大拖累；最后看08～10，只有跨年份和半年稳定的退出方法才值得进入下一轮。")


if __name__ == "__main__":
    main()
