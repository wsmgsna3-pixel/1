# -*- coding: utf-8 -*-
"""
科技股周线SKDJ核心池＋日线SKDJ买卖点审计器 V3.3

研究问题：
1. 周线SKDJ底部重置金叉出现时，日线SKDJ处于什么状态？
2. 直接买、机械等日线SKDJ金叉、状态分流买、普通KDJ金叉哪种滞后更小？
3. 日线SKDJ任意死叉、高位死叉、最高收盘回撤10%/15%及混合退出哪种更稳定？
4. 三仓条件下，同日竞争和随机顺序是否仍会改变结论？

所有指标信号均以收盘确认，下一个交易日开盘成交；不使用分钟数据。

运行：streamlit run app.py
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


TITLE = "科技股周线SKDJ核心池＋日线买卖点审计器 V3.3"
VERSION = "V3.3-WEEKLY-SKDJ-DAILY-ENTRY-EXIT-AUDIT"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# 复用同目录逐股票行情缓存，日期相同时无需重新下载。
CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")

SKDJ_N = 9
SKDJ_M = 3
SKDJ_BOTTOM = 25.0
RESET_LOOKBACK_WEEKS = 3
CROSS_ZONE_LOW = 20.0
CROSS_ZONE_HIGH = 35.0
DAILY_SKDJ_WAIT_DAYS = 10
STATE_FLOW_WAIT_DAYS = 5
DAILY_KDJ_WAIT_DAYS = 10
HIGH_ZONE = 75.0
HOLD_TRADING_DAYS = 40
INDICATOR_WARMUP_WEEKS = 40

INITIAL_CAPITAL = 300_000.0
MAX_POSITIONS = 3
POSITION_BUDGET = 100_000.0
LOT_SIZE = 100
MC_RUNS = 100
PRIMARY_SEED = 20260813

ENTRY_METHODS = {
    "Direct": "周线金叉次日直接买",
    "Daily_SKDJ_Gold": "等日线SKDJ金叉",
    "State_Flow": "日线SKDJ状态分流买",
    "Daily_KDJ_Gold": "等普通KDJ金叉",
}
EXIT_METHODS = {
    "Any_Death": "日线SKDJ任意死叉",
    "High_Death": "日线SKDJ高位死叉",
    "Trail_10": "最高收盘回撤10%",
    "Trail_15": "最高收盘回撤15%",
    "Hybrid_High_Death_Trail_15": "高位死叉或回撤15%",
}

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


# -----------------------------------------------------------------------------
# 通用工具
# -----------------------------------------------------------------------------
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


def atomic_pickle(payload: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = f"{path}.tmp"
    with open(temp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


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


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(files: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in files.items():
            archive.writestr(name, csv_bytes(frame))
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# 历史科技股票池
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# 行情、缓存与完整周线
# -----------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600)
def load_trade_calendar(start_date: str, end_date: str) -> list[str]:
    frame = safe_get("trade_cal", required=True, exchange="SSE", start_date=start_date, end_date=end_date)
    if frame.empty:
        raise RuntimeError("交易日历为空")
    return sorted(frame.loc[frame["is_open"].eq(1), "cal_date"].astype(str).tolist())


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


def add_daily_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy().sort_values("trade_date").reset_index(drop=True)
    work = add_skdj(work).rename(columns={
        "SKDJ_K": "D_SKDJ_K", "SKDJ_D": "D_SKDJ_D",
        "SKDJ_Golden_Cross": "D_SKDJ_Golden_Cross",
        "SKDJ_Death_Cross": "D_SKDJ_Death_Cross",
    })
    # 普通KDJ只作为更敏感的对照组：9,3,3，J=3K-2D。
    lowv = work["low"].rolling(SKDJ_N, min_periods=SKDJ_N).min()
    highv = work["high"].rolling(SKDJ_N, min_periods=SKDJ_N).max()
    rsv = (work["close"] - lowv) / (highv - lowv).replace(0, np.nan) * 100.0
    work["D_KDJ_K"] = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=1).mean()
    work["D_KDJ_D"] = work["D_KDJ_K"].ewm(alpha=1 / 3, adjust=False, min_periods=1).mean()
    work["D_KDJ_J"] = 3.0 * work["D_KDJ_K"] - 2.0 * work["D_KDJ_D"]
    work["D_KDJ_Golden_Cross"] = work["D_KDJ_K"].gt(work["D_KDJ_D"]) & work["D_KDJ_K"].shift(1).le(work["D_KDJ_D"].shift(1))
    work["D_SKDJ_K_Change_1D"] = work["D_SKDJ_K"].diff()
    work["D_SKDJ_K_Change_3D"] = work["D_SKDJ_K"].diff(3)
    work["D_SKDJ_D_Change_1D"] = work["D_SKDJ_D"].diff()
    work["D_SKDJ_Level"] = (work["D_SKDJ_K"] + work["D_SKDJ_D"]) / 2.0
    work["D_SKDJ_Prev_Level"] = work["D_SKDJ_Level"].shift(1)
    return work


# -----------------------------------------------------------------------------
# 事件、日线状态与买卖规则
# -----------------------------------------------------------------------------
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


def value_bin(value: Any) -> str:
    number = finite_num(value)
    if not math.isfinite(number):
        return "缺失"
    if number < 25:
        return "<25"
    if number < 50:
        return "25-50"
    if number < 75:
        return "50-75"
    return "≥75"


def weekly_cross_bin(value: float) -> str:
    if value <= 20:
        return "≤20"
    if value <= 25:
        return "20-25"
    if value <= 35:
        return "25-35"
    if value <= 50:
        return "35-50"
    return ">50"


def reset_quadrant(touched_bottom: bool, in_cross_zone: bool) -> str:
    if touched_bottom and in_cross_zone:
        return "触及25_金叉20-35"
    if touched_bottom:
        return "触及25_金叉区间外"
    if in_cross_zone:
        return "未触及25_金叉20-35"
    return "未触及25_金叉区间外"


def next_weekly_death_date(weekly: pd.DataFrame, position: int) -> str:
    future = weekly.iloc[position + 1:]
    hits = future[future["SKDJ_Death_Cross"].map(to_bool)]
    return str(hits.iloc[0]["trade_date"]) if not hits.empty else ""


def daily_state_at_cross(daily_ind: pd.DataFrame, signal_date: str) -> dict[str, Any]:
    result = {
        "Daily_SKDJ_K_At_Cross": np.nan, "Daily_SKDJ_D_At_Cross": np.nan,
        "Daily_SKDJ_Level_At_Cross": np.nan, "Daily_SKDJ_Level_Bin": "缺失",
        "Daily_SKDJ_K_Change_1D": np.nan, "Daily_SKDJ_K_Change_3D": np.nan,
        "Daily_SKDJ_D_Change_1D": np.nan, "Daily_SKDJ_Golden_At_Cross": False,
        "Daily_SKDJ_Death_At_Cross": False, "Daily_SKDJ_State_At_Cross": "缺少日线行情",
        "Last_Daily_SKDJ_Golden_Date": "", "Days_Since_Daily_SKDJ_Golden": np.nan,
        "Daily_KDJ_K_At_Cross": np.nan, "Daily_KDJ_D_At_Cross": np.nan,
        "Daily_KDJ_J_At_Cross": np.nan, "Daily_KDJ_Golden_At_Cross": False,
    }
    positions = daily_ind.index[daily_ind["trade_date"].astype(str).eq(signal_date)].tolist()
    if not positions:
        return result
    pos = int(positions[-1])
    row = daily_ind.iloc[pos]
    k, d = finite_num(row["D_SKDJ_K"]), finite_num(row["D_SKDJ_D"])
    level, change = finite_num(row["D_SKDJ_Level"]), finite_num(row["D_SKDJ_K_Change_1D"])
    golden, death = to_bool(row["D_SKDJ_Golden_Cross"]), to_bool(row["D_SKDJ_Death_Cross"])
    if golden:
        state = "当日金叉"
    elif death:
        state = "当日死叉"
    elif k > d and change > 0:
        state = "K>D且K上升"
    elif k > d:
        state = "K>D但K不升"
    elif change > 0:
        state = "K≤D但K上升"
    else:
        state = "K≤D且K不升"
    prior = daily_ind.iloc[:pos + 1]
    prior = prior[prior["D_SKDJ_Golden_Cross"].map(to_bool)]
    if not prior.empty:
        last_pos = int(prior.index[-1])
        last_date = str(daily_ind.loc[last_pos, "trade_date"])
        age = float(pos - last_pos)
    else:
        last_date, age = "", np.nan
    result.update({
        "Daily_SKDJ_K_At_Cross": k, "Daily_SKDJ_D_At_Cross": d,
        "Daily_SKDJ_Level_At_Cross": level, "Daily_SKDJ_Level_Bin": value_bin(level),
        "Daily_SKDJ_K_Change_1D": change,
        "Daily_SKDJ_K_Change_3D": finite_num(row["D_SKDJ_K_Change_3D"]),
        "Daily_SKDJ_D_Change_1D": finite_num(row["D_SKDJ_D_Change_1D"]),
        "Daily_SKDJ_Golden_At_Cross": golden, "Daily_SKDJ_Death_At_Cross": death,
        "Daily_SKDJ_State_At_Cross": state,
        "Last_Daily_SKDJ_Golden_Date": last_date, "Days_Since_Daily_SKDJ_Golden": age,
        "Daily_KDJ_K_At_Cross": finite_num(row["D_KDJ_K"]),
        "Daily_KDJ_D_At_Cross": finite_num(row["D_KDJ_D"]),
        "Daily_KDJ_J_At_Cross": finite_num(row["D_KDJ_J"]),
        "Daily_KDJ_Golden_At_Cross": to_bool(row["D_KDJ_Golden_Cross"]),
    })
    return result


def find_entry_trigger(daily_ind: pd.DataFrame, signal_date: str, death_date: str,
                       market_end: str, open_dates: list[str], open_pos: dict[str, int],
                       method: str) -> dict[str, Any]:
    result = {"Triggered": False, "Trigger_Date": "", "Trigger_Mode": "未触发",
              "Wait_Market_Days": np.nan, "No_Trigger_Reason": "等待窗口内没有信号"}
    positions = daily_ind.index[daily_ind["trade_date"].astype(str).eq(signal_date)].tolist()
    if not positions or signal_date not in open_pos:
        result["No_Trigger_Reason"] = "周线信号日无个股行情或不在交易日历"
        return result
    pos = int(positions[-1])
    row = daily_ind.iloc[pos]
    if method == "Direct":
        result.update({"Triggered": True, "Trigger_Date": signal_date,
                       "Trigger_Mode": "周线金叉收盘确认", "Wait_Market_Days": 0.0,
                       "No_Trigger_Reason": ""})
        return result
    wait_days = STATE_FLOW_WAIT_DAYS if method == "State_Flow" else (
        DAILY_KDJ_WAIT_DAYS if method == "Daily_KDJ_Gold" else DAILY_SKDJ_WAIT_DAYS)
    deadline_pos = min(open_pos[signal_date] + wait_days, len(open_dates) - 1)
    deadline = min(open_dates[deadline_pos], market_end)
    if method == "State_Flow" and finite_num(row["D_SKDJ_K"]) > finite_num(row["D_SKDJ_D"]) \
            and finite_num(row["D_SKDJ_K_Change_1D"]) > 0:
        result.update({"Triggered": True, "Trigger_Date": signal_date,
                       "Trigger_Mode": "当日K>D且K上升_直接买", "Wait_Market_Days": 0.0,
                       "No_Trigger_Reason": ""})
        return result
    search = daily_ind[
        daily_ind["trade_date"].astype(str).gt(signal_date)
        & daily_ind["trade_date"].astype(str).le(deadline)
    ].copy()
    if death_date:
        search = search[search["trade_date"].astype(str).lt(death_date)]
    if method == "Daily_SKDJ_Gold":
        if to_bool(row["D_SKDJ_Golden_Cross"]):
            search = daily_ind.iloc[[pos]].copy()
        else:
            search = search[search["D_SKDJ_Golden_Cross"].map(to_bool)]
        mode = "日线SKDJ金叉"
    elif method == "Daily_KDJ_Gold":
        if to_bool(row["D_KDJ_Golden_Cross"]):
            search = daily_ind.iloc[[pos]].copy()
        else:
            search = search[search["D_KDJ_Golden_Cross"].map(to_bool)]
        mode = "普通KDJ金叉"
    elif method == "State_Flow":
        initial_above = finite_num(row["D_SKDJ_K"]) > finite_num(row["D_SKDJ_D"])
        if initial_above:
            search = search[search["D_SKDJ_K"].gt(search["D_SKDJ_D"])
                            & search["D_SKDJ_K_Change_1D"].gt(0)]
            mode = "K>D回落后K重新上升"
        else:
            search = search[search["D_SKDJ_Golden_Cross"].map(to_bool)]
            mode = "K≤D后等待日线SKDJ金叉"
    else:
        raise ValueError(f"不支持的买入方法: {method}")
    if search.empty:
        if death_date and death_date <= deadline:
            result["No_Trigger_Reason"] = "等待期间先出现完整周线SKDJ死叉"
        return result
    trigger_date = str(search.iloc[0]["trade_date"])
    result.update({"Triggered": True, "Trigger_Date": trigger_date, "Trigger_Mode": mode,
                   "Wait_Market_Days": float(open_pos[trigger_date] - open_pos[signal_date]),
                   "No_Trigger_Reason": ""})
    return result


def is_main_board(ts_code: str) -> bool:
    return not str(ts_code).startswith(("300", "301", "688", "689"))


def dynamic_exit(path: pd.DataFrame, entry_price: float, method: str,
                 sell_slippage_pct: float, open_pos: dict[str, int]) -> dict[str, Any]:
    ordered = path.sort_values("trade_date").reset_index(drop=True)
    if ordered.empty:
        return {"Signal_Date": "", "Date": "", "Price": np.nan, "Return_pct": np.nan,
                "Hold_Market_Days": np.nan, "Reason": "无行情"}
    peak_close = entry_price
    for position, row in ordered.iterrows():
        close = finite_num(row["close"])
        peak_close = max(peak_close, close) if math.isfinite(close) else peak_close
        drawdown = (close / peak_close - 1.0) * 100.0 if peak_close > 0 and math.isfinite(close) else np.nan
        death = to_bool(row.get("D_SKDJ_Death_Cross"))
        levels = [finite_num(row.get("D_SKDJ_Level")), finite_num(row.get("D_SKDJ_Prev_Level"))]
        finite_levels = [value for value in levels if math.isfinite(value)]
        high_death = death and bool(finite_levels) and max(finite_levels) >= HIGH_ZONE
        triggered = False
        reason = ""
        if method == "Any_Death" and death:
            triggered, reason = True, "日线SKDJ任意位置死叉"
        elif method == "High_Death" and high_death:
            triggered, reason = True, "日线SKDJ高位死叉"
        elif method == "Trail_10" and drawdown <= -10.0:
            triggered, reason = True, "最高收盘回撤10%"
        elif method == "Trail_15" and drawdown <= -15.0:
            triggered, reason = True, "最高收盘回撤15%"
        elif method == "Hybrid_High_Death_Trail_15" and (high_death or drawdown <= -15.0):
            triggered = True
            reason = "日线SKDJ高位死叉" if high_death else "最高收盘回撤15%"
        if not triggered or position + 1 >= len(ordered):
            continue
        exit_row = ordered.iloc[position + 1]
        price = float(exit_row["open"]) * (1 - sell_slippage_pct / 100)
        exit_date = str(exit_row["trade_date"])
        entry_date = str(ordered.iloc[0]["trade_date"])
        return {"Signal_Date": str(row["trade_date"]), "Date": exit_date, "Price": price,
                "Return_pct": (price / entry_price - 1.0) * 100.0,
                "Hold_Market_Days": float(open_pos[exit_date] - open_pos[entry_date] + 1),
                "Reason": reason}
    last = ordered.iloc[-1]
    price = float(last["close"]) * (1 - sell_slippage_pct / 100)
    exit_date, entry_date = str(last["trade_date"]), str(ordered.iloc[0]["trade_date"])
    return {"Signal_Date": exit_date, "Date": exit_date, "Price": price,
            "Return_pct": (price / entry_price - 1.0) * 100.0,
            "Hold_Market_Days": float(open_pos[exit_date] - open_pos[entry_date] + 1),
            "Reason": "40个市场交易日到期"}


def evaluate_entry(daily_ind: pd.DataFrame, trigger_date: str, ts_code: str,
                   open_dates: list[str], open_pos: dict[str, int], config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "Tradable": False, "Untradable_Reason": "未到买入条件", "Entry_Date": "",
        "Entry_Price": np.nan, "Has_40D_Future": False, "MFE_40D_pct": np.nan,
        "MAE_40D_pct": np.nan, "Hold40_Return_pct": np.nan,
    }
    for method in EXIT_METHODS:
        for suffix, default in (("Exit_Signal_Date", ""), ("Exit_Date", ""),
                                ("Exit_Price", np.nan), ("Exit_Return_pct", np.nan),
                                ("Hold_Market_Days", np.nan), ("Exit_Reason", "")):
            out[f"{method}_{suffix}"] = default
    if not trigger_date or trigger_date not in open_pos:
        return out
    entry_market_pos = open_pos[trigger_date] + 1
    if entry_market_pos >= len(open_dates):
        out["Untradable_Reason"] = "未来交易日不足"
        return out
    entry_date = open_dates[entry_market_pos]
    out["Entry_Date"] = entry_date
    rows = daily_ind[daily_ind["trade_date"].astype(str).eq(entry_date)]
    if rows.empty:
        out["Untradable_Reason"] = "D1停牌或无行情"
        return out
    first = rows.iloc[-1]
    if is_main_board(ts_code) and float(first["open"]) == float(first["high"]) == float(first["low"]):
        out["Untradable_Reason"] = "主板D1一字板"
        return out
    entry_price = float(first["open"]) * (1 + config["buy_slippage_pct"] / 100)
    out.update({"Tradable": True, "Untradable_Reason": "", "Entry_Price": entry_price})
    horizon_pos = entry_market_pos + HOLD_TRADING_DAYS - 1
    if horizon_pos >= len(open_dates):
        out["Untradable_Reason"] = "可买但未来不足40个市场交易日"
        return out
    horizon_date = open_dates[horizon_pos]
    path = daily_ind[daily_ind["trade_date"].astype(str).between(entry_date, horizon_date)].copy()
    if path.empty:
        out["Untradable_Reason"] = "40日窗口无行情"
        return out
    path = path.sort_values("trade_date")
    out.update({
        "Has_40D_Future": True,
        "MFE_40D_pct": (float(path["high"].max()) / entry_price - 1.0) * 100.0,
        "MAE_40D_pct": (float(path["low"].min()) / entry_price - 1.0) * 100.0,
        "Hold40_Return_pct": (float(path.iloc[-1]["close"]) * (1 - config["sell_slippage_pct"] / 100)
                              / entry_price - 1.0) * 100.0,
    })
    for method in EXIT_METHODS:
        result = dynamic_exit(path, entry_price, method, config["sell_slippage_pct"], open_pos)
        out[f"{method}_Exit_Signal_Date"] = result["Signal_Date"]
        out[f"{method}_Exit_Date"] = result["Date"]
        out[f"{method}_Exit_Price"] = result["Price"]
        out[f"{method}_Exit_Return_pct"] = result["Return_pct"]
        out[f"{method}_Hold_Market_Days"] = result["Hold_Market_Days"]
        out[f"{method}_Exit_Reason"] = result["Reason"]
    return out


def build_event(stock: pd.Series, membership: dict[str, str], weekly: pd.DataFrame,
                position: int, daily_ind: pd.DataFrame, daily_basic: pd.DataFrame,
                open_dates: list[str], open_pos: dict[str, int],
                config: dict[str, Any]) -> dict[str, Any] | None:
    row = weekly.iloc[position]
    signal_date = str(row["trade_date"])
    if not (config["signal_start"] <= signal_date <= config["signal_end"]):
        return None
    snapshot = market_snapshot(daily_basic, signal_date)
    passed, reason = signal_filter(snapshot, config["min_price"], config["min_mv"])
    if not passed:
        config["rejects"][reason] = config["rejects"].get(reason, 0) + 1
        return None
    level = (float(row["SKDJ_K"]) + float(row["SKDJ_D"])) / 2.0
    recent = weekly.iloc[max(0, position - RESET_LOOKBACK_WEEKS + 1):position + 1]
    recent_min = float(pd.concat([recent["SKDJ_K"], recent["SKDJ_D"]]).min())
    touched_bottom = recent_min <= SKDJ_BOTTOM
    in_cross_zone = CROSS_ZONE_LOW < level <= CROSS_ZONE_HIGH
    core = touched_bottom and in_cross_zone
    death_date = next_weekly_death_date(weekly, position)
    event: dict[str, Any] = {
        "ts_code": str(stock["ts_code"]), "name": str(stock["name"]),
        "Sample_Board": sample_board(stock), "SW_L1": membership["l1"],
        "SW_L2": membership["l2"], "SW_L3": membership["l3"],
        "Signal_Date": signal_date, "Weekly_Close": float(row["close"]),
        "Weekly_SKDJ_K": float(row["SKDJ_K"]), "Weekly_SKDJ_D": float(row["SKDJ_D"]),
        "Weekly_Cross_Level": level, "Weekly_Cross_Level_Bin": weekly_cross_bin(level),
        "Recent_3W_Min_SKDJ": recent_min, "Recent_3W_Touched_25": touched_bottom,
        "Cross_In_20_35": in_cross_zone, "Bottom_Reset_Core": core,
        "Bottom_Reset_Quadrant": reset_quadrant(touched_bottom, in_cross_zone),
        "Weekly_SKDJ_Death_Cross_Date": death_date,
        **snapshot,
    }
    if not core:
        return event
    event.update(daily_state_at_cross(daily_ind, signal_date))
    for method in ENTRY_METHODS:
        trigger = find_entry_trigger(daily_ind, signal_date, death_date, config["market_end"],
                                     open_dates, open_pos, method)
        path = evaluate_entry(daily_ind, trigger["Trigger_Date"], str(stock["ts_code"]),
                              open_dates, open_pos, config)
        event.update({f"{method}_{key}": value for key, value in trigger.items()})
        event.update({f"{method}_{key}": value for key, value in path.items()})
    for method in ENTRY_METHODS:
        if method == "Direct":
            continue
        if to_bool(event.get("Direct_Has_40D_Future")) and to_bool(event.get(f"{method}_Has_40D_Future")):
            event[f"{method}_Delta_Hold40_vs_Direct_pct"] = (
                finite_num(event[f"{method}_Hold40_Return_pct"])
                - finite_num(event["Direct_Hold40_Return_pct"]))
            event[f"{method}_Delta_MFE_vs_Direct_pct"] = (
                finite_num(event[f"{method}_MFE_40D_pct"])
                - finite_num(event["Direct_MFE_40D_pct"]))
        else:
            event[f"{method}_Delta_Hold40_vs_Direct_pct"] = np.nan
            event[f"{method}_Delta_MFE_vs_Direct_pct"] = np.nan
    return event


def analyze_stock(stock: pd.Series, periods: list[dict[str, str]], daily: pd.DataFrame,
                  daily_basic: pd.DataFrame, week_last_map: dict[pd.Timestamp, str],
                  open_dates: list[str], open_pos: dict[str, int],
                  config: dict[str, Any]) -> list[dict[str, Any]]:
    weekly = build_complete_weekly(daily, week_last_map)
    if len(weekly) < INDICATOR_WARMUP_WEEKS:
        config["rejects"]["周线不足"] = config["rejects"].get("周线不足", 0) + 1
        return []
    daily_ind = add_daily_indicators(daily)
    records: list[dict[str, Any]] = []
    for position in range(INDICATOR_WARMUP_WEEKS, len(weekly)):
        if not to_bool(weekly.iloc[position]["SKDJ_Golden_Cross"]):
            continue
        signal_date = str(weekly.iloc[position]["trade_date"])
        if not (config["signal_start"] <= signal_date <= config["signal_end"]):
            continue
        if not (str(stock["list_date"]) <= signal_date < str(stock["delist_date"])):
            config["rejects"]["当时未上市或已退市"] = config["rejects"].get("当时未上市或已退市", 0) + 1
            continue
        membership = membership_on_date(periods, signal_date)
        if membership is None:
            config["rejects"]["当时不在历史科技池"] = config["rejects"].get("当时不在历史科技池", 0) + 1
            continue
        event = build_event(stock, membership, weekly, position, daily_ind, daily_basic,
                            open_dates, open_pos, config)
        if event is not None:
            records.append(event)
    return records


# -----------------------------------------------------------------------------
# 审计汇总
# -----------------------------------------------------------------------------
def natural_week_calendar(open_dates: list[str], start: str, end: str,
                          events: pd.DataFrame) -> pd.DataFrame:
    days = pd.DataFrame({"trade_date": [day for day in open_dates if start <= day <= end]})
    days["dt"] = pd.to_datetime(days["trade_date"])
    days["week"] = days["dt"].dt.to_period("W-FRI")
    weeks = days.groupby("week")["trade_date"].max().rename("Week_Last_Trade_Date").reset_index(drop=True).to_frame()
    all_counts = events.groupby("Signal_Date").size()
    core = events[events["Bottom_Reset_Core"].map(to_bool)]
    core_counts = core.groupby("Signal_Date").size()
    weeks["All_Weekly_SKDJ_Cross_Count"] = weeks["Week_Last_Trade_Date"].map(all_counts).fillna(0).astype(int)
    weeks["Core_Count"] = weeks["Week_Last_Trade_Date"].map(core_counts).fillna(0).astype(int)
    weeks["All_Empty"] = weeks["All_Weekly_SKDJ_Cross_Count"].eq(0)
    weeks["Core_Empty"] = weeks["Core_Count"].eq(0)
    return weeks


def entry_stat(frame: pd.DataFrame, method: str, label: str) -> dict[str, Any]:
    triggered = frame[frame[f"{method}_Triggered"].map(to_bool)] if len(frame) else frame
    tradable = frame[frame[f"{method}_Tradable"].map(to_bool)] if len(frame) else frame
    mature = frame[frame[f"{method}_Has_40D_Future"].map(to_bool)] if len(frame) else frame
    returns = pd.to_numeric(mature.get(f"{method}_Hold40_Return_pct"), errors="coerce")
    return {
        "分组": label, "买入方法代码": method, "买入方法": ENTRY_METHODS[method],
        "候选数": len(frame), "触发数": len(triggered), "触发率(%)": len(triggered) / len(frame) * 100 if len(frame) else np.nan,
        "可买数": len(tradable), "40日成熟数": len(mature),
        "平均等待市场日": pd.to_numeric(triggered.get(f"{method}_Wait_Market_Days"), errors="coerce").mean() if len(triggered) else np.nan,
        "40日平均收益(%)": returns.mean(), "40日收益中位数(%)": returns.median(),
        "40日正收益比例(%)": returns.gt(0).mean() * 100 if len(returns) else np.nan,
        "平均MFE(%)": pd.to_numeric(mature.get(f"{method}_MFE_40D_pct"), errors="coerce").mean() if len(mature) else np.nan,
        "平均MAE(%)": pd.to_numeric(mature.get(f"{method}_MAE_40D_pct"), errors="coerce").mean() if len(mature) else np.nan,
    }


def build_entry_summaries(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame([entry_stat(core, method, "核心池") for method in ENTRY_METHODS])
    yearly_rows: list[dict[str, Any]] = []
    years = sorted(core["Signal_Date"].astype(str).str[:4].unique())
    for year in years:
        year_frame = core[core["Signal_Date"].astype(str).str[:4].eq(year)]
        for method in ENTRY_METHODS:
            row = entry_stat(year_frame, method, f"{year}_{ENTRY_METHODS[method]}")
            row["年份"] = year
            yearly_rows.append(row)
    paired_rows: list[dict[str, Any]] = []
    for method in ENTRY_METHODS:
        if method == "Direct":
            continue
        paired = core[core["Direct_Has_40D_Future"].map(to_bool)
                      & core[f"{method}_Has_40D_Future"].map(to_bool)].copy()
        delta = pd.to_numeric(paired.get(f"{method}_Delta_Hold40_vs_Direct_pct"), errors="coerce")
        mfe_delta = pd.to_numeric(paired.get(f"{method}_Delta_MFE_vs_Direct_pct"), errors="coerce")
        paired_rows.append({
            "买入方法": ENTRY_METHODS[method], "成对数": len(paired),
            "相对直接买平均40日收益差(百分点)": delta.mean(),
            "收益优于直接买比例(%)": delta.gt(0).mean() * 100 if len(delta) else np.nan,
            "收益差中位数(百分点)": delta.median(),
            "平均MFE差(百分点)": mfe_delta.mean(),
        })
    return overall, pd.DataFrame(yearly_rows), pd.DataFrame(paired_rows)


def build_state_summary(core: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions = {
        "周线金叉时日线SKDJ关系": "Daily_SKDJ_State_At_Cross",
        "周线金叉时日线SKDJ位置": "Daily_SKDJ_Level_Bin",
        "两维组合": None,
    }
    for dimension, column in dimensions.items():
        if column is None:
            groups = core.groupby(["Daily_SKDJ_State_At_Cross", "Daily_SKDJ_Level_Bin"], dropna=False)
        else:
            groups = core.groupby(column, dropna=False)
        for group, frame in groups:
            group_label = "|".join(map(str, group)) if isinstance(group, tuple) else str(group)
            row = entry_stat(frame, "Direct", group_label)
            row.update({"维度": dimension, "状态分组": group_label})
            rows.append(row)
    return pd.DataFrame(rows)


def exit_stat(frame: pd.DataFrame, entry_method: str, exit_method: str,
              label: str) -> dict[str, Any]:
    mature = frame[frame[f"{entry_method}_Has_40D_Future"].map(to_bool)] if len(frame) else frame
    returns = pd.to_numeric(mature.get(f"{entry_method}_{exit_method}_Exit_Return_pct"), errors="coerce")
    hold = pd.to_numeric(mature.get(f"{entry_method}_{exit_method}_Hold_Market_Days"), errors="coerce")
    return {
        "分组": label, "买入方法代码": entry_method, "买入方法": ENTRY_METHODS[entry_method],
        "卖出方法代码": exit_method, "卖出方法": EXIT_METHODS[exit_method],
        "成熟交易数": len(mature), "平均收益(%)": returns.mean(),
        "收益中位数(%)": returns.median(),
        "正收益比例(%)": returns.gt(0).mean() * 100 if len(returns) else np.nan,
        "最差收益(%)": returns.min(), "最好收益(%)": returns.max(),
        "平均持有市场日": hold.mean(), "持有日中位数": hold.median(),
    }


def build_exit_summaries(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame([
        exit_stat(core, entry_method, exit_method, "核心池")
        for entry_method in ENTRY_METHODS for exit_method in EXIT_METHODS
    ])
    yearly_rows: list[dict[str, Any]] = []
    for year in sorted(core["Signal_Date"].astype(str).str[:4].unique()):
        frame = core[core["Signal_Date"].astype(str).str[:4].eq(year)]
        for entry_method in ENTRY_METHODS:
            for exit_method in EXIT_METHODS:
                row = exit_stat(frame, entry_method, exit_method, str(year))
                row["年份"] = year
                yearly_rows.append(row)
    reason_rows: list[dict[str, Any]] = []
    for entry_method in ENTRY_METHODS:
        mature = core[core[f"{entry_method}_Has_40D_Future"].map(to_bool)]
        for exit_method in EXIT_METHODS:
            counts = mature[f"{entry_method}_{exit_method}_Exit_Reason"].fillna("缺失").value_counts()
            for reason, count in counts.items():
                reason_rows.append({"买入方法": ENTRY_METHODS[entry_method],
                                    "卖出方法": EXIT_METHODS[exit_method],
                                    "退出原因": reason, "数量": int(count),
                                    "占比(%)": count / len(mature) * 100 if len(mature) else np.nan})
    return overall, pd.DataFrame(yearly_rows), pd.DataFrame(reason_rows)


def build_entry_competition(core: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in ENTRY_METHODS:
        frame = core[core[f"{method}_Tradable"].map(to_bool)
                     & core[f"{method}_Entry_Date"].astype(str).ne("")]
        counts = frame.groupby(f"{method}_Entry_Date").agg(
            Signal_Count=("ts_code", "size"), Unique_Stocks=("ts_code", "nunique"),
            Signal_Weeks=("Signal_Date", "nunique")
        ).reset_index().rename(columns={f"{method}_Entry_Date": "Entry_Date"})
        for row in counts.itertuples(index=False):
            rows.append({"买入方法代码": method, "买入方法": ENTRY_METHODS[method],
                         "Entry_Date": row.Entry_Date, "Signal_Count": int(row.Signal_Count),
                         "Unique_Stocks": int(row.Unique_Stocks), "Signal_Weeks": int(row.Signal_Weeks),
                         "More_Than_3": int(row.Signal_Count) > 3})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 三仓组合与同日随机顺序
# -----------------------------------------------------------------------------
def fee(amount: float, rate_pct: float, minimum: float = 0.0) -> float:
    return max(minimum, amount * rate_pct / 100) if amount > 0 else 0.0


def build_mark_prices(histories: dict[str, pd.DataFrame], open_dates: list[str]) -> dict[str, dict[str, float]]:
    """预先将每只股票的最新收盘价映射到市场交易日，避免蒙特卡洛中重复扫描DataFrame。"""
    result: dict[str, dict[str, float]] = {}
    calendar = pd.Index(open_dates, dtype=str)
    for code, history in histories.items():
        if history.empty:
            result[code] = {}
            continue
        clean = history.drop_duplicates("trade_date", keep="last").copy()
        clean["trade_date"] = clean["trade_date"].astype(str)
        series = pd.to_numeric(clean.set_index("trade_date")["close"], errors="coerce").sort_index()
        result[code] = series.reindex(calendar).ffill().to_dict()
    return result


def mark_price(mark_prices: dict[str, dict[str, float]], code: str, trade_date: str) -> float:
    return finite_num(mark_prices.get(code, {}).get(trade_date))


def empty_portfolio_summary(seed: int, entry_method: str, exit_method: str) -> dict[str, Any]:
    return {
        "同日随机种子": seed, "买入方法代码": entry_method,
        "买入方法": ENTRY_METHODS[entry_method], "卖出方法代码": exit_method,
        "卖出方法": EXIT_METHODS[exit_method], "初始资金": INITIAL_CAPITAL,
        "实际买入": 0, "仓位满错过": 0, "期末权益": INITIAL_CAPITAL,
        "总收益率(%)": 0.0, "最大回撤(%)": 0.0, "交易胜率(%)": np.nan,
    }


def simulate_portfolio(core: pd.DataFrame, mark_prices: dict[str, dict[str, float]],
                       open_dates: list[str], config: dict[str, Any], seed: int,
                       entry_method: str, exit_method: str,
                       build_detail: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    entry_date_col = f"{entry_method}_Entry_Date"
    exit_date_col = f"{entry_method}_{exit_method}_Exit_Date"
    work = core[
        core[f"{entry_method}_Tradable"].map(to_bool)
        & core[f"{entry_method}_Has_40D_Future"].map(to_bool)
        & core[entry_date_col].astype(str).ne("")
        & core[exit_date_col].astype(str).ne("")
    ].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), empty_portfolio_summary(seed, entry_method, exit_method)
    rng = np.random.default_rng(seed)
    work["_tie"] = rng.random(len(work))
    work = work.sort_values([entry_date_col, "_tie", "ts_code"], kind="mergesort")
    entry_groups = {str(day): frame for day, frame in work.groupby(entry_date_col, sort=True)}
    first_day, last_day = str(work[entry_date_col].min()), str(work[exit_date_col].max())
    days = [day for day in open_dates if first_day <= day <= last_day]
    cash = INITIAL_CAPITAL
    active: dict[str, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    equity_values: list[float] = []
    for trade_date in days:
        # 同一开盘先执行已计划卖出，释放仓位和现金后再处理新买入。
        exiting = [code for code, trade in active.items() if trade["Planned_Exit_Date"] == trade_date]
        for code in exiting:
            trade = active.pop(code)
            gross = trade["Shares"] * trade["Exit_Price"]
            sell_fee = (fee(gross, config["commission_pct"], 5.0)
                        + fee(gross, config["transfer_fee_pct"])
                        + fee(gross, config["stamp_duty_pct"]))
            proceeds = gross - sell_fee
            cash += proceeds
            pnl = proceeds - trade["Entry_Amount"] - trade["Buy_Fees"]
            trade.update({"Sell_Fees": sell_fee, "Exit_Proceeds": proceeds, "PnL": pnl,
                          "Net_Return_pct": pnl / (trade["Entry_Amount"] + trade["Buy_Fees"]) * 100})
        for _, row in entry_groups.get(trade_date, pd.DataFrame()).iterrows():
            code = str(row["ts_code"])
            reason = "同一股票已持仓" if code in active else "3个仓位已满" if len(active) >= MAX_POSITIONS else ""
            price = finite_num(row[f"{entry_method}_Entry_Price"])
            budget = min(POSITION_BUDGET, cash)
            shares = int(math.floor(budget / price / LOT_SIZE) * LOT_SIZE) if price > 0 else 0
            while shares >= LOT_SIZE:
                amount = shares * price
                buy_fee = fee(amount, config["commission_pct"], 5.0) + fee(amount, config["transfer_fee_pct"])
                if amount + buy_fee <= budget + 1e-8:
                    break
                shares -= LOT_SIZE
            if not reason and shares < LOT_SIZE:
                reason = "现金不足一手"
            if build_detail:
                orders.append({
                    "Entry_Date": trade_date, "Signal_Date": row["Signal_Date"], "ts_code": code,
                    "name": row.get("name", ""), "买入方法": ENTRY_METHODS[entry_method],
                    "卖出方法": EXIT_METHODS[exit_method],
                    "Action": "未买入" if reason else "已买入",
                    "Reason": reason or f"随机同日顺序买入{shares}股",
                    "Prospective_Exit_Date": row[exit_date_col], "Tie_Seed": seed,
                })
            if reason:
                continue
            amount = shares * price
            buy_fee = fee(amount, config["commission_pct"], 5.0) + fee(amount, config["transfer_fee_pct"])
            cash -= amount + buy_fee
            trade = {
                "Signal_Date": row["Signal_Date"], "Entry_Date": trade_date, "ts_code": code,
                "name": row.get("name", ""), "买入方法": ENTRY_METHODS[entry_method],
                "卖出方法": EXIT_METHODS[exit_method], "Shares": shares,
                "Entry_Price": price, "Entry_Amount": amount, "Buy_Fees": buy_fee,
                "Planned_Exit_Date": str(row[exit_date_col]),
                "Exit_Price": finite_num(row[f"{entry_method}_{exit_method}_Exit_Price"]),
                "Exit_Reason": row[f"{entry_method}_{exit_method}_Exit_Reason"],
                "PnL": np.nan, "Net_Return_pct": np.nan,
            }
            active[code] = trade
            ledger.append(trade)
        market_value = 0.0
        for code, trade in active.items():
            mark = mark_price(mark_prices, code, trade_date)
            market_value += trade["Shares"] * (mark if math.isfinite(mark) else trade["Entry_Price"])
        equity = cash + market_value
        equity_values.append(equity)
        if build_detail:
            curve_rows.append({"Trade_Date": trade_date, "Cash": cash, "Market_Value": market_value,
                               "Equity": equity, "Positions": len(active),
                               "买入方法": ENTRY_METHODS[entry_method],
                               "卖出方法": EXIT_METHODS[exit_method]})
    ledger_frame, orders_frame = pd.DataFrame(ledger), pd.DataFrame(orders)
    curve_frame = pd.DataFrame(curve_rows)
    equity_series = pd.Series(equity_values, dtype=float)
    if len(equity_series):
        running_peak = equity_series.cummax().clip(lower=INITIAL_CAPITAL)
        drawdown = (equity_series / running_peak - 1.0) * 100.0
        max_dd = finite_num(drawdown.min())
        if build_detail:
            curve_frame["Drawdown_pct"] = drawdown.to_numpy()
    else:
        max_dd = np.nan
    final_equity = cash + sum(
        trade["Shares"] * (mark_price(mark_prices, code, last_day)
                            if math.isfinite(mark_price(mark_prices, code, last_day)) else trade["Entry_Price"])
        for code, trade in active.items()
    )
    summary = {
        "同日随机种子": seed, "买入方法代码": entry_method,
        "买入方法": ENTRY_METHODS[entry_method], "卖出方法代码": exit_method,
        "卖出方法": EXIT_METHODS[exit_method], "初始资金": INITIAL_CAPITAL,
        "实际买入": len(ledger_frame),
        "仓位满错过": int(orders_frame.get("Reason", pd.Series(dtype=str)).eq("3个仓位已满").sum()) if build_detail else np.nan,
        "期末权益": final_equity, "总收益率(%)": (final_equity / INITIAL_CAPITAL - 1.0) * 100.0,
        "最大回撤(%)": max_dd,
        "交易胜率(%)": float(ledger_frame["PnL"].gt(0).mean() * 100.0) if len(ledger_frame) else np.nan,
    }
    return curve_frame, ledger_frame, orders_frame, summary


def run_primary(core: pd.DataFrame, mark_prices: dict[str, dict[str, float]], open_dates: list[str],
                config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    orders: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    for entry_method in ENTRY_METHODS:
        for exit_method in EXIT_METHODS:
            curve, ledger, order, summary = simulate_portfolio(
                core, mark_prices, open_dates, config, PRIMARY_SEED,
                entry_method, exit_method, build_detail=True)
            summaries.append(summary)
            if len(ledger):
                ledgers.append(ledger)
            if len(order):
                orders.append(order)
            if len(curve):
                curves.append(curve)
    return (pd.DataFrame(summaries),
            pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(),
            pd.concat(orders, ignore_index=True) if orders else pd.DataFrame(),
            pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


def run_monte_carlo(core: pd.DataFrame, mark_prices: dict[str, dict[str, float]],
                    open_dates: list[str], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for entry_method in ENTRY_METHODS:
        for exit_method in EXIT_METHODS:
            for seed in range(MC_RUNS):
                _, _, _, summary = simulate_portfolio(
                    core, mark_prices, open_dates, config, seed,
                    entry_method, exit_method, build_detail=False)
                rows.append(summary)
    distribution = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (entry_method, exit_method), frame in distribution.groupby(
            ["买入方法代码", "卖出方法代码"], sort=False):
        returns = pd.to_numeric(frame["总收益率(%)"], errors="coerce")
        drawdowns = pd.to_numeric(frame["最大回撤(%)"], errors="coerce")
        summary_rows.append({
            "买入方法代码": entry_method, "买入方法": ENTRY_METHODS[entry_method],
            "卖出方法代码": exit_method, "卖出方法": EXIT_METHODS[exit_method],
            "随机次数": len(frame), "平均总收益率(%)": returns.mean(),
            "中位总收益率(%)": returns.median(), "标准差(百分点)": returns.std(),
            "5%分位收益(%)": returns.quantile(0.05), "95%分位收益(%)": returns.quantile(0.95),
            "最差总收益率(%)": returns.min(), "最好总收益率(%)": returns.max(),
            "盈利模拟比例(%)": returns.gt(0).mean() * 100.0,
            "平均最大回撤(%)": drawdowns.mean(), "最差最大回撤(%)": drawdowns.min(),
            "平均买入笔数": pd.to_numeric(frame["实际买入"], errors="coerce").mean(),
        })
    return pd.DataFrame(summary_rows), distribution


# -----------------------------------------------------------------------------
# Streamlit主程序
# -----------------------------------------------------------------------------
def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title="周线SKDJ池＋日线买卖点 V3.3", layout="wide")
    st.title(TITLE)
    st.caption("只研究买卖时机：周线SKDJ底部重置核心池保持不变，不评分、不根据本次收益调参。")
    with st.expander("冻结规则与本版对照", expanded=True):
        st.markdown(f"""
- **周线核心池**：完整周线SKDJ金叉；最近{RESET_LOOKBACK_WEEKS}个完整周K或D曾≤{SKDJ_BOTTOM:.0f}；金叉位置>{CROSS_ZONE_LOW:.0f}且≤{CROSS_ZONE_HIGH:.0f}。
- **SKDJ参数**：周线与日线均冻结 `N={SKDJ_N}, M={SKDJ_M}`。
- **四种买入**：周线金叉次日直买；最多等{DAILY_SKDJ_WAIT_DAYS}日的日线SKDJ金叉；最多等{STATE_FLOW_WAIT_DAYS}日的状态分流；最多等{DAILY_KDJ_WAIT_DAYS}日的普通KDJ金叉对照。
- **状态分流**：周线信号日K>D且K上升则直接触发；K≤D则等日线SKDJ金叉；K>D但K不升则等K重新上升。
- **五种卖出**：任意日线SKDJ死叉；{HIGH_ZONE:.0f}以上高位死叉；最高收盘回撤10%；回撤15%；高位死叉或回撤15%。
- **可执行口径**：所有信号收盘确认，次一交易日开盘成交；回撤按最高收盘计算；最长{HOLD_TRADING_DAYS}个市场交易日。
- **组合口径**：30万元、3个等额仓位；20种买卖组合各做{MC_RUNS}次同日随机顺序。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("信号开始", date(2023, 6, 5), key="v33_start")
        signal_end_date = st.date_input("信号截止", date(2026, 6, 5), key="v33_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v33_market_end")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v33_pause")
        use_cache = st.checkbox("复用逐股票缓存", True, key="v33_cache")
        st.divider()
        commission_pct = st.number_input("佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f")
        stamp_duty_pct = st.number_input("卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f")
        transfer_fee_pct = st.number_input("过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f")
        if st.button("清除本程序行情缓存", key="v33_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("缓存已清除")
    token = st.text_input("Tushare Token", type="password", key="v33_token")
    session_key = "weekly_skdj_entry_exit_v33_zip"
    if not token:
        st.info("请输入Tushare Token；V3.2相同日期的逐股票行情缓存可直接复用。")
        return
    if not st.button("开始V3.3买卖点验证", type="primary", key="v33_run"):
        if session_key in st.session_state:
            st.download_button("下载上一次结果ZIP", st.session_state[session_key],
                               file_name="weekly_skdj_daily_entry_exit_audit_v3_3_all_results.zip",
                               mime="application/zip", on_click="ignore")
        return
    error = validate_dates(signal_start_date, signal_end_date, market_end_date)
    if error:
        st.error(error)
        return
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
        "min_price": 10.0, "min_mv": 100.0, "buy_slippage_pct": 0.20,
        "sell_slippage_pct": 0.20, "commission_pct": float(commission_pct),
        "stamp_duty_pct": float(stamp_duty_pct), "transfer_fee_pct": float(transfer_fee_pct),
        "rejects": rejects,
    }
    try:
        with st.spinner("加载交易日历和历史科技股池..."):
            open_dates = load_trade_calendar(preload, market_end)
            full_open_dates = load_trade_calendar(
                preload, (market_end_date + timedelta(days=7)).strftime("%Y%m%d"))
            week_last_map = complete_week_last_dates(full_open_dates)
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
    events: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}
    cache_hits = data_failures = 0
    progress, status = st.progress(0.0), st.empty()
    for number, stock in stocks.iterrows():
        code = str(stock["ts_code"])
        progress.progress((number + 1) / max(len(stocks), 1), text=f"{number + 1}/{len(stocks)} {code}")
        current_core = sum(to_bool(item.get("Bottom_Reset_Core")) for item in events)
        status.caption(f"周线金叉 {len(events)}；核心池 {current_core}；缓存 {cache_hits}；失败 {data_failures}")
        daily, daily_basic, cache_hit = fetch_stock_history(
            code, preload, market_end, bool(use_cache), float(pause))
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        records = analyze_stock(stock, period_index.get(code, []), daily, daily_basic,
                                week_last_map, open_dates, open_pos, config)
        if records:
            events.extend(records)
            histories[code] = daily.copy()
    progress.empty()
    status.empty()
    if not events:
        st.error("研究区间没有生成符合历史科技池、价格和市值条件的周线SKDJ金叉。")
        return
    try:
        with st.spinner("生成日线SKDJ状态、四买五卖、三仓与蒙特卡洛审计..."):
            event_frame = pd.DataFrame(events).sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True)
            core = event_frame[event_frame["Bottom_Reset_Core"].map(to_bool)].copy()
            if core.empty:
                st.error("本次没有生成底部重置核心池事件。")
                return
            pool_calendar = natural_week_calendar(open_dates, signal_start, signal_end, event_frame)
            state_summary = build_state_summary(core)
            entry_summary, entry_yearly, entry_paired = build_entry_summaries(core)
            exit_summary, exit_yearly, exit_reasons = build_exit_summaries(core)
            competition = build_entry_competition(core)
            mark_prices = build_mark_prices(histories, open_dates)
            primary_summary, primary_ledger, primary_orders, primary_curves = run_primary(
                core, mark_prices, open_dates, config)
            mc_summary, mc_distribution = run_monte_carlo(core, mark_prices, open_dates, config)
    except Exception as exc:
        st.exception(exc)
        return
    competition_summary = []
    for method in ENTRY_METHODS:
        frame = competition[competition["买入方法代码"].eq(method)]
        competition_summary.append({
            "买入方法代码": method, "买入方法": ENTRY_METHODS[method],
            "入场日数": len(frame), "超过3只的入场日": int(frame["More_Than_3"].sum()) if len(frame) else 0,
            "单日最多信号": int(frame["Signal_Count"].max()) if len(frame) else 0,
            "平均每入场日信号": pd.to_numeric(frame.get("Signal_Count"), errors="coerce").mean() if len(frame) else np.nan,
        })
    competition_summary_frame = pd.DataFrame(competition_summary)
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "信号开始": signal_start, "信号截止": signal_end,
        "观察截止": market_end, "全部周线SKDJ金叉": len(event_frame),
        "底部重置核心池": len(core), "不同股票": core["ts_code"].nunique(),
        "自然周": len(pool_calendar), "核心平均每周": len(core) / len(pool_calendar) if len(pool_calendar) else np.nan,
        "核心空窗周": int(pool_calendar["Core_Empty"].sum()),
        "买入方法数": len(ENTRY_METHODS), "卖出方法数": len(EXIT_METHODS),
        "买卖组合数": len(ENTRY_METHODS) * len(EXIT_METHODS), "每组蒙特卡洛次数": MC_RUNS,
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    basic_columns = [
        "ts_code", "name", "Sample_Board", "SW_L1", "SW_L2", "SW_L3", "Signal_Date",
        "Weekly_Close", "Weekly_SKDJ_K", "Weekly_SKDJ_D", "Weekly_Cross_Level",
        "Weekly_Cross_Level_Bin", "Recent_3W_Min_SKDJ", "Recent_3W_Touched_25",
        "Cross_In_20_35", "Bottom_Reset_Core", "Bottom_Reset_Quadrant",
        "Weekly_SKDJ_Death_Cross_Date", "Raw_Close", "Circ_MV_Billion", "Turnover_Rate",
    ]
    all_event_basic = event_frame[[column for column in basic_columns if column in event_frame.columns]].copy()
    metadata = pd.DataFrame([
        ("候选池", "完整周线SKDJ底部重置核心池；日线指标不改变候选资格"),
        ("SKDJ公式", "LOWV=LLV(LOW,N); HIGHV=HHV(HIGH,N); RSV=EMA((C-LOWV)/(HIGHV-LOWV)*100,M); K=EMA(RSV,M); D=MA(K,M)"),
        ("SKDJ冻结参数", f"N={SKDJ_N},M={SKDJ_M}；周线和日线同参数"),
        ("普通KDJ对照", "9,3,3递归平滑；J=3K-2D；只作对照，不预设更优"),
        ("价格市值", "信号日原始收盘价≥10元；历史流通市值≥100亿元"),
        ("延迟买入失效", "等待期间先出现完整周线SKDJ死叉则不再买入"),
        ("买入执行", "收盘触发，下一市场交易日开盘＋0.20%滑点；D1停牌或主板一字板不买"),
        ("卖出执行", "收盘触发，下一个股可交易日开盘-0.20%滑点；到期按窗口末日收盘"),
        ("回撤口径", "持仓以来最高收盘价，不使用当日最高价，避免日内先后顺序假设"),
        ("高位死叉", f"死叉当日或前一日(K+D)/2曾≥{HIGH_ZONE:.0f}"),
        ("最长持有", f"{HOLD_TRADING_DAYS}个市场交易日"),
        ("三仓口径", "30万元，3个10万元等额仓位；同日无评分时随机顺序"),
        ("防未来函数", "周线只使用完整周；日线信号只在收盘后确认；交易统一在下一交易日"),
    ], columns=["项目", "值"])
    files = {
        "01_run_summary_v3_3.csv": run_summary,
        "02_daily_skdj_state_at_weekly_cross_v3_3.csv": state_summary,
        "03_entry_method_comparison_v3_3.csv": entry_summary,
        "04_entry_method_yearly_stability_v3_3.csv": entry_yearly,
        "05_entry_paired_vs_direct_v3_3.csv": entry_paired,
        "06_exit_method_comparison_v3_3.csv": exit_summary,
        "07_exit_method_yearly_stability_v3_3.csv": exit_yearly,
        "08_exit_reason_distribution_v3_3.csv": exit_reasons,
        "09_entry_competition_detail_v3_3.csv": competition,
        "10_entry_competition_summary_v3_3.csv": competition_summary_frame,
        "11_primary_3slot_20_strategy_summary_v3_3.csv": primary_summary,
        "12_primary_3slot_all_ledgers_v3_3.csv": primary_ledger,
        "13_primary_3slot_all_orders_v3_3.csv": primary_orders,
        "14_primary_3slot_all_equity_curves_v3_3.csv": primary_curves,
        "15_random_tie_mc_summary_v3_3.csv": mc_summary,
        "16_random_tie_mc_distribution_v3_3.csv": mc_distribution,
        "17_core_event_detail_v3_3.csv": core,
        "18_all_weekly_skdj_cross_basic_v3_3.csv": all_event_basic,
        "19_weekly_pool_calendar_v3_3.csv": pool_calendar,
        "20_full_tech_universe_v3_3.csv": stocks,
        "21_board_population_v3_3.csv": population,
        "22_rejection_audit_v3_3.csv": pd.DataFrame(
            [{"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]),
        "23_api_errors_v3_3.csv": pd.DataFrame({"错误": API_ERRORS}),
        "24_metadata_v3_3.csv": metadata,
    }
    result_zip = make_zip(files)
    st.session_state[session_key] = result_zip
    st.success(f"完成：全部周线SKDJ金叉{len(event_frame)}个；底部重置核心池{len(core)}个。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("全部周线金叉", len(event_frame))
    c2.metric("核心池事件", len(core))
    c3.metric("核心平均每周", f"{len(core) / len(pool_calendar):.2f}" if len(pool_calendar) else "-")
    c4.metric("核心空窗周", int(pool_calendar["Core_Empty"].sum()))
    st.subheader("周线金叉时的日线SKDJ状态")
    st.dataframe(state_summary, use_container_width=True, hide_index=True)
    st.subheader("买入方法对照（固定40日终点，隔离卖点影响）")
    st.dataframe(entry_summary, use_container_width=True, hide_index=True)
    st.subheader("五种卖出规则对照")
    st.dataframe(exit_summary, use_container_width=True, hide_index=True)
    st.subheader("三仓同日随机顺序敏感性")
    st.dataframe(mc_summary, use_container_width=True, hide_index=True)
    st.download_button("下载V3.3全部结果ZIP", result_zip,
                       file_name="weekly_skdj_daily_entry_exit_audit_v3_3_all_results.zip",
                       mime="application/zip", type="primary", key="v33_download", on_click="ignore")
    st.info("先看02确认周线金叉时日线SKDJ的真实状态；再看03/05判断等待是否滞后；卖点先看06/07，最后以15的蒙特卡洛结果判断三仓是否可执行。")


if __name__ == "__main__":
    main()
