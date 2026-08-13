# -*- coding: utf-8 -*-
"""
科技股完整周线SKDJ金叉股票池＋日线MACD进场审计器 V3.1

研究问题：
1. 完整周线SKDJ金叉能否比完整周线MACD首红更早识别上涨启动？
2. 金叉位置≤25与>25，哪一组的数量、收益、回撤和稳定性更好？
3. 周线SKDJ建池后，由日线MACD第一根红柱确认进场，信号是否仍然拥挤？

本文件是独立精简版，只保留本次研究必需的行情、事件、成交和审计代码。
周线MACD仅作为“迟到程度”的研究基准，不参与候选生成。

运行：streamlit run weekly_skdj_pool_daily_macd_audit_v3_1_single.py
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


TITLE = "科技股完整周线SKDJ金叉池＋日线MACD进场审计器 V3.1"
VERSION = "V3.1-WEEKLY-SKDJ-POOL-DAILY-MACD-AUDIT"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# 复用同目录逐股票行情缓存，日期相同时无需重新下载。
CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")

SKDJ_N = 9
SKDJ_M = 3
SKDJ_BOTTOM = 25.0
DAILY_CONFIRM_MAX_WAIT = 20
HOLD_TRADING_DAYS = 40
MACD_WARMUP_WEEKS = 40

INITIAL_CAPITAL = 300_000.0
MAX_POSITIONS = 3
POSITION_BUDGET = 100_000.0
LOT_SIZE = 100
MC_RUNS = 200
PRIMARY_SEED = 20260813

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


def add_weekly_macd(weekly: pd.DataFrame) -> pd.DataFrame:
    work = weekly.copy()
    ema12 = work["close"].ewm(span=12, adjust=False).mean()
    ema26 = work["close"].ewm(span=26, adjust=False).mean()
    work["W_DIFF"] = ema12 - ema26
    work["W_DEA"] = work["W_DIFF"].ewm(span=9, adjust=False).mean()
    work["W_MACD_Hist"] = (work["W_DIFF"] - work["W_DEA"]) * 2.0
    work["W_MACD_First_Red"] = work["W_MACD_Hist"].gt(0) & work["W_MACD_Hist"].shift(1).le(0)
    return work


def build_complete_weekly(daily: pd.DataFrame, week_last_map: dict[pd.Timestamp, str]) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    weekly = aggregate_weekly(daily)
    weekly["calendar_week_last"] = weekly["week_label"].map(week_last_map)
    weekly = weekly[
        weekly["calendar_week_last"].notna()
        & weekly["trade_date"].astype(str).eq(weekly["calendar_week_last"].astype(str))
    ].copy().reset_index(drop=True)
    return add_skdj(add_weekly_macd(weekly)) if not weekly.empty else weekly


def build_provisional_weekly(daily: pd.DataFrame, n: int, m: int) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    return add_skdj(aggregate_weekly(daily), n, m)


def add_daily_macd(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy().sort_values("trade_date").reset_index(drop=True)
    ema12 = work["close"].ewm(span=12, adjust=False).mean()
    ema26 = work["close"].ewm(span=26, adjust=False).mean()
    work["D_DIFF"] = ema12 - ema26
    work["D_DEA"] = work["D_DIFF"].ewm(span=9, adjust=False).mean()
    work["D_MACD_Hist"] = (work["D_DIFF"] - work["D_DEA"]) * 2.0
    work["D_MACD_First_Red"] = work["D_MACD_Hist"].gt(0) & work["D_MACD_Hist"].shift(1).le(0)
    return work


# -----------------------------------------------------------------------------
# 事件、迟到审计与进场
# -----------------------------------------------------------------------------
def market_snapshot(basic: pd.DataFrame, signal_date: str) -> dict[str, float]:
    row = basic[basic["trade_date"].astype(str).eq(signal_date)] if not basic.empty else pd.DataFrame()
    if row.empty:
        return {"Raw_Close": np.nan, "Circ_MV_Billion": np.nan, "Turnover_Rate": np.nan}
    row = row.iloc[-1]
    return {
        "Raw_Close": finite_num(row.get("close")),
        "Circ_MV_Billion": finite_num(row.get("circ_mv")) / 10000.0,
        "Turnover_Rate": finite_num(row.get("turnover_rate")),
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


def cross_level_bin(value: float) -> str:
    if value <= 20:
        return "≤20"
    if value <= 25:
        return "20-25"
    if value <= 35:
        return "25-35"
    if value <= 50:
        return "35-50"
    return ">50"


def lag_bin(value: Any) -> str:
    lag = finite_num(value)
    if not math.isfinite(lag):
        return "观察期内未首红"
    if lag == 0:
        return "0周_MACD已红"
    if lag <= 2:
        return "1-2周"
    if lag <= 4:
        return "3-4周"
    if lag <= 8:
        return "5-8周"
    return ">8周"


def weekly_macd_lag(weekly: pd.DataFrame, position: int) -> dict[str, Any]:
    row = weekly.iloc[position]
    hist = finite_num(row["W_MACD_Hist"])
    if hist > 0:
        status = "首红" if bool(row["W_MACD_First_Red"]) else "红柱延续"
        return {
            "Weekly_MACD_State_At_Cross": status, "Weekly_MACD_First_Red_Date": str(row["trade_date"]),
            "Lead_Weeks_To_Weekly_MACD": 0.0, "Price_Gain_To_Weekly_MACD_pct": 0.0,
            "Max_Runup_Before_Weekly_MACD_pct": 0.0, "Max_Drawdown_Before_Weekly_MACD_pct": 0.0,
            "Weekly_MACD_Confirm_Censored": False,
        }
    future = weekly.iloc[position + 1:].copy()
    hits = future[future["W_MACD_First_Red"].map(to_bool)]
    if hits.empty:
        return {
            "Weekly_MACD_State_At_Cross": "绿柱或零轴", "Weekly_MACD_First_Red_Date": "",
            "Lead_Weeks_To_Weekly_MACD": np.nan, "Price_Gain_To_Weekly_MACD_pct": np.nan,
            "Max_Runup_Before_Weekly_MACD_pct": np.nan, "Max_Drawdown_Before_Weekly_MACD_pct": np.nan,
            "Weekly_MACD_Confirm_Censored": True,
        }
    confirm_pos = int(hits.index[0])
    confirm = weekly.loc[confirm_pos]
    path = weekly.loc[position:confirm_pos]
    base = float(row["close"])
    return {
        "Weekly_MACD_State_At_Cross": "绿柱或零轴",
        "Weekly_MACD_First_Red_Date": str(confirm["trade_date"]),
        "Lead_Weeks_To_Weekly_MACD": float(confirm_pos - position),
        "Price_Gain_To_Weekly_MACD_pct": (float(confirm["close"]) / base - 1.0) * 100.0,
        "Max_Runup_Before_Weekly_MACD_pct": (float(path["high"].max()) / base - 1.0) * 100.0,
        "Max_Drawdown_Before_Weekly_MACD_pct": (float(path["low"].min()) / base - 1.0) * 100.0,
        "Weekly_MACD_Confirm_Censored": False,
    }


def next_death_cross_date(weekly: pd.DataFrame, position: int) -> str:
    hits = weekly.iloc[position + 1:]
    hits = hits[hits["SKDJ_Death_Cross"].map(to_bool)]
    return str(hits.iloc[0]["trade_date"]) if not hits.empty else ""


def find_daily_trigger(daily_ind: pd.DataFrame, signal_date: str, death_date: str,
                       market_end: str, open_dates: list[str], open_pos: dict[str, int]) -> dict[str, Any]:
    result = {
        "Daily_MACD_Confirmed": False, "Daily_Trigger_Date": "", "Daily_Trigger_Mode": "未确认",
        "Daily_Red_Run_Start": "", "Daily_Red_Age_At_Weekly_Cross": np.nan,
        "Wait_Market_Days": np.nan, "Daily_MACD_Hist_At_Cross": np.nan,
        "No_Confirm_Reason": "等待窗口内没有日线MACD首红",
    }
    hits = daily_ind.index[daily_ind["trade_date"].astype(str).eq(signal_date)].tolist()
    if not hits or signal_date not in open_pos:
        result["No_Confirm_Reason"] = "周线信号日无个股行情或不在市场日历"
        return result
    pos = int(hits[-1])
    hist = finite_num(daily_ind.iloc[pos]["D_MACD_Hist"])
    result["Daily_MACD_Hist_At_Cross"] = hist
    if hist > 0:
        run_start = pos
        while run_start > 0 and finite_num(daily_ind.iloc[run_start - 1]["D_MACD_Hist"]) > 0:
            run_start -= 1
        run_date = str(daily_ind.iloc[run_start]["trade_date"])
        age = pos - run_start
        result["Daily_Red_Run_Start"] = run_date
        result["Daily_Red_Age_At_Weekly_Cross"] = float(age)
        same_week = pd.Timestamp(run_date).to_period("W-FRI") == pd.Timestamp(signal_date).to_period("W-FRI")
        if same_week:
            result.update({
                "Daily_MACD_Confirmed": True, "Daily_Trigger_Date": signal_date,
                "Daily_Trigger_Mode": "周线金叉当周日线已首红", "Wait_Market_Days": 0.0,
                "No_Confirm_Reason": "",
            })
            return result

    deadline_pos = min(open_pos[signal_date] + DAILY_CONFIRM_MAX_WAIT, len(open_dates) - 1)
    deadline = min(open_dates[deadline_pos], market_end)
    search = daily_ind[
        daily_ind["trade_date"].astype(str).gt(signal_date)
        & daily_ind["trade_date"].astype(str).le(deadline)
        & daily_ind["D_MACD_First_Red"].map(to_bool)
    ].copy()
    if death_date:
        search = search[search["trade_date"].astype(str).lt(death_date)]
    if search.empty:
        if death_date and death_date <= deadline:
            result["No_Confirm_Reason"] = "周线SKDJ死叉前没有日线MACD首红"
        elif hist > 0:
            result["No_Confirm_Reason"] = "日线红柱早于周线金叉当周，20日内没有新首红"
        return result
    trigger_date = str(search.iloc[0]["trade_date"])
    result.update({
        "Daily_MACD_Confirmed": True, "Daily_Trigger_Date": trigger_date,
        "Daily_Trigger_Mode": "周线金叉后等待日线首红",
        "Daily_Red_Run_Start": trigger_date,
        "Wait_Market_Days": float(open_pos[trigger_date] - open_pos[signal_date]),
        "No_Confirm_Reason": "",
    })
    return result


def is_main_board(ts_code: str) -> bool:
    return not str(ts_code).startswith(("300", "301", "688", "689"))


def simulate_exit(path: pd.DataFrame, entry_price: float, target_pct: float,
                  stop_pct: float, sell_slippage_pct: float) -> dict[str, Any]:
    target, stop = entry_price * (1 + target_pct / 100), entry_price * (1 - stop_pct / 100)
    for day_no, row in enumerate(path.sort_values("trade_date").itertuples(index=False), start=1):
        op, high, low = float(row.open), float(row.high), float(row.low)
        if op <= stop:
            raw, reason = op, "跳空止损"
        elif op >= target:
            raw, reason = op, "跳空止盈"
        elif high >= target and low <= stop:
            raw, reason = stop, "同日双触发_按止损"
        elif low <= stop:
            raw, reason = stop, "止损"
        elif high >= target:
            raw, reason = target, "止盈"
        else:
            continue
        price = raw * (1 - sell_slippage_pct / 100)
        return {"date": str(row.trade_date), "price": price,
                "return_pct": (price / entry_price - 1) * 100, "days": float(day_no), "reason": reason}
    if path.empty:
        return {"date": "", "price": np.nan, "return_pct": np.nan, "days": np.nan, "reason": "无行情"}
    last = path.sort_values("trade_date").iloc[-1]
    price = float(last["close"]) * (1 - sell_slippage_pct / 100)
    return {"date": str(last["trade_date"]), "price": price,
            "return_pct": (price / entry_price - 1) * 100, "days": float(len(path)), "reason": "八周到期"}


def evaluate_entry(daily: pd.DataFrame, trigger_date: str, ts_code: str,
                   open_dates: list[str], open_pos: dict[str, int], config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "Tradable": False, "Untradable_Reason": "未来交易日不足", "Entry_Date": "",
        "Entry_Price": np.nan, "Has_8W_Future": False, "MFE_8W_pct": np.nan,
        "MAE_8W_pct": np.nan, "Return_8W_pct": np.nan,
    }
    for target in (10, 20, 30):
        out.update({f"First_{target}_vs_Stop": "", f"Exit_T{target}_Date": "",
                    f"Exit_T{target}_Price": np.nan, f"Exit_T{target}_Return_pct": np.nan,
                    f"Exit_T{target}_Reason": ""})
    if not trigger_date or trigger_date not in open_pos:
        out["Untradable_Reason"] = "没有有效触发日"
        return out
    entry_pos = open_pos[trigger_date] + 1
    if entry_pos >= len(open_dates):
        return out
    entry_date = open_dates[entry_pos]
    out["Entry_Date"] = entry_date
    rows = daily[daily["trade_date"].astype(str).eq(entry_date)]
    if rows.empty:
        out["Untradable_Reason"] = "D1停牌或无行情"
        return out
    row = rows.iloc[-1]
    if is_main_board(ts_code) and float(row["open"]) == float(row["high"]) == float(row["low"]):
        out["Untradable_Reason"] = "主板D1一字板"
        return out
    entry_price = float(row["open"]) * (1 + config["buy_slippage_pct"] / 100)
    out.update({"Tradable": True, "Untradable_Reason": "", "Entry_Price": entry_price})
    horizon_pos = entry_pos + HOLD_TRADING_DAYS - 1
    if horizon_pos >= len(open_dates):
        out["Untradable_Reason"] = "可买但未来不足40个市场交易日"
        return out
    horizon_date = open_dates[horizon_pos]
    path = daily[daily["trade_date"].astype(str).between(entry_date, horizon_date)].copy().sort_values("trade_date")
    if path.empty:
        out["Untradable_Reason"] = "八周窗口无行情"
        return out
    out["Has_8W_Future"] = True
    out["MFE_8W_pct"] = (float(path["high"].max()) / entry_price - 1) * 100
    out["MAE_8W_pct"] = (float(path["low"].min()) / entry_price - 1) * 100
    out["Return_8W_pct"] = (float(path.iloc[-1]["close"]) / entry_price - 1) * 100
    # 买入当日不能卖出，退出路径严格从下一交易日开始。
    exit_path = path[path["trade_date"].astype(str).gt(entry_date)]
    for target in (10, 20, 30):
        result = simulate_exit(exit_path, entry_price, float(target), config["stop_pct"], config["sell_slippage_pct"])
        out[f"First_{target}_vs_Stop"] = result["reason"]
        out[f"Exit_T{target}_Date"] = result["date"]
        out[f"Exit_T{target}_Price"] = result["price"]
        out[f"Exit_T{target}_Return_pct"] = result["return_pct"]
        out[f"Exit_T{target}_Reason"] = result["reason"]
    return out


def build_event(stock: pd.Series, membership: dict[str, str], weekly: pd.DataFrame,
                position: int, daily: pd.DataFrame, daily_basic: pd.DataFrame,
                daily_ind: pd.DataFrame, open_dates: list[str], open_pos: dict[str, int],
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
    recent = weekly.iloc[max(0, position - 2):position + 1]
    recent_min = float(pd.concat([recent["SKDJ_K"], recent["SKDJ_D"]]).min())
    death_date = next_death_cross_date(weekly, position)
    trigger = find_daily_trigger(daily_ind, signal_date, death_date, config["market_end"], open_dates, open_pos)
    base_path = evaluate_entry(daily, signal_date, str(stock["ts_code"]), open_dates, open_pos, config)
    daily_path = evaluate_entry(daily, trigger["Daily_Trigger_Date"], str(stock["ts_code"]), open_dates, open_pos, config)
    lag = weekly_macd_lag(weekly, position)
    event = {
        "ts_code": str(stock["ts_code"]), "name": str(stock["name"]),
        "Sample_Board": sample_board(stock), "SW_L1": membership["l1"],
        "SW_L2": membership["l2"], "SW_L3": membership["l3"],
        "Signal_Date": signal_date, "Weekly_Close": float(row["close"]),
        "Weekly_SKDJ_K": float(row["SKDJ_K"]), "Weekly_SKDJ_D": float(row["SKDJ_D"]),
        "Cross_Level": level, "Cross_25_Group": "≤25" if level <= SKDJ_BOTTOM else ">25",
        "Cross_Level_Bin": cross_level_bin(level), "Recent_3W_Min_SKDJ": recent_min,
        "Recent_3W_Touched_25": recent_min <= SKDJ_BOTTOM,
        "Weekly_SKDJ_Death_Cross_Date": death_date,
        **snapshot, **lag, **trigger,
    }
    event["Weekly_MACD_Lag_Group"] = lag_bin(event["Lead_Weeks_To_Weekly_MACD"])
    event.update({f"Cross_NextOpen_{key}": value for key, value in base_path.items()})
    event.update({f"Daily_Confirm_{key}": value for key, value in daily_path.items()})
    if base_path.get("Has_8W_Future") and daily_path.get("Has_8W_Future"):
        event["Delta_T20_Return_Daily_vs_Direct_pct"] = finite_num(daily_path["Exit_T20_Return_pct"]) - finite_num(base_path["Exit_T20_Return_pct"])
        event["Delta_MFE_Daily_vs_Direct_pct"] = finite_num(daily_path["MFE_8W_pct"]) - finite_num(base_path["MFE_8W_pct"])
    else:
        event["Delta_T20_Return_Daily_vs_Direct_pct"] = np.nan
        event["Delta_MFE_Daily_vs_Direct_pct"] = np.nan
    return event


def analyze_stock(stock: pd.Series, periods: list[dict[str, str]], daily: pd.DataFrame,
                  daily_basic: pd.DataFrame, week_last_map: dict[pd.Timestamp, str],
                  open_dates: list[str], open_pos: dict[str, int], config: dict[str, Any]) -> list[dict[str, Any]]:
    weekly = build_complete_weekly(daily, week_last_map)
    if len(weekly) < MACD_WARMUP_WEEKS:
        config["rejects"]["周线不足"] = config["rejects"].get("周线不足", 0) + 1
        return []
    daily_ind = add_daily_macd(daily)
    records = []
    for position in range(MACD_WARMUP_WEEKS, len(weekly)):
        if not bool(weekly.iloc[position]["SKDJ_Golden_Cross"]):
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
        event = build_event(stock, membership, weekly, position, daily, daily_basic,
                            daily_ind, open_dates, open_pos, config)
        if event is not None:
            records.append(event)
    return records


# -----------------------------------------------------------------------------
# 汇总与三仓随机顺序敏感性
# -----------------------------------------------------------------------------
def natural_week_calendar(open_dates: list[str], start: str, end: str, events: pd.DataFrame) -> pd.DataFrame:
    days = pd.DataFrame({"trade_date": [d for d in open_dates if start <= d <= end]})
    days["dt"] = pd.to_datetime(days["trade_date"])
    days["week"] = days["dt"].dt.to_period("W-FRI")
    weeks = days.groupby("week")["trade_date"].max().rename("Week_Last_Trade_Date").reset_index(drop=True).to_frame()
    counts = events.groupby("Signal_Date").size()
    weeks["All_Cross_Count"] = weeks["Week_Last_Trade_Date"].map(counts).fillna(0).astype(int)
    for group in ("≤25", ">25"):
        grouped = events[events["Cross_25_Group"].eq(group)].groupby("Signal_Date").size()
        weeks[f"Cross_{group}_Count"] = weeks["Week_Last_Trade_Date"].map(grouped).fillna(0).astype(int)
    weeks["Is_Empty_Week"] = weeks["All_Cross_Count"].eq(0)
    return weeks


def stat_row(frame: pd.DataFrame, label: str, natural_weeks: int) -> dict[str, Any]:
    direct = frame[frame["Cross_NextOpen_Has_8W_Future"].map(to_bool)] if len(frame) else frame
    confirmed = frame[frame["Daily_Confirm_Has_8W_Future"].map(to_bool)] if len(frame) else frame
    early = frame[frame["Weekly_MACD_State_At_Cross"].eq("绿柱或零轴") & ~frame["Weekly_MACD_Confirm_Censored"].map(to_bool)] if len(frame) else frame
    weekly_counts = frame.groupby("Signal_Date").size() if len(frame) else pd.Series(dtype=float)
    return {
        "分组": label, "周线SKDJ金叉数": len(frame), "不同股票": frame["ts_code"].nunique() if len(frame) else 0,
        "信号周": frame["Signal_Date"].nunique() if len(frame) else 0,
        "平均每自然周信号": len(frame) / natural_weeks if natural_weeks else np.nan,
        "空窗周": natural_weeks - frame["Signal_Date"].nunique() if natural_weeks else np.nan,
        "单周超过3只": int(weekly_counts.gt(3).sum()), "单周最多": int(weekly_counts.max()) if len(weekly_counts) else 0,
        "日线MACD确认数": int(frame["Daily_MACD_Confirmed"].map(to_bool).sum()) if len(frame) else 0,
        "日线MACD确认率(%)": float(frame["Daily_MACD_Confirmed"].map(to_bool).mean() * 100) if len(frame) else np.nan,
        "平均等待日线确认天数": pd.to_numeric(frame.get("Wait_Market_Days"), errors="coerce").mean() if len(frame) else np.nan,
        "MACD原为绿柱可比数": len(early),
        "平均领先周数": pd.to_numeric(early.get("Lead_Weeks_To_Weekly_MACD"), errors="coerce").mean() if len(early) else np.nan,
        "MACD首红前平均涨幅(%)": pd.to_numeric(early.get("Price_Gain_To_Weekly_MACD_pct"), errors="coerce").mean() if len(early) else np.nan,
        "MACD首红前涨幅中位数(%)": pd.to_numeric(early.get("Price_Gain_To_Weekly_MACD_pct"), errors="coerce").median() if len(early) else np.nan,
        "金叉次日买20%先到率(%)": float(direct["Cross_NextOpen_First_20_vs_Stop"].astype(str).str.contains("止盈").mean() * 100) if len(direct) else np.nan,
        "金叉次日买平均T20收益(%)": pd.to_numeric(direct.get("Cross_NextOpen_Exit_T20_Return_pct"), errors="coerce").mean() if len(direct) else np.nan,
        "金叉次日买平均MFE(%)": pd.to_numeric(direct.get("Cross_NextOpen_MFE_8W_pct"), errors="coerce").mean() if len(direct) else np.nan,
        "日线确认买成熟数": len(confirmed),
        "日线确认买20%先到率(%)": float(confirmed["Daily_Confirm_First_20_vs_Stop"].astype(str).str.contains("止盈").mean() * 100) if len(confirmed) else np.nan,
        "日线确认买平均T20收益(%)": pd.to_numeric(confirmed.get("Daily_Confirm_Exit_T20_Return_pct"), errors="coerce").mean() if len(confirmed) else np.nan,
        "日线确认买平均MFE(%)": pd.to_numeric(confirmed.get("Daily_Confirm_MFE_8W_pct"), errors="coerce").mean() if len(confirmed) else np.nan,
    }


def summaries(events: pd.DataFrame, pool_calendar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    natural_weeks = len(pool_calendar)
    rows = [stat_row(events, "全部金叉", natural_weeks)]
    for group, frame in events.groupby("Cross_25_Group", sort=False):
        rows.append(stat_row(frame, f"金叉位置{group}", natural_weeks))
    for group, frame in events.groupby("Cross_Level_Bin", sort=False):
        rows.append(stat_row(frame, f"细分_{group}", natural_weeks))
    overall = pd.DataFrame(rows)
    yearly_rows = []
    years = events["Signal_Date"].astype(str).str[:4]
    for (year, group), frame in events.groupby([years, "Cross_25_Group"]):
        year_weeks = int(pool_calendar["Week_Last_Trade_Date"].astype(str).str[:4].eq(str(year)).sum())
        row = stat_row(frame, f"{year}_{group}", natural_weeks=max(1, year_weeks))
        row["年份"] = year; row["25分组"] = group
        yearly_rows.append(row)
    yearly = pd.DataFrame(yearly_rows)
    lag_rows = []
    for (cross_group, lag_group), frame in events.groupby(["Cross_25_Group", "Weekly_MACD_Lag_Group"]):
        row = stat_row(frame, f"{cross_group}_{lag_group}", natural_weeks)
        row["25分组"] = cross_group; row["MACD迟到分组"] = lag_group
        lag_rows.append(row)
    return overall, yearly, pd.DataFrame(lag_rows)


def fee(amount: float, rate_pct: float, minimum: float = 0.0) -> float:
    return max(minimum, amount * rate_pct / 100) if amount > 0 else 0.0


def close_on_or_before(history: pd.DataFrame, trade_date: str) -> float:
    rows = history[history["trade_date"].astype(str).le(trade_date)] if not history.empty else pd.DataFrame()
    return finite_num(rows.iloc[-1]["close"]) if not rows.empty else np.nan


def simulate_portfolio(events: pd.DataFrame, histories: dict[str, pd.DataFrame], open_dates: list[str],
                       config: dict[str, Any], seed: int, build_detail: bool = False):
    work = events[
        events["Daily_Confirm_Tradable"].map(to_bool)
        & events["Daily_Confirm_Has_8W_Future"].map(to_bool)
        & events["Daily_Confirm_Entry_Date"].astype(str).ne("")
        & events["Daily_Confirm_Exit_T20_Date"].astype(str).ne("")
    ].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"实际买入": 0, "仓位满错过": 0, "总收益率(%)": np.nan}
    rng = np.random.default_rng(seed)
    work["_tie"] = rng.random(len(work))
    work = work.sort_values(["Daily_Confirm_Entry_Date", "_tie", "ts_code"], kind="mergesort")
    entry_groups = {day: frame for day, frame in work.groupby("Daily_Confirm_Entry_Date", sort=True)}
    first_day, last_day = str(work["Daily_Confirm_Entry_Date"].min()), str(work["Daily_Confirm_Exit_T20_Date"].max())
    days = [day for day in open_dates if first_day <= day <= last_day]
    cash = INITIAL_CAPITAL
    active: dict[str, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    for trade_date in days:
        for _, row in entry_groups.get(trade_date, pd.DataFrame()).iterrows():
            code = str(row["ts_code"])
            reason = "同一股票已持仓" if code in active else "3个仓位已满" if len(active) >= MAX_POSITIONS else ""
            price = finite_num(row["Daily_Confirm_Entry_Price"])
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
                    "name": row.get("name", ""), "Cross_25_Group": row["Cross_25_Group"],
                    "Action": "未买入" if reason else "已买入", "Reason": reason or f"随机同日顺序买入{shares}股",
                    "Prospective_Exit_Date": row["Daily_Confirm_Exit_T20_Date"], "Tie_Seed": seed,
                })
            if reason:
                continue
            amount = shares * price
            buy_fee = fee(amount, config["commission_pct"], 5.0) + fee(amount, config["transfer_fee_pct"])
            cash -= amount + buy_fee
            trade = {
                "Signal_Date": row["Signal_Date"], "Entry_Date": trade_date, "ts_code": code,
                "name": row.get("name", ""), "Cross_25_Group": row["Cross_25_Group"],
                "Shares": shares, "Entry_Price": price, "Entry_Amount": amount, "Buy_Fees": buy_fee,
                "Planned_Exit_Date": str(row["Daily_Confirm_Exit_T20_Date"]),
                "Exit_Price": finite_num(row["Daily_Confirm_Exit_T20_Price"]),
                "Exit_Reason": row["Daily_Confirm_Exit_T20_Reason"], "PnL": np.nan, "Net_Return_pct": np.nan,
            }
            active[code] = trade; ledger.append(trade)
        exiting = []
        for code, trade in active.items():
            if trade["Planned_Exit_Date"] != trade_date:
                continue
            gross = trade["Shares"] * trade["Exit_Price"]
            sell_fee = (fee(gross, config["commission_pct"], 5.0)
                        + fee(gross, config["transfer_fee_pct"])
                        + fee(gross, config["stamp_duty_pct"]))
            proceeds = gross - sell_fee
            cash += proceeds
            pnl = proceeds - trade["Entry_Amount"] - trade["Buy_Fees"]
            trade.update({"Sell_Fees": sell_fee, "Exit_Proceeds": proceeds, "PnL": pnl,
                          "Net_Return_pct": pnl / (trade["Entry_Amount"] + trade["Buy_Fees"]) * 100})
            exiting.append(code)
        for code in exiting:
            active.pop(code, None)
        if build_detail:
            market_value = 0.0
            for code, trade in active.items():
                mark = close_on_or_before(histories.get(code, pd.DataFrame()), trade_date)
                market_value += trade["Shares"] * (mark if math.isfinite(mark) else trade["Entry_Price"])
            curve.append({"Trade_Date": trade_date, "Cash": cash, "Market_Value": market_value,
                          "Equity": cash + market_value, "Positions": len(active)})
    ledger_frame, orders_frame, curve_frame = pd.DataFrame(ledger), pd.DataFrame(orders), pd.DataFrame(curve)
    final_equity = cash
    if len(curve_frame):
        curve_frame["Drawdown_pct"] = (curve_frame["Equity"] / curve_frame["Equity"].cummax().clip(lower=INITIAL_CAPITAL) - 1) * 100
        max_dd = finite_num(curve_frame["Drawdown_pct"].min())
    else:
        max_dd = np.nan
    summary = {
        "同日随机种子": seed, "初始资金": INITIAL_CAPITAL, "实际买入": len(ledger_frame),
        "仓位满错过": int(orders_frame.get("Reason", pd.Series(dtype=str)).eq("3个仓位已满").sum()) if build_detail else np.nan,
        "期末权益": final_equity, "总收益率(%)": (final_equity / INITIAL_CAPITAL - 1) * 100,
        "最大回撤(%)": max_dd, "交易胜率(%)": float(ledger_frame["PnL"].gt(0).mean() * 100) if len(ledger_frame) else np.nan,
    }
    return curve_frame, ledger_frame, orders_frame, summary


def run_monte_carlo(events: pd.DataFrame, histories: dict[str, pd.DataFrame], open_dates: list[str],
                    config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = {
        "全部金叉": events,
        "金叉≤25": events[events["Cross_25_Group"].eq("≤25")],
        "金叉>25": events[events["Cross_25_Group"].eq(">25")],
        "金叉25-35": events[events["Cross_Level_Bin"].eq("25-35")],
    }
    rows = []
    for name, frame in variants.items():
        for seed in range(MC_RUNS):
            _, ledger, _, summary = simulate_portfolio(frame, histories, open_dates, config, seed, False)
            rows.append({"策略组": name, "随机种子": seed, "总收益率(%)": summary["总收益率(%)"],
                         "实际买入": summary["实际买入"], "交易胜率(%)": summary["交易胜率(%)"]})
    distribution = pd.DataFrame(rows)
    summary_rows = []
    for name, frame in distribution.groupby("策略组"):
        returns = pd.to_numeric(frame["总收益率(%)"], errors="coerce")
        summary_rows.append({
            "策略组": name, "随机次数": len(frame), "平均总收益率(%)": returns.mean(),
            "中位总收益率(%)": returns.median(), "标准差(百分点)": returns.std(),
            "最差总收益率(%)": returns.min(), "最好总收益率(%)": returns.max(),
            "盈利模拟比例(%)": returns.gt(0).mean() * 100,
            "平均买入笔数": pd.to_numeric(frame["实际买入"], errors="coerce").mean(),
        })
    return pd.DataFrame(summary_rows), distribution


# -----------------------------------------------------------------------------
# SKDJ参数锚点审计
# -----------------------------------------------------------------------------
def anchor_values(daily_300458: pd.DataFrame, daily_688017: pd.DataFrame, n: int, m: int) -> dict[str, float]:
    out = {"Daily_K": np.nan, "Daily_D": np.nan, "Weekly_K": np.nan, "Weekly_D": np.nan}
    if not daily_300458.empty:
        daily_skdj = add_skdj(daily_300458, n, m)
        hit = daily_skdj[daily_skdj["trade_date"].astype(str).eq("20260813")]
        if not hit.empty:
            out["Daily_K"], out["Daily_D"] = finite_num(hit.iloc[-1]["SKDJ_K"]), finite_num(hit.iloc[-1]["SKDJ_D"])
    if not daily_688017.empty:
        weekly_skdj = build_provisional_weekly(daily_688017, n, m)
        hit = weekly_skdj[weekly_skdj["trade_date"].astype(str).eq("20260813")]
        if not hit.empty:
            out["Weekly_K"], out["Weekly_D"] = finite_num(hit.iloc[-1]["SKDJ_K"]), finite_num(hit.iloc[-1]["SKDJ_D"])
    return out


def parameter_anchor_audit(daily_300458: pd.DataFrame, daily_688017: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in range(2, 61):
        for m in range(2, 21):
            values = anchor_values(daily_300458, daily_688017, n, m)
            errors = []
            for key, target in (("Daily_K", 53.11), ("Daily_D", 42.92), ("Weekly_K", 33.22), ("Weekly_D", 33.07)):
                value = values[key]
                if math.isfinite(value):
                    errors.append(abs(value - target))
            rows.append({
                "N": n, "M": m, "全志日K": values["Daily_K"], "全志日D": values["Daily_D"],
                "绿谐周K": values["Weekly_K"], "绿谐周D": values["Weekly_D"],
                "可用锚点数": len(errors), "平均绝对误差": float(np.mean(errors)) if errors else np.nan,
                "冻结参数N9M3": n == SKDJ_N and m == SKDJ_M,
            })
    result = pd.DataFrame(rows).sort_values(["可用锚点数", "平均绝对误差", "N", "M"], ascending=[False, True, True, True])
    frozen = result[result["冻结参数N9M3"]]
    return pd.concat([result.head(30), frozen]).drop_duplicates(["N", "M"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Streamlit主程序
# -----------------------------------------------------------------------------
def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title="周线SKDJ池 V3.1", layout="wide")
    st.title(TITLE)
    st.caption("完整周线SKDJ金叉建池；比较金叉≤25与>25；日线MACD首红确认进场。周线MACD只测迟到，不参与选股。")
    with st.expander("冻结规则", expanded=True):
        st.markdown(f"""
- **股票池事件**：完整周线 `K>D 且上周K≤D`；SKDJ暂冻结 `N={SKDJ_N}, M={SKDJ_M}`。
- **位置比较**：主比较以金叉当周 `(K+D)/2` 是否≤25分组；同时报告≤20、20-25、25-35、35-50、>50。
- **日线确认**：若周线金叉当周日线MACD已经首红，周五确认；否则最多等待{DAILY_CONFIRM_MAX_WAIT}个市场交易日。
- **失效**：等待期间先出现完整周线SKDJ死叉，则不再买入；日线红柱若早于金叉当周，不视为新鲜确认。
- **成交与判卷**：触发后下一市场交易日开盘；-10%止损、+20%止盈、最长40个市场交易日；严格T+1。
- **禁止事项**：不评分、不按收益调参数、不把周线MACD首红作为候选条件。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("信号开始", date(2023, 6, 5), key="v31_start")
        signal_end_date = st.date_input("信号截止", date(2026, 6, 5), key="v31_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v31_market_end")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v31_pause")
        use_cache = st.checkbox("复用逐股票缓存", True, key="v31_cache")
        st.divider()
        commission_pct = st.number_input("佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f")
        stamp_duty_pct = st.number_input("卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f")
        transfer_fee_pct = st.number_input("过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f")
        if st.button("清除本程序行情缓存", key="v31_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True); st.success("缓存已清除")
    token = st.text_input("Tushare Token", type="password", key="v31_token")
    session_key = "weekly_skdj_pool_v31_zip"
    if not token:
        st.info("请输入Tushare Token；同日期逐股票行情缓存可以直接复用。")
        return
    if not st.button("开始V3.1验证", type="primary", key="v31_run"):
        if session_key in st.session_state:
            st.download_button("下载上一次结果ZIP", st.session_state[session_key],
                               file_name="weekly_skdj_pool_daily_macd_audit_v3_1_all_results.zip", mime="application/zip")
        return
    error = validate_dates(signal_start_date, signal_end_date, market_end_date)
    if error:
        st.error(error); return
    API_ERRORS = []
    ts.set_token(token); pro = ts.pro_api()
    signal_start, signal_end, market_end = (signal_start_date.strftime("%Y%m%d"),
                                             signal_end_date.strftime("%Y%m%d"),
                                             market_end_date.strftime("%Y%m%d"))
    preload = (signal_start_date - timedelta(days=3 * 365)).strftime("%Y%m%d")
    rejects: dict[str, int] = {}
    config = {
        "signal_start": signal_start, "signal_end": signal_end, "market_end": market_end,
        "min_price": 10.0, "min_mv": 100.0, "buy_slippage_pct": 0.20,
        "sell_slippage_pct": 0.20, "stop_pct": 10.0, "commission_pct": float(commission_pct),
        "stamp_duty_pct": float(stamp_duty_pct), "transfer_fee_pct": float(transfer_fee_pct),
        "rejects": rejects,
    }
    try:
        with st.spinner("加载交易日历和历史科技股池..."):
            open_dates = load_trade_calendar(preload, market_end)
            full_open_dates = load_trade_calendar(preload, (market_end_date + timedelta(days=7)).strftime("%Y%m%d"))
            week_last_map = complete_week_last_dates(full_open_dates)
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}"); return
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
    anchor_histories: dict[str, pd.DataFrame] = {}
    cache_hits = data_failures = 0
    progress, status = st.progress(0.0), st.empty()
    for number, stock in stocks.iterrows():
        code = str(stock["ts_code"])
        progress.progress((number + 1) / max(len(stocks), 1), text=f"{number + 1}/{len(stocks)} {code}")
        status.caption(f"周线SKDJ金叉 {len(events)}；缓存 {cache_hits}；失败 {data_failures}")
        daily, daily_basic, cache_hit = fetch_stock_history(code, preload, market_end, bool(use_cache), float(pause))
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1; continue
        if code in {"300458.SZ", "688017.SH"}:
            anchor_histories[code] = daily.copy()
        records = analyze_stock(stock, period_index.get(code, []), daily, daily_basic,
                                week_last_map, open_dates, open_pos, config)
        if records:
            events.extend(records); histories[code] = daily.copy()
    progress.empty(); status.empty()
    # 锚点股票即使不在事件样本中也强制取数，避免参数审计再次出现空表。
    for code in ("300458.SZ", "688017.SH"):
        if code not in anchor_histories:
            daily, _, hit = fetch_stock_history(code, preload, market_end, bool(use_cache), float(pause))
            cache_hits += int(hit)
            if not daily.empty:
                anchor_histories[code] = daily
    if not events:
        st.error("研究区间没有生成符合价格、市值和历史科技池条件的周线SKDJ金叉事件。"); return
    try:
        with st.spinner("生成位置比较、MACD迟到审计、三仓随机顺序敏感性..."):
            event_frame = pd.DataFrame(events).sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True)
            pool_calendar = natural_week_calendar(open_dates, signal_start, signal_end, event_frame)
            overall, yearly, lag_summary = summaries(event_frame, pool_calendar)
            executable = event_frame[
                event_frame["Daily_Confirm_Tradable"].map(to_bool)
                & event_frame["Daily_Confirm_Entry_Date"].astype(str).ne("")
            ]
            competition = executable.groupby("Daily_Confirm_Entry_Date").agg(
                Signal_Count=("ts_code", "size"), Unique_Stocks=("ts_code", "nunique"),
                Cross_Weeks=("Signal_Date", "nunique")
            ).reset_index().rename(columns={"Daily_Confirm_Entry_Date": "Entry_Date"})
            competition["More_Than_3"] = competition["Signal_Count"].gt(3)
            no_confirm = event_frame[~event_frame["Daily_MACD_Confirmed"].map(to_bool)].copy()
            curve, ledger, orders, primary_summary = simulate_portfolio(
                event_frame, histories, open_dates, config, PRIMARY_SEED, True)
            mc_summary, mc_distribution = run_monte_carlo(event_frame, histories, open_dates, config)
            anchor_audit = parameter_anchor_audit(
                anchor_histories.get("300458.SZ", pd.DataFrame()),
                anchor_histories.get("688017.SH", pd.DataFrame()))
    except Exception as exc:
        st.exception(exc); return
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "信号开始": signal_start, "信号截止": signal_end,
        "观察截止": market_end, "周线SKDJ金叉": len(event_frame), "不同股票": event_frame["ts_code"].nunique(),
        "自然周": len(pool_calendar), "平均每周金叉": len(event_frame) / len(pool_calendar),
        "空窗周": int(pool_calendar["Is_Empty_Week"].sum()), "金叉≤25": int(event_frame["Cross_25_Group"].eq("≤25").sum()),
        "金叉>25": int(event_frame["Cross_25_Group"].eq(">25").sum()),
        "日线MACD确认": int(event_frame["Daily_MACD_Confirmed"].map(to_bool).sum()),
        "超过3只的买入日": int(competition["More_Than_3"].sum()),
        "单日最多": int(competition["Signal_Count"].max()) if len(competition) else 0,
        "主三仓总收益率(%)": primary_summary.get("总收益率(%)", np.nan),
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    metadata = pd.DataFrame([
        ("候选池", "完整周线SKDJ金叉；周线MACD不参与候选生成"),
        ("SKDJ公式", "LOWV=LLV(LOW,N); HIGHV=HHV(HIGH,N); RSV=EMA((C-LOWV)/(HIGHV-LOWV)*100,M); K=EMA(RSV,M); D=MA(K,M)"),
        ("冻结参数", f"N={SKDJ_N},M={SKDJ_M}；不根据收益寻优"),
        ("25分组", "以金叉当周(K+D)/2≤25或>25分组；另报五档位置"),
        ("日线确认", f"金叉当周日线MACD已首红，或金叉后{DAILY_CONFIRM_MAX_WAIT}个市场交易日内出现首红"),
        ("候选失效", "等待期间先确认完整周线SKDJ死叉，则取消候选"),
        ("周线MACD", "只记录从SKDJ金叉到周线MACD首红的周数和期间涨幅，不参与交易"),
        ("判卷", "触发后下一市场交易日开盘；-10%止损、+20%止盈、40市场交易日；T+1"),
        ("参数锚点", "全志科技300458日线20260813 K53.11/D42.92；绿的谐波688017周线当周K33.22/D33.07"),
    ], columns=["项目", "值"])
    files = {
        "01_run_summary_v3_1.csv": run_summary,
        "02_cross_level_comparison_v3_1.csv": overall,
        "03_yearly_cross_level_stability_v3_1.csv": yearly,
        "04_weekly_macd_lag_audit_v3_1.csv": lag_summary,
        "05_weekly_pool_calendar_v3_1.csv": pool_calendar,
        "06_daily_entry_competition_v3_1.csv": competition,
        "07_all_weekly_skdj_events_v3_1.csv": event_frame,
        "08_no_daily_macd_confirmation_v3_1.csv": no_confirm,
        "09_primary_3slot_summary_v3_1.csv": pd.DataFrame([primary_summary]),
        "10_primary_3slot_ledger_v3_1.csv": ledger,
        "11_primary_3slot_orders_v3_1.csv": orders,
        "12_primary_3slot_equity_curve_v3_1.csv": curve,
        "13_random_tie_mc_summary_v3_1.csv": mc_summary,
        "14_random_tie_mc_distribution_v3_1.csv": mc_distribution,
        "15_skdj_two_anchor_parameter_audit_v3_1.csv": anchor_audit,
        "16_full_tech_universe_v3_1.csv": stocks,
        "17_board_population_v3_1.csv": population,
        "18_rejection_audit_v3_1.csv": pd.DataFrame([{"剔除原因": k, "次数": v} for k, v in sorted(rejects.items())]),
        "19_api_errors_v3_1.csv": pd.DataFrame({"错误": API_ERRORS}),
        "20_metadata_v3_1.csv": metadata,
    }
    result_zip = make_zip(files); st.session_state[session_key] = result_zip
    st.success(f"完成：{len(event_frame)}个完整周线SKDJ金叉；≤25有{event_frame['Cross_25_Group'].eq('≤25').sum()}个，>25有{event_frame['Cross_25_Group'].eq('>25').sum()}个。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("周线金叉", len(event_frame)); c2.metric("平均每周", f"{len(event_frame)/len(pool_calendar):.2f}")
    c3.metric("超过3只买入日", int(competition["More_Than_3"].sum()))
    c4.metric("单日最多", int(competition["Signal_Count"].max()) if len(competition) else 0)
    st.subheader("金叉位置比较")
    st.dataframe(overall, use_container_width=True, hide_index=True)
    st.subheader("随机同日顺序三仓敏感性")
    st.dataframe(mc_summary, use_container_width=True, hide_index=True)
    st.download_button("下载V3.1全部结果ZIP", result_zip,
                       file_name="weekly_skdj_pool_daily_macd_audit_v3_1_all_results.zip",
                       mime="application/zip", type="primary", key="v31_download")
    st.info("先看≤25与>25的逐年稳定性、MACD迟到期间涨幅、日线确认后的信号拥挤度；不要只看一次三仓总收益。")


if __name__ == "__main__":
    main()
