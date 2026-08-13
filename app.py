# -*- coding: utf-8 -*-
"""
科技股周线SKDJ核心池：日线SKDJ Top2/Top3真实组合回测 V3.6

目的：
1. 唯一候选池为“近3个完整周触及25且金叉位置20~35”的周线SKDJ核心池。
2. 同一信号周按日线SKDJ位置从高到低，分别选择Top2和Top3。
3. 资金30万元、最多3仓、单仓预算10万元；已有持仓和仓位占用按交易日真实推进。
4. 主退出为日线SKDJ在75以上形成死叉后下一可交易日开盘卖出。
5. 固定持有40个市场交易日、到期收盘卖出作为对照。

注意：Top2/Top3规则来自V3.5同一段历史，V3.6验证的是组合可执行性，不是新的独立样本外证明。

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
TITLE = "科技股周线SKDJ核心池：日线SKDJ Top2/Top3真实组合回测 V3.6"
VERSION = "V3.6-WEEKLY-SKDJ-DAILY-SKDJ-TOPK-PORTFOLIO"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")

SKDJ_N = 9
SKDJ_M = 3
SKDJ_BOTTOM = 25.0
RESET_LOOKBACK_WEEKS = 3
CROSS_ZONE_LOW = 20.0
CROSS_ZONE_HIGH = 35.0
INDICATOR_WARMUP_WEEKS = 40
HOLD_20D = 20
HOLD_40D = 40
HIGH_DEATH_ZONE = 75.0

INITIAL_CAPITAL = 300_000.0
MAX_POSITIONS = 3
POSITION_BUDGET = 100_000.0

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

STRATEGIES = {
    "Top2_HighDeath": (2, "HighDeath", "日线SKDJ Top2＋高位死叉"),
    "Top3_HighDeath": (3, "HighDeath", "日线SKDJ Top3＋高位死叉"),
    "Top2_Fixed40": (2, "Fixed40", "日线SKDJ Top2＋固定40日"),
    "Top3_Fixed40": (3, "Fixed40", "日线SKDJ Top3＋固定40日"),
}

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


def safe_div(numerator: Any, denominator: Any) -> float:
    a, b = finite_num(numerator), finite_num(denominator)
    return a / b if math.isfinite(a) and math.isfinite(b) and b != 0 else np.nan


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
        "SKDJ_K": "D_SKDJ_K", "SKDJ_D": "D_SKDJ_D", "SKDJ_Golden_Cross": "D_SKDJ_Golden_Cross",
    })
    close = work["close"]
    prev = close.shift(1)
    true_range = pd.concat([
        work["high"] - work["low"], (work["high"] - prev).abs(), (work["low"] - prev).abs()
    ], axis=1).max(axis=1)
    dif = close.ewm(span=12, adjust=False, min_periods=1).mean() - close.ewm(span=26, adjust=False, min_periods=1).mean()
    dea = dif.ewm(span=9, adjust=False, min_periods=1).mean()
    work["D_MACD_Hist"] = 2.0 * (dif - dea)
    work["D_MACD_Hist_Change_1D"] = work["D_MACD_Hist"].diff()
    work["D_SKDJ_Level"] = (work["D_SKDJ_K"] + work["D_SKDJ_D"]) / 2.0
    work["D_SKDJ_Prev_Level"] = work["D_SKDJ_Level"].shift(1)
    work["D_SKDJ_Death_Cross"] = work["D_SKDJ_K"].lt(work["D_SKDJ_D"]) & work[
        "D_SKDJ_K"].shift(1).ge(work["D_SKDJ_D"].shift(1))
    work["D_SKDJ_K_Change_3D"] = work["D_SKDJ_K"].diff(3)
    work["D_Return_5D_pct"] = close.pct_change(5, fill_method=None) * 100.0
    work["D_Return_20D_pct"] = close.pct_change(20, fill_method=None) * 100.0
    work["D_Return_60D_pct"] = close.pct_change(60, fill_method=None) * 100.0
    work["D_MA20_Bias_pct"] = (close / close.rolling(20).mean() - 1.0) * 100.0
    work["D_MA60_Bias_pct"] = (close / close.rolling(60).mean() - 1.0) * 100.0
    work["D_Volume_Ratio_5_20"] = work["vol"].rolling(5).mean() / work["vol"].rolling(20).mean().replace(0, np.nan)
    work["D_ATR14_pct"] = true_range.rolling(14).mean() / close.replace(0, np.nan) * 100.0
    work["D_Amplitude_10D_pct"] = (work["high"].rolling(10).max() / work["low"].rolling(10).min() - 1.0) * 100.0
    work["D_Distance_60D_High_pct"] = (close / work["high"].rolling(60).max() - 1.0) * 100.0
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


def reset_quadrant(touched: bool, zone: bool) -> str:
    if touched and zone:
        return "触及25_金叉20-35"
    if touched:
        return "触及25_金叉区间外"
    if zone:
        return "未触及25_金叉20-35"
    return "未触及25_金叉区间外"


def weeks_since_touch_25(weekly: pd.DataFrame, position: int) -> float:
    prior = weekly.iloc[:position + 1]
    touched = prior[prior[["SKDJ_K", "SKDJ_D"]].min(axis=1).le(SKDJ_BOTTOM)]
    return float(position - int(touched.index[-1])) if not touched.empty else np.nan


def weekly_features(weekly: pd.DataFrame, position: int) -> dict[str, Any]:
    row = weekly.iloc[position]
    close, vol = weekly["close"], weekly["vol"]
    recent3 = weekly.iloc[max(0, position - 2):position + 1]
    recent6 = weekly.iloc[max(0, position - 5):position + 1]
    recent3_min = float(recent3[["SKDJ_K", "SKDJ_D"]].min().min())
    recent6_min = float(recent6[["SKDJ_K", "SKDJ_D"]].min().min())
    level = (float(row["SKDJ_K"]) + float(row["SKDJ_D"])) / 2.0
    range_pct = (weekly["high"] / weekly["low"].replace(0, np.nan) - 1.0) * 100.0
    return {
        "Weekly_SKDJ_K": float(row["SKDJ_K"]), "Weekly_SKDJ_D": float(row["SKDJ_D"]),
        "Weekly_Cross_Level": level, "Weekly_Cross_Level_Bin": weekly_cross_bin(level),
        "Recent_3W_Min_SKDJ": recent3_min, "Recent_6W_Min_SKDJ": recent6_min,
        "Weeks_Since_Touch_25": weeks_since_touch_25(weekly, position),
        "Recent_3W_Touched_25": recent3_min <= SKDJ_BOTTOM,
        "Cross_In_20_35": CROSS_ZONE_LOW < level <= CROSS_ZONE_HIGH,
        "Weekly_SKDJ_K_Change_1W": float(weekly["SKDJ_K"].diff().iloc[position]),
        "Weekly_SKDJ_D_Change_1W": float(weekly["SKDJ_D"].diff().iloc[position]),
        "Weekly_Return_1W_pct": float(close.pct_change(1, fill_method=None).iloc[position] * 100.0),
        "Weekly_Return_4W_pct": float(close.pct_change(4, fill_method=None).iloc[position] * 100.0),
        "Weekly_Return_12W_pct": float(close.pct_change(12, fill_method=None).iloc[position] * 100.0),
        "Weekly_MA20_Bias_pct": float((close / close.rolling(20).mean() - 1.0).iloc[position] * 100.0),
        "Weekly_Volume_Ratio_4_12": float((vol.rolling(4).mean() / vol.rolling(12).mean().replace(0, np.nan)).iloc[position]),
        "Weekly_Contraction_4_12": float((range_pct.rolling(4).mean() / range_pct.rolling(12).mean().replace(0, np.nan)).iloc[position]),
    }


def daily_features_at_signal(daily: pd.DataFrame, signal_date: str) -> dict[str, Any]:
    rows = daily.index[daily["trade_date"].astype(str).eq(signal_date)].tolist()
    defaults = {
        "Daily_SKDJ_Level_At_Cross": np.nan, "Daily_SKDJ_K_At_Cross": np.nan,
        "Daily_SKDJ_D_At_Cross": np.nan, "Daily_SKDJ_K_Change_3D": np.nan,
        "Daily_MACD_Hist": np.nan, "Daily_MACD_Hist_Change_1D": np.nan,
        "Daily_Return_5D_pct": np.nan, "Daily_Return_20D_pct": np.nan,
        "Daily_Return_60D_pct": np.nan, "Daily_MA20_Bias_pct": np.nan,
        "Daily_MA60_Bias_pct": np.nan, "Daily_Volume_Ratio_5_20": np.nan,
        "Daily_ATR14_pct": np.nan, "Daily_Amplitude_10D_pct": np.nan,
        "Distance_60D_High_pct": np.nan,
    }
    if not rows:
        return defaults
    row = daily.iloc[int(rows[-1])]
    return {
        "Daily_SKDJ_Level_At_Cross": finite_num(row.get("D_SKDJ_Level")),
        "Daily_SKDJ_K_At_Cross": finite_num(row.get("D_SKDJ_K")),
        "Daily_SKDJ_D_At_Cross": finite_num(row.get("D_SKDJ_D")),
        "Daily_SKDJ_K_Change_3D": finite_num(row.get("D_SKDJ_K_Change_3D")),
        "Daily_MACD_Hist": finite_num(row.get("D_MACD_Hist")),
        "Daily_MACD_Hist_Change_1D": finite_num(row.get("D_MACD_Hist_Change_1D")),
        "Daily_Return_5D_pct": finite_num(row.get("D_Return_5D_pct")),
        "Daily_Return_20D_pct": finite_num(row.get("D_Return_20D_pct")),
        "Daily_Return_60D_pct": finite_num(row.get("D_Return_60D_pct")),
        "Daily_MA20_Bias_pct": finite_num(row.get("D_MA20_Bias_pct")),
        "Daily_MA60_Bias_pct": finite_num(row.get("D_MA60_Bias_pct")),
        "Daily_Volume_Ratio_5_20": finite_num(row.get("D_Volume_Ratio_5_20")),
        "Daily_ATR14_pct": finite_num(row.get("D_ATR14_pct")),
        "Daily_Amplitude_10D_pct": finite_num(row.get("D_Amplitude_10D_pct")),
        "Distance_60D_High_pct": finite_num(row.get("D_Distance_60D_High_pct")),
    }


def build_portfolio_exit_plans(path: pd.DataFrame, entry_date: str,
                               entry_price: float, open_pos: dict[str, int],
                               config: dict[str, Any]) -> dict[str, Any]:
    ordered = path.sort_values("trade_date").reset_index(drop=True)
    defaults: dict[str, Any] = {}
    for method in ("HighDeath", "Fixed40"):
        defaults.update({
            f"{method}_Exit_Signal_Date": "", f"{method}_Exit_Date": "",
            f"{method}_Exit_Price": np.nan, f"{method}_Exit_Session": "",
            f"{method}_Exit_Reason": "", f"{method}_Hold_Market_Days": np.nan,
        })
    if ordered.empty:
        return defaults
    last = ordered.iloc[-1]
    last_date = str(last["trade_date"])
    fixed_price = float(last["close"]) * (1 - config["sell_slippage_pct"] / 100.0)
    fixed_hold = float(open_pos[last_date] - open_pos[entry_date] + 1) \
        if last_date in open_pos and entry_date in open_pos else np.nan
    defaults.update({
        "Fixed40_Exit_Signal_Date": last_date, "Fixed40_Exit_Date": last_date,
        "Fixed40_Exit_Price": fixed_price, "Fixed40_Exit_Session": "CLOSE",
        "Fixed40_Exit_Reason": "固定40个市场交易日到期",
        "Fixed40_Hold_Market_Days": fixed_hold,
        "HighDeath_Exit_Signal_Date": last_date, "HighDeath_Exit_Date": last_date,
        "HighDeath_Exit_Price": fixed_price, "HighDeath_Exit_Session": "CLOSE",
        "HighDeath_Exit_Reason": "40日内未出现高位死叉_到期退出",
        "HighDeath_Hold_Market_Days": fixed_hold,
    })
    for position, row in ordered.iterrows():
        if not to_bool(row.get("D_SKDJ_Death_Cross")) or position + 1 >= len(ordered):
            continue
        levels = [finite_num(row.get("D_SKDJ_Level")), finite_num(row.get("D_SKDJ_Prev_Level"))]
        if not any(math.isfinite(value) and value >= HIGH_DEATH_ZONE for value in levels):
            continue
        exit_row = ordered.iloc[position + 1]
        exit_date = str(exit_row["trade_date"])
        exit_price = float(exit_row["open"]) * (1 - config["sell_slippage_pct"] / 100.0)
        hold = float(open_pos[exit_date] - open_pos[entry_date] + 1) \
            if exit_date in open_pos and entry_date in open_pos else np.nan
        defaults.update({
            "HighDeath_Exit_Signal_Date": str(row["trade_date"]),
            "HighDeath_Exit_Date": exit_date, "HighDeath_Exit_Price": exit_price,
            "HighDeath_Exit_Session": "OPEN", "HighDeath_Exit_Reason": "日线SKDJ高位死叉",
            "HighDeath_Hold_Market_Days": hold,
        })
        break
    return defaults


def direct_outcomes(daily: pd.DataFrame, signal_date: str, ts_code: str,
                    open_dates: list[str], open_pos: dict[str, int], config: dict[str, Any]) -> dict[str, Any]:
    out = {
        "Tradable": False, "Untradable_Reason": "", "Entry_Date": "", "Entry_Price": np.nan,
        "Outcome_20D_End_Date": "", "Outcome_40D_End_Date": "", "Has_20D_Future": False,
        "Has_40D_Future": False, "Return_20D_pct": np.nan, "Return_40D_pct": np.nan,
        "MFE_20D_pct": np.nan, "MAE_20D_pct": np.nan, "MFE_40D_pct": np.nan, "MAE_40D_pct": np.nan,
        "Portfolio_Entry_Price": np.nan,
    }
    for method in ("HighDeath", "Fixed40"):
        out.update({
            f"{method}_Exit_Signal_Date": "", f"{method}_Exit_Date": "",
            f"{method}_Exit_Price": np.nan, f"{method}_Exit_Session": "",
            f"{method}_Exit_Reason": "", f"{method}_Hold_Market_Days": np.nan,
        })
    if signal_date not in open_pos or open_pos[signal_date] + 1 >= len(open_dates):
        out["Untradable_Reason"] = "未来交易日不足"
        return out
    entry_market_pos = open_pos[signal_date] + 1
    entry_date = open_dates[entry_market_pos]
    out["Entry_Date"] = entry_date
    rows = daily[daily["trade_date"].astype(str).eq(entry_date)]
    if rows.empty:
        out["Untradable_Reason"] = "D1停牌或无行情"
        return out
    first = rows.iloc[-1]
    if is_main_board(ts_code) and float(first["open"]) == float(first["high"]) == float(first["low"]):
        out["Untradable_Reason"] = "主板D1一字板"
        return out
    buy_cost = (config["commission_pct"] + config["transfer_fee_pct"]) / 100.0
    sell_cost = (config["commission_pct"] + config["transfer_fee_pct"] + config["stamp_duty_pct"]) / 100.0
    portfolio_entry_price = float(first["open"]) * (1 + config["buy_slippage_pct"] / 100.0)
    entry_price = portfolio_entry_price * (1 + buy_cost)
    out.update({"Tradable": True, "Entry_Price": entry_price,
                "Portfolio_Entry_Price": portfolio_entry_price})
    for days in (HOLD_20D, HOLD_40D):
        end_pos = entry_market_pos + days - 1
        if end_pos >= len(open_dates):
            continue
        end_date = open_dates[end_pos]
        path = daily[daily["trade_date"].astype(str).between(entry_date, end_date)].sort_values("trade_date")
        if path.empty:
            continue
        exit_price = float(path.iloc[-1]["close"]) * (1 - config["sell_slippage_pct"] / 100.0) * (1 - sell_cost)
        out.update({
            f"Outcome_{days}D_End_Date": end_date, f"Has_{days}D_Future": True,
            f"Return_{days}D_pct": (exit_price / entry_price - 1.0) * 100.0,
            f"MFE_{days}D_pct": (float(path["high"].max()) / entry_price - 1.0) * 100.0,
            f"MAE_{days}D_pct": (float(path["low"].min()) / entry_price - 1.0) * 100.0,
        })
        if days == HOLD_40D:
            future_trade = daily[
                daily["trade_date"].astype(str).ge(end_date)
            ].sort_values("trade_date")
            if not future_trade.empty:
                actual_exit_date = str(future_trade.iloc[0]["trade_date"])
                portfolio_path = daily[
                    daily["trade_date"].astype(str).between(entry_date, actual_exit_date)
                ].sort_values("trade_date")
                out.update(build_portfolio_exit_plans(
                    portfolio_path, entry_date, portfolio_entry_price, open_pos, config))
    if out["Tradable"] and not out["Has_40D_Future"]:
        out["Untradable_Reason"] = "可买但未来不足40个市场交易日"
    return out


def build_event(stock: pd.Series, membership: dict[str, str], weekly: pd.DataFrame,
                position: int, daily: pd.DataFrame, daily_basic: pd.DataFrame,
                open_dates: list[str], open_pos: dict[str, int], config: dict[str, Any]) -> dict[str, Any] | None:
    row = weekly.iloc[position]
    signal_date = str(row["trade_date"])
    if not (config["signal_start"] <= signal_date <= config["signal_end"]):
        return None
    snapshot = market_snapshot(daily_basic, signal_date)
    passed, reason = signal_filter(snapshot, config["min_price"], config["min_mv"])
    if not passed:
        config["rejects"][reason] = config["rejects"].get(reason, 0) + 1
        return None
    wfeat = weekly_features(weekly, position)
    touched, zone = to_bool(wfeat["Recent_3W_Touched_25"]), to_bool(wfeat["Cross_In_20_35"])
    board = sample_board(stock)
    event = {
        "ts_code": str(stock["ts_code"]), "name": str(stock["name"]), "Sample_Board": board,
        "SW_L1": membership["l1"], "SW_L2": membership["l2"], "SW_L3": membership["l3"],
        "Signal_Date": signal_date, "Weekly_Close": float(row["close"]), **snapshot, **wfeat,
        "Bottom_Reset_Core": touched and zone, "Bottom_Reset_Quadrant": reset_quadrant(touched, zone),
        "Recent_3W_Touched_25_Num": float(touched), "Cross_In_20_35_Num": float(zone),
        "Log_Raw_Price": math.log1p(max(finite_num(snapshot["Raw_Close"]), 0.0)),
        "Log_Circ_MV": math.log1p(max(finite_num(snapshot["Circ_MV_Billion"]), 0.0)),
        "Board_Main": float(board == "主板"), "Board_ChiNext": float(board == "创业板"),
        "Board_STAR": float(board == "科创板"),
    }
    event.update(daily_features_at_signal(daily, signal_date))
    event.update(direct_outcomes(daily, signal_date, str(stock["ts_code"]), open_dates, open_pos, config))
    return event


def analyze_stock(stock: pd.Series, periods: list[dict[str, str]], daily_raw: pd.DataFrame,
                  daily_basic: pd.DataFrame, week_last_map: dict[pd.Timestamp, str],
                  open_dates: list[str], open_pos: dict[str, int], config: dict[str, Any]) -> list[dict[str, Any]]:
    weekly = build_complete_weekly(daily_raw, week_last_map)
    daily = add_daily_features(daily_raw)
    if len(weekly) < INDICATOR_WARMUP_WEEKS:
        config["rejects"]["周线不足"] = config["rejects"].get("周线不足", 0) + 1
        return []
    records = []
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
        event = build_event(stock, membership, weekly, position, daily, daily_basic, open_dates, open_pos, config)
        if event is not None:
            records.append(event)
    return records


def add_cross_section_features(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["Week_Signal_Count"] = frame.groupby("Signal_Date")["ts_code"].transform("size").astype(float)
    frame["Industry_Signal_Count"] = frame.groupby(["Signal_Date", "SW_L1"])["ts_code"].transform("size").astype(float)
    return frame


def rank_core_candidates(events: pd.DataFrame) -> pd.DataFrame:
    core = events[
        events["Bottom_Reset_Core"].map(to_bool)
        & events["Tradable"].map(to_bool)
        & events["Has_40D_Future"].map(to_bool)
    ].copy()
    core["Daily_SKDJ_Weekly_Rank"] = pd.to_numeric(
        core["Daily_SKDJ_Level_At_Cross"], errors="coerce").groupby(
        core["Signal_Date"]).rank(method="first", ascending=False, na_option="bottom")
    core["Core_Tradable_Count"] = core.groupby("Signal_Date")["ts_code"].transform("size")
    return core.sort_values(["Signal_Date", "Daily_SKDJ_Weekly_Rank", "ts_code"]).reset_index(drop=True)


def build_mark_prices(histories: dict[str, pd.DataFrame],
                      open_dates: list[str]) -> dict[str, dict[str, float]]:
    calendar = pd.Index(open_dates, dtype=str)
    result: dict[str, dict[str, float]] = {}
    for code, history in histories.items():
        if history.empty:
            result[code] = {}
            continue
        clean = history.drop_duplicates("trade_date", keep="last").copy()
        clean["trade_date"] = clean["trade_date"].astype(str)
        series = pd.to_numeric(clean.set_index("trade_date")["close"], errors="coerce").sort_index()
        result[code] = series.reindex(calendar).ffill().to_dict()
    return result


def fee(amount: float, rate_pct: float, minimum: float = 0.0) -> float:
    if amount <= 0:
        return 0.0
    return max(minimum, amount * rate_pct / 100.0)


def buy_fee(amount: float, config: dict[str, Any]) -> float:
    return fee(amount, config["commission_pct"], 5.0) + fee(amount, config["transfer_fee_pct"])


def sell_fee(amount: float, config: dict[str, Any]) -> float:
    return (fee(amount, config["commission_pct"], 5.0)
            + fee(amount, config["transfer_fee_pct"])
            + fee(amount, config["stamp_duty_pct"]))


def affordable_units(budget: float, price: float, config: dict[str, Any]) -> float:
    if budget <= 0 or not math.isfinite(price) or price <= 0:
        return 0.0
    low, high = 0.0, budget / price
    for _ in range(48):
        middle = (low + high) / 2.0
        amount = middle * price
        if amount + buy_fee(amount, config) <= budget:
            low = middle
        else:
            high = middle
    return low


def mark_price(mark_prices: dict[str, dict[str, float]], code: str, trade_date: str) -> float:
    return finite_num(mark_prices.get(code, {}).get(trade_date))


def empty_portfolio_summary(strategy_code: str) -> dict[str, Any]:
    topk, exit_method, label = STRATEGIES[strategy_code]
    return {
        "策略代码": strategy_code, "策略": label, "TopK": topk, "退出方法": exit_method,
        "初始资金": INITIAL_CAPITAL, "实际买入": 0, "已完成交易": 0,
        "期末权益": INITIAL_CAPITAL, "总收益率(%)": 0.0, "最大回撤(%)": 0.0,
    }


def simulate_portfolio(core: pd.DataFrame, mark_prices: dict[str, dict[str, float]],
                       open_dates: list[str], config: dict[str, Any],
                       strategy_code: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    topk, exit_method, label = STRATEGIES[strategy_code]
    work = core[pd.to_numeric(core["Daily_SKDJ_Weekly_Rank"], errors="coerce").le(topk)].copy()
    exit_date_col = f"{exit_method}_Exit_Date"
    exit_price_col = f"{exit_method}_Exit_Price"
    exit_session_col = f"{exit_method}_Exit_Session"
    exit_reason_col = f"{exit_method}_Exit_Reason"
    exit_hold_col = f"{exit_method}_Hold_Market_Days"
    work = work[
        work["Entry_Date"].astype(str).ne("")
        & work[exit_date_col].astype(str).ne("")
        & pd.to_numeric(work["Portfolio_Entry_Price"], errors="coerce").gt(0)
        & pd.to_numeric(work[exit_price_col], errors="coerce").gt(0)
    ].sort_values(["Entry_Date", "Daily_SKDJ_Weekly_Rank", "ts_code"], kind="mergesort")
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), empty_portfolio_summary(strategy_code)
    entry_groups = {str(day): frame for day, frame in work.groupby("Entry_Date", sort=True)}
    days = [day for day in open_dates if config["signal_start"] <= day <= config["market_end"]]
    cash = INITIAL_CAPITAL
    active: dict[str, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []

    def execute_exit(code: str, trade_date: str) -> None:
        nonlocal cash
        trade = active.pop(code)
        gross = trade["Units"] * trade["Planned_Exit_Price"]
        fees = sell_fee(gross, config)
        proceeds = gross - fees
        cash += proceeds
        pnl = proceeds - trade["Entry_Total"]
        trade.update({
            "Exit_Date": trade_date, "Exit_Gross": gross, "Sell_Fees": fees,
            "Exit_Proceeds": proceeds, "PnL": pnl,
            "Net_Return_pct": pnl / trade["Entry_Total"] * 100.0,
        })

    for trade_date in days:
        opening_exits = [code for code, trade in active.items()
                         if trade["Planned_Exit_Date"] == trade_date
                         and trade["Planned_Exit_Session"] == "OPEN"]
        for code in opening_exits:
            execute_exit(code, trade_date)

        for _, row in entry_groups.get(trade_date, pd.DataFrame()).iterrows():
            code = str(row["ts_code"])
            reason = ""
            if code in active:
                reason = "同一股票已持仓"
            elif len(active) >= MAX_POSITIONS:
                reason = "3个仓位已满"
            price = finite_num(row["Portfolio_Entry_Price"])
            budget = min(POSITION_BUDGET, cash)
            units = affordable_units(budget, price, config)
            amount = units * price
            fees = buy_fee(amount, config) if units > 0 else 0.0
            if not reason and (units <= 0 or amount + fees > cash + 1e-6):
                reason = "可用现金不足"
            orders.append({
                "策略代码": strategy_code, "策略": label, "Signal_Date": row["Signal_Date"],
                "Entry_Date": trade_date, "ts_code": code, "name": row.get("name", ""),
                "日线SKDJ位置": row["Daily_SKDJ_Level_At_Cross"],
                "同周排名": row["Daily_SKDJ_Weekly_Rank"], "Action": "未买入" if reason else "已买入",
                "Reason": reason or "按日线SKDJ同周排名买入",
            })
            if reason:
                continue
            total = amount + fees
            cash -= total
            trade = {
                "策略代码": strategy_code, "策略": label, "Signal_Date": row["Signal_Date"],
                "Entry_Date": trade_date, "ts_code": code, "name": row.get("name", ""),
                "Daily_SKDJ_Level": finite_num(row["Daily_SKDJ_Level_At_Cross"]),
                "Daily_SKDJ_Weekly_Rank": finite_num(row["Daily_SKDJ_Weekly_Rank"]),
                "Units": units, "Entry_Price": price, "Entry_Amount": amount,
                "Buy_Fees": fees, "Entry_Total": total,
                "Planned_Exit_Date": str(row[exit_date_col]),
                "Planned_Exit_Price": finite_num(row[exit_price_col]),
                "Planned_Exit_Session": str(row[exit_session_col]),
                "Exit_Reason": str(row[exit_reason_col]),
                "Hold_Market_Days": finite_num(row[exit_hold_col]),
                "Exit_Date": "", "PnL": np.nan, "Net_Return_pct": np.nan,
            }
            active[code] = trade
            ledger.append(trade)

        closing_exits = [code for code, trade in active.items()
                         if trade["Planned_Exit_Date"] == trade_date
                         and trade["Planned_Exit_Session"] == "CLOSE"]
        for code in closing_exits:
            execute_exit(code, trade_date)

        market_value = 0.0
        for code, trade in active.items():
            mark = mark_price(mark_prices, code, trade_date)
            market_value += trade["Units"] * (mark if math.isfinite(mark) else trade["Entry_Price"])
        equity = cash + market_value
        curves.append({
            "Trade_Date": trade_date, "策略代码": strategy_code, "策略": label,
            "Cash": cash, "Market_Value": market_value, "Equity": equity,
            "Positions": len(active),
            "Capital_Exposure_pct": market_value / equity * 100.0 if equity > 0 else np.nan,
        })

    curve = pd.DataFrame(curves)
    ledger_frame = pd.DataFrame(ledger)
    orders_frame = pd.DataFrame(orders)
    if curve.empty:
        return curve, ledger_frame, orders_frame, empty_portfolio_summary(strategy_code)
    running_peak = curve["Equity"].cummax().clip(lower=INITIAL_CAPITAL)
    curve["Drawdown_pct"] = (curve["Equity"] / running_peak - 1.0) * 100.0
    final_equity = float(curve.iloc[-1]["Equity"])
    completed = ledger_frame[pd.to_numeric(ledger_frame.get("PnL"), errors="coerce").notna()] \
        if not ledger_frame.empty else pd.DataFrame()
    years = max(len(curve) / 252.0, 1 / 252.0)
    reason_counts = orders_frame.get("Reason", pd.Series(dtype=str)).value_counts()
    summary = {
        "策略代码": strategy_code, "策略": label, "TopK": topk, "退出方法": exit_method,
        "初始资金": INITIAL_CAPITAL, "实际买入": len(ledger_frame), "已完成交易": len(completed),
        "期末未平仓": len(active), "期末权益": final_equity,
        "总收益率(%)": (final_equity / INITIAL_CAPITAL - 1.0) * 100.0,
        "年化收益率(%)": ((final_equity / INITIAL_CAPITAL) ** (1.0 / years) - 1.0) * 100.0,
        "最大回撤(%)": pd.to_numeric(curve["Drawdown_pct"], errors="coerce").min(),
        "交易胜率(%)": completed["PnL"].gt(0).mean() * 100.0 if len(completed) else np.nan,
        "平均单笔收益(%)": completed["Net_Return_pct"].mean() if len(completed) else np.nan,
        "单笔收益中位数(%)": completed["Net_Return_pct"].median() if len(completed) else np.nan,
        "平均持有市场日": completed["Hold_Market_Days"].mean() if len(completed) else np.nan,
        "空仓交易日": int(curve["Positions"].eq(0).sum()),
        "空仓率(%)": curve["Positions"].eq(0).mean() * 100.0,
        "平均持仓数": curve["Positions"].mean(),
        "平均资金暴露(%)": curve["Capital_Exposure_pct"].mean(),
        "仓位满错过": int(reason_counts.get("3个仓位已满", 0)),
        "重复持仓错过": int(reason_counts.get("同一股票已持仓", 0)),
    }
    return curve, ledger_frame, orders_frame, summary


def annual_portfolio_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if curves.empty:
        return pd.DataFrame()
    for strategy, frame in curves.groupby("策略代码", sort=False):
        ordered = frame.sort_values("Trade_Date").copy()
        ordered["Year"] = ordered["Trade_Date"].astype(str).str[:4]
        prior_equity = INITIAL_CAPITAL
        for year, group in ordered.groupby("Year", sort=True):
            equity = pd.to_numeric(group["Equity"], errors="coerce")
            peak = pd.concat([pd.Series([prior_equity]), equity], ignore_index=True).cummax().iloc[1:]
            drawdown = (equity.to_numpy() / peak.to_numpy() - 1.0) * 100.0
            end_equity = float(equity.iloc[-1])
            rows.append({
                "策略代码": strategy, "策略": group.iloc[0]["策略"], "年份": year,
                "年初权益": prior_equity, "年末权益": end_equity,
                "年度收益率(%)": (end_equity / prior_equity - 1.0) * 100.0,
                "年度最大回撤(%)": float(np.nanmin(drawdown)),
                "交易日": len(group), "空仓率(%)": group["Positions"].eq(0).mean() * 100.0,
                "平均持仓数": group["Positions"].mean(),
            })
            prior_equity = end_equity
    return pd.DataFrame(rows)


def run_portfolios(core: pd.DataFrame, mark_prices: dict[str, dict[str, float]],
                   open_dates: list[str], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries, curves, ledgers, orders = [], [], [], []
    for strategy_code in STRATEGIES:
        curve, ledger, order, summary = simulate_portfolio(
            core, mark_prices, open_dates, config, strategy_code)
        summaries.append(summary)
        if not curve.empty:
            curves.append(curve)
        if not ledger.empty:
            ledgers.append(ledger)
        if not order.empty:
            orders.append(order)
    curve_frame = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return (
        pd.DataFrame(summaries), annual_portfolio_summary(curve_frame), curve_frame,
        pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(),
        pd.concat(orders, ignore_index=True) if orders else pd.DataFrame(),
    )


def pool_calendar(open_dates: list[str], start: str, end: str, events: pd.DataFrame) -> pd.DataFrame:
    days = pd.DataFrame({"trade_date": [day for day in open_dates if start <= day <= end]})
    days["dt"] = pd.to_datetime(days["trade_date"])
    days["week"] = days["dt"].dt.to_period("W-FRI")
    weeks = days.groupby("week")["trade_date"].max().rename("Week_Last_Trade_Date").reset_index(drop=True).to_frame()
    counts = events.groupby("Signal_Date").size()
    core_counts = events[events["Bottom_Reset_Core"].map(to_bool)].groupby("Signal_Date").size()
    weeks["All_Cross_Count"] = weeks["Week_Last_Trade_Date"].map(counts).fillna(0).astype(int)
    weeks["Original_Core_Count"] = weeks["Week_Last_Trade_Date"].map(core_counts).fillna(0).astype(int)
    weeks["All_Empty"] = weeks["All_Cross_Count"].eq(0)
    weeks["Original_Core_Empty"] = weeks["Original_Core_Count"].eq(0)
    return weeks


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title="周线SKDJ TopK组合回测 V3.6", layout="wide")
    st.title(TITLE)
    st.caption("本版不再研究市场过滤和复杂评分，直接检验V3.5发现能否转化为30万元、最多3仓的真实资金曲线。")
    with st.expander("冻结规则与执行顺序", expanded=True):
        st.markdown(f"""
- **核心池硬条件**：完整周线SKDJ金叉；最近{RESET_LOOKBACK_WEEKS}个完整周K或D曾≤{SKDJ_BOTTOM:.0f}；金叉位置>{CROSS_ZONE_LOW:.0f}且≤{CROSS_ZONE_HIGH:.0f}。
- **排序**：仅按信号日的日线SKDJ `(K+D)/2` 从高到低，分别验证Top2和Top3；不拼接其他评分。
- **买入**：周线收盘确认后下一市场交易日开盘，买入滑点后成交；主板一字板不买。
- **组合**：初始资金{INITIAL_CAPITAL:,.0f}元，最多{MAX_POSITIONS}仓，每次预算不超过{POSITION_BUDGET:,.0f}元；同股已持仓不重复买。
- **当日顺序**：开盘先卖出到期仓位，再按排名买入；固定40日退出在到期日收盘执行。
- **高位死叉**：当日日线SKDJ死叉，并且当日或前一日SKDJ位置≥{HIGH_DEATH_ZONE:.0f}，下一只可交易日开盘卖出；40日内没有触发则到期退出。
- **对照组**：固定持有40个市场交易日，到期收盘卖出。
- **不使用**：市场强弱门槛、机器学习、线性综合评分、任意位置日线SKDJ死叉。
- **限制**：Top2/Top3来自V3.5同一历史段，本版验证组合可执行性，不能当作新的独立样本外证明。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("信号开始", date(2023, 6, 5), key="v36_start")
        signal_end_date = st.date_input("信号截止", date(2026, 6, 5), key="v36_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v36_market_end")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v36_pause")
        use_cache = st.checkbox("复用逐股票缓存", True, key="v36_cache")
        st.divider()
        commission_pct = st.number_input("佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f")
        stamp_duty_pct = st.number_input("卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f")
        transfer_fee_pct = st.number_input("过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f")
        if st.button("清除本程序行情缓存", key="v36_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("缓存已清除")
    token = st.text_input("Tushare Token", type="password", key="v36_token")
    session_key = "weekly_skdj_portfolio_v36_zip"
    if not token:
        st.info("请输入Tushare Token；V3.5相同日期范围的逐股票缓存可以直接复用。")
        return
    if not st.button("开始V3.6真实组合回测", type="primary", key="v36_run"):
        if session_key in st.session_state:
            st.download_button(
                "下载上一次结果ZIP", st.session_state[session_key],
                file_name="weekly_skdj_topk_portfolio_v3_6_all_results.zip",
                mime="application/zip", on_click="ignore",
            )
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
        "min_price": 10.0, "min_mv": 100.0, "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct), "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct), "rejects": rejects,
    }
    try:
        with st.spinner("加载交易日历与历史科技股池..."):
            open_dates = load_trade_calendar(preload, market_end)
            extended_end = (market_end_date + timedelta(days=7)).strftime("%Y%m%d")
            full_open_dates = load_trade_calendar(preload, extended_end)
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
    population = stocks.groupby("Sample_Board").size().reindex(BOARDS, fill_value=0).rename(
        "股票数").reset_index()
    open_pos = {day: position for position, day in enumerate(open_dates)}
    events: list[dict[str, Any]] = []
    mark_histories: dict[str, pd.DataFrame] = {}
    cache_hits = data_failures = 0
    progress, status = st.progress(0.0), st.empty()
    for number, stock in stocks.iterrows():
        code = str(stock["ts_code"])
        progress.progress((number + 1) / max(len(stocks), 1), text=f"{number + 1}/{len(stocks)} {code}")
        status.caption(f"全部金叉 {len(events)}；缓存 {cache_hits}；失败 {data_failures}")
        daily, daily_basic, cache_hit = fetch_stock_history(
            code, preload, market_end, bool(use_cache), float(pause))
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        stock_events = analyze_stock(
            stock, period_index.get(code, []), daily, daily_basic,
            week_last_map, open_dates, open_pos, config,
        )
        events.extend(stock_events)
        if any(to_bool(item.get("Bottom_Reset_Core")) and to_bool(item.get("Tradable"))
               for item in stock_events):
            mark_histories[code] = daily[["trade_date", "close"]].copy()
    progress.empty()
    status.empty()
    if not events:
        st.error("研究区间没有生成符合历史科技池、价格和市值条件的完整周线SKDJ金叉。")
        return
    try:
        with st.spinner("计算Top2/Top3排名、退出计划与四条真实资金曲线..."):
            event_frame = add_cross_section_features(
                pd.DataFrame(events).sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True))
            core = rank_core_candidates(event_frame)
            mark_prices = build_mark_prices(mark_histories, open_dates)
            portfolio_summary, annual_summary, curves, ledger, orders = run_portfolios(
                core, mark_prices, open_dates, config)
            calendar = pool_calendar(open_dates, signal_start, signal_end, event_frame)
    except Exception as exc:
        st.exception(exc)
        return
    exit_reasons = (ledger.groupby(["策略代码", "策略", "Exit_Reason"], as_index=False).agg(
        交易数=("ts_code", "size"), 平均净收益=("Net_Return_pct", "mean"),
        收益中位数=("Net_Return_pct", "median"), 胜率=("PnL", lambda values: values.gt(0).mean() * 100.0),
    ) if not ledger.empty else pd.DataFrame())
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "信号开始": signal_start, "信号截止": signal_end,
        "观察截止": market_end, "全部周线SKDJ金叉": len(event_frame), "核心可交易成熟事件": len(core),
        "核心不同股票": core["ts_code"].nunique(), "自然周": len(calendar),
        "核心有信号周": core["Signal_Date"].nunique(),
        "核心平均每自然周": len(core) / len(calendar) if len(calendar) else np.nan,
        "核心平均每信号周": len(core) / max(core["Signal_Date"].nunique(), 1),
        "核心空窗周": int(calendar["Original_Core_Empty"].sum()),
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    metadata = pd.DataFrame([
        ("核心候选", "完整周线SKDJ金叉，近3个完整周K或D最低值≤25，金叉当周(K+D)/2>20且≤35"),
        ("SKDJ参数", f"N={SKDJ_N},M={SKDJ_M}，冻结不寻优"),
        ("股票池", "申万2021历史科技池；主板/创业板/科创板；排除北交所"),
        ("价格市值", "信号日原始收盘价≥10元；历史流通市值≥100亿元"),
        ("排序", "同周核心可交易候选按信号日日线SKDJ位置从高到低；分别验证Top2与Top3"),
        ("组合", f"初始{INITIAL_CAPITAL:.0f}元；最多{MAX_POSITIONS}仓；单次预算不超过{POSITION_BUDGET:.0f}元"),
        ("买入", "完整周线收盘确认，下一市场交易日开盘；主板一字板不买"),
        ("高位死叉", f"日线死叉且当日或前一日(K+D)/2≥{HIGH_DEATH_ZONE:.0f}；下一可交易日开盘卖出"),
        ("固定40日", "买入日计第1个市场交易日；第40个市场交易日收盘卖出"),
        ("同日顺序", "开盘卖出→按排名买入→收盘到期卖出→收盘价估值"),
        ("成本", "买卖滑点、佣金最低5元、双边佣金与过户费、卖出印花税全部计入"),
        ("份额口径", "使用前复权价格的连续资金份额，不做100股整手取整；用于保证跨除权期收益连续"),
        ("市场门槛", "V3.5未验证通过，本版完全不使用"),
        ("独立性限制", "Top2/Top3来自V3.5同一历史段；本版验证组合可执行性，不是独立样本外证明"),
    ], columns=["项目", "值"])
    files = {
        "01_run_summary_v3_6.csv": run_summary,
        "02_portfolio_comparison_v3_6.csv": portfolio_summary,
        "03_portfolio_yearly_v3_6.csv": annual_summary,
        "04_daily_equity_curve_v3_6.csv": curves,
        "05_trade_ledger_v3_6.csv": ledger,
        "06_order_and_skip_audit_v3_6.csv": orders,
        "07_exit_reason_summary_v3_6.csv": exit_reasons,
        "08_core_candidate_rank_and_exit_plan_v3_6.csv": core,
        "09_weekly_pool_calendar_v3_6.csv": calendar,
        "10_all_weekly_skdj_events_v3_6.csv": event_frame,
        "11_full_tech_universe_v3_6.csv": stocks,
        "12_board_population_v3_6.csv": population,
        "13_rejection_audit_v3_6.csv": pd.DataFrame(
            [{"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]),
        "14_api_errors_v3_6.csv": pd.DataFrame({"错误": API_ERRORS}),
        "15_metadata_v3_6.csv": metadata,
    }
    result_zip = make_zip(files)
    st.session_state[session_key] = result_zip
    st.success(f"完成：核心可交易成熟事件{len(core)}个，已生成4条真实组合资金曲线。")
    st.subheader("四种组合结果")
    st.dataframe(portfolio_summary, use_container_width=True, hide_index=True)
    st.subheader("年度稳定性")
    st.dataframe(annual_summary, use_container_width=True, hide_index=True)
    st.subheader("退出原因")
    st.dataframe(exit_reasons, use_container_width=True, hide_index=True)
    st.download_button(
        "下载V3.6全部结果ZIP", result_zip,
        file_name="weekly_skdj_topk_portfolio_v3_6_all_results.zip",
        mime="application/zip", type="primary", key="v36_download", on_click="ignore",
    )
    st.info("先看02比较Top2/Top3的总收益与最大回撤，再看03是否跨年稳定；若高位死叉不能显著改善回撤，就不能因为平均收益较高而进入实盘。")


if __name__ == "__main__":
    main()
