# -*- coding: utf-8 -*-
"""
科技股周线SKDJ周级环境与同周排序审计 V3.9

目的：
1. 硬候选池冻结为：完整周线SKDJ金叉位置20~35，并排除下跌趋势。
2. 最近3周触及25为第一梯队；未触及25为第二梯队，仅在第一梯队不足时补位。
3. 先检验同周候选广度、第一梯队数量和行业覆盖能否判断“本周是否值得买”。
4. 梯队内部只比较板块共振、周波动收缩及二者等权，不再调整旧量价评分权重。
5. Top2为主观察口径，同时保留Top1/Top3，并与遵守相同周级门槛和梯队顺序的随机选择比较。
6. 所有事件统一在周线确认后的下一市场交易日开盘买入，以20/40日固定终点、MFE和MAE判卷。

注意：分层规则来自同一三年历史，本版是复核审计，不是独立样本外证明。
周线只使用完整周，所有特征只使用信号日及以前数据，未来结果只用于判卷。

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
TITLE = "科技股周线SKDJ周级环境与同周排序审计 V3.9"
VERSION = "V3.9-WEEKLY-SKDJ-WEEK-STATE-AND-RANK-AUDIT"
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
RANDOM_SEED = 20260813
RANDOM_RUNS = 300
HISTORY_WINDOW_WEEKS = 52
MIN_HISTORY_WEEKS = 12

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


def direct_outcomes(daily: pd.DataFrame, signal_date: str, ts_code: str,
                    open_dates: list[str], open_pos: dict[str, int], config: dict[str, Any]) -> dict[str, Any]:
    out = {
        "Tradable": False, "Untradable_Reason": "", "Entry_Date": "", "Entry_Price": np.nan,
        "Outcome_20D_End_Date": "", "Outcome_40D_End_Date": "", "Has_20D_Future": False,
        "Has_40D_Future": False, "Return_20D_pct": np.nan, "Return_40D_pct": np.nan,
        "MFE_20D_pct": np.nan, "MAE_20D_pct": np.nan, "MFE_40D_pct": np.nan, "MAE_40D_pct": np.nan,
    }
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
    entry_price = float(first["open"]) * (1 + config["buy_slippage_pct"] / 100.0) * (1 + buy_cost)
    out.update({"Tradable": True, "Entry_Price": entry_price})
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
    if len(weekly) < INDICATOR_WARMUP_WEEKS:
        config["rejects"]["周线不足"] = config["rejects"].get("周线不足", 0) + 1
        return []
    daily = add_daily_features(daily_raw)
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
    frame["Signal_Date_dt"] = pd.to_datetime(frame["Signal_Date"], format="%Y%m%d", errors="coerce")
    frame["Outcome_40D_End_dt"] = pd.to_datetime(frame["Outcome_40D_End_Date"], format="%Y%m%d", errors="coerce")
    frame["Half_Year"] = frame["Signal_Date_dt"].dt.year.astype("Int64").astype(str) + "H" + np.where(frame["Signal_Date_dt"].dt.month.le(6), "1", "2")
    weekly_bias = pd.to_numeric(frame["Weekly_MA20_Bias_pct"], errors="coerce")
    weekly_return = pd.to_numeric(frame["Weekly_Return_12W_pct"], errors="coerce")
    daily_bias = pd.to_numeric(frame["Daily_MA60_Bias_pct"], errors="coerce")
    downtrend = weekly_bias.lt(0) & weekly_return.lt(0) & daily_bias.lt(0)
    uptrend = weekly_bias.gt(0) & weekly_return.gt(0) & daily_bias.gt(0)
    frame["Trend_State"] = np.select(
        [uptrend, downtrend], ["上涨趋势", "下跌趋势"], default="震荡或转换")
    frame["Downtrend_Hard_Excluded"] = downtrend
    frame["Tiered_Hard_Pool"] = (
        frame["Cross_In_20_35"].map(to_bool)
        & ~downtrend
        & frame["Tradable"].map(to_bool)
        & frame["Has_40D_Future"].map(to_bool)
    )
    frame["Tier_Number"] = np.where(
        frame["Recent_3W_Touched_25"].map(to_bool), 1, 2).astype(int)
    frame["Tier_Label"] = np.where(
        frame["Tier_Number"].eq(1), "第一梯队_近3周触及25", "第二梯队_未触及25补位")
    return frame


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


def build_tiered_pool(events: pd.DataFrame) -> pd.DataFrame:
    pool = events[events["Tiered_Hard_Pool"].map(to_bool)].copy()
    if pool.empty:
        return pool
    group_keys = [pool["Signal_Date"], pool["Tier_Number"]]
    resonance = pd.to_numeric(pool["Industry_Signal_Count"], errors="coerce")
    contraction = pd.to_numeric(pool["Weekly_Contraction_4_12"], errors="coerce")
    pool["Score_Industry_Resonance"] = resonance.groupby(group_keys).rank(
        pct=True, method="average", ascending=True).fillna(0.0)
    pool["Score_Weekly_Contraction"] = contraction.groupby(group_keys).rank(
        pct=True, method="average", ascending=False).fillna(0.0)
    pool["Score_Resonance_Contraction"] = pool[
        ["Score_Industry_Resonance", "Score_Weekly_Contraction"]
    ].mean(axis=1)
    pool["Hard_Pool_Weekly_Count"] = pool.groupby("Signal_Date")["ts_code"].transform("size")
    tier1_counts = pool[pool["Tier_Number"].eq(1)].groupby("Signal_Date").size()
    pool["Tier1_Weekly_Count"] = pool["Signal_Date"].map(tier1_counts).fillna(0).astype(int)
    return pool.sort_values(["Signal_Date", "Tier_Number", "ts_code"]).reset_index(drop=True)


RANK_METHODS = {
    "板块共振": "Score_Industry_Resonance",
    "周波动收缩": "Score_Weekly_Contraction",
    "板块共振+周波动收缩等权": "Score_Resonance_Contraction",
}


WEEK_GATES = (
    "全部硬池周",
    "第一梯队存在",
    "硬池候选至少2只",
    "历史周环境分≥50",
    "历史周环境分≥70",
)


def trailing_percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    for position, current in enumerate(numeric):
        history = numeric.iloc[max(0, position - HISTORY_WINDOW_WEEKS):position]
        if len(history) >= MIN_HISTORY_WEEKS:
            result.iloc[position] = float(history.le(current).mean() * 100.0)
    return result


def build_week_state(calendar: pd.DataFrame, events: pd.DataFrame,
                     pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = calendar[["Week_Last_Trade_Date"]].copy()
    state["Week_Last_Trade_Date"] = state["Week_Last_Trade_Date"].astype(str)
    all_counts = events.groupby(events["Signal_Date"].astype(str)).size()
    hard_counts = pool.groupby(pool["Signal_Date"].astype(str)).size()
    tier1_counts = pool[pool["Tier_Number"].eq(1)].groupby(
        pool.loc[pool["Tier_Number"].eq(1), "Signal_Date"].astype(str)).size()
    industry_counts = pool.groupby(pool["Signal_Date"].astype(str))["SW_L1"].nunique()
    state["All_Cross_Count"] = state["Week_Last_Trade_Date"].map(all_counts).fillna(0).astype(int)
    state["Hard_Pool_Count"] = state["Week_Last_Trade_Date"].map(hard_counts).fillna(0).astype(int)
    state["Tier1_Count"] = state["Week_Last_Trade_Date"].map(tier1_counts).fillna(0).astype(int)
    state["Tier2_Count"] = state["Hard_Pool_Count"] - state["Tier1_Count"]
    state["Industry_Breadth"] = state["Week_Last_Trade_Date"].map(industry_counts).fillna(0).astype(int)
    state["Tier1_Ratio_pct"] = np.where(
        state["Hard_Pool_Count"].gt(0), state["Tier1_Count"] / state["Hard_Pool_Count"] * 100.0, np.nan)
    for source, target in (
        ("All_Cross_Count", "All_Cross_History_Pct"),
        ("Hard_Pool_Count", "Hard_Pool_History_Pct"),
        ("Industry_Breadth", "Industry_Breadth_History_Pct"),
    ):
        state[target] = trailing_percentile(state[source])
    state["Week_State_Score"] = state[
        ["All_Cross_History_Pct", "Hard_Pool_History_Pct", "Industry_Breadth_History_Pct"]
    ].mean(axis=1, skipna=False)
    merge_columns = [
        "Week_Last_Trade_Date", "All_Cross_Count", "Hard_Pool_Count", "Tier1_Count",
        "Tier2_Count", "Industry_Breadth", "Tier1_Ratio_pct", "All_Cross_History_Pct",
        "Hard_Pool_History_Pct", "Industry_Breadth_History_Pct", "Week_State_Score",
    ]
    enriched = pool.merge(
        state[merge_columns], left_on=pool["Signal_Date"].astype(str),
        right_on="Week_Last_Trade_Date", how="left").drop(columns=["key_0", "Week_Last_Trade_Date"], errors="ignore")
    return state, enriched


def gate_mask(pool: pd.DataFrame, gate: str) -> pd.Series:
    if gate == "全部硬池周":
        return pd.Series(True, index=pool.index)
    if gate == "第一梯队存在":
        return pd.to_numeric(pool["Tier1_Count"], errors="coerce").ge(1)
    if gate == "硬池候选至少2只":
        return pd.to_numeric(pool["Hard_Pool_Count"], errors="coerce").ge(2)
    if gate == "历史周环境分≥50":
        return pd.to_numeric(pool["Week_State_Score"], errors="coerce").ge(50)
    if gate == "历史周环境分≥70":
        return pd.to_numeric(pool["Week_State_Score"], errors="coerce").ge(70)
    raise ValueError(f"未知周级门槛: {gate}")


def selected_event_stats(frame: pd.DataFrame, method: str, topk: int,
                         gate: str = "") -> dict[str, Any]:
    returns20 = pd.to_numeric(frame.get("Return_20D_pct"), errors="coerce").dropna()
    returns = pd.to_numeric(frame.get("Return_40D_pct"), errors="coerce").dropna()
    mfe = pd.to_numeric(frame.get("MFE_40D_pct"), errors="coerce").dropna()
    mae = pd.to_numeric(frame.get("MAE_40D_pct"), errors="coerce").dropna()
    tier2 = frame.get("Tier_Number", pd.Series(dtype=float)).eq(2)
    return {
        "周级门槛": gate, "排序方法": method, "TopK": topk, "选择事件": len(frame),
        "覆盖信号周": frame["Signal_Date"].nunique() if len(frame) else 0,
        "第二梯队补位数": int(tier2.sum()) if len(frame) else 0,
        "第二梯队占比(%)": tier2.mean() * 100.0 if len(frame) else np.nan,
        "20日平均收益(%)": returns20.mean(), "20日收益中位数(%)": returns20.median(),
        "40日平均收益(%)": returns.mean(), "40日收益中位数(%)": returns.median(),
        "正收益比例(%)": returns.gt(0).mean() * 100.0 if len(returns) else np.nan,
        "收益≥10%比例(%)": returns.ge(10).mean() * 100.0 if len(returns) else np.nan,
        "收益≥20%比例(%)": returns.ge(20).mean() * 100.0 if len(returns) else np.nan,
        "亏损≤-10%比例(%)": returns.le(-10).mean() * 100.0 if len(returns) else np.nan,
        "亏损≤-20%比例(%)": returns.le(-20).mean() * 100.0 if len(returns) else np.nan,
        "MFE中位数(%)": mfe.median(), "MFE≥10%比例(%)": mfe.ge(10).mean() * 100.0 if len(mfe) else np.nan,
        "MFE≥20%比例(%)": mfe.ge(20).mean() * 100.0 if len(mfe) else np.nan,
        "平均MAE(%)": mae.mean(), "MAE≤-15%比例(%)": mae.le(-15).mean() * 100.0 if len(mae) else np.nan,
    }


def select_tiered_topk(pool: pd.DataFrame, score_column: str, topk: int) -> pd.DataFrame:
    if pool.empty:
        return pool.copy()
    ranked = pool.sort_values(
        ["Signal_Date", "Tier_Number", score_column, "ts_code"],
        ascending=[True, True, False, True], kind="mergesort")
    return ranked.groupby("Signal_Date", sort=False).head(topk).copy()


def deterministic_rank_audit(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    yearly: list[dict[str, Any]] = []
    half_yearly: list[dict[str, Any]] = []
    for gate in WEEK_GATES:
        gated = pool[gate_mask(pool, gate)].copy()
        for method, score_column in RANK_METHODS.items():
            for topk in (1, 2, 3):
                selected = select_tiered_topk(gated, score_column, topk)
                selected.insert(0, "周级门槛", gate)
                selected.insert(1, "排序方法", method)
                selected.insert(2, "TopK", topk)
                selected.insert(3, "排序分数", pd.to_numeric(selected[score_column], errors="coerce"))
                summaries.append(selected_event_stats(selected, method, topk, gate))
                details.append(selected)
                for year, group in selected.groupby(selected["Signal_Date"].astype(str).str[:4], sort=True):
                    row = selected_event_stats(group, method, topk, gate)
                    row["年份"] = year
                    yearly.append(row)
                for period, group in selected.groupby("Half_Year", sort=True):
                    row = selected_event_stats(group, method, topk, gate)
                    row["半年"] = period
                    half_yearly.append(row)
    return (
        pd.DataFrame(summaries),
        pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
        pd.DataFrame(yearly), pd.DataFrame(half_yearly),
    )


def random_tier_audit(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame()
    for gate_number, gate in enumerate(WEEK_GATES):
        gated = pool[gate_mask(pool, gate)].copy()
        for run in range(RANDOM_RUNS):
            rng = np.random.default_rng(RANDOM_SEED + gate_number * 10_000 + run)
            work = gated.copy()
            work["Random_Score"] = rng.random(len(work))
            for topk in (1, 2, 3):
                selected = select_tiered_topk(work, "Random_Score", topk)
                row = selected_event_stats(selected, "分层随机", topk, gate)
                row["随机轮次"] = run
                rows.append(row)
    detail = pd.DataFrame(rows)
    summary_rows = []
    metrics = [
        "40日平均收益(%)", "40日收益中位数(%)", "正收益比例(%)",
        "亏损≤-10%比例(%)", "亏损≤-20%比例(%)", "平均MAE(%)",
    ]
    for (gate, topk), group in detail.groupby(["周级门槛", "TopK"], sort=True):
        row: dict[str, Any] = {"周级门槛": gate, "TopK": topk, "随机轮数": len(group)}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_随机均值"] = values.mean()
            row[f"{metric}_P05"] = values.quantile(0.05)
            row[f"{metric}_P50"] = values.quantile(0.50)
            row[f"{metric}_P95"] = values.quantile(0.95)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), detail


def compare_with_random(deterministic: pd.DataFrame, random_detail: pd.DataFrame) -> pd.DataFrame:
    result = deterministic.copy()
    if result.empty or random_detail.empty:
        return result
    random_means, random_medians, random_wins, random_loss10 = [], [], [], []
    for _, row in result.iterrows():
        group = random_detail[
            random_detail["TopK"].eq(row["TopK"])
            & random_detail["周级门槛"].eq(row["周级门槛"])
        ]
        random_means.append((group["40日平均收益(%)"] <= row["40日平均收益(%)"]).mean() * 100.0)
        random_medians.append((group["40日收益中位数(%)"] <= row["40日收益中位数(%)"]).mean() * 100.0)
        random_wins.append((group["正收益比例(%)"] <= row["正收益比例(%)"]).mean() * 100.0)
        random_loss10.append((group["亏损≤-10%比例(%)"] >= row["亏损≤-10%比例(%)"]).mean() * 100.0)
    result["平均收益随机百分位"] = random_means
    result["收益中位数随机百分位"] = random_medians
    result["胜率随机百分位"] = random_wins
    result["控制10%亏损随机百分位"] = random_loss10
    return result


def week_state_bucket_audit(state: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    signal_state = state[state["Hard_Pool_Count"].gt(0)].copy()
    signal_state["硬池数量分组"] = pd.cut(
        signal_state["Hard_Pool_Count"], bins=[0, 1, 2, 5, 9, np.inf],
        labels=["1只", "2只", "3～5只", "6～9只", "至少10只"])
    signal_state["历史周环境分组"] = pd.cut(
        signal_state["Week_State_Score"], bins=[0, 20, 40, 60, 80, 100],
        labels=["0～20", "20～40", "40～60", "60～80", "80～100"], include_lowest=True)
    for dimension, column in (("硬池候选数量", "硬池数量分组"), ("历史周环境分", "历史周环境分组")):
        for bucket, weeks in signal_state.groupby(column, observed=True, sort=False):
            dates = set(weeks["Week_Last_Trade_Date"].astype(str))
            events = pool[pool["Signal_Date"].astype(str).isin(dates)]
            row = selected_event_stats(events, dimension, 0, str(bucket))
            weekly_returns = events.groupby(events["Signal_Date"].astype(str))["Return_40D_pct"].mean()
            row.update({
                "维度": dimension, "分组": str(bucket), "周数": len(weeks),
                "周等权平均收益(%)": weekly_returns.mean(),
                "周等权收益中位数(%)": weekly_returns.median(),
                "正收益周比例(%)": weekly_returns.gt(0).mean() * 100.0 if len(weekly_returns) else np.nan,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def scenario_and_coverage(events: pd.DataFrame, weeks: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    mature = events["Tradable"].map(to_bool) & events["Has_40D_Future"].map(to_bool)
    non_down = ~events["Downtrend_Hard_Excluded"].map(to_bool)
    touched = events["Recent_3W_Touched_25"].map(to_bool)
    zone = events["Cross_In_20_35"].map(to_bool)
    scenarios = {
        "仅排除下跌趋势": mature & non_down,
        "触底25硬条件＋排除下跌": mature & non_down & touched,
        "20～35硬条件＋排除下跌": mature & non_down & zone,
        "两个硬条件＋排除下跌": mature & non_down & touched & zone,
    }
    summary_rows: list[dict[str, Any]] = []
    count_maps: dict[str, pd.Series] = {}
    for name, mask in scenarios.items():
        group = events[mask]
        counts = group.groupby(group["Signal_Date"].astype(str)).size().reindex(weeks, fill_value=0)
        count_maps[name] = counts
        row = selected_event_stats(group, name, 0)
        row.update({
            "自然周": len(counts), "平均每周事件": counts.mean(), "每周中位数": counts.median(),
            "有信号周": int(counts.gt(0).sum()), "空窗周": int(counts.eq(0).sum()),
            "至少3只候选周": int(counts.ge(3).sum()), "至少10只候选周": int(counts.ge(10).sum()),
            "最大单周候选": int(counts.max()), "最多三只理论可选事件": int(counts.clip(upper=3).sum()),
        })
        summary_rows.append(row)
    calendar = pd.DataFrame({"Week_Last_Trade_Date": weeks})
    for name, counts in count_maps.items():
        calendar[name] = counts.to_numpy()
    calendar["第一梯队事件"] = count_maps["两个硬条件＋排除下跌"].to_numpy()
    calendar["第二梯队补位事件"] = (
        count_maps["20～35硬条件＋排除下跌"] - count_maps["两个硬条件＋排除下跌"]).to_numpy()
    return pd.DataFrame(summary_rows), calendar


def profit_concentration(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if detail.empty:
        return pd.DataFrame()
    for (gate, method, topk), group in detail.groupby(["周级门槛", "排序方法", "TopK"], sort=False):
        ordered = group.sort_values("Return_40D_pct", ascending=False)
        original = pd.to_numeric(ordered["Return_40D_pct"], errors="coerce").dropna()
        for remove_n in (0, 1, 3, 5, 10, 20):
            remaining = original.iloc[min(remove_n, len(original)):]
            removed = ordered.head(remove_n)
            rows.append({
                "周级门槛": gate, "排序方法": method, "TopK": topk, "排除最赚钱事件数": remove_n,
                "原始事件数": len(original), "剩余事件数": len(remaining),
                "原始平均收益(%)": original.mean(), "剩余平均收益(%)": remaining.mean(),
                "剩余收益中位数(%)": remaining.median(),
                "剩余正收益比例(%)": remaining.gt(0).mean() * 100.0 if len(remaining) else np.nan,
                "被排除股票": "；".join(
                    f"{row.get('name', '')}({row.get('ts_code', '')},{finite_num(row.get('Return_40D_pct')):.1f}%)"
                    for _, row in removed.iterrows()) if remove_n else "未排除",
            })
    return pd.DataFrame(rows)


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title="周线SKDJ周级环境审计 V3.9", layout="wide")
    st.title(TITLE)
    st.caption("先判断这个星期是否值得买，再判断同周买哪只；Top2为主观察口径，Top1/Top3只作敏感性对照。")
    with st.expander("冻结规则与评价顺序", expanded=True):
        st.markdown(f"""
- **唯一信号**：完整周线SKDJ金叉，参数冻结 `N={SKDJ_N}, M={SKDJ_M}`。
- **硬条件一**：金叉位置>{CROSS_ZONE_LOW:.0f}且≤{CROSS_ZONE_HIGH:.0f}。
- **硬条件二**：排除“周线低于MA20、近12周收益<0、日线低于MA60”三项同时成立的下跌趋势。
- **第一梯队**：最近{RESET_LOOKBACK_WEEKS}个完整周K或D曾≤{SKDJ_BOTTOM:.0f}。
- **第二梯队**：未触及25，仅在第一梯队不足TopK时补位；分层优先级高于任何个股分数。
- **周级环境**：同周全部金叉数、硬池候选数、行业覆盖数；历史百分位只使用此前最多{HISTORY_WINDOW_WEEKS}个自然周，至少{MIN_HISTORY_WEEKS}周后启用。
- **梯队内排序**：板块共振、周波动4/12周收缩、二者同周等权；不再使用V3.8旧量价评分。
- **评价**：五种周级门槛×三种排序×Top1/Top2/Top3，分别与{RANDOM_RUNS}轮遵守相同周级门槛和梯队顺序的随机选择比较。
- **统一执行**：周线收盘确认后下一市场交易日开盘买入；固定20/40个市场交易日判卷并计入成本。
- **不使用**：月线、市场月线门槛、机器学习、退出参数、资金曲线和事后优化权重。
- **限制**：分层规则由同一三年结果提出；即使优于随机，也必须再经过新的时间样本验证。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("信号开始", date(2023, 6, 5), key="v39_start")
        signal_end_date = st.date_input("信号截止", date(2026, 6, 5), key="v39_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v39_market_end")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v39_pause")
        use_cache = st.checkbox("复用逐股票缓存", True, key="v39_cache")
        st.divider()
        commission_pct = st.number_input("佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f")
        stamp_duty_pct = st.number_input("卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f")
        transfer_fee_pct = st.number_input("过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f")
        if st.button("清除本程序行情缓存", key="v39_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("缓存已清除")
    token = st.text_input("Tushare Token", type="password", key="v39_token")
    session_key = "weekly_skdj_week_state_rank_v39_zip"
    if not token:
        st.info("请输入Tushare Token；V3.4至V3.8相同日期范围的逐股票缓存可以直接复用。")
        return
    if not st.button("开始V3.9周级环境与同周排序审计", type="primary", key="v39_run"):
        if session_key in st.session_state:
            st.download_button(
                "下载上一次结果ZIP", st.session_state[session_key],
                file_name="weekly_skdj_week_state_rank_audit_v3_9_all_results.zip",
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
        "min_price": 10.0, "min_mv": 100.0,
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
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
    cache_hits = data_failures = 0
    progress, status = st.progress(0.0), st.empty()
    for number, stock in stocks.iterrows():
        code = str(stock["ts_code"])
        progress.progress((number + 1) / max(len(stocks), 1), text=f"{number + 1}/{len(stocks)} {code}")
        status.caption(f"全量金叉 {len(events)}；缓存 {cache_hits}；失败 {data_failures}")
        daily, daily_basic, cache_hit = fetch_stock_history(
            code, preload, market_end, bool(use_cache), float(pause))
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        events.extend(analyze_stock(
            stock, period_index.get(code, []), daily, daily_basic,
            week_last_map, open_dates, open_pos, config))
    progress.empty()
    status.empty()
    if not events:
        st.error("研究区间没有生成符合历史科技池、价格和市值条件的完整周线SKDJ金叉。")
        return
    try:
        with st.spinner("构建周级环境、双梯队与严格匹配的随机基准..."):
            event_frame = add_cross_section_features(
                pd.DataFrame(events).sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True))
            pool = build_tiered_pool(event_frame)
            base_calendar = pool_calendar(open_dates, signal_start, signal_end, event_frame)
            weeks = base_calendar["Week_Last_Trade_Date"].astype(str)
            scenarios, calendar = scenario_and_coverage(event_frame, weeks)
            week_state, pool = build_week_state(base_calendar, event_frame, pool)
            state_buckets = week_state_bucket_audit(week_state, pool)
            comparison_raw, selected_detail, yearly, half_yearly = deterministic_rank_audit(pool)
            random_summary, random_detail = random_tier_audit(pool)
            comparison = compare_with_random(comparison_raw, random_detail)
            concentration = profit_concentration(selected_detail)
    except Exception as exc:
        st.exception(exc)
        return
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "信号开始": signal_start, "信号截止": signal_end,
        "观察截止": market_end, "全部周线SKDJ金叉": len(event_frame),
        "40日成熟事件": int(event_frame["Has_40D_Future"].map(to_bool).sum()),
        "20～35且非下跌成熟事件": len(pool), "第一梯队事件": int(pool["Tier_Number"].eq(1).sum()),
        "第二梯队补位事件": int(pool["Tier_Number"].eq(2).sum()),
        "硬池不同股票": pool["ts_code"].nunique(), "自然周": len(calendar),
        "硬池有信号周": pool["Signal_Date"].nunique(),
        "硬池空窗周": int(calendar["20～35硬条件＋排除下跌"].eq(0).sum()),
        "Top2主观察": True, "周环境历史窗口": HISTORY_WINDOW_WEEKS,
        "随机轮数": RANDOM_RUNS, "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    metadata = pd.DataFrame([
        ("硬候选", "完整周线SKDJ金叉位置>20且≤35，并排除下跌趋势"),
        ("下跌趋势", "信号日周线MA20偏离<0、近12周收益<0、日线MA60偏离<0三项同时成立"),
        ("第一梯队", "最近3个完整周K或D最低值≤25；始终优先于第二梯队"),
        ("第二梯队", "最近3周未触及25；仅在第一梯队不足TopK时补位"),
        ("周级环境", f"全部金叉数、硬池数、行业覆盖数分别相对此前最多{HISTORY_WINDOW_WEEKS}周计算百分位；至少{MIN_HISTORY_WEEKS}周后启用"),
        ("周级门槛", "全部硬池周、第一梯队存在、硬池至少2只、历史周环境分≥50、历史周环境分≥70；均为并列审计，不事后择优"),
        ("梯队内因子", "板块共振、周振幅4/12周收缩、二者同周等权；分层优先于分数"),
        ("随机基准", f"{RANDOM_RUNS}轮；每轮遵守相同周级门槛、第一梯队优先和第二梯队补位，再随机Top1/2/3"),
        ("SKDJ参数", f"N={SKDJ_N},M={SKDJ_M}，冻结不寻优"),
        ("股票池", "申万2021历史科技池；主板/创业板/科创板；排除北交所"),
        ("价格市值", "信号日原始收盘价≥10元；历史流通市值≥100亿元"),
        ("买入", "完整周线确认后下一市场交易日开盘；主板一字板不买"),
        ("判卷", "固定20/40个市场交易日收益、MFE与MAE；计入滑点和交易成本"),
        ("防前视", "周线只用完整周；所有排名特征只用信号日及以前；未来40日只用于判卷"),
        ("月线", "完全不使用"),
        ("主观察口径", "Top2；Top1和Top3仅作敏感性对照；不生成资金曲线"),
        ("限制", "候选数量分组来自V3.8发现，本版仍是同一历史复核；历史百分位防止使用未来周，但不是独立新时期证明"),
    ], columns=["项目", "值"])
    files = {
        "01_run_summary_v3_9.csv": run_summary,
        "02_pool_scenario_coverage_quality_v3_9.csv": scenarios,
        "03_week_state_bucket_audit_v3_9.csv": state_buckets,
        "04_gate_rank_vs_matched_random_v3_9.csv": comparison,
        "05_gate_rank_yearly_stability_v3_9.csv": yearly,
        "06_gate_rank_half_year_stability_v3_9.csv": half_yearly,
        "07_matched_random_distribution_summary_v3_9.csv": random_summary,
        "08_matched_random_all_runs_v3_9.csv": random_detail,
        "09_profit_concentration_remove_top_v3_9.csv": concentration,
        "10_selected_event_detail_v3_9.csv": selected_detail,
        "11_tiered_hard_pool_with_week_state_v3_9.csv": pool,
        "12_week_state_calendar_v3_9.csv": week_state,
        "13_weekly_tier_coverage_calendar_v3_9.csv": calendar,
        "14_all_weekly_skdj_events_v3_9.csv": event_frame,
        "15_full_tech_universe_v3_9.csv": stocks,
        "16_board_population_v3_9.csv": population,
        "17_rejection_audit_v3_9.csv": pd.DataFrame(
            [{"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]),
        "18_api_errors_v3_9.csv": pd.DataFrame({"错误": API_ERRORS}),
        "19_metadata_v3_9.csv": metadata,
    }
    result_zip = make_zip(files)
    st.session_state[session_key] = result_zip
    st.success(
        f"完成：全量金叉{len(event_frame)}个；20～35且非下跌硬池{len(pool)}个；"
        f"第一梯队{pool['Tier_Number'].eq(1).sum()}个，第二梯队{pool['Tier_Number'].eq(2).sum()}个。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("硬池事件", len(pool))
    c2.metric("第一梯队", int(pool["Tier_Number"].eq(1).sum()))
    c3.metric("第二梯队补位", int(pool["Tier_Number"].eq(2).sum()))
    c4.metric("硬池空窗周", int(calendar["20～35硬条件＋排除下跌"].eq(0).sum()))
    st.subheader("候选池覆盖率与质量")
    st.dataframe(scenarios, use_container_width=True, hide_index=True)
    st.subheader("周级环境分组审计")
    st.dataframe(state_buckets, use_container_width=True, hide_index=True)
    st.subheader("周级门槛＋同周排序相对严格匹配随机")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.subheader("Top2年度稳定性（主观察）")
    st.dataframe(yearly[yearly["TopK"].eq(2)], use_container_width=True, hide_index=True)
    st.subheader("剔除最赚钱事件后的稳健性")
    st.dataframe(concentration, use_container_width=True, hide_index=True)
    st.download_button(
        "下载V3.9全部结果ZIP", result_zip,
        file_name="weekly_skdj_week_state_rank_audit_v3_9_all_results.zip",
        mime="application/zip", type="primary", key="v39_download", on_click="ignore")
    st.info("先看03确认周级环境是否跨分组呈稳定改善；再以04中的Top2为主，要求平均收益、中位数、胜率和10%亏损控制同时优于匹配随机；最后用05、06、09排除单一年份和少数牛股支撑。")


if __name__ == "__main__":
    main()
