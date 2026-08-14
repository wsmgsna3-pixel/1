# -*- coding: utf-8 -*-
"""
科技股周线SKDJ宽池、少硬筛与状态分层特征审计 V4.0

目的：
1. 恢复全部完整周线SKDJ金叉，不再用20~35、最近触底或周环境分截断候选池。
2. 基础宽池只排除明确个股下跌趋势；其余旧条件全部降为待验证特征。
3. 用上证指数完整周线把市场标记为上涨、震荡或下跌，只分层研究，不作为硬门槛。
4. 在每周候选股内部审计相对强度、量价、波动收缩、SKDJ结构和上次金叉股性。
5. 同时输出单特征Top3、反向Top3、随机基准、年度稳定性、覆盖率和三仓位容量诊断。
6. 所有事件统一在周线确认后的下一市场交易日开盘买入，以20/40日固定终点、MFE和MAE判卷。

注意：本版是特征发现审计，不生成最终评分公式；同一三年中表现好的特征仍需新时期验证。
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
TITLE = "科技股周线SKDJ宽池与状态分层特征审计 V4.0"
VERSION = "V4.0-WEEKLY-SKDJ-WIDE-POOL-REGIME-FEATURE-AUDIT"
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
RANDOM_SEED = 20260814
RANDOM_RUNS = 300
BENCHMARK_CODE = "000001.SH"
PORTFOLIO_SLOTS = 3

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
def load_benchmark_daily(start_date: str, end_date: str) -> pd.DataFrame:
    frame = safe_get(
        "index_daily", ts_code=BENCHMARK_CODE, start_date=start_date, end_date=end_date,
        fields="ts_code,trade_date,open,high,low,close,vol")
    if frame.empty:
        record_error(f"基准指数{BENCHMARK_CODE}为空，市场状态将标记为未知")
        return pd.DataFrame()
    for column in ("open", "high", "low", "close", "vol"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame.dropna(subset=["trade_date", "open", "high", "low", "close"]).drop_duplicates(
        "trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)


def build_market_week_state(daily: pd.DataFrame,
                            week_last_map: dict[pd.Timestamp, str]) -> pd.DataFrame:
    columns = [
        "Signal_Date", "Market_State", "Market_Weekly_Close", "Market_MA20_Bias_pct",
        "Market_MA20_Slope_4W_pct", "Market_Return_4W_pct", "Market_Return_12W_pct",
    ]
    if daily.empty:
        return pd.DataFrame(columns=columns)
    weekly = aggregate_weekly(daily)
    weekly["calendar_week_last"] = weekly["week_label"].map(week_last_map)
    weekly = weekly[
        weekly["calendar_week_last"].notna()
        & weekly["trade_date"].astype(str).eq(weekly["calendar_week_last"].astype(str))
    ].copy().reset_index(drop=True)
    close = weekly["close"]
    ma20 = close.rolling(20).mean()
    weekly["Market_MA20_Bias_pct"] = (close / ma20 - 1.0) * 100.0
    weekly["Market_MA20_Slope_4W_pct"] = (ma20 / ma20.shift(4) - 1.0) * 100.0
    weekly["Market_Return_4W_pct"] = close.pct_change(4, fill_method=None) * 100.0
    weekly["Market_Return_12W_pct"] = close.pct_change(12, fill_method=None) * 100.0
    up = (
        weekly["Market_MA20_Bias_pct"].gt(0)
        & weekly["Market_MA20_Slope_4W_pct"].gt(0)
        & weekly["Market_Return_12W_pct"].gt(0)
    )
    down = (
        weekly["Market_MA20_Bias_pct"].lt(0)
        & weekly["Market_MA20_Slope_4W_pct"].lt(0)
        & weekly["Market_Return_12W_pct"].lt(0)
    )
    weekly["Market_State"] = np.select([up, down], ["上涨", "下跌"], default="震荡")
    weekly["Signal_Date"] = weekly["trade_date"].astype(str)
    weekly["Market_Weekly_Close"] = weekly["close"]
    return weekly[columns]


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
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
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
        "Weekly_MA10_Bias_pct": float((close / ma10 - 1.0).iloc[position] * 100.0),
        "Weekly_MA20_Bias_pct": float((close / ma20 - 1.0).iloc[position] * 100.0),
        "Weekly_MA20_Slope_4W_pct": float((ma20 / ma20.shift(4) - 1.0).iloc[position] * 100.0),
        "Weekly_Volume_Ratio_4_12": float((vol.rolling(4).mean() / vol.rolling(12).mean().replace(0, np.nan)).iloc[position]),
        "Weekly_Contraction_4_12": float((range_pct.rolling(4).mean() / range_pct.rolling(12).mean().replace(0, np.nan)).iloc[position]),
    }


def previous_cross_features(weekly: pd.DataFrame, position: int, daily: pd.DataFrame,
                            signal_date: str) -> dict[str, Any]:
    """只使用当前信号日以前行情，刻画该股上一次周线SKDJ金叉后的真实股性。"""
    defaults = {
        "Previous_Cross_Date": "", "Weeks_Since_Previous_Cross": np.nan,
        "Previous_Cross_Observed_Days": np.nan, "Previous_Cross_End_Return_pct": np.nan,
        "Previous_Cross_MFE_pct": np.nan, "Previous_Cross_MAE_pct": np.nan,
    }
    prior_positions = [
        idx for idx in range(INDICATOR_WARMUP_WEEKS, position)
        if to_bool(weekly.iloc[idx].get("SKDJ_Golden_Cross"))
    ]
    if not prior_positions:
        return defaults
    previous_position = int(prior_positions[-1])
    previous_date = str(weekly.iloc[previous_position]["trade_date"])
    after_previous = daily[
        daily["trade_date"].astype(str).gt(previous_date)
        & daily["trade_date"].astype(str).le(signal_date)
    ].sort_values("trade_date").head(HOLD_40D)
    result = defaults.copy()
    result["Previous_Cross_Date"] = previous_date
    result["Weeks_Since_Previous_Cross"] = float(position - previous_position)
    if after_previous.empty:
        return result
    entry = finite_num(after_previous.iloc[0].get("open"))
    if not math.isfinite(entry) or entry <= 0:
        return result
    observed = len(after_previous)
    result.update({
        "Previous_Cross_Observed_Days": float(observed),
        "Previous_Cross_End_Return_pct": (finite_num(after_previous.iloc[-1].get("close")) / entry - 1.0) * 100.0,
        "Previous_Cross_MFE_pct": (pd.to_numeric(after_previous["high"], errors="coerce").max() / entry - 1.0) * 100.0,
        "Previous_Cross_MAE_pct": (pd.to_numeric(after_previous["low"], errors="coerce").min() / entry - 1.0) * 100.0,
    })
    return result


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
    event.update(previous_cross_features(weekly, position, daily, signal_date))
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


V40_FEATURE_SPECS: dict[str, str] = {
    "周线SKDJ金叉位置": "Weekly_Cross_Level",
    "近3周SKDJ最低值": "Recent_3W_Min_SKDJ",
    "周K一周斜率": "Weekly_SKDJ_K_Change_1W",
    "周D一周斜率": "Weekly_SKDJ_D_Change_1W",
    "周线4周涨幅": "Weekly_Return_4W_pct",
    "周线12周涨幅": "Weekly_Return_12W_pct",
    "周线MA10偏离": "Weekly_MA10_Bias_pct",
    "周线MA20偏离": "Weekly_MA20_Bias_pct",
    "周线MA20四周斜率": "Weekly_MA20_Slope_4W_pct",
    "周成交量4比12": "Weekly_Volume_Ratio_4_12",
    "周波动收缩4比12": "Weekly_Contraction_4_12",
    "日线SKDJ位置": "Daily_SKDJ_Level_At_Cross",
    "日线5日涨幅": "Daily_Return_5D_pct",
    "日线20日涨幅": "Daily_Return_20D_pct",
    "日线60日涨幅": "Daily_Return_60D_pct",
    "日线MA20偏离": "Daily_MA20_Bias_pct",
    "日线MA60偏离": "Daily_MA60_Bias_pct",
    "日成交量5比20": "Daily_Volume_Ratio_5_20",
    "换手率": "Turnover_Rate",
    "日ATR14": "Daily_ATR14_pct",
    "日线10日振幅": "Daily_Amplitude_10D_pct",
    "距离60日高点": "Distance_60D_High_pct",
    "同周候选20日相对强度": "Candidate_RS20_Pct",
    "同周候选60日相对强度": "Candidate_RS60_Pct",
    "相对同周同行业20日强度": "Industry_Relative_RS20_pct",
    "同周行业共振强度": "Industry_Resonance_Pct",
    "同周量比排名": "Candidate_Volume_Rank_Pct",
    "距上次周线金叉周数": "Weeks_Since_Previous_Cross",
    "上次金叉后最多40日收益": "Previous_Cross_End_Return_pct",
    "上次金叉后最多40日最大涨幅": "Previous_Cross_MFE_pct",
    "上次金叉后最多40日最大回撤": "Previous_Cross_MAE_pct",
}


def v40_add_research_features(events: pd.DataFrame, market_state: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["Signal_Date"] = frame["Signal_Date"].astype(str)
    frame["Signal_Date_dt"] = pd.to_datetime(frame["Signal_Date"], format="%Y%m%d", errors="coerce")
    frame["Year"] = frame["Signal_Date"].str[:4]
    frame["Half_Year"] = frame["Year"] + "H" + np.where(frame["Signal_Date_dt"].dt.month.le(6), "1", "2")
    frame["Week_Signal_Count"] = frame.groupby("Signal_Date")["ts_code"].transform("size").astype(float)
    frame["Industry_Signal_Count"] = frame.groupby(["Signal_Date", "SW_L1"])["ts_code"].transform("size").astype(float)
    if market_state.empty:
        frame["Market_State"] = "未知"
        for column in (
            "Market_Weekly_Close", "Market_MA20_Bias_pct", "Market_MA20_Slope_4W_pct",
            "Market_Return_4W_pct", "Market_Return_12W_pct",
        ):
            frame[column] = np.nan
    else:
        frame = frame.merge(market_state, on="Signal_Date", how="left")
        frame["Market_State"] = frame["Market_State"].fillna("未知")

    weekly_bias = pd.to_numeric(frame["Weekly_MA20_Bias_pct"], errors="coerce")
    weekly_slope = pd.to_numeric(frame["Weekly_MA20_Slope_4W_pct"], errors="coerce")
    weekly_return = pd.to_numeric(frame["Weekly_Return_12W_pct"], errors="coerce")
    daily_bias = pd.to_numeric(frame["Daily_MA60_Bias_pct"], errors="coerce")
    downtrend = weekly_bias.lt(0) & weekly_slope.lt(0) & weekly_return.lt(0) & daily_bias.lt(0)
    uptrend = weekly_bias.gt(0) & weekly_slope.gt(0) & weekly_return.gt(0) & daily_bias.gt(0)
    frame["Individual_Trend_State"] = np.select([uptrend, downtrend], ["上涨", "下跌"], default="震荡或转换")
    frame["Individual_Downtrend"] = downtrend
    frame["Mature_Tradable"] = frame["Tradable"].map(to_bool) & frame["Has_40D_Future"].map(to_bool)
    frame["Wide_Pool"] = frame["Mature_Tradable"] & ~frame["Individual_Downtrend"]

    for source, target in (
        ("Daily_Return_20D_pct", "Candidate_RS20_Pct"),
        ("Daily_Return_60D_pct", "Candidate_RS60_Pct"),
        ("Daily_Volume_Ratio_5_20", "Candidate_Volume_Rank_Pct"),
    ):
        values = pd.to_numeric(frame[source], errors="coerce")
        frame[target] = values.groupby(frame["Signal_Date"]).rank(pct=True, method="average") * 100.0
    industry_median = pd.to_numeric(frame["Daily_Return_20D_pct"], errors="coerce").groupby(
        [frame["Signal_Date"], frame["SW_L1"]]).transform("median")
    frame["Industry_Relative_RS20_pct"] = pd.to_numeric(
        frame["Daily_Return_20D_pct"], errors="coerce") - industry_median
    frame["Industry_Resonance_Pct"] = frame["Industry_Signal_Count"].groupby(
        frame["Signal_Date"]).rank(pct=True, method="average") * 100.0

    previous_observed = pd.to_numeric(frame["Previous_Cross_Observed_Days"], errors="coerce")
    frame["Previous_Cross_Feature_Valid"] = previous_observed.ge(20)
    frame["Extreme_Overextension"] = (
        pd.to_numeric(frame["Weekly_Return_4W_pct"], errors="coerce").gt(30)
        | pd.to_numeric(frame["Daily_MA20_Bias_pct"], errors="coerce").gt(20)
    )
    frame["Weak_Volume_Cross"] = (
        pd.to_numeric(frame["Weekly_Volume_Ratio_4_12"], errors="coerce").lt(0.75)
        & pd.to_numeric(frame["Daily_Volume_Ratio_5_20"], errors="coerce").lt(0.80)
    )
    frame["Previous_Cross_Weak"] = (
        frame["Previous_Cross_Feature_Valid"]
        & pd.to_numeric(frame["Previous_Cross_End_Return_pct"], errors="coerce").lt(0)
        & pd.to_numeric(frame["Previous_Cross_MFE_pct"], errors="coerce").lt(10)
    )
    returns = pd.to_numeric(frame["Return_40D_pct"], errors="coerce")
    frame["Outcome_Class_40D"] = np.select(
        [returns.isna(), returns.ge(10), returns.gt(0), returns.gt(-10)],
        ["未成熟", "盈利≥10%", "盈利0～10%", "亏损0～10%"], default="亏损≥10%")
    return frame.sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True)


def v40_event_stats(frame: pd.DataFrame) -> dict[str, Any]:
    returns20 = pd.to_numeric(frame.get("Return_20D_pct"), errors="coerce").dropna()
    returns = pd.to_numeric(frame.get("Return_40D_pct"), errors="coerce").dropna()
    mfe = pd.to_numeric(frame.get("MFE_40D_pct"), errors="coerce").dropna()
    mae = pd.to_numeric(frame.get("MAE_40D_pct"), errors="coerce").dropna()
    return {
        "事件数": len(returns), "股票数": frame["ts_code"].nunique() if len(frame) else 0,
        "信号周": frame["Signal_Date"].nunique() if len(frame) else 0,
        "20日平均收益(%)": returns20.mean(), "20日中位数(%)": returns20.median(),
        "40日平均收益(%)": returns.mean(), "40日中位数(%)": returns.median(),
        "正收益比例(%)": returns.gt(0).mean() * 100.0 if len(returns) else np.nan,
        "收益≥10%比例(%)": returns.ge(10).mean() * 100.0 if len(returns) else np.nan,
        "亏损≤-10%比例(%)": returns.le(-10).mean() * 100.0 if len(returns) else np.nan,
        "MFE中位数(%)": mfe.median(), "MAE中位数(%)": mae.median(),
        "平均MAE(%)": mae.mean(),
    }


def v40_max_empty_run(counts: pd.Series) -> int:
    longest = current = 0
    for empty in counts.eq(0).tolist():
        current = current + 1 if empty else 0
        longest = max(longest, current)
    return int(longest)


def v40_week_calendar(open_dates: list[str], start: str, end: str) -> pd.DataFrame:
    days = pd.DataFrame({"trade_date": [day for day in open_dates if start <= day <= end]})
    days["dt"] = pd.to_datetime(days["trade_date"])
    days["week"] = days["dt"].dt.to_period("W-FRI")
    return days.groupby("week")["trade_date"].max().rename(
        "Week_Last_Trade_Date").reset_index(drop=True).to_frame()


def v40_coverage_audit(events: pd.DataFrame, calendar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mature = events["Mature_Tradable"].map(to_bool)
    non_down = ~events["Individual_Downtrend"].map(to_bool)
    touched = events["Recent_3W_Touched_25"].map(to_bool)
    zone = events["Cross_In_20_35"].map(to_bool)
    scenarios = {
        "完整SKDJ金叉成熟池": mature,
        "宽池_仅排除明确个股下跌": mature & non_down,
        "旧条件_触底25并排除下跌": mature & non_down & touched,
        "旧条件_金叉20至35并排除下跌": mature & non_down & zone,
        "旧条件_触底且20至35并排除下跌": mature & non_down & touched & zone,
    }
    detail = calendar.copy()
    rows: list[dict[str, Any]] = []
    weeks = detail["Week_Last_Trade_Date"].astype(str)
    for name, mask in scenarios.items():
        group = events[mask]
        counts = group.groupby("Signal_Date").size().reindex(weeks, fill_value=0)
        detail[name] = counts.to_numpy()
        nonzero = counts[counts.gt(0)]
        row = {"候选池": name, **v40_event_stats(group)}
        row.update({
            "自然周": len(counts), "有信号周": int(counts.gt(0).sum()),
            "覆盖率(%)": counts.gt(0).mean() * 100.0, "空窗周": int(counts.eq(0).sum()),
            "最长连续空窗周": v40_max_empty_run(counts), "平均每周候选": counts.mean(),
            "有信号周平均候选": nonzero.mean() if len(nonzero) else np.nan,
            "每周候选中位数": counts.median(), "至少3只候选周": int(counts.ge(3).sum()),
            "最大单周候选": int(counts.max()) if len(counts) else 0,
        })
        rows.append(row)
    return pd.DataFrame(rows), detail


def v40_market_state_audit(pool: pd.DataFrame, calendar: pd.DataFrame,
                           market_state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_calendar = calendar.merge(
        market_state, left_on="Week_Last_Trade_Date", right_on="Signal_Date", how="left")
    state_calendar["Market_State"] = state_calendar["Market_State"].fillna("未知")
    counts = pool.groupby("Signal_Date").size()
    state_calendar["Wide_Pool_Count"] = state_calendar["Week_Last_Trade_Date"].map(counts).fillna(0).astype(int)
    rows = []
    for state, weeks in state_calendar.groupby("Market_State", sort=False):
        dates = set(weeks["Week_Last_Trade_Date"].astype(str))
        group = pool[pool["Signal_Date"].isin(dates)]
        row = {"大盘周线状态": state, **v40_event_stats(group)}
        row.update({
            "自然周": len(weeks), "有信号周": int(weeks["Wide_Pool_Count"].gt(0).sum()),
            "空窗周": int(weeks["Wide_Pool_Count"].eq(0).sum()),
            "平均每周候选": weeks["Wide_Pool_Count"].mean(),
        })
        rows.append(row)
    return pd.DataFrame(rows), state_calendar


def v40_veto_candidate_audit(events: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    base = events[events["Mature_Tradable"].map(to_bool)].copy()
    weeks = calendar["Week_Last_Trade_Date"].astype(str)
    vetoes = {
        "明确个股下跌趋势": base["Individual_Downtrend"].map(to_bool),
        "极端追高_4周涨幅大于30或日线偏离MA20大于20": base["Extreme_Overextension"].map(to_bool),
        "周日线同时明显缩量": base["Weak_Volume_Cross"].map(to_bool),
        "上次金叉表现弱且历史观察至少20日": base["Previous_Cross_Weak"].map(to_bool),
        "旧规则_最近3周未触及25": ~base["Recent_3W_Touched_25"].map(to_bool),
        "旧规则_金叉不在20至35": ~base["Cross_In_20_35"].map(to_bool),
    }
    total_losses = pd.to_numeric(base["Return_40D_pct"], errors="coerce").le(0).sum()
    total_winners = pd.to_numeric(base["Return_40D_pct"], errors="coerce").gt(0).sum()
    rows = []
    for name, veto_mask in vetoes.items():
        removed, retained = base[veto_mask], base[~veto_mask]
        counts = retained.groupby("Signal_Date").size().reindex(weeks, fill_value=0)
        removed_returns = pd.to_numeric(removed["Return_40D_pct"], errors="coerce")
        row = {
            "候选剔除条件": name, "基础成熟事件": len(base), "剔除事件": len(removed),
            "剔除比例(%)": len(removed) / len(base) * 100.0 if len(base) else np.nan,
            "捕获全部亏损事件比例(%)": removed_returns.le(0).sum() / total_losses * 100.0 if total_losses else np.nan,
            "误杀全部盈利事件比例(%)": removed_returns.gt(0).sum() / total_winners * 100.0 if total_winners else np.nan,
            "被剔除40日平均(%)": removed_returns.mean(), "被剔除40日中位数(%)": removed_returns.median(),
            "被剔除正收益比例(%)": removed_returns.gt(0).mean() * 100.0 if len(removed_returns) else np.nan,
            **{f"保留_{key}": value for key, value in v40_event_stats(retained).items()},
            "保留有信号周": int(counts.gt(0).sum()), "保留空窗周": int(counts.eq(0).sum()),
            "保留覆盖率(%)": counts.gt(0).mean() * 100.0, "保留最长连续空窗周": v40_max_empty_run(counts),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def v40_feature_profile(pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in ("全部", "上涨", "震荡", "下跌"):
        state_pool = pool if state == "全部" else pool[pool["Market_State"].eq(state)]
        for label, column in V40_FEATURE_SPECS.items():
            values = pd.to_numeric(state_pool.get(column), errors="coerce")
            returns = pd.to_numeric(state_pool.get("Return_40D_pct"), errors="coerce")
            valid = pd.DataFrame({"x": values, "r": returns}).dropna()
            if len(valid) < 30 or valid["x"].nunique() < 5:
                continue
            q20, q80 = valid["x"].quantile([0.20, 0.80])
            bottom, top = valid[valid["x"].le(q20)], valid[valid["x"].ge(q80)]
            top_median, bottom_median = top["r"].median(), bottom["r"].median()
            direction = "数值越高越好" if top_median >= bottom_median else "数值越低越好"
            rows.append({
                "大盘状态": state, "特征": label, "字段": column, "有效样本": len(valid),
                "特征中位数": valid["x"].median(), "收益秩相关": valid["x"].rank().corr(valid["r"].rank()),
                "底部20%样本": len(bottom), "底部20%收益平均(%)": bottom["r"].mean(),
                "底部20%收益中位数(%)": bottom_median,
                "底部20%胜率(%)": bottom["r"].gt(0).mean() * 100.0,
                "顶部20%样本": len(top), "顶部20%收益平均(%)": top["r"].mean(),
                "顶部20%收益中位数(%)": top_median,
                "顶部20%胜率(%)": top["r"].gt(0).mean() * 100.0,
                "顶部减底部中位数差(百分点)": top_median - bottom_median,
                "探索性建议方向": direction,
            })
    return pd.DataFrame(rows)


def v40_profit_loss_feature_compare(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ("上涨", "震荡", "下跌", "全部"):
        group = pool if state == "全部" else pool[pool["Market_State"].eq(state)]
        returns = pd.to_numeric(group["Return_40D_pct"], errors="coerce")
        for label, column in V40_FEATURE_SPECS.items():
            values = pd.to_numeric(group.get(column), errors="coerce")
            valid = pd.DataFrame({"x": values, "r": returns}).dropna()
            winners, losers = valid[valid["r"].gt(0)], valid[valid["r"].le(0)]
            if min(len(winners), len(losers)) < 10:
                continue
            rows.append({
                "大盘状态": state, "特征": label, "字段": column,
                "盈利样本": len(winners), "亏损样本": len(losers),
                "盈利股特征中位数": winners["x"].median(),
                "亏损股特征中位数": losers["x"].median(),
                "盈利减亏损中位数": winners["x"].median() - losers["x"].median(),
                "盈利股特征均值": winners["x"].mean(), "亏损股特征均值": losers["x"].mean(),
            })
    return pd.DataFrame(rows)


def v40_direction_lookup(profile: pd.DataFrame) -> dict[tuple[str, str], str]:
    return {
        (str(row["大盘状态"]), str(row["字段"])): str(row["探索性建议方向"])
        for _, row in profile.iterrows()
    }


def v40_select_weekly_top3(pool: pd.DataFrame, column: str, direction: str) -> pd.DataFrame:
    work = pool.copy()
    work["Feature_Value"] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["Feature_Value"])
    ascending = direction == "数值越低越好"
    selected = work.sort_values(
        ["Signal_Date", "Feature_Value", "ts_code"],
        ascending=[True, ascending, True], kind="mergesort").groupby(
        "Signal_Date", sort=False).head(3).copy()
    selected["Selection_Priority"] = selected.groupby("Signal_Date").cumcount() * -1.0
    return selected


def v40_random_top3_distribution(pool: pd.DataFrame, state: str) -> pd.DataFrame:
    state_pool = pool if state == "全部" else pool[pool["Market_State"].eq(state)]
    rows = []
    for run in range(RANDOM_RUNS):
        rng = np.random.default_rng(RANDOM_SEED + run + sum(ord(ch) for ch in state) * 1000)
        work = state_pool.copy()
        work["Random_Score"] = rng.random(len(work))
        selected = work.sort_values(
            ["Signal_Date", "Random_Score", "ts_code"],
            ascending=[True, False, True], kind="mergesort").groupby(
            "Signal_Date", sort=False).head(3)
        row = v40_event_stats(selected)
        row["随机轮次"] = run
        rows.append(row)
    return pd.DataFrame(rows)


def v40_capacity_diagnostic(selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {"三仓接受事件": 0, "因仓位错过": 0, "三仓接受信号周": 0,
                "三仓40日平均(%)": np.nan, "三仓40日中位数(%)": np.nan, "三仓胜率(%)": np.nan}
    active: list[dict[str, str]] = []
    accepted_rows: list[pd.Series] = []
    rejected = 0
    ordered = selected.sort_values(
        ["Entry_Date", "Selection_Priority", "ts_code"], ascending=[True, False, True], kind="mergesort")
    for entry_date, day_group in ordered.groupby("Entry_Date", sort=True):
        active = [position for position in active if position["end_date"] >= str(entry_date)]
        active_codes = {position["ts_code"] for position in active}
        slots = PORTFOLIO_SLOTS - len(active)
        for _, row in day_group.iterrows():
            code = str(row["ts_code"])
            if slots <= 0 or code in active_codes:
                rejected += 1
                continue
            end_date = str(row.get("Outcome_40D_End_Date", ""))
            if not end_date:
                rejected += 1
                continue
            accepted_rows.append(row)
            active.append({"ts_code": code, "end_date": end_date})
            active_codes.add(code)
            slots -= 1
    accepted = pd.DataFrame(accepted_rows)
    returns = pd.to_numeric(accepted.get("Return_40D_pct"), errors="coerce").dropna()
    return {
        "三仓接受事件": len(accepted), "因仓位错过": rejected,
        "三仓接受信号周": accepted["Signal_Date"].nunique() if len(accepted) else 0,
        "三仓40日平均(%)": returns.mean(), "三仓40日中位数(%)": returns.median(),
        "三仓胜率(%)": returns.gt(0).mean() * 100.0 if len(returns) else np.nan,
    }


def v40_single_feature_top3_audit(pool: pd.DataFrame, profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directions = v40_direction_lookup(profile)
    random_by_state = {state: v40_random_top3_distribution(pool, state) for state in ("全部", "上涨", "震荡", "下跌")}
    rows, detail_frames, yearly_rows = [], [], []
    for state in ("全部", "上涨", "震荡", "下跌"):
        state_pool = pool if state == "全部" else pool[pool["Market_State"].eq(state)]
        random_detail = random_by_state[state]
        for label, column in V40_FEATURE_SPECS.items():
            direction = directions.get((state, column))
            if direction is None or column not in state_pool.columns:
                continue
            selected = v40_select_weekly_top3(state_pool, column, direction)
            if selected.empty:
                continue
            selected.insert(0, "审计大盘状态", state)
            selected.insert(1, "特征", label)
            selected.insert(2, "探索方向", direction)
            detail_frames.append(selected)
            stat = {"大盘状态": state, "特征": label, "字段": column, "方向": direction, **v40_event_stats(selected)}
            stat.update(v40_capacity_diagnostic(selected))
            if not random_detail.empty:
                stat["平均收益随机百分位"] = (
                    pd.to_numeric(random_detail["40日平均收益(%)"], errors="coerce")
                    .le(stat["40日平均收益(%)"]).mean() * 100.0)
                stat["中位数随机百分位"] = (
                    pd.to_numeric(random_detail["40日中位数(%)"], errors="coerce")
                    .le(stat["40日中位数(%)"]).mean() * 100.0)
                stat["胜率随机百分位"] = (
                    pd.to_numeric(random_detail["正收益比例(%)"], errors="coerce")
                    .le(stat["正收益比例(%)"]).mean() * 100.0)
            rows.append(stat)
            for year, year_group in selected.groupby("Year", sort=True):
                yearly_rows.append({
                    "大盘状态": state, "特征": label, "字段": column, "方向": direction,
                    "年份": year, **v40_event_stats(year_group),
                })
    random_all = pd.concat(
        [frame.assign(大盘状态=state) for state, frame in random_by_state.items()], ignore_index=True)
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows), pd.concat([detail], ignore_index=True), random_all


def v40_feature_year_stability(profile: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    if yearly.empty:
        return pd.DataFrame()
    rows = []
    for (state, label, column, direction), group in yearly.groupby(
            ["大盘状态", "特征", "字段", "方向"], sort=False):
        medians = pd.to_numeric(group["40日中位数(%)"], errors="coerce")
        wins = pd.to_numeric(group["正收益比例(%)"], errors="coerce")
        rows.append({
            "大盘状态": state, "特征": label, "字段": column, "方向": direction,
            "有样本年份": len(group), "正中位数年份": int(medians.gt(0).sum()),
            "正中位数年份占比(%)": medians.gt(0).mean() * 100.0,
            "年度中位数最差值(%)": medians.min(), "年度中位数平均值(%)": medians.mean(),
            "年度胜率最低值(%)": wins.min(),
        })
    return pd.DataFrame(rows)


def v40_profit_concentration(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for (state, feature), group in detail.groupby(["审计大盘状态", "特征"], sort=False):
        ordered = group.sort_values("Return_40D_pct", ascending=False)
        original = pd.to_numeric(ordered["Return_40D_pct"], errors="coerce").dropna()
        for remove_n in (0, 5, 10, 20):
            remaining = original.iloc[min(remove_n, len(original)):]
            rows.append({
                "大盘状态": state, "特征": feature, "排除最赚钱事件数": remove_n,
                "原始事件": len(original), "剩余事件": len(remaining),
                "原始平均收益(%)": original.mean(), "剩余平均收益(%)": remaining.mean(),
                "剩余中位数(%)": remaining.median(),
                "剩余胜率(%)": remaining.gt(0).mean() * 100.0 if len(remaining) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title="周线SKDJ宽池特征审计 V4.0", layout="wide")
    st.title(TITLE)
    st.caption("恢复完整周线SKDJ金叉宽池；少量硬筛只清除明显风险，其余条件全部进入上涨/震荡/下跌状态下的特征审计。")
    with st.expander("V4.0研究规则", expanded=True):
        st.markdown(f"""
- **唯一原始信号**：完整周线SKDJ金叉，参数冻结 `N={SKDJ_N}, M={SKDJ_M}`。
- **基础股票池**：申万历史科技池，主板/创业板/科创板；信号日原始股价≥10元、流通市值≥100亿元。
- **主要宽池**：只排除“周线低于MA20、MA20四周斜率<0、近12周收益<0、日线低于MA60”四项同时成立的个股下跌趋势。
- **不再硬筛**：金叉20～35、最近3周触及25、周环境分≥50/70；这些全部降为普通特征或对照场景。
- **大盘状态**：只用{BENCHMARK_CODE}完整周线；上涨/震荡/下跌只用于分层研究，不决定是否允许买入。
- **重点特征**：个股相对同周候选强度、相对同行业强度、量比、板块共振、波动收缩、均线位置、SKDJ结构、上次金叉股性。
- **排序审计**：每项特征分别选同周Top3，与{RANDOM_RUNS}轮同状态随机Top3比较；不把未来收益训练成综合评分。
- **执行与判卷**：周线确认后下一市场交易日开盘，固定20/40日，计入滑点和成本，并附最多{PORTFOLIO_SLOTS}仓的容量诊断。
- **限制**：探索方向由同一三年样本产生，Top3结果是特征发现，不是独立样本外证明。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("信号开始", date(2023, 6, 5), key="v40_start")
        signal_end_date = st.date_input("信号截止", date(2026, 6, 5), key="v40_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v40_market_end")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v40_pause")
        use_cache = st.checkbox("复用逐股票缓存", True, key="v40_cache")
        st.divider()
        commission_pct = st.number_input("佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f", key="v40_commission")
        stamp_duty_pct = st.number_input("卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f", key="v40_stamp")
        transfer_fee_pct = st.number_input("过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f", key="v40_transfer")
        if st.button("清除本程序行情缓存", key="v40_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("缓存已清除")

    token = st.text_input("Tushare Token", type="password", key="v40_token")
    session_key = "weekly_skdj_wide_pool_feature_v40_zip"
    if not token:
        st.info("请输入Tushare Token；V3.4至V3.9相同日期范围的逐股票缓存可以直接复用。")
        return
    if not st.button("开始V4.0宽池与分状态特征审计", type="primary", key="v40_run"):
        if session_key in st.session_state:
            st.download_button(
                "下载上一次结果ZIP", st.session_state[session_key],
                file_name="weekly_skdj_wide_pool_feature_audit_v4_0_all_results.zip",
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
        with st.spinner("加载交易日历、上证指数周状态与历史科技股池..."):
            open_dates = load_trade_calendar(preload, market_end)
            extended_end = (market_end_date + timedelta(days=7)).strftime("%Y%m%d")
            full_open_dates = load_trade_calendar(preload, extended_end)
            week_last_map = complete_week_last_dates(full_open_dates)
            benchmark_daily = load_benchmark_daily(preload, market_end)
            market_state = build_market_week_state(benchmark_daily, week_last_map)
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
        status.caption(f"完整金叉 {len(events)}；缓存 {cache_hits}；失败 {data_failures}")
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
        with st.spinner("构建宽池覆盖率、市场状态与单特征Top3审计..."):
            event_frame = v40_add_research_features(pd.DataFrame(events), market_state)
            calendar = v40_week_calendar(open_dates, signal_start, signal_end)
            coverage, coverage_calendar = v40_coverage_audit(event_frame, calendar)
            wide_pool = event_frame[event_frame["Wide_Pool"].map(to_bool)].copy()
            state_audit, state_calendar = v40_market_state_audit(wide_pool, calendar, market_state)
            veto_audit = v40_veto_candidate_audit(event_frame, calendar)
            feature_profile = v40_feature_profile(wide_pool)
            profit_loss_compare = v40_profit_loss_feature_compare(wide_pool)
            top3_audit, top3_yearly, top3_detail, random_detail = v40_single_feature_top3_audit(
                wide_pool, feature_profile)
            stability = v40_feature_year_stability(feature_profile, top3_yearly)
            concentration = v40_profit_concentration(top3_detail)
    except Exception as exc:
        st.exception(exc)
        return

    wide_counts = wide_pool.groupby("Signal_Date").size().reindex(
        calendar["Week_Last_Trade_Date"].astype(str), fill_value=0)
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "信号开始": signal_start, "信号截止": signal_end,
        "观察截止": market_end, "全部完整周线SKDJ金叉": len(event_frame),
        "40日成熟可买事件": int(event_frame["Mature_Tradable"].sum()),
        "仅排除明确下跌后的宽池事件": len(wide_pool), "宽池不同股票": wide_pool["ts_code"].nunique(),
        "自然周": len(calendar), "宽池有信号周": int(wide_counts.gt(0).sum()),
        "宽池空窗周": int(wide_counts.eq(0).sum()), "最长连续空窗周": v40_max_empty_run(wide_counts),
        "平均每周宽池候选": wide_counts.mean(), "每周宽池候选中位数": wide_counts.median(),
        "至少3只候选周": int(wide_counts.ge(3).sum()), "特征数": len(V40_FEATURE_SPECS),
        "随机轮数": RANDOM_RUNS, "最多仓位": PORTFOLIO_SLOTS,
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    feature_dictionary = pd.DataFrame([
        {"特征": label, "字段": column, "说明": "信号周收盘时已知；按大盘状态分别审计"}
        for label, column in V40_FEATURE_SPECS.items()
    ])
    metadata = pd.DataFrame([
        ("原始信号", f"完整周线SKDJ金叉；N={SKDJ_N},M={SKDJ_M}，参数冻结"),
        ("主要宽池", "成熟可买事件只排除明确个股下跌趋势；不再要求触底25、金叉20至35或高环境分"),
        ("明确个股下跌", "周线MA20偏离<0、MA20四周斜率<0、近12周收益<0、日线MA60偏离<0四项同时成立"),
        ("大盘状态", f"{BENCHMARK_CODE}完整周线：MA20位置、MA20四周斜率和12周收益同正为上涨，同负为下跌，其余为震荡；只分层不硬筛"),
        ("相对强度限制", "同周候选20/60日强度是SKDJ候选池内百分位；相对同行业20日强度是同周同行业候选的相对值，不冒充全市场标准RPS"),
        ("上次金叉股性", "只截取上次金叉之后、当前信号日之前最多40个交易日；超过40日固定取前40日，至少观察20日才标记有效，绝不使用当前信号后的行情"),
        ("候选硬筛审计", "明确下跌、极端追高、明显缩量、上次金叉弱及两个旧规则均单独展示剔除亏损能力和覆盖率损失；除明确下跌外均未正式启用"),
        ("单特征Top3", "每个大盘状态内用全样本顶部/底部20%中位数确定探索方向，再在每周选Top3；属于同样本探索，不是样本外结论"),
        ("随机基准", f"每种大盘状态{RANDOM_RUNS}轮，每周从同一宽池随机最多3只"),
        ("三仓诊断", "按固定40日终点释放仓位，最多3只；仅检查信号拥堵和可接受事件收益，不是完整逐日净值曲线"),
        ("价格市值", "信号日原始收盘价≥10元；历史流通市值≥100亿元"),
        ("买入", "完整周线确认后下一市场交易日开盘；主板一字板不买"),
        ("判卷", "固定20/40个市场交易日收益、MFE与MAE；计入滑点和交易成本"),
        ("不使用", "月线、机器学习、未来行情特征、环境分硬门槛、事后综合权重"),
    ], columns=["项目", "值"])

    files = {
        "01_run_summary_v4_0.csv": run_summary,
        "02_pool_coverage_and_quality_v4_0.csv": coverage,
        "03_weekly_coverage_calendar_v4_0.csv": coverage_calendar,
        "04_market_state_quality_v4_0.csv": state_audit,
        "05_market_state_calendar_v4_0.csv": state_calendar,
        "06_candidate_veto_audit_v4_0.csv": veto_audit,
        "07_feature_top_bottom_profile_v4_0.csv": feature_profile,
        "08_profit_vs_loss_feature_compare_v4_0.csv": profit_loss_compare,
        "09_single_feature_weekly_top3_vs_random_v4_0.csv": top3_audit,
        "10_single_feature_yearly_stability_v4_0.csv": top3_yearly,
        "11_feature_year_stability_summary_v4_0.csv": stability,
        "12_single_feature_top3_profit_concentration_v4_0.csv": concentration,
        "13_single_feature_selected_event_detail_v4_0.csv": top3_detail,
        "14_matched_random_all_runs_v4_0.csv": random_detail,
        "15_wide_pool_all_events_v4_0.csv": wide_pool,
        "16_all_weekly_skdj_events_v4_0.csv": event_frame,
        "17_feature_dictionary_v4_0.csv": feature_dictionary,
        "18_full_tech_universe_v4_0.csv": stocks,
        "19_board_population_v4_0.csv": population,
        "20_rejection_audit_v4_0.csv": pd.DataFrame(
            [{"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]),
        "21_api_errors_v4_0.csv": pd.DataFrame({"错误": API_ERRORS}),
        "22_metadata_v4_0.csv": metadata,
    }
    result_zip = make_zip(files)
    st.session_state[session_key] = result_zip
    st.success(
        f"完成：全部完整金叉{len(event_frame)}个；成熟可买{int(event_frame['Mature_Tradable'].sum())}个；"
        f"仅排除明确下跌后的宽池{len(wide_pool)}个，覆盖{int(wide_counts.gt(0).sum())}/{len(calendar)}周。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("宽池事件", len(wide_pool))
    c2.metric("有信号周", int(wide_counts.gt(0).sum()))
    c3.metric("空窗周", int(wide_counts.eq(0).sum()))
    c4.metric("每周候选中位数", f"{wide_counts.median():.1f}")
    st.subheader("候选池覆盖率与质量")
    st.dataframe(coverage, use_container_width=True, hide_index=True)
    st.subheader("大盘上涨/震荡/下跌分层")
    st.dataframe(state_audit, use_container_width=True, hide_index=True)
    st.subheader("候选硬筛：剔除亏损能力与覆盖率代价")
    st.dataframe(veto_audit, use_container_width=True, hide_index=True)
    st.subheader("单特征每周Top3与随机基准")
    st.dataframe(top3_audit, use_container_width=True, hide_index=True)
    st.subheader("特征年度稳定性")
    st.dataframe(stability, use_container_width=True, hide_index=True)
    st.download_button(
        "下载V4.0全部结果ZIP", result_zip,
        file_name="weekly_skdj_wide_pool_feature_audit_v4_0_all_results.zip",
        mime="application/zip", type="primary", key="v40_download", on_click="ignore")
    st.info("先看02确认宽池是否恢复足够覆盖；再看06决定第二个硬条件是否值得启用；最后用07～12寻找上涨和震荡状态下跨年份稳定、三仓位仍有效且不依赖少数牛股的排序特征。")


if __name__ == "__main__":
    main()
