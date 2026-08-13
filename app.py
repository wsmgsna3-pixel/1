# -*- coding: utf-8 -*-
"""
科技股全量周线SKDJ金叉特征审计与样本外排序器 V3.4

目的：
1. 完整保留所有满足科技池、价格和流通市值要求的周线SKDJ金叉。
2. “最近3周触及25”和“金叉位置20~35”只作为特征；原交集保留为基准组。
3. 所有事件统一在周线信号确认后的下一市场交易日开盘买入。
4. 用20/40日固定终点、MFE和MAE分析高收益与亏损事件的买入时特征。
5. 只用过去已成熟样本训练，按半年度扩展窗口产生严格样本外预测。
6. 比较全量、原核心、简单涨幅量能Top3、模型Top1/Top3及允许空仓的模型Top3。

注意：本版验证的是“候选排序能力”，不是持仓重叠后的30万元三仓资金曲线。
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance


TITLE = "科技股全量周线SKDJ金叉特征审计与样本外排序器 V3.4"
VERSION = "V3.4-WEEKLY-SKDJ-ALL-EVENT-OOS-RANK"
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
MIN_TRAIN_EVENTS = 500
MIN_TEST_EVENTS = 30
MODEL_SEED = 20260813

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

MODEL_FEATURES = [
    "Weekly_Cross_Level", "Recent_3W_Min_SKDJ", "Recent_6W_Min_SKDJ",
    "Weeks_Since_Touch_25", "Weekly_SKDJ_K_Change_1W", "Weekly_SKDJ_D_Change_1W",
    "Weekly_Return_1W_pct", "Weekly_Return_4W_pct", "Weekly_Return_12W_pct",
    "Weekly_MA20_Bias_pct", "Weekly_Volume_Ratio_4_12", "Weekly_Contraction_4_12",
    "Daily_SKDJ_Level_At_Cross", "Daily_SKDJ_K_Change_3D",
    "Daily_MACD_Hist", "Daily_MACD_Hist_Change_1D",
    "Daily_Return_5D_pct", "Daily_Return_20D_pct", "Daily_Return_60D_pct",
    "Daily_MA20_Bias_pct", "Daily_MA60_Bias_pct", "Daily_Volume_Ratio_5_20",
    "Daily_ATR14_pct", "Daily_Amplitude_10D_pct", "Distance_60D_High_pct",
    "Turnover_Rate", "Log_Raw_Price", "Log_Circ_MV", "Candidate_RS20_PctRank", "Candidate_RS60_PctRank",
    "Industry_Signal_Count", "Week_Signal_Count", "Recent_3W_Touched_25_Num",
    "Cross_In_20_35_Num", "Board_Main", "Board_ChiNext", "Board_STAR",
]

FEATURE_LABELS = {
    "Weekly_Cross_Level": "周线金叉位置", "Recent_3W_Min_SKDJ": "近3周SKDJ最低值",
    "Recent_6W_Min_SKDJ": "近6周SKDJ最低值", "Weeks_Since_Touch_25": "距最近触及25周数",
    "Weekly_SKDJ_K_Change_1W": "周K一周变化", "Weekly_SKDJ_D_Change_1W": "周D一周变化",
    "Weekly_Return_1W_pct": "近1周涨幅", "Weekly_Return_4W_pct": "近4周涨幅",
    "Weekly_Return_12W_pct": "近12周涨幅", "Weekly_MA20_Bias_pct": "周线距MA20",
    "Weekly_Volume_Ratio_4_12": "周量4/12周比例", "Weekly_Contraction_4_12": "周振幅4/12周收缩比",
    "Daily_SKDJ_Level_At_Cross": "日线SKDJ位置", "Daily_SKDJ_K_Change_3D": "日K三日变化",
    "Daily_MACD_Hist": "日线MACD柱", "Daily_MACD_Hist_Change_1D": "日线MACD柱一日变化",
    "Daily_Return_5D_pct": "近5日涨幅", "Daily_Return_20D_pct": "近20日涨幅",
    "Daily_Return_60D_pct": "近60日涨幅", "Daily_MA20_Bias_pct": "日线距MA20",
    "Daily_MA60_Bias_pct": "日线距MA60", "Daily_Volume_Ratio_5_20": "日量5/20日比例",
    "Daily_ATR14_pct": "ATR14占价格", "Daily_Amplitude_10D_pct": "近10日振幅",
    "Distance_60D_High_pct": "距60日高点", "Turnover_Rate": "换手率",
    "Log_Raw_Price": "原始股价对数", "Log_Circ_MV": "流通市值对数", "Candidate_RS20_PctRank": "候选内20日相对强度",
    "Candidate_RS60_PctRank": "候选内60日相对强度", "Industry_Signal_Count": "同行业同期信号数",
    "Week_Signal_Count": "全池同期信号数", "Recent_3W_Touched_25_Num": "近3周触及25",
    "Cross_In_20_35_Num": "金叉位于20至35", "Board_Main": "主板",
    "Board_ChiNext": "创业板", "Board_STAR": "科创板",
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
    for source, target in (("Daily_Return_20D_pct", "Candidate_RS20_PctRank"),
                           ("Daily_Return_60D_pct", "Candidate_RS60_PctRank")):
        frame[target] = frame.groupby("Signal_Date")[source].rank(pct=True, method="average")
    simple_parts = []
    for source in ("Daily_Return_5D_pct", "Daily_Volume_Ratio_5_20"):
        simple_parts.append(frame.groupby("Signal_Date")[source].rank(pct=True, method="average"))
    frame["Simple_PriceVolume_Score"] = pd.concat(simple_parts, axis=1).mean(axis=1)
    ret40 = pd.to_numeric(frame["Return_40D_pct"], errors="coerce").clip(-20, 40)
    mae40 = pd.to_numeric(frame["MAE_40D_pct"], errors="coerce").clip(-25, 0)
    frame["Risk_Adjusted_40"] = ret40 + 0.75 * mae40
    frame["Label_High_Return"] = pd.to_numeric(frame["Return_40D_pct"], errors="coerce").ge(10)
    frame["Label_Good_Low_DD"] = frame["Label_High_Return"] & pd.to_numeric(frame["MAE_40D_pct"], errors="coerce").ge(-10)
    frame["Label_Loss"] = pd.to_numeric(frame["Return_40D_pct"], errors="coerce").le(0)
    frame["Label_Severe_DD"] = pd.to_numeric(frame["MAE_40D_pct"], errors="coerce").le(-15)
    frame["Label_Bad"] = frame["Label_Loss"] | frame["Label_Severe_DD"]
    frame["Signal_Date_dt"] = pd.to_datetime(frame["Signal_Date"], format="%Y%m%d", errors="coerce")
    frame["Outcome_40D_End_dt"] = pd.to_datetime(frame["Outcome_40D_End_Date"], format="%Y%m%d", errors="coerce")
    frame["Half_Year"] = frame["Signal_Date_dt"].dt.year.astype("Int64").astype(str) + "H" + np.where(frame["Signal_Date_dt"].dt.month.le(6), "1", "2")
    return frame


def event_stats(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    ret = pd.to_numeric(frame.get("Return_40D_pct"), errors="coerce").dropna()
    mae = pd.to_numeric(frame.get("MAE_40D_pct"), errors="coerce").dropna()
    quality = pd.to_numeric(frame.get("Risk_Adjusted_40"), errors="coerce").dropna()
    return {
        "方法": label, "事件数": len(frame), "信号周": frame["Signal_Date"].nunique() if len(frame) else 0,
        "40日平均收益(%)": ret.mean(), "40日收益中位数(%)": ret.median(),
        "正收益比例(%)": ret.gt(0).mean() * 100 if len(ret) else np.nan,
        "收益≥10%比例(%)": ret.ge(10).mean() * 100 if len(ret) else np.nan,
        "平均MAE(%)": mae.mean(), "MAE≤-15%比例(%)": mae.le(-15).mean() * 100 if len(mae) else np.nan,
        "平均风险调整标签": quality.mean(),
    }


def feature_audit(frame: pd.DataFrame) -> pd.DataFrame:
    mature = frame[frame["Has_40D_Future"].map(to_bool)].copy()
    rows = []
    years = sorted(mature["Signal_Date"].astype(str).str[:4].unique())
    for feature in MODEL_FEATURES:
        values = pd.to_numeric(mature[feature], errors="coerce")
        good = values[mature["Label_Good_Low_DD"].map(to_bool)].dropna()
        bad = values[mature["Label_Bad"].map(to_bool)].dropna()
        valid = values.notna() & pd.to_numeric(mature["Risk_Adjusted_40"], errors="coerce").notna()
        corr = (values[valid].corr(pd.to_numeric(mature.loc[valid, "Risk_Adjusted_40"], errors="coerce"), method="spearman")
                if values[valid].nunique() > 1 else np.nan)
        signs = []
        for year in years:
            mask = mature["Signal_Date"].astype(str).str[:4].eq(year) & valid
            if mask.sum() >= 30 and values[mask].nunique() > 1:
                yearly_corr = values[mask].corr(
                    pd.to_numeric(mature.loc[mask, "Risk_Adjusted_40"], errors="coerce"), method="spearman")
                if math.isfinite(finite_num(yearly_corr)) and yearly_corr != 0:
                    signs.append(np.sign(yearly_corr))
        overall_sign = np.sign(corr) if math.isfinite(finite_num(corr)) else 0
        consistency = np.mean([sign == overall_sign for sign in signs if sign != 0]) * 100 if overall_sign and signs else np.nan
        rows.append({
            "特征": FEATURE_LABELS.get(feature, feature), "字段": feature, "有效样本": int(valid.sum()),
            "优质低回撤组中位数": good.median(), "亏损或大回撤组中位数": bad.median(),
            "中位数差": good.median() - bad.median(), "与风险调整标签Spearman": corr,
            "年度同方向比例(%)": consistency, "年度可比较数": len(signs),
        })
    return pd.DataFrame(rows).sort_values(["年度同方向比例(%)", "与风险调整标签Spearman"], ascending=[False, False])


def feature_quintiles(frame: pd.DataFrame) -> pd.DataFrame:
    mature = frame[frame["Has_40D_Future"].map(to_bool)].copy()
    rows = []
    for feature in MODEL_FEATURES:
        valid = mature[pd.to_numeric(mature[feature], errors="coerce").notna()].copy()
        if len(valid) < 50 or pd.to_numeric(valid[feature], errors="coerce").nunique() < 4:
            continue
        try:
            valid["Feature_Bin"] = pd.qcut(pd.to_numeric(valid[feature], errors="coerce"), 5, duplicates="drop")
        except ValueError:
            continue
        for number, (_, group) in enumerate(valid.groupby("Feature_Bin", observed=True), start=1):
            row = event_stats(group, f"Q{number}")
            rows.append({"特征": FEATURE_LABELS.get(feature, feature), "字段": feature,
                         "分位组": f"Q{number}", "特征最小值": pd.to_numeric(group[feature], errors="coerce").min(),
                         "特征最大值": pd.to_numeric(group[feature], errors="coerce").max(), **row})
    return pd.DataFrame(rows)


def half_year_starts(frame: pd.DataFrame) -> list[pd.Timestamp]:
    dates = frame["Signal_Date_dt"].dropna()
    if dates.empty:
        return []
    first_year, last_year = int(dates.min().year), int(dates.max().year)
    starts = []
    for year in range(first_year, last_year + 1):
        starts.extend([pd.Timestamp(year, 1, 1), pd.Timestamp(year, 7, 1)])
    return [value for value in starts if dates.min() <= value <= dates.max()]


def oos_rank(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = events.copy()
    frame["OOS_Predicted_Quality"] = np.nan
    frame["OOS_Fold"] = ""
    mature = frame[frame["Has_40D_Future"].map(to_bool)].copy()
    fold_rows, importance_rows = [], []
    for test_start in half_year_starts(mature):
        test_end = test_start + pd.offsets.DateOffset(months=6)
        train_mask = mature["Outcome_40D_End_dt"].lt(test_start)
        test_mask = mature["Signal_Date_dt"].ge(test_start) & mature["Signal_Date_dt"].lt(test_end)
        train, test = mature[train_mask].copy(), mature[test_mask].copy()
        fold = f"{test_start.year}H{1 if test_start.month == 1 else 2}"
        if len(train) < MIN_TRAIN_EVENTS or len(test) < MIN_TEST_EVENTS:
            continue
        X_train = train[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
        X_test = test[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
        y_train = pd.to_numeric(train["Risk_Adjusted_40"], errors="coerce")
        week_n = train.groupby("Signal_Date")["ts_code"].transform("size").astype(float)
        stock_n = train.groupby("ts_code")["Signal_Date"].transform("size").astype(float)
        weights = 1.0 / np.sqrt(week_n.clip(lower=1) * stock_n.clip(lower=1))
        model = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=160, max_leaf_nodes=7, max_depth=3,
            min_samples_leaf=30, l2_regularization=8.0, random_state=MODEL_SEED,
        )
        model.fit(X_train, y_train, sample_weight=weights)
        predictions = model.predict(X_test)
        frame.loc[test.index, "OOS_Predicted_Quality"] = predictions
        frame.loc[test.index, "OOS_Fold"] = fold
        actual = pd.to_numeric(test["Risk_Adjusted_40"], errors="coerce")
        fold_rows.append({
            "样本外阶段": fold, "训练事件": len(train), "训练截止结果日期": train["Outcome_40D_End_Date"].max(),
            "测试事件": len(test), "测试信号周": test["Signal_Date"].nunique(),
            "预测与实际Spearman": pd.Series(predictions, index=test.index).corr(actual, method="spearman"),
            "预测均值": float(np.mean(predictions)), "实际风险调整标签均值": actual.mean(),
        })
        try:
            perm = permutation_importance(model, X_test, actual, scoring="neg_mean_absolute_error",
                                          n_repeats=3, random_state=MODEL_SEED, n_jobs=1)
            for feature, mean, std in zip(MODEL_FEATURES, perm.importances_mean, perm.importances_std):
                importance_rows.append({"样本外阶段": fold, "特征": FEATURE_LABELS.get(feature, feature),
                                        "字段": feature, "置换重要性": mean, "重要性标准差": std})
        except Exception as exc:
            record_error(f"{fold}置换重要性失败: {exc}")
    predicted = frame[frame["OOS_Predicted_Quality"].notna()].copy()
    if not predicted.empty:
        predicted["OOS_Weekly_Rank"] = predicted.groupby("Signal_Date")["OOS_Predicted_Quality"].rank(
            method="first", ascending=False)
        frame.loc[predicted.index, "OOS_Weekly_Rank"] = predicted["OOS_Weekly_Rank"]
    else:
        frame["OOS_Weekly_Rank"] = np.nan
    return frame, pd.DataFrame(fold_rows), pd.DataFrame(importance_rows)


def selection_comparison(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mature = events[events["Has_40D_Future"].map(to_bool)].copy()
    oos = mature[mature["OOS_Predicted_Quality"].notna()].copy()
    if not oos.empty:
        oos["OOS_Weekly_Rank"] = oos.groupby("Signal_Date")["OOS_Predicted_Quality"].rank(method="first", ascending=False)
        oos["Simple_Weekly_Rank"] = oos.groupby("Signal_Date")["Simple_PriceVolume_Score"].rank(method="first", ascending=False)
    selections = {
        "样本外阶段全部事件": oos,
        "原核心条件全部事件": oos[oos["Bottom_Reset_Core"].map(to_bool)],
        "全量简单涨幅量能Top3": oos[oos["Simple_Weekly_Rank"].le(3)],
        "模型Top1": oos[oos["OOS_Weekly_Rank"].le(1)],
        "模型Top3": oos[oos["OOS_Weekly_Rank"].le(3)],
        "模型Top3且预测质量>0": oos[oos["OOS_Weekly_Rank"].le(3) & oos["OOS_Predicted_Quality"].gt(0)],
    }
    summary = pd.DataFrame([event_stats(group, label) for label, group in selections.items()])
    detail_frames = []
    for label, group in selections.items():
        if group.empty:
            continue
        part = group.copy()
        part.insert(0, "选择方法", label)
        detail_frames.append(part)
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary, detail


def yearly_selection_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for (method, year), group in detail.groupby(["选择方法", detail["Signal_Date"].astype(str).str[:4]]):
        row = event_stats(group, method)
        row["年份"] = year
        rows.append(row)
    return pd.DataFrame(rows)


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
    st.set_page_config(page_title="周线SKDJ全量特征排序 V3.4", layout="wide")
    st.title(TITLE)
    st.caption("本版不再用两个经验阈值提前剔除股票；先保留全部金叉，再用严格时间样本外结果检验特征与排序。")
    with st.expander("冻结规则与防前视设计", expanded=True):
        st.markdown(f"""
- **唯一候选信号**：完整周线SKDJ金叉，参数冻结 `N={SKDJ_N}, M={SKDJ_M}`。
- **原核心条件降级为特征**：最近{RESET_LOOKBACK_WEEKS}周K或D曾≤{SKDJ_BOTTOM:.0f}；金叉位置>{CROSS_ZONE_LOW:.0f}且≤{CROSS_ZONE_HIGH:.0f}。
- **股票池**：历史科技池，信号日原始股价≥10元、流通市值≥100亿元。
- **统一买入**：周线收盘确认，下一市场交易日开盘成交；不等待日线指标。
- **统一判卷**：固定20/40市场交易日收益、MFE、MAE；风险标签=`截尾40日收益 + 0.75×截尾MAE`。
- **样本外训练**：每半年重训；训练样本的40日结果必须在测试期开始前已经结束，禁止随机拆分。
- **排序评价**：原核心、简单涨幅量能Top3、模型Top1/Top3及允许空仓Top3；本版不生成资金曲线。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("信号开始", date(2023, 6, 5), key="v34_start")
        signal_end_date = st.date_input("信号截止", date(2026, 6, 5), key="v34_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v34_market_end")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v34_pause")
        use_cache = st.checkbox("复用逐股票缓存", True, key="v34_cache")
        st.divider()
        commission_pct = st.number_input("佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f")
        stamp_duty_pct = st.number_input("卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f")
        transfer_fee_pct = st.number_input("过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f")
        if st.button("清除本程序行情缓存", key="v34_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("缓存已清除")
    token = st.text_input("Tushare Token", type="password", key="v34_token")
    session_key = "weekly_skdj_feature_rank_v34_zip"
    if not token:
        st.info("请输入Tushare Token；相同日期的V3.2/V3.3逐股票行情缓存可直接复用。")
        return
    if not st.button("开始V3.4全量特征与样本外排序", type="primary", key="v34_run"):
        if session_key in st.session_state:
            st.download_button("下载上一次结果ZIP", st.session_state[session_key],
                               file_name="weekly_skdj_all_event_oos_rank_v3_4_all_results.zip",
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
        "min_price": 10.0, "min_mv": 100.0, "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct), "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct), "rejects": rejects,
    }
    try:
        with st.spinner("加载交易日历和历史科技股池..."):
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
    population = stocks.groupby("Sample_Board").size().reindex(BOARDS, fill_value=0).rename("股票数").reset_index()
    open_pos = {day: position for position, day in enumerate(open_dates)}
    events: list[dict[str, Any]] = []
    cache_hits = data_failures = 0
    progress, status = st.progress(0.0), st.empty()
    for number, stock in stocks.iterrows():
        code = str(stock["ts_code"])
        progress.progress((number + 1) / max(len(stocks), 1), text=f"{number + 1}/{len(stocks)} {code}")
        status.caption(f"全量金叉 {len(events)}；缓存 {cache_hits}；失败 {data_failures}")
        daily, daily_basic, cache_hit = fetch_stock_history(code, preload, market_end, bool(use_cache), float(pause))
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        events.extend(analyze_stock(stock, period_index.get(code, []), daily, daily_basic,
                                    week_last_map, open_dates, open_pos, config))
    progress.empty()
    status.empty()
    if not events:
        st.error("研究区间没有生成符合历史科技池、价格和市值条件的完整周线SKDJ金叉。")
        return
    try:
        with st.spinner("计算结果标签、特征审计和严格样本外排序..."):
            event_frame = add_cross_section_features(
                pd.DataFrame(events).sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True))
            event_frame, folds, importance = oos_rank(event_frame)
            comparison, selection_detail = selection_comparison(event_frame)
            yearly = yearly_selection_summary(selection_detail)
            audit = feature_audit(event_frame)
            quintiles = feature_quintiles(event_frame)
            calendar = pool_calendar(open_dates, signal_start, signal_end, event_frame)
    except Exception as exc:
        st.exception(exc)
        return
    mature = event_frame[event_frame["Has_40D_Future"].map(to_bool)]
    core = event_frame[event_frame["Bottom_Reset_Core"].map(to_bool)]
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "信号开始": signal_start, "信号截止": signal_end,
        "观察截止": market_end, "全部周线SKDJ金叉": len(event_frame), "40日成熟事件": len(mature),
        "原核心事件": len(core), "不同股票": event_frame["ts_code"].nunique(), "自然周": len(calendar),
        "全量平均每周": len(event_frame) / len(calendar) if len(calendar) else np.nan,
        "全量空窗周": int(calendar["All_Empty"].sum()), "原核心空窗周": int(calendar["Original_Core_Empty"].sum()),
        "样本外预测事件": int(event_frame["OOS_Predicted_Quality"].notna().sum()),
        "样本外阶段数": len(folds), "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    if not importance.empty:
        importance_summary = importance.groupby(["字段", "特征"], as_index=False).agg(
            平均置换重要性=("置换重要性", "mean"), 正重要性阶段数=("置换重要性", lambda x: int((x > 0).sum())),
            总阶段数=("置换重要性", "size"), 重要性标准差=("置换重要性", "std"),
        ).sort_values("平均置换重要性", ascending=False)
    else:
        importance_summary = pd.DataFrame()
    metadata = pd.DataFrame([
        ("候选池", "全部完整周线SKDJ金叉；近3周触及25和20-35金叉不再硬剔除"),
        ("原核心基准", "近3个完整周K或D最低值≤25，且金叉当周(K+D)/2>20且≤35"),
        ("SKDJ参数", f"N={SKDJ_N},M={SKDJ_M}，冻结不寻优"),
        ("价格市值", "信号日原始收盘价≥10元；历史流通市值≥100亿元"),
        ("买入", "完整周线收盘确认，下一市场交易日开盘；主板一字板不买"),
        ("成本", "买卖滑点、双边佣金与过户费、卖出印花税均计入固定终点收益"),
        ("风险调整标签", "clip(40日收益,-20,40)+0.75*clip(40日MAE,-25,0)"),
        ("优质低回撤", "40日收益≥10%且40日MAE≥-10%"),
        ("坏事件", "40日收益≤0或40日MAE≤-15%"),
        ("样本外", f"半年扩展窗口；至少{MIN_TRAIN_EVENTS}个训练事件；训练结果必须在测试期前成熟"),
        ("模型", "小型HistGradientBoostingRegressor；拥挤周和重复股票降权；不使用未来特征"),
        ("简单基准", "同一信号周内，近5日涨幅百分位与5/20日量能百分位各50%"),
        ("Top3说明", "事件级选择质量审计，不是持仓重叠后的30万元三仓资金曲线"),
        ("资金约束备忘", f"后续组合仍为{INITIAL_CAPITAL:.0f}元、最多{MAX_POSITIONS}仓、每仓约{POSITION_BUDGET:.0f}元"),
    ], columns=["项目", "值"])
    files = {
        "01_run_summary_v3_4.csv": run_summary,
        "02_oos_selection_comparison_v3_4.csv": comparison,
        "03_oos_yearly_stability_v3_4.csv": yearly,
        "04_oos_fold_diagnostics_v3_4.csv": folds,
        "05_feature_good_bad_audit_v3_4.csv": audit,
        "06_feature_quintiles_v3_4.csv": quintiles,
        "07_oos_feature_importance_summary_v3_4.csv": importance_summary,
        "08_oos_feature_importance_by_fold_v3_4.csv": importance,
        "09_oos_selected_event_detail_v3_4.csv": selection_detail,
        "10_all_weekly_skdj_event_features_v3_4.csv": event_frame,
        "11_weekly_pool_calendar_v3_4.csv": calendar,
        "12_full_tech_universe_v3_4.csv": stocks,
        "13_board_population_v3_4.csv": population,
        "14_rejection_audit_v3_4.csv": pd.DataFrame([{"剔除原因": k, "次数": v} for k, v in sorted(rejects.items())]),
        "15_api_errors_v3_4.csv": pd.DataFrame({"错误": API_ERRORS}),
        "16_metadata_v3_4.csv": metadata,
    }
    result_zip = make_zip(files)
    st.session_state[session_key] = result_zip
    st.success(f"完成：全量金叉{len(event_frame)}个；原核心{len(core)}个；样本外预测{event_frame['OOS_Predicted_Quality'].notna().sum()}个。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("全部金叉", len(event_frame))
    c2.metric("40日成熟", len(mature))
    c3.metric("原核心事件", len(core))
    c4.metric("样本外阶段", len(folds))
    st.subheader("样本外选择方法比较")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.subheader("样本外阶段诊断")
    st.dataframe(folds, use_container_width=True, hide_index=True)
    st.subheader("高收益低回撤与亏损/大回撤特征差异")
    st.dataframe(audit.head(20), use_container_width=True, hide_index=True)
    st.subheader("样本外特征重要性")
    st.dataframe(importance_summary.head(20), use_container_width=True, hide_index=True)
    st.download_button("下载V3.4全部结果ZIP", result_zip,
                       file_name="weekly_skdj_all_event_oos_rank_v3_4_all_results.zip",
                       mime="application/zip", type="primary", key="v34_download", on_click="ignore")
    st.info("先看02判断模型Top3是否同时改善收益中位数与MAE；再看03确认跨年稳定；只有05、06、07方向一致的特征，才有资格进入下一版可解释评分卡。")


if __name__ == "__main__":
    main()
