# -*- coding: utf-8 -*-
"""
科技股周线SKDJ核心池：市场开仓过滤与单因子排序审计 V3.5

目的：
1. 恢复“近3个完整周触及25且金叉位置20~35”为唯一核心候选池。
2. 将“本周是否开仓”和“同周买哪只”拆开，避免把市场环境误当成个股排序能力。
3. 市场层只审计预先冻结的中证500、创业板指、科创50趋势及科技池宽度。
4. 排序层只在同周核心池内比较单因子Top1/Top2/Top3，并与同周随机TopK公平比较。
5. 所有事件统一在完整周信号确认后的下一市场交易日开盘买入，计算20/40日收益、MFE与MAE。

注意：V3.5规则是在V3.4结果上提出，复跑同一三年不是新的独立样本外检验。
本版是事件级审计，不是持仓重叠后的30万元三仓资金曲线。

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
TITLE = "科技股周线SKDJ核心池：市场开仓过滤与单因子排序审计 V3.5"
VERSION = "V3.5-WEEKLY-SKDJ-MARKET-GATE-SINGLE-FACTOR-RANK"
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
RANDOM_RUNS = 1000

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

INDEX_SPECS = {
    "CSI500": ("000905.SH", "中证500"),
    "ChiNext": ("399006.SZ", "创业板指"),
    "STAR50": ("000688.SH", "科创50"),
}

# 方向在运行前冻结；每个因子单独比较，禁止事后拼权重。
RANK_FACTORS = {
    "Simple_PriceVolume_Score": ("涨幅量能基准", False),
    "Daily_SKDJ_Level_At_Cross": ("日线SKDJ位置", False),
    "Daily_Return_60D_pct": ("60日相对强度", False),
    "Daily_Volume_Ratio_5_20": ("日量5/20比例", False),
    "Weekly_Cross_Level": ("周线金叉位置", False),
    "Recent_3W_Min_SKDJ": ("近3周SKDJ最低值", False),
    "Distance_60D_High_pct": ("距60日高点", False),
    "Raw_Close": ("原始股价", False),
    "Turnover_Rate": ("换手率", False),
    "Daily_MACD_Hist_Change_1D": ("MACD柱一日变化（低优先诊断）", True),
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


def index_cache_path(start_date: str, end_date: str) -> str:
    return os.path.join(CACHE_DIR, f"market_indices_v3_5_{start_date}_{end_date}.pkl")


def load_market_indices(start_date: str, end_date: str, use_cache: bool,
                        api_pause: float) -> dict[str, pd.DataFrame]:
    path = index_cache_path(start_date, end_date)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            record_error(f"指数缓存损坏: {exc}")
    result: dict[str, pd.DataFrame] = {}
    for prefix, (code, name) in INDEX_SPECS.items():
        frame = safe_get(
            "index_daily", ts_code=code, start_date=start_date, end_date=end_date,
            fields="ts_code,trade_date,open,high,low,close,vol",
        )
        time.sleep(api_pause)
        if frame.empty:
            record_error(f"{name}({code})行情为空")
            continue
        frame["trade_date"] = frame["trade_date"].astype(str)
        for column in ("open", "high", "low", "close", "vol"):
            frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
        frame = frame.dropna(subset=["trade_date", "close"]).drop_duplicates(
            "trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)
        result[prefix] = frame
    if use_cache and result:
        atomic_pickle(result, path)
    return result


def build_index_context(indices: dict[str, pd.DataFrame], signal_dates: list[str]) -> pd.DataFrame:
    context = pd.DataFrame({"Signal_Date": sorted(set(signal_dates))})
    for prefix, (_, name) in INDEX_SPECS.items():
        frame = indices.get(prefix, pd.DataFrame()).copy()
        if frame.empty:
            continue
        close = frame["close"]
        ma20, ma60 = close.rolling(20).mean(), close.rolling(60).mean()
        dif = close.ewm(span=12, adjust=False, min_periods=1).mean() - close.ewm(
            span=26, adjust=False, min_periods=1).mean()
        dea = dif.ewm(span=9, adjust=False, min_periods=1).mean()
        frame[f"{prefix}_Name"] = name
        frame[f"{prefix}_Return20_pct"] = close.pct_change(20, fill_method=None) * 100.0
        frame[f"{prefix}_Return60_pct"] = close.pct_change(60, fill_method=None) * 100.0
        frame[f"{prefix}_MA20_Bias_pct"] = (close / ma20 - 1.0) * 100.0
        frame[f"{prefix}_MA60_Bias_pct"] = (close / ma60 - 1.0) * 100.0
        frame[f"{prefix}_MA20_Slope10_pct"] = (ma20 / ma20.shift(10) - 1.0) * 100.0
        frame[f"{prefix}_MACD_Hist"] = 2.0 * (dif - dea)
        columns = ["trade_date"] + [column for column in frame.columns if column.startswith(f"{prefix}_")]
        piece = frame[columns].rename(columns={"trade_date": "Signal_Date"})
        context = context.merge(piece, on="Signal_Date", how="left")
    above20, above60, rising20, positive20 = [], [], [], []
    for prefix in INDEX_SPECS:
        if f"{prefix}_MA20_Bias_pct" in context:
            above20.append(pd.to_numeric(context[f"{prefix}_MA20_Bias_pct"], errors="coerce").gt(0))
            above60.append(pd.to_numeric(context[f"{prefix}_MA60_Bias_pct"], errors="coerce").gt(0))
            rising20.append(pd.to_numeric(context[f"{prefix}_MA20_Slope10_pct"], errors="coerce").gt(0))
            positive20.append(pd.to_numeric(context[f"{prefix}_Return20_pct"], errors="coerce").gt(0))
    if above20:
        context["Index_Available_Count"] = len(above20)
        context["Index_Above_MA20_Count"] = pd.concat(above20, axis=1).sum(axis=1)
        context["Index_Above_MA60_Count"] = pd.concat(above60, axis=1).sum(axis=1)
        context["Index_MA20_Rising_Count"] = pd.concat(rising20, axis=1).sum(axis=1)
        context["Index_Positive_20D_Count"] = pd.concat(positive20, axis=1).sum(axis=1)
        return20 = [pd.to_numeric(context[f"{prefix}_Return20_pct"], errors="coerce")
                    for prefix in INDEX_SPECS if f"{prefix}_Return20_pct" in context]
        context["Index_Composite_Return20_pct"] = pd.concat(return20, axis=1).mean(axis=1)
    else:
        context["Index_Available_Count"] = 0
    return context


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


def update_tech_breadth(stock: pd.Series, periods: list[dict[str, str]], daily: pd.DataFrame,
                        daily_basic: pd.DataFrame, signal_week_dates: set[str],
                        config: dict[str, Any]) -> None:
    """按股票逐只累加周末科技池宽度，只使用当日及以前数据。"""
    selected = daily[daily["trade_date"].astype(str).isin(signal_week_dates)].copy()
    if selected.empty or daily_basic.empty:
        return
    basic = daily_basic[["trade_date", "close", "circ_mv"]].copy().rename(
        columns={"close": "Raw_Close_Breadth", "circ_mv": "Circ_MV_Breadth"})
    selected = selected.merge(basic, on="trade_date", how="left")
    selected = selected[
        pd.to_numeric(selected["Raw_Close_Breadth"], errors="coerce").ge(config["min_price"])
        & pd.to_numeric(selected["Circ_MV_Breadth"], errors="coerce").ge(config["min_mv"] * 10000.0)
    ]
    accumulator = config["breadth_accumulator"]
    for row in selected.itertuples(index=False):
        signal_date = str(row.trade_date)
        if not (str(stock["list_date"]) <= signal_date < str(stock["delist_date"])):
            continue
        if membership_on_date(periods, signal_date) is None:
            continue
        item = accumulator.setdefault(signal_date, {
            "Tech_Eligible_Count": 0, "Tech_Above_MA20_Count": 0,
            "Tech_Above_MA60_Count": 0, "Tech_Positive_20D_Count": 0,
            "Tech_Positive_60D_Count": 0, "Tech_MACD_Red_Count": 0,
        })
        item["Tech_Eligible_Count"] += 1
        item["Tech_Above_MA20_Count"] += int(finite_num(row.D_MA20_Bias_pct) > 0)
        item["Tech_Above_MA60_Count"] += int(finite_num(row.D_MA60_Bias_pct) > 0)
        item["Tech_Positive_20D_Count"] += int(finite_num(row.D_Return_20D_pct) > 0)
        item["Tech_Positive_60D_Count"] += int(finite_num(row.D_Return_60D_pct) > 0)
        item["Tech_MACD_Red_Count"] += int(finite_num(row.D_MACD_Hist) > 0)


def build_tech_breadth(accumulator: dict[str, dict[str, int]],
                       signal_dates: list[str]) -> pd.DataFrame:
    rows = []
    for signal_date in sorted(set(signal_dates)):
        item = dict(accumulator.get(signal_date, {}))
        eligible = int(item.get("Tech_Eligible_Count", 0))
        row: dict[str, Any] = {"Signal_Date": signal_date, "Tech_Eligible_Count": eligible}
        for count_column, pct_column in (
            ("Tech_Above_MA20_Count", "Tech_Above_MA20_pct"),
            ("Tech_Above_MA60_Count", "Tech_Above_MA60_pct"),
            ("Tech_Positive_20D_Count", "Tech_Positive_20D_pct"),
            ("Tech_Positive_60D_Count", "Tech_Positive_60D_pct"),
            ("Tech_MACD_Red_Count", "Tech_MACD_Red_pct"),
        ):
            count = int(item.get(count_column, 0))
            row[count_column] = count
            row[pct_column] = count / eligible * 100.0 if eligible else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


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
    daily = add_daily_features(daily_raw)
    update_tech_breadth(stock, periods, daily, daily_basic, config["signal_week_dates"], config)
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


def add_market_context(events: pd.DataFrame, index_context: pd.DataFrame,
                       breadth: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    core_counts = frame[frame["Bottom_Reset_Core"].map(to_bool)].groupby("Signal_Date").size()
    frame["Core_Signal_Count"] = frame["Signal_Date"].map(core_counts).fillna(0).astype(float)
    frame = frame.merge(index_context, on="Signal_Date", how="left")
    frame = frame.merge(breadth, on="Signal_Date", how="left")
    for column in ("Index_Available_Count", "Index_Above_MA20_Count", "Index_MA20_Rising_Count",
                   "Tech_Eligible_Count", "Tech_Above_MA20_pct"):
        if column not in frame:
            frame[column] = np.nan
    required = (
        pd.to_numeric(frame.get("Index_Available_Count"), errors="coerce").ge(len(INDEX_SPECS))
        & pd.to_numeric(frame.get("Tech_Eligible_Count"), errors="coerce").gt(0)
    )
    strong = (
        pd.to_numeric(frame.get("Index_Above_MA20_Count"), errors="coerce").ge(2)
        & pd.to_numeric(frame.get("Index_MA20_Rising_Count"), errors="coerce").ge(2)
        & pd.to_numeric(frame.get("Tech_Above_MA20_pct"), errors="coerce").ge(50)
    )
    weak = (
        pd.to_numeric(frame.get("Index_Above_MA20_Count"), errors="coerce").le(1)
        & pd.to_numeric(frame.get("Index_MA20_Rising_Count"), errors="coerce").le(1)
        & pd.to_numeric(frame.get("Tech_Above_MA20_pct"), errors="coerce").lt(40)
    )
    frame["Market_State"] = np.select(
        [~required, strong, weak], ["数据不足", "强势", "弱势"], default="中性")
    frame["Market_Gate_Strong"] = required & strong
    frame["Market_Gate_Not_Weak"] = required & ~weak
    return frame


def week_equal_stats(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"周等权平均收益(%)": np.nan, "周等权收益中位数(%)": np.nan,
                "周等权平均MAE(%)": np.nan, "周等权风险调整标签": np.nan}
    weekly = frame.groupby("Signal_Date", as_index=False).agg(
        Weekly_Return=("Return_40D_pct", "mean"), Weekly_MAE=("MAE_40D_pct", "mean"),
        Weekly_Quality=("Risk_Adjusted_40", "mean"),
    )
    return {
        "周等权平均收益(%)": pd.to_numeric(weekly["Weekly_Return"], errors="coerce").mean(),
        "周等权收益中位数(%)": pd.to_numeric(weekly["Weekly_Return"], errors="coerce").median(),
        "周等权平均MAE(%)": pd.to_numeric(weekly["Weekly_MAE"], errors="coerce").mean(),
        "周等权风险调整标签": pd.to_numeric(weekly["Weekly_Quality"], errors="coerce").mean(),
    }


def market_gate_summary(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mature = core[core["Has_40D_Future"].map(to_bool)].copy()
    groups: dict[str, pd.DataFrame] = {
        "核心池全部": mature,
        "冻结强势门槛": mature[mature["Market_Gate_Strong"].map(to_bool)],
        "冻结非弱势门槛": mature[mature["Market_Gate_Not_Weak"].map(to_bool)],
    }
    for state in ("强势", "中性", "弱势", "数据不足"):
        groups[f"市场状态={state}"] = mature[mature["Market_State"].eq(state)]
    rows = []
    for label, group in groups.items():
        rows.append({**event_stats(group, label), **week_equal_stats(group)})
    yearly = []
    for label, group in groups.items():
        for year, piece in group.groupby(group["Signal_Date"].astype(str).str[:4]):
            yearly.append({"年份": year, **event_stats(piece, label), **week_equal_stats(piece)})
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def market_feature_bins(core: pd.DataFrame) -> pd.DataFrame:
    mature = core[core["Has_40D_Future"].map(to_bool)].copy()
    weekly = mature.groupby("Signal_Date", as_index=False).agg(
        Core_Events=("ts_code", "size"), Weekly_Return=("Return_40D_pct", "mean"),
        Weekly_MAE=("MAE_40D_pct", "mean"), Weekly_Quality=("Risk_Adjusted_40", "mean"),
    )
    market_columns = [
        "Index_Composite_Return20_pct", "Index_Above_MA20_Count", "Index_MA20_Rising_Count",
        "Tech_Above_MA20_pct", "Tech_Above_MA60_pct", "Tech_Positive_20D_pct",
        "Tech_MACD_Red_pct", "Core_Signal_Count", "Week_Signal_Count",
    ]
    available = ["Signal_Date"] + [column for column in market_columns if column in mature.columns]
    weekly = weekly.merge(mature[available].drop_duplicates("Signal_Date"), on="Signal_Date", how="left")
    rows = []
    for feature in market_columns:
        if feature not in weekly:
            continue
        valid = weekly[pd.to_numeric(weekly[feature], errors="coerce").notna()].copy()
        if len(valid) < 20 or pd.to_numeric(valid[feature], errors="coerce").nunique() < 3:
            continue
        try:
            valid["Bin"] = pd.qcut(pd.to_numeric(valid[feature], errors="coerce"), 5, duplicates="drop")
        except ValueError:
            continue
        for number, (_, group) in enumerate(valid.groupby("Bin", observed=True), start=1):
            rows.append({
                "市场特征": feature, "分位组": f"Q{number}", "信号周": len(group),
                "特征最小值": pd.to_numeric(group[feature], errors="coerce").min(),
                "特征最大值": pd.to_numeric(group[feature], errors="coerce").max(),
                "平均核心候选数": group["Core_Events"].mean(),
                "周等权平均收益(%)": group["Weekly_Return"].mean(),
                "周等权收益中位数(%)": group["Weekly_Return"].median(),
                "周等权平均MAE(%)": group["Weekly_MAE"].mean(),
                "周等权风险调整标签": group["Weekly_Quality"].mean(),
            })
    return pd.DataFrame(rows)


def rank_core_events(core: pd.DataFrame) -> pd.DataFrame:
    frame = core.copy()
    for feature, (_, ascending) in RANK_FACTORS.items():
        values = pd.to_numeric(frame.get(feature), errors="coerce")
        frame[f"Rank__{feature}"] = values.groupby(frame["Signal_Date"]).rank(
            method="first", ascending=ascending, na_option="bottom")
    return frame


def random_topk_distribution(frame: pd.DataFrame, k: int, runs: int,
                             seed: int) -> pd.DataFrame:
    work = frame.reset_index(drop=True)
    groups = [group.index.to_numpy() for _, group in work.groupby("Signal_Date") if len(group)]
    if not groups:
        return pd.DataFrame()
    returns = pd.to_numeric(work["Return_40D_pct"], errors="coerce").to_numpy(dtype=float)
    maes = pd.to_numeric(work["MAE_40D_pct"], errors="coerce").to_numpy(dtype=float)
    qualities = pd.to_numeric(work["Risk_Adjusted_40"], errors="coerce").to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for run in range(runs):
        chosen = np.concatenate([rng.choice(index, size=min(k, len(index)), replace=False) for index in groups])
        ret, mae, quality = returns[chosen], maes[chosen], qualities[chosen]
        rows.append({
            "run": run, "Mean_Return": np.nanmean(ret), "Median_Return": np.nanmedian(ret),
            "Positive_Rate": np.nanmean(ret > 0) * 100.0, "Mean_MAE": np.nanmean(mae),
            "Severe_MAE_Rate": np.nanmean(mae <= -15) * 100.0,
            "Mean_Quality": np.nanmean(quality),
        })
    return pd.DataFrame(rows)


def percentile_vs_random(distribution: pd.DataFrame, column: str, actual: float,
                         higher_is_better: bool = True) -> float:
    values = pd.to_numeric(distribution.get(column), errors="coerce").dropna()
    if values.empty or not math.isfinite(finite_num(actual)):
        return np.nan
    return float((values.le(actual) if higher_is_better else values.ge(actual)).mean() * 100.0)


def factor_rank_audit(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mature = core[core["Has_40D_Future"].map(to_bool)].copy()
    ranked = rank_core_events(mature)
    overall_rows, yearly_rows, random_rows, detail_rows = [], [], [], []
    scopes: list[tuple[str, pd.DataFrame]] = [("全部", ranked)]
    scopes.extend((str(year), group) for year, group in ranked.groupby(ranked["Signal_Date"].astype(str).str[:4]))
    random_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for scope_number, (scope, scope_frame) in enumerate(scopes, start=1):
        for k in (1, 2, 3):
            distribution = random_topk_distribution(
                scope_frame, k, RANDOM_RUNS, RANDOM_SEED + scope_number * 100 + k)
            random_cache[(scope, k)] = distribution
            random_rows.append({
                "范围": scope, "排序因子": "同周随机基准", "排序字段": "无", "TopK": k,
                "随机次数": len(distribution),
                "随机平均收益均值(%)": distribution.get("Mean_Return", pd.Series(dtype=float)).mean(),
                "随机平均收益P95(%)": distribution.get("Mean_Return", pd.Series(dtype=float)).quantile(0.95),
                "随机风险调整均值": distribution.get("Mean_Quality", pd.Series(dtype=float)).mean(),
                "随机风险调整P95": distribution.get("Mean_Quality", pd.Series(dtype=float)).quantile(0.95),
            })
    for feature, (label, _) in RANK_FACTORS.items():
        for k in (1, 2, 3):
            selected_all = ranked[pd.to_numeric(ranked[f"Rank__{feature}"], errors="coerce").le(k)].copy()
            if not selected_all.empty:
                detail = selected_all.copy()
                detail.insert(0, "TopK", k)
                detail.insert(0, "排序字段", feature)
                detail.insert(0, "排序因子", label)
                detail_rows.append(detail)
            for scope, scope_frame in scopes:
                selected = scope_frame[pd.to_numeric(scope_frame[f"Rank__{feature}"], errors="coerce").le(k)]
                if selected.empty:
                    continue
                actual = event_stats(selected, f"{label} Top{k}")
                distribution = random_cache[(scope, k)]
                row = {
                    "范围": scope, "排序因子": label, "排序字段": feature, "TopK": k, **actual,
                    "平均收益随机百分位(%)": percentile_vs_random(
                        distribution, "Mean_Return", actual["40日平均收益(%)"]),
                    "收益中位数随机百分位(%)": percentile_vs_random(
                        distribution, "Median_Return", actual["40日收益中位数(%)"]),
                    "平均MAE随机百分位(%)": percentile_vs_random(
                        distribution, "Mean_MAE", actual["平均MAE(%)"]),
                    "风险调整随机百分位(%)": percentile_vs_random(
                        distribution, "Mean_Quality", actual["平均风险调整标签"]),
                }
                (overall_rows if scope == "全部" else yearly_rows).append(row)
    detail_frame = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return pd.DataFrame(overall_rows), pd.DataFrame(yearly_rows), pd.DataFrame(random_rows), detail_frame


def factor_market_state_summary(selection_detail: pd.DataFrame) -> pd.DataFrame:
    if selection_detail.empty:
        return pd.DataFrame()
    rows = []
    for (factor, field, k, state), group in selection_detail.groupby(
            ["排序因子", "排序字段", "TopK", "Market_State"], dropna=False):
        rows.append({"排序因子": factor, "排序字段": field, "TopK": k,
                     "市场状态": state, **event_stats(group, str(factor))})
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
    st.set_page_config(page_title="周线SKDJ市场过滤与排序 V3.5", layout="wide")
    st.title(TITLE)
    st.caption("先判断这一周是否值得开仓，再比较同一周核心候选买哪只；不再使用V3.4失败的机器学习排名。")
    with st.expander("冻结规则与本版边界", expanded=True):
        st.markdown(f"""
- **核心候选硬条件**：完整周线SKDJ金叉；最近{RESET_LOOKBACK_WEEKS}个完整周K或D曾≤{SKDJ_BOTTOM:.0f}；金叉位置>{CROSS_ZONE_LOW:.0f}且≤{CROSS_ZONE_HIGH:.0f}。
- **股票池**：历史科技池，主板/创业板/科创板，排除北交所；信号日原始股价≥10元、流通市值≥100亿元。
- **买入与判卷**：周线收盘确认后下一市场交易日开盘；固定20/40日收益、MFE、MAE并计交易成本。
- **市场层**：中证500、创业板指、科创50，加历史科技池MA20/MA60宽度；门槛在运行前冻结，不按结果寻优。
- **冻结强势门槛**：3个指数数据齐全，其中至少2个站上MA20且MA20上升；科技池站上MA20比例≥50%。
- **冻结弱势定义**：至多1个指数站上MA20且MA20上升；科技池站上MA20比例<40%。
- **排序层**：只在核心池同一信号周内逐个比较单因子Top1/Top2/Top3，并进行{RANDOM_RUNS}次同周随机TopK基准。
- **重要限制**：规则来自V3.4同一段历史的发现，因此本次属于复核审计，不应称为新的样本外证明，也不生成资金曲线。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("信号开始", date(2023, 6, 5), key="v35_start")
        signal_end_date = st.date_input("信号截止", date(2026, 6, 5), key="v35_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v35_market_end")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v35_pause")
        use_cache = st.checkbox("复用逐股票与指数缓存", True, key="v35_cache")
        st.divider()
        commission_pct = st.number_input("佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f")
        stamp_duty_pct = st.number_input("卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f")
        transfer_fee_pct = st.number_input("过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f")
        if st.button("清除本程序行情缓存", key="v35_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("缓存已清除")
    token = st.text_input("Tushare Token", type="password", key="v35_token")
    session_key = "weekly_skdj_market_gate_rank_v35_zip"
    if not token:
        st.info("请输入Tushare Token；相同日期范围的旧版逐股票缓存可以直接复用。")
        return
    if not st.button("开始V3.5双层验证", type="primary", key="v35_run"):
        if session_key in st.session_state:
            st.download_button(
                "下载上一次结果ZIP", st.session_state[session_key],
                file_name="weekly_skdj_market_gate_rank_v3_5_all_results.zip",
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
    breadth_accumulator: dict[str, dict[str, int]] = {}
    try:
        with st.spinner("加载交易日历、历史科技池与三只市场指数..."):
            open_dates = load_trade_calendar(preload, market_end)
            extended_end = (market_end_date + timedelta(days=7)).strftime("%Y%m%d")
            full_open_dates = load_trade_calendar(preload, extended_end)
            week_last_map = complete_week_last_dates(full_open_dates)
            signal_week_dates = {
                str(day) for day in week_last_map.values() if signal_start <= str(day) <= signal_end
            }
            stock_basic = load_stock_basic()
            memberships = load_tech_memberships(float(pause))
            indices = load_market_indices(preload, market_end, bool(use_cache), float(pause))
            index_context = build_index_context(indices, sorted(signal_week_dates))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return
    config = {
        "signal_start": signal_start, "signal_end": signal_end, "market_end": market_end,
        "min_price": 10.0, "min_mv": 100.0, "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "commission_pct": float(commission_pct), "stamp_duty_pct": float(stamp_duty_pct),
        "transfer_fee_pct": float(transfer_fee_pct), "rejects": rejects,
        "signal_week_dates": signal_week_dates, "breadth_accumulator": breadth_accumulator,
    }
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
        status.caption(f"全部金叉 {len(events)}；缓存 {cache_hits}；失败 {data_failures}")
        daily, daily_basic, cache_hit = fetch_stock_history(
            code, preload, market_end, bool(use_cache), float(pause))
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        events.extend(analyze_stock(
            stock, period_index.get(code, []), daily, daily_basic,
            week_last_map, open_dates, open_pos, config,
        ))
    progress.empty()
    status.empty()
    if not events:
        st.error("研究区间没有生成符合历史科技池、价格和市值条件的完整周线SKDJ金叉。")
        return
    try:
        with st.spinner("计算市场环境、核心池、单因子TopK和随机基准..."):
            event_frame = add_cross_section_features(
                pd.DataFrame(events).sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True))
            breadth = build_tech_breadth(breadth_accumulator, sorted(signal_week_dates))
            event_frame = add_market_context(event_frame, index_context, breadth)
            core = event_frame[event_frame["Bottom_Reset_Core"].map(to_bool)].copy()
            core_ranked = rank_core_events(core)
            market_summary, market_yearly = market_gate_summary(core_ranked)
            market_bins = market_feature_bins(core_ranked)
            factor_summary, factor_yearly, random_summary, selection_detail = factor_rank_audit(core_ranked)
            factor_by_market = factor_market_state_summary(selection_detail)
            calendar = pool_calendar(open_dates, signal_start, signal_end, event_frame)
    except Exception as exc:
        st.exception(exc)
        return
    mature_core = core_ranked[core_ranked["Has_40D_Future"].map(to_bool)]
    weekly_market = core_ranked.drop_duplicates("Signal_Date")[
        [column for column in core_ranked.columns if column == "Signal_Date"
         or column.startswith(("CSI500_", "ChiNext_", "STAR50_", "Index_", "Tech_", "Market_"))]
    ]
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "信号开始": signal_start, "信号截止": signal_end,
        "观察截止": market_end, "全部周线SKDJ金叉": len(event_frame), "核心事件": len(core_ranked),
        "核心40日成熟事件": len(mature_core), "核心不同股票": core_ranked["ts_code"].nunique(),
        "自然周": len(calendar), "核心有信号周": int(calendar["Original_Core_Count"].gt(0).sum()),
        "核心平均每自然周": len(core_ranked) / len(calendar) if len(calendar) else np.nan,
        "核心平均每信号周": len(core_ranked) / max(core_ranked["Signal_Date"].nunique(), 1),
        "核心空窗周": int(calendar["Original_Core_Empty"].sum()),
        "冻结强势门槛事件": int(core_ranked["Market_Gate_Strong"].map(to_bool).sum()),
        "指数数据齐全周": int(weekly_market.get("Index_Available_Count", pd.Series(dtype=float)).eq(3).sum()),
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    metadata = pd.DataFrame([
        ("核心候选", "完整周线SKDJ金叉，近3个完整周K或D最低值≤25，金叉当周(K+D)/2>20且≤35"),
        ("SKDJ参数", f"N={SKDJ_N},M={SKDJ_M}，冻结不寻优"),
        ("股票池", "申万2021历史科技池；主板/创业板/科创板；排除北交所"),
        ("价格市值", "信号日原始收盘价≥10元；历史流通市值≥100亿元"),
        ("买入", "完整周线收盘确认，下一市场交易日开盘；主板一字板不买"),
        ("成本", "买卖滑点、双边佣金与过户费、卖出印花税均计入固定终点收益"),
        ("市场指数", "中证500=000905.SH；创业板指=399006.SZ；科创50=000688.SH"),
        ("科技宽度", "逐股票、逐历史周末按历史行业成员身份及同一价格市值门槛统计，不使用今日成分回填"),
        ("强势门槛", "三指数齐全；至少2个站上MA20且MA20较10日前上升；科技池站上MA20比例≥50%"),
        ("弱势定义", "至多1个指数站上MA20且MA20上升；科技池站上MA20比例<40%"),
        ("排序", "只在同周核心池比较预先冻结的单因子Top1/Top2/Top3，不拼接线性总分"),
        ("随机基准", f"每个因子、每个TopK、每个年度均在相同可用候选中做{RANDOM_RUNS}次同周随机抽样"),
        ("统计口径", "同时报告事件加权和周等权；排序报告跨年稳定及随机百分位"),
        ("独立性限制", "V3.5规则由V3.4同一历史段提出；本次是复核，不是新的独立样本外证明"),
        ("资金约束备忘", f"后续组合仍为{INITIAL_CAPITAL:.0f}元、最多{MAX_POSITIONS}仓、每仓约{POSITION_BUDGET:.0f}元"),
    ], columns=["项目", "值"])
    files = {
        "01_run_summary_v3_5.csv": run_summary,
        "02_market_gate_comparison_v3_5.csv": market_summary,
        "03_market_gate_yearly_stability_v3_5.csv": market_yearly,
        "04_market_feature_week_quintiles_v3_5.csv": market_bins,
        "05_single_factor_topk_vs_random_v3_5.csv": factor_summary,
        "06_single_factor_topk_yearly_v3_5.csv": factor_yearly,
        "07_random_topk_benchmark_v3_5.csv": random_summary,
        "08_single_factor_by_market_state_v3_5.csv": factor_by_market,
        "09_selected_event_detail_v3_5.csv": selection_detail,
        "10_core_event_rank_detail_v3_5.csv": core_ranked,
        "11_weekly_market_environment_v3_5.csv": weekly_market,
        "12_all_weekly_skdj_events_v3_5.csv": event_frame,
        "13_weekly_pool_calendar_v3_5.csv": calendar,
        "14_full_tech_universe_v3_5.csv": stocks,
        "15_board_population_v3_5.csv": population,
        "16_rejection_audit_v3_5.csv": pd.DataFrame(
            [{"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]),
        "17_api_errors_v3_5.csv": pd.DataFrame({"错误": API_ERRORS}),
        "18_metadata_v3_5.csv": metadata,
    }
    result_zip = make_zip(files)
    st.session_state[session_key] = result_zip
    st.success(
        f"完成：全部金叉{len(event_frame)}个；核心{len(core_ranked)}个；"
        f"核心成熟{len(mature_core)}个；强势门槛{core_ranked['Market_Gate_Strong'].map(to_bool).sum()}个。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("核心事件", len(core_ranked))
    c2.metric("核心有信号周", core_ranked["Signal_Date"].nunique())
    c3.metric("核心空窗周", int(calendar["Original_Core_Empty"].sum()))
    c4.metric("强势门槛事件", int(core_ranked["Market_Gate_Strong"].map(to_bool).sum()))
    st.subheader("第一层：市场开仓过滤")
    st.dataframe(market_summary, use_container_width=True, hide_index=True)
    st.subheader("第二层：核心池单因子TopK与随机基准")
    st.dataframe(factor_summary, use_container_width=True, hide_index=True)
    st.subheader("跨年稳定性")
    st.dataframe(factor_yearly, use_container_width=True, hide_index=True)
    st.download_button(
        "下载V3.5全部结果ZIP", result_zip,
        file_name="weekly_skdj_market_gate_rank_v3_5_all_results.zip",
        mime="application/zip", type="primary", key="v35_download", on_click="ignore",
    )
    st.info("先看02、03判断市场门槛能否跨年减少回撤；再看05、06中Top3是否连续多年高于随机基准。任何只靠平均收益、却没有改善中位数和MAE的因子都不进入实盘。")


if __name__ == "__main__":
    main()
