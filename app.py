# -*- coding: utf-8 -*-
"""科技股周线SKDJ低位金叉后上穿25确认审计 V4.1。

验证假设：周线K、D均在25以下发生金叉后，不立即买；等待1~6个完整周，
当K第一次由25下方上穿25、K>D且K继续上升时，下一市场交易日开盘买入。

本版输出逐周W1~W5最高收益、期末收益、最大回撤、10/20/30%命中率，
并与同一低位金叉的“立即买入”进行配对比较。固定20/40交易日仍保留为对照。
周线只使用完整周；信号、价格、市值及趋势特征都只使用当时已知数据。

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
TITLE = "科技股周线SKDJ低位金叉后上穿25确认审计 V4.1"
VERSION = "V4.1-WEEKLY-SKDJ-LOW-CROSS-DELAYED-25-CONFIRMATION-AUDIT"
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
    work = daily.copy().sort_values("trade_date").reset_index(drop=True)
    work["D_MA60_Bias_pct"] = (work["close"] / work["close"].rolling(60).mean() - 1.0) * 100.0
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
                  market_weeks: list[tuple[pd.Period, str]],
                  config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weekly = build_complete_weekly(daily_raw, week_last_map)
    if len(weekly) < INDICATOR_WARMUP_WEEKS:
        config["rejects"]["周线不足"] = config["rejects"].get("周线不足", 0) + 1
        return [], []
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

    anchors: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    code, board = str(stock["ts_code"]), sample_board(stock)
    for anchor_position, trigger_position in anchors_all:
        anchor = weekly.iloc[anchor_position]
        anchor_date = str(anchor["trade_date"])
        trigger_date = str(weekly.iloc[trigger_position]["trade_date"]) if trigger_position is not None else ""
        lag = float(trigger_position - anchor_position) if trigger_position is not None else np.nan
        selected_anchor = trigger_position is None or latest_anchor_by_trigger.get(trigger_position) == anchor_position
        full_6w = anchor_position + WAIT_MAX_WEEKS < len(weekly)

        if config["signal_start"] <= anchor_date <= config["signal_end"]:
            membership = membership_on_date(periods, anchor_date)
            snapshot = market_snapshot(daily_basic, anchor_date)
            passed, reason = signal_filter(snapshot, config["min_price"], config["min_mv"])
            listed = str(stock["list_date"]) <= anchor_date < str(stock["delist_date"])
            eligible = bool(membership is not None and listed and passed)
            if not eligible:
                reject_reason = reason or ("当时不在历史科技池" if membership is None else "当时未上市或已退市")
                config["rejects"][f"低位金叉:{reject_reason}"] = config["rejects"].get(f"低位金叉:{reject_reason}", 0) + 1
            anchors.append({
                "ts_code": code, "name": str(stock["name"]), "Sample_Board": board,
                "Low_Cross_Date": anchor_date, "Low_Cross_K": float(anchor["SKDJ_K"]),
                "Low_Cross_D": float(anchor["SKDJ_D"]), "Low_Cross_Level": (float(anchor["SKDJ_K"]) + float(anchor["SKDJ_D"])) / 2.0,
                "Has_Full_6W_Observation": full_6w, "Confirmed_Within_6W": trigger_position is not None,
                "Selected_As_Latest_Anchor": selected_anchor, "Wait_Weeks": lag,
                "Confirm_Date": trigger_date, "Eligible_Low_Cross_Pool": eligible,
                "Low_Cross_Filter_Reason": "" if eligible else reject_reason, **snapshot,
            })

        if trigger_position is None or not selected_anchor:
            continue
        if not (config["signal_start"] <= trigger_date <= config["signal_end"]):
            continue
        if not (str(stock["list_date"]) <= trigger_date < str(stock["delist_date"])):
            config["rejects"]["确认日未上市或已退市"] = config["rejects"].get("确认日未上市或已退市", 0) + 1
            continue
        membership = membership_on_date(periods, trigger_date)
        if membership is None:
            config["rejects"]["确认日不在历史科技池"] = config["rejects"].get("确认日不在历史科技池", 0) + 1
            continue
        anchor_snapshot = market_snapshot(daily_basic, anchor_date)
        trigger_snapshot = market_snapshot(daily_basic, trigger_date)
        anchor_passed, anchor_reason = signal_filter(anchor_snapshot, config["min_price"], config["min_mv"])
        trigger_passed, trigger_reason = signal_filter(trigger_snapshot, config["min_price"], config["min_mv"])
        if not anchor_passed or not trigger_passed:
            reason = f"低位金叉:{anchor_reason}" if not anchor_passed else f"确认日:{trigger_reason}"
            config["rejects"][reason] = config["rejects"].get(reason, 0) + 1
            continue
        trigger = weekly.iloc[trigger_position]
        event = {
            "ts_code": code, "name": str(stock["name"]), "Sample_Board": board,
            "SW_L1": membership["l1"], "SW_L2": membership["l2"], "SW_L3": membership["l3"],
            "Low_Cross_Date": anchor_date, "Confirm_Date": trigger_date, "Wait_Weeks": int(lag),
            "Low_Cross_K": float(anchor["SKDJ_K"]), "Low_Cross_D": float(anchor["SKDJ_D"]),
            "Confirm_K": float(trigger["SKDJ_K"]), "Confirm_D": float(trigger["SKDJ_D"]),
            "Confirm_K_Change_1W": float(trigger["SKDJ_K"] - weekly.iloc[trigger_position - 1]["SKDJ_K"]),
            "Low_Cross_Raw_Close": anchor_snapshot["Raw_Close"],
            "Confirm_Raw_Close": trigger_snapshot["Raw_Close"],
            "Confirm_Circ_MV_Billion": trigger_snapshot["Circ_MV_Billion"],
            "Confirm_Turnover_Rate": trigger_snapshot["Turnover_Rate"],
            "Price_Change_Cross_to_Confirm_pct": (
                (trigger_snapshot["Raw_Close"] / anchor_snapshot["Raw_Close"] - 1.0) * 100.0
                if anchor_snapshot["Raw_Close"] > 0 else np.nan),
            "Period_Group": "2025-06以后" if trigger_date >= config["split_date"] else "2025-06以前",
            **stock_trend(weekly, trigger_position, daily, trigger_date),
        }
        event.update(prefix_keys(entry_outcomes(
            daily, anchor_date, code, open_dates, open_pos, market_weeks, config), "Immediate"))
        event.update(prefix_keys(entry_outcomes(
            daily, trigger_date, code, open_dates, open_pos, market_weeks, config), "Confirmed"))
        events.append(event)
    return anchors, events


def max_empty_run(counts: pd.Series) -> int:
    longest = current = 0
    for value in counts.tolist():
        current = current + 1 if int(value) == 0 else 0
        longest = max(longest, current)
    return longest


def signal_week_calendar(open_dates: list[str], start: str, end: str, events: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"trade_date": [day for day in open_dates if start <= day <= end]})
    frame["period"] = pd.to_datetime(frame["trade_date"]).dt.to_period("W-FRI")
    calendar = frame.groupby("period", as_index=False)["trade_date"].max().rename(columns={"trade_date": "Week_Last_Trade_Date"})
    counts = events.groupby("Confirm_Date").size() if not events.empty else pd.Series(dtype=int)
    mature = events[events["Confirmed_Tradable"].map(to_bool) & events[f"Confirmed_Has_W{AUDIT_WEEKS}"].map(to_bool)] if not events.empty else events
    mature_counts = mature.groupby("Confirm_Date").size() if not mature.empty else pd.Series(dtype=int)
    calendar["Confirmed_Signals"] = calendar["Week_Last_Trade_Date"].map(counts).fillna(0).astype(int)
    calendar["Mature_Tradable_Signals"] = calendar["Week_Last_Trade_Date"].map(mature_counts).fillna(0).astype(int)
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


def grouped_outcome_audit(events: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in events.groupby(column, dropna=False, sort=True):
        rows.append(outcome_summary(group, "Confirmed", str(value)))
    return pd.DataFrame(rows)


def weekly_path_audit(events: pd.DataFrame) -> pd.DataFrame:
    mature = events[events["Confirmed_Tradable"].map(to_bool)].copy()
    rows = []
    for week in range(1, AUDIT_WEEKS + 1):
        sample = mature[mature[f"Confirmed_Has_W{week}"].map(to_bool)]
        mfe = pd.to_numeric(sample[f"Confirmed_W{week}_Cum_Max_High_Raw_pct"], errors="coerce")
        close_ret = pd.to_numeric(sample[f"Confirmed_W{week}_Close_Return_Net_pct"], errors="coerce")
        mae = pd.to_numeric(sample[f"Confirmed_W{week}_Cum_MAE_Raw_pct"], errors="coerce")
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
    st.set_page_config(page_title="周线SKDJ上穿25确认审计 V4.1", layout="wide")
    st.title(TITLE)
    st.caption("验证：周线低位金叉后等待1～6周，首次上穿25再买，是否比低位金叉立即买更可靠。")
    with st.expander("V4.1验证规则", expanded=True):
        st.markdown(f"""
- **低位结构**：完整周线K上穿D，而且当周K、D均不高于25；参数固定 `N={SKDJ_N}, M={SKDJ_M}`。
- **确认信号**：低位金叉之后第1～6个完整周中，K第一次从25下方上穿25，同时K>D且K继续上升。
- **去重**：同一只股票多个低位金叉指向同一个确认周时，只保留距离确认周最近的一次，不重复买入。
- **股票池**：申万历史科技行业；主板、创业板、科创板；低位金叉日和确认日均要求原始股价≥10元、流通市值≥100亿元。
- **买入**：周线确认结束后的下一市场交易日开盘；主板一字板不买。
- **核心判卷**：买入后W1～W5累计最高价收益、期末净收益、最大回撤，以及达到10%/20%/30%的比例。
- **严格对照**：同一批最终确认事件，同时计算低位金叉后立即买入的结果；固定20/40交易日结果继续保留。
- **执行口径**：最高价收益同时输出不计成本的原始价差和计入成本/滑点的潜在收益；期末收益计入完整成本。
- **本版不做**：不优化止盈止损、不评分、不使用月线；先确认买点是否真实存在。
""")
    with st.sidebar:
        st.header("运行参数")
        signal_start_date = st.date_input("确认信号开始", date(2023, 6, 5), key="v41_start")
        signal_end_date = st.date_input("确认信号截止", date(2026, 6, 5), key="v41_end")
        market_end_date = st.date_input("行情观察截止", date.today(), key="v41_market_end")
        split_date_value = st.date_input("近期行情分界", date(2025, 6, 1), key="v41_split")
        pause = st.number_input("接口间隔(秒)", 0.0, 2.0, 0.12, 0.02, key="v41_pause")
        use_cache = st.checkbox("复用逐股票缓存", True, key="v41_cache")
        st.divider()
        commission_pct = st.number_input("佣金率(%)", 0.0, 0.20, 0.025, 0.005, format="%.3f", key="v41_commission")
        stamp_duty_pct = st.number_input("卖出印花税率(%)", 0.0, 0.20, 0.05, 0.01, format="%.3f", key="v41_stamp")
        transfer_fee_pct = st.number_input("过户费率(%)", 0.0, 0.05, 0.001, 0.001, format="%.3f", key="v41_transfer")
        if st.button("清除本程序行情缓存", key="v41_clear"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("缓存已清除")

    token = st.text_input("Tushare Token", type="password", key="v41_token")
    session_key = "weekly_skdj_delayed_25_confirmation_v41_zip"
    if not token:
        st.info("请输入Tushare Token；日期范围一致时可直接复用V3.4～V4.0逐股票缓存。")
        return
    if not st.button("开始V4.1两阶段买点验证", type="primary", key="v41_run"):
        if session_key in st.session_state:
            st.download_button(
                "下载上一次结果ZIP", st.session_state[session_key],
                file_name="weekly_skdj_delayed_25_confirmation_audit_v4_1_all_results.zip",
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
    event_rows: list[dict[str, Any]] = []
    cache_hits = data_failures = 0
    progress, status = st.progress(0.0), st.empty()
    for number, stock in stocks.iterrows():
        code = str(stock["ts_code"])
        progress.progress((number + 1) / max(len(stocks), 1), text=f"{number + 1}/{len(stocks)} {code}")
        status.caption(f"低位金叉 {len(anchor_rows)}；上穿25确认 {len(event_rows)}；缓存 {cache_hits}；失败 {data_failures}")
        daily, daily_basic, cache_hit = fetch_stock_history(
            code, preload, market_end, bool(use_cache), float(pause))
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        anchors, events = analyze_stock(
            stock, period_index.get(code, []), daily, daily_basic, week_last_map,
            open_dates, open_pos, market_weeks, config)
        anchor_rows.extend(anchors)
        event_rows.extend(events)
    progress.empty()
    status.empty()
    if not anchor_rows:
        st.error("研究区间没有找到符合基础股票池的周线SKDJ低位金叉。")
        return

    anchors = pd.DataFrame(anchor_rows)
    events = pd.DataFrame(event_rows)
    if events.empty:
        st.error("研究区间存在低位金叉，但没有形成符合条件的1～6周上穿25确认事件。")
        return
    confirmation_summary, confirmation_lag = confirmation_audit(anchors)
    paired_events = events[
        events["Immediate_Tradable"].map(to_bool)
        & events["Confirmed_Tradable"].map(to_bool)
        & events[f"Immediate_Has_W{AUDIT_WEEKS}"].map(to_bool)
        & events[f"Confirmed_Has_W{AUDIT_WEEKS}"].map(to_bool)
    ].copy()
    paired = pd.DataFrame([
        outcome_summary(paired_events, "Immediate", "严格配对：低位金叉后立即买"),
        outcome_summary(paired_events, "Confirmed", "严格配对：1～6周上穿25后买"),
    ])
    path_audit = weekly_path_audit(events)
    lag_quality = grouped_outcome_audit(events, "Wait_Weeks")
    period_quality = grouped_outcome_audit(events, "Period_Group")
    trend_quality = grouped_outcome_audit(events, "Individual_Trend")
    yearly_events = events.copy()
    yearly_events["Confirm_Year"] = yearly_events["Confirm_Date"].astype(str).str[:4]
    yearly_quality = grouped_outcome_audit(yearly_events, "Confirm_Year")
    calendar = signal_week_calendar(open_dates, signal_start, signal_end, events)
    signal_counts = calendar["Confirmed_Signals"]
    mature = events[events["Confirmed_Tradable"].map(to_bool) & events[f"Confirmed_Has_W{AUDIT_WEEKS}"].map(to_bool)]
    anchor_eligible = anchors[anchors["Eligible_Low_Cross_Pool"].map(to_bool)]

    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "确认开始": signal_start, "确认截止": signal_end,
        "观察截止": market_end, "符合池的低位金叉": len(anchor_eligible),
        "去重后上穿25确认事件": len(events), "确认事件不同股票": events["ts_code"].nunique(),
        "W5成熟可买事件": len(mature), "自然周": len(calendar),
        "有确认信号周": int(signal_counts.gt(0).sum()), "空窗周": int(signal_counts.eq(0).sum()),
        "最长连续空窗周": max_empty_run(signal_counts), "平均每周确认信号": signal_counts.mean(),
        "每周确认信号中位数": signal_counts.median(), "单周最多确认信号": signal_counts.max(),
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    unconfirmed = anchors[
        anchors["Eligible_Low_Cross_Pool"].map(to_bool)
        & anchors["Has_Full_6W_Observation"].map(to_bool)
        & ~anchors["Confirmed_Within_6W"].map(to_bool)
    ].copy()
    metadata = pd.DataFrame([
        ("低位金叉", f"周线SKDJ(N={SKDJ_N},M={SKDJ_M}) K上穿D，且当周K、D均≤25"),
        ("确认", "低位金叉后第1至6周，K首次从25下方上穿25，同时K>D且K继续上升"),
        ("去重", "多个低位金叉若指向同一只股票同一个确认周，只保留最近的低位金叉"),
        ("股票池", "申万历史科技行业；主板/创业板/科创板；低位金叉日和确认日均为股价≥10元、流通市值≥100亿元"),
        ("买入", "确认周结束后的下一市场交易日开盘；主板一字板不买"),
        ("W1至W5", "W1是买入所在市场周，W2至W5依次为后续完整市场周；最高价为截至该周的累计最高价"),
        ("原始最高收益", "未来累计最高复权价/下一市场交易日复权开盘价-1，不扣成本，直接对应人工看图口径"),
        ("净收益", "买卖均计0.20%滑点、佣金和过户费，卖出另计印花税"),
        ("同日双触发", "同一日最高价达到+10%且最低价达到-10%时无法判断先后，保守按止损先"),
        ("趋势", "确认周：周线MA20位置、MA20四周斜率、12周收益和日线MA60位置全正为上涨，全负为下跌，其余为震荡/过渡"),
        ("配对比较", "只保留立即买与确认买都可交易且都有完整W5的同一批事件，避免样本数量不同造成假改善"),
        ("日期边界", "确认率以区间内低位金叉为锚点并可观察其后六周；收益与覆盖率只纳入确认日本身位于信号区间的事件"),
        ("固定对照", "继续输出20/40个市场交易日净收益与原始MFE/MAE，但不作为本版唯一结论"),
        ("未使用", "月线、分钟线、评分模型、止盈止损参数优化、未来信息"),
    ], columns=["项目", "说明"])

    files = {
        "01_run_summary_v4_1.csv": run_summary,
        "02_low_cross_confirmation_rate_v4_1.csv": confirmation_summary,
        "03_confirmation_count_by_wait_week_v4_1.csv": confirmation_lag,
        "04_immediate_vs_confirmed_paired_v4_1.csv": paired,
        "05_confirmed_w1_w5_path_v4_1.csv": path_audit,
        "06_wait_week_quality_v4_1.csv": lag_quality,
        "07_recent_period_quality_v4_1.csv": period_quality,
        "08_individual_trend_quality_v4_1.csv": trend_quality,
        "09_yearly_quality_v4_1.csv": yearly_quality,
        "10_weekly_signal_calendar_v4_1.csv": calendar,
        "11_confirmed_event_detail_v4_1.csv": events,
        "12_all_low_cross_anchor_detail_v4_1.csv": anchors,
        "13_unconfirmed_after_6w_v4_1.csv": unconfirmed,
        "14_full_tech_universe_v4_1.csv": stocks,
        "15_board_population_v4_1.csv": population,
        "16_rejection_audit_v4_1.csv": pd.DataFrame(
            [{"剔除原因": key, "次数": value} for key, value in sorted(rejects.items())]),
        "17_api_errors_v4_1.csv": pd.DataFrame({"错误": API_ERRORS}),
        "18_metadata_v4_1.csv": metadata,
    }
    result_zip = make_zip(files)
    st.session_state[session_key] = result_zip
    st.success(
        f"完成：符合池低位金叉{len(anchor_eligible)}个；去重后上穿25确认{len(events)}个；"
        f"W5成熟可买{len(mature)}个；覆盖{int(signal_counts.gt(0).sum())}/{len(calendar)}周。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("确认事件", len(events))
    c2.metric("W5成熟可买", len(mature))
    c3.metric("有信号周", int(signal_counts.gt(0).sum()))
    c4.metric("空窗周", int(signal_counts.eq(0).sum()))
    st.subheader("低位金叉六周内确认率")
    st.dataframe(confirmation_summary, use_container_width=True, hide_index=True)
    st.subheader("同一事件：立即买与上穿25后买")
    st.dataframe(paired, use_container_width=True, hide_index=True)
    st.subheader("确认买入后的W1～W5路径")
    st.dataframe(path_audit, use_container_width=True, hide_index=True)
    st.subheader("等待第1～6周分别表现")
    st.dataframe(lag_quality, use_container_width=True, hide_index=True)
    st.subheader("2025年6月前后表现")
    st.dataframe(period_quality, use_container_width=True, hide_index=True)
    st.download_button(
        "下载V4.1全部结果ZIP", result_zip,
        file_name="weekly_skdj_delayed_25_confirmation_audit_v4_1_all_results.zip",
        mime="application/zip", type="primary", key="v41_download", on_click="ignore")
    st.info("先看02确认六周内上穿25是否常见；再看04判断延迟确认是否优于立即买；05验证10%～30%最高涨幅；06与07检查结果是否只集中在某个等待周或2025年6月以后。")


if __name__ == "__main__":
    main()
