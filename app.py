# -*- coding: utf-8 -*-
"""
科技股周线MACD第二周持有退出验证器 V1.0
===================================

本程序固定在第一根真正周线红柱确认后的下一交易日开盘买入。第二根完整
周线红柱若严格长于第一根则继续持有；若缩短、持平或翻绿，则在第二周
确认后的下一可交易日开盘退出。它验证提前退出减少了多少弱周期损失、
误杀了多少长周期，以及退出后还剩多少上涨空间。继续持有组仍只统计至
第一根完整绿柱前的机会空间，不把最高价伪装成可实现卖价。
以下底层统计仍保留在程序中，用于生成事件和未来路径：
1. 上升/下降趋势中，第一根周线红柱后，下一周继续红柱的概率。
2. 两类趋势中，未来八周触及 +10%/+20%/+30% 的概率。
3. 第一根红柱买入 vs 同一红柱周期第一次红柱缩短买入。
4. 第一根红柱后一周立即翻绿的比例。
5. DIF、DEA 位于零轴上方/下方的差异。
6. 第一根红柱前回调深度对未来八周表现的影响。
7. 不随机抽样，研究期历史科技股票池中符合价格、市值条件的事件全部纳入。
8. 对基准组、上升趋势组、回调<30%组及红柱缩短组做稳健性比较。
9. 模拟 +10%/+20%/+30% 止盈与统一止损后的实际退出收益。
10. 将第一根红柱后的完整周期事后划分为 A/B/C1/C2，验证哪类利润最大。
11. 只用当时已知数据记录第2—5周状态，检验能否提前识别弱反弹。
12. 信号截止日与行情截止日分离：股票池和信号严格停在前者，后者仅用于观察未来结果。

严格口径：
- 周线信号只使用已经结束的完整周，绝不使用周一至周四的临时周K。
- 信号周最后一个交易日为 D0，下一市场交易日开盘为买入价。
- 未来八周按 40 个市场交易日计算，而不是按个股实际成交天数计算。
- 主板 D1 一字板视为无法成交；双创板沿用既有口径，不做该项剔除。
- 同一天同时触及止损和目标价时，日线无法判断先后，屏障统计按止损先到处理，
  避免乐观偏差；单纯“八周内曾达到目标”的统计仍如实记录。
- “红柱缩短”默认只记录同一轮红柱中的第一次缩短，避免重复样本。

运行：
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import shutil
import time
import zipfile
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts


VERSION = "V1.0-WEEK2-HOLD-EXIT"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1_1")
OUTPUT_DIR = os.path.join(APP_DIR, "weekly_macd_validation_outputs")

HOLD_WEEKS = 8
HOLD_TRADING_DAYS = HOLD_WEEKS * 5
MACD_WARMUP_WEEKS = 80

CORE_TECH_L1 = {"电子", "计算机", "通信", "国防军工"}
EXTENDED_TECH_L1 = {
    "机械设备", "电力设备", "医药生物", "汽车",
    "基础化工", "有色金属", "建筑材料",
}
TECH_INDUSTRY_KEYWORDS = {
    "半导体", "电子元件", "元件", "光学光电子", "消费电子", "电子化学品",
    "计算机设备", "软件开发", "IT服务", "通信设备", "军工电子", "航空装备",
    "航天装备", "自动化设备", "机器人", "激光设备", "工控设备", "仪器仪表",
    "电池", "光伏设备", "风电设备", "电网设备", "电机", "医疗器械",
    "生物制品", "汽车电子", "金属新材料", "非金属材料", "膜材料", "碳纤维",
}

SAMPLE_BOARDS = ("主板", "创业板", "科创板")
DEFAULT_SAMPLE_PER_BOARD = 0
DEFAULT_SAMPLE_SEED = 20260806
DEFAULT_LONG_CYCLE_MIN_WEEKS = 9
DEFAULT_MATERIAL_HIST_CHANGE_PCT = 10.0
DEFAULT_SHORT_STRENGTH_RATIO = 0.50
CHECKPOINT_WEEKS = (2, 3, 4, 5)

TITLE = "科技股周线MACD第二周持有退出验证器 V1.0"
INITIAL_CAPITAL = 300_000.0
MAX_POSITIONS = 3
POSITION_BUDGET = 100_000.0
LOT_SIZE = 100
BOARD_INDEX = {"主板": "000905.SH", "创业板": "399006.SZ", "科创板": "000688.SH"}
# 只有信号真实性和次日可执行性可以一票否决。V40.6形态全部改为评分项。
V50_EXECUTION_GATES = [
    "True_First_Red_Audit", "One_Word_Pass", "Gap_Pass",
]

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


def validate_research_dates(
    signal_start_date: date,
    signal_end_date: date,
    market_end_date: date,
) -> str:
    """返回日期配置错误；空字符串表示配置有效。"""
    if signal_start_date >= signal_end_date:
        return "信号开始日期必须早于信号截止日期"
    if market_end_date <= signal_end_date:
        return "行情截止日期必须晚于信号截止日期"
    return ""


def record_error(message: str) -> None:
    if len(API_ERRORS) < 200:
        API_ERRORS.append(message)


def safe_get(func_name: str, retries: int = 3, required: bool = False, **kwargs) -> pd.DataFrame:
    global pro
    if pro is None:
        if required:
            raise RuntimeError("Tushare 尚未初始化")
        return pd.DataFrame()
    try:
        func = getattr(pro, func_name)
    except AttributeError as exc:
        if required:
            raise RuntimeError(f"当前 Tushare SDK 不支持接口 {func_name}") from exc
        record_error(f"缺少接口 {func_name}")
        return pd.DataFrame()

    last_error = None
    for attempt in range(retries):
        try:
            result = func(**kwargs)
            if result is None:
                return pd.DataFrame()
            return result
        except Exception as exc:  # 网络与频率错误统一重试
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    message = f"{func_name} 失败: {last_error}"
    record_error(message)
    if required:
        raise RuntimeError(message)
    return pd.DataFrame()


def atomic_pickle(payload: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = f"{path}.tmp"
    with open(temp, "wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def atomic_csv(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = f"{path}.tmp"
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def cache_key(*parts: Any) -> str:
    raw = "|".join(str(x) for x in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def pct_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean() * 100.0) if len(values) else np.nan


def numeric_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def numeric_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


# -----------------------------------------------------------------------------
# 股票池：与既有项目一致的历史申万科技池
# -----------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600)
def load_stock_basic() -> pd.DataFrame:
    frames = []
    fields = "ts_code,symbol,name,market,exchange,list_status,list_date,delist_date"
    for status in ["L", "P", "D"]:
        frame = safe_get("stock_basic", list_status=status, fields=fields)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("stock_basic 加载失败")
    result = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
    result = result[
        result["market"].isin(["主板", "创业板", "科创板"])
        & result["exchange"].ne("BSE")
        & ~result["ts_code"].astype(str).str.endswith(".BJ", na=False)
        & ~result["name"].astype(str).str.contains("ST|退", na=False)
    ].copy()
    result["list_date"] = result["list_date"].apply(lambda x: normalize_date(x, "19000101"))
    result["delist_date"] = result["delist_date"].apply(lambda x: normalize_date(x, "99991231"))
    return result


def industry_row_is_tech(row: pd.Series) -> bool:
    l1 = str(row.get("l1_name", ""))
    l2 = str(row.get("l2_name", ""))
    l3 = str(row.get("l3_name", ""))
    if l1 in CORE_TECH_L1:
        return True
    if l1 not in EXTENDED_TECH_L1:
        return False
    return any(keyword in f"{l2}|{l3}" for keyword in TECH_INDUSTRY_KEYWORDS)


@st.cache_data(ttl=7 * 24 * 3600)
def load_sw_tech_memberships(api_pause: float) -> pd.DataFrame:
    l1 = safe_get("index_classify", required=True, level="L1", src="SW2021")
    if l1.empty:
        raise RuntimeError("申万 2021 行业目录为空")
    targets = l1[l1["industry_name"].isin(CORE_TECH_L1 | EXTENDED_TECH_L1)]
    if targets.empty:
        raise RuntimeError("未找到目标申万行业")

    frames = []
    target_rows = targets[["index_code", "industry_name"]].to_dict("records")
    progress = st.progress(0.0, text="正在构建申万科技历史成分池...")
    total = max(len(target_rows) * 2, 1)
    count = 0
    for item in target_rows:
        for flag in ["Y", "N"]:
            frame = safe_get("index_member_all", l1_code=item["index_code"], is_new=flag)
            count += 1
            progress.progress(count / total, text=f"行业池：{item['industry_name']} {flag}")
            if not frame.empty:
                if "ts_code" not in frame.columns and "con_code" in frame.columns:
                    frame = frame.rename(columns={"con_code": "ts_code"})
                frames.append(frame)
            time.sleep(api_pause)
    progress.empty()
    if not frames:
        raise RuntimeError("index_member_all 未返回数据，请确认积分权限和 SDK 版本")

    result = pd.concat(frames, ignore_index=True)
    for column in ["ts_code", "l1_name", "l2_name", "l3_name", "in_date", "out_date"]:
        if column not in result.columns:
            result[column] = ""
    result = result[result.apply(industry_row_is_tech, axis=1)].copy()
    result["in_date"] = result["in_date"].apply(lambda x: normalize_date(x, "19000101"))
    result["out_date"] = result["out_date"].apply(lambda x: normalize_date(x, "99991231"))
    result = result.drop_duplicates(
        ["ts_code", "l1_name", "l2_name", "l3_name", "in_date", "out_date"]
    )
    if result.empty:
        raise RuntimeError("科技关键词过滤后行业池为空")
    return result


def build_period_index(memberships: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in memberships.itertuples(index=False):
        result.setdefault(str(row.ts_code), []).append({
            "in_date": str(row.in_date),
            "out_date": str(row.out_date),
            "l1": str(row.l1_name),
            "l2": str(row.l2_name),
            "l3": str(row.l3_name),
        })
    return result


def membership_on_date(periods: list[dict[str, str]], trade_date: str):
    for period in periods:
        if period["in_date"] <= trade_date < period["out_date"]:
            return period
    return None


def sample_board(row: pd.Series) -> str:
    """统一板块名称；优先使用 Tushare market，代码前缀只作兜底。"""
    market = str(row.get("market", "")).strip()
    if market in SAMPLE_BOARDS:
        return market
    ts_code = str(row.get("ts_code", ""))
    if ts_code.startswith(("300", "301")):
        return "创业板"
    if ts_code.startswith(("688", "689")):
        return "科创板"
    return "主板"


def representative_membership(
    periods: list[dict[str, str]],
    reference_date: str,
) -> dict[str, str]:
    active = membership_on_date(periods, reference_date)
    if active is not None:
        return active
    eligible = [period for period in periods if period["in_date"] <= reference_date]
    if eligible:
        return max(eligible, key=lambda item: item["in_date"])
    return periods[-1] if periods else {"l1": "", "l2": "", "l3": ""}


def build_stratified_sample(
    stocks: pd.DataFrame,
    period_index: dict[str, list[dict[str, str]]],
    reference_date: str,
    per_board: int = DEFAULT_SAMPLE_PER_BOARD,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    主板、创业板、科创板分别固定随机抽样。

    不足 per_board 的板块取全部，不把空缺名额转给其他板块。每只样本的
    Sample_Weight=N_board/n_board，用于还原完整股票池的板块实际占比。
    """
    if stocks.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    if per_board < 0:
        raise ValueError("每板块抽样数不能为负数")

    universe = stocks.copy()
    universe["Sample_Board"] = universe.apply(sample_board, axis=1)
    universe = universe[universe["Sample_Board"].isin(SAMPLE_BOARDS)].copy()
    universe = universe.sort_values("ts_code").drop_duplicates("ts_code", keep="first")

    sampled_frames: list[pd.DataFrame] = []
    population_rows: list[dict[str, Any]] = []
    for board_no, board in enumerate(SAMPLE_BOARDS):
        group = universe[universe["Sample_Board"].eq(board)].copy()
        population_size = int(len(group))
        requested = population_size if per_board == 0 else int(per_board)
        sample_size = min(requested, population_size)
        if sample_size:
            # 不使用 Python hash，确保不同机器和进程得到同一名单。
            board_seed_raw = hashlib.sha256(f"{seed}|{board}|{board_no}".encode("utf-8")).hexdigest()
            board_seed = int(board_seed_raw[:8], 16)
            chosen = group.sample(n=sample_size, replace=False, random_state=board_seed).copy()
            chosen["Sample_Seed"] = int(seed)
            chosen["Board_Universe_Size"] = population_size
            chosen["Board_Sample_Size"] = sample_size
            chosen["Sample_Weight"] = population_size / sample_size
            sampled_frames.append(chosen)
        population_rows.append({
            "Sample_Board": board,
            "Board_Universe_Size": population_size,
            "Board_Sample_Size": sample_size,
            "Sampling_Fraction_pct": (
                sample_size / population_size * 100.0 if population_size else np.nan
            ),
            "Sample_Weight": population_size / sample_size if sample_size else np.nan,
        })

    if not sampled_frames:
        empty = pd.DataFrame()
        return empty, empty, pd.DataFrame(population_rows)

    sampled = pd.concat(sampled_frames, ignore_index=True)
    sampled["Sample_Order"] = np.arange(1, len(sampled) + 1)

    audit = sampled.copy()
    sw_rows = []
    for code in audit["ts_code"].astype(str):
        period = representative_membership(period_index.get(code, []), reference_date)
        sw_rows.append((period.get("l1", ""), period.get("l2", ""), period.get("l3", "")))
    audit[["Sample_SW_L1", "Sample_SW_L2", "Sample_SW_L3"]] = pd.DataFrame(
        sw_rows, index=audit.index
    )
    audit_columns = [
        "Sample_Order", "ts_code", "symbol", "name", "Sample_Board", "market",
        "exchange", "list_status", "list_date", "delist_date", "Sample_SW_L1",
        "Sample_SW_L2", "Sample_SW_L3", "Sample_Seed", "Board_Universe_Size",
        "Board_Sample_Size", "Sampling_Fraction_pct", "Sample_Weight",
    ]
    for column in audit_columns:
        if column not in audit.columns:
            audit[column] = ""
    audit = audit[audit_columns].sort_values("Sample_Order").reset_index(drop=True)
    return sampled.reset_index(drop=True), audit, pd.DataFrame(population_rows)


# -----------------------------------------------------------------------------
# 交易日与个股历史数据
# -----------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600)
def load_trade_calendar(start_date: str, end_date: str) -> list[str]:
    frame = safe_get(
        "trade_cal", required=True, exchange="SSE",
        start_date=start_date, end_date=end_date,
    )
    if frame.empty:
        raise RuntimeError("交易日历为空")
    return sorted(frame.loc[frame["is_open"] == 1, "cal_date"].astype(str).tolist())


def stock_cache_path(ts_code: str, start_date: str, end_date: str) -> str:
    safe_code = ts_code.replace(".", "_")
    return os.path.join(CACHE_DIR, f"{safe_code}_{start_date}_{end_date}.pkl")


def fetch_pro_bar(ts_code: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    last_error = None
    for attempt in range(retries):
        try:
            frame = ts.pro_bar(
                api=pro, ts_code=ts_code, start_date=start_date, end_date=end_date,
                adj="qfq", freq="D", factors=["tor"],
            )
            return pd.DataFrame() if frame is None else frame
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    record_error(f"pro_bar {ts_code} 失败: {last_error}")
    return pd.DataFrame()


def fetch_stock_history(
    ts_code: str,
    start_date: str,
    end_date: str,
    use_cache: bool,
    api_pause: float,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    path = stock_cache_path(ts_code, start_date, end_date)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as file:
                payload = pickle.load(file)
            return payload.get("daily", pd.DataFrame()), payload.get("basic", pd.DataFrame()), True
        except Exception as exc:
            record_error(f"缓存损坏 {ts_code}: {exc}")

    daily = fetch_pro_bar(ts_code, start_date, end_date)
    time.sleep(api_pause)
    basic = safe_get(
        "daily_basic", ts_code=ts_code, start_date=start_date, end_date=end_date,
        fields="ts_code,trade_date,close,circ_mv,turnover_rate",
    )
    time.sleep(api_pause)

    if not daily.empty:
        for column in ["open", "high", "low", "close", "vol"]:
            if column in daily.columns:
                daily[column] = pd.to_numeric(daily[column], errors="coerce")
        daily["trade_date"] = daily["trade_date"].astype(str)
        daily = daily.dropna(subset=["trade_date", "open", "high", "low", "close"])
        daily = daily.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    if not basic.empty:
        basic["trade_date"] = basic["trade_date"].astype(str)
        for column in ["close", "circ_mv", "turnover_rate"]:
            if column in basic.columns:
                basic[column] = pd.to_numeric(basic[column], errors="coerce")
        basic = basic.drop_duplicates("trade_date", keep="last").sort_values("trade_date")

    if use_cache and not daily.empty:
        atomic_pickle({"daily": daily, "basic": basic}, path)
    return daily, basic, False


def complete_week_last_dates(open_dates: list[str]) -> dict[pd.Timestamp, str]:
    frame = pd.DataFrame({"trade_date": open_dates})
    frame["dt"] = pd.to_datetime(frame["trade_date"])
    frame["week_label"] = frame["dt"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    return frame.groupby("week_label")["trade_date"].max().to_dict()


def build_weekly(daily: pd.DataFrame, week_last_map: dict[pd.Timestamp, str]) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    work["dt"] = pd.to_datetime(work["trade_date"])
    weekly = work.set_index("dt").resample("W-FRI").agg({
        "trade_date": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vol": "sum",
    }).dropna(subset=["close"]).reset_index().rename(columns={"dt": "week_label"})
    weekly["calendar_week_last"] = weekly["week_label"].map(week_last_map)
    weekly = weekly[
        weekly["calendar_week_last"].notna()
        & weekly["trade_date"].astype(str).eq(weekly["calendar_week_last"].astype(str))
    ].copy()
    if weekly.empty:
        return weekly

    weekly["ema12"] = weekly["close"].ewm(span=12, adjust=False).mean()
    weekly["ema26"] = weekly["close"].ewm(span=26, adjust=False).mean()
    weekly["dif"] = weekly["ema12"] - weekly["ema26"]
    weekly["dea"] = weekly["dif"].ewm(span=9, adjust=False).mean()
    weekly["hist"] = (weekly["dif"] - weekly["dea"]) * 2.0
    weekly["ma20"] = weekly["close"].rolling(20).mean()
    weekly["ma40"] = weekly["close"].rolling(40).mean()
    weekly["ma20_slope4_pct"] = (weekly["ma20"] / weekly["ma20"].shift(4) - 1.0) * 100.0
    weekly["return_13w_pct"] = (weekly["close"] / weekly["close"].shift(13) - 1.0) * 100.0
    weekly["return_26w_pct"] = (weekly["close"] / weekly["close"].shift(26) - 1.0) * 100.0
    return weekly.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 事件定义与未来路径
# -----------------------------------------------------------------------------
def trend_state(row: pd.Series, price_tolerance_pct: float) -> str:
    values = [row.get("close"), row.get("ma20"), row.get("ma40"), row.get("ma20_slope4_pct")]
    if any(pd.isna(value) for value in values):
        return "数据不足"
    close, ma20, ma40, slope = map(float, values)
    tolerance = price_tolerance_pct / 100.0
    if ma20 > ma40 and slope > 0 and close >= ma20 * (1.0 - tolerance):
        return "上升趋势"
    if ma20 < ma40 and slope < 0 and close <= ma20 * (1.0 + tolerance):
        return "下降趋势"
    return "中性趋势"


def zero_axis_state(row: pd.Series) -> str:
    dif, dea = float(row["dif"]), float(row["dea"])
    if dif > 0 and dea > 0:
        return "DIF与DEA均在零轴上"
    if dif < 0 and dea < 0:
        return "DIF与DEA均在零轴下"
    return "DIF与DEA跨零轴"


def pullback_before_first_red(weekly: pd.DataFrame, position: int) -> dict[str, Any]:
    # position 是第一根红柱；向前寻找与其直接相连的连续绿柱段。
    green_start = position - 1
    while green_start >= 0 and float(weekly.iloc[green_start]["hist"]) <= 0:
        green_start -= 1
    first_green = green_start + 1
    green_weeks = max(position - first_green, 0)
    if green_weeks == 0:
        return {
            "Green_Weeks": 0,
            "Pullback_Depth_pct": np.nan,
            "Pre_Peak": np.nan,
            "Pullback_Low": np.nan,
        }

    # 用绿柱开始前最多4周的最高点作为回调起点，避免把多年以前的高点带入。
    peak_start = max(0, first_green - 4)
    peak_end = max(first_green, 1)
    pre_peak = pd.to_numeric(
        weekly.iloc[peak_start:peak_end]["high"], errors="coerce"
    ).max()
    pullback_low = pd.to_numeric(
        weekly.iloc[first_green:position]["low"], errors="coerce"
    ).min()
    depth = (1.0 - pullback_low / pre_peak) * 100.0 if pre_peak and pre_peak > 0 else np.nan
    return {
        "Green_Weeks": int(green_weeks),
        "Pullback_Depth_pct": float(depth) if pd.notna(depth) else np.nan,
        "Pre_Peak": float(pre_peak) if pd.notna(pre_peak) else np.nan,
        "Pullback_Low": float(pullback_low) if pd.notna(pullback_low) else np.nan,
    }


def material_hist_moves(
    red_values: list[float],
    material_change_pct: float,
) -> dict[str, Any]:
    """识别有意义的缩短及“缩短后再扩张”，忽略小于阈值的轻微噪声。"""
    first_shrink_week = np.nan
    shrink_count = 0
    reexpansion_count = 0
    armed_for_reexpansion = False
    for index in range(1, len(red_values)):
        previous = float(red_values[index - 1])
        current = float(red_values[index])
        if previous <= 0:
            continue
        change_pct = (current / previous - 1.0) * 100.0
        if change_pct <= -material_change_pct:
            shrink_count += 1
            if not np.isfinite(first_shrink_week):
                first_shrink_week = float(index + 1)
            armed_for_reexpansion = True
        elif change_pct >= material_change_pct and armed_for_reexpansion:
            reexpansion_count += 1
            armed_for_reexpansion = False
    return {
        "First_Material_Shrink_Week": first_shrink_week,
        "Material_Shrink_Count": int(shrink_count),
        "ReExpansion_Count": int(reexpansion_count),
    }


def build_red_cycle_features(
    weekly: pd.DataFrame,
    cycle_start_position: int,
    long_cycle_min_weeks: int,
    material_change_pct: float,
    short_strength_ratio: float,
) -> dict[str, Any]:
    """完整红柱周期的事后特征；Cycle_Type 只能用于研究，不能回填到买入日。"""
    start = int(cycle_start_position)
    if start < 0 or start >= len(weekly) or float(weekly.iloc[start]["hist"]) <= 0:
        return {}

    red_end = start
    while red_end + 1 < len(weekly) and float(weekly.iloc[red_end + 1]["hist"]) > 0:
        red_end += 1
    cycle_completed = red_end + 1 < len(weekly) and float(weekly.iloc[red_end + 1]["hist"]) <= 0
    red_rows = weekly.iloc[start:red_end + 1]
    red_values = pd.to_numeric(red_rows["hist"], errors="coerce").astype(float).tolist()

    green_start = start - 1
    while green_start >= 0 and float(weekly.iloc[green_start]["hist"]) <= 0:
        green_start -= 1
    previous_green = weekly.iloc[green_start + 1:start]
    green_values = pd.to_numeric(previous_green.get("hist", pd.Series(dtype=float)), errors="coerce")
    green_abs_peak = abs(float(green_values.min())) if len(green_values) else np.nan
    green_abs_area = float(green_values.abs().sum()) if len(green_values) else np.nan

    moves = material_hist_moves(red_values, material_change_pct)
    cycle_weeks = len(red_values)
    peak_hist = max(red_values)
    peak_week = red_values.index(peak_hist) + 1
    red_area = float(sum(red_values))
    peak_to_green = (
        peak_hist / green_abs_peak if np.isfinite(green_abs_peak) and green_abs_peak > 0 else np.nan
    )
    area_to_green = (
        red_area / green_abs_area if np.isfinite(green_abs_area) and green_abs_area > 0 else np.nan
    )
    strength_class = (
        "短小" if np.isfinite(peak_to_green) and peak_to_green < short_strength_ratio
        else "正常或较强" if np.isfinite(peak_to_green)
        else "前绿柱不足"
    )

    if not cycle_completed:
        cycle_type = "未完成_截至行情末日仍为红柱"
    elif cycle_weeks == 1:
        cycle_type = "A_第一根红柱后立即翻绿"
    elif cycle_weeks < long_cycle_min_weeks:
        cycle_type = f"B_2至{long_cycle_min_weeks - 1}周短周期"
    elif moves["ReExpansion_Count"] > 0:
        cycle_type = "C1_长周期_缩短后再扩张"
    else:
        cycle_type = "C2_长周期_单峰扩张后缩短"

    signal_close = float(weekly.iloc[start]["close"])
    last_red_close = float(red_rows.iloc[-1]["close"])
    first_green_date = (
        str(weekly.iloc[red_end + 1]["trade_date"]) if cycle_completed else ""
    )
    hist_changes = [
        (red_values[index] / red_values[index - 1] - 1.0) * 100.0
        for index in range(1, len(red_values))
        if red_values[index - 1] > 0
    ]
    return {
        "Cycle_Completed": bool(cycle_completed),
        "Cycle_Censored": bool(not cycle_completed),
        "Cycle_Type": cycle_type,
        "Cycle_Strength_Class": strength_class,
        "Red_Cycle_Weeks": int(cycle_weeks),
        "Last_Red_Date": str(red_rows.iloc[-1]["trade_date"]),
        "First_Green_Date": first_green_date,
        "Red_Hist_Sequence": "|".join(f"{value:.8f}" for value in red_values),
        "Red_Hist_Change_Sequence_pct": "|".join(
            f"{value:.2f}" for value in hist_changes
        ),
        "Peak_Red_Hist": float(peak_hist),
        "Peak_Red_Week": int(peak_week),
        "Red_Hist_Area": red_area,
        "PreGreen_Abs_Peak_Hist": green_abs_peak,
        "PreGreen_Abs_Hist_Area": green_abs_area,
        "Peak_Red_Hist_to_Close_pct": peak_hist / signal_close * 100.0,
        "Peak_Red_to_PreGreen_Peak_Ratio": peak_to_green,
        "Red_Area_to_PreGreen_Area_Ratio": area_to_green,
        "Cycle_Close_Return_pct": (last_red_close / signal_close - 1.0) * 100.0,
        "Cycle_Max_High_Return_pct": (
            float(pd.to_numeric(red_rows["high"], errors="coerce").max()) / signal_close - 1.0
        ) * 100.0,
        "Cycle_Min_Low_Return_pct": (
            float(pd.to_numeric(red_rows["low"], errors="coerce").min()) / signal_close - 1.0
        ) * 100.0,
        **moves,
    }


def is_main_board(ts_code: str) -> bool:
    return not ts_code.startswith(("300", "301", "688", "689"))


def first_hit_result(
    future: pd.DataFrame,
    entry_price: float,
    target_pct: float,
    stop_pct: float,
) -> tuple[str, str, float]:
    target_price = entry_price * (1.0 + target_pct / 100.0)
    stop_price = entry_price * (1.0 - stop_pct / 100.0)
    for day_no, row in enumerate(future.itertuples(index=False), start=1):
        hit_target = float(row.high) >= target_price
        hit_stop = float(row.low) <= stop_price
        trade_date = str(row.trade_date)
        if hit_target and hit_stop:
            return "同日不确定_按止损", trade_date, float(day_no)
        if hit_stop:
            return "止损先到", trade_date, float(day_no)
        if hit_target:
            return "目标先到", trade_date, float(day_no)
    return "八周均未触发", "", np.nan


def simulate_fixed_exit(
    future: pd.DataFrame,
    entry_price: float,
    target_pct: float,
    stop_pct: float,
    sell_slippage_pct: float = 0.0,
) -> dict[str, Any]:
    """
    模拟固定止盈/止损的可执行退出。

    - 跳空低开越过止损：按开盘价退出，保留真实跳空损失。
    - 跳空高开越过止盈：按开盘价退出，保留价格改善。
    - 日内同时触及止损和止盈：无法判定先后，保守按止损价退出。
    - 八周均未触发：按八周窗口最后一个可用收盘价退出。
    """
    target_price = entry_price * (1.0 + target_pct / 100.0)
    stop_price = entry_price * (1.0 - stop_pct / 100.0)
    ordered = future.sort_values("trade_date")
    for day_no, row in enumerate(ordered.itertuples(index=False), start=1):
        open_price = float(row.open)
        high_price = float(row.high)
        low_price = float(row.low)
        trade_date = str(row.trade_date)
        if open_price <= stop_price:
            exit_price, reason = open_price, "跳空止损"
        elif open_price >= target_price:
            exit_price, reason = open_price, "跳空止盈"
        else:
            hit_target = high_price >= target_price
            hit_stop = low_price <= stop_price
            if hit_target and hit_stop:
                exit_price, reason = stop_price, "同日双触发_按止损"
            elif hit_stop:
                exit_price, reason = stop_price, "止损"
            elif hit_target:
                exit_price, reason = target_price, "止盈"
            else:
                continue
        executable_price = float(exit_price) * (1.0 - sell_slippage_pct / 100.0)
        return {
            "date": trade_date,
            "price": executable_price,
            "return_pct": (executable_price / entry_price - 1.0) * 100.0,
            "holding_days": float(day_no),
            "reason": reason,
        }

    if ordered.empty:
        return {
            "date": "", "price": np.nan, "return_pct": np.nan,
            "holding_days": np.nan, "reason": "无行情",
        }
    last = ordered.iloc[-1]
    exit_price = float(last["close"]) * (1.0 - sell_slippage_pct / 100.0)
    return {
        "date": str(last["trade_date"]),
        "price": exit_price,
        "return_pct": (exit_price / entry_price - 1.0) * 100.0,
        "holding_days": float(len(ordered)),
        "reason": "八周到期",
    }


def path_on_or_before(daily: pd.DataFrame, checkpoint: str, entry_date: str) -> float:
    subset = daily[(daily["trade_date"] >= entry_date) & (daily["trade_date"] <= checkpoint)]
    return float(subset.iloc[-1]["close"]) if not subset.empty else np.nan


def evaluate_event_path(
    daily: pd.DataFrame,
    signal_date: str,
    open_dates: list[str],
    open_pos: dict[str, int],
    buy_slippage_pct: float,
    sell_slippage_pct: float,
    stop_threshold_pct: float,
    ts_code: str,
) -> dict[str, Any]:
    empty = {
        "Tradable": False,
        "Untradable_Reason": "未来交易日不足",
        "Entry_Date": "",
        "Entry_Price": np.nan,
        "Has_8W_Future": False,
        "MFE_8W_pct": np.nan,
        "MAE_8W_pct": np.nan,
        "Return_8W_pct": np.nan,
        "Hit_Stop_8W": np.nan,
    }
    for week in range(1, HOLD_WEEKS + 1):
        empty[f"Return_W{week}_pct"] = np.nan
    for target in [10, 20, 30]:
        empty[f"Hit_{target}_8W"] = np.nan
        empty[f"Hit_{target}_Date"] = ""
        empty[f"Days_To_{target}"] = np.nan
        empty[f"First_{target}_vs_Stop"] = ""
        empty[f"Exit_T{target}_Date"] = ""
        empty[f"Exit_T{target}_Price"] = np.nan
        empty[f"Exit_T{target}_Return_pct"] = np.nan
        empty[f"Exit_T{target}_Holding_Days"] = np.nan
        empty[f"Exit_T{target}_Reason"] = ""

    if signal_date not in open_pos:
        empty["Untradable_Reason"] = "信号日不在交易日历"
        return empty
    signal_pos = open_pos[signal_date]
    entry_pos = signal_pos + 1
    horizon_pos = entry_pos + HOLD_TRADING_DAYS - 1
    if entry_pos >= len(open_dates):
        return empty
    entry_date = open_dates[entry_pos]
    entry_rows = daily[daily["trade_date"] == entry_date]
    if entry_rows.empty:
        empty["Untradable_Reason"] = "D1停牌或无行情"
        empty["Entry_Date"] = entry_date
        return empty

    entry_row = entry_rows.iloc[-1]
    if (
        is_main_board(ts_code)
        and float(entry_row["open"]) == float(entry_row["high"]) == float(entry_row["low"])
    ):
        empty["Untradable_Reason"] = "主板D1一字板"
        empty["Entry_Date"] = entry_date
        return empty

    entry_price = float(entry_row["open"]) * (1.0 + buy_slippage_pct / 100.0)
    empty.update({
        "Tradable": True,
        "Untradable_Reason": "",
        "Entry_Date": entry_date,
        "Entry_Price": entry_price,
    })
    if horizon_pos >= len(open_dates):
        empty["Untradable_Reason"] = "可买但未来不足40个市场交易日"
        return empty

    horizon_date = open_dates[horizon_pos]
    future = daily[
        (daily["trade_date"] >= entry_date) & (daily["trade_date"] <= horizon_date)
    ].copy().sort_values("trade_date")
    if future.empty:
        empty["Untradable_Reason"] = "八周窗口无行情"
        return empty

    empty["Has_8W_Future"] = True
    empty["MFE_8W_pct"] = (float(future["high"].max()) / entry_price - 1.0) * 100.0
    empty["MAE_8W_pct"] = (float(future["low"].min()) / entry_price - 1.0) * 100.0
    empty["Hit_Stop_8W"] = bool(float(future["low"].min()) <= entry_price * (1 - stop_threshold_pct / 100.0))

    end_close = path_on_or_before(daily, horizon_date, entry_date)
    empty["Return_8W_pct"] = (end_close / entry_price - 1.0) * 100.0 if pd.notna(end_close) else np.nan
    for week in range(1, HOLD_WEEKS + 1):
        checkpoint = open_dates[entry_pos + week * 5 - 1]
        close = path_on_or_before(daily, checkpoint, entry_date)
        empty[f"Return_W{week}_pct"] = (
            (close / entry_price - 1.0) * 100.0 if pd.notna(close) else np.nan
        )

    for target in [10, 20, 30]:
        target_price = entry_price * (1.0 + target / 100.0)
        hit_rows = future[future["high"] >= target_price]
        empty[f"Hit_{target}_8W"] = bool(not hit_rows.empty)
        if not hit_rows.empty:
            hit_date = str(hit_rows.iloc[0]["trade_date"])
            empty[f"Hit_{target}_Date"] = hit_date
            empty[f"Days_To_{target}"] = float(open_pos[hit_date] - entry_pos + 1)
        result, _, _ = first_hit_result(
            future, entry_price, float(target), float(stop_threshold_pct)
        )
        empty[f"First_{target}_vs_Stop"] = result
        exit_result = simulate_fixed_exit(
            future, entry_price, float(target), float(stop_threshold_pct),
            sell_slippage_pct=float(sell_slippage_pct),
        )
        if exit_result["date"] in open_pos:
            exit_result["holding_days"] = float(open_pos[exit_result["date"]] - entry_pos + 1)
        empty[f"Exit_T{target}_Date"] = exit_result["date"]
        empty[f"Exit_T{target}_Price"] = exit_result["price"]
        empty[f"Exit_T{target}_Return_pct"] = exit_result["return_pct"]
        empty[f"Exit_T{target}_Holding_Days"] = exit_result["holding_days"]
        empty[f"Exit_T{target}_Reason"] = exit_result["reason"]
    return empty


def checkpoint_state_from_hist(
    observed_hist: list[float],
    material_change_pct: float,
) -> tuple[str, dict[str, Any]]:
    """只根据截至检查周已经出现的柱体给出实时状态。"""
    first_green = next((i for i, value in enumerate(observed_hist) if value <= 0), None)
    if first_green is not None:
        red_values = observed_hist[:first_green]
        return f"已翻绿_第{first_green + 1}周", material_hist_moves(red_values, material_change_pct)
    red_values = observed_hist
    moves = material_hist_moves(red_values, material_change_pct)
    if moves["ReExpansion_Count"] > 0:
        state = "缩短后再扩张"
    elif moves["Material_Shrink_Count"] > 0:
        state = "缩短未再扩张"
    elif len(red_values) >= 2 and red_values[-1] >= red_values[0] * (1.0 + material_change_pct / 100.0):
        state = "持续扩张"
    else:
        state = "红柱平缓延续"
    return state, moves


def build_checkpoint_features(
    weekly: pd.DataFrame,
    cycle_start_position: int,
    daily: pd.DataFrame,
    path_result: dict[str, Any],
    open_dates: list[str],
    open_pos: dict[str, int],
    buy_slippage_pct: float,
    stop_threshold_pct: float,
    material_change_pct: float,
    short_strength_ratio: float,
    cycle_features: dict[str, Any],
) -> dict[str, Any]:
    """生成第2—5周可实时观察的状态，以及该时点之后的独立结果。"""
    output: dict[str, Any] = {}
    start = int(cycle_start_position)
    pre_green_peak = float(cycle_features.get("PreGreen_Abs_Peak_Hist", np.nan))

    for checkpoint_week in CHECKPOINT_WEEKS:
        prefix = f"CP_W{checkpoint_week}"
        position = start + checkpoint_week - 1
        defaults = {
            f"{prefix}_Observed": False,
            f"{prefix}_Date": "",
            f"{prefix}_State": "未观察到",
            f"{prefix}_Hist": np.nan,
            f"{prefix}_Hist_vs_W1_pct": np.nan,
            f"{prefix}_Peak_to_PreGreen_Ratio": np.nan,
            f"{prefix}_Material_Shrink_Count": np.nan,
            f"{prefix}_ReExpansion_Count": np.nan,
            f"{prefix}_Weak_Candidate": np.nan,
            f"{prefix}_Return_From_Entry_pct": np.nan,
            f"{prefix}_Remaining_MFE_pct": np.nan,
            f"{prefix}_Remaining_MAE_pct": np.nan,
            f"{prefix}_Remaining_Return_pct": np.nan,
            f"{prefix}_Stop_Hit_Before": np.nan,
            f"{prefix}_Remaining_Stop_Hit": np.nan,
            f"{prefix}_Delayed_Entry_Date": "",
            f"{prefix}_Delayed_Entry_Price": np.nan,
            f"{prefix}_Delayed_Has_8W_Future": False,
            f"{prefix}_Delayed_MFE_8W_pct": np.nan,
            f"{prefix}_Delayed_MAE_8W_pct": np.nan,
            f"{prefix}_Delayed_Return_8W_pct": np.nan,
            f"{prefix}_Delayed_Hit_Stop_8W": np.nan,
        }
        for target in (10, 20, 30):
            defaults[f"{prefix}_T{target}_Still_Open"] = np.nan
            defaults[f"{prefix}_T{target}_Future_Result"] = ""
            defaults[f"{prefix}_T{target}_Future_Target_First"] = np.nan
            defaults[f"{prefix}_Delayed_Hit_{target}_8W"] = np.nan
            defaults[f"{prefix}_Delayed_First_{target}_vs_Stop"] = ""
        output.update(defaults)
        if position >= len(weekly):
            continue

        checkpoint_row = weekly.iloc[position]
        checkpoint_date = str(checkpoint_row["trade_date"])
        observed = pd.to_numeric(
            weekly.iloc[start:position + 1]["hist"], errors="coerce"
        ).astype(float).tolist()
        state, moves = checkpoint_state_from_hist(observed, material_change_pct)
        positive_observed = [value for value in observed if value > 0]
        observed_peak = max(positive_observed) if positive_observed else np.nan
        peak_ratio = (
            observed_peak / pre_green_peak
            if np.isfinite(observed_peak) and np.isfinite(pre_green_peak) and pre_green_peak > 0
            else np.nan
        )
        weak_candidate = bool(
            state == "缩短未再扩张"
            and np.isfinite(peak_ratio)
            and peak_ratio < short_strength_ratio
        )
        output.update({
            f"{prefix}_Observed": True,
            f"{prefix}_Date": checkpoint_date,
            f"{prefix}_State": state,
            f"{prefix}_Hist": float(observed[-1]),
            f"{prefix}_Hist_vs_W1_pct": (
                (observed[-1] / observed[0] - 1.0) * 100.0 if observed[0] != 0 else np.nan
            ),
            f"{prefix}_Peak_to_PreGreen_Ratio": peak_ratio,
            f"{prefix}_Material_Shrink_Count": moves["Material_Shrink_Count"],
            f"{prefix}_ReExpansion_Count": moves["ReExpansion_Count"],
            f"{prefix}_Weak_Candidate": weak_candidate,
        })

        if not bool(path_result.get("Tradable", False)):
            continue
        entry_date = str(path_result.get("Entry_Date", ""))
        entry_price = float(path_result.get("Entry_Price", np.nan))
        if entry_date not in open_pos or checkpoint_date not in open_pos or not np.isfinite(entry_price):
            continue
        checkpoint_close = float(checkpoint_row["close"])
        output[f"{prefix}_Return_From_Entry_pct"] = (
            checkpoint_close / entry_price - 1.0
        ) * 100.0

        # 独立延迟买点：确认第N根周线状态后，下一市场交易日开盘买入，
        # 从这个新买点重新观察40个交易日，而不是沿用第一根红柱的原八周终点。
        checkpoint_market_pos = open_pos[checkpoint_date]
        delayed_entry_pos = checkpoint_market_pos + 1
        delayed_horizon_pos = delayed_entry_pos + HOLD_TRADING_DAYS - 1
        if delayed_horizon_pos < len(open_dates):
            delayed_entry_date = open_dates[delayed_entry_pos]
            delayed_horizon_date = open_dates[delayed_horizon_pos]
            delayed_entry_row = daily[daily["trade_date"].eq(delayed_entry_date)]
            delayed_path = daily[
                (daily["trade_date"] >= delayed_entry_date)
                & (daily["trade_date"] <= delayed_horizon_date)
            ].copy().sort_values("trade_date")
            if not delayed_entry_row.empty and not delayed_path.empty:
                delayed_entry_price = float(delayed_entry_row.iloc[-1]["open"]) * (
                    1.0 + buy_slippage_pct / 100.0
                )
                delayed_stop_price = delayed_entry_price * (1.0 - stop_threshold_pct / 100.0)
                output.update({
                    f"{prefix}_Delayed_Entry_Date": delayed_entry_date,
                    f"{prefix}_Delayed_Entry_Price": delayed_entry_price,
                    f"{prefix}_Delayed_Has_8W_Future": True,
                    f"{prefix}_Delayed_MFE_8W_pct": (
                        float(delayed_path["high"].max()) / delayed_entry_price - 1.0
                    ) * 100.0,
                    f"{prefix}_Delayed_MAE_8W_pct": (
                        float(delayed_path["low"].min()) / delayed_entry_price - 1.0
                    ) * 100.0,
                    f"{prefix}_Delayed_Return_8W_pct": (
                        float(delayed_path.iloc[-1]["close"]) / delayed_entry_price - 1.0
                    ) * 100.0,
                    f"{prefix}_Delayed_Hit_Stop_8W": bool(
                        float(delayed_path["low"].min()) <= delayed_stop_price
                    ),
                })
                for target in (10, 20, 30):
                    output[f"{prefix}_Delayed_Hit_{target}_8W"] = bool(
                        float(delayed_path["high"].max())
                        >= delayed_entry_price * (1.0 + target / 100.0)
                    )
                    result, _, _ = first_hit_result(
                        delayed_path, delayed_entry_price,
                        float(target), float(stop_threshold_pct),
                    )
                    output[f"{prefix}_Delayed_First_{target}_vs_Stop"] = result

        entry_position = open_pos[entry_date]
        horizon_position = entry_position + HOLD_TRADING_DAYS - 1
        if horizon_position >= len(open_dates):
            continue
        horizon_date = open_dates[horizon_position]
        history = daily[
            (daily["trade_date"] >= entry_date) & (daily["trade_date"] <= checkpoint_date)
        ].copy().sort_values("trade_date")
        remaining = daily[
            (daily["trade_date"] > checkpoint_date) & (daily["trade_date"] <= horizon_date)
        ].copy().sort_values("trade_date")
        stop_price = entry_price * (1.0 - stop_threshold_pct / 100.0)
        output[f"{prefix}_Stop_Hit_Before"] = bool(
            not history.empty and float(history["low"].min()) <= stop_price
        )
        if not remaining.empty:
            output[f"{prefix}_Remaining_MFE_pct"] = (
                float(remaining["high"].max()) / checkpoint_close - 1.0
            ) * 100.0
            output[f"{prefix}_Remaining_MAE_pct"] = (
                float(remaining["low"].min()) / checkpoint_close - 1.0
            ) * 100.0
            output[f"{prefix}_Remaining_Return_pct"] = (
                float(remaining.iloc[-1]["close"]) / checkpoint_close - 1.0
            ) * 100.0
            output[f"{prefix}_Remaining_Stop_Hit"] = bool(
                float(remaining["low"].min()) <= stop_price
            )

        for target in (10, 20, 30):
            prior_result, _, _ = first_hit_result(
                history, entry_price, float(target), float(stop_threshold_pct)
            ) if not history.empty else ("八周均未触发", "", np.nan)
            still_open = prior_result == "八周均未触发"
            output[f"{prefix}_T{target}_Still_Open"] = bool(still_open)
            if still_open and not remaining.empty:
                future_result, _, _ = first_hit_result(
                    remaining, entry_price, float(target), float(stop_threshold_pct)
                )
                output[f"{prefix}_T{target}_Future_Result"] = future_result
                output[f"{prefix}_T{target}_Future_Target_First"] = (
                    future_result == "目标先到"
                )
            elif not still_open:
                output[f"{prefix}_T{target}_Future_Result"] = "此前已退出"
    return output


def signal_market_snapshot(
    basic: pd.DataFrame,
    signal_date: str,
) -> dict[str, float]:
    if basic.empty:
        return {"Raw_Close": np.nan, "Circ_MV_Billion": np.nan, "Turnover_Rate": np.nan}
    row = basic[basic["trade_date"] == signal_date]
    if row.empty:
        return {"Raw_Close": np.nan, "Circ_MV_Billion": np.nan, "Turnover_Rate": np.nan}
    row = row.iloc[-1]
    circ = float(row["circ_mv"]) / 10000.0 if pd.notna(row.get("circ_mv")) else np.nan
    return {
        "Raw_Close": float(row["close"]) if pd.notna(row.get("close")) else np.nan,
        "Circ_MV_Billion": circ,
        "Turnover_Rate": float(row["turnover_rate"]) if pd.notna(row.get("turnover_rate")) else np.nan,
    }


def passes_signal_filters(
    snapshot: dict[str, float],
    min_price: float,
    min_mv: float,
    max_mv: float,
) -> tuple[bool, str]:
    price = snapshot["Raw_Close"]
    circ = snapshot["Circ_MV_Billion"]
    if not np.isfinite(price):
        return False, "缺少信号日原始收盘价"
    if price < min_price:
        return False, "低于最低股价"
    if not np.isfinite(circ):
        return False, "缺少历史流通市值"
    if circ < min_mv or circ > max_mv:
        return False, "流通市值不在范围"
    return True, ""


def build_event_record(
    *,
    event_type: str,
    cycle_id: str,
    cycle_start_position: int,
    position: int,
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    stock: pd.Series,
    membership: dict[str, str],
    pullback: dict[str, Any],
    signal_start: str,
    signal_end: str,
    open_dates: list[str],
    open_pos: dict[str, int],
    price_tolerance_pct: float,
    min_price: float,
    min_mv: float,
    max_mv: float,
    buy_slippage_pct: float,
    sell_slippage_pct: float,
    stop_threshold_pct: float,
    long_cycle_min_weeks: int,
    material_hist_change_pct: float,
    short_strength_ratio: float,
) -> tuple[dict[str, Any] | None, str]:
    row = weekly.iloc[position]
    signal_date = str(row["trade_date"])
    if signal_date < signal_start or signal_date > signal_end:
        return None, "事件不在研究区间"
    if not (str(stock["list_date"]) <= signal_date < str(stock["delist_date"])):
        return None, "当时未上市或已退市"
    if membership is None:
        return None, "当时不在历史科技池"

    snapshot = signal_market_snapshot(basic, signal_date)
    passed, reason = passes_signal_filters(snapshot, min_price, min_mv, max_mv)
    if not passed:
        return None, reason

    next_hist = float(weekly.iloc[position + 1]["hist"]) if position + 1 < len(weekly) else np.nan
    next_red = bool(next_hist > 0) if np.isfinite(next_hist) else np.nan
    immediate_green = bool(next_hist <= 0) if np.isfinite(next_hist) else np.nan
    cycle_features = build_red_cycle_features(
        weekly=weekly,
        cycle_start_position=cycle_start_position,
        long_cycle_min_weeks=long_cycle_min_weeks,
        material_change_pct=material_hist_change_pct,
        short_strength_ratio=short_strength_ratio,
    )

    record = {
        "Event_Type": event_type,
        "Cycle_ID": cycle_id,
        "ts_code": str(stock["ts_code"]),
        "name": str(stock["name"]),
        "market": str(stock["market"]),
        "exchange": str(stock.get("exchange", "")),
        "Sample_Board": str(stock.get("Sample_Board", sample_board(stock))),
        "Sample_Weight": float(stock.get("Sample_Weight", 1.0)),
        "Sample_Seed": int(stock.get("Sample_Seed", DEFAULT_SAMPLE_SEED)),
        "SW_L1": membership["l1"],
        "SW_L2": membership["l2"],
        "SW_L3": membership["l3"],
        "Signal_Date": signal_date,
        "Cycle_Start_Signal_Date": str(weekly.iloc[cycle_start_position]["trade_date"]),
        "Event_Week_In_Cycle": int(position - cycle_start_position + 1),
        "Weekly_Trend": trend_state(row, price_tolerance_pct),
        "Zero_Axis": zero_axis_state(row),
        "Hist": float(row["hist"]),
        "Hist_Prev": float(weekly.iloc[position - 1]["hist"]),
        "DIF": float(row["dif"]),
        "DEA": float(row["dea"]),
        "Weekly_Close": float(row["close"]),
        "W_MA20": float(row["ma20"]) if pd.notna(row["ma20"]) else np.nan,
        "W_MA40": float(row["ma40"]) if pd.notna(row["ma40"]) else np.nan,
        "W_MA20_Slope4_pct": float(row["ma20_slope4_pct"]) if pd.notna(row["ma20_slope4_pct"]) else np.nan,
        "Pre_13W_Return_pct": float(row["return_13w_pct"]) if pd.notna(row["return_13w_pct"]) else np.nan,
        "Pre_26W_Return_pct": float(row["return_26w_pct"]) if pd.notna(row["return_26w_pct"]) else np.nan,
        "Next_Week_Hist": next_hist,
        "Next_Week_Red": next_red,
        "Immediate_Green": immediate_green,
        **pullback,
        **snapshot,
        **cycle_features,
    }
    path_result = evaluate_event_path(
        daily=daily,
        signal_date=signal_date,
        open_dates=open_dates,
        open_pos=open_pos,
        buy_slippage_pct=buy_slippage_pct,
        sell_slippage_pct=sell_slippage_pct,
        stop_threshold_pct=stop_threshold_pct,
        ts_code=str(stock["ts_code"]),
    )
    record.update(path_result)
    if event_type == "第一根红柱":
        record.update(build_checkpoint_features(
            weekly=weekly,
            cycle_start_position=cycle_start_position,
            daily=daily,
            path_result=path_result,
            open_dates=open_dates,
            open_pos=open_pos,
            buy_slippage_pct=buy_slippage_pct,
            stop_threshold_pct=stop_threshold_pct,
            material_change_pct=material_hist_change_pct,
            short_strength_ratio=short_strength_ratio,
            cycle_features=cycle_features,
        ))
    return record, ""


def analyze_stock(
    stock: pd.Series,
    periods: list[dict[str, str]],
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    week_last_map: dict[pd.Timestamp, str],
    open_dates: list[str],
    open_pos: dict[str, int],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int], pd.DataFrame]:
    weekly = build_weekly(daily, week_last_map)
    if len(weekly) < MACD_WARMUP_WEEKS:
        return [], {"周线不足": 1}, weekly

    records: list[dict[str, Any]] = []
    rejects: dict[str, int] = {}
    active_cycle_id = ""
    active_first_red_position = -1
    active_pullback: dict[str, Any] = {}
    first_shrink_emitted = False

    for position in range(MACD_WARMUP_WEEKS, len(weekly)):
        hist = float(weekly.iloc[position]["hist"])
        hist_prev = float(weekly.iloc[position - 1]["hist"])
        hist_prev2 = float(weekly.iloc[position - 2]["hist"])
        signal_date = str(weekly.iloc[position]["trade_date"])

        is_first_red = hist > 0 and hist_prev <= 0
        is_first_shrink = (
            hist > 0 and hist_prev > 0 and hist < hist_prev
            and not (hist_prev2 > 0 and hist_prev < hist_prev2)
        )

        if is_first_red:
            active_cycle_id = f"{stock['ts_code']}|{signal_date}"
            active_first_red_position = position
            active_pullback = pullback_before_first_red(weekly, position)
            first_shrink_emitted = False
            membership = membership_on_date(periods, signal_date)
            record, reason = build_event_record(
                event_type="第一根红柱",
                cycle_id=active_cycle_id,
                cycle_start_position=active_first_red_position,
                position=position,
                weekly=weekly,
                daily=daily,
                basic=basic,
                stock=stock,
                membership=membership,
                pullback=active_pullback,
                signal_start=config["signal_start"],
                signal_end=config["signal_end"],
                open_dates=open_dates,
                open_pos=open_pos,
                price_tolerance_pct=config["price_tolerance_pct"],
                min_price=config["min_price"],
                min_mv=config["min_mv"],
                max_mv=config["max_mv"],
                buy_slippage_pct=config["buy_slippage_pct"],
                sell_slippage_pct=config["sell_slippage_pct"],
                stop_threshold_pct=config["stop_threshold_pct"],
                long_cycle_min_weeks=config["long_cycle_min_weeks"],
                material_hist_change_pct=config["material_hist_change_pct"],
                short_strength_ratio=config["short_strength_ratio"],
            )
            if record is not None:
                records.append(record)
            elif reason not in {"事件不在研究区间"}:
                key = f"第一根红柱|{reason}"
                rejects[key] = rejects.get(key, 0) + 1

        elif is_first_shrink and active_cycle_id and not first_shrink_emitted:
            first_shrink_emitted = True
            membership = membership_on_date(periods, signal_date)
            record, reason = build_event_record(
                event_type="红柱首次缩短",
                cycle_id=active_cycle_id,
                cycle_start_position=active_first_red_position,
                position=position,
                weekly=weekly,
                daily=daily,
                basic=basic,
                stock=stock,
                membership=membership,
                pullback=active_pullback,
                signal_start=config["signal_start"],
                signal_end=config["signal_end"],
                open_dates=open_dates,
                open_pos=open_pos,
                price_tolerance_pct=config["price_tolerance_pct"],
                min_price=config["min_price"],
                min_mv=config["min_mv"],
                max_mv=config["max_mv"],
                buy_slippage_pct=config["buy_slippage_pct"],
                sell_slippage_pct=config["sell_slippage_pct"],
                stop_threshold_pct=config["stop_threshold_pct"],
                long_cycle_min_weeks=config["long_cycle_min_weeks"],
                material_hist_change_pct=config["material_hist_change_pct"],
                short_strength_ratio=config["short_strength_ratio"],
            )
            if record is not None:
                records.append(record)
            elif reason not in {"事件不在研究区间"}:
                key = f"红柱首次缩短|{reason}"
                rejects[key] = rejects.get(key, 0) + 1

        if hist <= 0 and not is_first_red:
            active_cycle_id = ""
            active_first_red_position = -1
            active_pullback = {}
            first_shrink_emitted = False

    return records, rejects, weekly


# -----------------------------------------------------------------------------
# 汇总报表
# -----------------------------------------------------------------------------
def aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    grouped = frame.groupby(group_columns, dropna=False, sort=False)
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        full = group[group["Tradable"].eq(True) & group["Has_8W_Future"].eq(True)]
        next_known = group[group["Next_Week_Red"].notna()]
        row.update({
            "信号数": int(len(group)),
            "次周状态有效样本": int(len(next_known)),
            "下一周继续红柱(%)": pct_mean(next_known["Next_Week_Red"]),
            "下一周立即翻绿(%)": pct_mean(next_known["Immediate_Green"]),
            "八周完整可交易样本": int(len(full)),
            "八周触及10%(%)": pct_mean(full["Hit_10_8W"]),
            "八周触及20%(%)": pct_mean(full["Hit_20_8W"]),
            "八周触及30%(%)": pct_mean(full["Hit_30_8W"]),
            "10%先于止损(%)": float(
                full["First_10_vs_Stop"].eq("目标先到").mean() * 100.0
            ) if len(full) else np.nan,
            "触发止损阈值(%)": pct_mean(full["Hit_Stop_8W"]),
            "八周末平均收益(%)": numeric_mean(full["Return_8W_pct"]),
            "八周末收益中位数(%)": numeric_median(full["Return_8W_pct"]),
            "八周最大浮盈均值(%)": numeric_mean(full["MFE_8W_pct"]),
            "八周最大回撤均值(%)": numeric_mean(full["MAE_8W_pct"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_comparison(events: pd.DataFrame) -> pd.DataFrame:
    full = events[events["Tradable"].eq(True) & events["Has_8W_Future"].eq(True)].copy()
    first = full[full["Event_Type"].eq("第一根红柱")].drop_duplicates("Cycle_ID")
    shrink = full[full["Event_Type"].eq("红柱首次缩短")].drop_duplicates("Cycle_ID")
    if first.empty or shrink.empty:
        return pd.DataFrame()
    columns = [
        "Cycle_ID", "ts_code", "name", "Signal_Date", "Entry_Date", "Entry_Price",
        "Weekly_Trend", "Zero_Axis", "Pullback_Depth_pct",
        "Return_8W_pct", "MFE_8W_pct", "MAE_8W_pct", "Hit_Stop_8W",
        "Hit_10_8W", "Hit_20_8W", "Hit_30_8W",
    ]
    paired = first[columns].merge(
        shrink[columns], on="Cycle_ID", how="inner", suffixes=("_FirstRed", "_Shrink")
    )
    paired["Shrink_Minus_FirstRed_Return_pct"] = (
        paired["Return_8W_pct_Shrink"] - paired["Return_8W_pct_FirstRed"]
    )
    paired["Shrink_Minus_FirstRed_MFE_pct"] = (
        paired["MFE_8W_pct_Shrink"] - paired["MFE_8W_pct_FirstRed"]
    )
    paired["Shrink_Entry_Price_Advantage_pct"] = (
        paired["Entry_Price_FirstRed"] / paired["Entry_Price_Shrink"] - 1.0
    ) * 100.0
    paired["Shrink_Failure_Increase"] = (
        paired["Hit_Stop_8W_Shrink"].astype(int)
        - paired["Hit_Stop_8W_FirstRed"].astype(int)
    )
    return paired


def paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "完整配对周期": int(len(paired)),
        "缩短买入平均价格优势(%)": numeric_mean(paired["Shrink_Entry_Price_Advantage_pct"]),
        "缩短买入八周收益变化(百分点)": numeric_mean(paired["Shrink_Minus_FirstRed_Return_pct"]),
        "缩短买入最大浮盈变化(百分点)": numeric_mean(paired["Shrink_Minus_FirstRed_MFE_pct"]),
        "第一根红柱止损触发率(%)": pct_mean(paired["Hit_Stop_8W_FirstRed"]),
        "红柱缩短止损触发率(%)": pct_mean(paired["Hit_Stop_8W_Shrink"]),
        "缩短买入失败率增加(百分点)": numeric_mean(paired["Shrink_Failure_Increase"]) * 100.0,
        "第一根红柱触及20%(%)": pct_mean(paired["Hit_20_8W_FirstRed"]),
        "红柱缩短触及20%(%)": pct_mean(paired["Hit_20_8W_Shrink"]),
    }])


def pullback_bin(series: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=["<10%", "10%-20%", "20%-30%", ">=30%"],
        right=False,
    )


def style_percent_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    numeric = frame.copy()
    for column in numeric.columns:
        if "(%)" in column or "百分点" in column:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce").round(2)
    return numeric


def build_all_summaries(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    first_red = events[events["Event_Type"].eq("第一根红柱")].copy()
    first_red["回调深度分组"] = pullback_bin(first_red["Pullback_Depth_pct"])
    return {
        "趋势_第一根红柱": aggregate(first_red, ["Weekly_Trend"]),
        "趋势与零轴_第一根红柱": aggregate(first_red, ["Weekly_Trend", "Zero_Axis"]),
        "回调深度_第一根红柱": aggregate(first_red, ["Weekly_Trend", "回调深度分组"]),
        "买点类型对比": aggregate(events, ["Event_Type"]),
        "买点类型与趋势": aggregate(events, ["Event_Type", "Weekly_Trend"]),
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    weight_values = pd.to_numeric(weights, errors="coerce")
    valid = numeric.notna() & weight_values.notna() & weight_values.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(numeric[valid], weights=weight_values[valid]))


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    weight_values = pd.to_numeric(weights, errors="coerce")
    valid = numeric.notna() & weight_values.notna() & weight_values.gt(0)
    if not valid.any():
        return np.nan
    order = np.argsort(numeric[valid].to_numpy())
    sorted_values = numeric[valid].to_numpy()[order]
    sorted_weights = weight_values[valid].to_numpy()[order]
    cutoff = sorted_weights.sum() / 2.0
    position = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(position, len(sorted_values) - 1)])


def weighted_rate(mask: pd.Series, weights: pd.Series) -> float:
    values = mask.astype(float)
    result = weighted_mean(values, weights)
    return result * 100.0 if np.isfinite(result) else np.nan


def weighted_top_removed_mean(
    values: pd.Series,
    weights: pd.Series,
    remove_n: int,
) -> float:
    frame = pd.DataFrame({
        "value": pd.to_numeric(values, errors="coerce"),
        "weight": pd.to_numeric(weights, errors="coerce"),
    }).dropna()
    if len(frame) <= remove_n:
        return np.nan
    frame = frame.sort_values("value", ascending=False).iloc[remove_n:]
    return weighted_mean(frame["value"], frame["weight"])


def weighted_trimmed_mean(
    values: pd.Series,
    weights: pd.Series,
    trim_fraction: float = 0.10,
) -> float:
    frame = pd.DataFrame({
        "value": pd.to_numeric(values, errors="coerce"),
        "weight": pd.to_numeric(weights, errors="coerce"),
    }).dropna().sort_values("value")
    trim_n = int(math.floor(len(frame) * trim_fraction))
    if trim_n and len(frame) > trim_n * 2:
        frame = frame.iloc[trim_n:-trim_n]
    return weighted_mean(frame["value"], frame["weight"]) if len(frame) else np.nan


def top_positive_contribution(
    values: pd.Series,
    weights: pd.Series,
    top_n: int,
) -> float:
    frame = pd.DataFrame({
        "value": pd.to_numeric(values, errors="coerce"),
        "weight": pd.to_numeric(weights, errors="coerce"),
    }).dropna()
    frame = frame[frame["value"].gt(0)].copy()
    if frame.empty:
        return np.nan
    frame["contribution"] = frame["value"] * frame["weight"]
    denominator = float(frame["contribution"].sum())
    if denominator <= 0:
        return np.nan
    numerator = float(frame.nlargest(top_n, "contribution")["contribution"].sum())
    return numerator / denominator * 100.0


def strategy_definitions(events: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    first_red = events["Event_Type"].eq("第一根红柱")
    uptrend = events["Weekly_Trend"].eq("上升趋势")
    pullback = pd.to_numeric(events["Pullback_Depth_pct"], errors="coerce")
    return [
        ("基准_全部第一根红柱", first_red),
        ("实验A_上升趋势第一根红柱", first_red & uptrend),
        ("实验B_上升趋势第一根红柱且回调<30%", first_red & uptrend & pullback.lt(30.0)),
        ("对照_红柱首次缩短", events["Event_Type"].eq("红柱首次缩短")),
    ]


def strategy_scope_row(
    strategy_name: str,
    scope_name: str,
    group: pd.DataFrame,
    use_population_weights: bool,
) -> dict[str, Any]:
    if group.empty:
        return {"策略组": strategy_name, "统计口径": scope_name, "信号数": 0}
    weights = (
        pd.to_numeric(group["Sample_Weight"], errors="coerce").fillna(1.0)
        if use_population_weights else pd.Series(1.0, index=group.index)
    )
    next_known = group[group["Next_Week_Red"].notna()]
    next_weights = weights.loc[next_known.index]
    full = group[group["Tradable"].eq(True) & group["Has_8W_Future"].eq(True)]
    full_weights = weights.loc[full.index]
    row: dict[str, Any] = {
        "策略组": strategy_name,
        "统计口径": scope_name,
        "信号数": int(len(group)),
        "估算股票池信号数": float(weights.sum()),
        "次周状态有效样本": int(len(next_known)),
        "下一周继续红柱(%)": weighted_rate(next_known["Next_Week_Red"], next_weights),
        "下一周立即翻绿(%)": weighted_rate(next_known["Immediate_Green"], next_weights),
        "八周完整可交易样本": int(len(full)),
        "估算股票池完整样本数": float(full_weights.sum()),
    }
    for target in (10, 20, 30):
        row[f"八周触及{target}(%)"] = weighted_rate(full[f"Hit_{target}_8W"], full_weights)
        row[f"{target}%先于止损(%)"] = weighted_rate(
            full[f"First_{target}_vs_Stop"].eq("目标先到"), full_weights
        )
        exit_column = f"Exit_T{target}_Return_pct"
        row[f"T{target}止盈策略平均收益(%)"] = weighted_mean(full[exit_column], full_weights)
        row[f"T{target}止盈策略收益中位数(%)"] = weighted_median(full[exit_column], full_weights)
        row[f"T{target}止盈策略胜率(%)"] = weighted_rate(full[exit_column].gt(0), full_weights)
        row[f"T{target}平均持有交易日"] = weighted_mean(
            full[f"Exit_T{target}_Holding_Days"], full_weights
        )
    row.update({
        "触发止损阈值(%)": weighted_rate(full["Hit_Stop_8W"], full_weights),
        "八周末平均收益(%)": weighted_mean(full["Return_8W_pct"], full_weights),
        "八周末收益中位数(%)": weighted_median(full["Return_8W_pct"], full_weights),
        "八周末10%截尾均值(%)": weighted_trimmed_mean(
            full["Return_8W_pct"], full_weights, 0.10
        ),
        "去掉最高3个后均值(%)": weighted_top_removed_mean(
            full["Return_8W_pct"], full_weights, 3
        ),
        "去掉最高5个后均值(%)": weighted_top_removed_mean(
            full["Return_8W_pct"], full_weights, 5
        ),
        "八周末盈利率(%)": weighted_rate(full["Return_8W_pct"].gt(0), full_weights),
        "最高3个占正收益比例(%)": top_positive_contribution(
            full["Return_8W_pct"], full_weights, 3
        ),
        "最高5个占正收益比例(%)": top_positive_contribution(
            full["Return_8W_pct"], full_weights, 5
        ),
        "八周最大浮盈均值(%)": weighted_mean(full["MFE_8W_pct"], full_weights),
        "八周最大回撤均值(%)": weighted_mean(full["MAE_8W_pct"], full_weights),
    })
    return row


def build_strategy_report(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy_name, mask in strategy_definitions(events):
        strategy_events = events[mask].copy()
        rows.append(strategy_scope_row(
            strategy_name, "全量科技股等权", strategy_events, False
        ))
        rows.append(strategy_scope_row(
            strategy_name, "按完整股票池板块占比加权", strategy_events, True
        ))
        for board in SAMPLE_BOARDS:
            board_events = strategy_events[strategy_events["Sample_Board"].eq(board)]
            rows.append(strategy_scope_row(
                strategy_name, f"分板块_{board}", board_events, True
            ))
    return pd.DataFrame(rows)


def cycle_report_row(
    cycle_type: str,
    scope_name: str,
    group: pd.DataFrame,
    use_population_weights: bool,
    trend_scope: str = "全部趋势",
) -> dict[str, Any]:
    weights = (
        pd.to_numeric(group["Sample_Weight"], errors="coerce").fillna(1.0)
        if use_population_weights else pd.Series(1.0, index=group.index)
    )
    full = group[group["Tradable"].eq(True) & group["Has_8W_Future"].eq(True)]
    full_weights = weights.loc[full.index]
    row: dict[str, Any] = {
        "周期类型": cycle_type,
        "趋势范围": trend_scope,
        "统计口径": scope_name,
        "周期样本数": int(len(group)),
        "估算股票池周期数": float(weights.sum()),
        "完整红柱周期数": int(group["Cycle_Completed"].eq(True).sum()),
        "八周完整可交易样本": int(len(full)),
        "红柱周期周数均值": weighted_mean(group["Red_Cycle_Weeks"], weights),
        "红柱周期周数中位数": weighted_median(group["Red_Cycle_Weeks"], weights),
        "红柱峰值周中位数": weighted_median(group["Peak_Red_Week"], weights),
        "红柱峰值/前绿柱峰值中位数": weighted_median(
            group["Peak_Red_to_PreGreen_Peak_Ratio"], weights
        ),
        "红柱面积/前绿柱面积中位数": weighted_median(
            group["Red_Area_to_PreGreen_Area_Ratio"], weights
        ),
        "八周触及止损(%)": weighted_rate(full["Hit_Stop_8W"], full_weights),
        "八周末平均收益(%)": weighted_mean(full["Return_8W_pct"], full_weights),
        "八周末收益中位数(%)": weighted_median(full["Return_8W_pct"], full_weights),
        "八周末10%截尾均值(%)": weighted_trimmed_mean(
            full["Return_8W_pct"], full_weights
        ),
        "八周末去最高3只均值(%)": weighted_top_removed_mean(
            full["Return_8W_pct"], full_weights, 3
        ),
        "最高3只占正收益贡献(%)": top_positive_contribution(
            full["Return_8W_pct"], full_weights, 3
        ),
        "八周末盈利率(%)": weighted_rate(full["Return_8W_pct"].gt(0), full_weights),
        "八周最大浮盈均值(%)": weighted_mean(full["MFE_8W_pct"], full_weights),
        "八周最大回撤均值(%)": weighted_mean(full["MAE_8W_pct"], full_weights),
    }
    for target in (10, 20, 30):
        row[f"{target}%先于止损(%)"] = weighted_rate(
            full[f"First_{target}_vs_Stop"].eq("目标先到"), full_weights
        )
        row[f"T{target}退出平均收益(%)"] = weighted_mean(
            full[f"Exit_T{target}_Return_pct"], full_weights
        )
    return row


def build_cycle_report(events: pd.DataFrame) -> pd.DataFrame:
    first_red = events[events["Event_Type"].eq("第一根红柱")].copy()
    rows: list[dict[str, Any]] = []
    trend_scopes = [
        ("全部趋势", first_red),
        ("仅上升趋势", first_red[first_red["Weekly_Trend"].eq("上升趋势")]),
    ]
    for trend_scope, scoped in trend_scopes:
        for cycle_type, group in scoped.groupby("Cycle_Type", dropna=False, sort=False):
            label = str(cycle_type)
            rows.append(cycle_report_row(
                label, "全量科技股等权", group, False, trend_scope
            ))
            rows.append(cycle_report_row(
                label, "按完整股票池板块占比加权", group, True, trend_scope
            ))
    return pd.DataFrame(rows)


def build_cycle_strength_report(events: pd.DataFrame) -> pd.DataFrame:
    first_red = events[events["Event_Type"].eq("第一根红柱")].copy()
    rows: list[dict[str, Any]] = []
    grouped = first_red.groupby(["Cycle_Type", "Cycle_Strength_Class"], dropna=False, sort=False)
    for (cycle_type, strength), group in grouped:
        for scope, weighted in [
            ("全量科技股等权", False),
            ("按完整股票池板块占比加权", True),
        ]:
            row = cycle_report_row(str(cycle_type), scope, group, weighted)
            row["红柱相对强度"] = str(strength)
            rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_report_row(
    checkpoint_week: int,
    state: str,
    group: pd.DataFrame,
    use_population_weights: bool,
    trend_scope: str = "全部趋势",
) -> dict[str, Any]:
    prefix = f"CP_W{checkpoint_week}"
    weights = (
        pd.to_numeric(group["Sample_Weight"], errors="coerce").fillna(1.0)
        if use_population_weights else pd.Series(1.0, index=group.index)
    )
    full = group[group["Tradable"].eq(True) & group["Has_8W_Future"].eq(True)]
    full_weights = weights.loc[full.index]
    stop_live = full[full[f"{prefix}_Stop_Hit_Before"].eq(False)]
    stop_live_weights = weights.loc[stop_live.index]
    row: dict[str, Any] = {
        "检查周": checkpoint_week,
        "当周可知状态": state,
        "趋势范围": trend_scope,
        "统计口径": "按完整股票池板块占比加权" if use_population_weights else "全量科技股等权",
        "观察样本数": int(len(group)),
        "八周完整样本数": int(len(full)),
        "弱反弹候选比例(%)": weighted_rate(full[f"{prefix}_Weak_Candidate"].eq(True), full_weights),
        "红柱峰值/前绿柱峰值中位数": weighted_median(
            full[f"{prefix}_Peak_to_PreGreen_Ratio"], full_weights
        ),
        "截至当周收益均值(%)": weighted_mean(full[f"{prefix}_Return_From_Entry_pct"], full_weights),
        "截至当周收益中位数(%)": weighted_median(full[f"{prefix}_Return_From_Entry_pct"], full_weights),
        "截至当周已触及止损(%)": weighted_rate(
            full[f"{prefix}_Stop_Hit_Before"].eq(True), full_weights
        ),
        "未止损仍存活样本": int(len(stop_live)),
        "存活样本之后再触及止损(%)": weighted_rate(
            stop_live[f"{prefix}_Remaining_Stop_Hit"].eq(True), stop_live_weights
        ),
        "当周以后最大浮盈均值(%)": weighted_mean(
            stop_live[f"{prefix}_Remaining_MFE_pct"], stop_live_weights
        ),
        "当周以后最大回撤均值(%)": weighted_mean(
            stop_live[f"{prefix}_Remaining_MAE_pct"], stop_live_weights
        ),
        "当周至第八周收益均值(%)": weighted_mean(
            stop_live[f"{prefix}_Remaining_Return_pct"], stop_live_weights
        ),
        "当周至第八周收益中位数(%)": weighted_median(
            stop_live[f"{prefix}_Remaining_Return_pct"], stop_live_weights
        ),
    }
    for target in (10, 20, 30):
        open_group = full[full[f"{prefix}_T{target}_Still_Open"].eq(True)]
        open_weights = weights.loc[open_group.index]
        known = open_group[open_group[f"{prefix}_T{target}_Future_Target_First"].notna()]
        known_weights = weights.loc[known.index]
        row[f"T{target}当周仍持仓样本"] = int(len(open_group))
        row[f"T{target}此后目标先于止损(%)"] = weighted_rate(
            known[f"{prefix}_T{target}_Future_Target_First"].eq(True), known_weights
        )
    return row


def build_checkpoint_report(events: pd.DataFrame) -> pd.DataFrame:
    first_red = events[events["Event_Type"].eq("第一根红柱")].copy()
    rows: list[dict[str, Any]] = []
    trend_scopes = [
        ("全部趋势", first_red),
        ("仅上升趋势", first_red[first_red["Weekly_Trend"].eq("上升趋势")]),
    ]
    for trend_scope, scoped in trend_scopes:
        for checkpoint_week in CHECKPOINT_WEEKS:
            prefix = f"CP_W{checkpoint_week}"
            observed = scoped[scoped[f"{prefix}_Observed"].eq(True)]
            for state, group in observed.groupby(f"{prefix}_State", dropna=False, sort=False):
                rows.append(checkpoint_report_row(
                    checkpoint_week, str(state), group, False, trend_scope
                ))
                rows.append(checkpoint_report_row(
                    checkpoint_week, str(state), group, True, trend_scope
                ))
    return pd.DataFrame(rows)


def build_delayed_entry_report(events: pd.DataFrame) -> pd.DataFrame:
    """严格比较第2—5周确认后、下一交易日开盘买入并重新持有八周。"""
    first_red = events[events["Event_Type"].eq("第一根红柱")].copy()
    rows: list[dict[str, Any]] = []
    for checkpoint_week in CHECKPOINT_WEEKS:
        prefix = f"CP_W{checkpoint_week}"
        observed = first_red[first_red[f"{prefix}_Observed"].eq(True)]
        for state, group in observed.groupby(f"{prefix}_State", dropna=False, sort=False):
            full = group[group[f"{prefix}_Delayed_Has_8W_Future"].eq(True)].copy()
            row = {
                "确认周": checkpoint_week,
                "当周可知状态": str(state),
                "观察样本": int(len(group)),
                "延迟买入完整八周样本": int(len(full)),
                # 下面一列是事后验证标签，不能作为实时选股条件。
                "最终持续至少9周比例(%)": pct_mean(full["Red_Cycle_Weeks"].ge(9)),
                "延迟买入八周触及10%(%)": pct_mean(full[f"{prefix}_Delayed_Hit_10_8W"]),
                "延迟买入八周触及20%(%)": pct_mean(full[f"{prefix}_Delayed_Hit_20_8W"]),
                "延迟买入八周触及30%(%)": pct_mean(full[f"{prefix}_Delayed_Hit_30_8W"]),
                "延迟买入八周平均收益(%)": numeric_mean(full[f"{prefix}_Delayed_Return_8W_pct"]),
                "延迟买入八周收益中位数(%)": numeric_median(full[f"{prefix}_Delayed_Return_8W_pct"]),
                "延迟买入八周盈利率(%)": pct_mean(full[f"{prefix}_Delayed_Return_8W_pct"].gt(0)),
                "延迟买入最大浮盈中位数(%)": numeric_median(full[f"{prefix}_Delayed_MFE_8W_pct"]),
                "延迟买入最大回撤中位数(%)": numeric_median(full[f"{prefix}_Delayed_MAE_8W_pct"]),
                "延迟买入触及止损(%)": pct_mean(full[f"{prefix}_Delayed_Hit_Stop_8W"]),
            }
            for target in (10, 20, 30):
                row[f"延迟买入{target}%先于止损(%)"] = pct_mean(
                    full[f"{prefix}_Delayed_First_{target}_vs_Stop"].eq("目标先到")
                )
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# V5.0透明评分：第一根红柱产生候选，V40.6形态只评分不否决
# -----------------------------------------------------------------------------
def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "是"}


def finite_num(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def fetch_index_history(
    ts_code: str, start_date: str, end_date: str,
    use_cache: bool, api_pause: float,
) -> pd.DataFrame:
    path = os.path.join(
        CACHE_DIR,
        f"index_{ts_code.replace('.', '_')}_{cache_key(start_date, end_date)}.pkl",
    )
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as handle:
                cached = pickle.load(handle)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached
        except Exception:
            pass
    frame = safe_get(
        "index_daily", ts_code=ts_code, start_date=start_date, end_date=end_date,
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
    )
    if frame.empty:
        return frame
    frame["trade_date"] = frame["trade_date"].astype(str)
    for column in ("open", "high", "low", "close", "vol"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["trade_date", "open", "high", "low", "close"])
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date").reset_index(drop=True)
    )
    if use_cache and not frame.empty:
        atomic_pickle(frame, path)
    time.sleep(api_pause)
    return frame


def add_daily_v406_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy().sort_values("trade_date").reset_index(drop=True)
    frame["ma20_v49"] = frame["close"].rolling(20).mean()
    frame["ma60_v49"] = frame["close"].rolling(60).mean()
    frame["ma120_v49"] = frame["close"].rolling(120).mean()
    frame["ma5_vol_v49"] = frame["vol"].shift(1).rolling(5).mean()
    frame["box_high_10_v49"] = frame["high"].shift(1).rolling(10).max()
    ema12 = frame["close"].ewm(span=12, adjust=False).mean()
    ema26 = frame["close"].ewm(span=26, adjust=False).mean()
    frame["dif_v49"] = ema12 - ema26
    dea = frame["dif_v49"].ewm(span=9, adjust=False).mean()
    frame["macd_v49"] = (frame["dif_v49"] - dea) * 2.0
    price_range = frame["high"] - frame["low"]
    frame["body_ratio_v49"] = np.where(
        price_range.gt(0), (frame["close"] - frame["open"]) / price_range, np.nan
    )
    return frame


def weekly_row_on_date(weekly: pd.DataFrame, signal_date: str, exact: bool) -> tuple[int, pd.Series | None]:
    if weekly.empty:
        return -1, None
    eligible = weekly.index[weekly["trade_date"].astype(str).le(signal_date)]
    if len(eligible) == 0:
        return -1, None
    position = int(eligible[-1])
    row = weekly.iloc[position]
    if exact and str(row["trade_date"]) != signal_date:
        return -1, None
    return position, row


def v49_wave_count(weekly: pd.DataFrame, position: int) -> int:
    if position < 1:
        return -1
    window = weekly.iloc[max(0, position - 51):position + 1].copy().reset_index(drop=True)
    if len(window) < 26:
        return -1
    start = int(window["low"].idxmin())
    segment = window.iloc[start:].reset_index(drop=True)
    if len(segment) < 5:
        return 0
    running_max = finite_num(segment.iloc[0]["high"])
    in_pullback = False
    count = 0
    for index in range(1, len(segment)):
        row = segment.iloc[index]
        high = finite_num(row["high"])
        low = finite_num(row["low"])
        hist = finite_num(row["hist"])
        if not all(math.isfinite(value) for value in (high, low, hist, running_max)):
            continue
        if high > running_max:
            running_max = high
            in_pullback = False
            continue
        drawdown = (running_max - low) / running_max if running_max > 0 else 0.0
        if hist < 0 and drawdown >= 0.05 and not in_pullback:
            count += 1
            in_pullback = True
    return count


def bucket_rs(percentile: float) -> float:
    """板块内相对强度位置，最高20分；避免只追逐最极端的强势股。"""
    if not math.isfinite(percentile):
        return 0.0
    if 60 < percentile <= 80:
        return 20.0
    if 40 < percentile <= 60 or 80 < percentile <= 90:
        return 15.0
    if 20 < percentile <= 40 or 90 < percentile <= 100:
        return 10.0
    return 5.0


def bucket_breakout(value: float) -> float:
    """突破幅度质量，作为突破组中的0—4分，不再作为硬门槛。"""
    if not math.isfinite(value) or value <= 0:
        return 0.0
    if value <= 1:
        return 3.0
    if value <= 3:
        return 4.0
    if value <= 5:
        return 2.0
    return 1.0


def bucket_volume(value: float) -> float:
    """成交量质量0—15分；非标准放量只是不加高分，不再直接剔除。"""
    if not math.isfinite(value) or value < 0:
        return 0.0
    if value < 1.0:
        return 0.0
    if 1.0 <= value < 1.3:
        return 5.0
    if value < 1.6:
        return 12.0
    if value <= 2.2:
        return 15.0
    if value <= 2.6:
        return 10.0
    if value <= 3.0:
        return 6.0
    if value <= 4.0:
        return 3.0
    return 0.0


def bucket_bias(value: float) -> float:
    """周线乖离风险0—6分；高乖离不再一票否决。"""
    if not math.isfinite(value):
        return 0.0
    if value <= 15:
        return 6.0
    if value <= 25:
        return 4.0
    if value <= 35:
        return 2.0
    if value <= 45:
        return 1.0
    return 0.0


def bucket_body(value: float) -> float:
    """K线实体质量0—10分；实体不足60%仍可参加排序。"""
    if not math.isfinite(value):
        return 0.0
    if value <= 0:
        return 0.0
    if value < 0.30:
        return 2.0
    if value < 0.60:
        return 5.0
    if value <= 0.80:
        return 10.0
    if value <= 0.95:
        return 8.0
    return 5.0


def enrich_first_red_v49(
    events: pd.DataFrame,
    daily_histories: dict[str, pd.DataFrame],
    weekly_histories: dict[str, pd.DataFrame],
    board_weeklies: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    first = events[
        events["Event_Type"].astype(str).eq("第一根红柱")
        & events["Tradable"].map(to_bool)
        & events["Has_8W_Future"].map(to_bool)
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, event in first.iterrows():
        code = str(event["ts_code"])
        signal = normalize_date(event["Signal_Date"])
        daily = daily_histories.get(code, pd.DataFrame())
        weekly = weekly_histories.get(code, pd.DataFrame())
        dpos_rows = daily.index[daily["trade_date"].astype(str).eq(signal)]
        wpos, wrow = weekly_row_on_date(weekly, signal, exact=True)
        board_code = BOARD_INDEX.get(str(event.get("Sample_Board", "")), "")
        _, board_row = weekly_row_on_date(
            board_weeklies.get(board_code, pd.DataFrame()), signal, exact=False
        )
        if len(dpos_rows) == 0 or wrow is None or board_row is None or wpos < 1:
            continue
        dpos = int(dpos_rows[-1])
        if dpos < 121 or dpos + 1 >= len(daily):
            continue
        row = daily.iloc[dpos]
        previous = daily.iloc[dpos - 1]
        next_row = daily.iloc[dpos + 1]
        wprevious = weekly.iloc[wpos - 1]

        true_first = finite_num(wprevious["hist"]) <= 0 < finite_num(wrow["hist"])
        weekly_bias = (
            (finite_num(wrow["close"]) / finite_num(wrow["ma20"]) - 1.0) * 100.0
            if finite_num(wrow["ma20"]) > 0 else np.nan
        )
        previous_range = finite_num(wprevious["high"]) - finite_num(wprevious["low"])
        upper_shadow = finite_num(wprevious["high"]) - max(
            finite_num(wprevious["open"]), finite_num(wprevious["close"])
        )
        shadow_ratio = upper_shadow / previous_range if previous_range > 0 else 0.0
        trend_pass = finite_num(row["ma60_v49"]) > finite_num(row["ma120_v49"])
        box_pass = (
            finite_num(row["close"]) > finite_num(row["box_high_10_v49"])
            and finite_num(previous["close"]) <= finite_num(previous["box_high_10_v49"])
        )
        daily_breakout = finite_num(row["close"]) > finite_num(row["ma20_v49"]) * 1.02
        ma20_healthy = finite_num(row["ma20_v49"]) >= finite_num(previous["ma20_v49"])
        volume_ratio = (
            finite_num(row["vol"]) / finite_num(row["ma5_vol_v49"])
            if finite_num(row["ma5_vol_v49"]) > 0 else np.nan
        )
        volume_pass = math.isfinite(volume_ratio) and 1.3 <= volume_ratio <= 3.0
        body_ratio = finite_num(row["body_ratio_v49"])
        solid_pass = (
            finite_num(row["close"]) > finite_num(row["open"])
            and math.isfinite(body_ratio) and body_ratio >= 0.60
        )
        daily_macd = (
            finite_num(row["dif_v49"]) > 0
            and finite_num(row["macd_v49"]) > finite_num(previous["macd_v49"])
        )
        is_main = not code.startswith(("300", "301", "688", "689"))
        one_word = (
            is_main
            and finite_num(next_row["open"]) == finite_num(next_row["high"])
            == finite_num(next_row["low"])
        )
        gap = (
            (finite_num(next_row["open"]) / finite_num(row["close"]) - 1.0) * 100.0
            if finite_num(row["close"]) > 0 else np.nan
        )
        gap_pass = math.isfinite(gap) and -3.0 <= gap <= 5.0
        daily_pct = (
            (finite_num(row["close"]) / finite_num(previous["close"]) - 1.0) * 100.0
            if finite_num(previous["close"]) > 0 else np.nan
        )
        breakout_pct = (
            (finite_num(row["close"]) / finite_num(row["box_high_10_v49"]) - 1.0) * 100.0
            if finite_num(row["box_high_10_v49"]) > 0 else np.nan
        )
        board_rs13 = finite_num(wrow["return_13w_pct"]) - finite_num(board_row["return_13w_pct"])
        rows.append({
            **event.to_dict(),
            "True_First_Red_Audit": bool(true_first),
            "Wave_Count_True_Weekly": v49_wave_count(weekly, wpos),
            "Weekly_Bias_pct": weekly_bias,
            "Weekly_Bias_Pass": bool(not math.isfinite(weekly_bias) or weekly_bias <= 45.0),
            "Prev_Upper_Shadow_Ratio": shadow_ratio,
            "Weekly_Shadow_Pass": bool(shadow_ratio < 0.60),
            "Daily_Trend_Pass": bool(trend_pass),
            "Box_Breakout_Pass": bool(box_pass),
            "Daily_Breakout_Pass": bool(daily_breakout),
            "MA20_Healthy_Pass": bool(ma20_healthy),
            "Volume_Pass": bool(volume_pass),
            "Solid_Yang_Pass": bool(solid_pass),
            "Daily_MACD_Pass": bool(daily_macd),
            "One_Word_Pass": not one_word,
            "Gap_pct": gap,
            "Gap_Pass": bool(gap_pass),
            "Daily_pct": daily_pct,
            "Volume_Ratio": volume_ratio,
            "Box_Breakout_pct": breakout_pct,
            "Body_Ratio": body_ratio,
            "Board_Index": board_code,
            "Board_RS_13W_pct": board_rs13,
            "Original_V406_Score": (
                daily_pct * 10.0 + volume_ratio * 10.0
                if math.isfinite(daily_pct) and math.isfinite(volume_ratio) else np.nan
            ),
        })
    enriched = pd.DataFrame(rows)
    if enriched.empty:
        return enriched
    enriched = enriched.dropna(subset=["Board_RS_13W_pct"]).copy()
    enriched["RS13_Weekly_Percentile"] = (
        enriched.groupby("Signal_Date")["Board_RS_13W_pct"]
        .rank(method="average", pct=True) * 100.0
    )
    enriched["RS_Position_Score"] = enriched["RS13_Weekly_Percentile"].map(bucket_rs)
    enriched["Daily_Long_Trend_Score"] = np.where(enriched["Daily_Trend_Pass"], 10.0, 0.0)
    enriched["MA20_Trend_Score"] = np.where(enriched["MA20_Healthy_Pass"], 5.0, 0.0)
    enriched["Trend_Group_Score"] = (
        enriched["Daily_Long_Trend_Score"] + enriched["MA20_Trend_Score"]
    )
    enriched["Box_First_Breakout_Score"] = np.where(enriched["Box_Breakout_Pass"], 12.0, 0.0)
    enriched["Daily_Breakout_Score"] = np.where(enriched["Daily_Breakout_Pass"], 4.0, 0.0)
    enriched["Breakout_Quality_Score"] = enriched["Box_Breakout_pct"].map(bucket_breakout)
    enriched["Breakout_Group_Score"] = enriched[[
        "Box_First_Breakout_Score", "Daily_Breakout_Score", "Breakout_Quality_Score",
    ]].sum(axis=1)
    enriched["Volume_Quality_Score"] = enriched["Volume_Ratio"].map(bucket_volume)
    enriched["Daily_MACD_Score"] = np.where(enriched["Daily_MACD_Pass"], 10.0, 0.0)
    enriched["Weekly_Shadow_Score"] = np.where(enriched["Weekly_Shadow_Pass"], 4.0, 0.0)
    enriched["Weekly_Bias_Score"] = enriched["Weekly_Bias_pct"].map(bucket_bias)
    enriched["Weekly_Risk_Group_Score"] = (
        enriched["Weekly_Shadow_Score"] + enriched["Weekly_Bias_Score"]
    )
    enriched["Body_Quality_Score"] = enriched["Body_Ratio"].map(bucket_body)
    enriched["New_Score_100"] = enriched[[
        "RS_Position_Score", "Trend_Group_Score", "Breakout_Group_Score",
        "Volume_Quality_Score", "Daily_MACD_Score", "Weekly_Risk_Group_Score",
        "Body_Quality_Score",
    ]].sum(axis=1)
    enriched["V50_Candidate_Pass"] = enriched[V50_EXECUTION_GATES].all(axis=1)
    candidates = enriched[enriched["V50_Candidate_Pass"]].copy().sort_values(
        ["Signal_Date", "New_Score_100", "Original_V406_Score", "Board_RS_13W_pct", "ts_code"],
        ascending=[True, False, False, False, True], kind="mergesort",
    )
    candidates["V50_Weekly_Rank"] = candidates.groupby("Signal_Date").cumcount() + 1
    rank_map = candidates.set_index("Cycle_ID")["V50_Weekly_Rank"]
    enriched["V50_Weekly_Rank"] = enriched["Cycle_ID"].map(rank_map)
    enriched["V50_Selected"] = enriched["V50_Weekly_Rank"].eq(1)
    return enriched.sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True)


def close_on_or_before(history: pd.DataFrame, trade_date: str) -> float:
    if history.empty:
        return np.nan
    subset = history[history["trade_date"].astype(str).le(trade_date)]
    return finite_num(subset.iloc[-1]["close"]) if not subset.empty else np.nan


def open_on_date(history: pd.DataFrame, trade_date: str) -> float:
    if history.empty:
        return np.nan
    subset = history[history["trade_date"].astype(str).eq(trade_date)]
    return finite_num(subset.iloc[-1]["open"]) if not subset.empty else np.nan


def simulate_v49_portfolio(
    selected: pd.DataFrame,
    daily_histories: dict[str, pd.DataFrame],
    open_dates: list[str],
    signal_start: str,
    signal_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = selected.copy()
    for column in ("Signal_Date", "Entry_Date", "Exit_T30_Date"):
        work[column] = work[column].map(normalize_date)
    for column in (
        "Entry_Price", "Exit_T30_Price", "Exit_T30_Return_pct",
        "New_Score_100", "Original_V406_Score",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.sort_values(
        ["Entry_Date", "Signal_Date", "New_Score_100", "ts_code"],
        ascending=[True, True, False, True], kind="mergesort",
    ).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    last_exit = work["Exit_T30_Date"].max()
    days = sorted({
        day for day in open_dates if signal_start <= day <= last_exit
    } | set(work["Entry_Date"]) | set(work["Exit_T30_Date"]))
    entry_groups = {
        day: group.sort_values(
            ["New_Score_100", "Original_V406_Score", "Board_RS_13W_pct", "ts_code"],
            ascending=[False, False, False, True], kind="mergesort",
        )
        for day, group in work.groupby("Entry_Date", sort=True)
    }
    cash = INITIAL_CAPITAL
    active: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    full_reason = f"{MAX_POSITIONS}个仓位已满"

    def audit_order(row: pd.Series, action: str, reason: str, positions_before: int) -> None:
        orders.append({
            "Signal_Date": row.get("Signal_Date", ""),
            "Entry_Date": row.get("Entry_Date", ""),
            "ts_code": row.get("ts_code", ""), "name": row.get("name", ""),
            "Sample_Board": row.get("Sample_Board", ""),
            "New_Score_100": row.get("New_Score_100", np.nan),
            "Portfolio_Action": action, "Portfolio_Reason": reason,
            "Positions_Before": positions_before,
            "Prospective_Exit_Date": row.get("Exit_T30_Date", ""),
            "Prospective_Return_pct": row.get("Exit_T30_Return_pct", np.nan),
            "Prospective_Exit_Reason": row.get("Exit_T30_Reason", ""),
        })

    for trade_date in days:
        # 与既有三仓口径一致：开盘先买，旧仓在当日稍后退出，不提前释放位置。
        for _, row in entry_groups.get(trade_date, pd.DataFrame()).iterrows():
            code = str(row["ts_code"])
            before = len(active)
            if code in active:
                audit_order(row, "未买入", "同一股票已在持仓", before)
                continue
            if len(active) >= MAX_POSITIONS:
                audit_order(row, "未买入", full_reason, before)
                continue
            entry_price = finite_num(row["Entry_Price"])
            exit_price = finite_num(row["Exit_T30_Price"])
            if not math.isfinite(entry_price) or entry_price <= 0 or not math.isfinite(exit_price) or exit_price <= 0:
                audit_order(row, "未买入", "买入或退出价格无效", before)
                continue
            budget = min(POSITION_BUDGET, cash)
            shares = int(math.floor(budget / entry_price / LOT_SIZE) * LOT_SIZE)
            if shares < LOT_SIZE:
                audit_order(row, "未买入", "现金不足一手", before)
                continue
            cost = shares * entry_price
            cash -= cost
            raw_open = open_on_date(daily_histories.get(code, pd.DataFrame()), trade_date)
            scale = entry_price / raw_open if raw_open > 0 else 1.0
            trade = {
                "Signal_Date": row["Signal_Date"], "Entry_Date": trade_date,
                "ts_code": code, "name": row.get("name", ""),
                "Sample_Board": row.get("Sample_Board", ""),
                "New_Score_100": row.get("New_Score_100", np.nan),
                "Entry_Price": entry_price, "Shares": shares, "Entry_Cost": cost,
                "Planned_Exit_Date": row["Exit_T30_Date"], "Actual_Exit_Date": "",
                "Net_Exit_Price": np.nan, "Exit_Proceeds": np.nan,
                "PnL": np.nan, "Portfolio_Return_pct": np.nan,
                "Exit_Reason": row.get("Exit_T30_Reason", ""),
                "Portfolio_Status": "持仓中", "_scale": scale, "_mark": entry_price,
            }
            active[code] = trade
            trades.append(trade)
            audit_order(row, "已买入", f"买入{shares}股", before)

        exiting: list[str] = []
        for code, trade in active.items():
            if trade["Planned_Exit_Date"] != trade_date:
                continue
            match = work[
                work["Signal_Date"].eq(trade["Signal_Date"])
                & work["ts_code"].astype(str).eq(code)
            ]
            exit_price = finite_num(match.iloc[-1]["Exit_T30_Price"])
            proceeds = trade["Shares"] * exit_price
            cash += proceeds
            pnl = proceeds - trade["Entry_Cost"]
            trade.update({
                "Actual_Exit_Date": trade_date, "Net_Exit_Price": exit_price,
                "Exit_Proceeds": proceeds, "PnL": pnl,
                "Portfolio_Return_pct": pnl / trade["Entry_Cost"] * 100.0,
                "Portfolio_Status": "已平仓",
            })
            exiting.append(code)
        for code in exiting:
            active.pop(code, None)

        market_value = 0.0
        for code, trade in active.items():
            raw_close = close_on_or_before(daily_histories.get(code, pd.DataFrame()), trade_date)
            mark = raw_close * trade["_scale"] if raw_close > 0 else trade["_mark"]
            trade["_mark"] = mark
            market_value += trade["Shares"] * mark
        equity = cash + market_value
        curve_rows.append({
            "Trade_Date": trade_date, "Cash": cash, "Market_Value": market_value,
            "Equity": equity, "Positions": len(active),
            "Exposure_pct": market_value / equity * 100.0 if equity > 0 else np.nan,
            "Is_Empty": len(active) == 0,
        })

    curve = pd.DataFrame(curve_rows)
    curve["Daily_Return_pct"] = curve["Equity"].pct_change().fillna(
        curve["Equity"].iloc[0] / INITIAL_CAPITAL - 1.0
    ) * 100.0
    curve["Drawdown_pct"] = (
        curve["Equity"] / curve["Equity"].cummax().clip(lower=INITIAL_CAPITAL) - 1.0
    ) * 100.0
    ledger = pd.DataFrame(trades)
    if not ledger.empty:
        ledger = ledger.drop(columns=[column for column in ledger.columns if column.startswith("_")])
    orders_frame = pd.DataFrame(orders)
    missed = orders_frame[orders_frame["Portfolio_Action"].eq("未买入")].copy()
    closed = ledger[ledger["Portfolio_Status"].eq("已平仓")].copy()
    profits = closed.loc[closed["PnL"].gt(0), "PnL"].sort_values(ascending=False)
    losses = -closed.loc[closed["PnL"].lt(0), "PnL"].sum()
    gross_profit = profits.sum()
    net_pnl = closed["PnL"].sum()
    final_equity = float(curve.iloc[-1]["Equity"])
    summary = {
        "程序": TITLE, "信号开始": signal_start, "信号截止": signal_end,
        "初始资金": INITIAL_CAPITAL, "单仓目标资金": POSITION_BUDGET,
        "最多持仓": MAX_POSITIONS, "评分第一名信号": len(work),
        "实际买入": len(ledger), "仓位满错过": int(missed["Portfolio_Reason"].eq(full_reason).sum()) if len(missed) else 0,
        "期末权益": final_equity,
        "总收益率(%)": (final_equity / INITIAL_CAPITAL - 1.0) * 100.0,
        "年化收益率(%)": ((final_equity / INITIAL_CAPITAL) ** (252.0 / max(1, len(curve))) - 1.0) * 100.0,
        "最大回撤(%)": float(curve["Drawdown_pct"].min()),
        "交易胜率(%)": float(closed["PnL"].gt(0).mean() * 100.0) if len(closed) else np.nan,
        "止损率(%)": float(closed["Exit_Reason"].astype(str).str.contains("止损").mean() * 100.0) if len(closed) else np.nan,
        "盈利因子": gross_profit / losses if losses > 0 else np.nan,
        "平均持仓数": float(curve["Positions"].mean()),
        "平均资金暴露(%)": float(curve["Exposure_pct"].mean()),
        "空仓交易日比例(%)": float(curve["Is_Empty"].mean() * 100.0),
        "已实现净利润": net_pnl,
        "最大盈利一笔占全部正利润(%)": profits.head(1).sum() / gross_profit * 100.0 if gross_profit > 0 else np.nan,
        "前三笔占全部正利润(%)": profits.head(3).sum() / gross_profit * 100.0 if gross_profit > 0 else np.nan,
        "扣除前三笔后的净利润": net_pnl - profits.head(3).sum(),
    }
    return curve, ledger, orders_frame, missed, summary


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_result_zip(files: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, frame in files.items():
            archive.writestr(filename, csv_bytes(frame))
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# 首红柱至第一根完整绿柱：最高利润机会空间
# -----------------------------------------------------------------------------
OPPORTUNITY_TARGETS = (5, 10, 20, 30, 50, 100)


def first_red_strength_group(value: float) -> str:
    if not math.isfinite(value):
        return "无法计算"
    if value < 0.25:
        return "很短_<前绿柱峰值25%"
    if value < 0.50:
        return "偏短_25%至50%"
    if value < 1.00:
        return "正常_50%至100%"
    return "较强_不低于前绿柱峰值"


def build_cycle_opportunities(
    events: pd.DataFrame,
    daily_histories: dict[str, pd.DataFrame],
    observation_end: str,
    sell_slippage_pct: float,
) -> pd.DataFrame:
    """计算首红柱买入后，到第一根完整绿柱确认日为止的事后最高利润。"""
    first = events[events["Event_Type"].astype(str).eq("第一根红柱")].copy()
    first = first.drop_duplicates("Cycle_ID", keep="first")
    rows: list[dict[str, Any]] = []
    keep_columns = [
        "Cycle_ID", "ts_code", "name", "Sample_Board", "SW_L1", "SW_L2", "SW_L3",
        "Signal_Date", "Weekly_Trend", "Zero_Axis", "Hist", "Hist_Prev", "DIF", "DEA",
        "Weekly_Close", "W_MA20", "W_MA40", "W_MA20_Slope4_pct",
        "Pre_13W_Return_pct", "Pre_26W_Return_pct", "Pullback_Depth_pct",
        "Raw_Close", "Circ_MV_Billion", "Turnover_Rate", "Tradable", "Untradable_Reason",
        "Entry_Date", "Entry_Price", "Cycle_Completed", "Cycle_Censored", "Cycle_Type",
        "Cycle_Strength_Class", "Red_Cycle_Weeks", "Last_Red_Date", "First_Green_Date",
        "Red_Hist_Sequence", "Peak_Red_Hist", "Peak_Red_Week", "Red_Hist_Area",
        "PreGreen_Abs_Peak_Hist", "Peak_Red_to_PreGreen_Peak_Ratio",
        "First_Material_Shrink_Week", "Material_Shrink_Count", "ReExpansion_Count",
        "CP_W2_Date", "CP_W2_State", "CP_W2_Hist", "CP_W2_Hist_vs_W1_pct",
        "CP_W2_Weak_Candidate",
    ]

    for _, event in first.iterrows():
        base = {column: event.get(column, np.nan) for column in keep_columns}
        code = str(event.get("ts_code", ""))
        entry_date = normalize_date(event.get("Entry_Date"))
        entry_price = finite_num(event.get("Entry_Price"))
        completed = to_bool(event.get("Cycle_Completed"))
        first_green = normalize_date(event.get("First_Green_Date"))
        last_red = normalize_date(event.get("Last_Red_Date"))
        end_date = first_green if completed and first_green else observation_end
        daily = daily_histories.get(code, pd.DataFrame())

        pre_green_peak = abs(finite_num(event.get("PreGreen_Abs_Peak_Hist")))
        first_hist = finite_num(event.get("Hist"))
        first_strength_ratio = (
            first_hist / pre_green_peak
            if math.isfinite(first_hist) and math.isfinite(pre_green_peak) and pre_green_peak > 0
            else np.nan
        )
        result: dict[str, Any] = {
            **base,
            "Opportunity_Status": "完整周期" if completed else "截至观察日仍未翻绿",
            "Opportunity_Valid": False,
            "Observation_End_Date": end_date,
            "First_Red_to_PreGreen_Peak_Ratio": first_strength_ratio,
            "First_Red_Strength_Group": first_red_strength_group(first_strength_ratio),
        }
        for target in OPPORTUNITY_TARGETS:
            result[f"Reached_{target}_pct"] = False
            result[f"First_{target}_Date"] = ""
            result[f"Trading_Days_To_{target}"] = np.nan

        if not to_bool(event.get("Tradable")) or not entry_date or not math.isfinite(entry_price):
            result["Opportunity_Invalid_Reason"] = str(
                event.get("Untradable_Reason", "无法按次日开盘买入")
            )
            rows.append(result)
            continue
        if daily.empty:
            result["Opportunity_Invalid_Reason"] = "个股日线不存在"
            rows.append(result)
            continue

        path = daily[
            daily["trade_date"].astype(str).ge(entry_date)
            & daily["trade_date"].astype(str).le(end_date)
        ].copy().sort_values("trade_date").reset_index(drop=True)
        for column in ("high", "low", "close"):
            path[column] = pd.to_numeric(path[column], errors="coerce")
        path = path.dropna(subset=["high", "low", "close"])
        if path.empty:
            result["Opportunity_Invalid_Reason"] = "买入日至观察结束日无行情"
            rows.append(result)
            continue

        peak_position = int(path["high"].idxmax())
        trough_position = int(path["low"].idxmin())
        peak_price = float(path.loc[peak_position, "high"])
        trough_price = float(path.loc[trough_position, "low"])
        peak_date = str(path.loc[peak_position, "trade_date"])
        trough_date = str(path.loc[trough_position, "trade_date"])
        end_close = float(path.iloc[-1]["close"])
        peak_net_price = peak_price * (1.0 - sell_slippage_pct / 100.0)
        result.update({
            "Opportunity_Valid": True,
            "Opportunity_Invalid_Reason": "",
            "Observation_Trading_Days": int(len(path)),
            "Peak_Date": peak_date,
            "Peak_High": peak_price,
            "Peak_MFE_pct": (peak_price / entry_price - 1.0) * 100.0,
            "Peak_MFE_After_Sell_Slippage_pct": (peak_net_price / entry_price - 1.0) * 100.0,
            "Trading_Days_To_Peak": int(peak_position + 1),
            "Calendar_Days_To_Peak": int(
                (pd.to_datetime(peak_date) - pd.to_datetime(entry_date)).days
            ),
            "Peak_During_Green_Confirmation_Week": bool(
                completed and bool(last_red) and peak_date > last_red
            ),
            "Trough_Date": trough_date,
            "Trough_Low": trough_price,
            "Path_MAE_pct": (trough_price / entry_price - 1.0) * 100.0,
            "Observation_End_Close": end_close,
            "End_Close_Return_pct": (end_close / entry_price - 1.0) * 100.0,
            "Peak_to_End_Close_Giveback_pct_points": (
                (peak_price / entry_price - 1.0) - (end_close / entry_price - 1.0)
            ) * 100.0,
        })
        for target in OPPORTUNITY_TARGETS:
            hit = path[path["high"].ge(entry_price * (1.0 + target / 100.0))]
            if not hit.empty:
                first_position = int(hit.index[0])
                result[f"Reached_{target}_pct"] = True
                result[f"First_{target}_Date"] = str(path.loc[first_position, "trade_date"])
                result[f"Trading_Days_To_{target}"] = int(first_position + 1)
        rows.append(result)

    return pd.DataFrame(rows).sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True)


def opportunity_summary(frame: pd.DataFrame, group_columns: list[str] | None = None) -> pd.DataFrame:
    """只汇总完整且可计算的周期；未翻绿周期另表保留。"""
    valid = frame[
        frame["Opportunity_Valid"].map(to_bool)
        & frame["Cycle_Completed"].map(to_bool)
    ].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["Peak_MFE_pct"] = pd.to_numeric(valid["Peak_MFE_pct"], errors="coerce")
    valid["Path_MAE_pct"] = pd.to_numeric(valid["Path_MAE_pct"], errors="coerce")
    valid["Trading_Days_To_Peak"] = pd.to_numeric(
        valid["Trading_Days_To_Peak"], errors="coerce"
    )
    valid["Red_Cycle_Weeks"] = pd.to_numeric(valid["Red_Cycle_Weeks"], errors="coerce")

    if group_columns:
        grouped = valid.groupby(group_columns, dropna=False, sort=False)
    else:
        grouped = [((), valid)]
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        mfe = group["Peak_MFE_pct"].dropna()
        row = {
            column: key for column, key in zip(group_columns or [], keys)
        }
        row.update({
            "完整周期数": int(len(group)),
            "涉及股票数": int(group["ts_code"].nunique()),
            "最高利润均值(%)": mfe.mean(),
            "最高利润中位数(%)": mfe.median(),
            "最高利润P25(%)": mfe.quantile(0.25),
            "最高利润P75(%)": mfe.quantile(0.75),
            "最高利润P90(%)": mfe.quantile(0.90),
            "最高利润最大值(%)": mfe.max(),
            "最大浮亏中位数(%)": group["Path_MAE_pct"].median(),
            "到最高价交易日中位数": group["Trading_Days_To_Peak"].median(),
            "红柱持续周数中位数": group["Red_Cycle_Weeks"].median(),
            "峰值发生在翻绿确认周(%)": pct_mean(
                group["Peak_During_Green_Confirmation_Week"].map(to_bool)
            ),
        })
        for target in OPPORTUNITY_TARGETS:
            row[f"曾达到{target}%(%)"] = pct_mean(
                group[f"Reached_{target}_pct"].map(to_bool)
            )
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Streamlit 页面
# -----------------------------------------------------------------------------
def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "无需上传任何前置结果：选择任意完整历史区间，程序自行扫描当时的全量科技股票池、"
        "对所有可买的第一根真正周线红柱透明评分，每周选择第一名，并以30万元、最多三仓完成组合回测。"
    )

    with st.sidebar:
        st.header("研究区间")
        SIGNAL_START_DATE = st.date_input("回测开始日期", value=date(2022, 6, 5))
        SIGNAL_END_DATE = st.date_input("回测截止日期", value=date(2023, 6, 5))
        MARKET_END_DATE = SIGNAL_END_DATE + timedelta(days=70)
        st.caption(
            f"行情自动观察至{MARKET_END_DATE.isoformat()}，只用于完成截止日前信号的八周退出，"
            "不会产生区间外新信号。"
        )

        st.header("历史科技股票池口径（固定）")
        st.info("不抽样：主板、创业板、科创板历史科技股票全部检查。")
        MIN_PRICE = 10.0
        MIN_MV = 100.0
        MAX_MV = 1_000_000_000.0
        SAMPLE_PER_BOARD = 0
        SAMPLE_SEED = DEFAULT_SAMPLE_SEED
        st.write("信号日原始股价：≥10元")
        st.write("信号日流通市值：≥100亿元（不设上限）")

        st.header("固定交易与评分规则")
        PRICE_TOLERANCE = 3.0
        STOP_THRESHOLD = 10.0
        BUY_SLIPPAGE = 0.20
        SELL_SLIPPAGE = 0.20
        LONG_CYCLE_MIN_WEEKS = DEFAULT_LONG_CYCLE_MIN_WEEKS
        MATERIAL_HIST_CHANGE = DEFAULT_MATERIAL_HIST_CHANGE_PCT
        SHORT_STRENGTH_RATIO = DEFAULT_SHORT_STRENGTH_RATIO
        st.write("第一根真正周线红柱产生候选；V40.6日线形态只评分、不再一票否决")
        st.write("每周评分第一名；30万元；最多3只；单只目标10万元")
        st.write("-10%止损；+30%止盈；最长8周；买卖各0.2%滑点")

        st.header("数据与缓存")
        USE_CACHE = st.checkbox("使用逐股票缓存", value=True)
        API_PAUSE = st.number_input(
            "每次API调用后暂停(秒)", min_value=0.0, max_value=3.0,
            value=0.12, step=0.05,
        )
        if st.button("清除本程序缓存"):
            if os.path.isdir(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            st.success("验证器专用缓存已清除")

    TS_TOKEN = st.text_input("Tushare Token", type="password")
    if not TS_TOKEN:
        st.info("请输入Tushare Token。程序会检查完整历史科技股票池，不限制股票数量。")
        return

    run_requested = st.button("开始一体化选股与三仓回测", type="primary")
    if not run_requested:
        if "v50_result_zip" in st.session_state:
            st.success("上一次回测结果仍然保留，可直接下载，无需重新运行。")
            st.download_button(
                "下载1号：上一次全部结果ZIP",
                data=st.session_state["v50_result_zip"],
                file_name="weekly_macd_rank_first_v5_0_all_results.zip",
                mime="application/zip",
                type="primary",
                on_click="ignore",
            )
            return
        with st.expander("本程序的固定交易口径"):
            st.markdown(
                """
                - **第一根红柱**：完整周MACD柱本周 `>0`，上周 `<=0`。
                - **实际买点**：信号周结束后的下一市场交易日开盘。
                - **退出模拟**：+30%止盈、-10%止损、最长40个交易日；同日双触发保守按止损。
                - **候选门槛**：真正第一根周线红柱、T+1非一字板、T+1开盘跳空在-3%至+5%。
                - **只评分不否决**：日线趋势、箱体突破、MA20、成交量、日线MACD、周线乖离、上影线及实体质量。
                - **明确取消**：波浪2—5次完全不参与选股；实体阳线≥60%不再作为硬门槛。
                - **组合**：每周最多产生一个第一名；30万元；最多3只；每只目标10万元；100股整数倍。
                """
            )
        return

    API_ERRORS = []
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    try:
        # 防止SDK内部网络请求无限等待；不存在该属性时不影响运行。
        if hasattr(pro, "_DataApi__http_url"):
            pass
    except Exception:
        pass

    date_error = validate_research_dates(
        SIGNAL_START_DATE, SIGNAL_END_DATE, MARKET_END_DATE
    )
    if date_error:
        st.error(date_error)
        return
    if MARKET_END_DATE > date.today():
        st.error(
            "回测截止日距离今天不足70天，后续八周尚未完整发生。"
            "请选择更早的截止日期；实时选股模式将在历史回测验证后单独加入。"
        )
        return
    signal_start = SIGNAL_START_DATE.strftime("%Y%m%d")
    signal_end = SIGNAL_END_DATE.strftime("%Y%m%d")
    market_end = MARKET_END_DATE.strftime("%Y%m%d")
    preload_start = (SIGNAL_START_DATE - timedelta(days=3 * 365)).strftime("%Y%m%d")

    config = {
        "signal_start": signal_start,
        "signal_end": signal_end,
        "market_end": market_end,
        "preload_start": preload_start,
        "min_price": float(MIN_PRICE),
        "min_mv": float(MIN_MV),
        "max_mv": float(MAX_MV),
        "price_tolerance_pct": float(PRICE_TOLERANCE),
        "stop_threshold_pct": float(STOP_THRESHOLD),
        "buy_slippage_pct": float(BUY_SLIPPAGE),
        "sell_slippage_pct": float(SELL_SLIPPAGE),
        "sample_per_board": int(SAMPLE_PER_BOARD),
        "sample_seed": int(SAMPLE_SEED),
        "long_cycle_min_weeks": int(LONG_CYCLE_MIN_WEEKS),
        "material_hist_change_pct": float(MATERIAL_HIST_CHANGE),
        "short_strength_ratio": float(SHORT_STRENGTH_RATIO),
    }

    try:
        with st.spinner("正在加载交易日历、历史股票池和申万历史成分..."):
            open_dates = load_trade_calendar(preload_start, market_end)
            stock_basic = load_stock_basic()
            memberships = load_sw_tech_memberships(float(API_PAUSE))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return

    week_last_map = complete_week_last_dates(open_dates)
    board_weeklies: dict[str, pd.DataFrame] = {}
    try:
        with st.spinner("正在加载主板、创业板、科创板历史基准..."):
            for board_code in sorted(set(BOARD_INDEX.values())):
                index_daily = fetch_index_history(
                    board_code, preload_start, market_end,
                    bool(USE_CACHE), float(API_PAUSE),
                )
                if index_daily.empty:
                    raise RuntimeError(f"基准指数{board_code}行情为空")
                board_weeklies[board_code] = build_weekly(index_daily, week_last_map)
    except Exception as exc:
        st.error(f"板块基准加载失败：{exc}")
        return

    period_index = build_period_index(memberships)
    universe_codes = sorted(set(period_index) & set(stock_basic["ts_code"].astype(str)))
    universe_stocks = stock_basic[stock_basic["ts_code"].isin(universe_codes)].copy()
    universe_stocks = universe_stocks.sort_values("ts_code").reset_index(drop=True)
    stocks, sample_audit, population_summary = build_stratified_sample(
        stocks=universe_stocks,
        period_index=period_index,
        reference_date=signal_end,
        per_board=int(SAMPLE_PER_BOARD),
        seed=int(SAMPLE_SEED),
    )
    if stocks.empty:
        st.error("历史科技股票池为空")
        return

    # 当前股票基础表包含回测结束后才上市的公司。它们不可能在信号区间产生事件，
    # 不应发起行情请求，更不能被误报为“数据获取失败”。
    list_dates = stocks["list_date"].apply(lambda value: normalize_date(value, "19000101"))
    delist_dates = stocks["delist_date"].apply(lambda value: normalize_date(value, "99991231"))
    listed_after_signal = list_dates.gt(signal_end)
    listed_after_market = list_dates.gt(market_end)
    no_history_overlap = delist_dates.lt(preload_start)
    post_signal_listings = int(listed_after_signal.sum())
    post_market_listings = int(listed_after_market.sum())
    no_overlap_stocks = int(no_history_overlap.sum())
    stocks_to_fetch = stocks[~listed_after_signal & ~no_history_overlap].copy().reset_index(drop=True)

    sample_hash = cache_key(
        int(SAMPLE_SEED), int(SAMPLE_PER_BOARD),
        "|".join(sample_audit["ts_code"].astype(str)),
    )
    sample_path = os.path.join(OUTPUT_DIR, f"weekly_macd_sample_{sample_hash}.csv")
    population_path = os.path.join(OUTPUT_DIR, f"weekly_macd_population_{sample_hash}.csv")
    # 在耗时行情循环之前保存，程序中断后仍能核对和复用本次样本。
    atomic_csv(sample_audit, sample_path)
    atomic_csv(population_summary, population_path)

    open_pos = {trade_date: position for position, trade_date in enumerate(open_dates)}
    st.write(
        f"完整历史科技池：{len(universe_stocks)}只；本次全量检查：{len(stocks)}只；"
        f"实际读取历史行情：{len(stocks_to_fetch)}只；"
        f"信号区间：{signal_start}—{signal_end}；"
        f"行情结果观察至：{market_end}；行情预热起点：{preload_start}。"
    )
    if post_signal_listings:
        st.caption(
            f"有{post_signal_listings}只股票在信号截止日后才上市，已按时间事实跳过；"
            f"其中{post_market_listings}只在行情观察截止日后才上市，不再计为行情失败。"
        )
    st.dataframe(style_percent_table(population_summary), use_container_width=True, hide_index=True)

    all_records: list[dict[str, Any]] = []
    daily_histories: dict[str, pd.DataFrame] = {}
    weekly_histories: dict[str, pd.DataFrame] = {}
    reject_totals: dict[str, int] = {}
    cache_hits = 0
    data_failures = 0
    progress = st.progress(0.0, text="正在逐股票验证周线状态...")
    status = st.empty()

    for idx, stock in stocks_to_fetch.iterrows():
        ts_code = str(stock["ts_code"])
        progress.progress(
            (idx + 1) / len(stocks_to_fetch),
            text=f"{idx + 1}/{len(stocks_to_fetch)} {ts_code}",
        )
        status.caption(
            f"已产生事件 {len(all_records)} 条；缓存命中 {cache_hits}；行情失败 {data_failures}"
        )
        daily, basic, cache_hit = fetch_stock_history(
            ts_code, preload_start, market_end, bool(USE_CACHE), float(API_PAUSE)
        )
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        records, rejects, weekly = analyze_stock(
            stock=stock,
            periods=period_index.get(ts_code, []),
            daily=daily,
            basic=basic,
            week_last_map=week_last_map,
            open_dates=open_dates,
            open_pos=open_pos,
            config=config,
        )
        all_records.extend(records)
        if records:
            daily_histories[ts_code] = add_daily_v406_indicators(daily)
            weekly_histories[ts_code] = weekly
        for reason, count in rejects.items():
            reject_totals[reason] = reject_totals.get(reason, 0) + count

    progress.empty()
    status.empty()
    if not all_records:
        st.error("没有生成有效事件。请检查日期、价格市值范围、权限和API错误。")
        if API_ERRORS:
            st.code("\n".join(API_ERRORS[:50]))
        return

    events = pd.DataFrame(all_records).sort_values(
        ["Signal_Date", "ts_code", "Event_Type"]
    ).reset_index(drop=True)
    with st.spinner("正在对全部可买第一根红柱计算透明评分和每周排名..."):
        enriched = enrich_first_red_v49(
            events, daily_histories, weekly_histories, board_weeklies
        )
    if enriched.empty:
        st.error("没有形成可评分的完整第一根红柱事件。")
        return
    candidates = enriched[enriched["V50_Candidate_Pass"]].copy().sort_values(
        ["Signal_Date", "V50_Weekly_Rank", "ts_code"]
    )
    top3 = candidates[candidates["V50_Weekly_Rank"].le(3)].copy().sort_values(
        ["Signal_Date", "V50_Weekly_Rank", "ts_code"]
    )
    selected = enriched[enriched["V50_Selected"]].copy().sort_values("Signal_Date")
    if selected.empty:
        st.error("所选区间没有形成可执行的评分第一名，组合保持空仓。")
        return

    with st.spinner("正在执行30万元、最多三仓的真实资金占用模拟..."):
        curve, ledger, orders, missed, portfolio_summary = simulate_v49_portfolio(
            selected, daily_histories, open_dates, signal_start, signal_end
        )

    accepted = ledger.copy()
    accepted["Entry_dt"] = pd.to_datetime(accepted["Entry_Date"], format="%Y%m%d", errors="coerce")
    accepted["Exit_dt"] = pd.to_datetime(accepted["Actual_Exit_Date"], format="%Y%m%d", errors="coerce")
    weekly_rows: list[dict[str, Any]] = []
    for week_end in pd.date_range(SIGNAL_START_DATE, SIGNAL_END_DATE, freq="W-FRI"):
        week_start = week_end - pd.Timedelta(days=6)
        held = accepted[
            accepted["Entry_dt"].le(week_end) & accepted["Exit_dt"].ge(week_start)
        ]
        weekly_rows.append({
            "Week_End": week_end.strftime("%Y%m%d"), "Covered": not held.empty,
            "Trades_Touching_Week": len(held),
            "Stocks": "|".join(held["name"].astype(str)),
        })
    weekly_coverage = pd.DataFrame(weekly_rows)
    covered_weeks = int(weekly_coverage["Covered"].sum())
    portfolio_summary.update({
        "区间周数": len(weekly_coverage), "实际持仓覆盖周": covered_weeks,
        "完全空仓周": len(weekly_coverage) - covered_weeks,
        "持仓周覆盖率(%)": covered_weeks / len(weekly_coverage) * 100.0,
        "全量第一根红柱": len(enriched), "可评分候选": len(candidates),
        "信号截止日后上市股票": post_signal_listings,
        "其中行情观察截止后上市": post_market_listings,
        "历史无重叠股票": no_overlap_stocks,
        "真实无行情或接口失败股票": data_failures, "缓存命中股票": cache_hits,
    })
    summary_frame = pd.DataFrame([portfolio_summary])

    def stage_row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
        returns = pd.to_numeric(frame["Exit_T30_Return_pct"], errors="coerce")
        return {
            "阶段": name, "事件数": len(frame),
            "有信号周": frame["Signal_Date"].nunique(),
            "平均收益(%)": returns.mean(), "中位收益(%)": returns.median(),
            "胜率(%)": returns.gt(0).mean() * 100.0,
            "止损率(%)": frame["Exit_T30_Reason"].astype(str).str.contains("止损").mean() * 100.0,
        }

    rank2 = candidates[candidates["V50_Weekly_Rank"].eq(2)].copy()
    rank3 = candidates[candidates["V50_Weekly_Rank"].eq(3)].copy()
    candidate_summary = pd.DataFrame([
        stage_row("全部完整第一根红柱", enriched),
        stage_row("通过T+1执行限制的评分池", candidates),
        stage_row("每周评分第一名", selected),
        stage_row("每周评分第二名", rank2),
        stage_row("每周评分第三名", rank3),
        stage_row("每周评分前三名合计", top3),
    ])
    score_definition = pd.DataFrame([
        {"评分组": "板块相对强度", "最高分": 20, "规则": "信号周相对所属板块指数的13周强度位置"},
        {"评分组": "日线趋势", "最高分": 15, "规则": "MA60>MA120得10分；MA20不下降得5分"},
        {"评分组": "突破质量", "最高分": 20, "规则": "十日箱体首次突破12分；站上MA20 2%得4分；突破幅度质量0—4分"},
        {"评分组": "成交量质量", "最高分": 15, "规则": "量比1.0—4.0分档；1.6—2.2倍最高，不符合不剔除"},
        {"评分组": "日线MACD", "最高分": 10, "规则": "DIF>0且MACD柱增长得10分"},
        {"评分组": "周线风险", "最高分": 10, "规则": "前周上影线合格4分；周线乖离分档0—6分"},
        {"评分组": "实体质量", "最高分": 10, "规则": "实体比例分档0—10分；不足60%仍可参加排序"},
    ])
    trade_year = ledger.assign(
        Entry_Year=ledger["Entry_Date"].astype(str).str[:4]
    ).groupby("Entry_Year", as_index=False).agg(
        交易数=("ts_code", "size"),
        平均单笔收益=("Portfolio_Return_pct", "mean"),
        中位单笔收益=("Portfolio_Return_pct", "median"),
        已实现利润=("PnL", "sum"),
    )
    reject_frame = pd.DataFrame(
        [{"剔除原因": reason, "次数": count} for reason, count in reject_totals.items()]
    ).sort_values("次数", ascending=False) if reject_totals else pd.DataFrame(
        columns=["剔除原因", "次数"]
    )
    metadata = pd.DataFrame([
        {"项目": "程序", "值": TITLE}, {"项目": "生成时间", "值": datetime.now().isoformat(timespec="seconds")},
        {"项目": "信号区间", "值": f"{signal_start}—{signal_end}"},
        {"项目": "行情观察截止", "值": market_end},
        {"项目": "股票池", "值": "历史申万科技成分；主板/创业板/科创板；信号日股价≥10元、流通市值≥100亿元"},
        {"项目": "候选规则", "值": "第一根真正周线红柱；T+1非一字板且跳空-3%至+5%；其他V40.6形态全部改为评分"},
        {"项目": "评分规则", "值": "固定透明100分；同时输出每周第一、第二、第三名的未来表现"},
        {"项目": "组合", "值": "30万元；最多3只；单只目标10万元；100股整数倍"},
        {"项目": "退出", "值": "+30%止盈；-10%止损；最长40个交易日；同日双触发按止损"},
        {"项目": "摩擦", "值": "买入和卖出各0.2%滑点；未另计佣金与印花税"},
    ])
    files = {
        "01_portfolio_summary_v5_0.csv": summary_frame,
        "02_portfolio_curve_v5_0.csv": curve,
        "03_portfolio_ledger_v5_0.csv": ledger,
        "04_portfolio_orders_v5_0.csv": orders,
        "05_missed_signals_v5_0.csv": missed,
        "06_selected_top1_v5_0.csv": selected,
        "07_weekly_top3_v5_0.csv": top3,
        "08_scored_candidate_pool_v5_0.csv": candidates,
        "09_first_red_audit_v5_0.csv": enriched,
        "10_weekly_coverage_v5_0.csv": weekly_coverage,
        "11_trade_year_v5_0.csv": trade_year,
        "12_rank_summary_v5_0.csv": candidate_summary,
        "13_score_definition_v5_0.csv": score_definition,
        "14_full_tech_universe_v5_0.csv": sample_audit,
        "15_population_v5_0.csv": population_summary,
        "16_rejection_audit_v5_0.csv": reject_frame,
        "17_metadata_v5_0.csv": metadata,
    }
    result_zip_bytes = make_result_zip(files)
    st.session_state["v50_result_zip"] = result_zip_bytes

    st.success(
        f"回测完成：完整第一根红柱{len(enriched)}个，可评分候选{len(candidates)}个，"
        f"评分第一名{len(selected)}个，实际买入{len(ledger)}个。"
    )
    metrics1 = st.columns(5)
    metrics1[0].metric("评分第一名", f"{len(selected)}")
    metrics1[1].metric("实际买入", f"{len(ledger)}")
    metrics1[2].metric("期末权益", f"¥{portfolio_summary['期末权益']:,.0f}")
    metrics1[3].metric("总收益", f"{portfolio_summary['总收益率(%)']:.2f}%")
    metrics1[4].metric("最大回撤", f"{portfolio_summary['最大回撤(%)']:.2f}%")
    metrics2 = st.columns(5)
    metrics2[0].metric("胜率", f"{portfolio_summary['交易胜率(%)']:.2f}%")
    metrics2[1].metric("止损率", f"{portfolio_summary['止损率(%)']:.2f}%")
    metrics2[2].metric("持仓覆盖周", f"{covered_weeks}/{len(weekly_coverage)}")
    metrics2[3].metric("完全空仓周", f"{len(weekly_coverage)-covered_weeks}")
    metrics2[4].metric("仓位满错过", f"{portfolio_summary['仓位满错过']}")

    st.subheader("组合资金曲线")
    chart = curve[["Trade_Date", "Equity"]].copy()
    chart["Trade_Date"] = pd.to_datetime(chart["Trade_Date"], format="%Y%m%d")
    st.line_chart(chart.set_index("Trade_Date"))
    st.subheader("组合总表与候选质量")
    st.dataframe(summary_frame, use_container_width=True, hide_index=True)
    st.dataframe(candidate_summary, use_container_width=True, hide_index=True)
    with st.expander("实际成交、错过信号和每周覆盖"):
        st.dataframe(ledger, use_container_width=True, hide_index=True)
        st.dataframe(missed, use_container_width=True, hide_index=True)
        st.dataframe(weekly_coverage, use_container_width=True, hide_index=True)

    st.subheader("下载结果")
    st.download_button(
        "下载1号：全部结果ZIP", result_zip_bytes,
        file_name="weekly_macd_rank_first_v5_0_all_results.zip",
        mime="application/zip", type="primary", on_click="ignore",
    )
    labels = [
        "2号：组合总表", "3号：资金曲线", "4号：成交账本", "5号：下单审计",
        "6号：错过信号", "7号：评分第一名", "8号：每周前三名", "9号：全部评分池",
        "10号：第一红柱审计", "11号：持仓周覆盖", "12号：分年交易", "13号：排名摘要",
        "14号：评分说明", "15号：科技股票池", "16号：板块统计", "17号：剔除审计",
        "18号：运行信息",
    ]
    columns = st.columns(4)
    for index, (filename, frame) in enumerate(files.items()):
        with columns[index % 4]:
            st.download_button(
                labels[index], csv_bytes(frame), file_name=filename,
                mime="text/csv", key=f"v50_{filename}", on_click="ignore",
            )
    st.warning(
        "这是历史回测，不是未来保证。为了保持跨年度可比性，请只修改起止日期，不调整评分权重。"
    )


def opportunity_main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "只验证首红柱周期内曾经出现的最高利润空间：不设八周、不止损、不止盈、"
        "不评分、不做资金组合。最高价是事后机会空间上限，不是卖出信号。"
    )

    with st.sidebar:
        st.header("信号与观察区间")
        SIGNAL_START_DATE = st.date_input(
            "首红柱信号开始日期", value=date(2022, 6, 5), key="opp_signal_start"
        )
        SIGNAL_END_DATE = st.date_input(
            "首红柱信号截止日期", value=date(2023, 6, 5), key="opp_signal_end"
        )
        suggested_observation_end = min(
            date.today(), SIGNAL_END_DATE + timedelta(days=550)
        )
        OBSERVATION_END_DATE = st.date_input(
            "周期观察截止日期",
            value=suggested_observation_end,
            max_value=date.today(),
            key="opp_observation_end",
        )
        st.caption(
            "观察截止日不是持有上限，只是可用行情的边界。到该日仍未翻绿的周期会单列为截断样本。"
        )

        st.header("固定股票池")
        st.info("不抽样：历史科技板块中的主板、创业板、科创板全部检查。")
        MIN_PRICE = 10.0
        MIN_MV = 100.0
        MAX_MV = 1_000_000_000.0
        SAMPLE_PER_BOARD = 0
        SAMPLE_SEED = DEFAULT_SAMPLE_SEED
        st.write("信号日原始股价≥10元")
        st.write("信号日流通市值≥100亿元，不设上限")

        st.header("唯一买入口径")
        PRICE_TOLERANCE = 3.0
        BUY_SLIPPAGE = 0.20
        SELL_SLIPPAGE = 0.20
        STOP_THRESHOLD = 10.0  # 仅供底层兼容字段计算，不参与本验证
        LONG_CYCLE_MIN_WEEKS = DEFAULT_LONG_CYCLE_MIN_WEEKS
        MATERIAL_HIST_CHANGE = DEFAULT_MATERIAL_HIST_CHANGE_PCT
        SHORT_STRENGTH_RATIO = DEFAULT_SHORT_STRENGTH_RATIO
        st.write("完整周MACD：上周柱≤0，本周柱>0")
        st.write("信号周结束后的下一交易日开盘买入，计0.2%买入滑点")
        st.write("统计至第一根完整绿柱确认日，不模拟卖出")

        st.header("数据与缓存")
        USE_CACHE = st.checkbox("使用逐股票缓存", value=True, key="opp_use_cache")
        API_PAUSE = st.number_input(
            "每次API调用后暂停(秒)", min_value=0.0, max_value=3.0,
            value=0.12, step=0.05, key="opp_api_pause",
        )
        if st.button("清除本程序缓存", key="opp_clear_cache"):
            if os.path.isdir(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            st.success("验证器专用缓存已清除")

    TS_TOKEN = st.text_input("Tushare Token", type="password", key="opp_token")
    if not TS_TOKEN:
        st.info("请输入Tushare Token。")
        return

    run_requested = st.button("开始验证首红柱周期最高利润", type="primary")
    if not run_requested:
        if "cycle_opportunity_v1_zip" in st.session_state:
            st.success("上一次验证结果仍然保留，可直接下载。")
            st.download_button(
                "下载1号：上一次全部结果ZIP",
                data=st.session_state["cycle_opportunity_v1_zip"],
                file_name="weekly_macd_cycle_opportunity_v1_0_all_results.zip",
                mime="application/zip", type="primary", on_click="ignore",
            )
            return
        with st.expander("本验证器回答什么问题"):
            st.markdown(
                """
                - 信号周结束后，以下一交易日开盘价作为买入成本。
                - 从买入日开始，观察到第一根完整绿柱确认日（含确认周）。
                - 只统计这段期间的最高价与买入价差距，不假设能够卖在最高价。
                - 完整周期用于核心统计；到观察截止日仍为红柱的周期单独列出。
                - 均值、中位数、分位数以及达到5%/10%/20%/30%/50%/100%的比例同时输出。
                """
            )
        return

    if SIGNAL_START_DATE >= SIGNAL_END_DATE:
        st.error("信号开始日期必须早于信号截止日期。")
        return
    if OBSERVATION_END_DATE <= SIGNAL_END_DATE:
        st.error("周期观察截止日期必须晚于信号截止日期。")
        return
    if OBSERVATION_END_DATE > date.today():
        st.error("周期观察截止日期不能晚于今天。")
        return

    API_ERRORS = []
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    signal_start = SIGNAL_START_DATE.strftime("%Y%m%d")
    signal_end = SIGNAL_END_DATE.strftime("%Y%m%d")
    observation_end = OBSERVATION_END_DATE.strftime("%Y%m%d")
    preload_start = (SIGNAL_START_DATE - timedelta(days=3 * 365)).strftime("%Y%m%d")
    calendar_tail = (OBSERVATION_END_DATE + timedelta(days=7)).strftime("%Y%m%d")
    config = {
        "signal_start": signal_start,
        "signal_end": signal_end,
        "market_end": observation_end,
        "preload_start": preload_start,
        "min_price": float(MIN_PRICE), "min_mv": float(MIN_MV), "max_mv": float(MAX_MV),
        "price_tolerance_pct": float(PRICE_TOLERANCE),
        "stop_threshold_pct": float(STOP_THRESHOLD),
        "buy_slippage_pct": float(BUY_SLIPPAGE),
        "sell_slippage_pct": float(SELL_SLIPPAGE),
        "sample_per_board": int(SAMPLE_PER_BOARD), "sample_seed": int(SAMPLE_SEED),
        "long_cycle_min_weeks": int(LONG_CYCLE_MIN_WEEKS),
        "material_hist_change_pct": float(MATERIAL_HIST_CHANGE),
        "short_strength_ratio": float(SHORT_STRENGTH_RATIO),
    }

    try:
        with st.spinner("正在加载交易日历、历史科技股票池和申万历史成分..."):
            open_dates = load_trade_calendar(preload_start, observation_end)
            full_calendar = load_trade_calendar(preload_start, calendar_tail)
            stock_basic = load_stock_basic()
            memberships = load_sw_tech_memberships(float(API_PAUSE))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return

    # 用多取7天的交易日历识别完整周；观察截止日落在周中时，该临时周不会被误当成完整周。
    week_last_map = complete_week_last_dates(full_calendar)
    period_index = build_period_index(memberships)
    universe_codes = sorted(set(period_index) & set(stock_basic["ts_code"].astype(str)))
    universe_stocks = stock_basic[stock_basic["ts_code"].isin(universe_codes)].copy()
    universe_stocks = universe_stocks.sort_values("ts_code").reset_index(drop=True)
    stocks, sample_audit, population_summary = build_stratified_sample(
        stocks=universe_stocks, period_index=period_index, reference_date=signal_end,
        per_board=int(SAMPLE_PER_BOARD), seed=int(SAMPLE_SEED),
    )
    if stocks.empty:
        st.error("历史科技股票池为空。")
        return

    list_dates = stocks["list_date"].apply(lambda value: normalize_date(value, "19000101"))
    delist_dates = stocks["delist_date"].apply(lambda value: normalize_date(value, "99991231"))
    listed_after_signal = list_dates.gt(signal_end)
    listed_after_observation = list_dates.gt(observation_end)
    no_history_overlap = delist_dates.lt(preload_start)
    post_signal_listings = int(listed_after_signal.sum())
    post_observation_listings = int(listed_after_observation.sum())
    no_overlap_stocks = int(no_history_overlap.sum())
    stocks_to_fetch = stocks[~listed_after_signal & ~no_history_overlap].copy().reset_index(drop=True)

    st.write(
        f"完整历史科技池{len(universe_stocks)}只；实际读取{len(stocks_to_fetch)}只；"
        f"首红柱信号区间{signal_start}—{signal_end}；周期观察至{observation_end}。"
    )
    st.dataframe(population_summary, use_container_width=True, hide_index=True)

    open_pos = {trade_date: position for position, trade_date in enumerate(open_dates)}
    all_records: list[dict[str, Any]] = []
    daily_histories: dict[str, pd.DataFrame] = {}
    reject_totals: dict[str, int] = {}
    cache_hits = 0
    data_failures = 0
    progress = st.progress(0.0, text="正在逐股票验证完整周线周期...")
    status = st.empty()

    for idx, stock in stocks_to_fetch.iterrows():
        ts_code = str(stock["ts_code"])
        progress.progress(
            (idx + 1) / len(stocks_to_fetch),
            text=f"{idx + 1}/{len(stocks_to_fetch)} {ts_code}",
        )
        status.caption(
            f"已产生底层事件{len(all_records)}条；缓存命中{cache_hits}；真实行情失败{data_failures}"
        )
        daily, basic, cache_hit = fetch_stock_history(
            ts_code, preload_start, observation_end, bool(USE_CACHE), float(API_PAUSE)
        )
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        records, rejects, _ = analyze_stock(
            stock=stock, periods=period_index.get(ts_code, []), daily=daily, basic=basic,
            week_last_map=week_last_map, open_dates=open_dates, open_pos=open_pos,
            config=config,
        )
        all_records.extend(records)
        if records:
            daily_histories[ts_code] = daily.copy()
        for reason, count in rejects.items():
            reject_totals[reason] = reject_totals.get(reason, 0) + count

    progress.empty()
    status.empty()
    if not all_records:
        st.error("没有生成有效事件，请检查区间、Token权限和价格市值条件。")
        if API_ERRORS:
            st.code("\n".join(API_ERRORS[:50]))
        return

    events = pd.DataFrame(all_records).sort_values(
        ["Signal_Date", "ts_code", "Event_Type"]
    ).reset_index(drop=True)
    with st.spinner("正在计算每一轮首红柱周期的最高价、分位数和目标命中率..."):
        opportunities = build_cycle_opportunities(
            events, daily_histories, observation_end, float(SELL_SLIPPAGE)
        )
    if opportunities.empty:
        st.error("没有形成第一根红柱机会样本。")
        return

    opportunities["Signal_Year"] = opportunities["Signal_Date"].astype(str).str[:4]
    opportunities["Circ_MV_Group"] = pd.cut(
        pd.to_numeric(opportunities["Circ_MV_Billion"], errors="coerce"),
        bins=[0, 100, 200, 500, 1000, np.inf],
        labels=["低于100亿", "100—200亿", "200—500亿", "500—1000亿", "1000亿以上"],
        right=False,
    ).astype(str)
    completed = opportunities[
        opportunities["Opportunity_Valid"].map(to_bool)
        & opportunities["Cycle_Completed"].map(to_bool)
    ].copy()
    censored = opportunities[
        opportunities["Opportunity_Valid"].map(to_bool)
        & ~opportunities["Cycle_Completed"].map(to_bool)
    ].copy()
    invalid = opportunities[~opportunities["Opportunity_Valid"].map(to_bool)].copy()
    if completed.empty:
        st.error("没有完整结束的红柱周期；请把周期观察截止日期向后延长。")
        return

    overall_summary = opportunity_summary(opportunities)
    overall_summary.insert(0, "范围", "全部完整首红柱周期")
    year_summary = opportunity_summary(opportunities, ["Signal_Year"])
    board_summary = opportunity_summary(opportunities, ["Sample_Board"])
    mv_summary = opportunity_summary(opportunities, ["Circ_MV_Group"])
    cycle_summary = opportunity_summary(opportunities, ["Cycle_Type"])
    w2_summary = opportunity_summary(opportunities, ["CP_W2_State"])
    first_strength_summary = opportunity_summary(
        opportunities, ["First_Red_Strength_Group"]
    )
    threshold_rows = []
    for target in OPPORTUNITY_TARGETS:
        hit = completed[f"Reached_{target}_pct"].map(to_bool)
        days = pd.to_numeric(
            completed.loc[hit, f"Trading_Days_To_{target}"], errors="coerce"
        )
        threshold_rows.append({
            "目标涨幅": f"+{target}%", "完整周期数": len(completed),
            "曾达到数量": int(hit.sum()), "曾达到比例(%)": pct_mean(hit),
            "达到目标交易日中位数": days.median(),
        })
    threshold_summary = pd.DataFrame(threshold_rows)
    distribution = pd.cut(
        pd.to_numeric(completed["Peak_MFE_pct"], errors="coerce"),
        bins=[-np.inf, 0, 5, 10, 20, 30, 50, 100, np.inf],
        labels=["≤0%", "0—5%", "5—10%", "10—20%", "20—30%", "30—50%", "50—100%", ">100%"],
        right=False,
    ).value_counts(sort=False).rename_axis("最高利润区间").reset_index(name="周期数")
    distribution["占完整周期比例(%)"] = distribution["周期数"] / len(completed) * 100.0
    reject_frame = pd.DataFrame(
        [{"剔除原因": reason, "次数": count} for reason, count in reject_totals.items()]
    ).sort_values("次数", ascending=False) if reject_totals else pd.DataFrame(
        columns=["剔除原因", "次数"]
    )
    run_summary = pd.DataFrame([{
        "程序": TITLE, "信号开始": signal_start, "信号截止": signal_end,
        "观察截止": observation_end, "首红柱事件": len(opportunities),
        "完整可计算周期": len(completed), "未翻绿截断周期": len(censored),
        "无法按口径计算": len(invalid), "涉及股票": opportunities["ts_code"].nunique(),
        "信号截止日后上市股票": post_signal_listings,
        "其中观察截止日后上市": post_observation_listings,
        "历史无重叠股票": no_overlap_stocks,
        "真实无行情或接口失败": data_failures, "缓存命中股票": cache_hits,
    }])
    metadata = pd.DataFrame([
        {"项目": "程序", "值": TITLE},
        {"项目": "生成时间", "值": datetime.now().isoformat(timespec="seconds")},
        {"项目": "首红柱信号区间", "值": f"{signal_start}—{signal_end}"},
        {"项目": "周期观察截止", "值": observation_end},
        {"项目": "买入价", "值": "信号周结束后下一交易日开盘价×1.002"},
        {"项目": "观察窗口", "值": "买入日至第一根完整绿柱确认日（含确认日）；未翻绿者截至观察日并单列"},
        {"项目": "核心指标", "值": "窗口内最高日线价相对买入价的涨幅；最高价仅为事后机会空间"},
        {"项目": "明确不包含", "值": "无八周上限、无止损、无止盈、无评分、无资金组合、无最高价卖出假设"},
        {"项目": "股票池", "值": "历史科技板块；主板/创业板/科创板；信号日价≥10元、流通市值≥100亿元"},
    ])

    files = {
        "01_run_summary_cycle_opportunity_v1_0.csv": run_summary,
        "02_overall_summary_cycle_opportunity_v1_0.csv": overall_summary,
        "03_threshold_summary_cycle_opportunity_v1_0.csv": threshold_summary,
        "04_peak_distribution_cycle_opportunity_v1_0.csv": distribution,
        "05_year_summary_cycle_opportunity_v1_0.csv": year_summary,
        "06_board_summary_cycle_opportunity_v1_0.csv": board_summary,
        "07_market_cap_summary_cycle_opportunity_v1_0.csv": mv_summary,
        "08_cycle_type_summary_cycle_opportunity_v1_0.csv": cycle_summary,
        "09_week2_state_summary_cycle_opportunity_v1_0.csv": w2_summary,
        "10_first_red_strength_summary_cycle_opportunity_v1_0.csv": first_strength_summary,
        "11_completed_cycle_events_v1_0.csv": completed,
        "12_censored_open_cycles_v1_0.csv": censored,
        "13_invalid_events_v1_0.csv": invalid,
        "14_full_first_red_opportunities_v1_0.csv": opportunities,
        "15_full_tech_universe_v1_0.csv": sample_audit,
        "16_population_v1_0.csv": population_summary,
        "17_rejection_audit_v1_0.csv": reject_frame,
        "18_metadata_v1_0.csv": metadata,
    }
    result_zip = make_result_zip(files)
    st.session_state["cycle_opportunity_v1_zip"] = result_zip

    overall = overall_summary.iloc[0]
    st.success(
        f"验证完成：首红柱{len(opportunities)}个；完整周期{len(completed)}个；"
        f"截至观察日仍未翻绿{len(censored)}个。"
    )
    metrics = st.columns(6)
    metrics[0].metric("完整周期", f"{len(completed)}")
    metrics[1].metric("最高利润中位数", f"{overall['最高利润中位数(%)']:.2f}%")
    metrics[2].metric("最高利润P75", f"{overall['最高利润P75(%)']:.2f}%")
    metrics[3].metric("曾达到20%", f"{overall['曾达到20%(%)']:.2f}%")
    metrics[4].metric("曾达到30%", f"{overall['曾达到30%(%)']:.2f}%")
    metrics[5].metric("到峰值中位日数", f"{overall['到最高价交易日中位数']:.0f}")

    st.subheader("核心结果：最高利润空间")
    st.dataframe(overall_summary, use_container_width=True, hide_index=True)
    st.dataframe(threshold_summary, use_container_width=True, hide_index=True)
    chart = distribution.set_index("最高利润区间")[["周期数"]]
    st.bar_chart(chart)
    with st.expander("年度、板块、市值、周期类型与第二周状态"):
        st.dataframe(year_summary, use_container_width=True, hide_index=True)
        st.dataframe(board_summary, use_container_width=True, hide_index=True)
        st.dataframe(mv_summary, use_container_width=True, hide_index=True)
        st.dataframe(cycle_summary, use_container_width=True, hide_index=True)
        st.dataframe(w2_summary, use_container_width=True, hide_index=True)
        st.dataframe(first_strength_summary, use_container_width=True, hide_index=True)

    st.subheader("下载结果")
    st.download_button(
        "下载1号：全部结果ZIP", result_zip,
        file_name="weekly_macd_cycle_opportunity_v1_0_all_results.zip",
        mime="application/zip", type="primary", on_click="ignore",
    )
    labels = [
        "2号：运行总表", "3号：总体机会空间", "4号：目标命中率", "5号：最高利润分布",
        "6号：年度汇总", "7号：板块汇总", "8号：市值汇总", "9号：周期类型汇总",
        "10号：第二周状态", "11号：首红柱强度", "12号：完整周期明细", "13号：未翻绿周期",
        "14号：无效事件", "15号：全部首红柱机会", "16号：科技股票池", "17号：板块数量",
        "18号：剔除审计", "19号：运行口径",
    ]
    columns = st.columns(4)
    for index, (filename, frame) in enumerate(files.items()):
        with columns[index % 4]:
            st.download_button(
                labels[index], csv_bytes(frame), file_name=filename,
                mime="text/csv", key=f"cycle_opp_{filename}", on_click="ignore",
            )
    if API_ERRORS:
        with st.expander("接口错误记录（最多显示50条）"):
            st.code("\n".join(API_ERRORS[:50]))
    st.warning(
        "最高价只能事后知道。本验证器只判断红柱周期是否存在足够大的机会空间，"
        "不能把最高利润当成可实现交易收益。"
    )


W2_LOOSE_STATES = {"持续扩张", "红柱平缓延续", "缩短未再扩张", "缩短后再扩张"}


def cycle_family(value: Any) -> str:
    text_value = str(value)
    if text_value.startswith(("C1_", "C2_")):
        return "C1C2_长周期"
    if text_value.startswith(("A_", "B_")):
        return "AB_弱或短周期"
    return "其他或未完成"


def delayed_cycle_path(
    event: pd.Series,
    daily: pd.DataFrame,
    open_dates: list[str],
    open_pos: dict[str, int],
    observation_end: str,
    buy_slippage_pct: float,
) -> dict[str, Any]:
    """第二根完整周线确认后，于下一市场交易日开盘买入并观察至首根完整绿柱。"""
    output: dict[str, Any] = {
        "W2_Delayed_Tradable": False,
        "W2_Delayed_Invalid_Reason": "",
        "W2_Delayed_Entry_Date": "",
        "W2_Delayed_Entry_Price": np.nan,
        "W2_Entry_Cost_vs_W1_pct": np.nan,
        "W2_Observation_End_Date": "",
        "W2_Observation_Trading_Days": np.nan,
        "W2_Peak_Date": "",
        "W2_Peak_MFE_pct": np.nan,
        "W2_Path_MAE_pct": np.nan,
        "W2_End_Close_Return_pct": np.nan,
        "W2_Trading_Days_To_Peak": np.nan,
        "W2_Lost_MFE_vs_W1_pct_points": np.nan,
    }
    for target in OPPORTUNITY_TARGETS:
        output[f"W2_Reached_{target}_pct"] = False
        output[f"W2_First_{target}_Date"] = ""
        output[f"W2_Trading_Days_To_{target}"] = np.nan

    w2_date = normalize_date(event.get("CP_W2_Date"))
    baseline_entry = finite_num(event.get("Entry_Price"))
    completed = to_bool(event.get("Cycle_Completed"))
    first_green = normalize_date(event.get("First_Green_Date"))
    end_date = first_green if completed and first_green else observation_end
    output["W2_Observation_End_Date"] = end_date

    if not w2_date or w2_date not in open_pos:
        output["W2_Delayed_Invalid_Reason"] = "缺少第二根完整周线日期"
        return output
    next_pos = open_pos[w2_date] + 1
    if next_pos >= len(open_dates):
        output["W2_Delayed_Invalid_Reason"] = "第二周确认后无下一市场交易日"
        return output
    entry_date = open_dates[next_pos]
    if entry_date > end_date:
        output["W2_Delayed_Invalid_Reason"] = "确认后的买入日已晚于观察终点"
        return output
    if daily.empty:
        output["W2_Delayed_Invalid_Reason"] = "个股日线不存在"
        return output
    entry_row = daily[daily["trade_date"].astype(str).eq(entry_date)]
    if entry_row.empty:
        output["W2_Delayed_Invalid_Reason"] = "确认后下一市场交易日停牌或无行情"
        return output
    raw_open = finite_num(entry_row.iloc[-1].get("open"))
    if not math.isfinite(raw_open) or raw_open <= 0:
        output["W2_Delayed_Invalid_Reason"] = "确认后买入日开盘价无效"
        return output
    entry_price = raw_open * (1.0 + buy_slippage_pct / 100.0)
    path = daily[
        daily["trade_date"].astype(str).ge(entry_date)
        & daily["trade_date"].astype(str).le(end_date)
    ].copy().sort_values("trade_date").reset_index(drop=True)
    for column in ("high", "low", "close"):
        path[column] = pd.to_numeric(path[column], errors="coerce")
    path = path.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
    if path.empty:
        output["W2_Delayed_Invalid_Reason"] = "确认买入日至观察终点无行情"
        return output

    peak_pos = int(path["high"].idxmax())
    peak_price = float(path.loc[peak_pos, "high"])
    peak_mfe = (peak_price / entry_price - 1.0) * 100.0
    w1_mfe = finite_num(event.get("Peak_MFE_pct"))
    output.update({
        "W2_Delayed_Tradable": True,
        "W2_Delayed_Invalid_Reason": "",
        "W2_Delayed_Entry_Date": entry_date,
        "W2_Delayed_Entry_Price": entry_price,
        "W2_Entry_Cost_vs_W1_pct": (
            (entry_price / baseline_entry - 1.0) * 100.0
            if math.isfinite(baseline_entry) and baseline_entry > 0 else np.nan
        ),
        "W2_Observation_Trading_Days": int(len(path)),
        "W2_Peak_Date": str(path.loc[peak_pos, "trade_date"]),
        "W2_Peak_MFE_pct": peak_mfe,
        "W2_Path_MAE_pct": (float(path["low"].min()) / entry_price - 1.0) * 100.0,
        "W2_End_Close_Return_pct": (
            float(path.iloc[-1]["close"]) / entry_price - 1.0
        ) * 100.0,
        "W2_Trading_Days_To_Peak": int(peak_pos + 1),
        "W2_Lost_MFE_vs_W1_pct_points": w1_mfe - peak_mfe if math.isfinite(w1_mfe) else np.nan,
    })
    for target in OPPORTUNITY_TARGETS:
        hit = path[path["high"].ge(entry_price * (1.0 + target / 100.0))]
        if not hit.empty:
            first_pos = int(hit.index[0])
            output[f"W2_Reached_{target}_pct"] = True
            output[f"W2_First_{target}_Date"] = str(path.loc[first_pos, "trade_date"])
            output[f"W2_Trading_Days_To_{target}"] = int(first_pos + 1)
    return output


def build_week2_paired_results(
    opportunities: pd.DataFrame,
    daily_histories: dict[str, pd.DataFrame],
    open_dates: list[str],
    observation_end: str,
    buy_slippage_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_pos = {trade_date: position for position, trade_date in enumerate(open_dates)}
    for _, event in opportunities.iterrows():
        row = event.to_dict()
        state = str(event.get("CP_W2_State", ""))
        row["Cycle_Family"] = cycle_family(event.get("Cycle_Type"))
        row["W2_Loose_Eligible"] = state in W2_LOOSE_STATES
        row["W2_Expansion_Eligible"] = state == "持续扩张"
        if row["W2_Loose_Eligible"]:
            row.update(delayed_cycle_path(
                event=event,
                daily=daily_histories.get(str(event.get("ts_code", "")), pd.DataFrame()),
                open_dates=open_dates,
                open_pos=open_pos,
                observation_end=observation_end,
                buy_slippage_pct=buy_slippage_pct,
            ))
        else:
            row.update(delayed_cycle_path(
                event=pd.Series({}), daily=pd.DataFrame(), open_dates=open_dates,
                open_pos=open_pos, observation_end=observation_end,
                buy_slippage_pct=buy_slippage_pct,
            ))
            row["W2_Delayed_Invalid_Reason"] = "第二周未保持红柱"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True)


def confirmation_summary(paired: pd.DataFrame) -> pd.DataFrame:
    complete = paired[
        paired["Opportunity_Valid"].map(to_bool)
        & paired["Cycle_Completed"].map(to_bool)
    ].copy()
    strong_total = int(complete["Cycle_Family"].eq("C1C2_长周期").sum())
    weak_total = int(complete["Cycle_Family"].eq("AB_弱或短周期").sum())
    definitions = [
        ("首红柱立即买入_全部", pd.Series(True, index=complete.index), False),
        ("第二周仍红后买入_宽松确认", complete["W2_Loose_Eligible"].map(to_bool), True),
        ("第二周扩张后买入_核心方案", complete["W2_Expansion_Eligible"].map(to_bool), True),
    ]
    rows: list[dict[str, Any]] = []
    for name, eligible_mask, delayed in definitions:
        selected = complete[eligible_mask].copy()
        if delayed:
            selected = selected[selected["W2_Delayed_Tradable"].map(to_bool)].copy()
            mfe_col, mae_col, days_col, hit_prefix = (
                "W2_Peak_MFE_pct", "W2_Path_MAE_pct", "W2_Trading_Days_To_Peak", "W2_Reached_"
            )
        else:
            mfe_col, mae_col, days_col, hit_prefix = (
                "Peak_MFE_pct", "Path_MAE_pct", "Trading_Days_To_Peak", "Reached_"
            )
        strong = int(selected["Cycle_Family"].eq("C1C2_长周期").sum())
        weak = int(selected["Cycle_Family"].eq("AB_弱或短周期").sum())
        mfe = pd.to_numeric(selected[mfe_col], errors="coerce")
        row = {
            "方案": name,
            "完整可执行周期": int(len(selected)),
            "占全部首红柱周期(%)": len(selected) / len(complete) * 100.0 if len(complete) else np.nan,
            "C1C2数量": strong,
            "AB数量": weak,
            "C1C2占入选比例(%)": strong / len(selected) * 100.0 if len(selected) else np.nan,
            "C1C2保留率(%)": strong / strong_total * 100.0 if strong_total else np.nan,
            "AB保留率(%)": weak / weak_total * 100.0 if weak_total else np.nan,
            "机会最高涨幅均值(%)": mfe.mean(),
            "机会最高涨幅中位数(%)": mfe.median(),
            "机会最高涨幅P25(%)": mfe.quantile(0.25),
            "机会最高涨幅P75(%)": mfe.quantile(0.75),
            "最大浮亏中位数(%)": pd.to_numeric(selected[mae_col], errors="coerce").median(),
            "到峰值交易日中位数": pd.to_numeric(selected[days_col], errors="coerce").median(),
            "追高成本中位数(%)": (
                pd.to_numeric(selected["W2_Entry_Cost_vs_W1_pct"], errors="coerce").median()
                if delayed else 0.0
            ),
            "损失最高利润中位数(百分点)": (
                pd.to_numeric(selected["W2_Lost_MFE_vs_W1_pct_points"], errors="coerce").median()
                if delayed else 0.0
            ),
        }
        for target in OPPORTUNITY_TARGETS:
            row[f"曾达到{target}%(%)"] = pct_mean(selected[f"{hit_prefix}{target}_pct"].map(to_bool))
        rows.append(row)
    return pd.DataFrame(rows)


def selection_effect_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """用所有方案共同的首红柱买价，只检查筛选质量，不混入延迟买入成本。"""
    complete = paired[
        paired["Opportunity_Valid"].map(to_bool)
        & paired["Cycle_Completed"].map(to_bool)
    ].copy()
    definitions = [
        ("全部首红柱", pd.Series(True, index=complete.index)),
        ("第二周仍红所保留的周期", complete["W2_Loose_Eligible"].map(to_bool)),
        ("第二周扩张所保留的周期", complete["W2_Expansion_Eligible"].map(to_bool)),
        ("第二周未扩张而被核心方案过滤", ~complete["W2_Expansion_Eligible"].map(to_bool)),
    ]
    rows = []
    for name, mask in definitions:
        group = complete[mask].copy()
        mfe = pd.to_numeric(group["Peak_MFE_pct"], errors="coerce")
        strong = group["Cycle_Family"].eq("C1C2_长周期")
        row = {
            "分组": name, "周期数": len(group),
            "C1C2比例(%)": pct_mean(strong),
            "按首红柱买价的最高涨幅均值(%)": mfe.mean(),
            "按首红柱买价的最高涨幅中位数(%)": mfe.median(),
            "按首红柱买价的最大浮亏中位数(%)": pd.to_numeric(
                group["Path_MAE_pct"], errors="coerce"
            ).median(),
        }
        for target in OPPORTUNITY_TARGETS:
            row[f"按首红柱买价曾达到{target}%(%)"] = pct_mean(
                group[f"Reached_{target}_pct"].map(to_bool)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_year_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    complete = paired[
        paired["Opportunity_Valid"].map(to_bool)
        & paired["Cycle_Completed"].map(to_bool)
    ].copy()
    complete["Signal_Year"] = complete["Signal_Date"].astype(str).str[:4]
    for year, year_frame in complete.groupby("Signal_Year", sort=True):
        for scheme, mask in [
            ("首红柱立即买入", pd.Series(True, index=year_frame.index)),
            ("第二周仍红后买入", year_frame["W2_Loose_Eligible"].map(to_bool)),
            ("第二周扩张后买入", year_frame["W2_Expansion_Eligible"].map(to_bool)),
        ]:
            group = year_frame[mask].copy()
            delayed = scheme != "首红柱立即买入"
            if delayed:
                group = group[group["W2_Delayed_Tradable"].map(to_bool)]
            mfe_col = "W2_Peak_MFE_pct" if delayed else "Peak_MFE_pct"
            hit_prefix = "W2_Reached_" if delayed else "Reached_"
            rows.append({
                "年份": year, "方案": scheme, "周期数": len(group),
                "C1C2比例(%)": pct_mean(group["Cycle_Family"].eq("C1C2_长周期")),
                "最高涨幅中位数(%)": pd.to_numeric(group[mfe_col], errors="coerce").median(),
                "曾达到10%(%)": pct_mean(group[f"{hit_prefix}10_pct"].map(to_bool)),
                "曾达到20%(%)": pct_mean(group[f"{hit_prefix}20_pct"].map(to_bool)),
                "曾达到30%(%)": pct_mean(group[f"{hit_prefix}30_pct"].map(to_bool)),
            })
    return pd.DataFrame(rows)


def week2_confirmation_main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "同一批首红柱周期做一对一配对：基准在第一根红柱确认后买入；第二周方案等待"
        "第二根完整周线确认，再于下一交易日开盘买入。只比较周期机会空间，不设置止盈止损。"
    )

    with st.sidebar:
        st.header("信号与观察区间")
        signal_start_date = st.date_input("首红柱信号开始日期", value=date(2022, 6, 5))
        signal_end_date = st.date_input("首红柱信号截止日期", value=date(2023, 6, 5))
        suggested_end = min(date.today(), signal_end_date + timedelta(days=550))
        observation_end_date = st.date_input(
            "周期观察截止日期", value=suggested_end, max_value=date.today()
        )
        st.header("固定研究口径")
        st.write("历史科技板块全量检查，不抽样")
        st.write("信号日股价≥10元、流通市值≥100亿元")
        st.write("真正完整周MACD(12,26,9)")
        st.write("核心确认：第二周红柱比第一周至少增加10%")
        st.write("买入价均计0.2%滑点")
        use_cache = st.checkbox("使用逐股票缓存", value=True)
        api_pause = st.number_input(
            "每次API调用后暂停(秒)", min_value=0.0, max_value=3.0,
            value=0.12, step=0.05,
        )
        if st.button("清除本程序缓存"):
            if os.path.isdir(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            st.success("缓存已清除")

    token = st.text_input("Tushare Token", type="password")
    if not token:
        st.info("请输入Tushare Token。")
        return
    run_requested = st.button("开始第二周确认法配对验证", type="primary")
    if not run_requested:
        if "week2_confirmation_v1_1_zip" in st.session_state:
            st.success("上一次结果仍在，可直接下载。")
            st.download_button(
                "下载1号：上一次全部结果ZIP",
                st.session_state["week2_confirmation_v1_1_zip"],
                file_name="weekly_macd_week2_confirmation_v1_1_all_results.zip",
                mime="application/zip", type="primary", on_click="ignore",
            )
        else:
            st.info(
                "重点看三项：C1C2占比提高多少、第二周确认的追高成本、确认后仍能达到20%/30%的比例。"
            )
        return
    if signal_start_date >= signal_end_date:
        st.error("信号开始日期必须早于截止日期。")
        return
    if observation_end_date <= signal_end_date:
        st.error("观察截止日期必须晚于信号截止日期。")
        return

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    signal_start = signal_start_date.strftime("%Y%m%d")
    signal_end = signal_end_date.strftime("%Y%m%d")
    observation_end = observation_end_date.strftime("%Y%m%d")
    preload_start = (signal_start_date - timedelta(days=3 * 365)).strftime("%Y%m%d")
    calendar_tail = (observation_end_date + timedelta(days=7)).strftime("%Y%m%d")
    config = {
        "signal_start": signal_start, "signal_end": signal_end,
        "market_end": observation_end, "preload_start": preload_start,
        "min_price": 10.0, "min_mv": 100.0, "max_mv": 1_000_000_000.0,
        "price_tolerance_pct": 3.0, "stop_threshold_pct": 10.0,
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "sample_per_board": 0, "sample_seed": DEFAULT_SAMPLE_SEED,
        "long_cycle_min_weeks": DEFAULT_LONG_CYCLE_MIN_WEEKS,
        "material_hist_change_pct": DEFAULT_MATERIAL_HIST_CHANGE_PCT,
        "short_strength_ratio": DEFAULT_SHORT_STRENGTH_RATIO,
    }
    try:
        with st.spinner("加载交易日历、历史科技股票池和申万历史成分..."):
            open_dates = load_trade_calendar(preload_start, observation_end)
            full_calendar = load_trade_calendar(preload_start, calendar_tail)
            stock_basic = load_stock_basic()
            memberships = load_sw_tech_memberships(float(api_pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return

    week_last_map = complete_week_last_dates(full_calendar)
    period_index = build_period_index(memberships)
    universe_codes = sorted(set(period_index) & set(stock_basic["ts_code"].astype(str)))
    universe = stock_basic[stock_basic["ts_code"].isin(universe_codes)].copy()
    stocks, sample_audit, population_summary = build_stratified_sample(
        stocks=universe, period_index=period_index, reference_date=signal_end,
        per_board=0, seed=DEFAULT_SAMPLE_SEED,
    )
    list_dates = stocks["list_date"].apply(lambda x: normalize_date(x, "19000101"))
    delist_dates = stocks["delist_date"].apply(lambda x: normalize_date(x, "99991231"))
    stocks_to_fetch = stocks[
        ~list_dates.gt(signal_end) & ~delist_dates.lt(preload_start)
    ].copy().reset_index(drop=True)
    st.write(
        f"历史科技池{len(universe)}只；实际读取{len(stocks_to_fetch)}只；"
        f"首红柱区间{signal_start}—{signal_end}；观察至{observation_end}。"
    )

    open_pos = {trade_date: position for position, trade_date in enumerate(open_dates)}
    all_records: list[dict[str, Any]] = []
    daily_histories: dict[str, pd.DataFrame] = {}
    reject_totals: dict[str, int] = {}
    cache_hits = data_failures = 0
    progress = st.progress(0.0, text="逐股票生成真正周线首红柱事件...")
    status = st.empty()
    for idx, stock in stocks_to_fetch.iterrows():
        code = str(stock["ts_code"])
        progress.progress((idx + 1) / len(stocks_to_fetch), text=f"{idx + 1}/{len(stocks_to_fetch)} {code}")
        status.caption(f"底层事件{len(all_records)}；缓存命中{cache_hits}；行情失败{data_failures}")
        daily, basic, cache_hit = fetch_stock_history(
            code, preload_start, observation_end, bool(use_cache), float(api_pause)
        )
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        records, rejects, _ = analyze_stock(
            stock=stock, periods=period_index.get(code, []), daily=daily, basic=basic,
            week_last_map=week_last_map, open_dates=open_dates, open_pos=open_pos,
            config=config,
        )
        all_records.extend(records)
        if records:
            daily_histories[code] = daily.copy()
        for reason, count in rejects.items():
            reject_totals[reason] = reject_totals.get(reason, 0) + count
    progress.empty()
    status.empty()
    if not all_records:
        st.error("没有生成有效事件，请检查日期、Token权限和价格市值条件。")
        return

    events = pd.DataFrame(all_records).sort_values(["Signal_Date", "ts_code", "Event_Type"])
    with st.spinner("计算首红柱基准机会与第二周确认后的剩余机会..."):
        opportunities = build_cycle_opportunities(
            events, daily_histories, observation_end, config["sell_slippage_pct"]
        )
        paired = build_week2_paired_results(
            opportunities, daily_histories, open_dates, observation_end,
            config["buy_slippage_pct"],
        )
        scheme_summary = confirmation_summary(paired)
        selection_summary = selection_effect_summary(paired)
        year_summary = paired_year_summary(paired)

    complete = paired[
        paired["Opportunity_Valid"].map(to_bool)
        & paired["Cycle_Completed"].map(to_bool)
    ].copy()
    if complete.empty:
        st.error("没有完整可计算周期，请延长观察截止日期。")
        return
    class_table = pd.crosstab(
        complete["CP_W2_State"], complete["Cycle_Family"], margins=True
    ).reset_index()
    w2_invalid = complete[
        complete["W2_Loose_Eligible"].map(to_bool)
        & ~complete["W2_Delayed_Tradable"].map(to_bool)
    ].copy()
    reject_frame = pd.DataFrame([
        {"剔除原因": reason, "次数": count} for reason, count in reject_totals.items()
    ]).sort_values("次数", ascending=False) if reject_totals else pd.DataFrame(
        columns=["剔除原因", "次数"]
    )
    run_summary = pd.DataFrame([{
        "程序": TITLE, "信号开始": signal_start, "信号截止": signal_end,
        "观察截止": observation_end, "首红柱事件": len(paired),
        "完整可计算周期": len(complete),
        "第二周仍红": int(complete["W2_Loose_Eligible"].map(to_bool).sum()),
        "第二周扩张": int(complete["W2_Expansion_Eligible"].map(to_bool).sum()),
        "涉及股票": paired["ts_code"].nunique(), "真实行情失败": data_failures,
        "缓存命中": cache_hits,
    }])
    metadata = pd.DataFrame([
        {"项目": "程序", "值": TITLE},
        {"项目": "基准买点", "值": "第一根完整红柱确认后的下一交易日开盘×1.002"},
        {"项目": "宽松确认", "值": "第二根完整周线仍为红柱，确认后的下一交易日开盘×1.002"},
        {"项目": "核心确认", "值": "第二根红柱≥第一根红柱×1.10，确认后的下一交易日开盘×1.002"},
        {"项目": "机会终点", "值": "第一根完整绿柱确认日（含该日）；未翻绿样本单列，不进入核心汇总"},
        {"项目": "C1C2用途", "值": "仅作事后审查标签，绝不参与第二周选股"},
        {"项目": "明确不包含", "值": "无止盈、无止损、无资金组合、无最高价卖出假设"},
        {"项目": "股票池", "值": "历史科技板块全量；信号日价≥10元、流通市值≥100亿元"},
    ])
    files = {
        "01_run_summary_week2_v1_1.csv": run_summary,
        "02_scheme_result_summary_week2_v1_1.csv": scheme_summary,
        "03_selection_effect_week2_v1_1.csv": selection_summary,
        "04_year_robustness_week2_v1_1.csv": year_summary,
        "05_week2_state_vs_cycle_class_v1_1.csv": class_table,
        "06_complete_paired_events_week2_v1_1.csv": complete,
        "07_all_paired_events_week2_v1_1.csv": paired,
        "08_delayed_entry_invalid_week2_v1_1.csv": w2_invalid,
        "09_full_tech_universe_week2_v1_1.csv": sample_audit,
        "10_population_week2_v1_1.csv": population_summary,
        "11_rejection_audit_week2_v1_1.csv": reject_frame,
        "12_metadata_week2_v1_1.csv": metadata,
    }
    result_zip = make_result_zip(files)
    st.session_state["week2_confirmation_v1_1_zip"] = result_zip

    core = scheme_summary[scheme_summary["方案"].eq("第二周扩张后买入_核心方案")].iloc[0]
    st.success(
        f"验证完成：完整周期{len(complete)}个；核心方案可执行{int(core['完整可执行周期'])}个；"
        f"其中C1C2占{core['C1C2占入选比例(%)']:.2f}%。"
    )
    metrics = st.columns(6)
    metrics[0].metric("核心入选周期", f"{int(core['完整可执行周期'])}")
    metrics[1].metric("C1C2比例", f"{core['C1C2占入选比例(%)']:.2f}%")
    metrics[2].metric("C1C2保留率", f"{core['C1C2保留率(%)']:.2f}%")
    metrics[3].metric("追高成本中位数", f"{core['追高成本中位数(%)']:.2f}%")
    metrics[4].metric("剩余最高涨幅中位数", f"{core['机会最高涨幅中位数(%)']:.2f}%")
    metrics[5].metric("确认后曾达到30%", f"{core['曾达到30%(%)']:.2f}%")

    st.subheader("核心比较：实际按各自买点计算")
    st.dataframe(scheme_summary, use_container_width=True, hide_index=True)
    st.subheader("拆分筛选效果：全部改用首红柱买价，暂时不计算追高成本")
    st.dataframe(selection_summary, use_container_width=True, hide_index=True)
    with st.expander("年度稳定性与第二周状态×周期类型"):
        st.dataframe(year_summary, use_container_width=True, hide_index=True)
        st.dataframe(class_table, use_container_width=True, hide_index=True)

    st.subheader("下载结果")
    st.download_button(
        "下载1号：全部结果ZIP", result_zip,
        file_name="weekly_macd_week2_confirmation_v1_1_all_results.zip",
        mime="application/zip", type="primary", on_click="ignore",
    )
    labels = [
        "2号：运行总表", "3号：三方案实际结果", "4号：纯筛选效果", "5号：年度稳健性",
        "6号：第二周状态与周期类型", "7号：完整周期配对明细", "8号：全部配对明细",
        "9号：延迟买入无效", "10号：科技股票池", "11号：板块数量",
        "12号：剔除审计", "13号：验证口径",
    ]
    columns = st.columns(4)
    for index, (filename, frame) in enumerate(files.items()):
        with columns[index % 4]:
            st.download_button(
                labels[index], csv_bytes(frame), file_name=filename,
                mime="text/csv", key=f"w2_v11_{filename}", on_click="ignore",
            )
    st.warning(
        "第二周确认只使用当时可见信息；C1/C2和周期最高价只用于事后验证。"
        "本程序比较机会空间，不等于已经解决止盈问题。"
    )


def w2_exact_expansion(event: pd.Series) -> bool:
    """第二根完整柱严格长于第一根；不使用10%或其他幅度门槛。"""
    first_hist = finite_num(event.get("Hist"))
    second_hist = finite_num(event.get("CP_W2_Hist"))
    return bool(
        math.isfinite(first_hist)
        and math.isfinite(second_hist)
        and first_hist > 0
        and second_hist > first_hist
    )


def empty_w2_exit_result(reason: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {
        "W2_Decision_Valid": False,
        "W2_Decision_Action": "",
        "W2_Exact_Expansion": False,
        "W2_Expansion_pct": np.nan,
        "W2_Exit_Required": False,
        "W2_Exit_Executable": False,
        "W2_Exit_Invalid_Reason": reason,
        "W2_Exit_Scheduled_Date": "",
        "W2_Exit_Date": "",
        "W2_Exit_Delay_Market_Days": np.nan,
        "W2_Exit_Raw_Open": np.nan,
        "W2_Exit_Net_Price": np.nan,
        "W2_Exit_Return_pct": np.nan,
        "W2_Holding_Market_Days": np.nan,
        "W2_PreExit_MFE_pct": np.nan,
        "W2_PreExit_MAE_pct": np.nan,
        "W2_Exit_vs_Cycle_End_Close_pct_points": np.nan,
        "W2_Full_MFE_minus_Exit_Return_pct_points": np.nan,
        "W2_PostExit_Path_Exists": False,
        "W2_PostExit_Peak_Date": "",
        "W2_PostExit_Peak_From_W1_pct": np.nan,
        "W2_PostExit_Peak_From_Exit_Open_pct": np.nan,
        "W2_PostExit_Trough_From_Exit_Open_pct": np.nan,
    }
    for target in OPPORTUNITY_TARGETS:
        output[f"W2_PostExit_Reached_{target}_From_W1"] = False
    return output


def evaluate_w2_exit(
    event: pd.Series,
    daily: pd.DataFrame,
    open_dates: list[str],
    open_pos: dict[str, int],
    sell_slippage_pct: float,
) -> dict[str, Any]:
    """第二周不扩张时，在下一市场交易日起寻找首个可交易开盘并卖出。"""
    output = empty_w2_exit_result()
    first_hist = finite_num(event.get("Hist"))
    second_hist = finite_num(event.get("CP_W2_Hist"))
    w2_date = normalize_date(event.get("CP_W2_Date"))
    expands = w2_exact_expansion(event)
    output.update({
        "W2_Decision_Valid": bool(
            w2_date and math.isfinite(first_hist) and math.isfinite(second_hist)
        ),
        "W2_Decision_Action": "继续持有" if expands else "退出",
        "W2_Exact_Expansion": expands,
        "W2_Expansion_pct": (
            (second_hist / first_hist - 1.0) * 100.0
            if math.isfinite(first_hist) and first_hist != 0 and math.isfinite(second_hist)
            else np.nan
        ),
        "W2_Exit_Required": not expands,
    })
    if not output["W2_Decision_Valid"]:
        output["W2_Exit_Invalid_Reason"] = "缺少第二根完整周线或柱值"
        return output
    if expands:
        return output
    if w2_date not in open_pos:
        output["W2_Exit_Invalid_Reason"] = "第二周确认日不在市场交易日历"
        return output
    scheduled_pos = open_pos[w2_date] + 1
    if scheduled_pos >= len(open_dates):
        output["W2_Exit_Invalid_Reason"] = "第二周确认后无下一市场交易日"
        return output
    scheduled_date = open_dates[scheduled_pos]
    output["W2_Exit_Scheduled_Date"] = scheduled_date
    if daily.empty:
        output["W2_Exit_Invalid_Reason"] = "个股日线不存在"
        return output

    daily_work = daily.copy()
    daily_work["trade_date"] = daily_work["trade_date"].astype(str)
    candidates = daily_work[daily_work["trade_date"].ge(scheduled_date)].copy()
    candidates["open"] = pd.to_numeric(candidates["open"], errors="coerce")
    candidates = candidates[candidates["open"].gt(0)].sort_values("trade_date")
    if candidates.empty:
        output["W2_Exit_Invalid_Reason"] = "确认后没有可交易开盘"
        return output
    exit_row = candidates.iloc[0]
    exit_date = str(exit_row["trade_date"])
    raw_open = float(exit_row["open"])
    net_exit = raw_open * (1.0 - sell_slippage_pct / 100.0)
    entry_date = normalize_date(event.get("Entry_Date"))
    entry_price = finite_num(event.get("Entry_Price"))
    if not entry_date or not math.isfinite(entry_price) or entry_price <= 0:
        output["W2_Exit_Invalid_Reason"] = "首红柱买入价无效"
        return output

    before = daily_work[
        daily_work["trade_date"].ge(entry_date)
        & daily_work["trade_date"].lt(exit_date)
    ].copy()
    for column in ("high", "low"):
        before[column] = pd.to_numeric(before[column], errors="coerce")
    peak_candidates = [raw_open]
    trough_candidates = [raw_open]
    if not before.empty:
        if before["high"].notna().any():
            peak_candidates.append(float(before["high"].max()))
        if before["low"].notna().any():
            trough_candidates.append(float(before["low"].min()))

    full_mfe = finite_num(event.get("Peak_MFE_pct"))
    end_close_return = finite_num(event.get("End_Close_Return_pct"))
    exit_return = (net_exit / entry_price - 1.0) * 100.0
    delay_days = (
        open_pos.get(exit_date, scheduled_pos) - scheduled_pos
        if exit_date in open_pos else np.nan
    )
    holding_days = (
        open_pos[exit_date] - open_pos[entry_date] + 1
        if exit_date in open_pos and entry_date in open_pos else np.nan
    )
    output.update({
        "W2_Exit_Executable": True,
        "W2_Exit_Invalid_Reason": "",
        "W2_Exit_Date": exit_date,
        "W2_Exit_Delay_Market_Days": delay_days,
        "W2_Exit_Raw_Open": raw_open,
        "W2_Exit_Net_Price": net_exit,
        "W2_Exit_Return_pct": exit_return,
        "W2_Holding_Market_Days": holding_days,
        "W2_PreExit_MFE_pct": (max(peak_candidates) / entry_price - 1.0) * 100.0,
        "W2_PreExit_MAE_pct": (min(trough_candidates) / entry_price - 1.0) * 100.0,
        "W2_Exit_vs_Cycle_End_Close_pct_points": (
            exit_return - end_close_return if math.isfinite(end_close_return) else np.nan
        ),
        "W2_Full_MFE_minus_Exit_Return_pct_points": (
            full_mfe - exit_return if math.isfinite(full_mfe) else np.nan
        ),
    })

    cycle_end = normalize_date(event.get("Observation_End_Date"))
    if cycle_end and cycle_end >= exit_date:
        post = daily_work[
            daily_work["trade_date"].ge(exit_date)
            & daily_work["trade_date"].le(cycle_end)
        ].copy().sort_values("trade_date").reset_index(drop=True)
        for column in ("high", "low"):
            post[column] = pd.to_numeric(post[column], errors="coerce")
        post = post.dropna(subset=["high", "low"])
        if not post.empty:
            peak_pos = int(post["high"].idxmax())
            peak_price = float(post.loc[peak_pos, "high"])
            trough_price = float(post["low"].min())
            output.update({
                "W2_PostExit_Path_Exists": True,
                "W2_PostExit_Peak_Date": str(post.loc[peak_pos, "trade_date"]),
                "W2_PostExit_Peak_From_W1_pct": (peak_price / entry_price - 1.0) * 100.0,
                "W2_PostExit_Peak_From_Exit_Open_pct": (peak_price / raw_open - 1.0) * 100.0,
                "W2_PostExit_Trough_From_Exit_Open_pct": (trough_price / raw_open - 1.0) * 100.0,
            })
            for target in OPPORTUNITY_TARGETS:
                output[f"W2_PostExit_Reached_{target}_From_W1"] = bool(
                    peak_price >= entry_price * (1.0 + target / 100.0)
                )
    return output


def build_week2_hold_exit_results(
    opportunities: pd.DataFrame,
    daily_histories: dict[str, pd.DataFrame],
    open_dates: list[str],
    sell_slippage_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_pos = {trade_date: position for position, trade_date in enumerate(open_dates)}
    for _, event in opportunities.iterrows():
        row = event.to_dict()
        row["Cycle_Family"] = cycle_family(event.get("Cycle_Type"))
        row.update(evaluate_w2_exit(
            event=event,
            daily=daily_histories.get(str(event.get("ts_code", "")), pd.DataFrame()),
            open_dates=open_dates,
            open_pos=open_pos,
            sell_slippage_pct=sell_slippage_pct,
        ))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True)


def complete_decision_events(results: pd.DataFrame) -> pd.DataFrame:
    return results[
        results["Opportunity_Valid"].map(to_bool)
        & results["Cycle_Completed"].map(to_bool)
        & results["W2_Decision_Valid"].map(to_bool)
    ].copy()


def build_decision_matrix(results: pd.DataFrame) -> pd.DataFrame:
    complete = complete_decision_events(results)
    matrix = pd.crosstab(
        complete["W2_Decision_Action"], complete["Cycle_Family"], margins=True
    ).reset_index()
    strong = complete["Cycle_Family"].eq("C1C2_长周期")
    weak = complete["Cycle_Family"].eq("AB_弱或短周期")
    exit_mask = complete["W2_Decision_Action"].eq("退出")
    audit = pd.DataFrame([{
        "完整可判断周期": int(len(complete)),
        "继续持有": int((~exit_mask).sum()),
        "提前退出": int(exit_mask.sum()),
        "正确退出AB": int((exit_mask & weak).sum()),
        "误退出C1C2": int((exit_mask & strong).sum()),
        "仍持有AB": int((~exit_mask & weak).sum()),
        "仍持有C1C2": int((~exit_mask & strong).sum()),
        "退出组AB纯度(%)": pct_mean(weak[exit_mask]),
        "AB提前退出率(%)": pct_mean(exit_mask[weak]),
        "C1C2误退出率(%)": pct_mean(exit_mask[strong]),
        "继续持有组C1C2比例(%)": pct_mean(strong[~exit_mask]),
    }])
    return audit, matrix


def build_early_exit_summary(results: pd.DataFrame) -> pd.DataFrame:
    complete = complete_decision_events(results)
    exited = complete[
        complete["W2_Decision_Action"].eq("退出")
        & complete["W2_Exit_Executable"].map(to_bool)
    ].copy()
    definitions: list[tuple[str, pd.Series]] = [
        ("全部提前退出", pd.Series(True, index=exited.index)),
        ("提前退出_AB", exited["Cycle_Family"].eq("AB_弱或短周期")),
        ("提前退出_C1C2", exited["Cycle_Family"].eq("C1C2_长周期")),
    ]
    for state in sorted(exited["CP_W2_State"].dropna().astype(str).unique()):
        definitions.append((f"第二周状态_{state}", exited["CP_W2_State"].astype(str).eq(state)))
    rows: list[dict[str, Any]] = []
    for name, mask in definitions:
        group = exited[mask].copy()
        exit_return = pd.to_numeric(group["W2_Exit_Return_pct"], errors="coerce")
        row = {
            "分组": name,
            "可执行退出周期": int(len(group)),
            "C1C2比例(%)": pct_mean(group["Cycle_Family"].eq("C1C2_长周期")),
            "退出收益均值(%)": exit_return.mean(),
            "退出收益中位数(%)": exit_return.median(),
            "退出收益P25(%)": exit_return.quantile(0.25),
            "退出收益P75(%)": exit_return.quantile(0.75),
            "退出盈利比例(%)": pct_mean(exit_return.gt(0)),
            "退出亏损不超过-5%(%)": pct_mean(exit_return.le(-5)),
            "退出亏损不超过-10%(%)": pct_mean(exit_return.le(-10)),
            "退出前最高浮盈中位数(%)": pd.to_numeric(
                group["W2_PreExit_MFE_pct"], errors="coerce"
            ).median(),
            "退出前最大浮亏中位数(%)": pd.to_numeric(
                group["W2_PreExit_MAE_pct"], errors="coerce"
            ).median(),
            "若持有至周期结束收盘收益中位数(%)": pd.to_numeric(
                group["End_Close_Return_pct"], errors="coerce"
            ).median(),
            "提前退出相对周期结束改善中位数(百分点)": pd.to_numeric(
                group["W2_Exit_vs_Cycle_End_Close_pct_points"], errors="coerce"
            ).median(),
            "退出后周期内最高涨幅_相对首买价中位数(%)": pd.to_numeric(
                group["W2_PostExit_Peak_From_W1_pct"], errors="coerce"
            ).median(),
            "退出后周期内最高涨幅_相对退出开盘中位数(%)": pd.to_numeric(
                group["W2_PostExit_Peak_From_Exit_Open_pct"], errors="coerce"
            ).median(),
            "完整机会最高涨幅减退出收益中位数(百分点)": pd.to_numeric(
                group["W2_Full_MFE_minus_Exit_Return_pct_points"], errors="coerce"
            ).median(),
        }
        for target in (10, 20, 30):
            valid_future = group[group["W2_PostExit_Path_Exists"].map(to_bool)]
            row[f"退出后仍曾达到{target}%_占有后续路径比例(%)"] = pct_mean(
                valid_future[f"W2_PostExit_Reached_{target}_From_W1"].map(to_bool)
            ) if len(valid_future) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_held_opportunity_summary(results: pd.DataFrame) -> pd.DataFrame:
    complete = complete_decision_events(results)
    held = complete[complete["W2_Decision_Action"].eq("继续持有")].copy()
    definitions = [
        ("全部继续持有", pd.Series(True, index=held.index)),
        ("继续持有_C1C2", held["Cycle_Family"].eq("C1C2_长周期")),
        ("继续持有_AB", held["Cycle_Family"].eq("AB_弱或短周期")),
    ]
    rows: list[dict[str, Any]] = []
    for name, mask in definitions:
        group = held[mask].copy()
        mfe = pd.to_numeric(group["Peak_MFE_pct"], errors="coerce")
        row = {
            "分组": name,
            "周期数": int(len(group)),
            "C1C2比例(%)": pct_mean(group["Cycle_Family"].eq("C1C2_长周期")),
            "机会最高涨幅均值(%)": mfe.mean(),
            "机会最高涨幅中位数(%)": mfe.median(),
            "机会最高涨幅P25(%)": mfe.quantile(0.25),
            "机会最高涨幅P75(%)": mfe.quantile(0.75),
            "全周期最大浮亏中位数(%)": pd.to_numeric(
                group["Path_MAE_pct"], errors="coerce"
            ).median(),
            "周期结束收盘收益中位数(%)": pd.to_numeric(
                group["End_Close_Return_pct"], errors="coerce"
            ).median(),
        }
        for target in OPPORTUNITY_TARGETS:
            row[f"曾达到{target}%(%)"] = pct_mean(group[f"Reached_{target}_pct"].map(to_bool))
        rows.append(row)
    return pd.DataFrame(rows)


def build_hold_exit_year_summary(results: pd.DataFrame) -> pd.DataFrame:
    complete = complete_decision_events(results)
    complete["Signal_Year"] = complete["Signal_Date"].astype(str).str[:4]
    rows: list[dict[str, Any]] = []
    for year, group in complete.groupby("Signal_Year", sort=True):
        strong = group["Cycle_Family"].eq("C1C2_长周期")
        weak = group["Cycle_Family"].eq("AB_弱或短周期")
        exit_mask = group["W2_Decision_Action"].eq("退出")
        held = group[~exit_mask]
        exited = group[exit_mask & group["W2_Exit_Executable"].map(to_bool)]
        rows.append({
            "年份": year,
            "完整可判断周期": int(len(group)),
            "提前退出": int(exit_mask.sum()),
            "继续持有": int((~exit_mask).sum()),
            "AB提前退出率(%)": pct_mean(exit_mask[weak]),
            "C1C2误退出率(%)": pct_mean(exit_mask[strong]),
            "继续持有组C1C2比例(%)": pct_mean(strong[~exit_mask]),
            "提前退出收益中位数(%)": pd.to_numeric(
                exited["W2_Exit_Return_pct"], errors="coerce"
            ).median(),
            "继续持有组最高涨幅中位数(%)": pd.to_numeric(
                held["Peak_MFE_pct"], errors="coerce"
            ).median(),
            "继续持有组曾达到20%(%)": pct_mean(held["Reached_20_pct"].map(to_bool)),
            "继续持有组曾达到30%(%)": pct_mean(held["Reached_30_pct"].map(to_bool)),
        })
    return pd.DataFrame(rows)


def week2_hold_exit_main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "固定首红柱后次日开盘买入。第二根完整红柱严格长于第一根则继续持有；"
        "缩短、持平或翻绿则在下一可交易日开盘退出。无10%扩张门槛。"
    )

    with st.sidebar:
        st.header("信号与观察区间")
        signal_start_date = st.date_input("首红柱信号开始日期", value=date(2022, 6, 5))
        signal_end_date = st.date_input("首红柱信号截止日期", value=date(2023, 6, 5))
        suggested_end = min(date.today(), signal_end_date + timedelta(days=550))
        observation_end_date = st.date_input(
            "周期观察截止日期", value=suggested_end, max_value=date.today()
        )
        st.header("固定研究口径")
        st.write("历史科技板块全量；信号日股价≥10元、流通市值≥100亿元")
        st.write("第一根完整红柱后下一交易日开盘买入，计0.2%滑点")
        st.write("第二根柱严格长于第一根：继续持有")
        st.write("第二根缩短、持平或翻绿：下一可交易日开盘退出，计0.2%滑点")
        use_cache = st.checkbox("使用逐股票缓存", value=True)
        api_pause = st.number_input(
            "每次API调用后暂停(秒)", min_value=0.0, max_value=3.0,
            value=0.12, step=0.05,
        )
        if st.button("清除本程序缓存"):
            if os.path.isdir(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            st.success("缓存已清除")

    token = st.text_input("Tushare Token", type="password")
    if not token:
        st.info("请输入Tushare Token。")
        return
    run_requested = st.button("开始第二周持有退出验证", type="primary")
    session_key = "week2_hold_exit_v1_0_zip"
    if not run_requested:
        if session_key in st.session_state:
            st.success("上一次结果仍在，可直接下载。")
            st.download_button(
                "下载1号：上一次全部结果ZIP", st.session_state[session_key],
                file_name="weekly_macd_week2_hold_exit_v1_0_all_results.zip",
                mime="application/zip", type="primary", on_click="ignore",
            )
        else:
            st.info("重点看：AB提前退出率、C1/C2误退出率、退出收益及退出后的剩余上涨空间。")
        return
    if signal_start_date >= signal_end_date:
        st.error("信号开始日期必须早于截止日期。")
        return
    if observation_end_date <= signal_end_date:
        st.error("观察截止日期必须晚于信号截止日期。")
        return

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    signal_start = signal_start_date.strftime("%Y%m%d")
    signal_end = signal_end_date.strftime("%Y%m%d")
    observation_end = observation_end_date.strftime("%Y%m%d")
    preload_start = (signal_start_date - timedelta(days=3 * 365)).strftime("%Y%m%d")
    calendar_tail = (observation_end_date + timedelta(days=7)).strftime("%Y%m%d")
    config = {
        "signal_start": signal_start, "signal_end": signal_end,
        "market_end": observation_end, "preload_start": preload_start,
        "min_price": 10.0, "min_mv": 100.0, "max_mv": 1_000_000_000.0,
        "price_tolerance_pct": 3.0, "stop_threshold_pct": 10.0,
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "sample_per_board": 0, "sample_seed": DEFAULT_SAMPLE_SEED,
        "long_cycle_min_weeks": DEFAULT_LONG_CYCLE_MIN_WEEKS,
        "material_hist_change_pct": DEFAULT_MATERIAL_HIST_CHANGE_PCT,
        "short_strength_ratio": DEFAULT_SHORT_STRENGTH_RATIO,
    }
    try:
        with st.spinner("加载交易日历、历史科技股票池和申万历史成分..."):
            open_dates = load_trade_calendar(preload_start, observation_end)
            full_calendar = load_trade_calendar(preload_start, calendar_tail)
            stock_basic = load_stock_basic()
            memberships = load_sw_tech_memberships(float(api_pause))
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return

    week_last_map = complete_week_last_dates(full_calendar)
    period_index = build_period_index(memberships)
    universe_codes = sorted(set(period_index) & set(stock_basic["ts_code"].astype(str)))
    universe = stock_basic[stock_basic["ts_code"].isin(universe_codes)].copy()
    stocks, sample_audit, population_summary = build_stratified_sample(
        stocks=universe, period_index=period_index, reference_date=signal_end,
        per_board=0, seed=DEFAULT_SAMPLE_SEED,
    )
    list_dates = stocks["list_date"].apply(lambda x: normalize_date(x, "19000101"))
    delist_dates = stocks["delist_date"].apply(lambda x: normalize_date(x, "99991231"))
    stocks_to_fetch = stocks[
        ~list_dates.gt(signal_end) & ~delist_dates.lt(preload_start)
    ].copy().reset_index(drop=True)
    st.write(
        f"历史科技池{len(universe)}只；实际读取{len(stocks_to_fetch)}只；"
        f"首红柱区间{signal_start}—{signal_end}；观察至{observation_end}。"
    )

    open_pos = {trade_date: position for position, trade_date in enumerate(open_dates)}
    all_records: list[dict[str, Any]] = []
    daily_histories: dict[str, pd.DataFrame] = {}
    reject_totals: dict[str, int] = {}
    cache_hits = data_failures = 0
    progress = st.progress(0.0, text="逐股票生成真正周线首红柱事件...")
    status = st.empty()
    for idx, stock in stocks_to_fetch.iterrows():
        code = str(stock["ts_code"])
        progress.progress((idx + 1) / len(stocks_to_fetch), text=f"{idx + 1}/{len(stocks_to_fetch)} {code}")
        status.caption(f"底层事件{len(all_records)}；缓存命中{cache_hits}；行情失败{data_failures}")
        daily, basic, cache_hit = fetch_stock_history(
            code, preload_start, observation_end, bool(use_cache), float(api_pause)
        )
        cache_hits += int(cache_hit)
        if daily.empty:
            data_failures += 1
            continue
        records, rejects, _ = analyze_stock(
            stock=stock, periods=period_index.get(code, []), daily=daily, basic=basic,
            week_last_map=week_last_map, open_dates=open_dates, open_pos=open_pos,
            config=config,
        )
        all_records.extend(records)
        if records:
            daily_histories[code] = daily.copy()
        for reason, count in rejects.items():
            reject_totals[reason] = reject_totals.get(reason, 0) + count
    progress.empty()
    status.empty()
    if not all_records:
        st.error("没有生成有效事件，请检查日期、Token权限和价格市值条件。")
        return

    events = pd.DataFrame(all_records).sort_values(["Signal_Date", "ts_code", "Event_Type"])
    with st.spinner("计算第二周持有或退出及退出后的完整路径..."):
        opportunities = build_cycle_opportunities(
            events, daily_histories, observation_end, config["sell_slippage_pct"]
        )
        results = build_week2_hold_exit_results(
            opportunities, daily_histories, open_dates, config["sell_slippage_pct"]
        )
        decision_audit, decision_matrix = build_decision_matrix(results)
        exit_summary = build_early_exit_summary(results)
        held_summary = build_held_opportunity_summary(results)
        year_summary = build_hold_exit_year_summary(results)

    complete = complete_decision_events(results)
    if complete.empty:
        st.error("没有完整可判断周期，请延长观察截止日期。")
        return
    exit_required = complete[complete["W2_Exit_Required"].map(to_bool)]
    invalid_exit = exit_required[~exit_required["W2_Exit_Executable"].map(to_bool)].copy()
    censored = results[
        results["Opportunity_Valid"].map(to_bool)
        & ~results["Cycle_Completed"].map(to_bool)
    ].copy()
    reject_frame = pd.DataFrame([
        {"剔除原因": reason, "次数": count} for reason, count in reject_totals.items()
    ]).sort_values("次数", ascending=False) if reject_totals else pd.DataFrame(
        columns=["剔除原因", "次数"]
    )
    run_summary = pd.DataFrame([{
        "程序": TITLE, "信号开始": signal_start, "信号截止": signal_end,
        "观察截止": observation_end, "首红柱事件": len(results),
        "完整可判断周期": len(complete),
        "继续持有": int(complete["W2_Decision_Action"].eq("继续持有").sum()),
        "提前退出": int(complete["W2_Decision_Action"].eq("退出").sum()),
        "退出不可执行": len(invalid_exit),
        "涉及股票": results["ts_code"].nunique(), "真实行情失败": data_failures,
        "缓存命中": cache_hits,
    }])
    metadata = pd.DataFrame([
        {"项目": "程序", "值": TITLE},
        {"项目": "买入", "值": "第一根完整红柱确认后的下一交易日开盘×1.002"},
        {"项目": "继续持有", "值": "第二根完整红柱柱值严格大于第一根；无10%扩张门槛"},
        {"项目": "提前退出", "值": "第二根柱缩短、持平或翻绿；确认后的下一可交易日开盘×0.998"},
        {"项目": "停牌处理", "值": "计划退出日无个股行情时，顺延至首次存在有效开盘价的交易日"},
        {"项目": "继续持有组终点", "值": "第一根完整绿柱确认日（含该日），仅统计机会空间"},
        {"项目": "C1C2用途", "值": "仅作事后审查标签，绝不参与第二周决策"},
        {"项目": "明确不包含", "值": "无固定止损、无最终止盈、无资金组合、无最高价卖出假设"},
        {"项目": "股票池", "值": "历史科技板块全量；信号日价≥10元、流通市值≥100亿元"},
    ])
    files = {
        "01_run_summary_week2_hold_exit_v1_0.csv": run_summary,
        "02_decision_audit_week2_hold_exit_v1_0.csv": decision_audit,
        "03_decision_matrix_week2_hold_exit_v1_0.csv": decision_matrix,
        "04_early_exit_outcome_week2_hold_exit_v1_0.csv": exit_summary,
        "05_held_opportunity_week2_hold_exit_v1_0.csv": held_summary,
        "06_year_robustness_week2_hold_exit_v1_0.csv": year_summary,
        "07_complete_decision_events_week2_hold_exit_v1_0.csv": complete,
        "08_all_events_week2_hold_exit_v1_0.csv": results,
        "09_exit_invalid_week2_hold_exit_v1_0.csv": invalid_exit,
        "10_censored_cycles_week2_hold_exit_v1_0.csv": censored,
        "11_full_tech_universe_week2_hold_exit_v1_0.csv": sample_audit,
        "12_population_week2_hold_exit_v1_0.csv": population_summary,
        "13_rejection_audit_week2_hold_exit_v1_0.csv": reject_frame,
        "14_metadata_week2_hold_exit_v1_0.csv": metadata,
    }
    result_zip = make_result_zip(files)
    st.session_state[session_key] = result_zip

    audit = decision_audit.iloc[0]
    all_exit = exit_summary[exit_summary["分组"].eq("全部提前退出")].iloc[0]
    st.success(
        f"验证完成：完整可判断{len(complete)}个；提前退出{int(audit['提前退出'])}个；"
        f"其中正确退出AB {int(audit['正确退出AB'])}个，误退出C1C2 {int(audit['误退出C1C2'])}个。"
    )
    metrics = st.columns(6)
    metrics[0].metric("AB提前退出率", f"{audit['AB提前退出率(%)']:.2f}%")
    metrics[1].metric("C1C2误退出率", f"{audit['C1C2误退出率(%)']:.2f}%")
    metrics[2].metric("退出组AB纯度", f"{audit['退出组AB纯度(%)']:.2f}%")
    metrics[3].metric("退出收益中位数", f"{all_exit['退出收益中位数(%)']:.2f}%")
    metrics[4].metric("退出盈利比例", f"{all_exit['退出盈利比例(%)']:.2f}%")
    metrics[5].metric("持有组C1C2比例", f"{audit['继续持有组C1C2比例(%)']:.2f}%")

    st.subheader("第二周决策效果")
    st.dataframe(decision_audit, use_container_width=True, hide_index=True)
    st.dataframe(decision_matrix, use_container_width=True, hide_index=True)
    st.subheader("提前退出后的实际结果与后来走势")
    st.dataframe(exit_summary, use_container_width=True, hide_index=True)
    st.subheader("第二周扩张后继续持有组的机会空间")
    st.dataframe(held_summary, use_container_width=True, hide_index=True)
    with st.expander("年度稳定性"):
        st.dataframe(year_summary, use_container_width=True, hide_index=True)

    st.subheader("下载结果")
    st.download_button(
        "下载1号：全部结果ZIP", result_zip,
        file_name="weekly_macd_week2_hold_exit_v1_0_all_results.zip",
        mime="application/zip", type="primary", on_click="ignore",
    )
    labels = [
        "2号：运行总表", "3号：决策审计", "4号：决策矩阵", "5号：提前退出结果",
        "6号：继续持有机会", "7号：年度稳健性", "8号：完整决策明细", "9号：全部事件",
        "10号：退出无效", "11号：未翻绿周期", "12号：科技股票池", "13号：板块数量",
        "14号：剔除审计", "15号：验证口径",
    ]
    columns = st.columns(4)
    for index, (filename, frame) in enumerate(files.items()):
        with columns[index % 4]:
            st.download_button(
                labels[index], csv_bytes(frame), file_name=filename,
                mime="text/csv", key=f"w2_hold_exit_{filename}", on_click="ignore",
            )
    st.warning(
        "提前退出组使用真实开盘卖价计算；继续持有组尚未设计最终止盈，只统计红柱周期内机会空间。"
        "两者不能混合解释为完整策略收益。"
    )


if __name__ == "__main__":
    week2_hold_exit_main()
