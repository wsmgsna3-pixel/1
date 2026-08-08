# -*- coding: utf-8 -*-
"""
全量科技股周线 MACD 红柱周期验证器 V4.0
========================================

研究目的（纯统计，不是选股实盘版）：
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
    streamlit run weekly_macd_hypothesis_validator_v3_1.py
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


VERSION = "V4.1-ALL-TECH-DELAYED-ENTRY"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "weekly_macd_validation_cache_v1")
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
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    weekly = build_weekly(daily, week_last_map)
    if len(weekly) < MACD_WARMUP_WEEKS:
        return [], {"周线不足": 1}

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

    return records, rejects


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
# Streamlit 页面
# -----------------------------------------------------------------------------
def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title="全量科技股周线MACD验证器 V4.1", layout="wide")
    st.title("全量科技股周线MACD红柱周期验证器 V4.1")
    st.caption(
        "历史科技股票池全部纳入，不随机抽样；信号日股价≥10元、流通市值≥100亿元。"
    )

    with st.sidebar:
        st.header("研究区间")
        SIGNAL_START_DATE = st.date_input("信号开始日期", value=date(2024, 6, 5))
        SIGNAL_END_DATE = st.date_input("信号截止日期", value=date(2026, 6, 5))
        MARKET_END_DATE = st.date_input("行情数据截止日期", value=date(2026, 8, 7))
        st.caption(
            "默认验证2024-06-05至2026-06-05两年信号；行情延长至2026-08-07，"
            "只用于计算后续八周收益和完成红柱周期，不会生成新的研究信号。"
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

        st.header("统计定义")
        PRICE_TOLERANCE = st.number_input(
            "趋势判断允许偏离周MA20(%)", min_value=0.0, max_value=15.0,
            value=3.0, step=0.5,
        )
        STOP_THRESHOLD = st.number_input(
            "失败/止损统计阈值(%)", min_value=1.0, max_value=30.0,
            value=10.0, step=1.0,
            help="用于屏障先后判断和固定止损止盈退出模拟。",
        )
        slip1, slip2 = st.columns(2)
        BUY_SLIPPAGE = slip1.number_input(
            "D1买入滑点(%)", min_value=0.0, max_value=2.0,
            value=0.20, step=0.05,
        )
        SELL_SLIPPAGE = slip2.number_input(
            "卖出滑点(%)", min_value=0.0, max_value=2.0,
            value=0.20, step=0.05,
        )

        st.header("红柱周期定义")
        LONG_CYCLE_MIN_WEEKS = st.number_input(
            "长红柱周期最少周数", min_value=3, max_value=12,
            value=DEFAULT_LONG_CYCLE_MIN_WEEKS, step=1,
            help="依据审计结果默认9周：1周=A，2—8周=B，9周及以上再区分C1/C2。",
        )
        MATERIAL_HIST_CHANGE = st.number_input(
            "有效缩短/扩张幅度(%)", min_value=1.0, max_value=50.0,
            value=DEFAULT_MATERIAL_HIST_CHANGE_PCT, step=1.0,
            help="柱体相对上周变化不足该幅度视为噪声；默认10%。",
        )
        SHORT_STRENGTH_RATIO = st.number_input(
            "短小红柱阈值（红峰/前绿峰）", min_value=0.05, max_value=2.0,
            value=DEFAULT_SHORT_STRENGTH_RATIO, step=0.05,
            help="默认0.5：红柱峰值不到前一轮绿柱绝对峰值的一半，标记为短小。",
        )

        st.header("数据与缓存")
        USE_CACHE = st.checkbox("使用逐股票缓存", value=True)
        API_PAUSE = st.number_input(
            "每次API调用后暂停(秒)", min_value=0.0, max_value=3.0,
            value=0.12, step=0.05,
        )
        if st.button("清除本验证器缓存"):
            if os.path.isdir(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            st.success("验证器专用缓存已清除")

    TS_TOKEN = st.text_input("Tushare Token", type="password")
    if not TS_TOKEN:
        st.info("请输入Tushare Token。程序将检查完整历史科技股票池，不限制股票数量。")
        return

    run_requested = st.button("开始全量科技股验证", type="primary")
    if not run_requested:
        if "v41_result_zip" in st.session_state:
            st.success("上一次验证结果仍然保留，可直接下载，无需重新运行。")
            st.download_button(
                "下载1号：上一次全部结果ZIP",
                data=st.session_state["v41_result_zip"],
                file_name="weekly_macd_all_tech_v4_1_all_results.zip",
                mime="application/zip",
                type="primary",
                on_click="ignore",
            )
            return
        with st.expander("本程序的关键统计口径"):
            st.markdown(
                """
                - **第一根红柱**：完整周MACD柱本周 `>0`，上周 `<=0`。
                - **红柱首次缩短**：同一轮红柱中第一次出现本周红柱小于上周。
                - **上升趋势**：周MA20>周MA40、周MA20四周斜率>0、价格未明显跌破周MA20。
                - **下降趋势**：周MA20<周MA40、斜率<0、价格未明显站上周MA20。
                - **回调深度**：绿柱段最低价相对绿柱开始前最多四周最高价的跌幅。
                - **实际买点**：信号周结束后的下一市场交易日开盘。
                - **退出模拟**：跳空穿越止损按开盘价；同日双触发保守按止损；计入买卖滑点，不含佣金和印花税。
                - **A/B/C1/C2**：完整周期结束后才知道，只用于验证形态与收益关系，绝不作为买入日条件。
                - **第2—5周状态**：每个检查点只使用当时已经完成的周线柱，后续收益从检查点以后单独计算。
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
    week_last_map = complete_week_last_dates(open_dates)
    st.write(
        f"完整历史科技池：{len(universe_stocks)}只；本次全量检查：{len(stocks)}只；"
        f"信号区间：{signal_start}—{signal_end}；"
        f"行情结果观察至：{market_end}；行情预热起点：{preload_start}。"
    )
    st.dataframe(style_percent_table(population_summary), use_container_width=True, hide_index=True)

    all_records: list[dict[str, Any]] = []
    reject_totals: dict[str, int] = {}
    cache_hits = 0
    data_failures = 0
    progress = st.progress(0.0, text="正在逐股票验证周线状态...")
    status = st.empty()

    for idx, stock in stocks.iterrows():
        ts_code = str(stock["ts_code"])
        progress.progress((idx + 1) / len(stocks), text=f"{idx + 1}/{len(stocks)} {ts_code}")
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
        records, rejects = analyze_stock(
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
    summaries = build_all_summaries(events)
    paired = build_paired_comparison(events)
    pair_summary = paired_summary(paired)
    strategy_report = build_strategy_report(events)
    cycle_report = build_cycle_report(events)
    cycle_strength_report = build_cycle_strength_report(events)
    checkpoint_report = build_checkpoint_report(events)
    delayed_entry_report = build_delayed_entry_report(events)

    run_hash = cache_key(
        json.dumps(config, ensure_ascii=False, sort_keys=True), sample_hash,
        "|".join(sample_audit["ts_code"].astype(str)),
    )
    event_path = os.path.join(OUTPUT_DIR, f"weekly_macd_events_{run_hash}.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"weekly_macd_summary_{run_hash}.csv")
    paired_path = os.path.join(OUTPUT_DIR, f"weekly_macd_paired_{run_hash}.csv")
    strategy_path = os.path.join(OUTPUT_DIR, f"weekly_macd_strategy_{run_hash}.csv")
    cycle_path = os.path.join(OUTPUT_DIR, f"weekly_macd_cycles_{run_hash}.csv")
    cycle_strength_path = os.path.join(OUTPUT_DIR, f"weekly_macd_cycle_strength_{run_hash}.csv")
    checkpoint_path = os.path.join(OUTPUT_DIR, f"weekly_macd_checkpoints_{run_hash}.csv")
    combined_summaries = []
    for title, frame in summaries.items():
        temp = frame.copy()
        temp.insert(0, "报表", title)
        combined_summaries.append(temp)
    if not pair_summary.empty:
        temp = pair_summary.copy()
        temp.insert(0, "报表", "严格配对比较")
        combined_summaries.append(temp)
    summary_export = pd.concat(combined_summaries, ignore_index=True, sort=False)
    atomic_csv(events, event_path)
    atomic_csv(summary_export, summary_path)
    atomic_csv(paired, paired_path)
    atomic_csv(strategy_report, strategy_path)
    atomic_csv(cycle_report, cycle_path)
    atomic_csv(cycle_strength_report, cycle_strength_path)
    atomic_csv(checkpoint_report, checkpoint_path)

    st.success(
        f"验证完成：事件{len(events)}条；完整八周可交易样本"
        f"{len(events[events['Tradable'].eq(True) & events['Has_8W_Future'].eq(True)])}条。"
    )

    first_red = events[events["Event_Type"].eq("第一根红柱")]
    full_first = first_red[first_red["Tradable"].eq(True) & first_red["Has_8W_Future"].eq(True)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("第一根红柱事件", f"{len(first_red):,}")
    c2.metric("完整八周样本", f"{len(full_first):,}")
    c3.metric("下一周继续红柱", f"{pct_mean(first_red['Next_Week_Red']):.2f}%")
    c4.metric("下一周立即翻绿", f"{pct_mean(first_red['Immediate_Green']):.2f}%")

    st.subheader("1—3、5：趋势与零轴对第一根红柱的影响")
    st.dataframe(
        style_percent_table(summaries["趋势与零轴_第一根红柱"]),
        use_container_width=True, hide_index=True,
    )

    st.subheader("4：第一根红柱买入 vs 红柱首次缩短买入")
    st.caption(
        "第一张表为全部可用事件的非配对比较；第二张表只比较同一股票、同一轮红柱周期，"
        "能够更直接回答多等几周换来了多少买价优势、又增加了多少失败率。"
    )
    st.dataframe(
        style_percent_table(summaries["买点类型对比"]),
        use_container_width=True, hide_index=True,
    )
    if not pair_summary.empty:
        st.dataframe(style_percent_table(pair_summary), use_container_width=True, hide_index=True)
    else:
        st.info("当前区间没有形成足够的完整配对周期。")

    st.subheader("6：第一根红柱前回调深度")
    st.dataframe(
        style_percent_table(summaries["回调深度_第一根红柱"]),
        use_container_width=True, hide_index=True,
    )

    st.subheader("三组实验与红柱缩短对照")
    st.caption(
        "同时给出全量科技股等权结果、三个板块各自结果。"
        "T10/T20/T30策略收益已经执行固定止损止盈；跳空止损按实际开盘价，"
        "同日双触发保守按止损处理。"
    )
    st.dataframe(
        style_percent_table(strategy_report), use_container_width=True, hide_index=True,
    )

    st.subheader("红柱完整周期A/B/C1/C2：事后验证")
    st.warning(
        "周期类型必须等到红柱转绿后才能确定，只能回答哪种形态历史利润最大，"
        "不能直接用于第一根红柱当天选股。未完成周期单独标记，避免把长周期误删。"
    )
    st.dataframe(style_percent_table(cycle_report), use_container_width=True, hide_index=True)

    st.subheader("周期类型 × 红柱相对强度")
    st.dataframe(
        style_percent_table(cycle_strength_report), use_container_width=True, hide_index=True,
    )

    st.subheader("第2—5周实时状态：无未来数据")
    st.caption(
        "仅统计在相应止盈/止损方案下截至检查周仍未退出的股票，"
        "再观察检查周之后目标先于止损的概率。"
    )
    st.dataframe(
        style_percent_table(checkpoint_report), use_container_width=True, hide_index=True,
    )

    st.subheader("第2—5周确认后再买入：独立八周验证")
    st.caption(
        "每个检查周结束后，下一市场交易日开盘并计入买入滑点；"
        "从这个新买点重新计算40个交易日，解决过去只观察原始第八周的问题。"
    )
    st.dataframe(
        style_percent_table(delayed_entry_report), use_container_width=True, hide_index=True,
    )

    st.subheader("逐周收益路径")
    full_events = events[events["Tradable"].eq(True) & events["Has_8W_Future"].eq(True)]
    weekly_rows = []
    for (event_type, trend), group in full_events.groupby(["Event_Type", "Weekly_Trend"]):
        row = {"买点": event_type, "趋势": trend, "样本数": len(group)}
        for week in range(1, HOLD_WEEKS + 1):
            row[f"W{week}平均收益(%)"] = numeric_mean(group[f"Return_W{week}_pct"])
            row[f"W{week}胜率(%)"] = float((group[f"Return_W{week}_pct"] > 0).mean() * 100.0)
        weekly_rows.append(row)
    weekly_report = pd.DataFrame(weekly_rows)
    st.dataframe(style_percent_table(weekly_report), use_container_width=True, hide_index=True)

    with st.expander("样本明细与剔除原因"):
        st.dataframe(events, use_container_width=True, hide_index=True)
        if reject_totals:
            reject_frame = pd.DataFrame(
                [{"剔除原因": reason, "次数": count} for reason, count in reject_totals.items()]
            ).sort_values("次数", ascending=False)
            st.dataframe(reject_frame, use_container_width=True, hide_index=True)
        st.write(f"个股行情获取失败：{data_failures}；缓存命中：{cache_hits}。")
        if API_ERRORS:
            st.code("\n".join(API_ERRORS[:100]))

    reject_frame = pd.DataFrame(
        [{"事件与剔除原因": reason, "次数": count} for reason, count in reject_totals.items()]
    ).sort_values("次数", ascending=False) if reject_totals else pd.DataFrame(
        columns=["事件与剔除原因", "次数"]
    )
    raw_first_red = len(first_red) + int(reject_frame.loc[
        reject_frame["事件与剔除原因"].astype(str).str.startswith("第一根红柱|"), "次数"
    ].sum())
    st.info(
        f"审计漏斗：原始第一根红柱 {raw_first_red:,} 次；"
        f"满足股价≥10元且流通市值≥100亿元 {len(first_red):,} 次；"
        f"条件删除 {raw_first_red - len(first_red):,} 次。"
    )

    st.subheader("下载结果")
    zip_buffer = io.BytesIO()
    zip_items = {
        "01_events.csv": events,
        "02_summary.csv": summary_export,
        "03_paired.csv": paired,
        "04_strategy.csv": strategy_report,
        "05_full_tech_universe.csv": sample_audit,
        "06_population.csv": population_summary,
        "07_cycles.csv": cycle_report,
        "08_cycle_strength.csv": cycle_strength_report,
        "09_checkpoints_w2_w5.csv": checkpoint_report,
        "10_rejection_audit.csv": reject_frame,
        "11_weekly_path.csv": weekly_report,
        "12_delayed_entry_w2_w5.csv": delayed_entry_report,
    }
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, frame in zip_items.items():
            archive.writestr(
                filename,
                frame.to_csv(index=False).encode("utf-8-sig"),
            )
    result_zip_bytes = zip_buffer.getvalue()
    st.session_state["v41_result_zip"] = result_zip_bytes
    st.download_button(
        "下载1号：全部结果ZIP",
        data=result_zip_bytes,
        file_name="weekly_macd_all_tech_v4_1_all_results.zip",
        mime="application/zip",
        type="primary",
        on_click="ignore",
    )
    st.warning(
        "A/B/C1/C2是事后标签；可以用于发现规律，不能直接当作实时信号。"
        "一年或两年数据仍只代表有限市场阶段。"
        "任何少于30条的细分样本都只能视为线索；最终规则应再做跨年份样本外验证。"
    )


if __name__ == "__main__":
    main()
