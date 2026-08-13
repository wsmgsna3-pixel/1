# -*- coding: utf-8 -*-
"""
科技股周线MACD同周RPS/VCP/LTR排序审计器 V2.9（单文件版）
=====================================================

本程序保持第二根完整周线红柱严格扩张候选池不变，只研究同一信号周内的排序：
RPS、VCP、RPS+VCP，以及按周分组的XGBoost LambdaMART。所有排序特征只使用
信号日及以前的数据；未来八周路径只用于结果标签和样本外判卷。
以下底层统计仍保留在程序中，用于生成事件和未来路径：
1. 上升/下降趋势中，第一根周线红柱后，下一周继续红柱的概率。
2. 两类趋势中，未来八周触及 +10%/+20%/+30%/+50%/+100% 的概率。
3. 第一根红柱买入 vs 同一红柱周期第一次红柱缩短买入。
4. 第一根红柱后一周立即翻绿的比例。
5. DIF、DEA 位于零轴上方/下方的差异。
6. 第一根红柱前回调深度对未来八周表现的影响。
7. 不随机抽样，研究期历史科技股票池中符合价格、市值条件的事件全部纳入。
8. 对基准组、上升趋势组、回调<30%组及红柱缩短组做稳健性比较。
9. 区分评分验证与实际退出；达到20%后仍继续记录完整40交易日影子路径。
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
    streamlit run weekly_macd_same_week_rank_audit_v2_9_single.py
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


VERSION = "V1.0-WEEKLY-SCORE-RANK"
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

TITLE = "科技股周线MACD硬条件＋周内评分排名验证器 V1.0"
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
    # 评分实验版新增的实时量价与风险字段。所有滚动基准均先 shift(1)，
    # 避免把当前完整周自身混入“历史常态”。
    weekly["week_return_pct"] = (weekly["close"] / weekly["close"].shift(1) - 1.0) * 100.0
    weekly["vol_median20_prev"] = weekly["vol"].shift(1).rolling(20).median()
    weekly["vol_ratio20"] = weekly["vol"] / weekly["vol_median20_prev"]
    previous_close = weekly["close"].shift(1)
    true_range = pd.concat(
        [
            weekly["high"] - weekly["low"],
            (weekly["high"] - previous_close).abs(),
            (weekly["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    weekly["atr14_pct"] = true_range.rolling(14).mean() / weekly["close"] * 100.0
    weekly_range = weekly["high"] - weekly["low"]
    weekly["close_location"] = np.where(
        weekly_range.gt(0), (weekly["close"] - weekly["low"]) / weekly_range, np.nan
    )
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
            f"{prefix}_Close": np.nan,
            f"{prefix}_DIF": np.nan,
            f"{prefix}_DEA": np.nan,
            f"{prefix}_DIF_to_Price_pct": np.nan,
            f"{prefix}_DEA_to_Price_pct": np.nan,
            f"{prefix}_MA20": np.nan,
            f"{prefix}_MA20_Slope4_pct": np.nan,
            f"{prefix}_Return_13W_pct": np.nan,
            f"{prefix}_Return_26W_pct": np.nan,
            f"{prefix}_Close_vs_MA20_pct": np.nan,
            f"{prefix}_Week_Return_pct": np.nan,
            f"{prefix}_Volume_Ratio20": np.nan,
            f"{prefix}_ATR14_pct": np.nan,
            f"{prefix}_Close_Location": np.nan,
            f"{prefix}_Return_From_Entry_pct": np.nan,
            f"{prefix}_Remaining_MFE_pct": np.nan,
            f"{prefix}_Remaining_MAE_pct": np.nan,
            f"{prefix}_Remaining_Return_pct": np.nan,
            f"{prefix}_Stop_Hit_Before": np.nan,
            f"{prefix}_Remaining_Stop_Hit": np.nan,
            f"{prefix}_Delayed_Entry_Date": "",
            f"{prefix}_Delayed_Observation_End_Date": "",
            f"{prefix}_Delayed_Entry_Price": np.nan,
            f"{prefix}_Delayed_Has_8W_Future": False,
            f"{prefix}_Delayed_MFE_8W_pct": np.nan,
            f"{prefix}_Delayed_MAE_8W_pct": np.nan,
            f"{prefix}_Delayed_Return_8W_pct": np.nan,
            f"{prefix}_Delayed_Hit_Stop_8W": np.nan,
        }
        for target in (10, 20, 30, 50, 100):
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
            f"{prefix}_Close": finite_num(checkpoint_row.get("close")),
            f"{prefix}_DIF": finite_num(checkpoint_row.get("dif")),
            f"{prefix}_DEA": finite_num(checkpoint_row.get("dea")),
            f"{prefix}_DIF_to_Price_pct": (
                finite_num(checkpoint_row.get("dif")) / finite_num(checkpoint_row.get("close")) * 100.0
                if finite_num(checkpoint_row.get("close")) > 0 else np.nan
            ),
            f"{prefix}_DEA_to_Price_pct": (
                finite_num(checkpoint_row.get("dea")) / finite_num(checkpoint_row.get("close")) * 100.0
                if finite_num(checkpoint_row.get("close")) > 0 else np.nan
            ),
            f"{prefix}_MA20": finite_num(checkpoint_row.get("ma20")),
            f"{prefix}_MA20_Slope4_pct": finite_num(checkpoint_row.get("ma20_slope4_pct")),
            f"{prefix}_Return_13W_pct": finite_num(checkpoint_row.get("return_13w_pct")),
            f"{prefix}_Return_26W_pct": finite_num(checkpoint_row.get("return_26w_pct")),
            f"{prefix}_Close_vs_MA20_pct": (
                (finite_num(checkpoint_row.get("close"))
                 / finite_num(checkpoint_row.get("ma20")) - 1.0) * 100.0
                if finite_num(checkpoint_row.get("ma20")) > 0 else np.nan
            ),
            f"{prefix}_Week_Return_pct": finite_num(checkpoint_row.get("week_return_pct")),
            f"{prefix}_Volume_Ratio20": finite_num(checkpoint_row.get("vol_ratio20")),
            f"{prefix}_ATR14_pct": finite_num(checkpoint_row.get("atr14_pct")),
            f"{prefix}_Close_Location": finite_num(checkpoint_row.get("close_location")),
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
                    f"{prefix}_Delayed_Observation_End_Date": delayed_horizon_date,
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
                for target in (10, 20, 30, 50, 100):
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

        for target in (10, 20, 30, 50, 100):
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
        "W1_Week_Return_pct": float(row["week_return_pct"]) if pd.notna(row["week_return_pct"]) else np.nan,
        "W1_Volume_Ratio20": float(row["vol_ratio20"]) if pd.notna(row["vol_ratio20"]) else np.nan,
        "W1_ATR14_pct": float(row["atr14_pct"]) if pd.notna(row["atr14_pct"]) else np.nan,
        "W1_Close_Location": float(row["close_location"]) if pd.notna(row["close_location"]) else np.nan,
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
        "W1_Week_Return_pct", "W1_Volume_Ratio20", "W1_ATR14_pct", "W1_Close_Location",
        "Raw_Close", "Circ_MV_Billion", "Turnover_Rate", "Tradable", "Untradable_Reason",
        "Entry_Date", "Entry_Price", "Cycle_Completed", "Cycle_Censored", "Cycle_Type",
        "Cycle_Strength_Class", "Red_Cycle_Weeks", "Last_Red_Date", "First_Green_Date",
        "Red_Hist_Sequence", "Peak_Red_Hist", "Peak_Red_Week", "Red_Hist_Area",
        "PreGreen_Abs_Peak_Hist", "Peak_Red_to_PreGreen_Peak_Ratio",
        "First_Material_Shrink_Week", "Material_Shrink_Count", "ReExpansion_Count",
        "CP_W2_Date", "CP_W2_State", "CP_W2_Hist", "CP_W2_Hist_vs_W1_pct",
        "CP_W2_Weak_Candidate", "CP_W2_Close", "CP_W2_MA20",
        "CP_W2_DIF", "CP_W2_DEA", "CP_W2_DIF_to_Price_pct", "CP_W2_DEA_to_Price_pct",
        "CP_W2_MA20_Slope4_pct", "CP_W2_Return_13W_pct", "CP_W2_Return_26W_pct",
        "CP_W2_Close_vs_MA20_pct", "CP_W2_Week_Return_pct",
        "CP_W2_Volume_Ratio20", "CP_W2_ATR14_pct", "CP_W2_Close_Location",
        "CP_W2_Return_From_Entry_pct", "CP_W2_Delayed_Entry_Date",
        "CP_W2_Delayed_Entry_Price", "CP_W2_Delayed_Has_8W_Future",
        "CP_W2_Delayed_Observation_End_Date",
        "CP_W2_Delayed_MFE_8W_pct", "CP_W2_Delayed_MAE_8W_pct",
        "CP_W2_Delayed_Return_8W_pct", "CP_W2_Delayed_Hit_Stop_8W",
        "CP_W2_Delayed_Hit_10_8W", "CP_W2_Delayed_Hit_20_8W",
        "CP_W2_Delayed_Hit_30_8W", "CP_W2_Delayed_Hit_50_8W",
        "CP_W2_Delayed_Hit_100_8W", "CP_W2_Delayed_First_10_vs_Stop",
        "CP_W2_Delayed_First_20_vs_Stop", "CP_W2_Delayed_First_30_vs_Stop",
        "CP_W2_Delayed_First_50_vs_Stop", "CP_W2_Delayed_First_100_vs_Stop",
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
# Streamlit 页面（旧基线，V2.5不调用）
# -----------------------------------------------------------------------------
def v10_main_legacy() -> None:
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


FROZEN_REFERENCE_RATES = {
    "全部首红柱": 58.70253164556962,
    "DEA/股价≤-8%": 92.17391304347827,
    "MA20四周斜率≤-8%": 90.19607843137256,
    "前26周收益≤-30%": 84.72222222222221,
    "下降趋势且零轴下": 71.29032258064515,
    "DEA/股价≤-5%且MA20斜率≤-5%": 83.91608391608392,
    "首红柱在零轴上": 31.818181818181817,
    "首红柱为上升趋势": 28.225806451612907,
    "第二根柱严格扩张": 65.28776978417267,
}


def oos_half_year(value: Any) -> str:
    text_value = normalize_date(value)
    if len(text_value) != 8:
        return "未知"
    return f"{text_value[:4]}{'H1' if text_value[4:6] <= '06' else 'H2'}"


def add_oos_research_features(opportunities: pd.DataFrame) -> pd.DataFrame:
    frame = opportunities.copy()
    for column in [
        "Weekly_Close", "DEA", "DIF", "W_MA20", "W_MA40",
        "W_MA20_Slope4_pct", "Pre_13W_Return_pct", "Pre_26W_Return_pct",
        "Pullback_Depth_pct", "CP_W2_Hist", "Hist",
        "CP_W2_Hist_vs_W1_pct", "Red_Cycle_Weeks",
    ]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["DEA_to_Price_pct"] = frame["DEA"] / frame["Weekly_Close"] * 100.0
    frame["DIF_to_Price_pct"] = frame["DIF"] / frame["Weekly_Close"] * 100.0
    frame["Close_vs_MA20_pct"] = (frame["Weekly_Close"] / frame["W_MA20"] - 1.0) * 100.0
    frame["Close_vs_MA40_pct"] = (frame["Weekly_Close"] / frame["W_MA40"] - 1.0) * 100.0
    frame["MA20_vs_MA40_pct"] = (frame["W_MA20"] / frame["W_MA40"] - 1.0) * 100.0
    frame["W2_Observed"] = (
        frame.get("CP_W2_Date", pd.Series("", index=frame.index)).fillna("").astype(str).ne("")
        & frame["CP_W2_Hist"].notna()
    )
    frame["W2_Exact_Expansion"] = (
        frame["W2_Observed"]
        & frame["Hist"].gt(0)
        & frame["CP_W2_Hist"].gt(frame["Hist"])
    )
    frame["Signal_Half_Year"] = frame["Signal_Date"].map(oos_half_year)

    research_family: list[str] = []
    research_status: list[str] = []
    for _, row in frame.iterrows():
        completed = to_bool(row.get("Cycle_Completed"))
        weeks = finite_num(row.get("Red_Cycle_Weeks"))
        if completed:
            family = cycle_family(row.get("Cycle_Type"))
            if family == "C1C2_长周期":
                research_family.append("C1C2_长周期")
                research_status.append("完整C1C2")
            elif family == "AB_弱或短周期":
                research_family.append("AB_弱或短周期")
                research_status.append("完整AB")
            else:
                research_family.append("未决")
                research_status.append("完整但无法分类")
        elif math.isfinite(weeks) and weeks >= DEFAULT_LONG_CYCLE_MIN_WEEKS:
            research_family.append("C1C2_长周期")
            research_status.append("进行中但已满9周_确认C类")
        else:
            research_family.append("未决")
            research_status.append("进行中不足9周_未决")
    frame["Research_Family"] = research_family
    frame["Research_Status"] = research_status

    high = frame["DEA_to_Price_pct"].le(-8.0)
    low = (
        frame["Zero_Axis"].astype(str).eq("DIF与DEA均在零轴上")
        | frame["Weekly_Trend"].astype(str).eq("上升趋势")
    )
    frame["Exploratory_Tier"] = np.select(
        [high, ~high & low],
        ["高概率C候选_DEA深于-8%", "低概率C候选_零轴上或上升趋势"],
        default="中间未定区",
    )
    return frame


def frozen_oos_rule_masks(frame: pd.DataFrame) -> list[tuple[str, pd.Series, str]]:
    return [
        ("全部首红柱", pd.Series(True, index=frame.index), "基准"),
        ("DEA/股价≤-8%", frame["DEA_to_Price_pct"].le(-8.0), "预期C1C2显著更高"),
        ("MA20四周斜率≤-8%", frame["W_MA20_Slope4_pct"].le(-8.0), "预期C1C2显著更高"),
        ("前26周收益≤-30%", frame["Pre_26W_Return_pct"].le(-30.0), "预期C1C2显著更高"),
        (
            "下降趋势且零轴下",
            frame["Weekly_Trend"].astype(str).eq("下降趋势")
            & frame["Zero_Axis"].astype(str).eq("DIF与DEA均在零轴下"),
            "预期C1C2更高",
        ),
        (
            "DEA/股价≤-5%且MA20斜率≤-5%",
            frame["DEA_to_Price_pct"].le(-5.0)
            & frame["W_MA20_Slope4_pct"].le(-5.0),
            "预期C1C2显著更高",
        ),
        (
            "首红柱在零轴上",
            frame["Zero_Axis"].astype(str).eq("DIF与DEA均在零轴上"),
            "预期AB显著更多",
        ),
        (
            "首红柱为上升趋势",
            frame["Weekly_Trend"].astype(str).eq("上升趋势"),
            "预期AB显著更多",
        ),
        ("第二根柱严格扩张", frame["W2_Exact_Expansion"].map(to_bool), "预期C1C2更高"),
    ]


def oos_rule_summary(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["Opportunity_Valid"].map(to_bool)].copy()
    resolved_all = valid[valid["Research_Family"].ne("未决")]
    total_c = int(resolved_all["Research_Family"].eq("C1C2_长周期").sum())
    total_ab = int(resolved_all["Research_Family"].eq("AB_弱或短周期").sum())
    base_rate = pct_mean(resolved_all["Research_Family"].eq("C1C2_长周期"))
    rows: list[dict[str, Any]] = []
    for name, mask, expectation in frozen_oos_rule_masks(valid):
        selected = valid[mask].copy()
        resolved = selected[selected["Research_Family"].ne("未决")]
        complete = resolved[resolved["Cycle_Completed"].map(to_bool)]
        c_mask = resolved["Research_Family"].eq("C1C2_长周期")
        ab_mask = resolved["Research_Family"].eq("AB_弱或短周期")
        new_rate = pct_mean(c_mask)
        reference_rate = FROZEN_REFERENCE_RATES.get(name, np.nan)
        if name == "全部首红柱":
            repeated = "基准"
        elif "AB" in expectation:
            repeated = "是" if math.isfinite(new_rate) and new_rate < base_rate else "否"
        else:
            repeated = "是" if math.isfinite(new_rate) and new_rate > base_rate else "否"
        row = {
            "冻结条件": name,
            "旧样本预期": expectation,
            "新样本信号数": int(len(selected)),
            "新样本已决数": int(len(resolved)),
            "新样本未决数": int(selected["Research_Family"].eq("未决").sum()),
            "新样本未决比例(%)": pct_mean(selected["Research_Family"].eq("未决")),
            "新样本C1C2数": int(c_mask.sum()),
            "新样本AB数": int(ab_mask.sum()),
            "新样本C1C2比例(%)": new_rate,
            "旧样本C1C2比例(%)": reference_rate,
            "新旧差异(百分点)": (
                new_rate - reference_rate
                if math.isfinite(new_rate) and math.isfinite(reference_rate) else np.nan
            ),
            "相对新样本总体提升(百分点)": (
                new_rate - base_rate if math.isfinite(new_rate) else np.nan
            ),
            "方向是否重复": repeated,
            "覆盖全部C1C2(%)": c_mask.sum() / total_c * 100.0 if total_c else np.nan,
            "纳入全部AB(%)": ab_mask.sum() / total_ab * 100.0 if total_ab else np.nan,
            "利润完整周期数": int(len(complete)),
            "最高利润中位数(%)": pd.to_numeric(complete["Peak_MFE_pct"], errors="coerce").median(),
            "最大浮亏中位数(%)": pd.to_numeric(complete["Path_MAE_pct"], errors="coerce").median(),
        }
        for target in (10, 20, 30):
            row[f"曾达到{target}%(%)"] = pct_mean(
                complete[f"Reached_{target}_pct"].map(to_bool)
            ) if len(complete) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def oos_period_rule_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = frame[frame["Opportunity_Valid"].map(to_bool)].copy()
    for period, period_frame in valid.groupby("Signal_Half_Year", sort=True):
        base_resolved = period_frame[period_frame["Research_Family"].ne("未决")]
        base_rate = pct_mean(base_resolved["Research_Family"].eq("C1C2_长周期"))
        for name, mask, _ in frozen_oos_rule_masks(period_frame):
            selected = period_frame[mask]
            resolved = selected[selected["Research_Family"].ne("未决")]
            rate = pct_mean(resolved["Research_Family"].eq("C1C2_长周期"))
            rows.append({
                "信号半年": period,
                "冻结条件": name,
                "信号数": int(len(selected)),
                "已决数": int(len(resolved)),
                "未决数": int(selected["Research_Family"].eq("未决").sum()),
                "C1C2比例(%)": rate,
                "同期总体C1C2比例(%)": base_rate,
                "相对同期总体差异(百分点)": (
                    rate - base_rate if math.isfinite(rate) and math.isfinite(base_rate) else np.nan
                ),
            })
    return pd.DataFrame(rows)


def rank_auc(values: pd.Series, target: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.notna() & target.notna()
    numeric = numeric[mask]
    binary = target[mask].astype(int)
    n1 = int(binary.sum())
    n0 = int(len(binary) - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = numeric.rank(method="average")
    rank_sum = float(ranks[binary.eq(1)].sum())
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def oos_numeric_feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    resolved = frame[
        frame["Opportunity_Valid"].map(to_bool)
        & frame["Research_Family"].ne("未决")
    ].copy()
    target = resolved["Research_Family"].eq("C1C2_长周期").astype(int)
    definitions = [
        ("DEA/股价(%)", "DEA_to_Price_pct", "越低越偏C1C2"),
        ("DIF/股价(%)", "DIF_to_Price_pct", "越低越偏C1C2"),
        ("MA20四周斜率(%)", "W_MA20_Slope4_pct", "越低越偏C1C2"),
        ("MA20相对MA40(%)", "MA20_vs_MA40_pct", "越低越偏C1C2"),
        ("收盘相对MA40(%)", "Close_vs_MA40_pct", "越低越偏C1C2"),
        ("前26周收益(%)", "Pre_26W_Return_pct", "越低越偏C1C2"),
        ("前13周收益(%)", "Pre_13W_Return_pct", "越低越偏C1C2"),
        ("回调深度(%)", "Pullback_Depth_pct", "越高越偏C1C2"),
        ("第二根相对第一根增幅(%)", "CP_W2_Hist_vs_W1_pct", "越高越偏C1C2"),
        ("第一根相对前绿柱峰值", "First_Red_to_PreGreen_Peak_Ratio", "旧样本几乎无效"),
        ("流通市值(亿元)", "Circ_MV_Billion", "旧样本较弱"),
        ("换手率", "Turnover_Rate", "旧样本较弱"),
    ]
    rows = []
    for label, column, expected in definitions:
        values = pd.to_numeric(resolved[column], errors="coerce")
        auc = rank_auc(values, target)
        observed_direction = "越高越偏C1C2" if auc >= 0.5 else "越低越偏C1C2"
        if expected.startswith("越高"):
            repeated = "是" if auc >= 0.5 else "否"
        elif expected.startswith("越低"):
            repeated = "是" if auc < 0.5 else "否"
        else:
            repeated = "不设方向"
        rows.append({
            "实时特征": label,
            "旧样本发现": expected,
            "AB中位数": values[target.eq(0)].median(),
            "C1C2中位数": values[target.eq(1)].median(),
            "原始AUC_高值偏C": auc,
            "分离度AUC": max(auc, 1.0 - auc) if math.isfinite(auc) else np.nan,
            "新样本方向": observed_direction,
            "方向是否重复": repeated,
            "有效样本数": int(values.notna().sum()),
        })
    return pd.DataFrame(rows).sort_values("分离度AUC", ascending=False)


def oos_categorical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    resolved = frame[
        frame["Opportunity_Valid"].map(to_bool)
        & frame["Research_Family"].ne("未决")
    ].copy()
    rows: list[dict[str, Any]] = []
    for column, label in [
        ("Weekly_Trend", "周线趋势"),
        ("Zero_Axis", "零轴位置"),
        ("Sample_Board", "上市板块"),
        ("First_Red_Strength_Group", "第一根红柱强度"),
        ("CP_W2_State", "第二周状态"),
        ("Exploratory_Tier", "探索性三层"),
    ]:
        for value, group in resolved.groupby(column, dropna=False, sort=False):
            rows.append({
                "特征": label,
                "分组": value,
                "已决数": int(len(group)),
                "C1C2数": int(group["Research_Family"].eq("C1C2_长周期").sum()),
                "AB数": int(group["Research_Family"].eq("AB_弱或短周期").sum()),
                "C1C2比例(%)": pct_mean(group["Research_Family"].eq("C1C2_长周期")),
                "最高利润中位数_仅完整周期(%)": pd.to_numeric(
                    group[group["Cycle_Completed"].map(to_bool)]["Peak_MFE_pct"], errors="coerce"
                ).median(),
            })
    return pd.DataFrame(rows)


def oos_status_summary(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["Opportunity_Valid"].map(to_bool)].copy()
    return valid.groupby("Research_Status", dropna=False).agg(
        事件数=("Cycle_ID", "size"),
        涉及股票=("ts_code", "nunique"),
        红柱周数中位数=("Red_Cycle_Weeks", "median"),
    ).reset_index()


def c1c2_ab_oos_main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "冻结旧样本阈值，在2024-06-05至2026-06-05的新样本上验证。"
        "不重新调参；未翻绿且不足9周的周期一律标为未决。"
    )
    with st.sidebar:
        st.header("样本外信号区间")
        signal_start_date = st.date_input("信号开始日期", value=date(2024, 6, 5))
        signal_end_date = st.date_input("信号截止日期", value=date(2026, 6, 5))
        observation_end_date = st.date_input(
            "可用行情观察截止", value=date.today(), max_value=date.today()
        )
        st.header("固定口径")
        st.write("历史科技板块全量，不抽样")
        st.write("信号日股价≥10元、流通市值≥100亿元")
        st.write("周MACD(12,26,9)，仅使用完整周")
        st.write("C类长周期门槛固定为9周")
        st.write("所有特征阈值冻结，禁止新样本调参")
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
    session_key = "c1c2_ab_oos_v1_0_zip"
    run_requested = st.button("开始样本外特征验证", type="primary")
    if not run_requested:
        if session_key in st.session_state:
            st.success("上一次结果仍在，可直接下载。")
            st.download_button(
                "下载1号：上一次全部结果ZIP", st.session_state[session_key],
                file_name="weekly_macd_c1c2_ab_oos_v1_0_all_results.zip",
                mime="application/zip", type="primary", on_click="ignore",
            )
        else:
            st.info("重点看冻结条件是否在新样本中保持同方向，以及2024H2—2026H1各阶段是否稳定。")
        return
    if signal_start_date >= signal_end_date:
        st.error("信号开始日期必须早于截止日期。")
        return
    if signal_end_date > observation_end_date:
        st.error("信号截止日期不能晚于可用行情观察截止。")
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
        f"信号区间{signal_start}—{signal_end}；行情观察至{observation_end}。"
    )

    open_pos = {trade_date: position for position, trade_date in enumerate(open_dates)}
    all_records: list[dict[str, Any]] = []
    daily_histories: dict[str, pd.DataFrame] = {}
    reject_totals: dict[str, int] = {}
    cache_hits = data_failures = 0
    progress = st.progress(0.0, text="逐股票生成首红柱事件...")
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
    with st.spinner("计算周期标签、冻结条件与样本外稳定性..."):
        opportunities = build_cycle_opportunities(
            events, daily_histories, observation_end, config["sell_slippage_pct"]
        )
        research = add_oos_research_features(opportunities)
        rule_summary = oos_rule_summary(research)
        period_summary = oos_period_rule_summary(research)
        numeric_summary = oos_numeric_feature_summary(research)
        categorical_summary = oos_categorical_summary(research)
        status_summary = oos_status_summary(research)

    valid = research[research["Opportunity_Valid"].map(to_bool)].copy()
    resolved = valid[valid["Research_Family"].ne("未决")].copy()
    unresolved = valid[valid["Research_Family"].eq("未决")].copy()
    ongoing_confirmed = valid[
        valid["Research_Status"].eq("进行中但已满9周_确认C类")
    ].copy()
    reject_frame = pd.DataFrame([
        {"剔除原因": reason, "次数": count} for reason, count in reject_totals.items()
    ]).sort_values("次数", ascending=False) if reject_totals else pd.DataFrame(
        columns=["剔除原因", "次数"]
    )
    base_row = rule_summary[rule_summary["冻结条件"].eq("全部首红柱")].iloc[0]
    direction_repeated = int(rule_summary["方向是否重复"].eq("是").sum())
    tested_direction_rules = int(rule_summary["方向是否重复"].isin(["是", "否"]).sum())
    run_summary = pd.DataFrame([{
        "程序": TITLE,
        "信号开始": signal_start,
        "信号截止": signal_end,
        "行情观察截止": observation_end,
        "首红柱事件": len(research),
        "可买入有效事件": len(valid),
        "已决事件": len(resolved),
        "完整周期": int(valid["Cycle_Completed"].map(to_bool).sum()),
        "进行中满9周确认C": len(ongoing_confirmed),
        "进行中不足9周未决": len(unresolved),
        "已决C1C2比例(%)": base_row["新样本C1C2比例(%)"],
        "冻结方向重复数": direction_repeated,
        "冻结方向检验数": tested_direction_rules,
        "涉及股票": research["ts_code"].nunique(),
        "真实行情失败": data_failures,
        "缓存命中": cache_hits,
    }])
    metadata = pd.DataFrame([
        {"项目": "程序", "值": TITLE},
        {"项目": "用途", "值": "验证旧样本发现是否在2024-06-05至2026-06-05的新时期重复"},
        {"项目": "禁止事项", "值": "不在新样本重新选择阈值，不根据结果回改条件"},
        {"项目": "A/B", "值": "完整红柱周期少于9周；1周为A，2至8周为B"},
        {"项目": "C1/C2", "值": "完整周期不少于9周；进行中已经满9周也可安全确认属于C类"},
        {"项目": "未决", "值": "截至行情末日仍为红柱且不足9周；绝不强行归入A/B"},
        {"项目": "实时特征", "值": "只使用首红柱与第二周当时可见信息"},
        {"项目": "利润指标", "值": "仅完整周期用于最高利润与最大浮亏比较；最高价不是卖出价"},
        {"项目": "股票池", "值": "历史科技板块全量；信号日价≥10元、流通市值≥100亿元"},
    ])
    files = {
        "01_run_summary_c1c2_ab_oos_v1_0.csv": run_summary,
        "02_frozen_rule_validation_c1c2_ab_oos_v1_0.csv": rule_summary,
        "03_half_year_robustness_c1c2_ab_oos_v1_0.csv": period_summary,
        "04_numeric_feature_separation_c1c2_ab_oos_v1_0.csv": numeric_summary,
        "05_categorical_feature_separation_c1c2_ab_oos_v1_0.csv": categorical_summary,
        "06_cycle_resolution_status_c1c2_ab_oos_v1_0.csv": status_summary,
        "07_resolved_event_detail_c1c2_ab_oos_v1_0.csv": resolved,
        "08_unresolved_event_detail_c1c2_ab_oos_v1_0.csv": unresolved,
        "09_ongoing_confirmed_c_detail_c1c2_ab_oos_v1_0.csv": ongoing_confirmed,
        "10_all_event_detail_c1c2_ab_oos_v1_0.csv": research,
        "11_full_tech_universe_c1c2_ab_oos_v1_0.csv": sample_audit,
        "12_population_c1c2_ab_oos_v1_0.csv": population_summary,
        "13_rejection_audit_c1c2_ab_oos_v1_0.csv": reject_frame,
        "14_metadata_c1c2_ab_oos_v1_0.csv": metadata,
    }
    result_zip = make_result_zip(files)
    st.session_state[session_key] = result_zip

    high_row = rule_summary[rule_summary["冻结条件"].eq("DEA/股价≤-8%")].iloc[0]
    low_row = rule_summary[rule_summary["冻结条件"].eq("首红柱在零轴上")].iloc[0]
    st.success(
        f"验证完成：有效事件{len(valid)}个，已决{len(resolved)}个，未决{len(unresolved)}个；"
        f"冻结方向{direction_repeated}/{tested_direction_rules}项重复。"
    )
    metrics = st.columns(6)
    metrics[0].metric("已决总体C1C2", f"{base_row['新样本C1C2比例(%)']:.2f}%")
    metrics[1].metric("DEA深于-8%的C1C2", f"{high_row['新样本C1C2比例(%)']:.2f}%")
    metrics[2].metric("零轴上C1C2", f"{low_row['新样本C1C2比例(%)']:.2f}%")
    metrics[3].metric("已决事件", f"{len(resolved)}")
    metrics[4].metric("未决事件", f"{len(unresolved)}")
    metrics[5].metric("方向重复", f"{direction_repeated}/{tested_direction_rules}")

    st.subheader("冻结条件样本外验证")
    st.dataframe(rule_summary, use_container_width=True, hide_index=True)
    st.subheader("2024H2—2026H1分阶段稳定性")
    st.dataframe(period_summary, use_container_width=True, hide_index=True)
    with st.expander("数值特征、分类特征与周期解决状态"):
        st.dataframe(numeric_summary, use_container_width=True, hide_index=True)
        st.dataframe(categorical_summary, use_container_width=True, hide_index=True)
        st.dataframe(status_summary, use_container_width=True, hide_index=True)

    st.subheader("下载结果")
    st.download_button(
        "下载1号：全部结果ZIP", result_zip,
        file_name="weekly_macd_c1c2_ab_oos_v1_0_all_results.zip",
        mime="application/zip", type="primary", on_click="ignore",
    )
    labels = [
        "2号：运行总表", "3号：冻结条件验证", "4号：半年稳定性", "5号：数值特征",
        "6号：分类特征", "7号：周期解决状态", "8号：已决明细", "9号：未决明细",
        "10号：进行中已确认C", "11号：全部事件", "12号：科技股票池", "13号：板块数量",
        "14号：剔除审计", "15号：验证口径",
    ]
    columns = st.columns(4)
    for index, (filename, frame) in enumerate(files.items()):
        with columns[index % 4]:
            st.download_button(
                labels[index], csv_bytes(frame), file_name=filename,
                mime="text/csv", key=f"c1c2_ab_oos_{filename}", on_click="ignore",
            )
    st.warning(
        "接近2026-06-05的信号可能尚未满9周，因此未决比例必须与命中率一起看。"
        "若某条件未决比例过高，不能提前宣布规律成立。"
    )


SCORE_WEIGHTS = {
    "第二根柱扩张幅度": 40.0,
    "DEA相对股价深度": 25.0,
    "MA20四周斜率": 20.0,
    "前26周收益": 10.0,
    "前期回调深度": 5.0,
}
TOLERANT_NONEXPANSION_PENALTY = 25.0
SCORE_POOL_STRICT = "严格池_第二周严格扩张"
SCORE_POOL_TOLERANT = "宽容池_第二周仍红"
SCORE_SCOPES = ("全部候选", "Top1", "Top3", "Top5")


def weekly_score_percentile(
    frame: pd.DataFrame,
    column: str,
    higher_is_better: bool,
) -> pd.Series:
    """在同一个第二周确认日内计算0—1百分位；缺失值按中性0.5处理。"""
    values = pd.to_numeric(frame[column], errors="coerce")
    oriented = values if higher_is_better else -values
    result = oriented.groupby(frame["Selection_Date"]).rank(
        method="average", pct=True, ascending=True,
    )
    return result.fillna(0.5)


def add_weekly_score_features(opportunities: pd.DataFrame) -> pd.DataFrame:
    """只用第二周确认时已知数据计算固定权重周内评分。"""
    frame = add_oos_research_features(opportunities)
    frame["Selection_Date"] = frame.get(
        "CP_W2_Date", pd.Series("", index=frame.index)
    ).map(normalize_date)
    frame["Selection_Half_Year"] = frame["Selection_Date"].map(oos_half_year)
    valid = frame["Opportunity_Valid"].map(to_bool)
    w2_observed = frame["W2_Observed"].map(to_bool)
    w2_red = pd.to_numeric(frame["CP_W2_Hist"], errors="coerce").gt(0)
    has_date = frame["Selection_Date"].ne("")
    frame["Score_Eligible_Tolerant"] = valid & w2_observed & w2_red & has_date
    frame["Score_Eligible_Strict"] = (
        frame["Score_Eligible_Tolerant"] & frame["W2_Exact_Expansion"].map(to_bool)
    )

    eligible = frame[frame["Score_Eligible_Tolerant"]].copy()
    component_specs = [
        ("ScorePct_W2_Expansion", "CP_W2_Hist_vs_W1_pct", True, 40.0),
        ("ScorePct_DEA_Depth", "DEA_to_Price_pct", False, 25.0),
        ("ScorePct_MA20_Slope", "W_MA20_Slope4_pct", False, 20.0),
        ("ScorePct_Pre26_Return", "Pre_26W_Return_pct", False, 10.0),
        ("ScorePct_Pullback_Depth", "Pullback_Depth_pct", True, 5.0),
    ]
    component_columns: list[str] = []
    for percentile_column, source_column, higher_is_better, weight in component_specs:
        eligible[percentile_column] = weekly_score_percentile(
            eligible, source_column, higher_is_better
        )
        points_column = percentile_column.replace("ScorePct_", "ScorePoints_")
        eligible[points_column] = eligible[percentile_column] * weight
        component_columns.append(points_column)
    eligible["Score_Base_100"] = eligible[component_columns].sum(axis=1)
    eligible["Score_NonExpansion_Penalty"] = np.where(
        eligible["W2_Exact_Expansion"].map(to_bool),
        0.0,
        TOLERANT_NONEXPANSION_PENALTY,
    )
    eligible["Score_Strict_Final"] = eligible["Score_Base_100"]
    eligible["Score_Tolerant_Final"] = (
        eligible["Score_Base_100"] - eligible["Score_NonExpansion_Penalty"]
    )

    new_columns = [
        "ScorePct_W2_Expansion", "ScorePct_DEA_Depth", "ScorePct_MA20_Slope",
        "ScorePct_Pre26_Return", "ScorePct_Pullback_Depth",
        "ScorePoints_W2_Expansion", "ScorePoints_DEA_Depth",
        "ScorePoints_MA20_Slope", "ScorePoints_Pre26_Return",
        "ScorePoints_Pullback_Depth", "Score_Base_100",
        "Score_NonExpansion_Penalty", "Score_Strict_Final", "Score_Tolerant_Final",
    ]
    for column in new_columns:
        frame[column] = np.nan
        frame.loc[eligible.index, column] = eligible[column]
    return frame


def build_score_pool(
    scored: pd.DataFrame,
    pool_name: str,
    eligible_column: str,
    score_column: str,
) -> pd.DataFrame:
    pool = scored[scored[eligible_column].map(to_bool)].copy()
    if pool.empty:
        return pool
    pool["Candidate_Pool"] = pool_name
    pool["Final_Score"] = pd.to_numeric(pool[score_column], errors="coerce")
    pool["_Tie_W2"] = pd.to_numeric(pool["CP_W2_Hist_vs_W1_pct"], errors="coerce")
    pool["_Tie_DEA"] = pd.to_numeric(pool["DEA_to_Price_pct"], errors="coerce")
    pool["_Tie_Slope"] = pd.to_numeric(pool["W_MA20_Slope4_pct"], errors="coerce")
    pool = pool.sort_values(
        ["Selection_Date", "Final_Score", "_Tie_W2", "_Tie_DEA", "_Tie_Slope", "ts_code"],
        ascending=[True, False, False, True, True, True],
        na_position="last",
    ).copy()
    pool["Weekly_Rank"] = pool.groupby("Selection_Date").cumcount() + 1
    pool["Selected_Top1"] = pool["Weekly_Rank"].le(1)
    pool["Selected_Top3"] = pool["Weekly_Rank"].le(3)
    pool["Selected_Top5"] = pool["Weekly_Rank"].le(5)
    pool["Rank_Cohort"] = np.select(
        [
            pool["Weekly_Rank"].eq(1),
            pool["Weekly_Rank"].between(2, 3),
            pool["Weekly_Rank"].between(4, 5),
        ],
        ["第1名", "第2—3名", "第4—5名"],
        default="第6名以后",
    )
    return pool.drop(columns=["_Tie_W2", "_Tie_DEA", "_Tie_Slope"])


def score_scope_mask(pool: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "全部候选":
        return pd.Series(True, index=pool.index)
    return pool[f"Selected_{scope}"].map(to_bool)


def score_reference_weeks(scored: pd.DataFrame) -> pd.PeriodIndex:
    dates = pd.to_datetime(
        scored.loc[
            scored["Opportunity_Valid"].map(to_bool)
            & scored["W2_Observed"].map(to_bool)
            & scored["Selection_Date"].ne(""),
            "Selection_Date",
        ],
        format="%Y%m%d", errors="coerce",
    ).dropna()
    if dates.empty:
        return pd.PeriodIndex([], freq="W-SUN")
    periods = dates.dt.to_period("W-SUN")
    return pd.period_range(periods.min(), periods.max(), freq="W-SUN")


def longest_blank_run(counts: pd.Series) -> tuple[int, str, str]:
    best_length = current_length = 0
    best_start = best_end = current_start = None
    for period, value in counts.items():
        if int(value) == 0:
            if current_length == 0:
                current_start = period
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
                best_end = period
        else:
            current_length = 0
            current_start = None
    def label(period: Any) -> str:
        if period is None:
            return ""
        return f"{period.start_time:%Y-%m-%d}—{period.end_time:%Y-%m-%d}"
    return best_length, label(best_start), label(best_end)


def score_strategy_summary(
    pools: dict[str, pd.DataFrame],
    reference_weeks: pd.PeriodIndex,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pool_name, pool in pools.items():
        base_resolved = pool[pool["Research_Family"].ne("未决")]
        base_c = int(base_resolved["Research_Family"].eq("C1C2_长周期").sum())
        base_ab = int(base_resolved["Research_Family"].eq("AB_弱或短周期").sum())
        base_rate = pct_mean(base_resolved["Research_Family"].eq("C1C2_长周期"))
        for scope in SCORE_SCOPES:
            selected = pool[score_scope_mask(pool, scope)].copy()
            resolved = selected[selected["Research_Family"].ne("未决")]
            complete = resolved[resolved["Cycle_Completed"].map(to_bool)]
            c_count = int(resolved["Research_Family"].eq("C1C2_长周期").sum())
            ab_count = int(resolved["Research_Family"].eq("AB_弱或短周期").sum())
            c_rate = pct_mean(resolved["Research_Family"].eq("C1C2_长周期"))
            periods = pd.to_datetime(
                selected["Selection_Date"], format="%Y%m%d", errors="coerce"
            ).dt.to_period("W-SUN")
            counts = periods.value_counts().reindex(reference_weeks, fill_value=0).sort_index()
            nonzero = counts[counts.gt(0)]
            blank_length, blank_start, blank_end = longest_blank_run(counts)
            maximum = int(counts.max()) if len(counts) else 0
            max_weeks = [
                f"{period.start_time:%Y-%m-%d}—{period.end_time:%Y-%m-%d}"
                for period in counts[counts.eq(maximum)].index
            ] if maximum else []
            row = {
                "候选池": pool_name,
                "选择范围": scope,
                "入选事件": int(len(selected)),
                "不同股票": int(selected["ts_code"].nunique()),
                "已决事件": int(len(resolved)),
                "未决事件": int(selected["Research_Family"].eq("未决").sum()),
                "C1C2数": c_count,
                "AB数": ab_count,
                "C1C2比例(%)": c_rate,
                "候选池基准C1C2比例(%)": base_rate,
                "相对候选池提升(百分点)": c_rate - base_rate if math.isfinite(c_rate) else np.nan,
                "保留候选池C1C2(%)": c_count / base_c * 100.0 if base_c else np.nan,
                "保留候选池AB(%)": ab_count / base_ab * 100.0 if base_ab else np.nan,
                "完整周期数": int(len(complete)),
                "最高利润中位数(%)": numeric_median(complete["Peak_MFE_pct"]),
                "最大浮亏中位数(%)": numeric_median(complete["Path_MAE_pct"]),
                "达到10%(%)": pct_mean(complete["Reached_10_pct"].map(to_bool)) if len(complete) else np.nan,
                "达到20%(%)": pct_mean(complete["Reached_20_pct"].map(to_bool)) if len(complete) else np.nan,
                "达到30%(%)": pct_mean(complete["Reached_30_pct"].map(to_bool)) if len(complete) else np.nan,
                "总统计周数": int(len(reference_weeks)),
                "非空周": int(len(nonzero)),
                "空窗周": int(counts.eq(0).sum()),
                "非空周最少入选": int(nonzero.min()) if len(nonzero) else 0,
                "非空周平均入选": float(nonzero.mean()) if len(nonzero) else 0.0,
                "单周最多入选": maximum,
                "最多入选周": "|".join(max_weeks),
                "最长连续空窗(周)": int(blank_length),
                "最长空窗开始周": blank_start,
                "最长空窗结束周": blank_end,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def score_half_year_summary(pools: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pool_name, pool in pools.items():
        for half_year, half in pool.groupby("Selection_Half_Year", sort=True):
            base_resolved = half[half["Research_Family"].ne("未决")]
            base_rate = pct_mean(base_resolved["Research_Family"].eq("C1C2_长周期"))
            for scope in SCORE_SCOPES:
                selected = half[score_scope_mask(half, scope)]
                resolved = selected[selected["Research_Family"].ne("未决")]
                complete = resolved[resolved["Cycle_Completed"].map(to_bool)]
                c_rate = pct_mean(resolved["Research_Family"].eq("C1C2_长周期"))
                rows.append({
                    "确认半年": half_year,
                    "候选池": pool_name,
                    "选择范围": scope,
                    "入选事件": int(len(selected)),
                    "已决事件": int(len(resolved)),
                    "C1C2数": int(resolved["Research_Family"].eq("C1C2_长周期").sum()),
                    "AB数": int(resolved["Research_Family"].eq("AB_弱或短周期").sum()),
                    "C1C2比例(%)": c_rate,
                    "同期候选池基准(%)": base_rate,
                    "相对同期基准(百分点)": c_rate - base_rate if math.isfinite(c_rate) else np.nan,
                    "最高利润中位数(%)": numeric_median(complete["Peak_MFE_pct"]),
                    "最大浮亏中位数(%)": numeric_median(complete["Path_MAE_pct"]),
                    "达到10%(%)": pct_mean(complete["Reached_10_pct"].map(to_bool)) if len(complete) else np.nan,
                    "达到20%(%)": pct_mean(complete["Reached_20_pct"].map(to_bool)) if len(complete) else np.nan,
                    "达到30%(%)": pct_mean(complete["Reached_30_pct"].map(to_bool)) if len(complete) else np.nan,
                })
    return pd.DataFrame(rows)


def score_weekly_detail(pools: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pool_name, pool in pools.items():
        for selection_date, week in pool.groupby("Selection_Date", sort=True):
            for scope in SCORE_SCOPES:
                selected = week[score_scope_mask(week, scope)]
                resolved = selected[selected["Research_Family"].ne("未决")]
                rows.append({
                    "确认日期": selection_date,
                    "确认半年": oos_half_year(selection_date),
                    "候选池": pool_name,
                    "选择范围": scope,
                    "入选数": int(len(selected)),
                    "不同股票": int(selected["ts_code"].nunique()),
                    "已决数": int(len(resolved)),
                    "C1C2数": int(resolved["Research_Family"].eq("C1C2_长周期").sum()),
                    "AB数": int(resolved["Research_Family"].eq("AB_弱或短周期").sum()),
                    "C1C2比例(%)": pct_mean(resolved["Research_Family"].eq("C1C2_长周期")),
                    "平均评分": numeric_mean(selected["Final_Score"]),
                    "最低入选评分": pd.to_numeric(selected["Final_Score"], errors="coerce").min(),
                    "最高入选评分": pd.to_numeric(selected["Final_Score"], errors="coerce").max(),
                })
    return pd.DataFrame(rows)


def score_rank_cohort_summary(pools: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pool_name, pool in pools.items():
        for cohort, group in pool.groupby("Rank_Cohort", sort=False):
            resolved = group[group["Research_Family"].ne("未决")]
            complete = resolved[resolved["Cycle_Completed"].map(to_bool)]
            rows.append({
                "候选池": pool_name,
                "周排名分层": cohort,
                "事件数": int(len(group)),
                "已决数": int(len(resolved)),
                "C1C2数": int(resolved["Research_Family"].eq("C1C2_长周期").sum()),
                "AB数": int(resolved["Research_Family"].eq("AB_弱或短周期").sum()),
                "C1C2比例(%)": pct_mean(resolved["Research_Family"].eq("C1C2_长周期")),
                "评分中位数": numeric_median(group["Final_Score"]),
                "最高利润中位数(%)": numeric_median(complete["Peak_MFE_pct"]),
                "最大浮亏中位数(%)": numeric_median(complete["Path_MAE_pct"]),
                "达到10%(%)": pct_mean(complete["Reached_10_pct"].map(to_bool)) if len(complete) else np.nan,
                "达到20%(%)": pct_mean(complete["Reached_20_pct"].map(to_bool)) if len(complete) else np.nan,
                "达到30%(%)": pct_mean(complete["Reached_30_pct"].map(to_bool)) if len(complete) else np.nan,
            })
    return pd.DataFrame(rows)


def score_concentration_summary(pools: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions = [
        ("申万一级行业", "SW_L1"),
        ("申万二级行业", "SW_L2"),
        ("上市板块", "Sample_Board"),
        ("单只股票", "ts_code"),
        ("确认日期", "Selection_Date"),
    ]
    for pool_name, pool in pools.items():
        for scope in SCORE_SCOPES:
            selected = pool[score_scope_mask(pool, scope)]
            for label, column in dimensions:
                counts = selected[column].fillna("未知").astype(str).value_counts()
                top_value = counts.index[0] if len(counts) else ""
                top_count = int(counts.iloc[0]) if len(counts) else 0
                rows.append({
                    "候选池": pool_name,
                    "选择范围": scope,
                    "集中维度": label,
                    "入选总数": int(len(selected)),
                    "不同类别数": int(len(counts)),
                    "最大类别": top_value,
                    "最大类别数量": top_count,
                    "最大类别占比(%)": top_count / len(selected) * 100.0 if len(selected) else np.nan,
                })
    return pd.DataFrame(rows)


def score_component_audit(scored: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("第二根柱相对第一根增幅", "CP_W2_Hist_vs_W1_pct", "越高越好", 40.0),
        ("DEA/股价", "DEA_to_Price_pct", "越低越好", 25.0),
        ("MA20四周斜率", "W_MA20_Slope4_pct", "越低越好", 20.0),
        ("前26周收益", "Pre_26W_Return_pct", "越低越好", 10.0),
        ("前期回调深度", "Pullback_Depth_pct", "越高越好", 5.0),
    ]
    eligible = scored[scored["Score_Eligible_Tolerant"].map(to_bool)]
    rows = []
    for name, column, direction, weight in definitions:
        values = pd.to_numeric(eligible[column], errors="coerce")
        rows.append({
            "评分项": name,
            "原始字段": column,
            "方向": direction,
            "固定权重": weight,
            "有效数": int(values.notna().sum()),
            "缺失数": int(values.isna().sum()),
            "中位数": values.median(),
            "最小值": values.min(),
            "最大值": values.max(),
            "周内百分位规则": "同一第二周确认日内排名；缺失按中性50%",
        })
    rows.append({
        "评分项": "第二周仍红但不扩张扣分",
        "原始字段": "W2_Exact_Expansion",
        "方向": "宽容池固定扣分",
        "固定权重": -TOLERANT_NONEXPANSION_PENALTY,
        "有效数": int(len(eligible)),
        "缺失数": 0,
        "中位数": np.nan,
        "最小值": np.nan,
        "最大值": np.nan,
        "周内百分位规则": "严格池不适用；宽容池第二周仍红但未严格扩张固定扣25分",
    })
    return pd.DataFrame(rows)


def score_ablation_summary(pools: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """固定权重逐项删除，检查Top3恶化来自哪一个评分项；不据此自动调参。"""
    components = [
        ("删除第二根柱扩张40分", "ScorePoints_W2_Expansion"),
        ("删除DEA深度25分", "ScorePoints_DEA_Depth"),
        ("删除MA20斜率20分", "ScorePoints_MA20_Slope"),
        ("删除前26周收益10分", "ScorePoints_Pre26_Return"),
        ("删除回调深度5分", "ScorePoints_Pullback_Depth"),
    ]
    rows: list[dict[str, Any]] = []
    for pool_name, pool in pools.items():
        base_resolved = pool[pool["Research_Family"].ne("未决")]
        base_rate = pct_mean(base_resolved["Research_Family"].eq("C1C2_长周期"))
        variants = [("完整固定评分", None)] + components
        for variant_name, removed_column in variants:
            working = pool.copy()
            score = pd.to_numeric(working["Final_Score"], errors="coerce")
            if removed_column is not None:
                score = score - pd.to_numeric(working[removed_column], errors="coerce").fillna(0.0)
            working["Ablation_Score"] = score
            working = working.sort_values(
                ["Selection_Date", "Ablation_Score", "CP_W2_Hist_vs_W1_pct", "ts_code"],
                ascending=[True, False, False, True], na_position="last",
            )
            working["Ablation_Rank"] = working.groupby("Selection_Date").cumcount() + 1
            selected = working[working["Ablation_Rank"].le(3)]
            resolved = selected[selected["Research_Family"].ne("未决")]
            complete = resolved[resolved["Cycle_Completed"].map(to_bool)]
            c_rate = pct_mean(resolved["Research_Family"].eq("C1C2_长周期"))
            rows.append({
                "候选池": pool_name,
                "评分变体": variant_name,
                "Top3事件": int(len(selected)),
                "已决事件": int(len(resolved)),
                "C1C2数": int(resolved["Research_Family"].eq("C1C2_长周期").sum()),
                "AB数": int(resolved["Research_Family"].eq("AB_弱或短周期").sum()),
                "C1C2比例(%)": c_rate,
                "候选池基准(%)": base_rate,
                "相对候选池提升(百分点)": c_rate - base_rate if math.isfinite(c_rate) else np.nan,
                "最高利润中位数(%)": numeric_median(complete["Peak_MFE_pct"]),
                "最大浮亏中位数(%)": numeric_median(complete["Path_MAE_pct"]),
                "达到10%(%)": pct_mean(complete["Reached_10_pct"].map(to_bool)) if len(complete) else np.nan,
                "达到20%(%)": pct_mean(complete["Reached_20_pct"].map(to_bool)) if len(complete) else np.nan,
                "达到30%(%)": pct_mean(complete["Reached_30_pct"].map(to_bool)) if len(complete) else np.nan,
                "说明": "只做诊断，不自动采用表现最好的变体",
            })
    return pd.DataFrame(rows)


def weekly_score_rank_main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "固定硬条件与100分制，在每个第二周确认日内比较全部候选、Top1、Top3和Top5。"
        "严格池要求第二根红柱严格扩张；宽容池只淘汰翻绿，对不扩张固定扣25分。"
    )
    with st.sidebar:
        st.header("验证区间")
        signal_start_date = st.date_input("首红柱信号开始", value=date(2024, 6, 5))
        signal_end_date = st.date_input("首红柱信号截止", value=date(2026, 6, 5))
        observation_end_date = st.date_input(
            "行情观察截止", value=date.today(), max_value=date.today()
        )
        st.header("冻结硬条件")
        st.write("历史科技板块全量；股价≥10元；流通市值≥100亿元；可交易")
        st.write("严格池：第二根红柱严格长于第一根")
        st.write("宽容池：第二周仍为红柱；不扩张扣25分")
        st.header("冻结评分")
        for name, weight in SCORE_WEIGHTS.items():
            st.write(f"{name}：{weight:.0f}分")
        st.caption("所有评分均为同一确认周内的百分位，不使用绝对阈值。")
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
    session_key = "weekly_score_rank_v1_0_zip"
    run_requested = st.button("开始周内评分排名验证", type="primary")
    if not run_requested:
        if session_key in st.session_state:
            st.success("上一次结果仍在，可直接下载。")
            st.download_button(
                "下载上一次全部结果ZIP", st.session_state[session_key],
                file_name="weekly_macd_score_rank_v1_0_all_results.zip",
                mime="application/zip", type="primary", on_click="ignore",
            )
        else:
            st.info("重点比较两套候选池的Top3能否提高C1/C2比例，同时保持较少空窗。")
        return
    date_error = validate_research_dates(
        signal_start_date, signal_end_date, observation_end_date
    )
    if date_error:
        st.error(date_error)
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
        f"首红柱信号{signal_start}—{signal_end}；观察至{observation_end}。"
    )

    open_pos = {trade_date: position for position, trade_date in enumerate(open_dates)}
    all_records: list[dict[str, Any]] = []
    daily_histories: dict[str, pd.DataFrame] = {}
    reject_totals: dict[str, int] = {}
    cache_hits = data_failures = 0
    progress = st.progress(0.0, text="逐股票生成首红柱事件...")
    status = st.empty()
    for idx, stock in stocks_to_fetch.iterrows():
        code = str(stock["ts_code"])
        progress.progress(
            (idx + 1) / len(stocks_to_fetch),
            text=f"{idx + 1}/{len(stocks_to_fetch)} {code}",
        )
        status.caption(
            f"底层事件{len(all_records)}；缓存命中{cache_hits}；行情失败{data_failures}"
        )
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
    with st.spinner("计算周期标签、周内百分位评分和Top1/Top3/Top5..."):
        opportunities = build_cycle_opportunities(
            events, daily_histories, observation_end, config["sell_slippage_pct"]
        )
        scored = add_weekly_score_features(opportunities)
        strict_pool = build_score_pool(
            scored, SCORE_POOL_STRICT, "Score_Eligible_Strict", "Score_Strict_Final"
        )
        tolerant_pool = build_score_pool(
            scored, SCORE_POOL_TOLERANT, "Score_Eligible_Tolerant", "Score_Tolerant_Final"
        )
        if strict_pool.empty or tolerant_pool.empty:
            st.error("评分候选池为空，请检查第二周数据和日期区间。")
            return
        pools = {SCORE_POOL_STRICT: strict_pool, SCORE_POOL_TOLERANT: tolerant_pool}
        reference_weeks = score_reference_weeks(scored)
        strategy_summary = score_strategy_summary(pools, reference_weeks)
        half_year_summary = score_half_year_summary(pools)
        weekly_detail = score_weekly_detail(pools)
        rank_cohort = score_rank_cohort_summary(pools)
        concentration = score_concentration_summary(pools)
        component_audit = score_component_audit(scored)
        ablation = score_ablation_summary(pools)

    valid = scored[scored["Opportunity_Valid"].map(to_bool)].copy()
    unresolved = valid[valid["Research_Family"].eq("未决")].copy()
    reject_frame = pd.DataFrame([
        {"剔除原因": reason, "次数": count} for reason, count in reject_totals.items()
    ]).sort_values("次数", ascending=False) if reject_totals else pd.DataFrame(
        columns=["剔除原因", "次数"]
    )
    strict_top3 = strategy_summary[
        strategy_summary["候选池"].eq(SCORE_POOL_STRICT)
        & strategy_summary["选择范围"].eq("Top3")
    ].iloc[0]
    tolerant_top3 = strategy_summary[
        strategy_summary["候选池"].eq(SCORE_POOL_TOLERANT)
        & strategy_summary["选择范围"].eq("Top3")
    ].iloc[0]
    run_summary = pd.DataFrame([{
        "程序": TITLE,
        "版本": VERSION,
        "信号开始": signal_start,
        "信号截止": signal_end,
        "观察截止": observation_end,
        "首红柱事件": int(len(scored)),
        "有效事件": int(len(valid)),
        "严格池候选": int(len(strict_pool)),
        "严格池不同股票": int(strict_pool["ts_code"].nunique()),
        "宽容池候选": int(len(tolerant_pool)),
        "宽容池不同股票": int(tolerant_pool["ts_code"].nunique()),
        "严格池Top3事件": int(strict_top3["入选事件"]),
        "严格池Top3_C1C2比例(%)": strict_top3["C1C2比例(%)"],
        "严格池Top3空窗周": int(strict_top3["空窗周"]),
        "宽容池Top3事件": int(tolerant_top3["入选事件"]),
        "宽容池Top3_C1C2比例(%)": tolerant_top3["C1C2比例(%)"],
        "宽容池Top3空窗周": int(tolerant_top3["空窗周"]),
        "未决事件": int(len(unresolved)),
        "真实行情失败": int(data_failures),
        "缓存命中": int(cache_hits),
    }])
    metadata = pd.DataFrame([
        {"项目": "程序", "值": TITLE},
        {"项目": "目的", "值": "验证少量硬条件＋连续评分＋每周Top1/Top3/Top5能否减少AB且保持信号连续"},
        {"项目": "严格池", "值": "第二周红柱严格长于第一根"},
        {"项目": "宽容池", "值": "第二周仍为红柱；未严格扩张固定扣25分；直接翻绿淘汰"},
        {"项目": "评分方法", "值": "同一第二周确认日内计算百分位；缺失值按中性50%"},
        {"项目": "固定权重", "值": "第二周扩张40、DEA深度25、MA20斜率20、前26周收益10、回调深度5"},
        {"项目": "排名并列处理", "值": "总分、第二周扩张幅度、DEA深度、MA20斜率、股票代码依次破同分"},
        {"项目": "A/B与C", "值": "A/B为完整红柱周期少于9周；C1/C2为不少于9周；进行中满9周可确认C"},
        {"项目": "利润口径", "值": "最高利润和最大浮亏只用于事后评价，不是最终止盈止损收益"},
        {"项目": "组合限制", "值": "本版不模拟三仓占用、资金曲线和最终退出"},
        {"项目": "股票池", "值": "历史科技板块全量；信号日股价≥10元、流通市值≥100亿元"},
    ])
    files = {
        "01_run_summary_score_rank_v1_0.csv": run_summary,
        "02_strategy_comparison_score_rank_v1_0.csv": strategy_summary,
        "03_half_year_robustness_score_rank_v1_0.csv": half_year_summary,
        "04_weekly_selection_detail_score_rank_v1_0.csv": weekly_detail,
        "05_rank_cohort_quality_score_rank_v1_0.csv": rank_cohort,
        "06_concentration_audit_score_rank_v1_0.csv": concentration,
        "07_score_component_audit_score_rank_v1_0.csv": component_audit,
        "08_score_ablation_top3_score_rank_v1_0.csv": ablation,
        "09_strict_pool_candidate_detail_score_rank_v1_0.csv": strict_pool,
        "10_tolerant_pool_candidate_detail_score_rank_v1_0.csv": tolerant_pool,
        "11_unresolved_event_detail_score_rank_v1_0.csv": unresolved,
        "12_all_event_score_detail_score_rank_v1_0.csv": scored,
        "13_full_tech_universe_score_rank_v1_0.csv": sample_audit,
        "14_population_score_rank_v1_0.csv": population_summary,
        "15_rejection_audit_score_rank_v1_0.csv": reject_frame,
        "16_metadata_score_rank_v1_0.csv": metadata,
    }
    result_zip = make_result_zip(files)
    st.session_state[session_key] = result_zip

    st.success(
        f"验证完成：严格池{len(strict_pool)}个，宽容池{len(tolerant_pool)}个；"
        f"严格池Top3的C1/C2比例{strict_top3['C1C2比例(%)']:.2f}%，"
        f"宽容池Top3为{tolerant_top3['C1C2比例(%)']:.2f}%。"
    )
    metrics = st.columns(6)
    metrics[0].metric("严格池候选", f"{len(strict_pool)}")
    metrics[1].metric("严格池Top3 C1C2", f"{strict_top3['C1C2比例(%)']:.2f}%")
    metrics[2].metric("严格池Top3空窗", f"{int(strict_top3['空窗周'])}周")
    metrics[3].metric("宽容池候选", f"{len(tolerant_pool)}")
    metrics[4].metric("宽容池Top3 C1C2", f"{tolerant_top3['C1C2比例(%)']:.2f}%")
    metrics[5].metric("宽容池Top3空窗", f"{int(tolerant_top3['空窗周'])}周")

    st.subheader("候选池与Top1/Top3/Top5总比较")
    st.dataframe(strategy_summary, use_container_width=True, hide_index=True)
    st.subheader("半年稳定性")
    st.dataframe(half_year_summary, use_container_width=True, hide_index=True)
    with st.expander("周度明细、排名分层与集中度审计"):
        st.dataframe(weekly_detail, use_container_width=True, hide_index=True)
        st.dataframe(rank_cohort, use_container_width=True, hide_index=True)
        st.dataframe(concentration, use_container_width=True, hide_index=True)
        st.dataframe(ablation, use_container_width=True, hide_index=True)

    st.subheader("下载结果")
    st.download_button(
        "下载1号：全部结果ZIP", result_zip,
        file_name="weekly_macd_score_rank_v1_0_all_results.zip",
        mime="application/zip", type="primary", on_click="ignore",
    )
    labels = [
        "2号：运行总表", "3号：策略总比较", "4号：半年稳定性", "5号：每周选择明细",
        "6号：排名分层质量", "7号：集中度审计", "8号：评分项审计", "9号：Top3删项诊断",
        "10号：严格池明细", "11号：宽容池明细", "12号：未决事件", "13号：全部评分事件",
        "14号：科技股票池", "15号：板块数量", "16号：剔除审计", "17号：验证口径",
    ]
    columns = st.columns(4)
    for index, (filename, frame) in enumerate(files.items()):
        with columns[index % 4]:
            st.download_button(
                labels[index], csv_bytes(frame), file_name=filename,
                mime="text/csv", key=f"score_rank_{filename}", on_click="ignore",
            )
    st.warning(
        "本版只验证选股排序质量。Top3不是三仓资金组合：没有处理持仓重叠、退出、"
        "资金占用和收益复利；只有评分稳定后才应进入组合回测。"
    )

# ===== V2.3 nonlinear expert scoring lab (single file) =====
TITLE = "科技股周线MACD双专家非线性评分实验器 V2.3"
VERSION = "V2.3-NONLINEAR-EXPERT-SCORING-LAB"
TREND, REPAIR, AUTO = "趋势延续", "超跌修复", "自动选择"
TRAIN_WEEKS, PERF_WEEKS = 52, 26
TRAIN_HALF_LIFE, PERF_HALF_LIFE = 26.0, 13.0
REFIT_EVERY_WEEKS = 13
MIN_WEEKS, MIN_ROWS, RIDGE = 26, 180, 12.0
LEARNED_WEIGHT = 0.35
WARMUP_DAYS = 600
BIG_MOVE_TARGETS = (20, 30, 50, 100)
SCORE_VARIANTS = ("非线性基础", "过热惩罚5", "过热惩罚10")
DEFAULT_VARIANT = "过热惩罚5"
TREND_FEATURES = (
    "TF_Return13", "TF_Return26", "TF_MA20Slope", "TF_W2Return",
    "TF_CloseLocation", "TF_BoardRS", "TF_Volume", "TF_W2Expansion",
)
REPAIR_FEATURES = (
    "RF_DEADepth", "RF_Pullback", "RF_PriorLoss", "RF_SlopeImprove",
    "RF_DEAImprove", "RF_ReclaimMA20", "RF_W2Return", "RF_CloseLocation",
    "RF_BoardRS", "RF_Volume", "RF_RepairConfirmation",
)
TREND_PRIOR = np.array([0.18, 0.10, 0.18, 0.08, 0.08, 0.16, 0.08, 0.14], dtype=float)
REPAIR_PRIOR = np.array([0.10, 0.08, 0.06, 0.14, 0.14, 0.12, 0.08, 0.06, 0.10, 0.05, 0.07], dtype=float)


def num(v: Any, index: pd.Index | None = None) -> pd.Series:
    return pd.to_numeric(v, errors="coerce") if isinstance(v, pd.Series) else pd.Series(v, index=index, dtype=float)


def bool_value(v: Any) -> bool:
    return to_bool(v)


def rate(v: pd.Series) -> float:
    return float(v.astype(bool).mean() * 100.0) if len(v) else np.nan


def exp_weights(dates: pd.Series, current: pd.Timestamp, half_life: float) -> np.ndarray:
    age = (current - pd.to_datetime(dates)).dt.days.to_numpy(float) / 7.0
    return np.power(0.5, np.maximum(age, 0.0) / half_life)


def wmean(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() else np.nan


def week_pct(frame: pd.DataFrame, values: Any, higher: bool = True) -> pd.Series:
    x = num(values, frame.index)
    oriented = x if higher else -x
    return oriented.groupby(frame["Selection_Date"]).rank(method="average", pct=True).fillna(0.5)


def add_board_data(frame: pd.DataFrame, board_weeklies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = frame.copy()
    out["Board_Index"] = out["Sample_Board"].map(BOARD_INDEX).fillna("")
    rows = []
    for code, weekly in board_weeklies.items():
        for _, r in weekly.iterrows():
            rows.append({
                "Board_Index": code, "Selection_Date": normalize_date(r.get("trade_date")),
                "Board_Return_13W_pct": finite_num(r.get("return_13w_pct")),
                "Board_MA20_Slope4_pct": finite_num(r.get("ma20_slope4_pct")),
            })
    env = pd.DataFrame(rows)
    if env.empty:
        out["Board_Return_13W_pct"] = np.nan
        out["Board_MA20_Slope4_pct"] = np.nan
        return out
    return out.merge(env.drop_duplicates(["Board_Index", "Selection_Date"]),
                     on=["Board_Index", "Selection_Date"], how="left")


def utility(frame: pd.DataFrame) -> pd.Series:
    state = frame["CP_W2_Delayed_First_20_vs_Stop"].fillna("").astype(str)
    ending = num(frame["CP_W2_Delayed_Return_8W_pct"]).clip(-10, 20)
    return pd.Series(np.select(
        [state.eq("目标先到"), state.isin(["止损先到", "同日不确定_按止损"])],
        [20.0, -10.0], default=ending), index=frame.index, dtype=float)


def target_first(frame: pd.DataFrame, target: int) -> pd.Series:
    column = f"CP_W2_Delayed_First_{target}_vs_Stop"
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna("").astype(str).eq("目标先到")


def big_move_target(frame: pd.DataFrame) -> pd.Series:
    """有界分层目标：20/30是主体，50仅小幅加分，翻倍只审计不额外加权。"""
    win20 = target_first(frame, 20)
    win30 = target_first(frame, 30)
    win50 = target_first(frame, 50)
    state20 = frame["CP_W2_Delayed_First_20_vs_Stop"].fillna("").astype(str)
    stopped = state20.isin(["止损先到", "同日不确定_按止损"])
    ending = num(frame["CP_W2_Delayed_Return_8W_pct"]).clip(-10, 20).fillna(0.0)
    fallback = ending * 0.5
    layered = 60.0 * win20.astype(float) + 25.0 * win30.astype(float) + 15.0 * win50.astype(float)
    return pd.Series(np.select([win20, stopped], [layered, -40.0], default=fallback),
                     index=frame.index, dtype=float)


def prepare_features(opportunities: pd.DataFrame, board_weeklies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = add_oos_research_features(opportunities)
    frame["Selection_Date"] = frame.get("CP_W2_Date", pd.Series("", index=frame.index)).map(normalize_date)
    frame["Selection_Date_dt"] = pd.to_datetime(frame["Selection_Date"], format="%Y%m%d", errors="coerce")
    frame["Selection_Year"] = frame["Selection_Date"].str[:4]
    frame["Strict_Eligible"] = (
        frame["Opportunity_Valid"].map(bool_value) & frame["W2_Observed"].map(bool_value)
        & frame["W2_Exact_Expansion"].map(bool_value) & frame["Selection_Date_dt"].notna()
    )
    frame = add_board_data(frame, board_weeklies)
    x = frame[frame["Strict_Eligible"]].copy()
    if x.empty:
        return frame
    x["Board_RS"] = num(x.get("CP_W2_Return_13W_pct"), x.index) - num(x.get("Board_Return_13W_pct"), x.index)
    x["Slope_Improve"] = num(x.get("CP_W2_MA20_Slope4_pct"), x.index) - num(x.get("W_MA20_Slope4_pct"), x.index)
    x["DEA_Improve"] = num(x.get("CP_W2_DEA_to_Price_pct"), x.index) - num(x.get("DEA_to_Price_pct"), x.index)
    specs = {
        "TF_Return13": (x.get("CP_W2_Return_13W_pct"), True),
        "TF_Return26": (x.get("CP_W2_Return_26W_pct"), True),
        "TF_MA20Slope": (x.get("CP_W2_MA20_Slope4_pct"), True),
        "TF_W2Return": (x.get("CP_W2_Week_Return_pct"), True),
        "TF_CloseLocation": (x.get("CP_W2_Close_Location"), True),
        "TF_BoardRS": (x["Board_RS"], True),
        "TF_Volume": (num(x.get("CP_W2_Volume_Ratio20"), x.index).clip(0, 5), True),
        "TF_W2Expansion": (x.get("CP_W2_Hist_vs_W1_pct"), True),
        "RF_DEADepth": (x.get("CP_W2_DEA_to_Price_pct"), False),
        "RF_Pullback": (x.get("Pullback_Depth_pct"), True),
        "RF_PriorLoss": (x.get("CP_W2_Return_26W_pct"), False),
        "RF_SlopeImprove": (x["Slope_Improve"], True),
        "RF_DEAImprove": (x["DEA_Improve"], True),
        "RF_ReclaimMA20": (x.get("CP_W2_Close_vs_MA20_pct"), True),
        "RF_W2Return": (x.get("CP_W2_Week_Return_pct"), True),
        "RF_CloseLocation": (x.get("CP_W2_Close_Location"), True),
        "RF_BoardRS": (x["Board_RS"], True),
        "RF_Volume": (num(x.get("CP_W2_Volume_Ratio20"), x.index).clip(0, 5), True),
    }
    for name, (values, direction) in specs.items():
        x[name] = week_pct(x, values, direction)
    improve = 0.5 * x["RF_SlopeImprove"] + 0.5 * x["RF_DEAImprove"]
    x["RF_RepairConfirmation"] = x["RF_DEADepth"] * improve
    x["Outcome_Mature"] = x["CP_W2_Delayed_Has_8W_Future"].map(bool_value)
    maturity_source = x.get(
        "CP_W2_Delayed_Observation_End_Date", pd.Series("", index=x.index)
    )
    maturity = pd.to_datetime(maturity_source.astype(str), format="%Y%m%d", errors="coerce")
    x["Outcome_Maturity_Date_dt"] = maturity.fillna(x["Selection_Date_dt"] + pd.Timedelta(days=56))
    x["Realised_Utility"] = utility(x)
    for target in BIG_MOVE_TARGETS:
        x[f"Win{target}BeforeStop"] = target_first(x, target)
    x["BigMove_Target"] = big_move_target(x)
    new = ["Board_RS", "Slope_Improve", "DEA_Improve", *TREND_FEATURES, *REPAIR_FEATURES,
           "Outcome_Mature", "Outcome_Maturity_Date_dt", "Realised_Utility", "BigMove_Target",
           *[f"Win{target}BeforeStop" for target in BIG_MOVE_TARGETS]]
    for c in new:
        frame[c] = False if c == "Outcome_Mature" or c.startswith("Win") else (
            pd.NaT if c == "Outcome_Maturity_Date_dt" else np.nan)
        frame.loc[x.index, c] = x[c]
    return frame


def fit_stable_expert(
    train: pd.DataFrame,
    features: tuple[str, ...],
    prior: np.ndarray,
    current: pd.Timestamp,
) -> dict[str, Any]:
    """季度慢更新、非负方向、先验锚定；极少数翻倍股不会得到额外权重。"""
    prior = np.asarray(prior, dtype=float)
    prior = prior / prior.sum()
    base = {"ready": False, "features": features, "weights": prior,
            "rows": len(train), "r2": np.nan, "fit_date": current}
    if len(train) < MIN_ROWS or train.Selection_Date_dt.nunique() < MIN_WEEKS:
        return base
    X = train.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(0.5).to_numpy(float)
    y = num(train["BigMove_Target"]).to_numpy(float)
    w = exp_weights(train["Selection_Date_dt"], current, TRAIN_HALF_LIFE)
    ok = np.isfinite(y) & np.isfinite(w)
    X, y, w = X[ok], y[ok], w[ok]
    if len(y) < MIN_ROWS:
        return base
    mu = (X * w[:, None]).sum(0) / w.sum()
    scale = np.sqrt(np.maximum((((X - mu) ** 2) * w[:, None]).sum(0) / w.sum(), 1e-8))
    Z = (X - mu) / scale
    D = np.column_stack([np.ones(len(Z)), Z])
    Dw, yw = D * np.sqrt(w[:, None]), y * np.sqrt(w)
    penalty = np.eye(D.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(Dw.T @ Dw + penalty) @ (Dw.T @ yw)
    learned = np.maximum(beta[1:], 0.0)
    learned = learned / learned.sum() if learned.sum() > 1e-12 else prior.copy()
    final_weights = (1.0 - LEARNED_WEIGHT) * prior + LEARNED_WEIGHT * learned
    final_weights = final_weights / final_weights.sum()
    raw_score = X @ final_weights
    weighted_mean_y = wmean(y, w)
    weighted_mean_s = wmean(raw_score, w)
    cov = wmean((raw_score - weighted_mean_s) * (y - weighted_mean_y), w)
    var = wmean((raw_score - weighted_mean_s) ** 2, w)
    slope = max(cov / var, 0.0) if math.isfinite(cov) and math.isfinite(var) and var > 1e-12 else 0.0
    fitted = weighted_mean_y + slope * (raw_score - weighted_mean_s)
    denom = np.sum(w * (y - weighted_mean_y) ** 2)
    r2 = 1.0 - np.sum(w * (y - fitted) ** 2) / denom if denom > 0 else np.nan
    return {"ready": True, "features": features, "weights": final_weights,
            "learned_weights": learned, "rows": len(y), "r2": r2,
            "fit_date": current, "train_mean_target": weighted_mean_y}


def predict(frame: pd.DataFrame, features: tuple[str, ...], state: dict[str, Any]) -> np.ndarray:
    X = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(0.5).to_numpy(float)
    return (X @ np.asarray(state["weights"], dtype=float)) * 100.0


def coefficient_rows(dt: pd.Timestamp, model: str, state: dict[str, Any], train: pd.DataFrame) -> list[dict[str, Any]]:
    base = {"生效日期": dt.strftime("%Y%m%d"), "专家模型": model,
            "训练完成": bool(state.get("ready")), "训练周数": train.Selection_Date.nunique(),
            "训练候选": len(train), "训练开始": train.Selection_Date.min() if len(train) else "",
            "训练截止": train.Selection_Date.max() if len(train) else "", "加权R2": state.get("r2", np.nan)}
    learned = state.get("learned_weights", np.full(len(state["features"]), np.nan))
    return [{**base, "特征": feature, "最终权重(%)": float(weight * 100.0),
             "纯数据权重(%)": float(data_weight * 100.0) if math.isfinite(data_weight) else np.nan}
            for feature, weight, data_weight in zip(state["features"], state["weights"], learned)]


def recent_edges(history: list[dict[str, Any]], current: pd.Timestamp) -> tuple[dict[str, float], int, str, str]:
    h = pd.DataFrame(history)
    empty = {TREND: np.nan, REPAIR: np.nan}
    if h.empty:
        return empty, 0, "", ""
    h = h[h["Maturity_Date_dt"].lt(current)].sort_values("Date_dt")
    dates = h.Date_dt.drop_duplicates().tail(PERF_WEEKS)
    h = h[h.Date_dt.isin(dates)]
    if len(dates) < MIN_WEEKS:
        return empty, len(dates), "", ""
    wd = pd.Series(exp_weights(pd.Series(dates), current, PERF_HALF_LIFE), index=dates.to_numpy())
    edges = {}
    for model in (TREND, REPAIR):
        g = h[h["专家模型"].eq(model)]
        edges[model] = wmean(g["相对候选效用"].to_numpy(float), g.Date_dt.map(wd).to_numpy(float))
    return edges, len(dates), dates.min().strftime("%Y%m%d"), dates.max().strftime("%Y%m%d")


def capped_rise(values: pd.Series, floor: float, full: float) -> pd.Series:
    return ((num(values) - floor) / max(full - floor, 1e-8)).clip(0.0, 1.0)


def sweet_spot(values: pd.Series, low: float, ideal_low: float,
               ideal_high: float, high: float) -> pd.Series:
    x = num(values)
    left = ((x - low) / max(ideal_low - low, 1e-8)).clip(0.0, 1.0)
    right = ((high - x) / max(high - ideal_high, 1e-8)).clip(0.0, 1.0)
    return pd.concat([left, right], axis=1).min(axis=1)


def score_model(week: pd.DataFrame, model: str, variant: str) -> pd.DataFrame:
    g = week.copy()
    if model == TREND:
        components = {
            "趋势_13周强度得分": 30.0 * capped_rise(g.TF_Return13, 0.25, 0.75),
            "趋势_板块强度得分": 30.0 * capped_rise(g.TF_BoardRS, 0.25, 0.85),
            "趋势_成交量得分": 15.0 * sweet_spot(g.TF_Volume, 0.10, 0.45, 0.80, 1.20),
            "趋势_MA20斜率得分": 10.0 * capped_rise(g.TF_MA20Slope, 0.25, 0.80),
            "趋势_第二周涨幅得分": 10.0 * sweet_spot(g.TF_W2Return, 0.05, 0.25, 0.80, 1.25),
            "趋势_红柱扩张得分": 5.0 * sweet_spot(g.TF_W2Expansion, 0.05, 0.25, 0.75, 1.25),
        }
        extreme_features = ["TF_Return13", "TF_BoardRS", "TF_Volume", "TF_MA20Slope",
                            "TF_W2Return", "TF_W2Expansion"]
    else:
        components = {
            "修复_收复MA20得分": 25.0 * capped_rise(g.RF_ReclaimMA20, 0.25, 0.80),
            "修复_板块强度得分": 20.0 * capped_rise(g.RF_BoardRS, 0.25, 0.85),
            "修复_DEA改善得分": 15.0 * capped_rise(g.RF_DEAImprove, 0.35, 0.85),
            "修复_斜率改善得分": 15.0 * capped_rise(g.RF_SlopeImprove, 0.30, 0.80),
            "修复_成交量得分": 10.0 * sweet_spot(g.RF_Volume, 0.10, 0.45, 0.80, 1.20),
            "修复_回调甜蜜区得分": 10.0 * sweet_spot(g.RF_Pullback, 0.15, 0.45, 0.80, 1.25),
            "修复_第二周涨幅得分": 5.0 * sweet_spot(g.RF_W2Return, 0.05, 0.25, 0.80, 1.25),
        }
        extreme_features = ["RF_ReclaimMA20", "RF_BoardRS", "RF_DEAImprove",
                            "RF_SlopeImprove", "RF_Volume", "RF_Pullback", "RF_W2Return"]
    for name, values in components.items():
        g[name] = values
    component_names = list(components)
    g["基础非线性得分"] = g[component_names].sum(axis=1)
    g["极端指标数"] = g[extreme_features].apply(pd.to_numeric, errors="coerce").ge(0.90).sum(axis=1)
    penalty_per_item = 0.0 if variant == "非线性基础" else (5.0 if variant == "过热惩罚5" else 10.0)
    g["过热扣分"] = (g["极端指标数"] - 2).clip(lower=0) * penalty_per_item
    g["Expert_Score"] = g["基础非线性得分"] - g["过热扣分"]
    g["专家模型"] = model
    g["评分变体"] = variant
    g["评分项明细"] = g[component_names].round(2).astype(str).agg("|".join, axis=1)
    g = g.sort_values(["Expert_Score", "CP_W2_Hist_vs_W1_pct", "ts_code"],
                      ascending=[False, False, True])
    g["Expert_Rank"] = np.arange(1, len(g) + 1)
    for n in (1, 2, 3):
        g[f"Selected_Top{n}"] = g.Expert_Rank.le(n)
    return g


def build_rankings(candidates: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, week in candidates.sort_values(["Selection_Date_dt", "ts_code"]).groupby("Selection_Date_dt"):
        for model in (TREND, REPAIR):
            for variant in SCORE_VARIANTS:
                parts.append(score_model(week, model, variant))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summary_row(frame: pd.DataFrame, plan: str, scope: str, period: str) -> dict[str, Any]:
    m = frame[frame.Outcome_Mature.map(bool_value)].copy(); ret = num(m.CP_W2_Delayed_Return_8W_pct)
    resolved = m[m.Research_Family.ne("未决")]
    ordered = m.assign(_mfe=num(m.CP_W2_Delayed_MFE_8W_pct)).sort_values("_mfe", ascending=False)
    without3 = ordered.iloc[min(3, len(ordered)):]
    without5 = ordered.iloc[min(5, len(ordered)):]
    return {"统计期": period, "方案": plan, "选择范围": scope, "入选事件": len(frame),
            "入选周数": frame.Selection_Date.nunique(), "不同股票": frame.ts_code.nunique(), "成熟结果": len(m),
            "大涨目标均值": num(m.BigMove_Target).mean(),
            "交易效用均值": num(m.Realised_Utility).mean(), "交易效用中位数": num(m.Realised_Utility).median(),
            "8周收益均值(%)": ret.mean(), "8周收益中位数(%)": ret.median(), "盈利比例(%)": rate(ret.gt(0)),
            "20%先于-10%(%)": rate(m.CP_W2_Delayed_First_20_vs_Stop.astype(str).eq("目标先到")),
            "30%先于-10%(%)": rate(target_first(m, 30)),
            "50%先于-10%(%)": rate(target_first(m, 50)),
            "100%先于-10%(%)": rate(target_first(m, 100)),
            "触及-10%(%)": rate(m.CP_W2_Delayed_Hit_Stop_8W.map(bool_value)),
            "MFE中位数(%)": num(m.CP_W2_Delayed_MFE_8W_pct).median(),
            "MAE中位数(%)": num(m.CP_W2_Delayed_MAE_8W_pct).median(),
            "删除最高3只后大涨目标均值": num(without3.BigMove_Target).mean(),
            "删除最高5只后大涨目标均值": num(without5.BigMove_Target).mean(),
            "C1C2比例(%)": rate(resolved.Research_Family.eq("C1C2_长周期"))}


def make_summary(rankings: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    r = rankings[rankings.Selection_Date_dt.ge(start)]
    periods = [("全部", None)] + [(y, y) for y in sorted(r.Selection_Year.dropna().unique())]
    rows = []
    for label, year in periods:
        for variant in SCORE_VARIANTS:
            for model in (TREND, REPAIR):
                g = r[r.专家模型.eq(model) & r.评分变体.eq(variant)]
                g = g if year is None else g[g.Selection_Year.eq(year)]
                for n in (1, 2, 3):
                    row = summary_row(g[g.Expert_Rank.le(n)], model, f"Top{n}", label)
                    row["评分变体"] = variant
                    rows.append(row)
    return pd.DataFrame(rows)


def exact_rank_quality(rankings: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    r = rankings[rankings.Selection_Date_dt.ge(start)]
    rows = []
    for variant in SCORE_VARIANTS:
        for model in (TREND, REPAIR):
            g = r[r.评分变体.eq(variant) & r.专家模型.eq(model)]
            for rank in range(1, 6):
                z = g[g.Expert_Rank.eq(rank)]
                row = summary_row(z, model, f"精确第{rank}名", "全部")
                row["评分变体"] = variant
                row["精确名次"] = rank
                rows.append(row)
    return pd.DataFrame(rows)


def allocation_comparison(rankings: pd.DataFrame, start: pd.Timestamp,
                          variant: str = DEFAULT_VARIANT) -> pd.DataFrame:
    r = rankings[rankings.Selection_Date_dt.ge(start) & rankings.评分变体.eq(variant)]
    plans = (("趋势Top1", 1, 0), ("修复Top1", 0, 1), ("各取Top1", 1, 1),
             ("趋势Top1＋修复Top2", 1, 2), ("趋势Top2＋修复Top1", 2, 1))
    rows = []
    for label, trend_n, repair_n in plans:
        parts = []
        for _, week in r.groupby("Selection_Date"):
            chosen = []
            if trend_n:
                chosen.append(week[week.专家模型.eq(TREND)].nsmallest(trend_n, "Expert_Rank"))
            if repair_n:
                chosen.append(week[week.专家模型.eq(REPAIR)].nsmallest(repair_n, "Expert_Rank"))
            if chosen:
                parts.append(pd.concat(chosen).drop_duplicates("Cycle_ID", keep="first"))
        selected = pd.concat(parts, ignore_index=True) if parts else r.head(0)
        row = summary_row(selected, "双专家并行", label, "全部")
        row["评分变体"] = variant
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_acceptance_audit(exact: pd.DataFrame, candidates: pd.DataFrame,
                             start: pd.Timestamp) -> pd.DataFrame:
    pool = candidates[candidates.Selection_Date_dt.ge(start) & candidates.Outcome_Mature.map(bool_value)]
    pool_big = num(pool.BigMove_Target).mean()
    pool_win20 = rate(target_first(pool, 20))
    rows = []
    for variant in SCORE_VARIANTS:
        for model in (TREND, REPAIR):
            g = exact[(exact.评分变体.eq(variant)) & (exact.方案.eq(model))].set_index("精确名次")
            big = [g.loc[n, "大涨目标均值"] for n in (1, 2, 3)]
            util = [g.loc[n, "交易效用均值"] for n in (1, 2, 3)]
            win20 = [g.loc[n, "20%先于-10%(%)"] for n in (1, 2, 3)]
            checks = {
                "大涨目标第1>第2>第3": bool(big[0] > big[1] > big[2]),
                "交易效用第1>第2>第3": bool(util[0] > util[1] > util[2]),
                "20%成功率第1>第2>第3": bool(win20[0] > win20[1] > win20[2]),
                "第一名大涨目标超过候选池": bool(big[0] > pool_big),
                "第一名20%成功率超过候选池": bool(win20[0] > pool_win20),
            }
            rows.append({"评分变体": variant, "专家模型": model,
                         "候选池大涨目标": pool_big, "候选池20%成功率(%)": pool_win20,
                         "第一名大涨目标": big[0], "第一名20%成功率(%)": win20[0],
                         **checks, "严格结论": "通过" if all(checks.values()) else "不通过"})
    return pd.DataFrame(rows)


def feature_commonality_audit(candidates: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """用中位数、分位组成功率和年度方向检查共同点，避免均值被少数牛股主导。"""
    m = candidates[candidates.Selection_Date_dt.ge(start) & candidates.Outcome_Mature.map(bool_value)].copy()
    if m.empty:
        return pd.DataFrame()
    rows = []
    for model, features in ((TREND, TREND_FEATURES), (REPAIR, REPAIR_FEATURES)):
        for feature in features:
            values = num(m[feature])
            win20 = target_first(m, 20)
            win30 = target_first(m, 30)
            fail20 = ~win20
            trimmed = m.assign(_mfe=num(m.CP_W2_Delayed_MFE_8W_pct)).sort_values("_mfe", ascending=False).iloc[min(5, len(m)):]
            trimmed_win = target_first(trimmed, 20)
            yearly_positive = yearly_comparable = 0
            for _, group in m.groupby("Selection_Year"):
                yw = target_first(group, 20)
                if int(yw.sum()) >= 5 and int((~yw).sum()) >= 5:
                    yearly_comparable += 1
                    yearly_positive += int(num(group.loc[yw, feature]).median() > num(group.loc[~yw, feature]).median())
            top_quartile = values.ge(0.75)
            bottom_quartile = values.le(0.25)
            rows.append({
                "专家模型": model, "特征": feature, "成熟样本": len(m),
                "20%成功数": int(win20.sum()), "30%成功数": int(win30.sum()),
                "20%成功组中位数": num(m.loc[win20, feature]).median(),
                "非20%组中位数": num(m.loc[fail20, feature]).median(),
                "20%组中位数差": num(m.loc[win20, feature]).median() - num(m.loc[fail20, feature]).median(),
                "30%成功组中位数": num(m.loc[win30, feature]).median(),
                "最高四分位20%成功率(%)": rate(win20.loc[top_quartile]),
                "最低四分位20%成功率(%)": rate(win20.loc[bottom_quartile]),
                "删除MFE最高5只后中位数差": (
                    num(trimmed.loc[trimmed_win, feature]).median()
                    - num(trimmed.loc[~trimmed_win, feature]).median()),
                "年度同方向数": yearly_positive, "年度可比较数": yearly_comparable,
            })
    return pd.DataFrame(rows).sort_values(["专家模型", "20%组中位数差"], ascending=[True, False])


def feature_dict() -> pd.DataFrame:
    return pd.DataFrame([
        (TREND, "13周强度30分", "达到较强区间后封顶，不再因极端强势无限加分"),
        (TREND, "板块相对强度30分", "主要趋势因子；高位封顶"),
        (TREND, "成交量15分", "甜蜜区评分，极端放量降分"),
        (TREND, "MA20斜率10分", "辅助确认；高位封顶"),
        (TREND, "第二周涨幅10分/红柱扩张5分", "宽甜蜜区；避免追逐最极端周涨幅"),
        (REPAIR, "收复MA20 25分/板块强度20分", "修复的主要确认"),
        (REPAIR, "DEA改善15分/斜率改善15分", "奖励改善速度，不再奖励DEA绝对深度"),
        (REPAIR, "成交量10分/回调甜蜜区10分", "适度放量和适度深回调，最极端区间降分"),
        (REPAIR, "第二周涨幅5分", "只作辅助确认"),
        ("已删除", "26周收益/收盘位置/DEA深度/前期亏损/修复交互", "V2.2共同性审计未显示稳定正作用"),
        ("共同目标", "20/30/50分层", "20%成功为主体、30%强化、50%小幅加分；100%只审计不训练"),
        ("A/B变体", "基础/每项扣5/每项扣10", "超过两个90%分位极端指标后开始过热扣分"),
    ], columns=["专家模型", "特征组", "含义"])


def v23_main_legacy() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide"); st.title(TITLE)
    st.caption("本版只验证评分顺序：固定候选池、关闭学习、取消模式切换，比较透明非线性评分和两档过热惩罚。")
    with st.sidebar:
        st.header("正式评价区间")
        eval_date = st.date_input("评价开始", value=date(2023, 6, 5))
        end_date = st.date_input("信号截止", value=date(2026, 6, 5))
        obs_date = st.date_input("行情观察截止", value=date.today(), max_value=date.today())
        view_variant = st.radio("界面重点查看评分", list(SCORE_VARIANTS), index=1)
        view_model = st.radio("界面重点查看专家", [TREND, REPAIR], index=0)
        st.header("冻结规则")
        st.write("硬条件与V2.2完全相同；不新增过滤")
        st.write("取消季度学习；不进行趋势/修复切换")
        st.write("基础评分、每项扣5分、每项扣10分同时输出")
        st.write("精确比较第1—5名，不用累计Top3掩盖顺序错误")
        st.write("删除MFE最高3只、5只后再次审计")
        cache = st.checkbox("使用逐股票缓存", value=True)
        pause = st.number_input("每次API调用后暂停(秒)", 0.0, 3.0, 0.12, 0.05)
        if st.button("清除本程序缓存"):
            if os.path.isdir(CACHE_DIR): shutil.rmtree(CACHE_DIR)
            st.success("缓存已清除")
    token = st.text_input("Tushare Token", type="password")
    if not token: st.info("请输入Tushare Token。"); return
    key = "nonlinear_experts_v23_zip"
    if not st.button("开始V2.3非线性评分验证", type="primary"):
        if key in st.session_state:
            st.download_button("下载上一次全部结果ZIP", st.session_state[key],
                               file_name="weekly_macd_nonlinear_experts_v2_3_all_results.zip", mime="application/zip")
        return
    if eval_date >= end_date or end_date > obs_date: st.error("日期关系不正确。"); return
    API_ERRORS = []; ts.set_token(token); pro = ts.pro_api()
    eval_start = pd.Timestamp(eval_date)
    research, end, obs = eval_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), obs_date.strftime("%Y%m%d")
    preload = (eval_date - timedelta(days=3*365)).strftime("%Y%m%d")
    config = {"signal_start": research, "signal_end": end, "market_end": obs, "preload_start": preload,
              "min_price":10.0, "min_mv":100.0, "max_mv":1_000_000_000.0, "price_tolerance_pct":3.0,
              "stop_threshold_pct":10.0, "buy_slippage_pct":0.20, "sell_slippage_pct":0.20,
              "sample_per_board":0, "sample_seed":DEFAULT_SAMPLE_SEED,
              "long_cycle_min_weeks":DEFAULT_LONG_CYCLE_MIN_WEEKS,
              "material_hist_change_pct":DEFAULT_MATERIAL_HIST_CHANGE_PCT,
              "short_strength_ratio":DEFAULT_SHORT_STRENGTH_RATIO}
    try:
        with st.spinner("加载基础数据和板块指数..."):
            opens=load_trade_calendar(preload,obs); full=load_trade_calendar(preload,(obs_date+timedelta(days=7)).strftime("%Y%m%d"))
            basic=load_stock_basic(); memberships=load_sw_tech_memberships(float(pause)); week_map=complete_week_last_dates(full)
            boards={}
            for code in sorted(set(BOARD_INDEX.values())):
                d=fetch_index_history(code,preload,obs,bool(cache),float(pause))
                if not d.empty: boards[code]=build_weekly(d,week_map)
    except Exception as exc: st.error(f"基础数据加载失败：{exc}"); return
    periods=build_period_index(memberships); codes=sorted(set(periods)&set(basic.ts_code.astype(str)))
    universe=basic[basic.ts_code.isin(codes)].copy(); stocks,audit,pop=build_stratified_sample(universe,periods,end,0,DEFAULT_SAMPLE_SEED)
    listed=stocks.list_date.apply(lambda x:normalize_date(x,"19000101")); delisted=stocks.delist_date.apply(lambda x:normalize_date(x,"99991231"))
    stocks=stocks[~listed.gt(end)&~delisted.lt(preload)].reset_index(drop=True)
    pos={d:i for i,d in enumerate(opens)}; records=[]; histories={}; rejects={}; hits=fails=0
    bar=st.progress(0.0); status=st.empty()
    for i,stock in stocks.iterrows():
        code=str(stock.ts_code); bar.progress((i+1)/len(stocks),text=f"{i+1}/{len(stocks)} {code}")
        status.caption(f"事件{len(records)}；缓存{hits}；失败{fails}")
        daily,db,hit=fetch_stock_history(code,preload,obs,bool(cache),float(pause)); hits+=int(hit)
        if daily.empty: fails+=1; continue
        rr,rj,_=analyze_stock(stock,periods.get(code,[]),daily,db,week_map,opens,pos,config); records.extend(rr)
        if rr: histories[code]=daily.copy()
        for reason,n in rj.items(): rejects[reason]=rejects.get(reason,0)+n
    bar.empty(); status.empty()
    if not records: st.error("没有生成事件。"); return
    with st.spinner("计算两个专家的三组非线性评分..."):
        events=pd.DataFrame(records).sort_values(["Signal_Date","ts_code","Event_Type"])
        opp=build_cycle_opportunities(events,histories,obs,config["sell_slippage_pct"])
        featured=prepare_features(opp,boards); candidates=featured[featured.Strict_Eligible.map(bool_value)].copy()
        rankings=build_rankings(candidates); summary=make_summary(rankings,eval_start)
    ec=candidates[candidates.Selection_Date_dt.ge(eval_start)]; er=rankings[rankings.Selection_Date_dt.ge(eval_start)]
    exact=exact_rank_quality(rankings,eval_start)
    acceptance=ranking_acceptance_audit(exact,candidates,eval_start)
    allocation=allocation_comparison(rankings,eval_start,DEFAULT_VARIANT)
    commonality=feature_commonality_audit(candidates,eval_start)
    default_top3=er[er.评分变体.eq(DEFAULT_VARIANT)&er.Expert_Rank.le(3)].copy()
    tail=default_top3[default_top3.Outcome_Mature.map(bool_value) & num(default_top3.CP_W2_Delayed_MFE_8W_pct).ge(50)].copy()
    tail=tail.sort_values("CP_W2_Delayed_MFE_8W_pct",ascending=False)
    component_columns=[c for c in er.columns if c.startswith("趋势_") or c.startswith("修复_")]
    component_detail=default_top3[["Selection_Date","ts_code","name","专家模型","评分变体",
                                  "Expert_Rank","Expert_Score","基础非线性得分","极端指标数","过热扣分",
                                  *component_columns,"BigMove_Target","Realised_Utility",
                                  "CP_W2_Delayed_Return_8W_pct","CP_W2_Delayed_MFE_8W_pct",
                                  "CP_W2_Delayed_MAE_8W_pct","CP_W2_Delayed_First_20_vs_Stop",
                                  "CP_W2_Delayed_First_30_vs_Stop","CP_W2_Delayed_First_50_vs_Stop",
                                  "CP_W2_Delayed_First_100_vs_Stop"]]
    reject=pd.DataFrame([{"剔除原因":k,"次数":v} for k,v in rejects.items()])
    total=summary[(summary.统计期=="全部")&(summary.方案==TREND)&
                  (summary.选择范围=="Top1")&(summary.评分变体==DEFAULT_VARIANT)].iloc[0]
    run=pd.DataFrame([{"程序":TITLE,"版本":VERSION,"评价开始":eval_date.strftime("%Y%m%d"),"信号开始":research,
                       "信号截止":end,"观察截止":obs,"评价候选":len(ec),"候选周":ec.Selection_Date.nunique(),
                       "评分变体数":len(SCORE_VARIANTS),"默认查看":DEFAULT_VARIANT,
                       "默认趋势Top1大涨目标":total.大涨目标均值,
                       "默认趋势Top1效用":total.交易效用均值,
                       "默认趋势Top1收益中位数(%)":total["8周收益中位数(%)"],
                       "行情失败":fails,"缓存命中":hits}])
    meta=pd.DataFrame([
        ("程序",TITLE),("硬条件","科技池、价≥10元、流通市值≥100亿元、第二根完整红柱严格扩张"),
        ("实验范围","只验证评分排序；关闭机器学习和自动模式切换"),
        ("趋势评分","13周强度30、板块强度30、成交量15、MA20斜率10、第二周涨幅10、红柱扩张5"),
        ("修复评分","收复MA20 25、板块强度20、DEA改善15、斜率改善15、成交量10、回调甜蜜区10、第二周涨幅5"),
        ("删除特征","26周收益、收盘位置、DEA绝对深度、前期亏损、修复交互"),
        ("过热A/B","超过两个90%分位指标后，分别不扣分、每项扣5分、每项扣10分"),
        ("评价目标","20%先于止损记60；30%再加25；50%再加15；翻倍只审计；止损先到-40"),
        ("影子观察","不在20%停止读取行情；同时记录20/30/50/100先于止损及8周MFE"),
        ("风险","ATR不混入Alpha；成交量进入两专家评分；换手率保留给后续仓位模块")],columns=["项目","值"])
    files={"01_run_summary_nonlinear_experts_v2_3.csv":run,
           "02_variant_year_comparison_nonlinear_experts_v2_3.csv":summary,
           "03_exact_rank_quality_nonlinear_experts_v2_3.csv":exact,
           "04_ranking_acceptance_audit_nonlinear_experts_v2_3.csv":acceptance,
           "05_parallel_allocation_comparison_nonlinear_experts_v2_3.csv":allocation,
           "06_default_top3_rankings_nonlinear_experts_v2_3.csv":default_top3,
           "07_default_top3_score_components_nonlinear_experts_v2_3.csv":component_detail,
           "08_score_dictionary_nonlinear_experts_v2_3.csv":feature_dict(),
           "09_feature_commonality_audit_nonlinear_experts_v2_3.csv":commonality,
           "10_default_top3_tail_50plus_nonlinear_experts_v2_3.csv":tail,
           "11_all_variant_rankings_nonlinear_experts_v2_3.csv":er,
           "12_evaluation_candidate_features_nonlinear_experts_v2_3.csv":ec,
           "13_all_event_features_nonlinear_experts_v2_3.csv":featured,
           "14_full_tech_universe_nonlinear_experts_v2_3.csv":audit,
           "15_population_nonlinear_experts_v2_3.csv":pop,
           "16_rejection_audit_nonlinear_experts_v2_3.csv":reject,
           "17_metadata_nonlinear_experts_v2_3.csv":meta}
    z=make_result_zip(files); st.session_state[key]=z
    st.success(f"完成：评价候选{len(ec)}个；两个专家×三组评分全部完成。")
    table=summary[(summary.统计期.eq("全部"))&(summary.评分变体.eq(view_variant))&(summary.方案.eq(view_model))]
    st.subheader("所选评分的Top1/Top2/Top3"); st.dataframe(table,use_container_width=True,hide_index=True)
    st.subheader("精确名次质量"); st.dataframe(exact[(exact.评分变体.eq(view_variant))&(exact.方案.eq(view_model))],use_container_width=True,hide_index=True)
    st.subheader("严格排序验收"); st.dataframe(acceptance,use_container_width=True,hide_index=True)
    st.subheader("年度比较"); st.dataframe(summary[(summary.统计期.ne("全部"))&(summary.评分变体.eq(view_variant))],use_container_width=True,hide_index=True)
    st.subheader("并行候选组合诊断"); st.dataframe(allocation,use_container_width=True,hide_index=True)
    st.download_button("下载全部结果ZIP",z,file_name="weekly_macd_nonlinear_experts_v2_3_all_results.zip",mime="application/zip",type="primary")
    st.warning("本版只验证评分顺序，不决定实盘模式；4周/8周影子切换器将在评分通过后单独验证。")


# ===== V2.5 two-feature / leave-one-year-out validation lab (single file) =====
TITLE = "科技股周线MACD三形态双特征跨年验证器 V2.5"
VERSION = "V2.5-THREE-STATE-TWO-FEATURE-LOYO"

STATE_ORDER = ("上升趋势", "中性趋势", "下降趋势")
LOSS_GROUP = "亏损且最高涨幅<30%"
LOW_PROFIT_GROUP = "盈利但最高涨幅<30%"
MID_GROUP = "30%～50%"
HIGH_GROUP = "50%～100%"
DOUBLE_GROUP = "翻倍以上"
GROUP_ORDER = (LOSS_GROUP, LOW_PROFIT_GROUP, MID_GROUP, HIGH_GROUP, DOUBLE_GROUP)

NUMERIC_MINING_FEATURES = {
    "流通市值(亿元)": "Circ_MV_Billion",
    "换手率(%)": "Turnover_Rate",
    "第一红柱前回调深度(%)": "Pullback_Depth_pct",
    "第一周涨幅(%)": "W1_Week_Return_pct",
    "第一周量比20": "W1_Volume_Ratio20",
    "第一周ATR14(%)": "W1_ATR14_pct",
    "第一周收盘位置": "W1_Close_Location",
    "第二红柱/第一红柱增幅(%)": "CP_W2_Hist_vs_W1_pct",
    "第二周DIF/价格(%)": "CP_W2_DIF_to_Price_pct",
    "第二周DEA/价格(%)": "CP_W2_DEA_to_Price_pct",
    "第二周MA20四周斜率(%)": "CP_W2_MA20_Slope4_pct",
    "第二周前13周涨幅(%)": "CP_W2_Return_13W_pct",
    "第二周前26周涨幅(%)": "CP_W2_Return_26W_pct",
    "第二周收盘/MA20(%)": "CP_W2_Close_vs_MA20_pct",
    "第二周涨幅(%)": "CP_W2_Week_Return_pct",
    "第二周量比20": "CP_W2_Volume_Ratio20",
    "第二周ATR14(%)": "CP_W2_ATR14_pct",
    "第二周收盘位置": "CP_W2_Close_Location",
    "第二周确认时浮盈(%)": "CP_W2_Return_From_Entry_pct",
    "板块13周涨幅(%)": "Board_Return_13W_pct",
    "板块MA20四周斜率(%)": "Board_MA20_Slope4_pct",
    "个股相对板块强度": "Board_RS",
    "MA20斜率改善": "Slope_Improve",
    "DEA位置改善": "DEA_Improve",
}

CATEGORICAL_MINING_FEATURES = {
    "上市板块": "Sample_Board",
    "零轴位置": "Zero_Axis",
    "申万一级行业": "SW_L1",
    "申万二级行业": "SW_L2",
}

CORE_COMPARISONS = (
    (MID_GROUP, LOSS_GROUP), (MID_GROUP, LOW_PROFIT_GROUP),
    (HIGH_GROUP, LOSS_GROUP), (HIGH_GROUP, LOW_PROFIT_GROUP),
    (DOUBLE_GROUP, LOSS_GROUP), (DOUBLE_GROUP, LOW_PROFIT_GROUP),
)

ADJACENT_COMPARISONS = (
    (MID_GROUP, LOW_PROFIT_GROUP),
    (HIGH_GROUP, MID_GROUP),
    (DOUBLE_GROUP, HIGH_GROUP),
)

RISK_COMPARISONS = ((LOSS_GROUP, LOW_PROFIT_GROUP),)


def assign_five_outcome_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """严格复现已确认口径：高涨幅按8周MFE分层，MFE<30再按交易效用分亏损/盈利。"""
    out = frame.copy()
    mfe = num(out.get("CP_W2_Delayed_MFE_8W_pct"), out.index)
    realised = num(out.get("Realised_Utility"), out.index)
    conditions = [
        mfe.lt(30) & realised.le(0),
        mfe.lt(30) & realised.gt(0),
        mfe.ge(30) & mfe.lt(50),
        mfe.ge(50) & mfe.lt(100),
        mfe.ge(100),
    ]
    out["结果组"] = np.select(conditions, GROUP_ORDER, default="未分组")
    out["结果组顺序"] = out["结果组"].map({name: i + 1 for i, name in enumerate(GROUP_ORDER)})
    out["三形态有效"] = out["Weekly_Trend"].isin(STATE_ORDER)
    return out


def _safe_ratio(a: float, b: float) -> float:
    return float(a / b) if math.isfinite(a) and math.isfinite(b) and abs(b) > 1e-12 else np.nan


def _auc(values: pd.Series, labels: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce")
    y = labels.astype(bool)
    ok = x.notna() & y.notna()
    x, y = x.loc[ok], y.loc[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = x.rank(method="average")
    return float((ranks.loc[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _effect_size(target: pd.Series, control: pd.Series) -> float:
    a = pd.to_numeric(target, errors="coerce").dropna()
    b = pd.to_numeric(control, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                       max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else np.nan


def _year_direction(frame: pd.DataFrame, feature: str, target_group: str,
                    control_group: str, full_direction: int) -> tuple[int, int]:
    same = comparable = 0
    for _, year in frame.groupby("Selection_Year"):
        a = num(year.loc[year["结果组"].eq(target_group), feature]).dropna()
        b = num(year.loc[year["结果组"].eq(control_group), feature]).dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        diff = a.median() - b.median()
        if not math.isfinite(diff) or abs(diff) < 1e-12:
            continue
        comparable += 1
        same += int((1 if diff > 0 else -1) == full_direction)
    return same, comparable


def numeric_feature_comparisons(frame: pd.DataFrame,
                                comparisons: tuple[tuple[str, str], ...],
                                comparison_set: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in STATE_ORDER:
        state_frame = frame[frame["Weekly_Trend"].eq(state)].copy()
        for target_group, control_group in comparisons:
            target = state_frame[state_frame["结果组"].eq(target_group)]
            control = state_frame[state_frame["结果组"].eq(control_group)]
            for label, feature in NUMERIC_MINING_FEATURES.items():
                if feature not in state_frame:
                    continue
                a = num(target[feature]).dropna()
                b = num(control[feature]).dropna()
                if len(a) < 2 or len(b) < 2:
                    continue
                median_diff = float(a.median() - b.median())
                direction = 1 if median_diff >= 0 else -1
                combined = pd.concat([a, b], ignore_index=True)
                threshold = float(combined.quantile(0.75 if direction > 0 else 0.25))
                a_hit = a.ge(threshold) if direction > 0 else a.le(threshold)
                b_hit = b.ge(threshold) if direction > 0 else b.le(threshold)
                coverage = float(a_hit.mean())
                false_positive = float(b_hit.mean())
                precision = _safe_ratio(float(a_hit.sum()), float(a_hit.sum() + b_hit.sum()))
                raw_auc = _auc(pd.concat([a, b], ignore_index=True),
                               pd.Series([True] * len(a) + [False] * len(b)))
                oriented_auc = raw_auc if direction > 0 else (1.0 - raw_auc)
                same, comparable = _year_direction(
                    state_frame, feature, target_group, control_group, direction
                )
                if min(len(a), len(b)) >= 30 and comparable >= 2 and same == comparable:
                    reliability = "较高"
                elif min(len(a), len(b)) >= 15 and (comparable == 0 or same >= max(1, comparable - 1)):
                    reliability = "中等"
                elif min(len(a), len(b)) >= 5:
                    reliability = "探索"
                else:
                    reliability = "案例"
                rows.append({
                    "对比集合": comparison_set, "红柱形态": state,
                    "目标组": target_group, "对照组": control_group,
                    "特征": label, "字段": feature,
                    "目标样本": len(a), "对照样本": len(b),
                    "目标中位数": a.median(), "对照中位数": b.median(),
                    "中位数差": median_diff, "有利方向": "较高" if direction > 0 else "较低",
                    "标准化效应": _effect_size(a, b), "方向化AUC": oriented_auc,
                    "固定分位阈值": threshold, "目标覆盖率(%)": coverage * 100.0,
                    "对照误入率(%)": false_positive * 100.0,
                    "阈值命中精度(%)": precision * 100.0 if math.isfinite(precision) else np.nan,
                    "覆盖/误入倍数": _safe_ratio(coverage, false_positive),
                    "年度同方向数": same, "年度可比较数": comparable,
                    "证据等级": reliability,
                })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["红柱形态", "目标组", "对照组", "方向化AUC", "目标覆盖率(%)"],
        ascending=[True, True, True, False, False],
    )


def categorical_feature_comparisons(frame: pd.DataFrame,
                                    comparisons: tuple[tuple[str, str], ...],
                                    comparison_set: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in STATE_ORDER:
        sf = frame[frame["Weekly_Trend"].eq(state)].copy()
        for target_group, control_group in comparisons:
            a = sf[sf["结果组"].eq(target_group)]
            b = sf[sf["结果组"].eq(control_group)]
            for label, feature in CATEGORICAL_MINING_FEATURES.items():
                if feature not in sf:
                    continue
                av = a[feature].fillna("未知").astype(str)
                bv = b[feature].fillna("未知").astype(str)
                levels = sorted(set(av) | set(bv))
                for level in levels:
                    ah, bh = int(av.eq(level).sum()), int(bv.eq(level).sum())
                    if ah + bh < 5:
                        continue
                    ar = ah / len(av) if len(av) else np.nan
                    br = bh / len(bv) if len(bv) else np.nan
                    precision = _safe_ratio(float(ah), float(ah + bh))
                    full_sign = 1 if ar >= br else -1
                    same = comparable = 0
                    for _, year in sf.groupby("Selection_Year"):
                        ya = year[year["结果组"].eq(target_group)]
                        yb = year[year["结果组"].eq(control_group)]
                        if len(ya) < 5 or len(yb) < 5:
                            continue
                        yar = ya[feature].fillna("未知").astype(str).eq(level).mean()
                        ybr = yb[feature].fillna("未知").astype(str).eq(level).mean()
                        if abs(yar - ybr) < 1e-12:
                            continue
                        comparable += 1
                        same += int((1 if yar > ybr else -1) == full_sign)
                    if min(len(av), len(bv)) >= 30 and comparable >= 2 and same == comparable:
                        reliability = "较高"
                    elif min(len(av), len(bv)) >= 15 and (comparable == 0 or same >= max(1, comparable - 1)):
                        reliability = "中等"
                    elif min(len(av), len(bv)) >= 5:
                        reliability = "探索"
                    else:
                        reliability = "案例"
                    rows.append({
                        "对比集合": comparison_set, "红柱形态": state,
                        "目标组": target_group, "对照组": control_group,
                        "特征": label, "字段": feature, "类别": level,
                        "目标样本": len(av), "对照样本": len(bv),
                        "目标类别数": ah, "对照类别数": bh,
                        "目标覆盖率(%)": ar * 100.0, "对照误入率(%)": br * 100.0,
                        "阈值命中精度(%)": precision * 100.0 if math.isfinite(precision) else np.nan,
                        "覆盖/误入倍数": _safe_ratio(ar, br),
                        "年度同方向数": same, "年度可比较数": comparable,
                        "证据等级": reliability,
                    })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["红柱形态", "目标组", "对照组", "覆盖/误入倍数", "目标覆盖率(%)"],
        ascending=[True, True, True, False, False],
    )


def five_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in STATE_ORDER:
        sf = frame[frame["Weekly_Trend"].eq(state)]
        for group_name in GROUP_ORDER:
            g = sf[sf["结果组"].eq(group_name)]
            mfe = num(g.get("CP_W2_Delayed_MFE_8W_pct"), g.index)
            mae = num(g.get("CP_W2_Delayed_MAE_8W_pct"), g.index)
            ending = num(g.get("CP_W2_Delayed_Return_8W_pct"), g.index)
            rows.append({
                "红柱形态": state, "结果组": group_name, "样本数": len(g),
                "不同股票": g.ts_code.nunique() if "ts_code" in g else np.nan,
                "涉及周数": g.Selection_Date.nunique() if "Selection_Date" in g else np.nan,
                "MFE均值(%)": mfe.mean(), "MFE中位数(%)": mfe.median(),
                "MAE均值(%)": mae.mean(), "MAE中位数(%)": mae.median(),
                "8周末收益均值(%)": ending.mean(), "8周末收益中位数(%)": ending.median(),
                "8周末盈利率(%)": rate(ending.gt(0)),
                "触及-10%(%)": rate(g.CP_W2_Delayed_Hit_Stop_8W.map(bool_value)) if len(g) else np.nan,
            })
    return pd.DataFrame(rows)


def year_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(frame.Selection_Year.dropna().unique()):
        yf = frame[frame.Selection_Year.eq(year)]
        for state in STATE_ORDER:
            sf = yf[yf.Weekly_Trend.eq(state)]
            for group_name in GROUP_ORDER:
                g = sf[sf["结果组"].eq(group_name)]
                rows.append({
                    "年份": year, "红柱形态": state, "结果组": group_name,
                    "样本数": len(g), "不同股票": g.ts_code.nunique() if len(g) else 0,
                    "占该形态比例(%)": len(g) / len(sf) * 100.0 if len(sf) else np.nan,
                    "MFE中位数(%)": num(g.get("CP_W2_Delayed_MFE_8W_pct"), g.index).median(),
                    "8周末收益中位数(%)": num(g.get("CP_W2_Delayed_Return_8W_pct"), g.index).median(),
                })
    return pd.DataFrame(rows)


def path_audit(frame: pd.DataFrame) -> pd.DataFrame:
    target_by_group = {MID_GROUP: 30, HIGH_GROUP: 50, DOUBLE_GROUP: 100}
    rows = []
    for state in STATE_ORDER:
        for group_name, target in target_by_group.items():
            g = frame[frame.Weekly_Trend.eq(state) & frame["结果组"].eq(group_name)]
            status = g.get(f"CP_W2_Delayed_First_{target}_vs_Stop", pd.Series("", index=g.index)).astype(str)
            target_first_mask = status.eq("目标先到")
            stop_first_mask = status.isin(["止损先到", "同日不确定_按止损"])
            rows.append({
                "红柱形态": state, "结果组": group_name, "目标涨幅(%)": target,
                "样本数": len(g), "目标先到数": int(target_first_mask.sum()),
                "目标先到比例(%)": rate(target_first_mask),
                "止损先到后来达到数": int(stop_first_mask.sum()),
                "止损先到后来达到比例(%)": rate(stop_first_mask),
                "MFE中位数(%)": num(g.get("CP_W2_Delayed_MFE_8W_pct"), g.index).median(),
                "MAE中位数(%)": num(g.get("CP_W2_Delayed_MAE_8W_pct"), g.index).median(),
            })
    return pd.DataFrame(rows)


def feature_gradient_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in STATE_ORDER:
        sf = frame[frame.Weekly_Trend.eq(state)].copy()
        for label, feature in NUMERIC_MINING_FEATURES.items():
            values = num(sf.get(feature), sf.index)
            valid = values.notna()
            if valid.sum() < 20 or values.loc[valid].nunique() < 5:
                continue
            try:
                bins = pd.qcut(values.loc[valid], q=5, labels=False, duplicates="drop")
            except ValueError:
                continue
            work = sf.loc[valid].copy()
            work["特征分位组"] = bins.astype(int) + 1
            for bucket, g in work.groupby("特征分位组"):
                rows.append({
                    "红柱形态": state, "特征": label, "字段": feature,
                    "特征分位组": int(bucket), "样本数": len(g),
                    "特征最小值": num(g[feature]).min(), "特征最大值": num(g[feature]).max(),
                    "亏损比例(%)": rate(g["结果组"].eq(LOSS_GROUP)),
                    "盈利不足30%比例(%)": rate(g["结果组"].eq(LOW_PROFIT_GROUP)),
                    "30%～50%比例(%)": rate(g["结果组"].eq(MID_GROUP)),
                    "50%～100%比例(%)": rate(g["结果组"].eq(HIGH_GROUP)),
                    "翻倍比例(%)": rate(g["结果组"].eq(DOUBLE_GROUP)),
                    "50%以上比例(%)": rate(g["结果组"].isin([HIGH_GROUP, DOUBLE_GROUP])),
                    "MFE中位数(%)": num(g.CP_W2_Delayed_MFE_8W_pct).median(),
                })
    return pd.DataFrame(rows)


def candidate_distinguishers(numeric_core: pd.DataFrame,
                             numeric_adjacent: pd.DataFrame,
                             categorical_core: pd.DataFrame,
                             categorical_adjacent: pd.DataFrame) -> pd.DataFrame:
    numeric = pd.concat([numeric_core, numeric_adjacent], ignore_index=True)
    if not numeric.empty:
        numeric = numeric.copy()
        numeric["候选证据分"] = (
            (num(numeric["方向化AUC"]) - 0.5).clip(lower=0) * 100.0
            + (num(numeric["目标覆盖率(%)"]) - num(numeric["对照误入率(%)"])).clip(lower=0)
        )
        numeric["类别"] = ""
        numeric["证据类型"] = "数值"
    categorical = pd.concat([categorical_core, categorical_adjacent], ignore_index=True)
    if not categorical.empty:
        categorical = categorical.copy()
        categorical["候选证据分"] = (
            num(categorical["目标覆盖率(%)"]) - num(categorical["对照误入率(%)"])
        ).clip(lower=0)
        categorical["证据类型"] = "类别"
        categorical["有利方向"] = "属于该类别"
    common = ["对比集合", "红柱形态", "目标组", "对照组", "特征", "字段", "类别",
              "证据类型", "目标样本", "对照样本", "有利方向", "目标覆盖率(%)",
              "对照误入率(%)", "阈值命中精度(%)", "覆盖/误入倍数",
              "年度同方向数", "年度可比较数", "证据等级", "候选证据分"]
    parts = []
    for data in (numeric, categorical):
        if data.empty:
            continue
        for col in common:
            if col not in data:
                data[col] = np.nan
        parts.append(data[common])
    if not parts:
        return pd.DataFrame(columns=common)
    result = pd.concat(parts, ignore_index=True)
    result = result[result["候选证据分"].gt(0)].copy()
    result["组内审计排名"] = result.groupby(
        ["对比集合", "红柱形态", "目标组", "对照组"]
    )["候选证据分"].rank(method="first", ascending=False)
    return result.sort_values(
        ["对比集合", "红柱形态", "目标组", "对照组", "组内审计排名"]
    )


def mining_feature_dictionary() -> pd.DataFrame:
    rows = []
    for label, field in NUMERIC_MINING_FEATURES.items():
        rows.append({"特征类型": "数值", "特征": label, "字段": field,
                     "使用时点": "第二根完整红柱确认时已知", "是否参与评分": "否"})
    for label, field in CATEGORICAL_MINING_FEATURES.items():
        rows.append({"特征类型": "类别", "特征": label, "字段": field,
                     "使用时点": "第二根完整红柱确认时已知", "是否参与评分": "否"})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# V2.5：双特征组合与整年留出验证
# -----------------------------------------------------------------------------
V25_TASKS = (
    {
        "任务": "避免亏损",
        "目标组": (LOW_PROFIT_GROUP,),
        "对照组": (LOSS_GROUP,),
        "含义": "在MFE都不足30%的股票中，区分盈利与亏损",
        "正式验收": True,
    },
    {
        "任务": "达到30%基础",
        "目标组": (MID_GROUP, HIGH_GROUP, DOUBLE_GROUP),
        "对照组": (LOW_PROFIT_GROUP,),
        "含义": "在已经避免亏损后，区分能否进入30%以上行情",
        "正式验收": True,
    },
    {
        "任务": "进入50%强化",
        "目标组": (HIGH_GROUP, DOUBLE_GROUP),
        "对照组": (MID_GROUP,),
        "含义": "在已经达到30%的股票中，区分能否继续进入50%以上行情",
        "正式验收": True,
    },
    {
        "任务": "翻倍附加",
        "目标组": (DOUBLE_GROUP,),
        "对照组": (HIGH_GROUP,),
        "含义": "50%～100%与翻倍组的探索性比较；样本不足，不作正式验收",
        "正式验收": False,
    },
)

# 所有规则形状在看留出年以前固定。阈值只能由训练年份的分位数生成。
V25_RULE_SHAPES = (
    ("低端40%", None, 0.40),
    ("低端20%", None, 0.20),
    ("高端40%", 0.60, None),
    ("高端20%", 0.80, None),
    ("中间60%", 0.20, 0.80),
    ("中间40%", 0.30, 0.70),
    ("偏弱中段", 0.20, 0.60),
    ("偏强中段", 0.40, 0.80),
    ("排除最高20%", None, 0.80),
    ("排除最低20%", 0.20, None),
)

V25_MIN_TRAIN_EACH = 15
V25_MIN_TEST_EACH = 5
V25_TOP_SINGLE_FEATURES = 10
V25_TOP_PAIRS_PER_FOLD = 10
V25_MAX_PAIR_CORRELATION = 0.90


def v25_task_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "分层任务": task["任务"],
            "目标组": " + ".join(task["目标组"]),
            "对照组": " + ".join(task["对照组"]),
            "研究含义": task["含义"],
            "是否正式验收": "是" if task["正式验收"] else "否，只作探索",
        }
        for task in V25_TASKS
    ])


def v25_rule_mask(values: pd.Series, lower: float, upper: float) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    selected = pd.Series(True, index=x.index, dtype=bool)
    if math.isfinite(lower):
        selected &= x.ge(lower)
    if math.isfinite(upper):
        selected &= x.le(upper)
    return selected & x.notna()


def v25_binary_metrics(labels: pd.Series, selected: pd.Series) -> dict[str, Any]:
    y = labels.astype(bool)
    picked = selected.reindex(y.index, fill_value=False).astype(bool)
    target_n = int(y.sum())
    control_n = int((~y).sum())
    selected_target = int((picked & y).sum())
    selected_control = int((picked & ~y).sum())
    selected_n = selected_target + selected_control
    baseline = target_n / (target_n + control_n) if target_n + control_n else np.nan
    precision = selected_target / selected_n if selected_n else np.nan
    target_coverage = selected_target / target_n if target_n else np.nan
    control_entry = selected_control / control_n if control_n else np.nan
    lift = precision / baseline if selected_n and baseline > 0 else np.nan
    return {
        "目标样本": target_n,
        "对照样本": control_n,
        "入选目标": selected_target,
        "误入对照": selected_control,
        "入选总数": selected_n,
        "基准目标率(%)": baseline * 100.0 if math.isfinite(baseline) else np.nan,
        "入选目标率(%)": precision * 100.0 if math.isfinite(precision) else np.nan,
        "目标覆盖率(%)": target_coverage * 100.0 if math.isfinite(target_coverage) else np.nan,
        "对照误入率(%)": control_entry * 100.0 if math.isfinite(control_entry) else np.nan,
        "相对基准提升倍数": lift,
    }


def v25_training_year_consistency(frame: pd.DataFrame,
                                  selected: pd.Series) -> tuple[int, int]:
    work = frame.copy()
    work["_Selected"] = selected.reindex(work.index, fill_value=False).astype(bool)
    improved = comparable = 0
    for _, year in work.groupby("Selection_Year"):
        labels = year["_Target"].astype(bool)
        if int(labels.sum()) < 5 or int((~labels).sum()) < 5:
            continue
        metrics = v25_binary_metrics(labels, year["_Selected"])
        if metrics["入选总数"] < 3 or not math.isfinite(metrics["相对基准提升倍数"]):
            continue
        comparable += 1
        improved += int(metrics["相对基准提升倍数"] > 1.0)
    return improved, comparable


def v25_learn_single_rules(train: pd.DataFrame) -> pd.DataFrame:
    """只用训练年份选择每个数值特征的最佳预设形状与训练分位阈值。"""
    rows: list[dict[str, Any]] = []
    labels = train["_Target"].astype(bool)
    for feature_label, field in NUMERIC_MINING_FEATURES.items():
        if field not in train:
            continue
        values = pd.to_numeric(train[field], errors="coerce")
        if values.notna().sum() < 30 or values.nunique(dropna=True) < 8:
            continue
        for shape_name, lower_q, upper_q in V25_RULE_SHAPES:
            lower = float(values.quantile(lower_q)) if lower_q is not None else np.nan
            upper = float(values.quantile(upper_q)) if upper_q is not None else np.nan
            selected = v25_rule_mask(values, lower, upper)
            metrics = v25_binary_metrics(labels, selected)
            target_coverage = metrics["目标覆盖率(%)"] / 100.0
            edge = (metrics["目标覆盖率(%)"] - metrics["对照误入率(%)"]) / 100.0
            lift = metrics["相对基准提升倍数"]
            if (target_coverage < 0.20 or metrics["入选总数"] < 8
                    or not math.isfinite(lift) or edge <= 0 or lift <= 1.02):
                continue
            improved, comparable = v25_training_year_consistency(train, selected)
            consistency = improved / comparable if comparable else 0.0
            score = edge * 100.0 + (lift - 1.0) * 20.0 + consistency * 5.0
            rows.append({
                "特征": feature_label, "字段": field, "规则形状": shape_name,
                "训练下限": lower, "训练上限": upper,
                **metrics,
                "训练改善年份": improved, "训练可比较年份": comparable,
                "训练规则分": score,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # 每个字段只留下一个形状，避免同一指标用十种切法制造虚假候选。
    result = result.sort_values("训练规则分", ascending=False)
    result = result.drop_duplicates("字段", keep="first")
    return result.head(V25_TOP_SINGLE_FEATURES).reset_index(drop=True)


def v25_apply_rule(frame: pd.DataFrame, rule: pd.Series | dict[str, Any]) -> pd.Series:
    field = str(rule["字段"])
    lower = finite_num(rule.get("训练下限", np.nan))
    upper = finite_num(rule.get("训练上限", np.nan))
    return v25_rule_mask(frame.get(field, pd.Series(np.nan, index=frame.index)), lower, upper)


def v25_learn_pairs(train: pd.DataFrame,
                    single_rules: pd.DataFrame) -> pd.DataFrame:
    """在训练年内组合两个不同字段；相关性过高的重复指标禁止配对。"""
    if len(single_rules) < 2:
        return pd.DataFrame()
    labels = train["_Target"].astype(bool)
    rows: list[dict[str, Any]] = []
    for left_i in range(len(single_rules)):
        for right_i in range(left_i + 1, len(single_rules)):
            left = single_rules.iloc[left_i]
            right = single_rules.iloc[right_i]
            left_values = pd.to_numeric(train[left["字段"]], errors="coerce")
            right_values = pd.to_numeric(train[right["字段"]], errors="coerce")
            correlation = left_values.corr(right_values, method="spearman")
            if math.isfinite(correlation) and abs(correlation) > V25_MAX_PAIR_CORRELATION:
                continue
            selected = v25_apply_rule(train, left) & v25_apply_rule(train, right)
            metrics = v25_binary_metrics(labels, selected)
            target_coverage = metrics["目标覆盖率(%)"] / 100.0
            lift = metrics["相对基准提升倍数"]
            if (target_coverage < 0.10 or metrics["入选总数"] < 6
                    or not math.isfinite(lift) or lift <= 1.02):
                continue
            improved, comparable = v25_training_year_consistency(train, selected)
            consistency = improved / comparable if comparable else 0.0
            edge = (metrics["目标覆盖率(%)"] - metrics["对照误入率(%)"]) / 100.0
            precision_edge = metrics["入选目标率(%)"] - metrics["基准目标率(%)"]
            score = precision_edge + edge * 50.0 + consistency * 5.0
            signature_parts = sorted([
                (str(left["字段"]), str(left["特征"]), str(left["规则形状"])),
                (str(right["字段"]), str(right["特征"]), str(right["规则形状"])),
            ])
            rows.append({
                "特征1": left["特征"], "字段1": left["字段"],
                "形状1": left["规则形状"], "下限1": left["训练下限"], "上限1": left["训练上限"],
                "特征2": right["特征"], "字段2": right["字段"],
                "形状2": right["规则形状"], "下限2": right["训练下限"], "上限2": right["训练上限"],
                "组合签名": f"{signature_parts[0][0]}[{signature_parts[0][2]}] + "
                            f"{signature_parts[1][0]}[{signature_parts[1][2]}]",
                "字段组合": " + ".join(sorted([str(left["字段"]), str(right["字段"])])),
                "训练Spearman相关": correlation,
                **metrics,
                "训练改善年份": improved, "训练可比较年份": comparable,
                "训练组合分": score,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["训练组合分", "相对基准提升倍数", "目标覆盖率(%)"],
        ascending=False,
    ).head(V25_TOP_PAIRS_PER_FOLD).reset_index(drop=True)
    result["训练排名"] = np.arange(1, len(result) + 1)
    return result


def v25_apply_pair(frame: pd.DataFrame, pair: pd.Series | dict[str, Any]) -> pd.Series:
    left = {
        "字段": pair["字段1"], "训练下限": pair.get("下限1", np.nan),
        "训练上限": pair.get("上限1", np.nan),
    }
    right = {
        "字段": pair["字段2"], "训练下限": pair.get("下限2", np.nan),
        "训练上限": pair.get("上限2", np.nan),
    }
    return v25_apply_rule(frame, left) & v25_apply_rule(frame, right)


def v25_prepare_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "Selection_Date" not in out:
        raise ValueError("输入缺少 Selection_Date")
    out["Selection_Date"] = out["Selection_Date"].map(normalize_date)
    out["Selection_Date_dt"] = pd.to_datetime(
        out["Selection_Date"], format="%Y%m%d", errors="coerce"
    )
    out["Selection_Year"] = out["Selection_Date"].str[:4]
    required = {
        "Weekly_Trend", "CP_W2_Delayed_MFE_8W_pct", "Realised_Utility",
        *NUMERIC_MINING_FEATURES.values(),
    }
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError("输入缺少V2.5所需字段：" + "、".join(missing))
    out = assign_five_outcome_groups(out)
    out = out[
        out["三形态有效"] & out["结果组"].isin(GROUP_ORDER)
        & out["Selection_Date_dt"].notna()
    ].copy()
    identity = [c for c in ("Cycle_ID", "Selection_Date", "ts_code") if c in out]
    if identity:
        out = out.drop_duplicates(identity, keep="first")
    return out.sort_values(["Selection_Date", "ts_code"]).reset_index(drop=True)


def v25_load_v24_upload(uploaded_file: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = uploaded_file.getvalue()
    input_summary = pd.DataFrame()
    if str(uploaded_file.name).lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            names = archive.namelist()
            candidate_names = [name for name in names if
                               "15_all_evaluation_candidate_features" in name]
            if not candidate_names:
                candidate_names = [name for name in names if
                                   "all_evaluation_candidate_features" in name]
            if not candidate_names:
                raise ValueError("ZIP中没有找到V2.4的15号候选特征文件")
            with archive.open(candidate_names[0]) as source:
                candidates = pd.read_csv(source, low_memory=False)
            summary_names = [name for name in names if "01_run_summary" in name]
            if summary_names:
                with archive.open(summary_names[0]) as source:
                    input_summary = pd.read_csv(source, low_memory=False)
    else:
        candidates = pd.read_csv(io.BytesIO(raw), low_memory=False)
    return v25_prepare_candidates(candidates), input_summary


def v25_run_loyo(candidates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """逐形态、逐层级、逐年留出；留出年的标签不参与规则和阈值选择。"""
    fold_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    years = sorted(candidates["Selection_Year"].dropna().astype(str).unique())
    for state in STATE_ORDER:
        state_frame = candidates[candidates["Weekly_Trend"].eq(state)].copy()
        for task in V25_TASKS:
            target_groups = tuple(task["目标组"])
            control_groups = tuple(task["对照组"])
            allowed_groups = target_groups + control_groups
            task_frame = state_frame[state_frame["结果组"].isin(allowed_groups)].copy()
            task_frame["_Target"] = task_frame["结果组"].isin(target_groups)
            for holdout in years:
                fold_id = f"{state}|{task['任务']}|留出{holdout}"
                train = task_frame[task_frame["Selection_Year"].ne(holdout)].copy()
                test = task_frame[task_frame["Selection_Year"].eq(holdout)].copy()
                train_target = int(train["_Target"].sum())
                train_control = int((~train["_Target"]).sum())
                test_target = int(test["_Target"].sum())
                test_control = int((~test["_Target"]).sum())
                min_train = 5 if not task["正式验收"] else V25_MIN_TRAIN_EACH
                min_test = 2 if not task["正式验收"] else V25_MIN_TEST_EACH
                eligible = (
                    train_target >= min_train and train_control >= min_train
                    and test_target >= min_test and test_control >= min_test
                )
                fold_row = {
                    "折叠ID": fold_id, "红柱形态": state, "分层任务": task["任务"],
                    "留出年份": holdout, "正式验收任务": bool(task["正式验收"]),
                    "训练目标": train_target, "训练对照": train_control,
                    "测试目标": test_target, "测试对照": test_control,
                    "样本门槛通过": eligible,
                    "处理状态": "样本不足" if not eligible else "待训练",
                }
                if not eligible:
                    fold_rows.append(fold_row)
                    continue
                single = v25_learn_single_rules(train)
                if single.empty:
                    fold_row["处理状态"] = "训练年没有合格单特征"
                    fold_rows.append(fold_row)
                    continue
                single = single.copy()
                single.insert(0, "折叠ID", fold_id)
                single.insert(1, "红柱形态", state)
                single.insert(2, "分层任务", task["任务"])
                single.insert(3, "留出年份", holdout)
                single["训练单特征排名"] = np.arange(1, len(single) + 1)
                for _, rule in single.iterrows():
                    test_metrics = v25_binary_metrics(test["_Target"], v25_apply_rule(test, rule))
                    row = rule.to_dict()
                    row.update({f"测试_{key}": value for key, value in test_metrics.items()})
                    single_rows.append(row)
                pairs = v25_learn_pairs(train, single)
                if pairs.empty:
                    fold_row["处理状态"] = "训练年没有合格双特征组合"
                    fold_rows.append(fold_row)
                    continue
                fold_row["处理状态"] = "完成"
                fold_row["训练单特征数"] = len(single)
                fold_row["训练组合数"] = len(pairs)
                fold_rows.append(fold_row)
                best_single = single.iloc[0]
                best_single_test = v25_binary_metrics(
                    test["_Target"], v25_apply_rule(test, best_single)
                )
                for _, pair in pairs.iterrows():
                    test_selected = v25_apply_pair(test, pair)
                    test_metrics = v25_binary_metrics(test["_Target"], test_selected)
                    row = pair.to_dict()
                    row.update({
                        "折叠ID": fold_id, "红柱形态": state,
                        "分层任务": task["任务"], "留出年份": holdout,
                        "正式验收任务": bool(task["正式验收"]),
                        "测试判卷有效": test_metrics["入选总数"] >= (
                            5 if task["正式验收"] else 2
                        ),
                        **{f"测试_{key}": value for key, value in test_metrics.items()},
                    })
                    pair_rows.append(row)
                    if int(pair["训练排名"]) == 1:
                        best = row.copy()
                        best.update({
                            "最佳单特征": best_single["特征"],
                            "最佳单特征形状": best_single["规则形状"],
                            **{f"单特征测试_{key}": value
                               for key, value in best_single_test.items()},
                        })
                        best_rows.append(best)
                        picked = test[test_selected].copy()
                        for _, event in picked.iterrows():
                            selected_rows.append({
                                "折叠ID": fold_id, "红柱形态": state,
                                "分层任务": task["任务"], "留出年份": holdout,
                                "组合签名": pair["组合签名"],
                                "是否目标组": bool(event["_Target"]),
                                "结果组": event["结果组"],
                                "Selection_Date": event.get("Selection_Date", ""),
                                "ts_code": event.get("ts_code", ""),
                                "name": event.get("name", ""),
                                "MFE8周(%)": event.get("CP_W2_Delayed_MFE_8W_pct", np.nan),
                                "MAE8周(%)": event.get("CP_W2_Delayed_MAE_8W_pct", np.nan),
                                "8周末收益(%)": event.get("CP_W2_Delayed_Return_8W_pct", np.nan),
                            })
    folds = pd.DataFrame(fold_rows)
    singles = pd.DataFrame(single_rows)
    pairs = pd.DataFrame(pair_rows)
    best = pd.DataFrame(best_rows)
    selected = pd.DataFrame(selected_rows)
    aggregate = v25_aggregate_acceptance(best)
    stability = v25_pair_stability(pairs)
    return {
        "folds": folds, "singles": singles, "pairs": pairs,
        "best": best, "aggregate": aggregate,
        "stability": stability, "selected": selected,
    }


def v25_aggregate_acceptance(best: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if best.empty:
        return pd.DataFrame()
    for (state, task_name), group in best.groupby(["红柱形态", "分层任务"]):
        target_n = int(num(group["测试_目标样本"]).sum())
        control_n = int(num(group["测试_对照样本"]).sum())
        selected_target = int(num(group["测试_入选目标"]).sum())
        selected_control = int(num(group["测试_误入对照"]).sum())
        selected_n = selected_target + selected_control
        baseline = target_n / (target_n + control_n) if target_n + control_n else np.nan
        precision = selected_target / selected_n if selected_n else np.nan
        coverage = selected_target / target_n if target_n else np.nan
        lift = precision / baseline if selected_n and baseline > 0 else np.nan
        adequate_mask = num(group["测试_入选总数"]).ge(5)
        adequate_folds = int(adequate_mask.sum())
        improved = int((adequate_mask & num(group["测试_相对基准提升倍数"]).gt(1.0)).sum())
        folds = len(group)
        single_target = int(num(group["单特征测试_入选目标"]).sum())
        single_control = int(num(group["单特征测试_误入对照"]).sum())
        single_n = single_target + single_control
        single_precision = single_target / single_n if single_n else np.nan
        formal = bool(group["正式验收任务"].iloc[0])
        needed_improved = math.ceil(folds * 2 / 3)
        if not formal:
            verdict = "探索层：禁止正式通过"
        elif folds < 3:
            verdict = "样本不足：少于3个可判卷年份"
        elif adequate_folds < 3:
            verdict = "样本不足：少于3年各自入选至少5只"
        elif selected_n < 20:
            verdict = "样本不足：留出年合计入选少于20"
        elif improved < needed_improved:
            verdict = "未通过：跨年方向不稳定"
        elif not math.isfinite(lift) or lift < 1.15:
            verdict = "未通过：相对基准提升不足15%"
        elif not math.isfinite(coverage) or coverage < 0.12:
            verdict = "未通过：目标覆盖率不足12%"
        elif math.isfinite(single_precision) and precision <= single_precision:
            verdict = "未通过：双特征未优于最佳单特征"
        else:
            verdict = "通过候选验收，可进入下一阶段"
        rows.append({
            "红柱形态": state, "分层任务": task_name,
            "可判卷年份": folds, "入选至少5只年份": adequate_folds, "改善年份": improved,
            "要求改善年份": needed_improved,
            "测试目标样本": target_n, "测试对照样本": control_n,
            "双特征入选数": selected_n,
            "双特征目标率(%)": precision * 100.0 if math.isfinite(precision) else np.nan,
            "未筛选基准目标率(%)": baseline * 100.0 if math.isfinite(baseline) else np.nan,
            "双特征OOS提升倍数": lift,
            "双特征目标覆盖率(%)": coverage * 100.0 if math.isfinite(coverage) else np.nan,
            "最佳单特征目标率(%)": single_precision * 100.0
            if math.isfinite(single_precision) else np.nan,
            "验收结论": verdict,
        })
    return pd.DataFrame(rows).sort_values(["红柱形态", "分层任务"])


def v25_pair_stability(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    top = pairs[num(pairs["训练排名"]).le(5)].copy()
    top["测试改善"] = (
        num(top["测试_入选总数"]).ge(5)
        & num(top["测试_相对基准提升倍数"]).gt(1.0)
    )
    rows = []
    for keys, group in top.groupby(["红柱形态", "分层任务", "组合签名", "字段组合"]):
        rows.append({
            "红柱形态": keys[0], "分层任务": keys[1],
            "组合签名": keys[2], "字段组合": keys[3],
            "进入训练前5次数": len(group),
            "涉及留出年份": group["留出年份"].nunique(),
            "测试改善次数": int(group["测试改善"].sum()),
            "平均训练提升倍数": num(group["相对基准提升倍数"]).mean(),
            "平均OOS提升倍数": num(group["测试_相对基准提升倍数"]).mean(),
            "平均OOS目标覆盖率(%)": num(group["测试_目标覆盖率(%)"]).mean(),
            "OOS合计入选": int(num(group["测试_入选总数"]).sum()),
        })
    return pd.DataFrame(rows).sort_values(
        ["红柱形态", "分层任务", "涉及留出年份", "测试改善次数", "平均OOS提升倍数"],
        ascending=[True, True, False, False, False],
    )


def v24_main_legacy() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption("只挖掘区别特征，不评分、不排名、不切换模式。严格复现三形态、五结果组和8周路径审计口径。")
    with st.sidebar:
        st.header("正式评价区间")
        eval_date = st.date_input("评价开始", value=date(2023, 6, 5), key="v24_eval")
        end_date = st.date_input("信号截止", value=date(2026, 6, 5), key="v24_end")
        obs_date = st.date_input("行情观察截止", value=date.today(), max_value=date.today(), key="v24_obs")
        st.header("冻结研究口径")
        st.write("上涨、下降、中性三种形态分别研究")
        st.write("未来8周最高涨幅分为30～50、50～100、翻倍以上")
        st.write("MFE<30再分亏损组和盈利不足30%组")
        st.write("保留-10%目标/止损先后路径审计")
        st.write("本版不生成任何选股分数")
        cache = st.checkbox("使用逐股票缓存", value=True, key="v24_cache")
        pause = st.number_input("每次API调用后暂停(秒)", 0.0, 3.0, 0.12, 0.05, key="v24_pause")
        if st.button("清除本程序缓存", key="v24_clear"):
            if os.path.isdir(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            st.success("缓存已清除")
    token = st.text_input("Tushare Token", type="password", key="v24_token")
    if not token:
        st.info("请输入Tushare Token。")
        return
    session_key = "three_state_feature_mining_v24_zip"
    if not st.button("开始V2.4三形态五组特征挖掘", type="primary", key="v24_run"):
        if session_key in st.session_state:
            st.download_button(
                "下载上一次全部结果ZIP", st.session_state[session_key],
                file_name="weekly_macd_three_state_feature_mining_v2_4_all_results.zip",
                mime="application/zip", key="v24_previous_download",
            )
        return
    if eval_date >= end_date or end_date > obs_date:
        st.error("日期关系不正确。")
        return

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    eval_start = pd.Timestamp(eval_date)
    research = eval_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")
    obs = obs_date.strftime("%Y%m%d")
    preload = (eval_date - timedelta(days=3 * 365)).strftime("%Y%m%d")
    config = {
        "signal_start": research, "signal_end": end, "market_end": obs,
        "preload_start": preload, "min_price": 10.0, "min_mv": 100.0,
        "max_mv": 1_000_000_000.0, "price_tolerance_pct": 3.0,
        "stop_threshold_pct": 10.0, "buy_slippage_pct": 0.20,
        "sell_slippage_pct": 0.20, "sample_per_board": 0,
        "sample_seed": DEFAULT_SAMPLE_SEED,
        "long_cycle_min_weeks": DEFAULT_LONG_CYCLE_MIN_WEEKS,
        "material_hist_change_pct": DEFAULT_MATERIAL_HIST_CHANGE_PCT,
        "short_strength_ratio": DEFAULT_SHORT_STRENGTH_RATIO,
    }
    try:
        with st.spinner("加载基础数据和板块指数..."):
            opens = load_trade_calendar(preload, obs)
            full = load_trade_calendar(preload, (obs_date + timedelta(days=7)).strftime("%Y%m%d"))
            basic = load_stock_basic()
            memberships = load_sw_tech_memberships(float(pause))
            week_map = complete_week_last_dates(full)
            boards = {}
            for code in sorted(set(BOARD_INDEX.values())):
                board_daily = fetch_index_history(code, preload, obs, bool(cache), float(pause))
                if not board_daily.empty:
                    boards[code] = build_weekly(board_daily, week_map)
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return

    periods = build_period_index(memberships)
    codes = sorted(set(periods) & set(basic.ts_code.astype(str)))
    universe = basic[basic.ts_code.isin(codes)].copy()
    stocks, universe_audit, population = build_stratified_sample(
        universe, periods, end, 0, DEFAULT_SAMPLE_SEED
    )
    listed = stocks.list_date.apply(lambda x: normalize_date(x, "19000101"))
    delisted = stocks.delist_date.apply(lambda x: normalize_date(x, "99991231"))
    stocks = stocks[~listed.gt(end) & ~delisted.lt(preload)].reset_index(drop=True)

    open_pos = {trade_date: i for i, trade_date in enumerate(opens)}
    records: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}
    rejects: dict[str, int] = {}
    cache_hits = data_failures = 0
    progress = st.progress(0.0)
    status = st.empty()
    for i, stock in stocks.iterrows():
        code = str(stock.ts_code)
        progress.progress((i + 1) / len(stocks), text=f"{i + 1}/{len(stocks)} {code}")
        status.caption(f"事件{len(records)}；缓存{cache_hits}；失败{data_failures}")
        daily, daily_basic, hit = fetch_stock_history(
            code, preload, obs, bool(cache), float(pause)
        )
        cache_hits += int(hit)
        if daily.empty:
            data_failures += 1
            continue
        stock_records, stock_rejects, _ = analyze_stock(
            stock, periods.get(code, []), daily, daily_basic,
            week_map, opens, open_pos, config,
        )
        records.extend(stock_records)
        if stock_records:
            histories[code] = daily.copy()
        for reason, count in stock_rejects.items():
            rejects[reason] = rejects.get(reason, 0) + count
    progress.empty()
    status.empty()
    if not records:
        st.error("没有生成事件。")
        return

    with st.spinner("建立三形态五结果组并完成特征对比..."):
        events = pd.DataFrame(records).sort_values(["Signal_Date", "ts_code", "Event_Type"])
        opportunities = build_cycle_opportunities(
            events, histories, obs, config["sell_slippage_pct"]
        )
        featured = prepare_features(opportunities, boards)
        candidates = featured[
            featured.Strict_Eligible.map(bool_value)
            & featured.Outcome_Mature.map(bool_value)
            & featured.Selection_Date_dt.ge(eval_start)
        ].copy()
        candidates = assign_five_outcome_groups(candidates)
        candidates = candidates[
            candidates["三形态有效"] & candidates["结果组"].isin(GROUP_ORDER)
        ].copy()
        core_numeric = numeric_feature_comparisons(candidates, CORE_COMPARISONS, "核心18组")
        core_categorical = categorical_feature_comparisons(candidates, CORE_COMPARISONS, "核心18组")
        risk_numeric = numeric_feature_comparisons(candidates, RISK_COMPARISONS, "亏损风险")
        risk_categorical = categorical_feature_comparisons(candidates, RISK_COMPARISONS, "亏损风险")
        adjacent_numeric = numeric_feature_comparisons(candidates, ADJACENT_COMPARISONS, "相邻层级")
        adjacent_categorical = categorical_feature_comparisons(candidates, ADJACENT_COMPARISONS, "相邻层级")
        group_summary = five_group_summary(candidates)
        yearly = year_group_summary(candidates)
        paths = path_audit(candidates)
        gradients = feature_gradient_table(candidates)
        distinguishers = candidate_distinguishers(
            core_numeric, pd.concat([risk_numeric, adjacent_numeric], ignore_index=True),
            core_categorical, pd.concat([risk_categorical, adjacent_categorical], ignore_index=True)
        )

    expected_counts = candidates["结果组"].value_counts().reindex(GROUP_ORDER, fill_value=0)
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "评价开始": research,
        "信号截止": end, "观察截止": obs, "成熟候选": len(candidates),
        "候选周": candidates.Selection_Date.nunique(),
        "不同股票": candidates.ts_code.nunique(),
        "上升趋势": int(candidates.Weekly_Trend.eq("上升趋势").sum()),
        "中性趋势": int(candidates.Weekly_Trend.eq("中性趋势").sum()),
        "下降趋势": int(candidates.Weekly_Trend.eq("下降趋势").sum()),
        "亏损且MFE<30": int(expected_counts[LOSS_GROUP]),
        "盈利且MFE<30": int(expected_counts[LOW_PROFIT_GROUP]),
        "MFE30至50": int(expected_counts[MID_GROUP]),
        "MFE50至100": int(expected_counts[HIGH_GROUP]),
        "MFE翻倍": int(expected_counts[DOUBLE_GROUP]),
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    reject_frame = pd.DataFrame([{"剔除原因": k, "次数": v} for k, v in rejects.items()])
    metadata = pd.DataFrame([
        ("程序定位", "三形态五结果组特征挖掘；不评分、不排名、不切换模式"),
        ("硬条件", "科技池、价≥10元、流通市值≥100亿元、第二根完整红柱严格扩张"),
        ("三种形态", "上升趋势、中性趋势、下降趋势，互斥且覆盖全部候选"),
        ("高涨幅分组", "第二根红柱确认后次日开盘买入，未来40个市场交易日MFE分30～50、50～100、翻倍以上"),
        ("低涨幅分组", "MFE<30后按既有Realised_Utility分为亏损与盈利；严格复现已确认的2371样本口径"),
        ("路径审计", "-10%只用于目标先到/止损先到审计，不改变高涨幅MFE分组"),
        ("核心比较", "三形态×三个高涨幅组×亏损/低盈利两个对照组＝18组"),
        ("相邻比较", "盈利不足30→30～50→50～100→翻倍，识别基础、强化和超级行情特征"),
        ("未来信息隔离", "特征只使用第二根完整红柱确认时已知数据；未来8周字段只作结果标签"),
        ("本版禁止", "不产生分数、不选Top3、不用结果反向修改硬条件"),
    ], columns=["项目", "值"])

    detail_columns = [
        "Cycle_ID", "Selection_Date", "Selection_Year", "ts_code", "name",
        "Sample_Board", "SW_L1", "SW_L2", "Weekly_Trend", "结果组",
        "CP_W2_Delayed_MFE_8W_pct", "CP_W2_Delayed_MAE_8W_pct",
        "CP_W2_Delayed_Return_8W_pct", "Realised_Utility",
        "CP_W2_Delayed_First_30_vs_Stop", "CP_W2_Delayed_First_50_vs_Stop",
        "CP_W2_Delayed_First_100_vs_Stop", *NUMERIC_MINING_FEATURES.values(),
    ]
    detail_columns = list(dict.fromkeys(c for c in detail_columns if c in candidates.columns))
    files = {
        "01_run_summary_three_state_feature_mining_v2_4.csv": run_summary,
        "02_five_group_summary_three_state_feature_mining_v2_4.csv": group_summary,
        "03_year_five_group_summary_three_state_feature_mining_v2_4.csv": yearly,
        "04_core_18_numeric_comparisons_three_state_feature_mining_v2_4.csv": core_numeric,
        "05_core_18_categorical_comparisons_three_state_feature_mining_v2_4.csv": core_categorical,
        "06_loss_vs_low_profit_numeric_three_state_feature_mining_v2_4.csv": risk_numeric,
        "07_loss_vs_low_profit_categorical_three_state_feature_mining_v2_4.csv": risk_categorical,
        "08_adjacent_numeric_comparisons_three_state_feature_mining_v2_4.csv": adjacent_numeric,
        "09_adjacent_categorical_comparisons_three_state_feature_mining_v2_4.csv": adjacent_categorical,
        "10_target_stop_path_audit_three_state_feature_mining_v2_4.csv": paths,
        "11_feature_gradient_quintiles_three_state_feature_mining_v2_4.csv": gradients,
        "12_candidate_distinguishers_three_state_feature_mining_v2_4.csv": distinguishers,
        "13_all_group_candidate_detail_three_state_feature_mining_v2_4.csv": candidates[detail_columns],
        "14_feature_dictionary_three_state_feature_mining_v2_4.csv": mining_feature_dictionary(),
        "15_all_evaluation_candidate_features_three_state_feature_mining_v2_4.csv": candidates,
        "16_all_event_features_three_state_feature_mining_v2_4.csv": featured,
        "17_full_tech_universe_three_state_feature_mining_v2_4.csv": universe_audit,
        "18_population_three_state_feature_mining_v2_4.csv": population,
        "19_rejection_audit_three_state_feature_mining_v2_4.csv": reject_frame,
        "20_metadata_three_state_feature_mining_v2_4.csv": metadata,
    }
    result_zip = make_result_zip(files)
    st.session_state[session_key] = result_zip
    st.success(
        f"完成：{len(candidates)}个成熟候选；三形态五组、18组核心对比和相邻层级对比已生成。"
    )
    st.subheader("三形态五结果组")
    st.dataframe(group_summary, use_container_width=True, hide_index=True)
    st.subheader("30%～50%组目标先到路径核对")
    st.dataframe(paths[paths["结果组"].eq(MID_GROUP)], use_container_width=True, hide_index=True)
    st.subheader("候选区别特征（仅审计，不是评分）")
    st.dataframe(distinguishers[distinguishers["组内审计排名"].le(10)],
                 use_container_width=True, hide_index=True)
    st.download_button(
        "下载全部结果ZIP", result_zip,
        file_name="weekly_macd_three_state_feature_mining_v2_4_all_results.zip",
        mime="application/zip", type="primary", key="v24_download",
    )
    st.warning("本版只发现区别特征。任何特征进入评分前，必须等待跨年稳定性、覆盖率和误选率审查。")


def v25_main_legacy() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "每次把一个年份完全留出，只用其他年份学习特征形状、分位阈值和两个特征的组合；"
        "留出年份只负责判卷。本版不产生最终评分，也不选Top3。"
    )

    with st.sidebar:
        st.header("数据来源")
        source_mode = st.radio(
            "选择底层样本来源",
            ("上传V2.4全部结果ZIP（推荐）", "重新下载行情并生成样本"),
            key="v25_source_mode",
        )
        uploaded = None
        if source_mode.startswith("上传"):
            uploaded = st.file_uploader(
                "上传 weekly_macd_three_state_feature_mining_v2_4_all_results.zip",
                type=["zip", "csv"], key="v25_v24_upload",
                help="也可以直接上传V2.4的15号候选特征CSV。这样能严格复用2371个样本，且无需重新下载行情。",
            )
            st.info("推荐使用本次V2.4结果ZIP：样本完全一致，几分钟内即可完成组合验证。")
            eval_date = end_date = obs_date = None
            cache = True
            pause = 0.12
        else:
            st.header("正式评价区间")
            eval_date = st.date_input("评价开始", value=date(2023, 6, 5), key="v25_eval")
            end_date = st.date_input("信号截止", value=date(2026, 6, 5), key="v25_end")
            obs_date = st.date_input(
                "行情观察截止", value=date.today(), max_value=date.today(), key="v25_obs"
            )
            cache = st.checkbox("使用逐股票缓存", value=True, key="v25_cache")
            pause = st.number_input(
                "每次API调用后暂停(秒)", 0.0, 3.0, 0.12, 0.05, key="v25_pause"
            )
            if st.button("清除本程序缓存", key="v25_clear"):
                if os.path.isdir(CACHE_DIR):
                    shutil.rmtree(CACHE_DIR)
                st.success("缓存已清除")

        st.header("冻结验证规则")
        st.write("形态：上升、中性、下降分别验证")
        st.write("层级：避免亏损、达到30%、进入50%、翻倍附加")
        st.write("每条规则最多两个数值特征")
        st.write("阈值只来自训练年份分位数")
        st.write("行业名称不参与组合，避免把历史主线写死")
        st.write("翻倍层只探索，任何结果都不正式通过")

    token = ""
    if source_mode.startswith("重新"):
        token = st.text_input("Tushare Token", type="password", key="v25_token")

    session_key = "two_feature_oos_v25_zip"
    run_requested = st.button("开始V2.5双特征跨年验证", type="primary", key="v25_run")
    if not run_requested:
        if session_key in st.session_state:
            st.success("上一次V2.5结果仍然保留，可直接下载。")
            st.download_button(
                "下载上一次全部结果ZIP", st.session_state[session_key],
                file_name="weekly_macd_two_feature_oos_v2_5_all_results.zip",
                mime="application/zip", key="v25_previous_download",
            )
        return

    API_ERRORS = []
    input_summary = pd.DataFrame()
    universe_audit = pd.DataFrame()
    population = pd.DataFrame()
    reject_frame = pd.DataFrame()
    featured = pd.DataFrame()
    data_failures = cache_hits = 0

    if source_mode.startswith("上传"):
        if uploaded is None:
            st.error("请先上传V2.4全部结果ZIP或15号候选特征CSV。")
            return
        try:
            with st.spinner("读取并核对V2.4三形态五结果组样本..."):
                candidates, input_summary = v25_load_v24_upload(uploaded)
        except Exception as exc:
            st.error(f"V2.4结果读取失败：{exc}")
            return
        source_description = f"复用上传文件 {uploaded.name}"
        research = candidates["Selection_Date"].min()
        end = candidates["Selection_Date"].max()
        obs = "沿用V2.4"
    else:
        if not token:
            st.error("请输入Tushare Token。")
            return
        if eval_date is None or end_date is None or obs_date is None:
            st.error("日期配置不完整。")
            return
        if eval_date >= end_date or end_date > obs_date:
            st.error("日期关系不正确。")
            return
        ts.set_token(token)
        pro = ts.pro_api()
        eval_start = pd.Timestamp(eval_date)
        research = eval_date.strftime("%Y%m%d")
        end = end_date.strftime("%Y%m%d")
        obs = obs_date.strftime("%Y%m%d")
        preload = (eval_date - timedelta(days=3 * 365)).strftime("%Y%m%d")
        config = {
            "signal_start": research, "signal_end": end, "market_end": obs,
            "preload_start": preload, "min_price": 10.0, "min_mv": 100.0,
            "max_mv": 1_000_000_000.0, "price_tolerance_pct": 3.0,
            "stop_threshold_pct": 10.0, "buy_slippage_pct": 0.20,
            "sell_slippage_pct": 0.20, "sample_per_board": 0,
            "sample_seed": DEFAULT_SAMPLE_SEED,
            "long_cycle_min_weeks": DEFAULT_LONG_CYCLE_MIN_WEEKS,
            "material_hist_change_pct": DEFAULT_MATERIAL_HIST_CHANGE_PCT,
            "short_strength_ratio": DEFAULT_SHORT_STRENGTH_RATIO,
        }
        try:
            with st.spinner("加载基础数据、完整交易日历和板块指数..."):
                opens = load_trade_calendar(preload, obs)
                full = load_trade_calendar(
                    preload, (obs_date + timedelta(days=7)).strftime("%Y%m%d")
                )
                basic = load_stock_basic()
                memberships = load_sw_tech_memberships(float(pause))
                week_map = complete_week_last_dates(full)
                boards: dict[str, pd.DataFrame] = {}
                for code in sorted(set(BOARD_INDEX.values())):
                    board_daily = fetch_index_history(
                        code, preload, obs, bool(cache), float(pause)
                    )
                    if not board_daily.empty:
                        boards[code] = build_weekly(board_daily, week_map)
        except Exception as exc:
            st.error(f"基础数据加载失败：{exc}")
            return

        periods = build_period_index(memberships)
        codes = sorted(set(periods) & set(basic.ts_code.astype(str)))
        universe = basic[basic.ts_code.isin(codes)].copy()
        stocks, universe_audit, population = build_stratified_sample(
            universe, periods, end, 0, DEFAULT_SAMPLE_SEED
        )
        listed = stocks.list_date.apply(lambda x: normalize_date(x, "19000101"))
        delisted = stocks.delist_date.apply(lambda x: normalize_date(x, "99991231"))
        stocks = stocks[~listed.gt(end) & ~delisted.lt(preload)].reset_index(drop=True)
        open_pos = {trade_date: i for i, trade_date in enumerate(opens)}
        records: list[dict[str, Any]] = []
        histories: dict[str, pd.DataFrame] = {}
        rejects: dict[str, int] = {}
        progress = st.progress(0.0)
        status = st.empty()
        for i, stock in stocks.iterrows():
            code = str(stock.ts_code)
            progress.progress((i + 1) / len(stocks), text=f"{i + 1}/{len(stocks)} {code}")
            status.caption(f"事件{len(records)}；缓存{cache_hits}；失败{data_failures}")
            daily, daily_basic, hit = fetch_stock_history(
                code, preload, obs, bool(cache), float(pause)
            )
            cache_hits += int(hit)
            if daily.empty:
                data_failures += 1
                continue
            stock_records, stock_rejects, _ = analyze_stock(
                stock, periods.get(code, []), daily, daily_basic,
                week_map, opens, open_pos, config,
            )
            records.extend(stock_records)
            if stock_records:
                histories[code] = daily.copy()
            for reason, count in stock_rejects.items():
                rejects[reason] = rejects.get(reason, 0) + count
        progress.empty()
        status.empty()
        if not records:
            st.error("没有生成事件。")
            return
        with st.spinner("生成与V2.4完全相同的严格扩张成熟候选..."):
            events = pd.DataFrame(records).sort_values(["Signal_Date", "ts_code", "Event_Type"])
            opportunities = build_cycle_opportunities(
                events, histories, obs, config["sell_slippage_pct"]
            )
            featured = prepare_features(opportunities, boards)
            candidates = featured[
                featured.Strict_Eligible.map(bool_value)
                & featured.Outcome_Mature.map(bool_value)
                & featured.Selection_Date_dt.ge(eval_start)
            ].copy()
            candidates = v25_prepare_candidates(candidates)
        reject_frame = pd.DataFrame(
            [{"剔除原因": reason, "次数": count} for reason, count in rejects.items()]
        )
        source_description = "本次从Tushare重新生成底层样本"

    if candidates.empty:
        st.error("没有形成可验证的成熟候选。")
        return
    if candidates["Selection_Year"].nunique() < 3:
        st.error("至少需要三个不同年份，才能进行整年留出验证。")
        return

    with st.spinner("逐形态、逐层级、逐年份学习并验证双特征组合..."):
        reports = v25_run_loyo(candidates)
        group_summary = five_group_summary(candidates)
        yearly = year_group_summary(candidates)
        paths = path_audit(candidates)

    aggregate = reports["aggregate"]
    completed_folds = int(reports["folds"]["处理状态"].eq("完成").sum()) \
        if not reports["folds"].empty else 0
    passed = int(aggregate["验收结论"].str.startswith("通过候选").sum()) \
        if not aggregate.empty else 0
    counts = candidates["结果组"].value_counts().reindex(GROUP_ORDER, fill_value=0)
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "数据来源": source_description,
        "评价开始": research, "信号截止": end, "观察截止": obs,
        "成熟候选": len(candidates), "候选周": candidates.Selection_Date.nunique(),
        "不同股票": candidates.ts_code.nunique(),
        "上升趋势": int(candidates.Weekly_Trend.eq("上升趋势").sum()),
        "中性趋势": int(candidates.Weekly_Trend.eq("中性趋势").sum()),
        "下降趋势": int(candidates.Weekly_Trend.eq("下降趋势").sum()),
        "亏损且MFE<30": int(counts[LOSS_GROUP]),
        "盈利且MFE<30": int(counts[LOW_PROFIT_GROUP]),
        "MFE30至50": int(counts[MID_GROUP]),
        "MFE50至100": int(counts[HIGH_GROUP]),
        "MFE翻倍": int(counts[DOUBLE_GROUP]),
        "完成留出折叠": completed_folds,
        "通过候选验收任务": passed,
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    metadata = pd.DataFrame([
        ("程序定位", "三形态、四层级、双特征组合的整年留出验证；不生成最终评分"),
        ("底层口径", "严格沿用V2.4：科技池、价≥10元、流通市值≥100亿元、第二根完整红柱严格扩张"),
        ("结果标签", "第二根红柱确认后次日开盘买入，未来40个市场交易日五个互斥结果组"),
        ("验证隔离", "每个留出年份的标签、分布和阈值均不参与该折叠的训练"),
        ("规则形状", "仅允许预先固定的高端、低端、中间、偏强中段、偏弱中段、排除极端等分位形状"),
        ("组合限制", "每条组合严格两个不同字段；训练期Spearman绝对相关>0.90禁止配对"),
        ("特征限制", "只使用第二根完整红柱确认时已知的24个数值特征；行业名称不参与组合"),
        ("验收对象", "每个折叠仅训练排名第1的组合参与正式汇总；其余组合只作探索审计"),
        ("正式门槛", "至少3个可判卷年份且每年各入选≥5只、2/3年份改善、OOS提升≥1.15、覆盖≥12%、合计入选≥20、双特征优于单特征"),
        ("翻倍限制", "翻倍层无论结果如何都禁止正式通过；只输出低置信度探索"),
        ("禁止事项", "本版不评分、不选Top3、不做模式切换、不用留出年反向修改规则"),
    ], columns=["项目", "值"])

    files = {
        "01_run_summary_two_feature_oos_v2_5.csv": run_summary,
        "02_five_group_summary_two_feature_oos_v2_5.csv": group_summary,
        "03_layer_task_definitions_two_feature_oos_v2_5.csv": v25_task_definitions(),
        "04_fold_sample_audit_two_feature_oos_v2_5.csv": reports["folds"],
        "05_train_only_single_rules_two_feature_oos_v2_5.csv": reports["singles"],
        "06_all_pair_oos_results_two_feature_oos_v2_5.csv": reports["pairs"],
        "07_best_pair_each_fold_two_feature_oos_v2_5.csv": reports["best"],
        "08_oos_acceptance_summary_two_feature_oos_v2_5.csv": aggregate,
        "09_pair_cross_year_stability_two_feature_oos_v2_5.csv": reports["stability"],
        "10_best_pair_selected_event_detail_two_feature_oos_v2_5.csv": reports["selected"],
        "11_year_five_group_summary_two_feature_oos_v2_5.csv": yearly,
        "12_target_stop_path_audit_two_feature_oos_v2_5.csv": paths,
        "13_all_evaluation_candidate_features_two_feature_oos_v2_5.csv": candidates,
        "14_feature_dictionary_two_feature_oos_v2_5.csv": mining_feature_dictionary(),
        "15_metadata_two_feature_oos_v2_5.csv": metadata,
    }
    if not input_summary.empty:
        files["16_input_v2_4_run_summary_two_feature_oos_v2_5.csv"] = input_summary
    if not universe_audit.empty:
        files["17_full_tech_universe_two_feature_oos_v2_5.csv"] = universe_audit
    if not population.empty:
        files["18_population_two_feature_oos_v2_5.csv"] = population
    if not reject_frame.empty:
        files["19_rejection_audit_two_feature_oos_v2_5.csv"] = reject_frame
    if not featured.empty:
        files["20_all_event_features_two_feature_oos_v2_5.csv"] = featured

    result_zip = make_result_zip(files)
    st.session_state[session_key] = result_zip
    st.success(
        f"完成：{len(candidates)}个成熟候选，{completed_folds}个留出折叠完成；"
        f"{passed}个形态×层级任务通过候选验收。"
    )
    st.subheader("正式OOS验收结论")
    st.dataframe(aggregate, use_container_width=True, hide_index=True)
    st.subheader("每个留出年份的训练第一名组合")
    st.dataframe(reports["best"], use_container_width=True, hide_index=True)
    with st.expander("查看重复出现的组合与全部折叠样本审计"):
        st.dataframe(reports["stability"], use_container_width=True, hide_index=True)
        st.dataframe(reports["folds"], use_container_width=True, hide_index=True)
    st.download_button(
        "下载全部结果ZIP", result_zip,
        file_name="weekly_macd_two_feature_oos_v2_5_all_results.zip",
        mime="application/zip", type="primary", key="v25_download",
    )
    st.download_button(
        "单独下载8号OOS验收总表", csv_bytes(aggregate),
        file_name="08_oos_acceptance_summary_two_feature_oos_v2_5.csv",
        mime="text/csv", key="v25_acceptance_download",
    )
    st.warning(
        "如果没有任务通过，这不是程序失败，而是说明现有两个技术特征不足以跨年稳定区分层级。"
        "禁止为了让结果好看而在本轮查看留出年后修改阈值。"
    )


# =============================================================================
# V2.6：直接目标 vs 全部较差结果；亏损否决单独验证
# =============================================================================
TITLE = "科技股周线MACD直接目标与亏损否决跨年验证器 V2.6"
VERSION = "V2.6-DIRECT-TARGET-LOSS-VETO-LOYO"

V26_TASKS = (
    {
        "任务": "亏损否决",
        "任务类型": "否决",
        "目标组": (LOSS_GROUP,),
        "对照组": (LOW_PROFIT_GROUP, MID_GROUP, HIGH_GROUP, DOUBLE_GROUP),
        "含义": "寻找亏损组高发、同时极少误删30%以上股票的反向风险形状",
        "正式验收": True,
    },
    {
        "任务": "达到30%",
        "任务类型": "直接目标",
        "目标组": (MID_GROUP, HIGH_GROUP, DOUBLE_GROUP),
        "对照组": (LOSS_GROUP, LOW_PROFIT_GROUP),
        "含义": "30%以上直接对比全部30%以下；不存在前置风险条件",
        "正式验收": True,
    },
    {
        "任务": "达到50%",
        "任务类型": "直接目标",
        "目标组": (HIGH_GROUP, DOUBLE_GROUP),
        "对照组": (LOSS_GROUP, LOW_PROFIT_GROUP, MID_GROUP),
        "含义": "50%以上直接对比全部50%以下；不存在前置30%条件",
        "正式验收": True,
    },
    {
        "任务": "翻倍审计",
        "任务类型": "探索",
        "目标组": (DOUBLE_GROUP,),
        "对照组": (LOSS_GROUP, LOW_PROFIT_GROUP, MID_GROUP, HIGH_GROUP),
        "含义": "翻倍与全部非翻倍结果直接比较；只作低置信度审计",
        "正式验收": False,
    },
)


def v26_task_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "任务": task["任务"], "任务类型": task["任务类型"],
            "目标组": " + ".join(task["目标组"]),
            "对照组": " + ".join(task["对照组"]),
            "研究含义": task["含义"],
            "是否正式验收": "是" if task["正式验收"] else "否，只作探索",
        }
        for task in V26_TASKS
    ])


def v26_load_prior_upload(uploaded_file: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """接受V2.4、V2.5的全部结果ZIP，也接受其中的候选特征CSV。"""
    raw = uploaded_file.getvalue()
    input_summary = pd.DataFrame()
    if str(uploaded_file.name).lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            names = archive.namelist()
            priorities = (
                "13_all_evaluation_candidate_features_two_feature_oos_v2_5",
                "15_all_evaluation_candidate_features_three_state_feature_mining_v2_4",
                "all_evaluation_candidate_features",
            )
            candidate_name = ""
            for pattern in priorities:
                matches = [name for name in names if pattern in name]
                if matches:
                    candidate_name = matches[0]
                    break
            if not candidate_name:
                raise ValueError("ZIP中没有找到V2.4或V2.5的候选特征文件")
            with archive.open(candidate_name) as source:
                candidates = pd.read_csv(source, low_memory=False)
            summary_names = [name for name in names if "01_run_summary" in name]
            if summary_names:
                with archive.open(summary_names[0]) as source:
                    input_summary = pd.read_csv(source, low_memory=False)
    else:
        candidates = pd.read_csv(io.BytesIO(raw), low_memory=False)
    return v25_prepare_candidates(candidates), input_summary


def v26_selection_outcomes(frame: pd.DataFrame,
                           selected: pd.Series) -> dict[str, Any]:
    picked = selected.reindex(frame.index, fill_value=False).astype(bool)
    chosen = frame[picked]
    result = frame["结果组"]
    chosen_result = chosen["结果组"]
    masks = {
        "亏损": result.eq(LOSS_GROUP),
        "低盈利": result.eq(LOW_PROFIT_GROUP),
        "30至50": result.eq(MID_GROUP),
        "50至100": result.eq(HIGH_GROUP),
        "翻倍": result.eq(DOUBLE_GROUP),
        "30以上": result.isin([MID_GROUP, HIGH_GROUP, DOUBLE_GROUP]),
        "50以上": result.isin([HIGH_GROUP, DOUBLE_GROUP]),
    }
    chosen_masks = {
        "亏损": chosen_result.eq(LOSS_GROUP),
        "低盈利": chosen_result.eq(LOW_PROFIT_GROUP),
        "30至50": chosen_result.eq(MID_GROUP),
        "50至100": chosen_result.eq(HIGH_GROUP),
        "翻倍": chosen_result.eq(DOUBLE_GROUP),
        "30以上": chosen_result.isin([MID_GROUP, HIGH_GROUP, DOUBLE_GROUP]),
        "50以上": chosen_result.isin([HIGH_GROUP, DOUBLE_GROUP]),
    }
    out: dict[str, Any] = {"全体样本": len(frame), "入选样本": len(chosen)}
    for label in masks:
        total = int(masks[label].sum())
        selected_n = int(chosen_masks[label].sum())
        out[f"全体{label}数"] = total
        out[f"入选{label}数"] = selected_n
        out[f"全体{label}率(%)"] = total / len(frame) * 100.0 if len(frame) else np.nan
        out[f"入选{label}率(%)"] = selected_n / len(chosen) * 100.0 if len(chosen) else np.nan
        out[f"{label}覆盖率(%)"] = selected_n / total * 100.0 if total else np.nan
    baseline_loss = out["全体亏损率(%)"]
    selected_loss = out["入选亏损率(%)"]
    out["亏损率相对下降(%)"] = (
        (baseline_loss - selected_loss) / baseline_loss * 100.0
        if math.isfinite(baseline_loss) and baseline_loss > 0
        and math.isfinite(selected_loss) else np.nan
    )
    return out


def v26_rule_is_safe(task_type: str,
                     outcomes: dict[str, Any]) -> bool:
    if task_type == "否决":
        # 否决规则本身允许覆盖亏损，但不能在训练期顺便删除大量潜在牛股。
        cover30 = outcomes.get("30以上覆盖率(%)", np.nan)
        cover50 = outcomes.get("50以上覆盖率(%)", np.nan)
        return ((not math.isfinite(cover30) or cover30 <= 15.0)
                and (not math.isfinite(cover50) or cover50 <= 15.0))
    if task_type == "直接目标":
        reduction = outcomes.get("亏损率相对下降(%)", np.nan)
        return math.isfinite(reduction) and reduction >= 0.0
    return True


def v26_learn_single_rules(train: pd.DataFrame,
                           task_type: str) -> pd.DataFrame:
    labels = train["_Target"].astype(bool)
    rows: list[dict[str, Any]] = []
    for feature_label, field in NUMERIC_MINING_FEATURES.items():
        if field not in train:
            continue
        values = pd.to_numeric(train[field], errors="coerce")
        if values.notna().sum() < 30 or values.nunique(dropna=True) < 8:
            continue
        for shape_name, lower_q, upper_q in V25_RULE_SHAPES:
            lower = float(values.quantile(lower_q)) if lower_q is not None else np.nan
            upper = float(values.quantile(upper_q)) if upper_q is not None else np.nan
            selected = v25_rule_mask(values, lower, upper)
            metrics = v25_binary_metrics(labels, selected)
            target_coverage = metrics["目标覆盖率(%)"] / 100.0
            edge = (metrics["目标覆盖率(%)"] - metrics["对照误入率(%)"]) / 100.0
            lift = metrics["相对基准提升倍数"]
            if (target_coverage < 0.15 or metrics["入选总数"] < 8
                    or not math.isfinite(lift) or edge <= 0 or lift <= 1.02):
                continue
            outcomes = v26_selection_outcomes(train, selected)
            if not v26_rule_is_safe(task_type, outcomes):
                continue
            improved, comparable = v25_training_year_consistency(train, selected)
            consistency = improved / comparable if comparable else 0.0
            risk_bonus = 0.0
            if task_type == "直接目标":
                risk_bonus = max(outcomes["亏损率相对下降(%)"], 0.0) * 0.20
            elif task_type == "否决":
                risk_bonus = max(15.0 - outcomes["30以上覆盖率(%)"], 0.0) * 0.20
            score = edge * 100.0 + (lift - 1.0) * 20.0 + consistency * 5.0 + risk_bonus
            rows.append({
                "特征": feature_label, "字段": field, "规则形状": shape_name,
                "训练下限": lower, "训练上限": upper,
                **metrics, **{f"训练_{k}": v for k, v in outcomes.items()},
                "训练改善年份": improved, "训练可比较年份": comparable,
                "训练规则分": score,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return (result.sort_values("训练规则分", ascending=False)
            .drop_duplicates("字段", keep="first")
            .head(V25_TOP_SINGLE_FEATURES).reset_index(drop=True))


def v26_learn_pairs(train: pd.DataFrame,
                    single_rules: pd.DataFrame,
                    task_type: str) -> pd.DataFrame:
    if len(single_rules) < 2:
        return pd.DataFrame()
    labels = train["_Target"].astype(bool)
    rows: list[dict[str, Any]] = []
    for left_i in range(len(single_rules)):
        for right_i in range(left_i + 1, len(single_rules)):
            left = single_rules.iloc[left_i]
            right = single_rules.iloc[right_i]
            left_values = pd.to_numeric(train[left["字段"]], errors="coerce")
            right_values = pd.to_numeric(train[right["字段"]], errors="coerce")
            correlation = left_values.corr(right_values, method="spearman")
            if math.isfinite(correlation) and abs(correlation) > V25_MAX_PAIR_CORRELATION:
                continue
            selected = v25_apply_rule(train, left) & v25_apply_rule(train, right)
            metrics = v25_binary_metrics(labels, selected)
            target_coverage = metrics["目标覆盖率(%)"] / 100.0
            lift = metrics["相对基准提升倍数"]
            if (target_coverage < 0.08 or metrics["入选总数"] < 6
                    or not math.isfinite(lift) or lift <= 1.02):
                continue
            outcomes = v26_selection_outcomes(train, selected)
            if not v26_rule_is_safe(task_type, outcomes):
                continue
            improved, comparable = v25_training_year_consistency(train, selected)
            consistency = improved / comparable if comparable else 0.0
            edge = (metrics["目标覆盖率(%)"] - metrics["对照误入率(%)"]) / 100.0
            precision_edge = metrics["入选目标率(%)"] - metrics["基准目标率(%)"]
            risk_bonus = 0.0
            if task_type == "直接目标":
                risk_bonus = max(outcomes["亏损率相对下降(%)"], 0.0) * 0.35
            elif task_type == "否决":
                risk_bonus = max(15.0 - outcomes["30以上覆盖率(%)"], 0.0) * 0.35
            score = precision_edge + edge * 50.0 + consistency * 5.0 + risk_bonus
            signature_parts = sorted([
                (str(left["字段"]), str(left["规则形状"])),
                (str(right["字段"]), str(right["规则形状"])),
            ])
            rows.append({
                "特征1": left["特征"], "字段1": left["字段"],
                "形状1": left["规则形状"], "下限1": left["训练下限"], "上限1": left["训练上限"],
                "特征2": right["特征"], "字段2": right["字段"],
                "形状2": right["规则形状"], "下限2": right["训练下限"], "上限2": right["训练上限"],
                "组合签名": f"{signature_parts[0][0]}[{signature_parts[0][1]}] + "
                            f"{signature_parts[1][0]}[{signature_parts[1][1]}]",
                "字段组合": " + ".join(sorted([str(left["字段"]), str(right["字段"])])),
                "训练Spearman相关": correlation,
                **metrics, **{f"训练_{k}": v for k, v in outcomes.items()},
                "训练改善年份": improved, "训练可比较年份": comparable,
                "训练组合分": score,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["训练组合分", "相对基准提升倍数", "目标覆盖率(%)"], ascending=False
    ).head(V25_TOP_PAIRS_PER_FOLD).reset_index(drop=True)
    result["训练排名"] = np.arange(1, len(result) + 1)
    return result


def v26_run_loyo(candidates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    years = sorted(candidates["Selection_Year"].dropna().astype(str).unique())
    for state in STATE_ORDER:
        state_frame = candidates[candidates["Weekly_Trend"].eq(state)].copy()
        for task in V26_TASKS:
            task_frame = state_frame.copy()
            task_frame["_Target"] = task_frame["结果组"].isin(task["目标组"])
            for holdout in years:
                fold_id = f"{state}|{task['任务']}|留出{holdout}"
                train = task_frame[task_frame["Selection_Year"].ne(holdout)].copy()
                test = task_frame[task_frame["Selection_Year"].eq(holdout)].copy()
                train_target = int(train["_Target"].sum())
                train_control = int((~train["_Target"]).sum())
                test_target = int(test["_Target"].sum())
                test_control = int((~test["_Target"]).sum())
                min_train = 5 if not task["正式验收"] else V25_MIN_TRAIN_EACH
                min_test = 2 if not task["正式验收"] else V25_MIN_TEST_EACH
                eligible = (train_target >= min_train and train_control >= min_train
                            and test_target >= min_test and test_control >= min_test)
                fold_row = {
                    "折叠ID": fold_id, "红柱形态": state, "任务": task["任务"],
                    "任务类型": task["任务类型"], "留出年份": holdout,
                    "正式验收任务": bool(task["正式验收"]),
                    "训练目标": train_target, "训练对照": train_control,
                    "测试目标": test_target, "测试对照": test_control,
                    "样本门槛通过": eligible,
                    "处理状态": "样本不足" if not eligible else "待训练",
                }
                if not eligible:
                    fold_rows.append(fold_row)
                    continue
                single = v26_learn_single_rules(train, task["任务类型"])
                if single.empty:
                    fold_row["处理状态"] = "没有满足训练风险约束的单特征"
                    fold_rows.append(fold_row)
                    continue
                single = single.copy()
                single.insert(0, "折叠ID", fold_id)
                single.insert(1, "红柱形态", state)
                single.insert(2, "任务", task["任务"])
                single.insert(3, "任务类型", task["任务类型"])
                single.insert(4, "留出年份", holdout)
                single["训练单特征排名"] = np.arange(1, len(single) + 1)
                for _, rule in single.iterrows():
                    test_selected = v25_apply_rule(test, rule)
                    test_metrics = v25_binary_metrics(test["_Target"], test_selected)
                    test_outcomes = v26_selection_outcomes(test, test_selected)
                    row = rule.to_dict()
                    row.update({f"测试_{key}": value for key, value in test_metrics.items()})
                    row.update({f"测试_{key}": value for key, value in test_outcomes.items()})
                    single_rows.append(row)
                pairs = v26_learn_pairs(train, single, task["任务类型"])
                if pairs.empty:
                    fold_row["处理状态"] = "没有满足训练风险约束的双特征"
                    fold_rows.append(fold_row)
                    continue
                fold_row.update({"处理状态": "完成", "训练单特征数": len(single),
                                 "训练组合数": len(pairs)})
                fold_rows.append(fold_row)
                best_single = single.iloc[0]
                best_single_selected = v25_apply_rule(test, best_single)
                best_single_test = v25_binary_metrics(test["_Target"], best_single_selected)
                best_single_outcomes = v26_selection_outcomes(test, best_single_selected)
                for _, pair in pairs.iterrows():
                    test_selected = v25_apply_pair(test, pair)
                    test_metrics = v25_binary_metrics(test["_Target"], test_selected)
                    test_outcomes = v26_selection_outcomes(test, test_selected)
                    row = pair.to_dict()
                    row.update({
                        "折叠ID": fold_id, "红柱形态": state, "任务": task["任务"],
                        "分层任务": task["任务"], "任务类型": task["任务类型"],
                        "留出年份": holdout, "正式验收任务": bool(task["正式验收"]),
                        "测试判卷有效": test_metrics["入选总数"] >= (
                            5 if task["正式验收"] else 2),
                        **{f"测试_{key}": value for key, value in test_metrics.items()},
                        **{f"测试_{key}": value for key, value in test_outcomes.items()},
                    })
                    pair_rows.append(row)
                    if int(pair["训练排名"]) == 1:
                        best = row.copy()
                        best.update({
                            "最佳单特征": best_single["特征"],
                            "最佳单特征形状": best_single["规则形状"],
                            **{f"单特征测试_{key}": value for key, value in best_single_test.items()},
                            **{f"单特征测试_{key}": value for key, value in best_single_outcomes.items()},
                        })
                        best_rows.append(best)
                        for _, event in test[test_selected].iterrows():
                            selected_rows.append({
                                "折叠ID": fold_id, "红柱形态": state,
                                "任务": task["任务"], "任务类型": task["任务类型"],
                                "留出年份": holdout, "组合签名": pair["组合签名"],
                                "是否任务目标": bool(event["_Target"]),
                                "结果组": event["结果组"],
                                "Selection_Date": event.get("Selection_Date", ""),
                                "ts_code": event.get("ts_code", ""), "name": event.get("name", ""),
                                "MFE8周(%)": event.get("CP_W2_Delayed_MFE_8W_pct", np.nan),
                                "MAE8周(%)": event.get("CP_W2_Delayed_MAE_8W_pct", np.nan),
                                "8周末收益(%)": event.get("CP_W2_Delayed_Return_8W_pct", np.nan),
                            })
    folds = pd.DataFrame(fold_rows)
    singles = pd.DataFrame(single_rows)
    pairs = pd.DataFrame(pair_rows)
    best = pd.DataFrame(best_rows)
    selected = pd.DataFrame(selected_rows)
    return {
        "folds": folds, "singles": singles, "pairs": pairs, "best": best,
        "aggregate": v26_acceptance(best),
        "stability": v25_pair_stability(pairs) if not pairs.empty else pd.DataFrame(),
        "selected": selected,
        "concentration": v26_selection_concentration(selected),
    }


def v26_acceptance(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (state, task_name), group in best.groupby(["红柱形态", "任务"]):
        task_type = str(group["任务类型"].iloc[0])
        target_n = int(num(group["测试_目标样本"]).sum())
        control_n = int(num(group["测试_对照样本"]).sum())
        selected_target = int(num(group["测试_入选目标"]).sum())
        selected_control = int(num(group["测试_误入对照"]).sum())
        selected_n = selected_target + selected_control
        baseline = target_n / (target_n + control_n) if target_n + control_n else np.nan
        precision = selected_target / selected_n if selected_n else np.nan
        coverage = selected_target / target_n if target_n else np.nan
        lift = precision / baseline if selected_n and baseline > 0 else np.nan
        adequate_mask = num(group["测试_入选总数"]).ge(5)
        adequate_folds = int(adequate_mask.sum())
        improved = int((adequate_mask & num(group["测试_相对基准提升倍数"]).gt(1.0)).sum())
        folds = len(group)
        needed_improved = math.ceil(folds * 2 / 3)
        single_target = int(num(group["单特征测试_入选目标"]).sum())
        single_control = int(num(group["单特征测试_误入对照"]).sum())
        single_n = single_target + single_control
        single_precision = single_target / single_n if single_n else np.nan
        totals = {label: int(num(group[f"测试_全体{label}数"]).sum())
                  for label in ("亏损", "低盈利", "30至50", "50至100", "翻倍", "30以上", "50以上")}
        selected_counts = {label: int(num(group[f"测试_入选{label}数"]).sum())
                           for label in ("亏损", "低盈利", "30至50", "50至100", "翻倍", "30以上", "50以上")}
        all_n = totals["亏损"] + totals["低盈利"] + totals["30至50"] + totals["50至100"] + totals["翻倍"]
        baseline_loss = totals["亏损"] / all_n if all_n else np.nan
        selected_loss = selected_counts["亏损"] / selected_n if selected_n else np.nan
        loss_reduction = ((baseline_loss - selected_loss) / baseline_loss
                          if math.isfinite(baseline_loss) and baseline_loss > 0
                          and math.isfinite(selected_loss) else np.nan)
        false_delete30 = selected_counts["30以上"] / totals["30以上"] if totals["30以上"] else np.nan
        false_delete50 = selected_counts["50以上"] / totals["50以上"] if totals["50以上"] else np.nan
        false_delete_double = selected_counts["翻倍"] / totals["翻倍"] if totals["翻倍"] else np.nan
        if task_type == "探索":
            verdict = "探索层：禁止正式通过"
        elif folds < 3:
            verdict = "样本不足：少于3个可判卷年份"
        elif adequate_folds < 3:
            verdict = "样本不足：少于3年各自入选至少5只"
        elif selected_n < 20:
            verdict = "样本不足：OOS合计入选少于20只"
        elif improved < needed_improved:
            verdict = "未通过：跨年方向不稳定"
        elif not math.isfinite(lift) or lift < 1.25:
            verdict = "未通过：OOS提升不足25%"
        elif not math.isfinite(coverage) or coverage < 0.12:
            verdict = "未通过：目标覆盖率不足12%"
        elif task_type == "否决" and (
                (math.isfinite(false_delete30) and false_delete30 > 0.10)
                or (math.isfinite(false_delete50) and false_delete50 > 0.10)):
            verdict = "未通过：否决规则误删大涨股超过10%"
        elif task_type == "直接目标" and (
                not math.isfinite(loss_reduction) or loss_reduction < 0.20):
            verdict = "潜力有提升，但亏损率下降不足20%"
        elif math.isfinite(single_precision) and precision <= single_precision:
            verdict = "未通过：双特征未优于最佳单特征"
        else:
            verdict = "通过候选验收，可进入下一阶段"
        row = {
            "红柱形态": state, "任务": task_name, "任务类型": task_type,
            "可判卷年份": folds, "入选至少5只年份": adequate_folds,
            "改善年份": improved, "要求改善年份": needed_improved,
            "测试目标样本": target_n, "测试对照样本": control_n,
            "双特征入选数": selected_n,
            "双特征目标率(%)": precision * 100.0 if math.isfinite(precision) else np.nan,
            "未筛选基准目标率(%)": baseline * 100.0 if math.isfinite(baseline) else np.nan,
            "双特征OOS提升倍数": lift,
            "双特征目标覆盖率(%)": coverage * 100.0 if math.isfinite(coverage) else np.nan,
            "最佳单特征目标率(%)": single_precision * 100.0 if math.isfinite(single_precision) else np.nan,
            "未筛选亏损率(%)": baseline_loss * 100.0 if math.isfinite(baseline_loss) else np.nan,
            "入选亏损率(%)": selected_loss * 100.0 if math.isfinite(selected_loss) else np.nan,
            "亏损率相对下降(%)": loss_reduction * 100.0 if math.isfinite(loss_reduction) else np.nan,
            "误删30%以上比例(%)": false_delete30 * 100.0 if math.isfinite(false_delete30) else np.nan,
            "误删50%以上比例(%)": false_delete50 * 100.0 if math.isfinite(false_delete50) else np.nan,
            "误删翻倍比例(%)": false_delete_double * 100.0 if math.isfinite(false_delete_double) else np.nan,
            "验收结论": verdict,
        }
        for label in ("亏损", "低盈利", "30至50", "50至100", "翻倍"):
            row[f"入选{label}数"] = selected_counts[label]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["红柱形态", "任务"])


def v26_selection_concentration(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    rows = []
    for (state, task), group in selected.groupby(["红柱形态", "任务"]):
        counts = group["ts_code"].astype(str).value_counts()
        total = len(group)
        rows.append({
            "红柱形态": state, "任务": task, "入选事件": total,
            "不同股票": group["ts_code"].nunique(),
            "单一股票最多入选": int(counts.max()) if len(counts) else 0,
            "单一股票最大占比(%)": counts.max() / total * 100.0 if total else np.nan,
            "前5只股票占比(%)": counts.head(5).sum() / total * 100.0 if total else np.nan,
        })
    return pd.DataFrame(rows)


def v26_main_legacy() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "30%以上直接对比全部30%以下，50%以上直接对比全部50%以下；"
        "亏损形状只作为反向否决实验。每个留出年份不参与特征、阈值和组合学习。"
    )
    with st.sidebar:
        st.header("底层样本")
        uploaded = st.file_uploader(
            "上传V2.5或V2.4全部结果ZIP",
            type=["zip", "csv"], key="v26_prior_upload",
            help="推荐上传刚生成的 weekly_macd_two_feature_oos_v2_5_all_results.zip；也兼容V2.4全部结果ZIP和候选特征CSV。",
        )
        st.info("本版直接复用冻结的2371个候选，不重新下载行情，避免股票池和数据修订造成漂移。")
        st.header("冻结验证规则")
        st.write("三种形态完全分开")
        st.write("达到30%：对比全部30%以下")
        st.write("达到50%：对比全部50%以下")
        st.write("亏损否决：大涨股训练误删率不得超过15%")
        st.write("每条规则最多两个数值特征")
        st.write("行业名称不参与规则")
        st.write("留出年每年至少入选5只")
        st.write("翻倍只审计，不正式通过")

    session_key = "direct_target_oos_v26_zip"
    run_requested = st.button("开始V2.6直接目标跨年验证", type="primary", key="v26_run")
    if not run_requested:
        if session_key in st.session_state:
            st.success("上一次V2.6结果仍然保留，可直接下载。")
            st.download_button(
                "下载上一次全部结果ZIP", st.session_state[session_key],
                file_name="weekly_macd_direct_target_oos_v2_6_all_results.zip",
                mime="application/zip", key="v26_previous_download",
            )
        return
    if uploaded is None:
        st.error("请先上传V2.5或V2.4全部结果ZIP。")
        return
    try:
        with st.spinner("读取并核对冻结候选样本..."):
            candidates, input_summary = v26_load_prior_upload(uploaded)
    except Exception as exc:
        st.error(f"结果文件读取失败：{exc}")
        return
    if candidates.empty or candidates["Selection_Year"].nunique() < 3:
        st.error("成熟候选为空，或不足三个不同年份，无法进行整年留出验证。")
        return

    with st.spinner("逐形态、逐任务、逐年份学习并独立判卷..."):
        reports = v26_run_loyo(candidates)
        group_summary = five_group_summary(candidates)
        yearly = year_group_summary(candidates)
        paths = path_audit(candidates)
    aggregate = reports["aggregate"]
    completed = int(reports["folds"]["处理状态"].eq("完成").sum()) \
        if not reports["folds"].empty else 0
    passed = int(aggregate["验收结论"].str.startswith("通过候选").sum()) \
        if not aggregate.empty else 0
    potential_only = int(aggregate["验收结论"].str.startswith("潜力有提升").sum()) \
        if not aggregate.empty else 0
    counts = candidates["结果组"].value_counts().reindex(GROUP_ORDER, fill_value=0)
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "输入文件": uploaded.name,
        "成熟候选": len(candidates), "候选周": candidates.Selection_Date.nunique(),
        "不同股票": candidates.ts_code.nunique(),
        "上升趋势": int(candidates.Weekly_Trend.eq("上升趋势").sum()),
        "中性趋势": int(candidates.Weekly_Trend.eq("中性趋势").sum()),
        "下降趋势": int(candidates.Weekly_Trend.eq("下降趋势").sum()),
        "亏损且MFE<30": int(counts[LOSS_GROUP]),
        "盈利且MFE<30": int(counts[LOW_PROFIT_GROUP]),
        "MFE30至50": int(counts[MID_GROUP]),
        "MFE50至100": int(counts[HIGH_GROUP]),
        "MFE翻倍": int(counts[DOUBLE_GROUP]),
        "完成留出折叠": completed, "正式通过任务": passed,
        "仅潜力提升但风险不合格": potential_only,
    }])
    metadata = pd.DataFrame([
        ("程序定位", "直接目标与亏损否决的整年留出验证；不生成最终评分、不选Top3"),
        ("输入", "复用V2.4/V2.5冻结候选，避免重新下载造成数据和股票池漂移"),
        ("达到30%", "目标为30～50、50～100、翻倍；对照为亏损和低盈利的全部30%以下"),
        ("达到50%", "目标为50～100和翻倍；对照为亏损、低盈利、30～50的全部50%以下"),
        ("亏损否决", "目标为亏损组；对照为其余四组；训练期误删30%和50%以上均不得超过15%"),
        ("翻倍", "翻倍直接对比全部非翻倍；无论结果如何都禁止正式通过"),
        ("验证隔离", "每个留出年份的标签、分布和阈值不参与该折叠训练"),
        ("组合限制", "最多两个不同数值字段；训练期Spearman绝对相关>0.90禁止配对；行业名称不参与"),
        ("正式基础门槛", "至少3个可判卷年份且每年入选≥5、2/3年份改善、OOS提升≥25%、覆盖≥12%"),
        ("直接目标风险门槛", "入选亏损率相对未筛选候选至少下降20%"),
        ("否决风险门槛", "OOS误删30%以上和50%以上股票均不得超过10%"),
        ("集中度", "另行输出同一股票重复入选、单股占比和前5股票占比"),
    ], columns=["项目", "值"])
    files = {
        "01_run_summary_direct_target_oos_v2_6.csv": run_summary,
        "02_five_group_summary_direct_target_oos_v2_6.csv": group_summary,
        "03_task_definitions_direct_target_oos_v2_6.csv": v26_task_definitions(),
        "04_fold_sample_audit_direct_target_oos_v2_6.csv": reports["folds"],
        "05_train_only_single_rules_direct_target_oos_v2_6.csv": reports["singles"],
        "06_all_pair_oos_results_direct_target_oos_v2_6.csv": reports["pairs"],
        "07_best_pair_each_fold_direct_target_oos_v2_6.csv": reports["best"],
        "08_oos_acceptance_summary_direct_target_oos_v2_6.csv": aggregate,
        "09_pair_cross_year_stability_direct_target_oos_v2_6.csv": reports["stability"],
        "10_best_pair_selected_event_detail_direct_target_oos_v2_6.csv": reports["selected"],
        "11_selected_stock_concentration_direct_target_oos_v2_6.csv": reports["concentration"],
        "12_year_five_group_summary_direct_target_oos_v2_6.csv": yearly,
        "13_target_stop_path_audit_direct_target_oos_v2_6.csv": paths,
        "14_all_evaluation_candidate_features_direct_target_oos_v2_6.csv": candidates,
        "15_feature_dictionary_direct_target_oos_v2_6.csv": mining_feature_dictionary(),
        "16_metadata_direct_target_oos_v2_6.csv": metadata,
    }
    if not input_summary.empty:
        files["17_input_run_summary_direct_target_oos_v2_6.csv"] = input_summary
    result_zip = make_result_zip(files)
    st.session_state[session_key] = result_zip
    st.success(
        f"完成：{len(candidates)}个成熟候选，{completed}个留出折叠完成；"
        f"正式通过{passed}项，仅潜力提升但风险不合格{potential_only}项。"
    )
    st.subheader("直接目标与否决规则OOS验收")
    st.dataframe(aggregate, use_container_width=True, hide_index=True)
    st.subheader("各留出年份训练第一名组合")
    st.dataframe(reports["best"], use_container_width=True, hide_index=True)
    with st.expander("组合重复性、样本门槛与股票集中度"):
        st.dataframe(reports["stability"], use_container_width=True, hide_index=True)
        st.dataframe(reports["folds"], use_container_width=True, hide_index=True)
        st.dataframe(reports["concentration"], use_container_width=True, hide_index=True)
    st.download_button(
        "下载全部结果ZIP", result_zip,
        file_name="weekly_macd_direct_target_oos_v2_6_all_results.zip",
        mime="application/zip", type="primary", key="v26_download",
    )
    st.download_button(
        "单独下载8号验收总表", csv_bytes(aggregate),
        file_name="08_oos_acceptance_summary_direct_target_oos_v2_6.csv",
        mime="text/csv", key="v26_acceptance_download",
    )
    st.warning(
        "目标率提高但亏损率没有同步下降，只能称为行情潜力识别，不能进入实盘评分。"
        "禁止为了让某个任务通过而查看留出年后修改分位阈值。"
    )


# =============================================================================
# V2.7：止损先到负向特征、环境变量与误删代价前沿
# =============================================================================
TITLE = "科技股周线MACD止损先到特征与误删代价前沿实验器 V2.7"
VERSION = "V2.7-STOP-FIRST-VETO-COST-FRONTIER"
V27_BUDGETS = (20, 30, 40, 50)
V27_MAX_BASE_RULES = 20
V27_MIN_TRAIN_RESOLVED_EACH = 30
V27_MIN_TEST_RESOLVED_EACH = 5

V27_ENV_FEATURES = {
    "科技池样本数": "Env_Tech_N",
    "科技池站上MA20比例(%)": "Env_Tech_Above_MA20_pct",
    "科技池站上MA60比例(%)": "Env_Tech_Above_MA60_pct",
    "科技池红柱比例(%)": "Env_Tech_Red_pct",
    "科技池红柱扩张比例(%)": "Env_Tech_Red_Expand_pct",
    "科技池4周收益中位数(%)": "Env_Tech_Return4W_Median_pct",
    "科技池8周收益中位数(%)": "Env_Tech_Return8W_Median_pct",
    "科技池13周收益中位数(%)": "Env_Tech_Return13W_Median_pct",
    "科技池量能扩张中位数": "Env_Tech_Volume_Ratio_Median",
    "科技池换手扩张中位数": "Env_Tech_Turnover_Ratio_Median",
    "行业样本数": "Env_Industry_N",
    "行业站上MA20比例(%)": "Env_Industry_Above_MA20_pct",
    "行业站上MA60比例(%)": "Env_Industry_Above_MA60_pct",
    "行业红柱比例(%)": "Env_Industry_Red_pct",
    "行业红柱扩张比例(%)": "Env_Industry_Red_Expand_pct",
    "行业4周收益中位数(%)": "Env_Industry_Return4W_Median_pct",
    "行业8周收益中位数(%)": "Env_Industry_Return8W_Median_pct",
    "行业13周收益中位数(%)": "Env_Industry_Return13W_Median_pct",
    "行业量能扩张中位数": "Env_Industry_Volume_Ratio_Median",
    "行业换手扩张中位数": "Env_Industry_Turnover_Ratio_Median",
    "个股相对行业4周强度": "Env_Stock_RS_Industry4W",
    "个股相对行业8周强度": "Env_Stock_RS_Industry8W",
    "个股相对行业13周强度": "Env_Stock_RS_Industry13W",
    "板块样本数": "Env_Board_N",
    "板块站上MA20比例(%)": "Env_Board_Above_MA20_pct",
    "板块站上MA60比例(%)": "Env_Board_Above_MA60_pct",
    "板块红柱扩张比例(%)": "Env_Board_Red_Expand_pct",
    "板块13周收益中位数(%)": "Env_Board_Return13W_Median_pct",
    "板块整体风险得分": "Env_Board_Risk_Score",
}
V27_ALL_FEATURES = {**NUMERIC_MINING_FEATURES, **V27_ENV_FEATURES}


def v27_stock_environment_rows(stock: pd.Series,
                               stock_periods: list[dict[str, str]],
                               daily: pd.DataFrame,
                               daily_basic: pd.DataFrame,
                               week_map: dict[pd.Timestamp, str],
                               start_date: str,
                               end_date: str) -> list[dict[str, Any]]:
    """构造一只股票在每个完整周末的当时环境贡献；不使用未来数据。"""
    weekly = build_weekly(daily, week_map)
    if weekly.empty:
        return []
    weekly = weekly.copy()
    weekly["return_4w_pct"] = (weekly["close"] / weekly["close"].shift(4) - 1.0) * 100.0
    weekly["return_8w_pct"] = (weekly["close"] / weekly["close"].shift(8) - 1.0) * 100.0
    weekly["ma60"] = weekly["close"].rolling(60).mean()
    weekly["hist_prev"] = weekly["hist"].shift(1)
    if daily_basic.empty:
        weekly["raw_close"] = np.nan
        weekly["circ_mv"] = np.nan
        weekly["turnover_rate"] = np.nan
    else:
        basic = daily_basic[[c for c in ["trade_date", "close", "circ_mv", "turnover_rate"]
                             if c in daily_basic]].copy()
        basic = basic.rename(columns={"close": "raw_close"})
        weekly = weekly.merge(basic.drop_duplicates("trade_date", keep="last"),
                              on="trade_date", how="left")
    turnover = pd.to_numeric(weekly.get("turnover_rate"), errors="coerce")
    weekly["turnover_median13_prev"] = turnover.shift(1).rolling(13).median()
    weekly["turnover_ratio13"] = turnover / weekly["turnover_median13_prev"]
    board = sample_board(stock)
    rows: list[dict[str, Any]] = []
    for _, row in weekly.iterrows():
        trade_date = normalize_date(row.get("trade_date"))
        if not trade_date or trade_date < start_date or trade_date > end_date:
            continue
        membership = membership_on_date(stock_periods, trade_date)
        if membership is None:
            continue
        raw_close = finite_num(row.get("raw_close"))
        circ_mv = finite_num(row.get("circ_mv"))
        circ_billion = circ_mv / 10000.0 if math.isfinite(circ_mv) else np.nan
        # 环境宽度与策略股票池保持同一价格、市值口径。
        if (not math.isfinite(raw_close) or raw_close < 10.0
                or not math.isfinite(circ_billion) or circ_billion < 100.0):
            continue
        close = finite_num(row.get("close"))
        ma20 = finite_num(row.get("ma20"))
        ma60 = finite_num(row.get("ma60"))
        hist = finite_num(row.get("hist"))
        hist_prev = finite_num(row.get("hist_prev"))
        rows.append({
            "Selection_Date": trade_date, "ts_code": str(stock.get("ts_code", "")),
            "Sample_Board": board, "SW_L1": membership.get("l1", ""),
            "SW_L2": membership.get("l2", ""),
            "EnvStock_Return4W_pct": finite_num(row.get("return_4w_pct")),
            "EnvStock_Return8W_pct": finite_num(row.get("return_8w_pct")),
            "EnvStock_Return13W_pct": finite_num(row.get("return_13w_pct")),
            "EnvStock_Above_MA20": math.isfinite(close) and math.isfinite(ma20) and close > ma20,
            "EnvStock_Above_MA60": math.isfinite(close) and math.isfinite(ma60) and close > ma60,
            "EnvStock_Red": math.isfinite(hist) and hist > 0,
            "EnvStock_Red_Expand": (math.isfinite(hist) and math.isfinite(hist_prev)
                                    and hist > 0 and hist > hist_prev),
            "EnvStock_Volume_Ratio": finite_num(row.get("vol_ratio20")),
            "EnvStock_Turnover_Ratio": finite_num(row.get("turnover_ratio13")),
        })
    return rows


def v27_environment_aggregate(detail: pd.DataFrame,
                              group_columns: list[str],
                              prefix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if detail.empty:
        return pd.DataFrame()
    grouped = detail.groupby(group_columns, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update({
            f"{prefix}_N": len(group),
            f"{prefix}_Above_MA20_pct": rate(group["EnvStock_Above_MA20"]),
            f"{prefix}_Above_MA60_pct": rate(group["EnvStock_Above_MA60"]),
            f"{prefix}_Red_pct": rate(group["EnvStock_Red"]),
            f"{prefix}_Red_Expand_pct": rate(group["EnvStock_Red_Expand"]),
            f"{prefix}_Return4W_Median_pct": num(group["EnvStock_Return4W_pct"]).median(),
            f"{prefix}_Return8W_Median_pct": num(group["EnvStock_Return8W_pct"]).median(),
            f"{prefix}_Return13W_Median_pct": num(group["EnvStock_Return13W_pct"]).median(),
            f"{prefix}_Volume_Ratio_Median": num(group["EnvStock_Volume_Ratio"]).median(),
            f"{prefix}_Turnover_Ratio_Median": num(group["EnvStock_Turnover_Ratio"]).median(),
        })
        row[f"{prefix}_Risk_Score"] = (
            0.35 * row[f"{prefix}_Above_MA20_pct"]
            + 0.25 * row[f"{prefix}_Above_MA60_pct"]
            + 0.25 * row[f"{prefix}_Red_pct"]
            + 0.15 * row[f"{prefix}_Red_Expand_pct"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def v27_enrich_environment(candidates: pd.DataFrame,
                           detail: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    out = candidates.copy()
    out["Selection_Date"] = out["Selection_Date"].map(normalize_date)
    if detail.empty:
        for field in V27_ENV_FEATURES.values():
            out[field] = np.nan
        return out, {"detail": detail, "tech": pd.DataFrame(),
                     "industry": pd.DataFrame(), "board": pd.DataFrame()}
    detail = detail.copy()
    detail["Selection_Date"] = detail["Selection_Date"].map(normalize_date)
    tech = v27_environment_aggregate(detail, ["Selection_Date"], "Env_Tech")
    industry = v27_environment_aggregate(
        detail, ["Selection_Date", "SW_L2"], "Env_Industry"
    )
    board = v27_environment_aggregate(
        detail, ["Selection_Date", "Sample_Board"], "Env_Board"
    )
    stock_fields = detail[["Selection_Date", "ts_code", "EnvStock_Return4W_pct",
                           "EnvStock_Return8W_pct", "EnvStock_Return13W_pct"]].drop_duplicates(
                               ["Selection_Date", "ts_code"], keep="last")
    out = out.merge(stock_fields, on=["Selection_Date", "ts_code"], how="left")
    out = out.merge(tech, on="Selection_Date", how="left")
    out = out.merge(industry, on=["Selection_Date", "SW_L2"], how="left")
    out = out.merge(board, on=["Selection_Date", "Sample_Board"], how="left")
    out["Env_Stock_RS_Industry4W"] = (
        num(out["EnvStock_Return4W_pct"]) - num(out["Env_Industry_Return4W_Median_pct"])
    )
    out["Env_Stock_RS_Industry8W"] = (
        num(out["EnvStock_Return8W_pct"]) - num(out["Env_Industry_Return8W_Median_pct"])
    )
    out["Env_Stock_RS_Industry13W"] = (
        num(out["EnvStock_Return13W_pct"]) - num(out["Env_Industry_Return13W_Median_pct"])
    )
    return out, {"detail": detail, "tech": tech, "industry": industry, "board": board}


def v27_path_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    status30 = out["CP_W2_Delayed_First_30_vs_Stop"].fillna("").astype(str)
    out["V27_Path_Label"] = np.select(
        [status30.eq("目标先到"), status30.isin(["止损先到", "同日不确定_按止损"])],
        ["30%目标先到", "-10%止损先到"], default="八周均未触发",
    )
    out["V27_Resolved"] = out["V27_Path_Label"].isin(["30%目标先到", "-10%止损先到"])
    out["V27_Stop_First"] = out["V27_Path_Label"].eq("-10%止损先到")
    out["V27_Target_First"] = out["V27_Path_Label"].eq("30%目标先到")
    return out


def v27_rule_mask(frame: pd.DataFrame, rule: pd.Series | dict[str, Any]) -> pd.Series:
    field = str(rule["字段"])
    return v25_rule_mask(
        frame.get(field, pd.Series(np.nan, index=frame.index)),
        finite_num(rule.get("下限", np.nan)), finite_num(rule.get("上限", np.nan)),
    )


def v27_veto_metrics(frame: pd.DataFrame,
                     veto: pd.Series) -> dict[str, Any]:
    veto = veto.reindex(frame.index, fill_value=False).astype(bool)
    resolved = frame["V27_Resolved"].astype(bool)
    stops = frame["V27_Stop_First"].astype(bool)
    targets = frame["V27_Target_First"].astype(bool)
    retained = ~veto
    stop_total = int(stops.sum())
    target_total = int(targets.sum())
    veto_stop = int((veto & stops).sum())
    veto_target = int((veto & targets).sum())
    retained_stop = int((retained & stops).sum())
    retained_target = int((retained & targets).sum())
    baseline_target_rate = target_total / (stop_total + target_total) if stop_total + target_total else np.nan
    retained_target_rate = retained_target / (retained_stop + retained_target) \
        if retained_stop + retained_target else np.nan
    status50 = frame["CP_W2_Delayed_First_50_vs_Stop"].fillna("").astype(str)
    status100 = frame["CP_W2_Delayed_First_100_vs_Stop"].fillna("").astype(str)
    target50 = status50.eq("目标先到")
    target100 = status100.eq("目标先到")
    utility = num(frame.get("Realised_Utility"), frame.index)
    selection_dates = frame.get("Selection_Date", pd.Series("", index=frame.index)).astype(str)
    all_weeks = selection_dates[selection_dates.ne("")].nunique()
    retained_weeks = selection_dates[retained & selection_dates.ne("")].nunique()
    retained_count = int(retained.sum())
    result = {
        "全部样本": len(frame), "已决样本": int(resolved.sum()),
        "止损先到总数": stop_total, "目标先到总数": target_total,
        "否决总数": int(veto.sum()), "否决止损先到数": veto_stop,
        "误删目标先到数": veto_target,
        "止损剔除率(%)": veto_stop / stop_total * 100.0 if stop_total else np.nan,
        "目标误删率(%)": veto_target / target_total * 100.0 if target_total else np.nan,
        "否决命中止损精度(%)": veto_stop / int((veto & resolved).sum()) * 100.0
        if int((veto & resolved).sum()) else np.nan,
        "保留止损先到数": retained_stop, "保留目标先到数": retained_target,
        "原始目标先到率(%)": baseline_target_rate * 100.0
        if math.isfinite(baseline_target_rate) else np.nan,
        "保留池目标先到率(%)": retained_target_rate * 100.0
        if math.isfinite(retained_target_rate) else np.nan,
        "目标先到率提升(百分点)": (retained_target_rate - baseline_target_rate) * 100.0
        if math.isfinite(baseline_target_rate) and math.isfinite(retained_target_rate) else np.nan,
        "止损剔除/目标误删倍数": (veto_stop / stop_total) / (veto_target / target_total)
        if stop_total and target_total and veto_target else np.inf if veto_stop else np.nan,
        "50%目标误删率(%)": int((veto & target50).sum()) / int(target50.sum()) * 100.0
        if int(target50.sum()) else np.nan,
        "100%目标误删率(%)": int((veto & target100).sum()) / int(target100.sum()) * 100.0
        if int(target100.sum()) else np.nan,
        "原始效用均值": utility.mean(), "保留池效用均值": utility[retained].mean(),
        "原始效用中位数": utility.median(), "保留池效用中位数": utility[retained].median(),
        "保留事件": retained_count, "原始信号周": all_weeks,
        "保留信号周": retained_weeks, "新增空窗周": all_weeks - retained_weeks,
        "保留周平均事件": retained_count / retained_weeks if retained_weeks else np.nan,
    }
    for group_name in GROUP_ORDER:
        group_mask = frame["结果组"].eq(group_name)
        total = int(group_mask.sum())
        result[f"{group_name}_总数"] = total
        result[f"{group_name}_误删数"] = int((veto & group_mask).sum())
        result[f"{group_name}_误删率(%)"] = int((veto & group_mask).sum()) / total * 100.0 \
            if total else np.nan
    return result


def v27_learn_base_rules(train: pd.DataFrame) -> pd.DataFrame:
    resolved = train[train["V27_Resolved"]].copy()
    labels = resolved["V27_Stop_First"].astype(bool)
    rows: list[dict[str, Any]] = []
    for feature_label, field in V27_ALL_FEATURES.items():
        if field not in train:
            continue
        values = pd.to_numeric(resolved[field], errors="coerce")
        if values.notna().sum() < 40 or values.nunique(dropna=True) < 8:
            continue
        for shape_name, lower_q, upper_q in V25_RULE_SHAPES:
            lower = float(values.quantile(lower_q)) if lower_q is not None else np.nan
            upper = float(values.quantile(upper_q)) if upper_q is not None else np.nan
            veto_resolved = v25_rule_mask(values, lower, upper)
            binary = v25_binary_metrics(labels, veto_resolved)
            if (binary["入选总数"] < 8 or binary["目标覆盖率(%)"] < 5.0
                    or not math.isfinite(binary["相对基准提升倍数"])
                    or binary["相对基准提升倍数"] <= 1.02):
                continue
            rule = {"字段": field, "下限": lower, "上限": upper}
            veto_all = v27_rule_mask(train, rule)
            metrics = v27_veto_metrics(train, veto_all)
            efficiency = metrics["止损剔除/目标误删倍数"]
            efficiency_score = min(efficiency, 10.0) if math.isfinite(efficiency) else 10.0
            score = metrics["止损剔除率(%)"] + efficiency_score * 3.0
            rows.append({
                "特征": feature_label, "字段": field, "规则形状": shape_name,
                "下限": lower, "上限": upper, "规则类型": "单规则",
                "规则签名": f"{field}[{shape_name}]", "训练风险分": score,
                **{f"训练_{key}": value for key, value in metrics.items()},
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # 分别为20/30/40/50%误删预算保留候选，避免高预算规则把低误删规则挤出组合搜索。
    result = result.drop_duplicates(["字段", "规则形状"], keep="first")
    selected: list[pd.DataFrame] = []
    per_budget = max(4, V27_MAX_BASE_RULES // len(V27_BUDGETS) + 1)
    for budget in V27_BUDGETS:
        eligible = result[num(result["训练_目标误删率(%)"]).le(float(budget))].copy()
        eligible = eligible.sort_values(
            ["训练_止损剔除率(%)", "训练_止损剔除/目标误删倍数", "训练风险分"],
            ascending=False,
        )
        selected.append(eligible.head(per_budget))
    result = pd.concat(selected, ignore_index=True) if selected else result.head(0)
    result = result.drop_duplicates("规则签名", keep="first")
    result = result.sort_values("训练风险分", ascending=False)
    return result.head(V27_MAX_BASE_RULES).reset_index(drop=True)


def v27_candidate_veto_rules(train: pd.DataFrame,
                             base_rules: pd.DataFrame) -> pd.DataFrame:
    if base_rules.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    base_masks: list[pd.Series] = []
    for _, rule in base_rules.iterrows():
        mask = v27_rule_mask(train, rule)
        base_masks.append(mask)
        rows.append(rule.to_dict())
    for left_i in range(len(base_rules)):
        for right_i in range(left_i + 1, len(base_rules)):
            left = base_rules.iloc[left_i]
            right = base_rules.iloc[right_i]
            if left["字段"] == right["字段"]:
                continue
            for logic, mask in (
                ("任一命中OR", base_masks[left_i] | base_masks[right_i]),
                ("同时命中AND", base_masks[left_i] & base_masks[right_i]),
            ):
                metrics = v27_veto_metrics(train, mask)
                if metrics["否决总数"] < 8 or metrics["否决止损先到数"] < 5:
                    continue
                efficiency = metrics["止损剔除/目标误删倍数"]
                efficiency_score = min(efficiency, 10.0) if math.isfinite(efficiency) else 10.0
                score = metrics["止损剔除率(%)"] + efficiency_score * 3.0
                rows.append({
                    "特征": f"{left['特征']} + {right['特征']}",
                    "字段": f"{left['字段']}|{right['字段']}",
                    "规则形状": f"{left['规则形状']}|{right['规则形状']}",
                    "下限": np.nan, "上限": np.nan, "规则类型": logic,
                    "字段1": left["字段"], "形状1": left["规则形状"],
                    "下限1": left["下限"], "上限1": left["上限"],
                    "字段2": right["字段"], "形状2": right["规则形状"],
                    "下限2": right["下限"], "上限2": right["上限"],
                    "规则签名": f"{left['规则签名']} {logic} {right['规则签名']}",
                    "训练风险分": score,
                    **{f"训练_{key}": value for key, value in metrics.items()},
                })
    return pd.DataFrame(rows)


def v27_apply_veto(frame: pd.DataFrame, rule: pd.Series | dict[str, Any]) -> pd.Series:
    rule_type = str(rule.get("规则类型", "单规则"))
    if rule_type == "单规则":
        return v27_rule_mask(frame, rule)
    left = {"字段": rule["字段1"], "下限": rule.get("下限1", np.nan),
            "上限": rule.get("上限1", np.nan)}
    right = {"字段": rule["字段2"], "下限": rule.get("下限2", np.nan),
             "上限": rule.get("上限2", np.nan)}
    left_mask = v27_rule_mask(frame, left)
    right_mask = v27_rule_mask(frame, right)
    return left_mask | right_mask if rule_type == "任一命中OR" else left_mask & right_mask


def v27_frontier_choice(candidates: pd.DataFrame,
                        budget: int) -> pd.Series | None:
    if candidates.empty:
        return None
    eligible = candidates[
        num(candidates["训练_目标误删率(%)"]).le(float(budget))
        & num(candidates["训练_止损剔除率(%)"]).gt(0)
    ].copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values(
        ["训练_止损剔除率(%)", "训练_止损剔除/目标误删倍数", "训练风险分"],
        ascending=False,
    )
    return eligible.iloc[0]


def v27_time_folds(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    dates = pd.to_datetime(candidates["Selection_Date"], format="%Y%m%d", errors="coerce")
    min_date, max_date = dates.min(), dates.max()
    proposed = [
        ("2025上半年", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
        ("2025下半年", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31")),
        ("2026上半年", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-30")),
    ]
    folds = []
    for name, start, end in proposed:
        if start > max_date or end < min_date:
            continue
        folds.append({"测试期": name, "测试开始": start,
                      "测试结束": min(end, max_date)})
    return folds


def v27_run_frontier(candidates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = v27_path_labels(candidates)
    data["Selection_Date_dt"] = pd.to_datetime(
        data["Selection_Date"], format="%Y%m%d", errors="coerce"
    )
    maturity = pd.to_datetime(
        data.get("Outcome_Maturity_Date_dt"), errors="coerce"
    )
    if maturity.isna().all():
        maturity = data["Selection_Date_dt"] + pd.Timedelta(days=56)
    data["V27_Maturity_Date"] = maturity
    folds = v27_time_folds(data)
    fold_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for state in STATE_ORDER:
        state_data = data[data["Weekly_Trend"].eq(state)].copy()
        for fold in folds:
            train = state_data[
                state_data["Selection_Date_dt"].lt(fold["测试开始"])
                & state_data["V27_Maturity_Date"].lt(fold["测试开始"])
            ].copy()
            test = state_data[
                state_data["Selection_Date_dt"].ge(fold["测试开始"])
                & state_data["Selection_Date_dt"].le(fold["测试结束"])
            ].copy()
            train_stop = int(train["V27_Stop_First"].sum())
            train_target = int(train["V27_Target_First"].sum())
            test_stop = int(test["V27_Stop_First"].sum())
            test_target = int(test["V27_Target_First"].sum())
            eligible = (train_stop >= V27_MIN_TRAIN_RESOLVED_EACH
                        and train_target >= V27_MIN_TRAIN_RESOLVED_EACH
                        and test_stop >= V27_MIN_TEST_RESOLVED_EACH
                        and test_target >= V27_MIN_TEST_RESOLVED_EACH)
            audit = {
                "红柱形态": state, "测试期": fold["测试期"],
                "训练截止": (fold["测试开始"] - pd.Timedelta(days=1)).strftime("%Y%m%d"),
                "测试开始": fold["测试开始"].strftime("%Y%m%d"),
                "测试结束": fold["测试结束"].strftime("%Y%m%d"),
                "训练样本": len(train), "训练止损先到": train_stop,
                "训练目标先到": train_target, "测试样本": len(test),
                "测试止损先到": test_stop, "测试目标先到": test_target,
                "样本门槛通过": eligible,
            }
            if not eligible:
                audit["处理状态"] = "样本不足"
                fold_rows.append(audit)
                continue
            base = v27_learn_base_rules(train)
            if base.empty:
                audit["处理状态"] = "无有效负向单规则"
                fold_rows.append(audit)
                continue
            all_rules = v27_candidate_veto_rules(train, base)
            if all_rules.empty:
                audit["处理状态"] = "无有效候选否决规则"
                fold_rows.append(audit)
                continue
            audit.update({"处理状态": "完成", "基础规则": len(base),
                          "单项与组合规则": len(all_rules)})
            fold_rows.append(audit)
            for _, row in base.iterrows():
                base_rows.append({"红柱形态": state, "测试期": fold["测试期"], **row.to_dict()})
            for _, row in all_rules.iterrows():
                candidate_rows.append({"红柱形态": state, "测试期": fold["测试期"], **row.to_dict()})
            for budget in V27_BUDGETS:
                chosen = v27_frontier_choice(all_rules, budget)
                if chosen is None:
                    frontier_rows.append({
                        "红柱形态": state, "测试期": fold["测试期"],
                        "允许目标误删预算(%)": budget, "处理状态": "训练期无规则满足预算",
                    })
                    continue
                test_veto = v27_apply_veto(test, chosen)
                test_metrics = v27_veto_metrics(test, test_veto)
                frontier_rows.append({
                    "红柱形态": state, "测试期": fold["测试期"],
                    "允许目标误删预算(%)": budget, "处理状态": "完成",
                    "规则签名": chosen["规则签名"], "规则类型": chosen["规则类型"],
                    **{key: chosen.get(key, np.nan) for key in
                       ["字段", "规则形状", "下限", "上限", "字段1", "形状1", "下限1", "上限1",
                        "字段2", "形状2", "下限2", "上限2"]},
                    **{key: value for key, value in chosen.items() if str(key).startswith("训练_")},
                    **{f"测试_{key}": value for key, value in test_metrics.items()},
                })
                retained = test[~test_veto]
                for _, event in retained.iterrows():
                    selected_rows.append({
                        "红柱形态": state, "测试期": fold["测试期"],
                        "允许目标误删预算(%)": budget, "规则签名": chosen["规则签名"],
                        "Selection_Date": event.get("Selection_Date", ""),
                        "ts_code": event.get("ts_code", ""), "name": event.get("name", ""),
                        "结果组": event.get("结果组", ""), "路径标签": event.get("V27_Path_Label", ""),
                        "Realised_Utility": event.get("Realised_Utility", np.nan),
                        "MFE8周(%)": event.get("CP_W2_Delayed_MFE_8W_pct", np.nan),
                        "MAE8周(%)": event.get("CP_W2_Delayed_MAE_8W_pct", np.nan),
                    })
    return {
        "labeled": data, "folds": pd.DataFrame(fold_rows),
        "base_rules": pd.DataFrame(base_rows), "candidate_rules": pd.DataFrame(candidate_rows),
        "frontier": pd.DataFrame(frontier_rows), "retained": pd.DataFrame(selected_rows),
    }


def v27_frontier_summary(frontier: pd.DataFrame) -> pd.DataFrame:
    if frontier.empty:
        return pd.DataFrame()
    complete = frontier[frontier["处理状态"].eq("完成")].copy()
    rows: list[dict[str, Any]] = []
    for (state, budget), group in complete.groupby(["红柱形态", "允许目标误删预算(%)"]):
        stop_total = num(group["测试_止损先到总数"]).sum()
        target_total = num(group["测试_目标先到总数"]).sum()
        veto_stop = num(group["测试_否决止损先到数"]).sum()
        veto_target = num(group["测试_误删目标先到数"]).sum()
        retained_stop = stop_total - veto_stop
        retained_target = target_total - veto_target
        stop_rate = veto_stop / stop_total * 100.0 if stop_total else np.nan
        target_error = veto_target / target_total * 100.0 if target_total else np.nan
        base_rate = target_total / (target_total + stop_total) * 100.0 \
            if target_total + stop_total else np.nan
        kept_rate = retained_target / (retained_target + retained_stop) * 100.0 \
            if retained_target + retained_stop else np.nan
        improved_folds = int((num(group["测试_目标先到率提升(百分点)"]) > 0).sum())
        efficiency = stop_rate / target_error if math.isfinite(target_error) and target_error > 0 \
            else np.inf if math.isfinite(stop_rate) and stop_rate > 0 else np.nan
        if (len(group) >= 2 and math.isfinite(stop_rate) and math.isfinite(target_error)
                and stop_rate >= target_error * 1.25 and improved_folds >= 2
                and math.isfinite(kept_rate) and kept_rate >= base_rate + 3.0):
            verdict = "OOS有效前沿候选"
        elif math.isfinite(stop_rate) and math.isfinite(target_error) and stop_rate > target_error:
            verdict = "有一定净剔除，但证据不足"
        else:
            verdict = "误删代价不划算"
        rows.append({
            "红柱形态": state, "允许目标误删预算(%)": budget,
            "完成测试期": len(group), "改善测试期": improved_folds,
            "OOS止损先到总数": int(stop_total), "OOS目标先到总数": int(target_total),
            "OOS剔除止损数": int(veto_stop), "OOS误删目标数": int(veto_target),
            "OOS止损剔除率(%)": stop_rate, "OOS目标误删率(%)": target_error,
            "OOS净剔除优势(百分点)": stop_rate - target_error,
            "OOS止损剔除/目标误删倍数": efficiency,
            "OOS原始目标先到率(%)": base_rate, "OOS保留池目标先到率(%)": kept_rate,
            "OOS目标先到率提升(百分点)": kept_rate - base_rate,
            "30至50组OOS误删数": int(num(group.get("测试_30%～50%_误删数", 0)).sum()),
            "50至100组OOS误删数": int(num(group.get("测试_50%～100%_误删数", 0)).sum()),
            "翻倍组OOS误删数": int(num(group.get("测试_翻倍以上_误删数", 0)).sum()),
            "前沿判断": verdict,
        })
    return pd.DataFrame(rows).sort_values(["红柱形态", "允许目标误删预算(%)"])


def v27_path_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    labeled = v27_path_labels(candidates)
    rows = []
    for state in STATE_ORDER:
        group = labeled[labeled["Weekly_Trend"].eq(state)]
        for label in ("30%目标先到", "-10%止损先到", "八周均未触发"):
            count = int(group["V27_Path_Label"].eq(label).sum())
            rows.append({"红柱形态": state, "路径": label, "数量": count,
                         "形态内占比(%)": count / len(group) * 100.0 if len(group) else np.nan})
    return pd.DataFrame(rows)


def v27_retained_concentration(retained: pd.DataFrame) -> pd.DataFrame:
    if retained.empty:
        return pd.DataFrame()
    rows = []
    keys = ["红柱形态", "测试期", "允许目标误删预算(%)"]
    for values, group in retained.groupby(keys):
        counts = group["ts_code"].astype(str).value_counts()
        rows.append({
            "红柱形态": values[0], "测试期": values[1],
            "允许目标误删预算(%)": values[2], "保留事件": len(group),
            "不同股票": group["ts_code"].nunique(),
            "单股最大占比(%)": counts.iloc[0] / len(group) * 100.0 if len(group) else np.nan,
            "前5股占比(%)": counts.head(5).sum() / len(group) * 100.0 if len(group) else np.nan,
        })
    return pd.DataFrame(rows)


def v27_feature_dictionary() -> pd.DataFrame:
    rows = []
    for label, field in V27_ALL_FEATURES.items():
        source = "当周全科技股/行业/板块截面" if field.startswith("Env_") else "个股第二根完整红柱"
        rows.append({"特征": label, "字段": field, "使用时点": "选股当时已知",
                     "来源": source, "用途": "仅训练负向否决规则，不评分"})
    return pd.DataFrame(rows)


def v27_main_legacy() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption(
        "分别在上升、中性、下降形态内，学习“-10%止损先到”特征；"
        "用20%/30%/40%/50%的目标误删预算检查代价前沿。本版不评分、不排名。"
    )
    with st.sidebar:
        st.header("正式评价区间")
        eval_date = st.date_input("评价开始", value=date(2023, 6, 5), key="v27_eval")
        end_date = st.date_input("信号截止", value=date(2026, 6, 5), key="v27_end")
        obs_date = st.date_input("行情观察截止", value=date.today(), max_value=date.today(), key="v27_obs")
        st.header("冻结实验口径")
        st.write("坏样本：-10%止损早于+30%目标")
        st.write("好样本：+30%目标早于-10%止损")
        st.write("八周均未触发单独审计，不参与学习标签")
        st.write("训练误删预算：20% / 30% / 40% / 50%")
        st.write("时间折：2025上半年、2025下半年、2026上半年")
        st.write("每个测试期只用当时已成熟的历史样本")
        st.write("加入科技宽度、行业强度、红柱扩散、量价和板块风险")
        cache = st.checkbox("使用逐股票缓存", value=True, key="v27_cache")
        pause = st.number_input("每次API调用后暂停(秒)", 0.0, 3.0, 0.12, 0.05, key="v27_pause")
        if st.button("清除本程序缓存", key="v27_clear"):
            if os.path.isdir(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            st.success("缓存已清除")
    token = st.text_input("Tushare Token", type="password", key="v27_token")
    if not token:
        st.info("请输入Tushare Token。本版要重建每周全科技股环境，不能只上传V2.6结果。")
        return
    session_key = "stop_first_veto_frontier_v27_zip"
    if not st.button("开始V2.7止损先到特征与代价前沿实验", type="primary", key="v27_run"):
        if session_key in st.session_state:
            st.download_button("下载上一次全部结果ZIP", st.session_state[session_key],
                               file_name="weekly_macd_stop_first_veto_frontier_v2_7_all_results.zip",
                               mime="application/zip", key="v27_previous_download")
        return
    if eval_date >= end_date or end_date > obs_date:
        st.error("日期关系不正确。")
        return

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    eval_start = pd.Timestamp(eval_date)
    research, end, obs = (eval_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"),
                          obs_date.strftime("%Y%m%d"))
    preload = (eval_date - timedelta(days=3 * 365)).strftime("%Y%m%d")
    config = {
        "signal_start": research, "signal_end": end, "market_end": obs,
        "preload_start": preload, "min_price": 10.0, "min_mv": 100.0,
        "max_mv": 1_000_000_000.0, "price_tolerance_pct": 3.0,
        "stop_threshold_pct": 10.0, "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20,
        "sample_per_board": 0, "sample_seed": DEFAULT_SAMPLE_SEED,
        "long_cycle_min_weeks": DEFAULT_LONG_CYCLE_MIN_WEEKS,
        "material_hist_change_pct": DEFAULT_MATERIAL_HIST_CHANGE_PCT,
        "short_strength_ratio": DEFAULT_SHORT_STRENGTH_RATIO,
    }
    try:
        with st.spinner("加载交易日历、历史科技股池和板块指数..."):
            opens = load_trade_calendar(preload, obs)
            full = load_trade_calendar(preload, (obs_date + timedelta(days=7)).strftime("%Y%m%d"))
            basic = load_stock_basic()
            memberships = load_sw_tech_memberships(float(pause))
            week_map = complete_week_last_dates(full)
            boards: dict[str, pd.DataFrame] = {}
            for code in sorted(set(BOARD_INDEX.values())):
                board_daily = fetch_index_history(code, preload, obs, bool(cache), float(pause))
                if not board_daily.empty:
                    boards[code] = build_weekly(board_daily, week_map)
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return

    periods = build_period_index(memberships)
    codes = sorted(set(periods) & set(basic.ts_code.astype(str)))
    universe = basic[basic.ts_code.isin(codes)].copy()
    stocks, universe_audit, population = build_stratified_sample(
        universe, periods, end, 0, DEFAULT_SAMPLE_SEED
    )
    listed = stocks.list_date.apply(lambda x: normalize_date(x, "19000101"))
    delisted = stocks.delist_date.apply(lambda x: normalize_date(x, "99991231"))
    stocks = stocks[~listed.gt(end) & ~delisted.lt(preload)].reset_index(drop=True)
    open_pos = {trade_date: i for i, trade_date in enumerate(opens)}
    records: list[dict[str, Any]] = []
    environment_rows: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}
    rejects: dict[str, int] = {}
    cache_hits = data_failures = 0
    progress = st.progress(0.0)
    status = st.empty()
    for i, stock in stocks.iterrows():
        code = str(stock.ts_code)
        progress.progress((i + 1) / len(stocks), text=f"{i + 1}/{len(stocks)} {code}")
        status.caption(f"事件{len(records)}；环境截面{len(environment_rows)}；缓存{cache_hits}；失败{data_failures}")
        daily, daily_basic, hit = fetch_stock_history(code, preload, obs, bool(cache), float(pause))
        cache_hits += int(hit)
        if daily.empty:
            data_failures += 1
            continue
        environment_rows.extend(v27_stock_environment_rows(
            stock, periods.get(code, []), daily, daily_basic, week_map, research, end
        ))
        stock_records, stock_rejects, _ = analyze_stock(
            stock, periods.get(code, []), daily, daily_basic, week_map, opens, open_pos, config
        )
        records.extend(stock_records)
        if stock_records:
            histories[code] = daily.copy()
        for reason, count in stock_rejects.items():
            rejects[reason] = rejects.get(reason, 0) + count
    progress.empty()
    status.empty()
    if not records:
        st.error("没有生成事件。")
        return

    try:
        with st.spinner("重建当时环境、生成三形态路径标签并运行三段按时间OOS验证..."):
            events = pd.DataFrame(records).sort_values(["Signal_Date", "ts_code", "Event_Type"])
            opportunities = build_cycle_opportunities(events, histories, obs, config["sell_slippage_pct"])
            featured = prepare_features(opportunities, boards)
            candidates = featured[
                featured.Strict_Eligible.map(bool_value)
                & featured.Outcome_Mature.map(bool_value)
                & featured.Selection_Date_dt.ge(eval_start)
            ].copy()
            candidates = v25_prepare_candidates(candidates)
            environment_detail = pd.DataFrame(environment_rows)
            enriched, panels = v27_enrich_environment(candidates, environment_detail)
            results = v27_run_frontier(enriched)
            frontier_summary = v27_frontier_summary(results["frontier"])
            path_summary = v27_path_summary(enriched)
            concentration = v27_retained_concentration(results["retained"])
    except Exception as exc:
        st.error(f"V2.7实验失败：{exc}")
        return

    env_coverage = pd.DataFrame([
        {"特征": label, "字段": field, "候选非空数": int(enriched[field].notna().sum()) if field in enriched else 0,
         "候选覆盖率(%)": enriched[field].notna().mean() * 100.0 if field in enriched and len(enriched) else 0.0}
        for label, field in V27_ENV_FEATURES.items()
    ])
    group_counts = enriched["结果组"].value_counts().reindex(GROUP_ORDER, fill_value=0)
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "评价开始": research, "信号截止": end,
        "观察截止": obs, "成熟候选": len(enriched), "候选周": enriched.Selection_Date.nunique(),
        "不同股票": enriched.ts_code.nunique(), "上升趋势": int(enriched.Weekly_Trend.eq("上升趋势").sum()),
        "中性趋势": int(enriched.Weekly_Trend.eq("中性趋势").sum()),
        "下降趋势": int(enriched.Weekly_Trend.eq("下降趋势").sum()),
        "亏损且MFE<30": int(group_counts[LOSS_GROUP]), "盈利且MFE<30": int(group_counts[LOW_PROFIT_GROUP]),
        "MFE30至50": int(group_counts[MID_GROUP]), "MFE50至100": int(group_counts[HIGH_GROUP]),
        "MFE翻倍": int(group_counts[DOUBLE_GROUP]), "环境股周截面": len(environment_detail),
        "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    reject_frame = pd.DataFrame([{"剔除原因": key, "次数": value} for key, value in rejects.items()])
    metadata = pd.DataFrame([
        ("研究问题", "能否用当时已知特征剔除-10%止损先到样本，并明示为此误删多少+30%目标先到样本"),
        ("三形态", "上升、中性、下降完全分开训练与验证"),
        ("正标签", "+30%目标早于-10%止损"),
        ("负标签", "-10%止损早于+30%目标；即使后来大涨也仍属于当前买点的负样本"),
        ("未决样本", "八周内均未触发，不进入标签学习，但进入保留池和空窗审计"),
        ("环境特征", "科技宽度；行业4/8/13周强度、MA20/60覆盖、红柱扩散、量价扩张；个股相对行业；板块风险"),
        ("误删代价前沿", "训练期允许误删+30%目标先到样本20%/30%/40%/50%；测试期如实报告实际代价"),
        ("未来隔离", "测试前只能使用测试开始日之前已走完8周观察窗的样本"),
        ("本版禁止", "不生成评分、不选Top3、不把OOS结果反向写入当期规则"),
    ], columns=["项目", "值"])
    files = {
        "01_run_summary_stop_first_veto_frontier_v2_7.csv": run_summary,
        "02_path_label_summary_stop_first_veto_frontier_v2_7.csv": path_summary,
        "03_chronological_fold_audit_stop_first_veto_frontier_v2_7.csv": results["folds"],
        "04_oos_cost_frontier_stop_first_veto_frontier_v2_7.csv": results["frontier"],
        "05_oos_frontier_aggregate_stop_first_veto_frontier_v2_7.csv": frontier_summary,
        "06_training_base_veto_rules_stop_first_veto_frontier_v2_7.csv": results["base_rules"],
        "07_training_all_candidate_rules_stop_first_veto_frontier_v2_7.csv": results["candidate_rules"],
        "08_oos_retained_event_detail_stop_first_veto_frontier_v2_7.csv": results["retained"],
        "09_oos_retained_concentration_stop_first_veto_frontier_v2_7.csv": concentration,
        "10_environment_feature_coverage_stop_first_veto_frontier_v2_7.csv": env_coverage,
        "11_feature_dictionary_stop_first_veto_frontier_v2_7.csv": v27_feature_dictionary(),
        "12_all_enriched_candidate_features_stop_first_veto_frontier_v2_7.csv": results["labeled"],
        "13_tech_weekly_environment_stop_first_veto_frontier_v2_7.csv": panels["tech"],
        "14_industry_weekly_environment_stop_first_veto_frontier_v2_7.csv": panels["industry"],
        "15_board_weekly_environment_stop_first_veto_frontier_v2_7.csv": panels["board"],
        "16_stock_weekly_environment_detail_stop_first_veto_frontier_v2_7.csv": panels["detail"],
        "17_full_tech_universe_stop_first_veto_frontier_v2_7.csv": universe_audit,
        "18_population_stop_first_veto_frontier_v2_7.csv": population,
        "19_rejection_audit_stop_first_veto_frontier_v2_7.csv": reject_frame,
        "20_metadata_stop_first_veto_frontier_v2_7.csv": metadata,
    }
    result_zip = make_result_zip(files)
    st.session_state[session_key] = result_zip
    st.success(f"完成：{len(enriched)}个成熟候选；三形态×三个时间折×四档误删预算已输出。")
    st.subheader("路径底数")
    st.dataframe(path_summary, use_container_width=True, hide_index=True)
    st.subheader("跨时间OOS误删代价前沿")
    st.dataframe(frontier_summary, use_container_width=True, hide_index=True)
    st.download_button("下载全部结果ZIP", result_zip,
                       file_name="weekly_macd_stop_first_veto_frontier_v2_7_all_results.zip",
                       mime="application/zip", type="primary", key="v27_download")
    st.warning("即使某档前沿有效，也只证明负向否决有研究价值；还不是可直接实盘使用的最终评分。")


# ===== V2.8 frozen monthly-MACD state audit (single file) =====
TITLE = "科技股周线MACD月线状态冻结审计器 V2.8"
VERSION = "V2.8-FROZEN-ASOF-MONTHLY-MACD-STATE-AUDIT"
V28_NEAR_ZERO_RATIO = 0.20
V28_STABLE_RATIO = 0.90
V28_MIN_MONTHS = 30

M_GREEN_NEW = "M1_刚由红转绿"
M_GREEN_EXPAND = "M2_绿柱扩大或稳定"
M_GREEN_SHRINK_FAR = "M3_绿柱缩短_远离零轴"
M_GREEN_SHRINK_NEAR = "M4_绿柱缩短_接近零轴"
M_RED_NEW = "M5_刚由绿转红"
M_RED_EXPAND_NEAR = "M6_红柱扩大或稳定_接近零轴"
M_RED_EXPAND_FAR = "M7_红柱扩大或稳定_远离零轴"
M_RED_SHRINK_NEAR = "M8_红柱缩短_接近零轴"
M_RED_SHRINK_FAR = "M9_红柱缩短_远离零轴"
V28_MONTHLY_STATES = (
    M_GREEN_NEW, M_GREEN_EXPAND, M_GREEN_SHRINK_FAR, M_GREEN_SHRINK_NEAR,
    M_RED_NEW, M_RED_EXPAND_NEAR, M_RED_EXPAND_FAR,
    M_RED_SHRINK_NEAR, M_RED_SHRINK_FAR,
)

V28_SCHEMES = {
    "S1_用户原始假设": {
        "states": {M_GREEN_SHRINK_NEAR, M_RED_EXPAND_FAR, M_RED_SHRINK_FAR},
        "meaning": "绿柱缩短且接近零轴，或红柱仍远离零轴（不限扩大/缩短）",
    },
    "S2_结构修正假设": {
        "states": {M_GREEN_SHRINK_NEAR, M_RED_NEW, M_RED_EXPAND_NEAR, M_RED_EXPAND_FAR},
        "meaning": "绿柱缩短接近零轴，刚由绿转红，或红柱仍在扩大/稳定",
    },
    "S3_保守趋势假设": {
        "states": {M_RED_EXPAND_FAR},
        "meaning": "只保留红柱远离零轴且扩大/稳定",
    },
}


def v28_monthly_snapshot(daily: pd.DataFrame, signal_date: str) -> dict[str, Any]:
    """用截至信号日的日线实时合成当月未完成月K；不读取信号日后价格。"""
    empty = {"Monthly_State": "无法计算", "Monthly_Audit_Reason": "无日线"}
    if daily.empty or "trade_date" not in daily or "close" not in daily:
        return empty
    cutoff = normalize_date(signal_date)
    work = daily.copy()
    work["trade_date"] = work["trade_date"].astype(str).map(normalize_date)
    work = work[work["trade_date"].le(cutoff)].copy()
    if work.empty:
        return empty
    work["dt"] = pd.to_datetime(work["trade_date"], format="%Y%m%d", errors="coerce")
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.dropna(subset=["dt", "close"]).sort_values("dt")
    work["month"] = work["dt"].dt.to_period("M")
    monthly = work.groupby("month", as_index=False).agg(
        trade_date=("trade_date", "last"), close=("close", "last"),
        month_trading_days_seen=("trade_date", "size"),
    )
    if len(monthly) < V28_MIN_MONTHS:
        return {"Monthly_State": "无法计算", "Monthly_Audit_Reason": f"月线预热不足<{V28_MIN_MONTHS}月",
                "Monthly_Months_Seen": len(monthly)}
    monthly["ema12"] = monthly["close"].ewm(span=12, adjust=False).mean()
    monthly["ema26"] = monthly["close"].ewm(span=26, adjust=False).mean()
    monthly["dif"] = monthly["ema12"] - monthly["ema26"]
    monthly["dea"] = monthly["dif"].ewm(span=9, adjust=False).mean()
    monthly["hist"] = (monthly["dif"] - monthly["dea"]) * 2.0
    current = monthly.iloc[-1]
    previous = monthly.iloc[-2]
    previous2 = monthly.iloc[-3]
    hist = finite_num(current["hist"])
    prev_hist = finite_num(previous["hist"])
    prev2_hist = finite_num(previous2["hist"])
    reference = pd.to_numeric(monthly["hist"].iloc[max(0, len(monthly) - 13):-1], errors="coerce").abs().max()
    near_ratio = abs(hist) / reference if math.isfinite(hist) and math.isfinite(reference) and reference > 1e-12 else np.nan
    near_zero = math.isfinite(near_ratio) and near_ratio <= V28_NEAR_ZERO_RATIO
    abs_ratio_prev = abs(hist) / abs(prev_hist) if math.isfinite(prev_hist) and abs(prev_hist) > 1e-12 else np.nan
    if hist < 0 and prev_hist >= 0:
        state = M_GREEN_NEW
    elif hist < 0:
        shrinking = math.isfinite(abs_ratio_prev) and abs_ratio_prev < V28_STABLE_RATIO
        state = (M_GREEN_SHRINK_NEAR if near_zero else M_GREEN_SHRINK_FAR) if shrinking else M_GREEN_EXPAND
    elif hist > 0 and prev_hist <= 0:
        state = M_RED_NEW
    elif hist > 0:
        expanding_or_stable = math.isfinite(abs_ratio_prev) and abs_ratio_prev >= V28_STABLE_RATIO
        if expanding_or_stable:
            state = M_RED_EXPAND_NEAR if near_zero else M_RED_EXPAND_FAR
        else:
            state = M_RED_SHRINK_NEAR if near_zero else M_RED_SHRINK_FAR
    else:
        state = "无法计算"
    dif = finite_num(current["dif"])
    dea = finite_num(current["dea"])
    if dif > 0 and dea > 0:
        line_zone = "DIF_DEA均在零轴上"
    elif dif < 0 and dea < 0:
        line_zone = "DIF_DEA均在零轴下"
    else:
        line_zone = "DIF_DEA分居零轴两侧"
    if hist > 0 and prev_hist > 0 and prev2_hist > 0:
        persistent = (abs(hist) < abs(prev_hist) * V28_STABLE_RATIO
                      and abs(prev_hist) < abs(prev2_hist) * V28_STABLE_RATIO)
    elif hist < 0 and prev_hist < 0 and prev2_hist < 0:
        persistent = (abs(hist) < abs(prev_hist) * V28_STABLE_RATIO
                      and abs(prev_hist) < abs(prev2_hist) * V28_STABLE_RATIO)
    else:
        persistent = False
    signal_dt = pd.to_datetime(cutoff, format="%Y%m%d")
    return {
        "Monthly_State": state, "Monthly_Audit_Reason": "完成",
        "Monthly_AsOf_Date": cutoff, "Monthly_Month": str(current["month"]),
        "Monthly_Months_Seen": len(monthly), "Monthly_Current_Close": finite_num(current["close"]),
        "Monthly_DIF": dif, "Monthly_DEA": dea, "Monthly_Hist": hist,
        "Monthly_Prev_Hist": prev_hist, "Monthly_Prev2_Hist": prev2_hist,
        "Monthly_Hist_Abs_vs_Prev": abs_ratio_prev,
        "Monthly_Zero_Distance_Ratio12": near_ratio,
        "Monthly_Near_Zero_Frozen20": near_zero,
        "Monthly_Line_Zone": line_zone, "Monthly_Two_Step_Shrink": persistent,
        "Monthly_Calendar_Progress_pct": signal_dt.day / signal_dt.days_in_month * 100.0,
        "Monthly_Trading_Days_Seen": int(current["month_trading_days_seen"]),
    }


def v28_add_monthly_states(candidates: pd.DataFrame,
                           histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, event in candidates.iterrows():
        code = str(event.get("ts_code", ""))
        snapshot = v28_monthly_snapshot(histories.get(code, pd.DataFrame()),
                                        normalize_date(event.get("Selection_Date")))
        rows.append(snapshot)
    monthly = pd.DataFrame(rows, index=candidates.index)
    out = candidates.copy()
    for column in monthly.columns:
        out[column] = monthly[column]
    return v27_path_labels(out)


def v28_period_masks(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    dates = pd.to_datetime(frame["Selection_Date"], format="%Y%m%d", errors="coerce")
    return [
        ("全部", pd.Series(True, index=frame.index)),
        ("2023下半年", dates.between("2023-06-05", "2023-12-31")),
        ("2024", dates.between("2024-01-01", "2024-12-31")),
        ("2025", dates.between("2025-01-01", "2025-12-31")),
        ("2026上半年", dates.between("2026-01-01", "2026-06-30")),
    ]


def v28_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    n = len(frame)
    targets = frame["V27_Target_First"].astype(bool)
    stops = frame["V27_Stop_First"].astype(bool)
    resolved = targets | stops
    status50 = frame["CP_W2_Delayed_First_50_vs_Stop"].fillna("").astype(str).eq("目标先到")
    status100 = frame["CP_W2_Delayed_First_100_vs_Stop"].fillna("").astype(str).eq("目标先到")
    return {
        "候选数": n, "不同股票": frame["ts_code"].nunique() if n else 0,
        "候选周": frame["Selection_Date"].nunique() if n else 0,
        "30%目标先到数": int(targets.sum()), "-10%止损先到数": int(stops.sum()),
        "八周均未触发数": int((~resolved).sum()),
        "30%目标先到率_全部(%)": targets.mean() * 100.0 if n else np.nan,
        "-10%止损先到率_全部(%)": stops.mean() * 100.0 if n else np.nan,
        "30%目标先到率_已决(%)": targets.sum() / resolved.sum() * 100.0 if resolved.sum() else np.nan,
        "50%目标先到数": int(status50.sum()), "100%目标先到数": int(status100.sum()),
        "MFE30至50数": int(frame["结果组"].eq(MID_GROUP).sum()),
        "MFE50至100数": int(frame["结果组"].eq(HIGH_GROUP).sum()),
        "MFE翻倍数": int(frame["结果组"].eq(DOUBLE_GROUP).sum()),
        "8周MFE均值(%)": num(frame.get("CP_W2_Delayed_MFE_8W_pct"), frame.index).mean(),
        "8周MFE中位数(%)": num(frame.get("CP_W2_Delayed_MFE_8W_pct"), frame.index).median(),
        "交易效用均值": num(frame.get("Realised_Utility"), frame.index).mean(),
        "交易效用中位数": num(frame.get("Realised_Utility"), frame.index).median(),
    }


def v28_state_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, period_mask in v28_period_masks(frame):
        period_data = frame[period_mask]
        for state in STATE_ORDER:
            state_data = period_data[period_data["Weekly_Trend"].eq(state)]
            baseline = v28_metrics(state_data)
            rows.append({"统计期": period, "周线形态": state,
                         "月线状态": "BASE_形态全部候选", "形态内占比(%)": 100.0,
                         **baseline})
            for monthly_state in V28_MONTHLY_STATES:
                selected = state_data[state_data["Monthly_State"].eq(monthly_state)]
                metrics = v28_metrics(selected)
                rows.append({"统计期": period, "周线形态": state,
                             "月线状态": monthly_state,
                             "形态内占比(%)": len(selected) / len(state_data) * 100.0 if len(state_data) else np.nan,
                             **metrics})
    return pd.DataFrame(rows)


def v28_scheme_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, period_mask in v28_period_masks(frame):
        period_data = frame[period_mask]
        for state in STATE_ORDER:
            baseline = period_data[period_data["Weekly_Trend"].eq(state)]
            base = v28_metrics(baseline)
            base_weeks = set(baseline["Selection_Date"].astype(str))
            for scheme_name, spec in V28_SCHEMES.items():
                selected = baseline[baseline["Monthly_State"].isin(spec["states"])]
                metrics = v28_metrics(selected)
                selected_weeks = set(selected["Selection_Date"].astype(str))
                row = {
                    "统计期": period, "周线形态": state, "冻结方案": scheme_name,
                    "方案含义": spec["meaning"],
                    "原始候选数": len(baseline), "保留比例(%)": len(selected) / len(baseline) * 100.0 if len(baseline) else np.nan,
                    "原始候选周": len(base_weeks), "保留候选周": len(selected_weeks),
                    "新增空窗周": len(base_weeks - selected_weeks),
                    "原始30%目标先到率_已决(%)": base["30%目标先到率_已决(%)"],
                    "原始止损先到率_全部(%)": base["-10%止损先到率_全部(%)"],
                    **{f"保留_{key}": value for key, value in metrics.items()},
                }
                row["30%目标先到率提升(百分点)"] = (
                    row["保留_30%目标先到率_已决(%)"]
                    - row["原始30%目标先到率_已决(%)"]
                )
                row["止损先到率下降(百分点)"] = (
                    row["原始止损先到率_全部(%)"]
                    - row["保留_-10%止损先到率_全部(%)"]
                )
                for outcome, short in ((MID_GROUP, "30至50"), (HIGH_GROUP, "50至100"), (DOUBLE_GROUP, "翻倍")):
                    total = int(baseline["结果组"].eq(outcome).sum())
                    kept = int(selected["结果组"].eq(outcome).sum())
                    row[f"{short}组保留数"] = kept
                    row[f"{short}组保留率(%)"] = kept / total * 100.0 if total else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def v28_acceptance(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    period_order = ["2024", "2025", "2026上半年"]
    for state in STATE_ORDER:
        for scheme in V28_SCHEMES:
            group = audit[(audit["周线形态"].eq(state)) & (audit["冻结方案"].eq(scheme))]
            overall = group[group["统计期"].eq("全部")]
            if overall.empty:
                continue
            overall = overall.iloc[0]
            period_group = group[group["统计期"].isin(period_order)].copy()
            period_group["改善"] = (
                num(period_group["30%目标先到率提升(百分点)"]).gt(0)
                & num(period_group["止损先到率下降(百分点)"]).gt(0)
                & num(period_group["保留_候选数"]).ge(5)
            )
            flags = []
            for period in period_order:
                item = period_group[period_group["统计期"].eq(period)]
                flags.append(bool(item.iloc[0]["改善"]) if not item.empty else False)
            max_streak = streak = 0
            for flag in flags:
                streak = streak + 1 if flag else 0
                max_streak = max(max_streak, streak)
            retain = finite_num(overall["保留比例(%)"])
            target_lift = finite_num(overall["30%目标先到率提升(百分点)"])
            stop_drop = finite_num(overall["止损先到率下降(百分点)"])
            pass_all = (retain >= 30.0 and target_lift >= 5.0 and stop_drop >= 10.0
                        and max_streak >= 2)
            if pass_all:
                verdict = "通过月线状态候选验收"
            elif retain < 30.0:
                verdict = "未通过：保留候选不足30%"
            elif target_lift < 5.0:
                verdict = "未通过：目标先到率提升不足5个百分点"
            elif stop_drop < 10.0:
                verdict = "未通过：止损率下降不足10个百分点"
            else:
                verdict = "未通过：没有连续两期同向改善"
            rows.append({
                "周线形态": state, "冻结方案": scheme,
                "全部保留比例(%)": retain, "全部目标先到率提升(百分点)": target_lift,
                "全部止损率下降(百分点)": stop_drop,
                "2024改善": flags[0], "2025改善": flags[1], "2026上半年改善": flags[2],
                "最长连续改善期": max_streak,
                "30至50组保留率(%)": overall["30至50组保留率(%)"],
                "50至100组保留率(%)": overall["50至100组保留率(%)"],
                "翻倍组保留率(%)": overall["翻倍组保留率(%)"],
                "新增空窗周": overall["新增空窗周"], "验收结论": verdict,
            })
    return pd.DataFrame(rows)


def v28_month_progress_audit(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    progress = num(out.get("Monthly_Calendar_Progress_pct"), out.index)
    out["月内信号阶段"] = pd.cut(progress, bins=[-np.inf, 33.333, 66.667, np.inf],
                                  labels=["月初三分之一", "月中三分之一", "月末三分之一"])
    rows = []
    for (weekly, monthly, progress_label), group in out.groupby(
            ["Weekly_Trend", "Monthly_State", "月内信号阶段"], observed=True):
        rows.append({"周线形态": weekly, "月线状态": monthly,
                     "月内信号阶段": progress_label, **v28_metrics(group)})
    return pd.DataFrame(rows)


def v28_state_dictionary() -> pd.DataFrame:
    descriptions = {
        M_GREEN_NEW: "当月柱为绿，上月为红；刚转弱",
        M_GREEN_EXPAND: "绿柱未缩短至上月的90%以下；下跌动能未明显收缩",
        M_GREEN_SHRINK_FAR: "绿柱绝对值<上月90%，但仍>12月参考峰值20%",
        M_GREEN_SHRINK_NEAR: "绿柱绝对值<上月90%，且≤12月参考峰值20%",
        M_RED_NEW: "当月柱为红，上月为绿；刚转强",
        M_RED_EXPAND_NEAR: "红柱≥上月90%，且≤12月参考峰值20%",
        M_RED_EXPAND_FAR: "红柱≥上月90%，且>12月参考峰值20%",
        M_RED_SHRINK_NEAR: "红柱<上月90%，且≤12月参考峰值20%",
        M_RED_SHRINK_FAR: "红柱<上月90%，但仍>12月参考峰值20%",
    }
    rows = [{"类型": "月线状态", "名称": state, "冻结定义": descriptions[state]} for state in V28_MONTHLY_STATES]
    rows.extend({"类型": "固定方案", "名称": name, "冻结定义": spec["meaning"]}
                for name, spec in V28_SCHEMES.items())
    rows.extend([
        {"类型": "固定阈值", "名称": "接近零轴", "冻结定义": "|当月MACD柱| ≤ 过去12个已知月最大|柱|的20%"},
        {"类型": "固定阈值", "名称": "扩大或稳定", "冻结定义": "|当月MACD柱| ≥ |上月柱|的90%"},
        {"类型": "信息时点", "名称": "实时月K", "冻结定义": "仅用第二根完整周红柱确认日及之前日线合成当月未完成月K"},
    ])
    return pd.DataFrame(rows)


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption("月线状态和阈值事先冻结；只审计、不训练、不搜索最优切分点、不生成评分。")
    with st.sidebar:
        st.header("正式评价区间")
        eval_date = st.date_input("评价开始", value=date(2023, 6, 5), key="v28_eval")
        end_date = st.date_input("信号截止", value=date(2026, 6, 5), key="v28_end")
        obs_date = st.date_input("行情观察截止", value=date.today(), max_value=date.today(), key="v28_obs")
        st.header("冻结月线定义")
        st.write("接近零轴：柱长≤过去12月峰值20%")
        st.write("扩大/稳定：当月柱长≥上月90%")
        st.write("当月使用信号日实时未完成月K")
        st.write("月线MACD至少预热30个月")
        st.write("三个固定方案同时判卷")
        st.write("不使用V2.7的负向否决规则")
        cache = st.checkbox("使用逐股票缓存", value=True, key="v28_cache")
        pause = st.number_input("每次API调用后暂停(秒)", 0.0, 3.0, 0.12, 0.05, key="v28_pause")
        if st.button("清除本程序缓存", key="v28_clear"):
            if os.path.isdir(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            st.success("缓存已清除")
    token = st.text_input("Tushare Token", type="password", key="v28_token")
    if not token:
        st.info("请输入Tushare Token。V2.7逐股票缓存可直接复用，不要清除缓存。")
        return
    session_key = "monthly_state_audit_v28_zip"
    if not st.button("开始V2.8月线状态冻结审计", type="primary", key="v28_run"):
        if session_key in st.session_state:
            st.download_button("下载上一次全部结果ZIP", st.session_state[session_key],
                               file_name="weekly_macd_monthly_state_audit_v2_8_all_results.zip",
                               mime="application/zip", key="v28_previous_download")
        return
    if eval_date >= end_date or end_date > obs_date:
        st.error("日期关系不正确。")
        return

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    eval_start = pd.Timestamp(eval_date)
    research, end, obs = eval_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), obs_date.strftime("%Y%m%d")
    # 与V2.7保持相同下载起点，可复用已有缓存；最早候选仍有约36个月MACD预热。
    preload = (eval_date - timedelta(days=3 * 365)).strftime("%Y%m%d")
    config = {
        "signal_start": research, "signal_end": end, "market_end": obs, "preload_start": preload,
        "min_price": 10.0, "min_mv": 100.0, "max_mv": 1_000_000_000.0,
        "price_tolerance_pct": 3.0, "stop_threshold_pct": 10.0,
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20, "sample_per_board": 0,
        "sample_seed": DEFAULT_SAMPLE_SEED, "long_cycle_min_weeks": DEFAULT_LONG_CYCLE_MIN_WEEKS,
        "material_hist_change_pct": DEFAULT_MATERIAL_HIST_CHANGE_PCT,
        "short_strength_ratio": DEFAULT_SHORT_STRENGTH_RATIO,
    }
    try:
        with st.spinner("加载交易日历、历史科技股池和板块指数..."):
            opens = load_trade_calendar(preload, obs)
            full = load_trade_calendar(preload, (obs_date + timedelta(days=7)).strftime("%Y%m%d"))
            basic = load_stock_basic()
            memberships = load_sw_tech_memberships(float(pause))
            week_map = complete_week_last_dates(full)
            boards: dict[str, pd.DataFrame] = {}
            for code in sorted(set(BOARD_INDEX.values())):
                board_daily = fetch_index_history(code, preload, obs, bool(cache), float(pause))
                if not board_daily.empty:
                    boards[code] = build_weekly(board_daily, week_map)
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return
    periods = build_period_index(memberships)
    codes = sorted(set(periods) & set(basic.ts_code.astype(str)))
    universe = basic[basic.ts_code.isin(codes)].copy()
    stocks, universe_audit, population = build_stratified_sample(universe, periods, end, 0, DEFAULT_SAMPLE_SEED)
    listed = stocks.list_date.apply(lambda x: normalize_date(x, "19000101"))
    delisted = stocks.delist_date.apply(lambda x: normalize_date(x, "99991231"))
    stocks = stocks[~listed.gt(end) & ~delisted.lt(preload)].reset_index(drop=True)
    open_pos = {trade_date: i for i, trade_date in enumerate(opens)}
    records: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}
    rejects: dict[str, int] = {}
    cache_hits = data_failures = 0
    progress = st.progress(0.0)
    status = st.empty()
    for i, stock in stocks.iterrows():
        code = str(stock.ts_code)
        progress.progress((i + 1) / len(stocks), text=f"{i + 1}/{len(stocks)} {code}")
        status.caption(f"事件{len(records)}；缓存{cache_hits}；失败{data_failures}")
        daily, daily_basic, hit = fetch_stock_history(code, preload, obs, bool(cache), float(pause))
        cache_hits += int(hit)
        if daily.empty:
            data_failures += 1
            continue
        stock_records, stock_rejects, _ = analyze_stock(
            stock, periods.get(code, []), daily, daily_basic, week_map, opens, open_pos, config
        )
        records.extend(stock_records)
        if stock_records:
            histories[code] = daily.copy()
        for reason, count in stock_rejects.items():
            rejects[reason] = rejects.get(reason, 0) + count
    progress.empty(); status.empty()
    if not records:
        st.error("没有生成事件。")
        return
    try:
        with st.spinner("按信号日实时合成月K，计算冻结月线状态并判卷..."):
            events = pd.DataFrame(records).sort_values(["Signal_Date", "ts_code", "Event_Type"])
            opportunities = build_cycle_opportunities(events, histories, obs, config["sell_slippage_pct"])
            featured = prepare_features(opportunities, boards)
            candidates = featured[
                featured.Strict_Eligible.map(bool_value) & featured.Outcome_Mature.map(bool_value)
                & featured.Selection_Date_dt.ge(eval_start)
                & featured.Selection_Date_dt.le(pd.Timestamp(end_date))
            ].copy()
            candidates = v25_prepare_candidates(candidates)
            audited = v28_add_monthly_states(candidates, histories)
            valid = audited[audited["Monthly_State"].isin(V28_MONTHLY_STATES)].copy()
            state_summary = v28_state_summary(valid)
            scheme_audit = v28_scheme_audit(valid)
            acceptance = v28_acceptance(scheme_audit)
            progress_audit = v28_month_progress_audit(valid)
    except Exception as exc:
        st.error(f"V2.8月线审计失败：{exc}")
        return

    monthly_counts = audited["Monthly_State"].value_counts()
    group_counts = valid["结果组"].value_counts().reindex(GROUP_ORDER, fill_value=0)
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "评价开始": research, "信号截止": end,
        "观察截止": obs, "严格成熟候选": len(candidates), "月线状态有效候选": len(valid),
        "月线无法计算": int(len(audited) - len(valid)), "候选周": valid.Selection_Date.nunique(),
        "不同股票": valid.ts_code.nunique(), "上升趋势": int(valid.Weekly_Trend.eq("上升趋势").sum()),
        "中性趋势": int(valid.Weekly_Trend.eq("中性趋势").sum()),
        "下降趋势": int(valid.Weekly_Trend.eq("下降趋势").sum()),
        "亏损且MFE<30": int(group_counts[LOSS_GROUP]), "盈利且MFE<30": int(group_counts[LOW_PROFIT_GROUP]),
        "MFE30至50": int(group_counts[MID_GROUP]), "MFE50至100": int(group_counts[HIGH_GROUP]),
        "MFE翻倍": int(group_counts[DOUBLE_GROUP]), "缓存命中": cache_hits, "行情失败": data_failures,
    }])
    monthly_count_table = pd.DataFrame([
        {"月线状态": state, "数量": int(monthly_counts.get(state, 0)),
         "占有效候选(%)": monthly_counts.get(state, 0) / len(valid) * 100.0 if len(valid) else np.nan}
        for state in V28_MONTHLY_STATES
    ])
    line_zone_summary = valid.groupby(["Weekly_Trend", "Monthly_State", "Monthly_Line_Zone"], dropna=False).size().reset_index(name="数量")
    reject_frame = pd.DataFrame([{"剔除原因": key, "次数": value} for key, value in rejects.items()])
    metadata = pd.DataFrame([
        ("研究定位", "月线状态最后假设的一次冻结审计；不搜索最优阈值"),
        ("买点", "第二根完整周线红柱严格扩张确认后，次日开盘"),
        ("月线时点", "仅用信号日及之前日线，实时合成未完成月K；不使用月底后来数据"),
        ("接近零轴", "|当月柱|≤过去12个当时已知月份最大|柱|的20%"),
        ("扩大或稳定", "|当月柱|≥|上月柱|的90%"),
        ("验收门槛", "保留≥30%；整体目标先到率+5个百分点；整体止损率-10个百分点；2024/2025/2026上半年至少连续两期同向改善"),
        ("信号截止", "Selection_Date严格不晚于用户设定信号截止日，已修正V2.7多纳6个次周候选的边界问题"),
        ("本版禁止", "不训练、不评分、不选Top3、不根据结果改动20%/90%阈值"),
    ], columns=["项目", "值"])
    files = {
        "01_run_summary_monthly_state_audit_v2_8.csv": run_summary,
        "02_monthly_state_count_monthly_state_audit_v2_8.csv": monthly_count_table,
        "03_state_period_quality_monthly_state_audit_v2_8.csv": state_summary,
        "04_frozen_scheme_period_audit_monthly_state_audit_v2_8.csv": scheme_audit,
        "05_frozen_scheme_acceptance_monthly_state_audit_v2_8.csv": acceptance,
        "06_month_progress_stability_monthly_state_audit_v2_8.csv": progress_audit,
        "07_monthly_dif_dea_zone_monthly_state_audit_v2_8.csv": line_zone_summary,
        "08_monthly_state_dictionary_monthly_state_audit_v2_8.csv": v28_state_dictionary(),
        "09_all_valid_candidate_monthly_detail_monthly_state_audit_v2_8.csv": valid,
        "10_all_candidate_including_monthly_failures_monthly_state_audit_v2_8.csv": audited,
        "11_full_tech_universe_monthly_state_audit_v2_8.csv": universe_audit,
        "12_population_monthly_state_audit_v2_8.csv": population,
        "13_rejection_audit_monthly_state_audit_v2_8.csv": reject_frame,
        "14_metadata_monthly_state_audit_v2_8.csv": metadata,
    }
    result_zip = make_result_zip(files)
    st.session_state[session_key] = result_zip
    st.success(f"完成：{len(valid)}个月线状态有效候选；9个冻结状态和3个固定方案已判卷。")
    st.subheader("三个固定方案验收")
    st.dataframe(acceptance, use_container_width=True, hide_index=True)
    st.subheader("月线状态数量")
    st.dataframe(monthly_count_table, use_container_width=True, hide_index=True)
    st.download_button("下载全部结果ZIP", result_zip,
                       file_name="weekly_macd_monthly_state_audit_v2_8_all_results.zip",
                       mime="application/zip", type="primary", key="v28_download")
    st.warning("如果三个事先冻结方案均未通过，应停止继续挖掘技术评分，不再事后改月线阈值。")


v28_main_legacy_entry = main  # 保留V2.8入口供代码审计；V2.9不调用。


# ===== V2.9 same-week RPS / VCP / LambdaMART ranking audit =====
TITLE = "科技股周线MACD同周RPS/VCP/LTR排序审计器 V2.9"
VERSION = "V2.9-SAME-WEEK-RPS-VCP-LAMBDAMART-OOS"
V29_RANDOM_REPEATS = 200
V29_TOP_K = 3
V29_MIN_TRAIN_ROWS = 300
V29_MIN_TRAIN_WEEKS = 45
V29_MODEL_FEATURES = (
    "V29_RPS20", "V29_RPS60", "V29_RPS120", "V29_RPS250",
    "V29_VCP_Tight10", "V29_VCP_Contract5_10", "V29_VCP_Contract10_20",
    "V29_VCP_DryVolume", "V29_VCP_NearHigh",
    "V29_MV_Rank", "V29_Turnover_Rank", "V29_ATR_Rank",
    "V29_CloseLocation_Rank", "V29_VolumeRatio_Rank", "V29_BoardRS_Rank",
    "V29_Trend_Up", "V29_Trend_Neutral", "V29_Trend_Down",
)


def v29_asof_daily_features(daily: pd.DataFrame, signal_date: str) -> dict[str, Any]:
    """只读取信号日及以前日线，生成RPS原始动量与VCP收缩特征。"""
    keys = [
        "V29_Return20_pct", "V29_Return60_pct", "V29_Return120_pct",
        "V29_Return250_pct", "V29_Range5_pct", "V29_Range10_pct",
        "V29_Range20_pct", "V29_Range5_to10", "V29_Range10_to20",
        "V29_Volume5_to20", "V29_ATR10_pct", "V29_ATR20_pct",
        "V29_Close_to_High20", "V29_Low5_to_Low20",
        "V29_Daily_Feature_Days", "V29_Daily_Feature_Reason",
    ]
    empty = {key: np.nan for key in keys}
    empty["V29_Daily_Feature_Reason"] = "无日线"
    if daily.empty or "trade_date" not in daily or "close" not in daily:
        return empty
    cutoff = normalize_date(signal_date)
    work = daily.copy()
    work["trade_date"] = work["trade_date"].astype(str).map(normalize_date)
    work = work[work["trade_date"].le(cutoff)].copy()
    for column in ("open", "high", "low", "close", "vol"):
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    work = work.dropna(subset=["trade_date", "high", "low", "close"]).sort_values("trade_date")
    if len(work) < 25:
        empty["V29_Daily_Feature_Days"] = len(work)
        empty["V29_Daily_Feature_Reason"] = "日线预热不足25日"
        return empty

    close = work["close"].to_numpy(float)
    high = work["high"].to_numpy(float)
    low = work["low"].to_numpy(float)
    volume = work["vol"].fillna(0.0).to_numpy(float)

    def trailing_return(days: int) -> float:
        if len(close) <= days or not math.isfinite(close[-days - 1]) or close[-days - 1] <= 0:
            return np.nan
        return (close[-1] / close[-days - 1] - 1.0) * 100.0

    def range_pct(days: int) -> float:
        if len(close) < days:
            return np.nan
        hi = float(np.nanmax(high[-days:]))
        lo = float(np.nanmin(low[-days:]))
        return (hi / lo - 1.0) * 100.0 if math.isfinite(hi) and math.isfinite(lo) and lo > 0 else np.nan

    previous = np.r_[close[0], close[:-1]]
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))

    def atr_pct(days: int) -> float:
        if len(close) < days or close[-1] <= 0:
            return np.nan
        return float(np.nanmean(true_range[-days:]) / close[-1] * 100.0)

    range5, range10, range20 = range_pct(5), range_pct(10), range_pct(20)
    vol5 = float(np.nanmean(volume[-5:])) if len(volume) >= 5 else np.nan
    vol20 = float(np.nanmean(volume[-20:])) if len(volume) >= 20 else np.nan
    high20 = float(np.nanmax(high[-20:]))
    low5 = float(np.nanmin(low[-5:]))
    low20 = float(np.nanmin(low[-20:]))
    return {
        "V29_Return20_pct": trailing_return(20),
        "V29_Return60_pct": trailing_return(60),
        "V29_Return120_pct": trailing_return(120),
        "V29_Return250_pct": trailing_return(250),
        "V29_Range5_pct": range5, "V29_Range10_pct": range10,
        "V29_Range20_pct": range20,
        "V29_Range5_to10": _safe_ratio(range5, range10),
        "V29_Range10_to20": _safe_ratio(range10, range20),
        "V29_Volume5_to20": _safe_ratio(vol5, vol20),
        "V29_ATR10_pct": atr_pct(10), "V29_ATR20_pct": atr_pct(20),
        "V29_Close_to_High20": _safe_ratio(close[-1], high20),
        "V29_Low5_to_Low20": _safe_ratio(low5, low20),
        "V29_Daily_Feature_Days": len(work), "V29_Daily_Feature_Reason": "完成",
    }


def v29_week_rank(frame: pd.DataFrame, values: Any, higher_is_better: bool = True) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    oriented = series if higher_is_better else -series
    ranked = oriented.groupby(frame["Selection_Date"]).rank(method="average", pct=True)
    return ranked.fillna(0.5).clip(0.0, 1.0)


def v29_relevance(frame: pd.DataFrame) -> pd.Series:
    win20 = target_first(frame, 20)
    win30 = target_first(frame, 30)
    win50 = target_first(frame, 50)
    win100 = target_first(frame, 100)
    positive = num(frame.get("Realised_Utility"), frame.index).gt(0)
    labels = np.select(
        [win100, win50, win30, win20, positive],
        [5, 4, 3, 2, 1], default=0,
    )
    return pd.Series(labels, index=frame.index, dtype=int)


def v29_enrich_candidates(candidates: pd.DataFrame,
                          histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    snapshots: list[dict[str, Any]] = []
    for _, event in candidates.iterrows():
        code = str(event.get("ts_code", ""))
        snapshots.append(v29_asof_daily_features(
            histories.get(code, pd.DataFrame()), normalize_date(event.get("Selection_Date"))
        ))
    extra = pd.DataFrame(snapshots, index=candidates.index)
    out = candidates.copy()
    for column in extra.columns:
        out[column] = extra[column]

    for days in (20, 60, 120, 250):
        out[f"V29_RPS{days}"] = v29_week_rank(out, out[f"V29_Return{days}_pct"], True)
    out["V29_RPS_Composite"] = (
        0.20 * out["V29_RPS60"] + 0.50 * out["V29_RPS120"] + 0.30 * out["V29_RPS250"]
    )
    out["V29_VCP_Tight10"] = v29_week_rank(out, out["V29_Range10_pct"], False)
    out["V29_VCP_Contract5_10"] = v29_week_rank(out, out["V29_Range5_to10"], False)
    out["V29_VCP_Contract10_20"] = v29_week_rank(out, out["V29_Range10_to20"], False)
    out["V29_VCP_DryVolume"] = v29_week_rank(out, out["V29_Volume5_to20"], False)
    out["V29_VCP_NearHigh"] = v29_week_rank(out, out["V29_Close_to_High20"], True)
    out["V29_VCP_Score"] = (
        0.25 * out["V29_VCP_Tight10"]
        + 0.15 * out["V29_VCP_Contract5_10"]
        + 0.20 * out["V29_VCP_Contract10_20"]
        + 0.20 * out["V29_VCP_DryVolume"]
        + 0.20 * out["V29_VCP_NearHigh"]
    )
    out["V29_RPS_VCP_Score"] = 0.50 * out["V29_RPS_Composite"] + 0.50 * out["V29_VCP_Score"]

    rank_specs = {
        "V29_MV_Rank": (out.get("Circ_MV_Billion"), False),
        "V29_Turnover_Rank": (out.get("Turnover_Rate"), True),
        "V29_ATR_Rank": (out.get("CP_W2_ATR14_pct"), True),
        "V29_CloseLocation_Rank": (out.get("CP_W2_Close_Location"), True),
        "V29_VolumeRatio_Rank": (out.get("CP_W2_Volume_Ratio20"), True),
        "V29_BoardRS_Rank": (out.get("Board_RS"), True),
    }
    for name, (values, direction) in rank_specs.items():
        if values is None:
            out[name] = 0.5
        else:
            out[name] = v29_week_rank(out, values, direction)
    out["V29_Trend_Up"] = out["Weekly_Trend"].eq("上升趋势").astype(int)
    out["V29_Trend_Neutral"] = out["Weekly_Trend"].eq("中性趋势").astype(int)
    out["V29_Trend_Down"] = out["Weekly_Trend"].eq("下降趋势").astype(int)
    out["V29_Relevance"] = v29_relevance(out)
    out = v27_path_labels(out)
    maturity = pd.to_datetime(out.get("Outcome_Maturity_Date_dt"), errors="coerce")
    fallback = pd.to_datetime(out["Selection_Date"], format="%Y%m%d", errors="coerce") + pd.Timedelta(days=56)
    out["V29_Maturity_Date"] = maturity.fillna(fallback)
    return out


def v29_hash_score(frame: pd.DataFrame, seed: int) -> pd.Series:
    def one(row: pd.Series) -> float:
        text_value = f"{seed}|{row.get('Selection_Date', '')}|{row.get('ts_code', '')}"
        digest = hashlib.sha256(text_value.encode("utf-8")).hexdigest()[:15]
        return int(digest, 16) / float(16 ** 15)
    return frame.apply(one, axis=1)


def v29_select_top3(frame: pd.DataFrame, score: pd.Series,
                    method: str, fold_name: str) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["V29_Method"] = method
    ranked["V29_Fold"] = fold_name
    ranked["V29_Score"] = pd.to_numeric(score, errors="coerce").fillna(-np.inf)
    ranked = ranked.sort_values(
        ["Selection_Date", "V29_Score", "ts_code"], ascending=[True, False, True]
    )
    ranked["V29_Weekly_Rank"] = ranked.groupby("Selection_Date").cumcount() + 1
    return ranked[ranked["V29_Weekly_Rank"].le(V29_TOP_K)].copy()


def v29_ndcg_at3(frame: pd.DataFrame, score_column: str = "V29_Score") -> float:
    if score_column not in frame.columns or "V29_Relevance" not in frame.columns:
        return np.nan
    values = []
    for _, group in frame.groupby("Selection_Date"):
        if group.empty:
            continue
        ordered = group.sort_values(score_column, ascending=False).head(V29_TOP_K)
        ideal = group.sort_values("V29_Relevance", ascending=False).head(V29_TOP_K)
        discounts = 1.0 / np.log2(np.arange(2, len(ordered) + 2))
        ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
        dcg = float(np.sum((np.power(2.0, ordered["V29_Relevance"].to_numpy(float)) - 1.0) * discounts))
        idcg = float(np.sum((np.power(2.0, ideal["V29_Relevance"].to_numpy(float)) - 1.0) * ideal_discounts))
        values.append(dcg / idcg if idcg > 0 else 1.0)
    return float(np.mean(values)) if values else np.nan


def v29_top3_capture_ndcg(selected: pd.DataFrame, pool: pd.DataFrame) -> float:
    """入选Top3相对于当周事后理想Top3的收益等级捕获率；仅用于判卷。"""
    if "V29_Weekly_Rank" not in selected.columns:
        return np.nan
    values: list[float] = []
    for week, group in selected.groupby("Selection_Date"):
        whole = pool[pool["Selection_Date"].eq(week)]
        if whole.empty:
            continue
        ordered = group.sort_values("V29_Weekly_Rank").head(V29_TOP_K)
        ideal = whole.sort_values("V29_Relevance", ascending=False).head(V29_TOP_K)
        discounts = 1.0 / np.log2(np.arange(2, len(ordered) + 2))
        ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
        dcg = float(np.sum((np.power(2.0, ordered["V29_Relevance"].to_numpy(float)) - 1.0) * discounts))
        idcg = float(np.sum((np.power(2.0, ideal["V29_Relevance"].to_numpy(float)) - 1.0) * ideal_discounts))
        values.append(dcg / idcg if idcg > 0 else 1.0)
    return float(np.mean(values)) if values else np.nan


def v29_selection_metrics(selected: pd.DataFrame, pool: pd.DataFrame,
                           method: str, fold_name: str) -> dict[str, Any]:
    n = len(selected)
    target30 = target_first(selected, 30)
    status30 = selected["CP_W2_Delayed_First_30_vs_Stop"].fillna("").astype(str)
    stop30 = status30.isin(["止损先到", "同日不确定_按止损"])
    resolved = target30 | stop30
    target50 = target_first(selected, 50)
    target100 = target_first(selected, 100)
    pool50 = int(target_first(pool, 50).sum())
    pool100 = int(target_first(pool, 100).sum())
    return {
        "测试期": fold_name, "方法": method,
        "测试候选": len(pool), "测试候选周": pool["Selection_Date"].nunique(),
        "入选事件": n, "入选周": selected["Selection_Date"].nunique() if n else 0,
        "周均入选": n / selected["Selection_Date"].nunique() if n and selected["Selection_Date"].nunique() else np.nan,
        "30%目标先到数": int(target30.sum()), "-10%止损先到数": int(stop30.sum()),
        "30%目标先到率_全部(%)": target30.mean() * 100.0 if n else np.nan,
        "-10%止损先到率_全部(%)": stop30.mean() * 100.0 if n else np.nan,
        "30%目标先到率_已决(%)": target30.sum() / resolved.sum() * 100.0 if resolved.sum() else np.nan,
        "50%目标先到数": int(target50.sum()), "100%目标先到数": int(target100.sum()),
        "50%牛股池捕获率(%)": target50.sum() / pool50 * 100.0 if pool50 else np.nan,
        "100%牛股池捕获率(%)": target100.sum() / pool100 * 100.0 if pool100 else np.nan,
        "标签均值": num(selected.get("V29_Relevance"), selected.index).mean(),
        "交易效用均值": num(selected.get("Realised_Utility"), selected.index).mean(),
        "交易效用中位数": num(selected.get("Realised_Utility"), selected.index).median(),
        "8周收益均值(%)": num(selected.get("CP_W2_Delayed_Return_8W_pct"), selected.index).mean(),
        "8周收益中位数(%)": num(selected.get("CP_W2_Delayed_Return_8W_pct"), selected.index).median(),
        "8周MFE均值(%)": num(selected.get("CP_W2_Delayed_MFE_8W_pct"), selected.index).mean(),
        "8周MAE均值(%)": num(selected.get("CP_W2_Delayed_MAE_8W_pct"), selected.index).mean(),
        "NDCG@3": v29_top3_capture_ndcg(selected, pool),
    }


def v29_import_xgboost():
    try:
        import xgboost as xgb  # type: ignore
        return xgb, "可用"
    except Exception as exc:
        return None, f"不可用：{exc}"


def v29_fit_ltr(train: pd.DataFrame):
    xgb, status = v29_import_xgboost()
    if xgb is None:
        return None, status, pd.DataFrame()
    work = train.sort_values(["Selection_Date", "ts_code"]).copy()
    varying = work.groupby("Selection_Date")["V29_Relevance"].nunique()
    useful_weeks = set(varying[varying.gt(1)].index.astype(str))
    work = work[work["Selection_Date"].astype(str).isin(useful_weeks)].copy()
    if len(work) < V29_MIN_TRAIN_ROWS or work["Selection_Date"].nunique() < V29_MIN_TRAIN_WEEKS:
        return None, "训练样本或有效周不足", pd.DataFrame()
    x = work.loc[:, V29_MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.5)
    y = work["V29_Relevance"].astype(int)
    qid = pd.factorize(work["Selection_Date"], sort=True)[0]
    model = xgb.XGBRanker(
        objective="rank:ndcg", eval_metric="ndcg@3",
        n_estimators=80, learning_rate=0.035, max_depth=2,
        min_child_weight=12.0, subsample=0.80, colsample_bytree=0.80,
        reg_alpha=1.0, reg_lambda=12.0, random_state=20260812,
        tree_method="hist", n_jobs=4,
    )
    model.fit(x, y, qid=qid, verbose=False)
    importance = pd.DataFrame({
        "字段": list(V29_MODEL_FEATURES),
        "模型重要性": getattr(model, "feature_importances_", np.zeros(len(V29_MODEL_FEATURES))),
    }).sort_values("模型重要性", ascending=False)
    return model, f"完成：{len(work)}行/{work['Selection_Date'].nunique()}个有效周", importance


def v29_predict_ltr(model: Any, frame: pd.DataFrame) -> pd.Series:
    x = frame.loc[:, V29_MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.5)
    return pd.Series(model.predict(x), index=frame.index, dtype=float)


def v29_random_mc(pool: pd.DataFrame, fold_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    first_detail = pd.DataFrame()
    for seed in range(V29_RANDOM_REPEATS):
        selected = v29_select_top3(pool, v29_hash_score(pool, seed), f"随机三只_种子{seed}", fold_name)
        if seed == 0:
            first_detail = selected.copy()
            first_detail["V29_Method"] = "随机三只_固定种子"
        metrics = v29_selection_metrics(selected, pool, "随机三只", fold_name)
        metrics["随机种子"] = seed
        rows.append(metrics)
    distribution = pd.DataFrame(rows)
    mean_row = {"测试期": fold_name, "方法": "随机三只_MC均值", "随机重复": V29_RANDOM_REPEATS}
    for column in distribution.columns:
        if column in ("测试期", "方法", "随机种子"):
            continue
        values = pd.to_numeric(distribution[column], errors="coerce")
        mean_row[column] = values.mean()
        mean_row[f"{column}_标准差"] = values.std(ddof=1)
    return pd.DataFrame([mean_row]), distribution


def v29_run_oos(candidates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = candidates.copy()
    data["Selection_Date_dt"] = pd.to_datetime(data["Selection_Date"], format="%Y%m%d", errors="coerce")
    folds = v27_time_folds(data)
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selected_parts: list[pd.DataFrame] = []
    random_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []

    score_methods = {
        "RPS120": "V29_RPS120",
        "RPS复合": "V29_RPS_Composite",
        "VCP紧凑度": "V29_VCP_Score",
        "RPS+VCP等权": "V29_RPS_VCP_Score",
    }
    for fold in folds:
        name, start, end = fold["测试期"], fold["测试开始"], fold["测试结束"]
        test = data[data["Selection_Date_dt"].between(start, end)].copy()
        train = data[
            data["Selection_Date_dt"].lt(start)
            & data["V29_Maturity_Date"].lt(start)
        ].copy()
        model, model_status, importance = v29_fit_ltr(train)
        fold_rows.append({
            "测试期": name, "测试开始": start.strftime("%Y%m%d"),
            "测试结束": end.strftime("%Y%m%d"), "训练候选": len(train),
            "训练周": train["Selection_Date"].nunique(), "测试候选": len(test),
            "测试周": test["Selection_Date"].nunique(), "LTR状态": model_status,
            "训练最新成熟日": train["V29_Maturity_Date"].max() if len(train) else pd.NaT,
        })
        if test.empty:
            continue
        pool_row = v29_selection_metrics(test, test, "全部候选池", name)
        summary_rows.append(pool_row)
        mc_mean, mc_distribution = v29_random_mc(test, name)
        summary_rows.extend(mc_mean.to_dict("records"))
        random_parts.append(mc_distribution)
        random_seed_detail = v29_select_top3(test, v29_hash_score(test, 0), "随机三只_固定种子", name)
        selected_parts.append(random_seed_detail)
        for method, column in score_methods.items():
            chosen = v29_select_top3(test, test[column], method, name)
            summary_rows.append(v29_selection_metrics(chosen, test, method, name))
            selected_parts.append(chosen)
        if model is not None:
            prediction = v29_predict_ltr(model, test)
            chosen = v29_select_top3(test, prediction, "LTR_LambdaMART", name)
            summary_rows.append(v29_selection_metrics(chosen, test, "LTR_LambdaMART", name))
            selected_parts.append(chosen)
            importance = importance.copy()
            importance.insert(0, "测试期", name)
            importance_parts.append(importance)

    summary = pd.DataFrame(summary_rows)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    random_distribution = pd.concat(random_parts, ignore_index=True) if random_parts else pd.DataFrame()
    importance = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    aggregate_rows: list[dict[str, Any]] = []
    if not selected.empty:
        test_pool = data[data["Selection_Date_dt"].between(
            min(f["测试开始"] for f in folds), max(f["测试结束"] for f in folds)
        )].copy() if folds else data.iloc[0:0].copy()
        for method, group in selected.groupby("V29_Method"):
            aggregate_rows.append(v29_selection_metrics(group, test_pool, method, "全部OOS合并"))
    return {
        "summary": summary, "aggregate": pd.DataFrame(aggregate_rows),
        "folds": pd.DataFrame(fold_rows), "selected": selected,
        "random": random_distribution, "importance": importance,
    }


def v29_acceptance(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = results["summary"]
    rows: list[dict[str, Any]] = []
    folds = [name for name in summary.get("测试期", pd.Series(dtype=str)).dropna().unique()]
    improved = comparable = 0
    for fold in folds:
        group = summary[summary["测试期"].eq(fold)]
        ltr = group[group["方法"].eq("LTR_LambdaMART")]
        transparent = group[group["方法"].isin(["RPS120", "RPS复合", "VCP紧凑度", "RPS+VCP等权"])]
        random_row = group[group["方法"].eq("随机三只_MC均值")]
        if ltr.empty or transparent.empty or random_row.empty:
            rows.append({"测试期": fold, "可比较": False, "结论": "LTR未完成或基准缺失"})
            continue
        comparable += 1
        ltr_row = ltr.iloc[0]
        best = transparent.sort_values(["交易效用均值", "标签均值"], ascending=False).iloc[0]
        random_item = random_row.iloc[0]
        passes = (
            finite_num(ltr_row["交易效用均值"]) > finite_num(best["交易效用均值"])
            and finite_num(ltr_row["标签均值"]) > finite_num(best["标签均值"])
            and finite_num(ltr_row["30%目标先到率_全部(%)"]) > finite_num(random_item["30%目标先到率_全部(%)"])
            and finite_num(ltr_row["-10%止损先到率_全部(%)"]) <= finite_num(random_item["-10%止损先到率_全部(%)"])
        )
        improved += int(passes)
        rows.append({
            "测试期": fold, "可比较": True, "最佳透明基准": best["方法"],
            "LTR交易效用": ltr_row["交易效用均值"], "透明基准交易效用": best["交易效用均值"],
            "LTR标签均值": ltr_row["标签均值"], "透明基准标签均值": best["标签均值"],
            "LTR目标先到率(%)": ltr_row["30%目标先到率_全部(%)"],
            "随机目标先到率(%)": random_item["30%目标先到率_全部(%)"],
            "LTR止损先到率(%)": ltr_row["-10%止损先到率_全部(%)"],
            "随机止损先到率(%)": random_item["-10%止损先到率_全部(%)"],
            "本期通过": passes, "结论": "本期优于透明基准和随机" if passes else "本期未形成净优势",
        })
    overall_pass = comparable >= 3 and improved >= 2
    rows.append({
        "测试期": "总验收", "可比较": comparable >= 3,
        "可比较测试期": comparable, "通过测试期": improved,
        "本期通过": overall_pass,
        "结论": "排序突破候选：至少两期形成净优势" if overall_pass else "未通过：没有稳定战胜透明基准和随机三只",
    })
    return pd.DataFrame(rows)


def v29_feature_dictionary() -> pd.DataFrame:
    rows = [
        ("RPS20/60/120/250", "各周期截至信号日收益在同周候选中的百分位；无未来数据"),
        ("RPS复合", "20%×RPS60 + 50%×RPS120 + 30%×RPS250；权重冻结不寻优"),
        ("VCP紧凑度", "10日箱体、5/10与10/20日收缩、5/20量能萎缩、接近20日高点的冻结组合"),
        ("RPS+VCP", "RPS复合与VCP各50%；只作透明组合基准"),
        ("LTR标签0", "未实现20%目标且交易效用不为正；包含止损先到"),
        ("LTR标签1", "未到20%，但既有交易效用为正"),
        ("LTR标签2/3/4/5", "依次为20%/30%/50%/100%目标先于-10%止损"),
        ("LTR分组", "qid=Selection_Date；模型只学习同一信号周候选间的顺序"),
        ("未来隔离", "训练样本的八周观察成熟日必须早于测试期开始日"),
    ]
    return pd.DataFrame(rows, columns=["项目", "冻结定义"])


def main() -> None:
    global pro, API_ERRORS
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.caption("红柱只建立候选池；本版严格比较随机、RPS、VCP、RPS+VCP与同周LambdaMART，不改候选硬条件。")
    with st.sidebar:
        st.header("正式评价区间")
        eval_date = st.date_input("评价开始", value=date(2023, 6, 5), key="v29_eval")
        end_date = st.date_input("信号截止", value=date(2026, 6, 5), key="v29_end")
        obs_date = st.date_input("行情观察截止", value=date.today(), max_value=date.today(), key="v29_obs")
        st.header("冻结排序口径")
        st.write("每个信号周为一个排序组；每周选前三名")
        st.write("RPS：60/120/250日冻结复合")
        st.write("VCP：价格收缩、量能萎缩、接近突破位")
        st.write("LTR：80棵深度2浅树；不搜索参数")
        st.write("测试期：2025上半年、2025下半年、2026上半年")
        st.write("训练只使用测试前已经走完8周路径的样本")
        cache = st.checkbox("使用逐股票缓存", value=True, key="v29_cache")
        pause = st.number_input("每次API调用后暂停(秒)", 0.0, 3.0, 0.12, 0.05, key="v29_pause")
        if st.button("清除本程序缓存", key="v29_clear"):
            if os.path.isdir(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            st.success("缓存已清除")
    xgb_module, xgb_status = v29_import_xgboost()
    if xgb_module is None:
        st.warning("未检测到xgboost。RPS/VCP仍可运行，但LTR会跳过。请先执行：pip install xgboost")
    else:
        st.caption(f"XGBoost状态：{xgb_status}；版本 {getattr(xgb_module, '__version__', '')}")
    token = st.text_input("Tushare Token", type="password", key="v29_token")
    if not token:
        st.info("请输入Tushare Token。V2.8逐股票缓存可直接复用，不要清除缓存。")
        return
    session_key = "same_week_rank_audit_v29_zip"
    if not st.button("开始V2.9同周排序实验", type="primary", key="v29_run"):
        if session_key in st.session_state:
            st.download_button("下载上一次全部结果ZIP", st.session_state[session_key],
                               file_name="weekly_macd_same_week_rank_audit_v2_9_all_results.zip",
                               mime="application/zip", key="v29_previous_download")
        return
    if eval_date >= end_date or end_date > obs_date:
        st.error("日期关系不正确。")
        return

    API_ERRORS = []
    ts.set_token(token)
    pro = ts.pro_api()
    eval_start = pd.Timestamp(eval_date)
    research, end, obs = eval_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), obs_date.strftime("%Y%m%d")
    preload = (eval_date - timedelta(days=3 * 365)).strftime("%Y%m%d")
    config = {
        "signal_start": research, "signal_end": end, "market_end": obs, "preload_start": preload,
        "min_price": 10.0, "min_mv": 100.0, "max_mv": 1_000_000_000.0,
        "price_tolerance_pct": 3.0, "stop_threshold_pct": 10.0,
        "buy_slippage_pct": 0.20, "sell_slippage_pct": 0.20, "sample_per_board": 0,
        "sample_seed": DEFAULT_SAMPLE_SEED, "long_cycle_min_weeks": DEFAULT_LONG_CYCLE_MIN_WEEKS,
        "material_hist_change_pct": DEFAULT_MATERIAL_HIST_CHANGE_PCT,
        "short_strength_ratio": DEFAULT_SHORT_STRENGTH_RATIO,
    }
    try:
        with st.spinner("加载交易日历、历史科技股池和板块指数..."):
            opens = load_trade_calendar(preload, obs)
            full = load_trade_calendar(preload, (obs_date + timedelta(days=7)).strftime("%Y%m%d"))
            basic = load_stock_basic()
            memberships = load_sw_tech_memberships(float(pause))
            week_map = complete_week_last_dates(full)
            boards: dict[str, pd.DataFrame] = {}
            for code in sorted(set(BOARD_INDEX.values())):
                board_daily = fetch_index_history(code, preload, obs, bool(cache), float(pause))
                if not board_daily.empty:
                    boards[code] = build_weekly(board_daily, week_map)
    except Exception as exc:
        st.error(f"基础数据加载失败：{exc}")
        return

    periods = build_period_index(memberships)
    codes = sorted(set(periods) & set(basic.ts_code.astype(str)))
    universe = basic[basic.ts_code.isin(codes)].copy()
    stocks, universe_audit, population = build_stratified_sample(universe, periods, end, 0, DEFAULT_SAMPLE_SEED)
    listed = stocks.list_date.apply(lambda x: normalize_date(x, "19000101"))
    delisted = stocks.delist_date.apply(lambda x: normalize_date(x, "99991231"))
    stocks = stocks[~listed.gt(end) & ~delisted.lt(preload)].reset_index(drop=True)
    open_pos = {trade_date: i for i, trade_date in enumerate(opens)}
    records: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}
    rejects: dict[str, int] = {}
    cache_hits = data_failures = 0
    progress = st.progress(0.0)
    status = st.empty()
    for i, stock in stocks.iterrows():
        code = str(stock.ts_code)
        progress.progress((i + 1) / len(stocks), text=f"{i + 1}/{len(stocks)} {code}")
        status.caption(f"事件{len(records)}；缓存{cache_hits}；失败{data_failures}")
        daily, daily_basic, hit = fetch_stock_history(code, preload, obs, bool(cache), float(pause))
        cache_hits += int(hit)
        if daily.empty:
            data_failures += 1
            continue
        stock_records, stock_rejects, _ = analyze_stock(
            stock, periods.get(code, []), daily, daily_basic, week_map, opens, open_pos, config
        )
        records.extend(stock_records)
        if stock_records:
            histories[code] = daily.copy()
        for reason, count in stock_rejects.items():
            rejects[reason] = rejects.get(reason, 0) + count
    progress.empty(); status.empty()
    if not records:
        st.error("没有生成事件。")
        return
    try:
        with st.spinner("生成截至信号日的RPS/VCP特征，并运行三段同周样本外排序..."):
            events = pd.DataFrame(records).sort_values(["Signal_Date", "ts_code", "Event_Type"])
            opportunities = build_cycle_opportunities(events, histories, obs, config["sell_slippage_pct"])
            featured = prepare_features(opportunities, boards)
            candidates = featured[
                featured.Strict_Eligible.map(bool_value) & featured.Outcome_Mature.map(bool_value)
                & featured.Selection_Date_dt.ge(eval_start)
                & featured.Selection_Date_dt.le(pd.Timestamp(end_date))
            ].copy()
            candidates = v25_prepare_candidates(candidates)
            enriched = v29_enrich_candidates(candidates, histories)
            results = v29_run_oos(enriched)
            acceptance = v29_acceptance(results)
    except Exception as exc:
        st.error(f"V2.9同周排序实验失败：{exc}")
        return

    reject_frame = pd.DataFrame([{"剔除原因": key, "次数": value} for key, value in rejects.items()])
    run_summary = pd.DataFrame([{
        "程序": TITLE, "版本": VERSION, "评价开始": research, "信号截止": end,
        "观察截止": obs, "成熟候选": len(enriched), "候选周": enriched.Selection_Date.nunique(),
        "不同股票": enriched.ts_code.nunique(), "RPS120有效": int(enriched.V29_Return120_pct.notna().sum()),
        "RPS250有效": int(enriched.V29_Return250_pct.notna().sum()),
        "VCP有效": int(enriched.V29_Range20_pct.notna().sum()),
        "XGBoost": xgb_status, "行情失败": data_failures, "缓存命中": cache_hits,
    }])
    metadata = pd.DataFrame([
        ("研究定位", "红柱池不变；只验证同周前三名排序能否突破"),
        ("透明基准", "随机三只200次、RPS120、RPS复合、VCP、RPS+VCP等权"),
        ("RPS口径", "同周候选百分位；用于前三名排序时与放入更大市场池的单周期顺序一致"),
        ("VCP口径", "只量化截至信号日已经形成的日线收缩；不声称低波动必然上涨"),
        ("LTR口径", "XGBoost rank:ndcg；qid=信号周；80棵、深度2；参数冻结不寻优"),
        ("标签", "0/1/2/3/4/5分别对应弱或止损、低盈利、20/30/50/100%目标先到"),
        ("未来隔离", "T1-T5及未来八周不进入特征；训练结果必须在测试开始前走完八周观察窗"),
        ("验收", "LTR至少三个测试期可比，并至少两期同时优于最佳透明基准与随机三只"),
        ("本版禁止", "不搜索RPS/VCP权重、不调树深、不根据OOS结果回写参数、不改变红柱候选池"),
    ], columns=["项目", "值"])
    files = {
        "01_run_summary_same_week_rank_v2_9.csv": run_summary,
        "02_oos_method_summary_same_week_rank_v2_9.csv": results["summary"],
        "03_oos_aggregate_same_week_rank_v2_9.csv": results["aggregate"],
        "04_acceptance_same_week_rank_v2_9.csv": acceptance,
        "05_fold_audit_same_week_rank_v2_9.csv": results["folds"],
        "06_selected_top3_detail_same_week_rank_v2_9.csv": results["selected"],
        "07_ltr_feature_importance_same_week_rank_v2_9.csv": results["importance"],
        "08_random_mc_distribution_same_week_rank_v2_9.csv": results["random"],
        "09_feature_dictionary_same_week_rank_v2_9.csv": v29_feature_dictionary(),
        "10_all_candidate_features_same_week_rank_v2_9.csv": enriched,
        "11_full_tech_universe_same_week_rank_v2_9.csv": universe_audit,
        "12_population_same_week_rank_v2_9.csv": population,
        "13_rejection_audit_same_week_rank_v2_9.csv": reject_frame,
        "14_metadata_same_week_rank_v2_9.csv": metadata,
    }
    result_zip = make_result_zip(files)
    st.session_state[session_key] = result_zip
    st.success(f"完成：{len(enriched)}个成熟候选；{enriched.Selection_Date.nunique()}个候选周已完成同周排序判卷。")
    st.subheader("最终验收")
    st.dataframe(acceptance, use_container_width=True, hide_index=True)
    st.subheader("各测试期方法比较")
    st.dataframe(results["summary"], use_container_width=True, hide_index=True)
    st.download_button("下载全部结果ZIP", result_zip,
                       file_name="weekly_macd_same_week_rank_audit_v2_9_all_results.zip",
                       mime="application/zip", type="primary", key="v29_download")
    st.warning("只有LTR在连续未来阶段稳定优于RPS、VCP和随机三只，才说明评分排序取得突破。")


if __name__ == "__main__":
    main()
