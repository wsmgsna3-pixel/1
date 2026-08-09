from __future__ import annotations

import hashlib
import io
import math
import os
import pickle
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts


TITLE = "周线MACD强弱反弹四因子验证器 V4.4"
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "weekly_factor_cache_v44"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BOARD_INDEX = {"主板": "000905.SH", "创业板": "399006.SZ", "科创板": "000688.SH"}
BROAD_INDEX = "000300.SH"
INDEX_NAME = {
    "000300.SH": "沪深300", "000905.SH": "中证500",
    "399006.SZ": "创业板指", "000688.SH": "科创50",
}

REQUIRED = {
    "Event_Type", "Cycle_ID", "ts_code", "name", "Sample_Board", "Signal_Date",
    "Tradable", "Has_8W_Future", "Cycle_Completed", "Cycle_Type", "Red_Cycle_Weeks",
    "Return_8W_pct", "Hit_Stop_8W", "First_10_vs_Stop", "First_20_vs_Stop",
    "First_30_vs_Stop",
}


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "是"}


def num(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def date8(value: Any) -> str:
    if pd.isna(value):
        return ""
    digits = "".join(ch for ch in str(value).split(".")[0] if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def read_csv(raw: bytes) -> pd.DataFrame:
    error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False)
        except Exception as exc:
            error = exc
    raise ValueError(f"CSV无法读取：{error}")


def load_events(uploaded: Any) -> tuple[pd.DataFrame, str]:
    raw = uploaded.getvalue()
    if uploaded.name.lower().endswith(".csv"):
        return read_csv(raw), uploaded.name
    if not uploaded.name.lower().endswith(".zip"):
        raise ValueError("请上传V4.1全部结果ZIP或01_events.csv。")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        candidates = [n for n in names if Path(n).name.lower() == "01_events.csv"]
        if not candidates:
            candidates = [n for n in names if "events" in Path(n).name.lower() and n.endswith(".csv")]
        if not candidates:
            raise ValueError("ZIP中没有找到V4.1的01_events.csv。")
        target = sorted(candidates, key=len)[0]
        return read_csv(archive.read(target)), f"{uploaded.name}/{target}"


def validate(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError("缺少V4.1字段：" + "、".join(missing))


def prepare_events(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw[raw["Event_Type"].astype(str).eq("第一根红柱")].copy()
    frame = frame[frame["Tradable"].map(to_bool) & frame["Has_8W_Future"].map(to_bool)]
    frame = frame.drop_duplicates("Cycle_ID").reset_index(drop=True)
    frame["Signal_Date"] = frame["Signal_Date"].map(date8)
    frame["Signal_Year"] = pd.to_numeric(frame["Signal_Date"].str[:4], errors="coerce").astype("Int64")
    red_weeks = pd.to_numeric(frame["Red_Cycle_Weeks"], errors="coerce")
    completed = frame["Cycle_Completed"].map(to_bool)
    frame["Strong_Sustained"] = red_weeks.ge(9) & frame["First_30_vs_Stop"].astype(str).eq("目标先到")
    frame["Weak_Red_3W"] = completed & red_weeks.le(3)
    frame["Stop_Before_10"] = ~frame["First_10_vs_Stop"].astype(str).eq("目标先到")
    frame["Short_Profitable_Rebound"] = (
        ~frame["Strong_Sustained"]
        & red_weeks.lt(9)
        & frame["First_20_vs_Stop"].astype(str).eq("目标先到")
    )
    frame["Weak_Rebound"] = (
        ~frame["Strong_Sustained"]
        & ~frame["Short_Profitable_Rebound"]
        & (frame["Weak_Red_3W"] | frame["Stop_Before_10"])
    )
    frame["Outcome_Class"] = np.select(
        [
            frame["Strong_Sustained"], frame["Short_Profitable_Rebound"],
            frame["Weak_Rebound"],
        ],
        ["持续强上涨", "短期盈利反弹", "弱反弹或失败"], default="其他中间结果",
    )
    frame["Target30_Before_Stop"] = frame["First_30_vs_Stop"].astype(str).eq("目标先到")
    frame["Stop_8W"] = frame["Hit_Stop_8W"].map(to_bool)
    frame["Return_8W_pct"] = pd.to_numeric(frame["Return_8W_pct"], errors="coerce")
    return frame


def cache_path(code: str, asset: str, start: str, end: str) -> Path:
    key = hashlib.sha1(f"{code}|{asset}|{start}|{end}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{code.replace('.', '_')}_{asset}_{key}.pkl"


def atomic_pickle(value: Any, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def fetch_history(
    pro: Any, code: str, asset: str, start: str, end: str,
    use_cache: bool, pause: float, retries: int = 3,
) -> tuple[pd.DataFrame, bool, str]:
    path = cache_path(code, asset, start, end)
    if use_cache and path.exists():
        try:
            with path.open("rb") as handle:
                return pickle.load(handle), True, ""
        except Exception:
            pass
    error = ""
    frame = pd.DataFrame()
    for attempt in range(retries):
        try:
            if asset == "I":
                data = ts.pro_bar(
                    api=pro, ts_code=code, asset="I", start_date=start,
                    end_date=end, freq="D",
                )
                if data is None or pd.DataFrame(data).empty:
                    data = pro.index_daily(
                        ts_code=code, start_date=start, end_date=end,
                        fields="ts_code,trade_date,open,high,low,close,vol,amount",
                    )
            else:
                data = ts.pro_bar(
                    api=pro, ts_code=code, start_date=start, end_date=end,
                    adj="qfq", freq="D", factors=["tor"],
                )
            frame = pd.DataFrame() if data is None else data.copy()
            if not frame.empty:
                break
            error = "返回空行情"
        except Exception as exc:
            error = str(exc)
        time.sleep(0.7 * (attempt + 1))
    time.sleep(pause)
    if frame.empty:
        return frame, False, error or "返回空行情"
    frame["trade_date"] = frame["trade_date"].astype(str)
    for column in ("open", "high", "low", "close", "vol", "amount", "turnover_rate"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("open", "high", "low", "close"):
        if column not in frame.columns:
            frame[column] = frame.get("close", np.nan)
    if "vol" not in frame.columns:
        frame["vol"] = np.nan
    if "turnover_rate" not in frame.columns:
        frame["turnover_rate"] = np.nan
    frame = (
        frame.dropna(subset=["trade_date", "open", "high", "low", "close"])
        .drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)
    )
    if use_cache and not frame.empty:
        atomic_pickle(frame, path)
    return frame, False, ""


def recursive_kdj(rsv: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    k_values: list[float] = []
    d_values: list[float] = []
    k_prev = 50.0
    d_prev = 50.0
    for value in rsv:
        if not math.isfinite(num(value)):
            k_values.append(np.nan)
            d_values.append(np.nan)
            continue
        k_prev = 2.0 / 3.0 * k_prev + 1.0 / 3.0 * float(value)
        d_prev = 2.0 / 3.0 * d_prev + 1.0 / 3.0 * k_prev
        k_values.append(k_prev)
        d_values.append(d_prev)
    k = pd.Series(k_values, index=rsv.index)
    d = pd.Series(d_values, index=rsv.index)
    return k, d, 3.0 * k - 2.0 * d


def build_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    work["dt"] = pd.to_datetime(work["trade_date"])
    aggregations: dict[str, str] = {
        "trade_date": "last", "open": "first", "high": "max",
        "low": "min", "close": "last", "vol": "sum",
    }
    if work["turnover_rate"].notna().any():
        aggregations["turnover_rate"] = "sum"
    weekly = (
        work.set_index("dt").resample("W-FRI").agg(aggregations)
        .dropna(subset=["close"]).reset_index().rename(columns={"dt": "week_label"})
    )
    weekly["ema12"] = weekly["close"].ewm(span=12, adjust=False).mean()
    weekly["ema26"] = weekly["close"].ewm(span=26, adjust=False).mean()
    weekly["dif"] = weekly["ema12"] - weekly["ema26"]
    weekly["dea"] = weekly["dif"].ewm(span=9, adjust=False).mean()
    weekly["hist"] = (weekly["dif"] - weekly["dea"]) * 2.0
    weekly["ret4"] = (weekly["close"] / weekly["close"].shift(4) - 1.0) * 100.0
    weekly["ret13"] = (weekly["close"] / weekly["close"].shift(13) - 1.0) * 100.0
    weekly["ret26"] = (weekly["close"] / weekly["close"].shift(26) - 1.0) * 100.0

    low9 = weekly["low"].rolling(9).min()
    high9 = weekly["high"].rolling(9).max()
    spread = (high9 - low9).replace(0, np.nan)
    weekly["rsv9"] = (weekly["close"] - low9) / spread * 100.0
    weekly["k"], weekly["d"], weekly["j"] = recursive_kdj(weekly["rsv9"])

    previous_close = weekly["close"].shift(1)
    tr = pd.concat([
        weekly["high"] - weekly["low"],
        (weekly["high"] - previous_close).abs(),
        (weekly["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    weekly["atr14_pct"] = tr.rolling(14).mean() / weekly["close"] * 100.0
    weekly["atr14_change4_pct"] = (weekly["atr14_pct"] / weekly["atr14_pct"].shift(4) - 1.0) * 100.0
    ma20 = weekly["close"].rolling(20).mean()
    std20 = weekly["close"].rolling(20).std(ddof=0)
    weekly["bb_width20_pct"] = 4.0 * std20 / ma20 * 100.0
    weekly["vol_ma8"] = weekly["vol"].shift(1).rolling(8).mean()
    weekly["w1_volume_ratio8"] = weekly["vol"] / weekly["vol_ma8"]
    if "turnover_rate" in weekly.columns:
        weekly["turnover_ma8"] = weekly["turnover_rate"].shift(1).rolling(8).mean()
        weekly["w1_turnover_ratio8"] = weekly["turnover_rate"] / weekly["turnover_ma8"]
    return weekly.reset_index(drop=True)


def asof_row(weekly: pd.DataFrame, signal_date: str) -> tuple[int, pd.Series | None]:
    if weekly.empty:
        return -1, None
    eligible = weekly.index[weekly["trade_date"].astype(str).le(signal_date)]
    if len(eligible) == 0:
        return -1, None
    position = int(eligible[-1])
    row = weekly.iloc[position]
    # 股票信号必须精确匹配信号周最后交易日，防止误用前一周。
    if str(row["trade_date"]) != signal_date:
        return -1, None
    return position, row


def index_row(weekly: pd.DataFrame, signal_date: str) -> pd.Series | None:
    if weekly.empty:
        return None
    rows = weekly[weekly["trade_date"].astype(str).le(signal_date)]
    return None if rows.empty else rows.iloc[-1]


def green_volume_features(weekly: pd.DataFrame, position: int) -> dict[str, float]:
    start = position - 1
    while start >= 0 and num(weekly.iloc[start]["hist"]) <= 0:
        start -= 1
    green = weekly.iloc[start + 1:position]
    prior = weekly.iloc[max(0, start - 7):start + 1]
    green_mean = pd.to_numeric(green.get("vol"), errors="coerce").mean()
    prior_mean = pd.to_numeric(prior.get("vol"), errors="coerce").mean()
    ratio = green_mean / prior_mean if prior_mean and math.isfinite(prior_mean) else np.nan
    if len(green) >= 4:
        split = max(1, len(green) // 2)
        first = pd.to_numeric(green.iloc[:split]["vol"], errors="coerce").mean()
        last = pd.to_numeric(green.iloc[split:]["vol"], errors="coerce").mean()
        late_early = last / first if first and math.isfinite(first) else np.nan
    else:
        late_early = np.nan
    return {
        "Green_Weeks_Rebuilt": len(green),
        "Green_Volume_vs_Prior_Ratio": ratio,
        "Green_Late_vs_Early_Volume_Ratio": late_early,
    }


def percentile_last(series: pd.Series, position: int, window: int = 52) -> float:
    values = pd.to_numeric(series.iloc[max(0, position - window + 1):position + 1], errors="coerce").dropna()
    current = num(series.iloc[position])
    if len(values) < 20 or not math.isfinite(current):
        return np.nan
    return float((values <= current).mean() * 100.0)


def event_factors(
    event: pd.Series, stock_weekly: pd.DataFrame,
    board_weekly: pd.DataFrame, broad_weekly: pd.DataFrame,
) -> tuple[dict[str, Any] | None, str]:
    signal = str(event["Signal_Date"])
    position, row = asof_row(stock_weekly, signal)
    if row is None or position < 52:
        return None, "信号周未匹配或周线不足52周"
    board = index_row(board_weekly, signal)
    broad = index_row(broad_weekly, signal)
    if board is None or broad is None:
        return None, "基准指数周线不足"
    green = green_volume_features(stock_weekly, position)
    k, d, j = num(row["k"]), num(row["d"]), num(row["j"])
    prev = stock_weekly.iloc[position - 1]
    prev2 = stock_weekly.iloc[position - 2]
    k_prev, d_prev = num(prev["k"]), num(prev["d"])
    cross_now = k > d and k_prev <= d_prev
    cross_prev = k_prev > d_prev and num(prev2["k"]) <= num(prev2["d"])
    level = (k + d) / 2.0 if math.isfinite(k) and math.isfinite(d) else np.nan
    if not math.isfinite(level):
        zone = "数据不足"
    elif level < 20:
        zone = "低位<20"
    elif level < 50:
        zone = "20—50"
    elif level < 80:
        zone = "50—80"
    else:
        zone = "高位≥80"

    board_rs13 = num(row["ret13"]) - num(board["ret13"])
    board_rs26 = num(row["ret26"]) - num(board["ret26"])
    broad_rs13 = num(row["ret13"]) - num(broad["ret13"])
    broad_rs26 = num(row["ret26"]) - num(broad["ret26"])
    w1_volume_ratio = num(row.get("w1_volume_ratio8"))
    green_volume_ratio = num(green["Green_Volume_vs_Prior_Ratio"])
    atr_change = num(row["atr14_change4_pct"])
    bb_percentile = percentile_last(stock_weekly["bb_width20_pct"], position, 52)
    kdj_bullish = bool(k > d and k > k_prev and d > d_prev)

    rs_pass = bool(board_rs13 > 0 and board_rs26 > 0)
    volume_pass = bool(green_volume_ratio <= 1.0 and w1_volume_ratio >= 1.0)
    volatility_pass = bool(atr_change <= 0 and bb_percentile <= 50.0)
    kdj_pass = kdj_bullish
    score = int(rs_pass) + int(volume_pass) + int(volatility_pass) + int(kdj_pass)
    return {
        "Board_Index": BOARD_INDEX.get(str(event.get("Sample_Board")), ""),
        "Stock_Return_4W_pct": num(row["ret4"]),
        "Stock_Return_13W_pct": num(row["ret13"]),
        "Stock_Return_26W_pct": num(row["ret26"]),
        "Board_RS_13W_pct": board_rs13,
        "Board_RS_26W_pct": board_rs26,
        "Broad_RS_13W_pct": broad_rs13,
        "Broad_RS_26W_pct": broad_rs26,
        "RS_Pass": rs_pass,
        **green,
        "W1_Volume_Ratio_8W": w1_volume_ratio,
        "W1_Turnover_Ratio_8W": num(row.get("w1_turnover_ratio8")),
        "Volume_Pass": volume_pass,
        "ATR14_pct": num(row["atr14_pct"]),
        "ATR14_Change4W_pct": atr_change,
        "BB_Width20_pct": num(row["bb_width20_pct"]),
        "BB_Width_Percentile52": bb_percentile,
        "Volatility_Pass": volatility_pass,
        "KDJ_K": k, "KDJ_D": d, "KDJ_J": j, "KDJ_Level": level,
        "KDJ_Zone": zone, "KDJ_Cross_This_Week": cross_now,
        "KDJ_Cross_Previous_Week": cross_prev,
        "KDJ_Cross_Within_2W": bool(cross_now or cross_prev),
        "KDJ_KD_Both_Rising": bool(k > k_prev and d > d_prev),
        "KDJ_Bullish_Pass": kdj_pass,
        "J_Distance_From_50": abs(j - 50.0),
        "Factor_Score_0_4": score,
        "All_4_Pass": score == 4,
    }, ""


def enrich_events(
    events: pd.DataFrame, histories: dict[str, pd.DataFrame],
    indexes: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        code = str(event["ts_code"])
        board_code = BOARD_INDEX.get(str(event.get("Sample_Board")), "")
        factors, reason = event_factors(
            event, histories.get(code, pd.DataFrame()),
            indexes.get(board_code, pd.DataFrame()), indexes.get(BROAD_INDEX, pd.DataFrame()),
        )
        if factors is None:
            failures.append({
                "Cycle_ID": event["Cycle_ID"], "ts_code": code,
                "name": event.get("name", ""), "Signal_Date": event["Signal_Date"],
                "失败原因": reason,
            })
            continue
        rows.append({**event.to_dict(), **factors})
    return pd.DataFrame(rows), pd.DataFrame(failures)


def rate(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.astype(bool).mean() * 100.0) if len(clean) else np.nan


def metrics(group: pd.DataFrame) -> dict[str, Any]:
    return {
        "样本数": len(group),
        "持续强上涨(%)": rate(group["Strong_Sustained"]),
        "弱反弹或失败(%)": rate(group["Weak_Rebound"]),
        "短期盈利反弹(%)": rate(group["Short_Profitable_Rebound"]),
        "30%先于止损(%)": rate(group["Target30_Before_Stop"]),
        "八周止损率(%)": rate(group["Stop_8W"]),
        "八周收益均值(%)": pd.to_numeric(group["Return_8W_pct"], errors="coerce").mean(),
        "八周收益中位数(%)": pd.to_numeric(group["Return_8W_pct"], errors="coerce").median(),
        "红柱周数中位数": pd.to_numeric(group["Red_Cycle_Weeks"], errors="coerce").median(),
    }


def group_report(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(columns, keys)), **metrics(group)})
    return pd.DataFrame(rows)


def factor_pass_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [{"因子": "全部样本", "状态": "基准", **metrics(frame)}]
    definitions = [
        ("相对强度", "RS_Pass"), ("回调缩量+启动放量", "Volume_Pass"),
        ("波动率收缩", "Volatility_Pass"), ("KDJ方向", "KDJ_Bullish_Pass"),
    ]
    for name, column in definitions:
        rows.append({"因子": name, "状态": "通过", **metrics(frame[frame[column].map(to_bool)])})
        rows.append({"因子": name, "状态": "未通过", **metrics(frame[~frame[column].map(to_bool)])})
    return pd.DataFrame(rows)


def safe_quartiles(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    result = pd.Series("缺失", index=series.index, dtype="object")
    if valid.nunique() < 4:
        result.loc[valid.index] = "有效值"
        return result
    try:
        result.loc[valid.index] = pd.qcut(valid.rank(method="first"), 4, labels=["Q1低", "Q2", "Q3", "Q4高"])
    except ValueError:
        result.loc[valid.index] = "有效值"
    return result


def quartile_report(frame: pd.DataFrame) -> pd.DataFrame:
    fields = {
        "相对板块13周": "Board_RS_13W_pct",
        "相对板块26周": "Board_RS_26W_pct",
        "绿柱期成交量比": "Green_Volume_vs_Prior_Ratio",
        "首红柱启动量比": "W1_Volume_Ratio_8W",
        "ATR四周变化": "ATR14_Change4W_pct",
        "布林带宽年度分位": "BB_Width_Percentile52",
        "KDJ位置": "KDJ_Level",
        "J距50": "J_Distance_From_50",
    }
    rows = []
    for name, column in fields.items():
        bins = safe_quartiles(frame[column])
        for bucket, group in frame.groupby(bins, dropna=False):
            values = pd.to_numeric(group[column], errors="coerce")
            rows.append({
                "连续因子": name, "分组": str(bucket), "因子最小值": values.min(),
                "因子中位数": values.median(), "因子最大值": values.max(), **metrics(group),
            })
    return pd.DataFrame(rows)


def yearly_score_report(frame: pd.DataFrame) -> pd.DataFrame:
    return group_report(frame, ["Signal_Year", "Factor_Score_0_4"])


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def build_result(
    enriched: pd.DataFrame, failures: pd.DataFrame, source: str,
    fetch_failures: list[dict[str, Any]], start_date: str, end_date: str,
    cache_hits: int,
) -> dict[str, Any]:
    pass_report = factor_pass_report(enriched)
    score_report = group_report(enriched, ["Factor_Score_0_4"])
    quartiles = quartile_report(enriched)
    kdj = group_report(enriched, ["KDJ_Zone", "KDJ_Cross_Within_2W", "KDJ_KD_Both_Rising"])
    yearly = yearly_score_report(enriched)
    boards = group_report(enriched, ["Sample_Board", "Factor_Score_0_4"])
    fetch_failure_frame = pd.DataFrame(fetch_failures)
    metadata = pd.DataFrame([
        {"项目": "程序", "值": TITLE}, {"项目": "输入", "值": source},
        {"项目": "生成时间", "值": datetime.now().isoformat(timespec="seconds")},
        {"项目": "成功计算事件", "值": len(enriched)},
        {"项目": "因子计算失败事件", "值": len(failures)},
        {"项目": "行情失败代码", "值": len(fetch_failure_frame)},
        {"项目": "股票缓存命中", "值": cache_hits},
        {"项目": "行情开始", "值": start_date}, {"项目": "行情结束", "值": end_date},
        {"项目": "KDJ参数", "值": "9,3,3；J上穿K/D不作为独立因子"},
        {"项目": "评分", "值": "相对强度、量价结构、波动率收缩、KDJ方向各1分；不调参"},
        {"项目": "用途", "值": "因子审查和排序研究，不是硬过滤或交易策略"},
    ])
    files = {
        "01_factor_events.csv": csv_bytes(enriched),
        "02_factor_pass_report.csv": csv_bytes(pass_report),
        "03_score_report.csv": csv_bytes(score_report),
        "04_continuous_quartiles.csv": csv_bytes(quartiles),
        "05_kdj_detail.csv": csv_bytes(kdj),
        "06_year_score_stability.csv": csv_bytes(yearly),
        "07_board_score_stability.csv": csv_bytes(boards),
        "08_event_failures.csv": csv_bytes(failures),
        "09_fetch_failures.csv": csv_bytes(fetch_failure_frame),
        "10_metadata.csv": csv_bytes(metadata),
    }
    return {
        "enriched": enriched, "pass_report": pass_report, "score_report": score_report,
        "quartiles": quartiles, "kdj": kdj, "yearly": yearly, "boards": boards,
        "failures": failures, "fetch_failures": fetch_failure_frame, "metadata": metadata,
        "files": files, "zip": make_zip(files),
    }


def show(frame: pd.DataFrame) -> None:
    formats = {
        column: "{:.2f}" for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and ("(%)" in column or column in {"因子最小值", "因子中位数", "因子最大值", "红柱周数中位数"})
    }
    st.dataframe(frame.style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)


def render(result: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("成功事件", f"{len(result['enriched']):,}")
    c2.metric("计算失败事件", f"{len(result['failures']):,}")
    c3.metric("四项全通过", f"{int(result['enriched']['All_4_Pass'].sum()):,}")
    c4.metric("持续强上涨基准", f"{rate(result['enriched']['Strong_Sustained']):.2f}%")
    st.subheader("四类因素分别是否有帮助")
    show(result["pass_report"])
    st.subheader("固定0—4分")
    show(result["score_report"])
    st.subheader("逐年稳定性")
    show(result["yearly"])
    st.subheader("连续因子四分位")
    show(result["quartiles"])
    st.subheader("KDJ位置与方向")
    show(result["kdj"])
    with st.expander("分板块和失败审计"):
        show(result["boards"])
        show(result["failures"])
        show(result["fetch_failures"])
    st.subheader("下载")
    st.download_button(
        "下载全部结果ZIP", result["zip"],
        file_name="weekly_macd_strength_factors_v4_4_all_results.zip",
        mime="application/zip", type="primary", key="v44_all", on_click="ignore",
    )
    labels = [
        "1号：事件明细", "2号：四因素", "3号：总评分", "4号：连续分组", "5号：KDJ",
        "6号：逐年", "7号：分板块", "8号：事件失败", "9号：行情失败", "10号：运行信息",
    ]
    columns = st.columns(5)
    for index, (name, data) in enumerate(result["files"].items()):
        with columns[index % 5]:
            st.download_button(
                labels[index], data, file_name=name, mime="text/csv",
                key=f"v44_{name}", on_click="ignore",
            )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.info(
        "本版不减少第一根红柱事件，只验证四类信息是否能区分弱反弹、短期盈利反弹和持续强上涨。"
        "固定自然阈值形成0—4分，不搜索最佳参数。"
    )
    with st.expander("四项固定评分", expanded=False):
        st.markdown(
            """
            - **相对强度1分**：个股过去13周、26周均跑赢所属板块基准。
            - **量价结构1分**：绿柱回调期均量不高于此前阶段，第一根红柱成交量不低于此前8周均量。
            - **波动率收缩1分**：周ATR较四周前下降，布林带宽处于过去52周下半区。
            - **KDJ方向1分**：K>D并且K、D同时上升。J与K/D交叉数学上等价，不重复加分。

            持续强上涨定义为：红柱至少9周，并且+30%先于-10%止损。
            弱反弹或失败定义为：红柱不超过3周，或者未能在止损前达到+10%。
            """
        )
    with st.sidebar:
        token = st.text_input("Tushare Token", type="password")
        use_cache = st.checkbox("使用逐股票缓存和中断续跑", value=True)
        pause = st.number_input("每次行情请求后暂停(秒)", 0.0, 2.0, 0.10, 0.05)
    with st.form("v44_form"):
        upload = st.file_uploader(
            "上传V4.1全部结果ZIP或01_events.csv", type=["zip", "csv"],
            help="只补取事件涉及股票的历史行情，不重新筛选全市场。",
        )
        submitted = st.form_submit_button("开始四因子验证", type="primary")
    if submitted:
        if not token:
            st.error("请输入Tushare Token。")
        elif upload is None:
            st.error("请上传V4.1结果。")
        else:
            try:
                raw, source = load_events(upload)
                validate(raw)
                events = prepare_events(raw)
                if events.empty:
                    raise ValueError("没有完整八周的第一根红柱事件。")
                min_signal = datetime.strptime(events["Signal_Date"].min(), "%Y%m%d")
                start_date = (min_signal - timedelta(days=3 * 365)).strftime("%Y%m%d")
                end_date = events["Signal_Date"].max()
                ts.set_token(token)
                pro = ts.pro_api()
                status = st.empty()
                progress = st.progress(0.0)

                indexes: dict[str, pd.DataFrame] = {}
                fetch_failures: list[dict[str, Any]] = []
                for code in sorted(set(BOARD_INDEX.values()) | {BROAD_INDEX}):
                    daily, _, error = fetch_history(
                        pro, code, "I", start_date, end_date, bool(use_cache), float(pause)
                    )
                    indexes[code] = build_weekly(daily)
                    if daily.empty:
                        fetch_failures.append({"代码": code, "名称": INDEX_NAME.get(code, ""), "失败原因": error})
                codes = sorted(events["ts_code"].astype(str).unique())
                histories: dict[str, pd.DataFrame] = {}
                cache_hits = 0
                for index, code in enumerate(codes, start=1):
                    status.write(f"正在计算 {index}/{len(codes)}：{code}；缓存命中 {cache_hits}")
                    daily, hit, error = fetch_history(
                        pro, code, "E", start_date, end_date, bool(use_cache), float(pause)
                    )
                    cache_hits += int(hit)
                    if daily.empty:
                        fetch_failures.append({"代码": code, "名称": "", "失败原因": error})
                    histories[code] = build_weekly(daily)
                    progress.progress(index / len(codes))
                enriched, event_failures = enrich_events(events, histories, indexes)
                if enriched.empty:
                    raise ValueError("没有事件成功计算四因子，请检查指数权限和行情。")
                st.session_state["v44_result"] = build_result(
                    enriched, event_failures, source, fetch_failures,
                    start_date, end_date, cache_hits,
                )
                status.success(f"完成：{len(enriched):,}个事件；缓存命中{cache_hits:,}只股票。")
            except Exception as exc:
                st.exception(exc)
    if "v44_result" in st.session_state:
        render(st.session_state["v44_result"])
    else:
        st.caption("本次只涉及V4.1事件中的约661只股票；首次运行需要补取行情，后续可从缓存续跑。")


if __name__ == "__main__":
    main()
