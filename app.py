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


TITLE = "真正周线MACD × V40.6混合验证器 V4.6"
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "hybrid_v46_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 20260810
TOP_N = 3

REQUIRED_BASE = {
    "Cycle_ID", "ts_code", "name", "Sample_Board", "Signal_Date",
    "Board_RS_13W_pct", "Red_Cycle_Weeks", "Return_8W_pct",
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
        raise ValueError("请上传V4.4/V4.5全部结果ZIP，或事件CSV。")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        preferences = [
            "01_factor_events.csv", "01_ranked_candidates.csv", "01_events.csv",
        ]
        target = ""
        for preferred in preferences:
            matches = [name for name in names if Path(name).name.lower() == preferred]
            if matches:
                target = sorted(matches, key=len)[0]
                break
        if not target:
            matches = [
                name for name in names
                if name.lower().endswith(".csv")
                and any(word in Path(name).name.lower() for word in ("event", "candidate"))
            ]
            if matches:
                target = sorted(matches, key=len)[0]
        if not target:
            raise ValueError("ZIP中没有找到V4.4/V4.5事件文件。")
        return read_csv(archive.read(target)), f"{uploaded.name}/{target}"


def validate_events(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_BASE - set(frame.columns))
    if missing:
        raise ValueError("输入事件文件缺少字段：" + "、".join(missing))
    if "Exit_T30_Return_pct" not in frame.columns and "Primary_Return_pct" not in frame.columns:
        raise ValueError("缺少Exit_T30_Return_pct或Primary_Return_pct，无法统一比较收益。")


def prepare_events(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    if "Event_Type" in frame.columns:
        frame = frame[frame["Event_Type"].astype(str).eq("第一根红柱")]
    if "Tradable" in frame.columns:
        frame = frame[frame["Tradable"].map(to_bool)]
    if "Has_8W_Future" in frame.columns:
        frame = frame[frame["Has_8W_Future"].map(to_bool)]
    frame = frame.drop_duplicates("Cycle_ID").reset_index(drop=True)
    frame["Signal_Date"] = frame["Signal_Date"].map(date8)
    frame = frame[frame["Signal_Date"].str.len().eq(8)].copy()
    frame["Signal_Year"] = pd.to_numeric(
        frame.get("Signal_Year", frame["Signal_Date"].str[:4]), errors="coerce"
    ).astype("Int64")

    numeric = [
        "Board_RS_13W_pct", "Red_Cycle_Weeks", "Return_8W_pct",
        "Exit_T30_Return_pct", "Primary_Return_pct", "Entry_Price",
    ]
    for column in numeric:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    red_weeks = pd.to_numeric(frame["Red_Cycle_Weeks"], errors="coerce")
    completed = frame.get("Cycle_Completed", pd.Series(False, index=frame.index)).map(to_bool)
    first_10 = frame.get("First_10_vs_Stop", pd.Series("", index=frame.index)).astype(str)
    first_20 = frame.get("First_20_vs_Stop", pd.Series("", index=frame.index)).astype(str)
    first_30 = frame.get("First_30_vs_Stop", pd.Series("", index=frame.index)).astype(str)
    if "Strong_Sustained" not in frame.columns:
        frame["Strong_Sustained"] = red_weeks.ge(9) & first_30.eq("目标先到")
    else:
        frame["Strong_Sustained"] = frame["Strong_Sustained"].map(to_bool)
    if "Short_Profitable_Rebound" not in frame.columns:
        frame["Short_Profitable_Rebound"] = (
            ~frame["Strong_Sustained"]
            & red_weeks.lt(9)
            & first_20.eq("目标先到")
        )
    else:
        frame["Short_Profitable_Rebound"] = frame["Short_Profitable_Rebound"].map(to_bool)
    if "Weak_Rebound" not in frame.columns:
        weak_three = completed & red_weeks.le(3)
        stop_before_10 = ~first_10.eq("目标先到")
        frame["Weak_Rebound"] = (
            ~frame["Strong_Sustained"]
            & ~frame["Short_Profitable_Rebound"]
            & (weak_three | stop_before_10)
        )
    else:
        frame["Weak_Rebound"] = frame["Weak_Rebound"].map(to_bool)
    if "Target30_Before_Stop" not in frame.columns:
        frame["Target30_Before_Stop"] = first_30.eq("目标先到")
    else:
        frame["Target30_Before_Stop"] = frame["Target30_Before_Stop"].map(to_bool)
    if "Stop_8W" not in frame.columns:
        frame["Stop_8W"] = frame.get(
            "Hit_Stop_8W", pd.Series(False, index=frame.index)
        ).map(to_bool)
    else:
        frame["Stop_8W"] = frame["Stop_8W"].map(to_bool)

    primary = "Exit_T30_Return_pct" if "Exit_T30_Return_pct" in frame.columns else "Primary_Return_pct"
    frame["Primary_Return_pct"] = pd.to_numeric(frame[primary], errors="coerce")
    frame = frame.dropna(subset=["Primary_Return_pct", "Board_RS_13W_pct"]).copy()
    frame["RS13_Weekly_Percentile"] = (
        frame.groupby("Signal_Date")["Board_RS_13W_pct"]
        .rank(method="average", pct=True) * 100.0
    )
    return frame.sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True)


def cache_path(code: str, start: str, end: str) -> Path:
    key = hashlib.sha1(f"{code}|{start}|{end}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{code.replace('.', '_')}_{key}.pkl"


def atomic_pickle(value: Any, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def fetch_daily(
    pro: Any, code: str, start: str, end: str, use_cache: bool,
    pause: float, retries: int = 3,
) -> tuple[pd.DataFrame, bool, str]:
    path = cache_path(code, start, end)
    if use_cache and path.exists():
        try:
            with path.open("rb") as handle:
                cached = pickle.load(handle)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached, True, ""
        except Exception:
            pass
    error = ""
    frame = pd.DataFrame()
    for attempt in range(retries):
        try:
            data = ts.pro_bar(
                api=pro, ts_code=code, start_date=start, end_date=end,
                adj="qfq", freq="D",
            )
            frame = pd.DataFrame() if data is None else pd.DataFrame(data).copy()
            if not frame.empty:
                break
            error = "返回空行情"
        except Exception as exc:
            error = str(exc)
        time.sleep(0.8 * (attempt + 1))
    time.sleep(pause)
    if frame.empty:
        return frame, False, error or "返回空行情"
    frame["trade_date"] = frame["trade_date"].astype(str)
    for column in ("open", "high", "low", "close", "pre_close", "vol"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["trade_date", "open", "high", "low", "close", "vol"])
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date").reset_index(drop=True)
    )
    if use_cache and not frame.empty:
        atomic_pickle(frame, path)
    return frame, False, ""


def build_daily(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy().sort_values("trade_date").reset_index(drop=True)
    frame["ma20"] = frame["close"].rolling(20).mean()
    frame["ma60"] = frame["close"].rolling(60).mean()
    frame["ma120"] = frame["close"].rolling(120).mean()
    frame["ma5_vol"] = frame["vol"].shift(1).rolling(5).mean()
    frame["box_high_10"] = frame["high"].shift(1).rolling(10).max()
    frame["ema12"] = frame["close"].ewm(span=12, adjust=False).mean()
    frame["ema26"] = frame["close"].ewm(span=26, adjust=False).mean()
    frame["dif"] = frame["ema12"] - frame["ema26"]
    frame["dea"] = frame["dif"].ewm(span=9, adjust=False).mean()
    frame["macd"] = (frame["dif"] - frame["dea"]) * 2.0
    frame["range"] = frame["high"] - frame["low"]
    frame["body"] = frame["close"] - frame["open"]
    frame["body_ratio"] = np.where(frame["range"].gt(0), frame["body"] / frame["range"], np.nan)
    return frame


def build_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    work["dt"] = pd.to_datetime(work["trade_date"])
    weekly = (
        work.set_index("dt").resample("W-FRI").agg({
            "trade_date": "last", "open": "first", "high": "max",
            "low": "min", "close": "last", "vol": "sum",
        }).dropna(subset=["close"]).reset_index(drop=True)
    )
    weekly["ema12"] = weekly["close"].ewm(span=12, adjust=False).mean()
    weekly["ema26"] = weekly["close"].ewm(span=26, adjust=False).mean()
    weekly["dif"] = weekly["ema12"] - weekly["ema26"]
    weekly["dea"] = weekly["dif"].ewm(span=9, adjust=False).mean()
    weekly["hist"] = (weekly["dif"] - weekly["dea"]) * 2.0
    weekly["w_ma20"] = weekly["close"].rolling(20).mean()
    return weekly.reset_index(drop=True)


def true_weekly_wave_count(weekly: pd.DataFrame, position: int) -> int:
    if position < 1:
        return -1
    window = weekly.iloc[max(0, position - 51):position + 1].copy().reset_index(drop=True)
    if len(window) < 26:
        return -1
    start = int(window["low"].idxmin())
    segment = window.iloc[start:].reset_index(drop=True)
    if len(segment) < 5:
        return 0
    running_max = num(segment.iloc[0]["high"])
    in_pullback = False
    count = 0
    for index in range(1, len(segment)):
        row = segment.iloc[index]
        high, low, hist = num(row["high"]), num(row["low"]), num(row["hist"])
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
    if not math.isfinite(percentile):
        return 0.0
    if 60 < percentile <= 80:
        return 30.0
    if 40 < percentile <= 60 or 80 < percentile <= 90:
        return 20.0
    if 20 < percentile <= 40 or 90 < percentile <= 100:
        return 10.0
    return 5.0


def bucket_breakout(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        return 0.0
    if value <= 1:
        return 18.0
    if value <= 3:
        return 25.0
    if value <= 5:
        return 15.0
    return 5.0


def bucket_volume(value: float) -> float:
    if not math.isfinite(value) or value < 1.3 or value > 3.0:
        return 0.0
    if value < 1.6:
        return 15.0
    if value <= 2.2:
        return 20.0
    if value <= 2.6:
        return 12.0
    return 5.0


def bucket_bias(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if value <= 15:
        return 15.0
    if value <= 25:
        return 10.0
    if value <= 35:
        return 5.0
    if value <= 45:
        return 2.0
    return 0.0


def bucket_body(value: float) -> float:
    if not math.isfinite(value) or value < 0.60:
        return 0.0
    if value <= 0.80:
        return 10.0
    if value <= 0.95:
        return 8.0
    return 5.0


def event_indicators(event: pd.Series, daily: pd.DataFrame, weekly: pd.DataFrame) -> dict[str, Any]:
    signal = str(event["Signal_Date"])
    output: dict[str, Any] = {
        "Data_Available": False, "Data_Error": "", "True_First_Red_Audit": False,
        "Wave_Count_True_Weekly": np.nan, "Wave_2_5_Pass": False,
        "Weekly_Bias_pct": np.nan, "Weekly_Bias_Pass": False,
        "Prev_Upper_Shadow_Ratio": np.nan, "Weekly_Shadow_Pass": False,
        "Weekly_Safe_Pass": False, "Daily_Trend_Pass": False,
        "Box_Breakout_Pass": False, "Daily_Breakout_Pass": False,
        "MA20_Healthy_Pass": False, "Volume_Pass": False,
        "Solid_Yang_Pass": False, "Daily_MACD_Pass": False,
        "V406_All_Pass_Before_Gap": False, "Gap_pct": np.nan,
        "One_Word_Pass": False, "Gap_Pass": False, "V406_Tradable_Pass": False,
        "Daily_pct": np.nan, "Volume_Ratio": np.nan,
        "Box_Breakout_pct": np.nan, "Body_Ratio": np.nan,
        "Original_V406_Score": np.nan, "RS_Position_Score": np.nan,
        "Breakout_Quality_Score": np.nan, "Volume_Quality_Score": np.nan,
        "Weekly_Bias_Score": np.nan, "Body_Quality_Score": np.nan,
        "New_Score_100": np.nan, "First_Fail_Stage": "数据不可用",
    }
    if daily.empty or weekly.empty:
        output["Data_Error"] = "行情为空"
        return output
    daily_positions = daily.index[daily["trade_date"].astype(str).eq(signal)]
    weekly_positions = weekly.index[weekly["trade_date"].astype(str).le(signal)]
    if len(daily_positions) == 0 or len(weekly_positions) == 0:
        output["Data_Error"] = "信号日不在行情中"
        return output
    dpos, wpos = int(daily_positions[-1]), int(weekly_positions[-1])
    if dpos < 121 or wpos < 26:
        output["Data_Error"] = "历史长度不足"
        return output
    row, previous = daily.iloc[dpos], daily.iloc[dpos - 1]
    wrow, wprevious = weekly.iloc[wpos], weekly.iloc[wpos - 1]
    output["Data_Available"] = True
    output["True_First_Red_Audit"] = bool(num(wprevious["hist"]) <= 0 < num(wrow["hist"]))

    wave_count = true_weekly_wave_count(weekly, wpos)
    output["Wave_Count_True_Weekly"] = wave_count
    output["Wave_2_5_Pass"] = 2 <= wave_count <= 5

    weekly_bias = (
        (num(wrow["close"]) / num(wrow["w_ma20"]) - 1.0) * 100.0
        if num(wrow["w_ma20"]) > 0 else np.nan
    )
    previous_range = num(wprevious["high"]) - num(wprevious["low"])
    previous_shadow = num(wprevious["high"]) - max(num(wprevious["open"]), num(wprevious["close"]))
    shadow_ratio = previous_shadow / previous_range if previous_range > 0 else 0.0
    output["Weekly_Bias_pct"] = weekly_bias
    output["Weekly_Bias_Pass"] = bool(not math.isfinite(weekly_bias) or weekly_bias <= 45.0)
    output["Prev_Upper_Shadow_Ratio"] = shadow_ratio
    output["Weekly_Shadow_Pass"] = bool(shadow_ratio < 0.60)
    output["Weekly_Safe_Pass"] = output["Weekly_Bias_Pass"] and output["Weekly_Shadow_Pass"]

    trend = num(row["ma60"]) > num(row["ma120"])
    box = num(row["close"]) > num(row["box_high_10"]) and num(previous["close"]) <= num(previous["box_high_10"])
    daily_breakout = num(row["close"]) > num(row["ma20"]) * 1.02
    ma20_healthy = num(row["ma20"]) >= num(previous["ma20"])
    volume_ratio = num(row["vol"]) / num(row["ma5_vol"]) if num(row["ma5_vol"]) > 0 else np.nan
    volume_pass = math.isfinite(volume_ratio) and 1.3 <= volume_ratio <= 3.0
    body_ratio = num(row["body_ratio"])
    solid = num(row["close"]) > num(row["open"]) and math.isfinite(body_ratio) and body_ratio >= 0.60
    daily_macd = num(row["dif"]) > 0 and num(row["macd"]) > num(previous["macd"])
    output.update({
        "Daily_Trend_Pass": bool(trend), "Box_Breakout_Pass": bool(box),
        "Daily_Breakout_Pass": bool(daily_breakout), "MA20_Healthy_Pass": bool(ma20_healthy),
        "Volume_Pass": bool(volume_pass), "Solid_Yang_Pass": bool(solid),
        "Daily_MACD_Pass": bool(daily_macd), "Volume_Ratio": volume_ratio,
        "Body_Ratio": body_ratio,
    })
    before_gap = all([
        output["True_First_Red_Audit"], output["Wave_2_5_Pass"],
        output["Weekly_Safe_Pass"], output["Daily_Trend_Pass"],
        output["Box_Breakout_Pass"], output["Daily_Breakout_Pass"],
        output["MA20_Healthy_Pass"], output["Volume_Pass"],
        output["Solid_Yang_Pass"], output["Daily_MACD_Pass"],
    ])
    output["V406_All_Pass_Before_Gap"] = before_gap

    signal_close = num(row["close"])
    daily_pct = (signal_close / num(previous["close"]) - 1.0) * 100.0 if num(previous["close"]) > 0 else np.nan
    breakout_pct = (signal_close / num(row["box_high_10"]) - 1.0) * 100.0 if num(row["box_high_10"]) > 0 else np.nan
    # 跳空和一字板必须使用未加买入滑点的下一交易日原始开盘价。
    # 输入事件的Entry_Price只在行情尾端缺少下一日时作兜底。
    entry_price = np.nan
    one_word_pass = False
    if dpos + 1 < len(daily):
        next_row = daily.iloc[dpos + 1]
        entry_price = num(daily.iloc[dpos + 1]["open"])
        is_main_board = not str(event["ts_code"]).startswith(("300", "301", "688", "689"))
        is_one_word = (
            is_main_board
            and num(next_row["open"]) == num(next_row["high"]) == num(next_row["low"])
        )
        one_word_pass = not is_one_word
    else:
        entry_price = num(event.get("Entry_Price", np.nan))
        one_word_pass = math.isfinite(entry_price)
    gap = (entry_price / signal_close - 1.0) * 100.0 if signal_close > 0 and math.isfinite(entry_price) else np.nan
    gap_pass = math.isfinite(gap) and -3.0 <= gap <= 5.0
    output.update({
        "Daily_pct": daily_pct, "Box_Breakout_pct": breakout_pct,
        "Gap_pct": gap, "One_Word_Pass": one_word_pass, "Gap_Pass": gap_pass,
        "V406_Tradable_Pass": before_gap and one_word_pass and gap_pass,
        "Original_V406_Score": daily_pct * 10.0 + volume_ratio * 10.0
        if math.isfinite(daily_pct) and math.isfinite(volume_ratio) else np.nan,
    })
    rs_score = bucket_rs(num(event["RS13_Weekly_Percentile"]))
    breakout_score = bucket_breakout(breakout_pct)
    volume_score = bucket_volume(volume_ratio)
    bias_score = bucket_bias(weekly_bias)
    body_score = bucket_body(body_ratio)
    output.update({
        "RS_Position_Score": rs_score, "Breakout_Quality_Score": breakout_score,
        "Volume_Quality_Score": volume_score, "Weekly_Bias_Score": bias_score,
        "Body_Quality_Score": body_score,
        "New_Score_100": rs_score + breakout_score + volume_score + bias_score + body_score,
    })

    stages = [
        ("真正周线第一红柱核对", output["True_First_Red_Audit"]),
        ("真正周线2—5次波浪", output["Wave_2_5_Pass"]),
        ("周线风控", output["Weekly_Safe_Pass"]),
        ("日线MA60>MA120", output["Daily_Trend_Pass"]),
        ("十日箱体首发", output["Box_Breakout_Pass"]),
        ("收盘高于MA20的2%", output["Daily_Breakout_Pass"]),
        ("MA20向上", output["MA20_Healthy_Pass"]),
        ("1.3—3倍温和放量", output["Volume_Pass"]),
        ("实体阳线", output["Solid_Yang_Pass"]),
        ("日线MACD健康", output["Daily_MACD_Pass"]),
        ("非一字板可买", output["One_Word_Pass"]),
        ("T+1开盘−3%至+5%", output["Gap_Pass"]),
    ]
    output["First_Fail_Stage"] = "全部通过"
    for name, passed in stages:
        if not passed:
            output["First_Fail_Stage"] = name
            break
    return output


def enrich_events(
    events: pd.DataFrame, histories: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        daily, weekly = histories.get(str(event["ts_code"]), (pd.DataFrame(), pd.DataFrame()))
        rows.append({**event.to_dict(), **event_indicators(event, daily, weekly)})
    return pd.DataFrame(rows)


def full_week_count(frame: pd.DataFrame) -> int:
    dates = pd.to_datetime(frame["Signal_Date"], format="%Y%m%d", errors="coerce").dropna()
    if dates.empty:
        return 0
    return len(pd.period_range(dates.min(), dates.max(), freq="W-FRI"))


def rate(series: pd.Series) -> float:
    return float(series.map(to_bool).mean() * 100.0) if len(series) else np.nan


def profit_factor(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    positive = clean[clean > 0].sum()
    negative = -clean[clean < 0].sum()
    return float(positive / negative) if negative > 0 else np.nan


def contribution_ratio(series: pd.Series, top_n: int) -> float:
    positive = pd.to_numeric(series, errors="coerce").dropna()
    positive = positive[positive > 0].sort_values(ascending=False)
    return float(positive.head(top_n).sum() / positive.sum() * 100.0) if positive.sum() > 0 else np.nan


def stock_concentration(frame: pd.DataFrame, top_n: int) -> float:
    if frame.empty:
        return np.nan
    values = frame.groupby("ts_code")["Primary_Return_pct"].sum().sort_values(ascending=False)
    positive = values[values > 0]
    return float(positive.head(top_n).sum() / positive.sum() * 100.0) if positive.sum() > 0 else np.nan


def mean_without_top_stocks(frame: pd.DataFrame, top_n: int) -> float:
    if frame.empty:
        return np.nan
    values = frame.groupby("ts_code")["Primary_Return_pct"].sum().sort_values(ascending=False)
    remaining = frame[~frame["ts_code"].isin(set(values.head(top_n).index))]
    return float(remaining["Primary_Return_pct"].mean()) if len(remaining) else np.nan


def metrics(frame: pd.DataFrame, interval_weeks: int) -> dict[str, Any]:
    if frame.empty:
        return {
            "事件数": 0, "有候选周": 0, "区间空窗周": interval_weeks,
            "每个有候选周平均数": np.nan, "涉及股票": 0,
        }
    returns = pd.to_numeric(frame["Primary_Return_pct"], errors="coerce")
    weeks = frame["Signal_Date"].nunique()
    return {
        "事件数": len(frame), "有候选周": weeks,
        "区间空窗周": max(0, interval_weeks - weeks),
        "每个有候选周平均数": len(frame) / weeks if weeks else np.nan,
        "涉及股票": frame["ts_code"].nunique(),
        "持续强上涨(%)": rate(frame["Strong_Sustained"]),
        "弱反弹或失败(%)": rate(frame["Weak_Rebound"]),
        "30%先于止损(%)": rate(frame["Target30_Before_Stop"]),
        "八周止损率(%)": rate(frame["Stop_8W"]),
        "可执行收益均值(%)": returns.mean(),
        "可执行收益中位数(%)": returns.median(),
        "可执行胜率(%)": float(returns.gt(0).mean() * 100.0),
        "可执行盈亏比": profit_factor(returns),
        "前10笔盈利贡献(%)": contribution_ratio(returns, 10),
        "前10只股票盈利贡献(%)": stock_concentration(frame, 10),
        "剔除前10只股票后收益均值(%)": mean_without_top_stocks(frame, 10),
    }


def select_top(frame: pd.DataFrame, score: str, scheme: str, name: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("Signal_Date", sort=True):
        selected = group.sort_values(
            [score, "Board_RS_13W_pct", "ts_code"],
            ascending=[False, False, True], kind="mergesort",
        ).head(TOP_N).copy()
        selected["Weekly_Rank"] = np.arange(1, len(selected) + 1)
        pieces.append(selected)
    if not pieces:
        return pd.DataFrame(columns=[*frame.columns, "Weekly_Rank", "Scheme", "Scheme_Name"])
    result = pd.concat(pieces, ignore_index=True).copy()
    result["Scheme"] = scheme
    result["Scheme_Name"] = name
    result["Ranking_Score"] = result[score]
    return result


def random_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    return [group.index.to_numpy() for _, group in frame.groupby("Signal_Date", sort=True)]


def random_simulation(
    frame: pd.DataFrame, repetitions: int, seed: int, interval_weeks: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    groups = random_groups(frame)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        indexes = np.concatenate([
            rng.choice(group, size=min(TOP_N, len(group)), replace=False) for group in groups
        ])
        rows.append({
            "Random_Repetition": repetition + 1,
            **metrics(frame.loc[indexes], interval_weeks),
        })
    return pd.DataFrame(rows)


def random_row(samples: pd.DataFrame, scheme: str, name: str) -> dict[str, Any]:
    row: dict[str, Any] = {"方案": scheme, "方案名称": name}
    if samples.empty:
        return row
    for column in samples.columns:
        if column == "Random_Repetition" or not pd.api.types.is_numeric_dtype(samples[column]):
            continue
        values = pd.to_numeric(samples[column], errors="coerce")
        row[column] = values.mean()
        row[f"{column}_随机2.5%"] = values.quantile(0.025)
        row[f"{column}_随机97.5%"] = values.quantile(0.975)
    return row


def funnel_report(frame: pd.DataFrame, interval_weeks: int) -> pd.DataFrame:
    stages = [
        ("01 第一根红柱输入", None),
        ("02 行情数据可用", "Data_Available"),
        ("03 真正周线第一红柱核对", "True_First_Red_Audit"),
        ("04 真正周线2—5次波浪", "Wave_2_5_Pass"),
        ("05 周线乖离与上影安全", "Weekly_Safe_Pass"),
        ("06 日线MA60>MA120", "Daily_Trend_Pass"),
        ("07 十日箱体首发", "Box_Breakout_Pass"),
        ("08 收盘高于MA20的2%", "Daily_Breakout_Pass"),
        ("09 MA20向上", "MA20_Healthy_Pass"),
        ("10 量比1.3—3倍", "Volume_Pass"),
        ("11 实体阳线≥60%", "Solid_Yang_Pass"),
        ("12 日线MACD健康", "Daily_MACD_Pass"),
        ("13 非一字板可买", "One_Word_Pass"),
        ("14 T+1开盘−3%至+5%", "Gap_Pass"),
    ]
    current = pd.Series(True, index=frame.index)
    previous_count = len(frame)
    rows: list[dict[str, Any]] = []
    for stage, column in stages:
        if column:
            current &= frame[column].map(to_bool)
        subset = frame[current]
        count = len(subset)
        weeks = subset["Signal_Date"].nunique()
        rows.append({
            "阶段": stage, "剩余事件": count,
            "本层淘汰": previous_count - count,
            "保留率(%)": count / len(frame) * 100.0 if len(frame) else np.nan,
            "有候选周": weeks, "区间空窗周": max(0, interval_weeks - weeks),
            "每个有候选周平均数": count / weeks if weeks else np.nan,
        })
        previous_count = count
    return pd.DataFrame(rows)


def grouped_report(frame: pd.DataFrame, columns: list[str], interval_weeks: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(columns, keys)), **metrics(group, interval_weeks)})
    return pd.DataFrame(rows)


def rank_report(selections: pd.DataFrame, interval_weeks: int) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame()
    return grouped_report(selections, ["Scheme", "Scheme_Name", "Weekly_Rank"], interval_weeks)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def build_result(
    enriched: pd.DataFrame, source: str, repetitions: int,
    fetch_failures: pd.DataFrame, cache_hits: int,
) -> dict[str, Any]:
    interval_weeks = full_week_count(enriched)
    gate_pool = enriched[enriched["V406_Tradable_Pass"].map(to_bool)].copy()
    original = select_top(gate_pool, "Original_V406_Score", "O1", "V40.6原评分前三")
    revised = select_top(gate_pool, "New_Score_100", "N1", "新100分前三")
    selections = pd.concat([original, revised], ignore_index=True, sort=False)

    random_all = random_simulation(enriched, repetitions, RANDOM_SEED, interval_weeks)
    random_gate = random_simulation(gate_pool, repetitions, RANDOM_SEED + 1, interval_weeks)
    rows = [
        {"方案": "A", "方案名称": "全部第一根红柱事件", **metrics(enriched, interval_weeks)},
        random_row(random_all, "R0", "全部第一根红柱内随机前三"),
        {"方案": "G", "方案名称": "通过V40.6全部门槛事件", **metrics(gate_pool, interval_weeks)},
        random_row(random_gate, "RG", "V40.6门槛内随机前三"),
    ]
    for scheme, name, frame in [
        ("O1", "V40.6原评分前三", original), ("N1", "新100分前三", revised),
    ]:
        rows.append({"方案": scheme, "方案名称": name, **metrics(frame, interval_weeks)})
    comparison = pd.DataFrame(rows)

    base_all = comparison[comparison["方案"].eq("R0")].iloc[0]
    base_gate_rows = comparison[comparison["方案"].eq("RG")]
    base_gate = base_gate_rows.iloc[0] if len(base_gate_rows) else base_all
    delta_columns = [
        "持续强上涨(%)", "弱反弹或失败(%)", "30%先于止损(%)", "八周止损率(%)",
        "可执行收益均值(%)", "可执行收益中位数(%)", "可执行胜率(%)",
        "剔除前10只股票后收益均值(%)",
    ]
    for index, row in comparison.iterrows():
        baseline = base_gate if row["方案"] in {"O1", "N1"} else base_all
        comparison.loc[index, "对应随机基准"] = "RG" if row["方案"] in {"O1", "N1"} else "R0"
        for column in delta_columns:
            if column in comparison.columns and column in baseline.index:
                comparison.loc[index, f"较对应随机_{column}"] = row.get(column, np.nan) - baseline.get(column, np.nan)

    funnel = funnel_report(enriched, interval_weeks)
    pool_report = pd.DataFrame([
        {"候选池": "全部第一根红柱", **metrics(enriched, interval_weeks)},
        {"候选池": "真正周线2—5次波浪", **metrics(enriched[enriched["Wave_2_5_Pass"].map(to_bool)], interval_weeks)},
        {"候选池": "V40.6全部门槛且T+1可买", **metrics(gate_pool, interval_weeks)},
    ])
    yearly = grouped_report(selections, ["Signal_Year", "Scheme", "Scheme_Name"], interval_weeks)
    boards = grouped_report(selections, ["Sample_Board", "Scheme", "Scheme_Name"], interval_weeks)
    ranks = rank_report(selections, interval_weeks)
    failure_reasons = (
        enriched.groupby("First_Fail_Stage", dropna=False).size().rename("事件数")
        .reset_index().sort_values("事件数", ascending=False)
    )
    weekly_rows: list[dict[str, Any]] = []
    for (date, scheme), group in selections.groupby(["Signal_Date", "Scheme"], sort=True):
        group = group.sort_values("Weekly_Rank")
        weekly_rows.append({
            "Signal_Date": date, "方案": scheme, "方案名称": group["Scheme_Name"].iloc[0],
            "入选数": len(group), "股票代码": "|".join(group["ts_code"].astype(str)),
            "股票名称": "|".join(group["name"].astype(str)),
            "板块": "|".join(group["Sample_Board"].astype(str)),
            "评分": "|".join(group["Ranking_Score"].round(2).astype(str)),
            "可执行收益均值(%)": group["Primary_Return_pct"].mean(),
        })
    weekly = pd.DataFrame(weekly_rows)
    metadata = pd.DataFrame([
        {"项目": "程序", "值": TITLE}, {"项目": "输入", "值": source},
        {"项目": "生成时间", "值": datetime.now().isoformat(timespec="seconds")},
        {"项目": "输入事件", "值": len(enriched)}, {"项目": "区间周数", "值": interval_weeks},
        {"项目": "通过V40.6门槛事件", "值": len(gate_pool)},
        {"项目": "行情缓存命中股票", "值": cache_hits},
        {"项目": "行情失败股票", "值": len(fetch_failures)},
        {"项目": "随机重复", "值": repetitions}, {"项目": "随机种子", "值": RANDOM_SEED},
        {"项目": "真正周线", "值": "周OHLCV后计算EMA12/EMA26/DEA9，第一红柱=前周≤0且本周>0"},
        {"项目": "波浪门槛", "值": "最近52周最低点后，真正周线绿柱且回撤≥5%的独立阶段2—5次"},
        {"项目": "退出口径", "值": "沿用输入事件的+30%/-10%/最长八周可执行收益"},
        {"项目": "限制", "值": "每周独立前三，尚未处理三仓持仓重叠和资金占用"},
    ])
    files = {
        "01_hybrid_event_audit.csv": csv_bytes(enriched),
        "02_filter_funnel.csv": csv_bytes(funnel),
        "03_candidate_pool_comparison.csv": csv_bytes(pool_report),
        "04_top3_main_comparison.csv": csv_bytes(comparison),
        "05_selected_events.csv": csv_bytes(selections),
        "06_year_stability.csv": csv_bytes(yearly),
        "07_board_stability.csv": csv_bytes(boards),
        "08_rank_performance.csv": csv_bytes(ranks),
        "09_weekly_choices.csv": csv_bytes(weekly),
        "10_random_all_distribution.csv": csv_bytes(random_all),
        "11_random_gate_distribution.csv": csv_bytes(random_gate),
        "12_first_fail_reasons.csv": csv_bytes(failure_reasons),
        "13_fetch_failures.csv": csv_bytes(fetch_failures),
        "14_metadata.csv": csv_bytes(metadata),
    }
    return {
        "enriched": enriched, "gate_pool": gate_pool, "selections": selections,
        "comparison": comparison, "funnel": funnel, "pool_report": pool_report,
        "yearly": yearly, "boards": boards, "ranks": ranks, "weekly": weekly,
        "failure_reasons": failure_reasons, "metadata": metadata,
        "files": files, "zip": make_zip(files), "interval_weeks": interval_weeks,
    }


def show(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("本表没有记录。")
        return
    formats = {
        column: "{:.2f}" for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and any(word in column for word in ("(%)", "平均", "得分", "Score", "比例", "率", "中位数"))
    }
    st.dataframe(frame.style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)


def render(result: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("第一根红柱事件", f"{len(result['enriched']):,}")
    c2.metric("真正周线2—5波浪", f"{int(result['enriched']['Wave_2_5_Pass'].sum()):,}")
    c3.metric("V40.6最终候选", f"{len(result['gate_pool']):,}")
    c4.metric("最终候选周", f"{result['gate_pool']['Signal_Date'].nunique():,}/{result['interval_weeks']:,}")
    st.subheader("过滤漏斗")
    show(result["funnel"])
    st.subheader("候选池变化")
    show(result["pool_report"])
    st.subheader("随机基准、原评分与新评分")
    show(result["comparison"])
    st.subheader("逐年和分板块")
    show(result["yearly"])
    show(result["boards"])
    with st.expander("排名表现、失败原因和每周选择"):
        show(result["ranks"])
        show(result["failure_reasons"])
        show(result["weekly"])

    st.subheader("下载")
    st.download_button(
        "下载全部结果ZIP", result["zip"],
        file_name="weekly_macd_v406_hybrid_v4_6_all_results.zip",
        mime="application/zip", type="primary", key="v46_all", on_click="ignore",
    )
    labels = [
        "1号：事件审计", "2号：过滤漏斗", "3号：候选池比较", "4号：前三主比较",
        "5号：入选事件", "6号：逐年", "7号：分板块", "8号：排名表现",
        "9号：每周选择", "10号：全部随机分布", "11号：门槛内随机分布",
        "12号：首次失败原因", "13号：行情失败", "14号：运行信息",
    ]
    columns = st.columns(4)
    for index, (name, data) in enumerate(result["files"].items()):
        with columns[index % 4]:
            st.download_button(
                labels[index], data, file_name=name, mime="text/csv",
                key=f"v46_{name}", on_click="ignore",
            )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.info(
        "本版使用V4.4/V4.5已经识别的第一根红柱事件，重新下载逐股票日线，"
        "以真正周线MACD核对信号和2—5次波浪，再应用V40.6硬门槛。"
    )
    with st.expander("固定实验口径", expanded=False):
        st.markdown(
            """
            - 周线：先合成周OHLCV，再计算12/26/9周MACD。
            - 波浪：最近52周最低点后，真正周线绿柱且回撤至少5%的独立阶段2—5次。
            - V40.6硬门槛保持：周线风控、日线趋势、10日箱体首发、MA20、量比、实体阳线、日线MACD、T+1开盘。
            - O1继续使用原V40.6线性评分；N1使用新的非线性100分。
            - 退出统一沿用事件文件中的+30%止盈、-10%止损、最长八周，不混入V40.6动态退出。
            """
        )
    with st.form("v46_form"):
        upload = st.file_uploader(
            "上传V4.4/V4.5全部结果ZIP或事件CSV", type=["zip", "csv"]
        )
        token = st.text_input("Tushare Token", type="password")
        c1, c2, c3 = st.columns(3)
        use_cache = c1.checkbox("使用逐股票缓存和断点续跑", value=True)
        pause = c2.number_input("每只股票请求后暂停(秒)", min_value=0.05, max_value=2.0, value=0.12, step=0.05)
        repetitions = c3.number_input("随机前三重复次数", min_value=200, max_value=3000, value=1000, step=200)
        submitted = st.form_submit_button("开始混合验证", type="primary")
    if submitted:
        if upload is None:
            st.error("请上传V4.4/V4.5结果。")
        elif not token.strip():
            st.error("请输入Tushare Token。")
        else:
            try:
                raw, source = load_events(upload)
                validate_events(raw)
                events = prepare_events(raw)
                if events.empty:
                    raise ValueError("没有完整、可交易的第一根红柱事件。")
                ts.set_token(token.strip())
                pro = ts.pro_api(token.strip())
                earliest = datetime.strptime(events["Signal_Date"].min(), "%Y%m%d")
                latest = datetime.strptime(events["Signal_Date"].max(), "%Y%m%d")
                # 与产生V4.1/V4.4事件的程序保持三年预热，避免EMA初始化长度不同
                # 导致靠近零轴的第一根红柱审计出现不必要偏差。
                start = (earliest - timedelta(days=3 * 365)).strftime("%Y%m%d")
                end = (latest + timedelta(days=20)).strftime("%Y%m%d")
                codes = sorted(events["ts_code"].astype(str).unique())
                histories: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
                failures: list[dict[str, str]] = []
                cache_hits = 0
                progress = st.progress(0.0, text="准备逐股票行情...")
                for index, code in enumerate(codes, start=1):
                    daily_raw, hit, error = fetch_daily(
                        pro, code, start, end, bool(use_cache), float(pause)
                    )
                    cache_hits += int(hit)
                    if daily_raw.empty:
                        histories[code] = (pd.DataFrame(), pd.DataFrame())
                        failures.append({"ts_code": code, "错误": error})
                    else:
                        daily = build_daily(daily_raw)
                        histories[code] = (daily, build_weekly(daily))
                    progress.progress(
                        index / len(codes),
                        text=f"行情与真正周线：{index}/{len(codes)}；缓存命中{cache_hits}",
                    )
                progress.empty()
                with st.spinner("正在执行过滤漏斗、新旧评分和双随机基准..."):
                    enriched = enrich_events(events, histories)
                    failure_frame = pd.DataFrame(failures, columns=["ts_code", "错误"])
                    result = build_result(
                        enriched, source, int(repetitions), failure_frame, cache_hits
                    )
                    st.session_state["v46_result"] = result
                st.success(
                    f"完成：{len(enriched):,}个事件，{len(codes):,}只股票，"
                    f"V40.6最终候选{len(result['gate_pool']):,}个。"
                )
            except Exception as exc:
                st.exception(exc)
    if "v46_result" in st.session_state:
        render(st.session_state["v46_result"])
    else:
        st.caption("首次需要逐股票下载；缓存后重复运行会明显加快。下载按钮不会清空结果。")


if __name__ == "__main__":
    main()
