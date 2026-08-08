"""周线 MACD 第一根红柱信号审计器 V1.0。

用途：验证股票软件肉眼看到的“绿柱翻红”是否被代码识别，并逐项说明
旧 V3/V5 管道为什么删除某个事件。本程序不做机器学习，也不把价格、市值、
趋势、回调作为原始事件过滤条件。
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts


APP_VERSION = "1.0"
CACHE_DIR = Path(".macd_signal_audit_cache")
CACHE_DIR.mkdir(exist_ok=True)
WARMUP_WEEKS = 80
FOCUS_CODES = ["600552.SH", "301251.SZ", "688036.SH"]


def norm_code(value: object) -> str:
    text = str(value).strip().upper()
    if text.endswith((".SH", ".SZ", ".BJ")):
        return text
    digits = "".join(ch for ch in text if ch.isdigit()).zfill(6)
    if digits.startswith(("5", "6", "9")):
        return f"{digits}.SH"
    if digits.startswith(("0", "1", "2", "3")):
        return f"{digits}.SZ"
    if digits.startswith(("4", "8")):
        return f"{digits}.BJ"
    return text


def board_from_code(code: str) -> str:
    digits = code.split(".")[0]
    if digits.startswith(("300", "301")):
        return "创业板"
    if digits.startswith(("688", "689")):
        return "科创板"
    return "主板"


def cache_path(code: str, start: str, end: str) -> Path:
    key = hashlib.sha256(f"{code}|{start}|{end}|audit-v1".encode()).hexdigest()[:20]
    return CACHE_DIR / f"{key}.pkl"


def atomic_pickle(obj: object, path: Path) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def fetch_history(
    pro, code: str, start: str, end: str, use_cache: bool, pause: float
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    path = cache_path(code, start, end)
    if use_cache and path.exists():
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            return payload["daily"], payload["basic"], "缓存命中"
        except Exception:
            pass

    try:
        daily = ts.pro_bar(
            ts_code=code, start_date=start, end_date=end,
            adj="qfq", freq="D",
        )
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"日线下载失败：{exc}"
    time.sleep(pause)
    try:
        basic = pro.daily_basic(
            ts_code=code, start_date=start, end_date=end,
            fields="ts_code,trade_date,close,circ_mv,turnover_rate",
        )
    except Exception as exc:
        basic = pd.DataFrame()
        basic.attrs["error"] = str(exc)
    time.sleep(pause)

    if daily is None or daily.empty:
        return pd.DataFrame(), basic, "没有日线数据"
    daily = daily.copy()
    daily["trade_date"] = daily["trade_date"].astype(str)
    for col in ["open", "high", "low", "close", "vol"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = (daily.dropna(subset=["trade_date", "open", "high", "low", "close"])
             .drop_duplicates("trade_date", keep="last")
             .sort_values("trade_date").reset_index(drop=True))

    if basic is None:
        basic = pd.DataFrame()
    if not basic.empty:
        basic = basic.copy()
        basic["trade_date"] = basic["trade_date"].astype(str)
        for col in ["close", "circ_mv", "turnover_rate"]:
            basic[col] = pd.to_numeric(basic[col], errors="coerce")
        basic = (basic.drop_duplicates("trade_date", keep="last")
                 .sort_values("trade_date").reset_index(drop=True))
    if use_cache:
        atomic_pickle({"daily": daily, "basic": basic}, path)
    return daily, basic, "下载成功"


def load_calendar(pro, start: str, end: str) -> list[str]:
    cal = pro.trade_cal(
        exchange="SSE", start_date=start, end_date=end,
        fields="cal_date,is_open",
    )
    cal = cal[pd.to_numeric(cal["is_open"], errors="coerce").eq(1)].copy()
    return sorted(cal["cal_date"].astype(str).tolist())


def complete_week_map(open_dates: list[str]) -> dict[pd.Timestamp, str]:
    frame = pd.DataFrame({"trade_date": open_dates})
    frame["dt"] = pd.to_datetime(frame["trade_date"])
    frame["week_label"] = frame["dt"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    return frame.groupby("week_label")["trade_date"].max().to_dict()


def build_weekly(daily: pd.DataFrame, week_map: dict[pd.Timestamp, str]) -> pd.DataFrame:
    work = daily.copy()
    work["dt"] = pd.to_datetime(work["trade_date"])
    weekly = (work.set_index("dt").resample("W-FRI").agg({
        "trade_date": "last", "open": "first", "high": "max",
        "low": "min", "close": "last", "vol": "sum",
    }).dropna(subset=["close"]).reset_index().rename(columns={"dt": "week_label"}))
    weekly["calendar_week_last"] = weekly["week_label"].map(week_map)
    weekly = weekly[
        weekly["calendar_week_last"].notna()
        & weekly["trade_date"].astype(str).eq(weekly["calendar_week_last"].astype(str))
    ].copy().reset_index(drop=True)
    weekly["EMA12"] = weekly["close"].ewm(span=12, adjust=False).mean()
    weekly["EMA26"] = weekly["close"].ewm(span=26, adjust=False).mean()
    weekly["DIF"] = weekly["EMA12"] - weekly["EMA26"]
    weekly["DEA"] = weekly["DIF"].ewm(span=9, adjust=False).mean()
    # 与截图软件一致：MACD = 2 * (DIF - DEA)
    weekly["MACD"] = 2.0 * (weekly["DIF"] - weekly["DEA"])
    weekly["MA20"] = weekly["close"].rolling(20).mean()
    weekly["MA40"] = weekly["close"].rolling(40).mean()
    return weekly


def lookup_basic(basic: pd.DataFrame, signal_date: str) -> tuple[float, float, float]:
    if basic.empty:
        return np.nan, np.nan, np.nan
    row = basic[basic["trade_date"].eq(signal_date)]
    if row.empty:
        return np.nan, np.nan, np.nan
    row = row.iloc[-1]
    raw_close = float(row["close"]) if pd.notna(row.get("close")) else np.nan
    circ = float(row["circ_mv"]) / 10000.0 if pd.notna(row.get("circ_mv")) else np.nan
    turnover = float(row["turnover_rate"]) if pd.notna(row.get("turnover_rate")) else np.nan
    return raw_close, circ, turnover


def old_filter_reason(raw_close: float, circ: float,
                      min_price: float, min_mv: float, max_mv: float) -> str:
    reasons = []
    if not np.isfinite(raw_close):
        reasons.append("缺少信号日原始收盘价")
    elif raw_close < min_price:
        reasons.append(f"股价{raw_close:.2f}<最低{min_price:.2f}")
    if not np.isfinite(circ):
        reasons.append("缺少历史流通市值")
    elif circ < min_mv:
        reasons.append(f"流通市值{circ:.2f}亿<最低{min_mv:.2f}亿")
    elif circ > max_mv:
        reasons.append(f"流通市值{circ:.2f}亿>最高{max_mv:.2f}亿")
    return "；".join(reasons) if reasons else "通过旧价格市值条件"


def future_path(daily: pd.DataFrame, signal_date: str, weeks: int = 8) -> dict:
    future = daily[daily["trade_date"] > signal_date].head(weeks * 5).copy()
    if future.empty:
        return {
            "Entry_Date": "", "Entry_Open": np.nan, "Future_Days": 0,
            "Has_8W_Future": False, "MFE_8W_pct": np.nan,
            "MAE_8W_pct": np.nan, "End_8W_Return_pct": np.nan,
            "Hit_30pct_Within_8W": False,
        }
    entry = float(future.iloc[0]["open"])
    mfe = (float(future["high"].max()) / entry - 1.0) * 100.0
    mae = (float(future["low"].min()) / entry - 1.0) * 100.0
    end_ret = (float(future.iloc[-1]["close"]) / entry - 1.0) * 100.0
    return {
        "Entry_Date": str(future.iloc[0]["trade_date"]),
        "Entry_Open": entry, "Future_Days": int(len(future)),
        "Has_8W_Future": bool(len(future) >= 35),
        "MFE_8W_pct": mfe, "MAE_8W_pct": mae,
        "End_8W_Return_pct": end_ret,
        "Hit_30pct_Within_8W": bool(mfe >= 30.0),
    }


def cycle_stats(weekly: pd.DataFrame, position: int) -> dict:
    end = position
    while end + 1 < len(weekly) and float(weekly.iloc[end + 1]["MACD"]) > 0:
        end += 1
    cycle = weekly.iloc[position:end + 1]
    start_close = float(weekly.iloc[position]["close"])
    peak_macd = float(cycle["MACD"].max())
    prev_green = []
    k = position - 1
    while k >= 0 and float(weekly.iloc[k]["MACD"]) <= 0:
        prev_green.append(abs(float(weekly.iloc[k]["MACD"])))
        k -= 1
    green_peak = max(prev_green) if prev_green else np.nan
    strength = peak_macd / green_peak if np.isfinite(green_peak) and green_peak > 0 else np.nan
    weeks = int(len(cycle))
    return {
        "Red_Cycle_Weeks": weeks,
        "Red_Cycle_End": str(cycle.iloc[-1]["trade_date"]),
        "Cycle_Max_Return_pct": (float(cycle["high"].max()) / start_close - 1.0) * 100.0,
        "Red_Peak_MACD": peak_macd,
        "Previous_Green_Peak_Abs": green_peak,
        "Red_Green_Strength_Ratio": strength,
        "PostKnown_Weak_Rebound": bool(weeks <= 5 and np.isfinite(strength) and strength < 0.5),
        "Cycle_Complete": bool(end + 1 < len(weekly)),
    }


def audit_stock(stock: pd.Series, daily: pd.DataFrame, basic: pd.DataFrame,
                week_map: dict[pd.Timestamp, str], signal_start: str, signal_end: str,
                min_price: float, min_mv: float, max_mv: float) -> tuple[pd.DataFrame, dict]:
    code = norm_code(stock["ts_code"])
    weekly = build_weekly(daily, week_map)
    records = []
    if len(weekly) >= 3:
        start_pos = min(WARMUP_WEEKS, len(weekly) - 1)
        for pos in range(max(1, start_pos), len(weekly)):
            hist = float(weekly.iloc[pos]["MACD"])
            prev = float(weekly.iloc[pos - 1]["MACD"])
            signal_date = str(weekly.iloc[pos]["trade_date"])
            if not (hist > 0 and prev <= 0):
                continue
            if not (signal_start <= signal_date <= signal_end):
                continue
            raw_close, circ, turnover = lookup_basic(basic, signal_date)
            reason = old_filter_reason(raw_close, circ, min_price, min_mv, max_mv)
            record = {
                "ts_code": code,
                "name": stock.get("name", ""),
                "Sample_Board": stock.get("Sample_Board", board_from_code(code)),
                "Signal_Date": signal_date,
                "Weekly_Close_QFQ": float(weekly.iloc[pos]["close"]),
                "MACD": hist, "MACD_Prev": prev,
                "DIF": float(weekly.iloc[pos]["DIF"]),
                "DEA": float(weekly.iloc[pos]["DEA"]),
                "DIF_DEA_Above_Zero": bool(
                    weekly.iloc[pos]["DIF"] > 0 and weekly.iloc[pos]["DEA"] > 0
                ),
                "Raw_Close": raw_close,
                "Circ_MV_Billion": circ,
                "Turnover_Rate": turnover,
                "Old_Filter_Reason": reason,
                "Would_Old_V3_Record": reason == "通过旧价格市值条件",
                **future_path(daily, signal_date),
                **cycle_stats(weekly, pos),
            }
            records.append(record)
    event_frame = pd.DataFrame(records)
    info = {
        "ts_code": code, "name": stock.get("name", ""),
        "Sample_Board": stock.get("Sample_Board", board_from_code(code)),
        "Daily_Rows": len(daily), "Weekly_Rows": len(weekly),
        "Warmup_OK": len(weekly) >= WARMUP_WEEKS,
        "Raw_First_Red_Events": len(event_frame),
        "Old_V3_Passed_Events": int(event_frame.get("Would_Old_V3_Record", pd.Series(dtype=bool)).sum()),
        "Old_V3_Deleted_Events": int(len(event_frame) - event_frame.get("Would_Old_V3_Record", pd.Series(dtype=bool)).sum()),
    }
    return event_frame, info


def read_sample(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame({
            "ts_code": FOCUS_CODES,
            "name": ["凯盛科技", "威尔高", "传音控股"],
            "Sample_Board": ["主板", "创业板", "科创板"],
        })
    frame = pd.read_csv(upload, dtype=str)
    code_col = next((c for c in ["ts_code", "code", "symbol"] if c in frame.columns), None)
    if code_col is None:
        raise ValueError("样本CSV必须包含 ts_code、code 或 symbol 列")
    out = pd.DataFrame({"ts_code": frame[code_col].map(norm_code)})
    out["name"] = frame["name"] if "name" in frame else ""
    out["Sample_Board"] = frame["Sample_Board"] if "Sample_Board" in frame else out["ts_code"].map(board_from_code)
    return out.drop_duplicates("ts_code").reset_index(drop=True)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(outputs: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, frame in outputs.items():
            zf.writestr(filename, csv_bytes(frame))
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="周线MACD信号审计器 V1.0", layout="wide")
    st.title("周线MACD第一根红柱信号审计器 V1.0")
    st.caption("原始翻红一律保留；价格、市值等旧规则只标记删除原因，不执行删除。")

    with st.sidebar:
        st.header("审计范围")
        signal_start_date = st.date_input("信号开始日期", date(2024, 4, 1))
        signal_end_date = st.date_input("信号截止日期", date(2026, 8, 7))
        market_end_date = st.date_input("行情截止日期", date(2026, 8, 7))
        mode = st.radio("运行范围", ["只检查三只重点股票", "检查上传样本全部股票"])
        uploaded = st.file_uploader("600只样本CSV（全量模式需要）", type=["csv"])
        st.header("复现旧版删除条件")
        min_price = st.number_input("旧版最低股价（元）", 0.0, value=20.0, step=1.0)
        min_mv = st.number_input("旧版最小流通市值（亿元）", 0.0, value=200.0, step=50.0)
        max_mv = st.number_input("旧版最大流通市值（亿元）", 0.0, value=1000.0, step=50.0)
        st.header("数据")
        use_cache = st.checkbox("使用缓存", True)
        pause = st.number_input("API调用间隔（秒）", 0.0, 3.0, 0.12, 0.05)

    token = st.text_input("Tushare Token", type="password")
    if not token:
        st.info("输入Token后运行。首次建议只检查三只重点股票，通常几分钟内完成。")
        return
    if not st.button("开始审计", type="primary"):
        st.warning("本工具不会训练模型，也不会用趋势、回调、价格或市值删除原始第一根红柱。")
        return

    if market_end_date < signal_end_date:
        st.error("行情截止日期不能早于信号截止日期")
        return
    try:
        sample = read_sample(uploaded)
        if mode == "只检查三只重点股票":
            sample = sample[sample["ts_code"].isin(FOCUS_CODES)]
            if len(sample) < 3:
                sample = read_sample(None)
    except Exception as exc:
        st.error(str(exc))
        return

    ts.set_token(token)
    pro = ts.pro_api()
    signal_start = signal_start_date.strftime("%Y%m%d")
    signal_end = signal_end_date.strftime("%Y%m%d")
    market_end = market_end_date.strftime("%Y%m%d")
    preload_start = (signal_start_date - timedelta(days=4 * 365)).strftime("%Y%m%d")
    try:
        open_dates = load_calendar(pro, preload_start, market_end)
    except Exception as exc:
        st.error(f"交易日历下载失败：{exc}")
        return
    week_map = complete_week_map(open_dates)

    all_events, stock_rows, failures = [], [], []
    progress = st.progress(0.0)
    for idx, stock in sample.iterrows():
        code = norm_code(stock["ts_code"])
        progress.progress((idx + 1) / len(sample), text=f"{idx + 1}/{len(sample)} {code}")
        daily, basic, status = fetch_history(
            pro, code, preload_start, market_end, use_cache, float(pause)
        )
        if daily.empty:
            failures.append({"ts_code": code, "name": stock.get("name", ""), "Reason": status})
            continue
        events, info = audit_stock(
            stock, daily, basic, week_map, signal_start, signal_end,
            float(min_price), float(min_mv), float(max_mv),
        )
        info["Fetch_Status"] = status
        stock_rows.append(info)
        if not events.empty:
            all_events.append(events)
    progress.empty()

    event_columns = [
        "ts_code", "name", "Sample_Board", "Signal_Date", "Weekly_Close_QFQ",
        "MACD", "MACD_Prev", "DIF", "DEA", "DIF_DEA_Above_Zero",
        "Raw_Close", "Circ_MV_Billion", "Turnover_Rate",
        "Old_Filter_Reason", "Would_Old_V3_Record", "Entry_Date", "Entry_Open",
        "Future_Days", "Has_8W_Future", "MFE_8W_pct", "MAE_8W_pct",
        "End_8W_Return_pct", "Hit_30pct_Within_8W", "Red_Cycle_Weeks",
        "Red_Cycle_End", "Cycle_Max_Return_pct", "Red_Peak_MACD",
        "Previous_Green_Peak_Abs", "Red_Green_Strength_Ratio",
        "PostKnown_Weak_Rebound", "Cycle_Complete",
    ]
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(columns=event_columns)
    stocks = pd.DataFrame(stock_rows)
    failure_frame = pd.DataFrame(failures, columns=["ts_code", "name", "Reason"])
    if not events.empty:
        events = events.sort_values(["Signal_Date", "ts_code"]).reset_index(drop=True)
        weekly_counts = events.groupby("Signal_Date").agg(
            Raw_First_Red=("ts_code", "size"),
            Old_V3_Passed=("Would_Old_V3_Record", "sum"),
            Stocks=("ts_code", lambda x: "、".join(x)),
        ).reset_index()
        reason_counts = (events.assign(
            Delete_Category=np.where(events["Would_Old_V3_Record"], "通过", events["Old_Filter_Reason"])
        ).groupby("Delete_Category").size().rename("Events").reset_index())
    else:
        weekly_counts = pd.DataFrame(columns=["Signal_Date", "Raw_First_Red", "Old_V3_Passed", "Stocks"])
        reason_counts = pd.DataFrame(columns=["Delete_Category", "Events"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("原始第一根红柱", len(events))
    c2.metric("旧规则会保留", int(events.get("Would_Old_V3_Record", pd.Series(dtype=bool)).sum()))
    c3.metric("旧规则会删除", int((~events.get("Would_Old_V3_Record", pd.Series(dtype=bool))).sum()))
    c4.metric("八周内最高涨幅≥30%", int(events.get("Hit_30pct_Within_8W", pd.Series(dtype=bool)).sum()))

    st.subheader("① 三只股票/样本逐股审计")
    st.dataframe(stocks, use_container_width=True, hide_index=True)
    st.subheader("② 所有原始第一根红柱及旧版删除原因")
    st.dataframe(events, use_container_width=True, hide_index=True)
    st.subheader("③ 删除原因汇总")
    st.dataframe(reason_counts, use_container_width=True, hide_index=True)
    if not failure_frame.empty:
        st.subheader("④ 行情下载失败")
        st.dataframe(failure_frame, use_container_width=True, hide_index=True)

    outputs = {
        "01_raw_first_red_events.csv": events,
        "02_stock_audit.csv": stocks,
        "03_weekly_signal_counts.csv": weekly_counts,
        "04_old_filter_reasons.csv": reason_counts,
        "05_data_failures.csv": failure_frame,
    }
    st.download_button(
        "下载 1号：全部审计结果ZIP",
        data=make_zip(outputs),
        file_name="weekly_macd_signal_audit_all_results.zip",
        mime="application/zip",
        type="primary",
    )


if __name__ == "__main__":
    main()
