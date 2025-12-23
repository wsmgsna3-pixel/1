# -*- coding: utf-8 -*-
"""
强势股评分 + 回测 · 稳定工程版
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# =========================
# 页面
# =========================
st.set_page_config(layout="wide")
st.title("📊 强势股评分 + 回测 · 稳定工程版")

# =========================
# Token
# =========================
TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN:
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# =========================
# 参数
# =========================
with st.sidebar:
    END_DATE = st.date_input("回测结束日", datetime.now().date())
    BACKTEST_DAYS = st.number_input("回测天数", 50, 500, 200, 50)

    STOP_LOSS = -3.0
    TAKE_PROFIT = 6.0
    HOLD_DAYS = 3
    SCORE_THRESHOLD = 5

# =========================
# 工具
# =========================
@st.cache_data(ttl=3600)
def get_trade_days(end, n):
    start = (end - timedelta(days=n * 3)).strftime("%Y%m%d")
    cal = pro.trade_cal(start_date=start, end_date=end.strftime("%Y%m%d"))
    return cal[cal["is_open"] == 1].sort_values("cal_date", ascending=False)["cal_date"].head(n).tolist()

@st.cache_data(ttl=3600)
def load_data(start, end):
    daily = pro.daily(start_date=start, end_date=end)
    basic = pro.daily_basic(start_date=start, end_date=end,
                            fields="ts_code,trade_date,turnover_rate,circ_mv")
    df = daily.merge(basic, on=["ts_code","trade_date"], how="left")
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["ts_code","trade_date"])

# =========================
# 主逻辑
# =========================
if st.button("🚀 开始回测"):

    trade_days = get_trade_days(END_DATE, BACKTEST_DAYS)
    start_date = (datetime.strptime(trade_days[-1], "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
    end_date = trade_days[0]

    data = load_data(start_date, end_date)
    grouped = data.groupby("ts_code")

    results = []
    candidates_count = 0

    for ts_code, df in grouped:
        df = df.reset_index(drop=True)
        if len(df) < 15:
            continue

        for i in range(10, len(df) - HOLD_DAYS):
            today = df.iloc[i]
            if today["trade_date"] not in trade_days:
                continue

            # 基础过滤
            if not (8 <= today["close"] <= 80):
                continue
            if today["turnover_rate"] < 1.5:
                continue
            if not (30 <= today["circ_mv"]/10000 <= 500):
                continue

            # ===== 评分 =====
            score = 0
            ret_5 = today["close"] / df.iloc[i-5]["close"] - 1
            if ret_5 > 0.04:
                score += 2
            if ret_5 > 0.08:
                score += 2
            if ret_5 > 0.12:
                score += 3

            pct = (today["close"] / df.iloc[i-1]["close"] - 1) * 100
            if -4 <= pct <= -1:
                score += 2

            ma5 = df["close"].iloc[i-5:i].mean()
            if today["low"] >= ma5:
                score += 2

            vol_ma5 = df["vol"].iloc[i-5:i].mean()
            if today["vol"] <= vol_ma5:
                score += 2

            if score < SCORE_THRESHOLD:
                continue

            candidates_count += 1

            # ===== 回测 =====
            buy = df.iloc[i+1]["open"]
            sl = buy * (1 + STOP_LOSS / 100)
            tp = buy * (1 + TAKE_PROFIT / 100)

            exit_ret = None
            for j in range(1, HOLD_DAYS+1):
                row = df.iloc[i+j]
                if row["low"] <= sl:
                    exit_ret = STOP_LOSS
                    break
                if row["high"] >= tp:
                    exit_ret = TAKE_PROFIT
                    break

            if exit_ret is None:
                exit_ret = (df.iloc[i+HOLD_DAYS]["close"] / buy - 1) * 100

            results.append(exit_ret)

    # =========================
    # 结果
    # =========================
    st.metric("候选信号数量", candidates_count)

    if not results:
        st.warning("⚠️ 没有产生任何成交，请降低评分阈值或拉长回测区间")
        st.stop()

    res = pd.Series(results)
    st.metric("平均收益%", round(res.mean(), 2))
    st.metric("胜率%", round((res > 0).mean() * 100, 1))
    st.metric("交易次数", len(res))
