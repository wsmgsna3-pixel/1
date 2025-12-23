# -*- coding: utf-8 -*-
"""
工程级 · 强势股回踩评分回测系统
目标：
- 可观测
- 可调试
- 不黑箱
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 页面
# =====================================================
st.set_page_config(layout="wide")
st.title("🧠 工程级 · 强势股回踩评分回测系统")

# =====================================================
# Token
# =====================================================
TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN:
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# =====================================================
# 参数区
# =====================================================
with st.sidebar:
    st.header("回测区间")
    END_DATE = st.date_input("结束日", datetime.now().date())
    BACKTEST_DAYS = st.number_input("回测天数", 50, 500, 200, 50)

    st.markdown("---")
    st.header("交易参数")
    STOP_LOSS = -3.0
    TAKE_PROFIT = 6.0
    HOLD_DAYS = 3

    st.markdown("---")
    st.header("评分阈值")
    SCORE_THRESHOLD = st.slider("最低评分", 1, 10, 4)

# =====================================================
# 工具函数
# =====================================================
@st.cache_data(ttl=3600)
def get_trade_days(end, n):
    start = (end - timedelta(days=n * 3)).strftime("%Y%m%d")
    cal = pro.trade_cal(start_date=start, end_date=end.strftime("%Y%m%d"))
    return cal[cal["is_open"] == 1].sort_values("cal_date", ascending=False)["cal_date"].head(n).tolist()

@st.cache_data(ttl=3600)
def load_market_data(start, end):
    daily = pro.daily(start_date=start, end_date=end)
    basic = pro.daily_basic(
        start_date=start, end_date=end,
        fields="ts_code,trade_date,turnover_rate,circ_mv"
    )
    df = daily.merge(basic, on=["ts_code","trade_date"], how="left")
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["ts_code","trade_date"])

# =====================================================
# 主程序
# =====================================================
if st.button("🚀 开始工程级回测"):

    trade_days = get_trade_days(END_DATE, BACKTEST_DAYS)
    start_date = (datetime.strptime(trade_days[-1], "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    end_date = trade_days[0]

    st.info("📥 一次性加载全市场数据")
    data = load_market_data(start_date, end_date)

    grouped = data.groupby("ts_code")

    # 统计器
    stats = {
        "L2_股票池": 0,
        "L3_强势股": 0,
        "L4_回踩候选": 0,
        "L5_成交": 0
    }

    trades = []

    for ts_code, df in grouped:
        df = df.reset_index(drop=True)
        if len(df) < 20:
            continue

        for i in range(10, len(df) - HOLD_DAYS):
            today = df.iloc[i]
            if today["trade_date"] not in trade_days:
                continue

            # ===== L2 股票池 =====
            if not (8 <= today["close"] <= 80):
                continue
            if today["turnover_rate"] < 1.5:
                continue
            if not (30 <= today["circ_mv"]/10000 <= 500):
                continue

            stats["L2_股票池"] += 1

            # ===== L3 强势 =====
            ret_5 = today["close"] / df.iloc[i-5]["close"] - 1
            if ret_5 < 0.04:
                continue

            stats["L3_强势股"] += 1

            # ===== L4 回踩评分 =====
            score = 0
            if ret_5 > 0.08: score += 2
            if ret_5 > 0.12: score += 2

            pct = (today["close"] / df.iloc[i-1]["close"] - 1) * 100
            if -5 <= pct <= -1: score += 2

            ma5 = df["close"].iloc[i-5:i].mean()
            if today["low"] >= ma5: score += 2

            vol_ma5 = df["vol"].iloc[i-5:i].mean()
            if today["vol"] <= vol_ma5: score += 2

            if score < SCORE_THRESHOLD:
                continue

            stats["L4_回踩候选"] += 1

            # ===== L5 交易模拟 =====
            buy = df.iloc[i+1]["open"]
            sl = buy * (1 + STOP_LOSS / 100)
            tp = buy * (1 + TAKE_PROFIT / 100)

            exit_ret = None
            for j in range(1, HOLD_DAYS + 1):
                row = df.iloc[i + j]
                if row["low"] <= sl:
                    exit_ret = STOP_LOSS
                    break
                if row["high"] >= tp:
                    exit_ret = TAKE_PROFIT
                    break

            if exit_ret is None:
                exit_ret = (df.iloc[i + HOLD_DAYS]["close"] / buy - 1) * 100

            trades.append(exit_ret)
            stats["L5_成交"] += 1

    # =====================================================
    # 结果展示
    # =====================================================
    st.subheader("📊 各层信号数量")
    st.json(stats)

    if not trades:
        st.warning("⚠️ 有信号但未成交，请继续降低评分阈值或扩大区间")
        st.stop()

    res = pd.Series(trades)
    st.metric("平均收益%", round(res.mean(), 2))
    st.metric("胜率%", round((res > 0).mean() * 100, 1))
    st.metric("交易次数", len(res))
