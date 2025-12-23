# -*- coding: utf-8 -*-
"""
回踩强势股 · 稳定极速回测版
特点：
- 一次拉取全市场数据
- 无交易也不报错
- 适合新手长期反复回测
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
st.title("🛡 回踩强势股 · 稳定极速回测版")

# =====================================================
# Token
# =====================================================
TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN:
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# =====================================================
# 参数
# =====================================================
with st.sidebar:
    st.header("回测参数")
    END_DATE = st.date_input("回测结束日", datetime.now().date())
    BACKTEST_DAYS = st.number_input("回测天数", 20, 300, 100, 10)

    st.markdown("---")
    st.header("交易规则")
    STOP_LOSS = -3.0
    TAKE_PROFIT = 6.0
    HOLD_DAYS = 3

# =====================================================
# 工具函数
# =====================================================
@st.cache_data(ttl=3600)
def get_trade_days(end, n):
    start = (end - timedelta(days=n * 3)).strftime("%Y%m%d")
    cal = pro.trade_cal(start_date=start, end_date=end.strftime("%Y%m%d"))
    return cal[cal["is_open"] == 1].sort_values("cal_date", ascending=False)["cal_date"].head(n).tolist()

@st.cache_data(ttl=3600)
def load_daily(start, end):
    df = pro.daily(start_date=start, end_date=end)
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["ts_code","trade_date"])

@st.cache_data(ttl=3600)
def load_basic(start, end):
    return pro.daily_basic(
        start_date=start,
        end_date=end,
        fields="ts_code,trade_date,turnover_rate,circ_mv"
    )

# =====================================================
# 主回测
# =====================================================
if st.button("🚀 开始稳定极速回测"):

    trade_days = get_trade_days(END_DATE, BACKTEST_DAYS)

    if not trade_days:
        st.error("无法获取交易日")
        st.stop()

    start_date = (datetime.strptime(trade_days[-1], "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
    end_date = trade_days[0]

    st.info("📥 正在加载全市场历史数据（仅一次）")
    daily = load_daily(start_date, end_date)
    basic = load_basic(start_date, end_date)

    data = daily.merge(basic, on=["ts_code","trade_date"], how="left")

    results = []
    grouped = data.groupby("ts_code")

    bar = st.progress(0)
    total = len(grouped)

    for i, (ts_code, df) in enumerate(grouped):
        df = df.reset_index(drop=True)

        if len(df) < 15:
            bar.progress(i / total)
            continue

        for idx in range(10, len(df) - HOLD_DAYS):
            today = df.iloc[idx]
            trade_date = today["trade_date"]

            if trade_date not in trade_days:
                continue

            # ===== 静态过滤 =====
            if not (8 <= today["close"] <= 80):
                continue
            if today["turnover_rate"] < 2:
                continue
            if not (30 <= today["circ_mv"] / 10000 <= 500):
                continue

            # ===== 强势 =====
            ret_5 = today["close"] / df.iloc[idx-5]["close"] - 1
            if not 0.06 <= ret_5 <= 0.25:
                continue

            # ===== 回踩 =====
            pct = (today["close"] / df.iloc[idx-1]["close"] - 1) * 100
            ma5 = df["close"].iloc[idx-5:idx].mean()
            vol_ma5 = df["vol"].iloc[idx-5:idx].mean()

            if not (-3 <= pct <= -0.5):
                continue
            if today["vol"] > vol_ma5:
                continue
            if today["low"] < ma5:
                continue

            # ===== 模拟交易 =====
            buy = df.iloc[idx+1]["open"]
            sl = buy * (1 + STOP_LOSS / 100)
            tp = buy * (1 + TAKE_PROFIT / 100)

            exit_ret = None
            for j in range(1, HOLD_DAYS + 1):
                row = df.iloc[idx + j]
                if row["low"] <= sl:
                    exit_ret = STOP_LOSS
                    break
                if row["high"] >= tp:
                    exit_ret = TAKE_PROFIT
                    break

            if exit_ret is None:
                close_p = df.iloc[idx + HOLD_DAYS]["close"]
                exit_ret = (close_p / buy - 1) * 100

            results.append({
                "交易日": trade_date,
                "股票": ts_code,
                "收益%": round(exit_ret, 2)
            })

        bar.progress(i / total)

    bar.empty()

    # =========================
    # 结果展示（关键稳健点）
    # =========================
    if not results:
        st.warning("⚠️ 回测区间内未产生任何交易（策略条件较严格，这是正常现象）")
        st.stop()

    res = pd.DataFrame(results)

    st.header("📊 回测结果")
    st.metric("平均收益%", round(res["收益%"].mean(), 2))
    st.metric("胜率%", round((res["收益%"] > 0).mean() * 100, 1))
    st.metric("交易次数", len(res))

    st.dataframe(res.sort_values("交易日", ascending=False), use_container_width=True)
