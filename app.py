# -*- coding: utf-8 -*-
"""
=====================================================
工程级 · 强势股回踩评分 + 回测系统（完整版）
-----------------------------------------------------
设计目标：
1. 结构厚、逻辑清晰
2. 每一层都有“数量反馈”
3. 永不黑箱、永不 0 成交无解释
4. 接近传统 400+ 行交易系统工程结构
=====================================================
"""

# =====================================================
# 0. 基础库
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 1. 页面 & 全局设置
# =====================================================
st.set_page_config(
    page_title="工程级回测系统 · 强势回踩",
    layout="wide"
)

st.title("🧱 工程级 · 强势股回踩评分 + 回测系统")
st.caption("不是最短代码，而是可调、可解释、可验证的完整系统")

# =====================================================
# 2. Token 管理
# =====================================================
TS_TOKEN = st.text_input("请输入 Tushare Token", type="password")
if not TS_TOKEN:
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# =====================================================
# 3. 参数区（完整）
# =====================================================
with st.sidebar:
    st.header("📅 回测区间")
    END_DATE = st.date_input("回测结束日", datetime.now().date())
    BACKTEST_DAYS = st.number_input("回测天数", 50, 600, 200, 50)

    st.markdown("---")
    st.header("📦 股票池过滤")
    MIN_PRICE = st.number_input("最低股价", 3.0, 20.0, 8.0, 1.0)
    MAX_PRICE = st.number_input("最高股价", 30.0, 300.0, 80.0, 5.0)
    MIN_TURNOVER = st.number_input("最低换手率(%)", 0.5, 10.0, 1.5, 0.5)
    MIN_CIRC_MV = st.number_input("最低流通市值(亿)", 10, 200, 30, 5)
    MAX_CIRC_MV = st.number_input("最高流通市值(亿)", 100, 2000, 500, 50)

    st.markdown("---")
    st.header("📈 强势定义")
    RET5_MIN = st.number_input("5日涨幅下限(%)", 2.0, 10.0, 4.0, 1.0)
    RET5_STRONG = st.number_input("5日强势加分(%)", 6.0, 20.0, 8.0, 1.0)
    RET5_SUPER = st.number_input("5日超强加分(%)", 10.0, 30.0, 12.0, 2.0)

    st.markdown("---")
    st.header("📉 回踩定义")
    PULLBACK_MIN = st.number_input("回调下限(%)", -8.0, -1.0, -5.0, 0.5)
    PULLBACK_MAX = st.number_input("回调上限(%)", -3.0, -0.2, -1.0, 0.2)

    st.markdown("---")
    st.header("⭐ 评分与交易")
    SCORE_THRESHOLD = st.slider("最低评分阈值", 1, 10, 4)
    HOLD_DAYS = st.number_input("最大持有天数", 1, 5, 3)
    STOP_LOSS = st.number_input("止损(%)", -10.0, -1.0, -3.0, 0.5)
    TAKE_PROFIT = st.number_input("止盈(%)", 2.0, 15.0, 6.0, 1.0)

    st.markdown("---")
    DEBUG_MODE = st.checkbox("开启调试输出", value=True)

# =====================================================
# 4. 工具函数：交易日
# =====================================================
@st.cache_data(ttl=3600)
def get_trade_days(end_date, n_days):
    start = (end_date - timedelta(days=n_days * 3)).strftime("%Y%m%d")
    cal = pro.trade_cal(
        start_date=start,
        end_date=end_date.strftime("%Y%m%d")
    )
    return (
        cal[cal["is_open"] == 1]
        .sort_values("cal_date", ascending=False)["cal_date"]
        .head(n_days)
        .tolist()
    )

# =====================================================
# 5. 工具函数：加载市场数据
# =====================================================
@st.cache_data(ttl=3600)
def load_market_data(start_date, end_date):
    """
    一次性加载：
    - daily
    - daily_basic
    """
    daily = pro.daily(start_date=start_date, end_date=end_date)
    basic = pro.daily_basic(
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,turnover_rate,circ_mv"
    )

    df = daily.merge(basic, on=["ts_code", "trade_date"], how="left")

    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values(["ts_code", "trade_date"])

# =====================================================
# 6. 评分函数（独立模块）
# =====================================================
def score_stock(df, idx):
    """
    给单一股票、单一交易日打分
    """
    score = 0

    # ---- 强势评分 ----
    ret5 = df.iloc[idx]["close"] / df.iloc[idx - 5]["close"] - 1
    if ret5 >= RET5_MIN / 100:
        score += 1
    if ret5 >= RET5_STRONG / 100:
        score += 2
    if ret5 >= RET5_SUPER / 100:
        score += 2

    # ---- 回踩评分 ----
    pct = (df.iloc[idx]["close"] / df.iloc[idx - 1]["close"] - 1) * 100
    if PULLBACK_MIN <= pct <= PULLBACK_MAX:
        score += 2

    ma5 = df["close"].iloc[idx - 5:idx].mean()
    if df.iloc[idx]["low"] >= ma5:
        score += 2

    vol_ma5 = df["vol"].iloc[idx - 5:idx].mean()
    if df.iloc[idx]["vol"] <= vol_ma5:
        score += 2

    return score, ret5 * 100, pct

# =====================================================
# 7. 主回测流程
# =====================================================
if st.button("🚀 开始工程级回测"):

    # ---------- 7.1 准备交易日 ----------
    trade_days = get_trade_days(END_DATE, BACKTEST_DAYS)
    if not trade_days:
        st.error("无法获取交易日")
        st.stop()

    start_date = (
        datetime.strptime(trade_days[-1], "%Y%m%d")
        - timedelta(days=40)
    ).strftime("%Y%m%d")
    end_date = trade_days[0]

    st.info("📥 正在加载全市场数据（一次性）")
    data = load_market_data(start_date, end_date)
    grouped = data.groupby("ts_code")

    # ---------- 7.2 分层统计 ----------
    stats = {
        "L1_总样本": 0,
        "L2_股票池": 0,
        "L3_强势": 0,
        "L4_评分达标": 0,
        "L5_成交": 0,
    }

    trades = []

    # ---------- 7.3 主循环 ----------
    for ts_code, df in grouped:
        df = df.reset_index(drop=True)
        if len(df) < 15:
            continue

        for i in range(10, len(df) - HOLD_DAYS):
            today = df.iloc[i]
            if today["trade_date"] not in trade_days:
                continue

            stats["L1_总样本"] += 1

            # ===== L2 股票池 =====
            if not (MIN_PRICE <= today["close"] <= MAX_PRICE):
                continue
            if today["turnover_rate"] < MIN_TURNOVER:
                continue
            circ_mv_billion = today["circ_mv"] / 10000
            if not (MIN_CIRC_MV <= circ_mv_billion <= MAX_CIRC_MV):
                continue

            stats["L2_股票池"] += 1

            # ===== L3 强势 =====
            ret5 = today["close"] / df.iloc[i - 5]["close"] - 1
            if ret5 < RET5_MIN / 100:
                continue

            stats["L3_强势"] += 1

            # ===== L4 评分 =====
            score, ret5_pct, pullback_pct = score_stock(df, i)
            if score < SCORE_THRESHOLD:
                continue

            stats["L4_评分达标"] += 1

            # ===== L5 交易模拟 =====
            buy_price = df.iloc[i + 1]["open"]
            sl = buy_price * (1 + STOP_LOSS / 100)
            tp = buy_price * (1 + TAKE_PROFIT / 100)

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
                exit_ret = (df.iloc[i + HOLD_DAYS]["close"] / buy_price - 1) * 100

            trades.append({
                "交易日": today["trade_date"],
                "股票": ts_code,
                "评分": score,
                "5日涨幅%": round(ret5_pct, 2),
                "回调%": round(pullback_pct, 2),
                "收益%": round(exit_ret, 2),
            })

            stats["L5_成交"] += 1

    # =================================================
    # 8. 结果展示
    # =================================================
    st.subheader("📊 分层统计（关键调参依据）")
    st.json(stats)

    if not trades:
        st.warning("⚠️ 本次回测未产生任何成交，请降低评分阈值或放宽条件")
        st.stop()

    res = pd.DataFrame(trades)

    st.subheader("📈 回测结果")
    st.metric("平均收益%", round(res["收益%"].mean(), 2))
    st.metric("胜率%", round((res["收益%"] > 0).mean() * 100, 1))
    st.metric("交易次数", len(res))

    st.dataframe(
        res.sort_values("交易日", ascending=False),
        use_container_width=True
    )
