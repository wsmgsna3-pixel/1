# -*- coding: utf-8 -*-
"""
回踩强势股 · 新手稳定实用版
目标：胜率 > 收益率，不追涨，不赌博
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
st.set_page_config(page_title="回踩强势股 · 稳定版", layout="wide")
st.title("📈 回踩强势股 · 新手稳定实用版")
st.markdown("""
**策略说明：**
- 只做强势股回踩
- 不追高
- 强制止损止盈
- 持有不超过 3 天
""")

# =====================================================
# Token
# =====================================================
TS_TOKEN = st.text_input("请输入 Tushare Token", type="password")
if not TS_TOKEN:
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# =====================================================
# 回测参数（少而稳）
# =====================================================
with st.sidebar:
    st.header("回测参数")
    END_DATE = st.date_input("回测结束日期", datetime.now().date())
    BACKTEST_DAYS = st.number_input("回测天数", 20, 200, 50, 5)

    st.markdown("---")
    st.header("交易参数")
    STOP_LOSS = -3.0
    TAKE_PROFIT = 6.0
    HOLD_DAYS = 3

# =====================================================
# 工具函数
# =====================================================
@st.cache_data(ttl=3600)
def safe_get(func, **kwargs):
    try:
        df = getattr(pro, func)(**kwargs)
        return df if df is not None else pd.DataFrame()
    except:
        return pd.DataFrame()

def get_trade_days(end_date, n):
    start = (end_date - timedelta(days=n * 3)).strftime("%Y%m%d")
    cal = safe_get("trade_cal", start_date=start, end_date=end_date.strftime("%Y%m%d"))
    return cal[cal["is_open"] == 1].sort_values("cal_date", ascending=False)["cal_date"].head(n).tolist()

def get_hist(ts_code, start, end):
    df = safe_get("daily", ts_code=ts_code, start_date=start, end_date=end)
    if df.empty:
        return df
    df = df.sort_values("trade_date")
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =====================================================
# 核心选股逻辑
# =====================================================
def select_stocks(trade_date):
    daily = safe_get("daily", trade_date=trade_date)
    basic = safe_get(
        "daily_basic",
        trade_date=trade_date,
        fields="ts_code,turnover_rate,circ_mv"
    )

    df = daily.merge(basic, on="ts_code", how="left")

    # 基础过滤
    df = df[
        (df["close"] >= 8) &
        (df["close"] <= 80) &
        (df["turnover_rate"] >= 2) &
        (df["circ_mv"] / 10000 >= 30) &
        (df["circ_mv"] / 10000 <= 500)
    ]

    results = []

    for ts_code in df["ts_code"]:
        hist = get_hist(
            ts_code,
            (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d"),
            trade_date
        )

        if len(hist) < 10:
            continue

        close = hist["close"]
        vol = hist["vol"]

        # 强势
        ret_5 = close.iloc[-1] / close.iloc[-6] - 1
        if not 0.06 <= ret_5 <= 0.25:
            continue

        ma5 = close.rolling(5).mean()
        vol_ma5 = vol.rolling(5).mean()

        today = hist.iloc[-1]
        yesterday = hist.iloc[-2]

        pct = (today["close"] / yesterday["close"] - 1) * 100

        # 回踩
        if not (-3 <= pct <= -0.5):
            continue
        if today["vol"] > vol_ma5.iloc[-1]:
            continue
        if today["low"] < ma5.iloc[-1]:
            continue

        results.append({
            "ts_code": ts_code,
            "close": today["close"],
            "ret_5%": round(ret_5 * 100, 2)
        })

    return pd.DataFrame(results)

# =====================================================
# 收益模拟
# =====================================================
def simulate_trade(ts_code, trade_date):
    hist = get_hist(
        ts_code,
        (datetime.strptime(trade_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d"),
        (datetime.strptime(trade_date, "%Y%m%d") + timedelta(days=5)).strftime("%Y%m%d")
    )

    if hist.empty:
        return None

    buy = hist.iloc[0]["open"]
    sl = buy * (1 + STOP_LOSS / 100)
    tp = buy * (1 + TAKE_PROFIT / 100)

    for i in range(min(HOLD_DAYS, len(hist))):
        row = hist.iloc[i]
        if row["low"] <= sl:
            return STOP_LOSS
        if row["high"] >= tp:
            return TAKE_PROFIT

    return (hist.iloc[min(HOLD_DAYS-1, len(hist)-1)]["close"] / buy - 1) * 100

# =====================================================
# 主回测
# =====================================================
if st.button("🚀 开始回测"):
    trade_days = get_trade_days(END_DATE, BACKTEST_DAYS)
    all_trades = []

    bar = st.progress(0)

    for i, d in enumerate(trade_days):
        picks = select_stocks(d)
        for row in picks.itertuples():
            ret = simulate_trade(row.ts_code, d)
            if ret is not None:
                all_trades.append({
                    "交易日": d,
                    "股票": row.ts_code,
                    "5日涨幅%": row._3,
                    "收益%": round(ret, 2)
                })
        bar.progress((i + 1) / len(trade_days))

    bar.empty()

    if not all_trades:
        st.warning("没有产生任何交易")
        st.stop()

    df = pd.DataFrame(all_trades)

    st.header("📊 回测结果")
    st.metric("平均收益%", round(df["收益%"].mean(), 2))
    st.metric("胜率%", round((df["收益%"] > 0).mean() * 100, 1))
    st.metric("交易次数", len(df))

    st.dataframe(df.sort_values("交易日", ascending=False), use_container_width=True)
