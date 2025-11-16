import streamlit as st
import tushare as ts
import pandas as pd
import time

st.set_page_config(page_title="选股王 · 极速版", layout="wide")

st.title("选股王 · 极速版（手动输入 Token 版）")

# 手动输入 token
ts_token = st.text_input("请输入你的 TuShare Token：", type="password")

if not ts_token:
    st.info("请输入 Token 后继续")
    st.stop()

# 初始化 tushare
ts.set_token(ts_token)
pro = ts.pro_api()

# --------------------
# 功能函数
# --------------------

@st.cache_data(show_spinner=False)
def load_trade_calendar():
    df = pro.trade_cal(exchange='', is_open='1')
    return df.sort_values("cal_date", ascending=True)

@st.cache_data(show_spinner=False)
def get_last_trade_date():
    cal = load_trade_calendar()
    return cal["cal_date"].iloc[-1]

@st.cache_data(show_spinner=False)
def load_daily(ts_code, start_date):
    df = pro.daily(ts_code=ts_code, start_date=start_date)
    return df.sort_values("trade_date")

@st.cache_data(show_spinner=False)
def get_yesterday_top500(trade_date):
    df = pro.daily(trade_date=trade_date)
    df = df.sort_values("pct_chg", ascending=False).head(500)
    return df

# --------------------
# 主逻辑：方案 B（你昨天用的那套逻辑）
# --------------------

def select_stocks(df):
    """
    你的逻辑：昨天涨幅前500 → 剔除 ST → 剔除10元以下 → 剔除200元以上
    """
    df = df.copy()

    # 去除ST
    df = df[~df["ts_code"].str.contains("ST")]

    # 过滤价格
    df = df[(df["close"] >= 10) & (df["close"] <= 200)]

    return df


# --------------------
# UI 操作
# --------------------

st.subheader("正在运行选股…")

with st.spinner("正在获取最新交易日…"):
    trade_date = get_last_trade_date()

st.write(f"检测到最新交易日：**{trade_date}**")

with st.spinner("正在获取昨日涨幅前500…"):
    df500 = get_yesterday_top500(trade_date)

st.write(f"昨日涨幅前500 共：**{df500.shape[0]}** 只")

with st.spinner("正在筛选符合条件的股票…"):
    result = select_stocks(df500)

st.success(f"筛选完成，共 {result.shape[0]} 只股票")

st.dataframe(result)
