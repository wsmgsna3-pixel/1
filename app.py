import streamlit as st
import pandas as pd
import numpy as np

# 页面基本设置
st.set_page_config(page_title="A股短线选股小工具", layout="wide")

st.title("A股短线选股 + 回测 Demo（第 1 步）")
st.write("我目前只用示例数据，先把界面跑起来，后面再接 Tushare。")

# 侧边栏参数
st.sidebar.header("参数设置")

start_date = st.sidebar.text_input("开始日期 (YYYYMMDD)", "20230101")
hold_days = st.sidebar.slider("持股天数", 1, 5, 2)
initial_capital = st.sidebar.number_input("初始资金(元)", value=100000.0, step=10000.0)

# 点击按钮后执行
if st.sidebar.button("开始回测（示例数据）"):
    st.success("按钮已点击：这是第 1 步，我先用示例数据演示。")

    # ========= 下面是“假数据”收益曲线 =========
    dates = pd.date_range("2023-01-01", periods=100)
    # 假设从初始资金慢慢涨到 50% 收益
    equity = initial_capital * (1 + np.linspace(0, 0.5, 100))
    df = pd.DataFrame({"date": dates, "equity": equity}).set_index("date")

    st.subheader("示例收益曲线（假数据）")
    st.line_chart(df["equity"])

    st.subheader("示例交易记录（假数据）")
    trades = pd.DataFrame({
        "买入日期": ["2023-01-10", "2023-02-05"],
        "卖出日期": ["2023-01-12", "2023-02-07"],
        "股票代码": ["000001.SZ", "600000.SH"],
        "单笔收益率": [0.05, -0.02],
    })
    st.dataframe(trades)

else:
    st.info("在左侧设置参数后，点击【开始回测（示例数据）】。")
