import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts


# ========== Tushare 相关 ==========

def get_pro(token: str):
    """
    使用你在侧边栏输入的 token 创建 Tushare pro 对象。
    不会把 token 保存到任何文件或页面上。
    """
    if not token:
        st.error("请在左侧输入你的 Tushare token。")
        return None

    try:
        ts.set_token(token)
        pro = ts.pro_api()
        return pro
    except Exception as e:
        st.error(f"创建 Tushare 接口失败：{e}")
        return None


# ========== 页面布局 ==========

st.set_page_config(page_title="A股短线选股小工具", layout="wide")

st.title("A股短线选股 + 回测 Demo（第 2 步）")
st.write("现在开始接入 Tushare，先拉一天天的 A股日线数据展示。")

# 侧边栏参数
st.sidebar.header("参数设置")

# 手动输入 Tushare token（密码模式，不会显示明文）
ts_token = st.sidebar.text_input(
    "Tushare Token（只在本次会话使用，不会保存）",
    value="",
    type="password"
)

start_date = st.sidebar.text_input("开始日期 (YYYYMMDD)", "20240102")
hold_days = st.sidebar.slider("持股天数", 1, 5, 2)
initial_capital = st.sidebar.number_input("初始资金(元)", value=100000.0, step=10000.0)

# 按钮
run = st.sidebar.button("开始测试 Tushare 数据（示例版）")

if run:
    st.success("按钮已点击：先用示例收益曲线 + 一天的真实 Tushare 行情演示。")

    # ========= 1. 示例收益曲线（还是假数据） =========
    dates = pd.date_range("2023-01-01", periods=100)
    equity = initial_capital * (1 + np.linspace(0, 0.5, 100))
    df_equity = pd.DataFrame({"date": dates, "equity": equity}).set_index("date")

    st.subheader("示例收益曲线（假数据）")
    st.line_chart(df_equity["equity"])

    # ========= 2. 用 Tushare 拉取一天的 A股日线数据 =========
    st.subheader("Tushare A股日线示例数据")

    pro = get_pro(ts_token)
    if pro is not None:
        with st.spinner(f"正在从 Tushare 拉取 {start_date} 的日线数据..."):
            try:
                df_daily = pro.daily(trade_date=start_date)
            except Exception as e:
                st.error(f"拉取 Tushare 数据失败：{e}")
                df_daily = None

        if df_daily is not None:
            if df_daily.empty:
                st.warning(
                    f"{start_date} 这一天可能不是交易日，或者没有数据。"
                    "你可以换一个日期，比如 20240102 再试试。"
                )
            else:
                st.write(f"以下是 {start_date} 的部分日线数据（前 20 行）：")
                st.dataframe(df_daily.head(20))

else:
    st.info("在左侧输入 Tushare token 和开始日期，然后点击【开始测试 Tushare 数据（示例版）】。")
