import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts


# ========== Tushare 相关函数 ==========

def get_pro(token: str):
    """
    使用侧边栏输入的 token 创建 Tushare pro 对象。
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


def load_daily_range(pro, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    从 Tushare 拉取一段时间内所有 A股的日线数据。
    start_date, end_date 形如 '20230901'
    """
    try:
        df = pro.daily(start_date=start_date, end_date=end_date)
        if df.empty:
            st.warning("这一段时间内没有日线数据，请检查开始/结束日期是否正确。")
            return None
        # 统一一下类型
        df["trade_date"] = df["trade_date"].astype(str)
        return df
    except Exception as e:
        st.error(f"拉取日线数据失败：{e}")
        return None


# ========== 选股逻辑（强势突破 + 放量） ==========

def select_strong_breakout(df_all: pd.DataFrame,
                           target_date: str,
                           min_amount: float = 3e7,
                           min_price: float = 5,
                           max_price: float = 150,
                           ma_short: int = 20,
                           ma_long: int = 60,
                           breakout_n: int = 20,
                           vol_ratio: float = 1.5,
                           min_chg: float = 3,
                           max_chg: float = 9) -> pd.DataFrame:
    """
    在 df_all 里，选出 target_date 这一天符合条件的股票。
    """

    # 按股票+日期排序
    df = df_all.sort_values(["ts_code", "trade_date"]).copy()

    # 计算均线
    df[f"ma_{ma_short}"] = df.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(ma_short, min_periods=1).mean()
    )
    df[f"ma_{ma_long}"] = df.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(ma_long, min_periods=1).mean()
    )

    # 计算过去 N 日平均成交量 & 最高价
    df["avg_vol_n"] = df.groupby("ts_code")["vol"].transform(
        lambda x: x.rolling(breakout_n, min_periods=1).mean()
    )
    df["high_n"] = df.groupby("ts_code")["high"].transform(
        lambda x: x.rolling(breakout_n, min_periods=1).max()
    )

    # 取出目标日期数据
    today = df[df["trade_date"] == target_date].copy()
    if today.empty:
        return today

    # 1）基础过滤：成交额、价格区间
    # amount 单位通常是“千元”，这里乘以 1000 约成元
    today = today[today["amount"] * 1000 >= min_amount]
    today = today[(today["close"] >= min_price) & (today["close"] <= max_price)]

    # 2）涨跌幅过滤
    today = today[(today["pct_chg"] >= min_chg) & (today["pct_chg"] <= max_chg)]

    # 3）趋势过滤：close > MA20 > MA60
    today = today[
        (today[f"ma_{ma_short}"] > today[f"ma_{ma_long}"]) &
        (today["close"] > today[f"ma_{ma_short}"])
    ]

    # 4）收盘在 K 线高位： (close - low) / (high - low) >= 0.6
    price_range = today["high"] - today["low"]
    pos_in_bar = (today["close"] - today["low"]) / price_range.replace(0, np.nan)
    today = today[pos_in_bar >= 0.6]

    # 5）放量：当日成交量 >= N 日平均量 * vol_ratio
    today = today[today["vol"] >= today["avg_vol_n"] * vol_ratio]

    # 6）突破：收盘价 >= 过去 N 日最高价 * 1.01
    today = today[today["close"] >= today["high_n"] * 1.01]

    # 挑一些主要字段展示
    show_cols = [
        "ts_code", "trade_date", "open", "high", "low", "close",
        "pct_chg", "vol", "amount",
        f"ma_{ma_short}", f"ma_{ma_long}", "avg_vol_n", "high_n"
    ]
    today = today[show_cols].sort_values("pct_chg", ascending=False)

    return today


# ========== 页面布局 ==========

st.set_page_config(page_title="A股短线选股小工具", layout="wide")

st.title("A股短线选股 + 回测 Demo（第 3 步）")
st.write("已经接入 Tushare：这一步我们拉一段时间的日线，并在最后一个交易日做一次“强势突破+放量”选股。")

# 侧边栏参数
st.sidebar.header("参数设置")

# 手动输入 Tushare token（密码模式）
ts_token = st.sidebar.text_input(
    "Tushare Token（只在本次会话使用，不会保存）",
    value="",
    type="password"
)

start_date = st.sidebar.text_input("开始日期 (YYYYMMDD)", "20230901")
end_date = st.sidebar.text_input("结束日期 (YYYYMMDD)", "20240201")
hold_days = st.sidebar.slider("未来计划持股天数（先不用于计算，预留）", 1, 5, 2)
initial_capital = st.sidebar.number_input("初始资金(元)", value=100000.0, step=10000.0)

run = st.sidebar.button("开始选股（强势突破 + 放量 示例）")

if run:
    st.success("按钮已点击：示例收益曲线 + 区间日线 + 最后一个交易日的选股结果。")

    # ========= 1. 示例收益曲线（仍然是假数据） =========
    dates = pd.date_range("2023-01-01", periods=100)
    equity = initial_capital * (1 + np.linspace(0, 0.5, 100))
    df_equity = pd.DataFrame({"date": dates, "equity": equity}).set_index("date")

    st.subheader("示例收益曲线（假数据）")
    st.line_chart(df_equity["equity"])

    # ========= 2. 拉取一段时间的日线数据 =========
    st.subheader("从 Tushare 拉取的区间日线数据概况")

    pro = get_pro(ts_token)
    if pro is not None:
        with st.spinner(f"正在拉取 {start_date} ~ {end_date} 的日线数据..."):
            df_all = load_daily_range(pro, start_date, end_date)

        if df_all is not None:
            # 展示一些基本信息
            trade_dates = sorted(df_all["trade_date"].unique().tolist())
            first_date = trade_dates[0]
            last_date = trade_dates[-1]

            st.write(f"成功获取 {len(trade_dates)} 个交易日，"
                     f"从 **{first_date}** 到 **{last_date}**。")
            st.write("下面是这段时间日线数据的前 20 行：")
            st.dataframe(df_all.head(20))

            # ========= 3. 在最后一个交易日做一次“强势突破+放量”选股 =========
            st.subheader(f"强势突破 + 放量 选股结果（交易日：{last_date}）")

            selected = select_strong_breakout(df_all, last_date)

            if selected.empty:
                st.warning("在最后一个交易日，没有股票符合当前示例策略的条件。"
                           "这很正常，说明条件比较严格，你可以之后调参数放宽。")
            else:
                st.write(f"共选出 **{len(selected)}** 只股票（已按当日涨跌幅从高到低排序）：")
                st.dataframe(selected.reset_index(drop=True))
else:
    st.info("在左侧输入 token、开始/结束日期，然后点击【开始选股】。")
