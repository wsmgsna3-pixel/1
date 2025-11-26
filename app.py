import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts


# ========== Tushare 辅助函数 ==========

def get_pro(token: str):
    """用侧边栏输入的 token 创建 Tushare pro 对象。"""
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


def load_daily_range(pro, start_date: str, end_date: str):
    """从 Tushare 拉一段时间内所有 A 股的日线数据。"""
    try:
        df = pro.daily(start_date=start_date, end_date=end_date)
        if df.empty:
            st.warning("这一段时间内没有日线数据，请检查开始/结束日期是否正确。")
            return None
        df["trade_date"] = df["trade_date"].astype(str)
        return df
    except Exception as e:
        st.error(f"拉取日线数据失败：{e}")
        return None


# ========== 特征计算 & 选股逻辑（强势突破 + 放量，放宽版） ==========

def add_features(df_all: pd.DataFrame,
                 ma_short: int = 20,
                 ma_long: int = 60,
                 breakout_n: int = 20) -> pd.DataFrame:
    """一次性为所有股票计算均线、过去 N 日均量和最高价。"""
    df = df_all.sort_values(["ts_code", "trade_date"]).copy()
    # 均线
    df[f"ma_{ma_short}"] = df.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(ma_short, min_periods=1).mean()
    )
    df[f"ma_{ma_long}"] = df.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(ma_long, min_periods=1).mean()
    )
    # 过去 N 日平均成交量 & 最高价
    df["avg_vol_n"] = df.groupby("ts_code")["vol"].transform(
        lambda x: x.rolling(breakout_n, min_periods=1).mean()
    )
    df["high_n"] = df.groupby("ts_code")["high"].transform(
        lambda x: x.rolling(breakout_n, min_periods=1).max()
    )
    return df


def select_strong_breakout(df_all_feat: pd.DataFrame,
                           target_date: str,
                           min_amount: float = 2e7,
                           min_price: float = 3,
                           max_price: float = 200,
                           ma_short: int = 20,
                           ma_long: int = 60,
                           breakout_n: int = 20,
                           vol_ratio: float = 1.2,
                           min_chg: float = 1,
                           max_chg: float = 11) -> pd.DataFrame:
    """
    在某一天做“强势突破 + 放量”选股（放宽条件版）：
    - 成交额 >= 2000 万
    - 价格 3~200 元
    - 涨幅 1%~11%
    - 放量 >= 1.2 * 20 日均量
    - 收盘价 >= 近 20 日最高价 * 1.005
    """
    today = df_all_feat[df_all_feat["trade_date"] == target_date].copy()
    if today.empty:
        return today

    # 1) 基础过滤：成交额、价格区间
    today = today[today["amount"] * 1000 >= min_amount]
    today = today[(today["close"] >= min_price) & (today["close"] <= max_price)]

    # 2) 涨跌幅过滤
    today = today[(today["pct_chg"] >= min_chg) & (today["pct_chg"] <= max_chg)]

    # 3) 趋势过滤：close > MA20 > MA60
    today = today[
        (today[f"ma_{ma_short}"] > today[f"ma_{ma_long}"]) &
        (today["close"] > today[f"ma_{ma_short}"])
    ]

    # 4) 收盘在 K 线高位
    price_range = today["high"] - today["low"]
    pos_in_bar = (today["close"] - today["low"]) / price_range.replace(0, np.nan)
    today = today[pos_in_bar >= 0.6]

    # 5) 放量（1.2 倍 20 日均量）
    today = today[today["vol"] >= today["avg_vol_n"] * vol_ratio]

    # 6) 突破（收盘价 >= 近 20 日高点 * 1.005）
    today = today[today["close"] >= today["high_n"] * 1.005]

    show_cols = [
        "ts_code", "trade_date", "open", "high", "low", "close",
        "pct_chg", "vol", "amount",
        f"ma_{ma_short}", f"ma_{ma_long}", "avg_vol_n", "high_n"
    ]
    today = today[show_cols].sort_values("pct_chg", ascending=False)
    return today


# ========== 简单回测：每天选股 -> 次日开盘买入 -> 持 N 天后收盘卖出 ==========

def run_backtest(df_all_feat: pd.DataFrame,
                 trade_dates,
                 hold_days: int = 2,
                 initial_capital: float = 100000.0):
    """
    - 在 select_date 当天收盘后，根据示例策略选股
    - buy_date = 下一交易日：按开盘价等权买入
    - 持有 hold_days 天，在 sell_date 收盘卖出
    """
    if len(trade_dates) <= hold_days + 1:
        return None, None

    capital = initial_capital
    equity_curve = []
    trade_records = []

    # 初始一条权益点
    first_date = trade_dates[0]
    equity_curve.append({"trade_date": first_date, "equity": capital})

    for i in range(len(trade_dates) - hold_days - 1):
        select_date = trade_dates[i]
        buy_date = trade_dates[i + 1]
        sell_date = trade_dates[i + hold_days]

        # 1. 选股
        candidates = select_strong_breakout(df_all_feat, select_date)
        if candidates.empty:
            equity_curve.append({"trade_date": sell_date, "equity": capital})
            continue

        ts_list = candidates["ts_code"].unique().tolist()

        # 2. 买入（开盘价）
        buy_df = df_all_feat[
            (df_all_feat["trade_date"] == buy_date) &
            (df_all_feat["ts_code"].isin(ts_list))
        ][["ts_code", "open"]].copy()

        if buy_df.empty:
            equity_curve.append({"trade_date": sell_date, "equity": capital})
            continue

        # 3. 卖出（收盘价）
        sell_df = df_all_feat[
            (df_all_feat["trade_date"] == sell_date) &
            (df_all_feat["ts_code"].isin(ts_list))
        ][["ts_code", "close"]].copy()

        merged = pd.merge(buy_df, sell_df, on="ts_code", how="inner")
        if merged.empty:
            equity_curve.append({"trade_date": sell_date, "equity": capital})
            continue

        # 等权分配资金
        n = len(merged)
        capital_per_stock = capital / n
        merged["buy_price"] = merged["open"]
        merged = merged[merged["buy_price"] > 0]
        if merged.empty:
            equity_curve.append({"trade_date": sell_date, "equity": capital})
            continue

        merged["position"] = capital_per_stock / merged["buy_price"]
        merged["sell_price"] = merged["close"]
        merged["pnl"] = (merged["sell_price"] - merged["buy_price"]) * merged["position"]
        merged["ret"] = merged["pnl"] / (merged["buy_price"] * merged["position"])

        total_pnl = merged["pnl"].sum()
        capital = capital + total_pnl

        # 记录每一只的交易
        for _, row in merged.iterrows():
            trade_records.append({
                "select_date": select_date,
                "buy_date": buy_date,
                "sell_date": sell_date,
                "ts_code": row["ts_code"],
                "buy_price": round(row["buy_price"], 3),
                "sell_price": round(row["sell_price"], 3),
                "return": round(row["ret"], 4),
            })

        equity_curve.append({"trade_date": sell_date, "equity": capital})

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trade_records)
    return equity_df, trades_df


def calc_stats(equity_df: pd.DataFrame, initial_capital: float):
    """根据权益曲线计算总收益率和最大回撤。"""
    equity_df = equity_df.sort_values("trade_date")
    equity_series = equity_df["equity"].astype(float)
    if equity_series.empty:
        return None, None
    total_return = equity_series.iloc[-1] / initial_capital - 1.0
    cum_max = equity_series.cummax()
    drawdown = equity_series / cum_max - 1.0
    max_dd = drawdown.min()
    return float(total_return), float(max_dd)


# ========== Streamlit 界面 ==========

st.set_page_config(page_title="A股短线选股小工具", layout="wide")

st.title("A股短线选股 + 回测 Demo（放宽条件版）")
st.write(
    "已经接入 Tushare，用“强势突破 + 放量（放宽条件版）”做一个简单的区间回测："
    "每天选股 -> 次日开盘买入 -> 持 N 天后收盘卖出。"
)

st.sidebar.header("参数设置")

# 手动输入 token（只在本次会话里用，不保存）
ts_token = st.sidebar.text_input(
    "Tushare Token（只在本次会话使用，不会保存）",
    value="",
    type="password"
)

# 把默认时间拉长一点，方便有更多交易
start_date = st.sidebar.text_input("开始日期 (YYYYMMDD)", "20220101")
end_date = st.sidebar.text_input("结束日期 (YYYYMMDD)", "20240201")
hold_days = st.sidebar.slider("持股天数（示例策略）", 1, 5, 2)
initial_capital = st.sidebar.number_input("初始资金(元)", value=100000.0, step=10000.0)

run = st.sidebar.button("开始回测（示例策略）")

if run:
    st.success("按钮已点击：开始拉取区间日线数据并进行简单回测。")

    # 1. 示意收益曲线（假数据，对比用）
    dates_demo = pd.date_range("2023-01-01", periods=100)
    equity_demo = initial_capital * (1 + np.linspace(0, 0.5, 100))
    df_equity_demo = pd.DataFrame({"date": dates_demo, "equity": equity_demo}).set_index("date")
    st.subheader("示例收益曲线（假数据，用于对比）")
    st.line_chart(df_equity_demo["equity"])

    # 2. 拉取区间日线
    st.subheader("从 Tushare 拉取的区间日线数据概况")
    pro = get_pro(ts_token)
    if pro is not None:
        with st.spinner(f"正在拉取 {start_date} ~ {end_date} 的日线数据..."):
            df_all = load_daily_range(pro, start_date, end_date)

        if df_all is not None:
            trade_dates = sorted(df_all["trade_date"].unique().tolist())
            first_date = trade_dates[0]
            last_date = trade_dates[-1]
            st.write(f"成功获取 {len(trade_dates)} 个交易日，从 **{first_date}** 到 **{last_date}**。")
            st.write("下面是这段时间日线数据的前 20 行：")
            st.dataframe(df_all.head(20))

            # 3. 计算特征
            with st.spinner("正在为所有股票计算均线和成交量特征..."):
                df_feat = add_features(df_all)

            # 4. 看看最后一个交易日的选股结果
            st.subheader(f"示例选股结果（最后一个交易日：{last_date}）")
            selected_last = select_strong_breakout(df_feat, last_date)
            if selected_last.empty:
                st.warning("在最后一个交易日，没有股票符合当前示例策略的条件（放宽版）。")
            else:
                st.write(f"共选出 **{len(selected_last)}** 只股票：")
                st.dataframe(selected_last.reset_index(drop=True))

            # 5. 做区间回测
            st.subheader("策略回测结果（示例策略）")
            with st.spinner("正在进行简单回测计算..."):
                equity_df, trades_df = run_backtest(
                    df_feat,
                    trade_dates,
                    hold_days=hold_days,
                    initial_capital=initial_capital,
                )

            if equity_df is None or trades_df is None or equity_df.empty:
                st.warning("回测没有产生有效结果，可能是选股太少或日期区间太短。")
            else:
                st.write("策略真实收益曲线：")
                st.line_chart(equity_df.set_index("trade_date")["equity"])

                total_ret, max_dd = calc_stats(equity_df, initial_capital)
                n_trades = len(trades_df)

                st.write(f"总收益率：**{total_ret:.2%}**")
                st.write(f"最大回撤：**{max_dd:.2%}**")
                st.write(f"成交笔数：**{n_trades}**")

                st.write(
                    "说明：这是一个非常简化的示例回测，只用于学习思路，"
                    "尚未考虑手续费、滑点、仓位限制等。"
                )

                st.subheader("部分交易记录示例")
                st.dataframe(trades_df.head(50))

else:
    st.info("在左侧输入 token、开始/结束日期和持股天数，然后点击【开始回测】。")
