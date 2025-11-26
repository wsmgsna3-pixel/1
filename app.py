# 拉取数据后立即检查
def load_daily_range(pro, start_date: str, end_date: str):
    """从 Tushare 拉取指定日期区间的所有A股日线数据。"""
    try:
        st.write(f"正在拉取数据：从 {start_date} 到 {end_date}")  # 输出调试信息
        df = pro.daily(start_date=start_date, end_date=end_date)
        
        # 确保数据不为空
        if df.empty:
            st.warning(f"没有获取到数据，检查日期区间是否正确：{start_date} ~ {end_date}")
            return None
        
        # 显示部分数据，检查数据是否完整
        st.write(f"成功获取 {len(df)} 条数据")
        st.dataframe(df.head(10))  # 查看前10条数据
        
        # 确保日期列是字符串格式，并按日期排序
        df["trade_date"] = df["trade_date"].astype(str)
        df = df.sort_values("trade_date")
        return df
    except Exception as e:
        st.error(f"拉取日线数据失败：{e}")
        return None

# 特征计算部分（无数据丢失的情况下，确认日期区间）
def add_features(df_all: pd.DataFrame,
                 ma_short: int = 20,
                 ma_long: int = 60,
                 breakout_n: int = 20) -> pd.DataFrame:
    """一次性为所有股票计算均线、过去 N 日均量和最高价。"""
    df = df_all.sort_values(["ts_code", "trade_date"]).copy()
    
    st.write(f"开始为数据计算特征，当前数据集大小: {len(df)}")

    # 均线计算
    df[f"ma_{ma_short}"] = df.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(ma_short, min_periods=1).mean()
    )
    df[f"ma_{ma_long}"] = df.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(ma_long, min_periods=1).mean()
    )
    
    # 过去 N 日均量和最高价
    df["avg_vol_n"] = df.groupby("ts_code")["vol"].transform(
        lambda x: x.rolling(breakout_n, min_periods=1).mean()
    )
    df["high_n"] = df.groupby("ts_code")["high"].transform(
        lambda x: x.rolling(breakout_n, min_periods=1).max()
    )

    st.write(f"计算特征后的数据集大小: {len(df)}")
    st.dataframe(df.head(10))  # 显示前10条特征计算后的数据
    return df

# 选股部分（确保过滤条件不会不合理地减少数据）
def select_strong_breakout(df_all_feat: pd.DataFrame,
                           target_date: str,
                           min_amount: float = 5e6,  # 成交额进一步降低
                           min_price: float = 1,  # 最低价降到 1
                           max_price: float = 1000,  # 价格上限 1000
                           ma_short: int = 20,
                           ma_long: int = 60,
                           breakout_n: int = 20,
                           vol_ratio: float = 0.8,  # 放量要求进一步放宽
                           min_chg: float = 0,  # 涨幅从 0 开始
                           max_chg: float = 20) -> pd.DataFrame:
    """
    在某一天做“强势突破 + 放量”选股（放宽条件版）：
    - 成交额 >= 500 万
    - 价格 1~1000 元
    - 涨幅 0%~20%
    - 放量 >= 0.8 * 20 日均量
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

    # 5) 放量（0.8 倍 20 日均量）
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
