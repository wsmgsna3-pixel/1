import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# 设置Tushare Token
ts.set_token('your_tushare_token')
pro = ts.pro_api()

# 拉取数据函数
def load_data(pro, start_date, end_date):
    """
    拉取指定日期区间的股票数据
    """
    try:
        st.write(f"正在拉取数据：从 {start_date} 到 {end_date}")
        df = pro.daily(start_date=start_date, end_date=end_date)
        
        if df.empty:
            st.warning(f"没有获取到数据，检查日期区间是否正确：{start_date} ~ {end_date}")
            return None
        
        st.write(f"成功获取 {len(df)} 条数据")
        st.dataframe(df.head())  # 显示前几条数据进行检查
        
        df["trade_date"] = df["trade_date"].astype(str)
        df = df.sort_values(by=["ts_code", "trade_date"])
        return df
    except Exception as e:
        st.error(f"获取数据失败：{e}")
        return None

# 特征计算函数
def add_features(df):
    """
    计算股票数据的均线、过去 N 日均量等特征
    """
    df["ma_20"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).mean())
    df["ma_60"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(60).mean())
    df["avg_vol_20"] = df.groupby("ts_code")["vol"].transform(lambda x: x.rolling(20).mean())
    df["high_20"] = df.groupby("ts_code")["high"].transform(lambda x: x.rolling(20).max())
    
    st.write(f"计算后的数据特征: {df.columns.tolist()}")
    st.dataframe(df.head())  # 显示数据查看特征是否计算成功
    return df

# 选股策略函数
def select_stocks(df, target_date):
    """
    按照强势突破 + 放量的策略进行选股
    """
    today = df[df["trade_date"] == target_date]
    
    if today.empty:
        st.warning(f"没有找到符合条件的数据: {target_date}")
        return today

    # 筛选条件：成交额、价格、涨幅、均线等
    today = today[today["amount"] * 1000 >= 5e6]  # 筛选成交额大于500万的股票
    today = today[(today["close"] >= 1) & (today["close"] <= 1000)]  # 筛选价格区间
    today = today[(today["pct_chg"] >= 0) & (today["pct_chg"] <= 20)]  # 筛选涨幅0~20%

    # 筛选趋势条件：20日均线 > 60日均线，且收盘价大于20日均线
    today = today[(today["ma_20"] > today["ma_60"]) & (today["close"] > today["ma_20"])]
    
    # 放量：放量要求 >= 20日均量的0.8倍
    today = today[today["vol"] >= today["avg_vol_20"] * 0.8]
    
    # 突破：收盘价大于过去20日的最高价 * 1.005
    today = today[today["close"] >= today["high_20"] * 1.005]

    st.write(f"符合选股条件的股票：{today[['ts_code', 'trade_date', 'close', 'pct_chg']].head()}")
    return today

# 主函数
def main():
    st.title("股票回测与选股系统")
    
    # Token输入框在侧边栏
    token = st.sidebar.text_input("请输入 Tushare Token")

    # 检查是否输入了 Token
    if token:
        st.sidebar.success("Token 已输入！")
    else:
        st.sidebar.warning("请先输入您的 Token")

    # 输入框：开始日期、结束日期、持股天数、初始资金等
    start_date = st.sidebar.text_input("开始日期 (YYYYMMDD)", "20220101")
    end_date = st.sidebar.text_input("结束日期 (YYYYMMDD)", "20220201")
    holding_days = st.sidebar.slider("持股天数", 1, 20, 5)
    initial_capital = st.sidebar.number_input("初始资金 (元)", min_value=10000, value=100000)

    # 获取数据
    df = load_data(pro, start_date, end_date)
    
    if df is not None:
        df = add_features(df)
        
        # 选股
        selected_stocks = select_stocks(df, end_date)

        # 如果选股成功，展示选股结果
        if not selected_stocks.empty:
            st.write("选股结果：", selected_stocks)
        else:
            st.write("没有找到符合条件的股票。")

# 运行主函数
if __name__ == "__main__":
    main()
