import streamlit as st
import tushare as ts
import pandas as pd
from datetime import datetime

st.title("Tushare 短线王（1-5天操作）")

# ==================== 输入 Token ====================
TUSHARE_TOKEN = st.text_input("请输入你的 Tushare Token", type="password", value="")
if not TUSHARE_TOKEN:
    st.warning("请输入 Token 后点击下方按钮运行")
    st.stop()

pro = ts.pro_api(TUSHARE_TOKEN)

# ==================== 交易时间提示 ====================
now = datetime.now()
if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 15:
    st.warning("当前非交易时间，数据为昨收价，建议 9:30-15:00 运行")

# ==================== 获取实时行情（最新写法）===================
@st.cache_data(ttl=300)
def get_realtime_data():
    try:
        # 1. 获取所有股票最新行情（含今开、昨收、成交量）
        df = pro.daily(trade_date='', fields='ts_code,open,pre_close,vol')
        if df.empty:
            st.error("数据为空")
            return None

        # 2. 获取股票名称
        name_df = pro.stock_basic(fields='ts_code,name')
        df = df.merge(name_df, on='ts_code', how='left')

        # 3. 格式化
        df['code'] = df['ts_code'].str.replace('.SH','sh').str.replace('.SZ','sz').str.lower()
        df['close_yesterday'] = df['pre_close']
        df['volume'] = df['vol']
        df['volume_ratio'] = df['volume'] / df['volume'].median()
        df['price_momentum'] = (df['open'] - df['close_yesterday']) / df['close_yesterday']

        st.success(f"Tushare 数据获取成功！共 {len(df)} 只股票")
        return df

    except Exception as e:
        st.error(f"Tushare 失败：{e}")
        return None

# ==================== 主逻辑 ====================
if st.button("一键生成短线王"):
    with st.spinner("正在拉取 Tushare 实时行情..."):
        data = get_realtime_data()
        if data is None:
            st.stop()

        df = data.copy()
        st.write(f"**初始股票数**：{len(df)}")

        # 过滤条件
        df = df[df["open"] > 10]
        st.write(f"**股价 > 10元**：{len(df)}")

        df = df[df["open"] < df["close_yesterday"] * 1.099]
        st.write(f"**未涨停**：{len(df)}")

        df = df[df["price_momentum"] > 0.02]
        st.write(f"**涨幅 > 2%**：{len(df)}")

        df = df[df["volume_ratio"] > 2]
        st.write(f"**放量 > 2倍**：{len(df)}")

        if len(df) == 0:
            st.warning("无符合条件股票，建议开盘运行或放宽条件")
            st.stop()

        # 打分
        df["score"] = df["price_momentum"] * 100 + df["volume_ratio"] * 10
        result = df.sort_values("score", ascending=False).head(30)

        st.success(f"找到 {len(result)} 只短线王！")
        st.dataframe(result[["code", "name", "open", "price_momentum", "volume_ratio", "score"]])

        # 下载
        csv = result.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载CSV", csv, "tushare_kings.csv", "text/csv")
