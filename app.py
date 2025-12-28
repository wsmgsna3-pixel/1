import tushare as ts
import pandas as pd
import streamlit as st

# 请填入您的 Token
ts.set_token('您的Token')
pro = ts.pro_api()

st.title("Tushare 接口健康度体检")

try:
    st.write("正在尝试获取最近交易日的日线数据...")
    # 尝试裸取一次数据，不加任何保护
    df = pro.daily(trade_date='20241220') 
    
    if df.empty:
        st.error("❌ 接口返回了空数据！可能是权限受限或数据未更新。")
    else:
        st.success(f"✅ 接口正常！获取到 {len(df)} 条数据。")
        st.dataframe(df.head())

except Exception as e:
    st.error("❌ 接口报错！这就是选不出股的真实原因：")
    st.code(str(e)) # 这里会显示具体的报错信息，比如 "每分钟访问超限" 或 "TCP连接重置"
