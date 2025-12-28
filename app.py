import streamlit as st
import tushare as ts
import pandas as pd

st.title("🏥 Tushare 接口健康度体检中心")

# 1. 这里加了输入框，您可以在网页上粘贴 Token
token = st.text_input("请输入您的 Tushare Token:", type="password")

if st.button("开始体检"):
    if not token:
        st.error("请先输入 Token！")
        st.stop()
    
    # 设置 Token
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        
        st.info("正在尝试连接 Tushare 服务器...")
        
        # 2. 尝试获取最近一个确定的交易日数据 (比如 2024-12-20)
        # 这里的日期选一个绝对过去的日期，确保有数据
        test_date = '20241220'
        df = pro.daily(trade_date=test_date)
        
        if df.empty:
            st.error(f"❌ 连接成功，但没有获取到数据！(日期: {test_date})")
            st.warning("原因分析：\n1. 您的积分可能不足以支持某些高频接口。\n2. 您的 IP 可能被暂时限流了（休息20分钟再试）。")
        else:
            st.success(f"✅ 接口完全正常！成功获取到 {len(df)} 行数据。")
            st.write(f"数据样例 ({test_date}):")
            st.dataframe(df.head())
            
            st.balloons()
            st.markdown("### 🎉 结论：您的 Token 和网络都没问题！")
            st.markdown("如果主程序选不出股，那一定是 **筛选条件太严** 或者 **日期设置到了未来**。")

    except Exception as e:
        st.error("❌ 接口报错！请截图以下信息：")
        st.code(str(e))
