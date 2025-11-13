import streamlit as st
import akshare as ak
import pandas as pd
import requests
from datetime import datetime
import time

st.title("短线王（多源稳定版）")

# 交易时间提示
now = datetime.now()
if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 15:
    st.warning("当前非交易时间，数据静态，建议 9:30-15:00 运行")

def get_realtime_data():
    sources = [
        ("新浪", lambda: ak.stock_zh_a_spot()),
        ("东方财富", lambda: ak.stock_zh_a_spot_em()),
    ]
    for name, func in sources:
        try:
            df = func()
            if not df.empty and len(df) > 100:  # 验证数据量
                st.success(f"使用 {name} 数据源成功！共 {len(df)} 只股票")
                return df
        except Exception as e:
            st.warning(f"{name} 失败：{e}，尝试下一个...")
    
    # 兜底：腾讯 HTTP 直接调用（无库）
    try:
        st.info("兜底使用腾讯 HTTP 接口...")
        r = requests.get('http://qt.gtimg.cn/q=s_sh000001,s_sz399001')  # 示例全市场
        data_str = r.text.split('=')[1].strip(';')
        # 解析示例（简化，实际需扩展）
        st.warning("腾讯 HTTP 需自定义解析，建议升级 akshare")
        return pd.DataFrame()  # 临时空 DF
    except:
        st.error("所有源失败，请升级 akshare 或检查网络")
        return None

if st.button("一键生成短线王"):
    with st.spinner("拉取实时行情..."):
        data = get_realtime_data()
        if data is None:
            st.stop()

        # 列处理（通用中文列名）
        try:
            df = data[["代码", "名称", "今开", "昨收", "成交量"]].copy()
            df.rename(columns={
                "代码": "code", "名称": "name", "今开": "open",
                "昨收": "close_yesterday", "成交量": "volume"
            }, inplace=True)
        except KeyError as e:
            st.error(f"列名问题：{e}")
            st.stop()

        df["open"] = pd.to_numeric(df["open"], errors='coerce')
        df["close_yesterday"] = pd.to_numeric(df["close_yesterday"], errors='coerce')
        df["volume"] = pd.to_numeric(df["volume"], errors='coerce')
        df.dropna(subset=["open", "close_yesterday", "volume"], inplace=True)

        # 指标计算
        df["price_momentum"] = (df["open"] - df["close_yesterday"]) / df["close_yesterday"]
        median_vol = df["volume"].median()
        df["volume_ratio"] = df["volume"] / median_vol

        # 过滤（宽松版）
        df_filtered = df[df["open"] > 10]
        df_filtered = df_filtered[df_filtered["open"] < df_filtered["close_yesterday"] * 1.099]
        df_filtered = df_filtered[df_filtered["price_momentum"] > 0.01]
        df_filtered = df_filtered[df_filtered["volume_ratio"] > 1.5]

        st.write(f"初始：{len(df)} | 最终：{len(df_filtered)}")

        if len(df_filtered) == 0:
            st.warning("无结果，建议开盘运行")
            st.stop()

        # 打分
        df_filtered["score"] = df_filtered["price_momentum"] * 100 + df_filtered["volume_ratio"] * 10
        result = df_filtered.sort_values("score", ascending=False).head(30)

        st.success(f"找到 {len(result)} 只短线王！")
        st.dataframe(result[["code", "name", "open", "price_momentum", "volume_ratio", "score"]])

        csv = result.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载CSV", csv, "multi_source_kings.csv", "text/csv")
