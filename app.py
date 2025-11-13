import streamlit as st
import akshare as ak
import pandas as pd
from datetime import datetime

st.title("短线王（腾讯源，稳定版）")

# 交易时间提示
now = datetime.now()
if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 15:
    st.warning("当前非交易时间，数据静态，建议 9:30-15:00 运行")

if st.button("一键生成短线王"):
    with st.spinner("拉取腾讯实时行情..."):
        try:
            df = ak.stock_zh_a_spot_tx()  # 腾讯源
            if df.empty:
                st.error("数据为空")
                st.stop()

            # 列处理（腾讯中文列名）
            df = df[["代码", "名称", "今开", "昨收", "成交量"]].copy()
            df.rename(columns={
                "代码": "code", "名称": "name", "今开": "open",
                "昨收": "close_yesterday", "成交量": "volume"
            }, inplace=True)

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
            df_filtered = df_filtered[df_filtered["price_momentum"] > 0.01]  # > 1%
            df_filtered = df_filtered[df_filtered["volume_ratio"] > 1.5]  # > 1.5 倍

            st.write(f"初始：{len(df)} | 股价 > 10：{len(df_filtered)} | 涨幅 > 1%：{len(df_filtered)} | 放量 > 1.5：{len(df_filtered)}")

            if len(df_filtered) == 0:
                st.warning("无结果，建议开盘运行")
                st.stop()

            # 打分
            df_filtered["score"] = df_filtered["price_momentum"] * 100 + df_filtered["volume_ratio"] * 10
            result = df_filtered.sort_values("score", ascending=False).head(30)

            st.success(f"找到 {len(result)} 只短线王！")
            st.dataframe(result[["code", "name", "open", "price_momentum", "volume_ratio", "score"]])

            csv = result.to_csv(index=False).encode('utf-8-sig')
            st.download_button("下载CSV", csv, "tencent_kings.csv", "text/csv")
        except Exception as e:
            st.error(f"腾讯源失败：{e}")
