import streamlit as st
import akshare as ak
import pandas as pd
from datetime import datetime

st.title("全市场短线王（仅用新浪源 · 超宽松版）")

# ==================== 1. 交易时间检测 ====================
def is_trading_time():
    now = datetime.now()
    if now.weekday() >= 5:  # 周末
        return False
    hour, minute = now.hour, now.minute
    # 早盘 9:30-11:30
    if 9 <= hour < 11 or (hour == 11 and minute <= 30):
        return True
    # 午盘 13:00-15:00
    if 13 <= hour < 15 or (hour == 15 and minute == 0):
        return True
    return False

if not is_trading_time():
    st.warning("当前非交易时间（9:30-11:30 或 13:00-15:00），数据为昨收价，动量无效！建议开盘后运行。")
    st.info("提示：非交易时间可用于测试代码结构，正式选股请在开盘时运行。")

# ==================== 2. 获取新浪数据 ====================
@st.cache_data(ttl=300)  # 5分钟缓存
def get_sina_data():
    try:
        df = ak.stock_zh_a_spot()
        if df.empty:
            st.error("新浪返回空数据")
            return None
        st.success(f"新浪数据获取成功，共 {len(df)} 只股票")
        return df
    except Exception as e:
        st.error(f"新浪数据失败：{e}")
        return None

# ==================== 3. 主逻辑 ====================
if st.button("一键生成潜力股（超宽松版）"):
    with st.spinner("正在获取新浪实时行情..."):
        raw_data = get_sina_data()
        if raw_data is None:
            st.stop()

        # 列名处理（新浪中文）
        try:
            df = raw_data[["代码", "名称", "今开", "昨收", "成交量"]].copy()
            df.rename(columns={
                "代码": "code",
                "名称": "name",
                "今开": "open",
                "昨收": "close_yesterday",
                "成交量": "volume"
            }, inplace=True)
        except KeyError as e:
            st.error(f"列名缺失：{e}，请检查新浪接口")
            st.stop()

        # 转数值
        df["open"] = pd.to_numeric(df["open"], errors='coerce')
        df["close_yesterday"] = pd.to_numeric(df["close_yesterday"], errors='coerce')
        df["volume"] = pd.to_numeric(df["volume"], errors='coerce')
        df.dropna(subset=["open", "close_yesterday", "volume"], inplace=True)

        # 基础指标
        df["price_momentum"] = (df["open"] - df["close_yesterday"]) / df["close_yesterday"]
        median_vol = df["volume"].median()
        df["volume_ratio"] = df["volume"] / median_vol

        # ==================== 4. 超宽松过滤（保证出结果）===================
        df_filtered = df.copy()

        # 条件1：股价 > 5元（避免仙股）
        df_filtered = df_filtered[df_filtered["open"] > 5]
        st.write(f"股价 > 5元：{len(df_filtered)} 只")

        # 条件2：未涨停
        df_filtered = df_filtered[df_filtered["open"] < df_filtered["close_yesterday"] * 1.099]
        st.write(f"未涨停：{len(df_filtered)} 只")

        # 条件3：开盘涨幅 > 0%（只要高开就行）
        df_filtered = df_filtered[df_filtered["price_momentum"] > 0]
        st.write(f"高开 > 0%：{len(df_filtered)} 只")

        # 条件4：放量 > 1.5 倍（超宽松）
        df_filtered = df_filtered[df_filtered["volume_ratio"] > 1.5]
        st.write(f"放量 > 1.5倍：{len(df_filtered)} 只")

        if len(df_filtered) == 0:
            st.warning("仍无结果，建议在开盘时运行")
            st.stop()

        # 打分排序
        df_filtered["score"] = (
            df_filtered["price_momentum"] * 100 +
            df_filtered["volume_ratio"] * 10
        )
        result = df_filtered.sort_values("score", ascending=False).head(30)

        st.success(f"成功筛选出 {len(result)} 只潜力股！")
        st.dataframe(result[["code", "name", "open", "price_momentum", "volume_ratio", "score"]])
        csv = result.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载CSV", csv, "potential_stocks.csv", "text/csv")
