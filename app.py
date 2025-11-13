import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.title("iTick 短线王（实时选股）")

# ==================== 输入你的 iTick Key ====================
ITICK_KEY = st.text_input("请输入你的 iTick API Key", type="password", value="")
if not ITICK_KEY:
    st.warning("请输入 Key 后点击下方按钮运行")
    st.stop()

# ==================== 交易时间提示 ====================
now = datetime.now()
if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 15:
    st.warning("当前非交易时间，数据为昨收价，建议 9:30-15:00 运行")

# ==================== 获取 iTick 实时行情 ====================
@st.cache_data(ttl=300)  # 5分钟缓存
def get_itick_data():
    try:
        # 1. 获取所有 A股 代码列表（iTick 提供）
        codes_url = "https://api.itick.org/stock/codes"
        codes_resp = requests.get(codes_url, headers={"Authorization": ITICK_KEY})
        if codes_resp.status_code != 200:
            st.error("获取代码列表失败")
            return None
        codes = [item['code'] for item in codes_resp.json()['data'] if item['code'].startswith(('sh', 'sz'))]

        # 2. 批量查询实时行情（单次最多 500 只）
        batch_size = 500
        all_data = []
        for i in range(0, len(codes), batch_size):
            batch_codes = ','.join(codes[i:i+batch_size])
            url = f"https://api.itick.org/stock/realtime?codes={batch_codes}"
            resp = requests.get(url, headers={"Authorization": ITICK_KEY})
            if resp.status_code == 200:
                all_data.extend(resp.json()['data'])
        
        if not all_data:
            st.error("无数据返回")
            return None

        df = pd.DataFrame(all_data)
        # 字段映射
        df['code'] = df['code'].str.lower()
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['close_yesterday'] = pd.to_numeric(df['pre_close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df.dropna(subset=['open', 'close_yesterday', 'volume'], inplace=True)

        # 计算指标
        df['price_momentum'] = (df['open'] - df['close_yesterday']) / df['close_yesterday']
        median_vol = df['volume'].median()
        df['volume_ratio'] = df['volume'] / median_vol

        st.success(f"iTick 数据获取成功！共 {len(df)} 只股票")
        return df

    except Exception as e:
        st.error(f"iTick 失败：{e}")
        return None

# ==================== 主逻辑 ====================
if st.button("一键生成短线王"):
    with st.spinner("正在拉取 iTick 实时行情..."):
        data = get_itick_data()
        if data is None:
            st.stop()

        df = data.copy()
        st.write(f"**初始股票数**：{len(df)}")

        # 你的核心过滤条件
        df = df[df["open"] > 10]
        st.write(f"**股价 > 10元**：{len(df)}")

        df = df[df["open"] < df["close_yesterday"] * 1.099]
        st.write(f"**未涨停**：{len(df)}")

        df = df[df["price_momentum"] > 0.02]
        st.write(f"**涨幅 > 2%**：{len(df)}")

        df = df[df["volume_ratio"] > 2]
        st.write(f"**放量 > 2倍**：{len(df)}")

        if len(df) == 0:
            st.warning("无符合条件股票，建议开盘运行")
            st.stop()

        # 打分排序
        df["score"] = df["price_momentum"] * 100 + df["volume_ratio"] * 10
        result = df.sort_values("score", ascending=False).head(30)

        st.success(f"找到 {len(result)} 只短线王！")
        st.dataframe(result[["code", "name", "open", "price_momentum", "volume_ratio", "score"]])

        # 下载 CSV
        csv = result.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载CSV", csv, "itick_kings.csv", "text/csv")
