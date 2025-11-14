import streamlit as st
import akshare as ak
import pandas as pd
from datetime import datetime

st.title("短線王（終極 5 條版）")

# 交易時間提示
now = datetime.now()
if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 15:
    st.warning("當前非交易時間，數據靜態，建議 9:30-15:00 運行")

# 獲取實時數據
def get_realtime_data():
    sources = [
        ("新浪", lambda: ak.stock_zh_a_spot()),
        ("東方財富", lambda: ak.stock_zh_a_spot_em()),
    ]
    for name, func in sources:
        try:
            df = func()
            if not df.empty and len(df) > 100:
                st.success(f"使用 {name} 數據源成功！共 {len(df)} 只股票")
                return df
        except Exception as e:
            st.warning(f"{name} 失敗：{e}")
    st.error("所有源失敗")
    return None

# 新增：獲取 10 天歷史數據
def get_10d_data(symbol):
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
        if len(df) < 10:
            return None
        df = df.tail(10)
        return {
            "10d_return": (df.iloc[-1]['收盘'] / df.iloc[0]['开盘']) - 1,
            "10d_avg_turnover": df['换手率'].mean(),
            "high_yesterday": df.iloc[-2]['最高'],
            "volume_yesterday": df.iloc[-2]['成交量']
        }
    except:
        return None

if st.button("一鍵生成短線王"):
    with st.spinner("拉取實時 + 歷史數據..."):
        # 1. 獲取實時數據
        data = get_realtime_data()
        if data is None:
            st.stop()

        df = data[["代码", "名称", "今开", "昨收", "成交量"]].copy()
        df.rename(columns={
            "代码": "code", "名称": "name", "今开": "open",
            "昨收": "close_yesterday", "成交量": "volume"
        }, inplace=True)

        df["open"] = pd.to_numeric(df["open"], errors='coerce')
        df["close_yesterday"] = pd.to_numeric(df["close_yesterday"], errors='coerce')
        df["volume"] = pd.to_numeric(df["volume"], errors='coerce')
        df.dropna(subset=["open", "close_yesterday", "volume"], inplace=True)

        # 2. 循環獲取 10 天數據（限前 1000 隻，避免超時）
        history_data = []
        codes = df["code"].tolist()[:1000]
        for i, code in enumerate(codes):
            st.write(f"處理 {i+1}/{len(codes)}: {code}")
            hist = get_10d_data(code)
            if hist:
                hist["code"] = code
                history_data.append(hist)
            time.sleep(0.1)  # 防限流

        if not history_data:
            st.error("無歷史數據")
            st.stop()

        hist_df = pd.DataFrame(history_data)
        df = df.merge(hist_df, on="code", how="inner")

        # 3. 終極 5 條過濾
        df = df[df["open"] > 10]                                      # 你的條件 1
        df = df[df["10d_return"] <= 0.50]                             # 你的條件 2
        df = df[df["10d_avg_turnover"] >= 3.0]                        # 你的條件 3
        df = df[df["volume"] > 3 * df["volume_yesterday"]]            # 我的條件 1
        df = df[df["open"] > df["high_yesterday"]]                    # 我的條件 2

        if len(df) == 0:
            st.warning("無符合條件的股票")
            st.stop()

        # 打分
        df["score"] = (df["open"] - df["close_yesterday"]) / df["close_yesterday"] * 100 + \
                      df["volume"] / df["volume_yesterday"] * 10
        result = df.sort_values("score", descending=True).head(30)

        st.success(f"找到 {len(result)} 只終極短線王！")
        st.dataframe(result[["code", "name", "open", "10d_return", "10d_avg_turnover", "score"]])
        st.download_button("下載CSV", result.to_csv(index=False).encode('utf-8-sig'), "ultimate_kings.csv")
