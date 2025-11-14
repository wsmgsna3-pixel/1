import streamlit as st
import akshare as ak
import pandas as pd
import time
from datetime import datetime

# 頁面設置
st.set_page_config(page_title="短線王 終極5條版", layout="wide")

# 警告：請勿離開頁面
st.markdown("""
<style>
    .big-warning {
        font-size: 18px !important;
        color: red;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border: 2px solid red;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="big-warning">請勿離開本頁面！短線王正在篩選，60 秒內出結果！</div>', unsafe_allow_html=True)

st.title("短線王（終極 5 條版）")

# 交易時間提示
now = datetime.now()
if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 15:
    st.warning("當前非交易時間，數據靜態，建議 9:30-15:00 運行")

# 獲取實時數據（多源 + 重試）
def get_realtime_data():
    sources = [
        ("新浪", lambda: ak.stock_zh_a_spot()),
        ("東方財富", lambda: ak.stock_zh_a_spot_em()),
    ]
    for name, func in sources:
        for retry in range(3):
            try:
                df = func()
                if not df.empty and len(df) > 100:
                    st.success(f"使用 {name} 數據源成功！共 {len(df)} 只股票")
                    return df
                time.sleep(1)
            except Exception as e:
                st.warning(f"{name} 第 {retry+1} 次失敗：{e}")
    st.error("所有數據源失敗")
    return None

# 獲取單股 10 天歷史數據（加重試）
def get_10d_data(symbol):
    for retry in range(3):
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
            time.sleep(1)
    return None

# 主邏輯
if st.button("一鍵生成短線王"):
    today = datetime.now().date()
    if 'last_run_date' in st.session_state and st.session_state.last_run_date == today and 'result' in st.session_state:
        st.success("使用今日快取結果！")
        result = st.session_state.result
    else:
        with st.spinner("拉取實時 + 歷史數據..."):
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

            codes = [c for c in df["code"].tolist() if c.startswith(('sh', 'sz'))][:300]
            history_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, code in enumerate(codes):
                status_text.text(f"拉取歷史數據 {i+1}/{len(codes)}: {code}")
                hist = get_10d_data(code)
                if hist:
                    hist["code"] = code
                    history_data.append(hist)
                progress_bar.progress((i + 1) / len(codes))
                time.sleep(0.05)

            progress_bar.empty()
            status_text.empty()

            if not history_data:
                st.warning("歷史數據拉取失敗，使用實時數據過濾")
                df["10d_return"] = 0.4
                df["10d_avg_turnover"] = 2.5
                df["volume_yesterday"] = df["volume"] / 2
                df["high_yesterday"] = df["close_yesterday"] * 0.98
            else:
                hist_df = pd.DataFrame(history_data)
                df = df.merge(hist_df, on="code", how="inner")

            df = df[df["open"] > 10]
            df = df[df["10d_return"] <= 0.50]
            df = df[df["10d_avg_turnover"] >= 3.0]
            df = df[df["volume"] > 3 * df["volume_yesterday"]]
            df = df[df["open"] > df["high_yesterday"]]

            if len(df) == 0:
                st.warning("今日無符合條件的短線王")
                st.stop()

            df["score"] = (df["open"] - df["close_yesterday"]) / df["close_yesterday"] * 100 + \
                          df["volume"] / df["volume_yesterday"] * 10
            result = df.sort_values("score", ascending=False).head(30)

            st.session_state.result = result
            st.session_state.last_run_date = today

    st.success(f"找到 {len(result)} 只終極短線王！")
    st.dataframe(result[["code", "name", "open", "10d_return", "10d_avg_turnover", "score"]], use_container_width=True)
    csv = result.to_csv(index=False).encode('utf-8-sig')
    st.download_button("下載 CSV", csv, "短線王_終極5條.csv", "text/csv")
