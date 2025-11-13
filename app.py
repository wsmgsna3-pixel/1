import streamlit as st
import akshare as ak
import pandas as pd
from datetime import datetime

st.title("全市场短线王（1-5天操作）")

# 交易时间检查
def is_trading_time():
    now = datetime.now()
    hour, minute = now.hour, now.minute
    if now.weekday() >= 5:  # 周末
        return False
    if hour < 9 or hour > 15:
        return False
    if hour == 11 and minute > 30:  # 午休
        return False
    if hour < 13 and hour > 11:  # 午休
        return False
    return True

if not is_trading_time():
    st.warning("⚠️ 当前非交易时间（9:30-11:30 或 13:00-15:00），数据为静态，短线指标无效！建议开盘后运行。")

# 缓存历史数据
@st.cache_data(ttl=3600)
def get_hist_data(code):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="")
        return df.tail(15)
    except:
        return pd.DataFrame()

# 获取实时数据（新浪）
def get_realtime_data():
    try:
        raw_data = ak.stock_zh_a_spot()
        if raw_data.empty:
            st.error("新浪数据为空")
            st.stop()

        st.write("**新浪列名**：", raw_data.columns.tolist()[:10])

        data = process_columns(raw_data)
        if not data.empty:
            st.success(f"新浪数据源成功！共 {len(data)} 只股票")
            return data
        else:
            st.error("列处理失败")
            st.stop()
    except Exception as e:
        st.error(f"新浪数据源失败：{e}")
        st.stop()
    return pd.DataFrame()

def process_columns(df):
    data = df.copy()
    data['code'] = data['代码'].astype(str) if '代码' in data.columns else None
    data['name'] = data['名称'] if '名称' in data.columns else None
    data['open'] = pd.to_numeric(data['今开'] if '今开' in data.columns else data['open'], errors='coerce')
    data['close_yesterday'] = pd.to_numeric(data['昨收'] if '昨收' in data.columns else data['pre_close'], errors='coerce')
    data['volume'] = pd.to_numeric(data['成交量'] if '成交量' in data.columns else data['volume'], errors='coerce')

    data = data.dropna(subset=['code', 'open', 'close_yesterday', 'volume'])
    data = data[data['open'] > 0]
    
    return data[['code', 'name', 'open', 'close_yesterday', 'volume']]

if st.button("一键生成全市场潜力股"):
    with st.spinner("正在加载新浪数据源..."):
        data = get_realtime_data()

        data["pure_code"] = data["code"].astype(str).str.extract(r'(\d{6})')[0]
        data["price_momentum"] = (data["open"] - data["close_yesterday"]) / data["close_yesterday"]
        median_vol = data["volume"].median()
        data["volume_ratio"] = data["volume"] / median_vol

        df = data.copy()
        st.write(f"**初始股票数**：{len(df)}")

        df = df[df["open"] > 10]
        st.write(f"**股价 > 10元后**：{len(df)}")

        df = df[df["open"] < df["close_yesterday"] * 1.099]
        st.write(f"**未涨停后**：{len(df)}")

        df = df[df["price_momentum"] > 0.02]  # 涨幅 > 2%（进一步放宽）
        st.write(f"**涨幅 > 2% 后**：{len(df)}")

        df = df[df["volume_ratio"] > 2]  # 放量 > 2倍（进一步放宽）
        st.write(f"**放量 > 2倍后**：{len(df)}")

        if len(df) == 0:
            st.warning("实时过滤后无股票，建议开盘时运行或进一步放宽")
            st.stop()

        # 历史过滤（前 50 只）
        df_top = df.head(50).copy()
        df_top["cumulative_rise_10d"] = 0.0
        df_top["avg_turnover_14d"] = 0.0
        valid = []

        for idx, row in df_top.iterrows():
            hist = get_hist_data(row["code"])
            if len(hist) < 10:
                continue
            rise_10d = (hist.iloc[-1]["close"] / hist.iloc[-11]["close"]) - 1
            df_top.loc[idx, "cumulative_rise_10d"] = rise_10d

            avg_turn = hist["换手率"].tail(14).mean() if "换手率" in hist.columns else 1.0
            df_top.loc[idx, "avg_turnover_14d"] = avg_turn

            if rise_10d <= 0.50 and avg_turn > 1.0:  # 换手 > 1%（放宽）
                valid.append(idx)

        if not valid:
            st.warning("历史过滤后无股票，使用实时结果")
            df_final = df.head(30)
        else:
            df_final = df_top.loc[valid].head(30)

        df_final["score"] = df_final["price_momentum"] * 100 + df_final["volume_ratio"] + df_final["avg_turnover_14d"] * 0.5
        df_final = df_final.sort_values("score", ascending=False)

        st.success(f"找到 {len(df_final)} 只短线王")
        st.dataframe(df_final[["code", "name", "open", "price_momentum", "volume_ratio", "avg_turnover_14d", "score"]])

        csv = df_final.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载CSV", csv, "stocks.csv", "text/csv")
