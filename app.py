import streamlit as st
import tushare as ts
import pandas as pd
from datetime import datetime

st.title("Tushare 短线王（1-5天操作）")

# ==================== 输入你的 Token ====================
TUSHARE_TOKEN = st.text_input("请输入你的 Tushare Token", type="password", value="")
if not TUSHARE_TOKEN:
    st.warning("请输入 Token 后点击下方按钮运行")
    st.stop()

# 初始化 Tushare
pro = ts.pro_api(TUSHARE_TOKEN)

# ==================== 交易时间检测 ====================
def is_trading_time():
    now = datetime.now()
    if now.weekday() >= 5: return False
    hour, minute = now.hour, now.minute
    if 9 <= hour < 11 or (hour == 11 and minute <= 30):
        return True
    if 13 <= hour < 15:
        return True
    return False

if not is_trading_time():
    st.warning("当前非交易时间，数据为昨收价，建议 9:30-15:00 运行")

# ==================== 获取实时行情 ====================
@st.cache_data(ttl=300)  # 5分钟缓存
def get_realtime_data():
    try:
        # 获取实时行情（含今开、昨收、成交量）
        df = pro.realtime_quote(fields='ts_code,name,open,pre_close,volume,turnover_rate')
        if df.empty:
            st.error("数据为空")
            return None
        
        # 转换代码格式
        df['code'] = df['ts_code'].str.replace('.SH', 'sh').str.replace('.SZ', 'sz').str.lower()
        df['close_yesterday'] = df['pre_close']
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

        # 你的核心过滤条件
        df = df[df["open"] > 10]                                      # 股价 > 10元
        st.write(f"**股价 > 10元**：{len(df)}")

        df = df[df["open"] < df["close_yesterday"] * 1.099]           # 未涨停
        st.write(f"**未涨停**：{len(df)}")

        df = df[df["price_momentum"] > 0.02]                          # 开盘涨幅 > 2%
        st.write(f"**涨幅 > 2%**：{len(df)}")

        df = df[df["volume_ratio"] > 2]                               # 放量 > 2倍
        st.write(f"**放量 > 2倍**：{len(df)}")

        if len(df) == 0:
            st.warning("无符合条件股票，建议放宽条件或开盘运行")
            st.stop()

        # 历史数据：10天涨幅 + 换手率
        df["cumulative_rise_10d"] = 0.0
        df["avg_turnover_14d"] = 0.0
        valid = []

        for idx, row in df.head(50).iterrows():  # 前50只
            try:
                hist = pro.daily(ts_code=row['ts_code'], start_date='', end_date='', fields='close,turnover_rate')
                if len(hist) < 10:
                    continue
                rise_10d = (hist['close'].iloc[-1] / hist['close'].iloc[-11]) - 1
                avg_turn = hist['turnover_rate'].tail(14).mean()
                df.loc[idx, "cumulative_rise_10d"] = rise_10d
                df.loc[idx, "avg_turnover_14d"] = avg_turn
                if rise_10d <= 0.50 and avg_turn > 1.0:
                    valid.append(idx)
            except:
                continue

        # 兜底：用实时结果
        if not valid:
            st.warning("历史过滤无结果，使用实时前30只")
            df_final = df.head(30).copy()
            df_final["avg_turnover_14d"] = df["turnover_rate"]
        else:
            df_final = df.loc[valid].head(30)

        # 打分
        df_final["score"] = (
            df_final["price_momentum"] * 100 +
            df_final["volume_ratio"] * 10 +
            df_final["avg_turnover_14d"] * 0.5
        )
        df_final = df_final.sort_values("score", ascending=False)

        st.success(f"找到 {len(df_final)} 只短线王！")
        st.dataframe(df_final[["code", "name", "open", "price_momentum", "volume_ratio", "avg_turnover_14d", "score"]])

        # 下载
        csv = df_final.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载CSV", csv, "tushare_kings.csv", "text/csv")
