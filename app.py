import streamlit as st
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

# ==============================
# 页面配置
# ==============================
st.set_page_config(page_title="短线王（Tushare 版）", layout="wide")
st.title("短线王（Tushare 极速 300 只股票版）")

# ==============================
# 运行时输入 Tushare Token
# ==============================
TS_TOKEN = st.text_input("请输入你的 Tushare Token", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 后才能运行")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ==============================
# 获取最近交易日（周末/节假日安全）
# ==============================
def get_last_trade_day():
    today = datetime.now()
    # 周六
    if today.weekday() == 5:
        last_trade_day = today - timedelta(days=1)
    # 周日
    elif today.weekday() == 6:
        last_trade_day = today - timedelta(days=2)
    # 平日
    else:
        last_trade_day = today - timedelta(days=1)
    return last_trade_day.strftime("%Y%m%d")

last_trade_day = get_last_trade_day()
st.info(f"当前使用最近交易日: {last_trade_day}")

# ==============================
# 拉取当天行情（限制 300 只）
# ==============================
@st.cache_data(ttl=60)
def get_today_data(trade_date):
    try:
        df = pro.daily(trade_date=trade_date)
    except:
        df = pd.DataFrame()
    return df

# ==============================
# 拉取最近 10 根 K 线历史数据
# ==============================
def get_10d(ts_code):
    try:
        end_date = last_trade_day
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or len(df) < 10:
            return None
        df = df.sort_values("trade_date").tail(10)
        row = {}
        row["10d_return"] = df.iloc[-1]["close"] / df.iloc[0]["open"] - 1
        row["volume_yesterday"] = df.iloc[-2]["vol"]
        row["high_yesterday"] = df.iloc[-2]["high"]
        row["10d_avg_turnover"] = df["pct_chg"].abs().mean()  # 近似指标
        return row
    except:
        return None

# ==============================
# 主入口
# ==============================
if st.button("一键生成短线王"):
    with st.spinner("正在获取 A 股行情..."):
        df = get_today_data(last_trade_day)

    if df is None or df.empty:
        st.error("未获取到行情数据")
        st.stop()

    df = df.head(300)  # 限制 300 只股票

    result = []
    progress = st.progress(0)

    for i, (idx, row) in enumerate(df.iterrows()):
        ts_code = row["ts_code"]
        hist = get_10d(ts_code)
        if not hist:
            progress.progress((i+1)/len(df))
            continue

        # 五条选股条件（开盘价条件改为 ≥ 昨日最高价的 70%）
        try:
            cond1 = row["open"] > 10
            cond2 = hist["10d_return"] <= 0.50
            cond3 = hist["10d_avg_turnover"] >= 3
            cond4 = row["vol"] > hist["volume_yesterday"] * 3
            cond5 = row["open"] >= hist["high_yesterday"] * 0.7  # 改动
        except:
            progress.progress((i+1)/len(df))
            continue

        if cond1 and cond2 and cond3 and cond4 and cond5:
            score = (row["open"] - row["pre_close"]) / row["pre_close"] * 100 \
                    + row["vol"] / hist["volume_yesterday"] * 10
            result.append({
                "ts_code": ts_code,
                "name": row["ts_code"].split(".")[0],
                "open": row["open"],
                "10d_return": round(hist["10d_return"], 4),
                "10d_avg_turnover": round(hist["10d_avg_turnover"], 2),
                "volume_yesterday": hist["volume_yesterday"],
                "volume_today": row["vol"],
                "score": round(score, 2)
            })

        progress.progress((i+1)/len(df))

    if not result:
        st.warning("未找到符合条件的短线王")
        st.stop()

    result_df = pd.DataFrame(result).sort_values("score", ascending=False)
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载 CSV", data=csv, file_name="短线王.csv", mime="text/csv")
