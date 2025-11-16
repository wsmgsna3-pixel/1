import streamlit as st
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="短线王·权限兼容版", layout="wide")

# =========================
# 0. 初始化
# =========================
ts.set_token(st.secrets["TS_TOKEN"])
pro = ts.pro_api()

st.title("短线王选股器（2100积分兼容版）")
st.write("所有字段均已自动适配你的权限，不调用 daily_basic / moneyflow / 市值 / 行业。")

# =========================
# 1. 参数区域
# =========================
col1, col2, col3, col4 = st.columns(4)

min_price = col1.number_input("最低股价", 1.0, 200.0, 3.0)
max_price = col2.number_input("最高股价", 1.0, 500.0, 80.0)
min_amount = col3.number_input("最低成交额（万元）", 0.0, 500000.0, 8000.0)
top_n = col4.number_input("输出前 N 名", 1, 200, 30)

days = st.slider("计算周期（天）", 5, 30, 15)

# =========================
# 2. 获取股票基本信息（低权限字段）
# =========================
@st.cache_data
def load_stock_basic():
    df = pro.stock_basic(fields="ts_code,symbol,name")
    return df

stock_basic_df = load_stock_basic()

# =========================
# 3. 拉取最近 N 天全部日线
# =========================
@st.cache_data
def load_recent_daily(days=15):
    end = datetime.now()
    start = end - timedelta(days=days*2)  # 留余量避免周末
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    df = pro.daily(start_date=start_str, end_date=end_str)
    df = df.sort_values(["ts_code", "trade_date"])
    return df

daily_df = load_recent_daily(days)

# =========================
# 4. 过滤当天可交易股票
# =========================
today = daily_df["trade_date"].max()
today_df = daily_df[daily_df["trade_date"] == today].copy()

# 价格过滤
today_df = today_df[(today_df["close"] >= min_price) & (today_df["close"] <= max_price)]

# 成交额过滤
today_df = today_df[today_df["amount"] >= min_amount]

if today_df.empty:
    st.error("无符合条件的股票，请放宽条件")
    st.stop()

# =========================
# 5. 计算技术指标
# =========================
result_list = []

for ts_code in today_df["ts_code"]:
    df = daily_df[daily_df["ts_code"] == ts_code].tail(days)

    if len(df) < days:
        continue

    # 涨幅（总涨幅）
    pct_total = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    # 量比评分：今日成交量 / 均量
    vol_ratio = df["vol"].iloc[-1] / max(df["vol"].mean(), 1)

    # 成交活跃度：今日 amount / 均 amount
    act_ratio = df["amount"].iloc[-1] / max(df["amount"].mean(), 1)

    score = (
        pct_total * 0.4 +
        vol_ratio * 2 +
        act_ratio * 0.3 +
        df["pct_chg"].iloc[-1] * 0.3
    )

    result_list.append({
        "ts_code": ts_code,
        "name": stock_basic_df.loc[stock_basic_df.ts_code == ts_code, "name"].values[0],
        "close": df["close"].iloc[-1],
        "pct_chg_today": df["pct_chg"].iloc[-1],
        "pct_total": pct_total,
        "vol_ratio": vol_ratio,
        "act_ratio": act_ratio,
        "score": score
    })

# =========================
# 6. 输出结果
# =========================
res = pd.DataFrame(result_list)
res = res.sort_values("score", ascending=False).head(top_n)

st.subheader(f"今日结果（前 {top_n} 名）")
st.dataframe(res)

st.success("运行成功！所有字段均已自动适配你的实际权限，不会报 KeyError。")
