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
# 获取最近交易日（简单推算）
# ==============================
def get_last_trade_day():
    today = datetime.now()
    if today.weekday() == 5:       # 周六
        last_trade_day = today - timedelta(days=1)
    elif today.weekday() == 6:     # 周日
        last_trade_day = today - timedelta(days=2)
    else:                          # 工作日取前一天
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
        row["10d_avg_turnover"] = df["pct_chg"].abs().mean()
        return row
    except:
        return None

# ==============================
# 筛选函数
# ==============================
def select_stocks(df, vol_multiplier=1.5, open_multiplier=0.3, fallback=False):
    result = []
    for i, (idx, row) in enumerate(df.iterrows()):
        ts_code = row["ts_code"]
        hist = get_10d(ts_code)
        if not hist:
            continue

        try:
            cond1 = row["open"] > 10
            cond2 = hist["10d_return"] <= 0.50
            cond3 = hist["10d_avg_turnover"] >= 3
            cond4 = row["vol"] > hist["volume_yesterday"] * vol_multiplier
            if not fallback:
                cond5 = row["open"] >= hist["high_yesterday"] * open_multiplier
            else:
                # 放宽条件：开盘价 ≥ 昨日收盘价
                cond5 = row["open"] >= row["pre_close"]
        except:
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
    return result

# ==============================
# 主入口
# ==============================
if st.button("一键生成短线王"):
    with st.spinner("正在获取 A 股行情..."):
        df = get_today_data(last_trade_day)
    st.write(f"获取到 {len(df)} 条行情数据")  # 输出调试信息

    if df is None or df.empty:
        st.error("未获取到行情数据")
        st.stop()

    df = df.head(300)

    # 首轮筛选
    result = select_stocks(df, vol_multiplier=1.5, open_multiplier=0.3)

    # 自动放宽条件
    if not result:
        st.warning("首次筛选无候选，自动放宽成交量和开盘价条件...")
        result = select_stocks(df, vol_multiplier=1.0, open_multiplier=0.0, fallback=True)

    if not result:
        st.error("即便放宽条件，仍未找到符合条件的短线王")
        st.stop()

    result_df = pd.DataFrame(result).sort_values("score", ascending=False)
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载 CSV", data=csv, file_name="短线王.csv", mime="text/csv")
