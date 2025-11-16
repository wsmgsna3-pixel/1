import streamlit as st
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------
# Streamlit 界面
# ---------------------------
st.title("选股王 · 极速版（适配 2100 积分）")

token = st.text_input("请输入你的 Tushare Token（不会上传，很安全）", type="password")
if not token:
    st.stop()

ts.set_token(token)
pro = ts.pro_api()

st.write("正在获取最新交易日…")

# ---------------------------
# 自动获取最近一个交易日
# ---------------------------
today_str = datetime.now().strftime("%Y%m%d")

cal = pro.trade_cal(start_date="20240101", end_date=today_str)
open_days = cal[cal["is_open"] == 1]["cal_date"].tolist()
last_trade_day = open_days[-1]

st.write(f"使用交易日：{last_trade_day}")

# ---------------------------
# 获取昨日全市场日线
# ---------------------------
st.write("获取昨日行情…")

df = pro.daily(trade_date=last_trade_day)

# 排除ST
df = df[~df['ts_code'].str.contains("ST")]

# 价格过滤 10~200 元
df = df[(df["close"] >= 10) & (df["close"] <= 200)]

# 如果过滤后剩很少，直接提示
if df.empty:
    st.error("价格过滤后无股票，请检查。")
    st.stop()

# ---------------------------
# 取昨日涨幅前500
# ---------------------------
df = df.sort_values("pct_chg", ascending=False).head(500)

st.write(f"昨日涨幅前500中，共 {len(df)} 只符合价格区间")

# 若500中可能仍大幅减少，也正常
if df.empty:
    st.warning("昨日涨幅前500中，没有符合条件的股票。")
    st.stop()

# ---------------------------
# 获取均线、成交量等批量数据（极速）
# ---------------------------
st.write("正在获取近20日K线以筛选趋势…（耗时约 1～2 秒）")

codes = df["ts_code"].tolist()

# 批量取K线（不会卡，因为只请求500个）
all_k = []
for code in codes:
    k = pro.daily(ts_code=code, start_date=(datetime.now() - timedelta(days=40)).strftime("%Y%m%d"))
    if k is not None and len(k) > 0:
        all_k.append(k)

if len(all_k) == 0:
    st.error("未能获取 K 线数据，请稍后重试")
    st.stop()

full_k = pd.concat(all_k)

# 按股票分组求 MA5 & MA10
def calc_ma(group):
    group = group.sort_values("trade_date")
    group["ma5"] = group["close"].rolling(5).mean()
    group["ma10"] = group["close"].rolling(10).mean()
    return group

full_k = full_k.groupby("ts_code").apply(calc_ma)

# 取最后一天的数据（即昨日）
latest_k = full_k.groupby("ts_code").tail(1)

# ---------------------------
# 趋势过滤：MA5 > MA10（短期趋势向上）
# ---------------------------
result = latest_k[
    (latest_k["ma5"] > latest_k["ma10"]) &
    (latest_k["close"] >= 10) &
    (latest_k["close"] <= 200)
]

# ---------------------------
# 输出最终结果
# ---------------------------
st.subheader("选股结果")

if result.empty:
    st.warning("筛选完成，0 只股票（可能是市场整体疲弱导致 MA 条件无法满足）")
else:
    st.success(f"筛选完成，共 {len(result)} 只股票")
    st.dataframe(result[["ts_code", "close", "pct_chg", "ma5", "ma10"]])

    # 下载 CSV
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("下载结果 CSV", csv, "stock_result.csv", "text/csv")
