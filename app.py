import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="选股王（极速版）", layout="wide")
st.title("🔥 极速版选股王（2100 积分专属优化）")

# -------------------------------
# 1. 手动输入 Token
# -------------------------------
token = st.text_input("请输入 TuShare Token（不会保存，非常安全）", type="password")

if not token:
    st.info("输入 Token 后开始选股。")
    st.stop()

ts.set_token(token)
pro = ts.pro_api(token)

# -------------------------------
# 2. 日期区间
# -------------------------------
today = datetime.today()
yesterday = (today - timedelta(days=1)).strftime("%Y%m%d")
start_60 = (today - timedelta(days=120)).strftime("%Y%m%d")  # 用 120 天够算 MA60

# -------------------------------
# 3. 批量拉取全市场日线 —— 关键优化！
# -------------------------------
st.write("📡 正在批量获取行情（不会卡，请稍候几秒）...")

df_daily = pro.daily(start_date=start_60, end_date=yesterday)
df_daily.sort_values(["ts_code", "trade_date"], inplace=True)

# -------------------------------
# 4. 股票基本信息
# -------------------------------
df_basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name")

# 合并
df = df_daily.merge(df_basic, on="ts_code", how="left")

# -------------------------------
# 5. 价格过滤（你自定义）
# -------------------------------
# 最新一天的收盘价
last_day = df[df.trade_date == df.trade_date.max()]
last_day = last_day[(last_day["close"] >= 10) & (last_day["close"] <= 200)]
last_codes = last_day.ts_code.unique()

df = df[df.ts_code.isin(last_codes)]

# -------------------------------
# 6. 计算涨幅、均线、量能等全部指标（批量计算，不循环）
# -------------------------------
df["pct_chg"] = df.groupby("ts_code")["close"].pct_change() * 100
df["vol_ma5"] = df.groupby("ts_code")["vol"].rolling(5).mean().reset_index(0, drop=True)
df["vol_ma10"] = df.groupby("ts_code")["vol"].rolling(10).mean().reset_index(0, drop=True)
df["ma20"] = df.groupby("ts_code")["close"].rolling(20).mean().reset_index(0, drop=True)
df["ma60"] = df.groupby("ts_code")["close"].rolling(60).mean().reset_index(0, drop=True)

# -------------------------------
# 7. 取昨日的所有数据
# -------------------------------
df_y = df[df.trade_date == df.trade_date.max()].copy()

# -------------------------------
# 8. 昨日涨幅前 500 名
# -------------------------------
df_top = df_y.sort_values("pct_chg", ascending=False).head(500)

# -------------------------------
# 9. 高级策略过滤（批量，不循环接口）
# -------------------------------
df_sel = df_top[
    (df_top["vol"] > df_top["vol_ma5"]) &          # 放量
    (df_top["close"] > df_top["ma20"]) &          # 收盘价站上20日均线
    (df_top["ma20"] > df_top["ma60"])             # 20日线上穿60日（趋势向上）
]

st.success(f"筛选完成，共 {len(df_sel)} 只股票")

st.dataframe(
    df_sel[["ts_code", "name", "close", "pct_chg", "vol", "vol_ma5", "ma20", "ma60"]],
    height=600
)
