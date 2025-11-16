import streamlit as st
import pandas as pd
import tushare as ts
import datetime

# ---------------------------
# 初始化
# ---------------------------

st.set_page_config(page_title="选股系统（含过滤统计）", layout="wide")
ts.set_token(st.secrets["tushare"]["token"])
pro = ts.pro_api()

today = datetime.datetime.now().strftime("%Y%m%d")
st.write(f"当日日期：{today}")

# ---------------------------
# 读取数据
# ---------------------------

@st.cache_data
def get_daily(date):
    df = pro.daily(trade_date=date)
    st.write(f"📌 daily 记录数：{len(df)}")
    return df

@st.cache_data
def get_daily_basic(date):
    df = pro.daily_basic(trade_date=date, fields="ts_code,turnover_rate,circ_mv")
    st.write(f"📌 daily_basic 记录数：{len(df)}")
    return df

# 主数据
df = get_daily(today)
df_db = get_daily_basic(today)

if df is None or len(df) == 0:
    st.error("❌ 今日没有 daily 数据")
    st.stop()

# ---------------------------
# 合并 daily_basic（修复版）
# ---------------------------

db_needed = ["ts_code", "turnover_rate", "circ_mv"]
db_exist = [c for c in db_needed if c in df_db.columns]

if len(db_exist) < len(db_needed):
    missing = set(db_needed) - set(db_exist)
    st.warning(f"⚠️ daily_basic 缺少字段：{missing}（已自动跳过缺失字段）")

if "ts_code" in df_db.columns:
    df = df.merge(df_db[db_exist], on="ts_code", how="left")
else:
    st.warning("⚠️ daily_basic 缺少 ts_code，跳过合并")

# ---------------------------
# 过滤统计工具
# ---------------------------

def step_filter(df, cond, name):
    before = len(df)
    df = df[cond]
    after = len(df)
    st.write(f"➡️ {name}： {before} → {after}")
    return df

st.header("📊 过滤过程统计")

# ---------------------------
# Step 1：涨停 or 跌停剔除
# ---------------------------

df = step_filter(df, (df["pct_chg"] < 9.9) & (df["pct_chg"] > -9.9), "剔除涨跌停")

# ---------------------------
# Step 2：开盘价过滤
# ---------------------------

df = step_filter(df, df["open"] > 1, "开盘价 > 1")

# ---------------------------
# Step 3：成交额过滤（使用 daily 的 amount，不会缺）
# ---------------------------

df = step_filter(df, df["amount"] > 1_000_000, "成交额 > 100万")

# ---------------------------
# Step 4：市值过滤
# ---------------------------

if "circ_mv" in df.columns:
    df = step_filter(df, df["circ_mv"] < 800, "流通市值 < 800亿")
else:
    st.warning("⚠️ circ_mv 缺失，跳过市值过滤")

# ---------------------------
# Step 5：换手率过滤
# ---------------------------

if "turnover_rate" in df.columns:
    df = step_filter(df, df["turnover_rate"] > 0.5, "换手率 > 0.5%")
else:
    st.warning("⚠️ turnover_rate 缺失，跳过换手率过滤")

# ---------------------------
# Step 6：最终排序
# ---------------------------

df = df.sort_values(by="amount", ascending=False)

st.header("📈 最终选股结果")

if len(df) == 0:
    st.error("❌ 没有选出股票，请适当降低筛选参数。")
else:
    st.success(f"🎉 共选出 {len(df)} 只股票")
    st.dataframe(df)
