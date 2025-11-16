import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
import datetime

# ===========================
# 配置 Tushare
# ===========================
TS_TOKEN = os.getenv("TS_TOKEN")
if TS_TOKEN is None:
    raise ValueError("没有读取到 TS_TOKEN，请先在系统环境变量里配置。")

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ===========================
# 基础函数：安全获取
# ===========================
def safe_get(api_func, **kwargs):
    try:
        df = api_func(**kwargs)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

# ===========================
# 获取每日行情 + 涨幅榜前1000
# ===========================
def get_daily_data(date):
    df = safe_get(pro.daily, trade_date=date)
    if df.empty:
        return df
    return df.sort_values("pct_chg", ascending=False).head(1000)

# ===========================
# 获取 daily_basic（换手率、市值）
# ===========================
def get_daily_basic(date):
    return safe_get(pro.daily_basic, trade_date=date)

# ===========================
# 获取主力净流
# ===========================
def try_get_moneyflow(date):
    df = safe_get(pro.moneyflow, trade_date=date)
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "net_mf"])
    df["net_mf"] = df["net_mf"].fillna(0)
    return df[["ts_code", "net_mf"]]

# ===========================
# 评分函数
# ===========================
def score_stock(row):
    score = 0

    # 涨幅
    if "pct_chg" in row and not np.isnan(row["pct_chg"]):
        score += row["pct_chg"] * 1.0

    # 换手率
    if "turnover_rate" in row and not np.isnan(row["turnover_rate"]):
        score += row["turnover_rate"] * 0.8

    # 成交额
    if "amount" in row and not np.isnan(row["amount"]):
        score += np.log1p(row["amount"]) * 1.2

    # 主力净流
    if "net_mf" in row and not np.isnan(row["net_mf"]):
        score += row["net_mf"] * 0.5

    # 控制流通市值（越小越好）
    if "circ_mv" in row and not np.isnan(row["circ_mv"]):
        score += max(0, 50 - np.log1p(row["circ_mv"]))

    return score

# ===========================
# Streamlit UI
# ===========================

st.title("📈 简洁版 · 评分制选股王（自动版）")

# 默认取最近一个交易日（自动兼容周末）
today = datetime.datetime.now().date()
offset = 0
while True:
    d = today - datetime.timedelta(days=offset)
    trade_date = d.strftime("%Y%m%d")
    df = safe_get(pro.daily, trade_date=trade_date)
    if not df.empty:
        break
    offset += 1

st.write(f"当前使用交易日：**{trade_date}**")

# ===========================
# 获取全部数据
# ===========================

daily_df = get_daily_data(trade_date)
basic_df = get_daily_basic(trade_date)
money_df = try_get_moneyflow(trade_date)

if daily_df.empty:
    st.error("Tushare 无法获取当日数据。请稍后再试。")
    st.stop()

# 合并：使用 ts_code 左连接，缺什么补什么
pool = daily_df.copy().set_index("ts_code")

def safe_merge(df, cols):
    if df.empty:
        for c in cols:
            pool[c] = np.nan
        return
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    pool[cols] = df.set_index("ts_code")[cols]

# 合并 daily_basic
safe_merge(basic_df, ["turnover_rate", "amount", "circ_mv"])

# 合并 moneyflow
safe_merge(money_df, ["net_mf"])

pool = pool.reset_index()

# ===========================
# 评分
# ===========================
pool["score"] = pool.apply(score_stock, axis=1)
pool = pool.sort_values("score", ascending=False).head(20)

# ===========================
# 展示结果
# ===========================
st.subheader("今日推荐股票 TOP 20")
st.dataframe(pool[["ts_code", "pct_chg", "turnover_rate", "amount", "net_mf", "circ_mv", "score"]])
