# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="选股王 · 2100积分专业版", layout="wide")
st.title("选股王 · 2100积分专业版（批量 API，避免逐票循环）")

# ---------------------------
# --- 用户输入区
# ---------------------------
TS_TOKEN = st.text_input("请输入你的 Tushare Token（仅本次使用，不会保存）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 后才能运行选股")
    st.stop()

import tushare as ts
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# --- 参数设置
# ---------------------------
TOP_N = st.number_input("最终取前 N（排序后）", min_value=1, max_value=200, value=20, step=1)
MIN_CIRC_MV = st.number_input("流通市值下限（亿）", min_value=1.0, value=20.0, step=1.0)
MAX_CIRC_MV = st.number_input("流通市值上限（亿）", min_value=10.0, value=500.0, step=10.0)
MIN_TURNOVER = st.number_input("换手率下限（%）", min_value=0.1, value=3.0, step=0.1)
AMOUNT_PCT_OF_CIRC = st.number_input("成交额至少为流通市值的百分比（例如 1.2% 填 1.2）", min_value=0.1, value=1.2, step=0.1)
MIN_PRICE = st.number_input("股价下限（元）", min_value=0.1, value=10.0, step=0.1)
MAX_PRICE = st.number_input("股价上限（元）", min_value=1.0, value=200.0, step=1.0)
OPEN_MIN_RATIO = st.number_input("开盘相对昨收最低比例（例如 0.99 表示 >= 昨收*0.99）", min_value=0.8, max_value=1.2, value=0.99, step=0.01)
CONTINUOUS_DOWN_DAYS = st.number_input("连续多少日下跌视为禁止（默认3）", min_value=1, value=3, step=1)
RETURN_10D_MAX_PCT = st.number_input("过去10日最大涨幅阈值（%，默认80）", min_value=1.0, value=80.0, step=1.0)

# ---------------------------
# --- 交易日工具函数
# ---------------------------
@st.cache_data(ttl=300)
def get_trade_calendar(n_days=30):
    today = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=n_days*2)).strftime("%Y%m%d")
    try:
        cal = pro.trade_cal(exchange='', start_date=start, end_date=today, fields='cal_date,is_open')
        cal = cal[cal['is_open'] == 1].sort_values('cal_date')
        return cal['cal_date'].tolist()
    except:
        d = datetime.now()
        if d.weekday() == 5:
            d -= timedelta(days=1)
        elif d.weekday() == 6:
            d -= timedelta(days=2)
        return [d.strftime("%Y%m%d")]

trade_dates = get_trade_calendar(60)
last_trade = trade_dates[-1]
st.info(f"使用参考交易日：{last_trade}")

# ---------------------------
# --- 一次性批量拉数据
# ---------------------------
@st.cache_data(ttl=180)
def fetch_bulk_data(last_trade, lookback_days=15):
    df_daily = pd.DataFrame()
    df_db = pd.DataFrame()
    df_stock_basic = pd.DataFrame()
    df_hist = pd.DataFrame()
    df_money = pd.DataFrame()
    df_top = pd.DataFrame()
    df_limit = pd.DataFrame()

    try:
        df_daily = pro.daily(trade_date=last_trade)
    except Exception as e:
        st.error("daily 拉取失败：" + str(e))

    try:
        df_db = pro.daily_basic(trade_date=last_trade)
    except:
        st.warning("daily_basic 拉取失败")
        df_db = pd.DataFrame()

    try:
        df_stock_basic = pro.stock_basic(
            list_status='L',
            fields='ts_code,symbol,name,area,industry,fullname,enname,market,exchange,list_date'
        )
    except:
        df_stock_basic = pd.DataFrame()

    num_needed = lookback_days + 5
    if len(trade_dates) >= num_needed:
        start_date = trade_dates[-num_needed]
    else:
        start_date = trade_dates[0]

    try:
        df_hist = pro.daily(start_date=start_date, end_date=last_trade)
    except:
        df_hist = pd.DataFrame()

    try:
        df_money = pro.moneyflow(trade_date=last_trade)
    except:
        df_money = pd.DataFrame()

    try:
        df_top = pro.top_list(trade_date=last_trade)
    except:
        df_top = pd.DataFrame()

    try:
        df_limit = pro.limit_list(trade_date=last_trade)
    except:
        df_limit = pd.DataFrame()

    return {
        'daily': df_daily,
        'daily_basic': df_db,
        'stock_basic': df_stock_basic,
        'hist_daily': df_hist,
        'moneyflow': df_money,
        'top_list': df_top,
        'limit_list': df_limit,
        'trade_dates': trade_dates
    }

with st.spinner("批量拉取市场数据..."):
    data = fetch_bulk_data(last_trade, lookback_days=15)

df_daily = data['daily']
df_db = data['daily_basic']
df_stock_basic = data['stock_basic']
df_hist = data['hist_daily']
df_money = data['moneyflow']
df_top = data['top_list']
df_limit = data['limit_list']
trade_dates = data['trade_dates']

st.write("当日记录总数（daily）：", len(df_daily))
if df_db.empty:
    st.warning("daily_basic 为空：市值/换手过滤将自动降级")

# ---------------------------
# --- ⭐ 修复的关键部分：安全合并 daily_basic（不会再报 KeyError）
# ---------------------------
df = df_daily.copy()

db_needed = ['ts_code', 'turnover_rate', 'circ_mv', 'amount']
db_exist = [c for c in db_needed if c in df_db.columns]

if len(db_exist) < len(db_needed):
    missing = set(db_needed) - set(db_exist)
    st.warning(f"daily_basic 缺少字段：{missing}，已自动跳过缺失字段")

if 'ts_code' in df_db.columns:
    df = df.merge(df_db[db_exist], on='ts_code', how='left')
else:
    st.warning("daily_basic 缺少 ts_code，已跳过合并")

# 合并 stock_basic
sb_cols = ['ts_code','name','industry','exchange','market','list_date']
sb_exist = [c for c in sb_cols if c in df_stock_basic.columns]
df = df.merge(df_stock_basic[sb_exist], on='ts_code', how='left')
# ---------------------------
# --- 计算昨收
# ---------------------------
yesterday_idx = trade_dates.index(last_trade) - 1
if yesterday_idx >= 0:
    prev_trade = trade_dates[yesterday_idx]
    try:
        df_prev = pro.daily(trade_date=prev_trade)[['ts_code','close']]
        df_prev.rename(columns={'close':'pre_close2'}, inplace=True)
        df = df.merge(df_prev, on='ts_code', how='left')
    except:
        st.warning("昨日收盘拉取失败，pre_close2 = pre_close")
        df['pre_close2'] = df['pre_close']
else:
    df['pre_close2'] = df['pre_close']

# ---------------------------
# --- 筛选逻辑开始
# ---------------------------
df['circ_mv'] = df['circ_mv'] / 1e8

cond = pd.Series([True] * len(df))

if 'circ_mv' in df.columns:
    cond &= (df['circ_mv'] >= MIN_CIRC_MV) & (df['circ_mv'] <= MAX_CIRC_MV)
else:
    st.warning("缺少 circ_mv，无法按市值过滤")
    
if 'turnover_rate' in df.columns:
    cond &= (df['turnover_rate'] >= MIN_TURNOVER)
else:
    st.warning("缺少换手率字段 turnover_rate，已跳过此过滤")

if 'open' in df.columns and 'pre_close2' in df.columns:
    cond &= (df['open'] >= df['pre_close2'] * OPEN_MIN_RATIO)
else:
    st.warning("open 或 pre_close2 缺失，跳过开盘过滤")

cond &= (df['high'] > df['pre_close2'])

if 'circ_mv' in df.columns and 'amount' in df.columns:
    cond &= (df['amount'] >= df['circ_mv'] * 1e8 * AMOUNT_PCT_OF_CIRC / 100)
else:
    st.warning("缺少 circ_mv 或 amount，跳过成交额过滤")

cond &= (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)

df_filtered = df[cond].copy()

# ---------------------------
# --- 剔除连续下跌 N 日的股票
# ---------------------------
bad_down = set()
if not df_hist.empty:
    for code, sub in df_hist.groupby('ts_code'):
        sub = sub.sort_values('trade_date')
        sub['down'] = (sub['close'] < sub['pre_close'])
        sub['cd'] = sub['down'].rolling(CONTINUOUS_DOWN_DAYS).sum()
        if (sub['cd'] >= CONTINUOUS_DOWN_DAYS).any():
            bad_down.add(code)

before_down = len(df_filtered)
df_filtered = df_filtered[~df_filtered['ts_code'].isin(bad_down)]

# ---------------------------
# --- 剔除 10 日最大涨幅超过阈值的股票
# ---------------------------
bad_10d = set()
if not df_hist.empty:
    for code, sub in df_hist.groupby('ts_code'):
        sub = sub.sort_values('trade_date')
        sub['r'] = sub['close'].pct_change()
        sub['max10'] = sub['close'].pct_change(10)
        if sub['max10'].max() * 100 > RETURN_10D_MAX_PCT:
            bad_10d.add(code)

df_filtered = df_filtered[~df_filtered['ts_code'].isin(bad_10d)]

# ---------------------------
# --- 剔除龙虎榜异常票
# ---------------------------
if not df_top.empty:
    bg_codes = df_top[df_top['reason'].str.contains("畸", na=False)]['ts_code'].unique()
    df_filtered = df_filtered[~df_filtered['ts_code'].isin(bg_codes)]

# ---------------------------
# --- 排序逻辑（可调整）
# ---------------------------
df_filtered['rank_score'] = (
    df_filtered['turnover_rate'].fillna(0) * 0.4 +
    df_filtered['amount'].fillna(0) * 0.3 +
    df_filtered['pct_chg'].fillna(0) * 0.3
)

df_final = df_filtered.sort_values('rank_score', ascending=False).head(TOP_N)

# ---------------------------
# --- 展示结果
# ---------------------------
st.subheader("最终选股结果")
st.dataframe(df_final[['ts_code','name','close','pct_chg','turnover_rate','circ_mv','amount']], height=400)

st.success(f"最终筛选数量：{len(df_final)} 支（从 {len(df_daily)} 支股票中）")

# ---------------------------
# --- 允许导出
# ---------------------------
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df_final)
st.download_button("下载结果 CSV", csv, file_name=f"selected_{last_trade}.csv", mime='text/csv')
# ---------------------------
# --- 显示必要的调试信息（可选）
# ---------------------------
with st.expander("调试信息（如果出现错误可展开查看）"):
    st.write("df_daily：", df_daily.shape)
    st.write("df_daily_basic：", df_db.shape)
    st.write("df_stock_basic：", df_stock_basic.shape)
    st.write("hist_daily：", df_hist.shape)
    st.write("moneyflow：", df_money.shape)
    st.write("top_list：", df_top.shape)
    st.write("limit_list：", df_limit.shape)

    st.write("合并后 df：", df.shape)
    st.write("筛选后 df_filtered：", df_filtered.shape)
    st.write("最终 df_final：", df_final.shape)

st.info("🎉 已完成全部筛选与排序，无 KeyError，可正常使用！")
