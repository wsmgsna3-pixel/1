import streamlit as st
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="选股王 v4", layout="wide")
st.title("选股王 · 双模核武库 v4")
st.caption("2100积分 | 日期永不错 | 一键出肉")

user_token = st.text_input("Tushare Token", type="password")
if not user_token:
    st.stop()

pro = ts.pro_api(user_token)

col1, col2 = st.columns([1, 3])
with col1:
    mode = st.radio("模式", ["核弹模式", "狙击枪模式"], index=1)
with col2:
    st.markdown("**狙击枪模式** 8~20 只主升浪" if mode == "狙击枪模式" else "**核弹模式** 0~5 只妖股")

# 强制使用最新可用交易日（避免权限回退）
last_trade_day = "20251114"  # 周五
yesterday = "20251113"       # 周四

@st.cache_data(ttl=3600)
def run_selection():
    try:
        top_n = 500 if mode == "核弹模式" else 1000
        vol_ratio = 1.3 if mode == "核弹模式" else 1.25
        amt_thr = 100 if mode == "核弹模式" else 50
        gold_days = 1 if mode == "核弹模式" else 2

        # 最新数据
        daily_basic = pro.daily_basic(trade_date=last_trade_day, fields='ts_code,close,total_mv')
        if daily_basic.empty: return pd.DataFrame()

        stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        if stock_basic.empty: return pd.DataFrame()

        df = daily_basic.merge(stock_basic, on='ts_code').rename(columns={'close': 'price'})

        # 初筛
        df = df[
            (~df['name'].str.contains('ST', na=False)) &
            (~df['name'].str.contains(r'\*ST', na=False, regex=True)) &
            (~df['ts_code'].str.startswith('8')) &
            (~df['ts_code'].str.startswith('4')) &
            (df['price'] >= 10) & (df['price'] <= 200) &
            (df['total_mv'] >= 100000) & (df['total_mv'] <= 50000000)
        ].copy()

        if df.empty: return pd.DataFrame()

        # 昨日涨幅
        daily = pro.daily(trade_date=yesterday, fields='ts_code,pct_chg')
        daily = daily[daily['pct_chg'] < 9.8]
        daily = daily[daily['ts_code'].isin(df['ts_code'])]
        if daily.empty: return pd.DataFrame()

        top_codes = daily.nlargest(top_n, 'pct_chg')['ts_code'].tolist()
        df = df[df['ts_code'].isin(top_codes)].merge(daily[['ts_code','pct_chg']], on='ts_code')

        # 拉日线
        codes = ','.join(df['ts_code'].tolist())
        start = "20250901"
        data = pro.daily(ts_code=codes, start_date=start, end_date=last_trade_day)
        if data.empty: return pd.DataFrame()

        data = data.merge(df[['ts_code','name','industry']], on='ts_code').sort_values(['ts_code','trade_date'])

        # 指标
        data['ma5'] = data.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
        data['ma10'] = data.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
        data['ma20'] = data.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
        data['vol_ma5'] = data.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5).mean())

        latest = data.groupby('ts_code').tail(1).copy()
        prev = data.groupby('ts_code').apply(lambda x: x.tail(2).iloc[:-1] if len(x)>=2 else pd.DataFrame()).reset_index(drop=True)
        prev = prev[['ts_code','ma5','ma10']].rename(columns={'ma5':'ma5_prev','ma10':'ma10_prev'})
        latest = latest.merge(prev, on='ts_code', how='left')

        # 条件
        c1 = latest['ma5'] > latest['ma10']
        c2 = (latest['ma5_prev'] <= latest['ma10_prev']) if gold_days == 1 else pd.Series([True]*len(latest), index=latest.index)
        c3 = latest['close'] >= latest['ma20']
        c4 = latest['vol'] >= latest['vol_ma5'] * vol_ratio

        amt_mean = data.groupby('ts_code')['amount'].tail(20).mean()
        latest['amt_ok'] = latest['ts_code'].map(amt_mean >= amt_thr)

        res = latest[c1 & c2 & c3 & c4 & latest['amt_ok']].copy()
        if res.empty: return pd.DataFrame()

        res['vol_ratio'] = (res['vol'] / res['vol_ma5']).round(2)
        res = res.merge(df[['ts_code','pct_chg','price']], on='ts_code')

        out = res[['ts_code','name','price','vol_ratio','pct_chg','industry']]
        out.columns = ['代码','名称','现价','放量倍数','昨日涨幅%','行业']
        out = out.sort_values('放量倍数', ascending=False).reset_index(drop=True)
        return out

    except Exception as e:
        st.error(f"错误：{e}")
        return pd.DataFrame()

if st.button("开始选股", type="primary", use_container_width=True):
    st.caption(f"数据：2025-11-14 | 昨日：2025-11-13")
    with st.spinner("选股中..."):
        result = run_selection()
    st.success("完成！")
    if result.empty:
        st.warning("无符合股票")
    else:
        st.dataframe(
            result,
            use_container_width=True,
            column_config={
                "昨日涨幅%": st.column_config.NumberColumn(format="%.2f%%"),
                "现价": st.column_config.NumberColumn(format="%.2f"),
                "放量倍数": st.column_config.NumberColumn(format="%.2fx")
            },
            hide_index=True
        )
        st.balloons()
        st.caption(f"命中 {len(result)} 只 | {mode}")
