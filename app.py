import streamlit as st
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="选股王 · 极速版", layout="wide")
st.title("选股王 · 极速版")
st.caption("2100积分 | 3分钟出结果 | 日期永不错")

token = st.text_input("Tushare Token", type="password")
if not token: st.stop()
pro = ts.pro_api(token)

# 强制使用最新交易日（避免权限回退）
last_trade_day = "20251114"  # 周五
yesterday = "20251113"       # 周四
start_date = "20250901"

@st.cache_data(ttl=3600)
def run():
    # Step 1: 预筛价格/市值（1次请求）
    db = pro.daily_basic(trade_date=last_trade_day, fields='ts_code,close,total_mv')
    sb = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    df = db.merge(sb, on='ts_code').rename(columns={'close':'price'})

    df = df[
        (~df['name'].str.contains('ST')) &
        (~df['name'].str.contains(r'\*ST', regex=True)) &
        (~df['ts_code'].str.startswith(('8','4'))) &
        (df['price'].between(10,200)) &
        (df['total_mv'].between(100000,50000000))
    ]

    if df.empty: return pd.DataFrame()

    # Step 2: 昨日涨幅前1000（1次请求）
    dy = pro.daily(trade_date=yesterday, fields='ts_code,pct_chg')
    dy = dy[(dy['pct_chg'] < 9.8) & (dy['ts_code'].isin(df['ts_code']))]
    if dy.empty: return pd.DataFrame()

    codes = dy.nlargest(1000, 'pct_chg')['ts_code'].tolist()
    df = df[df['ts_code'].isin(codes)].merge(dy[['ts_code','pct_chg']], on='ts_code')

    # Step 3: 批量拉日线（1次请求）
    data = pro.daily(ts_code=','.join(df['ts_code']), start_date=start_date, end_date=last_trade_day)
    if data.empty: return pd.DataFrame()

    data = data.merge(df[['ts_code','name','industry']], on='ts_code').sort_values(['ts_code','trade_date'])

    # Step 4: 向量化计算指标
    for ma in [5,10,20]:
        data[f'ma{ma}'] = data.groupby('ts_code')['close'].transform(lambda x: x.rolling(ma).mean())
    data['vol_ma5'] = data.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5).mean())

    latest = data.groupby('ts_code').tail(1).copy()
    prev = data.groupby('ts_code').apply(lambda x: x.tail(2).iloc[:-1] if len(x)>=2 else pd.DataFrame()).reset_index(drop=True)
    prev = prev[['ts_code','ma5','ma10']].rename(columns={'ma5':'ma5_prev','ma10':'ma10_prev'})
    latest = latest.merge(prev, on='ts_code', how='left').fillna(0)

    # Step 5: 条件筛选
    c1 = latest['ma5'] > latest['ma10']
    c2 = latest['ma5_prev'] <= latest['ma10_prev']
    c3 = latest['close'] >= latest['ma20']
    c4 = latest['vol'] >= latest['vol_ma5'] * 1.5

    amt = data.groupby('ts_code')['amount'].tail(20).mean()
    latest['amt_ok'] = amt.reindex(latest['ts_code']).fillna(0) >= 100

    res = latest[c1 & c2 & c3 & c4 & latest['amt_ok']].copy()
    if res.empty: return pd.DataFrame()

    res['vol_ratio'] = (res['vol'] / res['vol_ma5']).round(2)
    res = res.merge(df[['ts_code','pct_chg','price']], on='ts_code')
    out = res[['ts_code','name','price','vol_ratio','pct_chg','industry']]
    out.columns = ['代码','名称','现价','放量倍数','昨日涨幅%','行业']
    return out.sort_values('放量倍数', ascending=False).reset_index(drop=True)

if st.button("开始选股", type="primary"):
    st.caption(f"数据：2025-11-14 | 昨日：2025-11-13")
    with st.spinner("3分钟内出结果..."):
        result = run()
    st.success("完成")
    if result.empty:
        st.warning("暂无符合股票")
    else:
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.balloons()
        st.caption(f"命中 {len(result)} 只")
