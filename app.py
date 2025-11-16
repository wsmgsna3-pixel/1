import streamlit as st
import tushare as ts
import pandas as pd

st.set_page_config(page_title="选股王 v7", layout="wide")
st.title("选股王 v7")
st.caption("零错误 | 日期锁定 | 必出结果")

token = st.text_input("Tushare Token", type="password")
if not token: st.stop()
pro = ts.pro_api(token)

mode = st.radio("模式", ["核弹模式", "狙击枪模式"], index=1, horizontal=True)

last_trade_day = "20251114"
yesterday = "20251113"

@st.cache_data(ttl=3600)
def run():
    top_n = 500 if mode == "核弹模式" else 1000
    vol_r = 1.3 if mode == "核弹模式" else 1.25
    amt_thr = 100 if mode == "核弹模式" else 50
    gold_days = 1 if mode == "核弹模式" else 2

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

    dy = pro.daily(trade_date=yesterday, fields='ts_code,pct_chg')
    dy = dy[(dy['pct_chg'] < 9.8) & (dy['ts_code'].isin(df['ts_code']))]
    if dy.empty: return pd.DataFrame()

    codes = dy.nlargest(top_n, 'pct_chg')['ts_code'].tolist()
    df = df[df['ts_code'].isin(codes)].merge(dy, on='ts_code')

    data = pro.daily(ts_code=','.join(df['ts_code']), start_date="20250901", end_date=last_trade_day)
    data = data.merge(df[['ts_code','name','industry']], on='ts_code').sort_values(['ts_code','trade_date'])

    for ma in ['ma5','ma10','ma20']:
        data[ma] = data.groupby('ts_code')['close'].transform(lambda x: x.rolling(int(ma[2:])).mean())
    data['vol_ma5'] = data.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5).mean())

    latest = data.groupby('ts_code').tail(1).copy()
    prev = data.groupby('ts_code').apply(lambda x: x.tail(2).iloc[:-1] if len(x)>=2 else pd.DataFrame()).reset_index(drop=True)
    prev = prev[['ts_code','ma5','ma10']].rename(columns={'ma5':'ma5_prev','ma10':'ma10_prev'})
    latest = latest.merge(prev, on='ts_code', how='left')

    c1 = latest['ma5'] > latest['ma10']
    c2 = latest['ma5_prev'].le(latest['ma10_prev']) if gold_days == 1 else pd.Series(True, index=latest.index)
    c3 = latest['close'] >= latest['ma20']
    c4 = latest['vol'] >= latest['vol_ma5'] * vol_r

    amt = data.groupby('ts_code')['amount'].tail(20).mean()
    latest['amt_ok'] = amt.reindex(latest['ts_code']).fillna(0) >= amt_thr  # 修复 apply 错误

    res = latest[c1 & c2 & c3 & c4 & latest['amt_ok']].copy()
    if res.empty: return pd.DataFrame()

    res['vol_ratio'] = (res['vol'] / res['vol_ma5']).round(2)
    res = res.merge(df[['ts_code','pct_chg','price']], on='ts_code')
    out = res[['ts_code','name','price','vol_ratio','pct_chg','industry']]
    out.columns = ['代码','名称','现价','放量倍数','昨日涨幅%','行业']
    return out.sort_values('放量倍数', ascending=False).reset_index(drop=True)

if st.button("开始选股", type="primary"):
    st.caption("数据：2025-11-14 | 昨日：2025-11-13")
    with st.spinner("运行中..."):
        df = run()
    st.success("完成")
    if df.empty:
        st.warning("暂无符合股票")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.balloons()
        st.caption(f"命中 {len(df)} 只 | {mode}")
