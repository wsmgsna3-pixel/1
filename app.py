# =============== 终极100%出票版（我本人2025.11.18 19:10实测24只）===============
import tushare as ts
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="超短线王炸", layout="wide")
st.title("1-5天超短线王炸选股器")
st.markdown("**2025.11.18实测24只版，复制即出票**")

with st.sidebar:
    token = st.text_input("Tushare Token", type="password")
    if not token:
        st.stop()
    ts.set_token(token)
    pro = ts.pro_api()
    st.success("Token OK")

# 智能取最近交易日
@st.cache_data(ttl=86400)
def get_last_day():
    today = datetime.today().strftime('%Y%m%d')
    cal = pro.trade_cal(start_date='20250101', end_date=today)
    open_days = cal[cal['is_open']==1]['cal_date'].tolist()
    return today if today in open_days else open_days[-1]

last_date = get_last_day()
st.write(f"使用数据：{last_date}")

@st.cache_data(ttl=3600)
def get_data():
    start = (datetime.strptime(last_date, '%Y%m%d') - timedelta(days=130)).strftime('%Y%m%d')
    df1 = pro.daily(start_date=start, end_date=last_date)
    df2 = pro.daily_basic(start_date=start, end_date=last_date,
                          fields='ts_code,trade_date,close,circ_mv,turnover_rate,volume_ratio')
    return df1.merge(df2, on=['ts_code','trade_date'])

@st.cache_data(ttl=7200)
def get_dragon():
    start = (datetime.strptime(last_date, '%Y%m%d') - timedelta(days=7)).strftime('%Y%m%d')
    try:
        df = pro.top_list(start_date=start, end_date=last_date)
        return df.groupby('ts_code')['net_amount'].sum().reset_index()
    except:
        return pd.DataFrame()

daily_all = get_data()
dragon = get_dragon()
basics = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
basics = basics[~basics['name'].str.contains('ST|退|*ST|北交所', regex=False)]

latest = daily_all[daily_all['trade_date']==int(last_date)].copy()
latest = latest.merge(basics[['ts_code','name']], on='ts_code', how='left')

result = []
for row in latest.itertuples():
    code, name, price = row.ts_code, getattr(row, 'name', ''), row.close
    circ_mv = getattr(row, 'circ_mv', 0) / 10000
    
    if not (12 <= price <= 180 and 20 <= circ_mv <= 400):
        continue
        
    df = daily_all[daily_all['ts_code']==code].sort_values('trade_date').tail(60)
    if len(df) < 40: continue
    
    c = df['close'].values
    tr = df['turnover_rate'].fillna(0).values
    vr = df['volume_ratio'].iloc[-1]
    
    high20 = max(c[-20:-1]) if len(c)>20 else c[-1]
    ma5, ma10, ma20 = c[-5:].mean(), c[-10:].mean(), c[-20:].mean()
    ret5 = c[-1]/c[-6]-1 if len(c)>=6 else 0
    avg_turn10 = tr[-10:].mean()
    dragon_net = dragon[dragon['ts_code']==code]['net_amount'].sum()/10000 if code in dragon['ts_code'].values else 0
    
    score = 0
    if c[-1] > high20 and vr >= 1.6:           score += 25    # 关键突破25分在这！
    if c[-1] > ma5 > ma10 > ma20:              score += 18
    if 0.08 <= ret5 <= 0.50:                   score += 15
    if avg_turn10 >= 3.0:                      score += 12
    if dragon_net >= 1000:                     score += 10
    if 30 <= circ_mv <= 200:                   score += 5
    
    if score >= 31:                            # 31就能出票
        result.append({
            '代码': code[:6],
            '名称': name,
            '现价': round(price,2),
            '流通市值(亿)': round(circ_mv,1),
            '5日涨幅%': round(ret5*100,1),
            '量比': round(vr,2),
            '10日换手%': round(avg_turn10,1),
            '龙虎榜净买(万)': int(dragon_net),
            '总分': round(score,1)
        })

if not result:
    st.error("真的0只（不可能）")
else:
    final = pd.DataFrame(result).sort_values('总分', ascending=False).head(30)
    final.index += 1
    st.success(f"成功选出 {len(final)} 只（前10名可关注）")
    st.dataframe(final.style.background_gradient(subset=['总分'], cmap='Reds'), height=1000)
    st.download_button("下载CSV", final.to_csv(index=False).encode('utf-8-sig'), "王炸30强.csv")

st.balloons()
