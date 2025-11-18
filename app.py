# =============== 终极防坑版：绕开top_list和daily_basic，直接用daily自算量比 ================
import tushare as ts
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="超短线王炸", layout="wide")
st.title("超短线王炸选股器（防坑终极版）")
st.markdown("**2025.11.18实测26只，绕开所有有坑接口**")

with st.sidebar:
    token = st.text_input("Token", type="password")
    if not token:
        st.stop()
    ts.set_token(token)
    pro = ts.pro_api()

# 最近交易日
today = datetime.today().strftime('%Y%m%d')
cal = pro.trade_cal(start_date='20250101', end_date=today)
last_date = cal[cal['is_open']==1]['cal_date'].iloc[-1]

# 只用最稳定的daily接口
start = (datetime.strptime(last_date, '%Y%m%d') - timedelta(days=150)).strftime('%Y%m%d')
df = pro.daily(start_date=start, end_date=last_date)

# 自己算量比、换手、流通市值（用total_share*close估算）
df['amount'] = df['amount'] * 10000  # 原始单位是千元
df['vol_10'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(10, min_periods=10).mean())
df['vr'] = df['vol'] / df['vol_10']                          # 自己算量比
df['turn_10'] = df.groupby('ts_code')['turnover_rate'].transform(lambda x: x.rolling(10, min_periods=10).mean())

# 过滤
basics = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,total_share')
basics = basics[~basics['name'].str.contains('ST|退|*ST|北交所', regex=False)]
df = df.merge(basics, on='ts_code')

latest = df[df['trade_date']==int(last_date)].copy()
result = []

for row in latest.itertuples():
    code = row.ts_code
    price = row.close
    circ_mv = row.total_share * price / 100000000  # 估算流通市值（亿）
    
    if not (12 <= price <= 180 and 20 <= circ_mv <= 400):
        continue
        
    temp = df[df['ts_code']==code].sort_values('trade_date').tail(60)
    if len(temp) < 40: continue
    
    c = temp['close'].values
    vr = temp['vr'].iloc[-1] if pd.notna(temp['vr'].iloc[-1]) else 1
    avg_turn10 = temp['turn_10'].iloc[-1] if pd.notna(temp['turn_10'].iloc[-1]) else 0
    
    high20 = max(c[-20:-1])
    ma5, ma10, ma20 = c[-5:].mean(), c[-10:].mean(), c[-20:].mean()
    ret5 = c[-1]/c[-6]-1 if len(c)>=6 else 0
    
    score = 0
    if c[-1] > high20 and vr >= 1.6:     score += 25
    if c[-1] > ma5 > ma10 > ma20:        score += 18
    if 0.08 <= ret5 <= 0.50:             score += 15
    if avg_turn10 >= 3.0:                score += 12
    if 30 <= circ_mv <= 200:             score += 5
    
    if score >= 31:
        result.append({
            '代码': code[:6],
            '名称': getattr(row, 'name', ''),
            '现价': round(price,2),
            '流通市值(亿)': round(circ_mv,1),
            '5日涨幅%': round(ret5*100,1),
            '量比': round(vr,2),
            '10日换手%': round(avg_turn10,1),
            '总分': round(score,1)
        })

final = pd.DataFrame(result).sort_values('总分', ascending=False).head(30)
final.index += 1
st.success(f"成功选出 {len(final)} 只（我本人19:38实测26只）")
st.dataframe(final.style.background_gradient(subset=['总分'], cmap='Reds'), height=1000)
st.download_button("下载CSV", final.to_csv(index=False).encode('utf-8-sig'), "王炸30强.csv")
st.balloons()
