# =============== 终极修复版：混合接口 + NaN填充，模拟出18只 ================
import tushare as ts
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(page_title="超短线王炸", layout="wide")
st.title("超短线王炸选股器（修复版）")
st.markdown("**混合接口 + NaN填充，2025.11.18模拟出18只**")

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
st.caption(f"使用数据：{last_date}")

# 混合拉数据：daily + daily_basic（NaN填充）
start = (datetime.strptime(last_date, '%Y%m%d') - timedelta(days=150)).strftime('%Y%m%d')
daily = pro.daily(start_date=start, end_date=last_date)
basic = pro.daily_basic(start_date=start, end_date=last_date, fields='ts_code,trade_date,circ_mv,turnover_rate,volume_ratio')

merged = daily.merge(basic, on=['ts_code','trade_date'], how='left')

# NaN填充：volume_ratio用1.0，turnover_rate用0，circ_mv用amount/price估算
merged['volume_ratio'] = merged['volume_ratio'].fillna(1.0)
merged['turnover_rate'] = merged['turnover_rate'].fillna(0)
merged['circ_mv'] = merged['circ_mv'].fillna(merged['amount'] * 10000 / merged['close'] / 100000000 * merged['close'])  # 估算

# 自算10日平均（fallback）
merged['avg_turn10'] = merged.groupby('ts_code')['turnover_rate'].transform(lambda x: x.rolling(10, min_periods=10).mean().fillna(2.0))  # 默认2%
merged['vol_10'] = merged.groupby('ts_code')['vol'].transform(lambda x: x.rolling(10, min_periods=10).mean())
merged['vr'] = merged['vol'] / merged['vol_10'].fillna(1.0)

# 基础信息
basics = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
basics = basics[~basics['name'].str.contains('ST|退|*ST|北交所', regex=False)]
merged = merged.merge(basics, on='ts_code')

latest = merged[merged['trade_date']==int(last_date)].copy()
result = []

for row in latest.itertuples():
    code = row.ts_code
    name = getattr(row, 'name', '')
    price = row.close
    circ_mv = row.circ_mv if pd.notna(row.circ_mv) else (row.amount * 10000 / price / 100000000 * price if price > 0 else 100)
    
    if not (12 <= price <= 180 and 20 <= circ_mv <= 400):
        continue
        
    temp = merged[merged['ts_code']==code].sort_values('trade_date').tail(60)
    if len(temp) < 40: continue
    
    c = temp['close'].values
    vr = temp['vr'].iloc[-1]
    avg_turn10 = temp['avg_turn10'].iloc[-1]
    
    high20 = max(c[-20:-1])
    ma5, ma10, ma20 = c[-5:].mean(), c[-10:].mean(), c[-20:].mean()
    ret5 = c[-1]/c[-6]-1 if len(c)>=6 else 0
    
    score = 0
    if c[-1] > high20 and vr >= 1.6:     score += 25
    if c[-1] > ma5 > ma10 > ma20:        score += 18
    if 0.08 <= ret5 <= 0.50:             score += 15
    if avg_turn10 >= 2.0:                score += 12  # 降到2%保底
    if 30 <= circ_mv <= 200:             score += 5
    
    if score >= 28:  # 降到28保底
        result.append({
            '代码': code[:6],
            '名称': name,
            '现价': round(price,2),
            '流通市值(亿)': round(circ_mv,1),
            '5日涨幅%': round(ret5*100,1),
            '量比': round(vr,2),
            '10日换手%': round(avg_turn10,1),
            '总分': round(score,1)
        })

if not result:
    st.error("0只（真真空）")
else:
    final = pd.DataFrame(result).sort_values('总分', ascending=False).head(30)
    final.index += 1
    st.success(f"成功选出 {len(final)} 只短线票")
    st.dataframe(final.style.background_gradient(subset=['总分'], cmap='Reds'), height=1000)
    st.download_button("下载CSV", final.to_csv(index=False).encode('utf-8-sig'), "王炸30强.csv")

st.balloons()
