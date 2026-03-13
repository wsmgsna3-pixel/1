# -*- coding: utf-8 -*-

# 智选股 V1.0 - 鱼身策略选股系统

# 收盘后运行，次日高开+冲高1.5%触发买入，止损5%

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import os
import pickle

warnings.filterwarnings(“ignore”)

# —————————

# 全局变量

# —————————

pro = None
CHECKPOINT_FILE = “bt_checkpoint.csv”

# —————————

# 页面设置

# —————————

st.set_page_config(page_title=“智选股 V1.0”, layout=“wide”)
st.title(“智选股 V1.0 - 鱼身策略”)
st.caption(“收盘后运行，次日高开+冲高1.5%触发买入，止损5%”)

# —————————

# 基础API

# —————————

@st.cache_data(ttl=3600*12)
def safe_api(func_name, **kwargs):
global pro
if pro is None:
return pd.DataFrame()
func = getattr(pro, func_name)
for _ in range(3):
try:
df = func(**kwargs)
if df is not None and not df.empty:
return df
time.sleep(0.5)
except:
time.sleep(1)
return pd.DataFrame()

@st.cache_data(ttl=3600*24*7)
def load_stock_basic():
df = safe_api(‘stock_basic’, list_status=‘L’, fields=‘ts_code,name,list_date’)
if df.empty:
return pd.DataFrame()
df = df[~df[‘name’].str.contains(‘ST|退’, na=False)]
df = df[~df[‘ts_code’].str.startswith(‘43’)]
df = df[~df[‘ts_code’].str.startswith(‘83’)]
df = df[~df[‘ts_code’].str.startswith(‘87’)]
df = df[~df[‘ts_code’].str.startswith(‘92’)]
return df

@st.cache_data(ttl=3600*24*7)
def load_industry_map():
global pro
if pro is None:
return {}
try:
sw = pro.index_classify(level=‘L1’, src=‘SW2021’)
if sw.empty:
return {}
result = {}
for code in sw[‘index_code’].tolist():
members = pro.index_member(index_code=code, is_new=‘Y’)
if not members.empty:
for c in members[‘con_code’]:
result[c] = code
time.sleep(0.05)
return result
except:
return {}

# —————————

# 历史数据获取

# —————————

@st.cache_data(ttl=3600*12)
def get_stock_history(ts_code, end_date, lookback=90):
start = (datetime.strptime(end_date, “%Y%m%d”) - timedelta(days=lookback*2)).strftime(”%Y%m%d”)
daily = safe_api(‘daily’, ts_code=ts_code, start_date=start, end_date=end_date)
if daily is None or daily.empty or len(daily) < 30:
return pd.DataFrame()
adj = safe_api(‘adj_factor’, ts_code=ts_code, start_date=start, end_date=end_date)
daily = daily.sort_values(‘trade_date’).reset_index(drop=True)
if adj.empty:
return daily
daily = daily.merge(adj[[‘trade_date’, ‘adj_factor’]], on=‘trade_date’, how=‘left’)
daily[‘adj_factor’] = daily[‘adj_factor’].fillna(method=‘ffill’).fillna(1.0)
latest_adj = daily[‘adj_factor’].iloc[-1]
for col in [‘open’, ‘high’, ‘low’, ‘close’]:
daily[col] = daily[col] * daily[‘adj_factor’] / latest_adj
return daily

# —————————

# 技术指标计算

# —————————

def calc_indicators(hist, trade_date, winner_rate_dict):
if hist.empty or len(hist) < 30:
return None
close = hist[‘close’]
today = hist[hist[‘trade_date’] == trade_date]
if today.empty:
return None
idx = today.index[0]
if idx < 19:
return None

```
ma20 = close.iloc[idx-19:idx+1].mean()
ma20_prev = close.iloc[idx-20:idx].mean() if idx >= 20 else ma20
ma60 = close.iloc[max(0,idx-59):idx+1].mean()

ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
dif = ema12 - ema26
dea = dif.ewm(span=9, adjust=False).mean()
macd_bar = ((dif - dea) * 2).iloc[idx]

delta = close.diff()
gain = delta.where(delta > 0, 0).ewm(alpha=1/12, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/12, adjust=False).mean()
rsi = (100 - 100 / (1 + gain / (loss + 1e-9))).iloc[idx]

row = hist.iloc[idx]
c = float(row['close'])
h = float(row['high'])
l = float(row['low'])
upper_shadow = (h - c) / c * 100
body_range = h - l
body_pos = (c - l) / body_range if body_range > 0 else 0

pct_1d = float(row['pct_chg']) if 'pct_chg' in row else 0
pct_5d = (c / float(hist.iloc[idx-5]['close']) - 1) * 100 if idx >= 5 else 0
pct_20d = (c / float(hist.iloc[idx-20]['close']) - 1) * 100 if idx >= 20 else 0

low_60 = hist['low'].iloc[max(0,idx-59):idx+1].min()
from_bottom = (c - low_60) / low_60 * 100 if low_60 > 0 else 0

vol_5 = hist['vol'].iloc[max(0,idx-4):idx+1].mean()
vol_20 = hist['vol'].iloc[max(0,idx-19):idx+1].mean()
vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1

consec_limit = False
if idx >= 1 and 'pct_chg' in hist.columns:
    prev_pct = float(hist.iloc[idx-1]['pct_chg'])
    if pct_1d >= 9.5 and prev_pct >= 9.5:
        consec_limit = True

ts_code = hist['ts_code'].iloc[0] if 'ts_code' in hist.columns else ''
winner_rate = winner_rate_dict.get(ts_code, 60)

return {
    'close': c, 'high': h, 'low': l,
    'ma20': ma20, 'ma60': ma60, 'ma20_prev': ma20_prev,
    'macd_bar': macd_bar, 'rsi': rsi,
    'upper_shadow': upper_shadow, 'body_pos': body_pos,
    'pct_1d': pct_1d, 'pct_5d': pct_5d, 'pct_20d': pct_20d,
    'from_bottom': from_bottom,
    'vol_ratio': vol_ratio,
    'consec_limit': consec_limit,
    'winner_rate': winner_rate,
}
```

# —————————

# 六维评分

# —————————

def calc_score(ind, sector_score, market_strong):
detail = {}

```
tech = 0
if ind['close'] > ind['ma20'] and ind['ma20'] > ind['ma20_prev']:
    tech += 10
if ind['close'] > ind['ma60']:
    tech += 5
if ind['macd_bar'] > 0:
    tech += 5
rsi = ind['rsi']
if 55 <= rsi <= 70:
    tech += 5
elif 70 < rsi <= 80:
    tech += 3
else:
    tech += 1
detail['tech'] = min(tech, 25)

timing = 0
dev = (ind['close'] - ind['ma20']) / ind['ma20'] * 100
if dev <= 3:
    timing += 10
elif dev <= 7:
    timing += 7
elif dev <= 12:
    timing += 4
p5 = ind['pct_5d']
if 3 <= p5 <= 10:
    timing += 10
elif p5 < 3:
    timing += 6
elif p5 <= 15:
    timing += 4
detail['timing'] = min(timing, 20)

vr = ind['vol_ratio']
if 1.2 <= vr <= 2.5:
    vol = 15
elif vr > 3.5:
    vol = 0
elif 1.0 <= vr < 1.2 or 2.5 < vr <= 3.5:
    vol = 8
else:
    vol = 3
detail['vol'] = min(vol, 15)

fb = ind['from_bottom']
if fb <= 30:
    fish = 15
elif fb <= 60:
    fish = 10
elif fb <= 100:
    fish = 4
else:
    fish = 0
detail['fish'] = fish

detail['sector'] = min(sector_score, 15)
detail['market'] = 10 if market_strong else 3

total = sum(detail.values())
return total, detail
```

# —————————

# 风险标签

# —————————

def risk_tag(ind):
if ind[‘from_bottom’] > 100 or ind[‘consec_limit’]:
return “高风险”
if ind[‘from_bottom’] > 60 or ind[‘pct_5d’] > 15 or ind[‘rsi’] > 80:
return “谨慎”
return “安全”

# —————————

# 大盘状态

# —————————

@st.cache_data(ttl=3600*12)
def get_market_strong(trade_date):
start = (datetime.strptime(trade_date, “%Y%m%d”) - timedelta(days=60)).strftime(”%Y%m%d”)
df = safe_api(‘index_daily’, ts_code=‘000300.SH’, start_date=start, end_date=trade_date)
if df.empty or len(df) < 20:
return False
df = df.sort_values(‘trade_date’)
return float(df.iloc[-1][‘close’]) > df[‘close’].tail(20).mean()

# —————————

# 板块热度

# —————————

@st.cache_data(ttl=3600*12)
def get_sector_scores(trade_date):
global pro
start = (datetime.strptime(trade_date, “%Y%m%d”) - timedelta(days=20)).strftime(”%Y%m%d”)
hs300 = safe_api(‘index_daily’, ts_code=‘000300.SH’, start_date=start, end_date=trade_date)
if hs300.empty or len(hs300) < 5:
mkt_ret = 0
else:
hs300 = hs300.sort_values(‘trade_date’)
mkt_ret = (float(hs300.iloc[-1][‘close’]) / float(hs300.iloc[-5][‘close’]) - 1) * 100
scores = {}
try:
sw = pro.index_classify(level=‘L1’, src=‘SW2021’)
for code in sw[‘index_code’].tolist():
idf = safe_api(‘sw_daily’, index_code=code, start_date=start, end_date=trade_date)
if idf.empty or len(idf) < 5:
continue
idf = idf.sort_values(‘trade_date’)
ret = (float(idf.iloc[-1][‘close’]) / float(idf.iloc[-5][‘close’]) - 1) * 100
excess = ret - mkt_ret
if excess >= 3:
scores[code] = 15
elif excess >= 1:
scores[code] = 10
elif excess >= 0:
scores[code] = 5
else:
scores[code] = 0
except:
pass
return scores

# —————————

# 核心选股（单日）

# —————————

def run_screen(trade_date, top_n, min_price, min_mv, max_mv, for_backtest=False):
global pro
daily_all = safe_api(‘daily’, trade_date=trade_date)
if daily_all.empty:
return pd.DataFrame()

```
daily_basic = safe_api('daily_basic', trade_date=trade_date, fields='ts_code,circ_mv,pe,pb')
chip_df = safe_api('cyq_perf', trade_date=trade_date)
winner_rate_dict = {}
if not chip_df.empty and 'winner_rate' in chip_df.columns:
    winner_rate_dict = dict(zip(chip_df['ts_code'], chip_df['winner_rate']))

basics = load_stock_basic()
industry_map = load_industry_map()
market_strong = get_market_strong(trade_date)
sector_scores = get_sector_scores(trade_date)

df = daily_all.copy()
if not daily_basic.empty:
    df = df.merge(daily_basic, on='ts_code', how='left')
else:
    df['circ_mv'] = 0

df = df.merge(basics[['ts_code', 'name']], on='ts_code', how='inner')
df['circ_mv_b'] = df['circ_mv'] / 10000
df = df[df['close'] >= min_price]
df = df[df['circ_mv_b'] >= min_mv]
df = df[df['circ_mv_b'] <= max_mv]
df['winner_rate'] = df['ts_code'].map(winner_rate_dict).fillna(60)
df = df[(df['winner_rate'] >= 50) & (df['winner_rate'] <= 85)]
df = df.sort_values('pct_chg', ascending=False).head(200)

records = []
for row in df.itertuples():
    hist = get_stock_history(row.ts_code, trade_date, lookback=90)
    if hist.empty:
        continue
    ind = calc_indicators(hist, trade_date, winner_rate_dict)
    if ind is None:
        continue
    if ind['close'] <= ind['ma20']:
        continue
    if ind['ma20'] <= ind['ma20_prev']:
        continue
    if ind['upper_shadow'] > 5:
        continue
    if ind['body_pos'] < 0.6:
        continue
    if ind['consec_limit']:
        continue
    if ind['pct_20d'] > 30:
        continue
    if ind['pct_5d'] > 20:
        continue

    ind_code = industry_map.get(row.ts_code, '')
    s_score = sector_scores.get(ind_code, 5)
    score, detail = calc_score(ind, s_score, market_strong)
    tag = risk_tag(ind)

    rec = {
        'ts_code': row.ts_code,
        'name': row.name,
        'close': ind['close'],
        'pct_1d': ind['pct_1d'],
        'pct_5d': ind['pct_5d'],
        'pct_20d': ind['pct_20d'],
        'from_bottom': ind['from_bottom'],
        'rsi': round(ind['rsi'], 1),
        'winner_rate': ind['winner_rate'],
        'vol_ratio': round(ind['vol_ratio'], 2),
        'score': score,
        'tag': tag,
        'market': "强势" if market_strong else "弱势",
        'tech': detail['tech'],
        'timing': detail['timing'],
        'vol_score': detail['vol'],
        'fish': detail['fish'],
        'sector': detail['sector'],
        'mkt_score': detail['market'],
        'buy_low': round(ind['close'], 2),
        'buy_high': round(ind['close'] * 1.02, 2),
        'stop_loss': round(ind['close'] * 0.95, 2),
        'target': round(ind['close'] * 1.08, 2),
    }

    if for_backtest:
        d0_close = ind['close']
        d0 = datetime.strptime(trade_date, "%Y%m%d")
        start_f = (d0 + timedelta(days=1)).strftime("%Y%m%d")
        end_f = (d0 + timedelta(days=20)).strftime("%Y%m%d")
        fut = safe_api('daily', ts_code=row.ts_code, start_date=start_f, end_date=end_f)
        rec['R_D1'] = np.nan
        rec['R_D3'] = np.nan
        rec['R_D5'] = np.nan
        rec['triggered'] = False
        if not fut.empty:
            fut = fut.sort_values('trade_date').reset_index(drop=True)
            if len(fut) >= 1:
                next_open = float(fut.iloc[0]['open'])
                next_high = float(fut.iloc[0]['high'])
                if next_open > d0_close:
                    trigger = next_open * 1.015
                    if next_high >= trigger:
                        rec['triggered'] = True
                        for n, key in [(1,'R_D1'),(3,'R_D3'),(5,'R_D5')]:
                            if len(fut) >= n:
                                sell = float(fut.iloc[n-1]['close'])
                                rec[key] = round((sell - trigger) / trigger * 100, 2)

    records.append(rec)

if not records:
    return pd.DataFrame()

result = pd.DataFrame(records)
result = result.sort_values('score', ascending=False).head(top_n).reset_index(drop=True)
result.insert(0, 'rank', range(1, len(result)+1))
return result
```

# —————————

# 侧边栏参数

# —————————

with st.sidebar:
st.header(“参数设置”)
st.subheader(“Tushare Token”)
token = st.text_input(“Token”, type=“password”)
st.subheader(“股票池过滤”)
min_price = st.number_input(“最低股价(元)”, value=10.0, min_value=1.0, step=1.0)
min_mv = st.number_input(“最小流通市值(亿)”, value=50.0, min_value=10.0, step=10.0)
max_mv = st.number_input(“最大流通市值(亿)”, value=1000.0, min_value=100.0, step=100.0)
st.subheader(“实盘选股”)
top_n = st.slider(“输出Top N候选股”, 3, 10, 5)
st.subheader(“回测设置”)
bt_end = st.date_input(“回测截止日期”, value=datetime.now().date())
bt_days = st.number_input(“回测天数”, value=30, min_value=5, max_value=90, step=5)
bt_top_n = st.number_input(“每日推荐数”, value=4, min_value=1, max_value=10)
resume = st.checkbox(“开启断点续传”, value=True)
if st.button(“清除缓存”):
if os.path.exists(CHECKPOINT_FILE):
os.remove(CHECKPOINT_FILE)
st.success(“缓存已清除”)

# —————————

# Token初始化

# —————————

TS_TOKEN = st.text_input(“Tushare Token”, type=“password”)
if not TS_TOKEN:
st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# —————————

# 主界面

# —————————

tab1, tab2 = st.tabs([“实盘选股”, “历史回测”])

with tab1:
st.subheader(“实盘选股 - 今日候选”)
st.caption(“收盘后运行，第二天9:25判断高开，9:30后冲高1.5%触发买入”)
screen_date = st.date_input(“选股日期”, value=datetime.now().date())
if st.button(“开始选股”):
date_str = screen_date.strftime(”%Y%m%d”)
basics = load_stock_basic()
if basics.empty:
st.error(“无法获取股票列表，请检查Token权限”)
else:
with st.spinner(“正在分析全市场数据，请稍候…”):
result = run_screen(date_str, top_n, min_price, min_mv, max_mv, for_backtest=False)
market_state = get_market_strong(date_str)
if result.empty:
st.warning(“今日未找到符合条件的股票，可适当放宽参数”)
else:
ms = “强势（沪深300站上MA20）” if market_state else “弱势（沪深300跌破MA20）”
st.success(f”筛选完成，共推荐 {len(result)} 只候选股”)
st.info(f”当前大盘状态：{ms}”)
for *, row in result.iterrows():
with st.expander(f”No.{row[‘rank’]}  {row[‘name’]}（{row[‘ts_code’]}）  {row[‘tag’]}  评分:{round(row[‘score’],0)}”):
c1,c2,c3,c4 = st.columns(4)
c1.metric(“今日涨幅”, f”{row[‘pct_1d’]:+.2f}%”)
c2.metric(“5日涨幅”, f”{row[‘pct_5d’]:+.2f}%”)
c3.metric(“20日涨幅”, f”{row[‘pct_20d’]:+.2f}%”)
c4.metric(“本轮涨幅”, f”{row[‘from_bottom’]:+.1f}%”)
c5,c6,c7,c8 = st.columns(4)
c5.metric(“RSI”, row[‘rsi’])
c6.metric(“筹码获利比”, f”{row[‘winner_rate’]:.1f}%”)
c7.metric(“量比(5/20日)”, f”{row[‘vol_ratio’]:.2f}x”)
c8.metric(“大盘环境”, row[‘market’])
cc1,cc2,cc3,cc4 = st.columns(4)
cc1.metric(“建议买入区间”, f”{row[‘buy_low’]}~{row[‘buy_high’]}”)
cc2.metric(“止损价(-5%)”, row[‘stop_loss’])
cc3.metric(“目标价(+8%)”, row[‘target’])
cc4.metric(“风险收益比”, “1:1.6”)
dims  = [‘tech’,‘timing’,‘vol_score’,‘fish’,‘sector’,‘mkt_score’]
labels = [‘技术面/25’,‘买入时机/20’,‘量能/15’,‘鱼身/15’,‘板块/15’,‘大盘/10’]
maxes  = [25,20,15,15,15,10]
dcols  = st.columns(6)
for i,(d,lb,mx) in enumerate(zip(dims,labels,maxes)):
dcols[i].metric(lb, f”{row[d]}/{mx}”)
show_df = result[[‘rank’,‘name’,‘ts_code’,‘close’,‘pct_1d’,‘pct_5d’,‘pct_20d’,
‘from_bottom’,‘rsi’,‘winner_rate’,‘score’,‘tag’,
‘buy_low’,‘stop_loss’,‘target’]].copy()
show_df.columns = [‘排名’,‘名称’,‘代码’,‘现价’,‘今日%’,‘5日%’,‘20日%’,
‘本轮%’,‘RSI’,‘筹码%’,‘评分’,‘风险’,‘买入’,‘止损’,‘目标’]
st.dataframe(show_df, use_container_width=True)
csv = show_df.to_csv(index=False).encode(‘utf-8-sig’)
st.download_button(“导出CSV”, csv, f”result*{date_str}.csv”, “text/csv”)
st.caption(“本工具仅供学习研究，不构成投资建议。股市有风险，投资需谨慎。”)

with tab2:
st.subheader(“历史回测 - 策略验证”)
st.caption(“模拟：次日高开+盘中冲高1.5%触发买入，持有N天后收盘价卖出，止损5%”)
if st.button(“启动回测”):
end_str = bt_end.strftime(”%Y%m%d”)
cal = safe_api(‘trade_cal’,
start_date=(bt_end - timedelta(days=int(bt_days)*3)).strftime(”%Y%m%d”),
end_date=end_str)
if cal.empty:
st.error(“无法获取交易日历，请检查Token”)
st.stop()
dates = cal[cal[‘is_open’]==1].sort_values(‘cal_date’)[‘cal_date’].tail(int(bt_days)).tolist()
processed = set()
results = []
if resume and os.path.exists(CHECKPOINT_FILE):
try:
ex = pd.read_csv(CHECKPOINT_FILE)
processed = set(ex[‘trade_date’].astype(str).unique())
results.append(ex)
st.success(f”读取断点存档，已跳过 {len(processed)} 个交易日”)
except:
pass
dates_to_run = [d for d in dates if d not in processed]
if not dates_to_run:
st.success(“所有日期已计算完毕！”)
else:
bar = st.progress(0, text=“回测中…”)
err_ph = st.empty()
for i, date in enumerate(dates_to_run):
try:
res = run_screen(date, int(bt_top_n), min_price, min_mv, max_mv, for_backtest=True)
if not res.empty:
res[‘trade_date’] = date
first = not os.path.exists(CHECKPOINT_FILE)
res.to_csv(CHECKPOINT_FILE, mode=‘a’, index=False, header=first, encoding=‘utf-8-sig’)
results.append(res)
except Exception as e:
err_ph.warning(f”{date} 处理异常: {e}”)
bar.progress((i+1)/len(dates_to_run), text=f”回测中: {date} ({i+1}/{len(dates_to_run)})”)
bar.empty()
if results:
final = pd.concat(results).reset_index(drop=True)
final = final.sort_values([‘trade_date’,‘rank’], ascending=[False,True])
st.header(“回测统计报告”)
col1, col2 = st.columns(2)
col1.metric(“总选股记录”, f”{len(final)} 条”)
col1.metric(“涉及交易日”, f”{final[‘trade_date’].nunique()} 天”)
triggered = final[final[‘triggered’]==True] if ‘triggered’ in final.columns else pd.DataFrame()
if not triggered.empty:
col2.metric(“触发买入比例”, f”{len(triggered)/len(final)*100:.1f}%”)
cols3 = st.columns(3)
for i, n in enumerate([1,3,5]):
key = f’R_D{n}’
if key in triggered.columns:
valid = triggered.dropna(subset=[key])
if not valid.empty:
avg = valid[key].mean()
win = (valid[key] > 0).mean() * 100
loss = (valid[key] < -5).mean() * 100
cols3[i].metric(f”D+{n} 均收益/胜率”,
f”{avg:.2f}% / {win:.1f}%”,
delta=f”亏损>5%: {loss:.1f}%”)
st.subheader(“按排名分层分析（触发买入）”)
for n in [1,3,5]:
key = f’R_D{n}’
if key not in triggered.columns:
continue
valid = triggered.dropna(subset=[key])
if valid.empty:
continue
grp = valid.groupby(‘rank’)[key].agg(
avg=‘mean’,
win_rate=lambda x: (x > 0).mean() * 100,
n=‘count’
).round(2)
grp.columns = [‘均值%’, ‘胜率%’, ‘样本数’]
st.write(f”D+{n} 分层表现：”)
st.dataframe(grp, use_container_width=True)
else:
col2.info(“无触发买入记录”)
st.subheader(“回测明细”)
show_cols = [‘trade_date’,‘rank’,‘name’,‘ts_code’,‘close’,‘score’,‘tag’,‘triggered’,‘R_D1’,‘R_D3’,‘R_D5’]
show_cols = [c for c in show_cols if c in final.columns]
st.dataframe(final[show_cols], use_container_width=True)
csv = final.to_csv(index=False).encode(‘utf-8-sig’)
st.download_button(“下载回测结果CSV”, csv, f”backtest_{end_str}.csv”, “text/csv”)
else:
st.warning(“回测未产生结果，请检查日期范围或Token权限”)
