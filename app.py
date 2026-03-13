# -*- coding: utf-8 -*-

“””
Smart Screener V1.0
“””
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

pro = None

st.set_page_config(page_title=“Smart Screener V1.0”, layout=“wide”)
st.title(“Smart Screener V1.0”)
st.caption(“Run after market close. Buy on next-day gap-up + 1.5% surge. Stop loss 5%.”)

# —————————

# 全局变量

# —————————

CHECKPOINT_FILE = “bt_checkpoint.csv”
CACHE_FILE = “stock_cache.pkl”

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
df = df[~df[‘name’].str.contains(‘ST’, na=False)]
bse = df[‘ts_code’].str.startswith
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

# 技术指标

# —————————

def calc_indicators(hist, trade_date, winner_rate_dict):
if hist.empty or len(hist) < 30:
return None
close = hist[‘close’]
today = hist[hist[‘trade_date’] == trade_date]
if today.empty:
return None
idx = today.index[0]

```
ma20 = close.iloc[max(0, idx-19):idx+1].mean() if idx >= 19 else None
ma60 = close.iloc[max(0, idx-59):idx+1].mean() if idx >= 59 else None
if ma20 is None or ma60 is None:
    return None

ma20_prev = close.iloc[max(0, idx-20):idx].mean() if idx >= 20 else ma20

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
o = float(row['open'])
upper_shadow = (h - c) / c * 100
body_range = h - l
body_pos = (c - l) / body_range if body_range > 0 else 0

pct_1d = float(row['pct_chg']) if 'pct_chg' in row else 0
pct_5d = (c / float(hist.iloc[max(0,idx-5)]['close']) - 1) * 100 if idx >= 5 else 0
pct_20d = (c / float(hist.iloc[max(0,idx-20)]['close']) - 1) * 100 if idx >= 20 else 0

low_60 = hist['low'].iloc[max(0,idx-59):idx+1].min()
from_bottom = (c - low_60) / low_60 * 100 if low_60 > 0 else 0

vol_5 = hist['vol'].iloc[max(0,idx-4):idx+1].mean()
vol_20 = hist['vol'].iloc[max(0,idx-19):idx+1].mean()
vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1

consec_limit = False
if idx >= 1:
    prev_pct = float(hist.iloc[idx-1]['pct_chg']) if 'pct_chg' in hist.columns else 0
    if pct_1d >= 9.5 and prev_pct >= 9.5:
        consec_limit = True

ts_code = hist['ts_code'].iloc[0] if 'ts_code' in hist.columns else ''
winner_rate = winner_rate_dict.get(ts_code, 60)

return {
    'close': c, 'high': h, 'low': l, 'open': o,
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
# 1. 技术面 25分
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
elif rsi > 80:
    tech += 1
detail['tech'] = min(tech, 25)

# 2. 买入时机 20分
timing = 0
dev = (ind['close'] - ind['ma20']) / ind['ma20'] * 100
if dev <= 3:
    timing += 10
elif dev <= 7:
    timing += 7
elif dev <= 12:
    timing += 4
else:
    timing += 0
p5 = ind['pct_5d']
if 3 <= p5 <= 10:
    timing += 10
elif p5 < 3:
    timing += 6
elif p5 <= 15:
    timing += 4
else:
    timing += 0
detail['timing'] = min(timing, 20)

# 3. 量能健康 15分
vr = ind['vol_ratio']
if 1.2 <= vr <= 2.5:
    vol = 15
elif 1.0 <= vr < 1.2 or 2.5 < vr <= 3.5:
    vol = 8
elif vr > 3.5:
    vol = 0
else:
    vol = 3
detail['vol'] = min(vol, 15)

# 4. 鱼身判断 15分
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

# 5. 板块热度 15分
detail['sector'] = min(sector_score, 15)

# 6. 大盘环境 10分
detail['market'] = 10 if market_strong else 3

total = sum(detail.values())
return total, detail
```

# —————————

# 风险标签

# —————————

def risk_tag(ind):
fb = ind[‘from_bottom’]
p5 = ind[‘pct_5d’]
rsi = ind[‘rsi’]
if fb > 100 or ind[‘consec_limit’]:
return “High Risk”
if fb > 60 or p5 > 15 or rsi > 80:
return “Caution”
return “Safe”

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
def get_sector_scores(trade_date, industry_map):
start = (datetime.strptime(trade_date, “%Y%m%d”) - timedelta(days=20)).strftime(”%Y%m%d”)
hs300 = safe_api(‘index_daily’, ts_code=‘000300.SH’, start_date=start, end_date=trade_date)
if hs300.empty or len(hs300) < 5:
mkt_ret = 0
else:
hs300 = hs300.sort_values(‘trade_date’)
mkt_ret = (float(hs300.iloc[-1][‘close’]) / float(hs300.iloc[-5][‘close’]) - 1) * 100

```
scores = {}
try:
    sw = pro.index_classify(level='L1', src='SW2021')
    for code in sw['index_code'].tolist():
        idf = safe_api('sw_daily', index_code=code, start_date=start, end_date=trade_date)
        if idf.empty or len(idf) < 5:
            continue
        idf = idf.sort_values('trade_date')
        ret = (float(idf.iloc[-1]['close']) / float(idf.iloc[-5]['close']) - 1) * 100
        excess = ret - mkt_ret
        if excess >= 3:
            s = 15
        elif excess >= 1:
            s = 10
        elif excess >= 0:
            s = 5
        else:
            s = 0
        scores[code] = s
except:
    pass
return scores
```

# —————————

# 核心选股（单日）

# —————————

def run_screen(trade_date, top_n, min_price, min_mv, max_mv, for_backtest=False):
global pro
daily_all = safe_api(‘daily’, trade_date=trade_date)
if daily_all.empty:
return pd.DataFrame()

```
daily_basic = safe_api('daily_basic', trade_date=trade_date,
                       fields='ts_code,circ_mv,pe,pb')
chip_df = safe_api('cyq_perf', trade_date=trade_date)
winner_rate_dict = {}
if not chip_df.empty and 'winner_rate' in chip_df.columns:
    winner_rate_dict = dict(zip(chip_df['ts_code'], chip_df['winner_rate']))

basics = load_stock_basic()
industry_map = load_industry_map()
market_strong = get_market_strong(trade_date)

try:
    sector_scores = get_sector_scores(trade_date, industry_map)
except:
    sector_scores = {}

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

# winner_rate 硬性过滤
df['winner_rate'] = df['ts_code'].map(winner_rate_dict).fillna(60)
df = df[(df['winner_rate'] >= 50) & (df['winner_rate'] <= 85)]

# 取涨幅前200深度分析
df = df.sort_values('pct_chg', ascending=False).head(200)

records = []
for row in df.itertuples():
    hist = get_stock_history(row.ts_code, trade_date, lookback=90)
    if hist.empty:
        continue

    ind = calc_indicators(hist, trade_date, winner_rate_dict)
    if ind is None:
        continue

    # 硬性技术过滤
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

    buy_low = round(ind['close'], 2)
    buy_high = round(ind['close'] * 1.02, 2)
    stop_loss = round(ind['close'] * 0.95, 2)
    target = round(ind['close'] * 1.08, 2)

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
        'market': "Strong" if market_strong else "Weak",
        'tech': detail['tech'],
        'timing': detail['timing'],
        'vol_score': detail['vol'],
        'fish': detail['fish'],
        'sector': detail['sector'],
        'mkt_score': detail['market'],
        'buy_low': buy_low,
        'buy_high': buy_high,
        'stop_loss': stop_loss,
        'target': target,
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

# 侧边栏

# —————————

with st.sidebar:
st.header(“Settings”)
st.subheader(“Tushare Token”)
token = st.text_input(“Token”, type=“password”)

```
st.subheader("Stock Filter")
min_price = st.number_input("Min Price (CNY)", value=10.0, min_value=1.0, step=1.0)
min_mv = st.number_input("Min Mkt Cap (B)", value=50.0, min_value=10.0, step=10.0)
max_mv = st.number_input("Max Mkt Cap (B)", value=1000.0, min_value=100.0, step=100.0)

st.subheader("Live Screen")
top_n = st.slider("Top N Stocks", 3, 10, 5)

st.subheader("Backtest Settings")
bt_end = st.date_input("Backtest End Date", value=datetime.now().date())
bt_days = st.number_input("Backtest Days", value=30, min_value=5, max_value=90, step=5)
bt_top_n = st.number_input("Daily Top N", value=4, min_value=1, max_value=10)
resume = st.checkbox("Resume Checkpoint", value=True)

if st.button("Clear Cache"):
    for f in [CACHE_FILE, CHECKPOINT_FILE]:
        if os.path.exists(f):
            os.remove(f)
    st.success("Cache cleared")
```

# —————————

# Token初始化

# —————————

if not token:
st.info(“Please enter Tushare Token in the sidebar.”)
st.stop()

ts.set_token(token)
pro = ts.pro_api()

# —————————

# 主界面

# —————————

tab1, tab2 = st.tabs([“Live Screen”, “Backtest”])

# TAB1

with tab1:
st.subheader(“Live Screening - Today Candidates”)
st.caption(“Run after close. Check gap-up at 9:25. Buy trigger: +1.5% above open after 9:30.”)

```
col_l, col_r = st.columns([2, 1])
with col_l:
    screen_date = st.date_input("Screen Date (closing day)", value=datetime.now().date())
with col_r:
    run_screen_btn = st.button("Run Screen", use_container_width=True)

if run_screen_btn:
    date_str = screen_date.strftime("%Y%m%d")
    with st.spinner("Loading stock data..."):
        basics = load_stock_basic()
    if basics.empty:
        st.error("Cannot load stock list. Check Token.")
    else:
        with st.spinner("Analyzing market data, please wait..."):
            result = run_screen(date_str, top_n, min_price, min_mv, max_mv, for_backtest=False)
            market_state = get_market_strong(date_str)

        if result.empty:
            st.warning("No stocks found today. Try relaxing the filters.")
        else:
            st.success("Screening done, " + str(len(result)) + " candidates found")
            ms = "Strong (CSI300 above MA20)" if market_state else "Weak (CSI300 below MA20)"
            st.info("Market Status: " + ms)

            for _, row in result.iterrows():
                with st.expander("No." + str(row['rank']) + "  " + str(row['name']) + " (" + str(row['ts_code']) + ")  " + str(row['tag']) + "  Score:" + str(round(row['score'],0))):
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("D0 Chg%", str(round(row['pct_1d'],2)) + "%")
                    c2.metric("5D Chg%", str(round(row['pct_5d'],2)) + "%")
                    c3.metric("20D Chg%", str(round(row['pct_20d'],2)) + "%")
                    c4.metric("From Low%", str(round(row['from_bottom'],1)) + "%")

                    c5,c6,c7,c8 = st.columns(4)
                    c5.metric("RSI", str(row['rsi']))
                    c6.metric("Win Rate%", str(row['winner_rate']) + "%")
                    c7.metric("Vol Ratio", str(row['vol_ratio']) + "x")
                    c8.metric("Market", row['market'])

                    cc1,cc2,cc3,cc4 = st.columns(4)
                    cc1.metric("Buy Range", str(row['buy_low']) + "~" + str(row['buy_high']))
                    cc2.metric("Stop Loss", str(row['stop_loss']))
                    cc3.metric("Target", str(row['target']))
                    cc4.metric("R:R", "1:1.6")

                    st.write("6-Dim Score Breakdown:")
                    dims = ['tech','timing','vol_score','fish','sector','mkt_score']
                    labels = ['Tech/25','Timing/20','Volume/15','FishBody/15','Sector/15','Market/10']
                    maxes = [25,20,15,15,15,10]
                    cols = st.columns(6)
                    for i, (d, lb, mx) in enumerate(zip(dims, labels, maxes)):
                        val = row[d]
                        pct = int(val/mx*100)
                        cols[i].metric(lb, str(val) + "/" + str(mx))

            show_df = result[['rank','name','ts_code','close','pct_1d','pct_5d','pct_20d',
                               'from_bottom','rsi','winner_rate','score','tag','buy_low','stop_loss','target']].copy()
            show_df.columns = ['Rank','Name','Code','Price','D0%','5D%','20D%','FromLow%','RSI','WinRate%','Score','Risk','BuyRange','Stop','Target']
            st.dataframe(show_df, use_container_width=True)
            csv = show_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("Export CSV", csv, "result_" + date_str + ".csv", "text/csv")

    st.caption("For research only. Not financial advice.")
```

# TAB2

with tab2:
st.subheader(“Backtest - Strategy Validation”)
st.caption(“Simulate: next-day gap-up + intraday +1.5% trigger, sell at close after N days, stop loss 5%”)

```
run_bt = st.button("Run Backtest", use_container_width=True)

if run_bt:
    end_str = bt_end.strftime("%Y%m%d")
    cal = safe_api('trade_cal', start_date=(bt_end - timedelta(days=int(bt_days)*3)).strftime("%Y%m%d"),
                   end_date=end_str)
    if cal.empty:
        st.error("Cannot load trade calendar. Check Token.")
        st.stop()

    dates = cal[cal['is_open']==1].sort_values('cal_date', ascending=False)['cal_date'].head(int(bt_days)).tolist()
    dates = sorted(dates)

    processed = set()
    results = []

    if resume and os.path.exists(CHECKPOINT_FILE):
        try:
            ex = pd.read_csv(CHECKPOINT_FILE)
            processed = set(ex['trade_date'].astype(str).unique())
            results.append(ex)
            st.success("Resuming from checkpoint, skipped " + str(len(processed)) + " trading days")
        except:
            pass

    dates_to_run = [d for d in dates if d not in processed]

    if not dates_to_run:
        st.success("All dates processed!")
    else:
        basics = load_stock_basic()
        bar = st.progress(0, text="Backtesting...")
        err_ph = st.empty()

        for i, date in enumerate(dates_to_run):
            try:
                res = run_screen(date, int(bt_top_n), min_price, min_mv, max_mv, for_backtest=True)
                if not res.empty:
                    res['trade_date'] = date
                    first = not os.path.exists(CHECKPOINT_FILE)
                    res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=first, encoding='utf-8-sig')
                    results.append(res)
            except Exception as e:
                err_ph.warning(str(date) + " error: " + str(e))

            bar.progress((i+1)/len(dates_to_run), text="Backtesting: " + str(date) + " (" + str(i+1) + "/" + str(len(dates_to_run)) + ")")

        bar.empty()

    if results:
        final = pd.concat(results).reset_index(drop=True)
        final = final.sort_values(['trade_date','rank'], ascending=[False,True])

        st.header("Backtest Report")

        col1, col2 = st.columns(2)
        col1.metric("Total Records", str(len(final)) + " records")
        col1.metric("Trading Days", str(final['trade_date'].nunique()) + " days")

        triggered = final[final['triggered']==True] if 'triggered' in final.columns else pd.DataFrame()
        if not triggered.empty:
            trig_rate = len(triggered) / len(final) * 100
            col2.metric("Trigger Rate", str(round(trig_rate,1)) + "%")
            cols3 = st.columns(3)
            for i, n in enumerate([1,3,5]):
                key = 'R_D' + str(n)
                if key in triggered.columns:
                    valid = triggered.dropna(subset=[key])
                    if not valid.empty:
                        avg = valid[key].mean()
                        win = (valid[key] > 0).mean() * 100
                        loss = (valid[key] < -5).mean() * 100
                        cols3[i].metric("D+" + str(n) + " Avg/WinRate",
                                        str(round(avg,2)) + "% / " + str(round(win,1)) + "%",
                                        delta="Loss>5%: " + str(round(loss,1)) + "%")

            st.subheader("Performance by Rank (Triggered)")
            for n in [1,3,5]:
                key = 'R_D' + str(n)
                if key not in triggered.columns:
                    continue
                valid = triggered.dropna(subset=[key])
                if valid.empty:
                    continue
                grp = valid.groupby('rank')[key].agg(
                    avg='mean',
                    win_rate=lambda x: (x > 0).mean() * 100,
                    n='count'
                ).round(2)
                grp.columns = ['Avg%', 'WinRate%', 'Count']
                st.write("D+" + str(n) + " Performance by Rank:")
                st.dataframe(grp, use_container_width=True)
        else:
            col2.info("No triggered trades")

        st.subheader("Backtest Details")
        show_cols = ['trade_date','rank','name','ts_code','close','pct_1d','score','tag','triggered','R_D1','R_D3','R_D5']
        show_cols = [c for c in show_cols if c in final.columns]
        st.dataframe(final[show_cols], use_container_width=True)
        csv = final.to_csv(index=False).encode('utf-8-sig')
        st.download_button("Download Backtest CSV", csv, "backtest_" + end_str + ".csv", "text/csv")
    else:
        st.warning("No backtest results. Check date range or Token.")
```
