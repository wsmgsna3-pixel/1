# REBUILT_1772897923 - force cache bust

# -*- coding: utf-8 -*-

# Smart Screener V1.0

# Core idea: Find stocks just launched with room to grow, not the hottest ones.

# Universe: All A-shares, exclude ST and BSE, min price 10, min circ_mv 50bn

# Hard filters: 20d return max 30pct, 5d return max 20pct, no consecutive limit-up

# Tech filters: Price above MA20 upward, upper shadow max 5pct, body pos min 60pct

# Chip filter: winner_rate between 50 and 85 pct

# Six-dim score: Tech 25 + Timing 20 + Volume 15 + FishBody 15 + Sector 15 + Market 10

# Buy simulation: Next day gap-up plus intraday plus 1.5pct trigger, stop loss 5pct

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

# ==========================================

# \u9875\u9762\u914d\u7f6e

# ==========================================

st.set_page_config(
page_title=”\u667a\u9009\u80a1 V1.0”,
page_icon=”\U0001f4c8”,
layout=“wide”,
initial_sidebar_state=“expanded”
)

st.markdown(”””

<style>
    .main > div { padding: 0.8rem 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .score-high { color: #00d4aa; font-weight: bold; }
    .score-mid  { color: #ffa500; font-weight: bold; }
    .score-low  { color: #ff4b4b; font-weight: bold; }
    .tag-safe   { background:#00d4aa22; color:#00d4aa; padding:2px 8px; border-radius:4px; font-size:0.85rem; }
    .tag-warn   { background:#ffa50022; color:#ffa500; padding:2px 8px; border-radius:4px; font-size:0.85rem; }
    .tag-danger { background:#ff4b4b22; color:#ff4b4b; padding:2px 8px; border-radius:4px; font-size:0.85rem; }
    h1 { font-size: 1.5rem !important; }
    .stButton > button { border-radius: 8px; font-weight: bold; }
</style>

“””, unsafe_allow_html=True)

st.title(”\U0001f4c8 \u667a\u9009\u80a1 V1.0 | \u9c7c\u8eab\u7b56\u7565”)
st.caption(”\u6838\u5fc3\u7406\u5ff5\uff1a\u627e\u521a\u542f\u52a8\u3001\u8fd8\u6709\u7a7a\u95f4\u7684\u5f3a\u52bf\u80a1 | \u9ad8\u5f00+\u51b2\u9ad81.5%\u89e6\u53d1\u4e70\u5165 | \u6b62\u635f5%”)

# ==========================================

# \u5168\u5c40\u53d8\u91cf

# ==========================================

pro = None
CACHE_FILE = “smart_screener_cache.pkl”
CHECKPOINT_FILE = “smart_screener_checkpoint.csv”

# ==========================================

# Tushare \u57fa\u7840\u51fd\u6570

# ==========================================

def safe_api(func_name, max_retry=3, sleep=0.5, **kwargs):
global pro
if pro is None:
return pd.DataFrame()
func = getattr(pro, func_name, None)
if func is None:
return pd.DataFrame()
for i in range(max_retry):
try:
df = func(**kwargs)
if df is not None and not df.empty:
return df
time.sleep(sleep)
except Exception as e:
time.sleep(sleep * (i + 1))
return pd.DataFrame()

@st.cache_data(ttl=3600 * 24 * 7)
def load_stock_basic():
“””\u52a0\u8f7d\u5168A\u80a1\u57fa\u7840\u4fe1\u606f,\u6392\u9664ST\u548c\u5317\u4ea4\u6240”””
df = safe_api(‘stock_basic’, list_status=‘L’,
fields=‘ts_code,name,list_date,exchange’)
if df.empty:
return pd.DataFrame()
# \u6392\u9664ST
df = df[~df[‘name’].str.contains(‘ST|\u9000’, na=False)]
# \u6392\u9664\u5317\u4ea4\u6240(\u4ee543/83/87/92\u5f00\u5934)
df = df[~df[‘ts_code’].str.match(r’^(43|83|87|92)’)]
# \u6392\u9664\u5317\u4ea4\u6240exchange
df = df[df[‘exchange’].isin([‘SSE’, ‘SZSE’])]
return df

@st.cache_data(ttl=3600 * 24 * 7)
def load_sw_industry():
“””\u52a0\u8f7d\u7533\u4e07\u884c\u4e1a\u6620\u5c04”””
global pro
if pro is None:
return {}
try:
sw = pro.index_classify(level=‘L1’, src=‘SW2021’)
if sw.empty:
return {}
all_members = []
for idx_code in sw[‘index_code’].tolist():
df = pro.index_member(index_code=idx_code, is_new=‘Y’)
if not df.empty:
all_members.append(df[[‘con_code’, ‘index_code’]])
time.sleep(0.05)
if not all_members:
return {}
full = pd.concat(all_members).drop_duplicates(‘con_code’)
return dict(zip(full[‘con_code’], full[‘index_code’]))
except:
return {}

@st.cache_data(ttl=3600 * 24)
def get_trade_calendar(start_date, end_date):
df = safe_api(‘trade_cal’, start_date=start_date, end_date=end_date, is_open=‘1’)
if df.empty:
return []
return sorted(df[‘cal_date’].tolist())

def get_recent_trade_days(end_date_str, n):
start = (datetime.strptime(end_date_str, “%Y%m%d”) - timedelta(days=n * 3)).strftime(”%Y%m%d”)
days = get_trade_calendar(start, end_date_str)
return sorted(days, reverse=True)[:n]

# ==========================================

# \u6570\u636e\u7f13\u5b58\u7ba1\u7406

# ==========================================

def load_market_cache():
if os.path.exists(CACHE_FILE):
try:
with open(CACHE_FILE, ‘rb’) as f:
return pickle.load(f)
except:
os.remove(CACHE_FILE)
return None

def save_market_cache(data):
try:
with open(CACHE_FILE, ‘wb’) as f:
pickle.dump(data, f)
except Exception as e:
st.warning(f”\u7f13\u5b58\u4fdd\u5b58\u5931\u8d25: {e}”)

@st.cache_data(ttl=3600 * 12)
def fetch_daily_for_date(date):
daily = safe_api(‘daily’, trade_date=date)
adj   = safe_api(‘adj_factor’, trade_date=date)
basic = safe_api(‘daily_basic’, trade_date=date,
fields=‘ts_code,turnover_rate,circ_mv,pe_ttm,pb’)
chip  = safe_api(‘cyq_perf’, trade_date=date)
return {‘daily’: daily, ‘adj’: adj, ‘basic’: basic, ‘chip’: chip}

# ==========================================

# \u6280\u672f\u6307\u6807\u8ba1\u7b97

# ==========================================

def calc_indicators(close_series, vol_series, open_series, high_series, low_series):
“””\u8ba1\u7b97\u6240\u9700\u6280\u672f\u6307\u6807,\u8fd4\u56dedict”””
res = {}
n = len(close_series)
if n < 30:
return res

```
close = close_series.values.astype(float)
vol   = vol_series.values.astype(float)

# MA
res['ma5']  = np.mean(close[-5:])  if n >= 5  else np.nan
res['ma10'] = np.mean(close[-10:]) if n >= 10 else np.nan
res['ma20'] = np.mean(close[-20:]) if n >= 20 else np.nan
res['ma60'] = np.mean(close[-60:]) if n >= 60 else np.nan

# MA20\u65b9\u5411(\u7528\u6700\u8fd15\u65e5MA20\u659c\u7387)
if n >= 25:
    ma20_5d_ago = np.mean(close[-25:-5])
    res['ma20_slope'] = res['ma20'] - ma20_5d_ago
else:
    res['ma20_slope'] = 0

# MACD
def ema(arr, span):
    k = 2 / (span + 1)
    e = arr[0]
    result = [e]
    for v in arr[1:]:
        e = v * k + e * (1 - k)
        result.append(e)
    return np.array(result)

if n >= 35:
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    dif = ema12 - ema26
    dea = ema(dif, 9)
    res['macd_bar'] = (dif[-1] - dea[-1]) * 2
    res['dif'] = dif[-1]
    res['dea'] = dea[-1]
else:
    res['macd_bar'] = 0
    res['dif'] = 0
    res['dea'] = 0

# RSI(14)
if n >= 15:
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:])
    avg_loss = np.mean(loss[-14:])
    rs = avg_gain / (avg_loss + 1e-9)
    res['rsi'] = 100 - 100 / (1 + rs)
else:
    res['rsi'] = 50

# \u91cf\u80fd\uff1a\u8fd15\u65e5\u5747\u91cf vs \u8fd120\u65e5\u5747\u91cf
if n >= 20:
    avg_vol_5  = np.mean(vol[-5:])
    avg_vol_20 = np.mean(vol[-20:])
    res['vol_ratio'] = avg_vol_5 / (avg_vol_20 + 1e-9)
else:
    res['vol_ratio'] = 1.0

# \u4eca\u65e5K\u7ebf\u5f62\u6001
res['last_close'] = close[-1]
res['last_open']  = open_series.values[-1]
res['last_high']  = high_series.values[-1]
res['last_low']   = low_series.values[-1]

upper_shadow = (res['last_high'] - res['last_close']) / (res['last_close'] + 1e-9) * 100
res['upper_shadow'] = upper_shadow

range_len = res['last_high'] - res['last_low']
if range_len > 0:
    res['body_pos'] = (res['last_close'] - res['last_low']) / range_len
else:
    res['body_pos'] = 1.0

# \u6da8\u5e45\u8ba1\u7b97
if n >= 2:
    res['pct_1d'] = (close[-1] - close[-2]) / (close[-2] + 1e-9) * 100
if n >= 6:
    res['pct_5d'] = (close[-1] - close[-6]) / (close[-6] + 1e-9) * 100
if n >= 21:
    res['pct_20d'] = (close[-1] - close[-21]) / (close[-21] + 1e-9) * 100

# \u9c7c\u8eab\u5224\u65ad\uff1a60\u65e5\u5185\u4f4e\u70b9\u5230\u73b0\u5728\u7684\u6da8\u5e45
if n >= 60:
    low_60 = np.min(close[-60:])
    res['from_bottom_pct'] = (close[-1] - low_60) / (low_60 + 1e-9) * 100
elif n >= 20:
    low_n = np.min(close[-n:])
    res['from_bottom_pct'] = (close[-1] - low_n) / (low_n + 1e-9) * 100
else:
    res['from_bottom_pct'] = 0

# \u8fde\u7eed\u6da8\u505c\u68c0\u6d4b(\u7528\u6700\u8fd12\u65e5\u6da8\u5e45\u5224\u65ad)
if n >= 3:
    pct_today = (close[-1] - close[-2]) / (close[-2] + 1e-9) * 100
    pct_yest  = (close[-2] - close[-3]) / (close[-3] + 1e-9) * 100
    res['consecutive_limit'] = (pct_today >= 9.5 and pct_yest >= 9.5)
else:
    res['consecutive_limit'] = False

return res
```

# ==========================================

# \u516d\u7ef4\u8bc4\u5206\u7cfb\u7edf

# ==========================================

def calc_score(ind, winner_rate, sector_boost, market_strong):
“””
\u516d\u7ef4\u8bc4\u5206(\u6ee1\u5206100)\uff1a
- \u6280\u672f\u9762        25\u5206
- \u4e70\u5165\u65f6\u673a      20\u5206
- \u91cf\u80fd\u5065\u5eb7      15\u5206
- \u9c7c\u8eab\u5224\u65ad      15\u5206
- \u677f\u5757\u70ed\u5ea6      15\u5206
- \u5927\u76d8\u73af\u5883      10\u5206
“””
score = 0
detail = {}

```
# ===== 1. \u6280\u672f\u9762 (25\u5206) =====
tech = 0
# MA\u8d8b\u52bf(\u80a1\u4ef7\u5728MA20\u4e0a\u65b9\u4e14MA20\u5411\u4e0a)
if ind.get('last_close', 0) > ind.get('ma20', 0) and ind.get('ma20_slope', 0) > 0:
    tech += 10
elif ind.get('last_close', 0) > ind.get('ma20', 0):
    tech += 5
# MA20\u5728MA60\u4e0a\u65b9(\u4e2d\u957f\u671f\u8d8b\u52bf)
if ind.get('ma20', 0) > ind.get('ma60', 0):
    tech += 5
# MACD\uff1aDIF\u5728\u96f6\u8f74\u4e0a\u65b9\u4e14MACD\u67f1\u4e3a\u6b63
if ind.get('dif', 0) > 0 and ind.get('macd_bar', 0) > 0:
    tech += 7
elif ind.get('dif', 0) > 0 or ind.get('macd_bar', 0) > 0:
    tech += 3
# RSI\u5065\u5eb7\u533a\u95f455-70\u5f97\u6ee1\u5206,\u504f\u79bb\u6263\u5206
rsi = ind.get('rsi', 50)
if 55 <= rsi <= 70:
    tech += 3
elif 50 <= rsi < 55 or 70 < rsi <= 75:
    tech += 1
detail['\u6280\u672f\u9762'] = min(tech, 25)

# ===== 2. \u4e70\u5165\u65f6\u673a (20\u5206) =====
timing = 0
# \u8dddMA20\u504f\u79bb\u5ea6\uff1a\u504f\u79bb\u8d8a\u5c0f\u8d8a\u597d
ma20 = ind.get('ma20', 0)
last_close = ind.get('last_close', 0)
if ma20 > 0:
    deviation = (last_close - ma20) / ma20 * 100
    if 0 <= deviation <= 3:
        timing += 12
    elif 3 < deviation <= 6:
        timing += 8
    elif 6 < deviation <= 10:
        timing += 4
    elif deviation < 0:
        timing += 0  # \u8dcc\u7834MA20\u4e0d\u52a0\u5206(\u5df2\u88ab\u786c\u6027\u8fc7\u6ee4)
# \u8fd15\u65e5\u6da8\u5e45\uff1a\u6e29\u548c\u4e0a\u6da8\u5f97\u5206,\u8fc7\u70ed\u964d\u5206
pct_5d = ind.get('pct_5d', 0)
if 3 <= pct_5d <= 10:
    timing += 8
elif 0 <= pct_5d < 3:
    timing += 5
elif 10 < pct_5d <= 15:
    timing += 3
elif pct_5d > 15:
    timing += 0
detail['\u4e70\u5165\u65f6\u673a'] = min(timing, 20)

# ===== 3. \u91cf\u80fd\u5065\u5eb7 (15\u5206) =====
vol = 0
vol_ratio = ind.get('vol_ratio', 1.0)
# \u8fd15\u65e5\u5747\u91cf\u662f\u8fd120\u65e5\u5747\u91cf\u76841.2-2.5\u500d\uff1a\u6e29\u548c\u653e\u5927
if 1.2 <= vol_ratio <= 2.5:
    vol += 15
elif 1.0 <= vol_ratio < 1.2:
    vol += 8
elif 2.5 < vol_ratio <= 3.5:
    vol += 5  # \u653e\u91cf\u504f\u5927,\u7565\u964d\u5206
elif vol_ratio > 3.5:
    vol += 0  # \u7206\u91cf,\u5371\u9669
elif vol_ratio < 1.0:
    vol += 3  # \u7f29\u91cf,\u52a8\u80fd\u4e0d\u8db3
detail['\u91cf\u80fd\u5065\u5eb7'] = min(vol, 15)

# ===== 4. \u9c7c\u8eab\u5224\u65ad (15\u5206) =====
fish = 0
from_bottom = ind.get('from_bottom_pct', 0)
if from_bottom <= 30:
    fish += 15   # \u9c7c\u8eab\u524d\u6bb5,\u7a7a\u95f4\u5145\u8db3
elif from_bottom <= 60:
    fish += 10   # \u9c7c\u8eab\u4e2d\u6bb5,\u4ecd\u6709\u673a\u4f1a
elif from_bottom <= 100:
    fish += 4    # \u9c7c\u5c3e\u8b66\u544a,\u964d\u5206
else:
    fish += 0    # \u6da8\u5e45\u8fc7\u5927,\u5371\u9669
detail['\u9c7c\u8eab\u5224\u65ad'] = fish

# ===== 5. \u677f\u5757\u70ed\u5ea6 (15\u5206) =====
sector = 0
if sector_boost > 3:
    sector += 15
elif sector_boost > 1.5:
    sector += 10
elif sector_boost > 0:
    sector += 5
detail['\u677f\u5757\u70ed\u5ea6'] = sector

# ===== 6. \u5927\u76d8\u73af\u5883 (10\u5206) =====
mkt = 10 if market_strong else 3
detail['\u5927\u76d8\u73af\u5883'] = mkt

total = sum(detail.values())

# \u98ce\u9669\u6807\u7b7e
if from_bottom > 100 or ind.get('consecutive_limit', False):
    tag = '\U0001f534 \u9ad8\u98ce\u9669'
elif from_bottom > 60 or pct_5d > 15 or rsi > 80:
    tag = '\U0001f7e1 \u8c28\u614e'
else:
    tag = '\U0001f7e2 \u5b89\u5168'

return total, detail, tag
```

# ==========================================

# \u5355\u53ea\u80a1\u7968\u5b8c\u6574\u6570\u636e\u83b7\u53d6(\u7528\u4e8e\u56de\u6d4b\u548c\u5b9e\u76d8)

# ==========================================

@st.cache_data(ttl=3600 * 12)
def get_stock_history(ts_code, end_date, lookback_days=120):
start = (datetime.strptime(end_date, “%Y%m%d”) - timedelta(days=lookback_days * 2)).strftime(”%Y%m%d”)
daily = safe_api(‘daily’, ts_code=ts_code, start_date=start, end_date=end_date)
if daily is None or daily.empty or len(daily) < 30:
return pd.DataFrame()
adj = safe_api(‘adj_factor’, ts_code=ts_code, start_date=start, end_date=end_date)
if adj.empty:
return daily.sort_values(‘trade_date’).reset_index(drop=True)
# \u524d\u590d\u6743\u5904\u7406
daily = daily.merge(adj[[‘ts_code’, ‘trade_date’, ‘adj_factor’]], on=[‘ts_code’, ‘trade_date’], how=‘left’)
daily[‘adj_factor’] = daily[‘adj_factor’].fillna(method=‘ffill’).fillna(1.0)
latest_adj = daily[‘adj_factor’].iloc[0]  # \u6570\u636e\u964d\u5e8f,\u7b2c\u4e00\u6761\u662f\u6700\u65b0
daily = daily.sort_values(‘trade_date’).reset_index(drop=True)
latest_adj = daily[‘adj_factor’].iloc[-1]
for col in [‘open’, ‘high’, ‘low’, ‘close’]:
daily[col] = daily[col] * daily[‘adj_factor’] / latest_adj
return daily

# ==========================================

# \u677f\u5757\u70ed\u5ea6\u8ba1\u7b97

# ==========================================

@st.cache_data(ttl=3600 * 12)
def get_sector_performance(trade_date, industry_map):
“””\u8ba1\u7b97\u5404\u7533\u4e07\u884c\u4e1a5\u65e5\u76f8\u5bf9\u5927\u76d8\u8868\u73b0”””
start = (datetime.strptime(trade_date, “%Y%m%d”) - timedelta(days=15)).strftime(”%Y%m%d”)
# \u5927\u76d85\u65e5\u6da8\u5e45
hs300 = safe_api(‘daily’, ts_code=‘000300.SH’, start_date=start, end_date=trade_date)
if hs300.empty or len(hs300) < 5:
return {}
hs300 = hs300.sort_values(‘trade_date’)
mkt_5d = (hs300[‘close’].iloc[-1] - hs300[‘close’].iloc[-5]) / hs300[‘close’].iloc[-5] * 100

```
# \u7533\u4e07\u884c\u4e1a\u6307\u65705\u65e5\u6da8\u5e45
try:
    sw_daily = pro.sw_daily(trade_date=trade_date)
    if sw_daily.empty:
        return {}
    sw_daily['relative_5d'] = sw_daily.get('pct_chg', 0)
    return {row['index_code']: row.get('pct_chg', 0) - mkt_5d
            for _, row in sw_daily.iterrows()}
except:
    return {}
```

# ==========================================

# \u5927\u76d8\u72b6\u6001\u5224\u65ad

# ==========================================

@st.cache_data(ttl=3600 * 12)
def get_market_state(trade_date):
start = (datetime.strptime(trade_date, “%Y%m%d”) - timedelta(days=60)).strftime(”%Y%m%d”)
df = safe_api(‘daily’, ts_code=‘000300.SH’, start_date=start, end_date=trade_date)
if df.empty or len(df) < 20:
return False
df = df.sort_values(‘trade_date’)
return df[‘close’].iloc[-1] > df[‘close’].tail(20).mean()

# ==========================================

# \u6838\u5fc3\u9009\u80a1\u51fd\u6570(\u5355\u65e5)

# ==========================================

def screen_one_day(trade_date, stock_basic_df, industry_map,
min_price, min_mv, max_mv,
top_n=5, for_backtest=False):
“””
\u6267\u884c\u5355\u65e5\u9009\u80a1,\u8fd4\u56deDataFrame
for_backtest=True\u65f6\u540c\u65f6\u83b7\u53d6\u672a\u6765\u6536\u76ca
“””
# \u83b7\u53d6\u5f53\u65e5\u884c\u60c5
daily_all = safe_api(‘daily’, trade_date=trade_date)
if daily_all.empty:
return pd.DataFrame()

```
daily_basic = safe_api('daily_basic', trade_date=trade_date,
                       fields='ts_code,turnover_rate,circ_mv,pe_ttm,pb')
chip_df = safe_api('cyq_perf', trade_date=trade_date)

chip_dict = {}
if not chip_df.empty and 'winner_rate' in chip_df.columns:
    chip_dict = dict(zip(chip_df['ts_code'], chip_df['winner_rate']))

# \u677f\u5757\u70ed\u5ea6
sector_perf = {}
try:
    sector_perf = get_sector_performance(trade_date, industry_map)
except:
    pass

# \u5927\u76d8\u72b6\u6001
market_strong = get_market_state(trade_date)

# \u5408\u5e76\u57fa\u7840\u4fe1\u606f
df = daily_all.merge(stock_basic_df[['ts_code', 'name']], on='ts_code', how='inner')
if not daily_basic.empty:
    df = df.merge(daily_basic, on='ts_code', how='left')
else:
    df['circ_mv'] = np.nan
    df['turnover_rate'] = np.nan

# ===== \u786c\u6027\u8fc7\u6ee4 =====
# \u4ef7\u683c\u8fc7\u6ee4
df = df[df['close'] >= min_price]
# \u5e02\u503c\u8fc7\u6ee4(circ_mv\u5355\u4f4d\uff1a\u4e07\u5143)
if 'circ_mv' in df.columns:
    df = df[df['circ_mv'].notna()]
    df = df[(df['circ_mv'] >= min_mv * 10000) & (df['circ_mv'] <= max_mv * 10000)]
# \u53ea\u4fdd\u7559\u6709\u540d\u5b57\u7684(\u5df2\u6392\u9664ST)
df = df[df['name'].notna() & (df['name'] != '')]

if df.empty:
    return pd.DataFrame()

# \u6309\u5f53\u65e5\u6da8\u5e45\u6392\u5e8f,\u53d6\u524d200\u53ea\u505a\u6df1\u5ea6\u5206\u6790(\u63d0\u5347\u6548\u7387)
df = df.sort_values('pct_chg', ascending=False).head(200)

records = []
for row in df.itertuples():
    ts_code = row.ts_code

    # winner_rate\u786c\u6027\u8fc7\u6ee4\uff1a50-85\u4e4b\u95f4
    wr = chip_dict.get(ts_code, 70)
    if not (50 <= wr <= 85):
        continue

    # \u83b7\u53d6\u5386\u53f2\u6570\u636e
    hist = get_stock_history(ts_code, trade_date, lookback_days=90)
    if hist.empty or len(hist) < 30:
        continue

    # \u8ba1\u7b97\u6280\u672f\u6307\u6807
    ind = calc_indicators(
        hist['close'], hist['vol'],
        hist['open'], hist['high'], hist['low']
    )
    if not ind:
        continue

    # ===== \u786c\u6027\u6280\u672f\u8fc7\u6ee4 =====
    # \u80a1\u4ef7\u5fc5\u987b\u5728MA20\u4e0a\u65b9
    if ind.get('last_close', 0) < ind.get('ma20', 1):
        continue
    # MA20\u5fc5\u987b\u5411\u4e0a
    if ind.get('ma20_slope', 0) <= 0:
        continue
    # \u4e0a\u5f71\u7ebf<=5%
    if ind.get('upper_shadow', 99) > 5:
        continue
    # \u5b9e\u4f53\u4f4d\u7f6e>=60%
    if ind.get('body_pos', 0) < 0.6:
        continue
    # \u6392\u9664\u8fde\u7eed\u6da8\u505c
    if ind.get('consecutive_limit', False):
        continue
    # 20\u65e5\u6da8\u5e45<=30%
    if ind.get('pct_20d', 0) > 30:
        continue
    # 5\u65e5\u6da8\u5e45<=20%
    if ind.get('pct_5d', 0) > 20:
        continue

    # \u677f\u5757\u70ed\u5ea6
    ind_code = industry_map.get(ts_code, '')
    s_boost = sector_perf.get(ind_code, 0) if ind_code else 0

    # \u8ba1\u7b97\u7efc\u5408\u8bc4\u5206
    total_score, detail, tag = calc_score(ind, wr, s_boost, market_strong)

    # \u5efa\u8bae\u4e70\u5165\u4ef7\u533a\u95f4(\u6628\u6536+-0.5%\u5230+2%)
    buy_low  = ind['last_close'] * 1.000
    buy_high = ind['last_close'] * 1.020
    stop_loss = buy_low * 0.95
    target    = buy_low * 1.08

    rec = {
        'ts_code': ts_code,
        'name': row.name,
        'close': ind['last_close'],
        'pct_1d': round(ind.get('pct_1d', 0), 2),
        'pct_5d': round(ind.get('pct_5d', 0), 2),
        'pct_20d': round(ind.get('pct_20d', 0), 2),
        'from_bottom': round(ind.get('from_bottom_pct', 0), 1),
        'rsi': round(ind.get('rsi', 0), 1),
        'winner_rate': round(wr, 1),
        'vol_ratio': round(ind.get('vol_ratio', 1), 2),
        'score': round(total_score, 1),
        'tag': tag,
        'detail': detail,
        'buy_low': round(buy_low, 2),
        'buy_high': round(buy_high, 2),
        'stop_loss': round(stop_loss, 2),
        'target': round(target, 2),
        'market': '\u5f3a\u52bf' if market_strong else '\u5f31\u52bf',
        'sector_boost': round(s_boost, 2),
        '\u6280\u672f\u9762': detail.get('\u6280\u672f\u9762', 0),
        '\u4e70\u5165\u65f6\u673a': detail.get('\u4e70\u5165\u65f6\u673a', 0),
        '\u91cf\u80fd\u5065\u5eb7': detail.get('\u91cf\u80fd\u5065\u5eb7', 0),
        '\u9c7c\u8eab\u5224\u65ad': detail.get('\u9c7c\u8eab\u5224\u65ad', 0),
        '\u677f\u5757\u70ed\u5ea6': detail.get('\u677f\u5757\u70ed\u5ea6', 0),
        '\u5927\u76d8\u73af\u5883': detail.get('\u5927\u76d8\u73af\u5883', 0),
    }

    # \u56de\u6d4b\u6a21\u5f0f\uff1a\u6a21\u62df\u6b21\u65e5\u9ad8\u5f00+\u51b2\u9ad81.5%\u4e70\u5165
    if for_backtest:
        future = simulate_buy(ts_code, trade_date, ind['last_close'])
        rec.update(future)

    records.append(rec)

if not records:
    return pd.DataFrame()

result = pd.DataFrame(records)
result = result.sort_values('score', ascending=False).head(top_n).reset_index(drop=True)
result.insert(0, 'rank', range(1, len(result) + 1))
return result
```

# ==========================================

# \u56de\u6d4b\u6a21\u62df\u4e70\u5165\u51fd\u6570

# ==========================================

def simulate_buy(ts_code, selection_date, d0_close):
“””\u6a21\u62df\u6b21\u65e5\u9ad8\u5f00+\u76d8\u4e2d\u51b2\u9ad81.5%\u89e6\u53d1\u4e70\u5165,\u8ba1\u7b97D1/D3/D5\u6536\u76ca”””
d0 = datetime.strptime(selection_date, “%Y%m%d”)
start = (d0 + timedelta(days=1)).strftime(”%Y%m%d”)
end   = (d0 + timedelta(days=20)).strftime(”%Y%m%d”)

```
hist = safe_api('daily', ts_code=ts_code, start_date=start, end_date=end)
result = {'R_D1': np.nan, 'R_D3': np.nan, 'R_D5': np.nan,
          'triggered': False, 'buy_price': np.nan}

if hist is None or hist.empty:
    return result

hist = hist.sort_values('trade_date').reset_index(drop=True)
if len(hist) < 1:
    return result

d1 = hist.iloc[0]
next_open  = float(d1['open'])
next_high  = float(d1['high'])

# \u89e6\u53d1\u6761\u4ef6\uff1a\u9ad8\u5f00(\u5f00\u76d8\u4ef7>\u6628\u6536)\u4e14\u76d8\u4e2d\u51b2\u9ad8\u8d85\u8fc7\u5f00\u76d8\u4ef71.5%
if next_open <= d0_close:
    return result
trigger_price = next_open * 1.015
if next_high < trigger_price:
    return result

# \u89e6\u53d1\u6210\u529f
result['triggered'] = True
result['buy_price'] = round(trigger_price, 2)

for n, key in [(1, 'R_D1'), (3, 'R_D3'), (5, 'R_D5')]:
    if len(hist) >= n:
        sell_price = float(hist.iloc[n - 1]['close'])
        result[key] = round((sell_price - trigger_price) / trigger_price * 100, 2)

return result
```

# ==========================================

# Streamlit UI

# ==========================================

# ===== \u4fa7\u8fb9\u680f\u53c2\u6570 =====

with st.sidebar:
st.header(”\u2699\ufe0f \u53c2\u6570\u8bbe\u7f6e”)

```
st.subheader("\U0001f511 Tushare Token")
ts_token = st.text_input("Token", type="password", key="token")

st.divider()
st.subheader("\U0001f4b0 \u80a1\u7968\u6c60\u8fc7\u6ee4")
min_price = st.number_input("\u6700\u4f4e\u80a1\u4ef7(\u5143)", value=10.0, min_value=1.0, step=1.0)
min_mv    = st.number_input("\u6700\u5c0f\u6d41\u901a\u5e02\u503c(\u4ebf)", value=50.0, min_value=10.0, step=10.0)
max_mv    = st.number_input("\u6700\u5927\u6d41\u901a\u5e02\u503c(\u4ebf)", value=1000.0, min_value=100.0, step=100.0)

st.divider()
st.subheader("\U0001f4ca \u5b9e\u76d8\u9009\u80a1")
top_n = st.slider("\u8f93\u51faTop N\u5019\u9009\u80a1", 3, 10, 5)

st.divider()
st.subheader("\U0001f52c \u56de\u6d4b\u8bbe\u7f6e")
bt_end   = st.date_input("\u56de\u6d4b\u622a\u6b62\u65e5\u671f", value=datetime.now().date())
bt_days  = st.number_input("\u56de\u6d4b\u5929\u6570", value=30, min_value=5, max_value=90, step=5)
bt_top_n = st.number_input("\u6bcf\u65e5\u63a8\u8350\u6570", value=4, min_value=1, max_value=10)
resume   = st.checkbox("\u5f00\u542f\u65ad\u70b9\u7eed\u4f20", value=True)

if st.button("\U0001f5d1\ufe0f \u6e05\u9664\u7f13\u5b58"):
    for f in [CACHE_FILE, CHECKPOINT_FILE]:
        if os.path.exists(f):
            os.remove(f)
    st.cache_data.clear()
    st.success("\u7f13\u5b58\u5df2\u6e05\u9664")
```

# ===== Token\u521d\u59cb\u5316 =====

if not ts_token:
st.info(”\U0001f448 \u8bf7\u5728\u5de6\u4fa7\u8f93\u5165 Tushare Token \u540e\u5f00\u59cb\u4f7f\u7528”)
st.stop()

ts.set_token(ts_token)
pro = ts.pro_api()

# ===== \u4e3b\u754c\u9762\u6807\u7b7e\u9875 =====

tab1, tab2 = st.tabs([”\U0001f4e1 \u5b9e\u76d8\u9009\u80a1”, “\U0001f52c \u5386\u53f2\u56de\u6d4b”])

# ==========================================

# TAB1\uff1a\u5b9e\u76d8\u9009\u80a1

# ==========================================

with tab1:
st.subheader(”\U0001f4e1 \u5b9e\u76d8\u9009\u80a1 | \u4eca\u65e5\u5019\u9009”)
st.caption(”\u6536\u76d8\u540e\u8fd0\u884c,\u7b2c\u4e8c\u59299:25\u96c6\u5408\u7ade\u4ef7\u5224\u65ad\u9ad8\u5f00,9:30\u540e\u51b2\u9ad81.5%\u89e6\u53d1\u4e70\u5165”)

```
col1, col2 = st.columns([3, 1])
with col1:
    screen_date = st.date_input("\u9009\u80a1\u65e5\u671f(\u9009\u5f53\u5929\u6536\u76d8\u65e5)",
                                 value=datetime.now().date(), key='screen_date')
with col2:
    st.write("")
    st.write("")
    run_screen = st.button("\U0001f680 \u5f00\u59cb\u9009\u80a1", use_container_width=True)

if run_screen:
    date_str = screen_date.strftime("%Y%m%d")

    with st.spinner("\u52a0\u8f7d\u80a1\u7968\u57fa\u7840\u6570\u636e..."):
        stock_basic = load_stock_basic()
        industry_map = load_sw_industry()

    if stock_basic.empty:
        st.error("\u65e0\u6cd5\u83b7\u53d6\u80a1\u7968\u5217\u8868,\u8bf7\u68c0\u67e5Token\u6743\u9650")
        st.stop()

    with st.spinner(f"\u6b63\u5728\u5206\u6790 {date_str} \u5168\u5e02\u573a\u6570\u636e,\u8bf7\u7a0d\u5019..."):
        result = screen_one_day(
            date_str, stock_basic, industry_map,
            min_price, min_mv, max_mv,
            top_n=top_n, for_backtest=False
        )

    if result.empty:
        st.warning("\u4eca\u65e5\u672a\u627e\u5230\u7b26\u5408\u6761\u4ef6\u7684\u80a1\u7968,\u53ef\u9002\u5f53\u653e\u5bbd\u53c2\u6570")
    else:
        st.success(f"\u2705 \u7b5b\u9009\u5b8c\u6210,\u5171\u63a8\u8350 {len(result)} \u53ea\u5019\u9009\u80a1")

        market_state = get_market_state(date_str)
        st.info(f"\U0001f4ca \u5f53\u524d\u5927\u76d8\u72b6\u6001\uff1a{'\U0001f7e2 \u5f3a\u52bf(\u6caa\u6df1300\u7ad9\u4e0aMA20)' if market_state else '\U0001f534 \u5f31\u52bf(\u6caa\u6df1300\u8dcc\u7834MA20)'}")

        # \u8be6\u7ec6\u5361\u7247\u5c55\u793a
        for _, row in result.iterrows():
            with st.expander(
                f"#{int(row['rank'])}  {row['name']} ({row['ts_code']})  "
                f"CNY{row['close']:.2f}  {row['tag']}  \u7efc\u5408\u8bc4\u5206: {row['score']:.0f}\u5206",
                expanded=(row['rank'] <= 2)
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("\u4eca\u65e5\u6da8\u5e45", f"{row['pct_1d']:+.2f}%")
                c2.metric("5\u65e5\u6da8\u5e45",  f"{row['pct_5d']:+.2f}%")
                c3.metric("20\u65e5\u6da8\u5e45", f"{row['pct_20d']:+.2f}%")
                c4.metric("\u672c\u8f6e\u6da8\u5e45", f"{row['from_bottom']:+.1f}%",
                          help="\u4ece\u8fd160\u65e5\u4f4e\u70b9\u5230\u73b0\u5728\u7684\u6da8\u5e45,\u5224\u65ad\u9c7c\u8eab/\u9c7c\u5c3e\u4f4d\u7f6e")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("RSI(14)", f"{row['rsi']:.1f}")
                c6.metric("\u7b79\u7801\u83b7\u5229\u6bd4", f"{row['winner_rate']:.1f}%")
                c7.metric("\u91cf\u6bd4(5/20\u65e5)", f"{row['vol_ratio']:.2f}x")
                c8.metric("\u5927\u76d8\u73af\u5883", row['market'])

                st.divider()
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("\U0001f4cc \u5efa\u8bae\u4e70\u5165\u4ef7\u533a\u95f4",
                           f"CNY{row['buy_low']:.2f} ~ CNY{row['buy_high']:.2f}")
                cc2.metric("\U0001f6e1\ufe0f \u6b62\u635f\u4ef7(-5%)", f"CNY{row['stop_loss']:.2f}")
                cc3.metric("\U0001f3af \u76ee\u6807\u4ef7(+8%)", f"CNY{row['target']:.2f}")
                cc4.metric("\u98ce\u9669\u6536\u76ca\u6bd4", "1 : 1.6")

                st.divider()
                st.write("**\u516d\u7ef4\u8bc4\u5206\u5206\u89e3\uff1a**")
                score_cols = st.columns(6)
                dims = ['\u6280\u672f\u9762','\u4e70\u5165\u65f6\u673a','\u91cf\u80fd\u5065\u5eb7','\u9c7c\u8eab\u5224\u65ad','\u677f\u5757\u70ed\u5ea6','\u5927\u76d8\u73af\u5883']
                maxs = [25, 20, 15, 15, 15, 10]
                for i, (dim, mx) in enumerate(zip(dims, maxs)):
                    val = row.get(dim, 0)
                    pct = val / mx * 100
                    color = "\U0001f7e2" if pct >= 70 else "\U0001f7e1" if pct >= 40 else "\U0001f534"
                    score_cols[i].metric(f"{color} {dim}", f"{val}/{mx}")

        # \u6c47\u603b\u8868\u683c
        st.divider()
        st.write("**\u6c47\u603b\u8868\u683c\uff1a**")
        show_df = result[['rank','name','ts_code','close','pct_1d','pct_5d',
                           'pct_20d','from_bottom','rsi','winner_rate',
                           'score','tag','buy_low','stop_loss','target']].copy()
        show_df.columns = ['\u6392\u540d','\u540d\u79f0','\u4ee3\u7801','\u73b0\u4ef7','\u4eca\u65e5%','5\u65e5%','20\u65e5%',
                            '\u672c\u8f6e\u6da8\u5e45%','RSI','\u7b79\u7801\u83b7\u5229%','\u8bc4\u5206','\u98ce\u9669','\u5efa\u8bae\u4e70\u5165','\u6b62\u635f','\u76ee\u6807']
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        csv = show_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("\U0001f4e5 \u5bfc\u51faCSV", csv,
                           f"\u667a\u9009\u80a1_{date_str}.csv", "text/csv")

    st.divider()
    st.caption("\u26a0\ufe0f \u672c\u5de5\u5177\u4ec5\u4f9b\u5b66\u4e60\u7814\u7a76,\u4e0d\u6784\u6210\u6295\u8d44\u5efa\u8bae\u3002\u80a1\u5e02\u6709\u98ce\u9669,\u6295\u8d44\u9700\u8c28\u614e\u3002")
```

# ==========================================

# TAB2\uff1a\u5386\u53f2\u56de\u6d4b

# ==========================================

with tab2:
st.subheader(”\U0001f52c \u5386\u53f2\u56de\u6d4b | \u7b56\u7565\u9a8c\u8bc1”)
st.caption(”\u6a21\u62df\uff1a\u6b21\u65e5\u9ad8\u5f00 + \u76d8\u4e2d\u51b2\u9ad81.5% \u89e6\u53d1\u4e70\u5165,\u6301\u6709N\u5929\u540e\u6536\u76d8\u4ef7\u5356\u51fa,\u6b62\u635f5%”)

```
run_bt = st.button("\U0001f680 \u542f\u52a8\u56de\u6d4b", use_container_width=True)

if run_bt:
    end_str = bt_end.strftime("%Y%m%d")
    trade_days = get_recent_trade_days(end_str, int(bt_days))

    if not trade_days:
        st.error("\u65e0\u6cd5\u83b7\u53d6\u4ea4\u6613\u65e5\u5386,\u8bf7\u68c0\u67e5Token")
        st.stop()

    # \u65ad\u70b9\u7eed\u4f20
    processed = set()
    all_results = []

    if resume and os.path.exists(CHECKPOINT_FILE):
        try:
            ckpt = pd.read_csv(CHECKPOINT_FILE)
            ckpt['trade_date'] = ckpt['trade_date'].astype(str)
            processed = set(ckpt['trade_date'].unique())
            all_results.append(ckpt)
            st.success(f"\u2705 \u8bfb\u53d6\u65ad\u70b9\u5b58\u6863,\u5df2\u8df3\u8fc7 {len(processed)} \u4e2a\u4ea4\u6613\u65e5")
        except:
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
    elif not resume and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    dates_to_run = [d for d in trade_days if d not in processed]

    if not dates_to_run:
        st.success("\U0001f389 \u6240\u6709\u65e5\u671f\u5df2\u8ba1\u7b97\u5b8c\u6bd5\uff01")
    else:
        with st.spinner("\u52a0\u8f7d\u80a1\u7968\u57fa\u7840\u6570\u636e..."):
            stock_basic = load_stock_basic()
            industry_map = load_sw_industry()

        bar = st.progress(0, text="\u56de\u6d4b\u4e2d...")
        err_placeholder = st.empty()

        for i, date in enumerate(dates_to_run):
            bar.progress((i + 1) / len(dates_to_run),
                          text=f"\u56de\u6d4b\u4e2d: {date}  ({i+1}/{len(dates_to_run)})")
            try:
                day_result = screen_one_day(
                    date, stock_basic, industry_map,
                    min_price, min_mv, max_mv,
                    top_n=int(bt_top_n), for_backtest=True
                )
                if not day_result.empty:
                    day_result['trade_date'] = date
                    is_first = not os.path.exists(CHECKPOINT_FILE)
                    day_result.to_csv(CHECKPOINT_FILE, mode='a',
                                      index=False, header=is_first,
                                      encoding='utf-8-sig')
                    all_results.append(day_result)
            except Exception as e:
                err_placeholder.warning(f"{date} \u5904\u7406\u5f02\u5e38: {e}")

        bar.empty()

    # ===== \u56de\u6d4b\u7ed3\u679c\u5c55\u793a =====
    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        final['trade_date'] = final['trade_date'].astype(str)
        final = final.sort_values(['trade_date', 'rank'], ascending=[False, True])

        st.divider()
        st.header("\U0001f4ca \u56de\u6d4b\u7edf\u8ba1\u62a5\u544a")

        # \u603b\u4f53\u7edf\u8ba1
        triggered = final[final['triggered'] == True] if 'triggered' in final.columns else pd.DataFrame()

        col1, col2, col3 = st.columns(3)
        col1.metric("\u603b\u9009\u80a1\u8bb0\u5f55", f"{len(final)} \u6761")
        col1.metric("\u6d89\u53ca\u4ea4\u6613\u65e5", f"{final['trade_date'].nunique()} \u5929")

        if not triggered.empty:
            col2.metric("\u89e6\u53d1\u4e70\u5165\u6bd4\u4f8b",
                        f"{len(triggered)/len(final)*100:.1f}%",
                        help="\u6ee1\u8db3\u9ad8\u5f00+\u51b2\u9ad81.5%\u7684\u6bd4\u4f8b")
            for n, key in [(1,'R_D1'), (3,'R_D3'), (5,'R_D5')]:
                valid = triggered.dropna(subset=[key])
                if not valid.empty:
                    avg  = valid[key].mean()
                    wr   = (valid[key] > 0).mean() * 100
                    loss = (valid[key] < -5).mean() * 100
                    col3.metric(
                        f"D+{n} \u5747\u6536\u76ca/\u80dc\u7387",
                        f"{avg:+.2f}% / {wr:.1f}%",
                        delta=f"\u4e8f\u635f>5%: {loss:.1f}%"
                    )
        else:
            col2.info("\u65e0\u89e6\u53d1\u4e70\u5165\u8bb0\u5f55(\u53ef\u80fd\u65e5\u671f\u8303\u56f4\u5185\u4fe1\u53f7\u8f83\u5c11)")

        # Rank\u5206\u5c42\u5206\u6790
        if not triggered.empty:
            st.divider()
            st.subheader("\U0001f4c8 \u6309\u6392\u540d\u5206\u5c42\u5206\u6790(\u89e6\u53d1\u4e70\u5165)")
            for n, key in [(1,'R_D1'), (3,'R_D3'), (5,'R_D5')]:
                valid = triggered.dropna(subset=[key])
                if valid.empty:
                    continue
                grp = valid.groupby('rank')[key].agg(
                    avg='mean', win_rate=lambda x: (x > 0).mean() * 100, n_count='count'
                ).round(2)
                grp.columns = ['\u5747\u503c', '\u80dc\u7387%', '\u6837\u672c\u6570']
                st.write(f"**D+{n} \u5206\u5c42\u8868\u73b0\uff1a**")
                st.dataframe(grp, use_container_width=True)

        # \u98ce\u9669\u6807\u7b7e\u5206\u6790
        if not triggered.empty and 'tag' in triggered.columns:
            st.divider()
            st.subheader("\U0001f3f7\ufe0f \u6309\u98ce\u9669\u6807\u7b7e\u5206\u6790(D+3\u6536\u76ca)")
            valid = triggered.dropna(subset=['R_D3'])
            if not valid.empty:
                grp = valid.groupby('tag')['R_D3'].agg(
                    avg='mean', win_rate=lambda x: (x > 0).mean() * 100, n_count='count'
                ).round(2)
                grp.columns = ['\u5747\u503c', '\u80dc\u7387%', '\u6837\u672c\u6570']
                st.dataframe(grp, use_container_width=True)

        # \u660e\u7ec6\u8868
        st.divider()
        st.subheader("\U0001f4cb \u56de\u6d4b\u660e\u7ec6")
        show_cols = ['trade_date','rank','name','ts_code','close','pct_1d',
                     'pct_20d','from_bottom','rsi','winner_rate','score','tag',
                     'triggered','buy_price','R_D1','R_D3','R_D5']
        show_cols = [c for c in show_cols if c in final.columns]
        st.dataframe(final[show_cols], use_container_width=True, hide_index=True)

        csv = final[show_cols].to_csv(index=False).encode('utf-8-sig')
        st.download_button("\U0001f4e5 \u4e0b\u8f7d\u56de\u6d4b\u7ed3\u679cCSV", csv,
                           f"\u56de\u6d4b\u7ed3\u679c_{end_str}.csv", "text/csv")
    else:
        st.warning("\u26a0\ufe0f \u56de\u6d4b\u672a\u4ea7\u751f\u7ed3\u679c,\u8bf7\u68c0\u67e5\u65e5\u671f\u8303\u56f4\u6216Token\u6743\u9650")
```
