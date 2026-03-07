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

# 页面配置

# ==========================================

st.set_page_config(
page_title=“智选股 V1.0”,
page_icon=“📈”,
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

st.title(“📈 智选股 V1.0 · 鱼身策略”)
st.caption(“核心理念：找刚启动、还有空间的强势股 | 高开+冲高1.5%触发买入 | 止损5%”)

# ==========================================

# 全局变量

# ==========================================

pro = None
CACHE_FILE = “smart_screener_cache.pkl”
CHECKPOINT_FILE = “smart_screener_checkpoint.csv”

# ==========================================

# Tushare 基础函数

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
“”“加载全A股基础信息，排除ST和北交所”””
df = safe_api(‘stock_basic’, list_status=‘L’,
fields=‘ts_code,name,list_date,exchange’)
if df.empty:
return pd.DataFrame()
# 排除ST
df = df[~df[‘name’].str.contains(‘ST|退’, na=False)]
# 排除北交所（以43/83/87/92开头）
df = df[~df[‘ts_code’].str.match(r’^(43|83|87|92)’)]
# 排除北交所exchange
df = df[df[‘exchange’].isin([‘SSE’, ‘SZSE’])]
return df

@st.cache_data(ttl=3600 * 24 * 7)
def load_sw_industry():
“”“加载申万行业映射”””
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

# 数据缓存管理

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
st.warning(f”缓存保存失败: {e}”)

@st.cache_data(ttl=3600 * 12)
def fetch_daily_for_date(date):
daily = safe_api(‘daily’, trade_date=date)
adj   = safe_api(‘adj_factor’, trade_date=date)
basic = safe_api(‘daily_basic’, trade_date=date,
fields=‘ts_code,turnover_rate,circ_mv,pe_ttm,pb’)
chip  = safe_api(‘cyq_perf’, trade_date=date)
return {‘daily’: daily, ‘adj’: adj, ‘basic’: basic, ‘chip’: chip}

# ==========================================

# 技术指标计算

# ==========================================

def calc_indicators(close_series, vol_series, open_series, high_series, low_series):
“”“计算所需技术指标，返回dict”””
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

# MA20方向（用最近5日MA20斜率）
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

# 量能：近5日均量 vs 近20日均量
if n >= 20:
    avg_vol_5  = np.mean(vol[-5:])
    avg_vol_20 = np.mean(vol[-20:])
    res['vol_ratio'] = avg_vol_5 / (avg_vol_20 + 1e-9)
else:
    res['vol_ratio'] = 1.0

# 今日K线形态
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

# 涨幅计算
if n >= 2:
    res['pct_1d'] = (close[-1] - close[-2]) / (close[-2] + 1e-9) * 100
if n >= 6:
    res['pct_5d'] = (close[-1] - close[-6]) / (close[-6] + 1e-9) * 100
if n >= 21:
    res['pct_20d'] = (close[-1] - close[-21]) / (close[-21] + 1e-9) * 100

# 鱼身判断：60日内低点到现在的涨幅
if n >= 60:
    low_60 = np.min(close[-60:])
    res['from_bottom_pct'] = (close[-1] - low_60) / (low_60 + 1e-9) * 100
elif n >= 20:
    low_n = np.min(close[-n:])
    res['from_bottom_pct'] = (close[-1] - low_n) / (low_n + 1e-9) * 100
else:
    res['from_bottom_pct'] = 0

# 连续涨停检测（用最近2日涨幅判断）
if n >= 3:
    pct_today = (close[-1] - close[-2]) / (close[-2] + 1e-9) * 100
    pct_yest  = (close[-2] - close[-3]) / (close[-3] + 1e-9) * 100
    res['consecutive_limit'] = (pct_today >= 9.5 and pct_yest >= 9.5)
else:
    res['consecutive_limit'] = False

return res
```

# ==========================================

# 六维评分系统

# ==========================================

def calc_score(ind, winner_rate, sector_boost, market_strong):
“””
六维评分（满分100）：
- 技术面        25分
- 买入时机      20分
- 量能健康      15分
- 鱼身判断      15分
- 板块热度      15分
- 大盘环境      10分
“””
score = 0
detail = {}

```
# ===== 1. 技术面 (25分) =====
tech = 0
# MA趋势（股价在MA20上方且MA20向上）
if ind.get('last_close', 0) > ind.get('ma20', 0) and ind.get('ma20_slope', 0) > 0:
    tech += 10
elif ind.get('last_close', 0) > ind.get('ma20', 0):
    tech += 5
# MA20在MA60上方（中长期趋势）
if ind.get('ma20', 0) > ind.get('ma60', 0):
    tech += 5
# MACD：DIF在零轴上方且MACD柱为正
if ind.get('dif', 0) > 0 and ind.get('macd_bar', 0) > 0:
    tech += 7
elif ind.get('dif', 0) > 0 or ind.get('macd_bar', 0) > 0:
    tech += 3
# RSI健康区间55-70得满分，偏离扣分
rsi = ind.get('rsi', 50)
if 55 <= rsi <= 70:
    tech += 3
elif 50 <= rsi < 55 or 70 < rsi <= 75:
    tech += 1
detail['技术面'] = min(tech, 25)

# ===== 2. 买入时机 (20分) =====
timing = 0
# 距MA20偏离度：偏离越小越好
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
        timing += 0  # 跌破MA20不加分（已被硬性过滤）
# 近5日涨幅：温和上涨得分，过热降分
pct_5d = ind.get('pct_5d', 0)
if 3 <= pct_5d <= 10:
    timing += 8
elif 0 <= pct_5d < 3:
    timing += 5
elif 10 < pct_5d <= 15:
    timing += 3
elif pct_5d > 15:
    timing += 0
detail['买入时机'] = min(timing, 20)

# ===== 3. 量能健康 (15分) =====
vol = 0
vol_ratio = ind.get('vol_ratio', 1.0)
# 近5日均量是近20日均量的1.2-2.5倍：温和放大
if 1.2 <= vol_ratio <= 2.5:
    vol += 15
elif 1.0 <= vol_ratio < 1.2:
    vol += 8
elif 2.5 < vol_ratio <= 3.5:
    vol += 5  # 放量偏大，略降分
elif vol_ratio > 3.5:
    vol += 0  # 爆量，危险
elif vol_ratio < 1.0:
    vol += 3  # 缩量，动能不足
detail['量能健康'] = min(vol, 15)

# ===== 4. 鱼身判断 (15分) =====
fish = 0
from_bottom = ind.get('from_bottom_pct', 0)
if from_bottom <= 30:
    fish += 15   # 鱼身前段，空间充足
elif from_bottom <= 60:
    fish += 10   # 鱼身中段，仍有机会
elif from_bottom <= 100:
    fish += 4    # 鱼尾警告，降分
else:
    fish += 0    # 涨幅过大，危险
detail['鱼身判断'] = fish

# ===== 5. 板块热度 (15分) =====
sector = 0
if sector_boost > 3:
    sector += 15
elif sector_boost > 1.5:
    sector += 10
elif sector_boost > 0:
    sector += 5
detail['板块热度'] = sector

# ===== 6. 大盘环境 (10分) =====
mkt = 10 if market_strong else 3
detail['大盘环境'] = mkt

total = sum(detail.values())

# 风险标签
if from_bottom > 100 or ind.get('consecutive_limit', False):
    tag = '🔴 高风险'
elif from_bottom > 60 or pct_5d > 15 or rsi > 80:
    tag = '🟡 谨慎'
else:
    tag = '🟢 安全'

return total, detail, tag
```

# ==========================================

# 单只股票完整数据获取（用于回测和实盘）

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
# 前复权处理
daily = daily.merge(adj[[‘ts_code’, ‘trade_date’, ‘adj_factor’]], on=[‘ts_code’, ‘trade_date’], how=‘left’)
daily[‘adj_factor’] = daily[‘adj_factor’].fillna(method=‘ffill’).fillna(1.0)
latest_adj = daily[‘adj_factor’].iloc[0]  # 数据降序，第一条是最新
daily = daily.sort_values(‘trade_date’).reset_index(drop=True)
latest_adj = daily[‘adj_factor’].iloc[-1]
for col in [‘open’, ‘high’, ‘low’, ‘close’]:
daily[col] = daily[col] * daily[‘adj_factor’] / latest_adj
return daily

# ==========================================

# 板块热度计算

# ==========================================

@st.cache_data(ttl=3600 * 12)
def get_sector_performance(trade_date, industry_map):
“”“计算各申万行业5日相对大盘表现”””
start = (datetime.strptime(trade_date, “%Y%m%d”) - timedelta(days=15)).strftime(”%Y%m%d”)
# 大盘5日涨幅
hs300 = safe_api(‘daily’, ts_code=‘000300.SH’, start_date=start, end_date=trade_date)
if hs300.empty or len(hs300) < 5:
return {}
hs300 = hs300.sort_values(‘trade_date’)
mkt_5d = (hs300[‘close’].iloc[-1] - hs300[‘close’].iloc[-5]) / hs300[‘close’].iloc[-5] * 100

```
# 申万行业指数5日涨幅
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

# 大盘状态判断

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

# 核心选股函数（单日）

# ==========================================

def screen_one_day(trade_date, stock_basic_df, industry_map,
min_price, min_mv, max_mv,
top_n=5, for_backtest=False):
“””
执行单日选股，返回DataFrame
for_backtest=True时同时获取未来收益
“””
# 获取当日行情
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

# 板块热度
sector_perf = {}
try:
    sector_perf = get_sector_performance(trade_date, industry_map)
except:
    pass

# 大盘状态
market_strong = get_market_state(trade_date)

# 合并基础信息
df = daily_all.merge(stock_basic_df[['ts_code', 'name']], on='ts_code', how='inner')
if not daily_basic.empty:
    df = df.merge(daily_basic, on='ts_code', how='left')
else:
    df['circ_mv'] = np.nan
    df['turnover_rate'] = np.nan

# ===== 硬性过滤 =====
# 价格过滤
df = df[df['close'] >= min_price]
# 市值过滤（circ_mv单位：万元）
if 'circ_mv' in df.columns:
    df = df[df['circ_mv'].notna()]
    df = df[(df['circ_mv'] >= min_mv * 10000) & (df['circ_mv'] <= max_mv * 10000)]
# 只保留有名字的（已排除ST）
df = df[df['name'].notna() & (df['name'] != '')]

if df.empty:
    return pd.DataFrame()

# 按当日涨幅排序，取前200只做深度分析（提升效率）
df = df.sort_values('pct_chg', ascending=False).head(200)

records = []
for row in df.itertuples():
    ts_code = row.ts_code

    # winner_rate硬性过滤：50-85之间
    wr = chip_dict.get(ts_code, 70)
    if not (50 <= wr <= 85):
        continue

    # 获取历史数据
    hist = get_stock_history(ts_code, trade_date, lookback_days=90)
    if hist.empty or len(hist) < 30:
        continue

    # 计算技术指标
    ind = calc_indicators(
        hist['close'], hist['vol'],
        hist['open'], hist['high'], hist['low']
    )
    if not ind:
        continue

    # ===== 硬性技术过滤 =====
    # 股价必须在MA20上方
    if ind.get('last_close', 0) < ind.get('ma20', 1):
        continue
    # MA20必须向上
    if ind.get('ma20_slope', 0) <= 0:
        continue
    # 上影线≤5%
    if ind.get('upper_shadow', 99) > 5:
        continue
    # 实体位置≥60%
    if ind.get('body_pos', 0) < 0.6:
        continue
    # 排除连续涨停
    if ind.get('consecutive_limit', False):
        continue
    # 20日涨幅≤30%
    if ind.get('pct_20d', 0) > 30:
        continue
    # 5日涨幅≤20%
    if ind.get('pct_5d', 0) > 20:
        continue

    # 板块热度
    ind_code = industry_map.get(ts_code, '')
    s_boost = sector_perf.get(ind_code, 0) if ind_code else 0

    # 计算综合评分
    total_score, detail, tag = calc_score(ind, wr, s_boost, market_strong)

    # 建议买入价区间（昨收±0.5%到+2%）
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
        'market': '强势' if market_strong else '弱势',
        'sector_boost': round(s_boost, 2),
        '技术面': detail.get('技术面', 0),
        '买入时机': detail.get('买入时机', 0),
        '量能健康': detail.get('量能健康', 0),
        '鱼身判断': detail.get('鱼身判断', 0),
        '板块热度': detail.get('板块热度', 0),
        '大盘环境': detail.get('大盘环境', 0),
    }

    # 回测模式：模拟次日高开+冲高1.5%买入
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

# 回测模拟买入函数

# ==========================================

def simulate_buy(ts_code, selection_date, d0_close):
“”“模拟次日高开+盘中冲高1.5%触发买入，计算D1/D3/D5收益”””
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

# 触发条件：高开（开盘价>昨收）且盘中冲高超过开盘价1.5%
if next_open <= d0_close:
    return result
trigger_price = next_open * 1.015
if next_high < trigger_price:
    return result

# 触发成功
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

# ===== 侧边栏参数 =====

with st.sidebar:
st.header(“⚙️ 参数设置”)

```
st.subheader("🔑 Tushare Token")
ts_token = st.text_input("Token", type="password", key="token")

st.divider()
st.subheader("💰 股票池过滤")
min_price = st.number_input("最低股价(元)", value=10.0, min_value=1.0, step=1.0)
min_mv    = st.number_input("最小流通市值(亿)", value=50.0, min_value=10.0, step=10.0)
max_mv    = st.number_input("最大流通市值(亿)", value=1000.0, min_value=100.0, step=100.0)

st.divider()
st.subheader("📊 实盘选股")
top_n = st.slider("输出Top N候选股", 3, 10, 5)

st.divider()
st.subheader("🔬 回测设置")
bt_end   = st.date_input("回测截止日期", value=datetime.now().date())
bt_days  = st.number_input("回测天数", value=30, min_value=5, max_value=90, step=5)
bt_top_n = st.number_input("每日推荐数", value=4, min_value=1, max_value=10)
resume   = st.checkbox("开启断点续传", value=True)

if st.button("🗑️ 清除缓存"):
    for f in [CACHE_FILE, CHECKPOINT_FILE]:
        if os.path.exists(f):
            os.remove(f)
    st.cache_data.clear()
    st.success("缓存已清除")
```

# ===== Token初始化 =====

if not ts_token:
st.info(“👈 请在左侧输入 Tushare Token 后开始使用”)
st.stop()

ts.set_token(ts_token)
pro = ts.pro_api()

# ===== 主界面标签页 =====

tab1, tab2 = st.tabs([“📡 实盘选股”, “🔬 历史回测”])

# ==========================================

# TAB1：实盘选股

# ==========================================

with tab1:
st.subheader(“📡 实盘选股 · 今日候选”)
st.caption(“收盘后运行，第二天9:25集合竞价判断高开，9:30后冲高1.5%触发买入”)

```
col1, col2 = st.columns([3, 1])
with col1:
    screen_date = st.date_input("选股日期（选当天收盘日）",
                                 value=datetime.now().date(), key='screen_date')
with col2:
    st.write("")
    st.write("")
    run_screen = st.button("🚀 开始选股", use_container_width=True)

if run_screen:
    date_str = screen_date.strftime("%Y%m%d")

    with st.spinner("加载股票基础数据..."):
        stock_basic = load_stock_basic()
        industry_map = load_sw_industry()

    if stock_basic.empty:
        st.error("无法获取股票列表，请检查Token权限")
        st.stop()

    with st.spinner(f"正在分析 {date_str} 全市场数据，请稍候..."):
        result = screen_one_day(
            date_str, stock_basic, industry_map,
            min_price, min_mv, max_mv,
            top_n=top_n, for_backtest=False
        )

    if result.empty:
        st.warning("今日未找到符合条件的股票，可适当放宽参数")
    else:
        st.success(f"✅ 筛选完成，共推荐 {len(result)} 只候选股")

        market_state = get_market_state(date_str)
        st.info(f"📊 当前大盘状态：{'🟢 强势（沪深300站上MA20）' if market_state else '🔴 弱势（沪深300跌破MA20）'}")

        # 详细卡片展示
        for _, row in result.iterrows():
            with st.expander(
                f"#{int(row['rank'])}  {row['name']} ({row['ts_code']})  "
                f"¥{row['close']:.2f}  {row['tag']}  综合评分: {row['score']:.0f}分",
                expanded=(row['rank'] <= 2)
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("今日涨幅", f"{row['pct_1d']:+.2f}%")
                c2.metric("5日涨幅",  f"{row['pct_5d']:+.2f}%")
                c3.metric("20日涨幅", f"{row['pct_20d']:+.2f}%")
                c4.metric("本轮涨幅", f"{row['from_bottom']:+.1f}%",
                          help="从近60日低点到现在的涨幅，判断鱼身/鱼尾位置")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("RSI(14)", f"{row['rsi']:.1f}")
                c6.metric("筹码获利比", f"{row['winner_rate']:.1f}%")
                c7.metric("量比(5/20日)", f"{row['vol_ratio']:.2f}x")
                c8.metric("大盘环境", row['market'])

                st.divider()
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("📌 建议买入价区间",
                           f"¥{row['buy_low']:.2f} ~ ¥{row['buy_high']:.2f}")
                cc2.metric("🛡️ 止损价(-5%)", f"¥{row['stop_loss']:.2f}")
                cc3.metric("🎯 目标价(+8%)", f"¥{row['target']:.2f}")
                cc4.metric("风险收益比", "1 : 1.6")

                st.divider()
                st.write("**六维评分分解：**")
                score_cols = st.columns(6)
                dims = ['技术面','买入时机','量能健康','鱼身判断','板块热度','大盘环境']
                maxs = [25, 20, 15, 15, 15, 10]
                for i, (dim, mx) in enumerate(zip(dims, maxs)):
                    val = row.get(dim, 0)
                    pct = val / mx * 100
                    color = "🟢" if pct >= 70 else "🟡" if pct >= 40 else "🔴"
                    score_cols[i].metric(f"{color} {dim}", f"{val}/{mx}")

        # 汇总表格
        st.divider()
        st.write("**汇总表格：**")
        show_df = result[['rank','name','ts_code','close','pct_1d','pct_5d',
                           'pct_20d','from_bottom','rsi','winner_rate',
                           'score','tag','buy_low','stop_loss','target']].copy()
        show_df.columns = ['排名','名称','代码','现价','今日%','5日%','20日%',
                            '本轮涨幅%','RSI','筹码获利%','评分','风险','建议买入','止损','目标']
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        csv = show_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 导出CSV", csv,
                           f"智选股_{date_str}.csv", "text/csv")

    st.divider()
    st.caption("⚠️ 本工具仅供学习研究，不构成投资建议。股市有风险，投资需谨慎。")
```

# ==========================================

# TAB2：历史回测

# ==========================================

with tab2:
st.subheader(“🔬 历史回测 · 策略验证”)
st.caption(“模拟：次日高开 + 盘中冲高1.5% 触发买入，持有N天后收盘价卖出，止损5%”)

```
run_bt = st.button("🚀 启动回测", use_container_width=True)

if run_bt:
    end_str = bt_end.strftime("%Y%m%d")
    trade_days = get_recent_trade_days(end_str, int(bt_days))

    if not trade_days:
        st.error("无法获取交易日历，请检查Token")
        st.stop()

    # 断点续传
    processed = set()
    all_results = []

    if resume and os.path.exists(CHECKPOINT_FILE):
        try:
            ckpt = pd.read_csv(CHECKPOINT_FILE)
            ckpt['trade_date'] = ckpt['trade_date'].astype(str)
            processed = set(ckpt['trade_date'].unique())
            all_results.append(ckpt)
            st.success(f"✅ 读取断点存档，已跳过 {len(processed)} 个交易日")
        except:
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
    elif not resume and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    dates_to_run = [d for d in trade_days if d not in processed]

    if not dates_to_run:
        st.success("🎉 所有日期已计算完毕！")
    else:
        with st.spinner("加载股票基础数据..."):
            stock_basic = load_stock_basic()
            industry_map = load_sw_industry()

        bar = st.progress(0, text="回测中...")
        err_placeholder = st.empty()

        for i, date in enumerate(dates_to_run):
            bar.progress((i + 1) / len(dates_to_run),
                          text=f"回测中: {date}  ({i+1}/{len(dates_to_run)})")
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
                err_placeholder.warning(f"{date} 处理异常: {e}")

        bar.empty()

    # ===== 回测结果展示 =====
    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        final['trade_date'] = final['trade_date'].astype(str)
        final = final.sort_values(['trade_date', 'rank'], ascending=[False, True])

        st.divider()
        st.header("📊 回测统计报告")

        # 总体统计
        triggered = final[final['triggered'] == True] if 'triggered' in final.columns else pd.DataFrame()

        col1, col2, col3 = st.columns(3)
        col1.metric("总选股记录", f"{len(final)} 条")
        col1.metric("涉及交易日", f"{final['trade_date'].nunique()} 天")

        if not triggered.empty:
            col2.metric("触发买入比例",
                        f"{len(triggered)/len(final)*100:.1f}%",
                        help="满足高开+冲高1.5%的比例")
            for n, key in [(1,'R_D1'), (3,'R_D3'), (5,'R_D5')]:
                valid = triggered.dropna(subset=[key])
                if not valid.empty:
                    avg  = valid[key].mean()
                    wr   = (valid[key] > 0).mean() * 100
                    loss = (valid[key] < -5).mean() * 100
                    col3.metric(
                        f"D+{n} 均收益/胜率",
                        f"{avg:+.2f}% / {wr:.1f}%",
                        delta=f"亏损>5%: {loss:.1f}%"
                    )
        else:
            col2.info("无触发买入记录（可能日期范围内信号较少）")

        # Rank分层分析
        if not triggered.empty:
            st.divider()
            st.subheader("📈 按排名分层分析（触发买入）")
            for n, key in [(1,'R_D1'), (3,'R_D3'), (5,'R_D5')]:
                valid = triggered.dropna(subset=[key])
                if valid.empty:
                    continue
                grp = valid.groupby('rank')[key].agg(
                    均值='mean', 胜率=lambda x: (x > 0).mean() * 100, 样本='count'
                ).round(2)
                st.write(f"**D+{n} 分层表现：**")
                st.dataframe(grp, use_container_width=True)

        # 风险标签分析
        if not triggered.empty and 'tag' in triggered.columns:
            st.divider()
            st.subheader("🏷️ 按风险标签分析（D+3收益）")
            valid = triggered.dropna(subset=['R_D3'])
            if not valid.empty:
                grp = valid.groupby('tag')['R_D3'].agg(
                    均值='mean', 胜率=lambda x: (x > 0).mean() * 100, 样本='count'
                ).round(2)
                st.dataframe(grp, use_container_width=True)

        # 明细表
        st.divider()
        st.subheader("📋 回测明细")
        show_cols = ['trade_date','rank','name','ts_code','close','pct_1d',
                     'pct_20d','from_bottom','rsi','winner_rate','score','tag',
                     'triggered','buy_price','R_D1','R_D3','R_D5']
        show_cols = [c for c in show_cols if c in final.columns]
        st.dataframe(final[show_cols], use_container_width=True, hide_index=True)

        csv = final[show_cols].to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载回测结果CSV", csv,
                           f"回测结果_{end_str}.csv", "text/csv")
    else:
        st.warning("⚠️ 回测未产生结果，请检查日期范围或Token权限")
```
