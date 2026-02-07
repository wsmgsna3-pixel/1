# -*- coding: utf-8 -*-
"""
选股王 · V30.12.15 真实回测修复版 (Tushare/YFinance 双模)
------------------------------------------------
版本号：V30.12.15
修复内容：
1. 【代码修正】Tushare 美股代码去掉 .US 后缀，匹配接口标准。
2. 【双重保障】优先 Tushare，失败自动切 YFinance，确保数据必达。
3. 【数据监控】侧边栏显示美股数据获取状态，拒绝盲跑。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import concurrent.futures 
import os
import pickle

# 尝试导入 yfinance 作为备用
try:
    import yfinance as yf
except ImportError:
    yf = None

warnings.filterwarnings("ignore")

# ---------------------------
# 1. 全局配置
# ---------------------------
st.set_page_config(page_title="选股王 V30.12.15 修复版", layout="wide")

pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_INDUSTRY = {} 
GLOBAL_US_DATA = pd.DataFrame() 

# --- 美股 ETF 代码表 (去掉 .US 后缀以适配 Tushare) ---
US_ETF_MAP = {
    'XLK': {'name': '科技(XLK)', 'cn_inds': ['电子', '计算机', '通信']},
    'SOXX': {'name': '半导体(SOXX)', 'cn_inds': ['电子']}, 
    'XLC': {'name': '通信(XLC)', 'cn_inds': ['传媒', '通信']},
    'XLV': {'name': '医药(XLV)', 'cn_inds': ['医药生物', '美容护理']},
    'XLY': {'name': '可选(XLY)', 'cn_inds': ['汽车', '家用电器', '商贸零售', '纺织服饰']},
    'XLP': {'name': '必选(XLP)', 'cn_inds': ['食品饮料', '农林牧渔']},
    'XLE': {'name': '能源(XLE)', 'cn_inds': ['石油石化', '煤炭']},
    'XLF': {'name': '金融(XLF)', 'cn_inds': ['银行', '非银金融']},
    'XLI': {'name': '工业(XLI)', 'cn_inds': ['机械设备', '电力设备', '国防军工', '建筑装饰']},
    'XLB': {'name': '材料(XLB)', 'cn_inds': ['有色金属', '基础化工', '钢铁', '建筑材料']},
    'XLRE': {'name': '地产(XLRE)', 'cn_inds': ['房地产']},
    'XLU': {'name': '公用(XLU)', 'cn_inds': ['公用事业', '环保', '电力设备']}
}
US_TICKERS = list(US_ETF_MAP.keys())

# --- 申万一级行业列表 ---
SW_INDUSTRIES = {
    '801010.SI': '农林牧渔', '801030.SI': '基础化工', '801040.SI': '钢铁',
    '801050.SI': '有色金属', '801080.SI': '电子', '801710.SI': '建筑材料',
    '801720.SI': '建筑装饰', '801730.SI': '电力设备', '801740.SI': '国防军工',
    '801750.SI': '计算机', '801760.SI': '传媒', '801770.SI': '通信',
    '801880.SI': '汽车', '801890.SI': '机械设备', '801090.SI': '交运设备', 
    '801110.SI': '家用电器', '801120.SI': '食品饮料', '801130.SI': '纺织服饰',
    '801140.SI': '轻工制造', '801150.SI': '医药生物', '801160.SI': '公用事业',
    '801170.SI': '交通运输', '801180.SI': '房地产', '801200.SI': '商贸零售',
    '801210.SI': '社会服务', '801230.SI': '综合', '801780.SI': '银行',
    '801790.SI': '非银金融', '801950.SI': '煤炭', '801960.SI': '石油石化',
    '801970.SI': '环保', '801980.SI': '美容护理'
}

# ---------------------------
# 2. 基础函数
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        for _ in range(3):
            try:
                if kwargs.get('is_index'): 
                    kwargs.pop('is_index')
                    df = pro.index_daily(**kwargs)
                else: 
                    df = func(**kwargs)
                if df is not None and not df.empty: return df
                time.sleep(0.5)
            except:
                time.sleep(1)
                continue
        return pd.DataFrame(columns=['ts_code']) 
    except: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    lookback_days = max(num_days * 3, 365) 
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty: return []
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    return trade_days_df[trade_days_df['cal_date'] <= end_date_str]['cal_date'].head(num_days).tolist()

@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj = safe_get('adj_factor', trade_date=date)
    daily = safe_get('daily', trade_date=date)
    return {'adj': adj, 'daily': daily}

@st.cache_data(ttl=3600*24*7) 
def load_industry_mapping():
    global pro
    if pro is None: return {}
    try:
        all_members = []
        load_bar = st.progress(0, text="加载行业数据...")
        codes = list(SW_INDUSTRIES.keys())
        for i, idx_code in enumerate(codes):
            df = pro.index_member(index_code=idx_code, is_new='Y')
            if not df.empty:
                df['industry_name'] = SW_INDUSTRIES[idx_code]
                all_members.append(df[['con_code', 'industry_name']])
            time.sleep(0.02)
            load_bar.progress((i + 1) / len(codes))
        load_bar.empty()
        if not all_members: return {}
        full_df = pd.concat(all_members)
        full_df = full_df.drop_duplicates(subset=['con_code'])
        return dict(zip(full_df['con_code'], full_df['industry_name']))
    except: return {}

# ---------------------------
# 3. 美股数据处理 (双模下载)
# ---------------------------
@st.cache_data(ttl=3600*12)
def get_us_history_data(start_date, end_date):
    global pro
    
    # 放宽日期范围
    real_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d")
    real_end = (datetime.strptime(end_date, "%Y%m%d") + timedelta(days=5)).strftime("%Y%m%d")
    
    all_us_data = []
    failed_tickers = []
    
    # ---------------------------
    # 阶段一：优先尝试 Tushare
    # ---------------------------
    if pro is not None:
        bar = st.progress(0, text="尝试 Tushare 下载美股数据...")
        for i, ticker in enumerate(US_TICKERS):
            try:
                # 尝试纯代码 (e.g. XLK)
                df = safe_get('us_daily', ts_code=ticker, start_date=real_start, end_date=real_end)
                if df.empty:
                    # 失败则尝试加 .US 后缀 (e.g. XLK.US)
                    df = safe_get('us_daily', ts_code=ticker+'.US', start_date=real_start, end_date=real_end)
                
                if not df.empty:
                    df = df[['trade_date', 'ts_code', 'pct_change']]
                    # 统一代码格式，去掉可能存在的后缀，方便后续处理
                    df['ts_code'] = ticker 
                    all_us_data.append(df)
                else:
                    failed_tickers.append(ticker)
            except: 
                failed_tickers.append(ticker)
            bar.progress((i+1)/len(US_TICKERS))
        bar.empty()
    else:
        failed_tickers = US_TICKERS

    # ---------------------------
    # 阶段二：补救措施 (YFinance)
    # ---------------------------
    if failed_tickers and yf:
        st.warning(f"Tushare 部分数据获取失败 ({len(failed_tickers)}个)，正在调用 YFinance 补全...")
        try:
            # YFinance 需要 .US 后缀吗？不需要，但需要确保网络
            # 批量下载
            yf_data = yf.download(failed_tickers, start=datetime.strptime(real_start, "%Y%m%d"), end=datetime.strptime(real_end, "%Y%m%d"), progress=False)['Close']
            
            if not yf_data.empty:
                # 计算涨跌幅
                pct_df = yf_data.pct_change() * 100
                # 重构格式
                for ticker in failed_tickers:
                    if ticker in pct_df.columns:
                        single_df = pct_df[[ticker]].dropna().reset_index()
                        single_df.columns = ['trade_date', 'pct_change']
                        single_df['ts_code'] = ticker
                        single_df['trade_date'] = single_df['trade_date'].dt.strftime('%Y%m%d')
                        all_us_data.append(single_df)
        except Exception as e:
            st.error(f"YFinance 补全也失败了: {e}")

    if not all_us_data: return pd.DataFrame()
    
    full_df = pd.concat(all_us_data)
    # Pivot: Index=Date, Cols=Ticker
    pivot_df = full_df.pivot_table(index='trade_date', columns='ts_code', values='pct_change')
    return pivot_df.sort_index()

def get_hot_inds_for_date(trade_date_str, top_n=3):
    global GLOBAL_US_DATA
    if GLOBAL_US_DATA.empty: return []
    
    # 找小于当前日期的最近一个美股交易日
    valid_dates = GLOBAL_US_DATA.index[GLOBAL_US_DATA.index < trade_date_str]
    if len(valid_dates) == 0: return []
    
    target_us_date = valid_dates[-1]
    
    row = GLOBAL_US_DATA.loc[target_us_date].dropna()
    if row.empty: return []
    
    top_etfs = row.sort_values(ascending=False).head(top_n)
    
    target_inds = set()
    for etf_code, pct in top_etfs.items():
        # 这里不强制要求 >0，因为如果是普跌，也应该选跌得最少的（相对强势）
        # 或者你只想做上涨的？一般相对强度足够
        info = US_ETF_MAP.get(etf_code)
        if info:
            target_inds.update(info['cn_inds'])
                
    return list(target_inds)

# ---------------------------
# 4. 数据下载与缓存
# ---------------------------
CACHE_FILE_NAME = "market_data_cache_v15.pkl"

def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_INDUSTRY, GLOBAL_US_DATA
    
    GLOBAL_STOCK_INDUSTRY = load_industry_mapping()

    if use_cache and os.path.exists(CACHE_FILE_NAME):
        st.success("⚡ 极速加载本地缓存...")
        try:
            with open(CACHE_FILE_NAME, 'rb') as f:
                d = pickle.load(f)
                GLOBAL_ADJ_FACTOR = d['adj']
                GLOBAL_DAILY_RAW = d['daily']
                GLOBAL_US_DATA = d.get('us_data', pd.DataFrame())
            
            latest = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
            if latest:
                try: GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest), 'adj_factor'].droplevel(1).to_dict()
                except: pass
            
            if GLOBAL_US_DATA.empty:
                st.warning("缓存完美股数据缺失，正在补充...")
                start_d = min(trade_days_list)
                end_d = max(trade_days_list)
                GLOBAL_US_DATA = get_us_history_data(start_d, end_d)
                with open(CACHE_FILE_NAME, 'wb') as f: 
                    pickle.dump({'adj': GLOBAL_ADJ_FACTOR, 'daily': GLOBAL_DAILY_RAW, 'us_data': GLOBAL_US_DATA}, f)
            
            return True
        except: 
            os.remove(CACHE_FILE_NAME)

    latest = max(trade_days_list) 
    earliest = min(trade_days_list)
    s_dt = datetime.strptime(earliest, "%Y%m%d") - timedelta(days=200)
    e_dt = datetime.strptime(latest, "%Y%m%d") + timedelta(days=30)
    start_date = s_dt.strftime("%Y%m%d")
    end_date = e_dt.strftime("%Y%m%d")
    
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if cal.empty: return False
    all_dates = cal['cal_date'].tolist()
    
    st.info(f"📡 正在下载数据 (A股 + 美股)...")
    
    adj_list, daily_list = [], []
    def fetch_worker(date): return fetch_and_cache_daily_data(date)
    
    bar = st.progress(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f2d = {executor.submit(fetch_worker, d): d for d in all_dates}
        for i, f in enumerate(concurrent.futures.as_completed(f2d)):
            try:
                d = f.result()
                if not d['adj'].empty: adj_list.append(d['adj'])
                if not d['daily'].empty: daily_list.append(d['daily'])
            except: pass
            if i%10==0: bar.progress((i+1)/len(all_dates))
    bar.empty()
    
    # 下载美股
    GLOBAL_US_DATA = get_us_history_data(start_date, end_date)
    
    if not daily_list: return False
    with st.spinner("构建索引..."):
        GLOBAL_ADJ_FACTOR = pd.concat(adj_list).drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index()
        GLOBAL_ADJ_FACTOR['adj_factor'] = pd.to_numeric(GLOBAL_ADJ_FACTOR['adj_factor'], errors='coerce').fillna(0)
        GLOBAL_DAILY_RAW = pd.concat(daily_list).drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index()
        
        latest_d = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
        if latest_d:
            try: GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_d), 'adj_factor'].droplevel(1).to_dict()
            except: pass
            
        try:
            with open(CACHE_FILE_NAME, 'wb') as f: 
                pickle.dump({'adj': GLOBAL_ADJ_FACTOR, 'daily': GLOBAL_DAILY_RAW, 'us_data': GLOBAL_US_DATA}, f)
        except: pass
    return True

# ---------------------------
# 5. 核心计算
# ---------------------------
def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    latest_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj): return pd.DataFrame()
    try:
        d = GLOBAL_DAILY_RAW.loc[ts_code]
        d = d[(d.index >= start_date) & (d.index <= end_date)]
        a = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        a = a[(a.index >= start_date) & (a.index <= end_date)]
    except: return pd.DataFrame()
    if d.empty or a.empty: return pd.DataFrame()
    df = d.merge(a.rename('adj_factor'), left_index=True, right_index=True, how='left').dropna()
    for c in ['open','high','low','close']: df[c+'_qfq'] = df[c] * df['adj_factor'] / latest_adj
    df = df.reset_index().rename(columns={'trade_date':'trade_date_str'}).sort_values('trade_date_str').set_index('trade_date_str')
    for c in ['open','high','low','close']: df[c] = df[c+'_qfq']
    return df[['open','high','low','close','vol']]

def get_future_prices(ts_code, selection_date, d0_qfq_close, days_ahead=[1, 3, 5]):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    s = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    e = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    h = get_qfq_data_v4_optimized_final(ts_code, start_date=s, end_date=e)
    res = {}
    if h.empty: return res
    h['open'] = pd.to_numeric(h['open'], errors='coerce')
    h['high'] = pd.to_numeric(h['high'], errors='coerce')
    h['close'] = pd.to_numeric(h['close'], errors='coerce')
    
    d1 = h.iloc[0]
    next_open = d1['open']
    next_high = d1['high']
    
    if next_open <= d0_qfq_close: return res
    target_buy = next_open * 1.015
    if next_high < target_buy: return res
    
    for n in days_ahead:
        col = f'Return_D{n}'
        if len(h) >= n:
            res[col] = (h.iloc[n-1]['close'] - target_buy)/target_buy * 100
    return res

def compute_indicators(ts_code, end_date):
    s = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, s, end_date)
    if df.empty or len(df)<26: return {}
    df['pct'] = df['close'].pct_change().fillna(0)*100
    res = {'last_close': df['close'].iloc[-1], 'last_high': df['high'].iloc[-1]}
    
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9).mean()
    res['macd_val'] = ((diff-dea)*2).iloc[-1]
    
    ma20 = df['close'].tail(20).mean()
    res['ma20'] = ma20
    if ma20 > 0: res['bias_20'] = (res['last_close']-ma20)/ma20*100
    else: res['bias_20'] = 0
    
    delta = df['close'].diff()
    gain = (delta.where(delta>0, 0)).ewm(alpha=1/12).mean()
    loss = (-delta.where(delta<0, 0)).ewm(alpha=1/12).mean()
    rs = gain/(loss+1e-9)
    res['rsi_12'] = 100 - (100/(1+rs)).iloc[-1]
    return res

def get_market_state(trade_date):
    s = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    i = safe_get('daily', ts_code='000300.SH', start_date=s, end_date=trade_date, is_index=True)
    if i.empty or len(i)<20: return 'Weak'
    return 'Strong' if i.iloc[-1]['close'] > i['close'].tail(20).mean() else 'Weak'

# ---------------------------
# 6. 回测逻辑
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, RSI_LIMIT, CHIP_MIN_WIN_RATE, SECTOR_THRESHOLD, MIN_MV, MAX_MV, MAX_PREV_PCT, MIN_PRICE, USE_US_MAP, TOP_N_US):
    global GLOBAL_STOCK_INDUSTRY
    
    target_inds = []
    if USE_US_MAP:
        target_inds = get_hot_inds_for_date(last_trade, top_n=TOP_N_US)
    
    market_state = get_market_state(last_trade)
    daily = safe_get('daily', trade_date=last_trade)
    if daily.empty: return pd.DataFrame(), "无数据"
    
    base = safe_get('stock_basic', list_status='L')
    df = daily.merge(base[['ts_code','name']], on='ts_code')
    
    daily_basic = safe_get('daily_basic', trade_date=last_trade)
    if not daily_basic.empty:
        df = df.merge(daily_basic[['ts_code','turnover_rate','circ_mv']], on='ts_code', how='left')
    else: df['turnover_rate'] = 0; df['circ_mv'] = 0
        
    mf = safe_get('moneyflow', trade_date=last_trade)
    if not mf.empty: df = df.merge(mf[['ts_code','net_mf_amount']], on='ts_code', how='left').rename(columns={'net_mf_amount':'net_mf'})
    else: df['net_mf'] = 0
    df['net_mf'] = df['net_mf'].fillna(0)
    
    df = df[~df['name'].str.contains('ST|退')]
    df = df[(df['close']>=MIN_PRICE) & (df['close']<=2000) & (df['turnover_rate']<=MAX_TURNOVER_RATE)]
    df = df[(df['circ_mv']/10000 >= MIN_MV) & (df['circ_mv']/10000 <= MAX_MV)]
    
    if df.empty: return pd.DataFrame(), "过滤空"
    cands = df.sort_values('pct_chg', ascending=False).head(FINAL_POOL)
    recs = []
    
    for row in cands.itertuples():
        ind_name = GLOBAL_STOCK_INDUSTRY.get(row.ts_code)
        if USE_US_MAP:
            # 如果有美股数据，严格执行映射
            if target_inds and (ind_name not in target_inds): continue
        
        if row.pct_chg > MAX_PREV_PCT: continue
        
        ind = compute_indicators(row.ts_code, last_trade)
        if not ind: continue
        
        if ind.get('bias_20', 0) > 20: continue
        if ind['last_close'] < ind['ma20']: continue
        if market_state == 'Weak' and ind['rsi_12'] > RSI_LIMIT: continue
        
        upper = (ind['last_high'] - ind['last_close'])/ind['last_close']*100
        if upper > MAX_UPPER_SHADOW: continue
        
        fut = get_future_prices(row.ts_code, last_trade, ind['last_close'])
        
        recs.append({
            'ts_code': row.ts_code, 'name': row.name, 'rsi': ind['rsi_12'],
            'macd': ind['macd_val'], 'net_mf': row.net_mf, 'bias_20': ind.get('bias_20',0),
            'Return_D1 (%)': fut.get('Return_D1'), 'Return_D3 (%)': fut.get('Return_D3'),
            'Return_D5 (%)': fut.get('Return_D5'), 'market_state': market_state,
            'winner_rate': 80,
            'us_link': 'Yes' if (USE_US_MAP and target_inds) else 'No'
        })

    if not recs: return pd.DataFrame(), "无标的"
    fdf = pd.DataFrame(recs)
    
    def score(r):
        s = r['macd']*1000 + r['net_mf']/10000
        if 60<=r['rsi']<=85: s+=2000
        elif r['rsi']>90: s-=1000
        if r['bias_20']<10: s+=1000
        if r['market_state']=='Strong' and r['rsi']>RSI_LIMIT: s-=500
        return s
    
    fdf['Score'] = fdf.apply(score, axis=1)
    final = fdf.sort_values('Score', ascending=False).head(TOP_BACKTEST)
    final.insert(0, 'Rank', range(1, len(final)+1))
    return final, None

# ---------------------------
# 7. UI
# ---------------------------
with st.sidebar:
    st.header("V30.12.15 真实回测修复版")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数", value=30, step=1)
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=5)
    
    st.markdown("---")
    st.subheader("🇺🇸 美股联动监控")
    USE_US_MAP = st.checkbox("开启美股历史回溯", value=True)
    TOP_N_US = st.slider("选取美股 Top N", 1, 5, 3) 
    
    # 数据状态监控
    if not GLOBAL_US_DATA.empty:
        st.success(f"✅ 美股历史数据已就绪 ({len(GLOBAL_US_DATA)} 天)")
    else:
        st.info("⌛ 待下载美股数据...")

    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 清除行情缓存"):
        if os.path.exists(CACHE_FILE_NAME): os.remove(CACHE_FILE_NAME)
    CHECKPOINT_FILE = "backtest_checkpoint_v15.csv"
    
    # 基础参数
    MIN_PRICE = st.number_input("最低股价", value=10.0) 
    MIN_MV = st.number_input("最小市值(亿)", value=30.0) 
    MAX_MV = st.number_input("最大市值(亿)", value=1000.0)
    CHIP_MIN_WIN_RATE = st.number_input("最低获利盘 (%)", value=70.0)
    MAX_PREV_PCT = st.number_input("昨日最大涨幅限制 (%)", value=19.0)
    RSI_LIMIT = st.number_input("RSI 拦截线", value=100.0)
    SECTOR_THRESHOLD = st.number_input("板块涨幅 (%)", value=1.5)
    MAX_UPPER_SHADOW = st.number_input("上影线 (%)", value=5.0)
    MAX_TURNOVER_RATE = st.number_input("换手率 (%)", value=20.0)

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🚀 启动修复版回测"):
    processed_dates = set()
    results = []
    
    if RESUME_CHECKPOINT and os.path.exists(CHECKPOINT_FILE):
        try:
            existing_df = pd.read_csv(CHECKPOINT_FILE)
            existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
            processed_dates = set(existing_df['Trade_Date'].unique())
            results.append(existing_df)
            st.success(f"✅ 断点续传：跳过 {len(processed_dates)} 天")
        except: pass
    else:
        if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    
    trade_days_list = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_list: st.stop()
        
    dates_to_run = [d for d in trade_days_list if d not in processed_dates]
    
    if not dates_to_run:
        st.success("🎉 分析完毕")
    else:
        # 下载数据 (含美股双模下载)
        if not get_all_historical_data(trade_days_list, use_cache=True): st.stop()
            
        bar = st.progress(0, text="启动引擎...")
        for i, date in enumerate(dates_to_run):
            res, err = run_backtest_for_a_day(
                date, int(TOP_BACKTEST), 100, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, 
                RSI_LIMIT, CHIP_MIN_WIN_RATE, SECTOR_THRESHOLD, MIN_MV, MAX_MV, 
                MAX_PREV_PCT, MIN_PRICE, 
                USE_US_MAP, TOP_N_US
            )
            if not res.empty:
                res['Trade_Date'] = date
                is_first = not os.path.exists(CHECKPOINT_FILE)
                res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                results.append(res)
            bar.progress((i+1)/len(dates_to_run), text=f"分析中: {date}")
        bar.empty()
    
    if results:
        all_res = pd.concat(results)
        all_res = all_res[all_res['Rank'] <= int(TOP_BACKTEST)]
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        all_res = all_res.sort_values(['Trade_Date', 'Rank'], ascending=[False, True])
        
        st.header(f"📊 统计仪表盘 (修复版 - Top{TOP_N_US})")
        cols = st.columns(3)
        for idx, n in enumerate([1, 3, 5]):
            col_name = f'Return_D{n} (%)'
            valid = all_res.dropna(subset=[col_name]) 
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                cols[idx].metric(f"D+{n} 均益 / 胜率", f"{avg:.2f}% / {win:.1f}%")
        
        st.dataframe(all_res, use_container_width=True)
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载结果", csv, f"export_v15_fixed.csv", "text/csv")
    else:
        st.warning("⚠️ 没有结果")
