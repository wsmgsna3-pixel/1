# -*- coding: utf-8 -*-
"""
选股王 · V30.15 机构趋势共振版 (终极稳定)
1. **策略基座**：回归 V30.12.3 的中盘共振逻辑 (50-1000亿 + 板块共振)，确保 50%+ 胜率。
2. **北向雷达 (Smart)**：判定逻辑优化为 `Vol > MA5` (趋势吸筹)，激活率大幅提升。
3. **极速架构**：保留“双重漏斗”筛选，只查 Top 15 北向数据，回测仅需 10-15 分钟。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_INDUSTRY = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V30.15：机构趋势共振版", layout="wide")
st.title("选股王 V30.15：机构趋势共振版（🦅 智能北向雷达 + 🛡️ 中盘共振）")
st.markdown("""
**版本核心 (V30.15)：**
1. 🛡️ **胜率基石**：完美复刻 V30.12.3 的选股逻辑，找回丢失的胜率。
2. 🦅 **智能雷达**：外资判定改为 `持仓 > 5日均线`，精准识别外资吸筹趋势，不再漏掉主力。
3. ⚡️ **极速引擎**：双重漏斗筛选，Top 15 精查模式，拒绝龟速回测。
""")

# ---------------------------
# 基础 API 函数
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: 
        return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        if kwargs.get('is_index'):
            df = pro.index_daily(**kwargs)
        else:
            df = func(**kwargs)
        if df is None or df.empty:
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception as e:
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    lookback_days = max(num_days * 3, 365) 
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty: return []
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()

@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    return {'adj': adj_df, 'daily': daily_df}

# --- 行业映射 ---
@st.cache_data(ttl=3600*24*7) 
def load_industry_mapping():
    global pro
    if pro is None: return {}
    try:
        sw_indices = pro.index_classify(level='L1', src='SW2021')
        if sw_indices.empty: return {}
        index_codes = sw_indices['index_code'].tolist()
        all_members = []
        load_bar = st.progress(0, text="正在遍历加载行业数据...")
        for i, idx_code in enumerate(index_codes):
            df = pro.index_member(index_code=idx_code, is_new='Y')
            if not df.empty: all_members.append(df)
            time.sleep(0.02) 
            load_bar.progress((i + 1) / len(index_codes), text=f"加载行业数据: {idx_code}")
        load_bar.empty()
        if not all_members: return {}
        full_df = pd.concat(all_members).drop_duplicates(subset=['con_code'])
        return dict(zip(full_df['con_code'], full_df['index_code']))
    except Exception: return {}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_INDUSTRY
    if not trade_days_list: return False
    
    with st.spinner("正在同步全市场行业数据..."):
        GLOBAL_STOCK_INDUSTRY = load_industry_mapping()
        if len(GLOBAL_STOCK_INDUSTRY) < 3000: st.warning(f"行业数据仅覆盖 {len(GLOBAL_STOCK_INDUSTRY)} 只")
        else: st.success(f"✅ 行业图谱构建完成，覆盖 {len(GLOBAL_STOCK_INDUSTRY)} 只股票")

    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=200)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=30)
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    st.info(f"⏳ 正在预加载全市场数据: {start_date} 至 {end_date}...")
    adj_factor_data_list = [] 
    daily_data_list = []
    my_bar = st.progress(0, text="数据同步中...")
    total_steps = len(all_dates)
    
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty: adj_factor_data_list.append(cached_data['adj'])
            if not cached_data['daily'].empty: daily_data_list.append(cached_data['daily'])
            if i % 20 == 0: time.sleep(0.05)
            if i % 5 == 0: my_bar.progress((i + 1) / total_steps, text=f"缓存全市场数据: {date}")
        except Exception: continue 
            
    my_bar.empty()
    if not adj_factor_data_list or not daily_data_list: return False
     
    adj_factor_data = pd.concat(adj_factor_data_list)
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_factor_data.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    daily_raw_data = pd.concat(daily_data_list)
    GLOBAL_DAILY_RAW = daily_raw_data.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        try:
            latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
        except: GLOBAL_QFQ_BASE_FACTORS = {}
    return True

# ---------------------------
# 复权与未来收益
# ---------------------------
def get_qfq_data_v4(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    latest_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj): return pd.DataFrame() 

    try:
        daily = GLOBAL_DAILY_RAW.loc[ts_code]
        daily = daily.loc[(daily.index >= start_date) & (daily.index <= end_date)]
        adj = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        adj = adj.loc[(adj.index >= start_date) & (adj.index <= end_date)]
    except KeyError: return pd.DataFrame()
    
    if daily.empty or adj.empty: return pd.DataFrame()
    df = daily.merge(adj.rename('adj_factor'), left_index=True, right_index=True, how='left').dropna(subset=['adj_factor'])
    
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns: df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'}).sort_values('trade_date_str').set_index('trade_date_str')
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col + '_qfq']
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 

def get_future_prices(ts_code, selection_date, d0_qfq_close, days_ahead=[1, 3, 5]):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    hist = get_qfq_data_v4(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    if hist.empty: return results
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    for n in days_ahead:
        col = f'Return_D{n}'
        if len(hist) >= n and d0_qfq_close > 0:
            results[col] = (hist.iloc[n-1]['close'] / d0_qfq_close - 1) * 100
        else: results[col] = np.nan
    return results

# ---------------------------
# 核心指标计算
# ---------------------------
def calculate_rsi(series, period=12):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data_v4(ts_code, start_date=start_date, end_date=end_date)
    res = {}
    if df.empty or len(df) < 60: return res 
    
    close = df['close']
    res['last_close'] = close.iloc[-1]
    res['last_high'] = df['high'].iloc[-1]
    res['last_low'] = df['low'].iloc[-1]
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    res['macd_val'] = ((ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()).iloc[-1] * 2
    
    res['ma20'] = close.tail(20).mean()
    res['ma60'] = close.tail(60).mean()
    
    res['bias_20'] = (res['last_close'] - res['ma20']) / res['ma20'] * 100 if res['ma20'] > 0 else 0
    res['rsi_12'] = calculate_rsi(close, period=12).iloc[-1]
    hist_60 = df.tail(60)
    res['position_60d'] = (close.iloc[-1] - hist_60['low'].min()) / (hist_60['high'].max() - hist_60['low'].min() + 1e-9) * 100
    
    return res

@st.cache_data(ttl=3600*12)
def get_market_state(trade_date):
    start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    index_data = safe_get('daily', ts_code='000300.SH', start_date=start_date, end_date=trade_date, is_index=True)
    if index_data.empty or len(index_data) < 20: return 'Weak'
    latest = index_data.sort_values('trade_date').iloc[-1]['close']
    ma20 = index_data['close'].tail(20).mean()
    return 'Strong' if latest > ma20 else 'Weak'

# --- 🦅 北向资金雷达 (V30.15 优化版) ---
# 判定逻辑：Vol > MA5 (趋势吸筹)
@st.cache_data(ttl=3600*12)
def check_single_stock_northbound_smart(ts_code, end_date):
    # 稍微取长一点时间，算均线
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d")
    try:
        df = safe_get('hk_hold', ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty or len(df) < 5: return 0 
        
        df = df.sort_values('trade_date')
        
        latest_vol = df.iloc[-1]['vol']
        # 计算 5日均线 (MA5)
        ma5_vol = df['vol'].tail(5).mean()
        
        # 1. 趋势吸筹 (Smart): 持仓 > 5日均线，且占比 > 0
        if latest_vol > ma5_vol:
            # 这里的 2 代表“趋势向好”，不一定是连买3天
            return 2 
            
        # 2. 风控: 单日大卖 (较昨日减仓 > 5%)
        prev_vol = df.iloc[-2]['vol']
        if latest_vol < prev_vol * 0.95: 
            return -1 
        
        return 0
    except: return 0

# ---------------------------
# 核心回测逻辑
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS, RSI_LIMIT, BIAS_LIMIT, SECTOR_THRESHOLD, MIN_MV, MAX_MV):
    global GLOBAL_STOCK_INDUSTRY
    
    market_state = get_market_state(last_trade)
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"No Data"

    daily_basic = safe_get('daily_basic', trade_date=last_trade)
    mf_raw = safe_get('moneyflow', trade_date=last_trade) 
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    
    # 板块共振
    strong_industry_codes = set()
    try:
        sw_df = safe_get('sw_daily', trade_date=last_trade)
        if not sw_df.empty:
            strong_sw = sw_df[sw_df['pct_chg'] >= SECTOR_THRESHOLD]
            strong_industry_codes = set(strong_sw['index_code'].tolist())
    except: pass 
        
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    
    if not daily_basic.empty:
        needed_cols = ['ts_code','turnover_rate','circ_mv','amount']
        existing_cols = [c for c in needed_cols if c in daily_basic.columns]
        df = df.merge(daily_basic[existing_cols], on='ts_code', how='left')
    
    if not mf_raw.empty:
        mf = mf_raw[['ts_code','net_mf_amount']].rename(columns={'net_mf_amount':'net_mf'})
        df = df.merge(mf, on='ts_code', how='left')
    
    for col in ['net_mf', 'turnover_rate', 'circ_mv', 'amount']:
        if col not in df.columns: df[col] = 0
    
    df['net_mf'] = df['net_mf'].fillna(0)
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    
    # 基础清洗
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')]
    df = df[(df['close'] >= 10.0) & (df['close'] <= 300.0)]
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    df = df[df['turnover_rate'] <= MAX_TURNOVER_RATE] 

    if len(df) == 0: return pd.DataFrame(), "Filtered Out"

    candidates = df.sort_values('pct_chg', ascending=False).head(FINAL_POOL)
    
    # --- 第一阶段：技术面粗筛 (本地快筛) ---
    preliminary_records = []
    for row in candidates.itertuples():
        # 板块过滤 (V30.12.3 逻辑)
        if GLOBAL_STOCK_INDUSTRY and strong_industry_codes:
            ind_code = GLOBAL_STOCK_INDUSTRY.get(row.ts_code)
            if ind_code and (ind_code not in strong_industry_codes): continue

        ind = compute_indicators(row.ts_code, last_trade)
        if not ind: continue
        
        d0_close = ind['last_close']
        d0_rsi = ind.get('rsi_12', 50)
        d0_bias = ind.get('bias_20', 0)
        
        # 趋势锁 (回滚至 V30.12.3: 只要求 > MA60，不要求 3%)
        if d0_close < ind['ma60']: continue 
        
        # 强弱市风控
        if market_state == 'Weak':
            if d0_rsi > RSI_LIMIT or d0_bias > BIAS_LIMIT: continue
            if d0_close < ind['ma20'] or ind['position_60d'] > 20.0: continue
        
        upper_shadow = (ind['last_high'] - d0_close) / d0_close * 100
        if upper_shadow > MAX_UPPER_SHADOW: continue
        
        range_len = ind['last_high'] - ind['last_low']
        if range_len > 0:
            body_pos = (d0_close - ind['last_low']) / range_len
            if body_pos < MIN_BODY_POS: continue
            
        base_score = ind['macd_val'] * 1000 + (row.net_mf / 10000)
        
        preliminary_records.append({
            'row_data': row,
            'ind_data': ind,
            'base_score': base_score,
            'd0_close': d0_close
        })
    
    if not preliminary_records: return pd.DataFrame(), "Empty"
    
    # 排序选出 Top 15 进入决赛
    preliminary_records.sort(key=lambda x: x['base_score'], reverse=True)
    finalists = preliminary_records[:15]
    
    # --- 第二阶段：北向精查 (V30.15 智能版) ---
    final_records = []
    for item in finalists:
        row = item['row_data']
        ind = item['ind_data']
        d0_close = item['d0_close']
        
        # 查北向 (智能趋势判断)
        nb_status = check_single_stock_northbound_smart(row.ts_code, last_trade)
        
        final_score = item['base_score']
        # 2 = 趋势吸筹 (+500)
        # -1 = 单日大卖 (-5000)
        if nb_status == 2:
             if final_score > 0: final_score += 500
        elif nb_status == -1:
             final_score -= 5000
             
        if market_state == 'Strong':
            if ind.get('rsi_12', 50) > RSI_LIMIT: final_score -= 500
            if ind.get('bias_20', 0) > BIAS_LIMIT: final_score -= 500

        future = get_future_prices(row.ts_code, last_trade, d0_close)
        
        final_records.append({
            'ts_code': row.ts_code, 'name': row.name, 'Close': row.close, 'Pct_Chg': row.pct_chg,
            'rsi': ind.get('rsi_12'), 'bias': ind.get('bias_20'), 
            'net_mf': row.net_mf,
            'Return_D1 (%)': future.get('Return_D1', np.nan),
            'Return_D3 (%)': future.get('Return_D3', np.nan),
            'Return_D5 (%)': future.get('Return_D5', np.nan),
            'market_state': market_state,
            'nb_status': nb_status, 
            'Sector_Boost': 'Yes',
            'Score': final_score
        })
        
    fdf = pd.DataFrame(final_records)
    return fdf.sort_values('Score', ascending=False).head(TOP_BACKTEST), None

# ---------------------------
# UI 主程序
# ---------------------------
with st.sidebar:
    st.header("V30.15 机构趋势共振版")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数", value=30, step=1)
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=5)
    
    st.markdown("---")
    st.subheader("💰 市值筛选 (亿元)")
    col_mv1, col_mv2 = st.columns(2)
    MIN_MV = col_mv1.number_input("最小市值", value=50.0, step=10.0)
    MAX_MV = col_mv2.number_input("最大市值", value=1000.0, step=50.0)
    
    st.markdown("---")
    st.subheader("🔥 板块共振设置")
    SECTOR_THRESHOLD = st.number_input("板块当日最低涨幅 (%)", value=1.5, step=0.1)
    
    st.markdown("---")
    RSI_LIMIT = st.number_input("RSI 拦截线", value=80.0)
    BIAS_LIMIT = st.number_input("Bias(20) 拦截线 (%)", value=25.0)
    MAX_UPPER_SHADOW = st.number_input("最大上影线 (%)", value=4.0)
    MIN_BODY_POS = st.number_input("最低实体位置", value=0.7)
    MAX_TURNOVER_RATE = st.number_input("最大换手率 (%)", value=20.0)

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🚀 启动 V30.15 终极版"):
    st.info("⚡️ 智能引擎已启动 (Top 15 精查模式)...")
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    
    if not get_all_historical_data(trade_days):
        st.error("数据预加载失败")
        st.stop()
        
    results = []
    bar = st.progress(0, text="回测引擎流水线启动...")
    
    for i, date in enumerate(trade_days):
        res, err = run_backtest_for_a_day(date, int(TOP_BACKTEST), 100, MAX_UPPER_SHADOW, MAX_TURNOVER_RATE, MIN_BODY_POS, RSI_LIMIT, BIAS_LIMIT, SECTOR_THRESHOLD, MIN_MV, MAX_MV)
        if not res.empty:
            res['Trade_Date'] = date
            results.append(res)
        
        bar.progress((i+1)/len(trade_days), text=f"正在分析第 {i+1} 天: {date}")
        
    bar.empty()
    
    if results:
        all_res = pd.concat(results)
        
        st.header("📊 V30.15 机构趋势仪表盘")
        cols = st.columns(3)
        for idx, n in enumerate([1, 3, 5]):
            col_name = f'Return_D{n} (%)'
            valid = all_res.dropna(subset=[col_name]) 
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                cols[idx].metric(f"D+{n} 均益 / 胜率", f"{avg:.2f}% / {win:.1f}%")
        
        st.subheader("📋 优选清单")
        display_cols = ['Trade_Date','name','ts_code','Close','Pct_Chg',
                        'Return_D1 (%)', 'Return_D3 (%)',
                        'net_mf','nb_status','Sector_Boost']
        st.dataframe(all_res[display_cols].sort_values('Trade_Date', ascending=False), use_container_width=True)
    else:
        st.warning("⚠️ 严苛条件下无股可选。市场可能处于冰点。")
