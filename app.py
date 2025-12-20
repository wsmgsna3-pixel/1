# -*- coding: utf-8 -*-
"""
选股王 · V30.10 纯粹趋势版 (统一赛道 + 市值门槛)
V30.10 核心重构：
1. [入围重构] 取消“涨幅榜赛道”，全市场只选“资金流最强”且“涨幅适中”的前 100 名。
   - 逻辑：确保所有入围者都是“有主力真金白银买入”的，且大家在同一起跑线竞争。
2. [市值门槛] 默认只选流通市值 50亿 - 2000亿 的股票，剔除小盘妖股和巨无霸。
3. [评分逻辑] 维持 V30.7 的 MACD * 10000，优选趋势最强的龙头。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 


# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V30.10 纯粹趋势版", layout="wide")
st.title("选股王 · V30.10 纯粹趋势版（🦄 资金流+中阳线统一赛道）")
st.markdown("🎯 **V30.10 策略：** 只有 **市值达标** 且 **主力大买** 的 **中阳线** 股票才能入围，最后按 MACD 决胜负。")


# ---------------------------
# 辅助函数 
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        if kwargs.get('is_index'): df = pro.index_daily(**kwargs)
        else: df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历。")
        return []
    return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()


# ----------------------------------------------------------------------
# 数据拉取
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=20)).strftime("%Y%m%d")
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    st.info(f"⏳ 正在按日期循环下载 {start_date} 到 {end_date} 间的全市场数据...")

    adj_list, daily_list = [], []
    download_progress = st.progress(0, text="下载进度...")
    
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty: adj_list.append(cached_data['adj'])
            if not cached_data['daily'].empty: daily_list.append(cached_data['daily'])
            download_progress.progress((i + 1) / len(all_dates))
        except: continue 
    download_progress.empty()

    if not adj_list or not daily_list:
        st.error("无法获取历史数据。")
        return False
        
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    cols_to_keep = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
    valid_cols = [c for c in cols_to_keep if c in daily_list[0].columns]
    daily_raw = pd.concat(daily_list)[valid_cols]
    
    for col in ['open', 'high', 'low', 'close', 'vol']:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        try:
            latest_adj = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj.droplevel(1).to_dict()
        except: GLOBAL_QFQ_BASE_FACTORS = {}
    
    return True

# ----------------------------------------------------------------------
# 数据处理
# ----------------------------------------------------------------------
def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty: return pd.DataFrame()
        
    base_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(base_adj) or base_adj < 1e-9: return pd.DataFrame() 

    try:
        daily = GLOBAL_DAILY_RAW.loc[ts_code]
        daily = daily.loc[(daily.index >= start_date) & (daily.index <= end_date)]
        adj = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        adj = adj.loc[(adj.index >= start_date) & (adj.index <= end_date)]
    except KeyError: return pd.DataFrame()
    
    if daily.empty or adj.empty: return pd.DataFrame()
    
    df = daily.merge(adj.rename('adj_factor'), left_index=True, right_index=True, how='left').dropna(subset=['adj_factor'])
    df = df.sort_index()
    
    factor = df['adj_factor'] / base_adj
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns: df[col] = df[col] * factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df['trade_date'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')
    return df.sort_values('trade_date').set_index('trade_date_str')[['open', 'high', 'low', 'close', 'vol']]

# ----------------------------------------------------------------------
# 右侧收益
# ----------------------------------------------------------------------
def get_future_prices_right_side(ts_code, selection_date, days_ahead=[1, 3, 5], buy_threshold_pct=1.5):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    for n in days_ahead: results[f'Return_D{n}'] = np.nan

    if hist.empty: return results
        
    d1_data = hist.iloc[0]
    buy_price_threshold = d1_data['open'] * (1 + buy_threshold_pct / 100.0)
    
    if d1_data['high'] < buy_price_threshold: return results 

    for n in days_ahead:
        idx = n - 1
        if len(hist) > idx:
            results[f'Return_D{n}'] = (hist.iloc[idx]['close'] / buy_price_threshold - 1) * 100
            
    return results

# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    res = {}
    if df.empty or len(df) < 26: return res
         
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    close = df['close']
    res['last_close'] = close.iloc[-1] 
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    res['macd_val'] = ((ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()).iloc[-1] * 2
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    return res

@st.cache_data(ttl=3600*12)
def get_market_state(trade_date):
    start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    index_data = safe_get('daily', ts_code='000300.SH', start_date=start_date, end_date=trade_date, is_index=True)
    if index_data.empty or len(index_data) < 20: return 'Weak'
    index_data = index_data.sort_values('trade_date')
    return 'Strong' if index_data.iloc[-1]['close'] > index_data['close'].tail(20).mean() else 'Weak'
      
        
# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测设置")
    backtest_date_end = st.date_input("回测结束日期", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("**回测天数 (N)**", value=50, step=1))
    
    st.markdown("---")
    st.header("2. 纯粹选股门槛 (V30.10)")
    st.info("💡 **只选符合以下所有条件的股票**")
    
    col1, col2 = st.columns(2)
    with col1: MIN_PCT_CHG = st.number_input("最小涨幅 (%)", value=3.0)
    with col2: MAX_PCT_CHG = st.number_input("最大涨幅 (%)", value=8.0)
    
    col3, col4 = st.columns(2)
    with col3: MIN_MV_BILLION = st.number_input("最小市值(亿)", value=50.0)
    with col4: MAX_MV_BILLION = st.number_input("最大市值(亿)", value=2000.0)
    
    st.header("3. 实战参数")
    BUY_THRESHOLD_PCT = st.number_input("买入确认阈值 (%)", value=1.5, step=0.1)
    
    st.markdown("---")
    st.header("4. 基础过滤")
    FINAL_POOL = int(st.number_input("入围数量", value=100)) 
    TOP_BACKTEST = int(st.number_input("Top K", value=5))
    MIN_PRICE = st.number_input("最低股价", value=10.0, step=0.5) 
    MAX_PRICE = st.number_input("最高股价", value=300.0, step=5.0)
    MIN_TURNOVER = st.number_input("最低换手 (%)", value=3.0) 
    MIN_AMOUNT = st.number_input("最低成交额 (亿)", value=1.0) * 100000000 

# ---------------------------
# Token 
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ----------------------------------------------------------------------
# 核心逻辑
# ----------------------------------------------------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, buy_threshold):
    # 1. 弱市熔断
    market_state = get_market_state(last_trade)
    if market_state == 'Weak':
        return pd.DataFrame(), f"弱市避险：指数 < MA20，全天空仓。"

    # 2. 拉取数据
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失"

    pool = daily_all.reset_index(drop=True)
    basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date') 
    pool = pool.merge(basic, on='ts_code', how='left') if not basic.empty else pool
    
    d_basic = safe_get('daily_basic', trade_date=last_trade, fields='ts_code,turnover_rate,circ_mv,total_mv')
    pool = pool.merge(d_basic, on='ts_code', how='left') if not d_basic.empty else pool
    
    mf = safe_get('moneyflow', trade_date=last_trade)
    if not mf.empty and 'net_mf' in mf.columns:
        mf = mf[['ts_code', 'net_mf']].fillna(0)
        pool = pool.merge(mf, on='ts_code', how='left')
    
    for c in ['turnover_rate','circ_mv','net_mf']: 
        if c not in pool.columns: pool[c] = 0.0

    # 3. 严格过滤 (V30.10 核心：统一赛道 + 市值门槛)
    df = pool.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce') 
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000 
    
    df = df[~df['name'].str.contains('ST|退', case=False, na=False)]
    df = df[~df['ts_code'].str.startswith('92')]
    
    if 'list_date' in df.columns:
        df['days_listed'] = (datetime.strptime(last_trade, "%Y%m%d") - pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')).dt.days
        df = df[df['days_listed'] >= 120]

    df = df[
        (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE) & 
        (df['circ_mv_billion'] >= MIN_MV_BILLION) & (df['circ_mv_billion'] <= MAX_MV_BILLION) & # 市值门槛
        (df['turnover_rate'] >= MIN_TURNOVER) &
        (df['amount'] * 1000 >= MIN_AMOUNT)
    ]

    # [涨幅区间]：必须是中阳线
    df = df[(df['pct_chg'] >= MIN_PCT_CHG) & (df['pct_chg'] <= MAX_PCT_CHG)]
    
    if len(df) == 0: return pd.DataFrame(), f"过滤后无股票"

    # 4. 统一入围 (V30.10)
    # 不再分赛道，直接选池子里的“吸金王”
    candidates = df.sort_values('net_mf', ascending=False).head(FINAL_POOL).reset_index(drop=True)
    
    if not GLOBAL_DAILY_RAW.empty:
        try:
            available = GLOBAL_DAILY_RAW.loc[(slice(None), last_trade), :].index.get_level_values('ts_code').unique()
            candidates = candidates[candidates['ts_code'].isin(available)]
        except: return pd.DataFrame(), "缓存缺失"

    # 5. 深度计算
    records = []
    for row in candidates.itertuples():
        ind = compute_indicators(row.ts_code, last_trade) 
        if pd.isna(ind.get('macd_val')) or ind.get('macd_val') <= 0: continue
        
        future = get_future_prices_right_side(row.ts_code, last_trade, buy_threshold_pct=buy_threshold)
        
        records.append({
            'ts_code': row.ts_code, 'name': getattr(row, 'name', row.ts_code),
            'Close': row.close, 'Pct_Chg (%)': getattr(row, 'pct_chg', 0),
            'Circ_MV': row.circ_mv_billion,
            'macd': ind['macd_val'], 'volatility': ind['volatility'],
            'Return_D1 (%)': future.get('Return_D1'), 'Return_D3 (%)': future.get('Return_D3')
        })
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), "无正向MACD股票"

    # 6. 评分 (回归 MACD * 10000)
    s_vol = fdf['volatility']
    if s_vol.max() != s_vol.min():
        s_vol = (s_vol - s_vol.min()) / (s_vol.max() - s_vol.min())
    else: s_vol = 0.5
    
    fdf['综合评分'] = fdf['macd'] * 10000 + (1 - s_vol) * 0.3
    fdf['策略'] = '纯粹趋势(资金流+MACD)'
    
    fdf = fdf.sort_values('综合评分', ascending=False).head(TOP_BACKTEST)
    return fdf.reset_index(drop=True), None

# ---------------------------
# 主程序
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日纯粹回测"):
    
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days: st.stop()
    
    if not get_all_historical_data(trade_days): st.stop()
    st.success("✅ 数据就绪！开始 V30.10 纯粹版回测...")
    
    results = []
    bar = st.progress(0)
    
    for i, date in enumerate(trade_days):
        df, msg = run_backtest_for_a_day(date, TOP_BACKTEST, FINAL_POOL, BUY_THRESHOLD_PCT)
        if not df.empty:
            df['Trade_Date'] = date
            results.append(df)
        bar.progress((i + 1) / len(trade_days))
    bar.empty()
    
    if not results:
        st.error("区间内无有效强市交易日。")
        st.stop()
        
    all_res = pd.concat(results)
    if all_res['Trade_Date'].dtype != 'object': all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
    st.header(f"📊 V30.10 回测报告 (纯粹趋势 + 市值门槛)")
    st.markdown(f"**有效交易天数：** {all_res['Trade_Date'].nunique()} 天")

    cols = st.columns(2)
    for idx, n in enumerate([1, 3]):
        col = f'Return_D{n} (%)' 
        valid = all_res.dropna(subset=[col])
        if not valid.empty:
            avg_ret = valid[col].mean()
            hit_rate = (valid[col] > 0).sum() / len(valid) * 100
            count = len(valid)
        else: avg_ret, hit_rate, count = 0, 0, 0
        with cols[idx]:
            st.metric(f"D+{n} 收益 / 胜率", f"{avg_ret:.2f}% / {hit_rate:.1f}%", help=f"成交：{count} 笔")

    st.header("📋 每日成交明细")
    st.dataframe(all_res.sort_values('Trade_Date', ascending=False), use_container_width=True)
