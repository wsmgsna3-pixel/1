# -*- coding: utf-8 -*-
"""
周线波段选股王 · SKDJ(W22) 低位金叉实战版
------------------------------------------------
核心策略架构:
1. [股票池底座]:
   - 申万一级 6 大硬科技赛道白名单 (电子/计算机/通信/医药生物/国防军工/机械设备)
   - 流通市值锁定: 100亿 ~ 1000亿
   - 股价门槛: >= 10.0 元 (剔除低价股、ST、*ST、退市、北交所)
2. [周线 SKDJ(W22) 核心模型]:
   - 周期参数: N=22 (约半年周期极值), M=3 平滑
   - 买入信号: 周线 K 线上穿 D 线 (低位金叉)，且 D 值 <= 30 (筑底超卖区)
   - 形态过滤: 当周周 K 线必须收阳，且周成交量 >= 5周均量 * 0.85
3. [T+1 开盘竞价拦截]:
   - 开盘跳空在 [-3%, +5%] 之间正常买入，超出范围直接判定为极端情绪并剔除
4. [波段阶梯止盈止损]:
   - 固定硬止损: 主板 -8%, 双创(20cm) -12%
   - 保本止盈: 浮盈峰值达到 +12% 后，回落至成本线即保本离场
   - 移动止盈: 浮盈峰值达到 +25% 后，自高点回撤 15% 止盈
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import concurrent.futures 
import os
import pickle

warnings.filterwarnings("ignore")

CACHE_FILE_NAME = "weekly_skdj_market_data_cache.pkl" 
CHECKPOINT_FILE = "weekly_skdj_backtest_checkpoint.csv"

# ---------------------------
# 全局变量
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_INDUSTRY = {} 
SINA_STATUS = {'success': 0, 'fail': 0} 

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="周线 SKDJ(W22) 波段选股系统", layout="wide")
st.title("📈 周线 SKDJ(W22) 低位金叉波段选股系统")

# ---------------------------
# 新浪实时行情获取
# ---------------------------
def get_sina_realtime_kline(ts_code):
    global SINA_STATUS
    code_split = ts_code.split('.')
    if len(code_split) != 2: return None
    sina_code = code_split[1].lower() + code_split[0]
    
    url = f"http://hq.sinajs.cn/list={sina_code}"
    headers = {'Referer': 'https://finance.sina.com.cn'}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = 'gbk'
        data_str = response.text.split('="')[1].split('";')[0]
        if not data_str: 
            SINA_STATUS['fail'] += 1
            return None
        data_list = data_str.split(',')
        
        SINA_STATUS['success'] += 1
        return {
            'trade_date_str': datetime.now().strftime('%Y%m%d'),
            'open': float(data_list[1]),
            'pre_close': float(data_list[2]),
            'close': float(data_list[3]),
            'high': float(data_list[4]),
            'low': float(data_list[5]),
            'vol': (float(data_list[8]) / 100) * (240 / 225) 
        }
    except Exception:
        SINA_STATUS['fail'] += 1
        return None

# ---------------------------
# 基础 API 与行业映射
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        for _ in range(3):
            try:
                if func_name == 'index_daily': 
                    df = pro.index_daily(**kwargs)
                else: 
                    df = func(**kwargs)
                if df is not None and not df.empty: return df
                time.sleep(0.5)
            except: time.sleep(1); continue
        return pd.DataFrame(columns=['ts_code']) 
    except Exception: return pd.DataFrame(columns=['ts_code'])

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

@st.cache_data(ttl=3600*24*7) 
def load_industry_mapping():
    global pro
    if pro is None: return {}
    try:
        sw_indices = pro.index_classify(level='L1', src='SW2021')
        if sw_indices.empty: return {}
        white_list_names = ['电子', '计算机', '通信', '医药生物', '国防军工', '机械设备']
        target_indices = sw_indices[sw_indices['industry_name'].isin(white_list_names)]
        index_codes = target_indices['index_code'].tolist()
        
        all_members = []
        load_bar = st.progress(0, text="正在加载 6 大硬科技白名单数据...")
        for i, idx_code in enumerate(index_codes):
            df = pro.index_member(index_code=idx_code, is_new='Y')
            if not df.empty: 
                df['industry_code'] = idx_code
                all_members.append(df)
            time.sleep(0.05) 
            load_bar.progress((i + 1) / len(index_codes))
        load_bar.empty()
        
        if not all_members: return {}
        full_df = pd.concat(all_members).drop_duplicates(subset=['con_code'])
        return dict(zip(full_df['con_code'], full_df['industry_code']))
    except Exception:
        return {}

# ---------------------------
# 行情下载与前复权处理
# ---------------------------
def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_INDUSTRY
    if not trade_days_list: return False
    
    with st.spinner("正在同步全市场行业数据 (白名单)..."):
        GLOBAL_STOCK_INDUSTRY = load_industry_mapping()

    if use_cache and os.path.exists(CACHE_FILE_NAME):
        st.success("⚡ 发现本地行情缓存，极速加载中...")
        try:
            with open(CACHE_FILE_NAME, 'rb') as f:
                cached_data = pickle.load(f)
                GLOBAL_ADJ_FACTOR = cached_data['adj']
                GLOBAL_DAILY_RAW = cached_data['daily']
                
            latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
            if latest_global_date:
                try:
                    latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
                    GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
                except: GLOBAL_QFQ_BASE_FACTORS = {}
            return True
        except Exception:
            os.remove(CACHE_FILE_NAME)

    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 周线需要向前回溯至少 1.5 年以保证 22 周指标充分预热
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=550)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=150)).strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty: return False
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    st.info("📡 [首次运行] 正在下载全市场行情与复权数据...")
    adj_factor_data_list, daily_data_list = [], []

    my_bar = st.progress(0, text="Tushare 数据下载中...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_date = {executor.submit(fetch_and_cache_daily_data, date): date for date in all_dates}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_date)):
            try:
                data = future.result()
                if not data['adj'].empty: adj_factor_data_list.append(data['adj'])
                if not data['daily'].empty: daily_data_list.append(data['daily'])
            except Exception: pass
            if i % 5 == 0 or i == len(all_dates) - 1:
                my_bar.progress((i + 1) / len(all_dates), text=f"下载进度: {i+1}/{len(all_dates)}")
    my_bar.empty()
    
    if not daily_data_list: return False
   
    with st.spinner("正在构建多重索引并持久化缓存..."):
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
        
        try:
            with open(CACHE_FILE_NAME, 'wb') as f:
                pickle.dump({'adj': GLOBAL_ADJ_FACTOR, 'daily': GLOBAL_DAILY_RAW}, f)
        except Exception: pass
            
    return True

def get_qfq_data(ts_code, start_date, end_date, use_sina=False):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    
    latest_adj_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj_factor): return pd.DataFrame() 

    try:
        daily_df = GLOBAL_DAILY_RAW.loc[ts_code]
        daily_df = daily_df.loc[(daily_df.index >= start_date) & (daily_df.index <= end_date)].copy()
        adj_series = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        adj_series = adj_series.loc[(adj_series.index >= start_date) & (adj_series.index <= end_date)]
    except KeyError: return pd.DataFrame()
    
    if daily_df.empty or adj_series.empty: return pd.DataFrame()
    
    df = daily_df.merge(adj_series.rename('adj_factor'), left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df = df.sort_values('trade_date_str').set_index('trade_date_str')
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col + '_qfq']
        
    final_df = df[['open', 'high', 'low', 'close', 'pre_close', 'vol']].copy() 

    if use_sina:
        today_str = datetime.now().strftime('%Y%m%d')
        if end_date == today_str:
            sina_data = get_sina_realtime_kline(ts_code)
            if sina_data and sina_data['close'] > 0:
                sina_row = pd.DataFrame([sina_data]).set_index('trade_date_str')
                if today_str in final_df.index:
                    final_df.loc[today_str] = sina_row.iloc[0]
                else:
                    final_df = pd.concat([final_df, sina_row])
                    
    return final_df

# ---------------------------
# 周线 SKDJ (W22) 计算引擎
# ---------------------------
@st.cache_data(ttl=3600*12) 
def compute_weekly_skdj_signal(ts_code, end_date, n=22, m=3, use_sina=False, _run_id=None):
    """
    计算周线 SKDJ (N=22, M=3)
    LOWV:=LLV(LOW,N); HIGHV:=HHV(HIGH,N);
    RSV:=EMA((CLOSE-LOWV)/(HIGHV-LOWV)*100,M);
    K:EMA(RSV,M); D:MA(K,M);
    """
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=550)).strftime("%Y%m%d")
    df_daily = get_qfq_data(ts_code, start_date, end_date, use_sina=use_sina)
    res = {}
    if df_daily.empty or len(df_daily) < 100: return res

    # 日线合成为周线
    df = df_daily.copy().reset_index()
    df['dt'] = pd.to_datetime(df['trade_date_str'])
    iso_cal = df['dt'].dt.isocalendar()
    df['year_week'] = iso_cal.year.astype(str) + "_" + iso_cal.week.astype(str).str.zfill(2)

    weekly_df = df.groupby('year_week', as_index=False).agg({
        'trade_date_str': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum'
    }).sort_values('trade_date_str').reset_index(drop=True)

    if len(weekly_df) < n + 5: return res

    # 指标核心公式
    weekly_df['lowv'] = weekly_df['low'].rolling(window=n).min()
    weekly_df['highv'] = weekly_df['high'].rolling(window=n).max()

    diff = weekly_df['highv'] - weekly_df['lowv']
    diff = diff.replace(0, 0.001)

    raw_rsv = (weekly_df['close'] - weekly_df['lowv']) / diff * 100
    weekly_df['rsv'] = raw_rsv.ewm(span=m, adjust=False).mean()
    weekly_df['k'] = weekly_df['rsv'].ewm(span=m, adjust=False).mean()
    weekly_df['d'] = weekly_df['k'].rolling(window=m).mean()
    weekly_df['ma5_vol'] = weekly_df['vol'].shift(1).rolling(window=5).mean()

    # 获取当周与前一周数据
    curr_w = weekly_df.iloc[-1]
    prev_w = weekly_df.iloc[-2]

    if pd.isna(curr_w['k']) or pd.isna(curr_w['d']) or pd.isna(prev_w['k']) or pd.isna(prev_w['d']):
        return res

    # 1. 核心信号：当周 K 上穿 D (金叉)，且处于低位区 (D <= 30 或前一周 D <= 30)
    is_golden_cross = (curr_w['k'] > curr_w['d']) and (prev_w['k'] <= prev_w['d'])
    is_low_zone = (prev_w['d'] <= 30.0) or (curr_w['d'] <= 30.0)

    # 2. 周 K 线收阳 (确保周多头动能确立)
    is_yang = curr_w['close'] > curr_w['open']

    # 3. 周成交量配合 (不能过度缩量)
    is_vol_ok = True
    if pd.notna(curr_w['ma5_vol']) and curr_w['ma5_vol'] > 0:
        is_vol_ok = curr_w['vol'] >= curr_w['ma5_vol'] * 0.85

    is_buy_signal = is_golden_cross and is_low_zone and is_yang and is_vol_ok

    res['is_buy_signal'] = is_buy_signal
    res['k'] = round(curr_w['k'], 2)
    res['d'] = round(curr_w['d'], 2)
    res['signal_close'] = curr_w['close']
    res['pre_close'] = prev_w['close']
    res['bottom_line'] = curr_w['low']  # 当周最低价作为形态防守线
    res['vol_ratio'] = round(curr_w['vol'] / curr_w['ma5_vol'], 2) if curr_w['ma5_vol'] > 0 else 1.0

    return res

# ---------------------------
# 阶梯止盈止损回测跟踪系统
# ---------------------------
def track_medium_term_future(ts_code, selection_date, signal_close, bottom_line, hold_weeks=8, use_sina=False):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_fetch = (d0 - timedelta(days=60)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=150)).strftime("%Y%m%d") 
    
    hist_full = get_qfq_data(ts_code, start_date=start_fetch, end_date=end_future, use_sina=use_sina)
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    results['Exit_Reason'] = "持仓中"
    results['Buy_Price'] = np.nan
    results['Gap_pct (%)'] = np.nan
    
    if hist_full.empty or len(hist_full) < 30: return results
    
    hist_full['open'] = pd.to_numeric(hist_full['open'], errors='coerce')
    hist_full['high'] = pd.to_numeric(hist_full['high'], errors='coerce')
    hist_full['low'] = pd.to_numeric(hist_full['low'], errors='coerce')
    hist_full['close'] = pd.to_numeric(hist_full['close'], errors='coerce')
    
    hist_future = hist_full[hist_full.index > selection_date]
    if hist_future.empty: return results

    next_row = hist_future.iloc[0]

    # 一字涨停无法买入检测
    is_main_board = not (ts_code.startswith('300') or ts_code.startswith('301') 
                          or ts_code.startswith('688') or ts_code.startswith('689'))
    is_one_word_limit = (is_main_board and pd.notna(next_row['open']) and pd.notna(next_row['high']) 
                          and pd.notna(next_row['low']) and next_row['open'] == next_row['high'] == next_row['low'])
    if is_one_word_limit:
        results['Exit_Reason'] = "一字板无法买入(剔除)"
        results['Buy_Price'] = round(next_row['open'], 2)  
        return results

    buy_price = next_row['open']
    if pd.isna(buy_price) or buy_price <= 0:
        return results

    # T+1 开盘集合竞价拦截器 [-3%, +5%]
    if signal_close and signal_close > 0:
        gap_pct = (buy_price - signal_close) / signal_close * 100
        results['Gap_pct (%)'] = round(gap_pct, 2)
        if gap_pct < -3.0 or gap_pct > 5.0:
            results['Exit_Reason'] = f"开盘幅度不符(剔除: {round(gap_pct, 2)}%)"
            results['Buy_Price'] = round(buy_price, 2)
            return results

    results['Buy_Price'] = round(buy_price, 2)

    exit_triggered = False
    tier = 0  
    peak_close = buy_price
    pending_exit_reason = None  
    
    # 创业板/科创板 12% 止损，主板 8%
    is_20cm = ts_code.startswith('300') or ts_code.startswith('301') or ts_code.startswith('688')
    hard_stop_limit = -0.12 if is_20cm else -0.08
    
    for i in range(len(hist_future)):
        if i >= hold_weeks * 5: break 
            
        row = hist_future.iloc[i]
        day_count = i + 1
        current_week = ((day_count - 1) // 5) + 1 
        
        curr_open = row['open']
        curr_close = row['close']
        curr_high = row['high']
        curr_low = row['low']
        
        if pending_exit_reason is not None:
            final_return = (curr_open - buy_price) / buy_price * 100.0
            exit_triggered = True
            results['Exit_Reason'] = pending_exit_reason
            results[f'Return_W{current_week} (%)'] = final_return
            break
        
        peak_close = max(peak_close, curr_high)
        peak_profit_pct = (peak_close - buy_price) / buy_price
        
        # 1. 触及固定硬止损
        if (curr_low - buy_price) / buy_price <= hard_stop_limit:
            final_return = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
            exit_triggered = True
            results['Exit_Reason'] = f"固定止损(破{int(hard_stop_limit*100)}%)"
            results[f'Return_W{current_week} (%)'] = final_return
            break
        
        # 2. 达到 +12% 激活保本止盈
        if tier == 0 and peak_profit_pct >= 0.12:
            tier = 1
                
        if tier == 1:
            if curr_close < buy_price * 0.995:
                pending_exit_reason = "保本止盈"
            elif peak_profit_pct >= 0.25:
                tier = 2
                
        # 3. 达到 +25% 激活移动止盈 (回撤 15%)
        if tier == 2:
            giveback = (peak_close - curr_close) / peak_close
            if giveback >= 0.15:
                pending_exit_reason = "移动止盈(回撤15%)"
            
        if day_count % 5 == 0:
            results[f'Return_W{current_week} (%)'] = (curr_close - buy_price) / buy_price * 100.0
            
    if not exit_triggered and len(hist_future) >= hold_weeks * 5:
        last_price = hist_future.iloc[hold_weeks * 5 - 1]['close']
        results[f'Return_W{hold_weeks} (%)'] = (last_price - buy_price) / buy_price * 100.0
        results['Exit_Reason'] = "周期结束平仓"
        
    return results

# ---------------------------
# 每日/周选股核心执行循环
# ---------------------------
def run_scan_for_a_day(last_trade, TOP_BACKTEST, MIN_MV, MAX_MV, MIN_PRICE, use_sina=False, run_timestamp=None):
    global GLOBAL_STOCK_INDUSTRY

    query_date = last_trade
    daily_all = safe_get('daily', trade_date=query_date) 
    
    if use_sina and daily_all.empty:
        for i in range(1, 10):
            temp_date = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=i)).strftime("%Y%m%d")
            daily_all = safe_get('daily', trade_date=temp_date)
            if not daily_all.empty:
                query_date = temp_date
                break
                
    if daily_all.empty: return pd.DataFrame(), "数据缺失"

    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name')
    if stock_basic.empty: stock_basic = safe_get('stock_basic', list_status='L')
        
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    
    daily_basic = safe_get('daily_basic', trade_date=query_date)
    if not daily_basic.empty:
        df = df.merge(daily_basic[['ts_code','circ_mv']], on='ts_code', how='left')
    else: 
        return pd.DataFrame(), "市值数据缺失"
    
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    
    # 基础过滤：剔除 ST、退市、北交所 (92 开头)
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')] 
    
    # 股价与市值门槛 (100亿 ~ 1000亿，股价 >= 10元)
    df = df[(df['close'] >= MIN_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    
    records = []
    for row in df.itertuples():
        # 6 大硬科技白名单赛道过滤
        if GLOBAL_STOCK_INDUSTRY and row.ts_code not in GLOBAL_STOCK_INDUSTRY: 
            continue
            
        ind = compute_weekly_skdj_signal(row.ts_code, last_trade, n=22, m=3, use_sina=use_sina, _run_id=run_timestamp)
        if not ind or not ind.get('is_buy_signal'): 
            continue
            
        if use_sina and ind['signal_close'] < MIN_PRICE:
            continue
            
        # 打分机制：低位程度 (30-D) + 涨幅动能 + 量能评分
        score_low_zone = max(0, (30.0 - ind['d'])) * 2.0
        pct_chg = (ind['signal_close'] - ind['pre_close']) / ind['pre_close'] * 100
        score_gain = pct_chg * 5.0
        score_vol = ind['vol_ratio'] * 10.0
        total_score = score_low_zone + score_gain + score_vol
            
        future_returns = track_medium_term_future(row.ts_code, last_trade, ind['signal_close'], ind['bottom_line'], hold_weeks=8, use_sina=use_sina)
        
        record_dict = {
            'ts_code': row.ts_code, 'name': row.name, 'Signal_Close': ind['signal_close'], 
            'SKDJ_K': ind['k'], 'SKDJ_D': ind['d'],
            'circ_mv': round(row.circ_mv_billion, 2),
            'Total_Score': round(total_score, 1),
            'LowZone_S': round(score_low_zone, 1),
            'Volume_S': round(score_vol, 1)
        }
        record_dict.update(future_returns)
        records.append(record_dict)
            
    if not records: return pd.DataFrame(), "无标的"
    
    fdf = pd.DataFrame(records)
    final_df = fdf.sort_values('Total_Score', ascending=False).head(TOP_BACKTEST).copy()
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    return final_df, None

# ---------------------------
# UI 侧边栏与主控制流
# ---------------------------
with st.sidebar:
    st.header("⚙️ 参数配置")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数 (设为 1 即启动实盘周线雷达)", value=60, step=5)
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=3)
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 清除行情缓存"):
        if os.path.exists(CACHE_FILE_NAME):
            os.remove(CACHE_FILE_NAME)
            st.success("缓存已清除，下次运行将重新下载最新数据。")
    if st.button("🗑️ 清除断点记录 (重新回测)"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            st.success("旧进度已清理！")
            
    st.markdown("---")
    st.subheader("💰 股票池与护城河门槛")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=100.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: 
    st.warning("👈 请先在侧边栏输入您的 Tushare Token 启动程序。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button("🚀 启动周线 SKDJ(W22) 波段选股扫描"):
    SINA_STATUS = {'success': 0, 'fail': 0}
    processed_dates = set()
    results = []
    
    if RESUME_CHECKPOINT and os.path.exists(CHECKPOINT_FILE):
        try:
            existing_df = pd.read_csv(CHECKPOINT_FILE)
            existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
            processed_dates = set(existing_df['Trade_Date'].unique())
            results.append(existing_df)
        except:
            if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    else:
        if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
        
    trade_days_list = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_list: 
        st.error("未能获取交易日历，请检查日期或 Token。")
        st.stop()
    
    if not get_all_historical_data(trade_days_list, use_cache=True): 
        st.error("行情数据同步失败。")
        st.stop()
            
    dates_to_run = [d for d in trade_days_list if d not in processed_dates]
    if not dates_to_run:
        st.success("🎉 历史扫描已全部完毕！")
    else:
        bar = st.progress(0, text="正在运行周线 SKDJ(W22) 多周期扫描...")
        for i, date in enumerate(dates_to_run):
            is_realtime_radar = (int(BACKTEST_DAYS) == 1 and date == datetime.now().strftime("%Y%m%d"))
            run_timestamp = time.time() if is_realtime_radar else None
            
            res, err = run_scan_for_a_day(
                date, int(TOP_BACKTEST), MIN_MV, MAX_MV, MIN_PRICE,
                use_sina=is_realtime_radar, run_timestamp=run_timestamp
            )
            
            if not res.empty:
                res['Trade_Date'] = date
                is_first = not os.path.exists(CHECKPOINT_FILE)
                res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                results.append(res)
            bar.progress((i+1)/len(dates_to_run), text=f"扫描中: {date}")
        bar.empty()
    
    if int(BACKTEST_DAYS) == 1:
        st.markdown("---")
        if SINA_STATUS['success'] > 0:
            st.success(f"✅ **实时行情响应正常**：成功接入底层数据 {SINA_STATUS['success']} 次。")
        elif SINA_STATUS['fail'] > 0:
            st.error(f"❌ **实时行情抓取失败** {SINA_STATUS['fail']} 次。请确认当前网络与交易时间。")
        st.markdown("---")
    
    if results:
        all_res = pd.concat(results)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
        st.header("📊 周线 SKDJ(W22) 实战榜单与周期表现")
        st.subheader("🗓️ 周度生存与收益切片 (已剔除开盘异常标的)")
        cols_row1 = st.columns(4)
        cols_row2 = st.columns(4)
        
        valid_trades_only = all_res[~all_res['Exit_Reason'].str.contains('剔除', na=False)]
        
        for w in range(1, 9):
            col_name = f'Return_W{w} (%)'
            if col_name in valid_trades_only.columns:
                valid = valid_trades_only.dropna(subset=[col_name]) 
                target_col = cols_row1[w-1] if w <= 4 else cols_row2[w-5]
                with target_col:
                    if not valid.empty:
                        avg = valid[col_name].mean()
                        win = (valid[col_name] > 0).mean() * 100
                        st.metric(f"W{w} 均益/胜率 (存活{len(valid)}只)", f"{avg:.2f}% / {win:.1f}%")
                    else:
                        st.metric(f"W{w} 无持仓", "N/A")
                        
        st.subheader("📋 详细回测轨迹")
        display_cols = [
            'Rank', 'Trade_Date', 'name', 'ts_code', 'SKDJ_K', 'SKDJ_D', 'Signal_Close', 'Buy_Price', 'Gap_pct (%)',
            'Total_Score', 'LowZone_S', 'Volume_S', 'circ_mv', 'Exit_Reason'
        ] + [f'Return_W{w} (%)' for w in range(1, 9)]
        final_cols = [c for c in display_cols if c in all_res.columns]
    
        display_df = all_res[final_cols].sort_values(['Trade_Date', 'Rank'], ascending=[False, True]).reset_index(drop=True)
        
        def color_exit(val):
            if isinstance(val, str):
                if '剔除' in val: return 'color: white; background-color: darkgray'
                elif '固定止损' in val: return 'color: white; background-color: darkred'
                elif '保本止盈' in val: return 'color: orange'
                elif '移动止盈' in val: return 'color: green'
                elif '周期结束平仓' in val: return 'color: blue'
            return ''
        
        try:
            st.dataframe(display_df.style.map(color_exit, subset=['Exit_Reason']), use_container_width=True)
        except AttributeError:
            st.dataframe(display_df.style.applymap(color_exit, subset=['Exit_Reason']), use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 导出完整回测记录 (CSV)", csv, "weekly_skdj_w22_export.csv", "text/csv")
    else:
        st.warning("⚠️ 选股周期内暂无符合所有条件的标的。")
