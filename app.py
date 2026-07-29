# -*- coding: utf-8 -*-
"""
选股王 · V40.3 修复转折点重复触发版
------------------------------------------------
核心改进 (基于V40.2 + 深度讨论成果):
1. [MACD周线波浪识别] 引入周线 MACD 红绿柱交替（由红转绿）作为客观界定波浪洗盘的裁判。
2. [渐进式加分，非硬门槛] 洗盘次数不做筛选门槛，只按可靠程度分级加分（详见V40.1改动）。
3. [买入时机与周线转折点对齐] 用波浪转折点替代原来偏日线噪音的"回调2天"判断（详见V40.2改动）。
4. [★V40.3修复] V40.2的转折判断(w_curr['macd']>=0 and w_prev['macd']<0)会在"本周"这几天里
   天天成立——"本周"是用截至当天的数据滚动算出来的，只要本周还没走完、macd保持红柱，从周一
   到周五每天单独检查都符合条件，导致同一次转折被重复触发好几天(实测中江丰电子、北京君正、
   飞凯材料都在相邻日期各触发了2~4次，稀释了整体质量，也让原本表现最好的那一次触发被淹没，
   光迅科技这次只抓到转折后第9个交易日的次优信号，没能复现之前扛住172%涨幅的表现)。
   现在改成：用日线macd自己"今天由负转正"这个更精确的边界(只会在唯一的那一天成立)，配合
   "上一个完整周确实是绿柱"这个周线背景确认，两者结合，既贴合周线波浪转折的大方向，又不会
   把同一次转折重复计数。
5. [T+1开盘买入 & 一字板过滤 & 三层止盈止损] 完美继承 V39.6 的所有风控与实盘防错机制。
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

CACHE_FILE_NAME = "market_data_cache_v40_3.pkl" 

# ---------------------------
# 全局变量与探针
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_INDUSTRY = {} 
SINA_STATUS = {'success': 0, 'fail': 0} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V40.3 修复重复触发版", layout="wide")
st.title("选股王 V40.3：修复波浪转折点重复触发漏洞")

# ---------------------------
# 新浪实时行情引擎
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
# 基础 API 函数
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
    except Exception as e: return pd.DataFrame(columns=['ts_code'])

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
        load_bar = st.progress(0, text="正在加载硬科技白名单赛道数据...")
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
# 数据获取与复权引擎
# ---------------------------
def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_INDUSTRY
    if not trade_days_list: return False
    
    with st.spinner("正在同步全市场行业数据 (白名单)..."):
        GLOBAL_STOCK_INDUSTRY = load_industry_mapping()

    if use_cache and os.path.exists(CACHE_FILE_NAME):
        st.success(f"⚡ 发现本地行情缓存，极速加载中...")
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
    
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=150)).strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty: return False
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    st.info(f"📡 [首次运行] 正在下载复权行情数据...")
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
                my_bar.progress((i + 1) / len(all_dates), text=f"下载中: {i+1}/{len(all_dates)}")
    my_bar.empty()
    
    if not daily_data_list: return False
   
    with st.spinner("正在构建索引并保存缓存..."):
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

def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date, use_sina=False):
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
# 【V40新增】周线 MACD 波浪洗盘次数统计函数
# ---------------------------
def count_macd_wave_pullbacks(df_calc):
    """
    通过周线重采样并计算 MACD，统计从绝对低点到当前信号日之间，
    经历了多少次周线 MACD 由红转绿（洗盘结束/调整期）。
    返回洗盘次数 (int)
    """
    if len(df_calc) < 60: return -1 # 数据不够
    
    # 计算日线 MACD 基础指标
    df = df_calc.copy()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    df['dt'] = pd.to_datetime(df['trade_date_str'])
    iso_cal = df['dt'].dt.isocalendar()
    df['year_week'] = iso_cal.year.astype(str) + "_" + iso_cal.week.astype(str).str.zfill(2)
    
    weekly_df = df.groupby('year_week', as_index=False).agg({
        'trade_date_str': 'last',
        'low': 'min',
        'high': 'max',
        'macd': 'last'
    }).sort_values('trade_date_str').reset_index(drop=True)
    
    if len(weekly_df) < 10: return -1
    
    # 寻找过去周期内的最低点作为波浪起点
    min_idx = weekly_df['low'].idxmin()
    sub_df = weekly_df.loc[min_idx:].reset_index(drop=True)
    if len(sub_df) < 5: return -1
    
    running_max = sub_df['high'].iloc[0]
    in_pullback = False
    pullback_count = 0
    
    for i in range(1, len(sub_df)):
        curr_high = sub_df.loc[i, 'high']
        curr_low = sub_df.loc[i, 'low']
        curr_macd = sub_df.loc[i, 'macd']
        
        if curr_high > running_max:
            running_max = curr_high
            if in_pullback:
                in_pullback = False
        else:
            drawdown = (running_max - curr_low) / running_max
            # 满足 MACD 翻绿(<0) 且空间跌幅超过 5% 即确认为一次有效洗盘
            if curr_macd < 0 and drawdown >= 0.05:
                if not in_pullback:
                    in_pullback = True
                    pullback_count += 1
                    
    return pullback_count

# ---------------------------
# 核心指标计算 (温和周线过滤 + V40波浪动能机制)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def compute_trend_indicators(ts_code, end_date, use_sina=False, _run_id=None):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date, end_date, use_sina=use_sina)
    res = {}
    if df.empty or len(df) < 120: return res 
    
    # 1. 日线指标计算
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['ma120'] = df['close'].rolling(120).mean()
    df['ma5_vol'] = df['vol'].shift(1).rolling(5).mean()  
    
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    df_calc = df.dropna().copy().reset_index()
    if len(df_calc) < 20: return res

    # 【V40.1修正】：不再用洗盘次数做硬性筛选门槛——翻倍股回溯研究是"事后已知结局"算出来的，
    # 用它排除当下还不知道结局的候选票，等于假设"这只票一定会走完全程"，这是我们讨论中明确
    # 排除掉的做法（也是V39.3"近期新高确认"那次硬性条件误杀光迅科技的同类错误）。
    # 现在改成：始终计算 wave_count 并保留下来，不因为它不在某个区间就直接剔除信号，
    # 交给下面打分环节做"加分项"处理。
    wave_count = count_macd_wave_pullbacks(df_calc)

    # 2. 周线重采样合成（用于风控 + V40.2新增的波浪转折点判断）
    df_calc['dt'] = pd.to_datetime(df_calc['trade_date_str'])
    iso_cal = df_calc['dt'].dt.isocalendar()
    df_calc['year_week'] = iso_cal.year.astype(str) + "_" + iso_cal.week.astype(str).str.zfill(2)
    
    weekly_df = df_calc.groupby('year_week', as_index=False).agg({
        'trade_date_str': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'macd': 'last'  # 【V40.2新增】取当周最后一个交易日的日线macd值，作为该周(含未走完的本周)的MACD状态，
                         # 口径和 count_macd_wave_pullbacks、翻倍股统计工具保持一致，不引入第三种算法
    }).sort_values('trade_date_str').reset_index(drop=True)
    
    if len(weekly_df) < 10: return res
    
    weekly_df['w_ma20'] = weekly_df['close'].rolling(20).mean()
    
    w_curr = weekly_df.iloc[-1]
    w_prev = weekly_df.iloc[-2] if len(weekly_df) >= 2 else w_curr
    
    # 周线风控 1：防高位过度偏离
    w_bias_safe = True
    if not pd.isna(w_curr['w_ma20']) and w_curr['w_ma20'] > 0:
        w_bias = (w_curr['close'] - w_curr['w_ma20']) / w_curr['w_ma20']
        if w_bias > 0.45:
            w_bias_safe = False
            
    # 周线风控 2：防高位长上影线
    w_shadow_safe = True
    w_prev_range = w_prev['high'] - w_prev['low']
    w_prev_upper_shadow = w_prev['high'] - max(w_prev['open'], w_prev['close'])
    if w_prev_range > 0 and (w_prev_upper_shadow / w_prev_range) >= 0.60:
        w_shadow_safe = False

    is_weekly_safe = w_bias_safe and w_shadow_safe

    # 3. 日线突破点火信号
    row = df_calc.iloc[-1]
    prev_row = df_calc.iloc[-2]
    prev2_row = df_calc.iloc[-3]

    # 【V40.3修复】V40.2原来的写法(w_curr['macd']>=0 and w_prev['macd']<0)会在"本周"这几天
    # 里天天成立——因为"本周"是用截至当天的数据滚动算出来的，只要本周还没走完、macd保持红柱，
    # 从周一到周五每天单独检查都符合条件，导致同一次转折被重复触发好几天(江丰电子、北京君正、
    # 飞凯材料这几只票都在相邻日期各触发了2~4次，就是这个bug)。
    # 现在改成：用日线macd自己"今天由负转正"这个更精确的边界(只会在唯一的那一天成立)，
    # 配合"上一个完整周确实是绿柱"这个周线背景确认，两者结合，既贴合周线波浪转折的大方向，
    # 又不会把同一次转折重复计数。
    is_weekly_wave_turn = bool(row['macd'] >= 0 and prev_row['macd'] < 0 and w_prev['macd'] < 0)

    is_daily_trend_up = row['ma60'] > row['ma120']
    is_daily_breakout = row['close'] > row['ma20'] * 1.02
    is_daily_ma20_healthy = row['ma20'] >= prev_row['ma20']
    is_daily_vol_strong = row['vol'] > (1.2 * row['ma5_vol'])
    
    candle_range = row['high'] - row['low']
    candle_body = row['close'] - row['open']
    is_solid_yang = (row['close'] > row['open']) and (candle_body >= candle_range * 0.6 if candle_range > 0 else True)
    is_macd_healthy = (row['dif'] > 0) and (row['macd'] > prev_row['macd'])
    
    res['is_v38_buy_signal'] = (is_weekly_safe and 
                                is_daily_trend_up and is_weekly_wave_turn and 
                                is_daily_breakout and is_daily_ma20_healthy and 
                                is_daily_vol_strong and is_solid_yang and is_macd_healthy)
    
    if res['is_v38_buy_signal']:
        res['vol_ratio'] = row['vol'] / row['ma5_vol']  
        res['pre_close'] = prev_row['close']            
        
    res['wave_count'] = wave_count  # 【V40.1修正】始终记录，不管信号是否成立，且不再用默认值3掩盖"数据不足"的情况
    res['last_close'] = row['close']
    res['bottom_line'] = row['low'] 
    res['ma20'] = row['ma20']
    
    return res

# ---------------------------
# 三层简化止盈止损系统 (T+1开盘买入)
# ---------------------------
def get_medium_term_future(ts_code, selection_date, signal_close, bottom_line, hold_weeks=8, use_sina=False):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_fetch = (d0 - timedelta(days=60)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=150)).strftime("%Y%m%d") 
    
    hist_full = get_qfq_data_v4_optimized_final(ts_code, start_date=start_fetch, end_date=end_future, use_sina=use_sina)
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    results['Exit_Reason'] = "持仓中"
    results['Buy_Price'] = np.nan
    results['Gap_pct (%)'] = np.nan
    
    if hist_full.empty or len(hist_full) < 30: return results
    
    hist_full['open'] = pd.to_numeric(hist_full['open'], errors='coerce')
    hist_full['high'] = pd.to_numeric(hist_full['high'], errors='coerce')
    hist_full['low'] = pd.to_numeric(hist_full['low'], errors='coerce')
    hist_full['close'] = pd.to_numeric(hist_full['close'], errors='coerce')
    
    hist_full['ma10'] = hist_full['close'].rolling(10).mean()
    hist_full['ma20'] = hist_full['close'].rolling(20).mean()
    
    hist_future = hist_full[hist_full.index > selection_date]

    if hist_future.empty:
        return results

    next_row = hist_future.iloc[0]

    # 主板一字板过滤
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

    results['Buy_Price'] = round(buy_price, 2)
    if signal_close and signal_close > 0:
        results['Gap_pct (%)'] = round((buy_price - signal_close) / signal_close * 100, 2)

    exit_triggered = False
    tier = 0  
    peak_close = buy_price
    pending_exit_reason = None  
    
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
        
        final_return = np.nan
        
        if pending_exit_reason is not None:
            final_return = (curr_open - buy_price) / buy_price * 100.0
            exit_triggered = True
            results['Exit_Reason'] = pending_exit_reason
            results[f'Return_W{current_week} (%)'] = final_return
            break
        
        peak_close = max(peak_close, curr_high)
        peak_profit_pct = (peak_close - buy_price) / buy_price
        
        if (curr_low - buy_price) / buy_price <= hard_stop_limit:
            final_return = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
            exit_triggered = True
            results['Exit_Reason'] = f"固定止损(破{int(hard_stop_limit*100)}%)"
            results[f'Return_W{current_week} (%)'] = final_return
            break
        
        if tier == 0 and peak_profit_pct >= 0.10:
            tier = 1
                
        if tier == 1:
            if curr_close < buy_price * 0.995:
                pending_exit_reason = "保本止盈"
            elif peak_profit_pct >= 0.20:
                tier = 2
                
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
# 核心回测循环
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, MIN_MV, MAX_MV, MIN_PRICE, use_sina=False, run_timestamp=None):
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
    
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')] 
    
    df = df[(df['close'] >= MIN_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    
    records = []
    for row in df.itertuples():
        if GLOBAL_STOCK_INDUSTRY and row.ts_code not in GLOBAL_STOCK_INDUSTRY: 
            continue
            
        ind = compute_trend_indicators(row.ts_code, last_trade, use_sina=use_sina, _run_id=run_timestamp)
        if not ind or not ind.get('is_v38_buy_signal'): 
            continue
            
        if use_sina and ind['last_close'] < MIN_PRICE:
            continue
            
        pct_chg = (ind['last_close'] - ind['pre_close']) / ind['pre_close'] * 100
        score_breakout = pct_chg * 10 
        score_vol = ind['vol_ratio'] * 10
        total_score = score_breakout + score_vol
        
        # 【V40.1修正】洗盘次数改为渐进式加分曲线，只加分不做硬门槛：
        # 洗盘0~1次(尚未形成有效浪型，对应"潜伏爆破型")不加分；
        # 洗盘2~5次(对应第3~6浪)是翻倍股回溯统计里样本量和平均涨幅都相对扎实的区间，给予加分，
        # 其中洗盘4次(第5浪)在统计里平均涨幅最高(84%，179个样本)，加分也最高；
        # 洗盘6次以上样本量断崖下降(第7浪47个、第8浪18个、第9/10浪仅4个和1个)，规律不可信，不加分但也不排除。
        wave_cnt = ind.get('wave_count', -1)
        wave_bonus_map = {2: 10.0, 3: 20.0, 4: 20.0, 5: 10.0}
        total_score += wave_bonus_map.get(wave_cnt, 0.0)
            
        future_returns = get_medium_term_future(row.ts_code, last_trade, ind['last_close'], ind['bottom_line'], hold_weeks=8, use_sina=use_sina)
        
        record_dict = {
            'ts_code': row.ts_code, 'name': row.name, 'Signal_Close': ind['last_close'], 
            'Wave_Count': wave_cnt,
            'circ_mv': row.circ_mv_billion,
            'Total_Score': round(total_score, 1),
            'Breakout_S': round(score_breakout, 1),
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
# UI 及 主程序
# ---------------------------
with st.sidebar:
    st.header("V40.3 修复重复触发版")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数 (设为 1 即启动实盘雷达)", value=100, step=1)
    
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=3)
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 清除行情缓存"):
        if os.path.exists(CACHE_FILE_NAME):
            os.remove(CACHE_FILE_NAME)
            st.success("缓存已清除，下次运行将重新下载最新数据。")
    CHECKPOINT_FILE = "backtest_checkpoint_v40_3_fixdup.csv" 
    if st.button("🗑️ 清除断点记录 (重新回测)"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            st.success("旧进度已清理！")
            
    st.markdown("---")
    st.subheader("💰 核心护城河门槛")
    MIN_PRICE = st.number_input("最低股价 (元)", value=20.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=200.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🚀 启动 V40.3 追踪(修复重复触发)"):
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
    if not trade_days_list: st.stop()
    
    if not get_all_historical_data(trade_days_list, use_cache=True): st.stop()
            
    dates_to_run = [d for d in trade_days_list if d not in processed_dates]
    if not dates_to_run:
        st.success("🎉 扫描已全部完毕！")
    else:
        bar = st.progress(0, text="MACD波浪动能与爆量点火计算中...")
        for i, date in enumerate(dates_to_run):
            
            is_realtime_radar = (int(BACKTEST_DAYS) == 1 and date == datetime.now().strftime("%Y%m%d"))
            run_timestamp = time.time() if is_realtime_radar else None
            
            res, err = run_backtest_for_a_day(
                date, int(TOP_BACKTEST), MIN_MV, MAX_MV, MIN_PRICE,
                use_sina=is_realtime_radar, run_timestamp=run_timestamp
            )
            
            if not res.empty:
                res['Trade_Date'] = date
                is_first = not os.path.exists(CHECKPOINT_FILE)
                res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                results.append(res)
            bar.progress((i+1)/len(dates_to_run), text=f"分析中: {date}")
        bar.empty()
    
    if int(BACKTEST_DAYS) == 1:
        st.markdown("---")
        if SINA_STATUS['success'] > 0:
            st.success(f"✅ **盘中实时探针响应正常**：成功接入新浪底层数据 {SINA_STATUS['success']} 次，行情已接管。")
        elif SINA_STATUS['fail'] > 0:
            st.error(f"❌ **盘中实时探针警告**：新浪数据抓取失败 {SINA_STATUS['fail']} 次。请确认当前是否在交易时间。")
        else:
            st.info("ℹ️ 实时探针未触发（可能由于基础选股条件未通过）。")
        st.markdown("---")
    
    if results:
        all_res = pd.concat(results)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
        st.header(f"📊 V40.3 修复重复触发版")
        st.subheader("🗓️ 周度生存与收益切片")
        cols_row1 = st.columns(4)
        cols_row2 = st.columns(4)
        
        for w in range(1, 9):
            col_name = f'Return_W{w} (%)'
            if col_name in all_res.columns:
                valid = all_res.dropna(subset=[col_name]) 
                target_col = cols_row1[w-1] if w <= 4 else cols_row2[w-5]
                with target_col:
                    if not valid.empty:
                        avg = valid[col_name].mean()
                        win = (valid[col_name] > 0).mean() * 100
                        st.metric(f"W{w} 均益/胜率 (存活{len(valid)}只)", f"{avg:.2f}% / {win:.1f}%")
                    else:
                        st.metric(f"W{w} 无持仓", "N/A")
                        
        st.subheader("📋 优等生清单 (含波浪洗盘次数)")
        display_cols = [
            'Rank', 'Trade_Date', 'name', 'ts_code', 'Wave_Count', 'Signal_Close', 'Buy_Price', 'Gap_pct (%)',
            'Total_Score', 'Breakout_S', 'Volume_S', 'circ_mv', 'Exit_Reason'
        ] + [f'Return_W{w} (%)' for w in range(1, 9)]
        final_cols = [c for c in display_cols if c in all_res.columns]
    
        display_df = all_res[final_cols].sort_values(['Trade_Date', 'Rank'], ascending=[False, True]).reset_index(drop=True)
        
        def color_exit(val):
            if isinstance(val, str):
                if '固定止损' in val: return 'color: white; background-color: darkred'
                elif '一字板' in val: return 'color: white; background-color: gray'
                elif '保本止盈' in val: return 'color: orange'
                elif '移动止盈' in val: return 'color: green'
                elif '周期结束平仓' in val: return 'color: blue'
            return ''
        
        if 'Exit_Reason' in display_df.columns:
            try:
                st.dataframe(display_df.style.map(color_exit, subset=['Exit_Reason']), use_container_width=True)
            except AttributeError:
                st.dataframe(display_df.style.applymap(color_exit, subset=['Exit_Reason']), use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载完整轨迹 (CSV)", csv, f"export_v40_3_fixdup.csv", "text/csv")
    else:
        st.warning("⚠️ 暂无符合条件的标的。")
