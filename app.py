# -*- coding: utf-8 -*-
"""
周线 SKDJ(W22) 底部脱离定型版
------------------------------------------------
核心实战闭环设计:
1. 扫描时机锁定: 仅在每周最后一个交易日(周五)运行，确认 K 线定型。
2. 买入时机延后: 信号确立后，统一按次周第一天的【开盘价】执行买入。
3. 差异化开盘拦截: 
   - 科创/创业板高开 > 8% 放弃
   - 主板高开 > 5% 放弃
   - 遭遇核按钮低开 < -4% 放弃
4. 右侧脱离确认 (突破25线法):
   - K值本周 > 25 且 上周 <= 25
   - 近10周D值触及过 <= 20
   - 金叉多头 (K > D) 且 周线收阳、量能配合。
5. 铁律出局系统:
   - 认栽出局: 跌破真实买入价 -10% 无条件止损。
   - 保本/止盈: 浮盈 12% 设保本，浮盈 25% 后开启 15% 移动止盈。
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

CACHE_FILE_NAME = "skdj_final_cache.pkl" 
CHECKPOINT_FILE = "skdj_final_checkpoint.csv"

# ---------------------------
# 全局变量
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_INDUSTRY = {} 

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="SKDJ 底部突破定型版", layout="wide")
st.title("📈 周线 SKDJ 底部脱离右侧确认系统 (周末选股·周一执行)")
st.markdown("🔒 **回测引擎已加锁：严格过滤周中假信号，100% 模拟周末选股 + 周一开盘竞价买入的实战流程。**")

# ---------------------------
# 基础 API 与数据管理
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
        load_bar = st.progress(0, text="正在加载硬科技白名单...")
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

def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_INDUSTRY
    if not trade_days_list: return False
    
    with st.spinner("同步全市场行业数据..."):
        GLOBAL_STOCK_INDUSTRY = load_industry_mapping()

    if use_cache and os.path.exists(CACHE_FILE_NAME):
        st.success("⚡ 本地行情引擎已就绪。")
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
    
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=550)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=150)).strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty: return False
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    my_bar = st.progress(0, text="Tushare 数据极速下载中...")
    adj_factor_data_list, daily_data_list = [], []
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
   
    with st.spinner("处理前复权数据..."):
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

def get_qfq_data(ts_code, start_date, end_date):
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
    return final_df

# ---------------------------
# 核心引擎：上穿 25 线突破判断
# ---------------------------
@st.cache_data(ttl=3600*12) 
def compute_breakout_signal(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=550)).strftime("%Y%m%d")
    df_daily = get_qfq_data(ts_code, start_date, end_date)
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

    n, m = 22, 3
    if len(weekly_df) < n + 15: return res

    # 指标计算
    weekly_df['lowv'] = weekly_df['low'].rolling(window=n).min()
    weekly_df['highv'] = weekly_df['high'].rolling(window=n).max()
    diff = (weekly_df['highv'] - weekly_df['lowv']).replace(0, 0.001)

    raw_rsv = (weekly_df['close'] - weekly_df['lowv']) / diff * 100
    weekly_df['rsv'] = raw_rsv.ewm(span=m, adjust=False).mean()
    weekly_df['k'] = weekly_df['rsv'].ewm(span=m, adjust=False).mean()
    weekly_df['d'] = weekly_df['k'].rolling(window=m).mean()
    weekly_df['ma5_vol'] = weekly_df['vol'].shift(1).rolling(window=5).mean()

    curr_w = weekly_df.iloc[-1]
    prev_w = weekly_df.iloc[-2]
    
    if pd.isna(curr_w['k']) or pd.isna(prev_w['k']): return res

    # --- 右侧突破核心法则 ---
    # 1. 本周 K 突破 25，且上周仍在 25 及以下
    is_breakout_25 = (curr_w['k'] > 25.0) and (prev_w['k'] <= 25.0)
    
    # 2. 多头排列：确保 K > D
    is_bullish = curr_w['k'] > curr_w['d']
    
    # 3. 底部基因：过去 10 周内，D 值曾去过 20 极度超卖区
    recent_d_min = weekly_df['d'].tail(10).min()
    has_bottom_gene = recent_d_min <= 20.0
    
    # 4. 形态与量能：周线收实体阳线，成交量达到均量 85% 以上
    is_yang = curr_w['close'] > curr_w['open']
    is_vol_ok = True
    if pd.notna(curr_w['ma5_vol']) and curr_w['ma5_vol'] > 0:
        is_vol_ok = curr_w['vol'] >= curr_w['ma5_vol'] * 0.85

    is_buy_signal = is_breakout_25 and is_bullish and has_bottom_gene and is_yang and is_vol_ok

    res['is_buy_signal'] = is_buy_signal
    res['k'] = round(curr_w['k'], 2)
    res['d'] = round(curr_w['d'], 2)
    res['recent_d_min'] = round(recent_d_min, 2)
    res['signal_close'] = curr_w['close'] # 周五收盘价
    res['vol_ratio'] = round(curr_w['vol'] / curr_w['ma5_vol'], 2) if curr_w['ma5_vol'] > 0 else 1.0

    return res

# ---------------------------
# 严格出局跟踪系统 (次周一买入执行)
# ---------------------------
def track_future_performance(ts_code, selection_date, signal_close, hold_weeks=8):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_fetch = (d0 - timedelta(days=30)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=150)).strftime("%Y%m%d") 
    
    hist_full = get_qfq_data(ts_code, start_date=start_fetch, end_date=end_future)
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    results['Exit_Reason'] = "持仓中"
    results['Buy_Price'] = np.nan
    results['Gap_pct (%)'] = np.nan
    
    if hist_full.empty or len(hist_full) < 10: return results
    
    for col in ['open', 'high', 'low', 'close']:
        hist_full[col] = pd.to_numeric(hist_full[col], errors='coerce')
        
    # 获取选股日(周五)之后的未来数据，第一条记录即为次周第一天(通常为周一)
    hist_future = hist_full[hist_full.index > selection_date]
    if hist_future.empty: return results

    next_row = hist_future.iloc[0]
    buy_price = next_row['open']
    if pd.isna(buy_price) or buy_price <= 0: return results

    # 1. 剔除开盘一字涨停/跌停无法买入的情况
    if next_row['open'] == next_row['high'] == next_row['low']:
        results['Exit_Reason'] = "一字板无法买入(剔除)"
        results['Buy_Price'] = round(buy_price, 2)  
        return results

    # 2. 差异化开盘跳空拦截
    is_20cm = any(ts_code.startswith(prefix) for prefix in ['300', '301', '688', '689'])
    gap_pct = (buy_price - signal_close) / signal_close * 100
    results['Gap_pct (%)'] = round(gap_pct, 2)
    
    # 科创/创业板高开 > 8% 放弃
    if is_20cm and gap_pct > 8.0:
        results['Exit_Reason'] = f"双创高开过大(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results
        
    # 主板高开 > 5% 放弃
    elif not is_20cm and gap_pct > 5.0:
        results['Exit_Reason'] = f"主板高开过大(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results
        
    # 遭遇恶劣核按钮低开 < -4% 放弃
    if gap_pct < -4.0:
        results['Exit_Reason'] = f"恶劣低开(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results

    # --- 顺利在周一开盘买入 ---
    results['Buy_Price'] = round(buy_price, 2)

    exit_triggered = False
    tier = 0  
    peak_close = buy_price
    pending_exit_reason = None  
    hard_stop_limit = -0.10 # 统一的认栽出局线
    
    for i in range(len(hist_future)):
        if i >= hold_weeks * 5: break 
            
        row = hist_future.iloc[i]
        day_count = i + 1
        current_week = ((day_count - 1) // 5) + 1 
        
        curr_open, curr_close, curr_high, curr_low = row['open'], row['close'], row['high'], row['low']
        
        if pending_exit_reason is not None:
            final_return = (curr_open - buy_price) / buy_price * 100.0
            exit_triggered = True
            results['Exit_Reason'] = pending_exit_reason
            results[f'Return_W{current_week} (%)'] = final_return
            break
        
        peak_close = max(peak_close, curr_high)
        peak_profit_pct = (peak_close - buy_price) / buy_price
        
        # 1. 认栽出局：触及买入价的 -10% 硬止损
        if (curr_low - buy_price) / buy_price <= hard_stop_limit:
            final_return = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
            exit_triggered = True
            results['Exit_Reason'] = "认栽出局(破-10%)"
            results[f'Return_W{current_week} (%)'] = final_return
            break
        
        # 2. 阶段保本：浮盈超 +12%
        if tier == 0 and peak_profit_pct >= 0.12:
            tier = 1
                
        if tier == 1:
            if curr_close < buy_price * 1.00:
                pending_exit_reason = "保本离场"
            elif peak_profit_pct >= 0.25:
                tier = 2
                
        # 3. 移动止盈：最高点回撤 15%
        if tier == 2:
            giveback = (peak_close - curr_close) / peak_close
            if giveback >= 0.15:
                pending_exit_reason = "移动止盈(回撤15%)"
            
        if day_count % 5 == 0:
            results[f'Return_W{current_week} (%)'] = (curr_close - buy_price) / buy_price * 100.0
            
    if not exit_triggered and len(hist_future) >= hold_weeks * 5:
        last_price = hist_future.iloc[hold_weeks * 5 - 1]['close']
        results[f'Return_W{hold_weeks} (%)'] = (last_price - buy_price) / buy_price * 100.0
        results['Exit_Reason'] = "期满平仓"
        
    return results

# ---------------------------
# 周线定格全市场扫描流程
# ---------------------------
def run_weekly_scan(last_trade, TOP_BACKTEST, MIN_MV, MAX_MV, MIN_PRICE):
    global GLOBAL_STOCK_INDUSTRY

    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), "数据缺失"

    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name')
    if stock_basic.empty: stock_basic = safe_get('stock_basic', list_status='L')
        
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    
    daily_basic = safe_get('daily_basic', trade_date=last_trade)
    if not daily_basic.empty:
        df = df.merge(daily_basic[['ts_code','circ_mv']], on='ts_code', how='left')
    else: 
        return pd.DataFrame(), "市值数据缺失"
    
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    
    # 基础门槛
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')] 
    df = df[(df['close'] >= MIN_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    
    records = []
    for row in df.itertuples():
        if GLOBAL_STOCK_INDUSTRY and row.ts_code not in GLOBAL_STOCK_INDUSTRY: 
            continue
            
        ind = compute_breakout_signal(row.ts_code, last_trade)
        if not ind or not ind.get('is_buy_signal'): 
            continue
            
        score_k_break = (ind['k'] - 25.0) * 5.0
        score_vol = ind['vol_ratio'] * 10.0
        total_score = score_k_break + score_vol
            
        # 传递周五定格的收盘价 signal_close 给未来的周一作对比
        future_returns = track_future_performance(row.ts_code, last_trade, ind['signal_close'], hold_weeks=8)
        
        record_dict = {
            'ts_code': row.ts_code, 'name': row.name, 'Signal_Close': ind['signal_close'], 
            'SKDJ_K': ind['k'], 'SKDJ_D': ind['d'], 'D_Min(10W)': ind['recent_d_min'],
            'circ_mv': round(row.circ_mv_billion, 2),
            'Total_Score': round(total_score, 1)
        }
        record_dict.update(future_returns)
        records.append(record_dict)
            
    if not records: return pd.DataFrame(), "无标的"
    
    fdf = pd.DataFrame(records)
    final_df = fdf.sort_values('Total_Score', ascending=False).head(TOP_BACKTEST).copy()
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    return final_df, None

# ---------------------------
# UI 控制流
# ---------------------------
with st.sidebar:
    st.header("⚙️ 周末确认系统参数")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("追溯自然天数", value=180, step=30)
    TOP_BACKTEST = st.number_input("每周优选 TopK", value=3)
    
    st.markdown("---")
    if st.button("🗑️ 清除断点重新回测"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            st.success("进度已清理！")
            
    st.markdown("---")
    st.subheader("💰 护城河底座")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=100.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)

TS_TOKEN = st.text_input("🔑 Tushare Token", type="password")
if not TS_TOKEN: 
    st.info("👈 请在左侧填入 Token 激活程序")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button("🚀 启动周线 SKDJ 突破 25 定型回测"):
    processed_dates = set()
    results = []
    
    if os.path.exists(CHECKPOINT_FILE):
        try:
            existing_df = pd.read_csv(CHECKPOINT_FILE)
            existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
            processed_dates = set(existing_df['Trade_Date'].unique())
            results.append(existing_df)
        except:
            pass
        
    trade_days_list = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_list: st.stop()
    
    if not get_all_historical_data(trade_days_list, use_cache=True): st.stop()
            
    # 【核心修正】：智能提取每周最后一个交易日，跳过周一至周四
    td_df = pd.DataFrame({'cal_date': trade_days_list})
    td_df['dt'] = pd.to_datetime(td_df['cal_date'])
    td_df['year_week'] = td_df['dt'].dt.isocalendar().year.astype(str) + "_" + td_df['dt'].dt.isocalendar().week.astype(str).str.zfill(2)
    valid_scan_dates = set(td_df.groupby('year_week')['cal_date'].max().tolist())
    
    dates_to_run = [d for d in trade_days_list if d not in processed_dates and d in valid_scan_dates]
    dates_to_run.sort() 
    
    if not dates_to_run:
        st.success("🎉 数据全部拉取完毕！")
    else:
        bar = st.progress(0, text="严格过滤周中数据，执行周末扫描引擎...")
        for i, date in enumerate(dates_to_run):
            res, err = run_weekly_scan(
                date, int(TOP_BACKTEST), MIN_MV, MAX_MV, MIN_PRICE
            )
            
            if not res.empty:
                res['Trade_Date'] = date
                is_first = not os.path.exists(CHECKPOINT_FILE)
                res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                results.append(res)
            bar.progress((i+1)/len(dates_to_run), text=f"处理周线数据: {date}")
        bar.empty()
    
    if results:
        all_res = pd.concat(results)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
        st.header("📊 定型版战绩追踪")
        st.subheader("🗓️ 周度胜率分布 (已精准剔除周一至周四所有假信号)")
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
                        st.metric(f"W{w}", "空缺")
                        
        st.subheader("📋 详细真实回测轨迹")
        display_cols = [
            'Rank', 'Trade_Date', 'name', 'ts_code', 'SKDJ_K', 'SKDJ_D', 'D_Min(10W)', 'Signal_Close', 'Buy_Price', 'Gap_pct (%)',
            'Total_Score', 'circ_mv', 'Exit_Reason'
        ] + [f'Return_W{w} (%)' for w in range(1, 9)]
        final_cols = [c for c in display_cols if c in all_res.columns]
    
        display_df = all_res[final_cols].sort_values(['Trade_Date', 'Rank'], ascending=[False, True]).reset_index(drop=True)
        
        def color_exit(val):
            if isinstance(val, str):
                if '剔除' in val: return 'color: white; background-color: darkgray'
                elif '认栽' in val: return 'color: white; background-color: darkred'
                elif '保本' in val: return 'color: orange'
                elif '移动止盈' in val: return 'color: green'
                elif '期满' in val: return 'color: blue'
            return ''
        
        try:
            st.dataframe(display_df.style.map(color_exit, subset=['Exit_Reason']), use_container_width=True)
        except AttributeError:
            st.dataframe(display_df.style.applymap(color_exit, subset=['Exit_Reason']), use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 导出完整回测记录 (CSV)", csv, "skdj_final_v1_export.csv", "text/csv")
    else:
        st.warning("⚠️ 此区间内无标的触发真实周末脱离信号。")
