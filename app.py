# -*- coding: utf-8 -*-
"""
选股王 · V37.0 中线共振狙击版 (轨迹追踪系统) - Pandas 最新兼容版
------------------------------------------------
逻辑说明:
1. [中线股票池] 流通市值 200亿-1000亿，股价 >= 20元。
2. [白名单赛道] 仅保留：电子、计算机、通信、电力设备、医药生物、汽车、国防军工。
3. [周线设伏] 周线 MACD 多头，股价回踩至 10周或20周均线附近 (上下 5% 空间)。
4. [日线发车] 右侧形态：收盘站上 5日线 + 突破昨日最高价(阳吞阴) + 温和放量。
5. [铁血出局] 硬止损 15%；盘中跌破 20日线无条件按均价止盈/止损出局。
6. [持仓轨迹] 引入时间胶囊机制，切片记录 W1 - W8 周度收益，出局后隐藏后续数据。
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

warnings.filterwarnings("ignore")

CACHE_FILE_NAME = "market_data_cache_v37.pkl" 

# ---------------------------
# 全局变量
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_INDUSTRY = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V37.0 中线共振", layout="wide")
st.title("选股王 V37.0：中线趋势狙击与周度轨迹追踪")

# ---------------------------
# 基础 API 与 辅助函数
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        for _ in range(3):
            try:
                if kwargs.get('is_index'): df = pro.index_daily(**kwargs)
                else: df = func(**kwargs)
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
        # 🌟 白名单赛道：只保留核心科技、医药与高端制造
        white_list_names = ['电子', '计算机', '通信', '电力设备', '医药生物', '汽车', '国防军工']
        target_indices = sw_indices[sw_indices['industry_name'].isin(white_list_names)]
        index_codes = target_indices['index_code'].tolist()
        
        all_members = []
        load_bar = st.progress(0, text="正在加载白名单赛道数据...")
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
    except Exception as e:
        return {}

# ---------------------------
# 数据获取与缓存核心
# ---------------------------
def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_INDUSTRY
    if not trade_days_list: return False
    
    with st.spinner("正在同步全市场行业数据 (白名单)..."):
        GLOBAL_STOCK_INDUSTRY = load_industry_mapping()

    if use_cache and os.path.exists(CACHE_FILE_NAME):
        st.success(f"⚡ 发现本地行情缓存 ({CACHE_FILE_NAME})，极速加载中...")
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
            st.info("✅ 本地缓存加载成功！")
            return True
        except Exception as e:
            st.warning("缓存文件损坏，将重新下载...")
            os.remove(CACHE_FILE_NAME)

    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 获取长达 250 天的数据以支撑周线指标计算
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=250)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=120) # 往后多取以便结算
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty: return False
        
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    st.info(f"📡 [首次运行] 正在下载复权行情数据: {start_date} 至 {end_date} (下载后将自动缓存)...")
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

def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    
    latest_adj_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj_factor): return pd.DataFrame() 

    try:
        daily_df = GLOBAL_DAILY_RAW.loc[ts_code]
        daily_df = daily_df.loc[(daily_df.index >= start_date) & (daily_df.index <= end_date)]
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
        
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 

# ---------------------------
# 核心指标计算 (日线 + 周线共振)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def compute_trend_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=250)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date, end_date)
    res = {}
    if df.empty or len(df) < 60: return res 
    
    close_d = df['close']
    res['last_close'] = close_d.iloc[-1]
    res['last_high'] = df['high'].iloc[-1]
    res['last_vol'] = df['vol'].iloc[-1]
    res['prev_high'] = df['high'].iloc[-2]
    res['prev_vol'] = df['vol'].iloc[-2]
    res['prev_close'] = close_d.iloc[-2]
    
    # 均线与日线 MACD 用于打分
    df['ma5'] = close_d.rolling(5).mean()
    res['ma5'] = df['ma5'].iloc[-1]
    
    ema12_d = close_d.ewm(span=12, adjust=False).mean()
    ema26_d = close_d.ewm(span=26, adjust=False).mean()
    diff_d = ema12_d - ema26_d
    dea_d = diff_d.ewm(span=9, adjust=False).mean()
    res['macd_val'] = ((diff_d - dea_d) * 2).iloc[-1]
    
    # 🌟 周线合成与指标
    df.index = pd.to_datetime(df.index)
    weekly_df = df.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'}).dropna()
    if len(weekly_df) < 26: return res
    
    close_w = weekly_df['close']
    weekly_df['ma10_w'] = close_w.rolling(10).mean()
    weekly_df['ma20_w'] = close_w.rolling(20).mean()
    
    # 周线 MACD
    ema12_w = close_w.ewm(span=12, adjust=False).mean()
    ema26_w = close_w.ewm(span=26, adjust=False).mean()
    diff_w = ema12_w - ema26_w
    dea_w = diff_w.ewm(span=9, adjust=False).mean()
    weekly_macd = (diff_w - dea_w) * 2
    res['weekly_macd_val'] = weekly_macd.iloc[-1]
    
    # 空间确认：回踩 10周或20周线附近 (5%内)
    curr_w_close = close_w.iloc[-1]
    ma10_w = weekly_df['ma10_w'].iloc[-1]
    ma20_w = weekly_df['ma20_w'].iloc[-1]
    
    dist_10 = abs(curr_w_close - ma10_w) / ma10_w if ma10_w > 0 else 1
    dist_20 = abs(curr_w_close - ma20_w) / ma20_w if ma20_w > 0 else 1
    
    res['dist_to_support'] = min(dist_10, dist_20)
    res['is_near_support'] = res['dist_to_support'] <= 0.05
    
    return res

# ---------------------------
# 实战波段胶囊 (时间追踪与铁血纪律)
# ---------------------------
def get_medium_term_future(ts_code, selection_date, buy_price, hold_weeks=8, stop_loss_pct=15.0):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=150)).strftime("%Y%m%d") 
    
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_future, end_date=end_future)
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    
    if hist.empty or len(hist) < 2: return results
    
    hist['open'] = pd.to_numeric(hist['open'], errors='coerce')
    hist['low'] = pd.to_numeric(hist['low'], errors='coerce')
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    
    # 🌟 修复 Pandas 2.0+ 版本的 bfill 写法
    hist['ma20'] = hist['close'].rolling(20).mean().bfill()
    
    stop_loss_price = buy_price * (1 - stop_loss_pct / 100.0)
    exit_triggered = False
    
    for i in range(len(hist)):
        if i >= hold_weeks * 5: break # 最多统计 8 周 (40个交易日)
            
        row = hist.iloc[i]
        day_count = i + 1
        current_week = ((day_count - 1) // 5) + 1 
        current_ma20 = row['ma20']
        
        # 🌟 铁血纪律：盘中跌破 20日线 或 触及 15% 硬止损
        final_return = np.nan
        if row['low'] <= stop_loss_price:
            final_return = -stop_loss_pct
            exit_triggered = True
        elif row['low'] <= current_ma20 and current_ma20 > 0:
            final_return = (current_ma20 - buy_price) / buy_price * 100.0
            exit_triggered = True
            
        if exit_triggered:
            results[f'Return_W{current_week} (%)'] = final_return
            break 
            
        # 正常持仓切片
        if day_count % 5 == 0:
            week_num = day_count // 5
            results[f'Return_W{week_num} (%)'] = (row['close'] - buy_price) / buy_price * 100.0
            
    # 如果到了最后一天仍未触发离场，填入最后一天的结算
    if not exit_triggered and len(hist) >= hold_weeks * 5:
        last_price = hist.iloc[hold_weeks * 5 - 1]['close']
        results[f'Return_W{hold_weeks} (%)'] = (last_price - buy_price) / buy_price * 100.0
        
    return results

# ---------------------------
# 核心回测循环
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, MIN_MV, MAX_MV, MIN_PRICE, STOP_LOSS_PCT):
    global GLOBAL_STOCK_INDUSTRY
    
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), "数据缺失"

    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name')
    if stock_basic.empty: stock_basic = safe_get('stock_basic', list_status='L')
        
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    
    daily_basic = safe_get('daily_basic', trade_date=last_trade)
    if not daily_basic.empty:
        df = df.merge(daily_basic[['ts_code','circ_mv']], on='ts_code', how='left')
    else: return pd.DataFrame(), "市值数据缺失"
    
    mf_raw = safe_get('moneyflow', trade_date=last_trade)
    if not mf_raw.empty:
        mf = mf_raw[['ts_code','net_mf_amount']].rename(columns={'net_mf_amount':'net_mf'})
        df = df.merge(mf, on='ts_code', how='left')
    else: df['net_mf'] = 0 
    
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    
    # 🌟 强硬护城河过滤
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')] 
    df = df[(df['close'] >= MIN_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    
    records = []
    for row in df.itertuples():
        # 白名单过滤
        if GLOBAL_STOCK_INDUSTRY and row.ts_code not in GLOBAL_STOCK_INDUSTRY: 
            continue
            
        ind = compute_trend_indicators(row.ts_code, last_trade)
        if not ind: continue
        
        # 1. 周线多头且回踩到位 (设伏)
        if ind.get('weekly_macd_val', -1) <= 0: continue
        if not ind.get('is_near_support', False): continue
        
        # 2. 日线形态发车 (阳吞阴 + 站上5日线 + 放量)
        if ind['last_close'] <= ind['ma5']: continue
        if ind['last_close'] <= ind['prev_high']: continue
        if ind['last_vol'] <= ind['prev_vol']: continue
        
        # 记录未来 8 周轨迹
        future_returns = get_medium_term_future(row.ts_code, last_trade, ind['last_close'], 8, STOP_LOSS_PCT)
        
        record_dict = {
            'ts_code': row.ts_code, 'name': row.name, 'Close': row.close, 
            'circ_mv': row.circ_mv_billion, 'net_mf': row.net_mf,
            'macd_val': ind['macd_val'], 'dist_to_support': ind['dist_to_support']
        }
        record_dict.update(future_returns)
        records.append(record_dict)
            
    if not records: return pd.DataFrame(), "无标的"
    fdf = pd.DataFrame(records)
    
    # 🌟 中线三维打分系统
    def dynamic_score_medium_term(r):
        mf_ratio = r['net_mf'] / (r['circ_mv'] * 10000 + 1) if r['circ_mv'] > 0 else 0
        score_mf = min(max(mf_ratio * 10000, -500), 1000) 
        score_macd = r['macd_val'] * 100 
        # 距离支撑越近得分越高
        score_support = (0.05 - r['dist_to_support']) * 10000 
        return score_mf + score_macd + score_support

    fdf['Score'] = fdf.apply(dynamic_score_medium_term, axis=1)
    final_df = fdf.sort_values('Score', ascending=False).head(TOP_BACKTEST).copy()
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    
    return final_df, None

# ---------------------------
# UI 及 主程序
# ---------------------------
with st.sidebar:
    st.header("V37.0 中线共振狙击")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数", value=100, step=1, help="中线推荐测百日")
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=3, help="中线选股少而精")
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 清除行情缓存"):
        if os.path.exists(CACHE_FILE_NAME):
            os.remove(CACHE_FILE_NAME)
            st.success("缓存已清除，下次运行将重新下载最新数据。")
    CHECKPOINT_FILE = "backtest_checkpoint_v37.csv" 
    
    st.markdown("---")
    st.subheader("💰 核心门槛")
    MIN_PRICE = st.number_input("最低股价", value=20.0, help="20元护城河") 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=200.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)
    
    st.markdown("---")
    st.subheader("🛡️ 纪律风控")
    STOP_LOSS_PCT = st.number_input("硬性止损线 (%)", value=15.0)
    st.info("动态止盈：盘中跌破 20日线无条件结算")

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🚀 启动 V37.0 轨迹追踪"):
    processed_dates = set()
    results = []
    
    if RESUME_CHECKPOINT and os.path.exists(CHECKPOINT_FILE):
        try:
            existing_df = pd.read_csv(CHECKPOINT_FILE)
            existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
            processed_dates = set(existing_df['Trade_Date'].unique())
            results.append(existing_df)
            st.success(f"✅ 断点续传，跳过 {len(processed_dates)} 个交易日...")
        except:
            if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    else:
        if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
        
    trade_days_list = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_list: st.stop()
    
    if not get_all_historical_data(trade_days_list, use_cache=True): st.stop()
            
    dates_to_run = [d for d in trade_days_list if d not in processed_dates]
    if not dates_to_run:
        st.success("🎉 回测已全部完毕！")
    else:
        bar = st.progress(0, text="回测引擎启动...")
        for i, date in enumerate(dates_to_run):
            res, err = run_backtest_for_a_day(date, int(TOP_BACKTEST), MIN_MV, MAX_MV, MIN_PRICE, STOP_LOSS_PCT)
            if not res.empty:
                res['Trade_Date'] = date
                is_first = not os.path.exists(CHECKPOINT_FILE)
                res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                results.append(res)
            bar.progress((i+1)/len(dates_to_run), text=f"分析中: {date}")
        bar.empty()
    
    if results:
        all_res = pd.concat(results)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
        st.header(f"📊 V37.0 中线共振狙击 (白名单+生存轨迹)")
        st.subheader("🗓️ 周度生存与收益切片")
        
        cols = st.columns(4)
        for w in range(1, 9):
            col_name = f'Return_W{w} (%)'
            valid = all_res.dropna(subset=[col_name]) 
            with cols[(w-1) % 4]:
                if not valid.empty:
                    avg = valid[col_name].mean()
                    win = (valid[col_name] > 0).mean() * 100
                    st.metric(f"W{w} 均益/胜率 (存活{len(valid)}只)", f"{avg:.2f}% / {win:.1f}%")
                else:
                    st.metric(f"W{w} 无持仓", "N/A")
            if w == 4: st.write("")
 
        st.subheader("📋 回测清单 (W1-W8 完整轨迹)")
        display_cols = ['Rank', 'Trade_Date', 'name', 'ts_code', 'Close', 'circ_mv'] + [f'Return_W{w} (%)' for w in range(1, 9)]
        final_cols = [c for c in display_cols if c in all_res.columns]
    
        display_df = all_res[final_cols].sort_values(['Trade_Date', 'Rank'], ascending=[False, True])
        st.dataframe(display_df, use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载完整轨迹 (CSV)", csv, f"export_v37_0.csv", "text/csv")
    else:
        st.warning("⚠️ 深度筛选后未发现符合大资金共振的标的。请耐心等待。")
