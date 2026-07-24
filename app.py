# -*- coding: utf-8 -*-
"""
选股王 · V38.1 中线共振狙击版 (终极公平打分 + 强过滤假突破)
------------------------------------------------
逻辑说明:
1. [中线股票池] 严格锁定流通市值 200亿-1000亿，股价 >= 20元。
2. [白名单赛道] 电子、计算机、通信、医药生物、国防军工、机械设备。
3. [三维打分取Top3] 公平起跑线，取消一切市值与板块偏见，纯粹考核当天攻击力！
   - 涨幅动能：突破日真实涨幅转换的分数。
   - 攻击量能：突破日成交量对比5日均量的放大倍数。
4. [绞杀假突破与回踩动能确认] 
   - MA20必须走平或向上 (杜绝下跌趋势反抽)。
   - K线必须是饱满阳线（实体占振幅60%以上，杜绝长上影线）。
   - MACD大趋势在水上 (DIF > 0)，且动能拐头向上 (今日MACD > 昨日MACD，绿柱缩短或红柱放大)。
   - 突破日真实放量（>1.2倍五日均量）。
5. [三级防守] 初始装死 -> 挂20日线 -> 偏离15%挂10日线。
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

# 更新缓存文件名以防止冲突
CACHE_FILE_NAME = "market_data_cache_v38_1_final.pkl" 

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
st.set_page_config(page_title="选股王 V38.1 终极纯粹", layout="wide")
st.title("选股王 V38.1：绞杀假突破与公平动能打分")

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
        except Exception:
            st.warning("缓存文件损坏，将重新下载...")
            os.remove(CACHE_FILE_NAME)

    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=250)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=150) 
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty: return False
        
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    st.info(f"📡 [首次运行] 正在下载复权行情数据: {start_date} 至 {end_date}...")
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
        
    return df[['open', 'high', 'low', 'close', 'pre_close', 'vol']].copy() 

# ---------------------------
# 核心指标计算 (V38.1 强过滤版)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def compute_trend_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=250)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date, end_date)
    res = {}
    if df.empty or len(df) < 120: return res 
    
    # 均线
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['ma120'] = df['close'].rolling(120).mean()
    df['ma5_vol'] = df['vol'].rolling(5).mean()
    
    # MACD计算
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    df = df.dropna().reset_index()
    if len(df) < 4: return res
    
    row = df.iloc[-1]
    prev_row = df.iloc[-2]
    prev2_row = df.iloc[-3]
    
    # --- 原有形态条件 ---
    is_trend_up = row['ma60'] > row['ma120']
    is_pulled_back = (prev2_row['close'] < prev2_row['ma20']) and (prev_row['close'] < prev_row['ma20'])
    is_breakout = row['close'] > row['ma20']
    
    # --- 🚨 核心修正：绞杀假突破与回踩反转确认 ---
    # 1. 均线方向：今天的 MA20 必须不低于昨天 (走平或向上)，过滤下跌趋势反抽
    is_ma20_healthy = row['ma20'] >= prev_row['ma20']
    
    # 2. 真实放量：今日成交量必须明显放大 (>1.2倍五日均量)
    is_vol_strong = row['vol'] > (1.2 * row['ma5_vol'])
    
    # 3. 饱满K线过滤长上影线：必须是红盘，且实体占全天振幅的60%以上
    candle_range = row['high'] - row['low']
    candle_body = row['close'] - row['open']
    is_solid_yang = (row['close'] > row['open']) and (candle_body >= candle_range * 0.6 if candle_range > 0 else True)
    
    # 4. MACD逻辑修正 (契合回踩反转)：
    #    - 大趋势在水上 (dif > 0)
    #    - 动能拐头向上 (今天的macd柱子大于昨天的，即绿柱缩短或红柱变长)
    is_macd_healthy = (row['dif'] > 0) and (row['macd'] > prev_row['macd'])
    
    # 综合判定
    res['is_v38_buy_signal'] = (is_trend_up and is_pulled_back and is_breakout 
                                and is_ma20_healthy and is_vol_strong 
                                and is_solid_yang and is_macd_healthy)
    
    if res['is_v38_buy_signal']:
        # 记录纯粹打分需要的数据
        res['vol_ratio'] = row['vol'] / row['ma5_vol']  # 量比
        res['pre_close'] = prev_row['close']            # 昨日收盘价用于计算涨幅
        
    res['last_close'] = row['close']
    res['bottom_line'] = row['low'] 
    res['ma20'] = row['ma20']
    
    return res

# ---------------------------
# V38.1 核心大脑：三级动态防御系统
# ---------------------------
def get_medium_term_future(ts_code, selection_date, buy_price, bottom_line, hold_weeks=8):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_fetch = (d0 - timedelta(days=60)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=150)).strftime("%Y%m%d") 
    
    hist_full = get_qfq_data_v4_optimized_final(ts_code, start_date=start_fetch, end_date=end_future)
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    results['Exit_Reason'] = "持仓中"
    
    if hist_full.empty or len(hist_full) < 30: return results
    
    hist_full['open'] = pd.to_numeric(hist_full['open'], errors='coerce')
    hist_full['high'] = pd.to_numeric(hist_full['high'], errors='coerce')
    hist_full['low'] = pd.to_numeric(hist_full['low'], errors='coerce')
    hist_full['close'] = pd.to_numeric(hist_full['close'], errors='coerce')
    
    hist_full['ma10'] = hist_full['close'].rolling(10).mean()
    hist_full['ma20'] = hist_full['close'].rolling(20).mean()
    
    hist_future = hist_full[hist_full.index > selection_date]
    
    ma20_active = False
    ma10_active = False
    exit_triggered = False
    
    for i in range(len(hist_future)):
        if i >= hold_weeks * 5: break 
            
        row = hist_future.iloc[i]
        day_count = i + 1
        current_week = ((day_count - 1) // 5) + 1 
        
        curr_close = row['close']
        curr_high = row['high']
        curr_ma10 = row['ma10']
        curr_ma20 = row['ma20']
        
        if not ma10_active:
            current_bias = (curr_high - curr_ma20) / curr_ma20
            if current_bias >= 0.15:
                ma10_active = True
                ma20_active = True 
                
        if not ma20_active and not ma10_active:
            profit_pct = (curr_close - buy_price) / buy_price
            if profit_pct >= 0.10 or curr_ma20 > buy_price:
                ma20_active = True 
                
        final_return = np.nan
        if ma10_active:
            if curr_close < curr_ma10:
                final_return = (curr_close - buy_price) / buy_price * 100.0
                exit_triggered = True
                results['Exit_Reason'] = "二档止盈(破10日线)"
        elif ma20_active:
            if curr_close < curr_ma20:
                final_return = (curr_close - buy_price) / buy_price * 100.0
                exit_triggered = True
                results['Exit_Reason'] = "一档防守(破20日线)"
        else:
            if curr_close < bottom_line:
                final_return = (curr_close - buy_price) / buy_price * 100.0
                exit_triggered = True
                results['Exit_Reason'] = "假突破(破底线)"
                
        if exit_triggered:
            results[f'Return_W{current_week} (%)'] = final_return
            break 
            
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
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, MIN_MV, MAX_MV, MIN_PRICE):
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
    
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    
    # 严格锁死物理护城河条件
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')] 
    df = df[(df['close'] >= MIN_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    
    records = []
    for row in df.itertuples():
        if GLOBAL_STOCK_INDUSTRY and row.ts_code not in GLOBAL_STOCK_INDUSTRY: 
            continue
            
        ind = compute_trend_indicators(row.ts_code, last_trade)
        if not ind or not ind.get('is_v38_buy_signal'): 
            continue
            
        # ✅ V38.1 纯粹动能公平打分系统
        # 1. 突破涨幅得分 (涨幅10%即得100分，客观反映攻击力)
        pct_chg = (ind['last_close'] - ind['pre_close']) / ind['pre_close'] * 100
        score_breakout = pct_chg * 10 
        
        # 2. 攻击量能得分 (突破日量比放大倍数)
        score_vol = ind['vol_ratio'] * 10
        
        # 总分：没有任何外在偏见，纯粹用K线实体涨幅与量能说话
        total_score = score_breakout + score_vol
        
        future_returns = get_medium_term_future(row.ts_code, last_trade, ind['last_close'], ind['bottom_line'], 8)
        
        record_dict = {
            'ts_code': row.ts_code, 'name': row.name, 'Close': row.close, 
            'circ_mv': row.circ_mv_billion,
            'Total_Score': round(total_score, 1),
            'Breakout_S': round(score_breakout, 1),
            'Volume_S': round(score_vol, 1)
        }
        record_dict.update(future_returns)
        records.append(record_dict)
            
    if not records: return pd.DataFrame(), "无标的"
    
    fdf = pd.DataFrame(records)
    
    # 按照公平竞技的总分排序，选取TopK
    final_df = fdf.sort_values('Total_Score', ascending=False).head(TOP_BACKTEST).copy()
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    
    return final_df, None

# ---------------------------
# UI 及 主程序
# ---------------------------
with st.sidebar:
    st.header("V38.1 终极公平过滤版")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数", value=100, step=1)
    
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=3, help="只取公平动能打分最高的前三名")
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 清除行情缓存"):
        if os.path.exists(CACHE_FILE_NAME):
            os.remove(CACHE_FILE_NAME)
            st.success("缓存已清除，下次运行将重新下载最新数据。")
    CHECKPOINT_FILE = "backtest_checkpoint_v38_1_final.csv" 
    if st.button("🗑️ 清除断点记录 (重新回测)"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            st.success("旧进度已清理！")
            
    st.markdown("---")
    st.subheader("💰 核心护城河门槛")
    MIN_PRICE = st.number_input("最低股价 (元)", value=20.0) 
    col1, col2 = st.columns(2)
    # 严格贯彻：这里只作为准入通道，不再进行加减分干预
    MIN_MV = col1.number_input("最小市值(亿)", value=200.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🚀 启动 V38.1 极简动能追踪"):
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
        st.success("🎉 回测已全部完毕！")
    else:
        bar = st.progress(0, text="强力过滤与公平打分引擎运行中...")
        for i, date in enumerate(dates_to_run):
            res, err = run_backtest_for_a_day(date, int(TOP_BACKTEST), MIN_MV, MAX_MV, MIN_PRICE)
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
        
        st.header(f"📊 V38.1 终极版 (严打假突破 + 公平竞技)")
        st.subheader("🗓️ 周度生存与收益切片")
        
        cols_row1 = st.columns(4)
        cols_row2 = st.columns(4)
        
        for w in range(1, 9):
            col_name = f'Return_W{w} (%)'
            valid = all_res.dropna(subset=[col_name]) 
            
            target_col = cols_row1[w-1] if w <= 4 else cols_row2[w-5]
            
            with target_col:
                if not valid.empty:
                    avg = valid[col_name].mean()
                    win = (valid[col_name] > 0).mean() * 100
                    st.metric(f"W{w} 均益/胜率 (存活{len(valid)}只)", f"{avg:.2f}% / {win:.1f}%")
                else:
                    st.metric(f"W{w} 无持仓", "N/A")
 
        st.subheader("📋 优等生清单")
        display_cols = ['Rank', 'Trade_Date', 'name', 'ts_code', 'Close', 'Total_Score', 'Breakout_S', 'Volume_S', 'circ_mv', 'Exit_Reason'] + [f'Return_W{w} (%)' for w in range(1, 9)]
        final_cols = [c for c in display_cols if c in all_res.columns]
    
        display_df = all_res[final_cols].sort_values(['Trade_Date', 'Rank'], ascending=[False, True]).reset_index(drop=True)
        
        def color_exit(val):
            if isinstance(val, str):
                if '假突破' in val: return 'color: red'
                elif '二档' in val: return 'color: magenta'
                elif '一档' in val: return 'color: green'
            return ''
        
        if 'Exit_Reason' in display_df.columns:
            try:
                st.dataframe(display_df.style.map(color_exit, subset=['Exit_Reason']), use_container_width=True)
            except AttributeError:
                st.dataframe(display_df.style.applymap(color_exit, subset=['Exit_Reason']), use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载完整轨迹 (CSV)", csv, f"export_v38_1_final.csv", "text/csv")
    else:
        st.warning("⚠️ 强力过滤后未发现符合真实大资金共振的标的。请耐心等待确定性更高的机会。")
