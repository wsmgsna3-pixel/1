# -*- coding: utf-8 -*-
"""
选股王 · N型主升浪起爆版
------------------------------------------------
融合机制:
1. 继承 V35.1 框架的缓存、前复权、断点续传、UI系统。
2. 植入全新核心策略: N型突破前高(120日) + 缩量回踩颈线确认 + 突破日筹码获利盘>95%。
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
st.set_page_config(page_title="选股王 N型主升版", layout="wide")
st.title("选股王：N型主升浪起爆版 🚀")

# ---------------------------
# 基础 API 函数 (继承V35)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: 
        return pd.DataFrame(columns=['ts_code']) 
   
    func = getattr(pro, func_name) 
    try:
        for _ in range(3):
            try:
                if kwargs.get('is_index'):
                    df = pro.index_daily(**kwargs)
                else:
                    df = func(**kwargs)
                
                if df is not None and not df.empty:
                    return df
                time.sleep(0.5)
            except:
                time.sleep(1)
                continue
        return pd.DataFrame(columns=['ts_code']) 
    except Exception as e:
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    lookback_days = max(num_days * 3, 365) 
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    
    if cal.empty or 'cal_date' not in cal.columns:
        return []
        
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()

@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    return {'adj': adj_df, 'daily': daily_df}

# ---------------------------
# 数据获取核心 (本地缓存版)
# ---------------------------
CACHE_FILE_NAME = "market_data_cache_N_type.pkl"

def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    
    if not trade_days_list: return False

    if use_cache and os.path.exists(CACHE_FILE_NAME):
        st.success(f"⚡ 发现本地行情缓存，正在极速加载...")
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
        except Exception as e:
            st.warning(f"缓存损坏，将重新下载...")
            os.remove(CACHE_FILE_NAME)

    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=200) # 为120日均线多留数据
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=30)
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    
    if all_trade_dates_df.empty: return False
        
    all_dates = all_trade_dates_df['cal_date'].tolist()
    adj_factor_data_list, daily_data_list = [], []

    progress_text = "Tushare 行情底座下载中 (首次较慢，后续秒开)..."
    my_bar = st.progress(0, text=progress_text)
    total_steps = len(all_dates)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_date = {executor.submit(fetch_and_cache_daily_data, date): date for date in all_dates}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_date)):
            try:
                data = future.result()
                if not data['adj'].empty: adj_factor_data_list.append(data['adj'])
                if not data['daily'].empty: daily_data_list.append(data['daily'])
            except: pass
            if i % 5 == 0 or i == total_steps - 1:
                my_bar.progress((i + 1) / total_steps, text=f"下载中: {i+1}/{total_steps}")
    my_bar.empty()
    
    if not daily_data_list: return False
   
    with st.spinner("正在构建高速缓存索引..."):
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
        except: pass
    return True

# ---------------------------
# 复权核心
# ---------------------------
def get_qfq_data(ts_code, start_date, end_date):
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
    
    df = daily_df.merge(adj_series.rename('adj_factor'), left_index=True, right_index=True, how='left').dropna(subset=['adj_factor'])
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] * df['adj_factor'] / latest_adj_factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'}).sort_values('trade_date_str').set_index('trade_date_str')
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 

def get_future_prices(ts_code, selection_date, d0_close):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    hist = get_qfq_data(ts_code, start_future, end_future)
    results = {'Return_D3 (%)': np.nan, 'Return_D5 (%)': np.nan, 'Return_D10 (%)': np.nan, 'Max_Drawdown_3D (%)': np.nan}
    if hist.empty: return results
    
    next_open = hist.iloc[0]['open']
    if next_open <= d0_close: return results # 规避一字跌停
    
    # 模拟实盘以次日开盘价偏上一点买入
    buy_price = next_open * 1.01
    
    for n, key in zip([3, 5, 10], ['Return_D3 (%)', 'Return_D5 (%)', 'Return_D10 (%)']):
        if len(hist) >= n:
            results[key] = (hist.iloc[n-1]['close'] - buy_price) / buy_price * 100
            
    # 计算3天内最大回撤（扫损测试）
    if len(hist) >= 3:
        min_low_3d = hist.iloc[:3]['low'].min()
        results['Max_Drawdown_3D (%)'] = (min_low_3d - buy_price) / buy_price * 100
        
    return results

# ---------------------------
# N型主升浪起爆点 核心引擎
# ---------------------------
def run_n_type_backtest_for_a_day(last_trade, TOP_K, MIN_PRICE, MIN_MV, MAX_MV, NECK_BUFFER, WIN_RATE_LIMIT, VOL_MULTIPLIER):
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), "数据缺失"

    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name')
    daily_basic = safe_get('daily_basic', trade_date=last_trade)
    
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    if not daily_basic.empty:
        df = df.merge(daily_basic[['ts_code','turnover_rate','circ_mv']], on='ts_code', how='left')
    
    if 'circ_mv' not in df.columns: return pd.DataFrame(), "市值数据缺失"
    
    # 第一层：基础过滤网
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')] 
    df = df[(df['close'] >= MIN_PRICE) & (df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    
    if len(df) == 0: return pd.DataFrame(), "过滤后无标的"

    records = []
    # 提取150天前的数据起点，用于计算120日均线和5日均量
    lookback_start = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=250)).strftime("%Y%m%d")

    for row in df.itertuples():
        hist = get_qfq_data(row.ts_code, start_date=lookback_start, end_date=last_trade)
        if len(hist) < 125: continue # 上市不足半年不碰
        
        hist['vol_ma5'] = hist['vol'].rolling(window=5).mean()
        # 计算120日最高点（颈线），shift(1)表示不包含当天
        hist['high_120'] = hist['high'].shift(1).rolling(window=120).max()
        
        today_data = hist.iloc[-1]
        today_close = today_data['close']
        today_vol = today_data['vol']
        today_vol_ma5 = today_data['vol_ma5']
        neckline = today_data['high_120']
        
        # 第二层：形态过滤（今天必须是缩量，且价格刚好停在颈线上方不远处）
        if pd.isna(neckline): continue
        if today_vol >= today_vol_ma5: continue # 必须缩量回踩
        
        # 回踩空间判断：收盘价必须在颈线以上，且不超过设定的容忍度（默认5%）
        if not (neckline <= today_close <= neckline * (1 + NECK_BUFFER/100)): continue
        
        # 第三层：历史追溯（寻找过去5天内的有效突破日）
        breakout_date_str = None
        breakout_win_rate = 0
        
        # 往回找1到5天
        for i in range(2, 7):
            past_data = hist.iloc[-i]
            past_close = past_data['close']
            past_vol = past_data['vol']
            past_vol_ma5 = past_data['vol_ma5']
            past_neckline = past_data['high_120']
            
            # 判定突破：收盘越过前高，且成交量必须爆量
            if past_close > past_neckline and past_vol > (past_vol_ma5 * VOL_MULTIPLIER):
                p_date = hist.index[-i]
                
                # 第四层：筹码定海神针（定向调用那天的cyq_perf检查获利盘）
                chip_df = safe_get('cyq_perf', ts_code=row.ts_code, trade_date=p_date)
                if not chip_df.empty:
                    p_win_rate = chip_df.iloc[0]['winner_rate']
                    if p_win_rate >= WIN_RATE_LIMIT:
                        breakout_date_str = p_date
                        breakout_win_rate = p_win_rate
                        break
        
        if breakout_date_str:
            # 满足所有条件，入围待评
            neckline_deviation = (today_close - neckline) / neckline * 100
            future = get_future_prices(row.ts_code, last_trade, today_close)
            
            records.append({
                'ts_code': row.ts_code, 'name': row.name, 'Close': today_close, 
                'Neckline': round(neckline, 2), 
                'Dev_Neck(%)': round(neckline_deviation, 2),
                'Breakout_Date': breakout_date_str,
                'Breakout_WinRate': round(breakout_win_rate, 2),
                'Return_D3 (%)': future.get('Return_D3 (%)', np.nan),
                'Return_D5 (%)': future.get('Return_D5 (%)', np.nan),
                'Return_D10 (%)': future.get('Return_D10 (%)', np.nan),
                'Max_Drawdown_3D (%)': future.get('Max_Drawdown_3D (%)', np.nan)
            })

    if not records: return pd.DataFrame(), "深度筛选后无标的"
    
    fdf = pd.DataFrame(records)
    
    # 动态评分系统 (胜率越高越好，乖离率越小越好)
    def calculate_score(r):
        score = r['Breakout_WinRate'] * 10 # 获利盘占比极大权重
        score -= (r['Dev_Neck(%)'] * 20)   # 偏离颈线越多扣分越多
        return score

    fdf['Score'] = fdf.apply(calculate_score, axis=1)
    final_df = fdf.sort_values('Score', ascending=False).head(TOP_K).copy()
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    
    return final_df, None

# ---------------------------
# UI 及 主程序
# ---------------------------
with st.sidebar:
    st.header("N型主升浪起爆版 🚀")
    backtest_date_end = st.date_input("回测截止日", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("连续测试天数", value=30, step=1)
    TOP_BACKTEST = st.number_input("每日优选名额 (Top K)", value=5, help="输出最终前5名")
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 重置策略缓存"):
        if os.path.exists(CACHE_FILE_NAME): os.remove(CACHE_FILE_NAME)
        st.success("底层数据已清空，即将重载。")
    CHECKPOINT_FILE = "backtest_checkpoint_N_type.csv" 
    
    st.markdown("---")
    st.subheader("💰 资产底座")
    col1, col2 = st.columns(2)
    MIN_PRICE = col1.number_input("最低股价", value=10.0) 
    MIN_MV = col2.number_input("最小市值(亿)", value=30.0) 
    MAX_MV = st.number_input("最大市值(亿)", value=500.0)
    
    st.markdown("---")
    st.subheader("⚔️ N型突破参数")
    WIN_RATE_LIMIT = st.number_input("突破日极值获利盘 (%)", value=95.0, help="主力解放全套牢盘的诚意")
    VOL_MULTIPLIER = st.number_input("突破爆量倍数", value=1.5, help="突破日成交量必须是5日均量的多少倍")
    NECK_BUFFER = st.number_input("回踩颈线容错率 (%)", value=5.0, help="最高点到颈线的距离0-5%")

TS_TOKEN = st.text_input("Tushare Token (安全加密层)", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🎯 启动系统"):
    processed_dates = set()
    results = []
    
    if RESUME_CHECKPOINT and os.path.exists(CHECKPOINT_FILE):
        try:
            existing_df = pd.read_csv(CHECKPOINT_FILE)
            existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
            processed_dates = set(existing_df['Trade_Date'].unique())
            results.append(existing_df)
            st.success(f"✅ 检测到断点存档，跳过 {len(processed_dates)} 天...")
        except:
            if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    else:
        if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    
    trade_days_list = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_list: st.stop()
         
    dates_to_run = [d for d in trade_days_list if d not in processed_dates]
    
    if not dates_to_run:
        st.success("🎉 数据集已全部覆盖测试完成！")
    else:
        if not get_all_historical_data(trade_days_list, use_cache=True): st.stop()
            
        bar = st.progress(0, text="猎手引擎正在全市场检索...")
        
        for i, date in enumerate(dates_to_run):
            res, err = run_n_type_backtest_for_a_day(
                date, int(TOP_BACKTEST), MIN_PRICE, MIN_MV, MAX_MV, NECK_BUFFER, WIN_RATE_LIMIT, VOL_MULTIPLIER
            )
            if not res.empty:
                res['Trade_Date'] = date
                is_first = not os.path.exists(CHECKPOINT_FILE)
                res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                results.append(res)
            
            bar.progress((i+1)/len(dates_to_run), text=f"量化检索中: {date}")
        
        bar.empty()
    
    if results:
        all_res = pd.concat(results)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        all_res = all_res.sort_values(['Trade_Date', 'Rank'], ascending=[False, True])
        
        st.header(f"📊 N型战法 统计结果 (Top {TOP_BACKTEST})")
        cols = st.columns(3)
        for idx, n in enumerate([3, 5, 10]):
            col_name = f'Return_D{n} (%)'
            valid = all_res.dropna(subset=[col_name]) 
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                cols[idx].metric(f"买入后 {n} 天 均益/胜率", f"{avg:.2f}% / {win:.1f}%")
 
        st.subheader("📋 实盘买点清单")
        
        show_cols = ['Rank', 'Trade_Date','name','ts_code','Close','Neckline','Dev_Neck(%)',
                     'Breakout_Date','Breakout_WinRate',
                     'Return_D3 (%)', 'Return_D5 (%)','Return_D10 (%)', 'Max_Drawdown_3D (%)']
        final_cols = [c for c in show_cols if c in all_res.columns]
    
        st.dataframe(all_res[final_cols], use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 导出清单到 Excel", csv, f"N_Type_export.csv", "text/csv")
    else:
        st.warning("⚠️ 在指定的历史阶段内，市场没有产生符合如此苛刻条件的股票。")
