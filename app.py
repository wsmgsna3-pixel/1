# -*- coding: utf-8 -*-
"""
选股王 · V36 黄金坑主升版
------------------------------------------------
实战逻辑极致升级 (User Customized):
1. 锁定“黄金坑”：拒绝慢牛爬坡，前高(坑沿)必须在10个交易日之前，且坑深必须>15%。
2. 动量突破：今天一根大阳线，放量打穿这个蓄谋已久的坑沿，获利盘瞬间>90%。
3. 次日防守：第二天开盘价如果低于突破日收盘价的 -2%，直接视为诱多废弃。
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
# 全局变量与配置
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

st.set_page_config(page_title="选股王 V36 黄金坑主升版", layout="wide")
st.title("选股王：V36 黄金坑主升版 🏆")

# ---------------------------
# 基础 API 与 缓存底座 (与V35一致，稳定极速)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        for _ in range(3):
            try:
                df = pro.index_daily(**kwargs) if kwargs.get('is_index') else func(**kwargs)
                if df is not None and not df.empty: return df
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
    if cal.empty or 'cal_date' not in cal.columns: return []
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    return trade_days_df[trade_days_df['cal_date'] <= end_date_str]['cal_date'].head(num_days).tolist()

@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    return {'adj': safe_get('adj_factor', trade_date=date), 'daily': safe_get('daily', trade_date=date)}

CACHE_FILE_NAME = "market_data_cache_V36.pkl"

def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    if use_cache and os.path.exists(CACHE_FILE_NAME):
        st.success(f"⚡ 发现本地缓存，极速加载中...")
        try:
            with open(CACHE_FILE_NAME, 'rb') as f:
                cached_data = pickle.load(f)
                GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW = cached_data['adj'], cached_data['daily']
            latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
            if latest_global_date:
                latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
                GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
            return True
        except:
            st.warning(f"缓存损坏，将重新下载...")
            os.remove(CACHE_FILE_NAME)

    latest_trade_date, earliest_trade_date = max(trade_days_list), min(trade_days_list)
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=30)).strftime("%Y%m%d")
    
    all_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_dates_df.empty: return False
        
    adj_list, daily_list = [], []
    all_dates = all_dates_df['cal_date'].tolist()
    my_bar = st.progress(0, text="底座数据下载中 (首次较慢)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_date = {executor.submit(fetch_and_cache_daily_data, date): date for date in all_dates}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_date)):
            try:
                data = future.result()
                if not data['adj'].empty: adj_list.append(data['adj'])
                if not data['daily'].empty: daily_list.append(data['daily'])
            except: pass
            if i % 5 == 0 or i == len(all_dates) - 1: my_bar.progress((i + 1) / len(all_dates))
    my_bar.empty()
    if not daily_list: return False
   
    with st.spinner("构建高速索引..."):
        GLOBAL_ADJ_FACTOR = pd.concat(adj_list)
        GLOBAL_ADJ_FACTOR['adj_factor'] = pd.to_numeric(GLOBAL_ADJ_FACTOR['adj_factor'], errors='coerce').fillna(0)
        GLOBAL_ADJ_FACTOR = GLOBAL_ADJ_FACTOR.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
        GLOBAL_DAILY_RAW = pd.concat(daily_list).drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
        latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
        if latest_global_date:
            try:
                latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
                GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
            except: GLOBAL_QFQ_BASE_FACTORS = {}
        try:
            with open(CACHE_FILE_NAME, 'wb') as f: pickle.dump({'adj': GLOBAL_ADJ_FACTOR, 'daily': GLOBAL_DAILY_RAW}, f)
        except: pass
    return True

def get_qfq_data(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    latest_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj): return pd.DataFrame() 
    try:
        df_d = GLOBAL_DAILY_RAW.loc[ts_code]
        df_d = df_d.loc[(df_d.index >= start_date) & (df_d.index <= end_date)]
        df_a = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        df_a = df_a.loc[(df_a.index >= start_date) & (df_a.index <= end_date)]
    except KeyError: return pd.DataFrame()
    if df_d.empty or df_a.empty: return pd.DataFrame()
    df = df_d.merge(df_a.rename('adj_factor'), left_index=True, right_index=True, how='left').dropna(subset=['adj_factor'])
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col] * df['adj_factor'] / latest_adj
    return df.reset_index().rename(columns={'trade_date': 'trade_date_str'}).sort_values('trade_date_str').set_index('trade_date_str')[['open', 'high', 'low', 'close', 'vol']].copy() 

# ---------------------------
# 买入与防守测算核心
# ---------------------------
def get_future_prices_with_defense(ts_code, selection_date, d0_close):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=30)).strftime("%Y%m%d") # 跨越长假取足数据
    
    hist = get_qfq_data(ts_code, start_future, end_future)
    if hist.empty: return None 
    
    next_open = hist.iloc[0]['open']
    
    # 防守网：次日开盘如果低于突破日收盘的-2%，直接判定为诱多，放弃！
    if next_open < d0_close * 0.98: return None 
    
    buy_price = next_open
    results = {
        'Buy_Price': round(buy_price, 2),
        'Return_D1 (%)': np.nan, 'Return_D3 (%)': np.nan, 
        'Return_D5 (%)': np.nan, 'Return_D10 (%)': np.nan, 
        'Max_Drawdown_3D (%)': np.nan
    }
    
    results['Return_D1 (%)'] = (hist.iloc[0]['close'] - buy_price) / buy_price * 100
    for n, key in zip([3, 5, 10], ['Return_D3 (%)', 'Return_D5 (%)', 'Return_D10 (%)']):
        if len(hist) >= n: results[key] = (hist.iloc[n-1]['close'] - buy_price) / buy_price * 100
    if len(hist) >= 3:
        results['Max_Drawdown_3D (%)'] = (hist.iloc[:3]['low'].min() - buy_price) / buy_price * 100
        
    return results

# ---------------------------
# 🌟 核心引擎：寻找“黄金坑的突破”
# ---------------------------
def run_golden_pit_backtest(last_trade, TOP_K, MIN_PRICE, MIN_MV, MAX_MV, WIN_RATE_LIMIT, VOL_MULTIPLIER, PIT_DAYS_MIN, PIT_DEPTH_MIN):
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), "数据缺失"

    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name')
    daily_basic = safe_get('daily_basic', trade_date=last_trade)
    df = daily_all.merge(stock_basic, on='ts_code', how='left')
    if not daily_basic.empty:
        df = df.merge(daily_basic[['ts_code','turnover_rate','circ_mv']], on='ts_code', how='left')
    if 'circ_mv' not in df.columns: return pd.DataFrame(), "市值缺失"
    
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')] 
    df = df[(df['close'] >= MIN_PRICE) & (df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    
    if len(df) == 0: return pd.DataFrame(), "无标的"

    chip_df = safe_get('cyq_perf', trade_date=last_trade)
    chip_dict = dict(zip(chip_df['ts_code'], chip_df['winner_rate'])) if not chip_df.empty else {}

    records = []
    lookback_start = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=250)).strftime("%Y%m%d")

    for row in df.itertuples():
        win_rate = chip_dict.get(row.ts_code, 0)
        if win_rate < WIN_RATE_LIMIT: continue 

        hist = get_qfq_data(row.ts_code, start_date=lookback_start, end_date=last_trade)
        if len(hist) < 125: continue 
        
        today_data = hist.iloc[-1]
        today_close = today_data['close']
        today_vol = today_data['vol']
        
        # 获取过去120日的数据（不含今天），用来寻找“坑沿”
        past_120 = hist.iloc[-121:-1]
        if past_120.empty: continue
        
        max_price = past_120['high'].max()
        max_date = past_120['high'].idxmax()
        max_idx = past_120.index.get_loc(max_date) 
        
        # 1. 坑沿时间防伪：最高点必须发生在 N 天以前（剔除每天爬坡创新高的股票）
        days_since_high = len(past_120) - 1 - max_idx 
        if days_since_high < PIT_DAYS_MIN: continue
            
        # 2. 坑底深度防伪：在最高点之后，必须砸出一个真坑
        pit_data = past_120.iloc[max_idx:]
        min_price = pit_data['low'].min()
        pit_depth = (max_price - min_price) / max_price * 100
        if pit_depth < PIT_DEPTH_MIN: continue

        # 3. 突破判定：今天放量打穿蓄谋已久的坑沿
        vol_ma5 = past_120['vol'].tail(5).mean()
        is_breakout = (today_close > max_price) and (today_vol > vol_ma5 * VOL_MULTIPLIER)
        
        if is_breakout:
            future = get_future_prices_with_defense(row.ts_code, last_trade, today_close)
            if future is None: continue # 次日防守放弃
            
            records.append({
                'ts_code': row.ts_code, 'name': row.name, 
                'Break_Close': round(today_close, 2), 
                'Pit_Lip(High)': round(max_price, 2),
                'Days_Since_High': days_since_high,
                'Pit_Depth(%)': round(pit_depth, 2),
                'WinRate(%)': round(win_rate, 2),
                'Buy_Price': future['Buy_Price'],
                'Return_D1 (%)': future['Return_D1 (%)'],
                'Return_D3 (%)': future['Return_D3 (%)'],
                'Return_D5 (%)': future['Return_D5 (%)'],
                'Return_D10 (%)': future['Return_D10 (%)'],
                'Max_Drawdown_3D (%)': future['Max_Drawdown_3D (%)']
            })

    if not records: return pd.DataFrame(), "无爆点标的"
    
    fdf = pd.DataFrame(records)
    # 评分规则：坑越深（洗盘越狠），沉淀越久，突破获利盘越高，评分越高
    fdf['Score'] = fdf['WinRate(%)'] * 5 + fdf['Pit_Depth(%)'] * 2 + fdf['Days_Since_High']
    final_df = fdf.sort_values('Score', ascending=False).head(TOP_K).copy()
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    
    return final_df, None

# ---------------------------
# UI 面板
# ---------------------------
with st.sidebar:
    st.header("V36 黄金坑主升版 🏆")
    backtest_date_end = st.date_input("回测截止日", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("测试交易日跨度", value=30, step=1)
    TOP_BACKTEST = st.number_input("每日最多入围数", value=5)
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 清空所有缓存"):
        if os.path.exists(CACHE_FILE_NAME): os.remove(CACHE_FILE_NAME)
        st.success("数据已清空。")
    CHECKPOINT_FILE = "backtest_checkpoint_V36.csv" 
    
    st.markdown("---")
    st.subheader("💰 资产底座")
    col1, col2 = st.columns(2)
    MIN_PRICE = col1.number_input("最低股价", value=10.0) 
    MIN_MV = col2.number_input("最小市值(亿)", value=30.0) 
    MAX_MV = st.number_input("最大市值(亿)", value=500.0)
    
    st.markdown("---")
    st.subheader("⚔️ 黄金坑形态定义")
    PIT_DAYS_MIN = st.number_input("坑沿至少在N天前", value=10, help="剔除天天创新高的爬坡股，最少洗盘2周")
    PIT_DEPTH_MIN = st.number_input("坑深至少跌幅(%)", value=15.0, help="跌幅必须够大才能洗掉散户")
    
    st.markdown("---")
    st.subheader("🚀 突破点判定")
    WIN_RATE_LIMIT = st.number_input("筹码极值获利盘(%)", value=90.0)
    VOL_MULTIPLIER = st.number_input("突破爆量倍数", value=1.5)

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🎯 启动黄金坑侦测"):
    processed_dates = set()
    results = []
    
    if RESUME_CHECKPOINT and os.path.exists(CHECKPOINT_FILE):
        try:
            existing_df = pd.read_csv(CHECKPOINT_FILE)
            existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
            processed_dates = set(existing_df['Trade_Date'].unique())
            results.append(existing_df)
            st.success(f"✅ 断点生效，跳过 {len(processed_dates)} 天...")
        except:
            if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    
    trade_days_list = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_list: st.stop()
         
    dates_to_run = [d for d in trade_days_list if d not in processed_dates]
    
    if not dates_to_run:
        st.success("🎉 指定范围全部测试完成！")
    else:
        if not get_all_historical_data(trade_days_list, use_cache=True): st.stop()
        bar = st.progress(0, text="黄金坑主升浪扫描启动...")
        
        for i, date in enumerate(dates_to_run):
            res, err = run_golden_pit_backtest(
                date, int(TOP_BACKTEST), MIN_PRICE, MIN_MV, MAX_MV, WIN_RATE_LIMIT, VOL_MULTIPLIER, PIT_DAYS_MIN, PIT_DEPTH_MIN
            )
            if not res.empty:
                res['Trade_Date'] = date
                is_first = not os.path.exists(CHECKPOINT_FILE)
                res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                results.append(res)
            bar.progress((i+1)/len(dates_to_run), text=f"量化检索: {date}")
        bar.empty()
    
    if results:
        all_res = pd.concat(results)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        all_res = all_res.sort_values(['Trade_Date', 'Rank'], ascending=[False, True])
        
        st.header(f"📊 黄金坑战法 统计结果 (共捕获 {len(all_res)} 次绝佳机会)")
        cols = st.columns(4)
        for idx, (n, title) in enumerate([(1, 'D1(验妖)'), (3, 'D3'), (5, 'D5'), (10, 'D10')]):
            col_name = f'Return_D{n} (%)'
            valid = all_res.dropna(subset=[col_name]) 
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                cols[idx].metric(f"{title} 均益/胜率", f"{avg:.2f}% / {win:.1f}%")
 
        st.subheader("📋 实战出击清单 (剔除了低开-2%的诱多股，并显示坑深与洗盘天数)")
        show_cols = ['Trade_Date','name','ts_code','Break_Close','Pit_Lip(High)','Days_Since_High','Pit_Depth(%)','WinRate(%)','Buy_Price',
                     'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)','Return_D10 (%)', 'Max_Drawdown_3D (%)']
        final_cols = [c for c in show_cols if c in all_res.columns]
        st.dataframe(all_res[final_cols], use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 导出清单到 Excel", csv, f"Golden_Pit_V36.csv", "text/csv")
    else:
        st.warning("⚠️ 市场目前没有走出完美的黄金坑突破形态。")
