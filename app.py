# -*- coding: utf-8 -*-
"""
周线 SKDJ 底部脱离定型版 (V4.2 终极防崩溃·时间轴校正版)
------------------------------------------------
1. 【核心修复】：为 trade_cal 及所有 Tushare 接口增加安全重试与异常包裹，杜绝崩溃白屏。
2. 【时间轴校正】：交易日历强制本地升序排序 (ascending=True)，彻底解决日期截断至 2025 年的 Bug。
3. 【数据防穿越】：严格禁止向 Tushare 请求未来日期，未发生的走势自动归为“持仓中”。
4. 【UI 持久化】：回测结果与导出下载模块完全独立，点击下载不重置页面。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import os
import pickle

warnings.filterwarnings("ignore")
CHECKPOINT_FILE = "skdj_robust_checkpoint.csv"
MARKET_CACHE_FILE = "skdj_market_data_v4.pkl"

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="SKDJ 底部突破实战版", layout="wide")
st.title("📈 周线 SKDJ 底部脱离右侧确认系统 (定型版)")
st.markdown("🔒 **系统引擎已全面加固：时间轴已校准，接口全异常拦截保护。**")

# ---------------------------
# 通用安全 API 请求封装
# ---------------------------
def safe_tushare_call(func, max_retries=3, sleep_time=1.0, **kwargs):
    """安全调用 Tushare 接口，防网络抖动、限流及异常崩溃"""
    for attempt in range(max_retries):
        try:
            df = func(**kwargs)
            if df is not None and not df.empty:
                return df
            time.sleep(sleep_time)
        except Exception as e:
            time.sleep(sleep_time * (attempt + 1))
    return pd.DataFrame()

# ---------------------------
# 硬科技白名单 (内存缓存)
# ---------------------------
@st.cache_data(ttl=3600*24*7, show_spinner=False) 
def load_industry_mapping(token):
    ts.set_token(token)
    pro = ts.pro_api()
    sw_indices = safe_tushare_call(pro.index_classify, level='L1', src='SW2021')
    if sw_indices.empty: return {}
    white_list_names = ['电子', '计算机', '通信', '医药生物', '国防军工', '机械设备']
    target_indices = sw_indices[sw_indices['industry_name'].isin(white_list_names)]
    index_codes = target_indices['index_code'].tolist()
    
    all_members = []
    load_bar = st.progress(0, text="正在加载硬科技白名单...")
    for i, idx_code in enumerate(index_codes):
        df = safe_tushare_call(pro.index_member, index_code=idx_code, is_new='Y')
        if not df.empty: 
            df['industry_code'] = idx_code
            all_members.append(df)
        time.sleep(0.05)
        load_bar.progress((i + 1) / len(index_codes))
    load_bar.empty()
    
    if not all_members: return {}
    full_df = pd.concat(all_members).drop_duplicates(subset=['con_code'])
    return dict(zip(full_df['con_code'], full_df['industry_code']))

# ---------------------------
# 增量下载引擎 (防未来穿越 + 节流存盘)
# ---------------------------
def sync_market_data_incrementally(start_date, end_date, token):
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 获取日历数据并本地过滤
    cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_date, end_date=end_date)
    if cal_raw.empty:
        return {'daily': [], 'adj': [], 'fetched_dates': set()}
        
    cal_open = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date', ascending=True)
    all_dates = cal_open['cal_date'].tolist()
    
    today_str = datetime.now().strftime("%Y%m%d")
    valid_dates = [d for d in all_dates if d <= today_str]
    
    cache = {'daily': [], 'adj': [], 'fetched_dates': set()}
    if os.path.exists(MARKET_CACHE_FILE):
        try:
            with open(MARKET_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            pass 
            
    missing_dates = [d for d in valid_dates if d not in cache['fetched_dates']]
    
    if missing_dates:
        my_bar = st.progress(0, text=f"📥 发现 {len(missing_dates)} 天缺失行情，启动增量引擎...")
        
        for i, d in enumerate(missing_dates):
            df_d = safe_tushare_call(pro.daily, max_retries=3, sleep_time=0.8, trade_date=d)
            df_a = safe_tushare_call(pro.adj_factor, max_retries=3, sleep_time=0.8, trade_date=d)
                
            if not df_d.empty and not df_a.empty:
                cache['daily'].append(df_d)
                cache['adj'].append(df_a)
                cache['fetched_dates'].add(d)
            
            if (i + 1) % 10 == 0 or i == len(missing_dates) - 1:
                my_bar.progress((i+1)/len(missing_dates), text=f"📥 行情同步中: {i+1}/{len(missing_dates)} (已存盘)")
                try:
                    with open(MARKET_CACHE_FILE + ".tmp", 'wb') as f:
                        pickle.dump(cache, f)
                    os.replace(MARKET_CACHE_FILE + ".tmp", MARKET_CACHE_FILE)
                except Exception:
                    pass
            
            time.sleep(0.25)
            
        my_bar.empty()
        
    return cache

@st.cache_data(ttl=3600*12, show_spinner=False)
def load_and_process_market_data(start_date, end_date, token, _dummy_trigger):
    cache = sync_market_data_incrementally(start_date, end_date, token)
    
    with st.spinner("正在构建全市场前复权多重索引..."):
        daily_raw = pd.concat(cache['daily']) if cache['daily'] else pd.DataFrame()
        adj_raw = pd.concat(cache['adj']) if cache['adj'] else pd.DataFrame()
        
        if not daily_raw.empty:
            daily_raw = daily_raw.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index()
        else:
            daily_raw = pd.DataFrame(columns=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']).set_index(['ts_code', 'trade_date'])
            
        if not adj_raw.empty:
            adj_raw = adj_raw.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index()
        else:
            adj_raw = pd.DataFrame(columns=['ts_code', 'trade_date', 'adj_factor']).set_index(['ts_code', 'trade_date'])
            
    return daily_raw, adj_raw

def get_qfq_data(ts_code, start_date, end_date, daily_raw, adj_raw):
    try:
        stock_daily = daily_raw.loc[ts_code].copy()
        stock_adj = adj_raw.loc[ts_code].copy()
    except KeyError:
        return pd.DataFrame()
        
    stock_daily = stock_daily[(stock_daily.index >= start_date) & (stock_daily.index <= end_date)]
    stock_adj = stock_adj[(stock_adj.index >= start_date) & (stock_adj.index <= end_date)]
    
    if stock_daily.empty or stock_adj.empty: return pd.DataFrame()
    
    df = stock_daily.merge(stock_adj[['adj_factor']], left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    if df.empty: return pd.DataFrame()
    
    latest_adj = df['adj_factor'].iloc[-1]
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            df[col] = df[col] * df['adj_factor'] / latest_adj
            
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df = df.sort_values('trade_date_str').set_index('trade_date_str')
    return df

# ---------------------------
# 核心引擎：突破 25 线算法
# ---------------------------
def compute_breakout_signal(ts_code, end_date, daily_raw, adj_raw):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=550)).strftime("%Y%m%d")
    df_daily = get_qfq_data(ts_code, start_date, end_date, daily_raw, adj_raw)
    res = {}
    if df_daily.empty or len(df_daily) < 100: return res

    df = df_daily.copy().reset_index()
    df['dt'] = pd.to_datetime(df['trade_date_str'])
    df['year_week'] = df['dt'].dt.strftime('%G_%V') 

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

    # --- 突破判定法则 ---
    is_breakout_25 = (curr_w['k'] > 25.0) and (prev_w['k'] <= 25.0)
    is_bullish = curr_w['k'] > curr_w['d']
    recent_d_min = weekly_df['d'].tail(10).min()
    has_bottom_gene = recent_d_min <= 20.0
    
    is_yang = curr_w['close'] > curr_w['open']
    is_vol_ok = True
    if pd.notna(curr_w['ma5_vol']) and curr_w['ma5_vol'] > 0:
        is_vol_ok = curr_w['vol'] >= curr_w['ma5_vol'] * 0.85

    res['is_buy_signal'] = is_breakout_25 and is_bullish and has_bottom_gene and is_yang and is_vol_ok
    res['k'] = round(curr_w['k'], 2)
    res['d'] = round(curr_w['d'], 2)
    res['recent_d_min'] = round(recent_d_min, 2)
    res['signal_close'] = curr_w['close'] 
    res['vol_ratio'] = round(curr_w['vol'] / curr_w['ma5_vol'], 2) if curr_w['ma5_vol'] > 0 else 1.0

    return res

# ---------------------------
# 次周一买入与防守出局系统
# ---------------------------
def track_future_performance(ts_code, selection_date, signal_close, daily_raw, adj_raw, hold_weeks=8):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_fetch = (d0 - timedelta(days=30)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=150)).strftime("%Y%m%d") 
    
    hist_full = get_qfq_data(ts_code, start_fetch, end_future, daily_raw, adj_raw)
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    results['Exit_Reason'] = "持仓中"
    results['Buy_Price'] = np.nan
    results['Gap_pct (%)'] = np.nan
    
    if hist_full.empty or len(hist_full) < 10: return results
    
    hist_future = hist_full[hist_full.index > selection_date]
    if hist_future.empty: return results

    next_row = hist_future.iloc[0]
    buy_price = next_row['open']
    if pd.isna(buy_price) or buy_price <= 0: return results

    is_20cm = any(ts_code.startswith(prefix) for prefix in ['300', '301', '688', '689'])
    
    if not is_20cm and next_row['open'] == next_row['high'] == next_row['low']:
        results['Exit_Reason'] = "一字板无法买入(剔除)"
        results['Buy_Price'] = round(buy_price, 2)  
        return results

    gap_pct = (buy_price - signal_close) / signal_close * 100
    results['Gap_pct (%)'] = round(gap_pct, 2)
    
    if is_20cm and gap_pct > 8.0:
        results['Exit_Reason'] = f"双创高开过大(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results
    elif not is_20cm and gap_pct > 5.0:
        results['Exit_Reason'] = f"主板高开过大(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results
        
    if gap_pct < -4.0:
        results['Exit_Reason'] = f"恶劣低开(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results

    results['Buy_Price'] = round(buy_price, 2)
    exit_triggered = False
    tier = 0  
    peak_close = buy_price
    pending_exit_reason = None  
    hard_stop_limit = -0.10 
    
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
        
        if (curr_low - buy_price) / buy_price <= hard_stop_limit:
            final_return = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
            exit_triggered = True
            results['Exit_Reason'] = "认栽出局(破-10%)"
            results[f'Return_W{current_week} (%)'] = final_return
            break
        
        if tier == 0 and peak_profit_pct >= 0.12: tier = 1
        if tier == 1:
            if curr_close < buy_price * 1.00:
                pending_exit_reason = "保本离场"
            elif peak_profit_pct >= 0.25: tier = 2
                
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
# UI 控制流与主循环
# ---------------------------
with st.sidebar:
    st.header("⚙️ 系统配置参数")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("追溯交易天数", value=100, step=20)
    TOP_BACKTEST = st.number_input("每周优选 TopK", value=3)
    
    st.markdown("---")
    if st.button("🗑️ 清空行情缓存 (重新下载)"):
        if os.path.exists(MARKET_CACHE_FILE):
            os.remove(MARKET_CACHE_FILE)
            st.success("底层行情缓存已清空！")
            
    if st.button("🗑️ 清除所有回测记录"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            st.success("回测历史记录已清理！")
            
    st.markdown("---")
    st.subheader("💰 护城河底座")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=100.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)

TS_TOKEN = st.text_input("🔑 Tushare Token", type="password")
if not TS_TOKEN: 
    st.info("👈 请在侧边栏填入 Token 激活程序")
    st.stop()

token_clean = TS_TOKEN.strip()

if st.button("🚀 启动周末定型回测"):
    pro = ts.pro_api(token_clean)
    
    # 扩大日历获取跨度，确保周线预热有充足基座
    lookback_days = max(int(BACKTEST_DAYS) * 3, 500) 
    start_cal = (datetime.strptime(backtest_date_end.strftime("%Y%m%d"), "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_cal = backtest_date_end.strftime("%Y%m%d")
    
    # 【核心修复】：安全获取日历，并在本地进行 is_open 过滤与强制升序排列
    cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_cal, end_date=end_cal)
    if cal_raw.empty:
        st.error("❌ 无法获取交易日历，请检查网络或 Token 积分是否充足。")
        st.stop()
        
    cal_open = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date', ascending=True)
    trade_days_list = cal_open['cal_date'].tolist()
    
    if not trade_days_list:
        st.error("❌ 未获取到有效的开市交易日。")
        st.stop()
        
    # 提取每周最后一个交易日
    td_df = pd.DataFrame({'cal_date': trade_days_list})
    td_df['dt'] = pd.to_datetime(td_df['cal_date'])
    td_df['year_week'] = td_df['dt'].dt.strftime('%G_%V')
    valid_scan_dates = set(td_df.groupby('year_week')['cal_date'].max().tolist())
    
    processed_dates = set()
    if os.path.exists(CHECKPOINT_FILE):
        try:
            existing_df = pd.read_csv(CHECKPOINT_FILE)
            existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
            processed_dates = set(existing_df['Trade_Date'].unique())
        except Exception:
            pass
            
    # 【核心修复】：由于列表严格升序，[-BACKTEST_DAYS:] 准确截取最近 N 个交易日
    recent_trade_days = trade_days_list[-int(BACKTEST_DAYS):]
    dates_to_run = [d for d in recent_trade_days if d not in processed_dates and d in valid_scan_dates]
    dates_to_run.sort()
    
    if not dates_to_run:
        st.success("🎉 指定区间的所有周末数据已回测完毕！查看下方榜单即可。")
    else:
        fetch_start = (datetime.strptime(min(dates_to_run), "%Y%m%d") - timedelta(days=550)).strftime("%Y%m%d")
        fetch_end = (datetime.strptime(max(dates_to_run), "%Y%m%d") + timedelta(days=150)).strftime("%Y%m%d")
        
        dummy_trigger = time.time()
        daily_raw, adj_raw = load_and_process_market_data(fetch_start, fetch_end, token_clean, dummy_trigger)
        
        if daily_raw.empty:
            st.warning("⚠️ 未能加载到有效行情数据，请点击左侧清空缓存后重试。")
            st.stop()
            
        stock_basic = safe_tushare_call(pro.stock_basic, list_status='L', fields='ts_code,name')
        stock_industry_map = load_industry_mapping(token_clean)
        
        bar = st.progress(0, text="执行周末精选引擎...")
        
        for i, date in enumerate(dates_to_run):
            try:
                daily_all = daily_raw.xs(date, level='trade_date').reset_index()
            except (KeyError, TypeError):
                bar.progress((i+1)/len(dates_to_run), text=f"跳过无数据日期: {date}")
                continue
                
            df = daily_all.merge(stock_basic, on='ts_code', how='inner')
            daily_basic = safe_tushare_call(pro.daily_basic, trade_date=date)
            
            if daily_basic.empty or 'circ_mv' not in daily_basic.columns:
                continue
                
            df = df.merge(daily_basic[['ts_code', 'circ_mv']], on='ts_code', how='inner')
            df['circ_mv_billion'] = df['circ_mv'] / 10000 
            
            df = df[~df['name'].str.contains('ST|退', na=False)]
            df = df[~df['ts_code'].str.startswith('92')] 
            df = df[(df['close'] >= MIN_PRICE)]
            df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
            
            records = []
            for row in df.itertuples():
                if stock_industry_map and row.ts_code not in stock_industry_map: continue
                    
                ind = compute_breakout_signal(row.ts_code, date, daily_raw, adj_raw)
                if not ind or not ind.get('is_buy_signal'): continue
                    
                score_k_break = (ind['k'] - 25.0) * 5.0
                score_vol = ind['vol_ratio'] * 10.0
                total_score = score_k_break + score_vol
                    
                future_returns = track_future_performance(row.ts_code, date, ind['signal_close'], daily_raw, adj_raw, hold_weeks=8)
                
                record_dict = {
                    'ts_code': row.ts_code, 'name': row.name, 'Signal_Close': ind['signal_close'], 
                    'SKDJ_K': ind['k'], 'SKDJ_D': ind['d'], 'D_Min(10W)': ind['recent_d_min'],
                    'circ_mv': round(row.circ_mv_billion, 2), 'Total_Score': round(total_score, 1)
                }
                record_dict.update(future_returns)
                records.append(record_dict)
                    
            if records:
                fdf = pd.DataFrame(records).sort_values('Total_Score', ascending=False).head(TOP_BACKTEST)
                fdf.insert(0, 'Rank', range(1, len(fdf) + 1))
                fdf['Trade_Date'] = date
                is_first = not os.path.exists(CHECKPOINT_FILE)
                fdf.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                
            bar.progress((i+1)/len(dates_to_run), text=f"扫描中: {date} (捕获 {len(records)} 只目标)")
            
        bar.empty()
        st.success("🎉 指定区间的行情已经扫描完毕！")

# ---------------------------
# 结果呈现模块 (独立作用域，保证导出下载不丢失视图)
# ---------------------------
if os.path.exists(CHECKPOINT_FILE):
    st.markdown("---")
    try:
        all_res = pd.read_csv(CHECKPOINT_FILE)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
        st.header("📊 定型版战绩追踪")
        st.subheader("🗓️ 周度胜率分布 (严格对齐周末信号)")
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
        st.download_button(
            label="📥 导出完整回测记录 (CSV)", 
            data=csv, 
            file_name="skdj_final_v4_2_export.csv", 
            mime="text/csv"
        )
    except pd.errors.EmptyDataError:
        st.info("🕒 当前暂无满足条件的回测记录。")
