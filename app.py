# -*- coding: utf-8 -*-
"""
周线 SKDJ 底部脱离定型版 (V3 终极防卡死·增量断点版)
------------------------------------------------
【核心修复】：
1. 采用“原子化增量落地”引擎：每下载 10 天数据自动存盘，卡死刷新后直接接续下载进度！
2. 加入 Tushare 限流节流阀，彻底解决因请求过快导致的前端卡死问题。
3. 完美保留：周末选股 + 周一开盘竞价买入 + 差异化竞价拦截。
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
MARKET_CACHE_FILE = "skdj_market_data_v3.pkl"

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="SKDJ 底部突破定型版", layout="wide")
st.title("📈 周线 SKDJ 底部脱离右侧确认系统 (稳健断点版)")
st.markdown("🔒 **增量引擎已启动：支持行情下载断点续传，彻底告别卡死与重复下载！**")

# ---------------------------
# 硬科技白名单 (内存缓存)
# ---------------------------
@st.cache_data(ttl=3600*24*7, show_spinner=False) 
def load_industry_mapping(token):
    ts.set_token(token)
    pro = ts.pro_api()
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

# ---------------------------
# 增量下载引擎 (解决卡死与重传的核心)
# ---------------------------
def sync_market_data_incrementally(start_date, end_date, token):
    ts.set_token(token)
    pro = ts.pro_api()
    cal = pro.trade_cal(start_date=start_date, end_date=end_date, is_open='1')
    all_dates = cal['cal_date'].tolist()
    
    # 1. 安全读取本地进度
    cache = {'daily': [], 'adj': [], 'fetched_dates': set()}
    if os.path.exists(MARKET_CACHE_FILE):
        try:
            with open(MARKET_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            pass # 如果文件损坏，自动重新开始
            
    # 2. 计算需要补充下载的日期
    missing_dates = [d for d in all_dates if d not in cache['fetched_dates']]
    
    if missing_dates:
        my_bar = st.progress(0, text=f"📥 发现 {len(missing_dates)} 天缺失数据，启动断点续传...")
        
        for i, d in enumerate(missing_dates):
            # 获取日线
            df_d = pd.DataFrame()
            for _ in range(3):
                try:
                    df_d = pro.daily(trade_date=d)
                    break
                except Exception: time.sleep(0.5)
            
            # 获取复权因子
            df_a = pd.DataFrame()
            for _ in range(3):
                try:
                    df_a = pro.adj_factor(trade_date=d)
                    break
                except Exception: time.sleep(0.5)
                
            if not df_d.empty: cache['daily'].append(df_d)
            if not df_a.empty: cache['adj'].append(df_a)
            cache['fetched_dates'].add(d)
            
            # 【核心安全机制】：每 10 天或结束时，安全存盘一次
            if (i + 1) % 10 == 0 or i == len(missing_dates) - 1:
                my_bar.progress((i+1)/len(missing_dates), text=f"📥 增量下载中: {i+1}/{len(missing_dates)} (进度已存盘)")
                try:
                    # 原子写入：先写入临时文件，再替换，防止写入中途卡死导致文件损坏
                    with open(MARKET_CACHE_FILE + ".tmp", 'wb') as f:
                        pickle.dump(cache, f)
                    os.replace(MARKET_CACHE_FILE + ".tmp", MARKET_CACHE_FILE)
                except Exception:
                    pass
            
            # 【节流阀】：强制休眠，防止 Tushare 限流卡死前端
            time.sleep(0.2) 
            
        my_bar.empty()
        
    return cache

# 使用内存加速，避免每次交互都去合并巨大的 DataFrame
@st.cache_data(ttl=3600*12, show_spinner=False)
def load_and_process_market_data(start_date, end_date, token, _dummy_trigger):
    # 先运行增量落地引擎确保数据完整
    cache = sync_market_data_incrementally(start_date, end_date, token)
    
    with st.spinner("正在构建高速索引..."):
        daily_raw = pd.concat(cache['daily']) if cache['daily'] else pd.DataFrame()
        adj_raw = pd.concat(cache['adj']) if cache['adj'] else pd.DataFrame()
        
        if not daily_raw.empty:
            daily_raw = daily_raw.set_index(['ts_code', 'trade_date']).sort_index()
        if not adj_raw.empty:
            adj_raw = adj_raw.set_index(['ts_code', 'trade_date']).sort_index()
            
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
# 核心引擎：上穿 25 线突破判断
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

    # --- 突破核心 ---
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
    BACKTEST_DAYS = st.number_input("追溯自然天数", value=180, step=30)
    TOP_BACKTEST = st.number_input("每周优选 TopK", value=3)
    
    st.markdown("---")
    if st.button("🗑️ 清理行情重新下载"):
        if os.path.exists(MARKET_CACHE_FILE):
            os.remove(MARKET_CACHE_FILE)
            st.success("底层数据已清空，下次运行将重新拉取！")
            
    if st.button("🗑️ 清除断点重新回测"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            st.success("回测进度已清理！")
            
    st.markdown("---")
    st.subheader("💰 护城河底座")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=100.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)

TS_TOKEN = st.text_input("🔑 Tushare Token", type="password")
if not TS_TOKEN: 
    st.info("👈 请填入 Token 激活程序")
    st.stop()

if st.button("🚀 启动周末定型回测"):
    pro = ts.pro_api(TS_TOKEN)
    
    lookback_days = max(int(BACKTEST_DAYS) * 3, 365) 
    start_cal = (datetime.strptime(backtest_date_end.strftime("%Y%m%d"), "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal_df = pro.trade_cal(start_date=start_cal, end_date=backtest_date_end.strftime("%Y%m%d"), is_open='1')
    trade_days_list = cal_df['cal_date'].tolist()
    if not trade_days_list: st.stop()
    
    # 智能提取每周最后一个交易日
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
        except: pass
            
    dates_to_run = [d for d in trade_days_list[-int(BACKTEST_DAYS):] if d not in processed_dates and d in valid_scan_dates]
    dates_to_run.sort()
    
    if not dates_to_run:
        st.success("🎉 指定区间的数据已回测完毕！查看下方榜单即可。")
    else:
        # 1. 触发增量下载引擎获取全量数据 (带进度保存与节流)
        fetch_start = (datetime.strptime(min(dates_to_run), "%Y%m%d") - timedelta(days=550)).strftime("%Y%m%d")
        fetch_end = (datetime.strptime(max(dates_to_run), "%Y%m%d") + timedelta(days=150)).strftime("%Y%m%d")
        
        # 使用虚拟触发器确保每次点击都能刷新内存
        dummy_trigger = time.time()
        daily_raw, adj_raw = load_and_process_market_data(fetch_start, fetch_end, TS_TOKEN, dummy_trigger)
        stock_industry_map = load_industry_mapping(TS_TOKEN)
    
        bar = st.progress(0, text="执行周末精筛引擎...")
        results = []
        
        for i, date in enumerate(dates_to_run):
            try:
                daily_all = daily_raw.xs(date, level='trade_date').reset_index()
            except KeyError:
                bar.progress((i+1)/len(dates_to_run), text=f"扫描中: {date}")
                continue
                
            stock_basic = pro.stock_basic(list_status='L', fields='ts_code,name')
            df = daily_all.merge(stock_basic, on='ts_code', how='inner')
            
            # 使用 try-except 处理单日 daily_basic 可能的缺失
            for _ in range(3):
                try:
                    daily_basic = pro.daily_basic(trade_date=date)[['ts_code','circ_mv']]
                    break
                except Exception: time.sleep(0.5)
                
            if daily_basic.empty: continue
            
            df = df.merge(daily_basic, on='ts_code', how='inner')
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
                results.append(fdf)
                
            bar.progress((i+1)/len(dates_to_run), text=f"回测中: {date} (找到 {len(records)} 只符合)")
            
        bar.empty()
        st.success("🎉 本次扫描已顺利完成！")
    
    # 结果呈现
    if os.path.exists(CHECKPOINT_FILE):
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
        st.download_button("📥 导出完整回测记录 (CSV)", csv, "skdj_robust_export.csv", "text/csv")
