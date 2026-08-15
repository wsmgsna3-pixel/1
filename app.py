# -*- coding: utf-8 -*-
"""
周线 SKDJ 底部脱离定型版 (V2 稳健重构版)
------------------------------------------------
1. 修复了 Pickle 本地缓存导致内存溢出和文件损坏崩溃的致命 Bug。
2. 移除了 Streamlit 不兼容的 ThreadPool 多线程组件，采用原生极速缓存。
3. 严格执行：周末定格周线 K、D 指标，下周一早盘集合竞价执行买点。
4. 竞价拦截：双创板高开>8%剔除，主板高开>5%剔除，核按钮<-4%剔除。
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

warnings.filterwarnings("ignore")
CHECKPOINT_FILE = "skdj_robust_checkpoint.csv"

# ---------------------------
# 页面配置 (必须是第一句)
# ---------------------------
st.set_page_config(page_title="SKDJ 底部突破定型版", layout="wide")
st.title("📈 周线 SKDJ 底部脱离右侧确认系统 (稳健运行版)")
st.markdown("🔒 **回测引擎已加锁：严格过滤周中假信号，100% 模拟周末选股 + 周一开盘竞价买入。**")

# ---------------------------
# 极速数据获取引擎 (抛弃全局变量，使用 Streamlit 原生缓存)
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

@st.cache_data(ttl=3600*12, show_spinner=False)
def load_market_history(start_date, end_date, token):
    ts.set_token(token)
    pro = ts.pro_api()
    cal = pro.trade_cal(start_date=start_date, end_date=end_date, is_open='1')
    dates = cal['cal_date'].tolist()
    
    daily_list, adj_list = [], []
    my_bar = st.progress(0, text="📥 正在全量同步市场历史数据 (首次运行约需 3 分钟，请耐心等待)...")
    
    for i, d in enumerate(dates):
        # 仅在网络异常时重试，空数据不重试以防死循环
        for _ in range(3):
            try:
                df_d = pro.daily(trade_date=d)
                if not df_d.empty: daily_list.append(df_d)
                break
            except Exception: time.sleep(0.5)
            
        for _ in range(3):
            try:
                df_a = pro.adj_factor(trade_date=d)
                if not df_a.empty: adj_list.append(df_a)
                break
            except Exception: time.sleep(0.5)
            
        if i % 10 == 0 or i == len(dates) - 1:
            my_bar.progress((i+1)/len(dates), text=f"📥 数据同步中: {i+1}/{len(dates)}")
            
    my_bar.empty()
    daily_raw = pd.concat(daily_list) if daily_list else pd.DataFrame()
    adj_raw = pd.concat(adj_list) if adj_list else pd.DataFrame()
    
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
    # 采用更兼容的 ISO 日历提取方式，防止 pandas 版本报错
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

    # --- 核心买入法则 ---
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
    
    # 截取周末选股日之后的行情，第一条即为次周一的数据
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
    
    # 【核心逻辑】：针对个股本身的板块属性与跳空幅度进行独立拦截
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

    # 顺利按次周一开盘价成交
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
        
        # 1. 认栽出局
        if (curr_low - buy_price) / buy_price <= hard_stop_limit:
            final_return = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
            exit_triggered = True
            results['Exit_Reason'] = "认栽出局(破-10%)"
            results[f'Return_W{current_week} (%)'] = final_return
            break
        
        # 2. 保本控制机制：超过 +12% 后保本
        if tier == 0 and peak_profit_pct >= 0.12: tier = 1
        if tier == 1:
            if curr_close < buy_price * 1.00:
                pending_exit_reason = "保本离场"
            elif peak_profit_pct >= 0.25: tier = 2
                
        # 3. 移动止盈机制：最高点回撤 15%
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
    if st.button("🗑️ 清除断点重新回测"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            st.success("进度已清理，可重新测试！")
            
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

if st.button("🚀 启动 SKDJ 突破 25 定型回测"):
    pro = ts.pro_api(TS_TOKEN)
    
    # 获取交易日历
    lookback_days = max(int(BACKTEST_DAYS) * 3, 365) 
    start_cal = (datetime.strptime(backtest_date_end.strftime("%Y%m%d"), "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal_df = pro.trade_cal(start_date=start_cal, end_date=backtest_date_end.strftime("%Y%m%d"), is_open='1')
    trade_days_list = cal_df['cal_date'].tolist()
    if not trade_days_list: st.stop()
    
    # 智能提取包含该区间的每周最后一个交易日 (只处理周五)
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
        st.success("🎉 指定区间的数据已拉取完毕！")
        st.stop()
        
    # 一次性获取全量数据，彻底解决单日循环请求导致的卡顿
    fetch_start = (datetime.strptime(min(dates_to_run), "%Y%m%d") - timedelta(days=550)).strftime("%Y%m%d")
    fetch_end = (datetime.strptime(max(dates_to_run), "%Y%m%d") + timedelta(days=150)).strftime("%Y%m%d")
    daily_raw, adj_raw = load_market_history(fetch_start, fetch_end, TS_TOKEN)
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
        daily_basic = pro.daily_basic(trade_date=date)[['ts_code','circ_mv']]
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
            
        bar.progress((i+1)/len(dates_to_run), text=f"扫描中: {date} (找到 {len(records)} 只符合)")
        
    bar.empty()
    st.success("🎉 本次扫描已顺利完成！")
    
    # 结果呈现
    if os.path.exists(CHECKPOINT_FILE):
        all_res = pd.read_csv(CHECKPOINT_FILE)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
        st.header("📊 定型版战绩追踪")
        st.subheader("🗓️ 周度胜率分布 (已锁定周末数据计算)")
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
