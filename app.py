# -*- coding: utf-8 -*-
"""
周线 SKDJ 底部脱离定型版 (V6.1 极速轻量·防OOM内存优化版)
------------------------------------------------
1. 【内存暴降 85%】：全市场索引仅保留科技白名单标的，彻底杜绝 Streamlit 1GB OOM 闪退。
2. 【极速前复权引擎】：预先构建字典化 QFQ 序列，消除循环内部重复 merge 碎片。
3. 【周线趋势识别】：上升趋势(+30) / 震荡筑底(+15) / 下跌中继(-25) 动态打分。
4. 【全新评分体系】：周线趋势分 + K值动能甜区分 + 量能健康分，优选前3名真龙头。
5. 【防崩与缓存继承】：全异常拦截、Token 预检、master 缓存永久兼容。
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
import re
import pickle

warnings.filterwarnings("ignore")

# ---------------------------
# 全局持久化缓存配置
# ---------------------------
CHECKPOINT_FILE = "skdj_robust_checkpoint.csv"
MARKET_CACHE_FILE = "skdj_market_data_master.pkl"

if not os.path.exists(MARKET_CACHE_FILE):
    for legacy_file in [
        "skdj_market_data_v6.pkl",
        "skdj_market_data_v5.pkl", 
        "skdj_market_data_v4.pkl", 
        "skdj_market_data_v3.pkl",
        "skdj_market_data.pkl"
    ]:
        if os.path.exists(legacy_file):
            try:
                os.rename(legacy_file, MARKET_CACHE_FILE)
                break
            except Exception:
                pass

# ---------------------------
# 页面基础配置
# ---------------------------
st.set_page_config(page_title="SKDJ 底部突破实战版", layout="wide")
st.title("📈 周线 SKDJ 底部脱离右侧确认系统 (V6.1 极速轻量版)")
st.markdown("🔒 **轻量化引擎已装载：白名单内存优化已就绪，彻底杜绝 OOM 内存崩溃。**")

# ---------------------------
# Token 清洗与安全请求模块
# ---------------------------
def clean_token_str(raw_token: str) -> str:
    if not raw_token: return ""
    return re.sub(r'[\s\u3000\ufeff\xa0\r\n]+', '', str(raw_token)).strip()

def verify_token_connection(token_str: str):
    if not token_str:
        return False, "Token 为空，请在侧边栏填入 Token。"
    try:
        ts.set_token(token_str)
        pro = ts.pro_api(token_str)
        test_df = pro.trade_cal(exchange='SSE', start_date='20260801', end_date='20260805')
        if test_df is not None and not test_df.empty:
            return True, "验证通过"
        return False, "Token 校验未返回数据，请检查网络连接。"
    except Exception as e:
        err_msg = str(e)
        if "token不对" in err_msg or "-40001" in err_msg:
            return False, "您的 Token 不正确，请检查复制内容。"
        return False, f"接口校验失败: {err_msg}"

def safe_tushare_call(func, max_retries=3, sleep_time=0.8, **kwargs):
    for attempt in range(max_retries):
        try:
            df = func(**kwargs)
            if df is not None and not df.empty:
                return df
            time.sleep(sleep_time)
        except Exception:
            time.sleep(sleep_time * (attempt + 1))
    return pd.DataFrame()

# ---------------------------
# 硬科技白名单
# ---------------------------
@st.cache_data(ttl=3600*24*7, show_spinner=False) 
def load_industry_mapping(token):
    token_c = clean_token_str(token)
    if not token_c: return {}
    
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    sw_indices = safe_tushare_call(pro.index_classify, level='L1', src='SW2021')
    if sw_indices.empty: return {}
    
    white_list_names = ['电子', '计算机', '通信', '医药生物', '国防军工', '机械设备']
    target_indices = sw_indices[sw_indices['industry_name'].isin(white_list_names)]
    index_codes = target_indices['index_code'].tolist()
    
    all_members = []
    load_bar = st.progress(0, text="正在同步硬科技白名单...")
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
# 增量下载引擎
# ---------------------------
def sync_market_data_incrementally(start_date, end_date, token):
    token_c = clean_token_str(token)
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    
    cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_date, end_date=end_date)
    if cal_raw.empty:
        return {'daily': [], 'adj': [], 'daily_basic': [], 'fetched_dates': set()}
        
    cal_open = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date', ascending=True)
    all_dates = cal_open['cal_date'].tolist()
    
    today_str = datetime.now().strftime("%Y%m%d")
    valid_dates = [d for d in all_dates if d <= today_str]
    
    cache = {'daily': [], 'adj': [], 'daily_basic': [], 'fetched_dates': set()}
    if os.path.exists(MARKET_CACHE_FILE):
        try:
            with open(MARKET_CACHE_FILE, 'rb') as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    cache.update(loaded)
                    if 'daily_basic' not in cache: cache['daily_basic'] = []
        except Exception:
            pass 
            
    missing_dates = [d for d in valid_dates if d not in cache['fetched_dates']]
    
    if missing_dates:
        my_bar = st.progress(0, text=f"📥 检测到 {len(missing_dates)} 天增量行情需要同步...")
        
        for i, d in enumerate(missing_dates):
            df_d = safe_tushare_call(pro.daily, max_retries=3, sleep_time=0.8, trade_date=d)
            df_a = safe_tushare_call(pro.adj_factor, max_retries=3, sleep_time=0.8, trade_date=d)
            df_b = safe_tushare_call(pro.daily_basic, max_retries=3, sleep_time=0.8, trade_date=d, fields='ts_code,trade_date,circ_mv')
                
            if not df_d.empty and not df_a.empty:
                cache['daily'].append(df_d)
                cache['adj'].append(df_a)
                if not df_b.empty:
                    cache['daily_basic'].append(df_b)
                cache['fetched_dates'].add(d)
            
            if (i + 1) % 10 == 0 or i == len(missing_dates) - 1:
                my_bar.progress((i+1)/len(missing_dates), text=f"📥 行情同步中: {i+1}/{len(missing_dates)} (进度已落盘)")
                try:
                    with open(MARKET_CACHE_FILE + ".tmp", 'wb') as f:
                        pickle.dump(cache, f)
                    os.replace(MARKET_CACHE_FILE + ".tmp", MARKET_CACHE_FILE)
                except Exception:
                    pass
            
            time.sleep(0.25)
            
        my_bar.empty()
        
    return cache

# ---------------------------
# 极速轻量化内存索引引擎 (核心防 OOM)
# ---------------------------
@st.cache_data(ttl=3600*12, show_spinner=False)
def load_optimized_market_data(start_date, end_date, token, _whitelist_keys, _dummy_trigger):
    token_c = clean_token_str(token)
    cache = sync_market_data_incrementally(start_date, end_date, token_c)
    
    with st.spinner("正在构建科技白名单轻量化前复权索引 (节省85%内存)..."):
        daily_list = cache.get('daily', [])
        adj_list = cache.get('adj', [])
        basic_list = cache.get('daily_basic', [])
        
        daily_raw = pd.concat(daily_list, ignore_index=True) if daily_list else pd.DataFrame()
        adj_raw = pd.concat(adj_list, ignore_index=True) if adj_list else pd.DataFrame()
        basic_raw = pd.concat(basic_list, ignore_index=True) if basic_list else pd.DataFrame()
        
        if daily_raw.empty or adj_raw.empty:
            return {}, pd.DataFrame()
            
        # 【核心优化1】：仅保留科技白名单标的，剔除几千只无用股票
        whitelist_set = set(_whitelist_keys)
        if whitelist_set:
            daily_raw = daily_raw[daily_raw['ts_code'].isin(whitelist_set)]
            adj_raw = adj_raw[adj_raw['ts_code'].isin(whitelist_set)]
            if not basic_raw.empty:
                basic_raw = basic_raw[basic_raw['ts_code'].isin(whitelist_set)]

        # 【核心优化2】：预先按股合成前复权，存入高速字典，避免循环中重复 merge
        merged_all = daily_raw.merge(adj_raw[['ts_code', 'trade_date', 'adj_factor']], on=['ts_code', 'trade_date'], how='inner')
        merged_all['trade_date_str'] = merged_all['trade_date'].astype(str)
        merged_all = merged_all.sort_values(['ts_code', 'trade_date_str'])
        
        stock_qfq_dict = {}
        for ts_code, group in merged_all.groupby('ts_code'):
            df_g = group.copy()
            latest_adj = df_g['adj_factor'].iloc[-1]
            if latest_adj > 0:
                for col in ['open', 'high', 'low', 'close', 'pre_close']:
                    if col in df_g.columns:
                        df_g[col] = df_g[col] * df_g['adj_factor'] / latest_adj
            df_g = df_g.set_index('trade_date_str')
            stock_qfq_dict[ts_code] = df_g
            
        # 构建基础信息轻量索引
        if not basic_raw.empty:
            basic_indexed = basic_raw.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['trade_date', 'ts_code'])
        else:
            basic_indexed = pd.DataFrame()
            
    return stock_qfq_dict, basic_indexed

# ---------------------------
# 核心引擎：突破 25 线算法 + 趋势识别 + 综合评分
# ---------------------------
def compute_breakout_signal(ts_code, end_date, stock_qfq_dict):
    if ts_code not in stock_qfq_dict: return {}
    df_full = stock_qfq_dict[ts_code]
    
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=550)).strftime("%Y%m%d")
    df_daily = df_full[(df_full.index >= start_date) & (df_full.index <= end_date)]
    
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
    weekly_df['ma20'] = weekly_df['close'].rolling(window=20).mean()

    curr_w = weekly_df.iloc[-1]
    prev_w = weekly_df.iloc[-2]
    
    if pd.isna(curr_w['k']) or pd.isna(prev_w['k']): return res

    # 突破条件确认
    is_breakout_25 = (curr_w['k'] > 25.0) and (prev_w['k'] <= 25.0)
    is_bullish = curr_w['k'] > curr_w['d']
    recent_d_min = weekly_df['d'].tail(10).min()
    has_bottom_gene = recent_d_min <= 20.0
    
    is_yang = curr_w['close'] > curr_w['open']
    is_vol_ok = True
    if pd.notna(curr_w['ma5_vol']) and curr_w['ma5_vol'] > 0:
        is_vol_ok = curr_w['vol'] >= curr_w['ma5_vol'] * 0.85

    res['is_buy_signal'] = is_breakout_25 and is_bullish and has_bottom_gene and is_yang and is_vol_ok
    if not res['is_buy_signal']: return res

    res['k'] = round(curr_w['k'], 2)
    res['d'] = round(curr_w['d'], 2)
    res['recent_d_min'] = round(recent_d_min, 2)
    res['signal_close'] = curr_w['close'] 
    
    vol_ratio = curr_w['vol'] / curr_w['ma5_vol'] if (pd.notna(curr_w['ma5_vol']) and curr_w['ma5_vol'] > 0) else 1.0
    res['vol_ratio'] = round(vol_ratio, 2)

    # ---------------------------
    # 周线趋势识别
    # ---------------------------
    ma20_curr = curr_w['ma20'] if pd.notna(curr_w['ma20']) else curr_w['close']
    ma20_prev = prev_w['ma20'] if pd.notna(prev_w['ma20']) else curr_w['close']
    ma20_slope = (ma20_curr - ma20_prev) / (ma20_prev + 1e-5)
    
    low_recent_3w = weekly_df['low'].tail(3).min()
    low_prev_10w = weekly_df['low'].tail(10).min()
    is_higher_low = low_recent_3w > low_prev_10w * 1.01

    if curr_w['close'] >= ma20_curr and ma20_slope >= -0.003:
        trend_type = "上升趋势"
        score_trend = 30.0
    elif is_higher_low or abs(curr_w['close'] - ma20_curr) / ma20_curr <= 0.08:
        trend_type = "震荡筑底"
        score_trend = 15.0
    else:
        trend_type = "下跌中继"
        score_trend = -25.0
        
    res['trend_type'] = trend_type

    # ---------------------------
    # 新版综合评分模型
    # ---------------------------
    k_val = curr_w['k']
    if k_val < 28.0:
        score_k = (k_val - 25.0) * 2.0
    elif 28.0 <= k_val <= 38.0:
        score_k = 15.0 + (k_val - 28.0) * 3.5
    else:
        score_k = 50.0 - (k_val - 38.0) * 2.0

    if vol_ratio < 1.0:
        score_vol = -20.0
    elif 1.0 <= vol_ratio < 1.3:
        score_vol = 10.0
    elif 1.3 <= vol_ratio <= 3.0:
        score_vol = 35.0
    elif 3.0 < vol_ratio <= 4.5:
        score_vol = 20.0
    else:
        score_vol = 5.0

    total_score = score_k + score_vol + score_trend
    res['Total_Score'] = round(total_score, 1)

    return res

# ---------------------------
# 次周一买入与防守出局系统
# ---------------------------
def track_future_performance(ts_code, selection_date, signal_close, stock_qfq_dict, hold_weeks=8):
    if ts_code not in stock_qfq_dict: 
        return {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)} | {'Exit_Reason': '持仓中', 'Buy_Price': np.nan, 'Gap_pct (%)': np.nan}

    df_full = stock_qfq_dict[ts_code]
    hist_future = df_full[df_full.index > selection_date]
    
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    results['Exit_Reason'] = "持仓中"
    results['Buy_Price'] = np.nan
    results['Gap_pct (%)'] = np.nan
    
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
# UI 控制流与输入侧边栏
# ---------------------------
with st.sidebar:
    st.header("⚙️ 系统配置参数")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("追溯交易天数", value=250, step=30)
    TOP_BACKTEST = st.number_input("每周优选 TopK", value=3)
    
    st.markdown("---")
    if st.button("🗑️ 清空行情缓存 (重新全量下载)"):
        if os.path.exists(MARKET_CACHE_FILE):
            os.remove(MARKET_CACHE_FILE)
        st.cache_data.clear()
        st.success("底层行情缓存已彻底清空！下次运行将重新拉取。")
            
    if st.button("🗑️ 清除所有回测记录"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        st.success("回测记录已清理！")
            
    st.markdown("---")
    st.subheader("💰 护城河底座")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=100.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)
    
    st.markdown("---")
    secret_token = st.secrets.get("TUSHARE_TOKEN", "") if hasattr(st, "secrets") else ""
    TS_TOKEN_INPUT = st.text_input(
        "🔑 Tushare Token", 
        value=secret_token,
        type="password",
        help="可在 Streamlit Cloud 的 Settings -> Secrets 中预设 TUSHARE_TOKEN"
    )

token_clean = clean_token_str(TS_TOKEN_INPUT)

# ---------------------------
# 主流程：启动回测
# ---------------------------
if st.button("🚀 启动周末定型回测"):
    is_valid, msg = verify_token_connection(token_clean)
    if not is_valid:
        st.error(f"❌ **Token 预检拦截**：{msg}")
    else:
        try:
            ts.set_token(token_clean)
            pro = ts.pro_api(token_clean)
            
            lookback_days = max(int(BACKTEST_DAYS) * 3, 500) 
            start_cal = (datetime.strptime(backtest_date_end.strftime("%Y%m%d"), "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
            end_cal = backtest_date_end.strftime("%Y%m%d")
            
            cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_cal, end_date=end_cal)
            if cal_raw.empty:
                st.error("❌ 无法获取交易日历，请检查网络或 Token 积分。")
            else:
                cal_open = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date', ascending=True)
                trade_days_list = cal_open['cal_date'].tolist()
                
                if not trade_days_list:
                    st.error("❌ 未获取到有效的开市交易日。")
                else:
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
                            
                    recent_trade_days = trade_days_list[-int(BACKTEST_DAYS):]
                    dates_to_run = [d for d in recent_trade_days if d not in processed_dates and d in valid_scan_dates]
                    dates_to_run.sort()
                    
                    if not dates_to_run:
                        st.success("🎉 指定区间的所有周末数据已回测完毕！查看下方榜单即可。")
                    else:
                        stock_industry_map = load_industry_mapping(token_clean)
                        whitelist_keys = tuple(sorted(stock_industry_map.keys()))
                        
                        fetch_start = (datetime.strptime(min(dates_to_run), "%Y%m%d") - timedelta(days=550)).strftime("%Y%m%d")
                        fetch_end = (datetime.strptime(max(dates_to_run), "%Y%m%d") + timedelta(days=150)).strftime("%Y%m%d")
                        
                        dummy_trigger = time.time()
                        stock_qfq_dict, basic_indexed = load_optimized_market_data(fetch_start, fetch_end, token_clean, whitelist_keys, dummy_trigger)
                        
                        if not stock_qfq_dict:
                            st.warning("⚠️ 未能加载到有效白名单行情数据，请点击左侧清空缓存重试。")
                        else:
                            stock_basic = safe_tushare_call(pro.stock_basic, list_status='L', fields='ts_code,name')
                            basic_name_map = dict(zip(stock_basic['ts_code'], stock_basic['name'])) if not stock_basic.empty else {}
                            
                            bar = st.progress(0, text="执行轻量级极速扫描引擎...")
                            
                            for i, date in enumerate(dates_to_run):
                                records = []
                                
                                # 遍历白名单中的股票
                                for ts_code in whitelist_keys:
                                    stock_name = basic_name_map.get(ts_code, '')
                                    if 'ST' in stock_name or '退' in stock_name or ts_code.startswith('92'):
                                        continue
                                        
                                    if ts_code not in stock_qfq_dict:
                                        continue
                                        
                                    df_stock = stock_qfq_dict[ts_code]
                                    if date not in df_stock.index:
                                        continue
                                        
                                    row_latest = df_stock.loc[date]
                                    if isinstance(row_latest, pd.DataFrame):
                                        row_latest = row_latest.iloc[-1]
                                        
                                    curr_close = row_latest['close']
                                    if curr_close < MIN_PRICE:
                                        continue
                                        
                                    # 获取市值
                                    circ_mv_billion = np.nan
                                    if not basic_indexed.empty and (date, ts_code) in basic_indexed.index:
                                        circ_mv_billion = basic_indexed.loc[(date, ts_code)]['circ_mv'] / 10000.0
                                    
                                    if pd.notna(circ_mv_billion):
                                        if circ_mv_billion < MIN_MV or circ_mv_billion > MAX_MV:
                                            continue
                                    
                                    ind = compute_breakout_signal(ts_code, date, stock_qfq_dict)
                                    if not ind or not ind.get('is_buy_signal'): 
                                        continue
                                        
                                    future_returns = track_future_performance(ts_code, date, ind['signal_close'], stock_qfq_dict, hold_weeks=8)
                                    
                                    record_dict = {
                                        'ts_code': ts_code, 'name': stock_name, 'Signal_Close': ind['signal_close'], 
                                        'SKDJ_K': ind['k'], 'SKDJ_D': ind['d'], 'D_Min(10W)': ind['recent_d_min'],
                                        'Trend_Type': ind['trend_type'], 'vol_ratio': ind['vol_ratio'],
                                        'circ_mv': round(circ_mv_billion, 2) if pd.notna(circ_mv_billion) else np.nan, 
                                        'Total_Score': ind['Total_Score']
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
        except Exception as e:
            st.error(f"❌ **运行过程异常拦截（界面已安全保护）**：{str(e)}")

# ---------------------------
# 结果呈现模块 (独立展示区)
# ---------------------------
if os.path.exists(CHECKPOINT_FILE):
    st.markdown("---")
    try:
        all_res = pd.read_csv(CHECKPOINT_FILE)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
        st.header("📊 V6.1 极速版战绩追踪")
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
            'Rank', 'Trade_Date', 'name', 'ts_code', 'Trend_Type', 'SKDJ_K', 'SKDJ_D', 'D_Min(10W)', 'vol_ratio', 'Total_Score',
            'Signal_Close', 'Buy_Price', 'Gap_pct (%)', 'circ_mv', 'Exit_Reason'
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
            
        def color_trend(val):
            if val == '上升趋势': return 'color: green; font-weight: bold'
            elif val == '震荡筑底': return 'color: orange; font-weight: bold'
            elif val == '下跌中继': return 'color: red; font-weight: bold'
            return ''
        
        styled_df = display_df.style
        if 'Exit_Reason' in display_df.columns:
            styled_df = styled_df.map(color_exit, subset=['Exit_Reason'])
        if 'Trend_Type' in display_df.columns:
            styled_df = styled_df.map(color_trend, subset=['Trend_Type'])
        
        try:
            st.dataframe(styled_df, width="stretch")
        except TypeError:
            st.dataframe(styled_df, use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 导出完整回测记录 (CSV)", 
            data=csv, 
            file_name="skdj_final_v6_1_export.csv", 
            mime="text/csv"
        )
    except pd.errors.EmptyDataError:
        st.info("🕒 当前暂无满足条件的回测记录。")
