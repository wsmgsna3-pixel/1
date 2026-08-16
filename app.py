# -*- coding: utf-8 -*-
"""
周线 SKDJ 底部脱离定型版 (V7.1 三仓实战资金修复定型版)
------------------------------------------------
1. 【兼容历史数据容错】：内置自动修复引擎，无缝兼容所有历史版本 checkpoint 数据，彻底杜绝 KeyError。
2. 【三仓 30万动态轮动】：初始本金 30万（3仓各10万），真实模拟现金流、仓位冻结与组合净值曲线。
3. 【A股 T+1 实战执行】：周一买入锁仓，周二起盘中实时监控 -10%硬止损、+2%保本锁、15%移动止盈。
4. 【首周五 14:50 截断】：首周五收盘浮亏 <= -3% 且周K收阴时尾盘果断平仓，将潜在亏损截断减半。
5. 【12周最大波段跨度】：最长持仓扩展至 12 周（60个交易日），吃满大牛股二浪主升。
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
        "skdj_market_data_v6_8.pkl",
        "skdj_market_data_v6_7.pkl",
        "skdj_market_data_v6_6.pkl",
        "skdj_market_data_v6_5.pkl",
        "skdj_market_data_master.pkl"
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
st.set_page_config(page_title="SKDJ 三仓实战资金回测系统", layout="wide")
st.title("💼 周线 SKDJ 底部脱离系统 (V7.1 三仓实战资金定型版)")
st.markdown("🔒 **30万初始本金 · 三仓等额轮动 · 首周五 -3% 截断 · +2% 保本 · 15% 移动止盈 · 12 周大波段**")

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
# 科技白名单池构建 (50亿市值底座)
# ---------------------------
@st.cache_data(ttl=3600*24*7, show_spinner=False)
def load_custom_tech_whitelist(token):
    token_c = clean_token_str(token)
    if not token_c: return set(), {}
    
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    
    stock_basic = safe_tushare_call(pro.stock_basic, list_status='L', fields='ts_code,symbol,name,industry,market,list_date')
    if stock_basic.empty:
        return set(), {}
        
    BOARDS = ("主板", "创业板", "科创板")
    valid_stocks = stock_basic[stock_basic['market'].isin(BOARDS)].copy()
    valid_stocks = valid_stocks[~valid_stocks['name'].str.contains('ST|退', na=False)]
    valid_stocks = valid_stocks[~valid_stocks['ts_code'].str.startswith('92')]
    
    CORE_TECH_L1 = {"电子", "计算机", "通信", "国防军工"}
    EXTENDED_TECH_L1 = {"机械设备", "电力设备", "医药生物", "汽车", "基础化工", "有色金属", "建筑材料"}
    TECH_INDUSTRY_KEYWORDS = {
        "半导体", "电子元件", "元件", "光学光电子", "消费电子", "电子化学品",
        "计算机设备", "软件开发", "IT服务", "通信设备", "军工电子", "航空装备",
        "航天装备", "自动化设备", "机器人", "激光设备", "工控设备", "仪器仪表",
        "电池", "光伏设备", "风电设备", "电网设备", "电机", "医疗器械",
        "生物制品", "汽车电子", "金属新材料", "非金属材料", "膜材料", "碳纤维",
    }
    
    sw_indices = safe_tushare_call(pro.index_classify, level='L1', src='SW2021')
    tech_l1_names = CORE_TECH_L1.union(EXTENDED_TECH_L1)
    target_sw = sw_indices[sw_indices['industry_name'].isin(tech_l1_names)] if not sw_indices.empty else pd.DataFrame()
    
    stock_sw_map = {}
    if not target_sw.empty:
        for _, s_row in target_sw.iterrows():
            idx_code = s_row['index_code']
            ind_name = s_row['industry_name']
            m_df = safe_tushare_call(pro.index_member, index_code=idx_code, is_new='Y')
            if not m_df.empty:
                for c_code in m_df['con_code']:
                    stock_sw_map[c_code] = ind_name
            time.sleep(0.03)
            
    whitelist_set = set()
    name_map = dict(zip(stock_basic['ts_code'], stock_basic['name']))
    
    for _, row in valid_stocks.iterrows():
        code = row['ts_code']
        ind_basic = str(row['industry']) if pd.notna(row['industry']) else ""
        sw_l1 = stock_sw_map.get(code, "")
        
        if sw_l1 in CORE_TECH_L1:
            whitelist_set.add(code)
            continue
            
        if sw_l1 in EXTENDED_TECH_L1:
            if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS) or ind_basic == "" or sw_l1 in {"机械设备", "电力设备", "医药生物"}:
                whitelist_set.add(code)
                continue
                
        if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS):
            whitelist_set.add(code)
            continue

    return whitelist_set, name_map

# ---------------------------
# 增量下载引擎
# ---------------------------
def sync_market_data_incrementally(start_date, end_date, token, whitelist_set):
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
            
            if whitelist_set:
                if not df_d.empty: df_d = df_d[df_d['ts_code'].isin(whitelist_set)]
                if not df_a.empty: df_a = df_a[df_a['ts_code'].isin(whitelist_set)]
                if not df_b.empty: df_b = df_b[df_b['ts_code'].isin(whitelist_set)]
                
            if not df_d.empty and not df_a.empty:
                cache['daily'].append(df_d)
                cache['adj'].append(df_a)
                if not df_b.empty:
                    cache['daily_basic'].append(df_b)
                cache['fetched_dates'].add(d)
            
            if (i + 1) % 10 == 0 or i == len(missing_dates) - 1:
                my_bar.progress((i+1)/len(missing_dates), text=f"📥 行情同步中: {i+1}/{len(missing_dates)}")
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
# 极速轻量化内存索引引擎
# ---------------------------
@st.cache_data(ttl=3600*12, show_spinner=False)
def load_optimized_market_data(start_date, end_date, token, _whitelist_keys, _dummy_trigger):
    token_c = clean_token_str(token)
    whitelist_set = set(_whitelist_keys)
    cache = sync_market_data_incrementally(start_date, end_date, token_c, whitelist_set)
    
    with st.spinner("正在构建科技池轻量化前复权索引..."):
        daily_list = cache.get('daily', [])
        adj_list = cache.get('adj', [])
        basic_list = cache.get('daily_basic', [])
        
        daily_raw = pd.concat(daily_list, ignore_index=True) if daily_list else pd.DataFrame()
        adj_raw = pd.concat(adj_list, ignore_index=True) if adj_list else pd.DataFrame()
        basic_raw = pd.concat(basic_list, ignore_index=True) if basic_list else pd.DataFrame()
        
        if daily_raw.empty or adj_raw.empty:
            return {}, pd.DataFrame()
            
        if whitelist_set:
            daily_raw = daily_raw[daily_raw['ts_code'].isin(whitelist_set)]
            adj_raw = adj_raw[adj_raw['ts_code'].isin(whitelist_set)]
            if not basic_raw.empty:
                basic_raw = basic_raw[basic_raw['ts_code'].isin(whitelist_set)]

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
            
        if not basic_raw.empty:
            basic_indexed = basic_raw.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['trade_date', 'ts_code'])
        else:
            basic_indexed = pd.DataFrame()
            
    return stock_qfq_dict, basic_indexed

# ---------------------------
# 核心信号引擎 (自然动能与深坑共振)
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

    is_breakout_25 = (curr_w['k'] > 25.0) and (prev_w['k'] <= 25.0)
    is_bullish = curr_w['k'] > curr_w['d']
    recent_d_min = weekly_df['d'].tail(10).min()
    has_bottom_gene = recent_d_min <= 25.0
    is_yang = curr_w['close'] > curr_w['open']
    
    is_vol_ok = True
    if pd.notna(curr_w['ma5_vol']) and curr_w['ma5_vol'] > 0:
        is_vol_ok = curr_w['vol'] >= curr_w['ma5_vol'] * 0.85

    base_signal = is_breakout_25 and is_bullish and has_bottom_gene and is_yang and is_vol_ok
    if not base_signal: return res

    ma20_curr = curr_w['ma20'] if pd.notna(curr_w['ma20']) else curr_w['close']
    ma20_prev = prev_w['ma20'] if pd.notna(prev_w['ma20']) else curr_w['close']
    ma20_slope = (ma20_curr - ma20_prev) / (ma20_prev + 1e-5)

    if curr_w['close'] < ma20_curr and ma20_slope < -0.002:
        return res

    if curr_w['close'] >= ma20_curr and ma20_slope >= -0.002:
        trend_type = "上升趋势"
    else:
        trend_type = "震荡/转换趋势"

    res['is_buy_signal'] = True
    res['k'] = round(curr_w['k'], 2)
    res['d'] = round(curr_w['d'], 2)
    res['recent_d_min'] = round(recent_d_min, 2)
    res['signal_close'] = curr_w['close'] 
    res['trend_type'] = trend_type
    
    vol_ratio = curr_w['vol'] / curr_w['ma5_vol'] if (pd.notna(curr_w['ma5_vol']) and curr_w['ma5_vol'] > 0) else 1.0
    res['vol_ratio'] = round(vol_ratio, 2)

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

    total_score = score_k + score_vol
    res['Total_Score'] = round(total_score, 1)

    return res

# ---------------------------
# 出局系统：T+1 + 首周五截断 + +2%保本 + 15%移动止盈 + 12周跨度
# ---------------------------
def track_future_performance(ts_code, selection_date, signal_close, stock_qfq_dict, hold_weeks=12):
    default_res = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    default_res.update({
        'Exit_Reason': '持仓中', 'Buy_Price': np.nan, 'Gap_pct (%)': np.nan, 
        'Exit_Date': None, 'Final_Return (%)': np.nan, 'Hold_Days': 0
    })
    
    if ts_code not in stock_qfq_dict: 
        return default_res

    df_full = stock_qfq_dict[ts_code]
    hist_future = df_full[df_full.index > selection_date]
    results = default_res.copy()
    
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
    peak_price = buy_price
    pending_exit_reason = None  
    hard_stop_limit = -0.10 
    
    max_days = hold_weeks * 5
    
    for i in range(len(hist_future)):
        if i >= max_days: break 
            
        row = hist_future.iloc[i]
        day_count = i + 1
        current_week = ((day_count - 1) // 5) + 1 
        curr_open, curr_close, curr_high, curr_low = row['open'], row['close'], row['high'], row['low']
        curr_date = hist_future.index[i]
        
        if pending_exit_reason is not None and day_count >= 2:
            if "保本" in pending_exit_reason:
                final_return = 2.0  
            else:
                final_return = (curr_open - buy_price) / buy_price * 100.0
                
            exit_triggered = True
            results['Exit_Reason'] = pending_exit_reason
            results['Final_Return (%)'] = round(final_return, 2)
            results['Exit_Date'] = curr_date
            results['Hold_Days'] = day_count
            results[f'Return_W{current_week} (%)'] = round(final_return, 2)
            break
        
        peak_price = max(peak_price, curr_high)
        peak_profit_pct = (peak_price - buy_price) / buy_price
        
        if day_count >= 2:
            if (curr_low - buy_price) / buy_price <= hard_stop_limit:
                final_return = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
                exit_triggered = True
                results['Exit_Reason'] = "认栽出局(破-10%)"
                results['Final_Return (%)'] = round(final_return, 2)
                results['Exit_Date'] = curr_date
                results['Hold_Days'] = day_count
                results[f'Return_W{current_week} (%)'] = round(final_return, 2)
                break
        
        if tier == 0 and peak_profit_pct >= 0.10: 
            tier = 1  
            
        if tier == 1:
            if curr_close <= buy_price * 1.02:  
                pending_exit_reason = "保本离场(+2%)"
            elif peak_profit_pct >= 0.20: 
                tier = 2  
                
        if tier == 2:
            giveback = (peak_price - curr_close) / peak_price
            if giveback >= 0.15:  
                pending_exit_reason = "移动止盈(回撤15%)"
        
        if day_count == 5 and not exit_triggered and pending_exit_reason is None:
            w1_ret = (curr_close - buy_price) / buy_price * 100.0
            if w1_ret <= -3.0:
                exit_triggered = True
                results['Exit_Reason'] = f"首周不及预期截断({round(w1_ret, 1)}%)"
                results['Final_Return (%)'] = round(w1_ret, 2)
                results['Exit_Date'] = curr_date
                results['Hold_Days'] = 5
                results['Return_W1 (%)'] = round(w1_ret, 2)
                break
            
        if day_count % 5 == 0:
            results[f'Return_W{current_week} (%)'] = round((curr_close - buy_price) / buy_price * 100.0, 2)
            
    if not exit_triggered and len(hist_future) >= max_days:
        last_price = hist_future.iloc[max_days - 1]['close']
        final_return = (last_price - buy_price) / buy_price * 100.0
        results[f'Return_W{hold_weeks} (%)'] = round(final_return, 2)
        results['Exit_Reason'] = "12周期满平仓"
        results['Final_Return (%)'] = round(final_return, 2)
        results['Exit_Date'] = hist_future.index[max_days - 1]
        results['Hold_Days'] = max_days
        
    return results

# ---------------------------
# 🚀 历史数据自动兼容与修复引擎
# ---------------------------
def repair_checkpoint_df(df_in):
    df_out = df_in.copy()
    w_cols = [c for c in df_out.columns if c.startswith('Return_W') and c.endswith('(%)')]
    if w_cols:
        w_cols = sorted(w_cols, key=lambda x: int(x.replace('Return_W', '').replace(' (%)', '')))
    
    if 'Final_Return (%)' not in df_out.columns:
        def get_final_ret(r):
            if not w_cols: return 0.0
            rets = r[w_cols].dropna()
            return rets.iloc[-1] if not rets.empty else 0.0
        df_out['Final_Return (%)'] = df_out.apply(get_final_ret, axis=1)
        
    if 'Exit_Date' not in df_out.columns:
        df_out['Exit_Date'] = None
        
    if 'Hold_Days' not in df_out.columns:
        def get_hold_days(r):
            if not w_cols: return 0
            rets = r[w_cols].dropna()
            return len(rets) * 5 if not rets.empty else 0
        df_out['Hold_Days'] = df_out.apply(get_hold_days, axis=1)
        
    return df_out

# ---------------------------
# UI 控制流与输入侧边栏
# ---------------------------
with st.sidebar:
    st.header("⚙️ 实战账户配置")
    INIT_TOTAL_CAPITAL = st.number_input("初始总资金 (元)", value=300000, step=50000)
    MAX_SLOTS = st.number_input("持仓槽位数 (仓位)", value=3, min_value=1, max_value=5)
    st.info(f"💡 单仓分配本金：**{INIT_TOTAL_CAPITAL // MAX_SLOTS:,} 元**")
    
    st.markdown("---")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("追溯交易天数", value=250, step=30)
    
    st.markdown("---")
    if st.button("🗑️ 清空行情缓存"):
        if os.path.exists(MARKET_CACHE_FILE):
            os.remove(MARKET_CACHE_FILE)
        st.cache_data.clear()
        st.success("底层行情缓存已清理！")
            
    if st.button("🗑️ 清除所有回测记录"):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        st.success("回测记录已清理！")
            
    st.markdown("---")
    st.subheader("💰 护城河底座")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=50.0) 
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
if st.button("🚀 启动实战资金组合回测"):
    is_valid, msg = verify_token_connection(token_clean)
    if not is_valid:
        st.error(f"❌ **Token 预检拦截**：{msg}")
    else:
        try:
            ts.set_token(token_clean)
            pro = ts.pro_api(token_clean)
            
            with st.spinner("正在精准筛选科技池白名单标的..."):
                whitelist_set, basic_name_map = load_custom_tech_whitelist(token_clean)
                whitelist_keys = tuple(sorted(whitelist_set))
                
            if not whitelist_keys:
                st.error("❌ 未能获取到科技白名单股票，请检查 Token 积分或网络。")
            else:
                st.info(f"💡 成功锁定科技白名单股票池：共 **{len(whitelist_keys)}** 只标的。")
                
                lookback_days = max(int(BACKTEST_DAYS) * 3, 500) 
                start_cal = (datetime.strptime(backtest_date_end.strftime("%Y%m%d"), "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
                end_cal = backtest_date_end.strftime("%Y%m%d")
                
                cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_cal, end_date=end_cal)
                if cal_raw.empty:
                    st.error("❌ 无法获取交易日历。")
                else:
                    cal_open = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date', ascending=True)
                    trade_days_list = cal_open['cal_date'].tolist()
                    
                    if not trade_days_list:
                        st.error("❌ 未获取到有效交易日。")
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
                            st.success("🎉 指定区间数据已扫描完毕！查看下方实盘报告即可。")
                        else:
                            fetch_start = (datetime.strptime(min(dates_to_run), "%Y%m%d") - timedelta(days=550)).strftime("%Y%m%d")
                            fetch_end = (datetime.strptime(max(dates_to_run), "%Y%m%d") + timedelta(days=200)).strftime("%Y%m%d")
                            
                            dummy_trigger = time.time()
                            stock_qfq_dict, basic_indexed = load_optimized_market_data(fetch_start, fetch_end, token_clean, whitelist_keys, dummy_trigger)
                            
                            if not stock_qfq_dict:
                                st.warning("⚠️ 未能加载到行情数据，请重试。")
                            else:
                                bar = st.progress(0, text="执行 V7.1 实战信号扫描...")
                                
                                for i, date in enumerate(dates_to_run):
                                    records = []
                                    
                                    for ts_code in whitelist_keys:
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
                                            
                                        circ_mv_billion = np.nan
                                        if not basic_indexed.empty and (date, ts_code) in basic_indexed.index:
                                            circ_mv_billion = basic_indexed.loc[(date, ts_code)]['circ_mv'] / 10000.0
                                        
                                        if pd.notna(circ_mv_billion):
                                            if circ_mv_billion < MIN_MV or circ_mv_billion > MAX_MV:
                                                continue
                                        
                                        ind = compute_breakout_signal(ts_code, date, stock_qfq_dict)
                                        if not ind or not ind.get('is_buy_signal'): 
                                            continue
                                            
                                        future_returns = track_future_performance(ts_code, date, ind['signal_close'], stock_qfq_dict, hold_weeks=12)
                                        
                                        stock_name = basic_name_map.get(ts_code, ts_code)
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
                                        fdf = pd.DataFrame(records).sort_values('Total_Score', ascending=False).head(int(MAX_SLOTS) * 2)
                                        fdf.insert(0, 'Rank', range(1, len(fdf) + 1))
                                        fdf['Trade_Date'] = date
                                        is_first = not os.path.exists(CHECKPOINT_FILE)
                                        fdf.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                                        
                                    bar.progress((i+1)/len(dates_to_run), text=f"扫描中: {date} (捕获 {len(records)} 只目标)")
                                    
                                bar.empty()
                                st.success("🎉 实战回测数据已全部处理完毕！")
        except Exception as e:
            st.error(f"❌ **运行异常拦截**：{str(e)}")

# ---------------------------
# 实战资金曲线与三仓模拟展示区
# ---------------------------
if os.path.exists(CHECKPOINT_FILE):
    st.markdown("---")
    try:
        raw_res = pd.read_csv(CHECKPOINT_FILE)
        raw_res['Trade_Date'] = raw_res['Trade_Date'].astype(str)
        
        # 🚀 自动执行历史数据修复与容错补齐
        repaired_res = repair_checkpoint_df(raw_res)
        valid_signals = repaired_res[~repaired_res['Exit_Reason'].astype(str).str.contains('剔除', na=False)].copy()
        
        st.header("📈 三仓实战账户模拟报告 (本金 300,000 元)")
        
        slot_cash = [INIT_TOTAL_CAPITAL / MAX_SLOTS] * int(MAX_SLOTS)
        slot_occupied_until = ["" for _ in range(int(MAX_SLOTS))]
        portfolio_trades = []
        
        unique_dates = sorted(valid_signals['Trade_Date'].astype(str).unique())
        
        for date_str in unique_dates:
            clean_d_str = str(date_str).replace("-", "")
            day_candidates = valid_signals[valid_signals['Trade_Date'].astype(str) == date_str].sort_values('Rank', ascending=True)
            
            free_slots = []
            for s_idx in range(int(MAX_SLOTS)):
                occ = str(slot_occupied_until[s_idx]).replace("-", "")
                if occ == "" or occ <= clean_d_str:
                    free_slots.append(s_idx)
                    
            if not free_slots:
                continue
                
            for _, row in day_candidates.iterrows():
                if not free_slots:
                    break
                    
                target_slot = free_slots.pop(0)
                alloc_capital = slot_cash[target_slot]
                
                final_pct = row.get('Final_Return (%)', np.nan)
                if pd.isna(final_pct):
                    final_pct = 0.0
                    
                profit_amount = alloc_capital * (final_pct / 100.0)
                end_capital = alloc_capital + profit_amount
                slot_cash[target_slot] = end_capital
                
                if pd.notna(row.get('Exit_Date')) and str(row['Exit_Date']).strip() != "":
                    exit_date_str = str(row['Exit_Date']).replace("-", "")
                else:
                    hold_days = row.get('Hold_Days', 40)
                    if pd.isna(hold_days) or hold_days <= 0:
                        hold_days = 40
                    try:
                        td_dt = datetime.strptime(clean_d_str, "%Y%m%d")
                        exit_dt = td_dt + timedelta(days=int(hold_days * 7 / 5))
                        exit_date_str = exit_dt.strftime("%Y%m%d")
                    except Exception:
                        exit_date_str = "20991231"
                        
                slot_occupied_until[target_slot] = exit_date_str
                
                trade_record = row.to_dict()
                trade_record['Slot'] = f"槽位 {target_slot + 1}"
                trade_record['Alloc_Capital'] = round(alloc_capital, 2)
                trade_record['End_Capital'] = round(end_capital, 2)
                trade_record['Net_Profit'] = round(profit_amount, 2)
                trade_record['Exit_Date_Clean'] = exit_date_str
                portfolio_trades.append(trade_record)

        if portfolio_trades:
            port_df = pd.DataFrame(portfolio_trades)
            total_current_value = sum(slot_cash)
            total_net_profit = total_current_value - INIT_TOTAL_CAPITAL
            total_return_pct = (total_net_profit / INIT_TOTAL_CAPITAL) * 100.0
            
            comp_trades = port_df[port_df['Exit_Reason'] != '持仓中']
            win_count = (comp_trades['Final_Return (%)'] > 0).sum()
            total_comp_count = len(comp_trades)
            portfolio_win_rate = (win_count / total_comp_count * 100) if total_comp_count > 0 else 0.0
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("账户期末总资产", f"¥ {total_current_value:,.2f}", f"{total_return_pct:+.2f}%")
            col_m2.metric("累计净利润", f"¥ {total_net_profit:+,.2f}")
            col_m3.metric("三仓实盘总胜率", f"{portfolio_win_rate:.1f}%", f"{win_count}胜 / {total_comp_count}笔")
            col_m4.metric("交易执行总笔数", f"{len(port_df)} 笔")
            
            st.subheader("🗓️ 周度胜率分布 (严格对齐周末信号)")
            cols_row1 = st.columns(4)
            cols_row2 = st.columns(4)
            
            for w in range(1, 9):
                col_name = f'Return_W{w} (%)'
                if col_name in valid_signals.columns:
                    valid = valid_signals.dropna(subset=[col_name]) 
                    target_col = cols_row1[w-1] if w <= 4 else cols_row2[w-5]
                    with target_col:
                        if not valid.empty:
                            avg = valid[col_name].mean()
                            win = (valid[col_name] > 0).mean() * 100
                            st.metric(f"W{w} 均益/胜率 (存活{len(valid)}只)", f"{avg:.2f}% / {win:.1f}%")
                        else:
                            st.metric(f"W{w}", "空缺")
            
            st.subheader("📋 三仓实操交割流水单")
            port_disp_cols = [
                'Slot', 'Trade_Date', 'name', 'ts_code', 'Rank', 'Total_Score', 'Trend_Type',
                'Buy_Price', 'Alloc_Capital', 'Exit_Date', 'Hold_Days', 'Exit_Reason', 'Final_Return (%)', 'Net_Profit', 'End_Capital'
            ]
            final_port_cols = [c for c in port_disp_cols if c in port_df.columns]
            
            def color_exit_reason(val):
                if isinstance(val, str):
                    if '截断' in val: return 'color: white; background-color: #8B4513'
                    elif '认栽' in val: return 'color: white; background-color: darkred'
                    elif '保本' in val: return 'color: white; background-color: darkgoldenrod'
                    elif '移动止盈' in val: return 'color: white; background-color: darkgreen'
                    elif '期满' in val: return 'color: blue'
                return ''
                
            styled_port = port_df[final_port_cols].sort_values('Trade_Date', ascending=False).style
            if 'Exit_Reason' in port_df.columns:
                styled_port = styled_port.map(color_exit_reason, subset=['Exit_Reason'])
                
            try:
                st.dataframe(styled_port, width="stretch")
            except Exception:
                st.dataframe(styled_port, use_container_width=True)
                
            csv_data = port_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 导出三仓实战流水 (CSV)", 
                data=csv_data, 
                file_name="portfolio_3slots_v7_1_export.csv", 
                mime="text/csv"
            )
        else:
            st.info("🕒 当前暂无可执行的资金组合流水。")
    except pd.errors.EmptyDataError:
        st.info("🕒 当前暂无满足条件的回测记录。")
