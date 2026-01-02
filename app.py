# -*- coding: utf-8 -*-
"""
选股王 · V30.25 完整修复版 (含收益计算函数)
核心修复：
1. [补全] 找回了丢失的 get_future_prices_real_combat 函数，现在点击按钮会有反应了。
2. [数据] 筹码数据分批获取 (Chunk Size=20)，解决批量失败导致全员60分的BUG。
3. [风控] 乖离率阈值从 18% 降至 12%，超过 20% 直接剔除。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 全局设置
# ---------------------------
st.set_page_config(page_title="选股王 · V30.25 完整版", layout="wide")
st.title("选股王 · V30.25 (🔧 完整运行 + 🛡️ 严厉风控)")

# 初始化变量
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

# ---------------------------
# 1. 基础辅助函数 
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
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    return pd.DataFrame(columns=['ts_code']) 
                return df
            except Exception:
                time.sleep(0.5)
                continue
        return pd.DataFrame(columns=['ts_code'])
    except Exception: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3 + 30)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历。")
        return []
    return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()

# ---------------------------
# 2. 数据拉取 (含缓存)
# ---------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    latest = max(trade_days_list) 
    earliest = min(trade_days_list)
    start_date = (datetime.strptime(earliest, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d") 
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    st.info(f"⏳ 正在拉取 {start_date} 到 {end_date} 行情...")

    adj_list, daily_list = [], []
    download_progress = st.progress(0, text="数据同步中...")
    total = len(all_dates)
    
    for i, date in enumerate(all_dates):
        try:
            res = fetch_and_cache_daily_data(date)
            if not res['adj'].empty: adj_list.append(res['adj'])
            if not res['daily'].empty: daily_list.append(res['daily'])
            if i % 10 == 0: download_progress.progress((i + 1) / total)
        except: continue 
    download_progress.empty()

    if not adj_list or not daily_list:
        st.error("无法获取历史数据。")
        return False
        
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    valid_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
    daily_raw = pd.concat(daily_list)
    daily_raw = daily_raw[[c for c in valid_cols if c in daily_raw.columns]]
    
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    try:
        latest_dt = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
        latest_adj = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_dt), 'adj_factor']
        GLOBAL_QFQ_BASE_FACTORS = latest_adj.droplevel(1).to_dict()
    except: GLOBAL_QFQ_BASE_FACTORS = {}
    
    return True

# ---------------------------
# 3. 复权数据计算 (关键函数)
# ---------------------------
def get_qfq_data_optimized(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    try:
        idx = pd.IndexSlice
        daily = GLOBAL_DAILY_RAW.loc[idx[ts_code, start_date:end_date], :]
        adj = GLOBAL_ADJ_FACTOR.loc[idx[ts_code, start_date:end_date], 'adj_factor']
    except: return pd.DataFrame()
    
    if daily.empty or adj.empty: return pd.DataFrame()
    common = daily.index.intersection(adj.index)
    if common.empty: return pd.DataFrame()
    
    daily, adj = daily.loc[common], adj.loc[common]
    base = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(base) or base < 1e-9: return pd.DataFrame()
    
    factor = adj / base
    df = daily.copy()
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns: df[col] = df[col] * factor
        
    df = df.reset_index().rename(columns={'trade_date': 'date'})
    df['trade_date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df.sort_values('trade_date').set_index('date')[['open', 'high', 'low', 'close', 'pre_close', 'vol']]

# ---------------------------
# 4. 收益计算函数 (之前缺失的!)
# ---------------------------
def get_future_prices_real_combat(ts_code, selection_date, days_ahead=[1, 3, 5], buy_threshold_pct=1.5):
    """
    计算买入后的未来收益
    """
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=25)).strftime("%Y%m%d")
    
    hist = get_qfq_data_optimized(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    for n in days_ahead: results[f'Return_D{n}'] = np.nan

    if hist.empty: return results
    
    d1_data = hist.iloc[0]
    
    # 1. 拒绝低开
    if d1_data['open'] <= d1_data['pre_close']: return results 
    
    # 2. 确认 +1.5% 买入
    buy_price_threshold = d1_data['open'] * (1 + buy_threshold_pct / 100.0)
    if d1_data['high'] < buy_price_threshold: return results 

    # 3. 计算收益
    buy_price = buy_price_threshold
    
    for n in days_ahead:
        idx = n - 1
        if len(hist) > idx:
            current_close = hist.iloc[idx]['close']
            results[f'Return_D{n}'] = (current_close / buy_price - 1) * 100
            
    return results

# ---------------------------
# 5. 指标计算 (归一化MACD + 乖离率)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data_optimized(ts_code, start_date=start_date, end_date=end_date)
    res = {}
    if df.empty or len(df) < 26: return res
         
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    close = df['close']
    
    # 暴力MACD (8,17,5)
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd_val = (diff - dea) * 2
    res['macd_val'] = macd_val.iloc[-1]
    
    # 均线与乖离
    ma20 = close.rolling(window=20).mean()
    res['ma20_current'] = ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else 0
    res['close_current'] = close.iloc[-1]
    if res['ma20_current'] > 0:
        res['bias_20'] = (res['close_current'] - res['ma20_current']) / res['ma20_current'] * 100
    else: res['bias_20'] = 0
        
    vol = df['vol']
    ma5_vol = vol.rolling(window=5).mean()
    res['vol_current'] = vol.iloc[-1]
    res['ma5_vol_current'] = ma5_vol.iloc[-1] if not pd.isna(ma5_vol.iloc[-1]) else 0
    res['pct_chg_current'] = df['pct_chg'].iloc[-1]
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    return res

@st.cache_data(ttl=3600*12)
def get_market_state(trade_date):
    start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    index_data = safe_get('daily', ts_code='000300.SH', start_date=start_date, end_date=trade_date, is_index=True)
    if index_data.empty or len(index_data) < 20: return 'Weak'
    index_data = index_data.sort_values('trade_date')
    return 'Strong' if index_data.iloc[-1]['close'] > index_data['close'].tail(20).mean() else 'Weak'

# ---------------------------
# 6. 核心运行逻辑 (分批筹码 + 风控)
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, buy_threshold):
    # 1. 弱市熔断
    market_state = get_market_state(last_trade)
    if market_state == 'Weak': return pd.DataFrame(), f"弱市避险"

    # 2. 拉取数据
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失"
    
    pool = daily_all.reset_index(drop=True)
    pool = pool[~pool['ts_code'].str.startswith('92')]
    
    # 补充基础信息
    pool['close'] = pd.to_numeric(pool['close'], errors='coerce')
    pool['amount'] = pd.to_numeric(pool['amount'], errors='coerce').fillna(0)
    pool['pct_chg'] = pd.to_numeric(pool['pct_chg'], errors='coerce').fillna(0)
    
    # 补充名字
    basic = safe_get('stock_basic', list_status='L', fields='ts_code,name')
    if not basic.empty: pool = pool.merge(basic, on='ts_code', how='left')
    pool = pool[~pool['name'].str.contains('ST|退', case=False, na=False)]

    # 初筛条件
    pool = pool[
        (pool['close'] >= 5) & (pool['close'] <= 200) & 
        (pool['amount'] >= 100000) & (pool['pct_chg'] > 0)
    ]
    
    # 优选Candidates：优先看活跃度 (3% < 涨幅 < 9.6%)
    candidates = pool[(pool['pct_chg'] >= 3.0) & (pool['pct_chg'] <= 9.6)]
    if len(candidates) < FINAL_POOL:
        candidates = pool.sort_values('pct_chg', ascending=False).head(FINAL_POOL)
    else:
        d_basic = safe_get('daily_basic', trade_date=last_trade, fields='ts_code,turnover_rate')
        if not d_basic.empty: candidates = candidates.merge(d_basic, on='ts_code', how='left')
        candidates = candidates.sort_values('turnover_rate', ascending=False).head(FINAL_POOL)

    # --- 🚀 核心修复：分批获取筹码数据 ---
    cyq_map = {}
    code_list = candidates['ts_code'].tolist()
    
    if code_list:
        chunk_size = 20 # 每次请求20个
        for i in range(0, len(code_list), chunk_size):
            chunk = code_list[i:i+chunk_size]
            try:
                chunk_str = ",".join(chunk)
                cyq_df = safe_get('cyq_perf', ts_code=chunk_str, trade_date=last_trade)
                if not cyq_df.empty:
                    batch_map = cyq_df.set_index('ts_code')['winner_rate'].to_dict()
                    cyq_map.update(batch_map)
                time.sleep(0.1) 
            except: pass
    # ---------------------------------------

    records = []
    for row in candidates.itertuples():
        ind = compute_indicators(row.ts_code, last_trade)
        
        # 硬门槛
        if ind.get('close_current', 0) <= ind.get('ma20_current', 0): continue
        if ind.get('vol_current', 0) <= ind.get('ma5_vol_current', 0) * 1.1: continue
        if pd.isna(ind.get('macd_val')) or ind.get('macd_val') <= 0: continue
        
        # ⛔ [风控核心] 筹码过滤
        winner_rate = cyq_map.get(row.ts_code, -1) # 默认 -1 表示未知
        
        # 如果是未知数据，默认给 50 分，但如果有10000积分不应如此
        if winner_rate == -1: winner_rate = 50.0 
        
        # 过滤套牢盘严重的 ( < 40% )
        if winner_rate < 40.0: continue

        # ⛔ [风控核心] 乖离率直接剔除 ( > 20% )
        if ind['bias_20'] > 20.0: continue
        
        # 计算未来收益
        future = get_future_prices_real_combat(row.ts_code, last_trade, buy_threshold_pct=buy_threshold)

        records.append({
            'ts_code': row.ts_code, 'name': getattr(row, 'name', row.ts_code),
            'Close': row.close, 'Pct_Chg (%)': row.pct_chg,
            'macd': ind['macd_val'], 'volatility': ind['volatility'],
            'bias_20': ind['bias_20'], 'winner_rate': winner_rate,
            'Return_D1 (%)': future.get('Return_D1'),
            'Return_D3 (%)': future.get('Return_D3'),
            'Return_D5 (%)': future.get('Return_D5')
        })
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), "无标的"

    # [评分系统 V30.25]
    fdf['macd_ratio'] = (fdf['macd'] / fdf['Close']) * 100
    fdf['base_score'] = np.log1p(fdf['macd_ratio']) * 10000 
    
    def calc_score(row):
        score = row['base_score']
        tags = []
        # 1. 筹码加分
        if row['winner_rate'] >= 85: 
            score *= 1.2; tags.append('筹码佳')
        
        # 2. 乖离率惩罚 (更严厉)
        if 12.0 < row['bias_20'] <= 20.0:
            score *= 0.7; tags.append('过热惩罚')
        
        return score, "+".join(tags)

    fdf[['综合评分', '加分项']] = fdf.apply(lambda x: pd.Series(calc_score(x)), axis=1)
    
    # 返回前 N 名
    return fdf.sort_values('综合评分', ascending=False).head(TOP_BACKTEST), None

# ---------------------------
# 7. 主程序界面
# ---------------------------
with st.sidebar:
    st.header("1. 回测设置")
    backtest_date_end = st.date_input("回测结束日期", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("**回测天数 (N)**", value=30, step=1))
    
    st.markdown("---")
    st.header("2. 实战参数 (V30.25)")
    BUY_THRESHOLD_PCT = st.number_input("买入确认阈值 (%)", value=1.5, step=0.1)
    
    st.markdown("---")
    st.header("3. 基础过滤")
    FINAL_POOL = int(st.number_input("入围数量", value=100)) 
    TOP_BACKTEST = int(st.number_input("Top K", value=5))

TS_TOKEN = st.text_input("Tushare Token (需10000积分)", type="password")

if st.button("🚀 启动 V30.25 完整版"):
    if not TS_TOKEN:
        st.error("请输入 Token")
        st.stop()
        
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days: st.stop()
    if not get_all_historical_data(trade_days): st.stop()
    
    st.success(f"✅ V30.25 启动! 回测日期: {trade_days[-1]} 至 {trade_days[0]}")
    results = []
    bar = st.progress(0)
    
    for i, date in enumerate(trade_days):
        try:
            df, msg = run_backtest_for_a_day(date, TOP_BACKTEST, FINAL_POOL, BUY_THRESHOLD_PCT)
            if not df.empty:
                df['Trade_Date'] = date
                results.append(df)
            time.sleep(0.1) 
        except Exception as e:
            st.error(f"{date} 出错: {e}")
        bar.progress((i + 1) / len(trade_days))
    bar.empty()
    
    if not results:
        st.error("无结果。")
        st.stop()
        
    all_res = pd.concat(results)
    if all_res['Trade_Date'].dtype != 'object': all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
    st.header(f"📊 V30.25 回测报告")
    st.markdown(f"**有效交易天数：** {all_res['Trade_Date'].nunique()} 天")

    cols = st.columns(3)
    for idx, n in enumerate([1, 3, 5]):
        col = f'Return_D{n} (%)' 
        valid = all_res.dropna(subset=[col])
        if not valid.empty:
            avg_ret = valid[col].mean()
            hit_rate = (valid[col] > 0).sum() / len(valid) * 100
            count = len(valid)
        else: avg_ret, hit_rate, count = 0, 0, 0
        with cols[idx]:
            st.metric(f"D+{n} 收益 / 胜率", f"{avg_ret:.2f}% / {hit_rate:.1f}%", help=f"成交：{count} 笔")

    st.header("📋 每日成交明细")
    # 动态获取存在的列
    cols_to_show = ['Trade_Date', 'ts_code', 'name', 'Close', 'Pct_Chg (%)', 'macd', 'bias_20', 'winner_rate', '综合评分', 'Return_D5 (%)']
    valid_cols = [c for c in cols_to_show if c in all_res.columns]
    
    st.dataframe(all_res[valid_cols].sort_values('Trade_Date', ascending=False), use_container_width=True)
