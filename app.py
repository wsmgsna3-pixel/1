# -*- coding: utf-8 -*-
"""
选股王 · V30.23 涅槃版 (防接盘优化 + 智能止损)
核心改进：
1. [风控] 增加刚性止损 (-5%)，拒绝死扛。
2. [避险] 增加 RSI>80 过滤，禁止在严重超买区接盘。
3. [评分] 使用 MACD/股价 归一化评分，不再盲目追求高 MACD 数值。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V30.23 涅槃版", layout="wide")
st.title("选股王 · V30.23 涅槃版（🛡️ 智能止损 + 📉 RSI避险）")
st.markdown("""
**🔥 V30.23 核心升级逻辑：**
1. **拒绝山顶：** 如果 **RSI > 80**，判定为情绪过热，**坚决不买**。
2. **智能止损：** 模拟实盘，一旦跌破买入价 **-5%**，强制止损离场。
3. **公平评分：** 使用 **MACD相对强度** (MACD/Price)，让低价妖股也能入围。
""")

# ---------------------------
# 辅助函数 
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        if kwargs.get('is_index'): df = pro.index_daily(**kwargs)
        else: df = func(**kwargs)
        
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历。")
        return []
    return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()

# ----------------------------------------------------------------------
# 数据拉取
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d") 
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    st.info(f"⏳ 正在下载 {start_date} 到 {end_date} 间的数据（请稍候）...")

    adj_list, daily_list = [], []
    download_progress = st.progress(0, text="数据同步中...")
    
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty: adj_list.append(cached_data['adj'])
            if not cached_data['daily'].empty: daily_list.append(cached_data['daily'])
            download_progress.progress((i + 1) / len(all_dates))
        except: continue 
    download_progress.empty()

    if not adj_list or not daily_list:
        st.error("无法获取历史数据。")
        return False
        
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    cols_to_keep = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol']
    valid_cols = [c for c in cols_to_keep if c in daily_list[0].columns]
    daily_raw = pd.concat(daily_list)[valid_cols]
    
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'vol']:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        try:
            latest_adj = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj.droplevel(1).to_dict()
        except: GLOBAL_QFQ_BASE_FACTORS = {}
    
    return True

# ----------------------------------------------------------------------
# 数据处理
# ----------------------------------------------------------------------
def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty: return pd.DataFrame()
        
    base_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(base_adj) or base_adj < 1e-9: return pd.DataFrame() 

    try:
        daily = GLOBAL_DAILY_RAW.loc[ts_code]
        daily = daily.loc[(daily.index >= start_date) & (daily.index <= end_date)]
        adj = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        adj = adj.loc[(adj.index >= start_date) & (adj.index <= end_date)]
    except KeyError: return pd.DataFrame()
    
    if daily.empty or adj.empty: return pd.DataFrame()
    
    df = daily.merge(adj.rename('adj_factor'), left_index=True, right_index=True, how='left').dropna(subset=['adj_factor'])
    df = df.sort_index()
    
    factor = df['adj_factor'] / base_adj
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns: df[col] = df[col] * factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df['trade_date'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')
    return df.sort_values('trade_date').set_index('trade_date_str')[['open', 'high', 'low', 'close', 'pre_close', 'vol']]

# ----------------------------------------------------------------------
# [重点修改] 核心买入计算 (含止损)
# ----------------------------------------------------------------------
def get_future_prices_real_combat(ts_code, selection_date, days_ahead=[1, 3, 5], buy_threshold_pct=1.5, stop_loss_pct=5.0):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    # 多取一些数据防止跨周末不够
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=35)).strftime("%Y%m%d")
    
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    for n in days_ahead: results[f'Return_D{n}'] = np.nan

    if hist.empty: return results
    d1_data = hist.iloc[0]
    
    # 1. 拒绝低开
    if d1_data['open'] <= d1_data['pre_close']: return results 
    
    # 2. 确认突破 +1.5%
    buy_price = d1_data['open'] * (1 + buy_threshold_pct / 100.0)
    if d1_data['high'] < buy_price: return results 

    # 3. 计算收益 (加入止损逻辑)
    stop_price = buy_price * (1 - stop_loss_pct / 100.0)

    for n in days_ahead:
        idx = n - 1
        if len(hist) > idx:
            # 截取持有期内的数据
            period_data = hist.iloc[:n]
            # 检查是否有任何一天的最低价击穿止损线
            if (period_data['low'] < stop_price).any():
                # 触发止损，亏损固定为 stop_loss_pct
                results[f'Return_D{n}'] = -stop_loss_pct
            else:
                # 未触发止损，按第N天收盘价结算
                results[f'Return_D{n}'] = (hist.iloc[idx]['close'] / buy_price - 1) * 100
            
    return results

# ----------------------------------------------------------------------
# 指标计算 (加入 RSI)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    res = {}
    if df.empty or len(df) < 26: return res
         
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    close = df['close']
    vol = df['vol']
    
    # 1. 改进版 MACD (8, 17, 5)
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd_val = (diff - dea) * 2
    res['macd_val'] = macd_val.iloc[-1]
    
    # 2. [新增] RSI 指标 (14天)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # 使用 com=13 对应 span=27 左右的平滑，接近标准 RSI
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    res['rsi_current'] = rsi.iloc[-1]
    
    # 3. 均线/量能/其他特征
    ma20 = close.rolling(window=20).mean()
    ma5_vol = vol.rolling(window=5).mean()
    
    res['close_current'] = close.iloc[-1]
    res['ma20_current'] = ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else 0
    res['vol_current'] = vol.iloc[-1]
    res['ma5_vol_current'] = ma5_vol.iloc[-1] if not pd.isna(ma5_vol.iloc[-1]) else 0
    res['pct_chg_current'] = df['pct_chg'].iloc[-1]
    
    # 波动率
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    return res

@st.cache_data(ttl=3600*12)
def get_market_state(trade_date):
    start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    index_data = safe_get('daily', ts_code='000300.SH', start_date=start_date, end_date=trade_date, is_index=True)
    if index_data.empty or len(index_data) < 20: return 'Weak'
    index_data = index_data.sort_values('trade_date')
    return 'Strong' if index_data.iloc[-1]['close'] > index_data['close'].tail(20).mean() else 'Weak'
       
# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测设置")
    backtest_date_end = st.date_input("回测结束日期", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("**回测天数 (N)**", value=50, step=1))
    
    st.markdown("---")
    st.header("2. 实战参数 (V30.23)")
    BUY_THRESHOLD_PCT = st.number_input("买入确认阈值 (%)", value=1.5, step=0.1)
    STOP_LOSS_PCT = st.number_input("🛑 止损阈值 (%)", value=5.0, step=0.5, help="盘中跌破买入价多少比例强制止损")
    
    st.markdown("---")
    st.header("3. 基础过滤")
    FINAL_POOL = int(st.number_input("入围数量", value=100)) 
    TOP_BACKTEST = int(st.number_input("Top K", value=5))
    MIN_PRICE = st.number_input("最低股价", value=8.0, step=0.5) 
    MAX_PRICE = st.number_input("最高股价", value=300.0, step=5.0)
    MIN_TURNOVER = st.number_input("最低换手 (%)", value=3.0) 
    # [修改] 默认值设为 30亿，兼顾活跃性
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿)", value=30.0)
    MIN_AMOUNT = st.number_input("最低成交额 (亿)", value=1.0) * 100000000 

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ----------------------------------------------------------------------
# 核心逻辑 (V30.23 涅槃版)
# ----------------------------------------------------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, buy_threshold):
    # 1. 弱市熔断
    market_state = get_market_state(last_trade)
    if market_state == 'Weak': return pd.DataFrame(), f"弱市避险"

    # 2. 拉取数据
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失"
    pool = daily_all.reset_index(drop=True)
    basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date') 
    if not basic.empty: pool = pool.merge(basic, on='ts_code', how='left')
    if 'name' not in pool.columns: pool['name'] = 'Unknown'
    d_basic = safe_get('daily_basic', trade_date=last_trade, fields='ts_code,turnover_rate,circ_mv,total_mv')
    if not d_basic.empty: pool = pool.merge(d_basic, on='ts_code', how='left')
    mf = safe_get('moneyflow', trade_date=last_trade)
    if not mf.empty and 'net_mf' in mf.columns:
        mf = mf[['ts_code', 'net_mf']].fillna(0)
        pool = pool.merge(mf, on='ts_code', how='left')
    for c in ['turnover_rate','circ_mv','net_mf']: 
        if c not in pool.columns: pool[c] = 0.0
    
    # 3. 基础过滤
    df = pool.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce') 
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000 
    df = df[~df['name'].str.contains('ST|退', case=False, na=False)]
    df = df[~df['ts_code'].str.startswith('92')] # 过滤北交所
    if 'list_date' in df.columns:
        df['days_listed'] = (datetime.strptime(last_trade, "%Y%m%d") - pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')).dt.days
        df = df[df['days_listed'] >= 120]
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE) & 
        (df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS) &
        (df['turnover_rate'] >= MIN_TURNOVER) & (df['turnover_rate'] <= 25.0) &
        (df['amount'] * 1000 >= MIN_AMOUNT)]
    if len(df) == 0: return pd.DataFrame(), f"过滤后无股票"

    limit_mf = int(FINAL_POOL * 0.5)
    df_mf = df.sort_values('net_mf', ascending=False).head(limit_mf)
    df_pct = df[~df['ts_code'].isin(df_mf['ts_code'])].sort_values('pct_chg', ascending=False).head(FINAL_POOL - len(df_mf))
    candidates = pd.concat([df_mf, df_pct]).reset_index(drop=True)
    if not GLOBAL_DAILY_RAW.empty:
        try:
            available = GLOBAL_DAILY_RAW.loc[(slice(None), last_trade), :].index.get_level_values('ts_code').unique()
            candidates = candidates[candidates['ts_code'].isin(available)]
        except: return pd.DataFrame(), "缓存缺失"

    # 4. 深度计算 (硬门槛)
    records = []
    for row in candidates.itertuples():
        ind = compute_indicators(row.ts_code, last_trade) 
        
        # [V30.20 硬门槛 - 保留]
        if ind.get('close_current', 0) <= ind.get('ma20_current', 0): continue
        if ind.get('vol_current', 0) <= ind.get('ma5_vol_current', 0) * 1.2: continue
        
        # [新增] RSI 过滤：防止接盘
        # 如果 RSI > 80，说明短期情绪极其过热，随时可能回调，放弃
        if ind.get('rsi_current', 50) > 80: continue

        if pd.isna(ind.get('macd_val')) or ind.get('macd_val') <= 0: continue
        
        # 传入 stop_loss_pct
        future = get_future_prices_real_combat(row.ts_code, last_trade, buy_threshold_pct=buy_threshold, stop_loss_pct=STOP_LOSS_PCT)
        
        records.append({
            'ts_code': row.ts_code, 'name': getattr(row, 'name', row.ts_code),
            'Close': row.close, 
            'Pct_Chg (%)': getattr(row, 'pct_chg', 0),
            'macd': ind['macd_val'], 
            'rsi': ind.get('rsi_current', 0), # 记录RSI
            'volatility': ind['volatility'],
            'Return_D1 (%)': future.get('Return_D1'), 
            'Return_D3 (%)': future.get('Return_D3'),
            'Return_D5 (%)': future.get('Return_D5')
        })
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), "无优质放量股票"

    # 5. [核心修改] 终极评分系统优化
    # 新逻辑：MACD 相对强度 (MACD / Close)，公平对待低价股
    fdf['macd_norm'] = fdf['macd'] / fdf['Close']
    base_score = fdf['macd_norm'] * 10000 
    
    # 动态加分
    def calculate_smart_bonus(row):
        bonus = 1.0
        tags = []
        
        # 涨停板确认 (强者恒强)
        if row['Pct_Chg (%)'] >= 9.5:
            bonus += 0.20
            tags.append('板确认')
            
        # 波动率适中 (4.0-8.0)
        if 4.0 <= row['volatility'] <= 8.0:
            bonus += 0.10
            tags.append('波适中')
        
        # [删除了价格加分] -> 避免歧视低价股
            
        return bonus, "+".join(tags)

    fdf[['bonus', '加分项']] = fdf.apply(lambda x: pd.Series(calculate_smart_bonus(x)), axis=1)
    fdf['综合评分'] = base_score * fdf['bonus']
    
    # 排序取前几名
    fdf = fdf.sort_values('综合评分', ascending=False).head(TOP_BACKTEST)
    return fdf.reset_index(drop=True), None

# ---------------------------
# 主程序
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日 V30.23 涅槃回测"):
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days: st.stop()
    if not get_all_historical_data(trade_days): st.stop()
    
    st.success("✅ V30.23 (涅槃版) 启动... 正在为您避开山顶...")
    results = []
    bar = st.progress(0)
    
    for i, date in enumerate(trade_days):
        try:
            df, msg = run_backtest_for_a_day(date, TOP_BACKTEST, FINAL_POOL, BUY_THRESHOLD_PCT)
            if not df.empty:
                df['Trade_Date'] = date
                results.append(df)
        except Exception: pass
        bar.progress((i + 1) / len(trade_days))
    bar.empty()
    
    if not results:
        st.error("无结果。")
        st.stop()
        
    all_res = pd.concat(results)
    if all_res['Trade_Date'].dtype != 'object': all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
    st.header(f"📊 V30.23 回测报告 (RSI优化 + 止损)")
    st.markdown(f"**有效交易天数：** {all_res['Trade_Date'].nunique()} 天 | **止损设置：** -{STOP_LOSS_PCT}%")

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
    st.dataframe(all_res.sort_values('Trade_Date', ascending=False), use_container_width=True)
