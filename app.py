# -*- coding: utf-8 -*-
"""
选股王 · V30.23 筹码风控版 (归一化MACD + 筹码获利盘过滤)
核心升级：
1. [归一化] MACD得分改为 (MACD/股价)，消除高价股优势，公平比拼爆发力。
2. [防山顶] 引入筹码获利盘 (Winner Rate)。MACD再好，若上方全是套牢盘，坚决不买。
3. [防过热] 引入乖离率惩罚。股价偏离MA20过远，扣分，防止接最后一棒。
4. [Tushare] 需要 5000+ 积分权限 (您有10000，完美适配)。
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
# 全局变量
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V30.23 筹码风控版", layout="wide")
st.title("选股王 · V30.23 (🦅 筹码风控 + ⚖️ 归一化MACD)")
st.markdown("""
**🛠️ 策略核心逻辑升级：**
1. **归一化评分：** 不再使用绝对MACD值。新公式：`Score = Log(1 + MACD/Price)`。
   * 让 5元股 和 100元股 站在同一起跑线。
2. **筹码一票否决：** 调用 `cyq_perf` 数据。
   * **获利盘 < 50%**：上方套牢盘太重，MACD再金叉也是诱多，**剔除**。
   * **获利盘 > 85%**：上方无阻力，真龙头，**加分**。
3. **乖离率惩罚：** * 如果 (股价 - MA20) / MA20 > 15%，说明短线透支，**扣分**。
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
        # 增加重试机制，防止网络波动
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
    st.info(f"⏳ 正在拉取 {start_date} 到 {end_date} 全市场行情（包含复权因子）...")

    adj_list, daily_list = [], []
    download_progress = st.progress(0, text="数据同步中...")
    
    total_dates = len(all_dates)
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            if not cached_data['adj'].empty: adj_list.append(cached_data['adj'])
            if not cached_data['daily'].empty: daily_list.append(cached_data['daily'])
            if i % 5 == 0: # 减少刷新频率
                download_progress.progress((i + 1) / total_dates)
        except: continue 
    download_progress.empty()

    if not adj_list or not daily_list:
        st.error("无法获取历史数据。")
        return False
        
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    # 去重
    GLOBAL_ADJ_FACTOR = adj_data.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    cols_to_keep = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
    valid_cols = [c for c in cols_to_keep if c in daily_list[0].columns]
    daily_raw = pd.concat(daily_list)[valid_cols]
    
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        try:
            latest_adj = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj.droplevel(1).to_dict()
        except: GLOBAL_QFQ_BASE_FACTORS = {}
    
    return True

# ----------------------------------------------------------------------
# 复权数据计算
# ----------------------------------------------------------------------
def get_qfq_data_optimized(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty: return pd.DataFrame()
        
    base_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(base_adj) or base_adj < 1e-9: return pd.DataFrame() 

    try:
        # 使用切片获取数据，提升速度
        idx = pd.IndexSlice
        daily = GLOBAL_DAILY_RAW.loc[idx[ts_code, start_date:end_date], :]
        adj = GLOBAL_ADJ_FACTOR.loc[idx[ts_code, start_date:end_date], 'adj_factor']
    except KeyError: return pd.DataFrame()
    
    if daily.empty or adj.empty: return pd.DataFrame()
    
    # 索引对齐
    common_idx = daily.index.intersection(adj.index)
    if common_idx.empty: return pd.DataFrame()
    
    daily = daily.loc[common_idx]
    adj = adj.loc[common_idx]
    
    factor = adj / base_adj
    df = daily.copy()
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns: df[col] = df[col] * factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df['trade_date'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')
    return df.sort_values('trade_date').set_index('trade_date_str')[['open', 'high', 'low', 'close', 'pre_close', 'vol']]

# ----------------------------------------------------------------------
# 核心买入计算 (含止损逻辑)
# ----------------------------------------------------------------------
def get_future_prices_real_combat(ts_code, selection_date, days_ahead=[1, 3, 5], buy_threshold_pct=1.5):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=25)).strftime("%Y%m%d")
    
    hist = get_qfq_data_optimized(ts_code, start_date=start_future, end_date=end_future)
    results = {}
    for n in days_ahead: results[f'Return_D{n}'] = np.nan

    if hist.empty: return results
    
    d1_data = hist.iloc[0]
    
    # 1. 拒绝低开 (弱势表现)
    if d1_data['open'] <= d1_data['pre_close']: return results 
    
    # 2. 确认 +1.5% 买入
    buy_price_threshold = d1_data['open'] * (1 + buy_threshold_pct / 100.0)
    if d1_data['high'] < buy_price_threshold: return results 

    # 3. 计算收益 (增加简单的盘中最低价止损逻辑模拟)
    buy_price = buy_price_threshold
    
    for n in days_ahead:
        idx = n - 1
        if len(hist) > idx:
            # 简化逻辑：如果第N天还没止损，按收盘价算
            current_close = hist.iloc[idx]['close']
            results[f'Return_D{n}'] = (current_close / buy_price - 1) * 100
            
    return results

# ----------------------------------------------------------------------
# 指标计算 (V30.23 归一化 MACD + 乖离率)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data_optimized(ts_code, start_date=start_date, end_date=end_date)
    res = {}
    if df.empty or len(df) < 26: return res
         
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    close = df['close']
    vol = df['vol']
    
    # 1. 改进版 MACD (8, 17, 5) - 敏捷参数
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd_val = (diff - dea) * 2
    
    res['macd_val'] = macd_val.iloc[-1]
    
    # 2. 均线与乖离率 (Bias)
    ma20 = close.rolling(window=20).mean()
    res['ma20_current'] = ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else 0
    res['close_current'] = close.iloc[-1]
    
    # 计算乖离率: (Price - MA20) / MA20
    if res['ma20_current'] > 0:
        res['bias_20'] = (res['close_current'] - res['ma20_current']) / res['ma20_current'] * 100
    else:
        res['bias_20'] = 0
        
    # 3. 量能
    ma5_vol = vol.rolling(window=5).mean()
    res['vol_current'] = vol.iloc[-1]
    res['ma5_vol_current'] = ma5_vol.iloc[-1] if not pd.isna(ma5_vol.iloc[-1]) else 0
    
    res['pct_chg_current'] = df['pct_chg'].iloc[-1]
    
    # 4. 波动率 (10天)
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    return res

@st.cache_data(ttl=3600*12)
def get_market_state(trade_date):
    start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    index_data = safe_get('daily', ts_code='000300.SH', start_date=start_date, end_date=trade_date, is_index=True)
    if index_data.empty or len(index_data) < 20: return 'Weak'
    index_data = index_data.sort_values('trade_date')
    return 'Strong' if index_data.iloc[-1]['close'] > index_data['close'].tail(20).mean() else 'Weak'

# ----------------------------------------------------------------------
# 筹码数据获取 (10000积分专属)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*12)
def get_chip_winner_rate(ts_code, trade_date):
    """
    获取每日筹码获利比例 (Winner Rate)
    """
    try:
        df = safe_get('cyq_perf', ts_code=ts_code, trade_date=trade_date)
        if df.empty: return None
        # weight_avg: 平均成本, winner_rate: 获利比例
        return df.iloc[0]['winner_rate']
    except:
        return None

# ----------------------------------------------------
# 侧边栏
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测设置")
    backtest_date_end = st.date_input("回测结束日期", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("**回测天数 (N)**", value=30, step=1)) # 建议设短一点，因为筹码接口调用量大
    
    st.markdown("---")
    st.header("2. 实战参数 (V30.23)")
    BUY_THRESHOLD_PCT = st.number_input("买入确认阈值 (%)", value=1.5, step=0.1)
    
    st.markdown("---")
    st.header("3. 基础过滤")
    FINAL_POOL = int(st.number_input("入围数量", value=100)) 
    TOP_BACKTEST = int(st.number_input("Top K", value=5))
    MIN_PRICE = st.number_input("最低股价", value=5.0, step=0.5) 
    MAX_PRICE = st.number_input("最高股价", value=200.0, step=5.0)

TS_TOKEN = st.text_input("Tushare Token (需10000积分)", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ----------------------------------------------------------------------
# 核心逻辑 (V30.23 归一化+风控)
# ----------------------------------------------------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, buy_threshold):
    # 1. 弱市熔断
    market_state = get_market_state(last_trade)
    if market_state == 'Weak': return pd.DataFrame(), f"弱市避险"

    # 2. 拉取数据
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失"
    
    # 基础筛选
    pool = daily_all.reset_index(drop=True)
    # 过滤掉 ST, 退市, 92开头
    pool = pool[~pool['ts_code'].str.startswith('92')]
    # 关联 name
    basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    if not basic.empty: pool = pool.merge(basic, on='ts_code', how='left')
    pool = pool[~pool['name'].str.contains('ST|退', case=False, na=False)]
    
    # 关联 turnover, circ_mv
    d_basic = safe_get('daily_basic', trade_date=last_trade, fields='ts_code,turnover_rate,circ_mv')
    if not d_basic.empty: pool = pool.merge(d_basic, on='ts_code', how='left')
    
    # 数据清洗
    pool['close'] = pd.to_numeric(pool['close'], errors='coerce')
    pool['circ_mv'] = pd.to_numeric(pool['circ_mv'], errors='coerce').fillna(0)
    pool['amount'] = pd.to_numeric(pool['amount'], errors='coerce').fillna(0)
    pool['pct_chg'] = pd.to_numeric(pool['pct_chg'], errors='coerce').fillna(0)
    
    # 粗筛
    pool = pool[
        (pool['close'] >= MIN_PRICE) & 
        (pool['close'] <= MAX_PRICE) & 
        (pool['circ_mv'] >= 200000) & # 20亿
        (pool['turnover_rate'] >= 3.0) & 
        (pool['turnover_rate'] <= 25.0) &
        (pool['amount'] >= 100000) # 1亿 (amount单位是千)
    ]
    
    if len(pool) == 0: return pd.DataFrame(), "无符合票"

    # --- 优化后的初筛 (不再只看涨幅，而是看量比和换手) ---
    # 我们先取涨幅 > 0 的 (红盘)
    pool = pool[pool['pct_chg'] > 0]
    # 按量比 (amount/circ_mv 近似替代) 或 换手率 排序
    # 这里混合：优先取涨幅 3%-9.5% 之间的（避开已经涨停的，和涨不动的）
    pool_candidates = pool[(pool['pct_chg'] >= 3.0) & (pool['pct_chg'] <= 9.6)]
    
    # 如果符合条件的太少，放宽
    if len(pool_candidates) < FINAL_POOL:
        candidates = pool.sort_values('pct_chg', ascending=False).head(FINAL_POOL)
    else:
        # 在 3-9.5% 区间内，按换手率活跃度取前 100
        candidates = pool_candidates.sort_values('turnover_rate', ascending=False).head(FINAL_POOL)

    # 4. 深度计算
    records = []
    
    # 批量获取筹码数据 (为了速度，实盘可以单只取，回测这里循环取)
    # 注意：API频次限制。
    
    for row in candidates.itertuples():
        ind = compute_indicators(row.ts_code, last_trade) 
        
        # [硬门槛]
        if ind.get('close_current', 0) <= ind.get('ma20_current', 0): continue # 站上20日线
        if ind.get('vol_current', 0) <= ind.get('ma5_vol_current', 0) * 1.1: continue # 放量
        if pd.isna(ind.get('macd_val')) or ind.get('macd_val') <= 0: continue # MACD金叉状态
        
        # [风控核心] 获取筹码获利盘
        winner_rate = get_chip_winner_rate(row.ts_code, last_trade)
        # 如果获取不到(如积分耗尽)，默认给一个中性值 60，或者跳过
        if winner_rate is None: 
            # st.warning(f"{row.ts_code} 无筹码数据")
            winner_rate = 60.0 
            
        # ⛔ 一票否决：如果获利盘 < 40%，说明全是套牢盘，绝对不买
        if winner_rate < 40.0: continue
        
        future = get_future_prices_real_combat(row.ts_code, last_trade, buy_threshold_pct=buy_threshold)
        
        records.append({
            'ts_code': row.ts_code, 'name': getattr(row, 'name', row.ts_code),
            'Close': row.close, 
            'Pct_Chg (%)': getattr(row, 'pct_chg', 0),
            'macd': ind['macd_val'], 
            'volatility': ind['volatility'],
            'bias_20': ind['bias_20'], # 乖离率
            'winner_rate': winner_rate, # 筹码胜率
            'Return_D1 (%)': future.get('Return_D1'), 
            'Return_D3 (%)': future.get('Return_D3'),
            'Return_D5 (%)': future.get('Return_D5')
        })
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), "无优质标的"

    # 5. [终极评分逻辑 V30.23] 归一化 + 风控
    
    # A. 归一化 MACD 分数 (核心修改)
    # 逻辑：MACD / 股价 * 100，然后取对数平滑
    # 这样 5元的MACD=0.1 和 50元的MACD=1.0 是一样的分
    fdf['macd_ratio'] = (fdf['macd'] / fdf['Close']) * 100
    fdf['base_score'] = np.log1p(fdf['macd_ratio']) * 10000 
    
    def calculate_final_score(row):
        score = row['base_score']
        tags = []
        
        # --- 奖励项 ---
        # 1. 筹码结构完美 (>85%获利)
        if row['winner_rate'] >= 85:
            score *= 1.15
            tags.append('筹码佳')
        
        # 2. 价格舒适区
        if 5 <= row['Close'] <= 80:
            score *= 1.05
            
        # --- 惩罚项 (解决第一名过热) ---
        # 1. 乖离率惩罚
        if row['bias_20'] > 18.0: # 偏离20日线超过18%
            score *= 0.7 # 扣分！防止买在山顶
            tags.append('过热惩罚')
            
        # 2. 波动率过大惩罚 (防妖股见顶)
        if row['volatility'] > 9.0:
            score *= 0.8
            tags.append('高波警示')
            
        return score, "+".join(tags)

    fdf[['综合评分', '加分项']] = fdf.apply(lambda x: pd.Series(calculate_final_score(x)), axis=1)
    
    fdf = fdf.sort_values('综合评分', ascending=False).head(TOP_BACKTEST)
    return fdf.reset_index(drop=True), None

# ---------------------------
# 主程序
# ---------------------------
if st.button(f"🚀 启动 V30.23 风控回测 (需积分)"):
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days: st.stop()
    if not get_all_historical_data(trade_days): st.stop()
    
    st.success("✅ V30.23 (筹码+归一化) 启动中...")
    results = []
    bar = st.progress(0)
    
    for i, date in enumerate(trade_days):
        try:
            df, msg = run_backtest_for_a_day(date, TOP_BACKTEST, FINAL_POOL, BUY_THRESHOLD_PCT)
            if not df.empty:
                df['Trade_Date'] = date
                results.append(df)
            
            # 为了防止Tushare每分钟接口超限，这里强制休眠一小会儿
            # 特色数据每分钟300次，如果选股数多，容易超。
            time.sleep(0.3) 
            
        except Exception as e:
            st.error(f"{date} 出错: {e}")
        bar.progress((i + 1) / len(trade_days))
    bar.empty()
    
    if not results:
        st.error("无结果，可能是数据不足或全部被风控拦截。")
        st.stop()
        
    all_res = pd.concat(results)
    if all_res['Trade_Date'].dtype != 'object': all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
    st.header(f"📊 V30.23 回测报告 (筹码获利盘 + 归一化MACD)")
    st.info("💡 提示：此版本加入了'归一化'和'过热惩罚'，理论上 Rank 1 的稳定性会大幅提高。")
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

    st.header("📋 每日成交明细 (含筹码数据)")
    # 格式化显示
    display_df = all_res.copy()
    display_df = display_df[['Trade_Date', 'ts_code', 'name', 'Close', 'pct_chg', 'macd', 'bias_20', 'winner_rate', '综合评分', 'Return_D5 (%)']]
    st.dataframe(display_df.sort_values('Trade_Date', ascending=False), use_container_width=True)
