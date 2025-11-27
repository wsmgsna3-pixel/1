# -*- coding: utf-8 -*-
"""
选股王 · V9.8 稳定加速回退版 (保留 V9.7 逻辑 + 回退到 V9.0 稳定数据获取)

说明：
1. 【稳定回退】移除 V9.7 的批量历史数据预加载 (bulk_fetch_daily) 机制。
2. 【V9.0 稳定模式】在评分循环中，回归 V9.0 的逐个获取历史数据模式，确保数据不缺失。
3. 【策略逻辑】保持 V9.7 最终逻辑：严格市值防御 + 极限宽松流动性 + 风控因子改为软性评分。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import joblib 
import os

warnings.filterwarnings("ignore")

# ---------------------------
# 外部缓存配置 (用于历史数据)
# ---------------------------
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
# joblib 封装 Tushare 接口，实现磁盘持久化缓存
memory = joblib.Memory(CACHE_DIR, verbose=0)

# ---------------------------
# 页面设置 (UI 空间最大化)
# ---------------------------
st.set_page_config(page_title="选股王（V9.8 稳定加速回退版）", layout="wide")
st.markdown("### 选股王（V9.8 稳定加速回退版）") 

# ---------------------------
# 侧边栏参数（V9.8 策略：极限宽松流动性，保留风控参数用于评分）
# ---------------------------
with st.sidebar:
    st.header("可调参数（实时）")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=500, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    
    # V9.6 调整：极限宽松流动性
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=0.5, step=0.1)) 
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=20_000_000.0, step=5_000_000.0)) # 2000万
    
    # 风控参数保留，但仅用于评分 (不再硬性排除)
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.4, step=0.1)) 
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=6.0, step=0.5)) 
    
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    
    BACKTEST_DAYS = int(st.number_input("回测：最近 N 个交易日", value=10, step=1))
    st.markdown("---")
    st.caption("提示：策略已调整至 'V9.8 稳定回退' 模式，回退到 V9.0 的数据获取方式，预计运行成功。")

# ---------------------------
# Token 输入 (保留)
# ---------------------------
st.markdown("请输入 Tushare Token。")
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password", label_visibility="collapsed")

if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

# 初始化 tushare
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 依赖函数：数据安全获取和交易日查找 (保留)
# ---------------------------
def safe_get(func, **kwargs):
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def find_last_trade_day(max_days=20):
    today = datetime.now().date()
    for i in range(max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        # --- V9.8 稳定：这里确保能找到交易日 ---
        df = safe_get(pro.daily, trade_date=ds)
        if not df.empty:
            return ds
    return None

last_trade = find_last_trade_day()
if not last_trade:
    st.error("无法找到最近交易日，检查网络或 Token 权限。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")


# ----------------------------------------------------
# V9.8: 移除 bulk_fetch_daily 函数，不再使用批量预加载
# ----------------------------------------------------


# ----------------------------------------------------
# 按钮控制模块 (Session State 断点续传) (保留)
# ----------------------------------------------------
if 'run_selection' not in st.session_state: st.session_state['run_selection'] = False
if 'run_backtest' not in st.session_state: st.session_state['run_backtest'] = False
if 'backtest_status' not in st.session_state: 
    # V9.8: 移除 bulk_data 相关的状态
    st.session_state['backtest_status'] = {'progress': 0.0, 'results': [], 'current_index': 0, 'total_days': 0}

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 运行当日选股", use_container_width=True):
        st.session_state['run_selection'] = True
        st.session_state['run_backtest'] = False
        st.session_state['backtest_status'] = {'progress': 0.0, 'results': [], 'current_index': 0, 'total_days': 0}
        st.rerun()

with col2:
    if st.button(f"✅ 运行历史回测 ({BACKTEST_DAYS} 日)", use_container_width=True):
        st.session_state['run_backtest'] = True
        st.session_state['run_selection'] = False
        if st.session_state['backtest_status']['progress'] == 1.0 or st.session_state['backtest_status']['total_days'] == 0:
             st.session_state['backtest_status'] = {'progress': 0.0, 'results': [], 'current_index': 0, 'total_days': 0}
        st.rerun()

st.markdown("---")

# ---------------------------
# 指标计算和归一化 (保留)
# ---------------------------
def compute_indicators(df):
    res = {}
    if df.empty or len(df) < 3: return res
    close = df['close'].astype(float); high = df['high'].astype(float); low = df['low'].astype(float)
    try: res['last_close'] = close.iloc[-1]
    except: res['last_close'] = np.nan
    for n in (5,10,20):
        if len(close) >= n: res[f'ma{n}'] = close.rolling(window=n).mean().iloc[-1]
        else: res[f'ma{n}'] = np.nan
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_val = (diff - dea) * 2
        res['macd'] = macd_val.iloc[-1]; res['diff'] = diff.iloc[-1]; res['dea'] = dea.iloc[-1]
    else: res['macd'] = res['diff'] = res['dea'] = np.nan
    n = 9
    if len(close) >= n:
        low_n = low.rolling(window=n).min()
        high_n = high.rolling(window=n).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
        rsv = rsv.fillna(50)
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3*k - 2*d
        res['k'] = k.iloc[-1]; res['d'] = d.iloc[-1]; res['j'] = j.iloc[-1]
    else: res['k'] = res['d'] = res['j'] = np.nan
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        avg_prev5 = np.mean(vols[-6:-1])
        res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9)
        res['vol_last'] = vols[-1]; res['vol_ma5'] = avg_prev5
    else: res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan
    if len(close) >= 10: res['10d_return'] = close.iloc[-1] / close.iloc[-10] - 1
    else: res['10d_return'] = np.nan
    if 'pct_chg' in df.columns and len(df) >= 4:
        try: res['prev3_sum'] = df['pct_chg'].astype(float).iloc[-4:-1].sum()
        except: res['prev3_sum'] = np.nan
    else: res['prev3_sum'] = np.nan
    try:
        if 'pct_chg' in df.columns and len(df) >= 10:
            res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()
        else: res['volatility_10'] = np.nan
    except: res['volatility_10'] = np.nan
    return res

def safe_merge_pool(pool_df, other_df, cols):
    pool = pool_df.set_index('ts_code').copy()
    if other_df is None or other_df.empty:
        for c in cols: pool[c] = np.nan
        return pool.reset_index()
    if 'ts_code' not in other_df.columns:
        try: other_df = other_df.reset_index()
        except:
            for c in cols: pool[c] = np.nan
            return pool.reset_index()
    for c in cols:
        if c not in other_df.columns: other_df[c] = np.nan
    try: joined = pool.join(other_df.set_index('ts_code')[cols], how='left')
    except Exception:
        for c in cols: pool[c] = np.nan
        return pool.reset_index()
    for c in cols:
        if c not in joined.columns: joined[c] = np.nan
    return joined.reset_index()

def norm_col(s):
    s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    mn = s.min(); mx = s.max()
    if mx - mn < 1e-9: return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)

# ----------------------------------------------------
# 核心评分函数 (V9.8：数据获取逻辑回退 + 逻辑修正保留)
# ----------------------------------------------------
@memory.cache # V9.0 的缓存是加在整个评分函数上的
def run_scoring_for_date(trade_date, params):
    """
    V9.8 评分函数：回退到 V9.0 的数据获取模式，保留 V9.7 的逻辑修正和诊断。
    """
    
    # 解包参数
    initial_top_n, final_pool_limit, min_price, max_price, min_turnover, min_amount, vol_spike_mult, volatility_max, high_pct_threshold = \
        params['INITIAL_TOP_N'], params['FINAL_POOL'], params['MIN_PRICE'], params['MAX_PRICE'], \
        params['MIN_TURNOVER'], params['MIN_AMOUNT'], params['VOL_SPIKE_MULT'], \
        params['VOLATILITY_MAX'], params['HIGH_PCT_THRESHOLD']
    
    # 1. 拉取当日涨幅榜初筛
    daily_all = safe_get(pro.daily, trade_date=trade_date)
    daily_basic = safe_get(pro.daily_basic, trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    mf_raw = safe_get(pro.moneyflow, trade_date=trade_date)
    
    # --- 诊断 1：检查 Tushare 数据是否拉取成功 ---
    if daily_all.empty: 
        # Tushare pro.daily 失败是最大的问题
        st.error(f"诊断：Tushare 无法获取 {trade_date} 的日线数据，请检查 Token 权限或网络。")
        return pd.DataFrame()
    
    daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
    pool0 = daily_all.head(int(initial_top_n)).copy().reset_index(drop=True)

    # 2. 合并高级接口数据
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,total_mv,circ_mv')
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in','net_mf_out']
        col = next((c for c in possible if c in mf_raw.columns), None)
        if col: moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)
    
    if not stock_basic.empty:
        keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_basic.columns]
        try: pool0 = pool0.merge(stock_basic[keep], on='ts_code', how='left')
        except Exception: pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
    else: pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
        
    pool_merged = safe_merge_pool(pool0, daily_basic, ['turnover_rate','amount','total_mv','circ_mv'])
    
    if moneyflow.empty: moneyflow = pd.DataFrame({'ts_code': pool_merged['ts_code'].tolist(), 'net_mf': [0.0]*len(pool_merged)})
    try: pool_merged = pool_merged.set_index('ts_code').join(moneyflow.set_index('ts_code'), how='left').reset_index()
    except: pool_merged['net_mf'] = 0.0
    pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0.0)

    # --- 诊断 2：检查原始池大小 ---
    st.info(f"诊断：原始涨幅榜初筛并合并后，股票数量: **{len(pool_merged)}** 支。")
    
    # 3. 清洗 (只保留必要的硬性过滤：价格、ST、市值、流动性、当日上涨)
    
    # --- V9.8 严格市值防御 (800亿) ---
    MAX_TOTAL_MV_YUAN = 80000000000.0 
    pool_merged['total_mv_yuan'] = pool_merged['total_mv'].apply(
        lambda tv: tv * 10000.0 if not pd.isna(tv) and tv > 1e6 else tv)
    pool_merged['amount_yuan'] = pool_merged['amount'].apply(
        lambda amt: amt * 10000.0 if not pd.isna(amt) and amt > 0 and amt < 1e5 else amt)

    clean_df = pool_merged.copy()
    
    # 价格、ST、停牌过滤
    clean_df = clean_df[~(
        (clean_df['close'].isna()) | 
        (clean_df['close'] < min_price) | 
        (clean_df['close'] > max_price) | 
        (clean_df['name'].str.contains('ST|退', case=False, na=False))
    )]
    
    # 涨跌幅过滤 (当日上涨)
    clean_df = clean_df[~((clean_df['pct_chg'].isna()) | (clean_df['pct_chg'] < 0))]
    
    # V9.8 市值防御过滤 
    clean_df = clean_df[~((clean_df['total_mv_yuan'].notna()) & (clean_df['total_mv_yuan'] > MAX_TOTAL_MV_YUAN))]

    # V9.8 极限宽松流动性过滤 
    clean_df = clean_df[~((clean_df['turnover_rate'].isna()) | (clean_df['turnover_rate'] < min_turnover))]
    clean_df = clean_df[~((clean_df['amount_yuan'].isna()) | (clean_df['amount_yuan'] < min_amount))]
    
    # --- 诊断 3：检查硬性过滤后的数量 ---
    if clean_df.empty: 
        st.error(f"诊断：所有硬性过滤后，剩余股票数量为 **0** 支。")
        return pd.DataFrame()

    st.info(f"诊断：硬性过滤后，剩余股票数量: **{len(clean_df)}** 支，开始计算指标并评分...")

    score_pool_n = min(int(final_pool_limit), 300)
    clean_df = clean_df.sort_values('pct_chg', ascending=False).head(score_pool_n).reset_index(drop=True)
    
    # 4. 指标计算与评分
    records = []
    
    # V9.8: 恢复 V9.0 的逐个获取历史数据模式
    start_dt = datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=60 * 1.5) # 60天指标
    start_date_hist = start_dt.strftime("%Y%m%d")
    
    pbar = st.progress(0.0, text=f"正在计算 {len(clean_df)} 支股票的指标...")

    for i, row in enumerate(clean_df.itertuples()):
        ts_code = getattr(row, 'ts_code'); pct_chg = getattr(row, 'pct_chg', 0.0);
        turnover_rate = getattr(row, 'turnover_rate', np.nan); net_mf = float(getattr(row, 'net_mf', 0.0));
        amount_raw = getattr(row, 'amount', np.nan)
        amount = amount_raw * 10000.0 if not pd.isna(amount_raw) and amount_raw > 0 and amount_raw < 1e5 else amount_raw
        amount = amount if not pd.isna(amount) else 0.0
        name = getattr(row, 'name', ts_code)

        # 核心：V9.8 稳定模式：逐个获取历史数据
        hist = safe_get(pro.daily, ts_code=ts_code, start_date=start_date_hist, end_date=trade_date)
        
        ind = compute_indicators(hist)

        vol_ratio, ten_return, macd, k, d, j, vol_last, vol_ma5, prev3_sum, volatility_10, ma20, last_close = \
            ind.get('vol_ratio', np.nan), ind.get('10d_return', np.nan), ind.get('macd', np.nan), \
            ind.get('k', np.nan), ind.get('d', np.nan), ind.get('j', np.nan), \
            ind.get('vol_last', np.nan), ind.get('vol_ma5', np.nan), ind.get('prev3_sum', np.nan), ind.get('volatility_10', np.nan), ind.get('ma20', np.nan), ind.get('last_close', np.nan)

        try: proxy_money = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
        except: proxy_money = 0.0

        rec = {'ts_code': ts_code, 'pct_chg': pct_chg, 'turnover_rate': turnover_rate, 'net_mf': net_mf, 'amount': amount,
               'vol_ratio': vol_ratio, '10d_return': ten_return, 'macd': macd, 'k': k, 'd': d, 'j': j,
               'vol_last': vol_last, 'vol_ma5': vol_ma5, 'prev3_sum': prev3_sum, 'volatility_10': volatility_10,
               'proxy_money': proxy_money, 'name': name,
               'last_close': last_close, 'ma20': ma20}
        records.append(rec)
        
        pbar.progress((i + 1) / len(clean_df), text=f"指标计算进度：[{i+1}/{len(clean_df)}]...")
        
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame()
    pbar.empty()

    # 5. 风险过滤 (V9.8 修正：将所有风控硬性过滤逻辑移除，仅保留到评分阶段)
    # 6. RSL & 归一化 (逻辑不变)
    if '10d_return' in fdf.columns:
        try:
            market_mean_10d = fdf['10d_return'].replace([np.inf,-np.inf], np.nan).dropna().mean()
            fdf['rsl'] = fdf['10d_return'] / (market_mean_10d if abs(market_mean_10d) >= 1e-9 else 1e-9)
        except: fdf['rsl'] = 1.0
    else: fdf['rsl'] = 1.0

    fdf['s_pct'] = norm_col(fdf.get('pct_chg', pd.Series([0]*len(fdf))))
    fdf['s_volratio'] = norm_col(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))
    fdf['s_turn'] = norm_col(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))
    fdf['s_money'] = norm_col(fdf.get('net_mf', pd.Series([0]*len(fdf)))) if fdf['net_mf'].abs().sum() > 0 else norm_col(fdf.get('proxy_money', pd.Series([0]*len(fdf))))
    fdf['s_amount'] = norm_col(fdf.get('amount', pd.Series([0]*len(fdf))))
    fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))
    fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
    fdf['s_rsl'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))
    
    fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))

    # 7. 综合评分 (V9.0 权重调整：极限防御：w_turn (0.35), w_volatility (0.25))
    w_pct, w_volratio, w_turn, w_money, w_10d, w_macd, w_rsl, w_volatility = 0.05, 0.10, 0.35, 0.10, 0.05, 0.10, 0.05, 0.25
    
    fdf['综合评分'] = (fdf['s_pct'] * w_pct + fdf['s_volratio'] * w_volratio + fdf['s_turn'] * w_turn + fdf['s_money'] 
        * w_money + fdf['s_10d'] * w_10d + fdf['s_macd'] * w_macd + fdf['s_rsl'] * w_rsl + fdf['s_volatility'] * w_volatility)
    
    return fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)


# ----------------------------------------------------
# 简易回测模块 (V9.8：移除批量数据依赖)
# ----------------------------------------------------
def run_simple_backtest(days, params):
    status = st.session_state['backtest_status']
    
    container = st.empty()
    with container.container():
        st.subheader("📈 简易历史回测结果")
        
        # 1. 获取交易日历
        trade_dates_df = safe_get(pro.trade_cal, exchange='SSE', is_open='1', end_date=find_last_trade_day(), fields='cal_date')
        if trade_dates_df.empty:
            st.error("无法获取历史交易日历。")
            return

        trade_dates = trade_dates_df['cal_date'].sort_values(ascending=False).head(days + 1).tolist()
        trade_dates.reverse() 
        total_iterations = len(trade_dates) - 1
        
        if total_iterations < 1:
            st.warning("交易日不足，无法进行回测。")
            return
            
        status['total_days'] = total_iterations
        
        # V9.8: 移除数据预加载逻辑
        start_index = status['current_index']
        
        if start_index >= total_iterations:
             st.success(f"回测已完成。累计收益率请查看下方。")
        else:
             st.info(f"回测周期：**{trade_dates[0]}** 至 **{trade_dates[-2]}**。正在从第 {start_index+1} 天继续...")

        pbar = st.progress(status['progress'], text=f"回测进度：[{status['current_index']}/{status['total_days']}]...")
        
        # 4. 参数打包
        params_dict = {
            'INITIAL_TOP_N': params['INITIAL_TOP_N'], 'FINAL_POOL': params['FINAL_POOL'], 'MIN_PRICE': params['MIN_PRICE'], 
            'MAX_PRICE': params['MAX_PRICE'], 'MIN_TURNOVER': params['MIN_TURNOVER'], 'MIN_AMOUNT': params['MIN_AMOUNT'], 
            'VOL_SPIKE_MULT': params['VOL_SPIKE_MULT'], 'VOLATILITY_MAX': params['VOLATILITY_MAX'], 
            'HIGH_PCT_THRESHOLD': params['HIGH_PCT_THRESHOLD']
        }
        
        for i in range(start_index, total_iterations):
            select_date = trade_dates[i]
            next_trade_date = trade_dates[i+1]
            
            # 核心步骤：调用 V9.8 评分函数，无需传入 bulk_data
            select_df_full = run_scoring_for_date(select_date, params_dict) 

            # T+1 收益计算逻辑 (不变)
            return_pct = 0.0
            buy_price, sell_price = np.nan, np.nan

            if select_df_full.empty:
                result = {'选股日': select_date, '股票': '无符合条件', 'T+1 收益率': 0.0, '买入价 (T+1 开盘)': np.nan, '卖出价 (T+1 收盘)': np.nan, '评分': np.nan}
            else:
                top_pick = select_df_full.iloc[0] 
                ts_code = top_pick['ts_code']
                
                next_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=next_trade_date)
                

                if not next_day_data.empty and 'open' in next_day_data.columns and 'close' in next_day_data.columns:
                    buy_price = next_day_data.iloc[0]['open']
                    sell_price = next_day_data.iloc[0]['close']
                    
                    if buy_price > 0 and not pd.isna(sell_price):
                        return_pct = (sell_price / buy_price) - 1.0

                result = {
                    '选股日': select_date,
                    '股票': f"{top_pick.get('name', 'N/A')}({ts_code})",
                    'T+1 收益率': return_pct * 100,
                    '买入价 (T+1 开盘)': buy_price,
                    '卖出价 (T+1 收盘)': sell_price,
                    '评分': top_pick['综合评分']
                }

            # 5. 更新状态和进度条
            status['results'].append(result)
            status['current_index'] = i + 1
            status['progress'] = (i + 1) / total_iterations
            
            pbar.progress(status['progress'], text=f"正在回测 {select_date}... [{i+1}/{total_iterations}]")
            
            if (i+1) % 2 == 0: 
                 st.rerun() 
        
        status['progress'] = 1.0
        status['current_index'] = total_iterations
        pbar.progress(1.0, text="回测完成。")
        
        # 6. 结果展示 (不变)
        results_df = pd.DataFrame(status['results'])
        
        if results_df.empty:
            st.warning("回测结果为空。")
            return
            
        results_df['T+1 收益率'] = results_df['T+1 收益率'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        cumulative_return = (results_df['T+1 收益率'] / 100 + 1).product() - 1
        wins = (results_df['T+1 收益率'] > 0).sum()
        total_trades = len(results_df)
        win_rate = wins / total_trades if total_trades > 0 else 0

        st.markdown("---")
        st.subheader("💡 最终回测指标")
        colA, colB, colC = st.columns(3)
        colA.metric("累计收益率 (T+1)", f"{cumulative_return*100:.2f}%")
        colB.metric("胜率", f"{win_rate*100:.2f}%")
        colC.metric("交易次数", f"{total_trades}")
        
        st.subheader("📋 每日交易记录")
        st.dataframe(results_df, use_container_width=True)

# ----------------------------------------------------
# 实时选股模块 (V9.8：移除批量数据依赖)
# ----------------------------------------------------
def run_live_selection(last_trade, params):
    st.write(f"正在运行实时选股（最近交易日：{last_trade}）...")
    
    # V9.8: 移除预加载逻辑

    # 5. 调用 V9.8 评分
    params_dict = {
        'INITIAL_TOP_N': params['INITIAL_TOP_N'], 'FINAL_POOL': params['FINAL_POOL'], 'MIN_PRICE': params['MIN_PRICE'], 
        'MAX_PRICE': params['MAX_PRICE'], 'MIN_TURNOVER': params['MIN_TURNOVER'], 'MIN_AMOUNT': params['MIN_AMOUNT'], 
        'VOL_SPIKE_MULT': params['VOL_SPIKE_MULT'], 'VOLATILITY_MAX': params['VOLATILITY_MAX'], 
        'HIGH_PCT_THRESHOLD': params['HIGH_PCT_THRESHOLD']
    }
    # V9.8: 只传参数
    fdf_full = run_scoring_for_date(last_trade, params_dict)

    if fdf_full.empty:
        st.error("清洗和评分后没有候选。请参考上方的诊断信息，检查是 Tushare API 问题还是过滤条件过于严格。")
        st.stop()

    fdf = fdf_full.head(params['TOP_DISPLAY']).copy()
    fdf.index = fdf.index + 1

    st.success(f"评分完成：总候选 {len(fdf_full)} 支，显示 Top {min(params['TOP_DISPLAY'], len(fdf))}。")
    display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','proxy_money','amount','10d_return','macd','k','d','j','rsl','volatility_10']
    for c in display_cols:
        if c not in fdf.columns: fdf[c] = np.nan

    st.dataframe(fdf[display_cols], use_container_width=True)

    out_csv = fdf_full[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

    st.markdown("### 小结与操作提示（简洁）")
    st.markdown("""
- **【策略风格】** 本版本为 **V9.8 稳定回退版**，同时拥有 V9.0 的数据获取稳定性，以及 V9.7 的策略防御修正。
- **【风控提示】** 风控指标（波动率、放量倍数等）已全部纳入**评分体系**。
- **【重要纪律】** 9:40 前不买 → 观察 9:40-10:05 的量价节奏 → 10:05 后择优介入。
""")


# ----------------------------------------------------
# 主程序控制逻辑 (保留)
# ----------------------------------------------------
params = {
    'INITIAL_TOP_N': INITIAL_TOP_N, 'FINAL_POOL': FINAL_POOL, 'TOP_DISPLAY': TOP_DISPLAY,
    'MIN_PRICE': MIN_PRICE, 'MAX_PRICE': MAX_PRICE, 'MIN_TURNOVER': MIN_TURNOVER,
    'MIN_AMOUNT': MIN_AMOUNT, 'VOL_SPIKE_MULT': VOL_SPIKE_MULT, 'VOLATILITY_MAX': VOLATILITY_MAX,
    'HIGH_PCT_THRESHOLD': HIGH_PCT_THRESHOLD
}

if st.session_state.get('run_backtest', False):
    run_simple_backtest(BACKTEST_DAYS, params)
    
elif st.session_state.get('run_selection', False):
    run_live_selection(last_trade, params)
    
else:
    st.info("请点击上方的按钮开始运行。")
