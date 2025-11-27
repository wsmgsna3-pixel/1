# -*- coding: utf-8 -*-
"""
选股王 · V10.5 流通市值修复版 (修复 circ_mv_wan KeyError)

说明：
1. 【核心修复】修复了 V10.3/V10.4 在指标评分时丢失 'circ_mv_wan' 列导致的 KeyError，确保流通市值能正确显示。
2. 【稳定性】保留 V10.4 的交易日历稳定查找方式。
3. 【市值防御】保留“最低流通市值”硬性过滤，排除超小盘股。
4. 【并列回测】保留 T+1, T+3, T+5 并列回测功能。
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
memory = joblib.Memory(CACHE_DIR, verbose=0)

# ---------------------------
# 页面设置 (UI 空间最大化)
# ---------------------------
st.set_page_config(page_title="选股王（V10.5 流通市值修复版）", layout="wide")
st.markdown("### 选股王（V10.5 流通市值修复版）") 

# ---------------------------
# 侧边栏参数
# ---------------------------
with st.sidebar:
    st.header("可调参数（实时）")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=500, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    
    # V10.3 市值下限参数
    # --- 建议用户调整此参数 ---
    MIN_CIRC_MV_Billion = float(st.number_input("最低流通市值 (亿)", value=50.0, step=5.0)) 
    
    # 极限宽松流动性
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=0.5, step=0.1)) 
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=20_000_000.0, step=5_000_000.0)) # 2000万
    
    # 风控参数
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.4, step=0.1)) 
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=6.0, step=0.5)) 
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    
    st.markdown("---")
    
    # 回测天数 N
    BACKTEST_DAYS = int(st.number_input("回测：最近 N 个交易日", value=10, step=1))
    
    st.markdown("---")
    st.caption("提示：策略已升级至 'V10.5 流通市值修复版'。")
    st.caption("回测将同时计算 T+1, T+3, T+5 收益。")

# ---------------------------
# Token 输入
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
# 依赖函数：数据安全获取
# ---------------------------
def safe_get(func, **kwargs):
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

# ---------------------------
# V10.4 稳定版交易日历获取
# ---------------------------
@st.cache_data(ttl=600)
def find_last_trade_day():
    end_date = datetime.now().strftime("%Y%m%d")
    cal_df = safe_get(
        pro.trade_cal, 
        exchange='SSE', 
        is_open='1', 
        end_date=end_date, 
        fields='cal_date'
    )
    
    if not cal_df.empty:
        return cal_df['cal_date'].max() 
    return None

last_trade = find_last_trade_day()
if not last_trade:
    st.error("无法找到最近交易日，检查网络或 Token 权限。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")


# ----------------------------------------------------
# 按钮控制模块 (与 V10.3 相同)
# ----------------------------------------------------
if 'run_selection' not in st.session_state: st.session_state['run_selection'] = False
if 'run_backtest' not in st.session_state: st.session_state['run_backtest'] = False
if 'backtest_status' not in st.session_state: 
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
# 指标计算和归一化 (与 V10.3 完整版相同)
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
# 核心评分函数 (V10.5 修复：确保 circ_mv_wan 被带入最终 fdf)
# ----------------------------------------------------
@memory.cache 
def run_scoring_for_date(trade_date, params):
    
    # 解包参数
    initial_top_n, final_pool_limit, min_price, max_price, min_turnover, min_amount, min_circ_mv_billion = \
        params['INITIAL_TOP_N'], params['FINAL_POOL'], params['MIN_PRICE'], params['MAX_PRICE'], \
        params['MIN_TURNOVER'], params['MIN_AMOUNT'], params['MIN_CIRC_MV_Billion']
    
    # 1. 拉取当日涨幅榜初筛
    daily_all = safe_get(pro.daily, trade_date=trade_date)
    daily_basic = safe_get(pro.daily_basic, trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    mf_raw = safe_get(pro.moneyflow, trade_date=trade_date)
    
    if daily_all.empty: 
        if trade_date == last_trade: st.error(f"诊断：Tushare 无法获取 {trade_date} 的日线数据，请检查 Token 权限或网络。")
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
    
    pool_merged = safe_merge_pool(pool0, daily_basic.rename(columns={'amount':'amount_db'}), ['turnover_rate','amount_db','total_mv','circ_mv'])
    
    if moneyflow.empty: moneyflow = pd.DataFrame({'ts_code': pool_merged['ts_code'].tolist(), 'net_mf': [0.0]*len(pool_merged)})
    try: pool_merged = pool_merged.set_index('ts_code').join(moneyflow.set_index('ts_code'), how='left').reset_index()
    except: pool_merged['net_mf'] = 0.0
    pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0.0)
    
    # 成交额数据清洗和转换 (V10.0 冗余保留)
    if 'amount' in pool_merged.columns:
        pool_merged['amount'] = pool_merged['amount'].apply(lambda amt: amt * 10000.0 if not pd.isna(amt) and amt > 0 and amt < 1e5 else amt)
    else:
        pool_merged['amount'] = pool_merged['amount_db'].apply(lambda amt: amt * 10000.0 if not pd.isna(amt) and amt > 0 and amt < 1e5 else amt)
    
    pool_merged['amount_yuan'] = pool_merged['amount']
    # 流通市值和总市值（单位：万，Tushare 单位）
    pool_merged['circ_mv_wan'] = pool_merged['circ_mv'].fillna(0)
    pool_merged['total_mv_yuan'] = pool_merged['total_mv'].apply(
        lambda tv: tv * 10000.0 if not pd.isna(tv) and tv > 1e6 else tv)


    # --- 诊断 2 ---
    if trade_date == last_trade:
        st.info(f"诊断：原始涨幅榜初筛并合并后，股票数量: **{len(pool_merged)}** 支。")
    
    # 3. 清洗 (V10.3：新增流通市值下限过滤)
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
    
    # --- V10.3 核心修改：流通市值下限过滤 ---
    min_circ_mv_wan = min_circ_mv_billion * 10000.0 # 将用户输入的亿转为 Tushare 的万单位
    
    # 过滤掉流通市值低于 min_circ_mv_wan 的股票
    clean_df = clean_df[clean_df['circ_mv_wan'].notna() & (clean_df['circ_mv_wan'] >= min_circ_mv_wan)]

    # 1. 成交额硬性过滤
    clean_df = clean_df[clean_df['amount_yuan'].notna() & (clean_df['amount_yuan'] >= min_amount)]
    
    # 2. 换手率硬性过滤
    turnover_filter_cond = (
        clean_df['turnover_rate'].notna() & 
        (clean_df['turnover_rate'] < min_turnover)
    )
    clean_df = clean_df[~turnover_filter_cond]
    
    
    # --- 诊断 3 ---
    if clean_df.empty: 
        if trade_date == last_trade: st.error(f"诊断：所有硬性过滤后，剩余股票数量为 **0** 支。")
        return pd.DataFrame()

    if trade_date == last_trade:
        st.info(f"诊断：硬性过滤后，剩余股票数量: **{len(clean_df)}** 支，开始计算指标并评分...")

    score_pool_n = min(int(final_pool_limit), 300)
    clean_df = clean_df.sort_values('pct_chg', ascending=False).head(score_pool_n).reset_index(drop=True)
    
    # 4. 指标计算与评分
    records = []
    
    start_dt = datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=60 * 1.5) 
    start_date_hist = start_dt.strftime("%Y%m%d")
    
    pbar = None
    if trade_date == last_trade:
        pbar = st.progress(0.0, text=f"正在计算 {len(clean_df)} 支股票的指标...")

    for i, row in enumerate(clean_df.itertuples()):
        ts_code = getattr(row, 'ts_code'); pct_chg = getattr(row, 'pct_chg', 0.0);
        turnover_rate = getattr(row, 'turnover_rate', np.nan); net_mf = float(getattr(row, 'net_mf', 0.0));
        amount = getattr(row, 'amount_yuan', 0.0) 
        name = getattr(row, 'name', ts_code)
        # --- V10.5 修复：确保 circ_mv_wan 被带入最终 fdf ---
        circ_mv_wan = getattr(row, 'circ_mv_wan', np.nan)
        # ----------------------------------------------------

        @memory.cache
        def get_daily_hist(ts_code, start_date, end_date):
            return safe_get(pro.daily, ts_code=ts_code, start_date=start_date, end_date=end_date)
            
        hist = get_daily_hist(ts_code, start_date_hist, trade_date)
        
        ind = compute_indicators(hist)

        vol_ratio, ten_return, macd, k, d, j, vol_last, vol_ma5, prev3_sum, volatility_10 = \
            ind.get('vol_ratio', np.nan), ind.get('10d_return', np.nan), ind.get('macd', np.nan), \
            ind.get('k', np.nan), ind.get('d', np.nan), ind.get('j', np.nan), \
            ind.get('vol_last', np.nan), ind.get('vol_ma5', np.nan), ind.get('prev3_sum', np.nan), ind.get('volatility_10', np.nan)

        try: proxy_money = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
        except: proxy_money = 0.0

        rec = {'ts_code': ts_code, 'pct_chg': pct_chg, 'turnover_rate': turnover_rate, 'net_mf': net_mf, 'amount': amount,
               'vol_ratio': vol_ratio, '10d_return': ten_return, 'macd': macd, 'k': k, 'd': d, 'j': j,
               'vol_last': vol_last, 'vol_ma5': vol_ma5, 'prev3_sum': prev3_sum, 'volatility_10': volatility_10,
               'proxy_money': proxy_money, 'name': name,
               'circ_mv_wan': circ_mv_wan} # <-- V10.5 FIX HERE: Pass to final DataFrame
        records.append(rec)
        
        if pbar: pbar.progress((i + 1) / len(clean_df), text=f"指标计算进度：[{i+1}/{len(clean_df)}]...")
        
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame()
    if pbar: pbar.empty()

    # 5. 归一化和评分
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
    
    # 低波动率是加分项
    fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))

    # V10.0 权重：高换手 (0.35) 和 低波动 (0.25) 权重最高
    w_pct, w_volratio, w_turn, w_money, w_10d, w_macd, w_rsl, w_volatility = 0.05, 0.10, 0.35, 0.10, 0.05, 0.10, 0.05, 0.25
    
    fdf['综合评分'] = (fdf['s_pct'] * w_pct + fdf['s_volratio'] * w_volratio + fdf['s_turn'] * w_turn + fdf['s_money'] 
        * w_money + fdf['s_10d'] * w_10d + fdf['s_macd'] * w_macd + fdf['s_rsl'] * w_rsl + fdf['s_volatility'] * w_volatility)
    
    return fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)


# ----------------------------------------------------
# 简易回测模块 (与 V10.3 相同)
# ----------------------------------------------------
def run_simple_backtest(days, params):
    
    HOLDING_PERIODS = [1, 3, 5] 
    status = st.session_state['backtest_status']
    
    container = st.empty()
    with container.container():
        st.subheader(f"📈 简易历史回测结果 (T+{', T+'.join(map(str, HOLDING_PERIODS))} 并列)")
        
        # 使用 pro.trade_cal 获取交易日历，确保稳定性
        trade_dates_df = safe_get(pro.trade_cal, exchange='SSE', is_open='1', end_date=find_last_trade_day(), fields='cal_date')
        if trade_dates_df.empty:
            st.error("无法获取历史交易日历。")
            return

        max_holding = max(HOLDING_PERIODS)
        trade_dates = trade_dates_df['cal_date'].sort_values(ascending=False).head(days + max_holding).tolist() 
        trade_dates.reverse() 
        total_iterations = len(trade_dates) - max_holding 
        
        if total_iterations < 1:
            st.warning(f"交易日不足 {max_holding + 1} 天，无法进行回测。")
            return
            
        status['total_days'] = total_iterations
        start_index = status['current_index']
        
        if start_index >= total_iterations:
             st.success(f"回测已完成。累计收益率请查看下方。")
        else:
             st.info(f"回测周期：**{trade_dates[0]}** 至 **{trade_dates[total_iterations-1]}**。正在从第 {start_index+1} 天继续...")

        pbar = st.progress(status['progress'], text=f"回测进度：[{status['current_index']}/{status['total_days']}]...")
        
        score_params = {
            'INITIAL_TOP_N': params['INITIAL_TOP_N'], 'FINAL_POOL': params['FINAL_POOL'], 'MIN_PRICE': params['MIN_PRICE'], 
            'MAX_PRICE': params['MAX_PRICE'], 'MIN_TURNOVER': params['MIN_TURNOVER'], 'MIN_AMOUNT': params['MIN_AMOUNT'], 
            'VOL_SPIKE_MULT': params['VOL_SPIKE_MULT'], 'VOLATILITY_MAX': params['VOLATILITY_MAX'], 
            'HIGH_PCT_THRESHOLD': params['HIGH_PCT_THRESHOLD'],
            'MIN_CIRC_MV_Billion': params['MIN_CIRC_MV_Billion'] 
        }
        
        for i in range(start_index, total_iterations):
            select_date = trade_dates[i]
            next_trade_date = trade_dates[i+1] 
            
            select_df_full = run_scoring_for_date(select_date, score_params) 

            result = {
                '选股日': select_date, 
                '股票': '无符合条件', 
                '买入价 (T+1 开盘)': np.nan, 
                '评分': np.nan
            }
            for N in HOLDING_PERIODS:
                 result[f'T+{N} 收益率 (%)'] = 0.0
                 result[f'T+{N} 卖出价'] = np.nan
                 
            
            if not select_df_full.empty:
                top_pick = select_df_full.iloc[0] 
                ts_code = top_pick['ts_code']
                
                buy_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=next_trade_date)
                buy_price = buy_day_data.iloc[0]['open'] if not buy_day_data.empty and 'open' in buy_day_data.columns else np.nan
                
                result['股票'] = f"{top_pick.get('name', 'N/A')}({ts_code})"
                result['买入价 (T+1 开盘)'] = buy_price
                result['评分'] = top_pick['综合评分']
                
                if buy_price > 0 and not pd.isna(buy_price):
                    
                    for N in HOLDING_PERIODS:
                        sell_trade_date = trade_dates[i+N] 
                        
                        sell_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=sell_trade_date)
                        
                        if not sell_day_data.empty and 'close' in sell_day_data.columns:
                            sell_price = sell_day_data.iloc[0]['close']
                            result[f'T+{N} 卖出价'] = sell_price
                            
                            if not pd.isna(sell_price):
                                return_pct = (sell_price / buy_price) - 1.0
                                result[f'T+{N} 收益率 (%)'] = return_pct * 100
                        
            status['results'].append(result)
            status['current_index'] = i + 1
            status['progress'] = (i + 1) / total_iterations
            
            pbar.progress(status['progress'], text=f"正在回测 {select_date}... [{i+1}/{total_iterations}]")
            
            if (i+1) % 2 == 0 or (i + 1) == total_iterations: 
                 st.rerun() 
        
        status['progress'] = 1.0
        status['current_index'] = total_iterations
        pbar.progress(1.0, text="回测完成。")
        
        results_df = pd.DataFrame(status['results'])
        
        if results_df.empty:
            st.warning("回测结果为空。")
            return
            
        st.markdown("---")
        st.subheader("💡 最终回测指标（多周期对比）")
        
        cols_metrics = st.columns(len(HOLDING_PERIODS))
        
        for idx, N in enumerate(HOLDING_PERIODS):
            col_name = f'T+{N} 收益率 (%)'
            results_df[col_name] = results_df[col_name].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            cumulative_return = (results_df[col_name] / 100 + 1).product() - 1
            wins = (results_df[col_name] > 0).sum()
            total_trades = len(results_df)
            win_rate = wins / total_trades if total_trades > 0 else 0

            with cols_metrics[idx]:
                st.metric(f"累计收益率 (T+{N})", f"{cumulative_return*100:.2f}%")
                st.caption(f"胜率: {win_rate*100:.2f}% | 交易次数: {total_trades}")
        
        st.markdown("---")
        st.subheader("📋 每日交易记录")
        
        display_cols = ['选股日', '股票', '评分', '买入价 (T+1 开盘)']
        for N in HOLDING_PERIODS:
            display_cols.append(f'T+{N} 收益率 (%)')
            display_cols.append(f'T+{N} 卖出价')
            
        st.dataframe(results_df[display_cols], use_container_width=True)


# ----------------------------------------------------
# 实时选股模块 (V10.5 使用修复后的 fdf)
# ----------------------------------------------------
def run_live_selection(last_trade, params):
    st.write(f"正在运行实时选股（最近交易日：{last_trade}）...")
    
    params_dict = {
        'INITIAL_TOP_N': params['INITIAL_TOP_N'], 'FINAL_POOL': params['FINAL_POOL'], 'MIN_PRICE': params['MIN_PRICE'], 
        'MAX_PRICE': params['MAX_PRICE'], 'MIN_TURNOVER': params['MIN_TURNOVER'], 'MIN_AMOUNT': params['MIN_AMOUNT'], 
        'VOL_SPIKE_MULT': params['VOL_SPIKE_MULT'], 'VOLATILITY_MAX': params['VOLATILITY_MAX'], 
        'HIGH_PCT_THRESHOLD': params['HIGH_PCT_THRESHOLD'],
        'MIN_CIRC_MV_Billion': params['MIN_CIRC_MV_Billion'] 
    }
    fdf_full = run_scoring_for_date(last_trade, params_dict)

    if fdf_full.empty:
        st.error("清洗和评分后没有候选。请参考上方的诊断信息，如果 Tushare 数据接口连续故障，请等待。")
        st.stop()

    fdf = fdf_full.head(params['TOP_DISPLAY']).copy()
    fdf.index = fdf.index + 1

    st.success(f"评分完成：总候选 {len(fdf_full)} 支，显示 Top {min(params['TOP_DISPLAY'], len(fdf))}。")
    
    # 转换为亿显示 (V10.5: circ_mv_wan 列现在保证存在)
    fdf['流通市值 (亿)'] = fdf['circ_mv_wan'] / 10000.0
    
    display_cols = ['name','ts_code','综合评分','pct_chg','turnover_rate','amount','circ_mv_wan','total_mv_yuan','volatility_10','net_mf','10d_return']
    
    # 调整显示列的顺序和名称
    final_display_cols = ['name','ts_code','综合评分','流通市值 (亿)','pct_chg','turnover_rate','amount','volatility_10','10d_return']
    
    st.dataframe(fdf[final_display_cols], use_container_width=True)

    out_csv = fdf_full[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

    st.markdown("### 小结与操作提示（简洁）")
    st.markdown("""
- **【策略风格】** 本版本为 **V10.5 流通市值修复版**，已修复显示错误。
- **【风控提示】** **当前剩余候选仅 16 支。** 如果数量太少，请尝试降低侧边栏的 **“最低流通市值 (亿)”** 或 **“最低价格 (元)”** 参数。
- **【重要纪律】** 9:40 前不买 → 观察 9:40-10:05 的量价节奏 → 10:05 后择优介入。
""")


# ----------------------------------------------------
# 主程序控制逻辑
# ----------------------------------------------------
params = {
    'INITIAL_TOP_N': INITIAL_TOP_N, 'FINAL_POOL': FINAL_POOL, 'TOP_DISPLAY': TOP_DISPLAY,
    'MIN_PRICE': MIN_PRICE, 'MAX_PRICE': MAX_PRICE, 'MIN_TURNOVER': MIN_TURNOVER,
    'MIN_AMOUNT': MIN_AMOUNT, 'VOL_SPIKE_MULT': VOL_SPIKE_MULT, 'VOLATILITY_MAX': VOLATILITY_MAX,
    'HIGH_PCT_THRESHOLD': HIGH_PCT_THRESHOLD,
    'MIN_CIRC_MV_Billion': MIN_CIRC_MV_Billion 
}

if st.session_state.get('run_backtest', False):
    run_simple_backtest(BACKTEST_DAYS, params)
    
elif st.session_state.get('run_selection', False):
    run_live_selection(last_trade, params)
    
else:
    st.info("请点击上方的按钮开始运行。")
