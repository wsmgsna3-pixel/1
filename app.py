# -*- coding: utf-8 -*-
"""
选股王 · V13.8（数据校验稳定版）
核心：修复日期获取逻辑。采用 '数据校验回溯法'，确保选股日是 Tushare 接口实际有数据的最近日期。
"""
import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import joblib 
import os
import math
import time 

warnings.filterwarnings("ignore")

# ---------------------------
# 外部缓存配置 (joblib 仅用于历史数据)
# ---------------------------
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0) 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王（V13.8 数据校验稳定版）", layout="wide")
st.markdown("### 选股王（V13.8 数据校验稳定版）- 修复未来日期问题") 

# ---------------------------
# 默认参数定义 (保持 V13.6/V13.7)
# ---------------------------
DEFAULT_FINAL_POOL = 500
DEFAULT_TOP_DISPLAY = 30
DEFAULT_MIN_PRICE = 10.0
DEFAULT_MAX_PRICE = 200.0
DEFAULT_MIN_CIRC_MV_B = 40.0 
DEFAULT_MAX_CIRC_MV_B = 500.0 
DEFAULT_MIN_TURNOVER = 3.0 
DEFAULT_MIN_AMOUNT = 200_000_000.0 
DEFAULT_MA_PERIOD = 20
DEFAULT_MIN_LIST_DAYS = 180
DEFAULT_BACKTEST_DAYS = 10

DEFAULT_VOL_SPIKE_MULT = 1.7
DEFAULT_HIGH_PCT_THRESHOLD = 6.0
DEFAULT_MAX_VOLATILITY_10D = 8.0 

# ---------------------------
# 侧边栏参数 
# ---------------------------
with st.sidebar:
    st.header("可调参数（V13.8 默认值）")
    INITIAL_TOP_N = 99999 
    
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=DEFAULT_FINAL_POOL, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=DEFAULT_TOP_DISPLAY, step=5))
    
    st.markdown("---")
    st.subheader("基础过滤 (硬性要求)")
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=DEFAULT_MIN_PRICE, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=DEFAULT_MAX_PRICE, step=10.0))
    
    # 请根据需要调整流通市值范围
    MIN_CIRC_MV_Billion = float(st.number_input("最低流通市值 (亿)", value=DEFAULT_MIN_CIRC_MV_B, step=5.0)) 
    MAX_CIRC_MV_Billion = float(st.number_input("最高流通市值 (亿)", value=DEFAULT_MAX_CIRC_MV_B, step=50.0)) 
    
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=DEFAULT_MIN_TURNOVER, step=0.1)) 
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=DEFAULT_MIN_AMOUNT, step=50_000_000.0))
    
    MA_TREND_PERIOD = int(st.number_input("趋势过滤：MA 周期", value=DEFAULT_MA_PERIOD, step=5))
    MIN_LIST_DAYS = int(st.number_input("次新股排除：最低上市天数 (天)", value=DEFAULT_MIN_LIST_DAYS, step=30))
    
    st.markdown("---")
    st.subheader("短线风控参数 (BC 增强)")
    
    VOL_SPIKE_MULT = float(st.number_input("巨量冲高：放量倍数阈值", value=DEFAULT_VOL_SPIKE_MULT, step=0.1))
    HIGH_PCT_THRESHOLD = float(st.number_input("大阳线/反弹定义 (%变化)", value=DEFAULT_HIGH_PCT_THRESHOLD, step=0.5))
    MAX_VOLATILITY_10D = float(st.number_input("极端波动：10日波动 std 阈值 (%)", value=DEFAULT_MAX_VOLATILITY_10D, step=0.5))
    
    st.markdown("---")
    
    BACKTEST_DAYS = int(st.number_input("回测：最近 N 个交易日", value=DEFAULT_BACKTEST_DAYS, step=1))
    
    st.markdown("---")
    st.caption("提示：策略已升级至 'V13.8 数据校验稳定版'。")


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
    """尝试获取数据，失败则返回空 DataFrame"""
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

# ----------------------------------------------------
# 交易日历获取 (V13.8: 获取日历的逻辑不变，但用于稳定校验)
# ----------------------------------------------------
@st.cache_data(ttl=600)
def get_trade_cal_dates():
    """安全地从 Tushare 获取所有开放交易日，并按降序排列。"""
    end_date = datetime.now().strftime("%Y%m%d")
    cal_df = safe_get(
        pro.trade_cal, 
        exchange='SSE', 
        is_open='1', 
        end_date=end_date, 
        fields='cal_date'
    )
    if cal_df.empty: return []
    return cal_df['cal_date'].sort_values(ascending=False).tolist()

# ----------------------------------------------------
# 核心修正：数据校验回溯函数 (V13.8 新增)
# ----------------------------------------------------
@st.cache_data(ttl=3600)
def find_last_trade_day_robust(pro_api):
    """
    V13.8 核心修正：迭代最近交易日，直到找到 Tushare 接口实际能提供数据的日期。
    """
    trade_dates = get_trade_cal_dates()
    
    if not trade_dates: return None
    
    # 尝试最近的 5 个交易日
    for date_str in trade_dates[:5]: 
        
        # 尝试拉取全市场日线数据
        daily_all = safe_get(pro_api.daily, trade_date=date_str)
        
        if not daily_all.empty:
            # 找到有数据的日期，返回
            return date_str
        
        # 否则，继续回溯到前一个日期
            
    return None # 5 天内都没有数据，返回 None

# V13.8 运行稳定的日期函数
last_trade = find_last_trade_day_robust(pro)

if not last_trade:
    st.error("无法获取最近交易日。已尝试回溯最近 5 个交易日，但 Tushare 接口均无数据。请检查 Tushare Token 或等待数据更新。")
    st.stop()
st.info(f"参考最近交易日（经数据校验）：**{last_trade}**")


# ----------------------------------------------------
# 按钮控制模块 
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
# 辅助函数 (保持 V13.7 逻辑)
# ---------------------------
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
    
    try: 
        joined = pool.join(other_df.set_index('ts_code')[cols], how='left')
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


# ---------------------------
# V13.8 增强：指标计算和归一化 (保持 V13.7 逻辑)
# ---------------------------
def compute_indicators(df, ma_period):
    res = {}
    if df.empty or len(df) < 3: return res
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    vols = df['vol'].astype(float).tolist()

    res['last_close'] = close.iloc[-1]
    
    if len(close) >= ma_period:
        res[f'ma{ma_period}'] = close.rolling(window=ma_period).mean().iloc[-1]
    else:
        res[f'ma{ma_period}'] = np.nan
        
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd'] = (diff - dea).iloc[-1] * 2
    else: res['macd'] = np.nan

    n_kdj = 9
    if len(close) >= n_kdj:
        low_n = low.rolling(window=n_kdj).min()
        high_n = high.rolling(window=n_kdj).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
        rsv = rsv.fillna(50)
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3*k - 2*d
        res['k'] = k.iloc[-1]; res['d'] = d.iloc[-1]; res['j'] = j.iloc[-1]
    else:
        res['k'] = res['d'] = res['j'] = np.nan

    if len(vols) >= 6:
        avg_prev5 = np.mean(vols[-6:-1])
        res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9)
        res['vol_last'] = vols[-1]
        res['vol_ma5'] = avg_prev5
    else:
        res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan

    if len(close) >= 10:
        res['10d_return'] = close.iloc[-1] / close.iloc[-10] - 1
    else:
        res['10d_return'] = np.nan
    
    if 'pct_chg' in df.columns and len(df) >= 4:
        try:
            pct = df['pct_chg'].astype(float)
            res['prev3_sum'] = pct.iloc[-4:-1].sum()
        except:
            res['prev3_sum'] = np.nan
    else:
        res['prev3_sum'] = np.nan

    try:
        if 'pct_chg' in df.columns and len(df) >= 10:
            res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()
        else:
            res['volatility_10'] = np.nan
    except: res['volatility_10'] = np.nan
    
    return res

# ----------------------------------------------------
# 核心评分函数 (V13.8: 逻辑保持 V13.7)
# ----------------------------------------------------
@st.cache_data(show_spinner=False, ttl=600)
def run_scoring_for_date(trade_date, params):
    
    # 参数安全解包
    min_price = params.get('MIN_PRICE', DEFAULT_MIN_PRICE)
    max_price = params.get('MAX_PRICE', DEFAULT_MAX_PRICE)
    min_turnover = params.get('MIN_TURNOVER', DEFAULT_MIN_TURNOVER)
    min_amount = params.get('MIN_AMOUNT', DEFAULT_MIN_AMOUNT)
    min_circ_mv_billion = params.get('MIN_CIRC_MV_Billion', DEFAULT_MIN_CIRC_MV_B)
    max_circ_mv_billion = params.get('MAX_CIRC_MV_Billion', DEFAULT_MAX_CIRC_MV_B)
    ma_trend_period = params.get('MA_TREND_PERIOD', DEFAULT_MA_PERIOD)
    min_list_days = params.get('MIN_LIST_DAYS', DEFAULT_MIN_LIST_DAYS)
    final_pool_size = params.get('FINAL_POOL', DEFAULT_FINAL_POOL) 

    vol_spike_mult = params.get('VOL_SPIKE_MULT', DEFAULT_VOL_SPIKE_MULT)
    high_pct_threshold = params.get('HIGH_PCT_THRESHOLD', DEFAULT_HIGH_PCT_THRESHOLD)
    max_volatility_10d = params.get('MAX_VOLATILITY_10D', DEFAULT_MAX_VOLATILITY_10D)
    
    # 1. 拉取数据 (Daily 提供 open/high/low/pre_close)
    daily_all = safe_get(pro.daily, trade_date=trade_date)
    daily_basic = safe_get(pro.daily_basic, trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    moneyflow = safe_get(pro.moneyflow, trade_date=trade_date, fields='ts_code,net_mf_amount') 

    if daily_all.empty: 
        # V13.8: 只有在选股日不等于全局 last_trade (即回测中调用) 才会警告
        if trade_date == last_trade: 
             # 理论上被 find_last_trade_day_robust 过滤了，此处为双重保险
             st.error(f"诊断：Tushare 无法获取 {trade_date} 的日线数据。请检查 Token 权限或等待数据更新。")
        return pd.DataFrame()
    
    pool0 = daily_all.copy().reset_index(drop=True)

    # 2. 合并基本信息 
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')
    
    if not stock_basic.empty:
        keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv','list_date'] if c in stock_basic.columns]
        try: pool0 = pool0.merge(stock_basic[keep], on='ts_code', how='left')
        except Exception: 
            pool0['name'] = pool0['ts_code']; pool0['industry'] = ''; pool0['list_date'] = '20000101'
    else: 
        pool0['name'] = pool0['ts_code']; pool0['industry'] = ''; pool0['list_date'] = '20000101'
    
    pool_merged = safe_merge_pool(pool0, daily_basic.rename(columns={'amount':'amount_db'}), ['turnover_rate','amount_db','total_mv','circ_mv'])

    if not moneyflow.empty:
        pool_merged = safe_merge_pool(pool_merged, moneyflow, ['net_mf_amount'])
    else:
        pool_merged['net_mf_amount'] = 0.0

    
    if 'amount' in pool_merged.columns:
        pool_merged['amount'] = pool_merged['amount'].apply(lambda amt: amt * 10000.0 if not pd.isna(amt) and amt > 0 and amt < 1e5 else amt)
    else:
        pool_merged['amount'] = pool_merged['amount_db'].apply(lambda amt: amt * 10000.0 if not pd.isna(amt) and amt > 0 and amt < 1e5 else amt)
    
    pool_merged['amount_yuan'] = pool_merged['amount']
    pool_merged['circ_mv_wan'] = pool_merged['circ_mv'].fillna(0)


    # 3. V13.8 硬性过滤
    clean_df = pool_merged.copy()
    
    # 基础风险过滤 (ST, 价格, 北交所)
    clean_df = clean_df[~(
        (clean_df['close'].isna()) | 
        (clean_df['close'] < min_price) | 
        (clean_df['close'] > max_price) | 
        (clean_df['name'].str.contains('ST|退', case=False, na=False)) |
        (clean_df['ts_code'].str.endswith('.BJ', na=False)) 
    )]
    
    # 今日必须上涨（pct_chg > 0）
    clean_df = clean_df[~((clean_df['pct_chg'].isna()) | (clean_df['pct_chg'] < 0))]
    
    # 排除一字板 (open == high == low == pre_close)
    mask_yiziban = (clean_df['open'] == clean_df['high']) & \
                   (clean_df['high'] == clean_df['low']) & \
                   (clean_df['low'] == clean_df['pre_close']) & \
                   (clean_df['high'] > clean_df['pre_close']) 
    clean_df = clean_df[~mask_yiziban.fillna(False)]
    
    # 次新股过滤 
    current_date = datetime.strptime(trade_date, "%Y%m%d")
    clean_df['list_date'] = pd.to_datetime(clean_df['list_date'], format='%Y%m%d', errors='coerce')
    clean_df['days_since_list'] = (current_date - clean_df['list_date']).dt.days
    clean_df = clean_df[clean_df['days_since_list'].notna() & (clean_df['days_since_list'] >= min_list_days)]
    
    # 流通市值上下限过滤 (这是您要关注的核心范围)
    min_circ_mv_wan = min_circ_mv_billion * 10000.0 
    max_circ_mv_wan = max_circ_mv_billion * 10000.0 
    clean_df = clean_df[clean_df['circ_mv_wan'].notna() & 
                        (clean_df['circ_mv_wan'] >= min_circ_mv_wan) &
                        (clean_df['circ_mv_wan'] <= max_circ_mv_wan)]

    # 流动性过滤
    clean_df = clean_df[clean_df['amount_yuan'].notna() & (clean_df['amount_yuan'] >= min_amount)]
    clean_df = clean_df[clean_df['turnover_rate'].notna() & (clean_df['turnover_rate'] >= min_turnover)]
    
    
    if clean_df.empty: 
        if trade_date == last_trade: st.error(f"诊断：所有硬性过滤后，剩余股票数量为 **0** 支。请检查侧边栏参数。")
        return pd.DataFrame()

    if trade_date == last_trade:
        st.info(f"诊断：硬性过滤 (已包含次新股、市值收紧) 后，剩余股票数量: **{len(clean_df)}** 支，开始计算指标...")
        
    # 4. 指标计算与 MA 趋势硬性过滤 
    score_pool = clean_df.sort_values('pct_chg', ascending=False).head(min(len(clean_df), 300)).copy().reset_index(drop=True)

    records = []
    start_dt = datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=60 * 1.5) 
    start_date_hist = start_dt.strftime("%Y%m%d")
    
    pbar = None
    if trade_date == last_trade:
        pbar = st.progress(0.0, text=f"正在计算 {len(score_pool)} 支股票的指标...")

    for i, row in enumerate(score_pool.itertuples()):
        ts_code = getattr(row, 'ts_code');
        close_price = getattr(row, 'close', np.nan)
        
        @memory.cache 
        def get_daily_hist(ts_code, start_date, end_date):
            return safe_get(pro.daily, ts_code=ts_code, start_date=start_date, end_date=end_date)
            
        hist = get_daily_hist(ts_code, start_date_hist, trade_date)
        
        ind = compute_indicators(hist, ma_trend_period)
        ma_trend_val = ind.get(f'ma{ma_trend_period}', np.nan)
        
        # --- MA 趋势硬性过滤 --- 
        if not pd.isna(close_price) and not pd.isna(ma_trend_val) and (close_price < ma_trend_val):
             if pbar: pbar.progress((i + 1) / len(score_pool), text=f"指标计算进度：[{i+1}/{len(score_pool)}]... (已排除趋势向下股)")
             continue 

        rec = {
            'ts_code': ts_code, 
            'pct_chg': getattr(row, 'pct_chg', np.nan),
            'turnover_rate': getattr(row, 'turnover_rate', np.nan),
            'circ_mv_wan': getattr(row, 'circ_mv_wan', np.nan),
            'amount_yuan': getattr(row, 'amount_yuan', np.nan),
            'net_mf_amount': getattr(row, 'net_mf_amount', np.nan),
            'name': getattr(row, 'name', ts_code),
            f'ma{ma_trend_period}': ma_trend_val,
            
            'last_close': ind.get('last_close', np.nan),
            'macd': ind.get('macd', np.nan), 
            'k': ind.get('k', np.nan), 'd': ind.get('d', np.nan), 'j': ind.get('j', np.nan),
            'vol_ratio': ind.get('vol_ratio', np.nan),
            'vol_last': ind.get('vol_last', np.nan),
            'vol_ma5': ind.get('vol_ma5', np.nan),
            '10d_return': ind.get('10d_return', np.nan),
            'prev3_sum': ind.get('prev3_sum', np.nan),
            'volatility_10': ind.get('volatility_10', np.nan),
        }
        records.append(rec)
        
        if pbar: pbar.progress((i + 1) / len(score_pool), text=f"指标计算进度：[{i+1}/{len(score_pool)}]... (已排除趋势向下股)")
        
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame()
    if pbar: pbar.empty()

    if trade_date == last_trade:
        st.info(f"诊断：通过 {ma_trend_period} 日均线趋势过滤后，剩余股票数量: **{len(fdf)}** 支，开始高级风险过滤...")
        
    
    # 5. V13.8 高级风险过滤
    try:
        before_cnt = len(fdf)
        
        if all(c in fdf.columns for c in [f'ma{ma_trend_period}','last_close','pct_chg']):
            mask_high_big = (fdf['last_close'] > fdf[f'ma{ma_trend_period}'] * 1.10) & (fdf['pct_chg'] > high_pct_threshold)
            fdf = fdf[~mask_high_big.fillna(False)]

        if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
            mask_down_rebound = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > high_pct_threshold)
            fdf = fdf[~mask_down_rebound.fillna(False)]

        if 'vol_ratio' in fdf.columns:
            mask_vol_spike = (fdf['vol_ratio'] > vol_spike_mult)
            fdf = fdf[~mask_vol_spike.fillna(False)]

        if 'volatility_10' in fdf.columns:
            mask_volatility = fdf['volatility_10'] > max_volatility_10d
            fdf = fdf[~mask_volatility.fillna(False)]

        after_cnt = len(fdf)
        if trade_date == last_trade:
            st.info(f"诊断：高级风险过滤后，剩余股票数量: **{after_cnt}** 支，开始评分...")
    except Exception as e:
        if trade_date == last_trade: st.warning(f"高级风险过滤模块异常，跳过过滤。错误：{e}")
    
    if fdf.empty: return pd.DataFrame()


    # 6. RSL（相对强弱）计算
    if '10d_return' in fdf.columns:
        try:
            fdf['proxy_money'] = (abs(fdf['pct_chg']) + 1e-9) * fdf['vol_ratio'].fillna(0) * fdf['turnover_rate'].fillna(0)
            
            market_mean_10d = fdf['10d_return'].replace([np.inf,-np.inf], np.nan).dropna().mean()
            if np.isnan(market_mean_10d) or abs(market_mean_10d) < 1e-9:
                market_mean_10d = 1e-9
            fdf['rsl'] = fdf['10d_return'] / market_mean_10d
        except:
            fdf['rsl'] = 1.0
            fdf['proxy_money'] = 0.0
    else:
        fdf['rsl'] = 1.0
        fdf['proxy_money'] = 0.0


    # 7. 归一化
    fdf['s_pct'] = norm_col(fdf.get('pct_chg', pd.Series([0]*len(fdf))))
    fdf['s_volratio'] = norm_col(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))
    fdf['s_turn'] = norm_col(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))
    
    if 'net_mf_amount' in fdf.columns and fdf['net_mf_amount'].abs().sum() > 0:
        fdf['s_money'] = norm_col(fdf.get('net_mf_amount', pd.Series([0]*len(fdf))))
    else:
        fdf['s_money'] = norm_col(fdf.get('proxy_money', pd.Series([0]*len(fdf))))

    fdf['s_amount'] = norm_col(fdf.get('amount_yuan', pd.Series([0]*len(fdf))))
    fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))
    fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
    fdf['s_rsl'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))
    fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf)))) 
    
    
    # 8. 综合评分
    w_pct = 0.18        
    w_volratio = 0.18   
    w_turn = 0.12       
    w_money = 0.14      
    w_10d = 0.12        
    w_macd = 0.06       
    w_rsl = 0.12        
    w_volatility = 0.08 

    fdf['综合评分'] = (
        fdf['s_pct'] * w_pct +
        fdf['s_volratio'] * w_volratio +
        fdf['s_turn'] * w_turn +
        fdf['s_money'] * w_money +
        fdf['s_10d'] * w_10d +
        fdf['s_macd'] * w_macd +
        fdf['s_rsl'] * w_rsl +
        fdf['s_volatility'] * w_volatility
    )
    
    return fdf.sort_values('综合评分', ascending=False).head(final_pool_size).reset_index(drop=True)


# ----------------------------------------------------
# 简易回测模块 (保持 V13.7 逻辑)
# ----------------------------------------------------
def run_simple_backtest(days, params):
    
    HOLDING_PERIODS = [1, 3, 5]
    status = st.session_state['backtest_status']
    
    container = st.empty()
    with container.container():
        st.subheader(f"📈 简易历史回测结果 (V13.8 数据校验稳定版)")
        
        trade_dates_all = get_trade_cal_dates()
        
        if not trade_dates_all:
             st.error("无法获取历史交易日历。")
             return

        try:
            current_trade_idx = trade_dates_all.index(last_trade)
            # 确保 trade_dates_all 的第一个日期就是 last_trade，只保留有数据的日期
            trade_dates_all = trade_dates_all[current_trade_idx:]
        except ValueError:
            st.error(f"内部错误：无法定位最近有效交易日 {last_trade}。")
            return


        max_holding = max(HOLDING_PERIODS)
        trade_dates = trade_dates_all[:days + max_holding]
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
            'MIN_PRICE': params.get('MIN_PRICE', DEFAULT_MIN_PRICE), 
            'MAX_PRICE': params.get('MAX_PRICE', DEFAULT_MAX_PRICE), 
            'MIN_TURNOVER': params.get('MIN_TURNOVER', DEFAULT_MIN_TURNOVER), 
            'MIN_AMOUNT': params.get('MIN_AMOUNT', DEFAULT_MIN_AMOUNT), 
            'MIN_CIRC_MV_Billion': params.get('MIN_CIRC_MV_Billion', DEFAULT_MIN_CIRC_MV_B),
            'MAX_CIRC_MV_Billion': params.get('MAX_CIRC_MV_Billion', DEFAULT_MAX_CIRC_MV_B),
            'MA_TREND_PERIOD': params.get('MA_TREND_PERIOD', DEFAULT_MA_PERIOD),
            'MIN_LIST_DAYS': params.get('MIN_LIST_DAYS', DEFAULT_MIN_LIST_DAYS),
            'FINAL_POOL': params.get('FINAL_POOL', DEFAULT_FINAL_POOL),
            'VOL_SPIKE_MULT': params.get('VOL_SPIKE_MULT', DEFAULT_VOL_SPIKE_MULT),
            'HIGH_PCT_THRESHOLD': params.get('HIGH_PCT_THRESHOLD', DEFAULT_HIGH_PCT_THRESHOLD),
            'MAX_VOLATILITY_10D': params.get('MAX_VOLATILITY_10D', DEFAULT_MAX_VOLATILITY_10D)
        }
        
        for i in range(start_index, total_iterations):
            select_date = trade_dates[i]
            next_trade_date = trade_dates[i+1] 
            
            select_df_full = run_scoring_for_date(select_date, score_params) 

            result = {
                '选股日': select_date, 
                '股票': '无符合条件', 
                '买入价 (T+1 开盘)': np.nan, 
                '评分': np.nan,
                '市值 (亿)': np.nan
            }
            for N in HOLDING_PERIODS:
                 result[f'T+{N} 收益率 (%)'] = 0.0
                 result[f'T+{N} 卖出价'] = np.nan
                 
            
            if not select_df_full.empty:
                top_pick = select_df_full.iloc[0] 
                ts_code = top_pick['ts_code']
                
                max_retries = 3 
                buy_day_data = pd.DataFrame()
                for attempt in range(max_retries):
                    buy_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=next_trade_date)
                    if not buy_day_data.empty: break
                    time.sleep(1) 
                    
                buy_price = buy_day_data.iloc[0]['open'] if not buy_day_data.empty and 'open' in buy_day_data.columns else np.nan
                
                result['股票'] = f"{top_pick.get('name', 'N/A')}({ts_code})"
                result['买入价 (T+1 开盘)'] = buy_price
                result['评分'] = top_pick['综合评分']
                result['市值 (亿)'] = top_pick['circ_mv_wan'] / 10000.0 if not pd.isna(top_pick['circ_mv_wan']) else np.nan
                
                if buy_price > 0 and not pd.isna(buy_price):
                    
                    for N in HOLDING_PERIODS:
                        sell_trade_date = trade_dates[i+N] 
                        
                        sell_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=sell_trade_date)
                        
                        if not sell_day_data.empty and 'close' in sell_day_data.columns:
                            sell_price = sell_day_data.iloc[0]['close']
                            result[f'T+{N} 卖出价'] = sell_price
                            
                            if not pd.isna(sell_price):
                                return_pct = (sell_price / buy_price) - 1.0
                                return_pct = max(-0.10, return_pct) 
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
        
        display_cols = ['选股日', '股票', '市值 (亿)', '评分', '买入价 (T+1 开盘)']
        for N in HOLDING_PERIODS:
            display_cols.append(f'T+{N} 收益率 (%)')
            
        st.dataframe(results_df[display_cols], use_container_width=True)


# ----------------------------------------------------
# 实时选股模块 (V13.8)
# ----------------------------------------------------
def run_live_selection(last_trade, params):
    st.write(f"正在运行实时选股（最近有效交易日：{last_trade}）...")
    
    params_dict = {
        'MIN_PRICE': params.get('MIN_PRICE', DEFAULT_MIN_PRICE), 
        'MAX_PRICE': params.get('MAX_PRICE', DEFAULT_MAX_PRICE), 
        'MIN_TURNOVER': params.get('MIN_TURNOVER', DEFAULT_MIN_TURNOVER), 
        'MIN_AMOUNT': params.get('MIN_AMOUNT', DEFAULT_MIN_AMOUNT), 
        'MIN_CIRC_MV_Billion': params.get('MIN_CIRC_MV_Billion', DEFAULT_MIN_CIRC_MV_B),
        'MAX_CIRC_MV_Billion': params.get('MAX_CIRC_MV_Billion', DEFAULT_MAX_CIRC_MV_B),
        'MA_TREND_PERIOD': params.get('MA_TREND_PERIOD', DEFAULT_MA_PERIOD),
        'MIN_LIST_DAYS': params.get('MIN_LIST_DAYS', DEFAULT_MIN_LIST_DAYS),
        'FINAL_POOL': params.get('FINAL_POOL', DEFAULT_FINAL_POOL),
        'VOL_SPIKE_MULT': params.get('VOL_SPIKE_MULT', DEFAULT_VOL_SPIKE_MULT),
        'HIGH_PCT_THRESHOLD': params.get('HIGH_PCT_THRESHOLD', DEFAULT_HIGH_PCT_THRESHOLD),
        'MAX_VOLATILITY_10D': params.get('MAX_VOLATILITY_10D', DEFAULT_MAX_VOLATILITY_10D)
    }
    
    fdf_full = run_scoring_for_date(last_trade, params_dict)

    if fdf_full.empty:
        st.error(f"清洗和评分后没有候选。请检查硬性过滤参数是否过于严格。")
        st.stop()

    fdf = fdf_full.head(params.get('TOP_DISPLAY', DEFAULT_TOP_DISPLAY)).copy()
    fdf.index = fdf.index + 1

    st.success(f"评分完成：总候选 {len(fdf_full)} 支，显示 Top {min(params.get('TOP_DISPLAY', DEFAULT_TOP_DISPLAY), len(fdf))}。")
    
    fdf['流通市值 (亿)'] = fdf['circ_mv_wan'] / 10000.0
    
    final_display_cols = [
        'name','ts_code','综合评分','pct_chg','流通市值 (亿)','turnover_rate',
        'vol_ratio','10d_return','net_mf_amount','macd','volatility_10'
    ]
    
    for c in final_display_cols:
        if c not in fdf.columns: fdf[c] = np.nan
    
    if 'net_mf_amount' in fdf.columns:
        fdf['净流入 (亿)'] = fdf['net_mf_amount'] / 1e8
        final_display_cols[final_display_cols.index('net_mf_amount')] = '净流入 (亿)'
        
    
    st.dataframe(fdf[final_display_cols], use_container_width=True, height=500)

    download_cols = [c for c in fdf_full.columns if c not in ['list_date', 'days_since_list', 'circ_mv_wan']]
    out_csv = fdf_full[download_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}_V13_8.csv", mime="text/csv")

    st.markdown("### 小结与操作提示（V13.8 数据校验稳定版）")
    st.markdown(f"""
- **【核心修正】** 已修复“未来日期”问题，现在选股日 **{last_trade}** 是 Tushare 接口实际有数据可用的日期。
- **【市值范围】** 当前流通市值范围：**{params_dict['MIN_CIRC_MV_Billion']} 亿 到 {params_dict['MAX_CIRC_MV_Billion']} 亿**。
- **【操作建议】** **如果您仍未看到 100 亿 - 200 亿的股票**，请在左侧边栏将参数调整为：
    - 最低流通市值 (亿) = **100.0**
    - 最高流通市值 (亿) = **200.0**
    - 然后重新运行。
""")


# ----------------------------------------------------
# 主程序控制逻辑
# ----------------------------------------------------
params = {
    'FINAL_POOL': FINAL_POOL, 'TOP_DISPLAY': TOP_DISPLAY,
    'MIN_PRICE': MIN_PRICE, 'MAX_PRICE': MAX_PRICE, 'MIN_TURNOVER': MIN_TURNOVER,
    'MIN_AMOUNT': MIN_AMOUNT, 
    'MIN_CIRC_MV_Billion': MIN_CIRC_MV_Billion,
    'MAX_CIRC_MV_Billion': MAX_CIRC_MV_Billion,
    'MA_TREND_PERIOD': MA_TREND_PERIOD,
    'MIN_LIST_DAYS': MIN_LIST_DAYS,
    'VOL_SPIKE_MULT': VOL_SPIKE_MULT,
    'HIGH_PCT_THRESHOLD': HIGH_PCT_THRESHOLD,
    'MAX_VOLATILITY_10D': MAX_VOLATILITY_10D
}

if st.session_state.get('run_backtest', False):
    run_simple_backtest(BACKTEST_DAYS, params)
    
elif st.session_state.get('run_selection', False):
    run_live_selection(last_trade, params)
    
else:
    st.info("请点击上方的按钮开始运行。")
