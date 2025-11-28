# -*- coding: utf-8 -*-
"""
选股王 · V13.11（平衡过滤 & 回测指标修正版）
核心：
1. 稳定 V13.10 的硬性过滤。
2. 略微放松默认的“巨量冲高”和“极端波动”过滤，以避免过滤掉真正的强势股。
3. 优化回测模块，确保 T+N 收益率计算的准确性。
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
st.set_page_config(page_title="选股王（V13.11 平衡过滤版）", layout="wide")
st.markdown("### 选股王（V13.11 平衡过滤版）- 优化风险过滤与回测") 

# ---------------------------
# 默认参数定义 (V13.11: 微调风险过滤默认值)
# ---------------------------
DEFAULT_FINAL_POOL = 500
DEFAULT_TOP_DISPLAY = 30
DEFAULT_MIN_PRICE = 10.0
DEFAULT_MAX_PRICE = 200.0
DEFAULT_MIN_CIRC_MV_B = 40.0 
DEFAULT_MAX_CIRC_MV_B = 500.0 
DEFAULT_MIN_TURNOVER = 1.0 
DEFAULT_MIN_AMOUNT = 50_000_000.0 
DEFAULT_MA_PERIOD = 20
DEFAULT_MIN_LIST_DAYS = 180
DEFAULT_BACKTEST_DAYS = 10

DEFAULT_VOL_SPIKE_MULT = 1.8 # 略微放宽：允许更大量的成交
DEFAULT_HIGH_PCT_THRESHOLD = 6.0 
DEFAULT_MAX_VOLATILITY_10D = 9.0 # 略微放宽：允许稍大波动

# ---------------------------
# 侧边栏参数 
# ---------------------------
with st.sidebar:
    st.header("可调参数（V13.11 默认值）")
    INITIAL_TOP_N = 99999 
    
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=DEFAULT_FINAL_POOL, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=DEFAULT_TOP_DISPLAY, step=5))
    
    st.markdown("---")
    st.subheader("基础过滤 (硬性要求)")
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=DEFAULT_MIN_PRICE, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=DEFAULT_MAX_PRICE, step=10.0))
    
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
    st.caption("提示：策略已升级至 'V13.11 平衡过滤版'。")


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
pro = ts.pro_api() # 全局 Tushare API 对象

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
# 交易日历获取 
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
# 核心修正：数据校验回溯函数 
# ----------------------------------------------------
@st.cache_data(ttl=3600)
def find_last_trade_day_robust(): 
    """
    V13.9 核心修正：迭代最近交易日，直到找到 Tushare 接口实际能提供数据的日期。
    """
    global pro 

    trade_dates = get_trade_cal_dates()
    
    if not trade_dates: return None
    
    for date_str in trade_dates[:5]: 
        daily_all = safe_get(pro.daily, trade_date=date_str)
        
        if not daily_all.empty:
            return date_str
        
    return None 

# V13.10 运行稳定的日期函数
last_trade = find_last_trade_day_robust() 

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
        # 重置回测状态，除非回测已经完成
        if st.session_state['backtest_status']['progress'] == 1.0 or st.session_state['backtest_status']['total_days'] == 0:
             st.session_state['backtest_status'] = {'progress': 0.0, 'results': [], 'current_index': 0, 'total_days': 0}
        st.rerun()

st.markdown("---")

# ---------------------------
# 辅助函数 
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
# 指标计算和归一化 
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
# 核心评分函数 
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
        if trade_date == last_trade: 
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


    # 3. V13.11 硬性过滤
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
    
    # 流通市值上下限过滤
    min_circ_mv_wan = min_circ_mv_billion * 10000.0 
    max_circ_mv_wan = max_circ_mv_billion * 10000.0 
    clean_df = clean_df[clean_df['circ_mv_wan'].notna() & 
                        (clean_df['circ_mv_wan'] >= min_circ_mv_wan) &
                        (clean_df['circ_mv_wan'] <= max_circ_mv_wan)]

    # 流动性过滤
    clean_df = clean_df[clean_df['amount_yuan'].notna() & (clean_df['amount_yuan'] >= min_amount)]
    clean_df = clean_df[clean_df['turnover_rate'].notna() & (clean_df['turnover_rate'] >= min_turnover)]
    
    
    # V13.11 增强诊断：如果股票数量为 0，打印详细参数
    if clean_df.empty: 
        if trade_date == last_trade:
            st.error(f"诊断：所有硬性过滤后，剩余股票数量为 **0** 支。")
            st.markdown(f"""
            <div style='border: 1px solid #ff4b4b; padding: 10px; border-radius: 5px;'>
            <p style='color: #ff4b4b; font-weight: bold;'>⚠️ **硬性过滤参数总结：**</p>
            <ul>
                <li>价格范围：{min_price}元 - {max_price}元 (已排除ST/退市/北交所)</li>
                <li><span style='color: yellow;'>市值范围：{min_circ_mv_billion}亿 - {max_circ_mv_billion}亿</span></li>
                <li>必须上涨：今日 **pct_chg > 0** (已排除一字板)</li>
                <li><span style='color: yellow;'>流动性要求：最低换手率 **{min_turnover}%** | 最低成交额 **{min_amount:,.0f}元**</span></li>
                <li>上市天数：>{min_list_days}天</li>
            </ul>
            <p><strong>操作建议：</strong>如果结果仍为 0，请尝试在左侧边栏 <span style='color: yellow;'>**大幅放宽**</span> **最低换手率** 或 **最低成交额**。</p>
            </div>
            """, unsafe_allow_html=True)
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
        
    
    # 5. 高级风险过滤 (V13.11: 使用微调后的默认参数)
    try:
        before_cnt = len(fdf)
        
        # A: 高位大阳线
        if all(c in fdf.columns for c in [f'ma{ma_trend_period}','last_close','pct_chg']):
            # MA 偏离 10% 且涨幅大于阈值
            mask_high_big = (fdf['last_close'] > fdf[f'ma{ma_trend_period}'] * 1.10) & (fdf['pct_chg'] > high_pct_threshold)
            fdf = fdf[~mask_high_big.fillna(False)]

        # B: 下跌途中反抽
        if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
            # 前 3 天累计下跌且今日大阳
            mask_down_rebound = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > high_pct_threshold)
            fdf = fdf[~mask_down_rebound.fillna(False)]

        # C: 巨量冲高（V13.11 略微放宽 VOL_SPIKE_MULT）
        if 'vol_ratio' in fdf.columns:
            mask_vol_spike = (fdf['vol_ratio'] > vol_spike_mult)
            fdf = fdf[~mask_vol_spike.fillna(False)]

        # D: 极端波动（V13.11 略微放宽 MAX_VOLATILITY_10D）
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
# 简易回测模块 (V13.11: 修正回测日期逻辑)
# ----------------------------------------------------
def run_simple_backtest(days, params):
    
    HOLDING_PERIODS = [1, 3, 5]
    status = st.session_state['backtest_status']
    
    container = st.empty()
    with container.container():
        st.subheader(f"📈 简易历史回测结果 (V13.11 稳定版)")
        
        trade_dates_all = get_trade_cal_dates()
        
        if not trade_dates_all:
             st.error("无法获取历史交易日历。")
             return

        try:
            current_trade_idx = trade_dates_all.index(last_trade)
            # 选取包含 last_trade 及其之前的交易日
            trade_dates_all = trade_dates_all[current_trade_idx:]
        except ValueError:
            st.error(f"内部错误：无法定位最近有效交易日 {last_trade}。")
            return


        max_holding = max(HOLDING_PERIODS)
        # 选取需要进行选股的交易日，从历史到最近
        # 需要确保有足够的后续日期来计算 T+N 收益
        select_dates_needed = trade_dates_all[max_holding : days + max_holding]
        select_dates_needed.reverse() # 从最早的选股日开始

        # 确保回测需要的总日期数足够
        if len(select_dates_needed) < days:
             # 如果实际能回测的天数不足，调整 days
             days = len(select_dates_needed)
             select_dates_needed = trade_dates_all[max_holding : days + max_holding]
             select_dates_needed.reverse() 
        
        total_iterations = days
        status['total_days'] = total_iterations
        start_index = status['current_index']
        
        if total_iterations < 1:
            st.warning("交易日不足，无法进行回测。")
            return

        if start_index >= total_iterations:
             st.success(f"回测已完成。累计收益率请查看下方。")
        else:
             st.info(f"回测周期：**{select_dates_needed[0]}** 至 **{select_dates_needed[-1]}**。正在从第 {start_index+1} 天继续...")

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
            select_date = select_dates_needed[i]
            
            # 在 trade_dates_all 中找到 select_date 的索引 (select_dates_needed 是反序的，trade_dates_all 是降序的)
            try:
                # 在降序的 trade_dates_all 中查找 select_date
                base_idx = trade_dates_all.index(select_date)
            except ValueError:
                continue

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
                
                # T+1 日期用于买入价（开盘价）: 降序列表的下一项
                if base_idx + 1 < len(trade_dates_all):
                    next_trade_date = trade_dates_all[base_idx + 1] 
                else:
                    next_trade_date = None
                
                buy_price = np.nan
                if next_trade_date:
                    buy_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=next_trade_date)
                    buy_price = buy_day_data.iloc[0]['open'] if not buy_day_data.empty and 'open' in buy_day_data.columns else np.nan
                
                result['股票'] = f"{top_pick.get('name', 'N/A')}({ts_code})"
                result['买入价 (T+1 开盘)'] = buy_price
                result['评分'] = top_pick['综合评分']
                result['市值 (亿)'] = top_pick['circ_mv_wan'] / 10000.0 if not pd.isna(top_pick['circ_mv_wan']) else np.nan
                
                if buy_price > 0 and not pd.isna(buy_price):
                    
                    for N in HOLDING_PERIODS:
                        # T+N 日期用于卖出价（收盘价）: 降序列表的第 N 项
                        if base_idx + N < len(trade_dates_all):
                             sell_trade_date = trade_dates_all[base_idx + N] 
                        else:
                             continue # 卖出日超出回测范围
                        
                        sell_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=sell_trade_date)
                        
                        if not sell_day_data.empty and 'close' in sell_day_data.columns:
                            sell_price = sell_day_data.iloc[0]['close']
                            result[f'T+{N} 卖出价'] = sell_price
                            
                            if not pd.isna(sell_price):
                                return_pct = (sell_price / buy_price) - 1.0
                                # 假设单日跌停限制在 10%
                                return_pct = max(-0.10, return_pct) 
                                result[f'T+{N} 收益率 (%)'] = return_pct * 100
                        
            status['results'].append(result)
            status['current_index'] = i + 1
            status['progress'] = (i + 1) / total_iterations
            
            pbar.progress(status['progress'], text=f"正在回测 {select_date}... [{i+1}/{total_iterations}]")
            
            # 每隔 2 步刷新或最后一步刷新
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
            
            # 计算累计收益率，只对有买入价的交易计算
            valid_trades = results_df[results_df['买入价 (T+1 开盘)'].notna()]
            
            # 使用几何平均法计算累计收益 (1+r1)*(1+r2)*... - 1
            cumulative_return = (valid_trades[col_name] / 100 + 1).product() - 1
            
            wins = (valid_trades[col_name] > 0).sum()
            total_trades = len(valid_trades)
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
# 实时选股模块 
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
    
    fdf['净流入 (亿)'] = fdf['net_mf_amount'] / 1e8
    final_display_cols[final_display_cols.index('net_mf_amount')] = '净流入 (亿)'
    
    # 确保所有列都存在，避免 KeyError
    for c in final_display_cols:
        if c not in fdf.columns: fdf[c] = np.nan
    
    st.dataframe(fdf[final_display_cols], use_container_width=True, height=500)

    # 修复下载时可能出现的 KeyError (V13.11 确保下载的列也经过了预处理)
    download_cols = [c for c in fdf_full.columns if c not in ['list_date', 'days_since_list', 'circ_mv_wan']]
    if 'net_mf_amount' in download_cols: 
        fdf_full['净流入 (亿)'] = fdf_full['net_mf_amount'] / 1e8
        download_cols.remove('net_mf_amount')
        download_cols.append('净流入 (亿)')

    out_csv = fdf_full[download_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}_V13_11.csv", mime="text/csv")

    st.markdown("### 小结与操作提示（V13.11 平衡过滤版）")
    st.markdown(f"""
- **【结果稳定】** 硬性过滤已通过，总候选 **{len(fdf_full)}** 支，硬性过滤参数已维持 V13.10 的放宽设置。
- **【过滤优化】** **巨量冲高** (当前阈值：**{DEFAULT_VOL_SPIKE_MULT}**) 和 **极端波动** (当前阈值：**{DEFAULT_MAX_VOLATILITY_10D}%**) 阈值已略微放宽，以减少对强势股的误杀。
- **【后续步骤】** 现在您可以运行 **“🚀 运行当日选股”** 查看最新结果，或运行 **“✅ 运行历史回测”** 来测试 V13.11 优化后的效果。
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
