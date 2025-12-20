# -*- coding: utf-8 -*-
"""
选股王 · V30.5 强弱市自适应策略 (实战修正版)
V30.5 更新内容：
1. [核心修复] 去除未来函数：收益计算由 D0收盘价 改为 D1开盘价买入。
2. [性能优化] 内存占用大幅降低 (Float32 + 列裁剪)，支持更长周期回测。
3. [功能增强] 侧边栏增加策略权重微调参数。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time  
warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} # {ts_code: latest_adj_factor}


# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V30.5 实战修正版", layout="wide")
st.title("选股王 · V30.5 强弱市自适应策略（🛡️ 实战修正 / ⚡ 内存优化）")
st.markdown("🎯 **V30.5 策略说明：** 修复未来函数，采用 **次日开盘价** 模拟买入，回测结果更贴近实盘。")
st.markdown("✅ **性能说明：** 已针对 10000 积分权限优化，支持全速拉取和长周期回测。")


# ---------------------------
# 辅助函数 (API调用和数据获取)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    """安全调用 Tushare API"""
    global pro
    if pro is None:
        return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        if kwargs.get('is_index'):
             df = pro.index_daily(**kwargs)
        else:
            df = func(**kwargs)

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception as e:
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    """获取 num_days 个交易日作为选股日"""
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历，请检查 Token 或 Tushare 权限。")
        return []
    
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()


# ----------------------------------------------------------------------
# V30.4.4 新增：按日缓存数据函数 (解决长回测中断问题)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    """安全拉取并缓存单个交易日的数据"""
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    
    return {
        'adj': adj_df,
        'daily': daily_df,
    }


# ----------------------------------------------------------------------
# 核心加速函数：按日期循环拉取历史数据 
# ----------------------------------------------------------------------
def get_all_historical_data(trade_days_list):
    """
    通过循环调用 fetch_and_cache_daily_data 构建全局数据，
    [V30.5 优化]：只保留核心列并压缩数据类型，防止内存溢出。
    """
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 扩大数据获取范围 (150天历史 + 20天未来)
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=20)
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    # 1. 获取所有交易日列表
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty:
        st.error("无法获取交易日历。")
        return False
    
    all_dates = all_trade_dates_df['cal_date'].tolist()
    st.info(f"⏳ 正在按日期循环下载 {start_date} 到 {end_date} 间的**全市场历史数据** (增量缓存)...")

    # 2. 循环获取复权因子 (adj_factor) 和日线行情 (daily)
    adj_factor_data_list = []
    daily_data_list = []
    
    download_progress = st.progress(0, text="下载进度 (按日期循环)...")
    
    for i, date in enumerate(all_dates):
        try:
            cached_data = fetch_and_cache_daily_data(date)
            
            if not cached_data['adj'].empty:
                adj_factor_data_list.append(cached_data['adj'])
                
            if not cached_data['daily'].empty:
                daily_data_list.append(cached_data['daily'])
                
            download_progress.progress((i + 1) / len(all_dates), text=f"下载进度：处理日期 {date}")
        
        except Exception as e:
            st.error(f"❌ 警告：日期 {date} 的数据拉取失败。错误：{e}")
            continue 
            
    download_progress.progress(1.0, text="下载进度：合并数据...")
    download_progress.empty()

    # 3. 合并和处理数据
    if not adj_factor_data_list:
        st.error("❌ 严重错误：无法获取任何复权因子数据。")
        return False
        
    adj_factor_data = pd.concat(adj_factor_data_list)
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_factor_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    if not daily_data_list:
        st.error("❌ 严重错误：无法获取任何历史日线数据。")
        return False

    # [V30.5] 内存优化：只保留核心列
    cols_to_keep = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
    # 过滤掉不存在的列
    valid_cols = [c for c in cols_to_keep if c in daily_data_list[0].columns]
    
    daily_raw_data = pd.concat(daily_data_list)[valid_cols]
    
    # [V30.5] 内存优化：强制转换类型为 float32
    float_cols = ['open', 'high', 'low', 'close', 'vol']
    for col in float_cols:
        if col in daily_raw_data.columns:
            daily_raw_data[col] = pd.to_numeric(daily_raw_data[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])


    # 4. 计算并存储全局固定 QFQ 基准因子
    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    
    if latest_global_date:
        try:
            latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
            st.info(f"✅ 全局 QFQ 基准因子已设置。基准日期: {latest_global_date}，股票数量: {len(GLOBAL_QFQ_BASE_FACTORS)}")
        except Exception as e:
            GLOBAL_QFQ_BASE_FACTORS = {}
    
    st.info(f"✅ 数据预加载完成。内存优化已启用。")
    return True


# ----------------------------------------------------------------------
# 优化的数据获取函数：只从内存中切片 
# ----------------------------------------------------------------------
def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    """ 
    日线数据和复权因子均从预加载的全局变量中切片获取
    """
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
  
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty or not GLOBAL_QFQ_BASE_FACTORS:
        return pd.DataFrame()
        
    latest_adj_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj_factor) or latest_adj_factor < 1e-9:
        return pd.DataFrame() 

    try:
        # 切片数据
        daily_df_full = GLOBAL_DAILY_RAW.loc[ts_code]
        daily_df = daily_df_full.loc[(daily_df_full.index >= start_date) & (daily_df_full.index <= end_date)]
      
        adj_factor_series_full = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        adj_factor_series = adj_factor_series_full.loc[(adj_factor_series_full.index >= start_date) & (adj_factor_series_full.index <= end_date)]
        
    except KeyError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    
    if daily_df.empty or adj_factor_series.empty: return pd.DataFrame()
            
    # 合并
    df = daily_df.merge(adj_factor_series.rename('adj_factor'), left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    if df.empty: return pd.DataFrame()
    
    # 复权计算
    df = df.sort_index()
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df['trade_date'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')
    df = df.sort_values('trade_date').set_index('trade_date_str')
    
    # 覆盖原列
    for col in ['open', 'high', 'low', 'close']:
        if col + '_qfq' in df.columns:
            df[col] = df[col + '_qfq']
            
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 

# ----------------------------------------------------------------------
# [V30.5 核心修改] 计算未来收益 (D1 Open 买入)
# ----------------------------------------------------------------------
def get_future_prices(ts_code, selection_date, d0_qfq_close, days_ahead=[1, 3, 5]):
    
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_date_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date_future = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    # 获取未来 N 日数据
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date_future, end_date=end_date_future)
    
    results = {}
    for n in days_ahead: results[f'Return_D{n}'] = np.nan

    if hist.empty or len(hist) < 1:
        return results
        
    # [修正]：买入价格为 D1 的 Open (即 hist 的第 0 行 open)
    try:
        buy_price = hist.iloc[0]['open']
    except:
        return results

    if buy_price <= 1e-9: # 避免除以0
        return results

    # 计算收益
    for n in days_ahead:
        col_name = f'Return_D{n}'
        # D1 收益通常指持有1天，即 D1 Open 买，D1 Close 卖 (日内) 或 D2 Close 卖?
        # 惯例：D1 收益 = (D1 Close - D1 Open) / D1 Open (日内浮盈)
        # 或者 D+N 代表第 N 天的收盘价相对于买入价
        
        idx = n - 1
        if len(hist) > idx:
            sell_price = hist.iloc[idx]['close']
            results[col_name] = (sell_price / buy_price - 1) * 100
            
    return results

# ----------------------------------------------------------------------
# 核心函数 2: compute_indicators
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    """计算 MACD, MA20, 波动率, 60日位置等指标"""
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    
    res = {}
    if df.empty or len(df) < 3 or 'close' not in df.columns: 
        return res
        
    if len(df) >= 2:
       df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    else:
         df['pct_chg'] = 0.0
         
    close = df['close']
    res['last_close'] = close.iloc[-1] 
    
    # MACD 计算 
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    else: res['macd_val'] = np.nan
        
    # MA20 计算
    if len(close) >= 20:
        res['ma20'] = close.tail(20).mean()
    else: res['ma20'] = np.nan
        
    # 波动率
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    # 60日位置
    if len(df) >= 60:
        hist_60 = df.tail(60)
        min_low = hist_60['low'].min()
        max_high = hist_60['high'].max()
        current_close = hist_60['close'].iloc[-1]
        
        if max_high == min_low: res['position_60d'] = 50.0 
        else: res['position_60d'] = (current_close - min_low) / (max_high - min_low) * 100
    else: res['position_60d'] = np.nan 
    
    return res

# ----------------------------------------------------------------------
# 核心函数 3: get_market_state
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*12)
def get_market_state(trade_date):
    """判断沪深300指数是否处于 MA20 之上"""
    start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    index_data = safe_get('daily', ts_code='000300.SH', start_date=start_date, end_date=trade_date, is_index=True)
    
    if index_data.empty or 'close' not in index_data.columns:
        return 'Weak'

    index_data['close'] = pd.to_numeric(index_data['close'], errors='coerce').astype(float)
    index_data = index_data.sort_values('trade_date', ascending=True)

    if len(index_data) < 20: return 'Weak' 

    latest_close = index_data.iloc[-1]['close']
    ma20 = index_data['close'].tail(20).mean()

    return 'Strong' if latest_close > ma20 else 'Weak'
      
        
# ----------------------------------------------------
# 侧边栏参数 
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input("选择回测结束日期", value=datetime.now().date(), max_value=datetime.now().date())
    
    BACKTEST_DAYS = int(st.number_input(
        "**自动回测天数 (N)**", 
        value=30, 
        step=1, 
        min_value=1, 
        help="建议拥有10000积分的用户尝试 50-100 天长回测。"
    ))
    
    st.markdown("---")
    st.header("⚙️ 策略微调 (V30.5)")
    st.info("调整弱市防御模式下的因子权重：")
    WEIGHT_MACD = st.slider("MACD 权重 (反弹力度)", 0.0, 1.0, 0.45, step=0.05)
    WEIGHT_VOL = st.slider("低波动权重 (安全度)", 0.0, 1.0, 0.45, step=0.05)
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = int(st.number_input("最终入围评分数量", value=100, step=1)) 
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=10, step=1))
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=5, step=1))
    
    st.markdown("---")
    st.header("🛒 过滤条件")
    MIN_PRICE = st.number_input("最低股价", value=5.0, step=0.5) 
    MAX_PRICE = st.number_input("最高股价", value=300.0, step=5.0)
    MIN_TURNOVER = st.number_input("最低换手率 (%)", value=3.0, step=0.5) 
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿)", value=20.0, step=1.0)
    MIN_AMOUNT_MILLIONS = st.number_input("最低成交额 (亿)", value=1.0, step=0.1) 
    MIN_AMOUNT = MIN_AMOUNT_MILLIONS * 100000000 

# ---------------------------
# Token 输入
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ----------------------------------------------------------------------
# 核心回测逻辑函数 
# ----------------------------------------------------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS):
    """为单个交易日运行选股和回测逻辑"""
    global GLOBAL_DAILY_RAW
    
    market_state = get_market_state(last_trade)
 
    # 拉取全市场 Daily 数据 
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失或拉取失败：{last_trade}"

    pool_raw = daily_all.reset_index(drop=True) 
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date') 
    REQUIRED_BASIC_COLS = ['ts_code','turnover_rate','amount','total_mv','circ_mv'] 
    daily_basic = safe_get('daily_basic', trade_date=last_trade, fields=','.join(REQUIRED_BASIC_COLS))
    mf_raw = safe_get('moneyflow', trade_date=last_trade)
    
    pool_merged = pool_raw.copy()

    if not stock_basic.empty and 'name' in stock_basic.columns:
        pool_merged = pool_merged.merge(stock_basic[['ts_code','name','list_date']], on='ts_code', how='left')
    else:
        pool_merged['name'] = pool_merged['ts_code']
        pool_merged['list_date'] = '20000101'
        
    if not daily_basic.empty:
        cols_to_merge = [c for c in REQUIRED_BASIC_COLS if c in daily_basic.columns]
        if 'amount' in pool_merged.columns and 'amount' in cols_to_merge: 
            pool_merged = pool_merged.drop(columns=['amount'])
        pool_merged = pool_merged.merge(daily_basic[cols_to_merge], on='ts_code', how='left')
    
    required_daily_basic_cols = ['turnover_rate','amount','total_mv','circ_mv']
    for col in required_daily_basic_cols:
        if col not in pool_merged.columns:
            pool_merged[col] = 0.0
            
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in']
        for c in possible:
            if c in mf_raw.columns:
                moneyflow = mf_raw[['ts_code', c]].rename(columns={c:'net_mf'}).fillna(0)
                break            
    
    if not moneyflow.empty:
        pool_merged = pool_merged.merge(moneyflow, on='ts_code', how='left')
    
    if 'net_mf' not in pool_merged.columns:
        pool_merged['net_mf'] = 0.0 
    pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0) 
    
    df = pool_merged.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce') 
    df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000 
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000 
    df['name'] = df['name'].astype(str)
    
    # 过滤
    mask_st = df['name'].str.contains('ST|退', case=False, na=False)
    df = df[~mask_st]
    mask_bj = df['ts_code'].str.startswith('92') 
    df = df[~mask_bj]
    
    TODAY = datetime.strptime(last_trade, "%Y%m%d")
    MIN_LIST_DAYS = 120 
    df['list_date_dt'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
    df['days_listed'] = (TODAY - df['list_date_dt']).dt.days
    mask_new_all = df['days_listed'] < MIN_LIST_DAYS
    df = df[~mask_new_all] 
    
    mask_price = (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)
    df = df[mask_price]
    mask_circ_mv = df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS
    df = df[mask_circ_mv] 
    mask_turn = df['turnover_rate'] >= MIN_TURNOVER 
    df = df[mask_turn]
    mask_amt = df['amount'] * 1000 >= MIN_AMOUNT
    df = df[mask_amt]
    
    df = df.reset_index(drop=True)
    if len(df) == 0: return pd.DataFrame(), f"硬性过滤后无股票"

    # 初选
    limit_mf = int(FINAL_POOL * 0.5)
    df_mf = df.sort_values('net_mf', ascending=False).head(limit_mf).copy()
    limit_pct = FINAL_POOL - len(df_mf)
    existing_codes = set(df_mf['ts_code'])
    df_pct = df[~df['ts_code'].isin(existing_codes)].sort_values('pct_chg', ascending=False).head(limit_pct).copy()
    final_candidates = pd.concat([df_mf, df_pct]).reset_index(drop=True)
    
    if not GLOBAL_DAILY_RAW.empty:
        try:
            codes_with_d0_data = GLOBAL_DAILY_RAW.loc[(slice(None), last_trade), :].index.get_level_values('ts_code').unique()
            final_candidates = final_candidates[final_candidates['ts_code'].isin(codes_with_d0_data)].copy()
        except KeyError:
             return pd.DataFrame(), f"缓存中缺失数据"
            
    if final_candidates.empty:
        return pd.DataFrame(), f"初步筛选为空"

    # 深度评分
    records = []
    for row in final_candidates.itertuples():
        ts_code = row.ts_code
        raw_close = getattr(row, 'close', np.nan)
        
        ind = compute_indicators(ts_code, last_trade) 
        d0_qfq_close = ind.get('last_close', np.nan)
        d0_ma20 = ind.get('ma20', np.nan) 
        d0_position_60d = ind.get('position_60d', np.nan)

        # 弱市严格过滤
        if market_state == 'Weak':
            if pd.isna(d0_ma20) or d0_ma20 == 0 or d0_qfq_close < d0_ma20:
                 continue 
            if pd.isna(d0_position_60d) or d0_position_60d > 20.0:
                continue 

        if pd.notna(d0_qfq_close) and d0_qfq_close > 1e-9:
            # 收益率计算 (内部已改为 D1 Open 买入)
            future_returns = get_future_prices(ts_code, last_trade, d0_qfq_close) 
            
            rec = {
                'ts_code': ts_code, 'name': getattr(row, 'name', ts_code),
                'Close': raw_close, 
                'Circ_MV (亿)': getattr(row, 'circ_mv_billion', np.nan),
                'Pct_Chg (%)': getattr(row, 'pct_chg', 0),
                'net_mf': getattr(row, 'net_mf', 0),
                'macd': ind.get('macd_val', np.nan), 
                'volatility': ind.get('volatility', np.nan),
                'position_60d': d0_position_60d, 
                'Return_D1 (%)': future_returns.get('Return_D1', np.nan),
                'Return_D3 (%)': future_returns.get('Return_D3', np.nan),
                'Return_D5 (%)': future_returns.get('Return_D5', np.nan),
            }
            records.append(rec)
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), f"评分列表为空"

    # 归一化
    def normalize(series):
        series_nn = series.dropna() 
        if series_nn.empty or series_nn.max() == series_nn.min(): return pd.Series([0.5] * len(series), index=series.index)
        return (series - series_nn.min()) / (series_nn.max() - series_nn.min() + 1e-9)

    fdf['s_mf'] = normalize(fdf['net_mf'])
    fdf['s_volatility'] = normalize(fdf['volatility']) 
    
    # 策略评分
    if market_state == 'Strong':
        fdf['策略'] = '绝对MACD优势'
        fdf_strong = fdf[fdf['macd'] > 0].copy()
        if fdf_strong.empty:
            fdf['综合评分'] = 0.0 
            fdf = fdf[fdf['综合评分'] > 10000000] 
        else:
            fdf_strong['Score_MACD'] = fdf_strong['macd'] * 10000
            fdf_strong['Score_Aux'] = (fdf_strong['s_volatility'].rsub(1) * 0.3) + (fdf_strong['s_mf'] * 0.7)
            fdf_strong['综合评分'] = fdf_strong['Score_MACD'] + fdf_strong['Score_Aux']
            fdf = fdf_strong.sort_values('综合评分', ascending=False)
            
    else: # Weak Market
        fdf['策略'] = '极致反弹防御'
        fdf['s_macd'] = normalize(fdf['macd']) 
        
        # [V30.5] 使用侧边栏参数
        w_volatility = WEIGHT_VOL  
        w_macd = WEIGHT_MACD
        w_mf = max(0, 1.0 - w_macd - w_volatility)
        
        score = (
            fdf['s_volatility'].rsub(1).fillna(0.5) * w_volatility + 
            fdf['s_macd'].fillna(0.5) * w_macd +
            fdf['s_mf'].fillna(0.5) * w_mf 
        )
        
        fdf['综合评分'] = score * 100
        fdf = fdf.sort_values('综合评分', ascending=False)
        
    fdf = fdf.reset_index(drop=True)
    fdf.index += 1

    return fdf.head(TOP_BACKTEST).copy(), None

# ---------------------------
# 主运行块 
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测"):
    
    trade_days_str = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days_str:
        st.error("无法获取交易日列表，请检查日期或 Token。")
        st.stop()
    
    preload_success = get_all_historical_data(trade_days_str)
    if not preload_success:
        st.error("❌ 历史数据预加载失败。")
        st.stop()
    st.success("✅ 数据就绪！开始内存优化回测...")
    
    st.header(f"📈 回测进度 ({BACKTEST_DAYS} days)")
    
    results_list = []
    total_days = len(trade_days_str)
    
    progress_text = st.empty()
    my_bar = st.progress(0)
    
    for i, trade_date in enumerate(trade_days_str):
        progress_text.text(f"⏳ 处理中：{trade_date}")
        daily_result_df, error = run_backtest_for_a_day(
            trade_date, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS
        )
        if error:
            pass # 忽略单日错误，保持界面整洁
        elif not daily_result_df.empty:
            daily_result_df['Trade_Date'] = trade_date
            results_list.append(daily_result_df)
            
        my_bar.progress((i + 1) / total_days)

    progress_text.text("✅ 完成！")
    my_bar.empty()
    
    if not results_list:
        st.error("无回测结果。")
        st.stop()
        
    all_results = pd.concat(results_list)
    if all_results['Trade_Date'].dtype != 'object':
        all_results['Trade_Date'] = all_results['Trade_Date'].astype(str)
        
    valid_days_count = len(all_results['Trade_Date'].unique())
    
    st.header(f"📊 最终回测报告 (Top {TOP_BACKTEST} | {valid_days_count} 个交易日)")
    
    cols = st.columns(3)
    for idx, n in enumerate([1, 3, 5]):
        col_name = f'Return_D{n} (%)' 
        valid_returns = all_results.dropna(subset=[col_name])

        if not valid_returns.empty:
            avg_return = valid_returns[col_name].mean()
            # 胜率计算：收益 > 0
            hit_rate = (valid_returns[col_name] > 0).sum() / len(valid_returns) * 100 
            total_count = len(valid_returns)
        else:
            avg_return = 0.0
            hit_rate = 0.0
        
        with cols[idx]:
            st.metric(
                f"D+{n} 收益/胜率", 
                f"{avg_return:.2f}% / {hit_rate:.1f}%",
                help=f"基于次日开盘价买入。样本数：{len(valid_returns)}"
            )

    st.header("📋 每日明细 (按综合评分排序)")
    display_cols = ['Trade_Date', '策略', 'name', 'ts_code', '综合评分', 
                    'Close', 'Pct_Chg (%)', 'Return_D1 (%)', 'Return_D3 (%)']
    st.dataframe(all_results[display_cols].sort_values('Trade_Date', ascending=False), use_container_width=True)
