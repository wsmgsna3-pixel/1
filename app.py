# -*- coding: utf-8 -*-
"""
选股王 · V15.0 动量趋势增强版：中期动量 + 趋势突破 (鲁棒性修复+策略优化)
核心优化：
1. 【**策略优化 V15.0**】：从资金流主导转向动量趋势主导，采用20日动量+均线趋势+突破信号组合
   - 新权重：动量(0.40) + 趋势(0.25) + 量价(0.15) + 突破(0.10) + 防御(0.10) = 1.00
   - 新增20日动量、均线排列、量比、突破新高等多个有效因子
   
2. 【**过滤条件优化**】：放宽选股范围，提高策略灵活性
   - 最低股价从10元降至5元
   - 最低流通市值从20亿降至10亿
   - 最低换手率从2%降至1%
   - 最低成交额从0.6亿降至0.3亿
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
st.set_page_config(page_title="选股王 · V15.0 动量趋势增强版", layout="wide")
st.title("选股王 · V15.0 动量趋势增强版（🚀 中期动量 / 趋势突破 - 策略优化）")
st.markdown("🎯 **V15.0 策略说明：** **动量趋势主导，注重中期动能。** 核心权重：**20日动量 0.40** + **趋势排列 0.25** + **量价配合 0.15** + **突破新高 0.10** + **防御因子 0.10**。")
st.markdown("✅ **技术说明：** 启动加载时间较长 (5-8 分钟)，但数据可靠，回测计算速度极快。")


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
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception as e:
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    """获取 num_days 个交易日作为选股日"""
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 2)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历，请检查 Token 或 Tushare 权限。")
        return []
    
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()


# ----------------------------------------------------------------------
# ⭐️ V15.0 核心：按日期循环拉取历史数据 (鲁棒性保证)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def get_all_historical_data(trade_days_list):
    """
    V15.0 鲁棒修复：改用按日期循环拉取日线和复权因子，确保数据完整性。
    """
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 扩大数据获取范围
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=20)
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    st.info(f"⏳ 正在按日期循环下载 {start_date} 到 {end_date} 间的**全市场历史数据**...")
    
    # 1. 获取所有交易日列表
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty:
        st.error("无法获取交易日历。")
        return False
    
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    # 2. 循环获取复权因子 (adj_factor) 和日线行情 (daily)
    adj_factor_data_list = []
    daily_data_list = []
    
    download_progress = st.progress(0, text="下载进度 (按日期循环)...")
    
    for i, date in enumerate(all_dates):
        download_progress.progress((i + 1) / len(all_dates), text=f"下载进度：处理日期 {date}")
        
        # 获取复权因子
        adj_df = safe_get('adj_factor', trade_date=date)
        if not adj_df.empty:
            adj_factor_data_list.append(adj_df)
            
        # 获取日线行情
        daily_df = safe_get('daily', trade_date=date)
        if not daily_df.empty:
            daily_data_list.append(daily_df)
            
        # 避免过于频繁的 API 调用，Tushare 有 QPS 限制
    
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

    daily_raw_data = pd.concat(daily_data_list)
    GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])


    # 4. 计算并存储全局固定 QFQ 基准因子
    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    
    if latest_global_date:
        try:
            latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
            st.info(f"✅ 全局 QFQ 基准因子已设置。基准日期: {latest_global_date}，股票数量: {len(GLOBAL_QFQ_BASE_FACTORS)}")
        except Exception as e:
            st.error(f"无法设置全局 QFQ 基准因子: {e}")
            GLOBAL_QFQ_BASE_FACTORS = {}
    
    
    # 5. 诊断信息
    st.info(f"✅ 数据预加载完成。日线数据总条目：{len(GLOBAL_DAILY_RAW)}，复权因子总条目：{len(GLOBAL_ADJ_FACTOR)}")

    # 检查数据条目是否足够 
    if len(GLOBAL_DAILY_RAW) < 100000:
         st.warning("⚠️ 警告：总条目数偏低。请再次确认 Tushare 积分和 API 访问权限。")
         
    return True


# ----------------------------------------------------------------------
# 优化的数据获取函数：只从内存中切片 (前复权计算核心)
# ----------------------------------------------------------------------
def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    """ 
    日线数据和复权因子均从预加载的全局变量中切片获取，
    复权基准使用 GLOBAL_QFQ_BASE_FACTORS 中存储的统一因子。
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
            
    # 合并原始价格和复权因子
    df = daily_df.merge(adj_factor_series.rename('adj_factor'), left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    if df.empty: return pd.DataFrame()
    
    # 复权计算逻辑
    df = df.sort_index()
    
    # 使用全局固定基准进行向量化复权计算
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            # QFQ Price = Raw Price * (Adj Factor / Global Base Factor)
            df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df['trade_date'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')
    df = df.sort_values('trade_date').set_index('trade_date_str')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col + '_qfq']
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 

# ----------------------------------------------------------------------
# 核心函数：get_future_prices (接受 D0 QFQ 价格)
# ----------------------------------------------------------------------

def get_future_prices(ts_code, selection_date, d0_qfq_close, days_ahead=[1, 3, 5]):
    
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_date_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date_future = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    
    selection_price_adj = d0_qfq_close 
    
    # 1. 获取未来 N 日数据 (用于计算 D+N 的分子)
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date_future, end_date=end_date_future)
    if hist.empty or 'close' not in hist.columns:
        results = {}
        for n in days_ahead: results[f'Return_D{n}'] = np.nan
        return results
        
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    hist = hist.dropna(subset=['close'])
    hist = hist.reset_index(drop=True) 
    results = {}
    
    # 2. 计算收益
    for n in days_ahead:
        col_name = f'Return_D{n}'
        
        if pd.notna(selection_price_adj) and selection_price_adj > 1e-9:
            if len(hist) >= n:
                future_price = hist.iloc[n-1]['close']
                results[col_name] = (future_price / selection_price_adj - 1) * 100
            else:
                results[col_name] = np.nan
        else:
            results[col_name] = np.nan 
            
    return results


# ----------------------------------------------------------------------
# ⭐️ V15.0 新增：增强版指标计算函数
# ----------------------------------------------------------------------
def compute_indicators_v2(ts_code, end_date):
    """增强版指标计算 - 新增动量、趋势、量价、突破等因子"""
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    
    # 获取 QFQ 数据，用于计算所有指标
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    
    res = {}
    if df.empty or 'close' not in df.columns: 
        return res
        
    df['close'] = pd.to_numeric(df['close'], errors='coerce').astype(float)
    df['high'] = pd.to_numeric(df['high'], errors='coerce').astype(float)
    df['low'] = pd.to_numeric(df['low'], errors='coerce').astype(float)
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce').fillna(0)
    
    if len(df) >= 2:
         df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    else:
         df['pct_chg'] = 0.0
         
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['vol']
    
    res['last_close'] = close.iloc[-1] if len(close) > 0 else np.nan
    
    # 1. 动量因子 (20日涨幅)
    if len(close) >= 20:
        res['momentum_20d'] = (close.iloc[-1] / close.iloc[-20] - 1) * 100
    else:
        res['momentum_20d'] = 0
    
    # 2. 趋势因子 (均线排列)
    if len(close) >= 20:
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        
        # 均线多头排列得分
        trend_score = 0
        if ma5.iloc[-1] > ma10.iloc[-1]: trend_score += 1
        if ma10.iloc[-1] > ma20.iloc[-1]: trend_score += 1
        if close.iloc[-1] > ma5.iloc[-1]: trend_score += 1
        res['trend_score'] = trend_score / 3 * 100  # 归一化到0-100
    else:
        res['trend_score'] = 0
    
    # 3. 量价关系
    if len(vol) >= 10:
        # 量比：当日成交量/5日均量
        avg_vol_5d = vol.rolling(5).mean().iloc[-1]
        if avg_vol_5d > 0:
            res['volume_ratio'] = vol.iloc[-1] / avg_vol_5d
        else:
            res['volume_ratio'] = 1
        
        # 换手率稳定性
        vol_std = vol.tail(10).std()
        vol_mean = vol.tail(10).mean()
        if vol_mean > 0:
            res['volume_stability'] = (1 - vol_std / vol_mean) * 100
    else:
        res['volume_ratio'] = 1
        res['volume_stability'] = 50
    
    # 4. 突破因子 (创20日新高)
    if len(high) >= 20:
        highest_20d = high.tail(20).max()
        current_high = high.iloc[-1]
        res['breakout_score'] = 100 if current_high >= highest_20d else 0
    else:
        res['breakout_score'] = 0
    
    # 5. 防御因子 (60日位置 + 波动率)
    if len(df) >= 60:
        hist_60 = df.tail(60)
        min_low = hist_60['low'].min()
        max_high = hist_60['high'].max()
        current_close = hist_60['close'].iloc[-1]
        
        if max_high > min_low:
            res['position_60d'] = (current_close - min_low) / (max_high - min_low) * 100
        else:
            res['position_60d'] = 50
    
    # 计算波动率 (20日)
    if len(df) >= 20:
        returns = close.pct_change().dropna()
        res['volatility_20d'] = returns.tail(20).std() * np.sqrt(252) * 100  # 年化波动率
    
    # 保留原有指标用于兼容
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    else: 
        res['macd_val'] = np.nan
        
    vols = df['vol'].tolist()
    if len(vols) >= 6 and vols[-6:-1] and np.mean(vols[-6:-1]) > 1e-9:
        res['vol_ratio'] = vols[-1] / np.mean(vols[-6:-1])
    else: 
        res['vol_ratio'] = np.nan
        
    res['10d_return'] = (close.iloc[-1]/close.iloc[-10] - 1) * 100 if len(close)>=10 and close.iloc[-10]!=0 else 0
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    return res


# ----------------------------------------------------
# 侧边栏参数 (V15.0 优化：更宽松的过滤条件)
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input("选择**回测结束日期**", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("**自动回测天数 (N)**", value=20, step=1, min_value=1, max_value=50, help="程序将自动回测最近 N 个交易日。建议设置为 20 天以获得更可靠的统计数据。"))
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = int(st.number_input("最终入围评分数量 (M)", value=100, step=1, min_value=1)) 
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=10, step=1))
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=3, step=1, min_value=1)) 
    
    st.markdown("---")
    st.header("🛒 V15.0 灵活过滤条件 (更宽松)")
    MIN_PRICE = st.number_input("最低股价 (元)", value=5.0, step=0.5, min_value=0.1, help="从10元降至5元，扩大选股范围")
    MAX_PRICE = st.number_input("最高股价 (元)", value=500.0, step=5.0, min_value=1.0, help="从300元升至500元，包含更多高价优质股")
    MIN_TURNOVER = st.number_input("最低换手率 (%)", value=1.0, step=0.5, min_value=0.1, help="从2%降至1%，减少过滤掉低换手潜力股")
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿元)", value=10.0, step=1.0, min_value=1.0, help="从20亿降至10亿，扩大中小盘股选择")
    MIN_AMOUNT_MILLIONS = st.number_input("最低成交额 (亿元)", value=0.3, step=0.1, min_value=0.1, help="从0.6亿降至0.3亿，提高策略灵活性")
    MIN_AMOUNT = MIN_AMOUNT_MILLIONS * 100000000 

# ---------------------------
# Token 输入与初始化 (保持不变)
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ---------------------------
# ⭐️ V15.0 核心回测逻辑函数 (增强版)
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS):
    """为单个交易日运行选股和回测逻辑 - V15.0 增强版"""
    global GLOBAL_DAILY_RAW
    
    # 1. 拉取全市场 Daily 数据 (今日快照)
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty or 'ts_code' not in daily_all.columns: 
        return pd.DataFrame(), f"数据缺失或拉取失败：{last_trade}"

    pool_raw = daily_all.reset_index(drop=True) 
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date') 
    REQUIRED_BASIC_COLS = ['ts_code','turnover_rate','amount','total_mv','circ_mv'] 
    daily_basic = safe_get('daily_basic', trade_date=last_trade, fields=','.join(REQUIRED_BASIC_COLS))
    mf_raw = safe_get('moneyflow', trade_date=last_trade)
    pool_merged = pool_raw.copy()

    # 数据合并
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
    
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in']
        for c in possible:
            if c in mf_raw.columns:
                moneyflow = mf_raw[['ts_code', c]].rename(columns={c:'net_mf'}).fillna(0)
                break            
    if not moneyflow.empty:
        pool_merged = pool_merged.merge(moneyflow, on='ts_code', how='left')
        
    pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0) 
    
    # 检查 'turnover_rate' 字段
    if 'turnover_rate' not in pool_merged.columns:
        pool_merged['turnover_rate'] = 0.0 
    
    pool_merged['turnover_rate'] = pool_merged['turnover_rate'].fillna(0) 
    
   
    # 3. 执行硬性条件过滤
    df = pool_merged.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce') 
    df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000 # 转换为万元
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000 
    df['name'] = df['name'].astype(str)
    
    # 过滤 ST 股/退市股/北交所
    mask_st = df['name'].str.contains('ST|退', case=False, na=False)
    df = df[~mask_st]
    mask_bj = df['ts_code'].str.startswith('92') 
    df = df[~mask_bj]
    
    # 新股过滤
    TODAY = datetime.strptime(last_trade, "%Y%m%d")
    MIN_LIST_DAYS = 120
    df['list_date_dt'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
    df['days_listed'] = (TODAY - df['list_date_dt']).dt.days
    
    mask_new_all = df['days_listed'] < MIN_LIST_DAYS
    df = df[~mask_new_all] 
    
    # 过滤价格
    mask_price = (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)
    df = df[mask_price]
    
    # 过滤流通市值
    mask_circ_mv = df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS
    df = df[mask_circ_mv] 
    
    # 过滤换手率
    mask_turn = df['turnover_rate'] >= MIN_TURNOVER 
    df = df[mask_turn]
    
    # 过滤成交额
    mask_amt = df['amount'] * 1000 >= MIN_AMOUNT
    df = df[mask_amt]
    
    df = df.reset_index(drop=True)
    initial_candidate_count = len(df)

    if initial_candidate_count == 0: 
        return pd.DataFrame(), f"硬性过滤后无股票：{last_trade}"

    # 4. V15.0 新增：计算动量趋势指标进行预筛选
    momentum_scores = []
    trend_scores = []
    
    for row in df.itertuples():
        ts_code = row.ts_code
        ind = compute_indicators_v2(ts_code, last_trade)
        momentum_scores.append(ind.get('momentum_20d', 0))
        trend_scores.append(ind.get('trend_score', 0))
    
    df['momentum_20d'] = momentum_scores
    df['trend_score'] = trend_scores
    
    # V15.0 动量趋势预筛选：要求20日动量>0且趋势得分>33
    if len(df) > 0:
        momentum_mask = df['momentum_20d'] > 0
        trend_mask = df['trend_score'] > 33  # 至少满足一个趋势条件
        df = df[momentum_mask & trend_mask].copy()
    
    if len(df) == 0:
        return pd.DataFrame(), f"动量趋势筛选后无股票：{last_trade}"

    # 5. 遴选决赛名单
    # V15.0 筛选：使用20日动量和趋势得分作为入围标准
    limit_momentum = int(FINAL_POOL * 0.6)  # 60% 按动量选
    limit_trend = FINAL_POOL - limit_momentum  # 40% 按趋势选
    
    df_momentum = df.sort_values('momentum_20d', ascending=False).head(limit_momentum).copy()
    
    existing_codes = set(df_momentum['ts_code'])
    df_trend = df[~df['ts_code'].isin(existing_codes)].sort_values('trend_score', ascending=False).head(limit_trend).copy()
    
    final_candidates = pd.concat([df_momentum, df_trend]).reset_index(drop=True)
    
    # 鲁棒性强化：检查候选股在内存中的 D0 QFQ 数据是否完整
    if not GLOBAL_DAILY_RAW.empty:
        try:
            codes_with_d0_data = GLOBAL_DAILY_RAW.loc[(slice(None), last_trade), :].index.get_level_values('ts_code').unique()
            final_candidates = final_candidates[final_candidates['ts_code'].isin(codes_with_d0_data)].copy()
        except KeyError:
            return pd.DataFrame(), f"跳过 {last_trade}：核心历史数据缓存中缺失回测日 {last_trade} 的全部数据 (已通过鲁棒性检查过滤)"
            
    if final_candidates.empty:
        return pd.DataFrame(), f"跳过 {last_trade}：评分列表为空. 原因：在 {len(final_candidates)} 个已检查的候选股中，所有股票的 D0 QFQ 价格均无效。"

    # 6. V15.0 深度评分 (使用新因子和新权重)
    records = []
    
    for row in final_candidates.itertuples():
        ts_code = row.ts_code
        
        raw_close = getattr(row, 'close', np.nan)
        
        # 计算增强版指标
        ind = compute_indicators_v2(ts_code, last_trade) 
        d0_qfq_close = ind.get('last_close', np.nan) # 提取 D0 QFQ Close Price

        # 仅当 D0 QFQ Close Price 有效且非零时，才进行收益率计算和记录
        if pd.notna(d0_qfq_close) and d0_qfq_close > 1e-9:
            
            future_returns = get_future_prices(ts_code, last_trade, d0_qfq_close) 
            
            rec = {
                'ts_code': ts_code, 
                'name': getattr(row, 'name', ts_code),
                'Close': raw_close, 
                'Circ_MV (亿)': getattr(row, 'circ_mv_billion', np.nan),
                'Pct_Chg (%)': getattr(row, 'pct_chg', 0), 
                'turnover': getattr(row, 'turnover_rate', 0),
                'net_mf': getattr(row, 'net_mf', 0),
                # V15.0 新增因子
                'momentum_20d': ind.get('momentum_20d', 0),
                'trend_score': ind.get('trend_score', 0),
                'volume_ratio': ind.get('volume_ratio', 1),
                'volume_stability': ind.get('volume_stability', 50),
                'breakout_score': ind.get('breakout_score', 0),
                'position_60d': ind.get('position_60d', 50),
                'volatility_20d': ind.get('volatility_20d', 30),
                # 保留原有因子
                'vol_ratio': ind.get('vol_ratio', np.nan), 
                'macd': ind.get('macd_val', np.nan),
                '10d_return': ind.get('10d_return', np.nan), 
                'volatility': ind.get('volatility', np.nan), 
            }
            
            rec.update({
                'Return_D1 (%)': future_returns.get('Return_D1', np.nan),
                'Return_D3 (%)': future_returns.get('Return_D3', np.nan),
                'Return_D5 (%)': future_returns.get('Return_D5', np.nan),
            })
            
            records.append(rec)
    
    fdf = pd.DataFrame(records)
    
    if fdf.empty: 
        return pd.DataFrame(), f"跳过 {last_trade}：评分列表为空. 原因：在 {len(final_candidates)} 个已检查的候选股中，所有股票的 D0 QFQ 价格均无效。"

    # 7. V15.0 归一化与策略精调评分 
    def normalize(series):
        series_nn = series.dropna() 
        if series_nn.empty or series_nn.max() == series_nn.min(): 
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series_nn.min()) / (series_nn.max() - series_nn.min() + 1e-9)

    # 归一化所有使用的因子
    fdf['s_momentum'] = normalize(fdf['momentum_20d'])          # 动量越大越好
    fdf['s_trend'] = normalize(fdf['trend_score'])              # 趋势越强越好
    
    # 量比得分：1.5-3.0为最佳区间
    fdf['s_volume'] = np.where(
        (fdf['volume_ratio'] >= 1.5) & (fdf['volume_ratio'] <= 3.0),
        1.0,
        np.where(
            fdf['volume_ratio'] < 1.5, 
            fdf['volume_ratio'] / 1.5, 
            3.0 / fdf['volume_ratio']
        )
    )
    
    fdf['s_breakout'] = fdf['breakout_score'] / 100           # 突破得分 (0或1)
    
    # 位置得分：40-70分最好，过高或过低都减分
    position_score = np.where(
        (fdf['position_60d'] >= 40) & (fdf['position_60d'] <= 70),
        1.0,
        np.where(
            fdf['position_60d'] < 40, 
            fdf['position_60d'] / 40, 
            (100 - fdf['position_60d']) / 30
        )
    )
    fdf['s_position'] = position_score
    
    # 波动率得分：越低越好
    fdf['s_volatility'] = 1 - normalize(fdf['volatility_20d'])
    
    # 🚨 V15.0 策略权重 (动量趋势增强)
    w_momentum = 0.40      # 动量因子 (正向)
    w_trend = 0.25         # 趋势因子 (正向)
    w_volume = 0.15        # 量价关系 (正向)
    w_breakout = 0.10      # 突破因子 (正向)
    w_defensive = 0.10     # 防御因子 (位置+波动率)
    
    # 计算综合评分
    score = (
        fdf['s_momentum'].fillna(0.5) * w_momentum +
        fdf['s_trend'].fillna(0.5) * w_trend +
        fdf['s_volume'].fillna(0.5) * w_volume +
        fdf['s_breakout'].fillna(0) * w_breakout +
        fdf['s_position'].fillna(0.5) * 0.05 +  # 位置因子占防御权重的一半
        fdf['s_volatility'].fillna(0.5) * 0.05   # 波动率因子占防御权重的一半
    )
    
    fdf['综合评分'] = score * 100
    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index += 1

    return fdf.head(TOP_BACKTEST).copy(), None

# ---------------------------
# 主运行块 
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测 (V15.0 动量趋势增强版)"):
    
    st.warning("⚠️ **请务必先清除 Streamlit 缓存！**（右上角三点菜单 -> Settings -> Clear Cache）这是让程序强制重新下载数据的关键一步。")
   
    trade_days_str = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days_str:
        st.error("无法获取交易日列表，请检查日期或 Token。")
        st.stop()
    
    # ----------------------------------------------------------------------
    # 核心优化步骤：预加载所有历史数据
    # ----------------------------------------------------------------------
    preload_success = get_all_historical_data(trade_days_str)
    if not preload_success:
        st.error("❌ 历史数据预加载失败，回测无法进行。请检查 Tushare Token 和权限。")
        st.stop()
    st.success("✅ 历史数据预加载完成！QFQ 基准已固定。现在开始极速回测...")
    # ----------------------------------------------------------------------
    
    st.header(f"📈 正在进行 {BACKTEST_DAYS} 个交易日的回测 (V15.0 动量趋势增强版)...")
    
    results_list = []
    total_days = len(trade_days_str)
    
    progress_text = st.empty()
    my_bar = st.progress(0)
    
    for i, trade_date in enumerate(trade_days_str):
        progress_text.text(f"🚀 正在处理第 {i+1}/{total_days} 个交易日：{trade_date} (纯内存计算)")
        
        daily_result_df, error = run_backtest_for_a_day(
            trade_date, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS
        )
        
        if error:
            st.warning(f"{error}")
        elif not daily_result_df.empty:
            daily_result_df['Trade_Date'] = trade_date
            results_list.append(daily_result_df)
            
        my_bar.progress((i + 1) / total_days)

    progress_text.text("✅ 回测完成，正在汇总结果...")
    my_bar.empty()
    
    if not results_list:
        st.error("所有交易日的回测均失败或无结果。")
        st.stop()
        
    all_results = pd.concat(results_list)
    
    st.header(f"📊 最终平均回测结果 (Top {TOP_BACKTEST}，共 {len(all_results['Trade_Date'].unique())} 个有效交易日)")
    
    # 显示所有返回因子的统计信息
    st.subheader("📈 选股因子统计")
    factor_cols = ['momentum_20d', 'trend_score', 'volume_ratio', 'breakout_score', 'position_60d', 'volatility_20d']
    factor_stats = {}
    
    for col in factor_cols:
        if col in all_results.columns:
            factor_stats[col] = {
                '均值': all_results[col].mean(),
                '中位数': all_results[col].median(),
                '标准差': all_results[col].std()
            }
    
    if factor_stats:
        factor_df = pd.DataFrame(factor_stats).T
        st.dataframe(factor_df.round(2), use_container_width=True)
    
    # 显示收益统计
    for n in [1, 3, 5]:
        col = f'Return_D{n} (%)' 
        
        filtered_returns = all_results.copy()
        valid_returns = filtered_returns.dropna(subset=[col])

        if not valid_returns.empty:
            avg_return = valid_returns[col].mean()
            hit_rate = (valid_returns[col] > 0).sum() / len(valid_returns) * 100 if len(valid_returns) > 0 else 0.0
            total_count = len(valid_returns)
            median_return = valid_returns[col].median()
            std_return = valid_returns[col].std()
        else:
            avg_return = np.nan
            hit_rate = 0.0
            total_count = 0
            median_return = np.nan
            std_return = np.nan
            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"D+{n} 平均收益", f"{avg_return:.2f}%")
        with col2:
            st.metric(f"D+{n} 胜率", f"{hit_rate:.1f}%")
        with col3:
            st.metric(f"D+{n} 中位数收益", f"{median_return:.2f}%")
        with col4:
            st.metric(f"D+{n} 样本数", f"{total_count}")
        
        # 显示收益分布
        if not valid_returns.empty and len(valid_returns) > 5:
            st.caption(f"D+{n} 收益分布：最低 {valid_returns[col].min():.2f}%，最高 {valid_returns[col].max():.2f}%，标准差 {std_return:.2f}%")

    st.header("📋 每日回测详情 (Top K 明细)")
    
    # 显示详细的回测结果
    display_cols = ['Trade_Date', 'name', 'ts_code', '综合评分', 
                    'Close', 'Pct_Chg (%)', 'Circ_MV (亿)',
                    'momentum_20d', 'trend_score', 'volume_ratio',
                    'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)']
    
    # 只显示存在的列
    available_cols = [col for col in display_cols if col in all_results.columns]
    
    st.dataframe(all_results[available_cols].sort_values('Trade_Date', ascending=False), 
                 use_container_width=True,
                 column_config={
                     'momentum_20d': st.column_config.NumberColumn(format="%.1f"),
                     'trend_score': st.column_config.NumberColumn(format="%.1f"),
                     'volume_ratio': st.column_config.NumberColumn(format="%.2f"),
                     'Return_D1 (%)': st.column_config.NumberColumn(format="%.2f"),
                     'Return_D3 (%)': st.column_config.NumberColumn(format="%.2f"),
                     'Return_D5 (%)': st.column_config.NumberColumn(format="%.2f"),
                 })
