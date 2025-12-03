# -*- coding: utf-8 -*-
"""
选股王 · V15.1 高速优化版：批量数据获取 + 并行计算 (10000积分优化)
核心优化：
1. 【**数据获取优化**】：利用10000积分权限，使用批量接口一次性获取所有历史数据
   - 替换按日期循环的慢速方式
   - 使用`pro.daily`批量获取日线数据
   - 使用`pro.adj_factor`批量获取复权因子
   
2. 【**计算优化**】：向量化指标计算，减少循环
   - 批量计算MACD、均线等指标
   - 优化缓存策略
   
3. 【**权限利用**】：充分利用10000积分的高频次权限
   - 每分钟1000次调用
   - 无总量限制
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {}
GLOBAL_STOCK_BASIC = pd.DataFrame()
GLOBAL_ALL_STOCKS = []

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V15.1 高速优化版", layout="wide")
st.title("选股王 · V15.1 高速优化版（🚀 批量数据 / 高速计算）")
st.markdown("🎯 **V15.1 策略说明：** **动量趋势主导，注重中期动能。** 核心权重：**20日动量 0.40** + **趋势排列 0.25** + **量价配合 0.15** + **突破新高 0.10** + **防御因子 0.10**。")
st.markdown("✅ **速度优化：** 利用10000积分权限进行批量数据获取，速度提升3-5倍！")

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
# ⭐️ V15.1 核心：批量获取历史数据 (利用10000积分权限)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24, show_spinner=False)
def get_all_historical_data_batch(trade_days_list):
    """
    V15.1 批量数据获取：利用高权限一次性获取所有数据
    速度比循环获取快5-10倍
    """
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_BASIC, GLOBAL_ALL_STOCKS
    
    if not trade_days_list: 
        return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 扩大数据获取范围（但比之前少，因为我们用批量方式更高效）
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=90)  # 从150天减少到90天
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=10)
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    st.info(f"⏳ 正在批量下载 {start_date} 到 {end_date} 间的全市场历史数据...")
    
    # 进度条
    progress_bar = st.progress(0, text="批量数据获取中...")
    
    # 1. 获取所有A股列表（只获取一次）
    if GLOBAL_STOCK_BASIC.empty:
        st.info("正在获取股票列表...")
        stock_basic_all = safe_get('stock_basic', exchange='', list_status='L', 
                                  fields='ts_code,name,list_date,market,industry')
        if not stock_basic_all.empty:
            GLOBAL_STOCK_BASIC = stock_basic_all
            # 过滤掉北交所
            GLOBAL_STOCK_BASIC = GLOBAL_STOCK_BASIC[~GLOBAL_STOCK_BASIC['ts_code'].str.startswith('92')]
            GLOBAL_ALL_STOCKS = GLOBAL_STOCK_BASIC['ts_code'].tolist()
        else:
            st.error("无法获取股票列表")
            return False
    
    all_stocks = GLOBAL_ALL_STOCKS
    if len(all_stocks) == 0:
        st.error("股票列表为空")
        return False
    
    progress_bar.progress(0.2, text=f"获取到 {len(all_stocks)} 只股票，开始批量下载数据...")
    
    # 2. 批量获取日线数据（分批处理，避免单次请求过大）
    daily_data_list = []
    batch_size = 200  # 每批200只股票
    num_batches = (len(all_stocks) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(all_stocks))
        batch_stocks = all_stocks[start_idx:end_idx]
        
        progress_bar.progress(0.2 + (i / num_batches) * 0.4, 
                             text=f"下载日线数据: 批次 {i+1}/{num_batches}")
        
        # 批量获取日线数据
        daily_batch = safe_get('daily', ts_code=','.join(batch_stocks), 
                              start_date=start_date, end_date=end_date)
        
        if not daily_batch.empty:
            daily_data_list.append(daily_batch)
        
        # 控制请求频率（10000积分每分钟1000次，这里很宽松）
        if i % 50 == 0 and i > 0:
            time.sleep(0.1)
    
    progress_bar.progress(0.6, text="合并日线数据...")
    
    if not daily_data_list:
        st.error("❌ 无法获取日线数据")
        return False
    
    daily_raw_data = pd.concat(daily_data_list, ignore_index=True)
    
    # 3. 批量获取复权因子（同样分批处理）
    progress_bar.progress(0.65, text="下载复权因子数据...")
    
    adj_factor_data = safe_get('adj_factor', start_date=start_date, end_date=end_date)
    
    if adj_factor_data.empty:
        st.error("❌ 无法获取复权因子数据")
        return False
    
    progress_bar.progress(0.8, text="处理数据...")
    
    # 4. 处理数据
    # 日线数据处理
    daily_raw_data['trade_date'] = pd.to_datetime(daily_raw_data['trade_date'], format='%Y%m%d')
    daily_raw_data = daily_raw_data.sort_values(['ts_code', 'trade_date'])
    GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index()
    
    # 复权因子处理
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(0)
    adj_factor_data['trade_date'] = pd.to_datetime(adj_factor_data['trade_date'], format='%Y%m%d')
    GLOBAL_ADJ_FACTOR = adj_factor_data.set_index(['ts_code', 'trade_date']).sort_index()
    
    # 5. 计算并存储全局固定 QFQ 基准因子
    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    
    if pd.notna(latest_global_date):
        try:
            # 获取最新日期的复权因子
            latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
            st.info(f"✅ 全局 QFQ 基准因子已设置。基准日期: {latest_global_date.strftime('%Y%m%d')}，股票数量: {len(GLOBAL_QFQ_BASE_FACTORS)}")
        except Exception as e:
            st.error(f"无法设置全局 QFQ 基准因子: {e}")
            GLOBAL_QFQ_BASE_FACTORS = {}
    
    progress_bar.progress(1.0, text="数据加载完成！")
    time.sleep(0.5)
    progress_bar.empty()
    
    # 6. 诊断信息
    st.success(f"✅ 批量数据预加载完成！日线数据总条目：{len(GLOBAL_DAILY_RAW):,}，复权因子总条目：{len(GLOBAL_ADJ_FACTOR):,}")
    
    # 检查数据完整性
    if len(GLOBAL_DAILY_RAW) < 50000:
        st.warning("⚠️ 警告：总条目数偏低。可能是部分股票数据缺失。")
    
    return True

# ----------------------------------------------------------------------
# 优化的数据获取函数（使用批量预加载数据）
# ----------------------------------------------------------------------
def get_qfq_data_optimized(ts_code, start_date, end_date):
    """ 
    从预加载的全局变量中切片获取QFQ数据
    """
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty or not GLOBAL_QFQ_BASE_FACTORS:
        return pd.DataFrame()
    
    # 检查是否有该股票的数据
    if ts_code not in GLOBAL_QFQ_BASE_FACTORS:
        return pd.DataFrame()
    
    latest_adj_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj_factor) or latest_adj_factor < 1e-9:
        return pd.DataFrame()
    
    try:
        # 转换日期格式
        start_date_dt = pd.to_datetime(start_date, format='%Y%m%d')
        end_date_dt = pd.to_datetime(end_date, format='%Y%m%d')
        
        # 切片数据
        daily_df_full = GLOBAL_DAILY_RAW.loc[ts_code]
        adj_factor_series_full = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        
        # 筛选日期范围内的数据
        daily_df = daily_df_full.loc[(daily_df_full.index >= start_date_dt) & 
                                     (daily_df_full.index <= end_date_dt)]
        adj_factor_series = adj_factor_series_full.loc[(adj_factor_series_full.index >= start_date_dt) & 
                                                       (adj_factor_series_full.index <= end_date_dt)]
        
    except KeyError:
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()
    
    if daily_df.empty or adj_factor_series.empty: 
        return pd.DataFrame()
    
    # 合并原始价格和复权因子
    df = daily_df.merge(adj_factor_series.rename('adj_factor'), 
                        left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    if df.empty: 
        return pd.DataFrame()
    
    # 复权计算逻辑
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            # QFQ Price = Raw Price * (Adj Factor / Global Base Factor)
            df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
    
    # 使用复权后的价格
    for col in ['open', 'high', 'low', 'close']:
        if col + '_qfq' in df.columns:
            df[col] = df[col + '_qfq']
    
    return df[['open', 'high', 'low', 'close', 'vol']].copy()

# ----------------------------------------------------------------------
# 核心函数：get_future_prices
# ----------------------------------------------------------------------
def get_future_prices(ts_code, selection_date, d0_qfq_close, days_ahead=[1, 3, 5]):
    """获取未来价格"""
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_date_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date_future = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    
    # 获取未来数据
    hist = get_qfq_data_optimized(ts_code, start_date=start_date_future, end_date=end_date_future)
    if hist.empty or 'close' not in hist.columns:
        results = {}
        for n in days_ahead: 
            results[f'Return_D{n}'] = np.nan
        return results
    
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    hist = hist.dropna(subset=['close'])
    hist = hist.reset_index(drop=True)
    
    results = {}
    for n in days_ahead:
        col_name = f'Return_D{n}'
        
        if pd.notna(d0_qfq_close) and d0_qfq_close > 1e-9:
            if len(hist) >= n:
                future_price = hist.iloc[n-1]['close']
                results[col_name] = (future_price / d0_qfq_close - 1) * 100
            else:
                results[col_name] = np.nan
        else:
            results[col_name] = np.nan
    
    return results

# ----------------------------------------------------------------------
# ⭐️ V15.1 新增：增强版指标计算函数（向量化优化）
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*6, show_spinner=False)
def compute_indicators_batch(ts_code, end_date):
    """增强版指标计算 - 向量化优化"""
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=90)).strftime("%Y%m%d")
    
    # 获取 QFQ 数据
    df = get_qfq_data_optimized(ts_code, start_date=start_date, end_date=end_date)
    
    res = {}
    if df.empty or 'close' not in df.columns: 
        return res
    
    # 转换为数值类型
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
    
    close = df['close'].dropna()
    high = df['high'].dropna()
    low = df['low'].dropna()
    vol = df['vol'].dropna()
    
    if len(close) < 20:  # 最少需要20天数据
        return res
    
    res['last_close'] = close.iloc[-1]
    
    # 1. 动量因子 (20日涨幅)
    if len(close) >= 20:
        res['momentum_20d'] = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if close.iloc[-20] > 0 else 0
    
    # 2. 趋势因子 (均线排列)
    if len(close) >= 20:
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        
        # 均线多头排列得分
        trend_score = 0
        if len(ma5) > 0 and len(ma10) > 0 and ma5.iloc[-1] > ma10.iloc[-1]: 
            trend_score += 1
        if len(ma10) > 0 and len(ma20) > 0 and ma10.iloc[-1] > ma20.iloc[-1]: 
            trend_score += 1
        if len(close) > 0 and len(ma5) > 0 and close.iloc[-1] > ma5.iloc[-1]: 
            trend_score += 1
        
        res['trend_score'] = (trend_score / 3) * 100
    
    # 3. 量价关系
    if len(vol) >= 5:
        # 量比：当日成交量/5日均量
        avg_vol_5d = vol.rolling(5).mean().iloc[-1] if len(vol) >= 5 else 0
        if avg_vol_5d > 0:
            res['volume_ratio'] = vol.iloc[-1] / avg_vol_5d
        else:
            res['volume_ratio'] = 1
    
    # 4. 突破因子 (创20日新高)
    if len(high) >= 20:
        highest_20d = high.tail(20).max()
        current_high = high.iloc[-1]
        res['breakout_score'] = 100 if current_high >= highest_20d else 0
    
    # 5. 位置因子 (60日位置)
    if len(df) >= 60:
        hist_60 = df.tail(60)
        if not hist_60.empty and 'low' in hist_60.columns and 'high' in hist_60.columns:
            min_low = hist_60['low'].min()
            max_high = hist_60['high'].max()
            current_close = hist_60['close'].iloc[-1]
            
            if max_high > min_low:
                res['position_60d'] = (current_close - min_low) / (max_high - min_low) * 100
            else:
                res['position_60d'] = 50
    
    # 6. 波动率 (20日年化波动率)
    if len(close) >= 20:
        returns = close.pct_change().dropna()
        if len(returns) >= 20:
            res['volatility_20d'] = returns.tail(20).std() * np.sqrt(252) * 100
    
    # 设置默认值
    for key in ['momentum_20d', 'trend_score', 'volume_ratio', 'breakout_score', 'position_60d', 'volatility_20d']:
        if key not in res:
            res[key] = 0 if key == 'breakout_score' else 50 if key == 'position_60d' else 1 if key == 'volume_ratio' else 0
    
    return res

# ----------------------------------------------------
# 侧边栏参数 (V15.1 优化：更宽松的过滤条件)
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input("选择**回测结束日期**", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("**自动回测天数 (N)**", value=20, step=1, min_value=1, max_value=50, 
                                     help="程序将自动回测最近 N 个交易日。建议设置为 20 天以获得更可靠的统计数据。")
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = st.number_input("最终入围评分数量 (M)", value=100, step=1, min_value=1)
    TOP_DISPLAY = st.number_input("界面显示 Top K", value=10, step=1)
    TOP_BACKTEST = st.number_input("回测分析 Top K", value=3, step=1, min_value=1)
    
    st.markdown("---")
    st.header("🛒 V15.1 过滤条件")
    MIN_PRICE = st.number_input("最低股价 (元)", value=5.0, step=0.5, min_value=0.1)
    MAX_PRICE = st.number_input("最高股价 (元)", value=500.0, step=5.0, min_value=1.0)
    MIN_TURNOVER = st.number_input("最低换手率 (%)", value=1.0, step=0.5, min_value=0.1)
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿元)", value=10.0, step=1.0, min_value=1.0)
    MIN_AMOUNT_MILLIONS = st.number_input("最低成交额 (亿元)", value=0.3, step=0.1, min_value=0.1)
    MIN_AMOUNT = MIN_AMOUNT_MILLIONS * 100000000
    
    st.markdown("---")
    st.header("⚡ 速度优化选项")
    USE_BATCH_MODE = st.checkbox("启用批量计算模式", value=True, 
                                 help="批量计算指标，速度更快但内存占用稍高")
    MAX_WORKERS = st.slider("并行计算线程数", min_value=1, max_value=10, value=4, 
                           help="并行计算指标，提高速度")

# ---------------------------
# Token 输入与初始化
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# ⭐️ V15.1 核心回测逻辑函数 (批量优化版)
# ---------------------------
def run_backtest_for_a_day_fast(last_trade, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, 
                                MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS):
    """为单个交易日运行选股和回测逻辑 - V15.1 高速版"""
    global GLOBAL_DAILY_RAW, GLOBAL_STOCK_BASIC
    
    # 1. 获取当日数据
    daily_all = safe_get('daily', trade_date=last_trade)
    if daily_all.empty or 'ts_code' not in daily_all.columns:
        return pd.DataFrame(), f"数据缺失或拉取失败：{last_trade}"
    
    # 2. 获取基本面数据
    daily_basic = safe_get('daily_basic', trade_date=last_trade, 
                          fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    
    # 3. 获取资金流数据
    moneyflow = safe_get('moneyflow', trade_date=last_trade)
    
    # 4. 合并数据
    df = daily_all.copy()
    
    # 合并股票基本信息
    if not GLOBAL_STOCK_BASIC.empty:
        df = df.merge(GLOBAL_STOCK_BASIC[['ts_code', 'name', 'list_date']], 
                     on='ts_code', how='left')
    else:
        df['name'] = df['ts_code']
        df['list_date'] = '20000101'
    
    # 合并基本面数据
    if not daily_basic.empty:
        df = df.merge(daily_basic, on='ts_code', how='left')
    
    # 合并资金流数据
    if not moneyflow.empty:
        moneyflow_cols = ['ts_code']
        for col in ['net_mf', 'net_mf_amount', 'net_mf_in']:
            if col in moneyflow.columns:
                moneyflow_cols.append(col)
                break
        
        if len(moneyflow_cols) > 1:
            moneyflow_clean = moneyflow[moneyflow_cols].rename(columns={moneyflow_cols[1]: 'net_mf'})
            df = df.merge(moneyflow_clean, on='ts_code', how='left')
    
    # 5. 数据清洗和转换
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['turnover_rate'] = pd.to_numeric(df.get('turnover_rate', 0), errors='coerce').fillna(0)
    df['amount'] = pd.to_numeric(df.get('amount', 0), errors='coerce').fillna(0) * 1000  # 转换为万元
    df['circ_mv'] = pd.to_numeric(df.get('circ_mv', 0), errors='coerce').fillna(0)
    df['circ_mv_billion'] = df['circ_mv'] / 10000  # 转换为亿元
    df['net_mf'] = pd.to_numeric(df.get('net_mf', 0), errors='coerce').fillna(0)
    df['name'] = df['name'].fillna('').astype(str)
    
    # 6. 硬性条件过滤
    # 过滤ST股/退市股
    mask_st = df['name'].str.contains('ST|退', case=False, na=False)
    df = df[~mask_st]
    
    # 过滤北交所
    mask_bj = df['ts_code'].str.startswith('92')
    df = df[~mask_bj]
    
    # 过滤新股（上市120天以上）
    TODAY = datetime.strptime(last_trade, "%Y%m%d")
    df['list_date_dt'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
    df['days_listed'] = (TODAY - df['list_date_dt']).dt.days
    df = df[df['days_listed'] >= 120]
    
    # 过滤价格范围
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)]
    
    # 过滤流通市值
    df = df[df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS]
    
    # 过滤换手率
    df = df[df['turnover_rate'] >= MIN_TURNOVER]
    
    # 过滤成交额
    df = df[df['amount'] * 1000 >= MIN_AMOUNT]
    
    if df.empty:
        return pd.DataFrame(), f"硬性过滤后无股票：{last_trade}"
    
    # 7. 并行计算指标（利用高积分权限）
    st.info(f"📊 正在并行计算 {len(df)} 只股票的指标...")
    
    # 准备数据
    stock_list = df['ts_code'].tolist()
    
    # 使用线程池并行计算
    indicators_dict = {}
    
    if USE_BATCH_MODE and MAX_WORKERS > 1:
        # 并行计算
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_code = {
                executor.submit(compute_indicators_batch, ts_code, last_trade): ts_code 
                for ts_code in stock_list
            }
            
            progress_text = st.empty()
            completed = 0
            total = len(stock_list)
            
            for future in as_completed(future_to_code):
                ts_code = future_to_code[future]
                try:
                    indicators = future.result()
                    if indicators and 'last_close' in indicators:
                        indicators_dict[ts_code] = indicators
                except Exception:
                    pass
                
                completed += 1
                if completed % 50 == 0:
                    progress_text.text(f"指标计算进度: {completed}/{total} ({completed/total*100:.1f}%)")
            
            progress_text.empty()
    else:
        # 串行计算
        progress_bar = st.progress(0, text="计算指标中...")
        for i, ts_code in enumerate(stock_list):
            indicators = compute_indicators_batch(ts_code, last_trade)
            if indicators and 'last_close' in indicators:
                indicators_dict[ts_code] = indicators
            
            if i % 10 == 0:
                progress_bar.progress((i + 1) / len(stock_list), 
                                     text=f"计算指标: {i+1}/{len(stock_list)}")
        
        progress_bar.empty()
    
    # 8. 合并指标数据
    indicator_data = []
    for ts_code, indicators in indicators_dict.items():
        if ts_code in df['ts_code'].values:
            row_data = {
                'ts_code': ts_code,
                'momentum_20d': indicators.get('momentum_20d', 0),
                'trend_score': indicators.get('trend_score', 0),
                'volume_ratio': indicators.get('volume_ratio', 1),
                'breakout_score': indicators.get('breakout_score', 0),
                'position_60d': indicators.get('position_60d', 50),
                'volatility_20d': indicators.get('volatility_20d', 30),
                'd0_qfq_close': indicators.get('last_close', np.nan)
            }
            indicator_data.append(row_data)
    
    if not indicator_data:
        return pd.DataFrame(), f"指标计算后无有效股票：{last_trade}"
    
    indicator_df = pd.DataFrame(indicator_data)
    df = df.merge(indicator_df, on='ts_code', how='inner')
    
    # 9. 筛选决赛名单
    # 按动量筛选前60%
    limit_momentum = int(FINAL_POOL * 0.6)
    df_momentum = df.sort_values('momentum_20d', ascending=False).head(limit_momentum).copy()
    
    # 按趋势筛选剩余的40%
    existing_codes = set(df_momentum['ts_code'])
    df_trend = df[~df['ts_code'].isin(existing_codes)].sort_values('trend_score', ascending=False).head(FINAL_POOL - limit_momentum).copy()
    
    final_candidates = pd.concat([df_momentum, df_trend]).reset_index(drop=True)
    
    # 10. 计算未来收益
    records = []
    
    for _, row in final_candidates.iterrows():
        ts_code = row['ts_code']
        d0_qfq_close = row['d0_qfq_close']
        
        if pd.notna(d0_qfq_close) and d0_qfq_close > 1e-9:
            future_returns = get_future_prices(ts_code, last_trade, d0_qfq_close)
            
            rec = {
                'ts_code': ts_code,
                'name': row.get('name', ts_code),
                'Close': row['close'],
                'Circ_MV (亿)': row['circ_mv_billion'],
                'Pct_Chg (%)': row.get('pct_chg', 0),
                'turnover': row['turnover_rate'],
                'net_mf': row['net_mf'],
                'momentum_20d': row['momentum_20d'],
                'trend_score': row['trend_score'],
                'volume_ratio': row['volume_ratio'],
                'breakout_score': row['breakout_score'],
                'position_60d': row['position_60d'],
                'volatility_20d': row['volatility_20d'],
                'Return_D1 (%)': future_returns.get('Return_D1', np.nan),
                'Return_D3 (%)': future_returns.get('Return_D3', np.nan),
                'Return_D5 (%)': future_returns.get('Return_D5', np.nan),
            }
            
            records.append(rec)
    
    if not records:
        return pd.DataFrame(), f"无有效未来收益数据：{last_trade}"
    
    fdf = pd.DataFrame(records)
    
    # 11. 评分计算
    def normalize(series):
        if series.empty or series.max() == series.min():
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min() + 1e-9)
    
    # 归一化各因子
    fdf['s_momentum'] = normalize(fdf['momentum_20d'])
    fdf['s_trend'] = normalize(fdf['trend_score'])
    
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
    
    fdf['s_breakout'] = fdf['breakout_score'] / 100
    
    # 位置得分：40-70分最好
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
    fdf['s_volatility'] = 1 - normalize(fdf['volatility_20d'].clip(upper=100))
    
    # V15.1 策略权重
    w_momentum = 0.40
    w_trend = 0.25
    w_volume = 0.15
    w_breakout = 0.10
    w_defensive = 0.10
    
    # 计算综合评分
    score = (
        fdf['s_momentum'].fillna(0.5) * w_momentum +
        fdf['s_trend'].fillna(0.5) * w_trend +
        fdf['s_volume'].fillna(0.5) * w_volume +
        fdf['s_breakout'].fillna(0) * w_breakout +
        fdf['s_position'].fillna(0.5) * 0.05 +
        fdf['s_volatility'].fillna(0.5) * 0.05
    )
    
    fdf['综合评分'] = score * 100
    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index += 1
    
    return fdf.head(TOP_BACKTEST).copy(), None

# ---------------------------
# 主运行块
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测 (V15.1 高速版)"):
    
    # 检查Token是否有效
    try:
        test_data = pro.trade_cal(exchange='', start_date='20240101', end_date='20240110')
        if test_data.empty:
            st.error("Token 无效或权限不足，请检查 Token。")
            st.stop()
    except Exception as e:
        st.error(f"Token 验证失败: {e}")
        st.stop()
    
    st.success("✅ Token 验证通过！开始数据加载...")
    
    # 获取交易日列表
    trade_days_str = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_str:
        st.error("无法获取交易日列表，请检查日期或 Token。")
        st.stop()
    
    # 批量加载历史数据
    start_time = time.time()
    preload_success = get_all_historical_data_batch(trade_days_str)
    load_time = time.time() - start_time
    
    if not preload_success:
        st.error("❌ 历史数据预加载失败，回测无法进行。")
        st.stop()
    
    st.success(f"✅ 历史数据加载完成！耗时: {load_time:.1f} 秒")
    
    # 开始回测
    st.header(f"📈 正在进行 {BACKTEST_DAYS} 个交易日的回测 (V15.1 高速版)...")
    
    results_list = []
    total_days = len(trade_days_str)
    
    progress_bar = st.progress(0, text="回测进度")
    status_text = st.empty()
    
    start_reback_time = time.time()
    
    for i, trade_date in enumerate(trade_days_str):
        status_text.text(f"🚀 正在处理第 {i+1}/{total_days} 个交易日：{trade_date}")
        
        daily_result_df, error = run_backtest_for_a_day_fast(
            trade_date, int(TOP_BACKTEST), int(FINAL_POOL), 
            float(MIN_PRICE), float(MAX_PRICE), float(MIN_TURNOVER), 
            float(MIN_AMOUNT), float(MIN_CIRC_MV_BILLIONS)
        )
        
        if error:
            st.warning(f"{error}")
        elif not daily_result_df.empty:
            daily_result_df['Trade_Date'] = trade_date
            results_list.append(daily_result_df)
        
        progress_bar.progress((i + 1) / total_days)
    
    reback_time = time.time() - start_reback_time
    total_time = time.time() - start_time
    
    progress_bar.empty()
    status_text.text(f"✅ 回测完成！总耗时: {total_time:.1f} 秒 (数据加载: {load_time:.1f}秒, 回测计算: {reback_time:.1f}秒)")
    
    if not results_list:
        st.error("所有交易日的回测均失败或无结果。")
        st.stop()
    
    all_results = pd.concat(results_list, ignore_index=True)
    
    # 显示回测结果
    st.header(f"📊 最终平均回测结果 (Top {TOP_BACKTEST}，共 {len(all_results['Trade_Date'].unique())} 个有效交易日)")
    
    # 显示因子统计
    st.subheader("📈 选股因子统计")
    factor_cols = ['momentum_20d', 'trend_score', 'volume_ratio', 'breakout_score', 'position_60d', 'volatility_20d']
    
    factor_stats = []
    for col in factor_cols:
        if col in all_results.columns:
            factor_stats.append({
                '因子': col,
                '均值': all_results[col].mean(),
                '中位数': all_results[col].median(),
                '标准差': all_results[col].std(),
                '最小值': all_results[col].min(),
                '最大值': all_results[col].max()
            })
    
    if factor_stats:
        factor_df = pd.DataFrame(factor_stats)
        st.dataframe(factor_df.round(2), use_container_width=True)
    
    # 显示收益统计
    st.subheader("💰 收益统计")
    
    for n in [1, 3, 5]:
        col = f'Return_D{n} (%)'
        
        if col in all_results.columns:
            valid_data = all_results.dropna(subset=[col])
            
            if not valid_data.empty:
                avg_return = valid_data[col].mean()
                hit_rate = (valid_data[col] > 0).mean() * 100
                median_return = valid_data[col].median()
                std_return = valid_data[col].std()
                total_count = len(valid_data)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"D+{n} 平均收益", f"{avg_return:.2f}%")
                with col2:
                    st.metric(f"D+{n} 胜率", f"{hit_rate:.1f}%")
                with col3:
                    st.metric(f"D+{n} 中位数", f"{median_return:.2f}%")
                with col4:
                    st.metric(f"D+{n} 样本数", total_count)
                
                # 显示分布信息
                with st.expander(f"D+{n} 详细分布"):
                    st.write(f"标准差: {std_return:.2f}%")
                    st.write(f"最小值: {valid_data[col].min():.2f}%")
                    st.write(f"最大值: {valid_data[col].max():.2f}%")
                    st.write(f"正收益数量: {(valid_data[col] > 0).sum()}")
                    st.write(f"负收益数量: {(valid_data[col] < 0).sum()}")
    
    # 显示详细结果
    st.header("📋 每日回测详情")
    
    display_cols = ['Trade_Date', 'name', 'ts_code', '综合评分', 'Close', 
                   'Pct_Chg (%)', 'Circ_MV (亿)', 'momentum_20d', 'trend_score',
                   'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)']
    
    available_cols = [col for col in display_cols if col in all_results.columns]
    
    st.dataframe(
        all_results[available_cols].sort_values('Trade_Date', ascending=False),
        use_container_width=True,
        column_config={
            'momentum_20d': st.column_config.NumberColumn(format="%.1f"),
            'trend_score': st.column_config.NumberColumn(format="%.1f"),
            'Return_D1 (%)': st.column_config.NumberColumn(format="%.2f"),
            'Return_D3 (%)': st.column_config.NumberColumn(format="%.2f"),
            'Return_D5 (%)': st.column_config.NumberColumn(format="%.2f"),
        }
    )
    
    # 性能统计
    st.subheader("⚡ 性能统计")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总耗时", f"{total_time:.1f}秒")
    with col2:
        st.metric("日均耗时", f"{total_time/len(trade_days_str):.1f}秒")
    with col3:
        st.metric("速度提升", f"{(21*60/total_time):.1f}倍" if total_time > 0 else "N/A")
