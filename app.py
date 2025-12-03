# -*- coding: utf-8 -*-
"""
选股王 · V15.2 高效数据版：修复数据获取问题 + 优化计算逻辑
核心修复：
1. 【**数据获取修复**】：修正批量数据获取逻辑，确保获取完整数据
   - 修复日线数据获取：使用正确的参数格式
   - 修复复权因子获取：确保覆盖所有日期
   
2. 【**性能优化**】：减少不必要的计算和API调用
   - 优化指标计算逻辑
   - 减少重复数据获取
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

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V15.2 高效数据版", layout="wide")
st.title("选股王 · V15.2 高效数据版（🚀 数据修复 / 性能优化）")

# ---------------------------
# 辅助函数
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
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历，请检查 Token 或 Tushare 权限。")
        return []
    
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()

# ----------------------------------------------------------------------
# ⭐️ V15.2 修复：高效数据获取
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24, show_spinner=False)
def get_all_historical_data_fixed(trade_days_list):
    """
    修复版数据获取：确保获取完整数据
    """
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_BASIC
    
    if not trade_days_list: 
        return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 计算需要的日期范围（回测日期前120天到后20天）
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=120)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=20)
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    st.info(f"⏳ 正在下载 {start_date} 到 {end_date} 的历史数据...")
    
    # 1. 获取股票列表
    if GLOBAL_STOCK_BASIC.empty:
        progress_bar = st.progress(0, text="获取股票列表...")
        stock_basic = safe_get('stock_basic', exchange='', list_status='L', 
                              fields='ts_code,name,list_date,market')
        if stock_basic.empty:
            st.error("无法获取股票列表")
            return False
        
        # 过滤掉北交所
        stock_basic = stock_basic[~stock_basic['ts_code'].str.startswith('92')]
        GLOBAL_STOCK_BASIC = stock_basic
        progress_bar.progress(0.2, text=f"获取到 {len(stock_basic)} 只股票")
    
    all_stocks = GLOBAL_STOCK_BASIC['ts_code'].tolist()
    
    # 2. 获取交易日历
    progress_bar.progress(0.3, text="获取交易日历...")
    trade_cal = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if trade_cal.empty:
        st.error("无法获取交易日历")
        return False
    
    all_trade_dates = trade_cal['cal_date'].tolist()
    
    # 3. 批量获取复权因子（使用日期范围，这是最高效的方式）
    progress_bar.progress(0.4, text="下载复权因子...")
    adj_factor_data = safe_get('adj_factor', start_date=start_date, end_date=end_date)
    
    if adj_factor_data.empty:
        st.warning("复权因子数据为空，尝试按股票获取...")
        # 如果批量获取失败，尝试按股票获取
        adj_factor_list = []
        batch_size = 100
        num_batches = (len(all_stocks) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(all_stocks))
            batch_stocks = all_stocks[start_idx:end_idx]
            
            for stock in batch_stocks:
                adj_df = safe_get('adj_factor', ts_code=stock, start_date=start_date, end_date=end_date)
                if not adj_df.empty:
                    adj_factor_list.append(adj_df)
            
            progress_bar.progress(0.4 + (i / num_batches) * 0.2, 
                                 text=f"获取复权因子: 批次 {i+1}/{num_batches}")
        
        if adj_factor_list:
            adj_factor_data = pd.concat(adj_factor_list, ignore_index=True)
        else:
            st.error("无法获取复权因子数据")
            return False
    
    # 处理复权因子数据
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(1.0)
    GLOBAL_ADJ_FACTOR = adj_factor_data.set_index(['ts_code', 'trade_date']).sort_index()
    
    # 4. 获取日线数据（分批获取，避免超时）
    progress_bar.progress(0.7, text="下载日线数据...")
    
    # 方法1：按日期批量获取（更高效）
    daily_data_list = []
    
    # 分批处理日期，避免单次请求太大
    date_batch_size = 20
    num_date_batches = (len(all_trade_dates) + date_batch_size - 1) // date_batch_size
    
    for i in range(num_date_batches):
        start_idx = i * date_batch_size
        end_idx = min((i + 1) * date_batch_size, len(all_trade_dates))
        date_batch = all_trade_dates[start_idx:end_idx]
        
        for date in date_batch:
            daily_df = safe_get('daily', trade_date=date)
            if not daily_df.empty:
                daily_data_list.append(daily_df)
        
        progress_bar.progress(0.7 + (i / num_date_batches) * 0.25, 
                             text=f"下载日线数据: {i+1}/{num_date_batches}")
    
    if not daily_data_list:
        st.error("无法获取日线数据")
        return False
    
    daily_raw_data = pd.concat(daily_data_list, ignore_index=True)
    
    # 5. 处理日线数据
    progress_bar.progress(0.95, text="处理数据...")
    daily_raw_data['trade_date'] = pd.to_datetime(daily_raw_data['trade_date'], format='%Y%m%d')
    GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index()
    
    # 6. 设置QFQ基准因子
    try:
        latest_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
        if pd.notna(latest_date):
            latest_adj = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj.droplevel(1).to_dict()
            st.success(f"✅ 设置基准因子，股票数: {len(GLOBAL_QFQ_BASE_FACTORS)}")
    except Exception as e:
        st.warning(f"设置基准因子失败: {e}")
        GLOBAL_QFQ_BASE_FACTORS = {}
    
    progress_bar.progress(1.0, text="数据加载完成！")
    time.sleep(0.5)
    progress_bar.empty()
    
    # 显示数据统计
    st.success(f"""
    ✅ 数据加载完成！
    - 日线数据: {len(GLOBAL_DAILY_RAW):,} 条记录
    - 复权因子: {len(GLOBAL_ADJ_FACTOR):,} 条记录
    - 基准因子: {len(GLOBAL_QFQ_BASE_FACTORS)} 只股票
    - 时间范围: {start_date} 到 {end_date}
    """)
    
    return True

# ----------------------------------------------------------------------
# 数据获取函数
# ----------------------------------------------------------------------
def get_qfq_data(ts_code, start_date, end_date):
    """获取前复权数据"""
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty:
        return pd.DataFrame()
    
    # 获取基准复权因子
    base_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, 1.0)
    if base_factor <= 0:
        return pd.DataFrame()
    
    try:
        # 转换日期格式
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        # 获取日线数据
        if ts_code not in GLOBAL_DAILY_RAW.index.get_level_values('ts_code'):
            return pd.DataFrame()
        
        daily_data = GLOBAL_DAILY_RAW.loc[ts_code].copy()
        daily_data = daily_data[(daily_data.index >= start_dt) & (daily_data.index <= end_dt)]
        
        if daily_data.empty:
            return pd.DataFrame()
        
        # 获取复权因子
        if ts_code not in GLOBAL_ADJ_FACTOR.index.get_level_values('ts_code'):
            return pd.DataFrame()
        
        adj_data = GLOBAL_ADJ_FACTOR.loc[ts_code].copy()
        adj_data = adj_data[(adj_data.index >= start_dt) & (adj_data.index <= end_dt)]
        
        if adj_data.empty:
            return pd.DataFrame()
        
        # 合并数据
        df = daily_data.merge(adj_data, left_index=True, right_index=True, how='left')
        df['adj_factor'] = df['adj_factor'].fillna(base_factor)
        
        # 计算前复权价格
        for col in ['open', 'high', 'low', 'close', 'pre_close']:
            if col in df.columns:
                df[f'{col}_qfq'] = df[col] * df['adj_factor'] / base_factor
        
        # 使用复权价格
        for col in ['open', 'high', 'low', 'close']:
            if f'{col}_qfq' in df.columns:
                df[col] = df[f'{col}_qfq']
        
        return df[['open', 'high', 'low', 'close', 'vol']].reset_index()
        
    except Exception as e:
        return pd.DataFrame()

# ----------------------------------------------------------------------
# 核心计算函数
# ----------------------------------------------------------------------
def compute_indicators_simple(ts_code, end_date):
    """简化的指标计算"""
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
    
    df = get_qfq_data(ts_code, start_date, end_date)
    if df.empty or 'close' not in df.columns:
        return {}
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    
    close = df['close'].dropna()
    if len(close) < 20:
        return {}
    
    res = {'last_close': close.iloc[-1]}
    
    # 1. 动量因子
    if len(close) >= 20:
        res['momentum_20d'] = (close.iloc[-1] / close.iloc[-20] - 1) * 100
    
    # 2. 趋势因子
    if len(close) >= 20:
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        
        trend_score = 0
        if len(ma5) > 0 and len(ma10) > 0 and ma5.iloc[-1] > ma10.iloc[-1]:
            trend_score += 1
        if len(ma10) > 0 and len(ma20) > 0 and ma10.iloc[-1] > ma20.iloc[-1]:
            trend_score += 1
        if len(close) > 0 and len(ma5) > 0 and close.iloc[-1] > ma5.iloc[-1]:
            trend_score += 1
        
        res['trend_score'] = (trend_score / 3) * 100
    
    # 3. 位置因子
    if len(df) >= 60:
        hist_60 = df.tail(60)
        min_low = hist_60['low'].min()
        max_high = hist_60['high'].max()
        current_close = hist_60['close'].iloc[-1]
        
        if max_high > min_low:
            res['position_60d'] = (current_close - min_low) / (max_high - min_low) * 100
        else:
            res['position_60d'] = 50
    
    # 设置默认值
    res.setdefault('momentum_20d', 0)
    res.setdefault('trend_score', 0)
    res.setdefault('position_60d', 50)
    
    return res

def get_future_returns(ts_code, selection_date, selection_price):
    """获取未来收益"""
    if pd.isna(selection_price) or selection_price <= 0:
        return {f'Return_D{n} (%)': np.nan for n in [1, 3, 5]}
    
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_date = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    
    df = get_qfq_data(ts_code, start_date, end_date)
    if df.empty or 'close' not in df.columns:
        return {f'Return_D{n} (%)': np.nan for n in [1, 3, 5]}
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    
    results = {}
    for n in [1, 3, 5]:
        if len(df) >= n:
            future_price = df.iloc[n-1]['close']
            results[f'Return_D{n} (%)'] = (future_price / selection_price - 1) * 100
        else:
            results[f'Return_D{n} (%)'] = np.nan
    
    return results

# ----------------------------------------------------
# 侧边栏参数
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input("选择**回测结束日期**", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("**自动回测天数 (N)**", value=10, step=1, min_value=1, max_value=30)
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = st.number_input("最终入围评分数量 (M)", value=50, step=10, min_value=10)
    TOP_BACKTEST = st.number_input("回测分析 Top K", value=3, step=1, min_value=1)
    
    st.markdown("---")
    st.header("🛒 过滤条件")
    MIN_PRICE = st.number_input("最低股价 (元)", value=5.0, step=1.0, min_value=1.0)
    MAX_PRICE = st.number_input("最高股价 (元)", value=200.0, step=10.0, min_value=10.0)
    MIN_TURNOVER = st.number_input("最低换手率 (%)", value=1.0, step=0.5, min_value=0.1)
    MIN_CIRC_MV = st.number_input("最低流通市值 (亿元)", value=20.0, step=5.0, min_value=5.0)

# ---------------------------
# Token 输入
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 验证Token
try:
    test = pro.trade_cal(exchange='', start_date='20240101', end_date='20240105')
    if test.empty:
        st.error("Token无效")
        st.stop()
except Exception as e:
    st.error(f"Token验证失败: {e}")
    st.stop()

# ---------------------------
# 回测函数
# ---------------------------
def run_backtest_single_day(trade_date):
    """单个交易日的回测"""
    # 获取当日数据
    daily_data = safe_get('daily', trade_date=trade_date)
    if daily_data.empty:
        return pd.DataFrame(), f"无日线数据: {trade_date}"
    
    daily_basic = safe_get('daily_basic', trade_date=trade_date, 
                          fields='ts_code,turnover_rate,circ_mv')
    
    # 合并数据
    df = daily_data.copy()
    if not daily_basic.empty:
        df = df.merge(daily_basic, on='ts_code', how='left')
    
    # 过滤ST股和北交所
    df = df[~df['ts_code'].str.startswith(('68', '200', '300', '400', '900', '92'))]
    
    # 转换为数值
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
    df['circ_mv'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000  # 转为亿元
    
    # 硬性过滤
    df = df[
        (df['close'] >= MIN_PRICE) & 
        (df['close'] <= MAX_PRICE) &
        (df['turnover_rate'] >= MIN_TURNOVER) &
        (df['circ_mv'] >= MIN_CIRC_MV)
    ].copy()
    
    if df.empty:
        return pd.DataFrame(), f"过滤后无股票: {trade_date}"
    
    # 计算指标
    records = []
    for _, row in df.iterrows():
        ts_code = row['ts_code']
        
        # 获取指标
        indicators = compute_indicators_simple(ts_code, trade_date)
        if not indicators or 'last_close' not in indicators:
            continue
        
        d0_price = indicators['last_close']
        
        # 获取未来收益
        future_returns = get_future_returns(ts_code, trade_date, d0_price)
        
        record = {
            'ts_code': ts_code,
            'name': row.get('name', ts_code),
            'Close': row['close'],
            'Circ_MV (亿)': row['circ_mv'],
            'Pct_Chg (%)': row.get('pct_chg', 0),
            'turnover': row['turnover_rate'],
            'momentum_20d': indicators.get('momentum_20d', 0),
            'trend_score': indicators.get('trend_score', 0),
            'position_60d': indicators.get('position_60d', 50),
            **future_returns
        }
        
        records.append(record)
    
    if not records:
        return pd.DataFrame(), f"无有效指标: {trade_date}"
    
    result_df = pd.DataFrame(records)
    
    # 评分
    def normalize(series):
        if series.empty or series.max() == series.min():
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min() + 1e-9)
    
    # 动量得分
    result_df['s_momentum'] = normalize(result_df['momentum_20d'])
    
    # 趋势得分
    result_df['s_trend'] = normalize(result_df['trend_score'])
    
    # 位置得分（40-70为佳）
    position_score = np.where(
        (result_df['position_60d'] >= 40) & (result_df['position_60d'] <= 70),
        1.0,
        np.where(
            result_df['position_60d'] < 40,
            result_df['position_60d'] / 40,
            (100 - result_df['position_60d']) / 30
        )
    )
    result_df['s_position'] = position_score
    
    # 综合评分
    result_df['综合评分'] = (
        result_df['s_momentum'] * 0.4 +
        result_df['s_trend'] * 0.3 +
        result_df['s_position'] * 0.3
    ) * 100
    
    result_df = result_df.sort_values('综合评分', ascending=False).head(TOP_BACKTEST)
    return result_df, None

# ---------------------------
# 主运行块
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日回测"):
    
    # 获取交易日
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days:
        st.error("无法获取交易日")
        st.stop()
    
    # 加载历史数据
    st.info("正在加载历史数据...")
    start_time = time.time()
    
    success = get_all_historical_data_fixed(trade_days)
    if not success:
        st.error("数据加载失败")
        st.stop()
    
    load_time = time.time() - start_time
    st.success(f"数据加载完成！耗时: {load_time:.1f}秒")
    
    # 开始回测
    st.header(f"📈 回测 {len(trade_days)} 个交易日")
    
    all_results = []
    valid_days = 0
    
    progress_bar = st.progress(0, text="回测进度")
    status_text = st.empty()
    
    for i, trade_date in enumerate(trade_days):
        status_text.text(f"处理: {trade_date} ({i+1}/{len(trade_days)})")
        
        result, error = run_backtest_single_day(trade_date)
        
        if error:
            st.warning(error)
        elif not result.empty:
            result['Trade_Date'] = trade_date
            all_results.append(result)
            valid_days += 1
        
        progress_bar.progress((i + 1) / len(trade_days))
    
    progress_bar.empty()
    status_text.text(f"回测完成！有效交易日: {valid_days}/{len(trade_days)}")
    
    if not all_results:
        st.error("所有交易日回测均失败")
        st.stop()
    
    # 合并结果
    final_results = pd.concat(all_results, ignore_index=True)
    
    # 显示统计
    st.header("📊 回测统计")
    
    # 收益统计
    for n in [1, 3, 5]:
        col = f'Return_D{n} (%)'
        if col in final_results.columns:
            valid = final_results.dropna(subset=[col])
            if not valid.empty:
                avg_return = valid[col].mean()
                hit_rate = (valid[col] > 0).mean() * 100
                count = len(valid)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"D+{n} 平均收益", f"{avg_return:.2f}%")
                with col2:
                    st.metric(f"D+{n} 胜率", f"{hit_rate:.1f}%")
                with col3:
                    st.metric(f"D+{n} 样本数", count)
    
    # 显示详细结果
    st.header("📋 详细结果")
    
    display_cols = ['Trade_Date', 'ts_code', 'name', '综合评分', 'Close', 
                   'Pct_Chg (%)', 'Circ_MV (亿)', 'momentum_20d', 'trend_score']
    
    # 添加收益列
    for n in [1, 3, 5]:
        col = f'Return_D{n} (%)'
        if col in final_results.columns:
            display_cols.append(col)
    
    available_cols = [col for col in display_cols if col in final_results.columns]
    
    st.dataframe(
        final_results[available_cols].sort_values('Trade_Date', ascending=False),
        use_container_width=True,
        column_config={
            'momentum_20d': st.column_config.NumberColumn(format="%.1f"),
            'trend_score': st.column_config.NumberColumn(format="%.1f"),
        }
    )
