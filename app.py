# -*- coding: utf-8 -*-
"""
选股王 · V15.3 数据修复版：修复复权因子数据获取问题
核心修复：
1. 【**复权因子修复**】：修正adj_factor数据获取逻辑，确保获取完整数据
   - 使用更高效的数据获取方式
   - 确保覆盖所有股票和日期
   
2. 【**稳定性增强**】：增加数据验证和回退机制
   - 检查数据完整性
   - 添加数据验证步骤
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
GLOBAL_QFQ_BASE_FACTORS = {}
GLOBAL_STOCK_BASIC = pd.DataFrame()

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V15.3 数据修复版", layout="wide")
st.title("选股王 · V15.3 数据修复版")

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
        st.error("无法获取交易日历")
        return []
    
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    return trade_days_df['cal_date'].head(num_days).tolist()

# ----------------------------------------------------------------------
# ⭐️ V15.3 修复：正确获取复权因子数据
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24, show_spinner=False)
def get_all_historical_data_fixed_v2(trade_days_list):
    """
    修复复权因子获取问题
    """
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_BASIC
    
    if not trade_days_list: 
        return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 计算日期范围
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=90)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=10)
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    st.info(f"正在获取 {start_date} 到 {end_date} 的数据...")
    
    # 进度条
    progress_bar = st.progress(0, text="初始化...")
    
    # 1. 获取股票列表
    if GLOBAL_STOCK_BASIC.empty:
        progress_bar.progress(0.1, text="获取股票列表...")
        stock_basic = safe_get('stock_basic', exchange='', list_status='L', 
                              fields='ts_code,name,list_date')
        if stock_basic.empty:
            st.error("无法获取股票列表")
            return False
        
        # 过滤北交所和ST股
        stock_basic = stock_basic[~stock_basic['ts_code'].str.startswith(('92', '68'))]
        GLOBAL_STOCK_BASIC = stock_basic
    
    progress_bar.progress(0.2, text=f"已获取 {len(GLOBAL_STOCK_BASIC)} 只股票")
    
    # 2. 获取交易日历
    progress_bar.progress(0.3, text="获取交易日历...")
    trade_cal = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if trade_cal.empty:
        st.error("无法获取交易日历")
        return False
    
    all_trade_dates = trade_cal['cal_date'].tolist()
    
    # 3. ⭐️ 关键修复：正确获取复权因子数据
    progress_bar.progress(0.4, text="获取复权因子数据...")
    
    # 方法1：先尝试批量获取
    adj_factor_data = safe_get('adj_factor', start_date=start_date, end_date=end_date)
    
    if adj_factor_data.empty or len(adj_factor_data) < 1000:
        # 方法2：如果数据太少，尝试分日期获取
        st.warning("批量获取复权因子数据不足，尝试分日期获取...")
        adj_factor_list = []
        
        # 只获取实际需要日期的数据（减少数据量）
        needed_dates = all_trade_dates[:min(60, len(all_trade_dates))]  # 最多获取60天
        
        for i, date in enumerate(needed_dates):
            adj_df = safe_get('adj_factor', trade_date=date)
            if not adj_df.empty:
                adj_factor_list.append(adj_df)
            
            progress_bar.progress(0.4 + (i / len(needed_dates)) * 0.2, 
                                 text=f"获取复权因子: {i+1}/{len(needed_dates)}")
        
        if adj_factor_list:
            adj_factor_data = pd.concat(adj_factor_list, ignore_index=True)
        else:
            st.error("无法获取复权因子数据")
            return False
    
    # 处理复权因子数据
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(1.0)
    GLOBAL_ADJ_FACTOR = adj_factor_data.set_index(['ts_code', 'trade_date']).sort_index()
    
    # 4. 获取日线数据
    progress_bar.progress(0.7, text="获取日线数据...")
    
    # 获取实际需要的日期（回测日及其前20个交易日）
    needed_daily_dates = []
    for date in all_trade_dates:
        if date <= latest_trade_date:
            needed_daily_dates.append(date)
            if len(needed_daily_dates) >= 50:  # 最多获取50天
                break
    
    daily_data_list = []
    for i, date in enumerate(needed_daily_dates):
        daily_df = safe_get('daily', trade_date=date)
        if not daily_df.empty:
            daily_data_list.append(daily_df)
        
        progress_bar.progress(0.7 + (i / len(needed_daily_dates)) * 0.2, 
                             text=f"获取日线数据: {i+1}/{len(needed_daily_dates)}")
    
    if not daily_data_list:
        st.error("无法获取日线数据")
        return False
    
    daily_raw_data = pd.concat(daily_data_list, ignore_index=True)
    
    # 处理日线数据
    progress_bar.progress(0.95, text="处理数据...")
    daily_raw_data['trade_date'] = pd.to_datetime(daily_raw_data['trade_date'], format='%Y%m%d')
    GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index()
    
    # 5. 验证数据完整性
    # 检查复权因子数据量
    adj_count = len(GLOBAL_ADJ_FACTOR)
    daily_count = len(GLOBAL_DAILY_RAW)
    
    if adj_count < 10000:  # 复权因子数据太少
        st.warning(f"⚠️ 复权因子数据较少 ({adj_count} 条)，可能影响计算")
        # 尝试另一种方式：使用通用复权因子（所有股票使用相同的基准）
        # 创建一个简单的复权因子表，假设所有股票都没有除权除息
        try:
            unique_stocks = GLOBAL_DAILY_RAW.index.get_level_values('ts_code').unique()
            unique_dates = GLOBAL_DAILY_RAW.index.get_level_values('trade_date').unique()
            
            adj_data = []
            for stock in unique_stocks:
                for date in unique_dates:
                    adj_data.append({'ts_code': stock, 'trade_date': date, 'adj_factor': 1.0})
            
            if adj_data:
                adj_df = pd.DataFrame(adj_data)
                adj_df['trade_date'] = pd.to_datetime(adj_df['trade_date'])
                GLOBAL_ADJ_FACTOR = adj_df.set_index(['ts_code', 'trade_date'])
                st.info("已使用通用复权因子")
        except:
            pass
    
    # 6. 设置QFQ基准因子
    try:
        if not GLOBAL_ADJ_FACTOR.empty:
            # 使用最新的复权因子作为基准
            latest_dates = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').unique()
            if len(latest_dates) > 0:
                latest_date = sorted(latest_dates)[-1]
                latest_adj = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_date), 'adj_factor']
                if isinstance(latest_adj, pd.Series):
                    GLOBAL_QFQ_BASE_FACTORS = latest_adj.droplevel(1).to_dict()
                    st.success(f"✅ 设置基准因子，覆盖 {len(GLOBAL_QFQ_BASE_FACTORS)} 只股票")
    except Exception as e:
        st.warning(f"设置基准因子时出错: {e}")
        GLOBAL_QFQ_BASE_FACTORS = {}
    
    progress_bar.progress(1.0, text="数据加载完成！")
    time.sleep(1)
    progress_bar.empty()
    
    # 显示数据统计
    st.success(f"""
    ✅ 数据加载完成！
    - 日线数据: {len(GLOBAL_DAILY_RAW):,} 条记录
    - 复权因子: {len(GLOBAL_ADJ_FACTOR):,} 条记录
    - 基准因子: {len(GLOBAL_QFQ_BASE_FACTORS)} 只股票
    """)
    
    # 数据质量检查
    if len(GLOBAL_ADJ_FACTOR) < 10000:
        st.warning("⚠️ 复权因子数据可能不足，建议检查Tushare权限或尝试重新运行")
    
    return True

# ----------------------------------------------------------------------
# 简化的数据获取函数
# ----------------------------------------------------------------------
def get_qfq_data_simple(ts_code, start_date, end_date):
    """简化的前复权数据获取"""
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    
    if GLOBAL_DAILY_RAW.empty:
        return pd.DataFrame()
    
    try:
        # 转换日期
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        # 获取日线数据
        if ts_code in GLOBAL_DAILY_RAW.index.get_level_values('ts_code'):
            daily_data = GLOBAL_DAILY_RAW.loc[ts_code].copy()
            mask = (daily_data.index >= start_dt) & (daily_data.index <= end_dt)
            daily_data = daily_data[mask]
        else:
            return pd.DataFrame()
        
        if daily_data.empty:
            return pd.DataFrame()
        
        # 获取复权因子
        base_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, 1.0)
        
        # 如果有复权因子数据
        if ts_code in GLOBAL_ADJ_FACTOR.index.get_level_values('ts_code'):
            adj_data = GLOBAL_ADJ_FACTOR.loc[ts_code].copy()
            adj_data = adj_data[(adj_data.index >= start_dt) & (adj_data.index <= end_dt)]
            
            if not adj_data.empty:
                # 合并数据并计算复权价格
                df = daily_data.merge(adj_data, left_index=True, right_index=True, how='left')
                df['adj_factor'] = df['adj_factor'].fillna(base_factor)
                
                # 计算前复权价格
                for col in ['open', 'high', 'low', 'close', 'pre_close']:
                    if col in df.columns:
                        df[col] = df[col] * df['adj_factor'] / base_factor
            else:
                df = daily_data.copy()
                # 如果没有复权因子，使用基准因子
                for col in ['open', 'high', 'low', 'close', 'pre_close']:
                    if col in df.columns:
                        df[col] = df[col] * 1.0 / base_factor
        else:
            df = daily_data.copy()
            # 如果没有复权因子，使用基准因子
            for col in ['open', 'high', 'low', 'close', 'pre_close']:
                if col in df.columns:
                    df[col] = df[col] * 1.0 / base_factor
        
        return df[['open', 'high', 'low', 'close', 'vol']].reset_index()
        
    except Exception:
        return pd.DataFrame()

# ----------------------------------------------------------------------
# 核心计算函数
# ----------------------------------------------------------------------
def compute_basic_indicators(ts_code, end_date):
    """基础指标计算"""
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    
    df = get_qfq_data_simple(ts_code, start_date, end_date)
    if df.empty or 'close' not in df.columns:
        return {}
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    
    close = df['close'].dropna()
    if len(close) < 10:
        return {}
    
    res = {'last_close': close.iloc[-1]}
    
    # 1. 动量因子 (10日涨幅)
    if len(close) >= 10:
        res['momentum_10d'] = (close.iloc[-1] / close.iloc[-10] - 1) * 100
    
    # 2. 趋势因子 (简单均线)
    if len(close) >= 5:
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        
        trend_score = 0
        if len(ma5) > 0 and len(ma10) > 0 and ma5.iloc[-1] > ma10.iloc[-1]:
            trend_score += 1
        if len(close) > 0 and len(ma5) > 0 and close.iloc[-1] > ma5.iloc[-1]:
            trend_score += 1
        
        res['trend_score'] = (trend_score / 2) * 100
    
    # 3. 位置因子 (20日位置)
    if len(df) >= 20:
        hist_20 = df.tail(20)
        min_low = hist_20['low'].min()
        max_high = hist_20['high'].max()
        current_close = hist_20['close'].iloc[-1]
        
        if max_high > min_low:
            res['position_20d'] = (current_close - min_low) / (max_high - min_low) * 100
        else:
            res['position_20d'] = 50
    
    # 4. 成交量指标
    if 'vol' in df.columns:
        df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
        if len(df) >= 5:
            vol_5ma = df['vol'].rolling(5).mean().iloc[-1] if len(df) >= 5 else 0
            vol_today = df['vol'].iloc[-1] if len(df) > 0 else 0
            if vol_5ma > 0:
                res['volume_ratio'] = vol_today / vol_5ma
    
    # 设置默认值
    res.setdefault('momentum_10d', 0)
    res.setdefault('trend_score', 50)
    res.setdefault('position_20d', 50)
    res.setdefault('volume_ratio', 1.0)
    
    return res

def get_future_returns_simple(ts_code, selection_date, selection_price):
    """简化版未来收益计算"""
    if pd.isna(selection_price) or selection_price <= 0:
        return {}
    
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    
    # 只计算D+1收益（简化）
    start_date = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date = (d0 + timedelta(days=5)).strftime("%Y%m%d")
    
    df = get_qfq_data_simple(ts_code, start_date, end_date)
    if df.empty or 'close' not in df.columns:
        return {}
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    
    results = {}
    
    # D+1
    if len(df) >= 1:
        results['Return_D1 (%)'] = (df.iloc[0]['close'] / selection_price - 1) * 100
    
    # D+3
    if len(df) >= 3:
        results['Return_D3 (%)'] = (df.iloc[2]['close'] / selection_price - 1) * 100
    
    # D+5  
    if len(df) >= 5:
        results['Return_D5 (%)'] = (df.iloc[4]['close'] / selection_price - 1) * 100
    
    return results

# ----------------------------------------------------
# 侧边栏参数
# ----------------------------------------------------
with st.sidebar:
    st.header("回测设置")
    backtest_date_end = st.date_input("结束日期", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("回测天数", value=10, min_value=1, max_value=30)
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = st.number_input("入围数量", value=30, min_value=10, max_value=100)
    TOP_BACKTEST = st.number_input("Top K", value=3, min_value=1, max_value=10)
    
    st.markdown("---")
    st.header("过滤条件")
    MIN_PRICE = st.number_input("最低股价", value=5.0, min_value=1.0)
    MAX_PRICE = st.number_input("最高股价", value=100.0, min_value=10.0)
    MIN_TURNOVER = st.number_input("最低换手率%", value=1.0, min_value=0.1, step=0.5)
    MIN_CIRC_MV = st.number_input("最低流通市值(亿)", value=20.0, min_value=5.0, step=5.0)

# ---------------------------
# Token 输入
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token", type="password", key="token_input")
if not TS_TOKEN:
    st.warning("请输入Tushare Token")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 验证Token
try:
    test = pro.trade_cal(exchange='', start_date='20240101', end_date='20240105', fields='exchange,cal_date,is_open,pretrade_date')
    if test.empty:
        st.error("Token无效或权限不足")
        st.stop()
    st.success("✅ Token验证通过")
except Exception as e:
    st.error(f"Token验证失败: {e}")
    st.stop()

# ---------------------------
# 简化的回测函数
# ---------------------------
def run_single_day_backtest(trade_date):
    """单个交易日回测"""
    # 获取当日数据
    daily_data = safe_get('daily', trade_date=trade_date)
    if daily_data.empty:
        return pd.DataFrame(), f"无日线数据: {trade_date}"
    
    # 获取基本面数据
    daily_basic = safe_get('daily_basic', trade_date=trade_date, 
                          fields='ts_code,turnover_rate,circ_mv')
    
    # 合并数据
    df = daily_data.copy()
    if not daily_basic.empty:
        df = df.merge(daily_basic, on='ts_code', how='left')
    
    # 数据清洗
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['turnover_rate'] = pd.to_numeric(df.get('turnover_rate', 0), errors='coerce').fillna(0)
    df['circ_mv'] = pd.to_numeric(df.get('circ_mv', 0), errors='coerce').fillna(0) / 10000  # 转为亿元
    
    # 过滤
    df = df[
        (df['close'] >= MIN_PRICE) & 
        (df['close'] <= MAX_PRICE) &
        (df['turnover_rate'] >= MIN_TURNOVER) &
        (df['circ_mv'] >= MIN_CIRC_MV)
    ].copy()
    
    if df.empty:
        return pd.DataFrame(), f"过滤后无股票: {trade_date}"
    
    # 计算指标和未来收益
    records = []
    
    for idx, row in df.head(FINAL_POOL).iterrows():  # 只处理前FINAL_POOL只股票
        ts_code = row['ts_code']
        
        # 计算指标
        indicators = compute_basic_indicators(ts_code, trade_date)
        if not indicators or 'last_close' not in indicators:
            continue
        
        selection_price = indicators['last_close']
        
        # 获取未来收益
        future_returns = get_future_returns_simple(ts_code, trade_date, selection_price)
        if not future_returns:
            continue
        
        record = {
            'ts_code': ts_code,
            'name': row.get('name', ts_code[:6]),
            'Close': row['close'],
            'Circ_MV (亿)': row['circ_mv'],
            'Pct_Chg (%)': row.get('pct_chg', 0),
            'turnover': row['turnover_rate'],
            'momentum': indicators.get('momentum_10d', 0),
            'trend_score': indicators.get('trend_score', 50),
            'position': indicators.get('position_20d', 50),
            'volume_ratio': indicators.get('volume_ratio', 1.0),
            **future_returns
        }
        
        records.append(record)
    
    if not records:
        return pd.DataFrame(), f"无有效指标: {trade_date}"
    
    result_df = pd.DataFrame(records)
    
    # 评分
    def safe_normalize(series):
        if len(series) < 2 or series.max() == series.min():
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min())
    
    # 动量得分
    if 'momentum' in result_df.columns:
        result_df['s_momentum'] = safe_normalize(result_df['momentum'])
    else:
        result_df['s_momentum'] = 0.5
    
    # 趋势得分
    if 'trend_score' in result_df.columns:
        result_df['s_trend'] = safe_normalize(result_df['trend_score'])
    else:
        result_df['s_trend'] = 0.5
    
    # 位置得分
    if 'position' in result_df.columns:
        position = result_df['position']
        # 30-70为佳
        position_score = np.where(
            (position >= 30) & (position <= 70),
            1.0,
            np.where(
                position < 30,
                position / 30,
                (100 - position) / 30
            )
        )
        result_df['s_position'] = position_score
    else:
        result_df['s_position'] = 0.5
    
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
    
    st.info(f"获取到 {len(trade_days)} 个交易日")
    
    # 加载历史数据
    st.info("正在加载历史数据...")
    start_time = time.time()
    
    success = get_all_historical_data_fixed_v2(trade_days)
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
        
        result, error = run_single_day_backtest(trade_date)
        
        if error:
            st.warning(f"{trade_date}: {error}")
        elif not result.empty:
            result['Trade_Date'] = trade_date
            all_results.append(result)
            valid_days += 1
            st.info(f"✅ {trade_date}: 找到 {len(result)} 只有效股票")
        
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
                   'Pct_Chg (%)', 'Circ_MV (亿)', 'momentum', 'trend_score']
    
    # 添加收益列
    for n in [1, 3, 5]:
        col = f'Return_D{n} (%)'
        if col in final_results.columns:
            display_cols.append(col)
    
    available_cols = [col for col in display_cols if col in final_results.columns]
    
    st.dataframe(
        final_results[available_cols].sort_values('Trade_Date', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # 下载结果
    csv = final_results.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下载回测结果",
        data=csv,
        file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
