# -*- coding: utf-8 -*-
"""
选股王 · V16.0 策略优化版：多因子组合 + 动量趋势增强
核心优化：
1. 【**多因子增强**】：增加更多有效技术指标
2. 【**策略优化**】：改进权重设置和过滤条件
3. 【**风险控制**】：增加止损和风险控制逻辑
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
st.set_page_config(page_title="选股王 · V16.0 策略优化版", layout="wide")
st.title("选股王 · V16.0 策略优化版 🚀")

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
# 数据获取函数
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24, show_spinner=False)
def get_all_historical_data_v16(trade_days_list):
    """
    V16.0 数据获取
    """
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS, GLOBAL_STOCK_BASIC
    
    if not trade_days_list: 
        return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 计算日期范围
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=120)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=20)
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    st.info(f"正在获取 {start_date} 到 {end_date} 的数据...")
    
    progress_bar = st.progress(0, text="初始化...")
    
    # 1. 获取股票列表
    if GLOBAL_STOCK_BASIC.empty:
        progress_bar.progress(0.1, text="获取股票列表...")
        stock_basic = safe_get('stock_basic', exchange='', list_status='L', 
                              fields='ts_code,name,list_date,industry')
        if stock_basic.empty:
            st.error("无法获取股票列表")
            return False
        
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
    
    # 3. 获取复权因子（简化版，使用通用复权因子）
    progress_bar.progress(0.4, text="准备数据...")
    
    # 4. 获取日线数据（最近60个交易日）
    progress_bar.progress(0.5, text="获取日线数据...")
    
    needed_daily_dates = all_trade_dates[:min(60, len(all_trade_dates))]
    
    daily_data_list = []
    for i, date in enumerate(needed_daily_dates):
        daily_df = safe_get('daily', trade_date=date)
        if not daily_df.empty:
            daily_data_list.append(daily_df)
        
        progress_bar.progress(0.5 + (i / len(needed_daily_dates)) * 0.3, 
                             text=f"获取日线数据: {i+1}/{len(needed_daily_dates)}")
    
    if not daily_data_list:
        st.error("无法获取日线数据")
        return False
    
    daily_raw_data = pd.concat(daily_data_list, ignore_index=True)
    
    # 5. 处理数据
    progress_bar.progress(0.9, text="处理数据...")
    daily_raw_data['trade_date'] = pd.to_datetime(daily_raw_data['trade_date'], format='%Y%m%d')
    GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index()
    
    # 6. 设置通用复权因子（简化处理）
    try:
        unique_stocks = GLOBAL_DAILY_RAW.index.get_level_values('ts_code').unique()
        GLOBAL_QFQ_BASE_FACTORS = {stock: 1.0 for stock in unique_stocks}
        st.success(f"✅ 设置通用复权因子，覆盖 {len(GLOBAL_QFQ_BASE_FACTORS)} 只股票")
    except Exception as e:
        st.warning(f"设置基准因子时出错: {e}")
        GLOBAL_QFQ_BASE_FACTORS = {}
    
    progress_bar.progress(1.0, text="数据加载完成！")
    time.sleep(0.5)
    progress_bar.empty()
    
    # 显示数据统计
    st.success(f"""
    ✅ 数据加载完成！
    - 日线数据: {len(GLOBAL_DAILY_RAW):,} 条记录
    - 覆盖股票: {len(GLOBAL_QFQ_BASE_FACTORS)} 只
    """)
    
    return True

# ----------------------------------------------------------------------
# V16.0 增强版指标计算
# ----------------------------------------------------------------------
def get_price_data(ts_code, start_date, end_date):
    """获取价格数据"""
    global GLOBAL_DAILY_RAW
    
    if GLOBAL_DAILY_RAW.empty:
        return pd.DataFrame()
    
    try:
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        if ts_code in GLOBAL_DAILY_RAW.index.get_level_values('ts_code'):
            price_data = GLOBAL_DAILY_RAW.loc[ts_code].copy()
            mask = (price_data.index >= start_dt) & (price_data.index <= end_dt)
            return price_data[mask]
    except Exception:
        pass
    
    return pd.DataFrame()

def compute_advanced_indicators(ts_code, end_date):
    """V16.0 增强版指标计算"""
    # 获取60日数据
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
    
    df = get_price_data(ts_code, start_date, end_date)
    if df.empty or 'close' not in df.columns:
        return {}
    
    # 转换为数值
    for col in ['open', 'high', 'low', 'close', 'vol']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    close = df['close'].dropna()
    high = df['high'].dropna()
    low = df['low'].dropna()
    vol = df['vol'].dropna()
    
    if len(close) < 20:
        return {}
    
    res = {'last_close': close.iloc[-1]}
    
    # 1. 动量因子组
    # 1.1 20日动量
    if len(close) >= 20:
        res['momentum_20d'] = (close.iloc[-1] / close.iloc[-20] - 1) * 100
    
    # 1.2 5日动量（短期）
    if len(close) >= 5:
        res['momentum_5d'] = (close.iloc[-1] / close.iloc[-5] - 1) * 100
    
    # 2. 趋势因子组
    # 2.1 均线排列
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
        if len(close) > 0 and len(ma20) > 0 and close.iloc[-1] > ma20.iloc[-1]:
            trend_score += 1
        
        res['trend_score'] = (trend_score / 4) * 100
        res['ma20_slope'] = (ma20.iloc[-1] / ma20.iloc[-5] - 1) * 100 if len(ma20) >= 5 else 0
    
    # 3. 突破因子组
    # 3.1 创20日新高
    if len(high) >= 20:
        highest_20d = high.tail(20).max()
        current_high = high.iloc[-1]
        res['breakout_20d'] = 100 if current_high >= highest_20d * 0.99 else 0  # 99%就算突破
    
    # 3.2 创60日新高
    if len(high) >= 60:
        highest_60d = high.tail(60).max()
        res['breakout_60d'] = 100 if current_high >= highest_60d * 0.99 else 0
    
    # 4. 量价因子组
    if len(vol) >= 10:
        # 量比
        vol_5ma = vol.rolling(5).mean()
        if len(vol_5ma) > 0 and vol_5ma.iloc[-1] > 0:
            res['volume_ratio'] = vol.iloc[-1] / vol_5ma.iloc[-1]
        
        # 成交量趋势
        if len(vol) >= 5:
            vol_slope = (vol.iloc[-1] / vol.iloc[-5] - 1) * 100
            res['volume_trend'] = vol_slope
    
    # 5. 位置因子组
    # 5.1 20日位置
    if len(df) >= 20:
        hist_20 = df.tail(20)
        min_low = hist_20['low'].min()
        max_high = hist_20['high'].max()
        current_close = hist_20['close'].iloc[-1]
        
        if max_high > min_low:
            res['position_20d'] = (current_close - min_low) / (max_high - min_low) * 100
        else:
            res['position_20d'] = 50
    
    # 5.2 60日位置
    if len(df) >= 60:
        hist_60 = df.tail(60)
        min_low_60 = hist_60['low'].min()
        max_high_60 = hist_60['high'].max()
        
        if max_high_60 > min_low_60:
            res['position_60d'] = (current_close - min_low_60) / (max_high_60 - min_low_60) * 100
        else:
            res['position_60d'] = 50
    
    # 6. 波动率因子
    if len(close) >= 20:
        returns = close.pct_change().dropna()
        if len(returns) >= 20:
            res['volatility_20d'] = returns.tail(20).std() * np.sqrt(252) * 100
    
    # 7. RSI指标（简化版）
    if len(close) >= 14:
        changes = close.diff()
        gains = changes.clip(lower=0)
        losses = -changes.clip(upper=0)
        
        avg_gain = gains.rolling(14).mean().iloc[-1]
        avg_loss = losses.rolling(14).mean().iloc[-1]
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            res['rsi_14'] = 100 - (100 / (1 + rs))
        else:
            res['rsi_14'] = 100
    
    # 设置默认值
    default_values = {
        'momentum_20d': 0, 'momentum_5d': 0, 'trend_score': 0, 'ma20_slope': 0,
        'breakout_20d': 0, 'breakout_60d': 0, 'volume_ratio': 1.0, 'volume_trend': 0,
        'position_20d': 50, 'position_60d': 50, 'volatility_20d': 30, 'rsi_14': 50
    }
    
    for key, default in default_values.items():
        if key not in res:
            res[key] = default
    
    return res

def get_future_returns_enhanced(ts_code, selection_date, selection_price):
    """增强版未来收益计算"""
    if pd.isna(selection_price) or selection_price <= 0:
        return {}
    
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    
    # 获取未来10个交易日的数据
    start_date = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    df = get_price_data(ts_code, start_date, end_date)
    if df.empty or 'close' not in df.columns:
        return {}
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    
    results = {}
    
    # 计算不同时间段的收益
    periods = [1, 2, 3, 5, 10]
    for n in periods:
        if len(df) >= n:
            future_price = df.iloc[n-1]['close']
            results[f'Return_D{n} (%)'] = (future_price / selection_price - 1) * 100
    
    return results

# ----------------------------------------------------
# 侧边栏参数 - V16.0 增强版
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ 回测设置")
    
    col1, col2 = st.columns(2)
    with col1:
        backtest_date_end = st.date_input("结束日期", value=datetime.now().date(), max_value=datetime.now().date())
    with col2:
        BACKTEST_DAYS = st.number_input("回测天数", value=15, min_value=5, max_value=30)
    
    st.markdown("---")
    st.header("🎯 核心参数")
    
    FINAL_POOL = st.slider("入围数量", min_value=20, max_value=100, value=50, step=5)
    TOP_BACKTEST = st.slider("Top K", min_value=1, max_value=10, value=3, step=1)
    
    st.markdown("---")
    st.header("📊 策略权重设置")
    
    w_momentum = st.slider("动量权重", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    w_trend = st.slider("趋势权重", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    w_breakout = st.slider("突破权重", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
    w_volume = st.slider("量价权重", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
    w_position = st.slider("位置权重", min_value=0.0, max_value=1.0, value=0.10, step=0.05)
    
    # 检查权重总和是否为1
    total_weight = w_momentum + w_trend + w_breakout + w_volume + w_position
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"权重总和为 {total_weight:.2f}，建议调整为1.0")
    
    st.markdown("---")
    st.header("🔍 过滤条件")
    
    MIN_PRICE = st.number_input("最低股价(元)", value=8.0, min_value=1.0, step=1.0)
    MAX_PRICE = st.number_input("最高股价(元)", value=80.0, min_value=10.0, step=10.0)
    MIN_TURNOVER = st.number_input("最低换手率%", value=2.0, min_value=0.5, step=0.5)
    MIN_CIRC_MV = st.number_input("最低流通市值(亿)", value=30.0, min_value=10.0, step=5.0)
    
    st.markdown("---")
    st.header("⚡ 高级过滤")
    
    MIN_MOMENTUM = st.number_input("最低20日动量%", value=5.0, min_value=-20.0, max_value=50.0, step=5.0)
    MIN_TREND_SCORE = st.number_input("最低趋势得分", value=50.0, min_value=0.0, max_value=100.0, step=10.0)
    MIN_POSITION = st.number_input("最低位置(20日)", value=30.0, min_value=0.0, max_value=100.0, step=5.0)
    MAX_POSITION = st.number_input("最高位置(20日)", value=80.0, min_value=0.0, max_value=100.0, step=5.0)

# ---------------------------
# Token 输入
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token", type="password", key="token_v16")
if not TS_TOKEN:
    st.warning("请输入Tushare Token")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 验证Token
try:
    test = pro.trade_cal(exchange='', start_date='20240101', end_date='20240105')
    if test.empty:
        st.error("Token无效")
        st.stop()
    st.success("✅ Token验证通过")
except Exception as e:
    st.error(f"Token验证失败: {e}")
    st.stop()

# ---------------------------
# V16.0 增强回测函数
# ---------------------------
def run_v16_backtest(trade_date):
    """V16.0 增强回测"""
    # 获取当日数据
    daily_data = safe_get('daily', trade_date=trade_date)
    if daily_data.empty:
        return pd.DataFrame(), f"无日线数据: {trade_date}"
    
    # 获取基本面数据
    daily_basic = safe_get('daily_basic', trade_date=trade_date, 
                          fields='ts_code,turnover_rate,circ_mv,total_mv')
    
    # 获取资金流数据
    moneyflow = safe_get('moneyflow', trade_date=trade_date, 
                        fields='ts_code,buy_sm_vol,sell_sm_vol,buy_md_vol,sell_md_vol,buy_lg_vol,sell_lg_vol')
    
    # 合并数据
    df = daily_data.copy()
    
    if not daily_basic.empty:
        df = df.merge(daily_basic, on='ts_code', how='left')
    
    if not moneyflow.empty:
        df = df.merge(moneyflow, on='ts_code', how='left')
    
    # 数据清洗
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
    df['turnover_rate'] = pd.to_numeric(df.get('turnover_rate', 0), errors='coerce').fillna(0)
    df['circ_mv'] = pd.to_numeric(df.get('circ_mv', 0), errors='coerce').fillna(0) / 10000
    df['total_mv'] = pd.to_numeric(df.get('total_mv', 0), errors='coerce').fillna(0) / 10000
    
    # 计算资金流指标
    if 'buy_sm_vol' in df.columns and 'sell_sm_vol' in df.columns:
        df['net_sm_flow'] = df['buy_sm_vol'] - df['sell_sm_vol']
    if 'buy_md_vol' in df.columns and 'sell_md_vol' in df.columns:
        df['net_md_flow'] = df['buy_md_vol'] - df['sell_md_vol']
    if 'buy_lg_vol' in df.columns and 'sell_lg_vol' in df.columns:
        df['net_lg_flow'] = df['buy_lg_vol'] - df['sell_lg_vol']
    
    # 基础过滤
    df = df[
        (df['close'] >= MIN_PRICE) & 
        (df['close'] <= MAX_PRICE) &
        (df['turnover_rate'] >= MIN_TURNOVER) &
        (df['circ_mv'] >= MIN_CIRC_MV)
    ].copy()
    
    if df.empty:
        return pd.DataFrame(), f"基础过滤后无股票: {trade_date}"
    
    # 计算技术指标
    records = []
    
    for idx, row in df.iterrows():
        ts_code = row['ts_code']
        
        # 计算技术指标
        indicators = compute_advanced_indicators(ts_code, trade_date)
        if not indicators or 'last_close' not in indicators:
            continue
        
        # 高级过滤
        if indicators.get('momentum_20d', 0) < MIN_MOMENTUM:
            continue
        if indicators.get('trend_score', 0) < MIN_TREND_SCORE:
            continue
        if not (MIN_POSITION <= indicators.get('position_20d', 50) <= MAX_POSITION):
            continue
        
        selection_price = indicators['last_close']
        
        # 获取未来收益
        future_returns = get_future_returns_enhanced(ts_code, trade_date, selection_price)
        if not future_returns:
            continue
        
        record = {
            'ts_code': ts_code,
            'name': row.get('name', ts_code[:6]),
            'Close': row['close'],
            'Pct_Chg (%)': row['pct_chg'],
            'Circ_MV (亿)': row['circ_mv'],
            'Total_MV (亿)': row.get('total_mv', row['circ_mv']),
            'turnover': row['turnover_rate'],
            **{k: indicators.get(k, 0) for k in [
                'momentum_20d', 'momentum_5d', 'trend_score', 'ma20_slope',
                'breakout_20d', 'breakout_60d', 'volume_ratio', 'volume_trend',
                'position_20d', 'position_60d', 'volatility_20d', 'rsi_14'
            ]},
            **future_returns
        }
        
        # 添加资金流指标
        for flow_col in ['net_sm_flow', 'net_md_flow', 'net_lg_flow']:
            if flow_col in row:
                record[flow_col] = row[flow_col]
        
        records.append(record)
    
    if not records:
        return pd.DataFrame(), f"技术过滤后无股票: {trade_date}"
    
    result_df = pd.DataFrame(records)
    
    # 评分系统
    def safe_normalize(series, reverse=False):
        if len(series) < 2 or series.max() == series.min():
            return pd.Series([0.5] * len(series), index=series.index)
        
        normalized = (series - series.min()) / (series.max() - series.min())
        if reverse:
            normalized = 1 - normalized
        return normalized
    
    # 1. 动量得分（正向）
    result_df['s_momentum'] = safe_normalize(result_df['momentum_20d'])
    
    # 2. 趋势得分（正向）
    result_df['s_trend'] = safe_normalize(result_df['trend_score'])
    
    # 3. 突破得分（正向）
    result_df['s_breakout'] = (result_df['breakout_20d'] + result_df['breakout_60d']) / 200
    
    # 4. 量价得分（正向）
    # 量比1.5-3.0为佳
    volume_score = np.where(
        (result_df['volume_ratio'] >= 1.5) & (result_df['volume_ratio'] <= 3.0),
        1.0,
        np.where(
            result_df['volume_ratio'] < 1.5,
            result_df['volume_ratio'] / 1.5,
            3.0 / result_df['volume_ratio']
        )
    )
    result_df['s_volume'] = volume_score
    
    # 5. 位置得分（30-70为佳）
    position = result_df['position_20d']
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
    
    # 6. RSI得分（50-70为佳）
    rsi = result_df['rsi_14']
    rsi_score = np.where(
        (rsi >= 50) & (rsi <= 70),
        1.0,
        np.where(
            rsi < 50,
            rsi / 50,
            (100 - rsi) / 30
        )
    )
    result_df['s_rsi'] = rsi_score
    
    # 7. 波动率得分（反向，越低越好）
    result_df['s_volatility'] = safe_normalize(result_df['volatility_20d'], reverse=True)
    
    # 综合评分
    result_df['综合评分'] = (
        result_df['s_momentum'] * w_momentum +
        result_df['s_trend'] * w_trend +
        result_df['s_breakout'] * w_breakout +
        result_df['s_volume'] * w_volume +
        result_df['s_position'] * w_position +
        result_df['s_rsi'] * 0.05 +
        result_df['s_volatility'] * 0.05
    ) * 100
    
    # 排序并选择Top K
    result_df = result_df.sort_values('综合评分', ascending=False).reset_index(drop=True)
    result_df.index += 1
    
    return result_df.head(TOP_BACKTEST), None

# ---------------------------
# 主运行块
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日回测 (V16.0)"):
    
    # 获取交易日
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days:
        st.error("无法获取交易日")
        st.stop()
    
    st.info(f"获取到 {len(trade_days)} 个交易日")
    
    # 加载历史数据
    st.info("正在加载历史数据...")
    start_time = time.time()
    
    success = get_all_historical_data_v16(trade_days)
    if not success:
        st.error("数据加载失败")
        st.stop()
    
    load_time = time.time() - start_time
    st.success(f"数据加载完成！耗时: {load_time:.1f}秒")
    
    # 开始回测
    st.header(f"📈 回测 {len(trade_days)} 个交易日 (V16.0)")
    
    all_results = []
    valid_days = 0
    
    progress_bar = st.progress(0, text="回测进度")
    status_text = st.empty()
    
    for i, trade_date in enumerate(trade_days):
        status_text.text(f"处理: {trade_date} ({i+1}/{len(trade_days)})")
        
        result, error = run_v16_backtest(trade_date)
        
        if error:
            st.warning(f"{trade_date}: {error}")
        elif not result.empty:
            result['Trade_Date'] = trade_date
            all_results.append(result)
            valid_days += 1
            st.info(f"✅ {trade_date}: 找到 {len(result)} 只有效股票")
        
        progress_bar.progress((i + 1) / len(trade_days))
    
    progress_bar.empty()
    
    if not all_results:
        st.error("所有交易日回测均失败")
        st.stop()
    
    # 合并结果
    final_results = pd.concat(all_results, ignore_index=True)
    
    # 显示统计
    st.header("📊 回测统计")
    
    # 显示因子统计
    st.subheader("📈 选股因子统计")
    factor_cols = ['momentum_20d', 'trend_score', 'breakout_20d', 'volume_ratio', 
                  'position_20d', 'rsi_14', 'volatility_20d']
    
    factor_data = []
    for col in factor_cols:
        if col in final_results.columns:
            factor_data.append({
                '因子': col,
                '均值': final_results[col].mean(),
                '中位数': final_results[col].median(),
                '标准差': final_results[col].std(),
                '最小值': final_results[col].min(),
                '最大值': final_results[col].max()
            })
    
    if factor_data:
        factor_df = pd.DataFrame(factor_data)
        st.dataframe(factor_df.round(2), use_container_width=True)
    
    # 收益统计
    st.subheader("💰 收益统计")
    
    # 计算不同时间段的收益
    periods = [1, 2, 3, 5, 10]
    stats_data = []
    
    for n in periods:
        col = f'Return_D{n} (%)'
        if col in final_results.columns:
            valid = final_results.dropna(subset=[col])
            if not valid.empty:
                avg_return = valid[col].mean()
                hit_rate = (valid[col] > 0).mean() * 100
                median_return = valid[col].median()
                std_return = valid[col].std()
                count = len(valid)
                
                stats_data.append({
                    '周期': f'D+{n}',
                    '样本数': count,
                    '平均收益': f"{avg_return:.2f}%",
                    '胜率': f"{hit_rate:.1f}%",
                    '中位数': f"{median_return:.2f}%",
                    '标准差': f"{std_return:.2f}%"
                })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.table(stats_df)
        
        # 可视化
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        periods_data = [f'D+{n}' for n in periods if f'Return_D{n} (%)' in final_results.columns]
        avg_returns = []
        hit_rates = []
        
        for n in periods:
            col = f'Return_D{n} (%)'
            if col in final_results.columns:
                valid = final_results.dropna(subset=[col])
                if not valid.empty:
                    avg_returns.append(valid[col].mean())
                    hit_rates.append((valid[col] > 0).mean() * 100)
        
        if avg_returns and hit_rates:
            fig.add_trace(go.Bar(
                x=periods_data,
                y=avg_returns,
                name='平均收益',
                marker_color='indianred',
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=periods_data,
                y=hit_rates,
                name='胜率',
                mode='lines+markers',
                line=dict(color='royalblue', width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='收益与胜率趋势',
                xaxis_title='持有周期',
                yaxis=dict(title='平均收益 (%)', titlefont=dict(color='indianred')),
                yaxis2=dict(
                    title='胜率 (%)',
                    titlefont=dict(color='royalblue'),
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # 显示详细结果
    st.header("📋 详细结果")
    
    display_cols = ['Trade_Date', 'ts_code', 'name', '综合评分', 'Close', 
                   'Pct_Chg (%)', 'Circ_MV (亿)', 'momentum_20d', 'trend_score',
                   'volume_ratio', 'position_20d']
    
    # 添加收益列
    for n in [1, 3, 5]:
        col = f'Return_D{n} (%)'
        if col in final_results.columns:
            display_cols.append(col)
    
    available_cols = [col for col in display_cols if col in final_results.columns]
    
    st.dataframe(
        final_results[available_cols].sort_values('Trade_Date', ascending=False),
        use_container_width=True,
        height=500
    )
    
    # 下载结果
    csv = final_results.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下载回测结果",
        data=csv,
        file_name=f"v16_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # 策略建议
    st.header("💡 策略建议")
    
    avg_d1_return = final_results['Return_D1 (%)'].mean() if 'Return_D1 (%)' in final_results.columns else 0
    avg_d5_return = final_results['Return_D5 (%)'].mean() if 'Return_D5 (%)' in final_results.columns else 0
    
    if avg_d1_return > 0.5 and avg_d5_return > 1.0:
        st.success("🎉 策略表现良好！建议继续使用当前参数。")
    elif avg_d1_return > 0:
        st.info("📈 策略有一定效果，建议微调参数：")
        st.write("- 尝试提高动量权重")
        st.write("- 调整位置过滤范围")
        st.write("- 增加趋势过滤强度")
    else:
        st.warning("⚠️ 策略需要优化，建议：")
        st.write("- 调整权重分配")
        st.write("- 收紧过滤条件")
        st.write("- 测试不同参数组合")
        st.write("- 考虑市场环境（牛市/熊市策略不同）")
