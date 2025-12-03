# -*- coding: utf-8 -*-
"""
选股王 · V16.1 简洁结果版：恢复原结果显示格式 + 优化策略
核心特点：
1. 【**结果显示格式**】：恢复原来的简洁格式，显示收益率和准确率
2. 【**过滤条件恢复**】：最低股价10元，最高300元，最低流通市值20亿
3. 【**策略优化**】：保持V16.0的多因子策略
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
st.set_page_config(page_title="选股王 · V16.1 简洁结果版", layout="wide")
st.title("选股王 · V16.1 简洁结果版")

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
def get_all_historical_data_simple(trade_days_list):
    """
    简化版数据获取
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
                              fields='ts_code,name,list_date')
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
    
    # 3. 获取日线数据（最近60个交易日）
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
    
    # 4. 处理数据
    progress_bar.progress(0.9, text="处理数据...")
    daily_raw_data['trade_date'] = pd.to_datetime(daily_raw_data['trade_date'], format='%Y%m%d')
    GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index()
    
    # 5. 设置通用复权因子
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
# 简化版指标计算
# ----------------------------------------------------------------------
def get_price_data_simple(ts_code, start_date, end_date):
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

def compute_basic_indicators_v2(ts_code, end_date):
    """简化版指标计算"""
    # 获取60日数据
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
    
    df = get_price_data_simple(ts_code, start_date, end_date)
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
    
    # 3. 突破因子
    if len(high) >= 20:
        highest_20d = high.tail(20).max()
        current_high = high.iloc[-1]
        res['breakout_20d'] = 100 if current_high >= highest_20d * 0.99 else 0
    
    # 4. 量价因子
    if len(vol) >= 5:
        vol_5ma = vol.rolling(5).mean()
        if len(vol_5ma) > 0 and vol_5ma.iloc[-1] > 0:
            res['volume_ratio'] = vol.iloc[-1] / vol_5ma.iloc[-1]
    
    # 5. 位置因子
    if len(df) >= 20:
        hist_20 = df.tail(20)
        min_low = hist_20['low'].min()
        max_high = hist_20['high'].max()
        current_close = hist_20['close'].iloc[-1]
        
        if max_high > min_low:
            res['position_20d'] = (current_close - min_low) / (max_high - min_low) * 100
        else:
            res['position_20d'] = 50
    
    # 设置默认值
    res.setdefault('momentum_20d', 0)
    res.setdefault('trend_score', 0)
    res.setdefault('breakout_20d', 0)
    res.setdefault('volume_ratio', 1.0)
    res.setdefault('position_20d', 50)
    
    return res

def get_future_returns_simple(ts_code, selection_date, selection_price):
    """简化版未来收益计算"""
    if pd.isna(selection_price) or selection_price <= 0:
        return {}
    
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    
    # 获取未来数据
    start_date = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    
    df = get_price_data_simple(ts_code, start_date, end_date)
    if df.empty or 'close' not in df.columns:
        return {}
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    
    results = {}
    
    # D+1, D+3, D+5收益
    for n in [1, 3, 5]:
        if len(df) >= n:
            future_price = df.iloc[n-1]['close']
            results[f'Return_D{n} (%)'] = (future_price / selection_price - 1) * 100
    
    return results

# ----------------------------------------------------
# 侧边栏参数 - 恢复原来的设定
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input("选择**回测结束日期**", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("**自动回测天数 (N)**", value=20, step=1, min_value=1, max_value=50, 
                                       help="程序将自动回测最近 N 个交易日。建议设置为 20 天以获得更可靠的统计数据。"))
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = int(st.number_input("最终入围评分数量 (M)", value=100, step=1, min_value=1)) 
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=10, step=1))
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=3, step=1, min_value=1)) 
    
    st.markdown("---")
    st.header("🛒 过滤条件")
    # 恢复原来的设定
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0, step=0.5, min_value=0.1)
    MAX_PRICE = st.number_input("最高股价 (元)", value=300.0, step=5.0, min_value=1.0)
    MIN_TURNOVER = st.number_input("最低换手率 (%)", value=2.0, step=0.5, min_value=0.1) 
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿元)", value=20.0, step=1.0, min_value=1.0, 
                                          help="例如：输入 20 代表流通市值必须大于等于 20 亿元。")
    MIN_AMOUNT_MILLIONS = st.number_input("最低成交额 (亿元)", value=0.6, step=0.1, min_value=0.1)
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
# 回测函数
# ---------------------------
def run_backtest_single_day_v2(trade_date):
    """单个交易日的回测"""
    # 获取当日数据
    daily_data = safe_get('daily', trade_date=trade_date)
    if daily_data.empty:
        return pd.DataFrame(), f"数据缺失或拉取失败：{trade_date}"
    
    # 获取基本面数据
    daily_basic = safe_get('daily_basic', trade_date=trade_date, 
                          fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    
    # 获取资金流数据
    moneyflow = safe_get('moneyflow', trade_date=trade_date)
    
    # 合并数据
    df = daily_data.copy()
    
    if not daily_basic.empty:
        df = df.merge(daily_basic, on='ts_code', how='left')
    
    # 处理资金流数据
    moneyflow_clean = pd.DataFrame(columns=['ts_code', 'net_mf'])
    if not moneyflow.empty:
        possible_cols = ['net_mf', 'net_mf_amount', 'net_mf_in']
        for col in possible_cols:
            if col in moneyflow.columns:
                moneyflow_clean = moneyflow[['ts_code', col]].rename(columns={col: 'net_mf'}).fillna(0)
                break
    
    if not moneyflow_clean.empty:
        df = df.merge(moneyflow_clean, on='ts_code', how='left')
    
    df['net_mf'] = df['net_mf'].fillna(0)
    
    # 数据清洗
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['turnover_rate'] = pd.to_numeric(df.get('turnover_rate', 0), errors='coerce').fillna(0)
    df['amount'] = pd.to_numeric(df.get('amount', 0), errors='coerce').fillna(0) * 1000  # 转换为万元
    df['circ_mv'] = pd.to_numeric(df.get('circ_mv', 0), errors='coerce').fillna(0)
    df['circ_mv_billion'] = df['circ_mv'] / 10000  # 转换为亿元
    
    # 基础过滤
    df = df[
        (df['close'] >= MIN_PRICE) & 
        (df['close'] <= MAX_PRICE) &
        (df['turnover_rate'] >= MIN_TURNOVER) &
        (df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS) &
        (df['amount'] * 1000 >= MIN_AMOUNT)
    ].copy()
    
    if df.empty:
        return pd.DataFrame(), f"硬性过滤后无股票：{trade_date}"
    
    # 计算指标和未来收益
    records = []
    
    for idx, row in df.iterrows():
        ts_code = row['ts_code']
        
        # 计算指标
        indicators = compute_basic_indicators_v2(ts_code, trade_date)
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
            'Circ_MV (亿)': row['circ_mv_billion'],
            'Pct_Chg (%)': row.get('pct_chg', 0),
            'turnover': row['turnover_rate'],
            'net_mf': row['net_mf'],
            'momentum_20d': indicators.get('momentum_20d', 0),
            'trend_score': indicators.get('trend_score', 0),
            'breakout_20d': indicators.get('breakout_20d', 0),
            'volume_ratio': indicators.get('volume_ratio', 1.0),
            'position_20d': indicators.get('position_20d', 50),
            **future_returns
        }
        
        records.append(record)
    
    if not records:
        return pd.DataFrame(), f"无有效指标: {trade_date}"
    
    result_df = pd.DataFrame(records)
    
    # 评分系统
    def normalize(series):
        if len(series) < 2 or series.max() == series.min():
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min() + 1e-9)
    
    # 归一化各因子
    result_df['s_momentum'] = normalize(result_df['momentum_20d'])
    result_df['s_trend'] = normalize(result_df['trend_score'])
    result_df['s_breakout'] = result_df['breakout_20d'] / 100
    
    # 量比得分
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
    
    # 位置得分
    position = result_df['position_20d']
    position_score = np.where(
        (position >= 40) & (position <= 70),
        1.0,
        np.where(
            position < 40,
            position / 40,
            (100 - position) / 30
        )
    )
    result_df['s_position'] = position_score
    
    # V16.1 策略权重
    w_momentum = 0.35      # 动量因子
    w_trend = 0.25         # 趋势因子
    w_breakout = 0.20      # 突破因子
    w_volume = 0.10        # 量价因子
    w_position = 0.10      # 位置因子
    
    # 综合评分
    result_df['综合评分'] = (
        result_df['s_momentum'].fillna(0.5) * w_momentum +
        result_df['s_trend'].fillna(0.5) * w_trend +
        result_df['s_breakout'].fillna(0) * w_breakout +
        result_df['s_volume'].fillna(0.5) * w_volume +
        result_df['s_position'].fillna(0.5) * w_position
    ) * 100
    
    result_df = result_df.sort_values('综合评分', ascending=False).reset_index(drop=True)
    result_df.index += 1
    
    return result_df.head(TOP_BACKTEST), None

# ---------------------------
# 主运行块 - 恢复原来的结果显示格式
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测"):
    
    st.warning("⚠️ **请务必先清除 Streamlit 缓存！**（右上角三点菜单 -> Settings -> Clear Cache）这是让程序强制重新下载数据的关键一步。")
   
    # 获取交易日
    trade_days = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days:
        st.error("无法获取交易日列表，请检查日期或 Token。")
        st.stop()
    
    st.info(f"获取到 {len(trade_days)} 个交易日")
    
    # 加载历史数据
    st.info("正在加载历史数据...")
    start_time = time.time()
    
    success = get_all_historical_data_simple(trade_days)
    if not success:
        st.error("数据加载失败")
        st.stop()
    
    load_time = time.time() - start_time
    st.success(f"数据加载完成！耗时: {load_time:.1f}秒")
    
    # 开始回测
    st.header(f"📈 正在进行 {len(trade_days)} 个交易日的回测...")
    
    results_list = []
    total_days = len(trade_days)
    
    progress_text = st.empty()
    my_bar = st.progress(0)
    
    for i, trade_date in enumerate(trade_days):
        progress_text.text(f"🚀 正在处理第 {i+1}/{total_days} 个交易日：{trade_date}")
        
        daily_result_df, error = run_backtest_single_day_v2(trade_date)
        
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
    
    all_results = pd.concat(results_list, ignore_index=True)
    
    # ✅ 恢复原来的结果显示格式
    st.header(f"📊 最终平均回测结果 (Top {TOP_BACKTEST}，共 {len(all_results['Trade_Date'].unique())} 个有效交易日)")
    
    for n in [1, 3, 5]:
        col = f'Return_D{n} (%)' 
        
        # 过滤掉无效数据
        filtered_returns = all_results.copy()
        valid_returns = filtered_returns.dropna(subset=[col])
        
        if not valid_returns.empty:
            avg_return = valid_returns[col].mean()
            hit_rate = (valid_returns[col] > 0).sum() / len(valid_returns) * 100 if len(valid_returns) > 0 else 0.0
            total_count = len(valid_returns)
        else:
            avg_return = np.nan
            hit_rate = 0.0
            total_count = 0
            
        # 使用st.metric显示结果，恢复原来的格式
        st.metric(f"Top {TOP_BACKTEST}：D+{n} 平均收益 / 准确率", 
                  f"{avg_return:.2f}% / {hit_rate:.1f}%", 
                  help=f"总有效样本数：{total_count}。**V16.1 多因子增强版**")
    
    # 显示每日回测详情
    st.header("📋 每日回测详情 (Top K 明细)")
    
    # 定义显示列，恢复原来的格式
    display_cols = ['Trade_Date', 'name', 'ts_code', '综合评分', 
                    'Close', 'Pct_Chg (%)', 'Circ_MV (亿)',
                    'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)']
    
    # 确保只显示存在的列
    available_cols = [col for col in display_cols if col in all_results.columns]
    
    # 显示数据表格，恢复原来的格式
    st.dataframe(all_results[available_cols].sort_values('Trade_Date', ascending=False), 
                 use_container_width=True,
                 column_config={
                     'Return_D1 (%)': st.column_config.NumberColumn(format="%.2f"),
                     'Return_D3 (%)': st.column_config.NumberColumn(format="%.2f"),
                     'Return_D5 (%)': st.column_config.NumberColumn(format="%.2f"),
                 })
    
    # 添加下载功能
    csv = all_results.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下载完整回测结果",
        data=csv,
        file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # 简单的策略评估
    st.header("💡 策略表现评估")
    
    d1_return = all_results['Return_D1 (%)'].mean() if 'Return_D1 (%)' in all_results.columns else 0
    d5_return = all_results['Return_D5 (%)'].mean() if 'Return_D5 (%)' in all_results.columns else 0
    
    if d1_return > 0.5 and d5_return > 1.0:
        st.success("✅ 策略表现优秀！D+1平均收益 > 0.5%，D+5平均收益 > 1.0%")
    elif d1_return > 0:
        st.info("📈 策略有一定效果，但仍有改进空间")
    else:
        st.warning("⚠️ 策略表现不佳，建议调整参数或策略逻辑")
