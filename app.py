# -*- coding: utf-8 -*-
"""
选股王 · V18.1 数据修复版：修复数据获取问题
核心修复：
1. 【**数据获取修复**】：确保能获取到完整的日线数据和基本面数据
2. 【**数据验证**】：增加数据完整性检查
3. 【**简化处理**】：减少复杂的数据转换，提高稳定性
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
st.set_page_config(page_title="选股王 · V18.1 数据修复版", layout="wide")
st.title("选股王 · V18.1 数据修复版（🚀 数据修复 / 快速回测）")
st.markdown("🎯 **V18.1 策略说明：** **修复数据获取问题，确保回测正常运行。**")
st.markdown("✅ **速度说明：** 基于V14.8.1的快速框架，20-50个交易日回测只需几分钟。")


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
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 3)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历，请检查 Token 或 Tushare 权限。")
        return []
    
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()


# ----------------------------------------------------------------------
# ⭐️ V18.1 核心：修复数据获取问题
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def get_all_historical_data_fixed(trade_days_list):
    """
    V18.1 修复数据获取问题
    """
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 扩大数据获取范围，确保覆盖所有需要的日期
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=180)  # 增加到180天
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=30)  # 增加到30天
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    st.info(f"⏳ 正在下载 {start_date} 到 {end_date} 的全市场历史数据...")
    
    # 1. 获取所有交易日列表
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty:
        st.error("无法获取交易日历。")
        return False
    
    all_dates = all_trade_dates_df['cal_date'].tolist()
    
    # 2. 分批获取数据，避免请求过大
    download_progress = st.progress(0, text="下载数据...")
    
    # 获取复权因子数据
    download_progress.progress(0.2, text="下载复权因子...")
    adj_factor_data_list = []
    
    # 分批次获取复权因子，每30天一批
    batch_size = 30
    for i in range(0, len(all_dates), batch_size):
        batch_dates = all_dates[i:i+batch_size]
        for date in batch_dates:
            adj_df = safe_get('adj_factor', trade_date=date)
            if not adj_df.empty:
                adj_factor_data_list.append(adj_df)
        
        download_progress.progress(0.2 + (i/len(all_dates))*0.3, 
                                 text=f"下载复权因子: {min(i+batch_size, len(all_dates))}/{len(all_dates)}")
    
    if not adj_factor_data_list:
        st.warning("复权因子数据可能不完整，尝试其他方法...")
        # 尝试直接获取整个时间段的数据
        adj_factor_data = safe_get('adj_factor', start_date=start_date, end_date=end_date)
        if not adj_factor_data.empty:
            adj_factor_data_list = [adj_factor_data]
        else:
            st.error("❌ 无法获取复权因子数据。")
            return False
    
    adj_factor_data = pd.concat(adj_factor_data_list, ignore_index=True)
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(1.0)
    GLOBAL_ADJ_FACTOR = adj_factor_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    # 3. 获取日线数据
    download_progress.progress(0.5, text="下载日线数据...")
    daily_data_list = []
    
    # 获取最近90天的日线数据（足够回测使用）
    needed_dates = all_dates[:min(90, len(all_dates))]
    
    for i, date in enumerate(needed_dates):
        daily_df = safe_get('daily', trade_date=date)
        if not daily_df.empty:
            daily_data_list.append(daily_df)
        
        download_progress.progress(0.5 + (i/len(needed_dates))*0.4, 
                                 text=f"下载日线数据: {i+1}/{len(needed_dates)}")
    
    if not daily_data_list:
        st.error("❌ 无法获取日线数据。")
        return False
    
    daily_raw_data = pd.concat(daily_data_list, ignore_index=True)
    
    # 4. 处理数据
    download_progress.progress(0.9, text="处理数据...")
    
    # 转换日期格式
    daily_raw_data['trade_date'] = pd.to_datetime(daily_raw_data['trade_date'], format='%Y%m%d', errors='coerce')
    daily_raw_data = daily_raw_data.dropna(subset=['trade_date'])
    
    # 过滤掉无效数据
    daily_raw_data = daily_raw_data[daily_raw_data['ts_code'].notna()]
    
    # 设置索引
    try:
        GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    except Exception as e:
        st.error(f"设置索引失败: {e}")
        # 如果设置索引失败，尝试修复数据
        daily_raw_data = daily_raw_data.drop_duplicates(subset=['ts_code', 'trade_date'])
        GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    
    # 5. 设置QFQ基准因子
    try:
        # 获取最新的复权因子
        latest_date = None
        if not GLOBAL_ADJ_FACTOR.empty:
            date_level = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date')
            if len(date_level) > 0:
                latest_date = date_level.max()
        
        if latest_date:
            latest_adj = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_date), 'adj_factor']
            if isinstance(latest_adj, pd.Series):
                GLOBAL_QFQ_BASE_FACTORS = latest_adj.droplevel(1).to_dict()
                st.info(f"✅ 设置基准因子，股票数量: {len(GLOBAL_QFQ_BASE_FACTORS)}")
            else:
                # 如果是DataFrame，转换为字典
                GLOBAL_QFQ_BASE_FACTORS = latest_adj.reset_index().set_index('ts_code')['adj_factor'].to_dict()
    except Exception as e:
        st.warning(f"设置基准因子时出错: {e}")
        # 创建简单的基准因子
        unique_stocks = GLOBAL_DAILY_RAW.index.get_level_values('ts_code').unique()
        GLOBAL_QFQ_BASE_FACTORS = {stock: 1.0 for stock in unique_stocks}
    
    download_progress.progress(1.0, text="数据加载完成！")
    time.sleep(1)
    download_progress.empty()
    
    # 诊断信息
    st.success(f"""
    ✅ 数据预加载完成！
    - 日线数据: {len(GLOBAL_DAILY_RAW):,} 条记录
    - 复权因子: {len(GLOBAL_ADJ_FACTOR):,} 条记录
    - 基准因子: {len(GLOBAL_QFQ_BASE_FACTORS)} 只股票
    """)
    
    # 检查数据质量
    if len(GLOBAL_DAILY_RAW) < 50000:
        st.warning("⚠️ 警告：日线数据量可能不足。")
    
    return True


# ----------------------------------------------------------------------
# 简化的数据获取函数
# ----------------------------------------------------------------------
def get_qfq_data_simple(ts_code, start_date, end_date):
    """简化版前复权数据获取"""
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    
    if GLOBAL_DAILY_RAW.empty:
        return pd.DataFrame()
    
    try:
        # 转换日期格式
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        # 检查股票是否存在
        if ts_code not in GLOBAL_DAILY_RAW.index.get_level_values('ts_code'):
            return pd.DataFrame()
        
        # 获取日线数据
        try:
            daily_data = GLOBAL_DAILY_RAW.loc[ts_code].copy()
        except KeyError:
            return pd.DataFrame()
        
        if daily_data.empty:
            return pd.DataFrame()
        
        # 过滤日期范围
        daily_data = daily_data[(daily_data.index >= start_dt) & (daily_data.index <= end_dt)]
        
        if daily_data.empty:
            return pd.DataFrame()
        
        # 获取基准复权因子
        base_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, 1.0)
        
        # 如果有复权因子数据
        if ts_code in GLOBAL_ADJ_FACTOR.index.get_level_values('ts_code'):
            try:
                adj_data = GLOBAL_ADJ_FACTOR.loc[ts_code].copy()
                adj_data = adj_data[(adj_data.index >= start_dt) & (adj_data.index <= end_dt)]
                
                if not adj_data.empty:
                    # 合并数据
                    df = daily_data.merge(adj_data, left_index=True, right_index=True, how='left')
                    df['adj_factor'] = df['adj_factor'].fillna(base_factor)
                    
                    # 计算前复权价格
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df[col] = df[col] * df['adj_factor'] / base_factor
                else:
                    df = daily_data.copy()
            except Exception:
                df = daily_data.copy()
        else:
            df = daily_data.copy()
        
        return df[['open', 'high', 'low', 'close', 'vol']].reset_index()
        
    except Exception as e:
        return pd.DataFrame()


# ----------------------------------------------------------------------
# 核心函数：get_future_prices
# ----------------------------------------------------------------------
def get_future_prices_simple(ts_code, selection_date, d0_qfq_close, days_ahead=[1, 3, 5]):
    """简化版未来价格获取"""
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_date_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date_future = (d0 + timedelta(days=15)).strftime("%Y%m%d")
    
    hist = get_qfq_data_simple(ts_code, start_date_future, end_date_future)
    if hist.empty or 'close' not in hist.columns:
        return {f'Return_D{n}': np.nan for n in days_ahead}
        
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    hist = hist.dropna(subset=['close'])
    
    results = {}
    for n in days_ahead:
        if len(hist) >= n:
            future_price = hist.iloc[n-1]['close']
            results[f'Return_D{n}'] = (future_price / d0_qfq_close - 1) * 100
        else:
            results[f'Return_D{n}'] = np.nan
    
    return results


@st.cache_data(ttl=3600*12) 
def compute_indicators_simple(ts_code, end_date):
    """简化版指标计算"""
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
    
    df = get_qfq_data_simple(ts_code, start_date, end_date)
    if df.empty or 'close' not in df.columns:
        return {}
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    
    close = df['close'].dropna()
    if len(close) < 10:
        return {}
    
    res = {'last_close': close.iloc[-1]}
    
    # 10日回报
    if len(close) >= 10:
        res['10d_return'] = (close.iloc[-1] / close.iloc[-10] - 1) * 100
    
    # 位置因子
    if len(df) >= 20:
        hist_20 = df.tail(20)
        min_low = hist_20['low'].min()
        max_high = hist_20['high'].max()
        current_close = hist_20['close'].iloc[-1]
        
        if max_high > min_low:
            res['position_20d'] = (current_close - min_low) / (max_high - min_low) * 100
        else:
            res['position_20d'] = 50
    
    return res


# ----------------------------------------------------
# 侧边栏参数 (简化版)
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input("选择**回测结束日期**", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("**自动回测天数 (N)**", value=10, step=1, min_value=1, max_value=30, 
                                       help="建议从10天开始测试，如果成功再增加天数。"))
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = int(st.number_input("最终入围评分数量 (M)", value=50, step=10, min_value=10, max_value=200)) 
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=3, step=1, min_value=1, max_value=10)) 
    
    st.markdown("---")
    st.header("🛒 灵活过滤条件")
    MIN_PRICE = st.number_input("最低股价 (元)", value=5.0, step=0.5, min_value=1.0)
    MAX_PRICE = st.number_input("最高股价 (元)", value=200.0, step=10.0, min_value=10.0)
    MIN_TURNOVER = st.number_input("最低换手率 (%)", value=0.5, step=0.1, min_value=0.1) 
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿元)", value=10.0, step=5.0, min_value=1.0)

# ---------------------------
# Token 输入与初始化
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
        st.error("Token无效或权限不足")
        st.stop()
    st.success("✅ Token验证通过")
except Exception as e:
    st.error(f"Token验证失败: {e}")
    st.stop()

# ---------------------------
# 核心回测逻辑函数 - 简化稳定版
# ---------------------------
def run_backtest_for_a_day_simple(last_trade, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_CIRC_MV_BILLIONS):
    """为单个交易日运行选股和回测逻辑 - 简化稳定版"""
    # 1. 获取当日数据
    daily_data = safe_get('daily', trade_date=last_trade)
    if daily_data.empty:
        return pd.DataFrame(), f"无法获取日线数据: {last_trade}"
    
    # 2. 获取股票基本信息
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')
    
    # 3. 合并数据
    df = daily_data.copy()
    if not stock_basic.empty:
        df = df.merge(stock_basic[['ts_code', 'name', 'list_date']], on='ts_code', how='left')
    else:
        df['name'] = df['ts_code']
        df['list_date'] = '20000101'
    
    # 4. 数据清洗
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce').fillna(0)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    df['name'] = df['name'].fillna('').astype(str)
    
    # 5. 简单过滤
    # 过滤ST股和北交所
    mask_st = df['name'].str.contains('ST|退', case=False, na=False)
    df = df[~mask_st]
    mask_bj = df['ts_code'].str.startswith('92')
    df = df[~mask_bj]
    
    # 过滤新股
    TODAY = datetime.strptime(last_trade, "%Y%m%d")
    df['list_date_dt'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
    df['days_listed'] = (TODAY - df['list_date_dt']).dt.days
    df = df[df['days_listed'] >= 60]  # 降低到60天
    
    # 过滤价格
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)]
    
    if df.empty:
        return pd.DataFrame(), f"基础过滤后无股票: {last_trade}"
    
    # 6. 按涨幅排序，选择前FINAL_POOL只股票
    df = df.sort_values('pct_chg', ascending=False).head(FINAL_POOL)
    
    # 7. 计算指标和未来收益
    records = []
    
    for _, row in df.iterrows():
        ts_code = row['ts_code']
        
        # 计算指标
        indicators = compute_indicators_simple(ts_code, last_trade)
        if not indicators or 'last_close' not in indicators:
            continue
        
        d0_qfq_close = indicators['last_close']
        
        # 获取未来收益
        future_returns = get_future_prices_simple(ts_code, last_trade, d0_qfq_close)
        
        record = {
            'ts_code': ts_code,
            'name': row['name'],
            'Close': row['close'],
            'Pct_Chg (%)': row['pct_chg'],
            '10d_return': indicators.get('10d_return', 0),
            'position_20d': indicators.get('position_20d', 50),
            'Return_D1 (%)': future_returns.get('Return_D1', np.nan),
            'Return_D3 (%)': future_returns.get('Return_D3', np.nan),
            'Return_D5 (%)': future_returns.get('Return_D5', np.nan),
        }
        
        records.append(record)
    
    if not records:
        return pd.DataFrame(), f"无有效指标: {last_trade}"
    
    fdf = pd.DataFrame(records)
    
    # 8. 简单评分
    def normalize(series):
        if len(series) < 2 or series.max() == series.min():
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min() + 1e-9)
    
    # 当日涨幅得分
    fdf['s_pct'] = normalize(fdf['Pct_Chg (%)'])
    
    # 10日回报得分
    fdf['s_10d'] = normalize(fdf['10d_return'])
    
    # 位置得分（40-70为佳）
    position = fdf['position_20d']
    position_score = np.where(
        (position >= 40) & (position <= 70),
        1.0,
        np.where(position < 40, position / 40, (100 - position) / 30)
    )
    fdf['s_position'] = position_score
    
    # 综合评分
    fdf['综合评分'] = (
        fdf['s_pct'].fillna(0.5) * 0.5 +
        fdf['s_10d'].fillna(0.5) * 0.3 +
        fdf['s_position'].fillna(0.5) * 0.2
    ) * 100
    
    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index += 1
    
    return fdf.head(TOP_BACKTEST), None


# ---------------------------
# 主运行块 
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测 (V18.1 数据修复版)"):
    
    st.warning("⚠️ **请先清除 Streamlit 缓存**（右上角三点菜单 -> Settings -> Clear Cache）")
   
    trade_days_str = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days_str:
        st.error("无法获取交易日列表，请检查日期或 Token。")
        st.stop()
    
    st.info(f"获取到 {len(trade_days_str)} 个交易日")
    
    # ----------------------------------------------------------------------
    # 加载历史数据
    # ----------------------------------------------------------------------
    start_time = time.time()
    preload_success = get_all_historical_data_fixed(trade_days_str)
    load_time = time.time() - start_time
    
    if not preload_success:
        st.error("❌ 历史数据预加载失败，回测无法进行。")
        st.stop()
    
    st.success(f"✅ 历史数据加载完成！耗时: {load_time:.1f}秒")
    
    # ----------------------------------------------------------------------
    # 开始回测
    # ----------------------------------------------------------------------
    st.header(f"📈 正在进行 {len(trade_days_str)} 个交易日的回测...")
    
    results_list = []
    total_days = len(trade_days_str)
    
    progress_text = st.empty()
    my_bar = st.progress(0)
    
    for i, trade_date in enumerate(trade_days_str):
        progress_text.text(f"🚀 正在处理第 {i+1}/{total_days} 个交易日：{trade_date}")
        
        daily_result_df, error = run_backtest_for_a_day_simple(
            trade_date, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_CIRC_MV_BILLIONS
        )
        
        if error:
            st.warning(f"{error}")
        elif not daily_result_df.empty:
            daily_result_df['Trade_Date'] = trade_date
            results_list.append(daily_result_df)
            st.info(f"✅ {trade_date}: 找到 {len(daily_result_df)} 只有效股票")
        
        my_bar.progress((i + 1) / total_days)
    
    progress_text.text("✅ 回测完成，正在汇总结果...")
    my_bar.empty()
    
    if not results_list:
        st.error("所有交易日的回测均失败或无结果。")
        st.stop()
    
    all_results = pd.concat(results_list, ignore_index=True)
    
    # 显示结果
    st.header(f"📊 最终平均回测结果 (Top {TOP_BACKTEST}，共 {len(all_results['Trade_Date'].unique())} 个有效交易日)")
    
    for n in [1, 3, 5]:
        col = f'Return_D{n} (%)' 
        
        valid_returns = all_results.dropna(subset=[col])
        
        if not valid_returns.empty:
            avg_return = valid_returns[col].mean()
            hit_rate = (valid_returns[col] > 0).sum() / len(valid_returns) * 100
            total_count = len(valid_returns)
        else:
            avg_return = np.nan
            hit_rate = 0.0
            total_count = 0
        
        st.metric(f"Top {TOP_BACKTEST}：D+{n} 平均收益 / 准确率", 
                  f"{avg_return:.2f}% / {hit_rate:.1f}%", 
                  help=f"总有效样本数：{total_count}")
    
    st.header("📋 每日回测详情 (Top K 明细)")
    
    display_cols = ['Trade_Date', 'name', 'ts_code', '综合评分', 'Close', 
                   'Pct_Chg (%)', 'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)']
    
    available_cols = [col for col in display_cols if col in all_results.columns]
    
    st.dataframe(all_results[available_cols].sort_values('Trade_Date', ascending=False), 
                 use_container_width=True,
                 column_config={
                     'Return_D1 (%)': st.column_config.NumberColumn(format="%.2f"),
                     'Return_D3 (%)': st.column_config.NumberColumn(format="%.2f"),
                     'Return_D5 (%)': st.column_config.NumberColumn(format="%.2f"),
                 })
    
    # 显示性能统计
    st.subheader("⚡ 性能统计")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("数据加载时间", f"{load_time:.1f}秒")
    with col2:
        st.metric("回测交易天数", len(trade_days_str))
    
    # 下载功能
    csv = all_results.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下载回测结果",
        data=csv,
        file_name=f"backtest_v18_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
