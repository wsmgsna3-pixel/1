# -*- coding: utf-8 -*-
"""
选股王 · V13.1 极致动量版：V13.0 框架 + 最终策略优化
核心修复：QFQ 基准修复已完成 (V12.9)。
策略优化：核心变动：
1. 极致动量：提升当日涨幅 (w_pct) 至 0.20。
2. 强化趋势：提升 MACD (w_macd) 至 0.28。
3. 持续减负：削弱资金流 (w_mf) 至 0.17。
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
st.set_page_config(page_title="选股王 · V13.1 极致动量版", layout="wide")
st.title("选股王 · V13.1 最终冲刺策略（🚀 极致动量 + 强趋势）")
st.markdown("🎯 **V13.1 策略说明：** 进一步加强短期动量和中期趋势权重，削弱资金流依赖，目标是将 D+1 平均收益率转正。")
st.markdown("✅ **技术说明：** 底层 QFQ 基准已固定（V12.9 修复），回测速度极快且结果稳定。")


# ---------------------------
# 辅助函数 (API调用和数据获取)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    """安全调用 Tushare API，已移除 time.sleep(0.5)"""
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
# ⭐️ 核心优化：预加载所有历史数据 (日线和复权因子)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def get_all_historical_data(trade_days_list):
    """
    一次性获取所有回测日所需的最大历史范围内的日线数据和复权因子。
    并计算 GLOBAL_QFQ_BASE_FACTORS。
    """
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) # 以回测结束日作为复权基准日
    earliest_trade_date = min(trade_days_list)
    
    # 扩大数据获取范围，确保能覆盖回测和未来收益计算
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=20)
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    st.info(f"💾 正在批量下载 {start_date} 到 {end_date} 间的**全市场历史数据**...")
    
    # 1. 获取所有交易日列表
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty:
        st.error("无法获取交易日历。")
        return False
    
    all_dates = all_trade_dates_df['cal_date'].tolist()
    if not all_dates:
        st.error("交易日列表为空。")
        return False

    # 2. 批量获取复权因子 (adj_factor) - 按日期循环获取
    adj_factor_data_list = []
    adj_download_progress = st.progress(0, text="下载进度 (复权因子)...")
    
    for i, date in enumerate(all_dates):
        adj_df = safe_get('adj_factor', trade_date=date) 
        if not adj_df.empty:
            adj_factor_data_list.append(adj_df)
        adj_download_progress.progress((i + 1) / len(all_dates), text=f"下载进度 (复权因子)：正在处理 {date} ({i+1}/{len(all_dates)})")
    
    adj_download_progress.empty()
    
    if not adj_factor_data_list:
        st.error("❌ 严重错误：无法获取任何复权因子数据。")
        return False
        
    adj_factor_data = pd.concat(adj_factor_data_list, ignore_index=True)
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(0)
    adj_factor_data = adj_factor_data.set_index(['ts_code', 'trade_date'])
    GLOBAL_ADJ_FACTOR = adj_factor_data.sort_index(level=[0, 1]) 
    
    # ⭐️ 核心修复步骤 1：计算并存储全局固定 QFQ 基准因子
    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    
    if latest_global_date:
        try:
            latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
            st.info(f"✅ 全局 QFQ 基准因子已设置。基准日期: {latest_global_date}，股票数量: {len(GLOBAL_QFQ_BASE_FACTORS)}")
        except Exception as e:
            st.error(f"无法设置全局 QFQ 基准因子: {e}")
            GLOBAL_QFQ_BASE_FACTORS = {}
    
    
    # 3. 批量获取日线行情 (daily) - 按日期循环获取
    daily_data_list = []
    download_progress = st.progress(0, text="下载进度 (日线行情)...")
    
    for i, date in enumerate(all_dates):
        daily_df = safe_get('daily', trade_date=date)
        if not daily_df.empty:
            daily_data_list.append(daily_df)
        download_progress.progress((i + 1) / len(all_dates), text=f"下载进度 (日线行情)：正在处理 {date} ({i+1}/{len(all_dates)})")
    
    download_progress.empty()
    
    if not daily_data_list:
        st.error("❌ 严重错误：无法获取任何历史日线数据。")
        return False

    daily_raw_data = pd.concat(daily_data_list, ignore_index=True)
    daily_raw_data = daily_raw_data.set_index(['ts_code', 'trade_date'])
    GLOBAL_DAILY_RAW = daily_raw_data.sort_index(level=[0, 1])

    # 4. 诊断信息
    st.info(f"✅ 数据预加载完成。日线数据总条目：{len(GLOBAL_DAILY_RAW)}，复权因子总条目：{len(GLOBAL_ADJ_FACTOR)}")

    return True


# ----------------------------------------------------------------------
# ⭐️ 优化的数据获取函数：只从内存中切片 (前复权计算核心)
# ----------------------------------------------------------------------
def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    """ 
    V12.9 修复版：日线数据和复权因子均从预加载的全局变量中切片获取，
    复权基准使用 GLOBAL_QFQ_BASE_FACTORS 中存储的统一因子。
    """
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty or not GLOBAL_QFQ_BASE_FACTORS:
        return pd.DataFrame()
        
    # ⭐️ 核心修复步骤 2：获取全局固定复权基准
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
    
    # ⭐️ 核心修复步骤 3：使用全局固定基准进行向量化复权计算
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
# ⭐️ 核心函数：get_future_prices (接受 D0 QFQ 价格)
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


@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    """计算 MACD, 10日回报, 波动率, 60日位置等指标 (使用优化版数据获取)"""
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    
    # 获取 QFQ 数据，用于计算所有指标
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    
    res = {}
    if df.empty or 'close' not in df.columns: 
        return res
        
    df['close'] = pd.to_numeric(df['close'], errors='coerce').astype(float)
    df['low'] = pd.to_numeric(df['low'], errors='coerce').astype(float)
    df['high'] = pd.to_numeric(df['high'], errors='coerce').astype(float)
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce').fillna(0)
    
    if len(df) >= 2:
         df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    else:
         df['pct_chg'] = 0.0
         
    close = df['close']
    
    res['last_close'] = close.iloc[-1] if len(close) > 0 else np.nan
    
    # MACD, 量比, 10日回报, 波动率, 60日位置计算（保持不变）
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    else: res['macd_val'] = np.nan
        
    vols = df['vol'].tolist()
    if len(vols) >= 6 and vols[-6:-1] and np.mean(vols[-6:-1]) > 1e-9:
        res['vol_ratio'] = vols[-1] / np.mean(vols[-6:-1])
    else: res['vol_ratio'] = np.nan
        
    res['10d_return'] = close.iloc[-1]/close.iloc[-10] - 1 if len(close)>=10 and close.iloc[-10]!=0 else 0
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    if len(df) >= 60:
        hist_60 = df.tail(60)
        min_low = hist_60['low'].min()
        max_high = hist_60['high'].max()
        current_close = hist_60['close'].iloc[-1]
        
        if max_high == min_low: res['position_60d'] = 50.0 
        else: res['position_60d'] = (current_close - min_low) / (max_high - min_low) * 100
    else: res['position_60d'] = np.nan 
    
    return res

# ----------------------------------------------------
# 侧边栏参数 (保持不变)
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input("选择**回测结束日期**", value=datetime.now().date(), max_value=datetime.now().date())
    BACKTEST_DAYS = int(st.number_input("**自动回测天数 (N)**", value=20, step=1, min_value=1, max_value=50, help="程序将自动回测最近 N 个交易日。建议设置为 20 天以获得更可靠的统计数据。"))
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = int(st.number_input("最终入围评分数量 (M)", value=10, step=1, min_value=1)) 
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=10, step=1))
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=3, step=1, min_value=1)) 
    
    st.markdown("---")
    st.header("🛒 灵活过滤条件")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0, step=0.5, min_value=0.1)
    MAX_PRICE = st.number_input("最高股价 (元)", value=300.0, step=5.0, min_value=1.0)
    MIN_TURNOVER = st.number_input("最低换手率 (%)", value=2.0, step=0.5, min_value=0.1) 
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿元)", value=20.0, step=1.0, min_value=1.0, help="例如：输入 20 代表流通市值必须大于等于 20 亿元。")
    MIN_AMOUNT_MILLIONS = st.number_input("最低成交额 (亿元)", value=0.6, step=0.1, min_value=0.1)
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
# 核心回测逻辑函数 (run_backtest_for_a_day)
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS):
    """为单个交易日运行选股和回测逻辑"""
    global GLOBAL_DAILY_RAW
    
    # 1. 拉取全市场 Daily 数据 (今日快照)
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty or 'ts_code' not in daily_all.columns: return pd.DataFrame(), f"数据缺失或拉取失败：{last_trade}"

    pool_raw = daily_all.reset_index(drop=True) 
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date') 
    REQUIRED_BASIC_COLS = ['ts_code','turnover_rate','amount','total_mv','circ_mv'] 
    daily_basic = safe_get('daily_basic', trade_date=last_trade, fields=','.join(REQUIRED_BASIC_COLS))
    mf_raw = safe_get('moneyflow', trade_date=last_trade)
    pool_merged = pool_raw.copy()

    # 数据合并 (保持不变)
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
    pool_merged['turnover_rate'] = pool_merged['turnover_rate'].fillna(0) 
   
    # 3. 执行硬性条件过滤
    df = pool_merged.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce') 
    df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000 # 转换为万元
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000 
    df['name'] = df['name'].astype(str)
    
    # 过滤 ST 股/退市股/北交所/次新股
    mask_st = df['name'].str.contains('ST|退', case=False, na=False)
    df = df[~mask_st]
    mask_bj = df['ts_code'].str.startswith('92') 
    df = df[~mask_bj]
    TODAY = datetime.strptime(last_trade, "%Y%m%d")
    MIN_LIST_DAYS = 120 
    df['list_date_dt'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
    df['days_listed'] = (TODAY - df['list_date_dt']).dt.days
    mask_cyb_kcb = df['ts_code'].str.startswith(('30','68'))
    mask_new = df['days_listed'] < MIN_LIST_DAYS
    df = df[~((mask_cyb_kcb) & (mask_new))]

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

    if len(df) == 0: return pd.DataFrame(), f"硬性过滤后无股票：{last_trade}"

    # 4. 遴选决赛名单
    limit_pct = int(FINAL_POOL * 0.7)
    df_pct = df.sort_values('pct_chg', ascending=False).head(limit_pct).copy()
    limit_turn = FINAL_POOL - len(df_pct)
    existing_codes = set(df_pct['ts_code'])
    df_turn = df[~df['ts_code'].isin(existing_codes)].sort_values('turnover_rate', ascending=False).head(limit_turn).copy()
    final_candidates = pd.concat([df_pct, df_turn]).reset_index(drop=True)

    # 4.5 数据完整性前置校验 
    if not GLOBAL_DAILY_RAW.empty:
        codes_in_global = GLOBAL_DAILY_RAW.index.get_level_values('ts_code').unique()
        final_candidates = final_candidates[final_candidates['ts_code'].isin(codes_in_global)].copy()

    if final_candidates.empty:
        return pd.DataFrame(), f"候选股票池中所有股票在缓存中均缺失历史数据：{last_trade}"

    # 5. 深度评分 
    records = []
    
    for row in final_candidates.itertuples():
        ts_code = row.ts_code
        
        raw_close = getattr(row, 'close', np.nan)
        
        # 计算指标
        ind = compute_indicators(ts_code, last_trade) 
        d0_qfq_close = ind.get('last_close', np.nan) # 提取 D0 QFQ Close Price

        # 仅当 D0 QFQ Close Price 有效且非零时，才进行收益率计算和记录
        if pd.notna(d0_qfq_close) and d0_qfq_close > 1e-9:
            
            future_returns = get_future_prices(ts_code, last_trade, d0_qfq_close) 
            
            rec = {
                'ts_code': ts_code, 'name': getattr(row, 'name', ts_code),
                'Close': raw_close, 
                'Circ_MV (亿)': getattr(row, 'circ_mv_billion', np.nan),
                'Pct_Chg (%)': getattr(row, 'pct_chg', 0), 
                'turnover': getattr(row, 'turnover_rate', 0),
                'net_mf': getattr(row, 'net_mf', 0)
            }
            
            rec.update({
                'vol_ratio': ind.get('vol_ratio', np.nan), 'macd': ind.get('macd_val', np.nan),
                '10d_return': ind.get('10d_return', np.nan),
                'volatility': ind.get('volatility', np.nan), 'position_60d': ind.get('position_60d', np.nan)
            })
            
            rec.update({
                'Return_D1 (%)': future_returns.get('Return_D1', np.nan),
                'Return_D3 (%)': future_returns.get('Return_D3', np.nan),
                'Return_D5 (%)': future_returns.get('Return_D5', np.nan),
            })
            
            records.append(rec)
    
    fdf = pd.DataFrame(records)
    
    if fdf.empty: 
        error_msg = f"评分列表为空：{last_trade}. 原因：所有 {len(final_candidates)} 个候选股票都无法获取有效的 D0 QFQ 价格（可能数据不足3天或复权失败）。"
        return pd.DataFrame(), error_msg

    # 6. 归一化与 V13.1 策略精调评分 
    def normalize(series):
        series_nn = series.dropna() 
        if series_nn.empty or series_nn.max() == series_nn.min(): return pd.Series([0.5] * len(series), index=series.index)
        return (series - series_nn.min()) / (series_nn.max() - series_nn.min() + 1e-9)

    fdf['s_pct'] = normalize(fdf['Pct_Chg (%)'])
    fdf['s_turn'] = normalize(fdf['turnover'])
    fdf['s_vol'] = normalize(fdf['vol_ratio'])
    fdf['s_mf'] = normalize(fdf['net_mf'])
    fdf['s_macd'] = normalize(fdf['macd'])
    fdf['s_trend'] = normalize(fdf['10d_return'])
    fdf['s_volatility'] = normalize(fdf['volatility'])
    fdf['s_position'] = fdf['position_60d'] / 100 
    
    # 🚨 V13.1 策略权重 (极致动量 + 强趋势)
    w_mf = 0.17           # 资金流 ↓↓
    w_pct = 0.20          # 当日涨幅 ↑↑
    w_turn = 0.15         # 换手率 ↑
    w_position = 0.10     # 60日位置 ↓ (反向)
    w_volatility = 0.15   # 波动率 ↑ (反向)
    w_macd = 0.28         # MACD ↑↑
    w_vol = 0.00          
    w_trend = 0.00          
    
    score = (
        fdf['s_pct'].fillna(0.5) * w_pct + fdf['s_turn'].fillna(0.5) * w_turn + 
        fdf['s_mf'].fillna(0.5) * w_mf + 
        fdf['s_macd'].fillna(0.5) * w_macd + 
        
        (1 - fdf['s_position'].fillna(0.5)) * w_position + # 倾向低位，但权重低
        (1 - fdf['s_volatility'].fillna(0.5)) * w_volatility + # 倾向低波动
        
        fdf['s_vol'].fillna(0.5) * w_vol + 
        fdf['s_trend'].fillna(0.5) * w_trend     
    )
    fdf['综合评分'] = score * 100
    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index += 1

    return fdf.head(TOP_BACKTEST).copy(), None

# ---------------------------
# 主运行块 
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测"):
    
    st.warning("⚠️ **请务必先清除 Streamlit 缓存！**（右上角三点菜单 -> Settings -> Clear Cache）这是让程序重新加载数据的关键一步。")
   
    trade_days_str = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days_str:
        st.error("无法获取交易日列表，请检查日期或 Token。")
        st.stop()
    
    # ----------------------------------------------------------------------
    # ⭐️ 核心优化步骤：预加载所有历史数据 (V12.9 基准修复)
    # ----------------------------------------------------------------------
    preload_success = get_all_historical_data(trade_days_str)
    if not preload_success:
        st.error("❌ 历史数据预加载失败，回测无法进行。请检查 Tushare Token 和权限。")
        st.stop()
    st.success("✅ 历史数据预加载完成！QFQ 基准已固定。现在开始极速回测...")
    # ----------------------------------------------------------------------
    
    st.header(f"📈 正在进行 {BACKTEST_DAYS} 个交易日的回测...")
    
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
            st.warning(f"跳过 {trade_date}：{error}")
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
    
    for n in [1, 3, 5]:
        col = f'Return_D{n} (%)' 
        
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
            
        st.metric(f"Top {TOP_BACKTEST}：D+{n} 平均收益 / 准确率", 
                  f"{avg_return:.2f}% / {hit_rate:.1f}%", 
                  help=f"总有效样本数：{total_count}。**V13.1 极致动量版**")

    st.header("📋 每日回测详情 (Top K 明细)")
    
    display_cols = ['Trade_Date', 'name', 'ts_code', '综合评分', 
                    'Close', 'Pct_Chg (%)', 'Circ_MV (亿)',
                    'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)']
    
    st.dataframe(all_results[display_cols].sort_values('Trade_Date', ascending=False), use_container_width=True)
