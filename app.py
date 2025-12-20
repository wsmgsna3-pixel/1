# -*- coding: utf-8 -*-
"""
选股王 · V30.6 强弱市自适应策略 (右侧实战模拟版)
V30.6 更新内容：
1. [实战模拟] 引入“右侧买入阈值”机制，模拟 9:40 确认上涨后买入。
   - 只有 D1 最高价 > 开盘价 * (1 + 阈值) 才成交，否则记为空仓。
2. [数据优化] 保持 V30.5 的内存优化方案，支持长周期回测。
3. [参数增强] 侧边栏可调整买入阈值。
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


# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V30.6 右侧实战版", layout="wide")
st.title("选股王 · V30.6 强弱市自适应策略（🏹 右侧确认买入 / ⚡ 内存优化）")
st.markdown("🎯 **V30.6 核心逻辑：** 模拟实战操作，只有在次日盘中涨幅达到设定阈值（如 +1.5%）时才买入，过滤掉开盘即下跌的无效交易。")


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
# 缓存与数据拉取
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    """安全拉取并缓存单个交易日的数据"""
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(trade_days_list):
    """
    预加载数据：保留核心列并压缩类型 (float32) 以节省内存
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
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty:
        st.error("无法获取交易日历。")
        return False
    
    all_dates = all_trade_dates_df['cal_date'].tolist()
    st.info(f"⏳ 正在按日期循环下载 {start_date} 到 {end_date} 间的全市场数据...")

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
        except Exception:
            continue 
            
    download_progress.progress(1.0, text="下载进度：合并数据...")
    download_progress.empty()

    if not adj_factor_data_list or not daily_data_list:
        st.error("❌ 严重错误：无法获取历史数据。")
        return False
        
    adj_factor_data = pd.concat(adj_factor_data_list)
    adj_factor_data['adj_factor'] = pd.to_numeric(adj_factor_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_factor_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    # [V30.5/6] 内存优化：只保留核心列
    cols_to_keep = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
    valid_cols = [c for c in cols_to_keep if c in daily_data_list[0].columns]
    daily_raw_data = pd.concat(daily_data_list)[valid_cols]
    
    # [V30.5/6] 内存优化：强制转换类型为 float32
    float_cols = ['open', 'high', 'low', 'close', 'vol']
    for col in float_cols:
        if col in daily_raw_data.columns:
            daily_raw_data[col] = pd.to_numeric(daily_raw_data[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])

    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_global_date:
        try:
            latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
            GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
        except Exception:
            GLOBAL_QFQ_BASE_FACTORS = {}
    
    return True


# ----------------------------------------------------------------------
# 数据切片函数
# ----------------------------------------------------------------------
def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
  
    if GLOBAL_DAILY_RAW.empty or GLOBAL_ADJ_FACTOR.empty: return pd.DataFrame()
        
    latest_adj_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj_factor) or latest_adj_factor < 1e-9: return pd.DataFrame() 

    try:
        daily_df_full = GLOBAL_DAILY_RAW.loc[ts_code]
        daily_df = daily_df_full.loc[(daily_df_full.index >= start_date) & (daily_df_full.index <= end_date)]
        adj_factor_series_full = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        adj_factor_series = adj_factor_series_full.loc[(adj_factor_series_full.index >= start_date) & (adj_factor_series_full.index <= end_date)]
    except KeyError:
        return pd.DataFrame()
    
    if daily_df.empty or adj_factor_series.empty: return pd.DataFrame()
            
    df = daily_df.merge(adj_factor_series.rename('adj_factor'), left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    
    # 复权计算
    df = df.sort_index()
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df['trade_date'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')
    df = df.sort_values('trade_date').set_index('trade_date_str')
    
    for col in ['open', 'high', 'low', 'close']:
        if col + '_qfq' in df.columns: df[col] = df[col + '_qfq']
            
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 

# ----------------------------------------------------------------------
# [V30.6 核心] 计算未来收益 (右侧交易模拟)
# ----------------------------------------------------------------------
def get_future_prices_right_side(ts_code, selection_date, days_ahead=[1, 3, 5], buy_threshold_pct=1.5):
    """
    模拟实战：只有当 D1 日内涨幅超过阈值 (buy_threshold_pct) 时才买入。
    """
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_date_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date_future = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    # 获取未来数据
    hist = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date_future, end_date=end_date_future)
    
    results = {}
    for n in days_ahead: results[f'Return_D{n}'] = np.nan

    if hist.empty or len(hist) < 1:
        return results
        
    # --- 右侧确认逻辑 ---
    d1_data = hist.iloc[0]
    d1_open = d1_data['open']
    d1_high = d1_data['high']
    
    # 设定买入价格：开盘价 * (1 + 阈值%)
    # 例如：开盘 10.0，阈值 1.5%，则必须涨到 10.15 才买入，买入价即为 10.15
    buy_price_threshold = d1_open * (1 + buy_threshold_pct / 100.0)
    
    if buy_price_threshold <= 1e-9: return results

    # [过滤]：如果当天最高价都没摸到买入价，说明全天弱势，未成交
    if d1_high < buy_price_threshold:
        return results # 返回 NaN，代表空仓/跳过

    # --- 成交，计算收益 ---
    for n in days_ahead:
        col_name = f'Return_D{n}'
        idx = n - 1
        if len(hist) > idx:
            sell_price = hist.iloc[idx]['close'] # 假设 N 天后收盘卖出
            # 收益率 = (卖出价 - 右侧确认买入价) / 右侧确认买入价
            results[col_name] = (sell_price / buy_price_threshold - 1) * 100
            
    return results

# ----------------------------------------------------------------------
# 指标计算
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*12) 
def compute_indicators(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    
    res = {}
    if df.empty or len(df) < 3 or 'close' not in df.columns: return res
        
    if len(df) >= 2:
       df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    else:
         df['pct_chg'] = 0.0
         
    close = df['close']
    res['last_close'] = close.iloc[-1] 
    
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    else: res['macd_val'] = np.nan
        
    if len(close) >= 20: res['ma20'] = close.tail(20).mean()
    else: res['ma20'] = np.nan
        
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

@st.cache_data(ttl=3600*12)
def get_market_state(trade_date):
    """判断市场状态"""
    start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    index_data = safe_get('daily', ts_code='000300.SH', start_date=start_date, end_date=trade_date, is_index=True)
    
    if index_data.empty or len(index_data) < 20: return 'Weak'

    index_data['close'] = pd.to_numeric(index_data['close'], errors='coerce').astype(float)
    index_data = index_data.sort_values('trade_date', ascending=True)

    latest_close = index_data.iloc[-1]['close']
    ma20 = index_data['close'].tail(20).mean()

    return 'Strong' if latest_close > ma20 else 'Weak'
      
        
# ----------------------------------------------------
# 侧边栏参数 
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测设置")
    backtest_date_end = st.date_input("回测结束日期", value=datetime.now().date(), max_value=datetime.now().date())
    
    BACKTEST_DAYS = int(st.number_input("**回测天数 (N)**", value=30, step=1, help="高积分用户建议设置 100-200 天"))
    
    st.markdown("---")
    st.header("2. 实战模拟设置 (V30.6)")
    st.info("💡 **右侧买入逻辑**：D1 开盘后，必须涨幅超过下方阈值才买入，否则空仓。")
    BUY_THRESHOLD_PCT = st.number_input(
        "**买入确认阈值 (%)**", 
        value=1.5, 
        step=0.1, 
        help="模拟 9:40 上涨确认。建议 1.0% - 2.0%。如果设置为 0 则代表开盘直接买。"
    )
    
    st.markdown("---")
    st.header("3. 策略权重 (弱市)")
    WEIGHT_MACD = st.slider("MACD 权重", 0.0, 1.0, 0.45)
    WEIGHT_VOL = st.slider("低波动权重", 0.0, 1.0, 0.45)
    
    st.markdown("---")
    st.header("4. 过滤条件")
    FINAL_POOL = int(st.number_input("入围数量", value=100)) 
    TOP_BACKTEST = int(st.number_input("每日持仓 Top K", value=5))
    MIN_PRICE = st.number_input("最低价", value=5.0) 
    MAX_PRICE = st.number_input("最高价", value=300.0)
    MIN_TURNOVER = st.number_input("最低换手 (%)", value=3.0) 
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿)", value=20.0)
    MIN_AMOUNT = st.number_input("最低成交额 (亿)", value=1.0) * 100000000 

# ---------------------------
# Token 输入
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Token。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ----------------------------------------------------------------------
# 核心回测逻辑函数 
# ----------------------------------------------------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, buy_threshold):
    """为单个交易日运行选股和回测逻辑"""
    global GLOBAL_DAILY_RAW
    
    market_state = get_market_state(last_trade)
 
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty: return pd.DataFrame(), f"数据缺失"

    pool_raw = daily_all.reset_index(drop=True) 
    stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date') 
    REQUIRED_BASIC_COLS = ['ts_code','turnover_rate','amount','total_mv','circ_mv'] 
    daily_basic = safe_get('daily_basic', trade_date=last_trade, fields=','.join(REQUIRED_BASIC_COLS))
    mf_raw = safe_get('moneyflow', trade_date=last_trade)
    
    pool_merged = pool_raw.copy()
    if not stock_basic.empty:
        pool_merged = pool_merged.merge(stock_basic[['ts_code','name','list_date']], on='ts_code', how='left')
    else:
        pool_merged['name'], pool_merged['list_date'] = pool_merged['ts_code'], '20000101'
        
    if not daily_basic.empty:
        cols = [c for c in REQUIRED_BASIC_COLS if c in daily_basic.columns]
        if 'amount' in pool_merged.columns and 'amount' in cols: pool_merged = pool_merged.drop(columns=['amount'])
        pool_merged = pool_merged.merge(daily_basic[cols], on='ts_code', how='left')
    
    for c in ['turnover_rate','amount','circ_mv','net_mf']: 
        if c not in pool_merged.columns: pool_merged[c] = 0.0
            
    if not mf_raw.empty:
        mf = mf_raw[['ts_code', 'net_mf']].fillna(0) if 'net_mf' in mf_raw.columns else pd.DataFrame()
        if not mf.empty: 
            pool_merged = pool_merged.drop(columns=['net_mf'], errors='ignore')
            pool_merged = pool_merged.merge(mf, on='ts_code', how='left')

    df = pool_merged.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce') 
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000 
    df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000 
    
    # 硬性过滤
    df = df[~df['name'].str.contains('ST|退', case=False, na=False)]
    df = df[~df['ts_code'].str.startswith('92')]
    df['days_listed'] = (datetime.strptime(last_trade, "%Y%m%d") - pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')).dt.days
    df = df[df['days_listed'] >= 120]
    
    df = df[(df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)]
    df = df[df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS]
    df = df[df['turnover_rate'] >= MIN_TURNOVER]
    df = df[df['amount'] >= MIN_AMOUNT]
    
    if len(df) == 0: return pd.DataFrame(), f"过滤后无股票"

    # 初选
    limit_mf = int(FINAL_POOL * 0.5)
    df_mf = df.sort_values('net_mf', ascending=False).head(limit_mf)
    df_pct = df[~df['ts_code'].isin(df_mf['ts_code'])].sort_values('pct_chg', ascending=False).head(FINAL_POOL - len(df_mf))
    final_candidates = pd.concat([df_mf, df_pct]).reset_index(drop=True)
    
    # 缓存数据检查
    if not GLOBAL_DAILY_RAW.empty:
        try:
            available = GLOBAL_DAILY_RAW.loc[(slice(None), last_trade), :].index.get_level_values('ts_code').unique()
            final_candidates = final_candidates[final_candidates['ts_code'].isin(available)]
        except: return pd.DataFrame(), f"缓存缺失"

    if final_candidates.empty: return pd.DataFrame(), f"筛选为空"

    # 深度评分
    records = []
    for row in final_candidates.itertuples():
        ts_code = row.ts_code
        ind = compute_indicators(ts_code, last_trade) 
        d0_close, d0_ma20, d0_pos = ind.get('last_close'), ind.get('ma20'), ind.get('position_60d')

        if market_state == 'Weak':
            if pd.isna(d0_ma20) or d0_close < d0_ma20 or d0_pos > 20.0: continue 

        if pd.notna(d0_close):
            # 核心：使用带阈值的右侧收益计算
            future_returns = get_future_prices_right_side(ts_code, last_trade, buy_threshold_pct=buy_threshold)
            
            rec = {
                'ts_code': ts_code, 'name': getattr(row, 'name', ts_code),
                'Close': row.close, 
                'Pct_Chg (%)': getattr(row, 'pct_chg', 0),
                'macd': ind.get('macd_val', np.nan), 
                'volatility': ind.get('volatility', np.nan),
                'Return_D1 (%)': future_returns.get('Return_D1', np.nan),
                'Return_D3 (%)': future_returns.get('Return_D3', np.nan),
            }
            records.append(rec)
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), f"评分后无股票"

    # 评分逻辑
    def normalize(s): 
        s = s.dropna()
        if s.empty or s.max() == s.min(): return pd.Series([0.5]*len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    fdf['s_vol'] = normalize(fdf['volatility'])
    
    if market_state == 'Strong':
        fdf['策略'] = '绝对MACD优势'
        fdf = fdf[fdf['macd'] > 0].copy()
        if not fdf.empty:
            fdf['综合评分'] = fdf['macd'] * 10000 + fdf['s_vol'].rsub(1) * 0.3
            fdf = fdf.sort_values('综合评分', ascending=False)
    else: 
        fdf['策略'] = '极致反弹防御'
        fdf['s_macd'] = normalize(fdf['macd'])
        score = fdf['s_vol'].rsub(1).fillna(0.5) * WEIGHT_VOL + fdf['s_macd'].fillna(0.5) * WEIGHT_MACD
        fdf['综合评分'] = score * 100
        fdf = fdf.sort_values('综合评分', ascending=False)
        
    return fdf.head(TOP_BACKTEST).copy(), None

# ---------------------------
# 主运行块 
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测"):
    
    trade_days_str = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days_str: st.stop()
    
    if not get_all_historical_data(trade_days_str): st.stop()
    st.success("✅ 数据就绪！开始右侧交易回测...")
    
    results_list = []
    my_bar = st.progress(0)
    
    for i, trade_date in enumerate(trade_days_str):
        daily_result_df, error = run_backtest_for_a_day(
            trade_date, TOP_BACKTEST, FINAL_POOL, BUY_THRESHOLD_PCT
        )
        if not daily_result_df.empty:
            daily_result_df['Trade_Date'] = trade_date
            results_list.append(daily_result_df)
        my_bar.progress((i + 1) / len(trade_days_str))
    my_bar.empty()
    
    if not results_list:
        st.error("无回测结果（可能全部未达到买入阈值）。")
        st.stop()
        
    all_results = pd.concat(results_list)
    if all_results['Trade_Date'].dtype != 'object': all_results['Trade_Date'] = all_results['Trade_Date'].astype(str)
        
    st.header(f"📊 右侧交易回测报告 (买入阈值: +{BUY_THRESHOLD_PCT}%)")
    
    cols = st.columns(2)
    for idx, n in enumerate([1, 3]):
        col_name = f'Return_D{n} (%)' 
        valid = all_results.dropna(subset=[col_name])
        
        # 计算逻辑：分母是“实际成交的交易次数”，而不是“所有推荐次数”
        if not valid.empty:
            avg_ret = valid[col_name].mean()
            hit_rate = (valid[col_name] > 0).sum() / len(valid) * 100
            count = len(valid)
        else:
            avg_ret, hit_rate, count = 0, 0, 0
            
        with cols[idx]:
            st.metric(
                f"D+{n} 收益 / 胜率", 
                f"{avg_ret:.2f}% / {hit_rate:.1f}%",
                help=f"实际成交笔数：{count}。未成交的交易已自动剔除。"
            )

    st.markdown(f"**注：** 交易笔数较少是正常的，因为代码过滤掉了 D1 最高价未触及 `开盘价 * {1+BUY_THRESHOLD_PCT/100}` 的所有股票。")
    
    st.header("📋 每日成交明细")
    # 只显示有实际回报率的记录（即实际成交的）
    mask_traded = all_results['Return_D1 (%)'].notna()
    st.dataframe(all_results[mask_traded].sort_values('Trade_Date', ascending=False), use_container_width=True)
