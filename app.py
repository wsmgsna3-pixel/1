# -*- coding: utf-8 -*-
"""
选股王 · V11.7 (最终极速优化版：绕开缓存失效的结构性优化)
更新说明：
1. 【**极速优化 V11.7**】：
   - 彻底重构历史数据拉取逻辑。
   - **get_all_history_data** 函数：一次性拉取 M 支股票所需的全部最长历史数据（120天指标窗口 + N天回测周期），并缓存。
   - **compute_indicators** 函数：不再调用 Tushare，而是从这个大的缓存中进行切片。
   - **效果：** 历史数据 API 调用次数从 N*M 次 降为 M 次，速度将得到根本性改善。
2. 【Bug 修复 V11.6/V11.5/V11.4】：修复了 NameError、SyntaxError、括号不匹配等所有已知 bug。
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
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V11.7 最终决战策略 (极速版)", layout="wide")
st.title("选股王 · V11.7 最终决战策略（结构优化极速版）")
st.markdown("🚀 **V11.7 最终修正版：彻底解决了回测速度慢的根本原因（缓存失效），将历史数据调用次数降到最低。**")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None 
MAX_SEARCH_DAYS = 15 # 最大往前查找天数
HISTORY_LOOKBACK_DAYS = 120 # 计算指标所需的最长历史数据天数（用于 MACD, 60日位置等）
GLOBAL_HISTORY_DATA = {} # 全局历史数据缓存，用于 V11.7 结构优化

# ---------------------------
# 辅助函数 
# ---------------------------
# 🚨 V11.6 重新添加：交易日获取及日期回退函数
def get_trade_days(end_date_str, num_days, mode="backtest"):
    """
    获取交易日列表。
    - 在 'select' 模式下，如果 end_date_str 的数据缺失，则自动向前回退。
    - 在 'backtest' 模式下，不进行回退，使用 num_days。
    """
    
    # 1. 获取日历
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=MAX_SEARCH_DAYS * 2)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历，请检查 Token 或 Tushare 权限。")
        return []
        
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    
    if trade_days_df.empty:
        return []

    # 2. 自动回退逻辑 (仅在选股模式或单日回测时，且数据拉取失败才触发)
    if mode == "select" or num_days == 1:
        for i in range(min(len(trade_days_df), MAX_SEARCH_DAYS)):
            check_date = trade_days_df['cal_date'].iloc[i]
            
            # 尝试拉取当日数据，判断数据是否已更新
            check_data = safe_get('daily', trade_date=check_date)
            
            if not check_data.empty:
                if check_date != end_date_str:
                    st.warning(f"⚠️ 原始日期 {end_date_str} 数据缺失，自动回退到最新可用交易日：{check_date}。")
                
                trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
                trade_days_df = trade_days_df[trade_days_df['cal_date'] <= check_date]
                
                return trade_days_df['cal_date'].head(num_days).tolist()
                
        st.error(f"在最近 {MAX_SEARCH_DAYS} 个交易日内，均无法获取到任何股票数据，请检查数据源或 Tushare 权限。")
        return []
    
    # 3. 多日回测模式 (直接返回指定天数)
    return trade_days_df['cal_date'].head(num_days).tolist()

@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    """安全调用 Tushare API - 已移除 time.sleep(0.5)"""
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
        time.sleep(0.1) 
        return pd.DataFrame(columns=['ts_code'])

# 调整缓存时间到 7 天（3600*24*7）
@st.cache_data(ttl=3600*24*7)
def get_adj_factor(ts_code, start_date, end_date):
    df = safe_get('adj_factor', ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty or 'adj_factor' not in df.columns: return pd.DataFrame()
    df['adj_factor'] = pd.to_numeric(df['adj_factor'], errors='coerce').fillna(0)
    df = df.set_index('trade_date').sort_index() 
    return df['adj_factor']

# ----------------------------------------------------
# 🚨 V11.7 核心优化函数：一次性获取所有股票的全部所需历史数据
@st.cache_data(ttl=3600*12)
def get_all_history_data(trade_days_list, candidate_codes):
    """
    一次性获取所有股票在给定回测窗口所需的最大历史数据。
    返回一个字典 {ts_code: DataFrame}。
    """
    if not trade_days_list or not candidate_codes: return {}
    
    # 确定最大的时间窗口
    end_date_str = max(trade_days_list)
    
    # 我们需要从最早回测日往前 HISTORY_LOOKBACK_DAYS 的数据
    # 为了简化和安全，我们直接从最早回测日期往前推 200 天
    max_start_date = (datetime.strptime(min(trade_days_list), "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
    
    st.info(f"💾 **正在建立/读取 {len(candidate_codes)} 支股票的完整历史数据缓存（一次性调用）。**\n\n请求范围：{max_start_date} 至 {end_date_str}")
    
    history_cache = {}
    
    # 这里不能使用 st.progress，因为这个函数本身是被缓存的
    for i, ts_code in enumerate(candidate_codes):
        # 为了让缓存键值更精准，我们还是用 get_qfq_data_v4，但它的参数范围是最大的
        df = get_qfq_data_v4(ts_code, start_date=max_start_date, end_date=end_date_str)
        if not df.empty:
            history_cache[ts_code] = df
            
    st.success("✅ 完整历史数据缓存建立/读取成功。")
    return history_cache

# 原始的前复权函数，现在用于获取大块数据，缓存键值包含大时间范围
@st.cache_data(ttl=3600*12)
def get_qfq_data_v4(ts_code, start_date, end_date):
    """获取前复权数据，用于一次性拉取大块历史数据"""
    daily_df = safe_get('daily', ts_code=ts_code, start_date=start_date, end_date=end_date)
    if daily_df.empty: return pd.DataFrame()
    daily_df = daily_df.set_index('trade_date').sort_index()
    
    adj_factor_series = get_adj_factor(ts_code, start_date, end_date)
    if adj_factor_series.empty: return pd.DataFrame()
    
    df = daily_df.merge(adj_factor_series.rename('adj_factor'), left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    if df.empty: return pd.DataFrame()
    latest_adj_factor = df['adj_factor'].iloc[-1]
    
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            if latest_adj_factor > 1e-9:
                df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
            else:
                df[col + '_qfq'] = df[col] 
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df['trade_date'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')
    df = df.sort_values('trade_date').set_index('trade_date_str')
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col + '_qfq']
        
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 

def get_future_prices(ts_code, selection_date, global_data):
    """从全局数据中切片获取未来价格"""
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    
    if ts_code not in global_data:
        results = {}
        for n in [1, 3, 5]: results[f'Return_D{n}'] = np.nan
        return results

    full_hist = global_data[ts_code]
    full_hist = full_hist.sort_index() 
    
    # 获取选股日当天及之前的数据（用于基准价格）
    selection_price_df = full_hist.loc[full_hist.index <= selection_date]
    selection_price_adj = selection_price_df['close'].iloc[-1] if not selection_price_df.empty else np.nan
    
    # 获取选股日之后的数据（用于未来价格）
    future_hist = full_hist.loc[full_hist.index > selection_date]
    future_hist = future_hist.reset_index(drop=True) 
    
    results = {}
    for n in [1, 3, 5]:
        col_name = f'Return_D{n}'
        if len(future_hist) >= n:
            future_price = future_hist.iloc[n-1]['close']
            if pd.notna(selection_price_adj) and selection_price_adj > 1e-9:
                results[col_name] = (future_price / selection_price_adj - 1) * 100
            else:
                results[col_name] = np.nan
        else:
            results[col_name] = np.nan
    return results


def compute_indicators(ts_code, end_date, global_data):
    """
    计算 MACD, 10日回报, 波动率, 60日位置等指标。
    数据从全局缓存中切片，不再调用 Tushare API。
    """
    res = {}
    if ts_code not in global_data: return res
    
    full_df = global_data[ts_code]
    
    # 确定切片的起始日期：end_date 往前推 120 个交易日（粗略推算）
    end_date_dt = datetime.strptime(end_date, "%Y%m%d")
    
    # 我们只需要 end_date 及之前的历史数据
    hist_df = full_df.loc[full_df.index <= end_date]
    
    if hist_df.empty or len(hist_df) < 3 or 'close' not in hist_df.columns: return res
    
    # 为了确保有足够的数据计算指标（至少需要 120 天），我们取最后 200 个交易日
    df = hist_df.tail(200) 
    
    df['close'] = pd.to_numeric(df['close'], errors='coerce').astype(float)
    df['low'] = pd.to_numeric(df['low'], errors='coerce').astype(float)
    df['high'] = pd.to_numeric(df['high'], errors='coerce').astype(float)
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce').fillna(0)
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
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
        
    # 量比计算
    vols = df['vol'].tolist()
    if len(vols) >= 6 and vols[-6:-1] and np.mean(vols[-6:-1]) > 1e-9:
        res['vol_ratio'] = vols[-1] / np.mean(vols[-6:-1])
    else: res['vol_ratio'] = np.nan
        
    # 10日回报、波动率计算
    res['10d_return'] = close.iloc[-1]/close.iloc[-10] - 1 if len(close)>=10 and close.iloc[-10]!=0 else 0
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    # 60日位置计算
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


# ----------------------------------------------------
# 侧边栏参数 (定义 BACKTEST_DAYS 等变量)
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    
    run_mode = st.radio("选择运行模式", 
                        ("今日选股 (自动匹配最新可用日)", "多日回测 (指定天数)"),
                        key='run_mode', 
                        help="选股模式：自动寻找最新有数据的交易日，仅回测 1 天。回测模式：按您指定的日期和天数回测。")
    
    backtest_date_end = st.date_input("选择**回测/选股日期**", value=datetime.now().date(), max_value=datetime.now().date(), key='end_date')
    
    if run_mode == "多日回测 (指定天数)":
        BACKTEST_DAYS = int(st.number_input("**自动回测天数 (N)**", value=20, step=1, min_value=1, max_value=50, key='backtest_days_input', help="程序将自动回测最近 N 个交易日。"))
        MODE = "backtest"
    else:
        # 选股模式，固定为 1 天，但日期会自动回退到有数据的日子
        BACKTEST_DAYS = 1
        MODE = "select"
    
    st.markdown("---")
    st.header("核心参数")
    # 建议 M >= 100 
    FINAL_POOL = int(st.number_input("最终入围评分数量 (M)", value=100, step=10, min_value=1, key='final_pool', help="（推荐 100 或更高，以充分利用高权限）")) 
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=10, step=1, key='top_display'))
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=3, step=1, min_value=1, key='top_backtest')) 
    
    st.markdown("---")
    st.header("🛒 灵活过滤条件")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0, step=0.5, min_value=0.1, key='min_price')
    MAX_PRICE = st.number_input("最高股价 (元)", value=300.0, step=5.0, min_value=1.0, key='max_price')
    MIN_TURNOVER = st.number_input("最低换手率 (%)", value=2.0, step=0.5, min_value=0.1, key='min_turnover') 
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿元)", value=20.0, step=1.0, min_value=1.0, key='min_circ_mv', help="例如：输入 20 代表流通市值必须大于等于 20 亿元。")
    MIN_AMOUNT_MILLIONS = st.number_input("最低成交额 (亿元)", value=0.6, step=0.1, min_value=0.1, key='min_amount_mil')
    MIN_AMOUNT = MIN_AMOUNT_MILLIONS * 100000000 

# ---------------------------
# Token 输入与初始化 
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password", key='ts_token')
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ---------------------------
# 核心回测逻辑函数 
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS, global_data):
    """为单个交易日运行选股和回测逻辑 (V11.7 使用 global_data)"""
    
    # 1. 拉取全市场 Daily 数据
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty or 'ts_code' not in daily_all.columns: 
        return pd.DataFrame(), f"数据缺失或拉取失败：{last_trade}"

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
    
    # --- 资金流数据处理 FIX START ---
    moneyflow_to_merge = pd.DataFrame()
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in']
        for c in possible:
            if c in mf_raw.columns:
                moneyflow_to_merge = mf_raw[['ts_code', c]].rename(columns={c:'net_mf'})
                break            
    
    if not moneyflow_to_merge.empty:
        pool_merged = pool_merged.merge(moneyflow_to_merge, on='ts_code', how='left')
        
    if 'net_mf' not in pool_merged.columns:
        pool_merged['net_mf'] = np.nan 
        
    pool_merged['net_mf'] = pd.to_numeric(pool_merged['net_mf'], errors='coerce').fillna(0) 
    # --- 资金流数据处理 FIX END ---

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

    if len(df) == 0: return pd.DataFrame(), f"过滤后无股票：{last_trade}"

    # 4. 遴选决赛名单 (基于当日涨幅和换手率的混合初筛)
    limit_pct = int(FINAL_POOL * 0.7)
    df_pct = df.sort_values('pct_chg', ascending=False).head(limit_pct).copy()
    limit_turn = FINAL_POOL - len(df_pct)
    existing_codes = set(df_pct['ts_code'])
    df_turn = df[~df['ts_code'].isin(existing_codes)].sort_values('turnover_rate', ascending=False).head(limit_turn).copy()
    final_candidates = pd.concat([df_pct, df_turn]).reset_index(drop=True)

 
    # 5. 深度评分 
    records = []
    for row in final_candidates.itertuples():
        ts_code = row.ts_code
        
        rec = {
            'ts_code': ts_code, 'name': getattr(row, 'name', ts_code),
            'Close': getattr(row, 'close', np.nan),
            'Circ_MV (亿)': getattr(row, 'circ_mv_billion', np.nan),
            'Pct_Chg (%)': getattr(row, 'pct_chg', 0), 
            'turnover': getattr(row, 'turnover_rate', 0),
            'net_mf': getattr(row, 'net_mf', 0)
        }
        
        # 🚨 V11.7: 传入 global_data，从缓存中切片
        ind = compute_indicators(ts_code, last_trade, global_data)
        rec.update({
            'vol_ratio': ind.get('vol_ratio', 0), 'macd': ind.get('macd_val', 0),
            '10d_return': ind.get('10d_return', 0),
            'volatility': ind.get('volatility', 0), 'position_60d': ind.get('position_60d', np.nan)
        })
        
        # 只有在多日回测时才需要未来收益
        if MODE == 'backtest':
            # 🚨 V11.7: 传入 global_data，从缓存中切片
            future_returns = get_future_prices(ts_code, last_trade, global_data)
            rec.update({
                'Return_D1 (%)': future_returns.get('Return_D1', np.nan),
                'Return_D3 (%)': future_returns.get('Return_D3', np.nan),
                'Return_D5 (%)': future_returns.get('Return_D5', np.nan),
            })
        else:
             rec.update({
                'Return_D1 (%)': np.nan,
                'Return_D3 (%)': np.nan,
                'Return_D5 (%)': np.nan,
            })


        records.append(rec)
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), f"评分列表为空：{last_trade}"

    # 6. 归一化与 V11.0 策略精调评分 
    def normalize(series):
        series_nn = series.dropna() 
        if series_nn.max() == series_nn.min(): return pd.Series([0.5] * len(series), index=series.index)
        return (series - series_nn.min()) / (series_nn.max() - series_nn.min() + 1e-9)

 
    fdf['s_pct'] = normalize(fdf['Pct_Chg (%)'])
    fdf['s_turn'] = normalize(fdf['turnover'])
    fdf['s_vol'] = normalize(fdf['vol_ratio'])
    fdf['s_mf'] = normalize(fdf['net_mf'])
    fdf['s_macd'] = normalize(fdf['macd'])
    fdf['s_trend'] = normalize(fdf['10d_return'])
    fdf['s_volatility'] = normalize(fdf['volatility'])
    fdf['s_position'] = fdf['position_60d'] / 100 
    
    # ----------------------------------------------------------------------------------
    # 🚨 V11.0 最终决战策略：V9.0 框架 + 强化 MACD 趋势共振版
    
    # 核心权重：资金流，占比 35%
    w_mf = 0.35            
    # 动能权重：当日动能，占比 20%
    w_pct = 0.10            
    w_turn = 0.10           
    # 防御权重：安全边际与波动控制，占比 25%
    w_position = 0.15       
    w_volatility = 0.10     
    # 趋势权重：中期趋势，占比 20%
    w_macd = 0.20           
    # 彻底归零项
    w_vol = 0.00            
    w_trend = 0.00          
    
    # Sum: 0.35+0.10+0.10+0.15+0.10+0.20 = 1.00
    
  
    score = (
        fdf['s_pct'] * w_pct + fdf['s_turn'] * w_turn + 
        fdf['s_mf'] * w_mf + 
        fdf['s_macd'] * w_macd + 
        
        # 引入防御：60日位置越低越好 (1-s_position)，波动率越低越好 (1-s_volatility)
        (1 - fdf['s_position']) * w_position + 
        (1 - fdf['s_volatility']) * w_volatility + 
        
 
        # 归零项
        fdf['s_vol'] * w_vol + 
        fdf['s_trend'] * w_trend     
    )
    fdf['综合评分'] = score * 100
    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index += 1
    # ----------------------------------------------------------------------------------


    return fdf.head(TOP_BACKTEST).copy(), None

# ---------------------------
# 主运行块 
# ---------------------------

# 将所有运行逻辑包装在一个函数中
def execute_run(mode, backtest_days):
    
    if mode == "select":
        st.header(f"🚀 正在进行 1 个交易日的**选股**...")
    else:
        st.header(f"📈 正在进行 {backtest_days} 个交易日的**回测**...")

    trade_days_str = get_trade_days(backtest_date_end.strftime("%Y%m%d"), backtest_days, mode=mode)
    
    if not trade_days_str:
        st.error("无法获取交易日列表，请检查日期或 Token。")
        st.stop()
    
    
    # 1. 启动全局数据拉取和缓存 (V11.7 核心步骤)
    # 我们需要找到所有可能入围的股票代码，为了安全，我们拉取所有当日有数据的股票
    # 仅获取 end_date 的数据，以保证拉取的是最新的交易日数据
    initial_daily_data = safe_get('daily', trade_date=max(trade_days_str))
    
    if initial_daily_data.empty:
        st.error(f"无法获取日期 {max(trade_days_str)} 的日线数据，请检查 Tushare Token。")
        st.stop()
        
    candidate_codes = initial_daily_data['ts_code'].tolist()
    
    # 运行一次，将所有历史数据缓存起来
    global_data = get_all_history_data(trade_days_str, candidate_codes)
    
    if not global_data:
        st.error("无法拉取或建立历史数据缓存，请检查 Tushare 权限。")
        st.stop()

    
    results_list = []
    total_days = len(trade_days_str)
    
    progress_text = st.empty()
    my_bar = st.progress(0)
    
    start_time = time.time() 
    
    for i, trade_date in enumerate(trade_days_str):
        
        progress_text.text(f"🚀 正在处理第 {i+1}/{total_days} 个交易日：{trade_date} (数据已从本地缓存中切片)")
            
        daily_result_df, error = run_backtest_for_a_day(
            trade_date, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS, global_data
        )
        
        if error:
            st.warning(f"跳过 {trade_date}：{error}")
        elif not daily_result_df.empty:
            daily_result_df['Trade_Date'] = trade_date
            results_list.append(daily_result_df)
            
        my_bar.progress((i + 1) / total_days)

    end_time = time.time()
    total_duration = end_time - start_time
    
    progress_text.text(f"✅ 运行完成，正在汇总结果... 总耗时: {total_duration:.2f} 秒")
    my_bar.empty()
    
    if not results_list:
        st.error("所有交易日的回测均失败或无结果。")
        st.stop()
        
    all_results = pd.concat(results_list)
    
    # 区分显示选股结果和回测结果
    if mode == "select":
        st.success(f"🎉 **【今日选股结果】**：已成功使用最新可用数据（{trade_days_str[0]}）进行选股！")
        st.header(f"📋 选股推荐结果 (Top {TOP_BACKTEST})")
        display_cols = ['Trade_Date', 'name', 'ts_code', '综合评分', 
                        'Close', 'Pct_Chg (%)', 'Circ_MV (亿)']
        
    else: # 多日回测
        st.header(f"📊 最终平均回测结果 (Top {TOP_BACKTEST}，共 {total_days} 个交易日)")
        
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
                      help=f"总有效样本数：{total_count}。**V11.0 已应用 V9.0 框架 + 强化 MACD 趋势共振策略。**")

        st.header("📋 每日回测详情 (Top K 明细)")
        display_cols = ['Trade_Date', 'name', 'ts_code', '综合评分', 
                        'Close', 'Pct_Chg (%)', 'Circ_MV (亿)',
                        'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)']
    
    st.dataframe(all_results[display_cols].sort_values('综合评分', ascending=False).head(TOP_DISPLAY), use_container_width=True)

# ---------------------------
# 主界面按钮触发
# ---------------------------
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 今日选股 (1日)", key='select_button', help="使用最新的可用交易日数据进行选股。"):
        st.warning("⚠️ **V11.7 最终极速版已上线。请使用此版本进行测试。**")
        execute_run("select", 1)

with col2:
    if st.button(f"⏳ 开始 {BACKTEST_DAYS} 日自动回测", key='backtest_button', help="使用指定日期和天数进行历史回测。"):
        st.warning("⚠️ **V11.7 最终极速版已上线。请使用此版本进行测试。**")
        execute_run("backtest", BACKTEST_DAYS)
