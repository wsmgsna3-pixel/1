# -*- coding: utf-8 -*-
"""
选股王 · V13.3 日期修正版 (核心：解决 Tushare 返回未来交易日的问题)

说明：
1. 【日期修复】修正 find_last_trade_day 逻辑，确保返回的日期是已收盘的、可查询到数据的日期。
2. 【保留 V13.2 核心风控】包含次新股过滤、市值收紧、理性权重。
"""
import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import joblib 
import os
import math

warnings.filterwarnings("ignore")

# ---------------------------
# 外部缓存配置 (用于历史数据)
# ---------------------------
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)

# ---------------------------
# 页面设置 (UI 空间最大化)
# ---------------------------
st.set_page_config(page_title="选股王（V13.3 日期修正版）", layout="wide")
st.markdown("### 选股王（V13.3 日期修正版）") 

# ---------------------------
# 侧边栏参数 
# ---------------------------
with st.sidebar:
    st.header("可调参数（V13.3 默认值）")
    INITIAL_TOP_N = 99999 
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=500, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    
    # V13.2 市值上下限参数
    MIN_CIRC_MV_Billion = float(st.number_input("最低流通市值 (亿)", value=40.0, step=5.0)) 
    MAX_CIRC_MV_Billion = float(st.number_input("最高流通市值 (亿)", value=500.0, step=50.0)) 
    
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=0.5, step=0.1)) 
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=20_000_000.0, step=5_000_000.0))
    
    MA_TREND_PERIOD = int(st.number_input("硬性趋势过滤：MA 周期", value=20, step=5))
    
    # V13.2 新增：次新股硬性过滤参数
    MIN_LIST_DAYS = int(st.number_input("次新股排除：最低上市天数 (天)", value=180, step=30))
    
    VOLATILITY_MAX = float(st.number_input("过去20日波动 std 阈值 (%)", value=6.0, step=0.5)) 
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5)) 
    
    st.markdown("---")
    
    # 回测天数 N
    BACKTEST_DAYS = int(st.number_input("回测：最近 N 个交易日", value=10, step=1))
    
    st.markdown("---")
    st.caption("提示：策略已升级至 'V13.3 日期修正版'。")
    st.caption("核心：修复了 Tushare 返回未来日期导致的选股失败。")

# ---------------------------
# Token 输入
# ---------------------------
st.markdown("请输入 Tushare Token。")
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password", label_visibility="collapsed")

if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

# 初始化 tushare
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 依赖函数：数据安全获取
# ---------------------------
def safe_get(func, **kwargs):
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

# ---------------------------
# 交易日历获取 (V13.3 修正)
# ---------------------------
@st.cache_data(ttl=600)
def find_last_trade_day():
    # 尝试获取最近的交易日
    today_date = datetime.now().strftime("%Y%m%d")
    cal_df = safe_get(
        pro.trade_cal, 
        exchange='SSE', 
        is_open='1', 
        end_date=today_date, 
        fields='cal_date'
    )
    
    if cal_df.empty: return None

    # 获取所有交易日，并降序排列
    trade_dates = cal_df['cal_date'].sort_values(ascending=False).tolist()
    
    # 确保返回的日期的数据已经可以获取
    for trade_date in trade_dates:
        # 简单检查该日期是否有数据 (例如检查沪深300指数数据)
        test_df = safe_get(pro.daily, ts_code='000300.SH', trade_date=trade_date)
        if not test_df.empty:
            return trade_date
            
    return None # 无法找到任何有数据的交易日

last_trade = find_last_trade_day()
if not last_trade:
    st.error("无法找到最近交易日，检查网络或 Token 权限。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")


# ----------------------------------------------------
# 按钮控制模块 (保持不变)
# ----------------------------------------------------
if 'run_selection' not in st.session_state: st.session_state['run_selection'] = False
if 'run_backtest' not in st.session_state: st.session_state['run_backtest'] = False
if 'backtest_status' not in st.session_state: 
    st.session_state['backtest_status'] = {'progress': 0.0, 'results': [], 'current_index': 0, 'total_days': 0}

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 运行当日选股", use_container_width=True):
        st.session_state['run_selection'] = True
        st.session_state['run_backtest'] = False
        st.session_state['backtest_status'] = {'progress': 0.0, 'results': [], 'current_index': 0, 'total_days': 0}
        st.rerun()

with col2:
    if st.button(f"✅ 运行历史回测 ({BACKTEST_DAYS} 日)", use_container_width=True):
        st.session_state['run_backtest'] = True
        st.session_state['run_selection'] = False
        if st.session_state['backtest_status']['progress'] == 1.0 or st.session_state['backtest_status']['total_days'] == 0:
             st.session_state['backtest_status'] = {'progress': 0.0, 'results': [], 'current_index': 0, 'total_days': 0}
        st.rerun()

st.markdown("---")

# ---------------------------
# 指标计算和归一化 (保持不变)
# ---------------------------
def compute_indicators(df, ma_period):
    res = {}
    if df.empty or len(df) < 3: return res
    close = df['close'].astype(float)
    
    if len(close) >= ma_period:
        res[f'ma{ma_period}'] = close.rolling(window=ma_period).mean().iloc[-1]
    else:
        res[f'ma{ma_period}'] = np.nan
        
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_val = (diff - dea) * 2
        res['macd'] = macd_val.iloc[-1]
    else: res['macd'] = np.nan

    try:
        if 'pct_chg' in df.columns and len(df) >= 20:
            res['volatility_20'] = df['pct_chg'].astype(float).tail(20).std()
        else: res['volatility_20'] = np.nan
    except: res['volatility_20'] = np.nan
    return res

def safe_merge_pool(pool_df, other_df, cols):
    pool = pool_df.set_index('ts_code').copy()
    if other_df is None or other_df.empty:
        for c in cols: pool[c] = np.nan
        return pool.reset_index()
    if 'ts_code' not in other_df.columns:
        try: other_df = other_df.reset_index()
        except:
            for c in cols: pool[c] = np.nan
            return pool.reset_index()
    
    for c in cols:
        if c not in other_df.columns: other_df[c] = np.nan
    
    try: 
        joined = pool.join(other_df.set_index('ts_code')[cols], how='left')
    except Exception:
        for c in cols: pool[c] = np.nan
        return pool.reset_index()
    
    for c in cols:
        if c not in joined.columns: joined[c] = np.nan
        
    return joined.reset_index()

def norm_col(s):
    s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    mn = s.min(); mx = s.max()
    if mx - mn < 1e-9: return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)


# ----------------------------------------------------
# 核心评分函数 (V13.3 = V13.2 逻辑)
# ----------------------------------------------------
@memory.cache 
def run_scoring_for_date(trade_date, params):
    
    # 解包参数
    min_price, max_price, min_turnover, min_amount, min_circ_mv_billion, max_circ_mv_billion, ma_trend_period, min_list_days = \
        params['MIN_PRICE'], params['MAX_PRICE'], params['MIN_TURNOVER'], params['MIN_AMOUNT'], \
        params['MIN_CIRC_MV_Billion'], params['MAX_CIRC_MV_Billion'], params['MA_TREND_PERIOD'], params['MIN_LIST_DAYS']
    
    # 1. 拉取数据
    daily_all = safe_get(pro.daily, trade_date=trade_date)
    daily_basic = safe_get(pro.daily_basic, trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    
    if daily_all.empty: 
        if trade_date == last_trade: st.error(f"诊断：Tushare 无法获取 {trade_date} 的日线数据。")
        return pd.DataFrame()
    
    pool0 = daily_all.copy().reset_index(drop=True)

    # 2. 合并高级接口数据 (包含 name, industry, list_date)
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,total_mv,circ_mv,list_date')
    
    if not stock_basic.empty:
        keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv','list_date'] if c in stock_basic.columns]
        try: pool0 = pool0.merge(stock_basic[keep], on='ts_code', how='left')
        except Exception: 
            pool0['name'] = pool0['ts_code']; pool0['industry'] = ''; pool0['list_date'] = '20000101'
    else: 
        pool0['name'] = pool0['ts_code']; pool0['industry'] = ''; pool0['list_date'] = '20000101'
    
    pool_merged = safe_merge_pool(pool0, daily_basic.rename(columns={'amount':'amount_db'}), ['turnover_rate','amount_db','total_mv','circ_mv'])
    
    # 数据清洗和转换 
    if 'amount' in pool_merged.columns:
        pool_merged['amount'] = pool_merged['amount'].apply(lambda amt: amt * 10000.0 if not pd.isna(amt) and amt > 0 and amt < 1e5 else amt)
    else:
        pool_merged['amount'] = pool_merged['amount_db'].apply(lambda amt: amt * 10000.0 if not pd.isna(amt) and amt > 0 and amt < 1e5 else amt)
    
    pool_merged['amount_yuan'] = pool_merged['amount']
    pool_merged['circ_mv_wan'] = pool_merged['circ_mv'].fillna(0)


    # 3. 硬性过滤（清洗）
    clean_df = pool_merged.copy()
    
    # 基础风险过滤 
    clean_df = clean_df[~(
        (clean_df['close'].isna()) | 
        (clean_df['close'] < min_price) | 
        (clean_df['close'] > max_price) | 
        (clean_df['name'].str.contains('ST|退', case=False, na=False)) |
        (clean_df['ts_code'].str.endswith('.BJ', na=False)) # 排除北交所
    )]
    
    # 涨跌幅过滤 (剔除停牌/未交易)
    clean_df = clean_df[~((clean_df['pct_chg'].isna()))]
    
    # --- V13.2 核心风控 ---
    
    # 修复 1: 次新股过滤 (排除上市不足 180 天的股票)
    current_date = datetime.strptime(trade_date, "%Y%m%d")
    clean_df['list_date'] = pd.to_datetime(clean_df['list_date'], format='%Y%m%d', errors='coerce')
    clean_df['days_since_list'] = (current_date - clean_df['list_date']).dt.days
    clean_df = clean_df[clean_df['days_since_list'].notna() & (clean_df['days_since_list'] >= min_list_days)]
    
    # 修复 2: 流通市值上下限过滤 (40亿 到 500亿)
    min_circ_mv_wan = min_circ_mv_billion * 10000.0 
    max_circ_mv_wan = max_circ_mv_billion * 10000.0 
    clean_df = clean_df[clean_df['circ_mv_wan'].notna() & 
                        (clean_df['circ_mv_wan'] >= min_circ_mv_wan) &
                        (clean_df['circ_mv_wan'] <= max_circ_mv_wan)]

    # 流动性过滤
    clean_df = clean_df[clean_df['amount_yuan'].notna() & (clean_df['amount_yuan'] >= min_amount)]
    clean_df = clean_df[clean_df['turnover_rate'].notna() & (clean_df['turnover_rate'] >= min_turnover)]
    
    
    if clean_df.empty: 
        if trade_date == last_trade: st.error(f"诊断：所有硬性过滤后，剩余股票数量为 **0** 支。请检查侧边栏参数。")
        return pd.DataFrame()

    if trade_date == last_trade:
        st.info(f"诊断：硬性过滤 (已包含次新股、市值收紧) 后，剩余股票数量: **{len(clean_df)}** 支，开始计算指标...")

    # 4. 指标计算与 MA20 趋势硬性过滤 
    records = []
    start_dt = datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=60 * 1.5) 
    start_date_hist = start_dt.strftime("%Y%m%d")
    
    pbar = None
    if trade_date == last_trade:
        pbar = st.progress(0.0, text=f"正在计算 {len(clean_df)} 支股票的指标...")

    for i, row in enumerate(clean_df.itertuples()):
        ts_code = getattr(row, 'ts_code'); turnover_rate = getattr(row, 'turnover_rate', np.nan);
        close_price = getattr(row, 'close', np.nan)
        
        @memory.cache
        def get_daily_hist(ts_code, start_date, end_date):
            return safe_get(pro.daily, ts_code=ts_code, start_date=start_date, end_date=end_date)
            
        hist = get_daily_hist(ts_code, start_date_hist, trade_date)
        
        # 计算指标
        ind = compute_indicators(hist, ma_trend_period)
        ma_trend_val = ind.get(f'ma{ma_trend_period}', np.nan)
        
        # --- MA 趋势硬性过滤 ---
        if not pd.isna(close_price) and not pd.isna(ma_trend_val) and (close_price < ma_trend_val):
             if pbar: pbar.progress((i + 1) / len(clean_df), text=f"指标计算进度：[{i+1}/{len(clean_df)}]... (已排除趋势向下股)")
             continue 

        # 如果通过 MA 过滤，继续存储所有指标
        macd, volatility_20 = ind.get('macd', np.nan), ind.get('volatility_20', np.nan)

        rec = {
            'ts_code': ts_code, 
            'pct_chg': getattr(row, 'pct_chg', np.nan),
            'turnover_rate': turnover_rate,
            'macd': macd, 
            'volatility_20': volatility_20,
            'name': getattr(row, 'name', ts_code),
            'circ_mv_wan': getattr(row, 'circ_mv_wan', np.nan),
            f'ma{ma_trend_period}': ma_trend_val
        }
        records.append(rec)
        
        if pbar: pbar.progress((i + 1) / len(clean_df), text=f"指标计算进度：[{i+1}/{len(clean_df)}]... (已排除趋势向下股)")
        
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame()
    if pbar: pbar.empty()

    if trade_date == last_trade:
        st.info(f"诊断：通过 {ma_trend_period} 日均线趋势过滤后，剩余股票数量: **{len(fdf)}** 支，开始评分...")
        
    # 5. 归一化和评分
    
    # Log-平滑流动性因子
    fdf['turnover_rate_clean'] = fdf['turnover_rate'].fillna(params['MIN_TURNOVER']) 
    fdf['log_turnover'] = fdf['turnover_rate_clean'].apply(lambda x: math.log(x) if x > 0 else math.log(1e-6))
    
    # 归一化指标
    fdf['s_log_turn'] = norm_col(fdf.get('log_turnover', pd.Series([0]*len(fdf)))) # Log-平滑流动性评分
    fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
    fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_20', pd.Series([0]*len(fdf)))) # 低波动评分

    # V13.2 核心权重：低波动(0.45) + MACD(0.35) + Log-流动性(0.20)
    w_log_turn = 0.20    
    w_macd = 0.35    
    w_volatility = 0.45 

    fdf['综合评分'] = (fdf['s_log_turn'] * w_log_turn + 
                     fdf['s_macd'] * w_macd + 
                     fdf['s_volatility'] * w_volatility)
    
    return fdf.sort_values('综合评分', ascending=False).head(params['FINAL_POOL']).reset_index(drop=True)


# ----------------------------------------------------
# 简易回测模块 (保持 V13.2 逻辑)
# ----------------------------------------------------
def run_simple_backtest(days, params):
    
    HOLDING_PERIODS = [1, 3, 5]
    status = st.session_state['backtest_status']
    
    container = st.empty()
    with container.container():
        st.subheader(f"📈 简易历史回测结果 (V13.3 日期修正版)")
        
        trade_dates_df = safe_get(pro.trade_cal, exchange='SSE', is_open='1', end_date=find_last_trade_day(), fields='cal_date')
        if trade_dates_df.empty:
            st.error("无法获取历史交易日历。")
            return

        max_holding = max(HOLDING_PERIODS)
        trade_dates = trade_dates_df['cal_date'].sort_values(ascending=False).head(days + max_holding).tolist() 
        trade_dates.reverse() 
        total_iterations = len(trade_dates) - max_holding 
        
        if total_iterations < 1:
            st.warning(f"交易日不足 {max_holding + 1} 天，无法进行回测。")
            return
            
        status['total_days'] = total_iterations
        start_index = status['current_index']
        
        if start_index >= total_iterations:
             st.success(f"回测已完成。累计收益率请查看下方。")
        else:
             st.info(f"回测周期：**{trade_dates[0]}** 至 **{trade_dates[total_iterations-1]}**。正在从第 {start_index+1} 天继续...")

        pbar = st.progress(status['progress'], text=f"回测进度：[{status['current_index']}/{status['total_days']}]...")
        
        score_params = {
            'MIN_PRICE': params['MIN_PRICE'], 'MAX_PRICE': params['MAX_PRICE'], 
            'MIN_TURNOVER': params['MIN_TURNOVER'], 'MIN_AMOUNT': params['MIN_AMOUNT'], 
            'MIN_CIRC_MV_Billion': params['MIN_CIRC_MV_Billion'],
            'MAX_CIRC_MV_Billion': params['MAX_CIRC_MV_Billion'],
            'MA_TREND_PERIOD': params['MA_TREND_PERIOD'],
            'MIN_LIST_DAYS': params['MIN_LIST_DAYS'] 
        }
        
        for i in range(start_index, total_iterations):
            select_date = trade_dates[i]
            next_trade_date = trade_dates[i+1] 
            
            select_df_full = run_scoring_for_date(select_date, score_params) 

            result = {
                '选股日': select_date, 
                '股票': '无符合条件', 
                '买入价 (T+1 开盘)': np.nan, 
                '评分': np.nan,
                '市值 (亿)': np.nan
            }
            for N in HOLDING_PERIODS:
                 result[f'T+{N} 收益率 (%)'] = 0.0
                 result[f'T+{N} 卖出价'] = np.nan
                 
            
            if not select_df_full.empty:
                top_pick = select_df_full.iloc[0] 
                ts_code = top_pick['ts_code']
                
                buy_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=next_trade_date)
                buy_price = buy_day_data.iloc[0]['open'] if not buy_day_data.empty and 'open' in buy_day_data.columns else np.nan
                
                result['股票'] = f"{top_pick.get('name', 'N/A')}({ts_code})"
                result['买入价 (T+1 开盘)'] = buy_price
                result['评分'] = top_pick['综合评分']
                result['市值 (亿)'] = top_pick['circ_mv_wan'] / 10000.0 if not pd.isna(top_pick['circ_mv_wan']) else np.nan
                
                if buy_price > 0 and not pd.isna(buy_price):
                    
                    for N in HOLDING_PERIODS:
                        sell_trade_date = trade_dates[i+N] 
                        
                        sell_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=sell_trade_date)
                        
                        if not sell_day_data.empty and 'close' in sell_day_data.columns:
                            sell_price = sell_day_data.iloc[0]['close']
                            result[f'T+{N} 卖出价'] = sell_price
                            
                            if not pd.isna(sell_price):
                                return_pct = (sell_price / buy_price) - 1.0
                                return_pct = max(-0.10, return_pct) 
                                result[f'T+{N} 收益率 (%)'] = return_pct * 100
                        
            status['results'].append(result)
            status['current_index'] = i + 1
            status['progress'] = (i + 1) / total_iterations
            
            pbar.progress(status['progress'], text=f"正在回测 {select_date}... [{i+1}/{total_iterations}]")
            
            if (i+1) % 2 == 0 or (i + 1) == total_iterations: 
                 st.rerun() 
        
        status['progress'] = 1.0
        status['current_index'] = total_iterations
        pbar.progress(1.0, text="回测完成。")
        
        results_df = pd.DataFrame(status['results'])
        
        if results_df.empty:
            st.warning("回测结果为空。")
            return
            
        st.markdown("---")
        st.subheader("💡 最终回测指标（多周期对比）")
        
        cols_metrics = st.columns(len(HOLDING_PERIODS))
        
        for idx, N in enumerate(HOLDING_PERIODS):
            col_name = f'T+{N} 收益率 (%)'
            results_df[col_name] = results_df[col_name].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            cumulative_return = (results_df[col_name] / 100 + 1).product() - 1
            wins = (results_df[col_name] > 0).sum()
            total_trades = len(results_df)
            win_rate = wins / total_trades if total_trades > 0 else 0

            with cols_metrics[idx]:
                st.metric(f"累计收益率 (T+{N})", f"{cumulative_return*100:.2f}%")
                st.caption(f"胜率: {win_rate*100:.2f}% | 交易次数: {total_trades}")
        
        st.markdown("---")
        st.subheader("📋 每日交易记录")
        
        display_cols = ['选股日', '股票', '市值 (亿)', '评分', '买入价 (T+1 开盘)']
        for N in HOLDING_PERIODS:
            display_cols.append(f'T+{N} 收益率 (%)')
            
        st.dataframe(results_df[display_cols], use_container_width=True)


# ----------------------------------------------------
# 实时选股模块 (保持 V13.2 逻辑)
# ----------------------------------------------------
def run_live_selection(last_trade, params):
    st.write(f"正在运行实时选股（最近交易日：{last_trade}）...")
    
    params_dict = {
        'MIN_PRICE': params['MIN_PRICE'], 'MAX_PRICE': params['MAX_PRICE'], 
        'MIN_TURNOVER': params['MIN_TURNOVER'], 'MIN_AMOUNT': params['MIN_AMOUNT'], 
        'MIN_CIRC_MV_Billion': params['MIN_CIRC_MV_Billion'],
        'MAX_CIRC_MV_Billion': params['MAX_CIRC_MV_Billion'],
        'MA_TREND_PERIOD': params['MA_TREND_PERIOD'],
        'MIN_LIST_DAYS': params['MIN_LIST_DAYS'] 
    }
    
    fdf_full = run_scoring_for_date(last_trade, params_dict)

    if fdf_full.empty:
        st.error(f"清洗和评分后没有候选。请检查硬性过滤参数是否过于严格（当前过滤后剩余：0 支）。")
        st.stop()

    fdf = fdf_full.head(params['TOP_DISPLAY']).copy()
    fdf.index = fdf.index + 1

    st.success(f"评分完成：总候选 {len(fdf_full)} 支，显示 Top {min(params['TOP_DISPLAY'], len(fdf))}。")
    
    # 转换为亿显示 
    fdf['流通市值 (亿)'] = fdf['circ_mv_wan'] / 10000.0
    
    display_cols_full = ['name','ts_code','综合评分','pct_chg','turnover_rate','circ_mv_wan','volatility_20', 'log_turnover']
    
    for c in display_cols_full:
        if c not in fdf_full.columns: fdf_full[c] = np.nan 

    final_display_cols = ['name','ts_code','综合评分','流通市值 (亿)','pct_chg','turnover_rate','volatility_20', f'ma{params["MA_TREND_PERIOD"]}']
    
    st.dataframe(fdf[final_display_cols], use_container_width=True)

    out_csv = fdf_full[display_cols_full].head(200).to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}_V13_3.csv", mime="text/csv")

    st.markdown("### 小结与操作提示（V13.3 日期修正版）")
    st.markdown(f"""
- **【市值范围】** 流通市值已收紧到 **{params['MIN_CIRC_MV_Billion']} 亿 到 {params['MAX_CIRC_MV_Billion']} 亿** 之间。
- **【风控已修复】** 次新股（上市不足 {params['MIN_LIST_DAYS']} 天）已被强制排除。
- **【评分权重】** **低波动 (45%)**、MACD (35%)、Log-流动性 (20%)。
""")


# ----------------------------------------------------
# 主程序控制逻辑
# ----------------------------------------------------
params = {
    'FINAL_POOL': FINAL_POOL, 'TOP_DISPLAY': TOP_DISPLAY,
    'MIN_PRICE': MIN_PRICE, 'MAX_PRICE': MAX_PRICE, 'MIN_TURNOVER': MIN_TURNOVER,
    'MIN_AMOUNT': MIN_AMOUNT, 'VOLATILITY_MAX': VOLATILITY_MAX,
    'HIGH_PCT_THRESHOLD': HIGH_PCT_THRESHOLD,
    'MIN_CIRC_MV_Billion': MIN_CIRC_MV_Billion,
    'MAX_CIRC_MV_Billion': MAX_CIRC_MV_Billion,
    'MA_TREND_PERIOD': MA_TREND_PERIOD,
    'MIN_LIST_DAYS': MIN_LIST_DAYS
}

if st.session_state.get('run_backtest', False):
    run_simple_backtest(BACKTEST_DAYS, params)
    
elif st.session_state.get('run_selection', False):
    run_live_selection(last_trade, params)
    
else:
    st.info("请点击上方的按钮开始运行。")
