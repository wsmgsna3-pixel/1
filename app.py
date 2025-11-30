# -*- coding: utf-8 -*-
"""
选股王 · V4.1e 安全低位蓄势版
更新说明：
1. 【**策略精调 V4.1e**】：核心变动：将短期趋势指标（10日回报）的权重从 0.25 降至 0.10；将长期安全边际（60日位置）的权重从 0.15 提高到 0.30。
   目标：彻底规避“买在短期高位末班车”的风险，优先选择在60日内处于较低位置、趋势稳定且活跃的股票。
2. 【**过滤 V4.1b**】：继续使用手动复权和市值硬过滤。
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
st.set_page_config(page_title="选股王 · V4.1e 安全低位蓄势版", layout="wide")
st.title("选股王 · V4.1e 安全低位蓄势版（低位安全边际主导）")
st.markdown("✅ **V4.1e 策略：已大幅提高 60 日位置（低位）权重，旨在规避短期暴涨末班车。**")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None 

# ---------------------------
# 辅助函数 
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    """安全调用 Tushare API (针对 pro 接口，如 daily, daily_basic, adj_factor)"""
    global pro
    if pro is None:
        return pd.DataFrame(columns=['ts_code']) 
        
    func = getattr(pro, func_name) 
    
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            time.sleep(0.5) 
            return pd.DataFrame(columns=['ts_code']) 
            
        time.sleep(0.5) 
        
        return df
    except Exception as e:
        time.sleep(0.5) 
        return pd.DataFrame(columns=['ts_code'])

# 此函数无需缓存
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


# ----------------------------------------------------
# 关键函数：手动复权数据源 (V4.0 核心)
# ----------------------------------------------------
@st.cache_data(ttl=3600*24) # 缓存24小时
def get_adj_factor(ts_code, start_date, end_date):
    """获取指定时间段的复权因子"""
    df = safe_get('adj_factor', ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty or 'adj_factor' not in df.columns:
        return pd.DataFrame()
    df['adj_factor'] = pd.to_numeric(df['adj_factor'], errors='coerce').fillna(0)
    df = df.set_index('trade_date').sort_index() 
    return df['adj_factor']

@st.cache_data(ttl=3600*12) # 缓存12小时
def get_qfq_data_v4(ts_code, start_date, end_date):
    """V4.0 核心：手动计算前复权价格"""
    
    # 1. 获取未复权价格数据 (使用 daily 接口)
    daily_df = safe_get('daily', ts_code=ts_code, start_date=start_date, end_date=end_date)
    if daily_df.empty: return pd.DataFrame()

    daily_df = daily_df.set_index('trade_date').sort_index()
    
    # 2. 获取复权因子
    adj_factor_series = get_adj_factor(ts_code, start_date, end_date)
    if adj_factor_series.empty: return pd.DataFrame()

    # 3. 合并数据
    df = daily_df.merge(adj_factor_series.rename('adj_factor'), 
                        left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    
    if df.empty: return pd.DataFrame()

    latest_adj_factor = df['adj_factor'].iloc[-1]
    
    # 4. 手动计算前复权价格
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            if latest_adj_factor > 1e-9:
                df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
            else:
                df[col + '_qfq'] = df[col] 
            
    # 5. 清理并保留需要的 QFQ 价格，并使用 trade_date 排序
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df['trade_date'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')
    df = df.sort_values('trade_date').set_index('trade_date_str')

    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col + '_qfq']
    
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 
# ----------------------------------------------------


# ----------------------------------------------------
# 关键函数 1：获取未来价格 
# ----------------------------------------------------
def get_future_prices(ts_code, selection_date, days_ahead=[1, 3, 5]):
    """拉取选股日之后 N 个交易日的复权收盘价，用于回测"""
    
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_date = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date = (d0 + timedelta(days=15)).strftime("%Y%m%d")

    hist = get_qfq_data_v4(ts_code, start_date=start_date, end_date=end_date)
    
    if hist.empty or 'close' not in hist.columns:
        results = {}
        for n in days_ahead: results[f'Return_D{n}'] = np.nan
        return results
    
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    hist = hist.dropna(subset=['close'])
    hist = hist.reset_index(drop=True) 
    
    results = {}
    
    for n in days_ahead:
        col_name = f'Return_D{n}'
        if len(hist) >= n:
            future_price = hist.iloc[n-1]['close']
            if future_price == 0: 
                results[col_name] = np.nan 
            else:
                results[col_name] = future_price
        else:
            results[col_name] = np.nan

    return results
# ----------------------------------------------------


# ----------------------------------------------------
# 关键函数 2：计算指标 
# ----------------------------------------------------
@st.cache_data(ttl=3600*12) # 缓存12小时
def compute_indicators(ts_code, end_date):
    """计算 MACD, 10日回报, 波动率, 60日位置等指标 (V4.0 使用 get_qfq_data_v4)"""
    
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    
    df = get_qfq_data_v4(ts_code, start_date=start_date, end_date=end_date)
    
    res = {}
    
    if df.empty or len(df) < 3 or 'close' not in df.columns: 
        return res
    
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
# 侧边栏参数 (定义 BACKTEST_DAYS 等变量)
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input(
        "选择**回测结束日期**", 
        value=datetime.now().date(), 
        max_value=datetime.now().date()
    )
    BACKTEST_DAYS = int(st.number_input(
        "**自动回测天数 (N)**", 
        value=5, 
        step=1, 
        min_value=1, 
        max_value=50, 
        help="程序将自动回测最近 N 个交易日。"
    ))
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = int(st.number_input("最终入围评分数量 (M)", value=10, step=1, min_value=1)) 
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=10, step=1))
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=3, step=1, min_value=1)) 
    
    st.markdown("---")
    st.header("🛒 灵活过滤条件")
    
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0, step=0.5, min_value=0.1)
    MAX_PRICE = st.number_input("最高股价 (元)", value=300.0, step=5.0, min_value=1.0)
    MIN_TURNOVER = st.number_input("最低换手率 (%)", value=3.0, step=0.5, min_value=0.1)
    
    # V4.1b 新增：最低流通市值
    MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿元)", value=20.0, step=1.0, min_value=1.0, help="例如：输入 20 代表流通市值必须大于等于 20 亿元。")

    MIN_AMOUNT_MILLIONS = st.number_input("最低成交额 (亿元)", value=0.6, step=0.1, min_value=0.1)
    MIN_AMOUNT = MIN_AMOUNT_MILLIONS * 100000000 
    
    st.markdown(f"> *提示：最低成交额/最低换手率的组合筛选，仍是一种强大的活跃度过滤方法。*")

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
# 核心回测逻辑函数 
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS):
    """为单个交易日运行选股和回测逻辑"""
    
    # 1. 拉取全市场 Daily 数据
    daily_all = safe_get('daily', trade_date=last_trade) 
    if daily_all.empty or 'ts_code' not in daily_all.columns:
        return pd.DataFrame(), f"数据缺失或拉取失败：{last_trade}"

    pool_raw = daily_all.reset_index(drop=True) 

    # 2. 合并基本面数据
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
    
    # 过滤 ST 股/退市股/北交所/次新股 (逻辑不变)
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

    # 过滤条件 (V4.1b 增加市值过滤)
    
    # 过滤价格
    mask_price = (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)
    df = df[mask_price]
    
    # V4.1b 过滤流通市值
    mask_circ_mv = df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS
    df = df[mask_circ_mv] 
    
    # 过滤换手率
    mask_turn = df['turnover_rate'] >= MIN_TURNOVER
    df = df[mask_turn]
    # 过滤成交额
    mask_amt = df['amount'] * 1000 >= MIN_AMOUNT
    df = df[mask_amt]
    
    df = df.reset_index(drop=True)

    if len(df) == 0:
        return pd.DataFrame(), f"过滤后无股票：{last_trade}"

    # 4. 遴选决赛名单 (逻辑不变)
    limit_pct = int(FINAL_POOL * 0.7)
    df_pct = df.sort_values('pct_chg', ascending=False).head(limit_pct).copy()
    limit_turn = FINAL_POOL - len(df_pct)
    existing_codes = set(df_pct['ts_code'])
    df_turn = df[~df['ts_code'].isin(existing_codes)].sort_values('turnover_rate', ascending=False).head(limit_turn).copy()
    final_candidates = pd.concat([df_pct, df_turn]).reset_index(drop=True)

    # 5. 深度评分 (逻辑不变)
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
        
        ind = compute_indicators(ts_code, last_trade)
        rec.update({
            'vol_ratio': ind.get('vol_ratio', 0), 'macd': ind.get('macd_val', 0),
            '10d_return': ind.get('10d_return', 0),
            'volatility': ind.get('volatility', 0), 'position_60d': ind.get('position_60d', np.nan)
        })
        
        selection_price_adj = ind.get('last_close', np.nan) 
        
        future_prices = get_future_prices(ts_code, last_trade)
        
        for n in [1, 3, 5]: 
            future_price = future_prices.get(f'Return_D{n}', np.nan)
            
            if pd.notna(selection_price_adj) and pd.notna(future_price) and selection_price_adj > 0.01:
                rec[f'Return_D{n} (%)'] = (future_price / selection_price_adj - 1) * 100
            else: 
                rec[f'Return_D{n} (%)'] = np.nan 

        records.append(rec)
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), f"评分列表为空：{last_trade}"

    # 6. 归一化与 V4.1e 策略精调评分 (强化安全低位蓄势)
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
    fdf['s_position'] = fdf['position_60d'] / 100 

    # 🚨 V4.1e 策略精调：强化安全低位蓄势信号 (防末班车)
    
    # 安全/防御指标：总权重 50%
    w_position = 0.30   # 30% - 60日位置 (低位置得分高 = 核心安全边际 - **大幅提高** $\text{0.15} \to \text{0.30}$)
    w_volatility = 0.20 # 20% - 波动率 (低波动率得分高 = 趋势稳定，抗跌性)
    
    # 趋势/动量/活跃度指标：总权重 50%
    w_trend = 0.10      # 10% - 10日回报 (核心趋势强度 - **大幅降低** $\text{0.25} \to \text{0.10}$)
    w_macd = 0.10       # 10% - MACD (趋势信号确认)
    w_turn = 0.10       # 10% - 换手率 (活跃度)
    w_mf = 0.10         # 10% - 资金流 (主力动向)
    w_pct = 0.05        # 5% - 当日涨幅 (右侧启动的第一信号 - 保持低位)
    w_vol = 0.05        # 5% - 量比 
    
    # Sum: 0.30+0.20 + 0.10+0.10+0.10+0.10+0.05+0.05 = 1.00
    
    score = (
        fdf['s_pct'] * w_pct + fdf['s_turn'] * w_turn + fdf['s_vol'] * w_vol + fdf['s_mf'] * w_mf +        
        fdf['s_macd'] * w_macd + fdf['s_trend'] * w_trend +     
        (1 - normalize(fdf['volatility'])) * w_volatility + 
        (1 - fdf['s_position']) * w_position                
    )
    fdf['综合评分'] = score * 100
    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index += 1

    return fdf.head(TOP_BACKTEST).copy(), None

# ---------------------------
# 主运行块 
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测"):
    
    st.warning("⚠️ **V4.1e 版本已更换为安全低位蓄势策略，请清除 Streamlit 缓存后运行，以使用新的权重计算评分。**")
    
    trade_days_str = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)
    if not trade_days_str:
        st.error("无法获取交易日列表，请检查日期或 Token。")
        st.stop()
    
    st.header(f"📈 正在进行 {BACKTEST_DAYS} 个交易日的回测...")
    
    results_list = []
    total_days = len(trade_days_str)
    
    progress_text = st.empty()
    my_bar = st.progress(0)
    
    for i, trade_date in enumerate(trade_days_str):
        progress_text.text(f"🚀 正在处理第 {i+1}/{total_days} 个交易日：{trade_date}")
        
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
                  help=f"总有效样本数：{total_count}。**V4.1e 已应用安全低位策略。**")

    st.header("📋 每日回测详情 (Top K 明细)")
    
    display_cols = ['Trade_Date', 'name', 'ts_code', '综合评分', 
                    'Close', 'Pct_Chg (%)', 'Circ_MV (亿)',
                    'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)']
    
    st.dataframe(all_results[display_cols].sort_values('Trade_Date', ascending=False), use_container_width=True)
