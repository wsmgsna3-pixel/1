# -*- coding: utf-8 -*-
"""
选股王 · 全市场扫描增强版 V3.9.2 (最终稳定版)
更新说明：
1. 【**功能升级**】：将股价、成交额、换手率等过滤参数移至侧边栏。
2. 【**修复 V3.9.1**】：修复了 get_future_prices 函数和主函数中收益计算的致命 bug。
3. 【**修复 V3.9.2**】：在最终汇总计算时，增加了收益过滤机制（自动剔除 >50% 或 <-50% 的异常 Tushare 数据），确保平均收益结果真实可靠。
4. 【**策略保持**】：核心 V3.7 权重 (极致保守) 保持不变。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · V3.9.2 最终稳定版", layout="wide")
st.title("选股王 · V3.9.2 最终稳定版（灵活过滤与多日验证）")
st.markdown("🚀 **当前版本已集成收益过滤，确保回测结果的真实性。**")

# ---------------------------
# 辅助函数 (移除了 @st.cache_data)
# ---------------------------
def safe_get(func, **kwargs):
    """安全调用 Tushare API"""
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception:
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    """获取 num_days 个交易日作为选股日"""
    
    # 获取一个较长时间范围内的交易日历
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 2)).strftime("%Y%m%d")
    cal = safe_get(ts.pro_api().trade_cal, start_date=start_date, end_date=end_date_str)
    
    if cal.empty or 'is_open' not in cal.columns:
        st.error("无法获取交易日历，请检查 Token 或 Tushare 权限。")
        return []

    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    
    # 过滤掉结束日期之后的日期（如果用户选择了未来日期）
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    
    # 取最近的 num_days 个交易日作为选股日
    return trade_days_df['cal_date'].head(num_days).tolist()

# ----------------------------------------------------
# ⚠️ 修复后的未来价格获取函数 (V3.9.1)
# ----------------------------------------------------
def get_future_prices(ts_code, selection_date, days_ahead=[1, 3, 5]):
    """拉取选股日之后 N 个交易日的收盘价，用于回测 (V3.9.1 修复版)"""
    
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_date = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date = (d0 + timedelta(days=15)).strftime("%Y%m%d")

    # 1. 尝试从日线数据拉取未来价格
    hist = safe_get(ts.pro_api().daily, ts_code=ts_code, start_date=start_date, end_date=end_date)
    
    if hist.empty or 'trade_date' not in hist.columns:
        results = {}
        for n in days_ahead: results[f'Return_D{n}'] = np.nan
        return results
    
    # 2. 确保价格数据是数值类型
    hist['close'] = pd.to_numeric(hist['close'], errors='coerce')
    hist = hist.dropna(subset=['close'])
    
    hist = hist.sort_values('trade_date').reset_index(drop=True)
    
    results = {}
    
    for n in days_ahead:
        col_name = f'Return_D{n}'
        # 3. 严格检查是否有足够的交易日数据
        if len(hist) >= n:
            future_price = hist.iloc[n-1]['close']
            if future_price == 0: # 避免除以零或异常低价
                results[col_name] = np.nan 
            else:
                results[col_name] = future_price
        else:
            results[col_name] = np.nan

    return results
# ----------------------------------------------------


def compute_indicators(df):
    """计算 MACD, 10日回报, 波动率, 60日位置等指标"""
    res = {}
    if df.empty or len(df) < 3: return res
    close = df['close'].astype(float)
    res['last_close'] = close.iloc[-1]
    
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    else: res['macd_val'] = np.nan
        
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        res['vol_ratio'] = vols[-1] / (np.mean(vols[-6:-1]) + 1e-9)
    else: res['vol_ratio'] = np.nan
        
    res['10d_return'] = close.iloc[-1]/close.iloc[-10] - 1 if len(close)>=10 else 0
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

# ---------------------------
# 侧边栏参数 (V3.9 灵活配置)
# ---------------------------
with st.sidebar:
    st.header("模式与日期选择")
    backtest_date_end = st.date_input(
        "选择**回测结束日期**", 
        value=datetime.now().date(), 
        max_value=datetime.now().date()
    )
    BACKTEST_DAYS = int(st.number_input(
        "**自动回测天数 (N)**", 
        value=5, # 默认设为5天，方便观察
        step=1, 
        min_value=1, 
        max_value=50, 
        help="程序将自动回测最近 N 个交易日。"
    ))
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = int(st.number_input("最终入围评分数量 (M)", value=50, step=10, min_value=10)) # 默认为50，保障稳定
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=10, step=1))
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=3, step=1, min_value=1)) # 默认设为3
    
    st.markdown("---")
    st.header("🛒 灵活过滤条件 (V3.9)")
    
    # 股价区间 (用户要求 10-300)
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0, step=0.5, min_value=0.1)
    MAX_PRICE = st.number_input("最高股价 (元)", value=300.0, step=5.0, min_value=1.0)
    
    # 最低换手率
    MIN_TURNOVER = st.number_input("最低换手率 (%)", value=3.0, step=0.5, min_value=0.1)
    
    # 最低成交额 (用户要求 20亿市值，故改为 0.6 亿)
    MIN_AMOUNT_MILLIONS = st.number_input("最低成交额 (亿元)", value=0.6, step=0.1, min_value=0.1)
    MIN_AMOUNT = MIN_AMOUNT_MILLIONS * 100000000 
    st.markdown(f"> *当前设置下，最低流通市值约为：{(MIN_AMOUNT/100000000)/ (MIN_TURNOVER/100):.1f} 亿*")

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
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT):
    """为单个交易日运行选股和回测逻辑"""
    
    # 1. 拉取全市场 Daily 数据
    daily_all = safe_get(pro.daily, trade_date=last_trade) 
    if daily_all.empty or 'ts_code' not in daily_all.columns:
        return pd.DataFrame(), f"数据缺失或拉取失败：{last_trade}"

    pool_raw = daily_all.reset_index(drop=True) 

    # 2. 合并基本面数据
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry')
    REQUIRED_BASIC_COLS = ['ts_code','turnover_rate','amount']
    daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields=','.join(REQUIRED_BASIC_COLS))
    mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)
    pool_merged = pool_raw.copy()

    if not stock_basic.empty and 'name' in stock_basic.columns:
        pool_merged = pool_merged.merge(stock_basic[['ts_code','name']], on='ts_code', how='left')
    else:
        pool_merged['name'] = pool_merged['ts_code']

    if not daily_basic.empty:
        cols_to_merge = [c for c in REQUIRED_BASIC_COLS if c in daily_basic.columns]
        if len(cols_to_merge) > 1:
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

    # 3. 执行硬性条件过滤 (使用侧边栏参数)
    df = pool_merged.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000 # 转换为万元
    df['name'] = df['name'].astype(str)
    
    # 过滤规则 (使用侧边栏传入的参数)
    mask_st = df['name'].str.contains('ST|退', case=False, na=False)
    df = df[~mask_st]
    mask_price = (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)
    df = df[mask_price]
    mask_turn = df['turnover_rate'] >= MIN_TURNOVER
    df = df[mask_turn]
    mask_amt = df['amount'] * 1000 >= MIN_AMOUNT # 确保使用传入的 MIN_AMOUNT
    df = df[mask_amt]
    df = df.reset_index(drop=True)

    if len(df) == 0:
        return pd.DataFrame(), f"过滤后无股票：{last_trade}"

    # 4. 遴选决赛名单 (保持 V3.8 逻辑)
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
            'pct_chg': getattr(row, 'pct_chg', 0), 'turnover': getattr(row, 'turnover_rate', 0),
            'net_mf': getattr(row, 'net_mf', 0)
        }
        hist = safe_get(pro.daily, ts_code=ts_code, end_date=last_trade) 
        ind = compute_indicators(hist)
        rec.update({
            'vol_ratio': ind.get('vol_ratio', 0), 'macd': ind.get('macd_val', 0),
            '10d_return': ind.get('10d_return', 0),
            'volatility': ind.get('volatility', 0), 'position_60d': ind.get('position_60d', np.nan)
        })
        
        rec['selection_price'] = ind.get('last_close', np.nan)
        future_prices = get_future_prices(ts_code, last_trade)
        
        # ⚠️ 修复后的收益计算逻辑 (V3.9.1)
        for n in [1, 3, 5]: 
            future_price = future_prices.get(f'Return_D{n}', np.nan)
            
            # 防御性检查：确保 P0 > 0.01 且价格不为 NaN
            if pd.notna(rec['selection_price']) and pd.notna(future_price) and rec['selection_price'] > 0.01:
                rec[f'Return_D{n}'] = (future_price / rec['selection_price'] - 1) * 100
            else: 
                rec[f'Return_D{n}'] = np.nan # 价格异常或数据缺失，标记为 NaN

        records.append(rec)
    
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame(), f"评分列表为空：{last_trade}"

    # 6. 归一化与 V3.7 评分 (权重保持不变)
    def normalize(series):
        series_nn = series.dropna() 
        if series_nn.max() == series_nn.min(): return pd.Series([0.5] * len(series), index=series.index)
        return (series - series_nn.min()) / (series_nn.max() - series_nn.min() + 1e-9)

    fdf['s_pct'] = normalize(fdf['pct_chg'])
    fdf['s_turn'] = normalize(fdf['turnover'])
    fdf['s_vol'] = normalize(fdf['vol_ratio'])
    fdf['s_mf'] = normalize(fdf['net_mf'])
    fdf['s_macd'] = normalize(fdf['macd'])
    fdf['s_trend'] = normalize(fdf['10d_return'])
    fdf['s_position'] = fdf['position_60d'] / 100 

    # V3.7 极致保守权重配置
    w_pct = 0.05; w_turn = 0.05; w_vol = 0.05; w_mf = 0.05; w_macd = 0.05; w_trend = 0.15      
    w_volatility = 0.30; w_position = 0.35   
    
    score = (
        fdf['s_pct'] * w_pct + fdf['s_turn'] * w_turn + fdf['s_vol'] * w_vol + fdf['s_mf'] * w_mf +        
        fdf['s_macd'] * w_macd + fdf['s_trend'] * w_trend +     
        (1 - normalize(fdf['volatility'])) * w_volatility + 
        (1 - fdf['s_position']) * w_position                
    )
    fdf['综合评分'] = score * 100
    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    fdf.index += 1

    # 返回 Top K 的回测结果
    return fdf.head(TOP_BACKTEST).copy(), None

# ---------------------------
# 主运行块
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测"):
    
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
        
        # 运行单日回测
        daily_result_df, error = run_backtest_for_a_day(
            trade_date, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT
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
    
    # 最终汇总计算
    st.header(f"📊 最终平均回测结果 (Top {TOP_BACKTEST}，共 {total_days} 个交易日)")
    
    # V3.9.2 最终修复：引入收益过滤机制 (剔除超过 50% 的异常值)
    for n in [1, 3, 5]:
        col = f'Return_D{n}'
        
        # 1. 复制数据，用于安全过滤
        filtered_returns = all_results.copy()
        
        # 2. 移除 NaN 值，确保只对有效数据进行操作
        valid_returns = filtered_returns.dropna(subset=[col])

        # 3. 过滤异常值：收益率必须在 -50% 到 50% 之间（排除不可能的 Tushare 错误数据）
        if not valid_returns.empty:
            valid_returns = valid_returns[
                (valid_returns[col] > -50) & 
                (valid_returns[col] < 50)
            ]
            avg_return = valid_returns[col].mean()
            
            # 重新计算准确率 (基于过滤后的数据)
            hit_rate = (valid_returns[col] > 0).sum() / len(valid_returns) * 100
            total_count = len(valid_returns)
        else:
            avg_return = np.nan
            hit_rate = 0.0
            total_count = 0
            
        st.metric(f"Top {TOP_BACKTEST}：D+{n} 平均收益 / 准确率", 
                  f"{avg_return:.2f}% / {hit_rate:.1f}%", 
                  help=f"总有效样本数：{total_count}。收益已剔除 >50% 或 <-50% 的异常数据。")

    st.header("📋 每日回测详情 (Top K 明细)")
    st.dataframe(all_results[['Trade_Date', 'name', 'ts_code', '综合评分', 'selection_price', 'Return_D1', 'Return_D3', 'Return_D5']].sort_values('Trade_Date', ascending=False), use_container_width=True)

