# -*- coding: utf-8 -*-
"""
选股王 · V10.2 多周期并列回测版 (T+1, T+3, T+5 同时展示)

说明：
1. 【回测升级】一次运行，同时展示 T+1, T+3, T+5 收益率，便于横向比较。
2. 【价格逻辑】买入价：T+1 开盘价；卖出价：T+N 收盘价。
3. 【稳定性】保留 V10.0 的数据冗余和容错机制。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import joblib 
import os

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
st.set_page_config(page_title="选股王（V10.2 多周期并列回测版）", layout="wide")
st.markdown("### 选股王（V10.2 多周期并列回测版）") 

# ---------------------------
# 侧边栏参数（移除 HOLDING_DAYS）
# ---------------------------
with st.sidebar:
    st.header("可调参数（实时）")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=500, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=0.5, step=0.1)) 
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=20_000_000.0, step=5_000_000.0))
    
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.4, step=0.1)) 
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=6.0, step=0.5)) 
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    
    st.markdown("---")
    
    # ***回测天数 N***
    BACKTEST_DAYS = int(st.number_input("回测：最近 N 个交易日", value=10, step=1))
    
    st.markdown("---")
    st.caption("提示：策略已升级至 'V10.2 多周期并列回测版'。")
    st.caption("回测将同时计算 T+1, T+3, T+5 收益。")

# ---------------------------
# Token 输入 & 依赖函数（省略，与 V10.1 相同）
# ---------------------------
# ... (TS_TOKEN, pro, safe_get, find_last_trade_day, last_trade, 按钮控制模块等与 V10.1 相同) ...
st.markdown("请输入 Tushare Token。")
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password", label_visibility="collapsed")

if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

# 初始化 tushare
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 依赖函数：数据安全获取和交易日查找
# ---------------------------
def safe_get(func, **kwargs):
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def find_last_trade_day(max_days=20):
    today = datetime.now().date()
    for i in range(max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        df = safe_get(pro.daily, trade_date=ds)
        if not df.empty:
            return ds
    return None

last_trade = find_last_trade_day()
if not last_trade:
    st.error("无法找到最近交易日，检查网络或 Token 权限。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")

# ----------------------------------------------------
# 按钮控制模块
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
# 指标计算和归一化（省略，与 V10.1 相同）
# ---------------------------
# ... (compute_indicators, safe_merge_pool, norm_col functions remain the same as V10.1) ...

# ----------------------------------------------------
# 核心评分函数（省略，与 V10.1 相同）
# ----------------------------------------------------
# ... (run_scoring_for_date function remains the same as V10.1) ...

# ----------------------------------------------------
# 简易回测模块 (V10.2 核心修改：实现 T+1, T+3, T+5 并列)
# ----------------------------------------------------
def run_simple_backtest(days, params):
    
    # ***V10.2 核心修改：定义需要回测的周期列表***
    HOLDING_PERIODS = [1, 3, 5] 
    
    status = st.session_state['backtest_status']
    
    container = st.empty()
    with container.container():
        st.subheader(f"📈 简易历史回测结果 (T+{', T+'.join(map(str, HOLDING_PERIODS))} 并列)")
        
        trade_dates_df = safe_get(pro.trade_cal, exchange='SSE', is_open='1', end_date=find_last_trade_day(), fields='cal_date')
        if trade_dates_df.empty:
            st.error("无法获取历史交易日历。")
            return

        max_holding = max(HOLDING_PERIODS)
        # 需要多拉取 max_holding 天来计算最长周期的收益
        trade_dates = trade_dates_df['cal_date'].sort_values(ascending=False).head(days + max_holding).tolist() 
        trade_dates.reverse() 
        # 最后一个选股日必须保证后面有 max_holding 天来计算收益
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
        
        # 传递给评分函数的参数 (不含回测专用参数)
        score_params = {
            'INITIAL_TOP_N': params['INITIAL_TOP_N'], 'FINAL_POOL': params['FINAL_POOL'], 'MIN_PRICE': params['MIN_PRICE'], 
            'MAX_PRICE': params['MAX_PRICE'], 'MIN_TURNOVER': params['MIN_TURNOVER'], 'MIN_AMOUNT': params['MIN_AMOUNT'], 
            'VOL_SPIKE_MULT': params['VOL_SPIKE_MULT'], 'VOLATILITY_MAX': params['VOLATILITY_MAX'], 
            'HIGH_PCT_THRESHOLD': params['HIGH_PCT_THRESHOLD']
        }
        
        for i in range(start_index, total_iterations):
            select_date = trade_dates[i]
            next_trade_date = trade_dates[i+1] # T+1 日，用于买入
            
            select_df_full = run_scoring_for_date(select_date, score_params) 

            # 初始化结果
            result = {
                '选股日': select_date, 
                '股票': '无符合条件', 
                '买入价 (T+1 开盘)': np.nan, 
                '评分': np.nan
            }
            for N in HOLDING_PERIODS:
                 result[f'T+{N} 收益率 (%)'] = 0.0
                 result[f'T+{N} 卖出价'] = np.nan
                 
            
            if not select_df_full.empty:
                top_pick = select_df_full.iloc[0] 
                ts_code = top_pick['ts_code']
                
                # --- 获取 T+1 买入价 ---
                buy_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=next_trade_date)
                buy_price = buy_day_data.iloc[0]['open'] if not buy_day_data.empty and 'open' in buy_day_data.columns else np.nan
                
                result['股票'] = f"{top_pick.get('name', 'N/A')}({ts_code})"
                result['买入价 (T+1 开盘)'] = buy_price
                result['评分'] = top_pick['综合评分']
                
                if buy_price > 0 and not pd.isna(buy_price):
                    
                    # --- 循环计算 T+N 收益 ---
                    for N in HOLDING_PERIODS:
                        sell_trade_date = trade_dates[i+N] # T+N 日
                        
                        sell_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=sell_trade_date)
                        
                        if not sell_day_data.empty and 'close' in sell_day_data.columns:
                            sell_price = sell_day_data.iloc[0]['close']
                            result[f'T+{N} 卖出价'] = sell_price
                            
                            if not pd.isna(sell_price):
                                return_pct = (sell_price / buy_price) - 1.0
                                result[f'T+{N} 收益率 (%)'] = return_pct * 100
                        
            # 5. 更新状态和进度条
            status['results'].append(result)
            status['current_index'] = i + 1
            status['progress'] = (i + 1) / total_iterations
            
            pbar.progress(status['progress'], text=f"正在回测 {select_date}... [{i+1}/{total_iterations}]")
            
            # 每 2 次迭代或回测完成时，刷新界面，实现断点续传效果
            if (i+1) % 2 == 0 or (i + 1) == total_iterations: 
                 st.rerun() 
        
        status['progress'] = 1.0
        status['current_index'] = total_iterations
        pbar.progress(1.0, text="回测完成。")
        
        # 6. 结果展示 (修改累计收益计算)
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
        
        # 整理显示列
        display_cols = ['选股日', '股票', '评分', '买入价 (T+1 开盘)']
        for N in HOLDING_PERIODS:
            display_cols.append(f'T+{N} 收益率 (%)')
            display_cols.append(f'T+{N} 卖出价')
            
        st.dataframe(results_df[display_cols], use_container_width=True)


# ----------------------------------------------------
# 实时选股模块 (省略，与 V10.1 相同)
# ----------------------------------------------------
# ... (run_live_selection function remains the same as V10.1) ...


# ----------------------------------------------------
# 主程序控制逻辑（省略，与 V10.1 相同）
# ----------------------------------------------------
params = {
    'INITIAL_TOP_N': INITIAL_TOP_N, 'FINAL_POOL': FINAL_POOL, 'TOP_DISPLAY': TOP_DISPLAY,
    'MIN_PRICE': MIN_PRICE, 'MAX_PRICE': MAX_PRICE, 'MIN_TURNOVER': MIN_TURNOVER,
    'MIN_AMOUNT': MIN_AMOUNT, 'VOL_SPIKE_MULT': VOL_SPIKE_MULT, 'VOLATILITY_MAX': VOLATILITY_MAX,
    'HIGH_PCT_THRESHOLD': HIGH_PCT_THRESHOLD
}

if st.session_state.get('run_backtest', False):
    run_simple_backtest(BACKTEST_DAYS, params)
    
elif st.session_state.get('run_selection', False):
    # 此处 run_live_selection 沿用 V9.0 的逻辑，需要补充其代码体
    # 鉴于 V10.1 已经提供 run_live_selection 的完整逻辑，此处假设它已存在于实际运行环境中
    pass 
    
else:
    st.info("请点击上方的按钮开始运行。")

