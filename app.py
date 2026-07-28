# -*- coding: utf-8 -*-
"""
翻倍黑马归类统计器 · 双重确认版 (带洗盘深度测算)
------------------------------------------------
采用“绝对空间(≥18%) + 结构破坏(≥12%且破10周线)”的双保险机制来识别真实波浪
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
# 1. 页面配置与侧边栏 UI
# ---------------------------
st.set_page_config(page_title="50周翻倍股票深度测算器 (双重确认版)", layout="wide")

with st.sidebar:
    st.header("⚙️ 参数设置")
    TS_TOKEN = st.text_input("🔑 Tushare Token", type="password")
    MIN_PRICE = st.number_input("最低股价限制 (元)", value=20.0)
    MIN_MV = st.number_input("最小市值 (亿元)", value=100.0) 
    MAX_MV = st.number_input("最大市值 (亿元)", value=1000.0)

st.title("📊 近50周翻倍股票测算 (波浪双重确认版)")

if not TS_TOKEN:
    st.info("👈 请先在左侧边栏输入您的 Tushare Token 以启动程序。")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 2. 数据获取与合成 
# ---------------------------
@st.cache_data(ttl=3600*12)
def safe_get(_pro, func_name, **kwargs):
    if _pro is None: return pd.DataFrame()
    func = getattr(_pro, func_name)
    try:
        for _ in range(3):
            df = func(**kwargs)
            if df is not None and not df.empty: return df
            time.sleep(0.3)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def get_stock_weekly_data(_pro, ts_code, start_date, end_date):
    df = safe_get(_pro, 'daily', ts_code=ts_code, start_date=start_date, end_date=end_date)
    adj = safe_get(_pro, 'adj_factor', ts_code=ts_code, start_date=start_date, end_date=end_date)
    
    if df.empty or adj.empty: return pd.DataFrame()
    
    df = df.merge(adj, on=['ts_code', 'trade_date'], how='left').sort_values('trade_date')
    latest_factor = df['adj_factor'].iloc[-1]
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] * df['adj_factor'] / latest_factor
        
    df['dt'] = pd.to_datetime(df['trade_date'])
    iso_cal = df['dt'].dt.isocalendar()
    df['year_week'] = iso_cal.year.astype(str) + "_" + iso_cal.week.astype(str).str.zfill(2)
    
    weekly = df.groupby('year_week', as_index=False).agg({
        'trade_date': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum'
    }).sort_values('trade_date').reset_index(drop=True)
    
    # 【新增】：为双重确认机制计算 10 周均线
    if len(weekly) >= 10:
        weekly['w_ma10'] = weekly['close'].rolling(10).mean()
    else:
        weekly['w_ma10'] = np.nan
        
    return weekly

# ---------------------------
# 3. 形态判断核心算法 (带双重确认机制)
# ---------------------------
def classify_double_pattern(weekly_df):
    if len(weekly_df) < 10: return None
    
    min_idx = weekly_df['low'].idxmin()
    sub_df = weekly_df.loc[min_idx:]
    if len(sub_df) < 2: return None
    
    max_idx = sub_df['high'].idxmax()
    min_price = weekly_df.loc[min_idx, 'low']
    max_price = sub_df.loc[max_idx, 'high']
    
    if min_price <= 0 or (max_price / min_price) < 2.0:
        return None
        
    weeks_taken = max_idx - min_idx + 1 
    ascent_df = weekly_df.loc[min_idx:max_idx].copy().reset_index(drop=True)
    
    running_max = ascent_df['high'].iloc[0]
    in_pullback = False
    
    drawdowns = []
    current_max_drawdown = 0.0
    
    for i in range(1, len(ascent_df)):
        curr_high = ascent_df.loc[i, 'high']
        curr_low = ascent_df.loc[i, 'low']
        curr_close = ascent_df.loc[i, 'close']
        curr_ma10 = ascent_df.loc[i, 'w_ma10']
        
        if curr_high > running_max:
            running_max = curr_high
            if in_pullback:
                drawdowns.append(current_max_drawdown)
                in_pullback = False
                current_max_drawdown = 0.0
        else:
            drawdown = (running_max - curr_low) / running_max
            
            # 【核心修改】：波浪双重确认机制
            # 条件1：绝对空间跌幅 >= 18%
            # 条件2：跌幅 >= 12% 且 当周收盘价向下跌破 10 周均线
            is_valid_pullback = False
            if drawdown >= 0.18:
                is_valid_pullback = True
            elif drawdown >= 0.12 and pd.notna(curr_ma10) and curr_close < curr_ma10:
                is_valid_pullback = True

            if is_valid_pullback:
                in_pullback = True
                if drawdown > current_max_drawdown:
                    current_max_drawdown = drawdown 
                
    if in_pullback and current_max_drawdown > 0:
        drawdowns.append(current_max_drawdown)
        
    if weeks_taken <= 6 or len(drawdowns) <= 1:
        pattern = "⚡ 潜伏爆破型 (形态一)"
    else:
        pattern = "🌊 波浪推进型 (形态二)"
        
    avg_drop = (sum(drawdowns) / len(drawdowns) * 100) if drawdowns else 0.0
    max_drop = (max(drawdowns) * 100) if drawdowns else 0.0
        
    return {
        'Min_Price': round(min_price, 2),
        'Max_Price': round(max_price, 2),
        'Max_Gain (%)': round((max_price - min_price) / min_price * 100, 1),
        'Weeks_Taken': weeks_taken,
        'Pullback_Count': len(drawdowns),
        'Avg_Drop (%)': round(avg_drop, 2), 
        'Max_Drop (%)': round(max_drop, 2), 
        'Pattern': pattern
    }

# ---------------------------
# 4. 主干回测与展示逻辑
# ---------------------------
if st.button("🚀 开始测算 (采用最新双重确认机制)"):
    today_str = datetime.now().strftime("%Y%m%d")
    start_date_str = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    
    with st.spinner("正在获取基础股票列表..."):
        basic = safe_get(pro, 'stock_basic', list_status='L', fields='ts_code,name,market')
        daily_basic = safe_get(pro, 'daily_basic', trade_date=today_str)
        if daily_basic.empty:
            for d in range(1, 10):
                prev_date = (datetime.now() - timedelta(days=d)).strftime("%Y%m%d")
                daily_basic = safe_get(pro, 'daily_basic', trade_date=prev_date)
                if not daily_basic.empty: break
                
    if basic.empty or daily_basic.empty:
        st.error("无法获取股票基础数据，请检查 Token 或网络状态。")
        st.stop()
        
    df_merged = basic.merge(daily_basic[['ts_code', 'close', 'circ_mv']], on='ts_code', how='inner')
    df_merged['circ_mv_billion'] = df_merged['circ_mv'] / 10000
    
    filtered_stocks = df_merged[
        (~df_merged['name'].str.contains('ST|退', na=False)) &
        (df_merged['close'] >= MIN_PRICE) &
        (df_merged['circ_mv_billion'] >= MIN_MV) &
        (df_merged['circ_mv_billion'] <= MAX_MV)
    ]
    
    st.write(f"🔍 符合过滤条件（股价≥{MIN_PRICE}元，市值{MIN_MV}-{MAX_MV}亿）的股票共 **{len(filtered_stocks)}** 只，开始执行结构扫描...")
    
    results = []
    progress_bar = st.progress(0)
    
    for idx, row in enumerate(filtered_stocks.itertuples()):
        weekly = get_stock_weekly_data(pro, row.ts_code, start_date_str, today_str)
        if not weekly.empty:
            analysis = classify_double_pattern(weekly)
            if analysis:
                analysis['ts_code'] = row.ts_code
                analysis['name'] = row.name
                results.append(analysis)
                
        progress_bar.progress((idx + 1) / len(filtered_stocks))
    progress_bar.empty()
    
    if results:
        res_df = pd.DataFrame(results)
        
        p1_count = len(res_df[res_df['Pattern'].str.contains('潜伏爆破')])
        p2_count = len(res_df[res_df['Pattern'].str.contains('波浪推进')])
        total = len(res_df)
        
        st.markdown("---")
        st.header("📈 统计结果汇总 (双重确认机制)")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("近50周翻倍股票总数", f"{total} 只")
        c2.metric("⚡ 潜伏爆破型 (形态一)", f"{p1_count} 只", f"{p1_count/total*100:.1f}%")
        c3.metric("🌊 波浪推进型 (形态二)", f"{p2_count} 只", f"{p2_count/total*100:.1f}%")
        
        st.subheader("📋 详细翻倍股票轨迹清单 (含洗盘深度与次数精准测算)")
        
        display_df = res_df[['ts_code', 'name', 'Pattern', 'Weeks_Taken', 'Pullback_Count', 
                             'Avg_Drop (%)', 'Max_Drop (%)', 'Max_Gain (%)', 'Min_Price', 'Max_Price']]
        
        display_df.columns = ['代码', '名称', '归类形态', '翻倍耗时(周)', '严选期间洗盘次数', 
                              '平均每次洗盘跌幅(%)', '极限单次洗盘跌幅(%)', '最大总涨幅(%)', '区间最低价', '区间最高价']
        
        st.dataframe(display_df.sort_values('翻倍耗时(周)').reset_index(drop=True), use_container_width=True)
        
        csv = display_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载深度分析明细 (CSV)", csv, "double_stocks_dual_confirm_analysis.csv", "text/csv")
    else:
        st.warning("⚠️ 在当前筛选条件下，近 50 周内没有扫描到翻倍的股票。")
