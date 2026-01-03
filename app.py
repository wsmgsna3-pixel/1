# -*- coding: utf-8 -*-
"""
选股王 · V30.12.3 Pro (最终修正版: T+3/T+5卖出)
------------------------------------------------
🔥 核心修正：
1. **卖出日期校准**：
   - D1: T+2 卖出 (受A股T+1限制，最早只能次日卖)
   - D3: T+3 卖出 (持股第3天)
   - D5: T+5 卖出 (持股第5天)
2. **仪表盘回归**：恢复 D1/D3/D5 的收益率和胜率看板。
3. **严格买入**：Open > Pre_Close 且 High > Open * 1.015。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import os
import pickle

warnings.filterwarnings("ignore")

# ---------------------------
# 全局配置 & 缓存
# ---------------------------
CACHE_DIR = "data_cache_2025"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

GLOBAL_DATA = {
    'daily': pd.DataFrame(),
    'daily_basic': pd.DataFrame(),
    'moneyflow': pd.DataFrame()
}
pro = None

st.set_page_config(page_title="选股王 最终版", layout="wide")

# ---------------------------
# 1. 基础函数
# ---------------------------
@st.cache_resource
def init_tushare(token):
    if not token: return None
    try:
        api = ts.pro_api(token)
        api.trade_cal(start_date='20250101', end_date='20250101')
        return api
    except Exception as e:
        st.error(f"Token 连接失败: {e}")
        return None

def get_real_trade_date(date_str):
    if pro is None: return date_str
    try:
        start = (datetime.strptime(date_str, '%Y%m%d') - timedelta(days=10)).strftime('%Y%m%d')
        end = date_str
        df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
        if not df.empty: return df['cal_date'].iloc[-1]
        return date_str
    except:
        return date_str

def get_trade_cal(start_date, end_date):
    if pro is None: return []
    try:
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
        df = df[df['is_open'] == 1]
        return sorted(df['cal_date'].tolist())
    except:
        return []

# ---------------------------
# 2. 数据拉取 (带缓存)
# ---------------------------
def fetch_and_cache(api_func, date, data_type, **kwargs):
    cache_file = os.path.join(CACHE_DIR, f"{date}_{data_type}.pkl")
    if os.path.exists(cache_file):
        try:
            df = pd.read_pickle(cache_file)
            if df is not None: return df, True
        except: os.remove(cache_file)
    
    for _ in range(3):
        try:
            df = api_func(**kwargs)
            if df is not None: 
                df.to_pickle(cache_file)
                return df, False
        except: time.sleep(1)
    return None, False

def prefetch_data_stable(trade_days):
    global pro, GLOBAL_DATA
    if not trade_days: return False
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    all_daily, all_basic, all_mf = [], [], []
    total_days = len(trade_days)
    cache_hits, net_hits = 0, 0
    
    for i, date in enumerate(trade_days):
        # Daily
        df_d, from_cache = fetch_and_cache(pro.daily, date, 'daily', trade_date=date)
        if df_d is not None and not df_d.empty: all_daily.append(df_d)
        
        # Basic
        df_b, _ = fetch_and_cache(pro.daily_basic, date, 'basic', trade_date=date, fields='ts_code,trade_date,turnover_rate,circ_mv,total_mv,pe,pb')
        if df_b is not None and not df_b.empty: all_basic.append(df_b)
            
        # Moneyflow
        df_m, _ = fetch_and_cache(pro.moneyflow, date, 'moneyflow', trade_date=date)
        if df_m is not None and not df_m.empty: all_mf.append(df_m)
        
        if from_cache: cache_hits += 1
        else:
            net_hits += 1
            time.sleep(0.05)
            
        progress_bar.progress((i + 1) / total_days, text=f"加载数据: {date} ({i+1}/{total_days})")

    status_text.info(f"数据就绪 | 缓存: {cache_hits} | 网络: {net_hits}")
    
    if all_daily:
        full_daily = pd.concat(all_daily)
        full_daily['trade_date'] = full_daily['trade_date'].astype(str).str.strip()
        full_daily['ts_code'] = full_daily['ts_code'].astype(str).str.strip()
        full_daily.drop_duplicates(subset=['trade_date', 'ts_code'], inplace=True)
        full_daily.set_index(['trade_date', 'ts_code'], inplace=True)
        full_daily.sort_index(inplace=True)
        GLOBAL_DATA['daily'] = full_daily
    else: return False
        
    if all_basic:
        full_basic = pd.concat(all_basic)
        full_basic['trade_date'] = full_basic['trade_date'].astype(str).str.strip()
        full_basic['ts_code'] = full_basic['ts_code'].astype(str).str.strip()
        full_basic.drop_duplicates(subset=['trade_date', 'ts_code'], inplace=True)
        full_basic.set_index(['trade_date', 'ts_code'], inplace=True)
        full_basic.sort_index(inplace=True)
        GLOBAL_DATA['daily_basic'] = full_basic
        
    if all_mf:
        full_mf = pd.concat(all_mf)
        full_mf['trade_date'] = full_mf['trade_date'].astype(str).str.strip()
        full_mf['ts_code'] = full_mf['ts_code'].astype(str).str.strip()
        full_mf.set_index(['trade_date', 'ts_code'], inplace=True)
        full_mf.sort_index(inplace=True)
        GLOBAL_DATA['moneyflow'] = full_mf

    status_text.success("✅ 数据加载完成！")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    return True

# ---------------------------
# 3. 策略核心
# ---------------------------
def run_strategy(current_date, params):
    try:
        idx = pd.IndexSlice
        if current_date not in GLOBAL_DATA['daily'].index.get_level_values(0): return pd.DataFrame()
        if current_date not in GLOBAL_DATA['daily_basic'].index.get_level_values(0): return pd.DataFrame()
            
        # 提取当日切片
        daily_today = GLOBAL_DATA['daily'].loc[idx[current_date, :]].copy().reset_index()
        basic_today = GLOBAL_DATA['daily_basic'].loc[idx[current_date, :]].copy().reset_index()
        
        # 补全列名
        if 'ts_code' not in daily_today.columns: daily_today['ts_code'] = daily_today.index
        if 'ts_code' not in basic_today.columns: basic_today['ts_code'] = basic_today.index
        
        # 基础合并
        df = pd.merge(daily_today, basic_today[['ts_code', 'circ_mv', 'turnover_rate']], on='ts_code', how='inner')
        
        # 资金流
        try:
            if 'moneyflow' in GLOBAL_DATA and not GLOBAL_DATA['moneyflow'].empty:
                if current_date in GLOBAL_DATA['moneyflow'].index.get_level_values(0):
                    mf_today = GLOBAL_DATA['moneyflow'].loc[idx[current_date, :]].copy().reset_index()
                    if 'ts_code' not in mf_today.columns: mf_today['ts_code'] = mf_today.index
                    mf_today['net_mf'] = mf_today['buy_lg_vol'] + mf_today['buy_elg_vol'] - mf_today['sell_lg_vol'] - mf_today['sell_elg_vol']
                    df = pd.merge(df, mf_today[['ts_code', 'net_mf']], on='ts_code', how='left')
                else: df['net_mf'] = 0
            else: df['net_mf'] = 0
        except: df['net_mf'] = 0

        # 筛选
        df = df[df['close'] >= params['min_price']]
        df = df[df['pct_chg'] < 9.5] 
        df = df[df['pct_chg'] > -9.5]
        df = df[(df['turnover_rate'] >= params['min_turnover']) & (df['turnover_rate'] <= params['max_turnover'])]
        df['circ_mv_yi'] = df['circ_mv'] / 10000
        df = df[(df['circ_mv_yi'] >= params['min_mv']) & (df['circ_mv_yi'] <= params['max_mv'])]
        
        df['max_oc'] = df[['open', 'close']].max(axis=1)
        df['upper_shadow'] = (df['high'] - df['max_oc']) / df['close']
        df = df[df['upper_shadow'] <= 0.05]
        
        # 评分
        df['score'] = df['turnover_rate']
        df.loc[df['net_mf'] > 0, 'score'] += 20
        df.loc[df['close'] > df['open'], 'score'] += 10
        
        return df.sort_values(by='score', ascending=False).head(params['top_k'])

    except Exception:
        return pd.DataFrame()

# ---------------------------
# 4. 主程序
# ---------------------------
def main():
    st.title("🚀 选股王 2025 (D3/D5修正版)")
    st.info("💡 严格规则：Open > Pre_Close 且 High > Open*1.015 方可买入")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        token = st.text_input("Tushare Token", value="", type="password")
    with c2:
        st.write("") 
        st.write("") 
        start_btn = st.button("开始回测 ▶", type="primary", use_container_width=True)

    with st.sidebar:
        st.header("⚙️ 参数设置")
        start_date = st.date_input("开始日期", datetime(2025, 1, 1))
        end_date = st.date_input("结束日期", datetime(2025, 12, 31))
        
        st.subheader("核心门槛")
        min_price = st.number_input("最低股价 (元)", 0.0, 500.0, 10.0)
        min_mv = st.number_input("最小流通市值 (亿)", 0.0, 1000.0, 20.0)
        max_mv = st.number_input("最大流通市值 (亿)", 0.0, 5000.0, 500.0)
        top_k = st.slider("每日持仓数", 1, 10, 5)

    if start_btn:
        if not token:
            st.error("请输入 Token")
            return
            
        global pro
        with st.spinner("连接 Tushare..."):
            pro = init_tushare(token)
            if not pro: return
        
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        today_str = datetime.now().strftime('%Y%m%d')
        if end_str >= today_str: end_str = get_real_trade_date(today_str)
        
        trade_days = get_trade_cal(start_str, end_str)
        if not trade_days:
            st.error("无有效交易日")
            return
        
        st.info(f"回测区间: {trade_days[0]} - {trade_days[-1]} | {len(trade_days)} 天")
        
        if not prefetch_data_stable(trade_days): return
        
        params = {'min_price': min_price, 'min_mv': min_mv, 'max_mv': max_mv, 
                  'min_turnover': 3.0, 'max_turnover': 30.0, 'top_k': top_k}
        
        results = []
        progress = st.progress(0)
        
        for i, date in enumerate(trade_days):
            progress.progress((i+1)/len(trade_days), text=f"分析: {date}")
            selected = run_strategy(date, params)
            
            if not selected.empty:
                # 【核心修改】：日期索引校准
                # T: 信号日 (i)
                # T+1: 买入日 (i+1)
                # T+2: D1卖出 (i+2) - A股T+1限制
                # T+3: D3卖出 (i+3) - 持股第3天
                # T+5: D5卖出 (i+5) - 持股第5天
                
                idx_buy = i + 1
                idx_sell1 = i + 2
                idx_sell3 = i + 3 # 修正：第3天直接卖
                idx_sell5 = i + 5 # 修正：第5天直接卖
                
                date_buy = trade_days[idx_buy] if idx_buy < len(trade_days) else None
                date_sell1 = trade_days[idx_sell1] if idx_sell1 < len(trade_days) else None
                date_sell3 = trade_days[idx_sell3] if idx_sell3 < len(trade_days) else None
                date_sell5 = trade_days[idx_sell5] if idx_sell5 < len(trade_days) else None
                
                if date_buy:
                    try:
                        idx = pd.IndexSlice
                        quotes_buy = GLOBAL_DATA['daily'].loc[idx[date_buy, :]] if date_buy else None
                        quotes_s1 = GLOBAL_DATA['daily'].loc[idx[date_sell1, :]] if date_sell1 else None
                        quotes_s3 = GLOBAL_DATA['daily'].loc[idx[date_sell3, :]] if date_sell3 else None
                        quotes_s5 = GLOBAL_DATA['daily'].loc[idx[date_sell5, :]] if date_sell5 else None
                        
                        for _, row in selected.iterrows():
                            code = row['ts_code']
                            ret_d1, ret_d3, ret_d5 = np.nan, np.nan, np.nan
                            status = "No Data"
                            
                            # 1. 验证买入条件
                            if quotes_buy is not None and code in quotes_buy.index:
                                try:
                                    bar_buy = quotes_buy.loc[code]
                                    if isinstance(bar_buy, pd.DataFrame): bar_buy = bar_buy.iloc[0]
                                    
                                    cond1 = bar_buy['open'] > bar_buy['pre_close']
                                    cond2 = bar_buy['high'] > bar_buy['open'] * 1.015
                                    
                                    if cond1 and cond2:
                                        buy_price = bar_buy['open'] * 1.015
                                        status = "Bought"
                                        
                                        # D+1 (T+2 卖)
                                        if quotes_s1 is not None and code in quotes_s1.index:
                                            bar_s1 = quotes_s1.loc[code]
                                            if isinstance(bar_s1, pd.DataFrame): bar_s1 = bar_s1.iloc[0]
                                            ret_d1 = (bar_s1['close'] - buy_price) / buy_price * 100
                                            
                                        # D+3 (T+3 卖)
                                        if quotes_s3 is not None and code in quotes_s3.index:
                                            bar_s3 = quotes_s3.loc[code]
                                            if isinstance(bar_s3, pd.DataFrame): bar_s3 = bar_s3.iloc[0]
                                            ret_d3 = (bar_s3['close'] - buy_price) / buy_price * 100
                                            
                                        # D+5 (T+5 卖)
                                        if quotes_s5 is not None and code in quotes_s5.index:
                                            bar_s5 = quotes_s5.loc[code]
                                            if isinstance(bar_s5, pd.DataFrame): bar_s5 = bar_s5.iloc[0]
                                            ret_d5 = (bar_s5['close'] - buy_price) / buy_price * 100
                                except: pass
                            
                            if status == "Bought":
                                results.append({
                                    'Trade_Date': date, 
                                    'ts_code': code, 
                                    'Return_D1': ret_d1,
                                    'Return_D3': ret_d3,
                                    'Return_D5': ret_d5,
                                    'Score': row['score']
                                })
                    except: pass
        
        progress.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            st.divider()
            st.header("📊 统计仪表盘")
            
            # 【回归经典】显示 D1/D3/D5
            cols = st.columns(3)
            periods = {'D+1 (T+2卖)': 'Return_D1', 
                       'D+3 (T+3卖)': 'Return_D3', 
                       'D+5 (T+5卖)': 'Return_D5'}
            
            for idx, (label, col_name) in enumerate(periods.items()):
                valid_data = df_res.dropna(subset=[col_name])
                if not valid_data.empty:
                    avg_ret = valid_data[col_name].mean()
                    win_rate = (valid_data[col_name] > 0).mean() * 100
                    cols[idx].metric(f"{label} 均益 / 胜率", f"{avg_ret:.2f}% / {win_rate:.1f}%")
            
            # 资金曲线
            df_curve = df_res.groupby('Trade_Date')['Return_D1'].mean().reset_index()
            df_curve['Equity'] = (1 + df_curve['Return_D1'].fillna(0)/100).cumprod()
            
            st.subheader("📈 资金曲线 (基准D+1)")
            st.area_chart(df_curve.set_index('Trade_Date')['Equity'])
            
            with st.expander("查看详细交易记录"):
                st.dataframe(df_res)
        else:
            st.warning("⚠️ 严格规则下未触发任何交易 (未满足 Open>PreClose 且 High>Open*1.015 条件)")

if __name__ == '__main__':
    main()
