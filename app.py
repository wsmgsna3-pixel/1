# -*- coding: utf-8 -*-
"""
选股王 · V30.12.3 Pro (MACD敏捷增强版)
------------------------------------------------
🔥 核心升级：
1. **敏捷版 MACD (8, 17, 5)**：
   - 专为超短线设计，捕捉起涨点，规避高位钝化。
   - 金叉大幅加分，死叉大幅扣分，旨在"清洗" Rank 1 的质量。
2. **动态历史回溯**：
   - 自动拉取候选股过去 40 个交易日数据，精准计算指标。
3. **实战保留**：
   - 严格买入 (Open > Pre_Close & High > 1.5%)
   - 硬盘断点续传
   - 20元/30亿市值门槛
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

warnings.filterwarnings("ignore")

# ---------------------------
# 全局配置 & 缓存
# ---------------------------
CACHE_DIR = "data_cache_2025_macd" # 新缓存目录
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

GLOBAL_DATA = {
    'daily': pd.DataFrame(),
    'daily_basic': pd.DataFrame(),
    'moneyflow': pd.DataFrame(),
    'index_daily': pd.DataFrame()
}
pro = None

st.set_page_config(page_title="选股王 MACD增强版", layout="wide")

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

def prefetch_index_data(start_date, end_date):
    """拉取指数风控数据"""
    global GLOBAL_DATA
    try:
        s_date = (datetime.strptime(start_date, '%Y%m%d') - timedelta(days=40)).strftime('%Y%m%d')
        df = pro.index_daily(ts_code='000001.SH', start_date=s_date, end_date=end_date)
        if df is not None and not df.empty:
            df = df.sort_values('trade_date')
            df['ma20'] = df['close'].rolling(window=20).mean()
            df.set_index('trade_date', inplace=True)
            GLOBAL_DATA['index_daily'] = df
            return True
    except: pass
    return False

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
    
    # 合并数据
    if all_daily:
        full_daily = pd.concat(all_daily)
        for col in ['trade_date', 'ts_code']:
            full_daily[col] = full_daily[col].astype(str).str.strip()
        full_daily.drop_duplicates(subset=['trade_date', 'ts_code'], inplace=True)
        full_daily.set_index(['trade_date', 'ts_code'], inplace=True)
        full_daily.sort_index(inplace=True)
        GLOBAL_DATA['daily'] = full_daily
    else: return False
        
    if all_basic:
        full_basic = pd.concat(all_basic)
        for col in ['trade_date', 'ts_code']:
            full_basic[col] = full_basic[col].astype(str).str.strip()
        full_basic.drop_duplicates(subset=['trade_date', 'ts_code'], inplace=True)
        full_basic.set_index(['trade_date', 'ts_code'], inplace=True)
        full_basic.sort_index(inplace=True)
        GLOBAL_DATA['daily_basic'] = full_basic
        
    if all_mf:
        full_mf = pd.concat(all_mf)
        for col in ['trade_date', 'ts_code']:
            full_mf[col] = full_mf[col].astype(str).str.strip()
        full_mf.set_index(['trade_date', 'ts_code'], inplace=True)
        full_mf.sort_index(inplace=True)
        GLOBAL_DATA['moneyflow'] = full_mf

    status_text.success("✅ 数据加载完成！")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    return True

# ---------------------------
# 3. 辅助计算: 敏捷版 MACD
# ---------------------------
def calculate_agile_macd(df_hist, fast=8, slow=17, signal=5):
    """
    计算敏捷版 MACD (8, 17, 5)
    返回带有 dif, dea, macd 列的 DataFrame (仅取最后一天)
    """
    if df_hist.empty or len(df_hist) < slow + 5:
        return None
    
    # 排序
    df = df_hist.sort_values('trade_date').copy()
    close = df['close']
    
    # 计算 EMA
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    # 计算 DIF, DEA, MACD
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    # 注意：国内软件通常是 (dif-dea)*2，这里保持标准，这不影响金叉判断
    macd_hist = (dif - dea) * 2 
    
    # 提取最后一天的数据
    last_idx = df.index[-1]
    last_dif = dif.iloc[-1]
    last_dea = dea.iloc[-1]
    last_hist = macd_hist.iloc[-1]
    
    # 提取前一天的数据 (用于判断金叉)
    prev_dif = dif.iloc[-2]
    prev_dea = dea.iloc[-2]
    
    is_gold_cross = (prev_dif < prev_dea) and (last_dif > last_dea)
    is_dead_cross = (prev_dif > prev_dea) and (last_dif < last_dea)
    
    return {
        'dif': last_dif,
        'dea': last_dea,
        'hist': last_hist,
        'gold_cross': is_gold_cross,
        'dead_cross': is_dead_cross
    }

# ---------------------------
# 4. 策略核心 (整合 MACD)
# ---------------------------
def run_strategy(current_date, params):
    try:
        # --- 0. 大盘风控 ---
        if params['use_market_control']:
            try:
                if current_date in GLOBAL_DATA['index_daily'].index:
                    idx_today = GLOBAL_DATA['index_daily'].loc[current_date]
                    if not np.isnan(idx_today['ma20']) and idx_today['close'] < idx_today['ma20']:
                        return pd.DataFrame() 
            except: pass

        idx = pd.IndexSlice
        if current_date not in GLOBAL_DATA['daily'].index.get_level_values(0): return pd.DataFrame()
        if current_date not in GLOBAL_DATA['daily_basic'].index.get_level_values(0): return pd.DataFrame()
            
        # --- 1. 初筛 (Price, MV, Turnover) ---
        daily_today = GLOBAL_DATA['daily'].loc[idx[current_date, :]].copy().reset_index()
        basic_today = GLOBAL_DATA['daily_basic'].loc[idx[current_date, :]].copy().reset_index()
        
        if 'ts_code' not in daily_today.columns: daily_today['ts_code'] = daily_today.index
        if 'ts_code' not in basic_today.columns: basic_today['ts_code'] = basic_today.index
        
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

        # 初步过滤
        df = df[df['close'] >= params['min_price']]
        df = df[df['pct_chg'] < 9.5] 
        df = df[df['pct_chg'] > -9.5]
        df = df[(df['turnover_rate'] >= params['min_turnover']) & (df['turnover_rate'] <= params['max_turnover'])]
        df['circ_mv_yi'] = df['circ_mv'] / 10000
        df = df[(df['circ_mv_yi'] >= params['min_mv']) & (df['circ_mv_yi'] <= params['max_mv'])]
        
        df['max_oc'] = df[['open', 'close']].max(axis=1)
        df['upper_shadow'] = (df['high'] - df['max_oc']) / df['close']
        df = df[df['upper_shadow'] <= 0.05]
        
        if df.empty: return pd.DataFrame()
        
        # --- 2. 深度计算: MACD (8, 17, 5) ---
        # 为了速度，只取初步评分前 50 名进行 MACD 计算
        df['temp_score'] = df['turnover_rate']
        candidates = df.sort_values(by='temp_score', ascending=False).head(50)['ts_code'].tolist()
        
        macd_scores = {}
        
        # 获取历史数据 (用于计算 MACD)
        # 这里需要在 GLOBAL_DATA['daily'] 中截取
        # 技巧：在内存中截取比请求 API 快得多
        
        # 获取过去 40 天的日期列表
        current_dt = datetime.strptime(str(current_date), '%Y%m%d')
        start_dt_limit = (current_dt - timedelta(days=60)).strftime('%Y%m%d')
        
        for code in candidates:
            try:
                # 在全局数据中切片该股票的历史数据
                # 注意：GLOBAL_DATA 是 MultiIndex, 用 xs 切片 ts_code
                # 这种操作在数据量大时可能略慢，但比网络请求快
                hist_data = GLOBAL_DATA['daily'].xs(code, level='ts_code')
                
                # 截取 start_dt_limit 到 current_date
                hist_slice = hist_data[(hist_data.index >= start_dt_limit) & (hist_data.index <= str(current_date))]
                
                res = calculate_agile_macd(hist_slice, fast=8, slow=17, signal=5)
                
                score_adj = 0
                if res:
                    # 策略核心：复活 Rank 1 的关键逻辑
                    
                    # 1. 金叉暴击 (+30分)
                    if res['gold_cross']:
                        score_adj += 30
                        
                    # 2. 多头趋势 (+10分)
                    elif res['dif'] > res['dea'] and res['dif'] > 0:
                        score_adj += 10
                        
                    # 3. 死叉惩罚 (-20分) -> 这里的关键！把高位死叉的踢下去
                    if res['dead_cross'] or res['dif'] < res['dea']:
                        score_adj -= 20
                        
                macd_scores[code] = score_adj
            except:
                macd_scores[code] = 0

        # --- 3. 最终评分 ---
        df['score'] = df['turnover_rate']
        df.loc[df['net_mf'] > 0, 'score'] += 20
        df.loc[df['close'] > df['open'], 'score'] += 10
        
        # 应用 MACD 分数
        df['macd_boost'] = df['ts_code'].map(macd_scores).fillna(0)
        df['score'] += df['macd_boost']
        
        return df.sort_values(by='score', ascending=False).head(params['top_k'])

    except Exception:
        return pd.DataFrame()

# ---------------------------
# 5. 主程序
# ---------------------------
def main():
    st.title("🚀 选股王 2025 (MACD敏捷版)")
    st.info("💡 策略核心：引入 MACD(8,17,5) 因子，金叉大幅加分，死叉大幅扣分，旨在净化 Rank 1。")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        token = st.text_input("Tushare Token", value="", type="password")
    with c2:
        st.write("") 
        st.write("") 
        start_btn = st.button("开始回测 ▶", type="primary", use_container_width=True)

    with st.sidebar:
        st.header("⚙️ 参数设置")
        use_market_control = st.checkbox("✅ 开启大盘风控 (MA20)", value=True)
        start_date = st.date_input("开始日期", datetime(2025, 1, 1))
        end_date = st.date_input("结束日期", datetime(2025, 12, 31))
        
        st.subheader("核心门槛")
        min_price = st.number_input("最低股价", 0.0, 500.0, 20.0)
        min_mv = st.number_input("最小流通市值", 0.0, 1000.0, 30.0)
        max_mv = st.number_input("最大流通市值", 0.0, 5000.0, 800.0)
        top_k = st.slider("每日持仓数", 1, 10, 5)
        min_turnover = st.number_input("最小换手", 0.0, 100.0, 3.0)
        max_turnover = st.number_input("最大换手", 0.0, 100.0, 30.0)

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
        
        with st.spinner("更新大盘指数数据..."):
            prefetch_index_data(start_str, end_str)
            
        st.info(f"回测区间: {trade_days[0]} - {trade_days[-1]} | {len(trade_days)} 天")
        
        if not prefetch_data_stable(trade_days): return
        
        params = {
            'min_price': min_price, 'min_mv': min_mv, 'max_mv': max_mv, 
            'min_turnover': min_turnover, 'max_turnover': max_turnover, 'top_k': top_k,
            'use_market_control': use_market_control
        }
        
        results = []
        progress = st.progress(0)
        
        for i, date in enumerate(trade_days):
            progress.progress((i+1)/len(trade_days), text=f"MACD 计算中: {date}")
            selected = run_strategy(date, params)
            
            if not selected.empty:
                idx_buy = i + 1
                idx_sell1 = i + 2
                idx_sell3 = i + 3
                idx_sell5 = i + 5
                
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
                            status = "Wait"
                            
                            if quotes_buy is not None and code in quotes_buy.index:
                                try:
                                    bar_buy = quotes_buy.loc[code]
                                    if isinstance(bar_buy, pd.DataFrame): bar_buy = bar_buy.iloc[0]
                                    
                                    if bar_buy['open'] > bar_buy['pre_close'] and bar_buy['high'] > bar_buy['open'] * 1.015:
                                        buy_price = bar_buy['open'] * 1.015
                                        status = "Bought"
                                        
                                        if quotes_s1 is not None and code in quotes_s1.index:
                                            bar_s1 = quotes_s1.loc[code]
                                            if isinstance(bar_s1, pd.DataFrame): bar_s1 = bar_s1.iloc[0]
                                            ret_d1 = (bar_s1['close'] - buy_price) / buy_price * 100
                                            
                                        if quotes_s3 is not None and code in quotes_s3.index:
                                            bar_s3 = quotes_s3.loc[code]
                                            if isinstance(bar_s3, pd.DataFrame): bar_s3 = bar_s3.iloc[0]
                                            ret_d3 = (bar_s3['close'] - buy_price) / buy_price * 100
                                            
                                        if quotes_s5 is not None and code in quotes_s5.index:
                                            bar_s5 = quotes_s5.loc[code]
                                            if isinstance(bar_s5, pd.DataFrame): bar_s5 = bar_s5.iloc[0]
                                            ret_d5 = (bar_s5['close'] - buy_price) / buy_price * 100
                                except: pass
                            
                            if status == "Bought":
                                results.append({
                                    'Trade_Date': date, 'ts_code': code, 
                                    'Return_D1': ret_d1, 'Return_D3': ret_d3, 'Return_D5': ret_d5,
                                    'Score': row['score']
                                })
                    except: pass
        
        progress.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            st.divider()
            st.header("📊 MACD 增强版回测报告")
            
            cols = st.columns(3)
            periods = {'D+1 (T+2卖)': 'Return_D1', 'D+3 (T+3卖)': 'Return_D3', 'D+5 (T+5卖)': 'Return_D5'}
            for idx, (label, col_name) in enumerate(periods.items()):
                valid_data = df_res.dropna(subset=[col_name])
                if not valid_data.empty:
                    avg_ret = valid_data[col_name].mean()
                    win_rate = (valid_data[col_name] > 0).mean() * 100
                    cols[idx].metric(f"{label} 均益 / 胜率", f"{avg_ret:.2f}% / {win_rate:.1f}%")
            
            df_curve = df_res.groupby('Trade_Date')['Return_D1'].mean().reset_index()
            df_curve['Equity'] = (1 + df_curve['Return_D1'].fillna(0)/100).cumprod()
            
            st.subheader("📈 资金曲线")
            st.area_chart(df_curve.set_index('Trade_Date')['Equity'])
            st.dataframe(df_res)
            
            # 导出功能
            csv = df_res.to_csv().encode('utf-8')
            st.download_button("下载 CSV", csv, "macd_backtest.csv", "text/csv")
        else:
            st.warning("未触发交易。请检查是否被大盘风控拦截。")

if __name__ == '__main__':
    main()
