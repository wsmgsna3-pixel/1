# -*- coding: utf-8 -*-
"""
选股王 · V30.12.18 涨幅熔断版
------------------------------------------------
基于 tb1.txt 修改
核心改动：
1. 【累计涨幅熔断】新增 方案A 风控：
   - 5日累计涨幅 > 40% -> 禁买
   - 10日累计涨幅 > 70% -> 禁买
2. 【保留原味】其他逻辑（Rank算法、RSI、MACD）与 tb1 完全一致。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import concurrent.futures 
import os
import pickle

warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_INDUSTRY = {} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V30.12.18 涨幅熔断版", layout="wide")
st.title("选股王 V30.12.18：涨幅熔断版 (方案A)")

# ---------------------------
# 基础 API 函数
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: 
        return pd.DataFrame(columns=['ts_code']) 
   
    func = getattr(pro, func_name) 
    try:
        for _ in range(3):
            try:
                if kwargs.get('is_index'):
                    df = pro.index_daily(**kwargs)
                else:
                    df = func(**kwargs)
                
                if df is not None and not df.empty:
                    return df
                time.sleep(0.5)
            except:
                time.sleep(1)
                continue
        return pd.DataFrame(columns=['ts_code']) 
    except Exception as e:
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    lookback_days = max(num_days * 3, 365) 
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty: return []
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    return trade_days_df[trade_days_df['cal_date'] <= end_date_str]['cal_date'].head(num_days).tolist()

@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj = safe_get('adj_factor', trade_date=date)
    daily = safe_get('daily', trade_date=date)
    return {'adj': adj, 'daily': daily}

# ---------------------------
# 数据缓存逻辑 (本地化)
# ---------------------------
CACHE_FILE_NAME = "market_data_cache.pkl"

def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    
    # 1. 尝试读取本地缓存
    if use_cache and os.path.exists(CACHE_FILE_NAME):
        st.success("⚡ 发现本地缓存，正在极速加载...")
        try:
            with open(CACHE_FILE_NAME, 'rb') as f:
                d = pickle.load(f)
                GLOBAL_ADJ_FACTOR = d['adj']
                GLOBAL_DAILY_RAW = d['daily']
            
            # 恢复基准复权因子
            latest = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
            if latest:
                try:
                    GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest), 'adj_factor'].droplevel(1).to_dict()
                except: pass
            return True
        except Exception as e:
            st.warning(f"缓存文件损坏，将重新下载: {e}")
            os.remove(CACHE_FILE_NAME)

    # 2. 如果没有缓存，则下载
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    # 向前多取数据以确保计算指标
    start_date_dt = datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=200)
    end_date_dt = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=30)
    
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = end_date_dt.strftime("%Y%m%d")
    
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if cal.empty: return False
    all_dates = cal['cal_date'].tolist()
    
    st.info(f"📡 [首次运行] 正在下载全市场数据: {start_date} 至 {end_date}...")
    
    adj_list = []
    daily_list = []
    
    def fetch_worker(date):
        return fetch_and_cache_daily_data(date)
    
    bar = st.progress(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f2d = {executor.submit(fetch_worker, d): d for d in all_dates}
        for i, f in enumerate(concurrent.futures.as_completed(f2d)):
            try:
                d = f.result()
                if not d['adj'].empty: adj_list.append(d['adj'])
                if not d['daily'].empty: daily_list.append(d['daily'])
            except: pass
            if i % 10 == 0:
                bar.progress((i + 1) / len(all_dates))
    bar.empty()
    
    if not daily_list: return False
    
    with st.spinner("正在构建数据索引并写入硬盘..."):
        # 合并
        GLOBAL_ADJ_FACTOR = pd.concat(adj_list).drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index()
        GLOBAL_ADJ_FACTOR['adj_factor'] = pd.to_numeric(GLOBAL_ADJ_FACTOR['adj_factor'], errors='coerce').fillna(0)
        
        GLOBAL_DAILY_RAW = pd.concat(daily_list).drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date']).sort_index()
        
        # 预计算最新复权因子
        latest = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
        if latest:
            try:
                GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest), 'adj_factor'].droplevel(1).to_dict()
            except: pass
            
        # 写入缓存
        try:
            with open(CACHE_FILE_NAME, 'wb') as f:
                pickle.dump({'adj': GLOBAL_ADJ_FACTOR, 'daily': GLOBAL_DAILY_RAW}, f)
        except Exception as e:
            st.error(f"写入缓存失败: {e}")
            
    return True

# ---------------------------
# 复权数据计算 (极速版)
# ---------------------------
def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    
    # 快速获取最新复权因子
    latest_adj_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj_factor): return pd.DataFrame() 

    try:
        # 切片获取个股数据
        daily_df = GLOBAL_DAILY_RAW.loc[ts_code]
        daily_df = daily_df.loc[(daily_df.index >= start_date) & (daily_df.index <= end_date)]
        
        adj_series = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        adj_series = adj_series.loc[(adj_series.index >= start_date) & (adj_series.index <= end_date)]
    except KeyError:
        return pd.DataFrame()
    
    if daily_df.empty or adj_series.empty: return pd.DataFrame()
    
    # 合并
    df = daily_df.merge(adj_series.rename('adj_factor'), left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    
    # 前复权计算: Price_QFQ = Price * (Adj / Latest_Adj)
    # 向量化计算比循环快
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df = df.sort_values('trade_date_str').set_index('trade_date_str')
    
    # 替换原列
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col + '_qfq']
        
    return df[['open', 'high', 'low', 'close', 'vol']].copy() 

# ---------------------------
# 指标计算 (含累计涨幅)
# ---------------------------
def compute_indicators(ts_code, end_date):
    # 多取一些数据用于计算累计涨幅
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date=start_date, end_date=end_date)
    
    res = {}
    if df.empty or len(df) < 26: return res 
    
    # 基础指标
    df['pct_chg'] = df['close'].pct_change().fillna(0) * 100 
    close = df['close']
    
    res['last_close'] = close.iloc[-1]
    res['last_open'] = df['open'].iloc[-1]
    res['last_high'] = df['high'].iloc[-1]
    res['last_low'] = df['low'].iloc[-1]
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    
    # 均线
    res['ma20'] = close.tail(20).mean()
    res['ma60'] = close.tail(60).mean()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/12, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/12, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    res['rsi_12'] = 100 - (100 / (1 + rs)).iloc[-1]
    
    # 【新增】累计涨幅计算 (用于熔断)
    # 5日累计涨幅 = (Today_Close - Close_5_days_ago) / Close_5_days_ago
    if len(close) >= 6:
        res['pct_chg_5d'] = (close.iloc[-1] / close.iloc[-6] - 1) * 100
    else:
        res['pct_chg_5d'] = 0
        
    # 10日累计涨幅
    if len(close) >= 11:
        res['pct_chg_10d'] = (close.iloc[-1] / close.iloc[-11] - 1) * 100
    else:
        res['pct_chg_10d'] = 0

    return res

def get_future_prices(ts_code, selection_date, d0_close, days_ahead=[1, 3, 5]):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    s = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    e = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    h = get_qfq_data_v4_optimized_final(ts_code, start_date=s, end_date=e)
    res = {}
    if h.empty: return res
    
    for n in days_ahead:
        col = f'Return_D{n}'
        if len(h) >= n:
            # 收益率 = (Dn_Close - D0_Close) / D0_Close
            # 注意：这里计算的是买入后持有 N 天的收益，基准是选股日的收盘价
            res[col] = (h.iloc[n-1]['close'] - d0_close) / d0_close * 100
    return res

def get_market_state(trade_date):
    # 简单判定：大盘(沪深300) 20日均线之上为强，之下为弱
    s = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
    idx_df = safe_get('daily', ts_code='000300.SH', start_date=s, end_date=trade_date, is_index=True)
    if idx_df.empty or len(idx_df) < 20: return 'Weak'
    
    current_close = idx_df['close'].iloc[-1]
    ma20 = idx_df['close'].tail(20).mean()
    return 'Strong' if current_close > ma20 else 'Weak'

# ---------------------------
# 回测主逻辑
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MAX_PREV_PCT, RSI_LIMIT, MAX_RET_5D, MAX_RET_10D):
    # 1. 获取市场状态
    market_state = get_market_state(last_trade)
    
    # 2. 获取当日全市场行情
    daily = safe_get('daily', trade_date=last_trade)
    if daily.empty: return pd.DataFrame(), "今日无数据"
    
    # 3. 基础过滤
    base = safe_get('stock_basic', list_status='L')
    df = daily.merge(base[['ts_code','name','industry']], on='ts_code')
    
    # 去除 ST 和 退市
    df = df[~df['name'].str.contains('ST|退')]
    
    # 4. 初筛: 涨幅降序，取前 FINAL_POOL 名
    cands = df.sort_values('pct_chg', ascending=False).head(FINAL_POOL)
    
    recs = []
    for row in cands.itertuples():
        # [过滤1] 昨日/今日单日涨幅过大 (防一字板)
        if row.pct_chg > MAX_PREV_PCT: continue
        
        # 计算指标
        ind = compute_indicators(row.ts_code, last_trade)
        if not ind: continue
        
        # [过滤2] RSI 拦截 (默认100，相当于不拦截)
        d0_rsi = ind['rsi_12']
        if d0_rsi > RSI_LIMIT: continue

        # [过滤3] 累计涨幅熔断 (方案A)
        # 5日涨幅过大 -> 熔断
        if ind.get('pct_chg_5d', 0) > MAX_RET_5D: continue
        # 10日涨幅过大 -> 熔断
        if ind.get('pct_chg_10d', 0) > MAX_RET_10D: continue

        # 获取未来收益 (用于回测验证)
        fut = get_future_prices(row.ts_code, last_trade, ind['last_close'])
        
        # 资金流 (需要额外获取，这里简化为 0 或需调用 moneyflow)
        # 为保持极速版速度，暂用成交量代替资金热度，或如果已有缓存可用
        net_mf = 0 # 简化
        
        recs.append({
            'ts_code': row.ts_code, 
            'name': row.name, 
            'Close': ind['last_close'],
            'Pct_Chg': row.pct_chg,
            'rsi': d0_rsi,
            'macd': ind['macd_val'],
            'pct_chg_5d': ind.get('pct_chg_5d', 0),   # 记录下来以便查看
            'pct_chg_10d': ind.get('pct_chg_10d', 0), # 记录下来以便查看
            'Return_D1 (%)': fut.get('Return_D1'), 
            'Return_D3 (%)': fut.get('Return_D3'),
            'Return_D5 (%)': fut.get('Return_D5'),
            'market_state': market_state,
        })

    if not recs: return pd.DataFrame(), "无符合标的"
    
    fdf = pd.DataFrame(recs)
    
    # ---------------------------
    # 打分排序 (Rank 核心)
    # ---------------------------
    def score(r):
        # 基础分：MACD 越强越好
        s = r['macd'] * 100 
        
        # RSI 加分项 (强者恒强)
        if r['rsi'] > 70: s += 50
        
        # 市场状态修正
        if r['market_state'] == 'Strong':
            if r['rsi'] > 80: s += 20
        else:
            if r['rsi'] > 85: s -= 50 # 弱市不做超买
            
        return s
    
    fdf['Score'] = fdf.apply(score, axis=1)
    
    # 排序取 Top K
    final = fdf.sort_values('Score', ascending=False).head(TOP_BACKTEST)
    final.insert(0, 'Rank', range(1, len(final)+1))
    
    return final, None

# ---------------------------
# UI 侧边栏
# ---------------------------
with st.sidebar:
    st.header("V30.12.18 涨幅熔断配置")
    
    # 熔断参数配置 (方案A默认值)
    st.subheader("🛡️ 累计涨幅熔断 (方案A)")
    MAX_RET_5D = st.number_input("5日累计涨幅上限 (%)", value=40.0, help="超过此值坚决不买")
    MAX_RET_10D = st.number_input("10日累计涨幅上限 (%)", value=70.0, help="超过此值坚决不买")
    
    st.markdown("---")
    
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("回测天数", value=10, step=1)
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=5)
    
    # 缓存管理
    if st.button("🗑️ 清除行情缓存"):
        if os.path.exists(CACHE_FILE_NAME): os.remove(CACHE_FILE_NAME)
        st.success("缓存已清除，下次运行将重新下载")
    
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    CHECKPOINT_FILE = "backtest_checkpoint_v18.csv"

    # 其他参数
    MAX_PREV_PCT = st.number_input("昨日最大涨幅限制 (%)", value=19.0)
    RSI_LIMIT = st.number_input("RSI 拦截线 (建议100)", value=100.0)

# ---------------------------
# Tushare Token
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 主程序
# ---------------------------
if st.button(f"🚀 启动 V30.12.18 回测"):
    
    # 处理断点续传
    processed_dates = set()
    results = []
    if RESUME_CHECKPOINT and os.path.exists(CHECKPOINT_FILE):
        try:
            existing_df = pd.read_csv(CHECKPOINT_FILE)
            existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
            processed_dates = set(existing_df['Trade_Date'].unique())
            results.append(existing_df)
            st.success(f"✅ 断点续传：已跳过 {len(processed_dates)} 天")
        except: pass
    else:
        if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    
    # 获取交易日历
    trade_days_list = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_list: st.stop()
        
    dates_to_run = [d for d in trade_days_list if d not in processed_dates]
    
    if not dates_to_run:
        st.success("🎉 分析完毕 (所有日期已在缓存中)")
    else:
        # 1. 准备数据
        if not get_all_historical_data(trade_days_list, use_cache=True): st.stop()
            
        # 2. 循环回测
        bar = st.progress(0, text="启动引擎...")
        for i, date in enumerate(dates_to_run):
            res, err = run_backtest_for_a_day(
                date, int(TOP_BACKTEST), 100, 
                MAX_PREV_PCT, RSI_LIMIT, 
                MAX_RET_5D, MAX_RET_10D # 传入熔断参数
            )
            
            if not res.empty:
                res['Trade_Date'] = date
                
                # 写入断点文件
                is_first = not os.path.exists(CHECKPOINT_FILE)
                res.to_csv(CHECKPOINT_FILE, mode='a', index=False, header=is_first, encoding='utf-8-sig')
                
                results.append(res)
            
            bar.progress((i+1)/len(dates_to_run), text=f"分析中: {date}")
        
        bar.empty()
    
    # 结果展示
    if results:
        all_res = pd.concat(results)
        
        # 实时过滤 Top K
        all_res = all_res[all_res['Rank'] <= int(TOP_BACKTEST)]
        
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        all_res = all_res.sort_values(['Trade_Date', 'Rank'], ascending=[False, True])
        
        st.header(f"📊 V30.12.18 统计仪表盘 (方案A: 5日<{MAX_RET_5D}% / 10日<{MAX_RET_10D}%)")
        cols = st.columns(3)
        for idx, n in enumerate([1, 3, 5]):
            col_name = f'Return_D{n} (%)'
            valid = all_res.dropna(subset=[col_name]) 
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                cols[idx].metric(f"D+{n} 均益 / 胜率", f"{avg:.2f}% / {win:.1f}%")
        
        st.subheader("📋 回测清单 (含累计涨幅)")
        
        show_cols = ['Rank', 'Trade_Date','name','ts_code','Close','Pct_Chg',
                     'pct_chg_5d', 'pct_chg_10d', # 显示累计涨幅
                     'Return_D1 (%)','Return_D3 (%)','Return_D5 (%)',
                     'rsi','macd','market_state']
        
        # 格式化显示
        display_df = all_res[show_cols].copy()
        display_df = display_df.style.format({
            'Close': '{:.2f}', 'Pct_Chg': '{:.2f}%',
            'pct_chg_5d': '{:.2f}%', 'pct_chg_10d': '{:.2f}%',
            'Return_D1 (%)': '{:.2f}%', 'Return_D3 (%)': '{:.2f}%', 'Return_D5 (%)': '{:.2f}%',
            'rsi': '{:.2f}', 'macd': '{:.2f}'
        })
        
        st.dataframe(display_df, use_container_width=True)
        
        # 下载
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载结果 CSV", csv, "export_v18.csv", "text/csv")
    else:
        st.warning("⚠️ 没有选出任何股票，可能是熔断阈值设置过低？")
