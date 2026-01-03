# -*- coding: utf-8 -*-
"""
选股王 · V30.12.3 Pro (硬盘断点续传·终极修正版)
------------------------------------------------
🔥 核心修复：
1. **硬盘断点续传**：数据拉取后存入本地 'data_cache_2025' 文件夹。
   - 即使程序崩溃，重启后也会直接读取本地文件，绝不从头开始！
2. **代码逻辑回填**：恢复了之前被精简掉的缓存管理逻辑，代码量恢复，功能完整。
3. **收益修正**：保持 .loc 读取方式，杜绝收益为 0。
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
# 全局配置 & 缓存初始化
# ---------------------------
# 定义缓存目录
CACHE_DIR = "data_cache_2025"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# 全局数据容器
GLOBAL_DATA = {
    'daily': pd.DataFrame(),
    'daily_basic': pd.DataFrame(),
    'moneyflow': pd.DataFrame()
}
pro = None

st.set_page_config(page_title="选股王 硬盘续传版", layout="wide")

# ---------------------------
# 1. 基础工具函数
# ---------------------------
@st.cache_resource
def init_tushare(token):
    if not token: return None
    try:
        api = ts.pro_api(token)
        # 测试连通性
        api.trade_cal(start_date='20250101', end_date='20250101')
        return api
    except Exception as e:
        st.error(f"Token 连接失败: {e}")
        return None

def get_real_trade_date(date_str):
    """自动修正非交易日"""
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
    """获取交易日历"""
    if pro is None: return []
    try:
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
        df = df[df['is_open'] == 1]
        return sorted(df['cal_date'].tolist())
    except:
        return []

# ---------------------------
# 2. 核心：带硬盘缓存的数据拉取
# ---------------------------
def fetch_and_cache(api_func, date, data_type, **kwargs):
    """
    智能拉取函数：
    1. 检查本地硬盘有没有缓存文件
    2. 有 -> 读取并返回 (0流量, 0耗时)
    3. 无 -> 联网下载 -> 存入硬盘 -> 返回
    """
    # 缓存文件名: data_cache_2025/20250101_daily.pkl
    cache_file = os.path.join(CACHE_DIR, f"{date}_{data_type}.pkl")
    
    # --- A. 尝试读取缓存 ---
    if os.path.exists(cache_file):
        try:
            df = pd.read_pickle(cache_file)
            # 简单校验，防止读取空文件
            if df is not None: 
                return df, True # True 代表来自缓存
        except Exception:
            # 如果缓存文件损坏，删掉它，准备重新下载
            os.remove(cache_file)
    
    # --- B. 联网下载 (带重试) ---
    for retries in range(3): # 重试3次
        try:
            df = api_func(**kwargs)
            if df is not None and not df.empty:
                # 下载成功，写入硬盘缓存
                df.to_pickle(cache_file)
                return df, False # False 代表来自网络
            elif df is not None and df.empty:
                # 空数据也缓存，避免重复请求空值
                df.to_pickle(cache_file)
                return df, False
        except Exception as e:
            time.sleep(1) # 失败歇1秒
            continue
            
    return None, False

def prefetch_data_stable(trade_days):
    """
    极其稳定的数据预加载流程
    """
    global pro, GLOBAL_DATA
    if not trade_days: return False
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    all_daily = []
    all_basic = []
    all_mf = []
    
    total_days = len(trade_days)
    cache_hits = 0
    network_hits = 0
    
    # 逐日循环
    for i, date in enumerate(trade_days):
        # 1. Daily 行情
        df_d, is_cache = fetch_and_cache(pro.daily, date, 'daily', trade_date=date)
        if df_d is not None and not df_d.empty:
            all_daily.append(df_d)
        
        # 2. Daily Basic 指标
        df_b, _ = fetch_and_cache(pro.daily_basic, date, 'basic', trade_date=date, fields='ts_code,trade_date,turnover_rate,circ_mv,total_mv,pe,pb')
        if df_b is not None and not df_b.empty:
            all_basic.append(df_b)
            
        # 3. Moneyflow 资金流
        df_m, _ = fetch_and_cache(pro.moneyflow, date, 'moneyflow', trade_date=date)
        if df_m is not None and not df_m.empty:
            all_mf.append(df_m)
        
        # 状态更新
        if is_cache:
            cache_hits += 1
            msg = f"⚡ 已读缓存: {date}"
            # 读缓存太快了，不需要 sleep
        else:
            network_hits += 1
            msg = f"🌐 网络下载: {date}"
            # 只有走网络时才需要休息，防止限流
            time.sleep(0.05)
            
        progress_bar.progress((i + 1) / total_days, text=f"{msg} ({i+1}/{total_days})")

    status_text.info(f"数据准备完毕！本地缓存命中: {cache_hits} 天 | 网络下载: {network_hits} 天")
    
    # 合并数据
    status_text.text("正在合并数据表...")
    
    if all_daily:
        full_daily = pd.concat(all_daily)
        # 清洗
        full_daily['trade_date'] = full_daily['trade_date'].astype(str).str.strip()
        full_daily['ts_code'] = full_daily['ts_code'].astype(str).str.strip()
        full_daily.drop_duplicates(subset=['trade_date', 'ts_code'], inplace=True)
        full_daily.set_index(['trade_date', 'ts_code'], inplace=True)
        full_daily.sort_index(inplace=True)
        GLOBAL_DATA['daily'] = full_daily
    else:
        st.error("❌ 行情数据为空")
        return False
        
    if all_basic:
        full_basic = pd.concat(all_basic)
        full_basic['trade_date'] = full_basic['trade_date'].astype(str).str.strip()
        full_basic['ts_code'] = full_basic['ts_code'].astype(str).str.strip()
        full_basic.drop_duplicates(subset=['trade_date', 'ts_code'], inplace=True)
        full_basic.set_index(['trade_date', 'ts_code'], inplace=True)
        full_basic.sort_index(inplace=True)
        GLOBAL_DATA['daily_basic'] = full_basic
    else:
        st.error("❌ 指标数据为空")
        return False
        
    if all_mf:
        full_mf = pd.concat(all_mf)
        full_mf['trade_date'] = full_mf['trade_date'].astype(str).str.strip()
        full_mf['ts_code'] = full_mf['ts_code'].astype(str).str.strip()
        full_mf.set_index(['trade_date', 'ts_code'], inplace=True)
        full_mf.sort_index(inplace=True)
        GLOBAL_DATA['moneyflow'] = full_mf
        
    status_text.success("✅ 数据加载成功！")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    return True

# ---------------------------
# 3. 策略执行逻辑
# ---------------------------
def run_strategy(current_date, params):
    try:
        idx = pd.IndexSlice
        # 检查
        if current_date not in GLOBAL_DATA['daily'].index.get_level_values(0): return pd.DataFrame()
        if current_date not in GLOBAL_DATA['daily_basic'].index.get_level_values(0): return pd.DataFrame()
            
        # 提取 Copy
        daily_today = GLOBAL_DATA['daily'].loc[idx[current_date, :]].copy()
        basic_today = GLOBAL_DATA['daily_basic'].loc[idx[current_date, :]].copy()
        
        # 重置索引
        daily_today = daily_today.reset_index()
        if 'ts_code' not in daily_today.columns: daily_today['ts_code'] = daily_today.index
        basic_today = basic_today.reset_index()
        if 'ts_code' not in basic_today.columns: basic_today['ts_code'] = basic_today.index
        
        # 1. 基础合并 (Inner Join)
        df = pd.merge(daily_today, basic_today[['ts_code', 'circ_mv', 'turnover_rate']], on='ts_code', how='inner')
        
        # 2. 资金流合并 (Left Join)
        try:
            if 'moneyflow' in GLOBAL_DATA and not GLOBAL_DATA['moneyflow'].empty:
                if current_date in GLOBAL_DATA['moneyflow'].index.get_level_values(0):
                    mf_today = GLOBAL_DATA['moneyflow'].loc[idx[current_date, :]].copy()
                    mf_today = mf_today.reset_index()
                    if 'ts_code' not in mf_today.columns: mf_today['ts_code'] = mf_today.index
                    mf_today['net_mf'] = mf_today['buy_lg_vol'] + mf_today['buy_elg_vol'] - mf_today['sell_lg_vol'] - mf_today['sell_elg_vol']
                    df = pd.merge(df, mf_today[['ts_code', 'net_mf']], on='ts_code', how='left')
                else:
                    df['net_mf'] = 0
            else:
                df['net_mf'] = 0
        except:
            df['net_mf'] = 0

        # --- 筛选与评分 ---
        
        # 过滤条件
        df = df[df['close'] >= params['min_price']]
        df = df[df['pct_chg'] < 9.5] 
        df = df[df['pct_chg'] > -9.5]
        df = df[(df['turnover_rate'] >= params['min_turnover']) & (df['turnover_rate'] <= params['max_turnover'])]
        df['circ_mv_yi'] = df['circ_mv'] / 10000
        df = df[(df['circ_mv_yi'] >= params['min_mv']) & (df['circ_mv_yi'] <= params['max_mv'])]
        
        # 形态：上影线
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
# 4. 主程序入口
# ---------------------------
def main():
    st.title("🚀 选股王 2025 (硬盘断点续传版)")
    st.caption("✅ 已启用本地缓存：程序崩溃重启后将自动跳过已下载日期")
    
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
        
        # 日期处理
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        today_str = datetime.now().strftime('%Y%m%d')
        if end_str >= today_str: end_str = get_real_trade_date(today_str)
        
        trade_days = get_trade_cal(start_str, end_str)
        if not trade_days:
            st.error("无有效交易日")
            return
        
        st.info(f"回测区间: {trade_days[0]} - {trade_days[-1]} | {len(trade_days)} 天")
        
        # 1. 智能预加载 (Disk Cache)
        if not prefetch_data_stable(trade_days): return
        
        # 2. 执行回测
        params = {'min_price': min_price, 'min_mv': min_mv, 'max_mv': max_mv, 
                  'min_turnover': 3.0, 'max_turnover': 30.0, 'top_k': top_k}
        
        results = []
        progress = st.progress(0)
        
        for i, date in enumerate(trade_days):
            progress.progress((i+1)/len(trade_days), text=f"回测分析: {date}")
            selected = run_strategy(date, params)
            
            if not selected.empty:
                # 收益计算 (保留 Loc 修复)
                if i + 1 < len(trade_days):
                    next_date = trade_days[i+1]
                    try:
                        idx = pd.IndexSlice
                        if next_date in GLOBAL_DATA['daily'].index.get_level_values(0):
                            next_quotes = GLOBAL_DATA['daily'].loc[idx[next_date, :]]
                            for _, row in selected.iterrows():
                                code = row['ts_code']
                                ret = 0.0
                                if code in next_quotes.index:
                                    try:
                                        nb = next_quotes.loc[code]
                                        if isinstance(nb, pd.DataFrame): nb = nb.iloc[0]
                                        if nb['open'] > 0:
                                            ret = (nb['close'] - nb['open']) / nb['open'] * 100
                                    except: pass
                                results.append({'日期': date, '代码': code, '收益(%)': ret})
                    except: pass
        
        progress.empty()
        
        # 3. 结果展示
        if results:
            df_res = pd.DataFrame(results)
            st.divider()
            
            daily_ret = df_res.groupby('日期')['收益(%)'].mean().reset_index()
            daily_ret['策略净值'] = (1 + daily_ret['收益(%)']/100).cumprod()
            
            total_ret = (daily_ret['策略净值'].iloc[-1] - 1) * 100
            win_rate = (daily_ret['收益(%)'] > 0).mean() * 100
            max_dd = ((daily_ret['策略净值'].cummax() - daily_ret['策略净值']) / daily_ret['策略净值'].cummax()).max() * 100
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("累计收益", f"{total_ret:.2f}%")
            k2.metric("日胜率", f"{win_rate:.1f}%")
            k3.metric("最大回撤", f"{max_dd:.2f}%")
            k4.metric("交易天数", len(daily_ret))
            
            st.area_chart(daily_ret.set_index('日期')['策略净值'])
            st.dataframe(df_res)
        else:
            st.warning("未触发选股信号")

if __name__ == '__main__':
    main()
