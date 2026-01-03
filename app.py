# -*- coding: utf-8 -*-
"""
选股王 · V30.12.3 Pro (最终收益修复版)
------------------------------------------------
修复日志：
1. **核心修复**：修正收益率计算中的索引错误 (xs -> loc)，彻底解决收益为 0 的问题。
2. **逻辑校正**：强制确保交易日历按日期升序排列，防止回测顺序错乱。
3. **稳健性**：保留了单线程自动补全数据的机制，确保数据完整。
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

warnings.filterwarnings("ignore")

# ---------------------------
# 全局数据存储
# ---------------------------
GLOBAL_DATA = {
    'daily': pd.DataFrame(),
    'daily_basic': pd.DataFrame(),
    'moneyflow': pd.DataFrame(),
    'adj_factor': pd.DataFrame()
}
pro = None

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="选股王 2025 最终版", layout="wide")

# ---------------------------
# 工具函数
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
        if not df.empty:
            return df['cal_date'].iloc[-1]
        return date_str
    except:
        return date_str

def get_trade_cal(start_date, end_date):
    if pro is None: return []
    try:
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
        df = df[df['is_open'] == 1]
        return sorted(df['cal_date'].tolist()) # 强制升序
    except:
        return []

# ---------------------------
# 核心：双模数据预加载
# ---------------------------
def fetch_worker(dt, api_type):
    try:
        if api_type == 'daily':
            return pro.daily(trade_date=dt)
        elif api_type == 'adj_factor':
            return pro.adj_factor(trade_date=dt)
        elif api_type == 'daily_basic':
            return pro.daily_basic(trade_date=dt, fields='ts_code,trade_date,turnover_rate,circ_mv,total_mv,pe,pb')
        elif api_type == 'moneyflow':
            return pro.moneyflow(trade_date=dt)
    except:
        return None

def prefetch_data(trade_days):
    global pro, GLOBAL_DATA
    if not trade_days: return False
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    data_types = ['daily', 'daily_basic', 'adj_factor', 'moneyflow']
    
    for d_type in data_types:
        status_text.text(f"🚀 正在拉取 {d_type} ...")
        results = []
        
        # 1. 并发拉取
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_date = {executor.submit(fetch_worker, d, d_type): d for d in trade_days}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_date):
                data = future.result()
                if data is not None and not data.empty:
                    results.append(data)
                completed += 1
                base_progress = data_types.index(d_type) * 0.25
                curr_progress = base_progress + (completed / len(trade_days)) * 0.25
                progress_bar.progress(min(curr_progress, 1.0))

        # 2. 补漏 (单线程)
        if d_type in ['daily', 'daily_basic'] and len(results) < len(trade_days):
            existing_dates = set()
            for df in results:
                if 'trade_date' in df.columns and not df.empty:
                    existing_dates.add(df['trade_date'].iloc[0])
            
            missing_dates = [d for d in trade_days if d not in existing_dates]
            if missing_dates:
                status_text.warning(f"⚠️ {d_type} 正在单线程补全 {len(missing_dates)} 天数据...")
                for md in missing_dates:
                    retry_data = fetch_worker(md, d_type)
                    if retry_data is not None and not retry_data.empty:
                        results.append(retry_data)

        # 合并
        if results:
            full_df = pd.concat(results)
            if 'trade_date' in full_df.columns:
                full_df['trade_date'] = full_df['trade_date'].astype(str).str.strip()
            if 'ts_code' in full_df.columns:
                full_df['ts_code'] = full_df['ts_code'].astype(str).str.strip()
            
            full_df.drop_duplicates(subset=['trade_date', 'ts_code'], inplace=True)
            full_df.set_index(['trade_date', 'ts_code'], inplace=True)
            full_df.sort_index(inplace=True)
            GLOBAL_DATA[d_type] = full_df
        else:
            if d_type == 'daily_basic':
                st.error("❌ 严重错误：daily_basic 数据拉取失败。")
                return False

    status_text.success("✅ 数据加载完成！")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    return True

# ---------------------------
# 策略核心
# ---------------------------
def run_strategy(current_date, params):
    try:
        idx = pd.IndexSlice
        # 必须有 daily 和 daily_basic
        if current_date not in GLOBAL_DATA['daily'].index.get_level_values(0): return pd.DataFrame()
        if current_date not in GLOBAL_DATA['daily_basic'].index.get_level_values(0): return pd.DataFrame()
            
        daily_today = GLOBAL_DATA['daily'].loc[idx[current_date, :]]
        basic_today = GLOBAL_DATA['daily_basic'].loc[idx[current_date, :]]
        
        # 合并
        df = daily_today.reset_index()
        if 'ts_code' not in df.columns: df['ts_code'] = df.index
        basic_temp = basic_today.reset_index()
        if 'ts_code' not in basic_temp.columns: basic_temp['ts_code'] = basic_temp.index
        
        df = pd.merge(df, basic_temp[['ts_code', 'circ_mv', 'turnover_rate']], on='ts_code', how='inner')
        
        # 资金流
        try:
            if current_date in GLOBAL_DATA['moneyflow'].index.get_level_values(0):
                mf_today = GLOBAL_DATA['moneyflow'].loc[idx[current_date, :]]
                mf_temp = mf_today.reset_index()
                if 'ts_code' not in mf_temp.columns: mf_temp['ts_code'] = mf_temp.index
                mf_temp['net_mf'] = mf_temp['buy_lg_vol'] + mf_temp['buy_elg_vol'] - mf_temp['sell_lg_vol'] - mf_temp['sell_elg_vol']
                df = pd.merge(df, mf_temp[['ts_code', 'net_mf']], on='ts_code', how='left')
            else:
                df['net_mf'] = 0
        except:
            df['net_mf'] = 0

        # 过滤
        df = df[df['close'] >= params['min_price']]
        df = df[df['pct_chg'] < 9.5]
        df = df[(df['turnover_rate'] >= params['min_turnover']) & (df['turnover_rate'] <= params['max_turnover'])]
        
        df['circ_mv_yi'] = df['circ_mv'] / 10000
        df = df[(df['circ_mv_yi'] >= params['min_mv']) & (df['circ_mv_yi'] <= params['max_mv'])]
        
        if df.empty: return pd.DataFrame()

        # 评分
        df['score'] = df['turnover_rate']
        df.loc[df['net_mf'] > 0, 'score'] += 20
        df['upper_shadow'] = (df['high'] - df['close']) / df['close']
        df.loc[df['upper_shadow'] < 0.01, 'score'] += 10
        
        return df.sort_values(by='score', ascending=False).head(params['top_k'])

    except Exception:
        return pd.DataFrame()

# ---------------------------
# 主程序
# ---------------------------
def main():
    st.title("🚀 选股王 2025 最终修复版")
    
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
        
        # 日期修正
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        today_str = datetime.now().strftime('%Y%m%d')
        if end_str >= today_str:
            end_str = get_real_trade_date(today_str)
        
        trade_days = get_trade_cal(start_str, end_str)
        if not trade_days:
            st.error("无有效交易日")
            return
        
        st.info(f"回测区间: {trade_days[0]} - {trade_days[-1]} | {len(trade_days)} 天")
        
        if not prefetch_data(trade_days): return
        
        # 回测
        params = {'min_price': min_price, 'min_mv': min_mv, 'max_mv': max_mv, 
                  'min_turnover': 3.0, 'max_turnover': 30.0, 'top_k': top_k}
        
        results = []
        progress = st.progress(0)
        
        for i, date in enumerate(trade_days):
            progress.progress((i+1)/len(trade_days), text=f"分析: {date}")
            selected = run_strategy(date, params)
            
            if not selected.empty:
                # 获取次日数据计算收益
                if i + 1 < len(trade_days):
                    next_date = trade_days[i+1]
                    try:
                        idx = pd.IndexSlice
                        if next_date in GLOBAL_DATA['daily'].index.get_level_values(0):
                            # 这里获取的已经是只有 ts_code 索引的 DF
                            next_quotes = GLOBAL_DATA['daily'].loc[idx[next_date, :]]
                            
                            for _, row in selected.iterrows():
                                code = row['ts_code']
                                ret = 0.0
                                # 【关键修复】使用 .loc 而不是 .xs
                                if code in next_quotes.index:
                                    try:
                                        nb = next_quotes.loc[code]
                                        # 如果有重复代码，取第一行
                                        if isinstance(nb, pd.DataFrame): nb = nb.iloc[0]
                                        # 计算当日涨幅 (Close - Open) / Open
                                        ret = (nb['close'] - nb['open']) / nb['open'] * 100
                                    except: pass
                                
                                results.append({'日期': date, '代码': code, '收益(%)': ret})
                    except: pass
        
        progress.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            st.divider()
            
            # 统计
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
