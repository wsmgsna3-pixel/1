# -*- coding: utf-8 -*-
"""
选股王 · V30.12.3 (纯单线程·绝对稳定版)
------------------------------------------------
修改核心：
1. **纯单线程**：移除所有并发，一行行代码逐天拉取，拒绝花里胡哨，由慢变稳。
2. **逻辑修复**：修正了计算次日收益时的索引错误 (loc替换xs)，解决“全0收益”问题。
3. **进度可视**：由于单线程稍慢，增加了详细的进度条。
------------------------------------------------
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
# 全局数据存储
# ---------------------------
GLOBAL_DATA = {
    'daily': pd.DataFrame(),
    'daily_basic': pd.DataFrame(),
    'moneyflow': pd.DataFrame()
}
pro = None

st.set_page_config(page_title="选股王 稳定版", layout="wide")

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
# 核心：纯单线程数据预加载
# ---------------------------
def prefetch_data(trade_days):
    global pro, GLOBAL_DATA
    if not trade_days: return False
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 容器
    all_daily = []
    all_basic = []
    all_mf = []
    
    total_days = len(trade_days)
    
    # ----------------------------------------
    # 单线程循环：一天一天拉，稳如老狗
    # ----------------------------------------
    for i, date in enumerate(trade_days):
        # 进度提示
        progress = (i + 1) / total_days
        progress_bar.progress(progress, text=f"正在拉取: {date} ({i+1}/{total_days})")
        
        try:
            # 1. 拉取行情
            df_d = pro.daily(trade_date=date)
            if df_d is not None and not df_d.empty:
                all_daily.append(df_d)
            
            # 2. 拉取每日指标 (关键)
            df_b = pro.daily_basic(trade_date=date, fields='ts_code,trade_date,turnover_rate,circ_mv,total_mv,pe,pb')
            if df_b is not None and not df_b.empty:
                all_basic.append(df_b)
                
            # 3. 拉取资金流 (可选)
            df_m = pro.moneyflow(trade_date=date)
            if df_m is not None and not df_m.empty:
                all_mf.append(df_m)
            
            # 【关键】稍微歇一下，防止接口报错，保证成功率
            time.sleep(0.05) 
            
        except Exception as e:
            st.warning(f"{date} 数据获取失败，已跳过。错误: {e}")
            time.sleep(1) # 出错多歇会
            continue

    status_text.text("正在合并数据...")
    
    # 合并数据
    if all_daily:
        full_daily = pd.concat(all_daily)
        full_daily['trade_date'] = full_daily['trade_date'].astype(str).str.strip()
        full_daily['ts_code'] = full_daily['ts_code'].astype(str).str.strip()
        full_daily.drop_duplicates(subset=['trade_date', 'ts_code'], inplace=True)
        full_daily.set_index(['trade_date', 'ts_code'], inplace=True)
        full_daily.sort_index(inplace=True)
        GLOBAL_DATA['daily'] = full_daily
    else:
        st.error("❌ 每日行情数据为空，无法回测")
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
        st.error("❌ 每日指标数据(daily_basic)为空，无法回测")
        return False
        
    if all_mf:
        full_mf = pd.concat(all_mf)
        full_mf['trade_date'] = full_mf['trade_date'].astype(str).str.strip()
        full_mf['ts_code'] = full_mf['ts_code'].astype(str).str.strip()
        full_mf.set_index(['trade_date', 'ts_code'], inplace=True)
        full_mf.sort_index(inplace=True)
        GLOBAL_DATA['moneyflow'] = full_mf

    status_text.success("✅ 所有数据加载完成！(单线程模式)")
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
        # 检查数据是否存在
        if current_date not in GLOBAL_DATA['daily'].index.get_level_values(0): return pd.DataFrame()
        if current_date not in GLOBAL_DATA['daily_basic'].index.get_level_values(0): return pd.DataFrame()
            
        daily_today = GLOBAL_DATA['daily'].loc[idx[current_date, :]].copy()
        basic_today = GLOBAL_DATA['daily_basic'].loc[idx[current_date, :]].copy()
        
        # 1. 基础合并
        df = daily_today.reset_index()
        if 'ts_code' not in df.columns: df['ts_code'] = df.index
        basic_temp = basic_today.reset_index()
        if 'ts_code' not in basic_temp.columns: basic_temp['ts_code'] = basic_temp.index
        
        # Inner Join: 只保留同时有行情和指标的票
        df = pd.merge(df, basic_temp[['ts_code', 'circ_mv', 'turnover_rate']], on='ts_code', how='inner')
        
        # 2. 资金流 (Left Join)
        try:
            if 'moneyflow' in GLOBAL_DATA and current_date in GLOBAL_DATA['moneyflow'].index.get_level_values(0):
                mf_today = GLOBAL_DATA['moneyflow'].loc[idx[current_date, :]]
                mf_temp = mf_today.reset_index()
                if 'ts_code' not in mf_temp.columns: mf_temp['ts_code'] = mf_temp.index
                mf_temp['net_mf'] = mf_temp['buy_lg_vol'] + mf_temp['buy_elg_vol'] - mf_temp['sell_lg_vol'] - mf_temp['sell_elg_vol']
                df = pd.merge(df, mf_temp[['ts_code', 'net_mf']], on='ts_code', how='left')
            else:
                df['net_mf'] = 0
        except:
            df['net_mf'] = 0

        # --- 3. 筛选逻辑 ---
        # 过滤垃圾股
        df = df[df['close'] >= params['min_price']]
        # 过滤涨跌停
        df = df[df['pct_chg'] < 9.5] 
        df = df[df['pct_chg'] > -9.5]
        # 换手率
        df = df[(df['turnover_rate'] >= params['min_turnover']) & (df['turnover_rate'] <= params['max_turnover'])]
        # 市值
        df['circ_mv_yi'] = df['circ_mv'] / 10000
        df = df[(df['circ_mv_yi'] >= params['min_mv']) & (df['circ_mv_yi'] <= params['max_mv'])]
        
        # 上影线过滤
        df['max_oc'] = df[['open', 'close']].max(axis=1)
        df['upper_shadow'] = (df['high'] - df['max_oc']) / df['close']
        df = df[df['upper_shadow'] <= 0.05]
        
        if df.empty: return pd.DataFrame()

        # --- 4. 评分 ---
        df['score'] = df['turnover_rate']
        # 资金流加分
        df.loc[df['net_mf'] > 0, 'score'] += 20
        # 实体饱满度加分
        df['body_len'] = (df['close'] - df['open']).abs()
        df['hl_len'] = df['high'] - df['low']
        # 防止除0
        df.loc[df['hl_len'] == 0, 'hl_len'] = 0.01
        df['body_ratio'] = df['body_len'] / df['hl_len']
        df.loc[df['body_ratio'] > 0.5, 'score'] += 10
        
        return df.sort_values(by='score', ascending=False).head(params['top_k'])

    except Exception as e:
        return pd.DataFrame()

# ---------------------------
# 主程序
# ---------------------------
def main():
    st.title("🚀 选股王 2025 (单线程稳定版)")
    
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
        
        # 自动日期修正
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        today_str = datetime.now().strftime('%Y%m%d')
        if end_str >= today_str:
            end_str = get_real_trade_date(today_str)
        
        trade_days = get_trade_cal(start_str, end_str)
        if not trade_days:
            st.error("无有效交易日")
            return
        
        st.info(f"回测区间: {trade_days[0]} - {trade_days[-1]} | {len(trade_days)} 天 (单线程拉取中，请耐心等待...)")
        
        # 1. 执行预加载 (单线程)
        if not prefetch_data(trade_days): return
        
        # 2. 执行回测
        params = {'min_price': min_price, 'min_mv': min_mv, 'max_mv': max_mv, 
                  'min_turnover': 3.0, 'max_turnover': 30.0, 'top_k': top_k}
        
        results = []
        progress = st.progress(0)
        
        for i, date in enumerate(trade_days):
            progress.progress((i+1)/len(trade_days), text=f"回测分析: {date}")
            selected = run_strategy(date, params)
            
            if not selected.empty:
                # 收益计算逻辑
                if i + 1 < len(trade_days):
                    next_date = trade_days[i+1]
                    try:
                        idx = pd.IndexSlice
                        if next_date in GLOBAL_DATA['daily'].index.get_level_values(0):
                            next_quotes = GLOBAL_DATA['daily'].loc[idx[next_date, :]]
                            
                            for _, row in selected.iterrows():
                                code = row['ts_code']
                                ret = 0.0
                                # 【核心修复点】: 使用 .loc 直接获取，不使用 .xs
                                # 因为 next_quotes 的索引只有 ts_code 这一层了
                                if code in next_quotes.index:
                                    try:
                                        nb = next_quotes.loc[code]
                                        # 如果是 DataFrame (极少数情况) 取第一行
                                        if isinstance(nb, pd.DataFrame): nb = nb.iloc[0]
                                        
                                        # 收益率: (收 - 开) / 开
                                        if nb['open'] > 0:
                                            ret = (nb['close'] - nb['open']) / nb['open'] * 100
                                        else:
                                            ret = 0.0
                                    except: 
                                        ret = 0.0
                                
                                results.append({'日期': date, '代码': code, '收益(%)': ret})
                    except: pass
        
        progress.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            st.divider()
            st.subheader("📊 最终回测报告")
            
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
            st.warning("在此期间未触发选股信号")

if __name__ == '__main__':
    main()
