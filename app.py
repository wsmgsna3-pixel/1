# -*- coding: utf-8 -*-
"""
选股王 · V30.12.3 Pro (2025实战修复版 - V2)
------------------------------------------------
修复日志：
1. **紧急修复**：修正第 109 行的 SyntaxError (for 循环语法错误)。
2. **核心逻辑**：保持按天并发拉取，10线程全速，剔除垃圾股。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime
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
st.set_page_config(page_title="选股王 2025 回测版", layout="wide")

# ---------------------------
# 工具函数
# ---------------------------
@st.cache_resource
def init_tushare(token):
    if not token: return None
    try:
        api = ts.pro_api(token)
        # 验证 Token
        api.trade_cal(start_date='20250101', end_date='20250101')
        return api
    except Exception as e:
        st.error(f"Token 无效: {e}")
        return None

def get_trade_cal(start_date, end_date):
    if pro is None: return []
    try:
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
        return df[df['is_open'] == 1]['cal_date'].tolist()
    except:
        return []

# ---------------------------
# 核心：按天并发预加载 (修复数据为空问题)
# ---------------------------
def prefetch_data(trade_days):
    global pro, GLOBAL_DATA
    if not trade_days: return False
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 任务定义：每天拉一次，确保不超过单次 6000 行的限制
    tasks_dates = [[d] for d in trade_days] 
    
    # 定义 API 调用包装器
    def fetch_worker(date_list, api_type):
        dt = date_list[0]
        try:
            if api_type == 'daily':
                return pro.daily(trade_date=dt)
            elif api_type == 'adj_factor':
                return pro.adj_factor(trade_date=dt)
            elif api_type == 'daily_basic':
                return pro.daily_basic(trade_date=dt, fields='ts_code,trade_date,turnover_rate,circ_mv,total_mv,pe,pb')
            elif api_type == 'moneyflow':
                return pro.moneyflow(trade_date=dt)
        except Exception:
            return None

    # 开始拉取
    try:
        data_types = ['daily', 'daily_basic', 'adj_factor', 'moneyflow']
        
        for d_type in data_types:
            status_text.text(f"🚀 正在拉取 {d_type} (按天并发，防超限)...")
            results = []
            
            # 10 线程并发
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_date = {executor.submit(fetch_worker, d, d_type): d for d in tasks_dates}
                
                completed_count = 0
                # --- 修复点：这里修正了语法错误 ---
                for future in concurrent.futures.as_completed(future_to_date):
                    try:
                        data = future.result()
                        if data is not None and not data.empty:
                            results.append(data)
                    except:
                        pass
                    
                    completed_count += 1
                    base_progress = data_types.index(d_type) * 0.25
                    current_progress = base_progress + (completed_count / len(tasks_dates)) * 0.25
                    progress_bar.progress(min(current_progress, 1.0))

            if results:
                full_df = pd.concat(results)
                # 格式清洗
                if 'trade_date' in full_df.columns:
                    full_df['trade_date'] = full_df['trade_date'].astype(str).str.strip()
                if 'ts_code' in full_df.columns:
                    full_df['ts_code'] = full_df['ts_code'].astype(str).str.strip()
                
                # 去重
                full_df.drop_duplicates(subset=['trade_date', 'ts_code'], inplace=True)
                
                # 建立索引
                full_df.set_index(['trade_date', 'ts_code'], inplace=True)
                full_df.sort_index(inplace=True)
                
                GLOBAL_DATA[d_type] = full_df
            else:
                st.warning(f"⚠️ {d_type} 数据拉取为空，可能是权限不足或当天无数据。")

        status_text.success("✅ 所有数据加载完成！")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        return True

    except Exception as e:
        st.error(f"严重错误: {e}")
        return False

# ---------------------------
# 策略核心
# ---------------------------
def run_strategy(current_date, params):
    try:
        idx = pd.IndexSlice
        # 1. 获取当日数据
        daily_today = GLOBAL_DATA['daily'].loc[idx[current_date, :]]
        basic_today = GLOBAL_DATA['daily_basic'].loc[idx[current_date, :]]
        
        # 2. 基础合并
        df = daily_today.reset_index()
        if 'ts_code' not in df.columns: df['ts_code'] = df.index
        
        basic_temp = basic_today.reset_index()
        if 'ts_code' not in basic_temp.columns: basic_temp['ts_code'] = basic_temp.index
        
        df = pd.merge(df, basic_temp[['ts_code', 'circ_mv', 'turnover_rate']], on='ts_code', how='inner')
        
        # 3. 资金流 (Optional)
        try:
            mf_today = GLOBAL_DATA['moneyflow'].loc[idx[current_date, :]]
            if not mf_today.empty:
                mf_temp = mf_today.reset_index()
                if 'ts_code' not in mf_temp.columns: mf_temp['ts_code'] = mf_temp.index
                mf_temp['net_mf'] = mf_temp['buy_lg_vol'] + mf_temp['buy_elg_vol'] - mf_temp['sell_lg_vol'] - mf_temp['sell_elg_vol']
                df = pd.merge(df, mf_temp[['ts_code', 'net_mf']], on='ts_code', how='left')
            else:
                df['net_mf'] = 0
        except:
            df['net_mf'] = 0

        # --- 过滤逻辑 ---
        # 1. 价格过滤
        df = df[df['close'] >= params['min_price']]
        # 2. 涨幅过滤
        df = df[df['pct_chg'] < 9.5]
        # 3. 换手率
        df = df[(df['turnover_rate'] >= params['min_turnover']) & (df['turnover_rate'] <= params['max_turnover'])]
        # 4. 市值
        df['circ_mv_yi'] = df['circ_mv'] / 10000
        df = df[(df['circ_mv_yi'] >= params['min_mv']) & (df['circ_mv_yi'] <= params['max_mv'])]
        
        if df.empty: return pd.DataFrame()

        # --- 评分逻辑 ---
        df['score'] = df['turnover_rate']
        df.loc[df['net_mf'] > 0, 'score'] += 20
        df['upper_shadow'] = (df['high'] - df['close']) / df['close']
        df.loc[df['upper_shadow'] < 0.01, 'score'] += 10
        
        return df.sort_values(by='score', ascending=False).head(params['top_k'])

    except KeyError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ---------------------------
# 主程序
# ---------------------------
def main():
    st.title("🚀 选股王 2025 实战回测")
    
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
        min_price = st.number_input("最低股价 (元)", 0.0, 500.0, 10.0, help="严格剔除 10 元以下股票")
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
        
        trade_days = get_trade_cal(start_str, end_str)
        if not trade_days:
            st.error("未获取到交易日历，请检查网络。")
            return
        
        st.info(f"📅 回测区间: {start_str} - {end_str} | 交易日: {len(trade_days)} 天")
        
        if not prefetch_data(trade_days): return
        
        if GLOBAL_DATA['daily'].empty:
            st.error("❌ 数据依然为空！请检查您的 Token 积分。")
            return
        else:
            st.success(f"📊 数据加载成功！共 {len(GLOBAL_DATA['daily'])} 条行情记录。")

        params = {
            'min_price': min_price,
            'min_mv': min_mv,
            'max_mv': max_mv,
            'min_turnover': 3.0,
            'max_turnover': 30.0,
            'top_k': top_k
        }
        
        results = []
        progress = st.progress(0)
        
        for i, date in enumerate(trade_days):
            progress.progress((i+1)/len(trade_days), text=f"正在选股: {date}")
            selected = run_strategy(date, params)
            
            if not selected.empty:
                if i + 1 < len(trade_days):
                    next_date = trade_days[i+1]
                    try:
                        idx = pd.IndexSlice
                        if next_date in GLOBAL_DATA['daily'].index.get_level_values(0):
                            next_quotes = GLOBAL_DATA['daily'].loc[idx[next_date, :]]
                            for _, row in selected.iterrows():
                                code = row['ts_code']
                                ret = 0.0
                                if code in next_quotes.index.get_level_values('ts_code'):
                                    try:
                                        nb = next_quotes.xs(code, level='ts_code')
                                        if isinstance(nb, pd.DataFrame): nb = nb.iloc[0]
                                        ret = (nb['close'] - nb['open']) / nb['open'] * 100
                                    except: pass
                                
                                results.append({'日期': date, '代码': code, '收益(%)': ret})
                    except: pass
        
        progress.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            st.divider()
            st.subheader("📈 回测报告")
            
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
            
            st.area_chart(daily_ret.set_index('日期')['策略净值'], color="#FF4B4B")
            
            with st.expander("查看详细交易记录"):
                st.dataframe(df_res)
        else:
            st.warning("在此期间未触发选股信号。")

if __name__ == '__main__':
    main()
