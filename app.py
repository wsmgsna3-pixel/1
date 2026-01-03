# -*- coding: utf-8 -*-
"""
选股王 · V30.12.3 Pro (智能稳健版)
------------------------------------------------
修复日志：
1. **智能重试**：并发拉取失败时，自动切换为单线程串行拉取，解决 rate limit 问题。
2. **假日修正**：如果所选日期是节假日/未来，自动修正为最近的一个交易日。
3. **数据熔断**：如果缺少市值/换手率数据，直接报错提示，不再输出 0 结果。
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
st.set_page_config(page_title="选股王 2025 稳健版", layout="wide")

# ---------------------------
# 工具函数
# ---------------------------
@st.cache_resource
def init_tushare(token):
    if not token: return None
    try:
        api = ts.pro_api(token)
        # 验证连通性
        api.trade_cal(start_date='20250101', end_date='20250101')
        return api
    except Exception as e:
        st.error(f"Token 连接失败: {e}")
        return None

def get_real_trade_date(date_str):
    """
    如果传入的日期是假日或未来，自动寻找最近的一个过去交易日
    """
    if pro is None: return date_str
    try:
        # 获取该日期前后10天的日历
        start = (datetime.strptime(date_str, '%Y%m%d') - timedelta(days=10)).strftime('%Y%m%d')
        end = date_str
        df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
        if not df.empty:
            return df['cal_date'].iloc[-1] # 返回最近的一个交易日
        return date_str
    except:
        return date_str

def get_trade_cal(start_date, end_date):
    if pro is None: return []
    try:
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
        return df[df['is_open'] == 1]['cal_date'].tolist()
    except:
        return []

# ---------------------------
# 核心：双模数据预加载 (并发+串行兜底)
# ---------------------------
def fetch_worker(dt, api_type):
    """ 单个任务函数 """
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

def prefetch_data(trade_days):
    global pro, GLOBAL_DATA
    if not trade_days: return False
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    data_types = ['daily', 'daily_basic', 'adj_factor', 'moneyflow']
    
    for d_type in data_types:
        status_text.text(f"🚀 正在拉取 {d_type} ...")
        results = []
        failed_dates = []
        
        # --- 第一阶段：并发拉取 (速度快) ---
        # 降级为 4 线程，提高成功率
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_date = {executor.submit(fetch_worker, d, d_type): d for d in trade_days}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_date):
                dt = future_to_date[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results.append(data)
                    else:
                        # 记录空数据或失败的日期
                        # 注意：有些日期确实可能没数据（比如刚开市），但通常 daily_basic 不会全空
                        failed_dates.append(dt)
                except:
                    failed_dates.append(dt)
                
                completed += 1
                # 进度条
                base_progress = data_types.index(d_type) * 0.25
                curr_progress = base_progress + (completed / len(trade_days)) * 0.25
                progress_bar.progress(min(curr_progress, 1.0))

        # --- 第二阶段：智能兜底 (串行重试) ---
        # 如果 daily_basic 这种关键数据缺失太多，尝试单线程重试
        if d_type in ['daily', 'daily_basic'] and len(results) < len(trade_days) * 0.9:
            status_text.warning(f"⚠️ {d_type} 并发拉取不完整，正在切换单线程补全...")
            
            # 对失败的日期进行重试 (最多重试前 10 个，防止卡死，或者全部重试)
            # 这里简单起见，如果整体数据量太少，我们针对 trade_days 里缺失的进行补录
            existing_dates = set()
            for df in results:
                if 'trade_date' in df.columns and not df.empty:
                    existing_dates.add(df['trade_date'].iloc[0])
            
            missing_dates = [d for d in trade_days if d not in existing_dates]
            
            for md in missing_dates:
                time.sleep(0.1) # 强制间隔
                retry_data = fetch_worker(md, d_type)
                if retry_data is not None and not retry_data.empty:
                    results.append(retry_data)

        # 合并数据
        if results:
            full_df = pd.concat(results)
            # 清洗
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
                st.error("❌ 严重错误：无法获取每日指标数据 (daily_basic)。请检查您的积分权限。此数据缺失会导致无法选股。")
                return False
            else:
                st.warning(f"⚠️ {d_type} 数据拉取为空，将跳过相关因子计算。")

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
        
        # 1. 检查当日数据是否存在
        if current_date not in GLOBAL_DATA['daily'].index.get_level_values(0):
            return pd.DataFrame()
            
        daily_today = GLOBAL_DATA['daily'].loc[idx[current_date, :]]
        
        # 关键修复：daily_basic 必须有
        if current_date not in GLOBAL_DATA['daily_basic'].index.get_level_values(0):
            # 尝试容错：如果是 adj_factor 缺了还能跑，basic 缺了不能跑
            return pd.DataFrame()
            
        basic_today = GLOBAL_DATA['daily_basic'].loc[idx[current_date, :]]
        
        # 2. 合并
        df = daily_today.reset_index()
        if 'ts_code' not in df.columns: df['ts_code'] = df.index
        
        basic_temp = basic_today.reset_index()
        if 'ts_code' not in basic_temp.columns: basic_temp['ts_code'] = basic_temp.index
        
        # Inner Join
        df = pd.merge(df, basic_temp[['ts_code', 'circ_mv', 'turnover_rate']], on='ts_code', how='inner')
        
        # 资金流 (可选)
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

        # --- 过滤逻辑 ---
        df = df[df['close'] >= params['min_price']]
        df = df[df['pct_chg'] < 9.5]
        df = df[(df['turnover_rate'] >= params['min_turnover']) & (df['turnover_rate'] <= params['max_turnover'])]
        
        df['circ_mv_yi'] = df['circ_mv'] / 10000
        df = df[(df['circ_mv_yi'] >= params['min_mv']) & (df['circ_mv_yi'] <= params['max_mv'])]
        
        if df.empty: return pd.DataFrame()

        # --- 评分 ---
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
    st.title("🚀 选股王 2025 稳健回测")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        token = st.text_input("Tushare Token", value="", type="password")
    with c2:
        st.write("")
        st.write("")
        start_btn = st.button("开始回测 ▶", type="primary", use_container_width=True)

    with st.sidebar:
        st.header("⚙️ 参数设置")
        # 默认设为历史区间
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
        
        # --- 智能修正日期 ---
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # 如果用户选了今天(假日)，自动修正结束日期为最近的交易日
        today_str = datetime.now().strftime('%Y%m%d')
        if end_str >= today_str:
            st.toast(f"检测到日期 {end_str} 可能无数据，正在自动校正...")
            end_str = get_real_trade_date(today_str)
            st.info(f"📅 日期已自动校正: 结束日期调整为 {end_str} (最近交易日)")
        
        trade_days = get_trade_cal(start_str, end_str)
        if not trade_days:
            st.error("未获取到交易日历，请检查日期范围。")
            return
        
        st.info(f"回测区间: {trade_days[0]} - {trade_days[-1]} | 共 {len(trade_days)} 个交易日")
        
        # 执行预加载
        if not prefetch_data(trade_days): return
        
        # 再次检查
        if GLOBAL_DATA['daily_basic'].empty:
            st.error("❌ 核心数据 daily_basic 为空！程序无法运行。请检查 Token 权限或稍后重试。")
            return
        
        # 回测循环
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
            
            st.area_chart(daily_ret.set_index('日期')['策略净值'])
            with st.expander("查看详细交易记录"):
                st.dataframe(df_res)
        else:
            st.warning("⚠️ 依然未触发选股信号。这通常意味着过滤条件过严 (如最低股价 10元 配合 20亿市值 可能筛掉了大部分票)。建议调低参数尝试。")

if __name__ == '__main__':
    main()
