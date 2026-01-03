# -*- coding: utf-8 -*-
"""
选股王 · V30.12.3 Pro (UI优化 & 5线程安全版)
------------------------------------------------
修改日志：
1. **UI调整**：Token输入和开始按钮移至主界面，不再隐藏在侧边栏。
2. **安全线程**：强制锁定为 5 线程，杜绝限流风险。
3. **数据诊断**：增加数据拉取成功后的样本展示，防止“0选股”问题。
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
st.set_page_config(page_title="选股王 Pro (UI优化版)", layout="wide")

# ---------------------------
# 工具函数
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
        st.error(f"Tushare Token 无效或连接失败: {e}")
        return None

def get_trade_cal(start_date, end_date):
    if pro is None: return []
    try:
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
        return df[df['is_open'] == 1]['cal_date'].tolist()
    except:
        return []

# ---------------------------
# 核心：批量数据预加载 (5线程安全版)
# ---------------------------
def prefetch_data(trade_days):
    global pro, GLOBAL_DATA
    if not trade_days: return False
    
    start_dt = trade_days[0]
    end_dt = trade_days[-1]
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # 切分时间段，每 20 天一段
        chunks = [trade_days[i:i + 20] for i in range(0, len(trade_days), 20)]
        
        tasks = {
            'daily': lambda s, e: pro.daily(start_date=s, end_date=e),
            'adj_factor': lambda s, e: pro.adj_factor(start_date=s, end_date=e),
            'daily_basic': lambda s, e: pro.daily_basic(start_date=s, end_date=e, fields='ts_code,trade_date,turnover_rate,circ_mv,total_mv,pe,pb'),
            'moneyflow': lambda s, e: pro.moneyflow(start_date=s, end_date=e),
        }
        
        total_steps = len(tasks) * len(chunks)
        current_step = 0
        
        # 强制使用 5 线程，绝对安全
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for key, api_call in tasks.items():
                status_text.text(f"正在拉取 {key} 数据 (5线程安全模式)...")
                futures = []
                for chunk in chunks:
                    if not chunk: continue
                    s, e = chunk[0], chunk[-1]
                    futures.append(executor.submit(api_call, s, e))
                
                results = []
                for f in concurrent.futures.as_completed(futures):
                    try:
                        res = f.result()
                        if res is not None and not res.empty:
                            results.append(res)
                    except Exception as e:
                        st.warning(f"部分数据拉取失败: {e}")
                    
                    current_step += 1
                    progress_bar.progress(min(current_step / total_steps, 1.0))
                
                if results:
                    full_df = pd.concat(results).drop_duplicates()
                    # 关键：确保 ts_code 和 trade_date 都是字符串且无空格
                    if 'trade_date' in full_df.columns:
                        full_df['trade_date'] = full_df['trade_date'].astype(str).str.strip()
                    if 'ts_code' in full_df.columns:
                        full_df['ts_code'] = full_df['ts_code'].astype(str).str.strip()
                        
                    # 建立索引
                    if 'ts_code' in full_df.columns and 'trade_date' in full_df.columns:
                        full_df.set_index(['trade_date', 'ts_code'], inplace=True)
                        full_df.sort_index(inplace=True)
                        
                    GLOBAL_DATA[key] = full_df
        
        status_text.success("数据加载完成！")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        return True

    except Exception as e:
        st.error(f"严重错误: 数据预加载失败 - {e}")
        return False

# ---------------------------
# 策略核心
# ---------------------------
def run_strategy(current_date, params):
    # 1. 获取数据
    try:
        # 使用 idx 切片，兼容性更强
        idx = pd.IndexSlice
        daily_today = GLOBAL_DATA['daily'].loc[idx[current_date, :]]
        basic_today = GLOBAL_DATA['daily_basic'].loc[idx[current_date, :]]
        # adj 和 moneyflow 是可选的，如果没有不报错
        try:
            adj_today = GLOBAL_DATA['adj_factor'].loc[idx[current_date, :]]
        except:
            adj_today = pd.DataFrame()
            
        try:
            mf_today = GLOBAL_DATA['moneyflow'].loc[idx[current_date, :]]
        except:
            mf_today = pd.DataFrame()
            
    except KeyError:
        return pd.DataFrame() # 当天无数据

    # 2. 合并数据
    # daily_today 的 index 是 (trade_date, ts_code)，loc 后变成了 ts_code (如果 trade_date 唯一)
    # 这里的 reset_index 很重要，确保 ts_code 变成列，方便 merge
    df = daily_today.reset_index()
    if 'ts_code' not in df.columns: # 只有一个 level
        df['ts_code'] = df.index
        
    # 合并 Basic
    basic_temp = basic_today.reset_index()
    if 'ts_code' not in basic_temp.columns: basic_temp['ts_code'] = basic_temp.index
    df = pd.merge(df, basic_temp[['ts_code', 'circ_mv', 'turnover_rate']], on='ts_code', how='inner')
    
    # 合并 Moneyflow (如有)
    if not mf_today.empty:
        mf_temp = mf_today.reset_index()
        if 'ts_code' not in mf_temp.columns: mf_temp['ts_code'] = mf_temp.index
        # 计算主力净流入
        mf_temp['net_mf'] = mf_temp['buy_lg_vol'] + mf_temp['buy_elg_vol'] - mf_temp['sell_lg_vol'] - mf_temp['sell_elg_vol']
        df = pd.merge(df, mf_temp[['ts_code', 'net_mf']], on='ts_code', how='left')
    else:
        df['net_mf'] = 0

    # 3. 筛选逻辑
    # 价格
    df = df[df['close'] >= params['min_price']]
    
    # 涨幅 (剔除已经涨停的，比如 > 9.8% 且 High=Close，这里简单剔除 > 9%)
    df = df[df['pct_chg'] < 9.5]
    
    # 换手率
    df = df[(df['turnover_rate'] >= params['min_turnover']) & (df['turnover_rate'] <= params['max_turnover'])]
    
    # 市值 (万 -> 亿)
    df['circ_mv_yi'] = df['circ_mv'] / 10000
    df = df[(df['circ_mv_yi'] >= params['min_mv']) & (df['circ_mv_yi'] <= params['max_mv'])]
    
    # 形态：上影线 < 3%
    # (High - Max(Open, Close)) / Close
    df['max_oc'] = df[['open', 'close']].max(axis=1)
    df['upper_shadow'] = (df['high'] - df['max_oc']) / df['close']
    df = df[df['upper_shadow'] <= 0.05]
    
    if df.empty: return pd.DataFrame()

    # 4. 评分
    # 简单评分：换手率 * 10 + 资金流得分
    df['score'] = df['turnover_rate']
    
    # 资金流加分
    df.loc[df['net_mf'] > 0, 'score'] += 50
    
    return df.sort_values(by='score', ascending=False).head(params['top_k'])

# ---------------------------
# 主程序
# ---------------------------
def main():
    st.title("🚀 选股王 Pro (极速回测版)")
    
    # --- 布局优化：Token 和 按钮 放在主区域 ---
    c1, c2 = st.columns([3, 1])
    with c1:
        token = st.text_input("在此输入 Tushare Token", value="", type="password", placeholder="粘贴您的 Token")
    with c2:
        st.write("") # 占位
        st.write("") 
        start_btn = st.button("开始回测 ▶", type="primary", use_container_width=True)

    # --- 侧边栏：仅放参数 ---
    with st.sidebar:
        st.header("⚙️ 策略参数")
        
        st.subheader("时间范围")
        start_date = st.date_input("开始日期", datetime(2025, 1, 1))
        end_date = st.date_input("结束日期", datetime(2025, 3, 1))
        
        st.subheader("选股条件")
        min_price = st.number_input("最低股价 (元)", 0.0, 1000.0, 5.0)
        min_mv = st.number_input("最小流通市值 (亿)", 0.0, 1000.0, 20.0)
        max_mv = st.number_input("最大流通市值 (亿)", 0.0, 5000.0, 500.0)
        min_turnover = st.number_input("最小换手率 (%)", 0.0, 100.0, 3.0)
        max_turnover = st.number_input("最大换手率 (%)", 0.0, 100.0, 25.0)
        
        st.subheader("风控")
        top_k = st.slider("每日持仓数", 1, 10, 5)

    # --- 点击开始后的逻辑 ---
    if start_btn:
        if not token:
            st.error("请先输入 Token！")
            return
            
        global pro
        with st.spinner("正在连接 Tushare..."):
            pro = init_tushare(token)
            if not pro: return
        
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # 1. 获取日历
        trade_days = get_trade_cal(start_str, end_str)
        if not trade_days:
            st.error("未获取到交易日历，请检查日期范围或网络。")
            return
        st.info(f"回测区间: {start_str} 至 {end_str}，共 {len(trade_days)} 个交易日")
        
        # 2. 拉取数据
        if not prefetch_data(trade_days): return
        
        # --- 调试：检查数据是否为空 ---
        st.write("--- 数据完整性检查 ---")
        if not GLOBAL_DATA['daily'].empty:
            st.success(f"✅ 行情数据已加载: {len(GLOBAL_DATA['daily'])} 条")
            st.dataframe(GLOBAL_DATA['daily'].head(3)) # 展示几条数据，确保不是空的
        else:
            st.error("❌ 行情数据 (daily) 为空！无法进行回测。")
            return
            
        if not GLOBAL_DATA['daily_basic'].empty:
            st.success(f"✅ 每日指标已加载: {len(GLOBAL_DATA['daily_basic'])} 条")
        else:
            st.warning("⚠️ 每日指标 (daily_basic) 为空，可能导致无法筛选市值和换手率。")

        # 3. 回测循环
        params = {
            'min_price': min_price,
            'min_mv': min_mv,
            'max_mv': max_mv,
            'min_turnover': min_turnover,
            'max_turnover': max_turnover,
            'top_k': top_k
        }
        
        results_log = []
        progress = st.progress(0)
        
        # 提前转换 full_trade_days 为 list 以便查找 next_day
        # 这里的 trade_days 已经是 list
        
        for i, date in enumerate(trade_days):
            progress.progress((i+1)/len(trade_days), text=f"正在分析: {date}")
            
            selected = run_strategy(date, params)
            
            if not selected.empty:
                # 计算次日收益
                if i + 1 < len(trade_days):
                    next_date = trade_days[i+1]
                    try:
                        idx = pd.IndexSlice
                        next_quotes = GLOBAL_DATA['daily'].loc[idx[next_date, :]]
                        
                        # 批量计算收益
                        for _, row in selected.iterrows():
                            code = row['ts_code']
                            # 尝试获取次日数据
                            if code in next_quotes.index.get_level_values('ts_code'):
                                # 注意：如果 loc 得到的是 Series (只有一只股票) 还是 DataFrame
                                # 使用 xs 安全获取
                                try:
                                    next_bar = next_quotes.xs(code, level='ts_code')
                                    # 如果 xs 结果是 DataFrame (很少见，除非数据重复)，取第一行
                                    if isinstance(next_bar, pd.DataFrame):
                                        next_bar = next_bar.iloc[0]
                                        
                                    buy_price = next_bar['open']
                                    sell_price = next_bar['close']
                                    ret = (sell_price - buy_price) / buy_price * 100
                                except:
                                    ret = 0.0
                            else:
                                ret = 0.0 # 停牌或缺失
                                
                            results_log.append({
                                '日期': date,
                                '代码': code,
                                '名称': code, # 暂无名称
                                '得分': row['score'],
                                '次日收益(%)': ret
                            })
                    except Exception as e:
                        pass # 某天数据缺失不影响整体
        
        progress.empty()
        
        if results_log:
            df_res = pd.DataFrame(results_log)
            st.success(f"回测完成！共产生 {len(df_res)} 次交易信号")
            
            # 展示统计
            avg_ret = df_res['次日收益(%)'].mean()
            win_rate = (df_res['次日收益(%)'] > 0).mean() * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("平均收益", f"{avg_ret:.2f}%")
            c2.metric("胜率", f"{win_rate:.1f}%")
            c3.metric("总信号数", len(df_res))
            
            st.dataframe(df_res)
            
            # 资金曲线
            df_curve = df_res.groupby('日期')['次日收益(%)'].mean().reset_index()
            df_curve['净值'] = (1 + df_curve['次日收益(%)']/100).cumprod()
            st.line_chart(df_curve.set_index('日期')['净值'])
            
        else:
            st.error("⚠️ 依然未选出股票。可能原因：\n1. 过滤条件太严苛（请尝试调低市值门槛或放宽价格限制）。\n2. 刚开年的几天可能数据不全。")

if __name__ == '__main__':
    main()
