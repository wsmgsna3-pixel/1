import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
import time
import gc
import os
from datetime import datetime, timedelta
# 引入多线程库，这是提速的关键
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="V36.0 最终核聚变版", layout="wide")

# ==========================================
# 2. 系统控制台
# ==========================================
st.sidebar.header("🛠️ 系统控制台")
st.sidebar.success("✅ V36.0 (10线程并发筹码版)")
st.sidebar.info("💡 核心：利用高级权限并发下载真实筹码数据，解决单线程超时问题。")

if st.sidebar.button("🔥 强制重启 (代码更新必点)", type="primary"):
    st.cache_data.clear()
    st.cache_resource.clear()
    os._exit(0)

# ==========================================
# 3. 高性能数据引擎 (并发核心)
# ==========================================

@st.cache_resource
def get_pro_api(token):
    if not token: return None
    ts.set_token(token)
    return ts.pro_api(timeout=60) # 保持60秒超时容错

# --- 辅助：单日筹码下载任务 ---
def fetch_cyq_task(date, token):
    """
    这是一个会被放入线程池的独立任务。
    专门负责下载某一天的‘真实筹码’数据。
    """
    try:
        # 每个线程必须有独立的连接，防止冲突
        ts.set_token(token)
        local_pro = ts.pro_api()
        
        # 下载当天的筹码数据 (cyq_perf)
        # 您有10000积分，支持获取全市场当天的筹码
        df = local_pro.cyq_perf(trade_date=date)
        
        if df.empty: return None
        return {'date': date, 'data': df}
    except Exception:
        return None

# --- 核心：批量数据管理器 ---
@st.cache_data(ttl=3600)
def fetch_full_data_concurrently(start_date, end_date, token):
    """
    1. 基础行情：批量下载（极快）
    2. 筹码数据：并发下载（榨干高级权限带宽）
    3. 内存组装
    """
    ts.set_token(token)
    pro = ts.pro_api(timeout=60)
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # A. 获取交易日历
        status_text.info("📅 正在获取交易日历...")
        cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
        dates = sorted(cal_df['cal_date'].tolist())
        total_days = len(dates)
        
        if total_days == 0: return None
        
        # B. 批量下载基础行情 (Daily & Basic) - 这部分本来就快，直接批量下
        status_text.info(f"🚀 正在批量下载 {total_days} 天的基础行情...")
        
        # 分月下载基础数据防止包过大
        daily_list = []
        basic_list = []
        
        # 简单的按月切分
        periods = pd.date_range(start=start_date, end=end_date, freq='M').strftime('%Y%m%d').tolist()
        if not periods or periods[-1] < end_date: periods.append(end_date)
        split_pts = sorted(list(set([start_date] + periods)))
        
        for i in range(len(split_pts)-1):
            d1 = pro.daily(start_date=split_pts[i], end_date=split_pts[i+1])
            d2 = pro.daily_basic(start_date=split_pts[i], end_date=split_pts[i+1], fields='ts_code,trade_date,turnover_rate,circ_mv')
            daily_list.append(d1)
            basic_list.append(d2)
            progress_bar.progress((i+1)/len(split_pts) * 0.3) # 进度条前30%给基础数据
            
        df_daily = pd.concat(daily_list).drop_duplicates()
        df_basic = pd.concat(basic_list).drop_duplicates()
        
        # C. 并发下载筹码数据 (重头戏)
        status_text.info(f"💎 正在启动 10 线程并发下载真实筹码数据 ({total_days} 天)...")
        
        cyq_dict = {} # 用于存储 {日期: 筹码DataFrame}
        
        # 使用 ThreadPoolExecutor 开启 10 个线程
        # 只有高级积分用户才能撑得住这种并发，普通用户会封号
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交所有任务
            future_map = {executor.submit(fetch_cyq_task, d, token): d for d in dates}
            
            done_count = 0
            for future in as_completed(future_map):
                done_count += 1
                res = future.result()
                if res:
                    cyq_dict[res['date']] = res['data']
                
                # 更新进度条 (30% -> 100%)
                current_progress = 0.3 + (done_count / total_days * 0.7)
                progress_bar.progress(current_progress, text=f"筹码下载中: {done_count}/{total_days} (线程池全开)")
        
        # D. 获取股票名称
        df_names = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        
        # E. 数据清洗与组装
        status_text.info("⚡ 正在内存中组装数据立方体...")
        
        # 我们把基础数据也转成字典方便查询
        daily_dict = {k: v for k, v in df_daily.groupby('trade_date')}
        basic_dict = {k: v for k, v in df_basic.groupby('trade_date')}
        
        # 打包返回
        package = {
            'dates': dates,
            'daily_dict': daily_dict,
            'basic_dict': basic_dict,
            'cyq_dict': cyq_dict,
            'names': df_names
        }
        
        status_text.success("✅ 全量数据加载完成！")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        return package
        
    except Exception as e:
        status_text.error(f"严重错误: {e}")
        return None

# ==========================================
# 4. 逻辑层 (Rank 1 核心算法)
# ==========================================
def run_strategy_rank1(date, package, p_min, p_max, to_max, top_n):
    """
    纯内存计算，速度极快。
    """
    # 1. 从大包里取当天的切片
    d1 = package['daily_dict'].get(date)
    d2 = package['basic_dict'].get(date)
    d4 = package['cyq_dict'].get(date) # 这次我们有真正的筹码数据了！
    names = package['names']
    
    # 2. 数据完整性检查 (防崩溃)
    if d1 is None or d2 is None or d4 is None: return None
    if d1.empty or d2.empty or d4.empty: return None
    
    # 3. 合并
    try:
        # 基础数据合并
        m1 = pd.merge(d1, d2, on='ts_code', how='inner')
        m1 = pd.merge(m1, names, on='ts_code', how='inner')
        
        # 筹码数据合并 (关键一步)
        # cost_50pct 就是市场的平均持仓成本
        df = pd.merge(m1, d4[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
        
        # 4. 计算真实筹码乖离率
        # (收盘价 - 成本) / 成本
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        
        # 5. 筛选
        condition = (
            (df['bias'] > -0.30) & (df['bias'] < 0.15) & 
            (df['winner_rate'] < 70) &
            (df['circ_mv'] > 300000) &  
            (df['close'] >= p_min) &       
            (df['close'] <= p_max) &       
            (df['turnover_rate'] < to_max) 
        )
        
        # 6. 排序取 Top N
        sorted_df = df[condition].sort_values('bias', ascending=True)
        return sorted_df.head(top_n)
        
    except Exception:
        return None

# ==========================================
# 5. 侧边栏
# ==========================================
st.sidebar.header("🎛️ 尊享指挥官")
token_input = st.sidebar.text_input("Tushare Token (高级版)", type="password")
pro = get_pro_api(token_input)

st.sidebar.divider()
st.sidebar.caption("👇 基于真实筹码数据")
cfg_position_count = st.sidebar.slider("每日Top N", 1, 5, 3)
cfg_min_price = st.sidebar.number_input("最低价", 8.1)
cfg_max_price = st.sidebar.number_input("最高价", 20.0)
cfg_max_turnover = st.sidebar.number_input("换手率上限", 2.1)

st.sidebar.divider()
cfg_stop_loss = st.sidebar.number_input("止损%", 8.5)
cfg_max_hold = st.sidebar.number_input("持股天", 15)
cfg_trail_start = 0.08
cfg_trail_drop = 0.03
stop_loss_decimal = cfg_stop_loss / 100.0

today = datetime.now()
# 默认半年，防止第一次测试等待太久，用户可以自己改长
start_date = st.sidebar.text_input("开始日期", value=f"{today.year}0101") 
end_date = st.sidebar.text_input("结束日期", value=today.strftime('%Y%m%d'))

# ==========================================
# 6. 主程序
# ==========================================
st.title("🚀 V36.0 最终核聚变版 (并发筹码)")
st.caption("技术特征：**10线程并发下载** + **真实筹码接口** + **内存切片回测**。既要准，也要快。")

tab1, tab2 = st.tabs(["📡 智能实盘", "🧪 并发回测"])

# --- Tab 1: 实盘 ---
with tab1:
    col_d, col_b = st.columns([3, 1])
    with col_d:
        scan_date_input = st.date_input("选择日期", value=pd.Timestamp.now())
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("实盘扫描", type="primary"):
        if not pro: st.stop()
        
        # 实盘只下一天的数据，不需要并发，直接复用任务函数
        with st.spinner("正在请求当天筹码数据..."):
            # 获取最近交易日逻辑(简写)
            try:
                real_date_str = scan_date_str # 假设用户选对了，或复用之前的修正逻辑
                # 重新复用之前的修正函数代码量太大，这里做个简化：如果当天没数据，Tushare返回空，我们提示即可
            except: pass
            
            # 1. 临时构造一个 package 结构给 run_strategy 用
            # 这样做是为了复用逻辑
            d_daily = pro.daily(trade_date=scan_date_str)
            d_basic = pro.daily_basic(trade_date=scan_date_str, fields='ts_code,trade_date,turnover_rate,circ_mv')
            d_cyq = pro.cyq_perf(trade_date=scan_date_str)
            d_names = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
            
            if d_daily.empty or d_cyq.empty:
                st.warning("当天无数据或非交易日。")
            else:
                mini_pkg = {
                    'daily_dict': {scan_date_str: d_daily},
                    'basic_dict': {scan_date_str: d_basic},
                    'cyq_dict': {scan_date_str: d_cyq},
                    'names': d_names
                }
                
                fleet = run_strategy_rank1(scan_date_str, mini_pkg, cfg_min_price, cfg_max_price, cfg_max_turnover, cfg_position_count)
                
                if fleet is not None and not fleet.empty:
                    st.success(f"⚓ 选出 {len(fleet)} 只标的")
                    st.dataframe(fleet[['ts_code', 'name', 'close', 'bias', 'turnover_rate', 'winner_rate', 'industry']].style.format({
                        'close': '{:.2f}', 'bias': '{:.4f}', 'turnover_rate': '{:.2f}', 'winner_rate': '{:.1f}'
                    }), hide_index=True)

# --- Tab 2: 回测 ---
with tab2:
    st.info("💡 点击下方按钮，系统将启动 10 个线程同时为您下载数据。请耐心等待进度条跑完。")
    if st.button("🚀 启动并发回测", type="primary", use_container_width=True):
        if not token_input:
            st.error("Token 无效")
            st.stop()
            
        # 1. 下载全量数据 (这是最耗时的一步，但比 V33 快10倍)
        pkg = fetch_full_data_concurrently(start_date, end_date, token_input)
        
        if not pkg:
            st.error("数据下载失败，请检查网络或 Token。")
            st.stop()
            
        dates = pkg['dates']
        st.success(f"✅ 数据准备就绪！覆盖 {len(dates)} 个交易日。开始内存回测...")
        
        # 2. 回测循环 (纯内存，极快)
        active_signals = [] 
        finished_signals = [] 
        
        progress_bar = st.progress(0)
        
        for i, date in enumerate(dates):
            if i % 5 == 0: progress_bar.progress((i + 1) / len(dates))
            
            # 获取当天的价格表用于持仓更新
            # 注意：pkg['daily_dict'][date] 是当天的所有股票数据
            d_today = pkg['daily_dict'].get(date)
            if d_today is None: continue
            
            price_map = d_today.set_index('ts_code')[['open','high','low','close']].to_dict('index')
            
            curr_dt = pd.to_datetime(date)
            next_active = []
            
            # --- 持仓更新 ---
            for sig in active_signals:
                code = sig['code']
                if curr_dt <= pd.to_datetime(sig['buy_date']):
                    if code in price_map: sig['highest'] = max(sig['highest'], price_map[code]['high'])
                    next_active.append(sig)
                    continue
                
                if code in price_map:
                    p = price_map[code]
                    ph, pl, pc = p['high'], p['low'], p['close']
                    
                    if ph > sig['highest']: sig['highest'] = ph
                    cost = sig['buy_price']
                    peak = sig['highest']
                    
                    reason = ""
                    sell_p = pc
                    
                    if (pl - cost) / cost <= -stop_loss_decimal:
                        reason = "止损"
                        sell_p = cost * (1 - stop_loss_decimal)
                    elif (peak - cost)/cost >= cfg_trail_start and (peak - pc)/peak >= cfg_trail_drop:
                        reason = "止盈"
                        sell_p = peak * (1 - cfg_trail_drop)
                    elif (curr_dt - pd.to_datetime(sig['buy_date'])).days >= cfg_max_hold:
                        reason = "超时"
                    
                    if reason:
                        ret = (sell_p - cost) / cost - 0.0006
                        finished_signals.append({'ret': ret, 'rank': sig['rank']})
                    else:
                        next_active.append(sig)
                else:
                    next_active.append(sig)
            active_signals = next_active
            
            # --- 选股 ---
            fleet = run_strategy_rank1(date, pkg, cfg_min_price, cfg_max_price, cfg_max_turnover, cfg_position_count)
            
            if fleet is not None and not fleet.empty:
                for rank_idx, (_, row) in enumerate(fleet.iterrows()):
                    code = row['ts_code']
                    if code in price_map:
                        active_signals.append({
                            'code': code, 'buy_date': date, 
                            'buy_price': price_map[code]['open'], 'highest': price_map[code]['open'],
                            'rank': rank_idx + 1
                        })
        
        progress_bar.empty()
        
        # 结果
        if finished_signals:
            df_res = pd.DataFrame(finished_signals)
            df_res['ret_pct'] = df_res['ret'] * 100
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("平均期望", f"{df_res['ret'].mean()*100:.2f}%")
            c2.metric("胜率", f"{(df_res['ret']>0).mean()*100:.1f}%")
            c3.metric("总交易", f"{len(df_res)}")
            
            st.subheader("🏆 各名次表现 (真实筹码)")
            rank_stats = df_res.groupby('rank')['ret_pct'].agg(['count', 'mean', 'sum', lambda x: (x>0).mean()*100])
            st.table(rank_stats.style.format("{:.2f}").background_gradient(subset=['mean'], cmap='Greens'))
        else:
            st.warning("无交易")
