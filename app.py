import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
import time
import gc
import os
from datetime import datetime, timedelta
# 引入多线程
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="V37.0 修正合体版", layout="wide")

# ==========================================
# 2. 系统控制台
# ==========================================
st.sidebar.header("🛠️ 系统控制台")
st.sidebar.success("✅ V37.0 (多线程真筹码修复版)")

if st.sidebar.button("🔥 强制重启 (更新代码后必点)", type="primary"):
    st.cache_data.clear()
    st.cache_resource.clear()
    os._exit(0)

# ==========================================
# 3. 数据引擎
# ==========================================

@st.cache_resource
def get_pro_api(token):
    if not token: return None
    ts.set_token(token)
    return ts.pro_api(timeout=60) 

# --- 核心修复：智能日期回溯 (从V33移植回来) ---
def get_latest_trade_date(_pro, curr_date_str):
    """
    解决 V35 傻傻地查周六导致报错的问题。
    """
    try:
        end_dt = pd.to_datetime(curr_date_str)
        start_dt = end_dt - timedelta(days=60)
        df = _pro.trade_cal(exchange='', start_date=start_dt.strftime('%Y%m%d'), 
                            end_date=curr_date_str, is_open='1')
        if df.empty: return curr_date_str
        # 强制倒序，取最近一天
        df = df.sort_values('cal_date', ascending=False)
        return df['cal_date'].iloc[0]
    except:
        return curr_date_str

# --- 单日下载任务 (保留真筹码接口) ---
def fetch_day_task(date, token):
    try:
        ts.set_token(token)
        local_pro = ts.pro_api()
        
        # 1. 基础数据
        d1 = local_pro.daily(trade_date=date)
        if d1.empty: return None # 如果当天真的没数据，直接返回
        
        d2 = local_pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
        
        # 2. 真筹码数据 (cyq_perf)
        # 既然您接受回测较慢，我们就必须用这个最准的数据
        d4 = local_pro.cyq_perf(trade_date=date)
        
        return {'date': date, 'daily': d1, 'basic': d2, 'cyq': d4}
    except:
        return None

# --- 多线程批量下载 (带进度条) ---
@st.cache_data(ttl=3600)
def fetch_data_parallel(dates, token):
    results = {}
    progress_bar = st.progress(0, text="正在启动多线程引擎...")
    
    # 既然您说不一定是多线程的问题，我们保守一点开 5 个线程
    # 既能加速，又比 10 个线程稳
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(fetch_day_task, d, token): d for d in dates}
        
        total = len(dates)
        done = 0
        
        for future in as_completed(future_map):
            done += 1
            data = future.result()
            if data:
                results[data['date']] = data
            
            # 更新进度
            pct = done / total
            progress_bar.progress(pct, text=f"📥 多线程下载中: {done}/{total} 天")
            
    progress_bar.empty()
    return results

# 辅助：获取名称
@st.cache_data(ttl=86400)
def get_names(token):
    try:
        ts.set_token(token)
        return ts.pro_api().stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    except: return pd.DataFrame()

# ==========================================
# 4. 逻辑层 (Rank 1 真筹码版)
# ==========================================
def run_strategy_rank1(snapshot, names_df, p_min, p_max, to_max, top_n):
    if not snapshot: return None
    d1 = snapshot.get('daily')
    d2 = snapshot.get('basic')
    d4 = snapshot.get('cyq')
    
    # 严格检查，因为我们要用真数据
    if d1 is None or d1.empty: return None
    if d4 is None or d4.empty: return None
    if 'cost_50pct' not in d4.columns: return None
    
    try:
        # 合并
        m1 = pd.merge(d1, d2, on='ts_code', how='inner')
        if names_df is not None:
            m1 = pd.merge(m1, names_df, on='ts_code', how='left')
            
        # 关联筹码
        df = pd.merge(m1, d4[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
        
        # 计算 Bias (真)
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        
        condition = (
            (df['bias'] > -0.30) & (df['bias'] < 0.15) & 
            (df['winner_rate'] < 70) &
            (df['circ_mv'] > 300000) &  
            (df['close'] >= p_min) &       
            (df['close'] <= p_max) &       
            (df['turnover_rate'] < to_max) 
        )
        
        sorted_df = df[condition].sort_values('bias', ascending=True)
        return sorted_df.head(top_n)
    except:
        return None

# ==========================================
# 5. 侧边栏
# ==========================================
st.sidebar.header("🎛️ 尊享控制台")
token_input = st.sidebar.text_input("Tushare Token", type="password")
pro = get_pro_api(token_input)

st.sidebar.divider()
cfg_position_count = st.sidebar.slider("Top N", 1, 5, 3)
cfg_min_price = st.sidebar.number_input("最低价", 8.1)
cfg_max_price = st.sidebar.number_input("最高价", 20.0)
cfg_max_turnover = st.sidebar.slider("换手率上限", 0.5, 5.0, 2.1)

st.sidebar.divider()
cfg_stop_loss = st.sidebar.number_input("止损%", 8.5)
cfg_max_hold = st.sidebar.number_input("持股天", 15)
cfg_trail_start = 0.08
cfg_trail_drop = 0.03
stop_loss_decimal = cfg_stop_loss / 100.0

today = datetime.now()
start_date = st.sidebar.text_input("开始日期", value=f"{today.year}0101")
end_date = st.sidebar.text_input("结束日期", value=today.strftime('%Y%m%d'))

# ==========================================
# 6. 主程序
# ==========================================
st.title("🚀 V37.0 修正合体版 (真筹码+智能日期)")

tab1, tab2 = st.tabs(["📡 智能实盘", "🧪 并发回测"])

# --- Tab 1: 实盘 (修复了“不是交易日”的报错) ---
with tab1:
    col_d, col_b = st.columns([3, 1])
    with col_d:
        # 用户依然可以选周六
        scan_date_input = st.date_input("选择日期", value=pd.Timestamp.now())
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("开始扫描", type="primary"):
        if not pro: st.stop()
        
        with st.spinner("正在校对日期并获取筹码数据..."):
            # 1. 智能修正日期 (V37 关键修复)
            real_date_str = get_latest_trade_date(pro, scan_date_str)
            
            if real_date_str != scan_date_str:
                st.info(f"📅 修正：您选择的 **{scan_date_str}** 非交易日，已自动切换至：**{real_date_str}**")
            
            # 2. 获取数据 (复用 fetch_day_task)
            # 这里我们不用多线程，直接单次调用，因为只查一天
            data = fetch_day_task(real_date_str, token_input)
            names_df = get_names(token_input)
            
            if data:
                fleet = run_strategy_rank1(data, names_df, cfg_min_price, cfg_max_price, cfg_max_turnover, cfg_position_count)
                if fleet is not None and not fleet.empty:
                    st.success(f"⚓ 成功选出 {len(fleet)} 只标的 (基于 {real_date_str})")
                    st.dataframe(fleet[['ts_code', 'name', 'close', 'bias', 'turnover_rate', 'winner_rate', 'industry']].style.format({
                        'close': '{:.2f}', 'bias': '{:.4f}', 'turnover_rate': '{:.2f}', 'winner_rate': '{:.1f}'
                    }), hide_index=True)
                else:
                    st.warning(f"在 {real_date_str} 未找到符合条件的标的。")
            else:
                st.error(f"无法获取 {real_date_str} 的数据，请检查网络或Token。")

# --- Tab 2: 回测 (保留多线程，接受 1 小时耗时) ---
with tab2:
    st.info("💡 系统将启动 5 线程并发下载真实筹码数据。预计耗时会比单线程快，但仍需耐心等待。")
    
    if st.button("🚀 启动并发回测", type="primary", use_container_width=True):
        if not token_input:
            st.error("Token 无效")
            st.stop()
            
        # 1. 获取日期
        try:
            ts.set_token(token_input)
            pro = ts.pro_api()
            cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
            dates = sorted(cal_df['cal_date'].tolist())
        except:
            st.error("网络初始化失败")
            st.stop()
            
        # 2. 多线程下载
        memory_db = fetch_data_parallel(dates, token_input)
        names_df = get_names(token_input)
        
        if not memory_db:
            st.error("数据下载失败")
            st.stop()
            
        st.success(f"✅ 数据下载完成！内存已加载 {len(memory_db)} 天真筹码数据。开始回测...")
        
        # 3. 内存回测
        active_signals = [] 
        finished_signals = [] 
        progress_bar = st.progress(0)
        
        valid_dates = sorted(list(memory_db.keys()))
        
        for i, date in enumerate(valid_dates):
            if i % 5 == 0: progress_bar.progress((i + 1) / len(valid_dates))
            
            snap = memory_db.get(date)
            # 价格表
            price_map = {}
            if snap and not snap['daily'].empty:
                price_map = snap['daily'].set_index('ts_code')[['open','high','low','close']].to_dict('index')
            
            curr_dt = pd.to_datetime(date)
            next_active = []
            
            # 持仓
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
                        finished_signals.append({'ret': ret, 'rank': sig.get('rank', 1)})
                    else:
                        next_active.append(sig)
                else:
                    next_active.append(sig)
            active_signals = next_active
            
            # 选股
            fleet = run_strategy_rank1(snap, names_df, cfg_min_price, cfg_max_price, cfg_max_turnover, cfg_position_count)
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
        
        if finished_signals:
            df_res = pd.DataFrame(finished_signals)
            df_res['ret_pct'] = df_res['ret'] * 100
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("单笔期望", f"{df_res['ret'].mean()*100:.2f}%")
            c2.metric("胜率", f"{(df_res['ret']>0).mean()*100:.1f}%")
            c3.metric("交易次数", f"{len(df_res)}")
            
            st.subheader("🏆 分名次表现")
            rank_stats = df_res.groupby('rank')['ret_pct'].agg(['count', 'mean', 'sum', lambda x: (x>0).mean()*100])
            st.table(rank_stats.style.format("{:.2f}"))
        else:
            st.warning("无交易")
