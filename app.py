import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
import time
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="V40.3 黄金击球实战版", layout="wide")

# ==========================================
# 2. 系统控制台
# ==========================================
st.sidebar.header("🔥 趋势狩猎 (V40.3)")
st.sidebar.success("✅ 多线程引擎已启动")
st.sidebar.success("✅ 真筹码数据已加载")
st.sidebar.info("核心：**获利盘>80%** + **涨幅2%~7%**")

if st.sidebar.button("🔄 强制重启系统", type="primary"):
    st.cache_data.clear()
    st.cache_resource.clear()
    os._exit(0)

# ==========================================
# 3. 数据引擎 (多线程 + 真筹码)
# ==========================================
@st.cache_resource
def get_pro_api(token):
    if not token: return None
    ts.set_token(token)
    return ts.pro_api(timeout=60) 

def fetch_day_task_right_side(date, token):
    """
    单日数据下载任务：同时获取行情 + 筹码
    """
    max_retries = 5
    for i in range(max_retries):
        try:
            time.sleep(0.1) # 防封
            ts.set_token(token)
            local_pro = ts.pro_api(timeout=30)
            
            # 1. 基础行情 (涨跌幅, 收盘价)
            d_today = local_pro.daily(trade_date=date)
            
            # 2. 每日指标 (换手率, 流通市值)
            d_basic = local_pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            
            # 3. 真筹码数据 (您的核心优势)
            # 尝试获取当日筹码
            d_cyq = local_pro.cyq_perf(trade_date=date)
            
            if d_cyq.empty:
                # 如果当日没出（比如盘中），尝试取前一日的作为参考
                prev_date = (pd.to_datetime(date) - timedelta(days=1)).strftime('%Y%m%d')
                d_cyq = local_pro.cyq_perf(trade_date=prev_date)

            if not d_today.empty and not d_cyq.empty:
                return {'date': date, 'daily': d_today, 'basic': d_basic, 'cyq': d_cyq}
            
            # 如果依然空，可能是周末或休市，跳过
            if d_today.empty: return None
            raise ValueError("Data incomplete") # 抛错重试
            
        except:
            if i == max_retries - 1: return None
            time.sleep(1 + i)
    return None

@st.cache_data(ttl=3600)
def fetch_data_parallel_right(dates, token):
    """
    5线程并发下载引擎
    """
    results = {}
    progress_bar = st.progress(0, text="🔥 多线程引擎启动：正在扫描全市场筹码...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(fetch_day_task_right_side, d, token): d for d in dates}
        total = len(dates)
        done = 0
        success = 0
        
        for future in as_completed(future_map):
            done += 1
            data = future.result()
            if data:
                results[data['date']] = data
                success += 1
            progress_bar.progress(done / total, text=f"📥 猎取进度: {done}/{total} (成功: {success})")
            
    progress_bar.empty()
    return results

@st.cache_data(ttl=86400)
def get_names(token):
    try:
        ts.set_token(token)
        return ts.pro_api().stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    except: return pd.DataFrame()

# ==========================================
# 4. 逻辑层 (黄金击球区 + 筹码背书)
# ==========================================
def run_strategy_golden_zone(snapshot, names_df, min_winner, min_chg, max_chg, top_n):
    if not snapshot: return None
    d_today = snapshot.get('daily') 
    d_basic = snapshot.get('basic')
    d_cyq = snapshot.get('cyq')   
    
    if d_today is None or d_today.empty or d_cyq is None or d_cyq.empty: return None
    
    try:
        # 合并三张表
        m1 = pd.merge(d_today, d_basic, on='ts_code', how='inner')
        if names_df is not None:
            m1 = pd.merge(m1, names_df, on='ts_code', how='left')
        
        # 这里的 d_cyq 就是您的 10000 积分换来的真数据
        df = pd.merge(m1, d_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
        
        # 核心逻辑：
        # 1. winner_rate >= 80% (真筹码背书)
        # 2. pct_chg 在 2%~7% (黄金击球区，拒绝骗炮)
        condition = (
            (df['winner_rate'] >= min_winner) &     
            (df['pct_chg'] >= min_chg) &            
            (df['pct_chg'] <= max_chg) &            
            (df['circ_mv'] > 300000) &              
            (~df['name'].str.contains('ST'))        
        )
        
        # 强者恒强：按获利盘排序
        sorted_df = df[condition].sort_values('winner_rate', ascending=False)
        return sorted_df.head(top_n)
    except:
        return None

# ==========================================
# 5. 侧边栏
# ==========================================
st.sidebar.header("🏹 黄金击球参数")
token_input = st.sidebar.text_input("Tushare Token", type="password")
pro = get_pro_api(token_input)

st.sidebar.divider()
cfg_position_count = st.sidebar.number_input("每日持仓数", value=3, min_value=1, step=1)

# 这就是您的“大价钱”起作用的地方
cfg_min_winner = st.sidebar.number_input("最低获利盘(%)", value=80.0, step=1.0, help="只有主力高度控盘的票才买")

st.sidebar.caption("👇 黄金击球区 (避开长上影)")
col_c1, col_c2 = st.sidebar.columns(2)
with col_c1:
    cfg_min_chg = st.sidebar.number_input("最小涨幅(%)", value=2.0, step=0.5, help="确认上涨")
with col_c2:
    cfg_max_chg = st.sidebar.number_input("最大涨幅(%)", value=7.0, step=0.5, help="拒绝追高")

st.sidebar.divider()
st.sidebar.caption("🛡️ 右侧风控")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    cfg_stop_loss = st.sidebar.number_input("止损线(%)", value=6.0, step=0.1)
with col_s2:
    cfg_max_hold = st.sidebar.number_input("持仓天数", value=5, min_value=1, step=1)

cfg_trail_start = 0.10 
cfg_trail_drop = 0.03  
stop_loss_decimal = cfg_stop_loss / 100.0

today = datetime.now()
start_date = st.sidebar.text_input("开始日期", value=f"{today.year}0101")
end_date = st.sidebar.text_input("结束日期", value=today.strftime('%Y%m%d'))

# ==========================================
# 6. 主程序
# ==========================================
st.title("🚀 V40.3 黄金击球实战版 (真筹码+多线程)")
st.info("💡 策略逻辑：利用 **真筹码数据** 筛选获利盘 > 80% 的股票，并在 **下午 14:30** 确认涨幅在 **2%~7%** 时买入。")

tab1, tab2 = st.tabs(["🏹 实盘扫描", "📈 趋势回测"])

with tab1:
    col_d, col_b = st.columns([3, 1])
    with col_d:
        # 默认选“昨天”，实盘时结合“今天实时涨幅”
        yesterday = datetime.now() - timedelta(days=1)
        scan_date_input = st.date_input("选择日期 (建议选昨天)", value=yesterday)
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("扫描黄金机会", type="primary"):
        if not pro: st.stop()
        with st.spinner(f"正在调取真筹码数据分析 {scan_date_str}..."):
            
            data = fetch_day_task_right_side(scan_date_str, token_input)
            names_df = get_names(token_input)
            
            if data:
                # 扫描
                fleet = run_strategy_golden_zone(data, names_df, cfg_min_winner, cfg_min_chg, cfg_max_chg, 20)
                
                if fleet is not None and not fleet.empty:
                    st.success(f"🔥 发现 {len(fleet)} 只筹码完美的潜力股")
                    st.markdown("👇 **实盘操作指南 (14:30 执行)：**")
                    st.markdown("""
                    请在交易软件中查看以下股票**今天的表现**：
                    1.  **涨幅在 2% ~ 7% 之间？** (确认趋势)
                    2.  **K线是实心阳线？** (拒绝避雷针)
                    3.  **满足则现价买入！**
                    """)
                    st.dataframe(fleet[['ts_code', 'name', 'close', 'pct_chg', 'winner_rate', 'industry']].style.format({
                        'close': '{:.2f}', 'pct_chg': '{:.2f}%', 'winner_rate': '{:.1f}%'
                    }), hide_index=True)
                else:
                    st.warning(f"昨日无符合条件的股票。")
            else:
                st.error("数据获取失败。")

with tab2:
    if st.button("🚀 启动模拟回测 (尾盘买入)", type="primary", use_container_width=True):
        if not token_input: st.stop()
        
        try:
            ts.set_token(token_input)
            pro = ts.pro_api()
            cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
            dates = sorted(cal_df['cal_date'].tolist())
        except: st.stop()
            
        memory_db = fetch_data_parallel_right(dates, token_input)
        names_df = get_names(token_input)
        
        if not memory_db: st.stop()
        
        active_signals = [] 
        finished_signals = [] 
        progress_bar = st.progress(0)
        valid_dates = sorted(list(memory_db.keys()))
        
        for i, date in enumerate(valid_dates):
            if i % 5 == 0: progress_bar.progress((i + 1) / len(valid_dates))
            
            snap = memory_db.get(date)
            price_map = {}
            if snap and not snap['daily'].empty:
                price_map = snap['daily'].set_index('ts_code')[['open','high','low','close']].to_dict('index')
            
            curr_dt = pd.to_datetime(date)
            next_active = []
            
            # --- 持仓管理 ---
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
                    
                    # 1. 破位止损
                    if (pl - cost) / cost <= -stop_loss_decimal:
                        reason = "破位止损"
                        sell_p = cost * (1 - stop_loss_decimal)
                    elif (peak - cost)/cost >= cfg_trail_start and (peak - pc)/peak >= cfg_trail_drop:
                        reason = "高位止盈"
                        sell_p = peak * (1 - cfg_trail_drop)
                    elif (curr_dt - pd.to_datetime(sig['buy_date'])).days >= cfg_max_hold:
                        reason = "动力不足"
                    
                    if reason:
                        ret = (sell_p - cost) / cost - 0.001
                        finished_signals.append({
                            'name': sig.get('name', code),
                            'code': code,
                            'buy_date': sig['buy_date'],
                            'sell_date': date,
                            'ret': ret, 
                            'reason': reason
                        })
                    else:
                        next_active.append(sig)
                else:
                    next_active.append(sig)
            active_signals = next_active
            
            # --- 选股 ---
            fleet = run_strategy_golden_zone(snap, names_df, cfg_min_winner, cfg_min_chg, cfg_max_chg, cfg_position_count)
            if fleet is not None and not fleet.empty:
                for _, row in fleet.iterrows():
                    code = row['ts_code']
                    if code in price_map:
                        # 模拟：14:30 确认在区间内，以【收盘价】买入
                        active_signals.append({
                            'code': code, 
                            'name': row['name'] if 'name' in row else code,
                            'buy_date': date, 
                            'buy_price': price_map[code]['close'], 
                            'highest': price_map[code]['close']
                        })
        
        progress_bar.empty()
        
        if finished_signals:
            df_res = pd.DataFrame(finished_signals)
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("单笔期望", f"{df_res['ret'].mean()*100:.2f}%")
            c2.metric("胜率", f"{(df_res['ret']>0).mean()*100:.1f}%")
            c3.metric("交易次数", f"{len(df_res)}")
            
            st.subheader("📋 交易详情")
            st.dataframe(df_res[['name', 'code', 'buy_date', 'sell_date', 'ret', 'reason']].style.format({'ret': '{:.2%}'}), use_container_width=True)
        else:
            st.warning("无交易")
