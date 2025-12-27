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
st.set_page_config(page_title="V40.4 避雷针过滤版", layout="wide")

# ==========================================
# 2. 系统控制台
# ==========================================
st.sidebar.header("🔥 趋势狩猎 (V40.4)")
st.sidebar.success("✅ 自动过滤长上影线")
st.sidebar.info("逻辑：获利盘>80% + 涨幅2%~7% + **最高点回撤<1.5%**")

if st.sidebar.button("🔄 强制重启系统", type="primary"):
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

def fetch_day_task_right_side(date, token):
    max_retries = 5
    for i in range(max_retries):
        try:
            time.sleep(0.1)
            ts.set_token(token)
            local_pro = ts.pro_api(timeout=30)
            
            # 我们需要 high 字段来计算上影线
            d_today = local_pro.daily(trade_date=date)
            d_basic = local_pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            
            d_cyq = local_pro.cyq_perf(trade_date=date)
            if d_cyq.empty:
                prev_date = (pd.to_datetime(date) - timedelta(days=1)).strftime('%Y%m%d')
                d_cyq = local_pro.cyq_perf(trade_date=prev_date)

            if not d_today.empty and not d_cyq.empty:
                return {'date': date, 'daily': d_today, 'basic': d_basic, 'cyq': d_cyq}
            if d_today.empty: return None
            raise ValueError("Incomplete")
        except:
            if i == max_retries - 1: return None
            time.sleep(1 + i)
    return None

@st.cache_data(ttl=3600)
def fetch_data_parallel_right(dates, token):
    results = {}
    progress_bar = st.progress(0, text="全市场扫描中...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(fetch_day_task_right_side, d, token): d for d in dates}
        total = len(dates)
        done = 0
        for future in as_completed(future_map):
            done += 1
            data = future.result()
            if data: results[data['date']] = data
            progress_bar.progress(done / total, text=f"📥 进度: {done}/{total}")
    progress_bar.empty()
    return results

@st.cache_data(ttl=86400)
def get_names(token):
    try:
        ts.set_token(token)
        return ts.pro_api().stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    except: return pd.DataFrame()

# ==========================================
# 4. 逻辑层 (增加上影线过滤)
# ==========================================
def run_strategy_golden_zone_strict(snapshot, names_df, min_winner, min_chg, max_chg, max_shadow, top_n):
    if not snapshot: return None
    d_today = snapshot.get('daily') 
    d_basic = snapshot.get('basic')
    d_cyq = snapshot.get('cyq')   
    
    if d_today is None or d_today.empty or d_cyq is None or d_cyq.empty: return None
    
    try:
        m1 = pd.merge(d_today, d_basic, on='ts_code', how='inner')
        if names_df is not None:
            m1 = pd.merge(m1, names_df, on='ts_code', how='left')
        
        df = pd.merge(m1, d_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
        
        # --- 核心计算 ---
        # 1. 计算回撤 (High - Close) / Close * 100
        # 如果这个值很大，说明上影线很长
        df['shadow_pct'] = (df['high'] - df['close']) / df['close'] * 100
        
        condition = (
            (df['winner_rate'] >= min_winner) &     
            (df['pct_chg'] >= min_chg) &            
            (df['pct_chg'] <= max_chg) &    
            (df['shadow_pct'] <= max_shadow) &      # <--- 关键过滤：回撤不能超过 1.5%
            (df['circ_mv'] > 300000) &              
            (~df['name'].str.contains('ST'))        
        )
        
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
cfg_min_winner = st.sidebar.number_input("最低获利盘(%)", value=80.0, step=1.0)

col_c1, col_c2 = st.sidebar.columns(2)
with col_c1:
    cfg_min_chg = st.sidebar.number_input("最小涨幅(%)", value=2.0, step=0.5)
with col_c2:
    cfg_max_chg = st.sidebar.number_input("最大涨幅(%)", value=7.0, step=0.5)

# --- 新增：上影线风控 ---
st.sidebar.caption("👇 避雷针风控")
cfg_max_shadow = st.sidebar.number_input("允许最大回落(%)", value=1.5, step=0.1, help="例如设为1.5%，如果最高涨7%，收盘必须在5.5%以上，否则不买")

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
st.title("🚀 V40.4 避雷针过滤版")
st.info("💡 新增风控：如果股价从当日最高点回落超过 **1.5%**，视为上涨乏力（上影线太长），系统将自动剔除，防止骗炮。")

tab1, tab2 = st.tabs(["🏹 实盘扫描", "📈 趋势回测"])

with tab1:
    col_d, col_b = st.columns([3, 1])
    with col_d:
        yesterday = datetime.now() - timedelta(days=1)
        scan_date_input = st.date_input("选择日期 (选昨天)", value=yesterday)
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("扫描纯净机会", type="primary"):
        if not pro: st.stop()
        with st.spinner(f"正在分析 {scan_date_str} (自动剔除避雷针)..."):
            
            data = fetch_day_task_right_side(scan_date_str, token_input)
            names_df = get_names(token_input)
            
            if data:
                fleet = run_strategy_golden_zone_strict(data, names_df, cfg_min_winner, cfg_min_chg, cfg_max_chg, cfg_max_shadow, 20)
                
                if fleet is not None and not fleet.empty:
                    st.success(f"🔥 发现 {len(fleet)} 只形态完美的股票")
                    st.dataframe(fleet[['ts_code', 'name', 'close', 'pct_chg', 'shadow_pct', 'winner_rate']].style.format({
                        'close': '{:.2f}', 'pct_chg': '{:.2f}%', 'shadow_pct': '{:.2f}%', 'winner_rate': '{:.1f}%'
                    }), hide_index=True)
                else:
                    st.warning(f"昨日无符合条件的股票 (可能都被上影线过滤掉了)。")
            else:
                st.error("数据获取失败。")

with tab2:
    if st.button("🚀 启动纯净回测", type="primary", use_container_width=True):
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
            
            # --- 持仓 ---
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
            fleet = run_strategy_golden_zone_strict(snap, names_df, cfg_min_winner, cfg_min_chg, cfg_max_chg, cfg_max_shadow, cfg_position_count)
            if fleet is not None and not fleet.empty:
                for _, row in fleet.iterrows():
                    code = row['ts_code']
                    if code in price_map:
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
