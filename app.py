import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
import altair as alt
import time
import gc
from datetime import datetime, timedelta

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="V31.0 智能指挥官", layout="wide")

# ==========================================
# 2. 全局缓存与智能工具
# ==========================================
@st.cache_resource
def get_pro_api(token):
    if not token: return None
    ts.set_token(token)
    return ts.pro_api()

# --- 核心新增：智能日期回溯 ---
def get_latest_trade_date(_pro, curr_date_str):
    """
    输入一个日期，返回最近的一个交易日。
    如果当天是交易日，返回当天；否则往前找。
    """
    try:
        # 获取从10天前到今天的日历
        end_dt = pd.to_datetime(curr_date_str)
        start_dt = end_dt - timedelta(days=15)
        
        df = _pro.trade_cal(exchange='', start_date=start_dt.strftime('%Y%m%d'), 
                            end_date=curr_date_str, is_open='1')
        
        if not df.empty:
            return df['cal_date'].iloc[-1] # 返回最后一个交易日
        return curr_date_str # 兜底
    except:
        return curr_date_str

@st.cache_data(ttl=86400 * 7)
def fetch_daily_atomic_data(date, _pro):
    if _pro is None: return {}
    try:
        df_daily = _pro.daily(trade_date=date)
        df_basic = _pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
        df_names = _pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        
        df_cyq = _pro.cyq_perf(trade_date=date)
        if df_cyq.empty: 
             for i in range(1, 4):
                 prev = (pd.to_datetime(date) - pd.Timedelta(days=i)).strftime('%Y%m%d')
                 df_cyq = _pro.cyq_perf(trade_date=prev)
                 if not df_cyq.empty: break
        
        return {'daily': df_daily, 'basic': df_basic, 'cyq': df_cyq, 'names': df_names}
    except Exception: return {}

@st.cache_data(ttl=86400)
def get_market_sentiment(start, end, _pro):
    if _pro is None: return {}
    try:
        real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        df = _pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
        df = df.sort_values('trade_date', ascending=True)
        df['ma20'] = df['close'].rolling(20).mean()
        return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
    except: return {}

# ==========================================
# 3. 逻辑层
# ==========================================
def run_strategy_logic(snapshot, p_min, p_max, to_max, top_n=1):
    if not snapshot: return None
    d1, d2, d3, d4 = snapshot.get('daily'), snapshot.get('basic'), snapshot.get('names'), snapshot.get('cyq')
    
    if d1 is None or d1.empty or d2 is None or d2.empty or d4 is None or d4.empty: return None
    if 'cost_50pct' not in d4.columns: return None
    
    m1 = pd.merge(d1, d2, on='ts_code')
    m2 = pd.merge(m1, d3, on='ts_code')
    df = pd.merge(m2, d4[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code')
    
    df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
    
    condition = (
        (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
        (df['winner_rate'] < 70) &
        (df['circ_mv'] > 300000) &  
        (df['close'] >= p_min) &       
        (df['close'] <= p_max) &       
        (df['turnover_rate'] < to_max) 
    )
    
    sorted_df = df[condition].sort_values('bias', ascending=True)
    if sorted_df.empty: return None
    return sorted_df.head(top_n)

# ==========================================
# 4. 侧边栏
# ==========================================
st.sidebar.header("🎛️ 智能指挥台")
token_input = st.sidebar.text_input("Token", type="password")
pro = get_pro_api(token_input)

st.sidebar.divider()
st.sidebar.subheader("⚓ 仓位管理")
cfg_position_count = st.sidebar.slider("每日买入数量 (Top N)", 1, 5, 1)

st.sidebar.divider()
st.sidebar.subheader("🎯 上帝参数 (默认最优)")
cfg_min_price = st.sidebar.number_input("最低价 (元)", value=8.1, step=0.1)
cfg_max_price = st.sidebar.number_input("最高价 (元)", value=20.0, step=0.5)
cfg_max_turnover = st.sidebar.slider("最大换手率 (%)", 0.5, 5.0, 2.1, step=0.1)

st.sidebar.divider()
st.sidebar.subheader("🛡️ 风控参数")
cfg_stop_loss = st.sidebar.slider("止损线 (-%)", 3.0, 15.0, 8.5, step=0.5)
cfg_max_hold = st.sidebar.slider("最长持股 (天)", 5, 30, 15)
cfg_trail_start = st.sidebar.slider("止盈启动 (+%)", 5.0, 15.0, 8.0, step=0.5) / 100.0
cfg_trail_drop = st.sidebar.slider("回落卖出 (-%)", 1.0, 5.0, 3.0, step=0.5) / 100.0
stop_loss_decimal = cfg_stop_loss / 100.0

start_date = st.sidebar.text_input("开始日期", value="20250101")
end_date = st.sidebar.text_input("结束日期", value="20251226")

# ==========================================
# 5. 主程序
# ==========================================
st.title("🚀 V31.0 智能指挥官 (自动识别交易日)")

tab1, tab2 = st.tabs(["📡 智能实盘扫描", "🧪 历史分仓回测"])

# --- Tab 1: 智能实盘 ---
with tab1:
    col_d, col_b = st.columns([3, 1])
    with col_d:
        # 用户依然可以随便选日期，哪怕选了周六
        scan_date_input = st.date_input("选择日期 (系统会自动修正)", value=pd.Timestamp.now())
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("开始扫描", type="primary", use_container_width=True):
        if not pro:
            st.error("Token 无效")
            st.stop()
            
        with st.spinner("正在校对交易日历..."):
            # 1. 智能修正日期
            real_date_str = get_latest_trade_date(pro, scan_date_str)
            
            # 如果日期变了，提示用户
            if real_date_str != scan_date_str:
                st.info(f"📅 检测到 **{scan_date_str}** 是非交易日，已自动为您切换到最近的交易日：**{real_date_str}**")
            
            # 2. 获取数据
            snap = fetch_daily_atomic_data(real_date_str, pro)
            # 3. 运行策略
            fleet = run_strategy_logic(snap, cfg_min_price, cfg_max_price, cfg_max_turnover, cfg_position_count)
            
            if fleet is not None and not fleet.empty:
                st.success(f"⚓ 锁定 {len(fleet)} 只标的 (基于 {real_date_str} 数据)")
                
                st.dataframe(fleet[['ts_code', 'name', 'close', 'bias', 'turnover_rate', 'winner_rate', 'industry']].style.format({
                    'close': '{:.2f}', 'bias': '{:.4f}', 'turnover_rate': '{:.2f}', 'winner_rate': '{:.1f}'
                }))
                
                st.info(f"""
                **📝 交易计划：**
                1.  **标的**：{', '.join(fleet['name'].tolist())}
                2.  **买入时机**：下个交易日开盘。
                3.  **风控**：止损 -{cfg_stop_loss}%，持股 {cfg_max_hold} 天。
                """)
            else:
                st.warning(f"在 {real_date_str} 未找到符合条件的标的。")

# --- Tab 2: 回测 ---
with tab2:
    if st.button("🚀 运行回测", type="primary", use_container_width=True):
        if not pro:
            st.error("Token 无效")
            st.stop()
            
        cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
        dates = sorted(cal_df['cal_date'].tolist())
        market_safe_map = get_market_sentiment(start_date, end_date, pro)
        
        active_signals = [] 
        finished_signals = [] 
        
        progress_bar = st.progress(0)
        
        for i, date in enumerate(dates):
            progress_bar.progress((i + 1) / len(dates))
            if i % 20 == 0: gc.collect()
            
            snap = fetch_daily_atomic_data(date, pro)
            price_map = {}
            if snap and not snap['daily'].empty:
                price_map = snap['daily'].set_index('ts_code')[['open','high','low','close']].to_dict('index')
            
            is_market_safe = market_safe_map.get(date, False)
            
            # 持仓
            signals_still_active = []
            curr_dt = pd.to_datetime(date)
            
            for sig in active_signals:
                code = sig['code']
                if curr_dt <= pd.to_datetime(sig['buy_date']):
                    if code in price_map: sig['highest'] = max(sig['highest'], price_map[code]['high'])
                    signals_still_active.append(sig)
                    continue

                if code in price_map:
                    ph, pl, pc = price_map[code]['high'], price_map[code]['low'], price_map[code]['close']
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
                        finished_signals.append({
                            'code': code, 'buy_date': sig['buy_date'], 
                            'ret': ret, 'reason': reason, 'rank': sig['rank']
                        })
                    else:
                        signals_still_active.append(sig)
                else:
                    signals_still_active.append(sig)
            active_signals = signals_still_active
            
            # 买入
            if is_market_safe:
                fleet = run_strategy_logic(snap, cfg_min_price, cfg_max_price, cfg_max_turnover, cfg_position_count)
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
            st.markdown(f"### 📊 报告 (Top {cfg_position_count})")
            
            avg_ret = df_res['ret'].mean() * 100
            win_rate = (df_res['ret'] > 0).mean() * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("单笔平均期望", f"{avg_ret:.2f}%")
            c2.metric("胜率", f"{win_rate:.1f}%")
            c3.metric("总交易次数", f"{len(df_res)}")
            
            st.subheader("🏆 各名次表现")
            rank_stats = df_res.groupby('rank')['ret_pct'].agg(['count', 'mean', 'sum', lambda x: (x>0).mean()*100])
            rank_stats.columns = ['交易数', '单笔期望%', '总收益%', '胜率%']
            st.table(rank_stats.style.format("{:.2f}").background_gradient(subset=['单笔期望%'], cmap='Greens'))
        else:
            st.warning("无交易数据")
