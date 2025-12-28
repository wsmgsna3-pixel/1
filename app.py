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
st.set_page_config(page_title="V46.0 竞价止损版", layout="wide")

# ==========================================
# 2. 系统控制台
# ==========================================
st.sidebar.header("🛡️ 趋势狩猎 (V46.0)")
st.sidebar.success("✅ **新增：T+1 竞价风控**")
st.sidebar.info("若次日大幅低开，开盘直接核按钮(止损)")

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

@st.cache_data(ttl=3600)
def fetch_index_data(token, start_date, end_date):
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        real_start = (pd.to_datetime(start_date) - timedelta(days=60)).strftime('%Y%m%d')
        df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end_date)
        if df.empty: return pd.DataFrame()
        df = df.sort_values('trade_date')
        df['ma20'] = df['close'].rolling(20).mean()
        df = df[df['trade_date'] >= start_date]
        return df.set_index('trade_date')
    except: return pd.DataFrame()

def fetch_day_task_right_side(date, token):
    max_retries = 5 
    for i in range(max_retries):
        try:
            time.sleep(0.1 + np.random.random() * 0.2)
            ts.set_token(token)
            local_pro = ts.pro_api(timeout=45)
            
            d_today = local_pro.daily(trade_date=date)
            if d_today.empty: return None 
            
            d_basic = local_pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            
            d_cyq = local_pro.cyq_perf(trade_date=date)
            if d_cyq.empty:
                prev_date = (pd.to_datetime(date) - timedelta(days=1)).strftime('%Y%m%d')
                d_cyq = local_pro.cyq_perf(trade_date=prev_date)

            if not d_today.empty and not d_cyq.empty:
                return {'date': date, 'daily': d_today, 'basic': d_basic, 'cyq': d_cyq}
            
            raise ValueError("Data incomplete")
        except:
            if i == max_retries - 1: return None 
            time.sleep(1 + i) 
    return None

@st.cache_data(ttl=3600)
def fetch_data_parallel_right(dates, token):
    results = {}
    progress_bar = st.progress(0, text="启动下载引擎...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(fetch_day_task_right_side, d, token): d for d in dates}
        total = len(dates)
        done = 0
        for future in as_completed(future_map):
            done += 1
            data = future.result()
            if data:
                results[data['date']] = data
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
# 4. 逻辑层
# ==========================================
def run_strategy_v46(snapshot, names_df, min_winner, min_chg, max_chg, max_shadow, min_price, top_n, index_df, curr_date, enable_market_filter, show_debug=False):
    # 1. 大盘风控
    market_status = "OK"
    if enable_market_filter and index_df is not None and not index_df.empty:
        if curr_date in index_df.index:
            idx_today = index_df.loc[curr_date]
            if idx_today['close'] < idx_today['ma20']:
                market_status = "BAD"
                if not show_debug: return "MARKET_BAD", None

    # 2. 个股筛选
    if not snapshot: return "NO_DATA", None
    d_today = snapshot.get('daily') 
    d_basic = snapshot.get('basic')
    d_cyq = snapshot.get('cyq')   
    if d_today is None or d_today.empty: return "NO_DATA", None
    
    try:
        m1 = pd.merge(d_today, d_basic, on='ts_code', how='inner')
        if names_df is not None:
            m1 = pd.merge(m1, names_df, on='ts_code', how='left')
        
        df = pd.merge(m1, d_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
        df['shadow_pct'] = (df['high'] - df['close']) / df['close'] * 100
        
        # Step 0: 剔除创业板 (保留之前优化的成果)
        df = df[~df['ts_code'].str.startswith('30')]

        total = len(df)
        
        # Step 1: 价格 > 10
        df = df[df['close'] >= min_price]
        c_price = len(df)
        
        # Step 2: 绝对涨幅 
        df = df[(df['pct_chg'] >= min_chg) & (df['pct_chg'] <= max_chg)]
        c_chg = len(df)
        
        # Step 3: 上影线
        df = df[df['shadow_pct'] <= max_shadow]
        c_shadow = len(df)
        
        # Step 4: 获利盘
        df = df[df['winner_rate'] >= min_winner]
        c_winner = len(df)
        
        # 排除ST
        df = df[~df['name'].str.contains('ST', na=False)]
        df = df[df['circ_mv'] > 300000]
        
        debug_info = {
            "total": total,
            "after_price": c_price,
            "after_chg": c_chg,
            "after_shadow": c_shadow,
            "after_winner": c_winner,
            "market_status": market_status
        }
        
        sorted_df = df.sort_values('winner_rate', ascending=False)
            
        if show_debug:
             return sorted_df.head(top_n), debug_info
        
        return sorted_df.head(top_n), None
        
    except Exception as e:
        return "ERROR", None

# ==========================================
# 5. 侧边栏
# ==========================================
st.sidebar.header("🏹 参数设置")
token_input = st.sidebar.text_input("Tushare Token", type="password")
pro = get_pro_api(token_input)

st.sidebar.divider()
use_market_filter = st.sidebar.checkbox("开启大盘风控 (上证20日线)", value=False)
cfg_position_count = st.sidebar.number_input("持仓数", value=3)
cfg_min_winner = st.sidebar.number_input("最低获利盘(%)", value=50.0, step=1.0) 

# --- 新增：竞价风控参数 ---
st.sidebar.divider()
st.sidebar.caption("👇 T+1 竞价风控")
cfg_open_stop = st.sidebar.number_input("竞价止损线(%)", value=-2.0, step=0.5, max_value=0.0, help="如果次日开盘低于买入价2%，直接卖出")

st.sidebar.divider()
col_h1, col_h2 = st.sidebar.columns(2)
with col_h1:
    cfg_min_pct_chg = st.sidebar.number_input("最小涨幅(%)", value=3.0, step=0.5)
with col_h2:
    cfg_max_pct_chg = st.sidebar.number_input("最大涨幅(%)", value=7.5, step=0.5)

cfg_min_price = st.sidebar.number_input("最低股价(元)", value=10.0, step=0.1)
cfg_max_shadow = st.sidebar.number_input("最大上影线(%)", value=1.5, step=0.1)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    cfg_stop_loss = st.sidebar.number_input("盘中止损(%)", value=6.0)
with col_s2:
    cfg_max_hold = st.sidebar.number_input("持仓天", value=5)

cfg_trail_start = 0.10 
cfg_trail_drop = 0.03  
stop_loss_decimal = cfg_stop_loss / 100.0
open_stop_decimal = cfg_open_stop / 100.0

today = datetime.now()
start_date = st.sidebar.text_input("开始日期", value=f"{today.year}0101")
end_date = st.sidebar.text_input("结束日期", value=today.strftime('%Y%m%d'))

# ==========================================
# 6. 主程序
# ==========================================
st.title("🚀 V46.0 竞价止损版")

tab1, tab2 = st.tabs(["🩺 实盘漏斗诊断", "📈 全年回测"])

with tab1:
    col_d, col_b = st.columns([3, 1])
    with col_d:
        def_date = datetime.now() - timedelta(days=2) 
        scan_date_input = st.date_input("选择诊断日期", value=def_date)
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("开始诊断", type="primary"):
        if not pro: st.stop()
        
        with st.spinner(f"正在分析 {scan_date_str} 数据..."):
            idx_start = (pd.to_datetime(scan_date_str) - timedelta(days=60)).strftime('%Y%m%d')
            idx_df = fetch_index_data(token_input, idx_start, scan_date_str)
            data = fetch_day_task_right_side(scan_date_str, token_input)
            names_df = get_names(token_input)
            
            if data:
                result, debug_info = run_strategy_v46(data, names_df, cfg_min_winner, cfg_min_pct_chg, cfg_max_pct_chg, cfg_max_shadow, cfg_min_price, 20, idx_df, scan_date_str, use_market_filter, show_debug=True)
                
                if debug_info:
                    st.divider()
                    if debug_info['market_status'] == "BAD": st.error("大盘风控：🔴 红灯")
                    else: st.success("大盘风控：🟢 绿灯")
                    
                    funnel_data = [
                        {"步骤": "1. 初始全市场", "剩余数量": debug_info['total']},
                        {"步骤": f"2. 价格>10元", "剩余数量": debug_info['after_price']},
                        {"步骤": f"3. 涨幅 {cfg_min_pct_chg}%~{cfg_max_pct_chg}%", "剩余数量": debug_info['after_chg']},
                        {"步骤": f"4. 上影线<{cfg_max_shadow}%", "剩余数量": debug_info['after_shadow']},
                        {"步骤": f"5. 获利盘>{cfg_min_winner}%", "剩余数量": debug_info['after_winner']},
                    ]
                    st.dataframe(pd.DataFrame(funnel_data), use_container_width=True, hide_index=True)
                    
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        st.dataframe(result[['ts_code', 'name', 'close', 'pct_chg', 'shadow_pct', 'winner_rate']].style.format({
                            'close': '{:.2f}', 'pct_chg': '{:.2f}%', 'shadow_pct': '{:.2f}%', 'winner_rate': '{:.1f}%'
                        }), hide_index=True)
            else:
                st.error(f"❌ 无法获取 {scan_date_str} 的数据。")

with tab2:
    if st.button("🚀 启动回测", type="primary", use_container_width=True):
        if not token_input: st.stop()
        try:
            ts.set_token(token_input)
            pro = ts.pro_api()
            cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
            dates = sorted(cal_df['cal_date'].tolist())
            index_df = fetch_index_data(token_input, start_date, end_date)
        except: st.stop()
            
        memory_db = fetch_data_parallel_right(dates, token_input)
        names_df = get_names(token_input)
        if not memory_db: st.stop()
        
        active_signals = [] 
        finished_signals = [] 
        progress_bar = st.progress(0)
        valid_dates = sorted(list(memory_db.keys()))
        
        skipped_days = 0
        
        for i, date in enumerate(valid_dates):
            if i % 5 == 0: progress_bar.progress((i + 1) / len(valid_dates))
            
            snap = memory_db.get(date)
            price_map = {}
            if snap and not snap['daily'].empty:
                price_map = snap['daily'].set_index('ts_code')[['open','high','low','close']].to_dict('index')
            
            curr_dt = pd.to_datetime(date)
            next_active = []
            
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
                    
                    # --- V46.0 新增逻辑: 竞价风控 ---
                    # 检查是否是持仓第一天 (T+1)
                    held_days = (curr_dt - pd.to_datetime(sig['buy_date'])).days
                    is_t1 = (held_days <= 1) # 简化判断，实际应判断交易日，但此处近似可行
                    
                    t1_open_pct = 0.0
                    reason = ""
                    sell_p = pc
                    
                    # 1. T+1 开盘大跌 -> 竞价止损
                    if is_t1:
                        open_p = p['open']
                        t1_ret = (open_p - cost) / cost
                        t1_open_pct = t1_ret # 记录下来给用户看
                        
                        if t1_ret < open_stop_decimal:
                            reason = "竞价止损"
                            sell_p = open_p # 按开盘价卖出
                    
                    # 2. 如果没触发竞价止损，继续走常规逻辑
                    if not reason:
                        peak = sig['highest']
                        if (pl - cost) / cost <= -stop_loss_decimal:
                            reason = "破位止损"
                            sell_p = cost * (1 - stop_loss_decimal)
                        elif (peak - cost)/cost >= cfg_trail_start and (peak - pc)/peak >= cfg_trail_drop:
                            reason = "高位止盈"
                            sell_p = peak * (1 - cfg_trail_drop)
                        elif held_days >= cfg_max_hold:
                            reason = "动力不足"
                    
                    if reason:
                        ret = (sell_p - cost) / cost - 0.001
                        # 记录 t1_open_pct 方便分析
                        finished_signals.append({
                            'name': sig.get('name', code), 
                            'code': code, 
                            'buy_date': sig['buy_date'], 
                            'sell_date': date, 
                            'ret': ret, 
                            'reason': reason,
                            't1_open_pct': f"{t1_open_pct*100:.2f}%" if is_t1 else "N/A"
                        })
                    else:
                        next_active.append(sig)
                else:
                    next_active.append(sig)
            active_signals = next_active
            
            result, _ = run_strategy_v46(snap, names_df, cfg_min_winner, cfg_min_pct_chg, cfg_max_pct_chg, cfg_max_shadow, cfg_min_price, cfg_position_count, index_df, date, use_market_filter, show_debug=False)
            
            if isinstance(result, str) and result == "MARKET_BAD":
                skipped_days += 1
            elif isinstance(result, pd.DataFrame) and not result.empty:
                for _, row in result.iterrows():
                    code = row['ts_code']
                    if code in price_map:
                        active_signals.append({'code': code, 'name': row['name'] if 'name' in row else code, 'buy_date': date, 'buy_price': price_map[code]['close'], 'highest': price_map[code]['close']})
        
        progress_bar.empty()
        
        if finished_signals:
            df_res = pd.DataFrame(finished_signals)
            st.divider()
            st.info(f"风控检测：{skipped_days} 天空仓")
            c1, c2, c3 = st.columns(3)
            c1.metric("单笔期望", f"{df_res['ret'].mean()*100:.2f}%")
            c2.metric("胜率", f"{(df_res['ret']>0).mean()*100:.1f}%")
            c3.metric("交易次数", f"{len(df_res)}")
            
            st.subheader("📋 交易详情 (含 T+1 开盘数据)")
            st.dataframe(df_res[['name', 'code', 'buy_date', 'sell_date', 'ret', 'reason', 't1_open_pct']].style.format({'ret': '{:.2%}'}), use_container_width=True)
        else:
            st.warning("无交易")
