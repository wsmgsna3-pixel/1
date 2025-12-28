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
st.set_page_config(page_title="V43.0 诊断调试版", layout="wide")

# ==========================================
# 2. 系统控制台
# ==========================================
st.sidebar.header("🛡️ 趋势狩猎 (V43.0)")
st.sidebar.success("✅ **功能：大盘风控可关闭**")
st.sidebar.info("新增：选股漏斗诊断，查看过滤详情")

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
    max_retries = 10 
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
# 4. 逻辑层 (带漏斗统计)
# ==========================================
def run_strategy_debug(snapshot, names_df, min_winner, min_chg, max_chg, max_shadow, min_price, top_n, index_df, curr_date, enable_market_filter, show_debug=False):
    # 1. 大盘风控
    market_status = "OK"
    if enable_market_filter and index_df is not None and not index_df.empty:
        if curr_date in index_df.index:
            idx_today = index_df.loc[curr_date]
            if idx_today['close'] < idx_today['ma20']:
                market_status = "BAD"
                if not show_debug: # 如果不是调试模式，直接返回
                    return "MARKET_BAD", None

    # 2. 个股筛选
    if not snapshot: return "NO_DATA", None
    d_today = snapshot.get('daily') 
    d_basic = snapshot.get('basic')
    d_cyq = snapshot.get('cyq')   
    
    if d_today is None or d_today.empty: return "NO_DATA", None
    
    try:
        # 合并
        m1 = pd.merge(d_today, d_basic, on='ts_code', how='inner')
        if names_df is not None:
            m1 = pd.merge(m1, names_df, on='ts_code', how='left')
        df = pd.merge(m1, d_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
        df['shadow_pct'] = (df['high'] - df['close']) / df['close'] * 100
        
        # --- 漏斗统计 ---
        total_count = len(df)
        
        # 步骤 1: 价格门槛
        df_price = df[df['close'] >= min_price]
        count_price = len(df_price)
        
        # 步骤 2: 涨幅门槛
        df_chg = df_price[(df_price['pct_chg'] >= min_chg) & (df_price['pct_chg'] <= max_chg)]
        count_chg = len(df_chg)
        
        # 步骤 3: 避雷针门槛
        df_shadow = df_chg[df_chg['shadow_pct'] <= max_shadow]
        count_shadow = len(df_shadow)
        
        # 步骤 4: 获利盘门槛
        df_final = df_shadow[df_shadow['winner_rate'] >= min_winner]
        count_final = len(df_final)
        
        # 排除 ST
        df_final = df_final[~df_final['name'].str.contains('ST', na=False)]
        df_final = df_final[df_final['circ_mv'] > 300000]
        
        debug_info = {
            "total": total_count,
            "after_price": count_price,
            "after_chg": count_chg,
            "after_shadow": count_shadow,
            "after_winner": count_final,
            "market_status": market_status
        }
        
        if market_status == "BAD" and not show_debug:
             return "MARKET_BAD", None

        sorted_df = df_final.sort_values('winner_rate', ascending=False)
        return sorted_df.head(top_n), debug_info
        
    except Exception as e:
        return "ERROR", None

# ==========================================
# 5. 侧边栏
# ==========================================
st.sidebar.header("🏹 参数设置")
token_input = st.sidebar.text_input("Tushare Token", type="password")
pro = get_pro_api(token_input)

st.sidebar.divider()
# --- 开关 ---
use_market_filter = st.sidebar.checkbox("开启大盘风控 (上证20日线)", value=True, help="取消勾选可强制回测所有日期")

cfg_position_count = st.sidebar.number_input("持仓数", value=3)
cfg_min_winner = st.sidebar.number_input("最低获利盘(%)", value=50.0, step=1.0) # 默认改低一点试试

col_c1, col_c2 = st.sidebar.columns(2)
with col_c1:
    cfg_min_chg = st.sidebar.number_input("最小涨幅(%)", value=2.0, step=0.5)
with col_c2:
    cfg_max_chg = st.sidebar.number_input("最大涨幅(%)", value=7.0, step=0.5)

cfg_min_price = st.sidebar.number_input("最低股价(元)", value=10.0, step=0.1)
cfg_max_shadow = st.sidebar.number_input("最大上影线(%)", value=1.5, step=0.1)

st.sidebar.divider()
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    cfg_stop_loss = st.sidebar.number_input("止损(%)", value=6.0)
with col_s2:
    cfg_max_hold = st.sidebar.number_input("持仓天", value=5)

cfg_trail_start = 0.10 
cfg_trail_drop = 0.03  
stop_loss_decimal = cfg_stop_loss / 100.0

today = datetime.now()
start_date = st.sidebar.text_input("开始日期", value=f"{today.year}0101")
end_date = st.sidebar.text_input("结束日期", value=today.strftime('%Y%m%d'))

# ==========================================
# 6. 主程序
# ==========================================
st.title("🚀 V43.0 诊断调试版")

tab1, tab2 = st.tabs(["🩺 实盘漏斗诊断", "📈 全年回测"])

with tab1:
    col_d, col_b = st.columns([3, 1])
    with col_d:
        yesterday = datetime.now() - timedelta(days=1)
        scan_date_input = st.date_input("选择诊断日期", value=yesterday)
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("开始诊断", type="primary"):
        if not pro: st.stop()
        
        with st.spinner("正在解剖数据..."):
            idx_start = (pd.to_datetime(scan_date_str) - timedelta(days=60)).strftime('%Y%m%d')
            idx_df = fetch_index_data(token_input, idx_start, scan_date_str)
            
            data = fetch_day_task_right_side(scan_date_str, token_input)
            names_df = get_names(token_input)
            
            if data:
                # 开启 show_debug=True，强制返回统计信息
                result, debug_info = run_strategy_debug(data, names_df, cfg_min_winner, cfg_min_chg, cfg_max_chg, cfg_max_shadow, cfg_min_price, 20, idx_df, scan_date_str, use_market_filter, show_debug=True)
                
                if debug_info:
                    st.subheader("🕵️‍♂️ 选股漏斗 (数据真相)")
                    
                    # 1. 大盘状态
                    market_str = "🟢 绿灯 (安全)" if debug_info['market_status'] == "OK" else "🔴 红灯 (危险)"
                    if not use_market_filter: market_str += " [已强制无视]"
                    st.metric("大盘环境", market_str)
                    
                    # 2. 漏斗图
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("1.全市场", debug_info['total'])
                    c2.metric("2.价格>10元", debug_info['after_price'], f"-{debug_info['total']-debug_info['after_price']} 只")
                    c3.metric("3.涨幅2~7%", debug_info['after_chg'], f"-{debug_info['after_price']-debug_info['after_chg']} 只")
                    c4.metric("4.上影线<1.5%", debug_info['after_shadow'], f"-{debug_info['after_chg']-debug_info['after_shadow']} 只")
                    c5.metric(f"5.获利盘>{cfg_min_winner}%", debug_info['after_winner'], f"-{debug_info['after_shadow']-debug_info['after_winner']} 只")
                    
                    if debug_info['after_winner'] == 0:
                        st.error("结论：没有任何股票满足条件。请根据上方漏斗查看是哪一步杀得太狠，并调整对应参数。")
                    else:
                        st.success(f"最终选出 {len(result)} 只股票")
                        st.dataframe(result[['ts_code', 'name', 'close', 'pct_chg', 'shadow_pct', 'winner_rate']], hide_index=True)

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
            
            # 持仓更新
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
                        finished_signals.append({'name': sig.get('name', code), 'code': code, 'buy_date': sig['buy_date'], 'sell_date': date, 'ret': ret, 'reason': reason})
                    else:
                        next_active.append(sig)
                else:
                    next_active.append(sig)
            active_signals = next_active
            
            # 选股
            result, _ = run_strategy_debug(snap, names_df, cfg_min_winner, cfg_min_chg, cfg_max_chg, cfg_max_shadow, cfg_min_price, cfg_position_count, index_df, date, use_market_filter, show_debug=False)
            
            if result == "MARKET_BAD":
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
            if use_market_filter:
                st.info(f"🛡️ 风控开启：共 {len(valid_dates)} 天，其中 {skipped_days} 天因大盘不好停止开仓。")
            else:
                st.warning("⚠️ 风控关闭：已无视大盘环境，强制全时段交易。")
                
            c1, c2, c3 = st.columns(3)
            c1.metric("单笔期望", f"{df_res['ret'].mean()*100:.2f}%")
            c2.metric("胜率", f"{(df_res['ret']>0).mean()*100:.1f}%")
            c3.metric("交易次数", f"{len(df_res)}")
            st.dataframe(df_res[['name', 'code', 'buy_date', 'sell_date', 'ret', 'reason']].style.format({'ret': '{:.2%}'}), use_container_width=True)
        else:
            st.warning("无交易")
