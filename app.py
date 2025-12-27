import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
import time
import gc
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="V37.4 精准输入版", layout="wide")

# ==========================================
# 2. 系统控制台
# ==========================================
st.sidebar.header("🛠️ 系统控制台")
st.sidebar.success("✅ V37.4 (全数字输入框)")

if st.sidebar.button("🔥 强制重启", type="primary"):
    st.cache_data.clear()
    st.cache_resource.clear()
    os._exit(0)

# ==========================================
# 3. 数据引擎 (稳定版)
# ==========================================
@st.cache_resource
def get_pro_api(token):
    if not token: return None
    ts.set_token(token)
    return ts.pro_api(timeout=60) 

def get_latest_trade_date(_pro, curr_date_str):
    try:
        end_dt = pd.to_datetime(curr_date_str)
        start_dt = end_dt - timedelta(days=60)
        df = _pro.trade_cal(exchange='', start_date=start_dt.strftime('%Y%m%d'), 
                            end_date=curr_date_str, is_open='1')
        if df.empty: return curr_date_str
        df = df.sort_values('cal_date', ascending=False)
        return df['cal_date'].iloc[0]
    except:
        return curr_date_str

def fetch_day_task_robust(date, token):
    max_retries = 5
    for i in range(max_retries):
        try:
            time.sleep(0.1)
            ts.set_token(token)
            local_pro = ts.pro_api(timeout=30)
            
            d1 = local_pro.daily(trade_date=date)
            if d1.empty: return None 
            
            d2 = local_pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            
            d4 = local_pro.cyq_perf(trade_date=date)
            if d4.empty:
                 prev_day = (pd.to_datetime(date) - timedelta(days=1)).strftime('%Y%m%d')
                 d4 = local_pro.cyq_perf(trade_date=prev_day)

            if not d1.empty and not d4.empty:
                return {'date': date, 'daily': d1, 'basic': d2, 'cyq': d4}
            raise ValueError("Empty Data")
        except:
            if i == max_retries - 1: return None
            time.sleep(1 + i)
    return None

@st.cache_data(ttl=3600)
def fetch_data_parallel_robust(dates, token):
    results = {}
    progress_bar = st.progress(0, text="启动下载引擎...")
    status_box = st.empty()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(fetch_day_task_robust, d, token): d for d in dates}
        total = len(dates)
        done = 0
        success = 0
        for future in as_completed(future_map):
            done += 1
            data = future.result()
            if data:
                results[data['date']] = data
                success += 1
            progress_bar.progress(done / total, text=f"📥 下载进度: {done}/{total} (成功: {success})")
            
    progress_bar.empty()
    status_box.success(f"✅ 数据就绪！成功获取 {success} 天真筹码数据。")
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
def run_strategy_rank1(snapshot, names_df, p_min, p_max, to_max, top_n):
    if not snapshot: return None
    d1 = snapshot.get('daily')
    d2 = snapshot.get('basic')
    d4 = snapshot.get('cyq')
    
    if d1 is None or d1.empty or d4 is None or d4.empty: return None
    if 'cost_50pct' not in d4.columns: return None
    
    try:
        m1 = pd.merge(d1, d2, on='ts_code', how='inner')
        if names_df is not None:
            m1 = pd.merge(m1, names_df, on='ts_code', how='left')
            
        df = pd.merge(m1, d4[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
        
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
# 5. 侧边栏 (全数字输入框版)
# ==========================================
st.sidebar.header("🎛️ 尊享控制台")
token_input = st.sidebar.text_input("Tushare Token", type="password")
pro = get_pro_api(token_input)

st.sidebar.divider()
st.sidebar.caption("👇 全键盘精准输入模式")

# 修改1：Top N 改为数字输入，step=1
cfg_position_count = st.sidebar.number_input("Top N (每日持仓数)", value=3, min_value=1, max_value=10, step=1)

# 修改2：最低价/最高价 保持数字输入，明确 step=0.1
col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    cfg_min_price = st.sidebar.number_input("最低价(元)", value=8.1, min_value=0.0, step=0.1)
with col_p2:
    cfg_max_price = st.sidebar.number_input("最高价(元)", value=20.0, min_value=0.0, step=0.1)

# 修改3：换手率 改为数字输入，不再是滑块！
cfg_max_turnover = st.sidebar.number_input("换手率上限(%)", value=2.1, min_value=0.1, max_value=20.0, step=0.1, help="直接输入数字，或用+/-微调")

st.sidebar.divider()

# 修改4：止损/持仓天数 保持数字输入
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    cfg_stop_loss = st.sidebar.number_input("止损线(%)", value=8.5, min_value=0.0, step=0.1)
with col_s2:
    cfg_max_hold = st.sidebar.number_input("持仓天数", value=15, min_value=1, step=1)

cfg_trail_start = 0.08
cfg_trail_drop = 0.03
stop_loss_decimal = cfg_stop_loss / 100.0

today = datetime.now()
start_date = st.sidebar.text_input("开始日期", value=f"{today.year}0101")
end_date = st.sidebar.text_input("结束日期", value=today.strftime('%Y%m%d'))

# ==========================================
# 6. 主程序
# ==========================================
st.title("🚀 V37.4 精准输入版")

tab1, tab2 = st.tabs(["📡 智能实盘", "🧪 并发回测"])

with tab1:
    col_d, col_b = st.columns([3, 1])
    with col_d:
        scan_date_input = st.date_input("选择日期", value=pd.Timestamp.now())
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("开始扫描", type="primary"):
        if not pro: st.stop()
        with st.spinner("智能校对日期..."):
            real_date_str = get_latest_trade_date(pro, scan_date_str)
            if real_date_str != scan_date_str:
                st.info(f"📅 修正：{scan_date_str} -> {real_date_str}")
            
            data = fetch_day_task_robust(real_date_str, token_input)
            names_df = get_names(token_input)
            
            if data:
                fleet = run_strategy_rank1(data, names_df, cfg_min_price, cfg_max_price, cfg_max_turnover, cfg_position_count)
                if fleet is not None and not fleet.empty:
                    st.success(f"⚓ 选出 {len(fleet)} 只标的")
                    st.dataframe(fleet[['ts_code', 'name', 'close', 'bias', 'turnover_rate', 'winner_rate', 'industry']].style.format({
                        'close': '{:.2f}', 'bias': '{:.4f}', 'turnover_rate': '{:.2f}', 'winner_rate': '{:.1f}'
                    }), hide_index=True)
                else:
                    st.warning(f"在 {real_date_str} 未找到标的。")
            else:
                st.error("无法获取数据。")

with tab2:
    if st.button("🚀 启动回测", type="primary", use_container_width=True):
        if not token_input:
            st.error("Token 无效")
            st.stop()
            
        try:
            ts.set_token(token_input)
            pro = ts.pro_api()
            cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
            dates = sorted(cal_df['cal_date'].tolist())
        except:
            st.error("网络初始化失败")
            st.stop()
            
        memory_db = fetch_data_parallel_robust(dates, token_input)
        names_df = get_names(token_input)
        
        if not memory_db:
            st.error("未下载到有效数据")
            st.stop()
        
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
                        finished_signals.append({
                            'name': sig.get('name', code),
                            'code': code,
                            'buy_date': sig['buy_date'],
                            'sell_date': date,
                            'ret': ret, 
                            'rank': sig.get('rank', 1),
                            'reason': reason
                        })
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
                            'code': code, 
                            'name': row['name'] if 'name' in row else code,
                            'buy_date': date, 
                            'buy_price': price_map[code]['open'], 
                            'highest': price_map[code]['open'],
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
            
            st.subheader("🏆 分名次统计")
            rank_stats = df_res.groupby('rank')['ret_pct'].agg(['count', 'mean', 'sum', lambda x: (x>0).mean()*100])
            st.table(rank_stats.style.format("{:.2f}").background_gradient(subset=['mean'], cmap='RdYlGn'))
            
            st.subheader("📋 交易详情 (复盘专用)")
            st.dataframe(
                df_res[['name', 'code', 'buy_date', 'sell_date', 'ret_pct', 'rank', 'reason']].sort_values('buy_date', ascending=False),
                use_container_width=True,
                column_config={
                    "ret_pct": st.column_config.NumberColumn("收益率%", format="%.2f"),
                    "name": "股票名称",
                    "reason": "卖出原因"
                }
            )
        else:
            st.warning("无交易")
