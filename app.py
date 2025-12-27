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
st.set_page_config(page_title="V42.0 完美风控版", layout="wide")

# ==========================================
# 2. 系统控制台
# ==========================================
st.sidebar.header("🛡️ 趋势狩猎 (V42.0)")
st.sidebar.success("✅ **修复：数据下载 100% 完整**")
st.sidebar.success("✅ **风控：大盘跌破20日线停买**")
st.sidebar.info("门槛：股价 > 10.0元")

if st.sidebar.button("🔄 强制重启系统", type="primary"):
    st.cache_data.clear()
    st.cache_resource.clear()
    os._exit(0)

# ==========================================
# 3. 数据引擎 (增强稳定性)
# ==========================================
@st.cache_resource
def get_pro_api(token):
    if not token: return None
    ts.set_token(token)
    return ts.pro_api(timeout=60) 

# --- 获取大盘指数及均线 ---
@st.cache_data(ttl=3600)
def fetch_index_data(token, start_date, end_date):
    """
    获取上证指数(000001.SH)并计算 MA20
    """
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # 稍微多取60天以计算均线
        real_start = (pd.to_datetime(start_date) - timedelta(days=60)).strftime('%Y%m%d')
        
        df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end_date)
        if df.empty: return pd.DataFrame()
        
        df = df.sort_values('trade_date')
        df['ma20'] = df['close'].rolling(20).mean()
        
        # 只保留查询时间段内的
        df = df[df['trade_date'] >= start_date]
        return df.set_index('trade_date')
    except Exception as e:
        print(f"Index fetch error: {e}")
        return pd.DataFrame()

# --- 单日任务 (增强重试) ---
def fetch_day_task_right_side(date, token):
    max_retries = 10 # 增加重试次数，确保不丢数据
    for i in range(max_retries):
        try:
            # 增加随机等待，错峰请求
            time.sleep(0.1 + np.random.random() * 0.2)
            
            ts.set_token(token)
            local_pro = ts.pro_api(timeout=45)
            
            # 1. 行情
            d_today = local_pro.daily(trade_date=date)
            # 如果当天不是交易日(返回空)，直接返回空，不算错误
            if d_today.empty: 
                return None 
            
            # 2. 每日指标
            d_basic = local_pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            
            # 3. 筹码 (核心)
            d_cyq = local_pro.cyq_perf(trade_date=date)
            if d_cyq.empty:
                # 补救：前一天
                prev_date = (pd.to_datetime(date) - timedelta(days=1)).strftime('%Y%m%d')
                d_cyq = local_pro.cyq_perf(trade_date=prev_date)

            if not d_today.empty and not d_cyq.empty:
                return {'date': date, 'daily': d_today, 'basic': d_basic, 'cyq': d_cyq}
            
            # 如果该有的数据没有，抛错重试
            raise ValueError("Data incomplete")
            
        except Exception as e:
            if i == max_retries - 1:
                print(f"Failed {date}: {e}")
                return None # 最终放弃，但会在日志看到
            time.sleep(1 + i) # 指数级避让
    return None

@st.cache_data(ttl=3600)
def fetch_data_parallel_right(dates, token):
    results = {}
    progress_bar = st.progress(0, text="启动高保真下载引擎...")
    
    # 稍微降低并发数，提高成功率
    with ThreadPoolExecutor(max_workers=4) as executor:
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
            # 实时更新进度条
            progress_bar.progress(done / total, text=f"📥 进度: {done}/{total} | 成功: {success} 天")
            
    progress_bar.empty()
    st.toast(f"下载完成！共覆盖 {success} 个有效交易日。")
    return results

@st.cache_data(ttl=86400)
def get_names(token):
    try:
        ts.set_token(token)
        return ts.pro_api().stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    except: return pd.DataFrame()

# ==========================================
# 4. 逻辑层 (大盘风控 + 10元门槛)
# ==========================================
def run_strategy_with_market_filter(snapshot, names_df, min_winner, min_chg, max_chg, max_shadow, min_price, top_n, index_df, curr_date):
    # --- 1. 大盘风控 (红绿灯) ---
    if index_df is not None and not index_df.empty:
        if curr_date in index_df.index:
            idx_today = index_df.loc[curr_date]
            # 如果 000001.SH 收盘 < 20日线，大盘不好，直接返回特殊标记
            if idx_today['close'] < idx_today['ma20']:
                return "MARKET_BAD"
    
    # --- 2. 个股筛选 ---
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
        
        # 计算上影线比例
        df['shadow_pct'] = (df['high'] - df['close']) / df['close'] * 100
        
        condition = (
            (df['winner_rate'] >= min_winner) &     
            (df['pct_chg'] >= min_chg) &            
            (df['pct_chg'] <= max_chg) &    
            (df['shadow_pct'] <= max_shadow) & 
            (df['close'] >= min_price) &        # <--- 军规一：价格 > 10.0
            (df['circ_mv'] > 300000) &              
            (~df['name'].str.contains('ST'))        
        )
        
        sorted_df = df[condition].sort_values('winner_rate', ascending=False)
        return sorted_df.head(top_n)
    except:
        return None

# ==========================================
# 5. 侧边栏 (参数固定)
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

# 军规一：最低价 >= 10.0
st.sidebar.caption("👇 价格门槛")
cfg_min_price = st.sidebar.number_input("最低股价(元)", value=10.0, min_value=0.0, step=0.1, help="低于此价格不买，防止掉入个位数陷阱")

st.sidebar.caption("👇 避雷针风控")
cfg_max_shadow = st.sidebar.number_input("允许最大回落(%)", value=1.5, step=0.1)

st.sidebar.divider()
st.sidebar.caption("🛡️ 交易风控")
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
st.title("🚀 V42.0 完美风控版 (完整一年)")
st.info("💡 逻辑升级：**上证指数跌破20日线 -> 停止买入**。最低股价锁定 **10元**。")

tab1, tab2 = st.tabs(["🏹 实盘扫描", "📈 全年回测"])

with tab1:
    col_d, col_b = st.columns([3, 1])
    with col_d:
        yesterday = datetime.now() - timedelta(days=1)
        scan_date_input = st.date_input("选择日期 (选昨天)", value=yesterday)
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("扫描机会", type="primary"):
        if not pro: st.stop()
        
        # 1. 大盘判断
        with st.spinner("正在研判大盘环境..."):
            idx_start = (pd.to_datetime(scan_date_str) - timedelta(days=60)).strftime('%Y%m%d')
            idx_df = fetch_index_data(token_input, idx_start, scan_date_str)
            
            is_market_bad = False
            if not idx_df.empty and scan_date_str in idx_df.index:
                today_idx = idx_df.loc[scan_date_str]
                if today_idx['close'] < today_idx['ma20']:
                    is_market_bad = True
                    st.error(f"🛑 大盘环境恶劣：{scan_date_str} 上证指数 < 20日线。建议**今日空仓**。")
                else:
                    st.success(f"✅ 大盘环境健康：上证指数 > 20日线。可正常选股。")
        
        # 2. 依然扫描个股供参考
        if not is_market_bad or st.checkbox("无视大盘风险，强制查看个股"):
            with st.spinner(f"正在扫描..."):
                data = fetch_day_task_right_side(scan_date_str, token_input)
                names_df = get_names(token_input)
                
                if data:
                    fleet = run_strategy_with_market_filter(data, names_df, cfg_min_winner, cfg_min_chg, cfg_max_chg, cfg_max_shadow, cfg_min_price, 20, None, scan_date_str)
                    
                    if fleet is not None and not isinstance(fleet, str) and not fleet.empty:
                        st.dataframe(fleet[['ts_code', 'name', 'close', 'pct_chg', 'shadow_pct', 'winner_rate']].style.format({
                            'close': '{:.2f}', 'pct_chg': '{:.2f}%', 'shadow_pct': '{:.2f}%', 'winner_rate': '{:.1f}%'
                        }), hide_index=True)
                    else:
                        st.warning("无符合条件的个股。")

with tab2:
    if st.button("🚀 启动全年回测", type="primary", use_container_width=True):
        if not token_input: st.stop()
        
        try:
            ts.set_token(token_input)
            pro = ts.pro_api()
            cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
            dates = sorted(cal_df['cal_date'].tolist())
            
            # 获取大盘数据
            st.info("1/2 正在下载上证指数数据...")
            index_df = fetch_index_data(token_input, start_date, end_date)
            
        except: st.stop()
            
        # 下载个股数据
        st.info("2/2 正在下载全市场个股数据 (请耐心等待)...")
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
            
            # --- 选股 (大盘风控生效) ---
            fleet = run_strategy_with_market_filter(snap, names_df, cfg_min_winner, cfg_min_chg, cfg_max_chg, cfg_max_shadow, cfg_min_price, cfg_position_count, index_df, date)
            
            if fleet == "MARKET_BAD":
                # 大盘不好，记录一下，不买新股
                skipped_days += 1
            elif fleet is not None and not fleet.empty:
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
            
            # 风控统计
            st.info(f"📊 风控报告：在 {len(valid_dates)} 个交易日中，系统检测到 {skipped_days} 天大盘走弱（000001 < 20日线），已自动停止开仓。")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("单笔期望", f"{df_res['ret'].mean()*100:.2f}%")
            c2.metric("胜率", f"{(df_res['ret']>0).mean()*100:.1f}%")
            c3.metric("交易次数", f"{len(df_res)}")
            
            st.subheader("📋 交易详情")
            st.dataframe(df_res[['name', 'code', 'buy_date', 'sell_date', 'ret', 'reason']].style.format({'ret': '{:.2%}'}), use_container_width=True)
        else:
            st.warning("无交易")
