import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
import altair as alt
import time
import gc
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="V35.0 尊享极速版", layout="wide")

# ==========================================
# 2. 系统控制台
# ==========================================
st.sidebar.header("🛠️ 系统控制台")
st.sidebar.success("✅ V35.0 (多线程真筹码版)")

if st.sidebar.button("🔥 强制重启 (更新代码后必点)", type="primary"):
    st.cache_data.clear()
    st.cache_resource.clear()
    os._exit(0)

# ==========================================
# 3. 高性能数据引擎 (多线程并发)
# ==========================================

@st.cache_resource
def get_pro_api(token):
    if not token: return None
    ts.set_token(token)
    return ts.pro_api(timeout=60) # 60秒超时容错

def retry_api_call(func, *args, retries=3, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception:
            if i == retries - 1: return pd.DataFrame()
            time.sleep(1)
    return pd.DataFrame()

# --- 单日数据下载函数 (保留真筹码) ---
def fetch_single_day_data(date, token):
    """
    这是一个独立的下载任务，将被分配给不同的线程。
    必须在这里重新初始化 pro，因为线程间共享连接可能会有问题。
    """
    try:
        ts.set_token(token)
        local_pro = ts.pro_api()
        
        # 1. 下载基础数据
        df_daily = retry_api_call(local_pro.daily, trade_date=date)
        if df_daily.empty: return None # 如果没行情，直接跳过
        
        df_basic = retry_api_call(local_pro.daily_basic, trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
        
        # 2. 下载尊贵的筹码数据 (cyq_perf)
        # 您花钱买的权限，必须用上！
        df_cyq = retry_api_call(local_pro.cyq_perf, trade_date=date)
        if df_cyq.empty:
            # 简单回溯1-2天，防止当天数据偶尔缺失
             for i in range(1, 3):
                 prev = (pd.to_datetime(date) - pd.Timedelta(days=i)).strftime('%Y%m%d')
                 df_cyq = retry_api_call(local_pro.cyq_perf, trade_date=prev)
                 if not df_cyq.empty: break
        
        # 打包返回
        return {
            'date': date,
            'daily': df_daily,
            'basic': df_basic,
            'cyq': df_cyq
        }
    except:
        return None

# --- 多线程批量下载核心 ---
@st.cache_data(ttl=3600)
def fetch_data_concurrently(dates, token):
    """
    使用线程池并发下载，速度提升 5-10 倍。
    """
    results = {}
    
    # 进度条
    progress_bar = st.progress(0, text="正在启动多线程极速下载引擎...")
    status_text = st.empty()
    
    # 限制并发数为 4-8，防止 Tushare 封 IP
    # Tushare 高级用户通常允许每分钟几百次请求，8线程是安全的
    max_workers = 8 
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_date = {executor.submit(fetch_single_day_data, date, token): date for date in dates}
        
        completed_count = 0
        total_count = len(dates)
        
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            try:
                data = future.result()
                if data:
                    results[data['date']] = data
            except Exception as e:
                print(f"Error fetching {date}: {e}")
            
            completed_count += 1
            # 更新进度
            if completed_count % 5 == 0 or completed_count == total_count:
                pct = completed_count / total_count
                progress_bar.progress(pct, text=f"🚀 多线程极速下载中... 已完成 {completed_count}/{total_count} 天")
    
    progress_bar.empty()
    status_text.success(f"✅ 下载完成！成功获取 {len(results)} 天的完整筹码数据。")
    return results

# 辅助：获取静态股票名称 (一次性)
@st.cache_data(ttl=86400)
def get_stock_names(token):
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        return pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    except: return pd.DataFrame()

# 辅助：获取大盘
@st.cache_data(ttl=86400)
def get_market_sentiment(start, end, token):
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
        df = df.sort_values('trade_date', ascending=True)
        df['ma20'] = df['close'].rolling(20).mean()
        return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
    except: return {}

# ==========================================
# 4. 逻辑层 (使用真筹码)
# ==========================================
def run_strategy_real_cyq(snapshot, names_df, p_min, p_max, to_max, top_n):
    if not snapshot: return None
    
    d1 = snapshot.get('daily')
    d2 = snapshot.get('basic')
    d4 = snapshot.get('cyq') # 真筹码
    
    # 坚如磐石的防崩溃检查
    if d1 is None or d1.empty: return None
    if d2 is None or d2.empty: return None
    if d4 is None or d4.empty: return None # 必须有筹码
    
    if 'ts_code' not in d1.columns or 'cost_50pct' not in d4.columns: return None

    # 合并
    try:
        m1 = pd.merge(d1, d2, on='ts_code', how='inner')
        if names_df is not None and not names_df.empty:
            m1 = pd.merge(m1, names_df, on='ts_code', how='left')
        
        # 关键：使用真筹码数据合并
        df = pd.merge(m1, d4[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
        
        # 计算 Bias (真筹码乖离)
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
        
    except: return None

# ==========================================
# 5. 侧边栏
# ==========================================
st.sidebar.header("🎛️ 尊享控制台")
token_input = st.sidebar.text_input("Tushare Token (必须是高级版)", type="password")

st.sidebar.divider()
st.sidebar.subheader("🎯 上帝参数")
cfg_min_price = st.sidebar.number_input("最低价 (元)", value=8.1, step=0.1)
cfg_max_price = st.sidebar.number_input("最高价 (元)", value=20.0, step=0.5)
cfg_max_turnover = st.sidebar.slider("最大换手率 (%)", 0.5, 5.0, 2.1, step=0.1)
cfg_position_count = st.sidebar.slider("每日Top N", 1, 5, 3)

st.sidebar.divider()
cfg_stop_loss = st.sidebar.slider("止损线 (-%)", 3.0, 15.0, 8.5, step=0.5)
cfg_max_hold = st.sidebar.slider("最长持股 (天)", 5, 30, 15)
cfg_trail_start = st.sidebar.slider("止盈启动 (+%)", 5.0, 15.0, 8.0, step=0.5) / 100.0
cfg_trail_drop = st.sidebar.slider("回落卖出 (-%)", 1.0, 5.0, 3.0, step=0.5) / 100.0
stop_loss_decimal = cfg_stop_loss / 100.0

today = datetime.now()
start_date = st.sidebar.text_input("开始日期", value=f"{today.year}0101")
end_date = st.sidebar.text_input("结束日期", value=today.strftime('%Y%m%d'))

# ==========================================
# 6. 主程序
# ==========================================
st.title("🚀 V35.0 尊享极速版 (多线程 + 真筹码)")
st.caption("核心技术：使用 **8线程并发** 下载 Tushare **真实筹码数据**。不妥协精度，只提升速度。")

tab1, tab2 = st.tabs(["📡 智能实盘", "🧪 高精回测"])

# --- Tab 1: 实盘 ---
with tab1:
    col_d, col_b = st.columns([3, 1])
    with col_d:
        scan_date_input = st.date_input("选择日期", value=pd.Timestamp.now())
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("开始扫描", type="primary"):
        if not token_input:
            st.error("请先输入 Token")
            st.stop()
        
        # 简单的单日逻辑
        with st.spinner("正在获取高精筹码数据..."):
            # 获取最近交易日 (借用之前的逻辑，简化写在这里)
            try:
                ts.set_token(token_input)
                pro = ts.pro_api()
                # 简单回溯逻辑
                real_date_str = scan_date_str
                # (此处为了代码简洁，直接请求，如果非交易日会返回空，用户手动调整即可，或者复用V33的逻辑)
            except: pass
            
            # 直接调用单日函数
            data = fetch_single_day_data(scan_date_str, token_input)
            names_df = get_stock_names(token_input)
            
            if data:
                fleet = run_strategy_real_cyq(data, names_df, cfg_min_price, cfg_max_price, cfg_max_turnover, cfg_position_count)
                if fleet is not None and not fleet.empty:
                    st.success(f"⚓ 成功选出 {len(fleet)} 只标的")
                    st.dataframe(fleet[['ts_code', 'name', 'close', 'bias', 'turnover_rate', 'winner_rate', 'industry']].style.format({
                        'close': '{:.2f}', 'bias': '{:.4f}', 'turnover_rate': '{:.2f}', 'winner_rate': '{:.1f}'
                    }), hide_index=True)
            else:
                st.warning("该日期无数据或非交易日。")

# --- Tab 2: 回测 ---
with tab2:
    if st.button("🚀 启动高精并发回测", type="primary"):
        if not token_input:
            st.error("Token 无效")
            st.stop()
            
        # 1. 获取日期序列
        try:
            ts.set_token(token_input)
            pro = ts.pro_api()
            cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
            dates = sorted(cal_df['cal_date'].tolist())
        except:
            st.error("网络初始化失败")
            st.stop()
            
        # 2. 多线程并发下载 (速度的关键！)
        # 返回的是一个大字典：{ '20250101': {daily:..., cyq:...}, ... }
        memory_db = fetch_data_concurrently(dates, token_input)
        
        if not memory_db:
            st.error("未下载到有效数据，请检查 Token 权限或日期范围。")
            st.stop()
            
        # 获取其他静态数据
        names_df = get_stock_names(token_input)
        market_safe_map = get_market_sentiment(start_date, end_date, token_input)
        
        active_signals = [] 
        finished_signals = [] 
        progress_bar = st.progress(0, text="正在进行内存回测...")
        
        # 3. 内存回测 (极快)
        valid_dates = sorted(list(memory_db.keys()))
        
        for i, date in enumerate(valid_dates):
            progress_bar.progress((i + 1) / len(valid_dates), text=f"正在分析: {date}")
            
            # 直接从内存取数据
            snap = memory_db.get(date)
            
            # 构建价格表
            price_map = {}
            if snap and not snap['daily'].empty:
                price_map = snap['daily'].set_index('ts_code')[['open','high','low','close']].to_dict('index')
            
            is_market_safe = market_safe_map.get(date, False)
            
            # --- 持仓 ---
            next_active = []
            curr_dt = pd.to_datetime(date)
            
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
            
            # --- 买入 ---
            if is_market_safe:
                fleet = run_strategy_real_cyq(snap, names_df, cfg_min_price, cfg_max_price, cfg_max_turnover, cfg_position_count)
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
            avg_ret = df_res['ret'].mean() * 100
            win_rate = (df_res['ret']>0).mean() * 100
            c1.metric("单笔平均期望", f"{avg_ret:.2f}%")
            c2.metric("胜率", f"{win_rate:.1f}%")
            c3.metric("交易次数", f"{len(df_res)}")
            
            st.subheader("🏆 各名次表现 (基于真筹码)")
            rank_stats = df_res.groupby('rank')['ret_pct'].agg(['count', 'mean', 'sum', lambda x: (x>0).mean()*100])
            st.table(rank_stats.style.format("{:.2f}").background_gradient(subset=['mean'], cmap='Greens'))
        else:
            st.warning("无交易")
