import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt
import datetime
import time

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="V24.0 原子战舰", layout="wide")

# ==========================================
# 2. 侧边栏：极速控制台
# ==========================================
st.sidebar.header("🎛️ V24.0 控制台")

# Token
my_token = st.sidebar.text_input("Tushare Token", type="password")

st.sidebar.divider()

# --- 选股参数 (热调整：绝不触发重下数据) ---
st.sidebar.subheader("🎯 选股标准 (热切换)")
cfg_min_price = st.sidebar.number_input("最低价 (元)", value=11.0, step=0.5)
cfg_max_price = st.sidebar.number_input("最高价 (元)", value=20.0, step=0.5)
cfg_max_turnover = st.sidebar.slider("最大换手率 (%)", 1.0, 10.0, 3.0, step=0.5)

st.sidebar.divider()

# --- 交易参数 (热调整) ---
st.sidebar.subheader("🛡️ 交易风控 (热切换)")
cfg_stop_loss = st.sidebar.slider("止损线 (-%)", 3.0, 15.0, 5.0, step=0.5) / 100.0
cfg_trail_start = st.sidebar.slider("止盈启动 (+%)", 5.0, 20.0, 8.0, step=1.0) / 100.0
cfg_trail_drop = st.sidebar.slider("回落卖出 (-%)", 1.0, 5.0, 3.0, step=0.5) / 100.0
cfg_max_hold = st.sidebar.slider("最长持股 (天)", 3, 20, 10)

# --- 回测区间 ---
st.sidebar.divider()
st.sidebar.subheader("⏳ 时间轴")
start_date = st.sidebar.text_input("开始日期", value="20240504")
end_date = st.sidebar.text_input("结束日期", value="20251226")

# ==========================================
# 3. 核心功能
# ==========================================
st.title("🚀 V24.0 原子战舰 (参考 V30 缓存架构)")
st.caption("核心逻辑：将数据拆解为‘单日原子快照’。调整侧边栏参数**不会**触发重新下载。")

if not my_token:
    st.warning("👈 请先在左侧输入 Tushare Token")
    st.stop()

ts.set_token(my_token)
try:
    pro = ts.pro_api()
except Exception as e:
    st.error(f"Token 无效: {e}")
    st.stop()

# --- 核心：复刻 V30.12.3 的缓存逻辑 ---
# 这是一个“原子化”的函数，只负责拿某一天的纯数据，不带任何业务逻辑参数！
@st.cache_data(ttl=86400 * 7) 
def fetch_daily_atomic_snapshot(date):
    """
    原子化获取单日全市场数据。
    参考 nb.txt 中的 fetch_and_cache_daily_data 设计。
    """
    try:
        # 1. 基础行情 (Open/High/Low/Close)
        df_daily = pro.daily(trade_date=date)
        
        # 2. 每日指标 (换手、市值、PE)
        df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
        
        # 3. 股票名称 (一次性获取，防止 KeyError)
        df_names = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        
        # 4. 筹码数据 (Rank 1 核心)
        df_cyq = pro.cyq_perf(trade_date=date)
        if df_cyq.empty: # 容错回溯
             for i in range(1, 4):
                 prev = (pd.to_datetime(date) - pd.Timedelta(days=i)).strftime('%Y%m%d')
                 df_cyq = pro.cyq_perf(trade_date=prev)
                 if not df_cyq.empty: break
        
        return {'daily': df_daily, 'basic': df_basic, 'names': df_names, 'cyq': df_cyq}
    except:
        return {}

# 辅助：获取大盘情绪
@st.cache_data(ttl=86400)
def get_market_sentiment_atomic(start, end):
    try:
        real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
        df = df.sort_values('trade_date', ascending=True)
        df['ma20'] = df['close'].rolling(20).mean()
        return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
    except: return {}

# --- 纯逻辑处理 (运行在内存中，极快) ---
def process_day_logic(snapshot, min_p, max_p, max_to):
    """
    这是纯计算逻辑，输入是 snapshot（缓存的数据）和 参数。
    """
    if not snapshot: return None
    
    d1 = snapshot.get('daily', pd.DataFrame())
    d2 = snapshot.get('basic', pd.DataFrame())
    d3 = snapshot.get('names', pd.DataFrame())
    d4 = snapshot.get('cyq', pd.DataFrame())
    
    if d1.empty or d2.empty or d3.empty or d4.empty: return None
    if 'cost_50pct' not in d4.columns: return None
    
    # 内存合并
    m1 = pd.merge(d1, d2, on='ts_code')
    m2 = pd.merge(m1, d3, on='ts_code')
    df = pd.merge(m2, d4[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code')
    
    # 计算 Bias
    df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
    
    # 筛选 (使用传入的参数)
    condition = (
        (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
        (df['winner_rate'] < 70) &
        (df['circ_mv'] > 300000) &  
        (df['close'] >= min_p) & 
        (df['close'] <= max_p) & 
        (df['turnover_rate'] < max_to)
    )
    
    sorted_df = df[condition].sort_values('bias', ascending=True)
    if sorted_df.empty: return None
    return sorted_df.iloc[0]

# ==========================================
# 4. 双塔显示
# ==========================================
tab1, tab2 = st.tabs(["📡 实盘扫描 (今日)", "🧪 历史回测 (参数热调整)"])

# --- Tab 1: 实盘 ---
with tab1:
    st.subheader("📡 实盘选股")
    scan_date_input = st.date_input("选择日期", value=pd.Timestamp.now())
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if st.button("开始扫描", type="primary"):
        with st.spinner("正在获取原子快照..."):
            # 1. 拿数据 (缓存)
            snap = fetch_daily_atomic_snapshot(scan_date_str)
            # 2. 跑逻辑 (实时)
            champion = process_day_logic(snap, cfg_min_price, cfg_max_price, cfg_max_turnover)
            
            if champion is not None:
                st.success(f"🏆 冠军代码：{champion['ts_code']} | {champion['name']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("现价", f"{champion['close']}元")
                c2.metric("Bias", f"{champion['bias']:.4f}")
                c3.metric("换手率", f"{champion['turnover_rate']:.2f}%")
                c4.metric("获利盘", f"{champion['winner_rate']:.1f}%")
            else:
                st.warning("无符合条件的标的。")

# --- Tab 2: 回测 ---
with tab2:
    st.subheader("🧪 极速回测")
    st.info("💡 提示：因为采用了 V30 的缓存架构，第一次运行会下载每一天的数据（有进度条）。跑完一次后，**调整任何侧边栏参数，都无需等待，立刻出结果**。")
    
    if st.button("🚀 运行回测", type="primary", use_container_width=True):
        
        # 1. 获取日期序列
        cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
        dates = sorted(cal_df['cal_date'].tolist())
        market_safe_map = get_market_sentiment_atomic(start_date, end_date)
        
        active_signals = [] 
        finished_signals = [] 
        
        # 进度条 (致敬 V30 风格)
        progress_bar = st.progress(0, text="启动回测引擎...")
        
        for i, date in enumerate(dates):
            progress_bar.progress((i + 1) / len(dates), text=f"正在分析: {date} (数据命中缓存)")
            
            # === A. 获取数据 (缓存命中率 100% 后极快) ===
            snap = fetch_daily_atomic_snapshot(date)
            
            # 构建价格查询字典 (加速卖出判断)
            price_map = {}
            if snap and not snap['daily'].empty:
                d_indexed = snap['daily'].set_index('ts_code')
                price_map = d_indexed[['open', 'high', 'low', 'close']].to_dict('index')
            
            is_market_safe = market_safe_map.get(date, False)
            
            # === B. 持仓处理 (实时风控参数) ===
            signals_still_active = []
            current_date_obj = pd.to_datetime(date)
            
            for sig in active_signals:
                code = sig['code']
                if current_date_obj <= pd.to_datetime(sig['buy_date']):
                    if code in price_map:
                         sig['highest'] = max(sig['highest'], price_map[code]['high'])
                    signals_still_active.append(sig)
                    continue

                if code in price_map:
                    curr_high = price_map[code]['high']
                    curr_low = price_map[code]['low']
                    curr_close = price_map[code]['close']
                    
                    if curr_high > sig['highest']: sig['highest'] = curr_high
                    
                    cost = sig['buy_price']
                    peak = sig['highest']
                    peak_ret = (peak - cost) / cost
                    drawdown = (peak - curr_close) / peak
                    
                    reason = ""
                    sell_price = curr_close
                    
                    # 实时参数
                    if (curr_low - cost) / cost <= -cfg_stop_loss:
                        reason = "止损"
                        sell_price = cost * (1 - cfg_stop_loss)
                    elif peak_ret >= cfg_trail_start and (peak - curr_close)/peak >= cfg_trail_drop:
                        reason = "止盈"
                        sell_price = peak * (1 - cfg_trail_drop)
                    elif (current_date_obj - pd.to_datetime(sig['buy_date'])).days >= cfg_max_hold:
                        reason = "超时"
                    
                    if reason:
                        ret = (sell_price - cost) / cost - 0.0006 
                        finished_signals.append({
                            'code': code, 'buy_date': sig['buy_date'], 'return': ret, 'reason': reason
                        })
                    else:
                        signals_still_active.append(sig)
                else:
                    signals_still_active.append(sig)
            active_signals = signals_still_active
            
            # === C. 买入逻辑 (实时筛选参数) ===
            if is_market_safe:
                # 这一步调用逻辑层，传入参数。
                # 无论参数怎么变，snap 是不变的，所以不需要重新下载。
                champion = process_day_logic(snap, cfg_min_price, cfg_max_price, cfg_max_turnover)
                
                if champion is not None:
                    code = champion['ts_code']
                    if code in price_map:
                        active_signals.append({
                            'code': code, 'buy_date': date,
                            'buy_price': price_map[code]['open'], 'highest': price_map[code]['open']
                        })

        progress_bar.empty()
        
        if finished_signals:
            df_res = pd.DataFrame(finished_signals)
            df_res['return_pct'] = df_res['return'] * 100
            
            win_rate = (df_res['return'] > 0).mean() * 100
            avg_ret = df_res['return'].mean() * 100
            total_ret = df_res['return'].sum() * 100
            
            st.divider()
            st.markdown("### 📊 V24.0 回测报告")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("真实胜率", f"{win_rate:.1f}%")
            col2.metric("单笔期望", f"{avg_ret:.2f}%")
            col3.metric("虚拟总收益", f"{total_ret:.1f}%")
            col4.metric("交易次数", f"{len(df_res)}")
            
            chart = alt.Chart(df_res).mark_bar().encode(
                x=alt.X("return_pct", bin=alt.Bin(maxbins=40)),
                y='count()',
                color=alt.condition(alt.datum.return_pct > 0, alt.value("#d32f2f"), alt.value("#2e7d32"))
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(df_res)
        else:
            st.warning("该区间内无交易。")
