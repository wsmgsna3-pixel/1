import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt
import time

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="V25.0 原子战舰", layout="wide")

# ==========================================
# 2. 侧边栏：极速控制台
# ==========================================
st.sidebar.header("🎛️ 极速参数面板")

# Token
my_token = st.sidebar.text_input("Tushare Token", type="password")

st.sidebar.divider()

# --- 选股参数 (热调整：绝不触发重下数据) ---
st.sidebar.subheader("🎯 选股标准 (热切换)")
# 使用 columns 让界面更紧凑，参考 V30
c1, c2 = st.sidebar.columns(2)
cfg_min_price = c1.number_input("最低价", value=11.0, step=0.5)
cfg_max_price = c2.number_input("最高价", value=20.0, step=0.5)
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
st.title("🚀 V25.0 原子战舰 (V30同款架构)")
st.caption("核心技术：数据层与逻辑层彻底分离。调整参数**无需**重新下载数据。")

if not my_token:
    st.warning("👈 请先在左侧输入 Tushare Token")
    st.stop()

ts.set_token(my_token)
try:
    pro = ts.pro_api()
except Exception as e:
    st.error(f"Token 无效: {e}")
    st.stop()

# ==========================================
# 4. 数据层 (Data Layer) - 只负责下载和缓存
# ==========================================

# 这里的参数只有 date！没有价格、换手率等业务参数。
# 所以无论业务参数怎么变，这个缓存永远有效！
@st.cache_data(ttl=86400 * 7) 
def fetch_daily_atomic_data(date):
    """
    原子化获取单日全市场数据。
    不做任何筛选，原样下载。
    """
    try:
        # 1. 基础行情
        df_daily = pro.daily(trade_date=date)
        
        # 2. 每日指标
        df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
        
        # 3. 股票名称
        df_names = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        
        # 4. 筹码数据
        df_cyq = pro.cyq_perf(trade_date=date)
        if df_cyq.empty: # 容错回溯
             for i in range(1, 4):
                 prev = (pd.to_datetime(date) - pd.Timedelta(days=i)).strftime('%Y%m%d')
                 df_cyq = pro.cyq_perf(trade_date=prev)
                 if not df_cyq.empty: break
        
        # 打包返回，不进行 merge，因为 merge 也可以在逻辑层做，保持数据层纯净
        return {'daily': df_daily, 'basic': df_basic, 'names': df_names, 'cyq': df_cyq}
    except:
        return {}

@st.cache_data(ttl=86400)
def get_market_sentiment(start, end):
    try:
        real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
        df = df.sort_values('trade_date', ascending=True)
        df['ma20'] = df['close'].rolling(20).mean()
        return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
    except: return {}

# ==========================================
# 5. 逻辑层 (Logic Layer) - 纯内存计算，极快
# ==========================================

def run_strategy_memory(snapshot, p_min, p_max, to_max):
    """
    纯内存筛选。速度是毫秒级的。
    """
    if not snapshot: return None
    
    d1 = snapshot.get('daily')
    d2 = snapshot.get('basic')
    d3 = snapshot.get('names')
    d4 = snapshot.get('cyq')
    
    if d1 is None or d1.empty: return None
    if d2 is None or d2.empty: return None
    if d4 is None or d4.empty or 'cost_50pct' not in d4.columns: return None
    
    # 内存合并 (Merge 是很快的)
    m1 = pd.merge(d1, d2, on='ts_code')
    if d3 is not None and not d3.empty:
        m1 = pd.merge(m1, d3, on='ts_code')
    df = pd.merge(m1, d4[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code')
    
    # 计算因子
    df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
    
    # === 核心：这里的筛选使用传入的参数 ===
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
    return sorted_df.iloc[0]

# ==========================================
# 6. 主程序
# ==========================================
tab1, tab2 = st.tabs(["📡 实盘扫描", "🧪 历史回测"])

# --- Tab 1: 实盘 ---
with tab1:
    col_d, col_b = st.columns([3,1])
    with col_d:
        scan_date_input = st.date_input("选择日期", value=pd.Timestamp.now())
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("开始扫描", type="primary", use_container_width=True):
        with st.spinner("读取原子数据..."):
            snap = fetch_daily_atomic_data(scan_date_str)
            # 调用逻辑层
            champion = run_strategy_memory(snap, cfg_min_price, cfg_max_price, cfg_max_turnover)
            
            if champion is not None:
                st.success(f"🏆 冠军：{champion['ts_code']} | {champion['name']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("现价", f"{champion['close']}")
                c2.metric("Bias", f"{champion['bias']:.4f}")
                c3.metric("换手", f"{champion['turnover_rate']:.2f}%")
                c4.metric("获利盘", f"{champion['winner_rate']:.1f}%")
            else:
                st.warning("无符合条件的标的。")

# --- Tab 2: 回测 (参数秒级调整) ---
with tab2:
    st.caption("ℹ️ 说明：第一次运行会下载数据。下载完成后，调整侧边栏参数，点击运行，结果秒出。")
    
    if st.button("🚀 运行回测", type="primary", use_container_width=True):
        
        # 1. 获取日期
        cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
        dates = sorted(cal_df['cal_date'].tolist())
        market_safe_map = get_market_sentiment(start_date, end_date)
        
        active_signals = [] 
        finished_signals = [] 
        
        # 进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, date in enumerate(dates):
            # 更新进度
            progress_bar.progress((i + 1) / len(dates))
            
            # === A. 数据层 (有缓存则极快，无缓存则下载) ===
            snap = fetch_daily_atomic_data(date)
            
            # 构建价格查询字典 (加速)
            price_map = {}
            if snap and not snap['daily'].empty:
                d_idx = snap['daily'].set_index('ts_code')
                price_map = d_idx[['open', 'high', 'low', 'close']].to_dict('index')
            
            is_market_safe = market_safe_map.get(date, False)
            
            # === B. 逻辑层 (纯内存计算) ===
            
            # 1. 持仓管理 (使用实时参数 cfg_xxx)
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
                    
                    reason = ""
                    sell_price = curr_close
                    
                    # 实时计算止盈止损
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
            
            # 2. 选股买入 (使用实时参数 cfg_xxx)
            if is_market_safe:
                # 这里传入的 snap 是缓存的，cfg 参数是实时的
                champion = run_strategy_memory(snap, cfg_min_price, cfg_max_price, cfg_max_turnover)
                
                if champion is not None:
                    code = champion['ts_code']
                    if code in price_map:
                        active_signals.append({
                            'code': code, 'buy_date': date,
                            'buy_price': price_map[code]['open'], 'highest': price_map[code]['open']
                        })

        status_text.text("分析完成")
        
        if finished_signals:
            df_res = pd.DataFrame(finished_signals)
            df_res['return_pct'] = df_res['return'] * 100
            
            win_rate = (df_res['return'] > 0).mean() * 100
            avg_ret = df_res['return'].mean() * 100
            total_ret = df_res['return'].sum() * 100
            
            st.divider()
            st.markdown("### 📊 回测结果")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("胜率", f"{win_rate:.1f}%")
            c2.metric("期望", f"{avg_ret:.2f}%")
            c3.metric("总收益", f"{total_ret:.1f}%")
            c4.metric("交易数", f"{len(df_res)}")
            
            chart = alt.Chart(df_res).mark_bar().encode(
                x=alt.X("return_pct", bin=alt.Bin(maxbins=30)),
                y='count()',
                color=alt.condition(alt.datum.return_pct > 0, alt.value("red"), alt.value("green"))
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(df_res)
        else:
            st.info("区间内无交易。")
