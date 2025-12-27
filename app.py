import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt
import datetime

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="V22.0 核动力回测", layout="wide")

# ==========================================
# 2. 侧边栏：参数 (随意拖动，不再卡顿)
# ==========================================
st.sidebar.header("🎛️ 极速控制台")

# 必须先设置 Token
my_token = st.sidebar.text_input("Tushare Token", type="password")

if st.sidebar.button("🧹 强制清空缓存 (重下数据)", type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

# --- 选股参数 (动这些不再需要重新下载！) ---
st.sidebar.subheader("🎯 选股标准 (秒级响应)")
cfg_min_price = st.sidebar.number_input("最低价 (元)", value=11.0, step=0.5)
cfg_max_price = st.sidebar.number_input("最高价 (元)", value=20.0, step=0.5)
cfg_max_turnover = st.sidebar.slider("最大换手率 (%)", 1.0, 10.0, 3.0, step=0.5)

st.sidebar.divider()

# --- 交易参数 (动这些更是秒级！) ---
st.sidebar.subheader("🛡️ 交易风控 (秒级响应)")
cfg_stop_loss = st.sidebar.slider("止损线 (-%)", 3.0, 15.0, 5.0, step=0.5) / 100.0
cfg_trail_start = st.sidebar.slider("止盈启动 (+%)", 5.0, 20.0, 8.0, step=1.0) / 100.0
cfg_trail_drop = st.sidebar.slider("回落卖出 (-%)", 1.0, 5.0, 3.0, step=0.5) / 100.0
cfg_max_hold = st.sidebar.slider("最长持股 (天)", 3, 20, 10)

# --- 核心：固定回测区间 ---
# 为了实现“一次下载，永久复用”，我们固定下载 20240501 到 20251231 的数据
# 您可以在这个大区间内任意回测
FIXED_START = "20240501"
FIXED_END = "20251231"

# ==========================================
# 3. 核心功能
# ==========================================
st.title("🚀 V22.0 核动力回测 (数据逻辑分离)")
st.caption(f"当前数据覆盖区间：{FIXED_START} ~ {FIXED_END}。在此区间内调整参数，无需重新下载。")

if not my_token:
    st.warning("👈 请先在左侧输入 Tushare Token")
    st.stop()

ts.set_token(my_token)
try:
    pro = ts.pro_api()
except Exception as e:
    st.error(f"Token 无效: {e}")
    st.stop()

# --- 核心黑科技：全量数据预加载 ---
# 这个函数没有任何参数！意味着只要代码不变，它永远只运行一次。
@st.cache_data(ttl=86400 * 30) # 缓存 30 天！
def download_all_data_v22():
    """
    一次性下载所有需要的交易日历、大盘数据。
    注意：由于个股日线数据量太大，我们采用“按日缓存”的策略，
    但把缓存粒度做到极致，不依赖任何选股参数。
    """
    # 1. 交易日历
    cal_df = pro.trade_cal(exchange='', start_date=FIXED_START, end_date=FIXED_END, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    
    # 2. 大盘情绪 (一次性下完)
    # 这里多取90天为了算MA20
    real_start = (pd.to_datetime(FIXED_START) - pd.Timedelta(days=90)).strftime('%Y%m%d')
    index_df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=FIXED_END)
    index_df = index_df.sort_values('trade_date', ascending=True)
    index_df['ma20'] = index_df['close'].rolling(20).mean()
    
    # 转成字典：{date: True/False}
    market_safe_map = index_df.set_index('trade_date')['close'].gt(index_df.set_index('trade_date')['ma20']).to_dict()
    
    return dates, market_safe_map

# --- 纯净的数据获取函数 (绝对不带任何选股参数) ---
@st.cache_data(ttl=86400 * 7) # 缓存 7 天
def fetch_daily_package_v22(date):
    """
    下载某一天的【全部】数据包。
    不管您选股条件是 11元还是20元，是3%换手还是5%换手，
    我都把这一天全市场的数据下载下来存好。
    这样您调整参数时，直接从这个包里拿数据，不用再找 Tushare 了。
    """
    try:
        # 1. 基础行情
        df_daily = pro.daily(trade_date=date)
        
        # 2. 每日指标 (换手、市值、PE)
        df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
        
        # 3. 筹码数据
        df_cyq = pro.cyq_perf(trade_date=date)
        if df_cyq.empty: # 容错回溯
             for i in range(1, 4):
                 prev = (pd.to_datetime(date) - pd.Timedelta(days=i)).strftime('%Y%m%d')
                 df_cyq = pro.cyq_perf(trade_date=prev)
                 if not df_cyq.empty: break
        
        # 4. 股票名称 (用于展示，非必须但体验好)
        # 注意：stock_basic 变动不大，其实可以单独缓存，这里为了省事合并放
        # 为了极速，这里暂不merge stock_basic，只在最后展示时取
        
        return df_daily, df_basic, df_cyq
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 纯逻辑计算 (毫秒级) ---
def calculate_signals_fast(date, df_daily, df_basic, df_cyq):
    """
    只在内存里做数学运算，不涉及网络 IO。
    """
    if df_daily.empty or df_basic.empty or df_cyq.empty: return None
    
    # 1. 内存合并 (极快)
    # 只保留需要的列以加速
    df_m1 = pd.merge(df_daily[['ts_code', 'close']], df_basic, on='ts_code')
    if 'cost_50pct' not in df_cyq.columns: return None
    df = pd.merge(df_m1, df_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code')
    
    # 2. 计算 Bias
    df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
    
    # 3. 筛选 (使用侧边栏的实时参数)
    condition = (
        (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
        (df['winner_rate'] < 70) &
        (df['circ_mv'] > 300000) &  
        (df['close'] >= cfg_min_price) &  # <--- 实时参数
        (df['close'] <= cfg_max_price) &  # <--- 实时参数
        (df['turnover_rate'] < cfg_max_turnover) # <--- 实时参数
    )
    
    sorted_df = df[condition].sort_values('bias', ascending=True)
    if sorted_df.empty: return None
    return sorted_df.iloc[0] # 返回冠军

# ==========================================
# 主程序
# ==========================================

# 1. 预加载基础数据 (只运行一次)
all_dates, market_safe_map = download_all_data_v22()

# 2. 界面控制
col1, col2 = st.columns([3, 1])
with col1:
    st.info(f"📅 数据就绪。覆盖交易日：{len(all_dates)} 天。")
with col2:
    start_btn = st.button("⚡ 极速回测", type="primary", use_container_width=True)

if start_btn:
    active_signals = [] 
    finished_signals = [] 
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 缓存“价格字典”以加速卖出判断，避免重复 query
    # 但为了逻辑简单，我们还是逐日 fetch_daily_package_v22 (它是缓存的，极快)
    
    for i, date in enumerate(all_dates):
        progress_bar.progress((i + 1) / len(all_dates))
        
        # 这一步虽然看起来在循环，但全是读内存缓存！快得飞起！
        df_daily, df_basic, df_cyq = fetch_daily_package_v22(date)
        
        # 构建价格查询字典 (O(1) 复杂度)
        price_map = {}
        if not df_daily.empty:
            df_daily = df_daily.set_index('ts_code')
            price_map = df_daily[['open', 'high', 'low', 'close']].to_dict('index')
            
        is_market_safe = market_safe_map.get(date, False)
        
        # --- 1. 卖出逻辑 (纯内存计算) ---
        signals_still_active = []
        current_date_obj = pd.to_datetime(date)
        
        for sig in active_signals:
            code = sig['code']
            # 如果还没到买入日期，跳过
            if current_date_obj <= pd.to_datetime(sig['buy_date']):
                # 更新最高价
                if code in price_map:
                     sig['highest'] = max(sig['highest'], price_map[code]['high'])
                signals_still_active.append(sig)
                continue

            if code in price_map:
                curr_high = price_map[code]['high']
                curr_low = price_map[code]['low']
                curr_close = price_map[code]['close']
                
                # 更新历史最高
                if curr_high > sig['highest']: sig['highest'] = curr_high
                
                cost = sig['buy_price']
                peak = sig['highest']
                peak_ret = (peak - cost) / cost
                drawdown = (peak - curr_close) / peak # 这里简化用收盘算回落，实盘可用high算
                
                reason = ""
                sell_price = curr_close
                
                # 动态参数判定
                if (curr_low - cost) / cost <= -cfg_stop_loss:
                    reason = "止损"
                    sell_price = cost * (1 - cfg_stop_loss)
                elif peak_ret >= cfg_trail_start and (peak - curr_close)/peak >= cfg_trail_drop:
                    # 注意：回测为了严谨，通常假设触发后以触发价成交，这里简化处理
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
                signals_still_active.append(sig) # 停牌等情况
        active_signals = signals_still_active
        
        # --- 2. 买入逻辑 (纯内存计算) ---
        if is_market_safe:
            # 这里调用的是纯逻辑函数，传入的是从缓存拿出的数据
            # 无论参数怎么变，数据源都不变
            champion = calculate_signals_fast(date, df_daily.reset_index() if not df_daily.empty else df_daily, df_basic, df_cyq)
            
            if champion is not None:
                code = champion['ts_code']
                if code in price_map:
                    active_signals.append({
                        'code': code, 'buy_date': date,
                        'buy_price': price_map[code]['open'], 'highest': price_map[code]['open']
                    })

    status_text.text("计算完成！")
    
    if finished_signals:
        df_res = pd.DataFrame(finished_signals)
        df_res['return_pct'] = df_res['return'] * 100
        
        win_rate = (df_res['return'] > 0).mean() * 100
        avg_ret = df_res['return'].mean() * 100
        total_ret = df_res['return'].sum() * 100
        
        st.divider()
        st.markdown(f"### 📊 极速回测报告 (区间 {FIXED_START}-{FIXED_END})")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("真实胜率", f"{win_rate:.1f}%")
        col2.metric("单笔期望", f"{avg_ret:.2f}%")
        col3.metric("虚拟总收益", f"{total_ret:.1f}%")
        col4.metric("交易次数", f"{len(df_res)}")
        
        # 图表
        chart = alt.Chart(df_res).mark_bar().encode(
            x=alt.X("return_pct", bin=alt.Bin(maxbins=40)),
            y='count()',
            color=alt.condition(alt.datum.return_pct > 0, alt.value("#d32f2f"), alt.value("#2e7d32"))
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df_res)
    else:
        st.warning("无交易信号")
