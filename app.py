import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt
import datetime

# ==========================================
# 页面配置 (必须第一行)
# ==========================================
st.set_page_config(page_title="V21.0 全能指挥官", layout="wide")

# ==========================================
# 侧边栏：参数控制中心
# ==========================================
st.sidebar.header("🎛️ 策略控制台")

# 1. 基础设置
my_token = st.sidebar.text_input("Tushare Token", type="password")
st.sidebar.divider()

# 2. 策略参数 (可调整，寻找最优解)
st.sidebar.subheader("🎯 选股标准")
cfg_min_price = st.sidebar.number_input("最低价 (元)", value=11.0, step=0.5)
cfg_max_price = st.sidebar.number_input("最高价 (元)", value=20.0, step=0.5)
cfg_max_turnover = st.sidebar.slider("最大换手率 (%)", 1.0, 10.0, 3.0, step=0.5, help="越低越缩量，越高越活跃")

st.sidebar.divider()

# 3. 交易参数 (影响回测结果)
st.sidebar.subheader("🛡️ 风控纪律")
cfg_stop_loss = st.sidebar.slider("止损线 (-%)", 3.0, 15.0, 5.0, step=0.5) / 100.0
cfg_trail_start = st.sidebar.slider("止盈启动 (+%)", 5.0, 20.0, 8.0, step=1.0) / 100.0
cfg_trail_drop = st.sidebar.slider("回落卖出 (-%)", 1.0, 5.0, 3.0, step=0.5) / 100.0
cfg_max_hold = st.sidebar.slider("最长持股 (天)", 3, 20, 10)

# 4. 回测区间 (用于测试牛熊)
st.sidebar.divider()
st.sidebar.subheader("⏳ 回测时间轴")
start_date = st.sidebar.text_input("开始日期", value="20240504")
end_date = st.sidebar.text_input("结束日期", value="20251226")

# ==========================================
# 核心功能区
# ==========================================
st.title("🚀 V21.0 全能指挥官 (缩量Rank1策略)")

if not my_token:
    st.warning("👈 请在左侧侧边栏输入 Tushare Token")
    st.stop()

ts.set_token(my_token)
try:
    pro = ts.pro_api()
except Exception as e:
    st.error(f"连接失败: {e}")
    st.stop()

# --- 智能工具函数 ---

def get_recent_trade_date(target_date_str):
    """智能回溯：如果当天是非交易日，自动寻找最近的一个交易日"""
    try:
        # 获取包含目标日期在内的过去10天交易日历
        end_dt = pd.to_datetime(target_date_str)
        start_dt = end_dt - pd.Timedelta(days=10)
        df = pro.trade_cal(exchange='', start_date=start_dt.strftime('%Y%m%d'), end_date=target_date_str, is_open='1')
        if not df.empty:
            return df['cal_date'].iloc[-1] # 返回最后一个（最近的）交易日
        return target_date_str
    except:
        return target_date_str

# --- 缓存数据函数 (纯净版) ---
@st.cache_data(ttl=86400)
def get_market_sentiment_v21(start, end):
    try:
        real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
        df = df.sort_values('trade_date', ascending=True)
        df['ma20'] = df['close'].rolling(20).mean()
        return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
    except: return {}

@st.cache_data(ttl=3600) # 短缓存，方便盘中更新
def fetch_daily_data_v21(date):
    try: return pro.daily(trade_date=date)
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_metrics_data_v21(date):
    try:
        df_basic = pro.daily_basic(trade_date=date, fields='ts_code,name,turnover_rate,circ_mv,pe_ttm,industry')
        df_cyq = pro.cyq_perf(trade_date=date)
        # 如果筹码没出，尝试找前几天的
        if df_cyq.empty:
             for i in range(1, 4):
                 prev = (pd.to_datetime(date) - pd.Timedelta(days=i)).strftime('%Y%m%d')
                 df_cyq = pro.cyq_perf(trade_date=prev)
                 if not df_cyq.empty: break
        
        if df_cyq.empty or 'cost_50pct' not in df_cyq.columns: return pd.DataFrame()
        
        df_merge = pd.merge(df_basic, df_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
        return df_merge
    except: return pd.DataFrame()

def run_strategy_logic(df_daily, df_metrics):
    """通用策略逻辑：输入行情和指标，返回 Rank 1"""
    if df_daily.empty or df_metrics.empty: return None
    
    df = pd.merge(df_daily, df_metrics, on='ts_code', how='inner')
    df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
    
    # === 使用侧边栏参数进行过滤 ===
    condition = (
        (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
        (df['winner_rate'] < 70) &
        (df['circ_mv'] > 300000) &  
        (df['close'] >= cfg_min_price) & 
        (df['close'] <= cfg_max_price) &
        (df['turnover_rate'] < cfg_max_turnover) # 动态参数
    )
    
    sorted_df = df[condition].sort_values('bias', ascending=True)
    return sorted_df

# ==========================================
# 双塔架构：Tab页切换
# ==========================================
tab1, tab2 = st.tabs(["📡 实盘扫描 (今日)", "🧪 历史回测 (验证)"])

# --- Tab 1: 实盘扫描 ---
with tab1:
    st.subheader("📡 实盘选股雷达")
    scan_date_input = st.date_input("选择日期 (周六日自动回溯)", value=pd.Timestamp.now())
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if st.button("开始扫描", type="primary"):
        # 1. 智能日期修正
        real_date_str = get_recent_trade_date(scan_date_str)
        if real_date_str != scan_date_str:
            st.info(f"📅 提示：{scan_date_str} 非交易日，已自动回溯至最近交易日：**{real_date_str}**")
        
        with st.spinner(f"正在扫描 {real_date_str} 全市场数据..."):
            df_daily = fetch_daily_data_v21(real_date_str)
            df_metrics = fetch_metrics_data_v21(real_date_str)
            
            result_df = run_strategy_logic(df_daily, df_metrics)
            
            if result_df is not None and not result_df.empty:
                champion = result_df.iloc[0]
                
                st.success(f"🏆 锁定冠军：**{champion['name']} ({champion['ts_code']})**")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("现价", f"{champion['close']}元")
                c2.metric("Bias", f"{champion['bias']:.4f}")
                c3.metric("换手率", f"{champion['turnover_rate']:.2f}%", delta=f"<{cfg_max_turnover}%")
                c4.metric("获利盘", f"{champion['winner_rate']:.1f}%")
                
                st.markdown(f"""
                **交易指令：**
                * 明日开盘买入。
                * **止损**：{champion['close'] * (1 - cfg_stop_loss):.2f} (跌 {cfg_stop_loss*100}%)
                * **止盈**：回落止盈 (触发 {cfg_trail_start*100}%, 回撤 {cfg_trail_drop*100}%)
                """)
                
                with st.expander("查看前10名备选"):
                    st.dataframe(result_df.head(10)[['ts_code', 'name', 'close', 'bias', 'turnover_rate', 'winner_rate']])
            else:
                st.warning("今日无符合条件的标的。请尝试在侧边栏放宽“最大换手率”或“价格区间”。")

# --- Tab 2: 历史回测 ---
with tab2:
    st.subheader("🧪 策略效能验证")
    st.caption("修改侧边栏参数，点击下方按钮，验证不同行情下的收益。")
    
    if st.button("🚀 运行全样本回测"):
        market_safe_map = get_market_sentiment_v21(start_date, end_date)
        cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
        dates = sorted(cal_df['cal_date'].tolist())
        
        active_signals = [] 
        finished_signals = [] 
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, date in enumerate(dates):
            progress_bar.progress((i + 1) / len(dates))
            is_market_safe = market_safe_map.get(date, False)
            
            # 每10天显示一次日志防止卡顿
            if i % 10 == 0: status_text.text(f"Backtesting: {date}")

            df_daily = fetch_daily_data_v21(date)
            df_metrics = fetch_metrics_data_v21(date)
            
            price_map_open = {}
            price_map_close = {}
            price_map_high = {}
            price_map_low = {}
            
            if not df_daily.empty:
                df_daily = df_daily.set_index('ts_code')
                price_map_open = df_daily['open'].to_dict()
                price_map_close = df_daily['close'].to_dict()
                price_map_high = df_daily['high'].to_dict()
                price_map_low = df_daily['low'].to_dict()
            
            # 1. 更新持仓
            signals_still_active = []
            current_date_obj = pd.to_datetime(date)
            
            for sig in active_signals:
                code = sig['code']
                if current_date_obj <= pd.to_datetime(sig['buy_date']):
                    if code in price_map_high:
                         sig['highest'] = max(sig['highest'], price_map_high[code])
                    signals_still_active.append(sig)
                    continue

                if code in price_map_close:
                    curr_price = price_map_close[code]
                    high_today = price_map_high.get(code, curr_price)
                    low_today = price_map_low.get(code, curr_price)
                    
                    if high_today > sig['highest']: sig['highest'] = high_today
                    
                    cost = sig['buy_price']
                    peak = sig['highest']
                    peak_ret = (peak - cost) / cost
                    drawdown = (peak - curr_price) / peak
                    
                    reason = ""
                    sell_price = curr_price
                    
                    # === 使用侧边栏配置的动态止盈止损 ===
                    if (low_today - cost) / cost <= -cfg_stop_loss:
                        reason = "止损"
                        sell_price = cost * (1 - cfg_stop_loss)
                    elif peak_ret >= cfg_trail_start and drawdown >= cfg_trail_drop:
                        reason = "止盈"
                        sell_price = peak * (1 - cfg_trail_drop)
                    elif (current_date_obj - pd.to_datetime(sig['buy_date'])).days >= cfg_max_hold:
                        reason = "超时"
                    
                    if reason:
                        ret = (sell_price - cost) / cost - 0.0006 # 双边手续费
                        finished_signals.append({
                            'code': code, 'buy_date': sig['buy_date'], 'return': ret, 'reason': reason
                        })
                    else:
                        signals_still_active.append(sig)
                else:
                    signals_still_active.append(sig)
            active_signals = signals_still_active

            # 2. 买入逻辑 (复用 run_strategy_logic)
            if is_market_safe:
                # 重新构造 DataFrame 传给策略函数
                # fetch_daily_data_v21 返回的是 Raw Data，需要配合 metrics
                # 这里为了性能，我们直接利用缓存的数据
                df_d = fetch_daily_data_v21(date)
                df_m = fetch_metrics_data_v21(date)
                
                target_df = run_strategy_logic(df_d, df_m)
                
                if target_df is not None and not target_df.empty:
                    # 取第一名 Rank 1
                    target_row = target_df.iloc[0]
                    code = target_row['ts_code']
                    if code in price_map_open:
                        active_signals.append({
                            'code': code, 'buy_date': date,
                            'buy_price': price_map_open[code], 'highest': price_map_open[code]
                        })

        status_text.empty()
        st.success("回测完成！")
        
        if finished_signals:
            df_res = pd.DataFrame(finished_signals)
            df_res['return_pct'] = df_res['return'] * 100
            
            win_rate = (df_res['return'] > 0).mean() * 100
            avg_ret = df_res['return'].mean() * 100
            total_ret = df_res['return'].sum() * 100
            
            st.markdown("### 📊 回测报告")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("真实胜率", f"{win_rate:.1f}%")
            col2.metric("单笔期望", f"{avg_ret:.2f}%", delta="关键指标")
            col3.metric("虚拟总收益", f"{total_ret:.1f}%")
            col4.metric("交易次数", f"{len(df_res)}")
            
            st.divider()
            
            # 分布图
            chart = alt.Chart(df_res).mark_bar().encode(
                x=alt.X("return_pct", bin=alt.Bin(maxbins=40), title="单笔收益分布(%)"),
                y='count()',
                color=alt.condition(alt.datum.return_pct > 0, alt.value("red"), alt.value("green"))
            )
            st.altair_chart(chart, use_container_width=True)
            
            with st.expander("查看交易明细"):
                st.dataframe(df_res)
        else:
            st.warning("该区间内未触发任何交易信号。")
