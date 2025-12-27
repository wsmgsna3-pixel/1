import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt
import datetime

# ==========================================
# 1. 页面配置 (必须放在代码的第一行)
# ==========================================
st.set_page_config(page_title="V21.1 全能指挥官(修复版)", layout="wide")

# ==========================================
# 2. 侧边栏：参数控制中心
# ==========================================
st.sidebar.header("🎛️ 策略控制台")

# --- 救命按钮 (专门解决缓存卡死问题) ---
if st.sidebar.button("🧹 强制刷新数据 (报错时点我)", type="primary"):
    st.cache_data.clear()
    st.rerun()

# 基础设置
my_token = st.sidebar.text_input("Tushare Token", type="password")
st.sidebar.divider()

# 策略参数
st.sidebar.subheader("🎯 选股标准")
cfg_min_price = st.sidebar.number_input("最低价 (元)", value=11.0, step=0.5)
cfg_max_price = st.sidebar.number_input("最高价 (元)", value=20.0, step=0.5)
cfg_max_turnover = st.sidebar.slider("最大换手率 (%)", 1.0, 10.0, 3.0, step=0.5, help="核心参数：越低代表主力控盘越稳")

st.sidebar.divider()

# 交易参数
st.sidebar.subheader("🛡️ 风控纪律")
cfg_stop_loss = st.sidebar.slider("止损线 (-%)", 3.0, 15.0, 5.0, step=0.5) / 100.0
cfg_trail_start = st.sidebar.slider("止盈启动 (+%)", 5.0, 20.0, 8.0, step=1.0) / 100.0
cfg_trail_drop = st.sidebar.slider("回落卖出 (-%)", 1.0, 5.0, 3.0, step=0.5) / 100.0
cfg_max_hold = st.sidebar.slider("最长持股 (天)", 3, 20, 10)

# 回测时间
st.sidebar.divider()
st.sidebar.subheader("⏳ 回测时间轴")
start_date = st.sidebar.text_input("开始日期", value="20240504")
end_date = st.sidebar.text_input("结束日期", value="20251226")

# ==========================================
# 3. 核心功能区
# ==========================================
st.title("🚀 V21.1 全能指挥官 (实盘修复版)")

if not my_token:
    st.warning("👈 请先在左侧输入 Tushare Token")
    st.stop()

ts.set_token(my_token)
try:
    pro = ts.pro_api()
except Exception as e:
    st.error(f"Token 无效或连接失败: {e}")
    st.stop()

# --- 智能工具函数 ---

def get_recent_trade_date(target_date_str):
    """智能回溯：如果当天是非交易日，自动寻找最近的一个交易日"""
    try:
        end_dt = pd.to_datetime(target_date_str)
        start_dt = end_dt - pd.Timedelta(days=10)
        df = pro.trade_cal(exchange='', start_date=start_dt.strftime('%Y%m%d'), end_date=target_date_str, is_open='1')
        if not df.empty:
            return df['cal_date'].iloc[-1]
        return target_date_str
    except:
        return target_date_str

# --- 核心数据获取 (已修复 Name 缺失问题) ---

@st.cache_data(ttl=86400)
def get_market_sentiment_v21(start, end):
    try:
        real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
        df = df.sort_values('trade_date', ascending=True)
        df['ma20'] = df['close'].rolling(20).mean()
        return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
    except: return {}

@st.cache_data(ttl=3600)
def fetch_daily_data_v21(date):
    try: return pro.daily(trade_date=date)
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_metrics_data_v21(date):
    """
    修复版：同时获取 指标(daily_basic)、名称(stock_basic) 和 筹码(cyq_perf)
    """
    try:
        # 1. 获取每日指标
        df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
        
        # 2. 获取股票名称 (关键修复步骤！)
        df_names = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        
        # 3. 获取筹码数据 (带简单的向前回溯容错)
        df_cyq = pro.cyq_perf(trade_date=date)
        if df_cyq.empty:
             for i in range(1, 4): # 如果当天没筹码，往前找3天
                 prev = (pd.to_datetime(date) - pd.Timedelta(days=i)).strftime('%Y%m%d')
                 df_cyq = pro.cyq_perf(trade_date=prev)
                 if not df_cyq.empty: break
        
        if df_cyq.empty or 'cost_50pct' not in df_cyq.columns: return pd.DataFrame()
        
        # 4. 合并三张表
        # 先把 指标 和 名称 合并
        df_temp = pd.merge(df_basic, df_names, on='ts_code', how='inner')
        # 再把 结果 和 筹码 合并
        df_final = pd.merge(df_temp, df_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
        
        return df_final
    except: return pd.DataFrame()

def run_strategy_logic(df_daily, df_metrics):
    """通用策略逻辑"""
    if df_daily.empty or df_metrics.empty: return None
    
    df = pd.merge(df_daily, df_metrics, on='ts_code', how='inner')
    df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
    
    # 筛选逻辑
    condition = (
        (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
        (df['winner_rate'] < 70) &
        (df['circ_mv'] > 300000) &  
        (df['close'] >= cfg_min_price) & 
        (df['close'] <= cfg_max_price) &
        (df['turnover_rate'] < cfg_max_turnover) # 核心缩量参数
    )
    
    sorted_df = df[condition].sort_values('bias', ascending=True)
    return sorted_df

# ==========================================
# 4. 双塔显示 (实盘 + 回测)
# ==========================================
tab1, tab2 = st.tabs(["📡 实盘扫描 (今日)", "🧪 历史回测 (验证)"])

# --- Tab 1: 实盘扫描 ---
with tab1:
    st.subheader("📡 实盘选股雷达")
    col_date, col_btn = st.columns([3, 1])
    with col_date:
        scan_date_input = st.date_input("选择日期 (周六日自动回溯)", value=pd.Timestamp.now())
    
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if st.button("开始扫描", type="primary", use_container_width=True):
        # 智能日期修正
        real_date_str = get_recent_trade_date(scan_date_str)
        if real_date_str != scan_date_str:
            st.info(f"📅 提示：{scan_date_str} 非交易日，已自动回溯至最近交易日：**{real_date_str}**")
        
        with st.spinner(f"正在扫描 {real_date_str} 全市场数据..."):
            df_daily = fetch_daily_data_v21(real_date_str)
            df_metrics = fetch_metrics_data_v21(real_date_str)
            
            result_df = run_strategy_logic(df_daily, df_metrics)
            
            if result_df is not None and not result_df.empty:
                champion = result_df.iloc[0]
                
                # 冠军展示区
                st.success(f"🏆 锁定冠军：**{champion['name']} ({champion['ts_code']})**")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("现价", f"{champion['close']}元")
                c2.metric("Bias", f"{champion['bias']:.4f}", help="越小越好")
                c3.metric("换手率", f"{champion['turnover_rate']:.2f}%", delta=f"<{cfg_max_turnover}%")
                c4.metric("获利盘", f"{champion['winner_rate']:.1f}%")
                
                st.markdown(f"""
                ---
                **📝 交易计划：**
                1.  **买入**：明日开盘买入 {champion['name']}。
                2.  **止损**：跌破 **{champion['close'] * (1 - cfg_stop_loss):.2f} 元** (-{cfg_stop_loss*100}%) 坚决离场。
                3.  **止盈**：当涨幅超过 **{cfg_trail_start*100}%** 后，若回撤 **{cfg_trail_drop*100}%** 则止盈卖出。
                """)
                
                with st.expander("查看前10名备选池"):
                    st.dataframe(result_df.head(10)[['ts_code', 'name', 'close', 'bias', 'turnover_rate', 'winner_rate', 'industry']])
            else:
                st.warning("今日无符合条件的标的。请尝试在侧边栏放宽“最大换手率”或“价格区间”。")

# --- Tab 2: 历史回测 ---
with tab2:
    st.subheader("🧪 策略效能验证")
    st.caption("调整侧边栏参数 -> 点击下方按钮 -> 验证不同行情下的表现")
    
    if st.button("🚀 运行全样本回测", use_container_width=True):
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
            
            if i % 10 == 0: status_text.text(f"正在回测: {date}")

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
            
            # 1. 持仓管理
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
                    
                    # 动态止盈止损逻辑
                    if (low_today - cost) / cost <= -cfg_stop_loss:
                        reason = "止损"
                        sell_price = cost * (1 - cfg_stop_loss)
                    elif peak_ret >= cfg_trail_start and drawdown >= cfg_trail_drop:
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

            # 2. 选股买入
            if is_market_safe:
                # 复用策略函数
                target_df = run_strategy_logic(fetch_daily_data_v21(date), fetch_metrics_data_v21(date))
                
                if target_df is not None and not target_df.empty:
                    target_row = target_df.iloc[0]
                    code = target_row['ts_code']
                    if code in price_map_open:
                        active_signals.append({
                            'code': code, 'buy_date': date,
                            'buy_price': price_map_open[code], 'highest': price_map_open[code]
                        })

        status_text.empty()
        st.success("✅ 回测完成！")
        
        if finished_signals:
            df_res = pd.DataFrame(finished_signals)
            df_res['return_pct'] = df_res['return'] * 100
            
            win_rate = (df_res['return'] > 0).mean() * 100
            avg_ret = df_res['return'].mean() * 100
            total_ret = df_res['return'].sum() * 100
            
            st.markdown("### 📊 验证报告")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("真实胜率", f"{win_rate:.1f}%")
            col2.metric("单笔期望", f"{avg_ret:.2f}%", delta="越高越好")
            col3.metric("虚拟总收益", f"{total_ret:.1f}%")
            col4.metric("交易次数", f"{len(df_res)}")
            
            st.divider()
            
            # 简单的红绿柱状图
            chart = alt.Chart(df_res).mark_bar().encode(
                x=alt.X("return_pct", bin=alt.Bin(maxbins=40), title="单笔收益分布(%)"),
                y='count()',
                color=alt.condition(alt.datum.return_pct > 0, alt.value("#d32f2f"), alt.value("#2e7d32"))
            )
            st.altair_chart(chart, use_container_width=True)
            
            with st.expander("查看交易明细"):
                st.dataframe(df_res)
        else:
            st.warning("该区间内未触发任何交易信号。")
