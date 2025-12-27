import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
import altair as alt
import time
import gc

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="V29.0 实战指挥官", layout="wide")

# ==========================================
# 2. 全局缓存架构 (核心稳固)
# ==========================================

@st.cache_resource
def get_pro_api(token):
    if not token: return None
    ts.set_token(token)
    return ts.pro_api()

@st.cache_data(ttl=86400 * 7)
def fetch_daily_atomic_data(date, _pro):
    if _pro is None: return {}
    try:
        # 1. 基础数据
        df_daily = _pro.daily(trade_date=date)
        df_basic = _pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
        
        # 2. 筹码数据 (Rank 1 灵魂)
        df_cyq = _pro.cyq_perf(trade_date=date)
        if df_cyq.empty: 
             for i in range(1, 4):
                 prev = (pd.to_datetime(date) - pd.Timedelta(days=i)).strftime('%Y%m%d')
                 df_cyq = _pro.cyq_perf(trade_date=prev)
                 if not df_cyq.empty: break
        
        # 3. 股票名称
        df_names = _pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        
        return {'daily': df_daily, 'basic': df_basic, 'cyq': df_cyq, 'names': df_names}
    except Exception:
        return {}

@st.cache_data(ttl=86400)
def get_market_sentiment(start, end, _pro):
    if _pro is None: return {}
    try:
        real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        df = _pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
        df = df.sort_values('trade_date', ascending=True)
        df['ma20'] = df['close'].rolling(20).mean()
        return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
    except: return {}

# ==========================================
# 3. 纯内存逻辑 (秒级响应)
# ==========================================
def run_strategy_logic(snapshot, p_min, p_max, to_max):
    if not snapshot: return None
    d1, d2, d3, d4 = snapshot.get('daily'), snapshot.get('basic'), snapshot.get('names'), snapshot.get('cyq')
    
    if d1 is None or d1.empty or d2 is None or d2.empty or d4 is None or d4.empty: return None
    if 'cost_50pct' not in d4.columns: return None
    
    # 内存合并
    m1 = pd.merge(d1, d2, on='ts_code')
    m2 = pd.merge(m1, d3, on='ts_code')
    df = pd.merge(m2, d4[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code')
    
    df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
    
    # === 动态筛选 (参数可调) ===
    condition = (
        (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
        (df['winner_rate'] < 70) &
        (df['circ_mv'] > 300000) &  
        (df['close'] >= p_min) &       # 动态最低价
        (df['close'] <= p_max) &       # 动态最高价
        (df['turnover_rate'] < to_max) # 动态换手率
    )
    
    sorted_df = df[condition].sort_values('bias', ascending=True)
    if sorted_df.empty: return None
    return sorted_df.iloc[0]

# ==========================================
# 4. 侧边栏：实战控制台 (上帝参数预设)
# ==========================================
st.sidebar.header("🎛️ 实战控制台")

# Token
token_input = st.sidebar.text_input("Tushare Token", type="password")
pro = get_pro_api(token_input)

st.sidebar.divider()
st.sidebar.subheader("🎯 选股标准 (默认最优解)")

# --- 核心参数 (预设为 294% 收益的上帝参数) ---
cfg_min_price = st.sidebar.number_input("最低价 (元)", value=8.1, step=0.1, help="回测最优解：8.1元")
cfg_max_price = st.sidebar.number_input("最高价 (元)", value=20.0, step=0.5)
cfg_max_turnover = st.sidebar.slider("最大换手率 (%)", 0.5, 5.0, 2.1, step=0.1, help="回测最优解：2.1% (极致缩量)")

st.sidebar.divider()
st.sidebar.subheader("🛡️ 交易风控")

# --- 风控参数 (预设为最优解) ---
cfg_stop_loss = st.sidebar.slider("止损线 (-%)", 3.0, 15.0, 8.5, step=0.5, help="回测最优解：8.5% (宽止损)")
cfg_max_hold = st.sidebar.slider("最长持股 (天)", 5, 30, 15, help="回测最优解：15天 (耐心持有)")

# 止盈参数 (这俩影响相对较小，保持默认即可，也可微调)
cfg_trail_start = st.sidebar.slider("止盈启动 (+%)", 5.0, 15.0, 8.0, step=0.5) / 100.0
cfg_trail_drop = st.sidebar.slider("回落卖出 (-%)", 1.0, 5.0, 3.0, step=0.5) / 100.0

# 转换止损为小数
stop_loss_decimal = cfg_stop_loss / 100.0

st.sidebar.divider()
# 时间轴
start_date = st.sidebar.text_input("回测开始", value="20250101")
end_date = st.sidebar.text_input("回测结束", value="20251226")

# ==========================================
# 5. 主程序
# ==========================================
st.title("🚀 V29.0 终极实战指挥官 (上帝参数版)")
st.caption("已预设【8.1元 + 2.1%换手 + 15天持股】的最优参数组合。您依然可以随时调整。")

tab1, tab2 = st.tabs(["📡 今日实盘扫描", "🧪 历史验证 (参数热调)"])

# --- Tab 1: 实盘扫描 ---
with tab1:
    col_d, col_b = st.columns([3, 1])
    with col_d:
        scan_date_input = st.date_input("选择日期", value=pd.Timestamp.now())
    scan_date_str = scan_date_input.strftime('%Y%m%d')
    
    if col_b.button("开始扫描", type="primary", use_container_width=True):
        if not pro:
            st.error("请先输入 Token")
            st.stop()
            
        with st.spinner("正在获取原子数据..."):
            snap = fetch_daily_atomic_data(scan_date_str, pro)
            champion = run_strategy_logic(snap, cfg_min_price, cfg_max_price, cfg_max_turnover)
            
            if champion is not None:
                st.success(f"🏆 锁定冠军：{champion['name']} ({champion['ts_code']})")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("现价", f"{champion['close']}元")
                c2.metric("Bias", f"{champion['bias']:.4f}")
                c3.metric("换手率", f"{champion['turnover_rate']:.2f}%", delta=f"<{cfg_max_turnover}%")
                c4.metric("获利盘", f"{champion['winner_rate']:.1f}%")
                
                st.info(f"""
                **📝 交易计划 (基于上帝参数)：**
                1.  **买入**：明日开盘买入。
                2.  **止损**：跌破 **{champion['close'] * (1 - stop_loss_decimal):.2f}** (-{cfg_stop_loss}%) 离场。
                3.  **持股**：耐心持有 **{cfg_max_hold}** 天。若不触发止损止盈，到期卖出。
                """)
            else:
                st.warning("今日无符合【上帝参数】的标的。建议空仓，或尝试在侧边栏微调参数。")

# --- Tab 2: 历史验证 ---
with tab2:
    st.info("💡 提示：调整侧边栏参数后，点击下方按钮，**秒出**新参数的回测结果。")
    
    if st.button("🚀 运行极速回测", type="primary", use_container_width=True):
        if not pro:
            st.error("请先输入 Token")
            st.stop()
            
        # 1. 准备数据
        cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
        dates = sorted(cal_df['cal_date'].tolist())
        market_safe_map = get_market_sentiment(start_date, end_date, pro)
        
        active_signals = [] 
        finished_signals = [] 
        
        # 进度条
        progress_bar = st.progress(0)
        
        # 2. 回测循环
        for i, date in enumerate(dates):
            progress_bar.progress((i + 1) / len(dates))
            
            # 显式 GC 防止卡顿
            if i % 20 == 0: gc.collect()
            
            # 获取数据 (缓存)
            snap = fetch_daily_atomic_data(date, pro)
            price_map = {}
            if snap and not snap['daily'].empty:
                price_map = snap['daily'].set_index('ts_code')[['open','high','low','close']].to_dict('index')
            
            is_market_safe = market_safe_map.get(date, False)
            
            # 持仓更新 (使用侧边栏实时参数)
            signals_still_active = []
            curr_dt = pd.to_datetime(date)
            
            for sig in active_signals:
                code = sig['code']
                if curr_dt <= pd.to_datetime(sig['buy_date']):
                    if code in price_map: sig['highest'] = max(sig['highest'], price_map[code]['high'])
                    signals_still_active.append(sig)
                    continue

                if code in price_map:
                    ph, pl, pc = price_map[code]['high'], price_map[code]['low'], price_map[code]['close']
                    if ph > sig['highest']: sig['highest'] = ph
                    
                    cost = sig['buy_price']
                    peak = sig['highest']
                    
                    reason = ""
                    sell_p = pc
                    
                    # === 实时风控参数 ===
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
                        finished_signals.append({'code': code, 'buy_date': sig['buy_date'], 'ret': ret, 'reason': reason})
                    else:
                        signals_still_active.append(sig)
                else:
                    signals_still_active.append(sig)
            active_signals = signals_still_active
            
            # 买入逻辑 (使用侧边栏实时参数)
            if is_market_safe:
                champion = run_strategy_logic(snap, cfg_min_price, cfg_max_price, cfg_max_turnover)
                if champion is not None:
                    code = champion['ts_code']
                    if code in price_map:
                        active_signals.append({
                            'code': code, 'buy_date': date,
                            'buy_price': price_map[code]['open'], 'highest': price_map[code]['open']
                        })
        
        progress_bar.empty()
        
        # 3. 结果展示
        if finished_signals:
            df_res = pd.DataFrame(finished_signals)
            df_res['ret_pct'] = df_res['ret'] * 100
            
            win_rate = (df_res['ret'] > 0).mean() * 100
            avg_ret = df_res['ret'].mean() * 100
            total_ret = df_res['ret'].sum() * 100
            
            st.divider()
            st.markdown("### 📊 回测报告 (当前参数)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("总收益", f"{total_ret:.1f}%", delta="累计复利源")
            c2.metric("胜率", f"{win_rate:.1f}%")
            c3.metric("单笔期望", f"{avg_ret:.2f}%", help="核心指标：必须 > 0.3%")
            c4.metric("交易次数", f"{len(df_res)}")
            
            # 图表
            chart = alt.Chart(df_res).mark_bar().encode(
                x=alt.X("ret_pct", bin=alt.Bin(maxbins=40), title="收益分布"),
                y='count()',
                color=alt.condition(alt.datum.ret_pct > 0, alt.value("#d32f2f"), alt.value("#2e7d32"))
            )
            st.altair_chart(chart, use_container_width=True)
            
            with st.expander("查看详细交易单"):
                st.dataframe(df_res)
        else:
            st.warning("该参数组合下无交易记录。")
