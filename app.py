import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt
import time
import random

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="V27.0 炼金术士", layout="wide")

# ==========================================
# 2. 侧边栏：参数边界设定
# ==========================================
st.sidebar.header("⚗️ 炼金实验室")

my_token = st.sidebar.text_input("Tushare Token", type="password")

st.sidebar.divider()
st.sidebar.subheader("1. 设定参数尝试范围")
st.sidebar.info("系统将在您设定的范围内随机抽取参数进行演练。")

# 价格范围
c1, c2 = st.sidebar.columns(2)
opt_min_price_low = c1.number_input("最低价下限", 3.0)
opt_min_price_high = c2.number_input("最低价上限", 15.0, value=11.0)

# 换手率范围
opt_turnover_low = st.sidebar.number_input("换手率下限", 0.5, value=1.0)
opt_turnover_high = st.sidebar.number_input("换手率上限", 10.0, value=5.0)

# 止损范围
opt_stop_low = st.sidebar.slider("止损范围 (%)", 1, 15, (3, 8))

# 持股天数
opt_hold_low = st.sidebar.slider("持股天数范围", 1, 20, (3, 15))

# 止盈参数 (固定或小范围微调)
opt_trail_start = 0.08 # 暂时固定，减少复杂度，也可以放开
opt_trail_drop = 0.03

st.sidebar.divider()
st.sidebar.subheader("2. 训练强度")
sim_rounds = st.sidebar.slider("模拟次数 (轮)", 50, 500, 100, help="次数越多越精准，但耗时越长。建议先跑100次看看。")

# 回测区间
start_date = st.sidebar.text_input("开始日期", value="20250101")
end_date = st.sidebar.text_input("结束日期", value="20251226")

# ==========================================
# 3. 核心功能
# ==========================================
st.title("⚗️ V27.0 炼金术士 (多参数蒙特卡洛优化)")
st.markdown("""
### 🚀 寻找最优解
不要手动一个一个试了。让算法帮您在 **7维参数空间** 中寻找收益最高的组合。
""")

if not my_token:
    st.warning("👈 请输入 Token")
    st.stop()

ts.set_token(my_token)
try:
    pro = ts.pro_api()
except:
    st.error("Token 无效")
    st.stop()

# --- 原子数据层 (缓存) ---
@st.cache_data(ttl=86400 * 7) 
def fetch_daily_atomic_data(date):
    try:
        df_daily = pro.daily(trade_date=date)
        df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
        df_cyq = pro.cyq_perf(trade_date=date)
        if df_cyq.empty:
             for i in range(1, 4):
                 prev = (pd.to_datetime(date) - pd.Timedelta(days=i)).strftime('%Y%m%d')
                 df_cyq = pro.cyq_perf(trade_date=prev)
                 if not df_cyq.empty: break
        return {'daily': df_daily, 'basic': df_basic, 'cyq': df_cyq}
    except: return {}

@st.cache_data(ttl=86400)
def get_market_sentiment(start, end):
    try:
        real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
        df = df.sort_values('trade_date', ascending=True)
        df['ma20'] = df['close'].rolling(20).mean()
        return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
    except: return {}

# --- 逻辑层 (极速运算) ---
def run_strategy_once(snapshot, p_min, p_max, to_max):
    if not snapshot: return None
    d1, d2, d4 = snapshot.get('daily'), snapshot.get('basic'), snapshot.get('cyq')
    if d1 is None or d1.empty or d2 is None or d2.empty or d4 is None or d4.empty or 'cost_50pct' not in d4.columns: return None
    
    m1 = pd.merge(d1, d2, on='ts_code')
    df = pd.merge(m1, d4[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code')
    df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
    
    # 动态参数筛选
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
# 4. 蒙特卡洛引擎
# ==========================================
if st.button("🔥 开始蒙特卡洛训练", type="primary"):
    
    # 1. 预加载数据 (只做一次)
    cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    market_safe_map = get_market_sentiment(start_date, end_date)
    
    cache_snapshots = {}
    preload_bar = st.progress(0, text="预加载数据中 (IO操作)...")
    for i, date in enumerate(dates):
        preload_bar.progress((i+1)/len(dates))
        cache_snapshots[date] = fetch_daily_atomic_data(date)
    preload_bar.empty()
    
    # 2. 生成随机参数组
    params_pool = []
    for _ in range(sim_rounds):
        params_pool.append({
            'min_price': round(random.uniform(opt_min_price_low, opt_min_price_high), 1),
            'max_turnover': round(random.uniform(opt_turnover_low, opt_turnover_high), 1),
            'stop_loss': round(random.uniform(opt_stop_low[0], opt_stop_low[1]), 1) / 100.0,
            'max_hold': random.randint(opt_hold_low[0], opt_hold_low[1])
        })
    
    results = []
    
    # 3. 疯狂训练
    train_bar = st.progress(0, text="AI 正在疯狂演练...")
    start_time = time.time()
    
    for idx, params in enumerate(params_pool):
        train_bar.progress((idx+1)/sim_rounds, text=f"正在演练第 {idx+1}/{sim_rounds} 组参数...")
        
        active_signals = [] 
        finished_returns = []
        
        # 极速回测循环
        for date in dates:
            snap = cache_snapshots.get(date)
            price_map = {}
            if snap and not snap['daily'].empty:
                 # 简单构建 price_map，这里为了速度只取需要的
                 # 注意：为了极速，这里可以进一步优化，但先保持逻辑清晰
                 price_map = snap['daily'].set_index('ts_code')[['open','high','low','close']].to_dict('index')
            
            is_market_safe = market_safe_map.get(date, False)
            
            # 持仓
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
                    
                    # 使用当前 params
                    if (pl - cost) / cost <= -params['stop_loss']:
                        reason = "止损"
                        sell_p = cost * (1 - params['stop_loss'])
                    elif (peak - cost)/cost >= opt_trail_start and (peak - pc)/peak >= opt_trail_drop:
                        reason = "止盈"
                        sell_p = peak * (1 - opt_trail_drop)
                    elif (curr_dt - pd.to_datetime(sig['buy_date'])).days >= params['max_hold']:
                        reason = "超时"
                    
                    if reason:
                        ret = (sell_p - cost) / cost - 0.0006
                        finished_returns.append(ret)
                    else:
                        signals_still_active.append(sig)
                else:
                    signals_still_active.append(sig)
            active_signals = signals_still_active
            
            # 买入
            if is_market_safe:
                champion = run_strategy_once(snap, params['min_price'], 20.0, params['max_turnover'])
                if champion is not None:
                    code = champion['ts_code']
                    if code in price_map:
                        active_signals.append({
                            'code': code, 'buy_date': date,
                            'buy_price': price_map[code]['open'], 'highest': price_map[code]['open']
                        })
        
        # 记录本轮结果
        if finished_returns:
            total_ret = sum(finished_returns) * 100
            win_rate = len([r for r in finished_returns if r > 0]) / len(finished_returns) * 100
            results.append({
                '最低价': params['min_price'],
                '最大换手': params['max_turnover'],
                '止损(%)': round(params['stop_loss']*100, 1),
                '持股天数': params['max_hold'],
                '总收益%': round(total_ret, 1),
                '胜率%': round(win_rate, 1),
                '交易次数': len(finished_returns)
            })
    
    train_bar.empty()
    st.success(f"演练完成！耗时 {time.time()-start_time:.1f} 秒")
    
    if results:
        df_res = pd.DataFrame(results)
        
        # 1. 敏感度分析 (Correlation)
        st.subheader("📊 敏感度分析：哪个参数最重要？")
        corr = df_res[['最低价', '最大换手', '止损(%)', '持股天数', '总收益%']].corr()['总收益%'].drop('总收益%')
        st.bar_chart(corr)
        st.caption("💡 柱子越高（或越低），说明该参数对收益率的影响越大！")
        
        # 2. 散点图 (寻找最优区域)
        st.subheader("🎯 参数分布图 (颜色越红收益越高)")
        chart = alt.Chart(df_res).mark_circle(size=60).encode(
            x='最低价',
            y='最大换手',
            color=alt.Color('总收益%', scale=alt.Scale(scheme='turbo')),
            tooltip=['最低价', '最大换手', '止损(%)', '持股天数', '总收益%', '胜率%']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        
        # 3. TOP 10 榜单
        st.subheader("🏆 上帝参数 TOP 10")
        top_10 = df_res.sort_values('总收益%', ascending=False).head(10)
        st.dataframe(top_10)
    else:
        st.warning("无有效回测数据")
