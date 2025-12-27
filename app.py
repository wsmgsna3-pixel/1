import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
import altair as alt
import time
import random
import gc  # 引入垃圾回收机制

# ==========================================
# 1. 页面配置 (严格遵循 V30 结构)
# ==========================================
st.set_page_config(page_title="V28.0 炼金术士(稳定版)", layout="wide")

# ==========================================
# 2. 全局缓存 (复刻 V30 架构)
# ==========================================

# 使用 cache_resource 缓存 API 连接，避免重复连接导致卡顿
@st.cache_resource
def get_pro_api(token):
    if not token: return None
    ts.set_token(token)
    return ts.pro_api()

# 原子化数据缓存：只认日期，不认参数
@st.cache_data(ttl=86400 * 7)
def fetch_daily_atomic_data(date, _pro):
    """
    获取单日全量数据。
    注意：_pro 参数前加下划线，告诉 Streamlit 不要对 API 对象进行哈希（防止卡死）。
    """
    if _pro is None: return {}
    try:
        # 1. 基础行情
        df_daily = _pro.daily(trade_date=date)
        
        # 2. 每日指标
        df_basic = _pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
        
        # 3. 筹码数据 (Rank 1 核心)
        df_cyq = _pro.cyq_perf(trade_date=date)
        if df_cyq.empty: 
             # 简单回溯3天
             for i in range(1, 4):
                 prev = (pd.to_datetime(date) - pd.Timedelta(days=i)).strftime('%Y%m%d')
                 df_cyq = _pro.cyq_perf(trade_date=prev)
                 if not df_cyq.empty: break
        
        return {'daily': df_daily, 'basic': df_basic, 'cyq': df_cyq}
    except Exception:
        return {}

# 大盘风控数据
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
# 3. 纯内存逻辑 (毫秒级运算)
# ==========================================
def run_strategy_logic(snapshot, params):
    """
    params: 字典，包含所有 7 个动态参数
    """
    if not snapshot: return None
    d1, d2, d4 = snapshot.get('daily'), snapshot.get('basic'), snapshot.get('cyq')
    
    if d1 is None or d1.empty or d2 is None or d2.empty or d4 is None or d4.empty: return None
    if 'cost_50pct' not in d4.columns: return None
    
    # 内存合并
    m1 = pd.merge(d1, d2, on='ts_code')
    df = pd.merge(m1, d4[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code')
    
    df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
    
    # === 7 参数过滤 ===
    condition = (
        (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
        (df['winner_rate'] < 70) &
        (df['circ_mv'] > 300000) &  
        (df['close'] >= params['min_price']) &      # 1. 最低价
        (df['close'] <= params['max_price']) &      # 2. 最高价
        (df['turnover_rate'] < params['turnover'])  # 3. 换手率
    )
    
    sorted_df = df[condition].sort_values('bias', ascending=True)
    if sorted_df.empty: return None
    return sorted_df.iloc[0]

# ==========================================
# 4. 侧边栏：炼金实验室
# ==========================================
st.sidebar.header("⚗️ 炼金参数配置")

token_input = st.sidebar.text_input("Tushare Token", type="password")
pro = get_pro_api(token_input)

st.sidebar.divider()
st.sidebar.info("💡 设定 7 个参数的尝试范围，系统将自动寻找最优解。")

# --- 7 参数 范围设定 ---
# 1. 最低价
p_min_range = st.sidebar.slider("1. 最低价范围", 4.0, 15.0, (5.0, 11.0))
# 2. 最高价 (通常 Rank 1 不会太高，固定一下上限即可)
p_max_fixed = st.sidebar.number_input("2. 最高价上限 (固定)", value=20.0)
# 3. 换手率
to_range = st.sidebar.slider("3. 换手率范围", 1.0, 10.0, (2.0, 5.0))
# 4. 止损
sl_range = st.sidebar.slider("4. 止损范围 (%)", 3.0, 15.0, (5.0, 10.0))
# 5. 止盈启动
tp_start_range = st.sidebar.slider("5. 止盈启动范围 (%)", 5.0, 15.0, (6.0, 10.0))
# 6. 回落卖出
tp_drop_range = st.sidebar.slider("6. 回落卖出范围 (%)", 1.0, 5.0, (2.0, 4.0))
# 7. 持股天数
hold_range = st.sidebar.slider("7. 持股天数范围", 5, 20, (8, 15))

st.sidebar.divider()
sim_rounds = st.sidebar.number_input("🤖 演练次数 (建议 50-100)", value=50, step=10)
start_date = st.sidebar.text_input("开始日期", value="20250101")
end_date = st.sidebar.text_input("结束日期", value="20251226")

# ==========================================
# 5. 主程序：蒙特卡洛引擎
# ==========================================
st.title("⚗️ V28.0 炼金术士 (稳定架构版)")
st.caption("针对您提出的 7 参数优化难题，使用随机演练算法寻找最优解。")

if st.button("🔥 开始寻找上帝参数", type="primary", use_container_width=True):
    
    if not pro:
        st.error("请先输入 Token")
        st.stop()

    # 1. 准备日期和风控数据
    with st.spinner("正在初始化数据..."):
        cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
        dates = sorted(cal_df['cal_date'].tolist())
        market_safe_map = get_market_sentiment(start_date, end_date, pro)
        
        # 预加载数据到内存 (加速后续计算)
        # 注意：这里我们做一个简单的内存缓存，避免重复调用 atomic 函数
        memory_snapshots = {}
        preload_bar = st.progress(0, "预加载数据 (IO)...")
        for i, date in enumerate(dates):
            memory_snapshots[date] = fetch_daily_atomic_data(date, pro)
            preload_bar.progress((i+1)/len(dates))
        preload_bar.empty()

    # 2. 生成随机参数池
    # 这是解决“不知道哪个参数好”的最佳办法：生成一堆随机组合去跑！
    param_pool = []
    for _ in range(sim_rounds):
        param_pool.append({
            'min_price': round(random.uniform(p_min_range[0], p_min_range[1]), 1),
            'max_price': p_max_fixed,
            'turnover': round(random.uniform(to_range[0], to_range[1]), 1),
            'stop_loss': round(random.uniform(sl_range[0], sl_range[1]), 1) / 100.0,
            'trail_start': round(random.uniform(tp_start_range[0], tp_start_range[1]), 1) / 100.0,
            'trail_drop': round(random.uniform(tp_drop_range[0], tp_drop_range[1]), 1) / 100.0,
            'max_hold': random.randint(hold_range[0], hold_range[1])
        })
    
    # 3. 疯狂演练
    results = []
    main_bar = st.progress(0, "AI 正在演练...")
    start_time = time.time()
    
    for idx, params in enumerate(param_pool):
        # 显式回收内存，防止卡死
        if idx % 10 == 0: gc.collect()
        
        main_bar.progress((idx+1)/sim_rounds, f"正在演练第 {idx+1}/{sim_rounds} 组策略...")
        
        active_signals = [] 
        returns = []
        
        for date in dates:
            snap = memory_snapshots.get(date)
            # 构建简易 Price Map
            price_map = {}
            if snap and not snap['daily'].empty:
                # 只取需要的列，节省内存
                price_map = snap['daily'].set_index('ts_code')[['open','high','low','close']].to_dict('index')
            
            is_market_safe = market_safe_map.get(date, False)
            
            # --- 持仓 ---
            signals_still_active = []
            curr_dt = pd.to_datetime(date)
            
            for sig in active_signals:
                code = sig['code']
                # 还没到买入日
                if curr_dt <= pd.to_datetime(sig['buy_date']):
                    if code in price_map: sig['highest'] = max(sig['highest'], price_map[code]['high'])
                    signals_still_active.append(sig)
                    continue

                if code in price_map:
                    ph = price_map[code]['high']
                    pl = price_map[code]['low']
                    pc = price_map[code]['close']
                    
                    if ph > sig['highest']: sig['highest'] = ph
                    
                    cost = sig['buy_price']
                    peak = sig['highest']
                    
                    reason = ""
                    sell_p = pc
                    
                    # 使用当前轮次的随机参数
                    if (pl - cost) / cost <= -params['stop_loss']:
                        reason = "止损"
                        sell_p = cost * (1 - params['stop_loss'])
                    elif (peak - cost)/cost >= params['trail_start'] and (peak - pc)/peak >= params['trail_drop']:
                        reason = "止盈"
                        sell_p = peak * (1 - params['trail_drop'])
                    elif (curr_dt - pd.to_datetime(sig['buy_date'])).days >= params['max_hold']:
                        reason = "超时"
                    
                    if reason:
                        ret = (sell_p - cost) / cost - 0.0006
                        returns.append(ret)
                    else:
                        signals_still_active.append(sig)
                else:
                    signals_still_active.append(sig)
            active_signals = signals_still_active
            
            # --- 买入 ---
            if is_market_safe:
                champion = run_strategy_logic(snap, params)
                if champion is not None:
                    code = champion['ts_code']
                    if code in price_map:
                        active_signals.append({
                            'code': code, 'buy_date': date,
                            'buy_price': price_map[code]['open'], 'highest': price_map[code]['open']
                        })
        
        # 统计本轮结果
        if returns:
            tot = sum(returns) * 100
            win = len([r for r in returns if r > 0]) / len(returns) * 100
            exp = np.mean(returns) * 100
            
            # 记录结果 (把参数和结果拍平存进去)
            record = params.copy()
            # 把百分比还原成阅读友好的数值
            record['stop_loss'] *= 100
            record['trail_start'] *= 100
            record['trail_drop'] *= 100
            record['total_ret'] = tot
            record['win_rate'] = win
            record['expectancy'] = exp
            record['trades'] = len(returns)
            results.append(record)
            
    main_bar.empty()
    st.success(f"演练完成！耗时 {time.time()-start_time:.1f}s")
    
    if results:
        df_res = pd.DataFrame(results)
        
        st.divider()
        st.subheader("🏆 参数排行榜 (按总收益)")
        
        # 格式化一下显示
        show_cols = ['total_ret', 'win_rate', 'expectancy', 'trades', 
                     'min_price', 'turnover', 'stop_loss', 'max_hold']
        
        df_show = df_res.sort_values('total_ret', ascending=False)[show_cols].head(10)
        st.dataframe(df_show.style.format("{:.2f}").background_gradient(subset=['total_ret'], cmap='Reds'))
        
        st.markdown("""
        **🔎 观察要点：**
        1. 看 **TOP 1** 的参数组合，这就是当前的“上帝参数”。
        2. 看 **前10名** 的参数有没有共性？(比如最低价是不是都在 5-6元？)
        """)
        
        # 简单的参数相关性图
        st.subheader("📊 哪个参数最重要？")
        corr = df_res.corr()['total_ret'].drop(['total_ret', 'win_rate', 'expectancy', 'trades'])
        st.bar_chart(corr)
        
    else:
        st.warning("无回测结果")
