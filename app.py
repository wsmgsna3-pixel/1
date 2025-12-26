import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V17.0 无限火力", layout="wide")
st.title("🧪 V17.0 黄金狙击 (全样本·无限火力版)")
st.markdown("""
### 🛡️ 寻找数学期望 (The Truth)
此版本采用 **"无限子弹"** 模式：
* 忽略资金限制，忽略仓位冲突。
* 只要 Rank 1 出现，**必买**。
* 统计每一笔交易的盈亏，还原策略的最真实面目。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 压力测试参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20240504")
    end_date = st.text_input("结束日期", value="20251226")
    
    st.divider()
    
    # === 自由调整区 ===
    min_price = st.number_input("最低股价限制 (元)", value=8.0, step=0.5, help="请尝试不同价格，寻找盈亏分界线")
    
    st.info("交易规则 (固定)")
    st.text("持股周期: 10天")
    st.text("止损: -5%")
    st.text("止盈: 涨8%回落3%")
    
    # 固定参数，控制变量
    STOP_LOSS_FIXED = -0.0501
    MAX_HOLD_DAYS = 10
    TRAIL_START = 0.08
    TRAIL_DROP = 0.03
    FEE_RATE = 0.0003

run_btn = st.button("🚀 启动全样本回测", type="primary", use_container_width=True)

if run_btn:
    if not my_token:
        st.error("请输入 Token")
        st.stop()
    ts.set_token(my_token)
    status_box = st.empty()
    try:
        pro = ts.pro_api()
    except Exception as e:
        st.error(f"连接失败: {e}")
        st.stop()

    class Config:
        START_DATE = start_date
        END_DATE = end_date
        MIN_PRICE = min_price

    cfg = Config()

    # --- 数据函数 ---
    @st.cache_data(ttl=60)
    def get_market_sentiment(start, end):
        try:
            real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma20'] = df['close'].rolling(20).mean()
            return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
        except: return {}

    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_price_data(date):
        try: return pro.daily(trade_date=date)
        except: return pd.DataFrame()

    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_strategy_data(date):
        try:
            df_daily = pro.daily(trade_date=date)
            if df_daily.empty: return pd.DataFrame()
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            df_cyq = pro.cyq_perf(trade_date=date)
            if df_cyq.empty or 'cost_50pct' not in df_cyq.columns: return pd.DataFrame()
            df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
            df_final = pd.merge(df_merge, df_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
            return df_final
        except: return pd.DataFrame()

    def select_rank_1(df):
        if df.empty: return None
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        condition = (
            (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
            (df['winner_rate'] < 70) &
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5) &
            (df['close'] >= cfg.MIN_PRICE) 
        )
        sorted_df = df[condition].sort_values('bias', ascending=True)
        if sorted_df.empty: return None
        return sorted_df.iloc[0] # 返回 Series

    # --- 无限火力 回测循环 ---
    market_safe_map = get_market_sentiment(cfg.START_DATE, cfg.END_DATE)
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    
    # 核心：这里不再存 positions，而是存 active_signals
    # 结构: {'code':..., 'buy_date':..., 'buy_price':..., 'highest':..., 'days_held':...}
    active_signals = [] 
    finished_signals = [] # 所有的历史战绩

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Scanning: {date} | Active Trades: {len(active_signals)}")

        df_price = fetch_price_data(date)
        df_strat = fetch_strategy_data(date)
        
        price_map_open = {}
        price_map_close = {}
        
        if not df_price.empty:
            df_price = df_price.set_index('ts_code')
            price_map_open = df_price['open'].to_dict()
            price_map_close = df_price['close'].to_dict()
            price_map_high = df_price['high'].to_dict()
            price_map_low = df_price['low'].to_dict()
        
        # 1. 更新所有在手信号 (无论多少个)
        signals_still_active = []
        current_date_obj = pd.to_datetime(date)
        
        for sig in active_signals:
            code = sig['code']
            
            # 如果是买入当天，跳过卖出判断，只更新收盘信息
            if current_date_obj <= pd.to_datetime(sig['buy_date']):
                if code in price_map_close:
                     sig['highest'] = max(sig['highest'], price_map_high.get(code, 0))
                signals_still_active.append(sig)
                continue

            # 卖出逻辑判断
            if code in price_map_close:
                curr_price = price_map_close[code]
                high_today = price_map_high.get(code, curr_price)
                low_today = price_map_low.get(code, curr_price)
                
                # 更新最高价
                if high_today > sig['highest']: sig['highest'] = high_today
                
                cost = sig['buy_price']
                peak = sig['highest']
                peak_ret = (peak - cost) / cost
                drawdown = (peak - curr_price) / peak
                
                reason = ""
                sell_price = curr_price
                pct_chg = 0.0
                
                # 检查卖出条件
                if (low_today - cost) / cost <= STOP_LOSS_FIXED:
                    reason = "止损"
                    sell_price = cost * (1 + STOP_LOSS_FIXED)
                elif peak_ret >= TRAIL_START and drawdown >= TRAIL_DROP:
                    reason = "止盈"
                    # 这里简化处理，按收盘价算，实盘可能更好
                elif (current_date_obj - pd.to_datetime(sig['buy_date'])).days >= MAX_HOLD_DAYS:
                    reason = "超时"
                
                if reason:
                    # 结算
                    ret = (sell_price - cost) / cost - FEE_RATE - FEE_RATE # 双边手续费
                    finished_signals.append({
                        'code': code,
                        'buy_date': sig['buy_date'],
                        'buy_price': cost,
                        'sell_date': date,
                        'sell_price': sell_price,
                        'return': ret,
                        'reason': reason
                    })
                else:
                    signals_still_active.append(sig)
            else:
                # 停牌等情况，继续持有
                signals_still_active.append(sig)
        
        active_signals = signals_still_active

        # 2. 发出新信号 (上帝视角：只要符合就买，不管有没有钱)
        if is_market_safe and not df_strat.empty:
            target_row = select_rank_1(df_strat.reset_index())
            if target_row is not None:
                code = target_row['ts_code']
                # 假设次日开盘买入，这里用当日Open模拟（因为我们是回测日循环，实际上是拿到信号的当日Open买入）
                # 这里的逻辑是：昨日收盘选股 -> 今日开盘买入。
                # 所以我们用当天的 Open 价买入。
                if code in price_map_open:
                    buy_price = price_map_open[code]
                    active_signals.append({
                        'code': code,
                        'buy_date': date,
                        'buy_price': buy_price,
                        'highest': buy_price
                    })

    # --- 结果统计 ---
    status_box.empty()
    st.balloons()
    
    if finished_signals:
        df_res = pd.DataFrame(finished_signals)
        df_res['return_pct'] = df_res['return'] * 100
        
        # 核心指标
        total_trades = len(df_res)
        win_trades = len(df_res[df_res['return'] > 0])
        win_rate = win_trades / total_trades * 100
        avg_ret = df_res['return'].mean() * 100
        total_virtual_ret = df_res['return'].sum() * 100 # 模拟单利累加
        
        st.subheader(f"🧪 全样本回测结果 (最低价 {cfg.MIN_PRICE}元)")
        
        # 1. 核心看板
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("样本总数", f"{total_trades} 笔", help="该周期内所有触发的信号总和")
        c2.metric("真实胜率", f"{win_rate:.1f}%", help="剥离运气后的纯胜率")
        c3.metric("单笔期望收益", f"{avg_ret:.2f}%", delta="关键指标", help="平均每次出手能赚多少点？如果是负数，策略必死。")
        c4.metric("虚拟总收益", f"{total_virtual_ret:.1f}%", help="假设每次投入固定金额的单利总和")
        
        # 2. 收益分布图
        st.subheader("📊 盈亏分布 (真相图)")
        chart = alt.Chart(df_res).mark_bar().encode(
            x=alt.X("return_pct", bin=alt.Bin(maxbins=30), title="单笔收益率 (%)"),
            y='count()',
            color=alt.condition(
                alt.datum.return_pct > 0,
                alt.value("#d32f2f"),  # Red for profit
                alt.value("#2e7d32")   # Green for loss
            )
        )
        st.altair_chart(chart, use_container_width=True)
        
        # 3. 详细数据
        with st.expander("查看每一笔交易详情"):
            st.dataframe(df_res.sort_values('buy_date'))
            
        # 4. 结论判断
        st.divider()
        if avg_ret > 0.5:
            st.success(f"✅ 结论：该价位 ({cfg.MIN_PRICE}元) 策略具有显著的正数学期望！是真正的印钞机。")
        elif avg_ret > 0:
            st.warning(f"⚠️ 结论：该价位 ({cfg.MIN_PRICE}元) 勉强盈利，但抗风险能力较弱。")
        else:
            st.error(f"🛑 结论：该价位 ({cfg.MIN_PRICE}元) 长期期望为负！之前的盈利纯属运气，请立即放弃。")
            
    else:
        st.warning("无交易记录。")
