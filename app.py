import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V16.4 持仓时间分析", layout="wide")
st.title("⏱️ V16.4 黄金狙击 (持仓时间透视版)")
st.markdown("""
### 🧠 核心问题：我们到底拿了多久？
此版本将重点分析 **“盈亏与时间”** 的关系：
1.  **亏损股** 是不是跑得很快？(截断亏损)
2.  **盈利股** 是不是拿得更久？(让利润奔跑)
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 策略参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20250101")
    end_date = st.text_input("结束日期", value="20251225")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    max_pos = st.slider("持仓上限", 1, 5, 3)
    max_hold_days = st.slider("最大持股天数", 3, 20, 10)
    
    st.info("硬止损: -5.01% | 移动止盈: 8%回撤3%")

run_btn = st.button("🚀 启动时间分析", type="primary", use_container_width=True)

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
        INITIAL_CASH = initial_cash
        MAX_POSITIONS = max_pos
        STOP_LOSS = -0.0501
        FEE_RATE = 0.0003
        MAX_HOLD_DAYS = max_hold_days
        TRAIL_START = 0.08
        TRAIL_DROP = 0.03

    cfg = Config()

    # --- 数据函数 ---
    @st.cache_data(ttl=60)
    def get_market_sentiment(start, end):
        try:
            real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma20'] = df['close'].rolling(20).mean()
            df['is_safe'] = df['close'] > df['ma20']
            return df.set_index('trade_date')['is_safe'].to_dict()
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

    def select_stocks(df):
        if df.empty: return []
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        condition = (
            (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
            (df['winner_rate'] < 70) &
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5)
        )
        return df[condition].sort_values('bias', ascending=True).head(5)

    # --- 回测循环 ---
    market_safe_map = get_market_sentiment(cfg.START_DATE, cfg.END_DATE)
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    
    cash = cfg.INITIAL_CASH
    positions = {} 
    trade_log = []
    buy_queue = [] 
    
    # 增加一个列表专门记录持股时间
    holding_stats = []

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Analyzing Time: {date}")

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
        
        # 1. Sell
        codes_to_sell = []
        current_date_obj = pd.to_datetime(date)
        
        for code, pos in positions.items():
            if current_date_obj <= pd.to_datetime(pos['date']): continue 
            
            if code in price_map_close:
                curr_price = price_map_close[code]
                high_today = price_map_high.get(code, curr_price)
                low_today = price_map_low.get(code, curr_price)
                if high_today > pos['high_since_buy']: pos['high_since_buy'] = high_today
                
                cost = pos['cost']
                peak = pos['high_since_buy']
                peak_ret = (peak - cost) / cost
                drawdown = (peak - curr_price) / peak
                
                reason = ""
                sell_price = curr_price
                
                if (low_today - cost) / cost <= cfg.STOP_LOSS: 
                    reason = "止损"
                    sell_price = cost * (1 + cfg.STOP_LOSS)
                elif peak_ret >= cfg.TRAIL_START and drawdown >= cfg.TRAIL_DROP:
                    reason = "止盈"
                elif (current_date_obj - pd.to_datetime(pos['date'])).days >= cfg.MAX_HOLD_DAYS:
                    reason = "超时"
                
                if reason:
                    revenue = pos['vol'] * sell_price * (1 - cfg.FEE_RATE)
                    profit = revenue - (pos['vol'] * cost)
                    cash += revenue
                    
                    # === 计算持股天数 ===
                    buy_date = pd.to_datetime(pos['date'])
                    sell_date = current_date_obj
                    days_held = (sell_date - buy_date).days
                    
                    trade_type = "盈利" if profit > 0 else "亏损"
                    
                    trade_log.append({
                        '代码': code, '方向': '卖出', '盈亏': profit, 
                        '持股天数': days_held, '类型': trade_type
                    })
                    codes_to_sell.append(code)
        
        for c in codes_to_sell: del positions[c]

        # 2. Buy
        if not is_market_safe: buy_queue = []
        for code in buy_queue:
            if len(positions) >= cfg.MAX_POSITIONS: break
            if code in price_map_open:
                buy_price = price_map_open[code]
                slot_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                vol = int(slot_cash / buy_price / 100) * 100
                if vol > 0 and cash >= vol * buy_price:
                    cost = vol * buy_price * (1 + cfg.FEE_RATE)
                    cash -= cost
                    positions[code] = {'cost': buy_price, 'vol': vol, 'date': date, 'high_since_buy': buy_price}
        buy_queue = []

        # 3. Select (用混合模式，样本更多)
        if is_market_safe and not df_strat.empty and len(positions) < cfg.MAX_POSITIONS:
            target_df = select_stocks(df_strat.reset_index())
            for i, row in target_df.iterrows():
                if row['ts_code'] not in positions: buy_queue.append(row['ts_code'])

    # --- 结果展示 ---
    status_box.empty()
    st.balloons()
    
    st.header("⏱️ 持仓时间透视")
    
    if trade_log:
        df_log = pd.DataFrame(trade_log)
        
        # 分组计算平均值
        stats = df_log.groupby('类型')['持股天数'].mean().reset_index()
        stats['持股天数'] = stats['持股天数'].round(1)
        
        # 1. 核心指标卡
        c1, c2 = st.columns(2)
        
        win_days = stats[stats['类型']=='盈利']['持股天数'].values
        loss_days = stats[stats['类型']=='亏损']['持股天数'].values
        
        val_win = win_days[0] if len(win_days)>0 else 0
        val_loss = loss_days[0] if len(loss_days)>0 else 0
        
        c1.metric("🔴 盈利单平均持仓", f"{val_win} 天", help="好股票我们拿得久")
        c2.metric("🟢 亏损单平均持仓", f"{val_loss} 天", help="坏股票我们跑得快")
        
        # 2. 图表可视化
        chart = alt.Chart(stats).mark_bar().encode(
            x='类型',
            y='持股天数',
            color=alt.Color('类型', scale=alt.Scale(domain=['盈利', '亏损'], range=['#e53935', '#43a047'])),
            tooltip=['类型', '持股天数']
        ).properties(title="盈亏单持股时间对比")
        
        st.altair_chart(chart, use_container_width=True)
        
        # 3. 详细分布表格
        st.subheader("详细分布数据")
        st.dataframe(stats)
        
        st.info(f"""
        **💡 数据解读：**
        * 如果 **盈利天数 >> 亏损天数**（例如 8天 vs 2天）：说明策略非常健康，做到了“截断亏损，让利润奔跑”。
        * 如果 **亏损天数** 也很长：说明止损太慢，正在扛单（这是大忌，但本策略有-5%硬止损，通常不会发生）。
        """)
    else:
        st.warning("暂无已完成的交易记录。")
