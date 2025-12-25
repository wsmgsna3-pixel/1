import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V16.0 排名深度分析", layout="wide")
st.title("🔬 V16.0 深度归因：谁才是真正的王牌？")
st.markdown("""
### 🕵️‍♂️ 核心任务：
统计 **Rank 1 vs Rank 2 vs Rank 3** 的表现差异。
* **疑问**：第 1 名是否遥遥领先？
* **目标**：决定实盘是“雨露均沾”还是“独宠一人”。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    my_token = st.text_input("Tushare Token", type="password")
    
    # 建议回测整年以获取充足样本
    start_date = st.text_input("开始日期", value="20250101")
    end_date = st.text_input("结束日期", value="20251225")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    # 保持冠军参数
    max_pos = 3
    max_hold_days = 10
    STOP_LOSS_FIXED = -0.0501
    
    st.success(f"持仓: {max_pos} | 持股: {max_hold_days}天 | 止损: {STOP_LOSS_FIXED*100}%")

run_btn = st.button("🚀 启动分层统计", type="primary", use_container_width=True)

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
        STOP_LOSS = STOP_LOSS_FIXED
        FEE_RATE = 0.0003
        MAX_HOLD_DAYS = max_hold_days
        TRAIL_START = 0.08
        TRAIL_DROP = 0.03

    cfg = Config()

    # --- 数据获取函数 (复用 V15.1) ---
    @st.cache_data(ttl=86400, persist=True)
    def get_market_sentiment(start, end):
        try:
            real_start = (pd.to_datetime(start) - pd.Timedelta(days=60)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma_safe'] = df['close'].rolling(20).mean()
            df['is_safe'] = df['close'] > df['ma_safe']
            return df.set_index('trade_date')['is_safe'].to_dict()
        except:
            return {}

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

    # --- 选股逻辑 (带排名) ---
    def select_stocks_ranked(df):
        if df.empty: return []
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        condition = (
            (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
            (df['winner_rate'] < 70) &
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5)
        )
        selected = df[condition].sort_values('bias', ascending=True).head(5)
        selected = selected.reset_index(drop=True)
        selected['day_rank'] = selected.index + 1 
        return selected

    # --- 回测 ---
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    market_safe_map = get_market_sentiment(cfg.START_DATE, cfg.END_DATE)

    cash = cfg.INITIAL_CASH
    positions = {} 
    trade_log = []
    buy_queue = [] 

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Day: {date} | Analyzing Rank Performance...")

        df_price = fetch_price_data(date)
        df_strat = fetch_strategy_data(date)
        
        price_map_open = {}
        price_map_close = {}
        price_map_high = {}
        price_map_low = {}
        
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
                    # 记录这一单的详细信息，包括当时买入的排名
                    trade_log.append({
                        'rank': pos['rank'], # 关键追踪
                        'profit': profit,
                        'profit_pct': (sell_price - cost) / cost,
                        'win': 1 if profit > 0 else 0
                    })
                    codes_to_sell.append(code)
        
        for c in codes_to_sell: del positions[c]

        # 2. Buy
        if not is_market_safe: buy_queue = [] 
        
        for item in buy_queue:
            code = item['code']
            rank = item['rank']
            if len(positions) >= cfg.MAX_POSITIONS: break
            if code in price_map_open:
                buy_price = price_map_open[code]
                # 这里为了统计公平，还是按1/3仓位买
                slot_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                vol = int(slot_cash / buy_price / 100) * 100
                if vol > 0 and cash >= vol * buy_price:
                    cost = vol * buy_price * (1 + cfg.FEE_RATE)
                    cash -= cost
                    # 存入排名信息
                    positions[code] = {
                        'cost': buy_price, 'vol': vol, 'date': date, 
                        'high_since_buy': buy_price, 'rank': rank
                    }
        buy_queue = []

        # 3. Select
        if is_market_safe and not df_strat.empty and len(positions) < cfg.MAX_POSITIONS:
            target_df = select_stocks_ranked(df_strat.reset_index())
            for i, row in target_df.iterrows():
                if row['ts_code'] not in positions: 
                    buy_queue.append({'code': row['ts_code'], 'rank': row['day_rank']})

    # --- 统计结果 ---
    status_box.empty()
    st.balloons()
    
    if trade_log:
        df_res = pd.DataFrame(trade_log)
        
        st.subheader("📊 排名绩效大比武")
        
        # 分组统计
        stats = df_res.groupby('rank').agg({
            'rank': 'count',                 # 交易次数
            'win': 'sum',                    # 盈利次数
            'profit': 'sum',                 # 总盈利金额
            'profit_pct': 'mean'             # 平均单笔收益率
        }).rename(columns={'rank': '交易次数', 'win': '盈利次数'})
        
        stats['胜率'] = (stats['盈利次数'] / stats['交易次数'] * 100).map('{:.1f}%'.format)
        stats['单笔平均收益'] = (stats['profit_pct'] * 100).map('{:.2f}%'.format)
        stats['总盈利贡献'] = stats['profit'].map('{:,.0f}'.format)
        
        # 重点展示 Rank 1, 2, 3
        st.table(stats.head(3))
        
        # 智能结论
        rank1_win = float(stats.loc[1, '胜率'].strip('%')) if 1 in stats.index else 0
        rank2_win = float(stats.loc[2, '胜率'].strip('%')) if 2 in stats.index else 0
        
        st.info("💡 **AI 策略建议**：")
        if rank1_win > rank2_win + 5:
            st.markdown(f"""
            **第1名遥遥领先！** (胜率 {rank1_win}% vs {rank2_win}%)
            * **建议**：实盘时，**资金向第1名倾斜**（例如：第1名买 50%，2/3名各买 25%）。
            * **甚至**：如果没有第1名，宁可不买第2/3名。
            """)
        else:
            st.markdown(f"""
            **差距不大，雨露均沾。**
            * 第1名和第2名胜率接近，说明 Bias 选股逻辑在整个前三名都有效。
            * **建议**：继续保持 **1/3 等分仓位**，分散风险。
            """)
    else:
        st.warning("没有交易记录，无法分析。")
