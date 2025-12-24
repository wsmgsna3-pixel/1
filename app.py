import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V7.1修复版", layout="wide")
st.title("🛡️ V7.1 双引擎实战 (修复持仓锁死BUG)")
st.markdown("""
### 修复说明：
* **分离数据流**：将“持仓监控价格”与“选股策略数据”解耦。
* **效果**：即使筹码数据缺失，只要股票正常交易，就能正常止盈止损。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20250101")
    end_date = st.text_input("结束日期", value="20251224")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 100) * 10000
    
    st.divider()
    max_pos = st.slider("最大持仓数", 3, 10, 5) 
    stop_loss = st.slider("止损阈值", -15.0, -3.0, -8.0) / 100.0
    take_profit = st.slider("止盈阈值", 5.0, 50.0, 15.0) / 100.0

run_btn = st.button("🚀 启动 V7.1 修复版", type="primary", use_container_width=True)

# ==========================================
# 核心逻辑
# ==========================================
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
        STOP_LOSS = stop_loss
        TAKE_PROFIT = take_profit
        FEE_RATE = 0.0003
        MAX_HOLD_DAYS = 20 

    cfg = Config()

    # --- 1. 获取大盘 ---
    @st.cache_data(ttl=86400, persist=True)
    def get_index_data(start, end):
        try:
            df = pro.index_daily(ts_code='000001.SH', start_date=start, end_date=end)
            df = df.sort_values('trade_date')
            df['ma20'] = df['close'].rolling(20).mean()
            return df.set_index('trade_date')[['close', 'ma20']].to_dict('index')
        except:
            return {}

    # --- 2. 基础行情 (用于持仓监控，不依赖筹码) ---
    # 这个函数只取 daily 数据，保证只要股票在交易就能查到价格
    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_price_data(date):
        try:
            df = pro.daily(trade_date=date)
            return df
        except:
            return pd.DataFrame()

    # --- 3. 策略数据 (用于选股，依赖筹码) ---
    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_strategy_data(date):
        try:
            # 这里必须重新调一次daily，虽然有点冗余，但为了确保 merging 的基准
            df_daily = pro.daily(trade_date=date)
            if df_daily.empty: return pd.DataFrame()

            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            df_cyq = pro.cyq_perf(trade_date=date)
            
            # 严格合并
            if df_cyq.empty or 'cost_50pct' not in df_cyq.columns:
                return pd.DataFrame()

            df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
            df_final = pd.merge(df_merge, df_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
            return df_final
        except:
            return pd.DataFrame()

    # --- 4. 选股逻辑 ---
    def select_stocks_dual(df, index_status):
        if df.empty: return []
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        
        cond_support = ((df['bias'] > -0.02) & (df['bias'] < 0.1) & (df['winner_rate'] < 60))
        cond_breakout = pd.Series([False] * len(df), index=df.index)
        
        market_safe = False
        if index_status and index_status['close'] > index_status['ma20']:
            market_safe = True
            cond_breakout = ((df['bias'] > 0.05) & (df['bias'] < 0.25) & (df['winner_rate'] > 80) & (df['turnover_rate'] > 3.0))

        df['strategy'] = ''
        df.loc[cond_support, 'strategy'] = '低吸'
        if market_safe:
            df.loc[cond_breakout, 'strategy'] = '突破'
            
        selected = df[df['strategy'] != ''].copy()
        selected = selected[(selected['circ_mv'] > 300000) & (selected['pe_ttm'] > 0) & (selected['pe_ttm'] < 80)]
        selected['rank_score'] = selected.apply(lambda x: 100 if x['strategy']=='突破' else (1 - abs(x['bias'])), axis=1)
        
        return selected.sort_values('rank_score', ascending=False).head(3)

    # --- 5. 回测循环 ---
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = cal_df['cal_date'].tolist()
    index_data = get_index_data(cfg.START_DATE, cfg.END_DATE)

    cash = cfg.INITIAL_CASH
    positions = {}
    history = []
    trade_log = []
    buy_queue = {} 

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        
        # --- 获取两套数据 ---
        # 1. 宽口径：只看价格 (用于监控)
        df_price = fetch_price_data(date)
        # 2. 窄口径：看策略指标 (用于选股)
        df_strategy = fetch_strategy_data(date)
        
        price_map_open = {}
        price_map_close = {}
        price_map_high = {}
        price_map_low = {}

        # 构建价格地图 (使用宽口径数据)
        if not df_price.empty:
            df_price = df_price.set_index('ts_code')
            price_map_open = df_price['open'].to_dict()
            price_map_close = df_price['close'].to_dict()
            price_map_high = df_price['high'].to_dict()
            price_map_low = df_price['low'].to_dict()

        status_box.text(f"Processing: {date} | Positions: {len(positions)}")

        # --- A. T+1 买入 ---
        for code in list(buy_queue.keys()):
            if len(positions) >= cfg.MAX_POSITIONS: 
                buy_queue.pop(code)
                continue
            if code in price_map_open:
                buy_price = price_map_open[code]
                strat = buy_queue[code]['strategy']
                slot_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                vol = int(slot_cash / buy_price / 100) * 100
                if vol > 0 and cash >= vol * buy_price:
                    cost = vol * buy_price * (1 + cfg.FEE_RATE)
                    cash -= cost
                    positions[code] = {'cost': buy_price, 'vol': vol, 'date': date, 'last_price': buy_price, 'strategy': strat}
                    trade_log.append({'date': date, 'code': code, 'action': 'BUY', 'price': buy_price, 'reason': f'{strat} (T+1开盘)'})
                buy_queue.pop(code)

        # --- B. 卖出监控 (现在只要有行情就能卖) ---
        codes_to_sell = []
        for code, pos in positions.items():
            if code in price_map_close: # 这里的 map 包含了所有正常交易股票
                pos['last_price'] = price_map_close[code]
                cost = pos['cost']
                high_p = price_map_high.get(code, pos['last_price'])
                low_p = price_map_low.get(code, pos['last_price'])
                
                reason = ""
                sell_price = pos['last_price']
                
                if (low_p - cost) / cost <= cfg.STOP_LOSS:
                    reason = "止损"
                    sell_price = cost * (1 + cfg.STOP_LOSS)
                elif (high_p - cost) / cost >= cfg.TAKE_PROFIT:
                    reason = "止盈"
                    sell_price = cost * (1 + cfg.TAKE_PROFIT)
                else:
                    hold_days = (pd.to_datetime(date) - pd.to_datetime(pos['date'])).days
                    limit_days = cfg.MAX_HOLD_DAYS if pos.get('strategy') == '突破' else 10
                    if hold_days >= limit_days:
                        reason = f"超时({pos.get('strategy')})"
                
                if reason:
                    revenue = pos['vol'] * sell_price * (1 - cfg.FEE_RATE)
                    profit = revenue - (pos['vol'] * cost)
                    cash += revenue
                    trade_log.append({'date': date, 'code': code, 'action': 'SELL', 'price': round(sell_price, 2), 'profit': round(profit, 2), 'reason': reason})
                    codes_to_sell.append(code)
        for c in codes_to_sell: del positions[c]

        # --- C. 每日选股 ---
        # 选股依然需要策略数据 (df_strategy)
        if not df_strategy.empty and len(positions) + len(buy_queue) < cfg.MAX_POSITIONS:
            idx_stat = index_data.get(date, None)
            targets = select_stocks_dual(df_strategy.reset_index(), idx_stat)
            for i, row in targets.iterrows():
                code = row['ts_code']
                if code not in positions and code not in buy_queue:
                    buy_queue[code] = {'date': date, 'strategy': row['strategy']}

        # --- D. 结算 ---
        total = cash
        for code, pos in positions.items():
            total += pos['vol'] * pos.get('last_price', pos['cost'])
        history.append({'date': pd.to_datetime(date), 'asset': total})

    # --- 结果 ---
    status_box.empty()
    st.balloons()
    
    if history:
        df_res = pd.DataFrame(history).set_index('date')
        ret = (df_res['asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        
        st.subheader("📊 V7.1 修复版报告")
        c1, c2, c3 = st.columns(3)
        c1.metric("区间收益", f"{ret:.2f}%")
        c2.metric("交易次数", len(trade_log))
        c3.metric("最终持仓", len(positions))
        
        st.line_chart(df_res['asset'])
        with st.expander("交易明细", expanded=True):
            st.dataframe(pd.DataFrame(trade_log))
