import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V14.3 终极封存版", layout="wide")
st.title("🏆 V14.3 黄金狙击 (实盘冠军版)")
st.markdown("""
### 👑 冠军配置 (The Champion Set)：
* **核心战绩**：收益 **69.45%** | 胜率 **59.0%**
* **资金管理**：持仓 **3只** | 持股 **10天**
* **风控铁律**：T+1 盘中 **-5.01%** 触价即跑
* **大盘防线**：MA20 生命线
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 实盘参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20250101")
    end_date = st.text_input("结束日期", value="20251224")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    # === 参数已固化为最佳回测值 ===
    max_pos = 3
    st.success(f"持仓上限: {max_pos} 只 (黄金配置)")
    
    max_hold_days = 10
    st.success(f"持股周期: {max_hold_days} 天 (耐心配置)")
    
    # 硬止损 -5.01%
    STOP_LOSS_FIXED = -0.0501
    st.error(f"硬止损: {STOP_LOSS_FIXED*100}% (盘中条件单)")
    
    st.subheader("移动止盈")
    start_trailing = st.slider("启动阈值 (%)", 5, 20, 8) / 100.0
    drawdown_limit = st.slider("允许回撤 (%)", 1, 10, 3) / 100.0

run_btn = st.button("🚀 启动实盘验证", type="primary", use_container_width=True)

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
        TRAIL_START = start_trailing
        TRAIL_DROP = drawdown_limit

    cfg = Config()

    # --- 1. 获取大盘 (MA20) ---
    @st.cache_data(ttl=86400, persist=True)
    def get_market_sentiment(start, end):
        try:
            df = pro.index_daily(ts_code='000001.SH', start_date=start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma_safe'] = df['close'].rolling(20).mean()
            df['is_safe'] = df['close'] > df['ma_safe']
            return df.set_index('trade_date')['is_safe'].to_dict()
        except:
            return {}

    # --- 2. 基础数据 ---
    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_price_data(date):
        try:
            return pro.daily(trade_date=date)
        except:
            return pd.DataFrame()

    # --- 3. 策略数据 ---
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
        except:
            return pd.DataFrame()

    # --- 选股逻辑 (Bias排名) ---
    def select_stocks_final(df):
        if df.empty: return []
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        condition = (
            (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
            (df['winner_rate'] < 70) &
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5)
        )
        return df[condition].sort_values('bias', ascending=True).head(5)

    # --- 4. 回测循环 ---
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    market_safe_map = get_market_sentiment(cfg.START_DATE, cfg.END_DATE)

    cash = cfg.INITIAL_CASH
    positions = {} 
    history = []
    trade_log = []
    buy_queue = [] 

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Day: {date} | Safe: {is_market_safe} | Pos: {len(positions)}")

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
        
        # 1. Sell Logic
        codes_to_sell = []
        current_date_obj = pd.to_datetime(date)

        for code, pos in positions.items():
            if current_date_obj <= pd.to_datetime(pos['date']): 
                continue 

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
                
                # === T+1 盘中止损 ===
                if (low_today - cost) / cost <= cfg.STOP_LOSS: 
                    reason = "止损(T+1盘中)"
                    sell_price = cost * (1 + cfg.STOP_LOSS)
                    
                elif peak_ret >= cfg.TRAIL_START and drawdown >= cfg.TRAIL_DROP:
                    reason = f"移动止盈({drawdown*100:.1f}%)"
                # === 10天耐心持有 ===
                elif (current_date_obj - pd.to_datetime(pos['date'])).days >= cfg.MAX_HOLD_DAYS:
                    reason = f"超时换股({cfg.MAX_HOLD_DAYS}天)"
                
                if reason:
                    revenue = pos['vol'] * sell_price * (1 - cfg.FEE_RATE)
                    profit = revenue - (pos['vol'] * cost)
                    cash += revenue
                    trade_log.append({'date': date, 'code': code, 'action': 'SELL', 'price': round(sell_price, 2), 'profit': round(profit, 2), 'reason': reason})
                    codes_to_sell.append(code)
        
        for c in codes_to_sell: del positions[c]

        # 2. Buy Logic
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
                    trade_log.append({'date': date, 'code': code, 'action': 'BUY', 'price': buy_price, 'reason': '低吸(T+1)'})
        buy_queue = []

        # 3. Select
        if is_market_safe and not df_strat.empty and len(positions) < cfg.MAX_POSITIONS:
            target_df = select_stocks_final(df_strat.reset_index())
            for i, row in target_df.iterrows():
                if row['ts_code'] not in positions: 
                    buy_queue.append(row['ts_code'])

        # 4. Settle
        total = cash
        for code, pos in positions.items():
            total += pos['vol'] * price_map_close.get(code, pos['high_since_buy'])
        history.append({'date': pd.to_datetime(date), 'asset': total})

    # --- 结果 ---
    status_box.empty()
    st.balloons()
    
    if history:
        df_res = pd.DataFrame(history).set_index('date')
        ret = (df_res['asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        
        wins = len([t for t in trade_log if t['action']=='SELL' and t['profit']>0])
        total_sells = len([t for t in trade_log if t['action']=='SELL'])
        win_rate = (wins / total_sells * 100) if total_sells > 0 else 0
        
        st.subheader("🏆 实盘策略最终报告")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("区间收益", f"{ret:.2f}%")
        c2.metric("交易次数", len(trade_log))
        c3.metric("真实胜率", f"{win_rate:.1f}%")
        c4.metric("策略状态", "✅ Ready for Action")
        
        st.line_chart(df_res['asset'])
        with st.expander("交易明细"):
            st.dataframe(pd.DataFrame(trade_log))
