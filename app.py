import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V11 激进游击", layout="wide")
st.title("⚡ V11.0 激进游击战 (高频高周转)")
st.markdown("""
### 🚀 提速策略：
1.  **降低门槛**：大盘风控降为 **5日线**，选股区间放宽至 **30%**。
2.  **极速轮动**：持股上限仅 **5天**，不涨就换股，拒绝死拿。
3.  **微利快跑**：赚 **5%** 就开启止盈监控，积少成多。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 激进参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20250101")
    end_date = st.text_input("结束日期", value="20251224")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    # 既然资金少，就集中火力
    max_pos = st.slider("持仓只数", 1, 3, 2) 
    
    # 止盈止损都要快
    stop_loss = st.slider("硬止损", -10.0, -3.0, -5.0) / 100.0
    
    st.subheader("超短止盈")
    start_trailing = st.slider("启动阈值", 3, 10, 5) / 100.0 # 5%就准备跑
    drawdown_limit = st.slider("允许回撤", 1, 5, 2) / 100.0

run_btn = st.button("🔥 启动 V11 激进版", type="primary", use_container_width=True)

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
        FEE_RATE = 0.0003
        # 核心修改：5天不涨就走
        MAX_HOLD_DAYS = 5 
        TRAIL_START = start_trailing
        TRAIL_DROP = drawdown_limit

    cfg = Config()

    # --- 1. 获取大盘 (改为 MA5 风控) ---
    @st.cache_data(ttl=86400, persist=True)
    def get_market_sentiment(start, end):
        try:
            df = pro.index_daily(ts_code='000001.SH', start_date=start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            # 改为 MA5，反应更快，交易机会更多
            df['ma_line'] = df['close'].rolling(5).mean() 
            df['is_safe'] = df['close'] > df['ma_line']
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

    def select_stocks_v11(df):
        if df.empty: return []
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        
        # 核心修改：大幅放宽选股条件
        condition = (
            (df['bias'] > -0.05) & (df['bias'] < 0.30) & # 放宽到 30%，捕捉强势股
            (df['winner_rate'] < 80) & # 获利盘限制也放宽
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 2.5) # 必须活跃
        )
        
        # 按换手率排序，优先买活跃的
        return df[condition].sort_values('turnover_rate', ascending=False).head(5)['ts_code'].tolist()

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
        
        # 风控检查
        is_market_safe = market_safe_map.get(date, False)
        
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

        status_box.text(f"Day: {date} | Market Safe: {is_market_safe} | Pos: {len(positions)}")

        # --- A. Buy ---
        # 激进版：即使大盘不好，只要不是暴跌(MA5能反应)，也允许少量尝试
        # 这里保留熔断，但因为用的是 MA5，熔断概率小很多
        if not is_market_safe:
            buy_queue = [] 
        
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
                    trade_log.append({'date': date, 'code': code, 'action': 'BUY', 'price': buy_price, 'reason': '游击(T+1)'})
        buy_queue = []

        # --- B. Sell ---
        codes_to_sell = []
        for code, pos in positions.items():
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
                    reason = f"快进快出({drawdown*100:.1f}%)"
                elif (pd.to_datetime(date) - pd.to_datetime(pos['date'])).days >= cfg.MAX_HOLD_DAYS:
                    reason = "超时换股"
                
                if reason:
                    revenue = pos['vol'] * sell_price * (1 - cfg.FEE_RATE)
                    profit = revenue - (pos['vol'] * cost)
                    cash += revenue
                    trade_log.append({'date': date, 'code': code, 'action': 'SELL', 'price': round(sell_price, 2), 'profit': round(profit, 2), 'reason': reason})
                    codes_to_sell.append(code)
        for c in codes_to_sell: del positions[c]

        # --- C. Select ---
        # 只要有空位就拼命选
        if is_market_safe and not df_strat.empty and len(positions) < cfg.MAX_POSITIONS:
            targets = select_stocks_v11(df_strat.reset_index())
            for code in targets:
                if code not in positions: buy_queue.append(code)

        # --- D. Settle ---
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
        
        st.subheader("🔥 V11 激进版报告")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("区间收益", f"{ret:.2f}%")
        c2.metric("交易次数", len(trade_log))
        c3.metric("胜率", f"{win_rate:.1f}%")
        c4.metric("周转率", f"{len(trade_log)/len(dates)*100:.0f}%", help="资金活跃度")
        
        st.line_chart(df_res['asset'])
        
        st.divider()
        st.subheader("🎒 当前持仓")
        if positions:
            pos_data = []
            for code, info in positions.items():
                pos_data.append({"代码": code, "日期": info['date'], "成本": f"{info['cost']:.2f}", "浮盈": f"{(price_map_close.get(code,0)-info['cost'])/info['cost']*100:.1f}%"})
            st.dataframe(pd.DataFrame(pos_data))
        else:
            st.info("空仓")

        with st.expander("交易明细"):
            st.dataframe(pd.DataFrame(trade_log))
