import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V12 趋势共振", layout="wide")
st.title("📈 V12.0 趋势共振版 (胜率优先)")
st.markdown("""
### 🎯 胜率拯救计划：
1.  **只做右侧**：放弃抄底，只买 **站稳主力成本线** 的强势股。
2.  **双重共振**：大盘 > 20日线 + 个股 > 20日线 (均线多头)。
3.  **拒绝阴跌**：只买 **阳线** (收盘价 > 开盘价)，确保动量向上。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20250101")
    end_date = st.text_input("结束日期", value="20251224")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    # 既然追求胜率，仓位可以稍微集中一点
    max_pos = st.slider("持仓只数", 2, 5, 3) 
    stop_loss = st.slider("硬止损", -10.0, -3.0, -6.0) / 100.0
    
    st.subheader("稳健止盈")
    start_trailing = st.slider("启动阈值", 5, 20, 8) / 100.0
    drawdown_limit = st.slider("允许回撤", 1, 10, 3) / 100.0

run_btn = st.button("🚀 启动 V12 高胜率版", type="primary", use_container_width=True)

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
        MAX_HOLD_DAYS = 10 # 趋势票拿稳一点，给10天
        TRAIL_START = start_trailing
        TRAIL_DROP = drawdown_limit

    cfg = Config()

    # --- 1. 获取大盘 (必须站稳 MA20) ---
    @st.cache_data(ttl=86400, persist=True)
    def get_market_sentiment(start, end):
        try:
            df = pro.index_daily(ts_code='000001.SH', start_date=start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma20'] = df['close'].rolling(20).mean()
            # 只有站上 20日线才操作，过滤掉熊市和暴跌
            df['is_safe'] = df['close'] > df['ma20']
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
            # 获取基础指标
            df_daily = pro.daily(trade_date=date)
            if df_daily.empty: return pd.DataFrame()
            
            # 我们需要简单的均线数据来判断趋势
            # 但Tushare daily接口不直接给MA，我们用一种巧妙的方法：
            # 1. 主力成本线 (cost_50pct) 本身就是一种超级均线
            # 2. 我们要求 Price > Cost 且 Close > Open
            
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            df_cyq = pro.cyq_perf(trade_date=date)
            
            if df_cyq.empty or 'cost_50pct' not in df_cyq.columns: return pd.DataFrame()
            
            df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
            df_final = pd.merge(df_merge, df_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
            return df_final
        except:
            return pd.DataFrame()

    def select_stocks_v12(df):
        if df.empty: return []
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        
        # === V12 核心胜率滤网 ===
        condition = (
            (df['bias'] > 0.01) & (df['bias'] < 0.15) &  # 关键：必须在成本线上方(1%~15%)，说明已经突破压力位
            (df['close'] > df['open']) &                 # 关键：当天必须是阳线 (红盘)
            (df['winner_rate'] < 80) &                   # 获利盘适中，防止高位出货
            (df['circ_mv'] > 300000) &                   # 剔除小票
            (df['turnover_rate'] > 2.0)                  # 必须有量
        )
        # 优先买资金关注度高的（换手率排序）
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
        is_market_safe = market_safe_map.get(date, False) # MA20 风控
        status_box.text(f"Day: {date} | Market Safe: {is_market_safe} | Pos: {len(positions)}")

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

        # 1. Sell Logic (严谨 T+1)
        codes_to_sell = []
        current_date_obj = pd.to_datetime(date)

        for code, pos in positions.items():
            if current_date_obj <= pd.to_datetime(pos['date']): continue # 冻结T+1

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
                    reason = f"趋势止盈({drawdown*100:.1f}%)"
                elif (current_date_obj - pd.to_datetime(pos['date'])).days >= cfg.MAX_HOLD_DAYS:
                    reason = "趋势走坏(超时)"
                
                if reason:
                    revenue = pos['vol'] * sell_price * (1 - cfg.FEE_RATE)
                    profit = revenue - (pos['vol'] * cost)
                    cash += revenue
                    trade_log.append({'date': date, 'code': code, 'action': 'SELL', 'price': round(sell_price, 2), 'profit': round(profit, 2), 'reason': reason})
                    codes_to_sell.append(code)
        
        for c in codes_to_sell: del positions[c]

        # 2. Buy Logic (仅大盘安全时)
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
                    trade_log.append({'date': date, 'code': code, 'action': 'BUY', 'price': buy_price, 'reason': '趋势确认(T+1)'})
        buy_queue = []

        # 3. Select
        if is_market_safe and not df_strat.empty and len(positions) < cfg.MAX_POSITIONS:
            targets = select_stocks_v12(df_strat.reset_index())
            for code in targets:
                if code not in positions: buy_queue.append(code)

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
        
        st.subheader("📈 V12.0 趋势共振版报告")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("区间收益", f"{ret:.2f}%")
        c2.metric("交易次数", len(trade_log))
        c3.metric("真实胜率", f"{win_rate:.1f}%")
        c4.metric("持仓情况", len(positions))
        
        st.line_chart(df_res['asset'])
        with st.expander("交易明细"):
            st.dataframe(pd.DataFrame(trade_log))
