import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V15.2 自由战术版", layout="wide")
st.title("📱 V15.2 黄金狙击 (参数完全解锁 + 微信排序)")
st.markdown("""
### 🔓 参数控制权已移交：
1.  **持仓数量**：您现在可以设为 **1只** (只买冠军)，执行“斩首行动”。
2.  **持股天数**：1-15天由您定。
3.  **显示模式**：依然保持“最新日期在最上方”的微信式排序，Rank 1 依然高亮。
""")

# ==========================================
# 侧边栏 (控制台)
# ==========================================
with st.sidebar:
    st.header("⚙️ 战术参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    # 日期设置提醒
    st.info("💡 提示：为了计算 MA20，开始日期建议往前推 60 天。")
    start_date = st.text_input("开始日期", value="20251101")
    end_date = st.text_input("结束日期 (设为明天)", value="20251225")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    
    # === 解锁 1: 持仓上限 ===
    # 之前回测显示 Rank 1 最好，所以默认值设为 1，方便您执行“单吊策略”
    max_pos = st.slider("持仓上限 (只)", 1, 3, 1, help="回测数据显示：只买第1名效益最高")
    
    # === 解锁 2: 持股周期 ===
    # 范围放宽到 1-15 天
    max_hold_days = st.slider("持股周期 (天)", 1, 15, 10, help="默认10天，也可尝试短线4-5天")
    
    # === 锁死: 硬止损 ===
    STOP_LOSS_FIXED = -0.0501
    st.error(f"硬止损: {STOP_LOSS_FIXED*100}% (已锁死 -5.01%)")
    
    st.subheader("移动止盈")
    start_trailing = st.slider("启动阈值 (%)", 5, 20, 8) / 100.0
    drawdown_limit = st.slider("允许回撤 (%)", 1, 10, 3) / 100.0

run_btn = st.button("🚀 启动自由回测", type="primary", use_container_width=True)

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
        MAX_POSITIONS = max_pos      # 动态获取侧边栏
        STOP_LOSS = STOP_LOSS_FIXED  # 锁死
        FEE_RATE = 0.0003
        MAX_HOLD_DAYS = max_hold_days # 动态获取侧边栏
        TRAIL_START = start_trailing
        TRAIL_DROP = drawdown_limit

    cfg = Config()

    # --- 1. 获取大盘 (MA20) ---
    @st.cache_data(ttl=86400, persist=True)
    def get_market_sentiment(start, end):
        try:
            # 自动前推60天获取数据以计算MA20
            real_start = (pd.to_datetime(start) - pd.Timedelta(days=60)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma_safe'] = df['close'].rolling(20).mean()
            df['is_safe'] = df['close'] > df['ma_safe']
            return df.set_index('trade_date')['is_safe'].to_dict()
        except:
            return {}

    # --- 2. 基础数据 ---
    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_price_data(date):
        try: return pro.daily(trade_date=date)
        except: return pd.DataFrame()

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
        # 选出前5名备用，具体买几个由 MAX_POSITIONS 控制
        selected = df[condition].sort_values('bias', ascending=True).head(5)
        
        selected = selected.reset_index(drop=True)
        selected['day_rank'] = selected.index + 1 
        return selected

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
                    reason = "止损(T+1盘中)"
                    sell_price = cost * (1 + cfg.STOP_LOSS)
                elif peak_ret >= cfg.TRAIL_START and drawdown >= cfg.TRAIL_DROP:
                    reason = f"移动止盈({drawdown*100:.1f}%)"
                elif (current_date_obj - pd.to_datetime(pos['date'])).days >= cfg.MAX_HOLD_DAYS:
                    reason = f"超时换股({cfg.MAX_HOLD_DAYS}天)"
                
                if reason:
                    revenue = pos['vol'] * sell_price * (1 - cfg.FEE_RATE)
                    profit = revenue - (pos['vol'] * cost)
                    cash += revenue
                    trade_log.append({
                        '日期': date, '代码': code, '方向': '卖出', 
                        '价格': round(sell_price, 2), '盈亏': round(profit, 2), 
                        '理由': reason, '排名': '-', 'Bias': '-'
                    })
                    codes_to_sell.append(code)
        
        for c in codes_to_sell: del positions[c]

        # 2. Buy Logic
        if not is_market_safe: 
            buy_queue = [] 
        
        for item in buy_queue:
            code = item['code']
            rank = item['rank']
            bias_val = item['bias']
            
            # === 关键：这里会根据您在侧边栏设置的 max_pos 自动停止买入 ===
            # 如果您设为1，买完Rank 1后，循环就会因为这个判断而break，不会买Rank 2
            if len(positions) >= cfg.MAX_POSITIONS: break
            
            if code in price_map_open:
                buy_price = price_map_open[code]
                slot_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                vol = int(slot_cash / buy_price / 100) * 100
                if vol > 0 and cash >= vol * buy_price:
                    cost = vol * buy_price * (1 + cfg.FEE_RATE)
                    cash -= cost
                    positions[code] = {'cost': buy_price, 'vol': vol, 'date': date, 'high_since_buy': buy_price}
                    trade_log.append({
                        '日期': date, '代码': code, '方向': '买入', 
                        '价格': buy_price, '盈亏': 0, 
                        '理由': '低吸(T+1)', 
                        '排名': f"第 {rank} 名", 
                        'Bias': f"{bias_val*100:.2f}%"
                    })
        buy_queue = []

        # 3. Select
        if is_market_safe and not df_strat.empty and len(positions) < cfg.MAX_POSITIONS:
            target_df = select_stocks_ranked(df_strat.reset_index())
            for i, row in target_df.iterrows():
                if row['ts_code'] not in positions: 
                    buy_queue.append({
                        'code': row['ts_code'], 
                        'rank': row['day_rank'],
                        'bias': row['bias']
                    })

        # 4. Settle
        total = cash
        for code, pos in positions.items():
            total += pos['vol'] * price_map_close.get(code, pos['high_since_buy'])
        history.append({'date': pd.to_datetime(date), 'asset': total})

    # --- 结果展示 ---
    status_box.empty()
    st.balloons()
    
    if history:
        df_res = pd.DataFrame(history).set_index('date')
        ret = (df_res['asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        
        st.subheader("📱 V15.2 实盘面板 (自由定制)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("区间收益", f"{ret:.2f}%")
        c2.metric("当前持仓", f"{len(positions)} / {cfg.MAX_POSITIONS}")
        c3.metric("持股上限", f"{cfg.MAX_HOLD_DAYS} 天")
        c4.metric("硬止损", "-5.01%")
        
        st.line_chart(df_res['asset'])
        
        st.divider()
        st.markdown("### 📋 交易明细 (微信排序：最新在最上)")
        
        if trade_log:
            df_log = pd.DataFrame(trade_log)
            # 微信排序：日期倒序，排名正序
            df_log = df_log.sort_values(by=['日期', '排名'], ascending=[False, True])
            
            def highlight_rows(row):
                if row['方向'] == '买入':
                    if '第 1 名' in str(row['排名']):
                        return ['background-color: #d4edda; color: green'] * len(row)
                    return ['background-color: #f0f8ff'] * len(row)
                elif row['理由'] and '止损' in str(row['理由']):
                     return ['background-color: #f8d7da; color: red'] * len(row)
                return [''] * len(row)

            st.dataframe(df_log.style.apply(highlight_rows, axis=1), height=600)
        else:
            st.info("近期无交易。请检查日期设置或大盘状态。")
