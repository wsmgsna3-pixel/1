import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V15.3 实盘同步版", layout="wide")
st.title("📡 V15.3 黄金狙击 (实盘信号同步版)")
st.markdown("""
### 🛠️ 解决“漏单”漏洞：
* **逻辑升级**：选股信号与持仓解绑。
* **效果**：即使前几天出过信号，只要今天它还是“冠军”，依然会显示在列表中。
* **操作**：您看到信号后，检查自己账户。
    * **没货** ➔ 视为新信号，买入！
    * **有货** ➔ 视为持仓信号，不动。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 战术参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    st.info("💡 建议开始日期往前推 60 天以计算 MA20")
    start_date = st.text_input("开始日期", value="20251101")
    end_date = st.text_input("结束日期 (设为明天)", value="20251225")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    max_pos = st.slider("持仓上限 (只)", 1, 3, 1)
    max_hold_days = st.slider("持股周期 (天)", 1, 15, 10)
    STOP_LOSS_FIXED = -0.0501
    st.error(f"硬止损: {STOP_LOSS_FIXED*100}%")
    
    st.subheader("移动止盈")
    start_trailing = 0.08
    drawdown_limit = 0.03

run_btn = st.button("🚀 启动信号扫描", type="primary", use_container_width=True)

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

    # --- 数据获取 ---
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

    # --- 核心逻辑 ---
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    market_safe_map = get_market_sentiment(cfg.START_DATE, cfg.END_DATE)

    cash = cfg.INITIAL_CASH
    positions = {} 
    history = []
    trade_log = []
    
    # === 新增：每日信号池 (用于展示当日所有符合条件的股，不管是否持仓) ===
    daily_signals = []

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
        
        # 1. Sell Logic (回测逻辑保持不变，用于画资金曲线)
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
                    reason = "移动止盈"
                elif (current_date_obj - pd.to_datetime(pos['date'])).days >= cfg.MAX_HOLD_DAYS:
                    reason = "超时"
                if reason:
                    revenue = pos['vol'] * sell_price * (1 - cfg.FEE_RATE)
                    profit = revenue - (pos['vol'] * cost)
                    cash += revenue
                    trade_log.append({'日期': date, '代码': code, '方向': '卖出', '理由': reason, '盈亏': profit, '排名': '-'})
                    codes_to_sell.append(code)
        for c in codes_to_sell: del positions[c]

        # 2. Select & Buy Logic
        # 只要是大盘安全，就去选股
        if is_market_safe and not df_strat.empty:
            target_df = select_stocks_ranked(df_strat.reset_index())
            
            # === 关键修正：无论买没买，如果是最后一天，都记录下来 ===
            if date == dates[-1]: # 如果是回测的最后一天(也就是今天/明天)
                for idx, row in target_df.iterrows():
                    code = row['ts_code']
                    rank = row['day_rank']
                    bias_val = row['bias']
                    # 添加到今日信号板
                    daily_signals.append({
                        '代码': code,
                        '排名': f"第 {rank} 名",
                        'Bias': f"{bias_val*100:.2f}%",
                        '状态': '持有中' if code in positions else '建议买入'
                    })
            
            # 正常的回测买入逻辑 (用于计算资金曲线)
            for i, row in target_df.iterrows():
                code = row['ts_code']
                if len(positions) < cfg.MAX_POSITIONS and code not in positions:
                    if code in price_map_open:
                        buy_price = price_map_open[code]
                        slot_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                        vol = int(slot_cash / buy_price / 100) * 100
                        if vol > 0 and cash >= vol * buy_price:
                            cash -= vol * buy_price * (1 + cfg.FEE_RATE)
                            positions[code] = {'cost': buy_price, 'vol': vol, 'date': date, 'high_since_buy': buy_price}
                            trade_log.append({'日期': date, '代码': code, '方向': '买入', '理由': '低吸', '盈亏': 0, '排名': f"第 {row['day_rank']} 名"})

        # Settle
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
        
        # === 核心变化：专门开辟一个区域展示“今日信号” ===
        st.subheader("📡 今日冠军信号 (实盘参考)")
        
        if not is_market_safe:
            st.error("🛑 大盘状态危险 (MA20下方) - 建议空仓")
        elif not daily_signals:
            st.warning("⚠️ 大盘安全，但今日无符合Bias条件的股票")
        else:
            df_sig = pd.DataFrame(daily_signals)
            
            def color_signal(row):
                if '第 1 名' in row['排名']:
                    return ['background-color: #d4edda; color: green; font-weight: bold'] * len(row)
                return [''] * len(row)

            st.dataframe(df_sig.style.apply(color_signal, axis=1), use_container_width=True)
            st.info("👆 请核对上表：如果您账户里没有这只“第 1 名”，请视为【买入信号】！")

        st.divider()
        st.markdown("### 📊 回测详情")
        c1, c2 = st.columns(2)
        c1.metric("回测区间收益", f"{ret:.2f}%")
        c2.metric("当前模拟持仓", f"{len(positions)} / {cfg.MAX_POSITIONS}")
        st.line_chart(df_res['asset'])
