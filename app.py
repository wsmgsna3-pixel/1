import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V16.2 全天候雷达", layout="wide")
st.title("📡 V16.2 黄金狙击 (全天候雷达版)")
st.markdown("""
### 👁️ 核心升级：所见即所得
无论当前是否满仓，系统都会强制计算 **今日(结束日期)** 的选股结果，并给出操作建议。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 实盘参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    # 建议回测整年
    start_date = st.text_input("开始日期", value="20251001")
    end_date = st.text_input("结束日期 (设为今天)", value="20251225")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    max_pos = 3
    st.success(f"持仓上限: {max_pos} 只")
    
    max_hold_days = 10
    STOP_LOSS_FIXED = -0.0501
    
    st.subheader("止盈参数")
    start_trailing = st.slider("启动阈值 (%)", 5, 20, 8) / 100.0
    drawdown_limit = st.slider("允许回撤 (%)", 1, 10, 3) / 100.0

run_btn = st.button("🚀 启动雷达扫描", type="primary", use_container_width=True)

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

    # --- 1. 获取大盘 ---
    @st.cache_data(ttl=60)
    def get_market_sentiment(start, end):
        try:
            real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma20'] = df['close'].rolling(20).mean()
            df['is_safe'] = df['close'] > df['ma20']
            
            last_row = df.iloc[-1]
            return {
                'map': df.set_index('trade_date')['is_safe'].to_dict(),
                'last_date': last_row['trade_date'],
                'last_close': last_row['close'],
                'last_ma20': last_row['ma20']
            }
        except Exception as e:
            return {'map': {}, 'error': str(e)}

    # --- 2. 数据获取 ---
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

    # --- 核心：只选 Rank 1 ---
    def select_rank_1_only(df):
        if df.empty: return []
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        condition = (
            (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
            (df['winner_rate'] < 70) &
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5)
        )
        sorted_df = df[condition].sort_values('bias', ascending=True)
        return sorted_df.head(1)

    # ==========================================
    # PART 1: 回测部分 (计算资金曲线和持仓状态)
    # ==========================================
    market_data = get_market_sentiment(cfg.START_DATE, cfg.END_DATE)
    market_safe_map = market_data.get('map', {})
    
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    
    cash = cfg.INITIAL_CASH
    positions = {} 
    history = []
    
    # 我们需要跑到今天，知道今天的持仓状态
    progress_bar = st.progress(0)
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Calculating History: {date}")

        df_price = fetch_price_data(date)
        df_strat = fetch_strategy_data(date)
        
        price_map_close = {}
        price_map_high = {}
        price_map_low = {}
        price_map_open = {}
        
        if not df_price.empty:
            df_price = df_price.set_index('ts_code')
            price_map_close = df_price['close'].to_dict()
            price_map_high = df_price['high'].to_dict()
            price_map_low = df_price['low'].to_dict()
            price_map_open = df_price['open'].to_dict()
        
        # --- 卖出逻辑 (简化版，用于更新持仓) ---
        codes_to_sell = []
        current_date_obj = pd.to_datetime(date)
        for code, pos in positions.items():
            if current_date_obj <= pd.to_datetime(pos['date']): continue
            if code in price_map_close:
                curr_price = price_map_close[code]
                high_today = price_map_high.get(code, curr_price)
                low_today = price_map_low.get(code, curr_price)
                if high_today > pos['high_since_buy']: pos['high_since_buy'] = high_today
                
                # 止损/止盈 check
                reason = ""
                cost = pos['cost']
                peak = pos['high_since_buy']
                if (low_today - cost) / cost <= cfg.STOP_LOSS: reason = "止损"
                elif (peak - cost)/cost >= cfg.TRAIL_START and (peak - curr_price)/peak >= cfg.TRAIL_DROP: reason = "止盈"
                elif (current_date_obj - pd.to_datetime(pos['date'])).days >= cfg.MAX_HOLD_DAYS: reason = "超时"
                
                if reason:
                    revenue = pos['vol'] * curr_price * (1 - cfg.FEE_RATE)
                    cash += revenue
                    codes_to_sell.append(code)
        for c in codes_to_sell: del positions[c]

        # --- 买入逻辑 (回测历史中的买入) ---
        # 注意：这里是为了算出今天还剩多少钱、满不满仓。
        # 今天的具体信号在下面单独算。
        if is_market_safe and not df_strat.empty and len(positions) < cfg.MAX_POSITIONS:
            # 只在回测循环里做简单的买入模拟
            target_df = select_rank_1_only(df_strat.reset_index())
            for i, row in target_df.iterrows():
                if row['ts_code'] not in positions:
                     if row['ts_code'] in price_map_open:
                        buy_price = price_map_open[row['ts_code']]
                        slot_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                        vol = int(slot_cash / buy_price / 100) * 100
                        if vol > 0 and cash >= vol * buy_price:
                            cash -= vol * buy_price * (1 + cfg.FEE_RATE)
                            positions[row['ts_code']] = {'cost': buy_price, 'vol': vol, 'date': date, 'high_since_buy': buy_price}

    status_box.empty()
    st.balloons()

    # ==========================================
    # PART 2: 今日信号独立雷达 (核心功能)
    # ==========================================
    st.divider()
    st.header("🔭 今日信号雷达 (无论满仓与否，强制扫描)")
    
    today_date = cfg.END_DATE
    today_market_safe = market_safe_map.get(today_date, False)
    
    # 1. 大盘状态
    c1, c2, c3 = st.columns(3)
    c1.metric("今日日期", today_date)
    if 'last_close' in market_data:
        c2.metric("大盘 Close", f"{market_data['last_close']:.2f}")
        c3.metric("大盘 MA20", f"{market_data['last_ma20']:.2f}")
    
    if not today_market_safe:
        st.error(f"🛑 警报：今日大盘位于 MA20 下方，系统建议空仓。")
        # 即使空仓，也展示选股结果供观察，但不建议买
        st.caption("以下为观察标的（仅供参考，不建议操作）：")
    else:
        st.success(f"✅ 状态：大盘安全，雷达开启。")

    # 2. 强制选股
    df_today_strat = fetch_strategy_data(today_date)
    if df_today_strat.empty:
        st.warning(f"⚠️ 尚未获取到 {today_date} 的个股数据，可能是盘中尚未收盘，或Tushare数据延迟。")
    else:
        target_today = select_rank_1_only(df_today_strat.reset_index())
        
        if target_today.empty:
            st.info("🤷 今日无 Rank 1 标的入围 (未满足 Bias 或 换手率条件)。")
        else:
            # 拿到今日冠军
            top_stock = target_today.iloc[0]
            code = top_stock['ts_code']
            name = top_stock.get('name', code) # 有时候df里没name，就显示code
            bias = top_stock['bias']
            
            # 3. 智能决策判断
            st.subheader(f"🏆 今日冠军：{code}")
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Bias (乖离率)", f"{bias*100:.2f}%")
            
            with col_b:
                # 核心逻辑：判断买不买
                if not today_market_safe:
                    st.error("🛑 风险提示：标的虽好，但大盘危险，禁止开新仓！")
                elif code in positions:
                    st.info("🔵 持仓中：您已经持有该股，今日无需操作，躺赢即可。")
                elif len(positions) < cfg.MAX_POSITIONS:
                    st.success(f"🟢 强烈建议买入：\n1. 它是 Rank 1\n2. 大盘安全\n3. 您有 {cfg.MAX_POSITIONS - len(positions)} 个空位\n🚀 目标：{code}")
                else:
                    st.warning(f"🟡 仓位已满 (3/3)：\n系统选出了 {code}，它是今日冠军。\n但您没有空位了。如果您认为它比手里的股票更好，可以考虑'卖弱换强'。")
            
            st.markdown("---")
            st.dataframe(target_today)

    # 3. 展示当前模拟持仓
    with st.expander("当前模拟账户持仓详情"):
        if positions:
            st.write(positions)
        else:
            st.write("目前空仓。")
