import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V16.3 完美融合", layout="wide")
st.title("🏆 V16.3 黄金狙击 (回测+实盘 完美融合版)")
st.markdown("""
### 💎 您的全能指挥台：
1.  **历史回测**：验证策略的长期收益率和准确率 (看上面)。
2.  **今日雷达**：锁定今天的 Rank 1 冠军股 (看下面)。
3.  **参数自由**：侧边栏参数已解锁，可自由调整。
""")

# ==========================================
# 侧边栏 (参数全部回归)
# ==========================================
with st.sidebar:
    st.header("⚙️ 策略参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    # 默认回测一整年，确保有数据
    start_date = st.text_input("回测开始", value="20250101")
    end_date = st.text_input("回测结束 (设为今天)", value="20251225")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    st.subheader("🎯 仓位与风控")
    
    # === 参数解锁 ===
    max_pos = st.slider("持仓上限 (只)", 1, 5, 3, help="建议设为3，既有容错又能抓连板")
    max_hold_days = st.slider("持股天数 (天)", 3, 20, 10, help="建议10天，给主力拉升时间")
    
    STOP_LOSS_FIXED = -0.0501
    st.error(f"硬止损: {STOP_LOSS_FIXED*100}%")
    
    st.subheader("移动止盈")
    start_trailing = st.slider("启动阈值 (%)", 5, 20, 8) / 100.0
    drawdown_limit = st.slider("允许回撤 (%)", 1, 10, 3) / 100.0

run_btn = st.button("🚀 启动回测 & 扫描今日", type="primary", use_container_width=True)

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

    # --- 核心：Rank 1 Only ---
    def select_rank_1_only(df):
        if df.empty: return []
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        condition = (
            (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
            (df['winner_rate'] < 70) &
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5)
        )
        # 排序并只取第一名
        sorted_df = df[condition].sort_values('bias', ascending=True)
        return sorted_df.head(1)

    # --- 回测循环 ---
    market_data = get_market_sentiment(cfg.START_DATE, cfg.END_DATE)
    market_safe_map = market_data.get('map', {})
    
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    
    cash = cfg.INITIAL_CASH
    positions = {} 
    history = []
    trade_log = []
    buy_queue = [] 

    progress_bar = st.progress(0)
    
    # 这里的循环是为了生成历史回测数据
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"回测进行中: {date}")

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
        
        # 1. 卖出
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
                    trade_log.append({'日期': date, '代码': code, '方向': '卖出', '价格': round(sell_price, 2), '盈亏': round(profit, 2), '理由': reason})
                    codes_to_sell.append(code)
        for c in codes_to_sell: del positions[c]

        # 2. 买入
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
                    trade_log.append({'日期': date, '代码': code, '方向': '买入', '价格': buy_price, '盈亏': 0, '理由': 'Rank1'})
        buy_queue = []

        # 3. 选股
        if is_market_safe and not df_strat.empty and len(positions) < cfg.MAX_POSITIONS:
            target_df = select_rank_1_only(df_strat.reset_index())
            for i, row in target_df.iterrows():
                if row['ts_code'] not in positions: buy_queue.append(row['ts_code'])

        # 4. 结算
        total = cash
        for code, pos in positions.items():
            total += pos['vol'] * price_map_close.get(code, pos['high_since_buy'])
        history.append({'date': pd.to_datetime(date), 'asset': total})

    # --- 结果展示 ---
    status_box.empty()
    st.balloons()
    
    # === 第一部分：历史回测报告 ===
    st.header("📊 历史战绩验证 (2025全年)")
    if history:
        df_res = pd.DataFrame(history).set_index('date')
        ret = (df_res['asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        
        # 计算胜率
        wins = len([t for t in trade_log if t['方向']=='卖出' and t['盈亏']>0])
        total_sells = len([t for t in trade_log if t['方向']=='卖出'])
        win_rate = (wins / total_sells * 100) if total_sells > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("回测总收益", f"{ret:.2f}%")
        c2.metric("交易准确率", f"{win_rate:.1f}%")
        c3.metric("总交易次数", f"{len(trade_log)}")
        c4.metric("当前策略", f"持仓{cfg.MAX_POSITIONS}只 | 仅买Rank1")
        
        with st.expander("查看详细交易流水"):
            st.dataframe(pd.DataFrame(trade_log))
    
    st.divider()

    # === 第二部分：今日雷达 (实盘核心) ===
    st.header(f"📡 今日雷达信号 ({cfg.END_DATE})")
    
    # 诊断大盘
    is_today_safe = market_safe_map.get(cfg.END_DATE, False)
    real_today_close = market_data.get('last_close', 0)
    real_today_ma20 = market_data.get('last_ma20', 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 大盘环境")
        if real_today_close > real_today_ma20:
            st.success(f"✅ 安全 (指数 {real_today_close:.0f} > MA20 {real_today_ma20:.0f})")
        else:
            st.error(f"🛑 危险 (指数 {real_today_close:.0f} <= MA20 {real_today_ma20:.0f})")
            st.caption("系统风控：今日禁止开新仓")

    with col2:
        st.subheader("2. 冠军扫描")
        # 重新跑一次今天的选股，无论有没有钱都显示出来
        df_today = fetch_strategy_data(cfg.END_DATE)
        target_df = select_rank_1_only(df_today.reset_index()) if not df_today.empty else pd.DataFrame()
        
        if not target_df.empty:
            champion_code = target_df.iloc[0]['ts_code']
            champion_bias = target_df.iloc[0]['bias']
            st.metric("今日 Rank 1", champion_code, delta=f"Bias: {champion_bias*100:.2f}%")
            
            # === 核心逻辑：给您的建议 ===
            current_holdings = list(positions.keys())
            if not is_today_safe:
                st.warning("⚠️ 建议：大盘危险，不要买入，即使有冠军股。")
            elif len(positions) < cfg.MAX_POSITIONS:
                if champion_code in current_holdings:
                     st.info("ℹ️ 建议：持有不动 (已在持仓中)。")
                else:
                     st.success(f"🚀 建议：买入 {champion_code} (仓位充足)")
            else:
                # 满仓时的建议
                st.error("⛔ 建议：仓位已满 (3/3)，系统自动放弃买入。")
                st.markdown(f"""
                **思考题：卖弱换强？**
                * 系统选出了 **{champion_code}**。
                * 但您手里有 3 只票。
                * 如果手里有跌破 -4% 快止损的，或者涨不动横盘的，**可以考虑**手动卖出它，换入这只 Rank 1。
                * *注意：这是手动操作，违反了系统全自动原则，但符合实战利益。*
                """)
        else:
            st.info("今日无符合条件的 Rank 1 股票。")
