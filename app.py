import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V16.5 高价尊享版", layout="wide")
st.title("💎 V16.5 黄金狙击 (高价股尊享版)")
st.markdown("""
### 💎 您的专属纪律：
1.  **拒绝低价股**：严格剔除股价 < **10元** 的标的 (垃圾股滚粗)。
2.  **只做第一名**：在剩下的大票里，选 Bias 最低的冠军。
3.  **单吊+10天**：维持 95% 收益率的核心参数。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 核心参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20250101")
    end_date = st.text_input("结束日期 (设为今天)", value="20251225")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    st.subheader("🚫 选股门槛")
    # === 新增：价格门槛 ===
    min_price = st.number_input("最低股价限制 (元)", value=10.0, step=1.0, help="低于此价格的股票看都不看")
    
    st.divider()
    st.subheader("🎯 交易模式")
    # 默认为 95% 收益率的黄金配置
    max_pos = st.slider("持仓上限 (只)", 1, 5, 1, help="单吊模式收益最高")
    max_hold_days = st.slider("持股天数 (天)", 3, 20, 10, help="耐心持有10天")
    
    STOP_LOSS_FIXED = -0.0501
    
    st.subheader("移动止盈")
    start_trailing = st.slider("启动阈值 (%)", 5, 20, 8) / 100.0
    drawdown_limit = st.slider("允许回撤 (%)", 1, 10, 3) / 100.0

run_btn = st.button("🚀 启动高价股回测", type="primary", use_container_width=True)

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
        MIN_PRICE = min_price # 新增配置

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

    # --- 核心：Rank 1 (带价格过滤) ---
    def select_rank_1_filtered(df):
        if df.empty: return []
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        
        # === 核心修改：增加价格过滤 ===
        condition = (
            (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
            (df['winner_rate'] < 70) &
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5) &
            (df['close'] >= cfg.MIN_PRICE) # 这里！只选大于设定价格的
        )
        
        # 在符合条件的“高价股”里，重新排座次，选第一名
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
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Day: {date} | Price Filter: >={cfg.MIN_PRICE}")

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
        
        # 1. 卖出逻辑
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
                    trade_log.append({
                        '日期': date, '代码': code, '名称': '高价股', # 实际应通过basic获取名称，这里简化
                        '方向': '卖出', '价格': round(sell_price, 2), 
                        '盈亏': round(profit, 2), '理由': reason
                    })
                    codes_to_sell.append(code)
        
        for c in codes_to_sell: del positions[c]

        # 2. 买入逻辑
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
                    trade_log.append({'日期': date, '代码': code, '方向': '买入', '价格': buy_price, '盈亏': 0, '理由': 'Rank1(>10元)'})
        buy_queue = []

        # 3. 选股 (调用新函数)
        if is_market_safe and not df_strat.empty and len(positions) < cfg.MAX_POSITIONS:
            # 使用带过滤的函数
            target_df = select_rank_1_filtered(df_strat.reset_index())
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
    
    if history:
        df_res = pd.DataFrame(history).set_index('date')
        ret = (df_res['asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        
        # 统计
        win_count = len([t for t in trade_log if t['方向']=='卖出' and t['盈亏']>0])
        total_sell = len([t for t in trade_log if t['方向']=='卖出'])
        acc = (win_count / total_sell * 100) if total_sell > 0 else 0
        
        st.subheader(f"💎 回测结果 (仅限 >{cfg.MIN_PRICE}元)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("回测收益率", f"{ret:.2f}%", delta="对比95%如何?")
        c2.metric("交易准确率", f"{acc:.1f}%")
        c3.metric("交易次数", f"{len(trade_log)}次")
        c4.metric("价格门槛", f"≥ {cfg.MIN_PRICE} 元")
        
        st.line_chart(df_res['asset'])
        
        with st.expander("查看交易明细 (验证股价是否都大于10元)"):
            st.dataframe(pd.DataFrame(trade_log))
        
        # 今日信号
        st.divider()
        st.subheader(f"📡 今日 ({cfg.END_DATE}) 高价冠军扫描")
        
        # 再次获取今日数据并过滤
        df_today = fetch_strategy_data(cfg.END_DATE)
        if not df_today.empty:
            df_today['bias'] = (df_today['close'] - df_today['cost_50pct']) / df_today['cost_50pct']
            
            # 手动过滤展示
            filtered_df = df_today[
                (df_today['bias'] > -0.03) & 
                (df_today['bias'] < 0.15) & 
                (df_today['winner_rate'] < 70) & 
                (df_today['turnover_rate'] > 1.5) & 
                (df_today['close'] >= cfg.MIN_PRICE) # 过滤
            ].sort_values('bias').head(1)
            
            if not filtered_df.empty:
                code_now = filtered_df.iloc[0]['ts_code']
                price_now = filtered_df.iloc[0]['close']
                st.success(f"🚀 选出：{code_now} | 现价：{price_now} 元 (符合 >{cfg.MIN_PRICE}元 要求)")
            else:
                st.warning("今日无符合条件的 >10元 股票。")
        else:
            st.info("今日无数据。")

