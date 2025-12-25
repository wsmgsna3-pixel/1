import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import time

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V15.3 信号修复版", layout="wide")
st.title("📡 V15.3 黄金狙击 (实盘信号同步版)")
st.markdown("""
### 🛠️ 修复与诊断：
1.  **大盘诊断器**：在侧边栏显示系统读取到的 **真实指数数据**。
2.  **表格修复**：解决交易记录不显示的问题。
3.  **模式确认**：当前为 **Rank 1 单吊模式 (1只持仓)** 还是 **均衡模式 (3只)**？请在侧边栏确认。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    my_token = st.text_input("Tushare Token", type="password")
    
    # 建议回测整年
    start_date = st.text_input("开始日期", value="20250101")
    end_date = st.text_input("结束日期 (设为今天)", value="20251225")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    st.subheader("🎯 仓位模式 (影响收益率的核心)")
    # 这里给您选择权：是要 71% 的稳健，还是 Rank 1 的刺激
    pos_mode = st.radio("选择持仓模式：", ["稳健型 (3只持仓)", "激进型 (1只持仓/单吊)"])
    
    if pos_mode == "稳健型 (3只持仓)":
        max_pos = 3
    else:
        max_pos = 1
        
    st.info(f"当前持仓上限: {max_pos} 只")
    
    # 保持冠军参数
    max_hold_days = 10
    STOP_LOSS_FIXED = -0.0501
    
    st.subheader("止盈参数")
    start_trailing = st.slider("启动阈值 (%)", 5, 20, 8) / 100.0
    drawdown_limit = st.slider("允许回撤 (%)", 1, 10, 3) / 100.0

run_btn = st.button("🚀 启动修复回测", type="primary", use_container_width=True)

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

    # --- 1. 获取大盘 (增加诊断信息) ---
    @st.cache_data(ttl=60) # 缩短缓存时间，确保实盘数据新鲜
    def get_market_sentiment(start, end):
        try:
            # 强制多取 60 天数据，确保 MA20 能算出来
            real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma20'] = df['close'].rolling(20).mean()
            df['is_safe'] = df['close'] > df['ma20']
            
            # === 诊断信息输出 ===
            last_row = df.iloc[-1]
            return {
                'map': df.set_index('trade_date')['is_safe'].to_dict(),
                'last_date': last_row['trade_date'],
                'last_close': last_row['close'],
                'last_ma20': last_row['ma20'],
                'data_count': len(df)
            }
        except Exception as e:
            return {'map': {}, 'error': str(e)}

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

    # --- 选股逻辑 ---
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

    # --- 4. 回测循环 ---
    market_data = get_market_sentiment(cfg.START_DATE, cfg.END_DATE)
    market_safe_map = market_data.get('map', {})
    
    # === 诊断信息展示 ===
    with st.expander("🩺 大盘数据诊断 (为什么显示危险?)", expanded=True):
        if 'error' in market_data:
            st.error(f"数据获取失败: {market_data['error']}")
        else:
            last_date = market_data.get('last_date', '未知')
            last_close = market_data.get('last_close', 0)
            last_ma20 = market_data.get('last_ma20', 0)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("系统读取的最新日期", f"{last_date}")
            c2.metric("最新收盘价", f"{last_close:.2f}")
            c3.metric("最新 MA20", f"{last_ma20:.2f}")
            
            if last_date != cfg.END_DATE:
                st.warning(f"⚠️ 警告：系统还没读到 {cfg.END_DATE} 的数据！目前只停留在 {last_date}。这可能是 Tushare 数据未更新，导致系统误判。")
            elif last_close > last_ma20:
                st.success("✅ 数据显示：大盘安全 (Close > MA20)")
            else:
                st.error("🛑 数据显示：大盘危险 (Close <= MA20)")

    # --- 开始回测 ---
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
        # 如果大盘不安全，清空买入队列 (实盘选股时，这里很关键)
        if not is_market_safe: 
            buy_queue = [] 
        
        for item in buy_queue:
            code = item['code']
            rank = item['rank']
            bias_val = item['bias']
            
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

        # 3. Select (实盘选股核心)
        # 即使大盘不安全，我们也可以算出哪些是"符合Bias"的，只是不买而已
        # 这里为了展示，我们把选股逻辑放出来，方便您看今天到底有没有好票
        if not df_strat.empty:
            target_df = select_stocks_ranked(df_strat.reset_index())
            for i, row in target_df.iterrows():
                # 只有在大盘安全 且 仓位未满时，才加入"待买入队列"
                # 但我们可以打印出来看看
                if is_market_safe and len(positions) < cfg.MAX_POSITIONS and row['ts_code'] not in positions:
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
        
        st.subheader("📡 实盘信号面板")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("区间收益", f"{ret:.2f}%")
        c2.metric("最新仓位", f"{len(positions)} / {cfg.MAX_POSITIONS}")
        c3.metric("模式", "激进(1只)" if cfg.MAX_POSITIONS==1 else "稳健(3只)")
        c4.metric("交易笔数", len(trade_log))
        
        st.line_chart(df_res['asset'])
        
        st.divider()
        st.markdown("### 📋 交易明细 (修复显示版)")
        
        if trade_log:
            df_log = pd.DataFrame(trade_log)
            # 确保日期列是 datetime 类型以便排序
            df_log['日期'] = pd.to_datetime(df_log['日期'])
            df_log['日期Str'] = df_log['日期'].dt.strftime('%Y%m%d') # 用于显示的字符串
            
            # 按日期倒序 (最新在最上)
            df_log = df_log.sort_values(by=['日期', '排名'], ascending=[False, True])
            
            # 样式
            def highlight_rows(row):
                if row['方向'] == '买入':
                    return ['background-color: #d4edda; color: green'] * len(row)
                elif '止损' in str(row['理由']):
                     return ['background-color: #f8d7da; color: red'] * len(row)
                return [''] * len(row)

            # 显示时去掉 timestamp 的时分秒
            display_df = df_log.drop(columns=['日期'])
            st.dataframe(display_df.style.apply(highlight_rows, axis=1), height=600)
        else:
            st.info("⚠️ 暂无交易记录。可能原因：大盘一直处于危险状态，或资金已满仓。")
