import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V15.1 实盘指挥部", layout="wide")
st.title("📱 V15.1 黄金狙击 (微信排序 + 冠军排名)")
st.markdown("""
### 👁️ 实盘看盘指南：
1.  **排序方式**：**最新日期在最上面** (像微信消息一样)。
2.  **排名指标**：请重点看 **【排名】** 列。
    * 🥇 **第 1 名**：当天的“金股”，Bias 最低，必须优先买。
    * 🥈 **第 2/3 名**：备选，有钱再买。
3.  **选股技巧**：如果只想看明天的票，把**结束日期设为明天**，**开始日期往前推 60 天**。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 实盘参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    # 建议默认跨度设大一点，防止MA20数据不足
    start_date = st.text_input("开始日期 (建议前推60天)", value="20251101")
    end_date = st.text_input("结束日期 (设为明天)", value="20251225")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 20) * 10000
    
    st.divider()
    max_pos = 3
    st.success(f"持仓上限: {max_pos} 只 (黄金配置)")
    
    max_hold_days = 10
    st.success(f"持股周期: {max_hold_days} 天 (耐心持有)")
    
    STOP_LOSS_FIXED = -0.0501
    st.error(f"硬止损: {STOP_LOSS_FIXED*100}% (盘中条件单)")
    
    st.subheader("移动止盈")
    start_trailing = st.slider("启动阈值 (%)", 5, 20, 8) / 100.0
    drawdown_limit = st.slider("允许回撤 (%)", 1, 10, 3) / 100.0

run_btn = st.button("🚀 启动选股/回测", type="primary", use_container_width=True)

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
            # 多取一点数据以计算MA20
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

    # --- 选股逻辑 (增加排名计算) ---
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
        
        # === 核心：给选出来的股票打上排名标签 ===
        # reset_index后，第一行就是第0个，rank = index + 1
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
    buy_queue = [] # 存储结构：[{'code': code, 'rank': 1, 'bias': -0.02}]

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
                    # 卖出记录不需要排名，填空
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
            
            if len(positions) >= cfg.MAX_POSITIONS: break
            if code in price_map_open:
                buy_price = price_map_open[code]
                slot_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                vol = int(slot_cash / buy_price / 100) * 100
                if vol > 0 and cash >= vol * buy_price:
                    cost = vol * buy_price * (1 + cfg.FEE_RATE)
                    cash -= cost
                    positions[code] = {'cost': buy_price, 'vol': vol, 'date': date, 'high_since_buy': buy_price}
                    # 记录买入时的排名和Bias
                    trade_log.append({
                        '日期': date, '代码': code, '方向': '买入', 
                        '价格': buy_price, '盈亏': 0, 
                        '理由': '低吸(T+1)', 
                        '排名': f"第 {rank} 名", 
                        'Bias': f"{bias_val*100:.2f}%"
                    })
        buy_queue = []

        # 3. Select (带排名的选股)
        if is_market_safe and not df_strat.empty and len(positions) < cfg.MAX_POSITIONS:
            target_df = select_stocks_ranked(df_strat.reset_index())
            for i, row in target_df.iterrows():
                if row['ts_code'] not in positions: 
                    # 将排名信息存入队列
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

    # --- 结果展示 (微信式排序) ---
    status_box.empty()
    st.balloons()
    
    if history:
        df_res = pd.DataFrame(history).set_index('date')
        ret = (df_res['asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        
        st.subheader("📱 实盘操作面板")
        c1, c2, c3 = st.columns(3)
        c1.metric("区间收益", f"{ret:.2f}%")
        c2.metric("最新仓位", f"{len(positions)} / {cfg.MAX_POSITIONS}")
        c3.metric("大盘状态", "安全" if is_market_safe else "危险(空仓)")
        
        st.line_chart(df_res['asset'])
        
        st.divider()
        st.markdown("### 📋 交易明细 (最新在最上)")
        
        if trade_log:
            df_log = pd.DataFrame(trade_log)
            # === 核心修改：按日期倒序排列，同一天按排名正序 ===
            # 这样今天的数据在最上面，且第1名排在第2名上面
            df_log = df_log.sort_values(by=['日期', '排名'], ascending=[False, True])
            
            # 高亮显示“买入”和“第 1 名”
            def highlight_rows(row):
                if row['方向'] == '买入':
                    if '第 1 名' in str(row['排名']):
                        return ['background-color: #d4edda; color: green'] * len(row) # 冠军买入亮绿色
                    return ['background-color: #f0f8ff'] * len(row) # 普通买入浅蓝色
                elif row['理由'] and '止损' in str(row['理由']):
                     return ['background-color: #f8d7da; color: red'] * len(row) # 止损浅红色
                return [''] * len(row)

            st.dataframe(df_log.style.apply(highlight_rows, axis=1), height=600)
        else:
            st.info("近期无交易，建议检查日期设置或休息观望。")
