import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V8 终极防御", layout="wide")
st.title("🛡️ V8.0 全局熔断 + 移动止盈系统")
st.markdown("""
### 核心升级：
1. **📉 回归初心**：只做【主力成本支撑】低吸，剔除诱多风险。
2. **🛑 全局熔断**：**大盘跌破20日线 = 空仓休息**。这是躲避股灾的唯一办法。
3. **🏃 移动止盈**：利润 > 8% 启动跟踪，回撤 3% 自动落袋，拒绝过山车。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20250101")
    end_date = st.text_input("结束日期", value="20251224")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 100) * 10000
    
    st.divider()
    max_pos = st.slider("持仓上限", 3, 5, 3) 
    stop_loss = st.slider("硬止损", -15.0, -5.0, -8.0) / 100.0
    
    st.subheader("移动止盈参数")
    start_trailing = st.slider("启动阈值 (盈利%)", 5, 20, 8) / 100.0
    drawdown_limit = st.slider("允许回撤 (%)", 1, 10, 3) / 100.0

run_btn = st.button("🚀 启动 V8 终极版", type="primary", use_container_width=True)

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
        MAX_HOLD_DAYS = 20 
        # 移动止盈
        TRAIL_START = start_trailing
        TRAIL_DROP = drawdown_limit

    cfg = Config()

    # --- 1. 获取大盘 & 熔断信号 ---
    @st.cache_data(ttl=86400, persist=True)
    def get_market_sentiment(start, end):
        try:
            # 上证指数
            df = pro.index_daily(ts_code='000001.SH', start_date=start, end_date=end)
            df = df.sort_values('trade_date')
            df['ma20'] = df['close'].rolling(20).mean()
            # 熔断标志: Close < MA20
            df['is_safe'] = df['close'] > df['ma20']
            return df.set_index('trade_date')['is_safe'].to_dict()
        except:
            return {}

    # --- 2. 基础行情 (监控用) ---
    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_price_data(date):
        try:
            return pro.daily(trade_date=date)
        except:
            return pd.DataFrame()

    # --- 3. 策略数据 (选股用) ---
    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_strategy_data(date):
        try:
            df_daily = pro.daily(trade_date=date)
            if df_daily.empty: return pd.DataFrame()

            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            df_cyq = pro.cyq_perf(trade_date=date)
            
            if df_cyq.empty or 'cost_50pct' not in df_cyq.columns:
                return pd.DataFrame()

            df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
            df_final = pd.merge(df_merge, df_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
            return df_final
        except:
            return pd.DataFrame()

    # --- 4. 纯低吸选股逻辑 ---
    def select_stocks_v8(df):
        if df.empty: return []
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        
        # 经典 V6 逻辑：跌到成本线，但没跌穿
        condition = (
            (df['bias'] > -0.02) & (df['bias'] < 0.1) & 
            (df['winner_rate'] < 60) &
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5)
        )
        
        selected = df[condition].sort_values('bias', ascending=True).head(3)
        return selected['ts_code'].tolist()

    # --- 5. 回测循环 ---
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = cal_df['cal_date'].tolist()
    market_safe_map = get_market_sentiment(cfg.START_DATE, cfg.END_DATE)

    cash = cfg.INITIAL_CASH
    positions = {}
    history = []
    trade_log = []
    buy_queue = [] # 简化为列表

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        
        # 1. 检查大盘状态
        is_market_safe = market_safe_map.get(date, False) # 默认不安全
        market_status = "🟢 安全" if is_market_safe else "🔴 熔断(只卖不买)"
        
        status_box.text(f"{date} | {market_status} | 持仓: {len(positions)}")

        # 2. 获取数据
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

        # --- A. 执行 T+1 买入 (受熔断控制) ---
        # 如果大盘熔断，清空待买入队列（不接飞刀）
        if not is_market_safe:
            if buy_queue:
                # 记录一下被熔断拦截的操作
                # trade_log.append({'date': date, 'action': 'INFO', 'reason': f'熔断拦截{len(buy_queue)}只买入'})
                buy_queue = []
        
        # 正常买入
        for code in buy_queue:
            if len(positions) >= cfg.MAX_POSITIONS: break
            
            if code in price_map_open:
                buy_price = price_map_open[code]
                slot_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                vol = int(slot_cash / buy_price / 100) * 100
                
                if vol > 0 and cash >= vol * buy_price:
                    cost = vol * buy_price * (1 + cfg.FEE_RATE)
                    cash -= cost
                    positions[code] = {
                        'cost': buy_price, 
                        'vol': vol, 
                        'date': date, 
                        'high_since_buy': buy_price # 初始化最高价
                    }
                    trade_log.append({'date': date, 'code': code, 'action': 'BUY', 'price': buy_price, 'reason': '主力成本(T+1)'})
        
        buy_queue = [] # 清空队列

        # --- B. 智能卖出 (移动止盈 + 硬止损) ---
        codes_to_sell = []
        for code, pos in positions.items():
            if code in price_map_close:
                curr_price = price_map_close[code]
                high_today = price_map_high.get(code, curr_price)
                low_today = price_map_low.get(code, curr_price)
                
                # 1. 更新持仓期间最高价 (用于移动止盈)
                if high_today > pos['high_since_buy']:
                    pos['high_since_buy'] = high_today
                
                cost = pos['cost']
                peak = pos['high_since_buy']
                
                # 计算各种收益率
                curr_ret = (curr_price - cost) / cost
                peak_ret = (peak - cost) / cost
                drawdown_from_peak = (peak - curr_price) / peak
                
                reason = ""
                sell_price = curr_price
                
                # --- 卖出逻辑链 ---
                
                # 1. 硬止损 (保命)
                if (low_today - cost) / cost <= cfg.STOP_LOSS:
                    reason = "止损"
                    sell_price = cost * (1 + cfg.STOP_LOSS)
                    
                # 2. 移动止盈 (保利润)
                # 条件：曾经盈利超过 阈值(8%) 且 从高点回撤超过 限制(3%)
                elif peak_ret >= cfg.TRAIL_START and drawdown_from_peak >= cfg.TRAIL_DROP:
                    reason = f"移动止盈(回撤{drawdown_from_peak*100:.1f}%)"
                    sell_price = curr_price # 按收盘价走
                    
                # 3. 超时
                elif (pd.to_datetime(date) - pd.to_datetime(pos['date'])).days >= cfg.MAX_HOLD_DAYS:
                    reason = "超时"
                
                # 执行卖出
                if reason:
                    revenue = pos['vol'] * sell_price * (1 - cfg.FEE_RATE)
                    profit = revenue - (pos['vol'] * cost)
                    cash += revenue
                    trade_log.append({'date': date, 'code': code, 'action': 'SELL', 'price': round(sell_price, 2), 'profit': round(profit, 2), 'reason': reason})
                    codes_to_sell.append(code)
        
        for c in codes_to_sell: del positions[c]

        # --- C. 每日选股 (仅当大盘安全时) ---
        if is_market_safe and not df_strat.empty and len(positions) < cfg.MAX_POSITIONS:
            targets = select_stocks_v8(df_strat.reset_index())
            for code in targets:
                if code not in positions:
                    buy_queue.append(code)

        # --- D. 结算 ---
        total = cash
        for code, pos in positions.items():
            # 用收盘价估值
            curr = price_map_close.get(code, pos.get('high_since_buy', pos['cost']))
            total += pos['vol'] * curr
        history.append({'date': pd.to_datetime(date), 'asset': total})

    # --- 结果 ---
    status_box.empty()
    st.balloons()
    
    if history:
        df_res = pd.DataFrame(history).set_index('date')
        ret = (df_res['asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        max_dd = ((df_res['asset'].cummax() - df_res['asset']) / df_res['asset'].cummax()).max() * 100
        
        st.subheader("🛡️ V8 终极回测报告")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("区间收益", f"{ret:.2f}%", delta=f"{int(df_res['asset'].iloc[-1]-cfg.INITIAL_CASH)}")
        c2.metric("最大回撤", f"{max_dd:.2f}%")
        c3.metric("交易次数", len(trade_log))
        
        # 统计止盈类型
        if trade_log:
            trail_count = len([t for t in trade_log if '移动' in t.get('reason', '')])
            stop_count = len([t for t in trade_log if '止损' in t.get('reason', '')])
            c4.metric("移动止盈触发", trail_count, help="成功保住利润的次数")

        st.line_chart(df_res['asset'])
        with st.expander("交易明细", expanded=True):
            st.dataframe(pd.DataFrame(trade_log))
