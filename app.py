import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="主力成本V6 (严谨版)", layout="wide")
st.title("⚓ Tushare V6.0 主力成本支撑 (T+1严谨回测)")
st.markdown("""
### V6 升级说明：
1. **真实模拟**：T日选股，**T+1日开盘价买入**（更符合实战）。
2. **硬盘缓存**：下载过的数据不再重复下载，大幅提升速度。
3. **更长周期**：建议测试 2024 全年。
""")

# ==========================================
# 侧边栏设置
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    my_token = st.text_input("Tushare Token", type="password")
    
    # 默认时间拉长，测试长期稳定性
    start_date = st.text_input("开始日期", value="20240101")
    end_date = st.text_input("结束日期", value="20241220")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 100) * 10000
    
    st.divider()
    stop_loss = st.slider("止损阈值", -15.0, -3.0, -8.0) / 100.0
    take_profit = st.slider("止盈阈值", 5.0, 50.0, 15.0) / 100.0
    max_hold_days = st.slider("最长持股天数", 5, 30, 10)

run_btn = st.button("🚀 启动 V6 回测", type="primary", use_container_width=True)

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
        MAX_POSITIONS = 3
        STOP_LOSS = stop_loss
        TAKE_PROFIT = take_profit
        FEE_RATE = 0.0003
        MAX_HOLD_DAYS = max_hold_days

    cfg = Config()

    # --- 1. 缓存交易日历 (硬盘缓存) ---
    @st.cache_data(ttl=86400, persist=True)
    def get_trading_days(start, end):
        try:
            df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
            return df['cal_date'].tolist()
        except:
            return []

    # --- 2. 获取数据 (核心耗时步骤，开启 persist=True) ---
    # 注意：如果这步报错，可能是磁盘写入权限问题，通常 Streamlit Cloud 没问题
    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_data_cached(date):
        try:
            # A. 基础行情
            df_daily = pro.daily(trade_date=date)
            if df_daily.empty: return pd.DataFrame()

            # B. 每日指标
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            
            # C. 筹码数据
            df_cyq = pro.cyq_perf(trade_date=date)
            if df_cyq.empty or 'cost_50pct' not in df_cyq.columns:
                return pd.DataFrame()

            # 合并
            df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
            df_final = pd.merge(df_merge, df_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
            
            return df_final
        except:
            return pd.DataFrame()

    # --- 3. 选股逻辑 (不变) ---
    def select_stocks(df):
        if df.empty: return []
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        
        condition = (
            (df['bias'] > 0) & (df['bias'] < 0.1) &  # 支撑位
            (df['winner_rate'] < 70) &               # 拒绝高位
            (df['circ_mv'] > 300000) &               # 只要中大盘
            (df['turnover_rate'] > 1.0)              # 有流动性
        )
        selected = df[condition].sort_values('bias', ascending=True).head(3)
        return selected['ts_code'].tolist()

    # --- 4. 回测引擎 (T+1 模式) ---
    dates = get_trading_days(cfg.START_DATE, cfg.END_DATE)
    if not dates:
        st.error("日期无效")
        st.stop()

    cash = cfg.INITIAL_CASH
    positions = {}
    history = []
    trade_log = []
    
    # 待买入队列 {code: signal_date}
    buy_queue = {} 

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        status_box.markdown(f"🗓️ **{date}** | 资产: {int(cash + sum([p['vol']*p['last_price'] for p in positions.values()])) if positions else int(cash)}")
        
        df_today = fetch_data_cached(date)
        
        price_map_close = {}
        price_map_open = {}
        price_map_high = {}
        price_map_low = {}
        
        if not df_today.empty:
            df_today = df_today.set_index('ts_code')
            price_map_close = df_today['close'].to_dict()
            price_map_open = df_today['open'].to_dict()
            price_map_high = df_today['high'].to_dict()
            price_map_low = df_today['low'].to_dict()

        # --- A. 处理昨日的买入信号 (T+1 开盘买入) ---
        # 遍历待买入队列
        for code in list(buy_queue.keys()):
            if len(positions) >= cfg.MAX_POSITIONS: 
                buy_queue.pop(code) # 没钱了，信号作废
                continue
                
            if code in price_map_open:
                # 按开盘价买入
                buy_price = price_map_open[code]
                
                # 资金分配
                slot_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                vol = int(slot_cash / buy_price / 100) * 100
                
                if vol > 0 and cash >= vol * buy_price:
                    cost = vol * buy_price * (1 + cfg.FEE_RATE)
                    cash -= cost
                    positions[code] = {
                        'cost': buy_price, 
                        'vol': vol, 
                        'date': date, 
                        'last_price': buy_price
                    }
                    trade_log.append({
                        'date': date, 'code': code, 'action': 'BUY', 
                        'price': buy_price, 'reason': 'T+1开盘'
                    })
                # 买完(或买不起)移出队列
                buy_queue.pop(code)
            else:
                # 停牌，保留信号到明天
                pass

        # --- B. 持仓监控 (卖出逻辑) ---
        codes_to_sell = []
        for code, pos in positions.items():
            if code in price_map_close:
                # 更新最新价用于计算市值
                pos['last_price'] = price_map_close[code]
                
                cost = pos['cost']
                high_p = price_map_high.get(code, pos['last_price'])
                low_p = price_map_low.get(code, pos['last_price'])
                close_p = pos['last_price']
                
                reason = ""
                sell_price = close_p
                
                # 1. 止损 (检查最低价)
                if (low_p - cost) / cost <= cfg.STOP_LOSS:
                    reason = "止损"
                    sell_price = cost * (1 + cfg.STOP_LOSS) # 模拟触价成交
                    
                # 2. 止盈 (检查最高价)
                elif (high_p - cost) / cost >= cfg.TAKE_PROFIT:
                    reason = "止盈"
                    sell_price = cost * (1 + cfg.TAKE_PROFIT)
                
                # 3. 超时 (检查收盘价)
                elif (pd.to_datetime(date) - pd.to_datetime(pos['date'])).days >= cfg.MAX_HOLD_DAYS:
                    reason = "超时"
                    sell_price = close_p
                
                if reason:
                    revenue = pos['vol'] * sell_price * (1 - cfg.FEE_RATE - 0.001)
                    profit = revenue - (pos['vol'] * cost)
                    cash += revenue
                    trade_log.append({
                        'date': date, 'code': code, 'action': 'SELL', 
                        'price': round(sell_price, 2), 
                        'profit': round(profit, 2), 
                        'reason': reason
                    })
                    codes_to_sell.append(code)
        
        for c in codes_to_sell: del positions[c]

        # --- C. 每日选股 (产生信号放进队列) ---
        if not df_today.empty and len(positions) + len(buy_queue) < cfg.MAX_POSITIONS:
            targets = select_stocks(df_today.reset_index())
            for code in targets:
                if code not in positions and code not in buy_queue:
                    # 放入待买入队列，明天开盘买
                    buy_queue[code] = date

        # --- D. 结算资产 ---
        total_asset = cash
        for code, pos in positions.items():
            total_asset += pos['vol'] * pos.get('last_price', pos['cost'])
        
        history.append({'date': pd.to_datetime(date), 'asset': total_asset})

    # ==========================================
    # 结果展示
    # ==========================================
    status_box.empty()
    st.balloons()
    
    if history:
        df_res = pd.DataFrame(history).set_index('date')
        ret = (df_res['asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        max_dd = ((df_res['asset'].cummax() - df_res['asset']) / df_res['asset'].cummax()).max() * 100
        
        st.subheader("📊 V6 严谨回测报告")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("区间收益", f"{ret:.2f}%", delta=f"{int(df_res['asset'].iloc[-1]-cfg.INITIAL_CASH)}")
        c2.metric("最大回撤", f"{max_dd:.2f}%")
        c3.metric("交易次数", len(trade_log))
        c4.metric("胜率", f"{len([t for t in trade_log if t['action']=='SELL' and t['profit']>0]) / len([t for t in trade_log if t['action']=='SELL']) * 100:.1f}%" if trade_log else "0.0%")
        
        st.line_chart(df_res['asset'])
        
        with st.expander("查看详细交易流水", expanded=True):
            if trade_log:
                df_tx = pd.DataFrame(trade_log)
                # 格式化一下显示
                st.dataframe(df_tx.style.format({'price': '{:.2f}', 'profit': '{:.2f}'}))
            else:
                st.info("区间内无交易")
    else:
        st.error("数据获取失败，请重试")
