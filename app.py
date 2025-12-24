import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="主力成本支撑V5", layout="wide")
st.title("⚓ Tushare V5.0 主力成本支撑系统")
st.markdown("""
### 策略核心：拒绝高位接盘，只做底部支撑
1. **安全垫**：买入价接近市场平均成本 (`cost_50pct`)。
2. **避高位**：剔除获利盘过高 (>60%) 的股票，防止主力出货。
3. **抓反弹**：在主力护盘线附近低吸。
""")

# ==========================================
# 参数设置
# ==========================================
with st.sidebar:
    st.header("⚙️ 抄底参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20241008")
    end_date = st.text_input("结束日期", value="20241130")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 100) * 10000
    
    st.divider()
    # 止盈止损可以稍微放大，因为是底部买入
    stop_loss = st.slider("止损阈值", -10.0, -3.0, -8.0) / 100.0
    take_profit = st.slider("止盈阈值", 5.0, 50.0, 15.0) / 100.0

# 按钮区
st.divider()
run_btn = st.button("🚀 启动 V5 抄底回测", type="primary", use_container_width=True)

# ==========================================
# 核心逻辑
# ==========================================
if run_btn:
    if not my_token:
        st.error("请先在左侧输入 Token")
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

    cfg = Config()

    @st.cache_data(ttl=3600)
    def get_trading_days(start, end):
        try:
            df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
            return df['cal_date'].tolist()
        except:
            return []

    # --- 数据获取 ---
    def fetch_data_support(date):
        try:
            # 1. 基础行情
            df_daily = pro.daily(trade_date=date)
            if df_daily.empty: return pd.DataFrame()

            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            
            # 2. 筹码数据 (获取 cost_50pct 成本均线)
            df_cyq = pd.DataFrame()
            try:
                df_cyq = pro.cyq_perf(trade_date=date)
            except:
                pass

            if df_cyq.empty or 'cost_50pct' not in df_cyq.columns:
                return pd.DataFrame() # 没筹码数据就不做

            # 3. 合并
            df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
            # 关键：获取成本数据
            df_final = pd.merge(df_merge, df_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
            
            return df_final
        except:
            return pd.DataFrame()

    # --- 选股逻辑 (V5 核心) ---
    def select_stocks_support(df):
        if df.empty: return []
        
        # 计算 乖离率：(当前价 - 平均成本) / 平均成本
        # 结果越接近 0，说明价格越接近成本线
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        
        condition = (
            # 1. 价格要在成本线附近 (支撑位)
            (df['bias'] > 0) &                # 股价刚站上成本线
            (df['bias'] < 0.1) &              # 距离成本线不超过 10% (安全区间)
            
            # 2. 拒绝高位票
            (df['winner_rate'] < 70) &        # 获利盘不要太多，防止砸盘
            
            # 3. 基本面过滤
            (df['circ_mv'] > 300000) &        # 剔除小票
            (df['pe_ttm'] > 0) & (df['pe_ttm'] < 60) & # 有业绩支撑
            
            # 4. 活跃度
            (df['turnover_rate'] > 2.0)
        )
        
        selected = df[condition].copy()
        
        # 优先选 bias 最小的 (离成本线最近的，最安全)
        selected = selected.sort_values(by='bias', ascending=True).head(3)
        return selected['ts_code'].tolist()

    # --- 回测执行 ---
    dates = get_trading_days(cfg.START_DATE, cfg.END_DATE)
    if not dates:
        st.error("日期范围无效")
        st.stop()

    cash = cfg.INITIAL_CASH
    positions = {}
    history = []
    trade_log = []
    
    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        status_box.caption(f"回测进度: {date}")
        
        df_today = fetch_data_support(date)
        
        price_map = {}
        if not df_today.empty:
            price_map = df_today.set_index('ts_code')['close'].to_dict()

        # 1. 卖出
        codes_to_del = []
        for code, pos in positions.items():
            if code in price_map:
                curr_p = price_map[code]
                cost = pos['cost']
                ret = (curr_p - cost) / cost
                
                reason = ""
                if ret <= cfg.STOP_LOSS: reason = "止损"
                elif ret >= cfg.TAKE_PROFIT: reason = "止盈"
                elif (pd.to_datetime(date) - pd.to_datetime(pos['date'])).days >= 10: reason = "超时换股"
                
                if reason:
                    revenue = pos['vol'] * curr_p * (1 - cfg.FEE_RATE - 0.001)
                    cash += revenue
                    profit = revenue - (pos['vol'] * cost)
                    trade_log.append({'date': date, 'code': code, 'action': 'SELL', 'price': curr_p, 'reason': reason, 'profit': profit})
                    codes_to_del.append(code)
        for c in codes_to_del: del positions[c]

        # 2. 买入
        if not df_today.empty and len(positions) < cfg.MAX_POSITIONS:
            targets = select_stocks_support(df_today)
            for code in targets:
                if code not in positions and code in price_map:
                    if len(positions) < cfg.MAX_POSITIONS:
                        price = price_map[code]
                        slot_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                        vol = int(slot_cash / price / 100) * 100
                        
                        if vol > 0 and cash >= vol * price:
                            cash -= vol * price * (1 + cfg.FEE_RATE)
                            positions[code] = {'cost': price, 'vol': vol, 'date': date}
                            trade_log.append({'date': date, 'code': code, 'action': 'BUY', 'price': price, 'reason': '成本支撑'})

        # 3. 结算
        total = cash
        for code in positions:
            p = price_map.get(code, positions[code]['cost'])
            total += positions[code]['vol'] * p
        history.append({'date': pd.to_datetime(date), 'asset': total})

    # --- 结果 ---
    status_box.empty()
    st.balloons()
    
    if history:
        df_res = pd.DataFrame(history).set_index('date')
        final_val = df_res['asset'].iloc[-1]
        ret = (final_val - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        
        st.subheader("📊 V5 回测报告")
        c1, c2, c3 = st.columns(3)
        c1.metric("最终收益", f"{ret:.2f}%", delta_color="normal")
        c2.metric("最大回撤", f"{((df_res['asset'].cummax() - df_res['asset'])/df_res['asset'].cummax()).max():.2%}")
        c3.metric("交易次数", len(trade_log))
        
        st.line_chart(df_res['asset'])
        if trade_log:
            st.dataframe(pd.DataFrame(trade_log))
