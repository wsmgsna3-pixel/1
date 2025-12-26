import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V18.7 最终稳定版", layout="wide")
st.title("🛡️ V18.7 黄金实验室 (宽止损·最终稳定版)")
st.markdown("""
### 📝 明日行动指南
1.  **首次运行**：请手动清除缓存 (Clear Cache)，耐心等待数据下载 (约1小时)。
2.  **快速测试**：数据下载完成后，**拖动左侧止损滑块**，结果将秒级更新。
3.  **核心目标**：找到让胜率 > 50% 的那个止损点 (可能是 -8% 或 -10%)。
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 核心参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20240504")
    end_date = st.text_input("结束日期", value="20251226")
    
    st.divider()
    st.success("🔒 黄金区间: 11.0 - 20.0 元")
    
    # === 关键：止损滑块 ===
    st.subheader("🛡️ 止损防线测试")
    stop_loss_input = st.slider("止损线 (-%)", 5.0, 15.0, 8.0, step=0.5, 
                                help="数值越大，给主力的空间越大。建议直接从 8.0% 开始测。")
    
    st.caption(f"当前设置：跌破 **-{stop_loss_input}%** 止损")
    
    # 其他固定参数
    TRAIL_START_PCT = 8.0 
    TRAIL_DROP_PCT = 3.0
    MAX_HOLD_DAYS = 10

run_btn = st.button("🚀 启动回测 (首次需等待)", type="primary", use_container_width=True)

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
        MIN_PRICE = 11.0
        MAX_PRICE = 20.0
        # 动态止损
        STOP_LOSS = - (stop_loss_input / 100.0) - 0.0001
        TRAIL_START = TRAIL_START_PCT / 100.0
        TRAIL_DROP = TRAIL_DROP_PCT / 100.0
        MAX_HOLD_DAYS = MAX_HOLD_DAYS
        FEE_RATE = 0.0003

    cfg = Config()

    # --- 标准函数名 (保证缓存稳定) ---
    @st.cache_data(ttl=86400) # 24小时缓存
    def get_market_sentiment_final(start, end):
        try:
            real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma20'] = df['close'].rolling(20).mean()
            return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
        except: return {}

    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_price_data_final(date):  
        try: return pro.daily(trade_date=date)
        except: return pd.DataFrame()

    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_strategy_data_final(date): 
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

    def select_rank_1(df):
        if df.empty: return None
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        condition = (
            (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
            (df['winner_rate'] < 70) &
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5) &
            (df['close'] >= cfg.MIN_PRICE) &
            (df['close'] <= cfg.MAX_PRICE) 
        )
        sorted_df = df[condition].sort_values('bias', ascending=True)
        if sorted_df.empty: return None
        return sorted_df.iloc[0]

    # --- 回测循环 ---
    market_safe_map = get_market_sentiment_final(cfg.START_DATE, cfg.END_DATE)
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    
    active_signals = [] 
    finished_signals = [] 

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Scanning: {date}")

        df_price = fetch_price_data_final(date)
        df_strat = fetch_strategy_data_final(date)
        
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
        
        # 1. 更新信号
        signals_still_active = []
        current_date_obj = pd.to_datetime(date)
        
        for sig in active_signals:
            code = sig['code']
            if current_date_obj <= pd.to_datetime(sig['buy_date']):
                if code in price_map_high:
                     sig['highest'] = max(sig['highest'], price_map_high[code])
                signals_still_active.append(sig)
                continue

            if code in price_map_close:
                curr_price = price_map_close[code]
                high_today = price_map_high.get(code, curr_price)
                low_today = price_map_low.get(code, curr_price)
                
                if high_today > sig['highest']: sig['highest'] = high_today
                
                cost = sig['buy_price']
                peak = sig['highest']
                peak_ret = (peak - cost) / cost
                drawdown = (peak - curr_price) / peak
                
                reason = ""
                sell_price = curr_price
                
                # === 动态止损 ===
                if (low_today - cost) / cost <= cfg.STOP_LOSS:
                    reason = "止损"
                    sell_price = cost * (1 + cfg.STOP_LOSS)
                elif peak_ret >= cfg.TRAIL_START and drawdown >= cfg.TRAIL_DROP:
                    reason = "止盈"
                    sell_price = peak * (1 - cfg.TRAIL_DROP) 
                elif (current_date_obj - pd.to_datetime(sig['buy_date'])).days >= cfg.MAX_HOLD_DAYS:
                    reason = "超时"
                
                if reason:
                    ret = (sell_price - cost) / cost - cfg.FEE_RATE * 2
                    finished_signals.append({
                        'code': code, 'buy_date': sig['buy_date'],
                        'return': ret, 'reason': reason
                    })
                else:
                    signals_still_active.append(sig)
            else:
                signals_still_active.append(sig)
        
        active_signals = signals_still_active

        # 2. 发出新信号
        if is_market_safe and not df_strat.empty:
            target_row = select_rank_1(df_strat.reset_index())
            if target_row is not None:
                code = target_row['ts_code']
                if code in price_map_open:
                    active_signals.append({
                        'code': code, 'buy_date': date,
                        'buy_price': price_map_open[code], 'highest': price_map_open[code]
                    })

    # --- 结果展示 ---
    status_box.empty()
    st.balloons()
    
    if finished_signals:
        df_res = pd.DataFrame(finished_signals)
        df_res['return_pct'] = df_res['return'] * 100
        
        total_trades = len(df_res)
        win_trades = len(df_res[df_res['return'] > 0])
        win_rate = win_trades / total_trades * 100
        avg_ret = df_res['return'].mean() * 100
        
        stop_loss_counts = len(df_res[df_res['reason']=='止损'])
        
        st.subheader(f"🛡️ 止损 {stop_loss_input}% 测试结果")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("真实胜率", f"{win_rate:.1f}%")
        c2.metric("单笔期望", f"{avg_ret:.2f}%")
        c3.metric("止损触发率", f"{stop_loss_counts/total_trades*100:.1f}%")
        
        st.divider()
        if win_rate > 50:
            st.success(f"✅ 胜率突破 50%！当前设置为：-{stop_loss_input}%")
        else:
            st.warning(f"⚠️ 胜率仍为 {win_rate:.1f}%。")
        
        st.subheader("📊 盈亏分布")
        chart = alt.Chart(df_res).mark_circle(size=60).encode(
            x=alt.X('return_pct', title='单笔收益 (%)'),
            y='count()',
            color=alt.condition(
                alt.datum.return_pct > 0,
                alt.value("#d32f2f"),
                alt.value("#2e7d32")
            ),
            tooltip=['code', 'buy_date', 'return_pct', 'reason']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
