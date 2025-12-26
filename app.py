import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V17.1 区间锁定", layout="wide")
st.title("🧪 V17.1 黄金狙击 (区间锁定版)")
st.markdown("""
### 🎯 策略进化：掐头去尾，吃中间
数据诊断显示：
* **< 8元**：垃圾股陷阱 (已剔除)
* **> 20元**：机构死鱼股 (本次剔除)
* **8-20元**：策略的真实盈利核心！
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 最终区间")
    my_token = st.text_input("Tushare Token", type="password")
    start_date = st.text_input("开始日期", value="20240504")
    end_date = st.text_input("结束日期", value="20251226")
    
    st.divider()
    
    # === 核心修改：价格双限 ===
    c1, c2 = st.columns(2)
    min_price = c1.number_input("最低价", value=8.0)
    max_price = c2.number_input("最高价", value=20.0)
    
    st.info(f"锁定区间: {min_price} - {max_price} 元")

run_btn = st.button("🚀 验证黄金区间 (8-20元)", type="primary", use_container_width=True)

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
        MIN_PRICE = min_price
        MAX_PRICE = max_price # 新增上限

    cfg = Config()

    # --- 数据函数 (缓存通用) ---
    @st.cache_data(ttl=60)
    def get_market_sentiment(start, end):
        try:
            real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma20'] = df['close'].rolling(20).mean()
            return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
        except: return {}

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

    def select_rank_1(df):
        if df.empty: return None
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        condition = (
            (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
            (df['winner_rate'] < 70) &
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5) &
            (df['close'] >= cfg.MIN_PRICE) &
            (df['close'] <= cfg.MAX_PRICE) # === 核心修改：增加上限 ===
        )
        sorted_df = df[condition].sort_values('bias', ascending=True)
        if sorted_df.empty: return None
        return sorted_df.iloc[0]

    # --- 无限火力 回测循环 ---
    market_safe_map = get_market_sentiment(cfg.START_DATE, cfg.END_DATE)
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    
    active_signals = [] 
    finished_signals = [] 

    progress_bar = st.progress(0)
    
    # 固定参数
    STOP_LOSS_FIXED = -0.0501
    MAX_HOLD_DAYS = 10
    TRAIL_START = 0.08
    TRAIL_DROP = 0.03
    FEE_RATE = 0.0003
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Scanning: {date}")

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
        
        # 1. 更新信号
        signals_still_active = []
        current_date_obj = pd.to_datetime(date)
        
        for sig in active_signals:
            code = sig['code']
            if current_date_obj <= pd.to_datetime(sig['buy_date']):
                if code in price_map_close:
                     sig['highest'] = max(sig['highest'], price_map_high.get(code, 0))
                signals_still_active.append(sig)
                continue

            if code in price_map_close:
                curr_price = price_map_close[code]
                high_today = price_map_high.get(code, curr_price)
                if high_today > sig['highest']: sig['highest'] = high_today
                
                cost = sig['buy_price']
                peak = sig['highest']
                peak_ret = (peak - cost) / cost
                drawdown = (peak - curr_price) / peak
                low_today = price_map_low.get(code, curr_price)
                
                reason = ""
                sell_price = curr_price
                
                if (low_today - cost) / cost <= STOP_LOSS_FIXED:
                    reason = "止损"
                    sell_price = cost * (1 + STOP_LOSS_FIXED)
                elif peak_ret >= TRAIL_START and drawdown >= TRAIL_DROP:
                    reason = "止盈"
                elif (current_date_obj - pd.to_datetime(sig['buy_date'])).days >= MAX_HOLD_DAYS:
                    reason = "超时"
                
                if reason:
                    ret = (sell_price - cost) / cost - FEE_RATE * 2
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

    # --- 结果 ---
    status_box.empty()
    st.balloons()
    
    if finished_signals:
        df_res = pd.DataFrame(finished_signals)
        df_res['return_pct'] = df_res['return'] * 100
        
        total_trades = len(df_res)
        avg_ret = df_res['return'].mean() * 100
        total_virtual_ret = df_res['return'].sum() * 100
        
        st.subheader(f"🎯 黄金区间回测 (8-20元)")
        c1, c2, c3 = st.columns(3)
        c1.metric("单笔期望收益", f"{avg_ret:.2f}%", help="这应该是正的！")
        c2.metric("虚拟总收益", f"{total_virtual_ret:.1f}%")
        c3.metric("总交易次数", f"{total_trades}")
        
        st.success("💡 见证奇迹的时刻：如果期望收益是正的（比如 >0.5%），这就是您要找的‘圣杯’。")
