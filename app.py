import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V18.4 决战高开", layout="wide")
st.title("🧪 V18.4 黄金实验室 (数据强制修复版)")
st.markdown("""
### 🕵️‍♂️ 决战时刻：抓捕“高开龙”
* **修复机制**：强制刷新所有缓存数据，确保 `pre_close` (昨收价) 100% 准确。
* **实验目标**：揭开 **Gap > 0** (高开) 的真实收益率面纱。
* **预期**：如果高开组胜率 > 60%，这就是策略的最后一块拼图！
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 实验参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20240504")
    end_date = st.text_input("结束日期", value="20251226")
    
    st.divider()
    st.success("🔒 价格锁定: 11.0 - 20.0 元")
    
    # 基础策略参数
    STOP_LOSS_PCT = 5.0
    TRAIL_START_PCT = 8.0
    TRAIL_DROP_PCT = 3.0
    MAX_HOLD_DAYS = 10

run_btn = st.button("🚀 启动最终验证", type="primary", use_container_width=True)

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
        STOP_LOSS = - (STOP_LOSS_PCT / 100.0) - 0.0001
        TRAIL_START = TRAIL_START_PCT / 100.0
        TRAIL_DROP = TRAIL_DROP_PCT / 100.0
        MAX_HOLD_DAYS = MAX_HOLD_DAYS
        FEE_RATE = 0.0003

    cfg = Config()

    # --- 数据函数 (重命名以强制刷新缓存) ---
    @st.cache_data(ttl=60)
    def get_market_sentiment_v4(start, end):
        try:
            real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma20'] = df['close'].rolling(20).mean()
            return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
        except: return {}

    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_price_data_v4(date):
        # 强制获取 pre_close
        try: 
            df = pro.daily(trade_date=date)
            # 简单清洗，确保 columns 存在
            if 'pre_close' not in df.columns:
                return pd.DataFrame()
            return df
        except: return pd.DataFrame()

    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_strategy_data_v4(date):
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
    market_safe_map = get_market_sentiment_v4(cfg.START_DATE, cfg.END_DATE)
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    
    active_signals = [] 
    finished_signals = [] 
    
    # 调试计数器
    debug_gap_calc_count = 0

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Scanning: {date} | Gap Calc: {debug_gap_calc_count}")

        df_price = fetch_price_data_v4(date)
        df_strat = fetch_strategy_data_v4(date)
        
        price_map_open = {}
        price_map_close = {}
        price_map_high = {}
        price_map_low = {}
        price_map_pre_close = {} 
        
        if not df_price.empty:
            df_price = df_price.set_index('ts_code')
            price_map_open = df_price['open'].to_dict()
            price_map_close = df_price['close'].to_dict()
            price_map_high = df_price['high'].to_dict()
            price_map_low = df_price['low'].to_dict()
            price_map_pre_close = df_price['pre_close'].to_dict()
        
        # 1. 更新信号
        signals_still_active = []
        current_date_obj = pd.to_datetime(date)
        
        for sig in active_signals:
            code = sig['code']
            
            # 补全 Gap 数据
            if current_date_obj <= pd.to_datetime(sig['buy_date']):
                if code in price_map_high:
                     sig['highest'] = max(sig['highest'], price_map_high[code])
                
                # === 强制计算 Gap ===
                if sig['gap'] is None:
                    if code in price_map_open and code in price_map_pre_close:
                        open_p = price_map_open[code]
                        pre_c = price_map_pre_close[code]
                        sig['gap'] = (open_p - pre_c) / pre_c * 100
                        debug_gap_calc_count += 1
                
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
                        'return': ret, 'reason': reason,
                        'gap': sig['gap']
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
                        'buy_price': price_map_open[code], 'highest': price_map_open[code],
                        'gap': None 
                    })

    # --- 结果展示 ---
    status_box.empty()
    st.balloons()
    
    if finished_signals:
        df_res = pd.DataFrame(finished_signals)
        df_res['return_pct'] = df_res['return'] * 100
        
        # 显式统计缺失情况
        missing_gap_count = df_res['gap'].isna().sum()
        df_res['gap'] = df_res['gap'].fillna(0) # 仅用于画图，不影响High组
        
        df_high = df_res[df_res['gap'] > 0]
        df_low = df_res[df_res['gap'] <= 0]
        
        st.subheader(f"🧠 最终情绪分析 (11-20元)")
        st.info(f"数据体检：成功计算Gap {len(df_res) - missing_gap_count} 笔，缺失 {missing_gap_count} 笔。")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.error(f"🔥 高开组 (Gap > 0)")
            if not df_high.empty:
                high_win = len(df_high[df_high['return']>0]) / len(df_high) * 100
                high_avg = df_high['return'].mean() * 100
                st.metric("胜率", f"{high_win:.1f}%", delta="VS 低开")
                st.metric("期望收益", f"{high_avg:.2f}%")
                st.metric("样本数", f"{len(df_high)}")
            else:
                st.write("依然无高开数据？(那可能真是策略选不到)")
                
        with c2:
            st.success(f"🧊 低开组 (Gap ≤ 0)")
            if not df_low.empty:
                low_win = len(df_low[df_low['return']>0]) / len(df_low) * 100
                low_avg = df_low['return'].mean() * 100
                st.metric("胜率", f"{low_win:.1f}%")
                st.metric("期望收益", f"{low_avg:.2f}%")
                st.metric("样本数", f"{len(df_low)}")
