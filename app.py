import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt

# 1. Page Config 必须放在最前面，且不再变动
st.set_page_config(page_title="V19.2 因子挖掘", layout="wide")

# ==========================================
# 侧边栏配置
# ==========================================
st.sidebar.header("⚙️ 核心参数")
my_token = st.sidebar.text_input("Tushare Token", type="password")

start_date = st.sidebar.text_input("开始日期", value="20240504")
end_date = st.sidebar.text_input("结束日期", value="20251226")

st.sidebar.divider()
st.sidebar.success("🔒 黄金区间: 11.0 - 20.0 元")
st.sidebar.info("🛡️ 止损: 固定 -5%")

run_btn = st.sidebar.button("🚀 启动因子扫描", type="primary", use_container_width=True)

# ==========================================
# 核心逻辑区
# ==========================================
if run_btn:
    if not my_token:
        st.error("请输入 Token")
        st.stop()
    
    # 设置 Token
    ts.set_token(my_token)
    try:
        pro = ts.pro_api()
    except Exception as e:
        st.error(f"连接失败: {e}")
        st.stop()

    # 配置参数
    CFG_START = start_date
    CFG_END = end_date
    CFG_MIN_PRICE = 11.0
    CFG_MAX_PRICE = 20.0
    CFG_STOP_LOSS = -0.0501
    CFG_TRAIL_START = 0.08
    CFG_TRAIL_DROP = 0.03
    CFG_MAX_HOLD = 10
    CFG_FEE = 0.0003

    status_box = st.empty()

    # --- 数据获取函数 (移除 persist=True，恢复纯净) ---
    @st.cache_data(ttl=86400)
    def get_market_sentiment_pure(start, end):
        try:
            real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma20'] = df['close'].rolling(20).mean()
            return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
        except: return {}

    @st.cache_data(ttl=86400)
    def fetch_price_data_pure(date):
        try: return pro.daily(trade_date=date)
        except: return pd.DataFrame()

    @st.cache_data(ttl=86400)
    def fetch_strategy_data_pure(date):
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

    def select_rank_1_features(df):
        if df.empty: return None
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        condition = (
            (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5) &
            (df['close'] >= CFG_MIN_PRICE) &
            (df['close'] <= CFG_MAX_PRICE) 
        )
        sorted_df = df[condition].sort_values('bias', ascending=True)
        if sorted_df.empty: return None
        return sorted_df.iloc[0]

    # --- 主循环 ---
    st.title("⛏️ V19.2 因子挖掘机 (纯净修复版)")
    
    market_safe_map = get_market_sentiment_pure(CFG_START, CFG_END)
    cal_df = pro.trade_cal(exchange='', start_date=CFG_START, end_date=CFG_END, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    
    active_signals = [] 
    finished_signals = [] 

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Scanning: {date}")

        df_price = fetch_price_data_pure(date)
        df_strat = fetch_strategy_data_pure(date)
        
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
        
        # 1. 更新在手信号
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
                
                if (low_today - cost) / cost <= CFG_STOP_LOSS:
                    reason = "止损"
                    sell_price = cost * (1 + CFG_STOP_LOSS)
                elif peak_ret >= CFG_TRAIL_START and drawdown >= CFG_TRAIL_DROP:
                    reason = "止盈"
                    sell_price = peak * (1 - CFG_TRAIL_DROP) 
                elif (current_date_obj - pd.to_datetime(sig['buy_date'])).days >= CFG_MAX_HOLD:
                    reason = "超时"
                
                if reason:
                    ret = (sell_price - cost) / cost - CFG_FEE * 2
                    finished_signals.append({
                        'code': code, 'buy_date': sig['buy_date'],
                        'return': ret, 'reason': reason,
                        'winner_rate': sig['winner_rate'],
                        'pe_ttm': sig['pe_ttm'],
                        'turnover_rate': sig['turnover_rate'],
                        'circ_mv': sig['circ_mv']
                    })
                else:
                    signals_still_active.append(sig)
            else:
                signals_still_active.append(sig)
        
        active_signals = signals_still_active

        # 2. 选股
        if is_market_safe and not df_strat.empty:
            target_row = select_rank_1_features(df_strat.reset_index())
            if target_row is not None:
                code = target_row['ts_code']
                if code in price_map_open:
                    active_signals.append({
                        'code': code, 'buy_date': date,
                        'buy_price': price_map_open[code], 'highest': price_map_open[code],
                        'winner_rate': target_row['winner_rate'],
                        'pe_ttm': target_row['pe_ttm'],
                        'turnover_rate': target_row['turnover_rate'],
                        'circ_mv': target_row['circ_mv']
                    })

    # --- 结果展示 ---
    status_box.empty()
    st.balloons()
    
    if finished_signals:
        df_res = pd.DataFrame(finished_signals)
        df_res['return_pct'] = df_res['return'] * 100
        
        st.subheader("🔍 因子体检报告 (基于 -5% 止损)")
        st.info("观察哪个分区的胜率显著高于 40%，那就是我们要找的胜率之钥！")
        
        # 1. 获利盘
        st.divider()
        st.markdown("### 1. 获利盘 (Winner Rate)")
        bins = [-1, 1, 5, 10, 100]
        labels = ['极低 (0-1%)', '低 (1-5%)', '中 (5-10%)', '高 (>10%)']
        df_res['group'] = pd.cut(df_res['winner_rate'], bins=bins, labels=labels)
        stats = df_res.groupby('group')['return'].agg(['count', lambda x: (x>0).mean()*100, 'mean'])
        stats.columns = ['样本数', '胜率%', '期望收益%']
        stats['期望收益%'] = stats['期望收益%'] * 100
        st.table(stats)
        
        # 2. 换手率
        st.divider()
        st.markdown("### 2. 换手率 (Turnover)")
        bins_to = [0, 3, 5, 8, 100]
        labels_to = ['缩量 (<3%)', '温和 (3-5%)', '活跃 (5-8%)', '放量 (>8%)']
        df_res['group'] = pd.cut(df_res['turnover_rate'], bins=bins_to, labels=labels_to)
        stats_to = df_res.groupby('group')['return'].agg(['count', lambda x: (x>0).mean()*100, 'mean'])
        stats_to.columns = ['样本数', '胜率%', '期望收益%']
        stats_to['期望收益%'] = stats_to['期望收益%'] * 100
        st.table(stats_to)
        
        # 3. 市盈率
        st.divider()
        st.markdown("### 3. 市盈率 (PE)")
        bins_pe = [-1000, 0, 30, 60, 10000]
        labels_pe = ['亏损股 (<0)', '绩优股 (0-30)', '成长股 (30-60)', '高估 (>60)']
        df_res['group'] = pd.cut(df_res['pe_ttm'], bins=bins_pe, labels=labels_pe)
        stats_pe = df_res.groupby('group')['return'].agg(['count', lambda x: (x>0).mean()*100, 'mean'])
        stats_pe.columns = ['样本数', '胜率%', '期望收益%']
        stats_pe['期望收益%'] = stats_pe['期望收益%'] * 100
        st.table(stats_pe)
    else:
        st.warning("无交易记录")

else:
    st.info("👈 请在左侧输入 Token 并点击启动")
