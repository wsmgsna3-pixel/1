import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V18.1 黄金实验室", layout="wide")
st.title("🧪 V18.1 黄金实验室 (11-20元 专享版)")
st.markdown("""
### 🎯 验证您的“新想法”
* **基石**：已锁定 **11-20元** 黄金区间 (历史期望 +0.52%)。
* **目标**：通过调整策略参数，进一步提升 **胜率** 和 **收益率**。
* **模式**：全样本回测 (无限火力，统计每一次买卖)。
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
    
    # === 这里的参数供您测试新想法 ===
    st.subheader("💡 您的胜率优化区")
    
    stop_loss_pct = st.slider("止损线 (%)", 3, 10, 5, help="默认 -5%。放宽止损可能提高胜率？")
    trail_start_pct = st.slider("止盈启动 (%)", 5, 20, 8, help="默认 +8%。降低门槛容易成交？")
    trail_drop_pct = st.slider("回落卖出 (%)", 1, 10, 3, help="默认 3%。回撤多少就跑？")
    hold_days = st.slider("最长持股 (天)", 3, 20, 10, help="默认 10天。")

run_btn = st.button("🚀 运行实验", type="primary", use_container_width=True)

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
        # 核心锁定
        MIN_PRICE = 11.0
        MAX_PRICE = 20.0
        # 实验参数
        STOP_LOSS = - (stop_loss_pct / 100.0) - 0.0001 # 微调防止浮点
        TRAIL_START = trail_start_pct / 100.0
        TRAIL_DROP = trail_drop_pct / 100.0
        MAX_HOLD_DAYS = hold_days
        FEE_RATE = 0.0003

    cfg = Config()

    # --- 数据函数 ---
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
            (df['close'] <= cfg.MAX_PRICE) 
        )
        sorted_df = df[condition].sort_values('bias', ascending=True)
        if sorted_df.empty: return None
        return sorted_df.iloc[0]

    # --- 回测循环 ---
    market_safe_map = get_market_sentiment(cfg.START_DATE, cfg.END_DATE)
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    
    active_signals = [] 
    finished_signals = [] 

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Scanning: {date}")

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
        
        # 1. 更新信号
        signals_still_active = []
        current_date_obj = pd.to_datetime(date)
        
        for sig in active_signals:
            code = sig['code']
            if current_date_obj <= pd.to_datetime(sig['buy_date']):
                # 买入当日只更新最高价
                if code in price_map_high:
                     sig['highest'] = max(sig['highest'], price_map_high[code])
                signals_still_active.append(sig)
                continue

            if code in price_map_close:
                curr_price = price_map_close[code]
                high_today = price_map_high.get(code, curr_price)
                low_today = price_map_low.get(code, curr_price)
                
                # 更新最高价
                if high_today > sig['highest']: sig['highest'] = high_today
                
                cost = sig['buy_price']
                peak = sig['highest']
                
                # 计算动态回撤
                peak_ret = (peak - cost) / cost
                drawdown = (peak - curr_price) / peak
                
                reason = ""
                sell_price = curr_price
                
                # === 这里的逻辑决定了胜率 ===
                if (low_today - cost) / cost <= cfg.STOP_LOSS:
                    reason = "止损"
                    sell_price = cost * (1 + cfg.STOP_LOSS)
                elif peak_ret >= cfg.TRAIL_START and drawdown >= cfg.TRAIL_DROP:
                    reason = "止盈"
                    # 这里按触发回落卖出价模拟
                    sell_price = peak * (1 - cfg.TRAIL_DROP) 
                elif (current_date_obj - pd.to_datetime(sig['buy_date'])).days >= cfg.MAX_HOLD_DAYS:
                    reason = "超时"
                
                if reason:
                    # 扣手续费
                    ret = (sell_price - cost) / cost - cfg.FEE_RATE * 2
                    finished_signals.append({
                        'code': code, 
                        'buy_date': sig['buy_date'],
                        'buy_price': cost,
                        'sell_date': date,
                        'sell_price': sell_price,
                        'return': ret, 
                        'reason': reason
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
        total_virtual_ret = df_res['return'].sum() * 100
        
        st.subheader(f"🧪 实验报告 (11-20元 | {cfg.MAX_HOLD_DAYS}天)")
        
        # 1. 核心四维数据
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("单笔期望收益", f"{avg_ret:.2f}%", help="正数即为正期望")
        c2.metric("真实准确率", f"{win_rate:.1f}%", help="盈亏单比例")
        c3.metric("虚拟总收益", f"{total_virtual_ret:.1f}%", help="无复利累加")
        c4.metric("交易次数", f"{total_trades}")
        
        # 2. 分布图
        st.subheader("📊 盈亏分布")
        chart = alt.Chart(df_res).mark_bar().encode(
            x=alt.X("return_pct", bin=alt.Bin(maxbins=40), title="收益率分布 (%)"),
            y='count()',
            color=alt.condition(
                alt.datum.return_pct > 0,
                alt.value("#d32f2f"),  # 红
                alt.value("#2e7d32")   # 绿
            ),
            tooltip=['count()', 'return_pct']
        )
        st.altair_chart(chart, use_container_width=True)
        
        # 3. 详细数据 (支持下载)
        st.subheader("📝 交易流水详情")
        st.dataframe(df_res.sort_values('buy_date'), use_container_width=True)
        
        # CSV下载按钮
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 下载回测详情 CSV",
            csv,
            "11_20_experiment.csv",
            "text/csv",
            key='download-csv'
        )
