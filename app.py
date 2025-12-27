import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import altair as alt

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V19.0 因子挖掘", layout="wide")
st.title("⛏️ V19.0 因子挖掘机 (寻找胜率之钥)")
st.markdown("""
### 🔍 寻找“X因子”
我们保持 **-5% 窄止损** (保护心态)，尝试通过添加 **过滤条件** 来提升胜率。
我们将测试以下四大金刚对胜率的影响：
1.  **获利盘 (Winner Rate)**: 筹码结构是否健康？
2.  **换手率 (Turnover)**: 人气是否还在？
3.  **市盈率 (PE)**: 是错杀绩优股还是垃圾股？
4.  **流通市值 (MV)**: 盘子大小的影响？
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
    st.info("🛡️ 止损: 固定 -5% (回归人性)")
    
    # 基础参数
    STOP_LOSS = -0.0501
    TRAIL_START = 0.08
    TRAIL_DROP = 0.03
    MAX_HOLD_DAYS = 10

run_btn = st.button("🚀 启动因子扫描", type="primary", use_container_width=True)

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
        STOP_LOSS = STOP_LOSS
        TRAIL_START = TRAIL_START
        TRAIL_DROP = TRAIL_DROP
        MAX_HOLD_DAYS = MAX_HOLD_DAYS
        FEE_RATE = 0.0003

    cfg = Config()

    # --- 缓存函数 ---
    @st.cache_data(ttl=86400)
    def get_market_sentiment_v19(start, end):
        try:
            real_start = (pd.to_datetime(start) - pd.Timedelta(days=90)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code='000001.SH', start_date=real_start, end_date=end)
            df = df.sort_values('trade_date', ascending=True)
            df['ma20'] = df['close'].rolling(20).mean()
            return df.set_index('trade_date')['close'].gt(df.set_index('trade_date')['ma20']).to_dict()
        except: return {}

    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_price_data_v19(date):
        try: return pro.daily(trade_date=date)
        except: return pd.DataFrame()

    @st.cache_data(ttl=86400, persist=True, show_spinner=False)
    def fetch_strategy_data_v19(date):
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
            # 暂时放宽 winner_rate 限制，以便测试它的分布
            (df['circ_mv'] > 300000) &  
            (df['turnover_rate'] > 1.5) &
            (df['close'] >= cfg.MIN_PRICE) &
            (df['close'] <= cfg.MAX_PRICE) 
        )
        sorted_df = df[condition].sort_values('bias', ascending=True)
        if sorted_df.empty: return None
        return sorted_df.iloc[0] # 返回 Series，包含所有特征

    # --- 回测循环 ---
    market_safe_map = get_market_sentiment_v19(cfg.START_DATE, cfg.END_DATE)
    cal_df = pro.trade_cal(exchange='', start_date=cfg.START_DATE, end_date=cfg.END_DATE, is_open='1')
    dates = sorted(cal_df['cal_date'].tolist())
    
    active_signals = [] 
    finished_signals = [] 

    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        is_market_safe = market_safe_map.get(date, False) 
        status_box.text(f"Mining Factors: {date}")

        df_price = fetch_price_data_v19(date)
        df_strat = fetch_strategy_data_v19(date)
        
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
                    # === 保存因子数据 ===
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

        # 2. 发出新信号
        if is_market_safe and not df_strat.empty:
            target_row = select_rank_1_features(df_strat.reset_index())
            if target_row is not None:
                code = target_row['ts_code']
                if code in price_map_open:
                    active_signals.append({
                        'code': code, 'buy_date': date,
                        'buy_price': price_map_open[code], 'highest': price_map_open[code],
                        # 记录买入时的身体指标
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
        df_res['is_win'] = df_res['return'] > 0
        
        base_win_rate = df_res['is_win'].mean() * 100
        
        st.subheader(f"📊 基础胜率 (止损 -5%): {base_win_rate:.1f}%")
        st.write("让我们看看能不能通过过滤因子把胜率提上去！")
        
        # === 因子 1: 获利盘 (Chip) ===
        st.divider()
        st.subheader("1. 获利盘 (Winner Rate) 分析")
        c1, c2 = st.columns(2)
        # 将获利盘分桶：0-1%, 1-5%, 5-10%, >10%
        bins = [-1, 1, 5, 10, 100]
        labels = ['极低 (0-1%)', '低 (1-5%)', '中 (5-10%)', '高 (>10%)']
        df_res['chip_group'] = pd.cut(df_res['winner_rate'], bins=bins, labels=labels)
        chip_stats = df_res.groupby('chip_group').apply(lambda x: pd.Series({
            '胜率': (x['return']>0).mean()*100, 
            '样本数': len(x),
            '期望收益': x['return'].mean()*100
        }))
        c1.table(chip_stats)
        c2.info("💡 假设：获利盘太低(0-1%)可能是‘死鱼’；稍微高一点(>5%)可能有资金护盘。")

        # === 因子 2: 换手率 (Turnover) ===
        st.divider()
        st.subheader("2. 换手率 (Turnover) 分析")
        c1, c2 = st.columns(2)
        bins_to = [0, 3, 5, 8, 100]
        labels_to = ['缩量 (<3%)', '温和 (3-5%)', '活跃 (5-8%)', '放量 (>8%)']
        df_res['turnover_group'] = pd.cut(df_res['turnover_rate'], bins=bins_to, labels=labels_to)
        to_stats = df_res.groupby('turnover_group').apply(lambda x: pd.Series({
            '胜率': (x['return']>0).mean()*100, 
            '样本数': len(x),
            '期望收益': x['return'].mean()*100
        }))
        c1.table(to_stats)
        c2.info("💡 假设：Rank 1 如果伴随‘缩量’ (<3%)，可能跌不动了；如果‘巨量’，可能还在出货。")
        
        # === 因子 3: 市盈率 (PE) ===
        st.divider()
        st.subheader("3. 估值 (PE) 分析")
        c1, c2 = st.columns(2)
        bins_pe = [-1000, 0, 30, 60, 10000]
        labels_pe = ['亏损股 (<0)', '绩优股 (0-30)', '成长股 (30-60)', '高估/泡沫 (>60)']
        df_res['pe_group'] = pd.cut(df_res['pe_ttm'], bins=bins_pe, labels=labels_pe)
        pe_stats = df_res.groupby('pe_group').apply(lambda x: pd.Series({
            '胜率': (x['return']>0).mean()*100, 
            '样本数': len(x),
            '期望收益': x['return'].mean()*100
        }))
        c1.table(pe_stats)
        c2.info("💡 假设：亏损股的反弹可能是‘诈尸’，胜率低；绩优股的反弹可能是‘错杀修复’。")

        # === 智能推荐 ===
        st.divider()
        st.subheader("🤖 AI 策略优化建议")
        best_filter = ""
        best_win_rate = 0
        
        # 简单的遍历寻找最佳单因子
        for g_name, stats in [('获利盘', chip_stats), ('换手率', to_stats), ('PE', pe_stats)]:
            for idx, row in stats.iterrows():
                if row['样本数'] > 20 and row['胜率'] > best_win_rate:
                    best_win_rate = row['胜率']
                    best_filter = f"{g_name} 为 {idx}"
        
        if best_win_rate > 50:
            st.success(f"🎉 发现潜力！如果只做 【{best_filter}】 的股票，胜率可达 {best_win_rate:.1f}%！")
            st.markdown(f"建议您在实盘代码中加入这个过滤条件，即可在 **-5% 止损** 下实现正收益。")
        else:
            st.warning(f"即便加了过滤，最高胜率也只有 {best_win_rate:.1f}%。可能 Rank 1 策略本身确实太激进了。")
