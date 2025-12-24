import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="主力锁仓V4.1", layout="wide")
st.title("🛡️ Tushare 主力锁仓系统 V4.1 (修正版)")
st.markdown("### 核心策略：寻找【获利盘 > 90%】且【未大涨】的潜伏机会")

# ==========================================
# 1. 参数设置 (依然在侧边栏，为了页面整洁)
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数配置")
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20241008")
    end_date = st.text_input("结束日期", value="20241130")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 100) * 10000
    
    st.markdown("---")
    st.markdown("**风控参数**")
    stop_loss = st.slider("止损阈值", -10.0, -2.0, -6.0) / 100.0
    take_profit = st.slider("止盈阈值", 5.0, 30.0, 10.0) / 100.0

# ==========================================
# 2. 按钮区域 (移至主页面)
# ==========================================
st.divider()
col_btn, col_info = st.columns([1, 3])
with col_btn:
    # 按钮现在非常显眼地放在这里
    run_btn = st.button("🚀 点击开始回测", type="primary", use_container_width=True)
with col_info:
    st.info("点击左侧按钮启动。程序将自动扫描【获利盘(winner_rate)】数据。")

# ==========================================
# 3. 核心逻辑
# ==========================================
if run_btn:
    if not my_token:
        st.error("请先在左侧侧边栏输入 Tushare Token！")
        st.stop()
        
    ts.set_token(my_token)
    status_box = st.empty()
    debug_expander = st.expander("🔍 实时数据日志 (点击查看)", expanded=True) # 默认展开方便看数据状态
    log_container = debug_expander.container()
    
    try:
        pro = ts.pro_api()
    except Exception as e:
        st.error(f"Token 连接失败: {e}")
        st.stop()

    class Config:
        START_DATE = start_date
        END_DATE = end_date
        INITIAL_CASH = initial_cash
        MAX_POSITIONS = 2  # 严控仓位
        STOP_LOSS = stop_loss
        TAKE_PROFIT = take_profit
        FEE_RATE = 0.0003

    cfg = Config()

    # --- 辅助函数 ---
    @st.cache_data(ttl=3600)
    def get_trading_days(start, end):
        try:
            df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
            return df['cal_date'].tolist()
        except:
            return []

    # --- 数据获取 (修复字段名 winner_rate) ---
    def fetch_data_strict(date):
        logs = []
        try:
            # 1. 基础数据
            df_daily = pro.daily(trade_date=date)
            if df_daily.empty: return pd.DataFrame(), ["无基础行情"]

            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv')
            
            # 2. 筹码数据
            df_cyq = pd.DataFrame()
            try:
                df_cyq = pro.cyq_perf(trade_date=date)
            except:
                pass

            if df_cyq.empty:
                return pd.DataFrame(), [f"⚠️ {date}: 筹码接口未返回数据"]
            
            # 关键修复：使用 winner_rate
            if 'winner_rate' not in df_cyq.columns:
                return pd.DataFrame(), [f"❌ {date}: 字段异常，可用字段: {list(df_cyq.columns)}"]

            # 3. 合并
            # 先合基本面
            df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
            # 再合筹码 (只取 ts_code 和 winner_rate)
            df_final = pd.merge(df_merge, df_cyq[['ts_code', 'winner_rate']], on='ts_code', how='inner')
            
            msg = f"✅ {date}: 数据获取成功! 共有 {len(df_final)} 只股票含筹码数据"
            return df_final, [msg]

        except Exception as e:
            return pd.DataFrame(), [f"❌ {date} 处理报错: {str(e)}"]

    # --- 选股逻辑 (使用 winner_rate) ---
    def select_stocks_strict(df):
        if df.empty: return []
        
        condition = (
            (df['winner_rate'] >= 90) &       # 获利盘 > 90%
            (df['pct_chg'] > -2.0) &          # 涨幅控制
            (df['pct_chg'] < 3.0) &           # 不追高
            (df['turnover_rate'] < 5.0) &     # 锁仓
            (df['circ_mv'] > 300000)          # 市值筛选
        )
        
        selected = df[condition].copy()
        # 排序
        selected = selected.sort_values(by='winner_rate', ascending=False).head(3)
        return selected['ts_code'].tolist()

    # --- 回测主循环 ---
    dates = get_trading_days(cfg.START_DATE, cfg.END_DATE)
    if not dates:
        st.error("日期范围内无交易日")
        st.stop()

    cash = cfg.INITIAL_CASH
    positions = {}
    history = []
    trade_log = []
    
    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        status_box.markdown(f"### 🔄 正在处理: `{date}` ...")
        
        df_today, logs = fetch_data_strict(date)
        # 实时打印日志
        for log in logs:
            if "✅" in log:
                log_container.success(log)
            elif "❌" in log:
                log_container.error(log)
            else:
                log_container.warning(log)
        
        price_map = {}
        if not df_today.empty:
            price_map = df_today.set_index('ts_code')['close'].to_dict()

        # --- 卖出逻辑 ---
        codes_to_del = []
        for code, pos in positions.items():
            if code in price_map:
                curr_p = price_map[code]
                cost = pos['cost']
                ret = (curr_p - cost) / cost
                
                reason = ""
                if ret <= cfg.STOP_LOSS: reason = "止损"
                elif ret >= cfg.TAKE_PROFIT: reason = "止盈"
                elif (pd.to_datetime(date) - pd.to_datetime(pos['date'])).days >= 8: reason = "超时平推"
                
                if reason:
                    revenue = pos['vol'] * curr_p * (1 - cfg.FEE_RATE - 0.001)
                    profit = revenue - (pos['vol'] * cost)
                    cash += revenue
                    trade_log.append({
                        'date': date, 'code': code, 'action': 'SELL', 
                        'price': curr_p, 'profit': profit, 'reason': reason
                    })
                    codes_to_del.append(code)
        
        for c in codes_to_del: del positions[c]

        # --- 买入逻辑 ---
        if not df_today.empty and len(positions) < cfg.MAX_POSITIONS:
            targets = select_stocks_strict(df_today)
            for code in targets:
                if code not in positions and code in price_map:
                    if len(positions) < cfg.MAX_POSITIONS:
                        price = price_map[code]
                        slot_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                        vol = int(slot_cash / price / 100) * 100
                        if vol > 0:
                            cost_val = vol * price * (1 + cfg.FEE_RATE)
                            if cash >= cost_val:
                                cash -= cost_val
                                positions[code] = {'cost': price, 'vol': vol, 'date': date}
                                trade_log.append({
                                    'date': date, 'code': code, 'action': 'BUY', 
                                    'price': price, 'reason': '主力锁仓'
                                })

        # --- 结算 ---
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
        
        st.divider()
        st.subheader("📊 最终回测报告")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("最终收益率", f"{ret:.2f}%", delta_color="normal" if ret > 0 else "inverse")
        c2.metric("最终资产", f"{int(final_val):,}")
        c3.metric("交易笔数", len(trade_log))
        
        st.line_chart(df_res['asset'])
        
        if trade_log:
            st.write("📋 **交易明细**")
            st.dataframe(pd.DataFrame(trade_log))
    else:
        st.error("未能生成回测结果，请检查日志。")
