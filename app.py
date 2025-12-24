import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="主力锁仓V4 - 纯净版", layout="wide")
st.title("🛡️ Tushare 主力锁仓 V4 (拒绝追高版)")
st.markdown("### 核心逻辑：只做【高控盘 + 低涨幅】 | 无数据则空仓")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 核心参数")
    my_token = st.text_input("Tushare Token", type="password")
    
    # 建议回测区间
    start_date = st.text_input("开始日期", value="20241008")
    end_date = st.text_input("结束日期", value="20241130")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 100) * 10000
    
    st.markdown("---")
    st.markdown("**严控风险：**")
    stop_loss = st.slider("止损线", -10.0, -2.0, -6.0) / 100.0
    take_profit = st.slider("止盈线", 5.0, 30.0, 10.0) / 100.0
    
    run_btn = st.button("🔴 启动严格回测", use_container_width=True)

# ==========================================
# 核心逻辑
# ==========================================
if run_btn and my_token:
    ts.set_token(my_token)
    status_box = st.empty()
    debug_box = st.expander("🔍 数据诊断日志 (点击展开)", expanded=False)
    
    try:
        pro = ts.pro_api()
    except Exception as e:
        st.error(f"Token 连接失败: {e}")
        st.stop()

    class Config:
        START_DATE = start_date
        END_DATE = end_date
        INITIAL_CASH = initial_cash
        MAX_POSITIONS = 2  # 降低持仓数，集中火力
        STOP_LOSS = stop_loss
        TAKE_PROFIT = take_profit
        FEE_RATE = 0.0003

    cfg = Config()

    # --- 1. 获取交易日历 ---
    @st.cache_data(ttl=3600)
    def get_trading_days(start, end):
        try:
            df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
            return df['cal_date'].tolist()
        except:
            return []

    # --- 2. 增强型数据获取 (带Debug) ---
    def fetch_data_strict(date):
        logs = []
        try:
            # A. 基础行情
            df_daily = pro.daily(trade_date=date)
            if df_daily.empty: return pd.DataFrame(), ["当日无基础行情"]

            # B. 每日指标 (市值、换手)
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv')
            
            # C. 筹码数据 (核心)
            df_cyq = pd.DataFrame()
            try:
                df_cyq = pro.cyq_perf(trade_date=date)
            except:
                pass

            # --- 诊断逻辑 ---
            if df_cyq.empty:
                return pd.DataFrame(), [f"⚠️ {date}: 筹码接口返回为空 (可能无权限或数据未生成)"]
            
            # 检查列名 (关键修复点)
            if 'win_rate' not in df_cyq.columns:
                return pd.DataFrame(), [f"❌ {date}: 筹码数据缺少 'win_rate' 字段。现有字段: {list(df_cyq.columns)}"]

            # --- 数据合并 ---
            # 1. 合并 Daily + Basic
            df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
            
            # 2. 合并 筹码 (使用 inner merge 确保只有有筹码数据的票才会被选中)
            df_final = pd.merge(df_final, df_cyq[['ts_code', 'win_rate']], on='ts_code', how='inner')
            
            logs.append(f"✅ {date}: 成功获取数据 {len(df_final)} 条 | 筹码覆盖率 {len(df_cyq)}/{len(df_daily)}")
            return df_final, logs

        except Exception as e:
            # 最后的 fallback：修正变量名错误
            try:
                # 再次尝试合并，防止变量名未定义
                df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
                df_final = pd.merge(df_merge, df_cyq[['ts_code', 'win_rate']], on='ts_code', how='inner')
                return df_final, [f"✅ (重试成功) {date}"]
            except:
                return pd.DataFrame(), [f"❌ {date} 数据处理崩溃: {str(e)}"]

    # --- 3. 选股逻辑 (只做低吸) ---
    def select_stocks_strict(df):
        if df.empty: return []
        
        # 严苛的选股条件
        condition = (
            (df['win_rate'] >= 90) &          # 1. 获利盘 > 90% (极度控盘)
            (df['pct_chg'] > -2.0) &          # 2. 涨跌幅在 -2% 到 +3% 之间
            (df['pct_chg'] < 3.0) &           #    (拒绝追高，只埋伏)
            (df['turnover_rate'] < 5.0) &     # 3. 换手率低 (主力锁仓，散户不卖)
            (df['circ_mv'] > 300000)          # 4. 市值 > 30亿 (剔除垃圾小票)
        )
        
        selected = df[condition].copy()
        
        # 优先选 win_rate 最高的
        selected = selected.sort_values(by='win_rate', ascending=False).head(3)
        return selected['ts_code'].tolist()

    # --- 4. 回测循环 ---
    dates = get_trading_days(cfg.START_DATE, cfg.END_DATE)
    if not dates:
        st.error("无法获取交易日，请检查网络或Token")
        st.stop()

    cash = cfg.INITIAL_CASH
    positions = {}
    history = []
    trade_log = []
    
    # 统计
    valid_data_days = 0
    
    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        # UI 更新
        progress_bar.progress((i + 1) / len(dates))
        status_box.text(f"正在扫描: {date} | 当前持仓: {len(positions)} 只")
        
        # 获取数据
        df_today, logs = fetch_data_strict(date)
        if logs: 
            for log in logs: debug_box.text(log)
            if "✅" in logs[0]: valid_data_days += 1
        
        price_map = {}
        if not df_today.empty:
            price_map = df_today.set_index('ts_code')['close'].to_dict()
            
        # --- A. 卖出逻辑 (止盈止损) ---
        codes_to_del = []
        for code, pos in positions.items():
            # 如果今日有价格，更新逻辑
            curr_p = price_map.get(code, pos['cost']) # 如果停牌用成本价暂代
            
            # 如果今日实际有交易数据（能获取到价格）
            if code in price_map:
                cost = pos['cost']
                ret = (curr_p - cost) / cost
                
                reason = ""
                if ret <= cfg.STOP_LOSS: reason = "止损"
                elif ret >= cfg.TAKE_PROFIT: reason = "止盈"
                # 持仓超过8天不涨，平推走人
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
        
        for code in codes_to_del:
            del positions[code]

        # --- B. 买入逻辑 (仅当有筹码数据时) ---
        if not df_today.empty and len(positions) < cfg.MAX_POSITIONS:
            targets = select_stocks_strict(df_today)
            for code in targets:
                if code not in positions and code in price_map:
                    if len(positions) < cfg.MAX_POSITIONS:
                        price = price_map[code]
                        # 资金管理：剩余资金均分
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

        # --- C. 结算 ---
        total_asset = cash
        for code in positions:
            # 这里的价格如果是停牌，就用成本价算市值，防止资产归零
            p = price_map.get(code, positions[code]['cost'])
            total_asset += positions[code]['vol'] * p
        
        history.append({'date': pd.to_datetime(date), 'asset': total_asset})

    # ==========================================
    # 结果展示
    # ==========================================
    status_box.empty()
    st.balloons()
    
    if history:
        df_res = pd.DataFrame(history).set_index('date')
        ret_pct = (df_res['asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        
        # 结果看板
        st.subheader("📊 回测报告 (V4 严格版)")
        c1, c2, c3 = st.columns(3)
        c1.metric("区间收益率", f"{ret_pct:.2f}%", 
                  delta=f"{df_res['asset'].iloc[-1] - cfg.INITIAL_CASH:.0f} 元")
        c2.metric("有效筹码数据天数", f"{valid_data_days} / {len(dates)}")
        c3.metric("总交易次数", len(trade_log))

        st.line_chart(df_res['asset'])
        
        with st.expander("📄 查看详细交易单 (CSV)", expanded=True):
            if trade_log:
                df_log = pd.DataFrame(trade_log)
                st.dataframe(df_log)
                st.download_button("下载交易记录", df_log.to_csv().encode('utf-8'), "trade_log_v4.csv")
            else:
                st.info("区间内未触发符合严格条件的交易 (这可能是件好事，说明没有乱买)")
    else:
        st.error("数据异常，未能生成回测结果。请展开上方的诊断日志查看原因。")

elif run_btn and not my_token:
    st.error("⚠️ 请输入 Token")
