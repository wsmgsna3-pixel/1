import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="自适应回测系统", layout="wide")
st.title("🚀 Tushare 智能双模回测系统 (Pro)")
st.caption("自动检测数据权限 | 筹码/动量双策略自动切换")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    my_token = st.text_input("Tushare Token", type="password")
    
    # 默认选一个稍微早一点的时间段，数据更全
    start_date = st.text_input("开始日期", value="20241008")
    end_date = st.text_input("结束日期", value="20241130")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 100) * 10000
    run_btn = st.button("🔴 点击开始回测", use_container_width=True)

# ==========================================
# 核心逻辑
# ==========================================
if run_btn and my_token:
    ts.set_token(my_token)
    status_area = st.empty()
    
    try:
        pro = ts.pro_api()
    except Exception as e:
        st.error(f"Token 连接失败: {e}")
        st.stop()

    class Config:
        START_DATE = start_date
        END_DATE = end_date
        INITIAL_CASH = initial_cash
        MAX_POSITIONS = 3
        STOP_LOSS = -0.05
        TAKE_PROFIT = 0.15
        FEE_RATE = 0.0003

    cfg = Config()

    # --- 数据获取函数 ---
    @st.cache_data(ttl=3600)
    def get_trading_days(start, end):
        try:
            df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
            return df['cal_date'].tolist()
        except:
            return []

    def fetch_data_debug(date):
        """
        带有诊断功能的数据获取
        """
        data_status = {'daily': False, 'cyq': False}
        
        try:
            # 1. 基础行情
            df_daily = pro.daily(trade_date=date)
            if not df_daily.empty: data_status['daily'] = True
            
            # 2. 每日指标
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')

            # 3. 尝试获取筹码数据
            df_cyq = pd.DataFrame()
            try:
                df_cyq = pro.cyq_perf(trade_date=date)
                if not df_cyq.empty: data_status['cyq'] = True
            except:
                pass # 接口报错则忽略

            # 合并逻辑
            if df_daily.empty or df_basic.empty:
                return pd.DataFrame(), data_status

            df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
            
            # 只有当筹码数据存在时才合并
            if not df_cyq.empty:
                df_merge = pd.merge(df_merge, df_cyq, on='ts_code', how='inner')
            
            return df_merge, data_status

        except Exception as e:
            return pd.DataFrame(), data_status

    def select_stocks_adaptive(df, use_cyq_strategy):
        """
        自适应选股：根据数据情况自动切换策略
        """
        if df.empty: return []
        
        selected = pd.DataFrame()

        if use_cyq_strategy and 'win_rate' in df.columns:
            # === 策略 A: 筹码穿透 (VIP模式) ===
            condition = (
                (df['win_rate'] >= 85) &          # 主力高控盘
                (df['turnover_rate'] < 8) &       # 缩量
                (df['circ_mv'] > 300000) &        # 30亿以上
                (df['pct_chg'] > 2.0)
            )
            selected = df[condition].sort_values('win_rate', ascending=False)
        else:
            # === 策略 B: 量价动量 (备用模式) ===
            # 逻辑：中盘股 + 底部放量启动 + 估值合理
            condition = (
                (df['pct_chg'] > 4.0) &           # 强势启动
                (df['pct_chg'] < 9.5) &           # 非一字板
                (df['turnover_rate'] > 3) &       # 换手活跃
                (df['turnover_rate'] < 12) &      # 非死亡换手
                (df['pe_ttm'] > 0) & (df['pe_ttm'] < 60) & # 剔除亏损和高估
                (df['circ_mv'] > 500000)          # 50亿以上
            )
            selected = df[condition].sort_values('pct_chg', ascending=False)
            
        return selected.head(3)['ts_code'].tolist()

    # --- 回测循环 ---
    dates = get_trading_days(cfg.START_DATE, cfg.END_DATE)
    if not dates:
        st.error("无法获取交易日，请检查日期或网络")
        st.stop()

    cash = cfg.INITIAL_CASH
    positions = {}
    history = []
    trade_log = []
    
    # 统计数据质量
    cyq_days = 0
    total_days = 0
    
    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        # 1. 获取数据与诊断
        df_today, status = fetch_data_debug(date)
        
        # UI 反馈
        mode_text = "🔥 筹码模式" if status['cyq'] else "🛡️ 备用模式"
        if status['cyq']: cyq_days += 1
        total_days += 1
        
        status_area.markdown(f"""
        **进度**: {date} ({i+1}/{len(dates)})
        **数据状态**: 行情 {'✅' if status['daily'] else '❌'} | 筹码 {'✅' if status['cyq'] else '❌'}
        **当前策略**: {mode_text}
        """)
        progress_bar.progress((i + 1) / len(dates))
        
        if df_today.empty: continue
        
        price_map = {}
        if 'close' in df_today.columns:
            price_map = df_today.set_index('ts_code')['close'].to_dict()
            
        # 2. 卖出
        for code in list(positions.keys()):
            if code in price_map:
                curr_p = price_map[code]
                cost = positions[code]['cost']
                
                reason = ""
                if (curr_p - cost)/cost <= cfg.STOP_LOSS: reason = "止损"
                elif (curr_p - cost)/cost >= cfg.TAKE_PROFIT: reason = "止盈"
                elif (pd.to_datetime(date) - pd.to_datetime(positions[code]['date'])).days >= 5: reason = "超时"
                
                if reason:
                    revenue = positions[code]['vol'] * curr_p * (1 - cfg.FEE_RATE)
                    profit = revenue - (positions[code]['vol'] * cost)
                    cash += revenue
                    del positions[code]
                    trade_log.append({'date': date, 'code': code, 'action': 'SELL', 'price': curr_p, 'profit': profit, 'reason': reason})

        # 3. 买入 (根据 win_rate 是否存在，自动选择策略)
        if len(positions) < cfg.MAX_POSITIONS:
            # 自动判断是否使用筹码策略
            use_cyq = ('win_rate' in df_today.columns)
            targets = select_stocks_adaptive(df_today, use_cyq)
            
            for code in targets:
                if code not in positions and code in price_map:
                    if len(positions) < cfg.MAX_POSITIONS:
                        price = price_map[code]
                        money_per_pos = cash / (cfg.MAX_POSITIONS - len(positions))
                        vol = int(money_per_pos / price / 100) * 100
                        if vol > 0:
                            cost = vol * price * (1 + cfg.FEE_RATE)
                            if cash >= cost:
                                cash -= cost
                                positions[code] = {'cost': price, 'vol': vol, 'date': date}
                                strat_name = "筹码" if use_cyq else "备用"
                                trade_log.append({'date': date, 'code': code, 'action': 'BUY', 'price': price, 'reason': strat_name})

        # 结算
        total_asset = cash
        for code in positions:
            total_asset += positions[code]['vol'] * price_map.get(code, positions[code]['cost'])
        history.append({'date': pd.to_datetime(date), 'asset': total_asset})

    # ==========================================
    # 结果展示
    # ==========================================
    status_area.empty()
    st.balloons()
    
    # 诊断报告
    st.info(f"📊 回测诊断报告：共 {total_days} 个交易日，其中 {cyq_days} 天成功获取 VIP 筹码数据。")
    if cyq_days == 0:
        st.warning("⚠️ 警告：全程未获取到筹码数据，系统已完全运行在【备用模式】。请检查积分权限或接口配额。")

    if history:
        df_res = pd.DataFrame(history).set_index('date')
        ret = (df_res['asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("最终收益", f"{ret:.2f}%")
        c2.metric("交易次数", len(trade_log))
        c3.metric("当前持仓", len(positions))

        st.subheader("资金曲线")
        st.line_chart(df_res['asset'])
        
        with st.expander("查看详细交易单"):
            st.dataframe(pd.DataFrame(trade_log))
    else:
        st.error("数据完全空白，请检查Token或日期范围。")

elif run_btn and not my_token:
    st.error("请输入 Token")
