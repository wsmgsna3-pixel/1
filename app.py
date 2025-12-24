import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="主力锁仓回测系统", layout="wide")
st.title("🚀 Tushare 10000积分·主力锁仓穿透系统")
st.markdown("---")

# ==========================================
# 1. 侧边栏配置与输入
# ==========================================
st.sidebar.header("⚙️ 策略配置")

# 获取Token (Streamlit 方式)
my_token = st.sidebar.text_input("请输入 Tushare Token", type="password", help="您的10000积分Token")

start_date = st.sidebar.text_input("开始日期 (YYYYMMDD)", value="20241101")
end_date = st.sidebar.text_input("结束日期 (YYYYMMDD)", value="20241220")
initial_cash = st.sidebar.number_input("初始资金", value=1000000)

run_btn = st.sidebar.button("开始回测")

# ==========================================
# 2. 核心逻辑 (保持不变，适配显示)
# ==========================================

if run_btn and my_token:
    ts.set_token(my_token)
    try:
        pro = ts.pro_api()
        st.success("Token 设置成功，开始初始化数据...")
    except Exception as e:
        st.error(f"Token 无效或连接失败: {e}")
        st.stop()

    # 配置参数类
    class Config:
        START_DATE = start_date
        END_DATE = end_date
        INITIAL_CASH = initial_cash
        MAX_POSITIONS = 5
        STOP_LOSS = -0.05
        TAKE_PROFIT = 0.15
        FEE_RATE = 0.0003

    cfg = Config()
    
    # 占位符：用于实时显示日志
    log_area = st.empty()
    progress_bar = st.progress(0)

    # --- 辅助函数 ---
    @st.cache_data(ttl=3600) # 缓存数据，避免重复消耗积分
    def get_trading_days(start, end):
        df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
        return df['cal_date'].tolist()

    def fetch_data_for_date(date):
        try:
            # 每日行情
            df_daily = pro.daily(trade_date=date)
            # 每日指标
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
            # 筹码分布 (VIP接口)
            df_cyq = pro.cyq_perf(trade_date=date) 
            
            if df_daily.empty or df_cyq.empty:
                return pd.DataFrame()

            df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
            df_merge = pd.merge(df_merge, df_cyq, on='ts_code', how='inner')
            return df_merge
        except Exception as e:
            return pd.DataFrame()

    def select_stocks(df_data):
        if df_data.empty: return []
        # 选股逻辑
        condition = (
            (df_data['win_rate'] >= 85) &
            (df_data['turnover_rate'] < 10) &
            (df_data['turnover_rate'] > 1) &
            (df_data['circ_mv'] > 500000) & 
            (df_data['circ_mv'] < 8000000) &
            (df_data['pct_chg'] > 2.0) &
            (df_data['pct_chg'] < 9.5)
        )
        selected = df_data[condition].copy()
        selected = selected.sort_values(by='win_rate', ascending=False).head(3)
        return selected['ts_code'].tolist()

    # --- 回测引擎 ---
    dates = get_trading_days(cfg.START_DATE, cfg.END_DATE)
    if not dates:
        st.error("未获取到交易日数据，请检查日期范围。")
        st.stop()

    cash = cfg.INITIAL_CASH
    positions = {} 
    history_value = []
    trade_log = []
    
    logs = [] # 收集日志用于显示

    for i, date in enumerate(dates):
        # 更新进度条
        progress = (i + 1) / len(dates)
        progress_bar.progress(progress)
        
        status_text = f"正在处理: {date} ({i+1}/{len(dates)})"
        log_area.text(status_text)

        df_today = fetch_data_for_date(date)
        if df_today.empty: continue
        
        price_map = df_today.set_index('ts_code')['close'].to_dict()
        high_map = df_today.set_index('ts_code')['high'].to_dict()
        low_map = df_today.set_index('ts_code')['low'].to_dict()

        # 卖出逻辑
        codes_to_sell = []
        current_codes = list(positions.keys())
        
        for code in current_codes:
            if code not in price_map: continue
            
            cost = positions[code]['cost']
            curr_p = price_map[code]
            low_p = low_map.get(code, curr_p)
            high_p = high_map.get(code, curr_p)
            
            reason = ""
            sell_p = curr_p
            
            # 止损
            if (low_p - cost)/cost <= cfg.STOP_LOSS:
                sell_p = cost * (1 + cfg.STOP_LOSS)
                reason = "止损触发"
            # 止盈
            elif (high_p - cost)/cost >= cfg.TAKE_PROFIT:
                sell_p = cost * (1 + cfg.TAKE_PROFIT)
                reason = "止盈触发"
            # 时间止损
            else:
                d1 = datetime.strptime(positions[code]['date'], '%Y%m%d')
                d2 = datetime.strptime(date, '%Y%m%d')
                if (d2 - d1).days >= 5:
                    reason = "持仓超时"
            
            if reason:
                vol = positions[code]['vol']
                revenue = vol * sell_p * (1 - cfg.FEE_RATE - 0.001)
                cash += revenue
                profit = revenue - (vol * cost)
                del positions[code]
                trade_log.append({'date': date, 'action': 'SELL', 'code': code, 'price': sell_p, 'reason': reason, 'profit': profit})

        # 买入逻辑
        if len(positions) < cfg.MAX_POSITIONS:
            targets = select_stocks(df_today)
            for code in targets:
                if len(positions) >= cfg.MAX_POSITIONS: break
                if code in positions: continue
                
                buy_price = price_map.get(code)
                if buy_price:
                    # 资金分配
                    slot = cfg.MAX_POSITIONS - len(positions)
                    target_val = cash / slot
                    vol = int(target_val / buy_price / 100) * 100
                    
                    if vol > 0:
                        cost_val = vol * buy_price * (1 + cfg.FEE_RATE)
                        if cash >= cost_val:
                            cash -= cost_val
                            positions[code] = {'cost': buy_price, 'vol': vol, 'date': date}
                            trade_log.append({'date': date, 'action': 'BUY', 'code': code, 'price': buy_price})

        # 结算
        total_asset = cash
        for code, pos in positions.items():
            current_p = price_map.get(code, pos['cost'])
            total_asset += pos['vol'] * current_p
        
        history_value.append({'date': pd.to_datetime(date), 'total_asset': total_asset})

    # ==========================================
    # 3. 结果展示
    # ==========================================
    st.success("回测完成！")
    
    # 数据处理
    df_res = pd.DataFrame(history_value).set_index('date')
    df_res['peak'] = df_res['total_asset'].cummax()
    df_res['drawdown'] = (df_res['total_asset'] - df_res['peak']) / df_res['peak']
    
    total_ret = (df_res['total_asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
    max_dd = df_res['drawdown'].min() * 100
    
    # 指标卡片
    col1, col2, col3 = st.columns(3)
    col1.metric("总收益率", f"{total_ret:.2f}%")
    col2.metric("最大回撤", f"{max_dd:.2f}%")
    col3.metric("总交易次数", len(trade_log))

    # 图表绘制 (Streamlit 原生图表)
    st.subheader("📈 资产曲线")
    st.line_chart(df_res['total_asset'])
    
    st.subheader("📉 回撤曲线")
    st.area_chart(df_res['drawdown'])

    # 交易记录表
    st.subheader("📋 交易明细")
    if trade_log:
        df_log = pd.DataFrame(trade_log)
        st.dataframe(df_log)
    else:
        st.write("无交易记录")

elif run_btn and not my_token:
    st.warning("⚠️ 请先在侧边栏输入 Tushare Token")

