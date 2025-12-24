import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="主力锁仓回测系统", layout="wide")
st.title("🚀 Tushare 10000分·主力锁仓穿透系统 (Pro版)")

# ==========================================
# 1. 侧边栏配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 策略设置")
    my_token = st.text_input("Tushare Token", type="password", help="需拥有10000积分权限")
    
    start_date = st.text_input("开始日期", value="20241101")
    end_date = st.text_input("结束日期", value="20241220")
    initial_cash = st.slider("初始资金 (万)", 10, 500, 100) * 10000
    
    run_btn = st.button("🔴 点击开始回测", use_container_width=True)

# ==========================================
# 2. 核心逻辑 (修复版)
# ==========================================

if run_btn and my_token:
    ts.set_token(my_token)
    
    # 状态显示区
    status_area = st.empty() 
    error_area = st.container() # 专门用于显示非致命错误

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
    
    # --- 辅助函数 ---
    @st.cache_data(ttl=3600)
    def get_trading_days(start, end):
        try:
            df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
            return df['cal_date'].tolist()
        except Exception as e:
            st.error(f"无法获取交易日历: {e}")
            return []

    def fetch_data_for_date(date):
        """
        修复版：增加异常处理和字段检查，防止KeyError
        """
        try:
            # 1. 获取基础行情
            df_daily = pro.daily(trade_date=date)
            if df_daily.empty:
                return pd.DataFrame() # 当天无行情数据

            # 2. 获取每日指标 (换手率、市值)
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv')
            
            # 3. 获取筹码数据 (核心高阶数据)
            # 注意：如果权限不足或数据未更新，这里可能返回空或报错
            try:
                df_cyq = pro.cyq_perf(trade_date=date)
            except:
                df_cyq = pd.DataFrame() # 获取失败则置空

            # --- 防御性合并 ---
            # 必须保证基础数据都在
            if df_basic.empty:
                return pd.DataFrame()

            # 合并行情和指标
            df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')

            # 只有当筹码数据成功获取，且包含 win_rate 时才合并
            if not df_cyq.empty and 'win_rate' in df_cyq.columns:
                df_merge = pd.merge(df_merge, df_cyq, on='ts_code', how='inner')
            else:
                # 标记该日数据缺失筹码信息，后续选股会识别
                df_merge['win_rate'] = np.nan 
                
            return df_merge

        except Exception as e:
            # 捕获所有API层面的错误，防止崩溃
            print(f"Data Fetch Error: {e}")
            return pd.DataFrame()

    def select_stocks(df):
        """
        修复版：增加字段存在性检查
        """
        if df.empty: return []

        # --- 致命检查：如果没有 win_rate 字段，说明当天无法进行筹码选股 ---
        if 'win_rate' not in df.columns or df['win_rate'].isnull().all():
            # 这种情况静默失败即可，不选股，不报错
            return []

        try:
            # 选股逻辑
            condition = (
                (df['win_rate'] >= 85) &          # 核心：85%获利盘
                (df['turnover_rate'] < 10) &      # 锁仓
                (df['turnover_rate'] > 1) &
                (df['circ_mv'] > 500000) &        # 50亿以上
                (df['pct_chg'] > 2.0)
            )
            
            selected = df[condition].copy()
            
            if not selected.empty:
                # 按获利盘比例排序
                selected = selected.sort_values(by='win_rate', ascending=False).head(3)
                return selected['ts_code'].tolist()
            else:
                return []
                
        except KeyError as e:
            # 双重保险
            return []

    # --- 回测引擎 ---
    dates = get_trading_days(cfg.START_DATE, cfg.END_DATE)
    
    if not dates:
        st.error("未获取到有效交易日，请检查日期或Token权限。")
        st.stop()

    cash = cfg.INITIAL_CASH
    positions = {}
    history_value = []
    trade_log = [] # 记录交易明细
    
    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        # UI 更新
        status_area.markdown(f"**🔄 正在回测: `{date}` ...**")
        progress_bar.progress((i + 1) / len(dates))
        
        df_today = fetch_data_for_date(date)
        
        # 建立价格字典 (如果没有数据，字典为空，不会报错，只会跳过当日逻辑)
        price_map = {}
        if not df_today.empty and 'close' in df_today.columns:
            price_map = df_today.set_index('ts_code')['close'].to_dict()
        
        # 1. 卖出检查 (止盈止损)
        codes_to_sell = []
        for code in list(positions.keys()):
            if code in price_map:
                curr_p = price_map[code]
                cost = positions[code]['cost']
                
                # 触发卖出条件
                if (curr_p - cost)/cost <= cfg.STOP_LOSS or (curr_p - cost)/cost >= cfg.TAKE_PROFIT:
                    revenue = positions[code]['vol'] * curr_p * (1 - cfg.FEE_RATE - 0.001)
                    cash += revenue
                    del positions[code]
                    trade_log.append({'date': date, 'code': code, 'action': 'SELL', 'price': curr_p})

        # 2. 买入检查
        if len(positions) < cfg.MAX_POSITIONS and not df_today.empty:
            targets = select_stocks(df_today)
            for code in targets:
                if code not in positions and code in price_map:
                    price = price_map[code]
                    if len(positions) < cfg.MAX_POSITIONS:
                        # 仓位计算
                        available_cash = cash / (cfg.MAX_POSITIONS - len(positions))
                        vol = int(available_cash / price / 100) * 100
                        
                        if vol > 0 and cash > vol * price:
                            cost_val = vol * price * (1 + cfg.FEE_RATE)
                            cash -= cost_val
                            positions[code] = {'cost': price, 'vol': vol, 'date': date}
                            trade_log.append({'date': date, 'code': code, 'action': 'BUY', 'price': price})

        # 3. 每日结算
        total_asset = cash
        for code, pos in positions.items():
            # 如果当日无价格，沿用成本价估算
            current_p = price_map.get(code, pos['cost'])
            total_asset += pos['vol'] * current_p
        
        history_value.append({'date': pd.to_datetime(date), 'total_asset': total_asset})

    # ==========================================
    # 3. 结果可视化
    # ==========================================
    status_area.empty()
    st.balloons()
    
    if history_value:
        df_res = pd.DataFrame(history_value).set_index('date')
        
        # 计算收益
        final_asset = df_res['total_asset'].iloc[-1]
        ret = (final_asset - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        
        # 指标展示
        col1, col2 = st.columns(2)
        col1.metric("最终资产", f"{int(final_asset):,}")
        col2.metric("区间收益率", f"{ret:.2f}%", delta_color="normal")
        
        st.subheader("📈 资金曲线")
        st.line_chart(df_res['total_asset'])
        
        # 交易明细
        with st.expander("查看详细交易记录"):
            if trade_log:
                st.dataframe(pd.DataFrame(trade_log))
            else:
                st.write("区间内无交易触发")
    else:
        st.warning("未能生成回测数据，可能是因为数据权限不足或日期范围内无数据。")

elif run_btn and not my_token:
    st.error("❌ 请先输入 Token")
