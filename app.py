import streamlit as st
import subprocess
import sys
import os

# ==========================================
# 0. 手机端/云端自动环境配置 (核心修复)
# ==========================================
# 检测并自动安装缺失的库，免去配置 requirements.txt 的麻烦
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import tushare as ts
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    st.warning("正在初始化云端环境，自动安装必要的量化库...（首次运行需要约30秒）")
    packages = ["tushare", "pandas", "numpy", "matplotlib"]
    for p in packages:
        try:
            __import__(p)
        except ImportError:
            install(p)
    st.success("环境安装完成！正在加载策略...")
    # 重新加载页面以应用新库
    st.rerun()

# 再次导入（确保安装后能引用）
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# 页面配置 (适配手机竖屏)
# ==========================================
st.set_page_config(page_title="主力锁仓系统", layout="wide")

# 手机端标题优化
st.title("🚀 Tushare 10000分·主力锁仓")
st.caption("检测到手机端/云端环境，已启用自动配置模式")

# ==========================================
# 1. 侧边栏配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 策略设置")
    # 密码框输入Token
    my_token = st.text_input("Tushare Token", type="password")
    
    start_date = st.text_input("开始日期", value="20241101")
    end_date = st.text_input("结束日期", value="20241220")
    
    # 手机屏幕小，用滑块更方便
    initial_cash = st.slider("初始资金 (万)", 10, 500, 100) * 10000
    
    run_btn = st.button("🔴 点击开始回测", use_container_width=True)

# ==========================================
# 2. 核心逻辑
# ==========================================

if run_btn and my_token:
    ts.set_token(my_token)
    status_area = st.empty() # 状态显示区
    
    try:
        pro = ts.pro_api()
    except Exception as e:
        st.error(f"Token错误: {e}")
        st.stop()

    # 配置参数
    class Config:
        START_DATE = start_date
        END_DATE = end_date
        INITIAL_CASH = initial_cash
        MAX_POSITIONS = 3 # 手机端建议持仓少一点，方便看
        STOP_LOSS = -0.05
        TAKE_PROFIT = 0.15
        FEE_RATE = 0.0003

    cfg = Config()
    
    # 缓存数据函数
    @st.cache_data(ttl=3600)
    def get_trading_days(start, end):
        try:
            df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
            return df['cal_date'].tolist()
        except:
            return []

    def fetch_data_for_date(date):
        try:
            df_daily = pro.daily(trade_date=date)
            # 10000积分核心数据：筹码胜率
            df_cyq = pro.cyq_perf(trade_date=date) 
            df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv')
            
            if df_daily.empty or df_cyq.empty: return pd.DataFrame()

            df = pd.merge(df_daily, df_cyq, on='ts_code')
            df = pd.merge(df, df_basic, on='ts_code')
            return df
        except:
            return pd.DataFrame()

    # 选股逻辑
    def select_stocks(df):
        if df.empty: return []
        condition = (
            (df['win_rate'] >= 85) &          # 核心：85%获利盘
            (df['turnover_rate'] < 10) &      # 锁仓
            (df['turnover_rate'] > 1) &
            (df['circ_mv'] > 500000) &        # 50亿以上
            (df['pct_chg'] > 2.0)
        )
        sel = df[condition].sort_values('win_rate', ascending=False).head(3)
        return sel['ts_code'].tolist()

    # --- 执行回测 ---
    dates = get_trading_days(cfg.START_DATE, cfg.END_DATE)
    if not dates:
        st.error("日期范围内无交易日或接口报错")
        st.stop()

    cash = cfg.INITIAL_CASH
    positions = {}
    history = []
    logs = []
    
    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        status_area.info(f"正在回测: {date} ...")
        progress_bar.progress((i + 1) / len(dates))
        
        df_today = fetch_data_for_date(date)
        if df_today.empty: continue
        
        # 简化版回测引擎
        price_map = df_today.set_index('ts_code')['close'].to_dict()
        
        # 1. 卖出检查
        for code in list(positions.keys()):
            if code in price_map:
                curr_p = price_map[code]
                cost = positions[code]['cost']
                # 简单止盈止损
                if (curr_p - cost)/cost <= cfg.STOP_LOSS or (curr_p - cost)/cost >= cfg.TAKE_PROFIT:
                    cash += positions[code]['vol'] * curr_p
                    del positions[code]
        
        # 2. 买入检查
        if len(positions) < cfg.MAX_POSITIONS:
            targets = select_stocks(df_today)
            for code in targets:
                if code not in positions and code in price_map:
                    price = price_map[code]
                    if len(positions) < cfg.MAX_POSITIONS:
                        vol = int((cash / (cfg.MAX_POSITIONS - len(positions))) / price / 100) * 100
                        if vol > 0:
                            cash -= vol * price * (1.0003)
                            positions[code] = {'cost': price, 'vol': vol}

        # 3. 结算
        total = cash
        for code in positions:
            total += positions[code]['vol'] * price_map.get(code, positions[code]['cost'])
        history.append({'date': date, 'asset': total})

    # ==========================================
    # 3. 手机端适配结果展示
    # ==========================================
    status_area.empty() # 清除加载提示
    st.balloons() # 庆祝完成
    
    if history:
        df_res = pd.DataFrame(history)
        df_res['date'] = pd.to_datetime(df_res['date'])
        df_res.set_index('date', inplace=True)
        
        ret = (df_res['asset'].iloc[-1] - cfg.INITIAL_CASH) / cfg.INITIAL_CASH * 100
        
        st.metric("最终收益率", f"🔥 {ret:.2f}%")
        
        st.subheader("资产曲线")
        # 使用 Streamlit 原生图表，手机交互更友好
        st.line_chart(df_res['asset'])
    else:
        st.warning("该时间段无数据或未触发交易")

elif run_btn and not my_token:
    st.error("请先在左侧输入 Token")
