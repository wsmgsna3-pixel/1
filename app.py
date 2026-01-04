# -*- coding: utf-8 -*-
"""
选股王 · V30.26 三天买入法验证版 (The 3-Day Sniper)
核心目标：验证"D+3 趋势确立后追涨，吃 D+5 鱼尾"的可行性。
策略逻辑：
1. 选股：V30.25 Rank 1 (最强评分)。
2. 观察：D+1 开盘 到 D+3 收盘。
3. 买入：若 (D+3收盘价 / D+1开盘价 - 1) > 阈值 (如5%)，则于 D+3 收盘买入。
4. 卖出：D+5 收盘无脑卖出。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta

# ---------------------------
# 页面配置
# ---------------------------
st.set_page_config(page_title="V30.26 三天买入法验证版", layout="wide")
st.title("🧪 V30.26 三天买入法验证版 (趋势追涨测试)")
st.markdown("""
**💡 验证思路：**
* **假设：** 如果一只 Rank 1 的股票在 D+3 时，价格比 D+1 开盘价涨了 **5%~10%**，说明趋势确立，主力介入。
* **操作：** 此时 (D+3 收盘) 追进去，博弈它 D+4/D+5 的加速浪。
* **核心：** 放弃鱼头，只吃鱼尾。
""")

# ---------------------------
# 全局缓存 & 工具
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 

@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 5)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty or 'is_open' not in cal.columns: return []
    return cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)['cal_date'].head(num_days).tolist()

# ----------------------------------------------------------------------
# 数据下载
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_and_cache_daily_data(date):
    adj_df = safe_get('adj_factor', trade_date=date)
    daily_df = safe_get('daily', trade_date=date)
    return {'adj': adj_df, 'daily': daily_df}

def get_all_historical_data(trade_days_list):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list: return False
    
    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=150)).strftime("%Y%m%d")
    end_date = (datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=25)).strftime("%Y%m%d") 
    
    all_dates = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')['cal_date'].tolist()
    st.info(f"⏳ 正在下载 {start_date} 到 {end_date} 全市场数据...")

    adj_list, daily_list = [], []
    bar = st.progress(0)
    
    for i, date in enumerate(all_dates):
        try:
            cached = fetch_and_cache_daily_data(date)
            if not cached['adj'].empty: adj_list.append(cached['adj'])
            if not cached['daily'].empty: daily_list.append(cached['daily'])
            if i % 10 == 0: bar.progress((i+1)/len(all_dates))
        except: continue 
    bar.empty()

    if not adj_list or not daily_list: return False
        
    adj_data = pd.concat(adj_list)
    adj_data['adj_factor'] = pd.to_numeric(adj_data['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = adj_data.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1]) 
    
    daily_raw = pd.concat(daily_list)
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'vol']:
        if col in daily_raw.columns:
            daily_raw[col] = pd.to_numeric(daily_raw[col], errors='coerce').astype('float32')

    GLOBAL_DAILY_RAW = daily_raw.set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    
    # 缓存最新复权因子基准
    latest_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    if latest_date:
        GLOBAL_QFQ_BASE_FACTORS = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_date), 'adj_factor'].droplevel(1).to_dict()
    
    return True

def get_qfq_data(ts_code, start_date, end_date):
    # 获取复权数据
    base_adj = GLOBAL_QFQ_BASE_FACTORS.get(ts_code)
    if not base_adj: return pd.DataFrame()

    try:
        daily = GLOBAL_DAILY_RAW.loc[(ts_code, slice(start_date, end_date)), :]
        adj = GLOBAL_ADJ_FACTOR.loc[(ts_code, slice(start_date, end_date)), 'adj_factor']
    except: return pd.DataFrame()
    
    if daily.empty or adj.empty: return pd.DataFrame()
    
    df = daily.join(adj, how='left').dropna(subset=['adj_factor'])
    factor = df['adj_factor'] / base_adj
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns: df[col] = df[col] * factor
    
    df = df.reset_index()
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    return df.set_index('trade_date').sort_index()

# ----------------------------------------------------------------------
# V30.25 核心选股指标 (MACD Score)
# ----------------------------------------------------------------------
def compute_score(ts_code, end_date):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
    df = get_qfq_data(ts_code, start_date, end_date)
    if df.empty or len(df) < 26: return 0
    
    close = df['close']
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=17, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=5, adjust=False).mean()
    macd_val = (diff - dea) * 2
    
    score = (macd_val.iloc[-1] / close.iloc[-1]) * 100000
    if pd.isna(score): score = 0
    return score

# ----------------------------------------------------------------------
# 三天买入法回测逻辑
# ----------------------------------------------------------------------
def run_3day_buy_test(ts_code, signal_date, trend_threshold_pct):
    d0 = datetime.strptime(signal_date, "%Y%m%d")
    start_fut = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_fut = (d0 + timedelta(days=20)).strftime("%Y%m%d")
    
    hist = get_qfq_data(ts_code, start_fut, end_fut)
    if hist.empty or len(hist) < 3: return None # 至少要有 D1, D2, D3 数据
    
    # 获取关键价格点
    # D1 (Signal + 1天): 取开盘价
    d1_open = hist.iloc[0]['open']
    
    # D3 (Signal + 3天, 即 hist 的第 2 行，索引是 2): 取收盘价
    # 注意: iloc[0]=D1, iloc[1]=D2, iloc[2]=D3
    if len(hist) < 3: return None
    d3_close = hist.iloc[2]['close']
    d3_date = hist.index[2]
    
    # 策略判断：D3收盘价 是否大于 D1开盘价 * (1 + 阈值)
    # 用户设定 "5-10"，即涨幅 5% - 10%
    trend_pct = (d3_close / d1_open - 1) * 100
    
    if trend_pct < trend_threshold_pct:
        return None # 趋势未达标，不买入
        
    # 执行买入：D3 收盘买入
    buy_price = d3_close
    
    # 执行卖出：D5 收盘卖出 (iloc[4])
    if len(hist) >= 5:
        sell_price = hist.iloc[4]['close']
        hold_days = 2 # D3 -> D5
    else:
        # 如果没有 D5 数据(比如停牌或数据未更新)，按最后一天算
        sell_price = hist.iloc[-1]['close']
        hold_days = len(hist) - 3
        
    profit_pct = (sell_price / buy_price - 1) * 100
    
    return {
        'ts_code': ts_code,
        'D1_Open': d1_open,
        'D3_Close': d3_close,
        'Trend_Pct': trend_pct, # D1到D3的涨幅
        'Buy_Price': buy_price,
        'Sell_Price': sell_price,
        'Profit': profit_pct,
        'Hold_Days': hold_days,
        'Trade_Date': d3_date.strftime("%Y-%m-%d") # 实际买入日期
    }

# ----------------------------------------------------
# 侧边栏设置
# ----------------------------------------------------
with st.sidebar:
    st.header("1. 回测设置")
    end_date = st.date_input("结束日期", value=datetime.now().date())
    days_back = int(st.number_input("回测天数", value=50))
    
    st.markdown("---")
    st.header("2. 策略参数")
    TREND_THRESHOLD = st.number_input("D3趋势确认涨幅(%)", value=5.0, step=0.5, help="D3收盘价必须比D1开盘价高出多少才买入？")
    st.caption("建议设为 5.0 - 10.0，代表确认上涨趋势。")

    st.markdown("---")
    TS_TOKEN = st.text_input("Tushare Token", type="password")

if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api() 

# ---------------------------
# 主程序
# ---------------------------
if st.button("🚀 运行三天买入法测试"):
    dates = get_trade_days(end_date.strftime("%Y%m%d"), days_back)
    if not dates: st.stop()
    if not get_all_historical_data(dates): st.stop()
    
    st.success(f"✅ 开始验证：只买 Rank 1 | 趋势门槛 > {TREND_THRESHOLD}% | D3进 D5出")
    
    results = []
    bar = st.progress(0)
    
    for i, date in enumerate(dates):
        # 1. 模拟 V30.25 选出 Rank 1
        daily = safe_get('daily', trade_date=date)
        if daily.empty: continue
        
        # 简单粗筛
        pool = daily[daily['pct_chg'] > 0] # 只要红盘
        if len(pool) > 200: pool = pool.sort_values('pct_chg', ascending=False).head(200)
        
        best_score = -1
        rank1_code = None
        
        # 找 Rank 1
        for row in pool.itertuples():
            score = compute_score(row.ts_code, date)
            if score > best_score:
                best_score = score
                rank1_code = row.ts_code
        
        if rank1_code:
            # 2. 验证三天买入法
            res = run_3day_buy_test(rank1_code, date, TREND_THRESHOLD)
            if res:
                res['Signal_Date'] = date
                results.append(res)
                
        bar.progress((i+1)/len(dates))
    
    bar.empty()
    
    if not results:
        st.warning("没有触发任何买入信号。可能是趋势门槛太高，或市场太弱。")
        st.stop()
        
    df_res = pd.DataFrame(results)
    
    # ---------------------------
    # 结果展示
    # ---------------------------
    st.header("📊 三天买入法 (V30.26) 测试报告")
    
    col1, col2, col3, col4 = st.columns(4)
    avg_ret = df_res['Profit'].mean()
    win_rate = (df_res['Profit'] > 0).mean() * 100
    total_trades = len(df_res)
    
    # 计算简单的累计复利 (假设每次全仓)
    equity = (1 + df_res['Profit']/100).cumprod().iloc[-1] - 1
    
    col1.metric("平均收益 (2天持仓)", f"{avg_ret:.2f}%")
    col2.metric("胜率 (D3->D5)", f"{win_rate:.1f}%")
    col3.metric("总交易次数", f"{total_trades}")
    col4.metric("策略累计收益", f"{equity:.2%}")
    
    st.subheader("💡 核心洞察")
    if avg_ret > 2:
        st.success("✅ 验证成功：平均收益 > 2%，说明鱼尾效应显著，值得小资金博弈！")
    elif avg_ret > 0:
        st.warning("⚠️ 验证存疑：虽有盈利但不够厚 ( < 2% )，扣除手续费可能不划算。")
    else:
        st.error("❌ 验证失败：平均亏损，说明这是高位接盘，请谨慎！")

    st.subheader("📋 详细交易单 (D3买入 -> D5卖出)")
    st.dataframe(df_res[['Signal_Date', 'Trade_Date', 'ts_code', 'D1_Open', 'D3_Close', 'Trend_Pct', 'Profit']], use_container_width=True)
