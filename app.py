# -*- coding: utf-8 -*-
"""
选股王 · V30.12.3 敏捷改进版 (MACD 8-17-5)
------------------------------------------------
版本特性 (Agile Edition):
1. **策略升级**：MACD 参数调整为 (8, 17, 5)，更灵敏捕捉起涨点。
2. **稳定并发**：2 线程下载，杜绝 Tushare 限流报错。
3. **向量化计算**：全市场矩阵计算，计算速度极快。
4. **特色数据**：利用 cyq_perf (筹码获利盘) 捕捉主升浪。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import concurrent.futures

warnings.filterwarnings("ignore")

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None
# 缓存全市场计算好的指标，避免重复计算
GLOBAL_INDICATORS = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame()
GLOBAL_STOCK_INDUSTRY = {}
GLOBAL_CHIP_DATA = {} # 筹码数据缓存

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V30.12.3 敏捷版", layout="wide")
st.title("⚡ 选股王 V30.12.3：MACD(8,17,5) 敏捷版")
st.markdown("""
**⚙️ 策略变更：**
* **MACD 参数**：由 (12,26,9) 调整为 **(8, 17, 5)**
* **逻辑**：更敏感的均线系统，旨在提前发现超短线爆发信号。
""")

# ---------------------------
# 基础 API 函数
# ---------------------------
@st.cache_data(ttl=3600*12)
def safe_get(func_name, **kwargs):
    global pro
    if pro is None:
        return pd.DataFrame()
   
    func = getattr(pro, func_name)
    try:
        return func(**kwargs)
    except Exception as e:
        time.sleep(0.5) # 稍微等待
        try:
            return func(**kwargs)
        except:
            return pd.DataFrame()

def get_trade_days(end_date_str, num_days):
    # 获取足够长的交易日历以确保指标计算（向前推 250 天）
    lookback_days = max(num_days + 250, 365)
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    
    if cal.empty or 'cal_date' not in cal.columns:
        return []
        
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    # 返回所有需要的日期（包括回测期和计算指标的缓冲期）
    return trade_days_df['cal_date'].tolist()

# --- 行业加载 ---
@st.cache_data(ttl=3600*24*7)
def load_industry_mapping():
    global pro
    if pro is None: return {}
    try:
        sw_indices = pro.index_classify(level='L1', src='SW2021')
        if sw_indices.empty: return {}
        index_codes = sw_indices['index_code'].tolist()
        all_members = []
        
        # 即使是行业获取，也限制一下并发，防止初始化就崩
        def fetch_member(idx_code):
            return safe_get('index_member', index_code=idx_code, is_new='Y')

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = executor.map(fetch_member, index_codes)
            for res in results:
                if not res.empty: all_members.append(res)
                
        if not all_members: return {}
        full_df = pd.concat(all_members)
        full_df = full_df.drop_duplicates(subset=['con_code'])
        return dict(zip(full_df['con_code'], full_df['index_code']))
    except Exception:
        return {}

# ---------------------------
# 核心：批量指标计算 (向量化) - 已修改 MACD
# ---------------------------
def calculate_all_indicators_vectorized(daily_df, adj_df):
    """
    一次性计算所有股票的 RSI, MACD, MA
    """
    st.info("⚡ 正在进行全市场向量化指标计算 (MACD 8-17-5)...")
    
    # 1. 准备数据：合并复权因子
    df = daily_df.copy()
    if not adj_df.empty:
        df = df.join(adj_df['adj_factor'])

    # 简单前复权处理计算用于指标的价格
    df['adj_factor'] = df['adj_factor'].fillna(1.0)
    df['close_calc'] = df['close'] * df['adj_factor']
    
    # 2. 按股票代码分组计算
    grouped = df.groupby(level='ts_code')
    
    # === [修改点] MACD (8, 17, 5) ===
    # 快线 8
    ema_fast = grouped['close_calc'].transform(lambda x: x.ewm(span=8, adjust=False).mean())
    # 慢线 17
    ema_slow = grouped['close_calc'].transform(lambda x: x.ewm(span=17, adjust=False).mean())
    
    df['diff'] = ema_fast - ema_slow
    # 信号线 5
    df['dea'] = df.groupby(level='ts_code')['diff'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df['macd'] = (df['diff'] - df['dea']) * 2
    
    # --- RSI (12) ---
    def calc_rsi_series(series, period=12):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))
    
    df['rsi'] = grouped['close_calc'].transform(lambda x: calc_rsi_series(x, 12))
    
    # --- MA20, MA60 ---
    df['ma20'] = grouped['close_calc'].transform(lambda x: x.rolling(window=20).mean())
    df['ma60'] = grouped['close_calc'].transform(lambda x: x.rolling(window=60).mean())
    
    # --- 实体位置 & 上影线 (基于原始 High/Low/Close 计算即可，比例不变) ---
    df['real_body_top'] = df[['open', 'close']].max(axis=1)
    df['upper_shadow_pct'] = (df['high'] - df['real_body_top']) / (df['real_body_top'] + 1e-9) * 100
    
    range_len = df['high'] - df['low']
    df['body_pos'] = (df['close'] - df['low']) / (range_len + 1e-9)
    
    return df[['close', 'pct_chg', 'rsi', 'macd', 'ma20', 'ma60', 'upper_shadow_pct', 'body_pos']]


# ---------------------------
# 数据获取核心 (双线程稳定版)
# ---------------------------
def get_all_data_and_calc(trade_days_full_list):
    global GLOBAL_DAILY_RAW, GLOBAL_INDICATORS, GLOBAL_CHIP_DATA, GLOBAL_STOCK_INDUSTRY
    
    if not trade_days_full_list: return False
    
    with st.spinner("🚀 [防限流模式] 正在拉取市场数据 (2线程)..."):
        GLOBAL_STOCK_INDUSTRY.update(load_industry_mapping())
        
        daily_list = []
        adj_list = []
        
        # 定义任务
        def fetch_daily(date):
            d = safe_get('daily', trade_date=date)
            a = safe_get('adj_factor', trade_date=date)
            return d, a

        # 改为 2 线程，极其安全
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_date = {executor.submit(fetch_daily, date): date for date in trade_days_full_list}
            
            bar = st.progress(0, text="数据同步中...")
            for i, future in enumerate(concurrent.futures.as_completed(future_to_date)):
                d, a = future.result()
                if not d.empty: daily_list.append(d)
                if not a.empty: adj_list.append(a)
                bar.progress((i+1)/len(trade_days_full_list))
            bar.empty()
            
    if not daily_list:
        st.error("数据获取失败，可能是网络问题或接口返回空")
        return False

    with st.spinner("正在构建全市场因子矩阵..."):
        # 合并数据
        daily_df = pd.concat(daily_list).drop_duplicates(subset=['ts_code', 'trade_date'])
        adj_df = pd.concat(adj_list).drop_duplicates(subset=['ts_code', 'trade_date'])
        
        # 设置索引
        daily_df = daily_df.set_index(['ts_code', 'trade_date']).sort_index()
        adj_df = adj_df.set_index(['ts_code', 'trade_date']).sort_index()
        
        GLOBAL_DAILY_RAW = daily_df # 保存原始数据用于后续查价格
        
        # 计算指标
        GLOBAL_INDICATORS = calculate_all_indicators_vectorized(daily_df, adj_df)
        
    return True

# ---------------------------
# 回测逻辑
# ---------------------------
def run_backtest_optimized(target_date, TOP_K, PARAMS):
    """
    针对单日进行筛选
    """
    global GLOBAL_INDICATORS, GLOBAL_DAILY_RAW, GLOBAL_STOCK_INDUSTRY
    
    # 1. 获取当天的截面数据
    try:
        idx = pd.IndexSlice
        today_data = GLOBAL_INDICATORS.loc[idx[:, target_date], :].reset_index(level='trade_date', drop=True)
    except KeyError:
        return pd.DataFrame(), "无当日数据"
        
    # 2. 基础过滤
    df = today_data[today_data['pct_chg'] <= PARAMS['max_prev_pct']]
    
    # 获取 daily_basic
    daily_basic = safe_get('daily_basic', trade_date=target_date, fields='ts_code,turnover_rate,circ_mv,name')
    if daily_basic.empty: return pd.DataFrame(), "无基础数据"
    
    df = df.join(daily_basic.set_index('ts_code'))
    df = df.dropna(subset=['close']) 
    
    # 市值与价格过滤
    df['circ_mv_billion'] = df['circ_mv'] / 10000
    df = df[(df['circ_mv_billion'] >= PARAMS['min_mv']) & (df['circ_mv_billion'] <= PARAMS['max_mv'])]
    df = df[df['turnover_rate'] <= PARAMS['max_turnover']]
    df = df[df['close'] >= PARAMS['min_price']]
    
    # 3. 形态风控
    df = df[df['upper_shadow_pct'] <= PARAMS['max_upper_shadow']]
    df = df[df['body_pos'] >= PARAMS['min_body_pos']]
    
    # 4. 筹码数据
    chip_df = safe_get('cyq_perf', trade_date=target_date)
    chip_map = {}
    if not chip_df.empty:
        chip_map = dict(zip(chip_df['ts_code'], chip_df['winner_rate']))
    
    candidates = []
    
    # 获取板块数据
    strong_industry_codes = set()
    try:
        sw_df = safe_get('sw_daily', trade_date=target_date)
        if not sw_df.empty:
            strong_sw = sw_df[sw_df['pct_chg'] >= PARAMS['sector_threshold']]
            strong_industry_codes = set(strong_sw['index_code'].tolist())
    except: pass
    
    # 5. 循环筛选
    for ts_code, row in df.iterrows():
        # 初始化变量
        ind_code = None
        
        # 板块过滤
        if GLOBAL_STOCK_INDUSTRY and strong_industry_codes:
            ind_code = GLOBAL_STOCK_INDUSTRY.get(ts_code)
            if ind_code and (ind_code not in strong_industry_codes): continue
            
        # 筹码过滤
        win_rate = chip_map.get(ts_code, 50)
        if win_rate < PARAMS['chip_min_win_rate']: continue
        
        # RSI 拦截
        if row['rsi'] > PARAMS['rsi_limit']: continue 
        
        # 均线多头
        if row['close'] < row['ma60']: continue
        
        # 计算得分 (注意：这里的 macd 已经是 8-17-5 的值了)
        score = row['macd'] * 1000
        if win_rate > 90: score += 1000
        if row['rsi'] > 90: score += 3000 
        
        candidates.append({
            'ts_code': ts_code,
            'name': row.get('name', ts_code),
            'Close': row['close'],
            'Pct_Chg': row['pct_chg'],
            'rsi': row['rsi'],
            'winner_rate': win_rate,
            'Score': score,
            'Sector_Boost': 'Yes' if (ind_code and ind_code in strong_industry_codes) else 'No'
        })
        
    if not candidates: return pd.DataFrame(), "无标的"
    
    final_df = pd.DataFrame(candidates).sort_values('Score', ascending=False).head(TOP_K)
    
    # 6. 计算未来收益 (通过闭包传递当前Close)
    def get_returns_safe(code, current_close):
        try:
            idx = pd.IndexSlice
            # 找到该股票在 target_date 之后的数据
            future_data = GLOBAL_DAILY_RAW.loc[idx[code, :]]
            future_data = future_data[future_data.index > target_date].head(6)
            
            if future_data.empty: return np.nan, np.nan, np.nan
            
            d1_data = future_data.iloc[0]
            
            # 一字涨停无法买入判断
            limit_ratio = 1.195 if code.startswith('688') or code.startswith('300') else 1.095
            
            # 使用 D1 的 pre_close，如果没有则用 T日的 close
            ref_close = d1_data.get('pre_close', current_close)
            if pd.isna(ref_close): ref_close = current_close
            
            if d1_data['open'] >= ref_close * limit_ratio:
                return np.nan, np.nan, np.nan # 一字板买不进
            
            # 买入价：开盘价 + 1.5% 滑点
            buy_price = d1_data['open'] * 1.015 
            
            # 确保买入价不超过涨停价
            limit_up_price = ref_close * (1.20 if limit_ratio > 1.1 else 1.10)
            if buy_price > limit_up_price:
                buy_price = limit_up_price 
                
            rets = []
            for d in [1, 3, 5]:
                if len(future_data) >= d:
                    sell_price = future_data.iloc[d-1]['close']
                    ret = (sell_price - buy_price) / buy_price * 100
                    rets.append(ret)
                else:
                    rets.append(np.nan)
            return rets
        except Exception as e:
            return np.nan, np.nan, np.nan

    # 批量计算收益
    returns = final_df.apply(lambda row: get_returns_safe(row['ts_code'], row['Close']), axis=1)
    
    if not returns.empty:
        # returns 是一个包含 list 的 Series，需要拆分
        final_df['Return_D1 (%)'] = returns.apply(lambda x: x[0])
        final_df['Return_D3 (%)'] = returns.apply(lambda x: x[1])
        final_df['Return_D5 (%)'] = returns.apply(lambda x: x[2])
    
    return final_df, None

# ---------------------------
# UI 主程序
# ---------------------------
with st.sidebar:
    st.header("⚙️ 敏捷版参数配置")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数", value=30, step=1)
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=5)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    MIN_PRICE = col1.number_input("最低股价", value=10.0)
    MIN_MV = col2.number_input("最小市值(亿)", value=50.0)
    MAX_MV = st.number_input("最大市值(亿)", value=1000.0)
    
    CHIP_MIN_WIN_RATE = st.number_input("最低获利盘 (%)", value=70.0)
    MAX_PREV_PCT = st.number_input("昨日涨幅限制 (%)", value=19.0)
    RSI_LIMIT = st.number_input("RSI 拦截线", value=100.0)
    
    SECTOR_THRESHOLD = st.number_input("板块涨幅 (%)", value=1.5)
    MAX_UPPER_SHADOW = st.number_input("上影线 (%)", value=5.0)
    MIN_BODY_POS = st.number_input("实体位置", value=0.6)
    MAX_TURNOVER_RATE = st.number_input("换手率 (%)", value=20.0)

TS_TOKEN = st.text_input("Tushare Token", type="password")

if TS_TOKEN:
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()

if st.button("🚀 启动敏捷回测"):
    if not TS_TOKEN: st.error("请输入 Token"); st.stop()
    
    full_dates = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not full_dates: st.error("日期获取失败"); st.stop()
    
    trade_dates = full_dates[:int(BACKTEST_DAYS)]
    
    # 1. 获取全量数据 (2线程)
    if not get_all_data_and_calc(full_dates): st.stop()
    
    # 2. 循环回测
    results = []
    params = {
        'min_price': MIN_PRICE, 'min_mv': MIN_MV, 'max_mv': MAX_MV,
        'chip_min_win_rate': CHIP_MIN_WIN_RATE, 'max_prev_pct': MAX_PREV_PCT,
        'rsi_limit': RSI_LIMIT, 'sector_threshold': SECTOR_THRESHOLD,
        'max_upper_shadow': MAX_UPPER_SHADOW, 'min_body_pos': MIN_BODY_POS,
        'max_turnover': MAX_TURNOVER_RATE
    }
    
    bar = st.progress(0, text="策略筛选中...")
    for i, date in enumerate(trade_dates):
        res, err = run_backtest_optimized(date, int(TOP_BACKTEST), params)
        if not res.empty:
            res['Trade_Date'] = date
            results.append(res)
        bar.progress((i+1)/len(trade_dates))
    bar.empty()
    
    if results:
        all_res = pd.concat(results)
        
        st.header("📊 V30.12.3 敏捷版仪表盘 (MACD 8-17-5)")
        
        cols = st.columns(3)
        for idx, n in enumerate([1, 3, 5]):
            col_name = f'Return_D{n} (%)'
            valid = all_res.dropna(subset=[col_name])
            if not valid.empty:
                avg = valid[col_name].mean()
                win = (valid[col_name] > 0).mean() * 100
                max_dd = valid[col_name].min()
                cols[idx].metric(f"D+{n} 均益/胜率", f"{avg:.2f}% / {win:.1f}%", f"最大回撤: {max_dd:.2f}%")

        st.dataframe(all_res, use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载回测结果 (CSV)",
            data=csv,
            file_name=f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_agile_export.csv",
            mime="text/csv",
        )
            
    else:
        st.warning("⚠️ 没有选出股票。")
