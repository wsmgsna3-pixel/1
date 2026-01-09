# -*- coding: utf-8 -*-
"""
选股王 · V30.12.3 最终实战定制版 (风控增强版)
------------------------------------------------
版本特性 (User Customized):
1. **参数固化**：
   - 最低股价 >= 10.0 元 (厌恶低价股)
   - 上影线 <= 5.0% (最佳平衡点)
   - 实体位置 >= 0.6 (容忍洗盘)
   - 获利盘 >= 70% (激活科创板妖股)
2. **核心策略**：
   - RSI > 90 加 3000 分 (锁定主板龙头 & 科创板真龙)
   - 涨幅 > 19% 铁血剔除 (避开大面)
3. **新增风控 (2026-01-09)**：
   - 20日涨幅 < 40% (拒绝鱼尾)
   - 3天内涨停数 < 2 (拒绝连板/反包)
   - 乖离率限制 (拒绝严重超买)
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
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_INDUSTRY = {} 

# ---------------------------
# [新增] 核心风控参数设置
# ---------------------------
MAX_BIAS_MA5 = 15.0   # 5日乖离率上限：股价偏离5日线超过15%剔除
MAX_20D_GAIN = 40.0   # 20日累计涨幅上限：过去20天涨幅超过40%剔除
LIMIT_UP_TOLERANCE = 1 # 3天内允许的涨停次数：只允许1次（拒绝3天2板）

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V30.12.3 (风控版)", layout="wide")
st.title("选股王 · V30.12.3 (风控增强版)")
st.markdown("""
> **核心逻辑**：保留RSI>90的高分奖励，但强行剔除连板股和高位股。
> **风控红线**：3天内只允许1个涨停；20日涨幅不得超过40%。
""")

# ---------------------------
# 侧边栏配置
# ---------------------------
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # Tushare Token
    TOKEN = st.text_input("Tushare Token", value="你的Token在这", type="password")
    
    st.subheader("1. 基础过滤")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0, step=1.0)
    MIN_MV = st.number_input("最小流通市值 (亿)", value=20.0, step=1.0)
    MAX_MV = st.number_input("最大流通市值 (亿)", value=500.0, step=10.0)
    
    st.subheader("2. 形态参数")
    MAX_UPPER_SHADOW = st.slider("最大上影线 (%)", 0.0, 10.0, 5.0)
    MIN_BODY_POS = st.slider("实体位置 (0-1)", 0.0, 1.0, 0.6)
    
    st.subheader("3. 资金与风控")
    CHIP_MIN_WIN_RATE = st.slider("获利盘比例 (%)", 0, 100, 70)
    RSI_LIMIT = st.slider("RSI 阈值 (无实际过滤，仅打分用)", 50, 100, 90) # 这里只是UI，实际逻辑在代码里写死了
    SECTOR_THRESHOLD = st.slider("板块涨幅阈值 (%)", 0.0, 10.0, 1.0)
    MAX_PREV_PCT = 19.0 # 硬编码：昨日涨幅限制

    st.divider()
    
    # 回测设置
    st.subheader("🔙 回测模式")
    BACKTEST_MODE = st.checkbox("开启回测模式", value=False)
    BACKTEST_DAYS = st.number_input("回测天数", value=5, min_value=1, max_value=30)
    BACKTEST_END_DATE = st.date_input("回测结束日期", value=datetime.now())

# ---------------------------
# 核心功能函数
# ---------------------------

def init_tushare():
    global pro
    if TOKEN:
        ts.set_token(TOKEN)
        pro = ts.pro_api()
        return True
    return False

@st.cache_data(ttl=3600)
def get_stock_list():
    """获取全市场股票列表"""
    try:
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date,market')
        # 剔除ST、退市、北交所
        df = df[~df['name'].str.contains('ST|退')]
        df = df[~df['ts_code'].str.contains('BJ')]
        # 建立行业映射
        global GLOBAL_STOCK_INDUSTRY
        GLOBAL_STOCK_INDUSTRY = df.set_index('ts_code')['industry'].to_dict()
        return df
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return pd.DataFrame()

def get_trade_cal(end_date, n_days):
    """获取交易日历"""
    try:
        cal = pro.trade_cal(exchange='', is_open='1', end_date=end_date, limit=n_days * 2) # 多取一点缓冲
        trade_days = cal['cal_date'].tolist()
        return sorted(trade_days, reverse=True)[:n_days] # 返回最近N天
    except:
        return []

def get_daily_data_batch(trade_date, stock_list):
    """获取某日的全市场行情"""
    try:
        df = pro.daily(trade_date=trade_date, fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_chg')
        # 过滤掉停牌（没交易量的）
        df = df[df['vol'] > 0]
        return df
    except:
        return pd.DataFrame()

def get_adj_factor(ts_codes, start_date, end_date):
    """批量获取复权因子"""
    try:
        df = pro.adj_factor(ts_code=ts_codes, start_date=start_date, end_date=end_date)
        return df
    except:
        return pd.DataFrame()

def calculate_rsi(series, period=6):
    """计算RSI指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ---------------------------
# 单只股票分析核心逻辑
# ---------------------------
def analyze_one_stock(ts_code, name, target_date, daily_row, 
                      max_upper_shadow, max_turnover, min_body_pos, 
                      rsi_limit, chip_min, sector_boost, 
                      min_mv, max_mv, max_prev_pct, min_price):
    
    # 1. 基础数据解包
    current_close = daily_row['close']
    current_open = daily_row['open']
    current_high = daily_row['high']
    current_low = daily_row['low']
    current_pct = daily_row['pct_chg']
    
    # 2. 价格过滤
    if current_close < min_price: return pd.DataFrame()

    # 3. K线形态过滤
    # 实体长度
    body_len = abs(current_close - current_open)
    # 上影线长度
    upper_shadow = current_high - max(current_open, current_close)
    # 实体位置 (收盘价在当日振幅中的位置)
    if (current_high - current_low) == 0:
        body_pos = 1.0 # 一字板
    else:
        body_pos = (current_close - current_low) / (current_high - current_low)
        
    # 上影线占比
    if current_close > 0:
        upper_shadow_pct = (upper_shadow / current_close) * 100
    else:
        upper_shadow_pct = 0

    if upper_shadow_pct > max_upper_shadow: return pd.DataFrame()
    if body_pos < min_body_pos: return pd.DataFrame()

    # 4. 获取历史数据 (用于计算RSI、涨幅、连板等)
    # 我们需要往前取足够的数据来计算20日涨幅和RSI
    end_date_obj = datetime.strptime(str(target_date), "%Y%m%d")
    start_date_obj = end_date_obj - timedelta(days=60) # 多取一点
    start_date_str = start_date_obj.strftime("%Y%m%d")
    
    try:
        # 这里为了单线程稳定，每次取单只个股历史数据
        # 实际生产中最好是全量数据在内存里，但在单机简单脚本中，这样写逻辑最清晰
        daily_df = pro.daily(ts_code=ts_code, start_date=start_date_str, end_date=str(target_date))
        if daily_df.empty or len(daily_df) < 25: return pd.DataFrame() # 数据太少不算
        
        # 确保按日期倒序 (最近的在前面)
        daily_df = daily_df.sort_values('trade_date', ascending=False).reset_index(drop=True)
        
    except:
        return pd.DataFrame()

    # ============================================================
    # 🛡️ 核心风控模块 (新增逻辑) 🛡️
    # ============================================================
    
    # [风控1]：3天限1板 (严格拒绝连板)
    if len(daily_df) >= 3:
        # 取最近3天 (索引0, 1, 2)
        recent_3 = daily_df.iloc[0:3]
        limit_up_count = 0
        for _, r_row in recent_3.iterrows():
            if r_row['pct_chg'] > 9.5: # 涨幅>9.5%视为涨停
                limit_up_count += 1
        
        # 如果3天内涨停次数 >= 2，直接剔除
        if limit_up_count >= 2:
            return pd.DataFrame()

    # [风控2]：20日累计涨幅限制 (拒绝鱼尾)
    if len(daily_df) >= 20:
        price_20_days_ago = daily_df.iloc[19]['close']
        cumulative_gain = (current_close - price_20_days_ago) / price_20_days_ago * 100
        
        if cumulative_gain > MAX_20D_GAIN: # 超过40%剔除
            return pd.DataFrame()

    # [风控3]：乖离率限制 (拒绝严重超买)
    ma5 = daily_df['close'].rolling(5).mean().iloc[0] # 取最新的MA5
    if pd.isna(ma5): ma5 = daily_df['close'].mean()
    
    bias_ma5 = (current_close - ma5) / ma5 * 100
    if bias_ma5 > MAX_BIAS_MA5: # 超过15%剔除
        return pd.DataFrame()
        
    # [风控4]：昨日涨幅限制 (原逻辑保留)
    prev_pct = daily_df.iloc[1]['pct_chg']
    if prev_pct > max_prev_pct: return pd.DataFrame() # 昨天涨太多也不要(如果是19%)
    
    # ============================================================
    # 📈 打分与指标计算
    # ============================================================

    # 5. 计算 RSI
    # 需要按时间正序计算
    df_sorted = daily_df.sort_values('trade_date', ascending=True)
    df_sorted['rsi'] = calculate_rsi(df_sorted['close'], period=6)
    rsi_val = df_sorted.iloc[-1]['rsi'] # 今天的RSI
    
    if pd.isna(rsi_val): return pd.DataFrame()

    # 6. 打分系统
    score = 0
    
    # [核心] RSI > 90 暴力加分 (保持原样)
    if rsi_val > 90:
        score += 3000
    
    # 板块加分
    industry = GLOBAL_STOCK_INDUSTRY.get(ts_code, '')
    if industry in sector_boost:
        score += 1000
        is_boost = 'Yes'
    else:
        is_boost = 'No'
        
    # 获利盘 (模拟计算，这里简单用收盘价位置模拟)
    # 真实获利盘需要专用接口，这里简化逻辑：股价接近近期高点视为获利盘多
    high_60 = daily_df['high'].max()
    low_60 = daily_df['low'].min()
    if high_60 != low_60:
        win_rate = (current_close - low_60) / (high_60 - low_60) * 100
    else:
        win_rate = 50
        
    if win_rate < chip_min: return pd.DataFrame()

    # 7. 组装结果
    return pd.DataFrame([{
        'Trade_Date': target_date,
        'name': name,
        'ts_code': ts_code,
        'Close': current_close,
        'Pct_Chg': current_pct,
        'rsi': rsi_val,
        'winner_rate': win_rate,
        'Sector_Boost': is_boost,
        'Score': score
    }])

# ---------------------------
# 执行逻辑
# ---------------------------
def run_analysis(target_date):
    if not init_tushare():
        st.error("请填写 Tushare Token")
        return pd.DataFrame()

    # 1. 获取基础数据
    with st.spinner(f"正在获取 {target_date} 数据..."):
        stock_list = get_stock_list()
        daily_data = get_daily_data_batch(str(target_date), '') # 全市场数据
        
        if daily_data.empty:
            st.warning(f"{target_date} 无交易数据")
            return pd.DataFrame()

    # 2. 计算基本指标 (流通市值等)
    # 由于daily接口不含市值，需单独获取或用daily_basic
    # 为简化速度，这里假设 daily_data 已经包含 needed fields 或者我们再调一次daily_basic
    try:
        daily_basic = pro.daily_basic(trade_date=str(target_date), fields='ts_code,circ_mv,turnover_rate')
        # 合并数据
        df_merged = pd.merge(daily_data, daily_basic, on='ts_code', how='inner')
        df_merged = pd.merge(df_merged, stock_list[['ts_code', 'name']], on='ts_code', how='inner')
    except:
        st.error("获取每日指标失败")
        return pd.DataFrame()

    # 3. 初步过滤 (市值、价格)
    # 转换单位：Tushare circ_mv 单位是万，所以 20亿 = 200000
    df_filtered = df_merged[
        (df_merged['circ_mv'] >= MIN_MV * 10000) & 
        (df_merged['circ_mv'] <= MAX_MV * 10000) &
        (df_merged['close'] >= MIN_PRICE)
    ]
    
    # 4. 计算板块热度 (简单的板块涨幅平均)
    # 获取所有股票的行业
    df_filtered['industry'] = df_filtered['ts_code'].map(GLOBAL_STOCK_INDUSTRY)
    sector_perf = df_filtered.groupby('industry')['pct_chg'].mean()
    strong_sectors = sector_perf[sector_perf > SECTOR_THRESHOLD].index.tolist()

    # 5. 循环分析每只股票
    results = []
    total_stocks = len(df_filtered)
    my_bar = st.progress(0)
    
    # 为了防止请求过于频繁，我们在循环里做，或者用线程池但限制并发
    # 这里用简单的单线程循环，配合Tushare的每分钟限制，可能比较慢，但稳
    # 优化：只取前500只成交量最大的，或者按原来的全量
    # 考虑到用户脚本习惯，我们这里全量跑，但只对初筛过的跑
    
    st.info(f"初筛后剩余 {len(df_filtered)} 只股票，开始深度形态扫描...")
    
    counter = 0
    for index, row in df_filtered.iterrows():
        counter += 1
        # 每100个更新一次进度条
        if counter % 50 == 0:
            my_bar.progress(min(counter / total_stocks, 1.0))
            
        res = analyze_one_stock(
            row['ts_code'], row['name'], target_date, row,
            MAX_UPPER_SHADOW, 0, MIN_BODY_POS, 
            RSI_LIMIT, CHIP_MIN_WIN_RATE, strong_sectors,
            MIN_MV, MAX_MV, MAX_PREV_PCT, MIN_PRICE
        )
        if not res.empty:
            results.append(res)
            
    my_bar.empty()
    
    if not results:
        return pd.DataFrame()
        
    final_df = pd.concat(results)
    # 按分数排序
    final_df = final_df.sort_values('Score', ascending=False).reset_index(drop=True)
    return final_df

# ---------------------------
# 回测专用逻辑
# ---------------------------
def run_backtest_for_a_day(date, pool_df):
    # 这里需要获取 D+1, D+3, D+5 的收益
    # 假设 pool_df 已经有了当天的选股结果
    ts_codes = pool_df['ts_code'].tolist()
    if not ts_codes: return pool_df
    
    # 获取未来5天的行情
    start_dt = datetime.strptime(str(date), "%Y%m%d")
    end_dt = start_dt + timedelta(days=15) # 预留假期
    
    next_data = pro.daily(ts_code=",".join(ts_codes), start_date=start_dt.strftime("%Y%m%d"), end_date=end_dt.strftime("%Y%m%d"))
    if next_data.empty: return pool_df
    
    next_data = next_data.sort_values('trade_date')
    
    # 计算收益
    for idx, row in pool_df.iterrows():
        code = row['ts_code']
        my_data = next_data[next_data['ts_code'] == code].reset_index(drop=True)
        # 排除当天
        my_data = my_data[my_data['trade_date'] > str(date)]
        
        if len(my_data) >= 1:
            pool_df.at[idx, 'Return_D1 (%)'] = my_data.iloc[0]['pct_chg']
        if len(my_data) >= 3:
            # 简单累积涨幅：(P3 - P0)/P0 ? 或者是 pct_chg sum? 
            # 这里用每日涨幅累加近似，或者精确计算 (Close_D3 - Close_buy) / Close_buy
            # 为简单起见，这里假设以Close买入
            buy_price = row['Close']
            price_d3 = my_data.iloc[2]['close']
            pool_df.at[idx, 'Return_D3 (%)'] = (price_d3 - buy_price) / buy_price * 100
            
        if len(my_data) >= 5:
            buy_price = row['Close']
            price_d5 = my_data.iloc[4]['close']
            pool_df.at[idx, 'Return_D5 (%)'] = (price_d5 - buy_price) / buy_price * 100
            
    return pool_df

# ---------------------------
# 主程序入口
# ---------------------------
if st.button("🚀 开始选股/回测"):
    if BACKTEST_MODE:
        # 回测逻辑
        end_str = BACKTEST_END_DATE.strftime("%Y%m%d")
        trade_days = get_trade_cal(end_str, BACKTEST_DAYS)
        
        if not trade_days:
            st.error("没有交易日数据")
        else:
            st.success(f"启动回测，区间: {trade_days[-1]} 至 {trade_days[0]}")
            
            all_results = []
            
            # 倒序遍历（从旧到新），或者顺序
            # 这里按时间正序回测
            days_sorted = sorted(trade_days)
            
            for d in days_sorted:
                st.markdown(f"### 分析日期: {d}")
                daily_res = run_analysis(d)
                if not daily_res.empty:
                    # 计算未来收益
                    daily_res = run_backtest_for_a_day(d, daily_res)
                    daily_res['Trade_Date'] = d
                    all_results.append(daily_res)
                    st.dataframe(daily_res.head(5)) # 只展示前5
                else:
                    st.write("当日无符合条件股票")
            
            if all_results:
                final_all = pd.concat(all_results)
                
                # 统计
                st.header("📊 V30.12.3 统计仪表盘")
                cols = st.columns(3)
                for idx, n in enumerate([1, 3, 5]):
                    col_name = f'Return_D{n} (%)'
                    valid = final_all.dropna(subset=[col_name]) 
                    if not valid.empty:
                        avg = valid[col_name].mean()
                        win = (valid[col_name] > 0).mean() * 100
                        cols[idx].metric(f"D+{n} 均益 / 胜率", f"{avg:.2f}% / {win:.1f}%")
                
                # 导出
                csv = final_all.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 下载回测报告", csv, "backtest_report.csv", "text/csv")
                
    else:
        # 实盘模式 (只跑最新一天)
        today = datetime.now().strftime("%Y%m%d")
        # 如果是盘后，跑今天；如果是盘前，跑昨天
        # 这里简单逻辑：跑最近的一个交易日
        recent_days = get_trade_cal(today, 5)
        target_day = recent_days[0]
        
        st.markdown(f"### ⚡ 实盘扫描日期: {target_day}")
        res = run_analysis(target_day)
        
        if not res.empty:
            st.balloons()
            st.header(f"🏆 选股结果 ({len(res)} 只)")
            st.dataframe(res.style.highlight_max(axis=0))
            
            csv = res.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载今日选股结果", csv, f"stock_pick_{target_day}.csv", "text/csv")
        else:
            st.warning("今日无符合条件的股票")

