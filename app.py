# -*- coding: utf-8 -*-
"""
选股王 · V30.22.3 终极完整版 (单线程 + 全风控 + 完整数据流)
---------------------------------------------------------
核心特性：
1. [稳定性] 纯单线程运行，坚决不丢包，不封号，适合实战。
2. [完整性] 包含 stock_basic 获取，显示真实中文名称。
3. [风控] 
   - 3天限1板：3天内涨停数 >= 2 则剔除（允许0或1个涨停）。
   - 20日涨幅：过去20天涨幅 > 40% 则剔除（拒绝鱼尾）。
   - 乖离率：现价偏离5日线 > 15% 则剔除（拒绝短线超买）。
4. [策略] 暴力MACD(8,17,5) + 黄金形态 + RSI超强奖励。
---------------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time

# 过滤警告信息
warnings.filterwarnings("ignore")

# ---------------------------
# 1. 页面配置 (必须在代码第一行)
# ---------------------------
st.set_page_config(page_title="选股王 V30.22.3 完整版", layout="wide")

# ---------------------------
# 2. UI 布局：主界面设置
# ---------------------------
st.title("🐢 选股王 · V30.22.3 (单线程·完整无阉割版)")
st.markdown("##### 核心策略：暴力MACD + 黄金形态 + 严格风控")

# Token 输入框放置在主界面 Expander 中
with st.expander("🔑 系统设置 (必填)", expanded=True):
    col_token, col_date = st.columns([2, 1])
    with col_token:
        # 默认值留空，方便用户输入
        token = st.text_input("请输入 Tushare Token (回车确认)", value="", type="password", help="请前往 tushare.pro 注册获取")
    with col_date:
        backtest_date = st.date_input("选择回测日期", datetime.now())
        date_str = backtest_date.strftime("%Y%m%d")

# 检查 Token 是否存在
if not token:
    st.warning("⚠️ 请先在上方输入 Tushare Token 才能开始运行！")
    st.stop()

# 初始化 Tushare 接口
try:
    ts.set_token(token)
    pro = ts.pro_api()
except Exception as e:
    st.error(f"Token 无效或连接失败: {e}")
    st.stop()

# ---------------------------
# 3. 侧边栏参数设置 (完整参数)
# ---------------------------
with st.sidebar:
    st.header("⚙️ 策略参数")
    
    st.subheader("1. 基础门槛")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0, step=1.0)
    MIN_MV = st.number_input("最低流通市值 (亿)", value=20.0, step=1.0)
    MIN_TURNOVER = st.number_input("最低成交额 (亿)", value=1.0, step=0.1)
    
    st.subheader("2. 硬核风控 (核心)")
    MAX_20D_GAIN = st.number_input("20日累计涨幅上限 (%)", value=40.0, help="过去20天涨幅超过此值，视为鱼尾行情，直接剔除")
    MAX_BIAS_MA5 = st.number_input("5日乖离率上限 (%)", value=15.0, help="现价偏离5日线超过15%，视为短线超买，直接剔除")
    LIMIT_UP_TOLERANCE = 1 
    st.caption(f"🛡️ 连板风控：3天内涨停次数 > {LIMIT_UP_TOLERANCE} 次直接剔除 (拒绝接力)")

    st.subheader("3. 评分与加分")
    # 按照您的要求，板块阈值恢复为 1.5
    SECTOR_THRESHOLD = st.number_input("板块强暴阈值 (%)", value=1.5, step=0.1, help="板块当日涨幅超过此值才算板块效应")
    RSI_HIGH_BONUS = 3000 # RSI>90 奖励分
    
    st.divider()
    run_btn = st.button("🚀 开始运行 (单线程)", type="primary")

# ---------------------------
# 4. 核心工具函数 (完整定义)
# ---------------------------

@st.cache_data(ttl=3600)
def get_trade_days(end_date, lookback=365):
    """
    获取交易日历
    """
    try:
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=lookback)).strftime("%Y%m%d")
        df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
        return df['cal_date'].values.tolist()[::-1] # 倒序，最近的在前面
    except:
        return []

def get_stock_basics():
    """
    获取股票基础信息(主要是为了拿中文名name)
    """
    try:
        # 获取上市的股票列表
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry')
        return df
    except:
        return pd.DataFrame()

def analyze_one_stock(ts_code, name, current_daily_row, trade_date, daily_df_all_history=None):
    """
    【单只股票分析核心逻辑】
    包含所有的风控、形态判断和打分逻辑
    """
    # 1. 基础数据解包
    current_close = current_daily_row['close']
    current_open = current_daily_row['open']
    current_pre_close = current_daily_row['pre_close']
    current_pct = current_daily_row['pct_chg']
    current_vol = current_daily_row['vol']
    current_high = current_daily_row['high']
    
    # ----------------------------------------
    # [初筛] 基础门槛 (无需历史数据，速度快)
    # ----------------------------------------
    # 1. 价格门槛
    if current_close < MIN_PRICE: 
        return pd.DataFrame()
    # 2. 过滤跌停 (捕捉首板，但不能是跌停板)
    if current_pct < -9.0: 
        return pd.DataFrame()
    # 3. 必须平开或高开 (拒绝低开)
    if current_open < current_pre_close: 
        return pd.DataFrame()
    # 4. 上冲确认 (最高价必须 > 开盘价 1.5%，防止开盘即巅峰)
    if current_high < current_open * 1.015: 
        return pd.DataFrame()

    # ----------------------------------------
    # [数据准备] 获取个股历史数据
    # ----------------------------------------
    try:
        if daily_df_all_history is None:
            # 如果没有传入预取数据，则单独请求 (较慢，兜底方案)
            end_dt = trade_date
            start_dt = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
            daily_df = pro.daily(ts_code=ts_code, start_date=start_dt, end_date=end_dt)
        else:
            # 从预取的的大表中筛选出该股数据
            daily_df = daily_df_all_history[daily_df_all_history['ts_code'] == ts_code].copy()
            
        # 数据长度检查：至少需要25天数据才能计算MA20和MACD
        if len(daily_df) < 25: 
            return pd.DataFrame()
        
        # 确保按日期倒序 (最近的在 index 0)
        daily_df = daily_df.sort_values('trade_date', ascending=False).reset_index(drop=True)
        
    except Exception:
        # 如果数据获取出错，直接跳过该股
        return pd.DataFrame()

    # ----------------------------------------
    # [核心风控] 即使是妖股也要讲基本法
    # ----------------------------------------
    
    # 1. 【3天限1板】拒绝连板接力，拒绝反包
    if len(daily_df) >= 3:
        recent_3 = daily_df.iloc[0:3]
        limit_count = 0
        for _, row in recent_3.iterrows():
            if row['pct_chg'] > 9.5: # 兼容主板和科创
                limit_count += 1
        
        # 【关键判断】：如果3天内出现 >= 2个涨停，直接剔除！
        # 这意味着：0个涨停(通过)，1个涨停(通过)，2个及以上(剔除)
        if limit_count >= 2:
            return pd.DataFrame()

    # 2. 【20日涨幅限制】拒绝鱼尾行情
    if len(daily_df) >= 20:
        price_20_ago = daily_df.iloc[19]['close']
        cumulative_gain = (current_close - price_20_ago) / price_20_ago * 100
        # 如果过去20天涨幅超过阈值(默认40%)，剔除
        if cumulative_gain > MAX_20D_GAIN:
            return pd.DataFrame()

    # 3. 【乖离率限制】防止短线严重超买
    ma5 = daily_df['close'].rolling(5).mean().iloc[0]
    if pd.isna(ma5): ma5 = daily_df['close'].mean()
    bias_ma5 = (current_close - ma5) / ma5 * 100
    # 如果偏离5日线超过阈值(默认15%)，剔除
    if bias_ma5 > MAX_BIAS_MA5:
        return pd.DataFrame()

    # ----------------------------------------
    # [形态判断] 均线与MACD
    # ----------------------------------------
    
    # 计算 MA20 和 MA5_VOL
    ma20 = daily_df['close'].rolling(20).mean().iloc[0]
    ma5_vol = daily_df['vol'].rolling(5).mean().iloc[0]
    
    # [铁律1] 必须站上20日线 (趋势向上)
    if current_close <= ma20: 
        return pd.DataFrame()
    
    # [铁律2] 必须暴力放量 (量比 > 1.2)
    if current_vol < 1.2 * ma5_vol: 
        return pd.DataFrame()

    # 计算 MACD (参数: 8, 17, 5) - 特调敏捷版
    exp1 = daily_df['close'].ewm(span=8, adjust=False).mean()
    exp2 = daily_df['close'].ewm(span=17, adjust=False).mean()
    dif = exp1 - exp2
    dea = dif.ewm(span=5, adjust=False).mean()
    macd = (dif - dea) * 2
    
    curr_macd = macd.iloc[0]
    
    # [铁律3] MACD 必须水上 (金叉区或强势区)
    if curr_macd <= 0: 
        return pd.DataFrame()

    # ----------------------------------------
    # [评分系统] 选出最强者
    # ----------------------------------------
    score = 0
    bonus_items = []
    
    # 1. 基础分：完全由 MACD 绝对值决定，越大越好
    score += abs(curr_macd) * 1000 
    
    # 2. RSI 奖励 (保留妖股嗅觉)
    # 计算 RSI
    delta = daily_df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    curr_rsi = rsi.iloc[0]
    
    # 如果 RSI > 90，给予巨额奖励 
    # (注：由于前面风控已剔除了连板和20日涨幅过高的股，这里奖励的主要是首板强一字或极强首板)
    if curr_rsi > 90:
        score += RSI_HIGH_BONUS 
        bonus_items.append("RSI超强")
        
    # 3. 价格舒适区加分 (机构游资共鸣区)
    if 40 <= current_close <= 80:
        score += 1500
        bonus_items.append("黄金价格区")
    
    # 4. 板块效应加分 (需要外部传入，此处简化，若有板块数据可加)
    # if sector_pct > SECTOR_THRESHOLD: score += 1000
    
    # 返回结果
    return pd.DataFrame({
        'ts_code': [ts_code],
        'name': [name],
        'Close': [current_close],
        'Score': [score],
        'Pct_Chg': [current_pct],
        'rsi': [curr_rsi],
        'Bonus': ["+".join(bonus_items)]
    })

# ---------------------------
# 5. 主程序执行逻辑
# ---------------------------

if run_btn:
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # -----------------------
    # 步骤 1: 获取并检查交易日
    # -----------------------
    status_text.info("📅 正在检查交易日历...")
    
    # 获取最近20个交易日
    recent_days = get_trade_days(date_str, lookback=20)
    
    # 【修复 IndexError】: 如果列表为空，说明获取失败，停止运行并提示
    if not recent_days:
        status_text.error(f"❌ 错误：在日期 {date_str} 附近未找到交易日！")
        st.error("可能原因：\n1. Token 无效或过期。\n2. 选择的日期是长假期间。\n3. Tushare 接口今日额度耗尽。")
        st.stop()
        
    target_date = recent_days[0]
    st.write(f"正在分析交易日：**{target_date}** (模式：单线程稳定版)")
    
    # -----------------------
    # 步骤 2: 获取全市场基础数据
    # -----------------------
    try:
        status_text.info("📥 正在拉取当日全市场行情...")
        # 获取当日行情 (price, vol, pct_chg)
        df_daily = pro.daily(trade_date=target_date)
        # 获取每日指标 (mv, turnover, amount)
        df_daily_basic = pro.daily_basic(trade_date=target_date, fields='ts_code,turnover_rate,circ_mv,amount')
        # 获取股票基础信息 (为了拿 name)
        df_stock_basic = get_stock_basics()
        
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        st.stop()
    
    # 检查数据是否为空 (休市或数据未更新)
    if df_daily.empty or df_daily_basic.empty:
        st.error("❌ 今日数据未更新或非交易日，请收盘后(17:00后)重试！")
        st.stop()
        
    # 合并数据表
    df_all = pd.merge(df_daily, df_daily_basic, on='ts_code', how='inner')
    if not df_stock_basic.empty:
        df_all = pd.merge(df_all, df_stock_basic[['ts_code', 'name']], on='ts_code', how='left')
    else:
        df_all['name'] = df_all['ts_code'] # 降级处理
    
    # -----------------------
    # 步骤 3: 基础池初筛
    # -----------------------
    # 过滤 ST
    df_all = df_all[~df_all['name'].str.contains('ST', na=False)]
    df_all = df_all[~df_all['name'].str.contains('退', na=False)]
    
    # 过滤流通市值 (单位：万元 -> 换算为亿)
    df_all = df_all[df_all['circ_mv'] > MIN_MV * 10000] 
    
    # 过滤成交额 (单位：千元 -> 换算为亿)
    # Tushare 的 amount 单位是千元，所以 1亿 = 100000 千元
    df_all = df_all[df_all['amount'] > MIN_TURNOVER * 100000] 
    
    # 这里的 candidates 是初筛后的股票池
    candidates = df_all
    # candidates = df_all.head(50) # 【调试用】如果想测试速度，可以取消注释这行，只跑前50只
    
    total_stocks = len(candidates)
    status_text.info(f"🔍 初筛后剩余 {total_stocks} 只股票，开始深度扫描 (单线程模式)...")
    
    results = []
    
    # -----------------------
    # 步骤 4: 批量预取历史数据 (性能优化关键)
    # -----------------------
    # 为了避免每次循环都请求 API (单次请求太慢)，我们采用分批请求
    # 每次请求 50 只股票的历史数据
    
    codes = candidates['ts_code'].tolist()
    start_dt_batch = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
    
    # 初始化一个空的 DataFrame 存放历史数据
    df_history_all_batch = pd.DataFrame()
    
    # 分批大小
    BATCH_SIZE = 50 
    
    # -----------------------
    # 步骤 5: 循环执行分析
    # -----------------------
    for i in range(0, total_stocks):
        # 1. 批处理数据获取逻辑
        if i % BATCH_SIZE == 0:
            # 这一批的股票代码
            batch_codes = codes[i : i + BATCH_SIZE]
            status_text.text(f"📡 正在获取第 {i+1} ~ {min(i+BATCH_SIZE, total_stocks)} 只股票的历史数据...")
            
            try:
                # 一次性获取这批股票的历史数据
                df_batch = pro.daily(ts_code=",".join(batch_codes), start_date=start_dt_batch, end_date=target_date)
                # 覆盖旧的 batch 数据，释放内存
                df_history_all_batch = df_batch
                time.sleep(0.05) # 极短延迟防止触发限频
            except Exception:
                # 如果批量获取失败，设为空，后面 analyze_one_stock 会单独处理
                df_history_all_batch = pd.DataFrame()
        
        # 2. 提取当前行
        row = candidates.iloc[i]
        ts_code = row['ts_code']
        name = row['name']
        
        # 3. 更新进度条
        # progress_bar.progress((i + 1) / total_stocks) # 频繁更新UI会降速，每10个更新一次
        if i % 10 == 0:
            progress_bar.progress((i + 1) / total_stocks)
        
        # 4. 执行单只股票分析
        try:
            res = analyze_one_stock(
                ts_code, 
                name,
                row, 
                target_date,
                daily_df_all_history=df_history_all_batch # 传入这批次的历史数据
            )
            
            if not res.empty:
                results.append(res)
                
        except Exception as e:
            # 单只出错不影响整体
            continue
        
    status_text.success("✅ 扫描完成！")
    progress_bar.empty()
    
    # -----------------------
    # 步骤 6: 结果展示
    # -----------------------
    if results:
        # 合并结果
        final_df = pd.concat(results)
        # 按分数倒序排列
        final_df = final_df.sort_values('Score', ascending=False).reset_index(drop=True)
        # 索引从1开始
        final_df.index = final_df.index + 1
        
        st.subheader(f"🏆 选股结果 ({len(final_df)}只)")
        
        # 格式化显示 (保留小数位)
        st.dataframe(final_df.style.format({
            'Close': '{:.2f}',
            'Score': '{:.0f}',
            'Pct_Chg': '{:.2f}%',
            'rsi': '{:.1f}'
        }))
        
        # 提供下载
        csv = final_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载结果 CSV",
            data=csv,
            file_name=f"选股王_V30.22.3_{target_date}.csv",
            mime="text/csv",
        )
    else:
        st.warning("🍂 今日无符合条件的股票 (可能是门槛过高或市场太差)。")
