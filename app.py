import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# 定义版本号
APP_VERSION = "V14" 

# Tushare 单次查询限制 (用于分页循环)
LIMIT_SIZE = 5000 

# ==========================================
# 1. 页面配置与工具函数
# ==========================================
st.set_page_config(page_title=f"趋势接力选股器 {APP_VERSION}", layout="wide", page_icon="📈")

# 自定义CSS以优化手机端体验
st.markdown("""
    <style>
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white;}
    .reportview-container .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1 {font-size: 1.5rem;}
    h2 {font-size: 1.2rem;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_tushare(token):
    """
    初始化 Tushare 客户端，设置更长的超时时间来解决网络连接问题。
    """
    try:
        ts.set_token(token)
        # 设置 timeout=30 秒
        return ts.pro_api(timeout=30) 
    except Exception as e:
        st.error(f"Token 设置失败: {e}")
        return None

# ==========================================
# 2. 核心数据获取逻辑 (V14: Tushare 分页循环查询)
# ==========================================

@st.cache_data(show_spinner=False) 
def get_stock_basic_data(_pro):
    """
    独立函数：获取全市场股票代码列表 (V14)
    """
    try:
        df_basic = _pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,market,list_date')
        if df_basic.empty:
             st.error("【错误定位 A】: pro.stock_basic 接口返回为空。Token 可能无法访问基础信息。")
        return df_basic
    except Exception as e:
        st.error(f"【致命错误】: pro.stock_basic 接口调用失败。请检查 Token。\n错误详情：{e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600) # 缓存1小时
def get_base_pool(token_input):
    """
    V14 核心：使用 Tushare 分页（limit/offset）查询 daily_basic 全量最新数据。
    """
    pro = init_tushare(token_input)
    if not pro: return pd.DataFrame(), "" 

    status_text = st.empty()
    status_text.info("正在建立连接，获取全市场基础数据...")

    # --- 尝试获取基础数据和交易日历 ---
    max_retries = 3
    df_basic_filtered, trade_date = pd.DataFrame(), ""

    for attempt in range(max_retries):
        try:
            # 1. 获取交易日历
            cal = pro.trade_cal(exchange='', is_open='1', end_date=datetime.now().strftime('%Y%m%d'), fields='cal_date')
            trade_date = cal['cal_date'].values[-1]
            
            # 2. 获取基础信息并排除 ST/北交所
            df_basic = get_stock_basic_data(pro)
            if df_basic.empty:
                return pd.DataFrame(), ""

            df_basic_filtered = df_basic[~df_basic['market'].str.contains('北|BJE', na=False)] 
            df_basic_filtered = df_basic_filtered[~df_basic_filtered['name'].str.contains('ST|退', na=False)]
            
            if df_basic_filtered.empty:
                status_text.error("【错误定位 B】: 基础代码池过滤后为空。")
                return pd.DataFrame(), ""

            # 3. V14 核心逻辑：分页查询 daily_basic
            status_text.info(f"基础代码池包含 {len(df_basic_filtered)} 只股票。正在分页获取最新日线指标...")
            
            df_daily_basic_chunks = []
            offset = 0
            
            while True:
                daily_basic_fields = 'ts_code,close,turnover_rate,total_mv,circ_mv'
                
                # 使用 limit 和 offset 进行分页查询
                chunk_df = pro.daily_basic(
                    trade_date=trade_date, 
                    fields=daily_basic_fields,
                    limit=LIMIT_SIZE, 
                    offset=offset
                )
                
                if chunk_df is None or chunk_df.empty:
                    # 退出循环的条件：API 返回空数据
                    break
                    
                df_daily_basic_chunks.append(chunk_df)
                status_text.info(f"已获取到 {offset + len(chunk_df)} 条数据...")
                
                # 如果返回的数据量小于限制，说明是最后一页
                if len(chunk_df) < LIMIT_SIZE:
                    break
                
                offset += LIMIT_SIZE
                time.sleep(1.2) # 避免 API 频率超限

            # 合并所有批次的数据
            if not df_daily_basic_chunks:
                status_text.error("【错误定位 C】: pro.daily_basic 分页查询未返回任何最新日线数据。请确认 Token 权限。")
                return pd.DataFrame(), ""
                
            df_daily_data = pd.concat(df_daily_basic_chunks, ignore_index=True)
            
            # 报告实际从 Tushare 获取到数据的数量
            st.info(f"Tushare 实际返回【最新价格/市值】数据的股票数量：{len(df_daily_data)} 只。")

            # 4. 整合数据
            # 只有同时拥有基础信息和最新价格的股票才能留下
            df = pd.merge(df_basic_filtered, df_daily_data, on='ts_code', how='inner', suffixes=('_basic', '_daily'))

            break
        except Exception as e:
            if attempt < max_retries - 1:
                status_text.warning(f"获取数据失败，正在重试 ({attempt+1}/{max_retries})...")
                time.sleep(2) 
            else:
                status_text.error(f"【致命错误】: 在获取最新日线指标过程中发生异常。\n错误详情：{e}")
                return pd.DataFrame(), ""
    
    # --- 核心数据清洗 ---
    
    # 强制转换数据类型
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['total_mv'] = pd.to_numeric(df['total_mv'], errors='coerce') 
    
    # total_mv 单位是万元，我们转换为亿元
    df['total_mv_billion'] = df['total_mv'] / 10000
    
    # 剔除价格或市值为空的NaN值
    df = df.dropna(subset=['close', 'total_mv_billion', 'turnover_rate'])
    
    status_text.success(f"基础数据获取和清洗完成！符合【非ST非北交所】的股票共：{len(df)} 只")
    return df, trade_date

# 其余函数（get_technical_and_flow, calculate_strategy, simple_backtest, 主界面逻辑）保持不变

def get_technical_and_flow(pro, ts_code, end_date):
    """
    获取单个股票的技术面和资金流数据
    """
    start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=120)).strftime('%Y%m%d')
    
    # 1. 日线行情
    df_daily = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if len(df_daily) < 60: return None, None 
    
    df_daily = df_daily.sort_values('trade_date') 
    
    # 2. 资金流向 (10000积分特权接口)
    df_flow = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df_flow = df_flow.sort_values('trade_date')
    
    return df_daily, df_flow

def calculate_strategy(df_daily, df_flow):
    """
    计算技术指标并判断是否符合策略
    """
    close = df_daily['close'].values
    
    # 1. 计算均线
    ma20 = pd.Series(close).rolling(window=20).mean().values
    ma60 = pd.Series(close).rolling(window=60).mean().values
    
    # 2. 计算 RSI (14) - 简单算法
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan) 
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.values[-1]
    
    # 3. 20日涨幅
    pct_change_20 = (close[-1] - close[-20]) / close[-20] * 100
        
    # --- 策略判断逻辑 ---
    
    # A. 趋势判断：收盘价 > 20日线 > 60日线 (多头排列，非下跌趋势)
    is_trend_up = (close[-1] > ma20[-1]) and (ma20[-1] > ma60[-1]) 
    
    # B. 排除反弹/超买：RSI < 75 且 20日涨幅 < 80% (非近期翻倍/非超买)
    is_safe_zone = (current_rsi < 75) and (pct_change_20 < 80)
    
    # C. 资金流向 (最近3天主力净流入累计为正)
    if not df_flow.empty:
        recent_flow = df_flow.tail(3)['net_mf_amount'].sum()
        is_money_in = recent_flow > 0
    else:
        is_money_in = False 

    result = {
        'trend_up': is_trend_up,
        'safe_zone': is_safe_zone,
        'money_in': is_money_in,
        'rsi': round(current_rsi, 2),
        'pct_20': round(pct_change_20, 2),
        'close': close[-1],
    }
    return result

def simple_backtest(df_daily):
    """
    简易回测：统计该股票过去半年，出现类似买点后的 T+N 表现
    """
    close = df_daily['close']
    ma20 = close.rolling(20).mean()
    
    returns = {'1d': [], '3d': [], '5d': []}
    
    for i in range(60, len(df_daily) - 5): 
        # 简化版买入条件：收盘价站上MA20 (模拟趋势突破)
        if close.iloc[i] > ma20.iloc[i] and close.iloc[i-1] <= ma20.iloc[i-1]:
            # 记录 T+1, T+3, T+5 收益
            r1 = (close.iloc[i+1] - close.iloc[i]) / close.iloc[i] * 100
            r3 = (close.iloc[i+3] - close.iloc[i]) / close.iloc[i] * 100
            r5 = (close.iloc[i+5] - close.iloc[i]) / close.iloc[i] * 100
            
            returns['1d'].append(r1)
            returns['3d'].append(r3)
            returns['5d'].append(r5)
            
    # 计算平均收益和 3日胜率
    avg_1d = np.mean(returns['1d']) if returns['1d'] else 0
    avg_3d = np.mean(returns['3d']) if returns['3d'] else 0
    avg_5d = np.mean(returns['5d']) if returns['5d'] else 0
    
    # 3日胜率：3日收益为正的次数 / 总次数
    win_rate = len([x for x in returns['3d'] if x > 0]) / len(returns['3d']) if returns['3d'] else 0
    
    return avg_1d, avg_3d, avg_5d, win_rate

# ==========================================
# 4. 主界面逻辑
# ==========================================

st.title(f"🚀 A股智能选股 - 趋势接力版 {APP_VERSION}")
st.markdown("策略：**20-500亿市值 + 趋势向上 + 资金流入 + 排除暴涨/ST**")

with st.sidebar:
    st.header("⚙️ 设置")
    token = st.text_input("请输入 TuShare Token", type="password")
    
    # 增加清除缓存按钮
    if st.button("清除缓存并重置", key="clear_cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()
        
    st.divider()
    st.write("📊 **筛选参数微调**")
    mkt_cap_min, mkt_cap_max = st.slider("市值范围 (亿元)", 10, 1000, (20, 500))
    price_min, price_max = st.slider("价格范围 (元)", 5, 300, (10, 200))
    
    run_btn = st.button("开始选股 (请耐心等待)", type="primary")

if run_btn and token:
    # 1. 初始化和获取基础池
    pro = init_tushare(token)
    df_base, trade_date = get_base_pool(token)
    
    if df_base.empty:
        st.error("数据获取失败，或当前交易日无非ST/非北交所股票数据。请检查 Tushare Token 积分和权限。")
        st.stop()
        
    # 应用侧边栏的动态过滤 
    # total_mv_billion 单位是亿元
    df_pool = df_base[
        (df_base['total_mv_billion'].fillna(0) >= mkt_cap_min) & 
        (df_base['total_mv_billion'].fillna(0) <= mkt_cap_max) &
        (df_base['close'].fillna(0) >= price_min) &
        (df_base['close'].fillna(0) <= price_max)
    ]
    
    if df_pool.empty:
        st.warning(f"基础池规模 {len(df_base)} 只。没有股票满足您设置的市值 ({mkt_cap_min}-{mkt_cap_max}亿) 和价格 ({price_min}-{price_max}元) 范围。请调整侧边栏滑块。")
        st.stop()

    
    st.write(f"📅 数据日期: {trade_date} | 基础池规模: {len(df_base)} 只 | 满足【市值+价格】筛选后剩余: {len(df_pool)} 只 | 正在进行深度分析...")
    
    # 2. 循环处理 (添加进度条)
    final_results = []
    
    # 选取换手率较高的前 200 只进行深度扫描
    target_pool = df_pool.sort_values('turnover_rate', ascending=False).head(200)
    
    total_scan = len(target_pool)
    progress_bar = st.progress(0, text=f"扫描进度：0/{total_scan} 只股票")
    
    for i, row in enumerate(target_pool.itertuples()):
        try:
            # 更新进度条
            progress_bar.progress((i + 1) / total_scan, text=f"扫描进度：{i+1}/{total_scan} 只股票 - 正在分析 {row.name}...")
            
            df_daily, df_flow = get_technical_and_flow(pro, row.ts_code, trade_date)
            
            if df_daily is not None and len(df_daily) >= 60:
                res = calculate_strategy(df_daily, df_flow)
                
                # 核心筛选条件 (趋势向上 AND 安全区间 AND 资金流入)
                if res['trend_up'] and res['safe_zone'] and res['money_in']:
                    
                    # 满足条件，跑一下简易回测
                    r1, r3, r5, win = simple_backtest(df_daily)
                    
                    # 最终筛选：要求历史胜率大于 40%
                    if win >= 0.4:
                        final_results.append({
                            '代码': row.ts_code,
                            '名称': row.name,
                            '行业': row.industry,
                            '现价': res['close'],
                            'RSI': res['rsi'],
                            '20日涨幅(%)': res['pct_20'],
                            '主力净流入(万)': round(df_flow.tail(1)['net_mf_amount'].values[0], 2) if not df_flow.empty else 0,
                            'T+1平均收益(%)': round(r1, 2),
                            'T+3平均收益(%)': round(r3, 2),
                            'T+5平均收益(%)': round(r5, 2),
                            '3日历史胜率': f"{round(win*100)}%"
                        })
        except Exception:
            # 捕获异常，跳过有问题的股票，继续下一只
            continue
            
    progress_bar.empty()
    
    # 3. 展示结果
    if len(final_results) > 0:
        st.success(f"🎉 扫描完成！发现 {len(final_results)} 只潜力股（历史胜率 > 40%）")
        df_res = pd.DataFrame(final_results)
        
        # 交互式表格，用颜色突出主力资金和预期收益
        st.dataframe(
            df_res.style.background_gradient(
                subset=['主力净流入(万)', 'T+1平均收益(%)', 'T+3平均收益(%)'], 
                cmap='RdYlGn'
            ),
            use_container_width=True,
            column_order=['代码', '名称', '现价', '主力净流入(万)', 'T+3平均收益(%)', '3日历史胜率', '行业', 'RSI', '20日涨幅(%)']
        )
        
        # 详细图表展示区 
        st.divider()
        st.subheader("📈 个股详情分析")
        selected_stock = st.selectbox("选择一只股票查看 K 线图", df_res['代码'].astype(str) + " | " + df_res['名称'])
        
        if selected_stock:
            code = selected_stock.split(" | ")[0]
            # 重新获取绘图数据
            df_chart, _ = get_technical_and_flow(pro, code, trade_date)
            
            # 使用 Candlestick 图表
            fig = go.Figure(data=[go.Candlestick(x=df_chart['trade_date'],
                            open=df_chart['open'],
                            high=df_chart['high'],
                            low=df_chart['low'],
                            close=df_chart['close'])])
            fig.update_layout(title=f"{selected_stock} 日线走势", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 交易建议：请参考 **T+3平均收益(%)** 和 **3日历史胜率** 来确定您的持股时间。")
            
    else:
        st.warning(f"满足【市值+价格】条件的股票共 {len(df_pool)} 只，但没有股票完全符合所有【趋势+资金+安全】条件。建议调整侧边栏参数或换个交易日再试。")

elif run_btn and not token:
    st.error("请先在左侧输入 TuShare Token")
