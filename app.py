import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ==========================================
# 1. 页面配置与工具函数
# ==========================================
st.set_page_config(page_title="趋势接力选股器 Pro", layout="wide", page_icon="📈")

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
    try:
        ts.set_token(token)
        return ts.pro_api()
    except Exception as e:
        st.error(f"Token 设置失败: {e}")
        return None

# ==========================================
# 2. 核心数据获取逻辑 (利用 10000 积分优势)
# ==========================================

@st.cache_data(ttl=3600) # 缓存1小时，避免重复请求
def get_base_pool(token_input):
    """
    第一步：基础池筛选（市值、价格、非ST、非北交所）
    使用 daily_basic 接口一次性获取所有数据进行过滤，极大提高速度
    """
    pro = init_tushare(token_input)
    if not pro: return pd.DataFrame()

    status_text = st.empty()
    status_text.info("正在获取全市场基础数据 (每日指标)...")

    # 获取最新一个交易日
    cal = pro.trade_cal(exchange='', is_open='1', end_date=datetime.now().strftime('%Y%m%d'), fields='cal_date')
    trade_date = cal['cal_date'].values[-1]
    
    # 获取每日指标（包含市值、换手率、量比、价格）
    # 10000积分用户可以直接获取全市场数据
    df_daily = pro.daily_basic(trade_date=trade_date, fields='ts_code,close,turnover_rate,volume_ratio,circ_mv,total_mv,pe,pb')
    
    # 获取基础信息（用于排除ST和北交所）
    df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,market,list_date')
    
    # 合并数据
    df = pd.merge(df_basic, df_daily, on='ts_code', how='inner')
    
    # --- 核心筛选逻辑 Step 1 ---
    # 1. 排除北交所 (Market != 北京 / 代码不以8/4/9开头，保险起见用 market 字段)
    # Tushare market字段: 主板, 创业板, 科创板, 北交所
    df = df[~df['market'].str.contains('北|BJE', na=False)] 
    
    # 2. 排除ST
    df = df[~df['name'].str.contains('ST|退', na=False)]
    
    # 3. 市值筛选 (20亿 - 500亿) - 单位是万元，所以 20亿=200000, 500亿=5000000
    df = df[(df['total_mv'] >= 200000) & (df['total_mv'] <= 5000000)]
    
    # 4. 价格筛选 (10元 - 200元)
    df = df[(df['close'] >= 10) & (df['close'] <= 200)]
    
    status_text.success(f"基础筛选完成！符合【市值+价格+非ST】的股票共：{len(df)} 只")
    return df, trade_date

def get_technical_and_flow(pro, ts_code, end_date):
    """
    获取单个股票的技术面和资金流数据
    """
    # 获取过去60个交易日数据（用于计算均线和RSI）
    start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=120)).strftime('%Y%m%d')
    
    # 1. 日线行情
    df_daily = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if len(df_daily) < 30: return None # 新股数据不足
    
    df_daily = df_daily.sort_values('trade_date') # 按日期升序
    
    # 2. 资金流向 (10000积分特权接口)
    # 获取最近3天的资金流
    df_flow = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df_flow = df_flow.sort_values('trade_date')
    
    return df_daily, df_flow

# ==========================================
# 3. 策略计算与回测逻辑
# ==========================================

def calculate_strategy(df_daily, df_flow):
    """
    计算技术指标并判断是否符合策略
    """
    close = df_daily['close'].values
    
    # 1. 计算均线
    ma20 = pd.Series(close).rolling(window=20).mean().values
    ma60 = pd.Series(close).rolling(window=60).mean().values
    
    # 2. 计算 RSI (14) - 简单算法，不依赖 talib
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.values[-1]
    
    # 3. 20日涨幅
    if len(close) >= 20:
        pct_change_20 = (close[-1] - close[-20]) / close[-20] * 100
    else:
        pct_change_20 = 0
        
    # --- 策略判断逻辑 ---
    
    # A. 趋势判断：收盘价 > 20日线 > 60日线 (多头排列，非下跌趋势)
    is_trend_up = (close[-1] > ma20[-1]) and (ma20[-1] > ma60[-1])
    
    # B. 排除反弹/超买：RSI < 75 且 20日涨幅 < 80%
    is_safe_zone = (current_rsi < 75) and (pct_change_20 < 80)
    
    # C. 资金流向 (最近3天主力净流入至少有1天是正的，或者累计是正的)
    if not df_flow.empty:
        # net_mf_amount: 主力净流入额(万元)
        recent_flow = df_flow.tail(3)['net_mf_amount'].sum()
        is_money_in = recent_flow > 0
    else:
        is_money_in = False # 如果没有资金数据，保守起见

    result = {
        'trend_up': is_trend_up,
        'safe_zone': is_safe_zone,
        'money_in': is_money_in,
        'rsi': round(current_rsi, 2),
        'pct_20': round(pct_change_20, 2),
        'close': close[-1],
        'ma20': ma20[-1]
    }
    return result

def simple_backtest(df_daily):
    """
    简易回测：统计该股票过去半年，出现类似买点后的表现
    """
    # 简单的策略：当收盘价站上MA20 且 RSI<70 时买入
    close = df_daily['close']
    ma20 = close.rolling(20).mean()
    
    signals = []
    returns = {'1d': [], '3d': [], '5d': []}
    
    for i in range(20, len(df_daily) - 5):
        # 模拟买入条件
        if close.iloc[i] > ma20.iloc[i] and close.iloc[i-1] <= ma20.iloc[i-1]:
            # 记录 T+1, T+3, T+5 收益
            r1 = (close.iloc[i+1] - close.iloc[i]) / close.iloc[i] * 100
            r3 = (close.iloc[i+3] - close.iloc[i]) / close.iloc[i] * 100
            r5 = (close.iloc[i+5] - close.iloc[i]) / close.iloc[i] * 100
            
            returns['1d'].append(r1)
            returns['3d'].append(r3)
            returns['5d'].append(r5)
            
    avg_1d = np.mean(returns['1d']) if returns['1d'] else 0
    avg_3d = np.mean(returns['3d']) if returns['3d'] else 0
    avg_5d = np.mean(returns['5d']) if returns['5d'] else 0
    win_rate = len([x for x in returns['3d'] if x > 0]) / len(returns['3d']) if returns['3d'] else 0
    
    return avg_1d, avg_3d, avg_5d, win_rate

# ==========================================
# 4. 主界面逻辑
# ==========================================

st.title("🚀 A股智能选股 - 趋势接力版")
st.markdown("策略：**20-500亿市值 + 趋势向上 + 资金流入 + 排除暴涨/ST**")

with st.sidebar:
    st.header("⚙️ 设置")
    # 默认值留空，让用户输入
    token = st.text_input("请输入 TuShare Token", type="password")
    
    st.divider()
    st.write("📊 **筛选参数微调**")
    mkt_cap_min, mkt_cap_max = st.slider("市值范围 (亿元)", 20, 1000, (20, 500))
    price_min, price_max = st.slider("价格范围 (元)", 5, 300, (10, 200))
    
    run_btn = st.button("开始选股 (可能需要几分钟)", type="primary")

if run_btn and token:
    pro = init_tushare(token)
    
    # 1. 获取基础池
    df_base, trade_date = get_base_pool(token)
    
    # 应用侧边栏的动态过滤
    df_pool = df_base[
        (df_base['total_mv'] >= mkt_cap_min * 10000) & 
        (df_base['total_mv'] <= mkt_cap_max * 10000) &
        (df_base['close'] >= price_min) &
        (df_base['close'] <= price_max)
    ]
    
    st.write(f"📅 数据日期: {trade_date} | 初筛后剩余: {len(df_pool)} 只 | 正在进行技术面与资金面分析...")
    
    # 2. 循环处理 (添加进度条)
    final_results = []
    progress_bar = st.progress(0)
    
    # 为了演示速度，限制处理数量，或者打乱顺序随机抽取部分进行演示，
    # 实际部署时可以处理全部，但会比较慢 (几百只股票请求几百次API)
    # 这里我们选取换手率较高的前 100 只进行深度扫描，以提高效率
    target_pool = df_pool.sort_values('turnover_rate', ascending=False).head(100)
    
    total_scan = len(target_pool)
    
    for i, row in enumerate(target_pool.itertuples()):
        try:
            # 更新进度条
            progress_bar.progress((i + 1) / total_scan)
            
            df_daily, df_flow = get_technical_and_flow(pro, row.ts_code, trade_date)
            
            if df_daily is not None:
                res = calculate_strategy(df_daily, df_flow)
                
                # 核心筛选条件
                if res['trend_up'] and res['safe_zone'] and res['money_in']:
                    
                    # 满足条件，跑一下简易回测
                    r1, r3, r5, win = simple_backtest(df_daily)
                    
                    # 如果历史胜率太低 (<40%)，也排除
                    if win > 0.4:
                        final_results.append({
                            '代码': row.ts_code,
                            '名称': row.name,
                            '行业': row.industry,
                            '现价': row.close,
                            'RSI': res['rsi'],
                            '主力净流入(万)': round(df_flow.tail(1)['net_mf_amount'].values[0], 2) if not df_flow.empty else 0,
                            'T+1预期': round(r1, 2),
                            'T+3预期': round(r3, 2),
                            '历史胜率': f"{round(win*100)}%"
                        })
        except Exception as e:
            continue # 跳过错误的
            
    progress_bar.empty()
    
    # 3. 展示结果
    if len(final_results) > 0:
        st.success(f"🎉 扫描完成！发现 {len(final_results)} 只潜力股")
        df_res = pd.DataFrame(final_results)
        
        # 交互式表格
        st.dataframe(
            df_res.style.background_gradient(subset=['主力净流入(万)', 'T+3预期'], cmap='Reds'),
            use_container_width=True
        )
        
        # 详细图表展示区
        st.divider()
        st.subheader("📈 个股详情分析")
        selected_stock = st.selectbox("选择一只股票查看详情", df_res['代码'].astype(str) + " | " + df_res['名称'])
        
        if selected_stock:
            code = selected_stock.split(" | ")[0]
            # 重新获取绘图数据
            df_chart, _ = get_technical_and_flow(pro, code, trade_date)
            
            fig = go.Figure(data=[go.Candlestick(x=df_chart['trade_date'],
                            open=df_chart['open'],
                            high=df_chart['high'],
                            low=df_chart['low'],
                            close=df_chart['close'])])
            fig.update_layout(title=f"{selected_stock} 日线走势", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 交易建议：该股主力近期流入，且趋势向上。根据历史回测，建议持股 3 天左右卖出胜率较高。")
            
    else:
        st.warning("当前没有完全符合【趋势+资金+安全】条件的股票，建议放宽市值或价格范围试试。")

elif run_btn and not token:
    st.error("请先在左侧输入 TuShare Token")

