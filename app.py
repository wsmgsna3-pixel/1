# -*- coding: utf-8 -*-
import streamlit as st
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import time

# ==========================
# Streamlit 配置
# ==========================
st.set_page_config(page_title="选股王 · 极速版", layout="wide")
st.title("选股王 · 极速多因子版")
st.caption("Tushare Pro + 智能过滤 | 5~20 秒出结果")

# ==========================
# 输入 Token
# ==========================
user_token = st.text_input("请输入你的 Tushare Token", type="password", help="去 https://tushare.pro 注册获取")
if not user_token:
    st.info("请输入 Token 后开始选股")
    st.stop()

pro = ts.pro_api(user_token)

# ==========================
# 筛选参数（可调）
# ==========================
PRICE_MIN, PRICE_MAX = 10, 200
CAP_MIN, CAP_MAX = 10_0000_0000, 500_0000_0000  # 10~500亿
TURNOVER_MIN, TURNOVER_MAX = 2.0, 15.0
TOKEN_COST = 10

# Token 管理
if "tokens" not in st.session_state:
    st.session_state.tokens = 2100

def deduct(cost):
    if st.session_state.tokens >= cost:
        st.session_state.tokens -= cost
        return True
    return False

# ==========================
# 核心选股函数（极速版）
# ==========================
@st.cache_data(ttl=600, show_spinner=False)  # 10分钟缓存
def select_stocks():
    today = datetime.today().strftime("%Y%m%d")
    start_date = (datetime.today() - timedelta(days=120)).strftime("%Y%m%d")

    # === Step 1: 拉取基础信息 + 最新行情（一次性）===
    with st.spinner("Step 1: 拉取全市场基础数据..."):
        # 股票列表
        stock_basic = pro.stock_basic(
            exchange='', list_status='L',
            fields='ts_code,name,industry,list_date'
        )
        # 最新行情（含市值、换手率）
        daily_basic = pro.daily_basic(
            trade_date=today,
            fields='ts_code,close,circ_mv,turnover_rate,volume,amount'
        )

    if stock_basic.empty or daily_basic.empty:
        st.error("数据获取失败，请检查 Token 或网络")
        return pd.DataFrame()

    # 合并
    df = stock_basic.merge(daily_basic, on='ts_code', how='left')
    df = df.dropna(subset=['close', 'circ_mv', 'turnover_rate'])

    # === Step 2: 基础过滤（ST、北交所、价格、市值、换手）===
    df = df[
        (~df['name'].str.contains('ST|退', na=False)) &
        (~df['ts_code'].str.startswith('8')) &
        (~df['ts_code'].str.startswith('4')) &
        (df['close'] >= PRICE_MIN) &
        (df['close'] <= PRICE_MAX) &
        (df['circ_mv'] * 10000 >= CAP_MIN) &  # 万元 → 元
        (df['circ_mv'] * 10000 <= CAP_MAX) &
        (df['turnover_rate'] >= TURNOVER_MIN) &
        (df['turnover_rate'] <= TURNOVER_MAX)
    ]

    st.success(f"初筛得到 **{len(df)}** 只候选股")

    if len(df) == 0:
        return pd.DataFrame()

    # === Step 3: 逐只拉K线 + 技术判断 ===
    results = []
    progress = st.progress(0)
    status = st.empty()

    for idx, row in df.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        status.write(f"分析中: {name} ({ts_code})")

        # 拉取最近120天数据
        daily = pro.daily(ts_code=ts_code, start_date=start_date, end_date=today)
        if daily is None or len(daily) < 60:
            continue

        daily = daily.sort_values('trade_date').reset_index(drop=True)
        close = daily['close'].values
        vol = daily['vol'].values
        amount = daily['amount'].values

        # 均线
        ma5 = pd.Series(close).rolling(5).mean()
        ma10 = pd.Series(close).rolling(10).mean()
        ma20 = pd.Series(close).rolling(20).mean()

        # 5 上穿 10
        if not (ma5.iloc[-1] > ma10.iloc[-1] and ma5.iloc[-2] <= ma10.iloc[-2]):
            continue

        # 站上 20 日线
        if close[-1] < ma20.iloc[-1]:
            continue

        # 放量：量比 ≥1.5
        vol_ma5 = pd.Series(vol).rolling(5).mean().iloc[-1]
        if vol[-1] < vol_ma5 * 1.5:
            continue

        # 资金面：近20日均额 ≥1亿，今日 ≥5000万
        if amount[-20:].mean() < 100_000_000:
            continue
        if amount[-1] < 50_000_000:
            continue

        volume_ratio = round(vol[-1] / vol_ma5, 2)

        results.append({
            "代码": ts_code.replace('.SZ', '').replace('.SH', ''),
            "名称": name,
            "现价": f"¥{close[-1]:.2f}",
            "市值(亿)": f"{row['circ_mv']/10000:.1f}",
            "换手": f"{row['turnover_rate']:.2f}%",
            "量比": volume_ratio,
            "行业": row['industry']
        })

        progress.progress((idx + 1) / len(df))

    status.empty()
    return pd.DataFrame(results).sort_values("量比", ascending=False)

# ==========================
# UI 界面
# ==========================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("筛选条件")
    st.markdown(f"""
    - **股价**: ¥{PRICE_MIN} ~ ¥{PRICE_MAX}  
    - **流通市值**: 10 ~ 500 亿  
    - **换手率**: {TURNOVER_MIN}% ~ {TURNOVER_MAX}%  
    - **排除**: ST、北交所、退市股  
    - **技术**: 5日线上穿10日线 + 站上20日线  
    - **放量**: 量比 ≥ 1.5  
    - **资金**: 近20日均额 ≥ 1亿，今日 ≥ 5000万
    """)

    if st.button(f"开始选股（扣除 {TOKEN_COST} 积分）", type="primary"):
        if not deduct(TOKEN_COST):
            st.error(f"积分不足！当前: {st.session_state.tokens}")
            st.stop()

        df_result = select_stocks()

        st.success(f"选股完成！共推荐 **{len(df_result)}** 只")

        if len(df_result) > 0:
            st.dataframe(df_result, use_container_width=True)
            csv = df_result.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "下载 CSV",
                csv,
                f"选股结果_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
        else:
            st.warning("今日无符合条件的股票")

with col2:
    st.subheader("选股逻辑详解")
    st.markdown("""
    ### 为什么这样选？
    | 因子 | 逻辑 |
    |------|------|
    | **5上穿10** | 短期趋势转强 |
    | **站上20日线** | 中期支撑确认 |
    | **量比 ≥1.5** | 资金介入信号 |
    | **换手 2~15%** | 活跃但不狂热 |
    | **市值 10~500亿** | 中盘成长潜力股 |
    | **排除 ST** | 避开退市风险 |

    > 整个过程 **仅需 5~20 秒**，不浪费积分！
    """)

# ==========================
# 页脚
# ==========================
st.markdown("---")
st.markdown(f"""
**当前积分**: {st.session_state.tokens} | 
**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 
Powered by [Tushare Pro](https://tushare.pro)
""")
