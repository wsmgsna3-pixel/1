import streamlit as st
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

# ==========================
# Streamlit 配置
# ==========================
st.set_page_config(page_title="选股王 · 核弹版", layout="wide")
st.title("选股王 · 2100积分核弹版 v2.0")
st.caption("中小盘 + 强势起爆 + 金叉放量，精准捕捉主升浪！")

# ==========================
# 输入 Token
# ==========================
user_token = st.text_input("请输入你的 Tushare Token（2100积分已就位）", type="password")
if not user_token:
    st.info("请输入 Token 后点击 **开始选股**")
    st.stop()

pro = ts.pro_api(user_token)

# ==========================
# 缓存函数
# ==========================
@st.cache_data(ttl=3600, show_spinner=False)  # 缓存1小时
def get_last_trade_day():
    today = datetime.today().strftime("%Y%m%d")
    start = (datetime.today() - timedelta(days=30)).strftime("%Y%m%d")
    cal = pro.trade_cal(start_date=start, end_date=today)
    return cal[cal['is_open'] == 1]['cal_date'].iloc[-1]

@st.cache_data(ttl=3600, show_spinner=False)
def run_selection(_pro, last_trade_day):
    # ---------- Step 1: 获取最新市值 + 价格 ----------
    with st.spinner("Step 1: 正在获取全市场最新数据…"):
        daily_basic = _pro.daily_basic(trade_date=last_trade_day,
                                       fields='ts_code,close,total_mv')
        if daily_basic.empty:
            return pd.DataFrame()

    # ---------- Step 2: 获取基础信息 ----------
    with st.spinner("Step 2: 正在获取股票名称与行业…"):
        stock_basic = _pro.stock_basic(exchange='', list_status='L',
                                       fields='ts_code,name,industry')
        if stock_basic.empty:
            return pd.DataFrame()

    # 合并
    df = daily_basic.merge(stock_basic, on='ts_code', how='left')

    # ---------- Step 3: 初筛（市值 + 价格 + 非ST + 非北交所） ----------
    df = df[
        (~df['name'].str.contains('ST', na=False)) &
        (~df['ts_code'].str.startswith('8')) &
        (~df['ts_code'].str.startswith('4')) &
        (df['close'] >= 10) & (df['close'] <= 200) &
        (df['total_mv'] >= 100000) & (df['total_mv'] <= 50000000)  # 10亿~500亿（万元）
    ]

    if df.empty:
        return pd.DataFrame()

    # ---------- Step 4: 获取昨日涨幅，筛选前500 ----------
    with st.spinner("Step 3: 正在获取昨日涨跌幅…"):
        yesterday = (datetime.strptime(last_trade_day, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        cal = _pro.trade_cal(start_date=yesterday, end_date=yesterday)
        if not cal.iloc[0]['is_open']:
            yesterday = get_last_trade_day()  # 再往前推
        daily = _pro.daily(trade_date=yesterday, fields='ts_code,pct_chg')
        if daily.empty:
            return pd.DataFrame()

        # 排除涨停 + 取前500
        daily = daily[daily['pct_chg'] < 9.8]
        top500 = daily.nlargest(500, 'pct_chg')['ts_code'].tolist()

    # 交集
    df = df[df['ts_code'].isin(top500)]
    if df.empty:
        return pd.DataFrame()

    # ---------- Step 5: 批量拉取120天日线（一次请求！） ----------
    ts_code_str = ','.join(df['ts_code'].tolist())
    start_date = (datetime.strptime(last_trade_day, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")

    with st.spinner(f"Step 4: 正在批量获取 {len(df)} 只股票的120天数据…"):
        daily_data = _pro.daily(ts_code=ts_code_str, start_date=start_date, end_date=last_trade_day)
        if daily_data.empty:
            return pd.DataFrame()

    # 合并
    daily_data = daily_data.merge(df[['ts_code', 'name', 'industry', 'close']], on='ts_code', how='left')
    daily_data = daily_data.sort_values(['ts_code', 'trade_date'])

    # ---------- Step 6: 向量化计算技术指标 ----------
    daily_data['ma5'] = daily_data.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    daily_data['ma10'] = daily_data.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
    daily_data['ma20'] = daily_data.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    daily_data['vol_ma5'] = daily_data.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5).mean())

    # 取最新一天
    latest = daily_data.groupby('ts_code').tail(1).copy()
    prev = daily_data.groupby('ts_code').apply(lambda x: x.iloc[-2] if len(x) > 1 else None).dropna()

    # 合并前后两天
    latest = latest.merge(prev[['ts_code', 'ma5', 'ma10']], on='ts_code', suffixes=('', '_prev'))

    # 条件筛选
    cond1 = latest['ma5'] > latest['ma10']  # 今日金叉
    cond2 = latest['ma5_prev'] <= latest['ma10_prev']  # 昨日未穿越
    cond3 = latest['close'] >= latest['ma20']  # 站上20日线
    cond4 = latest['vol'] >= latest['vol_ma5'] * 1.3  # 放量1.3倍
    cond5 = daily_data.groupby('ts_code')['amount'].tail(20).mean() >= 100  # 近20日均额 >= 1亿

    # 合并 amount 条件
    amount_check = daily_data.groupby('ts_code')['amount'].tail(20).mean().reset_index()
    amount_check = amount_check[amount_check['amount'] >= 100]
    latest = latest.merge(amount_check[['ts_code']], on='ts_code')

    # 最终筛选
    result = latest[cond1 & cond2 & cond3 & cond4].copy()
    if result.empty:
        return pd.DataFrame()

    # 合并昨日涨幅
    result = result.merge(daily[['ts_code', 'pct_chg']], on='ts_code')
    result['volume_ratio'] = (result['vol'] / result['vol_ma5']).round(2)

    # 最终输出
    result = result[['ts_code', 'name', 'close', 'volume_ratio', 'pct_chg', 'industry']]
    result.columns = ['代码', '名称', '现价', '放量倍数', '昨日涨幅%', '行业']
    result = result.sort_values('放量倍数', ascending=False).reset_index(drop=True)

    return result

# ==========================
# 执行按钮
# ==========================
if st.button("开始选股", type="primary", use_container_width=True):
    last_trade_day = get_last_trade_day()
    st.write(f"数据日期：**{last_trade_day[:4]}-{last_trade_day[4:6]}-{last_trade_day[6:]}**")

    with st.spinner("选股王正在启动核弹模式…"):
        df_result = run_selection(pro, last_trade_day)

    st.success("选股完成！")

    if df_result.empty:
        st.warning("今日无满足条件的股票，明天再来！")
    else:
        st.dataframe(
            df_result,
            use_container_width=True,
            column_config={
                "昨日涨幅%": st.column_config.NumberColumn(format="%.2f%%"),
                "现价": st.column_config.NumberColumn(format="%.2f"),
                "放量倍数": st.column_config.NumberColumn(format="%.2fx")
            }
        )
        st.balloons()
