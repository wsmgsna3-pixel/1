import streamlit as st
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

# ==========================
# Streamlit 配置
# ==========================
st.set_page_config(page_title="选股王 · 双模核武库 v3.3", layout="wide")
st.title("选股王 · 双模核武库 v3.3")
st.caption("2100积分驱动 | 中小盘主升浪狙击手 | 周一8:30 AM PST 必出肉")

# ==========================
# 输入 Token
# ==========================
user_token = st.text_input("请输入你的 Tushare Token（2100积分已就位）", type="password")
if not user_token:
    st.info("请输入 Token 后点击 **开始选股**")
    st.stop()

pro = ts.pro_api(user_token)

# ==========================
# 模式切换
# ==========================
col1, col2 = st.columns([1, 3])
with col1:
    mode = st.radio(
        "选择模式",
        ["核弹模式", "狙击枪模式"],
        index=1,
        help="核弹：极致精准 | 狙击枪：平衡吃肉"
    )
with col2:
    if mode == "核弹模式":
        st.markdown("**核弹模式** 0~5 只妖股")
    else:
        st.markdown("**狙击枪模式** 8~20 只主升浪")

# ==========================
# 缓存函数（已修复日期问题）
# ==========================
@st.cache_data(ttl=3600, show_spinner=False)
def get_last_trade_day():
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
    cal = pro.trade_cal(start_date=start_date, end_date=end_date)
    
    if cal.empty:
        st.error("无法获取交易日历，请检查网络或 Token")
        return None
    
    open_days = cal[cal['is_open'] == 1]
    if open_days.empty:
        st.error("近期无交易日数据")
        return None
    
    return open_days['cal_date'].iloc[-1]

@st.cache_data(ttl=3600, show_spinner=False)
def get_previous_trade_day(current):
    if not current:
        return None
    dt = datetime.strptime(current, "%Y%m%d")
    for i in range(1, 15):
        prev = (dt - timedelta(days=i)).strftime("%Y%m%d")
        cal = pro.trade_cal(start_date=prev, end_date=prev)
        if not cal.empty and cal.iloc[0]['is_open']:
            return prev
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def run_selection(_pro, last_trade_day, yesterday, mode):
    try:
        # 参数配置
        if mode == "核弹模式":
            top_n = 500
            volume_ratio = 1.3
            amount_threshold = 100  # 1亿
            gold_cross_days = 1
        else:  # 狙击枪模式
            top_n = 1000
            volume_ratio = 1.25
            amount_threshold = 50   # 5000万
            gold_cross_days = 2

        # Step 1: 最新数据
        with st.spinner("Step 1: 获取全市场最新数据…"):
            daily_basic = _pro.daily_basic(trade_date=last_trade_day,
                                           fields='ts_code,close,total_mv')
            if daily_basic.empty: return pd.DataFrame()

        # Step 2: 基础信息
        with st.spinner("Step 2: 获取股票名称与行业…"):
            stock_basic = _pro.stock_basic(exchange='', list_status='L',
                                           fields='ts_code,name,industry')
            if stock_basic.empty: return pd.DataFrame()

        df = daily_basic.merge(stock_basic, on='ts_code')
        df = df.rename(columns={'close': 'latest_close'})

        # Step 3: 初筛 → 300 只（修复 *ST 正则错误）
        df = df[
            (~df['name'].str.contains('ST', na=False)) &
            (~df['name'].str.contains(r'\*ST', na=False, regex=True)) &
            (~df['ts_code'].str.startswith('8')) &
            (~df['ts_code'].str.startswith('4')) &
            (df['latest_close'] >= 10) & (df['latest_close'] <= 200) &
            (df['total_mv'] >= 100000) & (df['total_mv'] <= 50000000)
        ].copy()

        if df.empty: 
            st.info("初筛后无候选股，市场可能休整中")
            return pd.DataFrame()

        # Step 4: 昨日涨幅（在 300 只中排序）
        with st.spinner("Step 3: 获取昨日涨跌幅…"):
            daily = _pro.daily(trade_date=yesterday, fields='ts_code,pct_chg')
            if daily.empty: return pd.DataFrame()

            daily = daily[daily['pct_chg'] < 9.8]
            daily = daily[daily['ts_code'].isin(df['ts_code'])]
            if daily.empty: return pd.DataFrame()

            top_codes = daily.nlargest(top_n, 'pct_chg')['ts_code'].tolist()

        df = df[df['ts_code'].isin(top_codes)]
        df = df.merge(daily[['ts_code', 'pct_chg']], on='ts_code', how='left')
        if df.empty: return pd.DataFrame()

        # Step 5: 批量拉日线
        ts_code_str = ','.join(df['ts_code'].tolist())
        start_date = (datetime.strptime(last_trade_day, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")

        with st.spinner(f"Step 4: 批量获取 {len(df)} 只股票数据…"):
            daily_data = _pro.daily(ts_code=ts_code_str, start_date=start_date, end_date=last_trade_day)
            if daily_data.empty: return pd.DataFrame()

        daily_data = daily_data.merge(df[['ts_code', 'name', 'industry']], on='ts_code')
        daily_data = daily_data.sort_values(['ts_code', 'trade_date'])

        # Step 6: 技术指标
        daily_data['ma5'] = daily_data.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
        daily_data['ma10'] = daily_data.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
        daily_data['ma20'] = daily_data.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
        daily_data['vol_ma5'] = daily_data.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5).mean())

        latest = daily_data.groupby('ts_code').tail(1).copy()
        if len(latest) < 2: return pd.DataFrame()

        prev = daily_data.groupby('ts_code').apply(
            lambda x: x.tail(2).iloc[:-1] if len(x) >= 2 else pd.DataFrame()
        ).reset_index(drop=True)
        prev = prev[['ts_code', 'ma5', 'ma10']].rename(columns={'ma5': 'ma5_prev', 'ma10': 'ma10_prev'})
        latest = latest.merge(prev, on='ts_code', how='left')

        # 条件筛选
        cond1 = latest['ma5'] > latest['ma10']
        cond2 = (latest['ma5_prev'] <= latest['ma10_prev']) if gold_cross_days == 1 else True
        cond3 = latest['close'] >= latest['ma20']
        cond4 = latest['vol'] >= latest['vol_ma5'] * volume_ratio

        amount_mean = daily_data.groupby('ts_code')['amount'].tail(20).mean()
        latest['amount_ok'] = latest['ts_code'].map(amount_mean >= amount_threshold)

        result = latest[cond1 & cond2 & cond3 & cond4 & latest['amount_ok']].copy()
        if result.empty: return pd.DataFrame()

        result['volume_ratio'] = (result['vol'] / result['vol_ma5']).round(2)
        result = result.merge(df[['ts_code', 'pct_chg', 'latest_close']], on='ts_code')

        output = result[['ts_code', 'name', 'latest_close', 'volume_ratio', 'pct_chg', 'industry']]
        output.columns = ['代码', '名称', '现价', '放量倍数', '昨日涨幅%', '行业']
        output = output.sort_values('放量倍数', ascending=False).reset_index(drop=True)
        return output

    except Exception as e:
        st.error(f"运行出错：{str(e)}")
        return pd.DataFrame()

# ==========================
# 执行按钮（已修复日期显示）
# ==========================
if st.button("开始选股", type="primary", use_container_width=True):
    last_trade_day = get_last_trade_day()
    if not last_trade_day:
        st.stop()
        
    yesterday = get_previous_trade_day(last_trade_day)
    if not yesterday:
        st.error("无法获取昨日交易日")
        st.stop()

    last_str = f"{last_trade_day[:4]}-{last_trade_day[4:6]}-{last_trade_day[6:]}"
    yest_str = f"{yesterday[:4]}-{yesterday[4:6]}-{yesterday[6:]}"
    today_str = datetime.now().strftime("%Y-%m-%d")

    if last_str != today_str[:10].replace('-', ''):
        st.info(f"今日非交易日，使用 **最近交易日 {last_str}** 数据")
    st.caption(f"数据：{last_str} | 昨日：{yest_str}")

    with st.spinner("选股王启动双模核武…"):
        df_result = run_selection(pro, last_trade_day, yesterday, mode)

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
            },
            hide_index=True
        )
        st.balloons()
        st.caption(f"命中 {len(df_result)} 只强势股 | 模式：{mode}")
