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
    try:
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

        # 合并（close 重命名为 latest_close 避冲突）
        df = daily_basic.merge(stock_basic, on='ts_code', how='left')
        df = df.rename(columns={'close': 'latest_close'})

        # ---------- Step 3: 初筛（市值 + 价格 + 非ST + 非北交所） ----------
        df = df[
            (~df['name'].str.contains('ST', na=False)) &
            (~df['ts_code'].str.startswith('8')) &
            (~df['ts_code'].str.startswith('4')) &
            (df['latest_close'] >= 10) & (df['latest_close'] <= 200) &
            (df['total_mv'] >= 100000) & (df['total_mv'] <= 50000000)  # 10亿~500亿（万元）
        ].copy()

        if df.empty:
            return pd.DataFrame()

        # ---------- Step 4: 获取昨日涨幅，筛选前500 ----------
        with st.spinner("Step 3: 正在获取昨日涨跌幅…"):
            yesterday_dt = datetime.strptime(last_trade_day, "%Y%m%d") - timedelta(days=1)
            yesterday = yesterday_dt.strftime("%Y%m%d")
            # 确认昨日是交易日
            cal_yest = _pro.trade_cal(start_date=yesterday, end_date=yesterday)
            if cal_yest.empty or not cal_yest.iloc[0]['is_open']:
                # 再往前找
                cal = _pro.trade_cal(start_date=(yesterday_dt - timedelta(days=7)).strftime("%Y%m%d"), 
                                     end_date=yesterday)
                yesterday = cal[cal['is_open'] == 1]['cal_date'].iloc[-1]
            
            daily = _pro.daily(trade_date=yesterday, fields='ts_code,pct_chg')
            if daily.empty:
                return pd.DataFrame()

            # 排除涨停 + 取前500
            daily = daily[daily['pct_chg'] < 9.8]
            top500 = daily.nlargest(500, 'pct_chg')['ts_code'].tolist()

        # 交集 + 合并昨日涨幅
        df = df[df['ts_code'].isin(top500)]
        df = df.merge(daily[['ts_code', 'pct_chg']], on='ts_code', how='left')
        if df.empty:
            return pd.DataFrame()

        # ---------- Step 5: 批量拉取120天日线（一次请求！） ----------
        ts_code_str = ','.join(df['ts_code'].tolist())
        start_date = (datetime.strptime(last_trade_day, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")

        with st.spinner(f"Step 4: 正在批量获取 {len(df)} 只股票的120天数据…"):
            daily_data = _pro.daily(ts_code=ts_code_str, start_date=start_date, end_date=last_trade_day)
            if daily_data.empty:
                return pd.DataFrame()

        # 合并（只合并 name/industry，避免 close 冲突）
        daily_data = daily_data.merge(df[['ts_code', 'name', 'industry']], on='ts_code', how='left')
        daily_data = daily_data.sort_values(['ts_code', 'trade_date'])

        # ---------- Step 6: 向量化计算技术指标（用历史 close） ----------
        daily_data['ma5'] = daily_data.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
        daily_data['ma10'] = daily_data.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
        daily_data['ma20'] = daily_data.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
        daily_data['vol_ma5'] = daily_data.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5).mean())

        # 过滤数据不足的股票
        daily_data = daily_data.dropna(subset=['ma20'])  # 至少20天数据
        valid_codes = daily_data.groupby('ts_code').filter(lambda x: len(x) >= 60)['ts_code'].unique()
        daily_data = daily_data[daily_data['ts_code'].isin(valid_codes)]

        if daily_data.empty:
            return pd.DataFrame()

        # 取最新一天（用历史最新 close 作为现价）
        latest = daily_data.groupby('ts_code').tail(1).copy()
        if len(latest) < 2:
            return pd.DataFrame()  # 至少2天数据防 -2

        # 前一天数据（用于金叉判断）
        prev_data = daily_data.groupby('ts_code').apply(lambda x: x.tail(2).iloc[:-1] if len(x) >= 2 else pd.DataFrame()).reset_index(drop=True)
        prev = prev_data[['ts_code', 'ma5', 'ma10']].rename(columns={'ma5': 'ma5_prev', 'ma10': 'ma10_prev'})

        # 合并前后两天
        latest = latest.merge(prev, on='ts_code', how='left')

        # ---------- Step 7: 筛选条件 ----------
        # 金叉：今日 ma5 > ma10 且 昨日 ma5 <= ma10
        cond1 = latest['ma5'] > latest['ma10']
        cond2 = latest['ma5_prev'] <= latest['ma10_prev']
        # 站上 MA20
        cond3 = latest['close'] >= latest['ma20']
        # 放量 1.3 倍
        cond4 = latest['vol'] >= latest['vol_ma5'] * 1.3
        # 近20日均额 >= 1亿（amount 已万元）
        amount_mean = daily_data.groupby('ts_code')['amount'].tail(20).mean()
        cond5 = amount_mean >= 100
        latest['amount_ok'] = latest['ts_code'].map(cond5)

        # 最终筛选
        result = latest[cond1 & cond2 & cond3 & cond4 & latest['amount_ok'].fillna(False)].copy()
        if result.empty:
            return pd.DataFrame()

        # 计算放量倍数 + 合并昨日涨幅
        result['volume_ratio'] = (result['vol'] / result['vol_ma5']).round(2)
        result = result.merge(df[['ts_code', 'pct_chg', 'latest_close']], on='ts_code', how='left')

        # 输出列
        output_cols = ['ts_code', 'name', 'latest_close', 'volume_ratio', 'pct_chg', 'industry']
        result = result[output_cols].rename(columns={
            'ts_code': '代码',
            'name': '名称',
            'latest_close': '现价',
            'volume_ratio': '放量倍数',
            'pct_chg': '昨日涨幅%',
            'industry': '行业'
        })

        result = result.sort_values('放量倍数', ascending=False).reset_index(drop=True)
        return result

    except Exception as e:
        st.error(f"运行出错：{str(e)}")
        return pd.DataFrame()

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
            },
            hide_index=True
        )
        st.balloons()
        st.caption(f"今日命中 {len(df_result)} 只强势股！")
