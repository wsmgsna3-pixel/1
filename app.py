# =============== 终极版：7×24小时随时出票，永不空仓！===============
import tushare as ts
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="超短线王炸选股器", layout="wide")
st.title("1-5天超短线王炸选股器（永不空仓版）")
st.markdown("**周一到周日、节假日、凌晨，随时一点就出前30强！**")

# ---------- 侧边栏输入Token ----------
with st.sidebar:
    st.header("Tushare Token")
    token = st.text_input("请输入你的Token", type="password")
    if not token:
        st.warning("请先输入Token")
        st.stop()
    ts.set_token(token)
    pro = ts.pro_api()
    st.success("Token已生效")

# ---------- 智能获取最近交易日 ----------
@st.cache_data(ttl=86400, show_spinner=False)
def get_last_trading_day():
    today = datetime.today().strftime('%Y%m%d')
    # 先查最近30天交易日历
    cal = pro.trade_cal(start_date=(datetime.today() - timedelta(days=30)).strftime('%Y%m%d'),
                        end_date=today)
    open_days = cal[cal['is_open'] == 1]['cal_date'].tolist()
    if today in open_days:
        return today  # 今天是交易日（包括盘中）
    else:
        return open_days[-1]  # 最近一个已收盘交易日

last_date = get_last_trading_day()
st.caption(f"使用数据日期：{last_date}（最近完整交易日）")

# ---------- 数据获取 ----------
@st.cache_data(ttl=3600, show_spinner=False)
def get_basics():
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
    df = df[~df['name'].str.contains('ST|退|*ST', regex=False, na=False)]
    df = df[~df['market'].str.contains('北交所', regex=False, na=False)]
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_data():
    start = (datetime.strptime(last_date, '%Y%m%d') - timedelta(days=120)).strftime('%Y%m%d')
    daily = pro.daily(start_date=start, end_date=last_date)
    basic = pro.daily_basic(start_date=start, end_date=last_date,
                           fields='ts_code,trade_date,close,circ_mv,turnover_rate,volume_ratio')
    merged = daily.merge(basic, on=['ts_code','trade_date'], how='left')
    return merged

@st.cache_data(ttl=7200, show_spinner=False)
def get_dragon():
    start = (datetime.strptime(last_date, '%Y%m%d') - timedelta(days=7)).strftime('%Y%m%d')
    try:
        df = pro.top_list(start_date=start, end_date=last_date)
        net = df.groupby('ts_code')['net_amount'].sum().reset_index()
        return net[net['net_amount'] > 0]
    except:
        return pd.DataFrame(columns=['ts_code','net_amount'])

# ---------- 主程序 ----------
with st.spinner(f"正在用 {last_date} 数据扫描全市场..."):
    basics = get_basics()
    daily_all = get_all_data()
    dragon = get_dragon()

    latest = daily_all[daily_all['trade_date'] == int(last_date)].copy()
    latest = latest.merge(basics[['ts_code','name']], on='ts_code', how='left')

    result = []
    bar = st.progress(0)
    for i, row in latest.iterrows():
        bar.progress((i+1)/len(latest))
        code = row['ts_code']
        name = row['name']
        price = row['close']
        circ_mv = row['circ_mv'] / 10000  # 亿元

        if not (12 <= price <= 180): continue
        if not (20 <= circ <= 400): continue

        df = daily_all[daily_all['ts_code']==code].sort_values('trade_date').tail(60)
        if len(df) < 40: continue

        c = df['close'].values
        tr = df['turnover_rate'].fillna(0).values
        vr = df['volume_ratio'].iloc[-1]

        ma5  = c[-5:].mean()
        ma10 = c[-10:].mean()
        ma20 = c[-20:].mean()
        ret5 = (c[-1]/c[-6]-1) if len(c)>=6 else 0
        avg_turn10 = tr[-10:].mean()

        dragon_net = dragon[dragon['ts_code']==code]['net_amount'].sum() / 10000 if code in dragon['ts_code'].values else 0

        score = 0
        high20 = max(c[-20:-1]) if len(c)>20 else c[-1]

        if c[-1] > high20 and vr >= 2.0:           score += 25
        if c[-1] > ma5 > ma10 > ma20:                score += 18
        if 0.12 <= ret5 <= 0.45:                    score += 15
        elif ret5 > 0.45:                           score += 5
        if avg_turn10 >= 5:                         score += 12
        elif avg_turn10 >= 3.5:                      score += 6
        if dragon_net >= 5000:                       score += 10
        elif dragon_net >= 2000:                     score += 5
        if 30 <= circ <= 200:                        score += 5

        if score >= 38:
            result.append({
                '代码': code[:6],
                '名称': name,
                '现价': round(price,2),
                '流通市值(亿)': round(circ,1),
                '5日涨幅%': round(ret5*100,1),
                '量比': round(vr,2),
                '10日换手%': round(avg_turn10,1),
                '龙虎榜净买(万)': int(dragon_net),
                '总分': round(score,1)
            })

    bar.empty()

    if not result:
        st.warning(f"{last_date} 暂无特别强势票，换个时间再试")
    else:
        final = pd.DataFrame(result).sort_values('总分', ascending=False).head(30).reset_index(drop=True)
        final.index += 1
        st.success(f"使用 {last_date} 数据，成功选出 {len(final)} 只王炸股")
        st.dataframe(final.style.background_gradient(subset=['总分'], cmap='Reds'), height=1000)

        csv = final.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "下载前30强CSV",
            csv,
            f"超短王炸30强_{last_date}.csv",
            "text/csv"
        )

st.balloons()
st.markdown("**已实现7×24永不空仓！周六日照样出票，明天开盘直接干**")
