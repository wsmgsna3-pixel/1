# =============== app.py 最终安全稳定版（直接复制运行）===============
import tushare as ts
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta

# ---------- 页面设置 ----------
st.set_page_config(page_title="1-5天超短线王炸选股器", layout="wide")
st.title("1-5天超短线王炸选股器（5100积分专属版）")
st.markdown("**每周一早跑一次，前30名直接干，持股1-5天吃肉走人**")

# ---------- 侧边栏输入Token（安全！不保存）----------
with st.sidebar:
    st.header("请先输入你的Tushare Token")
    token = st.text_input("Token（运行时输入，绝不保存）", type="password")
    if not token:
        st.warning("请输入Token后回车")
        st.stop()
    ts.set_token(token)
    pro = ts.pro_api()
    st.success("Token已生效，开始拉数据！")

st.caption(f"数据截止：{datetime.now().strftime('%Y年%m月%d日 %H:%M')}")

# ---------- 缓存函数 ----------
@st.cache_data(ttl=3600, show_spinner=False)
def get_basics():
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
    df = df[~df['name'].str.contains('ST|退|*ST')]
    df = df[~df['market'].str.contains('北交所')]
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_daily_data():
    end = datetime.today().strftime('%Y%m%d')
    start = (datetime.today() - timedelta(days=120)).strftime('%Y%m%d')
    daily = pro.daily(start_date=start, end_date=end)
    basic = pro.daily_basic(start_date=start, end_date=end, 
                           fields='ts_code,trade_date,close,circ_mv,turnover_rate,volume_ratio')
    return daily.merge(basic, on=['ts_code','trade_date'], how='left')

@st.cache_data(ttl=7200, show_spinner=False)
def get_dragon():
    start = (datetime.today() - timedelta(days=7)).strftime('%Y%m%d')
    end = datetime.today().strftime('%Y%m%d')
    try:
        df = pro.top_list(start_date=start, end_date=end)
        net = df.groupby('ts_code')['net_amount'].sum().reset_index()
        net = net[net['net_amount'] > 0]
        return net
    except:
        return pd.DataFrame(columns=['ts_code','net_amount'])

@st.cache_data(ttl=7200, show_spinner=False)
def get_sw_rank():
    start = (datetime.today() - timedelta(days=12)).strftime('%Y%m%d')
    end = datetime.today().strftime('%Y%m%d')
    try:
        df = pro.sw_daily(start_date=start, end_date=end, fields='index_code,name,pct_chg')
        recent5 = df.groupby('index_code').apply(lambda x: x.tail(5))
        rank = recent5.groupby('name')['pct_chg'].sum().sort_values(ascending=False).reset_index()
        rank['rank'] = rank.index + 1
        return dict(zip(rank['name'], rank['rank']))
    except:
        return {}

# ---------- 主逻辑 ----------
with st.spinner("5100积分全速拉取全市场数据..."):
    basics = get_basics()
    daily_all = get_daily_data()
    dragon = get_dragon()
    sw_rank_map = get_sw_rank()

    latest_date = daily_all['trade_date'].max()
    latest = daily_all[daily_all['trade_date'] == latest_date].copy()
    latest = latest.merge(basics[['ts_code','name']], on='ts_code', how='left')

    result = []
    bar = st.progress(0)
    total = len(latest)

    for i, row in enumerate(latest.itertuples()):
        bar.progress((i + 1) / total)
        code, name, price, circ_mv_b = getattr(row, 'ts_code'), getattr(row, 'name'), row.close, row.circ_mv
        circ_mv = circ_mv_b / 10000  # 转为亿元

        # 你的硬过滤
        if not (12 <= price <= 180): continue
        if not (20 <= circ_mv <= 400): continue

        df = daily_all[daily_all['ts_code']==code].sort_values('trade_date').tail(60)
        if len(df) < 40: continue

        c = df['close'].values
        tr = df['turnover_rate'].values
        vr = df['volume_ratio'].iloc[-1]

        ma5 = c[-5:].mean()
        ma10 = c[-10:].mean()
        ma20 = c[-20:].mean()
        ret5 = c[-1]/c[-6] - 1 if len(c)>=6 else 0
        avg_turn10 = tr[-10:].mean()

        # 龙虎榜净买（万）
        dragon_net = dragon[dragon['ts_code']==code]['net_amount'].sum() / 10000 if code in dragon['ts_code'].values else 0

        # 打分
        score = 0
        high20 = max(c[-20:-1]) if len(c)>20 else c[-1]

        # 1. 突破放量（25分）
        if c[-1] > high20 and vr >= 2.0:
            score += 25
        elif len(c)>=2 and c[-2] > max(c[-22:-2]) and df['volume_ratio'].iloc[-2] >= 1.8:
            score += 22

        # 2. 均线多头（18分）
        if c[-1] > ma5 > ma10 > ma20:
            score += 18

        # 3. 5日涨幅甜蜜区（15分）
        if 0.12 <= ret5 <= 0.45:
            score += 15
        elif ret5 > 0.45:
            score += 5

        # 4. 高换手（12分）
        if avg_turn10 >= 5:
            score += 12
        elif avg_turn10 >= 3.5:
            score += 6

        # 5. 龙虎榜（10分）
        if dragon_net >= 5000:
            score += 10
        elif dragon_net >= 2000:
            score += 5

        # 6. 市值甜蜜区（5分）
        if 30 <= circ_mv <= 200:
            score += 5

        if score >= 38:
            result.append({
                '代码': code[:6],
                '名称': name,
                '现价': round(price,2),
                '流通市值(亿)': round(circ_mv,1),
                '5日涨幅%': round(ret5*100,1),
                '量比': round(vr,2),
                '10日换手%': round(avg_turn10,1),
                '龙虎榜净买(万)': int(dragon_net),
                '总分': round(score,1)
            })

    bar.empty()

    if not result:
        st.error("今天还没出现特别强的短线票，明天再来！")
    else:
        final = pd.DataFrame(result).sort_values('总分', ascending=False).head(30).reset_index(drop=True)
        final.index += 1
        st.success(f"生成完毕！共选出 {len(final)} 只超短线王炸股（前30名）")
        st.dataframe(final.style.background_gradient(subset=['总分'], cmap='Reds'), height=1000)

        csv = final.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下载前30强CSV",
            data=csv,
            file_name=f"超短线王炸前30_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

st.balloons()
st.markdown("**每周一早跑一次，前10名直接干，1-5天吃肉走人！祝你发财**")
