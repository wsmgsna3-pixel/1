import tushare as ts
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta


# ==========================
# Streamlit 页面标题
# ==========================
st.title("📈 选股王 · 2100 积分旗舰版")
st.write("请输入你的 Tushare Token 后开始选股。")


# ==========================
# 手动输入 Token
# ==========================
user_token = st.text_input("请输入你的 TS_TOKEN", type="password")

if not user_token:
    st.stop()

# 初始化 API
pro = ts.pro_api(user_token)


# ==========================
# 核心函数
# ==========================
def fetch_daily(ts_code, start, end):
    for _ in range(3):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
            if df is not None and len(df) > 0:
                return df
        except:
            continue
    return pd.DataFrame()


def select_stocks():
    today = datetime.today()
    start_date = (today - timedelta(days=120)).strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")

    # 全市场股票
    stock_basic = pro.stock_basic(exchange='', list_status='L',
                                  fields='ts_code,name,area,industry,list_date')

    # 去掉 ST 和 北交所
    stock_basic = stock_basic[
        (~stock_basic['name'].str.contains('ST')) &
        (~stock_basic['ts_code'].str.startswith('8')) &
        (~stock_basic['ts_code'].str.startswith('4'))
    ]

    results = []

    for _, row in stock_basic.iterrows():
        ts_code = row['ts_code']

        df = fetch_daily(ts_code, start_date, end_date)
        if df is None or len(df) < 60:
            continue

        df = df.sort_values(by="trade_date")

        # ---- 价格区间过滤 ----
        price = df.iloc[-1]['close']
        if price < 10 or price > 200:
            continue

        # ---- 均线 ----
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()

        # 5 上穿 10
        if not (df.iloc[-1]['ma5'] > df.iloc[-1]['ma10'] and
                df.iloc[-2]['ma5'] <= df.iloc[-2]['ma10']):
            continue

        # 站上 20 日线
        if price < df.iloc[-1]['ma20']:
            continue

        # ---- 成交量过滤 ----
        df['vol_ma5'] = df['vol'].rolling(5).mean()
        if df.iloc[-1]['vol'] < df.iloc[-1]['vol_ma5'] * 1.5:
            continue

        df['amount'] = df['amount'] / 1e6  # 转百万
        if df['amount'].tail(20).mean() < 100:
            continue

        if df.iloc[-1]['amount'] < 50:
            continue

        volume_ratio = df.iloc[-1]['vol'] / df.iloc[-1]['vol_ma5']
        results.append({
            "ts_code": ts_code,
            "name": row['name'],
            "price": price,
            "volume_ratio": round(volume_ratio, 2)
        })

    return pd.DataFrame(sorted(results, key=lambda x: x['volume_ratio'], reverse=True))


# ==========================
# 执行按钮
# ==========================
if st.button("开始选股"):
    with st.spinner("正在分析全市场，请稍候…"):
        df = select_stocks()

    st.success("选股完成！")

    if len(df) == 0:
        st.write("今日无满足条件的股票。")
    else:
        st.dataframe(df, use_container_width=True)
