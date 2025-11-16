import streamlit as st
import tushare as ts
import pandas as pd

st.title("升级版选股王")

# 手动输入Token
token = st.text_input("请输入TuShare Token", type="password")
if not token:
    st.warning("请输入Token后才能运行选股")
    st.stop()

ts.set_token(token)
pro = ts.pro_api()

st.write("正在获取股票列表...")

# 1. 获取全部股票基本信息（低积分接口）
stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
# 假设取最新收盘价，10元-200元
df_price = pro.daily(ts_code='', start_date='20251115', end_date='20251115')  # 测试用当天数据
df = pd.merge(stocks, df_price, on='ts_code')

# 排除ST股
df = df[~df['name'].str.contains('ST')]

# 价格区间筛选
df = df[(df['close'] >= 10) & (df['close'] <= 200)]

st.write(f"初筛后股票数量: {len(df)}")

# 2. 获取昨日涨幅并选前500
df['pct_chg'] = df['close'].pct_change() * 100
df_sorted = df.sort_values(by='pct_chg', ascending=False).head(500)

st.write("昨日涨幅前500名股票：")
st.dataframe(df_sorted[['ts_code','name','close','pct_chg']])

# 3. 高级筛选示例（量价、财务指标）
st.write("正在进行高级筛选...")

selected_stocks = []
for ts_code in df_sorted['ts_code']:
    # 示例：获取每日行情和财务指标（高积分接口）
    try:
        daily = pro.daily(ts_code=ts_code, start_date='20251115', end_date='20251115')
        fin = pro.fina_indicator(ts_code=ts_code, start_date='20251115', end_date='20251115')
        # 简单策略：昨日成交量大于均量，ROE > 10%
        if (daily['vol'].iloc[-1] > daily['vol'].mean()) and (fin['roe'].iloc[-1] > 10):
            selected_stocks.append(ts_code)
    except:
        continue

st.write("最终候选股票：")
st.dataframe(df_sorted[df_sorted['ts_code'].isin(selected_stocks)][['ts_code','name','close','pct_chg']])
