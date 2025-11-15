!pip install tushare pandas

import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

# ===== 1. TuShare token =====
# 去 https://tushare.pro 注册，然后在个人中心复制 token
ts.set_token("在这里填你的token")
pro = ts.pro_api()

# ===== 2. 获取今日基本行情 =====
# 获取所有 A 股列表
stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')
codes = stocks['ts_code'].tolist()

print(f"共载入股票数量：{len(codes)}")

# 获取今日行情
today = datetime.now().strftime('%Y%m%d')
daily = pro.daily(trade_date=today)

# 如果当天未开盘，则抓最近一个交易日
if daily is None or daily.empty:
    import time
    time.sleep(1)
    daily = pro.daily(trade_date=(datetime.now() - timedelta(days=1)).strftime('%Y%m%d'))

print(f"今日行情数量：{len(daily)}")

# ===== 3. 获取过去 10 天 K 线 =====
start_date = (datetime.now() - timedelta(days=20)).strftime('%Y%m%d')
hist = pro.daily(start_date=start_date, end_date=today)

# ===== 4. 处理 10 日历史指标 =====
# 计算每只股票：10 日涨幅、10 日均换手、昨日最高、昨日成交量
hist['trade_date'] = pd.to_datetime(hist['trade_date'])
hist_sorted = hist.sort_values(['ts_code', 'trade_date'])

def calc_features(df):
    df = df.tail(10)
    if len(df) < 10:
        return pd.Series({
            '10d_return': None,
            '10d_avg_turnover': None,
            'high_yesterday': None,
            'volume_yesterday': None
        })
    return pd.Series({
        '10d_return': (df.iloc[-1]['close'] / df.iloc[0]['open']) - 1,
        '10d_avg_turnover': df['turnover_rate'].mean(),
        'high_yesterday': df.iloc[-2]['high'],
        'volume_yesterday': df.iloc[-2]['vol']
    })

features = hist_sorted.groupby('ts_code').apply(calc_features).reset_index()

# ===== 5. 合并数据 =====
df = daily.merge(features, on='ts_code', how='inner')

# ===== 6. 应用终极五条 =====
df = df[df['open'] > 10]                                  # 条件 1：股价 > 10
df = df[df['10d_return'] <= 0.50]                         # 条件 2：10 天涨幅 ≤ 50%
df = df[df['10d_avg_turnover'] >= 3.0]                    # 条件 3：10 日平均换手 ≥ 3%
df = df[df['vol'] > 3 * df['volume_yesterday']]           # 条件 4：今日放量 > 昨天 3 倍
df = df[df['open'] > df['high_yesterday']]                # 条件 5：开盘突破昨日最高价

if df.empty:
    print("今日无符合条件的股票。")
else:
    # ===== 7. 打分排序 =====
    df['score'] = (df['open'] - df['pre_close']) / df['pre_close'] * 100 + \
                  df['vol'] / df['volume_yesterday'] * 10
    df = df.sort_values('score', ascending=False)

    print("终极五条选股结果：")
    print(df[['ts_code', 'open', '10d_return', '10d_avg_turnover', 'vol', 'score']])

    # ===== 8. 导出 CSV =====
    df.to_csv("短线王_终极五条_TuShare.csv", index=False, encoding="utf-8-sig")
    print("\nCSV 已生成：短线王_终极五条_TuShare.csv")
