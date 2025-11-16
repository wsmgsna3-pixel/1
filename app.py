import os
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

# ==========================
# 读取 GitHub Secrets
# ==========================
TS_TOKEN = os.getenv("TS_TOKEN")
if not TS_TOKEN:
    raise ValueError("环境变量 TS_TOKEN 未设置，请在 GitHub Secrets 中配置。")

pro = ts.pro_api(TS_TOKEN)

# ==========================
# 核心选股函数
# ==========================
def fetch_daily(ts_code, start, end):
    """封装日线获取，自动重试"""
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

    # 1）拉全市场基础信息
    stock_basic = pro.stock_basic(exchange='', list_status='L',
                                  fields='ts_code,name,area,industry,list_date')

    # 2）排除 ST / 北交所
    stock_basic = stock_basic[
        (~stock_basic['name'].str.contains('ST')) &
        (~stock_basic['ts_code'].str.startswith('8')) &
        (~stock_basic['ts_code'].str.startswith('4'))
    ]

    results = []

    for _, row in stock_basic.iterrows():
        ts_code = row['ts_code']

        # 3）拉近四个月日线
        df = fetch_daily(ts_code, start_date, end_date)
        if df is None or len(df) < 60:
            continue

        df = df.sort_values(by="trade_date")

        # ================
        #   价格条件
        # ================
        price = df.iloc[-1]['close']
        if price < 10 or price > 200:
            continue

        # ================
        #   均线
        # ================
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()

        # 短期趋势刚启动：5 上穿 10
        if not (df.iloc[-1]['ma5'] > df.iloc[-1]['ma10'] and
                df.iloc[-2]['ma5'] <= df.iloc[-2]['ma10']):
            continue

        # 站上 20 日线
        if price < df.iloc[-1]['ma20']:
            continue

        # ================
        #   成交量过滤
        # ================
        df['vol_ma5'] = df['vol'].rolling(5).mean()
        if df.iloc[-1]['vol'] < df.iloc[-1]['vol_ma5'] * 1.5:
            continue

        # 过去 20 天成交额 > 1 亿
        df['amount'] = df['amount'] / 1e6  # 转成百万
        if df['amount'].tail(20).mean() < 100:
            continue

        # 今日成交额 > 5000 万
        if df.iloc[-1]['amount'] < 50:
            continue

        # ================
        #   排序依据
        # ================
        volume_ratio = df.iloc[-1]['vol'] / df.iloc[-1]['vol_ma5']
        results.append({
            "ts_code": ts_code,
            "name": row['name'],
            "price": price,
            "volume_ratio": round(volume_ratio, 2)
        })

    # ================ 排序：量能最强的放前面 ================
    results = sorted(results, key=lambda x: x['volume_ratio'], reverse=True)

    return pd.DataFrame(results)


# ==========================
# 主程序（GitHub 自动运行用）
# ==========================
def main():
    df = select_stocks()

    if len(df) == 0:
        print("今日无满足条件的股票。")
        return

    print("=== 今日选股结果（2100积分版本） ===")
    print(df)


if __name__ == "__main__":
    main()
