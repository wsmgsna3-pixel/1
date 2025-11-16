import os
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

# ==============================
# 读取环境变量中的 Token（安全）
# ==============================
TS_TOKEN = os.getenv("TS_TOKEN")
if not TS_TOKEN:
    raise ValueError("没有读取到 TS_TOKEN，请先在系统环境变量里配置。")

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ==============================
# 数据缓存路径
# ==============================
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)


# ==============================
# 获取所有股票列表
# ==============================
def load_stock_list():
    cache_file = f"{DATA_DIR}/stock_list.csv"
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    df = pro.stock_basic(exchange='', list_status='L',
                         fields='ts_code,symbol,name,area,industry,list_date')
    df.to_csv(cache_file, index=False)
    return df


# ==============================
# 拉取单只股票历史行情（带缓存，每天只更新一次）
# ==============================
def load_daily_data(ts_code):
    today = datetime.now().strftime("%Y%m%d")
    cache_file = f"{DATA_DIR}/{ts_code}.csv"

    # 如果已有缓存，检查是否需要更新
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        last_date = df['trade_date'].iloc[0]  # Tushare 是倒序的

        if last_date >= today:
            return df  # 不需要更新

        # 需要补数据
        start_date = (datetime.strptime(last_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        df_new = pro.daily(ts_code=ts_code, start_date=start_date, end_date=today)
        if not df_new.empty:
            df = pd.concat([df_new, df], ignore_index=True)
            df.to_csv(cache_file, index=False)
        return df

    # 没缓存，首次拉取全部数据
    df = pro.daily(ts_code=ts_code, start_date="20180101", end_date=today)
    df.to_csv(cache_file, index=False)
    return df


# ==============================
# 示例：你的选股逻辑（随便写个示例）
# 你之后可以替换成自己的逻辑
# ==============================
def select_stocks():
    stock_list = load_stock_list()
    chosen = []

    for _, row in stock_list.iterrows():
        ts_code = row["ts_code"]
        df = load_daily_data(ts_code)

        # 数据太少跳过
        if df.shape[0] < 20:
            continue

        # 示例策略：最近 10 天均线向上
        df_sort = df.sort_values(by="trade_date")
        df_sort["ma10"] = df_sort["close"].rolling(10).mean()

        if df_sort["ma10"].iloc[-1] > df_sort["ma10"].iloc[-2]:
            chosen.append(ts_code)

    return chosen


# ==============================
# 主函数
# ==============================
if __name__ == "__main__":
    print("正在选股，请稍候……")
    result = select_stocks()
    print("选出的股票：")
    print(result)
