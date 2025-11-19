import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="选股王 · 10000 积分旗舰 BC 混合增强版", layout="wide")

# ===== 1. 输入 Token =====
st.title("选股王 · 10000 积分旗舰 BC 混合增强版")

token = st.text_input("请输入 Tushare Token：", type="password")
if not token:
    st.stop()

ts.set_token(token)
pro = ts.pro_api()

# ===== 2. 通用安全调用函数 =====
def safe_api_call(func, default_df):
    try:
        df = func()
        if df is None or df.empty:
            return default_df
        return df
    except:
        return default_df

# ===== 3. 最近交易日 =====
def get_last_trade_date():
    today = datetime.now().strftime("%Y%m%d")
    cal = safe_api_call(
        lambda: pro.trade_cal(exchange='SSE', start_date='20240101', end_date=today),
        pd.DataFrame(columns=['cal_date','is_open'])
    )
    if cal.empty:
        return today
    cal = cal[cal["is_open"] == 1]
    return cal.iloc[-1]['cal_date']

last_trade = get_last_trade_date()
st.write(f"最近交易日：{last_trade}")

# ===== 4. 获取 daily（初筛） =====
st.write("正在拉取当日 daily（涨幅榜）作为初筛...")

daily_all = safe_api_call(
    lambda: pro.daily(trade_date=last_trade),
    pd.DataFrame(columns=["ts_code","pct_chg","close","open","high","low","vol","amount"])
)

if daily_all.empty:
    st.error("无法获取 daily 行情")
    st.stop()

st.write(f"当日记录：{len(daily_all)}，选涨幅前 1000 作为初筛。")

daily_all = daily_all.sort_values("pct_chg", ascending=False).head(1000)

# ===== 5. 获取 stock_basic / daily_basic =====
st.write("加载基础数据...")

stock_basic = safe_api_call(
    lambda: pro.stock_basic(exchange='', fields="ts_code,name,industry"),
    pd.DataFrame(columns=["ts_code","name","industry"])
)

daily_basic = safe_api_call(
    lambda: pro.daily_basic(trade_date=last_trade),
    pd.DataFrame(columns=["ts_code","turnover_rate","pe","pb","ps","dv_ratio","total_mv"])
)

# ===== 6. moneyflow 修复（本次最重要） =====
st.write("加载 moneyflow（主力净流）...")

moneyflow = safe_api_call(
    lambda: pro.moneyflow(trade_date=last_trade),
    pd.DataFrame()
)

# ▶ 若空 → 建标准形状
if moneyflow is None or moneyflow.empty:
    moneyflow = pd.DataFrame(columns=["ts_code","net_mf"])

# ▶ 若缺列 → 补列
if "ts_code" not in moneyflow.columns:
    moneyflow["ts_code"] = None

if "net_mf" not in moneyflow.columns:
    moneyflow["net_mf"] = 0

# ===== 7. 合并初筛池 =====
pool = daily_all.merge(stock_basic, on='ts_code', how='left') \
    .merge(daily_basic, on='ts_code', how='left') \
    .merge(moneyflow[["ts_code","net_mf"]], on='ts_code', how='left')

pool["net_mf"] = pool["net_mf"].fillna(0)

# ===== 8. 清洗 =====
st.write("对初筛池进行清洗...")

clean = pool.copy()
# 确保一定有 close 列
if "close" not in clean.columns:
    clean["close"] = clean["pre_close"] if "pre_close" in clean.columns else 0
clean = clean[~clean["name"].str.contains("ST", na=False)]
clean = clean[clean["open"] > 3]
clean = clean[clean["close"] > 3]
clean = clean[clean["high"] != clean["low"]]
clean = clean[clean["amount"] > 2e7]
clean = clean[clean["turnover_rate"] > 2]

clean = clean.dropna(subset=["pct_chg"])

st.write(f"清洗后剩余：{len(clean)}")

if clean.empty:
    st.warning("无候选股")
    st.stop()

# ===== 9. 历史行情与指标 =====
@st.cache_data(show_spinner=False)
def get_hist(ts_code, end_date, days=90):
    end = datetime.strptime(end_date, "%Y%m%d")
    start = (end - timedelta(days=days+30)).strftime("%Y%m%d")
    df = safe_api_call(
        lambda: pro.daily(ts_code=ts_code, start_date=start, end_date=end_date),
        pd.DataFrame(columns=["trade_date","open","close","high","low","vol"])
    )
    if df.empty:
        return df
    df = df.sort_values("trade_date")
    return df

def compute_indicators(df):
    if df is None or df.empty or len(df) < 20:
        return {}

    close = df["close"].values
    vol = df["vol"].values

    ma5 = close[-5:].mean()
    ma10 = close[-10:].mean()
    ma20 = close[-20:].mean()

    vol_ratio = vol[-1] / (vol[-6:-1].mean() + 1e-9)

    # EMA
    def EMA(arr, span):
        alpha = 2 / (span + 1)
        ema = []
        prev = arr[0]
        for x in arr:
            prev = alpha*x + (1-alpha)*prev
            ema.append(prev)
        return np.array(ema)

    # MACD
    ema12 = EMA(close, 12)
    ema26 = EMA(close, 26)
    dif = ema12 - ema26
    dea = EMA(dif, 9)
    macd = (dif[-1] - dea[-1]) * 2

    RSL = close[-1] / close[-20] - 1 if len(close) >= 20 else np.nan
    ten_ret = close[-1] / close[-11] - 1 if len(close) >= 11 else np.nan

    return {
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "vol_ratio": vol_ratio,
        "macd": macd,
        "rsl": RSL,
        "10d_return": ten_ret
    }

# ===== 10. BC 混合评分 =====
st.write("计算 BC 混合因子评分...")

records = []
pbar = st.progress(0)

for i, row in enumerate(clean.itertuples()):
    ts_code = row.ts_code
    name = row.name

    hist = get_hist(ts_code, last_trade, 90)
    ind = compute_indicators(hist)

    score = 0

    # B（趋势）
    if "rsl" in ind and ind["rsl"] > 0:
        score += ind["rsl"] * 60
    if "10d_return" in ind and ind["10d_return"] > 0:
        score += ind["10d_return"] * 40

    # C（确认）
    if "macd" in ind and ind["macd"] > 0:
        score += ind["macd"] * 10

    # 放量
    if "vol_ratio" in ind and ind["vol_ratio"] > 1.3:
        score += (ind["vol_ratio"] - 1.3) * 5

    records.append({
        "ts_code": ts_code,
        "name": name,
        "pct_chg": row.pct_chg,
        "score": score
    })

    pbar.progress((i+1) / len(clean))

pbar.progress(1.0)

fdf = pd.DataFrame(records)
fdf = fdf.sort_values("score", ascending=False)

# ===== 11. 输出 =====
st.subheader("最终评分结果（前 20）")
st.dataframe(fdf.head(20), use_container_width=True)

out_csv = fdf.head(20).to_csv(index=False).encode("utf-8")

st.download_button(
    "下载评分结果（前20）CSV",
    data=out_csv,
    file_name=f"score_result_{last_trade}.csv",
    mime="text/csv"
)
