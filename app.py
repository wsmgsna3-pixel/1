import streamlit as st
import akshare as ak
import pandas as pd
import re

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

st.title("全市场短线王（1-5天操作）")

# Tushare token（可选）
TUSHARE_TOKEN = 'YOUR_TUSHARE_TOKEN'  # 替换为你的 token，或留空

# 缓存历史数据
@st.cache_data(ttl=3600)
def get_hist_data(code):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="")
        return df.tail(15)
    except:
        return pd.DataFrame()

# 获取实时数据（多源 + 列处理）
def get_realtime_data():
    sources = [
        ("新浪", lambda: ak.stock_zh_a_spot()),
        ("腾讯", lambda: ak.stock_zh_a_spot_tx()),
    ]
    if TUSHARE_AVAILABLE and TUSHARE_TOKEN != 'YOUR_TUSHARE_TOKEN':
        sources.append(("Tushare", lambda: fetch_tushare_data()))
    
    for source_name, fetch_func in sources:
        try:
            raw_data = fetch_func()
            if raw_data.empty:
                st.warning(f"{source_name} 返回空数据")
                continue

            # 调试：打印列名（上线后可删）
            st.write(f"**{source_name} 列名**：", raw_data.columns.tolist()[:10])

            data = process_columns(raw_data)
            if not data.empty and 'code' in data.columns and len(data) > 100:
                st.success(f"使用 {source_name} 数据源成功！共 {len(data)} 只股票")
                return data
        except Exception as e:
            st.warning(f"{source_name} 处理失败：{e}")
    
    st.error("所有数据源均失败，请检查网络或 akshare 版本")
    st.stop()
    return pd.DataFrame()

def process_columns(df):
    """终极列名适配：支持中英文、任意顺序"""
    data = df.copy()
    
    # 1. 查找代码列（支持 symbol, 代码, ts_code, 股票代码）
    code_col = None
    for col in data.columns:
        if col.lower() in ['symbol', 'code', '代码', 'ts_code', '股票代码', '证券代码']:
            code_col = col
            break
    if not code_col:
        st.error("未找到股票代码列")
        return pd.DataFrame()
    data['code'] = data[code_col].astype(str)

    # 2. 查找名称列
    name_col = None
    for col in data.columns:
        if col.lower() in ['name', '名称', 'stock_name', '股票名称']:
            name_col = col
            break
    if not name_col:
        st.error("未找到股票名称列")
        return pd.DataFrame()
    data['name'] = data[name_col]

    # 3. 查找今开、昨收、成交量
    open_col = _find_col(data, ['open', '今开', 'open_price', '开盘'])
    close_yest_col = _find_col(data, ['pre_close', '昨收', 'close', '前收盘'])
    volume_col = _find_col(data, ['volume', '成交量', 'vol', '成交额'])

    if not all([open_col, close_yest_col, volume_col]):
        st.error(f"缺少关键列：开盘={open_col}, 昨收={close_yest_col}, 成交量={volume_col}")
        return pd.DataFrame()

    data['open'] = pd.to_numeric(data[open_col], errors='coerce')
    data['close_yesterday'] = pd.to_numeric(data[close_yest_col], errors='coerce')
    data['volume'] = pd.to_numeric(data[volume_col], errors='coerce')

    # 清理无效行
    data = data.dropna(subset=['code', 'open', 'close_yesterday', 'volume'])
    data = data[data['open'] > 0]

    # 保留必要列
    return data[['code', 'name', 'open', 'close_yesterday', 'volume']]

def _find_col(df, candidates):
    """辅助函数：找第一个匹配的列"""
    for col in df.columns:
        if col.lower() in [c.lower() for c in candidates]:
            return col
    return None

def fetch_tushare_data():
    pro = ts.pro_api(TUSHARE_TOKEN)
    basic = pro.stock_basic(fields='ts_code,symbol,name')
    daily = pro.daily_basic(fields='ts_code,open,close,volume,turnover_rate')
    data = pd.merge(basic, daily, on='ts_code')
    data['code'] = data['ts_code'].str.lower().str.replace('.sz', 'sz').str.replace('.sh', 'sh')
    data.rename(columns={'open': 'open', 'close': 'close_yesterday', 'volume': 'volume'}, inplace=True)
    return data[['code', 'name', 'open', 'close_yesterday', 'volume']]

# 主逻辑
if st.button("一键生成全市场潜力股"):
    with st.spinner("正在加载数据源..."):
        data = get_realtime_data()

        # 提取纯代码
        data["pure_code"] = data["code"].astype(str).str.extract(r'(\d{6})')[0]
        data["price_momentum"] = (data["open"] - data["close_yesterday"]) / data["close_yesterday"]
        median_vol = data["volume"].median()
        data["volume_ratio"] = data["volume"] / median_vol

        # 核心过滤
        df = data.copy()
        df = df[df["open"] > 10]
        df = df[df["open"] < df["close_yesterday"] * 1.099]
        df = df[df["price_momentum"] > 0.04]
        df = df[df["volume_ratio"] > 8]

        # 历史过滤
        df["cumulative_rise_10d"] = 0.0
        df["avg_turnover_14d"] = 0.0
        valid = []

        for idx, row in df.iterrows():
            hist = get_hist_data(row["code"])
            if len(hist) < 10:
                continue
            rise_10d = (hist.iloc[-1]["close"] / hist.iloc[-11]["close"]) - 1
            df.loc[idx, "cumulative_rise_10d"] = rise_10d

            if "换手率" in hist.columns:
                avg_turn = hist["换手率"].tail(14).mean()
            else:
                avg_turn = 0
            df.loc[idx, "avg_turnover_14d"] = avg_turn

            if rise_10d <= 0.50 and avg_turn > 3.0:
                valid.append(idx)

        if not valid:
            st.warning("无符合条件股票，建议降低放量要求")
            st.stop()

        df = df.loc[valid]

        # 打分
        df["score"] = df["price_momentum"] * 100 + df["volume_ratio"] + df["avg_turnover_14d"] * 0.5
        df_sorted = df.sort_values("score", ascending=False).head(30)

        st.success(f"找到 {len(df_sorted)} 只短线王")
        st.dataframe(df_sorted[["code", "name", "open", "price_momentum", "volume_ratio", "avg_turnover_14d", "score"]])
        st.download_button("下载CSV", df_sorted.to_csv(index=False).encode('utf-8-sig'), "stocks.csv", "text/csv")
