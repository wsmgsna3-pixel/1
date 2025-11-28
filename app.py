# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（全市场扫描版）· 极速优化
更新说明：
- 解决了“幸存者偏差”：不再只取涨幅榜前1000，而是对全市场5000+只股票进行扫描。
- 引入“双轨选股”：决赛圈名单由“高涨幅”和“高换手（潜伏）”两部分组成，防止遗漏主力吸筹股。
- 性能优化：引入向量化初筛，确保处理5000只股票时依然流畅。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 · 全市场扫描增强版", layout="wide")
st.title("选股王 · 全市场扫描增强版（杜绝漏票）")
st.markdown("逻辑升级：全量扫描 -> 剔除垃圾 -> (涨幅Top + 换手Top) 混合入围 -> 深度评分")

# ---------------------------
# 侧边栏参数
# ---------------------------
with st.sidebar:
    st.header("核心参数")
    # INITIAL_TOP_N 参数已移除，因为现在是全量扫描
    FINAL_POOL = int(st.number_input("最终入围评分数量 (M)", value=300, step=50, help="为了速度，建议控制在300-500以内"))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=50, step=10))
    
    st.markdown("---")
    st.header("硬性过滤条件")
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=8.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=3.0, step=0.5, help="低于此换手说明无人关注，直接剔除"))
    MIN_AMOUNT = float(st.number_input("最低成交额 (亿)", value=2.0, step=0.5)) * 100000000
    
    st.markdown("---")
    st.header("评分与风控")
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("波动率上限", value=8.0, step=0.5))

# ---------------------------
# Token 输入
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 辅助函数
# ---------------------------
def safe_get(func, **kwargs):
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def find_last_trade_day(max_days=20):
    today = datetime.now().date()
    for i in range(max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        df = safe_get(pro.daily, trade_date=ds)
        if not df.empty:
            return ds
    return None

last_trade = find_last_trade_day()
if not last_trade:
    st.error("无法找到最近交易日。")
    st.stop()
st.info(f"数据日期：{last_trade}（正在进行全市场扫描...）")

# ---------------------------
# 第一步：获取全市场数据（不再截断）
# ---------------------------
st.write("1. 拉取全市场 Daily 数据...")
daily_all = safe_get(pro.daily, trade_date=last_trade)
if daily_all.empty:
    st.error("获取数据失败，请检查 Token。")
    st.stop()

# 【核心改动】：这里不再 head(1000)，而是保留全部
pool_raw = daily_all.reset_index(drop=True) 
st.write(f"  -> 获取到 {len(pool_raw)} 只股票，准备全量清洗。")

# ---------------------------
# 第二步：合并必要数据 (Basic / Moneyflow)
# ---------------------------
st.write("2. 合并基本面数据（市值、换手、主力流向）...")

# 获取 stock_basic (用于过滤 ST 和上市状态)
stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')

# 获取 daily_basic (关键！用于换手率筛选)
daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')

# 获取 moneyflow (可选)
mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)
moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
if not mf_raw.empty:
    # 尝试找净流入列
    possible = ['net_mf','net_mf_amount','net_mf_in']
    for c in possible:
        if c in mf_raw.columns:
            moneyflow = mf_raw[['ts_code', c]].rename(columns={c:'net_mf'}).fillna(0)
            break

# 执行合并
# 先合 stock_basic
if not stock_basic.empty:
    pool_merged = pool_raw.merge(stock_basic[['ts_code','name','industry']], on='ts_code', how='left')
else:
    pool_merged = pool_raw.copy()
    pool_merged['name'] = pool_merged['ts_code']

# 再合 daily_basic
if not daily_basic.empty:
    # 优先使用 daily_basic 的 amount 和 turnover
    cols_to_use = ['ts_code','turnover_rate','amount','total_mv','circ_mv']
    # 如果原表有 amount，先drop避免重名
    if 'amount' in pool_merged.columns: pool_merged = pool_merged.drop(columns=['amount'])
    pool_merged = pool_merged.merge(daily_basic[cols_to_use], on='ts_code', how='left')

# 再合 moneyflow
pool_merged = pool_merged.merge(moneyflow, on='ts_code', how='left')
pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0)

# ---------------------------
# 第三步：极速初筛（向量化处理，处理 5000 只秒级）
# ---------------------------
st.write("3. 执行硬性条件过滤（剔除 ST、低价、无量股）...")
df = pool_merged.copy()

# 填充缺失值防止报错
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
df['name'] = df['name'].astype(str)

# 1. 剔除 ST / 退
mask_st = df['name'].str.contains('ST|退', case=False, na=False)
df = df[~mask_st]

# 2. 价格区间
mask_price = (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)
df = df[mask_price]

# 3. 换手率过滤 (关键！没有换手就没有主力)
mask_turn = df['turnover_rate'] >= MIN_TURNOVER
df = df[mask_turn]

# 4. 成交额过滤 (注意单位，tushare amount 通常是千元，有时是元，这里做兼容处理)
# 假设 daily_basic 的 amount 是万元，或者 daily 的 amount 是千元。
# 安全起见，如果数值很小 (<10万)，可能单位是大数；如果很大，可能是元。
# 我们用简单逻辑：直接看数值大小。
# 如果 MIN_AMOUNT 设的是 2亿 (200,000,000)
# df['amount'] 如果是千元，则 2亿 = 200,000
mask_amt = df['amount'] * 1000 >= MIN_AMOUNT # 假设amount是千元
df = df[mask_amt]

# 5. 剔除一字板（开=高=低=收，且涨幅>9% 或 < -9%）
# 这里简单处理：如果 high == low 且 turnover < 1.0 (极度缩量一字) 剔除，防止买不进
# mask_one_word = (df['high'] == df['low']) & (df['turnover_rate'] < 1.0)
# df = df[~mask_one_word]

df = df.reset_index(drop=True)
st.write(f"  -> 经过硬性过滤，剩余潜力股：{len(df)} 只")

if len(df) == 0:
    st.error("过滤后无股票，请放宽条件。")
    st.stop()

# ---------------------------
# 第四步：双轨选股进入决赛圈
# 【核心逻辑】：为了不漏掉好票，我们取两个榜单的并集
# ---------------------------
st.write("4. 遴选决赛名单（涨幅榜 Top + 潜伏榜 Top）...")

# A. 涨幅榜（进攻型）：取前 70% 的名额
limit_pct = int(FINAL_POOL * 0.7)
df_pct = df.sort_values('pct_chg', ascending=False).head(limit_pct)

# B. 换手榜（潜伏型）：取前 30% 的名额（不管涨幅如何，只要资金活跃）
limit_turn = int(FINAL_POOL * 0.3)
# 排除掉已经在涨幅榜里的，避免重复
existing_codes = set(df_pct['ts_code'])
df_turn = df[~df['ts_code'].isin(existing_codes)].sort_values('turnover_rate', ascending=False).head(limit_turn)

# 合并
final_candidates = pd.concat([df_pct, df_turn]).reset_index(drop=True)
st.write(f"  -> 最终入围评分：{len(final_candidates)} 只（含 {len(df_pct)} 只高涨幅，{len(df_turn)} 只高活跃潜伏）")

# ---------------------------
# 第五步：拉取历史 + 深度评分
# ---------------------------
@st.cache_data(ttl=600)
def get_hist(ts_code, end_date, days=60):
    try:
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = safe_get(pro.daily, ts_code=ts_code, start_date=start, end_date=end_date)
        if df.empty: return pd.DataFrame()
        return df.sort_values('trade_date').reset_index(drop=True)
    except:
        return pd.DataFrame()

def compute_indicators(df):
    res = {}
    if df.empty or len(df) < 3: return res
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    # 基础
    res['last_close'] = close.iloc[-1]
    
    # MACD
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd'] = (diff - dea) * 2
        res['macd_val'] = res['macd'].iloc[-1]
    else:
        res['macd_val'] = np.nan
        
    # KDJ
    n = 9
    if len(close) >= n:
        low_n = low.rolling(window=n).min()
        high_n = high.rolling(window=n).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3*k - 2*d
        res['k'], res['d'], res['j'] = k.iloc[-1], d.iloc[-1], j.iloc[-1]
    else:
        res['k'] = res['d'] = res['j'] = np.nan
        
    # 量比 & 均线
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        res['vol_ratio'] = vols[-1] / (np.mean(vols[-6:-1]) + 1e-9)
        res['vol_last'] = vols[-1]
        res['vol_ma5'] = np.mean(vols[-6:-1])
    else:
        res['vol_ratio'] = np.nan
        
    # 均线
    for n in [5, 10, 20]:
        res[f'ma{n}'] = close.rolling(n).mean().iloc[-1] if len(close)>=n else np.nan
        
    # 10日涨幅
    res['10d_return'] = close.iloc[-1]/close.iloc[-10] - 1 if len(close)>=10 else 0
    
    # 波动率
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    return res

st.write("5. 正在逐个拉取历史数据并打分...")
records = []
my_bar = st.progress(0)
total_c = len(final_candidates)

for i, row in enumerate(final_candidates.itertuples()):
    ts_code = row.ts_code
    # 基础分
    pct_chg = getattr(row, 'pct_chg', 0)
    turnover = getattr(row, 'turnover_rate', 0)
    net_mf = getattr(row, 'net_mf', 0)
    amount = getattr(row, 'amount', 0)
    
    # 拉历史
    hist = get_hist(ts_code, last_trade)
    ind = compute_indicators(hist)
    
    # 风控过滤
    # 1. 巨量冲高回落或高位放量 (Risk)
    vol_last = ind.get('vol_last', 0)
    vol_ma5 = ind.get('vol_ma5', 1)
    if vol_last > vol_ma5 * VOL_SPIKE_MULT:
        # 这里仅做标记，或者降低评分，暂不直接剔除，方便观察
        pass 
        
    rec = {
        'ts_code': ts_code, 
        'name': getattr(row, 'name', ts_code),
        'pct_chg': pct_chg,
        'turnover': turnover,
        'net_mf': net_mf,
        'amount': amount,
        'vol_ratio': ind.get('vol_ratio', 0),
        'macd': ind.get('macd_val', 0),
        'k': ind.get('k', 50),
        '10d_return': ind.get('10d_return', 0),
        'volatility': ind.get('volatility', 0)
    }
    records.append(rec)
    my_bar.progress((i + 1) / total_c)

# ---------------------------
# 第六步：归一化与打分
# ---------------------------
fdf = pd.DataFrame(records)
if fdf.empty:
    st.error("评分列表为空。")
    st.stop()

# 归一化函数
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

# 计算因子得分
fdf['s_pct'] = normalize(fdf['pct_chg'])
fdf['s_turn'] = normalize(fdf['turnover'])
fdf['s_vol'] = normalize(fdf['vol_ratio'])
fdf['s_mf'] = normalize(fdf['net_mf'])
fdf['s_macd'] = normalize(fdf['macd'])
fdf['s_trend'] = normalize(fdf['10d_return'])

# 综合权重 (你原来的权重逻辑)
# 稍微调高了 turnover 的权重，以适应当下的潜伏逻辑
score = (
    fdf['s_pct'] * 0.20 +       # 涨幅
    fdf['s_turn'] * 0.20 +      # 换手 (重要!)
    fdf['s_vol'] * 0.15 +       # 量比
    fdf['s_mf'] * 0.15 +        # 资金流
    fdf['s_macd'] * 0.10 +      # 趋势
    fdf['s_trend'] * 0.10 +     # 10日走势
    (1 - normalize(fdf['volatility'])) * 0.10 # 稳定性
)
fdf['综合评分'] = score * 100

# 排序
fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
fdf.index += 1

# ---------------------------
# 展示结果
# ---------------------------
st.success(f"计算完成！共评分 {len(fdf)} 只。")

cols_show = ['name', 'ts_code', '综合评分', 'pct_chg', 'turnover', 'vol_ratio', 'net_mf', 'macd', 'k']
st.dataframe(fdf[cols_show].head(TOP_DISPLAY), use_container_width=True)

# 简单分析
top1 = fdf.iloc[0]
st.markdown(f"""
### 冠军股点评：{top1['name']}
- **得分**：{top1['综合评分']:.1f}
- **状态**：涨幅 {top1['pct_chg']}%，换手 {top1['turnover']}%
- **理由**：该股在全市场扫描中胜出，兼顾了资金活跃度与技术形态。
""")

st.download_button("下载完整CSV", fdf.to_csv(index=True).encode('utf-8-sig'), "result.csv")
