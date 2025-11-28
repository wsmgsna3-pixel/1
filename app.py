# -*- coding: utf-8 -*-
"""
选股王 · 全市场扫描增强版 V3.2 (位置过滤旗舰版)
更新说明：
1. 核心修复：集成 **60日相对价格位置 (Position_60d)** 指标，并赋予高额反向权重，解决“红彤彤大涨后持续下跌”的高位陷阱问题。
2. 回测优化：新增 **“回测分析 Top K”** 参数，让回测结果更符合你的实际交易习惯（Top 3）。
3. 其余功能（双轨选股、D+30回测、健壮性）保持不变。
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
st.set_page_config(page_title="选股王 · V3.2 位置过滤旗舰版", layout="wide")
st.title("选股王 · V3.2 位置过滤旗舰版（高位陷阱终结者）")
st.markdown("🔥 **核心升级：引入 60日价格位置过滤。回测分析范围可自定义。**")

# ---------------------------
# 辅助函数（必须定义在调用之前）
# ---------------------------
def safe_get(func, **kwargs):
    """安全调用 Tushare API，在出错或返回空时返回带 'ts_code' 的空 DataFrame"""
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=['ts_code']) 
        return df
    except Exception:
        return pd.DataFrame(columns=['ts_code'])

@st.cache_data(ttl=600)
def get_selection_date(backtest_date_input, max_days=20):
    """根据用户输入或默认查找最近交易日作为选股日"""
    
    if backtest_date_input:
        ds = backtest_date_input.strftime("%Y%m%d")
        df = safe_get(pro.daily, trade_date=ds, limit=1) 
        if not df.empty and 'ts_code' in df.columns:
            return ds, True
    
    today = datetime.now().date()
    for i in range(max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        df = safe_get(pro.daily, trade_date=ds, limit=10) 
        if not df.empty and 'ts_code' in df.columns:
            return ds, backtest_date_input is not None
    return None, False

@st.cache_data(ttl=600)
def get_future_prices(ts_code, selection_date, days_ahead=[1, 3, 5, 30]):
    """拉取选股日之后 N 个交易日的收盘价，用于回测"""
    
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_date = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_date = (d0 + timedelta(days=45)).strftime("%Y%m%d")

    hist = safe_get(pro.daily, ts_code=ts_code, start_date=start_date, end_date=end_date)
    hist = hist.sort_values('trade_date').reset_index(drop=True)
    
    results = {}
    
    if hist.empty:
        for n in days_ahead: results[f'Return_D{n}'] = np.nan
        return results

    for n in days_ahead:
        col_name = f'Return_D{n}'
        if len(hist) >= n:
            results[col_name] = hist.iloc[n-1]['close']
        else:
            results[col_name] = np.nan

    return results

# ---------------------------
# 侧边栏参数
# ---------------------------
with st.sidebar:
    st.header("模式与日期选择")
    
    backtest_date = st.date_input(
        "选择**选股日** (留空为最新交易日)", 
        value=None, 
        max_value=datetime.now().date()
    )
    
    st.markdown("---")
    st.header("核心参数")
    FINAL_POOL = int(st.number_input("最终入围评分数量 (M)", value=300, step=50, help="为了速度，建议控制在300-500以内"))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=50, step=10))
    
    # 【新增回测分析范围参数】
    TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=3, step=1, min_value=1, help="仅回测分析这前 K 名股票的平均收益。"))
    
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
# Token 输入与初始化
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()


# ---------------------------
# 核心调用：获取选股日
# ---------------------------
last_trade, is_backtest = get_selection_date(backtest_date)

if not last_trade:
    st.error("无法找到最近交易日。")
    st.stop()
    
if is_backtest:
    st.info(f"✅ 当前模式：**历史回测**，选股日：{last_trade}")
else:
    st.success(f"🚀 当前模式：**实盘选股**，选股日：{last_trade}")


# ---------------------------
# 第一至第四步：数据拉取、清洗、双轨入围（逻辑不变）
# ---------------------------
st.write("1. 拉取全市场 Daily 数据...")
daily_all = safe_get(pro.daily, trade_date=last_trade) 
if daily_all.empty or 'ts_code' not in daily_all.columns:
    st.error("获取全市场数据失败，请检查 Token 或等待数据更新。")
    st.stop()
pool_raw = daily_all.reset_index(drop=True) 
st.write(f"  -> 获取到 {len(pool_raw)} 只股票，准备全量清洗。")

# 第二步：合并必要数据
st.write("2. 合并基本面数据（市值、换手、主力流向）...")
stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')
REQUIRED_BASIC_COLS = ['ts_code','turnover_rate','amount','total_mv','circ_mv']
daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields=','.join(REQUIRED_BASIC_COLS))
mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)
pool_merged = pool_raw.copy()

if not stock_basic.empty and 'name' in stock_basic.columns:
    pool_merged = pool_merged.merge(stock_basic[['ts_code','name','industry']], on='ts_code', how='left')
else:
    pool_merged['name'] = pool_merged['ts_code']

if not daily_basic.empty:
    cols_to_merge = [c for c in REQUIRED_BASIC_COLS if c in daily_basic.columns]
    if len(cols_to_merge) > 1:
        if 'amount' in pool_merged.columns and 'amount' in cols_to_merge: 
            pool_merged = pool_merged.drop(columns=['amount'])
        pool_merged = pool_merged.merge(daily_basic[cols_to_merge], on='ts_code', how='left')
moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
if not mf_raw.empty:
    possible = ['net_mf','net_mf_amount','net_mf_in']
    for c in possible:
        if c in mf_raw.columns:
            moneyflow = mf_raw[['ts_code', c]].rename(columns={c:'net_mf'}).fillna(0)
            break            
if not moneyflow.empty:
    pool_merged = pool_merged.merge(moneyflow, on='ts_code', how='left')
    
pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0) 
pool_merged['turnover_rate'] = pool_merged['turnover_rate'].fillna(0) 

# 第三步：极速初筛
st.write("3. 执行硬性条件过滤（剔除 ST、低价、无量股）...")
df = pool_merged.copy()
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
df['name'] = df['name'].astype(str)
mask_st = df['name'].str.contains('ST|退', case=False, na=False)
df = df[~mask_st]
mask_price = (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)
df = df[mask_price]
mask_turn = df['turnover_rate'] >= MIN_TURNOVER
df = df[mask_turn]
mask_amt = df['amount'] * 1000 >= MIN_AMOUNT 
df = df[mask_amt]
df = df.reset_index(drop=True)
st.write(f"  -> 经过硬性过滤，剩余潜力股：{len(df)} 只")
if len(df) == 0:
    st.error("过滤后无股票，请放宽条件。")
    st.stop()

# 第四步：双轨选股
st.write("4. 遴选决赛名单（涨幅榜 Top + 潜伏榜 Top）...")
limit_pct = int(FINAL_POOL * 0.7)
df_pct = df.sort_values('pct_chg', ascending=False).head(limit_pct).copy()
df_pct['Source_Type'] = 'A-进攻 (高涨幅)' 
limit_turn = FINAL_POOL - len(df_pct)
existing_codes = set(df_pct['ts_code'])
df_turn = df[~df['ts_code'].isin(existing_codes)].sort_values('turnover_rate', ascending=False).head(limit_turn).copy()
df_turn['Source_Type'] = 'B-潜伏 (高换手)' 
final_candidates = pd.concat([df_pct, df_turn]).reset_index(drop=True)
st.write(f"  -> 最终入围评分：{len(final_candidates)} 只（含 {len(df_pct)} 只高涨幅，{len(df_turn)} 只高活跃潜伏）")

# ---------------------------
# 第五步：拉取历史 + 深度评分 (新增 Position_60d 计算)
# ---------------------------
@st.cache_data(ttl=600)
def get_hist(ts_code, end_date, days=60):
    try:
        # 拉取 60 个交易日数据，约 120 个日历日
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = safe_get(pro.daily, ts_code=ts_code, start_date=start, end_date=end_date)
        if df.empty: return pd.DataFrame()
        return df.sort_values('trade_date').reset_index(drop=True)
    except:
        return pd.DataFrame()

def compute_indicators(df):
    res = {}
    if df.empty or len(df) < 3: 
        return res
        
    close = df['close'].astype(float)
    
    res['last_close'] = close.iloc[-1]
    
    # MACD, KDJ, 量比, 10日涨幅, 波动率 (逻辑不变)
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd_val'] = ((diff - dea) * 2).iloc[-1]
    else:
        res['macd_val'] = np.nan
        
    n = 9
    if len(close) >= n:
        low_n = df['low'].rolling(window=n).min()
        high_n = df['high'].rolling(window=n).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        res['k'] = k.iloc[-1]
    else:
        res['k'] = np.nan
        
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        res['vol_ratio'] = vols[-1] / (np.mean(vols[-6:-1]) + 1e-9)
    else:
        res['vol_ratio'] = np.nan
        
    res['10d_return'] = close.iloc[-1]/close.iloc[-10] - 1 if len(close)>=10 else 0
    res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0
    
    # 【V3.2 新增指标】 60日相对价格位置
    if len(df) >= 60:
        hist_60 = df.tail(60)
        min_low = hist_60['low'].min()
        max_high = hist_60['high'].max()
        current_close = hist_60['close'].iloc[-1]
        
        if max_high == min_low:
            res['position_60d'] = 50.0 # 波动为零，设为中性
        else:
            # Position = (收盘价 - 60日最低价) / (60日最高价 - 60日最低价) * 100
            res['position_60d'] = (current_close - min_low) / (max_high - min_low) * 100
    else:
        res['position_60d'] = np.nan # 数据不足
    
    return res

st.write("5. 正在逐个拉取历史数据并打分...")
records = []
my_bar = st.progress(0)
total_c = len(final_candidates)

for i, row in enumerate(final_candidates.itertuples()):
    ts_code = row.ts_code
    
    rec = {
        'ts_code': ts_code, 
        'name': getattr(row, 'name', ts_code),
        'pct_chg': getattr(row, 'pct_chg', 0),
        'turnover': getattr(row, 'turnover_rate', 0),
        'net_mf': getattr(row, 'net_mf', 0),
        'amount': getattr(row, 'amount', 0),
        'Source_Type': getattr(row, 'Source_Type', '未知') 
    }
    
    hist = get_hist(ts_code, last_trade)
    ind = compute_indicators(hist)
    rec.update({
        'vol_ratio': ind.get('vol_ratio', 0),
        'macd': ind.get('macd_val', 0),
        'k': ind.get('k', 50),
        '10d_return': ind.get('10d_return', 0),
        'volatility': ind.get('volatility', 0),
        'position_60d': ind.get('position_60d', np.nan) # 【新增指标】
    })
    
    if is_backtest:
        rec['selection_price'] = ind.get('last_close', np.nan)
        future_prices = get_future_prices(ts_code, last_trade)
        
        for n in [1, 3, 5, 30]:
            future_price = future_prices.get(f'Return_D{n}', np.nan)
            
            if pd.notna(rec['selection_price']) and pd.notna(future_price):
                rec[f'Return_D{n}'] = (future_price / rec['selection_price'] - 1) * 100
            else:
                rec[f'Return_D{n}'] = np.nan
    
    records.append(rec)
    my_bar.progress((i + 1) / total_c)

# ---------------------------
# 第六步：归一化与打分 (V3.2 位置过滤权重)
# ---------------------------
fdf = pd.DataFrame(records)
if fdf.empty:
    st.error("评分列表为空。")
    st.stop()

def normalize(series):
    series_nn = series.dropna() 
    if series_nn.max() == series_nn.min():
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series_nn.min()) / (series_nn.max() - series_nn.min() + 1e-9)

fdf['s_pct'] = normalize(fdf['pct_chg'])
fdf['s_turn'] = normalize(fdf['turnover'])
fdf['s_vol'] = normalize(fdf['vol_ratio'])
fdf['s_mf'] = normalize(fdf['net_mf'])
fdf['s_macd'] = normalize(fdf['macd'])
fdf['s_trend'] = normalize(fdf['10d_return'])
fdf['s_position'] = fdf['position_60d'] / 100 # 将 0-100% 转化为 0-1

# V3.2 稳定趋势 + 位置过滤权重配置
w_pct = 0.05        # 【大幅降低】 当日涨幅权重，削弱追高风险
w_turn = 0.15       # 换手率权重 
w_vol = 0.05        # 量比权重
w_mf = 0.15         # 资金流向权重（核心指标）
w_macd = 0.10       # MACD形态权重
w_trend = 0.15      # 10日涨幅权重 (看重持续趋势)
w_volatility = 0.10 # 波动率反向（稳定性）权重
w_position = 0.25   # 【极高】 60日位置反向权重 (过滤高位股)

# 确保总和为 1.00
score = (
    fdf['s_pct'] * w_pct +       
    fdf['s_turn'] * w_turn +      
    fdf['s_vol'] * w_vol +       
    fdf['s_mf'] * w_mf +        
    fdf['s_macd'] * w_macd +      
    fdf['s_trend'] * w_trend +     
    (1 - normalize(fdf['volatility'])) * w_volatility + # 稳定性是反向指标
    (1 - fdf['s_position']) * w_position                # 【新增】 价格位置是反向指标
)
fdf['综合评分'] = score * 100

fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
fdf.index += 1

# ---------------------------
# 第七步：展示结果 (使用 TOP_BACKTEST 参数)
# ---------------------------
st.success(f"计算完成！共评分 {len(fdf)} 只。")

cols_show = ['name', 'ts_code', '综合评分', 'Source_Type', 'pct_chg', 'turnover', 'vol_ratio', 'net_mf', 'macd', 'k', 'position_60d']

if is_backtest:
    st.header(f"回测结果分析（Top {TOP_BACKTEST}）")
    # 【使用动态参数 TOP_BACKTEST】
    top_k = fdf.head(TOP_BACKTEST) 
    
    for n in [1, 3, 5, 30]:
        col = f'Return_D{n}'
        if col in top_k.columns:
            avg_return = top_k[col].mean()
            hit_rate = (top_k[col] > 0).sum() / len(top_k[col].dropna()) * 100
            st.metric(f"Top {TOP_BACKTEST}：D+{n} 平均收益 / 准确率", f"{avg_return:.2f}%", help=f" Top {TOP_BACKTEST} 中有 {hit_rate:.1f}% 的股票在 {n} 个交易日内上涨。")
            cols_show.insert(4, col)

st.header("选股结果列表")
st.dataframe(fdf[cols_show].head(TOP_DISPLAY), use_container_width=True, column_config={
    "Return_D1": st.column_config.NumberColumn("D+1 回报率(%)", format="%.2f"),
    "Return_D3": st.column_config.NumberColumn("D+3 回报率(%)", format="%.2f"),
    "Return_D5": st.column_config.NumberColumn("D+5 回报率(%)", format="%.2f"),
    "Return_D30": st.column_config.NumberColumn("D+30 回报率(%)", format="%.2f"),
    "position_60d": st.column_config.NumberColumn("60日位置(%)", format="%.1f"), # 新增列展示
    "综合评分": st.column_config.ProgressColumn("综合评分", format="%.1f", min_value=0, max_value=100),
    "pct_chg": st.column_config.NumberColumn("当日涨幅(%)", format="%.2f"),
    "turnover": st.column_config.NumberColumn("换手率(%)", format="%.2f")
})

st.download_button("下载完整CSV", fdf.to_csv(index=True).encode('utf-8-sig'), f"选股王_V3.2_结果_{last_trade}.csv")
