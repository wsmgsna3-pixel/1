# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰 · BC 混合终极版（位置过滤 + 市值区间）
说明：
- 目标：短线爆发 (B) + 妖股捕捉 (C)，持股 1-5 天
- 新增：强制市值区间（默认 20亿 - 300亿）、位置过滤（避免已涨过高的尾段）
- 在界面输入 Tushare Token（仅本次运行使用）
- 高级接口若无权限会自动降级（proxy_money）
- 运行前请备份现有 app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 页面与说明
# ---------------------------
st.set_page_config(page_title="选股王 · 10000旗舰（BC终极）", layout="wide")
st.title("选股王 · 10000 积分旗舰（BC 混合终极版）")
st.markdown("输入你的 Tushare Token（仅本次运行使用）。阈值可在侧栏调整。")

# ---------------------------
# 侧边栏参数（实时调整）
# ---------------------------
with st.sidebar:
    st.header("参数设置（实时）")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=500, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=3.0, step=0.5))
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=200_000_000.0, step=50_000_000.0))
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=8.0, step=0.5))
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    st.markdown("---")
    st.subheader("市值与位置过滤")
    MIN_TOTAL_MV_YUAN = float(st.number_input("市值下限 (元)", value=20_0000_0000.0, step=1_0000_0000.0))  # default 20亿
    MAX_TOTAL_MV_YUAN = float(st.number_input("市值上限 (元)", value=300_0000_0000.0, step=50_0000_0000.0))  # default 300亿
    MAX_RUNUP_10D = float(st.number_input("过去10日涨幅上限 (%)（位置过滤）", value=100.0, step=10.0))
    MAX_RUNUP_20D = float(st.number_input("过去20日涨幅上限 (%)（位置过滤）", value=150.0, step=10.0))
    st.caption("注：单项过热（如最近10日涨幅>100%）会被排除，避免高位接力。")

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
# 安全调用与缓存
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
    st.error("无法找到最近交易日，检查网络或 Token 权限。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")

# ---------------------------
# 初筛：当日涨幅榜
# ---------------------------
st.write("拉取当日 daily（涨幅榜）作为初筛...")
daily_all = safe_get(pro.daily, trade_date=last_trade)
if daily_all.empty:
    st.error("无法获取当日 daily 数据（Tushare 返回空）。请确认 Token 权限。")
    st.stop()

daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
st.write(f"当日记录：{len(daily_all)}，取涨幅前 {INITIAL_TOP_N} 作为初筛。")
pool0 = daily_all.head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

# ---------------------------
# 尝试加载高级接口（若权限）
# ---------------------------
st.write("尝试加载 stock_basic / daily_basic / moneyflow / ths_member / chip 等高级接口（若权限允许）...")
stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')
daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)
try:
    ths_hot = safe_get(pro.ths_member)
except Exception:
    ths_hot = pd.DataFrame()

# moneyflow 处理
if not mf_raw.empty:
    possible = ['net_mf','net_mf_amount','net_mf_in','net_mf_out']
    col = None
    for c in possible:
        if c in mf_raw.columns:
            col = c; break
    if col is None:
        numcols = [c for c in mf_raw.columns if c != 'ts_code' and pd.api.types.is_numeric_dtype(mf_raw[c])]
        col = numcols[0] if numcols else None
    if col:
        moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)
    else:
        moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
else:
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    st.warning("moneyflow 未获取到，将把主力流向因子置为 0（若有权限请确认 Token/积分）。")

# ---------------------------
# 合并基本信息（safe helper）
# ---------------------------
def safe_merge_pool(pool_df, other_df, cols):
    pool = pool_df.set_index('ts_code').copy()
    if other_df is None or other_df.empty:
        for c in cols:
            pool[c] = np.nan
        return pool.reset_index()
    if 'ts_code' not in other_df.columns:
        try:
            other_df = other_df.reset_index()
        except:
            for c in cols:
                pool[c] = np.nan
            return pool.reset_index()
    for c in cols:
        if c not in other_df.columns:
            other_df[c] = np.nan
    try:
        joined = pool.join(other_df.set_index('ts_code')[cols], how='left')
    except Exception:
        for c in cols:
            pool[c] = np.nan
        return pool.reset_index()
    for c in cols:
        if c not in joined.columns:
            joined[c] = np.nan
    return joined.reset_index()

# merge stock_basic
if not stock_basic.empty:
    keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_basic.columns]
    try:
        pool0 = pool0.merge(stock_basic[keep], on='ts_code', how='left')
    except Exception:
        pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
else:
    pool0['name'] = pool0['ts_code']; pool0['industry'] = ''

# merge daily_basic
pool_merged = safe_merge_pool(pool0, daily_basic, ['turnover_rate','amount','total_mv','circ_mv'])

# merge moneyflow robustly
if moneyflow.empty:
    moneyflow = pd.DataFrame({'ts_code': pool_merged['ts_code'].tolist(), 'net_mf': [0.0]*len(pool_merged)})
else:
    if 'ts_code' not in moneyflow.columns:
        moneyflow['ts_code'] = None
try:
    pool_merged = pool_merged.set_index('ts_code').join(moneyflow.set_index('ts_code'), how='left').reset_index()
except Exception:
    if 'net_mf' not in pool_merged.columns:
        pool_merged['net_mf'] = 0.0
if 'net_mf' not in pool_merged.columns:
    pool_merged['net_mf'] = 0.0
pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0.0)

# ---------------------------
# 基本清洗（ST / 停牌 / 价格区间 / 一字板 / 换手 / 成交额 / 市值区间）
# ---------------------------
st.write("对初筛池进行清洗（ST/停牌/价格/一字板/换手/成交额/市值区间等）...")
clean_list = []
pbar = st.progress(0)
for i, r in enumerate(pool_merged.itertuples()):
    ts = getattr(r, 'ts_code')
    # try get vol
    try:
        vol_df = safe_get(pro.daily, ts_code=ts, trade_date=last_trade)
        vol = vol_df.get('vol', pd.Series([0])).iloc[0] if not vol_df.empty else getattr(r, 'vol') if 'vol' in pool_merged.columns else 0
    except:
        vol = getattr(r, 'vol') if 'vol' in pool_merged.columns else 0

    close = getattr(r, 'close', np.nan)
    open_p = getattr(r, 'open', np.nan)
    pre_close = getattr(r, 'pre_close', np.nan)
    pct = getattr(r, 'pct_chg', np.nan)
    amount = getattr(r, 'amount', np.nan)
    turnover = getattr(r, 'turnover_rate', np.nan)
    total_mv = getattr(r, 'total_mv', np.nan)
    name = getattr(r, 'name', ts)

    # skip no trading
    if vol == 0 or (isinstance(amount,(int,float)) and amount == 0):
        pbar.progress((i+1)/len(pool_merged)); continue

    # price filter
    if pd.isna(close): pbar.progress((i+1)/len(pool_merged)); continue
    if (close < MIN_PRICE) or (close > MAX_PRICE): pbar.progress((i+1)/len(pool_merged)); continue

    # exclude ST / delist
    if isinstance(name, str) and (('ST' in name.upper()) or ('退' in name)):
        pbar.progress((i+1)/len(pool_merged)); continue

    # one-word board (open==high==low==pre_close)
    try:
        high = getattr(r, 'high', np.nan); low = getattr(r, 'low', np.nan)
        if (not pd.isna(open_p) and not pd.isna(high) and not pd.isna(low) and not pd.isna(pre_close)):
            if (open_p == high == low == pre_close):
                pbar.progress((i+1)/len(pool_merged)); continue
    except:
        pass

    # market cap normalization and filter
    try:
        tv = total_mv
        if not pd.isna(tv):
            tv = float(tv)
            # some APIs return total_mv in 万元 units; if it's large number like >1e6, treat as 万元
            if tv > 1e6:
                tv_yuan = tv * 10000.0
            else:
                tv_yuan = tv
            # enforce user-specified total mv range
            if tv_yuan < MIN_TOTAL_MV_YUAN or tv_yuan > MAX_TOTAL_MV_YUAN:
                pbar.progress((i+1)/len(pool_merged)); continue
    except:
        pass

    # turnover
    if not pd.isna(turnover):
        try:
            if float(turnover) < MIN_TURNOVER: pbar.progress((i+1)/len(pool_merged)); continue
        except:
            pass

    # amount normalization
    if not pd.isna(amount):
        amt = amount
        if amt > 0 and amt < 1e5:
            amt = amt * 10000.0
        if amt < MIN_AMOUNT: pbar.progress((i+1)/len(pool_merged)); continue

    # exclude yesterday down
    try:
        if float(pct) < 0: pbar.progress((i+1)/len(pool_merged)); continue
    except:
        pass

    clean_list.append(r)
    pbar.progress((i+1)/len(pool_merged))
pbar.progress(1.0)

clean_df = pd.DataFrame([dict(zip(r._fields, r)) for r in clean_list])
st.write(f"清洗后候选数量：{len(clean_df)} （将从中取涨幅前 {FINAL_POOL} 进入评分阶段）")
if len(clean_df) == 0:
    st.error("清洗后没有候选，建议放宽条件或检查接口权限。")
    st.stop()

# ---------------------------
# 取涨幅前 FINAL_POOL 进入评分
# ---------------------------
clean_df = clean_df.sort_values('pct_chg', ascending=False).head(int(FINAL_POOL)).reset_index(drop=True)
st.write(f"用于评分的池子大小：{len(clean_df)}")

# ---------------------------
# 历史拉取与指标计算
# ---------------------------
@st.cache_data(ttl=600)
def get_hist(ts_code, end_date, days=60):
    try:
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = safe_get(pro.daily, ts_code=ts_code, start_date=start, end_date=end_date)
        if df.empty:
            return pd.DataFrame()
        df = df.sort_values('trade_date').reset_index(drop=True)
        return df
    except:
        return pd.DataFrame()

def compute_indicators(df):
    res = {}
    if df.empty or len(df) < 3:
        return res
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    # last close
    try: res['last_close'] = close.iloc[-1]
    except: res['last_close'] = np.nan

    # MA
    for n in (5,10,20):
        if len(close) >= n:
            res[f'ma{n}'] = close.rolling(window=n).mean().iloc[-1]
        else:
            res[f'ma{n}'] = np.nan

    # MACD
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_val = (diff - dea) * 2
        res['macd'] = macd_val.iloc[-1]; res['diff'] = diff.iloc[-1]; res['dea'] = dea.iloc[-1]
    else:
        res['macd'] = res['diff'] = res['dea'] = np.nan

    # KDJ
    n = 9
    if len(close) >= n:
        low_n = low.rolling(window=n).min()
        high_n = high.rolling(window=n).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
        rsv = rsv.fillna(50)
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3*k - 2*d
        res['k'] = k.iloc[-1]; res['d'] = d.iloc[-1]; res['j'] = j.iloc[-1]
    else:
        res['k'] = res['d'] = res['j'] = np.nan

    # vol ratio and metrics
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        avg_prev5 = np.mean(vols[-6:-1])
        res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9)
        res['vol_last'] = vols[-1]
        res['vol_ma5'] = avg_prev5
    else:
        res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan

    # 10d & 20d return (runup)
    if len(close) >= 10:
        res['10d_return'] = close.iloc[-1] / close.iloc[-10] - 1
    else:
        res['10d_return'] = np.nan
    if len(close) >= 20:
        res['20d_return'] = close.iloc[-1] / close.iloc[-20] - 1
    else:
        res['20d_return'] = np.nan

    # prev3_sum for down-then-bounce detection
    if 'pct_chg' in df.columns and len(df) >= 4:
        try:
            pct = df['pct_chg'].astype(float)
            res['prev3_sum'] = pct.iloc[-4:-1].sum()
        except:
            res['prev3_sum'] = np.nan
    else:
        res['prev3_sum'] = np.nan

    # volatility (std of last 10 pct_chg)
    try:
        if 'pct_chg' in df.columns and len(df) >= 10:
            res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()
        else:
            res['volatility_10'] = np.nan
    except:
        res['volatility_10'] = np.nan

    return res

# ---------------------------
# 为评分池逐票拉历史并计算指标
# ---------------------------
st.write("评分池逐票拉历史并计算指标（启用缓存）...")
records = []
pbar2 = st.progress(0)
for idx, row in enumerate(clean_df.itertuples()):
    ts_code = getattr(row, 'ts_code')
    name = getattr(row, 'name', ts_code)
    pct_chg = getattr(row, 'pct_chg', 0.0)
    amount = getattr(row, 'amount', np.nan)
    if amount is not None and not pd.isna(amount) and amount > 0 and amount < 1e5:
        amount = amount * 10000.0
    turnover_rate = getattr(row, 'turnover_rate', np.nan)
    net_mf = float(getattr(row, 'net_mf', 0.0))

    hist = get_hist(ts_code, last_trade, days=60)
    ind = compute_indicators(hist)

    vol_ratio = ind.get('vol_ratio', np.nan)
    ten_return = ind.get('10d_return', np.nan)
    twenty_return = ind.get('20d_return', np.nan)
    ma5 = ind.get('ma5', np.nan)
    ma10 = ind.get('ma10', np.nan)
    ma20 = ind.get('ma20', np.nan)
    macd = ind.get('macd', np.nan)
    k, d, j = ind.get('k', np.nan), ind.get('d', np.nan), ind.get('j', np.nan)
    last_close = ind.get('last_close', np.nan)
    vol_last = ind.get('vol_last', np.nan)
    vol_ma5 = ind.get('vol_ma5', np.nan)
    prev3_sum = ind.get('prev3_sum', np.nan)
    volatility_10 = ind.get('volatility_10', np.nan)

    # 资金强度代理（price move * vol_ratio * turnover）
    try:
        proxy_money = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
    except:
        proxy_money = 0.0

    rec = {
        'ts_code': ts_code, 'name': name, 'pct_chg': pct_chg,
        'amount': amount if not pd.isna(amount) else 0.0,
        'turnover_rate': turnover_rate if not pd.isna(turnover_rate) else np.nan,
        'net_mf': net_mf,
        'vol_ratio': vol_ratio if not pd.isna(vol_ratio) else np.nan,
        '10d_return': ten_return if not pd.isna(ten_return) else np.nan,
        '20d_return': twenty_return if not pd.isna(twenty_return) else np.nan,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
        'macd': macd, 'k': k, 'd': d, 'j': j,
        'last_close': last_close, 'vol_last': vol_last, 'vol_ma5': vol_ma5,
        'prev3_sum': prev3_sum, 'volatility_10': volatility_10,
        'proxy_money': proxy_money
    }

    records.append(rec)
    pbar2.progress((idx+1)/len(clean_df))
pbar2.progress(1.0)

fdf = pd.DataFrame(records)
if fdf.empty:
    st.error("评分计算失败或无数据，请检查 Token 权限与接口。")
    st.stop()

# ---------------------------
# 风险过滤（下跌途中大阳 / 巨量冲高 / 高位大阳 / 极端波动 / 位置过滤）
# ---------------------------
st.write("执行增强风险过滤（含位置与市值热度过滤）...")
try:
    before_cnt = len(fdf)

    # 高位大阳 -> last_close > ma20*1.10 且 pct_chg > HIGH_PCT_THRESHOLD
    if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):
        mask_high_big = (fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
        fdf = fdf[~mask_high_big]

    # 下跌途中反抽 -> prev3_sum < 0 且 pct_chg > HIGH_PCT_THRESHOLD
    if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
        mask_down_rebound = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
        fdf = fdf[~mask_down_rebound]

    # 巨量放量大阳 -> vol_last > vol_ma5 * VOL_SPIKE_MULT
    if all(c in fdf.columns for c in ['vol_last','vol_ma5']):
        mask_vol_spike = (fdf['vol_last'] > (fdf['vol_ma5'] * VOL_SPIKE_MULT))
        fdf = fdf[~mask_vol_spike]

    # 极端波动 -> volatility_10 > VOLATILITY_MAX
    if 'volatility_10' in fdf.columns:
        mask_volatility = fdf['volatility_10'] > VOLATILITY_MAX
        fdf = fdf[~mask_volatility]

    # 位置过滤：过去10/20日涨幅过高剔除
    if '10d_return' in fdf.columns:
        mask_runup10 = fdf['10d_return'] * 100.0 > MAX_RUNUP_10D
        fdf = fdf[~mask_runup10]
    if '20d_return' in fdf.columns:
        mask_runup20 = fdf['20d_return'] * 100.0 > MAX_RUNUP_20D
        fdf = fdf[~mask_runup20]

    after_cnt = len(fdf)
    st.write(f"过滤完成：{before_cnt} -> {after_cnt}（如过严请调整侧栏阈值）")
except Exception as e:
    st.warning(f"风险过滤模块运行异常，已跳过该步骤。错误：{e}")

# ---------------------------
# RSL（相对强弱） —— 基于池内 10d_return 的相对表现
# ---------------------------
if '10d_return' in fdf.columns:
    try:
        market_mean_10d = fdf['10d_return'].replace([np.inf,-np.inf], np.nan).dropna().mean()
        if np.isnan(market_mean_10d) or abs(market_mean_10d) < 1e-9:
            market_mean_10d = 1e-9
        fdf['rsl'] = fdf['10d_return'] / market_mean_10d
    except:
        fdf['rsl'] = 1.0
else:
    fdf['rsl'] = 1.0

# ---------------------------
# 子指标归一化（稳健）
# ---------------------------
def norm_col(s):
    s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    mn = s.min(); mx = s.max()
    if mx - mn < 1e-9:
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)

fdf['s_pct'] = norm_col(fdf.get('pct_chg', pd.Series([0]*len(fdf))))
fdf['s_volratio'] = norm_col(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))
fdf['s_turn'] = norm_col(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))
# moneyflow 优先，否则 proxy_money
if 'net_mf' in fdf.columns and fdf['net_mf'].abs().sum() > 0:
    fdf['s_money'] = norm_col(fdf.get('net_mf', pd.Series([0]*len(fdf))))
else:
    fdf['s_money'] = norm_col(fdf.get('proxy_money', pd.Series([0]*len(fdf))))
fdf['s_amount'] = norm_col(fdf.get('amount', pd.Series([0]*len(fdf))))
# 新鲜度：10d_return 越高越新鲜（但已位置过滤）
fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))
fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
fdf['s_rsl'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))
fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))

# ---------------------------
# 综合评分（BC 混合权重：偏向“新鲜动能 + 相对强弱”，降低对“单纯已涨幅”的偏好）
# ---------------------------
w_pct = 0.16
w_volratio = 0.18
w_turn = 0.12
w_money = 0.14
w_10d = 0.16   # 提高新鲜度权重
w_macd = 0.06
w_rsl = 0.12
w_volatility = 0.06

fdf['综合评分'] = (
    fdf['s_pct'] * w_pct +
    fdf['s_volratio'] * w_volratio +
    fdf['s_turn'] * w_turn +
    fdf['s_money'] * w_money +
    fdf['s_10d'] * w_10d +
    fdf['s_macd'] * w_macd +
    fdf['s_rsl'] * w_rsl +
    fdf['s_volatility'] * w_volatility
)

# ---------------------------
# 最终排序与展示
# ---------------------------
fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
fdf.index = fdf.index + 1

st.success(f"评分完成：总候选 {len(fdf)} 支，显示 Top {min(TOP_DISPLAY, len(fdf))}。")
display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','proxy_money','amount','10d_return','20d_return','macd','k','d','j','rsl','volatility_10']
for c in display_cols:
    if c not in fdf.columns:
        fdf[c] = np.nan

st.dataframe(fdf[display_cols].head(TOP_DISPLAY), use_container_width=True)

# 下载（仅导出前200以防太大）
out_csv = fdf[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{}.csv".format(last_trade), mime="text/csv")

# ---------------------------
# 小结与提示（简洁）
# ---------------------------
st.markdown("### 小结与操作提示")
st.markdown("""
- 本版增加：市值区间（默认 20亿–300亿）与位置过滤（剔除过去10/20日涨幅过高的票）。  
- 提高了新鲜动能权重（10日回报）以优先横盘突破与趋势中段的“新鲜”票。  
- moneyflow 若不可用，使用 proxy_money 代替，不会中断评分。  
- 实战纪律：**9:40 前不买 → 观察 9:40-10:05 的量价结构 → 10:05 后择优介入**。  
- 若你想把“市值上限/下限、位置阈值”改成别的数值，直接在侧栏修改后再跑一次即可。  
""")

st.info("运行出现问题请把 Streamlit 的错误日志或首段报错文本发我（或截图），我会在你剩余的修改机会内继续修。")
