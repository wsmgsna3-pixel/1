# -*- coding: utf-8 -*-
"""
选股王 · 5000 积分旗舰版（增强：MACD / RSL / 风险过滤）
说明：
- 将本文件替换到 app.py，直接运行（streamlit run app.py）
- 在界面输入 Tushare Token（仅本次运行使用）
- 额外增加：RSL（相对强弱）、极端波动过滤、更多兜底
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="选股王 · 5000旗舰（增强）", layout="wide")
st.title("选股王 · 5000 积分旗舰（增强版：MACD,RSL,过滤）")

# ---------------------------
# 用户可调参数（必要时可改）
# ---------------------------
INITIAL_TOP_N = 1000   # 从涨幅榜取前 N
FINAL_POOL = 500       # 清洗后取前 M 进入评分
TOP_DISPLAY = 30       # 界面显示 Top K
MIN_PRICE = 10.0
MAX_PRICE = 200.0
MIN_TURNOVER = 3.0     # %
MIN_AMOUNT = 200_000_000.0  # 成交额下限（元）
MAX_TOTAL_MV = 1000 * 1e8  # 1000 亿元 -> 元

VOL_SPIKE_MULT = 1.5   # 放量阈值（vol_last > vol_ma5 * VOL_SPIKE_MULT）
VOLATILITY_MAX = 8.0   # 过去10日 pct_chg 标准差超过这个（%）视为极端波动，剔除
HIGH_PCT_THRESHOLD = 5.0  # 视为“大阳线”的 pct_chg 阈值

# ---------------------------
# Token 输入（界面）
# ---------------------------
TS_TOKEN = st.text_input("请输入你的 Tushare Token（仅本次运行使用）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 后才能运行。")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# helper: 安全调用
# ---------------------------
def safe_get(func, **kwargs):
    try:
        df = func(**kwargs)
        if df is None:
            return pd.DataFrame()
        if isinstance(df, pd.DataFrame) and df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

# 最近交易日寻找（回退）
def find_last_trade_day(max_days=15):
    today = datetime.now().date()
    for i in range(max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        df = safe_get(pro.daily, trade_date=ds)
        if not df.empty:
            return ds
    return None

last_trade = find_last_trade_day(15)
if not last_trade:
    st.error("无法找到最近交易日，检查网络或 Token 权限。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")

# ---------------------------
# 拉涨幅榜初筛
# ---------------------------
st.write("拉取当日 daily（涨幅榜）作为初筛...")
daily_all = safe_get(pro.daily, trade_date=last_trade)
if daily_all.empty:
    st.error("无法获取当日 daily 数据（Tushare 返回空）。请确认 Token 权限。")
    st.stop()
daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
st.write(f"当日记录数：{len(daily_all)}，将取涨幅前 {INITIAL_TOP_N} 作为初筛池。")
pool0 = daily_all.head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

# ---------------------------
# 尝试获取其他高级表
# ---------------------------
st.write("尝试加载 stock_basic / daily_basic / moneyflow（若权限允许）...")
stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,list_date,total_mv,circ_mv')
daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)
if mf_raw.empty:
    st.warning("无法获取 moneyflow（主力净流），相关因子将降级为0。")

# 预处理 moneyflow：挑选可能的净流列
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

# ---------------------------
# safe merge helper
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

# merge moneyflow with robust fallback
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
# 基本清洗（ST / 停牌 / 价格区间 / 一字板 / 换手 / 成交额 / 市值）
# ---------------------------
st.write("对初筛池进行清洗（ST/停牌/价格过高或过低/一字板/低换手等）...")
clean_list = []
pbar = st.progress(0)
for i, r in enumerate(pool_merged.itertuples()):
    ts = getattr(r, 'ts_code')
    # try volume detection with fallback
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

    # market cap filter (兜底)
    try:
        tv = total_mv
        if not pd.isna(tv):
            tv = float(tv)
            if tv > 1e6:
                tv_yuan = tv * 10000.0
            else:
                tv_yuan = tv
            if tv_yuan > MAX_TOTAL_MV:
                pbar.progress((i+1)/len(pool_merged)); continue
    except:
        pass

    # turnover
    if not pd.isna(turnover):
        try:
            if float(turnover) < MIN_TURNOVER: pbar.progress((i+1)/len(pool_merged)); continue
        except:
            pass

    # amount (convert if likely in 万元)
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
# 历史拉取（缓存）与指标计算
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
        macd = (diff - dea) * 2
        res['macd'] = macd.iloc[-1]; res['diff'] = diff.iloc[-1]; res['dea'] = dea.iloc[-1]
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

    # vol ratio and vol metrics
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        avg_prev5 = np.mean(vols[-6:-1])
        res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9)
        res['vol_last'] = vols[-1]
        res['vol_ma5'] = avg_prev5
    else:
        res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan

    # 10d return
    if len(close) >= 10:
        res['10d_return'] = close.iloc[-1] / close.iloc[-10] - 1
    else:
        res['10d_return'] = np.nan

    # prev3_sum (用于下跌途中过滤)
    if 'pct_chg' in df.columns and len(df) >= 4:
        try:
            pct = df['pct_chg'].astype(float)
            res['prev3_sum'] = pct.iloc[-4:-1].sum()
        except:
            res['prev3_sum'] = np.nan
    else:
        res['prev3_sum'] = np.nan

    # volatility (std of last 10 pct_chg in %)
    try:
        if 'pct_chg' in df.columns and len(df) >= 10:
            res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()
        else:
            res['volatility_10'] = np.nan
    except:
        res['volatility_10'] = np.nan

    return res

# ---------------------------
# 评分池逐票计算因子
# ---------------------------
st.write("对评分池逐只拉历史并计算指标（注意：此步骤会调用历史接口，已启用缓存以减少调用）...")
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

    rec = {
        'ts_code': ts_code, 'name': name, 'pct_chg': pct_chg,
        'amount': amount if not pd.isna(amount) else 0.0,
        'turnover_rate': turnover_rate if not pd.isna(turnover_rate) else np.nan,
        'net_mf': net_mf,
        'vol_ratio': vol_ratio if not pd.isna(vol_ratio) else np.nan,
        '10d_return': ten_return if not pd.isna(ten_return) else np.nan,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
        'macd': macd, 'k': k, 'd': d, 'j': j,
        'last_close': last_close, 'vol_last': vol_last, 'vol_ma5': vol_ma5,
        'prev3_sum': prev3_sum, 'volatility_10': volatility_10
    }

    records.append(rec)
    pbar2.progress((idx+1)/len(clean_df))

pbar2.progress(1.0)
fdf = pd.DataFrame(records)
if fdf.empty:
    st.error("评分计算失败或无数据，请检查 Token 权限与接口。")
    st.stop()

# ---------------------------
# 在评分前加入风险过滤模块（下跌途中大阳/巨量冲高/高位大阳/极端波动）
# ---------------------------
st.write("执行增强风险过滤（下跌途中大阳 / 巨量冲高 / 高位大阳 / 极端波动）...")
try:
    before_cnt = len(fdf)

    # A: 高位大阳线 -> last_close > ma20*1.10 且 pct_chg > HIGH_PCT_THRESHOLD
    if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):
        mask_high_big = (fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
        fdf = fdf[~mask_high_big]

    # B: 下跌途中反抽 -> prev3_sum < 0 且 pct_chg > HIGH_PCT_THRESHOLD
    if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
        mask_down_rebound = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
        fdf = fdf[~mask_down_rebound]

    # C: 巨量放量大阳 -> vol_last > vol_ma5 * VOL_SPIKE_MULT
    if all(c in fdf.columns for c in ['vol_last','vol_ma5']):
        mask_vol_spike = (fdf['vol_last'] > (fdf['vol_ma5'] * VOL_SPIKE_MULT))
        fdf = fdf[~mask_vol_spike]

    # D: 极端波动 -> volatility_10 (std of pct) > VOLATILITY_MAX
    if 'volatility_10' in fdf.columns:
        mask_volatility = fdf['volatility_10'] > VOLATILITY_MAX
        fdf = fdf[~mask_volatility]

    after_cnt = len(fdf)
    st.write(f"过滤完成：从 {before_cnt} 条候选过滤至 {after_cnt} 条（若过度严格请调整阈值）。")
except Exception as e:
    st.warning(f"风险过滤模块运行异常，已跳过过滤。错误信息：{e}")

# ---------------------------
# RSL（相对强弱）：用 fdf 的 10d_return 均值作为市场基准
# ---------------------------
if '10d_return' in fdf.columns:
    try:
        market_mean_10d = fdf['10d_return'].replace([np.inf,-np.inf], np.nan).dropna().mean()
        if np.isnan(market_mean_10d) or abs(market_mean_10d) < 1e-9:
            market_mean_10d = 1e-9
        fdf['rsl'] = fdf['10d_return'] / market_mean_10d
    except Exception:
        fdf['rsl'] = 1.0
else:
    fdf['rsl'] = 1.0

# ---------------------------
# 子指标归一化（稳健）
# ---------------------------
def norm_col(s, clip_low=None, clip_high=None):
    s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    if clip_low is not None:
        s = s.clip(lower=clip_low)
    if clip_high is not None:
        s = s.clip(upper=clip_high)
    mn = s.min(); mx = s.max()
    if mx - mn < 1e-9:
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)

fdf['s_pct'] = norm_col(fdf['pct_chg'])
fdf['s_volratio'] = norm_col(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))
fdf['s_turn'] = norm_col(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))
fdf['s_money'] = norm_col(fdf.get('net_mf', pd.Series([0]*len(fdf))))
fdf['s_amount'] = norm_col(fdf.get('amount', pd.Series([0]*len(fdf))))
fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))
fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
fdf['s_ind'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))  # 用 rsl 作为行业/相对强度代理
# 额外加入波动惩罚（波动越大越低分）
fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))

# ---------------------------
# 综合评分权重（偏向动能 + 资金 + 相对强弱）
# 你可以在后续把这些权重放 UI 控件里
# ---------------------------
w_pct = 0.18
w_volratio = 0.18
w_turn = 0.14
w_money = 0.16
w_10d = 0.12
w_macd = 0.06
w_rsl = 0.10
w_volatility = 0.06

fdf['综合评分'] = (
    fdf['s_pct'] * w_pct +
    fdf['s_volratio'] * w_volratio +
    fdf['s_turn'] * w_turn +
    fdf['s_money'] * w_money +
    fdf['s_10d'] * w_10d +
    fdf['s_macd'] * w_macd +
    fdf['s_ind'] * w_rsl +
    fdf['s_volatility'] * w_volatility
)

# ---------------------------
# 最后排序并展示 Top
# ---------------------------
fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
fdf.index = fdf.index + 1

st.success(f"评分完成：总候选 {len(fdf)} 支，显示 Top {min(TOP_DISPLAY, len(fdf))}。")
display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','amount','10d_return','macd','k','d','j','rsl','volatility_10']
for c in display_cols:
    if c not in fdf.columns:
        fdf[c] = np.nan

st.dataframe(fdf[display_cols].head(TOP_DISPLAY), use_container_width=True)

# CSV 下载（仅前若干列）
csv = fdf[display_cols].to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载全部评分结果 CSV", data=csv, file_name=f"score_result_{}.csv".format(last_trade), mime="text/csv")

# ---------------------------
# 小结与提示
# ---------------------------
st.markdown("### 小结与操作提示")
st.markdown("""
- 新增：MACD、RSL（相对强弱）、下跌途中大阳线过滤、巨量冲高过滤、极端波动过滤。  
- 风险提示：即使候选评分高，也必须等待早盘节奏（建议 9:40 后或 10:05 后确认再进）。  
- 若发现今日候选普遍翻绿，请不要强行追涨；若多数红且量价齐升，则可择机介入。  
- 若希望我把阈值（VOL_SPIKE_MULT / VOLATILITY_MAX / HIGH_PCT_THRESHOLD / 权重）放到侧栏 UI，告诉我我会按你要求做。  
""")

st.info("如运行时报错或数据缺失，把报错信息或截图发给我（尽量完整），我在后续两次机会内继续帮你调优。")
