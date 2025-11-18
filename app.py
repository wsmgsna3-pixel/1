# -*- coding: utf-8 -*-
"""
短线王 · v3.0（1-5天短线强化版，Top20 输出）
特性：
- 强化行业权重与形态/量能/波动过滤
- ATR 波动率过滤（排除极端波动或极低波动）
- 排除过去 10-20 天内翻倍（可配置）
- 自动降级（缺字段不会崩）
- 手动输入 Token（界面）
"""
import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="短线王·v3.0 强化版（Top20）", layout="wide")
st.title("短线王 · v3.0（1-5 天短线强化）")

# ---------------------------
# Token
# ---------------------------
TS_TOKEN = st.text_input("请输入 Tushare Token（仅本次会话有效）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 并回车激活后继续。")
    st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# utils
# ---------------------------
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except:
        return default

def norm_series(s):
    s = pd.Series(s).astype(float)
    if s.isnull().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn = s.min()
    mx = s.max()
    if abs(mx - mn) < 1e-9:
        return pd.Series(np.ones(len(s)) * 0.5, index=s.index)
    return (s - mn) / (mx - mn)

def check_and_fill_data(df, required_cols):
    missing = []
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan
            missing.append(c)
    return df, missing

def get_last_trade_day(pro_obj, max_days=14):
    today = datetime.now()
    for i in range(0, max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        try:
            dd = pro_obj.daily(trade_date=ds)
            if dd is not None and len(dd) > 0:
                return ds
        except Exception:
            continue
    return None

# technical helpers
def calculate_macd(df, close_col='close', short=12, long=26, signal=9):
    close = df[close_col].astype(float)
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    dif = ema_short - ema_long
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    df = df.copy()
    df['DIF'] = dif
    df['DEA'] = dea
    df['MACD_HIST'] = macd_hist
    return df

def calculate_rsi(df, close_col='close', period=14):
    close = df[close_col].astype(float)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df = df.copy()
    df[f'RSI_{period}'] = rsi
    return df

def calculate_atr(df, period=14):
    df = df.copy()
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['prev_close'] = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['prev_close']).abs()
    tr3 = (df['low'] - df['prev_close']).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    df['ATR'] = atr
    return df

# ---------------------------
# Sidebar params (hard defaults tuned for 1-5 day)
# ---------------------------
st.sidebar.header("筛选 & 风格参数（v3.0）")
INITIAL_TOP_N = int(st.sidebar.number_input("初筛：涨幅榜取前 N", min_value=200, max_value=5000, value=1000, step=100))
FINAL_POOL = int(st.sidebar.number_input("进入评分池数量", min_value=50, max_value=2000, value=500, step=50))
TOP_K = int(st.sidebar.number_input("输出 Top K", min_value=5, max_value=50, value=20, step=1))

MIN_PRICE = float(st.sidebar.number_input("最低股价（元）", min_value=0.1, max_value=1000.0, value=10.0, step=0.1))
MAX_PRICE = float(st.sidebar.number_input("最高股价（元）", min_value=1.0, max_value=2000.0, value=200.0, step=1.0))
MIN_AMOUNT = float(st.sidebar.number_input("最低成交额（元）", min_value=0.0, max_value=1e11, value=200_000_000.0, step=10_000_000.0))
MIN_MV = float(st.sidebar.number_input("最小市值（元）", min_value=1e7, max_value=1e12, value=20_0000_0000.0, step=1e7))  # default 20亿
MAX_MV = float(st.sidebar.number_input("最大市值（元）", min_value=1e8, max_value=1e13, value=500_0000_00000.0, step=1e8))  # default 500亿

# exclude quick double in 10-20 days
EXCLUDE_DOUBLE_10_20 = st.sidebar.checkbox("排除过去10-20天内翻倍（建议勾选）", value=True)

st.sidebar.markdown("---")
st.sidebar.header("因子权重（默认已适配短线）")
# default: strengthen industry and tech
w_pct = st.sidebar.slider("涨幅/短期强度", 0.0, 1.0, 0.18)
w_volratio = st.sidebar.slider("量比（放量）", 0.0, 1.0, 0.16)
w_turn = st.sidebar.slider("活跃度/成交额", 0.0, 1.0, 0.12)
w_money = st.sidebar.slider("主力资金", 0.0, 1.0, 0.10)
w_ind = st.sidebar.slider("行业强度（加权）", 0.0, 1.0, 0.26)
w_tech = st.sidebar.slider("技术形态（MACD/RSI/形态）", 0.0, 1.0, 0.18)

total_w = w_pct + w_volratio + w_turn + w_money + w_ind + w_tech
if total_w == 0:
    st.sidebar.error("权重总和不可为0")
    st.stop()
w_pct /= total_w; w_volratio /= total_w; w_turn /= total_w; w_money /= total_w; w_ind /= total_w; w_tech /= total_w

st.sidebar.markdown("---")
st.sidebar.markdown("若接口缺失（moneyflow/daily_basic 等），脚本会自动降级并在界面提示。")

# ---------------------------
# get last trade day
# ---------------------------
with st.spinner("获取最近交易日..."):
    last_trade = get_last_trade_day(pro, max_days=14)
if not last_trade:
    st.error("无法获取最近交易日，请检查 Token 或网络")
    st.stop()
st.info(f"参考交易日：{last_trade}")

# ---------------------------
# load market daily
# ---------------------------
@st.cache_data(ttl=60)
def load_market_daily(trade_date):
    try:
        return pro.daily(trade_date=trade_date)
    except Exception:
        return pd.DataFrame()

market_df = load_market_daily(last_trade)
market_df, _ = check_and_fill_data(market_df, ['ts_code','pct_chg','vol','amount','open','high','low','pre_close','close'])
if market_df.empty:
    st.error("无法获取当日行情或数据为空")
    st.stop()
st.write(f"当日记录数：{len(market_df)}（后续从涨幅榜前 {INITIAL_TOP_N} 初筛）")

# ---------------------------
# stock_basic safe
# ---------------------------
def try_get_stock_basic():
    try:
        df = pro.stock_basic(list_status='L', fields='ts_code,symbol,name,market,industry,list_date')
        return df
    except Exception:
        try:
            return pro.stock_basic(list_status='L')
        except Exception:
            return pd.DataFrame()

stock_basic_df = try_get_stock_basic()
stock_basic_df, missing_sb = check_and_fill_data(stock_basic_df, ['ts_code','name','industry'])
if len(missing_sb) > 0:
    st.warning(f"stock_basic 缺失列：{missing_sb}，行业/名称可能受限，相关因子会降级。")

# ---------------------------
# try daily_basic & moneyflow (may be None)
# ---------------------------
def try_get_daily_basic(trade_date):
    try:
        return pro.daily_basic(trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv,pe,pb')
    except Exception:
        return None

def try_get_moneyflow(trade_date):
    try:
        mf = pro.moneyflow(trade_date=trade_date)
        for col in ['net_mf','net_mf_amount','net_amount']:
            if col in mf.columns:
                mf = mf[['ts_code', col]].drop_duplicates(subset=['ts_code']).set_index('ts_code')
                mf.columns = ['net_mf']
                return mf
        return None
    except Exception:
        return None

daily_basic_df = try_get_daily_basic(last_trade)
if daily_basic_df is None:
    st.warning("daily_basic 不可用，换手/市值/PE等因子将降级或用近似代替。")

moneyflow_df = try_get_moneyflow(last_trade)
if moneyflow_df is None:
    st.warning("moneyflow 不可用，主力资金因子将禁用。")

# ---------------------------
# initial top N
# ---------------------------
pool = market_df.sort_values('pct_chg', ascending=False).head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

# safe merge stock_basic
need_cols = ['ts_code','name','industry']
actual_cols = stock_basic_df.columns.tolist() if not stock_basic_df.empty else []
use_cols = [c for c in need_cols if c in actual_cols]
if 'ts_code' in use_cols and len(use_cols)>0:
    stock_basic_safe = stock_basic_df[use_cols]
    pool = pool.merge(stock_basic_safe, on='ts_code', how='left')
else:
    pool['name'] = pool['ts_code']
    pool['industry'] = ""

# join daily_basic
if daily_basic_df is not None:
    try:
        db = daily_basic_df.drop_duplicates(subset=['ts_code']).set_index('ts_code')
        pool = pool.set_index('ts_code').join(db[['turnover_rate','amount','total_mv','circ_mv']].rename(columns={'turnover_rate':'turnover_rate_db','amount':'amount_db'}), how='left').reset_index()
    except Exception:
        pool['turnover_rate_db'] = np.nan
        pool['amount_db'] = np.nan
else:
    pool['turnover_rate_db'] = np.nan
    pool['amount_db'] = np.nan

# join moneyflow
if moneyflow_df is not None:
    try:
        pool = pool.set_index('ts_code').join(moneyflow_df[['net_mf']], how='left').reset_index()
    except Exception:
        pool['net_mf'] = 0.0
else:
    pool['net_mf'] = 0.0

pool, _ = check_and_fill_data(pool, ['ts_code','pct_chg','vol','amount','open','high','low','pre_close','close','name','industry','turnover_rate_db','net_mf'])

# ---------------------------
# cleaning filters (hard rules)
# ---------------------------
cleaned = []
for idx, r in pool.iterrows():
    try:
        # volume/amount basic
        vol = safe_float(r.get('vol', 0))
        amt = safe_float(r.get('amount', 0))
        if vol == 0 or (amt == 0 or np.isnan(amt)):
            continue
        # price bounds
        price = safe_float(r.get('close', r.get('open', np.nan)))
        if np.isnan(price):
            continue
        if price < MIN_PRICE or price > MAX_PRICE:
            continue
        # name filters
        name = r.get('name', '') if 'name' in r else ''
        if isinstance(name, str) and name != "":
            up = name.upper()
            if 'ST' in up or '退' in up:
                continue
        # market cap filter: prefer daily_basic total_mv; try to normalize units
        tv = r.get('total_mv', np.nan)
        try:
            tvf = float(tv)
            if tvf > 1e6:  # likely 万元
                tv_yuan = tvf * 10000
            else:
                tv_yuan = tvf
            # apply range if parseable
            if not np.isnan(tv_yuan):
                if tv_yuan < MIN_MV or tv_yuan > MAX_MV:
                    continue
        except:
            pass
        # amount normalization (some APIs return 万元)
        if amt > 0 and amt < 1e5:
            amt *= 10000
        if amt < MIN_AMOUNT:
            continue
        # exclude one-word board (open==high==low==pre_close)
        try:
            if (safe_float(r.get('open',0)) == safe_float(r.get('high',0)) == safe_float(r.get('low',0)) == safe_float(r.get('pre_close',0))):
                continue
        except:
            pass
        cleaned.append(r)
    except Exception:
        continue

cleaned_df = pd.DataFrame(cleaned).reset_index(drop=True)
st.write(f"清洗后候选：{len(cleaned_df)}（将从中取涨幅前 {FINAL_POOL} 进入评分）")
if cleaned_df.empty:
    st.error("清洗后无候选，请放宽条件或检查权限")
    st.stop()

# ---------------------------
# reduce to final pool by pct_chg
# ---------------------------
cleaned_df = cleaned_df.sort_values('pct_chg', ascending=False).head(int(FINAL_POOL)).reset_index(drop=True)
st.write(f"评分池大小：{len(cleaned_df)}")

# ---------------------------
# history fetcher
# ---------------------------
@st.cache_data(ttl=600)
def get_hist(ts_code, end_date, days=60):
    try:
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        return df
    except Exception:
        return None

# ---------------------------
# scoring loop (with v3 filters)
# ---------------------------
records = []
pbar = st.progress(0)
N = len(cleaned_df)
for i, row in enumerate(cleaned_df.itertuples()):
    try:
        ts_code = getattr(row, 'ts_code')
        pct_chg = safe_float(getattr(row, 'pct_chg', 0))
        amount_val = safe_float(getattr(row, 'amount', 0))
        if amount_val > 0 and amount_val < 1e5:
            amount_val *= 10000

        hist = get_hist(ts_code, last_trade, days=60)
        if hist is None or len(hist) < 12:
            # fallback neutral scores
            vol_ratio = np.nan
            ma5 = np.nan
            ten_return = np.nan
            macd_score = 0.5
            rsi_score = 0.5
            atr = np.nan
            recent_low_rising = False
            exclude_double = False
        else:
            hist = hist.reset_index(drop=True)
            # calculate ATR
            hist_atr = calculate_atr(hist, period=14)
            atr = hist_atr['ATR'].iloc[-1] if 'ATR' in hist_atr.columns else np.nan

            # vol ratio: today / avg prev5
            tail = hist.tail(20).reset_index(drop=True)
            vols = tail['vol'].astype(float).tolist()
            if len(vols) >= 6:
                avg_vol5 = np.mean(vols[:-1][-5:])
            else:
                avg_vol5 = np.mean(vols[:-1]) if len(vols[:-1])>0 else np.nan
            vol_today = float(vols[-1]) if len(vols)>0 else np.nan
            vol_ratio = vol_today / (avg_vol5 + 1e-9) if not np.isnan(avg_vol5) and avg_vol5>0 else np.nan

            # ma5, ma10
            tail_close = tail['close'].astype(float)
            ma5 = tail_close.rolling(window=5).mean().iloc[-1] if len(tail_close)>=5 else np.nan
            ma10 = tail_close.rolling(window=10).mean().iloc[-1] if len(tail_close)>=10 else np.nan

            # 10-day return & 20-day return
            if len(tail_close) >= 10:
                ten_return = float(tail_close.iloc[-1]) / float(tail_close.iloc[0]) - 1.0
            else:
                ten_return = np.nan
            if len(tail_close) >= 20:
                twenty_return = float(tail_close.iloc[-1]) / float(tail_close.iloc[0]) - 1.0
            else:
                twenty_return = np.nan

            # exclude if 10-20日翻倍
            exclude_double = False
            if EXCLUDE_DOUBLE_10_20:
                if (not np.isnan(ten_return) and ten_return >= 1.0) or (not np.isnan(twenty_return) and twenty_return >= 1.0):
                    exclude_double = True

            # shape: recent lows rising? check last 5 lows vs prev 5 lows
            lows = tail['low'].astype(float).tolist()
            recent_low_rising = False
            if len(lows) >= 10:
                prev5 = lows[:-5]
                last5 = lows[-5:]
                if np.nanmean(last5) > np.nanmean(prev5):
                    recent_low_rising = True

            # MACD & RSI
            macd_tail = calculate_macd(tail, close_col='close')
            dif = macd_tail['DIF'].iloc[-1] if 'DIF' in macd_tail.columns else 0.0
            dea = macd_tail['DEA'].iloc[-1] if 'DEA' in macd_tail.columns else 0.0
            macd_hist_last = macd_tail['MACD_HIST'].iloc[-1] if 'MACD_HIST' in macd_tail.columns else 0.0
            if dif > dea:
                macd_score = 0.75 if macd_hist_last > 0 else 0.6
            else:
                macd_score = 0.25 if macd_hist_last < 0 else 0.4

            rsi_tail = calculate_rsi(tail, close_col='close', period=6)
            rsi6 = rsi_tail['RSI_6'].iloc[-1] if 'RSI_6' in rsi_tail.columns else 50.0
            if rsi6 < 30:
                rsi_score = 0.2
            elif rsi6 < 45:
                rsi_score = 0.55
            elif rsi6 < 70:
                rsi_score = 0.9
            else:
                rsi_score = 0.45

        # moneyflow
        net_mf = 0.0
        try:
            if moneyflow_df is not None and ts_code in moneyflow_df.index:
                net_mf = float(moneyflow_df.loc[ts_code,'net_mf'])
        except Exception:
            net_mf = 0.0

        # final hard exclusion rules
        # 1) exclude recent doubling if selected
        if EXCLUDE_DOUBLE_10_20 and exclude_double:
            continue
        # 2) ATR relative filter: require ATR/price between thresholds
        price_now = safe_float(getattr(row,'close', np.nan))
        atr_ratio = np.nan
        if not np.isnan(atr) and price_now>0:
            atr_ratio = atr / price_now
            if atr_ratio > 0.08:  # 太波动（>8% 日波动）排除
                continue
            if atr_ratio < 0.003:  # 过低波动（<0.3%）排除
                continue
        # 3) recent low rising required (trend quality)
        if not recent_low_rising:
            # allow a small fraction through but penalize later in score
            trend_ok = False
        else:
            trend_ok = True

        # 4) price range already checked earlier

        # compose record
        records.append({
            'ts_code': ts_code,
            'name': getattr(row,'name', ts_code),
            'pct_chg': safe_float(getattr(row,'pct_chg',0)),
            'amount': amount_val,
            'vol_ratio': vol_ratio if not pd.isna(vol_ratio) else 1.0,
            'turnover_rate': safe_float(getattr(row,'turnover_rate_db', np.nan)),
            'net_mf': net_mf,
            'ten_return': ten_return if not pd.isna(ten_return) else 0.0,
            'ma5': ma5,
            'ma10': ma10 if 'ma10' in locals() else np.nan,
            'macd_score': macd_score,
            'rsi_score': rsi_score,
            'atr_ratio': atr_ratio,
            'trend_ok': trend_ok,
            'price': price_now,
            'industry': getattr(row,'industry','')
        })
    except Exception:
        continue
    pbar.progress((i+1)/N if N>0 else 1.0)

pbar.progress(1.0)
score_df = pd.DataFrame(records)
if score_df.empty:
    st.error("评分池为空，请检查历史数据或参数设置")
    st.stop()

# ---------------------------
# industry strength
# ---------------------------
if 'industry' in score_df.columns and score_df['industry'].notnull().any():
    ind_mean = score_df.groupby('industry')['pct_chg'].transform('mean')
    score_df['industry_score'] = (ind_mean - ind_mean.min()) / (ind_mean.max() - ind_mean.min() + 1e-9)
    score_df['industry_score'] = score_df['industry_score'].fillna(0.0)
else:
    score_df['industry_score'] = 0.0
    st.warning("行业数据缺失或无效，行业因子被禁用")

# ---------------------------
# normalize subfactors & penalties
# ---------------------------
score_df['pct_rank'] = norm_series(score_df['pct_chg'])
score_df['volrank'] = norm_series(score_df['vol_ratio'].replace([np.inf,-np.inf], np.nan).fillna(0))
score_df['turn_rank'] = norm_series(score_df['turnover_rate'].fillna(0))
score_df['money_rank'] = norm_series(score_df['net_mf'].fillna(0))
score_df['tech_rank'] = norm_series(score_df['macd_score'] * 0.6 + score_df['rsi_score'] * 0.4)
score_df['industry_rank'] = norm_series(score_df['industry_score'].fillna(0))

# penalize non-trend_ok
score_df['trend_bonus'] = score_df['trend_ok'].apply(lambda x: 1.0 if x else 0.7)

# disable money factor if missing
if moneyflow_df is None:
    w_money = 0.0

# disable industry if missing
if 'industry' not in stock_basic_df.columns or stock_basic_df['industry'].isnull().all():
    w_ind = 0.0

# re-normalize weights
total_w = w_pct + w_volratio + w_turn + w_money + w_ind + w_tech
if total_w == 0:
    st.error("所有因子被禁用，无法评分")
    st.stop()
w_pct /= total_w; w_volratio /= total_w; w_turn /= total_w; w_money /= total_w; w_ind /= total_w; w_tech /= total_w

# compose final score with trend bonus and atr penalty
score_df['综合评分'] = (
    score_df['pct_rank'] * w_pct +
    score_df['volrank'] * w_volratio +
    score_df['turn_rank'] * w_turn +
    score_df['money_rank'] * w_money +
    score_df['industry_rank'] * w_ind +
    score_df['tech_rank'] * w_tech
) * score_df['trend_bonus']

# ATR-based slight penalty: if atr_ratio very low or moderately high, apply small correction (already filtered extremes)
score_df['综合评分'] = score_df['综合评分'] * (1 - 0.1 * (1 - np.tanh(10 * (score_df['atr_ratio'].fillna(0.01)-0.01))))  # smooth small adjustment

score_df = score_df.sort_values('综合评分', ascending=False).reset_index(drop=True)
score_df.index += 1

# ---------------------------
# Output Top K
# ---------------------------
display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','amount','price','ma5','ma10','ten_return','atr_ratio','industry']
for c in display_cols:
    if c not in score_df.columns:
        score_df[c] = np.nan

st.success(f"评分完成，候选 {len(score_df)} 支，展示 Top {min(TOP_K, len(score_df))}（按综合评分降序）")
st.dataframe(score_df[display_cols].head(int(TOP_K)).reset_index(drop=False), use_container_width=True)

# csv download
csv = score_df[display_cols].to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载全部评分结果 CSV", data=csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

# brief reasons for top picks
st.markdown("### Top 候选说明（示例）")
top_show = score_df.head(int(TOP_K))
for idx, r in top_show.reset_index().iterrows():
    reasons = []
    if r['pct_chg'] > 5:
        reasons.append("当日强势")
    if r['vol_ratio'] and r['vol_ratio'] > 1.5:
        reasons.append("放量")
    if r['ma5'] and r['price'] > r['ma5']:
        reasons.append("站上短均")
    if r['atr_ratio'] and r['atr_ratio'] < 0.03:
        reasons.append("波动适中")
    if r['net_mf'] and r['net_mf'] > 0:
        reasons.append("主力净流入")
    st.write(f"{int(r['index'])}. {r['name']} ({r['ts_code']}) — Score {r['综合评分']:.4f} — {'；'.join(reasons) if reasons else '常规强势'}")

st.markdown("### 小结")
st.markdown("""
- v3.0 针对 1-5 天短线强化了行业权重、形态与量能过滤、ATR 波动率过滤与翻倍排除等规则。  
- 若某些接口缺失（例：moneyflow/daily_basic），脚本会自动降级并提示；核心计算仍基于 daily/历史日线。  
- 建议每日早盘 9:30-10:15 运行一次以获得最佳候选池。  
""")
