# -*- coding: utf-8 -*-
import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="短线王·5000积分旗舰版（Top20）", layout="wide")
st.title("短线王 · 5000 积分旗舰版（Top20 输出）")

# ---------------------------
# User Token input (manual)
# ---------------------------
TS_TOKEN = st.text_input("在此输入你的 Tushare Token（仅本次会话使用）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 后运行。")
    st.stop()

# init tushare
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# Helper utilities
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
    """
    检查 DataFrame 是否包含 required_cols，
    若缺失则用 NaN 填列并返回缺失列表
    """
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

# MACD and RSI implementations
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

# ---------------------------
# Sidebar parameters & weights
# ---------------------------
st.sidebar.header("筛选参数（短线 1-5 天 风格）")
INITIAL_TOP_N = int(st.sidebar.number_input("初筛：涨幅榜取前 N", min_value=200, max_value=5000, value=1000, step=100))
FINAL_POOL = int(st.sidebar.number_input("进入评分池数量（最终取前多少入评分）", min_value=50, max_value=2000, value=500, step=50))
TOP_K = int(st.sidebar.number_input("界面展示 Top K（输出）", min_value=5, max_value=100, value=20, step=1))

MIN_PRICE = float(st.sidebar.number_input("最低价格（元）", min_value=0.1, max_value=1000.0, value=10.0, step=0.1))
MAX_PRICE = float(st.sidebar.number_input("最高价格（元）", min_value=1.0, max_value=2000.0, value=200.0, step=1.0))
MIN_AMOUNT = float(st.sidebar.number_input("成交额最低阈值（元）", min_value=0.0, max_value=1e11, value=200_000_000.0, step=10_000_000.0))
MIN_MV = float(st.sidebar.number_input("允许的最小总市值（元）", min_value=1e7, max_value=1e12, value=2e9, step=1e7))
MAX_MV = float(st.sidebar.number_input("允许的最大总市值（元）", min_value=1e8, max_value=1e13, value=5e10, step=1e8))

st.sidebar.markdown("---")
st.sidebar.header("因子权重（可调，界面会归一化）")
w_pct = st.sidebar.slider("涨幅权重（当日/短期强度）", 0.0, 1.0, 0.25)
w_volratio = st.sidebar.slider("量比权重", 0.0, 1.0, 0.18)
w_turn = st.sidebar.slider("换手/活跃度权重", 0.0, 1.0, 0.15)
w_money = st.sidebar.slider("主力资金权重", 0.0, 1.0, 0.12)
w_ind = st.sidebar.slider("行业强度权重", 0.0, 1.0, 0.15)
w_tech = st.sidebar.slider("技术形态（MACD/RSI）权重", 0.0, 1.0, 0.15)

total_w = w_pct + w_volratio + w_turn + w_money + w_ind + w_tech
if total_w == 0:
    st.sidebar.error("权重总和不可为0，请调整。")
    st.stop()
w_pct /= total_w; w_volratio /= total_w; w_turn /= total_w; w_money /= total_w; w_ind /= total_w; w_tech /= total_w

st.sidebar.markdown("---")
st.sidebar.markdown("注意：脚本会对每个外部接口做容错并提示降级（例如 moneyflow/daily_basic 缺失）。")

# ---------------------------
# Find last trade day
# ---------------------------
with st.spinner("正在获取最近交易日..."):
    last_trade = get_last_trade_day(pro, max_days=14)
if not last_trade:
    st.error("无法获取最近交易日，请检查 Token/网络。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")

# ---------------------------
# Load market daily (all)
# ---------------------------
@st.cache_data(ttl=60)
def load_market_daily(trade_date):
    try:
        df = pro.daily(trade_date=trade_date)
        return df
    except Exception:
        return pd.DataFrame()

market_df = load_market_daily(last_trade)
market_df, missing = check_and_fill_data(market_df, ['ts_code','pct_chg','vol','amount','open','high','low','pre_close','close'])
if market_df.empty:
    st.error("获取当日日线失败或为空，请检查 Token 权限。")
    st.stop()
st.write(f"当日记录数：{len(market_df)}（后续将从涨幅榜前 {INITIAL_TOP_N} 进行初筛）")

# ---------------------------
# Try stock_basic (safe)
# ---------------------------
def try_get_stock_basic():
    try:
        # 请求显式字段；如果接口无权则返回可用的列
        df = pro.stock_basic(list_status='L', fields='ts_code,symbol,name,market,industry,list_date')
        return df
    except Exception:
        try:
            df = pro.stock_basic(list_status='L')  # fallback
            return df
        except Exception:
            return pd.DataFrame()

stock_basic_df = try_get_stock_basic()
if stock_basic_df is None:
    stock_basic_df = pd.DataFrame()
stock_basic_df, missing_sb = check_and_fill_data(stock_basic_df, ['ts_code','name','industry'])

if len(missing_sb) > 0:
    st.warning(f"stock_basic 缺失字段：{missing_sb}。脚本将自动降级相关因子。")

# ---------------------------
# Try daily_basic
# ---------------------------
def try_get_daily_basic(trade_date):
    try:
        db = pro.daily_basic(trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv,pe,pb')
        return db
    except Exception:
        return None

daily_basic_df = try_get_daily_basic(last_trade)
if daily_basic_df is None:
    st.warning("无法获取 daily_basic（换手率/成交额/市值），相关因子将降级或用近似值代替。")

# ---------------------------
# Try moneyflow
# ---------------------------
def try_get_moneyflow(trade_date):
    try:
        mf = pro.moneyflow(trade_date=trade_date)
        # try common net field names
        for col in ['net_mf','net_mf_amount','net_amount']:
            if col in mf.columns:
                mf = mf[['ts_code', col]].drop_duplicates(subset=['ts_code']).set_index('ts_code')
                mf.columns = ['net_mf']
                return mf
        # fallback: try to compute approximate net from available buy/sell big
        # but if not present return None
        return None
    except Exception:
        return None

moneyflow_df = try_get_moneyflow(last_trade)
if moneyflow_df is None:
    st.warning("无法获取 moneyflow（主力净流），评分中此项将降级为0。")

# ---------------------------
# Initial top N by pct_chg
# ---------------------------
pool = market_df.sort_values('pct_chg', ascending=False).head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

# safe merge stock_basic
need_cols = ['ts_code','name','industry']
actual_cols = stock_basic_df.columns.tolist() if not stock_basic_df.empty else []
use_cols = [c for c in need_cols if c in actual_cols]
if 'ts_code' in use_cols and len(use_cols) > 0:
    stock_basic_safe = stock_basic_df[use_cols]
    pool = pool.merge(stock_basic_safe, on='ts_code', how='left')
else:
    pool['name'] = pool['ts_code']
    pool['industry'] = ""

# join daily_basic if available
if daily_basic_df is not None:
    try:
        daily_basic_df = daily_basic_df.drop_duplicates(subset=['ts_code']).set_index('ts_code')
        pool = pool.set_index('ts_code').join(daily_basic_df[['turnover_rate','amount','total_mv','circ_mv']].rename(columns={'turnover_rate':'turnover_rate_db'}), how='left').reset_index()
    except Exception:
        pool['turnover_rate_db'] = np.nan
        pool['amount_db'] = np.nan
else:
    pool['turnover_rate_db'] = np.nan

# join moneyflow if available
if moneyflow_df is not None:
    try:
        pool = pool.set_index('ts_code').join(moneyflow_df[['net_mf']], how='left').reset_index()
    except Exception:
        pool['net_mf'] = 0.0
else:
    pool['net_mf'] = 0.0

# ensure columns exist
pool, _ = check_and_fill_data(pool, ['ts_code','pct_chg','vol','amount','open','high','low','pre_close','close','name','industry','turnover_rate_db','net_mf'])

# ---------------------------
# Clean candidate pool (filtering)
# ---------------------------
cleaned = []
for idx, r in pool.iterrows():
    try:
        vol = safe_float(r.get('vol', 0))
        amount = safe_float(r.get('amount', r.get('amount', 0)))
        if vol == 0 or (amount == 0 or np.isnan(amount)):
            continue

        price = safe_float(r.get('close', r.get('open', np.nan)))
        if np.isnan(price):
            continue
        if price < MIN_PRICE or price > MAX_PRICE:
            continue

        name = r.get('name', '') if 'name' in r else ''
        if isinstance(name, str) and name != "":
            up = name.upper()
            if 'ST' in up or '退' in up:
                continue

        # total_mv handling: prefer daily_basic total_mv, fallback to None
        total_mv = r.get('total_mv', np.nan)
        if pd.isna(total_mv) and 'total_mv' in r:
            total_mv = r.get('total_mv')
        # normalize units if obviously in 万元
        try:
            tv = float(total_mv)
            # guess unit
            if tv > 1e6:
                tv_yuan = tv * 10000
            else:
                tv_yuan = tv
            if not np.isnan(tv_yuan):
                if tv_yuan < MIN_MV or tv_yuan > MAX_MV:
                    continue
        except:
            # if cannot parse, skip mv filter (don't filter out)
            pass

        # turnover: prefer daily_basic 'turnover_rate_db' else leave
        tr = safe_float(r.get('turnover_rate_db', np.nan))
        if not pd.isna(tr):
            if tr < 0.1:  # extremely low turnover
                continue

        # amount normalization
        amt = safe_float(r.get('amount', 0))
        if amt > 0 and amt < 1e5:
            amt *= 10000
        if amt < MIN_AMOUNT:
            continue

        # exclude one-word boards
        try:
            if (safe_float(r.get('open',0)) == safe_float(r.get('high',0)) == safe_float(r.get('low',0)) == safe_float(r.get('pre_close',0))):
                continue
        except:
            pass

        cleaned.append(r)
    except Exception:
        continue

cleaned_df = pd.DataFrame(cleaned).reset_index(drop=True)
st.write(f"清洗后候选数量：{len(cleaned_df)} （将从中取涨幅前 {FINAL_POOL} 进入评分阶段）")
if len(cleaned_df) == 0:
    st.error("清洗后无候选，请放宽条件或检查 Token 权限。")
    st.stop()

# ---------------------------
# reduce to FINAL_POOL by pct_chg
# ---------------------------
cleaned_df = cleaned_df.sort_values('pct_chg', ascending=False).head(int(FINAL_POOL)).reset_index(drop=True)
st.write(f"用于评分的池子大小：{len(cleaned_df)}")

# ---------------------------
# helper: get_hist with caching
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
# Scoring loop (compute factors)
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

        # get history
        hist = get_hist(ts_code, last_trade, days=60)
        if hist is None or len(hist) < 10:
            # fallback: use available daily snapshot only
            vol_ratio = np.nan
            ma5 = np.nan
            ten_return = np.nan
            macd_score = 0.5
            rsi_score = 0.5
            vol_mean = np.nan
        else:
            # compute vol ratio: today vol / avg(prev 5)
            hist_tail = hist.tail(20).reset_index(drop=True)
            vols = hist_tail['vol'].astype(float).tolist()
            if len(vols) >= 6:
                avg_vol_5 = np.mean(vols[:-1][-5:])
            else:
                avg_vol_5 = np.mean(vols[:-1]) if len(vols[:-1])>0 else np.nan
            vol_today = float(vols[-1])
            vol_ratio = vol_today / (avg_vol_5 + 1e-9) if not np.isnan(avg_vol_5) and avg_vol_5>0 else np.nan

            ma5 = hist_tail['close'].astype(float).rolling(window=5).mean().iloc[-1] if len(hist_tail)>=5 else np.nan
            ten_return = (hist_tail['close'].iloc[-1] / hist_tail['close'].iloc[0] - 1) if len(hist_tail)>=2 else np.nan

            # MACD & RSI
            hist_macd = calculate_macd(hist_tail, close_col='close')
            hist_macd = hist_macd.reset_index(drop=True)
            macd_hist_last = hist_macd['MACD_HIST'].iloc[-1] if 'MACD_HIST' in hist_macd.columns else 0.0
            dif = hist_macd['DIF'].iloc[-1] if 'DIF' in hist_macd.columns else 0.0
            dea = hist_macd['DEA'].iloc[-1] if 'DEA' in hist_macd.columns else 0.0
            macd_score = 0.5
            # simple scoring logic: DIF>DEA positive bias
            if dif > dea:
                macd_score = 0.7 if macd_hist_last > 0 else 0.6
            else:
                macd_score = 0.3 if macd_hist_last < 0 else 0.4

            hist_rsi = calculate_rsi(hist_tail, close_col='close', period=6)
            rsi6 = hist_rsi['RSI_6'].iloc[-1] if 'RSI_6' in hist_rsi.columns else 50.0
            # rsi score: 40-70 is fine
            if rsi6 < 30:
                rsi_score = 0.2
            elif rsi6 < 45:
                rsi_score = 0.6
            elif rsi6 < 70:
                rsi_score = 0.9
            else:
                rsi_score = 0.4

            vol_mean = np.mean(hist_tail['vol'].astype(float)) if len(hist_tail)>0 else np.nan

        # moneyflow
        net_mf = 0.0
        try:
            if moneyflow_df is not None and ts_code in moneyflow_df.index:
                net_mf = float(moneyflow_df.loc[ts_code,'net_mf'])
        except Exception:
            net_mf = 0.0

        # industry -- compute later
        industry = getattr(row, 'industry', '') if 'industry' in cleaned_df.columns else ''

        # compose record
        records.append({
            'ts_code': ts_code,
            'name': getattr(row, 'name', ts_code),
            'pct_chg': pct_chg,
            'amount': amount_val,
            'vol_ratio': vol_ratio if not pd.isna(vol_ratio) else 1.0,
            'turnover_rate': safe_float(getattr(row, 'turnover_rate_db', np.nan)),
            'net_mf': net_mf,
            'ten_return': ten_return if not pd.isna(ten_return) else 0.0,
            'ma5': ma5,
            'macd_score': macd_score,
            'rsi_score': rsi_score,
            'price': safe_float(getattr(row,'close',np.nan)),
            'industry': industry
        })
    except Exception:
        continue
    pbar.progress((i+1)/N if N>0 else 1.0)

pbar.progress(1.0)
score_df = pd.DataFrame(records)
if score_df.empty:
    st.error("评分数据为空，请检查历史数据拉取限制或 Token 权限。")
    st.stop()

# ---------------------------
# Industry strength
# ---------------------------
if 'industry' in score_df.columns and score_df['industry'].notnull().any():
    ind_mean = score_df.groupby('industry')['pct_chg'].transform('mean')
    score_df['industry_score'] = (ind_mean - ind_mean.min()) / (ind_mean.max() - ind_mean.min() + 1e-9)
    score_df['industry_score'] = score_df['industry_score'].fillna(0.0)
else:
    score_df['industry_score'] = 0.0
    st.warning("行业字段不可用或均为空，行业因子将被禁用。")

# ---------------------------
# Normalize subfactors
# ---------------------------
score_df['pct_rank'] = norm_series(score_df['pct_chg'])
score_df['volratio_rank'] = norm_series(score_df['vol_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0))
score_df['turn_rank'] = norm_series(score_df['turnover_rate'].fillna(0))
score_df['money_rank'] = norm_series(score_df['net_mf'].fillna(0))
score_df['tech_rank'] = norm_series(score_df['macd_score'] * 0.6 + score_df['rsi_score'] * 0.4)
score_df['industry_rank'] = norm_series(score_df['industry_score'].fillna(0))

# auto-disable factor if source missing
if moneyflow_df is None:
    w_money = 0.0
if 'industry' not in stock_basic_df.columns or stock_basic_df['industry'].isnull().all():
    w_ind = 0.0
if daily_basic_df is None:
    w_turn = w_turn  # we use vol/amount approximate for turn; keep weight but note degraded

# re-normalize after disables
total_w = w_pct + w_volratio + w_turn + w_money + w_ind + w_tech
if total_w == 0:
    st.error("所有因子均被禁用，无法评分。")
    st.stop()
w_pct /= total_w; w_volratio /= total_w; w_turn /= total_w; w_money /= total_w; w_ind /= total_w; w_tech /= total_w

# ---------------------------
# Compose final score
# ---------------------------
score_df['综合评分'] = (
    score_df['pct_rank'] * w_pct +
    score_df['volratio_rank'] * w_volratio +
    score_df['turn_rank'] * w_turn +
    score_df['money_rank'] * w_money +
    score_df['industry_rank'] * w_ind +
    score_df['tech_rank'] * w_tech
)
score_df = score_df.sort_values('综合评分', ascending=False).reset_index(drop=True)
score_df.index += 1

# ---------------------------
# Display Top K (Top 20 default)
# ---------------------------
display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','amount','price','ma5','ten_return','macd_score','rsi_score','industry']
for c in display_cols:
    if c not in score_df.columns:
        score_df[c] = np.nan

st.success(f"评分完成，候选 {len(score_df)} 支，展示 Top {min(TOP_K, len(score_df))}（按综合评分降序）。")
st.dataframe(score_df[display_cols].head(int(TOP_K)).reset_index(drop=False), use_container_width=True)

# CSV Download (Top all)
csv = score_df[display_cols].to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载全部评分结果 CSV", data=csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

# Short explanation for each top pick (why)
st.markdown("### Top 候选说明（示例）")
top_show = score_df.head(int(TOP_K))
for idx, r in top_show.reset_index().iterrows():
    reasons = []
    if r['pct_chg'] > 5:
        reasons.append("当日强势")
    if r['vol_ratio'] and r['vol_ratio'] > 1.5:
        reasons.append("放量")
    if r['macd_score'] > 0.6:
        reasons.append("MACD 支持")
    if r['rsi_score'] > 0.6:
        reasons.append("RSI 偏强")
    if r['net_mf'] and r['net_mf'] > 0:
        reasons.append("主力净流入")
    st.write(f"{int(r['index'])}. {r['name']} ({r['ts_code']}) — Score {r['综合评分']:.4f} — {'；'.join(reasons) if reasons else '常规强势'}")

st.markdown("### 小结与提示")
st.markdown("""
- 该脚本优先使用 daily、daily_basic、stock_basic、moneyflow 等接口；若某些接口权限不足脚本会自动降级并在界面提示。  
- 若要获得最稳定、最全面的多因子结果，建议确保你的 Token 属于 5000 积分档。  
- 权重可以在侧边栏调整；如果某因子被禁用（因接口缺失），权重会自动重新归一化。  
- 建议每天仅在一个时间段运行一次（例如早盘 9:30-10:30），避免重复拉取触发限额。  
""")
