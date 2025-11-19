# -*- coding: utf-8 -*-
"""
最终版：自动找交易日 + 自动字段修复 + 自动补缺列 + 自动防止清洗归零
请一次性覆盖仓库里的 app.py 并运行（运行前请备份旧文件）
"""

import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time
import math
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="选股王 · 最终稳健版", layout="wide")
st.title("选股王 · 最终稳健版（自动容错 + 分级放宽）")
st.markdown("说明：本脚本会自动寻找最近交易日、修补缺失列、并在清洗后自动分级放宽条件以保证不会无候选。")

# -------------------------
# 侧边栏参数（可调）
# -------------------------
with st.sidebar:
    st.header("参数（可调）")
    TOKEN = st.text_input("Tushare Token（本次运行使用）", type="password")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=500, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=20, step=5))
    MIN_PRICE = float(st.number_input("最低价格(元)（初始）", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格(元)（初始）", value=200.0, step=10.0))
    MIN_TURNOVER = float(st.number_input("最低换手率(％)（初始）", value=3.0, step=0.5))
    MIN_AMOUNT = float(st.number_input("最低成交额(元)（初始）", value=200_000_000.0, step=50_000_000.0))
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=8.0, step=0.5))
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    USE_RELAX = st.checkbox("启用分级放宽（若严格清洗后无候选）", value=True)
    st.caption("建议保持启用。若想严格把控可取消。")

if not TOKEN:
    st.warning("请输入 Tushare Token（侧栏）。")
    st.stop()

ts.set_token(TOKEN)
pro = ts.pro_api()

# -------------------------
# 辅助函数：安全调用与字段修复
# -------------------------
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

def ensure_columns(df, required_cols):
    """确保 df 包含 required_cols，中间缺的用 NaN 或合理默认补齐"""
    if df is None:
        df = pd.DataFrame()
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# -------------------------
# 稳健寻找最近交易日（trade_cal 优先 + 验证 daily 是否有数据）
# -------------------------
@st.cache_data(ttl=300)
def find_recent_trade_day(max_back_days=7):
    """返回 YYYYMMDD 格式的最近交易日。逻辑：
       1) 用 trade_cal 找到最近 open day
       2) 验证 pro.daily(trade_date=that_day) 确实有数据
       3) 若没有则回溯最多 max_back_days 天逐日检验
    """
    today = date.today()
    # try trade_cal first
    try:
        cal = safe_get(pro.trade_cal, exchange='SSE', start_date=(today - timedelta(days=30)).strftime("%Y%m%d"),
                       end_date=today.strftime("%Y%m%d"))
        if not cal.empty:
            cal = cal[cal['is_open'] == 1]
            # iterate from latest to earliest in cal
            dates = sorted(cal['cal_date'].unique(), reverse=True)
            for d in dates:
                # verify daily has data
                dd = str(d)
                try:
                    df = safe_get(pro.daily, trade_date=dd)
                    if not df.empty:
                        return dd
                except:
                    continue
    except Exception:
        pass
    # fallback: brute force back-check last max_back_days
    for i in range(0, max_back_days):
        dd = (today - timedelta(days=i)).strftime("%Y%m%d")
        try:
            df = safe_get(pro.daily, trade_date=dd)
            if not df.empty:
                return dd
        except:
            continue
    # last fallback: today as string (may be weekend)
    return today.strftime("%Y%m%d")

last_trade = find_recent_trade_day(max_back_days=14)
st.info(f"使用的最近交易日（已验证 daily 数据存在）：{last_trade}")

# -------------------------
# 拉取当日 daily 初筛（并标准化列）
# -------------------------
st.write("拉取当日 daily（用于涨幅榜初筛）...")
daily_all = safe_get(pro.daily, trade_date=last_trade)
# ensure core cols
daily_all = ensure_columns(daily_all, ["ts_code","trade_date","open","high","low","close","pre_close","vol","amount","pct_chg"])
if daily_all.empty:
    st.error("无法获取当日 daily（或 daily 返回为空）。请检查 Token/积分 或 稍后重试。")
    st.stop()

daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
st.write(f"当日记录：{len(daily_all)}，取涨幅前 {INITIAL_TOP_N} 作为初筛。")
pool0 = daily_all.head(INITIAL_TOP_N).copy()

# -------------------------
# 尝试拉取附表：stock_basic, daily_basic, moneyflow（均 safe）
# -------------------------
st.write("尝试加载 stock_basic / daily_basic / moneyflow（权限允许时可用）...")
stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,total_mv,circ_mv')
stock_basic = ensure_columns(stock_basic, ["ts_code","name","industry","total_mv","circ_mv"])

daily_basic = safe_get(pro.daily_basic, trade_date=last_trade, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
daily_basic = ensure_columns(daily_basic, ["ts_code","turnover_rate","amount","total_mv","circ_mv"])

mf_raw = safe_get(pro.moneyflow, trade_date=last_trade)
# standardize moneyflow
if mf_raw is None or mf_raw.empty:
    moneyflow = pd.DataFrame(columns=["ts_code","net_mf"])
else:
    # pick plausible numeric column as net_mf if net_mf missing
    if "net_mf" not in mf_raw.columns:
        numeric_candidates = [c for c in mf_raw.columns if c != "ts_code" and pd.api.types.is_numeric_dtype(mf_raw[c])]
        if numeric_candidates:
            mf_raw = mf_raw.rename(columns={numeric_candidates[0]:"net_mf"})
    moneyflow = ensure_columns(mf_raw, ["ts_code","net_mf"])
moneyflow["net_mf"] = moneyflow["net_mf"].fillna(0)

# -------------------------
# 合并并修复列（保证后续不会因为 KeyError 崩溃）
# -------------------------
# merge with pool0 safely
def safe_merge(pool_df, other_df, on='ts_code', how='left'):
    if other_df is None or other_df.empty:
        return pool_df
    if 'ts_code' not in other_df.columns:
        other_df['ts_code'] = None
    try:
        merged = pool_df.merge(other_df, on='ts_code', how=how)
    except Exception:
        # fallback: join by index
        merged = pool_df.copy()
    return merged

pool = safe_merge(pool0, stock_basic[['ts_code','name','industry']], on='ts_code', how='left')
pool = safe_merge(pool, daily_basic[['ts_code','turnover_rate','amount','total_mv','circ_mv']], on='ts_code', how='left')
pool = safe_merge(pool, moneyflow[['ts_code','net_mf']], on='ts_code', how='left')

# ensure columns exist
core_cols = ["ts_code","name","open","high","low","close","pre_close","pct_chg","vol","amount","turnover_rate","net_mf","total_mv","circ_mv"]
pool = ensure_columns(pool, core_cols)

# normalize amount: some APIs return amount in 万元 -> if median amount < 1e6 treat as 万元
if pool['amount'].dropna().shape[0] > 0:
    median_amt = pool['amount'].dropna().median()
    if median_amt < 1e6:
        pool['amount'] = pool['amount'].apply(lambda x: x*10000 if not pd.isna(x) and x>0 and x<1e6 else x)

# -------------------------
# 清洗函数（可被分级放宽）
# -------------------------
def do_clean(df, min_price, max_price, min_turnover, min_amount):
    df2 = df.copy()
    # repair missing close from pre_close
    if "close" not in df2.columns or df2['close'].isna().all():
        if "pre_close" in df2.columns:
            df2['close'] = df2['pre_close']
        else:
            df2['close'] = 0.0
    # basic filters
    df2 = df2[~df2['name'].str.contains("ST", na=False)]
    # price
    df2 = df2[(df2['close'] >= min_price) & (df2['close'] <= max_price)]
    # nonzero trading
    df2 = df2[(df2['vol'].fillna(0) > 0) & (df2['amount'].fillna(0) > 0)]
    # exclude one-word board where open==high==low==pre_close (likely 一字板)
    try:
        mask_one_word = (df2['open'] == df2['high']) & (df2['open'] == df2['low']) & (df2['open'] == df2['pre_close'])
        df2 = df2[~mask_one_word.fillna(False)]
    except:
        pass
    # turnover & amount filters (turnover likely in %)
    try:
        df2 = df2[df2['turnover_rate'].fillna(0) >= min_turnover]
    except:
        pass
    # amount filter (already normalized)
    df2 = df2[df2['amount'].fillna(0) >= min_amount]
    # exclude yesterday down
    df2 = df2[df2['pct_chg'].fillna(0) >= 0]
    return df2

# -------------------------
# 执行分级清洗：严格 -> 放宽 -> 回退到前一交易日 -> 最后兜底少量放宽到显示
# -------------------------
st.write("对初筛池进行清洗（含分级放宽与回退机制）...")
log_msgs = []
# 级别1：默认严格
min_price = MIN_PRICE; max_price = MAX_PRICE
min_turn = MIN_TURNOVER; min_amt = MIN_AMOUNT

clean_df = do_clean(pool, min_price, max_price, min_turn, min_amt)
log_msgs.append(f"严格过滤后数量：{len(clean_df)}")

# 级别2：若为空且允许放宽，逐步放宽阈值
relax_steps = [
    {"min_price": max(1, MIN_PRICE*0.8), "max_price": MAX_PRICE*1.2, "min_turn": max(0.5, MIN_TURNOVER*0.7), "min_amt": max(5e6, MIN_AMOUNT*0.5)},
    {"min_price": 1, "max_price": 9999, "min_turn": 0.5, "min_amt": 1e6}
]

used_relax = None
if len(clean_df) == 0 and USE_RELAX:
    for step in relax_steps:
        tmp = do_clean(pool, step["min_price"], step["max_price"], step["min_turn"], step["min_amt"])
        log_msgs.append(f"放宽尝试：min_price={step['min_price']}, min_turn={step['min_turn']}, min_amt={int(step['min_amt'])}  -> 数量 {len(tmp)}")
        if len(tmp) > 0:
            clean_df = tmp
            used_relax = step
            break

# 级别3：若仍为空，回退到前一个交易日（最多回退 3 日）
used_back_day = None
if len(clean_df) == 0:
    st.warning("严格与放宽后仍无候选，尝试回退到前一交易日获取数据。")
    for back in range(1,4):
        cand_day = (datetime.strptime(last_trade, "%Y%m%d").date() - timedelta(days=back)).strftime("%Y%m%d")
        daily_try = safe_get(pro.daily, trade_date=cand_day)
        if daily_try is None or daily_try.empty:
            log_msgs.append(f"回退 {back} 天 ({cand_day}) 无 daily 数据。")
            continue
        # rebuild pool for that day
        pool_try = daily_try.sort_values("pct_chg", ascending=False).head(INITIAL_TOP_N).reset_index(drop=True)
        pool_try = safe_merge(pool_try, stock_basic[['ts_code','name','industry']], on='ts_code', how='left')
        pool_try = safe_merge(pool_try, daily_basic[['ts_code','turnover_rate','amount','total_mv','circ_mv']], on='ts_code', how='left')
        pool_try = safe_merge(pool_try, moneyflow[['ts_code','net_mf']], on='ts_code', how='left')
        pool_try = ensure_columns(pool_try, core_cols)
        # apply relaxed cleaning (more permissive to avoid empty)
        tmp = do_clean(pool_try, max(1, MIN_PRICE*0.5), MAX_PRICE*2, max(0.5, MIN_TURNOVER*0.5), max(1e6, MIN_AMOUNT*0.2))
        log_msgs.append(f"回退 {back} 天 ({cand_day}) 过滤后数量：{len(tmp)}")
        if len(tmp) > 0:
            clean_df = tmp
            used_back_day = cand_day
            break

# 最后兜底：若仍空，宽松取 topN 显示（避免页面空白）
if len(clean_df) == 0:
    st.error("所有智能清洗与回退尝试均未得到候选，启用最终兜底：取初筛 top 50（仅供观察，不建议实盘）。")
    clean_df = pool.head(50).copy()
    log_msgs.append("启用最终兜底 top50。")

# 展示日志
st.write("清洗与容错日志：")
for m in log_msgs:
    st.write("- " + str(m))
if used_relax:
    st.info(f"已采用放宽规则：{used_relax}")
if used_back_day:
    st.info(f"已回退并使用交易日：{used_back_day}")

st.write(f"清洗后候选数量：{len(clean_df)} （将从中取涨幅前 {FINAL_POOL} 进入评分阶段）")
if clean_df.empty:
    st.error("清洗仍为空，程序终止。")
    st.stop()

# -------------------------
# 取涨幅前 FINAL_POOL 进入评分
# -------------------------
clean_df = clean_df.sort_values("pct_chg", ascending=False).head(FINAL_POOL).reset_index(drop=True)

# -------------------------
# 历史拉取与指标（缓存）
# -------------------------
@st.cache_data(ttl=600)
def get_hist(ts_code, end_date, days=60):
    try:
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = safe_get(pro.daily, ts_code=ts_code, start_date=start, end_date=end_date)
        if df.empty:
            return pd.DataFrame()
        return df.sort_values("trade_date").reset_index(drop=True)
    except:
        return pd.DataFrame()

def compute_indicators(df):
    res = {}
    if df is None or df.empty or len(df) < 6:
        return res
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    vols = df['vol'].astype(float)
    # MA
    for n in (5,10,20):
        res[f"ma{n}"] = close.rolling(window=n).mean().iloc[-1] if len(close) >= n else np.nan
    # MACD
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        res['macd'] = (diff - dea).iloc[-1] * 2
    else:
        res['macd'] = np.nan
    # vol_ratio
    if len(vols) >= 6:
        res['vol_ratio'] = vols.iloc[-1] / (vols.iloc[-6:-1].mean() + 1e-9)
        res['vol_last'] = vols.iloc[-1]
        res['vol_ma5'] = vols.iloc[-6:-1].mean()
    else:
        res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan
    # returns
    if len(close) >= 11:
        res['10d_return'] = close.iloc[-1]/close.iloc[-11] - 1
    else:
        res['10d_return'] = np.nan
    if len(close) >= 21:
        res['20d_return'] = close.iloc[-1]/close.iloc[-21] - 1
    else:
        res['20d_return'] = np.nan
    # prev3
    if 'pct_chg' in df.columns and len(df) >= 4:
        try:
            res['prev3_sum'] = df['pct_chg'].astype(float).iloc[-4:-1].sum()
        except:
            res['prev3_sum'] = np.nan
    else:
        res['prev3_sum'] = np.nan
    # volatility
    if 'pct_chg' in df.columns and len(df) >= 10:
        res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()
    else:
        res['volatility_10'] = np.nan
    return res

# -------------------------
# 评分池逐票计算因子
# -------------------------
st.write("为评分池逐票拉历史并计算指标（已启用缓存）...")
records = []
pbar = st.progress(0)
for idx, row in enumerate(clean_df.itertuples()):
    ts_code = getattr(row, 'ts_code')
    name = getattr(row, 'name', ts_code)
    pct_chg = getattr(row, 'pct_chg', 0.0)
    amount = getattr(row, 'amount', np.nan)
    if amount is not None and not pd.isna(amount) and amount > 0 and amount < 1e6:
        amount = amount * 10000.0
    turnover_rate = getattr(row, 'turnover_rate', np.nan)
    net_mf = float(getattr(row, 'net_mf', 0.0))

    hist = get_hist(ts_code, last_trade, days=60)
    ind = compute_indicators(hist)

    vol_ratio = ind.get('vol_ratio', np.nan)
    ten_return = ind.get('10d_return', np.nan)
    macd = ind.get('macd', np.nan)
    prev3_sum = ind.get('prev3_sum', np.nan)
    volatility_10 = ind.get('volatility_10', np.nan)

    # proxy money if no net_mf
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
        'macd': macd,
        'prev3_sum': prev3_sum,
        'volatility_10': volatility_10,
        'proxy_money': proxy_money
    }
    records.append(rec)
    pbar.progress((idx+1)/len(clean_df))

pbar.progress(1.0)
fdf = pd.DataFrame(records)
if fdf.empty:
    st.error("评分计算失败或无数据，请检查 Token 权限与接口。")
    st.stop()

# -------------------------
# 风险过滤（增强）
# -------------------------
st.write("执行增强型风险过滤...")
try:
    before = len(fdf)
    # 高位大阳
    if all(c in fdf.columns for c in ['pct_chg','10d_return']):
        mask_high = (fdf['10d_return']*100 > 100) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
        fdf = fdf[~mask_high]
    # 下跌途中反抽
    if 'prev3_sum' in fdf.columns:
        mask_down = (fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD)
        fdf = fdf[~mask_down]
    # 巨量放量
    if all(c in fdf.columns for c in ['vol_ratio']):
        mask_vol = fdf['vol_ratio'] > VOL_SPIKE_MULT
        fdf = fdf[~mask_vol]
    # 极端波动
    if 'volatility_10' in fdf.columns:
        mask_v = fdf['volatility_10'] > VOLATILITY_MAX
        fdf = fdf[~mask_v]
    after = len(fdf)
    st.write(f"风险过滤：{before} -> {after}")
except Exception as e:
    st.warning(f"风险过滤异常（忽略）：{e}")

# -------------------------
# 归一化与综合评分（BC 混合）
# -------------------------
def safe_norm(s):
    s = s.fillna(0.0).replace([np.inf,-np.inf],0.0)
    mn = s.min(); mx = s.max()
    if mx - mn < 1e-9:
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn)/(mx - mn)

fdf['s_pct'] = safe_norm(fdf.get('pct_chg', pd.Series([0]*len(fdf))))
fdf['s_volratio'] = safe_norm(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))
fdf['s_turn'] = safe_norm(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))
fdf['s_money'] = safe_norm(fdf.get('net_mf', pd.Series([0]*len(fdf)))) if fdf['net_mf'].abs().sum() > 0 else safe_norm(fdf.get('proxy_money', pd.Series([0]*len(fdf))))
fdf['s_10d'] = safe_norm(fdf.get('10d_return', pd.Series([0]*len(fdf))))
fdf['s_macd'] = safe_norm(fdf.get('macd', pd.Series([0]*len(fdf))))
fdf['s_volatility'] = 1 - safe_norm(fdf.get('volatility_10', pd.Series([0]*len(fdf))))

w_pct=0.16; w_volratio=0.18; w_turn=0.12; w_money=0.14; w_10d=0.16; w_macd=0.06; w_volatility=0.08
fdf['score'] = (
    fdf['s_pct']*w_pct + fdf['s_volratio']*w_volratio + fdf['s_turn']*w_turn +
    fdf['s_money']*w_money + fdf['s_10d']*w_10d + fdf['s_macd']*w_macd + fdf['s_volatility']*w_volatility
)

# -------------------------
# 输出与下载（只显示 TOP_DISPLAY，可调）
# -------------------------
fdf = fdf.sort_values('score', ascending=False).reset_index(drop=True)
fdf.index = fdf.index + 1

display_cols = ['name','ts_code','score','pct_chg','vol_ratio','turnover_rate','net_mf','proxy_money','10d_return','macd','volatility_10']
for c in display_cols:
    if c not in fdf.columns:
        fdf[c] = np.nan

st.success(f"评分完成，总候选 {len(fdf)}，显示 Top {min(TOP_DISPLAY, len(fdf))}")
st.dataframe(fdf[display_cols].head(TOP_DISPLAY), use_container_width=True)

# 下载（导出前 TOP_DISPLAY）
out_csv = fdf[display_cols].head(TOP_DISPLAY).to_csv(index=False, encoding='utf-8-sig')
st.download_button("下载评分结果（前N）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

# -------------------------
# 结尾提示
# -------------------------
st.markdown("### 运行说明")
st.markdown("""
- 已启用：自动寻找交易日、字段修复、分级放宽与回退机制。  
- 若出现“回退”或“放宽”日志，请优先检查 Token/积分与当日接口响应；脚本已尽量容错以保证页面不会返回空。  
- 想把输出默认改为 20 / 30 / 50，可在侧栏调整 Top K。  
""")
