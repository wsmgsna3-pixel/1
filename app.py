# -*- coding: utf-8 -*-
"""
选股王 · 2100 积分旗舰版 (Streamlit) - 稳健版
说明：此版本对缺失字段做了大量容错，适配 5000 积分 / 部分接口不可用的情况。
"""
import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="选股王 · 2100 旗舰版（稳健版）", layout="wide")
st.title("选股王 · 2100 积分旗舰版（先过滤→涨幅→评分）（稳健容错）")

# ---------------------------
# 运行时输入 Token（首选界面输入）
# ---------------------------
TS_TOKEN = st.text_input("请输入你的 Tushare Token（只在本次会话使用）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 后才能运行。若已输入请回车确保激活。")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 参数面板（侧边栏）
# ---------------------------
st.sidebar.header("筛选参数（可调）")
INITIAL_TOP_N = st.sidebar.number_input("初筛：涨幅榜取前 N（初筛数量）", min_value=200, max_value=3000, value=1000, step=100)
FINAL_POOL = st.sidebar.number_input("进入评分池数量（最终取前多少入评分）", min_value=50, max_value=1000, value=300, step=50)
TOP_K = st.sidebar.number_input("最终展示 Top K（界面显示）", min_value=10, max_value=200, value=30, step=5)

MIN_PRICE = st.sidebar.number_input("最低价格（元）", min_value=0.1, max_value=1000.0, value=10.0, step=0.1)
MAX_PRICE = st.sidebar.number_input("最高价格（元）", min_value=1.0, max_value=2000.0, value=200.0, step=1.0)
MIN_TURNOVER = st.sidebar.number_input("换手率最低阈值（%）", min_value=0.0, max_value=100.0, value=3.0, step=0.5)
MIN_AMOUNT = st.sidebar.number_input("成交额最低阈值（元，近似）", min_value=0.0, max_value=1e12, value=200_000_000.0, step=10_000_000.0)
MAX_TOTAL_MV = st.sidebar.number_input("排除总市值 > （元）", min_value=1e8, max_value=1e13, value=1000 * 100000000.0, step=100_000_000.0)  # default ~1000亿

st.sidebar.markdown("---")
st.sidebar.markdown("评分权重（可选）")
w_pct = st.sidebar.slider("涨幅权重", 0.0, 1.0, 0.25)
w_volratio = st.sidebar.slider("量比权重", 0.0, 1.0, 0.20)
w_turn = st.sidebar.slider("换手权重", 0.0, 1.0, 0.20)
w_money = st.sidebar.slider("主力资金权重", 0.0, 1.0, 0.15)
w_ind = st.sidebar.slider("行业强度权重", 0.0, 1.0, 0.10)
w_health = st.sidebar.slider("价格健康权重", 0.0, 1.0, 0.10)

# normalize weights
total_w = w_pct + w_volratio + w_turn + w_money + w_ind + w_health
if total_w == 0:
    st.sidebar.error("权重总和不可为0，请调整。")
    st.stop()
w_pct /= total_w; w_volratio /= total_w; w_turn /= total_w; w_money /= total_w; w_ind /= total_w; w_health /= total_w

st.sidebar.markdown("---")
st.sidebar.markdown("注意：字段命名和单位可能因 Tushare 接口版本而异，脚本对缺失字段会自动降级并发出提示。")

# ---------------------------
# helper: 最近交易日（回退查找）
# ---------------------------
def get_last_trade_day(pro, max_days=10):
    today = datetime.now()
    for i in range(0, max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        try:
            dd = pro.daily(trade_date=ds)
            if dd is not None and len(dd) > 0:
                return ds
        except Exception:
            continue
    return None

last_trade = get_last_trade_day(pro)
if not last_trade:
    st.error("无法获取最近交易日，请检查 Token 或网络。")
    st.stop()

st.info(f"参考最近交易日：{last_trade}")

# ---------------------------
# 尝试加载全市场日线（一次性）
# ---------------------------
@st.cache_data(ttl=60)
def load_market_daily(trade_date):
    try:
        df = pro.daily(trade_date=trade_date)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as e:
        st.warning(f"拉取 daily 失败：{e}")
        return pd.DataFrame()

market_df = load_market_daily(last_trade)
if market_df is None or market_df.empty:
    st.error("获取当日日线失败，请确认 Token 权限。")
    st.stop()

st.write(f"当日记录数：{len(market_df)}（后续将从涨幅榜前 {INITIAL_TOP_N} 进行初筛）")

# ---------------------------
# 尝试获取 stock_basic & daily_basic & moneyflow（若权限足够）
# ---------------------------
def try_get_stock_basic():
    # request broader fields but handle missing ones
    try:
        info = pro.stock_basic(list_status='L', fields='ts_code,name,market,industry,list_date,total_mv,circ_mv')
        if info is None:
            return pd.DataFrame()
        info = info.drop_duplicates(subset=['ts_code'])
        return info
    except Exception as e:
        st.warning(f"无法获取 stock_basic（名称/市值/行业）。说明：{e}")
        return pd.DataFrame()

def try_get_daily_basic(trade_date):
    try:
        db = pro.daily_basic(trade_date=trade_date, fields='ts_code,turnover_rate,amount')
        if db is None:
            return None
        db = db.drop_duplicates(subset=['ts_code']).set_index('ts_code')
        return db
    except Exception as e:
        st.warning(f"无法获取 daily_basic（换手率/成交额）：{e}")
        return None

def try_get_moneyflow(trade_date):
    """
    尝试检测 moneyflow 的合适字段名（不同权限下字段名可能不同）。
    返回的 DataFrame index 为 ts_code，包含一列 'net_mf'（统一列名）。
    """
    try:
        mf = pro.moneyflow(trade_date=trade_date)
        if mf is None or mf.empty:
            st.warning("moneyflow 返回为空（接口可用但无数据）。")
            return None
        mf = mf.drop_duplicates(subset=['ts_code'])
        # 尝试可能的主力列名
        possible = ['net_mf', 'net_mf_amount', 'net_mf_amt', 'net_mf_vol', 'net_mf_amt_main', 'net_mf_main']
        found = None
        for c in possible:
            if c in mf.columns:
                found = c
                break
        if found is None:
            # 尝试任何以 'net' 开头的列
            for c in mf.columns:
                if isinstance(c, str) and c.lower().startswith('net'):
                    found = c
                    break
        if found is None:
            st.warning("moneyflow 中未找到可识别的主力净流字段，资金项将降级为0。")
            return None
        mf2 = mf[['ts_code', found]].rename(columns={found: 'net_mf'}).set_index('ts_code')
        return mf2
    except Exception as e:
        st.warning(f"无法获取 moneyflow（主力净流），评分中此项将降级为0：{e}")
        return None

stock_basic_df = try_get_stock_basic()
daily_basic_df = try_get_daily_basic(last_trade)
moneyflow_df = try_get_moneyflow(last_trade)

# 友好提示已取得哪些关键表
st.write("stock_basic 行数：", 0 if stock_basic_df is None else len(stock_basic_df))
st.write("daily_basic 可用：" , "是" if (daily_basic_df is not None and not daily_basic_df.empty) else "否")
st.write("moneyflow 可用：" , "是" if (moneyflow_df is not None and not moneyflow_df.empty) else "否")

# ---------------------------
# 初筛：按涨幅取 top N（减少后续调用）
# ---------------------------
pool = market_df.sort_values('pct_chg', ascending=False).head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

# --- 合并部分基本信息（若 available），但先检测字段
if not stock_basic_df.empty:
    want_cols = ['ts_code','name','industry','total_mv','circ_mv']
    exist_cols = [c for c in want_cols if c in stock_basic_df.columns]
    if len(exist_cols) < len(want_cols):
        missing = set(want_cols) - set(exist_cols)
        st.warning(f"stock_basic 缺少字段：{missing}。会使用可用字段并对缺失项降级。")
    # merge only existing cols
    try:
        pool = pool.merge(stock_basic_df[exist_cols], on='ts_code', how='left')
    except Exception as e:
        st.warning(f"合并 stock_basic 失败（回退为不合并）：{e}")
else:
    # ensure name/industry columns exist to avoid later KeyError
    pool['name'] = pool['ts_code']
    pool['industry'] = ""

# 合并 daily_basic（换手率/amount）若可用
if daily_basic_df is not None and not daily_basic_df.empty:
    want_db_cols = ['turnover_rate','amount']
    exist_db_cols = [c for c in want_db_cols if c in daily_basic_df.columns]
    if len(exist_db_cols) < len(want_db_cols):
        st.warning(f"daily_basic 缺少字段：{set(want_db_cols)-set(exist_db_cols)}。将使用可用字段。")
    try:
        # join on index
        pool = pool.set_index('ts_code').join(daily_basic_df[exist_db_cols], how='left').reset_index()
    except Exception as e:
        st.warning(f"daily_basic 合并失败：{e}")
        # ensure the columns exist
        if 'turnover_rate' not in pool.columns:
            pool['turnover_rate'] = np.nan
        if 'amount' not in pool.columns:
            pool['amount'] = np.nan
else:
    # ensure columns exist
    if 'turnover_rate' not in pool.columns:
        pool['turnover_rate'] = np.nan
    if 'amount' not in pool.columns:
        pool['amount'] = np.nan

# 合并 moneyflow 若可用（moneyflow_df 已经将主力列重命名为 net_mf）
if moneyflow_df is not None and not moneyflow_df.empty:
    try:
        pool = pool.set_index('ts_code').join(moneyflow_df[['net_mf']], how='left').reset_index()
    except Exception as e:
        st.warning(f"moneyflow 合并失败：{e}")
        if 'net_mf' not in pool.columns:
            pool['net_mf'] = 0.0
else:
    if 'net_mf' not in pool.columns:
        pool['net_mf'] = 0.0

# ---------------------------
# 清洗候选池（按你要求的规则）
# ---------------------------
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except:
        return default

cleaned = []
for idx, r in pool.iterrows():
    ts = r.get('ts_code', None)
    if not ts:
        continue

    # 排除停牌（vol==0 或 amount==0）
    vol = safe_float(r.get('vol', 0))
    amount = safe_float(r.get('amount', np.nan))

    # 若 amount 为 NaN，但 market_df 有 amount 列则使用
    if pd.isna(amount) and 'amount' in r:
        amount = safe_float(r.get('amount', np.nan))

    # 如果仍然 NaN，尝试用成交额换算：vol * close（若 vol 与 close 存在且 vol 单位是手）
    if pd.isna(amount):
        if 'vol' in r and 'close' in r and not pd.isna(r.get('vol')) and not pd.isna(r.get('close')):
            try:
                amount = float(r.get('vol')) * float(r.get('close'))
            except:
                amount = np.nan

    # 认为停牌或无成交的直接跳过
    if vol == 0 or (pd.isna(amount) or amount == 0):
        continue

    # 价格区间
    price = safe_float(r.get('close', r.get('open', np.nan)))
    if pd.isna(price):
        continue
    if price < MIN_PRICE or price > MAX_PRICE:
        continue

    # 排除 ST（如果 name 可用）
    name = r.get('name', '') if 'name' in r else ''
    if isinstance(name, str) and name != "":
        up = name.upper()
        if 'ST' in up or '退' in up:
            continue

    # 排除总市值 > MAX_TOTAL_MV （需先确定 stock_basic 提供的单位）
    total_mv = r.get('total_mv', np.nan)
    if not (pd.isna(total_mv)):
        try:
            tv = float(total_mv)
            # 如果 tv 看起来像是万元级别（较大），把它 *10000 转换为元
            if tv > 1e6:
                tv_yuan = tv * 10000.0
            else:
                tv_yuan = tv
            if tv_yuan > MAX_TOTAL_MV:
                continue
        except:
            pass

    # 排除换手率低于 MIN_TURNOVER（若 daily_basic 有值）
    tr = safe_float(r.get('turnover_rate', np.nan))
    if not pd.isna(tr):
        if tr < MIN_TURNOVER:
            continue
    else:
        # 无 daily_basic 时，不强制排除（保守策略）
        pass

    # 排除成交额过低（近似）
    amt = amount if not pd.isna(amount) else 0.0
    # some API return amount in 万元; normalize: 若小于1e5则认为是万元，乘以10000
    if amt > 0 and amt < 1e5:
        amt = amt * 10000.0
    if amt < MIN_AMOUNT:
        continue

    # 排除一字板（open==high==low==pre_close）
    try:
        if (safe_float(r.get('open',0)) == safe_float(r.get('high',0)) == safe_float(r.get('low',0)) == safe_float(r.get('pre_close',0))):
            continue
    except:
        pass

    cleaned.append(r)

cleaned_df = pd.DataFrame(cleaned).reset_index(drop=True)
st.write(f"清洗后候选数量：{len(cleaned_df)} （将从中取涨幅前 {FINAL_POOL} 进入评分阶段）")
if len(cleaned_df) == 0:
    st.error("清洗后无候选，请放宽条件或检查 Token 权限（例如 daily_basic / stock_basic 是否完整）。")
    st.stop()

# ---------------------------
# 取涨幅前 FINAL_POOL
# ---------------------------
cleaned_df = cleaned_df.sort_values('pct_chg', ascending=False).head(int(FINAL_POOL)).reset_index(drop=True)
st.write(f"用于评分的池子大小：{len(cleaned_df)}")

# ---------------------------
# 辅助：获取单只历史（20 日）并缓存（用于量比/10日收益等）
# ---------------------------
@st.cache_data(ttl=600)
def get_hist(ts_code, end_date, days=20):
    try:
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date')
        return df
    except Exception:
        return None

# ---------------------------
# 评分计算（多因子，归一化为 0-1）
# ---------------------------
records = []
pbar = st.progress(0)
n = len(cleaned_df)
for i, row in enumerate(cleaned_df.itertuples()):
    ts = getattr(row, 'ts_code', None)
    pct = safe_float(getattr(row, 'pct_chg', 0))
    turnover = safe_float(getattr(row, 'turnover_rate', np.nan))
    # amount：优先 cleaned_df 的 amount，再尝试 row.amount 字段
    amount_val = safe_float(getattr(row, 'amount', np.nan))
    if pd.isna(amount_val):
        # try market_df
        try:
            amount_val = float(cleaned_df.loc[cleaned_df['ts_code']==ts, 'amount'].values[0])
        except:
            amount_val = np.nan
    if amount_val > 0 and amount_val < 1e5:
        amount_val *= 10000.0

    # try to get 10/20d history
    hist = get_hist(ts, last_trade, days=20)
    if hist is None or len(hist) < 5:
        avg_vol_5 = np.nan
        vol_ratio = np.nan
        ten_return = np.nan
    else:
        hist_tail = hist.tail(10)
        vols = hist_tail['vol'].astype(float).tolist()
        if len(vols) >= 6:
            avg_vol_5 = np.mean(vols[:-1][-5:])
        else:
            avg_vol_5 = np.mean(vols[:-1]) if len(vols[:-1])>0 else np.nan
        vol_today = float(vols[-1]) if len(vols)>0 else np.nan
        vol_ratio = vol_today / (avg_vol_5+1e-9) if not np.isnan(avg_vol_5) and avg_vol_5>0 else np.nan
        ten_return = (hist_tail.iloc[-1]['close'] / hist_tail.iloc[0]['open'] - 1) if len(hist_tail)>=2 else np.nan

    # moneyflow if available
    net_mf = 0.0
    if moneyflow_df is not None and not moneyflow_df.empty:
        try:
            if ts in moneyflow_df.index:
                net_mf = float(moneyflow_df.loc[ts, 'net_mf'])
            else:
                net_mf = 0.0
        except:
            net_mf = 0.0

    # industry if available in merged cleaned_df or stock_basic
    industry = ''
    if 'industry' in cleaned_df.columns:
        try:
            industry = cleaned_df.loc[cleaned_df['ts_code']==ts, 'industry'].values[0]
        except:
            industry = ''
    elif not stock_basic_df.empty and 'industry' in stock_basic_df.columns:
        try:
            industry = stock_basic_df.loc[stock_basic_df['ts_code']==ts, 'industry'].values[0]
        except:
            industry = ''

    records.append({
        'ts_code': ts,
        'name': getattr(row, 'name', ts),
        'pct_chg': pct,
        'turnover_rate': turnover,
        'amount': amount_val if not pd.isna(amount_val) else 0.0,
        'vol_ratio': vol_ratio if not pd.isna(vol_ratio) else 1.0,
        'net_mf': net_mf,
        'ten_return': ten_return if not pd.isna(ten_return) else 0.0,
        'price': safe_float(getattr(row, 'close', getattr(row, 'open', np.nan))),
        'open': safe_float(getattr(row, 'open', np.nan)),
        'pre_close': safe_float(getattr(row, 'pre_close', np.nan)),
        'industry': industry if industry is not None else ''
    })
    pbar.progress((i+1)/n if n>0 else 1.0)

pbar.progress(1.0)
score_df = pd.DataFrame(records)

# ---------------------------
# 计算行业热度（基础：行业内平均涨幅，用于 industry_score）
# ---------------------------
if 'industry' in score_df.columns:
    ind_mean = score_df.groupby('industry')['pct_chg'].transform('mean')
    # industry strength normalized
    if ind_mean.max() - ind_mean.min() < 1e-9:
        score_df['industry_score'] = 0.0
    else:
        score_df['industry_score'] = (ind_mean - ind_mean.min()) / (ind_mean.max() - ind_mean.min() + 1e-9)
    score_df['industry_score'] = score_df['industry_score'].fillna(0.0)
else:
    score_df['industry_score'] = 0.0

# ---------------------------
# 归一化每个子指标到 0-1（越大越好）
# ---------------------------
def norm_series(s):
    s = pd.Series(s).astype(float)
    if s.isnull().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn = s.min()
    mx = s.max()
    if mx - mn < 1e-9:
        return pd.Series(np.ones(len(s))*0.5, index=s.index)
    return (s - mn) / (mx - mn)

score_df['pct_rank'] = norm_series(score_df['pct_chg'].fillna(0))
score_df['volratio_rank'] = norm_series(score_df['vol_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0))
score_df['turn_rank'] = norm_series(score_df['turnover_rate'].fillna(0))
score_df['money_rank'] = norm_series(score_df['net_mf'].fillna(0))
score_df['amount_norm'] = norm_series(score_df['amount'].fillna(0))
score_df['health_score'] = 1.0 - 0.1 * score_df['amount_norm']
score_df['health_score'] = score_df['health_score'].clip(0,1)

# ---------------------------
# 综合评分（加权）
# ---------------------------
score_df['综合评分'] = (
    score_df['pct_rank'] * w_pct +
    score_df['volratio_rank'] * w_volratio +
    score_df['turn_rank'] * w_turn +
    score_df['money_rank'] * w_money +
    score_df['industry_score'] * w_ind +
    score_df['health_score'] * w_health
)

# 排序并输出 Top_K
score_df = score_df.sort_values('综合评分', ascending=False).reset_index(drop=True)
score_df.index += 1

st.success(f"评分完成，候选 {len(score_df)} 支，展示 Top {min(TOP_K, len(score_df))}。")
display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','amount','price','open','pre_close','ten_return']
for c in display_cols:
    if c not in score_df.columns:
        score_df[c] = np.nan

st.dataframe(score_df[display_cols].head(int(TOP_K)), use_container_width=True)

# CSV 下载
csv = score_df[display_cols].to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载全部评分结果 CSV", data=csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

# 小结与提示
st.markdown("### 小结与注意事项")
st.markdown("""
- 此版本已尽可能对 stock_basic/daily_basic/moneyflow 的字段差异做容错：  
  - 若 stock_basic 缺少某些列（如 total_mv/circ_mv/industry），脚本会跳过并在界面提醒。  
  - 若 daily_basic 或 moneyflow 不可用，相关因子会降级为保守默认（不再报 KeyError）。  
- 如果你仍然看到“清洗后无候选”或“选出数量为0”，请检查：  
  1. 权重与阈值是否过严（例如 MIN_AMOUNT 带来严格剪枝）  
  2. Token 是否确实有 daily_basic / stock_basic / moneyflow 的权限  
- 我可以继续为你做：  
  - “显示每一步被刷掉多少票”的更详细过滤统计版本，或  
  - 自动放宽阈值直到至少选出 N 支股票（自动调参版）。  
如需其中一个，请回复“过滤统计版”或“自动调参版”。  
""")

st.info("如需我继续优化（例如加入 MA 选股、放量判断或行业轮动逻辑），告诉我你的偏好，我会在此基础上给出完整可替换代码。")
