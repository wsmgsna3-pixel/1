# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="选股王 · 2100积分专业版", layout="wide")
st.title("选股王 · 2100积分专业版（批量 API，避免逐票循环）")

# ---------------------------
# --- 用户输入区（保留你习惯的 token 界面）
# ---------------------------
TS_TOKEN = st.text_input("请输入你的 Tushare Token（仅本次使用，不会保存）", type="password")
if not TS_TOKEN:
    st.info("请输入 Tushare Token 后才能运行选股")
    st.stop()

# 导入 tushare（在运行环境已有 tushare 库）
import tushare as ts
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# --- 参数（可按需调整）
# ---------------------------
TOP_N = st.number_input("最终取前 N（排序后）", min_value=1, max_value=200, value=20, step=1)
MIN_CIRC_MV = st.number_input("流通市值下限（亿）", min_value=1.0, value=20.0, step=1.0)
MAX_CIRC_MV = st.number_input("流通市值上限（亿）", min_value=10.0, value=500.0, step=10.0)
MIN_TURNOVER = st.number_input("换手率下限（%）", min_value=0.1, value=3.0, step=0.1)
AMOUNT_PCT_OF_CIRC = st.number_input("成交额至少为流通市值的百分比（例如 1.2% 填 1.2）", min_value=0.1, value=1.2, step=0.1)
MIN_PRICE = st.number_input("股价下限（元）", min_value=0.1, value=10.0, step=0.1)
MAX_PRICE = st.number_input("股价上限（元）", min_value=1.0, value=200.0, step=1.0)
OPEN_MIN_RATIO = st.number_input("开盘相对昨收最低比例（例如 0.99 表示 >= 昨收*0.99）", min_value=0.8, max_value=1.2, value=0.99, step=0.01)
CONTINUOUS_DOWN_DAYS = st.number_input("连续多少日下跌视为禁止（默认3）", min_value=1, value=3, step=1)
RETURN_10D_MAX_PCT = st.number_input("过去10日最大涨幅阈值（%，默认80）", min_value=1.0, value=80.0, step=1.0)

# ---------------------------
# --- 工具函数：取最近交易日与历史交易日列表
# ---------------------------
@st.cache_data(ttl=300)
def get_trade_calendar(n_days=30):
    # 获取近 n_days 的交易日（使用 trade_cal）
    today = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=n_days*2)).strftime("%Y%m%d")
    try:
        cal = pro.trade_cal(exchange='', start_date=start, end_date=today, fields='cal_date,is_open')
        cal = cal[cal['is_open'] == 1].sort_values('cal_date')
        return cal['cal_date'].tolist()
    except Exception as e:
        st.warning("无法调用 trade_cal（权限/网络问题）。将尝试简单后退到昨天作为交易日。")
        # 简单回退：昨天或上周五
        d = datetime.now()
        if d.weekday() == 5:
            d = d - timedelta(days=1)
        elif d.weekday() == 6:
            d = d - timedelta(days=2)
        return [d.strftime("%Y%m%d")]

trade_dates = get_trade_calendar(60)
if len(trade_dates) < 12:
    st.warning("可用的交易日很少，数据可能不完整：%s" % str(trade_dates[:5]))

last_trade = trade_dates[-1]
st.info(f"使用参考交易日（最近交易日）：{last_trade}")

# ---------------------------
# --- 批量拉取必需数据（一次性批量）
# ---------------------------
@st.cache_data(ttl=180)
def fetch_bulk_data(last_trade, lookback_days=15):
    # 1) 当日全市场日线
    df_daily = pd.DataFrame()
    try:
        df_daily = pro.daily(trade_date=last_trade)
    except Exception as e:
        st.error("拉取当日 daily 失败，请检查 Token/权限/网络。错误：" + str(e))
        return None

    # 2) 当日 daily_basic（换手/流通市值等）
    df_db = pd.DataFrame()
    try:
        df_db = pro.daily_basic(trade_date=last_trade)
    except Exception as e:
        st.warning("daily_basic 拉取失败，将无法使用真实换手/流通市值等字段，部分过滤将降级为近似。")
        df_db = pd.DataFrame()

    # 3) stock_basic（一次性拿全市场基本信息）
    df_stock_basic = pd.DataFrame()
    try:
        df_stock_basic = pro.stock_basic(list_status='L', fields='ts_code,symbol,name,area,industry,fullname,enname,market,exchange,list_date')
    except Exception as e:
        st.warning("stock_basic 拉取失败。部分板块/退市/ST 判断可能受影响。")
        df_stock_basic = pd.DataFrame()

    # 4) 多日全市场日线（用于计算 MA、10日涨幅等） - 按交易日范围拉
    #    使用 trade_dates 列表计算 start_date
    num_needed = lookback_days + 5
    if len(trade_dates) >= num_needed:
        start_date = trade_dates[-num_needed]
    else:
        start_date = trade_dates[0]
    start = start_date
    end = last_trade
    df_hist = pd.DataFrame()
    try:
        df_hist = pro.daily(start_date=start, end_date=end)
    except Exception as e:
        st.warning("拉历史多日 daily 失败，历史因子可能受限：" + str(e))
        df_hist = pd.DataFrame()

    # 5) moneyflow（当日主力资金流向）
    df_money = pd.DataFrame()
    try:
        df_money = pro.moneyflow(trade_date=last_trade)
    except Exception as e:
        st.warning("moneyflow 拉取失败：资金项将不可用或降级。")

    # 6) top_list（龙虎榜） - 可选
    df_top = pd.DataFrame()
    try:
        df_top = pro.top_list(trade_date=last_trade)
    except Exception as e:
        st.info("top_list 无法拉取或无权限：龙虎榜加分项将不可用。")

    # 7) limit_list（涨停信息）
    df_limit = pd.DataFrame()
    try:
        df_limit = pro.limit_list(trade_date=last_trade)
    except Exception as e:
        # 不是致命
        df_limit = pd.DataFrame()

    return {
        'daily': df_daily,
        'daily_basic': df_db,
        'stock_basic': df_stock_basic,
        'hist_daily': df_hist,
        'moneyflow': df_money,
        'top_list': df_top,
        'limit_list': df_limit,
        'trade_dates': trade_dates
    }

with st.spinner("批量拉取市场数据（一次性）..."):
    data = fetch_bulk_data(last_trade, lookback_days=15)
if not data:
    st.stop()

df_daily = data['daily']
df_db = data['daily_basic']
df_stock_basic = data['stock_basic']
df_hist = data['hist_daily']
df_money = data['moneyflow']
df_top = data['top_list']
df_limit = data['limit_list']
trade_dates = data['trade_dates']

# ---------------------------
# --- 基本数据存在性检查与提示（单位/字段可能因权限不同而变化）
# ---------------------------
st.write("当日记录总数（daily）：", len(df_daily))
if df_db is None or df_db.empty:
    st.warning("daily_basic 未能取得；请确保 2100 积分档有效。若 daily_basic 无法使用，流动性与换手率判断将被部分降级。")

# ---------------------------
# --- 合并基础信息：把 daily 与 daily_basic、stock_basic 合并在一起
# ---------------------------
df = df_daily.copy()
if not df_db.empty:
    # daily_basic 的 ts_code 索引可能是列
    if 'ts_code' in df_db.columns:
        df = df.merge(df_db[['ts_code', 'turnover_rate', 'circ_mv', 'amount']], on='ts_code', how='left', suffixes=('', '_db'))
    else:
        st.warning("daily_basic missing ts_code column: 跳过 merge")
if not df_stock_basic.empty:
    df = df.merge(df_stock_basic[['ts_code', 'name', 'industry', 'exchange', 'market', 'list_date']], on='ts_code', how='left')

# ---------------------------
# --- 板块/退市/ST/停牌排查
# ---------------------------
# ST 判定：尽量用 name 字段判断（若 stock_basic 可用）
def is_st(name):
    if not isinstance(name, str):
        return False
    return ('ST' in name.upper()) or ('☆' in name) or ('退' in name) or ('退市' in name)

df['is_st'] = df['name'].apply(lambda x: is_st(x) if 'name' in df.columns else False)
df['is_suspened'] = (df['vol'] == 0) | (df['amount'] == 0)
# 北交所判定：优先用 exchange/market 字段，否则按代码前缀回退
def is_bj(row):
    if 'exchange' in row and pd.notna(row['exchange']):
        return str(row['exchange']).lower() == 'bj'
    ts = row.get('ts_code', '')
    if isinstance(ts, str):
        return ts.startswith('4') or ts.startswith('8')
    return False

df['is_bj'] = df.apply(is_bj, axis=1)

# ---------------------------
# --- 基础过滤（第一层）
# ---------------------------
# 1) 排除 ST/退市/停牌/北交所
base_mask = (~df['is_st']) & (~df['is_suspened']) & (~df['is_bj'])

# 2) 价格范围
price_mask = (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)

# 3) 开盘不能太低
price_mask = price_mask & (df['open'] >= df['pre_close'] * OPEN_MIN_RATIO)

# 4) 连续下跌 N 天（使用 hist 数据）：构建最近 N 天是否收跌的判断
bad_continuous = set()
if (df_hist is not None) and (not df_hist.empty):
    # 最近交易日序列（按 trade_dates）
    recent_dates = trade_dates[-(CONTINUOUS_DOWN_DAYS+2):]  # 取略多些
    hist = df_hist[df_hist['trade_date'].isin(recent_dates)].copy()
    # pivot: ts_code x trade_date
    if not hist.empty:
        hist = hist.sort_values(['ts_code','trade_date'])
        # 对每只票判断最近 N 天是否均为收跌
        grouped = hist.groupby('ts_code')
        for ts, g in grouped:
            g = g.sort_values('trade_date')
            close = g['close'].tolist()
            # 取最后 CONTINUOUS_DOWN_DAYS 的相邻收盘，若长度不足就跳过
            if len(close) >= CONTINUOUS_DOWN_DAYS:
                last_n = close[-CONTINUOUS_DOWN_DAYS:]
                # 如果每天都低于前一天则视为连续下跌
                if all(last_n[i] < last_n[i-1] for i in range(1, len(last_n))):
                    bad_continuous.add(ts)
else:
    st.info("历史日线不完整：连续下跌的判断可能不准确。")

df['continuous_down'] = df['ts_code'].apply(lambda x: x in bad_continuous)
base_mask = base_mask & (~df['continuous_down'])

# 5) 10 日涨幅过大（过去 10 个交易日涨幅 >= RETURN_10D_MAX_PCT% 则排除）
# 计算 10 日涨幅（基于 hist）
ten_up_set = set()
if (df_hist is not None) and (not df_hist.empty):
    hist2 = df_hist.copy()
    hist2 = hist2.sort_values(['ts_code', 'trade_date'])
    tail = hist2.groupby('ts_code').tail(11)  # 包含起始日与结束日
    grouped = tail.groupby('ts_code')
    for ts, g in grouped:
        g = g.sort_values('trade_date')
        if len(g) >= 2:
            start_open = g.iloc[0]['open']
            end_close = g.iloc[-1]['close']
            if pd.notna(start_open) and start_open > 0:
                ret10 = (end_close / start_open - 1) * 100.0
                if ret10 >= RETURN_10D_MAX_PCT:
                    ten_up_set.add(ts)
df['ten_up_too_much'] = df['ts_code'].apply(lambda x: x in ten_up_set)
base_mask = base_mask & (~df['ten_up_too_much'])

# 6) 市值 / 换手 / 成交额 自适应过滤
# circ_mv 字段单位差异注意：尽量根据原始字段给出提示
if 'circ_mv' in df.columns:
    # circ_mv 单位在 tushare 通常是 万元或元，存在差异 —— 这里尝试推断：若 median 小于 1e6 则可能是万元单位
    med = df['circ_mv'].dropna().median() if df['circ_mv'].dropna().size>0 else 0
    unit_warn = ""
    if med > 0 and med < 1e6:
        # 可能 circ_mv 是万元单位，转换为元
        df['circ_mv_yuan'] = df['circ_mv'].astype(float) * 10000.0
    else:
        df['circ_mv_yuan'] = df['circ_mv'].astype(float)
    # 转换为亿单位
    df['circ_mv_yi'] = df['circ_mv_yuan'] / 1e8
    circ_mask = (df['circ_mv_yi'] >= MIN_CIRC_MV) & (df['circ_mv_yi'] <= MAX_CIRC_MV)
else:
    st.warning("daily_basic 中缺少 circ_mv 字段，市值过滤将无法使用，请检查权限或手动调整。")
    circ_mask = pd.Series([True] * len(df), index=df.index)

if 'turnover_rate' in df.columns:
    turnover_mask = df['turnover_rate'].astype(float) >= MIN_TURNOVER
else:
    st.warning("daily_basic 中缺少 turnover_rate 字段，换手率过滤将无法使用，请检查权限或手动调整。")
    turnover_mask = pd.Series([True] * len(df), index=df.index)

# 成交额阈值（动态）： amount_today >= circ_mv * (AMOUNT_PCT_OF_CIRC / 100)
if 'amount' in df.columns and 'circ_mv_yi' in df.columns:
    # amount 单位通常是 元；circ_mv_yi 是 亿。因此将 circ_mv_yi * 1e8 * pct
    df['amount'] = df['amount'].astype(float)
    df['amount_threshold'] = df['circ_mv_yi'] * 1e8 * (AMOUNT_PCT_OF_CIRC / 100.0)
    amount_mask = df['amount'] >= df['amount_threshold']
else:
    st.warning("daily 或 daily_basic 中缺少 amount 或 circ_mv 数据，成交额动态阈值将无法准确计算。")
    amount_mask = pd.Series([True] * len(df), index=df.index)

# 不涨停、不跌停（使用 limit_list 判断或用 pct_chg）
if (df_limit is not None) and (not df_limit.empty):
    limit_set = set(df_limit['ts_code'].unique())
    df['is_limited'] = df['ts_code'].apply(lambda x: x in limit_set)
    limit_mask = ~df['is_limited']
else:
    # fallback：用 pct_chg 接近限幅判断（绝对值 >= 9.5 表示涨停或跌停）
    limit_mask = df['pct_chg'].abs() < 9.5

# 价格区间同时检查
final_mask = base_mask & price_mask & circ_mask & turnover_mask & amount_mask & limit_mask

candidates = df[final_mask].copy()
st.write(f"第一轮筛选后候选数量：{len(candidates)}")

if candidates.empty:
    st.error("第一轮筛选没有候选。请放宽阈值或检查 daily_basic/circ_mv 等字段单位是否正确。")
    st.stop()

# ---------------------------
# --- 第二层：短期趋势 & 历史因子（一次性在 hist_daily 上处理）
# ---------------------------
# 以 df_hist 为基础计算 MA5/MA10、10日阳线占比、近3日最高价上移等
hist = df_hist.copy()
if hist.empty:
    st.warning("历史日线为空，无法计算短期趋势因子。将跳过第二层滤波。")
    candidates_trend = candidates.copy()
else:
    hist = hist.sort_values(['ts_code', 'trade_date'])
    # 计算 MA5、MA10、近10日阳线占比、近3日最高上移
    def compute_trend_factors(g):
        g = g.sort_values('trade_date').tail(20)  # 取最近 20 日足够计算
        res = {}
        closes = g['close'].astype(float).values
        opens = g['open'].astype(float).values
        highs = g['high'].astype(float).values
        lows = g['low'].astype(float).values
        if len(closes) >= 5:
            res['ma5'] = np.mean(closes[-5:])
        else:
            res['ma5'] = np.nan
        if len(closes) >= 10:
            res['ma10'] = np.mean(closes[-10:])
        else:
            res['ma10'] = np.nan
        # 10日涨幅
        if len(closes) >= 10:
            res['ret_10d_pct'] = (closes[-1] / closes[-10] - 1) * 100.0 if closes[-10] != 0 else np.nan
        else:
            res['ret_10d_pct'] = np.nan
        # 阳线占比（近10日）
        if len(closes) >= 10:
            up_days = sum(1 for i in range(-10, 0) if (closes[i] > opens[i]) )
            res['up_ratio_10d'] = up_days / 10.0
        else:
            res['up_ratio_10d'] = np.nan
        # 近3日最高价是否上移（last_high > prev_high）
        if len(highs) >= 4:
            res['three_high_up'] = (highs[-1] > highs[-2]) and (highs[-2] > highs[-3])
        else:
            res['three_high_up'] = False
        # 昨日上影线比（用于最后结构判断）
        if len(closes) >= 2:
            res['yesterday_upper_shadow_pct'] = (highs[-2] - closes[-2]) / closes[-2] * 100.0 if closes[-2] != 0 else np.nan
        else:
            res['yesterday_upper_shadow_pct'] = np.nan
        return pd.Series(res)

trend_feats = hist.groupby('ts_code').apply(compute_trend_factors).reset_index()
candidates_trend = candidates.merge(trend_feats, on='ts_code', how='left')

# 应用趋势过滤：
trend_mask = (
    (candidates_trend['ma5'] > candidates_trend['ma10']) &
    (candidates_trend['up_ratio_10d'] >= 0.40) &
    (candidates_trend['three_high_up'] == True)
)
# 对缺失值宽容：若 MA 缺失则保留（但会在评分里被弱化）
trend_mask = trend_mask.fillna(True)
candidates_trend = candidates_trend[trend_mask]
st.write(f"第二轮（趋势）后候选数量：{len(candidates_trend)}")

if candidates_trend.empty:
    st.warning("趋势层筛选后无候选，放宽趋势条件可能更适合短线。你可以调整 MA/阳线占比阈值。")

# ---------------------------
# --- 第三层：资金项（moneyflow/top_list）
# ---------------------------
cand = candidates_trend.copy()
# moneyflow 包含 net_mf_amount 等字段。若无则降级。
if (df_money is not None) and (not df_money.empty):
    # 合并当日 moneyflow（按 ts_code）
    if 'ts_code' in df_money.columns:
        mf_cols = [c for c in df_money.columns if c in ['ts_code','net_mf_amount','net_mf_vol','buy_sm_amount','sell_sm_amount','buy_lg_amount','sell_lg_amount']]
        mf_exist = [c for c in mf_cols if c in df_money.columns]
        cand = cand.merge(df_money[mf_exist], on='ts_code', how='left')
    else:
        st.info("moneyflow 无 ts_code 列，跳过资金合并。")
else:
    st.info("moneyflow 数据不可用，资金指标会降级。")

# 计算 5 日主力累计净流入（基于 hist 的 moneyflow，如不可用则为 NaN）
# 尝试一次性拉取近5日 moneyflow（若权限允许），否则跳过
try:
    recent_dates = trade_dates[-6:]
    mf_multi = pd.DataFrame()
    for td in recent_dates:
        try:
            dmf = pro.moneyflow(trade_date=td)
            if not dmf.empty:
                mf_multi = pd.concat([mf_multi, dmf], ignore_index=True)
        except Exception:
            pass
    if not mf_multi.empty:
        mf_5 = mf_multi.groupby('ts_code')['net_mf_amount'].sum().reset_index().rename(columns={'net_mf_amount':'net_mf_5d'})
        cand = cand.merge(mf_5, on='ts_code', how='left')
    else:
        cand['net_mf_5d'] = np.nan
except Exception:
    cand['net_mf_5d'] = np.nan

# top_list 加分项：是否在最近10日出现过龙虎榜
if (df_top is not None) and (not df_top.empty):
    top_set = set(df_top['ts_code'].unique())
    cand['top_recent'] = cand['ts_code'].apply(lambda x: x in top_set)
else:
    cand['top_recent'] = False

# ---------------------------
# --- 第四层：评分（向量化）
# ---------------------------
# 各类因子归一化/打分（简单线性归一化，保守权重）
def zscore_series(s):
    s = s.fillna(0)
    if s.std() == 0:
        return pd.Series([0]*len(s), index=s.index)
    return (s - s.mean()) / (s.std())

# 因子：短期趋势（ma5 - ma10）、主力 net_mf_5d、成交额相对于过去均值放大、行业景气（用 industry 聚合 moneyflow）
cand['trend_factor'] = (cand['ma5'] - cand['ma10']) / (cand['ma10'].replace(0, np.nan))
cand['mf5'] = cand['net_mf_5d'].fillna(0)
# 成交额放大：今日 amount vs hist 5 日平均（先计算 hist amount avg）
amount_avg = df_hist.groupby('ts_code')['amount'].mean().rename('amount_mean_hist').reset_index()
cand = cand.merge(amount_avg, on='ts_code', how='left')
cand['amount_mean_hist'] = cand['amount_mean_hist'].fillna(cand['amount'])
cand['amount_ratio'] = cand['amount'] / (cand['amount_mean_hist'].replace(0, np.nan))
cand['amount_ratio'] = cand['amount_ratio'].fillna(1.0)

# industry level money (行业景气度)
if ('industry' in df_stock_basic.columns) and (not df_money.empty):
    # industry aggregated net_mf_amount
    # 先把 moneyflow 与 stock_basic 合并，注意字段
    mf = df_money.copy()
    if 'ts_code' in mf.columns:
        mf = mf.merge(df_stock_basic[['ts_code','industry']], on='ts_code', how='left')
        ind_money = mf.groupby('industry')['net_mf_amount'].sum().rename('ind_net_mf').reset_index()
        cand = cand.merge(ind_money, on='industry', how='left')
        cand['ind_net_mf'] = cand['ind_net_mf'].fillna(0)
    else:
        cand['ind_net_mf'] = 0
else:
    cand['ind_net_mf'] = 0

# 标准化各因子
cand['f1'] = zscore_series(cand['trend_factor'].fillna(0))
cand['f2'] = zscore_series(cand['mf5'].fillna(0))
cand['f3'] = zscore_series(cand['amount_ratio'].fillna(1.0))
cand['f4'] = zscore_series(cand['ind_net_mf'].fillna(0))
# k线结构（昨天上影线越小越好 -> 转换成正向分数）
cand['f5'] = - zscore_series(cand['yesterday_upper_shadow_pct'].fillna(0))

# 合成评分（权重可调）
cand['score'] = (
    0.28 * cand['f1'].fillna(0) +
    0.25 * cand['f2'].fillna(0) +
    0.20 * cand['f3'].fillna(0) +
    0.15 * cand['f4'].fillna(0) +
    0.12 * cand['f5'].fillna(0)
)

# 排序并显示
final = cand.sort_values('score', ascending=False).reset_index(drop=True)
final.index += 1

st.success(f"选股完成：共找到 {len(final)} 支候选（按综合评分降序）。显示前 {min(len(final), TOP_N)} 支")
# 显示关键列
display_cols = ['name','ts_code','score','close','pct_chg','turnover_rate','circ_mv_yi','amount','amount_ratio','net_mf_5d','top_recent']
display_cols = [c for c in display_cols if c in final.columns]
st.dataframe(final[display_cols].head(TOP_N), use_container_width=True)

# 提供 CSV 下载
csv = final.to_csv(index=True, encoding='utf-8-sig')
st.download_button("下载候选 CSV（含综合评分）", data=csv, file_name=f"shortlist_{last_trade}.csv", mime="text/csv")

# 运行结束提示
st.caption("说明：若结果看起来不合理，请把界面顶部的 warnings/infos 的提示内容发给我（例如 circ_mv 单位或 daily_basic 字段缺失），我会帮你调整字段单位/阈值。")
