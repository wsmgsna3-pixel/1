# -*- coding: utf-8 -*-
import streamlit as st
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="选股王 · 干净候选池（增强版）", layout="wide")
st.title("选股王 · 干净候选池（增强版，120 积分友好）")

# -----------------------------
# 运行时输入 Token（界面输入）
# -----------------------------
TS_TOKEN = st.text_input("请输入你的 Tushare Token（仅本次使用，不会保存）", type="password")
if not TS_TOKEN:
    st.info("请输入 Token 后才能运行选股")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# -----------------------------
# 参数（可改）
# -----------------------------
INITIAL_TOP_N = 1000   # 从涨幅榜取前 N 名作为初始池（可以改小以省时间/积分）
FINAL_POOL = 300       # 过滤后取前多少作为进入评分池
MIN_PRICE = 10         # 排除低于此价格（元）
MAX_PRICE = 200        # 排除高于此价格（元）
MIN_TURNOVER = 3.0     # 换手率阈值（%）
MIN_VOL_RATIO = 1.5    # 在日换手不可用时，用成交量相比昨日的倍数作为近似（vol_today > vol_yesterday * MIN_VOL_RATIO）
OPEN_MULTIPLIER = 0.3  # 开盘与昨日最高价比率（0.3）
VOL_MULTIPLIER = 1.5   # 今天成交量需大于昨天成交量 × VOL_MULTIPLIER（评分条件）
# -----------------------------

# -----------------------------
# helper: 最近交易日（简单回退，不依赖 trade_cal）
# -----------------------------
def get_last_trade_day():
    today = datetime.now()
    # 周六 -> 周五，周日 -> 周五，平日 -> 昨天
    if today.weekday() == 5:
        d = today - timedelta(days=1)
    elif today.weekday() == 6:
        d = today - timedelta(days=2)
    else:
        d = today - timedelta(days=1)
    return d.strftime("%Y%m%d")

last_trade = get_last_trade_day()
st.info(f"使用参考交易日（最近交易日）：{last_trade}")

# -----------------------------
# 拉取当天日线（所有交易记录）
# -----------------------------
@st.cache_data(ttl=60)
def get_today_data(trade_date):
    try:
        df = pro.daily(trade_date=trade_date)
        return df
    except Exception as e:
        return pd.DataFrame()

# -----------------------------
# 拉取最近 10 根 K 线（用于历史判断）
# -----------------------------
@st.cache_data(ttl=600)
def fetch_10d(ts_code):
    """
    返回 DataFrame（按 trade_date 升序），或 None。
    """
    try:
        end = last_trade
        start = (datetime.strptime(end, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or len(df) < 5:
            return None
        df = df.sort_values("trade_date")
        return df
    except Exception:
        return None

# -----------------------------
# 尝试获取股票名称（批量），若失败回退
# -----------------------------
def try_get_names(ts_codes):
    """
    尝试通过 pro.stock_basic 获取 ts_code->name 映射。
    若权限不足或失败，返回空字典并在界面给出告警。
    """
    try:
        # 尝试一次性获取全部基本信息（按权限）
        info = pro.stock_basic(list_status='L', fields='ts_code,name')
        info = info.drop_duplicates(subset=['ts_code'])
        info = info.set_index('ts_code')['name'].to_dict()
        return {code: info.get(code, "") for code in ts_codes}
    except Exception as e:
        st.warning("注意：无法通过 pro.stock_basic 获取股票名称（可能是权限问题）。界面将显示代码代替名称。")
        return {}

# -----------------------------
# 尝试获取换手率数据（daily_basic），若失败回退到体量近似
# -----------------------------
def try_get_daily_basic(trade_date):
    """
    返回 DataFrame indexed by ts_code with columns including turnover_rate if possible.
    若失败返回 None（脚本会用体量近似规则替代换手率过滤）。
    """
    try:
        db = pro.daily_basic(trade_date=trade_date, fields='ts_code,turnover_rate')
        db = db.set_index('ts_code')
        return db
    except Exception:
        st.warning("注意：无法获取 daily_basic（换手率），将用成交量放大近似替代换手率过滤。若你有更高权限可解除此降级。")
        return None

# -----------------------------
# 判定一字板（基于单日数据）
# -----------------------------
def is_one_word_row(r):
    try:
        return (r['open'] == r['high']) and (r['high'] == r['low']) and (r['low'] == r['pre_close'])
    except Exception:
        return False

# -----------------------------
# 主要筛选流程
# -----------------------------
if st.button("一键生成干净候选池并评分"):
    with st.spinner("获取当天行情..."):
        today_df = get_today_data(last_trade)

    if today_df is None or today_df.empty:
        st.error("未能获取当天行情数据。请检查 Token 权限或网络。")
        st.stop()

    st.write(f"当日行情总记录数：{len(today_df)}（后续将从涨幅榜前 {INITIAL_TOP_N} 进行清洗）")

    # 先按涨幅取前 INITIAL_TOP_N 作为初始池（这一步轻量）
    pool = today_df.sort_values('pct_chg', ascending=False).head(INITIAL_TOP_N).copy()
    pool = pool.reset_index(drop=True)

    # 尝试获取名称 & daily_basic（换手率）
    names_map = try_get_names(pool['ts_code'].tolist())
    daily_basic_df = try_get_daily_basic(last_trade)  # may be None

    cleaned = []
    approx_turnover_fallback = daily_basic_df is None

    # 遍历池并做清洗（只用少数字段，尽量减少接口）
    progress = st.progress(0)
    for i, row in pool.iterrows():
        ts = row['ts_code']
        # 基础过滤（基于当日这条记录）
        # 1) 排除停牌（vol==0 或 amount==0）
        vol = row.get('vol', 0)
        amount = row.get('amount', 0)
        if (vol is None) or (vol == 0) or (amount is None) or (amount == 0):
            progress.progress((i+1)/len(pool))
            continue

        # 2) 排除价格区间（< MIN_PRICE or > MAX_PRICE）
        price = row.get('close', row.get('open', None))
        if price is None:
            progress.progress((i+1)/len(pool))
            continue
        if price < MIN_PRICE or price > MAX_PRICE:
            progress.progress((i+1)/len(pool))
            continue

        # 3) 排除一字板（基于当日）
        if is_one_word_row(row):
            progress.progress((i+1)/len(pool))
            continue

        # 4) 排除昨日下跌（即这条记录的 pct_chg < 0）
        if row.get('pct_chg', 0) < 0:
            progress.progress((i+1)/len(pool))
            continue

        # 5) 换手率判定：如果 daily_basic 可用，则直接判断；否则用体量近似（vol vs yesterday）
        turnover_ok = False
        if daily_basic_df is not None and ts in daily_basic_df.index:
            try:
                tr = float(daily_basic_df.loc[ts, 'turnover_rate'])
                if tr >= MIN_TURNOVER:
                    turnover_ok = True
            except Exception:
                turnover_ok = False
        else:
            # 近似：先拿最近 10 日看昨天的 vol 和过去的最大 vol 判断（需要历史）
            hist = fetch_10d(ts)
            if hist is None or len(hist) < 3:
                # 没历史数据：跳过这个票以防止风险
                progress.progress((i+1)/len(pool))
                continue
            # hist 按升序排列；最后一条是最近交易日
            vol_list = hist['vol'].astype(float).tolist()
            # 当日（参考日）成交量
            vol_today = vol_list[-1]
            vol_yesterday = vol_list[-2] if len(vol_list) >= 2 else 0
            # 排除昨日爆量：如果昨天（vol_yesterday）是 10 日的最大量，则视为爆量，跳过
            if vol_yesterday >= max(vol_list[:-1]):  # exclude the latest day itself
                progress.progress((i+1)/len(pool))
                continue
            # 近似换手判断（今天成交量 vs 昨日）
            if vol_today > vol_yesterday * MIN_VOL_RATIO:
                turnover_ok = True
            else:
                turnover_ok = False

        if not turnover_ok:
            progress.progress((i+1)/len(pool))
            continue

        # 6) 排除 ST —— 需要名称支持；若无法拿到名称（权限不足），则跳过这项并在顶部提示用户
        name = names_map.get(ts, "")
        if name:
            if "ST" in name.upper() or "退" in name:  # 简单匹配 ST/退等特殊标识
                progress.progress((i+1)/len(pool))
                continue
        # 如果 name 为空（权限不足），我们保留候选并在顶部提示（之前已提示）

        # 7) 排除昨日爆量：如果 daily_basic 不可用，我们已经在近似里判断过；如果 daily_basic 可用，这里还可以用历史判定
        #    为统一性，这里用历史函数再确认一次（不会产生额外接口，因为 fetch_10d 缓存）
        hist2 = fetch_10d(ts)
        if hist2 is None or len(hist2) < 3:
            progress.progress((i+1)/len(pool))
            continue
        vol_hist = hist2['vol'].astype(float).tolist()
        # 昨天（最后一条）之前的那天是昨天
        vol_yesterday = vol_hist[-2]
        # 如果 vol_yesterday 是过去 10 日的最大，则视为爆量 => 跳过
        if vol_yesterday >= max(vol_hist[:-1]):
            progress.progress((i+1)/len(pool))
            continue

        # 通过所有清洗检查，保留
        cleaned.append(row)
        progress.progress((i+1)/len(pool))

    progress.progress(1.0)
    st.write(f"清洗后候选数量：{len(cleaned)}（将从中取涨幅前 {FINAL_POOL} 进入评分阶段）")

    if len(cleaned) == 0:
        st.error("清洗后没有候选股票。建议放宽过滤条件（例如减小 MIN_TURNOVER / 放宽价格区间 / 增加初筛数量）。")
        st.stop()

    # 转为 DataFrame 并按 pct_chg 再取前 FINAL_POOL
    cleaned_df = pd.DataFrame(cleaned).sort_values('pct_chg', ascending=False).head(FINAL_POOL).reset_index(drop=True)
    st.write(f"用于评分的池子大小：{len(cleaned_df)}")

    # -----------------------------
    # 在 FINAL_POOL 里运行原始评分逻辑（更严格、逐股拉历史并评分）
    # -----------------------------
    result = []
    progress2 = st.progress(0)
    for i, r in enumerate(cleaned_df.iterrows()):
        idx, row = r
        ts = row['ts_code']
        # 尝试拿历史（缓存命中）
        hist = fetch_10d(ts)
        if hist is None or len(hist) < 10:
            progress2.progress((i+1)/len(cleaned_df))
            continue

        # 计算历史指标（与之前约定保持一致）
        hist_tail = hist.tail(10).reset_index(drop=True)
        ten_return = hist_tail.iloc[-1]['close'] / hist_tail.iloc[0]['open'] - 1
        vol_yesterday = hist_tail.iloc[-2]['vol']
        high_yesterday = hist_tail.iloc[-2]['high']
        avg_turnover_proxy = abs(hist_tail['pct_chg']).mean()  # 近似换手/活跃度

        # 评分条件（严格版）
        try:
            cond1 = row['open'] > 10
            cond2 = ten_return <= 0.50
            cond3 = avg_turnover_proxy >= 3  # 这里用 pct_chg 绝对均值当作近似换手率门槛
            cond4 = row['vol'] > vol_yesterday * VOL_MULTIPLIER
            cond5 = row['open'] >= high_yesterday * OPEN_MULTIPLIER
        except Exception:
            progress2.progress((i+1)/len(cleaned_df))
            continue

        if cond1 and cond2 and cond3 and cond4 and cond5:
            score = (row['open'] - row['pre_close']) / row['pre_close'] * 100 + (row['vol'] / (vol_yesterday+1)) * 10
            result.append({
                'ts_code': ts,
                'name': names_map.get(ts, ts),  # 若没有名称则显示代码
                '综合评分': round(score, 2),
                'pct_chg': row.get('pct_chg'),
                'open': row.get('open'),
                'close': row.get('close'),
                'volume_today': row.get('vol'),
                'volume_yesterday': vol_yesterday,
                '10d_return': round(ten_return, 4),
                '10d_avg_turnover_proxy': round(avg_turnover_proxy, 3)
            })
        progress2.progress((i+1)/len(cleaned_df))

    progress2.progress(1.0)

    if not result:
        st.warning("首次严格评分没有选出符合条件的股票，脚本将自动放宽成交量 & 开盘放宽条件并重试。")
        # 放宽条件重试：成交量 multiplier -> 1.0，开盘放宽 -> >= 昨日收盘价
        fallback_result = []
        progress3 = st.progress(0)
        for i, r in enumerate(cleaned_df.iterrows()):
            idx, row = r
            ts = row['ts_code']
            hist = fetch_10d(ts)
            if hist is None or len(hist) < 5:
                progress3.progress((i+1)/len(cleaned_df))
                continue
            hist_tail = hist.tail(10).reset_index(drop=True)
            vol_yesterday = hist_tail.iloc[-2]['vol']
            ten_return = hist_tail.iloc[-1]['close'] / hist_tail.iloc[0]['open'] - 1
            avg_turn = abs(hist_tail['pct_chg']).mean()
            try:
                cond1 = row['open'] > 10
                cond2 = ten_return <= 0.50
                cond3 = avg_turn >= 2  # 放宽
                cond4 = row['vol'] > vol_yesterday * 1.0
                cond5 = row['open'] >= row['pre_close']  # 放宽
            except Exception:
                progress3.progress((i+1)/len(cleaned_df))
                continue

            if cond1 and cond2 and cond3 and cond4 and cond5:
                score = (row['open'] - row['pre_close']) / row['pre_close'] * 100 + (row['vol'] / (vol_yesterday+1)) * 10
                fallback_result.append({
                    'ts_code': ts,
                    'name': names_map.get(ts, ts),
                    '综合评分': round(score,2),
                    'pct_chg': row.get('pct_chg'),
                    'open': row.get('open'),
                    'close': row.get('close'),
                    'volume_today': row.get('vol'),
                    'volume_yesterday': vol_yesterday,
                    '10d_return': round(ten_return, 4),
                    '10d_avg_turnover_proxy': round(avg_turn, 3)
                })
            progress3.progress((i+1)/len(cleaned_df))
        progress3.progress(1.0)
        final_list = fallback_result
    else:
        final_list = result

    if not final_list:
        st.error("即使放宽条件，也没有选出股票。建议进一步放宽过滤或检查 Token 权限。")
        st.stop()

    # DataFrame & 排序
    final_df = pd.DataFrame(final_list).sort_values('综合评分', ascending=False).reset_index(drop=True)
    final_df.index += 1  # 方便显示从1开始
    st.success(f"选股完成：共找到 {len(final_df)} 支候选（按综合评分降序）")

    # 显示前 20 用于快速查看
    st.dataframe(final_df.head(50), use_container_width=True)

    # 提供 CSV 下载
    csv = final_df.to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载候选 CSV（含综合评分）", data=csv, file_name=f"shortlist_{last_trade}.csv", mime="text/csv")
