# -*- coding: utf-8 -*-
"""
翻倍黑马归类统计器 · MACD波浪动能版 (逐浪拆解版)
------------------------------------------------
采用"周线 MACD 红绿柱交替（由红转绿）"作为客观识别波浪的硬核工具

【★本版新增】
1. 逐浪明细：每只票的每一段上涨浪，单独记录"第几浪、起点价、终点价、本浪涨幅%"，
   不再只有汇总的"洗盘次数/平均跌幅"这种统计量，可以直接回答"主升浪常出现在第几浪"这个问题。
2. 未走完的最后一浪会被标记为"进行中"：如果一只票最新一根周K线仍然是红柱(还没变绿)，
   说明它最后这段上涨还没走完、还没见顶，这一浪的涨幅是不完整的——参与"第几浪涨幅最大"的
   统计时会被排除，避免用一个"还没涨到头"的半截浪拉低或拉偏某一浪序号的平均涨幅，
   但仍然会在明细表里展示出来，方便你观察"哪些票正处在被认为可能即将进入主升浪的位置"。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings("ignore")

# ---------------------------
# 1. 页面配置与侧边栏 UI
# ---------------------------
st.set_page_config(page_title="50周翻倍股票深度测算器 (逐浪拆解版)", layout="wide")

with st.sidebar:
    st.header("⚙️ 参数设置")
    TS_TOKEN = st.text_input("🔑 Tushare Token", type="password")
    MIN_PRICE = st.number_input("最低股价限制 (元)", value=20.0)
    MIN_MV = st.number_input("最小市值 (亿元)", value=100.0)
    MAX_MV = st.number_input("最大市值 (亿元)", value=1000.0)

st.title("📊 近50周翻倍股票测算 (MACD动能·逐浪拆解版)")

if not TS_TOKEN:
    st.info("👈 请先在左侧边栏输入您的 Tushare Token 以启动程序。")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 2. 数据获取与合成
# ---------------------------
@st.cache_data(ttl=3600*12)
def safe_get(_pro, func_name, **kwargs):
    if _pro is None: return pd.DataFrame()
    func = getattr(_pro, func_name)
    try:
        for _ in range(3):
            df = func(**kwargs)
            if df is not None and not df.empty: return df
            time.sleep(0.3)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def get_stock_weekly_data(_pro, ts_code, start_date, end_date):
    df = safe_get(_pro, 'daily', ts_code=ts_code, start_date=start_date, end_date=end_date)
    adj = safe_get(_pro, 'adj_factor', ts_code=ts_code, start_date=start_date, end_date=end_date)

    if df.empty or adj.empty: return pd.DataFrame()

    df = df.merge(adj, on=['ts_code', 'trade_date'], how='left').sort_values('trade_date')
    latest_factor = df['adj_factor'].iloc[-1]

    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] * df['adj_factor'] / latest_factor

    # 日线 MACD 计算，用于合成周线 MACD 状态
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2

    df['dt'] = pd.to_datetime(df['trade_date'])
    iso_cal = df['dt'].dt.isocalendar()
    df['year_week'] = iso_cal.year.astype(str) + "_" + iso_cal.week.astype(str).str.zfill(2)

    # 周线合成：取每周最后一天的数据作为周收盘/MACD状态
    weekly = df.groupby('year_week', as_index=False).agg({
        'trade_date': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'macd': 'last'  # 采用周末最后一天的 MACD 柱子状态
    }).sort_values('trade_date').reset_index(drop=True)

    return weekly

# ---------------------------
# 3. 形态判断 + 逐浪拆解核心算法
# ---------------------------
def classify_double_pattern(weekly_df):
    """
    返回 (summary_dict, waves_list)
    summary_dict：和之前一样的汇总统计（形态归类、洗盘次数、跌幅等）
    waves_list：新增的逐浪明细，每个元素是一段上涨浪的 {wave_no, start_price, end_price, gain_pct, is_ongoing}
    """
    if len(weekly_df) < 10: return None, None

    min_idx = weekly_df['low'].idxmin()
    sub_df = weekly_df.loc[min_idx:]
    if len(sub_df) < 2: return None, None

    max_idx = sub_df['high'].idxmax()
    min_price = weekly_df.loc[min_idx, 'low']
    max_price = sub_df.loc[max_idx, 'high']

    if min_price <= 0 or (max_price / min_price) < 2.0:
        return None, None

    weeks_taken = max_idx - min_idx + 1
    ascent_df = weekly_df.loc[min_idx:max_idx].copy().reset_index(drop=True)

    # 【修复】原来要求"最新一周必须恰好是整个窗口的历史最高点"才算"进行中"，
    # 这个条件太苛刻——只要历史最高点不是精确落在最新一周（哪怕只差一点点，或者
    # 当前周K线还没走完），就永远判断为False，导致"进行中"几乎不可能被触发。
    # 现在改成：不要求创新高，只看最新一根周K线的MACD是否仍是红柱——只要还是红柱，
    # 就说明这段行情(不管有没有突破前高)还没走完，值得纳入"进行中"观察池。
    last_macd = weekly_df['macd'].iloc[-1]
    is_final_wave_ongoing = bool(last_macd >= 0)

    running_max = ascent_df['high'].iloc[0]
    in_pullback = False
    pullback_trough_price = None

    drawdowns = []
    current_max_drawdown = 0.0

    # 【新增】逐浪拆解所需的状态
    waves = []
    wave_start_price = ascent_df['low'].iloc[0]  # 第1浪起点＝整个区间的最低价

    for i in range(1, len(ascent_df)):
        curr_high = ascent_df.loc[i, 'high']
        curr_low = ascent_df.loc[i, 'low']
        curr_macd = ascent_df.loc[i, 'macd']

        if curr_high > running_max:
            if in_pullback:
                # 一次洗盘正式结束（创出新高）——把此前那段上涨记为完整的一浪
                drawdowns.append(current_max_drawdown)
                waves.append({
                    'start_price': wave_start_price,
                    'end_price': running_max,
                    'gain_pct': (running_max - wave_start_price) / wave_start_price * 100,
                    'is_ongoing': False
                })
                # 下一浪从这次洗盘的最低点开始起算
                wave_start_price = pullback_trough_price
                in_pullback = False
                current_max_drawdown = 0.0
            running_max = curr_high
        else:
            drawdown = (running_max - curr_low) / running_max
            # 只要周线 MACD 翻绿（< 0），且确实发生了空间回撤（>=5%），就确认为一次有效洗盘
            if curr_macd < 0 and drawdown >= 0.05:
                if (not in_pullback) or (pullback_trough_price is None) or (curr_low < pullback_trough_price):
                    pullback_trough_price = curr_low
                in_pullback = True
                if drawdown > current_max_drawdown:
                    current_max_drawdown = drawdown

    if in_pullback and current_max_drawdown > 0:
        drawdowns.append(current_max_drawdown)

    # 【新增】收尾：把最后一段（可能是完整浪，也可能是还没走完的浪）也记录进 waves
    final_gain = (running_max - wave_start_price) / wave_start_price * 100
    waves.append({
        'start_price': wave_start_price,
        'end_price': running_max,
        'gain_pct': final_gain,
        'is_ongoing': is_final_wave_ongoing
    })
    for w_no, w in enumerate(waves, start=1):
        w['wave_no'] = w_no

    # 如果没有检测到清晰的 MACD 绿柱交替，或者耗时太短，归为爆破型
    if weeks_taken <= 6 or len(drawdowns) <= 1:
        pattern = "⚡ 潜伏爆破型 (形态一)"
    else:
        pattern = "🌊 波浪推进型 (形态二)"

    avg_drop = (sum(drawdowns) / len(drawdowns) * 100) if drawdowns else 0.0
    max_drop = (max(drawdowns) * 100) if drawdowns else 0.0

    # 【新增】"进行中"只代表最新一周仍是红柱，不代表已经创了新高——
    # 补充"最新收盘价距历史最高点还差多少"，让你能分清两种情况：
    # 距离≤0(甚至为负)＝已经站上前高、正在创新高的强势票；距离>0＝还没突破前高，
    # 可能是刚从洗盘里爬升、正在挑战前高的途中，两种都值得关注，但含义不同。
    last_close = weekly_df['close'].iloc[-1]
    dist_from_peak_pct = (max_price - last_close) / max_price * 100 if max_price > 0 else np.nan

    summary = {
        'Min_Price': round(min_price, 2),
        'Max_Price': round(max_price, 2),
        'Max_Gain (%)': round((max_price - min_price) / min_price * 100, 1),
        'Weeks_Taken': weeks_taken,
        'Pullback_Count': len(drawdowns),
        'Avg_Drop (%)': round(avg_drop, 2),
        'Max_Drop (%)': round(max_drop, 2),
        'Pattern': pattern,
        'Total_Waves': len(waves),
        'Final_Wave_Ongoing': is_final_wave_ongoing,
        'Last_Close': round(last_close, 2),
        'Dist_From_Peak (%)': round(dist_from_peak_pct, 2)
    }

    return summary, waves

# ---------------------------
# 4. 主干回测与展示逻辑
# ---------------------------
if st.button("🚀 开始 MACD 动能·逐浪拆解测算"):
    today_str = datetime.now().strftime("%Y%m%d")
    start_date_str = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    with st.spinner("正在获取基础股票列表..."):
        basic = safe_get(pro, 'stock_basic', list_status='L', fields='ts_code,name,market')
        daily_basic = safe_get(pro, 'daily_basic', trade_date=today_str)
        if daily_basic.empty:
            for d in range(1, 10):
                prev_date = (datetime.now() - timedelta(days=d)).strftime("%Y%m%d")
                daily_basic = safe_get(pro, 'daily_basic', trade_date=prev_date)
                if not daily_basic.empty: break

    if basic.empty or daily_basic.empty:
        st.error("无法获取股票基础数据，请检查 Token 或网络状态。")
        st.stop()

    df_merged = basic.merge(daily_basic[['ts_code', 'close', 'circ_mv']], on='ts_code', how='inner')
    df_merged['circ_mv_billion'] = df_merged['circ_mv'] / 10000

    filtered_stocks = df_merged[
        (~df_merged['name'].str.contains('ST|退', na=False)) &
        (df_merged['close'] >= MIN_PRICE) &
        (df_merged['circ_mv_billion'] >= MIN_MV) &
        (df_merged['circ_mv_billion'] <= MAX_MV)
    ]

    st.write(f"🔍 符合过滤条件（股价≥{MIN_PRICE}元，市值{MIN_MV}-{MAX_MV}亿）的股票共 **{len(filtered_stocks)}** 只，开始 MACD 动能·逐浪拆解扫描...")

    summary_results = []
    wave_results = []  # 【新增】逐浪明细的长表
    progress_bar = st.progress(0)

    for idx, row in enumerate(filtered_stocks.itertuples()):
        weekly = get_stock_weekly_data(pro, row.ts_code, start_date_str, today_str)
        if not weekly.empty:
            summary, waves = classify_double_pattern(weekly)
            if summary:
                summary['ts_code'] = row.ts_code
                summary['name'] = row.name
                summary_results.append(summary)

                for w in waves:
                    wave_results.append({
                        'ts_code': row.ts_code,
                        'name': row.name,
                        '浪序号': w['wave_no'],
                        '起点价': round(w['start_price'], 2),
                        '终点价': round(w['end_price'], 2),
                        '本浪涨幅(%)': round(w['gain_pct'], 1),
                        '是否进行中': '是(未走完)' if w['is_ongoing'] else '否(已走完)'
                    })

        progress_bar.progress((idx + 1) / len(filtered_stocks))
    progress_bar.empty()

    if summary_results:
        res_df = pd.DataFrame(summary_results)
        wave_df = pd.DataFrame(wave_results)

        p1_count = len(res_df[res_df['Pattern'].str.contains('潜伏爆破')])
        p2_count = len(res_df[res_df['Pattern'].str.contains('波浪推进')])
        total = len(res_df)

        st.markdown("---")
        st.header("📈 统计结果汇总 (MACD 动能波浪版)")

        c1, c2, c3 = st.columns(3)
        c1.metric("近50周翻倍股票总数", f"{total} 只")
        c2.metric("⚡ 潜伏爆破型 (形态一)", f"{p1_count} 只", f"{p1_count/total*100:.1f}%")
        c3.metric("🌊 波浪推进型 (形态二)", f"{p2_count} 只", f"{p2_count/total*100:.1f}%")

        st.subheader("📋 汇总统计明细")
        display_df = res_df[['ts_code', 'name', 'Pattern', 'Weeks_Taken', 'Pullback_Count',
                             'Avg_Drop (%)', 'Max_Drop (%)', 'Max_Gain (%)', 'Min_Price', 'Max_Price',
                             'Total_Waves', 'Final_Wave_Ongoing', 'Last_Close', 'Dist_From_Peak (%)']]
        display_df.columns = ['代码', '名称', '归类形态', '翻倍耗时(周)', 'MACD洗盘次数',
                              '平均每次洗盘跌幅(%)', '极限单次洗盘跌幅(%)', '最大总涨幅(%)', '区间最低价',
                              '区间最高价', '总浪数', '最后一浪是否进行中', '最新收盘价', '距历史高点距离(%)']
        st.dataframe(display_df.sort_values('翻倍耗时(周)').reset_index(drop=True), use_container_width=True)

        csv_summary = display_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载汇总统计明细 (CSV)", csv_summary, "double_stocks_summary.csv", "text/csv")

        # ---------------------------
        # 【★新增核心板块】逐浪明细 + "第几浪涨幅最大"统计
        # ---------------------------
        st.markdown("---")
        st.header("🌊 逐浪拆解：每一浪究竟涨了多少？")

        st.subheader("📋 逐浪涨幅明细（长表，一行一浪）")
        st.dataframe(wave_df.sort_values(['ts_code', '浪序号']).reset_index(drop=True), use_container_width=True)

        csv_wave = wave_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载逐浪涨幅明细 (CSV)", csv_wave, "double_stocks_wave_breakdown.csv", "text/csv")

        # 只用"已走完"的浪来做"第几浪平均涨幅最大"的统计，进行中的半截浪不计入，避免拉偏
        completed_waves = wave_df[wave_df['是否进行中'] == '否(已走完)'].copy()

        if not completed_waves.empty:
            st.subheader("📊 分浪序号平均涨幅（只统计已走完的浪）")
            wave_avg = completed_waves.groupby('浪序号')['本浪涨幅(%)'].agg(['mean', 'count']).reset_index()
            wave_avg.columns = ['浪序号', '平均涨幅(%)', '样本数']
            wave_avg['平均涨幅(%)'] = wave_avg['平均涨幅(%)'].round(1)
            st.dataframe(wave_avg, use_container_width=True)
            st.bar_chart(wave_avg.set_index('浪序号')['平均涨幅(%)'])

            st.subheader("🎯 每只票「涨幅最大的一浪」分布在第几浪")
            # 每只票，在已走完的浪里，找出涨幅最大的那一浪是第几浪
            idx_max = completed_waves.groupby('ts_code')['本浪涨幅(%)'].idxmax()
            best_wave_per_stock = completed_waves.loc[idx_max]
            best_wave_dist = best_wave_per_stock['浪序号'].value_counts().sort_index().reset_index()
            best_wave_dist.columns = ['浪序号', '成为主升浪的次数']
            best_wave_dist['占比(%)'] = round(best_wave_dist['成为主升浪的次数'] / best_wave_dist['成为主升浪的次数'].sum() * 100, 1)
            st.dataframe(best_wave_dist, use_container_width=True)
            st.bar_chart(best_wave_dist.set_index('浪序号')['成为主升浪的次数'])

            st.caption("💡 这张表直接回答你的问题：主升浪最常出现在第几浪。数值越高，说明处在那个浪序号位置的票，"
                       "越有可能正准备走出最大的一段涨幅——但这里只统计了「已经走完」的浪，正在进行中的最后一浪不计入，"
                       "避免用一段还没涨到头的行情拉偏某个浪序号的统计。")
        else:
            st.info("暂无已走完的浪数据可供统计。")

        # 单独展示"最后一浪仍在进行中"的股票——这些是你最关心的、可能正准备走主升浪的候选
        ongoing_stocks = wave_df[wave_df['是否进行中'] == '是(未走完)'].copy()
        if not ongoing_stocks.empty:
            st.markdown("---")
            st.subheader("🚀 当前正处于「进行中」最后一浪的股票（候选观察池）")
            st.caption("这些票最新一根周K线仍是红柱，说明当前这一浪还没走完/还没见顶。"
                       "注意「距历史高点距离(%)」这一列：≤0表示已经站上前高、正在创新高，属于强势延续；"
                       ">0表示还没突破前高，可能是刚从洗盘里爬升、正在挑战前高的半路上——这一浪的涨幅数字"
                       "在后一种情况下还没算完整，仅供参考。结合上面「主升浪常见于第几浪」的统计，"
                       "如果这些票正处在容易成为主升浪的浪序号位置，可以重点关注。")
            ongoing_display = ongoing_stocks.merge(
                res_df[['ts_code', 'Total_Waves', 'Dist_From_Peak (%)']], on='ts_code', how='left'
            )
            st.dataframe(ongoing_display.sort_values('本浪涨幅(%)', ascending=False).reset_index(drop=True),
                        use_container_width=True)
    else:
        st.warning("⚠️ 在当前筛选条件下，近 50 周内没有扫描到符合条件的股票。")
