# -*- coding: utf-8 -*-
"""周线SKDJ信号有效性验证器（不是策略，不做选股，只测信号本身有没有优势）。

用途：在往上搭任何选股规则之前，先回答一个问题——
    "周线SKDJ的K上穿25之后，股价表现真的比随便买一只科技股更好吗？"

核心方法：超额收益（edge），而不是绝对收益。
    edge = 信号样本的未来N周平均收益 - 同期全池所有股票的未来N周平均收益
牛市里随便买什么都涨，只看绝对收益会把大盘的贡献误认成信号的功劳。
只有相对同期基准仍然领先，才说明信号本身携带信息。

买卖口径与主程序保持一致：信号周收盘产生信号，下一周开盘买入，
持有N周后按当周收盘卖出。不含交易成本（验证信号用，成本在策略层再算）。

运行方式与主程序相同：
    streamlit run validate_skdj_edge.py
本脚本复用主程序的行情缓存目录，已下载过的数据不会重复下载。
"""

from __future__ import annotations

import importlib
import io
import math
import zipfile
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 复用主程序的数据层（复权处理、缓存、股票池口径完全一致，避免口径差异干扰结论）
# -----------------------------------------------------------------------------
CANDIDATE_MODULE_NAMES = (
    "app",
    "app_r24_fix_branch_label_and_r15_gate",
    "app_r23_r3_r6_r15_overhaul",
    "app_r22_fix_export_scaling",
)

_engine = None
_engine_name = ""
for _name in CANDIDATE_MODULE_NAMES:
    try:
        _engine = importlib.import_module(_name)
        _engine_name = _name
        break
    except ImportError:
        continue

if _engine is None:
    st.error(
        "找不到主程序模块。请把本脚本和主程序 app.py 放在同一个目录下，"
        "或修改本文件顶部的 CANDIDATE_MODULE_NAMES 填入你的主程序文件名（不含.py）。"
    )
    st.stop()

clean_token_str = _engine.clean_token_str
verify_token_connection = _engine.verify_token_connection
load_custom_tech_whitelist = _engine.load_custom_tech_whitelist
load_optimized_market_data = _engine.load_optimized_market_data
_safe_float = _engine._safe_float


# -----------------------------------------------------------------------------
# 周线与SKDJ
# -----------------------------------------------------------------------------
def build_weekly_bars(daily_indexed: pd.DataFrame) -> pd.DataFrame:
    """一次性把整段日线聚合成周线（比逐个信号日重算快很多）。"""
    frame = daily_indexed.reset_index()
    if "trade_date_str" not in frame.columns:
        return pd.DataFrame()
    frame["dt"] = pd.to_datetime(frame["trade_date_str"], errors="coerce")
    frame = frame.dropna(subset=["dt"])
    if frame.empty:
        return pd.DataFrame()
    frame["year_week"] = frame["dt"].dt.strftime("%G_%V")
    aggregations = {
        "trade_date_str": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "raw_close" in frame.columns:
        aggregations["raw_close"] = "last"
    weekly = (
        frame.groupby("year_week", as_index=False)
        .agg(aggregations)
        .sort_values("trade_date_str")
        .reset_index(drop=True)
    )
    return weekly


def add_skdj(weekly: pd.DataFrame, n_period: int, m_period: int) -> pd.DataFrame:
    """严格按通达信公式实现：
        LOWV := LLV(LOW, N)
        HIGHV := HHV(HIGH, N)
        RSV := EMA((CLOSE-LOWV)/(HIGHV-LOWV)*100, M)
        K : EMA(RSV, M)
        D : MA(K, M)
    通达信的 EMA(X,M) 对应 pandas 的 ewm(span=M)，MA(X,M) 对应 rolling(M).mean()。
    """
    low = pd.to_numeric(weekly["low"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    close = pd.to_numeric(weekly["close"], errors="coerce")

    low_n = low.rolling(n_period).min()
    high_n = high.rolling(n_period).max()
    raw_rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100.0
    rsv = raw_rsv.ewm(span=m_period, adjust=False).mean()
    weekly["K"] = rsv.ewm(span=m_period, adjust=False).mean()
    weekly["D"] = weekly["K"].rolling(m_period).mean()
    return weekly


def add_forward_returns(weekly: pd.DataFrame, horizons) -> pd.DataFrame:
    """下一周开盘买入，持有N周后按当周收盘卖出。

    对信号周 i：买入价 = 第 i+1 周的开盘价，
                卖出价 = 第 i+N 周的收盘价。
    """
    open_next = pd.to_numeric(weekly["open"], errors="coerce").shift(-1)
    close = pd.to_numeric(weekly["close"], errors="coerce")
    weekly["Entry_Open_Next_Week"] = open_next
    for n_weeks in horizons:
        exit_close = close.shift(-n_weeks)
        weekly[f"Fwd_{n_weeks}W_pct"] = (
            exit_close / open_next.replace(0, np.nan) - 1.0
        ) * 100.0
    return weekly


# -----------------------------------------------------------------------------
# 统计
# -----------------------------------------------------------------------------
def describe_group(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"样本数": 0, "平均收益%": np.nan, "中位收益%": np.nan, "胜率%": np.nan}
    return {
        "样本数": int(len(numeric)),
        "平均收益%": float(numeric.mean()),
        "中位收益%": float(numeric.median()),
        "胜率%": float((numeric > 0).mean() * 100.0),
    }


def compare_signal_vs_baseline(
    panel: pd.DataFrame, signal_mask: pd.Series, horizons, label: str
) -> pd.DataFrame:
    """信号组 vs 全池基准组的逐周期对比，核心输出是超额收益。"""
    rows = []
    for n_weeks in horizons:
        column = f"Fwd_{n_weeks}W_pct"
        signal_values = pd.to_numeric(
            panel.loc[signal_mask, column], errors="coerce"
        ).dropna()
        base_values = pd.to_numeric(panel[column], errors="coerce").dropna()
        signal_stats = describe_group(signal_values)
        base_stats = describe_group(base_values)

        # 粗略的显著性参考：均值差 / 信号组均值的标准误。
        # 样本相关性（同一周多只股票同涨同跌）会让真实显著性低于这个数字，
        # 所以它只用来排除"明显是噪声"的情况，不能当作严格检验。
        if len(signal_values) > 1 and signal_values.std(ddof=1) > 0:
            std_error = signal_values.std(ddof=1) / math.sqrt(len(signal_values))
            t_stat = (
                (signal_stats["平均收益%"] - base_stats["平均收益%"]) / std_error
                if std_error > 0
                else np.nan
            )
        else:
            t_stat = np.nan

        rows.append(
            {
                "信号定义": label,
                "持有周数": n_weeks,
                "信号样本数": signal_stats["样本数"],
                "信号平均收益%": signal_stats["平均收益%"],
                "基准平均收益%": base_stats["平均收益%"],
                "超额收益%(核心)": signal_stats["平均收益%"] - base_stats["平均收益%"],
                "信号中位收益%": signal_stats["中位收益%"],
                "基准中位收益%": base_stats["中位收益%"],
                "超额中位收益%": signal_stats["中位收益%"] - base_stats["中位收益%"],
                "信号胜率%": signal_stats["胜率%"],
                "基准胜率%": base_stats["胜率%"],
                "胜率差%": signal_stats["胜率%"] - base_stats["胜率%"],
                "粗略t值": t_stat,
            }
        )
    return pd.DataFrame(rows)


def yearly_breakdown(
    panel: pd.DataFrame, signal_mask: pd.Series, hold_weeks: int
) -> pd.DataFrame:
    """按年度拆分——判断优势是长期稳定存在，还是只集中在某一两年。"""
    column = f"Fwd_{hold_weeks}W_pct"
    work = panel.copy()
    work["_year"] = work["Signal_Date"].astype(str).str[:4]
    work["_is_signal"] = signal_mask.values
    rows = []
    for year, group in work.groupby("_year", sort=True):
        signal_values = pd.to_numeric(
            group.loc[group["_is_signal"], column], errors="coerce"
        ).dropna()
        base_values = pd.to_numeric(group[column], errors="coerce").dropna()
        if base_values.empty:
            continue
        signal_mean = signal_values.mean() if len(signal_values) else np.nan
        rows.append(
            {
                "年份": year,
                "信号样本数": int(len(signal_values)),
                "信号平均收益%": signal_mean,
                "基准平均收益%": float(base_values.mean()),
                "超额收益%": (
                    signal_mean - base_values.mean()
                    if len(signal_values)
                    else np.nan
                ),
                "信号胜率%": (
                    float((signal_values > 0).mean() * 100.0)
                    if len(signal_values)
                    else np.nan
                ),
                "基准胜率%": float((base_values > 0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="周线SKDJ信号验证器", layout="wide")
    st.title("🔍 周线SKDJ信号有效性验证器")
    st.caption(
        "本工具不做选股、不做组合、不做资金管理，只回答一个问题："
        "周线SKDJ信号之后的表现，是否真的优于同期随便买一只科技股。"
    )
    st.info(
        "**为什么必须看超额收益**：牛市里随便买什么都涨，只看绝对收益会把大盘的功劳"
        "误认成信号的功劳。只有信号组相对同期全池基准仍然领先，才说明信号本身携带信息。"
        f"（数据层复用自 `{_engine_name}.py`，缓存共用，不会重复下载行情）"
    )

    with st.sidebar:
        st.header("验证配置")
        try:
            secret_token = st.secrets.get("TUSHARE_TOKEN", "")
        except Exception:
            secret_token = ""
        token_input = st.text_input("Tushare Token", value=secret_token, type="password")

        today = pd.Timestamp.now().date()
        start_input = st.date_input("开始日期", value=today - timedelta(days=365 * 4))
        end_input = st.date_input("结束日期", value=today)

        st.markdown("---")
        st.subheader("SKDJ 参数")
        n_period = st.number_input("N（LLV/HHV周期）", value=6, min_value=2, max_value=60, step=1)
        m_period = st.number_input("M（EMA/MA周期）", value=3, min_value=2, max_value=30, step=1)
        cross_level = st.number_input(
            "上穿判定线", value=25.0, min_value=1.0, max_value=90.0, step=5.0,
            help="你的原始想法是25。脚本会同时测试它附近的几个值，检验结论是否只在25成立。",
        )

        st.markdown("---")
        st.subheader("股票池硬条件（与主程序一致）")
        min_price = st.number_input("最低股价（元）", value=10.0, min_value=0.0, step=1.0)
        min_mv = st.number_input("最低流通市值（亿元）", value=100.0, min_value=0.0, step=10.0)
        max_mv = st.number_input("最高流通市值（亿元）", value=1000.0, min_value=100.0, step=100.0)

        st.markdown("---")
        run_clicked = st.button("开始验证", type="primary")

    if not run_clicked:
        st.markdown(
            """
### 这个工具会给你三张表

**1. 信号 vs 基准总表**  
最重要的一列是 **超额收益%**。它的含义：
- 明显为正（且样本足够多） → 信号确实有效，值得在它上面搭策略、加日线择时
- 接近 0 → 信号只是跟着大盘走，没有独立价值，再精妙的买点也救不回来
- 为负 → 信号方向是反的

**2. 分年度表**  
判断优势是长期稳定，还是只靠某一两年撑起来的。如果只有一年为正、其余年份都为负，
那和之前四年回测遇到的问题是同一个——收益集中在极少数窗口，不可复制。

**3. 参数敏感性表**  
同时测试上穿 15/20/25/30/35 各条线，以及"上穿且K>D"等变体。
如果只有25这一个值有效、旁边的20和30都失效，那这个25大概率是数据里的巧合而不是规律。

---
填好左侧配置后点击"开始验证"。首次运行需要下载行情（复用主程序缓存，之前下过的不会重复下载）。
            """
        )
        return

    token_clean = clean_token_str(token_input)
    valid, message = verify_token_connection(token_clean)
    if not valid:
        st.error(f"Token校验失败：{message}")
        return
    if max_mv <= min_mv:
        st.error("最高流通市值必须大于最低流通市值。")
        return

    horizons = [1, 2, 3, 4]
    start_date = start_input.strftime("%Y%m%d")
    end_date = end_input.strftime("%Y%m%d")
    # 预留周线指标预热窗口 + 未来收益观察窗口
    fetch_start = (
        pd.Timestamp(start_input) - timedelta(days=400)
    ).strftime("%Y%m%d")
    fetch_end = (pd.Timestamp(end_input) + timedelta(days=60)).strftime("%Y%m%d")

    with st.spinner("构建科技股研究池……"):
        whitelist_set, name_map, industry_map = load_custom_tech_whitelist(token_clean)
    if not whitelist_set:
        st.error("未取得科技股研究池，请检查Token权限。")
        return
    st.success(f"科技股研究池：{len(whitelist_set)}只")

    with st.spinner("加载行情（复用主程序缓存）……"):
        stocks, basic_indexed, _, _, failed_dates, sync_stats = load_optimized_market_data(
            fetch_start, fetch_end, token_clean, tuple(sorted(whitelist_set))
        )
    if not stocks:
        st.error("未加载到行情数据。")
        return
    st.caption(
        f"行情：复用{sync_stats.get('cached_days', 0)}天，"
        f"本次下载{sync_stats.get('downloaded_days', 0)}天。"
    )
    if failed_dates:
        st.warning(f"{len(failed_dates)}个交易日未取得，结果可能有少量缺口。")

    # ---- 逐股票构建周线面板 ----
    progress = st.progress(0.0, text="计算周线SKDJ与未来收益……")
    panel_parts = []
    codes = sorted(stocks.keys())
    for idx, ts_code in enumerate(codes):
        weekly = build_weekly_bars(stocks[ts_code])
        if weekly.empty or len(weekly) < max(n_period, m_period) + 15:
            continue
        weekly = add_skdj(weekly, int(n_period), int(m_period))
        weekly = add_forward_returns(weekly, horizons)
        weekly["ts_code"] = ts_code
        weekly["Signal_Date"] = weekly["trade_date_str"].astype(str)
        panel_parts.append(
            weekly[
                [
                    "ts_code", "Signal_Date", "close", "raw_close",
                    "K", "D", "Entry_Open_Next_Week",
                    *[f"Fwd_{n}W_pct" for n in horizons],
                ]
                if "raw_close" in weekly.columns
                else [
                    "ts_code", "Signal_Date", "close",
                    "K", "D", "Entry_Open_Next_Week",
                    *[f"Fwd_{n}W_pct" for n in horizons],
                ]
            ]
        )
        if idx % 50 == 0:
            progress.progress(
                min((idx + 1) / len(codes), 1.0),
                text=f"计算周线SKDJ与未来收益……{idx + 1}/{len(codes)}",
            )
    progress.empty()

    if not panel_parts:
        st.error("没有足够长的周线数据。")
        return
    panel = pd.concat(panel_parts, ignore_index=True)

    # ---- 应用股票池硬条件（与主程序口径一致）----
    panel = panel[
        (panel["Signal_Date"] >= start_date) & (panel["Signal_Date"] <= end_date)
    ].copy()
    price_column = "raw_close" if "raw_close" in panel.columns else "close"
    panel = panel[pd.to_numeric(panel[price_column], errors="coerce") >= min_price]

    if not basic_indexed.empty:
        basic_reset = basic_indexed.reset_index()
        basic_reset = basic_reset.rename(columns={"trade_date_str": "Signal_Date"})
        keep_columns = [
            column
            for column in ("Signal_Date", "ts_code", "circ_mv")
            if column in basic_reset.columns
        ]
        if len(keep_columns) == 3:
            panel = panel.merge(
                basic_reset[keep_columns].drop_duplicates(["Signal_Date", "ts_code"]),
                on=["Signal_Date", "ts_code"],
                how="left",
            )
            circ_mv_billion = pd.to_numeric(panel["circ_mv"], errors="coerce") / 10000.0
            panel = panel[
                circ_mv_billion.between(min_mv, max_mv) | circ_mv_billion.isna()
            ]

    panel = panel.dropna(subset=["K", f"Fwd_{horizons[0]}W_pct"]).reset_index(drop=True)
    if panel.empty:
        st.error("过滤后没有可用样本，请放宽股票池条件或时间范围。")
        return

    st.markdown("---")
    st.header("验证结果")
    st.caption(
        f"全池基准样本：{len(panel):,} 个「个股-周」观测。"
        f"每一个信号都会和同一批基准做对比。"
    )

    k_now = pd.to_numeric(panel["K"], errors="coerce")
    k_prev = k_now.groupby(panel["ts_code"]).shift(1)
    d_now = pd.to_numeric(panel["D"], errors="coerce")

    # ---- 表1：主假设 ----
    main_mask = (k_prev <= cross_level) & (k_now > cross_level)
    main_table = compare_signal_vs_baseline(
        panel, main_mask.fillna(False), horizons, f"K上穿{cross_level:.0f}"
    )
    st.subheader(f"表1 · 你的假设：K上穿{cross_level:.0f}")
    st.dataframe(
        main_table.round(3), width="stretch", hide_index=True
    )
    st.caption(
        "**看「超额收益%(核心)」这一列**：明显为正=信号有效；接近0=只是跟随大盘；为负=方向相反。"
        "粗略t值仅供排除明显噪声（同周个股涨跌高度相关，真实显著性低于该值），|t|<2 基本可认为没有说服力。"
    )

    # ---- 表2：分年度 ----
    st.subheader("表2 · 分年度拆解（优势是稳定的还是靠某一年？）")
    year_table = yearly_breakdown(panel, main_mask.fillna(False), 3)
    st.dataframe(year_table.round(3), width="stretch", hide_index=True)
    st.caption(
        "以持有3周为例。如果超额收益只有一两年为正、其余年份为负，"
        "说明优势不可复制——这正是之前四年回测暴露的问题。"
    )

    # ---- 表3：参数敏感性 ----
    st.subheader("表3 · 参数敏感性（25这条线是规律还是巧合？）")
    variant_tables = []
    for level in (15.0, 20.0, 25.0, 30.0, 35.0):
        mask = (k_prev <= level) & (k_now > level)
        variant_tables.append(
            compare_signal_vs_baseline(
                panel, mask.fillna(False), [3], f"K上穿{level:.0f}"
            )
        )
    cross_and_kd = (k_prev <= cross_level) & (k_now > cross_level) & (k_now > d_now)
    variant_tables.append(
        compare_signal_vs_baseline(
            panel, cross_and_kd.fillna(False), [3],
            f"K上穿{cross_level:.0f} 且 K>D",
        )
    )
    low_zone_turn = (k_prev <= k_now) & (k_now <= cross_level)
    variant_tables.append(
        compare_signal_vs_baseline(
            panel, low_zone_turn.fillna(False), [3],
            f"K在{cross_level:.0f}下方拐头（旧R6近似口径）",
        )
    )
    variant_table = pd.concat(variant_tables, ignore_index=True)
    st.dataframe(variant_table.round(3), width="stretch", hide_index=True)
    st.caption(
        "全部按持有3周计算。**如果只有25有效、20和30都失效，那25大概率是巧合而非规律**——"
        "一个真实的市场规律不会对阈值这么敏感。最后一行是旧R6分支的近似口径，"
        "可以直接看出它和你的原始想法差别有多大。"
    )

    # ---- 导出 ----
    st.markdown("---")
    signal_rows = panel.loc[main_mask.fillna(False)].copy()
    signal_rows["name"] = signal_rows["ts_code"].map(name_map)
    signal_rows["Industry"] = signal_rows["ts_code"].map(industry_map)

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_signal_vs_baseline.csv",
            main_table.to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "02_yearly_breakdown.csv",
            year_table.to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "03_parameter_sensitivity.csv",
            variant_table.to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "04_all_signal_rows.csv",
            signal_rows.to_csv(index=False, encoding="utf-8-sig"),
        )
    st.download_button(
        "下载完整验证结果",
        data=output.getvalue(),
        file_name="skdj_signal_edge_validation.zip",
        mime="application/zip",
    )

    with st.expander("查看全部信号明细"):
        st.dataframe(signal_rows, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
