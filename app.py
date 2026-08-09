from __future__ import annotations

import io
import math
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


TITLE = "周线MACD每周前三名排序验证器 V4.5"
TOP_N = 3
RANDOM_SEED = 20260809

SCHEMES = {
    "S1": "仅13周相对强度",
    "S2": "13周70%＋26周30%",
    "S3": "13周70%＋26周20%＋KDJ10%",
}

REQUIRED = {
    "Cycle_ID", "ts_code", "name", "Sample_Board", "Signal_Date", "Signal_Year",
    "Board_RS_13W_pct", "Board_RS_26W_pct", "KDJ_Zone", "KDJ_Cross_Within_2W",
    "KDJ_KD_Both_Rising", "Strong_Sustained", "Weak_Rebound",
    "Short_Profitable_Rebound", "Target30_Before_Stop", "Stop_8W",
    "Return_8W_pct", "Red_Cycle_Weeks",
}


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "是"}


def read_csv(raw: bytes) -> pd.DataFrame:
    error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False)
        except Exception as exc:
            error = exc
    raise ValueError(f"CSV无法读取：{error}")


def load_factor_events(uploaded: Any) -> tuple[pd.DataFrame, str]:
    raw = uploaded.getvalue()
    if uploaded.name.lower().endswith(".csv"):
        return read_csv(raw), uploaded.name
    if not uploaded.name.lower().endswith(".zip"):
        raise ValueError("请上传V4.4全部结果ZIP或01_factor_events.csv。")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        candidates = [n for n in names if Path(n).name.lower() == "01_factor_events.csv"]
        if not candidates:
            candidates = [n for n in names if "factor_events" in Path(n).name.lower() and n.endswith(".csv")]
        if not candidates:
            raise ValueError("ZIP中没有找到V4.4的01_factor_events.csv。")
        target = sorted(candidates, key=len)[0]
        return read_csv(archive.read(target)), f"{uploaded.name}/{target}"


def validate(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError("输入不是V4.4因子事件表，缺少字段：" + "、".join(missing))


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    work = frame.copy().drop_duplicates("Cycle_ID").reset_index(drop=True)
    work["Signal_Date"] = work["Signal_Date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    work["Signal_Year"] = pd.to_numeric(work["Signal_Year"], errors="coerce").astype("Int64")
    numeric = [
        "Board_RS_13W_pct", "Board_RS_26W_pct", "Return_8W_pct", "Red_Cycle_Weeks",
    ]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    booleans = [
        "KDJ_Cross_Within_2W", "KDJ_KD_Both_Rising", "Strong_Sustained",
        "Weak_Rebound", "Short_Profitable_Rebound", "Target30_Before_Stop", "Stop_8W",
    ]
    for column in booleans:
        work[column] = work[column].map(to_bool)
    work["KDJ_Focus"] = (
        work["KDJ_Zone"].astype(str).eq("50—80")
        & work["KDJ_Cross_Within_2W"] & work["KDJ_KD_Both_Rising"]
    )
    primary = "Exit_T30_Return_pct" if "Exit_T30_Return_pct" in work.columns else "Return_8W_pct"
    work["Primary_Return_pct"] = pd.to_numeric(work[primary], errors="coerce")
    work = work.dropna(subset=["Signal_Date", "Board_RS_13W_pct", "Board_RS_26W_pct", "Primary_Return_pct"])

    grouped = work.groupby("Signal_Date", sort=False)
    work["RS13_Weekly_Percentile"] = grouped["Board_RS_13W_pct"].rank(method="average", pct=True) * 100.0
    work["RS26_Weekly_Percentile"] = grouped["Board_RS_26W_pct"].rank(method="average", pct=True) * 100.0
    work["Score_S1"] = work["RS13_Weekly_Percentile"]
    work["Score_S2"] = 0.70 * work["RS13_Weekly_Percentile"] + 0.30 * work["RS26_Weekly_Percentile"]
    work["Score_S3"] = (
        0.70 * work["RS13_Weekly_Percentile"]
        + 0.20 * work["RS26_Weekly_Percentile"]
        + 10.0 * work["KDJ_Focus"].astype(float)
    )
    return work.reset_index(drop=True), primary


def select_top(frame: pd.DataFrame, scheme: str) -> pd.DataFrame:
    score = f"Score_{scheme}"
    pieces = []
    for _, group in frame.groupby("Signal_Date", sort=True):
        selected = group.sort_values(
            [score, "Board_RS_13W_pct", "Board_RS_26W_pct", "ts_code"],
            ascending=[False, False, False, True], kind="mergesort",
        ).head(TOP_N).copy()
        selected["Weekly_Rank"] = np.arange(1, len(selected) + 1)
        pieces.append(selected)
    # V4.4事件表列很多；copy会合并继承来的碎片化内存块，避免后续加列时
    # 反复触发PerformanceWarning并拖慢Streamlit运行。
    result = pd.concat(pieces, ignore_index=True).copy() if pieces else pd.DataFrame()
    result["Scheme"] = scheme
    result["Scheme_Name"] = SCHEMES[scheme]
    result["Ranking_Score"] = result[score]
    return result


def profit_factor(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    positive = clean[clean > 0].sum()
    negative = -clean[clean < 0].sum()
    return float(positive / negative) if negative > 0 else np.nan


def contribution_ratio(values: pd.Series, top_n: int) -> float:
    positive = pd.to_numeric(values, errors="coerce").dropna()
    positive = positive[positive > 0].sort_values(ascending=False)
    return float(positive.head(top_n).sum() / positive.sum() * 100.0) if positive.sum() > 0 else np.nan


def mean_without_top_trades(frame: pd.DataFrame, top_n: int) -> float:
    if len(frame) <= top_n:
        return np.nan
    remaining = frame.sort_values("Primary_Return_pct", ascending=False).iloc[top_n:]
    return float(remaining["Primary_Return_pct"].mean())


def stock_concentration(frame: pd.DataFrame, top_n: int) -> float:
    stock_returns = frame.groupby("ts_code")["Primary_Return_pct"].sum().sort_values(ascending=False)
    positive = stock_returns[stock_returns > 0]
    return float(positive.head(top_n).sum() / positive.sum() * 100.0) if positive.sum() > 0 else np.nan


def mean_without_top_stocks(frame: pd.DataFrame, top_n: int) -> float:
    stock_returns = frame.groupby("ts_code")["Primary_Return_pct"].sum().sort_values(ascending=False)
    remove = set(stock_returns.head(top_n).index)
    remaining = frame[~frame["ts_code"].isin(remove)]
    return float(remaining["Primary_Return_pct"].mean()) if len(remaining) else np.nan


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(frame["Primary_Return_pct"], errors="coerce")
    raw_returns = pd.to_numeric(frame["Return_8W_pct"], errors="coerce")
    return {
        "入选事件": len(frame),
        "涉及股票": frame["ts_code"].nunique(),
        "持续强上涨(%)": frame["Strong_Sustained"].mean() * 100.0,
        "弱反弹或失败(%)": frame["Weak_Rebound"].mean() * 100.0,
        "短期盈利反弹(%)": frame["Short_Profitable_Rebound"].mean() * 100.0,
        "30%先于止损(%)": frame["Target30_Before_Stop"].mean() * 100.0,
        "八周止损率(%)": frame["Stop_8W"].mean() * 100.0,
        "可执行收益均值(%)": returns.mean(),
        "可执行收益中位数(%)": returns.median(),
        "可执行胜率(%)": (returns > 0).mean() * 100.0,
        "可执行盈亏比": profit_factor(returns),
        "原始八周收益均值(%)": raw_returns.mean(),
        "原始八周收益中位数(%)": raw_returns.median(),
        "红柱周数中位数": pd.to_numeric(frame["Red_Cycle_Weeks"], errors="coerce").median(),
        "前3笔盈利贡献(%)": contribution_ratio(returns, 3),
        "前10笔盈利贡献(%)": contribution_ratio(returns, 10),
        "剔除前3笔后收益均值(%)": mean_without_top_trades(frame, 3),
        "剔除前10笔后收益均值(%)": mean_without_top_trades(frame, 10),
        "前3只股票盈利贡献(%)": stock_concentration(frame, 3),
        "前10只股票盈利贡献(%)": stock_concentration(frame, 10),
        "剔除前3只股票后收益均值(%)": mean_without_top_stocks(frame, 3),
        "剔除前10只股票后收益均值(%)": mean_without_top_stocks(frame, 10),
    }


def calendar_audit(frame: pd.DataFrame) -> dict[str, int | float]:
    dates = pd.to_datetime(frame["Signal_Date"], format="%Y%m%d", errors="coerce").dropna()
    periods = dates.dt.to_period("W-FRI")
    first, last = periods.min(), periods.max()
    full = pd.period_range(first, last, freq="W-FRI")
    candidate_weeks = periods.nunique()
    return {
        "区间周数": len(full), "有候选周数": candidate_weeks,
        "原始空窗周数": len(full) - candidate_weeks,
        "有候选周覆盖率(%)": candidate_weeks / len(full) * 100.0 if len(full) else np.nan,
    }


def random_index_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    return [group.index.to_numpy() for _, group in frame.groupby("Signal_Date", sort=True)]


def random_simulation(
    frame: pd.DataFrame, repetitions: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    groups = random_index_groups(frame)
    overall_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    example = pd.DataFrame()
    for repetition in range(repetitions):
        indexes = np.concatenate([
            rng.choice(group, size=min(TOP_N, len(group)), replace=False) for group in groups
        ])
        selected = frame.loc[indexes].copy()
        if repetition == 0:
            example = selected.copy()
            example["Scheme"] = "S0"
            example["Scheme_Name"] = "随机前三名示例（固定种子第1次）"
            example["Weekly_Rank"] = np.nan
            example["Ranking_Score"] = np.nan
        overall_rows.append({"Random_Repetition": repetition + 1, **metrics(selected)})
        for year, group in selected.groupby("Signal_Year", dropna=False):
            year_rows.append({"Random_Repetition": repetition + 1, "Signal_Year": year, **metrics(group)})
    return pd.DataFrame(overall_rows), pd.DataFrame(year_rows), example


def random_summary(samples: pd.DataFrame, group_columns: list[str] | None = None) -> pd.DataFrame:
    group_columns = group_columns or []
    metric_columns = [
        column for column in samples.columns
        if column not in {"Random_Repetition", *group_columns}
        and pd.api.types.is_numeric_dtype(samples[column])
    ]
    rows = []
    iterator = samples.groupby(group_columns, dropna=False) if group_columns else [((), samples)]
    for keys, group in iterator:
        if group_columns and not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys)) if group_columns else {}
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = values.mean()
            row[f"{column}_随机2.5%"] = values.quantile(0.025)
            row[f"{column}_随机97.5%"] = values.quantile(0.975)
        rows.append(row)
    return pd.DataFrame(rows)


def deterministic_summary(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme, name), group in selections.groupby(["Scheme", "Scheme_Name"], sort=True):
        rows.append({"方案": scheme, "方案名称": name, **metrics(group)})
    return pd.DataFrame(rows)


def main_comparison(
    frame: pd.DataFrame, deterministic: pd.DataFrame, random_samples: pd.DataFrame,
) -> pd.DataFrame:
    sample_summary = random_summary(random_samples)
    random_row = {"方案": "S0", "方案名称": "随机前三名蒙特卡洛均值"}
    for column in [c for c in sample_summary.columns if not c.endswith(("随机2.5%", "随机97.5%"))]:
        random_row[column] = sample_summary.iloc[0][column]
    rows = [random_row]
    rows.extend(deterministic_summary(deterministic).to_dict("records"))
    summary = pd.DataFrame(rows)
    base = summary[summary["方案"].eq("S0")].iloc[0]
    delta_metrics = [
        "持续强上涨(%)", "弱反弹或失败(%)", "30%先于止损(%)", "八周止损率(%)",
        "可执行收益均值(%)", "可执行收益中位数(%)", "可执行胜率(%)",
        "剔除前10只股票后收益均值(%)",
    ]
    for column in delta_metrics:
        summary[f"较随机_{column}"] = summary[column] - base[column]
    return summary


def year_report(
    selections: pd.DataFrame, random_year_samples: pd.DataFrame,
) -> pd.DataFrame:
    random_year = random_summary(random_year_samples, ["Signal_Year"])
    random_year["方案"] = "S0"
    random_year["方案名称"] = "随机前三名蒙特卡洛均值"
    rows = [random_year]
    det_rows = []
    for (year, scheme, name), group in selections.groupby(
        ["Signal_Year", "Scheme", "Scheme_Name"], dropna=False, sort=True
    ):
        det_rows.append({"Signal_Year": year, "方案": scheme, "方案名称": name, **metrics(group)})
    rows.append(pd.DataFrame(det_rows))
    report = pd.concat(rows, ignore_index=True, sort=False)
    base = report[report["方案"].eq("S0")][
        ["Signal_Year", "可执行收益均值(%)", "持续强上涨(%)", "弱反弹或失败(%)"]
    ].rename(columns={
        "可执行收益均值(%)": "__随机收益", "持续强上涨(%)": "__随机强上涨",
        "弱反弹或失败(%)": "__随机弱反弹",
    })
    report = report.merge(base, on="Signal_Year", how="left")
    report["较随机_可执行收益均值(%)"] = report["可执行收益均值(%)"] - report["__随机收益"]
    report["较随机_持续强上涨(百分点)"] = report["持续强上涨(%)"] - report["__随机强上涨"]
    report["较随机_弱反弹(百分点)"] = report["弱反弹或失败(%)"] - report["__随机弱反弹"]
    return report.drop(columns=["__随机收益", "__随机强上涨", "__随机弱反弹"])


def board_report(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (board, scheme, name), group in selections.groupby(
        ["Sample_Board", "Scheme", "Scheme_Name"], dropna=False, sort=True
    ):
        rows.append({"Sample_Board": board, "方案": scheme, "方案名称": name, **metrics(group)})
    return pd.DataFrame(rows)


def weekly_report(frame: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    counts = frame.groupby("Signal_Date").size().rename("候选数").reset_index()
    rows = []
    for (date, scheme), group in selections.groupby(["Signal_Date", "Scheme"], sort=True):
        ordered = group.sort_values("Weekly_Rank", na_position="last")
        rows.append({
            "Signal_Date": date, "方案": scheme, "方案名称": ordered["Scheme_Name"].iloc[0],
            "入选数": len(ordered),
            "股票代码": "|".join(ordered["ts_code"].astype(str)),
            "股票名称": "|".join(ordered["name"].astype(str)),
            "板块": "|".join(ordered["Sample_Board"].astype(str)),
            "平均13周相对强度": ordered["Board_RS_13W_pct"].mean(),
            "持续强上涨数": int(ordered["Strong_Sustained"].sum()),
            "弱反弹或失败数": int(ordered["Weak_Rebound"].sum()),
            "止损数": int(ordered["Stop_8W"].sum()),
            "可执行收益均值(%)": ordered["Primary_Return_pct"].mean(),
        })
    return pd.DataFrame(rows).merge(counts, on="Signal_Date", how="left")


def overlap_report(selections: pd.DataFrame) -> pd.DataFrame:
    sets = {
        scheme: set(group["Cycle_ID"].astype(str))
        for scheme, group in selections.groupby("Scheme")
    }
    rows = []
    for left in SCHEMES:
        for right in SCHEMES:
            union = sets[left] | sets[right]
            rows.append({
                "方案A": left, "方案B": right, "共同事件": len(sets[left] & sets[right]),
                "Jaccard重合率(%)": len(sets[left] & sets[right]) / len(union) * 100.0 if union else np.nan,
            })
    return pd.DataFrame(rows)


def acceptance_report(summary: pd.DataFrame, years: pd.DataFrame, calendar: dict[str, Any]) -> pd.DataFrame:
    base = summary[summary["方案"].eq("S0")].iloc[0]
    rows = []
    for scheme in SCHEMES:
        row = summary[summary["方案"].eq(scheme)].iloc[0]
        year_rows = years[years["方案"].eq(scheme)]
        positive_years = int(year_rows["较随机_可执行收益均值(%)"].gt(0).sum())
        tests = {
            "强上涨提高≥5个百分点": row["持续强上涨(%)"] - base["持续强上涨(%)"] >= 5.0,
            "弱反弹下降≥5个百分点": row["弱反弹或失败(%)"] - base["弱反弹或失败(%)"] <= -5.0,
            "止损率下降≥3个百分点": row["八周止损率(%)"] - base["八周止损率(%)"] <= -3.0,
            "收益中位数提高": row["可执行收益中位数(%)"] > base["可执行收益中位数(%)"],
            "至少两个年度收益提高": positive_years >= 2,
            "剔除前10只股票后仍优于随机": (
                row["剔除前10只股票后收益均值(%)"] > base["剔除前10只股票后收益均值(%)"]
            ),
            "不增加空窗": True,
        }
        passed = sum(bool(v) for v in tests.values())
        verdict = "通过：进入独立年份验证" if passed == len(tests) else (
            "观察：有增量但尚不完整" if passed >= 5 else "拒绝：不足以成为正式排序"
        )
        rows.append({
            "方案": scheme, "方案名称": SCHEMES[scheme], "通过条件数/7": passed,
            "结论": verdict, "正向年度数": positive_years, **tests,
        })
    return pd.DataFrame(rows).sort_values(["通过条件数/7", "方案"], ascending=[False, True])


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def build_result(frame: pd.DataFrame, source: str, repetitions: int) -> dict[str, Any]:
    deterministic_parts = [select_top(frame, scheme) for scheme in SCHEMES]
    deterministic = pd.concat(deterministic_parts, ignore_index=True)
    random_samples, random_year_samples, random_example = random_simulation(
        frame, repetitions, RANDOM_SEED
    )
    selections = pd.concat([random_example, deterministic], ignore_index=True, sort=False)
    summary = main_comparison(frame, deterministic, random_samples)
    years = year_report(deterministic, random_year_samples)
    boards = board_report(deterministic)
    weekly = weekly_report(frame, deterministic)
    overlap = overlap_report(deterministic)
    calendar = calendar_audit(frame)
    acceptance = acceptance_report(summary, years, calendar)
    candidates = frame.copy()
    for scheme in SCHEMES:
        selected_ids = set(deterministic.loc[deterministic["Scheme"].eq(scheme), "Cycle_ID"])
        candidates[f"Selected_{scheme}"] = candidates["Cycle_ID"].isin(selected_ids)
    random_stats = random_summary(random_samples)
    metadata = pd.DataFrame([
        {"项目": "程序", "值": TITLE}, {"项目": "输入", "值": source},
        {"项目": "生成时间", "值": datetime.now().isoformat(timespec="seconds")},
        {"项目": "候选事件", "值": len(frame)}, {"项目": "涉及股票", "值": frame["ts_code"].nunique()},
        {"项目": "蒙特卡洛重复", "值": repetitions}, {"项目": "随机种子", "值": RANDOM_SEED},
        {"项目": "可执行收益字段", "值": "Exit_T30_Return_pct（若不存在则Return_8W_pct）"},
        {"项目": "排序", "值": "S1=RS13；S2=70%RS13+30%RS26；S3=70%RS13+20%RS26+10%KDJ"},
        *({"项目": key, "值": value} for key, value in calendar.items()),
        {"项目": "限制", "值": "每周独立选前三，尚未处理最多三仓、持仓重叠和资金占用"},
    ])
    files = {
        "01_ranked_candidates.csv": csv_bytes(candidates),
        "02_selected_events.csv": csv_bytes(selections),
        "03_main_comparison.csv": csv_bytes(summary),
        "04_random_distribution.csv": csv_bytes(random_samples),
        "05_year_stability.csv": csv_bytes(years),
        "06_board_stability.csv": csv_bytes(boards),
        "07_weekly_choices.csv": csv_bytes(weekly),
        "08_bull_concentration_and_acceptance.csv": csv_bytes(acceptance),
        "09_scheme_overlap.csv": csv_bytes(overlap),
        "10_random_confidence_intervals.csv": csv_bytes(random_stats),
        "11_metadata.csv": csv_bytes(metadata),
    }
    return {
        "frame": frame, "deterministic": deterministic, "selections": selections,
        "summary": summary, "random_samples": random_samples, "years": years,
        "boards": boards, "weekly": weekly, "acceptance": acceptance,
        "overlap": overlap, "random_stats": random_stats, "metadata": metadata,
        "files": files, "zip": make_zip(files), "calendar": calendar,
    }


def show(frame: pd.DataFrame) -> None:
    formats = {
        column: "{:.2f}" for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and ("(%)" in column or "百分点" in column or "率" in column or "均值" in column or "中位数" in column)
    }
    st.dataframe(frame.style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)


def render(result: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("候选事件", f"{len(result['frame']):,}")
    c2.metric("有候选周", f"{result['calendar']['有候选周数']:,}")
    c3.metric("原始空窗周", f"{result['calendar']['原始空窗周数']:,}")
    c4.metric("每方案入选", f"{len(result['deterministic']) // len(SCHEMES):,}")
    st.subheader("预设门槛结论")
    show(result["acceptance"])
    st.subheader("随机前三名 vs 三种排序")
    show(result["summary"])
    st.subheader("逐年稳定性")
    show(result["years"])
    st.subheader("分板块")
    show(result["boards"])
    with st.expander("每周选择与方案重合率"):
        show(result["weekly"])
        show(result["overlap"])
    st.subheader("下载")
    st.download_button(
        "下载全部结果ZIP", result["zip"],
        file_name="weekly_macd_top3_rank_v4_5_all_results.zip",
        mime="application/zip", type="primary", key="v45_all", on_click="ignore",
    )
    labels = [
        "1号：全部候选评分", "2号：入选事件", "3号：主比较", "4号：随机分布",
        "5号：逐年", "6号：分板块", "7号：每周选择", "8号：门槛结论",
        "9号：方案重合", "10号：随机置信区间", "11号：运行信息",
    ]
    columns = st.columns(4)
    for index, (name, data) in enumerate(result["files"].items()):
        with columns[index % 4]:
            st.download_button(
                labels[index], data, file_name=name, mime="text/csv",
                key=f"v45_{name}", on_click="ignore",
            )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.info(
        "本版不设最低分，不减少有候选的星期。每周从全部第一根红柱事件中选择最多3只，"
        "比较相对强度排序与可复现的随机前三名。"
    )
    with st.expander("固定方案和通过门槛", expanded=False):
        st.markdown(
            """
            - S1：仅按当周13周相对强度百分位。
            - S2：13周70%＋26周30%。
            - S3：13周70%＋26周20%＋KDJ状态10%。
            - 随机基准：每周随机选择最多3只，默认重复2000次。

            通过需要：强上涨提高5个百分点、弱反弹下降5个百分点、止损下降3个百分点、
            中位数提高、至少两个年度收益提高、剔除前10只牛股后仍优于随机，并且不增加空窗。
            """
        )
    with st.form("v45_form"):
        upload = st.file_uploader(
            "上传V4.4全部结果ZIP或01_factor_events.csv", type=["zip", "csv"]
        )
        repetitions = st.number_input(
            "随机前三名重复次数", min_value=200, max_value=5000,
            value=2000, step=200,
        )
        submitted = st.form_submit_button("开始每周前三排序验证", type="primary")
    if submitted:
        if upload is None:
            st.error("请上传V4.4结果。")
        else:
            try:
                raw, source = load_factor_events(upload)
                validate(raw)
                frame, _ = prepare(raw)
                if frame.empty:
                    raise ValueError("没有可排序的因子事件。")
                with st.spinner("正在进行每周排序和随机基准模拟..."):
                    st.session_state["v45_result"] = build_result(
                        frame, source, int(repetitions)
                    )
            except Exception as exc:
                st.exception(exc)
    if "v45_result" in st.session_state:
        render(st.session_state["v45_result"])
    else:
        st.caption("不需要Token，不下载行情；2000次随机基准通常很快完成。")


if __name__ == "__main__":
    main()
