from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "周线MACD透明动态持仓验证器 V4.2"
CHECKPOINTS = (2, 3, 4, 5)
MIN_REQUIRED = {
    "Event_Type",
    "Cycle_ID",
    "Signal_Date",
    "Hist",
    "Red_Cycle_Weeks",
    "Cycle_Type",
    "Has_8W_Future",
    "MFE_8W_pct",
    "MAE_8W_pct",
    "Return_8W_pct",
    "Hit_Stop_8W",
    "Hit_30_8W",
}


@dataclass(frozen=True)
class Rule:
    code: str
    name: str
    explanation: str


RULES = (
    Rule("R0", "基准：检查周仍未止损", "不看柱体形态；只要此前没有触发-10%止损。"),
    Rule("R1", "红柱仍为正", "检查周标准MACD柱仍大于0。"),
    Rule("R2", "本周柱体扩张", "红柱为正，且本周柱高于上周。"),
    Rule("R3", "速度或加速度改善", "红柱为正，且柱体正在扩张，或收缩速度已经改善。"),
    Rule(
        "R4",
        "C1友好：只排除明确衰弱",
        "允许首次缩短和缩短后再扩张；只排除翻绿，或连续至少两周缩短且加速度不改善。",
    ),
    Rule("R5", "严格：扩张且加速", "红柱为正、柱体扩张，并且扩张速度不低于上周。"),
)


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "是"}


def num(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def rate(mask: pd.Series) -> float:
    clean = mask.dropna()
    return float(clean.astype(bool).mean() * 100.0) if len(clean) else np.nan


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin) * 100.0, min(1.0, centre + margin) * 100.0


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False)
        except Exception as exc:  # pragma: no cover - only used for fallback encodings
            last_error = exc
    raise ValueError(f"CSV无法读取：{last_error}")


def load_events(uploaded: Any) -> tuple[pd.DataFrame, str]:
    raw = uploaded.getvalue()
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return read_csv_bytes(raw), uploaded.name
    if not name.endswith(".zip"):
        raise ValueError("只接受V4.1结果ZIP或01_events.csv。")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        files = [n for n in archive.namelist() if not n.endswith("/")]
        preferred = [n for n in files if Path(n).name.lower() == "01_events.csv"]
        if not preferred:
            preferred = [n for n in files if "events" in Path(n).name.lower() and n.lower().endswith(".csv")]
        if not preferred:
            raise ValueError("ZIP中没有找到01_events.csv或events CSV。")
        target = sorted(preferred, key=len)[0]
        return read_csv_bytes(archive.read(target)), f"{uploaded.name}/{target}"


def validate_events(events: pd.DataFrame) -> None:
    missing = sorted(MIN_REQUIRED - set(events.columns))
    for week in CHECKPOINTS:
        for column in (f"CP_W{week}_Observed", f"CP_W{week}_Hist", f"CP_W{week}_Stop_Hit_Before"):
            if column not in events.columns:
                missing.append(column)
    if missing:
        raise ValueError("缺少V4.1字段：" + "、".join(sorted(set(missing))))


def signal_year(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"\D", "", regex=True)
    return pd.to_numeric(text.str[:4], errors="coerce").astype("Int64")


def hist_path(row: pd.Series, week: int) -> list[float]:
    values = [num(row.get("Hist"))]
    for k in range(2, week + 1):
        values.append(num(row.get(f"CP_W{k}_Hist")))
    return values


def consecutive_shrinks(values: list[float]) -> int:
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    streak = 0
    for value in reversed(diffs):
        if math.isfinite(value) and value < 0:
            streak += 1
        else:
            break
    return streak


def checkpoint_features(row: pd.Series, week: int) -> dict[str, Any]:
    values = hist_path(row, week)
    observed = to_bool(row.get(f"CP_W{week}_Observed")) and math.isfinite(values[-1])
    stopped_before = to_bool(row.get(f"CP_W{week}_Stop_Hit_Before"))
    h = values[-1]
    velocity = h - values[-2] if observed and math.isfinite(values[-2]) else np.nan
    previous_velocity = (
        values[-2] - values[-3]
        if week >= 3 and all(math.isfinite(v) for v in values[-3:-1])
        else np.nan
    )
    acceleration = velocity - previous_velocity if math.isfinite(previous_velocity) else np.nan
    shrink_streak = consecutive_shrinks(values) if observed else 0
    prior_diffs = [values[i] - values[i - 1] for i in range(1, len(values) - 1)]
    had_prior_shrink = any(math.isfinite(v) and v < 0 for v in prior_diffs)
    re_expansion = observed and h > 0 and velocity > 0 and had_prior_shrink

    if not observed:
        state = "数据不足"
    elif h <= 0:
        state = "已经翻绿"
    elif re_expansion:
        state = "C1式再扩张"
    elif velocity > 0 and (not math.isfinite(acceleration) or acceleration >= 0):
        state = "红柱扩张加速"
    elif velocity > 0:
        state = "红柱扩张减速"
    elif shrink_streak == 1:
        state = "首次缩短"
    else:
        state = "连续缩短"

    eligible = observed and not stopped_before
    red = eligible and h > 0
    v_improves = red and (velocity > 0 or (math.isfinite(acceleration) and acceleration > 0))
    clearly_weak = red and shrink_streak >= 2 and velocity < 0 and (
        not math.isfinite(acceleration) or acceleration <= 0
    )
    decisions = {
        "R0": eligible,
        "R1": red,
        "R2": red and velocity > 0,
        "R3": v_improves,
        "R4": red and not clearly_weak,
        "R5": red and velocity > 0 and math.isfinite(acceleration) and acceleration >= 0,
    }
    if week == 2:
        # W2只有一个速度，无法计算加速度；R5不参与W2比较。
        decisions["R5"] = False

    result: dict[str, Any] = {
        "Observed": observed,
        "Stopped_Before": stopped_before,
        "Eligible": eligible,
        "Hist": h,
        "Velocity": velocity,
        "Acceleration": acceleration,
        "Velocity_vs_W1": velocity / abs(values[0]) if math.isfinite(values[0]) and values[0] != 0 else np.nan,
        "Acceleration_vs_W1": acceleration / abs(values[0]) if math.isfinite(values[0]) and values[0] != 0 else np.nan,
        "Shrink_Streak": shrink_streak,
        "ReExpansion": re_expansion,
        "State": state,
    }
    result.update({f"Keep_{code}": value for code, value in decisions.items()})
    return result


def prepare_events(raw: pd.DataFrame, complete_8w_only: bool) -> pd.DataFrame:
    frame = raw[raw["Event_Type"].astype(str).eq("第一根红柱")].copy()
    if "Tradable" in frame.columns:
        frame = frame[frame["Tradable"].map(to_bool)]
    if complete_8w_only:
        frame = frame[frame["Has_8W_Future"].map(to_bool)]
    frame = frame.drop_duplicates("Cycle_ID").reset_index(drop=True)
    frame["Signal_Year"] = signal_year(frame["Signal_Date"])
    frame["Long_9W"] = pd.to_numeric(frame["Red_Cycle_Weeks"], errors="coerce").ge(9)
    frame["Cycle_Max_30"] = pd.to_numeric(frame.get("Cycle_Max_High_Return_pct"), errors="coerce").ge(30)
    frame["Hit_30"] = frame["Hit_30_8W"].map(to_bool)
    frame["Stop_10"] = frame["Hit_Stop_8W"].map(to_bool)
    if "First_30_vs_Stop" in frame.columns:
        frame["Target30_Before_Stop"] = frame["First_30_vs_Stop"].astype(str).eq("目标先到")
    else:
        frame["Target30_Before_Stop"] = frame["Hit_30"] & ~frame["Stop_10"]
    red_weeks = pd.to_numeric(frame["Red_Cycle_Weeks"], errors="coerce")
    completed = frame.get("Cycle_Completed", pd.Series(True, index=frame.index)).map(to_bool)
    frame["Weak_Rebound_3W"] = completed & red_weeks.le(3)
    frame["Is_C1"] = frame["Cycle_Type"].astype(str).str.startswith("C1_")
    frame["Is_C2"] = frame["Cycle_Type"].astype(str).str.startswith("C2_")
    for column in ("Return_8W_pct", "MFE_8W_pct", "MAE_8W_pct"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for week in CHECKPOINTS:
        features = frame.apply(lambda row: pd.Series(checkpoint_features(row, week)), axis=1)
        features = features.add_prefix(f"W{week}_")
        frame = pd.concat([frame, features], axis=1)
    return frame


def subset_metrics(group: pd.DataFrame) -> dict[str, Any]:
    n = len(group)
    long_n = int(group["Long_9W"].sum())
    ci_low, ci_high = wilson_interval(long_n, n)
    return {
        "入选数": n,
        "红柱持续至少9周(%)": rate(group["Long_9W"]),
        "九周概率95%下限(%)": ci_low,
        "九周概率95%上限(%)": ci_high,
        "八周最高涨幅≥30%(%)": rate(group["Hit_30"]),
        "30%先于止损(%)": rate(group["Target30_Before_Stop"]),
        "完整周期最高涨幅≥30%(%)": rate(group["Cycle_Max_30"]),
        "八周触及-10%(%)": rate(group["Stop_10"]),
        "八周收益均值(%)": float(group["Return_8W_pct"].mean()) if n else np.nan,
        "八周收益中位数(%)": float(group["Return_8W_pct"].median()) if n else np.nan,
        "弱反弹≤3周(%)": rate(group["Weak_Rebound_3W"]),
    }


def build_rule_report(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week in CHECKPOINTS:
        eligible = events[events[f"W{week}_Eligible"].map(to_bool)].copy()
        if eligible.empty:
            continue
        base = subset_metrics(eligible)
        c1_total = int(eligible["Is_C1"].sum())
        c2_total = int(eligible["Is_C2"].sum())
        winners_total = int(eligible["Target30_Before_Stop"].sum())
        for rule in RULES:
            if week == 2 and rule.code == "R5":
                continue
            chosen = eligible[eligible[f"W{week}_Keep_{rule.code}"].map(to_bool)].copy()
            metrics = subset_metrics(chosen)
            row = {
                "检查周": f"W{week}",
                "规则编号": rule.code,
                "规则": rule.name,
                "规则说明": rule.explanation,
                "可检查持仓数": len(eligible),
                "保留覆盖率(%)": len(chosen) / len(eligible) * 100.0,
                **metrics,
                "九周概率较基准变化(百分点)": metrics["红柱持续至少9周(%)"] - base["红柱持续至少9周(%)"],
                "30%先于止损较基准变化(百分点)": metrics["30%先于止损(%)"] - base["30%先于止损(%)"],
                "止损率较基准变化(百分点)": metrics["八周触及-10%(%)"] - base["八周触及-10%(%)"],
                "30%赢家保留率(%)": (
                    chosen["Target30_Before_Stop"].sum() / winners_total * 100.0 if winners_total else np.nan
                ),
                "C1保留率(%)": chosen["Is_C1"].sum() / c1_total * 100.0 if c1_total else np.nan,
                "C2保留率(%)": chosen["Is_C2"].sum() / c2_total * 100.0 if c2_total else np.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def build_state_report(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week in CHECKPOINTS:
        eligible = events[events[f"W{week}_Eligible"].map(to_bool)].copy()
        for state, group in eligible.groupby(f"W{week}_State", dropna=False):
            rows.append({"检查周": f"W{week}", "柱体状态": str(state), **subset_metrics(group)})
    return pd.DataFrame(rows)


def build_year_report(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, year_frame in events.groupby("Signal_Year", dropna=True):
        for week in CHECKPOINTS:
            eligible = year_frame[year_frame[f"W{week}_Eligible"].map(to_bool)].copy()
            if eligible.empty:
                continue
            baseline = rate(eligible["Long_9W"])
            for rule in RULES:
                if week == 2 and rule.code == "R5":
                    continue
                chosen = eligible[eligible[f"W{week}_Keep_{rule.code}"].map(to_bool)]
                metrics = subset_metrics(chosen)
                rows.append(
                    {
                        "年份": int(year),
                        "检查周": f"W{week}",
                        "规则编号": rule.code,
                        "规则": rule.name,
                        "可检查持仓数": len(eligible),
                        "保留覆盖率(%)": len(chosen) / len(eligible) * 100.0,
                        **metrics,
                        "九周概率较当年基准变化(百分点)": metrics["红柱持续至少9周(%)"] - baseline,
                    }
                )
    return pd.DataFrame(rows)


def build_verdicts(rule_report: pd.DataFrame, year_report: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = rule_report[~rule_report["规则编号"].eq("R0")]
    for _, row in candidates.iterrows():
        years = year_report[
            year_report["检查周"].eq(row["检查周"])
            & year_report["规则编号"].eq(row["规则编号"])
            & year_report["入选数"].ge(20)
        ]
        stable_years = int(years["九周概率较当年基准变化(百分点)"].gt(0).sum())
        testable_years = len(years)
        tests = {
            "覆盖率≥60%": row["保留覆盖率(%)"] >= 60,
            "九周概率提高≥5个百分点": row["九周概率较基准变化(百分点)"] >= 5,
            "止损率不升高": row["止损率较基准变化(百分点)"] <= 0,
            "保留≥75%的30%赢家": pd.isna(row["30%赢家保留率(%)"]) or row["30%赢家保留率(%)"] >= 75,
            "跨年方向稳定": testable_years >= 2 and stable_years / testable_years >= 0.67,
        }
        passed = sum(bool(v) for v in tests.values())
        if passed == 5:
            verdict = "通过：值得进入模拟交易回测"
        elif passed >= 3:
            verdict = "观察：有局部效果，暂不进入正式规则"
        else:
            verdict = "拒绝：效果或稳定性不足"
        rows.append(
            {
                "检查周": row["检查周"],
                "规则编号": row["规则编号"],
                "规则": row["规则"],
                "通过条件数/5": passed,
                "结论": verdict,
                "覆盖率≥60%": tests["覆盖率≥60%"],
                "九周概率提高≥5个百分点": tests["九周概率提高≥5个百分点"],
                "止损率不升高": tests["止损率不升高"],
                "保留≥75%的30%赢家": tests["保留≥75%的30%赢家"],
                "跨年方向稳定": tests["跨年方向稳定"],
                "可比较年份": testable_years,
                "正向年份": stable_years,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["通过条件数/5", "检查周", "规则编号"], ascending=[False, True, True]
    )


def display_frame(frame: pd.DataFrame, percent_decimals: int = 2) -> None:
    formats = {
        col: f"{{:.{percent_decimals}f}}"
        for col in frame.columns
        if "(%)" in col or "百分点" in col or "覆盖率" in col or "保留率" in col
    }
    st.dataframe(frame.style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def build_outputs(events: pd.DataFrame, source_name: str, complete_8w_only: bool) -> dict[str, Any]:
    rule_report = build_rule_report(events)
    state_report = build_state_report(events)
    year_report = build_year_report(events)
    verdicts = build_verdicts(rule_report, year_report)
    identity = [
        "Cycle_ID", "ts_code", "name", "Sample_Board", "SW_L1", "SW_L2", "SW_L3",
        "Signal_Date", "Signal_Year", "Cycle_Type", "Red_Cycle_Weeks", "Long_9W",
        "Return_8W_pct", "MFE_8W_pct", "MAE_8W_pct", "Hit_30", "Target30_Before_Stop",
        "Stop_10", "Weak_Rebound_3W", "Is_C1", "Is_C2",
    ]
    checkpoint_columns = [
        col for col in events.columns if any(col.startswith(f"W{k}_") for k in CHECKPOINTS)
    ]
    audit = events[[c for c in identity if c in events.columns] + checkpoint_columns].copy()
    metadata = pd.DataFrame(
        [
            {"项目": "程序", "值": APP_TITLE},
            {"项目": "输入文件", "值": source_name},
            {"项目": "生成时间", "值": datetime.now().isoformat(timespec="seconds")},
            {"项目": "第一根红柱事件数", "值": len(events)},
            {"项目": "仅保留完整八周", "值": complete_8w_only},
            {"项目": "快MACD", "值": "本版未加入；先验证标准柱体速度/加速度"},
            {"项目": "重要限制", "值": "规则结论是条件统计，不是可直接实盘的逐日成交回测"},
        ]
    )
    definitions = pd.DataFrame(
        [{"规则编号": r.code, "规则": r.name, "定义": r.explanation} for r in RULES]
    )
    files = {
        "01_event_audit.csv": csv_bytes(audit),
        "02_rule_report.csv": csv_bytes(rule_report),
        "03_state_report.csv": csv_bytes(state_report),
        "04_year_stability.csv": csv_bytes(year_report),
        "05_verdicts.csv": csv_bytes(verdicts),
        "06_rule_definitions.csv": csv_bytes(definitions),
        "07_metadata.csv": csv_bytes(metadata),
    }
    return {
        "events": events,
        "rule_report": rule_report,
        "state_report": state_report,
        "year_report": year_report,
        "verdicts": verdicts,
        "definitions": definitions,
        "metadata": metadata,
        "files": files,
        "zip": make_zip(files),
    }


def render_results(result: dict[str, Any]) -> None:
    events = result["events"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("第一根红柱样本", f"{len(events):,}")
    c2.metric("C1", f"{int(events['Is_C1'].sum()):,}")
    c3.metric("C2", f"{int(events['Is_C2'].sum()):,}")
    c4.metric("完整周期≥9周", f"{rate(events['Long_9W']):.2f}%")

    st.subheader("先看结论：哪些规则达到预设门槛")
    st.caption("门槛在查看结果之前已经固定。五项全部通过，才建议进入下一轮逐日模拟回测。")
    display_frame(result["verdicts"])

    st.subheader("规则总表")
    st.caption(
        "R0是同一检查周的基准。覆盖率和赢家保留率很重要：概率提高但把多数C1或大赢家删掉，同样不可接受。"
    )
    display_frame(result["rule_report"])

    st.subheader("柱体状态的真实后续表现")
    display_frame(result["state_report"])

    st.subheader("逐年稳定性")
    st.caption("单年入选数少于20时，只展示，不参与‘跨年方向稳定’判定。")
    display_frame(result["year_report"])

    st.subheader("规则定义")
    st.dataframe(result["definitions"], use_container_width=True, hide_index=True)

    st.subheader("下载（编号与压缩包内文件一致）")
    st.download_button(
        "下载全部结果 ZIP",
        data=result["zip"],
        file_name="weekly_macd_transparent_hold_v4_2_all_results.zip",
        mime="application/zip",
        type="primary",
        key="download_all_v42",
    )
    labels = {
        "01_event_audit.csv": "1号：逐事件审查",
        "02_rule_report.csv": "2号：规则总表",
        "03_state_report.csv": "3号：柱体状态",
        "04_year_stability.csv": "4号：逐年稳定性",
        "05_verdicts.csv": "5号：预设门槛结论",
        "06_rule_definitions.csv": "6号：规则定义",
        "07_metadata.csv": "7号：运行信息",
    }
    columns = st.columns(4)
    for index, (name, data) in enumerate(result["files"].items()):
        with columns[index % 4]:
            st.download_button(
                labels[name], data=data, file_name=name, mime="text/csv", key=f"download_{name}"
            )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.info(
        "本版不寻找最优参数，也不加入快MACD。先用V4.1现成结果验证：标准周线MACD柱体的速度和加速度，"
        "能否提高九周持续概率、降低止损，同时保住C1、C2和八周大赢家。"
    )
    with st.expander("本版如何防止过拟合", expanded=False):
        st.markdown(
            """
            - 只验证5条预先写死的规则，不扫描阈值。
            - 每条规则都与同一检查周R0基准比较。
            - 同时检查覆盖率、C1/C2保留率和30%赢家保留率，避免只追求漂亮概率。
            - 必须查看逐年结果；单年好看不算稳定。
            - 本轮通过后，才值得加入逐日成交、交易成本并做真正的模拟持仓。
            """
        )
    with st.form("analysis_form"):
        uploaded = st.file_uploader(
            "上传V4.1的全部结果ZIP（推荐）或01_events.csv",
            type=["zip", "csv"],
            help="直接使用刚才全量科技股V4.1结果，无需重新下载行情。",
        )
        complete_8w_only = st.checkbox("只使用具有完整未来八周的样本", value=True)
        submitted = st.form_submit_button("开始透明验证", type="primary")

    if submitted:
        if uploaded is None:
            st.error("请先上传V4.1结果ZIP或01_events.csv。")
        else:
            try:
                raw, source_name = load_events(uploaded)
                validate_events(raw)
                events = prepare_events(raw, complete_8w_only)
                if events.empty:
                    raise ValueError("筛选后没有第一根红柱事件。")
                st.session_state["v42_result"] = build_outputs(events, source_name, complete_8w_only)
                st.session_state["v42_source"] = source_name
            except Exception as exc:
                st.exception(exc)

    if "v42_result" in st.session_state:
        render_results(st.session_state["v42_result"])
    else:
        st.caption("上传后通常数秒即可完成。本程序不会访问行情接口。")


if __name__ == "__main__":
    main()
