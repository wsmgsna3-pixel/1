from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


TITLE = "周线MACD动态退出成交验证器 V4.3"
CHECKPOINTS = (2, 3, 4, 5)
TARGETS = (10, 20, 30)


@dataclass(frozen=True)
class Policy:
    code: str
    name: str
    explanation: str


POLICIES = (
    Policy("P0", "基准：固定止盈/止损/八周", "不使用MACD动态退出。"),
    Policy(
        "P1", "W2缩短立即退出",
        "W2红柱不扩张即全部退出；W3同样要求扩张；W4/W5只排明确衰弱。",
    ),
    Policy(
        "P2", "W2缩短减仓一半",
        "W2翻绿全部退出；W2首次缩短卖一半；W3不扩张退出余仓；W4/W5只排明确衰弱。",
    ),
    Policy(
        "P3", "W2缩短观察至W3",
        "W2只有翻绿才退出；W3若不扩张则退出；W4/W5只排明确衰弱。",
    ),
    Policy(
        "P4", "全程C1友好",
        "W2翻绿退出；W3—W5允许首次缩短和再扩张，只排翻绿或连续明确衰弱。",
    ),
    Policy("P5", "只在翻绿时退出", "W2—W5只要标准MACD柱翻绿，下一交易日开盘全部退出。"),
)


BASE_REQUIRED = {
    "Event_Type", "Cycle_ID", "Signal_Date", "Entry_Date", "Entry_Price",
    "Tradable", "Has_8W_Future", "Cycle_Type", "Red_Cycle_Weeks", "Hist",
}


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "是"}


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def date8(value: Any) -> str:
    if pd.isna(value):
        return ""
    digits = "".join(ch for ch in str(value).split(".")[0] if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def read_csv(raw: bytes) -> pd.DataFrame:
    error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False)
        except Exception as exc:
            error = exc
    raise ValueError(f"CSV无法读取：{error}")


def load_events(uploaded: Any) -> tuple[pd.DataFrame, str]:
    raw = uploaded.getvalue()
    if uploaded.name.lower().endswith(".csv"):
        return read_csv(raw), uploaded.name
    if not uploaded.name.lower().endswith(".zip"):
        raise ValueError("请上传V4.1全部结果ZIP或原始01_events.csv。")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        candidates = [n for n in names if Path(n).name.lower() == "01_events.csv"]
        if not candidates:
            candidates = [n for n in names if "events" in Path(n).name.lower() and n.endswith(".csv")]
        if not candidates:
            raise ValueError("ZIP中没有找到V4.1的01_events.csv。不要上传V4.2结果ZIP。")
        target = sorted(candidates, key=len)[0]
        return read_csv(archive.read(target)), f"{uploaded.name}/{target}"


def validate(frame: pd.DataFrame, targets: tuple[int, ...]) -> None:
    required = set(BASE_REQUIRED)
    for week in CHECKPOINTS:
        required.update({
            f"CP_W{week}_Observed", f"CP_W{week}_Hist",
            f"CP_W{week}_Delayed_Entry_Date", f"CP_W{week}_Delayed_Entry_Price",
        })
    for target in targets:
        required.update({
            f"Exit_T{target}_Date", f"Exit_T{target}_Price",
            f"Exit_T{target}_Return_pct", f"Exit_T{target}_Holding_Days",
            f"Exit_T{target}_Reason",
        })
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("输入不是V4.1原始事件表，缺少字段：" + "、".join(missing))


def velocity_features(row: pd.Series, week: int) -> dict[str, Any]:
    hist = [number(row.get("Hist"))]
    hist.extend(number(row.get(f"CP_W{k}_Hist")) for k in range(2, week + 1))
    observed = to_bool(row.get(f"CP_W{week}_Observed")) and math.isfinite(hist[-1])
    velocity = hist[-1] - hist[-2] if observed and math.isfinite(hist[-2]) else np.nan
    previous_velocity = (
        hist[-2] - hist[-3]
        if week >= 3 and all(math.isfinite(v) for v in hist[-3:-1]) else np.nan
    )
    acceleration = velocity - previous_velocity if math.isfinite(previous_velocity) else np.nan
    shrink_streak = 0
    for i in range(len(hist) - 1, 0, -1):
        if math.isfinite(hist[i]) and math.isfinite(hist[i - 1]) and hist[i] < hist[i - 1]:
            shrink_streak += 1
        else:
            break
    red = observed and hist[-1] > 0
    expanding = red and velocity > 0
    clearly_weak = red and shrink_streak >= 2 and velocity < 0 and (
        not math.isfinite(acceleration) or acceleration <= 0
    )
    c1_friendly = red and not clearly_weak
    return {
        "observed": observed, "hist": hist[-1], "velocity": velocity,
        "acceleration": acceleration, "shrink_streak": shrink_streak,
        "red": red, "expanding": expanding, "c1_friendly": c1_friendly,
    }


def policy_action(policy_code: str, week: int, feature: dict[str, Any]) -> tuple[float, str]:
    """返回本次卖出占当前剩余仓位的比例和原因。"""
    if policy_code == "P0" or not feature["observed"]:
        return 0.0, ""
    if policy_code == "P5":
        return (1.0, "MACD翻绿") if not feature["red"] else (0.0, "")
    if policy_code == "P4":
        keep = feature["red"] if week == 2 else feature["c1_friendly"]
        return (0.0, "") if keep else (1.0, "C1友好规则退出")
    if policy_code == "P1":
        keep = feature["expanding"] if week in (2, 3) else feature["c1_friendly"]
        return (0.0, "") if keep else (1.0, "动态衰弱退出")
    if policy_code == "P3":
        keep = feature["red"] if week == 2 else (
            feature["expanding"] if week == 3 else feature["c1_friendly"]
        )
        return (0.0, "") if keep else (1.0, "观察后衰弱退出")
    if policy_code == "P2":
        if week == 2:
            if not feature["red"]:
                return 1.0, "W2翻绿退出"
            if not feature["expanding"]:
                return 0.5, "W2首次缩短减半"
            return 0.0, ""
        keep = feature["expanding"] if week == 3 else feature["c1_friendly"]
        return (0.0, "") if keep else (1.0, "余仓动态退出")
    raise ValueError(f"未知策略：{policy_code}")


def scheduled_exit_return(
    row: pd.Series, week: int, source_buy_slippage_pct: float, sell_slippage_pct: float,
) -> tuple[str, float, float]:
    scheduled_date = date8(row.get(f"CP_W{week}_Delayed_Entry_Date"))
    delayed_buy_price = number(row.get(f"CP_W{week}_Delayed_Entry_Price"))
    entry_price = number(row.get("Entry_Price"))
    if not scheduled_date or not math.isfinite(delayed_buy_price) or not math.isfinite(entry_price):
        return "", np.nan, np.nan
    raw_open = delayed_buy_price / (1.0 + source_buy_slippage_pct / 100.0)
    sell_price = raw_open * (1.0 - sell_slippage_pct / 100.0)
    return scheduled_date, sell_price, (sell_price / entry_price - 1.0) * 100.0


def simulate_one(
    row: pd.Series, target: int, policy: Policy,
    source_buy_slippage_pct: float, sell_slippage_pct: float,
) -> dict[str, Any]:
    base_date = date8(row.get(f"Exit_T{target}_Date"))
    base_return = number(row.get(f"Exit_T{target}_Return_pct"))
    base_reason = str(row.get(f"Exit_T{target}_Reason", ""))
    base_holding = number(row.get(f"Exit_T{target}_Holding_Days"))
    if not base_date or not math.isfinite(base_return):
        return {"Comparable": False}

    remaining = 1.0
    total_return = 0.0
    weighted_days = 0.0
    macd_weight = 0.0
    stop_weight = 0.0
    target_weight = 0.0
    pieces: list[str] = []
    final_date = base_date
    final_reason = base_reason

    if policy.code != "P0":
        for week in CHECKPOINTS:
            feature = velocity_features(row, week)
            fraction, reason = policy_action(policy.code, week, feature)
            if fraction <= 0 or remaining <= 1e-12:
                continue
            scheduled_date, _, scheduled_return = scheduled_exit_return(
                row, week, source_buy_slippage_pct, sell_slippage_pct
            )
            if not scheduled_date or not math.isfinite(scheduled_return):
                return {"Comparable": False}
            # 同日时，周线决定在开盘执行；优先于当天盘中的止盈/止损。
            if base_date and base_date < scheduled_date:
                break
            sold_weight = remaining * fraction
            total_return += sold_weight * scheduled_return
            weighted_days += sold_weight * (5 * (week - 1) + 1)
            macd_weight += sold_weight
            remaining -= sold_weight
            pieces.append(f"W{week}:{reason}:{sold_weight:.2f}")
            final_date, final_reason = scheduled_date, f"W{week}_{reason}"

    if remaining > 1e-12:
        total_return += remaining * base_return
        weighted_days += remaining * base_holding
        if "止损" in base_reason:
            stop_weight += remaining
        elif "止盈" in base_reason:
            target_weight += remaining
        pieces.append(f"基准退出:{base_reason}:{remaining:.2f}")
        if base_date >= final_date:
            final_date, final_reason = base_date, base_reason

    return {
        "Comparable": True,
        "Strategy_Return_pct": total_return,
        "Weighted_Holding_Days": weighted_days,
        "MACD_Exit_Weight": macd_weight,
        "Stop_Exit_Weight": stop_weight,
        "Target_Exit_Weight": target_weight,
        "Any_Stop": stop_weight > 1e-12,
        "Any_MACD_Exit": macd_weight > 1e-12,
        "Final_Exit_Date": final_date,
        "Final_Exit_Reason": final_reason,
        "Exit_Pieces": "|".join(pieces),
    }


def common_sample(raw: pd.DataFrame, targets: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = raw[raw["Event_Type"].astype(str).eq("第一根红柱")].copy()
    first = first[first["Tradable"].map(to_bool) & first["Has_8W_Future"].map(to_bool)]
    first = first.drop_duplicates("Cycle_ID").reset_index(drop=True)
    reasons = []
    for _, row in first.iterrows():
        reason = ""
        for week in CHECKPOINTS:
            if not date8(row.get(f"CP_W{week}_Delayed_Entry_Date")) or not math.isfinite(
                number(row.get(f"CP_W{week}_Delayed_Entry_Price"))
            ):
                reason = f"缺少W{week}下一交易日开盘价"
                break
        if not reason:
            for target in targets:
                if not date8(row.get(f"Exit_T{target}_Date")) or not math.isfinite(
                    number(row.get(f"Exit_T{target}_Return_pct"))
                ):
                    reason = f"缺少T{target}基准退出"
                    break
        reasons.append(reason)
    first["V43_Exclusion_Reason"] = reasons
    excluded = first[first["V43_Exclusion_Reason"].ne("")].copy()
    included = first[first["V43_Exclusion_Reason"].eq("")].copy().reset_index(drop=True)
    included["Signal_Year"] = pd.to_numeric(
        included["Signal_Date"].astype(str).str.replace(r"\D", "", regex=True).str[:4], errors="coerce"
    ).astype("Int64")
    return included, excluded


def run_simulation(
    events: pd.DataFrame, targets: tuple[int, ...],
    source_buy_slippage_pct: float, sell_slippage_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identity = [
        "Cycle_ID", "ts_code", "name", "Sample_Board", "SW_L1", "SW_L2", "SW_L3",
        "Signal_Date", "Signal_Year", "Entry_Date", "Entry_Price", "Cycle_Type", "Red_Cycle_Weeks",
    ]
    for _, event in events.iterrows():
        base = {column: event.get(column, "") for column in identity}
        for target in targets:
            for policy in POLICIES:
                result = simulate_one(
                    event, target, policy, source_buy_slippage_pct, sell_slippage_pct
                )
                if result.get("Comparable"):
                    rows.append({
                        **base, "Target_pct": target, "Policy_Code": policy.code,
                        "Policy": policy.name, **result,
                    })
    return pd.DataFrame(rows)


def cvar(series: pd.Series, quantile: float = 0.10) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    cutoff = clean.quantile(quantile)
    return float(clean[clean <= cutoff].mean())


def summary_row(group: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(group["Strategy_Return_pct"], errors="coerce")
    return {
        "交易数": len(group),
        "平均收益(%)": returns.mean(),
        "收益中位数(%)": returns.median(),
        "胜率(%)": (returns > 0).mean() * 100.0,
        "盈利≥10%(%)": (returns >= 10).mean() * 100.0,
        "盈利≥20%(%)": (returns >= 20).mean() * 100.0,
        "亏损≤-10%(%)": (returns <= -10).mean() * 100.0,
        "最差10%平均收益(%)": cvar(returns),
        "平均持有交易日": pd.to_numeric(group["Weighted_Holding_Days"], errors="coerce").mean(),
        "MACD退出资金占比(%)": pd.to_numeric(group["MACD_Exit_Weight"], errors="coerce").mean() * 100.0,
        "止损退出资金占比(%)": pd.to_numeric(group["Stop_Exit_Weight"], errors="coerce").mean() * 100.0,
        "止盈退出资金占比(%)": pd.to_numeric(group["Target_Exit_Weight"], errors="coerce").mean() * 100.0,
        "发生MACD退出的交易(%)": group["Any_MACD_Exit"].map(to_bool).mean() * 100.0,
        "仍有仓位触发止损的交易(%)": group["Any_Stop"].map(to_bool).mean() * 100.0,
    }


def build_summary(trades: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in trades.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(groups, keys)), **summary_row(group)})
    return pd.DataFrame(rows)


def add_baseline_delta(summary: pd.DataFrame, scope: list[str]) -> pd.DataFrame:
    frame = summary.copy()
    metrics = [
        "平均收益(%)", "收益中位数(%)", "胜率(%)", "亏损≤-10%(%)",
        "最差10%平均收益(%)", "平均持有交易日", "止损退出资金占比(%)",
    ]
    baseline = frame[frame["Policy_Code"].eq("P0")][scope + metrics].copy()
    baseline = baseline.rename(columns={metric: f"__base_{metric}" for metric in metrics})
    frame = frame.merge(baseline, on=scope, how="left")
    for metric in metrics:
        frame[f"较基准_{metric}"] = frame[metric] - frame[f"__base_{metric}"]
    return frame.drop(columns=[f"__base_{metric}" for metric in metrics])


def build_reports(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    total = add_baseline_delta(
        build_summary(trades, ["Target_pct", "Policy_Code", "Policy"]), ["Target_pct"]
    )
    yearly = add_baseline_delta(
        build_summary(trades, ["Signal_Year", "Target_pct", "Policy_Code", "Policy"]),
        ["Signal_Year", "Target_pct"],
    )
    board = add_baseline_delta(
        build_summary(trades, ["Sample_Board", "Target_pct", "Policy_Code", "Policy"]),
        ["Sample_Board", "Target_pct"],
    )
    reasons = (
        trades.groupby(["Target_pct", "Policy_Code", "Policy", "Final_Exit_Reason"], dropna=False)
        .size().rename("交易数").reset_index()
    )
    definitions = pd.DataFrame(
        [{"策略编号": p.code, "策略": p.name, "定义": p.explanation} for p in POLICIES]
    )
    return {"total": total, "yearly": yearly, "board": board, "reasons": reasons, "definitions": definitions}


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def build_result(
    raw: pd.DataFrame, source: str, targets: tuple[int, ...],
    source_buy_slippage_pct: float, sell_slippage_pct: float,
) -> dict[str, Any]:
    events, excluded = common_sample(raw, targets)
    if events.empty:
        raise ValueError("共同可比较样本为空。")
    trades = run_simulation(events, targets, source_buy_slippage_pct, sell_slippage_pct)
    reports = build_reports(trades)
    exclusion_summary = (
        excluded.groupby("V43_Exclusion_Reason").size().rename("事件数").reset_index()
        if not excluded.empty else pd.DataFrame(columns=["V43_Exclusion_Reason", "事件数"])
    )
    metadata = pd.DataFrame([
        {"项目": "程序", "值": TITLE},
        {"项目": "输入", "值": source},
        {"项目": "生成时间", "值": datetime.now().isoformat(timespec="seconds")},
        {"项目": "第一根红柱完整八周事件", "值": len(events) + len(excluded)},
        {"项目": "共同可比较事件", "值": len(events)},
        {"项目": "因W2-W5开盘价不完整而剔除", "值": len(excluded)},
        {"项目": "目标止盈", "值": ",".join(f"+{t}%" for t in targets)},
        {"项目": "原V4.1买入滑点(%)", "值": source_buy_slippage_pct},
        {"项目": "本次动态卖出滑点(%)", "值": sell_slippage_pct},
        {"项目": "成交顺序", "值": "周末确认，下一交易日开盘退出；同日早于盘中止盈止损"},
        {"项目": "限制", "值": "逐事件等权，不模拟最多三仓的资金占用和信号冲突"},
    ])
    files = {
        "01_trade_ledger.csv": csv_bytes(trades),
        "02_strategy_summary.csv": csv_bytes(reports["total"]),
        "03_year_stability.csv": csv_bytes(reports["yearly"]),
        "04_board_stability.csv": csv_bytes(reports["board"]),
        "05_exit_reasons.csv": csv_bytes(reports["reasons"]),
        "06_policy_definitions.csv": csv_bytes(reports["definitions"]),
        "07_exclusion_summary.csv": csv_bytes(exclusion_summary),
        "08_metadata.csv": csv_bytes(metadata),
    }
    return {
        "events": events, "excluded": excluded, "trades": trades, **reports,
        "exclusion_summary": exclusion_summary, "metadata": metadata,
        "files": files, "zip": make_zip(files),
    }


def show_frame(frame: pd.DataFrame) -> None:
    formats = {
        column: "{:.2f}" for column in frame.columns
        if "(%)" in column or "交易日" in column
    }
    st.dataframe(frame.style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)


def render(result: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("原始完整八周事件", f"{len(result['events']) + len(result['excluded']):,}")
    c2.metric("共同可比较事件", f"{len(result['events']):,}")
    c3.metric("剔除事件", f"{len(result['excluded']):,}")
    c4.metric("模拟交易明细", f"{len(result['trades']):,}")
    st.subheader("策略总比较")
    st.caption("正的收益变化代表优于同一止盈目标下的P0；负的亏损率变化代表风险降低。")
    show_frame(result["total"])
    st.subheader("逐年稳定性")
    show_frame(result["yearly"])
    st.subheader("主板/创业板/科创板")
    show_frame(result["board"])
    with st.expander("退出原因与剔除原因"):
        show_frame(result["reasons"])
        show_frame(result["exclusion_summary"])
    st.subheader("策略定义")
    st.dataframe(result["definitions"], use_container_width=True, hide_index=True)
    st.subheader("下载")
    st.download_button(
        "下载全部结果ZIP", result["zip"],
        file_name="weekly_macd_dynamic_exit_v4_3_all_results.zip",
        mime="application/zip", type="primary", key="v43_all", on_click="ignore",
    )
    labels = [
        "1号：逐笔成交", "2号：策略总表", "3号：逐年", "4号：分板块",
        "5号：退出原因", "6号：策略定义", "7号：剔除审计", "8号：运行信息",
    ]
    columns = st.columns(4)
    for index, (name, data) in enumerate(result["files"].items()):
        with columns[index % 4]:
            st.download_button(
                labels[index], data, file_name=name, mime="text/csv",
                key=f"v43_{name}", on_click="ignore",
            )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.info(
        "本版使用V4.1已经保存的真实下一交易日开盘价，将周线决定推迟到下一交易日开盘执行；"
        "固定止盈和-10%止损仍按日线先后执行。重点比较W2缩短立即退出、减半和观察至W3。"
    )
    with st.expander("成交口径", expanded=False):
        st.markdown(
            """
            - 第一根红柱后的原始买点与V4.1完全相同。
            - 周线状态只能在该周收盘后确认，动态卖出使用下一交易日开盘价。
            - 动态退出价计入卖出滑点；固定止盈/止损直接沿用V4.1已计算的成交结果。
            - 若动态退出与止盈/止损发生在同一交易日，开盘动态退出先于盘中触发。
            - 所有策略使用完全相同的共同样本；结果是逐事件等权，尚未处理最多三仓和信号冲突。
            """
        )
    with st.form("v43_form"):
        upload = st.file_uploader(
            "上传V4.1全部结果ZIP或V4.1的01_events.csv",
            type=["zip", "csv"],
            help="必须是V4.1原始文件；V4.2结果已删除成交价字段，不能用于本模拟。",
        )
        selected = st.multiselect(
            "统一止盈目标", options=list(TARGETS), default=list(TARGETS),
            format_func=lambda x: f"+{x}%",
        )
        c1, c2 = st.columns(2)
        source_buy_slip = c1.number_input(
            "原V4.1买入滑点(%)", 0.0, 2.0, 0.20, 0.05,
            help="必须与生成V4.1结果时的设置一致，默认0.20%。",
        )
        sell_slip = c2.number_input("动态卖出滑点(%)", 0.0, 2.0, 0.20, 0.05)
        submitted = st.form_submit_button("开始动态退出模拟", type="primary")
    if submitted:
        if upload is None:
            st.error("请先上传V4.1结果。")
        elif not selected:
            st.error("至少选择一个止盈目标。")
        else:
            try:
                raw, source = load_events(upload)
                targets = tuple(sorted(set(int(v) for v in selected)))
                validate(raw, targets)
                st.session_state["v43_result"] = build_result(
                    raw, source, targets, float(source_buy_slip), float(sell_slip)
                )
            except Exception as exc:
                st.exception(exc)
    if "v43_result" in st.session_state:
        render(st.session_state["v43_result"])
    else:
        st.caption("使用现成结果，通常数秒完成，不需要Tushare，也不重新下载行情。")


if __name__ == "__main__":
    main()
