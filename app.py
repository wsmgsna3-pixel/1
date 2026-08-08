# -*- coding: utf-8 -*-
"""
4—8周路径概率研究器 V6.0
========================

定位
----
本程序不是一个承诺盈利的实时选股器，而是把 V5 生成的多个年度
``weekly_macd_events_*.csv`` 合并后，进行严格的逐年滚动概率验证。

它不再把周线 MACD 当作唯一依据。MACD 只是一组输入特征；模型同时使用：

1. 大盘、板块和行业趋势；
2. 市场、板块和行业广度；
3. 股票中长期趋势、相对强度和回调结构；
4. 市值、换手率、同周信号拥挤度；
5. MACD 第一根红柱当时的强度、零轴位置等。

预测目标
--------
以 -15% 止损为固定研究口径，分别估计：

* 前10个交易日先触发止损的概率；
* 八周内 +20% 先于 -15% 到达的路径概率；
* 八周内 +30% 先于 -15% 到达的路径概率；
* 八周始终没有触发 -15% 止损的概率；
* 到第4周仍可观察的事件，未来四周上涨、再涨10%、再跌10%的概率。

严格验证
--------
* 测试年份永远只使用更早年份训练；
* 测试年前留出56天隔离带，防止八周标签重叠；
* 每个测试周按预测期望收益最多选择3只；
* 输出概率校准、路径指标、每周前三和逐事件预测；
* 不使用股票代码和名称作为特征，避免模型记忆个股；
* 明确排除所有未来收益、止盈止损结果和事后周期分类字段。

建议运行
--------
图形界面：
    streamlit run weekly_path_probability_v6.py

命令行：
    python weekly_path_probability_v6.py --input file1.csv file2.csv file3.csv \
        --output-dir v6_outputs --models logit hgb

依赖：
    pip install pandas numpy scikit-learn streamlit
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  # 命令行模式不强制安装 Streamlit
    st = None

SKLEARN_IMPORT_ERROR: ImportError | None = None
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:  # Streamlit Cloud遗漏requirements.txt时给出明确提示
    SKLEARN_IMPORT_ERROR = exc


VERSION = "V6.0"
DEFAULT_STOP_PCT = 15
DEFAULT_MAX_PICKS = 3
DEFAULT_EMBARGO_DAYS = 56
DEFAULT_MIN_TRAIN_YEARS = 2
DEFAULT_MIN_TRAIN_SAMPLES = 80
RANDOM_STATE = 20260808

PATH_CLASSES = ("STOP", "TARGET", "TIMEOUT")


NUMERIC_FEATURE_CANDIDATES = [
    # 股票趋势与 MACD：全部是信号周结束时已知的数据
    "Hist", "Hist_Prev", "DIF", "DEA", "Stock_Dist_MA20_pct",
    "W_MA20_Slope4_pct", "Pre_13W_Return_pct", "Pre_26W_Return_pct",
    "Green_Weeks", "Pullback_Depth_pct", "Initial_Red_Strength",
    "PreGreen_Abs_Peak_Hist", "Raw_Close", "Circ_MV_Billion",
    "Turnover_Rate", "D1_Open_Gap_From_Weekly_Close_pct",
    # 相对强度
    "Stock_Excess_vs_Board_13W_pct", "Stock_Excess_vs_Board_26W_pct",
    "Stock_Excess_vs_Industry_13W_pct", "Stock_Excess_vs_Industry_26W_pct",
    # 宽基环境
    "Broad_Index_Return_13W_pct", "Broad_Index_Return_26W_pct",
    "Broad_Index_MA20_Slope4_pct", "Broad_Index_Dist_MA20_pct",
    "Broad_Index_MACD_Hist",
    # 板块环境
    "Board_Index_Return_13W_pct", "Board_Index_Return_26W_pct",
    "Board_Index_MA20_Slope4_pct", "Board_Index_Dist_MA20_pct",
    "Board_Index_MACD_Hist",
    # 行业环境
    "Industry_Index_Return_13W_pct", "Industry_Index_Return_26W_pct",
    "Industry_Index_MA20_Slope4_pct", "Industry_Index_Dist_MA20_pct",
    "Industry_Index_MACD_Hist",
    # 广度
    "Market_Breadth_Above_MA20_pct", "Market_Breadth_Uptrend_pct",
    "Market_Breadth_First_Red_pct", "Market_Breadth_Median_13W_Return_pct",
    "Market_Breadth_Median_26W_Return_pct",
    "Board_Breadth_Above_MA20_pct", "Board_Breadth_Uptrend_pct",
    "Board_Breadth_First_Red_pct", "Board_Breadth_Median_13W_Return_pct",
    "Board_Breadth_Median_26W_Return_pct",
    "Industry_Breadth_Above_MA20_pct", "Industry_Breadth_Uptrend_pct",
    "Industry_Breadth_First_Red_pct", "Industry_Breadth_Median_13W_Return_pct",
    "Industry_Breadth_Median_26W_Return_pct",
    # 同周拥挤度
    "First_Red_Signals_Market_Week", "First_Red_Signals_Board_Week",
    "First_Red_Signals_Industry_Week",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "Sample_Board", "SW_L1", "Weekly_Trend", "Zero_Axis",
    "Broad_Index_Trend", "Broad_Index_Zero_Axis",
    "Board_Index_Trend", "Board_Index_Zero_Axis",
    "Industry_Index_Trend", "Industry_Index_Zero_Axis",
]

W4_NUMERIC_FEATURE_CANDIDATES = NUMERIC_FEATURE_CANDIDATES + [
    "CP_W4_Hist", "CP_W4_Hist_vs_W1_pct", "CP_W4_Peak_to_PreGreen_Ratio",
    "CP_W4_Material_Shrink_Count", "CP_W4_ReExpansion_Count",
    "CP_W4_Return_From_Entry_pct",
]

W4_CATEGORICAL_FEATURE_CANDIDATES = CATEGORICAL_FEATURE_CANDIDATES + [
    "CP_W4_State",
]


@dataclass(frozen=True)
class Endpoint:
    name: str
    label_column: str
    kind: str  # binary | multiclass
    positive_class: str | int | None = None


ENTRY_ENDPOINTS = (
    Endpoint("stop_2w", "Label_Stop_2W", "binary", 1),
    Endpoint("survive_8w", "Label_Survive_8W", "binary", 1),
    Endpoint("path20_8w", "Label_Path20", "multiclass", "TARGET"),
    Endpoint("path30_8w", "Label_Path30", "multiclass", "TARGET"),
)

W4_ENDPOINTS = (
    Endpoint("w4_positive", "Label_W4_Remaining_Positive", "binary", 1),
    Endpoint("w4_gain10", "Label_W4_Remaining_Gain10", "binary", 1),
    Endpoint("w4_drawdown10", "Label_W4_Remaining_Drawdown10", "binary", 1),
)


def normalize_bool(series: pd.Series) -> pd.Series:
    """兼容 CSV 中的 True/False、1/0 和中英文布尔文本。"""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    mapping = {
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False, "是": True, "否": False,
    }
    text = series.astype(str).str.strip().str.lower()
    out = text.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    out = out.where(out.notna(), numeric.map({1.0: True, 0.0: False}))
    return out.astype("boolean")


def normalize_date_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def reason_to_path(reason: Any) -> str | float:
    text = str(reason)
    if "止盈" in text:
        return "TARGET"
    if "止损" in text:
        return "STOP"
    if "到期" in text:
        return "TIMEOUT"
    return np.nan


def read_event_files(paths: Sequence[str | os.PathLike[str]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path)
        frame = pd.read_csv(path, low_memory=False)
        frame["Source_File"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError("没有读取到CSV文件")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def validate_input_columns(frame: pd.DataFrame, stop_pct: int) -> list[str]:
    required = [
        "Signal_Date", "ts_code", "Event_Type", "Tradable", "Has_8W_Future",
        f"V5_S{stop_pct}_Survived_8W",
        f"V5_S{stop_pct}_T20_Fixed_Reason",
        f"V5_S{stop_pct}_T20_Fixed_Return_pct",
        f"V5_S{stop_pct}_T30_Fixed_Reason",
        f"V5_S{stop_pct}_T30_Fixed_Return_pct",
        f"V5_S{stop_pct}_T30_Fixed_Holding_Days",
    ]
    return [column for column in required if column not in frame.columns]


def prepare_entry_dataset(frame: pd.DataFrame, stop_pct: int = 15) -> pd.DataFrame:
    """从V5事件明细生成买入日可用特征与路径标签。"""
    missing = validate_input_columns(frame, stop_pct)
    if missing:
        raise ValueError(
            "输入文件不是完整的V5事件明细，缺少字段：" + "、".join(missing)
            + "。请上传 weekly_macd_events_*.csv，而不是仅上传v5_selected文件。"
        )

    work = frame.copy()
    work["Signal_DT"] = normalize_date_series(work["Signal_Date"])
    work["Tradable_Bool"] = normalize_bool(work["Tradable"])
    work["Future_Bool"] = normalize_bool(work["Has_8W_Future"])
    work = work[
        work["Event_Type"].astype(str).eq("第一根红柱")
        & work["Tradable_Bool"].eq(True)
        & work["Future_Bool"].eq(True)
        & work["Signal_DT"].notna()
    ].copy()

    # 相同事件跨文件重复时只保留一条；不使用股票代码作为模型特征。
    dedup_columns = [column for column in [
        "Signal_Date", "ts_code", "Entry_Date", "Event_Type"
    ] if column in work.columns]
    work = work.sort_values(["Signal_DT", "ts_code"]).drop_duplicates(
        dedup_columns, keep="last"
    )

    prefix = f"V5_S{stop_pct}"
    reason20 = work[f"{prefix}_T20_Fixed_Reason"]
    reason30 = work[f"{prefix}_T30_Fixed_Reason"]
    holding30 = pd.to_numeric(
        work[f"{prefix}_T30_Fixed_Holding_Days"], errors="coerce"
    )
    work["Label_Path20"] = reason20.map(reason_to_path)
    work["Label_Path30"] = reason30.map(reason_to_path)
    work["Label_Stop_2W"] = (
        reason30.astype(str).str.contains("止损", na=False) & holding30.le(10)
    ).astype(int)
    work["Label_Survive_8W"] = normalize_bool(
        work[f"{prefix}_Survived_8W"]
    ).astype("Int64")
    work["Actual_Return_T20"] = pd.to_numeric(
        work[f"{prefix}_T20_Fixed_Return_pct"], errors="coerce"
    )
    work["Actual_Return_T30"] = pd.to_numeric(
        work[f"{prefix}_T30_Fixed_Return_pct"], errors="coerce"
    )
    work["Signal_Year"] = work["Signal_DT"].dt.year.astype(int)

    complete = (
        work["Label_Path20"].notna()
        & work["Label_Path30"].notna()
        & work["Label_Survive_8W"].notna()
        & work["Actual_Return_T30"].notna()
    )
    return work[complete].reset_index(drop=True)


def prepare_week4_dataset(entry: pd.DataFrame) -> pd.DataFrame:
    """
    生成第4周动态复评数据。

    V5 的 CP_W4_Stop_Hit_Before 使用当次V5界面设定的兼容止损阈值；
    因而此处只研究“当时仍可观察”的事件，不把它混同为固定-15%组合回测。
    """
    required = [
        "CP_W4_Observed", "CP_W4_Stop_Hit_Before",
        "CP_W4_Remaining_Return_pct", "CP_W4_Remaining_MFE_pct",
        "CP_W4_Remaining_MAE_pct",
    ]
    if any(column not in entry.columns for column in required):
        return pd.DataFrame()
    work = entry.copy()
    observed = normalize_bool(work["CP_W4_Observed"])
    stopped = normalize_bool(work["CP_W4_Stop_Hit_Before"])
    work = work[observed.eq(True) & stopped.eq(False)].copy()
    remaining_return = pd.to_numeric(work["CP_W4_Remaining_Return_pct"], errors="coerce")
    remaining_mfe = pd.to_numeric(work["CP_W4_Remaining_MFE_pct"], errors="coerce")
    remaining_mae = pd.to_numeric(work["CP_W4_Remaining_MAE_pct"], errors="coerce")
    work["Label_W4_Remaining_Positive"] = (remaining_return > 0).astype(int)
    work["Label_W4_Remaining_Gain10"] = (remaining_mfe >= 10).astype(int)
    work["Label_W4_Remaining_Drawdown10"] = (remaining_mae <= -10).astype(int)
    work["Actual_W4_Remaining_Return"] = remaining_return
    return work[
        remaining_return.notna() & remaining_mfe.notna() & remaining_mae.notna()
    ].reset_index(drop=True)


def available_features(
    frame: pd.DataFrame,
    numeric_candidates: Sequence[str],
    categorical_candidates: Sequence[str],
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    for column in numeric_candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() >= max(10, int(len(frame) * 0.05)) and values.nunique() > 1:
            frame[column] = values
            numeric.append(column)
    categorical: list[str] = []
    for column in categorical_candidates:
        if column not in frame.columns:
            continue
        values = frame[column].astype("string").fillna("缺失")
        if values.nunique() > 1:
            frame[column] = values
            categorical.append(column)
    if not numeric and not categorical:
        raise ValueError("没有足够的买入日特征可用于建模")
    return numeric, categorical


def make_preprocessor(numeric: Sequence[str], categorical: Sequence[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, list(numeric)))
    if categorical:
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore", min_frequency=3, sparse_output=False
            )),
        ])
        transformers.append(("cat", categorical_pipe, list(categorical)))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def make_estimator(model_name: str, kind: str) -> Any:
    if model_name == "logit":
        # 小样本优先强正则化和可解释基线。
        return LogisticRegression(
            C=0.20,
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=7,
            min_samples_leaf=15,
            l2_regularization=2.0,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"未知模型：{model_name}")


def make_pipeline(
    model_name: str,
    kind: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> Pipeline:
    return Pipeline([
        ("features", make_preprocessor(numeric, categorical)),
        ("model", make_estimator(model_name, kind)),
    ])


def safe_auc(y_true: pd.Series, probability: np.ndarray) -> float:
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, probability))


def multiclass_brier(y_true: pd.Series, probability: np.ndarray, classes: Sequence[Any]) -> float:
    class_to_position = {value: position for position, value in enumerate(classes)}
    encoded = np.zeros_like(probability, dtype=float)
    for row_no, value in enumerate(y_true):
        if value in class_to_position:
            encoded[row_no, class_to_position[value]] = 1.0
    return float(np.mean(np.sum((probability - encoded) ** 2, axis=1)))


def class_probability(
    probabilities: np.ndarray,
    classes: Sequence[Any],
    wanted: Any,
) -> np.ndarray:
    lookup = {value: position for position, value in enumerate(classes)}
    if wanted not in lookup:
        return np.full(len(probabilities), np.nan)
    return probabilities[:, lookup[wanted]]


def endpoint_is_trainable(y: pd.Series, kind: str, min_class_samples: int = 8) -> bool:
    counts = y.value_counts(dropna=True)
    if kind == "binary":
        return len(counts) == 2 and int(counts.min()) >= min_class_samples
    return set(PATH_CLASSES).issubset(set(counts.index)) and int(counts.min()) >= min_class_samples


def fit_predict_endpoint(
    train: pd.DataFrame,
    test: pd.DataFrame,
    endpoint: Endpoint,
    model_name: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    y_train = train[endpoint.label_column]
    y_test = test[endpoint.label_column]
    if not endpoint_is_trainable(y_train, endpoint.kind):
        raise ValueError(f"{endpoint.name}训练集中某类样本少于8条")

    pipeline = make_pipeline(model_name, endpoint.kind, numeric, categorical)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(train[list(numeric) + list(categorical)], y_train)
    probability = pipeline.predict_proba(test[list(numeric) + list(categorical)])
    classes = list(pipeline.named_steps["model"].classes_)
    prediction = pipeline.predict(test[list(numeric) + list(categorical)])

    output = pd.DataFrame(index=test.index)
    output[f"Pred_{endpoint.name}"] = class_probability(
        probability, classes, endpoint.positive_class
    )
    for class_value in classes:
        safe_name = str(class_value).replace(" ", "_")
        output[f"Pred_{endpoint.name}_{safe_name}"] = class_probability(
            probability, classes, class_value
        )

    if endpoint.kind == "binary":
        positive_probability = output[f"Pred_{endpoint.name}"].to_numpy(float)
        metrics = {
            "Brier": float(brier_score_loss(y_test.astype(int), positive_probability)),
            "ROC_AUC": safe_auc(y_test.astype(int), positive_probability),
            "LogLoss": float(log_loss(y_test, probability, labels=classes)),
            "Accuracy": float(accuracy_score(y_test, prediction)),
            "Actual_Rate": float(pd.to_numeric(y_test, errors="coerce").mean()),
            "Predicted_Rate": float(np.nanmean(positive_probability)),
        }
    else:
        metrics = {
            "Brier": multiclass_brier(y_test, probability, classes),
            "ROC_AUC": np.nan,
            "LogLoss": float(log_loss(y_test, probability, labels=classes)),
            "Accuracy": float(accuracy_score(y_test, prediction)),
            "Actual_Rate": float((y_test == endpoint.positive_class).mean()),
            "Predicted_Rate": float(np.nanmean(output[f"Pred_{endpoint.name}"])),
        }

    coefficient_rows: list[dict[str, Any]] = []
    if model_name == "logit":
        feature_names = pipeline.named_steps["features"].get_feature_names_out()
        coefficients = pipeline.named_steps["model"].coef_
        model_classes = pipeline.named_steps["model"].classes_
        for class_no, class_value in enumerate(model_classes):
            # 二分类coef只有一行，对应classes_[1]
            if coefficients.shape[0] == 1 and class_no == 0:
                continue
            row_no = 0 if coefficients.shape[0] == 1 else class_no
            for feature, coefficient in zip(feature_names, coefficients[row_no]):
                coefficient_rows.append({
                    "Endpoint": endpoint.name,
                    "Class": class_value,
                    "Feature": str(feature),
                    "Coefficient": float(coefficient),
                })
    return output, metrics, coefficient_rows


def payoff_map(train: pd.DataFrame, target: int) -> dict[str, float]:
    label = f"Label_Path{target}"
    returns = f"Actual_Return_T{target}"
    result: dict[str, float] = {}
    for class_value in PATH_CLASSES:
        values = pd.to_numeric(
            train.loc[train[label].eq(class_value), returns], errors="coerce"
        ).dropna()
        result[class_value] = float(values.mean()) if len(values) else 0.0
    return result


def add_expected_value(
    predictions: pd.DataFrame,
    payoffs: dict[str, float],
    target: int,
) -> pd.DataFrame:
    output = predictions.copy()
    expected = np.zeros(len(output), dtype=float)
    for class_value in PATH_CLASSES:
        column = f"Pred_path{target}_8w_{class_value}"
        if column not in output.columns:
            output[column] = np.nan
        expected += output[column].fillna(0.0).to_numpy(float) * payoffs[class_value]
    output[f"EV_T{target}"] = expected
    return output


def walk_forward_predictions(
    data: pd.DataFrame,
    endpoints: Sequence[Endpoint],
    model_names: Sequence[str],
    numeric: Sequence[str],
    categorical: Sequence[str],
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    min_train_samples: int = DEFAULT_MIN_TRAIN_SAMPLES,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = sorted(data["Signal_Year"].dropna().astype(int).unique())
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for test_year in years:
        earlier_years = [year for year in years if year < test_year]
        if len(earlier_years) < min_train_years:
            continue
        test_start = pd.Timestamp(year=test_year, month=1, day=1)
        cutoff = test_start - pd.Timedelta(days=embargo_days)
        train = data[data["Signal_DT"] < cutoff].copy()
        test = data[data["Signal_Year"].eq(test_year)].copy()
        if len(train) < min_train_samples or test.empty:
            continue

        fold_rows.append({
            "Test_Year": test_year,
            "Train_Start": train["Signal_DT"].min(),
            "Train_End": train["Signal_DT"].max(),
            "Train_Samples": len(train),
            "Test_Samples": len(test),
            "Embargo_Days": embargo_days,
        })
        for model_name in model_names:
            endpoint_labels = [
                endpoint.label_column for endpoint in endpoints
                if endpoint.label_column in test.columns
            ]
            base_columns = list(dict.fromkeys(
                column for column in [
                    "Signal_Date", "Signal_DT", "Signal_Year", "ts_code", "name",
                    "Sample_Board", "SW_L1", "Weekly_Trend", "Source_File",
                    "Label_Stop_2W", "Label_Survive_8W", "Label_Path20",
                    "Label_Path30", "Actual_Return_T20", "Actual_Return_T30",
                    "Actual_W4_Remaining_Return",
                    *endpoint_labels,
                ] if column in test.columns
            ))
            base = test[base_columns].copy()
            base["Model"] = model_name
            base["Test_Year"] = test_year
            successful = 0
            for endpoint in endpoints:
                if endpoint.label_column not in train.columns:
                    continue
                try:
                    endpoint_prediction, metrics, coefficients = fit_predict_endpoint(
                        train, test, endpoint, model_name, numeric, categorical
                    )
                except ValueError as exc:
                    metric_rows.append({
                        "Test_Year": test_year, "Model": model_name,
                        "Endpoint": endpoint.name, "Status": str(exc),
                    })
                    continue
                base = base.join(endpoint_prediction)
                successful += 1
                metric_rows.append({
                    "Test_Year": test_year,
                    "Model": model_name,
                    "Endpoint": endpoint.name,
                    "Train_Samples": len(train),
                    "Test_Samples": len(test),
                    "Status": "OK",
                    **metrics,
                })
                for row in coefficients:
                    row.update({"Test_Year": test_year, "Model": model_name})
                    coefficient_rows.append(row)
            if successful:
                if all(f"Pred_path30_8w_{value}" in base.columns for value in PATH_CLASSES):
                    base = add_expected_value(base, payoff_map(train, 30), 30)
                if all(f"Pred_path20_8w_{value}" in base.columns for value in PATH_CLASSES):
                    base = add_expected_value(base, payoff_map(train, 20), 20)
                prediction_frames.append(base)

    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) \
        if prediction_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    folds = pd.DataFrame(fold_rows).drop_duplicates() if fold_rows else pd.DataFrame()
    return predictions, metrics, coefficients, folds


def select_weekly_top3(
    predictions: pd.DataFrame,
    max_picks: int = DEFAULT_MAX_PICKS,
    require_positive_ev: bool = True,
) -> pd.DataFrame:
    if predictions.empty or "EV_T30" not in predictions.columns:
        return pd.DataFrame()
    work = predictions.copy()
    if require_positive_ev:
        work = work[work["EV_T30"] > 0].copy()
    if work.empty:
        return work
    work = work.sort_values(
        ["Model", "Signal_DT", "EV_T30", "Pred_survive_8w", "ts_code"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    work["Weekly_Rank"] = work.groupby(["Model", "Signal_Date"]).cumcount() + 1
    return work[work["Weekly_Rank"] <= max_picks].reset_index(drop=True)


def top3_report(
    all_predictions: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    if all_predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (model, test_year), candidates in all_predictions.groupby(["Model", "Test_Year"]):
        picks = selected[
            selected["Model"].eq(model) & selected["Test_Year"].eq(test_year)
        ]
        candidate_weeks = candidates["Signal_Date"].nunique()
        returns = pd.to_numeric(picks.get("Actual_Return_T30"), errors="coerce").dropna()
        rows.append({
            "Model": model,
            "Test_Year": test_year,
            "Candidate_Samples": len(candidates),
            "Candidate_Weeks": candidate_weeks,
            "Selected_Samples": len(picks),
            "Selected_Weeks": picks["Signal_Date"].nunique() if len(picks) else 0,
            "Blank_Candidate_Weeks": candidate_weeks - picks["Signal_Date"].nunique(),
            "Average_Return_T30_pct": float(returns.mean()) if len(returns) else np.nan,
            "Median_Return_T30_pct": float(returns.median()) if len(returns) else np.nan,
            "Win_Rate_pct": float((returns > 0).mean() * 100.0) if len(returns) else np.nan,
            "Target_First_pct": float((picks.get("Label_Path30") == "TARGET").mean() * 100.0)
            if len(picks) else np.nan,
            "Stop_First_pct": float((picks.get("Label_Path30") == "STOP").mean() * 100.0)
            if len(picks) else np.nan,
            "Survive_8W_pct": float(pd.to_numeric(
                picks.get("Label_Survive_8W"), errors="coerce"
            ).mean() * 100.0) if len(picks) else np.nan,
            "Average_Predicted_EV_pct": float(pd.to_numeric(
                picks.get("EV_T30"), errors="coerce"
            ).mean()) if len(picks) else np.nan,
        })
    return pd.DataFrame(rows)


def calibration_report(
    predictions: pd.DataFrame,
    probability_column: str = "Pred_path30_8w_TARGET",
    actual_column: str = "Label_Path30",
    bins: int = 5,
) -> pd.DataFrame:
    if predictions.empty or probability_column not in predictions.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (model, test_year), group in predictions.groupby(["Model", "Test_Year"]):
        work = group[[probability_column, actual_column]].dropna().copy()
        if len(work) < bins:
            continue
        rank = work[probability_column].rank(method="first")
        work["Probability_Bin"] = pd.qcut(rank, q=min(bins, len(work)), duplicates="drop")
        for bin_value, part in work.groupby("Probability_Bin", observed=True):
            rows.append({
                "Model": model,
                "Test_Year": test_year,
                "Probability_Bin": str(bin_value),
                "Samples": len(part),
                "Average_Prediction": float(part[probability_column].mean()),
                "Actual_Target_Rate": float((part[actual_column] == "TARGET").mean()),
                "Calibration_Gap": float(
                    part[probability_column].mean()
                    - (part[actual_column] == "TARGET").mean()
                ),
            })
    return pd.DataFrame(rows)


def coefficient_report(coefficients: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if coefficients.empty:
        return coefficients
    work = coefficients.copy()
    work["Abs_Coefficient"] = work["Coefficient"].abs()
    group_columns = ["Test_Year", "Model", "Endpoint", "Class"]
    return work.sort_values(
        group_columns + ["Abs_Coefficient"],
        ascending=[True, True, True, True, False],
    ).groupby(group_columns, dropna=False).head(top_n).reset_index(drop=True)


def run_research(
    paths: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    model_names: Sequence[str] = ("logit", "hgb"),
    stop_pct: int = DEFAULT_STOP_PCT,
    max_picks: int = DEFAULT_MAX_PICKS,
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    min_train_samples: int = DEFAULT_MIN_TRAIN_SAMPLES,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> dict[str, pd.DataFrame]:
    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "缺少scikit-learn。请把requirements.txt与app.py放在同一GitHub仓库根目录，"
            "然后在Streamlit Cloud重新部署。"
        ) from SKLEARN_IMPORT_ERROR
    raw = read_event_files(paths)
    entry = prepare_entry_dataset(raw, stop_pct=stop_pct)
    numeric, categorical = available_features(
        entry, NUMERIC_FEATURE_CANDIDATES, CATEGORICAL_FEATURE_CANDIDATES
    )
    predictions, metrics, coefficients, folds = walk_forward_predictions(
        entry,
        endpoints=ENTRY_ENDPOINTS,
        model_names=model_names,
        numeric=numeric,
        categorical=categorical,
        min_train_years=min_train_years,
        min_train_samples=min_train_samples,
        embargo_days=embargo_days,
    )
    selected = select_weekly_top3(predictions, max_picks=max_picks, require_positive_ev=True)
    report = top3_report(predictions, selected)
    calibration = calibration_report(predictions)
    top_coefficients = coefficient_report(coefficients)

    # 第4周为独立的动态复评实验；样本不足时保留空表而不是降低标准。
    week4 = prepare_week4_dataset(entry)
    w4_predictions = pd.DataFrame()
    w4_metrics = pd.DataFrame()
    if not week4.empty:
        w4_numeric, w4_categorical = available_features(
            week4, W4_NUMERIC_FEATURE_CANDIDATES, W4_CATEGORICAL_FEATURE_CANDIDATES
        )
        w4_predictions, w4_metrics, _, _ = walk_forward_predictions(
            week4,
            endpoints=W4_ENDPOINTS,
            model_names=model_names,
            numeric=w4_numeric,
            categorical=w4_categorical,
            min_train_years=min_train_years,
            min_train_samples=max(50, min_train_samples // 2),
            embargo_days=embargo_days,
        )
        if not w4_predictions.empty:
            # 正值越高越值得继续持有；只是研究分数，不是冻结交易阈值。
            w4_predictions["Continue_Score"] = (
                w4_predictions.get("Pred_w4_positive", 0.0)
                + w4_predictions.get("Pred_w4_gain10", 0.0)
                - w4_predictions.get("Pred_w4_drawdown10", 0.0)
            )

    results = {
        "dataset": entry,
        "folds": folds,
        "metrics": metrics,
        "predictions": predictions,
        "selected_top3": selected,
        "top3_report": report,
        "calibration": calibration,
        "coefficients": top_coefficients,
        "week4_dataset": week4,
        "week4_metrics": w4_metrics,
        "week4_predictions": w4_predictions,
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    run_id = stable_hash({
        "version": VERSION,
        "files": [Path(path).name for path in paths],
        "models": list(model_names),
        "stop_pct": stop_pct,
        "max_picks": max_picks,
        "min_train_years": min_train_years,
        "min_train_samples": min_train_samples,
        "embargo_days": embargo_days,
    })
    for name, frame in results.items():
        frame.to_csv(
            destination / f"weekly_probability_v6_{name}_{run_id}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return results


def percent_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    for column in output.columns:
        if any(token in column.lower() for token in ["rate", "brier", "auc", "prediction"]):
            if pd.api.types.is_numeric_dtype(output[column]):
                output[column] = output[column].round(4)
        elif pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].round(3)
    return output


def streamlit_main() -> None:
    assert st is not None
    st.set_page_config(page_title="4—8周路径概率研究器 V6.0", layout="wide")
    st.title("4—8周路径概率研究器 V6.0")
    if SKLEARN_IMPORT_ERROR is not None:
        st.error(
            "当前运行环境没有安装 scikit-learn。请将 requirements.txt 与 app.py "
            "放在同一GitHub仓库根目录，然后在Streamlit Cloud重新启动应用。"
        )
        st.code(
            "streamlit>=1.32,<2\n"
            "pandas>=2.0,<3\n"
            "numpy>=1.24,<3\n"
            "scikit-learn>=1.4,<2",
            language="text",
        )
        st.stop()
    st.caption(
        "MACD只作为特征；按年份滚动预测止损、止盈和八周存活概率。"
        "本版用于验证概率是否有区分力，不直接生成未来实盘信号。"
    )

    with st.sidebar:
        st.header("滚动验证参数")
        models = []
        if st.checkbox("逻辑回归（主模型）", value=True):
            models.append("logit")
        if st.checkbox("梯度提升树（挑战模型）", value=True):
            models.append("hgb")
        min_train_years = st.number_input(
            "最少训练年份", min_value=1, max_value=10,
            value=DEFAULT_MIN_TRAIN_YEARS, step=1,
        )
        min_train_samples = st.number_input(
            "最少训练事件", min_value=30, max_value=5000,
            value=DEFAULT_MIN_TRAIN_SAMPLES, step=10,
        )
        embargo_days = st.number_input(
            "训练测试隔离天数", min_value=40, max_value=120,
            value=DEFAULT_EMBARGO_DAYS, step=1,
        )
        max_picks = st.number_input(
            "每周最多选择", min_value=1, max_value=10,
            value=DEFAULT_MAX_PICKS, step=1,
        )

    uploads = st.file_uploader(
        "上传多个年度的 weekly_macd_events_*.csv",
        type=["csv"], accept_multiple_files=True,
        help="必须是完整事件明细；v5_selected只有入选样本，会产生严重选择偏差。",
    )
    if not uploads:
        st.info(
            "建议至少上传连续3个年度，最好6—8年。测试年份只能使用更早年份训练，"
            "因此只有3年数据时通常只能真正测试最后一年。"
        )
        with st.expander("V6为什么不直接读取行情并立刻选股？"):
            st.markdown(
                """
                1. 先验证概率是否校准、每周前三是否真正优于全部候选；
                2. 若概率没有样本外区分力，接入实时行情只会制造更复杂的过拟合；
                3. V5已经生成所需原始事件，本程序复用结果可避免重复数小时抓取行情；
                4. 研究通过后，再把冻结模型接入实时数据收集器。
                """
            )
        return
    if not models:
        st.error("至少选择一个模型")
        return

    if not st.button("开始V6逐年概率验证", type="primary"):
        return

    temp_dir = Path("weekly_probability_v6_uploads")
    temp_dir.mkdir(exist_ok=True)
    paths: list[str] = []
    for upload in uploads:
        safe_name = Path(upload.name).name
        target = temp_dir / safe_name
        target.write_bytes(upload.getbuffer())
        paths.append(str(target))

    try:
        with st.spinner("正在合并事件、构造路径标签并逐年滚动训练……"):
            results = run_research(
                paths=paths,
                output_dir="weekly_probability_v6_outputs",
                model_names=models,
                max_picks=int(max_picks),
                min_train_years=int(min_train_years),
                min_train_samples=int(min_train_samples),
                embargo_days=int(embargo_days),
            )
    except Exception as exc:
        st.exception(exc)
        return

    dataset = results["dataset"]
    years = sorted(dataset["Signal_Year"].unique())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("完整第一红柱事件", f"{len(dataset):,}")
    c2.metric("股票数", f"{dataset['ts_code'].nunique():,}")
    c3.metric("年度数", f"{len(years):,}")
    c4.metric("覆盖区间", f"{years[0]}—{years[-1]}" if years else "—")

    st.subheader("训练—测试年度边界")
    st.dataframe(percent_display(results["folds"]), use_container_width=True, hide_index=True)

    st.subheader("概率模型样本外指标")
    st.caption(
        "Brier和LogLoss越低越好，AUC越高越好；Actual_Rate与Predicted_Rate接近，"
        "说明整体概率校准较好。不能只看Accuracy。"
    )
    st.dataframe(percent_display(results["metrics"]), use_container_width=True, hide_index=True)

    st.subheader("预测期望收益为正时，每周最多3只")
    st.dataframe(percent_display(results["top3_report"]), use_container_width=True, hide_index=True)

    st.subheader("+30%先到概率分箱校准")
    st.dataframe(percent_display(results["calibration"]), use_container_width=True, hide_index=True)

    st.subheader("逻辑回归主要特征系数")
    st.caption(
        "系数只表示模型内部方向，不等于因果关系；跨测试年份方向反复变化的特征不可靠。"
    )
    st.dataframe(percent_display(results["coefficients"]), use_container_width=True, hide_index=True)

    st.subheader("第4周动态复评")
    if results["week4_metrics"].empty:
        st.warning("第4周可观察样本或训练年份不足，程序没有降低标准强行训练。")
    else:
        st.dataframe(
            percent_display(results["week4_metrics"]),
            use_container_width=True, hide_index=True,
        )

    st.subheader("下载核心结果（点击不会清空结果页）")
    download_names = [
        (1, "predictions", "逐事件概率预测"),
        (2, "selected_top3", "每周最多3只"),
        (3, "top3_report", "每周前三汇总"),
        (4, "metrics", "样本外模型指标"),
        (5, "calibration", "概率校准"),
        (6, "coefficients", "主要特征系数"),
        (7, "week4_predictions", "第4周预测"),
        (8, "week4_metrics", "第4周模型指标"),
    ]
    download_payloads = [
        (
            number,
            name,
            title,
            results[name].to_csv(index=False).encode("utf-8-sig"),
        )
        for number, name, title in download_names
    ]
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for number, name, _, payload in download_payloads:
            archive.writestr(f"{number:02d}_{name}.csv", payload)
    zip_buffer.seek(0)
    st.download_button(
        "0号｜一键下载全部V6结果ZIP",
        data=zip_buffer.getvalue(),
        file_name="weekly_probability_v6_all_results.zip",
        mime="application/zip",
        key="v6_download_all",
        on_click="ignore",
        type="primary",
    )
    columns = st.columns(4)
    for position, (number, name, title, payload) in enumerate(download_payloads):
        columns[position % 4].download_button(
            f"{number}号｜{title}",
            payload,
            file_name=f"weekly_probability_v6_{name}.csv",
            mime="text/csv",
            key=f"v6_download_{number}_{name}",
            on_click="ignore",
        )

    st.warning(
        "如果只有3个年度，模型结果仍然非常脆弱；若每周前三没有同时改善存活率、"
        "止盈先到率和去极值收益，就不能进入实时选股。概率不是承诺，"
        "只是在固定数据口径下的历史条件频率估计。"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4—8周路径概率研究器 V6.0")
    parser.add_argument("--input", nargs="+", help="一个或多个V5事件明细CSV")
    parser.add_argument("--output-dir", default="weekly_probability_v6_outputs")
    parser.add_argument("--models", nargs="+", choices=["logit", "hgb"], default=["logit", "hgb"])
    parser.add_argument("--max-picks", type=int, default=DEFAULT_MAX_PICKS)
    parser.add_argument("--min-train-years", type=int, default=DEFAULT_MIN_TRAIN_YEARS)
    parser.add_argument("--min-train-samples", type=int, default=DEFAULT_MIN_TRAIN_SAMPLES)
    parser.add_argument("--embargo-days", type=int, default=DEFAULT_EMBARGO_DAYS)
    return parser.parse_args(argv)


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if SKLEARN_IMPORT_ERROR is not None:
        print(
            "缺少scikit-learn，请执行：pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2
    if not args.input:
        if st is None:
            print("请通过 --input 指定V5事件CSV，或安装Streamlit后运行图形界面。", file=sys.stderr)
            return 2
        streamlit_main()
        return 0
    results = run_research(
        paths=args.input,
        output_dir=args.output_dir,
        model_names=args.models,
        max_picks=args.max_picks,
        min_train_years=args.min_train_years,
        min_train_samples=args.min_train_samples,
        embargo_days=args.embargo_days,
    )
    print("\n训练—测试年度边界")
    print(results["folds"].to_string(index=False))
    print("\n样本外模型指标")
    print(results["metrics"].to_string(index=False))
    print("\n每周最多3只")
    print(results["top3_report"].to_string(index=False))
    print(f"\n结果已保存至：{Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
