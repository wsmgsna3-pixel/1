from __future__ import annotations

"""
独立选股研究系统 v1.2
构建编号：SELECTOR-V1.2-20260812-FINAL

本文件不是“ｖ1.0日线版.py”。
v1.2保持A/B模型与v1.1完全一致，只新增入场位置诊断、
B模型第一名与前三名等权对照，以及ZIP打包下载。
所有新增指标只用于事后研究，不参与选股和排名。
"""

import io
import json
import math
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts


APP_NAME = "独立选股研究系统 v1.2"
APP_VERSION = "1.2.0"
BUILD_ID = "SELECTOR-V1.2-20260812-FINAL"
CACHE_DIR = Path(__file__).resolve().parent / ".selector_research_cache_v1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TECH_INDUSTRIES = {
    "电子": "801080.SI",
    "计算机": "801750.SI",
    "通信": "801770.SI",
    "传媒": "801760.SI",
    "电力设备": "801730.SI",
    "国防军工": "801740.SI",
}

TECH_SUBINDUSTRIES = {
    "自动化设备（机器人/工控/激光）": "801078.SI",
}


@dataclass(frozen=True)
class Config:
    start_date: str
    end_date: str
    l1_codes: tuple[str, ...]
    l2_codes: tuple[str, ...]
    min_price: float
    min_circ_mv_yi: float
    max_circ_mv_yi: float
    min_listing_days: int
    min_amount_yi: float
    use_historical_st: bool
    success_mfe: float
    severe_mae: float


def ymd(value: date | datetime | str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def safe_number(value: object) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def cache_path(group: str, key: str) -> Path:
    folder = CACHE_DIR / group
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{key}.csv.gz"


def read_cache(group: str, key: str) -> pd.DataFrame | None:
    path = cache_path(group, key)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, dtype=str, compression="gzip")
    except Exception:
        return None


def write_cache(group: str, key: str, frame: pd.DataFrame) -> None:
    path = cache_path(group, key)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, compression="gzip")
    os.replace(temp, path)


def api_call(func: Callable, *, retries: int = 4, pause: float = 0.12, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            result = func(**kwargs)
            time.sleep(pause)
            if result is None:
                return pd.DataFrame()
            return result
        except Exception as exc:  # TuShare将权限、频率和网络错误都作为异常返回
            last_error = exc
            time.sleep(min(8.0, 0.8 * (2**attempt)))
    raise RuntimeError(str(last_error) if last_error else "TuShare请求失败")


def load_or_fetch(
    group: str,
    key: str,
    fetcher: Callable[[], pd.DataFrame],
    gaps: list[dict],
    required: bool = True,
) -> pd.DataFrame:
    cached = read_cache(group, key)
    if cached is not None:
        return cached
    try:
        frame = fetcher()
        if frame is None or frame.empty:
            raise RuntimeError("接口返回空数据")
        write_cache(group, key, frame)
        return frame.astype(str)
    except Exception as exc:
        gaps.append({"数据类型": group, "日期或代码": key, "错误": str(exc), "是否关键": required})
        return pd.DataFrame()


def get_trade_dates(pro, start_date: str, end_date: str, gaps: list[dict]) -> list[str]:
    key = f"SSE_{start_date}_{end_date}"
    frame = load_or_fetch(
        "trade_cal",
        key,
        lambda: api_call(
            pro.trade_cal,
            exchange="SSE",
            start_date=start_date,
            end_date=end_date,
            is_open="1",
            fields="cal_date,is_open",
        ),
        gaps,
    )
    if frame.empty:
        return []
    return sorted(frame.loc[frame["is_open"].astype(str) == "1", "cal_date"].astype(str).unique().tolist())


def get_stock_basic(pro, gaps: list[dict]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    fields = "ts_code,symbol,name,market,exchange,list_status,list_date,delist_date"
    for status in ("L", "D", "P"):
        frame = load_or_fetch(
            "stock_basic",
            status,
            lambda status=status: api_call(
                pro.stock_basic, exchange="", list_status=status, fields=fields
            ),
            gaps,
            required=False,
        )
        if not frame.empty:
            pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    stocks = pd.concat(pieces, ignore_index=True).drop_duplicates("ts_code", keep="first")
    stocks["symbol"] = stocks["symbol"].astype(str).str.zfill(6)
    board_ok = stocks["symbol"].str.match(r"^(00|30|60|68)")
    exchange_ok = stocks["exchange"].isin(["SSE", "SZSE"])
    return stocks.loc[board_ok & exchange_ok].copy()


def get_industry_members(
    pro,
    l1_codes: Iterable[str],
    l2_codes: Iterable[str],
    gaps: list[dict],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    fields = "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,ts_code,name,in_date,out_date,is_new"
    for code in l1_codes:
        for is_new in ("Y", "N"):
            key = f"{code}_{is_new}"
            frame = load_or_fetch(
                "index_member_all",
                key,
                lambda code=code, is_new=is_new: api_call(
                    pro.index_member_all,
                    l1_code=code,
                    is_new=is_new,
                    fields=fields,
                ),
                gaps,
                required=is_new == "Y",
            )
            if not frame.empty:
                pieces.append(frame)
    for code in l2_codes:
        for is_new in ("Y", "N"):
            key = f"L2_{code}_{is_new}"
            frame = load_or_fetch(
                "index_member_all",
                key,
                lambda code=code, is_new=is_new: api_call(
                    pro.index_member_all,
                    l2_code=code,
                    is_new=is_new,
                    fields=fields,
                ),
                gaps,
                required=is_new == "Y",
            )
            if not frame.empty:
                pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    members = pd.concat(pieces, ignore_index=True)
    members = members.drop_duplicates(
        ["ts_code", "l1_code", "l2_code", "l3_code", "in_date", "out_date"], keep="first"
    )
    members["in_date"] = members["in_date"].fillna("19000101").replace("nan", "19000101")
    members["out_date"] = members["out_date"].fillna("").replace("nan", "")
    return members


def active_members_on(members: pd.DataFrame, signal_date: str) -> pd.DataFrame:
    if members.empty:
        return members
    active = members.loc[
        (members["in_date"] <= signal_date)
        & ((members["out_date"] == "") | (members["out_date"] >= signal_date))
    ].copy()
    active = active.sort_values(["ts_code", "in_date"]).drop_duplicates("ts_code", keep="last")
    return active


def fetch_market_day(pro, trade_date: str, universe: set[str], gaps: list[dict]) -> pd.DataFrame:
    merged_cached = read_cache("market_day", trade_date)
    if merged_cached is not None:
        return merged_cached.loc[merged_cached["ts_code"].isin(universe)].copy()

    daily = load_or_fetch(
        "daily",
        trade_date,
        lambda: api_call(
            pro.daily,
            trade_date=trade_date,
            fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg,amount",
        ),
        gaps,
    )
    factor = load_or_fetch(
        "adj_factor",
        trade_date,
        lambda: api_call(
            pro.adj_factor,
            trade_date=trade_date,
            fields="ts_code,trade_date,adj_factor",
        ),
        gaps,
    )
    if daily.empty or factor.empty:
        return pd.DataFrame()
    merged = daily.merge(factor, on=["ts_code", "trade_date"], how="inner")
    if merged.empty:
        gaps.append({"数据类型": "market_day", "日期或代码": trade_date, "错误": "行情与复权因子无法合并", "是否关键": False})
        return merged
    write_cache("market_day", trade_date, merged)
    return merged.loc[merged["ts_code"].isin(universe)].copy()


def fetch_daily_basic(pro, trade_date: str, gaps: list[dict]) -> pd.DataFrame:
    return load_or_fetch(
        "daily_basic",
        trade_date,
        lambda: api_call(
            pro.daily_basic,
            trade_date=trade_date,
            fields="ts_code,trade_date,close,circ_mv,turnover_rate,limit_status",
        ),
        gaps,
    )


def fetch_st_codes(pro, trade_date: str, gaps: list[dict], enabled: bool) -> set[str]:
    if not enabled:
        return set()
    frame = load_or_fetch(
        "stock_st",
        trade_date,
        lambda: api_call(
            pro.stock_st,
            trade_date=trade_date,
            fields="ts_code,name,trade_date,type,type_name",
        ),
        gaps,
        required=False,
    )
    if frame.empty:
        return set()
    return set(frame["ts_code"].astype(str))


def max_drawdown(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size < 2:
        return np.nan
    peak = np.maximum.accumulate(clean)
    return float(np.min(clean / peak - 1.0))


def prepare_price_panels(history: pd.DataFrame, trade_dates: list[str]) -> dict[str, pd.DataFrame]:
    numeric_columns = ["open", "high", "low", "close", "pre_close", "pct_chg", "amount", "adj_factor"]
    for col in numeric_columns:
        history[col] = pd.to_numeric(history[col], errors="coerce")
    history["trade_date"] = history["trade_date"].astype(str)
    full_index = pd.Index(trade_dates, name="trade_date")
    panels: dict[str, pd.DataFrame] = {}

    for code, group in history.groupby("ts_code", sort=False):
        group = group.sort_values("trade_date").drop_duplicates("trade_date", keep="last").set_index("trade_date")
        aligned = group.reindex(full_index)
        aligned["traded"] = aligned["close"].notna()
        aligned["amount"] = aligned["amount"].fillna(0.0)
        aligned["listed_obs"] = aligned["traded"].cumsum()

        for col in ("open", "high", "low", "close"):
            aligned[f"adj_{col}"] = aligned[col] * aligned["adj_factor"]

        adjusted_close = aligned["adj_close"].ffill()
        aligned["feature_close"] = adjusted_close
        aligned["ret1"] = adjusted_close.pct_change(fill_method=None).fillna(0.0)
        aligned["ret5"] = adjusted_close / adjusted_close.shift(5) - 1.0
        aligned["ret20"] = adjusted_close / adjusted_close.shift(20) - 1.0
        aligned["ret60"] = adjusted_close / adjusted_close.shift(60) - 1.0
        aligned["ret120_ex5"] = adjusted_close.shift(5) / adjusted_close.shift(125) - 1.0
        prior15_return = adjusted_close.shift(5) / adjusted_close.shift(20) - 1.0
        prior15_equivalent_5 = (1.0 + prior15_return).pow(1.0 / 3.0) - 1.0
        aligned["acceleration5"] = aligned["ret5"] - prior15_equivalent_5
        aligned["distance_high20"] = (
            adjusted_close / adjusted_close.rolling(20, min_periods=18).max() - 1.0
        )
        aligned["distance_high60"] = (
            adjusted_close / adjusted_close.rolling(60, min_periods=55).max() - 1.0
        )
        aligned["distance_ma20"] = (
            adjusted_close / adjusted_close.rolling(20, min_periods=18).mean() - 1.0
        )
        aligned["vol60"] = aligned["ret1"].rolling(60, min_periods=55).std(ddof=0) * math.sqrt(252)
        aligned["mdd60"] = adjusted_close.rolling(60, min_periods=55).apply(max_drawdown, raw=True)
        aligned["amount20"] = aligned["amount"].rolling(20, min_periods=18).mean()
        aligned["recent_trade_count20"] = aligned["traded"].rolling(20, min_periods=1).sum()

        # 停牌日只用于保持市场交易日时间轴；最高/最低价以最近收盘价填充，不能制造虚假波动。
        aligned["path_high"] = aligned["adj_high"].where(aligned["traded"], adjusted_close)
        aligned["path_low"] = aligned["adj_low"].where(aligned["traded"], adjusted_close)
        aligned["path_close"] = adjusted_close
        panels[str(code)] = aligned
    return panels


def percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    if series.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    if higher_is_better:
        return series.rank(pct=True, method="average", ascending=True)
    return series.rank(pct=True, method="average", ascending=False)


def weekly_signal_dates(trade_dates: list[str], start_date: str, end_date: str) -> list[str]:
    dates = pd.to_datetime(pd.Series(trade_dates), format="%Y%m%d")
    frame = pd.DataFrame({"date": dates})
    frame = frame.loc[(frame["date"] >= pd.Timestamp(start_date)) & (frame["date"] <= pd.Timestamp(end_date))]
    if frame.empty:
        return []
    frame["week"] = frame["date"].dt.to_period("W-FRI")
    return frame.groupby("week")["date"].max().dt.strftime("%Y%m%d").tolist()


def score_one_week(
    signal_date: str,
    config: Config,
    members: pd.DataFrame,
    stocks: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    daily_basic: pd.DataFrame,
    st_codes: set[str],
) -> tuple[pd.DataFrame, dict]:
    active = active_members_on(members, signal_date)
    meta = stocks[["ts_code", "symbol", "name", "market", "list_date", "delist_date"]].copy()
    active = active.merge(meta, on="ts_code", how="inner", suffixes=("_行业", ""))

    rows: list[dict] = []
    for row in active.itertuples(index=False):
        code = str(row.ts_code)
        panel = panels.get(code)
        if panel is None or signal_date not in panel.index:
            continue
        point = panel.loc[signal_date]
        rows.append(
            {
                "信号日": signal_date,
                "ts_code": code,
                "股票名称": row.name,
                "板块": row.market,
                "一级行业": row.l1_name,
                "二级行业": row.l2_name,
                "三级行业": row.l3_name,
                "上市日期": row.list_date,
                "退市日期": row.delist_date,
                "信号前5日涨幅": safe_number(point.get("ret5")),
                "信号前20日涨幅": safe_number(point.get("ret20")),
                "信号前60日涨幅": safe_number(point.get("ret60")),
                "近5日加速度": safe_number(point.get("acceleration5")),
                "距20日最高价": safe_number(point.get("distance_high20")),
                "距60日最高价": safe_number(point.get("distance_high60")),
                "距20日均线": safe_number(point.get("distance_ma20")),
                "ret120_ex5": safe_number(point.get("ret120_ex5")),
                "vol60": safe_number(point.get("vol60")),
                "mdd60": safe_number(point.get("mdd60")),
                "amount20": safe_number(point.get("amount20")),
                "listed_obs": safe_number(point.get("listed_obs")),
                "recent_trade_count20": safe_number(point.get("recent_trade_count20")),
                "信号日有交易": bool(point.get("traded", False)),
            }
        )
    features = pd.DataFrame(rows)
    funnel = {
        "信号日": signal_date,
        "历史行业成员": int(len(active)),
        "有足够行情": int(len(features)),
        "价格市值合格": 0,
        "上市与流动性合格": 0,
        "绝对趋势为正": 0,
        "排除ST后合格": 0,
    }
    if features.empty or daily_basic.empty:
        return pd.DataFrame(), funnel

    basic = daily_basic.copy()
    for col in ("close", "circ_mv", "turnover_rate", "limit_status"):
        if col in basic.columns:
            basic[col] = pd.to_numeric(basic[col], errors="coerce")
    basic = basic.rename(columns={"close": "未复权收盘", "circ_mv": "流通市值万元", "limit_status": "涨跌状态"})
    features = features.merge(
        basic[[c for c in ["ts_code", "未复权收盘", "流通市值万元", "turnover_rate", "涨跌状态"] if c in basic.columns]],
        on="ts_code",
        how="inner",
    )

    price_mv = (
        (features["未复权收盘"] >= config.min_price)
        & (features["流通市值万元"] >= config.min_circ_mv_yi * 10000.0)
        & (features["流通市值万元"] <= config.max_circ_mv_yi * 10000.0)
    )
    features = features.loc[price_mv].copy()
    funnel["价格市值合格"] = int(len(features))

    liquidity = (
        (features["listed_obs"] >= config.min_listing_days)
        & (features["amount20"] >= config.min_amount_yi * 100000.0)
        & (features["recent_trade_count20"] >= 18)
        & features["信号日有交易"]
    )
    features = features.loc[liquidity].copy()
    funnel["上市与流动性合格"] = int(len(features))

    features = features.loc[
        features[["ret120_ex5", "vol60", "mdd60"]].notna().all(axis=1)
        & (features["ret120_ex5"] > 0)
    ].copy()
    funnel["绝对趋势为正"] = int(len(features))

    features = features.loc[~features["ts_code"].isin(st_codes)].copy()
    funnel["排除ST后合格"] = int(len(features))
    if features.empty:
        return features, funnel

    industry_median = features.groupby("二级行业")["ret120_ex5"].transform("median")
    features["行业超额强度"] = features["ret120_ex5"] - industry_median
    features["相对强度分"] = percentile(features["行业超额强度"], True)
    features["低波动分"] = percentile(features["vol60"], False)
    features["低回撤分"] = percentile(features["mdd60"], True)
    features["价格模型分"] = (
        0.50 * features["相对强度分"]
        + 0.25 * features["低波动分"]
        + 0.25 * features["低回撤分"]
    )

    return features, funnel


def assign_model_ranks(frame: pd.DataFrame, model: str, score_col: str) -> pd.DataFrame:
    ranked = frame.copy().sort_values(score_col, ascending=False).reset_index(drop=True)
    ranked["模型"] = model
    ranked["综合分"] = ranked[score_col]
    ranked["周排名"] = np.arange(1, len(ranked) + 1)
    ranked["排名百分位"] = ranked["综合分"].rank(pct=True, ascending=True, method="average")
    if len(ranked) >= 5:
        ranked["分数组"] = pd.qcut(
            ranked["综合分"].rank(method="first"),
            q=5,
            labels=["Q1最低", "Q2", "Q3", "Q4", "Q5最高"],
        ).astype(str)
    else:
        ranked["分数组"] = "样本不足5只"
    return ranked


def first_hit_day(values: np.ndarray, threshold: float, direction: str) -> float:
    if direction == "up":
        matches = np.flatnonzero(values >= threshold)
    else:
        matches = np.flatnonzero(values <= threshold)
    return float(matches[0] + 1) if matches.size else np.nan


def path_metrics(
    panel: pd.DataFrame,
    entry_pos: int,
    entry_price: float,
    horizon: int,
    success_mfe: float,
    severe_mae: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        f"{horizon}日完整": False,
        f"{horizon}日末收益": np.nan,
        f"{horizon}日MFE": np.nan,
        f"{horizon}日MAE": np.nan,
        f"{horizon}日峰值天数": np.nan,
        f"{horizon}日峰值前MAE": np.nan,
        f"{horizon}日峰后至期末回撤": np.nan,
        f"{horizon}日先+20还是-10": "数据不足",
    }
    stop = entry_pos + horizon
    if entry_pos < 0 or stop > len(panel) or not np.isfinite(entry_price) or entry_price <= 0:
        return result
    window = panel.iloc[entry_pos:stop]
    highs = pd.to_numeric(window["path_high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(window["path_low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(window["path_close"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(highs).any() and np.isfinite(lows).any() and np.isfinite(closes[-1])):
        return result

    high_returns = highs / entry_price - 1.0
    low_returns = lows / entry_price - 1.0
    peak_index = int(np.nanargmax(high_returns))
    mfe = float(np.nanmax(high_returns))
    mae = float(np.nanmin(low_returns))
    end_return = float(closes[-1] / entry_price - 1.0)
    before_peak_mae = float(np.nanmin(low_returns[: peak_index + 1]))
    peak_price = highs[peak_index]
    giveback = float(closes[-1] / peak_price - 1.0) if peak_price > 0 else np.nan

    up10 = first_hit_day(high_returns, 0.10, "up")
    up20 = first_hit_day(high_returns, 0.20, "up")
    up30 = first_hit_day(high_returns, 0.30, "up")
    down5 = first_hit_day(low_returns, -0.05, "down")
    down8 = first_hit_day(low_returns, -0.08, "down")
    down10 = first_hit_day(low_returns, -0.10, "down")
    if np.isnan(up20) and np.isnan(down10):
        order = "均未触及"
    elif np.isnan(down10):
        order = "先+20%"
    elif np.isnan(up20):
        order = "先-10%"
    else:
        order = "先+20%" if up20 <= down10 else "先-10%"

    if mfe >= success_mfe and end_return < 0:
        label = "选股成功_利润回吐"
    elif mfe >= success_mfe and before_peak_mae > severe_mae:
        label = "强势成功"
    elif mfe >= success_mfe:
        label = "震荡后成功"
    elif mfe < 0.10:
        label = "真正失败"
    else:
        label = "中等机会"

    result.update(
        {
            f"{horizon}日完整": True,
            f"{horizon}日末收益": end_return,
            f"{horizon}日MFE": mfe,
            f"{horizon}日MAE": mae,
            f"{horizon}日峰值天数": peak_index + 1,
            f"{horizon}日峰值前MAE": before_peak_mae,
            f"{horizon}日峰后至期末回撤": giveback,
            f"{horizon}日到+10天数": up10,
            f"{horizon}日到+20天数": up20,
            f"{horizon}日到+30天数": up30,
            f"{horizon}日到-5天数": down5,
            f"{horizon}日到-8天数": down8,
            f"{horizon}日到-10天数": down10,
            f"{horizon}日先+20还是-10": order,
            f"{horizon}日路径分类": label,
        }
    )
    return result


def attach_forward_paths(
    candidates: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    trade_dates: list[str],
    success_mfe: float,
    severe_mae: float,
    progress,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    date_to_pos = {value: index for index, value in enumerate(trade_dates)}
    rows: list[dict] = []
    records = candidates.to_dict("records")
    path_cache: dict[tuple[str, str], dict[str, object]] = {}
    for number, row in enumerate(records):
        signal_date = str(row["信号日"])
        code = str(row["ts_code"])
        cache_key = (signal_date, code)
        if cache_key not in path_cache:
            signal_pos = date_to_pos.get(signal_date, -1)
            entry_pos = signal_pos + 1
            panel = panels.get(code)
            path_row: dict[str, object] = {
                "买入日": trade_dates[entry_pos] if 0 <= entry_pos < len(trade_dates) else "",
                "研究买入价": np.nan,
                "能否按次日开盘买入": False,
                "无法买入原因": "",
            }
            if panel is None or not (0 <= entry_pos < len(panel)):
                path_row["无法买入原因"] = "没有下一交易日数据"
            else:
                entry = panel.iloc[entry_pos]
                entry_price = safe_number(entry.get("adj_open"))
                raw_open = safe_number(entry.get("open"))
                raw_high = safe_number(entry.get("high"))
                raw_low = safe_number(entry.get("low"))
                pct_chg = safe_number(entry.get("pct_chg"))
                traded = bool(entry.get("traded", False))
                one_price_limit_up = (
                    traded
                    and np.isfinite(raw_open)
                    and np.isfinite(raw_high)
                    and np.isfinite(raw_low)
                    and abs(raw_high - raw_low) < 1e-8
                    and np.isfinite(pct_chg)
                    and pct_chg >= 9.5
                )
                if not traded or not np.isfinite(entry_price):
                    path_row["无法买入原因"] = "次日停牌或缺少开盘价"
                elif one_price_limit_up:
                    path_row["无法买入原因"] = "次日一字涨停"
                else:
                    path_row["能否按次日开盘买入"] = True
                    path_row["研究买入价"] = entry_price
                    for horizon in (20, 60, 120):
                        path_row.update(
                            path_metrics(
                                panel,
                                entry_pos,
                                entry_price,
                                horizon,
                                success_mfe,
                                severe_mae,
                            )
                        )
            path_cache[cache_key] = path_row
        row.update(path_cache[cache_key])
        rows.append(row)
        if number % 250 == 0 or number + 1 == len(records):
            progress.progress((number + 1) / len(records), text=f"完整价格路径 {number + 1}/{len(records)}")
    return pd.DataFrame(rows)


def build_bucket_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if candidates.empty:
        return pd.DataFrame()
    tradable = candidates.loc[candidates["能否按次日开盘买入"] == True].copy()  # noqa: E712
    for (model, bucket), group in tradable.groupby(["模型", "分数组"], dropna=False):
        for horizon in (20, 60, 120):
            complete_col = f"{horizon}日完整"
            sample = group.loc[group.get(complete_col, False) == True].copy()  # noqa: E712
            if sample.empty:
                continue
            mfe = sample[f"{horizon}日MFE"]
            mae = sample[f"{horizon}日MAE"]
            end_ret = sample[f"{horizon}日末收益"]
            labels = sample[f"{horizon}日路径分类"]
            rows.append(
                {
                    "模型": model,
                    "分数组": bucket,
                    "观察窗口": horizon,
                    "样本数": len(sample),
                    "MFE均值": mfe.mean(),
                    "MFE中位数": mfe.median(),
                    "MAE中位数": mae.median(),
                    "期末收益均值": end_ret.mean(),
                    "期末收益中位数": end_ret.median(),
                    "达到20%比例": (mfe >= 0.20).mean(),
                    "先20%后-10%比较_先涨比例": (sample[f"{horizon}日先+20还是-10"] == "先+20%").mean(),
                    "利润回吐比例": (labels == "选股成功_利润回吐").mean(),
                    "真正失败比例": (labels == "真正失败").mean(),
                }
            )
    return pd.DataFrame(rows)


def add_weekly_benchmark(candidates: pd.DataFrame) -> pd.DataFrame:
    """以同一信号日的全部合格股票为等权基准，不使用未来数据参与选股。"""
    output = candidates.copy()
    for horizon in (20, 60, 120):
        complete = output[f"{horizon}日完整"] == True  # noqa: E712
        tradable = output["能否按次日开盘买入"] == True  # noqa: E712
        sample = output.loc[complete & tradable]
        benchmark = sample.groupby(["信号日", "模型"])[f"{horizon}日末收益"].mean()
        key = pd.MultiIndex.from_frame(output[["信号日", "模型"]])
        output[f"{horizon}日合格池基准收益"] = benchmark.reindex(key).to_numpy()
        output[f"{horizon}日超额收益"] = (
            output[f"{horizon}日末收益"] - output[f"{horizon}日合格池基准收益"]
        )
    return output


def build_model_summary(signals: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    rows: list[dict] = []
    for model, model_frame in signals.groupby("模型"):
        for horizon in (20, 60, 120):
            sample = model_frame.loc[
                (model_frame["能否按次日开盘买入"] == True)  # noqa: E712
                & (model_frame[f"{horizon}日完整"] == True)  # noqa: E712
            ].copy()
            if sample.empty:
                continue
            rows.append(
                {
                    "样本口径": sample_type,
                    "模型": model,
                    "观察窗口": horizon,
                    "样本数": len(sample),
                    "不同股票数": sample["ts_code"].nunique(),
                    "MFE均值": sample[f"{horizon}日MFE"].mean(),
                    "MFE中位数": sample[f"{horizon}日MFE"].median(),
                    "MAE中位数": sample[f"{horizon}日MAE"].median(),
                    "期末收益均值": sample[f"{horizon}日末收益"].mean(),
                    "期末收益中位数": sample[f"{horizon}日末收益"].median(),
                    "超额收益均值": sample[f"{horizon}日超额收益"].mean(),
                    "超额收益中位数": sample[f"{horizon}日超额收益"].median(),
                    "达到20%比例": (sample[f"{horizon}日MFE"] >= 0.20).mean(),
                    "先+20%比例": (sample[f"{horizon}日先+20还是-10"] == "先+20%").mean(),
                    "先-10%比例": (sample[f"{horizon}日先+20还是-10"] == "先-10%").mean(),
                    "真正失败比例": (sample[f"{horizon}日路径分类"] == "真正失败").mean(),
                    "利润回吐比例": (sample[f"{horizon}日路径分类"] == "选股成功_利润回吐").mean(),
                }
            )
    return pd.DataFrame(rows)


def mark_independent_events(signals: pd.DataFrame, max_gap_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    """同一模型、同一股票、间隔不超过14天的第一名信号合并为一次趋势事件。"""
    if signals.empty:
        return signals.copy(), signals.copy()
    marked_parts: list[pd.DataFrame] = []
    event_number = 0
    for model, model_frame in signals.groupby("模型", sort=False):
        frame = model_frame.copy()
        frame["_date"] = pd.to_datetime(frame["信号日"].astype(str), format="%Y%m%d")
        frame = frame.sort_values("_date")
        last_seen: dict[str, pd.Timestamp] = {}
        active_event: dict[str, str] = {}
        event_ids: list[str] = []
        first_flags: list[bool] = []
        for _, row in frame.iterrows():
            code = str(row["ts_code"])
            current = row["_date"]
            is_new = code not in last_seen or (current - last_seen[code]).days > max_gap_days
            if is_new:
                event_number += 1
                active_event[code] = f"E{event_number:04d}"
            event_ids.append(active_event[code])
            first_flags.append(is_new)
            last_seen[code] = current
        frame["事件编号"] = event_ids
        frame["是否事件首信号"] = first_flags
        frame["事件连续信号数"] = frame.groupby("事件编号")["事件编号"].transform("size")
        frame["事件最后信号日"] = frame.groupby("事件编号")["信号日"].transform("max")
        marked_parts.append(frame.drop(columns="_date"))
    marked = pd.concat(marked_parts, ignore_index=True).sort_values(["信号日", "模型"])
    events = marked.loc[marked["是否事件首信号"] == True].copy()  # noqa: E712
    return marked.reset_index(drop=True), events.reset_index(drop=True)


def build_rank_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    bands = pd.cut(
        candidates["周排名"],
        bins=[0, 1, 3, 10, 50, np.inf],
        labels=["第1名", "第2-3名", "第4-10名", "第11-50名", "第51名以后"],
    )
    work = candidates.copy()
    work["排名区间"] = bands.astype(str)
    for (model, band), group in work.groupby(["模型", "排名区间"], dropna=False):
        for horizon in (20, 60, 120):
            sample = group.loc[
                (group["能否按次日开盘买入"] == True)  # noqa: E712
                & (group[f"{horizon}日完整"] == True)  # noqa: E712
            ]
            if sample.empty:
                continue
            rows.append(
                {
                    "模型": model,
                    "排名区间": band,
                    "观察窗口": horizon,
                    "样本数": len(sample),
                    "不同股票数": sample["ts_code"].nunique(),
                    "MFE中位数": sample[f"{horizon}日MFE"].median(),
                    "MAE中位数": sample[f"{horizon}日MAE"].median(),
                    "期末收益中位数": sample[f"{horizon}日末收益"].median(),
                    "超额收益中位数": sample[f"{horizon}日超额收益"].median(),
                    "达到20%比例": (sample[f"{horizon}日MFE"] >= 0.20).mean(),
                    "先+20%比例": (sample[f"{horizon}日先+20还是-10"] == "先+20%").mean(),
                    "先-10%比例": (sample[f"{horizon}日先+20还是-10"] == "先-10%").mean(),
                    "真正失败比例": (sample[f"{horizon}日路径分类"] == "真正失败").mean(),
                }
            )
    return pd.DataFrame(rows)


def rank_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[0, 1, 3, 10, 50, np.inf],
        labels=["第1名", "第2-3名", "第4-10名", "第11-50名", "第51名以后"],
    ).astype(str)


def build_entry_diagnostics(candidates: pd.DataFrame) -> pd.DataFrame:
    """比较各排名区间的买入前位置；这些字段不参与评分。"""
    if candidates.empty:
        return pd.DataFrame()
    work = candidates.copy()
    work["排名区间"] = rank_band(work["周排名"])
    diagnostic_columns = [
        "信号前5日涨幅",
        "信号前20日涨幅",
        "信号前60日涨幅",
        "近5日加速度",
        "距20日最高价",
        "距60日最高价",
        "距20日均线",
    ]
    rows: list[dict] = []
    for (model, band), group in work.groupby(["模型", "排名区间"], dropna=False):
        for horizon in (20, 60, 120):
            sample = group.loc[
                (group["能否按次日开盘买入"] == True)  # noqa: E712
                & (group[f"{horizon}日完整"] == True)  # noqa: E712
            ].copy()
            if sample.empty:
                continue
            row: dict[str, object] = {
                "模型": model,
                "排名区间": band,
                "观察窗口": horizon,
                "样本数": len(sample),
                "不同股票数": sample["ts_code"].nunique(),
            }
            for column in diagnostic_columns:
                row[f"{column}中位数"] = sample[column].median()
            row.update(
                {
                    "未来MFE中位数": sample[f"{horizon}日MFE"].median(),
                    "未来MAE中位数": sample[f"{horizon}日MAE"].median(),
                    "期末收益中位数": sample[f"{horizon}日末收益"].median(),
                    "超额收益中位数": sample[f"{horizon}日超额收益"].median(),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_b_top3_comparison(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在完全相同的成熟周中，比较B模型第一名与前三名等权期末收益。"""
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    model_b = candidates.loc[candidates["模型"] == "B_原综合模型"].copy()
    rows: list[dict] = []
    for horizon in (20, 60, 120):
        for signal_date, week in model_b.groupby("信号日", sort=True):
            top3 = week.loc[week["周排名"] <= 3].sort_values("周排名").copy()
            paired = top3.loc[
                (top3["能否按次日开盘买入"] == True)  # noqa: E712
                & (top3[f"{horizon}日完整"] == True)  # noqa: E712
            ]
            if len(top3) != 3 or len(paired) != 3 or not (paired["周排名"] == 1).any():
                continue
            top1 = paired.loc[paired["周排名"] == 1].iloc[0]
            benchmark = safe_number(top1.get(f"{horizon}日合格池基准收益"))
            top1_return = safe_number(top1.get(f"{horizon}日末收益"))
            top3_return = safe_number(paired[f"{horizon}日末收益"].mean())
            difference = top3_return - top1_return
            definitions = [
                ("B_第一名", paired.loc[paired["周排名"] == 1], top1_return),
                ("B_前三名等权", paired, top3_return),
            ]
            for method, constituents, end_return in definitions:
                rows.append(
                    {
                        "信号日": signal_date,
                        "观察窗口": horizon,
                        "组合口径": method,
                        "成分数量": len(constituents),
                        "成分股票": "、".join(constituents["股票名称"].astype(str).tolist()),
                        "期末收益": end_return,
                        "合格池基准收益": benchmark,
                        "超额收益": end_return - benchmark,
                        "是否盈利": end_return > 0,
                        "是否跑赢合格池": end_return > benchmark,
                        "前三名减第一名": difference,
                    }
                )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary_rows: list[dict] = []
    for (horizon, method), group in detail.groupby(["观察窗口", "组合口径"], sort=True):
        summary_rows.append(
            {
                "观察窗口": horizon,
                "组合口径": method,
                "配对成熟周数": len(group),
                "期末收益均值": group["期末收益"].mean(),
                "期末收益中位数": group["期末收益"].median(),
                "超额收益均值": group["超额收益"].mean(),
                "超额收益中位数": group["超额收益"].median(),
                "盈利周比例": group["是否盈利"].mean(),
                "跑赢合格池比例": group["是否跑赢合格池"].mean(),
                "前三名减第一名中位数": group["前三名减第一名"].median(),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def build_factor_ic(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    factor_columns = ["相对强度分", "低波动分", "低回撤分", "综合分"]
    for model, model_frame in candidates.groupby("模型"):
        for horizon in (20, 60, 120):
            sample = model_frame.loc[
                (model_frame["能否按次日开盘买入"] == True)  # noqa: E712
                & (model_frame[f"{horizon}日完整"] == True)  # noqa: E712
            ]
            for factor in factor_columns:
                for target in (f"{horizon}日MFE", f"{horizon}日末收益", f"{horizon}日MAE"):
                    weekly_values: list[float] = []
                    for _, week in sample.groupby("信号日"):
                        if len(week) >= 20 and week[factor].nunique() > 1 and week[target].nunique() > 1:
                            value = week[factor].corr(week[target], method="spearman")
                            if pd.notna(value):
                                weekly_values.append(float(value))
                    if weekly_values:
                        values = pd.Series(weekly_values)
                        rows.append(
                            {
                                "模型": model,
                                "观察窗口": horizon,
                                "因子": factor,
                                "目标": target,
                                "成熟周数": len(values),
                                "逐周IC均值": values.mean(),
                                "逐周IC中位数": values.median(),
                                "IC为正周比例": (values > 0).mean(),
                            }
                        )
    return pd.DataFrame(rows)


def build_robustness(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model, model_frame in events.groupby("模型"):
        for horizon in (20, 60, 120):
            sample = model_frame.loc[
                (model_frame["能否按次日开盘买入"] == True)  # noqa: E712
                & (model_frame[f"{horizon}日完整"] == True)  # noqa: E712
            ].copy()
            if sample.empty:
                continue
            stock_performance = sample.groupby("ts_code")[f"{horizon}日末收益"].mean().sort_values(ascending=False)
            for remove_count in (0, 1, 3, 5):
                removed = set(stock_performance.head(remove_count).index) if remove_count else set()
                remaining = sample.loc[~sample["ts_code"].isin(removed)]
                if remaining.empty:
                    continue
                rows.append(
                    {
                        "模型": model,
                        "观察窗口": horizon,
                        "剔除最好股票数": remove_count,
                        "被剔除股票": "、".join(
                            sample.loc[sample["ts_code"].isin(removed), "股票名称"].drop_duplicates().tolist()
                        ),
                        "剩余事件数": len(remaining),
                        "剩余股票数": remaining["ts_code"].nunique(),
                        "MFE均值": remaining[f"{horizon}日MFE"].mean(),
                        "MFE中位数": remaining[f"{horizon}日MFE"].median(),
                        "期末收益均值": remaining[f"{horizon}日末收益"].mean(),
                        "期末收益中位数": remaining[f"{horizon}日末收益"].median(),
                        "达到20%比例": (remaining[f"{horizon}日MFE"] >= 0.20).mean(),
                        "先+20%比例": (remaining[f"{horizon}日先+20还是-10"] == "先+20%").mean(),
                    }
                )
    return pd.DataFrame(rows)


def format_percent_columns(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    keywords = ("收益", "MFE", "MAE", "回撤", "比例", "ret120", "涨幅", "加速度", "距")
    score_columns = {"综合分", "相对强度分", "低波动分", "低回撤分", "价格模型分", "排名百分位"}
    for col in display.columns:
        is_percent = any(key in str(col) for key in keywords) or str(col) in score_columns
        if is_percent and pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    return display


def result_files(result: dict) -> list[tuple[str, pd.DataFrame | bytes]]:
    run_id = str(result["run_id"])
    config_bytes = json.dumps(result["config"], ensure_ascii=False, indent=2).encode("utf-8")
    return [
        (f"weekly_signals_selector_v1_2_{run_id}.csv", result["signals"]),
        (f"independent_events_selector_v1_2_{run_id}.csv", result["events"]),
        (f"weekly_top10_selector_v1_2_{run_id}.csv", result["top10"]),
        (f"all_candidates_paths_selector_v1_2_{run_id}.csv", result["candidates"]),
        (f"model_comparison_selector_v1_2_{run_id}.csv", result["model_summary"]),
        (f"rank_bands_selector_v1_2_{run_id}.csv", result["rank_summary"]),
        (f"entry_diagnostics_selector_v1_2_{run_id}.csv", result["entry_diagnostics"]),
        (f"b_top3_comparison_selector_v1_2_{run_id}.csv", result["b_top3_summary"]),
        (f"b_top3_weekly_selector_v1_2_{run_id}.csv", result["b_top3_detail"]),
        (f"factor_ic_selector_v1_2_{run_id}.csv", result["factor_ic"]),
        (f"robustness_selector_v1_2_{run_id}.csv", result["robustness"]),
        (f"bucket_summary_selector_v1_2_{run_id}.csv", result["bucket"]),
        (f"funnel_selector_v1_2_{run_id}.csv", result["funnel"]),
        (f"data_gaps_selector_v1_2_{run_id}.csv", result["gaps"]),
        (f"research_config_selector_v1_2_{run_id}.json", config_bytes),
    ]


def build_zip(result: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in result_files(result):
            data = to_csv_bytes(content) if isinstance(content, pd.DataFrame) else content
            archive.writestr(filename, data)
    return buffer.getvalue()


def run_research(pro, config: Config) -> dict[str, pd.DataFrame | dict]:
    gaps: list[dict] = []
    status = st.status("准备数据", expanded=True)

    warmup_start = ymd(pd.Timestamp(config.start_date) - pd.Timedelta(days=500))
    forward_end = ymd(min(pd.Timestamp.today().normalize(), pd.Timestamp(config.end_date) + pd.Timedelta(days=190)))
    status.write("读取交易日历与历史申万行业成员")
    trade_dates = get_trade_dates(pro, warmup_start, forward_end, gaps)
    if not trade_dates:
        raise RuntimeError("无法取得交易日历，不能继续。")

    stocks = get_stock_basic(pro, gaps)
    members = get_industry_members(pro, config.l1_codes, config.l2_codes, gaps)
    if stocks.empty or members.empty:
        raise RuntimeError("无法建立股票池，请检查TuShare积分和接口权限。")
    members = members.loc[members["ts_code"].isin(set(stocks["ts_code"]))].copy()
    universe = set(members["ts_code"].astype(str))
    status.write(f"历史科技股票池并集：{len(universe)}只")

    price_progress = st.progress(0.0, text="下载复权行情")
    day_frames: list[pd.DataFrame] = []
    for index, trade_date in enumerate(trade_dates):
        frame = fetch_market_day(pro, trade_date, universe, gaps)
        if not frame.empty:
            day_frames.append(frame)
        if index % 5 == 0 or index + 1 == len(trade_dates):
            price_progress.progress((index + 1) / len(trade_dates), text=f"复权行情 {index + 1}/{len(trade_dates)}")
    if not day_frames:
        raise RuntimeError("没有取得任何行情数据。")
    history = pd.concat(day_frames, ignore_index=True)
    available_dates = sorted(history["trade_date"].astype(str).unique().tolist())
    trade_dates = [d for d in trade_dates if d in set(available_dates)]
    panels = prepare_price_panels(history, trade_dates)

    signals = weekly_signal_dates(trade_dates, config.start_date, config.end_date)
    if not signals:
        raise RuntimeError("所选期间内没有可用周末信号日。")

    scoring_progress = st.progress(0.0, text="逐周排名")
    ranked_weeks: list[pd.DataFrame] = []
    funnel_rows: list[dict] = []
    st_endpoint_available = config.use_historical_st
    for index, signal_date in enumerate(signals):
        basic = fetch_daily_basic(pro, signal_date, gaps)
        if index == 0 and basic.empty:
            raise RuntimeError("daily_basic不可用；股价和流通市值过滤无法执行，请检查2000积分权限。")
        gap_count_before_st = len(gaps)
        st_codes = fetch_st_codes(pro, signal_date, gaps, st_endpoint_available)
        if len(gaps) > gap_count_before_st and any(
            item.get("数据类型") == "stock_st" for item in gaps[gap_count_before_st:]
        ):
            st_endpoint_available = False
            status.write("历史ST接口不可用：已停止继续请求，研究仍运行并在数据缺口中标记。")
        base, funnel = score_one_week(
            signal_date,
            config,
            members,
            stocks,
            panels,
            basic,
            st_codes,
        )
        funnel_rows.append(funnel)
        if not base.empty:
            ranked_weeks.append(assign_model_ranks(base, "A_相对强度单因子", "相对强度分"))
            ranked_weeks.append(assign_model_ranks(base, "B_原综合模型", "价格模型分"))
        scoring_progress.progress((index + 1) / len(signals), text=f"逐周排名 {index + 1}/{len(signals)}")

    if not ranked_weeks:
        raise RuntimeError("所有周均无合格股票。请先查看数据缺口和筛选漏斗，不要直接放宽评分。")
    candidates = pd.concat(ranked_weeks, ignore_index=True)

    path_progress = st.progress(0.0, text="计算完整价格路径")
    candidates = attach_forward_paths(
        candidates,
        panels,
        trade_dates,
        config.success_mfe,
        config.severe_mae,
        path_progress,
    )
    candidates = add_weekly_benchmark(candidates)
    candidates = candidates.sort_values(["信号日", "模型", "周排名"]).reset_index(drop=True)
    weekly_top10 = candidates.loc[candidates["周排名"] <= 10].copy()
    weekly_signals = candidates.loc[candidates["周排名"] == 1].copy()
    weekly_signals, independent_events = mark_independent_events(weekly_signals)
    bucket_summary = build_bucket_summary(candidates)
    model_summary = pd.concat(
        [
            build_model_summary(weekly_signals, "全部周第一名"),
            build_model_summary(independent_events, "独立趋势事件"),
        ],
        ignore_index=True,
    )
    rank_summary = build_rank_summary(candidates)
    entry_diagnostics = build_entry_diagnostics(candidates)
    b_top3_detail, b_top3_summary = build_b_top3_comparison(candidates)
    factor_ic = build_factor_ic(candidates)
    robustness = build_robustness(independent_events)
    funnel = pd.DataFrame(funnel_rows)
    gap_frame = pd.DataFrame(gaps)

    status.update(label="研究完成", state="complete", expanded=False)
    return {
        "config": {
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "build_id": BUILD_ID,
            **asdict(config),
        },
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "candidates": candidates,
        "top10": weekly_top10,
        "signals": weekly_signals,
        "events": independent_events,
        "model_summary": model_summary,
        "rank_summary": rank_summary,
        "entry_diagnostics": entry_diagnostics,
        "b_top3_detail": b_top3_detail,
        "b_top3_summary": b_top3_summary,
        "factor_ic": factor_ic,
        "robustness": robustness,
        "bucket": bucket_summary,
        "funnel": funnel,
        "gaps": gap_frame,
    }


def render_results(result: dict) -> None:
    candidates: pd.DataFrame = result["candidates"]
    signals: pd.DataFrame = result["signals"]
    top10: pd.DataFrame = result["top10"]
    events: pd.DataFrame = result["events"]
    model_summary: pd.DataFrame = result["model_summary"]
    rank_summary: pd.DataFrame = result["rank_summary"]
    entry_diagnostics: pd.DataFrame = result["entry_diagnostics"]
    b_top3_detail: pd.DataFrame = result["b_top3_detail"]
    b_top3_summary: pd.DataFrame = result["b_top3_summary"]
    factor_ic: pd.DataFrame = result["factor_ic"]
    robustness: pd.DataFrame = result["robustness"]
    bucket: pd.DataFrame = result["bucket"]
    funnel: pd.DataFrame = result["funnel"]
    gaps: pd.DataFrame = result["gaps"]

    st.subheader("研究结论面板")
    tradable_signals = signals.loc[signals["能否按次日开盘买入"] == True].copy()  # noqa: E712
    if "120日完整" in tradable_signals.columns:
        complete120 = tradable_signals.loc[tradable_signals["120日完整"] == True].copy()  # noqa: E712
    else:
        complete120 = tradable_signals.iloc[0:0].copy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("研究周数", signals["信号日"].nunique())
    c2.metric("双模型周信号", len(signals))
    c3.metric("独立趋势事件", len(events))
    if not complete120.empty:
        c4.metric("120日MFE中位数", f"{complete120['120日MFE'].median():.2%}")
    else:
        c4.metric("120日MFE中位数", "样本不足")

    st.caption("A为行业相对强度单因子；B为原50%相对强度+25%低波动+25%低回撤。两者固定同时运行，不搜索权重。")
    st.dataframe(format_percent_columns(model_summary), use_container_width=True, hide_index=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        ["每周第一名", "独立趋势事件", "排名区间", "入场位置", "因子诊断", "牛股依赖", "每周Top10", "分组单调性", "数据审计"]
    )
    with tab1:
        preferred = [
            "信号日", "买入日", "模型", "周排名", "ts_code", "股票名称", "二级行业",
            "综合分", "未复权收盘", "流通市值万元", "能否按次日开盘买入", "无法买入原因",
            "信号前5日涨幅", "信号前20日涨幅", "信号前60日涨幅", "近5日加速度",
            "距20日最高价", "距60日最高价", "距20日均线",
            "20日MFE", "20日MAE", "20日路径分类", "60日MFE", "60日MAE", "60日路径分类",
            "120日MFE", "120日MAE", "120日末收益", "120日峰值天数", "120日峰值前MAE",
            "120日峰后至期末回撤", "120日超额收益", "120日路径分类",
            "事件编号", "是否事件首信号", "事件连续信号数",
        ]
        st.dataframe(format_percent_columns(signals[[c for c in preferred if c in signals.columns]]), use_container_width=True, hide_index=True)
    with tab2:
        st.caption("同一模型、同一股票、相邻信号间隔不超过14天，只保留第一次作为独立事件。")
        st.dataframe(format_percent_columns(events), use_container_width=True, hide_index=True)
    with tab3:
        st.caption("直接比较第1名、第2-3名、第4-10名；第一名应稳定优于后续排名才有实盘意义。")
        st.dataframe(format_percent_columns(rank_summary), use_container_width=True, hide_index=True)
    with tab4:
        st.caption("只诊断追高与入场位置，不改变A/B排名。近5日加速度=最近5日涨幅减去此前15日折算的5日涨幅。")
        st.dataframe(format_percent_columns(entry_diagnostics), use_container_width=True, hide_index=True)
        st.markdown("**B模型：第一名与前三名等权（相同成熟周配对比较）**")
        st.dataframe(format_percent_columns(b_top3_summary), use_container_width=True, hide_index=True)
        with st.expander("查看逐周配对明细"):
            st.dataframe(format_percent_columns(b_top3_detail), use_container_width=True, hide_index=True)
    with tab5:
        st.caption("逐周Spearman相关。MFE、期末收益希望为正；MAE希望为正，代表评分越高、亏损越小。")
        st.dataframe(format_percent_columns(factor_ic), use_container_width=True, hide_index=True)
    with tab6:
        st.caption("基于独立趋势事件，依次删除期末表现最好的1、3、5只股票，检查结果是否坍塌。")
        st.dataframe(format_percent_columns(robustness), use_container_width=True, hide_index=True)
    with tab7:
        st.dataframe(format_percent_columns(top10), use_container_width=True, hide_index=True)
    with tab8:
        st.caption("只有从Q1到Q5总体改善，才说明排名具有可信的横截面区分能力。")
        st.dataframe(format_percent_columns(bucket), use_container_width=True, hide_index=True)
    with tab9:
        st.markdown("**筛选漏斗**")
        st.dataframe(funnel, use_container_width=True, hide_index=True)
        st.markdown("**数据缺口**")
        if gaps.empty:
            st.success("没有记录到数据缺口。")
        else:
            st.warning("非关键日期缺失会被跳过并记录，不会强制整次研究失败。")
            st.dataframe(gaps, use_container_width=True, hide_index=True)

    st.subheader("下载结果")
    st.download_button(
        "一键下载全部研究结果（ZIP）",
        data=build_zip(result),
        file_name=f"selector_research_v1_2_{result['run_id']}.zip",
        mime="application/zip",
        type="primary",
        key=f"download_all_{result['run_id']}",
    )
    with st.expander("单独下载文件（备用）"):
        columns = st.columns(3)
        for index, (filename, content) in enumerate(result_files(result)):
            data = to_csv_bytes(content) if isinstance(content, pd.DataFrame) else content
            columns[index % 3].download_button(
                label=filename.rsplit("_selector", 1)[0],
                data=data,
                file_name=filename,
                mime="application/json" if filename.endswith(".json") else "text/csv",
                key=f"download_single_{index}_{result['run_id']}",
            )


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    st.title(APP_NAME)
    st.code(f"版本：{APP_VERSION}｜构建编号：{BUILD_ID}", language=None)
    st.caption("从零设计；不使用MACD、红绿柱、旧评分阈值或旧版本交易规则。当前版本只验证选股能力，不模拟组合卖出。")

    with st.sidebar:
        st.header("数据与股票池")
        token_default = os.environ.get("TUSHARE_TOKEN", "")
        token = st.text_input("TuShare Token", value=token_default, type="password")
        default_end = date.today()
        # 开发阶段默认只研究最近一年；需要最终验证时再由用户手动扩大区间。
        default_start = default_end - timedelta(days=365)
        start_value = st.date_input("信号开始日期", value=default_start)
        end_value = st.date_input("信号结束日期", value=default_end)
        industry_names = st.multiselect(
            "申万一级行业",
            options=list(TECH_INDUSTRIES.keys()),
            default=["电子", "计算机", "通信", "电力设备", "国防军工"],
        )
        subindustry_names = st.multiselect(
            "补充申万二级行业",
            options=list(TECH_SUBINDUSTRIES.keys()),
            default=list(TECH_SUBINDUSTRIES.keys()),
            help="只补入机械设备中的自动化设备，不把整个机械设备行业纳入。",
        )
        min_price = st.number_input("最低股价（元）", min_value=1.0, value=20.0, step=1.0)
        c1, c2 = st.columns(2)
        min_mv = c1.number_input("最低流通市值（亿元）", min_value=1.0, value=100.0, step=10.0)
        max_mv = c2.number_input("最高流通市值（亿元）", min_value=10.0, value=1000.0, step=50.0)
        min_listing = st.number_input("最少上市交易日", min_value=120, value=250, step=10)
        min_amount = st.number_input("20日平均成交额下限（亿元）", min_value=0.0, value=1.0, step=0.1)

        st.header("研究模型")
        st.caption("固定同时运行A：相对强度单因子；B：原综合模型。v1.2不改模型，只增加入场诊断。")
        use_st = st.checkbox(
            "尝试调用历史ST列表",
            value=True,
            help="stock_st通常需要3000积分；权限不足时记录缺口，研究继续运行。",
        )

        st.header("路径诊断口径")
        success_mfe_pct = st.number_input("显著上涨空间（%）", min_value=5.0, value=20.0, step=5.0)
        severe_mae_pct = st.number_input("严重前置回撤（%）", min_value=3.0, value=10.0, step=1.0)
        st.caption("这两个数只用于事后分类，不参与选股评分，也不是止盈止损。")

        run_button = st.button("开始研究", type="primary", use_container_width=True)

    st.info(
        "A模型只按行业相对强度排名；B模型固定为相对强度50% + 低波动25% + 低回撤25%。"
        "重点比较第一名是否稳定优于第2-10名，并诊断短期涨幅、加速度和距离近期高点。"
    )

    if run_button:
        if not token.strip():
            st.error("请填写TuShare Token。")
            return
        if start_value >= end_value:
            st.error("结束日期必须晚于开始日期。")
            return
        if not industry_names and not subindustry_names:
            st.error("至少选择一个申万行业。")
            return
        if max_mv <= min_mv:
            st.error("最高流通市值必须大于最低流通市值。")
            return

        ts.set_token(token.strip())
        pro = ts.pro_api(token.strip())
        config = Config(
            start_date=ymd(start_value),
            end_date=ymd(end_value),
            l1_codes=tuple(TECH_INDUSTRIES[name] for name in industry_names),
            l2_codes=tuple(TECH_SUBINDUSTRIES[name] for name in subindustry_names),
            min_price=float(min_price),
            min_circ_mv_yi=float(min_mv),
            max_circ_mv_yi=float(max_mv),
            min_listing_days=int(min_listing),
            min_amount_yi=float(min_amount),
            use_historical_st=bool(use_st),
            success_mfe=float(success_mfe_pct) / 100.0,
            severe_mae=-float(severe_mae_pct) / 100.0,
        )
        try:
            st.session_state["selector_v1_2_result"] = run_research(pro, config)
        except Exception as exc:
            st.exception(exc)

    if "selector_v1_2_result" in st.session_state:
        render_results(st.session_state["selector_v1_2_result"])


if __name__ == "__main__":
    main()
