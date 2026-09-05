# -*- coding: utf-8 -*-
"""周线SKDJ信号有效性验证器（单文件独立版，直接覆盖 app.py 运行即可）。

这个版本不做选股、不做组合、不做资金管理、不做回测收益曲线。
它只回答一个问题：

    "周线SKDJ的K上穿25之后，股价表现真的比同期随便买一只科技股更好吗？"

核心方法是超额收益（edge），而不是绝对收益：
    edge = 信号样本未来N周平均收益 - 同期全池所有股票未来N周平均收益

牛市里随便买什么都涨，只看绝对收益会把大盘的功劳误认成信号的功劳。
只有信号组相对同期全池基准仍然领先，才说明信号本身携带择股信息。

买卖口径：信号周收盘产生信号 -> 下一周开盘买入 -> 持有N周后按当周收盘卖出。
不计交易成本（验证信号阶段，成本留到策略层再算）。

SKDJ 严格按通达信公式实现：
    LOWV := LLV(LOW, N);  HIGHV := HHV(HIGH, N);
    RSV  := EMA((CLOSE-LOWV)/(HIGHV-LOWV)*100, M);
    K : EMA(RSV, M);   D : MA(K, M);
其中通达信 EMA(X,M) 对应 pandas ewm(span=M)，MA(X,M) 对应 rolling(M).mean()。

行情缓存目录与主程序一致，之前下载过的数据不会重复下载。
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import pickle
import re
import shutil
import tempfile
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts

warnings.filterwarnings("ignore")

APP_TITLE = "周线SKDJ信号有效性验证器"
# 与主程序共用同一个缓存目录，避免重复下载行情
MARKET_CACHE_ROOT = "r1_trend_entry_market_cache_v2"
CACHE_SCHEMA_VERSION = 3
DOWNLOAD_WORKERS = 4
DATA_READY_HOUR_SHANGHAI = 18

# -----------------------------------------------------------------------------
# 数据层（与主程序完全一致：复权处理、缓存分片、股票池口径）
# -----------------------------------------------------------------------------
def clean_token_str(raw_token: str) -> str:
    if not raw_token:
        return ""
    return re.sub(r"[\s\u3000\ufeff\xa0\r\n]+", "", str(raw_token)).strip()

def _shanghai_now():
    return datetime.now(ZoneInfo("Asia/Shanghai"))

def _latest_data_ready_date(now_shanghai=None):
    """Tushare日线在交易日盘中并不完整；18点前只使用上一自然日。"""
    current = now_shanghai or _shanghai_now()
    ready_date = current.date()
    if current.hour < DATA_READY_HOUR_SHANGHAI:
        ready_date -= timedelta(days=1)
    return ready_date

def safe_tushare_call(func, max_retries: int = 3, sleep_time: float = 0.8, **kwargs):
    for attempt in range(max_retries):
        try:
            frame = func(**kwargs)
            if frame is not None and not frame.empty:
                return frame
        except Exception:
            pass
        time.sleep(sleep_time * (attempt + 1))
    return pd.DataFrame()

def verify_token_connection(token_str: str):
    if not token_str:
        return False, "Token为空，请在侧边栏填入Token。"
    try:
        ts.set_token(token_str)
        pro = ts.pro_api(token_str)
        end_dt = datetime.now().date()
        start_dt = end_dt - timedelta(days=14)
        frame = pro.trade_cal(
            exchange="SSE",
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
        )
        if frame is not None and not frame.empty:
            return True, "验证通过"
        return False, "Token校验未返回交易日历。"
    except Exception as exc:
        message = str(exc)
        if "token不对" in message or "-40001" in message:
            return False, "Token不正确。"
        return False, f"接口校验失败：{message}"

def parse_yyyymmdd(value: Any):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\.0$", "", str(value)).replace("-", "")
    return text if re.fullmatch(r"\d{8}", text) else None

def atomic_write_csv(frame: pd.DataFrame, path: str):
    target_dir = os.path.dirname(os.path.abspath(path)) or "."

# -----------------------------------------------------------------------------
# 科技股固定研究池
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600 * 24 * 7, show_spinner=False)
def load_custom_tech_whitelist(token: str):
    token_c = clean_token_str(token)
    if not token_c:
        return set(), {}, {}
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)

    basic_parts = []
    for status in ("L", "D", "P"):
        part = safe_tushare_call(
            pro.stock_basic,
            list_status=status,
            fields="ts_code,symbol,name,industry,market,list_date,delist_date",
        )
        if not part.empty:
            part = part.copy()
            part["_list_status"] = status
            basic_parts.append(part)
    if not basic_parts:
        return set(), {}, {}
    stock_basic = pd.concat(basic_parts, ignore_index=True).drop_duplicates("ts_code", keep="first")

    boards = {"主板", "创业板", "科创板"}
    valid = stock_basic[stock_basic["market"].isin(boards)].copy()
    current_bad_name = (
        valid["_list_status"].astype(str).eq("L")
        & valid["name"].astype(str).str.contains("ST|退", na=False)
    )
    valid = valid[~current_bad_name]
    valid = valid[~valid["ts_code"].astype(str).str.startswith("92")]

    core_l1 = {"电子", "计算机", "通信", "国防军工"}
    extended_l1 = {"机械设备", "电力设备", "医药生物", "汽车", "基础化工", "有色金属", "建筑材料"}
    keywords = {
        "半导体", "电子元件", "元件", "光学光电子", "消费电子", "电子化学品",
        "计算机设备", "电脑设备", "软件开发", "软件服务", "IT服务", "互联网",
        "信息安全", "通信设备", "通信服务", "军工电子", "航空装备", "航空航天",
        "航天装备", "自动化设备", "机器人", "激光设备", "工控设备", "仪器仪表",
        "电器仪表", "专用机械", "通用机械", "工业机械", "电池", "光伏设备",
        "风电设备", "电网设备", "电气设备", "电机", "医疗器械", "医疗保健",
        "生物制品", "汽车电子", "汽车配件", "金属新材料", "非金属材料",
        "新材料", "膜材料", "碳纤维", "小金属",
    }

    stock_sw_map: dict[str, str] = {}
    sw_indices = safe_tushare_call(pro.index_classify, level="L1", src="SW2021")
    if not sw_indices.empty:
        target = sw_indices[sw_indices["industry_name"].isin(core_l1 | extended_l1)]
        for _, sw_row in target.iterrows():
            members = safe_tushare_call(
                pro.index_member, index_code=sw_row["index_code"], is_new="Y"
            )
            if not members.empty:
                for code in members["con_code"].astype(str):
                    stock_sw_map[code] = str(sw_row["industry_name"])
            time.sleep(0.02)

    whitelist: set[str] = set()
    name_map: dict[str, str] = {}
    industry_map: dict[str, str] = {}
    for _, row in valid.iterrows():
        code = str(row["ts_code"])
        name = str(row["name"])
        basic_industry = "" if pd.isna(row["industry"]) else str(row["industry"])
        sw_l1 = stock_sw_map.get(code, "")
        include = False
        if sw_l1 in core_l1:
            include = True
        elif sw_l1 in extended_l1:
            include = (
                any(word in basic_industry for word in keywords)
                or basic_industry == ""
                or sw_l1 in {"机械设备", "电力设备", "医药生物"}
            )
        elif any(word in basic_industry for word in keywords):
            include = True
        if include:
            whitelist.add(code)
            name_map[code] = name
            industry_map[code] = sw_l1 or basic_industry or "未分类"
    return whitelist, name_map, industry_map


# -----------------------------------------------------------------------------
# 行情分片缓存
# -----------------------------------------------------------------------------
def _pool_cache_dir(whitelist_set: set[str]):
    pool_hash = hashlib.sha1("|".join(sorted(whitelist_set)).encode("utf-8")).hexdigest()[:12]
    cache_dir = os.path.join(MARKET_CACHE_ROOT, pool_hash)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir, pool_hash

def _valid_market_partition(payload: Any, trade_date: str, pool_hash: str):
    if not isinstance(payload, dict):
        return False
    version = int(payload.get("version", 0))
    if version not in {2, CACHE_SCHEMA_VERSION}:
        return False
    if payload.get("trade_date") != str(trade_date) or payload.get("pool_hash") != pool_hash:
        return False
    daily, basic = payload.get("daily"), payload.get("daily_basic")
    if not isinstance(daily, pd.DataFrame) or not isinstance(basic, pd.DataFrame):
        return False
    if int(payload.get("raw_daily_count", 0)) < 1000:
        return False
    required_daily = {"ts_code", "trade_date", "open", "high", "low", "close", "vol"}
    required_basic = {"ts_code", "trade_date", "circ_mv", "turnover_rate"}
    if daily.empty or not required_daily.issubset(daily.columns):
        return False
    if not ({"pct_chg", "pre_close"} & set(daily.columns)):
        return False
    if version == 2:
        adj = payload.get("adj")
        return (
            isinstance(adj, pd.DataFrame)
            and not adj.empty
            and int(payload.get("raw_adj_count", 0)) >= 1000
            and {"ts_code", "trade_date", "adj_factor"}.issubset(adj.columns)
            and required_basic.issubset(basic.columns)
        )
    if bool(payload.get("need_basic", False)):
        return not basic.empty and required_basic.issubset(basic.columns)
    return required_basic.issubset(basic.columns)

def _atomic_write_pickle(payload: Any, path: str):
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir
    )
    os.close(fd)
    try:
        with open(tmp_path, "wb") as file_obj:
            pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def _read_market_partition(path: str, trade_date: str, pool_hash: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as file_obj:
            payload = pickle.load(file_obj)
        return payload if _valid_market_partition(payload, trade_date, pool_hash) else None
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        return None

def _download_one_market_partition(
    token: str,
    trade_date: str,
    whitelist_set: set[str],
    pool_hash: str,
    cache_dir: str,
    need_basic: bool,
):
    """下载单日后立即原子落盘；线程之间不共享大表。"""
    pro = ts.pro_api(token)
    daily_fields = "ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount"
    daily_all = safe_tushare_call(
        pro.daily,
        trade_date=trade_date,
        fields=daily_fields,
    )
    # 个别Tushare节点对trade_date查询偶发返回空表，再用等价日期区间补一次。
    if daily_all.empty:
        daily_all = safe_tushare_call(
            pro.daily,
            start_date=trade_date,
            end_date=trade_date,
            fields=daily_fields,
        )
    basic_columns = [
        "ts_code",
        "trade_date",
        "turnover_rate",
        "volume_ratio",
        "circ_mv",
        "total_mv",
    ]
    if need_basic:
        basic_all = safe_tushare_call(
            pro.daily_basic,
            trade_date=trade_date,
            fields=",".join(basic_columns),
        )
        if basic_all.empty:
            basic_all = safe_tushare_call(
                pro.daily_basic,
                start_date=trade_date,
                end_date=trade_date,
                fields=",".join(basic_columns),
            )
    else:
        basic_all = pd.DataFrame(columns=basic_columns)

    daily = (
        daily_all[daily_all["ts_code"].isin(whitelist_set)].copy()
        if not daily_all.empty and "ts_code" in daily_all.columns
        else pd.DataFrame()
    )
    basic = (
        basic_all[basic_all["ts_code"].isin(whitelist_set)].copy()
        if not basic_all.empty and "ts_code" in basic_all.columns
        else pd.DataFrame(columns=basic_columns)
    )
    for column in basic_columns:
        if column not in basic.columns:
            basic[column] = pd.Series(dtype="object")
    payload = {
        "version": CACHE_SCHEMA_VERSION,
        "trade_date": trade_date,
        "pool_hash": pool_hash,
        "raw_daily_count": int(len(daily_all)),
        "need_basic": bool(need_basic),
        "daily": daily,
        "daily_basic": basic[basic_columns],
    }
    if not _valid_market_partition(payload, trade_date, pool_hash):
        return trade_date, False
    _atomic_write_pickle(payload, os.path.join(cache_dir, f"{trade_date}.pkl"))
    return trade_date, True

def sync_market_data_incrementally(
    start_date: str,
    end_date: str,
    token: str,
    whitelist_set: set[str],
    lease_heartbeat=None,
):
    token_c = clean_token_str(token)
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    calendar = safe_tushare_call(
        pro.trade_cal, exchange="SSE", start_date=start_date, end_date=end_date
    )
    if calendar.empty:
        return [], "", "", [], {}
    data_ready_str = _latest_data_ready_date().strftime("%Y%m%d")
    open_calendar = calendar[
        pd.to_numeric(calendar["is_open"], errors="coerce").eq(1)
        & (calendar["cal_date"].astype(str) <= data_ready_str)
    ].copy()
    open_calendar["cal_date"] = open_calendar["cal_date"].astype(str)
    open_calendar = open_calendar.sort_values("cal_date")
    valid_dates = open_calendar["cal_date"].tolist()
    open_calendar["week_key"] = pd.to_datetime(
        open_calendar["cal_date"], format="%Y%m%d", errors="coerce"
    ).dt.strftime("%G_%V")
    week_end_dates = set(
        open_calendar.dropna(subset=["week_key"])
        .groupby("week_key")["cal_date"]
        .max()
        .astype(str)
        .tolist()
    )
    cache_dir, pool_hash = _pool_cache_dir(whitelist_set)
    missing_dates = []
    for trade_date in valid_dates:
        payload = _read_market_partition(
            os.path.join(cache_dir, f"{trade_date}.pkl"), trade_date, pool_hash
        )
        basic_missing = (
            trade_date in week_end_dates
            and payload is not None
            and payload.get("daily_basic", pd.DataFrame()).empty
        )
        if payload is None or basic_missing:
            missing_dates.append(trade_date)

    failed_dates: list[str] = []
    downloaded_dates: list[str] = []
    if missing_dates:
        progress = st.progress(
            0,
            text=(
                f"4路并发补充{len(missing_dates)}个交易日；每完成一天立即保存，"
                "中断后只补未完成日期……"
            ),
        )
        futures = {}
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            for trade_date in missing_dates:
                future = executor.submit(
                    _download_one_market_partition,
                    token_c,
                    trade_date,
                    whitelist_set,
                    pool_hash,
                    cache_dir,
                    trade_date in week_end_dates,
                )
                futures[future] = trade_date
            for idx, future in enumerate(as_completed(futures), start=1):
                trade_date = futures[future]
                try:
                    _, succeeded = future.result()
                except Exception:
                    succeeded = False
                if succeeded:
                    downloaded_dates.append(trade_date)
                else:
                    failed_dates.append(trade_date)
                if idx % 4 == 0 or idx == len(missing_dates):
                    if callable(lease_heartbeat) and lease_heartbeat() is False:
                        raise RuntimeError("任务租约已经转移，停止本页的行情同步。")
                    progress.progress(
                        idx / len(missing_dates),
                        text=(
                            f"行情同步 {idx}/{len(missing_dates)}：已保存{len(downloaded_dates)}天，"
                            f"待重试{len(failed_dates)}天"
                        ),
                    )
        progress.empty()
    stats = {
        "calendar_days": len(valid_dates),
        "cached_days": len(valid_dates) - len(missing_dates),
        "downloaded_days": len(downloaded_dates),
        "failed_days": len(failed_dates),
        "failed_dates": ",".join(sorted(failed_dates)),
        "data_ready_through": data_ready_str,
        "weekly_basic_days": sum(item in week_end_dates for item in missing_dates),
    }
    return valid_dates, cache_dir, pool_hash, failed_dates, stats

def _build_market_index_from_partitions(
    valid_dates_key, cache_dir, pool_hash
):
    merged_parts = []
    for trade_date in valid_dates_key:
        payload = _read_market_partition(
            os.path.join(cache_dir, f"{trade_date}.pkl"), trade_date, pool_hash
        )
        if payload is None:
            continue
        day = payload["daily"].copy()
        basic = payload.get("daily_basic", pd.DataFrame())
        if isinstance(basic, pd.DataFrame) and not basic.empty:
            basic_cols = [
                column
                for column in (
                    "ts_code",
                    "trade_date",
                    "turnover_rate",
                    "volume_ratio",
                    "circ_mv",
                    "total_mv",
                )
                if column in basic.columns
            ]
            if {"ts_code", "trade_date"}.issubset(basic_cols):
                day = day.merge(
                    basic[basic_cols].drop_duplicates(["ts_code", "trade_date"]),
                    on=["ts_code", "trade_date"],
                    how="left",
                )
        merged_parts.append(day)
    if not merged_parts:
        return {}, pd.DataFrame(), []
    merged = pd.concat(merged_parts, ignore_index=True)
    del merged_parts
    merged["trade_date_str"] = merged["trade_date"].astype(str)
    merged = merged.drop_duplicates(["ts_code", "trade_date_str"], keep="last")
    merged = merged.sort_values(["ts_code", "trade_date_str"])
    available_dates = sorted(merged["trade_date_str"].unique().tolist())

    basic_columns = [
        column
        for column in ("turnover_rate", "volume_ratio", "circ_mv", "total_mv")
        if column in merged.columns
    ]
    if basic_columns:
        basic_raw = merged.loc[
            merged[basic_columns].notna().any(axis=1),
            ["trade_date_str", "ts_code", *basic_columns],
        ].copy()
        basic_indexed = basic_raw.drop_duplicates(
            ["trade_date_str", "ts_code"]
        ).set_index(["trade_date_str", "ts_code"])
    else:
        basic_indexed = pd.DataFrame()

    stock_qfq_dict: dict[str, pd.DataFrame] = {}
    for ts_code, group in merged.groupby("ts_code", sort=False):
        stock = group.copy().sort_values("trade_date_str")
        for column in ("open", "high", "low", "close", "pre_close"):
            if column in stock.columns:
                stock[f"raw_{column}"] = pd.to_numeric(stock[column], errors="coerce")
        raw_close = pd.to_numeric(stock["raw_close"], errors="coerce")
        raw_pre_close = (
            pd.to_numeric(stock["raw_pre_close"], errors="coerce")
            if "raw_pre_close" in stock.columns
            else raw_close.shift(1)
        )
        pct_chg = (
            pd.to_numeric(stock["pct_chg"], errors="coerce")
            if "pct_chg" in stock.columns
            else pd.Series(np.nan, index=stock.index, dtype="float64")
        )
        fallback_pct = (raw_close / raw_pre_close.replace(0, np.nan) - 1.0) * 100.0
        pct_chg = pct_chg.fillna(fallback_pct)
        growth = (1.0 + pct_chg / 100.0).where(lambda values: values > 0)
        continuous_close = pd.Series(np.nan, index=stock.index, dtype="float64")
        if not raw_close.empty and pd.notna(raw_close.iloc[0]) and raw_close.iloc[0] > 0:
            continuous_close.iloc[0] = raw_close.iloc[0]
            if len(stock) > 1:
                continuous_close.iloc[1:] = (
                    raw_close.iloc[0] * growth.iloc[1:].fillna(1.0).cumprod()
                )
        price_scale = continuous_close / raw_close.replace(0, np.nan)
        for column in ("open", "high", "low", "close"):
            raw_column = f"raw_{column}"
            if raw_column in stock.columns:
                stock[column] = pd.to_numeric(stock[raw_column], errors="coerce") * price_scale
        stock["pre_close"] = continuous_close.shift(1)
        if len(stock) and "raw_pre_close" in stock.columns:
            stock.iloc[0, stock.columns.get_loc("pre_close")] = (
                _safe_float(stock.iloc[0].get("raw_pre_close"))
                * _safe_float(price_scale.iloc[0], 1.0)
            )
        # 周末daily_basic给出流通市值，由此反推流通股数并补算每天换手率；
        # 因而仍可保持原版“周换手率=日换手率之和”的评分口径。
        if "circ_mv" in stock.columns:
            circ_mv = pd.to_numeric(stock["circ_mv"], errors="coerce")
            implied_float_shares = (
                circ_mv * 10000.0 / raw_close.replace(0, np.nan)
            ).ffill().bfill()
            daily_turnover = (
                pd.to_numeric(stock.get("vol"), errors="coerce")
                * 10000.0
                / implied_float_shares.replace(0, np.nan)
            )
            if "turnover_rate" in stock.columns:
                existing_turnover = pd.to_numeric(stock["turnover_rate"], errors="coerce")
                stock["turnover_rate"] = existing_turnover.fillna(daily_turnover)
            else:
                stock["turnover_rate"] = daily_turnover
        for column in (
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "vol",
            "amount",
            "turnover_rate",
            "volume_ratio",
            "circ_mv",
            "total_mv",
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "raw_pre_close",
        ):
            if column in stock.columns:
                stock[column] = pd.to_numeric(stock[column], errors="coerce").astype("float32")
        stock_qfq_dict[str(ts_code)] = stock.set_index("trade_date_str")
    del merged
    gc.collect()
    return stock_qfq_dict, basic_indexed, available_dates

def load_optimized_market_data(
    start_date: str, end_date: str, token: str, whitelist_keys, lease_heartbeat=None
):
    whitelist_set = set(whitelist_keys)
    valid_dates, cache_dir, pool_hash, failed_dates, sync_stats = sync_market_data_incrementally(
        start_date, end_date, token, whitelist_set, lease_heartbeat=lease_heartbeat
    )
    if not valid_dates:
        return {}, pd.DataFrame(), [], [], failed_dates, sync_stats
    stocks, basic, available_dates = _build_market_index_from_partitions(
        tuple(valid_dates), cache_dir, pool_hash
    )
    return stocks, basic, valid_dates, available_dates, failed_dates, sync_stats


def _safe_float(value: Any, default: float = np.nan):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default

# -----------------------------------------------------------------------------
# 周线聚合与SKDJ指标
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


def add_skdj_suffixed(
    weekly: pd.DataFrame, n_period: int, m_period: int, suffix: str
) -> pd.DataFrame:
    """同一份周线上算多组N参数，列名加后缀，用于N参数扫描。"""
    low = pd.to_numeric(weekly["low"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    close = pd.to_numeric(weekly["close"], errors="coerce")
    low_n = low.rolling(n_period).min()
    high_n = high.rolling(n_period).max()
    raw_rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100.0
    rsv = raw_rsv.ewm(span=m_period, adjust=False).mean()
    weekly[f"K{suffix}"] = rsv.ewm(span=m_period, adjust=False).mean()
    weekly[f"D{suffix}"] = weekly[f"K{suffix}"].rolling(m_period).mean()
    return weekly


def add_skdj(weekly: pd.DataFrame, n_period: int, m_period: int) -> pd.DataFrame:
    """严格按通达信公式：
        LOWV := LLV(LOW,N); HIGHV := HHV(HIGH,N);
        RSV := EMA((CLOSE-LOWV)/(HIGHV-LOWV)*100, M);
        K : EMA(RSV,M);  D : MA(K,M);
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
    """对信号周 i：买入价 = 第 i+1 周开盘价，卖出价 = 第 i+N 周收盘价。"""
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
# 统计对比
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
    """信号组 vs 全池基准组，核心输出是超额收益。"""
    rows = []
    for n_weeks in horizons:
        column = f"Fwd_{n_weeks}W_pct"
        signal_values = pd.to_numeric(
            panel.loc[signal_mask, column], errors="coerce"
        ).dropna()
        base_values = pd.to_numeric(panel[column], errors="coerce").dropna()
        signal_stats = describe_group(signal_values)
        base_stats = describe_group(base_values)

        # 粗略显著性参考：同一周内个股涨跌高度相关，真实显著性低于该值，
        # 因此它只用来排除"明显是噪声"，不能当作严格统计检验。
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
    """按年度拆分——判断优势是长期稳定，还是只集中在某一两年。"""
    column = f"Fwd_{hold_weeks}W_pct"
    work = panel.copy()
    work["_year"] = work["Signal_Date"].astype(str).str[:4]
    work["_is_signal"] = np.asarray(signal_mask)
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
                    signal_mean - base_values.mean() if len(signal_values) else np.nan
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
# Streamlit 主程序
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🔍 {APP_TITLE}")
    st.caption(
        "本工具不做选股、不做组合、不做资金管理，只回答一个问题："
        "周线SKDJ信号之后的表现，是否真的优于同期随便买一只科技股。"
    )
    st.info(
        "**为什么必须看超额收益**：牛市里随便买什么都涨，只看绝对收益会把大盘的功劳"
        "误认成信号的功劳。只有信号组相对同期全池基准仍然领先，才说明信号本身携带择股信息。"
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
        n_period = st.number_input(
            "N（LLV/HHV周期）", value=6, min_value=2, max_value=60, step=1
        )
        m_period = st.number_input(
            "M（EMA/MA周期）", value=3, min_value=2, max_value=30, step=1
        )
        cross_level = st.number_input(
            "上穿判定线",
            value=25.0,
            min_value=1.0,
            max_value=90.0,
            step=5.0,
            help="你的原始想法是25。脚本会同时测试附近几个值，检验结论是否只在25成立。",
        )

        st.markdown("---")
        st.subheader("股票池硬条件")
        min_price = st.number_input("最低股价（元）", value=10.0, min_value=0.0, step=1.0)
        min_mv = st.number_input("最低流通市值（亿元）", value=100.0, min_value=0.0, step=10.0)
        max_mv = st.number_input(
            "最高流通市值（亿元）", value=1000.0, min_value=100.0, step=100.0
        )

        st.markdown("---")
        clear_cache_clicked = st.button("清空行情缓存")
        run_clicked = st.button("开始验证", type="primary")

    if clear_cache_clicked:
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.success("行情缓存已清空。")

    if not run_clicked:
        st.markdown(
            """
### 这个工具会给你三张表

**表1 · 信号 vs 基准**  
最重要的是 **超额收益%** 这一列：
- 明显为正（且样本足够多）→ 信号确实有效，值得在它上面搭策略、加日线择时
- 接近 0 → 信号只是跟着大盘走，没有独立价值，再精妙的买点也救不回来
- 为负 → 信号方向是反的

**表2 · 分年度拆解**  
判断优势是长期稳定，还是只靠某一两年撑起来。如果只有一年为正、其余年份为负，
那和之前四年回测遇到的问题是同一个——收益集中在极少数窗口，不可复制。

**表3 · 参数敏感性**  
同时测试上穿 15/20/25/30/35 各条线，以及"上穿且K>D"等变体。
如果只有25有效、旁边的20和30都失效，那这个25大概率是数据里的巧合而非规律。

---
填好左侧配置后点击「开始验证」。行情缓存与你原来的主程序共用，之前下载过的不会重复下载。
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
    if start_input >= end_input:
        st.error("开始日期必须早于结束日期。")
        return

    horizons = [1, 2, 3, 4]
    start_date = start_input.strftime("%Y%m%d")
    end_date = end_input.strftime("%Y%m%d")
    # 预留周线指标预热窗口 + 未来收益观察窗口
    fetch_start = (pd.Timestamp(start_input) - timedelta(days=400)).strftime("%Y%m%d")
    fetch_end = (pd.Timestamp(end_input) + timedelta(days=60)).strftime("%Y%m%d")

    with st.spinner("构建科技股研究池……"):
        whitelist_set, name_map, industry_map = load_custom_tech_whitelist(token_clean)
    if not whitelist_set:
        st.error("未取得科技股研究池，请检查Token权限。")
        return
    st.success(f"科技股研究池：{len(whitelist_set)}只")

    with st.spinner("加载行情（复用已有缓存）……"):
        (
            stocks,
            basic_indexed,
            _,
            _,
            failed_dates,
            sync_stats,
        ) = load_optimized_market_data(
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

    # N参数扫描范围：用来检验你选定的N是不是一个"幸运的尖峰"。
    # 如果超额收益随N平滑变化，说明是真实规律；如果只有某个N突出、
    # 相邻的N都很差，那这个N大概率是数据里的巧合。
    sweep_n_values = sorted({4, 5, 6, 7, 8, 9, 10, 12, int(n_period)})
    max_warmup = max(sweep_n_values) + int(m_period) + 12

    progress = st.progress(0.0, text="计算周线SKDJ与未来收益……")
    panel_parts = []
    codes = sorted(stocks.keys())
    for idx, ts_code in enumerate(codes):
        weekly = build_weekly_bars(stocks[ts_code])
        if weekly.empty or len(weekly) < max_warmup:
            continue
        weekly = add_skdj(weekly, int(n_period), int(m_period))
        for sweep_n in sweep_n_values:
            weekly = add_skdj_suffixed(
                weekly, sweep_n, int(m_period), f"_N{sweep_n}"
            )
        weekly = add_forward_returns(weekly, horizons)
        weekly["ts_code"] = ts_code
        weekly["Signal_Date"] = weekly["trade_date_str"].astype(str)
        keep = ["ts_code", "Signal_Date", "close", "K", "D", "Entry_Open_Next_Week"]
        if "raw_close" in weekly.columns:
            keep.insert(3, "raw_close")
        keep += [f"Fwd_{n}W_pct" for n in horizons]
        keep += [f"K_N{sweep_n}" for sweep_n in sweep_n_values]
        panel_parts.append(weekly[keep])
        if idx % 50 == 0:
            progress.progress(
                min((idx + 1) / len(codes), 1.0),
                text=f"计算周线SKDJ与未来收益……{idx + 1}/{len(codes)}",
            )
    progress.empty()

    del stocks
    gc.collect()

    if not panel_parts:
        st.error("没有足够长的周线数据。")
        return
    panel = pd.concat(panel_parts, ignore_index=True)
    del panel_parts
    gc.collect()

    # 应用股票池硬条件
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
            panel = panel[circ_mv_billion.between(min_mv, max_mv) | circ_mv_billion.isna()]

    panel = panel.dropna(subset=["K", f"Fwd_{horizons[0]}W_pct"]).reset_index(drop=True)
    if panel.empty:
        st.error("过滤后没有可用样本，请放宽股票池条件或时间范围。")
        return

    st.markdown("---")
    st.header("验证结果")
    st.caption(
        f"全池基准样本：{len(panel):,} 个「个股-周」观测，"
        f"覆盖 {panel['Signal_Date'].min()} — {panel['Signal_Date'].max()}。"
        "每个信号都与同一批基准对比。"
    )

    k_now = pd.to_numeric(panel["K"], errors="coerce")
    k_prev = k_now.groupby(panel["ts_code"]).shift(1)
    d_now = pd.to_numeric(panel["D"], errors="coerce")

    # 表1：主假设
    main_mask = ((k_prev <= cross_level) & (k_now > cross_level)).fillna(False)
    main_table = compare_signal_vs_baseline(
        panel, main_mask, horizons, f"K上穿{cross_level:.0f}"
    )
    st.subheader(f"表1 · 你的假设：K上穿{cross_level:.0f}")
    st.dataframe(main_table.round(3), width="stretch", hide_index=True)
    st.caption(
        "**看「超额收益%(核心)」这一列**：明显为正=信号有效；接近0=只是跟随大盘；为负=方向相反。"
        "粗略t值仅用于排除明显噪声（同周个股涨跌高度相关，真实显著性低于该值），"
        "|t|<2 基本可认为没有说服力。"
    )

    # 表2：分年度
    st.subheader("表2 · 分年度拆解（优势稳定吗？）")
    year_table = yearly_breakdown(panel, main_mask, 3)
    st.dataframe(year_table.round(3), width="stretch", hide_index=True)
    st.caption(
        "以持有3周为例。如果超额收益只有一两年为正、其余年份为负，"
        "说明优势不可复制——这正是之前四年回测暴露的问题。"
    )

    # 表3：参数敏感性
    st.subheader("表3 · 参数敏感性（25这条线是规律还是巧合？）")
    variant_tables = []
    for level in (15.0, 20.0, 25.0, 30.0, 35.0):
        mask = ((k_prev <= level) & (k_now > level)).fillna(False)
        variant_tables.append(
            compare_signal_vs_baseline(panel, mask, [3], f"K上穿{level:.0f}")
        )
    cross_and_kd = (
        (k_prev <= cross_level) & (k_now > cross_level) & (k_now > d_now)
    ).fillna(False)
    variant_tables.append(
        compare_signal_vs_baseline(
            panel, cross_and_kd, [3], f"K上穿{cross_level:.0f} 且 K>D"
        )
    )
    low_zone_turn = ((k_prev <= k_now) & (k_now <= cross_level)).fillna(False)
    variant_tables.append(
        compare_signal_vs_baseline(
            panel, low_zone_turn, [3], f"K在{cross_level:.0f}下方拐头（旧R6近似口径）"
        )
    )
    variant_table = pd.concat(variant_tables, ignore_index=True)
    st.dataframe(variant_table.round(3), width="stretch", hide_index=True)
    st.caption(
        "全部按持有3周计算。**如果只有25有效、20和30都失效，那25大概率是巧合而非规律**——"
        "真实的市场规律不会对阈值这么敏感。最后一行是旧R6分支的近似口径，"
        "可以直接看出它和你的原始想法差别有多大。"
    )

    # 表4：N参数扫描
    st.subheader("表4 · N参数扫描（你选的N是真规律还是幸运尖峰？）")
    sweep_rows = []
    for sweep_n in sweep_n_values:
        column = f"K_N{sweep_n}"
        if column not in panel.columns:
            continue
        k_sweep = pd.to_numeric(panel[column], errors="coerce")
        k_sweep_prev = k_sweep.groupby(panel["ts_code"]).shift(1)
        cross_mask = (
            (k_sweep_prev <= cross_level) & (k_sweep > cross_level)
        ).fillna(False)
        turn_mask = (
            (k_sweep_prev <= k_sweep) & (k_sweep <= cross_level)
        ).fillna(False)
        for mask, definition in (
            (cross_mask, f"K上穿{cross_level:.0f}"),
            (turn_mask, f"K在{cross_level:.0f}下方拐头"),
        ):
            one = compare_signal_vs_baseline(panel, mask, [3], definition)
            one.insert(0, "N", sweep_n)
            sweep_rows.append(one)
    sweep_table = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    if not sweep_table.empty:
        for definition in sweep_table["信号定义"].unique():
            subset = sweep_table[sweep_table["信号定义"] == definition]
            st.markdown(f"**{definition}**")
            st.dataframe(
                subset[
                    ["N", "信号样本数", "超额收益%(核心)", "超额中位收益%",
                     "胜率差%", "粗略t值"]
                ].round(3),
                width="stretch",
                hide_index=True,
            )
    st.caption(
        "全部按持有3周计算，M保持不变。**看超额收益随N的变化是否平滑**："
        "如果是单调或平缓的曲线（例如N越小越好），说明背后有真实逻辑（N越短指标越灵敏、"
        "越早捕捉拐点）；如果你选的那个N特别突出、相邻的N却明显更差，"
        "那这个N大概率是在历史数据上试出来的巧合，换一段行情就会失效。"
    )

    # 导出
    st.markdown("---")
    signal_rows = panel.loc[main_mask].copy()
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
            "05_n_period_sweep.csv",
            sweep_table.to_csv(index=False, encoding="utf-8-sig"),
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
