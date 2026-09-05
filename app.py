# -*- coding: utf-8 -*-
"""周线SKDJ最简策略回测器（单文件独立版，直接覆盖 app.py 运行）。

选股只用一个信号，不含任何其他条件：
    周线SKDJ  K <= 阈值  且  K >= 上周K（低位拐头）  且  K > D

该信号已由信号验证器在4年、24万个「个股-周」观测上验证：
    N=4, K<=20, 要求K>D, 持有3周
    -> 平均收益2.07%、胜率56.2%、相对全池超额+1.63%、t=9.25

本回测把策略拆成三层，用来分辨「策略好」与「运气好」：
    1. 信号层：无资金约束，每周买入当周全部信号 -> 策略本身的优势上限
    2. 三仓层：真实资金约束，每周最多N只      -> 实际能拿到多少
    3. 蒙特卡洛：把"选哪只"随机化重复数百次   -> 运气区间与排序规则的真实贡献

三仓存在路径依赖：某只股票占住仓位会挡掉后面的信号，这纯属时间巧合。
只看三仓的单次收益无法分辨优势来源，因此必须配合蒙特卡洛分布一起看。

行情缓存目录与之前一致，已下载的数据不会重复下载。
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

APP_TITLE = "周线SKDJ最简策略回测（三层分离验证）"
MARKET_CACHE_ROOT = "r1_trend_entry_market_cache_v2"
CACHE_SCHEMA_VERSION = 3
DOWNLOAD_WORKERS = 4
DATA_READY_HOUR_SHANGHAI = 18

# -----------------------------------------------------------------------------
# 数据层（与之前完全一致：复权、缓存分片、股票池口径）
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



# -----------------------------------------------------------------------------
# SKDJ 与信号构造
# -----------------------------------------------------------------------------
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


def build_stock_signals(
    weekly: pd.DataFrame,
    ts_code: str,
    level: float,
    hold_weeks: int,
    require_k_above_d: bool,
) -> pd.DataFrame:
    """信号定义（经四年24万观测验证的最优口径）：
        K <= 阈值   且   K >= 上周K（低位拐头）   且   K > D（可选确认）

    买卖口径：信号周收盘出信号 -> 下一周开盘买入 -> 持有hold_weeks周后按当周收盘卖出。
    与信号验证器完全一致，保证两边结果可直接对照。
    """
    k_now = pd.to_numeric(weekly["K"], errors="coerce")
    k_prev = k_now.shift(1)
    d_now = pd.to_numeric(weekly["D"], errors="coerce")

    signal = (k_prev <= k_now) & (k_now <= level)
    if require_k_above_d:
        signal = signal & (k_now > d_now)
    signal = signal.fillna(False)

    open_price = pd.to_numeric(weekly["open"], errors="coerce")
    close_price = pd.to_numeric(weekly["close"], errors="coerce")
    dates = weekly["trade_date_str"].astype(str)

    entry_open = open_price.shift(-1)
    exit_close = close_price.shift(-hold_weeks)

    frame = pd.DataFrame(
        {
            "ts_code": ts_code,
            "Signal_Week": dates,
            "Entry_Week": dates.shift(-1),
            "Exit_Week": dates.shift(-hold_weeks),
            "K": k_now,
            "D": d_now,
            "Entry_Open": entry_open,
            "Exit_Close": exit_close,
        }
    )
    # 持仓期内每周收盘，用于组合按周盯市
    for step in range(1, hold_weeks + 1):
        frame[f"Path_{step}"] = close_price.shift(-step)

    frame["Return_pct"] = (
        exit_close / entry_open.replace(0, np.nan) - 1.0
    ) * 100.0
    frame = frame[signal]
    frame = frame.dropna(subset=["Entry_Week", "Exit_Week", "Return_pct"])
    return frame


# -----------------------------------------------------------------------------
# 回测层一：信号层（无资金约束）——衡量策略本身的纯粹优势
# -----------------------------------------------------------------------------
def backtest_signal_layer(
    signals: pd.DataFrame, week_index: dict, hold_weeks: int, cost_pct: float
):
    """把资金分成hold_weeks份阶梯（每周投一份），每份等权买入当周全部信号。

    与三仓层使用完全相同的资金投放节奏，唯一区别是每份资金买入当周所有信号
    而不是只买1只。因此两者之差 = 集中持股（只买1只）带来的影响。
    """
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()
    cohort = (
        signals.groupby("Entry_Week")["Return_pct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Cohort_Return_pct", "count": "Signal_Count"})
    )
    cohort["Entry_Index"] = cohort["Entry_Week"].map(week_index)
    cohort = cohort.dropna(subset=["Entry_Index"]).sort_values("Entry_Index")
    cohort["Entry_Index"] = cohort["Entry_Index"].astype(int)
    cohort["Net_Return_pct"] = cohort["Cohort_Return_pct"] - cost_pct

    # hold_weeks条阶梯，各自独立复利
    tranche_values = {j: 1.0 for j in range(hold_weeks)}
    rows = []
    for _, row in cohort.iterrows():
        tranche = int(row["Entry_Index"]) % hold_weeks
        tranche_values[tranche] *= 1.0 + row["Net_Return_pct"] / 100.0
        rows.append(
            {
                "Entry_Week": row["Entry_Week"],
                "阶梯": tranche + 1,
                "当周信号数": int(row["Signal_Count"]),
                "当周等权收益%": row["Net_Return_pct"],
                "组合净值": sum(tranche_values.values()) / hold_weeks,
            }
        )
    curve = pd.DataFrame(rows)
    total_return = (sum(tranche_values.values()) / hold_weeks - 1.0) * 100.0
    returns = cohort["Net_Return_pct"]
    summary = pd.DataFrame(
        [
            {
                "层级": "信号层（无资金约束，每周买入全部信号）",
                "交易笔数": int(len(signals)),
                "调仓周数": int(len(cohort)),
                "每周平均信号数": float(cohort["Signal_Count"].mean()),
                "单笔平均收益%": float(
                    signals["Return_pct"].mean() - cost_pct
                ),
                "单笔胜率%": float(
                    (signals["Return_pct"] - cost_pct > 0).mean() * 100.0
                ),
                "周期等权平均收益%": float(returns.mean()),
                "周期胜率%": float((returns > 0).mean() * 100.0),
                "总收益率%": total_return,
            }
        ]
    )
    return summary, curve


# -----------------------------------------------------------------------------
# 回测层二：三仓组合（有资金约束）
# -----------------------------------------------------------------------------
def backtest_three_slot(
    signals: pd.DataFrame,
    week_index: dict,
    hold_weeks: int,
    cost_pct: float,
    slot_count: int,
    order_mode: str = "K",
    seed: int = 0,
):
    """三仓逐仓复投。order_mode='K' 按K值升序优选；'random' 随机顺序。

    随机顺序用于蒙特卡洛：它保持了完全相同的资金约束和路径依赖结构，
    只把"选哪只"变成随机，从而分离出「排序规则的贡献」与「运气的贡献」。
    """
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0

    work = signals.copy()
    work["Entry_Index"] = work["Entry_Week"].map(week_index)
    work["Exit_Index"] = work["Exit_Week"].map(week_index)
    work = work.dropna(subset=["Entry_Index", "Exit_Index"])
    work["Entry_Index"] = work["Entry_Index"].astype(int)
    work["Exit_Index"] = work["Exit_Index"].astype(int)

    rng = np.random.default_rng(seed)
    if order_mode == "random":
        work["_order"] = rng.random(len(work))
    else:
        work["_order"] = pd.to_numeric(work["K"], errors="coerce").fillna(999.0)

    by_entry = {
        idx: group.sort_values(["_order", "ts_code"], kind="mergesort")
        for idx, group in work.groupby("Entry_Index", sort=True)
    }

    slot_value = [1.0 / slot_count] * slot_count
    slot_free_at = [0] * slot_count
    slot_holding = [None] * slot_count
    trades = []
    taken_codes_by_slot = {}

    for entry_index in sorted(by_entry.keys()):
        group = by_entry[entry_index]
        for _, row in group.iterrows():
            free_slots = [
                i for i in range(slot_count) if slot_free_at[i] <= entry_index
            ]
            held_now = {
                taken_codes_by_slot[i]
                for i in range(slot_count)
                if slot_free_at[i] > entry_index and i in taken_codes_by_slot
            }
            if str(row["ts_code"]) in held_now:
                trades.append({**row.to_dict(), "执行": "跳过", "原因": "已持有同股"})
                continue
            if not free_slots:
                trades.append({**row.to_dict(), "执行": "跳过", "原因": "三仓已满"})
                continue
            slot = free_slots[0]
            net_return = float(row["Return_pct"]) - cost_pct
            before = slot_value[slot]
            slot_value[slot] = before * (1.0 + net_return / 100.0)
            slot_free_at[slot] = int(row["Exit_Index"]) + 1
            taken_codes_by_slot[slot] = str(row["ts_code"])
            trades.append(
                {
                    **row.to_dict(),
                    "执行": "买入",
                    "原因": "",
                    "仓位": slot + 1,
                    "净收益%": net_return,
                    "仓位买入前": before,
                    "仓位卖出后": slot_value[slot],
                }
            )

    ledger = pd.DataFrame(trades)
    total_return = (sum(slot_value) - 1.0) * 100.0
    bought = (
        ledger[ledger["执行"].eq("买入")]
        if not ledger.empty and "执行" in ledger.columns
        else pd.DataFrame()
    )
    net = (
        pd.to_numeric(bought["净收益%"], errors="coerce").dropna()
        if not bought.empty
        else pd.Series(dtype=float)
    )
    summary = pd.DataFrame(
        [
            {
                "层级": f"{slot_count}仓组合（{'按K值优选' if order_mode == 'K' else '随机选取'}）",
                "完整信号": int(len(work)),
                "实际买入": int(len(bought)),
                "仓位冲突跳过": int(len(ledger) - len(bought)) if not ledger.empty else 0,
                "单笔平均收益%": float(net.mean()) if len(net) else np.nan,
                "单笔中位收益%": float(net.median()) if len(net) else np.nan,
                "单笔胜率%": float((net > 0).mean() * 100.0) if len(net) else np.nan,
                "总收益率%": total_return,
            }
        ]
    )
    return summary, ledger, total_return


def monte_carlo_slots(
    signals: pd.DataFrame,
    week_index: dict,
    hold_weeks: int,
    cost_pct: float,
    slot_count: int,
    runs: int,
    progress_callback=None,
):
    """重复N次随机选取的三仓回测，得到"纯运气"情况下的收益分布。

    把按K值优选的真实结果放到这个分布里比较：
      - 落在分布中间 -> 排序规则没有贡献，结果主要由运气决定
      - 落在分布右尾 -> 排序规则确实带来了额外价值
    分布本身的宽度，就是三仓路径依赖引入的运气成分大小。
    """
    outcomes = []
    for run in range(runs):
        _, _, total = backtest_three_slot(
            signals, week_index, hold_weeks, cost_pct, slot_count,
            order_mode="random", seed=run + 1,
        )
        outcomes.append(total)
        if progress_callback and (run % 10 == 0 or run == runs - 1):
            progress_callback((run + 1) / runs)
    return np.array(outcomes, dtype=float)


# -----------------------------------------------------------------------------
# Streamlit 主程序
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🧪 {APP_TITLE}")
    st.caption(
        "只用一个信号：周线SKDJ低位拐头。不加MACD、不加ATR、不加六因子、不加市场状态分支。"
    )
    st.info(
        "**本工具把策略拆成三层分别测量**，用来回答"
        "「三仓买入法是否引入了运气成分」：\n\n"
        "1. **信号层**：不受资金约束，每周买入当周全部信号 → 策略本身的纯粹优势\n"
        "2. **三仓层**：真实资金约束，每周最多3只 → 实际能拿到多少\n"
        "3. **蒙特卡洛**：把「选哪只」改成随机，重复数百次 → 运气区间有多宽、"
        "按K值排序到底有没有用"
    )

    with st.sidebar:
        st.header("策略配置")
        try:
            secret_token = st.secrets.get("TUSHARE_TOKEN", "")
        except Exception:
            secret_token = ""
        token_input = st.text_input("Tushare Token", value=secret_token, type="password")

        today = pd.Timestamp.now().date()
        start_input = st.date_input("开始日期", value=today - timedelta(days=365 * 4))
        end_input = st.date_input("结束日期", value=today)

        st.markdown("---")
        st.subheader("信号参数（已由信号验证器确定）")
        n_period = st.number_input("N（LLV/HHV周期）", value=4, min_value=2, max_value=60, step=1)
        m_period = st.number_input("M（EMA/MA周期）", value=3, min_value=2, max_value=30, step=1)
        level = st.number_input(
            "K值阈值（低位判定）", value=20.0, min_value=1.0, max_value=90.0, step=5.0,
            help="验证结果：15最强但信号偏少，20是收益与信号密度的平衡点。",
        )
        require_kd = st.checkbox(
            "要求 K > D（确认）", value=True,
            help="验证显示加此确认后平均收益从1.79%升到2.07%、胜率升到56.2%。",
        )
        hold_weeks = st.number_input(
            "持有周数", value=3, min_value=1, max_value=8, step=1,
            help="验证显示3-4周最优；1周几乎无效。",
        )

        st.markdown("---")
        st.subheader("资金与成本")
        slot_count = st.number_input("仓位数", value=3, min_value=1, max_value=10, step=1)
        cost_pct = st.number_input(
            "往返交易成本%", value=0.20, min_value=0.0, max_value=2.0, step=0.05
        )
        mc_runs = st.number_input(
            "蒙特卡洛次数", value=200, min_value=20, max_value=1000, step=20,
            help="次数越多分布越稳定，200次通常足够。",
        )

        st.markdown("---")
        st.subheader("股票池硬条件")
        min_price = st.number_input("最低股价（元）", value=10.0, min_value=0.0, step=1.0)
        min_mv = st.number_input("最低流通市值（亿元）", value=100.0, min_value=0.0, step=10.0)
        max_mv = st.number_input("最高流通市值（亿元）", value=1000.0, min_value=100.0, step=100.0)

        st.markdown("---")
        clear_cache_clicked = st.button("清空行情缓存")
        run_clicked = st.button("开始回测", type="primary")

    if clear_cache_clicked:
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.success("行情缓存已清空。")

    if not run_clicked:
        st.markdown(
            """
### 为什么要这样拆开测

三仓买入法有一个内在问题：**哪只股票在哪天占住了仓位，会影响后面能不能买到别的信号**。
这是纯粹的时间巧合，和信号质量无关。在之前的四年回测里已经出现过实证——
因"三仓已满"被跳过的交易胜率56%，实际买入的胜率80%，差了24个百分点，
而跳过与否完全取决于运气。

所以直接看三仓的最终收益，是**分不清"策略好"和"运气好"**的。

这个工具的做法：
- **信号层**给出策略的天花板（如果资金无限）
- **三仓层**给出真实约束下的结果
- **蒙特卡洛**把"选哪只"随机化重复数百次，画出运气的分布区间

三个数字放在一起，就能明确回答：三仓损失了多少、排序规则值多少、运气占多大比重。

---
默认参数已是信号验证器确定的最优配置（N=4、K≤20、要求K>D、持有3周）。
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

    start_date = start_input.strftime("%Y%m%d")
    end_date = end_input.strftime("%Y%m%d")
    fetch_start = (pd.Timestamp(start_input) - timedelta(days=400)).strftime("%Y%m%d")
    fetch_end = (pd.Timestamp(end_input) + timedelta(days=90)).strftime("%Y%m%d")

    with st.spinner("构建科技股研究池……"):
        whitelist_set, name_map, industry_map = load_custom_tech_whitelist(token_clean)
    if not whitelist_set:
        st.error("未取得科技股研究池，请检查Token权限。")
        return
    st.success(f"科技股研究池：{len(whitelist_set)}只")

    with st.spinner("加载行情（复用已有缓存）……"):
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

    progress = st.progress(0.0, text="计算信号……")
    signal_parts = []
    all_weeks = set()
    codes = sorted(stocks.keys())
    for idx, ts_code in enumerate(codes):
        weekly = build_weekly_bars(stocks[ts_code])
        if weekly.empty or len(weekly) < int(n_period) + int(m_period) + int(hold_weeks) + 8:
            continue
        weekly = add_skdj(weekly, int(n_period), int(m_period))
        all_weeks.update(weekly["trade_date_str"].astype(str).tolist())
        rows = build_stock_signals(
            weekly, ts_code, float(level), int(hold_weeks), bool(require_kd)
        )
        if not rows.empty:
            # 只保留信号周落在回测区间内的
            rows = rows[
                (rows["Signal_Week"] >= start_date) & (rows["Signal_Week"] <= end_date)
            ]
            if not rows.empty:
                signal_parts.append(rows)
        if idx % 50 == 0:
            progress.progress(
                min((idx + 1) / len(codes), 1.0),
                text=f"计算信号……{idx + 1}/{len(codes)}",
            )
    progress.empty()
    del stocks
    gc.collect()

    if not signal_parts:
        st.error("回测区间内没有产生任何信号，请放宽阈值或时间范围。")
        return
    signals = pd.concat(signal_parts, ignore_index=True)
    del signal_parts
    gc.collect()

    # 股票池硬条件（按信号周的市值与股价）
    if not basic_indexed.empty:
        basic_reset = basic_indexed.reset_index().rename(
            columns={"trade_date_str": "Signal_Week"}
        )
        keep = [c for c in ("Signal_Week", "ts_code", "circ_mv") if c in basic_reset.columns]
        if len(keep) == 3:
            signals = signals.merge(
                basic_reset[keep].drop_duplicates(["Signal_Week", "ts_code"]),
                on=["Signal_Week", "ts_code"], how="left",
            )
            mv_billion = pd.to_numeric(signals["circ_mv"], errors="coerce") / 10000.0
            signals = signals[mv_billion.between(min_mv, max_mv) | mv_billion.isna()]
    signals = signals[
        pd.to_numeric(signals["Entry_Open"], errors="coerce") >= min_price
    ]
    signals = signals.reset_index(drop=True)
    if signals.empty:
        st.error("过滤后没有信号，请放宽股票池条件。")
        return

    week_list = sorted(all_weeks)
    week_index = {week: i for i, week in enumerate(week_list)}

    signals["name"] = signals["ts_code"].map(name_map)
    signals["Industry"] = signals["ts_code"].map(industry_map)

    st.markdown("---")
    st.header("回测结果")
    st.caption(
        f"信号总数 {len(signals):,} 笔，覆盖 "
        f"{signals['Signal_Week'].min()} — {signals['Signal_Week'].max()}"
    )

    # 三层结果
    sig_summary, sig_curve = backtest_signal_layer(
        signals, week_index, int(hold_weeks), float(cost_pct)
    )
    slot_summary, slot_ledger, slot_total = backtest_three_slot(
        signals, week_index, int(hold_weeks), float(cost_pct), int(slot_count),
        order_mode="K",
    )

    st.subheader("表1 · 信号层 vs 三仓层")
    combined = pd.concat(
        [
            sig_summary[["层级", "交易笔数", "单笔平均收益%", "单笔胜率%", "总收益率%"]]
            .rename(columns={"交易笔数": "实际买入"}),
            slot_summary[["层级", "实际买入", "单笔平均收益%", "单笔胜率%", "总收益率%"]],
        ],
        ignore_index=True,
    )
    st.dataframe(combined.round(2), width="stretch", hide_index=True)
    st.caption(
        "两层使用完全相同的资金投放节奏，区别只在于：信号层每份资金买入当周全部信号，"
        "三仓层每份资金只买1只。**两者之差就是资金约束+集中持股的代价。**"
    )

    st.dataframe(
        sig_summary[
            ["调仓周数", "每周平均信号数", "周期等权平均收益%", "周期胜率%"]
        ].round(2),
        width="stretch", hide_index=True,
    )

    # 蒙特卡洛
    st.subheader("表2 · 蒙特卡洛：运气占多大比重？")
    mc_progress = st.progress(0.0, text="随机重排选股顺序，重复回测……")
    outcomes = monte_carlo_slots(
        signals, week_index, int(hold_weeks), float(cost_pct), int(slot_count),
        int(mc_runs), progress_callback=lambda p: mc_progress.progress(p),
    )
    mc_progress.empty()

    percentile_of_real = float((outcomes < slot_total).mean() * 100.0)
    mc_table = pd.DataFrame(
        [
            {
                "指标": "随机选股 最差5%",
                "总收益率%": float(np.percentile(outcomes, 5)),
            },
            {"指标": "随机选股 中位数", "总收益率%": float(np.median(outcomes))},
            {"指标": "随机选股 平均", "总收益率%": float(outcomes.mean())},
            {
                "指标": "随机选股 最好5%",
                "总收益率%": float(np.percentile(outcomes, 95)),
            },
            {"指标": "▶ 按K值优选（真实规则）", "总收益率%": slot_total},
            {"指标": "▶ 信号层（无资金约束）",
             "总收益率%": float(sig_summary["总收益率%"].iloc[0])},
        ]
    )
    st.dataframe(mc_table.round(2), width="stretch", hide_index=True)

    spread = float(np.percentile(outcomes, 95) - np.percentile(outcomes, 5))
    st.markdown(
        f"""
**怎么读这张表：**

- 随机选股{int(mc_runs)}次，总收益率的90%区间宽度是 **{spread:.1f}个百分点**
  —— 这就是三仓路径依赖带来的**纯运气区间**。区间越宽，单次回测结果越不可信。
- 按K值优选的真实结果落在随机分布的 **{percentile_of_real:.0f}%分位**。
  - 接近50% → 排序规则基本没贡献，结果主要靠运气
  - 高于90% → 排序规则确实有效
- 信号层收益与三仓层之差 = 资金约束的代价。
"""
    )

    chart_frame = pd.DataFrame({"随机选股总收益率%": outcomes})
    st.bar_chart(
        chart_frame["随机选股总收益率%"].value_counts(bins=30, sort=False).rename("次数")
    )

    # 分年度
    st.subheader("表3 · 分年度（信号层）")
    year_frame = signals.copy()
    year_frame["年份"] = year_frame["Signal_Week"].astype(str).str[:4]
    year_frame["净收益%"] = pd.to_numeric(
        year_frame["Return_pct"], errors="coerce"
    ) - float(cost_pct)
    year_table = (
        year_frame.groupby("年份")["净收益%"]
        .agg(["count", "mean", "median", lambda s: (s > 0).mean() * 100.0])
        .reset_index()
    )
    year_table.columns = ["年份", "信号数", "平均收益%", "中位收益%", "胜率%"]
    st.dataframe(year_table.round(2), width="stretch", hide_index=True)

    # 导出
    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_layer_comparison.csv", combined.to_csv(index=False, encoding="utf-8-sig")
        )
        archive.writestr(
            "02_monte_carlo.csv", mc_table.to_csv(index=False, encoding="utf-8-sig")
        )
        archive.writestr(
            "03_monte_carlo_raw.csv",
            pd.DataFrame({"total_return_pct": outcomes}).to_csv(
                index=False, encoding="utf-8-sig"
            ),
        )
        archive.writestr(
            "04_yearly.csv", year_table.to_csv(index=False, encoding="utf-8-sig")
        )
        archive.writestr(
            "05_all_signals.csv", signals.to_csv(index=False, encoding="utf-8-sig")
        )
        archive.writestr(
            "06_three_slot_ledger.csv",
            slot_ledger.to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "07_signal_layer_curve.csv",
            sig_curve.to_csv(index=False, encoding="utf-8-sig"),
        )
    st.download_button(
        "下载完整回测结果",
        data=output.getvalue(),
        file_name="skdj_strategy_backtest.zip",
        mime="application/zip",
    )

    with st.expander("查看三仓逐笔台账"):
        st.dataframe(slot_ledger, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
