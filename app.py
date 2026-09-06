# -*- coding: utf-8 -*-
"""趋势启动信号验证器 V2（单文件独立版，直接覆盖 app.py 运行）。

上一轮（V1）的关键发现，两条都是干净单调、幅度接近两倍：
  - 越接近两年高点，突破后大涨概率越高（18.96% -> 35.08%）
  - 横盘区间越宽，大涨概率越高（18.93% -> 33.06%）

这两条都与从K线图得到的直觉相反（图上看是"低位极窄横盘后暴涨"），
属于典型的幸存者偏差：低位窄幅横盘的股票，突破后恰恰是大涨概率最低的一组。

而V1的最优组合⑦（突破+压缩+放量）完全没用到这两条，提升倍数仅1.46；
⑧因为加了反方向的"低位启动"条件，提升倍数掉到0.83、收益转负。

本轮用已验证方向的特征重新组合，目标是把涨超50%的提升倍数从1.46推到2以上。
同时对这两个关键特征做分年度单调性检验——V1只验证了全期。

所有特征只用当周及之前数据；前瞻指标从下一周开盘算起，无未来函数。
退出统一用移动止损（趋势型信号配趋势型退出）。

行情缓存与之前共用，已下载数据不会重复下载。
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

APP_TITLE = "趋势启动信号验证 V2"
MARKET_CACHE_ROOT = "r1_trend_entry_market_cache_v2"
CACHE_SCHEMA_VERSION = 3
DOWNLOAD_WORKERS = 4
DATA_READY_HOUR_SHANGHAI = 18

# -----------------------------------------------------------------------------
# 数据层（与之前完全一致）
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
    # 量能类列按周求和（换手率逐日相加即为周换手率），
    # 缺了这几列会导致后续量能因子拿到标量而不是序列。
    for column in ("vol", "amount", "turnover_rate"):
        if column in frame.columns:
            aggregations[column] = "sum"
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

# -----------------------------------------------------------------------------
# 指标计算
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 指标
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 个股周线面板
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 指标与信号
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 指标与信号
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 趋势启动特征与前瞻结果
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 特征与前瞻
# -----------------------------------------------------------------------------
def build_trend_panel(
    weekly: pd.DataFrame,
    ts_code: str,
    breakout_weeks: int,
    base_weeks: int,
    forward_weeks: int,
    trail_pct: float,
) -> pd.DataFrame:
    """逐周一行：趋势启动特征 + 未来大涨情况。无未来函数。"""
    close = pd.to_numeric(weekly["close"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    low = pd.to_numeric(weekly["low"], errors="coerce")
    open_p = pd.to_numeric(weekly["open"], errors="coerce")
    volume = (
        pd.to_numeric(weekly["vol"], errors="coerce")
        if "vol" in weekly.columns
        else pd.Series(np.nan, index=weekly.index, dtype="float64")
    )
    dates = weekly["trade_date_str"].astype(str)
    n = len(weekly)
    if n < base_weeks + 20:
        return pd.DataFrame()

    return_1w = (close / close.shift(1) - 1.0) * 100.0

    prior_high_close = close.shift(1).rolling(breakout_weeks).max()
    breakout = close > prior_high_close

    base_high = high.shift(1).rolling(base_weeks).max()
    base_low = low.shift(1).rolling(base_weeks).min()
    base_mid = close.shift(1).rolling(base_weeks).mean()
    base_range_pct = (base_high - base_low) / base_mid.replace(0, np.nan) * 100.0

    vol_recent = return_1w.shift(1).rolling(8).std()
    vol_earlier = return_1w.shift(9).rolling(18).std()
    vol_contraction = vol_recent / vol_earlier.replace(0, np.nan)

    volume_surge = volume / volume.shift(1).rolling(8).mean().replace(0, np.nan)

    long_high = high.shift(1).rolling(104).max()
    position_vs_2y_high = close / long_high.replace(0, np.nan)

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma_bull = (ma5 > ma10) & (ma10 > ma20)

    # V2新增：中期动量（接近高点的另一种表达，且不依赖104周历史）
    momentum_26w = (close / close.shift(26) - 1.0) * 100.0

    entry = open_p.shift(-1)
    max_gain = pd.Series(np.nan, index=weekly.index, dtype="float64")
    final_return = pd.Series(np.nan, index=weekly.index, dtype="float64")
    trail_return = pd.Series(np.nan, index=weekly.index, dtype="float64")

    high_values = high.to_numpy()
    close_values = close.to_numpy()
    entry_values = entry.to_numpy()

    for i in range(n):
        entry_price = entry_values[i]
        if not math.isfinite(entry_price) or entry_price <= 0:
            continue
        stop = min(i + forward_weeks, n - 1)
        if stop <= i:
            continue
        window_high = high_values[i + 1 : stop + 1]
        finite_high = window_high[np.isfinite(window_high)]
        if finite_high.size:
            max_gain.iloc[i] = (finite_high.max() / entry_price - 1.0) * 100.0
        final_close = close_values[stop]
        if math.isfinite(final_close):
            final_return.iloc[i] = (final_close / entry_price - 1.0) * 100.0
        peak = entry_price
        exit_price = None
        for j in range(i + 1, stop + 1):
            current = close_values[j]
            if not math.isfinite(current):
                continue
            peak = max(peak, current)
            if current <= peak * (1.0 - trail_pct / 100.0):
                exit_price = current
                break
        if exit_price is None:
            exit_price = close_values[stop]
        if math.isfinite(exit_price):
            trail_return.iloc[i] = (exit_price / entry_price - 1.0) * 100.0

    return pd.DataFrame(
        {
            "ts_code": ts_code,
            "Week": dates,
            "Breakout": breakout.fillna(False),
            "Base_Range_pct": base_range_pct,
            "Vol_Contraction": vol_contraction,
            "Volume_Surge": volume_surge,
            "Position_vs_2Y_High": position_vs_2y_high,
            "Momentum_26W_pct": momentum_26w,
            "MA_Bull": ma_bull.fillna(False),
            "Entry_Open": entry,
            "Max_Gain_pct": max_gain,
            "Final_Return_pct": final_return,
            "Trail_Return_pct": trail_return,
        }
    ).dropna(subset=["Max_Gain_pct"])


# -----------------------------------------------------------------------------
# 统计
# -----------------------------------------------------------------------------
def big_move_stats(values: pd.Series, thresholds=(30, 50, 100)):
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    stats = {"样本数": int(len(numeric))}
    for threshold in thresholds:
        stats[f"涨超{threshold}%概率"] = (
            float((numeric > threshold).mean() * 100.0) if len(numeric) else np.nan
        )
    stats["平均最大涨幅%"] = float(numeric.mean()) if len(numeric) else np.nan
    return stats


def v2_variants_test(panel: pd.DataFrame, cost_pct: float, quantile_cut: float):
    """按上一轮验证出的正确方向重新组合特征。

    上一轮关键发现（两条都是干净单调、幅度接近两倍）：
      - 越接近两年高点，大涨概率越高（18.96% -> 35.08%）
      - 横盘区间越宽，大涨概率越高（18.93% -> 33.06%）
    而上一轮最优组合⑦完全没用到这两条，⑧还用了反方向的"低位启动"导致失效。
    本轮用正确方向替换，看提升倍数能否从1.46推高。
    """
    baseline = big_move_stats(panel["Max_Gain_pct"])
    breakout = panel["Breakout"].astype(bool)
    base_range = pd.to_numeric(panel["Base_Range_pct"], errors="coerce")
    vol_contract = pd.to_numeric(panel["Vol_Contraction"], errors="coerce")
    volume_surge = pd.to_numeric(panel["Volume_Surge"], errors="coerce")
    position = pd.to_numeric(panel["Position_vs_2Y_High"], errors="coerce")
    momentum = pd.to_numeric(panel["Momentum_26W_pct"], errors="coerce")
    ma_bull = panel["MA_Bull"].astype(bool)

    # 方向已由上一轮数据确定：位置高、区间宽、压缩强、放量
    near_high = position >= position.quantile(1.0 - quantile_cut)
    wide_range = base_range >= base_range.quantile(1.0 - quantile_cut)
    strong_contract = vol_contract <= 0.8
    with_volume = volume_surge >= 1.5
    strong_momentum = momentum >= momentum.quantile(1.0 - quantile_cut)

    variants = [
        ("全池基准（不筛选）", pd.Series(True, index=panel.index)),
        ("上轮最优⑦：突破+压缩+放量", breakout & strong_contract & with_volume),
        ("A 突破 + 接近两年高点", breakout & near_high),
        ("B 突破 + 区间宽", breakout & wide_range),
        ("C 突破 + 接近高点 + 区间宽", breakout & near_high & wide_range),
        ("D 突破 + 接近高点 + 放量", breakout & near_high & with_volume),
        ("E 突破 + 接近高点 + 区间宽 + 放量", breakout & near_high & wide_range & with_volume),
        ("F 突破 + 接近高点 + 区间宽 + 压缩", breakout & near_high & wide_range & strong_contract),
        (
            "G 全条件：突破+接近高点+区间宽+压缩+放量",
            breakout & near_high & wide_range & strong_contract & with_volume,
        ),
        ("H 突破 + 26周动量强 + 放量", breakout & strong_momentum & with_volume),
        ("I 突破 + 接近高点 + 均线多头 + 放量", breakout & near_high & ma_bull & with_volume),
    ]

    rows = []
    for label, mask in variants:
        subset = panel.loc[mask.fillna(False)]
        if len(subset) < 30:
            continue
        stats = big_move_stats(subset["Max_Gain_pct"])
        trail = (
            pd.to_numeric(subset["Trail_Return_pct"], errors="coerce").dropna() - cost_pct
        )
        final = (
            pd.to_numeric(subset["Final_Return_pct"], errors="coerce").dropna() - cost_pct
        )
        row = {
            "信号定义": label,
            "样本数": stats["样本数"],
            "占全池%": float(len(subset) / len(panel) * 100.0),
        }
        for threshold in (50, 100):
            key = f"涨超{threshold}%概率"
            row[key] = stats[key]
            base_value = baseline[key]
            row[f"涨超{threshold}%提升倍数"] = (
                stats[key] / base_value if base_value and base_value > 0 else np.nan
            )
        row["移动止损收益%"] = float(trail.mean()) if len(trail) else np.nan
        row["移动止损胜率%"] = float((trail > 0).mean() * 100.0) if len(trail) else np.nan
        row["移动止损中位%"] = float(trail.median()) if len(trail) else np.nan
        row["持有到期收益%"] = float(final.mean()) if len(final) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def feature_yearly_monotonicity(panel: pd.DataFrame, feature: str, label: str, ascending: bool):
    """分年度检验特征的单调性——上一轮只看了全期，这次逐年验证。

    一个真规律应该每年都保持同方向；如果只有某几年单调，可信度就要打折。
    """
    breakout = panel[panel["Breakout"].astype(bool)].copy()
    breakout["年份"] = breakout["Week"].astype(str).str[:4]
    values = pd.to_numeric(breakout[feature], errors="coerce")
    work = breakout.assign(_v=values).dropna(subset=["_v"])
    rows = []
    for year, group in work.groupby("年份"):
        if len(group) < 100:
            continue
        try:
            group = group.assign(
                _q=pd.qcut(group["_v"].rank(method="first", ascending=ascending), 5, labels=False)
            )
        except ValueError:
            continue
        record = {"特征": label, "年份": year, "样本数": len(group)}
        means = []
        for q in range(5):
            sub = group[group["_q"] == q]["Max_Gain_pct"]
            value = float((sub > 50).mean() * 100.0) if len(sub) else np.nan
            record[f"第{q + 1}组涨超50%"] = value
            means.append(value)
        valid = [m for m in means if math.isfinite(m)]
        record["首尾差"] = (valid[-1] - valid[0]) if len(valid) >= 2 else np.nan
        diffs = [b - a for a, b in zip(valid, valid[1:])]
        record["单调性"] = (
            f"{sum(1 for d in diffs if d > 0)}升/{sum(1 for d in diffs if d < 0)}降"
        )
        rows.append(record)
    return pd.DataFrame(rows)


def best_variant_yearly(panel: pd.DataFrame, cost_pct: float, quantile_cut: float):
    """几个候选组合的分年度对比。"""
    work = panel.copy()
    work["年份"] = work["Week"].astype(str).str[:4]
    breakout = work["Breakout"].astype(bool)
    base_range = pd.to_numeric(work["Base_Range_pct"], errors="coerce")
    vol_contract = pd.to_numeric(work["Vol_Contraction"], errors="coerce")
    volume_surge = pd.to_numeric(work["Volume_Surge"], errors="coerce")
    position = pd.to_numeric(work["Position_vs_2Y_High"], errors="coerce")

    near_high = position >= position.quantile(1.0 - quantile_cut)
    wide_range = base_range >= base_range.quantile(1.0 - quantile_cut)
    strong_contract = vol_contract <= 0.8
    with_volume = volume_surge >= 1.5

    candidates = {
        "上轮⑦(压缩+放量)": breakout & strong_contract & with_volume,
        "E(高点+宽+放量)": breakout & near_high & wide_range & with_volume,
        "G(全条件)": breakout & near_high & wide_range & strong_contract & with_volume,
    }
    rows = []
    for year, group in work.groupby("年份"):
        base_stats = big_move_stats(group["Max_Gain_pct"])
        record = {
            "年份": year,
            "基准涨超50%": base_stats["涨超50%概率"],
        }
        for name, mask in candidates.items():
            sub = group[mask.reindex(group.index).fillna(False)]
            if len(sub) < 5:
                record[f"{name}信号数"] = len(sub)
                record[f"{name}提升"] = np.nan
                record[f"{name}止损收益%"] = np.nan
                continue
            stats = big_move_stats(sub["Max_Gain_pct"])
            trail = (
                pd.to_numeric(sub["Trail_Return_pct"], errors="coerce").dropna() - cost_pct
            )
            record[f"{name}信号数"] = int(len(sub))
            record[f"{name}提升"] = (
                stats["涨超50%概率"] / base_stats["涨超50%概率"]
                if base_stats["涨超50%概率"]
                else np.nan
            )
            record[f"{name}止损收益%"] = float(trail.mean()) if len(trail) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🚀 {APP_TITLE}")
    st.caption("用上一轮验证出的正确方向重新组合——不是碰运气加条件。")
    st.info(
        "**上一轮的关键发现（两条都是干净单调、幅度接近两倍）：**\n\n"
        "- 越**接近两年高点**，突破后大涨概率越高（18.96% → 35.08%）\n"
        "- 横盘**区间越宽**，大涨概率越高（18.93% → 33.06%）\n\n"
        "这两条都和从K线图得出的直觉相反（图上看是低位极窄横盘后暴涨），"
        "属于典型的幸存者偏差。而上一轮最优组合⑦完全没用到这两条，"
        "⑧还用了反方向的低位启动条件，提升倍数掉到0.83。\n\n"
        "**本轮用正确方向替换，看提升倍数能否从1.46推到2以上。**"
        "同时对这两个特征做分年度单调性检验——上一轮只看了全期。"
    )

    with st.sidebar:
        st.header("配置")
        try:
            secret_token = st.secrets.get("TUSHARE_TOKEN", "")
        except Exception:
            secret_token = ""
        token_input = st.text_input("Tushare Token", value=secret_token, type="password")
        today = pd.Timestamp.now().date()
        start_input = st.date_input("开始日期", value=today - timedelta(days=365 * 4))
        end_input = st.date_input("结束日期", value=today)

        st.markdown("---")
        st.subheader("信号定义")
        breakout_weeks = st.number_input("突破几周新高", value=26, min_value=4, max_value=104, step=2)
        base_weeks = st.number_input("横盘基底考察周数", value=26, min_value=8, max_value=104, step=2)
        quantile_cut = st.slider(
            "「接近高点」「区间宽」取前百分之几",
            min_value=0.10, max_value=0.60, value=0.33, step=0.05,
            help="0.33表示取该特征排名前33%的样本。放宽会增加信号数但降低纯度。",
        )

        st.markdown("---")
        st.subheader("前瞻与退出")
        forward_weeks = st.number_input("前瞻观察周数", value=26, min_value=8, max_value=104, step=2)
        trail_pct = st.number_input(
            "移动止损：从最高收盘价回撤%", value=20.0, min_value=5.0, max_value=50.0, step=5.0
        )
        cost_pct = st.number_input("往返成本%", value=0.20, min_value=0.0, max_value=2.0, step=0.05)

        st.markdown("---")
        st.subheader("股票池硬条件")
        min_price = st.number_input("最低股价（元）", value=10.0, min_value=0.0, step=1.0)
        min_mv = st.number_input("最低流通市值（亿元）", value=100.0, min_value=0.0, step=10.0)
        max_mv = st.number_input("最高流通市值（亿元）", value=1000.0, min_value=100.0, step=100.0)

        st.markdown("---")
        clear_cache_clicked = st.button("清空行情缓存")
        run_clicked = st.button("开始验证", type="primary")

    if clear_cache_clicked:
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.success("行情缓存已清空。")

    if not run_clicked:
        if st.session_state.get("trend2_result"):
            return
        st.markdown(
            """
### 本轮测试的组合

| 代号 | 组合 |
|---|---|
| ⑦ | 上轮最优：突破+压缩+放量（对照） |
| A | 突破 + 接近两年高点 |
| B | 突破 + 区间宽 |
| C | 突破 + 接近高点 + 区间宽 |
| D | 突破 + 接近高点 + 放量 |
| E | 突破 + 接近高点 + 区间宽 + 放量 |
| F | 突破 + 接近高点 + 区间宽 + 压缩 |
| G | 全条件叠加 |
| H | 突破 + 26周动量强 + 放量 |
| I | 突破 + 接近高点 + 均线多头 + 放量 |

### 三张表

**表1 · 各组合 vs 基准**　核心是**涨超50%/100%的提升倍数**，
以及**移动止损收益**（真正能落袋的）。

**表2 · 分年度单调性**　上一轮只看全期单调，这次逐年验证。
一个真规律应该每年都同方向；只有某几年单调的话，可信度要打折。

**表3 · 候选组合分年度**　对比⑦、E、G三个方案逐年的表现，
看新组合是否稳定优于上一轮的⑦。

---
**判断标准**：提升倍数>2 且分年度大部分>1.5，才算真正找到东西。
上一轮⑦只有1.46，且5年里3年移动止损是亏的。
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
    fetch_start = (
        pd.Timestamp(start_input) - timedelta(days=int(base_weeks) * 7 + 900)
    ).strftime("%Y%m%d")
    fetch_end = (
        pd.Timestamp(end_input) + timedelta(days=int(forward_weeks) * 7 + 60)
    ).strftime("%Y%m%d")

    with st.spinner("构建科技股研究池……"):
        whitelist_set, name_map, industry_map = load_custom_tech_whitelist(token_clean)
    if not whitelist_set:
        st.error("未取得研究池。")
        return
    st.success(f"科技股研究池：{len(whitelist_set)}只")

    with st.spinner("加载行情（复用缓存）……"):
        stocks, basic_indexed, _, _, failed_dates, sync_stats = load_optimized_market_data(
            fetch_start, fetch_end, token_clean, tuple(sorted(whitelist_set))
        )
    if not stocks:
        st.error("未加载到行情。")
        return
    st.caption(
        f"行情：复用{sync_stats.get('cached_days', 0)}天，"
        f"本次下载{sync_stats.get('downloaded_days', 0)}天。"
    )

    progress = st.progress(0.0, text="计算趋势启动特征……")
    parts = []
    codes = sorted(stocks.keys())
    for idx, ts_code in enumerate(codes):
        weekly = build_weekly_bars(stocks[ts_code])
        if weekly.empty:
            continue
        rows = build_trend_panel(
            weekly, ts_code, int(breakout_weeks), int(base_weeks),
            int(forward_weeks), float(trail_pct),
        )
        if not rows.empty:
            parts.append(rows)
        if idx % 40 == 0:
            progress.progress(
                min((idx + 1) / len(codes), 1.0),
                text=f"计算趋势启动特征……{idx + 1}/{len(codes)}",
            )
    progress.empty()
    del stocks
    gc.collect()

    if not parts:
        st.error("数据不足。")
        return
    panel = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    panel = panel[(panel["Week"] >= start_date) & (panel["Week"] <= end_date)]
    panel = panel[pd.to_numeric(panel["Entry_Open"], errors="coerce") >= min_price]
    if not basic_indexed.empty:
        basic_reset = basic_indexed.reset_index().rename(columns={"trade_date_str": "Week"})
        keep = [c for c in ("Week", "ts_code", "circ_mv") if c in basic_reset.columns]
        if len(keep) == 3:
            panel = panel.merge(
                basic_reset[keep].drop_duplicates(["Week", "ts_code"]),
                on=["Week", "ts_code"], how="left",
            )
            mv = pd.to_numeric(panel["circ_mv"], errors="coerce") / 10000.0
            panel = panel[mv.between(min_mv, max_mv) | mv.isna()]
    panel = panel.reset_index(drop=True)
    if panel.empty:
        st.error("过滤后无数据。")
        return

    variants = v2_variants_test(panel, float(cost_pct), float(quantile_cut))
    monotonic = pd.concat(
        [
            feature_yearly_monotonicity(
                panel, "Position_vs_2Y_High", "相对两年高点（第5组最接近高点）", True
            ),
            feature_yearly_monotonicity(
                panel, "Base_Range_pct", "横盘区间宽度（第5组最宽）", True
            ),
        ],
        ignore_index=True,
    )
    yearly = best_variant_yearly(panel, float(cost_pct), float(quantile_cut))

    st.session_state["trend2_result"] = {
        "panel_size": len(panel),
        "breakout_count": int(panel["Breakout"].astype(bool).sum()),
        "variants": variants,
        "monotonic": monotonic,
        "yearly": yearly,
        "params": {
            "突破周数": int(breakout_weeks), "基底周数": int(base_weeks),
            "前瞻周数": int(forward_weeks), "移动止损": float(trail_pct),
            "分位": float(quantile_cut),
        },
    }


def render_results():
    result = st.session_state.get("trend2_result")
    if not result:
        return False
    params = result["params"]

    st.markdown("---")
    st.header("趋势启动 V2 验证结果")
    st.caption(
        f"全池观测 {result['panel_size']:,} 个「个股-周」，突破样本 "
        f"{result['breakout_count']:,} 个　|　突破{params['突破周数']}周新高　"
        f"前瞻{params['前瞻周数']}周　移动止损{params['移动止损']:.0f}%　"
        f"特征取前{params['分位']*100:.0f}%"
    )

    st.subheader("表1 · 各组合 vs 全池基准")
    st.dataframe(result["variants"].round(2), width="stretch", hide_index=True)
    st.caption(
        "**核心两列**：「涨超50%提升倍数」要大于2才算真正有能力；"
        "「移动止损收益%」是实际能落袋的。\n\n"
        "第二行是上一轮最优的⑦（提升1.46倍、收益4.65%），"
        "新组合必须明显超过它才算这一轮有进展。"
        "同时留意「占全池%」——筛得太狠会导致信号太少没法交易。"
    )

    if not result["monotonic"].empty:
        st.subheader("表2 · 两个关键特征的分年度单调性")
        st.dataframe(result["monotonic"].round(2), width="stretch", hide_index=True)
        st.caption(
            "上一轮只验证了全期单调，这次逐年检查。**看「单调性」列**："
            "如果多数年份是4升/0降或3升/1降，说明规律稳定可信；"
            "如果各年方向不一致，那全期的单调可能是少数年份主导的假象。"
        )

    if not result["yearly"].empty:
        st.subheader("表3 · 三个候选方案的分年度对比")
        st.dataframe(result["yearly"].round(2), width="stretch", hide_index=True)
        st.caption(
            "对比上轮⑦与新组合E、G。**看新组合是否每年都优于⑦**，"
            "以及止损收益为负的年份是否减少（⑦是5年里3年亏）。"
        )

    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_variants_v2.csv",
            result["variants"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "02_yearly_monotonicity.csv",
            result["monotonic"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "03_yearly_variants.csv",
            result["yearly"].to_csv(index=False, encoding="utf-8-sig"),
        )
    st.download_button(
        "下载验证结果",
        data=output.getvalue(),
        file_name="trend_start_v2.zip",
        mime="application/zip",
        key="download_trend2",
    )
    return True


if __name__ == "__main__":
    main()
    render_results()
