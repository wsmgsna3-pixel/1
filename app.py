# -*- coding: utf-8 -*-
"""
R3 中性市场首红 Top2 与 W3 主目标验证器

R2 已证明“突破/回调/首红混合候选 + 横截面追强评分”在一年样本中会把较差股票
排到前面。R3 因此恢复经过逐行复核的 R1 六因子首红评分，并把排序简化为
“趋势分优先、风险分次优、六因子总分只破同分”。每周由 Top3 收缩为 Top2；
仅在科技池 13 周中位涨幅处于 -5% 到 +5% 的中性阶段
形成可交易组合。历史信号日为每周最后一个交易日，下一交易日开盘买入，W3 为
主验收目标，同时固定记录 W1—W8；任何买入后信息都不进入评分。
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
import uuid
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts

warnings.filterwarnings("ignore")

APP_VERSION = "R3.0-NEUTRAL-FIRST-RED-TOP2-W3-AUDIT"
APP_TITLE = "R3 中性市场首红Top2与W3主目标验证器"
ENGINE_PATCH = "R3.0-R1-EXACT-SCORE-LEASED-STABLE-ENGINE"

CHECKPOINT_FILE = "r3_neutral_first_red_candidates.csv"
SCAN_LEDGER_FILE = "r3_neutral_first_red_scanned_dates.csv"
RUN_TASK_FILE = "r3_neutral_first_red_running_task.json"
MARKET_CACHE_ROOT = "r1_trend_entry_market_cache_v2"

TOP_N = 2
MIN_VALID_SELECTION_SIZE = 2
PRIMARY_HOLD_WEEKS = 3
HOLD_WEEKS = 8
MARKET_DAYS_PER_WEEK = 5
WEEKS_PER_BATCH = 3
CACHE_SCHEMA_VERSION = 3
DOWNLOAD_WORKERS = 4
MARKET_NEUTRAL_LOWER_PCT = -5.0
MARKET_NEUTRAL_UPPER_PCT = 5.0
TASK_LEASE_SECONDS = 45
PRIMARY_RETURN_COLUMN = f"Fixed_Return_W{PRIMARY_HOLD_WEEKS}_Net_pct"


# -----------------------------------------------------------------------------
# 通用安全读写
# -----------------------------------------------------------------------------
def clean_token_str(raw_token: str) -> str:
    if not raw_token:
        return ""
    return re.sub(r"[\s\u3000\ufeff\xa0\r\n]+", "", str(raw_token)).strip()


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
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir
    )
    os.close(fd)
    try:
        frame.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        with open(tmp_path, "rb") as file_obj:
            os.fsync(file_obj.fileno())
        if os.path.exists(path):
            try:
                shutil.copy2(path, path + ".bak")
            except OSError:
                pass
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_csv_safe(path: str):
    for candidate in (path, path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            return pd.read_csv(candidate, encoding="utf-8-sig", low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError):
            continue
    return pd.DataFrame()


def atomic_write_json(value: dict[str, Any], path: str):
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir
    )
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        if os.path.exists(path):
            try:
                shutil.copy2(path, path + ".bak")
            except OSError:
                pass
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_json_safe(path: str):
    for candidate in (path, path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as file_obj:
                value = json.load(file_obj)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return {}


def remove_with_backup(path: str):
    for candidate in (path, path + ".bak"):
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except OSError:
            pass


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
    daily_all = safe_tushare_call(
        pro.daily,
        trade_date=trade_date,
        fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount",
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
    today_str = datetime.now().strftime("%Y%m%d")
    open_calendar = calendar[
        pd.to_numeric(calendar["is_open"], errors="coerce").eq(1)
        & (calendar["cal_date"].astype(str) <= today_str)
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


# -----------------------------------------------------------------------------
# 买入前特征：周线结构候选 + 趋势资格
# -----------------------------------------------------------------------------
def _safe_float(value: Any, default: float = np.nan):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _weekly_bars(stock: pd.DataFrame, end_date: str):
    daily = stock[stock.index <= end_date].tail(420).copy()
    if len(daily) < 180:
        return pd.DataFrame()
    daily = daily.reset_index()
    daily["dt"] = pd.to_datetime(daily["trade_date_str"], errors="coerce")
    daily = daily.dropna(subset=["dt"])
    daily["year_week"] = daily["dt"].dt.strftime("%G_%V")
    aggregations: dict[str, str] = {
        "trade_date_str": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vol": "sum",
    }
    if "turnover_rate" in daily.columns:
        aggregations["turnover_rate"] = "sum"
    weekly = (
        daily.groupby("year_week", as_index=False)
        .agg(aggregations)
        .sort_values("trade_date_str")
        .reset_index(drop=True)
    )
    if len(weekly) < 45:
        return pd.DataFrame()

    close = pd.to_numeric(weekly["close"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    low = pd.to_numeric(weekly["low"], errors="coerce")
    volume = pd.to_numeric(weekly["vol"], errors="coerce")

    weekly["ma10"] = close.rolling(10).mean()
    weekly["ma20"] = close.rolling(20).mean()
    weekly["ma40"] = close.rolling(40).mean()
    weekly["ma10_slope_2w_pct"] = (weekly["ma10"] / weekly["ma10"].shift(2) - 1.0) * 100.0
    weekly["ma20_slope_4w_pct"] = (weekly["ma20"] / weekly["ma20"].shift(4) - 1.0) * 100.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    weekly["dif"] = ema12 - ema26
    weekly["dea"] = weekly["dif"].ewm(span=9, adjust=False).mean()
    weekly["macd_hist"] = 2.0 * (weekly["dif"] - weekly["dea"])
    weekly["macd_impulse_pct"] = (
        (weekly["macd_hist"] - weekly["macd_hist"].shift(1))
        / close.replace(0, np.nan)
        * 100.0
    )

    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100.0
    weekly["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    weekly["kdj_d"] = weekly["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    weekly["atr3_pct"] = true_range.rolling(3).mean() / close * 100.0
    weekly["atr13_pct"] = true_range.rolling(13).mean() / close * 100.0
    weekly["atr_contraction"] = weekly["atr3_pct"] / weekly["atr13_pct"].replace(0, np.nan)

    weekly["prior_vol3"] = volume.shift(1).rolling(3).mean()
    weekly["prior_vol8"] = volume.shift(1).rolling(8).mean()
    weekly["volume_contraction"] = weekly["prior_vol3"] / weekly["prior_vol8"].replace(0, np.nan)
    weekly["startup_volume_ratio"] = volume / volume.shift(1).rolling(5).mean().replace(0, np.nan)

    if "turnover_rate" in weekly.columns:
        turnover = pd.to_numeric(weekly["turnover_rate"], errors="coerce")
        weekly["prior_turn3"] = turnover.shift(1).rolling(3).mean()
        weekly["prior_turn8"] = turnover.shift(1).rolling(8).mean()
        weekly["turnover_contraction"] = weekly["prior_turn3"] / weekly["prior_turn8"].replace(0, np.nan)
    else:
        weekly["turnover_contraction"] = np.nan

    weekly["return_1w_pct"] = (close / close.shift(1) - 1.0) * 100.0
    weekly["return_2w_pct"] = (close / close.shift(2) - 1.0) * 100.0
    weekly["return_4w_pct"] = (close / close.shift(4) - 1.0) * 100.0
    weekly["return_8w_pct"] = (close / close.shift(8) - 1.0) * 100.0
    weekly["return_13w_pct"] = (close / close.shift(13) - 1.0) * 100.0
    weekly["pre_signal_4w_return_pct"] = (close.shift(1) / close.shift(5) - 1.0) * 100.0
    weekly["prior_high_13w"] = high.shift(1).rolling(13).max()
    weekly["prior_high_26w"] = high.shift(1).rolling(26).max()
    weekly["breakout_13w_pct"] = (close / weekly["prior_high_13w"] - 1.0) * 100.0
    weekly["high_26w"] = high.rolling(26).max()
    weekly["drawdown_26w_pct"] = (close / weekly["high_26w"] - 1.0) * 100.0

    price_range = (high - low).replace(0, np.nan)
    weekly["close_location"] = (close - low) / price_range
    weekly["upper_shadow_ratio"] = (high - np.maximum(close, weekly["open"])) / price_range
    weekly["weekly_range_pct"] = price_range / close.replace(0, np.nan) * 100.0
    weekly["distance_ma20_pct"] = (close / weekly["ma20"] - 1.0) * 100.0
    return weekly


def compute_signal_snapshot(
    ts_code: str,
    end_date: str,
    stock_qfq_dict: dict[str, pd.DataFrame],
):
    if ts_code not in stock_qfq_dict:
        return {}
    weekly = _weekly_bars(stock_qfq_dict[ts_code], end_date)
    if weekly.empty or len(weekly) < 45:
        return {}
    current = weekly.iloc[-1]
    previous = weekly.iloc[-2]
    previous2 = weekly.iloc[-3]

    current_hist = _safe_float(current.get("macd_hist"))
    previous_hist = _safe_float(previous.get("macd_hist"))
    is_first_red = (
        math.isfinite(current_hist)
        and math.isfinite(previous_hist)
        and current_hist > 0.0
        and previous_hist <= 0.0
    )
    current_close = _safe_float(current.get("close"))
    previous_close_value = _safe_float(previous.get("close"))
    ma10 = _safe_float(current.get("ma10"))
    ma20 = _safe_float(current.get("ma20"))
    ma40 = _safe_float(current.get("ma40"))
    ma10_slope = _safe_float(current.get("ma10_slope_2w_pct"))
    ma20_slope = _safe_float(current.get("ma20_slope_4w_pct"))
    distance_ma20 = _safe_float(current.get("distance_ma20_pct"))
    weekly_range = _safe_float(current.get("weekly_range_pct"))
    # R1 的硬资格只有两个条件：收盘不低于 MA20，且 MA20 四周斜率为正。
    # MA20/MA40 排列、离均线距离和周振幅仍进入六因子评分，但不在这里二次加门。
    base_trend_eligible = (
        math.isfinite(current_close)
        and math.isfinite(ma20)
        and math.isfinite(ma20_slope)
        and current_close >= ma20
        and ma20_slope > 0.0
    )

    return_1w = _safe_float(current.get("return_1w_pct"))
    previous_return_1w = _safe_float(previous.get("return_1w_pct"))
    previous2_return_1w = _safe_float(previous2.get("return_1w_pct"))
    prior_high_13w = _safe_float(current.get("prior_high_13w"))
    previous_prior_high_13w = _safe_float(previous.get("prior_high_13w"))
    close_location_now = _safe_float(current.get("close_location"))
    fresh_breakout = (
        all(
            math.isfinite(item)
            for item in (
                current_close,
                previous_close_value,
                prior_high_13w,
                previous_prior_high_13w,
                return_1w,
            )
        )
        and current_close >= prior_high_13w * 0.995
        and previous_close_value < previous_prior_high_13w * 0.995
        and return_1w > 0.0
    )
    pullback_restart = (
        base_trend_eligible
        and all(
            math.isfinite(item)
            for item in (current_close, ma10, return_1w, current_hist, previous_hist, close_location_now)
        )
        and current_close >= ma10
        and 0.0 < return_1w <= 12.0
        and current_hist > previous_hist
        and close_location_now >= 0.55
        and (
            (math.isfinite(previous_return_1w) and previous_return_1w <= 0.0)
            or (math.isfinite(previous2_return_1w) and previous2_return_1w <= 0.0)
        )
    )
    setup_type = "趋势内MACD首红" if is_first_red else ""
    setup_candidate = bool(is_first_red)
    position_risk_ok = (
        math.isfinite(distance_ma20)
        and 0.0 <= distance_ma20 <= 25.0
        and math.isfinite(weekly_range)
        and weekly_range <= 25.0
    )
    trend_eligible = bool(base_trend_eligible and setup_candidate)

    recent_26 = weekly.tail(26).reset_index(drop=True)
    weeks_since_high = np.nan
    if not recent_26.empty and pd.to_numeric(recent_26["high"], errors="coerce").notna().any():
        high_position = int(pd.to_numeric(recent_26["high"], errors="coerce").values.argmax())
        weeks_since_high = len(recent_26) - 1 - high_position

    k_now = _safe_float(current.get("kdj_k"))
    d_now = _safe_float(current.get("kdj_d"))
    k_prev = _safe_float(previous.get("kdj_k"))
    d_prev = _safe_float(previous.get("kdj_d"))
    kdj_cross = (
        all(math.isfinite(item) for item in (k_now, d_now, k_prev, d_prev))
        and k_now > d_now
        and k_prev <= d_prev
    )

    return {
        "Is_First_Red": bool(is_first_red),
        "Fresh_13W_Breakout": bool(fresh_breakout),
        "Pullback_Restart": bool(pullback_restart),
        "R3_Setup_Candidate": bool(setup_candidate),
        "R3_Setup_Type": setup_type,
        "Base_Trend_Eligible": bool(base_trend_eligible),
        "Position_Risk_OK": bool(position_risk_ok),
        "Trend_Eligible": bool(trend_eligible),
        "Signal_Close": current_close,
        "Weekly_Date": str(current.get("trade_date_str")),
        "MACD_DIF": _safe_float(current.get("dif")),
        "MACD_DEA": _safe_float(current.get("dea")),
        "MACD_Hist": current_hist,
        "Previous_MACD_Hist": previous_hist,
        "Previous2_MACD_Hist": _safe_float(previous2.get("macd_hist")),
        "MACD_Impulse_pct": _safe_float(current.get("macd_impulse_pct")),
        "MA10": ma10,
        "MA20": ma20,
        "MA40": ma40,
        "MA10_Slope_2W_pct": ma10_slope,
        "MA20_Slope_4W_pct": ma20_slope,
        "Distance_MA20_pct": distance_ma20,
        "Drawdown_26W_pct": _safe_float(current.get("drawdown_26w_pct")),
        "Weeks_Since_26W_High": weeks_since_high,
        "PreSignal_4W_Return_pct": _safe_float(current.get("pre_signal_4w_return_pct")),
        "Return_1W_pct": return_1w,
        "Return_2W_pct": _safe_float(current.get("return_2w_pct")),
        "Return_4W_pct": _safe_float(current.get("return_4w_pct")),
        "Return_8W_pct": _safe_float(current.get("return_8w_pct")),
        "Return_13W_pct": _safe_float(current.get("return_13w_pct")),
        "Prior_High_13W": prior_high_13w,
        "Breakout_13W_pct": _safe_float(current.get("breakout_13w_pct")),
        "ATR_Contraction": _safe_float(current.get("atr_contraction")),
        "Volume_Contraction": _safe_float(current.get("volume_contraction")),
        "Turnover_Contraction": _safe_float(current.get("turnover_contraction")),
        "Startup_Volume_Ratio": _safe_float(current.get("startup_volume_ratio")),
        "Weekly_Close_Location": _safe_float(current.get("close_location")),
        "Weekly_Upper_Shadow_Ratio": _safe_float(current.get("upper_shadow_ratio")),
        "Weekly_Range_pct": _safe_float(current.get("weekly_range_pct")),
        "KDJ_K": k_now,
        "KDJ_D": d_now,
        "KDJ_Low_Cross": bool(kdj_cross and k_now <= 45.0),
    }


def _numeric_series(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _percentile_rank(values: pd.Series, higher_is_better: bool = True):
    numeric = pd.to_numeric(values, errors="coerce")
    ranked_source = numeric if higher_is_better else -numeric
    return ranked_source.rank(method="average", pct=True).fillna(0.5)


def _score_r1_six_factors(frame: pd.DataFrame):
    """R1 六因子原公式；已用 R1 导出的 2,103 条候选逐行精确复现。"""
    scored = frame.copy()
    close = _numeric_series(scored, "Signal_Close")
    ma20 = _numeric_series(scored, "MA20")
    ma40 = _numeric_series(scored, "MA40")
    slope20 = _numeric_series(scored, "MA20_Slope_4W_pct")
    dif = _numeric_series(scored, "MACD_DIF")
    drawdown = _numeric_series(scored, "Drawdown_26W_pct")
    weeks_high = _numeric_series(scored, "Weeks_Since_26W_High")
    presignal = _numeric_series(scored, "PreSignal_4W_Return_pct")
    atr = _numeric_series(scored, "ATR_Contraction")
    volume = _numeric_series(scored, "Volume_Contraction")
    turnover = _numeric_series(scored, "Turnover_Contraction")
    impulse_pct = _numeric_series(scored, "MACD_Impulse_Pct")
    startup = _numeric_series(scored, "Startup_Volume_Ratio")
    close_location = _numeric_series(scored, "Weekly_Close_Location")
    kdj_k = _numeric_series(scored, "KDJ_K")
    kdj_d = _numeric_series(scored, "KDJ_D")
    distance = _numeric_series(scored, "Distance_MA20_pct")
    week_range = _numeric_series(scored, "Weekly_Range_pct")
    upper_shadow = _numeric_series(scored, "Weekly_Upper_Shadow_Ratio")

    scored["Score_Trend_20"] = (
        1.0
        + (close >= ma20).astype(float) * 5.0
        + (close >= ma40).astype(float) * 3.0
        + (ma20 >= ma40).astype(float) * 4.0
        + np.select([slope20 > 1.0, slope20 > 0.0], [5.0, 3.0], default=0.0)
        + (dif > 0.0).astype(float) * 2.0
    )
    scored["Score_Pullback_15"] = (
        np.select(
            [
                drawdown <= -40.0,
                drawdown <= -30.0,
                drawdown <= -8.0,
                drawdown <= -3.0,
            ],
            [0.0, 3.0, 8.0, 4.0],
            default=1.0,
        )
        + np.select(
            [weeks_high <= 0.0, weeks_high <= 2.0, weeks_high <= 12.0],
            [0.0, 1.0, 3.0],
            default=0.0,
        )
        + np.select(
            [
                presignal <= -25.0,
                presignal <= -20.0,
                presignal <= -5.0,
                presignal <= 0.0,
            ],
            [0.0, 2.0, 4.0, 2.0],
            default=0.0,
        )
    )
    scored["Score_Contraction_15"] = (
        np.select([atr <= 0.8, atr <= 1.0, atr <= 1.2], [6.0, 4.0, 2.0], default=0.0)
        + np.select(
            [volume <= 0.8, volume <= 1.0, volume <= 1.2],
            [5.0, 3.0, 1.0],
            default=0.0,
        )
        + np.select(
            [turnover <= 0.8, turnover <= 1.0, turnover <= 1.2],
            [4.0, 3.0, 1.0],
            default=0.0,
        )
    )
    startup_score = np.select(
        [
            (startup > 0.8) & (startup <= 2.5),
            (startup > 2.5) & (startup <= 4.0),
            startup > 4.0,
        ],
        [4.0, 2.0, -1.0],
        default=0.0,
    )
    location_score = np.select(
        [close_location > 0.7, close_location > 0.5], [4.0, 2.0], default=0.0
    )
    kdj_score = np.select(
        [
            _bool_series(scored, "KDJ_Low_Cross"),
            (kdj_k <= 60.0) & (kdj_k > kdj_d),
        ],
        [4.0, 3.0],
        default=1.0,
    )
    scored["Score_Restart_15"] = (
        impulse_pct * 3.0 + startup_score + location_score + kdj_score
    )
    scored["Score_RS_25"] = (
        _numeric_series(scored, "RS_4W_Pct") * 5.0
        + _numeric_series(scored, "RS_8W_Pct") * 8.0
        + _numeric_series(scored, "RS_13W_Pct") * 8.0
        + _numeric_series(scored, "Industry_Excess_Pct") * 4.0
    )
    distance_score = np.select(
        [(distance > 0.0) & (distance <= 10.0), (distance > 10.0) & (distance <= 20.0)],
        [4.0, 2.0],
        default=0.0,
    )
    range_score = np.select(
        [week_range <= 8.0, week_range <= 12.0, week_range <= 18.0],
        [3.0, 2.0, 1.0],
        default=0.0,
    )
    shadow_score = np.select(
        [upper_shadow <= 0.20, upper_shadow <= 0.35], [3.0, 1.5], default=0.0
    )
    scored["Score_Risk_10"] = distance_score + range_score + shadow_score
    factor_columns = [
        "Score_Trend_20",
        "Score_Pullback_15",
        "Score_Contraction_15",
        "Score_Restart_15",
        "Score_RS_25",
        "Score_Risk_10",
    ]
    scored["Entry_Score_100"] = scored[factor_columns].sum(axis=1).clip(0.0, 100.0)
    return scored


def score_r3_candidates(pool_snapshots: pd.DataFrame):
    """用全池当周百分位复现 R1 评分，再用中性市场门决定是否形成 Top2。"""
    if pool_snapshots.empty:
        return pd.DataFrame(), 0, 0
    pool = pool_snapshots.copy()
    return_13w = _numeric_series(pool, "Return_13W_pct")
    industry_median = pool.groupby("Industry", dropna=False)["Return_13W_pct"].transform(
        "median"
    )
    pool["Industry_13W_Excess_pct"] = return_13w - pd.to_numeric(
        industry_median, errors="coerce"
    )
    pool["RS_4W_Pct"] = _percentile_rank(_numeric_series(pool, "Return_4W_pct"))
    pool["RS_8W_Pct"] = _percentile_rank(_numeric_series(pool, "Return_8W_pct"))
    pool["RS_13W_Pct"] = _percentile_rank(return_13w)
    pool["Industry_Excess_Pct"] = _percentile_rank(
        _numeric_series(pool, "Industry_13W_Excess_pct")
    )
    pool["MACD_Impulse_Pct"] = _percentile_rank(_numeric_series(pool, "MACD_Impulse_pct"))

    candidates = pool[_bool_series(pool, "R3_Setup_Candidate")].copy()
    raw_count = len(candidates)
    if candidates.empty:
        return candidates, 0, 0
    candidates = _score_r1_six_factors(candidates)
    candidates["Rank"] = np.nan
    candidates["Selected_Top2"] = False
    eligible = candidates[_bool_series(candidates, "Trend_Eligible")].copy()
    eligible_count = len(eligible)

    market_median = _safe_float(return_13w.median(), 0.0)
    if eligible_count < MIN_VALID_SELECTION_SIZE:
        block_reason = "趋势内首红候选不足2只"
    elif market_median <= MARKET_NEUTRAL_LOWER_PCT:
        block_reason = "科技池13周中位涨幅不高于-5%，弱势期只观察"
    elif market_median >= MARKET_NEUTRAL_UPPER_PCT:
        block_reason = "科技池13周中位涨幅不低于+5%，追涨期只观察"
    else:
        block_reason = ""
    selection_valid = block_reason == ""
    candidates["Selection_Valid"] = bool(selection_valid)
    candidates["Selection_Block_Reason"] = block_reason

    if eligible_count:
        # 不再让追强与启动分主导。R3 采用趋势、位置风险、R1 总分的词典序。
        eligible = candidates.loc[eligible.index].sort_values(
            [
                "Score_Trend_20",
                "Score_Risk_10",
                "Entry_Score_100",
                "ts_code",
            ],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        rank_map = pd.Series(np.arange(1, len(eligible) + 1, dtype=int), index=eligible.index)
        candidates.loc[rank_map.index, "Rank"] = rank_map.astype(float)
        if selection_valid:
            selected_index = rank_map[rank_map <= TOP_N].index
            candidates.loc[selected_index, "Selected_Top2"] = True

    candidates["Raw_Setup_Count"] = raw_count
    candidates["Eligible_Trend_Count"] = eligible_count
    candidates["Market_13W_Median_pct"] = market_median
    candidates["Market_Regime"] = (
        "强势"
        if market_median >= MARKET_NEUTRAL_UPPER_PCT
        else "弱势"
        if market_median <= MARKET_NEUTRAL_LOWER_PCT
        else "中性"
    )
    candidates = candidates.sort_values(
        ["Trend_Eligible", "Rank", "Entry_Score_100", "ts_code"],
        ascending=[False, True, False, True],
        na_position="last",
        kind="mergesort",
    )
    return candidates.reset_index(drop=True), raw_count, eligible_count


# -----------------------------------------------------------------------------
# 买入后固定路径标签：只评价入口，不构造退出策略
# -----------------------------------------------------------------------------
def track_fixed_future_path(
    ts_code: str,
    signal_date: str,
    signal_close: float,
    signal_raw_close: float,
    stock_qfq_dict: dict[str, pd.DataFrame],
    roundtrip_cost_pct: float,
    market_dates=None,
):
    result: dict[str, Any] = {
        "Entry_Tradable": False,
        "Entry_Date": None,
        "Entry_Open": np.nan,
        "Entry_Gap_pct": np.nan,
        "Outcome_Complete": False,
        "Outcome_Complete_8W": False,
        "Primary_Outcome_Date": None,
        "Primary_Return_Net_pct": np.nan,
        "Available_Future_Days": 0,
        "Available_Price_Days": 0,
        "Fixed_W8_Return_Net_pct": np.nan,
        "MFE_W3_Net_pct": np.nan,
        "MAE_W3_Raw_pct": np.nan,
        "MFE_8W_Net_pct": np.nan,
        "MAE_8W_Raw_pct": np.nan,
        "Path_10_vs_Minus5": "PENDING",
        "Early_Failure_2W": np.nan,
        "Outcome_Grade": "待完成",
    }
    for week in range(1, HOLD_WEEKS + 1):
        result[f"Fixed_Return_W{week}_Net_pct"] = np.nan
    if ts_code not in stock_qfq_dict:
        result["Entry_Status"] = "无行情"
        return result
    stock = stock_qfq_dict[ts_code]
    if market_dates is None:
        future_market_dates = stock.index[stock.index > signal_date].tolist()[
            : HOLD_WEEKS * MARKET_DAYS_PER_WEEK
        ]
    else:
        future_market_dates = [
            str(item) for item in market_dates if str(item) > signal_date
        ][: HOLD_WEEKS * MARKET_DAYS_PER_WEEK]
    result["Available_Future_Days"] = int(len(future_market_dates))
    if not future_market_dates:
        result["Entry_Status"] = "等待下一交易日"
        return result

    entry_date = future_market_dates[0]
    if entry_date not in stock.index:
        result["Entry_Date"] = entry_date
        result["Entry_Status"] = "下一交易日停牌或无行情，无法成交"
        return result
    future = stock.reindex(future_market_dates).copy()
    result["Available_Price_Days"] = int(future["close"].notna().sum())

    first = future.iloc[0]
    buy_price = _safe_float(first.get("open"))
    if not math.isfinite(buy_price) or buy_price <= 0:
        result["Entry_Status"] = "下一交易日开盘价缺失"
        return result
    result["Entry_Date"] = str(future.index[0])
    raw_buy_price = _safe_float(first.get("raw_open"), buy_price)
    raw_first_high = _safe_float(first.get("raw_high"), _safe_float(first.get("high")))
    raw_first_low = _safe_float(first.get("raw_low"), _safe_float(first.get("low")))
    raw_first_close = _safe_float(first.get("raw_close"), _safe_float(first.get("close")))
    result["Entry_Open"] = raw_buy_price
    result["Entry_Open_QFQ"] = buy_price
    result["Entry_Gap_pct"] = (raw_buy_price / signal_raw_close - 1.0) * 100.0

    is_20cm = ts_code.startswith(("300", "301", "688", "689"))
    limit_threshold = 0.195 if is_20cm else 0.095
    one_price_board = (
        all(math.isfinite(item) for item in (raw_first_high, raw_first_low, raw_first_close))
        and np.isclose(
            raw_first_high,
            raw_first_low,
            rtol=0,
            atol=max(0.001, raw_buy_price * 1e-5),
        )
        and (raw_first_close / signal_raw_close - 1.0) >= limit_threshold
    )
    if one_price_board:
        result["Entry_Status"] = "下一交易日一字涨停，无法成交"
        return result

    result["Entry_Tradable"] = True
    result["Entry_Status"] = "可成交"
    for week in range(1, HOLD_WEEKS + 1):
        day_index = week * MARKET_DAYS_PER_WEEK - 1
        if len(future) > day_index:
            marked_close = pd.to_numeric(future["close"], errors="coerce").ffill()
            exit_close = _safe_float(marked_close.iloc[day_index])
            if math.isfinite(exit_close):
                result[f"Fixed_Return_W{week}_Net_pct"] = (
                    (exit_close / buy_price - 1.0) * 100.0 - roundtrip_cost_pct
                )
    result["Fixed_W8_Return_Net_pct"] = _safe_float(
        result.get("Fixed_Return_W8_Net_pct")
    )

    highs = pd.to_numeric(future["high"], errors="coerce")
    lows = pd.to_numeric(future["low"], errors="coerce")
    if highs.notna().any():
        result["MFE_8W_Net_pct"] = (
            (highs.max() / buy_price - 1.0) * 100.0 - roundtrip_cost_pct
        )
    if lows.notna().any():
        result["MAE_8W_Raw_pct"] = (lows.min() / buy_price - 1.0) * 100.0

    primary_days = PRIMARY_HOLD_WEEKS * MARKET_DAYS_PER_WEEK
    primary_future = future.head(primary_days)
    primary_highs = pd.to_numeric(primary_future["high"], errors="coerce")
    primary_lows = pd.to_numeric(primary_future["low"], errors="coerce")
    if primary_highs.notna().any():
        result["MFE_W3_Net_pct"] = (
            (primary_highs.max() / buy_price - 1.0) * 100.0 - roundtrip_cost_pct
        )
    if primary_lows.notna().any():
        result["MAE_W3_Raw_pct"] = (primary_lows.min() / buy_price - 1.0) * 100.0

    path = "NEITHER"
    first_plus10_day = None
    first_minus5_day = None
    for day_number, (_, row) in enumerate(primary_future.iterrows(), start=1):
        hit_plus = _safe_float(row.get("high"), -np.inf) >= buy_price * 1.10
        hit_minus = _safe_float(row.get("low"), np.inf) <= buy_price * 0.95
        if hit_plus and first_plus10_day is None:
            first_plus10_day = day_number
        if hit_minus and first_minus5_day is None:
            first_minus5_day = day_number
        if hit_plus or hit_minus:
            if hit_plus and hit_minus:
                path = "SAME_DAY_AMBIGUOUS"
            elif hit_plus:
                path = "PLUS10_FIRST"
            else:
                path = "MINUS5_FIRST"
            break
    result["Path_10_vs_Minus5"] = path
    result["First_Plus10_Day"] = first_plus10_day
    result["First_Minus5_Day"] = first_minus5_day
    first_ten_days = future.head(10)
    early_low = pd.to_numeric(first_ten_days["low"], errors="coerce").min()
    result["Early_Failure_2W"] = bool(
        math.isfinite(_safe_float(early_low))
        and early_low <= buy_price * 0.95
        and path != "PLUS10_FIRST"
    )

    primary_return = _safe_float(result.get(PRIMARY_RETURN_COLUMN))
    primary_complete = len(future) >= primary_days and math.isfinite(primary_return)
    complete_8w = (
        len(future) >= HOLD_WEEKS * MARKET_DAYS_PER_WEEK
        and math.isfinite(_safe_float(result.get("Fixed_Return_W8_Net_pct")))
    )
    result["Outcome_Complete"] = bool(primary_complete)
    result["Outcome_Complete_8W"] = bool(complete_8w)
    result["Primary_Return_Net_pct"] = primary_return
    if len(future) >= primary_days:
        result["Primary_Outcome_Date"] = str(future.index[primary_days - 1])
    if primary_complete:
        mfe = _safe_float(result["MFE_W3_Net_pct"], -np.inf)
        if mfe >= 15.0 and primary_return >= 5.0:
            grade = "S"
        elif path == "PLUS10_FIRST" or primary_return >= 5.0:
            grade = "A"
        elif primary_return >= 0.0:
            grade = "B"
        else:
            grade = "F"
        result["Outcome_Grade"] = grade
    return result


# -----------------------------------------------------------------------------
# 扫描账本、断点和单周扫描
# -----------------------------------------------------------------------------
def make_config_id(min_price: float, min_mv: float, max_mv: float, roundtrip_cost_pct: float):
    payload = {
        "strategy": APP_VERSION,
        "min_price": float(min_price),
        "min_mv": float(min_mv),
        "max_mv": float(max_mv),
        "roundtrip_cost_pct": float(roundtrip_cost_pct),
        "top_n": TOP_N,
        "hold_weeks": HOLD_WEEKS,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def append_checkpoint_atomic(new_rows: pd.DataFrame):
    if new_rows.empty:
        return
    existing = read_csv_safe(CHECKPOINT_FILE)
    combined = (
        pd.concat([existing, new_rows], ignore_index=True, sort=False)
        if not existing.empty
        else new_rows.copy()
    )
    if "Signal_Date" in combined.columns:
        combined["Signal_Date"] = combined["Signal_Date"].map(parse_yyyymmdd)
    keys = [column for column in ("Config_ID", "Signal_Date", "ts_code") if column in combined.columns]
    if keys:
        combined = combined.drop_duplicates(keys, keep="last")
    sort_columns = [column for column in ("Signal_Date", "Rank", "ts_code") if column in combined.columns]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="mergesort", na_position="last")
    atomic_write_csv(combined.reset_index(drop=True), CHECKPOINT_FILE)


def replace_checkpoint_date(new_rows: pd.DataFrame, signal_date: str, config_id: str):
    """整周替换，避免重扫后旧候选残留。即使本周变成零候选也会删除旧行。"""
    existing = read_csv_safe(CHECKPOINT_FILE)
    if not existing.empty and {"Signal_Date", "Config_ID"}.issubset(existing.columns):
        normalized_dates = existing["Signal_Date"].map(parse_yyyymmdd)
        keep = ~(
            normalized_dates.eq(str(signal_date))
            & existing["Config_ID"].astype(str).eq(str(config_id))
        )
        existing = existing[keep].copy()
    if not new_rows.empty:
        combined = (
            pd.concat([existing, new_rows], ignore_index=True, sort=False)
            if not existing.empty
            else new_rows.copy()
        )
    else:
        combined = existing
    if combined.empty:
        remove_with_backup(CHECKPOINT_FILE)
        return
    combined["Signal_Date"] = combined["Signal_Date"].map(parse_yyyymmdd)
    keys = [column for column in ("Config_ID", "Signal_Date", "ts_code") if column in combined.columns]
    if keys:
        combined = combined.drop_duplicates(keys, keep="last")
    sort_columns = [column for column in ("Signal_Date", "Rank", "ts_code") if column in combined.columns]
    combined = combined.sort_values(sort_columns, kind="mergesort", na_position="last")
    atomic_write_csv(combined.reset_index(drop=True), CHECKPOINT_FILE)


def mark_scan_complete(
    signal_date: str,
    raw_signal_count: int,
    eligible_count: int,
    selected_count: int,
    config_id: str,
    selection_block_reason: str = "",
):
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    row = pd.DataFrame(
        [
            {
                "Signal_Date": str(signal_date),
                "Raw_Setup_Count": int(raw_signal_count),
                "Eligible_Trend_Count": int(eligible_count),
                "Selected_Count": int(selected_count),
                "Selection_Block_Reason": str(selection_block_reason or ""),
                "Scan_Status": "COMPLETED",
                "Config_ID": config_id,
                "Updated_At": datetime.now().isoformat(timespec="seconds"),
            }
        ]
    )
    ledger = pd.concat([ledger, row], ignore_index=True, sort=False) if not ledger.empty else row
    ledger["Signal_Date"] = ledger["Signal_Date"].map(parse_yyyymmdd)
    ledger = ledger.drop_duplicates(["Signal_Date", "Config_ID"], keep="last")
    atomic_write_csv(ledger.sort_values("Signal_Date").reset_index(drop=True), SCAN_LEDGER_FILE)


def completed_scan_dates(config_id: str):
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    if ledger.empty or not {"Signal_Date", "Config_ID", "Scan_Status"}.issubset(ledger.columns):
        return set()
    match = ledger[
        (ledger["Config_ID"].astype(str) == str(config_id))
        & (ledger["Scan_Status"].astype(str) == "COMPLETED")
    ]
    return set(filter(None, (parse_yyyymmdd(value) for value in match["Signal_Date"])))


def invalidate_recent_ledger_once(config_id: str, start_date: str, end_date: str):
    """新任务启动时让最近10周重算一次；自动批次续跑不会再次失效。"""
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    if ledger.empty or not {"Signal_Date", "Config_ID"}.issubset(ledger.columns):
        return
    dates = ledger["Signal_Date"].map(parse_yyyymmdd)
    recent_cutoff = (datetime.now() - timedelta(days=75)).strftime("%Y%m%d")
    lower = max(str(start_date), recent_cutoff)
    remove_mask = (
        ledger["Config_ID"].astype(str).eq(str(config_id))
        & dates.ge(lower)
        & dates.le(str(end_date))
    )
    if remove_mask.any():
        remaining = ledger[~remove_mask].copy()
        if remaining.empty:
            remove_with_backup(SCAN_LEDGER_FILE)
        else:
            atomic_write_csv(remaining.reset_index(drop=True), SCAN_LEDGER_FILE)


def save_task(task: dict[str, Any]):
    task = dict(task)
    task["Updated_At"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(task, RUN_TASK_FILE)


@contextmanager
def _task_file_guard():
    """只保护一次任务文件读改写；进程崩溃后 10 秒自动清理，不充当运行锁。"""
    guard_path = RUN_TASK_FILE + ".guard"
    acquired = False
    for _ in range(80):
        try:
            descriptor = os.open(guard_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            acquired = True
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(guard_path) > 10.0:
                    os.remove(guard_path)
                    continue
            except OSError:
                pass
            time.sleep(0.05)
    if not acquired:
        raise RuntimeError("任务状态文件暂时忙，请稍后重试。")
    try:
        yield
    finally:
        try:
            os.remove(guard_path)
        except OSError:
            pass


def _lease_is_fresh(task: dict[str, Any]):
    raw = str(task.get("Lease_Expires_At", "") or "")
    try:
        return datetime.fromisoformat(raw) > datetime.now()
    except (TypeError, ValueError):
        return False


def acquire_task_lease(worker_id: str):
    """同一 Streamlit 会话稳定续租；旧页面失联 45 秒后由新页面自动接管。"""
    with _task_file_guard():
        task = read_json_safe(RUN_TASK_FILE)
        if task.get("State") != "RUNNING":
            return False, task
        owner = str(task.get("Owner_ID", "") or "")
        if owner and owner != worker_id and _lease_is_fresh(task):
            return False, task
        task["Owner_ID"] = worker_id
        task["Lease_Expires_At"] = (
            datetime.now() + timedelta(seconds=TASK_LEASE_SECONDS)
        ).isoformat(timespec="seconds")
        save_task(task)
        return True, task


def refresh_task_lease(task_id: str, worker_id: str):
    with _task_file_guard():
        task = read_json_safe(RUN_TASK_FILE)
        if (
            task.get("State") != "RUNNING"
            or str(task.get("Task_ID", "")) != str(task_id)
            or str(task.get("Owner_ID", "")) != str(worker_id)
        ):
            return False
        task["Lease_Expires_At"] = (
            datetime.now() + timedelta(seconds=TASK_LEASE_SECONDS)
        ).isoformat(timespec="seconds")
        save_task(task)
        return True


def save_owned_task(task: dict[str, Any], worker_id: str):
    """仅允许当前租约持有者写任务，防止失联旧页面夺回新页面的租约。"""
    with _task_file_guard():
        current = read_json_safe(RUN_TASK_FILE)
        if (
            str(current.get("Task_ID", "")) != str(task.get("Task_ID", ""))
            or str(current.get("Owner_ID", "")) != str(worker_id)
        ):
            return False
        updated = dict(task)
        updated["Owner_ID"] = worker_id
        updated["Lease_Expires_At"] = (
            datetime.now() + timedelta(seconds=TASK_LEASE_SECONDS)
        ).isoformat(timespec="seconds")
        save_task(updated)
        return True


def resume_paused_task(worker_id: str):
    """用户明确点击继续时原子接管暂停任务。"""
    with _task_file_guard():
        task = read_json_safe(RUN_TASK_FILE)
        if task.get("State") != "PAUSED_ERROR":
            return False
        task["State"] = "RUNNING"
        task["Error_Count"] = 0
        task.pop("Last_Error", None)
        task["Owner_ID"] = worker_id
        task["Lease_Expires_At"] = (
            datetime.now() + timedelta(seconds=TASK_LEASE_SECONDS)
        ).isoformat(timespec="seconds")
        save_task(task)
        return True


def build_run_dates(
    pro,
    start_date: str,
    end_date: str,
    is_preview_mode: bool,
    config_id: str,
):
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    calendar_start = (start_dt - timedelta(days=14)).strftime("%Y%m%d")
    calendar_end = (end_dt + timedelta(days=14)).strftime("%Y%m%d")
    calendar = safe_tushare_call(
        pro.trade_cal,
        exchange="SSE",
        start_date=calendar_start,
        end_date=calendar_end,
    )
    if calendar.empty:
        raise RuntimeError("无法获取交易日历。")
    today_str = datetime.now().strftime("%Y%m%d")
    open_days = calendar[calendar["is_open"] == 1].copy()
    open_days["cal_date"] = open_days["cal_date"].astype(str)
    available_days = open_days[open_days["cal_date"] <= min(end_date, today_str)]
    if available_days.empty:
        raise RuntimeError("所选区间没有已完成的交易日。")
    open_days["dt"] = pd.to_datetime(open_days["cal_date"])
    open_days["year_week"] = open_days["dt"].dt.strftime("%G_%V")
    week_ends = set(open_days.groupby("year_week")["cal_date"].max().tolist())
    if is_preview_mode:
        latest = available_days["cal_date"].max()
        return [latest], [latest], latest in week_ends

    requested = sorted(
        item
        for item in available_days["cal_date"].tolist()
        if start_date <= item <= end_date and item in week_ends
    )
    processed = completed_scan_dates(config_id)
    pending = [item for item in requested if item not in processed]
    return requested, pending, True


def scan_one_date(
    signal_date: str,
    whitelist_keys,
    basic_name_map: dict[str, str],
    industry_map: dict[str, str],
    stock_qfq_dict: dict[str, pd.DataFrame],
    basic_indexed: pd.DataFrame,
    market_dates,
    min_price: float,
    min_mv: float,
    max_mv: float,
    roundtrip_cost_pct: float,
    is_preview_mode: bool,
    weekly_data_mode: str,
):
    pool_records: list[dict[str, Any]] = []
    for ts_code in whitelist_keys:
        stock = stock_qfq_dict.get(ts_code)
        if stock is None or signal_date not in stock.index:
            continue
        latest = stock.loc[signal_date]
        if isinstance(latest, pd.DataFrame):
            latest = latest.iloc[-1]
        raw_close = _safe_float(latest.get("raw_close"), _safe_float(latest.get("close")))
        if not math.isfinite(raw_close) or raw_close < min_price:
            continue

        circ_mv_billion = np.nan
        turnover_rate = np.nan
        if not basic_indexed.empty and (signal_date, ts_code) in basic_indexed.index:
            basic_row = basic_indexed.loc[(signal_date, ts_code)]
            if isinstance(basic_row, pd.DataFrame):
                basic_row = basic_row.iloc[-1]
            circ_mv_billion = _safe_float(basic_row.get("circ_mv")) / 10000.0
            turnover_rate = _safe_float(basic_row.get("turnover_rate"))
        if not math.isfinite(circ_mv_billion):
            continue
        if circ_mv_billion < min_mv or circ_mv_billion > max_mv:
            continue

        snapshot = compute_signal_snapshot(ts_code, signal_date, stock_qfq_dict)
        if not snapshot:
            continue
        snapshot.update(
            {
                "ts_code": ts_code,
                "name": basic_name_map.get(ts_code, ts_code),
                "Industry": industry_map.get(ts_code, "未分类"),
                "Signal_Date": signal_date,
                "Weekly_Data_Mode": weekly_data_mode,
                "Raw_Close": raw_close,
                "Circ_MV_Billion": circ_mv_billion,
                "Turnover_Rate": turnover_rate,
            }
        )
        pool_records.append(snapshot)

    if not pool_records:
        return pd.DataFrame(), 0, 0
    candidates, raw_count, eligible_count = score_r3_candidates(
        pd.DataFrame(pool_records)
    )
    if candidates.empty:
        return candidates, raw_count, eligible_count

    if is_preview_mode:
        for column, value in {
            "Entry_Tradable": np.nan,
            "Outcome_Complete": False,
            "Outcome_Complete_8W": False,
            "Primary_Outcome_Date": None,
            "Primary_Return_Net_pct": np.nan,
            "Entry_Status": "最新预览不计算未来结果",
            "Outcome_Grade": "待发生",
        }.items():
            candidates[column] = value
    else:
        outcome_rows = []
        for _, row in candidates.iterrows():
            outcome_rows.append(
                track_fixed_future_path(
                    str(row["ts_code"]),
                    signal_date,
                    _safe_float(row["Signal_Close"]),
                    _safe_float(row["Raw_Close"]),
                    stock_qfq_dict,
                    roundtrip_cost_pct,
                    market_dates,
                )
            )
        candidates = pd.concat(
            [candidates.reset_index(drop=True), pd.DataFrame(outcome_rows)],
            axis=1,
        )
    return candidates, raw_count, eligible_count


# -----------------------------------------------------------------------------
# R3稳健性报告
# -----------------------------------------------------------------------------
def _bool_series(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _profit_factor(returns: pd.Series):
    values = pd.to_numeric(returns, errors="coerce").dropna()
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def _trimmed_mean(returns: pd.Series, fraction: float = 0.05):
    values = pd.to_numeric(returns, errors="coerce").dropna().sort_values()
    if values.empty:
        return np.nan
    cut = int(math.floor(len(values) * fraction))
    if cut <= 0 or len(values) <= cut * 2:
        return values.mean()
    return values.iloc[cut:-cut].mean()


def _remove_best_fraction(returns: pd.Series, fraction: float):
    values = pd.to_numeric(returns, errors="coerce").dropna().sort_values(ascending=False)
    if values.empty:
        return values
    remove_count = max(1, int(math.ceil(len(values) * fraction)))
    return values.iloc[remove_count:]


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96):
    if total <= 0:
        return np.nan
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = probability + z * z / (2.0 * total)
    adjustment = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    )
    return (centre - adjustment) / denominator


def completed_research_rows(history: pd.DataFrame):
    if history.empty:
        return history.copy()
    complete = _bool_series(history, "Outcome_Complete")
    tradable = _bool_series(history, "Entry_Tradable")
    trend = _bool_series(history, "Trend_Eligible")
    valid_selection = _bool_series(history, "Selection_Valid")
    result = history[complete & tradable & trend & valid_selection].copy()
    result["Rank"] = pd.to_numeric(result.get("Rank"), errors="coerce")
    result[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        result.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    return result.dropna(subset=["Rank", PRIMARY_RETURN_COLUMN])


def cohort_summary(completed: pd.DataFrame):
    cohorts = [
        ("Top1", completed[completed["Rank"] == 1]),
        ("Top2", completed[completed["Rank"] == 2]),
        ("Top1—2合计", completed[completed["Rank"] <= TOP_N]),
        ("第3名以后", completed[completed["Rank"] > TOP_N]),
    ]
    rows = []
    for name, frame in cohorts:
        returns = pd.to_numeric(frame[PRIMARY_RETURN_COLUMN], errors="coerce").dropna()
        without_best = returns.drop(index=returns.idxmax()) if len(returns) > 1 else pd.Series(dtype=float)
        rows.append(
            {
                "排名组": name,
                "交易数": len(returns),
                "信号周": frame.loc[returns.index, "Signal_Date"].nunique() if len(returns) else 0,
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "5%截尾均益%": _trimmed_mean(returns),
                "去最佳一只均益%": without_best.mean() if len(without_best) else np.nan,
                "Profit_Factor": _profit_factor(returns),
                "+10先于-5比例%": (
                    frame.loc[returns.index, "Path_10_vs_Minus5"].astype(str).eq("PLUS10_FIRST").mean()
                    * 100.0
                    if len(returns)
                    else np.nan
                ),
                "两周早败率%": (
                    _bool_series(frame.loc[returns.index], "Early_Failure_2W").mean() * 100.0
                    if len(returns)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def outlier_audit(completed: pd.DataFrame):
    selected = completed[completed["Rank"] <= TOP_N].copy()
    returns = pd.to_numeric(selected[PRIMARY_RETURN_COLUMN], errors="coerce").dropna()
    if returns.empty:
        return pd.DataFrame(), {}
    positive_total = returns[returns > 0].sum()
    sorted_positive = returns[returns > 0].sort_values(ascending=False)
    best_contribution = (
        sorted_positive.iloc[0] / positive_total * 100.0
        if len(sorted_positive) and positive_total > 0
        else np.nan
    )
    top5_count = max(1, int(math.ceil(len(returns) * 0.05)))
    top5_contribution = (
        sorted_positive.head(top5_count).sum() / positive_total * 100.0
        if len(sorted_positive) and positive_total > 0
        else np.nan
    )

    variants = [
        ("原始Top2", returns),
        ("去掉最佳1只", returns.drop(index=returns.idxmax()) if len(returns) > 1 else pd.Series(dtype=float)),
        ("去掉收益最高1%", _remove_best_fraction(returns, 0.01)),
        ("去掉收益最高5%", _remove_best_fraction(returns, 0.05)),
    ]
    rows = []
    for name, values in variants:
        rows.append(
            {
                "口径": name,
                "样本数": len(values),
                "收益率简单合计%": values.sum() if len(values) else np.nan,
                "平均收益%": values.mean() if len(values) else np.nan,
                "中位收益%": values.median() if len(values) else np.nan,
                "胜率%": (values > 0).mean() * 100.0 if len(values) else np.nan,
                "Profit_Factor": _profit_factor(values),
            }
        )
    details = {
        "best_return": returns.max(),
        "best_contribution": best_contribution,
        "top5_contribution": top5_contribution,
        "positive_total": positive_total,
    }
    return pd.DataFrame(rows), details


def year_summary(completed: pd.DataFrame):
    if completed.empty:
        return pd.DataFrame()
    frame = completed[completed["Rank"] <= TOP_N].copy()
    frame["Year"] = frame["Signal_Date"].astype(str).str[:4]
    rows = []
    for year, group in frame.groupby("Year", sort=True):
        returns = group[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "年份": year,
                "交易数": len(group),
                "信号周": group["Signal_Date"].nunique(),
                "胜率%": (returns > 0).mean() * 100.0,
                "中位收益%": returns.median(),
                "平均收益%": returns.mean(),
                "去最佳一只均益%": returns.drop(index=returns.idxmax()).mean() if len(returns) > 1 else np.nan,
                "Profit_Factor": _profit_factor(returns),
                "F级比例%": group["Outcome_Grade"].astype(str).eq("F").mean() * 100.0,
            }
        )
    return pd.DataFrame(rows)


def regime_gate_summary(history: pd.DataFrame):
    """对中性门内外都按同一 R3 排名判卷，防止只展示被允许交易的样本。"""
    if history.empty:
        return pd.DataFrame()
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & _bool_series(history, "Trend_Eligible")
    ].copy()
    frame["Rank"] = pd.to_numeric(frame.get("Rank"), errors="coerce")
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    frame = frame[
        frame["Rank"].le(TOP_N) & frame[PRIMARY_RETURN_COLUMN].notna()
    ]
    rows = []
    for regime, group in frame.groupby("Market_Regime", sort=False):
        returns = group[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "市场状态": regime,
                "R3动作": "允许Top2" if str(regime) == "中性" else "只观察",
                "交易数": len(returns),
                "信号周": group["Signal_Date"].nunique(),
                "胜率%": (returns > 0).mean() * 100.0,
                "中位收益%": returns.median(),
                "平均收益%": returns.mean(),
                "Profit_Factor": _profit_factor(returns),
            }
        )
    return pd.DataFrame(rows)


def two_stock_group_summary(completed: pd.DataFrame):
    rows = []
    selected = completed[completed["Rank"] <= TOP_N].copy()
    for signal_date, group in selected.groupby("Signal_Date"):
        exact = group.sort_values("Rank").drop_duplicates("Rank")
        if set(exact["Rank"].astype(int)) != {1, 2}:
            continue
        returns = exact[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "Signal_Date": signal_date,
                "盈利只数": int((returns > 0).sum()),
                "两股平均收益%": returns.mean(),
                "两股中位收益%": returns.median(),
                "两股最差收益%": returns.min(),
                "两股最佳收益%": returns.max(),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, {}
    summary = {
        "完整两股组": len(detail),
        "平均盈利只数": detail["盈利只数"].mean(),
        "两只都盈利比例%": detail["盈利只数"].eq(2).mean() * 100.0,
        "两只全亏比例%": detail["盈利只数"].eq(0).mean() * 100.0,
        "两股组平均收益中位数%": detail["两股平均收益%"].median(),
    }
    return detail, summary


def horizon_summary(completed: pd.DataFrame):
    selected = completed[completed["Rank"] <= TOP_N].copy()
    rows = []
    for week in range(1, HOLD_WEEKS + 1):
        column = f"Fixed_Return_W{week}_Net_pct"
        if column not in selected.columns:
            continue
        values = pd.to_numeric(selected[column], errors="coerce").dropna()
        rows.append(
            {
                "持有周": f"W{week}",
                "完整样本": len(values),
                "胜率%": (values > 0).mean() * 100.0 if len(values) else np.nan,
                "中位收益%": values.median() if len(values) else np.nan,
                "平均收益%": values.mean() if len(values) else np.nan,
                "5%截尾均益%": _trimmed_mean(values),
                "Profit_Factor": _profit_factor(values),
            }
        )
    return pd.DataFrame(rows)


def ranking_diagnostics(completed: pd.DataFrame):
    if completed.empty:
        return {}
    # R3 是词典序，不把六因子总分伪装成唯一排序分；直接审计实际名次。
    score = -pd.to_numeric(completed.get("Rank"), errors="coerce")
    returns = pd.to_numeric(completed.get(PRIMARY_RETURN_COLUMN), errors="coerce")
    global_corr = score.rank(method="average").corr(returns.rank(method="average"))
    weekly_corrs = []
    for _, group in completed.groupby("Signal_Date"):
        if len(group) < 3:
            continue
        group_scores = -pd.to_numeric(group["Rank"], errors="coerce")
        group_returns = pd.to_numeric(
            group[PRIMARY_RETURN_COLUMN], errors="coerce"
        ).rank(method="average")
        corr = group_scores.corr(group_returns)
        if pd.notna(corr):
            weekly_corrs.append(corr)
    top2 = returns[completed["Rank"] <= TOP_N]
    rest = returns[completed["Rank"] > TOP_N]
    paired_advantages = []
    for _, group in completed.groupby("Signal_Date"):
        selected_returns = pd.to_numeric(
            group.loc[group["Rank"] <= TOP_N, PRIMARY_RETURN_COLUMN], errors="coerce"
        ).dropna()
        other_returns = pd.to_numeric(
            group.loc[group["Rank"] > TOP_N, PRIMARY_RETURN_COLUMN], errors="coerce"
        ).dropna()
        if len(selected_returns) and len(other_returns):
            paired_advantages.append(selected_returns.mean() - other_returns.mean())

    selected = completed[completed["Rank"] <= TOP_N].copy()
    signal_dates = sorted(selected["Signal_Date"].astype(str).unique().tolist())
    split_at = len(signal_dates) // 2
    first_dates = set(signal_dates[:split_at])
    second_dates = set(signal_dates[split_at:])
    first_half = pd.to_numeric(
        selected.loc[selected["Signal_Date"].astype(str).isin(first_dates), PRIMARY_RETURN_COLUMN],
        errors="coerce",
    ).dropna()
    second_half = pd.to_numeric(
        selected.loc[selected["Signal_Date"].astype(str).isin(second_dates), PRIMARY_RETURN_COLUMN],
        errors="coerce",
    ).dropna()
    return {
        "全局实际排名收益秩相关": global_corr,
        "逐周秩相关均值": np.mean(weekly_corrs) if weekly_corrs else np.nan,
        "逐周可计算周数": len(weekly_corrs),
        "Top2中位收益%": top2.median() if len(top2) else np.nan,
        "其余候选中位收益%": rest.median() if len(rest) else np.nan,
        "Top2相对其余中位优势百分点": (
            top2.median() - rest.median() if len(top2) and len(rest) else np.nan
        ),
        "Top2逐周平均收益战胜其余比例%": (
            np.mean(np.asarray(paired_advantages) > 0.0) * 100.0
            if paired_advantages
            else np.nan
        ),
        "Top2逐周平均收益优势均值百分点": (
            np.mean(paired_advantages) if paired_advantages else np.nan
        ),
        "前半段Top2中位收益%": first_half.median() if len(first_half) else np.nan,
        "后半段Top2中位收益%": second_half.median() if len(second_half) else np.nan,
    }


def research_gates(
    completed: pd.DataFrame,
    cohort: pd.DataFrame,
    outlier: pd.DataFrame,
    diagnostics: dict[str, Any],
):
    selected = completed[completed["Rank"] <= TOP_N]
    returns = selected[PRIMARY_RETURN_COLUMN] if not selected.empty else pd.Series(dtype=float)
    weeks = selected["Signal_Date"].nunique() if not selected.empty else 0
    wins = int((returns > 0).sum()) if len(returns) else 0
    lower_bound = _wilson_lower_bound(wins, len(returns)) * 100.0 if len(returns) else np.nan
    rest = completed[completed["Rank"] > TOP_N][PRIMARY_RETURN_COLUMN]
    remove5 = outlier[outlier["口径"] == "去掉收益最高5%"] if not outlier.empty else pd.DataFrame()
    remove5_mean = _safe_float(remove5["平均收益%"].iloc[0]) if not remove5.empty else np.nan
    remove5_median = _safe_float(remove5["中位收益%"].iloc[0]) if not remove5.empty else np.nan
    if not remove5.empty:
        try:
            remove5_pf = float(remove5["Profit_Factor"].iloc[0])
        except (TypeError, ValueError):
            remove5_pf = np.nan
    else:
        remove5_pf = np.nan
    rank_rows = cohort[cohort["排名组"].isin(["Top1", "Top2"])] if not cohort.empty else pd.DataFrame()
    all_rank_medians_positive = (
        len(rank_rows) == 2 and pd.to_numeric(rank_rows["中位收益%"], errors="coerce").gt(0).all()
    )
    rank_medians = (
        rank_rows.set_index("排名组")["中位收益%"].map(_safe_float)
        if len(rank_rows) == 2
        else pd.Series(dtype=float)
    )
    rank_ordered = (
        len(rank_medians) == 2
        and rank_medians.get("Top1", -np.inf) >= rank_medians.get("Top2", np.inf)
    )
    weekly_corr = _safe_float(diagnostics.get("逐周秩相关均值"))
    paired_beat = _safe_float(diagnostics.get("Top2逐周平均收益战胜其余比例%"))
    first_half_median = _safe_float(diagnostics.get("前半段Top2中位收益%"))
    second_half_median = _safe_float(diagnostics.get("后半段Top2中位收益%"))
    gates = [
        ("一年内至少18个独立中性信号周", weeks >= 18, f"当前{weeks}周"),
        ("至少36笔完整Top2交易", len(returns) >= 36, f"当前{len(returns)}笔"),
        ("W3 Top2胜率至少55%", len(returns) > 0 and (returns > 0).mean() >= 0.55, f"当前{((returns > 0).mean() * 100.0 if len(returns) else np.nan):.1f}%"),
        ("胜率95%下限高于50%", math.isfinite(lower_bound) and lower_bound > 50.0, f"当前{lower_bound:.1f}%"),
        ("W3 Top2中位收益大于0", len(returns) > 0 and returns.median() > 0, f"当前{(returns.median() if len(returns) else np.nan):.2f}%"),
        ("去掉最高5%后平均收益大于0", math.isfinite(remove5_mean) and remove5_mean > 0, f"当前{remove5_mean:.2f}%"),
        ("去掉最高5%后中位收益大于0", math.isfinite(remove5_median) and remove5_median > 0, f"当前{remove5_median:.2f}%"),
        ("去掉最高5%后PF至少1.2", not math.isnan(remove5_pf) and remove5_pf >= 1.2, f"当前{remove5_pf:.2f}"),
        ("Top1和Top2中位收益分别为正", all_rank_medians_positive, "逐名检查"),
        ("Top1中位收益不低于Top2", rank_ordered, "Top1≥Top2"),
        ("逐周实际排名收益秩相关至少0.05", math.isfinite(weekly_corr) and weekly_corr >= 0.05, f"当前{weekly_corr:.3f}"),
        ("Top2逐周战胜其余候选至少55%", math.isfinite(paired_beat) and paired_beat >= 55.0, f"当前{paired_beat:.1f}%"),
        ("前后半段Top2中位收益均为正", math.isfinite(first_half_median) and math.isfinite(second_half_median) and first_half_median > 0 and second_half_median > 0, f"前{first_half_median:.2f}% / 后{second_half_median:.2f}%"),
        ("Top2中位收益优于其余候选", len(returns) > 0 and len(rest) > 0 and returns.median() > rest.median(), f"差{(returns.median() - rest.median() if len(returns) and len(rest) else np.nan):.2f}个百分点"),
    ]
    return pd.DataFrame(
        [{"验收项目": name, "结果": "通过" if passed else "未通过", "当前值": value} for name, passed, value in gates]
    )


def build_export_zip(
    history: pd.DataFrame,
    cohort: pd.DataFrame,
    outlier: pd.DataFrame,
    yearly: pd.DataFrame,
    groups: pd.DataFrame,
    horizon: pd.DataFrame,
    gates: pd.DataFrame,
    diagnostics: dict[str, Any],
    regimes: pd.DataFrame,
):
    buffer = io.BytesIO()
    files = {
        "01_all_r3_first_red_candidates.csv": history,
        "02_rank_cohort_summary.csv": cohort,
        "03_outlier_dependency_audit.csv": outlier,
        "04_year_summary.csv": yearly,
        "05_complete_two_stock_groups.csv": groups,
        "06_w1_w8_fixed_horizon.csv": horizon,
        "07_research_acceptance_gates.csv": gates,
        "08_ranking_diagnostics.csv": pd.DataFrame(
            [{"指标": key, "数值": value} for key, value in diagnostics.items()]
        ),
        "09_market_regime_gate_audit.csv": regimes,
    }
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, frame in files.items():
            archive.writestr(filename, frame.to_csv(index=False).encode("utf-8-sig"))
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Streamlit 主程序
# -----------------------------------------------------------------------------
def _format_report_frame(frame: pd.DataFrame):
    result = frame.copy()
    for column in result.columns:
        if column.endswith("%") or column.endswith("收益%") or column.endswith("均益%"):
            result[column] = pd.to_numeric(result[column], errors="coerce").round(2)
        elif "Factor" in column or "相关" in column:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(3)
    return result


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🔬 {APP_TITLE}")
    st.caption(
        "R3只保留趋势内MACD第一根红柱；按趋势分、风险分、R1总分依次破同分。"
        "仅在科技池13周中位涨幅处于-5%到+5%时选择Top2。W3主验收，W1—W8固定记录。"
    )
    st.caption(f"运行引擎修订：{ENGINE_PATCH}")
    st.warning(
        "R3仍是研究验证版。当前规则来自R1/R2结果，只能用新的回测结果判卷；通过一年验收后仍需更长样本外验证。"
    )
    with st.expander("查看R3规则边界"):
        st.markdown(
            """
- **候选触发**：本周MACD柱由非正转正，且收盘不低于20周均线、20周均线四周斜率为正。
- **排序**：R1六因子完整保留用于审计；实际名次先看趋势20分，再看风险10分，最后才用六因子总分破同分。
- **市场门**：科技池13周涨幅中位数必须严格处于-5%到+5%；弱势和追涨阶段均只观察，不选股。
- **组合**：每个有效周只选Top2，不设置“候选拥挤”门，避免再次误删相对更好的周。
- **历史执行**：下一交易日开盘买入；W3为主目标，同时固定观察W1—W8并扣除往返成本。
- **明确排除**：买入后的走势、止损、止盈、移动保护、S/A/B/F结果均不参与入场评分。
            """
        )

    today = datetime.now().date()
    default_start = today - timedelta(days=365)
    with st.sidebar:
        st.header("研究配置")
        mode = st.radio(
            "运行模式",
            ["历史R3验证", "最新选股预览"],
            index=0,
            help="历史模式只使用完整周线；最新预览允许使用本周未完成周线且不写入回测。",
        )
        start_input = st.date_input("验证开始日期", value=default_start, disabled=mode != "历史R3验证")
        end_input = st.date_input("验证截止日期", value=today)

        st.markdown("---")
        st.subheader("基础股票池硬条件")
        min_price = st.number_input("最低股价（元）", value=10.0, min_value=0.0, step=1.0)
        min_mv = st.number_input("最低流通市值（亿元）", value=100.0, min_value=0.0, step=10.0)
        max_mv = st.number_input("最高流通市值（亿元）", value=1000.0, min_value=100.0, step=100.0)
        roundtrip_cost_pct = st.number_input(
            "往返交易成本（占买价%）",
            value=0.20,
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            help="R3所有W1—W8固定周收益都会扣除该成本。",
        )

        st.markdown("---")
        try:
            secret_token = st.secrets.get("TUSHARE_TOKEN", "")
        except Exception:
            secret_token = ""
        token_input = st.text_input("Tushare Token", value=secret_token, type="password")

        st.markdown("---")
        clear_market_clicked = st.button("清空行情缓存")
        clear_history_clicked = st.button("清除R3历史结果")

    if max_mv <= min_mv:
        st.error("最高流通市值必须大于最低流通市值。")
        return
    if start_input > end_input and mode == "历史R3验证":
        st.error("验证开始日期不能晚于截止日期。")
        return

    if clear_market_clicked:
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.success("行情缓存已清空。")

    if clear_history_clicked:
        for path in (CHECKPOINT_FILE, SCAN_LEDGER_FILE, RUN_TASK_FILE):
            remove_with_backup(path)
        st.session_state.pop("r3_preview", None)
        st.success("R3历史结果和断点任务已清除。")

    token_clean = clean_token_str(token_input)
    config_id = make_config_id(min_price, min_mv, max_mv, roundtrip_cost_pct)
    is_preview_mode = mode == "最新选股预览"
    if "r3_worker_id" not in st.session_state:
        st.session_state["r3_worker_id"] = uuid.uuid4().hex
    worker_id = str(st.session_state["r3_worker_id"])
    task_before = read_json_safe(RUN_TASK_FILE)

    if task_before.get("State") in {"RUNNING", "PAUSED_ERROR"}:
        done = int(task_before.get("Completed_Weeks", 0))
        total = int(task_before.get("Total_Weeks", 0))
        state_text = "运行中" if task_before.get("State") == "RUNNING" else "已暂停"
        st.info(f"检测到历史断点任务：{state_text}，已完成{done}/{total}周。")

    resume_clicked = False
    if task_before.get("State") == "PAUSED_ERROR":
        resume_clicked = st.button("从断点继续")
    stop_clicked = False
    if task_before.get("State") in {"RUNNING", "PAUSED_ERROR"}:
        stop_clicked = st.button("停止断点任务")
    if stop_clicked:
        stopped = read_json_safe(RUN_TASK_FILE)
        stopped["State"] = "STOPPED"
        save_task(stopped)
        st.warning("任务已停止，已经完成的数据仍保留。")
    if resume_clicked:
        if not resume_paused_task(worker_id):
            st.warning("任务状态已经变化，请刷新页面后再操作。")

    start_label = "运行最新选股预览" if is_preview_mode else "启动历史R3验证"
    start_clicked = st.button(start_label, type="primary")
    start_precheck_valid = False
    if start_clicked:
        valid, message = verify_token_connection(token_clean)
        start_precheck_valid = bool(valid)
        if not valid:
            st.error(f"Token预检失败：{message}")
        elif not is_preview_mode:
            task_start = start_input.strftime("%Y%m%d")
            task_end = end_input.strftime("%Y%m%d")
            invalidate_recent_ledger_once(config_id, task_start, task_end)
            task = {
                "Task_ID": uuid.uuid4().hex,
                "State": "RUNNING",
                "Config_ID": config_id,
                "Params": {
                    "Start_Date": task_start,
                    "End_Date": task_end,
                    "Min_Price": float(min_price),
                    "Min_MV": float(min_mv),
                    "Max_MV": float(max_mv),
                    "Roundtrip_Cost_pct": float(roundtrip_cost_pct),
                },
                "Completed_Weeks": 0,
                "Total_Weeks": 0,
                "Error_Count": 0,
                "Owner_ID": worker_id,
                "Lease_Expires_At": (
                    datetime.now() + timedelta(seconds=TASK_LEASE_SECONDS)
                ).isoformat(timespec="seconds"),
            }
            save_task(task)

    active_task = read_json_safe(RUN_TASK_FILE)
    run_history = False
    if active_task.get("State") == "RUNNING" and not stop_clicked:
        run_history, active_task = acquire_task_lease(worker_id)
        if not run_history:
            st.info(
                "另一个页面正在处理同一断点；本页不会重复写入。若原页面已崩溃，"
                f"租约最多{TASK_LEASE_SECONDS}秒自动失效，刷新本页即可从断点接管。"
            )
    run_preview = start_clicked and is_preview_mode and start_precheck_valid
    rerun_needed = False

    if run_history or run_preview:
        if not token_clean:
            if run_history:
                active_task["State"] = "PAUSED_ERROR"
                active_task["Last_Error"] = "Token为空。"
                save_owned_task(active_task, worker_id)
            st.error("Token为空，历史断点已经保留。")
        else:
            try:
                if run_history:
                    params = active_task["Params"]
                    run_start = str(params["Start_Date"])
                    run_end = str(params["End_Date"])
                    run_min_price = float(params["Min_Price"])
                    run_min_mv = float(params["Min_MV"])
                    run_max_mv = float(params["Max_MV"])
                    run_cost = float(params["Roundtrip_Cost_pct"])
                    run_config_id = str(active_task["Config_ID"])
                else:
                    run_start = end_input.strftime("%Y%m%d")
                    run_end = end_input.strftime("%Y%m%d")
                    run_min_price = float(min_price)
                    run_min_mv = float(min_mv)
                    run_max_mv = float(max_mv)
                    run_cost = float(roundtrip_cost_pct)
                    run_config_id = config_id

                ts.set_token(token_clean)
                pro = ts.pro_api(token_clean)
                with st.spinner("构建固定科技股研究池……"):
                    whitelist_set, name_map, industry_map = load_custom_tech_whitelist(token_clean)
                whitelist_keys = tuple(sorted(whitelist_set))
                if not whitelist_keys:
                    raise RuntimeError("未取得科技股研究池，请检查Token权限或网络。")
                st.info(f"科技股研究池：{len(whitelist_keys)}只。")

                requested_dates, pending_dates, latest_is_completed_week = build_run_dates(
                    pro, run_start, run_end, run_preview, run_config_id
                )
                if run_history:
                    active_task["Total_Weeks"] = len(requested_dates)
                    active_task["Completed_Weeks"] = len(requested_dates) - len(pending_dates)
                    save_owned_task(active_task, worker_id)

                if not pending_dates:
                    if run_history:
                        remove_with_backup(RUN_TASK_FILE)
                        st.success("所选区间已经全部完成。")
                    else:
                        st.warning("没有可扫描日期。")
                else:
                    batch_dates = pending_dates if run_preview else pending_dates[:WEEKS_PER_BATCH]
                    # 保留R1/R2稳定的420日指标预热窗口；每批只装载3个扫描周。
                    fetch_start = (
                        datetime.strptime(min(batch_dates), "%Y%m%d") - timedelta(days=420)
                    ).strftime("%Y%m%d")
                    requested_fetch_end = datetime.strptime(max(batch_dates), "%Y%m%d") + timedelta(days=75)
                    fetch_end = min(requested_fetch_end.date(), today).strftime("%Y%m%d")
                    st.caption(
                        f"本批扫描{batch_dates[0]}—{batch_dates[-1]}；"
                        f"只加载必要行情窗口{fetch_start}—{fetch_end}。"
                    )
                    lease_heartbeat = (
                        lambda: refresh_task_lease(
                            str(active_task.get("Task_ID", "")), worker_id
                        )
                    ) if run_history else None
                    (
                        stocks,
                        basic_indexed,
                        market_dates,
                        loaded_dates,
                        failed_dates,
                        sync_stats,
                    ) = load_optimized_market_data(
                        fetch_start,
                        fetch_end,
                        token_clean,
                        whitelist_keys,
                        lease_heartbeat=lease_heartbeat,
                    )
                    st.caption(
                        f"行情分片：复用{sync_stats.get('cached_days', 0)}天，"
                        f"本次保存{sync_stats.get('downloaded_days', 0)}天；"
                        f"daily_basic仅下载{sync_stats.get('weekly_basic_days', 0)}个周末交易日。"
                    )
                    if failed_dates:
                        raise RuntimeError(
                            f"仍有{len(failed_dates)}个交易日下载失败；成功分片已保存，"
                            "重试时只补这些日期。"
                        )
                    if not stocks:
                        raise RuntimeError("未加载到行情；已成功下载的分片仍然保留。")

                    loaded_date_set = set(loaded_dates)
                    progress = st.progress(0, text="开始扫描R3趋势内首红候选……")
                    stopped_during_batch = False
                    for idx, signal_date in enumerate(batch_dates):
                        if run_history and not refresh_task_lease(
                            str(active_task.get("Task_ID", "")), worker_id
                        ):
                            raise RuntimeError("任务租约已经转移，本页停止写入。")
                        if run_history and read_json_safe(RUN_TASK_FILE).get("State") == "STOPPED":
                            stopped_during_batch = True
                            break
                        if signal_date not in loaded_date_set:
                            raise RuntimeError(f"扫描日{signal_date}行情分片不完整，断点已保留。")
                        weekly_mode = (
                            "已完成周线"
                            if run_history or latest_is_completed_week
                            else "未完成周线预览"
                        )
                        candidates, raw_count, eligible_count = scan_one_date(
                            signal_date,
                            whitelist_keys,
                            name_map,
                            industry_map,
                            stocks,
                            basic_indexed,
                            market_dates,
                            run_min_price,
                            run_min_mv,
                            run_max_mv,
                            run_cost,
                            run_preview,
                            weekly_mode,
                        )
                        selected_count = (
                            int(_bool_series(candidates, "Selected_Top2").sum())
                            if not candidates.empty
                            else 0
                        )
                        if run_preview:
                            st.session_state["r3_preview"] = candidates
                        else:
                            if not candidates.empty:
                                candidates["Config_ID"] = run_config_id
                            if not refresh_task_lease(
                                str(active_task.get("Task_ID", "")), worker_id
                            ):
                                raise RuntimeError("任务租约已经转移，本页停止写入回测断点。")
                            replace_checkpoint_date(candidates, signal_date, run_config_id)
                            mark_scan_complete(
                                signal_date,
                                raw_count,
                                eligible_count,
                                selected_count,
                                run_config_id,
                                (
                                    str(candidates["Selection_Block_Reason"].iloc[0] or "")
                                    if not candidates.empty
                                    and "Selection_Block_Reason" in candidates.columns
                                    else "没有结构触发"
                                ),
                            )
                            active_task["Completed_Weeks"] = int(active_task.get("Completed_Weeks", 0)) + 1
                            active_task["Last_Date"] = signal_date
                            active_task["Error_Count"] = 0
                            save_owned_task(active_task, worker_id)
                        progress.progress(
                            (idx + 1) / len(batch_dates),
                            text=(
                                f"{signal_date}：首红触发{raw_count}只，"
                                f"合格候选{eligible_count}只，入选{selected_count}只"
                            ),
                        )
                    progress.empty()

                    # 进入下一批前主动释放股票字典，避免Streamlit反复rerun后内存累积。
                    del stocks, basic_indexed
                    gc.collect()
                    if run_preview:
                        st.success("最新候选预览完成，不会写入历史验证。")
                    elif stopped_during_batch:
                        st.warning("任务已停止，本批已完成结果仍然保留。")
                    else:
                        remaining = len(pending_dates) - len(batch_dates)
                        if remaining > 0:
                            st.success(f"本批完成{len(batch_dates)}周，剩余{remaining}周将自动续跑。")
                            rerun_needed = True
                        else:
                            remove_with_backup(RUN_TASK_FILE)
                            st.success("历史R3扫描完成。")
            except Exception as exc:
                gc.collect()
                if run_history:
                    latest_task = read_json_safe(RUN_TASK_FILE) or active_task
                    still_owner = (
                        str(latest_task.get("Task_ID", ""))
                        == str(active_task.get("Task_ID", ""))
                        and str(latest_task.get("Owner_ID", "")) == worker_id
                    )
                    if not still_owner:
                        st.warning(f"任务已由其他页面接管，本页停止：{exc}")
                    else:
                        errors = int(latest_task.get("Error_Count", 0)) + 1
                        latest_task["Error_Count"] = errors
                        latest_task["Last_Error"] = str(exc)
                        if errors < 3:
                            latest_task["State"] = "RUNNING"
                            rerun_needed = True
                            st.warning(f"临时异常，断点已保留，将自动重试（{errors}/3）：{exc}")
                        else:
                            latest_task["State"] = "PAUSED_ERROR"
                            st.error(f"连续3次失败，任务已暂停：{exc}")
                        save_owned_task(latest_task, worker_id)
                else:
                    st.error(f"运行失败：{exc}")

    preview = st.session_state.get("r3_preview")
    if is_preview_mode and isinstance(preview, pd.DataFrame):
        st.markdown("---")
        st.header("最新选股预览")
        if preview.empty:
            st.info("最新交易日没有R3趋势内首红候选。")
        else:
            selected_preview = preview[_bool_series(preview, "Selected_Top2")].copy()
            if selected_preview.empty:
                block_reason = ""
                if "Selection_Block_Reason" in preview.columns and not preview.empty:
                    block_reason = str(preview["Selection_Block_Reason"].iloc[0] or "")
                st.warning(block_reason or "本周没有形成有效Top2选股组。")
            else:
                preview_columns = [
                    "Signal_Date", "Weekly_Data_Mode", "Rank", "name", "ts_code", "Industry",
                    "R3_Setup_Type", "Score_Trend_20", "Score_Risk_10",
                    "Entry_Score_100", "Score_Pullback_15", "Score_Contraction_15",
                    "Score_Restart_15", "Score_RS_25",
                    "Raw_Close", "Circ_MV_Billion", "Market_Regime", "Selection_Block_Reason",
                ]
                st.dataframe(
                    selected_preview[[column for column in preview_columns if column in selected_preview.columns]],
                    width="stretch",
                )
            with st.expander("查看全部R3首红候选及未入选原因"):
                st.dataframe(preview, width="stretch")

    if rerun_needed:
        # 下一批前立即重跑，不在每个小批次重复构建整份历史报告和ZIP。
        gc.collect()
        time.sleep(0.3)
        st.rerun()

    raw_history = read_csv_safe(CHECKPOINT_FILE)
    if not raw_history.empty:
        raw_history["Signal_Date"] = raw_history["Signal_Date"].map(parse_yyyymmdd)
        raw_history = raw_history.dropna(subset=["Signal_Date"])
        report_config_id = config_id
        if "Config_ID" in raw_history.columns:
            matching = raw_history[raw_history["Config_ID"].astype(str) == report_config_id]
            if matching.empty:
                report_config_id = str(raw_history["Config_ID"].dropna().astype(str).iloc[-1])
            history = raw_history[raw_history["Config_ID"].astype(str) == report_config_id].copy()
        else:
            history = raw_history.copy()

        completed = completed_research_rows(history)
        cohort = cohort_summary(completed)
        outlier, outlier_details = outlier_audit(completed)
        yearly = year_summary(completed)
        regimes = regime_gate_summary(history)
        group_detail, group_stats = two_stock_group_summary(completed)
        horizons = horizon_summary(completed)
        diagnostics = ranking_diagnostics(completed)
        gates = research_gates(completed, cohort, outlier, diagnostics)

        st.markdown("---")
        st.header("R3历史验证报告")
        st.caption(
            "主报告以已经走满15个交易日的W3样本验收入场排序；W4—W8只统计实际已经走到"
            "相应周数的记录，并在表中明确显示每个持有期的完整样本数。"
        )

        ledger = read_csv_safe(SCAN_LEDGER_FILE)
        if not ledger.empty and "Config_ID" in ledger.columns:
            ledger = ledger[ledger["Config_ID"].astype(str) == report_config_id].copy()
        scanned_weeks = len(ledger)
        invalid_selection_weeks = (
            int(
                (
                    pd.to_numeric(ledger.get("Selected_Count"), errors="coerce")
                    .fillna(0)
                    .lt(TOP_N)
                ).sum()
            )
            if not ledger.empty
            else 0
        )
        selected = completed[completed["Rank"] <= TOP_N]
        selected_returns = selected[PRIMARY_RETURN_COLUMN] if not selected.empty else pd.Series(dtype=float)
        metric_columns = st.columns(5)
        metric_columns[0].metric("已扫描周数", scanned_weeks)
        metric_columns[1].metric("无有效Top2周", invalid_selection_weeks)
        metric_columns[2].metric("W3完整Top2交易", len(selected))
        metric_columns[3].metric(
            "Top2胜率",
            f"{((selected_returns > 0).mean() * 100.0 if len(selected_returns) else np.nan):.1f}%",
        )
        metric_columns[4].metric(
            "W3 Top2中位收益",
            f"{(selected_returns.median() if len(selected_returns) else np.nan):.2f}%",
        )

        st.subheader("研究验收门槛")
        st.dataframe(gates, width="stretch", hide_index=True)
        if not gates.empty and gates["结果"].eq("通过").all():
            st.success("R3一年快速验收全部通过，可以冻结规则并进入更长区间样本外验证。")
        else:
            st.error("R3尚未通过全部验收，当前代码不能进入实盘，也不应靠止盈止损掩盖入口问题。")

        st.subheader("Top1、Top2与其余候选")
        st.dataframe(_format_report_frame(cohort), width="stretch", hide_index=True)

        st.subheader("极端牛股依赖审计")
        if outlier_details:
            outlier_cols = st.columns(3)
            outlier_cols[0].metric("最佳一笔W3收益", f"{outlier_details['best_return']:.2f}%")
            outlier_cols[1].metric("最佳一笔占正利润", f"{outlier_details['best_contribution']:.1f}%")
            outlier_cols[2].metric("最高5%占正利润", f"{outlier_details['top5_contribution']:.1f}%")
        st.dataframe(_format_report_frame(outlier), width="stretch", hide_index=True)

        st.subheader("排名是否真的有效")
        if diagnostics:
            diagnostic_frame = pd.DataFrame(
                [{"指标": key, "数值": value} for key, value in diagnostics.items()]
            )
            st.dataframe(diagnostic_frame, width="stretch", hide_index=True)

        st.subheader("完整两股组")
        if group_stats:
            group_cols = st.columns(4)
            group_cols[0].metric("完整两股组", group_stats["完整两股组"])
            group_cols[1].metric("平均盈利只数", f"{group_stats['平均盈利只数']:.2f}/2")
            group_cols[2].metric("两只都盈利", f"{group_stats['两只都盈利比例%']:.1f}%")
            group_cols[3].metric("两只全亏", f"{group_stats['两只全亏比例%']:.1f}%")
        else:
            st.info("尚无两只都走满W3的完整选股组。")

        st.subheader("固定持有W1—W8路径")
        st.dataframe(_format_report_frame(horizons), width="stretch", hide_index=True)

        st.subheader("分年稳定性")
        st.dataframe(_format_report_frame(yearly), width="stretch", hide_index=True)

        st.subheader("中性市场门内外对照")
        st.dataframe(_format_report_frame(regimes), width="stretch", hide_index=True)

        with st.expander("查看Top2历史明细"):
            detail_columns = [
                "Signal_Date", "Entry_Date", "Rank", "name", "ts_code", "Industry",
                "R3_Setup_Type", "Score_Trend_20", "Score_Risk_10", "Entry_Score_100",
                "Score_Pullback_15", "Score_Contraction_15", "Score_Restart_15", "Score_RS_25",
                "Entry_Open", PRIMARY_RETURN_COLUMN, "MFE_W3_Net_pct", "MAE_W3_Raw_pct",
                "Path_10_vs_Minus5", "Early_Failure_2W", "Outcome_Grade", "Market_Regime",
            ]
            selected_detail = history[
                _bool_series(history, "Selected_Top2")
            ].copy()
            selected_detail = selected_detail.sort_values(
                ["Signal_Date", "Rank"], ascending=[False, True]
            )
            st.dataframe(
                selected_detail[[column for column in detail_columns if column in selected_detail.columns]],
                width="stretch",
            )

        export_bytes = build_export_zip(
            history.drop(columns=["Config_ID"], errors="ignore"),
            cohort,
            outlier,
            yearly,
            group_detail,
            horizons,
            gates,
            diagnostics,
            regimes,
        )
        st.download_button(
            "下载R3完整研究结果",
            data=export_bytes,
            file_name="r3_neutral_first_red_w3_audit_results.zip",
            mime="application/zip",
        )

if __name__ == "__main__":
    main()
