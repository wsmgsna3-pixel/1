# -*- coding: utf-8 -*-
"""
R9 强势有序个股再启动 + R7 早期强势回调 + R3 中性首红 + R6 N6 弱势
首次转折四分支、最多 Top2 验证器。

R3、R6 与 R7 的原触发、资格和排名逐行冻结。R9 删除 R8 的“前两周市场状态、
上一周普遍回调、本周同步再扩散”八条件同周硬门，只用当前可观测的宽市场阶段
排除弱扩散和整体过热；市场不再要求与个股事件在同一周精确同步。

R9 个股入口仍是一次性事件：趋势完整，上一周横盘或温和整理，本周第一次收复
上周高点并恢复动量。排名把 R8 被证明无效的“上一周越强越好”改为“上一周
波动越小越有序”，并保留相对强度、行业超额、ATR 收缩和本周不过度上涨。
第一名正常入选；第二名只有分数和距离 MA20 同时达标才入选，不再强行凑满。
信号日仍为每周最后一个交易日，下一交易日开盘买入。
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

APP_VERSION = "R9.0-ORDERLY-STOCK-RESTART-FOUR-BRANCH-MAX2-W3-AUDIT"
APP_TITLE = "R9强势有序个股再启动四分支最多Top2验证器"
ENGINE_PATCH = "R9.0-R3-R6-R7-FROZEN-BROAD-PHASE-OPTIONAL-SECOND-STABLE"

CHECKPOINT_FILE = "r9_four_branch_candidates.csv"
SCAN_LEDGER_FILE = "r9_four_branch_scanned_dates.csv"
OPPORTUNITY_FILE = "r9_four_branch_w3_major_winner_opportunities.csv"
RUN_TASK_FILE = "r9_four_branch_running_task.json"
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
STRONG_EARLY_UPPER_PCT = 10.0
STRONG_RESET_MAX_MARKET_1W_MEDIAN_PCT = 0.0
STRONG_RESET_MAX_POSITIVE_BREADTH = 0.55
STRONG_MAX_WEEKLY_RETURN_PCT = 15.0
STRONG_MAX_DISTANCE_MA20_PCT = 25.0
STRONG_MAX_WEEKLY_RANGE_PCT = 25.0
R9_MIN_MARKET_13W_PCT = 5.0
R9_MAX_MARKET_13W_PCT = 30.0
R9_MIN_MARKET_1W_PCT = 0.0
R9_MAX_MARKET_1W_PCT = 6.0
R9_MIN_POSITIVE_BREADTH = 0.50
R9_MAX_POSITIVE_BREADTH = 0.90
R9_MIN_VALID_SELECTION_SIZE = 1
R9_SECOND_MIN_SCORE = 60.0
R9_SECOND_MAX_DISTANCE_MA20_PCT = 18.0
REACCEL_MIN_PREVIOUS_RETURN_PCT = -8.0
REACCEL_MAX_PREVIOUS_RETURN_PCT = 5.0
REACCEL_MAX_WEEKLY_RETURN_PCT = 12.0
REACCEL_MAX_DISTANCE_MA20_PCT = 25.0
REACCEL_MAX_WEEKLY_RANGE_PCT = 25.0
REACCEL_MIN_CLOSE_LOCATION = 0.60
RECOVERY_OVERSOLD_LEVEL = 35.0
RECOVERY_DEEP_DRAWDOWN_PCT = -20.0
RECOVERY_MAX_WEEKLY_RETURN_PCT = 25.0
RECOVERY_MAX_LOW_REBOUND_PCT = 40.0
RECOVERY_STRONG_CLOSE_LOCATION = 0.70
R5_BASELINE_MIN_MARKET_1W_MEDIAN_PCT = 0.0
R5_BASELINE_MIN_POSITIVE_BREADTH = 0.55
R5_BASELINE_MIN_POOL_FRACTION = 0.005
R5_BASELINE_MIN_CANDIDATES = 3
MAJOR_WINNER_W3_PCT = 20.0
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

    # R3 原KDJ(9)继续用于六因子；复苏分支恢复历史验证过的 SKDJ N=6、M=3。
    # 精确口径：Raw RSV -> EMA(span=3) -> K再EMA(span=3) -> D为K的3周SMA。
    low6 = low.rolling(6).min()
    high6 = high.rolling(6).max()
    raw_rsv6 = (close - low6) / (high6 - low6).replace(0, 0.001) * 100.0
    weekly["skdj_rsv6"] = raw_rsv6.ewm(span=3, adjust=False).mean()
    weekly["skdj_k6"] = weekly["skdj_rsv6"].ewm(span=3, adjust=False).mean()
    weekly["skdj_d6"] = weekly["skdj_k6"].rolling(3).mean()

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
    previous3 = weekly.iloc[-4]

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
    previous_return_13w = _safe_float(previous.get("return_13w_pct"))
    previous2_return_13w = _safe_float(previous2.get("return_13w_pct"))
    prior_high_13w = _safe_float(current.get("prior_high_13w"))
    previous_prior_high_13w = _safe_float(previous.get("prior_high_13w"))
    previous_high_value = _safe_float(previous.get("high"))
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

    # R7 强势分支只增加“抗跌新高”事件，不改动上面的 R3 资格。科技池当周
    # 普遍回调时，仍能第一次创13周新高，代表相对韧性；但周涨幅、离MA20
    # 距离或周振幅过大时只观察，避免把高位加速误当成第二段启动。
    strong_trend_eligible = bool(
        base_trend_eligible
        and math.isfinite(ma40)
        and math.isfinite(ma20)
        and ma20 >= ma40
    )
    strong_resilience_trigger = bool(strong_trend_eligible and fresh_breakout)
    strong_risk_ok = bool(
        math.isfinite(return_1w)
        and 0.0 < return_1w <= STRONG_MAX_WEEKLY_RETURN_PCT
        and math.isfinite(distance_ma20)
        and 0.0 <= distance_ma20 <= STRONG_MAX_DISTANCE_MA20_PCT
        and math.isfinite(weekly_range)
        and weekly_range <= STRONG_MAX_WEEKLY_RANGE_PCT
    )
    strong_overheated = bool(strong_resilience_trigger and not strong_risk_ok)
    strong_eligible = bool(strong_resilience_trigger and strong_risk_ok)

    # R9 只定义个股的一次性“整理后再启动”事件，市场是否处于可交易的有序
    # 强势阶段在完整科技池形成后另行判定。这里不使用未来收益，也不把
    # K>D、MACD持续改善等可连续维持的状态单独当作买点。
    strong_reacceleration_trigger = bool(
        strong_trend_eligible
        and all(
            math.isfinite(item)
            for item in (
                current_close,
                previous_high_value,
                ma10,
                return_1w,
                previous_return_1w,
                current_hist,
                previous_hist,
                close_location_now,
            )
        )
        and REACCEL_MIN_PREVIOUS_RETURN_PCT
        <= previous_return_1w
        <= REACCEL_MAX_PREVIOUS_RETURN_PCT
        and current_close > previous_high_value
        and current_close >= ma10
        and 0.0 < return_1w <= REACCEL_MAX_WEEKLY_RETURN_PCT
        and current_hist > previous_hist
        and close_location_now >= REACCEL_MIN_CLOSE_LOCATION
    )
    strong_reacceleration_risk_ok = bool(
        strong_reacceleration_trigger
        and math.isfinite(distance_ma20)
        and 0.0 <= distance_ma20 <= REACCEL_MAX_DISTANCE_MA20_PCT
        and math.isfinite(weekly_range)
        and weekly_range <= REACCEL_MAX_WEEKLY_RANGE_PCT
    )
    strong_reacceleration_overheated = bool(
        strong_reacceleration_trigger and not strong_reacceleration_risk_ok
    )

    # R6弱势分支不等待MACD翻红或MA20斜率转正。实际入口必须是一个“事件”而
    # 不是能连续维持数周的状态：深跌且近期超卖后，K本周首次转升，同时价格
    # 至少出现周涨或强收之一；前两周若已有同类转升，本周不重复触发。
    # R5原宽触发继续单独计算，只用于同场对照，绝不参与R6入选。
    skdj_k6 = _safe_float(current.get("skdj_k6"))
    skdj_d6 = _safe_float(current.get("skdj_d6"))
    skdj_k6_prev = _safe_float(previous.get("skdj_k6"))
    skdj_d6_prev = _safe_float(previous.get("skdj_d6"))
    skdj_k6_prev2 = _safe_float(previous2.get("skdj_k6"))
    skdj_d6_prev2 = _safe_float(previous2.get("skdj_d6"))
    skdj_k6_prev3 = _safe_float(previous3.get("skdj_k6"))
    skdj_recent_values = [
        item
        for item in (
            skdj_k6,
            skdj_d6,
            skdj_k6_prev,
            skdj_d6_prev,
            skdj_k6_prev2,
            skdj_d6_prev2,
        )
        if math.isfinite(item)
    ]
    skdj_recent_min = min(skdj_recent_values) if skdj_recent_values else np.nan
    skdj_low_turn = bool(
        math.isfinite(skdj_recent_min)
        and skdj_recent_min <= RECOVERY_OVERSOLD_LEVEL
        and (
            (math.isfinite(skdj_k6_prev) and skdj_k6 > skdj_k6_prev)
            or (math.isfinite(skdj_d6) and skdj_k6 > skdj_d6)
        )
    )
    drawdown_26w = _safe_float(current.get("drawdown_26w_pct"))
    weekly_low = _safe_float(current.get("low"))
    rebound_from_week_low = (
        (current_close / weekly_low - 1.0) * 100.0
        if math.isfinite(current_close) and math.isfinite(weekly_low) and weekly_low > 0.0
        else np.nan
    )
    price_to_ma10_ratio = (
        current_close / ma10
        if math.isfinite(current_close) and math.isfinite(ma10) and ma10 > 0.0
        else np.nan
    )
    previous_close_location = _safe_float(previous.get("close_location"))
    previous2_close_location = _safe_float(previous2.get("close_location"))
    macd_hist_delta = (
        current_hist - previous_hist
        if math.isfinite(current_hist) and math.isfinite(previous_hist)
        else np.nan
    )
    macd_repairing = bool(math.isfinite(macd_hist_delta) and macd_hist_delta > 0.0)

    price_repair_now = bool(
        (math.isfinite(return_1w) and return_1w > 0.0)
        or (
            math.isfinite(close_location_now)
            and close_location_now >= RECOVERY_STRONG_CLOSE_LOCATION
        )
    )
    previous_turn_state = bool(
        all(math.isfinite(item) for item in (skdj_k6_prev, skdj_k6_prev2))
        and skdj_k6_prev > skdj_k6_prev2
        and (
            (math.isfinite(previous_return_1w) and previous_return_1w > 0.0)
            or (
                math.isfinite(previous_close_location)
                and previous_close_location >= RECOVERY_STRONG_CLOSE_LOCATION
            )
        )
    )
    previous2_turn_state = bool(
        all(math.isfinite(item) for item in (skdj_k6_prev2, skdj_k6_prev3))
        and skdj_k6_prev2 > skdj_k6_prev3
        and (
            (math.isfinite(previous2_return_1w) and previous2_return_1w > 0.0)
            or (
                math.isfinite(previous2_close_location)
                and previous2_close_location >= RECOVERY_STRONG_CLOSE_LOCATION
            )
        )
    )
    recovery_first_turn_event = bool(
        math.isfinite(drawdown_26w)
        and drawdown_26w <= RECOVERY_DEEP_DRAWDOWN_PCT
        and math.isfinite(skdj_recent_min)
        and skdj_recent_min <= RECOVERY_OVERSOLD_LEVEL
        and all(math.isfinite(item) for item in (skdj_k6, skdj_k6_prev))
        and skdj_k6 > skdj_k6_prev
        and price_repair_now
        and not previous_turn_state
        and not previous2_turn_state
    )
    recovery_overheated = bool(
        recovery_first_turn_event
        and (
            (math.isfinite(return_1w) and return_1w > RECOVERY_MAX_WEEKLY_RETURN_PCT)
            or (
                math.isfinite(rebound_from_week_low)
                and rebound_from_week_low > RECOVERY_MAX_LOW_REBOUND_PCT
            )
        )
    )
    recovery_eligible = bool(recovery_first_turn_event and not recovery_overheated)

    r5_near_ma10 = bool(
        math.isfinite(price_to_ma10_ratio) and price_to_ma10_ratio >= 0.75
    )
    r5_baseline_structure = bool(
        math.isfinite(drawdown_26w)
        and drawdown_26w <= RECOVERY_DEEP_DRAWDOWN_PCT
        and skdj_low_turn
        and math.isfinite(return_1w)
        and return_1w > 0.0
        and math.isfinite(close_location_now)
        and close_location_now >= 0.55
        and r5_near_ma10
    )
    r5_baseline_overheated = bool(
        r5_baseline_structure
        and (
            return_1w > RECOVERY_MAX_WEEKLY_RETURN_PCT
            or (
                math.isfinite(rebound_from_week_low)
                and rebound_from_week_low > RECOVERY_MAX_LOW_REBOUND_PCT
            )
        )
    )
    r5_baseline_eligible = bool(r5_baseline_structure and not r5_baseline_overheated)

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
        "Strong_Resilience_Trigger": strong_resilience_trigger,
        "Strong_Overheated": strong_overheated,
        "Strong_Eligible": strong_eligible,
        "Strong_Setup_Type": (
            "抗跌新高-过热观察"
            if strong_overheated
            else "抗跌新高"
            if strong_eligible
            else ""
        ),
        "Strong_Trend_Eligible": strong_trend_eligible,
        "Strong_Reacceleration_Trigger": strong_reacceleration_trigger,
        "Strong_Reacceleration_Risk_OK": strong_reacceleration_risk_ok,
        "Strong_Reacceleration_Overheated": strong_reacceleration_overheated,
        "Strong_Reacceleration_Setup_Type": (
            "整理后再加速-过热观察"
            if strong_reacceleration_overheated
            else "整理后再加速"
            if strong_reacceleration_trigger
            else ""
        ),
        "Recovery_Structure_Trigger": recovery_first_turn_event,
        "Recovery_Overheated": recovery_overheated,
        "Recovery_Eligible": recovery_eligible,
        "Recovery_Setup_Type": (
            "N6首次转折-过热观察"
            if recovery_overheated
            else "N6首次转折"
            if recovery_eligible
            else ""
        ),
        "R5_Baseline_Recovery_Structure_Trigger": r5_baseline_structure,
        "R5_Baseline_Recovery_Overheated": r5_baseline_overheated,
        "R5_Baseline_Recovery_Eligible": r5_baseline_eligible,
        "Recovery_Price_Repair": price_repair_now,
        "Recovery_Previous_Turn_State": previous_turn_state,
        "Recovery_Previous2_Turn_State": previous2_turn_state,
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
        "MACD_Hist_Delta": macd_hist_delta,
        "MACD_Repairing": macd_repairing,
        "MA10": ma10,
        "MA20": ma20,
        "MA40": ma40,
        "MA10_Slope_2W_pct": ma10_slope,
        "MA20_Slope_4W_pct": ma20_slope,
        "Distance_MA20_pct": distance_ma20,
        "Drawdown_26W_pct": drawdown_26w,
        "Weeks_Since_26W_High": weeks_since_high,
        "PreSignal_4W_Return_pct": _safe_float(current.get("pre_signal_4w_return_pct")),
        "Return_1W_pct": return_1w,
        "Previous_Return_1W_pct": previous_return_1w,
        "Return_2W_pct": _safe_float(current.get("return_2w_pct")),
        "Return_4W_pct": _safe_float(current.get("return_4w_pct")),
        "Return_8W_pct": _safe_float(current.get("return_8w_pct")),
        "Return_13W_pct": _safe_float(current.get("return_13w_pct")),
        "Previous_Return_13W_pct": previous_return_13w,
        "Previous2_Return_13W_pct": previous2_return_13w,
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
        "Weekly_SKDJ_K6": skdj_k6,
        "Weekly_SKDJ_D6": skdj_d6,
        "Previous_SKDJ_K6": skdj_k6_prev,
        "Previous_SKDJ_D6": skdj_d6_prev,
        "Previous2_SKDJ_K6": skdj_k6_prev2,
        "Previous2_SKDJ_D6": skdj_d6_prev2,
        "SKDJ_Recent_Min": skdj_recent_min,
        "SKDJ_Low_Turn": skdj_low_turn,
        "Rebound_From_Week_Low_pct": rebound_from_week_low,
        "Price_to_MA10_Ratio": price_to_ma10_ratio,
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


def _score_recovery_factors(frame: pd.DataFrame):
    """R5失败评分只作为基线对照；R6实际排名不得使用。"""
    scored = frame.copy()
    drawdown = _numeric_series(scored, "Drawdown_26W_pct")
    skdj_min = _numeric_series(scored, "SKDJ_Recent_Min")
    return_1w = _numeric_series(scored, "Return_1W_pct")
    rebound = _numeric_series(scored, "Rebound_From_Week_Low_pct")
    close_location = _numeric_series(scored, "Weekly_Close_Location")
    week_range = _numeric_series(scored, "Weekly_Range_pct")
    scored["Recovery_Score_Drawdown_20"] = (
        ((-drawdown - 20.0) / 30.0) * 20.0
    ).clip(0.0, 20.0)
    scored["Recovery_Score_SKDJ_20"] = (
        ((RECOVERY_OVERSOLD_LEVEL - skdj_min) / RECOVERY_OVERSOLD_LEVEL) * 15.0 + 5.0
    ).clip(0.0, 20.0)
    scored["Recovery_Score_Week_20"] = np.select(
        [
            (return_1w > 0.0) & (return_1w <= 3.0),
            return_1w <= 12.0,
            return_1w <= 20.0,
            return_1w <= RECOVERY_MAX_WEEKLY_RETURN_PCT,
        ],
        [10.0, 20.0, 15.0, 8.0],
        default=0.0,
    )
    scored["Recovery_Score_Early_15"] = np.select(
        [rebound <= 15.0, rebound <= 25.0, rebound <= RECOVERY_MAX_LOW_REBOUND_PCT],
        [15.0, 10.0, 5.0],
        default=0.0,
    )
    scored["Recovery_Score_MACD_10"] = np.where(
        _bool_series(scored, "MACD_Repairing"), 10.0, 3.0
    )
    scored["Recovery_Score_Close_10"] = (close_location * 10.0).clip(0.0, 10.0)
    scored["Recovery_Score_Risk_5"] = np.select(
        [week_range <= 12.0, week_range <= 20.0, week_range <= 30.0],
        [5.0, 3.0, 1.0],
        default=0.0,
    )
    columns = [
        "Recovery_Score_Drawdown_20",
        "Recovery_Score_SKDJ_20",
        "Recovery_Score_Week_20",
        "Recovery_Score_Early_15",
        "Recovery_Score_MACD_10",
        "Recovery_Score_Close_10",
        "Recovery_Score_Risk_5",
    ]
    scored["Recovery_Score_100"] = scored[columns].sum(axis=1).clip(0.0, 100.0)
    return scored


def _score_recovery_early_stage(frame: pd.DataFrame):
    """五项等权早期阶段指数；每项只使用当周横截面和买入前数据。"""
    scored = frame.copy()
    factors = [
        ("Return_2W_pct", "Recovery_Early_Return2W_20"),
        ("Price_to_MA10_Ratio", "Recovery_Early_MA10_Distance_20"),
        ("Weekly_SKDJ_K6", "Recovery_Early_SKDJ_20"),
        ("RS_8W_Pct", "Recovery_Early_RS8_20"),
        ("MACD_Impulse_Pct", "Recovery_Early_MACD_20"),
    ]
    component_columns = []
    for source, target in factors:
        # 五项均是数值越低代表反弹阶段越早；固定等权，不从W3收益拟合权重。
        scored[target] = _percentile_rank(
            _numeric_series(scored, source), higher_is_better=False
        ) * 20.0
        component_columns.append(target)
    scored["Recovery_Early_Stage_100"] = scored[component_columns].sum(axis=1)
    return scored


def _score_strong_resilience(frame: pd.DataFrame):
    """R7强势抗跌新高五项等权指数；只使用当周横截面和买入前数据。"""
    scored = frame.copy()
    factors = [
        ("PreSignal_4W_Return_pct", "Strong_Score_Pause20", False),
        ("ATR_Contraction", "Strong_Score_ATR20", False),
        ("Return_1W_pct", "Strong_Score_NonChase20", False),
        ("Industry_13W_Excess_pct", "Strong_Score_Industry20", True),
        ("Distance_MA20_pct", "Strong_Score_Position20", False),
    ]
    component_columns = []
    for source, target, higher_is_better in factors:
        scored[target] = _percentile_rank(
            _numeric_series(scored, source), higher_is_better=higher_is_better
        ) * 20.0
        component_columns.append(target)
    scored["Strong_Resilience_100"] = scored[component_columns].sum(axis=1)
    return scored


def _score_strong_reacceleration(frame: pd.DataFrame):
    """R9整理后再启动五项等权指数；权重固定且不读取未来收益。"""
    scored = frame.copy()
    pause_magnitude = _numeric_series(scored, "Previous_Return_1W_pct").abs()
    scored["Reaccel_Score_PauseControl20"] = (
        _percentile_rank(pause_magnitude, higher_is_better=False) * 20.0
    )
    factors = [
        ("RS_13W_Pct", "Reaccel_Score_RS13_20", True),
        ("Industry_13W_Excess_pct", "Reaccel_Score_Industry20", True),
        ("ATR_Contraction", "Reaccel_Score_ATR20", False),
        ("Return_1W_pct", "Reaccel_Score_NonChase20", False),
    ]
    component_columns = ["Reaccel_Score_PauseControl20"]
    for source, target, higher_is_better in factors:
        scored[target] = _percentile_rank(
            _numeric_series(scored, source), higher_is_better=higher_is_better
        ) * 20.0
        component_columns.append(target)
    scored["Strong_Reacceleration_100"] = scored[component_columns].sum(axis=1)
    return scored


def _market_state_metrics(pool: pd.DataFrame):
    """从当周完整科技池计算市场状态；全部字段在信号日已知。"""
    current_13w = _numeric_series(pool, "Return_13W_pct")
    previous_13w = _numeric_series(pool, "Previous_Return_13W_pct")
    previous2_13w = _numeric_series(pool, "Previous2_Return_13W_pct")
    current_1w = _numeric_series(pool, "Return_1W_pct")
    previous_1w = _numeric_series(pool, "Previous_Return_1W_pct")
    market_13w = _safe_float(current_13w.median(), 0.0)
    previous_market_13w = _safe_float(previous_13w.median(), 0.0)
    previous2_market_13w = _safe_float(previous2_13w.median(), 0.0)
    market_1w = _safe_float(current_1w.median(), 0.0)
    previous_market_1w = _safe_float(previous_1w.median(), 0.0)
    positive_breadth = float((current_1w > 0.0).mean()) if len(pool) else 0.0
    previous_positive_breadth = (
        float((previous_1w > 0.0).mean()) if len(pool) else 0.0
    )
    acceleration = market_13w - previous_market_13w
    regime = (
        "强势"
        if market_13w >= MARKET_NEUTRAL_UPPER_PCT
        else "弱势"
        if market_13w <= MARKET_NEUTRAL_LOWER_PCT
        else "中性"
    )
    reset_pass = bool(
        MARKET_NEUTRAL_UPPER_PCT <= market_13w < STRONG_EARLY_UPPER_PCT
        and market_1w <= STRONG_RESET_MAX_MARKET_1W_MEDIAN_PCT
        and positive_breadth < STRONG_RESET_MAX_POSITIVE_BREADTH
    )
    continuation_pass = bool(
        R9_MIN_MARKET_13W_PCT <= market_13w <= R9_MAX_MARKET_13W_PCT
        and R9_MIN_MARKET_1W_PCT < market_1w <= R9_MAX_MARKET_1W_PCT
        and R9_MIN_POSITIVE_BREADTH
        <= positive_breadth
        <= R9_MAX_POSITIVE_BREADTH
    )
    context = (
        "早期强势回调"
        if reset_pass
        else "早期强势扩张"
        if market_13w < STRONG_EARLY_UPPER_PCT
        else "延伸强势回调"
        if (
            market_1w <= STRONG_RESET_MAX_MARKET_1W_MEDIAN_PCT
            and positive_breadth < STRONG_RESET_MAX_POSITIVE_BREADTH
        )
        else "延伸强势扩张"
    )
    stage = (
        f"{regime}市场"
        if regime != "强势"
        else "早期强势回调-R7"
        if reset_pass
        else "强势有序扩张-R9"
        if continuation_pass
        else f"{context}-观察"
    )
    return {
        "Market_13W_Median_pct": market_13w,
        "Previous_Market_13W_Median_pct": previous_market_13w,
        "Previous2_Market_13W_Median_pct": previous2_market_13w,
        "Market_13W_Acceleration_pct": acceleration,
        "Market_1W_Median_pct": market_1w,
        "Previous_Market_1W_Median_pct": previous_market_1w,
        "Market_1W_Positive_Breadth": positive_breadth,
        "Previous_Market_1W_Positive_Breadth": previous_positive_breadth,
        "Market_Regime": regime,
        "Strong_Reset_Context_Pass": reset_pass,
        "Strong_Continuation_Context_Pass": continuation_pass,
        "Strong_Market_Context": context,
        "Strong_Market_Stage": stage,
    }


def score_r9_candidates(pool_snapshots: pd.DataFrame):
    """R9/R7/R3/R6按互斥市场阶段启用；既有三个分支计算保持不变。"""
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

    market_state = _market_state_metrics(pool)
    reacceleration_trigger = _bool_series(pool, "Strong_Reacceleration_Trigger")
    reacceleration_risk_ok = _bool_series(pool, "Strong_Reacceleration_Risk_OK")
    previous_market_1w = _safe_float(
        market_state.get("Previous_Market_1W_Median_pct"), 0.0
    )
    prior_resilience_pass = (
        _numeric_series(pool, "Previous_Return_1W_pct") >= previous_market_1w
    )
    pool["Strong_Reacceleration_Resilience_Pass"] = prior_resilience_pass
    # R8中“上一周必须强于市场中位数”在审计中反向失效。R9仅保留为观察字段，
    # 不再参与硬资格；真正资格只由一次性事件与不过热风险边界共同决定。
    pool["Strong_Reacceleration_Eligible"] = (
        reacceleration_trigger & reacceleration_risk_ok
    )
    pool["Strong_Reacceleration_Overheated"] = (
        reacceleration_trigger & ~reacceleration_risk_ok
    )

    r3_trigger = _bool_series(pool, "R3_Setup_Candidate")
    strong_trigger = _bool_series(pool, "Strong_Resilience_Trigger")
    reacceleration_trigger = _bool_series(pool, "Strong_Reacceleration_Trigger")
    recovery_trigger = _bool_series(pool, "Recovery_Structure_Trigger")
    r5_baseline_trigger = _bool_series(pool, "R5_Baseline_Recovery_Structure_Trigger")
    candidates = pool[
        r3_trigger
        | strong_trigger
        | reacceleration_trigger
        | recovery_trigger
        | r5_baseline_trigger
    ].copy()
    raw_count = int(
        (r3_trigger | strong_trigger | reacceleration_trigger | recovery_trigger).sum()
    )
    if candidates.empty:
        return candidates, 0, 0
    candidates = _score_r1_six_factors(candidates)
    candidates = _score_recovery_factors(candidates)
    candidates["Rank"] = np.nan
    candidates["R3_Rank"] = np.nan
    candidates["Strong_Rank"] = np.nan
    candidates["Strong_Reacceleration_Rank"] = np.nan
    candidates["Recovery_Rank"] = np.nan
    for column in (
        "Recovery_Early_Return2W_20",
        "Recovery_Early_MA10_Distance_20",
        "Recovery_Early_SKDJ_20",
        "Recovery_Early_RS8_20",
        "Recovery_Early_MACD_20",
        "Recovery_Early_Stage_100",
    ):
        candidates[column] = np.nan
    for column in (
        "Strong_Score_Pause20",
        "Strong_Score_ATR20",
        "Strong_Score_NonChase20",
        "Strong_Score_Industry20",
        "Strong_Score_Position20",
        "Strong_Resilience_100",
    ):
        candidates[column] = np.nan
    for column in (
        "Reaccel_Score_PauseControl20",
        "Reaccel_Score_RS13_20",
        "Reaccel_Score_Industry20",
        "Reaccel_Score_ATR20",
        "Reaccel_Score_NonChase20",
        "Strong_Reacceleration_100",
    ):
        candidates[column] = np.nan
    candidates["Selected_Top2"] = False
    candidates["Entry_Eligible"] = False
    candidates["R9_Second_Qualified"] = False
    candidates["R9_Selection_Qualified"] = False
    r3_eligible = candidates[_bool_series(candidates, "Trend_Eligible")].copy()
    strong_eligible = candidates[_bool_series(candidates, "Strong_Eligible")].copy()
    reacceleration_eligible = candidates[
        _bool_series(candidates, "Strong_Reacceleration_Eligible")
    ].copy()
    recovery_eligible = candidates[_bool_series(candidates, "Recovery_Eligible")].copy()
    r5_baseline_eligible = candidates[
        _bool_series(candidates, "R5_Baseline_Recovery_Eligible")
    ].copy()
    r3_eligible_count = len(r3_eligible)
    strong_eligible_count = len(strong_eligible)
    reacceleration_eligible_count = len(reacceleration_eligible)
    recovery_eligible_count = len(recovery_eligible)
    r5_baseline_eligible_count = len(r5_baseline_eligible)

    if r3_eligible_count:
        ordered = r3_eligible.sort_values(
            ["Score_Trend_20", "Score_Risk_10", "Entry_Score_100", "ts_code"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        rank_map = pd.Series(np.arange(1, len(ordered) + 1, dtype=int), index=ordered.index)
        candidates.loc[rank_map.index, "R3_Rank"] = rank_map.astype(float)
    if strong_eligible_count:
        strong_scored = _score_strong_resilience(strong_eligible)
        strong_columns = [
            "Strong_Score_Pause20",
            "Strong_Score_ATR20",
            "Strong_Score_NonChase20",
            "Strong_Score_Industry20",
            "Strong_Score_Position20",
            "Strong_Resilience_100",
        ]
        for column in strong_columns:
            candidates.loc[strong_scored.index, column] = strong_scored[column]
        ordered = strong_scored.sort_values(
            [
                "Strong_Resilience_100",
                "ATR_Contraction",
                "Return_1W_pct",
                "ts_code",
            ],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        rank_map = pd.Series(np.arange(1, len(ordered) + 1, dtype=int), index=ordered.index)
        candidates.loc[rank_map.index, "Strong_Rank"] = rank_map.astype(float)
    if reacceleration_eligible_count:
        reacceleration_scored = _score_strong_reacceleration(reacceleration_eligible)
        reacceleration_columns = [
            "Reaccel_Score_PauseControl20",
            "Reaccel_Score_RS13_20",
            "Reaccel_Score_Industry20",
            "Reaccel_Score_ATR20",
            "Reaccel_Score_NonChase20",
            "Strong_Reacceleration_100",
        ]
        for column in reacceleration_columns:
            candidates.loc[reacceleration_scored.index, column] = (
                reacceleration_scored[column]
            )
        ordered = reacceleration_scored.sort_values(
            [
                "Strong_Reacceleration_100",
                "RS_13W_Pct",
                "Return_1W_pct",
                "ts_code",
            ],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        rank_map = pd.Series(
            np.arange(1, len(ordered) + 1, dtype=int), index=ordered.index
        )
        candidates.loc[rank_map.index, "Strong_Reacceleration_Rank"] = (
            rank_map.astype(float)
        )
        r9_rank = pd.to_numeric(
            candidates["Strong_Reacceleration_Rank"], errors="coerce"
        )
        r9_score = pd.to_numeric(
            candidates["Strong_Reacceleration_100"], errors="coerce"
        )
        r9_distance = pd.to_numeric(
            candidates["Distance_MA20_pct"], errors="coerce"
        )
        second_qualified = (
            r9_rank.eq(2)
            & r9_score.ge(R9_SECOND_MIN_SCORE)
            & r9_distance.le(R9_SECOND_MAX_DISTANCE_MA20_PCT)
        )
        candidates.loc[second_qualified, "R9_Second_Qualified"] = True
        candidates.loc[
            r9_rank.eq(1) | second_qualified, "R9_Selection_Qualified"
        ] = True
    if recovery_eligible_count:
        early_scored = _score_recovery_early_stage(recovery_eligible)
        early_columns = [
            "Recovery_Early_Return2W_20",
            "Recovery_Early_MA10_Distance_20",
            "Recovery_Early_SKDJ_20",
            "Recovery_Early_RS8_20",
            "Recovery_Early_MACD_20",
            "Recovery_Early_Stage_100",
        ]
        for column in early_columns:
            candidates.loc[early_scored.index, column] = early_scored[column]
        ordered = early_scored.sort_values(
            [
                "Recovery_Early_Stage_100",
                "Price_to_MA10_Ratio",
                "Return_2W_pct",
                "ts_code",
            ],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        rank_map = pd.Series(np.arange(1, len(ordered) + 1, dtype=int), index=ordered.index)
        candidates.loc[rank_map.index, "Recovery_Rank"] = rank_map.astype(float)

    market_median = _safe_float(market_state.get("Market_13W_Median_pct"), 0.0)
    previous_market_13w = _safe_float(
        market_state.get("Previous_Market_13W_Median_pct"), 0.0
    )
    previous2_market_13w = _safe_float(
        market_state.get("Previous2_Market_13W_Median_pct"), 0.0
    )
    market_13w_acceleration = _safe_float(
        market_state.get("Market_13W_Acceleration_pct"), 0.0
    )
    market_1w_median = _safe_float(market_state.get("Market_1W_Median_pct"), 0.0)
    previous_market_1w = _safe_float(
        market_state.get("Previous_Market_1W_Median_pct"), 0.0
    )
    positive_breadth = _safe_float(
        market_state.get("Market_1W_Positive_Breadth"), 0.0
    )
    previous_positive_breadth = _safe_float(
        market_state.get("Previous_Market_1W_Positive_Breadth"), 0.0
    )
    r5_baseline_required_count = max(
        R5_BASELINE_MIN_CANDIDATES,
        int(math.ceil(len(pool) * R5_BASELINE_MIN_POOL_FRACTION)),
    )
    r5_baseline_market_gate_pass = bool(
        r5_baseline_eligible_count >= r5_baseline_required_count
        and market_1w_median > R5_BASELINE_MIN_MARKET_1W_MEDIAN_PCT
        and positive_breadth >= R5_BASELINE_MIN_POSITIVE_BREADTH
    )
    market_context = (
        "扩散修复"
        if market_1w_median > 0.0 and positive_breadth >= 0.55
        else "分化修复"
        if market_1w_median > 0.0 or positive_breadth >= 0.50
        else "逆风修复"
    )
    market_regime = str(market_state.get("Market_Regime", "中性"))
    strong_reset_context_pass = bool(
        market_state.get("Strong_Reset_Context_Pass", False)
    )
    strong_continuation_context_pass = bool(
        market_state.get("Strong_Continuation_Context_Pass", False)
    )
    strong_market_context = str(market_state.get("Strong_Market_Context", ""))
    strong_market_stage = str(market_state.get("Strong_Market_Stage", ""))
    if market_regime == "强势":
        if strong_reset_context_pass:
            active_eligible_count = strong_eligible_count
            candidates["Rank"] = candidates["Strong_Rank"]
            candidates["Entry_Eligible"] = _bool_series(candidates, "Strong_Eligible")
            active_branch = "R7强势抗跌新高"
            block_reason = (
                ""
                if strong_eligible_count >= MIN_VALID_SELECTION_SIZE
                else "强势抗跌新高候选不足2只"
            )
        elif strong_continuation_context_pass:
            active_eligible_count = reacceleration_eligible_count
            candidates["Rank"] = candidates["Strong_Reacceleration_Rank"]
            candidates["Entry_Eligible"] = _bool_series(
                candidates, "Strong_Reacceleration_Eligible"
            )
            active_branch = "R9强势有序个股再启动"
            block_reason = (
                ""
                if reacceleration_eligible_count >= R9_MIN_VALID_SELECTION_SIZE
                else "整理后再启动候选不足1只"
            )
        else:
            active_eligible_count = strong_eligible_count
            candidates["Rank"] = candidates["Strong_Rank"]
            candidates["Entry_Eligible"] = _bool_series(candidates, "Strong_Eligible")
            active_branch = "R9强势观察"
            block_reason = (
                f"{strong_market_stage}；仅R7早期回调或R9有序扩张允许入选"
            )
    elif market_regime == "中性":
        active_branch = "R3中性趋势"
        active_eligible_count = r3_eligible_count
        candidates["Rank"] = candidates["R3_Rank"]
        candidates["Entry_Eligible"] = _bool_series(candidates, "Trend_Eligible")
        block_reason = "" if r3_eligible_count >= MIN_VALID_SELECTION_SIZE else "趋势内首红候选不足2只"
    else:
        active_branch = "R6弱势首次转折-N6"
        active_eligible_count = recovery_eligible_count
        candidates["Rank"] = candidates["Recovery_Rank"]
        candidates["Entry_Eligible"] = _bool_series(candidates, "Recovery_Eligible")
        if recovery_eligible_count < MIN_VALID_SELECTION_SIZE:
            block_reason = "N6首次转折候选不足2只"
        else:
            block_reason = ""
    selection_valid = block_reason == ""
    candidates["Selection_Valid"] = bool(selection_valid)
    candidates["Selection_Block_Reason"] = block_reason
    candidates["Strategy_Branch"] = active_branch
    if selection_valid:
        if active_branch == "R9强势有序个股再启动":
            selected = (
                _bool_series(candidates, "Entry_Eligible")
                & _bool_series(candidates, "R9_Selection_Qualified")
            )
        else:
            selected = (
                _bool_series(candidates, "Entry_Eligible")
                & pd.to_numeric(candidates["Rank"], errors="coerce").le(TOP_N)
            )
        candidates.loc[selected, "Selected_Top2"] = True

    candidates["Raw_Setup_Count"] = raw_count
    candidates["Observation_Row_Count"] = len(candidates)
    candidates["R3_Raw_First_Red_Count"] = int(r3_trigger.sum())
    candidates["Strong_Structure_Count"] = int(strong_trigger.sum())
    candidates["Strong_Reacceleration_Structure_Count"] = int(
        reacceleration_trigger.sum()
    )
    candidates["Recovery_Structure_Count"] = int(recovery_trigger.sum())
    candidates["R5_Baseline_Recovery_Structure_Count"] = int(r5_baseline_trigger.sum())
    candidates["Recovery_Overheated_Count"] = int(_bool_series(pool, "Recovery_Overheated").sum())
    candidates["Strong_Overheated_Count"] = int(_bool_series(pool, "Strong_Overheated").sum())
    candidates["Strong_Reacceleration_Overheated_Count"] = int(
        _bool_series(pool, "Strong_Reacceleration_Overheated").sum()
    )
    candidates["Eligible_Trend_Count"] = r3_eligible_count
    candidates["Strong_Eligible_Count"] = strong_eligible_count
    candidates["Strong_Reacceleration_Eligible_Count"] = (
        reacceleration_eligible_count
    )
    candidates["Recovery_Eligible_Count"] = recovery_eligible_count
    candidates["R5_Baseline_Recovery_Eligible_Count"] = r5_baseline_eligible_count
    candidates["Active_Eligible_Count"] = active_eligible_count
    candidates["Strong_Required_Count"] = MIN_VALID_SELECTION_SIZE
    candidates["Strong_Reacceleration_Required_Count"] = R9_MIN_VALID_SELECTION_SIZE
    candidates["Recovery_Required_Count"] = MIN_VALID_SELECTION_SIZE
    candidates["R5_Baseline_Required_Count"] = r5_baseline_required_count
    candidates["R5_Baseline_Market_Gate_Pass"] = r5_baseline_market_gate_pass
    candidates["Market_13W_Median_pct"] = market_median
    candidates["Previous_Market_13W_Median_pct"] = previous_market_13w
    candidates["Previous2_Market_13W_Median_pct"] = previous2_market_13w
    candidates["Market_13W_Acceleration_pct"] = market_13w_acceleration
    candidates["Market_1W_Median_pct"] = market_1w_median
    candidates["Previous_Market_1W_Median_pct"] = previous_market_1w
    candidates["Market_1W_Positive_Breadth_pct"] = positive_breadth * 100.0
    candidates["Previous_Market_1W_Positive_Breadth_pct"] = (
        previous_positive_breadth * 100.0
    )
    candidates["Market_Recovery_Context"] = market_context
    candidates["Strong_Market_Context"] = strong_market_context
    candidates["Strong_Market_Stage"] = strong_market_stage
    candidates["Strong_Reset_Context_Pass"] = strong_reset_context_pass
    candidates["Strong_Continuation_Context_Pass"] = (
        strong_continuation_context_pass
    )
    candidates["Market_Regime"] = market_regime
    candidates = candidates.sort_values(
        [
            "Entry_Eligible",
            "Rank",
            "Strong_Reacceleration_100",
            "Strong_Resilience_100",
            "Recovery_Early_Stage_100",
            "Entry_Score_100",
            "ts_code",
        ],
        ascending=[False, True, False, False, False, False, True],
        na_position="last",
        kind="mergesort",
    )
    return candidates.reset_index(drop=True), raw_count, active_eligible_count


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


def track_primary_return_only(
    ts_code: str,
    signal_date: str,
    signal_raw_close: float,
    stock_qfq_dict: dict[str, pd.DataFrame],
    roundtrip_cost_pct: float,
    market_dates,
):
    """全池漏选审计的轻量W3标签；成交规则与完整路径函数保持一致。"""
    result: dict[str, Any] = {
        "Entry_Tradable": False,
        "Entry_Date": None,
        "Outcome_Complete": False,
        PRIMARY_RETURN_COLUMN: np.nan,
    }
    stock = stock_qfq_dict.get(ts_code)
    if stock is None:
        return result
    if market_dates is None:
        future_market_dates = stock.index[stock.index > signal_date].tolist()[:15]
    else:
        future_market_dates = [
            str(item) for item in market_dates if str(item) > signal_date
        ][:15]
    if not future_market_dates:
        return result

    entry_date = future_market_dates[0]
    result["Entry_Date"] = entry_date
    if entry_date not in stock.index:
        return result
    future = stock.reindex(future_market_dates).copy()
    first = future.iloc[0]
    buy_price = _safe_float(first.get("open"))
    if not math.isfinite(buy_price) or buy_price <= 0:
        return result

    raw_buy_price = _safe_float(first.get("raw_open"), buy_price)
    raw_first_high = _safe_float(first.get("raw_high"), _safe_float(first.get("high")))
    raw_first_low = _safe_float(first.get("raw_low"), _safe_float(first.get("low")))
    raw_first_close = _safe_float(first.get("raw_close"), _safe_float(first.get("close")))
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
        return result

    result["Entry_Tradable"] = True
    if len(future) < PRIMARY_HOLD_WEEKS * MARKET_DAYS_PER_WEEK:
        return result
    marked_close = pd.to_numeric(future["close"], errors="coerce").ffill()
    exit_close = _safe_float(marked_close.iloc[14])
    if not math.isfinite(exit_close):
        return result
    primary_return = (
        (exit_close / buy_price - 1.0) * 100.0 - roundtrip_cost_pct
    )
    result[PRIMARY_RETURN_COLUMN] = primary_return
    result["Outcome_Complete"] = True
    return result


def build_major_winner_audit(
    pool: pd.DataFrame,
    candidates: pd.DataFrame,
    signal_date: str,
    stock_qfq_dict: dict[str, pd.DataFrame],
    roundtrip_cost_pct: float,
    market_dates,
):
    """用买入后W3结果反查全池大牛；未来收益只用于审计，绝不进入候选或排名。"""
    if pool.empty:
        return pd.DataFrame()
    candidate_map = {}
    if not candidates.empty and "ts_code" in candidates.columns:
        candidate_map = {
            str(row["ts_code"]): row
            for _, row in candidates.drop_duplicates("ts_code", keep="last").iterrows()
        }
    market_state = _market_state_metrics(pool)
    market_13w_median = _safe_float(
        market_state.get("Market_13W_Median_pct"), 0.0
    )
    previous_market_13w = _safe_float(
        market_state.get("Previous_Market_13W_Median_pct"), 0.0
    )
    previous2_market_13w = _safe_float(
        market_state.get("Previous2_Market_13W_Median_pct"), 0.0
    )
    market_13w_acceleration = _safe_float(
        market_state.get("Market_13W_Acceleration_pct"), 0.0
    )
    market_1w_median = _safe_float(
        market_state.get("Market_1W_Median_pct"), 0.0
    )
    previous_market_1w = _safe_float(
        market_state.get("Previous_Market_1W_Median_pct"), 0.0
    )
    market_positive_breadth = (
        _safe_float(market_state.get("Market_1W_Positive_Breadth"), 0.0) * 100.0
    )
    previous_market_positive_breadth = (
        _safe_float(
            market_state.get("Previous_Market_1W_Positive_Breadth"), 0.0
        )
        * 100.0
    )
    market_regime = str(market_state.get("Market_Regime", "中性"))
    strong_reset_context_pass = bool(
        market_state.get("Strong_Reset_Context_Pass", False)
    )
    strong_continuation_context_pass = bool(
        market_state.get("Strong_Continuation_Context_Pass", False)
    )
    strong_market_context = str(market_state.get("Strong_Market_Context", ""))
    strong_market_stage = str(market_state.get("Strong_Market_Stage", ""))
    strategy_branch = {
        "强势": (
            "R7强势抗跌新高"
            if strong_reset_context_pass
            else "R9强势有序个股再启动"
            if strong_continuation_context_pass
            else "R9强势观察"
        ),
        "弱势": "R6弱势首次转折-N6",
        "中性": "R3中性趋势",
    }[market_regime]
    rows = []
    for _, pool_row in pool.iterrows():
        code = str(pool_row.get("ts_code", ""))
        candidate_row = candidate_map.get(code)
        if candidate_row is not None:
            outcome = candidate_row
        else:
            outcome = track_primary_return_only(
                code,
                signal_date,
                _safe_float(pool_row.get("Raw_Close")),
                stock_qfq_dict,
                roundtrip_cost_pct,
                market_dates,
            )
        if not bool(outcome.get("Outcome_Complete", False)):
            continue
        w3_return = _safe_float(outcome.get(PRIMARY_RETURN_COLUMN))
        if not math.isfinite(w3_return) or w3_return < MAJOR_WINNER_W3_PCT:
            continue
        selected = bool(candidate_row is not None and _bool_series(
            pd.DataFrame([candidate_row]), "Selected_Top2"
        ).iloc[0])
        entry_eligible = bool(candidate_row is not None and _bool_series(
            pd.DataFrame([candidate_row]), "Entry_Eligible"
        ).iloc[0])
        selection_valid = bool(candidate_row is not None and _bool_series(
            pd.DataFrame([candidate_row]), "Selection_Valid"
        ).iloc[0])
        r3_trigger = bool(pool_row.get("R3_Setup_Candidate", False))
        strong_trigger = bool(pool_row.get("Strong_Resilience_Trigger", False))
        reacceleration_trigger = bool(
            pool_row.get("Strong_Reacceleration_Trigger", False)
        )
        recovery_trigger = bool(pool_row.get("Recovery_Structure_Trigger", False))
        r5_baseline_trigger = bool(
            pool_row.get("R5_Baseline_Recovery_Structure_Trigger", False)
        )
        overheated = bool(pool_row.get("Recovery_Overheated", False))
        if selected:
            status = "已实际入选"
        elif entry_eligible and selection_valid:
            status = "合格但排名未入选"
        elif r3_trigger or strong_trigger or reacceleration_trigger or recovery_trigger:
            status = "已发现但被规则拦截"
        else:
            status = "完全未发现"
        if not (r3_trigger or strong_trigger or reacceleration_trigger or recovery_trigger):
            miss_reason = (
                "仅R5旧宽触发，R6首次事件未触发"
                if r5_baseline_trigger
                else "未触发R3首红、R7抗跌新高、R9再启动或R6-N6首次转折事件"
            )
        elif candidate_row is not None:
            miss_reason = str(candidate_row.get("Selection_Block_Reason", "") or "")
            if entry_eligible and selection_valid and not selected:
                is_r9_second = bool(
                    str(candidate_row.get("Strategy_Branch", ""))
                    == "R9强势有序个股再启动"
                    and _safe_float(candidate_row.get("Rank")) == 2.0
                    and not bool(candidate_row.get("R9_Second_Qualified", False))
                )
                miss_reason = (
                    "R9第二名未通过分数或MA20距离独立资格"
                    if is_r9_second
                    else "当周合格但排名未进入最多Top2"
                )
            elif selected:
                miss_reason = "已入选"
            elif not miss_reason:
                miss_reason = "个股结构触发但未通过资格"
        else:
            miss_reason = "未触发R3首红、R7抗跌新高、R9再启动或R6-N6首次转折事件"
        rows.append(
            {
                "Signal_Date": signal_date,
                "Entry_Date": outcome.get("Entry_Date"),
                "ts_code": code,
                "name": pool_row.get("name", code),
                "Industry": pool_row.get("Industry", "未分类"),
                PRIMARY_RETURN_COLUMN: w3_return,
                "Detection_Status": status,
                "Selected_Top2": selected,
                "Entry_Eligible": entry_eligible,
                "Selection_Valid": selection_valid,
                "R3_Setup_Candidate": r3_trigger,
                "Strong_Resilience_Trigger": strong_trigger,
                "Strong_Overheated": bool(pool_row.get("Strong_Overheated", False)),
                "Strong_Reacceleration_Trigger": reacceleration_trigger,
                "Strong_Reacceleration_Overheated": bool(
                    pool_row.get("Strong_Reacceleration_Overheated", False)
                ),
                "Recovery_Structure_Trigger": recovery_trigger,
                "Recovery_Overheated": overheated,
                "R5_Baseline_Recovery_Structure_Trigger": r5_baseline_trigger,
                "Miss_Reason": miss_reason,
                "Rank": candidate_row.get("Rank") if candidate_row is not None else np.nan,
                "Strategy_Branch": (
                    candidate_row.get("Strategy_Branch", "")
                    if candidate_row is not None
                    else strategy_branch
                ),
                "Market_Regime": (
                    candidate_row.get("Market_Regime", "")
                    if candidate_row is not None
                    else market_regime
                ),
                "Market_13W_Median_pct": market_13w_median,
                "Previous_Market_13W_Median_pct": previous_market_13w,
                "Previous2_Market_13W_Median_pct": previous2_market_13w,
                "Market_13W_Acceleration_pct": market_13w_acceleration,
                "Market_1W_Median_pct": market_1w_median,
                "Previous_Market_1W_Median_pct": previous_market_1w,
                "Market_1W_Positive_Breadth_pct": market_positive_breadth,
                "Previous_Market_1W_Positive_Breadth_pct": (
                    previous_market_positive_breadth
                ),
                "Strong_Market_Context": strong_market_context,
                "Strong_Market_Stage": strong_market_stage,
                "Strong_Reset_Context_Pass": strong_reset_context_pass,
                "Strong_Continuation_Context_Pass": (
                    strong_continuation_context_pass
                ),
                "Return_1W_pct": pool_row.get("Return_1W_pct"),
                "Return_13W_pct": pool_row.get("Return_13W_pct"),
                "Drawdown_26W_pct": pool_row.get("Drawdown_26W_pct"),
                "Price_to_MA10_Ratio": pool_row.get("Price_to_MA10_Ratio"),
                "Return_2W_pct": pool_row.get("Return_2W_pct"),
                "Weekly_SKDJ_K6": pool_row.get("Weekly_SKDJ_K6"),
                "Weekly_SKDJ_D6": pool_row.get("Weekly_SKDJ_D6"),
            }
        )
    return pd.DataFrame(rows)


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


def replace_opportunity_date(new_rows: pd.DataFrame, signal_date: str, config_id: str):
    """逐周替换未来W3大牛机会审计；仅保存收益达到20%的小表。"""
    existing = read_csv_safe(OPPORTUNITY_FILE)
    if not existing.empty and {"Signal_Date", "Config_ID"}.issubset(existing.columns):
        normalized_dates = existing["Signal_Date"].map(parse_yyyymmdd)
        existing = existing[
            ~(
                normalized_dates.eq(str(signal_date))
                & existing["Config_ID"].astype(str).eq(str(config_id))
            )
        ].copy()
    if not new_rows.empty:
        combined = (
            pd.concat([existing, new_rows], ignore_index=True, sort=False)
            if not existing.empty
            else new_rows.copy()
        )
    else:
        combined = existing
    if combined.empty:
        remove_with_backup(OPPORTUNITY_FILE)
        return
    combined["Signal_Date"] = combined["Signal_Date"].map(parse_yyyymmdd)
    keys = [
        column
        for column in ("Config_ID", "Signal_Date", "ts_code")
        if column in combined.columns
    ]
    combined = combined.drop_duplicates(keys, keep="last") if keys else combined
    atomic_write_csv(
        combined.sort_values(["Signal_Date", PRIMARY_RETURN_COLUMN], ascending=[True, False])
        .reset_index(drop=True),
        OPPORTUNITY_FILE,
    )


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
        return pd.DataFrame(), pd.DataFrame(), 0, 0
    pool = pd.DataFrame(pool_records)
    candidates, raw_count, eligible_count = score_r9_candidates(pool)

    if is_preview_mode:
        if not candidates.empty:
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
        major_winners = pd.DataFrame()
    else:
        if not candidates.empty:
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
        major_winners = build_major_winner_audit(
            pool,
            candidates,
            signal_date,
            stock_qfq_dict,
            roundtrip_cost_pct,
            market_dates,
        )
    return candidates, major_winners, raw_count, eligible_count


# -----------------------------------------------------------------------------
# R9四分支稳健性报告
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
    entry_eligible = (
        _bool_series(history, "Entry_Eligible")
        if "Entry_Eligible" in history.columns
        else _bool_series(history, "Trend_Eligible")
    )
    valid_selection = _bool_series(history, "Selection_Valid")
    result = history[complete & tradable & entry_eligible & valid_selection].copy()
    result["Rank"] = pd.to_numeric(result.get("Rank"), errors="coerce")
    result[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        result.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    return result.dropna(subset=["Rank", PRIMARY_RETURN_COLUMN])


def _actual_selected_mask(frame: pd.DataFrame):
    """兼容“最多Top2”：R9第二名未达标时不能被Rank<=2误算成实盘入选。"""
    if "Selected_Top2" in frame.columns:
        return _bool_series(frame, "Selected_Top2")
    return pd.to_numeric(frame.get("Rank"), errors="coerce").le(TOP_N)


def cohort_summary(completed: pd.DataFrame):
    actual = _actual_selected_mask(completed)
    rank = pd.to_numeric(completed.get("Rank"), errors="coerce")
    cohorts = [
        ("Top1", completed[actual & rank.eq(1)]),
        ("Top2", completed[actual & rank.eq(2)]),
        ("实际入选合计", completed[actual]),
        ("未入选候选", completed[~actual]),
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
    selected = completed[_actual_selected_mask(completed)].copy()
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
        ("原始实际入选", returns),
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
    frame = completed[_actual_selected_mask(completed)].copy()
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


def strategy_branch_summary(completed: pd.DataFrame):
    if completed.empty or "Strategy_Branch" not in completed.columns:
        return pd.DataFrame()
    rows = []
    selected = completed[_actual_selected_mask(completed)]
    for branch, group in selected.groupby("Strategy_Branch", sort=False):
        returns = pd.to_numeric(group[PRIMARY_RETURN_COLUMN], errors="coerce").dropna()
        rows.append(
            {
                "策略分支": branch,
                "交易数": len(returns),
                "信号周": group.loc[returns.index, "Signal_Date"].nunique(),
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "Profit_Factor": _profit_factor(returns),
            }
        )
    return pd.DataFrame(rows)


def strong_candidate_audit(history: pd.DataFrame):
    """R7抗跌新高分支单独判卷；观察期与过热候选不混入实际Top2。"""
    if history.empty or "Strong_Resilience_Trigger" not in history.columns:
        return pd.DataFrame()
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & _bool_series(history, "Strong_Resilience_Trigger")
    ].copy()
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    frame["Strong_Rank"] = pd.to_numeric(frame.get("Strong_Rank"), errors="coerce")
    actual = (
        _bool_series(frame, "Selected_Top2")
        & frame["Strategy_Branch"].astype(str).eq("R7强势抗跌新高")
    )
    eligible = _bool_series(frame, "Strong_Eligible")
    context_pass = _bool_series(frame, "Strong_Reset_Context_Pass")
    masks = [
        ("实际入选强势Top2", actual),
        ("早期强势回调合格但未入选", eligible & context_pass & ~actual),
        ("其他强势背景合格观察", eligible & ~context_pass),
        ("抗跌新高过热观察", _bool_series(frame, "Strong_Overheated")),
        ("全部抗跌新高结构", pd.Series(True, index=frame.index)),
    ]
    rows = []
    for label, mask in masks:
        group = frame[mask & frame[PRIMARY_RETURN_COLUMN].notna()]
        returns = group[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "强势审计组": label,
                "交易数": len(returns),
                "信号周": group["Signal_Date"].nunique(),
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "Profit_Factor": _profit_factor(returns),
            }
        )
    return pd.DataFrame(rows)


def strong_market_context_audit(history: pd.DataFrame):
    """所有强势周使用同一R7名次做反事实分层，检验市场门是否有效。"""
    if history.empty or "Strong_Rank" not in history.columns:
        return pd.DataFrame()
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & _bool_series(history, "Strong_Eligible")
        & history.get("Market_Regime", pd.Series("", index=history.index)).astype(str).eq(
            "强势"
        )
    ].copy()
    frame["Strong_Rank"] = pd.to_numeric(frame.get("Strong_Rank"), errors="coerce")
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    frame = frame[
        frame["Strong_Rank"].le(TOP_N) & frame[PRIMARY_RETURN_COLUMN].notna()
    ]
    rows = []
    for context, group in frame.groupby("Strong_Market_Context", dropna=False):
        returns = group[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "强势市场背景": context,
                "R7动作": "允许Top2" if str(context) == "早期强势回调" else "只观察",
                "交易数": len(returns),
                "信号周": group["Signal_Date"].nunique(),
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "Profit_Factor": _profit_factor(returns),
            }
        )
    return pd.DataFrame(rows)


def strong_baseline_comparison_audit(history: pd.DataFrame):
    """同一批强势周比较R7抗跌新高与R3首红基线，不改变任何入选结果。"""
    if history.empty:
        return pd.DataFrame()
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & history.get("Market_Regime", pd.Series("", index=history.index)).astype(str).eq(
            "强势"
        )
    ].copy()
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    strong_rank = pd.to_numeric(frame.get("Strong_Rank"), errors="coerce")
    r3_rank = pd.to_numeric(frame.get("R3_Rank"), errors="coerce")
    actual = (
        _bool_series(frame, "Selected_Top2")
        & frame.get("Strategy_Branch", pd.Series("", index=frame.index)).astype(str).eq(
            "R7强势抗跌新高"
        )
    )
    r7_all_contexts = _bool_series(frame, "Strong_Eligible") & strong_rank.le(TOP_N)
    r3_baseline = _bool_series(frame, "Trend_Eligible") & r3_rank.le(TOP_N)
    groups = [
        ("R7实际允许的抗跌新高Top2", actual),
        ("R7若覆盖全部强势背景", r7_all_contexts),
        ("R3旧首红若直接用于强势期", r3_baseline),
    ]
    rows = []
    for label, mask in groups:
        group = frame[mask & frame[PRIMARY_RETURN_COLUMN].notna()]
        returns = group[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "强势方案": label,
                "交易数": len(returns),
                "信号周": group["Signal_Date"].nunique(),
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "Profit_Factor": _profit_factor(returns),
            }
        )
    return pd.DataFrame(rows)


def reacceleration_candidate_audit(history: pd.DataFrame):
    """R9强势个股再启动分支独立判卷，不借用R7早期回调收益。"""
    if history.empty or "Strong_Reacceleration_Trigger" not in history.columns:
        return pd.DataFrame()
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & _bool_series(history, "Strong_Reacceleration_Trigger")
    ].copy()
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    actual = (
        _bool_series(frame, "Selected_Top2")
        & frame.get("Strategy_Branch", pd.Series("", index=frame.index))
        .astype(str)
        .eq("R9强势有序个股再启动")
    )
    eligible = _bool_series(frame, "Strong_Reacceleration_Eligible")
    context_pass = _bool_series(frame, "Strong_Continuation_Context_Pass")
    resilience_pass = _bool_series(
        frame, "Strong_Reacceleration_Resilience_Pass"
    )
    risk_ok = _bool_series(frame, "Strong_Reacceleration_Risk_OK")
    rank = pd.to_numeric(frame.get("Strong_Reacceleration_Rank"), errors="coerce")
    second_rejected = (
        eligible
        & context_pass
        & rank.eq(2)
        & ~_bool_series(frame, "R9_Second_Qualified")
    )
    masks = [
        ("实际入选R9最多Top2", actual),
        ("R9阶段内合格但未入选", eligible & context_pass & ~actual),
        ("R9第二名未通过独立资格", second_rejected),
        ("相同个股事件但R9市场阶段未通过", eligible & ~context_pass),
        ("R8旧抗跌条件通过", risk_ok & resilience_pass),
        ("R8旧抗跌条件未通过", risk_ok & ~resilience_pass),
        ("整理后再启动过热观察", _bool_series(frame, "Strong_Reacceleration_Overheated")),
        ("全部整理后再启动结构", pd.Series(True, index=frame.index)),
    ]
    rows = []
    for label, mask in masks:
        group = frame[mask & frame[PRIMARY_RETURN_COLUMN].notna()]
        returns = group[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "R9再启动审计组": label,
                "交易数": len(returns),
                "信号周": group["Signal_Date"].nunique(),
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "Profit_Factor": _profit_factor(returns),
            }
        )
    return pd.DataFrame(rows)


def reexpansion_comparison_audit(history: pd.DataFrame):
    """在同一R9强势阶段比较实际、强制Top2、R7和R3。"""
    if history.empty:
        return pd.DataFrame()
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & history.get("Market_Regime", pd.Series("", index=history.index))
        .astype(str)
        .eq("强势")
    ].copy()
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    reacceleration_rank = pd.to_numeric(
        frame.get("Strong_Reacceleration_Rank"), errors="coerce"
    )
    strong_rank = pd.to_numeric(frame.get("Strong_Rank"), errors="coerce")
    r3_rank = pd.to_numeric(frame.get("R3_Rank"), errors="coerce")
    context_pass = _bool_series(frame, "Strong_Continuation_Context_Pass")
    actual = (
        _bool_series(frame, "Selected_Top2")
        & frame.get("Strategy_Branch", pd.Series("", index=frame.index))
        .astype(str)
        .eq("R9强势有序个股再启动")
    )
    r9_eligible = _bool_series(frame, "Strong_Reacceleration_Eligible")
    groups = [
        ("R9实际最多Top2", actual),
        ("R9实际第一名", actual & reacceleration_rank.eq(1)),
        ("R9通过独立资格的第二名", actual & reacceleration_rank.eq(2)),
        (
            "R9若强制选满Top2",
            context_pass & r9_eligible & reacceleration_rank.le(TOP_N),
        ),
        (
            "同一R9阶段内R7抗跌新高Top2",
            context_pass
            & _bool_series(frame, "Strong_Eligible")
            & strong_rank.le(TOP_N),
        ),
        (
            "同一R9阶段内R3首红Top2",
            context_pass
            & _bool_series(frame, "Trend_Eligible")
            & r3_rank.le(TOP_N),
        ),
    ]
    rows = []
    for label, mask in groups:
        group = frame[mask & frame[PRIMARY_RETURN_COLUMN].notna()]
        returns = group[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "强势扩张方案": label,
                "交易数": len(returns),
                "信号周": group["Signal_Date"].nunique(),
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "Profit_Factor": _profit_factor(returns),
            }
        )
    return pd.DataFrame(rows)


def strong_stage_week_audit(history: pd.DataFrame):
    """逐个市场阶段核对周数、候选数与真实入选，防止观察样本被当作交易。"""
    if history.empty or "Strong_Market_Stage" not in history.columns:
        return pd.DataFrame()
    frame = history[
        history.get("Market_Regime", pd.Series("", index=history.index))
        .astype(str)
        .eq("强势")
    ].copy()
    rows = []
    for stage, group in frame.groupby("Strong_Market_Stage", dropna=False, sort=False):
        action = (
            "R7抗跌新高Top2"
            if str(stage) == "早期强势回调-R7"
            else "R9整理后再启动最多Top2"
            if str(stage) == "强势有序扩张-R9"
            else "只观察"
        )
        rows.append(
            {
                "强势市场阶段": stage,
                "R9动作": action,
                "市场周数": group["Signal_Date"].nunique(),
                "R7合格股票周数": int(_bool_series(group, "Strong_Eligible").sum()),
                "R9合格股票周数": int(
                    _bool_series(group, "Strong_Reacceleration_Eligible").sum()
                ),
                "实际入选交易": int(_bool_series(group, "Selected_Top2").sum()),
                "实际入选周": group.loc[
                    _bool_series(group, "Selected_Top2"), "Signal_Date"
                ].nunique(),
            }
        )
    return pd.DataFrame(rows)


def recovery_candidate_audit(history: pd.DataFrame):
    """R6首次转折分支单独判卷；过热观察不混入实际Top2收益。"""
    if history.empty or "Recovery_Structure_Trigger" not in history.columns:
        return pd.DataFrame()
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & _bool_series(history, "Recovery_Structure_Trigger")
    ].copy()
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    frame["Recovery_Rank"] = pd.to_numeric(
        frame.get("Recovery_Rank"), errors="coerce"
    )
    masks = [
        (
            "实际入选复苏Top2",
            _bool_series(frame, "Selected_Top2")
            & frame["Strategy_Branch"].astype(str).eq("R6弱势首次转折-N6"),
        ),
        (
            "合格但未入选",
            _bool_series(frame, "Recovery_Eligible")
            & ~_bool_series(frame, "Selected_Top2"),
        ),
        ("单周过热观察", _bool_series(frame, "Recovery_Overheated")),
        ("全部复苏结构", pd.Series(True, index=frame.index)),
    ]
    rows = []
    for label, mask in masks:
        group = frame[mask & frame[PRIMARY_RETURN_COLUMN].notna()]
        returns = group[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "复苏审计组": label,
                "交易数": len(returns),
                "信号周": group["Signal_Date"].nunique(),
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "Profit_Factor": _profit_factor(returns),
            }
        )
    return pd.DataFrame(rows)


def recovery_trigger_comparison_audit(history: pd.DataFrame):
    """同一批未来路径比较R6首次事件与R5旧宽触发，不改变任何入选结果。"""
    if history.empty:
        return pd.DataFrame()
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
    ].copy()
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    r6 = _bool_series(frame, "Recovery_Structure_Trigger")
    r5 = _bool_series(frame, "R5_Baseline_Recovery_Structure_Trigger")
    groups = [
        ("R6首次转折事件", r6),
        ("R5旧宽触发", r5),
        ("R5与R6重合", r5 & r6),
        ("仅R6新增", r6 & ~r5),
        ("仅R5旧规则", r5 & ~r6),
    ]
    rows = []
    for label, mask in groups:
        group = frame[mask & frame[PRIMARY_RETURN_COLUMN].notna()]
        returns = group[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "触发口径": label,
                "交易数": len(returns),
                "信号周": group["Signal_Date"].nunique(),
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "Profit_Factor": _profit_factor(returns),
            }
        )
    return pd.DataFrame(rows)


def recovery_market_context_audit(history: pd.DataFrame):
    """市场广度只做事后分层，检查旧55%门究竟帮助还是伤害。"""
    if history.empty:
        return pd.DataFrame()
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & history.get("Strategy_Branch", pd.Series("", index=history.index)).astype(str).eq(
            "R6弱势首次转折-N6"
        )
        & _bool_series(history, "Selected_Top2")
    ].copy()
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    if frame.empty:
        return pd.DataFrame()
    rows = []
    context_groups = [
        (f"市场背景：{label}", group)
        for label, group in frame.groupby("Market_Recovery_Context", dropna=False)
    ]
    gate_groups = [
        (
            "R5旧市场门：通过" if passed else "R5旧市场门：未通过",
            group,
        )
        for passed, group in frame.groupby(
            _bool_series(frame, "R5_Baseline_Market_Gate_Pass"), dropna=False
        )
    ]
    for label, group in context_groups + gate_groups:
        returns = group[PRIMARY_RETURN_COLUMN].dropna()
        rows.append(
            {
                "市场分层": label,
                "交易数": len(returns),
                "信号周": group.loc[returns.index, "Signal_Date"].nunique() if len(returns) else 0,
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "Profit_Factor": _profit_factor(returns),
            }
        )
    return pd.DataFrame(rows)


def major_winner_coverage_summary(opportunities: pd.DataFrame):
    """统计未来W3大涨机会的发现率；只用于事后漏选审计。"""
    columns = ["机会口径", "股票周数", "涉及周数", "占全部机会%", "W3中位收益%", "W3平均收益%"]
    if opportunities.empty:
        return pd.DataFrame(columns=columns)
    frame = opportunities.copy()
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    frame = frame[frame[PRIMARY_RETURN_COLUMN].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    total = len(frame)
    detected = frame[frame["Detection_Status"].astype(str).ne("完全未发现")]
    eligible = frame[
        _bool_series(frame, "Entry_Eligible")
        & _bool_series(frame, "Selection_Valid")
    ]
    selected = frame[_bool_series(frame, "Selected_Top2")]
    missed = frame[frame["Detection_Status"].astype(str).eq("完全未发现")]
    groups = [
        (f"全部未来W3≥{MAJOR_WINNER_W3_PCT:.0f}%机会", frame),
        ("任一结构已发现", detected),
        ("当周分支合格", eligible),
        ("最终实际入选", selected),
        ("完全未发现", missed),
    ]
    rows = []
    for label, group in groups:
        values = pd.to_numeric(group[PRIMARY_RETURN_COLUMN], errors="coerce").dropna()
        rows.append(
            {
                "机会口径": label,
                "股票周数": len(values),
                "涉及周数": group.loc[values.index, "Signal_Date"].nunique() if len(values) else 0,
                "占全部机会%": len(values) / total * 100.0,
                "W3中位收益%": values.median() if len(values) else np.nan,
                "W3平均收益%": values.mean() if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def regime_gate_summary(history: pd.DataFrame):
    """只统计真实入选；强势观察样本另表展示，避免把反事实写成实际交易。"""
    if history.empty:
        return pd.DataFrame()
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & _bool_series(history, "Selected_Top2")
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
        if str(regime) == "中性":
            action = "R3允许Top2"
        elif str(regime) == "弱势":
            action = "R6首次转折Top2；市场仅分层不否决"
        else:
            action = "R7早期回调Top2或R9有序再启动最多Top2"
        rows.append(
            {
                "市场状态": regime,
                "R9动作": action,
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
    selected = completed[_actual_selected_mask(completed)].copy()
    for signal_date, group in selected.groupby("Signal_Date"):
        exact = group.sort_values("Rank").drop_duplicates("Rank")
        returns = exact[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "Signal_Date": signal_date,
                "入选只数": len(returns),
                "盈利只数": int((returns > 0).sum()),
                "入选平均收益%": returns.mean(),
                "入选中位收益%": returns.median(),
                "入选最差收益%": returns.min(),
                "入选最佳收益%": returns.max(),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, {}
    summary = {
        "完整入选组": len(detail),
        "两股组": int(detail["入选只数"].eq(2).sum()),
        "单股组": int(detail["入选只数"].eq(1).sum()),
        "组平均收益为正比例%": detail["入选平均收益%"].gt(0).mean() * 100.0,
        "组平均收益中位数%": detail["入选平均收益%"].median(),
    }
    return detail, summary


def horizon_summary(completed: pd.DataFrame):
    selected = completed[_actual_selected_mask(completed)].copy()
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
    actual = _actual_selected_mask(completed)
    top2 = returns[actual]
    rest = returns[~actual]
    paired_advantages = []
    for _, group in completed.groupby("Signal_Date"):
        selected_returns = pd.to_numeric(
            group.loc[_actual_selected_mask(group), PRIMARY_RETURN_COLUMN], errors="coerce"
        ).dropna()
        other_returns = pd.to_numeric(
            group.loc[~_actual_selected_mask(group), PRIMARY_RETURN_COLUMN], errors="coerce"
        ).dropna()
        if len(selected_returns) and len(other_returns):
            paired_advantages.append(selected_returns.mean() - other_returns.mean())

    selected = completed[actual].copy()
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
        "实际入选中位收益%": top2.median() if len(top2) else np.nan,
        "未入选候选中位收益%": rest.median() if len(rest) else np.nan,
        "实际入选相对未入选中位优势百分点": (
            top2.median() - rest.median() if len(top2) and len(rest) else np.nan
        ),
        "实际入选逐周平均收益战胜未入选比例%": (
            np.mean(np.asarray(paired_advantages) > 0.0) * 100.0
            if paired_advantages
            else np.nan
        ),
        "实际入选逐周平均收益优势均值百分点": (
            np.mean(paired_advantages) if paired_advantages else np.nan
        ),
        "前半段实际入选中位收益%": first_half.median() if len(first_half) else np.nan,
        "后半段实际入选中位收益%": second_half.median() if len(second_half) else np.nan,
    }


def research_gates(
    completed: pd.DataFrame,
    cohort: pd.DataFrame,
    outlier: pd.DataFrame,
    diagnostics: dict[str, Any],
):
    selected = completed[_actual_selected_mask(completed)]
    returns = selected[PRIMARY_RETURN_COLUMN] if not selected.empty else pd.Series(dtype=float)
    weeks = selected["Signal_Date"].nunique() if not selected.empty else 0
    wins = int((returns > 0).sum()) if len(returns) else 0
    lower_bound = _wilson_lower_bound(wins, len(returns)) * 100.0 if len(returns) else np.nan
    rest = completed[~_actual_selected_mask(completed)][PRIMARY_RETURN_COLUMN]
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
    paired_beat = _safe_float(diagnostics.get("实际入选逐周平均收益战胜未入选比例%"))
    first_half_median = _safe_float(diagnostics.get("前半段实际入选中位收益%"))
    second_half_median = _safe_float(diagnostics.get("后半段实际入选中位收益%"))
    gates = [
        ("一年内至少18个独立四分支信号周", weeks >= 18, f"当前{weeks}周"),
        ("至少36笔完整实际入选交易", len(returns) >= 36, f"当前{len(returns)}笔"),
        ("W3实际入选胜率至少55%", len(returns) > 0 and (returns > 0).mean() >= 0.55, f"当前{((returns > 0).mean() * 100.0 if len(returns) else np.nan):.1f}%"),
        ("胜率95%下限高于50%", math.isfinite(lower_bound) and lower_bound > 50.0, f"当前{lower_bound:.1f}%"),
        ("W3实际入选中位收益大于0", len(returns) > 0 and returns.median() > 0, f"当前{(returns.median() if len(returns) else np.nan):.2f}%"),
        ("去掉最高5%后平均收益大于0", math.isfinite(remove5_mean) and remove5_mean > 0, f"当前{remove5_mean:.2f}%"),
        ("去掉最高5%后中位收益大于0", math.isfinite(remove5_median) and remove5_median > 0, f"当前{remove5_median:.2f}%"),
        ("去掉最高5%后PF至少1.2", not math.isnan(remove5_pf) and remove5_pf >= 1.2, f"当前{remove5_pf:.2f}"),
        ("Top1和Top2中位收益分别为正", all_rank_medians_positive, "逐名检查"),
        ("Top1中位收益不低于Top2", rank_ordered, "Top1≥Top2"),
        ("逐周实际排名收益秩相关至少0.05", math.isfinite(weekly_corr) and weekly_corr >= 0.05, f"当前{weekly_corr:.3f}"),
        ("实际入选逐周战胜未入选候选至少55%", math.isfinite(paired_beat) and paired_beat >= 55.0, f"当前{paired_beat:.1f}%"),
        ("前后半段实际入选中位收益均为正", math.isfinite(first_half_median) and math.isfinite(second_half_median) and first_half_median > 0 and second_half_median > 0, f"前{first_half_median:.2f}% / 后{second_half_median:.2f}%"),
        ("实际入选中位收益优于未入选候选", len(returns) > 0 and len(rest) > 0 and returns.median() > rest.median(), f"差{(returns.median() - rest.median() if len(returns) and len(rest) else np.nan):.2f}个百分点"),
    ]
    return pd.DataFrame(
        [{"验收项目": name, "结果": "通过" if passed else "未通过", "当前值": value} for name, passed, value in gates]
    )


def strong_research_gates(completed: pd.DataFrame):
    """R7强势抗跌新高模块必须独立过关，不能借用R3/R6结果。"""
    if completed.empty or "Strategy_Branch" not in completed.columns:
        selected = pd.DataFrame()
    else:
        selected = completed[
            completed["Strategy_Branch"].astype(str).eq("R7强势抗跌新高")
            & _actual_selected_mask(completed)
        ].copy()
    returns = pd.to_numeric(
        selected.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    weeks = selected.loc[returns.index, "Signal_Date"].nunique() if len(returns) else 0
    pf = _profit_factor(returns)
    without_best = (
        returns.drop(index=returns.idxmax())
        if len(returns) > 1
        else pd.Series(dtype=float)
    )
    gates = [
        ("强势分支至少6个独立信号周", weeks >= 6, f"当前{weeks}周"),
        ("强势分支至少12笔完整交易", len(returns) >= 12, f"当前{len(returns)}笔"),
        (
            "强势分支W3胜率至少55%",
            len(returns) > 0 and (returns > 0).mean() >= 0.55,
            f"当前{((returns > 0).mean() * 100.0 if len(returns) else np.nan):.1f}%",
        ),
        (
            "强势分支W3中位收益大于0",
            len(returns) > 0 and returns.median() > 0.0,
            f"当前{(returns.median() if len(returns) else np.nan):.2f}%",
        ),
        (
            "强势分支去最佳一只后平均收益大于0",
            len(without_best) > 0 and without_best.mean() > 0.0,
            f"当前{(without_best.mean() if len(without_best) else np.nan):.2f}%",
        ),
        (
            "强势分支PF至少1.2",
            pd.notna(pf) and pf >= 1.2,
            f"当前{pf:.2f}",
        ),
    ]
    return pd.DataFrame(
        [
            {"强势验收项目": name, "结果": "通过" if passed else "未通过", "当前值": value}
            for name, passed, value in gates
        ]
    )


def reacceleration_ranking_diagnostics(history: pd.DataFrame):
    if history.empty:
        return {}
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & _bool_series(history, "Strong_Reacceleration_Eligible")
        & _bool_series(history, "Strong_Continuation_Context_Pass")
    ].copy()
    if frame.empty:
        return {}
    frame["Rank"] = pd.to_numeric(
        frame.get("Strong_Reacceleration_Rank"), errors="coerce"
    )
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    frame = frame.dropna(subset=["Rank", PRIMARY_RETURN_COLUMN])
    return ranking_diagnostics(frame)


def reacceleration_research_gates(
    completed: pd.DataFrame,
    history: pd.DataFrame,
    diagnostics: dict[str, Any],
):
    """R9强势再启动分支独立过关；不能与其他分支合并凑门槛。"""
    if completed.empty or "Strategy_Branch" not in completed.columns:
        selected = pd.DataFrame()
    else:
        selected = completed[
            completed["Strategy_Branch"].astype(str).eq("R9强势有序个股再启动")
            & _bool_series(completed, "Selected_Top2")
        ].copy()
    returns = pd.to_numeric(
        selected.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    weeks = selected.loc[returns.index, "Signal_Date"].nunique() if len(returns) else 0
    pf = _profit_factor(returns)
    without_best = (
        returns.drop(index=returns.idxmax())
        if len(returns) > 1
        else pd.Series(dtype=float)
    )
    without_best_week = pd.Series(dtype=float)
    if len(returns) and weeks > 1:
        valid = selected.loc[returns.index].copy()
        valid[PRIMARY_RETURN_COLUMN] = returns
        best_week = (
            valid.groupby("Signal_Date")[PRIMARY_RETURN_COLUMN].mean().idxmax()
        )
        without_best_week = valid.loc[
            valid["Signal_Date"] != best_week, PRIMARY_RETURN_COLUMN
        ]
    without_best_week_pf = _profit_factor(without_best_week)
    weekly_corr = _safe_float(diagnostics.get("逐周秩相关均值"))
    paired_beat = _safe_float(
        diagnostics.get("实际入选逐周平均收益战胜未入选比例%")
    )
    top2_median = _safe_float(diagnostics.get("实际入选中位收益%"))
    rest_median = _safe_float(diagnostics.get("未入选候选中位收益%"))
    gates = [
        ("R9再启动至少6个独立信号周", weeks >= 6, f"当前{weeks}周"),
        ("R9再启动至少12笔完整交易", len(returns) >= 12, f"当前{len(returns)}笔"),
        (
            "R9再启动W3胜率至少55%",
            len(returns) > 0 and (returns > 0).mean() >= 0.55,
            f"当前{((returns > 0).mean() * 100.0 if len(returns) else np.nan):.1f}%",
        ),
        (
            "R9再启动W3中位收益大于0",
            len(returns) > 0 and returns.median() > 0.0,
            f"当前{(returns.median() if len(returns) else np.nan):.2f}%",
        ),
        (
            "R9去最佳一只后平均收益大于0",
            len(without_best) > 0 and without_best.mean() > 0.0,
            f"当前{(without_best.mean() if len(without_best) else np.nan):.2f}%",
        ),
        (
            "R9去最佳整周后平均收益大于0且PF至少1.2",
            len(without_best_week) > 0
            and without_best_week.mean() > 0.0
            and pd.notna(without_best_week_pf)
            and without_best_week_pf >= 1.2,
            (
                f"均益{(without_best_week.mean() if len(without_best_week) else np.nan):.2f}% / "
                f"PF{without_best_week_pf:.2f}"
            ),
        ),
        ("R9再启动PF至少1.2", pd.notna(pf) and pf >= 1.2, f"当前{pf:.2f}"),
        (
            "R9逐周排名收益秩相关至少0.05",
            math.isfinite(weekly_corr) and weekly_corr >= 0.05,
            f"当前{weekly_corr:.3f}",
        ),
        (
            "R9实际入选逐周战胜其余候选至少55%",
            math.isfinite(paired_beat) and paired_beat >= 55.0,
            f"当前{paired_beat:.1f}%",
        ),
        (
            "R9实际入选中位收益优于其余候选",
            math.isfinite(top2_median)
            and math.isfinite(rest_median)
            and top2_median > rest_median,
            f"差{(top2_median - rest_median):.2f}个百分点",
        ),
    ]
    return pd.DataFrame(
        [
            {"R9验收项目": name, "结果": "通过" if passed else "未通过", "当前值": value}
            for name, passed, value in gates
        ]
    )


def recovery_research_gates(completed: pd.DataFrame):
    """R6首次转折模块必须单独过关；总体结果不能替它掩盖样本不足。"""
    if completed.empty or "Strategy_Branch" not in completed.columns:
        selected = pd.DataFrame()
    else:
        selected = completed[
            completed["Strategy_Branch"].astype(str).eq("R6弱势首次转折-N6")
            & _actual_selected_mask(completed)
        ].copy()
    returns = pd.to_numeric(
        selected.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    weeks = selected.loc[returns.index, "Signal_Date"].nunique() if len(returns) else 0
    pf = _profit_factor(returns)
    gates = [
        ("复苏分支至少6个独立信号周", weeks >= 6, f"当前{weeks}周"),
        ("复苏分支至少12笔完整交易", len(returns) >= 12, f"当前{len(returns)}笔"),
        (
            "复苏分支W3胜率至少50%",
            len(returns) > 0 and (returns > 0).mean() >= 0.50,
            f"当前{((returns > 0).mean() * 100.0 if len(returns) else np.nan):.1f}%",
        ),
        (
            "复苏分支W3中位收益大于0",
            len(returns) > 0 and returns.median() > 0.0,
            f"当前{(returns.median() if len(returns) else np.nan):.2f}%",
        ),
        (
            "复苏分支PF至少1.2",
            pd.notna(pf) and pf >= 1.2,
            f"当前{pf:.2f}",
        ),
    ]
    return pd.DataFrame(
        [
            {"复苏验收项目": name, "结果": "通过" if passed else "未通过", "当前值": value}
            for name, passed, value in gates
        ]
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
    branches: pd.DataFrame,
    strong_audit: pd.DataFrame,
    strong_gates: pd.DataFrame,
    strong_context_audit: pd.DataFrame,
    strong_baseline_comparison: pd.DataFrame,
    reacceleration_audit: pd.DataFrame,
    reacceleration_gates: pd.DataFrame,
    reacceleration_diagnostics: dict[str, Any],
    reexpansion_comparison: pd.DataFrame,
    strong_stage_audit: pd.DataFrame,
    recovery_audit: pd.DataFrame,
    recovery_gates: pd.DataFrame,
    opportunities: pd.DataFrame,
    opportunity_summary: pd.DataFrame,
    trigger_comparison: pd.DataFrame,
    market_context_audit: pd.DataFrame,
):
    buffer = io.BytesIO()
    files = {
        "01_all_r9_four_branch_candidates.csv": history,
        "02_rank_cohort_summary.csv": cohort,
        "03_outlier_dependency_audit.csv": outlier,
        "04_year_summary.csv": yearly,
        "05_complete_selection_groups.csv": groups,
        "06_w1_w8_fixed_horizon.csv": horizon,
        "07_research_acceptance_gates.csv": gates,
        "08_ranking_diagnostics.csv": pd.DataFrame(
            [{"指标": key, "数值": value} for key, value in diagnostics.items()]
        ),
        "09_market_regime_gate_audit.csv": regimes,
        "10_strategy_branch_summary.csv": branches,
        "11_r7_reset_candidate_audit.csv": strong_audit,
        "12_r7_reset_acceptance_gates.csv": strong_gates,
        "13_r7_all_strong_context_audit.csv": strong_context_audit,
        "14_r7_r3_strong_baseline_comparison.csv": strong_baseline_comparison,
        "15_r9_restart_candidate_audit.csv": reacceleration_audit,
        "16_r9_restart_acceptance_gates.csv": reacceleration_gates,
        "17_r9_restart_ranking_diagnostics.csv": pd.DataFrame(
            [{"指标": key, "数值": value} for key, value in reacceleration_diagnostics.items()]
        ),
        "18_r9_restart_alternative_comparison.csv": reexpansion_comparison,
        "19_strong_market_stage_week_audit.csv": strong_stage_audit,
        "20_recovery_candidate_audit.csv": recovery_audit,
        "21_recovery_acceptance_gates.csv": recovery_gates,
        "22_w3_major_winner_opportunities.csv": opportunities,
        "23_major_winner_coverage_summary.csv": opportunity_summary,
        "24_r5_r6_trigger_comparison.csv": trigger_comparison,
        "25_recovery_market_context_audit.csv": market_context_audit,
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
        "R3、R6与R7逐行冻结；R9只重做强势扩张期的个股再启动分支。"
        "四个分支互斥启用、分别排名和独立验收；R9最多选2只，W3主验收，W1—W8固定记录。"
    )
    st.caption(f"运行引擎修订：{ENGINE_PATCH}")
    st.warning(
        "R9仍是研究验证版；R7早期回调、R9强势再启动与R6弱势转折都必须独立过关。"
    )
    with st.expander("查看R9四分支规则边界"):
        st.markdown(
            """
- **R3中性趋势**：原MACD首红、MA20资格和趋势/风险/总分词典序完全保留。
- **R6弱势首次转折**：26周深跌且N6 SKDJ近期进入35以下，K首次转升，并出现周涨或强收之一；此前两周已有同类转折则不重复触发。
- **R7强势抗跌新高**：科技池13周中位涨幅位于+5%至+10%，当周中位收益不高于0且上涨家数低于55%时，寻找趋势完整、逆势首次创13周新高且不过热的股票。
- **R7实际排名**：前四周停顿、ATR收缩、本周不过度上涨、行业相对强度、距离MA20五项等权；分数只决定强势候选内部名次。
- **R9宽市场阶段**：R7早期回调优先；其余强势周只要求科技池13周中位涨幅位于+5%至+30%，本周中位涨幅在0%至+6%，上涨家数在50%至90%。不再要求前两周同步、上一周全市场回调或13周动量只能增加0至5个百分点。
- **R9个股事件**：趋势完整，上一周涨幅位于-8%至+5%，本周首次收复上周高点、站上MA10、MACD柱增强且强收盘；删除“上一周必须强于市场中位数”的硬条件。
- **R9实际排名**：上周涨幅绝对值越小越优、13周相对强度、行业13周超额、ATR收缩、本周不过度上涨五项等权。
- **指标口径**：SKDJ固定N=6、M=3；Raw RSV两次EMA(span=3)得到K，D为K的3周简单均线。
- **删除硬门**：不再要求价格达到MA10的75%，不再用1周中位涨幅和55%上涨家数整周归零；市场只做分层审计。
- **实际排名**：两周涨幅、价格/MA10、K6、8周相对强度、MACD冲量五项均按越早越优等权；R5旧100分只保留对照。
- **防追高**：单周涨幅超过25%或收盘距离本周低点超过40%时只进入过热观察，不参与排名。
- **分支隔离**：中性期只运行R3；弱势期只运行R6-N6；早期强势回调运行R7；其余符合宽阶段的强势扩张周运行R9。
- **组合**：R3/R6/R7仍选Top2；R9第一名入选，第二名还必须达到60分且距离MA20不超过18%，因此R9每周可能只选1只，不再强行凑满。
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
            ["历史R9四分支验证", "最新选股预览"],
            index=0,
            help="历史模式只使用完整周线；最新预览允许使用本周未完成周线且不写入回测。",
        )
        start_input = st.date_input("验证开始日期", value=default_start, disabled=mode != "历史R9四分支验证")
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
            help="R9四个分支的W1—W8固定周收益都会扣除该成本。",
        )

        st.markdown("---")
        try:
            secret_token = st.secrets.get("TUSHARE_TOKEN", "")
        except Exception:
            secret_token = ""
        token_input = st.text_input("Tushare Token", value=secret_token, type="password")

        st.markdown("---")
        clear_market_clicked = st.button("清空行情缓存")
        clear_history_clicked = st.button("清除R9历史结果")

    if max_mv <= min_mv:
        st.error("最高流通市值必须大于最低流通市值。")
        return
    if start_input > end_input and mode == "历史R9四分支验证":
        st.error("验证开始日期不能晚于截止日期。")
        return

    if clear_market_clicked:
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.success("行情缓存已清空。")

    if clear_history_clicked:
        for path in (CHECKPOINT_FILE, SCAN_LEDGER_FILE, RUN_TASK_FILE, OPPORTUNITY_FILE):
            remove_with_backup(path)
        st.session_state.pop("r9_preview", None)
        st.success("R9历史结果和断点任务已清除。")

    token_clean = clean_token_str(token_input)
    config_id = make_config_id(min_price, min_mv, max_mv, roundtrip_cost_pct)
    is_preview_mode = mode == "最新选股预览"
    if "r9_worker_id" not in st.session_state:
        st.session_state["r9_worker_id"] = uuid.uuid4().hex
    worker_id = str(st.session_state["r9_worker_id"])
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

    start_label = "运行最新选股预览" if is_preview_mode else "启动历史R9四分支验证"
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
                    progress = st.progress(0, text="开始扫描R9四分支候选……")
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
                        candidates, major_winners, raw_count, eligible_count = scan_one_date(
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
                            st.session_state["r9_preview"] = candidates
                        else:
                            if not candidates.empty:
                                candidates["Config_ID"] = run_config_id
                            if not major_winners.empty:
                                major_winners["Config_ID"] = run_config_id
                            if not refresh_task_lease(
                                str(active_task.get("Task_ID", "")), worker_id
                            ):
                                raise RuntimeError("任务租约已经转移，本页停止写入回测断点。")
                            replace_checkpoint_date(candidates, signal_date, run_config_id)
                            replace_opportunity_date(
                                major_winners, signal_date, run_config_id
                            )
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
                                f"{signal_date}：R9/R7/R3/R6结构{raw_count}只，"
                                f"当前分支合格{eligible_count}只，入选{selected_count}只"
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
                            st.success("历史R9四分支扫描完成。")
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

    preview = st.session_state.get("r9_preview")
    if is_preview_mode and isinstance(preview, pd.DataFrame):
        st.markdown("---")
        st.header("最新选股预览")
        if preview.empty:
            st.info("最新交易日没有R9四分支结构候选。")
        else:
            selected_preview = preview[_bool_series(preview, "Selected_Top2")].copy()
            if selected_preview.empty:
                block_reason = ""
                if "Selection_Block_Reason" in preview.columns and not preview.empty:
                    block_reason = str(preview["Selection_Block_Reason"].iloc[0] or "")
                st.warning(block_reason or "本周没有形成有效入选组。")
            else:
                preview_columns = [
                    "Signal_Date", "Weekly_Data_Mode", "Rank", "name", "ts_code", "Industry",
                    "Strategy_Branch", "R3_Setup_Type", "Strong_Setup_Type",
                    "Strong_Reacceleration_Setup_Type", "Recovery_Setup_Type",
                    "Strong_Reacceleration_100",
                    "Reaccel_Score_PauseControl20", "Reaccel_Score_RS13_20",
                    "Reaccel_Score_Industry20", "Reaccel_Score_ATR20",
                    "Reaccel_Score_NonChase20", "R9_Second_Qualified",
                    "Strong_Resilience_100", "Strong_Score_Pause20",
                    "Strong_Score_ATR20", "Strong_Score_NonChase20",
                    "Strong_Score_Industry20", "Strong_Score_Position20",
                    "Recovery_Early_Stage_100", "Recovery_Score_100",
                    "Weekly_SKDJ_K6", "Weekly_SKDJ_D6",
                    "Drawdown_26W_pct", "Price_to_MA10_Ratio", "Return_1W_pct",
                    "Rebound_From_Week_Low_pct",
                    "Score_Trend_20", "Score_Risk_10",
                    "Entry_Score_100", "Score_Pullback_15", "Score_Contraction_15",
                    "Score_Restart_15", "Score_RS_25",
                    "Raw_Close", "Circ_MV_Billion", "Market_Regime",
                    "Strong_Market_Context", "Strong_Market_Stage",
                    "Market_1W_Median_pct", "Previous_Market_1W_Median_pct",
                    "Market_1W_Positive_Breadth_pct",
                    "Previous_Market_1W_Positive_Breadth_pct",
                    "Selection_Block_Reason",
                ]
                st.dataframe(
                    selected_preview[[column for column in preview_columns if column in selected_preview.columns]],
                    width="stretch",
                )
            with st.expander("查看全部四分支候选、过热观察及未入选原因"):
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

        opportunities = read_csv_safe(OPPORTUNITY_FILE)
        if not opportunities.empty:
            opportunities["Signal_Date"] = opportunities["Signal_Date"].map(parse_yyyymmdd)
            if "Config_ID" in opportunities.columns:
                opportunities = opportunities[
                    opportunities["Config_ID"].astype(str).eq(report_config_id)
                ].copy()
            opportunities[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
                opportunities.get(PRIMARY_RETURN_COLUMN), errors="coerce"
            )
            opportunities = opportunities.sort_values(
                ["Signal_Date", PRIMARY_RETURN_COLUMN],
                ascending=[False, False],
                kind="mergesort",
            )

        completed = completed_research_rows(history)
        cohort = cohort_summary(completed)
        outlier, outlier_details = outlier_audit(completed)
        yearly = year_summary(completed)
        regimes = regime_gate_summary(history)
        branches = strategy_branch_summary(completed)
        strong_audit = strong_candidate_audit(history)
        strong_context_audit = strong_market_context_audit(history)
        strong_baseline_comparison = strong_baseline_comparison_audit(history)
        reacceleration_audit = reacceleration_candidate_audit(history)
        reexpansion_comparison = reexpansion_comparison_audit(history)
        strong_stage_audit = strong_stage_week_audit(history)
        recovery_audit = recovery_candidate_audit(history)
        group_detail, group_stats = two_stock_group_summary(completed)
        horizons = horizon_summary(completed)
        diagnostics = ranking_diagnostics(completed)
        gates = research_gates(completed, cohort, outlier, diagnostics)
        strong_gates = strong_research_gates(completed)
        reacceleration_diagnostics = reacceleration_ranking_diagnostics(history)
        reacceleration_gates = reacceleration_research_gates(
            completed, history, reacceleration_diagnostics
        )
        recovery_gates = recovery_research_gates(completed)
        opportunity_summary = major_winner_coverage_summary(opportunities)
        trigger_comparison = recovery_trigger_comparison_audit(history)
        market_context_audit = recovery_market_context_audit(history)

        st.markdown("---")
        st.header("R9四分支历史验证报告")
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
                    .lt(1)
                ).sum()
            )
            if not ledger.empty
            else 0
        )
        selected = completed[_actual_selected_mask(completed)]
        selected_returns = selected[PRIMARY_RETURN_COLUMN] if not selected.empty else pd.Series(dtype=float)
        metric_columns = st.columns(5)
        metric_columns[0].metric("已扫描周数", scanned_weeks)
        metric_columns[1].metric("无有效入选周", invalid_selection_weeks)
        metric_columns[2].metric("W3完整入选交易", len(selected))
        metric_columns[3].metric(
            "实际入选胜率",
            f"{((selected_returns > 0).mean() * 100.0 if len(selected_returns) else np.nan):.1f}%",
        )
        metric_columns[4].metric(
            "W3实际入选中位收益",
            f"{(selected_returns.median() if len(selected_returns) else np.nan):.2f}%",
        )

        st.subheader("研究验收门槛")
        st.dataframe(gates, width="stretch", hide_index=True)
        st.subheader("R7早期强势回调分支独立验收")
        st.dataframe(strong_gates, width="stretch", hide_index=True)
        st.subheader("R9强势有序个股再启动分支独立验收")
        st.dataframe(reacceleration_gates, width="stretch", hide_index=True)
        st.subheader("弱势复苏分支独立验收")
        st.dataframe(recovery_gates, width="stretch", hide_index=True)
        all_overall_passed = not gates.empty and gates["结果"].eq("通过").all()
        all_strong_passed = (
            not strong_gates.empty and strong_gates["结果"].eq("通过").all()
        )
        all_reacceleration_passed = (
            not reacceleration_gates.empty
            and reacceleration_gates["结果"].eq("通过").all()
        )
        all_recovery_passed = (
            not recovery_gates.empty and recovery_gates["结果"].eq("通过").all()
        )
        if (
            all_overall_passed
            and all_strong_passed
            and all_reacceleration_passed
            and all_recovery_passed
        ):
            st.success("R9总体与R7、R9强势子分支及N6首次转折均通过，可进入样本外验证。")
        else:
            st.error("R9任一独立分支尚未通过全部验收，当前代码不能进入实盘。")

        st.subheader("R9、R7、R3与R6-N6分支分别表现")
        st.dataframe(_format_report_frame(branches), width="stretch", hide_index=True)

        st.subheader("R7抗跌新高、过热和观察候选审计")
        st.dataframe(_format_report_frame(strong_audit), width="stretch", hide_index=True)

        st.subheader("强势市场四种背景同场对照")
        st.caption("只有早期强势回调允许实际Top2；其余背景用同一R7名次做反事实观察。")
        st.dataframe(
            _format_report_frame(strong_context_audit), width="stretch", hide_index=True
        )

        st.subheader("R7强势方案与R3旧首红同场对照")
        st.dataframe(
            _format_report_frame(strong_baseline_comparison),
            width="stretch",
            hide_index=True,
        )

        st.subheader("R9整理后再启动候选独立审计")
        st.dataframe(
            _format_report_frame(reacceleration_audit), width="stretch", hide_index=True
        )

        st.subheader("强势扩张：R9实际、强制Top2、R7与R3同场对照")
        st.dataframe(
            _format_report_frame(reexpansion_comparison),
            width="stretch",
            hide_index=True,
        )

        st.subheader("R9强势再启动排名是否有效")
        if reacceleration_diagnostics:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"指标": key, "数值": value}
                        for key, value in reacceleration_diagnostics.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

        st.subheader("强势市场阶段与实际动作")
        st.dataframe(strong_stage_audit, width="stretch", hide_index=True)

        st.subheader("R6首次转折与过热观察审计")
        st.dataframe(_format_report_frame(recovery_audit), width="stretch", hide_index=True)

        st.subheader("R5旧触发与R6首次事件同场对照")
        st.dataframe(
            _format_report_frame(trigger_comparison), width="stretch", hide_index=True
        )

        st.subheader("弱势市场背景分层（不参与否决）")
        st.dataframe(
            _format_report_frame(market_context_audit), width="stretch", hide_index=True
        )

        st.subheader("未来W3大涨机会覆盖审计（仅事后判卷）")
        st.caption(
            f"从每周完整基础池反查未来W3净收益≥{MAJOR_WINNER_W3_PCT:.0f}%的股票；"
            "未来收益不参与当周候选、门控或排名。"
        )
        if opportunity_summary.empty:
            st.info("尚无走满W3的大涨机会样本。")
        else:
            st.dataframe(
                _format_report_frame(opportunity_summary),
                width="stretch",
                hide_index=True,
            )
            with st.expander("查看未来W3大涨机会及漏选原因"):
                st.dataframe(
                    opportunities.drop(columns=["Config_ID"], errors="ignore"),
                    width="stretch",
                    hide_index=True,
                )

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

        st.subheader("完整入选组（R9允许第二名不达标时只选第一名）")
        if group_stats:
            group_cols = st.columns(4)
            group_cols[0].metric("完整入选组", group_stats["完整入选组"])
            group_cols[1].metric("两股组", group_stats["两股组"])
            group_cols[2].metric("单股组", group_stats["单股组"])
            group_cols[3].metric(
                "组平均收益为正", f"{group_stats['组平均收益为正比例%']:.1f}%"
            )
        else:
            st.info("尚无走满W3的完整入选组。")

        st.subheader("固定持有W1—W8路径")
        st.dataframe(_format_report_frame(horizons), width="stretch", hide_index=True)

        st.subheader("分年稳定性")
        st.dataframe(_format_report_frame(yearly), width="stretch", hide_index=True)

        st.subheader("市场状态与对应分支对照")
        st.dataframe(_format_report_frame(regimes), width="stretch", hide_index=True)

        with st.expander("查看实际入选历史明细"):
            detail_columns = [
                "Signal_Date", "Entry_Date", "Rank", "name", "ts_code", "Industry",
                "Strategy_Branch", "R3_Setup_Type", "Strong_Setup_Type", "Recovery_Setup_Type",
                "Strong_Reacceleration_Setup_Type", "Strong_Reacceleration_100",
                "Reaccel_Score_PauseControl20", "Reaccel_Score_RS13_20",
                "Reaccel_Score_Industry20", "Reaccel_Score_ATR20",
                "Reaccel_Score_NonChase20", "R9_Second_Qualified",
                "Strong_Resilience_100", "Strong_Score_Pause20",
                "Strong_Score_ATR20", "Strong_Score_NonChase20",
                "Strong_Score_Industry20", "Strong_Score_Position20",
                "Recovery_Early_Stage_100", "Recovery_Score_100",
                "Weekly_SKDJ_K6", "Weekly_SKDJ_D6",
                "Drawdown_26W_pct", "Price_to_MA10_Ratio", "Return_1W_pct",
                "Rebound_From_Week_Low_pct",
                "Score_Trend_20", "Score_Risk_10", "Entry_Score_100",
                "Score_Pullback_15", "Score_Contraction_15", "Score_Restart_15", "Score_RS_25",
                "Entry_Open", PRIMARY_RETURN_COLUMN, "MFE_W3_Net_pct", "MAE_W3_Raw_pct",
                "Path_10_vs_Minus5", "Early_Failure_2W", "Outcome_Grade", "Market_Regime",
                "Strong_Market_Context", "Strong_Market_Stage",
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
            branches,
            strong_audit,
            strong_gates,
            strong_context_audit,
            strong_baseline_comparison,
            reacceleration_audit,
            reacceleration_gates,
            reacceleration_diagnostics,
            reexpansion_comparison,
            strong_stage_audit,
            recovery_audit,
            recovery_gates,
            opportunities.drop(columns=["Config_ID"], errors="ignore"),
            opportunity_summary,
            trigger_comparison,
            market_context_audit,
        )
        st.download_button(
            "下载R9四分支完整研究结果",
            data=export_bytes,
            file_name="r9_orderly_stock_restart_four_branch_w3_audit_results.zip",
            mime="application/zip",
        )

if __name__ == "__main__":
    main()
