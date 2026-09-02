# -*- coding: utf-8 -*-
"""R19 冻结三仓W3组合风险审计版。

只保留已经进入主方案的R3中性Top2、R6弱势Top2、R15强势Top1，买入次日起
执行日内-10%灾难止损，否则固定W3退出。R7/R9、R12/R13、R14、R17整仓W4和
R18盈利尾仓均已验证失败并从执行链、报告与导出中删除。

本版不优化任何入场或退出参数，只记录三仓逐仓复投的真实资金占用和每日净值，
审计最大回撤、恢复时间、连续亏损、月度收益与资金暴露。
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
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts

warnings.filterwarnings("ignore")

APP_VERSION = "R19-FROZEN-THREE-SLOT-W3-PORTFOLIO-RISK-AUDIT"
APP_TITLE = "R19三仓W3组合风险审计"
ENGINE_PATCH = "R19-FROZEN-ENTRY-STOP-W3-PORTFOLIO-RISK"

CHECKPOINT_FILE = "r19_three_slot_w3_risk_candidates.csv"
SCAN_LEDGER_FILE = "r19_three_slot_w3_risk_scanned_dates.csv"
RUN_TASK_FILE = "r19_three_slot_w3_risk_running_task.json"
RESULT_STATE_GUARD_FILE = "r19_three_slot_w3_risk_result_state.guard"
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
STRONG_ATR_CONTRACTION_MIN = 0.70
STRONG_ATR_CONTRACTION_MAX = 0.90
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
TASK_LEASE_SECONDS = 45
DATA_READY_HOUR_SHANGHAI = 18
PRIMARY_RETURN_COLUMN = f"Fixed_Return_W{PRIMARY_HOLD_WEEKS}_Net_pct"
R16_STOP_SLIPPAGE_PCT = 0.30
R16_PRIMARY_STOP_PCT = -10.0
R16_PRIMARY_EXIT_RULE = "日内-10%硬止损（主规则）"
PORTFOLIO_CAPITAL_DEFAULT = 200000.0
PORTFOLIO_SLOT_COUNT = 3

# -----------------------------------------------------------------------------
# 通用安全读写
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

def _atomic_replace_bytes(path: str, payload: bytes):
    """事务回滚专用：原子恢复原始字节，不再改写.bak。"""
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".restore.", suffix=".tmp", dir=target_dir
    )
    try:
        with os.fdopen(fd, "wb") as file_obj:
            file_obj.write(payload)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@contextmanager
def _result_state_guard():
    """导入、逐周落盘和清除结果共用短锁，防止三个结果文件交叉写入。"""
    acquired = False
    for _ in range(240):
        try:
            descriptor = os.open(
                RESULT_STATE_GUARD_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
            os.close(descriptor)
            acquired = True
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(RESULT_STATE_GUARD_FILE) > 120.0:
                    os.remove(RESULT_STATE_GUARD_FILE)
                    continue
            except OSError:
                pass
            time.sleep(0.05)
    if not acquired:
        raise RuntimeError("结果文件正在写入，请稍后重试。")
    try:
        yield
    finally:
        try:
            os.remove(RESULT_STATE_GUARD_FILE)
        except OSError:
            pass

@contextmanager
def _result_files_transaction(paths):
    """多文件写入失败时恢复目标和.bak，避免留下半份导入结果。"""
    tracked = []
    for path in dict.fromkeys(str(item) for item in paths):
        tracked.extend([path, path + ".bak"])
    with _result_state_guard():
        snapshots = {}
        for path in tracked:
            if os.path.exists(path):
                with open(path, "rb") as file_obj:
                    snapshots[path] = file_obj.read()
            else:
                snapshots[path] = None
        try:
            yield
        except Exception:
            for path, payload in snapshots.items():
                if payload is None:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except OSError:
                        pass
                else:
                    _atomic_replace_bytes(path, payload)
            raise

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
    stock = stock_qfq_dict[ts_code]
    weekly = _weekly_bars(stock, end_date)
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
    previous_high_value = _safe_float(previous.get("high"))
    close_location_now = _safe_float(current.get("close_location"))
    setup_type = "趋势内MACD首红" if is_first_red else ""
    setup_candidate = bool(is_first_red)
    position_risk_ok = (
        math.isfinite(distance_ma20)
        and 0.0 <= distance_ma20 <= 25.0
        and math.isfinite(weekly_range)
        and weekly_range <= 25.0
    )
    trend_eligible = bool(base_trend_eligible and setup_candidate)

    # R15强势分支沿用已冻结的一次性“整理后再启动”结构；持续K>D或MACD改善
    # 不会重复触发。这里只生成个股结构，最终仍按强势市场与ATR收缩Top1入选。
    strong_trend_eligible = bool(
        base_trend_eligible
        and math.isfinite(ma40)
        and math.isfinite(ma20)
        and ma20 >= ma40
    )
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

    snapshot = {
        "Is_First_Red": bool(is_first_red),
        "R3_Setup_Candidate": bool(setup_candidate),
        "R3_Setup_Type": setup_type,
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
    return snapshot

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

def _market_state_metrics(pool: pd.DataFrame):
    """仅保留三分支真正使用的市场状态，全部字段在信号日已知。"""
    current_13w = _numeric_series(pool, "Return_13W_pct")
    current_1w = _numeric_series(pool, "Return_1W_pct")
    market_13w = _safe_float(current_13w.median(), 0.0)
    market_1w = _safe_float(current_1w.median(), 0.0)
    positive_breadth = float((current_1w > 0.0).mean()) if len(pool) else 0.0
    regime = (
        "强势"
        if market_13w >= MARKET_NEUTRAL_UPPER_PCT
        else "弱势"
        if market_13w <= MARKET_NEUTRAL_LOWER_PCT
        else "中性"
    )
    return {
        "Market_13W_Median_pct": market_13w,
        "Market_1W_Median_pct": market_1w,
        "Market_1W_Positive_Breadth": positive_breadth,
        "Market_Regime": regime,
    }

def score_frozen_candidates(pool_snapshots: pd.DataFrame):
    """冻结R3/R6/R15入场；不再计算任何已否决研究排名。"""
    if pool_snapshots.empty:
        return pd.DataFrame(), 0, 0

    pool = pool_snapshots.copy()
    return_13w = _numeric_series(pool, "Return_13W_pct")
    industry_median = pool.groupby(
        "Industry", dropna=False
    )["Return_13W_pct"].transform("median")
    pool["Industry_13W_Excess_pct"] = return_13w - pd.to_numeric(
        industry_median, errors="coerce"
    )
    pool["RS_4W_Pct"] = _percentile_rank(_numeric_series(pool, "Return_4W_pct"))
    pool["RS_8W_Pct"] = _percentile_rank(_numeric_series(pool, "Return_8W_pct"))
    pool["RS_13W_Pct"] = _percentile_rank(return_13w)
    pool["Industry_Excess_Pct"] = _percentile_rank(
        _numeric_series(pool, "Industry_13W_Excess_pct")
    )
    pool["MACD_Impulse_Pct"] = _percentile_rank(
        _numeric_series(pool, "MACD_Impulse_pct")
    )
    market_state = _market_state_metrics(pool)

    r3_trigger = _bool_series(pool, "R3_Setup_Candidate")
    strong_trigger = _bool_series(pool, "Strong_Reacceleration_Trigger")
    recovery_trigger = _bool_series(pool, "Recovery_Structure_Trigger")
    observation = r3_trigger | strong_trigger | recovery_trigger
    candidates = pool.loc[observation].copy()
    raw_count = int(observation.sum())
    if candidates.empty:
        return candidates, 0, 0

    candidates = _score_r1_six_factors(candidates)
    candidates["Rank"] = np.nan
    candidates["R3_Rank"] = np.nan
    candidates["Recovery_Rank"] = np.nan
    candidates["R15_Strong_Rank"] = np.nan
    candidates["Selected_Top2"] = False
    candidates["R15_Strong_ATR_Top1"] = False
    candidates["R19_Selected"] = False
    candidates["Entry_Eligible"] = False

    r3_eligible = candidates[_bool_series(candidates, "Trend_Eligible")].copy()
    if not r3_eligible.empty:
        ordered = r3_eligible.sort_values(
            ["Score_Trend_20", "Score_Risk_10", "Entry_Score_100", "ts_code"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        candidates.loc[ordered.index, "R3_Rank"] = np.arange(
            1, len(ordered) + 1, dtype=float
        )

    recovery_eligible = candidates[
        _bool_series(candidates, "Recovery_Eligible")
    ].copy()
    if not recovery_eligible.empty:
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
        candidates.loc[ordered.index, "Recovery_Rank"] = np.arange(
            1, len(ordered) + 1, dtype=float
        )

    strong_eligible_mask = (
        _bool_series(candidates, "Strong_Reacceleration_Trigger")
        & _bool_series(candidates, "Strong_Reacceleration_Risk_OK")
    )
    strong_eligible = candidates.loc[strong_eligible_mask].copy()
    if not strong_eligible.empty:
        ordered = strong_eligible.sort_values(
            ["ATR_Contraction", "ts_code"],
            ascending=[True, True],
            na_position="last",
            kind="mergesort",
        )
        candidates.loc[ordered.index, "R15_Strong_Rank"] = np.arange(
            1, len(ordered) + 1, dtype=float
        )

    r15_atr = pd.to_numeric(candidates["ATR_Contraction"], errors="coerce")
    r15_top1 = (
        pd.to_numeric(candidates["R15_Strong_Rank"], errors="coerce").eq(1)
        & r15_atr.between(
            STRONG_ATR_CONTRACTION_MIN,
            STRONG_ATR_CONTRACTION_MAX,
            inclusive="both",
        )
    )

    market_regime = str(market_state.get("Market_Regime", "中性"))
    r3_count = len(r3_eligible)
    recovery_count = len(recovery_eligible)
    strong_count = len(strong_eligible)
    if market_regime == "强势":
        active_branch = "R15强势温和ATR Top1"
        active_count = strong_count
        candidates["Rank"] = candidates["R15_Strong_Rank"]
        candidates["Entry_Eligible"] = strong_eligible_mask
        candidates["R15_Strong_ATR_Top1"] = r15_top1
        candidates["R19_Selected"] = r15_top1
        selection_valid = bool(r15_top1.any())
        block_reason = (
            ""
            if selection_valid
            else "R15强势Top1的ATR3/ATR13不在0.70—0.90，保持空仓"
        )
    elif market_regime == "中性":
        active_branch = "R3中性趋势"
        active_count = r3_count
        candidates["Rank"] = candidates["R3_Rank"]
        candidates["Entry_Eligible"] = _bool_series(candidates, "Trend_Eligible")
        selection_valid = r3_count >= MIN_VALID_SELECTION_SIZE
        block_reason = "" if selection_valid else "R3中性候选不足2只"
        if selection_valid:
            selected = (
                _bool_series(candidates, "Entry_Eligible")
                & pd.to_numeric(candidates["Rank"], errors="coerce").le(TOP_N)
            )
            candidates.loc[selected, ["Selected_Top2", "R19_Selected"]] = True
    else:
        active_branch = "R6弱势首次转折-N6"
        active_count = recovery_count
        candidates["Rank"] = candidates["Recovery_Rank"]
        candidates["Entry_Eligible"] = _bool_series(
            candidates, "Recovery_Eligible"
        )
        selection_valid = recovery_count >= MIN_VALID_SELECTION_SIZE
        block_reason = "" if selection_valid else "R6弱势候选不足2只"
        if selection_valid:
            selected = (
                _bool_series(candidates, "Entry_Eligible")
                & pd.to_numeric(candidates["Rank"], errors="coerce").le(TOP_N)
            )
            candidates.loc[selected, ["Selected_Top2", "R19_Selected"]] = True

    candidates["Selection_Valid"] = bool(selection_valid)
    candidates["Selection_Block_Reason"] = block_reason
    candidates["Strategy_Branch"] = active_branch
    candidates["Raw_Setup_Count"] = raw_count
    candidates["Observation_Row_Count"] = len(candidates)
    candidates["R3_Raw_First_Red_Count"] = int(r3_trigger.sum())
    candidates["Strong_Reacceleration_Structure_Count"] = int(
        strong_trigger.sum()
    )
    candidates["Recovery_Structure_Count"] = int(recovery_trigger.sum())
    candidates["Eligible_Trend_Count"] = r3_count
    candidates["Strong_Reacceleration_Eligible_Count"] = strong_count
    candidates["Recovery_Eligible_Count"] = recovery_count
    candidates["Active_Eligible_Count"] = active_count
    for key, value in market_state.items():
        column = (
            f"{key}_pct"
            if key == "Market_1W_Positive_Breadth"
            else key
        )
        candidates[column] = value * 100.0 if column.endswith("_pct") else value

    candidates = candidates.sort_values(
        ["R19_Selected", "Entry_Eligible", "Rank", "ts_code"],
        ascending=[False, False, True, True],
        na_position="last",
        kind="mergesort",
    )
    return candidates.reset_index(drop=True), raw_count, active_count

# -----------------------------------------------------------------------------
# 买入后固定路径标签：只评价入口，不构造退出策略
# -----------------------------------------------------------------------------
def track_w3_future_path(
    ts_code: str,
    signal_date: str,
    signal_raw_close: float,
    stock_qfq_dict: dict[str, pd.DataFrame],
    roundtrip_cost_pct: float,
    market_dates=None,
):
    """固定下一交易日开盘、T+1 -10%止损和W3退出，并保存净值所需日线。"""
    result: dict[str, Any] = {
        "Entry_Tradable": False,
        "Entry_Date": None,
        "Entry_Open": np.nan,
        "Entry_Open_QFQ": np.nan,
        "Entry_Gap_pct": np.nan,
        "Outcome_Complete": False,
        "Primary_Outcome_Date": None,
        "Primary_Return_Net_pct": np.nan,
        "Available_Future_Days": 0,
        "Available_Price_Days": 0,
        "Fixed_Return_W3_Net_pct": np.nan,
        "Fixed_Exit_W3_Date": None,
        "MFE_W3_Net_pct": np.nan,
        "MAE_W3_Raw_pct": np.nan,
        "Outcome_Grade": "待完成",
        "R16_Lifecycle_Data_Available": False,
        "R16_Stop_Minus10_Triggered": False,
        "R16_Stop_Minus10_Trigger_Date": None,
        "R16_Stop_Minus10_Trigger_Day": np.nan,
        "R16_Stop_Minus10_Exit_Date": None,
        "R16_Stop_Minus10_Exit_Day": np.nan,
        "R16_Stop_Minus10_Exit_Price_QFQ": np.nan,
        "R16_Stop_Minus10_Return_Net_pct": np.nan,
        "R16_Stop_Minus10_Delay_Days": np.nan,
        "R16_Stop_Minus10_Blocked_Days": 0,
        "R19_Daily_Path_JSON": "",
        "R19_Daily_Path_Available": False,
        "R19_Roundtrip_Cost_pct": float(roundtrip_cost_pct),
    }
    stock = stock_qfq_dict.get(ts_code)
    if stock is None:
        result["Entry_Status"] = "无行情"
        return result

    if market_dates is None:
        future_dates = stock.index[stock.index > signal_date].tolist()[
            : HOLD_WEEKS * MARKET_DAYS_PER_WEEK
        ]
    else:
        future_dates = [
            str(item) for item in market_dates if str(item) > signal_date
        ][: HOLD_WEEKS * MARKET_DAYS_PER_WEEK]
    result["Available_Future_Days"] = len(future_dates)
    if not future_dates:
        result["Entry_Status"] = "等待下一交易日"
        return result

    entry_date = future_dates[0]
    result["Entry_Date"] = entry_date
    if entry_date not in stock.index:
        result["Entry_Status"] = "下一交易日停牌或无行情，无法成交"
        return result

    future = stock.reindex(future_dates).copy()
    result["Available_Price_Days"] = int(future["close"].notna().sum())
    first = future.iloc[0]
    buy_price = _safe_float(first.get("open"))
    if not math.isfinite(buy_price) or buy_price <= 0:
        result["Entry_Status"] = "下一交易日开盘价缺失"
        return result

    raw_buy_price = _safe_float(first.get("raw_open"), buy_price)
    raw_first_high = _safe_float(
        first.get("raw_high"), _safe_float(first.get("high"))
    )
    raw_first_low = _safe_float(
        first.get("raw_low"), _safe_float(first.get("low"))
    )
    raw_first_close = _safe_float(
        first.get("raw_close"), _safe_float(first.get("close"))
    )
    is_20cm = ts_code.startswith(("300", "301", "688", "689"))
    limit_threshold = 0.195 if is_20cm else 0.095
    one_price_board = (
        all(
            math.isfinite(item)
            for item in (raw_first_high, raw_first_low, raw_first_close)
        )
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
    result["Entry_Open"] = raw_buy_price
    result["Entry_Open_QFQ"] = buy_price
    result["Entry_Gap_pct"] = (
        (raw_buy_price / signal_raw_close - 1.0) * 100.0
        if signal_raw_close > 0
        else np.nan
    )

    marked_close = pd.to_numeric(future["close"], errors="coerce").ffill()
    path_rows = [
        [str(day), round(float(close), 8)]
        for day, close in marked_close.items()
        if math.isfinite(_safe_float(close))
    ]
    result["R19_Daily_Path_JSON"] = json.dumps(
        path_rows, ensure_ascii=False, separators=(",", ":")
    )
    result["R19_Daily_Path_Available"] = bool(path_rows)

    primary_days = PRIMARY_HOLD_WEEKS * MARKET_DAYS_PER_WEEK
    if len(future) >= primary_days:
        exit_close = _safe_float(marked_close.iloc[primary_days - 1])
        if math.isfinite(exit_close):
            result["Fixed_Exit_W3_Date"] = str(future.index[primary_days - 1])
            result["Fixed_Return_W3_Net_pct"] = (
                (exit_close / buy_price - 1.0) * 100.0 - roundtrip_cost_pct
            )

    stop_price = buy_price * (1.0 + R16_PRIMARY_STOP_PCT / 100.0)
    previous_raw_close = raw_first_close
    trigger_position = None
    trigger_date = None
    pending_exit = False
    blocked_days = 0
    for position, (_, stop_row) in enumerate(future.iterrows()):
        raw_close = _safe_float(
            stop_row.get("raw_close"), _safe_float(stop_row.get("close"))
        )
        if position == 0:
            if math.isfinite(raw_close) and raw_close > 0:
                previous_raw_close = raw_close
            continue

        exit_open = _safe_float(stop_row.get("open"))
        day_low = _safe_float(stop_row.get("low"))
        raw_open = _safe_float(stop_row.get("raw_open"), exit_open)
        raw_high = _safe_float(
            stop_row.get("raw_high"), _safe_float(stop_row.get("high"))
        )
        raw_low = _safe_float(
            stop_row.get("raw_low"), _safe_float(stop_row.get("low"))
        )
        one_price_limit_down = (
            math.isfinite(previous_raw_close)
            and previous_raw_close > 0
            and all(
                math.isfinite(item)
                for item in (raw_open, raw_high, raw_low, raw_close)
            )
            and np.isclose(
                raw_high,
                raw_low,
                rtol=0,
                atol=max(0.001, abs(raw_open) * 1e-5),
            )
            and (raw_close / previous_raw_close - 1.0) <= -limit_threshold
        )
        if (
            trigger_position is None
            and math.isfinite(day_low)
            and day_low <= stop_price
        ):
            trigger_position = position
            trigger_date = str(future.index[position])
            pending_exit = bool(one_price_limit_down)
            blocked_days += int(pending_exit)

        if trigger_position is not None:
            if one_price_limit_down:
                if position > trigger_position:
                    blocked_days += 1
                pending_exit = True
            elif math.isfinite(exit_open) and exit_open > 0:
                reference_price = (
                    exit_open
                    if pending_exit or exit_open <= stop_price
                    else stop_price
                )
                slipped_price = reference_price * (
                    1.0 - R16_STOP_SLIPPAGE_PCT / 100.0
                )
                exit_price = (
                    max(day_low, slipped_price)
                    if math.isfinite(day_low) and day_low > 0
                    else slipped_price
                )
                result.update(
                    {
                        "R16_Stop_Minus10_Triggered": True,
                        "R16_Stop_Minus10_Trigger_Date": trigger_date,
                        "R16_Stop_Minus10_Trigger_Day": trigger_position + 1,
                        "R16_Stop_Minus10_Exit_Date": str(
                            future.index[position]
                        ),
                        "R16_Stop_Minus10_Exit_Day": position + 1,
                        "R16_Stop_Minus10_Exit_Price_QFQ": exit_price,
                        "R16_Stop_Minus10_Return_Net_pct": (
                            (exit_price / buy_price - 1.0) * 100.0
                            - roundtrip_cost_pct
                        ),
                        "R16_Stop_Minus10_Delay_Days": (
                            position - trigger_position
                        ),
                        "R16_Stop_Minus10_Blocked_Days": blocked_days,
                    }
                )
                break
        if math.isfinite(raw_close) and raw_close > 0:
            previous_raw_close = raw_close

    if trigger_position is not None and not result[
        "R16_Stop_Minus10_Triggered"
    ]:
        result["R16_Stop_Minus10_Triggered"] = True
        result["R16_Stop_Minus10_Trigger_Date"] = trigger_date
        result["R16_Stop_Minus10_Trigger_Day"] = trigger_position + 1
        result["R16_Stop_Minus10_Blocked_Days"] = blocked_days

    primary_future = future.head(primary_days)
    highs = pd.to_numeric(primary_future["high"], errors="coerce")
    lows = pd.to_numeric(primary_future["low"], errors="coerce")
    if highs.notna().any():
        result["MFE_W3_Net_pct"] = (
            (highs.max() / buy_price - 1.0) * 100.0 - roundtrip_cost_pct
        )
    if lows.notna().any():
        result["MAE_W3_Raw_pct"] = (
            lows.min() / buy_price - 1.0
        ) * 100.0

    primary_return = _safe_float(result["Fixed_Return_W3_Net_pct"])
    complete = len(future) >= primary_days and math.isfinite(primary_return)
    result["Outcome_Complete"] = bool(complete)
    result["R16_Lifecycle_Data_Available"] = bool(complete)
    result["Primary_Return_Net_pct"] = primary_return
    result["Primary_Outcome_Date"] = result["Fixed_Exit_W3_Date"]
    if complete:
        mfe = _safe_float(result["MFE_W3_Net_pct"], -np.inf)
        if mfe >= 15.0 and primary_return >= 5.0:
            result["Outcome_Grade"] = "S"
        elif mfe >= 10.0 or primary_return >= 5.0:
            result["Outcome_Grade"] = "A"
        elif primary_return >= 0.0:
            result["Outcome_Grade"] = "B"
        else:
            result["Outcome_Grade"] = "F"
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
    scan_status: str = "COMPLETED",
    data_gap_dates=None,
    candidate_row_count: int | None = None,
):
    gap_dates = sorted(set(str(item) for item in (data_gap_dates or []) if item))
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    row = pd.DataFrame(
        [
            {
                "Signal_Date": str(signal_date),
                "Raw_Setup_Count": int(raw_signal_count),
                "Eligible_Trend_Count": int(eligible_count),
                "Selected_Count": int(selected_count),
                "Candidate_Row_Count": (
                    int(candidate_row_count)
                    if candidate_row_count is not None
                    else np.nan
                ),
                "Selection_Block_Reason": str(selection_block_reason or ""),
                "Scan_Status": str(scan_status),
                "Market_Data_Gap_Count": len(gap_dates),
                "Market_Data_Gap_Dates": ",".join(gap_dates),
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
        & ledger["Scan_Status"].astype(str).isin(
            {"COMPLETED", "COMPLETED_WITH_GAPS", "SKIPPED_DATA_GAP"}
        )
    ]
    return set(filter(None, (parse_yyyymmdd(value) for value in match["Signal_Date"])))

def invalidate_recent_ledger_once(config_id: str, start_date: str, end_date: str):
    """新任务重算最近10周，并重试此前因数据缺口降级或跳过的所有周。"""
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    if ledger.empty or not {"Signal_Date", "Config_ID"}.issubset(ledger.columns):
        return
    dates = ledger["Signal_Date"].map(parse_yyyymmdd)
    recent_cutoff = (datetime.now() - timedelta(days=75)).strftime("%Y%m%d")
    lower = max(str(start_date), recent_cutoff)
    same_range = (
        ledger["Config_ID"].astype(str).eq(str(config_id))
        & dates.ge(str(start_date))
        & dates.le(str(end_date))
    )
    recent = dates.ge(lower) & dates.le(str(end_date))
    status = ledger.get(
        "Scan_Status", pd.Series("COMPLETED", index=ledger.index)
    ).astype(str)
    # PENDING_R19_NAV必须保留在账本中，后续才能进入“只补路径、不重排”流程。
    # 其余近期完整周与真实数据缺口仍按原稳定机制重扫。
    remove_mask = same_range & (
        (recent & status.eq("COMPLETED"))
        | status.isin(
            {
                "COMPLETED_WITH_GAPS",
                "SKIPPED_DATA_GAP",
                "PENDING_RESCAN",
            }
        )
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
    data_ready_str = _latest_data_ready_date().strftime("%Y%m%d")
    open_days = calendar[calendar["is_open"] == 1].copy()
    open_days["cal_date"] = open_days["cal_date"].astype(str)
    available_days = open_days[
        open_days["cal_date"] <= min(end_date, data_ready_str)
    ]
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
    pool = pd.DataFrame(pool_records)
    candidates, raw_count, eligible_count = score_frozen_candidates(pool)

    if is_preview_mode:
        if not candidates.empty:
            for column, value in {
                "Entry_Tradable": np.nan,
                "Outcome_Complete": False,
                "Primary_Outcome_Date": None,
                "Primary_Return_Net_pct": np.nan,
                "Entry_Status": "最新预览不计算未来结果",
                "Outcome_Grade": "待发生",
            }.items():
                candidates[column] = value
    else:
        if not candidates.empty:
            outcome_rows = []
            for _, row in candidates.iterrows():
                if bool(row.get("R19_Selected", False)):
                    outcome_rows.append(
                        track_w3_future_path(
                            str(row["ts_code"]),
                            signal_date,
                            _safe_float(row["Raw_Close"]),
                            stock_qfq_dict,
                            roundtrip_cost_pct,
                            market_dates,
                        )
                    )
                else:
                    outcome_rows.append(
                        {
                            "Entry_Tradable": False,
                            "Outcome_Complete": False,
                            "R16_Lifecycle_Data_Available": False,
                            "R19_Daily_Path_Available": False,
                            "R19_Daily_Path_JSON": "",
                            "R19_Roundtrip_Cost_pct": float(
                                roundtrip_cost_pct
                            ),
                            "Entry_Status": "未入选，不计算未来路径",
                            "Outcome_Grade": "未入选",
                        }
                    )
            candidates = pd.concat(
                [candidates.reset_index(drop=True), pd.DataFrame(outcome_rows)],
                axis=1,
            )
    return candidates, raw_count, eligible_count


def r19_backfill_frozen_daily_paths(
    candidates: pd.DataFrame,
    signal_date: str,
    stock_qfq_dict: dict[str, pd.DataFrame],
    roundtrip_cost_pct: float,
    market_dates,
):
    """只补已冻结入选股的每日路径，不重算候选、市场分支或排名。"""
    result = candidates.copy()
    selected = _bool_series(result, "R19_Selected")
    path_columns = (
        "R19_Daily_Path_JSON",
        "R19_Daily_Path_Available",
        "R19_Roundtrip_Cost_pct",
    )
    for index, row in result.loc[selected].iterrows():
        outcome = track_w3_future_path(
            str(row.get("ts_code", "")),
            signal_date,
            _safe_float(row.get("Raw_Close")),
            stock_qfq_dict,
            roundtrip_cost_pct,
            market_dates,
        )
        for column in path_columns:
            result.loc[index, column] = outcome.get(column)
    return result


def r19_pending_nav_dates(config_id: str):
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    if ledger.empty:
        return set()
    if "Config_ID" in ledger.columns:
        ledger = ledger[ledger["Config_ID"].astype(str).eq(str(config_id))]
    status = ledger.get(
        "Scan_Status", pd.Series("COMPLETED", index=ledger.index)
    ).astype(str)
    return set(
        ledger.loc[status.eq("PENDING_R19_NAV"), "Signal_Date"]
        .map(parse_yyyymmdd)
        .dropna()
        .astype(str)
    )


def r19_frozen_candidates_for_date(signal_date: str, config_id: str):
    history = read_csv_safe(CHECKPOINT_FILE)
    if history.empty:
        return history
    history["Signal_Date"] = history["Signal_Date"].map(parse_yyyymmdd)
    mask = history["Signal_Date"].astype(str).eq(str(signal_date))
    if "Config_ID" in history.columns:
        mask &= history["Config_ID"].astype(str).eq(str(config_id))
    return history.loc[mask].copy().reset_index(drop=True)

# -----------------------------------------------------------------------------
# 冻结方案通用统计
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

# -----------------------------------------------------------------------------
# R19 三仓W3组合与每日风险审计
# -----------------------------------------------------------------------------
def _date_series(frame: pd.DataFrame, column: str):
    raw = frame.get(column, pd.Series(None, index=frame.index)).astype(str)
    compact = raw.str.replace(r"\.0$", "", regex=True).str.replace("-", "", regex=False)
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(raw.loc[missing], errors="coerce")
    return parsed

def _r19_selected(history: pd.DataFrame, require_complete: bool = False):
    """统一冻结后的三市场入场集合；兼容导入R18结果。"""
    if history.empty:
        return history.iloc[0:0].copy()
    if "R19_Selected" in history.columns:
        selected_mask = _bool_series(history, "R19_Selected")
    else:
        selected_mask = (
            _bool_series(history, "Selected_Top2")
            | _bool_series(history, "R15_Strong_ATR_Top1")
        )
    selected = history.loc[selected_mask].copy()
    regime = selected.get(
        "Market_Regime", pd.Series("", index=selected.index)
    ).astype(str)
    selected["R19_市场分支"] = regime.map(
        {"强势": "R15强势", "中性": "R3中性", "弱势": "R6弱势"}
    ).fillna("未知")
    if require_complete:
        complete = (
            _bool_series(selected, "Entry_Tradable")
            & _bool_series(selected, "R16_Lifecycle_Data_Available")
            & pd.to_numeric(
                selected.get(
                    "Fixed_Return_W3_Net_pct",
                    pd.Series(np.nan, index=selected.index),
                ),
                errors="coerce",
            ).notna()
        )
        selected = selected.loc[complete].copy()
    return selected

def r19_trade_universe(history: pd.DataFrame):
    """生成三仓调度器唯一允许使用的W3交易集合。"""
    selected = _r19_selected(history, require_complete=True)
    if selected.empty:
        return selected
    fixed = pd.to_numeric(selected["Fixed_Return_W3_Net_pct"], errors="coerce")
    trigger_day = pd.to_numeric(
        selected.get(
            "R16_Stop_Minus10_Trigger_Day",
            pd.Series(np.nan, index=selected.index),
        ),
        errors="coerce",
    )
    stop_return = pd.to_numeric(
        selected.get(
            "R16_Stop_Minus10_Return_Net_pct",
            pd.Series(np.nan, index=selected.index),
        ),
        errors="coerce",
    )
    stop_exit = _date_series(selected, "R16_Stop_Minus10_Exit_Date")
    use_stop = (
        _bool_series(selected, "R16_Stop_Minus10_Triggered")
        & trigger_day.le(PRIMARY_HOLD_WEEKS * MARKET_DAYS_PER_WEEK)
        & stop_return.notna()
        & stop_exit.notna()
    )
    selected["R19_Realized_Return_pct"] = fixed
    selected.loc[use_stop, "R19_Realized_Return_pct"] = stop_return.loc[
        use_stop
    ]
    selected["R19_Entry_Date"] = _date_series(selected, "Entry_Date")
    selected["R19_Exit_Date"] = _date_series(
        selected, "Fixed_Exit_W3_Date"
    )
    selected.loc[use_stop, "R19_Exit_Date"] = stop_exit.loc[use_stop]
    selected["R19_Exit_Reason"] = np.where(
        use_stop, "T+1日内-10%灾难止损", "W3到期"
    )
    rank = pd.to_numeric(
        selected.get("Rank", pd.Series(np.nan, index=selected.index)),
        errors="coerce",
    )
    for fallback in ("R3_Rank", "Recovery_Rank", "R15_Strong_Rank"):
        rank = rank.where(
            rank.notna(),
            pd.to_numeric(
                selected.get(
                    fallback, pd.Series(np.nan, index=selected.index)
                ),
                errors="coerce",
            ),
        )
    selected["R19_Priority_Rank"] = rank.fillna(999.0)
    selected["R19_Path_Available"] = selected.get(
        "R19_Daily_Path_JSON", pd.Series("", index=selected.index)
    ).fillna("").astype(str).str.startswith("[[")
    selected = selected.dropna(
        subset=[
            "R19_Entry_Date",
            "R19_Exit_Date",
            "R19_Realized_Return_pct",
        ]
    )
    selected = selected[
        selected["R19_Exit_Date"] >= selected["R19_Entry_Date"]
    ]
    return selected.sort_values(
        ["R19_Entry_Date", "R19_Priority_Rank", "ts_code"],
        kind="mergesort",
    )

def _r19_parse_daily_path(raw_value):
    try:
        rows = json.loads(str(raw_value))
        frame = pd.DataFrame(rows, columns=["Date", "Close"])
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
        return frame.dropna().drop_duplicates("Date", keep="last").sort_values(
            "Date"
        )
    except Exception:
        return pd.DataFrame(columns=["Date", "Close"])

def _r19_losing_streak(ledger: pd.DataFrame):
    bought = ledger[ledger.get("执行状态", pd.Series(dtype=str)).eq("买入")].copy()
    if bought.empty:
        return 0, 0.0
    bought["_exit"] = pd.to_datetime(bought["Exit_Date"], errors="coerce")
    bought = bought.sort_values(["_exit", "仓位编号", "ts_code"])
    returns = pd.to_numeric(bought["交易净收益%"], errors="coerce")
    amounts = pd.to_numeric(bought["复投盈亏"], errors="coerce").fillna(0.0)
    best_count = current_count = 0
    best_loss = current_loss = 0.0
    for value, amount in zip(returns, amounts):
        if math.isfinite(_safe_float(value)) and value < 0.0:
            current_count += 1
            current_loss += amount
            if current_count > best_count or (
                current_count == best_count and current_loss < best_loss
            ):
                best_count = current_count
                best_loss = current_loss
        else:
            current_count = 0
            current_loss = 0.0
    return best_count, best_loss

def r19_three_slot_portfolio(
    history: pd.DataFrame,
    total_capital: float = PORTFOLIO_CAPITAL_DEFAULT,
):
    """三仓逐仓复投；卖出日资金不能用于当日开盘的新信号。"""
    universe = r19_trade_universe(history)
    if universe.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    slot_count = PORTFOLIO_SLOT_COUNT
    initial_stake = float(total_capital) / slot_count
    balances = [initial_stake] * slot_count
    active: dict[int, dict[str, Any]] = {}
    ledger_rows: list[dict[str, Any]] = []

    def release_before(entry_date):
        for slot, position in list(active.items()):
            if position["exit_date"] < entry_date:
                balances[slot] = position["exit_amount"]
                active.pop(slot, None)

    for entry_date, rows in universe.groupby("R19_Entry_Date", sort=True):
        release_before(entry_date)
        rows = rows.sort_values(
            ["R19_Priority_Rank", "ts_code"], kind="mergesort"
        )
        for _, row in rows.iterrows():
            code = str(row.get("ts_code", ""))
            free_slots = [i for i in range(slot_count) if i not in active]
            base = {
                "Signal_Date": row.get("Signal_Date"),
                "Entry_Date": entry_date.strftime("%Y%m%d"),
                "Exit_Date": row["R19_Exit_Date"].strftime("%Y%m%d"),
                "Rank": row.get("R19_Priority_Rank"),
                "ts_code": code,
                "name": row.get("name"),
                "市场分支": row.get("R19_市场分支"),
                "Outcome_Grade": row.get("Outcome_Grade"),
                "退出原因": row.get("R19_Exit_Reason"),
                "交易净收益%": _safe_float(row.get("R19_Realized_Return_pct")),
                "Entry_Open_QFQ": _safe_float(row.get("Entry_Open_QFQ")),
                "R19_Roundtrip_Cost_pct": _safe_float(
                    row.get("R19_Roundtrip_Cost_pct"), 0.20
                ),
                "R19_Daily_Path_JSON": row.get("R19_Daily_Path_JSON", ""),
                "R19_Path_Available": bool(row.get("R19_Path_Available", False)),
            }
            held_codes = {
                str(position.get("ts_code", "")) for position in active.values()
            }
            if code and code in held_codes:
                ledger_rows.append(
                    {**base, "执行状态": "跳过", "跳过原因": "已有同股持仓"}
                )
                continue
            if not free_slots:
                ledger_rows.append(
                    {**base, "执行状态": "跳过", "跳过原因": "三仓已满"}
                )
                continue
            slot = free_slots[0]
            entry_amount = balances[slot]
            return_pct = _safe_float(row.get("R19_Realized_Return_pct"))
            exit_amount = entry_amount * (1.0 + return_pct / 100.0)
            active[slot] = {
                "ts_code": code,
                "exit_date": row["R19_Exit_Date"],
                "exit_amount": exit_amount,
            }
            ledger_rows.append(
                {
                    **base,
                    "执行状态": "买入",
                    "跳过原因": "",
                    "仓位编号": slot + 1,
                    "固定仓额": initial_stake,
                    "固定仓额盈亏": initial_stake * return_pct / 100.0,
                    "复投买入金额": entry_amount,
                    "复投卖出金额": exit_amount,
                    "复投盈亏": exit_amount - entry_amount,
                }
            )
    for slot, position in list(active.items()):
        balances[slot] = position["exit_amount"]

    ledger = pd.DataFrame(ledger_rows)
    bought = ledger[ledger["执行状态"].eq("买入")].copy()
    skipped = ledger[ledger["执行状态"].eq("跳过")].copy()
    returns = pd.to_numeric(bought["交易净收益%"], errors="coerce").dropna()
    fixed_profit = pd.to_numeric(
        bought["固定仓额盈亏"], errors="coerce"
    ).fillna(0.0).sum()
    top5 = returns.nlargest(min(5, len(returns))).sum() if len(returns) else 0.0
    loss_count, loss_amount = _r19_losing_streak(ledger)
    summary = pd.DataFrame(
        [
            {
                "方案": "冻结三仓+T+1日内-10%止损+固定W3",
                "初始资金": float(total_capital),
                "初始单仓": initial_stake,
                "完整候选": len(universe),
                "实际买入": len(bought),
                "仓位冲突错过": len(skipped),
                "错过第一名": int(
                    pd.to_numeric(skipped.get("Rank"), errors="coerce")
                    .eq(1.0)
                    .sum()
                ),
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "平均单笔收益%": returns.mean() if len(returns) else np.nan,
                "中位单笔收益%": returns.median() if len(returns) else np.nan,
                "固定仓额期末资金": float(total_capital) + fixed_profit,
                "固定仓额总收益率%": fixed_profit / float(total_capital) * 100.0,
                "逐仓复投期末资金": float(sum(balances)),
                "逐仓复投总收益率%": (
                    sum(balances) / float(total_capital) - 1.0
                ) * 100.0,
                "前五笔占净利润%": (
                    top5 / returns.sum() * 100.0
                    if len(returns) and not np.isclose(returns.sum(), 0.0)
                    else np.nan
                ),
                "剔除前五笔后固定仓额收益率%": (
                    (returns.sum() - top5) / slot_count if len(returns) else np.nan
                ),
                "最大连续亏损笔数": loss_count,
                "最大连续亏损金额": loss_amount,
                "日线路径完整买入": int(
                    _bool_series(bought, "R19_Path_Available").sum()
                ),
            }
        ]
    )

    daily, monthly, risk = r19_daily_equity_curve(
        bought, float(total_capital), initial_stake
    )
    return summary, ledger, daily, monthly, risk

def r19_daily_equity_curve(
    bought: pd.DataFrame, total_capital: float, initial_stake: float
):
    """按每日收盘估值；往返成本从持仓第一天即保守计提。"""
    if bought.empty or not _bool_series(bought, "R19_Path_Available").all():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    trades_by_slot: dict[int, list[dict[str, Any]]] = {}
    for _, row in bought.iterrows():
        path = _r19_parse_daily_path(row.get("R19_Daily_Path_JSON", ""))
        if path.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        trades_by_slot.setdefault(int(row["仓位编号"]), []).append(
            {
                "entry": pd.to_datetime(row["Entry_Date"]),
                "exit": pd.to_datetime(row["Exit_Date"]),
                "entry_amount": _safe_float(row["复投买入金额"]),
                "exit_amount": _safe_float(row["复投卖出金额"]),
                "buy": _safe_float(row["Entry_Open_QFQ"]),
                "cost": _safe_float(row.get("R19_Roundtrip_Cost_pct"), 0.20),
                "path": path.set_index("Date")["Close"],
            }
        )
    for rows in trades_by_slot.values():
        rows.sort(key=lambda item: item["entry"])
    first_date = min(pd.to_datetime(bought["Entry_Date"], errors="coerce"))
    last_date = max(pd.to_datetime(bought["Exit_Date"], errors="coerce"))
    calendar = pd.bdate_range(first_date, last_date)
    output = []
    for day in calendar:
        cash = market_value = 0.0
        open_positions = 0
        for slot in range(1, PORTFOLIO_SLOT_COUNT + 1):
            slot_value = initial_stake
            slot_is_open = False
            for trade in trades_by_slot.get(slot, []):
                if day < trade["entry"]:
                    break
                if day >= trade["exit"]:
                    slot_value = trade["exit_amount"]
                    continue
                prices = trade["path"].loc[trade["path"].index <= day]
                close = _safe_float(prices.iloc[-1]) if len(prices) else np.nan
                if math.isfinite(close) and trade["buy"] > 0:
                    slot_value = trade["entry_amount"] * (
                        close / trade["buy"] - trade["cost"] / 100.0
                    )
                else:
                    slot_value = trade["entry_amount"]
                slot_is_open = True
                break
            if slot_is_open:
                market_value += slot_value
                open_positions += 1
            else:
                cash += slot_value
        equity = cash + market_value
        output.append(
            {
                "日期": day.strftime("%Y%m%d"),
                "现金": cash,
                "持仓市值": market_value,
                "账户权益": equity,
                "净值": equity / total_capital,
                "持仓数": open_positions,
                "资金暴露%": market_value / equity * 100.0 if equity else np.nan,
            }
        )
    daily = pd.DataFrame(output)
    equity = pd.to_numeric(daily["账户权益"], errors="coerce")
    previous = equity.shift(1).fillna(total_capital)
    daily["单日收益%"] = (equity / previous - 1.0) * 100.0
    running_peak = pd.Series(
        np.maximum.accumulate(np.maximum(equity.to_numpy(), total_capital)),
        index=daily.index,
    )
    daily["历史峰值"] = running_peak
    daily["回撤%"] = (equity / running_peak - 1.0) * 100.0

    dated = daily.copy()
    dated["_date"] = pd.to_datetime(dated["日期"], format="%Y%m%d")
    dated["月份"] = dated["_date"].dt.to_period("M").astype(str)
    month_end = dated.groupby("月份", sort=True).tail(1).copy()
    month_previous = month_end["账户权益"].shift(1).fillna(total_capital)
    monthly = month_end[["月份", "日期", "账户权益", "净值"]].copy()
    monthly["月收益%"] = (
        month_end["账户权益"].to_numpy() / month_previous.to_numpy() - 1.0
    ) * 100.0

    trough_index = int(daily["回撤%"].idxmin())
    trough_date = pd.to_datetime(daily.loc[trough_index, "日期"])
    peak_value = _safe_float(daily.loc[trough_index, "历史峰值"])
    pre_trough = daily.loc[:trough_index]
    peak_rows = pre_trough[
        np.isclose(
            pd.to_numeric(pre_trough["账户权益"], errors="coerce"), peak_value
        )
    ]
    peak_date = (
        pd.to_datetime(peak_rows.iloc[-1]["日期"])
        if not peak_rows.empty
        else first_date - pd.offsets.BDay(1)
    )
    after = daily.loc[trough_index + 1 :]
    recovered = after[pd.to_numeric(after["账户权益"], errors="coerce") >= peak_value]
    recovery_date = (
        pd.to_datetime(recovered.iloc[0]["日期"])
        if not recovered.empty
        else pd.NaT
    )
    recovery_days = (
        int((recovery_date - peak_date).days)
        if pd.notna(recovery_date)
        else np.nan
    )
    risk = pd.DataFrame(
        [
            {
                "最大回撤%": _safe_float(daily.loc[trough_index, "回撤%"]),
                "回撤峰值日": peak_date.strftime("%Y%m%d"),
                "回撤谷底日": trough_date.strftime("%Y%m%d"),
                "恢复日": (
                    recovery_date.strftime("%Y%m%d")
                    if pd.notna(recovery_date)
                    else "尚未恢复"
                ),
                "峰谷回撤自然日": int((trough_date - peak_date).days),
                "完整恢复自然日": recovery_days,
                "最大资金暴露%": pd.to_numeric(
                    daily["资金暴露%"], errors="coerce"
                ).max(),
                "平均资金暴露%": pd.to_numeric(
                    daily["资金暴露%"], errors="coerce"
                ).mean(),
                "最多同时持仓": int(daily["持仓数"].max()),
                "空仓交易日": int(daily["持仓数"].eq(0).sum()),
                "日线审计交易日": len(daily),
                "期末权益核对": _safe_float(daily.iloc[-1]["账户权益"]),
            }
        ]
    )
    return daily, monthly.reset_index(drop=True), risk

def r19_branch_summary(history: pd.DataFrame):
    universe = r19_trade_universe(history)
    columns = [
        "市场分支", "完整交易", "信号周", "止损交易", "胜率%",
        "平均收益%", "中位收益%", "Profit_Factor", "最差收益%",
    ]
    if universe.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    groups = [("合计", universe)] + [
        (branch, group)
        for branch, group in universe.groupby("R19_市场分支", sort=False)
    ]
    for branch, group in groups:
        returns = pd.to_numeric(
            group["R19_Realized_Return_pct"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "市场分支": branch,
                "完整交易": len(returns),
                "信号周": group["Signal_Date"].nunique(),
                "止损交易": int(
                    group.get(
                        "R19_Exit_Reason",
                        pd.Series("", index=group.index),
                    ).astype(str).str.contains("止损").sum()
                ),
                "胜率%": (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "Profit_Factor": _profit_factor(returns),
                "最差收益%": returns.min() if len(returns) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)

def r19_integrity_gates(
    history: pd.DataFrame,
    ledger: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    portfolio_ledger: pd.DataFrame,
    daily: pd.DataFrame,
):
    bought = portfolio_ledger[
        portfolio_ledger.get(
            "执行状态", pd.Series(dtype=str)
        ).astype(str).eq("买入")
    ].copy()
    completed_status = ledger.get(
        "Scan_Status", pd.Series("COMPLETED", index=ledger.index)
    ).astype(str)
    data_complete = completed_status.eq("COMPLETED").all() if len(ledger) else False
    path_complete = (
        not bought.empty
        and _bool_series(bought, "R19_Path_Available").all()
    )
    summary_row = (
        portfolio_summary.iloc[0]
        if not portfolio_summary.empty
        else pd.Series(dtype=object)
    )
    end_expected = _safe_float(summary_row.get("逐仓复投期末资金"))
    end_daily = (
        _safe_float(daily.iloc[-1]["账户权益"])
        if not daily.empty
        else np.nan
    )
    gates = [
        ("冻结规则", "仓位数严格为3", PORTFOLIO_SLOT_COUNT == 3, f"当前{PORTFOLIO_SLOT_COUNT}仓"),
        ("冻结规则", "最长持有严格为W3", PRIMARY_HOLD_WEEKS == 3, f"当前W{PRIMARY_HOLD_WEEKS}"),
        ("冻结规则", "灾难止损严格为T+1日内-10%", R16_PRIMARY_STOP_PCT == -10.0, f"当前{R16_PRIMARY_STOP_PCT:.1f}%"),
        ("冻结规则", "止损计0.3%不利滑点", np.isclose(R16_STOP_SLIPPAGE_PCT, 0.30), f"当前{R16_STOP_SLIPPAGE_PCT:.2f}%"),
        ("数据完整", "全部扫描周无缺口且已完成", data_complete, f"完成{int(completed_status.eq('COMPLETED').sum())}/{len(ledger)}周"),
        ("数据完整", "扫描账本与候选明细一致", result_state_consistency_audit(history, ledger).empty, "已核对"),
        ("净值完整", "全部实际买入均保存每日路径", path_complete, f"完整{int(_bool_series(bought, 'R19_Path_Available').sum())}/{len(bought)}笔"),
        ("资金约束", "任一日同时持仓不超过3只", not daily.empty and int(daily['持仓数'].max()) <= 3, f"当前最多{int(daily['持仓数'].max()) if not daily.empty else 0}只"),
        ("资金核对", "逐仓复投期末资金与每日净值一致", math.isfinite(end_expected) and math.isfinite(end_daily) and abs(end_expected - end_daily) <= 0.02, f"差额{(end_daily - end_expected) if math.isfinite(end_expected) and math.isfinite(end_daily) else np.nan:.2f}元"),
    ]
    return pd.DataFrame(
        [
            {
                "验收阶段": phase,
                "R19完整性项目": name,
                "结果": "通过" if passed else "未通过",
                "当前值": value,
            }
            for phase, name, passed, value in gates
        ]
    )

def market_data_gap_audit(ledger: pd.DataFrame):
    columns = [
        "Signal_Date",
        "Scan_Status",
        "Market_Data_Gap_Count",
        "Market_Data_Gap_Dates",
        "Selection_Block_Reason",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    frame = ledger.copy()
    status = frame.get(
        "Scan_Status", pd.Series("COMPLETED", index=frame.index)
    ).astype(str)
    gap_source = (
        frame["Market_Data_Gap_Count"]
        if "Market_Data_Gap_Count" in frame.columns
        else pd.Series(0, index=frame.index, dtype=int)
    )
    gap_count = pd.to_numeric(gap_source, errors="coerce").fillna(0)
    result = frame[status.ne("COMPLETED") | gap_count.gt(0)].copy()
    for column in columns:
        if column not in result.columns:
            result[column] = "" if column != "Market_Data_Gap_Count" else 0
    return result[columns].sort_values("Signal_Date").reset_index(drop=True)

def result_state_consistency_audit(history: pd.DataFrame, ledger: pd.DataFrame):
    """核对账本与候选检查点；不允许“账本完成、候选明细消失”进入报告。"""
    columns = [
        "Signal_Date",
        "Ledger_Status",
        "Expected_Candidate_Rows",
        "Actual_Candidate_Rows",
        "Expected_Selected_Count",
        "Actual_Selected_Count",
        "Consistency_Issue",
    ]
    history_frame = history.copy()
    if not history_frame.empty and "Signal_Date" in history_frame.columns:
        history_frame["Signal_Date"] = history_frame["Signal_Date"].map(
            parse_yyyymmdd
        )
        history_frame = history_frame.dropna(subset=["Signal_Date"])
    if ledger.empty:
        if history_frame.empty:
            return pd.DataFrame(columns=columns)
        rows = [
            {
                "Signal_Date": signal_date,
                "Ledger_Status": "MISSING",
                "Expected_Candidate_Rows": np.nan,
                "Actual_Candidate_Rows": len(group),
                "Expected_Selected_Count": np.nan,
                "Actual_Selected_Count": int(
                    _bool_series(
                        group,
                        "R19_Selected" if "R19_Selected" in group.columns else "Selected_Top2",
                    ).sum()
                ),
                "Consistency_Issue": "候选明细存在，但扫描账本缺失",
            }
            for signal_date, group in history_frame.groupby("Signal_Date", sort=True)
        ]
        return pd.DataFrame(rows, columns=columns)

    ledger_frame = ledger.copy()
    ledger_frame["Signal_Date"] = ledger_frame["Signal_Date"].map(parse_yyyymmdd)
    ledger_frame = ledger_frame.dropna(subset=["Signal_Date"])
    completed_statuses = {"COMPLETED", "COMPLETED_WITH_GAPS"}
    ledger_status = ledger_frame.get(
        "Scan_Status", pd.Series("COMPLETED", index=ledger_frame.index)
    ).astype(str)
    completed = ledger_frame[ledger_status.isin(completed_statuses)].copy()
    pending_research_dates = set(
        ledger_frame.loc[
            ledger_status.eq("PENDING_R19_NAV"),
            "Signal_Date",
        ].astype(str)
    )

    actual_rows = (
        history_frame.groupby("Signal_Date").size().to_dict()
        if not history_frame.empty
        else {}
    )
    actual_selected = (
        history_frame.assign(
            _selected=_bool_series(
                history_frame,
                "R19_Selected" if "R19_Selected" in history_frame.columns else "Selected_Top2",
            )
        )
        .groupby("Signal_Date")["_selected"]
        .sum()
        .astype(int)
        .to_dict()
        if not history_frame.empty
        else {}
    )
    candidate_dates = set(actual_rows)
    completed_dates = set(completed["Signal_Date"].astype(str))
    rows = []
    for _, row in completed.iterrows():
        signal_date = str(row["Signal_Date"])
        actual_count = int(actual_rows.get(signal_date, 0))
        actual_selected_count = int(actual_selected.get(signal_date, 0))
        raw_count = int(_safe_float(row.get("Raw_Setup_Count"), 0.0))
        expected_selected = int(_safe_float(row.get("Selected_Count"), 0.0))
        expected_candidate_raw = pd.to_numeric(
            pd.Series([row.get("Candidate_Row_Count")]), errors="coerce"
        ).iloc[0]
        has_exact_candidate_count = pd.notna(expected_candidate_raw)
        expected_candidate = (
            int(expected_candidate_raw) if has_exact_candidate_count else np.nan
        )
        issues = []
        if has_exact_candidate_count and actual_count != expected_candidate:
            issues.append("候选行数与账本不一致")
        elif not has_exact_candidate_count and raw_count > 0 and actual_count == 0:
            issues.append("账本显示存在候选，但候选明细缺失")
        if actual_selected_count != expected_selected:
            issues.append("实际入选数量与账本不一致")
        if issues:
            rows.append(
                {
                    "Signal_Date": signal_date,
                    "Ledger_Status": str(row.get("Scan_Status", "COMPLETED")),
                    "Expected_Candidate_Rows": expected_candidate,
                    "Actual_Candidate_Rows": actual_count,
                    "Expected_Selected_Count": expected_selected,
                    "Actual_Selected_Count": actual_selected_count,
                    "Consistency_Issue": "；".join(issues),
                }
            )

    for signal_date in sorted(
        candidate_dates - completed_dates - pending_research_dates
    ):
        group = history_frame[
            history_frame["Signal_Date"].astype(str).eq(signal_date)
        ]
        rows.append(
            {
                "Signal_Date": signal_date,
                "Ledger_Status": "MISSING_OR_PENDING",
                "Expected_Candidate_Rows": np.nan,
                "Actual_Candidate_Rows": len(group),
                "Expected_Selected_Count": np.nan,
                "Actual_Selected_Count": int(
                    _bool_series(group, "Selected_Top2").sum()
                ),
                "Consistency_Issue": "候选明细存在，但账本尚未完成",
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        "Signal_Date"
    ).reset_index(drop=True)

def repair_inconsistent_completed_ledger(config_id: str):
    """删除伪完成账本行，使build_run_dates自动把相应日期重新列为待扫描。"""
    history = read_csv_safe(CHECKPOINT_FILE)
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    if ledger.empty or "Config_ID" not in ledger.columns:
        return []
    if not history.empty:
        history["Signal_Date"] = history["Signal_Date"].map(parse_yyyymmdd)
        if "Config_ID" in history.columns:
            history = history[
                history["Config_ID"].astype(str).eq(str(config_id))
            ].copy()
    ledger["Signal_Date"] = ledger["Signal_Date"].map(parse_yyyymmdd)
    target = ledger[ledger["Config_ID"].astype(str).eq(str(config_id))].copy()
    issues = result_state_consistency_audit(history, target)
    if issues.empty:
        return []
    bad_dates = sorted(
        set(
            issues.loc[
                issues["Ledger_Status"].astype(str).isin(
                    {"COMPLETED", "COMPLETED_WITH_GAPS"}
                ),
                "Signal_Date",
            ].astype(str)
        )
    )
    if not bad_dates:
        return []
    remove_mask = (
        ledger["Config_ID"].astype(str).eq(str(config_id))
        & ledger["Signal_Date"].astype(str).isin(bad_dates)
    )
    remaining = ledger[~remove_mask].copy()
    with _result_files_transaction([SCAN_LEDGER_FILE]):
        if remaining.empty:
            remove_with_backup(SCAN_LEDGER_FILE)
        else:
            atomic_write_csv(remaining.reset_index(drop=True), SCAN_LEDGER_FILE)
    return bad_dates

def _apply_r19_selection_policy(frame: pd.DataFrame):
    """兼容导入R18/R19；只重建冻结的R15强势Top1与统一主入选标记。"""
    result = frame.copy()
    market_regime = result.get(
        "Market_Regime", pd.Series("", index=result.index)
    ).astype(str)
    strong_market = market_regime.eq("强势")
    strong_eligible = (
        strong_market
        & _bool_series(result, "Strong_Reacceleration_Trigger")
        & _bool_series(result, "Strong_Reacceleration_Risk_OK")
    )
    result["R15_Strong_Rank"] = np.nan
    atr = pd.to_numeric(
        result.get("ATR_Contraction", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    rows = result.loc[strong_eligible].copy()
    if not rows.empty:
        rows["_atr"] = atr.loc[rows.index]
        rows["_code"] = rows.get(
            "ts_code", pd.Series("", index=rows.index)
        ).astype(str)
        rows = rows.sort_values(
            ["Signal_Date", "_atr", "_code"],
            ascending=[True, True, True],
            na_position="last",
            kind="mergesort",
        )
        rows["_rank"] = rows.groupby("Signal_Date", sort=False).cumcount() + 1
        result.loc[rows.index, "R15_Strong_Rank"] = rows["_rank"].astype(float)

    result["R15_Strong_ATR_Top1"] = (
        strong_market
        & pd.to_numeric(result["R15_Strong_Rank"], errors="coerce").eq(1)
        & atr.between(
            STRONG_ATR_CONTRACTION_MIN,
            STRONG_ATR_CONTRACTION_MAX,
            inclusive="both",
        )
    )
    if "Selected_Top2" not in result.columns:
        result["Selected_Top2"] = False
    result.loc[strong_market, "Selected_Top2"] = False
    result.loc[strong_market, "Rank"] = result.loc[
        strong_market, "R15_Strong_Rank"
    ]
    result.loc[strong_market, "Entry_Eligible"] = strong_eligible.loc[
        strong_market
    ]
    result["R19_Selected"] = (
        _bool_series(result, "Selected_Top2")
        | _bool_series(result, "R15_Strong_ATR_Top1")
    )
    result["Strategy_Branch"] = result.get(
        "Strategy_Branch", pd.Series("", index=result.index)
    )
    result.loc[strong_market, "Strategy_Branch"] = "R15强势温和ATR Top1"
    return result

def import_prior_results_zip(
    zip_bytes: bytes,
    config_id: str,
    roundtrip_cost_pct: float,
):
    """事务导入R18/R19；R18只补扫缺少每日净值路径的信号周。"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        infos = {
            info.filename: info
            for info in archive.infolist()
            if not info.is_dir()
        }
        candidate_names = [
            name
            for name in infos
            if name.startswith(("01_all_r18_", "01_all_r19_"))
            and name.endswith("_candidates.csv")
        ]
        if len(candidate_names) != 1:
            raise ValueError("结果包中未找到唯一的R18或R19候选明细。")
        info = infos[candidate_names[0]]
        if info.file_size > 200 * 1024 * 1024:
            raise ValueError("候选明细超过200MB，拒绝导入。")
        candidates = pd.read_csv(
            io.BytesIO(archive.read(info)),
            encoding="utf-8-sig",
            low_memory=False,
        )
        required = {
            "Signal_Date",
            "ts_code",
            "Market_Regime",
            "Entry_Tradable",
            "Entry_Date",
            "Fixed_Return_W3_Net_pct",
            "Fixed_Exit_W3_Date",
            "R16_Stop_Minus10_Triggered",
            "R16_Stop_Minus10_Trigger_Day",
            "R16_Stop_Minus10_Return_Net_pct",
            "R16_Stop_Minus10_Exit_Date",
            "Strong_Reacceleration_Trigger",
            "Strong_Reacceleration_Risk_OK",
            "ATR_Contraction",
        }
        missing = sorted(required.difference(candidates.columns))
        if missing:
            raise ValueError("结果包缺少冻结主策略字段：" + "、".join(missing))
        candidates["Signal_Date"] = candidates["Signal_Date"].map(
            parse_yyyymmdd
        )
        candidates = candidates.dropna(subset=["Signal_Date", "ts_code"]).copy()
        if candidates.empty:
            raise ValueError("候选明细为空。")
        if candidates.duplicated(["Signal_Date", "ts_code"]).any():
            raise ValueError("候选明细存在重复日期与股票代码。")
        if "R19_Daily_Path_JSON" not in candidates.columns:
            candidates["R19_Daily_Path_JSON"] = ""
        if "R19_Daily_Path_Available" not in candidates.columns:
            candidates["R19_Daily_Path_Available"] = False
        if "R19_Roundtrip_Cost_pct" not in candidates.columns:
            candidates["R19_Roundtrip_Cost_pct"] = float(
                roundtrip_cost_pct
            )
        candidates = _apply_r19_selection_policy(candidates)
        candidates["Config_ID"] = str(config_id)

        ledger_name = next(
            (
                name
                for name in (
                    "02_scan_ledger.csv",
                    "26_scan_ledger.csv",
                )
                if name in infos
            ),
            None,
        )
        if ledger_name is None:
            raise ValueError("结果包缺少扫描账本，拒绝伪造零候选周。")
        ledger = pd.read_csv(
            io.BytesIO(archive.read(infos[ledger_name])),
            encoding="utf-8-sig",
            low_memory=False,
        )
        if "Signal_Date" not in ledger.columns:
            raise ValueError("扫描账本缺少Signal_Date。")
        ledger["Signal_Date"] = ledger["Signal_Date"].map(parse_yyyymmdd)
        ledger = ledger.dropna(subset=["Signal_Date"]).copy()
        if ledger.empty or ledger.duplicated(["Signal_Date"]).any():
            raise ValueError("扫描账本为空或存在重复日期。")
        ledger["Config_ID"] = str(config_id)

        selected = candidates[_bool_series(candidates, "R19_Selected")].copy()
        complete_selected = selected[
            _bool_series(selected, "Entry_Tradable")
            & _bool_series(selected, "R16_Lifecycle_Data_Available")
            & pd.to_numeric(
                selected.get(
                    "Fixed_Return_W3_Net_pct",
                    pd.Series(np.nan, index=selected.index),
                ),
                errors="coerce",
            ).notna()
        ].copy()
        missing_path_dates = set(
            complete_selected.loc[
                ~_bool_series(complete_selected, "R19_Daily_Path_Available"),
                "Signal_Date",
            ].astype(str)
        )
        pending_mask = ledger["Signal_Date"].astype(str).isin(
            missing_path_dates
        )
        ledger.loc[pending_mask, "Scan_Status"] = "PENDING_R19_NAV"
        ledger.loc[
            pending_mask, "Selection_Block_Reason"
        ] = "冻结交易已恢复；等待补算R19每日净值路径"

        row_counts = candidates.groupby("Signal_Date").size().to_dict()
        selected_counts = (
            candidates.assign(
                _selected=_bool_series(candidates, "R19_Selected")
            )
            .groupby("Signal_Date")["_selected"]
            .sum()
            .astype(int)
            .to_dict()
        )
        ledger["Candidate_Row_Count"] = (
            ledger["Signal_Date"].map(row_counts).fillna(0).astype(int)
        )
        ledger["Selected_Count"] = (
            ledger["Signal_Date"].map(selected_counts).fillna(0).astype(int)
        )

        existing_candidates = read_csv_safe(CHECKPOINT_FILE)
        combined_candidates = (
            pd.concat(
                [existing_candidates, candidates],
                ignore_index=True,
                sort=False,
            )
            if not existing_candidates.empty
            else candidates.copy()
        )
        combined_candidates["Signal_Date"] = combined_candidates[
            "Signal_Date"
        ].map(parse_yyyymmdd)
        combined_candidates = combined_candidates.dropna(
            subset=["Signal_Date", "ts_code"]
        ).drop_duplicates(
            ["Config_ID", "Signal_Date", "ts_code"], keep="last"
        )
        combined_candidates = combined_candidates.sort_values(
            ["Signal_Date", "Rank", "ts_code"],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)

        existing_ledger = read_csv_safe(SCAN_LEDGER_FILE)
        combined_ledger = (
            pd.concat(
                [existing_ledger, ledger], ignore_index=True, sort=False
            )
            if not existing_ledger.empty
            else ledger.copy()
        )
        combined_ledger["Signal_Date"] = combined_ledger[
            "Signal_Date"
        ].map(parse_yyyymmdd)
        combined_ledger = combined_ledger.dropna(
            subset=["Signal_Date"]
        ).drop_duplicates(
            ["Config_ID", "Signal_Date"], keep="last"
        ).sort_values("Signal_Date").reset_index(drop=True)

        with _result_files_transaction(
            [CHECKPOINT_FILE, SCAN_LEDGER_FILE]
        ):
            atomic_write_csv(combined_candidates, CHECKPOINT_FILE)
            atomic_write_csv(combined_ledger, SCAN_LEDGER_FILE)
            check_history = combined_candidates[
                combined_candidates["Config_ID"].astype(str).eq(str(config_id))
            ]
            check_ledger = combined_ledger[
                combined_ledger["Config_ID"].astype(str).eq(str(config_id))
            ]
            issues = result_state_consistency_audit(
                check_history, check_ledger
            )
            if not issues.empty:
                raise RuntimeError("导入后一致性校验失败，已自动回滚。")

    return {
        "candidate_rows": len(candidates),
        "known_weeks": len(ledger),
        "selected_rows": len(selected),
        "pending_nav_weeks": len(missing_path_dates),
    }

def build_export_zip(
    history: pd.DataFrame,
    ledger: pd.DataFrame,
    data_gaps: pd.DataFrame,
    branch_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    portfolio_ledger: pd.DataFrame,
    daily_equity: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    risk_summary: pd.DataFrame,
    integrity_gates: pd.DataFrame,
):
    """R19只导出主方案与风险审计，不再携带失败研究分支。"""
    files = {
        "01_all_r19_three_slot_w3_risk_candidates.csv": history,
        "02_scan_ledger.csv": ledger,
        "03_market_data_gap_audit.csv": data_gaps,
        "04_three_regime_trade_summary.csv": branch_summary,
        "05_three_slot_portfolio_summary.csv": portfolio_summary,
        "06_three_slot_trade_ledger.csv": portfolio_ledger,
        "07_daily_equity_curve.csv": daily_equity,
        "08_monthly_returns.csv": monthly_returns,
        "09_portfolio_risk_summary.csv": risk_summary,
        "10_r19_integrity_gates.csv": integrity_gates,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, frame in files.items():
            archive.writestr(
                name,
                frame.to_csv(index=False, encoding="utf-8-sig"),
            )
    return output.getvalue()

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
        "入场、排名、三仓、T+1日内-10%止损和W3退出全部冻结；"
        "本版只审计账户每日净值与实盘风险。"
    )
    st.caption(f"运行引擎修订：{ENGINE_PATCH}")
    st.warning(
        "最大回撤是观察结果，不是调参目标；本版不会为了改善回撤修改选股或退出。"
    )
    with st.expander("查看冻结交易规则"):
        st.markdown(
            """
- **R3中性**：MACD首红趋势池按原词典序取Top2；不足2只则空仓。
- **R6弱势**：26周深跌、SKDJ固定N=6的首次转折池按原五项早期阶段排名取Top2；不足2只则空仓。
- **R15强势**：整理后首次再启动候选仅按ATR3/ATR13从小到大取Top1；第一名必须位于0.70—0.90，不递补。
- **买入**：下一交易日开盘；一字涨停不虚构成交。
- **止损**：买入日不可卖，从下一交易日起执行日内-10%；计0.3%不利滑点，停牌或一字跌停顺延。
- **退出**：未触发止损的交易固定W3收盘卖出；卖出日资金不能用于当日开盘新信号。
- **资金**：本金等分三仓，每个仓位卖出后连同盈亏投入下一次新信号；仓位满时不追买旧信号。
- **已删除**：R7/R9、R12/R13、R14周末退出、R17整仓W4、R18盈利尾仓及全池大牛机会反查。
            """
        )

    today = _shanghai_now().date()
    default_start = today - timedelta(days=365)
    with st.sidebar:
        st.header("研究配置")
        mode = st.radio(
            "运行模式",
            ["历史R19三仓W3风险审计", "最新选股预览"],
            index=0,
            help="历史模式只使用完整周线；最新预览允许使用本周未完成周线且不写入回测。",
        )
        start_input = st.date_input("验证开始日期", value=default_start, disabled=mode != "历史R19三仓W3风险审计")
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
            help="固定W3与-10%硬止损收益都扣除该往返成本。",
        )
        portfolio_capital_wan = st.number_input(
            "三仓组合本金（万元）",
            value=20.0,
            min_value=1.0,
            max_value=10000.0,
            step=1.0,
            help="只改变资金报告的金额，不改变候选、排名、缓存或回测配置。",
        )

        st.markdown("---")
        try:
            secret_token = st.secrets.get("TUSHARE_TOKEN", "")
        except Exception:
            secret_token = ""
        token_input = st.text_input("Tushare Token", value=secret_token, type="password")

        st.markdown("---")
        clear_market_clicked = st.button("清空行情缓存")
        clear_history_clicked = st.button("清除R19历史结果")
        imported_results = st.file_uploader(
            "导入R18或R19结果包",
            type=["zip"],
            help="部署更新导致本地断点丢失时，可导入此前下载的结果包后继续。",
        )
        import_results_clicked = st.button(
            "恢复结果包中的断点",
            disabled=imported_results is None,
        )

    if max_mv <= min_mv:
        st.error("最高流通市值必须大于最低流通市值。")
        return
    if start_input > end_input and mode == "历史R19三仓W3风险审计":
        st.error("验证开始日期不能晚于截止日期。")
        return

    if clear_market_clicked:
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.success("行情缓存已清空。")

    if clear_history_clicked:
        with _result_files_transaction(
            [CHECKPOINT_FILE, SCAN_LEDGER_FILE]
        ):
            for path in (
                CHECKPOINT_FILE,
                SCAN_LEDGER_FILE,
            ):
                remove_with_backup(path)
        remove_with_backup(RUN_TASK_FILE)
        st.session_state.pop("r19_preview", None)
        st.success("R19历史结果和断点任务已清除。")

    token_clean = clean_token_str(token_input)
    config_id = make_config_id(min_price, min_mv, max_mv, roundtrip_cost_pct)
    if import_results_clicked and imported_results is not None:
        try:
            import_stats = import_prior_results_zip(
                imported_results.getvalue(),
                config_id,
                float(roundtrip_cost_pct),
            )
            pending = int(import_stats["pending_nav_weeks"])
            note = (
                f"；其中{pending}个信号周需补算每日净值路径"
                if pending
                else "；每日净值路径完整"
            )
            st.success(
                f"已恢复{import_stats['candidate_rows']}条候选、"
                f"{import_stats['known_weeks']}个扫描周、"
                f"{import_stats['selected_rows']}笔冻结信号{note}。"
            )
        except Exception as exc:
            st.error(f"结果包恢复失败：{exc}")
    is_preview_mode = mode == "最新选股预览"
    if "r19_worker_id" not in st.session_state:
        st.session_state["r19_worker_id"] = uuid.uuid4().hex
    worker_id = str(st.session_state["r19_worker_id"])
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

    start_label = "运行最新选股预览" if is_preview_mode else "启动历史R19三仓W3风险审计"
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
            repaired_dates = repair_inconsistent_completed_ledger(config_id)
            if repaired_dates:
                preview_dates = "、".join(repaired_dates[:8])
                more_text = "……" if len(repaired_dates) > 8 else ""
                st.warning(
                    f"已发现{len(repaired_dates)}个伪完成日期并自动重置："
                    f"{preview_dates}{more_text}。本次只补扫这些日期。"
                )
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
                    pending_nav = (
                        r19_pending_nav_dates(run_config_id)
                        if run_history
                        else set()
                    )
                    frozen_batch_rows = {
                        signal_date: r19_frozen_candidates_for_date(
                            signal_date, run_config_id
                        )
                        for signal_date in batch_dates
                        if signal_date in pending_nav
                    }
                    path_only_batch = bool(frozen_batch_rows) and all(
                        not frozen_batch_rows.get(signal_date, pd.DataFrame()).empty
                        for signal_date in batch_dates
                    )
                    if path_only_batch:
                        # 旧结果只补已冻结交易的日线路径，不再加载420日指标预热。
                        frozen_rows = pd.concat(
                            frozen_batch_rows.values(), ignore_index=True, sort=False
                        )
                        frozen_universe = r19_trade_universe(frozen_rows)
                        last_exit = pd.to_datetime(
                            frozen_universe.get(
                                "R19_Exit_Date", pd.Series(dtype="datetime64[ns]")
                            ),
                            errors="coerce",
                        ).max()
                        fetch_start = min(batch_dates)
                        requested_fetch_end = (
                            last_exit.to_pydatetime()
                            if pd.notna(last_exit)
                            else datetime.strptime(max(batch_dates), "%Y%m%d")
                            + timedelta(days=30)
                        )
                    else:
                        # 新扫描保留R1/R2稳定的420日指标预热窗口。
                        fetch_start = (
                            datetime.strptime(min(batch_dates), "%Y%m%d")
                            - timedelta(days=420)
                        ).strftime("%Y%m%d")
                        requested_fetch_end = (
                            datetime.strptime(max(batch_dates), "%Y%m%d")
                            + timedelta(days=75)
                        )
                    data_ready_date = _latest_data_ready_date()
                    fetch_end = min(
                        requested_fetch_end.date(), data_ready_date
                    ).strftime("%Y%m%d")
                    st.caption(
                        f"本批扫描{batch_dates[0]}—{batch_dates[-1]}；"
                        f"只加载必要行情窗口{fetch_start}—{fetch_end}。"
                        + (
                            " 本批仅补冻结交易的每日净值，不重算入场与排名。"
                            if path_only_batch
                            else ""
                        )
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
                        f"daily_basic仅下载{sync_stats.get('weekly_basic_days', 0)}个周末交易日；"
                        f"数据就绪截止{sync_stats.get('data_ready_through', fetch_end)}。"
                    )
                    if failed_dates:
                        failed_preview = "、".join(sorted(failed_dates)[:8])
                        more_text = "……" if len(failed_dates) > 8 else ""
                        st.warning(
                            f"{len(failed_dates)}个历史交易日仍未取得："
                            f"{failed_preview}{more_text}。任务继续运行并写入缺口审计；"
                            "含缺口结果不能通过数据完整性验收。"
                        )
                    if not stocks:
                        raise RuntimeError("未加载到行情；已成功下载的分片仍然保留。")

                    loaded_date_set = set(loaded_dates)
                    batch_gap_dates = sorted(set(failed_dates))
                    progress = st.progress(0, text="开始扫描冻结入场、-10%止损与W3每日路径……")
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
                            if run_preview:
                                st.warning(
                                    f"预览日{signal_date}行情尚未就绪，本次预览跳过。"
                                )
                                continue
                            with _result_files_transaction(
                                [CHECKPOINT_FILE, SCAN_LEDGER_FILE]
                            ):
                                replace_checkpoint_date(
                                    pd.DataFrame(), signal_date, run_config_id
                                )
                                mark_scan_complete(
                                    signal_date,
                                    0,
                                    0,
                                    0,
                                    run_config_id,
                                    f"扫描日行情缺失，已跳过：{signal_date}",
                                    scan_status="SKIPPED_DATA_GAP",
                                    data_gap_dates=sorted(
                                        set(batch_gap_dates) | {signal_date}
                                    ),
                                    candidate_row_count=0,
                                )
                            active_task["Completed_Weeks"] = int(
                                active_task.get("Completed_Weeks", 0)
                            ) + 1
                            active_task["Last_Date"] = signal_date
                            active_task["Error_Count"] = 0
                            save_owned_task(active_task, worker_id)
                            progress.progress(
                                (idx + 1) / len(batch_dates),
                                text=f"{signal_date}：扫描日行情缺失，已记录并跳过",
                            )
                            continue
                        weekly_mode = (
                            "已完成周线"
                            if run_history or latest_is_completed_week
                            else "未完成周线预览"
                        )
                        frozen_candidates = frozen_batch_rows.get(
                            signal_date, pd.DataFrame()
                        )
                        if not frozen_candidates.empty:
                            candidates = r19_backfill_frozen_daily_paths(
                                frozen_candidates,
                                signal_date,
                                stocks,
                                run_cost,
                                market_dates,
                            )
                            raw_count = int(
                                _safe_float(
                                    candidates.get(
                                        "Raw_Setup_Count",
                                        pd.Series(len(candidates), index=candidates.index),
                                    ).iloc[0],
                                    len(candidates),
                                )
                            )
                            eligible_count = int(
                                _safe_float(
                                    candidates.get(
                                        "Active_Eligible_Count",
                                        pd.Series(0, index=candidates.index),
                                    ).iloc[0],
                                    0,
                                )
                            )
                        else:
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
                        if not candidates.empty:
                            candidates["Market_Data_Gap_Count"] = len(batch_gap_dates)
                            candidates["Market_Data_Gap_Dates"] = ",".join(
                                batch_gap_dates
                            )
                            candidates["Backtest_Data_Complete"] = not bool(
                                batch_gap_dates
                            )
                        selected_count = (
                            int(_bool_series(candidates, "R19_Selected").sum())
                            if not candidates.empty
                            else 0
                        )
                        if run_preview:
                            st.session_state["r19_preview"] = candidates
                        else:
                            if not candidates.empty:
                                candidates["Config_ID"] = run_config_id
                            if not refresh_task_lease(
                                str(active_task.get("Task_ID", "")), worker_id
                            ):
                                raise RuntimeError("任务租约已经转移，本页停止写入回测断点。")
                            with _result_files_transaction(
                                [CHECKPOINT_FILE, SCAN_LEDGER_FILE]
                            ):
                                replace_checkpoint_date(
                                    candidates, signal_date, run_config_id
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
                                    scan_status=(
                                        "COMPLETED_WITH_GAPS"
                                        if batch_gap_dates
                                        else "COMPLETED"
                                    ),
                                    data_gap_dates=batch_gap_dates,
                                    candidate_row_count=len(candidates),
                                )
                            active_task["Completed_Weeks"] = int(active_task.get("Completed_Weeks", 0)) + 1
                            active_task["Last_Date"] = signal_date
                            active_task["Error_Count"] = 0
                            save_owned_task(active_task, worker_id)
                        progress.progress(
                            (idx + 1) / len(batch_dates),
                            text=(
                                f"{signal_date}：冻结结构候选{raw_count}只，"
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
                            st.success("历史R19三仓W3风险审计扫描完成。")
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

    preview = st.session_state.get("r19_preview")
    if is_preview_mode and isinstance(preview, pd.DataFrame):
        st.markdown("---")
        st.header("最新选股预览")
        if preview.empty:
            st.info("最新交易日没有冻结结构候选。")
        else:
            selected_preview = preview[
                _bool_series(preview, "R19_Selected")
            ].copy()
            if selected_preview.empty:
                reason = str(
                    preview.get(
                        "Selection_Block_Reason",
                        pd.Series("", index=preview.index),
                    ).iloc[0]
                    or "本周没有形成有效入选。"
                )
                st.warning(reason)
            else:
                columns = [
                    "Signal_Date", "Weekly_Data_Mode", "Rank", "name",
                    "ts_code", "Industry", "Strategy_Branch",
                    "R3_Setup_Type", "Strong_Reacceleration_Setup_Type",
                    "Recovery_Setup_Type", "R15_Strong_Rank",
                    "ATR_Contraction", "Recovery_Early_Stage_100",
                    "Weekly_SKDJ_K6", "Weekly_SKDJ_D6",
                    "Drawdown_26W_pct", "Return_1W_pct",
                    "Score_Trend_20", "Score_Risk_10",
                    "Entry_Score_100", "Raw_Close",
                    "Circ_MV_Billion", "Market_Regime",
                ]
                st.dataframe(
                    selected_preview[
                        [column for column in columns if column in selected_preview.columns]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            with st.expander("查看全部冻结候选与未入选原因"):
                st.dataframe(preview, width="stretch", hide_index=True)

    if rerun_needed:
        # 下一批前立即重跑，不在每个小批次重复构建整份历史报告和ZIP。
        gc.collect()
        time.sleep(0.3)
        st.rerun()

    raw_history = read_csv_safe(CHECKPOINT_FILE)
    raw_ledger = read_csv_safe(SCAN_LEDGER_FILE)
    if raw_history.empty and not raw_ledger.empty:
        empty_report_config = config_id
        if "Config_ID" in raw_ledger.columns:
            matching_ledger = raw_ledger[
                raw_ledger["Config_ID"].astype(str).eq(empty_report_config)
            ]
            if matching_ledger.empty:
                empty_report_config = str(
                    raw_ledger["Config_ID"].dropna().astype(str).iloc[-1]
                )
            empty_ledger = raw_ledger[
                raw_ledger["Config_ID"].astype(str).eq(empty_report_config)
            ].copy()
        else:
            empty_ledger = raw_ledger.copy()
        empty_state_issues = result_state_consistency_audit(
            pd.DataFrame(), empty_ledger
        )
        if not empty_state_issues.empty:
            st.markdown("---")
            st.error(
                "扫描账本已存在，但候选检查点为空。"
                "当前禁止生成研究报告；重新启动历史验证后会自动补扫缺失日期。"
            )
            st.dataframe(empty_state_issues, width="stretch", hide_index=True)
            return
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

        ledger = raw_ledger.copy()
        if not ledger.empty and "Config_ID" in ledger.columns:
            ledger = ledger[
                ledger["Config_ID"].astype(str) == report_config_id
            ].copy()
        status = ledger.get(
            "Scan_Status", pd.Series("COMPLETED", index=ledger.index)
        ).astype(str)
        pending_nav_rows = ledger[status.eq("PENDING_R19_NAV")].copy()
        data_gap_rows = market_data_gap_audit(ledger)
        actual_data_gaps = data_gap_rows[
            ~data_gap_rows.get(
                "Scan_Status", pd.Series("", index=data_gap_rows.index)
            ).astype(str).eq("PENDING_R19_NAV")
        ].copy()
        state_issues = result_state_consistency_audit(history, ledger)
        if not state_issues.empty:
            st.markdown("---")
            st.error(
                f"发现{len(state_issues)}周账本与候选明细不一致。"
                "当前禁止生成组合结论；重新启动R19后只补扫异常周。"
            )
            st.dataframe(state_issues, width="stretch", hide_index=True)
            return

        total_capital = float(portfolio_capital_wan) * 10000.0
        branch_summary = r19_branch_summary(history)
        (
            portfolio_summary,
            portfolio_ledger,
            daily_equity,
            monthly_returns,
            risk_summary,
        ) = r19_three_slot_portfolio(
            history,
            total_capital=total_capital,
        )
        integrity_gates = r19_integrity_gates(
            history,
            ledger,
            portfolio_summary,
            portfolio_ledger,
            daily_equity,
        )

        st.markdown("---")
        st.header("R19 冻结三仓W3组合风险报告")
        st.caption(
            "本报告不比较新策略，只回答当前主方案在真实三仓调度下赚了多少、"
            "会承受多大账户回撤、多久恢复以及资金是否经常闲置。"
        )

        summary_row = (
            portfolio_summary.iloc[0]
            if not portfolio_summary.empty
            else pd.Series(dtype=object)
        )
        risk_row = (
            risk_summary.iloc[0]
            if not risk_summary.empty
            else pd.Series(dtype=object)
        )
        metric_columns = st.columns(10)
        metric_columns[0].metric("扫描周", len(ledger))
        metric_columns[1].metric(
            "冻结完整交易", len(r19_trade_universe(history))
        )
        metric_columns[2].metric(
            "三仓实际买入", int(_safe_float(summary_row.get("实际买入"), 0))
        )
        metric_columns[3].metric(
            "错过第一名", int(_safe_float(summary_row.get("错过第一名"), 0))
        )
        metric_columns[4].metric(
            "组合交易胜率",
            f"{_safe_float(summary_row.get('胜率%')):.1f}%",
        )
        metric_columns[5].metric(
            "固定仓额收益",
            f"{_safe_float(summary_row.get('固定仓额总收益率%')):.2f}%",
        )
        metric_columns[6].metric(
            "逐仓复投收益",
            f"{_safe_float(summary_row.get('逐仓复投总收益率%')):.2f}%",
        )
        metric_columns[7].metric(
            "最大回撤",
            (
                f"{_safe_float(risk_row.get('最大回撤%')):.2f}%"
                if not risk_summary.empty
                else "待补日线"
            ),
        )
        metric_columns[8].metric(
            "最大连续亏损",
            f"{int(_safe_float(summary_row.get('最大连续亏损笔数'), 0))}笔",
        )
        metric_columns[9].metric(
            "空仓交易日",
            (
                int(_safe_float(risk_row.get("空仓交易日"), 0))
                if not risk_summary.empty
                else "待补日线"
            ),
        )

        if not pending_nav_rows.empty:
            st.info(
                f"已恢复旧结果，但有{len(pending_nav_rows)}个信号周缺少每日净值路径。"
                "点击启动历史R19后只补扫这些周；入场、止损和W3收益不会重排。"
            )
        if not actual_data_gaps.empty:
            st.error(
                f"存在{len(actual_data_gaps)}个行情缺口或跳过周，当前组合结果不完整。"
            )
            with st.expander("查看行情缺口"):
                st.dataframe(actual_data_gaps, width="stretch", hide_index=True)

        st.subheader("冻结规则与结果完整性")
        st.dataframe(integrity_gates, width="stretch", hide_index=True)

        st.subheader(
            f"三仓资金结果（本金{float(portfolio_capital_wan):.0f}万元）"
        )
        portfolio_display = portfolio_summary.copy()
        for column in (
            "初始资金", "初始单仓", "固定仓额期末资金",
            "逐仓复投期末资金", "最大连续亏损金额",
        ):
            if column in portfolio_display.columns:
                portfolio_display[column] = pd.to_numeric(
                    portfolio_display[column], errors="coerce"
                ).round(0)
        st.dataframe(
            _format_report_frame(portfolio_display),
            width="stretch",
            hide_index=True,
        )

        st.subheader("强势、中性、弱势分支表现")
        st.dataframe(
            _format_report_frame(branch_summary),
            width="stretch",
            hide_index=True,
        )

        if risk_summary.empty:
            st.warning(
                "每日净值尚未完整，不能用单笔MAE代替账户最大回撤。"
                "完成R19补扫前只参考交易与资金调度结果。"
            )
        else:
            st.subheader("账户风险摘要")
            st.dataframe(
                _format_report_frame(risk_summary),
                width="stretch",
                hide_index=True,
            )
            chart_frame = daily_equity.copy()
            chart_frame["日期"] = pd.to_datetime(
                chart_frame["日期"], format="%Y%m%d", errors="coerce"
            )
            chart_frame = chart_frame.dropna(subset=["日期"]).set_index("日期")
            st.subheader("每日账户净值")
            st.line_chart(chart_frame[["净值"]])
            st.subheader("账户回撤")
            st.line_chart(chart_frame[["回撤%"]])
            st.subheader("月度收益")
            st.dataframe(
                _format_report_frame(monthly_returns),
                width="stretch",
                hide_index=True,
            )
            with st.expander("查看每日现金、持仓市值与资金暴露"):
                st.dataframe(
                    _format_report_frame(daily_equity),
                    width="stretch",
                    hide_index=True,
                )

        with st.expander("查看三仓逐笔买入、跳过与复投明细"):
            st.dataframe(
                _format_report_frame(portfolio_ledger),
                width="stretch",
                hide_index=True,
            )

        selected_detail = _r19_selected(history, require_complete=False)
        with st.expander("查看冻结入选信号明细"):
            detail_columns = [
                "Signal_Date", "Entry_Date", "Rank", "name", "ts_code",
                "Industry", "Market_Regime", "Strategy_Branch",
                "R15_Strong_Rank", "ATR_Contraction",
                "Recovery_Early_Stage_100", "Weekly_SKDJ_K6",
                "Weekly_SKDJ_D6", "Drawdown_26W_pct",
                "Entry_Open", "Fixed_Return_W3_Net_pct",
                "MFE_W3_Net_pct", "MAE_W3_Raw_pct",
                "R16_Stop_Minus10_Triggered",
                "R16_Stop_Minus10_Exit_Date",
                "R16_Stop_Minus10_Return_Net_pct",
                "Outcome_Grade",
            ]
            st.dataframe(
                selected_detail[
                    [
                        column
                        for column in detail_columns
                        if column in selected_detail.columns
                    ]
                ].sort_values(
                    ["Signal_Date", "Rank"],
                    ascending=[False, True],
                    kind="mergesort",
                ),
                width="stretch",
                hide_index=True,
            )

        export_bytes = build_export_zip(
            history.drop(columns=["Config_ID"], errors="ignore"),
            ledger.drop(columns=["Config_ID"], errors="ignore"),
            data_gap_rows,
            branch_summary,
            portfolio_summary,
            portfolio_ledger,
            daily_equity,
            monthly_returns,
            risk_summary,
            integrity_gates,
        )
        st.download_button(
            "下载R19三仓W3组合风险审计结果",
            data=export_bytes,
            file_name="r19_three_slot_w3_portfolio_risk_audit_results.zip",
            mime="application/zip",
        )

if __name__ == "__main__":
    main()
