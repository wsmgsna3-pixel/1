# -*- coding: utf-8 -*-
"""
R14 高弹性持仓生命周期研究版。

R3、R6与R11.1实际组合逐行保持不变；R11—R13继续冻结为研究对照。R14只把
R13已经预先声明并完成三年检验的“日线MACD柱加速度”单因子Top2单独标记为
高弹性观察组，不增加候选、硬门或选股参数。买入仍为信号后下一交易日开盘，
退出只比较三个事先固定的收盘规则，并在触发后的下一可交易日开盘成交：W1收盘
亏损、W1/W2连续收盘亏损、W2收盘不高于-5%。所有R14结果均为研究审计，绝不
覆盖R3/R6实际入选，也不使用买入后的路径反向修改排名。
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

APP_VERSION = "R14-MACD-ELASTIC-LIFECYCLE-RESEARCH"
APP_TITLE = "R14高弹性持仓生命周期验证器"
ENGINE_PATCH = "R14-MACD-ELASTIC-NEXT-OPEN-EXIT-AUDIT"

CHECKPOINT_FILE = "r14_macd_elastic_lifecycle_candidates.csv"
SCAN_LEDGER_FILE = "r14_macd_elastic_lifecycle_scanned_dates.csv"
OPPORTUNITY_FILE = "r14_macd_elastic_lifecycle_w3_major_winner_opportunities.csv"
RUN_TASK_FILE = "r14_macd_elastic_lifecycle_running_task.json"
RESULT_STATE_GUARD_FILE = "r14_macd_elastic_lifecycle_result_state.guard"
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
R11_ATR_CONTRACTION_MIN = 0.70
R11_ATR_CONTRACTION_MAX = 0.90
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
DATA_READY_HOUR_SHANGHAI = 18
PRIMARY_RETURN_COLUMN = f"Fixed_Return_W{PRIMARY_HOLD_WEEKS}_Net_pct"
R14_LIFECYCLE_HORIZONS = (3, 4, 6, 8)
R14_PRIMARY_EXIT_RULE = "W1/W2连续收盘亏损"
R14_EXIT_RULES = (
    ("固定持有", None, None),
    ("W1收盘亏损", "R14_Trigger_W1_Close_Loss", "R14_W1_Next_Open_Return_Net_pct"),
    (
        "W1/W2连续收盘亏损",
        "R14_Trigger_W1_W2_Both_Loss",
        "R14_W2_Next_Open_Return_Net_pct",
    ),
    (
        "W2收盘不高于-5%",
        "R14_Trigger_W2_Close_Minus5",
        "R14_W2_Next_Open_Return_Net_pct",
    ),
)


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


def _daily_restart_snapshot(stock: pd.DataFrame, end_date: str):
    """R13日线快照；所有滚动值严格截止信号日，不读取下一交易日。"""
    fields = {
        "Daily_Restart_Data_Available": False,
        "Daily_Close": np.nan,
        "Daily_MA5": np.nan,
        "Daily_MA10": np.nan,
        "Daily_MA20": np.nan,
        "Daily_MA30": np.nan,
        "Daily_Close_to_MA20_Ratio": np.nan,
        "Daily_MA5_Slope_3D_pct": np.nan,
        "Daily_MACD_Hist": np.nan,
        "Daily_Previous_MACD_Hist": np.nan,
        "Daily_MACD_Hist_Delta_pct": np.nan,
        "Daily_Return_5D_pct": np.nan,
        "Daily_Close_to_Prior_10D_High_Ratio": np.nan,
        "Daily_Higher_Low_5D_pct": np.nan,
        "Daily_Close_Location_5D": np.nan,
        "Daily_MA5_Above_MA10": False,
        "Daily_MACD_Improving": False,
    }
    daily = stock[stock.index <= end_date].tail(90).copy()
    if len(daily) < 35:
        return fields
    close = pd.to_numeric(daily.get("close"), errors="coerce")
    high = pd.to_numeric(daily.get("high"), errors="coerce")
    low = pd.to_numeric(daily.get("low"), errors="coerce")
    if close.isna().all() or high.isna().all() or low.isna().all():
        return fields

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma30 = close.rolling(30).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2.0 * (dif - dea)
    prior_high10 = high.shift(1).rolling(10).max()
    current_low5 = low.rolling(5).min()
    previous_low5 = low.shift(5).rolling(5).min()
    current_high5 = high.rolling(5).max()

    current_close = _safe_float(close.iloc[-1])
    current_ma5 = _safe_float(ma5.iloc[-1])
    current_ma10 = _safe_float(ma10.iloc[-1])
    current_ma20 = _safe_float(ma20.iloc[-1])
    current_ma30 = _safe_float(ma30.iloc[-1])
    current_hist = _safe_float(hist.iloc[-1])
    previous_hist = _safe_float(hist.iloc[-2])
    hist_delta_pct = (
        (current_hist - previous_hist) / current_close * 100.0
        if all(math.isfinite(item) for item in (current_hist, previous_hist, current_close))
        and current_close > 0.0
        else np.nan
    )
    close_to_ma20 = (
        current_close / current_ma20
        if all(math.isfinite(item) for item in (current_close, current_ma20))
        and current_ma20 > 0.0
        else np.nan
    )
    prior_high10_value = _safe_float(prior_high10.iloc[-1])
    close_to_high10 = (
        current_close / prior_high10_value
        if all(math.isfinite(item) for item in (current_close, prior_high10_value))
        and prior_high10_value > 0.0
        else np.nan
    )
    low5_value = _safe_float(current_low5.iloc[-1])
    prior_low5_value = _safe_float(previous_low5.iloc[-1])
    higher_low5 = (
        (low5_value / prior_low5_value - 1.0) * 100.0
        if all(math.isfinite(item) for item in (low5_value, prior_low5_value))
        and prior_low5_value > 0.0
        else np.nan
    )
    high5_value = _safe_float(current_high5.iloc[-1])
    close_location5 = (
        (current_close - low5_value) / (high5_value - low5_value)
        if all(math.isfinite(item) for item in (current_close, low5_value, high5_value))
        and high5_value > low5_value
        else np.nan
    )
    ma5_slope3 = (
        (current_ma5 / _safe_float(ma5.iloc[-4]) - 1.0) * 100.0
        if math.isfinite(current_ma5)
        and math.isfinite(_safe_float(ma5.iloc[-4]))
        and _safe_float(ma5.iloc[-4]) > 0.0
        else np.nan
    )
    return5 = (
        (current_close / _safe_float(close.iloc[-6]) - 1.0) * 100.0
        if math.isfinite(current_close)
        and math.isfinite(_safe_float(close.iloc[-6]))
        and _safe_float(close.iloc[-6]) > 0.0
        else np.nan
    )
    required = [
        close_to_ma20,
        ma5_slope3,
        hist_delta_pct,
        return5,
        higher_low5,
    ]
    fields.update(
        {
            "Daily_Restart_Data_Available": all(
                math.isfinite(item) for item in required
            ),
            "Daily_Close": current_close,
            "Daily_MA5": current_ma5,
            "Daily_MA10": current_ma10,
            "Daily_MA20": current_ma20,
            "Daily_MA30": current_ma30,
            "Daily_Close_to_MA20_Ratio": close_to_ma20,
            "Daily_MA5_Slope_3D_pct": ma5_slope3,
            "Daily_MACD_Hist": current_hist,
            "Daily_Previous_MACD_Hist": previous_hist,
            "Daily_MACD_Hist_Delta_pct": hist_delta_pct,
            "Daily_Return_5D_pct": return5,
            "Daily_Close_to_Prior_10D_High_Ratio": close_to_high10,
            "Daily_Higher_Low_5D_pct": higher_low5,
            "Daily_Close_Location_5D": close_location5,
            "Daily_MA5_Above_MA10": bool(
                all(math.isfinite(item) for item in (current_ma5, current_ma10))
                and current_ma5 >= current_ma10
            ),
            "Daily_MACD_Improving": bool(
                math.isfinite(hist_delta_pct) and hist_delta_pct > 0.0
            ),
        }
    )
    return fields


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

    snapshot = {
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
    snapshot.update(_daily_restart_snapshot(stock, end_date))
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


def _score_daily_restart_quality(frame: pd.DataFrame):
    """R13五项日线重启等权分；只在同周R6合格池内做横截面排序。"""
    scored = frame.copy()
    factors = [
        ("Daily_Close_to_MA20_Ratio", "R13_Daily_Price_Repair_20"),
        ("Daily_MA5_Slope_3D_pct", "R13_Daily_MA5_Slope_20"),
        ("Daily_MACD_Hist_Delta_pct", "R13_Daily_MACD_Accel_20"),
        ("Daily_RS_5D_Pct", "R13_Daily_RS5_20"),
        ("Daily_Higher_Low_5D_pct", "R13_Daily_Higher_Low_20"),
    ]
    component_columns = []
    for source, target in factors:
        scored[target] = (
            _percentile_rank(_numeric_series(scored, source), higher_is_better=True)
            * 20.0
        )
        component_columns.append(target)
    scored["R13_Daily_Restart_100"] = scored[component_columns].sum(axis=1)
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


def score_r13_candidates(pool_snapshots: pd.DataFrame):
    """R3/R6保持原实际基线；R11—R14全部只研究。"""
    if pool_snapshots.empty:
        return pd.DataFrame(), 0, 0
    pool = pool_snapshots.copy()
    daily_return5 = _numeric_series(pool, "Daily_Return_5D_pct")
    pool["Daily_RS_5D_Pct"] = _percentile_rank(daily_return5)
    daily_industry_median = pool.groupby(
        "Industry", dropna=False
    )["Daily_Return_5D_pct"].transform("median")
    pool["Daily_Industry_5D_Excess_pct"] = daily_return5 - pd.to_numeric(
        daily_industry_median, errors="coerce"
    )
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
    candidates["R11_Strong_Rank"] = np.nan
    candidates["Recovery_Rank"] = np.nan
    candidates["R12_Recovery_Repair_Rank"] = np.nan
    candidates["R13_Daily_Restart_Rank"] = np.nan
    candidates["R14_MACD_Elastic_Rank"] = np.nan
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
        "R13_Daily_Price_Repair_20",
        "R13_Daily_MA5_Slope_20",
        "R13_Daily_MACD_Accel_20",
        "R13_Daily_RS5_20",
        "R13_Daily_Higher_Low_20",
        "R13_Daily_Restart_100",
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
    candidates["R7_Research_Top2"] = False
    candidates["R9_Failure_Control_Top2"] = False
    candidates["R11_ATR_Band_Pass"] = False
    candidates["R11_Strong_Research_Top1"] = False
    candidates["R12_Recovery_Repair_Top2"] = False
    candidates["R13_Daily_Restart_Top1"] = False
    candidates["R13_Daily_Restart_Top2"] = False
    candidates["R14_MACD_Elastic_Top1"] = False
    candidates["R14_MACD_Elastic_Top2"] = False
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
        # R11只使用一个预先冻结的强势排名：ATR3/ATR13由小到大，代码仅作
        # 稳定并列项。先排全体合格候选，再审查第一名是否落在0.70—0.90；
        # 第一名越界时不允许用第二名递补。
        r11_ordered = reacceleration_eligible.sort_values(
            ["ATR_Contraction", "ts_code"],
            ascending=[True, True],
            kind="mergesort",
        )
        r11_rank_map = pd.Series(
            np.arange(1, len(r11_ordered) + 1, dtype=int),
            index=r11_ordered.index,
        )
        candidates.loc[r11_rank_map.index, "R11_Strong_Rank"] = (
            r11_rank_map.astype(float)
        )
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

        # R12只增加一个弱势研究排名，不改R6实际排名：优先保留R1既有
        # Score_Pullback_15较低的深跌结构，同分时选择价格相对MA10修复更充分者。
        # 两项均在信号周已知；不设新阈值，不读取未来收益，不允许递补扩容。
        repair_ordered = recovery_eligible.sort_values(
            ["Score_Pullback_15", "Price_to_MA10_Ratio", "ts_code"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        repair_rank_map = pd.Series(
            np.arange(1, len(repair_ordered) + 1, dtype=int),
            index=repair_ordered.index,
        )
        candidates.loc[repair_rank_map.index, "R12_Recovery_Repair_Rank"] = (
            repair_rank_map.astype(float)
        )

        # R13只挑战R6弱势合格池的名次，不改变触发、资格、原名次或实际组合。
        # 五项日线特征均严格截止信号日，并固定等权；缺少完整日线快照的旧结果
        # 不允许按股票代码伪造R13名次。
        daily_available = _bool_series(
            recovery_eligible, "Daily_Restart_Data_Available"
        )
        daily_rows = recovery_eligible[daily_available].copy()
        if not daily_rows.empty:
            daily_scored = _score_daily_restart_quality(daily_rows)
            daily_columns = [
                "R13_Daily_Price_Repair_20",
                "R13_Daily_MA5_Slope_20",
                "R13_Daily_MACD_Accel_20",
                "R13_Daily_RS5_20",
                "R13_Daily_Higher_Low_20",
                "R13_Daily_Restart_100",
            ]
            for column in daily_columns:
                candidates.loc[daily_scored.index, column] = daily_scored[column]
            daily_ordered = daily_scored.sort_values(
                [
                    "R13_Daily_Restart_100",
                    "Daily_Close_to_MA20_Ratio",
                    "ts_code",
                ],
                ascending=[False, False, True],
                kind="mergesort",
            )
            daily_rank_map = pd.Series(
                np.arange(1, len(daily_ordered) + 1, dtype=int),
                index=daily_ordered.index,
            )
            candidates.loc[
                daily_rank_map.index, "R13_Daily_Restart_Rank"
            ] = daily_rank_map.astype(float)

            # R14不再组合五项日线分，只把R13预先声明的MACD柱加速度单因子
            # 单独排序为高弹性观察组。股票代码只负责完全同值时的稳定并列，
            # 此名次永远不能覆盖R6原实际排名。
            elastic_ordered = daily_rows.sort_values(
                ["Daily_MACD_Hist_Delta_pct", "ts_code"],
                ascending=[False, True],
                na_position="last",
                kind="mergesort",
            )
            elastic_rank_map = pd.Series(
                np.arange(1, len(elastic_ordered) + 1, dtype=int),
                index=elastic_ordered.index,
            )
            candidates.loc[
                elastic_rank_map.index, "R14_MACD_Elastic_Rank"
            ] = elastic_rank_map.astype(float)

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
        active_eligible_count = reacceleration_eligible_count
        candidates["Rank"] = candidates["R11_Strong_Rank"]
        candidates["Entry_Eligible"] = _bool_series(
            candidates, "Strong_Reacceleration_Eligible"
        )
        active_branch = "R11强势温和收缩Top1观察"
        block_reason = "R11仍为研究验证，强势市场实际强制空仓"
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
    candidates["R7_Research_Top2"] = (
        strong_reset_context_pass
        & _bool_series(candidates, "Strong_Eligible")
        & pd.to_numeric(candidates["Strong_Rank"], errors="coerce").le(TOP_N)
    )
    candidates["R9_Failure_Control_Top2"] = (
        strong_continuation_context_pass
        & _bool_series(candidates, "Strong_Reacceleration_Eligible")
        & _bool_series(candidates, "R9_Selection_Qualified")
    )
    r11_atr = pd.to_numeric(candidates["ATR_Contraction"], errors="coerce")
    candidates["R11_ATR_Band_Pass"] = r11_atr.between(
        R11_ATR_CONTRACTION_MIN,
        R11_ATR_CONTRACTION_MAX,
        inclusive="both",
    )
    candidates["R11_Strong_Research_Top1"] = (
        market_regime == "强势"
    ) & pd.to_numeric(candidates["R11_Strong_Rank"], errors="coerce").eq(1) & _bool_series(
        candidates, "R11_ATR_Band_Pass"
    )
    candidates["R12_Recovery_Repair_Top2"] = (
        (market_regime == "弱势")
        & (recovery_eligible_count >= MIN_VALID_SELECTION_SIZE)
        & _bool_series(candidates, "Recovery_Eligible")
        & pd.to_numeric(
            candidates["R12_Recovery_Repair_Rank"], errors="coerce"
        ).le(TOP_N)
    )
    r13_rank = pd.to_numeric(
        candidates["R13_Daily_Restart_Rank"], errors="coerce"
    )
    r13_available_count = int(
        (
            _bool_series(candidates, "Recovery_Eligible")
            & _bool_series(candidates, "Daily_Restart_Data_Available")
        ).sum()
    )
    r13_research_context = (
        (market_regime == "弱势")
        & (recovery_eligible_count >= MIN_VALID_SELECTION_SIZE)
        & (r13_available_count >= MIN_VALID_SELECTION_SIZE)
        & _bool_series(candidates, "Recovery_Eligible")
        & _bool_series(candidates, "Daily_Restart_Data_Available")
    )
    candidates["R13_Daily_Restart_Top1"] = r13_research_context & r13_rank.eq(1)
    candidates["R13_Daily_Restart_Top2"] = r13_research_context & r13_rank.le(
        TOP_N
    )
    r14_rank = pd.to_numeric(candidates["R14_MACD_Elastic_Rank"], errors="coerce")
    candidates["R14_MACD_Elastic_Top1"] = r13_research_context & r14_rank.eq(1)
    candidates["R14_MACD_Elastic_Top2"] = r13_research_context & r14_rank.le(
        TOP_N
    )
    if selection_valid:
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
    candidates["R13_Daily_Available_Count"] = r13_available_count
    candidates["R14_MACD_Elastic_Count"] = r13_available_count
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
        "R14_W1_Close_Gross_pct": np.nan,
        "R14_W2_Close_Gross_pct": np.nan,
        "R14_Trigger_W1_Close_Loss": np.nan,
        "R14_Trigger_W1_W2_Both_Loss": np.nan,
        "R14_Trigger_W2_Close_Minus5": np.nan,
        "R14_W1_Next_Open_Exit_Date": None,
        "R14_W2_Next_Open_Exit_Date": None,
        "R14_W1_Next_Open_Return_Net_pct": np.nan,
        "R14_W2_Next_Open_Return_Net_pct": np.nan,
        "R14_W1_Exit_Delay_Days": np.nan,
        "R14_W2_Exit_Delay_Days": np.nan,
        "R14_Lifecycle_Data_Available": False,
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
    marked_close = pd.to_numeric(future["close"], errors="coerce").ffill()
    for week in range(1, HOLD_WEEKS + 1):
        day_index = week * MARKET_DAYS_PER_WEEK - 1
        if len(future) > day_index:
            exit_close = _safe_float(marked_close.iloc[day_index])
            if math.isfinite(exit_close):
                result[f"Fixed_Return_W{week}_Net_pct"] = (
                    (exit_close / buy_price - 1.0) * 100.0 - roundtrip_cost_pct
                )

    # R14退出只在周末收盘确认后，于下一可交易日开盘成交。停牌和一字跌停
    # 不假设能够卖出，而是顺延到第一天真实可交易的开盘价。
    def _next_tradable_exit(decision_day_index: int):
        previous_raw_close = signal_raw_close
        for position, (_, exit_row) in enumerate(future.iterrows()):
            raw_close = _safe_float(
                exit_row.get("raw_close"), _safe_float(exit_row.get("close"))
            )
            if position <= decision_day_index:
                if math.isfinite(raw_close) and raw_close > 0:
                    previous_raw_close = raw_close
                continue
            exit_open = _safe_float(exit_row.get("open"))
            raw_open = _safe_float(exit_row.get("raw_open"), exit_open)
            raw_high = _safe_float(
                exit_row.get("raw_high"), _safe_float(exit_row.get("high"))
            )
            raw_low = _safe_float(
                exit_row.get("raw_low"), _safe_float(exit_row.get("low"))
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
                    atol=max(0.001, raw_open * 1e-5),
                )
                and (raw_close / previous_raw_close - 1.0) <= -limit_threshold
            )
            if math.isfinite(exit_open) and exit_open > 0 and not one_price_limit_down:
                return (
                    str(future.index[position]),
                    (exit_open / buy_price - 1.0) * 100.0 - roundtrip_cost_pct,
                    position - decision_day_index - 1,
                )
            if math.isfinite(raw_close) and raw_close > 0:
                previous_raw_close = raw_close
        return None, np.nan, np.nan

    if len(future) >= MARKET_DAYS_PER_WEEK:
        w1_close = _safe_float(marked_close.iloc[MARKET_DAYS_PER_WEEK - 1])
        if math.isfinite(w1_close):
            w1_gross = (w1_close / buy_price - 1.0) * 100.0
            result["R14_W1_Close_Gross_pct"] = w1_gross
            result["R14_Trigger_W1_Close_Loss"] = bool(w1_gross < 0.0)
            exit_date, exit_return, delay_days = _next_tradable_exit(
                MARKET_DAYS_PER_WEEK - 1
            )
            result["R14_W1_Next_Open_Exit_Date"] = exit_date
            result["R14_W1_Next_Open_Return_Net_pct"] = exit_return
            result["R14_W1_Exit_Delay_Days"] = delay_days
    if len(future) >= 2 * MARKET_DAYS_PER_WEEK:
        w2_close = _safe_float(marked_close.iloc[2 * MARKET_DAYS_PER_WEEK - 1])
        w1_gross = _safe_float(result.get("R14_W1_Close_Gross_pct"))
        if math.isfinite(w2_close):
            w2_gross = (w2_close / buy_price - 1.0) * 100.0
            result["R14_W2_Close_Gross_pct"] = w2_gross
            result["R14_Trigger_W1_W2_Both_Loss"] = bool(
                math.isfinite(w1_gross) and w1_gross < 0.0 and w2_gross < 0.0
            )
            result["R14_Trigger_W2_Close_Minus5"] = bool(w2_gross <= -5.0)
            exit_date, exit_return, delay_days = _next_tradable_exit(
                2 * MARKET_DAYS_PER_WEEK - 1
            )
            result["R14_W2_Next_Open_Exit_Date"] = exit_date
            result["R14_W2_Next_Open_Return_Net_pct"] = exit_return
            result["R14_W2_Exit_Delay_Days"] = delay_days
            result["R14_Lifecycle_Data_Available"] = bool(
                math.isfinite(_safe_float(exit_return))
                and math.isfinite(w1_gross)
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
        "强势": "R11强势温和收缩Top1观察",
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
                    == "R9失败对照观察"
                    and _safe_float(candidate_row.get("Rank")) == 2.0
                    and not bool(candidate_row.get("R9_Second_Qualified", False))
                )
                miss_reason = (
                    "R9第二名未通过分数或MA20距离独立资格"
                    if is_r9_second
                    else "R11强势只研究Top1，实际组合强制空仓"
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
                "R11_Strong_Research_Top1": bool(
                    candidate_row is not None
                    and _bool_series(
                        pd.DataFrame([candidate_row]), "R11_Strong_Research_Top1"
                    ).iloc[0]
                ),
                "R12_Recovery_Repair_Rank": (
                    candidate_row.get("R12_Recovery_Repair_Rank")
                    if candidate_row is not None
                    else np.nan
                ),
                "R12_Recovery_Repair_Top2": bool(
                    candidate_row is not None
                    and _bool_series(
                        pd.DataFrame([candidate_row]),
                        "R12_Recovery_Repair_Top2",
                    ).iloc[0]
                ),
                "R13_Daily_Restart_Rank": (
                    candidate_row.get("R13_Daily_Restart_Rank")
                    if candidate_row is not None
                    else np.nan
                ),
                "R13_Daily_Restart_Top1": bool(
                    candidate_row is not None
                    and _bool_series(
                        pd.DataFrame([candidate_row]),
                        "R13_Daily_Restart_Top1",
                    ).iloc[0]
                ),
                "R13_Daily_Restart_Top2": bool(
                    candidate_row is not None
                    and _bool_series(
                        pd.DataFrame([candidate_row]),
                        "R13_Daily_Restart_Top2",
                    ).iloc[0]
                ),
                "R14_MACD_Elastic_Rank": (
                    candidate_row.get("R14_MACD_Elastic_Rank")
                    if candidate_row is not None
                    else np.nan
                ),
                "R14_MACD_Elastic_Top1": bool(
                    candidate_row is not None
                    and _bool_series(
                        pd.DataFrame([candidate_row]),
                        "R14_MACD_Elastic_Top1",
                    ).iloc[0]
                ),
                "R14_MACD_Elastic_Top2": bool(
                    candidate_row is not None
                    and _bool_series(
                        pd.DataFrame([candidate_row]),
                        "R14_MACD_Elastic_Top2",
                    ).iloc[0]
                ),
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
                "Score_Pullback_15": (
                    candidate_row.get("Score_Pullback_15")
                    if candidate_row is not None
                    else np.nan
                ),
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
    remove_mask = same_range & (recent | status.ne("COMPLETED"))
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
        return pd.DataFrame(), pd.DataFrame(), 0, 0
    pool = pd.DataFrame(pool_records)
    candidates, raw_count, eligible_count = score_r13_candidates(pool)

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
# R13实际失败基线与三项研究排名报告
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
    """R7抗跌新高只做研究判卷；任何强势候选都不计入实际组合。"""
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
    research = _bool_series(frame, "R7_Research_Top2")
    eligible = _bool_series(frame, "Strong_Eligible")
    context_pass = _bool_series(frame, "Strong_Reset_Context_Pass")
    masks = [
        ("R7研究Top2（不交易）", research),
        ("早期强势回调合格但未进研究Top2", eligible & context_pass & ~research),
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
                "R7动作": (
                    "研究Top2（不交易）"
                    if str(context) == "早期强势回调"
                    else "只观察"
                ),
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
    """同一批强势周比较R7研究组与R3首红基线，不改变实际空仓结果。"""
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
    research = _bool_series(frame, "R7_Research_Top2")
    r7_all_contexts = _bool_series(frame, "Strong_Eligible") & strong_rank.le(TOP_N)
    r3_baseline = _bool_series(frame, "Trend_Eligible") & r3_rank.le(TOP_N)
    groups = [
        ("R7早期回调研究Top2（不交易）", research),
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
    """R9再启动只作为失败对照判卷，不再产生实际交易。"""
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
    control = _bool_series(frame, "R9_Failure_Control_Top2")
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
        ("R9失败对照Top2（不交易）", control),
        ("R9阶段内合格但未进失败对照Top2", eligible & context_pass & ~control),
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
    """在同一R9强势阶段比较失败对照、强制Top2、R7和R3。"""
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
    control = _bool_series(frame, "R9_Failure_Control_Top2")
    r9_eligible = _bool_series(frame, "Strong_Reacceleration_Eligible")
    groups = [
        ("R9失败对照最多Top2", control),
        ("R9失败对照第一名", control & reacceleration_rank.eq(1)),
        ("R9失败对照第二名", control & reacceleration_rank.eq(2)),
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


def r11_strong_candidate_audit(history: pd.DataFrame):
    """R11温和ATR收缩Top1独立判卷；全部样本均为研究观察。"""
    if history.empty or "R11_Strong_Rank" not in history.columns:
        return pd.DataFrame()
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & history.get("Market_Regime", pd.Series("", index=history.index))
        .astype(str)
        .eq("强势")
        & _bool_series(history, "Strong_Reacceleration_Eligible")
    ].copy()
    frame["R11_Strong_Rank"] = pd.to_numeric(
        frame.get("R11_Strong_Rank"), errors="coerce"
    )
    frame["ATR_Contraction"] = pd.to_numeric(
        frame.get("ATR_Contraction"), errors="coerce"
    )
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    rank1 = frame["R11_Strong_Rank"].eq(1)
    research = _bool_series(frame, "R11_Strong_Research_Top1")
    selected_dates = set(frame.loc[research, "Signal_Date"].astype(str))
    same_week_rank2 = (
        frame["R11_Strong_Rank"].eq(2)
        & frame["Signal_Date"].astype(str).isin(selected_dates)
    )
    groups = [
        ("R11温和收缩研究Top1（不交易）", research),
        (
            f"Top1收缩过度（ATR比<{R11_ATR_CONTRACTION_MIN:.2f}）",
            rank1 & frame["ATR_Contraction"].lt(R11_ATR_CONTRACTION_MIN),
        ),
        (
            f"Top1收缩不足（ATR比>{R11_ATR_CONTRACTION_MAX:.2f}）",
            rank1 & frame["ATR_Contraction"].gt(R11_ATR_CONTRACTION_MAX),
        ),
        ("研究信号周同池第二名", same_week_rank2),
        ("全部强势周ATR最小第一名", rank1),
    ]
    rows = []
    for label, mask in groups:
        group = frame[mask & frame[PRIMARY_RETURN_COLUMN].notna()]
        returns = group[PRIMARY_RETURN_COLUMN]
        rows.append(
            {
                "R11强势研究组": label,
                "样本数": len(returns),
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
        action = "全部强势阶段实际空仓；R11温和收缩Top1仅研究观察"
        rows.append(
            {
                "强势市场阶段": stage,
                "R11动作": action,
                "市场周数": group["Signal_Date"].nunique(),
                "R7合格股票周数": int(_bool_series(group, "Strong_Eligible").sum()),
                "R9合格股票周数": int(
                    _bool_series(group, "Strong_Reacceleration_Eligible").sum()
                ),
                "R7研究Top2": int(_bool_series(group, "R7_Research_Top2").sum()),
                "R9失败对照Top2": int(
                    _bool_series(group, "R9_Failure_Control_Top2").sum()
                ),
                "R11研究Top1": int(
                    _bool_series(group, "R11_Strong_Research_Top1").sum()
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


def _r12_research_performance_row(label: str, group: pd.DataFrame):
    returns = pd.to_numeric(
        group.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    valid = group.loc[returns.index].copy() if len(returns) else group.iloc[0:0].copy()
    without_best = (
        returns.drop(index=returns.idxmax())
        if len(returns) > 1
        else pd.Series(dtype=float)
    )
    positive_total = returns[returns > 0.0].sum()
    best_contribution = (
        returns.max() / positive_total * 100.0
        if len(returns) and positive_total > 0.0
        else np.nan
    )
    return {
        "R12弱势修复研究组": label,
        "样本数": len(returns),
        "信号周": valid["Signal_Date"].nunique() if len(returns) else 0,
        "胜率%": (returns > 0.0).mean() * 100.0 if len(returns) else np.nan,
        "中位收益%": returns.median() if len(returns) else np.nan,
        "平均收益%": returns.mean() if len(returns) else np.nan,
        "Profit_Factor": _profit_factor(returns),
        "去最佳一只平均收益%": (
            without_best.mean() if len(without_best) else np.nan
        ),
        "去最佳一只PF": _profit_factor(without_best),
        "最佳一只占正利润%": best_contribution,
    }


def r12_recovery_repair_candidate_audit(history: pd.DataFrame):
    """R12只判卷弱势R6合格池的新研究名次，不改R6原实际收益。"""
    columns = [
        "R12弱势修复研究组",
        "样本数",
        "信号周",
        "胜率%",
        "中位收益%",
        "平均收益%",
        "Profit_Factor",
        "去最佳一只平均收益%",
        "去最佳一只PF",
        "最佳一只占正利润%",
    ]
    if history.empty or "R12_Recovery_Repair_Rank" not in history.columns:
        return pd.DataFrame(columns=columns)
    weak = history.get(
        "Market_Regime", pd.Series("", index=history.index)
    ).astype(str).eq("弱势")
    base = history[weak & _bool_series(history, "Recovery_Eligible")].copy()
    if base.empty:
        return pd.DataFrame(columns=columns)
    base["R12_Recovery_Repair_Rank"] = pd.to_numeric(
        base.get("R12_Recovery_Repair_Rank"), errors="coerce"
    )
    complete = _bool_series(base, "Outcome_Complete") & _bool_series(
        base, "Entry_Tradable"
    )
    evaluable = base[complete].copy()
    research = _bool_series(evaluable, "R12_Recovery_Repair_Top2")
    research_dates = set(
        evaluable.loc[research, "Signal_Date"].astype(str).tolist()
    )
    same_week = evaluable["Signal_Date"].astype(str).isin(research_dates)
    old_actual = (
        _bool_series(evaluable, "Selected_Top2")
        & evaluable.get(
            "Strategy_Branch", pd.Series("", index=evaluable.index)
        ).astype(str).eq("R6弱势首次转折-N6")
    )
    groups = [
        (
            "R12深跌结构优先、价格修复并列项Top2（只研究）",
            evaluable[research],
        ),
        (
            "R12研究Top1",
            evaluable[
                research & evaluable["R12_Recovery_Repair_Rank"].eq(1)
            ],
        ),
        (
            "R12研究Top2中的第二名",
            evaluable[
                research & evaluable["R12_Recovery_Repair_Rank"].eq(2)
            ],
        ),
        ("R6原早期阶段排名Top2（实际基线）", evaluable[old_actual]),
        (
            "R12信号周其余弱势合格候选",
            evaluable[same_week & ~research],
        ),
    ]
    return pd.DataFrame(
        [_r12_research_performance_row(label, group) for label, group in groups],
        columns=columns,
    )


def _r12_top2_by_sort(
    frame: pd.DataFrame,
    columns: list[str],
    ascending: list[bool],
):
    if frame.empty or any(column not in frame.columns for column in columns):
        return frame.iloc[0:0].copy()
    ordered = frame.copy()
    for column in columns:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    ordered["_code"] = ordered.get(
        "ts_code", pd.Series("", index=ordered.index)
    ).astype(str)
    ordered = ordered.sort_values(
        ["Signal_Date"] + columns + ["_code"],
        ascending=[True] + ascending + [True],
        na_position="last",
        kind="mergesort",
    )
    return ordered.groupby("Signal_Date", sort=False).head(TOP_N)


def r12_recovery_factor_comparison_audit(history: pd.DataFrame):
    """固定比较四种预先声明的弱势排名，避免只展示胜出的R12规则。"""
    if history.empty:
        return pd.DataFrame()
    weak = history.get(
        "Market_Regime", pd.Series("", index=history.index)
    ).astype(str).eq("弱势")
    eligible = history[weak & _bool_series(history, "Recovery_Eligible")].copy()
    if eligible.empty:
        return pd.DataFrame()
    counts = eligible.groupby("Signal_Date", sort=False).size()
    valid_dates = set(counts[counts >= MIN_VALID_SELECTION_SIZE].index.astype(str))
    eligible = eligible[
        eligible["Signal_Date"].astype(str).isin(valid_dates)
    ].copy()
    old_rank = pd.to_numeric(eligible.get("Recovery_Rank"), errors="coerce")
    groups = [
        ("R6原五项越早越好排名", eligible[old_rank.le(TOP_N)]),
        (
            "R12深跌结构优先、价格修复打破并列",
            _r12_top2_by_sort(
                eligible,
                ["Score_Pullback_15", "Price_to_MA10_Ratio"],
                [True, False],
            ),
        ),
        (
            "仅深跌结构分从低到高",
            _r12_top2_by_sort(eligible, ["Score_Pullback_15"], [True]),
        ),
        (
            "仅价格相对MA10从高到低",
            _r12_top2_by_sort(eligible, ["Price_to_MA10_Ratio"], [False]),
        ),
    ]
    rows = []
    for label, group in groups:
        evaluable = group[
            _bool_series(group, "Outcome_Complete")
            & _bool_series(group, "Entry_Tradable")
        ].copy()
        row = _r12_research_performance_row(label, evaluable)
        row["弱势排名方案"] = row.pop("R12弱势修复研究组")
        rows.append(row)
    return pd.DataFrame(rows)


def _r13_research_performance_row(label: str, group: pd.DataFrame):
    returns = pd.to_numeric(
        group.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    valid = group.loc[returns.index].copy() if len(returns) else group.iloc[0:0].copy()
    without_best = (
        returns.drop(index=returns.idxmax())
        if len(returns) > 1
        else pd.Series(dtype=float)
    )
    positive_total = returns[returns > 0.0].sum()
    best_contribution = (
        returns.max() / positive_total * 100.0
        if len(returns) and positive_total > 0.0
        else np.nan
    )
    return {
        "R13日线重启研究组": label,
        "样本数": len(returns),
        "信号周": valid["Signal_Date"].nunique() if len(returns) else 0,
        "胜率%": (returns > 0.0).mean() * 100.0 if len(returns) else np.nan,
        "中位收益%": returns.median() if len(returns) else np.nan,
        "平均收益%": returns.mean() if len(returns) else np.nan,
        "Profit_Factor": _profit_factor(returns),
        "去最佳一只平均收益%": (
            without_best.mean() if len(without_best) else np.nan
        ),
        "去最佳一只PF": _profit_factor(without_best),
        "最佳一只占正利润%": best_contribution,
    }


def r13_daily_restart_candidate_audit(history: pd.DataFrame):
    """R13只判卷同一R6合格池的日线挑战名次，不改变实际入选。"""
    columns = [
        "R13日线重启研究组",
        "样本数",
        "信号周",
        "胜率%",
        "中位收益%",
        "平均收益%",
        "Profit_Factor",
        "去最佳一只平均收益%",
        "去最佳一只PF",
        "最佳一只占正利润%",
    ]
    if history.empty or "R13_Daily_Restart_Rank" not in history.columns:
        return pd.DataFrame(columns=columns)
    weak = history.get(
        "Market_Regime", pd.Series("", index=history.index)
    ).astype(str).eq("弱势")
    base = history[
        weak
        & _bool_series(history, "Recovery_Eligible")
        & _bool_series(history, "Daily_Restart_Data_Available")
    ].copy()
    if base.empty:
        return pd.DataFrame(columns=columns)
    base["R13_Daily_Restart_Rank"] = pd.to_numeric(
        base.get("R13_Daily_Restart_Rank"), errors="coerce"
    )
    evaluable = base[
        _bool_series(base, "Outcome_Complete")
        & _bool_series(base, "Entry_Tradable")
    ].copy()
    research = _bool_series(evaluable, "R13_Daily_Restart_Top2")
    research_dates = set(
        evaluable.loc[research, "Signal_Date"].astype(str).tolist()
    )
    same_week = evaluable["Signal_Date"].astype(str).isin(research_dates)
    old_actual = (
        _bool_series(evaluable, "Selected_Top2")
        & evaluable.get(
            "Strategy_Branch", pd.Series("", index=evaluable.index)
        ).astype(str).eq("R6弱势首次转折-N6")
        & same_week
    )
    r12 = _bool_series(evaluable, "R12_Recovery_Repair_Top2") & same_week
    groups = [
        ("R13五项日线重启Top2（只研究）", evaluable[research]),
        (
            "R13日线重启Top1",
            evaluable[research & evaluable["R13_Daily_Restart_Rank"].eq(1)],
        ),
        (
            "R13日线重启第二名",
            evaluable[research & evaluable["R13_Daily_Restart_Rank"].eq(2)],
        ),
        ("同周R6原排名Top2（实际基线）", evaluable[old_actual]),
        ("同周R12弱势修复Top2（研究基线）", evaluable[r12]),
        ("R13同周其余弱势合格候选", evaluable[same_week & ~research]),
    ]
    return pd.DataFrame(
        [_r13_research_performance_row(label, group) for label, group in groups],
        columns=columns,
    )


def r13_daily_restart_factor_comparison_audit(history: pd.DataFrame):
    """预先固定五种同周排名，完整展示失败对照，避免事后挑选因子。"""
    if history.empty:
        return pd.DataFrame()
    weak = history.get(
        "Market_Regime", pd.Series("", index=history.index)
    ).astype(str).eq("弱势")
    eligible = history[
        weak
        & _bool_series(history, "Recovery_Eligible")
        & _bool_series(history, "Daily_Restart_Data_Available")
    ].copy()
    if eligible.empty:
        return pd.DataFrame()
    counts = eligible.groupby("Signal_Date", sort=False).size()
    valid_dates = set(counts[counts >= MIN_VALID_SELECTION_SIZE].index.astype(str))
    eligible = eligible[
        eligible["Signal_Date"].astype(str).isin(valid_dates)
    ].copy()
    old_rank = pd.to_numeric(eligible.get("Recovery_Rank"), errors="coerce")
    r12_rank = pd.to_numeric(
        eligible.get("R12_Recovery_Repair_Rank"), errors="coerce"
    )
    groups = [
        ("R6原五项早期阶段排名", eligible[old_rank.le(TOP_N)]),
        ("R12深跌+周线价格修复排名", eligible[r12_rank.le(TOP_N)]),
        (
            "R13五项日线重启等权排名",
            _r12_top2_by_sort(
                eligible,
                ["R13_Daily_Restart_100", "Daily_Close_to_MA20_Ratio"],
                [False, False],
            ),
        ),
        (
            "仅日线价格/MA20从高到低",
            _r12_top2_by_sort(
                eligible, ["Daily_Close_to_MA20_Ratio"], [False]
            ),
        ),
        (
            "仅日线MACD柱加速度从高到低",
            _r12_top2_by_sort(
                eligible, ["Daily_MACD_Hist_Delta_pct"], [False]
            ),
        ),
        (
            "仅全池日线五日相对强度从高到低",
            _r12_top2_by_sort(eligible, ["Daily_RS_5D_Pct"], [False]),
        ),
        (
            "仅日线五日低点抬升从高到低",
            _r12_top2_by_sort(
                eligible, ["Daily_Higher_Low_5D_pct"], [False]
            ),
        ),
    ]
    rows = []
    for label, group in groups:
        evaluable = group[
            _bool_series(group, "Outcome_Complete")
            & _bool_series(group, "Entry_Tradable")
        ].copy()
        row = _r13_research_performance_row(label, evaluable)
        row["弱势排名方案"] = row.pop("R13日线重启研究组")
        rows.append(row)
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
    r12_selected = frame[_bool_series(frame, "R12_Recovery_Repair_Top2")]
    missed = frame[frame["Detection_Status"].astype(str).eq("完全未发现")]
    groups = [
        (f"全部未来W3≥{MAJOR_WINNER_W3_PCT:.0f}%机会", frame),
        ("任一结构已发现", detected),
        ("当周分支合格", eligible),
        ("最终实际入选", selected),
        ("R12弱势修复研究Top2命中", r12_selected),
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
            action = "全部强势周持有现金"
        rows.append(
            {
                "市场状态": regime,
                "R11动作": action,
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
        ("一年内至少18个独立R3/R6实际信号周", weeks >= 18, f"当前{weeks}周"),
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


def neutral_research_gates(completed: pd.DataFrame):
    """R3中性分支单独验收，避免总体收益掩盖跨阶段失效。"""
    if completed.empty or "Strategy_Branch" not in completed.columns:
        selected = pd.DataFrame()
    else:
        selected = completed[
            completed["Strategy_Branch"].astype(str).eq("R3中性趋势")
            & _actual_selected_mask(completed)
        ].copy()
    returns = pd.to_numeric(
        selected.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    valid = selected.loc[returns.index].sort_values(["Signal_Date", "Rank"])
    weeks = valid["Signal_Date"].nunique() if len(valid) else 0
    pf = _profit_factor(returns)
    without_best = (
        returns.drop(index=returns.idxmax())
        if len(returns) > 1
        else pd.Series(dtype=float)
    )
    ordered_returns = pd.to_numeric(
        valid.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    midpoint = len(ordered_returns) // 2
    first_half = ordered_returns.iloc[:midpoint]
    second_half = ordered_returns.iloc[midpoint:]
    gates = [
        ("R3至少12个独立信号周", weeks >= 12, f"当前{weeks}周"),
        ("R3至少24笔完整交易", len(returns) >= 24, f"当前{len(returns)}笔"),
        (
            "R3 W3胜率至少55%",
            len(returns) > 0 and (returns > 0).mean() >= 0.55,
            f"当前{((returns > 0).mean() * 100.0 if len(returns) else np.nan):.1f}%",
        ),
        (
            "R3 W3中位收益大于0",
            len(returns) > 0 and returns.median() > 0.0,
            f"当前{(returns.median() if len(returns) else np.nan):.2f}%",
        ),
        (
            "R3 Profit Factor至少1.2",
            pd.notna(pf) and pf >= 1.2,
            f"当前{pf:.2f}",
        ),
        (
            "R3去最佳一只后平均收益大于0",
            len(without_best) > 0 and without_best.mean() > 0.0,
            f"当前{(without_best.mean() if len(without_best) else np.nan):.2f}%",
        ),
        (
            "R3前后半段中位收益均为正",
            len(first_half) > 0
            and len(second_half) > 0
            and first_half.median() > 0.0
            and second_half.median() > 0.0,
            (
                f"前{(first_half.median() if len(first_half) else np.nan):.2f}% / "
                f"后{(second_half.median() if len(second_half) else np.nan):.2f}%"
            ),
        ),
    ]
    return pd.DataFrame(
        [
            {"R3验收项目": name, "结果": "通过" if passed else "未通过", "当前值": value}
            for name, passed, value in gates
        ]
    )


def strong_research_gates(history: pd.DataFrame):
    """R7研究Top2单独判卷；结果仅决定是否继续研究，不影响实际组合。"""
    if history.empty:
        selected = pd.DataFrame()
    else:
        selected = history[
            _bool_series(history, "Outcome_Complete")
            & _bool_series(history, "Entry_Tradable")
            & _bool_series(history, "R7_Research_Top2")
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
        ("R7研究组至少6个独立信号周", weeks >= 6, f"当前{weeks}周"),
        ("R7研究组至少12笔完整样本", len(returns) >= 12, f"当前{len(returns)}笔"),
        (
            "R7研究组W3胜率至少55%",
            len(returns) > 0 and (returns > 0).mean() >= 0.55,
            f"当前{((returns > 0).mean() * 100.0 if len(returns) else np.nan):.1f}%",
        ),
        (
            "R7研究组W3中位收益大于0",
            len(returns) > 0 and returns.median() > 0.0,
            f"当前{(returns.median() if len(returns) else np.nan):.2f}%",
        ),
        (
            "R7研究组去最佳一只后平均收益大于0",
            len(without_best) > 0 and without_best.mean() > 0.0,
            f"当前{(without_best.mean() if len(without_best) else np.nan):.2f}%",
        ),
        (
            "R7研究组PF至少1.2",
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


def r11_strong_ranking_diagnostics(history: pd.DataFrame):
    if history.empty:
        return {}
    frame = history[
        _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & history.get("Market_Regime", pd.Series("", index=history.index))
        .astype(str)
        .eq("强势")
        & _bool_series(history, "Strong_Reacceleration_Eligible")
    ].copy()
    if frame.empty:
        return {}
    frame["Rank"] = pd.to_numeric(frame.get("R11_Strong_Rank"), errors="coerce")
    frame["Selected_Top2"] = _bool_series(frame, "R11_Strong_Research_Top1")
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    frame = frame.dropna(subset=["Rank", PRIMARY_RETURN_COLUMN])
    return ranking_diagnostics(frame)


def r11_strong_research_gates(history: pd.DataFrame):
    """R11强势Top1独立验收；仅判定是否值得继续研究，不影响实际组合。"""
    if history.empty:
        selected = pd.DataFrame()
        rank2 = pd.DataFrame()
    else:
        complete = _bool_series(history, "Outcome_Complete")
        tradable = _bool_series(history, "Entry_Tradable")
        research = _bool_series(history, "R11_Strong_Research_Top1")
        selected = history[complete & tradable & research].copy()
        selected_dates = set(selected.get("Signal_Date", pd.Series(dtype=str)).astype(str))
        r11_rank = pd.to_numeric(history.get("R11_Strong_Rank"), errors="coerce")
        rank2 = history[
            complete
            & tradable
            & r11_rank.eq(2)
            & history.get("Signal_Date", pd.Series("", index=history.index))
            .astype(str)
            .isin(selected_dates)
        ].copy()
    selected = selected.sort_values(["Signal_Date", "R11_Strong_Rank"])
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
    without_best_pf = _profit_factor(without_best)
    positive_profit = returns[returns > 0].sum()
    best_contribution = (
        returns.max() / positive_profit * 100.0
        if len(returns) and positive_profit > 0
        else np.nan
    )
    midpoint = len(returns) // 2
    first_half = returns.iloc[:midpoint]
    second_half = returns.iloc[midpoint:]
    rank2_returns = pd.to_numeric(
        rank2.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    rank2_median = rank2_returns.median() if len(rank2_returns) else np.nan
    gates = [
        ("R11强势研究信号至少8周", weeks >= 8, f"当前{weeks}周"),
        ("R11强势研究信号不超过14周", weeks <= 14, f"当前{weeks}周"),
        ("R11每个信号周严格只有Top1", len(returns) == weeks, f"当前{len(returns)}笔/{weeks}周"),
        (
            "R11 W3胜率至少50%",
            len(returns) > 0 and (returns > 0).mean() >= 0.50,
            f"当前{((returns > 0).mean() * 100.0 if len(returns) else np.nan):.1f}%",
        ),
        (
            "R11 W3中位收益大于0",
            len(returns) > 0 and returns.median() > 0.0,
            f"当前{(returns.median() if len(returns) else np.nan):.2f}%",
        ),
        (
            "R11 Profit Factor至少1.2",
            pd.notna(pf) and pf >= 1.2,
            f"当前{pf:.2f}",
        ),
        (
            "R11去最佳一只后平均收益大于0",
            len(without_best) > 0 and without_best.mean() > 0.0,
            f"当前{(without_best.mean() if len(without_best) else np.nan):.2f}%",
        ),
        (
            "R11去最佳一只后PF至少1.2",
            pd.notna(without_best_pf) and without_best_pf >= 1.2,
            f"当前{without_best_pf:.2f}",
        ),
        (
            "R11最佳一只占正利润不超过40%",
            pd.notna(best_contribution) and best_contribution <= 40.0,
            f"当前{best_contribution:.1f}%",
        ),
        (
            "R11前后半段中位收益均为正",
            len(first_half) > 0
            and len(second_half) > 0
            and first_half.median() > 0.0
            and second_half.median() > 0.0,
            (
                f"前{(first_half.median() if len(first_half) else np.nan):.2f}% / "
                f"后{(second_half.median() if len(second_half) else np.nan):.2f}%"
            ),
        ),
        (
            "R11 Top1中位收益高于同周第二名",
            len(returns) > 0
            and len(rank2_returns) > 0
            and returns.median() > rank2_median,
            (
                f"Top1{(returns.median() if len(returns) else np.nan):.2f}% / "
                f"第二名{rank2_median:.2f}%"
            ),
        ),
    ]
    return pd.DataFrame(
        [
            {"R11验收项目": name, "结果": "通过" if passed else "未通过", "当前值": value}
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
    frame["Selected_Top2"] = _bool_series(frame, "R9_Failure_Control_Top2")
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    frame = frame.dropna(subset=["Rank", PRIMARY_RETURN_COLUMN])
    return ranking_diagnostics(frame)


def reacceleration_research_gates(
    history: pd.DataFrame,
    diagnostics: dict[str, Any],
):
    """R9失败对照独立判卷；结果不影响R11实际组合。"""
    if history.empty:
        selected = pd.DataFrame()
    else:
        selected = history[
            _bool_series(history, "Outcome_Complete")
            & _bool_series(history, "Entry_Tradable")
            & _bool_series(history, "R9_Failure_Control_Top2")
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
        ("R9失败对照至少6个独立信号周", weeks >= 6, f"当前{weeks}周"),
        ("R9失败对照至少12笔完整样本", len(returns) >= 12, f"当前{len(returns)}笔"),
        (
            "R9失败对照W3胜率至少55%",
            len(returns) > 0 and (returns > 0).mean() >= 0.55,
            f"当前{((returns > 0).mean() * 100.0 if len(returns) else np.nan):.1f}%",
        ),
        (
            "R9失败对照W3中位收益大于0",
            len(returns) > 0 and returns.median() > 0.0,
            f"当前{(returns.median() if len(returns) else np.nan):.2f}%",
        ),
        (
            "R9失败对照去最佳一只后平均收益大于0",
            len(without_best) > 0 and without_best.mean() > 0.0,
            f"当前{(without_best.mean() if len(without_best) else np.nan):.2f}%",
        ),
        (
            "R9失败对照去最佳整周后均益为正且PF至少1.2",
            len(without_best_week) > 0
            and without_best_week.mean() > 0.0
            and pd.notna(without_best_week_pf)
            and without_best_week_pf >= 1.2,
            (
                f"均益{(without_best_week.mean() if len(without_best_week) else np.nan):.2f}% / "
                f"PF{without_best_week_pf:.2f}"
            ),
        ),
        ("R9失败对照PF至少1.2", pd.notna(pf) and pf >= 1.2, f"当前{pf:.2f}"),
        (
            "R9失败对照逐周排名收益秩相关至少0.05",
            math.isfinite(weekly_corr) and weekly_corr >= 0.05,
            f"当前{weekly_corr:.3f}",
        ),
        (
            "R9失败对照逐周战胜其余候选至少55%",
            math.isfinite(paired_beat) and paired_beat >= 55.0,
            f"当前{paired_beat:.1f}%",
        ),
        (
            "R9失败对照中位收益优于其余候选",
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


def r12_recovery_repair_ranking_diagnostics(history: pd.DataFrame):
    if history.empty or "R12_Recovery_Repair_Rank" not in history.columns:
        return {}
    weak = history.get(
        "Market_Regime", pd.Series("", index=history.index)
    ).astype(str).eq("弱势")
    frame = history[
        weak
        & _bool_series(history, "Recovery_Eligible")
        & _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
    ].copy()
    frame["Rank"] = pd.to_numeric(
        frame.get("R12_Recovery_Repair_Rank"), errors="coerce"
    )
    frame["Selected_Top2"] = _bool_series(
        frame, "R12_Recovery_Repair_Top2"
    )
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    frame = frame.dropna(subset=["Rank", PRIMARY_RETURN_COLUMN])
    return ranking_diagnostics(frame)


def r12_recovery_repair_gates(history: pd.DataFrame):
    """R12独立验收；通过只代表值得继续样本外研究，不改变实际组合。"""
    if history.empty:
        selected = pd.DataFrame()
    else:
        selected = history[
            _bool_series(history, "R12_Recovery_Repair_Top2")
            & _bool_series(history, "Outcome_Complete")
            & _bool_series(history, "Entry_Tradable")
        ].copy()
    returns = pd.to_numeric(
        selected.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    selected = selected.loc[returns.index].copy() if len(returns) else selected.iloc[0:0]
    weeks = selected["Signal_Date"].nunique() if len(returns) else 0
    pf = _profit_factor(returns)
    without_best = (
        returns.drop(index=returns.idxmax())
        if len(returns) > 1
        else pd.Series(dtype=float)
    )
    without_best_pf = _profit_factor(without_best)
    positive_total = returns[returns > 0.0].sum()
    best_contribution = (
        returns.max() / positive_total * 100.0
        if len(returns) and positive_total > 0.0
        else np.nan
    )
    diagnostics = r12_recovery_repair_ranking_diagnostics(history)
    first_median = _safe_float(diagnostics.get("前半段实际入选中位收益%"))
    second_median = _safe_float(diagnostics.get("后半段实际入选中位收益%"))
    paired_beat = _safe_float(
        diagnostics.get("实际入选逐周平均收益战胜未入选比例%")
    )
    selected_median = _safe_float(diagnostics.get("实际入选中位收益%"))
    rest_median = _safe_float(diagnostics.get("未入选候选中位收益%"))
    old_actual = history[
        _bool_series(history, "Selected_Top2")
        & _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & history.get(
            "Strategy_Branch", pd.Series("", index=history.index)
        ).astype(str).eq("R6弱势首次转折-N6")
    ].copy() if not history.empty else pd.DataFrame()
    old_returns = pd.to_numeric(
        old_actual.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    old_median = old_returns.median() if len(old_returns) else np.nan
    gates = [
        ("R12弱势研究至少6个独立信号周", weeks >= 6, f"当前{weeks}周"),
        ("R12弱势研究至少12笔完整样本", len(returns) >= 12, f"当前{len(returns)}笔"),
        (
            "R12每个信号周严格只有Top2",
            len(returns) == weeks * TOP_N,
            f"当前{len(returns)}笔/{weeks}周",
        ),
        (
            "R12 W3胜率至少50%",
            len(returns) > 0 and (returns > 0.0).mean() >= 0.50,
            f"当前{((returns > 0.0).mean() * 100.0 if len(returns) else np.nan):.1f}%",
        ),
        (
            "R12 W3中位收益大于0",
            len(returns) > 0 and returns.median() > 0.0,
            f"当前{(returns.median() if len(returns) else np.nan):.2f}%",
        ),
        (
            "R12 W3平均收益大于0",
            len(returns) > 0 and returns.mean() > 0.0,
            f"当前{(returns.mean() if len(returns) else np.nan):.2f}%",
        ),
        ("R12 Profit Factor至少1.2", pd.notna(pf) and pf >= 1.2, f"当前{pf:.2f}"),
        (
            "R12去最佳一只后平均收益大于0",
            len(without_best) > 0 and without_best.mean() > 0.0,
            f"当前{(without_best.mean() if len(without_best) else np.nan):.2f}%",
        ),
        (
            "R12去最佳一只后PF至少1.2",
            pd.notna(without_best_pf) and without_best_pf >= 1.2,
            f"当前{without_best_pf:.2f}",
        ),
        (
            "R12最佳一只占正利润不超过40%",
            math.isfinite(best_contribution) and best_contribution <= 40.0,
            f"当前{best_contribution:.1f}%",
        ),
        (
            "R12前后半段中位收益均为正",
            math.isfinite(first_median)
            and math.isfinite(second_median)
            and first_median > 0.0
            and second_median > 0.0,
            f"前{first_median:.2f}% / 后{second_median:.2f}%",
        ),
        (
            "R12逐周战胜其余弱势合格候选至少55%",
            math.isfinite(paired_beat) and paired_beat >= 55.0,
            f"当前{paired_beat:.1f}%",
        ),
        (
            "R12中位收益优于其余弱势合格候选",
            math.isfinite(selected_median)
            and math.isfinite(rest_median)
            and selected_median > rest_median,
            f"差{(selected_median - rest_median):.2f}个百分点",
        ),
        (
            "R12中位收益优于R6原排名",
            len(returns) > 0
            and math.isfinite(old_median)
            and returns.median() > old_median,
            f"R12{(returns.median() if len(returns) else np.nan):.2f}% / R6{old_median:.2f}%",
        ),
    ]
    return pd.DataFrame(
        [
            {"R12验收项目": name, "结果": "通过" if passed else "未通过", "当前值": value}
            for name, passed, value in gates
        ]
    )


def r13_daily_restart_ranking_diagnostics(history: pd.DataFrame):
    if history.empty or "R13_Daily_Restart_Rank" not in history.columns:
        return {}
    weak = history.get(
        "Market_Regime", pd.Series("", index=history.index)
    ).astype(str).eq("弱势")
    frame = history[
        weak
        & _bool_series(history, "Recovery_Eligible")
        & _bool_series(history, "Daily_Restart_Data_Available")
        & _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
    ].copy()
    frame["Rank"] = pd.to_numeric(
        frame.get("R13_Daily_Restart_Rank"), errors="coerce"
    )
    frame["Selected_Top2"] = _bool_series(frame, "R13_Daily_Restart_Top2")
    frame[PRIMARY_RETURN_COLUMN] = pd.to_numeric(
        frame.get(PRIMARY_RETURN_COLUMN), errors="coerce"
    )
    frame = frame.dropna(subset=["Rank", PRIMARY_RETURN_COLUMN])
    return ranking_diagnostics(frame)


def r13_daily_restart_gates(history: pd.DataFrame):
    """R13独立验收；通过也只代表值得继续样本外验证。"""
    if history.empty:
        selected = pd.DataFrame()
    else:
        selected = history[
            _bool_series(history, "R13_Daily_Restart_Top2")
            & _bool_series(history, "Outcome_Complete")
            & _bool_series(history, "Entry_Tradable")
        ].copy()
    returns = pd.to_numeric(
        selected.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    selected = selected.loc[returns.index].copy() if len(returns) else selected.iloc[0:0]
    weeks = selected["Signal_Date"].nunique() if len(returns) else 0
    selected_dates = set(selected["Signal_Date"].astype(str)) if len(selected) else set()
    pf = _profit_factor(returns)
    without_best = (
        returns.drop(index=returns.idxmax())
        if len(returns) > 1
        else pd.Series(dtype=float)
    )
    without_best_pf = _profit_factor(without_best)
    positive_total = returns[returns > 0.0].sum()
    best_contribution = (
        returns.max() / positive_total * 100.0
        if len(returns) and positive_total > 0.0
        else np.nan
    )
    diagnostics = r13_daily_restart_ranking_diagnostics(history)
    first_median = _safe_float(diagnostics.get("前半段实际入选中位收益%"))
    second_median = _safe_float(diagnostics.get("后半段实际入选中位收益%"))
    paired_beat = _safe_float(
        diagnostics.get("实际入选逐周平均收益战胜未入选比例%")
    )
    selected_median = _safe_float(diagnostics.get("实际入选中位收益%"))
    rest_median = _safe_float(diagnostics.get("未入选候选中位收益%"))
    old_actual = history[
        _bool_series(history, "Selected_Top2")
        & _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & history.get(
            "Strategy_Branch", pd.Series("", index=history.index)
        ).astype(str).eq("R6弱势首次转折-N6")
        & history.get(
            "Signal_Date", pd.Series("", index=history.index)
        ).astype(str).isin(selected_dates)
    ].copy() if not history.empty else pd.DataFrame()
    old_returns = pd.to_numeric(
        old_actual.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    old_median = old_returns.median() if len(old_returns) else np.nan
    top1 = selected[
        pd.to_numeric(selected.get("R13_Daily_Restart_Rank"), errors="coerce").eq(1)
    ]
    second = selected[
        pd.to_numeric(selected.get("R13_Daily_Restart_Rank"), errors="coerce").eq(2)
    ]
    top1_returns = pd.to_numeric(
        top1.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    second_returns = pd.to_numeric(
        second.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    gates = [
        ("R13弱势研究至少6个独立信号周", weeks >= 6, f"当前{weeks}周"),
        ("R13弱势研究至少12笔完整样本", len(returns) >= 12, f"当前{len(returns)}笔"),
        (
            "R13每个信号周严格只有Top2",
            len(returns) == weeks * TOP_N,
            f"当前{len(returns)}笔/{weeks}周",
        ),
        (
            "R13 W3胜率至少50%",
            len(returns) > 0 and (returns > 0.0).mean() >= 0.50,
            f"当前{((returns > 0.0).mean() * 100.0 if len(returns) else np.nan):.1f}%",
        ),
        (
            "R13 W3中位收益大于0",
            len(returns) > 0 and returns.median() > 0.0,
            f"当前{(returns.median() if len(returns) else np.nan):.2f}%",
        ),
        (
            "R13 W3平均收益大于0",
            len(returns) > 0 and returns.mean() > 0.0,
            f"当前{(returns.mean() if len(returns) else np.nan):.2f}%",
        ),
        ("R13 Profit Factor至少1.2", pd.notna(pf) and pf >= 1.2, f"当前{pf:.2f}"),
        (
            "R13去最佳一只后平均收益大于0",
            len(without_best) > 0 and without_best.mean() > 0.0,
            f"当前{(without_best.mean() if len(without_best) else np.nan):.2f}%",
        ),
        (
            "R13去最佳一只后PF至少1.2",
            pd.notna(without_best_pf) and without_best_pf >= 1.2,
            f"当前{without_best_pf:.2f}",
        ),
        (
            "R13最佳一只占正利润不超过40%",
            math.isfinite(best_contribution) and best_contribution <= 40.0,
            f"当前{best_contribution:.1f}%",
        ),
        (
            "R13前后半段中位收益均为正",
            math.isfinite(first_median)
            and math.isfinite(second_median)
            and first_median > 0.0
            and second_median > 0.0,
            f"前{first_median:.2f}% / 后{second_median:.2f}%",
        ),
        (
            "R13逐周战胜其余弱势合格候选至少55%",
            math.isfinite(paired_beat) and paired_beat >= 55.0,
            f"当前{paired_beat:.1f}%",
        ),
        (
            "R13中位收益优于其余弱势合格候选",
            math.isfinite(selected_median)
            and math.isfinite(rest_median)
            and selected_median > rest_median,
            f"差{(selected_median - rest_median):.2f}个百分点",
        ),
        (
            "R13同周中位收益优于R6原排名",
            len(returns) > 0
            and math.isfinite(old_median)
            and returns.median() > old_median,
            f"R13{(returns.median() if len(returns) else np.nan):.2f}% / R6{old_median:.2f}%",
        ),
        (
            "R13第一名中位收益不低于第二名",
            len(top1_returns) > 0
            and len(second_returns) > 0
            and top1_returns.median() >= second_returns.median(),
            f"第一名{(top1_returns.median() if len(top1_returns) else np.nan):.2f}% / 第二名{(second_returns.median() if len(second_returns) else np.nan):.2f}%",
        ),
    ]
    return pd.DataFrame(
        [
            {"R13验收项目": name, "结果": "通过" if passed else "未通过", "当前值": value}
            for name, passed, value in gates
        ]
    )


# -----------------------------------------------------------------------------
# R14 MACD高弹性组与下一开盘退出审计
# -----------------------------------------------------------------------------
def _r14_selected(history: pd.DataFrame, require_lifecycle: bool = False):
    if history.empty:
        return history.iloc[0:0].copy()
    mask = (
        _bool_series(history, "R14_MACD_Elastic_Top2")
        & _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
    )
    if require_lifecycle:
        mask &= _bool_series(history, "R14_Lifecycle_Data_Available")
    return history[mask].copy()


def _r14_performance_row(label: str, group: pd.DataFrame):
    returns = pd.to_numeric(
        group.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    valid = group.loc[returns.index].copy() if len(returns) else group.iloc[0:0].copy()
    grades = valid.get(
        "Outcome_Grade", pd.Series("", index=valid.index)
    ).astype(str)
    without_best = (
        returns.drop(index=returns.idxmax())
        if len(returns) > 1
        else pd.Series(dtype=float)
    )
    positive_total = returns[returns > 0.0].sum()
    best_contribution = (
        returns.max() / positive_total * 100.0
        if len(returns) and positive_total > 0.0
        else np.nan
    )
    mfe = pd.to_numeric(
        valid.get("MFE_W3_Net_pct", pd.Series(dtype=float)), errors="coerce"
    )
    mae = pd.to_numeric(
        valid.get("MAE_W3_Raw_pct", pd.Series(dtype=float)), errors="coerce"
    )
    return {
        "R14高弹性研究组": label,
        "样本数": len(returns),
        "信号周": valid["Signal_Date"].nunique() if len(valid) else 0,
        "胜率%": (returns > 0.0).mean() * 100.0 if len(returns) else np.nan,
        "S级比例%": grades.eq("S").mean() * 100.0 if len(grades) else np.nan,
        "A级比例%": grades.eq("A").mean() * 100.0 if len(grades) else np.nan,
        "S/A比例%": grades.isin({"S", "A"}).mean() * 100.0 if len(grades) else np.nan,
        "F级比例%": grades.eq("F").mean() * 100.0 if len(grades) else np.nan,
        "中位收益%": returns.median() if len(returns) else np.nan,
        "平均收益%": returns.mean() if len(returns) else np.nan,
        "Profit_Factor": _profit_factor(returns),
        "去最佳一只平均收益%": without_best.mean() if len(without_best) else np.nan,
        "去最佳一只PF": _profit_factor(without_best),
        "最佳一只占正利润%": best_contribution,
        "W3最大浮盈中位数%": mfe.median() if len(mfe.dropna()) else np.nan,
        "W3最大回撤中位数%": mae.median() if len(mae.dropna()) else np.nan,
    }


def r14_macd_elastic_candidate_audit(history: pd.DataFrame):
    columns = [
        "R14高弹性研究组", "样本数", "信号周", "胜率%", "S级比例%",
        "A级比例%", "S/A比例%", "F级比例%", "中位收益%", "平均收益%",
        "Profit_Factor", "去最佳一只平均收益%", "去最佳一只PF",
        "最佳一只占正利润%", "W3最大浮盈中位数%", "W3最大回撤中位数%",
    ]
    selected = _r14_selected(history)
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected_dates = set(selected["Signal_Date"].astype(str))
    same_week_r6 = history[
        _bool_series(history, "Selected_Top2")
        & _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & history.get(
            "Strategy_Branch", pd.Series("", index=history.index)
        ).astype(str).eq("R6弱势首次转折-N6")
        & history.get(
            "Signal_Date", pd.Series("", index=history.index)
        ).astype(str).isin(selected_dates)
    ].copy()
    rank = pd.to_numeric(
        selected.get("R14_MACD_Elastic_Rank"), errors="coerce"
    )
    groups = [
        ("R14日线MACD柱加速度Top2（只研究）", selected),
        ("R14高弹性Top1", selected[rank.eq(1)]),
        ("R14高弹性第二名", selected[rank.eq(2)]),
        ("同周R6原排名Top2（实际基线）", same_week_r6),
    ]
    return pd.DataFrame(
        [_r14_performance_row(label, group) for label, group in groups],
        columns=columns,
    )


def r14_exit_classification_audit(history: pd.DataFrame):
    columns = [
        "R14退出规则", "完整样本", "触发退出数", "触发退出比例%",
        "F级样本", "F级捕获率%", "非F误退率%", "S/A样本",
        "S/A误杀率%", "正常次日开盘退出比例%", "平均顺延交易日",
    ]
    selected = _r14_selected(history, require_lifecycle=True)
    if selected.empty:
        return pd.DataFrame(columns=columns)
    grades = selected.get(
        "Outcome_Grade", pd.Series("", index=selected.index)
    ).astype(str)
    f_mask = grades.eq("F")
    sa_mask = grades.isin({"S", "A"})
    rows = []
    for label, trigger_column, exit_column in R14_EXIT_RULES[1:]:
        trigger = _bool_series(selected, trigger_column)
        delay_column = (
            "R14_W1_Exit_Delay_Days"
            if exit_column == "R14_W1_Next_Open_Return_Net_pct"
            else "R14_W2_Exit_Delay_Days"
        )
        delays = pd.to_numeric(
            selected.loc[trigger].get(
                delay_column, pd.Series(dtype=float)
            ),
            errors="coerce",
        ).dropna()
        rows.append(
            {
                "R14退出规则": label,
                "完整样本": len(selected),
                "触发退出数": int(trigger.sum()),
                "触发退出比例%": trigger.mean() * 100.0,
                "F级样本": int(f_mask.sum()),
                "F级捕获率%": trigger[f_mask].mean() * 100.0 if f_mask.any() else np.nan,
                "非F误退率%": trigger[~f_mask].mean() * 100.0 if (~f_mask).any() else np.nan,
                "S/A样本": int(sa_mask.sum()),
                "S/A误杀率%": trigger[sa_mask].mean() * 100.0 if sa_mask.any() else np.nan,
                "正常次日开盘退出比例%": (
                    delays.eq(0).mean() * 100.0 if len(delays) else np.nan
                ),
                "平均顺延交易日": delays.mean() if len(delays) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _r14_lifecycle_returns(
    selected: pd.DataFrame,
    horizon: int,
    trigger_column: str | None,
    exit_column: str | None,
):
    fixed_column = f"Fixed_Return_W{horizon}_Net_pct"
    fixed = pd.to_numeric(
        selected.get(fixed_column, pd.Series(np.nan, index=selected.index)),
        errors="coerce",
    )
    matured = fixed.notna()
    cohort = selected.loc[matured].copy()
    fixed = fixed.loc[matured]
    if trigger_column is None or exit_column is None:
        return cohort, fixed, pd.Series(False, index=cohort.index, dtype=bool)
    trigger = _bool_series(cohort, trigger_column)
    exit_returns = pd.to_numeric(
        cohort.get(exit_column, pd.Series(np.nan, index=cohort.index)),
        errors="coerce",
    )
    valid = ~trigger | exit_returns.notna()
    cohort = cohort.loc[valid].copy()
    trigger = trigger.loc[valid]
    fixed = fixed.loc[valid]
    exit_returns = exit_returns.loc[valid]
    realized = fixed.copy()
    realized.loc[trigger] = exit_returns.loc[trigger]
    return cohort, realized, trigger


def r14_lifecycle_return_audit(history: pd.DataFrame):
    columns = [
        "持有上限", "R14退出规则", "可比样本", "信号周", "退出交易数",
        "胜率%", "中位收益%", "平均收益%", "5%截尾均益%",
        "Profit_Factor", "最大单笔亏损%", "相对固定平均改善百分点",
        "F级退出率%", "S/A保留率%",
    ]
    selected = _r14_selected(history, require_lifecycle=True)
    if selected.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for horizon in R14_LIFECYCLE_HORIZONS:
        baseline = pd.to_numeric(
            selected.get(
                f"Fixed_Return_W{horizon}_Net_pct",
                pd.Series(np.nan, index=selected.index),
            ),
            errors="coerce",
        )
        for label, trigger_column, exit_column in R14_EXIT_RULES:
            cohort, realized, trigger = _r14_lifecycle_returns(
                selected, horizon, trigger_column, exit_column
            )
            realized = pd.to_numeric(realized, errors="coerce").dropna()
            cohort = cohort.loc[realized.index]
            trigger = trigger.loc[realized.index]
            fixed_same = baseline.loc[realized.index]
            grades = cohort.get(
                "Outcome_Grade", pd.Series("", index=cohort.index)
            ).astype(str)
            f_mask = grades.eq("F")
            sa_mask = grades.isin({"S", "A"})
            rows.append(
                {
                    "持有上限": f"W{horizon}",
                    "R14退出规则": label,
                    "可比样本": len(realized),
                    "信号周": cohort["Signal_Date"].nunique() if len(cohort) else 0,
                    "退出交易数": int(trigger.sum()),
                    "胜率%": (realized > 0.0).mean() * 100.0 if len(realized) else np.nan,
                    "中位收益%": realized.median() if len(realized) else np.nan,
                    "平均收益%": realized.mean() if len(realized) else np.nan,
                    "5%截尾均益%": _trimmed_mean(realized),
                    "Profit_Factor": _profit_factor(realized),
                    "最大单笔亏损%": realized.min() if len(realized) else np.nan,
                    "相对固定平均改善百分点": (
                        (realized - fixed_same).mean() if len(realized) else np.nan
                    ),
                    "F级退出率%": trigger[f_mask].mean() * 100.0 if f_mask.any() else np.nan,
                    "S/A保留率%": (~trigger[sa_mask]).mean() * 100.0 if sa_mask.any() else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def r14_yearly_lifecycle_audit(history: pd.DataFrame):
    columns = [
        "年份", "持有上限", "R14退出规则", "样本数", "胜率%",
        "中位收益%", "平均收益%", "Profit_Factor", "最大单笔亏损%",
    ]
    selected = _r14_selected(history, require_lifecycle=True)
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["_year"] = selected["Signal_Date"].astype(str).str[:4]
    rows = []
    for year in sorted(selected["_year"].dropna().unique()):
        year_group = selected[selected["_year"].eq(year)]
        for horizon in R14_LIFECYCLE_HORIZONS:
            for label, trigger_column, exit_column in R14_EXIT_RULES:
                _, realized, _ = _r14_lifecycle_returns(
                    year_group, horizon, trigger_column, exit_column
                )
                realized = pd.to_numeric(realized, errors="coerce").dropna()
                rows.append(
                    {
                        "年份": year,
                        "持有上限": f"W{horizon}",
                        "R14退出规则": label,
                        "样本数": len(realized),
                        "胜率%": (realized > 0.0).mean() * 100.0 if len(realized) else np.nan,
                        "中位收益%": realized.median() if len(realized) else np.nan,
                        "平均收益%": realized.mean() if len(realized) else np.nan,
                        "Profit_Factor": _profit_factor(realized),
                        "最大单笔亏损%": realized.min() if len(realized) else np.nan,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def r14_lifecycle_acceptance_gates(history: pd.DataFrame):
    selected = _r14_selected(history)
    returns = pd.to_numeric(
        selected.get(PRIMARY_RETURN_COLUMN, pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    selected = selected.loc[returns.index] if len(returns) else selected.iloc[0:0]
    dates = set(selected["Signal_Date"].astype(str)) if len(selected) else set()
    r6 = history[
        _bool_series(history, "Selected_Top2")
        & _bool_series(history, "Outcome_Complete")
        & _bool_series(history, "Entry_Tradable")
        & history.get(
            "Strategy_Branch", pd.Series("", index=history.index)
        ).astype(str).eq("R6弱势首次转折-N6")
        & history.get(
            "Signal_Date", pd.Series("", index=history.index)
        ).astype(str).isin(dates)
    ].copy()
    selected_grades = selected.get(
        "Outcome_Grade", pd.Series("", index=selected.index)
    ).astype(str)
    r6_grades = r6.get("Outcome_Grade", pd.Series("", index=r6.index)).astype(str)
    sa_rate = selected_grades.isin({"S", "A"}).mean() * 100.0 if len(selected) else np.nan
    r6_sa_rate = r6_grades.isin({"S", "A"}).mean() * 100.0 if len(r6) else np.nan
    f_rate = selected_grades.eq("F").mean() * 100.0 if len(selected) else np.nan
    r6_f_rate = r6_grades.eq("F").mean() * 100.0 if len(r6) else np.nan
    without_best = returns.drop(index=returns.idxmax()) if len(returns) > 1 else pd.Series(dtype=float)
    positive_total = returns[returns > 0.0].sum()
    best_contribution = (
        returns.max() / positive_total * 100.0
        if len(returns) and positive_total > 0.0
        else np.nan
    )
    selected_week_return = (
        selected.groupby("Signal_Date")[PRIMARY_RETURN_COLUMN].mean()
        if len(selected)
        else pd.Series(dtype=float)
    )
    r6_week_return = (
        r6.groupby("Signal_Date")[PRIMARY_RETURN_COLUMN].mean()
        if len(r6)
        else pd.Series(dtype=float)
    )
    weekly_comparison = pd.concat(
        [
            selected_week_return.rename("R14"),
            r6_week_return.rename("R6"),
        ],
        axis=1,
    ).dropna()
    weekly_difference = weekly_comparison["R14"] - weekly_comparison["R6"]
    weekly_wins = int(weekly_difference.gt(1e-9).sum())
    weekly_losses = int(weekly_difference.lt(-1e-9).sum())
    weekly_ties = int(weekly_difference.abs().le(1e-9).sum())
    weekly_win_rate = (
        weekly_wins / len(weekly_comparison) * 100.0
        if len(weekly_comparison)
        else np.nan
    )

    classification = r14_exit_classification_audit(history)
    primary_class = classification[
        classification.get(
            "R14退出规则", pd.Series("", index=classification.index)
        ).astype(str).eq(R14_PRIMARY_EXIT_RULE)
    ]
    f_capture = _safe_float(
        primary_class["F级捕获率%"].iloc[0] if not primary_class.empty else np.nan
    )
    sa_false = _safe_float(
        primary_class["S/A误杀率%"].iloc[0] if not primary_class.empty else np.nan
    )
    lifecycle = r14_lifecycle_return_audit(history)
    base_w3 = lifecycle[
        lifecycle["持有上限"].astype(str).eq("W3")
        & lifecycle["R14退出规则"].astype(str).eq("固定持有")
    ] if not lifecycle.empty else pd.DataFrame()
    primary_w3 = lifecycle[
        lifecycle["持有上限"].astype(str).eq("W3")
        & lifecycle["R14退出规则"].astype(str).eq(R14_PRIMARY_EXIT_RULE)
    ] if not lifecycle.empty else pd.DataFrame()
    base_pf = _safe_float(base_w3["Profit_Factor"].iloc[0] if not base_w3.empty else np.nan)
    primary_pf = _safe_float(primary_w3["Profit_Factor"].iloc[0] if not primary_w3.empty else np.nan)
    base_min = _safe_float(base_w3["最大单笔亏损%"].iloc[0] if not base_w3.empty else np.nan)
    primary_min = _safe_float(primary_w3["最大单笔亏损%"].iloc[0] if not primary_w3.empty else np.nan)
    gates = [
        ("冻结入场8项", "R14高弹性组至少12个完整信号周", selected["Signal_Date"].nunique() >= 12 if len(selected) else False, f"当前{selected['Signal_Date'].nunique() if len(selected) else 0}周"),
        ("冻结入场8项", "R14高弹性组至少24笔完整交易", len(returns) >= 24, f"当前{len(returns)}笔"),
        ("冻结入场8项", "R14 S/A比例至少高于同周R6十个百分点", math.isfinite(sa_rate) and math.isfinite(r6_sa_rate) and sa_rate >= r6_sa_rate + 10.0, f"R14 {sa_rate:.1f}% / R6 {r6_sa_rate:.1f}%"),
        ("冻结入场8项", "R14 F级比例至少低于同周R6十个百分点", math.isfinite(f_rate) and math.isfinite(r6_f_rate) and f_rate <= r6_f_rate - 10.0, f"R14 {f_rate:.1f}% / R6 {r6_f_rate:.1f}%"),
        ("冻结入场8项", "R14固定W3中位收益大于0", len(returns) > 0 and returns.median() > 0.0, f"当前{returns.median() if len(returns) else np.nan:.2f}%"),
        ("冻结入场8项", "R14去最佳一只后PF至少1.2", pd.notna(_profit_factor(without_best)) and _profit_factor(without_best) >= 1.2, f"当前{_profit_factor(without_best):.2f}"),
        ("冻结入场8项", "R14最佳一只占正利润不超过40%", math.isfinite(best_contribution) and best_contribution <= 40.0, f"当前{best_contribution:.1f}%"),
        ("冻结入场8项", "R14逐周战胜同周R6至少55%", math.isfinite(weekly_win_rate) and weekly_win_rate >= 55.0, f"胜{weekly_wins}周 / 负{weekly_losses}周 / 平{weekly_ties}周；{weekly_win_rate:.1f}%"),
        ("新增退出4项", "主退出规则捕获至少65%的F级", math.isfinite(f_capture) and f_capture >= 65.0, f"当前{f_capture:.1f}%"),
        ("新增退出4项", "主退出规则误杀S/A不超过10%", math.isfinite(sa_false) and sa_false <= 10.0, f"当前{sa_false:.1f}%"),
        ("新增退出4项", "主退出规则W3 Profit Factor不低于固定持有", math.isfinite(primary_pf) and math.isfinite(base_pf) and primary_pf >= base_pf, f"退出{primary_pf:.2f} / 固定{base_pf:.2f}"),
        ("新增退出4项", "主退出规则改善W3最大单笔亏损", math.isfinite(primary_min) and math.isfinite(base_min) and primary_min > base_min, f"退出{primary_min:.2f}% / 固定{base_min:.2f}%"),
    ]
    return pd.DataFrame(
        [
            {
                "验收阶段": phase,
                "R14验收项目": name,
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
                    _bool_series(group, "Selected_Top2").sum()
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
            ledger_status.isin(
                {"PENDING_R13_DAILY", "PENDING_R14_LIFECYCLE"}
            ),
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
            _selected=_bool_series(history_frame, "Selected_Top2")
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


def _apply_r14_selection_policy(frame: pd.DataFrame):
    """兼容导入R9—R14：实际组合不变，重算可复原的研究标记。"""
    result = frame.copy()
    market_regime = result.get(
        "Market_Regime", pd.Series("", index=result.index)
    ).astype(str)
    strong_market = market_regime.eq("强势")
    reset_context = _bool_series(result, "Strong_Reset_Context_Pass")
    continuation_context = _bool_series(
        result, "Strong_Continuation_Context_Pass"
    )
    strong_rank = pd.to_numeric(
        result.get("Strong_Rank", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    result["R7_Research_Top2"] = (
        strong_market
        & reset_context
        & _bool_series(result, "Strong_Eligible")
        & strong_rank.le(TOP_N)
    )
    result["R9_Failure_Control_Top2"] = (
        strong_market
        & continuation_context
        & _bool_series(result, "Strong_Reacceleration_Eligible")
        & _bool_series(result, "R9_Selection_Qualified")
    )

    # 老结果包可能保存的是较早版本的“合格”字段；R11统一由同一时点的
    # 再启动触发与风险边界重算，随后按ATR3/ATR13从小到大排一次名。
    reacceleration_eligible = (
        strong_market
        & _bool_series(result, "Strong_Reacceleration_Trigger")
        & _bool_series(result, "Strong_Reacceleration_Risk_OK")
    )
    result["Strong_Reacceleration_Eligible"] = reacceleration_eligible
    result["R11_Strong_Rank"] = np.nan
    atr = pd.to_numeric(
        result.get("ATR_Contraction", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    eligible_rows = result.loc[reacceleration_eligible].copy()
    if not eligible_rows.empty:
        eligible_rows["_atr"] = atr.loc[eligible_rows.index]
        eligible_rows["_code"] = eligible_rows.get(
            "ts_code", pd.Series("", index=eligible_rows.index)
        ).astype(str)
        eligible_rows = eligible_rows.sort_values(
            ["Signal_Date", "_atr", "_code"],
            ascending=[True, True, True],
            na_position="last",
            kind="mergesort",
        )
        eligible_rows["_rank"] = (
            eligible_rows.groupby("Signal_Date", sort=False).cumcount() + 1
        )
        result.loc[eligible_rows.index, "R11_Strong_Rank"] = eligible_rows["_rank"]
    result["R11_ATR_Band_Pass"] = atr.between(
        R11_ATR_CONTRACTION_MIN, R11_ATR_CONTRACTION_MAX, inclusive="both"
    )
    result["R11_Strong_Research_Top1"] = (
        strong_market
        & pd.to_numeric(result["R11_Strong_Rank"], errors="coerce").eq(1)
        & _bool_series(result, "R11_ATR_Band_Pass")
    )

    # R12弱势研究排名必须从完整当周R6合格池重算。主键是既有深跌结构分，
    # 价格/MA10只负责打破同分；Top2研究标记不改变R6原实际入选。
    weak_market = market_regime.eq("弱势")
    recovery_eligible = weak_market & _bool_series(result, "Recovery_Eligible")
    result["R12_Recovery_Repair_Rank"] = np.nan
    repair_rows = result.loc[recovery_eligible].copy()
    if not repair_rows.empty:
        repair_rows["_pullback"] = pd.to_numeric(
            repair_rows.get(
                "Score_Pullback_15", pd.Series(np.nan, index=repair_rows.index)
            ),
            errors="coerce",
        )
        repair_rows["_price_repair"] = pd.to_numeric(
            repair_rows.get(
                "Price_to_MA10_Ratio", pd.Series(np.nan, index=repair_rows.index)
            ),
            errors="coerce",
        )
        repair_rows["_code"] = repair_rows.get(
            "ts_code", pd.Series("", index=repair_rows.index)
        ).astype(str)
        repair_rows = repair_rows.sort_values(
            ["Signal_Date", "_pullback", "_price_repair", "_code"],
            ascending=[True, True, False, True],
            na_position="last",
            kind="mergesort",
        )
        repair_rows["_rank"] = (
            repair_rows.groupby("Signal_Date", sort=False).cumcount() + 1
        )
        result.loc[
            repair_rows.index, "R12_Recovery_Repair_Rank"
        ] = repair_rows["_rank"]
    recovery_count_by_week = (
        result.loc[recovery_eligible]
        .groupby("Signal_Date", sort=False)
        .size()
    )
    enough_recovery = result["Signal_Date"].map(recovery_count_by_week).fillna(0).ge(
        MIN_VALID_SELECTION_SIZE
    )
    result["R12_Recovery_Repair_Top2"] = (
        recovery_eligible
        & enough_recovery
        & pd.to_numeric(
            result["R12_Recovery_Repair_Rank"], errors="coerce"
        ).le(TOP_N)
    )

    # R13日线分只能由结果包中保存的信号日日线快照重算。R9—R12旧包没有
    # 这些字段时全部保持不可用，绝不以代码顺序伪造研究名次。
    daily_required = [
        "Daily_Close_to_MA20_Ratio",
        "Daily_MA5_Slope_3D_pct",
        "Daily_MACD_Hist_Delta_pct",
        "Daily_RS_5D_Pct",
        "Daily_Higher_Low_5D_pct",
    ]
    daily_available = pd.Series(True, index=result.index, dtype=bool)
    for column in daily_required:
        if column not in result.columns:
            result[column] = np.nan
        daily_available &= pd.to_numeric(result[column], errors="coerce").notna()
    result["Daily_Restart_Data_Available"] = daily_available
    result["R13_Daily_Restart_Rank"] = np.nan
    for column in (
        "R13_Daily_Price_Repair_20",
        "R13_Daily_MA5_Slope_20",
        "R13_Daily_MACD_Accel_20",
        "R13_Daily_RS5_20",
        "R13_Daily_Higher_Low_20",
        "R13_Daily_Restart_100",
    ):
        result[column] = np.nan
    r13_eligible = recovery_eligible & daily_available
    for _, week_rows in result.loc[r13_eligible].groupby(
        "Signal_Date", sort=False
    ):
        scored = _score_daily_restart_quality(week_rows)
        score_columns = [
            "R13_Daily_Price_Repair_20",
            "R13_Daily_MA5_Slope_20",
            "R13_Daily_MACD_Accel_20",
            "R13_Daily_RS5_20",
            "R13_Daily_Higher_Low_20",
            "R13_Daily_Restart_100",
        ]
        for column in score_columns:
            result.loc[scored.index, column] = scored[column]
        ordered = scored.sort_values(
            [
                "R13_Daily_Restart_100",
                "Daily_Close_to_MA20_Ratio",
                "ts_code",
            ],
            ascending=[False, False, True],
            kind="mergesort",
        )
        result.loc[ordered.index, "R13_Daily_Restart_Rank"] = np.arange(
            1, len(ordered) + 1, dtype=float
        )
    r13_count_by_week = (
        result.loc[r13_eligible].groupby("Signal_Date", sort=False).size()
    )
    enough_r13 = result["Signal_Date"].map(r13_count_by_week).fillna(0).ge(
        MIN_VALID_SELECTION_SIZE
    )
    r13_rank = pd.to_numeric(result["R13_Daily_Restart_Rank"], errors="coerce")
    result["R13_Daily_Restart_Top1"] = (
        r13_eligible & enough_r13 & r13_rank.eq(1)
    )
    result["R13_Daily_Restart_Top2"] = (
        r13_eligible & enough_r13 & r13_rank.le(TOP_N)
    )

    # R14只需要信号日MACD柱加速度；从完整的同周R6合格池恢复名次，
    # 不允许利用结果包中的未来收益或保存的旧名次。
    elastic_available = pd.to_numeric(
        result.get(
            "Daily_MACD_Hist_Delta_pct",
            pd.Series(np.nan, index=result.index),
        ),
        errors="coerce",
    ).notna()
    r14_eligible = recovery_eligible & elastic_available
    result["R14_MACD_Elastic_Rank"] = np.nan
    elastic_rows = result.loc[r14_eligible].copy()
    if not elastic_rows.empty:
        elastic_rows["_macd_accel"] = pd.to_numeric(
            elastic_rows["Daily_MACD_Hist_Delta_pct"], errors="coerce"
        )
        elastic_rows["_code"] = elastic_rows.get(
            "ts_code", pd.Series("", index=elastic_rows.index)
        ).astype(str)
        elastic_rows = elastic_rows.sort_values(
            ["Signal_Date", "_macd_accel", "_code"],
            ascending=[True, False, True],
            na_position="last",
            kind="mergesort",
        )
        elastic_rows["_rank"] = (
            elastic_rows.groupby("Signal_Date", sort=False).cumcount() + 1
        )
        result.loc[
            elastic_rows.index, "R14_MACD_Elastic_Rank"
        ] = elastic_rows["_rank"]
    r14_count_by_week = (
        result.loc[r14_eligible].groupby("Signal_Date", sort=False).size()
    )
    enough_r14 = result["Signal_Date"].map(r14_count_by_week).fillna(0).ge(
        MIN_VALID_SELECTION_SIZE
    )
    r14_rank = pd.to_numeric(result["R14_MACD_Elastic_Rank"], errors="coerce")
    result["R14_MACD_Elastic_Top1"] = (
        r14_eligible & enough_r14 & r14_rank.eq(1)
    )
    result["R14_MACD_Elastic_Top2"] = (
        r14_eligible & enough_r14 & r14_rank.le(TOP_N)
    )
    result["R14_MACD_Elastic_Count"] = (
        result["Signal_Date"].map(r14_count_by_week).fillna(0).astype(int)
    )

    if "Selected_Top2" not in result.columns:
        result["Selected_Top2"] = False
    if "Selection_Valid" not in result.columns:
        result["Selection_Valid"] = False
    if "Strategy_Branch" not in result.columns:
        result["Strategy_Branch"] = ""
    if "Selection_Block_Reason" not in result.columns:
        result["Selection_Block_Reason"] = ""

    result.loc[strong_market, "Selected_Top2"] = False
    result.loc[strong_market, "Selection_Valid"] = False
    result.loc[strong_market, "Strategy_Branch"] = "R11强势温和收缩Top1观察"
    result.loc[
        strong_market, "Selection_Block_Reason"
    ] = "R11强势温和收缩Top1仅研究，实际组合强制空仓"
    result.loc[strong_market, "Rank"] = result.loc[
        strong_market, "R11_Strong_Rank"
    ]
    result.loc[strong_market, "Entry_Eligible"] = result.loc[
        strong_market, "Strong_Reacceleration_Eligible"
    ]
    return result


def import_r14_results_zip(zip_bytes: bytes, config_id: str):
    """先完整验证、后事务提交；失败时不允许留下候选或账本半成品。"""
    if not zip_bytes:
        raise ValueError("结果包为空。")
    task = read_json_safe(RUN_TASK_FILE)
    if task.get("State") in {"RUNNING", "PAUSED_ERROR"}:
        raise RuntimeError("历史任务仍在运行或暂停中，请先停止任务再导入结果包。")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        archive_items = archive.infolist()
        names = [item.filename for item in archive_items]
        if len(names) != len(set(names)):
            raise ValueError("结果包包含重复文件名，拒绝导入。")
        infos = {item.filename: item for item in archive_items}
        candidate_names = [
            name
            for name in infos
            if (
                name.startswith("01_all_r9")
                or name.startswith("01_all_r10")
                or name.startswith("01_all_r11")
                or name.startswith("01_all_r12")
                or name.startswith("01_all_r13")
                or name.startswith("01_all_r14")
            )
            and name.endswith("_candidates.csv")
        ]
        if len(candidate_names) != 1:
            raise ValueError("结果包中未找到唯一的R9/R10/R11/R12/R13/R14候选明细。")
        candidate_info = infos[candidate_names[0]]
        if candidate_info.file_size > 200 * 1024 * 1024:
            raise ValueError("候选明细超过200MB，拒绝导入。")
        candidates = pd.read_csv(
            io.BytesIO(archive.read(candidate_info)),
            encoding="utf-8-sig",
            low_memory=False,
        )
        if not {"Signal_Date", "ts_code"}.issubset(candidates.columns):
            raise ValueError("候选明细缺少Signal_Date或ts_code。")
        candidates["Signal_Date"] = candidates["Signal_Date"].map(parse_yyyymmdd)
        candidates = candidates.dropna(subset=["Signal_Date", "ts_code"]).copy()
        if candidates.empty:
            raise ValueError("候选明细为空。")
        if candidates.duplicated(["Signal_Date", "ts_code"]).any():
            raise ValueError("候选明细存在重复的日期与股票代码。")
        candidates = _apply_r14_selection_policy(candidates)
        legacy_missing_r13 = not _bool_series(
            candidates, "Daily_Restart_Data_Available"
        ).any()
        r14_lifecycle_required = {
            "R14_W1_Close_Gross_pct",
            "R14_W2_Close_Gross_pct",
            "R14_W1_Next_Open_Return_Net_pct",
            "R14_W2_Next_Open_Return_Net_pct",
            "R14_Lifecycle_Data_Available",
        }
        legacy_missing_r14 = not r14_lifecycle_required.issubset(
            candidates.columns
        )
        pending_r14_dates = set(
            candidates.loc[
                _bool_series(candidates, "R14_MACD_Elastic_Top2"),
                "Signal_Date",
            ].astype(str)
        ) if legacy_missing_r14 else set()
        candidates["Config_ID"] = str(config_id)

        opportunity_name = next(
            (
                name
                for name in infos
                if name.startswith("22_") and name.endswith("opportunities.csv")
            ),
            None,
        )
        opportunity_count = 0
        opportunities = pd.DataFrame()
        if opportunity_name:
            if infos[opportunity_name].file_size > 200 * 1024 * 1024:
                raise ValueError("大牛股机会明细超过200MB，拒绝导入。")
            opportunities = pd.read_csv(
                io.BytesIO(archive.read(infos[opportunity_name])),
                encoding="utf-8-sig",
                low_memory=False,
            )
            if not {"Signal_Date", "ts_code"}.issubset(opportunities.columns):
                raise ValueError("大牛股机会明细缺少Signal_Date或ts_code。")
            opportunities["Signal_Date"] = opportunities["Signal_Date"].map(
                parse_yyyymmdd
            )
            opportunities = opportunities.dropna(
                subset=["Signal_Date", "ts_code"]
            ).copy()
            if opportunities.duplicated(["Signal_Date", "ts_code"]).any():
                raise ValueError("大牛股机会明细存在重复的日期与股票代码。")
            opportunities = _apply_r14_selection_policy(opportunities)
            # 机会表只是全池子集，R11/R12/R13名次必须从完整候选表回填。
            research_map_columns = [
                "Signal_Date",
                "ts_code",
                "R11_Strong_Rank",
                "R11_ATR_Band_Pass",
                "R11_Strong_Research_Top1",
                "R12_Recovery_Repair_Rank",
                "R12_Recovery_Repair_Top2",
                "R13_Daily_Restart_Rank",
                "R13_Daily_Restart_Top1",
                "R13_Daily_Restart_Top2",
                "R14_MACD_Elastic_Rank",
                "R14_MACD_Elastic_Top1",
                "R14_MACD_Elastic_Top2",
            ]
            research_map = candidates[research_map_columns].drop_duplicates(
                ["Signal_Date", "ts_code"], keep="last"
            )
            mapped = opportunities[["Signal_Date", "ts_code"]].merge(
                research_map,
                on=["Signal_Date", "ts_code"],
                how="left",
                sort=False,
            )
            for column in research_map_columns[2:]:
                opportunities[column] = mapped[column].to_numpy()
            opportunities["R11_Strong_Research_Top1"] = _bool_series(
                opportunities, "R11_Strong_Research_Top1"
            )
            opportunities["R12_Recovery_Repair_Top2"] = _bool_series(
                opportunities, "R12_Recovery_Repair_Top2"
            )
            opportunities["R13_Daily_Restart_Top1"] = _bool_series(
                opportunities, "R13_Daily_Restart_Top1"
            )
            opportunities["R13_Daily_Restart_Top2"] = _bool_series(
                opportunities, "R13_Daily_Restart_Top2"
            )
            opportunities["R14_MACD_Elastic_Top1"] = _bool_series(
                opportunities, "R14_MACD_Elastic_Top1"
            )
            opportunities["R14_MACD_Elastic_Top2"] = _bool_series(
                opportunities, "R14_MACD_Elastic_Top2"
            )
            strong_opportunities = opportunities.get(
                "Market_Regime", pd.Series("", index=opportunities.index)
            ).astype(str).eq("强势")
            if "Detection_Status" in opportunities.columns:
                any_strong_trigger = (
                    _bool_series(opportunities, "Strong_Resilience_Trigger")
                    | _bool_series(opportunities, "Strong_Reacceleration_Trigger")
                )
                opportunities.loc[
                    strong_opportunities & any_strong_trigger,
                    "Detection_Status",
                ] = "已发现但被规则拦截"
            if "Miss_Reason" in opportunities.columns:
                opportunities.loc[
                    strong_opportunities, "Miss_Reason"
                ] = "R11强势Top1仅研究，实际组合强制空仓"
            opportunities["Config_ID"] = str(config_id)
            opportunity_count = len(opportunities)

        ledger_name = next(
            (name for name in infos if name == "26_scan_ledger.csv"), None
        )
        if ledger_name:
            if infos[ledger_name].file_size > 20 * 1024 * 1024:
                raise ValueError("扫描账本超过20MB，拒绝导入。")
            imported_ledger = pd.read_csv(
                io.BytesIO(archive.read(infos[ledger_name])),
                encoding="utf-8-sig",
                low_memory=False,
            )
            imported_ledger["Signal_Date"] = imported_ledger["Signal_Date"].map(
                parse_yyyymmdd
            )
            imported_ledger = imported_ledger.dropna(subset=["Signal_Date"]).copy()
            if imported_ledger.empty:
                raise ValueError("扫描账本为空。")
            if imported_ledger.duplicated(["Signal_Date"]).any():
                raise ValueError("扫描账本存在重复日期。")
            imported_ledger["Config_ID"] = str(config_id)
        else:
            # R9.0旧结果没有账本，只能把确实存在候选明细的周标为完成；
            # 零候选周会在新任务中自动重扫，不会伪造其完成状态。
            grouped = candidates.groupby("Signal_Date", as_index=False)
            imported_ledger = grouped.agg(
                Raw_Setup_Count=("ts_code", "size"),
                Eligible_Trend_Count=("Entry_Eligible", lambda values: int(_bool_series(pd.DataFrame({"v": values}), "v").sum())),
                Selected_Count=("Selected_Top2", lambda values: int(_bool_series(pd.DataFrame({"v": values}), "v").sum())),
            )
            imported_ledger["Selection_Block_Reason"] = "由R9.0结果包恢复；零候选周将重扫"
            imported_ledger["Scan_Status"] = "COMPLETED"
            imported_ledger["Market_Data_Gap_Count"] = 0
            imported_ledger["Market_Data_Gap_Dates"] = ""
            imported_ledger["Config_ID"] = str(config_id)
            imported_ledger["Updated_At"] = datetime.now().isoformat(
                timespec="seconds"
            )

        if legacy_missing_r13:
            imported_ledger["Scan_Status"] = "PENDING_R13_DAILY"
            imported_ledger["Selection_Block_Reason"] = (
                "旧结果包已恢复；缺少R13信号日日线快照，等待R13逐周重扫"
            )
        elif pending_r14_dates:
            pending_mask = imported_ledger["Signal_Date"].astype(str).isin(
                pending_r14_dates
            )
            imported_ledger.loc[
                pending_mask, "Scan_Status"
            ] = "PENDING_R14_LIFECYCLE"
            imported_ledger.loc[
                pending_mask, "Selection_Block_Reason"
            ] = "R13结果已恢复；等待补算R14下一可交易日开盘退出路径"

        candidate_rows_by_week = candidates.groupby("Signal_Date").size().to_dict()
        mapped_candidate_rows = imported_ledger["Signal_Date"].map(
            candidate_rows_by_week
        ).fillna(0).astype(int)
        raw_setup_counts = pd.to_numeric(
            imported_ledger.get(
                "Raw_Setup_Count", pd.Series(0, index=imported_ledger.index)
            ),
            errors="coerce",
        ).fillna(0).astype(int)
        if "Candidate_Row_Count" in imported_ledger.columns:
            source_candidate_rows = pd.to_numeric(
                imported_ledger["Candidate_Row_Count"], errors="coerce"
            )
            source_count_mismatch = source_candidate_rows.notna() & source_candidate_rows.ne(
                mapped_candidate_rows
            )
        else:
            source_count_mismatch = pd.Series(False, index=imported_ledger.index)
        missing_candidate_weeks = (
            (raw_setup_counts.gt(0) & mapped_candidate_rows.eq(0))
            | source_count_mismatch
        )
        if missing_candidate_weeks.any():
            bad_dates = imported_ledger.loc[
                missing_candidate_weeks, "Signal_Date"
            ].astype(str).tolist()
            preview_dates = "、".join(bad_dates[:8])
            raise ValueError(
                f"结果包中有{len(bad_dates)}周账本显示已扫描，但候选明细缺失；"
                f"示例：{preview_dates}。本次导入未写入任何文件。"
            )
        selected_by_week = (
            candidates.groupby("Signal_Date")["Selected_Top2"]
            .apply(lambda values: int(_bool_series(pd.DataFrame({"v": values}), "v").sum()))
            .to_dict()
        )
        imported_ledger["Selected_Count"] = imported_ledger["Signal_Date"].map(
            selected_by_week
        ).fillna(0).astype(int)
        imported_ledger["Candidate_Row_Count"] = mapped_candidate_rows
        block_reason_by_week = (
            candidates.assign(
                _block=candidates.get(
                    "Selection_Block_Reason", pd.Series("", index=candidates.index)
                ).fillna("").astype(str)
            )
            .groupby("Signal_Date")["_block"]
            .agg(lambda values: next((value for value in values if value), ""))
            .to_dict()
        )
        mapped_reasons = imported_ledger["Signal_Date"].map(block_reason_by_week)
        if "Selection_Block_Reason" not in imported_ledger.columns:
            imported_ledger["Selection_Block_Reason"] = ""
        preserve_pending_reason = imported_ledger.get(
            "Scan_Status", pd.Series("", index=imported_ledger.index)
        ).astype(str).isin({"PENDING_R13_DAILY", "PENDING_R14_LIFECYCLE"})
        imported_ledger.loc[
            mapped_reasons.fillna("").ne("") & ~preserve_pending_reason,
            "Selection_Block_Reason",
        ] = mapped_reasons[
            mapped_reasons.fillna("").ne("") & ~preserve_pending_reason
        ]

        # 任何“账本说有候选、明细却没有”的压缩包必须在写盘前拒绝。
        imported_issues = result_state_consistency_audit(
            candidates,
            imported_ledger,
        )
        if not imported_issues.empty:
            preview_dates = "、".join(
                imported_issues["Signal_Date"].astype(str).head(8)
            )
            raise ValueError(
                f"结果包状态不完整：{len(imported_issues)}周账本与候选明细不一致；"
                f"示例：{preview_dates}。本次导入未写入任何文件。"
            )

        existing_candidates = read_csv_safe(CHECKPOINT_FILE)
        combined_candidates = (
            pd.concat(
                [existing_candidates, candidates], ignore_index=True, sort=False
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

        existing_opportunities = read_csv_safe(OPPORTUNITY_FILE)
        if not opportunities.empty:
            combined_opportunities = (
                pd.concat(
                    [existing_opportunities, opportunities],
                    ignore_index=True,
                    sort=False,
                )
                if not existing_opportunities.empty
                else opportunities.copy()
            )
            combined_opportunities["Signal_Date"] = combined_opportunities[
                "Signal_Date"
            ].map(parse_yyyymmdd)
            combined_opportunities = combined_opportunities.dropna(
                subset=["Signal_Date", "ts_code"]
            ).drop_duplicates(
                ["Config_ID", "Signal_Date", "ts_code"], keep="last"
            )
            combined_opportunities = combined_opportunities.sort_values(
                ["Signal_Date", PRIMARY_RETURN_COLUMN],
                ascending=[True, False],
                kind="mergesort",
            ).reset_index(drop=True)
        else:
            combined_opportunities = existing_opportunities

        existing_ledger = read_csv_safe(SCAN_LEDGER_FILE)
        merged_ledger = (
            pd.concat([existing_ledger, imported_ledger], ignore_index=True, sort=False)
            if not existing_ledger.empty
            else imported_ledger
        )
        merged_ledger["Signal_Date"] = merged_ledger["Signal_Date"].map(
            parse_yyyymmdd
        )
        merged_ledger = merged_ledger.dropna(subset=["Signal_Date"])
        merged_ledger = merged_ledger.drop_duplicates(
            ["Signal_Date", "Config_ID"], keep="last"
        )
        merged_ledger = merged_ledger.sort_values("Signal_Date").reset_index(
            drop=True
        )

        write_paths = [CHECKPOINT_FILE, SCAN_LEDGER_FILE]
        if not opportunities.empty:
            write_paths.append(OPPORTUNITY_FILE)
        with _result_files_transaction(write_paths):
            atomic_write_csv(combined_candidates, CHECKPOINT_FILE)
            if not opportunities.empty:
                atomic_write_csv(combined_opportunities, OPPORTUNITY_FILE)
            atomic_write_csv(merged_ledger, SCAN_LEDGER_FILE)

            committed_candidates = combined_candidates[
                combined_candidates["Config_ID"].astype(str).eq(str(config_id))
            ].copy()
            committed_ledger = merged_ledger[
                merged_ledger["Config_ID"].astype(str).eq(str(config_id))
            ].copy()
            committed_issues = result_state_consistency_audit(
                committed_candidates, committed_ledger
            )
            if not committed_issues.empty:
                raise RuntimeError("导入后一致性校验失败，已自动回滚。")
    return {
        "candidate_rows": len(candidates),
        "known_weeks": candidates["Signal_Date"].nunique(),
        "r13_daily_rows": int(
            _bool_series(candidates, "Daily_Restart_Data_Available").sum()
        ),
        "pending_r14_weeks": len(pending_r14_dates),
        "opportunity_rows": opportunity_count,
        "ledger_rows": len(imported_ledger),
        "ledger_inferred": ledger_name is None,
    }


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
    scan_ledger: pd.DataFrame,
    data_gap_audit: pd.DataFrame,
    neutral_gates: pd.DataFrame,
    r11_audit: pd.DataFrame,
    r11_gates: pd.DataFrame,
    r11_diagnostics: dict[str, Any],
    state_consistency_audit: pd.DataFrame,
    r12_audit: pd.DataFrame,
    r12_gates: pd.DataFrame,
    r12_diagnostics: dict[str, Any],
    r12_factor_comparison: pd.DataFrame,
    r13_audit: pd.DataFrame,
    r13_gates: pd.DataFrame,
    r13_diagnostics: dict[str, Any],
    r13_factor_comparison: pd.DataFrame,
    r14_candidate_audit: pd.DataFrame,
    r14_exit_audit: pd.DataFrame,
    r14_lifecycle_audit: pd.DataFrame,
    r14_yearly_audit: pd.DataFrame,
    r14_gates: pd.DataFrame,
):
    buffer = io.BytesIO()
    files = {
        "01_all_r14_macd_elastic_lifecycle_candidates.csv": history,
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
        "26_scan_ledger.csv": scan_ledger,
        "27_market_data_gap_audit.csv": data_gap_audit,
        "28_r3_neutral_acceptance_gates.csv": neutral_gates,
        "29_r11_moderate_atr_top1_candidate_audit.csv": r11_audit,
        "30_r11_moderate_atr_top1_acceptance_gates.csv": r11_gates,
        "31_r11_moderate_atr_top1_ranking_diagnostics.csv": pd.DataFrame(
            [{"指标": key, "数值": value} for key, value in r11_diagnostics.items()]
        ),
        "32_result_state_consistency_audit.csv": state_consistency_audit,
        "33_r12_weak_repair_candidate_audit.csv": r12_audit,
        "34_r12_weak_repair_acceptance_gates.csv": r12_gates,
        "35_r12_weak_repair_ranking_diagnostics.csv": pd.DataFrame(
            [{"指标": key, "数值": value} for key, value in r12_diagnostics.items()]
        ),
        "36_r12_weak_repair_factor_comparison.csv": r12_factor_comparison,
        "37_r13_daily_restart_candidate_audit.csv": r13_audit,
        "38_r13_daily_restart_acceptance_gates.csv": r13_gates,
        "39_r13_daily_restart_ranking_diagnostics.csv": pd.DataFrame(
            [{"指标": key, "数值": value} for key, value in r13_diagnostics.items()]
        ),
        "40_r13_daily_restart_factor_comparison.csv": r13_factor_comparison,
        "41_r14_macd_elastic_candidate_audit.csv": r14_candidate_audit,
        "42_r14_exit_classification_audit.csv": r14_exit_audit,
        "43_r14_lifecycle_return_audit.csv": r14_lifecycle_audit,
        "44_r14_yearly_lifecycle_audit.csv": r14_yearly_audit,
        "45_r14_lifecycle_acceptance_gates.csv": r14_gates,
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
        "R14保持R11.1实际组合逐行不变：R3中性Top2与R6弱势Top2仍是历史基线；"
        "R14只研究已经预先声明的MACD高弹性Top2及三种退出规则。"
    )
    st.caption(f"运行引擎修订：{ENGINE_PATCH}")
    st.warning(
        "R14仍是研究验证版本，不是实盘版本；高弹性排名和退出规则均不进入原实际组合。"
    )
    with st.expander("查看R14交易与研究边界"):
        st.markdown(
            """
- **R3中性趋势**：原MACD首红、MA20资格和趋势/风险/总分词典序完全保留。
- **R6弱势首次转折**：26周深跌且N6 SKDJ近期进入35以下，K首次转升，并出现周涨或强收之一；此前两周已有同类转折则不重复触发。
- **R11统一强势事件**：沿用R9的“整理后首次再启动”触发和风险边界，不叠加新引擎。
- **R11排名**：在当周合格股中只按ATR3/ATR13从小到大排名，股票代码仅用于并列时确定顺序。
- **R11固定接受带**：只检查排名第一的股票；ATR3/ATR13位于0.70至0.90才记为研究信号。第一名不合格时不用第二名补位。
- **R12弱势研究排名**：只在R6-N6合格池中，先按既有Score_Pullback_15从低到高，再按价格/MA10从高到低打破并列，观察Top2。
- **R12不增加硬门**：没有新的阈值、市场门或候选数量扩容；新名次不覆盖R6原名次。
- **R13日线挑战排名**：仍在同一个R6-N6合格池内，把信号日已知的价格/MA20、MA5三日斜率、MACD柱加速度、全池五日相对强度、五日低点抬升五项做同周等权排名。
- **R13无个股特判**：不使用股票名称、行业故事或买入后涨跌；缺少完整日线快照的旧结果包不会伪造R13名次。
- **R14高弹性排名**：严格使用R13已经预先声明的单因子——信号日日线MACD柱加速度；仍只在同周R6-N6合格池中从高到低取Top2，不增加新入场条件。
- **R14退出成交**：W1或W2周末收盘后才判断，最早下一交易日开盘卖出；停牌和一字跌停顺延，不用当周收盘价虚构成交。
- **R14三种退出**：分别检验W1收盘亏损、W1/W2连续收盘亏损、W2收盘不高于-5%；主规则预先指定为“W1/W2连续收盘亏损”。
- **R14持有上限**：同时报告W3、W4、W6、W8固定上限；退出规则只允许提前离场，不延长相应持有上限。
- **R7/R9对照**：旧触发和旧排名仅保留在报告中用于比较，不下单、不影响R11。
- **指标口径**：SKDJ固定N=6、M=3；Raw RSV两次EMA(span=3)得到K，D为K的3周简单均线。
- **删除硬门**：不再要求价格达到MA10的75%，不再用1周中位涨幅和55%上涨家数整周归零；市场只做分层审计。
- **实际排名**：两周涨幅、价格/MA10、K6、8周相对强度、MACD冲量五项均按越早越优等权；R5旧100分只保留对照。
- **防追高**：单周涨幅超过25%或收盘距离本周低点超过40%时只进入过热观察，不参与排名。
- **实际组合**：中性期只运行R3 Top2；弱势期只运行R6-N6 Top2；所有强势周持有现金。
- **研究隔离**：R11、R12、R13、R14以及R7/R9对照均使用独立标记，永远不能进入R3/R6实际收益或持股统计。
- **历史执行**：下一交易日开盘买入；W3为主目标，同时固定观察W1—W8并扣除往返成本。
- **明确排除**：买入后的走势、止损、止盈、移动保护、S/A/B/F结果均不参与入场评分。
            """
        )

    today = _shanghai_now().date()
    default_start = today - timedelta(days=365)
    with st.sidebar:
        st.header("研究配置")
        mode = st.radio(
            "运行模式",
            ["历史R14高弹性持仓生命周期验证", "最新选股预览"],
            index=0,
            help="历史模式只使用完整周线；最新预览允许使用本周未完成周线且不写入回测。",
        )
        start_input = st.date_input("验证开始日期", value=default_start, disabled=mode != "历史R14高弹性持仓生命周期验证")
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
            help="R3/R6实际基线与R11—R14研究样本的W1—W8固定周收益及下一开盘退出收益都会扣除该成本。",
        )

        st.markdown("---")
        try:
            secret_token = st.secrets.get("TUSHARE_TOKEN", "")
        except Exception:
            secret_token = ""
        token_input = st.text_input("Tushare Token", value=secret_token, type="password")

        st.markdown("---")
        clear_market_clicked = st.button("清空行情缓存")
        clear_history_clicked = st.button("清除R14历史结果")
        imported_results = st.file_uploader(
            "导入已下载的R9/R10/R11/R12/R13/R14结果包",
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
    if start_input > end_input and mode == "历史R14高弹性持仓生命周期验证":
        st.error("验证开始日期不能晚于截止日期。")
        return

    if clear_market_clicked:
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.success("行情缓存已清空。")

    if clear_history_clicked:
        with _result_files_transaction(
            [CHECKPOINT_FILE, SCAN_LEDGER_FILE, OPPORTUNITY_FILE]
        ):
            for path in (
                CHECKPOINT_FILE,
                SCAN_LEDGER_FILE,
                OPPORTUNITY_FILE,
            ):
                remove_with_backup(path)
        remove_with_backup(RUN_TASK_FILE)
        st.session_state.pop("r14_preview", None)
        st.success("R14历史结果和断点任务已清除。")

    token_clean = clean_token_str(token_input)
    config_id = make_config_id(min_price, min_mv, max_mv, roundtrip_cost_pct)
    if import_results_clicked and imported_results is not None:
        try:
            import_stats = import_r14_results_zip(
                imported_results.getvalue(), config_id
            )
            inferred_note = (
                "；旧包没有扫描账本，已有候选周已恢复，零候选周会自动重扫"
                if import_stats["ledger_inferred"]
                else ""
            )
            r13_note = (
                "。已包含R13日线快照，可直接查看R13报告"
                if import_stats.get("r13_daily_rows", 0) > 0
                else "。旧结果包不含R13日线快照；R11/R12可直接查看，R13需重新扫描"
            )
            r14_note = (
                f"；其中{import_stats.get('pending_r14_weeks', 0)}周需补算R14下一开盘退出路径"
                if import_stats.get("pending_r14_weeks", 0) > 0
                else "；已包含R14持仓生命周期数据"
            )
            st.success(
                f"已恢复{import_stats['candidate_rows']}条候选、"
                f"{import_stats['known_weeks']}个已知候选周"
                f"{inferred_note}{r13_note}{r14_note}。"
            )
        except Exception as exc:
            st.error(f"结果包恢复失败：{exc}")
    is_preview_mode = mode == "最新选股预览"
    if "r14_worker_id" not in st.session_state:
        st.session_state["r14_worker_id"] = uuid.uuid4().hex
    worker_id = str(st.session_state["r14_worker_id"])
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

    start_label = "运行最新选股预览" if is_preview_mode else "启动历史R14高弹性持仓生命周期验证"
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
                    # 保留R1/R2稳定的420日指标预热窗口；每批只装载3个扫描周。
                    fetch_start = (
                        datetime.strptime(min(batch_dates), "%Y%m%d") - timedelta(days=420)
                    ).strftime("%Y%m%d")
                    requested_fetch_end = datetime.strptime(max(batch_dates), "%Y%m%d") + timedelta(days=75)
                    data_ready_date = _latest_data_ready_date()
                    fetch_end = min(
                        requested_fetch_end.date(), data_ready_date
                    ).strftime("%Y%m%d")
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
                    progress = st.progress(0, text="开始扫描R14实际基线、高弹性排名与退出路径……")
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
                                [CHECKPOINT_FILE, OPPORTUNITY_FILE, SCAN_LEDGER_FILE]
                            ):
                                replace_checkpoint_date(
                                    pd.DataFrame(), signal_date, run_config_id
                                )
                                replace_opportunity_date(
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
                        for frame in (candidates, major_winners):
                            if not frame.empty:
                                frame["Market_Data_Gap_Count"] = len(batch_gap_dates)
                                frame["Market_Data_Gap_Dates"] = ",".join(
                                    batch_gap_dates
                                )
                                frame["Backtest_Data_Complete"] = not bool(
                                    batch_gap_dates
                                )
                        selected_count = (
                            int(_bool_series(candidates, "Selected_Top2").sum())
                            if not candidates.empty
                            else 0
                        )
                        if run_preview:
                            st.session_state["r14_preview"] = candidates
                        else:
                            if not candidates.empty:
                                candidates["Config_ID"] = run_config_id
                            if not major_winners.empty:
                                major_winners["Config_ID"] = run_config_id
                            if not refresh_task_lease(
                                str(active_task.get("Task_ID", "")), worker_id
                            ):
                                raise RuntimeError("任务租约已经转移，本页停止写入回测断点。")
                            with _result_files_transaction(
                                [CHECKPOINT_FILE, OPPORTUNITY_FILE, SCAN_LEDGER_FILE]
                            ):
                                replace_checkpoint_date(
                                    candidates, signal_date, run_config_id
                                )
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
                                f"{signal_date}：R14结构与研究候选{raw_count}只，"
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
                            st.success("历史R14高弹性持仓生命周期扫描完成。")
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

    preview = st.session_state.get("r14_preview")
    if is_preview_mode and isinstance(preview, pd.DataFrame):
        st.markdown("---")
        st.header("最新选股预览")
        if preview.empty:
            st.info("最新交易日没有R14结构或研究观察候选。")
        else:
            selected_preview = preview[_bool_series(preview, "Selected_Top2")].copy()
            if selected_preview.empty:
                block_reason = ""
                if "Selection_Block_Reason" in preview.columns and not preview.empty:
                    block_reason = str(preview["Selection_Block_Reason"].iloc[0] or "")
                st.warning(block_reason or "本周没有形成有效入选组。")
                research_preview = preview[
                    _bool_series(preview, "R11_Strong_Research_Top1")
                ].copy()
                if not research_preview.empty:
                    st.caption("以下是R11强势温和收缩Top1，只记录未来收益，不属于买入名单。")
                    research_columns = [
                        "Signal_Date", "R11_Strong_Rank", "name", "ts_code", "Industry",
                        "Strategy_Branch", "ATR_Contraction", "R11_ATR_Band_Pass",
                        "R11_Strong_Research_Top1", "Strong_Reacceleration_100",
                        "Selection_Block_Reason",
                    ]
                    st.dataframe(
                        research_preview[
                            [
                                column
                                for column in research_columns
                                if column in research_preview.columns
                            ]
                        ],
                        width="stretch",
                        hide_index=True,
                    )
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
            r12_preview = preview[
                _bool_series(preview, "R12_Recovery_Repair_Top2")
            ].copy()
            if not r12_preview.empty:
                st.caption(
                    "以下是R12弱势深跌结构+价格修复Top2，只作研究对照，"
                    "不属于实际买入名单。"
                )
                r12_columns = [
                    "Signal_Date",
                    "R12_Recovery_Repair_Rank",
                    "name",
                    "ts_code",
                    "Industry",
                    "Score_Pullback_15",
                    "Price_to_MA10_Ratio",
                    "Weekly_SKDJ_K6",
                    "MACD_Impulse_Pct",
                    "R12_Recovery_Repair_Top2",
                ]
                st.dataframe(
                    r12_preview[
                        [column for column in r12_columns if column in r12_preview.columns]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            r13_preview = preview[
                _bool_series(preview, "R13_Daily_Restart_Top2")
            ].copy()
            if not r13_preview.empty:
                st.caption(
                    "以下是R13五项日线重启质量Top2，只作挑战排名研究，"
                    "不属于实际买入名单。"
                )
                r13_columns = [
                    "Signal_Date",
                    "R13_Daily_Restart_Rank",
                    "name",
                    "ts_code",
                    "Industry",
                    "R13_Daily_Restart_100",
                    "Daily_Close_to_MA20_Ratio",
                    "Daily_MA5_Slope_3D_pct",
                    "Daily_MACD_Hist_Delta_pct",
                    "Daily_Return_5D_pct",
                    "Daily_RS_5D_Pct",
                    "Daily_Higher_Low_5D_pct",
                    "Daily_Close_to_Prior_10D_High_Ratio",
                    "R13_Daily_Restart_Top1",
                    "R13_Daily_Restart_Top2",
                ]
                st.dataframe(
                    r13_preview[
                        [column for column in r13_columns if column in r13_preview.columns]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            r14_preview = preview[
                _bool_series(preview, "R14_MACD_Elastic_Top2")
            ].copy()
            if not r14_preview.empty:
                st.caption(
                    "以下是R14日线MACD柱加速度高弹性Top2，只作持仓生命周期研究，"
                    "不属于R3/R6实际买入名单。"
                )
                r14_columns = [
                    "Signal_Date",
                    "R14_MACD_Elastic_Rank",
                    "name",
                    "ts_code",
                    "Industry",
                    "Daily_MACD_Hist_Delta_pct",
                    "Daily_Return_5D_pct",
                    "Daily_Close_to_MA20_Ratio",
                    "Weekly_SKDJ_K6",
                    "R14_MACD_Elastic_Top1",
                    "R14_MACD_Elastic_Top2",
                ]
                st.dataframe(
                    r14_preview[
                        [column for column in r14_columns if column in r14_preview.columns]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            with st.expander("查看全部实际候选、研究观察及未入选原因"):
                st.dataframe(preview, width="stretch")

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

        ledger = raw_ledger.copy()
        if not ledger.empty and "Config_ID" in ledger.columns:
            ledger = ledger[
                ledger["Config_ID"].astype(str) == report_config_id
            ].copy()
        data_gap_rows = market_data_gap_audit(ledger)
        pending_r13_rows = data_gap_rows[
            data_gap_rows.get(
                "Scan_Status", pd.Series("", index=data_gap_rows.index)
            ).astype(str).eq("PENDING_R13_DAILY")
        ].copy()
        pending_r14_rows = data_gap_rows[
            data_gap_rows.get(
                "Scan_Status", pd.Series("", index=data_gap_rows.index)
            ).astype(str).eq("PENDING_R14_LIFECYCLE")
        ].copy()
        actual_data_gap_rows = data_gap_rows[
            ~data_gap_rows.index.isin(
                pending_r13_rows.index.union(pending_r14_rows.index)
            )
        ].copy()
        state_consistency_rows = result_state_consistency_audit(history, ledger)
        if not state_consistency_rows.empty:
            st.markdown("---")
            st.error(
                f"发现{len(state_consistency_rows)}周账本与候选明细不一致。"
                "当前结果禁止判定策略优劣，也不提供正式下载。"
                "点击“启动历史R14高弹性持仓生命周期验证”后，程序会只补扫这些日期。"
            )
            st.dataframe(
                state_consistency_rows, width="stretch", hide_index=True
            )
            return

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
        gates = pd.concat(
            [
                gates,
                pd.DataFrame(
                    [
                        {
                            "验收项目": "行情缺失或跳过周数必须为0",
                            "结果": "通过" if actual_data_gap_rows.empty else "未通过",
                            "当前值": f"当前{len(actual_data_gap_rows)}周",
                        },
                        {
                            "验收项目": "扫描账本与候选明细必须完全一致",
                            "结果": (
                                "通过" if state_consistency_rows.empty else "未通过"
                            ),
                            "当前值": f"当前{len(state_consistency_rows)}个异常周",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        neutral_gates = neutral_research_gates(completed)
        strong_gates = strong_research_gates(history)
        reacceleration_diagnostics = reacceleration_ranking_diagnostics(history)
        reacceleration_gates = reacceleration_research_gates(
            history, reacceleration_diagnostics
        )
        r11_audit = r11_strong_candidate_audit(history)
        r11_diagnostics = r11_strong_ranking_diagnostics(history)
        r11_gates = r11_strong_research_gates(history)
        recovery_gates = recovery_research_gates(completed)
        opportunity_summary = major_winner_coverage_summary(opportunities)
        trigger_comparison = recovery_trigger_comparison_audit(history)
        market_context_audit = recovery_market_context_audit(history)
        r12_audit = r12_recovery_repair_candidate_audit(history)
        r12_diagnostics = r12_recovery_repair_ranking_diagnostics(history)
        r12_gates = r12_recovery_repair_gates(history)
        r12_factor_comparison = r12_recovery_factor_comparison_audit(history)
        r13_audit = r13_daily_restart_candidate_audit(history)
        r13_diagnostics = r13_daily_restart_ranking_diagnostics(history)
        r13_gates = r13_daily_restart_gates(history)
        r13_factor_comparison = r13_daily_restart_factor_comparison_audit(history)
        r14_candidate_audit = r14_macd_elastic_candidate_audit(history)
        r14_exit_audit = r14_exit_classification_audit(history)
        r14_lifecycle_audit = r14_lifecycle_return_audit(history)
        r14_yearly_audit = r14_yearly_lifecycle_audit(history)
        r14_gates = r14_lifecycle_acceptance_gates(history)

        st.markdown("---")
        st.header("R14 MACD高弹性持仓生命周期验证报告")
        st.caption(
            "R14入场严格复用R13已经预先声明的MACD柱加速度单因子，"
            "不改R3/R6实际组合。退出在W1/W2周末确认，按下一可交易日开盘成交；"
            "W3、W4、W6、W8只统计已经走满相应期限的同批可比样本。"
        )

        scanned_weeks = len(ledger)
        ledger_status = ledger.get(
            "Scan_Status", pd.Series("COMPLETED", index=ledger.index)
        ).astype(str)
        pending_r13_weeks = int(ledger_status.eq("PENDING_R13_DAILY").sum())
        pending_r14_weeks = int(ledger_status.eq("PENDING_R14_LIFECYCLE").sum())
        data_gap_weeks = int(
            (
                ledger_status.ne("COMPLETED")
                & ~ledger_status.isin(
                    {"PENDING_R13_DAILY", "PENDING_R14_LIFECYCLE"}
                )
            ).sum()
        )
        skipped_data_weeks = int(ledger_status.eq("SKIPPED_DATA_GAP").sum())
        invalid_selection_weeks = (
            int(
                (
                    pd.to_numeric(ledger.get("Selected_Count"), errors="coerce")
                    .fillna(0)
                    .lt(1)
                    & ~ledger_status.eq("SKIPPED_DATA_GAP")
                ).sum()
            )
            if not ledger.empty
            else 0
        )
        selected = completed[_actual_selected_mask(completed)]
        selected_returns = selected[PRIMARY_RETURN_COLUMN] if not selected.empty else pd.Series(dtype=float)
        metric_columns = st.columns(9)
        metric_columns[0].metric("账本已知周数", scanned_weeks)
        metric_columns[1].metric("待补R13日线周", pending_r13_weeks)
        metric_columns[2].metric("待补R14退出周", pending_r14_weeks)
        metric_columns[3].metric("数据缺口周", data_gap_weeks)
        metric_columns[4].metric("跳过扫描周", skipped_data_weeks)
        metric_columns[5].metric("无有效入选周", invalid_selection_weeks)
        metric_columns[6].metric("W3完整入选交易", len(selected))
        metric_columns[7].metric(
            "实际入选胜率",
            f"{((selected_returns > 0).mean() * 100.0 if len(selected_returns) else np.nan):.1f}%",
        )
        metric_columns[8].metric(
            "W3实际入选中位收益",
            f"{(selected_returns.median() if len(selected_returns) else np.nan):.2f}%",
        )

        if not pending_r13_rows.empty:
            st.info(
                f"已恢复旧结果中的{len(pending_r13_rows)}周R3/R6、R11/R12数据；"
                "这些周缺少R13日线快照，启动R13后会逐周替换，不会被当作已完成。"
            )
        if not pending_r14_rows.empty:
            st.info(
                f"已恢复R13结果中的{len(pending_r14_rows)}个R14信号周；"
                "这些周尚缺W1/W2确认后的下一可交易日开盘退出路径，启动R14后只补扫相关日期。"
            )
        if not actual_data_gap_rows.empty:
            st.error(
                "回测已继续完成，但存在行情缺失或跳过周；这些周不会被伪装成完整样本，"
                "当前结果不能进入实盘判断。"
            )
            with st.expander("查看行情缺失与跳过明细", expanded=True):
                st.dataframe(actual_data_gap_rows, width="stretch", hide_index=True)

        st.subheader("R11.1原实际组合总体验收（仅R3/R6基线）")
        st.dataframe(gates, width="stretch", hide_index=True)
        st.subheader("R3中性分支独立验收")
        st.dataframe(neutral_gates, width="stretch", hide_index=True)
        st.subheader("R6弱势首次转折分支独立验收")
        st.dataframe(recovery_gates, width="stretch", hide_index=True)
        st.subheader("R14高弹性与持仓生命周期验收（只研究）")
        st.caption(
            "前8项检验高弹性入场是否稳定；后4项检验主退出规则“W1/W2连续收盘亏损”。"
            "任何一项失败都只说明继续研究，不会回头修改R3/R6或按结果挑股票。"
        )
        if _bool_series(history, "R14_Lifecycle_Data_Available").sum() == 0:
            st.info(
                "当前结果尚无R14下一开盘退出路径；高弹性入场前8项可以先查看，"
                "退出相关项目需启动R14补算后判断。"
            )
        st.dataframe(r14_gates, width="stretch", hide_index=True)
        st.subheader("R12弱势深跌结构+价格修复排名验收（只研究）")
        st.caption(
            "固定顺序：Score_Pullback_15从低到高；同分时Price_to_MA10_Ratio"
            "从高到低。没有阈值、没有第二套引擎，也不替换R6原实际名次。"
        )
        st.dataframe(r12_gates, width="stretch", hide_index=True)
        st.subheader("R13日线重启质量挑战排名验收（只研究）")
        st.caption(
            "固定五项等权：价格/MA20、MA5三日斜率、MACD柱加速度、"
            "全池五日相对强度、五日低点抬升。没有硬门，不覆盖R6或R12名次。"
        )
        if _bool_series(history, "Daily_Restart_Data_Available").sum() == 0:
            st.info("当前导入的是R12或更早结果包，不含信号日日线快照；需运行R13扫描后才能验收。")
        st.dataframe(r13_gates, width="stretch", hide_index=True)
        st.subheader("R11强势温和ATR收缩Top1验收（只研究）")
        st.caption(
            "固定顺序是：先按ATR3/ATR13排名，再检查第一名是否位于0.70—0.90；"
            "第一名失败时不补第二名。"
        )
        st.dataframe(r11_gates, width="stretch", hide_index=True)
        st.subheader("R7早期强势回调研究观察（不交易）")
        st.dataframe(strong_gates, width="stretch", hide_index=True)
        st.subheader("R9强势再启动失败对照（不交易）")
        st.dataframe(reacceleration_gates, width="stretch", hide_index=True)
        all_overall_passed = not gates.empty and gates["结果"].eq("通过").all()
        all_neutral_passed = (
            not neutral_gates.empty and neutral_gates["结果"].eq("通过").all()
        )
        all_recovery_passed = (
            not recovery_gates.empty and recovery_gates["结果"].eq("通过").all()
        )
        if (
            all_overall_passed
            and all_neutral_passed
            and all_recovery_passed
        ):
            st.success(
                "R3/R6实际分支与总体门槛均通过；强势仍保持空仓，可进入跨年度样本外验证。"
            )
        else:
            st.error("R3、R6或总体门槛尚未全部通过；R11—R14研究信号均禁止进入实盘。")

        st.subheader("R3与R6实际分支分别表现")
        st.dataframe(_format_report_frame(branches), width="stretch", hide_index=True)

        st.subheader("R11强势Top1候选审计")
        st.dataframe(_format_report_frame(r11_audit), width="stretch", hide_index=True)

        st.subheader("R11的ATR排名是否有效")
        if r11_diagnostics:
            st.dataframe(
                pd.DataFrame(
                    [{"指标": key, "数值": value} for key, value in r11_diagnostics.items()]
                ),
                width="stretch",
                hide_index=True,
            )

        st.subheader("R7抗跌新高、过热和观察候选审计")
        st.dataframe(_format_report_frame(strong_audit), width="stretch", hide_index=True)

        st.subheader("强势市场四种背景同场对照")
        st.caption("全部强势周实际空仓；表内R7名次全部属于反事实研究观察。")
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

        st.subheader("强势扩张：R9失败对照、强制Top2、R7与R3同场对照")
        st.dataframe(
            _format_report_frame(reexpansion_comparison),
            width="stretch",
            hide_index=True,
        )

        st.subheader("R9失败对照排名是否有效")
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

        st.subheader("R12弱势修复Top2与R6原排名同场对照")
        st.dataframe(_format_report_frame(r12_audit), width="stretch", hide_index=True)

        st.subheader("R12弱势排名是否真的有效")
        if r12_diagnostics:
            st.dataframe(
                pd.DataFrame(
                    [{"指标": key, "数值": value} for key, value in r12_diagnostics.items()]
                ),
                width="stretch",
                hide_index=True,
            )

        st.subheader("R12预先声明的弱势排名方案对照")
        st.caption("同时展示失败基线和两个单因子对照，避免只报告胜出的组合顺序。")
        st.dataframe(
            _format_report_frame(r12_factor_comparison),
            width="stretch",
            hide_index=True,
        )

        st.subheader("R13日线挑战Top1、第二名与R6/R12同周对照")
        st.dataframe(_format_report_frame(r13_audit), width="stretch", hide_index=True)

        st.subheader("R13日线挑战排名是否真的有效")
        if r13_diagnostics:
            st.dataframe(
                pd.DataFrame(
                    [{"指标": key, "数值": value} for key, value in r13_diagnostics.items()]
                ),
                width="stretch",
                hide_index=True,
            )

        st.subheader("R13预先声明的日线单因子与组合排名对照")
        st.caption("所有方案使用完全相同的R6合格周，完整展示失败项，不事后挑选赢家。")
        st.dataframe(
            _format_report_frame(r13_factor_comparison),
            width="stretch",
            hide_index=True,
        )

        st.subheader("R14 MACD高弹性Top2与同周R6对照")
        st.caption(
            "R14仅把R13对照表中已经预先声明的日线MACD柱加速度单因子独立冻结，"
            "仍限于同周R6合格池，不读取未来收益。"
        )
        st.dataframe(
            _format_report_frame(r14_candidate_audit),
            width="stretch",
            hide_index=True,
        )

        st.subheader("R14三种退出规则对F级与S/A级的识别")
        st.caption(
            "判断使用W1或W2周末收盘；执行使用下一可交易日开盘。"
            "F级捕获率越高越好，S/A误杀率越低越好。"
        )
        st.dataframe(
            _format_report_frame(r14_exit_audit),
            width="stretch",
            hide_index=True,
        )

        st.subheader("R14 W3/W4/W6/W8持有上限与提前退出")
        st.dataframe(
            _format_report_frame(r14_lifecycle_audit),
            width="stretch",
            hide_index=True,
        )

        st.subheader("R14分年持仓生命周期稳定性")
        st.dataframe(
            _format_report_frame(r14_yearly_audit),
            width="stretch",
            hide_index=True,
        )

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

        st.subheader("R3/R6实际完整入选组")
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
                "R13_Daily_Restart_Rank", "R13_Daily_Restart_100",
                "Daily_Close_to_MA20_Ratio", "Daily_MA5_Slope_3D_pct",
                "Daily_MACD_Hist_Delta_pct", "Daily_Return_5D_pct",
                "Daily_RS_5D_Pct", "Daily_Higher_Low_5D_pct",
                "Daily_Close_to_Prior_10D_High_Ratio",
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

        with st.expander("查看R14高弹性入场与下一开盘退出明细"):
            r14_detail_columns = [
                "Signal_Date", "Entry_Date", "R14_MACD_Elastic_Rank",
                "name", "ts_code", "Industry", "Daily_MACD_Hist_Delta_pct",
                "Daily_Return_5D_pct", "Daily_Close_to_MA20_Ratio",
                "Weekly_SKDJ_K6", "Weekly_SKDJ_D6", "Entry_Open",
                "R14_W1_Close_Gross_pct", "R14_W2_Close_Gross_pct",
                "R14_Trigger_W1_Close_Loss", "R14_W1_Next_Open_Exit_Date",
                "R14_W1_Next_Open_Return_Net_pct", "R14_W1_Exit_Delay_Days",
                "R14_Trigger_W1_W2_Both_Loss", "R14_Trigger_W2_Close_Minus5",
                "R14_W2_Next_Open_Exit_Date", "R14_W2_Next_Open_Return_Net_pct",
                "R14_W2_Exit_Delay_Days", "Fixed_Return_W3_Net_pct",
                "Fixed_Return_W4_Net_pct", "Fixed_Return_W6_Net_pct",
                "Fixed_Return_W8_Net_pct", "MFE_W3_Net_pct",
                "MAE_W3_Raw_pct", "Outcome_Grade",
            ]
            r14_detail = history[
                _bool_series(history, "R14_MACD_Elastic_Top2")
            ].copy()
            r14_detail = r14_detail.sort_values(
                ["Signal_Date", "R14_MACD_Elastic_Rank"],
                ascending=[False, True],
                kind="mergesort",
            )
            st.dataframe(
                r14_detail[
                    [
                        column
                        for column in r14_detail_columns
                        if column in r14_detail.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
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
            ledger.drop(columns=["Config_ID"], errors="ignore"),
            data_gap_rows,
            neutral_gates,
            r11_audit,
            r11_gates,
            r11_diagnostics,
            state_consistency_rows,
            r12_audit,
            r12_gates,
            r12_diagnostics,
            r12_factor_comparison,
            r13_audit,
            r13_gates,
            r13_diagnostics,
            r13_factor_comparison,
            r14_candidate_audit,
            r14_exit_audit,
            r14_lifecycle_audit,
            r14_yearly_audit,
            r14_gates,
        )
        st.download_button(
            "下载R14高弹性持仓生命周期完整研究结果",
            data=export_bytes,
            file_name="r14_macd_elastic_lifecycle_audit_results.zip",
            mime="application/zip",
        )

if __name__ == "__main__":
    main()
