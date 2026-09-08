# -*- coding: utf-8 -*-
"""R28 冻结R27.1规则的分层与随机基准诊断，不优化策略。

R3中性Top2和市场三分法保持不变；R6正式Top1，第二名保留影子。
强市完整整理再启动池先筛ATR3/ATR13在0.70—0.90，再按ATR升序取Top1。
R7与R11第二名只做影子观察；所有正式信号采用无限资金等额独立成交。
正式结果保持不变；增加全基础池B10标签、同日分层及随机对照，不自动选优。
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

APP_VERSION = "R28-FROZEN-EDGE-DIAGNOSTIC"
APP_TITLE = "R28 同条件选股增益诊断"
ENGINE_PATCH = "R27.1-FIXED-WEEK-WINDOW"
# R11正式强市入口与R22不同，必须使用新的配置身份和结果文件；行情缓存继续复用。
STRATEGY_CONFIG_VERSION = "R27.1-FROZEN13-FIXED60W-FLOAT64"

CHECKPOINT_FILE = "r28_candidates.csv"
SCAN_LEDGER_FILE = "r28_scanned_dates.csv"
RUN_TASK_FILE = "r28_running_task.json"
RESULT_STATE_GUARD_FILE = "r28_result_state.guard"
FROZEN_POOL_FILE = "r27_1_frozen_tech_pool.json"
SIGNAL_WINDOW_WEEKS = 60
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
REACCEL_MIN_PREVIOUS_RETURN_PCT = -8.0
REACCEL_MAX_PREVIOUS_RETURN_PCT = 5.0
REACCEL_MAX_WEEKLY_RETURN_PCT = 12.0
REACCEL_MAX_DISTANCE_MA20_PCT = 25.0
REACCEL_MAX_WEEKLY_RANGE_PCT = 25.0
REACCEL_MIN_CLOSE_LOCATION = 0.60
R11_ATR_CONTRACTION_MIN = 0.70
R11_ATR_CONTRACTION_MAX = 0.90
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
R27_BOOTSTRAP_REPETITIONS = 2000
R27_BOOTSTRAP_BLOCK_WEEKS = 4
R27_EXIT_SCHEMES = {
    "B10": "10%硬止损＋W3（可成交基准）",
    "L5": "收盘跌破买价5%→次日开盘退出",
    "W1": "第5交易日收盘未盈利→次日开盘退出",
}
R28_SCHEMA = "R28-EDGE-1"
R28_RANDOM_SEED = 280901
R28_RANDOM_REPETITIONS = 2000

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
            return pd.read_csv(candidate, encoding="utf-8-sig", low_memory=False, float_precision="round_trip")
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
def load_frozen_tech_whitelist(token):
    """冻结首次取得的研究池与行业映射。清除回测结果不删除该文件。"""
    payload = read_json_safe(FROZEN_POOL_FILE)
    if payload:
        codes = payload.get("codes", [])
        names, industries = payload.get("names", {}), payload.get("industries", {})
        if not codes or set(codes) != set(names) or set(codes) != set(industries):
            raise RuntimeError("冻结股票池文件不完整，不能静默替换研究池。")
        expected = hashlib.sha256(json.dumps([sorted(codes), names, industries], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if payload.get("hash") != expected:
            raise RuntimeError("冻结股票池校验失败，停止混用股票池。")
        return set(codes), names, industries
    codes, names, industries = load_custom_tech_whitelist(token)
    if codes:
        payload = {"codes": sorted(codes), "names": names, "industries": industries}
        payload["hash"] = hashlib.sha256(json.dumps([sorted(codes), names, industries], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        atomic_write_json(payload, FROZEN_POOL_FILE)
    return codes, names, industries


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
        # 保存真实输入；以后每个信号日自行补换手率，不使用批次范围内的填充值。
        for column in ("turnover_rate", "circ_mv"):
            stock["source_" + column] = pd.to_numeric(stock.get(column, pd.Series(np.nan, index=stock.index)), errors="coerce")
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
                stock[column] = pd.to_numeric(stock[column], errors="coerce").astype("float64")
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
    for stock in stocks.values():
        stock.attrs["Requested_Start"] = str(start_date)
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

def signal_window_start(signal_date):
    day = datetime.strptime(str(signal_date), "%Y%m%d")
    monday = day - timedelta(days=day.weekday())
    return (monday - timedelta(weeks=SIGNAL_WINDOW_WEEKS - 1)).strftime("%Y%m%d")


def canonical_signal_stock(stock, signal_date):
    """固定60个日历周；只用该信号日及以前数据确定指标与价格锚点。

    批次先前计算的连续价格、换手率不得用作输入。信号日之后的路径
    只向前递推，不参与历史指标的初始化。上游仍必须提供完整市场日历。
    """
    out = stock.loc[stock.index >= signal_window_start(signal_date)].copy().sort_index()
    if out.empty:
        return out
    raw = {}
    for c in ("open", "high", "low", "close"):
        raw[c] = pd.to_numeric(out.get("raw_" + c, out[c]), errors="coerce").astype("float64")
        out["raw_" + c] = raw[c]
    pre = pd.to_numeric(out.get("raw_pre_close", out.get("pre_close", raw["close"].shift())), errors="coerce")
    pct = pd.to_numeric(out.get("pct_chg", pd.Series(np.nan, index=out.index)), errors="coerce")
    growth = (1 + pct.fillna((raw["close"] / pre - 1) * 100) / 100).where(lambda s: s > 0)
    # 从窗口首日开始，旧批次的首个连续价格不会残留在这里。
    continuous = raw["close"].iloc[0] * growth.fillna(1).iloc[1:].cumprod()
    continuous = pd.concat([pd.Series([raw["close"].iloc[0]], index=out.index[:1]), continuous])
    past = continuous.loc[continuous.index <= str(signal_date)]
    if past.empty or not _safe_float(past.iloc[-1]) > 0:
        return out.iloc[:0]
    anchor_date = past.index[-1]
    continuous = continuous * (raw["close"].loc[anchor_date] / past.iloc[-1])
    scale = continuous / raw["close"].replace(0, np.nan)
    for c in raw:
        out[c] = raw[c] * scale
    out["pre_close"] = continuous.shift()
    # 全部历史填充局限于信号日以前，未来流通股本不能反填历史。
    historical = out.index <= str(signal_date)
    mv = pd.to_numeric(out.get("source_circ_mv", out.get("circ_mv", pd.Series(np.nan, index=out.index))), errors="coerce")
    shares = (mv.loc[historical] * 10000 / raw["close"].loc[historical]).ffill().bfill()
    turnover = pd.to_numeric(out.get("source_turnover_rate", out.get("turnover_rate", pd.Series(np.nan, index=out.index))), errors="coerce")
    estimate = pd.to_numeric(out.loc[historical, "vol"], errors="coerce") * 10000 / shares.replace(0, np.nan)
    out.loc[historical, "turnover_rate"] = turnover.loc[historical].fillna(estimate)
    return out


def _weekly_bars(stock: pd.DataFrame, end_date: str):
    if stock.attrs.get("Requested_Start", "00000000") > signal_window_start(end_date):
        raise RuntimeError(f"{end_date}指标预热数据不足，需从{signal_window_start(end_date)}加载；不能沿用短批次起点。")
    stock = canonical_signal_stock(stock, end_date)
    daily = stock[stock.index <= end_date].copy()
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
    weekly["return_6w_pct"] = (close / close.shift(6) - 1.0) * 100.0
    weekly["return_9w_pct"] = (close / close.shift(9) - 1.0) * 100.0
    weekly["return_11w_pct"] = (close / close.shift(11) - 1.0) * 100.0
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
    # 指纹只覆盖信号时点之前的价格/成交量/换手率输入，不包含未来收益。
    fingerprint_cols = [c for c in ("trade_date_str", "open", "high", "low", "close", "vol", "turnover_rate") if c in daily]
    input_bytes = daily[fingerprint_cols].to_csv(index=False, float_format="%.17g").encode()
    weekly.attrs = {}
    weekly.attrs["R271_Input_Hash"] = hashlib.sha256(input_bytes).hexdigest()
    weekly.attrs["R271_Window_Start"] = signal_window_start(end_date)
    weekly.attrs["R271_First_Price_Date"] = str(daily["trade_date_str"].iloc[0])
    weekly.attrs["R271_History_Days"] = len(daily)
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

    # R27恢复R11原始口径。R22简化触发仍单独保存为观察字段，只用于维持
    # R3/R6横截面候选口径和对照，不再进入强市正式交易。
    strong_trend_eligible = bool(
        base_trend_eligible
        and math.isfinite(ma40)
        and math.isfinite(ma20)
        and ma20 >= ma40
    )
    strong_core_observation_trigger = bool(
        strong_trend_eligible
        and all(
            math.isfinite(item)
            for item in (
                current_close,
                previous_high_value,
                previous_return_1w,
                current_hist,
                previous_hist,
            )
        )
        and previous_return_1w <= REACCEL_MAX_PREVIOUS_RETURN_PCT
        and current_close > previous_high_value
        and current_close >= ma10
        and current_hist > previous_hist
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

    # R7只做影子观察：科技池处于早期强势回调时，记录第一次接近13周新高
    # 且不过热的股票。它不能覆盖R11正式信号，也不能进入总收益。
    fresh_breakout = bool(
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
        **weekly.attrs,
        "Is_First_Red": bool(is_first_red),
        "R3_Setup_Candidate": bool(setup_candidate),
        "R3_Setup_Type": setup_type,
        "Fresh_13W_Breakout": fresh_breakout,
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
        "Strong_Core_Observation_Trigger": strong_core_observation_trigger,
        "Strong_Reacceleration_Trigger": strong_reacceleration_trigger,
        "Strong_Reacceleration_Risk_OK": strong_reacceleration_risk_ok,
        "Strong_Reacceleration_Overheated": strong_reacceleration_overheated,
        "Strong_MA10_Confirmed": bool(
            strong_reacceleration_trigger
            and math.isfinite(current_close)
            and math.isfinite(ma10)
            and current_close >= ma10
        ),
        "Strong_Reacceleration_Setup_Type": (
            "整理后再启动-过热观察"
            if strong_reacceleration_overheated
            else "整理后再启动"
            if strong_reacceleration_risk_ok
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
        "Return_6W_pct": _safe_float(current.get("return_6w_pct")),
        "Return_9W_pct": _safe_float(current.get("return_9w_pct")),
        "Return_11W_pct": _safe_float(current.get("return_11w_pct")),
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

def _score_r7_strong_resilience(frame: pd.DataFrame):
    """恢复R7原五项等权影子排名；只使用信号周已知横截面。"""
    scored = frame.copy()
    factors = [
        ("PreSignal_4W_Return_pct", "R7_Score_Pause20", False),
        ("ATR_Contraction", "R7_Score_ATR20", False),
        ("Return_1W_pct", "R7_Score_NonChase20", False),
        ("Industry_13W_Excess_pct", "R7_Score_Industry20", True),
        ("Distance_MA20_pct", "R7_Score_Position20", False),
    ]
    components = []
    for source, target, higher_is_better in factors:
        scored[target] = _percentile_rank(
            _numeric_series(scored, source), higher_is_better=higher_is_better
        ) * 20.0
        components.append(target)
    scored["R7_Strong_Resilience_100"] = scored[components].sum(axis=1)
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

def score_frozen_candidates(pool_snapshots: pd.DataFrame, market_override=None):
    """冻结R3/R6；恢复R11正式Top1，并把R7与R11第二名隔离为影子。"""
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
    if market_override is not None:
        market_state = {**market_state, "Market_Regime": market_override}

    r3_trigger = _bool_series(pool, "R3_Setup_Candidate")
    strong_core_trigger = _bool_series(pool, "Strong_Core_Observation_Trigger")
    strong_trigger = _bool_series(pool, "Strong_Reacceleration_Trigger")
    r7_trigger = _bool_series(pool, "Strong_Resilience_Trigger")
    recovery_trigger = _bool_series(pool, "Recovery_Structure_Trigger")
    # R3/R6仍在与R22完全相同的基础观察集合中计算六因子，避免新增R7影子行
    # 改变它们的横截面百分位。R7独有股票只在基础评分完成后追加。
    baseline_observation = r3_trigger | strong_core_trigger | recovery_trigger
    observation = baseline_observation | r7_trigger
    baseline_scored = pool.loc[baseline_observation].copy()
    if not baseline_scored.empty:
        baseline_scored = _score_r1_six_factors(baseline_scored)
    r7_only = pool.loc[r7_trigger & ~baseline_observation].copy()
    candidates = pd.concat([baseline_scored, r7_only], axis=0, sort=False)
    raw_count = int(observation.sum())
    if candidates.empty:
        return candidates, 0, 0

    candidates["Rank"] = np.nan
    candidates["R3_Rank"] = np.nan
    candidates["Recovery_Rank"] = np.nan
    candidates["R11_Strong_Rank"] = np.nan
    candidates["R7_Strong_Rank"] = np.nan
    candidates["Selected_Top2"] = False
    candidates["R11_ATR_Band_Pass"] = False
    candidates["R11_Strong_Top1"] = False
    candidates["R11_Second_Shadow"] = False
    candidates["R7_Early_Strong_Context"] = False
    candidates["R7_Shadow_Top2"] = False
    candidates["R27_Shadow_Tracked"] = False
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
        candidates.loc[ordered.index, "R11_Strong_Rank"] = np.arange(
            1, len(ordered) + 1, dtype=float
        )

    market_regime = str(market_state.get("Market_Regime", "中性"))
    market_13w = _safe_float(market_state.get("Market_13W_Median_pct"), 0.0)
    market_1w = _safe_float(market_state.get("Market_1W_Median_pct"), 0.0)
    positive_breadth = _safe_float(
        market_state.get("Market_1W_Positive_Breadth"), 0.0
    )
    r7_context = bool(
        MARKET_NEUTRAL_UPPER_PCT <= market_13w < STRONG_EARLY_UPPER_PCT
        and market_1w <= STRONG_RESET_MAX_MARKET_1W_MEDIAN_PCT
        and positive_breadth < STRONG_RESET_MAX_POSITIVE_BREADTH
    )
    candidates["R7_Early_Strong_Context"] = r7_context
    r7_eligible = candidates[_bool_series(candidates, "Strong_Eligible")].copy()
    if not r7_eligible.empty:
        r7_scored = _score_r7_strong_resilience(r7_eligible)
        r7_columns = [
            "R7_Score_Pause20", "R7_Score_ATR20", "R7_Score_NonChase20",
            "R7_Score_Industry20", "R7_Score_Position20",
            "R7_Strong_Resilience_100",
        ]
        for column in r7_columns:
            candidates.loc[r7_scored.index, column] = r7_scored[column]
        r7_ordered = r7_scored.sort_values(
            ["R7_Strong_Resilience_100", "ATR_Contraction", "Return_1W_pct", "ts_code"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        candidates.loc[r7_ordered.index, "R7_Strong_Rank"] = np.arange(
            1, len(r7_ordered) + 1, dtype=float
        )

    r11_atr = pd.to_numeric(candidates["ATR_Contraction"], errors="coerce")
    candidates["R11_ATR_Band_Pass"] = r11_atr.between(
        R11_ATR_CONTRACTION_MIN,
        R11_ATR_CONTRACTION_MAX,
        inclusive="both",
    )
    r11_rank = pd.to_numeric(candidates["R11_Strong_Rank"], errors="coerce")
    candidates["R27_R11_Baseline"] = (
        (market_regime == "强势")
        & r11_rank.eq(1)
        & _bool_series(candidates, "R11_ATR_Band_Pass")
    )
    band_pool = candidates.loc[strong_eligible_mask & candidates["R11_ATR_Band_Pass"]]
    band_order = band_pool.sort_values(["ATR_Contraction", "ts_code"], kind="mergesort")
    candidates["R27_Band_Rank"] = np.nan
    candidates.loc[band_order.index, "R27_Band_Rank"] = np.arange(1, len(band_order) + 1)
    r11_top1 = (market_regime == "强势") & candidates["R27_Band_Rank"].eq(1)
    candidates["R27_R11_Added"] = r11_top1 & ~candidates["R27_R11_Baseline"]
    candidates["R27_R6_Second"] = (
        (market_regime == "弱势") & (len(recovery_eligible) >= MIN_VALID_SELECTION_SIZE)
        & pd.to_numeric(candidates["Recovery_Rank"], errors="coerce").eq(2)
    )
    candidates["R27_R6_First"] = (
        (market_regime == "弱势") & (len(recovery_eligible) >= MIN_VALID_SELECTION_SIZE)
        & pd.to_numeric(candidates["Recovery_Rank"], errors="coerce").eq(1)
    )
    r7_shadow = (
        r7_context
        & _bool_series(candidates, "Strong_Eligible")
        & pd.to_numeric(candidates["R7_Strong_Rank"], errors="coerce").le(TOP_N)
    )
    r11_second_shadow = (
        bool(candidates["R27_R11_Baseline"].any())
        & (market_regime == "强势")
        & strong_eligible_mask
        & r11_rank.eq(2)
    )
    candidates["R11_Strong_Top1"] = r11_top1
    candidates["R11_Second_Shadow"] = r11_second_shadow
    candidates["R7_Shadow_Top2"] = r7_shadow
    candidates["R27_Shadow_Tracked"] = r11_second_shadow | r7_shadow | candidates["R27_R6_Second"]

    r3_count = len(r3_eligible)
    recovery_count = len(recovery_eligible)
    strong_count = len(strong_eligible)
    if market_regime == "强势":
        active_branch = "R11强势温和ATR Top1"
        active_count = strong_count
        candidates["Rank"] = candidates["R27_Band_Rank"]
        candidates["Entry_Eligible"] = strong_eligible_mask
        candidates["R19_Selected"] = r11_top1
        selection_valid = bool(r11_top1.any())
        if selection_valid:
            block_reason = ""
        elif strong_count == 0:
            block_reason = "R11强势没有完整的整理后再启动候选"
        else:
            first_atr = _safe_float(
                candidates.loc[r11_rank.eq(1), "ATR_Contraction"].iloc[0]
            )
            block_reason = "R11完整候选中没有ATR比例处于0.70—0.90的股票"
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
        active_branch = "R6弱势首次转折-N6-Top1"
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
                & pd.to_numeric(candidates["Rank"], errors="coerce").eq(1)
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
    candidates["R7_Strong_Structure_Count"] = int(r7_trigger.sum())
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
        # 只有上涨家数占比是0—1比例，需要转成百分数；市场1W/13W字段
        # 本身已经是百分数，不能因列名以_pct结尾而再次乘100。
        candidates[column] = (
            value * 100.0
            if key == "Market_1W_Positive_Breadth"
            else value
        )

    candidates = candidates.sort_values(
        ["R19_Selected", "R27_Shadow_Tracked", "Entry_Eligible", "Rank", "ts_code"],
        ascending=[False, False, False, True, True],
        na_position="last",
        kind="mergesort",
    )
    return candidates.reset_index(drop=True), raw_count, active_count

# -----------------------------------------------------------------------------
# 买入后固定路径标签：只评价入口，不构造退出策略
# -----------------------------------------------------------------------------
def r27_exit_simulation(path, buy_price, cost, ts_code, scheme):
    """按时间顺序模拟；只用已收盘数据触发软退出，下一开盘执行。

    path包含市场日历中的空行。开盘处于跌停价时保守延后软退出；
    日内硬止损沿用一字跌停不可成交模型。停牌/零量不按填充收盘成交。
    数据末尾尚未成交保持待完成，绝不用最后一天强平。
    """
    if scheme not in R27_EXIT_SCHEMES:
        raise ValueError("未知退出方案")
    out = {"Status": "待完成", "Exit_Date": None, "Exit_Day": np.nan,
           "Return_pct": np.nan, "Reason": "", "Trigger_Date": None,
           "Blocked_Days": 0, "Exit_Price": np.nan}
    if not math.isfinite(buy_price) or buy_price <= 0 or not path:
        out["Status"] = "路径缺失"
        return out
    threshold = .195 if str(ts_code).startswith(("300", "301", "688", "689")) else .095
    prev_raw = np.nan
    pending = None
    stop = buy_price * .90
    def finish(i, bar, price, reason):
        out.update(Status="已退出", Exit_Date=bar["date"], Exit_Day=i + 1,
                   Return_pct=(price / buy_price - 1) * 100 - cost,
                   Reason=reason, Exit_Price=price)
        return out
    for i, bar in enumerate(path):
        op, hi, lo, cl = [_safe_float(bar.get(k)) for k in ("open", "high", "low", "close")]
        ro, rh, rl, rc = [_safe_float(bar.get("raw_" + k), _safe_float(bar.get(k)))
                          for k in ("open", "high", "low", "close")]
        vol = _safe_float(bar.get("vol"))
        valid = all(math.isfinite(v) and v > 0 for v in (op, hi, lo, cl))
        if math.isfinite(vol) and vol <= 0:
            valid = False
        opening_down = valid and math.isfinite(prev_raw) and prev_raw > 0 and ro / prev_raw - 1 <= -threshold
        locked_down = opening_down and np.isclose(rh, rl, rtol=0, atol=max(.001, abs(ro) * 1e-5))
        if i == 0 and not valid:
            out["Status"] = "买入日行情无效"
            return out
        if i > 0:
            # 前收盘信号在开盘执行，先于当日日内止损；开盘滑点不借助当日最低价裁剪。
            if pending and valid and not opening_down:
                return finish(i, bar, op * (1 - R16_STOP_SLIPPAGE_PCT / 100), pending)
            if valid and lo <= stop:
                if not locked_down:
                    reference = min(op, stop)
                    price = max(lo, reference * (1 - R16_STOP_SLIPPAGE_PCT / 100))
                    if not out["Trigger_Date"]:
                        out["Trigger_Date"] = bar["date"]
                    return finish(i, bar, price, "10%硬止损")
                if pending is None:
                    pending = "10%硬止损延后成交"
                    out["Trigger_Date"] = bar["date"]
            if pending and (not valid or opening_down):
                out["Blocked_Days"] += 1
            if i == 14:
                if valid and not locked_down:
                    return finish(i, bar, cl, "W3收盘")
                if pending is None:
                    pending = "W3无法成交后延后退出"
                    out["Trigger_Date"] = bar["date"]
                    out["Blocked_Days"] += 1
        # 固定第5个市场交易日，不能把停牌前价格前填来制造触发。
        if valid and i < 14 and pending is None:
            soft = (scheme == "L5" and cl < buy_price * .95)
            time_fail = (scheme == "W1" and i == 4 and (cl / buy_price - 1) * 100 - cost <= 0)
            if soft or time_fail:
                pending = "收盘亏损超过5%" if soft else "第一周未盈利"
                out["Trigger_Date"] = bar["date"]
        if math.isfinite(rc) and rc > 0:
            prev_raw = rc
    return out


def r27_capture_exit_audit(future, buy_price, cost, ts_code):
    columns = [c for c in ("open", "high", "low", "close", "raw_open", "raw_high",
                          "raw_low", "raw_close", "vol") if c in future]
    bars = future[columns].copy()
    bars.insert(0, "date", future.index.astype(str))
    encoded = bars.to_json(orient="records", double_precision=10)
    path = json.loads(encoded)
    results = {"R27_Exit_Path_JSON": encoded, "R27_Exit_Path_Version": 1}
    for scheme in R27_EXIT_SCHEMES:
        results.update({f"R27_{scheme}_{key}": value for key, value in
                        r27_exit_simulation(path, buy_price, cost, ts_code, scheme).items()})
    return results


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
        "R27_Exit_Path_JSON": "",
        "R27_Exit_Path_Version": 1,
        **{f"R27_{scheme}_{key}": value for scheme in R27_EXIT_SCHEMES
           for key, value in {"Status": "未成交", "Exit_Date": None, "Exit_Day": np.nan,
                              "Return_pct": np.nan, "Reason": "", "Trigger_Date": None,
                              "Blocked_Days": 0, "Exit_Price": np.nan}.items()},
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
        "R19_Path_Entry_Open_QFQ": np.nan,
        "R19_Roundtrip_Cost_pct": float(roundtrip_cost_pct),
    }
    stock = stock_qfq_dict.get(ts_code)
    if stock is None:
        result["Entry_Status"] = "无行情"
        return result

    stock = canonical_signal_stock(stock, signal_date)

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
    result.update(r27_capture_exit_audit(future, buy_price, roundtrip_cost_pct, ts_code))
    result["Entry_Open"] = raw_buy_price
    result["Entry_Open_QFQ"] = buy_price
    # 每日净值必须使用生成该条路径时的同尺度买入价。旧版导入后补算的
    # 短窗口连续复权价格，不能与原420日窗口的Entry_Open_QFQ直接相除。
    result["R19_Path_Entry_Open_QFQ"] = buy_price
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
        "strategy": STRATEGY_CONFIG_VERSION,
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
    market_regime: str = "未知",
    r28_audit_json: str = "",
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
                "Market_Regime": str(market_regime or "未知"),
                "Scan_Status": str(scan_status),
                "Market_Data_Gap_Count": len(gap_dates),
                "Market_Data_Gap_Dates": ",".join(gap_dates),
                "Config_ID": config_id,
                "Updated_At": datetime.now().isoformat(timespec="seconds"),
                "R28_Audit_JSON": r28_audit_json,
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
    # 旧包仍可查看原报告，但没有全池对照，不能冒充R28已完成。
    available = match.get("R28_Audit_JSON", pd.Series("", index=match.index)).map(r28_payload_valid)
    # 本轮已记录的行情缺口必须允许任务走完；下次主动启动由原机制重试。
    # 否则缺失扫描日永远没有审计JSON，会在同一运行中无限循环。
    available |= match["Scan_Status"].isin({"SKIPPED_DATA_GAP", "COMPLETED_WITH_GAPS"})
    return set(filter(None, (parse_yyyymmdd(value) for value in match.loc[available, "Signal_Date"])))

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
    lease_heartbeat=None,
):
    pool_records: list[dict[str, Any]] = []
    for pool_index, ts_code in enumerate(whitelist_keys):
        if pool_index % 32 == 0:
            r28_heartbeat(lease_heartbeat)
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
        empty = pd.DataFrame()
        if not is_preview_mode:
            empty.attrs["R28_Audit_JSON"] = r28_pack(pd.DataFrame(), signal_date, "未知", 0)
        return empty, 0, 0
    pool = pd.DataFrame(pool_records)
    candidates, raw_count, eligible_count = score_frozen_candidates(pool)
    if not candidates.empty:
        pool_payload = pool[["ts_code", "Industry", "Raw_Close", "Circ_MV_Billion", "R271_Input_Hash"]].sort_values("ts_code")
        candidates["R271_Full_Pool_Hash"] = hashlib.sha256(pool_payload.to_csv(index=False, float_format="%.17g").encode()).hexdigest()
        candidates["R271_Pool_Size"] = len(pool)
        candidates["R271_Whitelist_Hash"] = hashlib.sha256(json.dumps([sorted(whitelist_keys), basic_name_map, industry_map], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

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
                # 正式信号与影子审计共用完全相同的买入、止损和W3退出路径；
                # 影子行只用于比较，R19_Selected仍是进入正式总收益的唯一开关。
                if bool(row.get("R19_Selected", False)) or bool(
                    row.get("R27_Shadow_Tracked", False)
                ):
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
                            "R19_Path_Entry_Open_QFQ": np.nan,
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
    if not is_preview_mode:
        candidates.attrs["R28_Audit_JSON"] = r28_scan_pool(
            pool, candidates, signal_date, stock_qfq_dict, roundtrip_cost_pct,
            market_dates, lease_heartbeat,
        )
    return candidates, raw_count, eligible_count


def r28_heartbeat(callback):
    if callback is not None and callback() is False:
        raise RuntimeError("任务租约已转移，R28停止计算；已完成周仍保留。")


def r28_track_baseline(code, signal_date, raw_close, stocks, cost, market_dates):
    """只算冻结B10，省去全池8周分级和两个已失败的软退出实验。

    买入检查和JSON价格精度与track_w3_future_path保持一致；退出直接调用
    原r27_exit_simulation，不另造一套成交规则。回归测试对照原完整函数。
    """
    out = dict(Entry_Tradable=False, Entry_Date=None, Entry_Status="无行情",
               Status="未成交", Return_pct=np.nan, Exit_Date=None, Exit_Day=np.nan,
               Reason="", Buy_Price=np.nan, Input_Path_Hash="")
    stock = stocks.get(code)
    if stock is None:
        return out
    stock = canonical_signal_stock(stock, signal_date)
    calendar = stock.index if market_dates is None else market_dates
    days = [str(d) for d in calendar if str(d) > signal_date][:HOLD_WEEKS * MARKET_DAYS_PER_WEEK]
    if not days:
        out.update(Entry_Status="等待下一交易日", Status="待完成")
        return out
    out["Entry_Date"] = days[0]
    if days[0] not in stock.index:
        out["Entry_Status"] = "下一交易日停牌或无行情，无法成交"
        return out
    future = stock.reindex(days)
    first = future.iloc[0]
    buy = _safe_float(first.get("open"))
    if not math.isfinite(buy) or buy <= 0:
        out["Entry_Status"] = "下一交易日开盘价缺失"
        return out
    ro, rh, rl, rc = [_safe_float(first.get("raw_" + c), _safe_float(first.get(c)))
                      for c in ("open", "high", "low", "close")]
    threshold = .195 if code.startswith(("300", "301", "688", "689")) else .095
    if (all(math.isfinite(v) for v in (rh, rl, rc))
            and np.isclose(rh, rl, rtol=0, atol=max(.001, ro * 1e-5))
            and rc / raw_close - 1 >= threshold):
        out["Entry_Status"] = "下一交易日一字涨停，无法成交"
        return out
    cols = [c for c in ("open", "high", "low", "close", "raw_open", "raw_high",
                        "raw_low", "raw_close", "vol") if c in future]
    bars = future[cols].copy()
    bars.insert(0, "date", future.index.astype(str))
    encoded = bars.to_json(orient="records", double_precision=10)
    out.update(Entry_Tradable=True, Entry_Status="可成交", Buy_Price=buy,
               Input_Path_Hash=hashlib.sha256(encoded.encode()).hexdigest())
    out.update(r27_exit_simulation(json.loads(encoded), buy, cost, code, "B10"))
    return out


def r28_pack(frame, signal_date, regime, future_days, cost=None):
    data = frame.to_json(orient="split", index=False, double_precision=15)
    return json.dumps(dict(schema=R28_SCHEMA, signal_date=str(signal_date),
                           regime=regime, future_days=int(future_days), cost=cost,
                           count=len(frame), data=data,
                           sha256=hashlib.sha256(data.encode()).hexdigest()),
                      ensure_ascii=False, separators=(",", ":"))


def r28_unpack(raw):
    obj = json.loads(str(raw))
    if not isinstance(obj, dict) or obj.get("schema") != R28_SCHEMA:
        raise ValueError("缺少R28全池对照，需补扫")
    if hashlib.sha256(obj["data"].encode()).hexdigest() != obj["sha256"]:
        raise ValueError("R28全池对照校验和不一致")
    data = json.loads(obj["data"])
    frame = pd.DataFrame(data["data"], columns=data["columns"])
    if len(frame) != obj["count"]:
        raise ValueError("R28全池行数不一致")
    if len(frame):
        required = {"ts_code", "Candidate", "Selected", "Entry_Tradable", "Status", "Return_pct"}
        if not required.issubset(frame) or frame.ts_code.duplicated().any():
            raise ValueError("R28缺少字段或重复股票")
        if (_bool_series(frame, "Selected") & ~_bool_series(frame, "Candidate")).any():
            raise ValueError("正式入选不在分支候选池")
        exited = frame.Status.eq("已退出")
        if not np.isfinite(pd.to_numeric(frame.loc[exited, "Return_pct"], errors="coerce")).all():
            raise ValueError("R28退出收益无效")
    return obj, frame


def r28_payload_valid(raw):
    try:
        r28_unpack(raw)
        return True
    except (ValueError, TypeError, KeyError):
        return False


def r28_scan_pool(pool, candidates, signal_date, stocks, cost, market_dates, heartbeat=None):
    """成员资格只来自已冻结的信号字段；未来路径只作标签，不参与筛选。"""
    scored = candidates.set_index("ts_code") if not candidates.empty else pd.DataFrame()
    regime = _market_state_metrics(pool)["Market_Regime"]
    rows = []
    for i, (_, row) in enumerate(pool.sort_values("ts_code").iterrows()):
        if i % 16 == 0:
            r28_heartbeat(heartbeat)
        code = str(row.ts_code)
        candidate = selected = False
        rank = np.nan
        if code in scored.index:
            s = scored.loc[code]
            candidate = bool(s.get("Entry_Eligible", False))
            # 强市ATR区间是资格，不把区间外股票混进排名对照。
            if regime == "强势":
                candidate = candidate and bool(s.get("R11_ATR_Band_Pass", False))
            selected = bool(s.get("R19_Selected", False))
            rank = _safe_float(s.get("Rank"))
        record = {k: row.get(k) for k in ("ts_code", "name", "Industry", "Raw_Close",
                                          "Circ_MV_Billion", "R271_Input_Hash")}
        record.update(Candidate=candidate, Selected=selected, Rank=rank)
        record.update(r28_track_baseline(code, signal_date, float(row.Raw_Close), stocks, cost, market_dates))
        rows.append(record)
    future_days = sum(str(d) > signal_date for d in market_dates) if market_dates is not None else 0
    return r28_pack(pd.DataFrame(rows), signal_date, regime, future_days, float(cost))


def r28_paired_interval(weekly, numerator, denominator):
    """固定4个扫描周区组；无信号周仍占位置，不能把相邻交易当独立样本。"""
    if np.count_nonzero(denominator) < 8 or len(weekly) < 4:
        return np.nan, np.nan
    n = len(weekly)
    rng = np.random.default_rng(R28_RANDOM_SEED + 4)
    starts = rng.integers(0, n - 3, size=(R28_RANDOM_REPETITIONS, math.ceil(n / 4)))
    idx = (starts[:, :, None] + np.arange(4)).reshape(R28_RANDOM_REPETITIONS, -1)[:, :n]
    den = denominator[idx].sum(axis=1)
    num = numerator[idx].sum(axis=1)
    ratios = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return tuple(np.nanquantile(ratios, [.025, .975]))


@st.cache_data(show_spinner=False, max_entries=2)
def r28_reports(history, ledger):
    """主比较只用相同成熟、有正式信号且全池结果可核实的星期。

    每周三组权重均为正式计划名额k；随机在全部信号日成员中无放回抽k只，
    不成交名额按0，不补位。任何已买未退出/无效路径使整周暂不配对，
    不把早止损股先收入样本。随机区间是条件随机分布，不是过拟合概率。
    """
    checks, weeks, details = [], [], []
    blocks = []
    selected_history = history.loc[_bool_series(history, "R19_Selected")].copy()
    if not selected_history.empty:
        selected_history["Signal_Date"] = selected_history.Signal_Date.map(parse_yyyymmdd)
    ordered_ledger = ledger.sort_values("Signal_Date") if "Signal_Date" in ledger else ledger
    for _, entry in ordered_ledger.iterrows():
        day = parse_yyyymmdd(entry.get("Signal_Date"))
        check = {"信号日": day, "核对": "待补算", "说明": "", "基础池": 0, "候选池": 0,
                 "正式名额": 0, "已成交未退出": 0, "可配对": False}
        try:
            meta, pool = r28_unpack(entry.get("R28_Audit_JSON", ""))
            if meta["signal_date"] != day:
                raise ValueError("信号日期与审计账本不一致")
            if len(pool):
                pool = pool.sort_values("ts_code").reset_index(drop=True)
            selected = _bool_series(pool, "Selected")
            eligible = _bool_series(pool, "Candidate")
            k = int(selected.sum())
            if k != int(_safe_float(entry.get("Selected_Count"), 0)):
                raise ValueError("正式名额与扫描账本不一致")
            original = selected_history.loc[selected_history.Signal_Date.eq(day)] if len(selected_history) else pd.DataFrame()
            if set(original.get("ts_code", [])) != set(pool.loc[selected, "ts_code"] if len(pool) else []):
                raise ValueError("全池对照改变了正式名单")
            for _, old in original.iterrows():
                new = pool.loc[pool.ts_code.eq(old.ts_code)].iloc[0]
                if "R271_Input_Hash" in old and "R271_Input_Hash" in new and str(old.R271_Input_Hash) != str(new.R271_Input_Hash):
                    raise ValueError(f"{old.ts_code}信号输入指纹不一致")
                if bool(new.Entry_Tradable) != bool(old.Entry_Tradable) or str(new.Status) != str(old.get("R27_B10_Status")):
                    raise ValueError(f"{old.ts_code}正式成交状态不一致")
                if new.Status == "已退出":
                    if (parse_yyyymmdd(new.Exit_Date) != parse_yyyymmdd(old.get("R27_B10_Exit_Date"))
                            or not math.isclose(float(new.Return_pct), float(old.R27_B10_Return_pct), abs_tol=1e-8, rel_tol=1e-9)):
                        raise ValueError(f"{old.ts_code}正式B10收益或退出日期不一致")
            tradable = _bool_series(pool, "Entry_Tradable")
            complete = pool.get("Status", pd.Series(dtype=str)).eq("已退出")
            pending = tradable & ~complete
            check.update(基础池=len(pool), 候选池=int(eligible.sum()), 正式名额=k,
                         已成交未退出=int(pending.sum()), 核对="通过")
            if len(pool):
                detail = pool.copy()
                detail.insert(0, "Signal_Date", day)
                detail.insert(1, "Market_Regime", meta["regime"])
                details.append(detail)
            clean = str(entry.get("Scan_Status")) == "COMPLETED" and _safe_float(entry.get("Market_Data_Gap_Count"), 0) == 0
            valid = clean and len(pool) > 0 and meta["future_days"] >= 15 and not pending.any()
            if not clean:
                check["说明"] = "扫描存在行情缺口，整周不参与比较"
            elif meta["future_days"] < 15:
                check["说明"] = "未满15个市场交易日，整周等待成熟"
            elif pending.any():
                check["说明"] = "有买入后未完成/无效路径，整周不参与比较"
            elif not k:
                check["说明"] = "没有正式信号，仅保留全池明细，不纳入主配对"
            if valid and k:
                ret = pd.to_numeric(pool.Return_pct, errors="coerce").where(tradable, 0).to_numpy(float)
                if not np.isfinite(ret).all() or int(eligible.sum()) < k:
                    raise ValueError("可配对收益无效或候选数不足")
                formal = ret[selected.to_numpy()]
                branch = ret[eligible.to_numpy()]
                week = dict(信号日=day, 市场=meta["regime"], 年份=day[:4], 名额=k,
                            基础池均益=float(ret.mean()), 候选池均益=float(branch.mean()),
                            正式均益=float(formal.mean()), 正式成交=int(tradable[selected].sum()))
                for label, values, count in (("基础池", ret, int(tradable.sum())),
                                              ("候选池", branch, int(tradable[eligible].sum())),
                                              ("正式", formal, int(tradable[selected].sum()))):
                    week[label + "名额胜率"] = float((values > 0).mean() * 100)
                    week[label + "成交胜率"] = float((values > 0).sum() / count * 100) if count else np.nan
                    week[label + "成交均益"] = float(values.sum() / count) if count else np.nan
                random_week = {}
                for label, values in (("基础池", ret), ("候选池", branch)):
                    seed = int(hashlib.sha256(f"{R28_RANDOM_SEED}:{day}:{label}".encode()).hexdigest()[:16], 16)
                    rng = np.random.default_rng(seed)
                    # 成员按代码排序，抽样索引不依赖任何未来标签。
                    draw = np.array([rng.choice(len(values), size=k, replace=False)
                                     for _ in range(R28_RANDOM_REPETITIONS)])
                    random_week[label] = values[draw].mean(axis=1)
                check["可配对"] = True
                weeks.append(week)
                blocks.append(random_week)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            check.update(核对="需补算或核查", 说明=str(exc))
        checks.append(check)
    weekly = pd.DataFrame(weeks)
    summary, random_summary, increments = [], [], []
    if len(weekly):
        scopes = [("全部", np.ones(len(weekly), dtype=bool))]
        scopes += [(r, weekly.市场.eq(r).to_numpy()) for r in ("强势", "中性", "弱势")]
        scopes += [(y, weekly.年份.eq(y).to_numpy()) for y in sorted(weekly.年份.unique())]
        scopes += [(y + "·" + r, (weekly.年份.eq(y) & weekly.市场.eq(r)).to_numpy())
                   for y in sorted(weekly.年份.unique()) for r in ("强势", "中性", "弱势")]
        for scope, mask in scopes:
            if not mask.any():
                continue
            weights = weekly.名额.to_numpy(float) * mask
            total = weights.sum()
            formal = np.average(weekly.正式均益, weights=weights)
            for label in ("基础池", "候选池", "正式"):
                summary.append({"范围": scope, "层级": label, "配对周数": int(mask.sum()),
                                "相同计划名额": int(total),
                                "名额加权净收益_pct": np.average(weekly[label + "均益"], weights=weights),
                                "名额加权胜率_pct": np.average(weekly[label + "名额胜率"], weights=weights)})
            for label in ("基础池", "候选池"):
                draws = np.array([b[label] for b in blocks])
                random_values = np.average(draws, axis=0, weights=weights)
                lo, med, hi = np.quantile(random_values, [.025, .5, .975])
                random_summary.append({"范围": scope, "随机来源": label, "重复次数": R28_RANDOM_REPETITIONS,
                                       "配对周数": int(mask.sum()), "正式均益_pct": formal,
                                       "随机均益P2.5_pct": lo, "随机均益P50_pct": med,
                                       "随机均益P97.5_pct": hi,
                                       "正式超过随机比例_pct": float((random_values < formal).mean() * 100)})
            # 完整扫描周索引保留无信号/其他分支周，按日期映射后做配对区组。
            scan_days = [parse_yyyymmdd(c["信号日"]) for c in checks]
            positions = {day: i for i, day in enumerate(scan_days)}
            for label, a, b in (("形态筛选增益", "候选池均益", "基础池均益"),
                                ("排名增益", "正式均益", "候选池均益"),
                                ("总选股增益", "正式均益", "基础池均益")):
                delta = (weekly[a] - weekly[b]).to_numpy(float)
                num, den = np.zeros(len(scan_days)), np.zeros(len(scan_days))
                for i, day in enumerate(weekly.信号日):
                    pos = positions[day]
                    num[pos], den[pos] = delta[i] * weights[i], weights[i]
                lo, hi = r28_paired_interval(scan_days, num, den)
                increments.append({"范围": scope, "比较": label, "配对周数": int(mask.sum()),
                                   "增益_百分点": float(np.average(delta, weights=weights)),
                                   "4周区组区间下限": lo, "4周区组区间上限": hi,
                                   "解释": "描述性诊断，未校正历次策略搜索，不判定通过；少于8周不估区间"})
    return {"31_r28_pool_outcomes.csv": pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
            "32_r28_week_integrity.csv": pd.DataFrame(checks),
            "33_r28_matched_weekly.csv": weekly,
            "34_r28_layer_comparison.csv": pd.DataFrame(summary),
            "35_r28_random_comparison.csv": pd.DataFrame(random_summary),
            "36_r28_paired_increment.csv": pd.DataFrame(increments),
            "37_r28_protocol.csv": pd.DataFrame([{
                "版本": R28_SCHEMA, "冻结策略": STRATEGY_CONFIG_VERSION,
                "随机种子": R28_RANDOM_SEED, "随机次数": R28_RANDOM_REPETITIONS,
                "主口径": "同信号周、同正式计划名额加权；无法买入名额0收益且不补位；非账户收益",
                "候选定义": "当前分支全部合格股票；强市含原ATR0.70—0.90资格，未应用排名和最少候选数门槛",
                "成熟口径": "至少15市场交易日，全基础池所有已买股票B10已退出，行情无缺口，否则整周暂不配对",
                "随机解释": "固定已实现路径条件下的随机分布；非过拟合概率、非未来盈利置信度",
                "数据边界": "当前冻结科技名单及行业分类，不是完整历史时点股票池；价格/市值为信号日数据",
                "决策边界": "不自动调参、不自动停用分支、不授予实盘合格；先比较筛选与排名增益"}])}


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
        "R19_Path_Entry_Open_QFQ",
        "R19_Roundtrip_Cost_pct",
    )
    for index, row in result.loc[selected].iterrows():
        frozen_cost = _safe_float(
            row.get("R19_Roundtrip_Cost_pct"), roundtrip_cost_pct
        )
        outcome = track_w3_future_path(
            str(row.get("ts_code", "")),
            signal_date,
            _safe_float(row.get("Raw_Close")),
            stock_qfq_dict,
            frozen_cost,
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
    """统一三市场入场集合。"""
    if history.empty:
        return history.iloc[0:0].copy()
    if "R19_Selected" in history.columns:
        selected_mask = _bool_series(history, "R19_Selected")
    else:
        selected_mask = (
            _bool_series(history, "Selected_Top2")
            | _bool_series(history, "R11_Strong_Top1")
        )
    selected = history.loc[selected_mask].copy()
    regime = selected.get(
        "Market_Regime", pd.Series("", index=selected.index)
    ).astype(str)
    selected["R19_市场分支"] = regime.map(
        {"强势": "R11强势", "中性": "R3中性", "弱势": "R6弱势"}
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


def _r19_path_ready_mask(frame: pd.DataFrame):
    """路径JSON与其同尺度买入基准必须同时存在。"""
    path = frame.get(
        "R19_Daily_Path_JSON", pd.Series("", index=frame.index)
    ).fillna("").astype(str).str.startswith("[[")
    baseline = pd.to_numeric(
        frame.get(
            "R19_Path_Entry_Open_QFQ",
            pd.Series(np.nan, index=frame.index),
        ),
        errors="coerce",
    )
    return path & baseline.gt(0.0) & np.isfinite(baseline)


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
    for fallback in ("R3_Rank", "Recovery_Rank", "R11_Strong_Rank"):
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
    selected["R19_Path_Available"] = _r19_candidate_path_scale_ready_mask(
        selected
    )
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


def _r19_candidate_path_scale_ready_mask(frame: pd.DataFrame):
    """候选路径不仅要存在，还必须能复算出被冻结的W3收益。"""
    ready = _r19_path_ready_mask(frame).copy()
    if frame.empty:
        return ready
    for index, row in frame.loc[ready].iterrows():
        path = _r19_parse_daily_path(row.get("R19_Daily_Path_JSON", ""))
        baseline = _safe_float(row.get("R19_Path_Entry_Open_QFQ"))
        exit_date = pd.to_datetime(
            str(row.get("Fixed_Exit_W3_Date", "")).replace(".0", ""),
            errors="coerce",
        )
        frozen = _safe_float(row.get("Fixed_Return_W3_Net_pct"))
        consistent = False
        if not path.empty and baseline > 0 and pd.notna(exit_date):
            prices = path.loc[path["Date"].le(exit_date), "Close"]
            if len(prices):
                calculated = (
                    (_safe_float(prices.iloc[-1]) / baseline - 1.0) * 100.0
                    - _safe_float(row.get("R19_Roundtrip_Cost_pct"), 0.20)
                )
                consistent = (
                    math.isfinite(calculated)
                    and math.isfinite(frozen)
                    and abs(calculated - frozen) <= 0.02
                )
        ready.loc[index] = consistent
    return ready


def recover_r19_1_path_baselines(frame: pd.DataFrame):
    """从旧路径W3收盘与冻结收益代数恢复同尺度买入价，不重算交易。"""
    result = frame.copy()
    if "R19_Path_Entry_Open_QFQ" not in result.columns:
        result["R19_Path_Entry_Open_QFQ"] = np.nan
    existing = pd.to_numeric(
        result["R19_Path_Entry_Open_QFQ"], errors="coerce"
    )
    has_path = result.get(
        "R19_Daily_Path_JSON", pd.Series("", index=result.index)
    ).fillna("").astype(str).str.startswith("[[")
    needs_recovery = ~existing.gt(0.0) & has_path
    recovered = 0
    for index, row in result.loc[needs_recovery].iterrows():
        path = _r19_parse_daily_path(row.get("R19_Daily_Path_JSON", ""))
        exit_text = parse_yyyymmdd(row.get("Fixed_Exit_W3_Date"))
        exit_date = pd.to_datetime(
            exit_text, format="%Y%m%d", errors="coerce"
        )
        frozen = _safe_float(row.get("Fixed_Return_W3_Net_pct"))
        cost = _safe_float(row.get("R19_Roundtrip_Cost_pct"), 0.20)
        if path.empty or pd.isna(exit_date) or not math.isfinite(frozen):
            continue
        exact_exit = path.loc[path["Date"].eq(exit_date), "Close"]
        denominator = 1.0 + (frozen + cost) / 100.0
        if not len(exact_exit) or denominator <= 0.0:
            continue
        baseline = _safe_float(exact_exit.iloc[-1]) / denominator
        if math.isfinite(baseline) and baseline > 0.0:
            result.loc[index, "R19_Path_Entry_Open_QFQ"] = baseline
            recovered += 1
    return result, recovered


def r19_w3_path_scale_audit(portfolio_ledger: pd.DataFrame):
    """复算W3路径收益，阻止不同复权尺度生成看似可对账的假净值。"""
    columns = [
        "Signal_Date", "ts_code", "name", "Exit_Date",
        "路径复算收益%", "冻结交易收益%", "差额百分点", "路径尺度一致",
    ]
    if portfolio_ledger.empty:
        return pd.DataFrame(columns=columns)
    rows = portfolio_ledger[
        portfolio_ledger.get(
            "执行状态", pd.Series("", index=portfolio_ledger.index)
        ).astype(str).eq("买入")
        & portfolio_ledger.get(
            "退出原因", pd.Series("", index=portfolio_ledger.index)
        ).astype(str).eq("W3到期")
    ]
    output = []
    for _, row in rows.iterrows():
        path = _r19_parse_daily_path(row.get("R19_Daily_Path_JSON", ""))
        baseline = _safe_float(row.get("R19_Path_Entry_Open_QFQ"))
        exit_date = pd.to_datetime(
            parse_yyyymmdd(row.get("Exit_Date")),
            format="%Y%m%d",
            errors="coerce",
        )
        calculated = np.nan
        if not path.empty and math.isfinite(baseline) and baseline > 0 and pd.notna(exit_date):
            prices = path.loc[path["Date"].le(exit_date), "Close"]
            if len(prices):
                calculated = (
                    (_safe_float(prices.iloc[-1]) / baseline - 1.0) * 100.0
                    - _safe_float(row.get("R19_Roundtrip_Cost_pct"), 0.20)
                )
        frozen = _safe_float(row.get("交易净收益%"))
        difference = calculated - frozen
        consistent = (
            math.isfinite(calculated)
            and math.isfinite(frozen)
            and abs(difference) <= 0.02
        )
        output.append(
            {
                "Signal_Date": row.get("Signal_Date"),
                "ts_code": row.get("ts_code"),
                "name": row.get("name"),
                "Exit_Date": row.get("Exit_Date"),
                "路径复算收益%": calculated,
                "冻结交易收益%": frozen,
                "差额百分点": difference,
                "路径尺度一致": consistent,
            }
        )
    return pd.DataFrame(output, columns=columns)


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
                "R19_Path_Entry_Open_QFQ": _safe_float(
                    row.get("R19_Path_Entry_Open_QFQ")
                ),
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
                "起算方式": "区间首笔信号前三仓均为空，不继承区间外持仓",
                "首笔实际买入日": (
                    str(bought["Entry_Date"].min()) if not bought.empty else ""
                ),
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


def r19_missing_bought_path_dates(history: pd.DataFrame):
    """只要求三仓实际买入的交易具备净值路径；被仓位跳过者不影响账户。"""
    if history.empty:
        return set()
    _, portfolio_ledger, _, _, _ = r19_three_slot_portfolio(
        history, total_capital=PORTFOLIO_CAPITAL_DEFAULT
    )
    if portfolio_ledger.empty:
        return set()
    bought = portfolio_ledger[
        portfolio_ledger.get(
            "执行状态", pd.Series("", index=portfolio_ledger.index)
        ).astype(str).eq("买入")
    ]
    missing = bought.loc[
        ~_bool_series(bought, "R19_Path_Available"), "Signal_Date"
    ]
    return set(filter(None, (parse_yyyymmdd(value) for value in missing)))

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
                # 只能与同一次路径计算生成的复权买入价相除。
                "buy": _safe_float(row["R19_Path_Entry_Open_QFQ"]),
                "cost": _safe_float(row.get("R19_Roundtrip_Cost_pct"), 0.20),
                "path": path.set_index("Date")["Close"],
            }
        )
    for rows in trades_by_slot.values():
        rows.sort(key=lambda item: item["entry"])
    first_date = min(pd.to_datetime(bought["Entry_Date"], errors="coerce"))
    last_date = max(pd.to_datetime(bought["Exit_Date"], errors="coerce"))
    # bdate_range会把春节、国庆等工作日休市错误当成交易日。净值只采用
    # 行情路径中真实出现过的日期；路径未覆盖的纯空仓间隔不虚构交易日数量。
    calendar_values = {first_date, last_date}
    for trades in trades_by_slot.values():
        for trade in trades:
            calendar_values.update(
                day
                for day in trade["path"].index
                if first_date <= day <= last_date
            )
            calendar_values.update((trade["entry"], trade["exit"]))
    calendar = pd.DatetimeIndex(sorted(calendar_values))
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
                "路径覆盖日平均资金暴露%": pd.to_numeric(
                    daily["资金暴露%"], errors="coerce"
                ).mean(),
                "最多同时持仓": int(daily["持仓数"].max()),
                "路径覆盖内空仓日": int(daily["持仓数"].eq(0).sum()),
                "日线审计估值日": len(daily),
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


# -----------------------------------------------------------------------------
# 全信号等权过拟合审计（不构造任何有限仓位或复投资金路径）
# -----------------------------------------------------------------------------
def _r20_summary_row(label: str, group: pd.DataFrame, notional: float):
    returns = pd.to_numeric(
        group.get("R19_Realized_Return_pct", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    gains = returns[returns > 0.0].sum()
    losses = -returns[returns < 0.0].sum()
    return {
        "分组": label,
        "完整交易": len(returns),
        "信号周": int(group["Signal_Date"].nunique()) if len(group) else 0,
        "止损交易": int(
            group.get("R19_Exit_Reason", pd.Series("", index=group.index))
            .astype(str)
            .str.contains("止损")
            .sum()
        ),
        "胜率%": (returns > 0.0).mean() * 100.0 if len(returns) else np.nan,
        "平均收益%": returns.mean() if len(returns) else np.nan,
        "中位收益%": returns.median() if len(returns) else np.nan,
        "收益点合计": returns.sum() if len(returns) else np.nan,
        "Profit_Factor": (
            gains / losses if losses > 0.0 else (np.inf if gains > 0.0 else np.nan)
        ),
        "最差收益%": returns.min() if len(returns) else np.nan,
        "最佳收益%": returns.max() if len(returns) else np.nan,
        "每笔名义本金": float(notional),
        "累计投入名义本金": float(notional) * len(returns),
        "名义总盈亏": float(notional) * returns.sum() / 100.0,
        "投入资金平均收益%": returns.mean() if len(returns) else np.nan,
    }


def r20_all_signal_ledger(history: pd.DataFrame, notional: float):
    """每个完整入选信号都投入相同名义本金；无仓位上限、无复投。"""
    universe = r19_trade_universe(history).copy()
    columns = [
        "Signal_Date", "Entry_Date", "Exit_Date", "Rank", "ts_code", "name",
        "Industry", "市场分支", "退出原因", "交易净收益%", "名义本金", "名义盈亏",
        "Outcome_Grade", "MFE_W3_Net_pct", "MAE_W3_Raw_pct",
    ]
    if universe.empty:
        return pd.DataFrame(columns=columns), universe
    universe["Signal_Date"] = universe["Signal_Date"].map(parse_yyyymmdd)
    universe["Entry_Date"] = universe["R19_Entry_Date"].dt.strftime("%Y%m%d")
    universe["Exit_Date"] = universe["R19_Exit_Date"].dt.strftime("%Y%m%d")
    universe["Rank"] = pd.to_numeric(
        universe["R19_Priority_Rank"], errors="coerce"
    )
    universe["市场分支"] = universe["R19_市场分支"]
    universe["退出原因"] = universe["R19_Exit_Reason"]
    universe["交易净收益%"] = pd.to_numeric(
        universe["R19_Realized_Return_pct"], errors="coerce"
    )
    universe["名义本金"] = float(notional)
    universe["名义盈亏"] = (
        float(notional) * universe["交易净收益%"] / 100.0
    )
    for column in columns:
        if column not in universe.columns:
            universe[column] = np.nan
    ledger = universe[columns].sort_values(
        ["Signal_Date", "Rank", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    return ledger, universe


def r20_group_summaries(universe: pd.DataFrame, notional: float):
    overall_columns = list(_r20_summary_row("合计", universe, notional).keys())
    if universe.empty:
        empty = pd.DataFrame(columns=overall_columns)
        return empty, empty.copy(), empty.copy(), empty.copy()

    total = pd.DataFrame([_r20_summary_row("合计", universe, notional)])

    branch_rows = [
        _r20_summary_row(str(label), group, notional)
        for label, group in universe.groupby("R19_市场分支", sort=False)
    ]
    branch = pd.DataFrame(branch_rows, columns=overall_columns)

    dated = universe.copy()
    signal_dt = pd.to_datetime(
        dated["Signal_Date"].map(parse_yyyymmdd), format="%Y%m%d", errors="coerce"
    )
    dated["_year"] = signal_dt.dt.year.astype("Int64").astype(str)
    dated["_half"] = (
        signal_dt.dt.year.astype("Int64").astype(str)
        + "H"
        + np.where(signal_dt.dt.month <= 6, "1", "2")
    )
    year = pd.DataFrame(
        [
            _r20_summary_row(str(label), group, notional)
            for label, group in dated.groupby("_year", sort=True)
        ],
        columns=overall_columns,
    )
    half = pd.DataFrame(
        [
            _r20_summary_row(str(label), group, notional)
            for label, group in dated.groupby("_half", sort=True)
        ],
        columns=overall_columns,
    )
    return total, branch, year, half


def r20_rank_summary(universe: pd.DataFrame, notional: float):
    if universe.empty:
        return pd.DataFrame()
    ranks = pd.to_numeric(universe["R19_Priority_Rank"], errors="coerce")
    frame = universe.copy()
    frame["_rank_label"] = ranks.map(
        lambda value: f"第{int(value)}名" if math.isfinite(_safe_float(value)) else "未知"
    )
    return pd.DataFrame(
        [
            _r20_summary_row(str(label), group, notional)
            for label, group in frame.groupby("_rank_label", sort=True)
        ]
    )


def r20_weekly_summary(universe: pd.DataFrame):
    columns = [
        "Signal_Date", "市场分支", "入选数", "盈利数", "止损数", "周内胜率%",
        "周内等权平均收益%", "周内中位收益%", "周内最差收益%", "周内最佳收益%",
    ]
    if universe.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for signal_date, group in universe.groupby("Signal_Date", sort=True):
        returns = pd.to_numeric(group["R19_Realized_Return_pct"], errors="coerce").dropna()
        branches = "/".join(sorted(group["R19_市场分支"].dropna().astype(str).unique()))
        rows.append(
            {
                "Signal_Date": parse_yyyymmdd(signal_date),
                "市场分支": branches,
                "入选数": len(returns),
                "盈利数": int((returns > 0.0).sum()),
                "止损数": int(group["R19_Exit_Reason"].astype(str).str.contains("止损").sum()),
                "周内胜率%": (returns > 0.0).mean() * 100.0,
                "周内等权平均收益%": returns.mean(),
                "周内中位收益%": returns.median(),
                "周内最差收益%": returns.min(),
                "周内最佳收益%": returns.max(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def r20_rolling_26week_summary(universe: pd.DataFrame, scan_ledger: pd.DataFrame):
    columns = [
        "窗口截止周", "窗口起始周", "扫描周数", "完整交易", "信号周", "胜率%",
        "平均收益%", "中位收益%", "收益点合计", "Profit_Factor", "最差收益%",
    ]
    if universe.empty or scan_ledger.empty:
        return pd.DataFrame(columns=columns)
    scan_dates = sorted(
        {
            value
            for value in scan_ledger["Signal_Date"].map(parse_yyyymmdd)
            if value
        }
    )
    if len(scan_dates) < 26:
        return pd.DataFrame(columns=columns)
    signal_text = universe["Signal_Date"].map(parse_yyyymmdd)
    rows = []
    for end_index in range(25, len(scan_dates)):
        window_dates = set(scan_dates[end_index - 25 : end_index + 1])
        group = universe.loc[signal_text.isin(window_dates)].copy()
        returns = pd.to_numeric(group["R19_Realized_Return_pct"], errors="coerce").dropna()
        gains = returns[returns > 0.0].sum()
        losses = -returns[returns < 0.0].sum()
        rows.append(
            {
                "窗口截止周": scan_dates[end_index],
                "窗口起始周": scan_dates[end_index - 25],
                "扫描周数": 26,
                "完整交易": len(returns),
                "信号周": int(group["Signal_Date"].nunique()) if len(group) else 0,
                "胜率%": (returns > 0.0).mean() * 100.0 if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "收益点合计": returns.sum() if len(returns) else 0.0,
                "Profit_Factor": gains / losses if losses > 0.0 else np.nan,
                "最差收益%": returns.min() if len(returns) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def r20_concentration_audit(universe: pd.DataFrame):
    columns = ["项目", "当前值", "说明"]
    if universe.empty:
        return pd.DataFrame(columns=columns)
    frame = universe.copy()
    frame["_return"] = pd.to_numeric(
        frame["R19_Realized_Return_pct"], errors="coerce"
    )
    frame = frame.dropna(subset=["_return"]).sort_values("_return", ascending=False)
    net = frame["_return"].sum()
    top1 = frame.head(1)["_return"].sum()
    top5 = frame.head(5)["_return"].sum()
    after1 = frame.iloc[1:]["_return"]
    after5 = frame.iloc[5:]["_return"]
    best = frame.iloc[0]
    return pd.DataFrame(
        [
            {"项目": "全部收益点合计", "当前值": net, "说明": "每笔等额时的净收益百分点之和"},
            {"项目": "第一大盈利交易", "当前值": top1, "说明": f"{best.get('name', '')} / {parse_yyyymmdd(best.get('Signal_Date'))}"},
            {"项目": "第一大盈利占净收益%", "当前值": top1 / net * 100.0 if net > 0 else np.nan, "说明": "越低越不依赖单一牛股"},
            {"项目": "前五大盈利占净收益%", "当前值": top5 / net * 100.0 if net > 0 else np.nan, "说明": "等额口径，不含复投放大"},
            {"项目": "删除第一大后收益点", "当前值": after1.sum(), "说明": f"剩余{len(after1)}笔"},
            {"项目": "删除第一大后平均收益%", "当前值": after1.mean(), "说明": "仍为正才说明不靠一只股票"},
            {"项目": "删除前五大后收益点", "当前值": after5.sum(), "说明": f"剩余{len(after5)}笔"},
            {"项目": "删除前五大后平均收益%", "当前值": after5.mean(), "说明": "仍为正才说明主体样本有贡献"},
        ],
        columns=columns,
    )


def r27_branch_robustness_audit(universe: pd.DataFrame):
    """逐分支检查典型收益、盈亏结构与去除头部盈利后的主体贡献。"""
    columns = [
        "市场分支", "完整交易", "平均收益%", "中位收益%", "Profit_Factor",
        "第一大盈利占净收益%", "前五大盈利占净收益%",
        "删除前五大后收益点", "删除前五大后平均收益%",
        "样本数检查", "平均收益检查", "中位收益检查", "PF检查",
        "去前五大检查", "分支综合结论",
    ]
    rows = []
    for branch in ("R11强势", "R3中性", "R6弱势"):
        group = universe[
            universe.get(
                "R19_市场分支", pd.Series("", index=universe.index)
            ).astype(str).eq(branch)
        ].copy()
        returns = pd.to_numeric(
            group.get("R19_Realized_Return_pct", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna().sort_values(ascending=False).reset_index(drop=True)
        net = returns.sum() if len(returns) else np.nan
        gains = returns[returns > 0.0].sum()
        losses = -returns[returns < 0.0].sum()
        pf = gains / losses if losses > 0.0 else (np.inf if gains > 0.0 else np.nan)
        after5 = returns.iloc[5:] if len(returns) > 5 else pd.Series(dtype=float)
        sample_ok = len(returns) >= 10
        mean_ok = len(returns) > 0 and returns.mean() > 0.0
        median_ok = len(returns) > 0 and returns.median() > 0.0
        pf_ok = pd.notna(pf) and pf > 1.2
        after5_ok = len(after5) > 0 and after5.sum() > 0.0
        all_ok = sample_ok and mean_ok and median_ok and pf_ok and after5_ok
        rows.append(
            {
                "市场分支": branch,
                "完整交易": len(returns),
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "Profit_Factor": pf,
                "第一大盈利占净收益%": (
                    returns.head(1).sum() / net * 100.0
                    if math.isfinite(_safe_float(net)) and net > 0.0
                    else np.nan
                ),
                "前五大盈利占净收益%": (
                    returns.head(5).sum() / net * 100.0
                    if math.isfinite(_safe_float(net)) and net > 0.0
                    else np.nan
                ),
                "删除前五大后收益点": after5.sum() if len(after5) else np.nan,
                "删除前五大后平均收益%": after5.mean() if len(after5) else np.nan,
                "样本数检查": "通过" if sample_ok else "未通过",
                "平均收益检查": "通过" if mean_ok else "未通过",
                "中位收益检查": "通过" if median_ok else "未通过",
                "PF检查": "通过" if pf_ok else "未通过",
                "去前五大检查": "通过" if after5_ok else "未通过",
                "分支综合结论": "通过" if all_ok else "未通过",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _r27_shadow_outcomes(history: pd.DataFrame, flag_column: str):
    """按正式交易同口径提取影子信号的真实退出收益。"""
    if history.empty or flag_column not in history.columns:
        return history.iloc[0:0].copy()
    frame = history.loc[_bool_series(history, flag_column)].copy()
    if frame.empty:
        return frame
    complete = (
        _bool_series(frame, "Entry_Tradable")
        & _bool_series(frame, "R16_Lifecycle_Data_Available")
        & pd.to_numeric(frame.get("Fixed_Return_W3_Net_pct"), errors="coerce").notna()
    )
    frame = frame.loc[complete].copy()
    if frame.empty:
        return frame
    fixed = pd.to_numeric(frame["Fixed_Return_W3_Net_pct"], errors="coerce")
    trigger_day = pd.to_numeric(
        frame.get("R16_Stop_Minus10_Trigger_Day", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    stop_return = pd.to_numeric(
        frame.get("R16_Stop_Minus10_Return_Net_pct", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    use_stop = (
        _bool_series(frame, "R16_Stop_Minus10_Triggered")
        & trigger_day.le(PRIMARY_HOLD_WEEKS * MARKET_DAYS_PER_WEEK)
        & stop_return.notna()
    )
    frame["R27_Realized_Return_pct"] = fixed
    frame.loc[use_stop, "R27_Realized_Return_pct"] = stop_return.loc[use_stop]
    return frame


def r27_shadow_group_summary(history: pd.DataFrame):
    """R11正式、R11第二名和R7影子必须分开判卷。"""
    columns = [
        "审计组", "完整交易", "信号周", "胜率%", "平均收益%", "中位收益%",
        "Profit_Factor", "第一大盈利占净收益%", "删除第一大后收益点",
        "删除前五大后收益点",
    ]
    rows = []
    for label, flag in (
        ("R11正式Top1", "R11_Strong_Top1"),
        ("R11原有信号", "R27_R11_Baseline"),
        ("R11新增信号", "R27_R11_Added"),
        ("R6正式第一名", "R27_R6_First"),
        ("R6第二名影子", "R27_R6_Second"),
        ("R11原有信号周原第二名影子", "R11_Second_Shadow"),
        ("R7早期强势Top2影子", "R7_Shadow_Top2"),
    ):
        frame = _r27_shadow_outcomes(history, flag)
        returns = pd.to_numeric(
            frame.get("R27_Realized_Return_pct", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna().sort_values(ascending=False).reset_index(drop=True)
        net = returns.sum() if len(returns) else np.nan
        rows.append(
            {
                "审计组": label,
                "完整交易": len(returns),
                "信号周": frame["Signal_Date"].nunique() if len(frame) else 0,
                "胜率%": (returns > 0.0).mean() * 100.0 if len(returns) else np.nan,
                "平均收益%": returns.mean() if len(returns) else np.nan,
                "中位收益%": returns.median() if len(returns) else np.nan,
                "Profit_Factor": _profit_factor(returns),
                "第一大盈利占净收益%": (
                    returns.head(1).sum() / net * 100.0
                    if len(returns) and math.isfinite(_safe_float(net)) and net > 0.0
                    else np.nan
                ),
                "删除第一大后收益点": returns.iloc[1:].sum() if len(returns) > 1 else np.nan,
                "删除前五大后收益点": returns.iloc[5:].sum() if len(returns) > 5 else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def r27_rank2_acceptance(history: pd.DataFrame):
    """提前冻结第二名准入条件；任何单项失败都不得升级为正式信号。"""
    second = _r27_shadow_outcomes(history, "R11_Second_Shadow")
    top1 = _r27_shadow_outcomes(history, "R11_Strong_Top1")
    values = pd.to_numeric(
        second.get("R27_Realized_Return_pct", pd.Series(dtype=float)), errors="coerce"
    ).dropna().sort_values(ascending=False).reset_index(drop=True)
    top1_values = pd.to_numeric(
        top1.get("R27_Realized_Return_pct", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    pf = _profit_factor(values)
    checks = [
        ("第二名至少10笔完整样本", len(values) >= 10, f"当前{len(values)}笔"),
        ("第二名平均收益为正", len(values) > 0 and values.mean() > 0.0, f"当前{values.mean() if len(values) else np.nan:.2f}%"),
        ("第二名中位收益为正", len(values) > 0 and values.median() > 0.0, f"当前{values.median() if len(values) else np.nan:.2f}%"),
        ("第二名Profit Factor高于1.2", pd.notna(pf) and pf > 1.2, f"当前{pf:.2f}"),
        ("第二名删除第一大盈利后仍为正", len(values) > 1 and values.iloc[1:].sum() > 0.0, f"剩余收益点{values.iloc[1:].sum() if len(values) > 1 else np.nan:.2f}"),
        ("第二名中位收益不低于正式Top1", len(values) > 0 and len(top1_values) > 0 and values.median() >= top1_values.median(), f"第二名{values.median() if len(values) else np.nan:.2f}% / Top1 {top1_values.median() if len(top1_values) else np.nan:.2f}%"),
    ]
    result = pd.DataFrame(
        [{"R11第二名验收项目": name, "结果": "通过" if passed else "未通过", "当前值": value} for name, passed, value in checks]
    )
    result["最终结论"] = "允许继续研究正式化" if result["结果"].eq("通过").all() else "保持影子，不得买入"
    return result


def _r27_zero_run_lengths(signal_flags):
    runs, current = [], 0
    for flag in signal_flags:
        if bool(flag):
            if current:
                runs.append(current)
            current = 0
        else:
            current += 1
    if current:
        runs.append(current)
    return runs


def r27_signal_gap_audit(history: pd.DataFrame, ledger: pd.DataFrame):
    """同时报告新信号空窗与市场状态序列空窗，不用交易数冒充覆盖率。"""
    columns = [
        "范围", "扫描周", "新信号周", "无新信号周", "新信号覆盖率%",
        "最长连续无新信号周", "1周空窗段", "2周空窗段", "3周空窗段",
        "4周及以上空窗段", "说明",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    ledger_columns = ["Signal_Date"] + (["Market_Regime"] if "Market_Regime" in ledger.columns else [])
    weeks = ledger[ledger_columns].copy()
    weeks["Signal_Date"] = weeks["Signal_Date"].map(parse_yyyymmdd)
    weeks = weeks.dropna().drop_duplicates().sort_values("Signal_Date")
    selected_dates = set(
        history.loc[_bool_series(history, "R19_Selected"), "Signal_Date"]
        .map(parse_yyyymmdd).dropna().astype(str)
    ) if not history.empty else set()
    regime_map = {}
    if not history.empty and "Market_Regime" in history.columns:
        regime_source = history.copy()
        regime_source["Signal_Date"] = regime_source["Signal_Date"].map(parse_yyyymmdd)
        regime_source = regime_source.dropna(subset=["Signal_Date"])
        regime_map = regime_source.groupby("Signal_Date")["Market_Regime"].first().astype(str).to_dict()
    ledger_regime = weeks.get(
        "Market_Regime", pd.Series("未知", index=weeks.index)
    ).fillna("未知").astype(str)
    history_regime = weeks["Signal_Date"].map(regime_map).fillna("未知").astype(str)
    weeks["Market_Regime"] = ledger_regime.where(
        ~ledger_regime.isin({"", "未知", "nan"}), history_regime
    )
    weeks["Has_Signal"] = weeks["Signal_Date"].astype(str).isin(selected_dates)
    rows = []
    for label, regime in (("全部市场", None), ("强势", "强势"), ("中性", "中性"), ("弱势", "弱势")):
        group = weeks if regime is None else weeks[weeks["Market_Regime"].eq(regime)]
        flags = group["Has_Signal"].tolist()
        runs = _r27_zero_run_lengths(flags)
        rows.append(
            {
                "范围": label,
                "扫描周": len(group),
                "新信号周": int(sum(flags)),
                "无新信号周": int(len(group) - sum(flags)),
                "新信号覆盖率%": float(np.mean(flags) * 100.0) if flags else np.nan,
                "最长连续无新信号周": max(runs) if runs else 0,
                "1周空窗段": sum(run == 1 for run in runs),
                "2周空窗段": sum(run == 2 for run in runs),
                "3周空窗段": sum(run == 3 for run in runs),
                "4周及以上空窗段": sum(run >= 4 for run in runs),
                "说明": "全部市场按自然扫描周连续计算" if regime is None else "只在该市场状态出现序列中计算",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def r27_w3_holding_coverage_audit(history: pd.DataFrame, ledger: pd.DataFrame):
    columns = ["持有上限", "可观察周", "理论有持仓周", "理论空仓周", "理论持仓覆盖率%", "计算边界"]
    if history.empty or ledger.empty:
        return pd.DataFrame(columns=columns)
    weeks = ledger[["Signal_Date"]].copy()
    weeks["Signal_Date"] = weeks["Signal_Date"].map(parse_yyyymmdd)
    weeks = weeks.dropna().drop_duplicates().sort_values("Signal_Date")
    if len(weeks) <= 1:
        return pd.DataFrame(columns=columns)
    selected_dates = set(
        history.loc[_bool_series(history, "R19_Selected"), "Signal_Date"]
        .map(parse_yyyymmdd).dropna().astype(str)
    )
    flags = weeks["Signal_Date"].astype(str).isin(selected_dates).to_numpy()
    active = np.zeros(len(weeks), dtype=bool)
    for position in np.flatnonzero(flags):
        active[position + 1 : min(len(weeks), position + PRIMARY_HOLD_WEEKS + 1)] = True
    observable = active[1:]
    return pd.DataFrame([{
        "持有上限": f"W{PRIMARY_HOLD_WEEKS}",
        "可观察周": len(observable),
        "理论有持仓周": int(observable.sum()),
        "理论空仓周": int((~observable).sum()),
        "理论持仓覆盖率%": observable.mean() * 100.0,
        "计算边界": "忽略资金上限、提前止损和区间开始前已有持仓",
    }], columns=columns)


def r27_actual_coverage(history: pd.DataFrame, observation_dates=None):
    """同一成熟观察窗口；按行情路径交易日和真实退出统计。退出当日仍算持仓日。"""
    tracked = history.loc[_bool_series(history, "R19_Selected") | _bool_series(history, "R27_R6_Second")].copy()
    if tracked.empty:
        return pd.DataFrame(), pd.DataFrame()
    tracked["_coverage_flag"] = True
    complete = _r27_shadow_outcomes(tracked, "_coverage_flag")
    if complete.empty:
        return pd.DataFrame(), pd.DataFrame()
    complete["_entry"] = complete["Entry_Date"].map(parse_yyyymmdd)
    complete["_exit"] = complete["Fixed_Exit_W3_Date"].map(parse_yyyymmdd)
    stop = (_bool_series(complete, "R16_Stop_Minus10_Triggered")
            & pd.to_numeric(complete["R16_Stop_Minus10_Trigger_Day"], errors="coerce").le(15)
            & pd.to_numeric(complete["R16_Stop_Minus10_Return_Net_pct"], errors="coerce").notna())
    complete.loc[stop, "_exit"] = complete.loc[stop, "R16_Stop_Minus10_Exit_Date"].map(parse_yyyymmdd)
    complete = complete.dropna(subset=["_entry", "_exit"])
    if complete.empty:
        return pd.DataFrame(), pd.DataFrame()
    start = complete["_entry"].min()
    end = complete["Fixed_Exit_W3_Date"].map(parse_yyyymmdd).max()
    # 末端未完成交易不视为空仓：在其最早入场前截断所有对照组。
    pending = tracked.loc[~tracked.index.isin(complete.index), "Entry_Date"].map(parse_yyyymmdd).dropna()
    cutoff = pending.min() if len(pending) else None
    dates = set()
    for raw in tracked.get("R19_Daily_Path_JSON", pd.Series(dtype=str)).dropna():
        try:
            dates.update(parse_yyyymmdd(v[0]) for v in json.loads(raw))
        except (ValueError, TypeError, IndexError):
            continue
    dates = sorted(d for d in dates if d and start <= d <= end and (cutoff is None or d < cutoff))
    if observation_dates is not None:
        dates = list(observation_dates)
    if not dates:
        return pd.DataFrame(), pd.DataFrame()
    daily = pd.DataFrame({"交易日": dates})
    formal = _bool_series(complete, "R19_Selected")
    weak = complete["Market_Regime"].eq("弱势")
    groups = {
        "正式全分支": formal,
        "全分支加R6第二名对照": formal | _bool_series(complete, "R27_R6_Second"),
        "R6仅第一名": formal & weak,
        "R6前两名对照": weak,
    }
    rows = []
    for label, mask in groups.items():
        count = np.zeros(len(dates), dtype=int)
        for _, row in complete.loc[mask].iterrows():
            count += np.array([row["_entry"] <= d <= row["_exit"] for d in dates], dtype=int)
        daily[label] = count
        runs = _r27_zero_run_lengths(count > 0)
        rows.append({"方案": label, "起始交易日": dates[0], "截止交易日": dates[-1],
                     "观察交易日": len(dates), "有持仓日": int((count > 0).sum()),
                     "空仓日": int((count == 0).sum()), "最长连续空仓交易日": max(runs, default=0),
                     "实际退出持仓覆盖率%": float((count > 0).mean() * 100),
                     "边界": "无限资金；完整交易公共窗口；含止损；退出日算持仓；不含区间前持仓"})
    return pd.DataFrame(rows), daily


def r27_exit_reports(history, ledger):
    """只比较同一批正式入选且W3成熟、三方案均完成的交易，不选择胜出规则。"""
    summaries, pairs, observations, coverage, checks, reconcile = [], [], [], [], [], []
    selected = history.loc[_bool_series(history, "R19_Selected")].copy()
    for col in ["Signal_Date", "Entry_Date"] + [f"R27_{s}_{suffix}" for s in R27_EXIT_SCHEMES for suffix in ("Exit_Date", "Trigger_Date")]:
        if col in selected:
            selected[col] = selected[col].map(parse_yyyymmdd)
    matched, parsed = [], {}
    for idx, row in selected.iterrows():
        identity = {k: row.get(k) for k in ("Signal_Date", "ts_code", "name", "Market_Regime", "Entry_Date")}
        try:
            path = json.loads(str(row.get("R27_Exit_Path_JSON", "")))
            if not isinstance(path, list):
                raise ValueError("路径格式错误")
        except (ValueError, TypeError):
            path = []
        states = [row.get(f"R27_{s}_Status") for s in R27_EXIT_SCHEMES]
        good = (bool(_bool_series(pd.DataFrame([row]), "Entry_Tradable").iloc[0])
                and len(path) >= 15 and all(s == "已退出" for s in states)
                and all(math.isfinite(_safe_float(row.get(f"R27_{s}_Return_pct"))) for s in R27_EXIT_SCHEMES))
        checks.append({**identity, "纳入配对": good,
                       "原因": "同一交易三方案均完成且W3已成熟" if good else "未成交、缺少完整路径或退出尚未完成",
                       **{R27_EXIT_SCHEMES[s]: row.get(f"R27_{s}_Status", "无退出路径") for s in R27_EXIT_SCHEMES}})
        if not good:
            continue
        matched.append(idx)
        parsed[idx] = path
        base = float(row["R27_B10_Return_pct"])
        old = _safe_float(row.get("Fixed_Return_W3_Net_pct"))
        trigger_day = _safe_float(row.get("R16_Stop_Minus10_Trigger_Day"))
        old_stop = _safe_float(row.get("R16_Stop_Minus10_Return_Net_pct"))
        if trigger_day <= 15 and math.isfinite(old_stop):
            old = old_stop
        reconcile.append({**identity, "原报表净收益%": old, "可成交基准净收益%": base,
                          "口径修正差值百分点": base - old,
                          "可成交基准退出日": row["R27_B10_Exit_Date"],
                          "可成交基准退出原因": row["R27_B10_Reason"]})
        for scheme in R27_EXIT_SCHEMES:
            ret = float(row[f"R27_{scheme}_Return_pct"])
            pairs.append({**identity, "方案代码": scheme, "方案": R27_EXIT_SCHEMES[scheme],
                          "退出日": row[f"R27_{scheme}_Exit_Date"],
                          "持有交易日": row[f"R27_{scheme}_Exit_Day"],
                          "触发日": row[f"R27_{scheme}_Trigger_Date"],
                          "退出原因": row[f"R27_{scheme}_Reason"],
                          "受阻交易日": row[f"R27_{scheme}_Blocked_Days"],
                          "净收益%": ret, "基准净收益%": base, "改善百分点": ret - base})
        # 诊断首周之后的增量，不能把首周涨幅与包含首周的W3总收益相关性当作预测。
        day5 = path[4]
        cl = _safe_float(day5.get("close"))
        buy = _safe_float(row.get("R19_Path_Entry_Open_QFQ"))
        cost = _safe_float(row.get("R19_Roundtrip_Cost_pct"), .2)
        if cl > 0 and buy > 0 and _safe_float(day5.get("vol"), 1) > 0:
            first = (cl / buy - 1) * 100 - cost
            survived = _safe_float(row.get("R27_B10_Exit_Day")) > 5
            exit_price = _safe_float(row.get("R27_B10_Exit_Price"))
            observations.append({**identity, "首周净收益%": first,
                "首周分组": "首周未盈利" if first <= 0 else "首周盈利",
                "第5日后仍持仓": survived,
                "后续增量毛收益%": (exit_price / cl - 1) * 100 if survived else np.nan,
                "基准最终净收益%": base,
                "说明": "后续增量仅统计第5日收盘后仍持仓样本；未扣重复费用"})
    pair_df = pd.DataFrame(pairs)
    if pairs:
        period_dates = pair_df.Signal_Date.map(parse_yyyymmdd)
        failure = period_dates.between("20250307", "20250425")
        for scheme, group in pair_df.groupby("方案代码", sort=False):
            subsets = {"全区间": group,
                       "已见失败段20250307—20250425": group.loc[failure.reindex(group.index)],
                       "失败段以外": group.loc[~failure.reindex(group.index)]}
            subsets.update({"分支：" + str(k): g for k, g in group.groupby("Market_Regime")})
            subsets.update({"信号年：" + str(k): g for k, g in group.groupby(group.Signal_Date.astype(str).str[:4])})
            for scope, g in subsets.items():
                r, b, delta = g["净收益%"], g["基准净收益%"], g["改善百分点"]
                loss = -r[r < 0].sum()
                summaries.append({"方案": R27_EXIT_SCHEMES[scheme], "范围": scope,
                    "配对交易数": len(g), "信号周": g.Signal_Date.nunique(),
                    "平均净收益%": r.mean(), "中位净收益%": r.median(),
                    "胜率%": r.gt(0).mean() * 100 if len(g) else np.nan,
                    "PF": r[r > 0].sum() / loss if loss > 0 else np.nan,
                    "平均改善百分点": delta.mean(),
                    "改善交易数": int(delta.gt(1e-8).sum()), "恶化交易数": int(delta.lt(-1e-8).sum()),
                    "减亏收益点": delta[(b < 0) & (delta > 0)].sum(),
                    "原盈利交易损失收益点": -delta[(b > 0) & (delta < 0)].sum(),
                    "原盈利变亏损笔数": int(((b > 0) & (r < 0)).sum()),
                    "亏损超过10%笔数": int(r.lt(-10).sum()),
                    "最差单笔%": r.min(), "平均持有交易日": g["持有交易日"].mean(),
                    "剔除最高5笔后均值%": r.sort_values().iloc[:-5].mean() if len(r) > 5 else np.nan})
        # 同一成熟配对样本、同一日历窗口；提前退出后的空窗如实计入。
        calendar = sorted({bar["date"] for path in parsed.values() for bar in path})
        start = min(str(selected.loc[i, "Entry_Date"]) for i in matched)
        end = max(str(parsed[i][14]["date"]) for i in matched)
        calendar = [d for d in calendar if start <= d <= end]
        for scheme in R27_EXIT_SCHEMES:
            run = longest = empty = 0
            intervals = [(str(selected.loc[i, "Entry_Date"]), str(selected.loc[i, f"R27_{scheme}_Exit_Date"])) for i in matched]
            for d in calendar:
                count = sum(a <= d <= b for a, b in intervals)
                run = run + 1 if count == 0 else 0
                empty += int(count == 0)
                longest = max(longest, run)
            coverage.append({"方案": R27_EXIT_SCHEMES[scheme], "观察起日": start, "观察止日": end,
                "配对交易数": len(matched), "观察交易日": len(calendar), "空仓交易日": empty,
                "最长连续空仓交易日": longest,
                "说明": "仅成熟配对样本；退出当天算持有；不是账户净值或全体信号覆盖"})
    return (pd.DataFrame(summaries), pair_df, pd.DataFrame(observations),
            pd.DataFrame(coverage), pd.DataFrame(checks), pd.DataFrame(reconcile))


def r20_block_bootstrap(universe: pd.DataFrame, scan_ledger: pd.DataFrame):
    """按连续4个扫描周成块重采样，保留同周股票及W3重叠的相关性。"""
    columns = ["统计量", "2.5%下界", "中位数", "97.5%上界", "重复次数", "区组周数"]
    if universe.empty or scan_ledger.empty:
        return pd.DataFrame(columns=columns)
    scan_dates = sorted(
        {value for value in scan_ledger["Signal_Date"].map(parse_yyyymmdd) if value}
    )
    if len(scan_dates) < R27_BOOTSTRAP_BLOCK_WEEKS:
        return pd.DataFrame(columns=columns)
    signal_text = universe["Signal_Date"].map(parse_yyyymmdd)
    returns_by_week = {
        day: pd.to_numeric(
            universe.loc[signal_text.eq(day), "R19_Realized_Return_pct"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        for day in scan_dates
    }
    block = R27_BOOTSTRAP_BLOCK_WEEKS
    starts = np.arange(0, len(scan_dates) - block + 1)
    rng = np.random.default_rng(20200907)
    mean_values, win_values, pf_values = [], [], []
    for _ in range(R27_BOOTSTRAP_REPETITIONS):
        sampled = []
        while len(sampled) < len(scan_dates):
            start = int(rng.choice(starts))
            sampled.extend(scan_dates[start : start + block])
        chunks = [returns_by_week[day] for day in sampled[: len(scan_dates)] if len(returns_by_week[day])]
        if not chunks:
            continue
        values = np.concatenate(chunks)
        gains = values[values > 0.0].sum()
        losses = -values[values < 0.0].sum()
        mean_values.append(float(values.mean()))
        win_values.append(float((values > 0.0).mean() * 100.0))
        pf_values.append(float(gains / losses) if losses > 0.0 else np.nan)

    def row(label, values):
        numeric = np.asarray(values, dtype=float)
        numeric = numeric[np.isfinite(numeric)]
        return {
            "统计量": label,
            "2.5%下界": np.quantile(numeric, 0.025) if len(numeric) else np.nan,
            "中位数": np.quantile(numeric, 0.50) if len(numeric) else np.nan,
            "97.5%上界": np.quantile(numeric, 0.975) if len(numeric) else np.nan,
            "重复次数": len(numeric),
            "区组周数": block,
        }
    return pd.DataFrame(
        [
            row("平均单笔收益%", mean_values),
            row("交易胜率%", win_values),
            row("Profit_Factor", pf_values),
        ],
        columns=columns,
    )


def r20_internal_robustness_scorecard(
    universe: pd.DataFrame,
    bootstrap: pd.DataFrame,
):
    returns = pd.to_numeric(
        universe.get("R19_Realized_Return_pct", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    ordered = returns.sort_values(ascending=False).reset_index(drop=True)
    gains = returns[returns > 0.0].sum()
    losses = -returns[returns < 0.0].sum()
    pf = gains / losses if losses > 0.0 else np.nan
    branch_means = (
        universe.assign(_ret=pd.to_numeric(universe["R19_Realized_Return_pct"], errors="coerce"))
        .groupby("R19_市场分支")["_ret"]
        .agg(["size", "mean"])
        if len(universe)
        else pd.DataFrame()
    )
    branch_audit = r27_branch_robustness_audit(universe)
    all_branch_medians_ok = (
        len(branch_audit) == 3
        and branch_audit["中位收益检查"].eq("通过").all()
    )
    all_branch_pf_ok = (
        len(branch_audit) == 3
        and branch_audit["PF检查"].eq("通过").all()
    )
    all_branch_after5_ok = (
        len(branch_audit) == 3
        and branch_audit["去前五大检查"].eq("通过").all()
    )
    boot_lower = np.nan
    if not bootstrap.empty:
        match = bootstrap[bootstrap["统计量"].eq("平均单笔收益%")]
        if not match.empty:
            boot_lower = _safe_float(match.iloc[0]["2.5%下界"])
    net = returns.sum()
    top5_share = ordered.head(5).sum() / net * 100.0 if net > 0 else np.nan
    checks = [
        ("完整交易不少于60笔", len(returns) >= 60, f"当前{len(returns)}笔"),
        ("信号覆盖不少于30周", universe["Signal_Date"].nunique() >= 30 if len(universe) else False, f"当前{universe['Signal_Date'].nunique() if len(universe) else 0}周"),
        ("平均收益为正", len(returns) > 0 and returns.mean() > 0.0, f"当前{returns.mean() if len(returns) else np.nan:.2f}%"),
        ("中位收益为正", len(returns) > 0 and returns.median() > 0.0, f"当前{returns.median() if len(returns) else np.nan:.2f}%"),
        ("Profit Factor高于1.5", math.isfinite(_safe_float(pf)) and pf > 1.5, f"当前{pf:.2f}"),
        ("删除第一大盈利后仍为正", len(ordered) > 1 and ordered.iloc[1:].sum() > 0.0, f"剩余收益点{ordered.iloc[1:].sum() if len(ordered) > 1 else np.nan:.2f}"),
        ("删除前五大盈利后仍为正", len(ordered) > 5 and ordered.iloc[5:].sum() > 0.0, f"剩余收益点{ordered.iloc[5:].sum() if len(ordered) > 5 else np.nan:.2f}"),
        ("前五大盈利占比不超过50%", math.isfinite(_safe_float(top5_share)) and top5_share <= 50.0, f"当前{top5_share:.2f}%"),
        ("三个分支均至少10笔且平均为正", not branch_means.empty and len(branch_means) == 3 and bool(((branch_means['size'] >= 10) & (branch_means['mean'] > 0.0)).all()), "; ".join(f"{idx}:{int(row['size'])}笔/{row['mean']:.2f}%" for idx, row in branch_means.iterrows())),
        ("三个分支中位收益均为正", all_branch_medians_ok, "; ".join(f"{row['市场分支']}:{row['中位收益%']:.2f}%" for _, row in branch_audit.iterrows())),
        ("三个分支Profit Factor均高于1.2", all_branch_pf_ok, "; ".join(f"{row['市场分支']}:{row['Profit_Factor']:.2f}" for _, row in branch_audit.iterrows())),
        ("三个分支删除前五大盈利后均为正", all_branch_after5_ok, "; ".join(f"{row['市场分支']}:{row['删除前五大后收益点']:.2f}" for _, row in branch_audit.iterrows())),
        ("4周区组自助95%下界为正", math.isfinite(boot_lower) and boot_lower > 0.0, f"当前下界{boot_lower:.2f}%"),
    ]
    return pd.DataFrame(
        [
            {
                "内部稳健性项目": name,
                "结果": "通过" if passed else "未通过",
                "当前值": value,
                "解释边界": "只检验当前样本内部稳健性，不能替代未见样本或前向验证",
            }
            for name, passed, value in checks
        ]
    )


def r27_integrity_gates(
    history: pd.DataFrame,
    scan_ledger: pd.DataFrame,
    all_signal_ledger: pd.DataFrame,
    notional: float,
):
    universe = r19_trade_universe(history)
    status = scan_ledger.get(
        "Scan_Status", pd.Series("COMPLETED", index=scan_ledger.index)
    ).astype(str)
    unique_trades = not all_signal_ledger.duplicated(["Signal_Date", "ts_code"]).any()
    returns_ok = (
        len(all_signal_ledger) > 0
        and pd.to_numeric(all_signal_ledger["交易净收益%"], errors="coerce").notna().all()
    )
    notional_values = pd.to_numeric(
        all_signal_ledger.get("名义本金", pd.Series(dtype=float)), errors="coerce"
    )
    regime = history.get("Market_Regime", pd.Series("", index=history.index)).astype(str)
    strong_rows = regime.eq("强势")
    strong_selected = strong_rows & _bool_series(history, "R19_Selected")
    r11_selected = strong_rows & _bool_series(history, "R11_Strong_Top1")
    strong_week_counts = (
        history.loc[strong_selected].groupby("Signal_Date").size()
        if strong_selected.any() else pd.Series(dtype=int)
    )
    r11_exact = bool(strong_selected.equals(r11_selected))
    r11_rule_ok = bool(
        (
            pd.to_numeric(history.loc[strong_selected, "R27_Band_Rank"], errors="coerce").eq(1)
            & _bool_series(history.loc[strong_selected], "R11_ATR_Band_Pass")
        ).all()
    ) if strong_selected.any() else True
    gates = [
        ("冻结规则", "最长持有严格为W3", PRIMARY_HOLD_WEEKS == 3, f"当前W{PRIMARY_HOLD_WEEKS}"),
        ("冻结规则", "灾难止损严格为T+1日内-10%", R16_PRIMARY_STOP_PCT == -10.0, f"当前{R16_PRIMARY_STOP_PCT:.1f}%"),
        ("冻结规则", "止损计0.3%不利滑点", np.isclose(R16_STOP_SLIPPAGE_PCT, 0.30), f"当前{R16_STOP_SLIPPAGE_PCT:.2f}%"),
        ("数据完整", "全部扫描周无缺口且已完成", len(scan_ledger) > 0 and status.eq("COMPLETED").all(), f"完成{int(status.eq('COMPLETED').sum())}/{len(scan_ledger)}周"),
        ("数据完整", "扫描账本与候选明细一致", result_state_consistency_audit(history, scan_ledger).empty, "已核对"),
        ("全量执行", "全部完整入选交易均纳入", len(all_signal_ledger) == len(universe), f"纳入{len(all_signal_ledger)}/{len(universe)}笔"),
        ("全量执行", "不存在仓位冲突或跳过交易", "执行状态" not in all_signal_ledger.columns and "仓位编号" not in all_signal_ledger.columns, "无限资金、无仓位路径"),
        ("等权口径", "每笔名义本金完全相同", len(notional_values) > 0 and np.allclose(notional_values, float(notional)), f"每笔{float(notional):.2f}元"),
        ("交易唯一", "同一信号周同一股票不重复", unique_trades, "已核对"),
        ("收益完整", "全部纳入交易均有真实退出收益", returns_ok, f"完整{int(pd.to_numeric(all_signal_ledger.get('交易净收益%', pd.Series(dtype=float)), errors='coerce').notna().sum())}/{len(all_signal_ledger)}笔"),
        ("R11隔离", "强市正式信号与R11 Top1完全一致", r11_exact, f"正式{int(strong_selected.sum())}笔 / R11标记{int(r11_selected.sum())}笔"),
        ("R11冻结", "强市每周至多一只且为ATR区间内第一名", (strong_week_counts.le(1).all() if len(strong_week_counts) else True) and r11_rule_ok, f"最多{int(strong_week_counts.max()) if len(strong_week_counts) else 0}只/周"),
        ("R6隔离", "弱市正式仅第一名，第二名不进入正式交易", bool((_bool_series(history, "R19_Selected") & regime.eq("弱势")).equals(_bool_series(history, "R27_R6_First"))), "保留原至少两只合格候选门槛"),
        ("影子隔离", "R7和R11第二名不能单独进入正式收益", r11_exact, "影子只计算同口径未来路径"),
    ]
    return pd.DataFrame(
        [
            {
                "验收阶段": phase,
                "R27完整性项目": name,
                "结果": "通过" if passed else "未通过",
                "当前值": value,
            }
            for phase, name, passed, value in gates
        ]
    )

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
    path_baselines = pd.to_numeric(
        bought.get(
            "R19_Path_Entry_Open_QFQ",
            pd.Series(np.nan, index=bought.index),
        ),
        errors="coerce",
    )
    baseline_complete = (
        not bought.empty
        and path_baselines.notna().all()
        and path_baselines.gt(0.0).all()
    )
    scale_audit = r19_w3_path_scale_audit(bought)
    scale_consistent = path_complete and (
        scale_audit.empty
        or scale_audit["路径尺度一致"].fillna(False).all()
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
        ("复权口径", "每日路径均保存同尺度买入基准", baseline_complete, f"完整{int(path_baselines.gt(0.0).sum())}/{len(bought)}笔"),
        ("复权口径", "W3路径复算与冻结交易收益一致", scale_consistent, f"通过{int(scale_audit['路径尺度一致'].fillna(False).sum()) if not scale_audit.empty else 0}/{len(scale_audit)}笔"),
        ("资金约束", "任一日同时持仓不超过3只", not daily.empty and int(daily['持仓数'].max()) <= 3, f"当前最多{int(daily['持仓数'].max()) if not daily.empty else 0}只"),
        ("资金核对", "逐仓复投期末资金与每日净值一致", math.isfinite(end_expected) and math.isfinite(end_daily) and abs(end_expected - end_daily) <= 0.02, f"差额{(end_daily - end_expected) if math.isfinite(end_expected) and math.isfinite(end_daily) else np.nan:.2f}元"),
    ]
    return pd.DataFrame(
        [
            {
                "验收阶段": phase,
                "R19.1完整性项目": name,
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

def import_prior_results_zip(
    zip_bytes: bytes,
    config_id: str,
    roundtrip_cost_pct: float,
):
    """事务导入同策略R27结果；旧版候选不得冒充恢复后的R11结果。"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        infos = {
            info.filename: info
            for info in archive.infolist()
            if not info.is_dir()
        }
        candidate_names = [
            name
            for name in infos
            if name.startswith("01_all_r27_1_")
            and name.endswith("_candidates.csv")
        ]
        if len(candidate_names) != 1:
            raise ValueError("只可恢复R27.1结果。R27使用不同的指标初始化口径，请重新扫描；旧包可用页面的历史结果对比功能查看。")
        info = infos[candidate_names[0]]
        if info.file_size > 200 * 1024 * 1024:
            raise ValueError("候选明细超过200MB，拒绝导入。")
        candidates = pd.read_csv(
            io.BytesIO(archive.read(info)),
            encoding="utf-8-sig",
            low_memory=False,
            float_precision="round_trip",
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
            "R11_Strong_Rank",
            "R11_ATR_Band_Pass",
            "R11_Strong_Top1",
            "R11_Second_Shadow",
            "R7_Shadow_Top2",
            "R27_Shadow_Tracked",
            "R27_Band_Rank", "R27_R11_Baseline", "R27_R11_Added",
            "R27_R6_First", "R27_R6_Second",
            "R19_Selected",
            "ATR_Contraction",
        }
        missing = sorted(required.difference(candidates.columns))
        missing += sorted({"R271_Window_Start", "R271_Input_Hash", "R271_Full_Pool_Hash", "R271_Whitelist_Hash"}.difference(candidates.columns))
        missing += sorted({"R27_Exit_Path_JSON", "R27_Exit_Path_Version"}.difference(candidates.columns))
        missing += sorted({f"R27_{k}_{suffix}" for k in R27_EXIT_SCHEMES for suffix in
                           ["Status", "Exit_Date", "Exit_Day", "Return_pct", "Reason", "Trigger_Date", "Blocked_Days", "Exit_Price"]}.difference(candidates.columns))
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
        if "R19_Path_Entry_Open_QFQ" not in candidates.columns:
            candidates["R19_Path_Entry_Open_QFQ"] = np.nan
        if "R19_Roundtrip_Cost_pct" not in candidates.columns:
            candidates["R19_Roundtrip_Cost_pct"] = float(
                roundtrip_cost_pct
            )
        candidates, recovered_path_rows = recover_r19_1_path_baselines(
            candidates
        )
        # 用已保存OHLC重新计算退出字段，不能信任损坏或被修改的汇总收益。
        for scheme in R27_EXIT_SCHEMES:
            for suffix in ("Status", "Exit_Date", "Trigger_Date", "Reason"):
                column = f"R27_{scheme}_{suffix}"
                candidates[column] = candidates[column].astype(object)
        for idx in candidates.index[_bool_series(candidates, "R19_Selected") & _bool_series(candidates, "Entry_Tradable")]:
            row = candidates.loc[idx]
            try:
                if _safe_float(row.get("R27_Exit_Path_Version")) != 1:
                    raise ValueError("路径版本不符")
                path = json.loads(str(row.get("R27_Exit_Path_JSON", "")))
                dates = [str(bar["date"]) for bar in path]
                entry = parse_yyyymmdd(row.get("Entry_Date"))
                buy = _safe_float(row.get("R19_Path_Entry_Open_QFQ"))
                if not dates or dates != sorted(set(dates)) or dates[0] != entry or not buy > 0:
                    raise ValueError("路径日期或买入锚点异常")
                if not math.isclose(_safe_float(path[0].get("open")), buy, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError("OHLC与买入价格复权尺度不一致")
                cost = _safe_float(row.get("R19_Roundtrip_Cost_pct"), roundtrip_cost_pct)
                for scheme in R27_EXIT_SCHEMES:
                    for key, value in r27_exit_simulation(path, buy, cost, str(row.ts_code), scheme).items():
                        candidates.at[idx, f"R27_{scheme}_{key}"] = value
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                raise ValueError(f"{row.get('Signal_Date')} {row.get('ts_code')} 退出路径校验失败：{exc}") from exc
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
        for _, imported_week in ledger.iterrows():
            raw_audit = imported_week.get("R28_Audit_JSON", "")
            if pd.notna(raw_audit) and str(raw_audit).strip():
                meta, _ = r28_unpack(raw_audit)
                if meta["signal_date"] != imported_week.Signal_Date:
                    raise ValueError("R28诊断日期不一致，未写入任何结果。")
                if meta.get("cost") is not None and not math.isclose(float(meta["cost"]), roundtrip_cost_pct):
                    raise ValueError("R28结果的交易成本与当前配置不同，请先恢复原配置。")

        selected = candidates[_bool_series(candidates, "R19_Selected")].copy()
        # R27只使用每笔实际退出收益，不构造三仓每日净值。
        missing_path_dates: set[str] = set()
        pending_mask = pd.Series(False, index=ledger.index)
        ledger.loc[
            ledger.get("Scan_Status", pd.Series("", index=ledger.index))
            .astype(str)
            .eq("PENDING_R19_NAV"),
            "Scan_Status",
        ] = "COMPLETED"

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
        if not existing_candidates.empty and "R271_Whitelist_Hash" in existing_candidates:
            hashes = set(existing_candidates["R271_Whitelist_Hash"].dropna()) | set(candidates["R271_Whitelist_Hash"].dropna())
            if len(hashes) > 1:
                raise ValueError("两份结果使用不同研究池，不能合并恢复；请使用只读对比。")
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
        "recovered_path_rows": recovered_path_rows,
        "pending_nav_weeks": len(missing_path_dates),
    }


def mark_legacy_r19_paths_pending():
    """部署覆盖升级时，把没有同尺度基准的旧R19路径自动转为只补路径。"""
    history = read_csv_safe(CHECKPOINT_FILE)
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    if history.empty or ledger.empty:
        return 0
    if "Signal_Date" not in history.columns or "Signal_Date" not in ledger.columns:
        return 0
    history["Signal_Date"] = history["Signal_Date"].map(parse_yyyymmdd)
    ledger["Signal_Date"] = ledger["Signal_Date"].map(parse_yyyymmdd)
    history, recovered_path_rows = recover_r19_1_path_baselines(history)
    if "Config_ID" not in history.columns:
        history["Config_ID"] = ""
    if "Config_ID" not in ledger.columns:
        ledger["Config_ID"] = ""
    keys: set[tuple[str, str]] = set()
    for config, group in history.groupby("Config_ID", dropna=False):
        missing_dates = r19_missing_bought_path_dates(group)
        config_text = "" if pd.isna(config) else str(config)
        keys.update((config_text, date_text) for date_text in missing_dates)
    if not keys:
        if recovered_path_rows:
            with _result_files_transaction([CHECKPOINT_FILE]):
                atomic_write_csv(history.reset_index(drop=True), CHECKPOINT_FILE)
        return 0
    mask = pd.Series(
        [
            (str(config), str(signal_date)) in keys
            for config, signal_date in zip(
                ledger["Config_ID"].fillna("").astype(str),
                ledger["Signal_Date"].astype(str),
            )
        ],
        index=ledger.index,
    )
    already_pending = ledger.get(
        "Scan_Status", pd.Series("", index=ledger.index)
    ).astype(str).eq("PENDING_R19_NAV")
    change = mask & ~already_pending
    if change.any() or recovered_path_rows:
        ledger.loc[change, "Scan_Status"] = "PENDING_R19_NAV"
        ledger.loc[
            change, "Selection_Block_Reason"
        ] = "R19.1检测到旧复权路径；等待只补同尺度每日净值"
        with _result_files_transaction(
            [CHECKPOINT_FILE, SCAN_LEDGER_FILE]
        ):
            if recovered_path_rows:
                atomic_write_csv(
                    history.reset_index(drop=True), CHECKPOINT_FILE
                )
            atomic_write_csv(ledger.reset_index(drop=True), SCAN_LEDGER_FILE)
    return int(mask.sum())

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
    """R19.1只导出主方案与风险审计，不再携带失败研究分支。"""
    files = {
        "01_all_r19_1_same_scale_three_slot_w3_risk_candidates.csv": history,
        "02_scan_ledger.csv": ledger,
        "03_market_data_gap_audit.csv": data_gaps,
        "04_three_regime_trade_summary.csv": branch_summary,
        "05_three_slot_portfolio_summary.csv": portfolio_summary,
        "06_three_slot_trade_ledger.csv": portfolio_ledger,
        "07_daily_equity_curve.csv": daily_equity,
        "08_monthly_returns.csv": monthly_returns,
        "09_portfolio_risk_summary.csv": risk_summary,
        "10_r19_1_integrity_gates.csv": integrity_gates,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, frame in files.items():
            archive.writestr(
                name,
                frame.to_csv(index=False, encoding="utf-8-sig"),
            )
    return output.getvalue()


def r271_input_manifest(history, ledger):
    rows = []
    groups = {parse_yyyymmdd(d): g for d, g in history.groupby("Signal_Date")} if "Signal_Date" in history else {}
    for d in ledger.get("Signal_Date", pd.Series(dtype=str)).map(parse_yyyymmdd):
        g = groups.get(d, pd.DataFrame())
        rows.append({"Signal_Date": d, "Window_Start": signal_window_start(d),
                     "Candidate_Count": len(g), "Selected_Count": int(_bool_series(g, "R19_Selected").sum()),
                     **{key: g[key].iloc[0] if key in g and len(g) else "无候选：需原始输入进一步核对" for key in
                        ("R271_Full_Pool_Hash", "R271_Whitelist_Hash", "R271_Pool_Size")}})
    return pd.DataFrame(rows)


def r271_compare_results(history, ledger, zip_bytes):
    """只读对照，不导入、不覆盖当前断点；只比较共同信号日的买入前结果。"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [i for i in archive.infolist() if i.filename.startswith("01_all_") and i.filename.endswith("_candidates.csv")]
        if len(names) != 1 or names[0].file_size > 200 * 1024 * 1024:
            raise ValueError("需要唯一且不超过200MB的候选表。")
        other = pd.read_csv(archive.open(names[0]), low_memory=False, float_precision="round_trip")
        info = archive.getinfo("02_scan_ledger.csv")
        if info.file_size > 10 * 1024 * 1024:
            raise ValueError("扫描账本超过大小限制。")
        old_ledger = pd.read_csv(archive.open(info))
    if not {"Signal_Date", "ts_code", "R19_Selected"}.issubset(other):
        raise ValueError("旧包缺少信号日期、股票代码或正式入选字段。")
    left, right = history.copy(), other.copy()
    for frame in (left, right):
        frame["Signal_Date"] = frame["Signal_Date"].map(parse_yyyymmdd)
        if frame.duplicated(["Signal_Date", "ts_code"]).any():
            raise ValueError("候选有重复键，不能完成对照。")
    common_dates = sorted(set(ledger.Signal_Date.map(parse_yyyymmdd)) & set(old_ledger.Signal_Date.map(parse_yyyymmdd)))
    lg = dict(tuple(left.groupby("Signal_Date")))
    rg = dict(tuple(right.groupby("Signal_Date")))
    rows = []
    for d in common_dates:
        a, b = lg.get(d, left.iloc[:0]), rg.get(d, right.iloc[:0])
        ac, bc = set(a.ts_code), set(b.ts_code)
        sa = set(a.loc[_bool_series(a, "R19_Selected"), "ts_code"])
        sb = set(b.loc[_bool_series(b, "R19_Selected"), "ts_code"])
        joined = a.set_index("ts_code").join(b.set_index("ts_code"), lsuffix="_now", rsuffix="_old", how="inner")
        diffs = []
        for col in ("MACD_DIF", "MACD_DEA", "MACD_Hist", "Previous_MACD_Hist", "Rank", "Entry_Score_100"):
            if col + "_now" in joined and col + "_old" in joined:
                x = pd.to_numeric(joined[col + "_now"], errors="coerce")
                y = pd.to_numeric(joined[col + "_old"], errors="coerce")
                diffs.append(bool((x.eq(y) | (x.isna() & y.isna())).all()))
        fingerprint_known = len(a) > 0 and len(b) > 0 and all(k in a and k in b for k in ("R271_Full_Pool_Hash", "R271_Whitelist_Hash"))
        inputs_equal = fingerprint_known and all(set(a[k].dropna()) == set(b[k].dropna()) and a[k].notna().all() and b[k].notna().all()
                                                for k in ("R271_Full_Pool_Hash", "R271_Whitelist_Hash"))
        fields_equal = bool(diffs) and all(diffs)
        rows.append({"Signal_Date": d, "当前候选": len(a), "对照候选": len(b),
                     "候选名单一致": ac == bc, "正式入选一致": sa == sb,
                     "当前新增入选": ",".join(sorted(sa - sb)), "当前减少入选": ",".join(sorted(sb - sa)),
                     "共同候选指标及排名一致": fields_equal,
                     "输入指纹一致": inputs_equal if fingerprint_known else "旧包无指纹或无候选",
                     "核对结果": "通过" if inputs_equal and ac == bc and sa == sb and fields_equal else "需核对"})
    return pd.DataFrame(rows)


def build_r27_export_zip(
    history: pd.DataFrame,
    ledger: pd.DataFrame,
    data_gaps: pd.DataFrame,
    all_signal_summary: pd.DataFrame,
    all_signal_ledger: pd.DataFrame,
    branch_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    halfyear_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    weekly_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    bootstrap: pd.DataFrame,
    robustness: pd.DataFrame,
    branch_robustness: pd.DataFrame,
    strong_shadow_summary: pd.DataFrame,
    rank2_acceptance: pd.DataFrame,
    signal_gap_audit: pd.DataFrame,
    holding_coverage: pd.DataFrame,
    integrity: pd.DataFrame,
    audit_metadata: pd.DataFrame,
    r28_diagnostics=None,
):
    """R27导出正式全信号、强市影子和空窗审计，不含三仓或复投。"""
    files = {
        "01_all_r27_1_reproducible_candidates.csv": history,
        "02_scan_ledger.csv": ledger,
        "03_market_data_gap_audit.csv": data_gaps,
        "04_all_signal_equal_notional_summary.csv": all_signal_summary,
        "05_all_signal_equal_notional_trade_ledger.csv": all_signal_ledger,
        "06_branch_stability.csv": branch_summary,
        "07_calendar_year_stability.csv": year_summary,
        "08_halfyear_stability.csv": halfyear_summary,
        "09_rank_stability.csv": rank_summary,
        "10_signal_week_equal_weight.csv": weekly_summary,
        "11_rolling_26_scan_week_stability.csv": rolling_summary,
        "12_profit_concentration_audit.csv": concentration,
        "13_four_week_block_bootstrap.csv": bootstrap,
        "14_internal_robustness_scorecard.csv": robustness,
        "15_branch_robustness_scorecard.csv": branch_robustness,
        "16_strong_shadow_group_summary.csv": strong_shadow_summary,
        "17_r11_rank2_acceptance_gates.csv": rank2_acceptance,
        "18_signal_gap_audit.csv": signal_gap_audit,
        "19_w3_holding_coverage_audit.csv": holding_coverage,
        "20_r27_integrity_gates.csv": integrity,
        "21_audit_metadata.csv": audit_metadata,
    }
    actual_summary, actual_daily = r27_actual_coverage(history)
    files["22_actual_exit_coverage_comparison.csv"] = actual_summary
    files["23_actual_exit_daily_holdings.csv"] = actual_daily
    reports = r27_exit_reports(history, ledger)
    for name, report in zip([
        "24_exit_paired_summary.csv", "25_exit_paired_trades.csv",
        "26_first_week_forward_observations.csv", "27_exit_paired_coverage.csv",
        "28_exit_sample_completeness.csv", "29_baseline_execution_reconciliation.csv"
    ], reports):
        files[name] = report
    files["30_signal_input_fingerprints.csv"] = r271_input_manifest(history, ledger)
    files.update(r28_diagnostics if isinstance(r28_diagnostics, dict) else r28_reports(history, ledger))
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
        "R3与市场三分法冻结；R6正式Top1及第二名影子；R11先筛ATR区间再取Top1；"
        "原T+1日内-10%止损和W3报表保留，新增全池、合格候选和正式排名的同条件对照；"
        "所有完整入选信号等额独立成交，不设仓位上限、不复投。"
    )
    st.caption(f"运行引擎修订：{ENGINE_PATCH}")
    st.warning(
        "R27.1在2022—2024留出区间未通过验证。R28不优化任何入场或退出条件，"
        "不计算三仓账户收益，不自动颁发实盘合格结论。已看过的年份属于研发数据。"
    )
    st.info("R28首次需补算基础股票池的B10结果，复用原行情缓存，不需新增依赖。"
            "旧R27.1包只能恢复原报告；启动相同区间后会补扫缺少全池对照的周。"
            "R28完整包可恢复新增诊断断点。建议分别运行2022—2024和2024—2026，参数与原测试完全一致。")
    with st.expander("查看冻结交易规则"):
        st.markdown(
            """
- **R3中性**：MACD首红趋势池按原词典序取Top2；不足2只则空仓。
- **R6弱势**：原买点、五项评分及至少2只合格门槛不变；正式只买第一名，第二名记录影子退出用于覆盖对照。
- **R11强势正式信号**：完整整理再启动池先筛ATR3/ATR13在0.70—0.90，再按ATR升序取第一名。
- **R11第二名影子**：只在原R23信号周记录原第二名；原有信号与新增信号分开审计。
- **R7影子**：仅记录早期强势回调中的抗跌新高Top2，永不进入正式收益。
- **买入**：下一交易日开盘；一字涨停不虚构成交。
- **止损**：买入日不可卖，从下一交易日起执行日内-10%；计0.3%不利滑点，停牌或一字跌停顺延。
- **退出**：未触发止损的交易固定W3收盘卖出。
- **R27退出研究**：L5在任一交易日收盘跌破买价5%时触发；W1在第5个市场交易日收盘扣费后未盈利时触发。均下一可成交日开盘退出、计0.3%不利滑点，同时保留10%硬止损与W3期限，不重买。
- **成交核对**：新退出对照不使用前填收盘价卖出；开盘软退出遇跌停价保守延后，W3停牌或一字跌停也延后。原报表与可成交基准的差异单列，提前退出只与可成交基准比较。
- **资金口径**：每笔完整入选信号投入相同名义本金；无限资金、无仓位冲突、无复投。
- **研究隔离**：R7和R11第二名即使表现优秀，也必须先独立通过预设验收，不能在本版升级为买入信号。
- **其他已删除研究**：R9、R12/R13、R14周末退出、R17整仓W4、R18盈利尾仓及全池大牛机会反查。
            """
        )

    today = _shanghai_now().date()
    default_start = today - timedelta(days=365)
    with st.sidebar:
        st.header("研究配置")
        mode = st.radio(
            "运行模式",
            ["历史R28同条件增益诊断", "最新选股预览"],
            index=0,
            help="历史模式只使用完整周线；最新预览允许使用本周未完成周线且不写入回测。",
        )
        start_input = st.date_input("验证开始日期", value=default_start, disabled=mode != "历史R28同条件增益诊断")
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
        equal_notional_wan = st.number_input(
            "每笔等额名义本金（万元）",
            value=1.0,
            min_value=0.01,
            max_value=10000.0,
            step=0.5,
            help="只用于把收益率换算成名义盈亏；不限制资金、不改变任何信号。",
        )
        sample_status = st.selectbox(
            "本次区间属性",
            ["研发样本（已经看过）", "冻结后未见样本", "冻结后前向记录"],
            index=0,
            help="标签只写入审计元数据，不改变计算。未见样本必须在查看结果前指定。",
        )

        st.markdown("---")
        try:
            secret_token = st.secrets.get("TUSHARE_TOKEN", "")
        except Exception:
            secret_token = ""
        token_input = st.text_input("Tushare Token", value=secret_token, type="password")

        st.markdown("---")
        clear_market_clicked = st.button("清空行情缓存")
        clear_history_clicked = st.button("清除R28历史结果（保留行情缓存和研究池）")
        imported_results = st.file_uploader(
            "恢复R27.1或R28结果包",
            type=["zip"],
            help="R27.1包恢复正式报告，仍须补算全池；R28包恢复全池诊断。两者都复用已有行情缓存。",
        )
        import_results_clicked = st.button(
            "恢复结果包中的断点",
            disabled=imported_results is None,
        )

    if max_mv <= min_mv:
        st.error("最高流通市值必须大于最低流通市值。")
        return
    if start_input > end_input and mode == "历史R28同条件增益诊断":
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
        st.session_state.pop("r27_preview", None)
        st.success("R27历史结果和断点任务已清除。")

    token_clean = clean_token_str(token_input)
    config_id = make_config_id(min_price, min_mv, max_mv, roundtrip_cost_pct)
    if import_results_clicked and imported_results is not None:
        try:
            import_stats = import_prior_results_zip(
                imported_results.getvalue(),
                config_id,
                float(roundtrip_cost_pct),
            )
            st.success(
                f"已恢复{import_stats['candidate_rows']}条候选、"
                f"{import_stats['known_weeks']}个扫描周、"
                f"{import_stats['selected_rows']}笔冻结信号。"
                "如旧包缺少R28全池对照，请启动相同区间补扫；已有行情缓存继续复用。"
            )
        except Exception as exc:
            st.error(f"结果包恢复失败：{exc}")
    is_preview_mode = mode == "最新选股预览"
    if "r27_worker_id" not in st.session_state:
        st.session_state["r27_worker_id"] = uuid.uuid4().hex
    worker_id = str(st.session_state["r27_worker_id"])
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

    start_label = "运行最新选股预览" if is_preview_mode else "启动历史R28同条件增益诊断"
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
                    whitelist_set, name_map, industry_map = load_frozen_tech_whitelist(token_clean)
                whitelist_keys = tuple(sorted(whitelist_set))
                if not whitelist_keys:
                    raise RuntimeError("未取得科技股研究池，请检查Token权限或网络。")
                if run_history:
                    current_pool_hash = hashlib.sha256(json.dumps([sorted(whitelist_keys), name_map, industry_map], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    expected_pool_hash = active_task.get("R271_Whitelist_Hash")
                    if not expected_pool_hash:
                        existing_pool_rows = read_csv_safe(CHECKPOINT_FILE)
                        if "R271_Whitelist_Hash" in existing_pool_rows:
                            known_hashes = set(existing_pool_rows["R271_Whitelist_Hash"].dropna())
                            if known_hashes and known_hashes != {current_pool_hash}:
                                raise RuntimeError("已有结果与本机冻结研究池不同，不能混合补扫；请先导出已有结果，再清除历史结果开始独立回测。")
                    elif expected_pool_hash != current_pool_hash:
                        raise RuntimeError("本次任务的研究池身份发生变化，暂停续跑。")
                    active_task["R271_Whitelist_Hash"] = current_pool_hash
                    save_owned_task(active_task, worker_id)
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
                        # 即使某只股票提前止损，也加载到冻结W3日，才能用W3收益
                        # 独立验算新路径与买入基准确实处于同一复权尺度。
                        frozen_selected = _r19_selected(
                            frozen_rows, require_complete=True
                        )
                        last_exit = _date_series(
                            frozen_selected, "Fixed_Exit_W3_Date"
                        ).max()
                        fetch_start = min(batch_dates)
                        requested_fetch_end = (
                            last_exit.to_pydatetime()
                            if pd.notna(last_exit)
                            else datetime.strptime(max(batch_dates), "%Y%m%d")
                            + timedelta(days=30)
                        )
                    else:
                        # 下载仍分批；指标起点由每个信号日本身确定，不能由此批决定。
                        fetch_start = signal_window_start(min(batch_dates))
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
                            repaired_complete = _r19_selected(
                                candidates, require_complete=True
                            )
                            repaired_ready = (
                                _r19_candidate_path_scale_ready_mask(
                                    repaired_complete
                                )
                                if not repaired_complete.empty
                                else pd.Series(dtype=bool)
                            )
                            if (
                                not repaired_complete.empty
                                and not repaired_ready.all()
                            ):
                                failed_codes = repaired_complete.loc[
                                    ~repaired_ready, "ts_code"
                                ].astype(str).head(5).tolist()
                                raise RuntimeError(
                                    "R19.1同尺度路径补算未通过："
                                    + "、".join(failed_codes)
                                    + "。行情分片已保留，重试时只补本批。"
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
                                lease_heartbeat=lease_heartbeat,
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
                            st.session_state["r27_preview"] = candidates
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
                                    r28_audit_json=candidates.attrs.get("R28_Audit_JSON", ""),
                                    market_regime=(
                                        str(candidates["Market_Regime"].iloc[0])
                                        if not candidates.empty and "Market_Regime" in candidates.columns
                                        else "未知"
                                    ),
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
                            st.success("历史R28同条件增益诊断扫描完成。")
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

    preview = st.session_state.get("r27_preview")
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
                regime = str(
                    preview.get("Market_Regime", pd.Series("", index=preview.index)).iloc[0]
                )
                if regime == "强势":
                    watch = preview[
                        pd.to_numeric(
                            preview.get("R11_Strong_Rank", pd.Series(np.nan, index=preview.index)),
                            errors="coerce",
                        ).le(3)
                    ].copy()
                    if not watch.empty:
                        st.caption(
                            "以下是R11观察候选Top3，仅解释本周为何空仓，不是买入名单。"
                        )
                        watch_columns = [
                            "R11_Strong_Rank", "name", "ts_code", "Industry",
                            "ATR_Contraction", "R11_ATR_Band_Pass",
                            "Return_1W_pct", "Distance_MA20_pct",
                            "Weekly_Close_Location", "Market_Regime",
                        ]
                        st.dataframe(
                            watch[[column for column in watch_columns if column in watch.columns]],
                            width="stretch", hide_index=True,
                        )
            else:
                columns = [
                    "Signal_Date", "Weekly_Data_Mode", "Rank", "name",
                    "ts_code", "Industry", "Strategy_Branch",
                    "R3_Setup_Type", "Strong_Reacceleration_Setup_Type",
                    "Recovery_Setup_Type", "R11_Strong_Rank",
                    "R11_ATR_Band_Pass", "R11_Strong_Top1",
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
                "当前禁止生成审计结论；重新启动R27后只补扫异常周。"
            )
            st.dataframe(state_issues, width="stretch", hide_index=True)
            return

        equal_notional = float(equal_notional_wan) * 10000.0
        all_signal_ledger, universe = r20_all_signal_ledger(
            history, equal_notional
        )
        (
            all_signal_summary,
            branch_summary,
            year_summary,
            halfyear_summary,
        ) = r20_group_summaries(universe, equal_notional)
        rank_summary = r20_rank_summary(universe, equal_notional)
        weekly_summary = r20_weekly_summary(universe)
        rolling_summary = r20_rolling_26week_summary(universe, ledger)
        concentration = r20_concentration_audit(universe)
        branch_robustness = r27_branch_robustness_audit(universe)
        strong_shadow_summary = r27_shadow_group_summary(history)
        rank2_acceptance = r27_rank2_acceptance(history)
        signal_gap_audit = r27_signal_gap_audit(history, ledger)
        holding_coverage = r27_w3_holding_coverage_audit(history, ledger)
        bootstrap = r20_block_bootstrap(universe, ledger)
        robustness = r20_internal_robustness_scorecard(universe, bootstrap)
        integrity_gates = r27_integrity_gates(
            history, ledger, all_signal_ledger, equal_notional
        )
        audit_metadata = pd.DataFrame(
            [
                {
                    "App_Version": APP_VERSION,
                    "指标预热": "固定信号日所在周及此前59周；周一开始；60周完整范围；不以扫描批次为起点",
                    "价格尺度": "原始日线float64重建连续价格，并锚定信号日原收盘；未来数据不参与指标初始化",
                    "Strategy_Config": STRATEGY_CONFIG_VERSION,
                    "区间属性": sample_status,
                    "声明": (
                        "区间属性必须在查看结果前确定；内部稳健性不能证明不存在过拟合。"
                    ),
                    "资金口径": "每笔等额独立名义本金；无限资金；无仓位上限；无复投",
                    "每笔名义本金": equal_notional,
                    "止损": "买入次日起日内-10%，另计0.3%不利滑点",
                    "退出": "未止损则固定W3收盘",
                    "R11规则": "完整整理再启动合格池先筛ATR0.70—0.90，再按ATR升序取Top1",
                    "R6规则": "保持原资格门槛和评分，正式Top1，第二名影子",
                    "R27对照计划": "固定13周选股；B10可成交基准；L5任一收盘较买价亏损>5%；W1第5市场交易日扣费后未盈利；软信号次日开盘执行，三组均保留10%硬止损与W3期限",
                    "R27实验边界": "不自动选优；同一成熟配对样本；检查分年分支、失败段以外、错杀原盈利股；无三仓无复投",
                    "R27成交假设": "开盘软退出0.3%不利滑点，开盘处跌停价保守延后；日内止损按原一字跌停模型；停牌或零量不能卖出；W3延后卖出不按前填价成交；日线无法证明排队成交",
                    "影子规则": "R11同信号周第二名与R7早期强势Top2只记录未来路径，不进入正式收益",
                }
            ]
        )

        st.markdown("---")
        st.subheader("R27.1 历史信号可重复性")
        st.caption("同一信号日固定使用60个日历周输入。首次研究池固定保存；完整结果包包含逐周输入指纹。修改初始化口径可能改变旧版临界信号，这不代表收益改善。")
        with st.expander("上传另一次结果，核对共同历史星期（不会覆盖当前结果）"):
            comparison_upload = st.file_uploader("历史结果只读对比", type=["zip"], key="r271_compare_upload")
            if comparison_upload is not None:
                try:
                    reproducibility = r271_compare_results(history, ledger, comparison_upload.getvalue())
                    if reproducibility.empty:
                        st.info("两份结果没有共同扫描周。")
                    else:
                        st.dataframe(reproducibility, width="stretch", hide_index=True)
                        st.caption(f"共同{len(reproducibility)}周；核对通过{int(reproducibility['核对结果'].eq('通过').sum())}周。旧版缺少指纹时，仅比较名单和指标，不宣称输入完全一致。")
                except (ValueError, KeyError, zipfile.BadZipFile) as exc:
                    st.error(f"无法比较：{exc}")
        st.header("R28 同条件选股增益诊断")
        st.caption("基础池是当周满足原价格、市值和至少45根周线要求的冻结科技池。"
                   "强市候选包含原ATR区间资格；三个层级只比较相同的正式信号周。"
                   "主表按正式计划名额加权，无法成交名额记0、不补位；成交均益/胜率在逐周表单列。"
                   "这不是三仓收益，也不等于预测未来的胜率。")
        with st.spinner("生成固定种子的分层、随机和配对区组诊断……"):
            r28_diagnostics = r28_reports(history, ledger)
        r28_checks = r28_diagnostics["32_r28_week_integrity.csv"]
        if len(r28_checks):
            st.write(f"扫描{len(r28_checks)}周；可配对{int(r28_checks['可配对'].sum())}周。")
            if not r28_checks["核对"].eq("通过").all():
                st.warning("部分周缺少R28对照或核对失败，当前只显示完整周的阶段性结果，不能代表整个区间。请补扫并查看完整性说明。")
        st.dataframe(_format_report_frame(r28_diagnostics["34_r28_layer_comparison.csv"]), width="stretch", hide_index=True)
        st.dataframe(_format_report_frame(r28_diagnostics["36_r28_paired_increment.csv"]), width="stretch", hide_index=True)
        with st.expander("随机基准、逐周明细与审计边界"):
            st.caption("2000次固定种子、同周无放回抽取相同名额；不成交不换股。正式超过随机的比例不是过拟合概率。"
                       "4周区组区间未校正过去多轮策略搜索；年份与分支小样本只作诊断，不自动判定通过。"
                       "当前冻结名单不等于历史时点名单，不能宣称已排除幸存者或行业归属偏差。")
            for key in ("35_r28_random_comparison.csv", "33_r28_matched_weekly.csv", "32_r28_week_integrity.csv", "37_r28_protocol.csv"):
                st.dataframe(_format_report_frame(r28_diagnostics[key]), width="stretch", hide_index=True)

        st.header("保留：R27.1 买入后失败退出对照审计")
        st.info("选股保留R24的13周规则。两个提前退出方案只作研究，不改变入选股票，不自动升级为正式退出。")
        comparison_summary, comparison_trades, comparison_weeks, comparison_coverage, comparison_checks, execution_reconciliation = r27_exit_reports(history, ledger)
        st.subheader("10%硬止损＋W3，与两种提前退出对照")
        st.caption("L5：收盘跌破买价5%；W1：第5个市场交易日扣费后未盈利。均下一可成交日开盘卖出，保留10%日内硬止损；节假日不计入交易日。")
        st.dataframe(_format_report_frame(comparison_summary), width="stretch", hide_index=True)
        st.dataframe(_format_report_frame(comparison_coverage), width="stretch", hide_index=True)
        st.caption("三组使用同一成熟配对样本；提前退出即使已经成交，未走完W3的交易也暂不纳入优劣比较。空窗只对应配对样本，不是账户收益率。")
        st.warning("下方原基准报表保留历史口径供核对。新退出表修正停牌、零成交与W3无法卖出问题；只在可成交基准上计算提前退出的改善。")
        with st.expander("逐笔退出与改善、恶化"):
            st.dataframe(_format_report_frame(comparison_trades), width="stretch", hide_index=True)
        with st.expander("第一周表现与后续增量（诊断，不调阈值）"):
            st.caption("只看第5日之后仍持仓股票的后续增量；不能把首周涨幅与包含首周的W3总收益相关性当作预测能力。")
            st.dataframe(_format_report_frame(comparison_weeks), width="stretch", hide_index=True)
        with st.expander("样本完成度与原基准成交口径差异"):
            st.dataframe(comparison_checks, width="stretch", hide_index=True)
            st.dataframe(_format_report_frame(execution_reconciliation), width="stretch", hide_index=True)
        st.info(
            "本报告把每一笔完整入选股票都视为等额独立交易。"
            "没有三仓、没有跳过、没有复投，因此不存在起始仓位路径和后期大仓位放大。"
        )
        st.warning(
            "“名义总盈亏”只是每笔投入相同金额后的加总；因为假设资金无限，"
            "不能把它解释为某个真实账户的年收益率或累计收益率。"
        )

        selected_all = _r19_selected(history, require_complete=False)
        pending_outcomes = max(0, len(selected_all) - len(universe))
        summary_row = (
            all_signal_summary.iloc[0]
            if not all_signal_summary.empty
            else pd.Series(dtype=object)
        )
        metric_columns = st.columns(10)
        metric_columns[0].metric("扫描周", len(ledger))
        metric_columns[1].metric("全部完整交易", int(_safe_float(summary_row.get("完整交易"), 0)))
        metric_columns[2].metric("信号周", int(_safe_float(summary_row.get("信号周"), 0)))
        metric_columns[3].metric("胜率", f"{_safe_float(summary_row.get('胜率%')):.1f}%")
        metric_columns[4].metric("平均单笔收益", f"{_safe_float(summary_row.get('平均收益%')):.2f}%")
        metric_columns[5].metric("中位收益", f"{_safe_float(summary_row.get('中位收益%')):.2f}%")
        metric_columns[6].metric("Profit Factor", f"{_safe_float(summary_row.get('Profit_Factor')):.2f}")
        metric_columns[7].metric("止损交易", int(_safe_float(summary_row.get("止损交易"), 0)))
        metric_columns[8].metric("内部稳健性", f"{int(robustness['结果'].eq('通过').sum())}/{len(robustness)}")
        metric_columns[9].metric("尚未完成", pending_outcomes)
        if pending_outcomes:
            st.caption(
                f"最近{pending_outcomes}笔入选信号尚未走完W3，当前全部统计只使用"
                "已经拥有真实退出结果的交易，不用未来价格填充。"
            )

        if not actual_data_gaps.empty:
            st.error(
                f"存在{len(actual_data_gaps)}个行情缺口或未完成周，当前审计不完整。"
            )
            with st.expander("查看行情缺口"):
                st.dataframe(actual_data_gaps, width="stretch", hide_index=True)

        st.subheader("R27完整性验收")
        st.dataframe(integrity_gates, width="stretch", hide_index=True)

        st.subheader("新信号空窗与W3理论持仓覆盖")
        actual_summary, _ = r27_actual_coverage(history)
        st.dataframe(_format_report_frame(actual_summary), width="stretch", hide_index=True)
        st.caption("上表按真实止损/到期退出；R6两名对照与正式方案采用相同成熟窗口。空表表示没有足够完整路径。")
        st.dataframe(_format_report_frame(signal_gap_audit), width="stretch", hide_index=True)
        st.dataframe(_format_report_frame(holding_coverage), width="stretch", hide_index=True)
        st.caption(
            "空窗按新信号计算；理论持仓覆盖只表示已有信号固定持有W3时可能跨越的星期，"
            "忽略提前止损和资金限制，不能冒充实际账户持仓率。"
        )

        st.subheader("全部入选信号等额结果")
        st.dataframe(
            _format_report_frame(all_signal_summary), width="stretch", hide_index=True
        )

        st.subheader("内部稳健性检查")
        st.dataframe(
            _format_report_frame(robustness), width="stretch", hide_index=True
        )
        st.caption(
            "这些项目只能发现样本过少、利润集中、分支失效或统计区间不稳定；"
            "即使全部通过，也必须再用冻结后未见样本或前向记录验证。"
        )

        st.subheader("分支级稳健性检查")
        st.dataframe(
            _format_report_frame(branch_robustness), width="stretch", hide_index=True
        )
        st.caption(
            "每个分支必须单独满足样本数、平均与中位收益、Profit Factor及删除前五大盈利后仍为正；"
            "总体盈利不能替代分支通过。"
        )

        st.subheader("强市正式信号与影子对照")
        st.dataframe(_format_report_frame(strong_shadow_summary), width="stretch", hide_index=True)
        st.subheader("R11第二名能否升级")
        st.dataframe(_format_report_frame(rank2_acceptance), width="stretch", hide_index=True)
        st.caption(
            "第二名只有全部预设项目通过才值得进入下一轮正式化研究；本版无论结果如何都不买第二名。"
        )

        st.subheader("利润集中度")
        st.dataframe(
            _format_report_frame(concentration), width="stretch", hide_index=True
        )

        st.subheader("强势、中性、弱势分支稳定性")
        st.dataframe(
            _format_report_frame(branch_summary), width="stretch", hide_index=True
        )

        st.subheader("年度与半年时间稳定性")
        st.dataframe(
            _format_report_frame(year_summary), width="stretch", hide_index=True
        )
        st.dataframe(
            _format_report_frame(halfyear_summary), width="stretch", hide_index=True
        )

        st.subheader("第一名、第二名稳定性")
        st.dataframe(
            _format_report_frame(rank_summary), width="stretch", hide_index=True
        )

        st.subheader("连续26个扫描周滚动稳定性")
        if rolling_summary.empty:
            st.info("扫描周不足26周，暂时不能生成滚动稳定性。")
        else:
            st.dataframe(
                _format_report_frame(rolling_summary), width="stretch", hide_index=True
            )

        st.subheader("4周区组自助置信区间")
        st.dataframe(
            _format_report_frame(bootstrap), width="stretch", hide_index=True
        )
        st.caption(
            "按连续4个扫描周成块重采样2000次，保留同周股票及W3持有期重叠的相关性。"
            "它衡量当前样本内部不确定性，不是未见年份测试。"
        )

        with st.expander("查看每个信号周的等权表现"):
            st.dataframe(
                _format_report_frame(weekly_summary), width="stretch", hide_index=True
            )

        with st.expander("查看全部等额独立交易"):
            st.dataframe(
                _format_report_frame(all_signal_ledger), width="stretch", hide_index=True
            )

        selected_detail = _r19_selected(history, require_complete=False)
        with st.expander("查看冻结入选信号明细"):
            detail_columns = [
                "Signal_Date", "Entry_Date", "Rank", "name", "ts_code",
                "Industry", "Market_Regime", "Strategy_Branch",
                "R11_Strong_Rank", "R11_ATR_Band_Pass",
                "R11_Strong_Top1", "ATR_Contraction",
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

        with st.expander("查看R11第二名与R7影子明细"):
            shadow_detail = history[_bool_series(history, "R27_Shadow_Tracked")].copy()
            shadow_columns = [
                "Signal_Date", "Entry_Date", "name", "ts_code", "Industry",
                "Market_Regime", "R11_Strong_Rank", "R11_Second_Shadow",
                "R7_Strong_Rank", "R7_Shadow_Top2", "ATR_Contraction",
                "Entry_Tradable", "Fixed_Return_W3_Net_pct",
                "R16_Stop_Minus10_Triggered", "R16_Stop_Minus10_Return_Net_pct",
                "MFE_W3_Net_pct", "MAE_W3_Raw_pct", "Outcome_Grade",
            ]
            st.dataframe(
                shadow_detail[[column for column in shadow_columns if column in shadow_detail.columns]]
                .sort_values(["Signal_Date", "R11_Strong_Rank", "R7_Strong_Rank"],
                             ascending=[False, True, True], na_position="last", kind="mergesort"),
                width="stretch", hide_index=True,
            )

        export_bytes = build_r27_export_zip(
            history.drop(columns=["Config_ID"], errors="ignore"),
            ledger.drop(columns=["Config_ID"], errors="ignore"),
            data_gap_rows,
            all_signal_summary,
            all_signal_ledger,
            branch_summary,
            year_summary,
            halfyear_summary,
            rank_summary,
            weekly_summary,
            rolling_summary,
            concentration,
            bootstrap,
            robustness,
            branch_robustness,
            strong_shadow_summary,
            rank2_acceptance,
            signal_gap_audit,
            holding_coverage,
            integrity_gates,
            audit_metadata,
            r28_diagnostics=r28_diagnostics,
        )
        st.download_button(
            "下载R28完整审计结果",
            data=export_bytes,
            file_name="r28_frozen_edge_diagnostic_audit_results.zip",
            mime="application/zip",
        )

if __name__ == "__main__":
    main()
