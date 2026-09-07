# -*- coding: utf-8 -*-
"""持有期对比验证器（单文件独立版，直接覆盖 app.py 运行）。

上一轮的死结：26周持有 + 3仓，四年只买了46笔（信号共1277个），
运气区间宽达152个百分点，各仓位数的优选分位在32%~96%之间乱跳
——说明三仓下80%的分位只是噪声，不是排序能力。

根本原因是交易频率：3仓 × 平均持有13周，一年最多11-12笔。
46笔样本量下，任何排序规则的优势都无法从运气中分离出来。

本轮唯一改变最长持有周数（8/10/12/13/26周），信号规则与市值排序完全不动。
要看两件事：
  1. 交易笔数能增加到多少
  2. 优选分位是否变稳定——这比单次收益数字重要得多

同时量化代价：缩短持有期会砍掉多少还在上涨的仓位，
用"买到的票在26周内本来能涨到多少"来衡量捕获率。

所有持有期在一次遍历中算出，避免重复扫描。移动止损15%始终生效，
最长持有周数只决定强制退出时点。

内存优化：边构建周线边释放日线。行情缓存与之前共用。
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
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts

warnings.filterwarnings("ignore")

APP_TITLE = "持有期对比：8-26周"
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

# -----------------------------------------------------------------------------
# 信号与路径
# -----------------------------------------------------------------------------





# =============================================================================
# 冻结规则（信号与排序不变，本轮唯一变量是最长持有周数）
# =============================================================================
FROZEN_BREAKOUT_WEEKS = 26
FROZEN_POSITION_QUANTILE = 0.33
FROZEN_VOL_CONTRACTION_MAX = 0.8
FROZEN_STOP_PCT = 15.0
FROZEN_RANK_COLUMN = "MV_Billion"

HOLD_OPTIONS = [8, 10, 12, 13, 26]


def compute_features(weekly: pd.DataFrame):
    close = pd.to_numeric(weekly["close"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    return_1w = (close / close.shift(1) - 1.0) * 100.0
    features = pd.DataFrame(index=weekly.index)
    features["breakout"] = close > close.shift(1).rolling(FROZEN_BREAKOUT_WEEKS).max()
    features["position_2y"] = close / high.shift(1).rolling(104).max().replace(0, np.nan)
    vol_recent = return_1w.shift(1).rolling(8).std()
    vol_earlier = return_1w.shift(9).rolling(18).std()
    features["vol_contraction"] = vol_recent / vol_earlier.replace(0, np.nan)
    return features


def evaluate_stock(weekly: pd.DataFrame, ts_code: str, position_threshold: float):
    """一次遍历算出所有持有期的结果，避免重复扫描。

    移动止损始终生效；最长持有周数只决定强制退出的时点。
    因此同一个信号在不同持有期下，若止损先触发则结果完全相同——
    这正是我们要观察的：缩短持有期到底砍掉了多少还在上涨的仓位。
    """
    if len(weekly) < 140:
        return pd.DataFrame()
    features = compute_features(weekly)
    close_values = pd.to_numeric(weekly["close"], errors="coerce").to_numpy()
    high_values = pd.to_numeric(weekly["high"], errors="coerce").to_numpy()
    open_values = pd.to_numeric(weekly["open"], errors="coerce").to_numpy()
    dates = weekly["trade_date_str"].astype(str).tolist()
    n = len(weekly)

    breakout = features["breakout"].fillna(False).to_numpy()
    position = features["position_2y"].to_numpy()
    contraction = features["vol_contraction"].to_numpy()
    max_hold = max(HOLD_OPTIONS)

    rows = []
    for i in range(n - 1):
        if not breakout[i]:
            continue
        if not (math.isfinite(position[i]) and position[i] >= position_threshold):
            continue
        if not (
            math.isfinite(contraction[i]) and contraction[i] <= FROZEN_VOL_CONTRACTION_MAX
        ):
            continue
        entry_price = open_values[i + 1]
        if not math.isfinite(entry_price) or entry_price <= 0:
            continue
        if i + 1 >= n:
            continue

        row = {
            "ts_code": ts_code,
            "Signal_Week": dates[i],
            "Entry_Week": dates[i + 1],
            "Entry_Price": entry_price,
        }
        # 最长窗口内的理论最大涨幅（用于统计抓到多少翻倍股）
        long_stop = min(i + max_hold, n - 1)
        window_high = high_values[i + 1 : long_stop + 1]
        finite_high = window_high[np.isfinite(window_high)]
        row["Max_Gain_26W"] = (
            (finite_high.max() / entry_price - 1.0) * 100.0 if finite_high.size else np.nan
        )

        ok = False
        for hold in HOLD_OPTIONS:
            stop_index = min(i + hold, n - 1)
            if stop_index <= i + 1:
                continue
            peak = entry_price
            exit_price = None
            exit_index = stop_index
            for j in range(i + 1, stop_index + 1):
                current = close_values[j]
                if not math.isfinite(current):
                    continue
                peak = max(peak, current)
                if current <= peak * (1.0 - FROZEN_STOP_PCT / 100.0):
                    exit_price = current
                    exit_index = j
                    break
            if exit_price is None:
                exit_price = close_values[stop_index]
            if not math.isfinite(exit_price):
                continue
            window = high_values[i + 1 : exit_index + 1]
            finite_window = window[np.isfinite(window)]
            row[f"Exit_Week_{hold}"] = dates[exit_index]
            row[f"Return_{hold}"] = (exit_price / entry_price - 1.0) * 100.0
            row[f"MaxGain_{hold}"] = (
                (finite_window.max() / entry_price - 1.0) * 100.0
                if finite_window.size
                else np.nan
            )
            row[f"Hold_{hold}"] = exit_index - i
            ok = True
        if ok:
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 仓位调度
# -----------------------------------------------------------------------------
def simulate_slots(
    signals, week_index, hold, slot_count, cost_pct,
    order_mode="rank", seed=0, keep_ledger=True,
):
    exit_col, ret_col = f"Exit_Week_{hold}", f"Return_{hold}"
    if exit_col not in signals.columns:
        return pd.DataFrame(), 0.0
    work = signals.dropna(subset=[exit_col, ret_col]).copy()
    work["Entry_Index"] = work["Entry_Week"].map(week_index)
    work["Exit_Index"] = work[exit_col].map(week_index)
    work = work.dropna(subset=["Entry_Index", "Exit_Index"])
    if work.empty:
        return pd.DataFrame(), 0.0
    work["Entry_Index"] = work["Entry_Index"].astype(int)
    work["Exit_Index"] = work["Exit_Index"].astype(int)

    rng = np.random.default_rng(seed)
    if order_mode == "random":
        work["_order"] = rng.random(len(work))
    else:
        work["_order"] = (
            -pd.to_numeric(work[FROZEN_RANK_COLUMN], errors="coerce")
        ).fillna(np.inf)

    work = work.sort_values(["Entry_Index", "_order", "ts_code"], kind="mergesort")
    records = work.to_dict("records")

    slot_value = [1.0 / slot_count] * slot_count
    slot_free_at = [0] * slot_count
    slot_code = [None] * slot_count
    trades = []
    for row in records:
        entry_index = int(row["Entry_Index"])
        free = [i for i in range(slot_count) if slot_free_at[i] <= entry_index]
        held = {
            slot_code[i]
            for i in range(slot_count)
            if slot_free_at[i] > entry_index and slot_code[i] is not None
        }
        if str(row["ts_code"]) in held or not free:
            continue
        slot = free[0]
        net = float(row[ret_col]) - cost_pct
        slot_value[slot] *= 1.0 + net / 100.0
        slot_free_at[slot] = int(row["Exit_Index"]) + 1
        slot_code[slot] = str(row["ts_code"])
        if keep_ledger:
            trades.append(
                {
                    "Signal_Week": row["Signal_Week"],
                    "ts_code": row["ts_code"],
                    "MV_Billion": row.get("MV_Billion"),
                    "净收益%": net,
                    "Max_Gain_26W": row.get("Max_Gain_26W"),
                    "持有周数": row.get(f"Hold_{hold}"),
                }
            )
        else:
            trades.append({"净收益%": net, "Max_Gain_26W": row.get("Max_Gain_26W")})
    return pd.DataFrame(trades), (sum(slot_value) - 1.0) * 100.0


def monte_carlo(signals, week_index, hold, slot_count, cost_pct, runs):
    outcomes = []
    for run in range(runs):
        _, total = simulate_slots(
            signals, week_index, hold, slot_count, cost_pct,
            order_mode="random", seed=run + 1, keep_ledger=False,
        )
        outcomes.append(total)
    return np.array(outcomes, dtype=float)


def hold_comparison(signals, week_index, slot_count, cost_pct, mc_runs, progress_callback=None):
    """核心表：不同持有期在三仓约束下的交易频率与统计可靠性。"""
    rows = []
    for step, hold in enumerate(HOLD_OPTIONS):
        ledger, total = simulate_slots(
            signals, week_index, hold, slot_count, cost_pct, order_mode="rank"
        )
        if ledger.empty:
            continue
        outcomes = monte_carlo(signals, week_index, hold, slot_count, cost_pct, mc_runs)
        net = pd.to_numeric(ledger["净收益%"], errors="coerce").dropna()
        gain26 = pd.to_numeric(ledger["Max_Gain_26W"], errors="coerce").dropna()
        held = pd.to_numeric(ledger["持有周数"], errors="coerce").dropna()
        low, high = np.percentile(outcomes, 5), np.percentile(outcomes, 95)
        rows.append(
            {
                "最长持有周数": hold,
                "四年买入笔数": int(len(net)),
                "年均笔数": round(len(net) / 4.0, 1),
                "平均实际持有周": float(held.mean()) if len(held) else np.nan,
                "单笔平均%": float(net.mean()),
                "单笔中位%": float(net.median()),
                "胜率%": float((net > 0).mean() * 100.0),
                "买到的票26周内翻倍数": int((gain26 > 100).sum()),
                "总收益%": total,
                "随机中位%": float(np.median(outcomes)),
                "运气区间宽度pp": float(high - low),
                "优选分位%": float((outcomes < total).mean() * 100.0),
            }
        )
        if progress_callback:
            progress_callback((step + 1) / len(HOLD_OPTIONS))
    return pd.DataFrame(rows)


def slot_stability(signals, week_index, hold, cost_pct, mc_runs, slot_values):
    """在选定持有期下，检验分位是否在各仓位数上稳定。

    上一轮26周持有时分位在32%~96%之间乱跳，正是这个不稳定暴露了
    "80%只是噪声"。交易笔数增加后，分位应当变得稳定才可信。
    """
    rows = []
    for slots in slot_values:
        _, total = simulate_slots(
            signals, week_index, hold, int(slots), cost_pct,
            order_mode="rank", keep_ledger=False,
        )
        outcomes = monte_carlo(signals, week_index, hold, int(slots), cost_pct, mc_runs)
        low, high = np.percentile(outcomes, 5), np.percentile(outcomes, 95)
        rows.append(
            {
                "仓位数": int(slots),
                "总收益%": total,
                "随机中位%": float(np.median(outcomes)),
                "运气区间宽度pp": float(high - low),
                "优选分位%": float((outcomes < total).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def yearly_by_hold(signals, week_index, hold, slot_count, cost_pct):
    ledger, _ = simulate_slots(
        signals, week_index, hold, slot_count, cost_pct, order_mode="rank"
    )
    if ledger.empty:
        return pd.DataFrame()
    ledger["年份"] = ledger["Signal_Week"].astype(str).str[:4]
    rows = []
    for year, group in ledger.groupby("年份"):
        net = pd.to_numeric(group["净收益%"], errors="coerce").dropna()
        gain = pd.to_numeric(group["Max_Gain_26W"], errors="coerce").dropna()
        rows.append(
            {
                "年份": year,
                "买入笔数": int(len(net)),
                "单笔平均%": float(net.mean()),
                "单笔中位%": float(net.median()),
                "胜率%": float((net > 0).mean() * 100.0),
                "买到的票26周内翻倍数": int((gain > 100).sum()),
                "最大单笔%": float(net.max()),
                "最差单笔%": float(net.min()),
            }
        )
    return pd.DataFrame(rows)


def _memory_usage_mb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"⏱ {APP_TITLE}")
    st.caption("缩短持有期换来更多交易笔数，能不能让排序规则的优势真正显现？")
    st.info(
        "**上一轮的死结**：26周持有 + 3仓，四年只买了46笔（信号有1277个），"
        "运气区间宽达152个百分点，各仓位数的分位在32%~96%之间乱跳——"
        "**说明3仓下80%的分位只是噪声，不是能力**。\n\n"
        "**本轮唯一改变的是最长持有周数**（8/10/12/13/26周），"
        "信号规则和市值排序完全不动。要看两件事：\n\n"
        "1. 交易笔数能增加到多少（笔数越多，优势才越可能显现）\n"
        "2. **分位是否变稳定**——这是判断排序规则是否真的有效的关键，"
        "比单次收益数字重要得多"
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
        slot_count = st.number_input("仓位数", value=3, min_value=1, max_value=20, step=1)
        detail_hold = st.selectbox(
            "详细分析哪个持有期", HOLD_OPTIONS, index=2,
            help="表3和表4会针对这个持有期展开。",
        )
        mc_runs = st.number_input(
            "蒙特卡洛次数", value=150, min_value=20, max_value=600, step=10,
            help="要跑5个持有期，次数太多会很慢。",
        )
        cost_pct = st.number_input("往返成本%", value=0.20, min_value=0.0, max_value=2.0, step=0.05)

        st.markdown("---")
        st.caption("信号与排序规则已锁死。")
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
        if st.session_state.get("hold_result"):
            return
        st.markdown(
            """
### 四张表

**表1 · 持有期对比**（核心）　8/10/12/13/26周各自的交易笔数、收益、
运气区间宽度、优选分位。重点看**笔数增加后运气区间有没有收窄**。

**表2 · 你要付出的代价**　缩短持有期会砍掉多少还在上涨的仓位——
用"买到的票在26周内本来能涨到多少"来衡量。

**表3 · 分位稳定性**　选定持有期下，1到10仓的分位是否稳定在高位。
上一轮26周时是32%~96%乱跳，这次如果稳定在70%以上，才说明排序真的有效。

**表4 · 分年度**　实盘会经历的样子。

---
**判断标准**：
- 笔数至少翻倍（46 → 90以上）
- 运气区间明显收窄（152pp → 100pp以内）
- **各仓位分位稳定在60%以上，不再乱跳**

三条都满足，这套东西才算可用。
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
    fetch_start = (pd.Timestamp(start_input) - timedelta(days=900)).strftime("%Y%m%d")
    fetch_end = (pd.Timestamp(end_input) + timedelta(days=220)).strftime("%Y%m%d")

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

    if not basic_indexed.empty and "circ_mv" in basic_indexed.columns:
        mv_lookup = (
            basic_indexed[["circ_mv"]].reset_index().rename(
                columns={"trade_date_str": "Signal_Week"}
            )
        )
        mv_lookup["circ_mv"] = pd.to_numeric(
            mv_lookup["circ_mv"], errors="coerce"
        ).astype("float32")
        mv_lookup = mv_lookup.drop_duplicates(["Signal_Week", "ts_code"])
    else:
        st.error("缺少市值数据。")
        return
    del basic_indexed
    gc.collect()

    needed = ["trade_date_str", "open", "high", "low", "close"]
    weekly_cache = {}
    position_samples = []
    all_weeks = set()
    codes = sorted(stocks.keys())
    prep = st.progress(0.0, text="构建周线……")
    for idx, ts_code in enumerate(codes):
        daily = stocks.pop(ts_code)
        weekly = build_weekly_bars(daily)
        del daily
        if weekly.empty or len(weekly) < 140:
            continue
        keep = [c for c in needed if c in weekly.columns]
        weekly = weekly[keep].copy()
        for column in ("open", "high", "low", "close"):
            if column in weekly.columns:
                weekly[column] = pd.to_numeric(weekly[column], errors="coerce").astype("float32")
        weekly_cache[ts_code] = weekly
        all_weeks.update(weekly["trade_date_str"].astype(str).tolist())
        features = compute_features(weekly)
        position_samples.append(
            features.loc[features["breakout"].fillna(False), "position_2y"]
            .dropna().astype("float32")
        )
        del features
        if idx % 60 == 0:
            prep.progress(min((idx + 1) / len(codes), 1.0))
    prep.empty()
    del stocks
    gc.collect()

    if not position_samples:
        st.error("数据不足。")
        return
    all_positions = pd.concat(position_samples, ignore_index=True)
    del position_samples
    gc.collect()
    position_threshold = float(all_positions.quantile(1.0 - FROZEN_POSITION_QUANTILE))
    del all_positions
    gc.collect()

    progress = st.progress(0.0, text="生成信号（一次算出全部持有期）……")
    parts = []
    cached = list(weekly_cache.keys())
    for idx, ts_code in enumerate(cached):
        weekly = weekly_cache.pop(ts_code)
        rows = evaluate_stock(weekly, ts_code, position_threshold)
        del weekly
        if not rows.empty:
            parts.append(rows)
        if idx % 40 == 0:
            progress.progress(min((idx + 1) / len(cached), 1.0))
    progress.empty()
    del weekly_cache
    gc.collect()

    if not parts:
        st.error("没有产生信号。")
        return
    signals = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    signals = signals[
        (signals["Signal_Week"] >= start_date) & (signals["Signal_Week"] <= end_date)
    ]
    signals = signals[pd.to_numeric(signals["Entry_Price"], errors="coerce") >= min_price]
    signals = signals.merge(mv_lookup, on=["Signal_Week", "ts_code"], how="left")
    del mv_lookup
    gc.collect()
    signals["MV_Billion"] = pd.to_numeric(signals["circ_mv"], errors="coerce") / 10000.0
    signals = signals[signals["MV_Billion"].between(min_mv, max_mv)].reset_index(drop=True)
    if signals.empty:
        st.error("过滤后无信号。")
        return

    week_index = {week: i for i, week in enumerate(sorted(all_weeks))}

    cmp_progress = st.progress(0.0, text="对比各持有期……")
    comparison = hold_comparison(
        signals, week_index, int(slot_count), float(cost_pct), int(mc_runs),
        progress_callback=lambda p: cmp_progress.progress(p),
    )
    cmp_progress.empty()

    stability = slot_stability(
        signals, week_index, int(detail_hold), float(cost_pct),
        max(30, int(mc_runs) // 3), [1, 2, 3, 4, 5, 6, 8, 10],
    )
    yearly = yearly_by_hold(
        signals, week_index, int(detail_hold), int(slot_count), float(cost_pct)
    )

    # 缩短持有期的代价：买到的票原本能涨多少 vs 实际拿到多少
    cost_rows = []
    for hold in HOLD_OPTIONS:
        ledger, _ = simulate_slots(
            signals, week_index, hold, int(slot_count), float(cost_pct), order_mode="rank"
        )
        if ledger.empty:
            continue
        net = pd.to_numeric(ledger["净收益%"], errors="coerce")
        gain26 = pd.to_numeric(ledger["Max_Gain_26W"], errors="coerce")
        valid = net.notna() & gain26.notna() & (gain26 > 0)
        cost_rows.append(
            {
                "最长持有周数": hold,
                "买入笔数": int(len(net.dropna())),
                "买到的票26周理论涨幅均值%": float(gain26[valid].mean()),
                "实际到手均值%": float(net[valid].mean()),
                "捕获率%": float((net[valid] / gain26[valid]).clip(-2, 2).mean() * 100.0),
                "26周内本可翻倍的票数": int((gain26 > 100).sum()),
                "其中实际赚超50%的": int(((gain26 > 100) & (net > 50)).sum()),
            }
        )

    st.session_state["hold_result"] = {
        "signal_count": len(signals),
        "comparison": comparison,
        "capture": pd.DataFrame(cost_rows),
        "stability": stability,
        "yearly": yearly,
        "slot_count": int(slot_count),
        "detail_hold": int(detail_hold),
        "period": f"{start_date} — {end_date}",
        "memory_mb": _memory_usage_mb(),
    }


def render_results():
    result = st.session_state.get("hold_result")
    if not result:
        return False

    st.markdown("---")
    st.header("持有期对比结果")
    st.caption(
        f"突破信号 {result['signal_count']:,} 个　|　区间 {result['period']}　|　"
        f"{result['slot_count']}仓"
        + (
            f"　|　内存 {result['memory_mb']:.0f} MB"
            if math.isfinite(result.get("memory_mb", float("nan")))
            else ""
        )
    )

    st.subheader("表1 · 各持有期在三仓约束下的表现")
    st.dataframe(result["comparison"].round(2), width="stretch", hide_index=True)
    st.caption(
        "**最该看的不是「总收益%」，而是「运气区间宽度」和「优选分位%」。**\n\n"
        "26周那行是上一轮的结果（46笔、区间152pp）。如果缩短持有期后笔数明显增加、"
        "区间明显收窄，说明这条路走通了；如果区间依然很宽，"
        "说明3仓的样本量问题不是靠缩短持有期能解决的。"
    )

    if not result["capture"].empty:
        st.subheader("表2 · 缩短持有期的代价")
        st.dataframe(result["capture"].round(2), width="stretch", hide_index=True)
        st.caption(
            "「捕获率%」= 实际到手 ÷ 该票26周内的理论涨幅。"
            "最后两列尤其重要：**买到的票里本来能翻倍的有几只，实际赚超50%的有几只**——"
            "这直接量化了你为提高交易频率放弃了多少大波段。"
        )

    if not result["stability"].empty:
        st.subheader(f"表3 · 分位稳定性（持有{result['detail_hold']}周）")
        st.dataframe(result["stability"].round(2), width="stretch", hide_index=True)
        st.caption(
            "**这是判断排序规则是否真有效的关键。**上一轮26周持有时，"
            "各仓位的分位在32%~96%之间乱跳，暴露了那个80%只是噪声。"
            "这次如果各仓位分位稳定在60%以上，才说明市值排序真的站得住。"
        )

    if not result["yearly"].empty:
        st.subheader(f"表4 · 分年度（持有{result['detail_hold']}周，{result['slot_count']}仓）")
        st.dataframe(result["yearly"].round(2), width="stretch", hide_index=True)
        st.caption("实盘会经历的样子，注意亏损年份的胜率和最差单笔。")

    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_hold_comparison.csv",
            result["comparison"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "02_capture_cost.csv",
            result["capture"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "03_slot_stability.csv",
            result["stability"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "04_yearly.csv", result["yearly"].to_csv(index=False, encoding="utf-8-sig")
        )
    st.download_button(
        "下载验证结果",
        data=output.getvalue(),
        file_name="hold_period_validation.zip",
        mime="application/zip",
        key="download_hold",
    )
    return True


if __name__ == "__main__":
    main()
    render_results()
