# -*- coding: utf-8 -*-
"""三仓约束下的排序层验证器（单文件独立版，直接覆盖 app.py 运行）。

上一轮（无资金约束）结论：十个候选排序变量中，只有流通市值（越大越优先）
三关全过——正向+1.31%/t=1.97、反向-2.02%、四年全部为正且逐年增强
（+0.26/+0.83/+3.15/+3.84）、TopN衰减平滑（Top3 7.85% → 全部 4.57%）。
而横盘区间宽度、距40周均线全期看着更好（+2.14/+1.64），
但一分年度就露馅：优势全部来自2026一年，2023-2025三年均为负。

但那是无资金约束下测的。SKDJ阶段的教训很清楚：
K值排序在无约束时看着可以，一加三仓就失效（蒙特卡洛分位仅45%，不如随机）。

因此本轮验证：市值排序在三仓约束下能否明显跑赢随机选股。
方法是蒙特卡洛——保持完全相同的资金约束与路径依赖结构，
只把"选哪只"换成随机，重复数百次得到运气分布，
再看真实规则落在这个分布的哪个分位。

判断标准：分位 > 80% 才算排序层在真实约束下真正有效。

信号规则与排序规则全部锁死为模块级常量，本轮不做任何修改。
内存优化：边构建周线边释放日线。

行情缓存与之前共用。
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

APP_TITLE = "三仓约束下的市值排序验证"
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
# 冻结规则（信号+排序，全部锁死）
# 信号规则来自样本外测试；排序规则来自上一轮验证：
#   流通市值是十个候选里唯一三关全过的（正向+1.31%/t=1.97、反向-2.02%、
#   四年全部为正且逐年增强、TopN衰减平滑），且与市值分层发现互相印证。
# =============================================================================
FROZEN_BREAKOUT_WEEKS = 26
FROZEN_POSITION_QUANTILE = 0.33
FROZEN_VOL_CONTRACTION_MAX = 0.8
FROZEN_FORWARD_WEEKS = 26
FROZEN_STOP_PCT = 15.0
FROZEN_RANK_COLUMN = "MV_Billion"
FROZEN_RANK_ASCENDING = False  # 市值越大越优先


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
    """输出符合冻结信号的周，含入场/退出所需的全部信息。

    退出用移动止损，并记录实际持有周数——三仓调度需要知道仓位何时释放。
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
        stop_index = min(i + FROZEN_FORWARD_WEEKS, n - 1)
        if stop_index <= i + 1:
            continue
        window_high = high_values[i + 1 : stop_index + 1]
        finite_high = window_high[np.isfinite(window_high)]
        if not finite_high.size:
            continue
        max_gain = (finite_high.max() / entry_price - 1.0) * 100.0

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

        rows.append(
            {
                "ts_code": ts_code,
                "Signal_Week": dates[i],
                "Entry_Week": dates[i + 1],
                "Exit_Week": dates[exit_index],
                "Entry_Price": entry_price,
                "Max_Gain_pct": max_gain,
                "Return_pct": (exit_price / entry_price - 1.0) * 100.0,
                "Hold_Weeks": exit_index - (i + 1) + 1,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 三仓调度
# -----------------------------------------------------------------------------
def simulate_slots(
    signals: pd.DataFrame, week_index: dict, slot_count: int, cost_pct: float,
    order_mode: str = "rank", seed: int = 0, keep_ledger: bool = True,
):
    """固定仓位调度。order_mode='rank'按冻结排序优选，'random'随机取。

    随机模式用于蒙特卡洛：保持完全相同的资金约束与路径依赖结构，
    只把"选哪只"变成随机，从而分离排序规则的贡献与运气的贡献。
    """
    if signals.empty:
        return pd.DataFrame(), 0.0

    work = signals.copy()
    work["Entry_Index"] = work["Entry_Week"].map(week_index)
    work["Exit_Index"] = work["Exit_Week"].map(week_index)
    work = work.dropna(subset=["Entry_Index", "Exit_Index"])
    if work.empty:
        return pd.DataFrame(), 0.0
    work["Entry_Index"] = work["Entry_Index"].astype(int)
    work["Exit_Index"] = work["Exit_Index"].astype(int)

    rng = np.random.default_rng(seed)
    if order_mode == "random":
        work["_order"] = rng.random(len(work))
    else:
        values = pd.to_numeric(work[FROZEN_RANK_COLUMN], errors="coerce")
        work["_order"] = (-values).fillna(np.inf) if not FROZEN_RANK_ASCENDING else values.fillna(np.inf)

    work = work.sort_values(["Entry_Index", "_order", "ts_code"], kind="mergesort")
    records = work.to_dict("records")

    slot_value = [1.0 / slot_count] * slot_count
    slot_free_at = [0] * slot_count
    slot_code = [None] * slot_count
    trades = []

    for row in records:
        entry_index = int(row["Entry_Index"])
        free_slots = [i for i in range(slot_count) if slot_free_at[i] <= entry_index]
        held = {
            slot_code[i]
            for i in range(slot_count)
            if slot_free_at[i] > entry_index and slot_code[i] is not None
        }
        if str(row["ts_code"]) in held:
            if keep_ledger:
                trades.append({**row, "执行": "跳过", "原因": "已持有同股"})
            continue
        if not free_slots:
            if keep_ledger:
                trades.append({**row, "执行": "跳过", "原因": "仓位已满"})
            continue
        slot = free_slots[0]
        net = float(row["Return_pct"]) - cost_pct
        slot_value[slot] *= 1.0 + net / 100.0
        slot_free_at[slot] = int(row["Exit_Index"]) + 1
        slot_code[slot] = str(row["ts_code"])
        if keep_ledger:
            trades.append({**row, "执行": "买入", "原因": "", "仓位": slot + 1, "净收益%": net})
        else:
            trades.append({"执行": "买入", "净收益%": net, "Max_Gain_pct": row["Max_Gain_pct"]})

    ledger = pd.DataFrame(trades)
    return ledger, (sum(slot_value) - 1.0) * 100.0


def monte_carlo(signals, week_index, slot_count, cost_pct, runs, progress_callback=None):
    outcomes = []
    for run in range(runs):
        _, total = simulate_slots(
            signals, week_index, slot_count, cost_pct,
            order_mode="random", seed=run + 1, keep_ledger=False,
        )
        outcomes.append(total)
        if progress_callback and (run % 10 == 0 or run == runs - 1):
            progress_callback((run + 1) / runs)
    return np.array(outcomes, dtype=float)


def slot_sweep(signals, week_index, cost_pct, slot_values, mc_runs, progress_callback=None):
    rows = []
    for step, slots in enumerate(slot_values):
        ledger, total = simulate_slots(
            signals, week_index, int(slots), cost_pct, order_mode="rank", keep_ledger=False
        )
        outcomes = monte_carlo(signals, week_index, int(slots), cost_pct, int(mc_runs))
        bought = ledger[ledger["执行"] == "买入"] if not ledger.empty else pd.DataFrame()
        net = (
            pd.to_numeric(bought["净收益%"], errors="coerce").dropna()
            if not bought.empty
            else pd.Series(dtype=float)
        )
        gain = (
            pd.to_numeric(bought["Max_Gain_pct"], errors="coerce").dropna()
            if not bought.empty
            else pd.Series(dtype=float)
        )
        low, high = float(np.percentile(outcomes, 5)), float(np.percentile(outcomes, 95))
        rows.append(
            {
                "仓位数": int(slots),
                "实际买入": int(len(net)),
                "单笔平均%": float(net.mean()) if len(net) else np.nan,
                "单笔胜率%": float((net > 0).mean() * 100.0) if len(net) else np.nan,
                "翻倍股数": int((gain > 100).sum()) if len(gain) else 0,
                "按市值优选总收益%": total,
                "随机中位数%": float(np.median(outcomes)),
                "运气区间宽度pp": high - low,
                "优选所处分位%": float((outcomes < total).mean() * 100.0),
            }
        )
        if progress_callback:
            progress_callback((step + 1) / max(len(slot_values), 1))
    return pd.DataFrame(rows)


def yearly_slots(ledger: pd.DataFrame):
    if ledger.empty:
        return pd.DataFrame()
    bought = ledger[ledger["执行"] == "买入"].copy()
    if bought.empty:
        return pd.DataFrame()
    bought["年份"] = bought["Signal_Week"].astype(str).str[:4]
    rows = []
    for year, group in bought.groupby("年份"):
        net = pd.to_numeric(group["净收益%"], errors="coerce").dropna()
        gain = pd.to_numeric(group["Max_Gain_pct"], errors="coerce").dropna()
        rows.append(
            {
                "年份": year,
                "买入笔数": int(len(net)),
                "单笔平均%": float(net.mean()) if len(net) else np.nan,
                "单笔中位%": float(net.median()) if len(net) else np.nan,
                "胜率%": float((net > 0).mean() * 100.0) if len(net) else np.nan,
                "翻倍股数": int((gain > 100).sum()),
                "最大单笔%": float(net.max()) if len(net) else np.nan,
                "最差单笔%": float(net.min()) if len(net) else np.nan,
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
    st.title(f"🎰 {APP_TITLE}")
    st.caption("市值排序在真实资金约束下还成立吗？——用蒙特卡洛分离能力与运气。")
    st.info(
        "**上一轮结论**：十个候选排序变量里，只有**流通市值（越大越优先）**三关全过——"
        "正向+1.31%/t=1.97、反向-2.02%、四年全部为正且逐年增强、TopN衰减平滑。"
        "而横盘区间宽度、距40周均线全期看着更好，一分年度就露馅（优势全部来自2026一年）。\n\n"
        "**但那是无资金约束下测的。** SKDJ阶段的教训：K值排序在无约束时看着可以，"
        "一加三仓就失效（蒙特卡洛分位仅45%，还不如随机）。\n\n"
        "**所以本轮必须验证**：市值排序在三仓约束下，能否明显跑赢随机选股。"
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
        st.subheader("资金设置")
        slot_count = st.number_input("仓位数", value=3, min_value=1, max_value=20, step=1)
        mc_runs = st.number_input(
            "蒙特卡洛次数", value=200, min_value=20, max_value=1000, step=20
        )
        do_sweep = st.checkbox("同时扫描 1-10 仓（较慢）", value=True)
        cost_pct = st.number_input("往返成本%", value=0.20, min_value=0.0, max_value=2.0, step=0.05)

        st.markdown("---")
        st.caption("信号与排序规则已锁死，不可修改。")
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
        if st.session_state.get("slot_result"):
            return
        st.markdown(
            """
### 冻结的完整策略

```
信号：突破26周新高 + 接近两年高点前33% + 波动率压缩≤0.8
排序：同一周多个信号时，按流通市值从大到小
入场：突破次周开盘买入
退出：移动止损15%，最长持有26周
```

### 四张表

**表1 · 三仓结果 vs 蒙特卡洛**　核心是「优选所处分位」：
- 接近50% → 市值排序在资金约束下没用，和随机一样
- 高于80% → 排序真正起作用，可以定下来
- 「运气区间宽度」反映单次回测结果有多不可信

**表2 · 仓位数扫描**　1到10仓的收益、运气区间、分位。
你只能开3仓，但看趋势能判断3仓是否已经太少。

**表3 · 分年度**　三仓约束下每年买了几笔、抓到几只翻倍股、
最大和最差单笔各是多少——这是你实盘会真实经历的。

**表4 · 逐笔台账**　可下载，看具体买了什么。

---
**判断标准**：分位>80% 且 三仓年度表现不比无约束差太多，
才算这个排序层真的可用。
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
        st.error("缺少市值数据，排序规则依赖它。")
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
                weekly[column] = pd.to_numeric(
                    weekly[column], errors="coerce"
                ).astype("float32")
        weekly_cache[ts_code] = weekly
        all_weeks.update(weekly["trade_date_str"].astype(str).tolist())
        features = compute_features(weekly)
        values = features.loc[features["breakout"].fillna(False), "position_2y"]
        position_samples.append(values.dropna().astype("float32"))
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

    progress = st.progress(0.0, text="生成信号……")
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
    signals = signals[signals["MV_Billion"].between(min_mv, max_mv)]
    signals = signals.reset_index(drop=True)
    if signals.empty:
        st.error("过滤后无信号。")
        return

    week_index = {week: i for i, week in enumerate(sorted(all_weeks))}

    ledger, total = simulate_slots(
        signals, week_index, int(slot_count), float(cost_pct), order_mode="rank"
    )
    mc_progress = st.progress(0.0, text="蒙特卡洛：随机选股重复回测……")
    outcomes = monte_carlo(
        signals, week_index, int(slot_count), float(cost_pct), int(mc_runs),
        progress_callback=lambda p: mc_progress.progress(p),
    )
    mc_progress.empty()

    sweep = pd.DataFrame()
    if do_sweep:
        sweep_progress = st.progress(0.0, text="扫描不同仓位数……")
        sweep = slot_sweep(
            signals, week_index, float(cost_pct), [1, 2, 3, 4, 5, 6, 8, 10],
            max(30, int(mc_runs) // 4),
            progress_callback=lambda p: sweep_progress.progress(p),
        )
        sweep_progress.empty()

    bought = ledger[ledger["执行"] == "买入"] if not ledger.empty else pd.DataFrame()
    net = (
        pd.to_numeric(bought["净收益%"], errors="coerce").dropna()
        if not bought.empty
        else pd.Series(dtype=float)
    )
    gain_all = pd.to_numeric(signals["Return_pct"], errors="coerce") - float(cost_pct)

    summary = pd.DataFrame(
        [
            {
                "口径": "信号层（无资金约束，全部信号）",
                "笔数": int(len(gain_all.dropna())),
                "单笔平均%": float(gain_all.mean()),
                "单笔胜率%": float((gain_all > 0).mean() * 100.0),
                "翻倍股数": int(
                    (pd.to_numeric(signals["Max_Gain_pct"], errors="coerce") > 100).sum()
                ),
                "总收益%": np.nan,
            },
            {
                "口径": f"{int(slot_count)}仓（按市值优选）",
                "笔数": int(len(net)),
                "单笔平均%": float(net.mean()) if len(net) else np.nan,
                "单笔胜率%": float((net > 0).mean() * 100.0) if len(net) else np.nan,
                "翻倍股数": int(
                    (pd.to_numeric(bought["Max_Gain_pct"], errors="coerce") > 100).sum()
                )
                if not bought.empty
                else 0,
                "总收益%": total,
            },
        ]
    )

    mc_table = pd.DataFrame(
        [
            {"指标": "随机选股 最差5%", "总收益率%": float(np.percentile(outcomes, 5))},
            {"指标": "随机选股 中位数", "总收益率%": float(np.median(outcomes))},
            {"指标": "随机选股 最好5%", "总收益率%": float(np.percentile(outcomes, 95))},
            {"指标": "▶ 按市值优选（真实规则）", "总收益率%": total},
        ]
    )

    st.session_state["slot_result"] = {
        "signal_count": len(signals),
        "summary": summary,
        "mc_table": mc_table,
        "percentile": float((outcomes < total).mean() * 100.0),
        "spread": float(np.percentile(outcomes, 95) - np.percentile(outcomes, 5)),
        "sweep": sweep,
        "yearly": yearly_slots(ledger),
        "ledger": ledger,
        "slot_count": int(slot_count),
        "mc_runs": int(mc_runs),
        "period": f"{start_date} — {end_date}",
        "memory_mb": _memory_usage_mb(),
    }


def render_results():
    result = st.session_state.get("slot_result")
    if not result:
        return False

    st.markdown("---")
    st.header("三仓约束下的排序层验证")
    st.caption(
        f"突破信号 {result['signal_count']:,} 个　|　区间 {result['period']}　|　"
        f"{result['slot_count']}仓，蒙特卡洛{result['mc_runs']}次"
        + (
            f"　|　内存 {result['memory_mb']:.0f} MB"
            if math.isfinite(result.get("memory_mb", float("nan")))
            else ""
        )
    )

    st.subheader("表1 · 信号层 vs 三仓层")
    st.dataframe(result["summary"].round(2), width="stretch", hide_index=True)
    st.caption("两者之差 = 资金约束的代价。重点看翻倍股数量被砍掉多少。")

    st.subheader("表2 · 蒙特卡洛：排序规则是真本事还是运气？")
    st.dataframe(result["mc_table"].round(2), width="stretch", hide_index=True)
    st.markdown(
        f"""
**按市值优选的结果落在随机分布的 {result['percentile']:.0f}% 分位**，
随机选股的90%区间宽度为 **{result['spread']:.1f} 个百分点**。

- 分位接近50% → 市值排序在资金约束下没用（K值排序当年就是45%）
- **分位高于80% → 排序真正起作用**
- 区间越宽，说明单次回测结果越不可信，运气成分越大
"""
    )

    if not result["sweep"].empty:
        st.subheader("表3 · 仓位数扫描")
        st.dataframe(result["sweep"].round(2), width="stretch", hide_index=True)
        st.caption(
            "你只能开3仓，但看趋势能判断3仓是否已经太少："
            "如果分位随仓位增加而明显上升、运气区间明显收窄，"
            "说明3仓的噪声仍然偏大，可考虑降低单仓占比而非增加仓位数。"
        )

    if not result["yearly"].empty:
        st.subheader("表4 · 三仓约束下的分年度（你实盘会经历的）")
        st.dataframe(result["yearly"].round(2), width="stretch", hide_index=True)
        st.caption(
            "**这张表最接近实盘感受**：每年买几笔、抓到几只翻倍股、"
            "最大和最差单笔各是多少。注意亏损年份的买入笔数和最差单笔。"
        )

    if not result["ledger"].empty:
        with st.expander("查看逐笔台账"):
            st.dataframe(result["ledger"], width="stretch", hide_index=True)

    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_summary.csv", result["summary"].to_csv(index=False, encoding="utf-8-sig")
        )
        archive.writestr(
            "02_monte_carlo.csv", result["mc_table"].to_csv(index=False, encoding="utf-8-sig")
        )
        archive.writestr(
            "03_slot_sweep.csv", result["sweep"].to_csv(index=False, encoding="utf-8-sig")
        )
        archive.writestr(
            "04_yearly.csv", result["yearly"].to_csv(index=False, encoding="utf-8-sig")
        )
        archive.writestr(
            "05_ledger.csv", result["ledger"].to_csv(index=False, encoding="utf-8-sig")
        )
    st.download_button(
        "下载验证结果",
        data=output.getvalue(),
        file_name="slot_ranking_validation.zip",
        mime="application/zip",
        key="download_slot",
    )
    return True


if __name__ == "__main__":
    main()
    render_results()
