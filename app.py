# -*- coding: utf-8 -*-
"""周线选股 + 日线买点 验证器（单文件独立版，直接覆盖 app.py 运行）。

验证一个此前从未测过、但你最初就提出的思路：
    周线SKDJ出信号后，用日线判断具体买入时机，而不是下周一开盘无脑买入。

已确立的前提（前几轮验证结论）：
- 周线SKDJ低位拐头信号有真实横截面优势，且是主因子（2x2析因：净贡献+1.06~+2.24%）
- 26周回撤只在SKDJ信号内部有效（无SKDJ时净贡献-0.15%），是附属排序工具
- 但整体信噪比偏低，相对随机买入的超额仅约+0.5%/笔

日线择时可能带来两类价值：
  ① 择价：拿到更好的成交价
  ② 筛选：周线出信号但日线迟迟不确认的，直接放弃——这层筛选比择价更有意义

测试口径（严格避免未来函数）：
- 确认统一为「当日收盘满足条件 -> 次日开盘买入」
- 回踩买入按限价单模拟：盘中触及目标价才成交，开盘已跳空低于目标价则按开盘价
- 所有入场方式共用完全相同的退出时点（第N周最后交易日收盘），
  因此收益差异只反映入场时机，不受持有时间不同影响
- 与基准的比较采用「同一批能成交的交易」配对比较，避免样本不同造成误判

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

APP_TITLE = "周线选股 + 日线买点 验证"
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
def add_skdj(weekly: pd.DataFrame, n_period: int, m_period: int) -> pd.DataFrame:
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


def prepare_daily_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """日线指标，只用于入场确认。全部基于当日及之前数据，无未来函数。"""
    frame = daily.copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = (
        pd.to_numeric(frame["vol"], errors="coerce")
        if "vol" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )

    frame["d_ma5"] = close.rolling(5).mean()
    frame["d_vol_ma5"] = volume.shift(1).rolling(5).mean()
    frame["d_vol_ratio"] = volume / frame["d_vol_ma5"].replace(0, np.nan)

    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100.0
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    frame["d_k"] = k
    frame["d_d"] = d
    frame["d_k_prev"] = k.shift(1)
    frame["d_d_prev"] = d.shift(1)
    frame["d_close_prev"] = close.shift(1)
    return frame


def daily_entry_variants(
    daily_window: pd.DataFrame, signal_close: float, pullback_pct: float
):
    """返回各种日线确认方式下的 (成交日, 成交价)；无法确认则为 (None, nan)。

    统一口径：当日收盘满足条件 -> 次日开盘买入。这样不存在用当日收盘价
    成交的未来函数问题。窗口内始终无法确认则放弃该笔交易。
    """
    results = {}
    if daily_window.empty:
        return results

    dates = daily_window.index.tolist()
    opens = pd.to_numeric(daily_window["open"], errors="coerce")
    closes = pd.to_numeric(daily_window["close"], errors="coerce")
    lows = pd.to_numeric(daily_window["low"], errors="coerce")

    # ① 基准：下周第一个交易日开盘直接买入
    first_open = _safe_float(opens.iloc[0]) if len(opens) else np.nan
    results["①立即买入（基准）"] = (
        (dates[0], first_open) if math.isfinite(first_open) else (None, np.nan)
    )

    def confirm_then_next_open(mask):
        """当日满足条件，次日开盘买入。"""
        for position in range(len(dates) - 1):
            if bool(mask.iloc[position]):
                price = _safe_float(opens.iloc[position + 1])
                if math.isfinite(price):
                    return dates[position + 1], price
        return None, np.nan

    # ② 收盘站上5日线
    ma5 = pd.to_numeric(daily_window["d_ma5"], errors="coerce")
    results["②收盘站上5日线"] = confirm_then_next_open(closes >= ma5)

    # ③ 日线KDJ金叉
    k = pd.to_numeric(daily_window["d_k"], errors="coerce")
    d_line = pd.to_numeric(daily_window["d_d"], errors="coerce")
    k_prev = pd.to_numeric(daily_window["d_k_prev"], errors="coerce")
    d_prev = pd.to_numeric(daily_window["d_d_prev"], errors="coerce")
    results["③日线KDJ金叉"] = confirm_then_next_open((k > d_line) & (k_prev <= d_prev))

    # ④ 放量阳线
    close_prev = pd.to_numeric(daily_window["d_close_prev"], errors="coerce")
    vol_ratio = pd.to_numeric(daily_window["d_vol_ratio"], errors="coerce")
    results["④放量阳线"] = confirm_then_next_open(
        (closes > close_prev) & (vol_ratio >= 1.2)
    )

    # ⑤ 回踩后买入：盘中触及信号周收盘价下方 pullback_pct，则以该限价成交
    target = signal_close * (1.0 - pullback_pct / 100.0)
    for position in range(len(dates)):
        day_low = _safe_float(lows.iloc[position])
        day_open = _safe_float(opens.iloc[position])
        if math.isfinite(day_low) and day_low <= target:
            # 开盘已低于目标价则按开盘价，否则按目标价成交
            fill = day_open if math.isfinite(day_open) and day_open < target else target
            results[f"⑤回踩{pullback_pct:.0f}%买入"] = (dates[position], fill)
            break
    else:
        results[f"⑤回踩{pullback_pct:.0f}%买入"] = (None, np.nan)

    return results


def build_signals_with_daily_entry(
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    ts_code: str,
    n_period: int,
    m_period: int,
    level: float,
    require_kd: bool,
    hold_weeks: int,
    pullback_pct: float,
) -> pd.DataFrame:
    """周线出信号，日线定买点；退出统一固定在第 hold_weeks 周最后一个交易日收盘。

    退出时点对所有入场方式完全相同，这样收益差异只反映"入场时机"，
    不会被"持有时间不同"污染。
    """
    weekly = add_skdj(weekly, n_period, m_period)
    if len(weekly) < 30:
        return pd.DataFrame()

    def column_series(frame: pd.DataFrame, name: str) -> pd.Series:
        """取列并转数值；列不存在时返回等长的全NaN序列。

        直接用 frame.get(name, np.nan) 在列缺失时会返回标量，
        后续 .shift()/.rolling() 就会抛 AttributeError。
        """
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    k_now = pd.to_numeric(weekly["K"], errors="coerce")
    k_prev = k_now.shift(1)
    d_now = pd.to_numeric(weekly["D"], errors="coerce")
    close_w = pd.to_numeric(weekly["close"], errors="coerce")
    high_w = pd.to_numeric(weekly["high"], errors="coerce")
    volume_w = column_series(weekly, "vol")
    turnover_w = column_series(weekly, "turnover_rate")
    amount_w = column_series(weekly, "amount")

    signal = (k_prev <= k_now) & (k_now <= level)
    if require_kd:
        signal = signal & (k_now > d_now)
    signal = signal.fillna(False)

    # 周线附加因子（全部只用信号周及之前数据）
    drawdown_26 = (close_w / high_w.rolling(26).max() - 1.0) * 100.0
    vol_surge = volume_w / volume_w.shift(1).rolling(8).mean().replace(0, np.nan)
    turnover_change = (
        turnover_w / turnover_w.shift(1).rolling(8).mean().replace(0, np.nan)
    )
    position_52w = (
        (close_w - close_w.rolling(52).min())
        / (close_w.rolling(52).max() - close_w.rolling(52).min()).replace(0, np.nan)
    )
    return_13w = (close_w / close_w.shift(13) - 1.0) * 100.0

    daily_ready = prepare_daily_indicators(daily)
    daily_index = daily_ready.index

    week_dates = weekly["trade_date_str"].astype(str).tolist()
    rows = []
    for i in range(len(weekly)):
        if not bool(signal.iloc[i]):
            continue
        if i + hold_weeks >= len(weekly):
            continue
        week_end = week_dates[i]
        next_week_end = week_dates[i + 1]
        exit_date = week_dates[i + hold_weeks]

        window_mask = (daily_index > week_end) & (daily_index <= next_week_end)
        window = daily_ready.loc[window_mask]
        if window.empty:
            continue
        exit_price = _safe_float(close_w.iloc[i + hold_weeks])
        if not math.isfinite(exit_price) or exit_price <= 0:
            continue

        signal_close = _safe_float(close_w.iloc[i])
        variants = daily_entry_variants(window, signal_close, pullback_pct)

        base = {
            "ts_code": ts_code,
            "Signal_Week": week_end,
            "Entry_Week": next_week_end,
            "Exit_Week": exit_date,
            "K": _safe_float(k_now.iloc[i]),
            "Drawdown_26W_pct": _safe_float(drawdown_26.iloc[i]),
            "Vol_Surge": _safe_float(vol_surge.iloc[i]),
            "Turnover_Change": _safe_float(turnover_change.iloc[i]),
            "Position_52W": _safe_float(position_52w.iloc[i]),
            "Return_13W_pct": _safe_float(return_13w.iloc[i]),
            "Amount_W": _safe_float(amount_w.iloc[i]),
            "Exit_Close": exit_price,
        }
        for label, (entry_date, entry_price) in variants.items():
            base[f"入场_{label}_日期"] = entry_date
            base[f"入场_{label}_价格"] = entry_price
            base[f"收益_{label}"] = (
                (exit_price / entry_price - 1.0) * 100.0
                if math.isfinite(_safe_float(entry_price)) and entry_price > 0
                else np.nan
            )
        rows.append(base)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 分析
# -----------------------------------------------------------------------------
def compare_entry_methods(signals: pd.DataFrame, cost_pct: float):
    """对比各日线入场方式：同样的退出时点，只有入场不同。"""
    labels = [
        column[3:] for column in signals.columns if column.startswith("收益_")
    ]
    rows = []
    baseline_label = next((x for x in labels if x.startswith("①")), None)
    base_returns = (
        pd.to_numeric(signals[f"收益_{baseline_label}"], errors="coerce")
        if baseline_label
        else pd.Series(dtype=float)
    )
    for label in labels:
        returns = pd.to_numeric(signals[f"收益_{label}"], errors="coerce")
        traded = returns.notna()
        net = returns[traded] - cost_pct
        # 与基准在"同一批能成交的交易"上对比，避免样本不同造成误判
        both = traded & base_returns.notna()
        paired_diff = (returns[both] - base_returns[both]).dropna()
        if len(paired_diff) > 1 and paired_diff.std(ddof=1) > 0:
            t_stat = paired_diff.mean() / (
                paired_diff.std(ddof=1) / math.sqrt(len(paired_diff))
            )
        else:
            t_stat = np.nan
        rows.append(
            {
                "入场方式": label,
                "成交笔数": int(traded.sum()),
                "成交率%": float(traded.mean() * 100.0),
                "平均收益%": float(net.mean()) if len(net) else np.nan,
                "中位收益%": float(net.median()) if len(net) else np.nan,
                "胜率%": float((net > 0).mean() * 100.0) if len(net) else np.nan,
                "标准差%": float(net.std(ddof=1)) if len(net) > 1 else np.nan,
                "vs基准差值%": float(paired_diff.mean()) if len(paired_diff) else np.nan,
                "差值t值": t_stat,
            }
        )
    return pd.DataFrame(rows)


def test_weekly_factors(signals: pd.DataFrame, top_n: int, cost_pct: float, label: str):
    """周线附加因子的周内选股能力检验（含反向对照）。"""
    if signals.empty or f"收益_{label}" not in signals.columns:
        return pd.DataFrame()
    work = signals.copy()
    work["_ret"] = pd.to_numeric(work[f"收益_{label}"], errors="coerce") - cost_pct
    work = work.dropna(subset=["_ret"])
    if work.empty:
        return pd.DataFrame()
    baseline = work.groupby("Entry_Week")["_ret"].mean()

    candidates = [
        ("Drawdown_26W_pct", "26周回撤最深优先", True),
        ("Vol_Surge", "成交量放大优先", False),
        ("Turnover_Change", "换手率放大优先", False),
        ("Position_52W", "52周位置最低优先", True),
        ("Return_13W_pct", "13周涨幅最高优先", False),
        ("Amount_W", "成交额最大优先", False),
        ("K", "K值最低优先", True),
    ]
    rows = []
    for column, name, ascending in candidates:
        if column not in work.columns:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        if values.notna().sum() < len(work) * 0.5:
            continue
        subset = work.assign(_v=values).dropna(subset=["_v"])
        rank_fwd = subset.groupby("Entry_Week")["_v"].rank(
            method="first", ascending=ascending
        )
        rank_rev = subset.groupby("Entry_Week")["_v"].rank(
            method="first", ascending=not ascending
        )
        top = subset[rank_fwd <= top_n].groupby("Entry_Week")["_ret"].mean()
        bottom = subset[rank_rev <= top_n].groupby("Entry_Week")["_ret"].mean()
        edge = (top - baseline.reindex(top.index)).dropna()
        edge_rev = (bottom - baseline.reindex(bottom.index)).dropna()
        if len(edge) > 1 and edge.std(ddof=1) > 0:
            t_stat = edge.mean() / (edge.std(ddof=1) / math.sqrt(len(edge)))
        else:
            t_stat = np.nan
        rows.append(
            {
                "附加因子": name,
                f"每周Top{top_n}收益%": float(top.mean()),
                "同周全部信号%": float(baseline.reindex(top.index).mean()),
                "选股超额%": float(edge.mean()),
                "反向对照%": float(edge_rev.mean()) if len(edge_rev) else np.nan,
                "有效周数": int(len(edge)),
                "粗略t值": t_stat,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("选股超额%", ascending=False).reset_index(drop=True)
    return result


def combo_test(signals: pd.DataFrame, top_n: int, cost_pct: float, label: str):
    """把日线确认与周线排序组合起来，看叠加后还剩多少优势。"""
    labels = [c[3:] for c in signals.columns if c.startswith("收益_")]
    rows = []
    for entry_label in labels:
        work = signals.copy()
        work["_ret"] = pd.to_numeric(work[f"收益_{entry_label}"], errors="coerce") - cost_pct
        work = work.dropna(subset=["_ret", "Drawdown_26W_pct"])
        if work.empty:
            continue
        rank = work.groupby("Entry_Week")["Drawdown_26W_pct"].rank(
            method="first", ascending=True
        )
        top = work[rank <= top_n]
        by_week = top.groupby("Entry_Week")["_ret"].mean()
        if by_week.empty:
            continue
        if len(by_week) > 1 and by_week.std(ddof=1) > 0:
            t_stat = by_week.mean() / (by_week.std(ddof=1) / math.sqrt(len(by_week)))
        else:
            t_stat = np.nan
        rows.append(
            {
                "组合方案": f"{entry_label} + 回撤最深Top{top_n}",
                "可交易周数": int(len(by_week)),
                "每周收益%": float(by_week.mean()),
                "周胜率%": float((by_week > 0).mean() * 100.0),
                "周标准差%": float(by_week.std(ddof=1)) if len(by_week) > 1 else np.nan,
                "收益/波动": (
                    by_week.mean() / by_week.std(ddof=1)
                    if len(by_week) > 1 and by_week.std(ddof=1) > 0
                    else np.nan
                ),
                "粗略t值": t_stat,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"📈 {APP_TITLE}")
    st.caption("周线SKDJ出信号，日线找买点——你最初的思路，这次正式验证。")
    st.info(
        "**为什么测这个**：此前所有回测都是「周线出信号→下周一开盘无脑买入」，"
        "从未验证过日线择时。日线确认可能带来两类价值："
        "①拿到更好的成交价；②过滤掉周线出了信号但日线迟迟不确认的假信号。"
        "第②类尤其重要——那不只是买便宜点，而是多了一层筛选。\n\n"
        "**测法**：所有入场方式共用**完全相同的退出时点**，"
        "因此收益差异只反映入场时机，不会被持有时间不同污染。"
        "确认口径统一为「当日收盘满足条件→次日开盘买入」，不存在未来函数。"
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
        st.subheader("周线信号（已验证的最优配置）")
        n_period = st.number_input("N", value=4, min_value=2, max_value=60, step=1)
        m_period = st.number_input("M", value=3, min_value=2, max_value=30, step=1)
        level = st.number_input("K值阈值", value=20.0, min_value=1.0, max_value=90.0, step=5.0)
        require_kd = st.checkbox("要求 K > D", value=True)
        hold_weeks = st.number_input("持有周数", value=3, min_value=1, max_value=8, step=1)

        st.markdown("---")
        st.subheader("日线入场")
        pullback_pct = st.number_input(
            "回踩买入：低于信号周收盘价百分之几",
            value=3.0, min_value=0.5, max_value=15.0, step=0.5,
        )
        top_n = st.number_input("每周选几只", value=3, min_value=1, max_value=20, step=1)
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
        if st.session_state.get("entry_result"):
            return
        st.markdown(
            """
### 五种入场方式的对比

| 方式 | 说明 |
|---|---|
| ①立即买入 | 下周第一个交易日开盘买入（现在的做法，基准） |
| ②收盘站上5日线 | 等日线收盘站上MA5，次日开盘买 |
| ③日线KDJ金叉 | 等日线KDJ(9,3,3)金叉，次日开盘买 |
| ④放量阳线 | 等一根放量上涨的日K，次日开盘买 |
| ⑤回踩买入 | 挂低于信号周收盘价N%的限价单，触及才成交 |

②③④如果在下一周内始终没确认，就**放弃这笔交易**——这是筛选效应，
会体现在「成交率」那一列。

### 三张表

**表1 · 入场方式对比**　同一批交易配对比较，看「vs基准差值」和t值。

**表2 · 周线附加因子**　除回撤外，再测量能、换手、52周位置、成交额等
是否有周内选股能力（全部带反向对照）。

**表3 · 组合效果**　把日线确认和回撤排序叠加，看最终还剩多少优势。
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
    fetch_start = (pd.Timestamp(start_input) - timedelta(days=450)).strftime("%Y%m%d")
    fetch_end = (pd.Timestamp(end_input) + timedelta(days=90)).strftime("%Y%m%d")

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

    progress = st.progress(0.0, text="周线出信号 + 日线定买点……")
    parts = []
    codes = sorted(stocks.keys())
    for idx, ts_code in enumerate(codes):
        daily = stocks[ts_code]
        weekly = build_weekly_bars(daily)
        if weekly.empty or len(weekly) < 60:
            continue
        rows = build_signals_with_daily_entry(
            weekly, daily, ts_code, int(n_period), int(m_period),
            float(level), bool(require_kd), int(hold_weeks), float(pullback_pct),
        )
        if not rows.empty:
            parts.append(rows)
        if idx % 40 == 0:
            progress.progress(
                min((idx + 1) / len(codes), 1.0),
                text=f"周线出信号 + 日线定买点……{idx + 1}/{len(codes)}",
            )
    progress.empty()
    del stocks
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
            mv = pd.to_numeric(signals["circ_mv"], errors="coerce") / 10000.0
            signals = signals[mv.between(min_mv, max_mv) | mv.isna()]
    base_col = next(
        (c for c in signals.columns if c.startswith("入场_①") and c.endswith("价格")), None
    )
    if base_col:
        signals = signals[pd.to_numeric(signals[base_col], errors="coerce") >= min_price]
    signals = signals.reset_index(drop=True)
    if signals.empty:
        st.error("过滤后无信号。")
        return

    entry_table = compare_entry_methods(signals, float(cost_pct))
    baseline_label = next(
        (c[3:] for c in signals.columns if c.startswith("收益_①")), None
    )
    factor_table = test_weekly_factors(
        signals, int(top_n), float(cost_pct), baseline_label
    )
    combo_table = combo_test(signals, int(top_n), float(cost_pct), baseline_label)

    signals["年份"] = signals["Signal_Week"].astype(str).str[:4]
    yearly_rows = []
    for year, group in signals.groupby("年份"):
        row = {"年份": year, "信号数": len(group)}
        for column in group.columns:
            if column.startswith("收益_"):
                row[column[3:]] = float(
                    pd.to_numeric(group[column], errors="coerce").mean() - cost_pct
                )
        yearly_rows.append(row)
    yearly_table = pd.DataFrame(yearly_rows)

    st.session_state["entry_result"] = {
        "signals": signals,
        "entry_table": entry_table,
        "factor_table": factor_table,
        "combo_table": combo_table,
        "yearly_table": yearly_table,
        "params": {
            "N": int(n_period), "M": int(m_period), "阈值": float(level),
            "K>D": bool(require_kd), "持有": int(hold_weeks),
            "回踩%": float(pullback_pct), "每周选": int(top_n),
        },
    }


def render_results():
    result = st.session_state.get("entry_result")
    if not result:
        return False
    params = result["params"]
    signals = result["signals"]

    st.markdown("---")
    st.header("验证结果")
    st.caption(
        f"信号 {len(signals):,} 笔，覆盖 {signals['Signal_Week'].min()} — "
        f"{signals['Signal_Week'].max()}　|　"
        f"SKDJ: N={params['N']} M={params['M']} K≤{params['阈值']:.0f}"
        f"{' 且K>D' if params['K>D'] else ''}　持有{params['持有']}周"
    )

    st.subheader("表1 · 日线入场方式对比")
    st.dataframe(result["entry_table"].round(3), width="stretch", hide_index=True)
    st.caption(
        "**重点看两列**：「vs基准差值%」是同一批交易的配对比较（比单纯比平均值更可靠），"
        "「成交率%」反映筛选强度。\n\n"
        "- 差值明显为正且 |t|>2 → 日线择时确实有用\n"
        "- 差值接近0 → 日线择时只是换个价格进场，不创造优势\n"
        "- 成交率低但收益高 → 价值来自**过滤**而非**择价**，这种更有意义"
    )

    if not result["factor_table"].empty:
        st.subheader("表2 · 周线附加因子（周内选股能力）")
        st.dataframe(result["factor_table"].round(3), width="stretch", hide_index=True)
        st.caption(
            "同样带反向对照：**正向超额为正、反向为负、|t|>2** 三条都满足才可信。"
            "只有正向为正而反向也为正的，是噪声。"
        )

    if not result["combo_table"].empty:
        st.subheader("表3 · 日线确认 + 回撤排序 的组合效果")
        st.dataframe(result["combo_table"].round(3), width="stretch", hide_index=True)
        st.caption(
            "「收益/波动」是风险调整后的比较，比单看收益率更能说明问题。"
            "可交易周数也要一起看——周数太少同样没有实操性。"
        )

    if not result["yearly_table"].empty:
        st.subheader("表4 · 各入场方式分年度表现")
        st.dataframe(result["yearly_table"].round(2), width="stretch", hide_index=True)
        st.caption("看某种入场方式是否每年都优于基准，还是只靠某一年。")

    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_entry_methods.csv",
            result["entry_table"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "02_weekly_factors.csv",
            result["factor_table"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "03_combo.csv",
            result["combo_table"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "04_yearly.csv",
            result["yearly_table"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "05_all_signals.csv",
            signals.to_csv(index=False, encoding="utf-8-sig"),
        )
    st.download_button(
        "下载验证结果",
        data=output.getvalue(),
        file_name="daily_entry_validation.zip",
        mime="application/zip",
        key="download_entry",
    )
    return True


if __name__ == "__main__":
    main()
    render_results()
