# -*- coding: utf-8 -*-
"""SKDJ vs 26周回撤 因子分解验证器（单文件独立版，直接覆盖 app.py 运行）。

回答一个问题：此前找到的选股优势，到底来自SKDJ信号，还是来自"跌得深"？
如果来自后者，SKDJ就可以扔掉——那样信号更多、逻辑更简单、实操性更好。

背景：
- SKDJ低位拐头信号本身有横截面超额（+1.34%/3周，t=8.9）
- 但周内用K值排序选股完全无效（超额-0.05%，t=-0.21）
- 真正有效的周内排序是26周回撤（超额+0.99%，t=2.51，五分组单调，反向对照为负）
- 所以必须搞清楚：SKDJ是否贡献了独立于"跌得深"的信息

方法：2x2析因设计。把"有无SKDJ信号"与"是否深度回撤"作为两个独立因子，
在控制另一因子不变的前提下，测量各自的净贡献。这比单独看任一方案的
总收益更能分辨因果。

对比全部采用横截面口径（每周选N只算平均，再对周求平均），不模拟仓位调度，
以剔除三仓路径依赖带来的巨大噪声（此前实测运气区间宽达80个百分点）。

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

APP_TITLE = "SKDJ vs 回撤 因子分解验证"
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
def add_skdj(weekly: pd.DataFrame, n_period: int, m_period: int) -> pd.DataFrame:
    """通达信 SKDJ：
        LOWV:=LLV(LOW,N); HIGHV:=HHV(HIGH,N);
        RSV:=EMA((CLOSE-LOWV)/(HIGHV-LOWV)*100,M); K:EMA(RSV,M); D:MA(K,M);
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


def build_panel_for_stock(
    weekly: pd.DataFrame, ts_code: str, n_period: int, m_period: int,
    level: float, require_kd: bool, hold_weeks: int,
) -> pd.DataFrame:
    """构建"全部个股-周"面板：不只是信号周，而是每一周都留下。

    这样才能做2x2析因：需要知道"有SKDJ信号"和"没有SKDJ信号"两组的表现，
    只保留信号周是无法做对照的。
    """
    weekly = add_skdj(weekly, n_period, m_period)
    close = pd.to_numeric(weekly["close"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    open_price = pd.to_numeric(weekly["open"], errors="coerce")
    dates = weekly["trade_date_str"].astype(str)

    k_now = pd.to_numeric(weekly["K"], errors="coerce")
    k_prev = k_now.shift(1)
    d_now = pd.to_numeric(weekly["D"], errors="coerce")

    skdj_signal = (k_prev <= k_now) & (k_now <= level)
    if require_kd:
        skdj_signal = skdj_signal & (k_now > d_now)

    entry_open = open_price.shift(-1)
    exit_close = close.shift(-hold_weeks)

    frame = pd.DataFrame(
        {
            "ts_code": ts_code,
            "Signal_Week": dates,
            "Entry_Week": dates.shift(-1),
            "K": k_now,
            "D": d_now,
            "SKDJ_Signal": skdj_signal.fillna(False),
            "Drawdown_26W_pct": (close / high.rolling(26).max() - 1.0) * 100.0,
            "Entry_Open": entry_open,
            "Fwd_Return_pct": (
                exit_close / entry_open.replace(0, np.nan) - 1.0
            ) * 100.0,
        }
    )
    return frame.dropna(subset=["Entry_Week", "Fwd_Return_pct", "Drawdown_26W_pct"])


# -----------------------------------------------------------------------------
# 分解分析
# -----------------------------------------------------------------------------
def factorial_2x2(panel: pd.DataFrame, deep_quantile: float, cost_pct: float):
    """2x2析因：把SKDJ信号和深度回撤当作两个独立因子，看各自的净贡献。

    这是判断"SKDJ该不该留"的核心测试。如果在同样是深度回撤的股票里，
    有没有SKDJ信号的表现差不多，那SKDJ就是多余的。
    """
    work = panel.copy()
    work["_ret"] = pd.to_numeric(work["Fwd_Return_pct"], errors="coerce") - cost_pct
    work = work.dropna(subset=["_ret"])
    # 每周横截面上定义"深度回撤"：回撤最深的 deep_quantile 部分
    work["_dd_rank"] = work.groupby("Entry_Week")["Drawdown_26W_pct"].rank(pct=True)
    work["深度回撤"] = work["_dd_rank"] <= deep_quantile
    work["SKDJ信号"] = work["SKDJ_Signal"].astype(bool)

    rows = []
    for dd in (True, False):
        for sk in (True, False):
            cell = work[(work["深度回撤"] == dd) & (work["SKDJ信号"] == sk)]["_ret"]
            rows.append(
                {
                    "深度回撤": "是" if dd else "否",
                    "SKDJ信号": "有" if sk else "无",
                    "样本数": int(len(cell)),
                    "平均收益%": float(cell.mean()) if len(cell) else np.nan,
                    "中位收益%": float(cell.median()) if len(cell) else np.nan,
                    "胜率%": float((cell > 0).mean() * 100.0) if len(cell) else np.nan,
                }
            )
    table = pd.DataFrame(rows)

    def cell_mean(dd, sk):
        row = table[(table["深度回撤"] == dd) & (table["SKDJ信号"] == sk)]
        return float(row["平均收益%"].iloc[0]) if len(row) else np.nan

    contrib = pd.DataFrame(
        [
            {
                "对比": "SKDJ的独立贡献（在深度回撤股票内部）",
                "有该因子%": cell_mean("是", "有"),
                "无该因子%": cell_mean("是", "无"),
                "净贡献%": cell_mean("是", "有") - cell_mean("是", "无"),
            },
            {
                "对比": "SKDJ的独立贡献（在非深度回撤股票内部）",
                "有该因子%": cell_mean("否", "有"),
                "无该因子%": cell_mean("否", "无"),
                "净贡献%": cell_mean("否", "有") - cell_mean("否", "无"),
            },
            {
                "对比": "深度回撤的独立贡献（在有SKDJ信号内部）",
                "有该因子%": cell_mean("是", "有"),
                "无该因子%": cell_mean("否", "有"),
                "净贡献%": cell_mean("是", "有") - cell_mean("否", "有"),
            },
            {
                "对比": "深度回撤的独立贡献（在无SKDJ信号内部）",
                "有该因子%": cell_mean("是", "无"),
                "无该因子%": cell_mean("否", "无"),
                "净贡献%": cell_mean("是", "无") - cell_mean("否", "无"),
            },
        ]
    )
    return table, contrib


def compare_selection_schemes(
    panel: pd.DataFrame, top_n: int, cost_pct: float, random_draws: int = 50
):
    """四种选股方案的正面对比（每周选top_n只，算平均收益，再对周求平均）。

    用横截面方式对比而不是模拟三仓，是为了剔除仓位路径依赖带来的噪声，
    让"选股方法本身"的差异干净地显现出来。
    """
    work = panel.copy()
    work["_ret"] = pd.to_numeric(work["Fwd_Return_pct"], errors="coerce") - cost_pct
    work = work.dropna(subset=["_ret"])
    rng = np.random.default_rng(20240101)

    def weekly_mean_of_top(subset: pd.DataFrame, sort_col: str | None):
        if subset.empty:
            return pd.Series(dtype=float)
        if sort_col is None:
            picks = []
            for _, group in subset.groupby("Entry_Week"):
                take = min(top_n, len(group))
                vals = group["_ret"].to_numpy()
                draws = [
                    rng.choice(vals, size=take, replace=False).mean()
                    for _ in range(random_draws)
                ]
                picks.append((group["Entry_Week"].iloc[0], float(np.mean(draws))))
            return pd.Series(dict(picks))
        ranked = subset.groupby("Entry_Week")[sort_col].rank(
            method="first", ascending=True
        )
        return subset[ranked <= top_n].groupby("Entry_Week")["_ret"].mean()

    skdj_only = work[work["SKDJ_Signal"].astype(bool)]

    schemes = [
        ("① SKDJ信号 + 回撤最深（当前策略）", skdj_only, "Drawdown_26W_pct"),
        ("② 全池 + 回撤最深（不用SKDJ）", work, "Drawdown_26W_pct"),
        ("③ SKDJ信号 + 随机取（只靠SKDJ）", skdj_only, None),
        ("④ 全池随机取（基准）", work, None),
    ]
    rows = []
    for label, subset, sort_col in schemes:
        weekly_returns = weekly_mean_of_top(subset, sort_col)
        weekly_returns = weekly_returns.dropna()
        if weekly_returns.empty:
            continue
        if len(weekly_returns) > 1 and weekly_returns.std(ddof=1) > 0:
            t_stat = weekly_returns.mean() / (
                weekly_returns.std(ddof=1) / math.sqrt(len(weekly_returns))
            )
        else:
            t_stat = np.nan
        rows.append(
            {
                "选股方案": label,
                "可交易周数": int(len(weekly_returns)),
                f"每周Top{top_n}平均收益%": float(weekly_returns.mean()),
                "周胜率%": float((weekly_returns > 0).mean() * 100.0),
                "周收益标准差%": float(weekly_returns.std(ddof=1)),
                "粗略t值": t_stat,
            }
        )
    return pd.DataFrame(rows)


def yearly_scheme_comparison(panel: pd.DataFrame, top_n: int, cost_pct: float):
    """两个主要方案的分年度对比：用SKDJ vs 不用SKDJ。"""
    work = panel.copy()
    work["_ret"] = pd.to_numeric(work["Fwd_Return_pct"], errors="coerce") - cost_pct
    work = work.dropna(subset=["_ret"])
    work["年份"] = work["Entry_Week"].astype(str).str[:4]

    def top_by_week(subset):
        ranked = subset.groupby("Entry_Week")["Drawdown_26W_pct"].rank(
            method="first", ascending=True
        )
        return subset[ranked <= top_n]

    rows = []
    for year, group in work.groupby("年份"):
        with_skdj = top_by_week(group[group["SKDJ_Signal"].astype(bool)])
        without_skdj = top_by_week(group)
        rows.append(
            {
                "年份": year,
                "①用SKDJ 周数": with_skdj["Entry_Week"].nunique(),
                "①用SKDJ 收益%": float(
                    with_skdj.groupby("Entry_Week")["_ret"].mean().mean()
                ) if not with_skdj.empty else np.nan,
                "②不用SKDJ 周数": without_skdj["Entry_Week"].nunique(),
                "②不用SKDJ 收益%": float(
                    without_skdj.groupby("Entry_Week")["_ret"].mean().mean()
                ) if not without_skdj.empty else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["差值(①-②)%"] = result["①用SKDJ 收益%"] - result["②不用SKDJ 收益%"]
    return result


# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🔬 {APP_TITLE}")
    st.caption(
        "回答一个问题：优势到底来自SKDJ，还是来自「跌得深」？SKDJ该不该保留？"
    )
    st.info(
        "**背景**：此前检验发现，真正稳定有效的是「26周回撤越深越好」"
        "（同周五分组单调、跨年稳定、反向对照为负），而SKDJ的K值排序完全无效。"
        "所以必须搞清楚：SKDJ是在贡献独立信息，还是只是个噪声筛子。\n\n"
        "**方法**：2x2析因设计。把「有无SKDJ信号」和「是否深度回撤」当作两个独立因子，"
        "看在**控制另一个因子不变**的前提下，各自还能带来多少收益差异。"
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
        st.subheader("SKDJ 信号定义")
        n_period = st.number_input("N", value=4, min_value=2, max_value=60, step=1)
        m_period = st.number_input("M", value=3, min_value=2, max_value=30, step=1)
        level = st.number_input("K值阈值", value=20.0, min_value=1.0, max_value=90.0, step=5.0)
        require_kd = st.checkbox("要求 K > D", value=True)

        st.markdown("---")
        st.subheader("对比设置")
        hold_weeks = st.number_input("持有周数", value=3, min_value=1, max_value=8, step=1)
        top_n = st.number_input("每周选几只", value=3, min_value=1, max_value=20, step=1)
        deep_quantile = st.slider(
            "「深度回撤」定义：每周回撤最深的百分之几",
            min_value=0.05, max_value=0.50, value=0.20, step=0.05,
        )
        cost_pct = st.number_input("往返成本%", value=0.20, min_value=0.0, max_value=2.0, step=0.05)

        st.markdown("---")
        st.subheader("股票池硬条件")
        min_price = st.number_input("最低股价（元）", value=10.0, min_value=0.0, step=1.0)
        min_mv = st.number_input("最低流通市值（亿元）", value=100.0, min_value=0.0, step=10.0)
        max_mv = st.number_input("最高流通市值（亿元）", value=1000.0, min_value=100.0, step=100.0)

        st.markdown("---")
        clear_cache_clicked = st.button("清空行情缓存")
        run_clicked = st.button("开始分解验证", type="primary")

    if clear_cache_clicked:
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.success("行情缓存已清空。")

    if not run_clicked:
        if st.session_state.get("decomp_result"):
            return
        st.markdown(
            """
### 三张表会告诉你什么

**表1 · 2x2析因**  
四个格子：深回撤×有SKDJ、深回撤×无SKDJ、浅回撤×有SKDJ、浅回撤×无SKDJ。
关键看**在同样是深度回撤的股票里，有没有SKDJ信号的收益差多少**。

**表2 · 因子净贡献**  
把表1换算成"控制另一因子后，每个因子还值多少"。
- SKDJ净贡献接近0 → **可以扔掉SKDJ**，直接用回撤选股，信号更多、更简单
- SKDJ净贡献明显为正 → SKDJ有独立价值，保留

**表3 · 四种选股方案正面对比**  
① SKDJ+回撤（当前策略）② 只用回撤 ③ 只用SKDJ ④ 纯随机基准。
直接比谁的每周Top3收益更高、可交易周数更多。

---
用横截面方式对比（每周选N只算平均），不模拟仓位调度——
这样能剔除路径依赖噪声，让选股方法本身的差异干净显现。
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
        st.error("未取得科技股研究池。")
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

    progress = st.progress(0.0, text="构建全池面板……")
    parts = []
    codes = sorted(stocks.keys())
    for idx, ts_code in enumerate(codes):
        weekly = build_weekly_bars(stocks[ts_code])
        if weekly.empty or len(weekly) < 26 + int(hold_weeks) + 6:
            continue
        parts.append(
            build_panel_for_stock(
                weekly, ts_code, int(n_period), int(m_period),
                float(level), bool(require_kd), int(hold_weeks),
            )
        )
        if idx % 50 == 0:
            progress.progress(
                min((idx + 1) / len(codes), 1.0),
                text=f"构建全池面板……{idx + 1}/{len(codes)}",
            )
    progress.empty()
    del stocks
    gc.collect()

    if not parts:
        st.error("没有足够数据。")
        return
    panel = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    panel = panel[
        (panel["Signal_Week"] >= start_date) & (panel["Signal_Week"] <= end_date)
    ]
    panel = panel[pd.to_numeric(panel["Entry_Open"], errors="coerce") >= min_price]
    if not basic_indexed.empty:
        basic_reset = basic_indexed.reset_index().rename(
            columns={"trade_date_str": "Signal_Week"}
        )
        keep = [c for c in ("Signal_Week", "ts_code", "circ_mv") if c in basic_reset.columns]
        if len(keep) == 3:
            panel = panel.merge(
                basic_reset[keep].drop_duplicates(["Signal_Week", "ts_code"]),
                on=["Signal_Week", "ts_code"], how="left",
            )
            mv = pd.to_numeric(panel["circ_mv"], errors="coerce") / 10000.0
            panel = panel[mv.between(min_mv, max_mv) | mv.isna()]
    panel = panel.reset_index(drop=True)
    if panel.empty:
        st.error("过滤后无数据。")
        return

    table_2x2, contrib = factorial_2x2(panel, float(deep_quantile), float(cost_pct))
    schemes = compare_selection_schemes(panel, int(top_n), float(cost_pct))
    yearly = yearly_scheme_comparison(panel, int(top_n), float(cost_pct))

    st.session_state["decomp_result"] = {
        "panel_size": len(panel),
        "weeks": panel["Entry_Week"].nunique(),
        "skdj_count": int(panel["SKDJ_Signal"].astype(bool).sum()),
        "table_2x2": table_2x2,
        "contrib": contrib,
        "schemes": schemes,
        "yearly": yearly,
        "params": {
            "N": int(n_period), "M": int(m_period), "阈值": float(level),
            "K>D": bool(require_kd), "持有": int(hold_weeks),
            "每周选": int(top_n), "深度回撤定义": float(deep_quantile),
        },
    }


def render_results():
    result = st.session_state.get("decomp_result")
    if not result:
        return False
    params = result["params"]

    st.markdown("---")
    st.header("分解验证结果")
    st.caption(
        f"全池观测 {result['panel_size']:,} 个「个股-周」，其中SKDJ信号 "
        f"{result['skdj_count']:,} 个，覆盖 {result['weeks']} 周　|　"
        f"SKDJ: N={params['N']} M={params['M']} K≤{params['阈值']:.0f}"
        f"{' 且K>D' if params['K>D'] else ''}　持有{params['持有']}周　"
        f"每周选{params['每周选']}只　深度回撤=最深{params['深度回撤定义']*100:.0f}%"
    )

    st.subheader("表1 · 2x2析因：四个格子的表现")
    st.dataframe(result["table_2x2"].round(3), width="stretch", hide_index=True)
    st.caption(
        "**重点看前两行**：同样是深度回撤的股票，有SKDJ信号和没有SKDJ信号，收益差多少。"
    )

    st.subheader("表2 · 因子净贡献（控制另一因子后）")
    st.dataframe(result["contrib"].round(3), width="stretch", hide_index=True)
    st.caption(
        "**这是决策依据。**\n"
        "- SKDJ净贡献接近0或为负 → 可以放弃SKDJ，直接用回撤选股：信号更多、逻辑更简单\n"
        "- SKDJ净贡献明显为正（两行都为正）→ SKDJ有独立价值，值得保留\n"
        "- 只有一行为正、另一行为负 → 不稳定，多半是噪声"
    )

    st.subheader("表3 · 四种选股方案正面对比")
    st.dataframe(result["schemes"].round(3), width="stretch", hide_index=True)
    st.caption(
        "①是当前策略，②是去掉SKDJ，③是只靠SKDJ不排序，④是纯随机基准。"
        "**如果②不比①差，那SKDJ就是多余的**——而且②的可交易周数会明显更多，实操性更好。"
    )

    if not result["yearly"].empty:
        st.subheader("表4 · 分年度：用SKDJ vs 不用SKDJ")
        st.dataframe(result["yearly"].round(3), width="stretch", hide_index=True)
        st.caption(
            "看「差值」这一列是否稳定为正。如果各年正负交替、幅度不大，"
            "说明SKDJ的贡献不可靠。"
        )

    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_factorial_2x2.csv",
            result["table_2x2"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "02_factor_contribution.csv",
            result["contrib"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "03_scheme_comparison.csv",
            result["schemes"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "04_yearly_comparison.csv",
            result["yearly"].to_csv(index=False, encoding="utf-8-sig"),
        )
    st.download_button(
        "下载分解验证结果",
        data=output.getvalue(),
        file_name="skdj_vs_drawdown_decomposition.zip",
        mime="application/zip",
        key="download_decomp",
    )
    return True


if __name__ == "__main__":
    main()
    render_results()
