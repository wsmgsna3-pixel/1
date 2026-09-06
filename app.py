# -*- coding: utf-8 -*-
"""洗盘验证器（单文件独立版，直接覆盖 app.py 运行）。

上一轮结论：突破+接近两年高点+波动压缩的组合，涨超100%的概率是全池基准的
2.37倍——选股确实找对了。但移动止损后的实际收益却只有3.95%，
且加入"区间宽"特征后收益转负、止损中位数从-7%恶化到-14%、胜率从40%掉到31%。

诊断：不是选错股票，而是固定止损把赢家提前赶走了。

本工具验证两个假设：
  1. 真正的上涨浪起来之前存在洗盘（即使最终大涨的股票，中途也深度回撤）
  2. 如果洗盘存在，能否等洗盘结束后再介入

第二点的关键优势：在更低位置进场后，同样15%的止损从更低成本算起，
被打掉的概率自然更小——不需要把止损放宽到实盘无法承受的25~40%。

必须同时验证的前提：洗盘和失败在事前是否可区分。如果不可区分，
"等回撤再买"就等于把失败的票也一起买了，这会体现在成交率和大涨比例上。

止损范围限定在5~25%（实盘可承受），不再测更宽的止损。

所有判断只用当周及之前数据；限价单按盘中触及成交，确认类按当周收盘确认、
次周开盘买入，无未来函数。

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

APP_TITLE = "洗盘验证：能否等洗盘后再介入"
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
def compute_features(weekly: pd.DataFrame, breakout_weeks: int, base_weeks: int):
    close = pd.to_numeric(weekly["close"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    low = pd.to_numeric(weekly["low"], errors="coerce")
    volume = (
        pd.to_numeric(weekly["vol"], errors="coerce")
        if "vol" in weekly.columns
        else pd.Series(np.nan, index=weekly.index, dtype="float64")
    )
    return_1w = (close / close.shift(1) - 1.0) * 100.0

    features = pd.DataFrame(index=weekly.index)
    features["breakout"] = close > close.shift(1).rolling(breakout_weeks).max()
    features["position_2y"] = close / high.shift(1).rolling(104).max().replace(0, np.nan)
    vol_recent = return_1w.shift(1).rolling(8).std()
    vol_earlier = return_1w.shift(9).rolling(18).std()
    features["vol_contraction"] = vol_recent / vol_earlier.replace(0, np.nan)
    features["volume_surge"] = (
        volume / volume.shift(1).rolling(8).mean().replace(0, np.nan)
    )
    # 个股自身的周波动率，用于判断回撤是否属于其正常波动范围
    features["weekly_vol"] = return_1w.rolling(26).std()
    return features


def analyze_signal_path(
    close_values, high_values, low_values, open_values,
    signal_index: int, forward_weeks: int, stop_pct: float,
    pullback_levels, confirm_modes,
):
    """对单个突破信号，模拟"立即买入"与各种"等洗盘后买入"的结果。

    统一口径：
      - 限价单：盘中最低价触及目标价即成交；若开盘已低于目标价则按开盘价
      - 确认类：当周收盘满足条件 -> 次周开盘买入
      - 所有方式共用同一个观察截止周，止损幅度也相同
    """
    n = len(close_values)
    stop_index = min(signal_index + forward_weeks, n - 1)
    if stop_index <= signal_index + 1:
        return None

    entry_index_immediate = signal_index + 1
    entry_immediate = open_values[entry_index_immediate]
    if not math.isfinite(entry_immediate) or entry_immediate <= 0:
        return None
    signal_close = close_values[signal_index]

    def outcome_from(entry_idx, entry_price):
        """给定入场点，算最大涨幅、止损收益、持有到期收益、以及最大逆向波动。"""
        if entry_idx is None or not math.isfinite(entry_price) or entry_price <= 0:
            return None
        if entry_idx > stop_index:
            return None
        window_high = high_values[entry_idx : stop_index + 1]
        window_low = low_values[entry_idx : stop_index + 1]
        finite_high = window_high[np.isfinite(window_high)]
        max_gain = (
            (finite_high.max() / entry_price - 1.0) * 100.0 if finite_high.size else np.nan
        )
        # 到达最高点之前的最大跌幅（衡量"要忍多少痛才等到涨"）
        if finite_high.size:
            peak_offset = int(np.nanargmax(window_high))
            pre_peak_low = window_low[: peak_offset + 1]
            finite_low = pre_peak_low[np.isfinite(pre_peak_low)]
            max_adverse = (
                (finite_low.min() / entry_price - 1.0) * 100.0 if finite_low.size else np.nan
            )
        else:
            max_adverse = np.nan
        # 止损收益
        peak = entry_price
        exit_price = None
        for j in range(entry_idx, stop_index + 1):
            current = close_values[j]
            if not math.isfinite(current):
                continue
            peak = max(peak, current)
            if current <= peak * (1.0 - stop_pct / 100.0):
                exit_price = current
                break
        if exit_price is None:
            exit_price = close_values[stop_index]
        trail_return = (
            (exit_price / entry_price - 1.0) * 100.0 if math.isfinite(exit_price) else np.nan
        )
        hold_return = (
            (close_values[stop_index] / entry_price - 1.0) * 100.0
            if math.isfinite(close_values[stop_index])
            else np.nan
        )
        return {
            "max_gain": max_gain,
            "max_adverse": max_adverse,
            "trail": trail_return,
            "hold": hold_return,
            "entry_index": entry_idx,
        }

    result = {"signal_index": signal_index}
    immediate = outcome_from(entry_index_immediate, entry_immediate)
    if immediate is None:
        return None
    result["立即买入"] = immediate

    # ---- 等回撤到指定幅度再买（限价单）----
    for level in pullback_levels:
        target = signal_close * (1.0 - level / 100.0)
        filled_index = None
        fill_price = np.nan
        for j in range(signal_index + 1, stop_index + 1):
            day_low = low_values[j]
            day_open = open_values[j]
            if math.isfinite(day_low) and day_low <= target:
                fill_price = (
                    day_open
                    if math.isfinite(day_open) and day_open < target
                    else target
                )
                filled_index = j
                break
        outcome = outcome_from(filled_index, fill_price) if filled_index else None
        result[f"回撤{level:.0f}%买入"] = outcome

    # ---- 洗盘后需要回升确认 ----
    for level, mode in confirm_modes:
        target = signal_close * (1.0 - level / 100.0)
        pullback_index = None
        for j in range(signal_index + 1, stop_index + 1):
            day_low = low_values[j]
            if math.isfinite(day_low) and day_low <= target:
                pullback_index = j
                break
        outcome = None
        if pullback_index is not None:
            for j in range(pullback_index, stop_index):
                current = close_values[j]
                if not math.isfinite(current):
                    continue
                confirmed = False
                if mode == "站回突破价":
                    confirmed = current >= signal_close
                elif mode == "周线收阳":
                    prev = close_values[j - 1] if j > 0 else np.nan
                    confirmed = math.isfinite(prev) and current > prev
                if confirmed:
                    entry_price = open_values[j + 1]
                    outcome = outcome_from(j + 1, entry_price)
                    break
        result[f"回撤{level:.0f}%后{mode}"] = outcome

    return result


def build_signals(
    weekly: pd.DataFrame, ts_code: str, breakout_weeks: int, base_weeks: int,
    forward_weeks: int, stop_pct: float, pullback_levels, confirm_modes,
    position_quantile_value: float, use_contraction: bool,
) -> pd.DataFrame:
    if len(weekly) < base_weeks + 30:
        return pd.DataFrame()
    features = compute_features(weekly, breakout_weeks, base_weeks)
    close_values = pd.to_numeric(weekly["close"], errors="coerce").to_numpy()
    high_values = pd.to_numeric(weekly["high"], errors="coerce").to_numpy()
    low_values = pd.to_numeric(weekly["low"], errors="coerce").to_numpy()
    open_values = pd.to_numeric(weekly["open"], errors="coerce").to_numpy()
    dates = weekly["trade_date_str"].astype(str).tolist()

    breakout = features["breakout"].fillna(False).to_numpy()
    position = features["position_2y"].to_numpy()
    contraction = features["vol_contraction"].to_numpy()
    volume_surge = features["volume_surge"].to_numpy()
    weekly_vol = features["weekly_vol"].to_numpy()

    rows = []
    for i in range(len(weekly)):
        if not breakout[i]:
            continue
        if math.isfinite(position_quantile_value) and not (
            math.isfinite(position[i]) and position[i] >= position_quantile_value
        ):
            continue
        if use_contraction and not (
            math.isfinite(contraction[i]) and contraction[i] <= 0.8
        ):
            continue
        path = analyze_signal_path(
            close_values, high_values, low_values, open_values,
            i, forward_weeks, stop_pct, pullback_levels, confirm_modes,
        )
        if path is None:
            continue
        row = {
            "ts_code": ts_code,
            "Week": dates[i],
            "Signal_Close": close_values[i],
            "Position_2Y": position[i],
            "Vol_Contraction": contraction[i],
            "Volume_Surge": volume_surge[i],
            "Weekly_Vol": weekly_vol[i],
        }
        for key, outcome in path.items():
            if key == "signal_index":
                continue
            if outcome is None:
                row[f"{key}_成交"] = False
                continue
            row[f"{key}_成交"] = True
            row[f"{key}_最大涨幅"] = outcome["max_gain"]
            row[f"{key}_最大逆向"] = outcome["max_adverse"]
            row[f"{key}_止损收益"] = outcome["trail"]
            row[f"{key}_持有到期"] = outcome["hold"]
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 分析
# -----------------------------------------------------------------------------
def washout_evidence(signals: pd.DataFrame):
    """洗盘存在性检验：把最终大涨的和没涨的分开，看它们中途各回撤了多少。

    如果连最终大涨的股票，中途也普遍深度回撤，
    就证明"固定止损把赢家提前赶走"这个诊断成立。
    """
    gain = pd.to_numeric(signals.get("立即买入_最大涨幅"), errors="coerce")
    adverse = pd.to_numeric(signals.get("立即买入_最大逆向"), errors="coerce")
    work = pd.DataFrame({"gain": gain, "adverse": adverse}).dropna()
    if work.empty:
        return pd.DataFrame()
    buckets = [
        ("失败（最大涨幅<10%）", work["gain"] < 10),
        ("一般（10~30%）", (work["gain"] >= 10) & (work["gain"] < 30)),
        ("良好（30~50%）", (work["gain"] >= 30) & (work["gain"] < 50)),
        ("大涨（50~100%）", (work["gain"] >= 50) & (work["gain"] < 100)),
        ("翻倍以上（>100%）", work["gain"] >= 100),
    ]
    rows = []
    for label, mask in buckets:
        subset = work.loc[mask, "adverse"]
        if subset.empty:
            continue
        rows.append(
            {
                "最终结果分组": label,
                "样本数": int(len(subset)),
                "中途最大跌幅 中位数%": float(subset.median()),
                "中途最大跌幅 平均%": float(subset.mean()),
                "跌超10%比例": float((subset <= -10).mean() * 100.0),
                "跌超15%比例": float((subset <= -15).mean() * 100.0),
                "跌超20%比例": float((subset <= -20).mean() * 100.0),
                "跌超30%比例": float((subset <= -30).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def entry_method_comparison(signals: pd.DataFrame, cost_pct: float):
    """各入场方式对比：洗盘后介入是否真的更好。"""
    methods = sorted(
        {
            column.rsplit("_", 1)[0]
            for column in signals.columns
            if column.endswith("_止损收益")
        }
    )
    total = len(signals)
    rows = []
    for method in methods:
        filled = signals.get(f"{method}_成交", pd.Series(False, index=signals.index))
        filled = filled.fillna(False).astype(bool)
        trail = (
            pd.to_numeric(signals.loc[filled, f"{method}_止损收益"], errors="coerce").dropna()
            - cost_pct
        )
        hold = (
            pd.to_numeric(signals.loc[filled, f"{method}_持有到期"], errors="coerce").dropna()
            - cost_pct
        )
        gain = pd.to_numeric(
            signals.loc[filled, f"{method}_最大涨幅"], errors="coerce"
        ).dropna()
        adverse = pd.to_numeric(
            signals.loc[filled, f"{method}_最大逆向"], errors="coerce"
        ).dropna()
        if trail.empty:
            continue
        rows.append(
            {
                "入场方式": method,
                "成交数": int(filled.sum()),
                "成交率%": float(filled.sum() / total * 100.0) if total else np.nan,
                "止损后收益%": float(trail.mean()),
                "止损后中位%": float(trail.median()),
                "止损后胜率%": float((trail > 0).mean() * 100.0),
                "持有到期收益%": float(hold.mean()) if len(hold) else np.nan,
                "涨超50%比例": float((gain > 50).mean() * 100.0) if len(gain) else np.nan,
                "涨超100%比例": float((gain > 100).mean() * 100.0) if len(gain) else np.nan,
                "中途最大跌幅中位%": float(adverse.median()) if len(adverse) else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("止损后收益%", ascending=False).reset_index(drop=True)
    return result


def stop_sensitivity(signals: pd.DataFrame, cost_pct: float):
    """在最优入场方式下，不同止损幅度的效果（只测实盘可承受的范围）。

    注意：这里用"中途最大跌幅"重新推算不同止损的触发情况是近似的，
    真实结果需要重跑；本表用于判断方向，不作为最终依据。
    """
    methods = sorted(
        {
            column.rsplit("_", 1)[0]
            for column in signals.columns
            if column.endswith("_最大逆向")
        }
    )
    rows = []
    for method in methods:
        filled = signals.get(f"{method}_成交", pd.Series(False, index=signals.index))
        filled = filled.fillna(False).astype(bool)
        adverse = pd.to_numeric(
            signals.loc[filled, f"{method}_最大逆向"], errors="coerce"
        )
        gain = pd.to_numeric(
            signals.loc[filled, f"{method}_最大涨幅"], errors="coerce"
        )
        work = pd.DataFrame({"adverse": adverse, "gain": gain}).dropna()
        if work.empty:
            continue
        big = work["gain"] >= 50
        row = {"入场方式": method, "成交数": int(len(work))}
        for stop in (8, 10, 12, 15, 20):
            killed = work["adverse"] <= -stop
            row[f"{stop}%止损误杀大涨比例"] = (
                float((killed & big).sum() / big.sum() * 100.0) if big.sum() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def yearly_entry_table(signals: pd.DataFrame, cost_pct: float):
    work = signals.copy()
    work["年份"] = work["Week"].astype(str).str[:4]
    methods = sorted(
        {
            column.rsplit("_", 1)[0]
            for column in signals.columns
            if column.endswith("_止损收益")
        }
    )
    rows = []
    for year, group in work.groupby("年份"):
        record = {"年份": year, "信号数": len(group)}
        for method in methods:
            filled = group.get(f"{method}_成交", pd.Series(False, index=group.index))
            filled = filled.fillna(False).astype(bool)
            trail = (
                pd.to_numeric(group.loc[filled, f"{method}_止损收益"], errors="coerce").dropna()
                - cost_pct
            )
            record[method] = float(trail.mean()) if len(trail) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------
def _memory_usage_mb():
    """当前进程内存占用（MB）。Streamlit Cloud 上限约1GB，超过会被直接杀掉，
    表现为日志无报错、应用消失、需要重新部署。用它做运行中的预警。"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🔍 {APP_TITLE}")
    st.caption("先证明洗盘存在，再测能不能等洗盘结束后再进场。")
    st.info(
        "**两步验证：**\n\n"
        "**第一步 · 洗盘是否真实存在**　把突破后最终大涨的股票和没涨的分开，"
        "看它们中途各自回撤了多少。如果连最终翻倍的股票中途也普遍跌15%以上，"
        "就证明固定止损确实在提前赶走赢家。\n\n"
        "**第二步 · 能否等洗盘后再买**　对比立即买入与几种等回撤后介入的方式。"
        "**关键优势**：在更低的位置进场，同样15%的止损从更低成本算起，"
        "被打掉的概率自然更小——不用放宽止损就能解决被洗出的问题。\n\n"
        "**但有个前提要验证**：洗盘和失败在事前长得一样。"
        "如果分不出来，等回撤再买就等于把失败的票也一起买了——"
        "这会体现在成交率和涨超50%比例的变化上。"
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
        st.subheader("突破信号")
        breakout_weeks = st.number_input("突破几周新高", value=26, min_value=4, max_value=104, step=2)
        base_weeks = st.number_input("基底考察周数", value=26, min_value=8, max_value=104, step=2)
        position_cut = st.slider(
            "只保留接近两年高点的前百分之几",
            min_value=0.10, max_value=1.00, value=0.33, step=0.05,
            help="上轮验证：越接近两年高点，大涨概率越高（5年里4年正向）。设为1.00则不筛选。",
        )
        use_contraction = st.checkbox(
            "同时要求波动率压缩（≤0.8）", value=True,
            help="上轮F组合的组成部分，涨超100%提升2.37倍。",
        )

        st.markdown("---")
        st.subheader("洗盘与止损")
        forward_weeks = st.number_input("前瞻观察周数", value=26, min_value=8, max_value=104, step=2)
        stop_pct = st.number_input(
            "移动止损%（实盘可承受范围）", value=15.0, min_value=5.0, max_value=25.0, step=1.0,
            help="不再测25%以上——单笔亏那么多实盘扛不住。",
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
        if st.session_state.get("wash_result"):
            return
        st.markdown(
            """
### 对比的入场方式

| 方式 | 说明 |
|---|---|
| 立即买入 | 突破次周开盘买（基准） |
| 回撤8/12/15%买入 | 挂低于突破周收盘价N%的限价单 |
| 回撤12%后站回突破价 | 洗完盘、价格重新站上突破价才买 |
| 回撤12%后周线收阳 | 洗完盘、出现一根周阳线才买 |

后两种是"等洗盘结束的信号"，比单纯挂限价单更谨慎，
代价是买得更高、可能错过。

### 四张表

**表1 · 洗盘存在性**　按最终涨幅分组，看各组中途跌了多少。
**这张表决定后面所有讨论有没有意义。**

**表2 · 入场方式对比**　重点看「止损后收益%」和「中途最大跌幅中位%」——
后者反映在更低位置进场是否真的减轻了持仓压力。

**表3 · 止损误杀分析**　在各入场方式下，8/10/12/15/20%的止损
分别会误杀掉多少比例的大涨行情。这直接回答"止损该设多少"。

**表4 · 分年度**　确认结论稳定。
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

    # 内存优化：basic_indexed 含多列且行数庞大，但后续只用到流通市值。
    # 立即裁成小表并释放原对象，避免它在整个回测过程中一直占内存。
    if not basic_indexed.empty and "circ_mv" in basic_indexed.columns:
        mv_lookup = (
            basic_indexed[["circ_mv"]]
            .reset_index()
            .rename(columns={"trade_date_str": "Week"})
        )
        mv_lookup["circ_mv"] = pd.to_numeric(
            mv_lookup["circ_mv"], errors="coerce"
        ).astype("float32")
        mv_lookup = mv_lookup.drop_duplicates(["Week", "ts_code"])
    else:
        mv_lookup = pd.DataFrame()
    del basic_indexed
    gc.collect()

    pullback_levels = [8.0, 12.0, 15.0]
    confirm_modes = [(12.0, "站回突破价"), (12.0, "周线收阳")]

    # 内存优化：日线数据体积远大于周线（约1300只×1200行×20列）。
    # 这里边构建周线边把对应日线从字典中弹出并释放，避免两份完整数据
    # 同时驻留内存——Streamlit Cloud 内存上限约1GB，同时保留会被OOM杀掉
    # （表现为日志无报错、应用直接消失、必须重新部署）。
    needed_columns = ["trade_date_str", "open", "high", "low", "close", "vol"]
    position_samples = []
    weekly_cache = {}
    codes = sorted(stocks.keys())
    total_codes = len(codes)
    prep = st.progress(0.0, text="构建周线并计算全池位置分布……")
    for idx, ts_code in enumerate(codes):
        daily = stocks.pop(ts_code)  # 弹出：字典中不再持有该股日线
        weekly = build_weekly_bars(daily)
        del daily
        if weekly.empty or len(weekly) < int(base_weeks) + 30:
            continue
        # 只保留后续真正用到的列，并降精度，进一步压缩占用
        keep = [c for c in needed_columns if c in weekly.columns]
        weekly = weekly[keep].copy()
        for column in ("open", "high", "low", "close", "vol"):
            if column in weekly.columns:
                weekly[column] = pd.to_numeric(
                    weekly[column], errors="coerce"
                ).astype("float32")
        weekly_cache[ts_code] = weekly
        features = compute_features(weekly, int(breakout_weeks), int(base_weeks))
        values = features.loc[features["breakout"].fillna(False), "position_2y"]
        position_samples.append(values.dropna().astype("float32"))
        del features
        if idx % 60 == 0:
            prep.progress(min((idx + 1) / total_codes, 1.0))
    prep.empty()
    del stocks
    gc.collect()

    if not position_samples:
        st.error("数据不足。")
        return
    all_positions = pd.concat(position_samples, ignore_index=True)
    del position_samples
    gc.collect()
    position_threshold = (
        float(all_positions.quantile(1.0 - float(position_cut)))
        if float(position_cut) < 1.0
        else float("-inf")
    )
    del all_positions
    gc.collect()
    st.caption(
        f"位置门槛：只保留突破时价格 ≥ 两年高点的 "
        f"{position_threshold:.3f} 倍（前{float(position_cut)*100:.0f}%）"
    )

    progress = st.progress(0.0, text="模拟洗盘与各种入场方式……")
    parts = []
    cached_codes = list(weekly_cache.keys())
    for idx, ts_code in enumerate(cached_codes):
        weekly = weekly_cache.pop(ts_code)  # 用完即释放
        rows = build_signals(
            weekly, ts_code, int(breakout_weeks), int(base_weeks),
            int(forward_weeks), float(stop_pct), pullback_levels, confirm_modes,
            position_threshold, bool(use_contraction),
        )
        del weekly
        if not rows.empty:
            parts.append(rows)
        if idx % 40 == 0:
            progress.progress(min((idx + 1) / len(cached_codes), 1.0))
    progress.empty()
    del weekly_cache
    gc.collect()

    if not parts:
        st.error("没有产生信号，请放宽筛选条件。")
        return
    signals = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    signals = signals[(signals["Week"] >= start_date) & (signals["Week"] <= end_date)]
    signals = signals[
        pd.to_numeric(signals["Signal_Close"], errors="coerce") >= min_price
    ]
    if not mv_lookup.empty:
        signals = signals.merge(mv_lookup, on=["Week", "ts_code"], how="left")
        mv = pd.to_numeric(signals["circ_mv"], errors="coerce") / 10000.0
        signals = signals[mv.between(min_mv, max_mv) | mv.isna()]
        del mv_lookup
        gc.collect()
    signals = signals.reset_index(drop=True)
    if signals.empty:
        st.error("过滤后无信号。")
        return

    used_mb = _memory_usage_mb()
    if math.isfinite(used_mb):
        if used_mb > 750:
            st.warning(
                f"当前内存占用 {used_mb:.0f} MB，已接近 Streamlit Cloud 约1GB的上限。"
                "如果之后出现应用崩溃需重新部署，请缩短回测时间范围"
                "（例如改成2年）或收紧股票池条件。"
            )
        else:
            st.caption(f"当前内存占用 {used_mb:.0f} MB（上限约1GB）")

    st.session_state["wash_result"] = {
        "signals": signals,
        "washout": washout_evidence(signals),
        "entries": entry_method_comparison(signals, float(cost_pct)),
        "stops": stop_sensitivity(signals, float(cost_pct)),
        "yearly": yearly_entry_table(signals, float(cost_pct)),
        "params": {
            "突破周数": int(breakout_weeks), "前瞻": int(forward_weeks),
            "止损": float(stop_pct), "位置前": float(position_cut),
            "要求压缩": bool(use_contraction),
        },
    }


def render_results():
    result = st.session_state.get("wash_result")
    if not result:
        return False
    params = result["params"]
    signals = result["signals"]

    st.markdown("---")
    st.header("洗盘验证结果")
    st.caption(
        f"突破信号 {len(signals):,} 个，覆盖 {signals['Week'].min()} — "
        f"{signals['Week'].max()}　|　突破{params['突破周数']}周新高　"
        f"前瞻{params['前瞻']}周　移动止损{params['止损']:.0f}%　"
        f"位置前{params['位置前']*100:.0f}%"
        f"{'　含波动压缩' if params['要求压缩'] else ''}"
    )

    st.subheader("表1 · 洗盘存在性：最终大涨的股票，中途跌了多少？")
    st.dataframe(result["washout"].round(2), width="stretch", hide_index=True)
    st.caption(
        "**这张表决定后面所有讨论有没有意义。**\n\n"
        "看「翻倍以上」那一行的「跌超15%比例」：如果很高，"
        "说明即使是最终翻倍的股票，中途也普遍深跌——"
        "固定15%止损确实在提前赶走赢家，你的洗盘假设成立。\n\n"
        "但同时对比「失败」那一行：如果失败组跌得更深、比例更高，"
        "说明深跌本身仍是负面信号，只是不够干净。"
    )

    st.subheader("表2 · 入场方式对比：等洗盘后再买是否更好？")
    st.dataframe(result["entries"].round(2), width="stretch", hide_index=True)
    st.caption(
        "**看三列**：「止损后收益%」是最终结果；"
        "「中途最大跌幅中位%」反映在更低位置进场是否真的减轻了持仓压力；"
        "「成交率%」反映会错过多少机会。\n\n"
        "如果等回撤的方式收益更高**且**中途跌幅更小，"
        "那就是双赢——不用放宽止损就解决了被洗出的问题。"
    )

    if not result["stops"].empty:
        st.subheader("表3 · 止损误杀分析：止损该设多少？")
        st.dataframe(result["stops"].round(2), width="stretch", hide_index=True)
        st.caption(
            "各入场方式下，不同止损幅度会误杀掉多少比例的大涨行情"
            "（那些最终涨超50%、但中途被止损打出的）。\n\n"
            "**对比同一行的8%到20%**：如果从15%放宽到20%误杀率大幅下降，"
            "说明止损确实偏紧；如果差别不大，说明问题不在止损幅度。"
        )

    if not result["yearly"].empty:
        st.subheader("表4 · 各入场方式分年度")
        st.dataframe(result["yearly"].round(2), width="stretch", hide_index=True)
        st.caption("确认最优方式不是靠某一年。")

    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_washout_evidence.csv",
            result["washout"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "02_entry_methods.csv",
            result["entries"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "03_stop_sensitivity.csv",
            result["stops"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "04_yearly.csv",
            result["yearly"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "05_all_signals.csv",
            signals.to_csv(index=False, encoding="utf-8-sig"),
        )
    st.download_button(
        "下载验证结果",
        data=output.getvalue(),
        file_name="washout_validation.zip",
        mime="application/zip",
        key="download_wash",
    )
    return True


if __name__ == "__main__":
    main()
    render_results()
