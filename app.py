# -*- coding: utf-8 -*-
"""退出规则对比验证器（单文件独立版，直接覆盖 app.py 运行）。

此前十几轮全部在优化"买什么""何时买"，退出一直雷打不动固定持有3周。
但固定持有期会把所有大赢家提前砍断——这可能正是
"157笔交易里剔掉最好的5笔就归零"的真正原因。

本次只改一件事：卖出方式。
同一批信号、同一个买入价、同一段行情，唯一差别是何时卖出，
因此收益差异纯粹来自退出规则。

对比的退出方式：
  固定持有 3/4/6/10/13 周
  移动止损（从持有期最高收盘价回撤N%）
  持有到SKDJ进入高位区（吃完一整个波浪）
  跌破N周均线

判断标准与之前不同：
本次改变的是收益分布的形状而非平均值，因此除了平均收益，更要看
  - 赚>50%的交易占比（能否留住大波段）
  - 大涨样本捕获率（在真正出现大涨的交易里，实际吃到了理论涨幅的多少）
  - 平均持有周数（决定交易节奏是否可接受）

成交口径：所有规则统一按周收盘价卖出，不假设能卖在盘中最高点。
最长持有周数为所有非固定规则设置上限，避免无限持有。

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

APP_TITLE = "退出规则对比验证"
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


def simulate_exits(
    weekly: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    max_weeks: int,
    trail_pct: float,
    skdj_top: float,
    ma_exit_weeks: int,
    fixed_weeks_list,
):
    """从买入那一周开始，逐周推进，模拟多种退出规则。

    所有规则共用同一个买入价和同一段行情，唯一差别是何时卖出，
    因此收益差异纯粹来自退出方式。

    统一口径：每周收盘判断是否满足退出条件，满足则按该周收盘价卖出。
    这样不存在"用盘中最高价卖出"这种做不到的假设。
    """
    results = {}
    n = len(weekly)
    close = pd.to_numeric(weekly["close"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    k_series = pd.to_numeric(weekly["K"], errors="coerce")
    ma_exit = close.rolling(ma_exit_weeks).mean()

    last_index = min(entry_index + max_weeks, n - 1)
    if last_index <= entry_index or not math.isfinite(entry_price) or entry_price <= 0:
        return results

    # ---- 固定持有N周 ----
    for weeks in fixed_weeks_list:
        exit_index = entry_index + weeks
        if exit_index < n:
            exit_price = _safe_float(close.iloc[exit_index])
            if math.isfinite(exit_price):
                results[f"固定持有{weeks}周"] = {
                    "return_pct": (exit_price / entry_price - 1.0) * 100.0,
                    "weeks_held": weeks,
                }

    # ---- 移动止损：从最高收盘价回撤trail_pct则卖出 ----
    peak = entry_price
    trail_result = None
    for step in range(1, last_index - entry_index + 1):
        idx = entry_index + step
        current = _safe_float(close.iloc[idx])
        if not math.isfinite(current):
            continue
        peak = max(peak, current)
        if current <= peak * (1.0 - trail_pct / 100.0):
            trail_result = {
                "return_pct": (current / entry_price - 1.0) * 100.0,
                "weeks_held": step,
            }
            break
    if trail_result is None:
        final = _safe_float(close.iloc[last_index])
        if math.isfinite(final):
            trail_result = {
                "return_pct": (final / entry_price - 1.0) * 100.0,
                "weeks_held": last_index - entry_index,
            }
    if trail_result:
        results[f"移动止损{trail_pct:.0f}%"] = trail_result

    # ---- 持有到SKDJ进入高位区（吃完一整个波浪）----
    skdj_result = None
    for step in range(1, last_index - entry_index + 1):
        idx = entry_index + step
        k_value = _safe_float(k_series.iloc[idx])
        current = _safe_float(close.iloc[idx])
        if not math.isfinite(k_value) or not math.isfinite(current):
            continue
        if k_value >= skdj_top:
            skdj_result = {
                "return_pct": (current / entry_price - 1.0) * 100.0,
                "weeks_held": step,
            }
            break
    if skdj_result is None:
        final = _safe_float(close.iloc[last_index])
        if math.isfinite(final):
            skdj_result = {
                "return_pct": (final / entry_price - 1.0) * 100.0,
                "weeks_held": last_index - entry_index,
            }
    if skdj_result:
        results[f"持有到SKDJ≥{skdj_top:.0f}"] = skdj_result

    # ---- 跌破均线卖出 ----
    ma_result = None
    for step in range(1, last_index - entry_index + 1):
        idx = entry_index + step
        current = _safe_float(close.iloc[idx])
        ma_value = _safe_float(ma_exit.iloc[idx])
        if not math.isfinite(current) or not math.isfinite(ma_value):
            continue
        if current < ma_value:
            ma_result = {
                "return_pct": (current / entry_price - 1.0) * 100.0,
                "weeks_held": step,
            }
            break
    if ma_result is None:
        final = _safe_float(close.iloc[last_index])
        if math.isfinite(final):
            ma_result = {
                "return_pct": (final / entry_price - 1.0) * 100.0,
                "weeks_held": last_index - entry_index,
            }
    if ma_result:
        results[f"跌破{ma_exit_weeks}周均线"] = ma_result

    # ---- 参考值：这段行情的理论最大涨幅（无法实际获得，仅用于衡量各规则吃到了多少）----
    window_high = pd.to_numeric(
        high.iloc[entry_index + 1 : last_index + 1], errors="coerce"
    )
    if window_high.notna().any():
        results["__max_possible__"] = {
            "return_pct": (window_high.max() / entry_price - 1.0) * 100.0,
            "weeks_held": np.nan,
        }
    return results


def build_signals_with_exits(
    weekly: pd.DataFrame, ts_code: str, n_period: int, m_period: int,
    level: float, require_kd: bool, max_weeks: int, trail_pct: float,
    skdj_top: float, ma_exit_weeks: int, fixed_weeks_list,
) -> pd.DataFrame:
    weekly = add_skdj(weekly, n_period, m_period)
    if len(weekly) < 60:
        return pd.DataFrame()

    close = pd.to_numeric(weekly["close"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    open_p = pd.to_numeric(weekly["open"], errors="coerce")
    k_now = pd.to_numeric(weekly["K"], errors="coerce")
    k_prev = k_now.shift(1)
    d_now = pd.to_numeric(weekly["D"], errors="coerce")
    dates = weekly["trade_date_str"].astype(str)

    signal = (k_prev <= k_now) & (k_now <= level)
    if require_kd:
        signal = signal & (k_now > d_now)
    signal = signal.fillna(False)

    drawdown_26w = (close / high.rolling(26).max() - 1.0) * 100.0
    ma40 = close.rolling(40).mean()

    rows = []
    for i in range(len(weekly)):
        if not bool(signal.iloc[i]):
            continue
        entry_index = i + 1
        if entry_index >= len(weekly):
            continue
        entry_price = _safe_float(open_p.iloc[entry_index])
        if not math.isfinite(entry_price) or entry_price <= 0:
            continue
        exits = simulate_exits(
            weekly, entry_index, entry_price, max_weeks, trail_pct,
            skdj_top, ma_exit_weeks, fixed_weeks_list,
        )
        if not exits:
            continue
        row = {
            "ts_code": ts_code,
            "Signal_Week": dates.iloc[i],
            "Entry_Week": dates.iloc[entry_index],
            "Entry_Price": entry_price,
            "K": _safe_float(k_now.iloc[i]),
            "Drawdown_26W_pct": _safe_float(drawdown_26w.iloc[i]),
            "Dist_MA40_pct": _safe_float((close.iloc[i] / ma40.iloc[i] - 1.0) * 100.0)
            if math.isfinite(_safe_float(ma40.iloc[i]))
            else np.nan,
        }
        for label, payload in exits.items():
            if label == "__max_possible__":
                row["理论最大涨幅%"] = payload["return_pct"]
                continue
            row[f"收益_{label}"] = payload["return_pct"]
            row[f"周数_{label}"] = payload["weeks_held"]
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 分析
# -----------------------------------------------------------------------------
def compare_exit_rules(signals: pd.DataFrame, cost_pct: float):
    """各退出规则的正面对比。

    重点不只是平均收益，更要看收益分布的形状：
    大赢家占比决定了能否留住"黄金"，这正是固定持有期最可能损害的地方。
    """
    labels = [c[3:] for c in signals.columns if c.startswith("收益_")]
    rows = []
    for label in labels:
        returns = pd.to_numeric(signals[f"收益_{label}"], errors="coerce").dropna()
        if returns.empty:
            continue
        net = returns - cost_pct
        weeks_column = f"周数_{label}"
        weeks = (
            pd.to_numeric(signals[weeks_column], errors="coerce").dropna()
            if weeks_column in signals.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "退出规则": label,
                "样本数": int(len(net)),
                "平均收益%": float(net.mean()),
                "中位收益%": float(net.median()),
                "胜率%": float((net > 0).mean() * 100.0),
                "平均持有周数": float(weeks.mean()) if len(weeks) else np.nan,
                "赚>30%比例": float((net > 30).mean() * 100.0),
                "赚>50%比例": float((net > 50).mean() * 100.0),
                "赚>100%比例": float((net > 100).mean() * 100.0),
                "亏>20%比例": float((net < -20).mean() * 100.0),
                "最大单笔%": float(net.max()),
                "收益/波动": (
                    net.mean() / net.std(ddof=1)
                    if len(net) > 1 and net.std(ddof=1) > 0
                    else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("平均收益%", ascending=False).reset_index(drop=True)
    return result


def capture_ratio_table(signals: pd.DataFrame, cost_pct: float):
    """各规则吃到了理论最大涨幅的多少——直接衡量"留住黄金"的能力。"""
    if "理论最大涨幅%" not in signals.columns:
        return pd.DataFrame()
    max_possible = pd.to_numeric(signals["理论最大涨幅%"], errors="coerce")
    labels = [c[3:] for c in signals.columns if c.startswith("收益_")]
    rows = []
    # 只在真正出现过大涨的样本上比较，否则会被大量平庸样本稀释
    big_mask = max_possible >= 30.0
    for label in labels:
        returns = pd.to_numeric(signals[f"收益_{label}"], errors="coerce") - cost_pct
        valid = returns.notna() & max_possible.notna() & (max_possible > 0)
        if valid.sum() == 0:
            continue
        ratio_all = (returns[valid] / max_possible[valid]).clip(-2, 2)
        big_valid = valid & big_mask
        rows.append(
            {
                "退出规则": label,
                "全部样本 捕获率%": float(ratio_all.mean() * 100.0),
                "大涨样本数": int(big_valid.sum()),
                "大涨样本 理论涨幅%": float(max_possible[big_valid].mean()),
                "大涨样本 实际收益%": float(returns[big_valid].mean()),
                "大涨样本 捕获率%": float(
                    (returns[big_valid] / max_possible[big_valid]).clip(-2, 2).mean()
                    * 100.0
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            "大涨样本 捕获率%", ascending=False
        ).reset_index(drop=True)
    return result


def exit_by_selection(signals: pd.DataFrame, cost_pct: float, top_n: int):
    """在实际选股条件下（每周按回撤最深取TopN）对比各退出规则。"""
    if signals.empty:
        return pd.DataFrame()
    ranked = signals.groupby("Entry_Week")["Drawdown_26W_pct"].rank(
        method="first", ascending=True
    )
    picked = signals[ranked <= top_n]
    if picked.empty:
        return pd.DataFrame()
    labels = [c[3:] for c in signals.columns if c.startswith("收益_")]
    rows = []
    for label in labels:
        returns = pd.to_numeric(picked[f"收益_{label}"], errors="coerce").dropna()
        if returns.empty:
            continue
        net = returns - cost_pct
        rows.append(
            {
                "退出规则": label,
                f"Top{top_n}样本数": int(len(net)),
                "平均收益%": float(net.mean()),
                "中位收益%": float(net.median()),
                "胜率%": float((net > 0).mean() * 100.0),
                "赚>50%比例": float((net > 50).mean() * 100.0),
                "亏>20%比例": float((net < -20).mean() * 100.0),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("平均收益%", ascending=False).reset_index(drop=True)
    return result


def yearly_exit_table(signals: pd.DataFrame, cost_pct: float):
    work = signals.copy()
    work["年份"] = work["Signal_Week"].astype(str).str[:4]
    labels = [c[3:] for c in work.columns if c.startswith("收益_")]
    rows = []
    for year, group in work.groupby("年份"):
        row = {"年份": year, "信号数": len(group)}
        for label in labels:
            row[label] = float(
                (pd.to_numeric(group[f"收益_{label}"], errors="coerce") - cost_pct).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🎯 {APP_TITLE}")
    st.caption("同一批信号、同一个买入价，只改卖出方式——看能不能把大波段留住。")
    st.info(
        "**为什么测这个**：此前十几轮全在优化买什么、何时买，"
        "退出一直雷打不动固定3周。但固定持有期会**把所有大赢家提前砍断**——"
        "这可能正是157笔里剔掉最好5笔就归零的原因。\n\n"
        "**这次改的是收益分布的形状**，不是平均值：让赚钱的单子跑得更远，"
        "亏钱的单子照样早砍。所以判断标准不只看平均收益，"
        "更要看**赚>50%的比例**和**大涨样本捕获率**。\n\n"
        "所有规则共用同一个买入价和同一段行情，且统一按**周收盘价**成交，"
        "不假设能卖在盘中最高点。"
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
        st.subheader("买入信号（已验证配置）")
        n_period = st.number_input("N", value=4, min_value=2, max_value=60, step=1)
        m_period = st.number_input("M", value=3, min_value=2, max_value=30, step=1)
        level = st.number_input("K值阈值", value=20.0, min_value=1.0, max_value=90.0, step=5.0)
        require_kd = st.checkbox("要求 K > D", value=True)

        st.markdown("---")
        st.subheader("退出规则参数")
        max_weeks = st.number_input(
            "最长持有周数（超过则强制卖出）", value=26, min_value=4, max_value=104, step=2,
            help="给移动止损、SKDJ高位等规则一个上限，避免无限持有。",
        )
        trail_pct = st.number_input(
            "移动止损：从最高收盘价回撤%", value=15.0, min_value=5.0, max_value=50.0, step=5.0,
        )
        skdj_top = st.number_input(
            "SKDJ高位线（K达到即卖）", value=75.0, min_value=50.0, max_value=95.0, step=5.0,
            help="你图中指标的顶部线就是75。",
        )
        ma_exit_weeks = st.number_input(
            "跌破几周均线卖出", value=10, min_value=3, max_value=40, step=1,
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
        if st.session_state.get("exit_result"):
            return
        st.markdown(
            """
### 对比的退出方式

| 规则 | 说明 |
|---|---|
| 固定持有3/4/6/10/13周 | 到期就卖，现在的做法是3周 |
| 移动止损15% | 从持有期最高收盘价回撤15%才卖，让利润奔跑 |
| 持有到SKDJ≥75 | 吃完一整个波浪再走（对应你图中的顶部线） |
| 跌破10周均线 | 趋势走坏才卖 |

### 四张表

**表1 · 退出规则总对比**　除了平均收益，重点看**赚>50%比例**和**亏>20%比例**。

**表2 · 大涨捕获率**　在那些理论上确实出现过30%以上涨幅的交易里，
每种规则实际吃到了多少。**这张表直接回答"能不能留住黄金"。**

**表3 · 实际选股条件下的对比**　每周按回撤最深取Top3，更接近真实操作。

**表4 · 分年度**　确认结论不是靠某一年撑起来的。

---
**你需要有心理准备**：如果移动止损或SKDJ高位退出确实更好，
那意味着平均持有周数会从3周拉长到可能十几周，交易节奏完全改变。
表1的"平均持有周数"一列会告诉你具体是多久。
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
    fetch_start = (pd.Timestamp(start_input) - timedelta(days=500)).strftime("%Y%m%d")
    fetch_end = (
        pd.Timestamp(end_input) + timedelta(days=int(max_weeks) * 7 + 60)
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

    fixed_weeks_list = [3, 4, 6, 10, 13]
    progress = st.progress(0.0, text="模拟各种退出规则……")
    parts = []
    codes = sorted(stocks.keys())
    for idx, ts_code in enumerate(codes):
        weekly = build_weekly_bars(stocks[ts_code])
        if weekly.empty or len(weekly) < 60:
            continue
        rows = build_signals_with_exits(
            weekly, ts_code, int(n_period), int(m_period), float(level),
            bool(require_kd), int(max_weeks), float(trail_pct),
            float(skdj_top), int(ma_exit_weeks), fixed_weeks_list,
        )
        if not rows.empty:
            parts.append(rows)
        if idx % 40 == 0:
            progress.progress(
                min((idx + 1) / len(codes), 1.0),
                text=f"模拟各种退出规则……{idx + 1}/{len(codes)}",
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
    signals = signals[
        pd.to_numeric(signals["Entry_Price"], errors="coerce") >= min_price
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
    signals = signals.reset_index(drop=True)
    if signals.empty:
        st.error("过滤后无信号。")
        return

    st.session_state["exit_result"] = {
        "signals": signals,
        "compare": compare_exit_rules(signals, float(cost_pct)),
        "capture": capture_ratio_table(signals, float(cost_pct)),
        "selection": exit_by_selection(signals, float(cost_pct), int(top_n)),
        "yearly": yearly_exit_table(signals, float(cost_pct)),
        "params": {
            "最长持有": int(max_weeks), "移动止损": float(trail_pct),
            "SKDJ高位": float(skdj_top), "均线": int(ma_exit_weeks),
            "每周选": int(top_n),
        },
    }


def render_results():
    result = st.session_state.get("exit_result")
    if not result:
        return False
    params = result["params"]
    signals = result["signals"]

    st.markdown("---")
    st.header("退出规则对比结果")
    st.caption(
        f"信号 {len(signals):,} 笔，覆盖 {signals['Signal_Week'].min()} — "
        f"{signals['Signal_Week'].max()}　|　最长持有{params['最长持有']}周　"
        f"移动止损{params['移动止损']:.0f}%　SKDJ高位{params['SKDJ高位']:.0f}"
    )

    st.subheader("表1 · 退出规则总对比")
    st.dataframe(result["compare"].round(2), width="stretch", hide_index=True)
    st.caption(
        "**不要只看平均收益。**「赚>50%比例」反映能否留住大波段，"
        "「亏>20%比例」反映代价，「平均持有周数」决定你的交易节奏是否能接受，"
        "「收益/波动」是风险调整后的综合比较。"
    )

    if not result["capture"].empty:
        st.subheader("表2 · 大涨捕获率（这张表直接回答能否留住黄金）")
        st.dataframe(result["capture"].round(2), width="stretch", hide_index=True)
        st.caption(
            "只在**理论最大涨幅≥30%**的那批交易上比较——这些就是真正的黄金。"
            "「大涨样本 捕获率%」= 实际收益 ÷ 理论最大涨幅。"
            "固定3周的捕获率如果很低，就证实了它在系统性地砍断大赢家。"
        )

    if not result["selection"].empty:
        st.subheader("表3 · 实际选股条件下（每周回撤最深Top3）")
        st.dataframe(result["selection"].round(2), width="stretch", hide_index=True)
        st.caption("更接近真实操作的结果。")

    if not result["yearly"].empty:
        st.subheader("表4 · 分年度")
        st.dataframe(result["yearly"].round(2), width="stretch", hide_index=True)
        st.caption("确认最优规则不是靠某一年撑起来的。")

    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_exit_comparison.csv",
            result["compare"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "02_capture_ratio.csv",
            result["capture"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "03_with_selection.csv",
            result["selection"].to_csv(index=False, encoding="utf-8-sig"),
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
        file_name="exit_rule_validation.zip",
        mime="application/zip",
        key="download_exit",
    )
    return True


if __name__ == "__main__":
    main()
    render_results()
