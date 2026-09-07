# -*- coding: utf-8 -*-
"""突破策略实盘版（单文件独立版，直接覆盖 app.py 运行）。

不做回测，只做两件事：
  1. 本周选股——扫描全池，列出符合条件的股票并给出前3只的买入指引
  2. 持仓管理——输入已持仓，计算当前移动止损位、持有周数、是否该退出

规则来源（全部经独立验证，不要随意改动）：
  信号  突破26周新高 + 接近两年高点前33% + 波动率压缩≤0.8
        样本外2018-2022：提升1.57倍、收益7.53%、胜率43.28%
  排序  同周多个信号按流通市值从大到小
        十个候选变量中唯一三关全过：正向+1.31%/t=1.97、反向-2.02%、
        四年全部为正、TopN衰减平滑；12周持有下各仓位分位稳定在82-98%
  退出  移动止损15%，最长持有12周
        持有期对比：10-12周胜率约50%（26周时仅39%）

已知的策略性格（回测与样本外验证得出，使用前必须接受）：
  胜率约50%；收益靠每年一两只大赢家；五年里两年亏损；
  四年内约36%概率遇到5连亏；单笔最差可能超过止损线（跌停/跳空，实测有-33%）。

时点要求：本策略按周线运作，应在周线收盘后扫描，下周一开盘买入。
若在周中扫描，信号可能在周五收盘前变化，程序会给出提示。

行情缓存与之前共用；实盘版只需约3.5年数据，比回测轻量。
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

APP_TITLE = "突破策略实盘助手"
MARKET_CACHE_ROOT = "r1_trend_entry_market_cache_v2"
CACHE_SCHEMA_VERSION = 3
DOWNLOAD_WORKERS = 4
DATA_READY_HOUR_SHANGHAI = 18

# -----------------------------------------------------------------------------
# 数据层（与回测完全一致，保证实盘与回测口径统一）
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
# 冻结的实盘规则 —— 全部经过独立验证，不要随意改动
#
# 信号：突破26周新高 + 接近两年高点前33% + 波动率压缩≤0.8
#   依据：样本外(2018-2022)提升1.57倍、收益7.53%、胜率43.28%
# 排序：同一周多个信号时按流通市值从大到小
#   依据：十个候选变量中唯一三关全过（正向+1.31%/t=1.97、反向-2.02%、
#         四年全部为正、TopN衰减平滑），且12周持有下各仓位分位稳定在82-98%
# 退出：移动止损15%，最长持有10-12周
#   依据：持有期对比显示10-12周胜率约50%（26周时仅39%），分位稳定
# =============================================================================
BREAKOUT_WEEKS = 26
POSITION_QUANTILE = 0.33
VOL_CONTRACTION_MAX = 0.8
STOP_PCT = 15.0
MAX_HOLD_WEEKS = 12
SLOT_COUNT = 3
MIN_PRICE = 10.0
MIN_MV_BILLION = 100.0
MAX_MV_BILLION = 1000.0


def compute_features(weekly: pd.DataFrame):
    close = pd.to_numeric(weekly["close"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    return_1w = (close / close.shift(1) - 1.0) * 100.0
    features = pd.DataFrame(index=weekly.index)
    prior_high = close.shift(1).rolling(BREAKOUT_WEEKS).max()
    features["prior_high"] = prior_high
    features["breakout"] = close > prior_high
    features["position_2y"] = close / high.shift(1).rolling(104).max().replace(0, np.nan)
    vol_recent = return_1w.shift(1).rolling(8).std()
    vol_earlier = return_1w.shift(9).rolling(18).std()
    features["vol_contraction"] = vol_recent / vol_earlier.replace(0, np.nan)
    return features


def scan_latest(weekly: pd.DataFrame, ts_code: str, target_week: str):
    """只判断指定的那一周是否出信号，返回该股的全部判定明细。

    明细包含每一项条件的实际数值，方便你人工复核，而不是只给一个是/否。
    """
    if len(weekly) < 140:
        return None
    features = compute_features(weekly)
    dates = weekly["trade_date_str"].astype(str).tolist()
    if target_week not in dates:
        return None
    i = dates.index(target_week)
    close = pd.to_numeric(weekly["close"], errors="coerce")
    return {
        "ts_code": ts_code,
        "信号周": target_week,
        "收盘价": float(close.iloc[i]) if math.isfinite(close.iloc[i]) else np.nan,
        "前26周最高收盘": float(features["prior_high"].iloc[i]),
        "突破": bool(features["breakout"].iloc[i]),
        "两年高点位置": float(features["position_2y"].iloc[i]),
        "波动率压缩": float(features["vol_contraction"].iloc[i]),
    }


def position_status(weekly: pd.DataFrame, buy_week: str, buy_price: float):
    """已持仓个股的当前状态：移动止损位、持有周数、是否该退出。

    移动止损口径与回测完全一致：以买入后各周收盘价的最高值为基准，
    回撤达到15%即在该周收盘卖出。
    """
    dates = weekly["trade_date_str"].astype(str).tolist()
    close = pd.to_numeric(weekly["close"], errors="coerce").tolist()
    if buy_week not in dates:
        # 买入周可能不是周线最后交易日，取其后最近的一周
        later = [d for d in dates if d >= buy_week]
        if not later:
            return None
        buy_week = later[0]
    start = dates.index(buy_week)
    peak = buy_price
    triggered_week = None
    for j in range(start, len(dates)):
        current = close[j]
        if not math.isfinite(current):
            continue
        peak = max(peak, current)
        if current <= peak * (1.0 - STOP_PCT / 100.0) and triggered_week is None:
            triggered_week = dates[j]
    latest_close = next(
        (close[j] for j in range(len(close) - 1, -1, -1) if math.isfinite(close[j])),
        np.nan,
    )
    held_weeks = len(dates) - start
    return {
        "持有周数": held_weeks,
        "最新收盘": latest_close,
        "期间最高收盘": peak,
        "当前移动止损位": peak * (1.0 - STOP_PCT / 100.0),
        "浮动盈亏%": (latest_close / buy_price - 1.0) * 100.0
        if math.isfinite(latest_close) and buy_price > 0
        else np.nan,
        "距止损位%": (latest_close / (peak * (1.0 - STOP_PCT / 100.0)) - 1.0) * 100.0
        if math.isfinite(latest_close)
        else np.nan,
        "已触发止损周": triggered_week,
        "是否到期": held_weeks >= MAX_HOLD_WEEKS,
    }


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
def render_rules():
    st.markdown(
        f"""
| 环节 | 规则 |
|---|---|
| 股票池 | 科技股（主板/创业板/科创板），流通市值 {MIN_MV_BILLION:.0f}-{MAX_MV_BILLION:.0f}亿，股价≥{MIN_PRICE:.0f}元 |
| 信号 | 周线收盘突破前{BREAKOUT_WEEKS}周最高收盘 |
| 条件① | 价格接近两年高点（全池前{POSITION_QUANTILE*100:.0f}%） |
| 条件② | 波动率压缩 ≤ {VOL_CONTRACTION_MAX} |
| 排序 | 同周多个信号时，按流通市值从大到小取前{SLOT_COUNT}只 |
| 买入 | 信号周的下一周开盘 |
| 止损 | 移动止损{STOP_PCT:.0f}%（以持有期内最高周收盘为基准） |
| 持有 | 最长{MAX_HOLD_WEEKS}周，到期卖出 |
"""
    )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"📋 {APP_TITLE}")
    st.caption("每周选股 + 持仓管理。规则已冻结，不做回测。")

    with st.expander("查看冻结的完整规则", expanded=False):
        render_rules()

    with st.expander("⚠️ 使用前必读：这套策略的真实性格", expanded=False):
        st.markdown(
            """
这些数字来自四年回测与2018-2022样本外验证，**请在开始前就接受它们**，
而不是等发生时才判断策略是否失效：

- **胜率约50%**，一半交易是亏的
- **多数交易赚得少，收益靠每年一两只大赢家**（单笔中位数常年接近0）
- **五年里两年是亏的**（2022约-8%，2023约-0.5%）
- **四年内有约36%的概率遇到5连亏**
- 单笔最差可能超过止损线（跌停/跳空），回测中出现过-33%
- 3仓满仓时，单笔止损=账户-5%；**如需降低冲击，可每仓只用20%资金**

**最危险的时刻不是连亏，而是连亏之后改规则。** 建议事先写下
"连亏5次、回撤X%以内属正常"，并在那之前不做任何修改。
            """
        )

    with st.sidebar:
        st.header("配置")
        try:
            secret_token = st.secrets.get("TUSHARE_TOKEN", "")
        except Exception:
            secret_token = ""
        token_input = st.text_input("Tushare Token", value=secret_token, type="password")
        mode = st.radio("功能", ["本周选股", "持仓管理"], index=0)
        st.markdown("---")
        st.caption("规则已锁死。如需调整请重新回测验证，不要直接改。")
        run_clicked = st.button("运行", type="primary")
        st.markdown("---")
        if st.button("清空行情缓存"):
            if os.path.isdir(MARKET_CACHE_ROOT):
                shutil.rmtree(MARKET_CACHE_ROOT)
            st.success("已清空。")

    if mode == "持仓管理":
        st.subheader("持仓管理")
        st.caption(
            "填入你的持仓，计算当前移动止损位与是否该退出。"
            "买入周填该笔交易买入那一周的任意日期即可（格式YYYYMMDD）。"
        )
        default = pd.DataFrame(
            {"股票代码": ["", "", ""], "买入周": ["", "", ""], "买入价": [0.0, 0.0, 0.0]}
        )
        holdings = st.data_editor(
            default, num_rows="dynamic", width="stretch", key="holdings_editor"
        )
        st.session_state["holdings_input"] = holdings

    if not run_clicked:
        if st.session_state.get("live_result"):
            render_results()
        else:
            st.info("填好Token后点击左侧「运行」。")
        return

    token_clean = clean_token_str(token_input)
    valid, message = verify_token_connection(token_clean)
    if not valid:
        st.error(f"Token校验失败：{message}")
        return

    data_ready = _latest_data_ready_date()
    fetch_end = data_ready.strftime("%Y%m%d")
    fetch_start = (data_ready - timedelta(days=1300)).strftime("%Y%m%d")

    with st.spinner("构建科技股研究池……"):
        whitelist_set, name_map, industry_map = load_custom_tech_whitelist(token_clean)
    if not whitelist_set:
        st.error("未取得研究池。")
        return

    with st.spinner("加载行情……"):
        stocks, basic_indexed, _, _, failed_dates, sync_stats = load_optimized_market_data(
            fetch_start, fetch_end, token_clean, tuple(sorted(whitelist_set))
        )
    if not stocks:
        st.error("未加载到行情。")
        return

    latest_mv = {}
    if not basic_indexed.empty and "circ_mv" in basic_indexed.columns:
        mv_reset = basic_indexed[["circ_mv"]].reset_index()
        mv_reset = mv_reset.sort_values("trade_date_str")
        for ts_code, group in mv_reset.groupby("ts_code"):
            value = pd.to_numeric(group["circ_mv"], errors="coerce").dropna()
            if len(value):
                latest_mv[str(ts_code)] = float(value.iloc[-1]) / 10000.0
    del basic_indexed
    gc.collect()

    # ---- 持仓管理 ----
    if mode == "持仓管理":
        holdings = st.session_state.get("holdings_input", pd.DataFrame())
        rows = []
        for _, item in holdings.iterrows():
            code = str(item.get("股票代码", "")).strip()
            buy_week = parse_yyyymmdd(item.get("买入周"))
            buy_price = _safe_float(item.get("买入价"))
            if not code or not buy_week or not math.isfinite(buy_price) or buy_price <= 0:
                continue
            daily = stocks.get(code)
            if daily is None:
                rows.append({"股票代码": code, "状态": "未找到行情（是否在科技股池内？）"})
                continue
            weekly = build_weekly_bars(daily)
            status = position_status(weekly, buy_week, buy_price)
            if status is None:
                rows.append({"股票代码": code, "状态": "数据不足"})
                continue
            action = "继续持有"
            if status["已触发止损周"]:
                action = f"⚠️ 已触发止损（{status['已触发止损周']}）"
            elif status["是否到期"]:
                action = f"⚠️ 已达最长持有{MAX_HOLD_WEEKS}周，应卖出"
            rows.append(
                {
                    "股票代码": code,
                    "名称": name_map.get(code, ""),
                    "买入价": round(buy_price, 2),
                    "最新收盘": round(status["最新收盘"], 2),
                    "浮动盈亏%": round(status["浮动盈亏%"], 2),
                    "持有周数": status["持有周数"],
                    "期间最高收盘": round(status["期间最高收盘"], 2),
                    "当前止损位": round(status["当前移动止损位"], 2),
                    "距止损位%": round(status["距止损位%"], 2),
                    "建议": action,
                }
            )
        st.session_state["live_result"] = {
            "mode": "持仓管理",
            "table": pd.DataFrame(rows),
            "data_through": fetch_end,
        }
        del stocks
        gc.collect()
        render_results()
        return

    # ---- 本周选股 ----
    progress = st.progress(0.0, text="扫描全池……")
    weekly_cache = {}
    position_values = []
    codes = sorted(stocks.keys())
    latest_weeks = []
    for idx, ts_code in enumerate(codes):
        daily = stocks.pop(ts_code)
        weekly = build_weekly_bars(daily)
        del daily
        if weekly.empty or len(weekly) < 140:
            continue
        weekly_cache[ts_code] = weekly
        latest_weeks.append(weekly["trade_date_str"].astype(str).iloc[-1])
        features = compute_features(weekly)
        values = features.loc[features["breakout"].fillna(False), "position_2y"]
        position_values.append(values.dropna())
        del features
        if idx % 60 == 0:
            progress.progress(min((idx + 1) / len(codes), 1.0))
    progress.empty()
    del stocks
    gc.collect()

    if not weekly_cache:
        st.error("数据不足。")
        return

    target_week = max(latest_weeks)
    all_positions = pd.concat(position_values, ignore_index=True)
    position_threshold = float(all_positions.quantile(1.0 - POSITION_QUANTILE))
    del all_positions, position_values
    gc.collect()

    records = []
    for ts_code, weekly in weekly_cache.items():
        info = scan_latest(weekly, ts_code, target_week)
        if info is None:
            continue
        mv = latest_mv.get(ts_code, np.nan)
        info["流通市值(亿)"] = mv
        info["名称"] = name_map.get(ts_code, "")
        info["行业"] = industry_map.get(ts_code, "")
        records.append(info)
    del weekly_cache
    gc.collect()

    frame = pd.DataFrame(records)
    if frame.empty:
        st.error("无数据。")
        return

    qualified = frame[
        frame["突破"]
        & (frame["两年高点位置"] >= position_threshold)
        & (frame["波动率压缩"] <= VOL_CONTRACTION_MAX)
        & (frame["收盘价"] >= MIN_PRICE)
        & frame["流通市值(亿)"].between(MIN_MV_BILLION, MAX_MV_BILLION)
    ].copy()
    qualified = qualified.sort_values("流通市值(亿)", ascending=False).reset_index(drop=True)
    qualified.insert(0, "排名", range(1, len(qualified) + 1))

    st.session_state["live_result"] = {
        "mode": "本周选股",
        "target_week": target_week,
        "data_through": fetch_end,
        "position_threshold": position_threshold,
        "pool_size": len(frame),
        "qualified": qualified,
        "memory_mb": _memory_usage_mb(),
    }
    render_results()


def render_results():
    result = st.session_state.get("live_result")
    if not result:
        return False

    if result["mode"] == "持仓管理":
        st.markdown("---")
        st.subheader("持仓状态")
        st.caption(f"行情数据截至 {result['data_through']}")
        if result["table"].empty:
            st.info("没有有效持仓记录。请在上方表格填入股票代码、买入周、买入价。")
        else:
            st.dataframe(result["table"], width="stretch", hide_index=True)
            st.caption(
                f"**止损位每周更新**：以买入后各周最高收盘价为基准下移{STOP_PCT:.0f}%。"
                "「距止损位%」为正表示还有缓冲，为负表示已跌破（应卖出）。"
            )
        return True

    st.markdown("---")
    st.header("本周选股结果")
    st.caption(
        f"信号周：**{result['target_week']}**　|　行情截至 {result['data_through']}　|　"
        f"扫描 {result['pool_size']} 只　|　"
        f"两年高点位置门槛 ≥{result['position_threshold']:.3f}"
    )
    st.warning(
        f"**请确认信号周 {result['target_week']} 已经收盘。** "
        "如果这一周还没结束，信号可能在周五收盘前发生变化。"
        "本策略按周线运作，应在周线收盘后（周五收盘或周末）扫描，下周一开盘买入。"
    )

    qualified = result["qualified"]
    if qualified.empty:
        st.info("本周没有符合条件的股票。空仓等待，不要降低标准。")
        return True

    st.subheader(f"符合全部条件的股票（共 {len(qualified)} 只）")
    show = qualified[
        ["排名", "股票代码" if "股票代码" in qualified.columns else "ts_code",
         "名称", "行业", "流通市值(亿)", "收盘价", "前26周最高收盘",
         "两年高点位置", "波动率压缩"]
    ].rename(columns={"ts_code": "股票代码"})
    st.dataframe(show.round(3), width="stretch", hide_index=True)

    st.subheader(f"🎯 建议买入（市值最大的前{SLOT_COUNT}只）")
    top = qualified.head(SLOT_COUNT)
    for _, row in top.iterrows():
        close_price = row["收盘价"]
        st.markdown(
            f"""
**{row['排名']}. {row.get('名称', '')}　{row['ts_code']}**　
行业：{row.get('行业', '—')}　流通市值：{row['流通市值(亿)']:.0f}亿

- 信号周收盘价：**{close_price:.2f}**（突破前26周最高 {row['前26周最高收盘']:.2f}）
- **买入方式：下周第一个交易日开盘价买入**
- **初始止损：按实际成交价 × 0.85**（例如成交价{close_price:.2f}则止损{close_price * 0.85:.2f}）
- 之后每周更新：以持有期内最高周收盘价 × 0.85 上移止损
- 最长持有 {MAX_HOLD_WEEKS} 周，到期无条件卖出
"""
        )

    st.info(
        f"**仓位提醒**：策略设计为{SLOT_COUNT}个仓位。如果当前已有持仓，"
        f"只买入空缺的仓位数量，按上面的排名顺序补。"
        f"**不要为了买满而降低排名标准，也不要因为看好某只而超配。**"
    )

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"signals_{result['target_week']}.csv",
            qualified.to_csv(index=False, encoding="utf-8-sig"),
        )
    st.download_button(
        "下载本周信号明细",
        data=output.getvalue(),
        file_name=f"weekly_signals_{result['target_week']}.zip",
        mime="application/zip",
        key="download_live",
    )
    return True


if __name__ == "__main__":
    main()
