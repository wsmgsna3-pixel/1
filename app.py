# -*- coding: utf-8 -*-
"""样本外测试器（单文件独立版，直接覆盖 app.py 运行）。

背景与目的：
突破策略在2022-2026期间表现为：翻倍概率15.41%（全池基准7.33%，提升2.10倍），
移动止损后收益6.47%。但77%的信号（988/1285）集中在2025-2026两年，
所谓"四年回测"在信息量上其实只有两年。

更值得注意的是：2024年全市场+4.12%而该策略-3.91%；
2026年全市场-2.29%而该策略+8.16%。说明它不是单纯的牛市beta，
2025-2026与2022-2024之间存在某种市场结构差异。

但用同一份数据无法分辨以下两种可能：
  (a) 过拟合——参数是在这份数据上试出来的，换时期即失效
  (b) 行情依赖——逻辑真实，但只在特定市场结构下有效

唯一的分辨方法是拿全新的数据做样本外测试。本工具即为此设计：
策略参数全部锁死为模块级常量，只允许修改测试区间。

使用纪律：绝对不要在新数据上调整这些参数。样本外测试的全部价值就在于
"规则不动，只换数据"；一旦调参，就等于又做了一轮样本内拟合，测试立即失效。

内存优化：边构建周线边释放日线，避免两份完整数据共存
（Streamlit Cloud 上限约1GB，超出会被静默杀死、需重新部署）。

行情缓存与之前共用，但测试早期区间需要额外下载历史数据。
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

APP_TITLE = "样本外测试：规则冻结，只换数据"
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
# 冻结的策略规则 —— 这些数值来自 2022-2026 样本内测试，本工具不允许修改。
# 样本外测试的全部意义就在于"规则不动，只换数据"。
# 一旦在新数据上调参数，这次测试就失去价值，等于又做了一轮样本内拟合。
# =============================================================================
FROZEN_BREAKOUT_WEEKS = 26          # 突破26周新高
FROZEN_POSITION_QUANTILE = 0.33     # 只取接近两年高点的前33%
FROZEN_VOL_CONTRACTION_MAX = 0.8    # 波动率压缩 ≤0.8
FROZEN_FORWARD_WEEKS = 26           # 前瞻26周
FROZEN_STOP_PCT = 15.0              # 移动止损15%（实盘可承受范围）
FROZEN_ENTRY = "立即买入"            # 突破次周开盘买入（洗盘验证已证明等回撤更差）

# 2022-2026 样本内基准结果，用于并排对照
IN_SAMPLE_REFERENCE = {
    "期间": "2022-2026（样本内，已反复优化）",
    "信号数": 1285,
    "翻倍概率%": 15.41,
    "全池基准翻倍概率%": 7.33,
    "提升倍数": 2.10,
    "止损后收益%": 6.47,
    "止损后胜率%": 38.75,
    "涨超50%比例": 36.34,
}


def compute_features(weekly: pd.DataFrame):
    close = pd.to_numeric(weekly["close"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    return_1w = (close / close.shift(1) - 1.0) * 100.0
    features = pd.DataFrame(index=weekly.index)
    features["breakout"] = (
        close > close.shift(1).rolling(FROZEN_BREAKOUT_WEEKS).max()
    )
    features["position_2y"] = close / high.shift(1).rolling(104).max().replace(0, np.nan)
    vol_recent = return_1w.shift(1).rolling(8).std()
    vol_earlier = return_1w.shift(9).rolling(18).std()
    features["vol_contraction"] = vol_recent / vol_earlier.replace(0, np.nan)
    return features


def evaluate_stock(
    weekly: pd.DataFrame, ts_code: str, position_threshold: float
) -> pd.DataFrame:
    """对单只股票：算出全部周的基准结果 + 符合冻结规则的信号结果。

    基准 = 该股所有周（无条件买入），用于计算"随机买入"的翻倍概率。
    信号 = 满足 突破+接近高点+波动压缩 的周。
    两者用完全相同的前瞻与止损口径，可直接比较。
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
        for j in range(i + 1, stop_index + 1):
            current = close_values[j]
            if not math.isfinite(current):
                continue
            peak = max(peak, current)
            if current <= peak * (1.0 - FROZEN_STOP_PCT / 100.0):
                exit_price = current
                break
        if exit_price is None:
            exit_price = close_values[stop_index]
        if not math.isfinite(exit_price):
            continue
        trail_return = (exit_price / entry_price - 1.0) * 100.0

        is_signal = bool(
            breakout[i]
            and math.isfinite(position[i])
            and position[i] >= position_threshold
            and math.isfinite(contraction[i])
            and contraction[i] <= FROZEN_VOL_CONTRACTION_MAX
        )
        rows.append(
            {
                "ts_code": ts_code,
                "Week": dates[i],
                "Is_Signal": is_signal,
                "Entry_Price": entry_price,
                "Max_Gain_pct": max_gain,
                "Trail_Return_pct": trail_return,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 汇总
# -----------------------------------------------------------------------------
def summarize(panel: pd.DataFrame, cost_pct: float, label: str):
    gain = pd.to_numeric(panel["Max_Gain_pct"], errors="coerce")
    trail = pd.to_numeric(panel["Trail_Return_pct"], errors="coerce") - cost_pct
    valid = gain.notna() & trail.notna()
    gain, trail = gain[valid], trail[valid]
    if gain.empty:
        return {}
    return {
        "组别": label,
        "样本数": int(len(gain)),
        "涨超50%比例": float((gain > 50).mean() * 100.0),
        "翻倍概率%": float((gain > 100).mean() * 100.0),
        "止损后收益%": float(trail.mean()),
        "止损后中位%": float(trail.median()),
        "止损后胜率%": float((trail > 0).mean() * 100.0),
    }


def main_comparison(panel: pd.DataFrame, cost_pct: float):
    signal = panel[panel["Is_Signal"].astype(bool)]
    baseline = panel
    rows = []
    base_stats = summarize(baseline, cost_pct, "全池基准（所有周无条件买入）")
    signal_stats = summarize(signal, cost_pct, "冻结规则信号")
    if base_stats:
        rows.append(base_stats)
    if signal_stats:
        rows.append(signal_stats)
    table = pd.DataFrame(rows)
    if len(table) == 2:
        lift = pd.Series(
            {
                "组别": "提升倍数（信号÷基准）",
                "样本数": np.nan,
                "涨超50%比例": table["涨超50%比例"].iloc[1] / table["涨超50%比例"].iloc[0]
                if table["涨超50%比例"].iloc[0]
                else np.nan,
                "翻倍概率%": table["翻倍概率%"].iloc[1] / table["翻倍概率%"].iloc[0]
                if table["翻倍概率%"].iloc[0]
                else np.nan,
                "止损后收益%": np.nan,
                "止损后中位%": np.nan,
                "止损后胜率%": np.nan,
            }
        )
        table = pd.concat([table, lift.to_frame().T], ignore_index=True)
    return table


def yearly_table(panel: pd.DataFrame, cost_pct: float):
    work = panel.copy()
    work["年份"] = work["Week"].astype(str).str[:4]
    rows = []
    for year, group in work.groupby("年份"):
        signal = group[group["Is_Signal"].astype(bool)]
        base_gain = pd.to_numeric(group["Max_Gain_pct"], errors="coerce").dropna()
        if signal.empty or base_gain.empty:
            continue
        gain = pd.to_numeric(signal["Max_Gain_pct"], errors="coerce").dropna()
        trail = (
            pd.to_numeric(signal["Trail_Return_pct"], errors="coerce").dropna() - cost_pct
        )
        base_double = float((base_gain > 100).mean() * 100.0)
        signal_double = float((gain > 100).mean() * 100.0) if len(gain) else np.nan
        rows.append(
            {
                "年份": year,
                "信号数": int(len(signal)),
                "信号翻倍概率%": signal_double,
                "基准翻倍概率%": base_double,
                "提升倍数": (
                    signal_double / base_double if base_double and base_double > 0 else np.nan
                ),
                "止损后收益%": float(trail.mean()) if len(trail) else np.nan,
                "止损后胜率%": float((trail > 0).mean() * 100.0) if len(trail) else np.nan,
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
    st.title(f"🧊 {APP_TITLE}")
    st.caption("规则完全冻结，只换数据——这是唯一能分辨真规律与过拟合的方法。")
    st.error(
        "**使用纪律：这个工具里的策略参数是锁死的，请不要修改代码去调它们。**\n\n"
        "样本外测试的全部价值就在于「规则不动，只换数据」。"
        "一旦在新数据上试参数，这次测试立刻失效，等于又做了一轮样本内拟合。"
        "如果结果不好，那就是不好——那本身就是最有价值的信息。"
    )
    st.info(
        "**冻结的规则**（来自2022-2026样本内测试）：\n\n"
        f"- 突破 {FROZEN_BREAKOUT_WEEKS} 周新高\n"
        f"- 价格接近两年高点（前 {FROZEN_POSITION_QUANTILE*100:.0f}%）\n"
        f"- 波动率压缩 ≤ {FROZEN_VOL_CONTRACTION_MAX}\n"
        f"- 突破次周开盘买入（洗盘验证已证明等回撤更差）\n"
        f"- 移动止损 {FROZEN_STOP_PCT:.0f}%，最长持有 {FROZEN_FORWARD_WEEKS} 周\n\n"
        "**要检验的问题**：2022-2026期间该规则翻倍概率15.41%，是基准7.33%的2.1倍。"
        "但77%的信号集中在2025-2026两年——这究竟是真规律，"
        "还是只在最近两年的市场结构下成立？"
    )

    with st.sidebar:
        st.header("配置")
        try:
            secret_token = st.secrets.get("TUSHARE_TOKEN", "")
        except Exception:
            secret_token = ""
        token_input = st.text_input("Tushare Token", value=secret_token, type="password")

        st.markdown("---")
        st.subheader("测试区间（这是唯一该改的东西）")
        st.caption("建议先跑 2018-01-01 至 2022-08-31，这段我们从未碰过。")
        start_input = st.date_input("开始日期", value=date(2018, 1, 1))
        end_input = st.date_input("结束日期", value=date(2022, 8, 31))

        st.markdown("---")
        st.subheader("股票池（与样本内保持一致）")
        min_price = st.number_input("最低股价（元）", value=10.0, min_value=0.0, step=1.0)
        min_mv = st.number_input("最低流通市值（亿元）", value=100.0, min_value=0.0, step=10.0)
        max_mv = st.number_input("最高流通市值（亿元）", value=1000.0, min_value=100.0, step=100.0)
        cost_pct = st.number_input("往返成本%", value=0.20, min_value=0.0, max_value=2.0, step=0.05)

        st.markdown("---")
        clear_cache_clicked = st.button("清空行情缓存")
        run_clicked = st.button("开始样本外测试", type="primary")

    if clear_cache_clicked:
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.success("行情缓存已清空。")

    if not run_clicked:
        if st.session_state.get("oos_result"):
            return
        st.markdown(
            """
### 结果会怎么读

跑完会得到「新区间」的三个数字，和2022-2026并排对照：

| 情形 | 含义 | 该怎么办 |
|---|---|---|
| 提升倍数 ≈ 2，收益为正 | **真规律**，跨越完全不同的市场环境仍成立 | 可以认真考虑 |
| 提升倍数 1.2~1.5 | 有效但弱，之前的2.1倍含运气成分 | 降低预期，小仓位试 |
| 提升倍数 ≈ 1 或收益为负 | **要么过拟合，要么市场结构已变** | 见下方 |

### 如果最后一种情况发生

那还需要再分辨一次：

- **过拟合**：策略从来就没用过，2025-2026是撞上的
- **结构变化**：老规律在老市场有效、在新市场失效，或反过来

区分方法：看2018-2022内部的分年度。如果那几年里也有某一两年特别好、其余年份亏，
说明这个策略**一直都是靠特定行情吃饭**，只是每个时代的好年份不同——
那它就是个高波动的行情依赖型策略，而不是坏策略，
但你必须接受连亏几年的可能。

### 数据量提醒

需要额外下载2016年起的行情（位置特征要104周历史），
首次运行会比较慢。如果崩溃，把区间缩短到2-3年分次跑。
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
    if start_input >= end_input:
        st.error("开始日期必须早于结束日期。")
        return

    start_date = start_input.strftime("%Y%m%d")
    end_date = end_input.strftime("%Y%m%d")
    # 位置特征需104周历史，前瞻需26周
    fetch_start = (pd.Timestamp(start_input) - timedelta(days=900)).strftime("%Y%m%d")
    fetch_end = (pd.Timestamp(end_input) + timedelta(days=220)).strftime("%Y%m%d")

    with st.spinner("构建科技股研究池……"):
        whitelist_set, name_map, industry_map = load_custom_tech_whitelist(token_clean)
    if not whitelist_set:
        st.error("未取得研究池。")
        return
    st.success(f"科技股研究池：{len(whitelist_set)}只")

    with st.spinner("加载行情（首次跑早期区间需要下载，较慢）……"):
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
                columns={"trade_date_str": "Week"}
            )
        )
        mv_lookup["circ_mv"] = pd.to_numeric(
            mv_lookup["circ_mv"], errors="coerce"
        ).astype("float32")
        mv_lookup = mv_lookup.drop_duplicates(["Week", "ts_code"])
    else:
        mv_lookup = pd.DataFrame()
    del basic_indexed
    gc.collect()

    # 边建周线边释放日线，避免两份数据共存导致内存溢出
    needed = ["trade_date_str", "open", "high", "low", "close"]
    weekly_cache = {}
    position_samples = []
    codes = sorted(stocks.keys())
    prep = st.progress(0.0, text="构建周线并计算位置分布……")
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
    position_threshold = float(
        all_positions.quantile(1.0 - FROZEN_POSITION_QUANTILE)
    )
    del all_positions
    gc.collect()

    progress = st.progress(0.0, text="应用冻结规则……")
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
        st.error("没有产生数据。")
        return
    panel = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    panel = panel[(panel["Week"] >= start_date) & (panel["Week"] <= end_date)]
    panel = panel[pd.to_numeric(panel["Entry_Price"], errors="coerce") >= min_price]
    if not mv_lookup.empty:
        panel = panel.merge(mv_lookup, on=["Week", "ts_code"], how="left")
        mv = pd.to_numeric(panel["circ_mv"], errors="coerce") / 10000.0
        panel = panel[mv.between(min_mv, max_mv) | mv.isna()]
        del mv_lookup
        gc.collect()
    panel = panel.reset_index(drop=True)
    if panel.empty:
        st.error("过滤后无数据。")
        return

    used_mb = _memory_usage_mb()
    st.session_state["oos_result"] = {
        "panel_rows": len(panel),
        "signal_count": int(panel["Is_Signal"].astype(bool).sum()),
        "comparison": main_comparison(panel, float(cost_pct)),
        "yearly": yearly_table(panel, float(cost_pct)),
        "period": f"{start_date} — {end_date}",
        "memory_mb": used_mb,
    }


def render_results():
    result = st.session_state.get("oos_result")
    if not result:
        return False

    st.markdown("---")
    st.header("样本外测试结果")
    st.caption(
        f"测试区间 {result['period']}　|　"
        f"全池观测 {result['panel_rows']:,} 个「个股-周」，"
        f"其中符合冻结规则的信号 {result['signal_count']:,} 个"
        + (
            f"　|　内存 {result['memory_mb']:.0f} MB"
            if math.isfinite(result.get("memory_mb", float("nan")))
            else ""
        )
    )

    st.subheader("表1 · 新区间：信号 vs 全池基准")
    st.dataframe(result["comparison"].round(2), width="stretch", hide_index=True)

    st.subheader("表2 · 与样本内结果并排对照")
    reference = pd.DataFrame(
        [
            IN_SAMPLE_REFERENCE,
            {
                "期间": f"{result['period']}（样本外，规则冻结）",
                "信号数": result["signal_count"],
                "翻倍概率%": (
                    result["comparison"]["翻倍概率%"].iloc[1]
                    if len(result["comparison"]) > 1
                    else np.nan
                ),
                "全池基准翻倍概率%": (
                    result["comparison"]["翻倍概率%"].iloc[0]
                    if len(result["comparison"]) > 0
                    else np.nan
                ),
                "提升倍数": (
                    result["comparison"]["翻倍概率%"].iloc[2]
                    if len(result["comparison"]) > 2
                    else np.nan
                ),
                "止损后收益%": (
                    result["comparison"]["止损后收益%"].iloc[1]
                    if len(result["comparison"]) > 1
                    else np.nan
                ),
                "止损后胜率%": (
                    result["comparison"]["止损后胜率%"].iloc[1]
                    if len(result["comparison"]) > 1
                    else np.nan
                ),
                "涨超50%比例": (
                    result["comparison"]["涨超50%比例"].iloc[1]
                    if len(result["comparison"]) > 1
                    else np.nan
                ),
            },
        ]
    )
    st.dataframe(reference.round(2), width="stretch", hide_index=True)
    st.caption(
        "**这是全部结论所在。**样本内提升2.10倍、收益6.47%。"
        "如果样本外也接近这个水平，说明是真规律；"
        "如果明显衰减或转负，那么之前的结果就含有大量运气或过拟合成分。"
    )

    if not result["yearly"].empty:
        st.subheader("表3 · 新区间分年度")
        st.dataframe(result["yearly"].round(2), width="stretch", hide_index=True)
        st.caption(
            "**即使整体结果不好，也要看这里**：如果新区间内部也是"
            "某一两年特别好、其余年份亏，说明这个策略一直都靠特定行情吃饭，"
            "只是每个时代的好年份不同——那它是行情依赖型策略，不是坏策略，"
            "但必须接受连亏几年的可能。"
        )

    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_comparison.csv",
            result["comparison"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "02_in_vs_out_sample.csv",
            reference.to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "03_yearly.csv",
            result["yearly"].to_csv(index=False, encoding="utf-8-sig"),
        )
    st.download_button(
        "下载样本外测试结果",
        data=output.getvalue(),
        file_name="out_of_sample_validation.zip",
        mime="application/zip",
        key="download_oos",
    )
    return True


if __name__ == "__main__":
    main()
    render_results()
