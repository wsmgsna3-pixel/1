# -*- coding: utf-8 -*-
"""市场择时层验证器（单文件独立版，直接覆盖 app.py 运行）。

这一层不选股，只回答：什么样的市场状态下，未来3-4周整体就不该买。

为什么先做这一层：
此前四年回测显示，2023年509个信号整年亏损（-1.53%/笔），
无论换哪种入场方式、哪个排序因子都救不回来——那一年的问题不是选错股，
是根本不该重仓。若能识别这类时期，收益改善空间可能大于继续优化选股因子。

设计要点：
- 完全独立于选股层：指标只用全池截面信息，预测"全市场未来N周平均收益"，
  因此可以单独验证，不会和选股效果混在一起说不清谁的功劳。
- 无未来函数：所有状态指标在当周收盘即可算出，预测的是之后N周。
- 未来收益口径与之前一致：下一周开盘买入，第N周收盘卖出。

必须正视的限制：
四年约200周，且持有期重叠，真正独立的观测只有约70个。择时层的统计功效
天然远低于选股层。因此判断标准侧重"经济逻辑 + 分年度一致性 + 五分组单调性"，
而不是单一t值或收益数字。

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

APP_TITLE = "市场择时层验证"
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
def build_stock_weekly_panel(
    weekly: pd.DataFrame, ts_code: str, hold_weeks: int
) -> pd.DataFrame:
    """每只股票逐周一行：本周状态 + 未来hold_weeks周收益。

    未来收益口径与之前完全一致：下一周开盘买入，第hold_weeks周收盘卖出。
    状态列只用本周及之前数据，不含未来函数。
    """
    close = pd.to_numeric(weekly["close"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    open_price = pd.to_numeric(weekly["open"], errors="coerce")
    dates = weekly["trade_date_str"].astype(str)

    return_1w = (close / close.shift(1) - 1.0) * 100.0
    drawdown_26w = (close / high.rolling(26).max() - 1.0) * 100.0
    entry_open = open_price.shift(-1)
    exit_close = close.shift(-hold_weeks)

    frame = pd.DataFrame(
        {
            "ts_code": ts_code,
            "Week": dates,
            "Return_1W_pct": return_1w,
            "Drawdown_26W_pct": drawdown_26w,
            "Fwd_Return_pct": (
                exit_close / entry_open.replace(0, np.nan) - 1.0
            ) * 100.0,
        }
    )
    return frame.dropna(subset=["Return_1W_pct"])


# -----------------------------------------------------------------------------
# 市场状态指标（全部只用当周及之前的横截面信息）
# -----------------------------------------------------------------------------
def build_market_state(panel: pd.DataFrame, index_ma_weeks: int) -> pd.DataFrame:
    """把个股面板压缩成"每周一行"的市场状态表。

    所有指标都在当周收盘即可算出，用来预测未来3-4周，不存在未来函数。
    """
    grouped = panel.groupby("Week")
    state = pd.DataFrame(
        {
            "个股数": grouped["Return_1W_pct"].size(),
            "本周涨跌中位数%": grouped["Return_1W_pct"].median(),
            "上涨家数占比": grouped["Return_1W_pct"].apply(lambda s: (s > 0).mean()),
            "超跌股占比": grouped["Drawdown_26W_pct"].apply(
                lambda s: (s <= -20.0).mean()
            ),
            "平均回撤%": grouped["Drawdown_26W_pct"].mean(),
            "未来收益%": grouped["Fwd_Return_pct"].mean(),
        }
    ).sort_index()

    # 等权全池指数：把每周中位涨跌累乘，用来衡量整体位置与动量
    weekly_return = state["本周涨跌中位数%"].fillna(0.0) / 100.0
    state["全池指数"] = (1.0 + weekly_return).cumprod()
    state[f"指数距{index_ma_weeks}周均线%"] = (
        state["全池指数"] / state["全池指数"].rolling(index_ma_weeks).mean() - 1.0
    ) * 100.0
    state["指数4周动量%"] = (
        state["全池指数"] / state["全池指数"].shift(4) - 1.0
    ) * 100.0
    state["指数13周动量%"] = (
        state["全池指数"] / state["全池指数"].shift(13) - 1.0
    ) * 100.0
    state["13周波动率%"] = state["本周涨跌中位数%"].rolling(13).std()
    state["广度4周均值"] = state["上涨家数占比"].rolling(4).mean()
    state["年份"] = state.index.astype(str).str[:4]
    return state.reset_index()


MARKET_INDICATORS = [
    ("上涨家数占比", "当周上涨家数占比"),
    ("广度4周均值", "近4周平均上涨家数占比"),
    ("指数距MA%", "全池指数距均线距离"),
    ("指数4周动量%", "全池指数4周动量"),
    ("指数13周动量%", "全池指数13周动量"),
    ("13周波动率%", "全池指数13周波动率"),
    ("超跌股占比", "超跌股(回撤<-20%)占比"),
    ("平均回撤%", "全池平均26周回撤"),
]


def quintile_analysis(state: pd.DataFrame, ma_weeks: int, buckets: int = 5):
    """把周按各指标分组，看不同市场状态下未来收益差多少。

    这是择时层的核心检验：如果某个指标的最差一组未来收益显著为负，
    就说明它能识别"不该买的时期"。
    """
    work = state.dropna(subset=["未来收益%"]).copy()
    work = work.rename(columns={f"指数距{ma_weeks}周均线%": "指数距MA%"})
    rows = []
    for column, label in MARKET_INDICATORS:
        if column not in work.columns:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        subset = work.assign(_v=values).dropna(subset=["_v"])
        if len(subset) < buckets * 4:
            continue
        try:
            subset["_q"] = pd.qcut(
                subset["_v"].rank(method="first"), buckets, labels=False
            )
        except ValueError:
            continue
        group_means = subset.groupby("_q")["未来收益%"].mean()
        group_counts = subset.groupby("_q")["未来收益%"].size()
        record = {"市场状态指标": label}
        for q in range(buckets):
            record[f"第{q + 1}组(低→高)%"] = float(group_means.get(q, np.nan))
        record["最低组周数"] = int(group_counts.get(0, 0))
        record["最高-最低%"] = float(
            group_means.get(buckets - 1, np.nan) - group_means.get(0, np.nan)
        )
        # 单调性：相邻组是否同向变化，衡量规律是否干净
        diffs = group_means.diff().dropna()
        record["单调性"] = (
            f"{int((diffs > 0).sum())}升/{int((diffs < 0).sum())}降"
        )
        rows.append(record)
    return pd.DataFrame(rows)


def timing_rule_simulation(
    state: pd.DataFrame, ma_weeks: int, hold_weeks: int, exclude_buckets: int = 1,
    buckets: int = 5,
):
    """简单择时规则模拟：在指标最差的若干组里空仓，其余时间满仓。

    与"永远满仓"对比，看择时是否真的改善了结果。
    """
    work = state.dropna(subset=["未来收益%"]).copy()
    work = work.rename(columns={f"指数距{ma_weeks}周均线%": "指数距MA%"})
    always = work["未来收益%"]
    rows = [
        {
            "择时规则": "永远满仓（基准）",
            "参与周数": int(len(always)),
            "参与比例%": 100.0,
            "参与期平均收益%": float(always.mean()),
            "参与期胜率%": float((always > 0).mean() * 100.0),
            "全期年化贡献%": float(always.mean()) * (52.0 / hold_weeks),
        }
    ]
    for column, label in MARKET_INDICATORS:
        if column not in work.columns:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        subset = work.assign(_v=values).dropna(subset=["_v"])
        if len(subset) < buckets * 4:
            continue
        try:
            subset["_q"] = pd.qcut(
                subset["_v"].rank(method="first"), buckets, labels=False
            )
        except ValueError:
            continue
        # 指标越低越差 -> 排除最低的若干组
        keep = subset[subset["_q"] >= exclude_buckets]["未来收益%"]
        if keep.empty:
            continue
        rows.append(
            {
                "择时规则": f"{label}：最差{exclude_buckets}组空仓",
                "参与周数": int(len(keep)),
                "参与比例%": float(len(keep) / len(subset) * 100.0),
                "参与期平均收益%": float(keep.mean()),
                "参与期胜率%": float((keep > 0).mean() * 100.0),
                # 空仓期收益按0计，折算到全期
                "全期年化贡献%": float(keep.sum() / len(subset)) * (52.0 / hold_weeks),
            }
        )
    return pd.DataFrame(rows)


def yearly_state_table(state: pd.DataFrame, ma_weeks: int):
    """分年度：各指标的年均水平 vs 当年实际收益。

    重点看2023这种全年亏损的年份，是否有指标提前给出了警示。
    """
    work = state.dropna(subset=["未来收益%"]).copy()
    work = work.rename(columns={f"指数距{ma_weeks}周均线%": "指数距MA%"})
    columns = ["未来收益%"] + [c for c, _ in MARKET_INDICATORS if c in work.columns]
    table = work.groupby("年份")[columns].mean().reset_index()
    table.insert(1, "周数", work.groupby("年份").size().values)
    return table


def bad_period_detection(state: pd.DataFrame, ma_weeks: int, worst_n: int = 30):
    """把未来收益最差的N周挑出来，看当时各指标处于什么水平。

    如果某指标在这些周明显偏离常态，它就有作为预警信号的价值。
    """
    work = state.dropna(subset=["未来收益%"]).copy()
    work = work.rename(columns={f"指数距{ma_weeks}周均线%": "指数距MA%"})
    worst = work.nsmallest(worst_n, "未来收益%")
    best = work.nlargest(worst_n, "未来收益%")
    rows = []
    for column, label in MARKET_INDICATORS:
        if column not in work.columns:
            continue
        overall = pd.to_numeric(work[column], errors="coerce")
        w = pd.to_numeric(worst[column], errors="coerce")
        b = pd.to_numeric(best[column], errors="coerce")
        std = overall.std(ddof=1)
        rows.append(
            {
                "市场状态指标": label,
                "全期平均": float(overall.mean()),
                f"最差{worst_n}周平均": float(w.mean()),
                f"最好{worst_n}周平均": float(b.mean()),
                "最差组偏离(标准差倍数)": (
                    float((w.mean() - overall.mean()) / std) if std and std > 0 else np.nan
                ),
                "最好组偏离(标准差倍数)": (
                    float((b.mean() - overall.mean()) / std) if std and std > 0 else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["区分度"] = (
            result["最好组偏离(标准差倍数)"] - result["最差组偏离(标准差倍数)"]
        ).abs()
        result = result.sort_values("区分度", ascending=False).reset_index(drop=True)
    return result


# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🌐 {APP_TITLE}")
    st.caption("不选股，只回答一个问题：什么样的市场状态下，未来3-4周整体不该买。")
    st.info(
        "**为什么先做这一层**：此前回测显示2023年509个信号整年亏损（-1.53%/笔），"
        "换任何入场方式、任何排序因子都救不回来。**那一年的问题不是选错了股，是根本不该重仓。**"
        "如果能识别这类时期，收益改善空间可能比继续优化选股因子更大。\n\n"
        "**这一层完全独立于选股**：指标只用全池截面信息，"
        "预测的是全市场未来N周平均收益，因此可以单独验证，不会和选股层的效果混在一起。"
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
        st.subheader("参数")
        hold_weeks = st.number_input(
            "持有周数（预测窗口）", value=3, min_value=1, max_value=8, step=1
        )
        ma_weeks = st.number_input(
            "全池指数均线周期", value=20, min_value=4, max_value=52, step=2
        )
        exclude_buckets = st.number_input(
            "择时规则：排除最差几组（共5组）", value=1, min_value=1, max_value=3, step=1
        )
        worst_n = st.number_input(
            "预警分析：取最差/最好各几周", value=30, min_value=10, max_value=80, step=5
        )

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
        if st.session_state.get("timing_result"):
            return
        st.markdown(
            """
### 测什么

把每一周压缩成一行"市场状态"，测8个指标能否预测**全市场未来3周的平均收益**：

| 指标 | 想法 |
|---|---|
| 当周上涨家数占比 | 广度，普涨还是分化 |
| 近4周平均上涨家数占比 | 平滑后的广度 |
| 全池指数距均线距离 | 整体位置高低 |
| 全池指数4周/13周动量 | 趋势方向 |
| 全池指数13周波动率 | 市场是否动荡 |
| 超跌股占比 | 是否已经跌透 |
| 全池平均26周回撤 | 整体受伤程度 |

### 四张表

**表1 · 五分组** 按各指标把周分成5组，看最差组的未来收益是否明显为负。

**表2 · 择时规则模拟** 在最差组空仓，与永远满仓对比。

**表3 · 分年度** 重点看2023：有没有指标当年就处于异常水平。

**表4 · 极端周诊断** 把未来收益最差的30周挑出来，看当时哪个指标偏离常态最远。

---
**一个必须先说的限制**：四年只有约200周，而且3周持有期意味着相邻观测高度重叠，
真正独立的样本大约只有70个。**择时层的统计功效天然远低于选股层**，
所以这次更看重"经济逻辑是否合理 + 分年度是否一致"，而不是t值。
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

    progress = st.progress(0.0, text="构建全池周线面板……")
    parts = []
    codes = sorted(stocks.keys())
    for idx, ts_code in enumerate(codes):
        weekly = build_weekly_bars(stocks[ts_code])
        if weekly.empty or len(weekly) < 30 + int(hold_weeks):
            continue
        parts.append(build_stock_weekly_panel(weekly, ts_code, int(hold_weeks)))
        if idx % 50 == 0:
            progress.progress(
                min((idx + 1) / len(codes), 1.0),
                text=f"构建全池周线面板……{idx + 1}/{len(codes)}",
            )
    progress.empty()
    del stocks
    gc.collect()

    if not parts:
        st.error("数据不足。")
        return
    panel = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    panel = panel[(panel["Week"] >= start_date) & (panel["Week"] <= end_date)]
    if not basic_indexed.empty:
        basic_reset = basic_indexed.reset_index().rename(
            columns={"trade_date_str": "Week"}
        )
        keep = [c for c in ("Week", "ts_code", "circ_mv") if c in basic_reset.columns]
        if len(keep) == 3:
            panel = panel.merge(
                basic_reset[keep].drop_duplicates(["Week", "ts_code"]),
                on=["Week", "ts_code"], how="left",
            )
            mv = pd.to_numeric(panel["circ_mv"], errors="coerce") / 10000.0
            panel = panel[mv.between(min_mv, max_mv) | mv.isna()]
    panel = panel.reset_index(drop=True)
    if panel.empty:
        st.error("过滤后无数据。")
        return

    state = build_market_state(panel, int(ma_weeks))
    quintiles = quintile_analysis(state, int(ma_weeks))
    rules = timing_rule_simulation(
        state, int(ma_weeks), int(hold_weeks), int(exclude_buckets)
    )
    yearly = yearly_state_table(state, int(ma_weeks))
    detection = bad_period_detection(state, int(ma_weeks), int(worst_n))

    st.session_state["timing_result"] = {
        "state": state,
        "quintiles": quintiles,
        "rules": rules,
        "yearly": yearly,
        "detection": detection,
        "params": {
            "持有": int(hold_weeks), "均线": int(ma_weeks),
            "排除组数": int(exclude_buckets), "极端周": int(worst_n),
        },
    }


def render_results():
    result = st.session_state.get("timing_result")
    if not result:
        return False
    params = result["params"]
    state = result["state"]

    st.markdown("---")
    st.header("市场择时层验证结果")
    valid_weeks = int(state["未来收益%"].notna().sum())
    st.caption(
        f"共 {len(state)} 周，其中 {valid_weeks} 周有完整的未来{params['持有']}周收益　|　"
        f"全池指数均线{params['均线']}周"
    )
    st.warning(
        f"**统计功效提醒**：{valid_weeks}周里，因为持有期{params['持有']}周相互重叠，"
        f"真正独立的观测大约只有 {valid_weeks // params['持有']} 个。"
        "择时层的样本量天生远小于选股层，所以下面更该看经济逻辑是否合理、"
        "分年度是否一致、五分组是否单调，而不是单一数字的大小。"
    )

    st.subheader("表1 · 五分组：不同市场状态下的未来收益")
    st.dataframe(result["quintiles"].round(3), width="stretch", hide_index=True)
    st.caption(
        "第1组=指标最低，第5组=指标最高。**关键看第1组（或第5组）的未来收益是否明显为负**，"
        "以及各组是否单调变化。「单调性」列显示相邻组的升降次数，"
        "接近全升或全降说明规律干净；升降交替说明多半是噪声。"
    )

    st.subheader("表2 · 择时规则模拟")
    st.dataframe(result["rules"].round(3), width="stretch", hide_index=True)
    st.caption(
        "在指标最差的组里空仓，其余时间满仓。**「全期年化贡献%」已把空仓期按0收益折算**，"
        "可以直接和永远满仓比较——如果择时后反而更低，说明这个指标不值得用。"
        "同时要看「参与比例%」：过度择时会导致大部分时间空仓，实操性差。"
    )

    st.subheader("表3 · 分年度：市场状态与实际收益")
    st.dataframe(result["yearly"].round(3), width="stretch", hide_index=True)
    st.caption(
        "**重点看2023行**（此前实测该年整年亏损）：哪些指标在那一年处于明显异常的水平？"
        "如果某指标在2023年偏离得很明显、而在盈利年份处于正常区间，它就有预警价值。"
    )

    st.subheader(f"表4 · 极端周诊断（最差/最好各{params['极端周']}周）")
    st.dataframe(result["detection"].round(3), width="stretch", hide_index=True)
    st.caption(
        "把未来收益最差和最好的周分别挑出来，看当时各指标偏离全期均值多少个标准差。"
        "**「区分度」越大，说明该指标越能分辨好坏时期。**"
        "注意：这是事后诊断，用来找线索，不能直接当作择时规则的证据。"
    )

    with st.expander("查看逐周市场状态明细"):
        st.dataframe(state.round(3), width="stretch", hide_index=True)

    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_quintile_analysis.csv",
            result["quintiles"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "02_timing_rules.csv",
            result["rules"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "03_yearly_state.csv",
            result["yearly"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "04_extreme_weeks.csv",
            result["detection"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "05_weekly_state.csv",
            state.to_csv(index=False, encoding="utf-8-sig"),
        )
    st.download_button(
        "下载择时验证结果",
        data=output.getvalue(),
        file_name="market_timing_validation.zip",
        mime="application/zip",
        key="download_timing",
    )
    return True


if __name__ == "__main__":
    main()
    render_results()
