# -*- coding: utf-8 -*-
"""突破策略排序层验证器（单文件独立版，直接覆盖 app.py 运行）。

当前突破策略只有筛选（能不能买），没有排序（先买哪个）。
每周平均产生6.27个信号，意味着在仓位受限时，一半以上的选择完全没有依据。

SKDJ阶段已验证排序层的决定性作用：同一批信号，用K值排序时三仓收益-7.3%、
蒙特卡洛分位45%（不如随机）；换成26周回撤排序后变成+45.7%、分位94%。
只换排序规则，结果天差地别。

本轮先不加资金约束，纯粹测量排序变量本身的选股能力：
每周按各变量取前N名，与"该周全部信号的平均"对比。

判断标准（三条同时满足）：
  1. 正向TopN收益明显高于全部信号，且 |t|>2
  2. 反向对照明显更差（正反都好说明是噪声）
  3. 分年度大部分为正

另设"随机取N只"作为参照，代表只受数量限制、完全无选股能力的水平；
任何排序变量跑不赢它即为无效。

信号规则与样本外测试完全一致并锁死，本轮不做任何修改。
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

APP_TITLE = "突破策略排序层验证"
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
# 冻结的信号规则（与样本外测试一致，不修改）
# 本轮唯一新增的是"排序层"：同一周有多个信号时，先买哪几个。
# =============================================================================
FROZEN_BREAKOUT_WEEKS = 26
FROZEN_POSITION_QUANTILE = 0.33
FROZEN_VOL_CONTRACTION_MAX = 0.8
FROZEN_FORWARD_WEEKS = 26
FROZEN_STOP_PCT = 15.0


def compute_features(weekly: pd.DataFrame):
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
    prior_high = close.shift(1).rolling(FROZEN_BREAKOUT_WEEKS).max()
    features["breakout"] = close > prior_high
    features["position_2y"] = close / high.shift(1).rolling(104).max().replace(0, np.nan)
    vol_recent = return_1w.shift(1).rolling(8).std()
    vol_earlier = return_1w.shift(9).rolling(18).std()
    features["vol_contraction"] = vol_recent / vol_earlier.replace(0, np.nan)
    features["volume_surge"] = volume / volume.shift(1).rolling(8).mean().replace(0, np.nan)
    # 突破幅度：本周收盘超出前高多少
    features["breakout_strength"] = (close / prior_high.replace(0, np.nan) - 1.0) * 100.0
    # 横盘区间宽度
    base_high = high.shift(1).rolling(26).max()
    base_low = low.shift(1).rolling(26).min()
    base_mid = close.shift(1).rolling(26).mean()
    features["base_range"] = (base_high - base_low) / base_mid.replace(0, np.nan) * 100.0
    features["momentum_26w"] = (close / close.shift(26) - 1.0) * 100.0
    features["weekly_vol"] = return_1w.rolling(26).std()
    features["dist_ma40"] = (close / close.rolling(40).mean() - 1.0) * 100.0
    features["return_1w"] = return_1w
    return features


def evaluate_stock(weekly: pd.DataFrame, ts_code: str, position_threshold: float):
    """只输出符合冻结信号的周，附带全部候选排序变量与结果。"""
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

        rows.append(
            {
                "ts_code": ts_code,
                "Week": dates[i],
                "Entry_Price": entry_price,
                "Position_2Y": position[i],
                "Vol_Contraction": contraction[i],
                "Volume_Surge": features["volume_surge"].iloc[i],
                "Breakout_Strength": features["breakout_strength"].iloc[i],
                "Base_Range": features["base_range"].iloc[i],
                "Momentum_26W": features["momentum_26w"].iloc[i],
                "Weekly_Vol": features["weekly_vol"].iloc[i],
                "Dist_MA40": features["dist_ma40"].iloc[i],
                "Return_1W": features["return_1w"].iloc[i],
                "Max_Gain_pct": max_gain,
                "Trail_Return_pct": (exit_price / entry_price - 1.0) * 100.0,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 排序层检验
# -----------------------------------------------------------------------------
RANK_CANDIDATES = [
    ("Position_2Y", "两年高点位置（越高越优先）", False),
    ("MV_Billion", "流通市值（越大越优先）", False),
    ("Vol_Contraction", "波动率压缩（越强越优先）", True),
    ("Volume_Surge", "突破量能（越大越优先）", False),
    ("Breakout_Strength", "突破幅度（超出前高越多越优先）", False),
    ("Base_Range", "横盘区间宽度（越宽越优先）", False),
    ("Momentum_26W", "26周动量（越强越优先）", False),
    ("Weekly_Vol", "个股波动率（越大越优先）", False),
    ("Dist_MA40", "距40周均线（越远越优先）", False),
    ("Return_1W", "信号周涨幅（越大越优先）", False),
]


def ranking_test(signals: pd.DataFrame, top_n: int, cost_pct: float, random_draws: int = 30):
    """无资金约束下，检验各排序变量能否在同一周内挑出更好的股票。

    对比三组，全部按周先取均值再对周求平均，因此不受"某些周信号特别多"的影响：
      - 全部信号：该周所有信号的平均（不做任何排序）
      - 正向TopN：按变量取前N名
      - 反向TopN：按相反方向取N名（噪声检测——真信号的反向应明显更差）
      - 随机TopN：随机取N名，反映"只受仓位数限制、无选股能力"的水平
    """
    work = signals.copy()
    work["_gain"] = pd.to_numeric(work["Max_Gain_pct"], errors="coerce")
    work["_ret"] = pd.to_numeric(work["Trail_Return_pct"], errors="coerce") - cost_pct
    work = work.dropna(subset=["_gain", "_ret"])
    if work.empty:
        return pd.DataFrame()

    baseline_ret = work.groupby("Week")["_ret"].mean()
    baseline_gain = work.groupby("Week")["_gain"].mean()
    baseline_double = work.groupby("Week")["_gain"].apply(lambda s: (s > 100).mean() * 100.0)

    rng = np.random.default_rng(20240501)

    def week_stats(subset: pd.DataFrame):
        by_week_ret = subset.groupby("Week")["_ret"].mean()
        by_week_gain = subset.groupby("Week")["_gain"].mean()
        by_week_double = subset.groupby("Week")["_gain"].apply(
            lambda s: (s > 100).mean() * 100.0
        )
        return by_week_ret, by_week_gain, by_week_double

    rows = []

    # 基准行：全部信号
    rows.append(
        {
            "排序规则": "【基准】全部信号（无排序）",
            "方向": "—",
            "覆盖周数": int(len(baseline_ret)),
            "止损后收益%": float(baseline_ret.mean()),
            "止损后胜率%": float((baseline_ret > 0).mean() * 100.0),
            "平均最大涨幅%": float(baseline_gain.mean()),
            "翻倍概率%": float(baseline_double.mean()),
            "vs基准收益差%": 0.0,
            "粗略t值": np.nan,
        }
    )

    # 随机TopN
    random_means = []
    for _ in range(random_draws):
        picks = work.groupby("Week", group_keys=False).apply(
            lambda g: g.sample(n=min(top_n, len(g)), random_state=int(rng.integers(1e9))),
            include_groups=False,
        )
        random_means.append(picks.groupby(picks.index if "Week" not in picks.columns else "Week")["_ret"].mean())
    if random_means:
        stacked = pd.concat(random_means, axis=1)
        random_by_week = stacked.mean(axis=1)
        aligned = baseline_ret.reindex(random_by_week.index)
        rows.append(
            {
                "排序规则": f"【对照】随机取{top_n}只",
                "方向": "随机",
                "覆盖周数": int(len(random_by_week)),
                "止损后收益%": float(random_by_week.mean()),
                "止损后胜率%": float((random_by_week > 0).mean() * 100.0),
                "平均最大涨幅%": np.nan,
                "翻倍概率%": np.nan,
                "vs基准收益差%": float((random_by_week - aligned).dropna().mean()),
                "粗略t值": np.nan,
            }
        )

    for column, label, ascending in RANK_CANDIDATES:
        if column not in work.columns:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        if values.notna().sum() < len(work) * 0.5:
            continue
        subset = work.assign(_v=values).dropna(subset=["_v"])
        rank_fwd = subset.groupby("Week")["_v"].rank(method="first", ascending=ascending)
        rank_rev = subset.groupby("Week")["_v"].rank(method="first", ascending=not ascending)

        for direction, mask in (("正向", rank_fwd <= top_n), ("反向", rank_rev <= top_n)):
            picked = subset[mask]
            if picked.empty:
                continue
            by_ret, by_gain, by_double = week_stats(picked)
            aligned = baseline_ret.reindex(by_ret.index)
            diff = (by_ret - aligned).dropna()
            if len(diff) > 1 and diff.std(ddof=1) > 0:
                t_stat = diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff)))
            else:
                t_stat = np.nan
            rows.append(
                {
                    "排序规则": label if direction == "正向" else f"　└ 反向对照",
                    "方向": direction,
                    "覆盖周数": int(len(by_ret)),
                    "止损后收益%": float(by_ret.mean()),
                    "止损后胜率%": float((by_ret > 0).mean() * 100.0),
                    "平均最大涨幅%": float(by_gain.mean()),
                    "翻倍概率%": float(by_double.mean()),
                    "vs基准收益差%": float(diff.mean()),
                    "粗略t值": t_stat,
                }
            )
    return pd.DataFrame(rows)


def top_n_sensitivity(signals: pd.DataFrame, cost_pct: float, best_columns):
    """同一排序变量在取前3/5/10名时的表现，判断优势是否随N平滑衰减。"""
    work = signals.copy()
    work["_ret"] = pd.to_numeric(work["Trail_Return_pct"], errors="coerce") - cost_pct
    work["_gain"] = pd.to_numeric(work["Max_Gain_pct"], errors="coerce")
    work = work.dropna(subset=["_ret", "_gain"])
    baseline = work.groupby("Week")["_ret"].mean()
    rows = []
    for column, label, ascending in best_columns:
        if column not in work.columns:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        subset = work.assign(_v=values).dropna(subset=["_v"])
        record = {"排序规则": label}
        for n in (3, 5, 10, 20):
            rank = subset.groupby("Week")["_v"].rank(method="first", ascending=ascending)
            picked = subset[rank <= n]
            if picked.empty:
                continue
            by_week = picked.groupby("Week")["_ret"].mean()
            record[f"Top{n}收益%"] = float(by_week.mean())
            record[f"Top{n}翻倍%"] = float(
                picked.groupby("Week")["_gain"].apply(lambda s: (s > 100).mean() * 100.0).mean()
            )
        record["全部信号收益%"] = float(baseline.mean())
        rows.append(record)
    return pd.DataFrame(rows)


def yearly_ranking(signals: pd.DataFrame, cost_pct: float, top_n: int, best_columns):
    work = signals.copy()
    work["年份"] = work["Week"].astype(str).str[:4]
    work["_ret"] = pd.to_numeric(work["Trail_Return_pct"], errors="coerce") - cost_pct
    work = work.dropna(subset=["_ret"])
    rows = []
    for year, group in work.groupby("年份"):
        record = {"年份": year, "信号数": len(group)}
        record["全部信号%"] = float(group.groupby("Week")["_ret"].mean().mean())
        for column, label, ascending in best_columns:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            subset = group.assign(_v=values).dropna(subset=["_v"])
            if subset.empty:
                continue
            rank = subset.groupby("Week")["_v"].rank(method="first", ascending=ascending)
            picked = subset[rank <= top_n]
            record[label] = (
                float(picked.groupby("Week")["_ret"].mean().mean())
                if not picked.empty
                else np.nan
            )
        rows.append(record)
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
    st.title(f"🏅 {APP_TITLE}")
    st.caption("无资金约束，只问一件事：同一周多个突破信号，先买哪几个更好？")
    st.info(
        "**为什么这是目前最大的空白**：突破策略现在只有筛选（能不能买），"
        "没有排序（先买哪个）。而每周平均有6.27个信号——"
        "**意味着一半以上的选择完全没有依据，等于随机**。\n\n"
        "SKDJ阶段已经验证过排序层的威力：同一批信号，K值排序时3仓收益-7.3%、"
        "蒙特卡洛分位45%（不如随机）；换成26周回撤排序后变成+45.7%、分位94%。"
        "**只换排序规则，结果天差地别。**\n\n"
        "本轮先不加三仓约束，纯粹测量排序变量本身的选股能力。"
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
        st.subheader("排序检验设置")
        top_n = st.number_input(
            "每周取前几名", value=5, min_value=1, max_value=30, step=1,
            help="先用5验证排序能力本身，暂不模拟三仓资金约束。",
        )
        st.caption("信号规则已锁死，与样本外测试一致，不可修改。")
        min_price = st.number_input("最低股价（元）", value=10.0, min_value=0.0, step=1.0)
        min_mv = st.number_input("最低流通市值（亿元）", value=100.0, min_value=0.0, step=10.0)
        max_mv = st.number_input("最高流通市值（亿元）", value=1000.0, min_value=100.0, step=100.0)
        cost_pct = st.number_input("往返成本%", value=0.20, min_value=0.0, max_value=2.0, step=0.05)

        st.markdown("---")
        clear_cache_clicked = st.button("清空行情缓存")
        run_clicked = st.button("开始验证", type="primary")

    if clear_cache_clicked:
        if os.path.isdir(MARKET_CACHE_ROOT):
            shutil.rmtree(MARKET_CACHE_ROOT)
        st.success("行情缓存已清空。")

    if not run_clicked:
        if st.session_state.get("rank_result"):
            return
        st.markdown(
            """
### 测试的十个排序变量

| 变量 | 已有线索 |
|---|---|
| 两年高点位置 | 五分组18.96%→35.08%，5年4年正向（现在只当筛选，没当排序） |
| 流通市值 | 分层显示提升倍数随市值单调上升（1.44→2.68） |
| 波动率压缩程度 | 弱 |
| 突破量能 | 弱 |
| 突破幅度 | 未测过 |
| 横盘区间宽度 | 全期单调但2022强烈反向，不稳 |
| 26周动量 | 未单独测过 |
| 个股波动率 | 未测过 |
| 距40周均线 | 未测过 |
| 信号周涨幅 | 未测过 |

### 判断标准（三条同时满足才可信）

1. **正向Top5收益明显高于「全部信号」**，且 |t|>2
2. **反向对照明显更差**——如果正反都好，那是噪声
3. **分年度大部分为正**——不能只靠某一年

### 三张表

**表1 · 排序变量检验**　每个变量都带反向对照，另有「随机取5只」作为参照，
反映"只受数量限制、无选股能力"的水平。

**表2 · Top N敏感性**　同一变量取前3/5/10/20名的表现。
真实的选股能力应该随N增大而平滑衰减；如果只有某个N好，那是噪声。

**表3 · 分年度**　确认最优排序不是靠某一年。
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

    needed = ["trade_date_str", "open", "high", "low", "close", "vol"]
    weekly_cache = {}
    position_samples = []
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
        for column in ("open", "high", "low", "close", "vol"):
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
    position_threshold = float(all_positions.quantile(1.0 - FROZEN_POSITION_QUANTILE))
    del all_positions
    gc.collect()

    progress = st.progress(0.0, text="生成信号与排序变量……")
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

    signals = signals[(signals["Week"] >= start_date) & (signals["Week"] <= end_date)]
    signals = signals[
        pd.to_numeric(signals["Entry_Price"], errors="coerce") >= min_price
    ]
    if not mv_lookup.empty:
        signals = signals.merge(mv_lookup, on=["Week", "ts_code"], how="left")
        signals["MV_Billion"] = pd.to_numeric(signals["circ_mv"], errors="coerce") / 10000.0
        signals = signals[
            signals["MV_Billion"].between(min_mv, max_mv) | signals["MV_Billion"].isna()
        ]
        del mv_lookup
        gc.collect()
    else:
        signals["MV_Billion"] = np.nan
    signals = signals.reset_index(drop=True)
    if signals.empty:
        st.error("过滤后无信号。")
        return

    table = ranking_test(signals, int(top_n), float(cost_pct))
    # 取正向组里收益最高的三个变量做后续分析
    forward = table[table["方向"] == "正向"].copy()
    best = forward.nlargest(3, "止损后收益%")["排序规则"].tolist() if not forward.empty else []
    best_columns = [
        (col, label, asc) for col, label, asc in RANK_CANDIDATES if label in best
    ]

    st.session_state["rank_result"] = {
        "signal_count": len(signals),
        "weeks": signals["Week"].nunique(),
        "table": table,
        "sensitivity": top_n_sensitivity(signals, float(cost_pct), best_columns),
        "yearly": yearly_ranking(signals, float(cost_pct), int(top_n), best_columns),
        "top_n": int(top_n),
        "period": f"{start_date} — {end_date}",
        "memory_mb": _memory_usage_mb(),
    }


def render_results():
    result = st.session_state.get("rank_result")
    if not result:
        return False

    st.markdown("---")
    st.header("排序层验证结果")
    st.caption(
        f"突破信号 {result['signal_count']:,} 个，覆盖 {result['weeks']} 周　|　"
        f"区间 {result['period']}　|　每周取前{result['top_n']}名"
        + (
            f"　|　内存 {result['memory_mb']:.0f} MB"
            if math.isfinite(result.get("memory_mb", float("nan")))
            else ""
        )
    )

    st.subheader(f"表1 · 十个排序变量的选股能力（每周取前{result['top_n']}名）")
    st.dataframe(result["table"].round(2), width="stretch", hide_index=True)
    st.caption(
        "**三条同时满足才可信**：\n"
        "1. 正向收益明显高于第一行「全部信号」，且 |t|>2\n"
        "2. 紧跟其下的「反向对照」明显更差——正反都好说明是噪声\n"
        "3. 分年度大部分为正（见表3）\n\n"
        "第二行「随机取N只」是重要参照：它代表**只受数量限制、完全没有选股能力**的水平。"
        "任何排序变量如果跑不赢它，就等于没用。"
    )

    if not result["sensitivity"].empty:
        st.subheader("表2 · Top N 敏感性（取最好的三个变量）")
        st.dataframe(result["sensitivity"].round(2), width="stretch", hide_index=True)
        st.caption(
            "真实的选股能力应该**随N增大平滑衰减**（取的越多越接近全池平均）。"
            "如果只有某一个N特别好、相邻的N反而差，那多半是噪声。"
        )

    if not result["yearly"].empty:
        st.subheader("表3 · 分年度")
        st.dataframe(result["yearly"].round(2), width="stretch", hide_index=True)
        st.caption(
            "**重点看2022-2024那三个亏损年**：好的排序规则应该至少减轻亏损。"
            "如果只在2025-2026有效，那它就只是又一个依赖行情的东西。"
        )

    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_ranking_test.csv",
            result["table"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "02_topn_sensitivity.csv",
            result["sensitivity"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "03_yearly.csv",
            result["yearly"].to_csv(index=False, encoding="utf-8-sig"),
        )
    st.download_button(
        "下载验证结果",
        data=output.getvalue(),
        file_name="ranking_validation.zip",
        mime="application/zip",
        key="download_rank",
    )
    return True


if __name__ == "__main__":
    main()
    render_results()
