# -*- coding: utf-8 -*-
"""强弱反弹判别验证器（单文件独立版，直接覆盖 app.py 运行）。

要解决的问题：周线上很多股票呈波浪起伏，在低点买入等待上涨是合理思路，
但反弹分两种——真强势和弱反弹。如何提前分辨？

此前已验证：
- 周线SKDJ低位拐头能较好地识别"波浪低点"（横截面超额+1.34%，t=8.9）
- 26周回撤可作为周内排序（超额+0.99%）
- 挂低于前收3%的限价单优于直接买入（配对+2.92%）
- 但整体超额仅约0.4~0.8%，接近"随便买一篮子科技股"的水平
- 市场择时层全部失效；成交量、换手率、收盘位置等个股自身指标也全部失效

本次测两个从未验证过的方向：

方向一 · 相对强度
  "弱反弹"很可能就是"只是跟着大盘一起涨"——大盘涨5%它也涨5%是beta不是强势。
  此前失效的指标（量能、收盘位置）都只看个股自己，没有和市场比较。
  本次加入：个股涨幅减全池中位数、减行业中位数。

方向二 · 观察一周再决定
  不在信号周就买，先看完下一周表现再定。日线层面的"等确认"已证明失败（追高吃亏），
  但一整周的表现比5天内的日线波动更能反映真实资金态度，值得单独验证。

同时补测趋势背景（距40周均线）——现有"回撤越深越买"的逻辑，
可能正在系统性地挑下降趋势里的股票。

判断标准（比之前更严格）：
  一个真能识别弱反弹的特征，必须同时做到 强组收益更高 + 胜率更高 + 大跌比例更低。
  因为"识别弱反弹"本质是降低失败率，不只是抬高平均值。

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

APP_TITLE = "强弱反弹判别验证"
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


def build_all_weeks(weekly: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """全部周（不只信号周），用于计算全池/行业中位数作为相对强度基准。"""
    close = pd.to_numeric(weekly["close"], errors="coerce")
    return pd.DataFrame(
        {
            "ts_code": ts_code,
            "Week": weekly["trade_date_str"].astype(str),
            "Return_1W_pct": (close / close.shift(1) - 1.0) * 100.0,
        }
    ).dropna(subset=["Return_1W_pct"])


def build_signal_features(
    weekly: pd.DataFrame, ts_code: str, n_period: int, m_period: int,
    level: float, require_kd: bool, hold_weeks: int,
) -> pd.DataFrame:
    """信号周特征 + 观察周(t+1)特征 + 两种入场方式的收益。

    立即买入：t+1周开盘买 -> t+hold周收盘卖
    观察后买：先看完t+1周表现，t+2周开盘买 -> t+1+hold周收盘卖
    两者持有周数相同，可直接比较。
    """
    weekly = add_skdj(weekly, n_period, m_period)
    if len(weekly) < 50:
        return pd.DataFrame()

    close = pd.to_numeric(weekly["close"], errors="coerce")
    high = pd.to_numeric(weekly["high"], errors="coerce")
    low = pd.to_numeric(weekly["low"], errors="coerce")
    open_p = pd.to_numeric(weekly["open"], errors="coerce")
    volume = (
        pd.to_numeric(weekly["vol"], errors="coerce")
        if "vol" in weekly.columns
        else pd.Series(np.nan, index=weekly.index, dtype="float64")
    )
    dates = weekly["trade_date_str"].astype(str)

    k_now = pd.to_numeric(weekly["K"], errors="coerce")
    k_prev = k_now.shift(1)
    d_now = pd.to_numeric(weekly["D"], errors="coerce")
    signal = (k_prev <= k_now) & (k_now <= level)
    if require_kd:
        signal = signal & (k_now > d_now)
    signal = signal.fillna(False)

    return_1w = (close / close.shift(1) - 1.0) * 100.0
    ma40 = close.rolling(40).mean()
    price_range = (high - low).replace(0, np.nan)
    close_location = (close - low) / price_range
    vol_ratio = volume / volume.shift(1).rolling(8).mean().replace(0, np.nan)
    prior_high_4w = high.shift(1).rolling(4).max()
    drawdown_26w = (close / high.rolling(26).max() - 1.0) * 100.0

    frame = pd.DataFrame(
        {
            "ts_code": ts_code,
            "Week": dates,
            "Signal": signal,
            # ---- 信号周(t)特征 ----
            "T_Return_1W_pct": return_1w,
            "T_Dist_MA40_pct": (close / ma40 - 1.0) * 100.0,
            "T_Close_Location": close_location,
            "T_Vol_Ratio": vol_ratio,
            "T_Breakout_4W": (close > prior_high_4w).astype(float),
            "T_Drawdown_26W_pct": drawdown_26w,
            # ---- 观察周(t+1)特征：只用t+1周收盘时已知的信息 ----
            "N_Return_1W_pct": return_1w.shift(-1),
            "N_Close_Location": close_location.shift(-1),
            "N_Vol_Ratio": vol_ratio.shift(-1),
            "N_Break_T_High": (close.shift(-1) > high).astype(float),
            # ---- 收益 ----
            "Ret_Immediate": (
                close.shift(-hold_weeks) / open_p.shift(-1).replace(0, np.nan) - 1.0
            ) * 100.0,
            "Ret_Delayed": (
                close.shift(-(hold_weeks + 1)) / open_p.shift(-2).replace(0, np.nan) - 1.0
            ) * 100.0,
        }
    )
    return frame[frame["Signal"]].drop(columns=["Signal"])


# -----------------------------------------------------------------------------
# 强弱判别检验
# -----------------------------------------------------------------------------
SIGNAL_WEEK_FEATURES = [
    ("RS_Pool_T", "相对全池强度（信号周）"),
    ("RS_Industry_T", "相对行业强度（信号周）"),
    ("T_Dist_MA40_pct", "距40周均线（趋势背景）"),
    ("T_Close_Location", "周内收盘位置"),
    ("T_Vol_Ratio", "成交量比（信号周）"),
    ("T_Breakout_4W", "突破前4周高点"),
    ("T_Drawdown_26W_pct", "26周回撤（越深越好，对照组）"),
]

OBSERVE_WEEK_FEATURES = [
    ("RS_Pool_N", "相对全池强度（观察周）"),
    ("RS_Industry_N", "相对行业强度（观察周）"),
    ("N_Return_1W_pct", "观察周涨幅"),
    ("N_Close_Location", "观察周收盘位置"),
    ("N_Vol_Ratio", "观察周成交量比"),
    ("N_Break_T_High", "观察周突破信号周高点"),
]


def discriminator_test(
    signals: pd.DataFrame, features, return_column: str, cost_pct: float,
    label_prefix: str,
):
    """把信号按各特征分成强/弱两组，看能否区分反弹质量。

    这不是排序检验（每周挑最好的几只），而是筛选检验（判断该不该买）。
    重点看：强组是否同时做到 平均收益更高 + 胜率更高 + 大跌概率更低。
    只有一项好可能是噪声；三项都好才是真的能识别弱反弹。
    """
    work = signals.copy()
    work["_ret"] = pd.to_numeric(work[return_column], errors="coerce") - cost_pct
    work = work.dropna(subset=["_ret"])
    if work.empty:
        return pd.DataFrame()

    overall_mean = work["_ret"].mean()
    rows = []
    for column, name in features:
        if column not in work.columns:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        subset = work.assign(_v=values).dropna(subset=["_v"])
        if len(subset) < 200:
            continue
        # 布尔型按0/1分组，连续型按中位数分组
        unique_values = subset["_v"].nunique()
        if unique_values <= 2:
            strong = subset[subset["_v"] > 0.5]["_ret"]
            weak = subset[subset["_v"] <= 0.5]["_ret"]
        else:
            median = subset["_v"].median()
            strong = subset[subset["_v"] > median]["_ret"]
            weak = subset[subset["_v"] <= median]["_ret"]
        if len(strong) < 100 or len(weak) < 100:
            continue

        diff = strong.mean() - weak.mean()
        pooled_se = math.sqrt(
            strong.var(ddof=1) / len(strong) + weak.var(ddof=1) / len(weak)
        )
        t_stat = diff / pooled_se if pooled_se > 0 else np.nan
        rows.append(
            {
                "判别特征": f"{label_prefix}{name}",
                "强组样本": int(len(strong)),
                "强组收益%": float(strong.mean()),
                "弱组收益%": float(weak.mean()),
                "收益差%": float(diff),
                "强组胜率%": float((strong > 0).mean() * 100.0),
                "弱组胜率%": float((weak > 0).mean() * 100.0),
                "强组大跌<-15%比例": float((strong < -15).mean() * 100.0),
                "弱组大跌<-15%比例": float((weak < -15).mean() * 100.0),
                "t值": t_stat,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("收益差%", ascending=False).reset_index(drop=True)
    return result


def entry_timing_comparison(signals: pd.DataFrame, cost_pct: float):
    """立即买入 vs 观察一周再买（不加任何筛选，纯比较入场时点）。"""
    rows = []
    for column, label in (
        ("Ret_Immediate", "立即买入（t+1周开盘）"),
        ("Ret_Delayed", "观察一周后买入（t+2周开盘）"),
    ):
        values = pd.to_numeric(signals[column], errors="coerce").dropna() - cost_pct
        if values.empty:
            continue
        rows.append(
            {
                "入场时点": label,
                "样本数": int(len(values)),
                "平均收益%": float(values.mean()),
                "中位收益%": float(values.median()),
                "胜率%": float((values > 0).mean() * 100.0),
                "标准差%": float(values.std(ddof=1)),
                "大涨>20%比例": float((values > 20).mean() * 100.0),
                "大跌<-15%比例": float((values < -15).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def combined_filter_test(
    signals: pd.DataFrame, cost_pct: float, top_n: int,
):
    """把最有希望的判别条件组合起来，看筛选后的实际效果。

    对比：不筛选 / 只用观察周强度筛选 / 观察周强度+回撤排序。
    """
    work = signals.copy()
    work["_imm"] = pd.to_numeric(work["Ret_Immediate"], errors="coerce") - cost_pct
    work["_del"] = pd.to_numeric(work["Ret_Delayed"], errors="coerce") - cost_pct

    schemes = []

    base = work.dropna(subset=["_imm"])
    schemes.append(("① 全部信号，立即买入（基准）", base, "_imm"))

    delayed = work.dropna(subset=["_del"])
    schemes.append(("② 全部信号，观察一周后买入", delayed, "_del"))

    if "RS_Pool_N" in work.columns:
        strong_rs = work[
            pd.to_numeric(work["RS_Pool_N"], errors="coerce") > 0
        ].dropna(subset=["_del"])
        schemes.append(("③ 观察周跑赢全池才买", strong_rs, "_del"))

    if "N_Break_T_High" in work.columns:
        breakout = work[
            pd.to_numeric(work["N_Break_T_High"], errors="coerce") > 0.5
        ].dropna(subset=["_del"])
        schemes.append(("④ 观察周突破信号周高点才买", breakout, "_del"))

    if {"RS_Pool_N", "N_Break_T_High"}.issubset(work.columns):
        both = work[
            (pd.to_numeric(work["RS_Pool_N"], errors="coerce") > 0)
            & (pd.to_numeric(work["N_Break_T_High"], errors="coerce") > 0.5)
        ].dropna(subset=["_del"])
        schemes.append(("⑤ 跑赢全池 且 突破信号周高点", both, "_del"))

    rows = []
    total_weeks = work["Week"].nunique()
    for label, subset, column in schemes:
        if subset.empty:
            continue
        values = subset[column].dropna()
        if values.empty:
            continue
        # 每周取回撤最深的top_n只，模拟实际选股
        ranked = subset.groupby("Week")["T_Drawdown_26W_pct"].rank(
            method="first", ascending=True
        )
        picked = subset[ranked <= top_n]
        by_week = picked.groupby("Week")[column].mean().dropna()
        rows.append(
            {
                "方案": label,
                "信号数": int(len(values)),
                "保留比例%": float(len(values) / len(base) * 100.0) if len(base) else np.nan,
                "单笔平均收益%": float(values.mean()),
                "单笔胜率%": float((values > 0).mean() * 100.0),
                "可交易周数": int(len(by_week)),
                "周覆盖率%": float(len(by_week) / total_weeks * 100.0),
                f"每周Top{top_n}收益%": float(by_week.mean()) if len(by_week) else np.nan,
                f"每周Top{top_n}胜率%": (
                    float((by_week > 0).mean() * 100.0) if len(by_week) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🌊 {APP_TITLE}")
    st.caption("周线低点买入之后，怎么分辨哪些是真强势、哪些只是弱反弹。")
    st.info(
        "**这次测两个此前从未验证过的方向：**\n\n"
        "**方向一 · 相对强度**　"
        "所谓弱反弹，很可能就是只是跟着大盘一起涨。大盘涨5%它也涨5%，那是beta不是强势。"
        "之前测过的量能、收盘位置都只看个股自己，全部失效；"
        "这次加入减去全池中位数、减去行业中位数的超额涨幅。\n\n"
        "**方向二 · 观察一周再决定**　"
        "不在信号周就买，先看完下一周的表现再定。日线层面的等确认已被证明失败（追高吃亏），"
        "但一整周的表现比5天内的日线波动更能反映真实资金态度，值得单独验证。\n\n"
        "同时补测**趋势背景**（距40周均线）——"
        "现在回撤越深越买的逻辑，可能正在系统性地挑下降趋势里的股票。"
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
        st.subheader("周线信号（波浪低点）")
        n_period = st.number_input("N", value=4, min_value=2, max_value=60, step=1)
        m_period = st.number_input("M", value=3, min_value=2, max_value=30, step=1)
        level = st.number_input("K值阈值", value=20.0, min_value=1.0, max_value=90.0, step=5.0)
        require_kd = st.checkbox("要求 K > D", value=True)
        hold_weeks = st.number_input("持有周数", value=3, min_value=1, max_value=8, step=1)
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
        if st.session_state.get("wave_result"):
            return
        st.markdown(
            """
### 四张表

**表1 · 信号周判别特征**　在信号周就能算出的强弱特征，能否区分后续表现。
新增：相对全池强度、相对行业强度、距40周均线（趋势背景）。

**表2 · 观察周判别特征**　用信号后第一周的表现来判断，能否区分。
新增：观察周超额涨幅、是否突破信号周高点。

**表3 · 立即买入 vs 观察一周**　纯粹比较入场时点，不加任何筛选。

**表4 · 组合筛选效果**　把有效的判别条件组合起来，看实际能提升多少，
以及**周覆盖率**下降多少（这决定实操性）。

### 判断标准

一个真正能识别弱反弹的特征，应该**同时满足三条**：
- 强组平均收益更高
- 强组胜率更高
- 强组大跌（<-15%）比例更低

只满足一条大概率是噪声。这个标准比之前只看平均收益更严格，
因为"识别弱反弹"本质上是要**降低失败率**，不只是提高平均值。
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

    progress = st.progress(0.0, text="计算信号与判别特征……")
    all_week_parts = []
    signal_parts = []
    codes = sorted(stocks.keys())
    for idx, ts_code in enumerate(codes):
        weekly = build_weekly_bars(stocks[ts_code])
        if weekly.empty or len(weekly) < 50 + int(hold_weeks):
            continue
        weekly = add_skdj(weekly, int(n_period), int(m_period))
        all_week_parts.append(build_all_weeks(weekly, ts_code))
        rows = build_signal_features(
            weekly, ts_code, int(n_period), int(m_period),
            float(level), bool(require_kd), int(hold_weeks),
        )
        if not rows.empty:
            signal_parts.append(rows)
        if idx % 50 == 0:
            progress.progress(
                min((idx + 1) / len(codes), 1.0),
                text=f"计算信号与判别特征……{idx + 1}/{len(codes)}",
            )
    progress.empty()
    del stocks
    gc.collect()

    if not signal_parts or not all_week_parts:
        st.error("没有产生信号。")
        return

    all_weeks = pd.concat(all_week_parts, ignore_index=True)
    signals = pd.concat(signal_parts, ignore_index=True)
    del all_week_parts, signal_parts
    gc.collect()

    # 相对强度基准：全池中位数与行业中位数
    all_weeks["Industry"] = all_weeks["ts_code"].map(industry_map).fillna("未分类")
    pool_median = all_weeks.groupby("Week")["Return_1W_pct"].median().rename("Pool_Median")
    industry_median = (
        all_weeks.groupby(["Week", "Industry"])["Return_1W_pct"]
        .median()
        .rename("Industry_Median")
        .reset_index()
    )
    # 观察周(t+1)的基准需要用下一周的中位数
    pool_median_frame = pool_median.reset_index()
    week_order = sorted(all_weeks["Week"].unique())
    next_week_map = {
        week: week_order[i + 1] for i, week in enumerate(week_order[:-1])
    }

    signals["Industry"] = signals["ts_code"].map(industry_map).fillna("未分类")
    signals = signals.merge(pool_median_frame, on="Week", how="left")
    signals = signals.merge(industry_median, on=["Week", "Industry"], how="left")
    signals["Next_Week"] = signals["Week"].map(next_week_map)
    signals = signals.merge(
        pool_median_frame.rename(
            columns={"Week": "Next_Week", "Pool_Median": "Pool_Median_N"}
        ),
        on="Next_Week", how="left",
    )
    signals = signals.merge(
        industry_median.rename(
            columns={"Week": "Next_Week", "Industry_Median": "Industry_Median_N"}
        ),
        on=["Next_Week", "Industry"], how="left",
    )

    signals["RS_Pool_T"] = signals["T_Return_1W_pct"] - signals["Pool_Median"]
    signals["RS_Industry_T"] = signals["T_Return_1W_pct"] - signals["Industry_Median"]
    signals["RS_Pool_N"] = signals["N_Return_1W_pct"] - signals["Pool_Median_N"]
    signals["RS_Industry_N"] = signals["N_Return_1W_pct"] - signals["Industry_Median_N"]

    signals = signals[
        (signals["Week"] >= start_date) & (signals["Week"] <= end_date)
    ]
    if not basic_indexed.empty:
        basic_reset = basic_indexed.reset_index().rename(
            columns={"trade_date_str": "Week"}
        )
        keep = [c for c in ("Week", "ts_code", "circ_mv") if c in basic_reset.columns]
        if len(keep) == 3:
            signals = signals.merge(
                basic_reset[keep].drop_duplicates(["Week", "ts_code"]),
                on=["Week", "ts_code"], how="left",
            )
            mv = pd.to_numeric(signals["circ_mv"], errors="coerce") / 10000.0
            signals = signals[mv.between(min_mv, max_mv) | mv.isna()]
    signals = signals.reset_index(drop=True)
    if signals.empty:
        st.error("过滤后无信号。")
        return

    table_t = discriminator_test(
        signals, SIGNAL_WEEK_FEATURES, "Ret_Immediate", float(cost_pct), ""
    )
    table_n = discriminator_test(
        signals, OBSERVE_WEEK_FEATURES, "Ret_Delayed", float(cost_pct), ""
    )
    timing = entry_timing_comparison(signals, float(cost_pct))
    combos = combined_filter_test(signals, float(cost_pct), int(top_n))

    st.session_state["wave_result"] = {
        "signals": signals,
        "table_t": table_t,
        "table_n": table_n,
        "timing": timing,
        "combos": combos,
        "params": {
            "N": int(n_period), "M": int(m_period), "阈值": float(level),
            "持有": int(hold_weeks), "每周选": int(top_n),
        },
    }


def render_results():
    result = st.session_state.get("wave_result")
    if not result:
        return False
    params = result["params"]
    signals = result["signals"]

    st.markdown("---")
    st.header("强弱反弹判别结果")
    st.caption(
        f"信号 {len(signals):,} 笔，覆盖 {signals['Week'].min()} — {signals['Week'].max()}"
        f"　|　持有{params['持有']}周"
    )

    st.subheader("表1 · 信号周就能看出的强弱特征")
    st.dataframe(result["table_t"].round(3), width="stretch", hide_index=True)
    st.caption(
        "**三条同时满足才算有效**：强组收益更高 + 强组胜率更高 + 强组大跌比例更低。"
        "「相对全池/行业强度」和「距40周均线」是本次新增的、此前从未测过的角度。"
    )

    st.subheader("表2 · 观察一周后才能看出的强弱特征")
    st.dataframe(result["table_n"].round(3), width="stretch", hide_index=True)
    st.caption(
        "用信号后第一周的表现判断。**注意**：这一层多等了一周，"
        "所以即使判别有效，也要和表3的延迟入场代价一起看净效果。"
    )

    st.subheader("表3 · 入场时点：立即买 vs 观察一周")
    st.dataframe(result["timing"].round(3), width="stretch", hide_index=True)
    st.caption(
        "不加任何筛选，纯粹比较入场时点的代价。"
        "如果观察一周本身就明显亏，那表2的判别能力必须大到能覆盖这个代价才划算。"
    )

    st.subheader("表4 · 组合筛选的实际效果")
    st.dataframe(result["combos"].round(3), width="stretch", hide_index=True)
    st.caption(
        "**重点看「周覆盖率%」**——筛选越严，能交易的周越少。"
        "之前的教训是：靠大幅牺牲交易机会换来的收益提升，没有实操价值。"
        "理想结果是收益和胜率明显提升，而周覆盖率保持在70%以上。"
    )

    st.markdown("---")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "01_signal_week_discriminators.csv",
            result["table_t"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "02_observe_week_discriminators.csv",
            result["table_n"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "03_entry_timing.csv",
            result["timing"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "04_combined_filters.csv",
            result["combos"].to_csv(index=False, encoding="utf-8-sig"),
        )
        archive.writestr(
            "05_all_signals.csv",
            signals.to_csv(index=False, encoding="utf-8-sig"),
        )
    st.download_button(
        "下载验证结果",
        data=output.getvalue(),
        file_name="wave_strength_validation.zip",
        mime="application/zip",
        key="download_wave",
    )
    return True


if __name__ == "__main__":
    main()
    render_results()
