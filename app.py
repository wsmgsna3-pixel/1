# -*- coding: utf-8 -*-
"""
=========================================================================
科技 / 军工 / 新能源 / 机器人  —— 排名轮动选股与回测系统
=========================================================================
单文件 Streamlit 应用。直接覆盖原 app.py，然后:

    pip install streamlit tushare pandas numpy pyarrow
    streamlit run app.py

设计要点
--------
1) 排名轮动，不是信号触发。每个调仓日对全池打分排序，永远持有分数最高的
   N 只。空窗由结构决定为 0，不存在"没信号"的问题。
2) 先验证因子，再谈策略。"因子分层检验"标签页可以在不写任何买卖逻辑的
   前提下，单独回答"这个打分有没有排序能力"。
3) 基准是股票池自身的等权指数，不是沪深300。跑赢沪深300可能只是吃了
   板块 beta，那不是 alpha。
4) 回测口径：次日开盘成交、涨停不买、跌停不卖、停牌顺延、佣金+印花税+
   滑点、时点市值筛选、含退市股票。
=========================================================================
"""
from __future__ import annotations

import os
import gc
import time
import concurrent.futures as cf
import pickle
import threading
import datetime as dt
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:  # 便于在无 streamlit 环境下单测纯计算函数
    st = None

# ----------------------------------------------------------------------
# 常量配置
# ----------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    import tushare as ts_mod
except Exception:
    ts_mod = None

CACHE_DIR = os.path.join(APP_DIR, ".cache_pool")
PX_DIR = os.path.join(CACHE_DIR, "px")     # 唯一的缓存：每只股票一个文件
os.makedirs(PX_DIR, exist_ok=True)

# 申万2021 一级行业（整体纳入）
SW_L1_ALL = ["电子", "计算机", "通信", "传媒", "国防军工", "电力设备", "机械设备",
             "汽车", "医药生物", "基础化工", "有色金属", "电子设备"]
SW_L1_DEFAULT = ["电子", "计算机", "通信", "传媒", "国防军工"]

# 申万2021 二级行业（只取子行业，避免把整个电力设备/机械设备灌进来）
SW_L2_ALL = ["电池", "光伏设备", "风电设备", "电网设备", "其他电源设备Ⅱ",
             "自动化设备", "通用设备", "专用设备", "工程机械", "汽车零部件",
             "半导体", "光学光电子", "元件", "消费电子", "军工电子Ⅱ"]
SW_L2_DEFAULT = ["电池", "光伏设备", "风电设备", "其他电源设备Ⅱ", "自动化设备"]

# 因子登记表: 内部名 -> (中文名, 默认权重)
# 默认权重一律为 0：不预设任何结论，权重必须由检验结果决定。
FACTOR_DEF = {
    "mom_ra":  ("风险调整动量(60日,跳过最近5日)", 0.0),
    "trend_q": ("趋势质量(斜率×R²)",             0.0),
    "rel_str": ("板块相对强度(60日超额)",         0.0),
    "vol_exp": ("量能扩张(5日额/60日额)",         0.0),
    "dist_hi": ("距60日高点(越近越高)",           0.0),
    "rev5":    ("5日反转(近5日涨幅)",            -0.0),
}
# 预设权重组，方便对比。名字即用途。
WEIGHT_PRESETS = {
    "A组：只用 距60日高点": {"dist_hi": -1.0},
    "B组：距60日高点 + 5日反转": {"dist_hi": -1.0, "rev5": -0.5},
    "全部清零": {},
}
FACTOR_KEYS = list(FACTOR_DEF.keys())

# 诊断因子：不参与打分，只用来排除混淆。市值是最关键的一个——
# 池子有 50-1000 亿的上下限，动量排名高的股票平均更靠近上沿，
# 所以"动量为负"很可能只是"小市值跑赢大市值"的伪装。
DIAG_DEF = {
    "logsize": ("对数流通市值（诊断用）", 0.0),
    "amihud":  ("非流动性(|收益|/成交额，诊断用)", 0.0),
}
DIAG_KEYS = list(DIAG_DEF.keys())
COMPOSITE_KEY = "composite"
ALL_DEF = {**FACTOR_DEF, **DIAG_DEF, COMPOSITE_KEY: ("综合打分（当前权重）", 0.0)}
TEST_KEYS = FACTOR_KEYS + DIAG_KEYS


# ======================================================================
# 一、Tushare 数据层
# ======================================================================
class Limiter:
    """
    线程安全的每分钟请求数限流器。
    Tushare 的频次限制是按账号算的，不是按连接算的，所以多线程必须共用同一个
    限流器，否则 4 条线程各跑各的会直接把账号打到限频。
    """

    def __init__(self, per_min: int = 400):
        self.per_min = max(1, int(per_min))
        self.calls: deque = deque()
        self.lock = threading.Lock()

    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                while self.calls and now - self.calls[0] >= 60.0:
                    self.calls.popleft()
                if len(self.calls) < self.per_min:
                    self.calls.append(now)
                    return
                sleep_s = 60.0 - (now - self.calls[0]) + 0.05
            time.sleep(min(max(sleep_s, 0.01), 5.0))   # 必须在锁外面睡


_ERR_LOCK = threading.Lock()
API_ERRORS: List[str] = []


def _safe_progress(cb, *a):
    """进度回调只是给人看的，任何异常都不该让下载失败。"""
    if cb is None:
        return
    try:
        cb(*a)
    except Exception:
        pass


def api_call(fn, lim: Limiter, retries: int = 3, **kwargs):
    """带限流与重试的 Tushare 调用。失败返回 None。可在工作线程中安全调用。"""
    last = None
    for k in range(retries):
        try:
            lim.wait()
            return fn(**kwargs)
        except Exception as e:  # 网络抖动 / 限频
            last = e
            time.sleep(2.0 * (k + 1))
    # 注意: 工作线程里不能碰 st.session_state（没有 ScriptRunContext）
    with _ERR_LOCK:
        if len(API_ERRORS) < 500:
            API_ERRORS.append(f"{getattr(fn, '__name__', 'api')} {kwargs.get('ts_code', '')}: {last}")
    return None


def fetch_universe(pro, lim: Limiter, l1_names: List[str], l2_names: List[str]) -> pd.DataFrame:
    """
    取申万成分股，返回 ts_code / ind_name / in_date / out_date。
    优先 index_member_all（自带进出日期），失败退回 index_member。
    """
    cls = api_call(pro.index_classify, lim, src="SW2021")
    if cls is None or len(cls) == 0:
        cls = api_call(pro.index_classify, lim)  # 老接口
    if cls is None or len(cls) == 0:
        return pd.DataFrame(columns=["ts_code", "ind_name", "in_date", "out_date"])

    name_col = "industry_name" if "industry_name" in cls.columns else "name"
    rows = []

    def _grab(code: str, level: str, ind_name: str):
        df = None
        try:
            kw = {"l1_code": code} if level == "L1" else {"l2_code": code}
            df = api_call(pro.index_member_all, lim, retries=1, **kw)
        except Exception:
            df = None
        if df is not None and len(df) > 0:
            code_col = "ts_code" if "ts_code" in df.columns else "con_code"
            out = df[[code_col]].copy()
            out.columns = ["ts_code"]
            out["in_date"] = df["in_date"] if "in_date" in df.columns else None
            out["out_date"] = df["out_date"] if "out_date" in df.columns else None
            out["ind_name"] = ind_name
            # 申万二级行业名 —— 板块层分析的基础。用 L1 查询时返回的成分股
            # 分属多个二级行业，只记 L1 名字会把几百只股票压成一个"板块"。
            out["l2_name"] = (df["l2_name"] if "l2_name" in df.columns
                              else (ind_name if level == "L2" else None))
            rows.append(out)
            return
        df = api_call(pro.index_member, lim, retries=1, index_code=code)
        if df is not None and len(df) > 0:
            code_col = "con_code" if "con_code" in df.columns else "ts_code"
            out = df[[code_col]].copy()
            out.columns = ["ts_code"]
            out["in_date"] = df["in_date"] if "in_date" in df.columns else None
            out["out_date"] = df["out_date"] if "out_date" in df.columns else None
            out["ind_name"] = ind_name
            out["l2_name"] = ind_name if level == "L2" else None
            rows.append(out)

    for lvl, names in (("L1", l1_names), ("L2", l2_names)):
        sub = cls[(cls.get("level") == lvl) & (cls[name_col].isin(names))]
        for _, r in sub.iterrows():
            _grab(r["index_code"], lvl, r[name_col])

    if not rows:
        return pd.DataFrame(columns=["ts_code", "ind_name", "in_date", "out_date"])

    uni = pd.concat(rows, ignore_index=True)
    uni = uni.dropna(subset=["ts_code"])
    uni = uni[~uni["ts_code"].str.endswith(".BJ")]          # 剔除北交所（流动性差、涨跌幅30%）
    uni = uni.sort_values(["ts_code", "in_date"]).drop_duplicates("ts_code", keep="first")
    return uni.reset_index(drop=True)


def fetch_stock_basic(pro, lim: Limiter) -> pd.DataFrame:
    """含退市/暂停上市，避免幸存者偏差。"""
    parts = []
    for status in ["L", "D", "P"]:
        df = api_call(pro.stock_basic, lim, exchange="", list_status=status,
                      fields="ts_code,name,industry,market,list_date,delist_date")
        if df is not None and len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["ts_code", "name", "list_date", "delist_date"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("ts_code")


def prescreen_by_mv(pro, lim: Limiter, codes: List[str], start: str, end: str,
                    mv_lo: float, mv_hi: float, n_samples: int = 16) -> List[str]:
    """
    用全市场快照做市值预筛：只保留「历史上曾经落在市值区间（放宽后）」的股票。
    每个采样日一次全市场 daily_basic，16 次调用就能砍掉三分之一以上的下载量。
    用的是各采样日的当期市值，不是今天的市值，所以不引入前视偏差；
    代价是两个采样点之间短暂进出区间的极少数股票会被漏掉。
    """
    cal = api_call(pro.trade_cal, lim, exchange="SSE", start_date=start,
                   end_date=end, is_open="1")
    if cal is None or len(cal) == 0:
        return codes
    days = sorted(cal["cal_date"].astype(str).tolist())
    if len(days) <= n_samples:
        picks = days
    else:
        idx = np.linspace(0, len(days) - 1, n_samples).astype(int)
        picks = [days[i] for i in idx]

    ever = set()
    lo_g, hi_g = mv_lo * 1e4 * 0.6, mv_hi * 1e4 * 1.5     # 上下各留足余量
    for d in picks:
        df = api_call(pro.daily_basic, lim, trade_date=d, fields="ts_code,circ_mv")
        if df is None or len(df) == 0:
            continue
        m = (df["circ_mv"] >= lo_g) & (df["circ_mv"] <= hi_g)
        ever |= set(df.loc[m, "ts_code"].tolist())
    if not ever:
        return codes
    kept = [c for c in codes if c in ever]
    return kept if len(kept) >= 50 else codes


def _px_path(ts_code: str) -> str:
    return os.path.join(PX_DIR, ts_code.replace(".", "_") + ".pkl")


PX_COLS = ["trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "amount", "circ_mv"]


def fetch_one_stock(pro_get: Callable, lim: Limiter, ts_code: str, start: str, end: str,
                    use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    单只股票的日线 + 每日指标，落地磁盘缓存（每股一个文件，中途崩溃可续传）。
    注意 Tushare 的 pre_close 已做除权处理，pct_chg 因此是正确的复权收益率，
    所以不需要额外拉 adj_factor。high/low 本系统用不到，不下载也不保存。
    """
    path = _px_path(ts_code)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
            # 必须比对「请求区间」而不是「数据区间」：股票首个交易日几乎不会
            # 正好等于请求起始日（节假日、上市较晚、已退市），拿数据区间去比会
            # 导致永远未命中、每次全量重下。
            if isinstance(blob, dict) and "df" in blob:
                if (blob.get("start", "99999999") <= start
                        and blob.get("end", "0") >= end):
                    return blob["df"]
        except Exception:
            pass

    pro = pro_get()
    d = api_call(pro.daily, lim, ts_code=ts_code, start_date=start, end_date=end)
    if d is None or len(d) == 0:
        return None
    b = api_call(pro.daily_basic, lim, ts_code=ts_code, start_date=start, end_date=end,
                 fields="trade_date,circ_mv")
    d = d[[c for c in PX_COLS if c in d.columns]].copy()
    if b is not None and len(b):
        d = d.merge(b[["trade_date", "circ_mv"]], on="trade_date", how="left")
    else:
        d["circ_mv"] = np.nan
    d["trade_date"] = pd.to_datetime(d["trade_date"], format="%Y%m%d")
    for c in d.columns:
        if c != "trade_date":
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(np.float32)   # 内存减半
    d = d.sort_values("trade_date").reset_index(drop=True)
    try:
        with open(path, "wb") as f:
            pickle.dump({"start": start, "end": end, "df": d}, f, protocol=4)
    except Exception:
        pass
    return d


def download_all(token: str, codes: List[str], start: str, end: str, lim: Limiter,
                 use_cache: bool, workers: int,
                 progress_cb: Optional[Callable] = None) -> Dict[str, pd.DataFrame]:
    """
    多线程下载。瓶颈是单次请求的网络往返（约 0.5-1 秒），不是频次上限，
    所以并发能把吞吐从「延迟受限」拉到「频次受限」，4 线程通常快 3-4 倍。
    限流器全局共享，账号层面的频次不会被突破。
    """
    import tushare as ts

    local = threading.local()

    def pro_get():
        if not hasattr(local, "pro"):
            local.pro = ts.pro_api(token)     # 每线程一个客户端，不共享连接
        return local.pro

    out: Dict[str, pd.DataFrame] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(fetch_one_stock, pro_get, lim, c, start, end, use_cache): c
                for c in codes}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                d = fut.result()
            except Exception as e:
                d = None
                with _ERR_LOCK:
                    API_ERRORS.append(f"{c}: {e}")
            if d is not None and len(d) > 30:
                out[c] = d
            done += 1
            if progress_cb is not None and (done % 10 == 0 or done == len(codes)):
                progress_cb(done, len(codes), len(out))   # 回调只在主线程消费
    return out


def build_panel(px: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    把逐股票的长表拼成宽表面板，并构造复权价。
    adj_close[t] = adj_close[t-1] * (1 + pct_chg/100)
    adj_open[t]  = adj_close[t-1] * open[t] / pre_close[t]
    """
    codes = sorted([c for c, d in px.items() if d is not None and len(d) > 30])
    if not codes:
        raise ValueError("没有可用的价格数据")

    idxed = {c: px[c].set_index("trade_date") for c in codes}

    def wide(col: str) -> pd.DataFrame:
        cols = [idxed[c][col] if col in idxed[c].columns
                else pd.Series(dtype=np.float32) for c in codes]
        w = pd.concat(cols, axis=1, keys=codes).sort_index()
        return w.astype(np.float32)

    raw_close = wide("close")
    amount = wide("amount")            # 单位: 千元
    circ_mv = wide("circ_mv")          # 单位: 万元

    cal = raw_close.index
    tradable = raw_close.notna()       # 有行情=可交易，NaN 视为停牌

    # 复权价：cumprod 用 float64 累乘 2000 步以免误差累积，存回 float32
    pct = wide("pct_chg")
    adj_close = (1.0 + pct.astype(np.float64).fillna(0.0) / 100.0).cumprod()
    adj_close = adj_close.where(tradable).ffill()
    del pct

    raw_open = wide("open")
    pre_close = wide("pre_close")
    # adj_open 只用来当成交价，不参与链式累乘，降到 float32 无妨；
    # adj_close 逐日累乘且要反复做 pct_change（相近数相减会放大误差），保持 float64。
    ratio = adj_close.shift(1) / pre_close.where(pre_close > 0)   # 当日复权换算比例
    adj_open = (raw_open * ratio).astype(np.float32)
    adj_high = (wide("high") * ratio).astype(np.float32)           # SKDJ 需要
    adj_low = (wide("low") * ratio).astype(np.float32)

    # 涨跌停判定（创业板/科创板 20%，其余 10%）
    lim_pct = pd.Series([0.20 if (c.startswith("30") or c.startswith("688")) else 0.10
                         for c in codes], index=codes, dtype=np.float32)
    limit_up_open = (raw_open >= pre_close.mul(1.0 + lim_pct, axis=1) - 0.004) & tradable
    limit_dn_open = (raw_open <= pre_close.mul(1.0 - lim_pct, axis=1) + 0.004) & tradable
    del pre_close, idxed                # 之后再也用不到，立刻释放
    gc.collect()

    return dict(cal=cal, codes=codes,
                raw_close=raw_close, raw_open=raw_open, amount=amount, circ_mv=circ_mv,
                adj_close=adj_close, adj_open=adj_open,
                adj_high=adj_high, adj_low=adj_low, tradable=tradable,
                limit_up_open=limit_up_open, limit_dn_open=limit_dn_open)


# ======================================================================
# 二、因子计算
# ======================================================================
def compute_factors(adj_close: pd.DataFrame, amount: pd.DataFrame, win: int = 60,
                    circ_mv: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """全部因子一次性向量化算完，返回 {因子名: 宽表}。"""
    A = adj_close
    out: Dict[str, pd.DataFrame] = {}

    ret = A.pct_change()
    vol = ret.rolling(win).std() * np.sqrt(252.0)
    out["mom_ra"] = ((A.shift(5) / A.shift(win + 5) - 1.0) / vol.where(vol > 1e-9))
    del ret, vol

    # 滚动线性回归: 斜率与 R²。窗口内自变量取全局序号，相关性对平移不变。
    # 方差用 E[y²]-E[y]² 会有抵消误差，float32 精度不够，这一段必须走 float64。
    logp = np.log(A.where(A > 0)).astype(np.float64)
    t = pd.Series(np.arange(len(A.index), dtype=np.float64), index=A.index)
    my = logp.rolling(win).mean()
    mt = t.rolling(win).mean()
    cov_ty = logp.mul(t, axis=0).rolling(win).mean().sub(my.mul(mt, axis=0))
    var_t = (win * win - 1.0) / 12.0
    var_y = (logp ** 2).rolling(win).mean() - my ** 2
    r2 = (cov_ty ** 2) / (var_t * var_y.where(var_y > 1e-12))
    out["trend_q"] = (cov_ty / var_t) * 252.0 * r2.clip(0.0, 1.0)
    del logp, my, mt, cov_ty, var_y, r2, t

    r60 = A / A.shift(win) - 1.0
    out["rel_str"] = r60.sub(r60.mean(axis=1), axis=0)
    del r60

    a60 = amount.rolling(win).mean()
    out["vol_exp"] = amount.rolling(5).mean() / a60.where(a60 > 1e-9)
    del a60

    out["dist_hi"] = A / A.rolling(win).max() - 1.0
    out["rev5"] = A / A.shift(5) - 1.0

    # 诊断因子
    if circ_mv is not None:
        out["logsize"] = np.log(circ_mv.where(circ_mv > 0))
    ar = (A / A.shift(1) - 1.0).abs()
    out["amihud"] = (ar / amount.where(amount > 1e-9)).rolling(20).mean() * 1e6
    del ar

    for k in list(out):
        out[k] = out[k].astype(np.float32)
    gc.collect()
    return out


def build_eligibility(panel: dict, basic: pd.DataFrame, uni: pd.DataFrame,
                      mv_lo: float, mv_hi: float, min_price: float,
                      min_amt_yi: float, min_list_days: int) -> pd.DataFrame:
    """
    时点可选性矩阵（日期 × 股票）。全部按当期数据判断，不用今天的市值回看历史。
    mv_lo/mv_hi 单位: 亿元。min_amt_yi 单位: 亿元。
    """
    cal = panel["cal"]
    codes = panel["codes"]
    cmv = panel["circ_mv"]        # 万元
    rc = panel["raw_close"]
    amt = panel["amount"]         # 千元

    ok = panel["tradable"].copy()
    # circ_mv 来自 daily_basic，它比 daily 发布得晚 —— 收盘后先有价、后有市值。
    # 那一天市值全空会让所有股票市值筛选不通过，候选直接崩掉。
    # 市值是慢变量，用前几天的完全够用；ffill 只取过去的值，不含未来信息。
    cmv = cmv.where(panel["tradable"]).ffill(limit=5)
    ok &= cmv.notna() & (cmv >= mv_lo * 1e4) & (cmv <= mv_hi * 1e4)
    ok &= rc >= min_price
    ok &= amt.rolling(20).mean() >= min_amt_yi * 1e5

    # ---- 以下全部按整块广播计算，不逐只股票循环 ----
    cal_col = cal.to_numpy(dtype="datetime64[ns]").reshape(-1, 1)
    FAR_PAST = np.datetime64("1990-01-01")
    FAR_FUTURE = np.datetime64("2099-01-01")

    def _dates(df: pd.DataFrame, col: str, fill) -> np.ndarray:
        """按 codes 顺序取一列日期，缺失填 fill，返回 (1, N) 便于广播。"""
        if not len(df) or col not in df.columns:
            return np.full((1, len(codes)), fill, dtype="datetime64[ns]")
        s = df.drop_duplicates("ts_code").set_index("ts_code")[col]
        s = pd.to_datetime(s.reindex(codes).astype(str), format="%Y%m%d", errors="coerce")
        return s.fillna(pd.Timestamp(fill)).to_numpy(dtype="datetime64[ns]").reshape(1, -1)

    # 上市满 N 天 + 未退市
    ld = _dates(basic, "list_date", FAR_PAST) + np.timedelta64(int(min_list_days), "D")
    dd = _dates(basic, "delist_date", FAR_FUTURE) - np.timedelta64(5, "D")
    mask = (cal_col >= ld) & (cal_col < dd)

    # 申万成分进出日期（时点行业归属）
    if len(uni):
        mask &= (cal_col >= _dates(uni, "in_date", FAR_PAST))
        mask &= (cal_col < _dates(uni, "out_date", FAR_FUTURE))

    ok &= pd.DataFrame(mask, index=cal, columns=codes)

    # 非 ST / 非退市整理股（按当前名称判断，见界面说明）
    if len(basic) and "name" in basic.columns:
        nm = basic.drop_duplicates("ts_code").set_index("ts_code")["name"].reindex(codes).fillna("")
        bad = nm.str.upper().str.contains("ST") | nm.str.contains("退")
        if bad.any():
            ok.loc[:, bad.to_numpy()] = False

    return ok.fillna(False)


def cs_zscore(df: pd.DataFrame, mask: pd.DataFrame, wins: float = 0.01) -> pd.DataFrame:
    """截面去极值 + 标准化，只在 mask 为真的样本上做。"""
    x = df.where(mask)
    lo = x.quantile(wins, axis=1)
    hi = x.quantile(1.0 - wins, axis=1)
    x = x.clip(lower=lo, upper=hi, axis=0)
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return x.sub(mu, axis=0).div(sd.where(sd > 1e-12), axis=0)


def composite_score(factors: Dict[str, pd.DataFrame], elig: pd.DataFrame,
                    weights: Dict[str, float]) -> pd.DataFrame:
    total = None
    wsum = sum(abs(w) for w in weights.values() if abs(w) > 1e-9)
    if wsum < 1e-9:
        wsum = 1.0
    for k, w in weights.items():
        if abs(w) < 1e-9 or k not in factors:
            continue
        z = cs_zscore(factors[k], elig)
        total = z * w if total is None else total.add(z * w, fill_value=0.0)
    if total is None:
        total = pd.DataFrame(np.nan, index=elig.index, columns=elig.columns)
    return (total / wsum).where(elig)


def weekly_rebal_dates(cal: pd.DatetimeIndex, every_n_weeks: int = 1) -> List[pd.Timestamp]:
    """每周最后一个交易日。"""
    s = pd.Series(cal, index=cal)
    wk = s.groupby([cal.isocalendar().year, cal.isocalendar().week]).max()
    ds = sorted(pd.DatetimeIndex(wk.values))
    return list(ds[::max(1, every_n_weeks)])


# ======================================================================
# 三、因子分层检验（写任何买卖逻辑之前先跑这个）
# ======================================================================
def _spearman(a: np.ndarray, b: np.ndarray, min_n: int = 20) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return np.nan
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def newey_west_t(x: pd.Series, lag: int) -> float:
    """
    IC 序列的 Newey-West 修正 t 值。
    周频调仓 + H 周持有期时，相邻 H-1 期的样本是重叠的，IC 序列有自相关，
    直接用 ICIR×√n 会把显著性高估约 √H 倍。这里按 lag=H-1 做 HAC 修正。
    """
    v = x.dropna().to_numpy(dtype=np.float64)
    n = len(v)
    if n < 12:
        return np.nan
    mu = v.mean()
    e = v - mu
    var = float(e @ e) / n                       # γ0
    for k in range(1, min(int(lag), n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if var <= 0:
        return np.nan
    return float(mu / np.sqrt(var / n))


def layered_test(fac: pd.DataFrame, adj_close: pd.DataFrame, elig: pd.DataFrame,
                 rebal: List[pd.Timestamp], horizon_weeks: int = 4,
                 n_group: int = 10, gap: int = 1) -> dict:
    """
    分层检验：每个调仓日按因子值分 n_group 组，看未来 horizon_weeks 周收益是否单调。

    gap = 入场延迟的交易日数，是这里最重要的参数。
    gap=0 时，因子的分子和未来收益的分母共用排名当日的价格 A[i]。A[i] 里任何
    噪音（买卖价差跳动、单日过度反应）都会同时抬高因子、压低未来收益，凭空造出
    负相关——反转研究里的经典陷阱。真实的反转效应能扛住延迟入场，微观结构噪音
    扛不住。默认 gap=1，与回测「次日开盘成交」的口径一致。
    """
    cal = adj_close.index
    pos = {d: i for i, d in enumerate(cal)}
    step = horizon_weeks * 5
    rows, ex_rows, ic_rows = [], [], []

    for d in rebal:
        i = pos.get(d)
        if i is None or i + gap + step >= len(cal):
            continue
        f = fac.iloc[i].where(elig.iloc[i]).to_numpy(dtype=float)
        j = i + gap                                    # 延迟 gap 个交易日才入场
        fwd = (adj_close.iloc[j + step] / adj_close.iloc[j] - 1.0).to_numpy(dtype=float)
        m = np.isfinite(f) & np.isfinite(fwd)
        if m.sum() < n_group * 3:
            continue
        fv, rv = f[m], fwd[m]
        grp = pd.qcut(pd.Series(fv).rank(method="first"), n_group,
                      labels=False, duplicates="drop").to_numpy()
        means = [np.nanmean(rv[grp == g]) if (grp == g).sum() else np.nan for g in range(n_group)]
        cols = [f"D{g+1}" for g in range(n_group)]
        rows.append(pd.Series(means, index=cols, name=d))
        # 相对当期截面均值的超额 —— 纯多头真正能吃到的部分
        mkt = float(np.nanmean(rv))
        ex_rows.append(pd.Series([m - mkt for m in means], index=cols, name=d))
        ic_rows.append(pd.Series({"date": d, "ic": _spearman(fv, rv)}))

    if not rows:
        return {"ok": False}

    grp_df = pd.DataFrame(rows)
    ex_df = pd.DataFrame(ex_rows)
    ic = pd.DataFrame(ic_rows).set_index("date")["ic"].dropna()

    # 不重叠累计曲线
    sub = grp_df.iloc[::max(1, horizon_weeks)]
    curve = (1.0 + sub).cumprod()

    ic_mean = float(ic.mean()) if len(ic) else np.nan
    ic_std = float(ic.std()) if len(ic) else np.nan
    icir = ic_mean / ic_std if ic_std and ic_std > 1e-12 else np.nan
    tstat = icir * np.sqrt(len(ic)) if np.isfinite(icir) else np.nan
    t_nw = newey_west_t(ic, lag=max(0, horizon_weeks - 1))      # 重叠窗口修正后
    ic_year = ic.groupby(ic.index.year).mean() if len(ic) else pd.Series(dtype=float)
    ic_pos = float((ic > 0).mean()) if len(ic) else np.nan
    top, bot = grp_df.columns[-1], grp_df.columns[0]
    # 钱在哪一端: IC 为负则做多最低组(D1), 为正则做多最高组(D10)。
    # A股融券做空不现实, 所以只有多头那一端的超额才是真正能拿到的。
    long_side = bot if (np.isfinite(ic_mean) and ic_mean < 0) else top
    short_side = top if long_side == bot else bot
    long_ex = float(ex_df[long_side].mean())
    short_ex = float(ex_df[short_side].mean())
    tot = abs(long_ex) + abs(short_ex)
    long_share = (abs(long_ex) / tot) if tot > 1e-12 else np.nan
    # 单调性: 各组均值与组序号的秩相关
    ordered = grp_df.mean(axis=0).to_numpy()
    mono = _spearman(np.arange(len(ordered), dtype=float), ordered, min_n=4)

    return {"ok": True, "group_mean": grp_df.mean(axis=0), "curve": curve, "ic": ic,
            "ic_mean": ic_mean, "icir": icir, "tstat": tstat, "t_nw": t_nw,
            "ic_year": ic_year, "ic_pos": ic_pos,
            "group_excess": ex_df.mean(axis=0), "long_side": long_side,
            "long_excess": long_ex, "short_excess": short_ex,
            "long_share": long_share,
            "spread": float(grp_df[top].mean() - grp_df[bot].mean()),
            "monotonic": mono, "n_period": len(grp_df)}


# ======================================================================
# 四、回测引擎
# ======================================================================


# ======================================================================
# 五、名次段检验 —— 只持 1-3 只时，必须知道最极端的那几名是好是坏
# ======================================================================
RANK_BANDS = [(1, 3), (4, 10), (11, 20), (21, 40), (41, 80), (81, 150)]


def rank_band_test(score: pd.DataFrame, adj_close: pd.DataFrame, elig: pd.DataFrame,
                   rebal: List[pd.Timestamp], horizon_weeks: int = 4, gap: int = 1,
                   bands=RANK_BANDS) -> pd.DataFrame:
    """
    按名次分段看未来收益（相对当期截面均值的超额）。

    分层检验测的是 D1 组（前 10%，上百只）的平均值。只买前 3 名时，买的是
    前 0.2%，那是我们从没测过的区间。最极端的几只可能是真出事的公司，
    不是被错杀的——这个函数就是查这件事。
    """
    cal = adj_close.index
    pos = {d: i for i, d in enumerate(cal)}
    step = horizon_weeks * 5
    rows = []
    for d in rebal:
        i = pos.get(d)
        if i is None or i + gap + step >= len(cal):
            continue
        sc = score.iloc[i].where(elig.iloc[i])
        sc = sc.dropna().sort_values(ascending=False)
        if len(sc) < max(b[1] for b in bands):
            continue
        j = i + gap
        fwd = (adj_close.iloc[j + step] / adj_close.iloc[j] - 1.0)
        mkt = float(fwd.reindex(sc.index).mean())
        rec = {"date": d}
        for lo, hi in bands:
            names = sc.index[lo - 1:hi]
            v = fwd.reindex(names).mean()
            rec[f"第{lo}-{hi}名"] = (float(v) - mkt) if pd.notna(v) else np.nan
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date")
    out = pd.DataFrame({
        "平均超额": df.mean(),
        "胜率": (df > 0).mean(),
        "标准误": df.std() / np.sqrt(df.notna().sum()),
    })
    out["t值"] = out["平均超额"] / out["标准误"].where(out["标准误"] > 1e-12)
    return out


def picked_path_stats(score: pd.DataFrame, adj_close: pd.DataFrame, elig: pd.DataFrame,
                      rebal: List[pd.Timestamp], top_n: int = 3, weeks: int = 8,
                      gap: int = 1) -> pd.DataFrame:
    """
    选出的股票在之后每周的「最大涨幅 / 最大跌幅」分布。
    用户自己设止盈止损，这张表就是定位的依据：止损设在历史上大多数
    正常波动之外，才不会被噪音扫出局。
    """
    cal = adj_close.index
    pos = {d: i for i, d in enumerate(cal)}
    recs = []
    for d in rebal:
        i = pos.get(d)
        if i is None or i + gap + weeks * 5 >= len(cal):
            continue
        sc = score.iloc[i].where(elig.iloc[i]).dropna().sort_values(ascending=False)
        if len(sc) < top_n:
            continue
        names = list(sc.index[:top_n])
        j = i + gap
        base = adj_close.iloc[j].reindex(names)
        for w in range(1, weeks + 1):
            seg = adj_close.iloc[j:j + w * 5].reindex(columns=names)
            up = (seg.max() / base - 1.0)
            dn = (seg.min() / base - 1.0)
            end = (adj_close.iloc[j + w * 5].reindex(names) / base - 1.0)
            for c in names:
                recs.append({"周": w, "最大涨幅": up.get(c), "最大跌幅": dn.get(c),
                             "期末收益": end.get(c)})
    if not recs:
        return pd.DataFrame()
    r = pd.DataFrame(recs)
    g = r.groupby("周")
    return pd.DataFrame({
        "最大涨幅_中位": g["最大涨幅"].median(),
        "最大跌幅_中位": g["最大跌幅"].median(),
        "最大跌幅_25分位": g["最大跌幅"].quantile(0.25),
        "期末收益_中位": g["期末收益"].median(),
        "期末为正比例": g["期末收益"].apply(lambda x: (x > 0).mean()),
    })


def empty_week_stats(score: pd.DataFrame, elig: pd.DataFrame,
                     rebal: List[pd.Timestamp], top_n: int = 3,
                     warmup_days: int = 70) -> pd.Series:
    """
    每年有多少周选不满 top_n 只。排名系统理论上恒为 0，但要用数据确认。
    前 warmup_days 个交易日是因子预热期（60日窗口尚未填满），那段没有分数
    不是真空窗，要排除，否则会把预热期误报成系统缺陷。
    """
    if len(score.index) > warmup_days:
        first_ok = score.index[warmup_days]
        rebal = [d for d in rebal if d >= first_ok]
    rows = []
    for d in rebal:
        if d not in score.index:
            continue
        n = int(score.loc[d].where(elig.loc[d]).notna().sum())
        rows.append({"date": d, "不足": 1 if n < top_n else 0})
    if not rows:
        return pd.Series(dtype=int)
    df = pd.DataFrame(rows).set_index("date")
    return df.groupby(df.index.year)["不足"].sum()


def clear_day_cache() -> int:
    """清空行情缓存，返回删除的文件数。"""
    n = 0
    try:
        for f in os.listdir(PX_DIR):
            if f.endswith(".pkl"):
                try:
                    os.remove(os.path.join(PX_DIR, f))
                    n += 1
                except Exception:
                    pass
    except Exception:
        pass
    return n


def day_cache_info() -> tuple:
    """返回 (缓存天数, 占用MB, 最早日期, 最晚日期)。"""
    try:
        fs = [f for f in os.listdir(PX_DIR) if f.endswith(".pkl")]
    except Exception:
        return 0, 0.0, None, None
    if not fs:
        return 0, 0.0, None, None
    mb = sum(os.path.getsize(os.path.join(PX_DIR, f)) for f in fs) / 1e6
    return len(fs), mb, None, None


# ======================================================================
# 板块层 —— 把选股单位从个股换成行业，目的是降噪
# ======================================================================
def build_sector_map(uni: pd.DataFrame, panel: dict, elig: pd.DataFrame,
                     min_members: int = 5) -> Dict[str, List[str]]:
    """
    code → 申万二级行业。成分股不足 min_members 的板块并入"其他"，
    因为几只股票的等权指数降不了多少噪音，还会制造伪板块。
    """
    if "l2_name" not in uni.columns:
        return {}
    m = uni.dropna(subset=["l2_name"]).drop_duplicates("ts_code")
    m = m[m["ts_code"].isin(panel["codes"])]
    grp: Dict[str, List[str]] = {}
    for sec, g in m.groupby("l2_name"):
        codes = [c for c in g["ts_code"] if c in panel["codes"]]
        # 用历史平均合格数判断板块够不够大
        if len(codes) and float(elig[codes].sum(axis=1).mean()) >= min_members:
            grp[str(sec)] = codes
    return grp


def build_sector_index(panel: dict, elig: pd.DataFrame,
                       sectors: Dict[str, List[str]]) -> tuple:
    """
    每个板块的等权指数（只用当期合格成分股，无幸存者偏差）。
    返回 (板块日收益表, 板块指数, 每日成分股数)。
    """
    ret = panel["adj_close"].pct_change()
    rows, cnts = {}, {}
    for sec, codes in sectors.items():
        m = elig[codes]
        r = ret[codes].where(m)
        n = m.sum(axis=1)
        rows[sec] = r.mean(axis=1).where(n >= 3)
        cnts[sec] = n
    R = pd.DataFrame(rows)
    IDX = (1.0 + R.fillna(0.0)).cumprod().where(R.notna()).ffill()
    return R, IDX, pd.DataFrame(cnts)


def sector_noise_check(panel: dict, elig: pd.DataFrame,
                       sectors: Dict[str, List[str]]) -> pd.DataFrame:
    """
    最关键的前置检验：板块指数的波动到底比个股小多少？
    如果降噪幅度不明显，"换单位"这个思路就不成立，后面不用做了。
    """
    ret = panel["adj_close"].pct_change()
    R, _, cnt = build_sector_index(panel, elig, sectors)
    rows = []
    for sec, codes in sectors.items():
        m = elig[codes]
        iv = ret[codes].where(m).std().mean() * np.sqrt(252)     # 成分股平均年化波动
        sv = R[sec].std() * np.sqrt(252)                          # 板块指数年化波动
        rows.append({"板块": sec, "平均成分股数": float(m.sum(axis=1).mean()),
                     "个股平均波动": float(iv), "板块指数波动": float(sv),
                     "降噪比": float(sv / iv) if iv > 0 else np.nan})
    d = pd.DataFrame(rows).set_index("板块").sort_values("平均成分股数", ascending=False)
    return d


def sector_factors(R: pd.DataFrame, IDX: pd.DataFrame,
                   amt_sec: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """板块层面的候选信号。全部只用过去数据。"""
    out = {}
    for w in (5, 10, 20, 60):
        out[f"板块{w}日动量"] = IDX / IDX.shift(w) - 1.0
    vol = R.rolling(20).std() * np.sqrt(252)
    out["板块风险调整动量"] = (IDX / IDX.shift(20) - 1.0) / vol.where(vol > 1e-9)
    # 注：曾有"板块相对强度"= 20日动量减截面均值，那是单调变换，
    # 截面排序与20日动量完全相同，等于同一个信号数了两遍，已删除。
    # 保留"创20日新高"：它单调性 −0.91、方向与动量相反，是有信息的反向对照。
    # 已移除「板块距60日高点」(单调性0.54,t−0.05) 和
    #        「板块成交额占比变化」(单调性−0.09,Q4−Q1 +0.02%) —— 两个都≈0。
    out["板块创20日新高"] = (IDX >= IDX.rolling(20).max()).astype(float)
    a5, a60 = amt_sec.rolling(5).mean(), amt_sec.rolling(60).mean()
    out["板块量能扩张"] = a5 / a60.where(a60 > 1e-9)
    return out


def sector_layer_test(fac: pd.DataFrame, IDX: pd.DataFrame, horizons=(3, 5, 8, 15),
                      n_group: int = 4, step: int = 5) -> pd.DataFrame:
    """
    板块分层检验：按信号把板块分组，看未来 N 个交易日板块指数的表现。
    这是"板块信号本身有没有预测力"的直接检验 —— 不涉及选股。
    收益取相对全部板块均值的超额。
    """
    dates = list(IDX.index)
    rows = []
    for i in range(120, len(dates) - max(horizons) - 1, step):
        f = fac.iloc[i]
        v = f.dropna()
        if len(v) < n_group * 2:
            continue
        try:
            g = pd.qcut(v.rank(method="first"), n_group, labels=False)
        except Exception:
            continue
        rec = {"date": dates[i]}
        for h in horizons:
            fwd = IDX.iloc[i + h] / IDX.iloc[i] - 1.0
            mkt = float(fwd.reindex(v.index).mean())
            for q in range(n_group):
                names = v.index[g == q]
                x = fwd.reindex(names).mean()
                rec[f"Q{q+1}_{h}日"] = (float(x) - mkt) if pd.notna(x) else np.nan
        rows.append(rec)
    if len(rows) < 50:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date")
    out = []
    for q in range(n_group):
        r = {"分组": f"Q{q+1}"}
        for h in horizons:
            col = f"Q{q+1}_{h}日"
            v = df[col].dropna()
            r[f"{h}日超额"] = v.mean()
            r[f"{h}日胜率"] = (v > 0).mean()
            if h == horizons[-1]:
                # 必须做重叠修正：前瞻 h 个交易日、每 step 天采样一次，
                # 相邻样本重叠 h/step 倍。用朴素标准误会把 t 放大约 sqrt(h/step)。
                lag = max(1, int(np.ceil(h / max(step, 1))))
                r["末期t(重叠修正)"] = newey_west_t(v, lag=lag)
                se = v.std(ddof=1) / np.sqrt(len(v))
                r["末期t(朴素)"] = v.mean() / se if se > 1e-12 else np.nan
        out.append(r)
    return pd.DataFrame(out).set_index("分组")


# ---------------- 买入位置过滤（日线 SKDJ K 值）----------------
def daily_kd(panel: dict, n: int = 9, m: int = 3):
    """日线 SKDJ 的 K 和 D。买入当天就已知，不含未来数据。"""
    A, H, L = panel["adj_close"], panel["adj_high"], panel["adj_low"]
    lo, hi = L.rolling(n).min(), H.rolling(n).max()
    rng = (hi - lo).where((hi - lo) > 1e-9)
    ema = lambda d, p: d.ewm(span=p, adjust=False, min_periods=p).mean()
    rsv = ema((A - lo) / rng * 100.0, m)
    k = ema(rsv, m)
    d = k.rolling(m).mean()
    return k.astype(np.float32), d.astype(np.float32)


def daily_k(panel: dict, n: int = 9, m: int = 3) -> pd.DataFrame:
    return daily_kd(panel, n, m)[0]


# 曾有 K1-K7 八个"带替补的买入位置过滤"，全部已移除：
#   八个条件里不过滤最好(+2.34%)，两个相反方向的过滤都变差 0.6-0.7%；
#   干净划分证明其中 87% 的表观效果来自"替补挖得太深"，不是条件本身。
# 教训：任何"要不要加条件"的问题，都用 split_by_mask 做干净划分，
#       不要跑带替补的过滤版——那个对比是被污染的。


def high_dead_cross_mask(kdf: pd.DataFrame, dkf: pd.DataFrame, win: int = 5):
    """K 从 75 上方下穿 D 之后 win 天内 —— 供干净划分用，不做过滤。"""
    dead = (kdf < dkf) & (kdf.shift(1) >= dkf.shift(1)) & (kdf.shift(1) >= 75)
    return dead.rolling(win).max().fillna(0).astype(bool)


STOCK_RULES = ["S1_板块内最强", "S2_板块内最弱(回调)", "S3_板块内随机"]


def sector_then_stock(panel: dict, elig: pd.DataFrame, sectors: Dict[str, List[str]],
                      sec_fac: pd.DataFrame, dates: List[pd.Timestamp],
                      top_sec: int = 2, top_n: int = 3,
                      stock_rule: str = "S1_板块内最强",
                      sec_rule: str = "最强", cooldown: int = 5,
                      seed: int = 20260910, kdf: pd.DataFrame = None,
                      per_sec_cap: int = 0) -> pd.DataFrame:
    """
    两层选股：先按 sec_fac 选出 top_sec 个板块，再在板块内按 stock_rule 选股。
    sec_rule="随机" 时板块层用随机选择 —— 这是判断"板块层有没有加分"的对照组。
    """
    A = panel["adj_close"]
    m20 = (A / A.shift(20) - 1.0)
    rng = np.random.default_rng(seed)
    cal = list(A.index)
    pos = {d: i for i, d in enumerate(cal)}
    last: Dict[str, int] = {}
    rows = []
    secnames = list(sectors)
    for d in dates:
        i = pos.get(d)
        if i is None or d not in sec_fac.index:
            continue
        f = sec_fac.loc[d].dropna()
        if len(f) < top_sec + 1:
            continue
        if sec_rule == "随机":
            picks_sec = list(rng.choice(f.index, min(top_sec, len(f)), replace=False))
        else:
            picks_sec = list(f.sort_values(ascending=False).index[:top_sec])
        cand = []
        for s in picks_sec:
            codes = [c for c in sectors[s] if elig.loc[d, c]]
            if not codes:
                continue
            v = m20.loc[d, codes].dropna()
            if not len(v):
                continue
            if stock_rule == "S1_板块内最强":
                order = v.sort_values(ascending=False).index
            elif stock_rule == "S2_板块内最弱(回调)":
                order = v.sort_values(ascending=True).index
            else:
                order = list(rng.permutation(list(v.index)))
            for c in order:
                cand.append((s, c, float(v[c])))
        taken = 0
        used: Dict[str, int] = {}
        for s, c, sc in cand:
            if taken >= top_n:
                break
            if c in last and i - last[c] < cooldown:
                continue
            # 候选是按板块顺序排的（最强板块的股票全部在前），
            # 不设上限时前 top_n 名往往全部来自最强板块。
            # per_sec_cap>0 则强制分散。默认 0 = 不限，保持与已验证口径一致。
            if per_sec_cap > 0 and used.get(s, 0) >= per_sec_cap:
                continue
            used[s] = used.get(s, 0) + 1
            rows.append({"date": d, "code": c, "板块": s, "rank": taken + 1,
                         "score": sc, "买入K": (float(kdf.loc[d, c])
                                                if kdf is not None and c in kdf.columns
                                                else np.nan)})
            last[c] = i
            taken += 1
    return pd.DataFrame(rows)


def run_age_diagnosis(picks: pd.DataFrame, tr: pd.DataFrame, sec_fac: pd.DataFrame,
                      cal) -> pd.DataFrame:
    """
    按「买入时，该板块已经连续霸榜几天」给同一批成交分组。

    回答的是执行时机问题：龙头刚换就买 vs 它已经领跑几天才买，差别多大。
    这是对同一批交易做划分，不做替补，所以没有"往排名深处挖"的污染。
    """
    if not len(tr):
        return pd.DataFrame()
    days = [d for d in cal if d in sec_fac.index and sec_fac.loc[d].notna().any()]
    top = {}
    for d in days:
        top[d] = sec_fac.loc[d].idxmax()
    age, prev, cnt = {}, None, 0
    for d in days:                      # 当天龙头已连续第几天（只用过去，无前视）
        cnt = cnt + 1 if top[d] == prev else 1
        age[d] = cnt
        prev = top[d]

    m = picks[["date", "code", "板块"]].drop_duplicates(["date", "code"])
    d0 = tr.merge(m, on=["date", "code"], how="left", suffixes=("", "_p"))
    d0 = d0.dropna(subset=["收益率"]).copy()
    d0["霸榜天数"] = d0["date"].map(age)
    d0 = d0.dropna(subset=["霸榜天数"])
    if len(d0) < 100:
        return pd.DataFrame()
    b = [0, 1, 2, 3, 5, 8, 999]
    lab = ["第1天", "第2天", "第3天", "第4-5天", "第6-8天", "第9天以上"]
    d0["档"] = pd.cut(d0["霸榜天数"], bins=b, labels=lab, right=True)
    out = []
    for g, sub in d0.groupby("档", observed=True):
        day = sub.groupby("date")["收益率"].mean().sort_index()
        se = day.std(ddof=1) / np.sqrt(len(day)) if len(day) > 3 else np.nan
        out.append({"买入时机": g, "笔数": len(sub), "占比": len(sub) / len(d0),
                    "平均收益": sub["收益率"].mean(),
                    "中位收益": sub["收益率"].median(),
                    "胜率": float((sub["收益率"] > 0).mean()),
                    "聚类t": float(day.mean() / se) if se and se > 1e-12 else np.nan})
    return pd.DataFrame(out).set_index("买入时机")


def split_by_mask(picks: pd.DataFrame, tr: pd.DataFrame, mask: pd.DataFrame,
                   lab_yes: str = "是", lab_no: str = "否") -> pd.DataFrame:
    """
    把**同一批**成交按某个条件劈成两半，不做替补。

    这才是干净的对比：K6/K7 那种带替补的跑法，两边都掺进了顶上来的
    第4、5名，替补的代价和条件本身的效果混在一起，分不开。
    直接划分同一批交易就没有这个问题。
    """
    if not len(tr) or mask is None:
        return pd.DataFrame()
    d = tr.dropna(subset=["收益率"]).copy()
    flag = []
    for _, r in d.iterrows():
        try:
            flag.append(bool(mask.loc[r["date"], r["code"]]))
        except Exception:
            flag.append(np.nan)
    d["组"] = [lab_yes if f is True else (lab_no if f is False else None) for f in flag]
    d = d.dropna(subset=["组"])
    if len(d) < 100:
        return pd.DataFrame()
    out = []
    for g, sub in d.groupby("组"):
        day = sub.groupby("date")["收益率"].mean().sort_index()
        se = day.std(ddof=1) / np.sqrt(len(day)) if len(day) > 3 else np.nan
        out.append({"组": g, "笔数": len(sub), "占比": len(sub) / len(d),
                    "平均收益": sub["收益率"].mean(),
                    "中位收益": sub["收益率"].median(),
                    "胜率": float((sub["收益率"] > 0).mean()),
                    "聚类t": float(day.mean() / se) if se and se > 1e-12 else np.nan})
    return pd.DataFrame(out).set_index("组")


def k_bucket_diagnosis(picks: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """按买入当天的日线 K 值分档，看后续收益。直接检验「K>75 买入是否更差」。"""
    if not len(tr) or "买入K" not in picks.columns:
        return pd.DataFrame()
    m = picks[["date", "code", "买入K"]].drop_duplicates(["date", "code"])
    d = tr.merge(m, on=["date", "code"], how="left").dropna(subset=["买入K", "收益率"])
    if len(d) < 100:
        return pd.DataFrame()
    bins = [0, 40, 55, 65, 75, 85, 101]
    lab = ["<40", "40-55", "55-65", "65-75", "75-85", ">85"]
    d["档"] = pd.cut(d["买入K"], bins=bins, labels=lab, right=False)
    g = d.groupby("档", observed=True)["收益率"]
    out = pd.DataFrame({"笔数": g.size(), "平均收益": g.mean(),
                        "中位收益": g.median(), "胜率": g.apply(lambda x: (x > 0).mean())})
    out["占比"] = out["笔数"] / out["笔数"].sum()
    return out


def flat_stock_pick(panel: dict, elig: pd.DataFrame, dates: List[pd.Timestamp],
                    top_n: int = 3, stock_rule: str = "S1_板块内最强",
                    cooldown: int = 5, seed: int = 20260910) -> pd.DataFrame:
    """
    对照组：不分板块，直接在全池用同样的选股规则挑。
    与两层结果对比，才知道"板块层"本身有没有贡献。
    """
    A = panel["adj_close"]
    m20 = (A / A.shift(20) - 1.0)
    rng = np.random.default_rng(seed)
    cal = list(A.index)
    pos = {d: i for i, d in enumerate(cal)}
    last: Dict[str, int] = {}
    rows = []
    for d in dates:
        i = pos.get(d)
        if i is None:
            continue
        v = m20.loc[d].where(elig.loc[d]).dropna()
        if len(v) < top_n:
            continue
        if stock_rule == "S1_板块内最强":
            order = v.sort_values(ascending=False).index
        elif stock_rule == "S2_板块内最弱(回调)":
            order = v.sort_values(ascending=True).index
        else:
            order = list(rng.permutation(list(v.index)))
        taken = 0
        for c in order:
            if taken >= top_n:
                break
            if c in last and i - last[c] < cooldown:
                continue
            rows.append({"date": d, "code": c, "板块": "-", "rank": taken + 1,
                         "score": float(v[c])})
            last[c] = i
            taken += 1
    return pd.DataFrame(rows)


def track_fixed(picks: pd.DataFrame, panel: dict, hold_days: int = 8,
                comm: float = 0.0003, stamp: float = 0.0005,
                slip: float = 0.001) -> pd.DataFrame:
    """固定持有 hold_days 个交易日，次日开盘买、到期次日开盘卖。不设止盈止损。"""
    cal = panel["adj_close"].index
    ci = {c: j for j, c in enumerate(panel["codes"])}
    AO = panel["adj_open"].to_numpy(dtype=np.float32)
    RO = panel["raw_open"].to_numpy(dtype=np.float32)      # 真实开盘价，仅用于显示
    TRD = panel["tradable"].to_numpy(dtype=bool)
    LU = panel["limit_up_open"].to_numpy(dtype=bool)
    LD = panel["limit_dn_open"].to_numpy(dtype=bool)
    pos = {d: i for i, d in enumerate(cal)}
    cin, cout = comm + slip, comm + stamp + slip
    out = []
    for _, p in picks.iterrows():
        i0 = pos.get(p["date"]); j = ci.get(p["code"])
        if i0 is None or j is None:
            continue
        b = i0 + 1
        if b >= len(cal) or not TRD[b, j] or LU[b, j] or not np.isfinite(AO[b, j]):
            continue
        e = b + hold_days
        while e < len(cal) and (not TRD[e, j] or LD[e, j] or not np.isfinite(AO[e, j])):
            e += 1
            if e - b > hold_days + 5:
                break
        if e >= len(cal) or not np.isfinite(AO[e, j]):
            continue
        entry = float(AO[b, j]) * (1 + cin)
        exit_ = float(AO[e, j]) * (1 - cout)
        # 上面两个是**复权价**（起点归一化为1.0），用来算收益才正确——
        # 它把分红送股都还原了。但它不是你在交易软件上看到的价格，
        # 所以额外给出当日真实开盘价，方便你逐笔核对。
        out.append({"date": p["date"], "code": p["code"], "板块": p.get("板块", "-"),
                    "买入日": cal[b], "卖出日": cal[e],
                    "买入价(实际)": round(float(RO[b, j]), 2) if np.isfinite(RO[b, j]) else np.nan,
                    "卖出价(实际)": round(float(RO[e, j]), 2) if np.isfinite(RO[e, j]) else np.nan,
                    "收益率": exit_ / entry - 1.0, "持有交易日": e - b,
                    "买入价(复权)": round(entry, 4), "卖出价(复权)": round(exit_, 4)})
    return pd.DataFrame(out)


# ---------------- 三项必做诊断 ----------------
def _clustered(d: pd.DataFrame) -> float:
    day = d.groupby("date")["收益率"].mean().sort_index()
    se = day.std(ddof=1) / np.sqrt(len(day)) if len(day) > 3 else np.nan
    return float(day.mean() / se) if se and se > 1e-12 else np.nan


def split_check(tr: pd.DataFrame, cut: str = "2023-01-01") -> pd.DataFrame:
    """样本内 / 样本外。全样本好而样本外垮，是最常见的自欺方式。"""
    d = tr.dropna(subset=["收益率"]) if len(tr) else pd.DataFrame()
    if not len(d):
        return pd.DataFrame()
    c = pd.Timestamp(cut)
    rows = {}
    for lab, seg in (("样本内", d[d["date"] < c]), ("样本外", d[d["date"] >= c])):
        if len(seg) < 50:
            continue
        rows[lab] = {"笔数": len(seg), "平均收益": seg["收益率"].mean(),
                     "中位收益": seg["收益率"].median(),
                     "胜率": (seg["收益率"] > 0).mean(), "聚类t": _clustered(seg)}
    return pd.DataFrame(rows).T


def yearly_check(tr: pd.DataFrame) -> pd.DataFrame:
    """逐年。靠单独一年撑起来的平均值只是那一年的行情。"""
    d = tr.dropna(subset=["收益率"]) if len(tr) else pd.DataFrame()
    if not len(d):
        return pd.DataFrame()
    rows = {}
    for y, g in d.groupby(pd.to_datetime(d["date"]).dt.year):
        if len(g) < 30:
            continue
        rows[y] = {"笔数": len(g), "平均收益": g["收益率"].mean(),
                   "中位收益": g["收益率"].median(),
                   "胜率": (g["收益率"] > 0).mean(), "聚类t": _clustered(g)}
    return pd.DataFrame(rows).T


def concentration_check(tr: pd.DataFrame, ks=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    """利润集中度。绝大部分利润来自极少数几笔的话，那是彩票不是策略。"""
    d = tr.dropna(subset=["收益率"]) if len(tr) else pd.DataFrame()
    if len(d) < 50:
        return pd.DataFrame()
    v = d["收益率"].sort_values(ascending=False).to_numpy()
    tot = v.sum()
    rows = []
    for k in ks:
        if k >= len(v):
            continue
        rows.append({"最赚的前N笔": k,
                     "占总利润": v[:k].sum() / tot if abs(tot) > 1e-12 else np.nan,
                     "剔除后单笔均值": v[k:].mean(),
                     "剩余比例": v[k:].mean() / v.mean() if abs(v.mean()) > 1e-12 else np.nan})
    out = pd.DataFrame(rows).set_index("最赚的前N笔")
    out.attrs["原均值"] = float(v.mean())
    out.attrs["总笔数"] = int(len(v))
    return out


# ---------------- 避免挑选：合成信号 + 熊市开关 ----------------
MOM_FAMILY = ["板块5日动量", "板块10日动量", "板块20日动量",
              "板块60日动量", "板块风险调整动量"]


# ======================================================================
# 滚动前推检验 —— 唯一能挽救"样本外已被搜索用掉"的办法
# ======================================================================
def walk_forward(panel: dict, elig: pd.DataFrame, sectors: Dict[str, List[str]],
                 SF: Dict[str, pd.DataFrame], dates: List[pd.Timestamp],
                 signals: List[str] = None, top_secs=(2, 3), top_ns=(3,),
                 holds=(15, 20), start_year: int = 2021, per_sec_cap: int = 0,
                 comm: float = 0.0003, stamp: float = 0.0005,
                 slip: float = 0.001, progress=None) -> tuple:
    """
    模拟"你当年真的会怎么做"：
      每年年初，只用**截至上一年底**的数据，在全部配置里挑成绩最好的那个，
      然后用它跑这一年，只记录这一年的结果。第二年重新挑。

    这才是真正的样本外。全样本上挑一个最优配置再看它的"样本外"，
    等于用样本外做了选择——那个数字已经不算数了。
    """
    sigs = [s for s in (signals or list(SF)) if s in SF]
    cfgs = [(sg, ts, tn, hd) for sg in sigs for ts in top_secs
            for tn in top_ns for hd in holds]
    # 每个配置的全期成交只算一次，之后按年份切片即可
    allt: Dict[tuple, pd.DataFrame] = {}
    for i, (sg, ts, tn, hd) in enumerate(cfgs):
        pk = sector_then_stock(panel, elig, sectors, SF[sg], dates, ts, tn,
                               "S1_板块内最强", "最强", per_sec_cap=per_sec_cap)
        tr = track_fixed(pk, panel, hd, comm=comm, stamp=stamp, slip=slip) if len(pk) else pd.DataFrame()
        if len(tr):
            tr = tr.dropna(subset=["收益率"]).copy()
            tr["年"] = pd.to_datetime(tr["date"]).dt.year
            allt[(sg, ts, tn, hd)] = tr
        if progress:
            progress((i + 1) / len(cfgs), f"{sg} {ts}板块 {tn}只 {hd}日")

    years = sorted({y for t in allt.values() for y in t["年"].unique()})
    years = [y for y in years if y >= start_year]
    rows, picked = [], []
    for y in years:
        best, bcfg = -9e9, None
        for cfg, tr in allt.items():
            hist = tr[tr["年"] < y]
            if len(hist) < 150:
                continue
            v = hist["收益率"]
            se = v.std(ddof=1) / np.sqrt(len(v))
            score = v.mean() / se if se > 1e-12 else -9e9   # 只用历史挑
            if score > best:
                best, bcfg = score, cfg
        if bcfg is None:
            continue
        cur = allt[bcfg][allt[bcfg]["年"] == y]
        if not len(cur):
            continue
        _cap = f"|每板块≤{per_sec_cap}" if per_sec_cap > 0 else "|板块不限"
        picked.append({"年": y,
                       "选中配置": f"{bcfg[0]}|{bcfg[1]}板块|{bcfg[2]}只|{bcfg[3]}日{_cap}",
                       "历史t": best, "当年笔数": len(cur),
                       "当年平均收益": cur["收益率"].mean(),
                       "当年胜率": (cur["收益率"] > 0).mean()})
        rows.append(cur.assign(年份=y))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    wf = pd.concat(rows, ignore_index=True)
    return pd.DataFrame(picked).set_index("年"), wf


def wf_summary(wf: pd.DataFrame, hold_days: int = 20, step_days: int = 3) -> dict:
    """
    滚动前推的统计。给三个 t，从宽松到严格：

      按日聚类(朴素)   —— 相邻批次重叠，会被放大约 sqrt(持有期/选股间隔) 倍
      按日聚类(重叠修正) —— Newey-West，我此前在这里漏做了
      按年聚类        —— 只有几个独立年份，最保守，也最难自欺

    真正该看的是**按年**：滚动前推每年重新挑一次配置，年与年之间才是
    真正独立的观测。按日算会把同一年内高度重叠的持仓当成几百个独立样本。
    """
    if not len(wf):
        return {}
    v = wf["收益率"]
    day = wf.groupby("date")["收益率"].mean().sort_index()
    se = day.std(ddof=1) / np.sqrt(len(day)) if len(day) > 3 else np.nan
    lag = max(1, int(np.ceil(hold_days / max(step_days, 1))))
    yr = wf.groupby(pd.to_datetime(wf["date"]).dt.year)["收益率"].mean()
    sey = yr.std(ddof=1) / np.sqrt(len(yr)) if len(yr) > 2 else np.nan
    return {"笔数": len(v), "平均收益": v.mean(), "中位收益": v.median(),
            "胜率": float((v > 0).mean()),
            "t(按日,朴素)": float(day.mean() / se) if se and se > 1e-12 else np.nan,
            "t(按日,重叠修正)": newey_west_t(day, lag=lag),
            "t(按年)": float(yr.mean() / sey) if sey and sey > 1e-12 else np.nan,
            "年数": len(yr), "逐年为正": int((yr > 0).sum())}


def export_all(tables: Dict[str, pd.DataFrame]) -> bytes:
    """把所有结果表打包成一个 zip（纯标准库，无额外依赖）。"""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            if df is None or not len(df):
                continue
            z.writestr(f"{name}.csv", df.to_csv().encode("utf-8-sig"))
    return buf.getvalue()


# ======================================================================
# 界面
# ======================================================================
def _st(tr: pd.DataFrame) -> dict:
    if tr is None or not len(tr):
        return {}
    d = tr.dropna(subset=["收益率"])
    if len(d) < 30:
        return {}
    day = d.groupby("date")["收益率"].mean().sort_index()
    se = day.std(ddof=1) / np.sqrt(len(day))
    return {"笔数": len(d), "平均收益": d["收益率"].mean(),
            "中位收益": d["收益率"].median(), "胜率": (d["收益率"] > 0).mean(),
            "聚类t": day.mean() / se if se > 1e-12 else np.nan}


def main():
    st.set_page_config(page_title="板块轮动选股", layout="wide")
    ss = st.session_state
    ss.setdefault("panel", None)
    st.title("板块轮动选股")
    st.caption("先选最强板块，再从板块内选动量最高的股票。"
               "科技/军工/新能源/机器人　·　流通市值 50-1000 亿　·　股价 10 元以上")

    with st.sidebar:
        token = st.text_input("Tushare Token", type="password",
                              value=os.environ.get("TUSHARE_TOKEN", ""))
        top_sec = st.slider("选几个板块", 1, 5, 3)
        top_n = st.slider("每次选几只", 1, 5, 3)
        hold = st.slider("持有交易日", 3, 30, 20)
        cap = st.slider("每个板块最多取几只（0=不限）", 0, 5, 0,
                        help="0 是已验证过的口径：候选按板块顺序取，"
                             "常常三只全来自最强板块。设为 1 则强制每个板块只取一只。"
                             "改了这个，前面所有回测结论都要重跑。")
        with st.expander("其他设置"):
            start = st.date_input("数据起始（选「验证」模式时生效）", dt.date(2018, 1, 1))
            end = st.date_input("数据结束", dt.date.today())
            min_mem = st.slider("板块最少成分股", 3, 20, 5)
            every = st.slider("每几个交易日选一次", 1, 10, 3)
            comm = st.number_input("佣金(单边,万分之)", 0.0, 10.0, 3.0, 0.1) / 1e4
            slip = st.number_input("滑点(单边,%)", 0.0, 0.5, 0.10, 0.01) / 100.0
            workers = st.slider("下载并发", 1, 8, 4,
                                help="按股票下载，每只 2 次调用。"
                                     "官方 daily 每分钟 500 次，4 线程足够跑满。")
            rate = st.slider("每分钟调用上限", 100, 800, 450, 50,
                             help="留在官方 500 次/分以下。频繁报错就调低。")
        run = st.button("下载数据", type="primary", use_container_width=True)
        _cn, _cmb, _c0, _c1 = day_cache_info()
        if _cn:
            st.caption(f"行情缓存 {_cn} 只股票 · {_cmb:.0f}MB")
        else:
            st.caption("行情缓存为空")
        if st.button("清除缓存并重新下载", use_container_width=True):
            n = clear_day_cache()
            for kk in ("panel", "sec", "res", "nz", "sigres", "wf", "kres",
                       "ksplit", "kbk", "elig", "elig_key", "kdf", "ddf"):
                ss.pop(kk, None)
            gc.collect()
            st.success(f"已清除 {n} 个缓存文件，请点「下载数据」。")
        if ss.get("panel") is not None:
            _p = ss["panel"]
            st.success(f"{len(_p['codes'])} 只 × {len(_p['cal'])} 日")
            st.caption(f"数据区间 {_p['cal'][0]:%Y-%m-%d} ~ **{_p['cal'][-1]:%Y-%m-%d}**")

    # ---------------- 下载 ----------------
    if run:
        if not token:
            st.error("请先填 Tushare Token"); st.stop()
        try:
            import tushare as ts
        except ImportError:
            st.error("未安装 tushare：pip install tushare"); st.stop()
        ts.set_token(token); pro = ts.pro_api(token)
        lim = Limiter(rate); API_ERRORS.clear()
        for kk in list(ss.keys()):
            if kk != "panel" or True:
                pass
        for kk in ("panel", "sec", "res", "nz", "sigres", "wf", "kres", "ksplit",
                   "elig", "elig_key", "kmask_key", "sec_mm"):
            ss.pop(kk, None)
        gc.collect()
        s_str, e_str = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        with st.status("下载中…", expanded=True) as stt:
            st.write("取申万行业成分股（含二级行业）…")
            uni = fetch_universe(pro, lim, SW_L1_DEFAULT, SW_L2_DEFAULT)
            if not len(uni):
                st.error("行业成分股为空，可能是 Tushare 积分不足。"); st.stop()
            n2 = uni["l2_name"].notna().sum() if "l2_name" in uni.columns else 0
            st.write(f"   {len(uni)} 只，其中 {n2} 只带二级行业")
            if n2 < len(uni) * 0.5:
                st.warning("多数股票没取到二级行业，板块层做不了——可能是积分不足。")
            st.write("取股票基础信息…")
            basic = fetch_stock_basic(pro, lim)
            st.write("市值预筛…")
            codes = prescreen_by_mv(pro, lim, list(uni["ts_code"]), s_str, e_str, 50, 1000)
            # 自动挑更省调用的下载方式：
            #   按股票取 = 2 × 股票数（固定）；按交易日取 = 2 × 交易日数
            ndays = int(np.busday_count(start, end) * 0.97)
            by_date = ndays < len(codes)
            # 缓存可写性自检：写不进去就会每次全量重下，必须提前告知
            try:
                os.makedirs(DAY_DIR, exist_ok=True)
                _t = os.path.join(DAY_DIR, ".wtest")
                with open(_t, "wb") as _f:
                    _f.write(b"1")
                os.remove(_t)
                _cached = len([x for x in os.listdir(DAY_DIR) if x.endswith(".pkl")])
                st.write(f"   磁盘缓存可用，已有 {_cached} 个交易日")
            except Exception as _e:
                st.warning(f"**磁盘缓存不可写（{_e}）**，每次都会全量重下。")
            st.write(f"下载 {len(codes)} 只（约 {ndays} 个交易日）…")
            st.write(f"   用**{'按交易日' if by_date else '按股票'}**取："
                     f"约 {2*ndays if by_date else 2*len(codes)} 次调用"
                     f"（另一种要 {2*len(codes) if by_date else 2*ndays} 次）")
            bar = st.progress(0.0); t0 = time.time()
            px = download_all(token, codes, s_str, e_str, lim, True, workers,
                              lambda a, b, c: bar.progress(a / b,
                                  text=f"{a}/{b}　{(time.time()-t0)/60:.1f} 分"))
            if not px:
                st.error("没下到数据。"); st.stop()
            panel = build_panel(px); px.clear(); del px; gc.collect()
            ss["panel"], ss["basic"], ss["uni"] = panel, basic, uni
            stt.update(label=f"完成，{(time.time()-t0)/60:.1f} 分钟", state="complete")

    if ss.get("panel") is None:
        st.info("左侧填 Token 后点「下载数据」。首次约 10-20 分钟，之后走本地缓存。")
        with st.expander("为什么有时候一打开就要重新下载？"):
            st.markdown(
                "有两层缓存，都可能丢：\n\n"
                "1. **内存里的 panel**：应用进程重启就没了。Streamlit Cloud 会在"
                "长时间没人访问后休眠，再次打开就是新进程。\n"
                "2. **磁盘上的分股缓存**：**每次你覆盖 app.py，云端都会重建容器，"
                "这份缓存一起被清掉**——这大概是你遇到最多的情况。\n\n"
                "我试过把整个数据打包供你下载/回传，但压缩后仍有 **87MB**"
                "（价格是随机游走，压不动），不实用。\n\n"
                "**实际的解法就是用「日常」模式**：近2年数据一两分钟下完，"
                "选股结果和8年完全一致。代码稳定不再频繁改动之后，缓存也就能留住了。")
        st.stop()

    panel, basic, uni = ss["panel"], ss["basic"], ss["uni"]
    dkey = f"{len(panel['codes'])}|{panel['cal'][-1]:%Y%m%d}|{min_mem}"


    # 合格池、板块、信号、日线KD —— 全部按数据版本缓存。
    # 不缓存的话每次交互都要重算 1400×2100 的全量矩阵，反复分配大数组
    # 会把云端内存顶爆、进程被杀、页面闪回初始状态。
    if ss.get("elig_key") != dkey:
        with st.spinner("构建合格池与板块…"):
            ss["elig"] = build_eligibility(panel, basic, uni, 50, 1000, 10.0, 2.0, 365)
            sectors = build_sector_map(uni, panel, ss["elig"], min_mem)
            if not sectors:
                st.error("没能建立板块映射——二级行业数据缺失。"); st.stop()
            R, IDX, cnt = build_sector_index(panel, ss["elig"], sectors)
            amt = pd.DataFrame({s: panel["amount"][c].where(ss["elig"][c]).sum(axis=1)
                                for s, c in sectors.items()})
            kdf, ddf = daily_kd(panel)
            ss["sec"] = (sectors, R, IDX, cnt, sector_factors(R, IDX, amt))
            ss["kdf"], ss["ddf"] = kdf, ddf
            ss["elig_key"] = dkey
            ss.pop("nz", None); ss.pop("sigres", None)
            ss.pop("res", None); ss.pop("wf", None)
            gc.collect()
    elig = ss["elig"]
    sectors, R, IDX, cnt, SF = ss["sec"]
    KDF, DDF = ss["kdf"], ss["ddf"]
    kw = dict(comm=comm, stamp=0.0005, slip=slip)
    dates = list(panel["cal"][130::every])
    st.caption(f"数据截至 **{panel['cal'][-1]:%Y-%m-%d}**　·　"
               f"{len(panel['codes'])} 只 × {len(panel['cal'])} 个交易日　·　"
               f"{len(sectors)} 个板块")
    DEF_SIG = "板块20日动量" if "板块20日动量" in SF else list(SF)[0]

    t1, t2, t3, t4 = st.tabs(["① 板块信号", "② 主回测", "③ 位置诊断", "④ 今日候选"])

    # ---------------- ① 板块信号 ----------------
    with t1:
        st.markdown("### 板块层面的信号有没有预测力")
        st.caption("**换新数据后跑一次就够，平时不用动。** 按信号把板块分四组，"
                   "看未来 3/5/8/15 天板块指数的表现——不涉及选股。")
        with st.expander("地基：板块指数比个股降噪多少（也是一次性的）"):
            if st.button("运行降噪检验"):
                with st.spinner("计算中…"):
                    ss["nz"] = sector_noise_check(panel, elig, sectors)
            if ss.get("nz") is not None:
                nz = ss["nz"]
                rr = float(nz["降噪比"].mean())
                st.metric("平均降噪比", f"{rr:.2f}", f"信噪比提升 {1/rr:.2f} 倍")
                st.dataframe(nz.style.format({"平均成分股数": "{:.0f}",
                                              "个股平均波动": "{:.1%}",
                                              "板块指数波动": "{:.1%}",
                                              "降噪比": "{:.2f}"}),
                             use_container_width=True, height=320)
                (st.success if rr <= 0.85 else st.error)(
                    f"板块指数波动是个股的 {rr:.0%}。" +
                    ("地基成立。" if rr <= 0.85 else "降噪不明显，板块思路不成立。"))
            else:
                st.caption(f"共 {len(sectors)} 个板块，"
                           f"{sum(len(v) for v in sectors.values())} 只股票纳入。"
                           "此前实测降噪比 0.60（波动降到个股的六成）。")

        _nd = len(panel["cal"])
        if _nd < 900:
            st.error(f"**当前只有 {_nd} 个交易日（约 {_nd/244:.1f} 年），这一页的结果不可信。** "
                     "分层检验的 t 值随样本量开平方缩水：8年约420个观测，"
                     f"1.7年只有约{int(_nd/5)}个，t 会缩到 {np.sqrt(_nd/5/420):.2f} 倍。"
                     "**信号之间的名次在这种样本下基本是噪音，不要据此换默认信号。** "
                     "要比较信号，把数据起始改回 2018-01-01。")
        if st.button("跑全部板块信号", type="primary"):
            bar = st.progress(0.0); out = {}
            for i, nm in enumerate(SF):
                t = sector_layer_test(SF[nm], IDX, horizons=(3, 5, 8, 15))
                if len(t):
                    out[nm] = t
                bar.progress((i + 1) / len(SF), text=nm)
            ss["sigres"] = out; bar.empty()
        if ss.get("sigres"):
            out = ss["sigres"]
            sm = pd.DataFrame([{
                "板块信号": nm,
                "Q4−Q1": t["15日超额"].iloc[-1] - t["15日超额"].iloc[0],
                "单调性": float(np.corrcoef(np.arange(len(t)),
                                          t["15日超额"].to_numpy())[0, 1]),
                "Q4超额": t["15日超额"].iloc[-1],
                "末期t(重叠修正)": t["末期t(重叠修正)"].iloc[-1],
                "末期t(朴素)": t["末期t(朴素)"].iloc[-1]} for nm, t in out.items()
            ]).set_index("板块信号").sort_values("Q4−Q1", key=abs, ascending=False)
            st.dataframe(sm.style.format({"Q4−Q1": "{:+.2%}", "单调性": "{:+.2f}",
                                          "Q4超额": "{:+.2%}",
                                          "末期t(重叠修正)": "{:.2f}",
                                          "末期t(朴素)": "{:.2f}"})
                         .background_gradient(subset=["Q4−Q1"], cmap="RdYlGn"),
                         use_container_width=True)
            st.warning("**这一页测的是「板块指数会不会涨」，不是「按它选股能赚多少」。** "
                       "8年数据上：这一页 60日动量最好（t 2.19）> 20日动量（t 1.89）；"
                       "但实际回测里 20日动量 +1.56% > 60日动量 +1.00%，"
                       "滚动前推六年里五年也选中 20日动量。\n\n"
                       "**板块指数涨得准，不等于按它选出的股票赚得多。** "
                       "换默认信号只应依据滚动前推，不要依据这一页。")
            st.info("**看重叠修正后的 t，不看朴素 t**（15日前瞻每几天采样一次，样本重叠）。"
                    "**真正的证据是一致性**：动量类信号如果单调性全部同号，"
                    "而「创20日新高」呈现相反的单调性——这种内部一致的结构"
                    "很难从噪音里产生，比某一个格子的高 t 值可信。")
            pick = st.selectbox("看哪个信号的分组明细", list(out))
            st.dataframe(out[pick].style.format(
                {**{f"{h}日超额": "{:+.2%}" for h in (3, 5, 8, 15)},
                 **{f"{h}日胜率": "{:.1%}" for h in (3, 5, 8, 15)},
                 "末期t(重叠修正)": "{:.2f}", "末期t(朴素)": "{:.2f}"}),
                use_container_width=True)

    # ---------------- ② 主回测 ----------------
    with t2:
        c0, c1 = st.columns(2)
        sig = c0.selectbox("板块信号", list(SF), index=list(SF).index(DEF_SIG))
        if sig not in SF:
            sig = DEF_SIG
        srule = c1.selectbox("板块内怎么选股", STOCK_RULES)
        if srule not in STOCK_RULES:
            srule = STOCK_RULES[0]
        st.caption(f"默认「{DEF_SIG}」不是我挑的——滚动前推六年里五年都选中它。"
                   "**不要再逐个试信号挑最好的**，那会让后面所有检验失效。")
        st.markdown("四个方案同时跑：两层 / 随机板块 / 不分板块 / 全池随机。"
                    "**两层减随机板块 = 板块层的净贡献**，这个对照能干净分离"
                    "「选板块」和「板块内选股」两层的功劳。")

        if st.button("运行对照实验", type="primary", use_container_width=True):
            bar = st.progress(0.0)
            plans = [
                ("两层：最强板块 + " + srule,
                 lambda: sector_then_stock(panel, elig, sectors, SF[sig], dates,
                                           top_sec, top_n, srule, "最强", kdf=KDF,
                                           per_sec_cap=cap)),
                ("对照A：随机板块 + " + srule,
                 lambda: sector_then_stock(panel, elig, sectors, SF[sig], dates,
                                           top_sec, top_n, srule, "随机", kdf=KDF,
                                           per_sec_cap=cap)),
                ("对照B：不分板块，全池 " + srule,
                 lambda: flat_stock_pick(panel, elig, dates, top_n, srule)),
                ("对照C：全池随机",
                 lambda: flat_stock_pick(panel, elig, dates, top_n, "S3_板块内随机")),
            ]
            rows, keep, pks = [], {}, {}
            for i, (lab, fn) in enumerate(plans):
                pk = fn()
                tr = track_fixed(pk, panel, hold, **kw) if len(pk) else pd.DataFrame()
                s_ = _st(tr)
                bar.progress((i + 1) / len(plans), text=lab)
                if s_:
                    rows.append({"方案": lab, **s_}); keep[lab] = tr; pks[lab] = pk
            ss["res"] = (pd.DataFrame(rows).set_index("方案"), keep, pks, sig, srule)
            ss.pop("wf", None); ss.pop("ksplit", None)
            bar.empty(); gc.collect()

        if ss.get("res"):
            df, keep, pks, sig_, sr_ = ss["res"]
            st.dataframe(df.style.format({"笔数": "{:.0f}", "平均收益": "{:+.2%}",
                                          "中位收益": "{:+.2%}", "胜率": "{:.1%}",
                                          "聚类t": "{:.2f}"})
                         .background_gradient(subset=["聚类t"], cmap="RdYlGn",
                                              vmin=-3, vmax=3),
                         use_container_width=True)
            try:
                two, ra, fb = df.iloc[0], df.iloc[1], df.iloc[2]
                st.metric("板块层净贡献（两层 − 随机板块）",
                          f"{two['平均收益']-ra['平均收益']:+.3%} / 笔",
                          f"两层 vs 全池直选 {two['平均收益']-fb['平均收益']:+.3%}")
                rt = (comm * 2 + 0.0005 + slip * 2)
                st.caption(f"单次往返成本 {rt:.2%}；持有 {hold} 日 → 年换手 "
                           f"{244/hold:.0f} 次 → 年成本 {rt*244/hold:.1%}。"
                           f"**每笔平均收益要超过 {rt:.2%} 才算真有边际。**")
            except Exception:
                pass

            st.divider()
            st.markdown("### 三项诊断")
            st.caption("前面几轮就是这三项戳破的幻觉：平均收益漂亮，但利润 94% 来自 "
                       "513 笔里的 5 笔、或者只靠一年撑着。"
                       "**注意：如果你试过多个配置再挑最好的，这里的「样本外」已经不算数**"
                       "——往下看滚动前推。")
            dsel = st.selectbox("诊断哪个方案", list(keep))
            trd = keep[dsel]
            cA, cB = st.columns(2)
            sp = split_check(trd)
            cA.markdown("**样本内 / 样本外**（2023-01-01 分界）")
            if len(sp):
                cA.dataframe(sp.style.format({"笔数": "{:.0f}", "平均收益": "{:+.2%}",
                                              "中位收益": "{:+.2%}", "胜率": "{:.1%}",
                                              "聚类t": "{:.2f}"}),
                             use_container_width=True)
            pc = concentration_check(trd)
            cB.markdown("**利润集中度**")
            if len(pc):
                cB.dataframe(pc.style.format({"占总利润": "{:.1%}",
                                              "剔除后单笔均值": "{:+.3%}",
                                              "剩余比例": "{:.0%}"}),
                             use_container_width=True)
            yy = yearly_check(trd)
            if len(yy):
                st.markdown("**逐年**")
                st.dataframe(yy.style.format({"笔数": "{:.0f}", "平均收益": "{:+.2%}",
                                              "中位收益": "{:+.2%}", "胜率": "{:.1%}",
                                              "聚类t": "{:.2f}"})
                             .background_gradient(subset=["平均收益"], cmap="RdYlGn"),
                             use_container_width=True)
                st.write(f"逐年为正 **{int((yy['平均收益']>0).sum())}/{len(yy)}**")
                st.caption(
                    f"**最后一年（{yy.index[-1]}）的数字会随数据更新而变，其余年份不会。** "
                    f"持有 {hold} 个交易日的规则下，数据末尾不足 {hold} 天的交易"
                    "整笔被丢弃；每多几天数据，就有几笔能完成、被纳入统计。"
                    f"当前数据截至 {panel['cal'][-1]:%Y-%m-%d}。"
                    "拿不同日期跑出的结果对比时，只看最后一年之前的部分。")

            st.divider()
            st.markdown("### 滚动前推（搜索过参数后唯一算数的检验）")
            st.caption(
                f"**滚动前推自己搜索**：板块数(2或3) × 持有期(15或20日) × 全部 {len(SF)} 个板块信号。\n\n"
                f"**按你侧边栏固定**：每次选 **{top_n}** 只 · 每板块最多 "
                f"**{cap if cap else '不限'}** 只 · 每 **{every}** 日选一次 · 成本设置。\n\n"
                "所以改「选几个板块」和「持有交易日」对这里没影响（它自己会搜）；"
                "改「每次选几只」「每板块最多几只」「每几日选一次」会改变结果。")
            st.error("**每年年初只用截至上一年底的数据挑配置，再用它跑这一年。** "
                     "全样本上挑一个最优配置再看它的「样本外」，等于用样本外做了选择，"
                     "那个数字不算数。")
            if st.button("运行滚动前推", type="primary"):
                bar3 = st.progress(0.0)
                picked, wf = walk_forward(
                    panel, elig, sectors, SF, dates,
                    top_secs=(2, 3), top_ns=(top_n,), holds=(15, 20),
                    start_year=2021, per_sec_cap=cap,
                    progress=lambda p, n2: bar3.progress(p, text=n2), **kw)
                ss["wf"] = (picked, wf); bar3.empty(); gc.collect()
            if ss.get("wf"):
                picked, wf = ss["wf"]
                if not len(wf):
                    st.warning("样本不足。")
                else:
                    s5 = wf_summary(wf, hold_days=hold, step_days=every)
                    m5 = st.columns(4)
                    m5[0].metric("平均收益", f"{s5['平均收益']:+.2%}")
                    m5[1].metric("胜率", f"{s5['胜率']:.1%}")
                    m5[2].metric("t(按年，最严格)", f"{s5['t(按年)']:.2f}",
                                 f"逐年为正 {s5['逐年为正']}/{s5['年数']}")
                    m5[3].metric("笔数", f"{s5['笔数']}")
                    st.caption(f"对照：按日朴素 t={s5['t(按日,朴素)']:.2f}，"
                               f"按日重叠修正 t={s5['t(按日,重叠修正)']:.2f}。"
                               "**该看按年**——每年重新挑一次配置，年与年之间才真正独立。")
                    st.dataframe(picked.style.format(
                        {"历史t": "{:.2f}", "当年笔数": "{:.0f}",
                         "当年平均收益": "{:+.2%}", "当年胜率": "{:.1%}"})
                        .background_gradient(subset=["当年平均收益"], cmap="RdYlGn"),
                        use_container_width=True)
                    nuniq = picked["选中配置"].nunique()
                    top1 = picked["选中配置"].value_counts()
                    if nuniq <= max(2, len(picked) // 3):
                        st.success(f"**配置稳定**：{len(picked)} 年里只用了 {nuniq} 种，"
                                   f"最常选中「{top1.index[0]}」{int(top1.iloc[0])} 次。"
                                   "这很难从噪音里得到——我验证时，纯噪音数据上配置年年换，"
                                   "植入真规律时六年只选中同一个。")
                    else:
                        st.error(f"**配置不稳定**：{len(picked)} 年里换了 {nuniq} 种。"
                                 "每年的「最优」都不一样，说明所谓最优只是当年的运气，"
                                 "当年你没有依据挑中它。**这比平均收益低更值得警惕。**")
                    yv = wf.groupby(pd.to_datetime(wf["date"]).dt.year)["收益率"].mean()
                    top2 = yv.nlargest(2).index
                    rest = wf[~pd.to_datetime(wf["date"]).dt.year.isin(top2)]
                    if len(rest):
                        st.warning(f"剔除最好的两年（{list(top2)}）后，其余年份平均 "
                                   f"{rest['收益率'].mean():+.3%}/笔。")
                    if s5["t(按年)"] >= 2 and s5["平均收益"] > 0:
                        st.success("**滚动前推也站得住。** 这是搜索过参数之后唯一算数的证据。")
                    elif s5["平均收益"] > 0 and s5["逐年为正"] >= s5["年数"] * 0.6:
                        need = int(np.ceil((2 * yv.std(ddof=1) / yv.mean()) ** 2)) \
                            if yv.mean() > 0 else 0
                        st.warning(
                            f"**为正但样本不足**（{s5['平均收益']:+.2%}，按年 "
                            f"t={s5['t(按年)']:.2f}）。**这不等于过拟合**——"
                            "过拟合是样本内好、样本外垮，而这里样本外为正、配置也稳定。"
                            f"问题是年度样本太少：以这个边际和年间波动，约需 **{need} 年** "
                            f"才能到 t=2，现在只有 {s5['年数']} 年。\n\n"
                            "**结论是「证据不足」，不是「已被证伪」。** "
                            "唯一能改变它的是新数据——往后每周记录名单，攒够年份再看。")
                    else:
                        st.error(f"**滚动前推没站住**（{s5['平均收益']:+.2%}，按年 "
                                 f"t={s5['t(按年)']:.2f}）。当年你没有能力挑中"
                                 "事后看最好的那个配置。")

            st.divider()
            d = st.selectbox("看哪个方案的成交明细", list(keep))
            _td = keep[d].copy()
            _order = [c for c in ["date", "code", "板块", "买入日", "卖出日",
                                  "买入价(实际)", "卖出价(实际)", "收益率",
                                  "持有交易日", "买入K", "买入价(复权)", "卖出价(复权)"]
                      if c in _td.columns]
            if "卖出日" in _td.columns and len(_td):
                st.caption(f"最晚卖出日 **{pd.to_datetime(_td['卖出日']).max():%Y-%m-%d}**"
                           f"　·　数据截至 {panel['cal'][-1]:%Y-%m-%d}"
                           f"　·　共 {len(_td)} 笔")
            st.dataframe(_td[_order].tail(300), use_container_width=True, height=300)
            st.caption("**买入价(实际)** 是当日真实开盘价，可直接与交易软件核对。"
                       "**买入价(复权)** 是起点归一化为 1.0 的前复权序列——"
                       "收益率必须用它算才正确（还原了分红送股），但它不是盘面价格。")
            if st.button("生成导出包"):
                with st.spinner("打包中…"):
                    tb = {"01_对照结果": df,
                          "02_成交明细": pd.concat([v.assign(方案=k)
                                                 for k, v in keep.items()],
                                                ignore_index=True),
                          "00_参数": pd.DataFrame([{
                              "板块信号": sig_, "选股规则": sr_, "选几个板块": top_sec,
                              "每次选几只": top_n, "持有交易日": hold,
                              "每几日选一次": every, "板块数": len(sectors),
                              "导出时间": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}]
                          ).T.rename(columns={0: "值"})}
                    for k2, v2 in keep.items():
                        tag = k2.split("：")[0]
                        for nm2, fn2 in (("样本内外", split_check), ("逐年", yearly_check),
                                         ("集中度", concentration_check)):
                            r2 = fn2(v2)
                            if len(r2):
                                tb[f"03_{nm2}_{tag}"] = r2
                    if ss.get("sigres"):
                        tb["04_板块信号分层"] = pd.concat(
                            [t.assign(信号=k) for k, t in ss["sigres"].items()])
                    if ss.get("wf"):
                        tb["05_滚动前推_逐年"] = ss["wf"][0]
                    if ss.get("ksplit") is not None and len(ss["ksplit"]):
                        tb["06_位置诊断_干净划分"] = ss["ksplit"]
                    if ss.get("kbk") is not None and len(ss["kbk"]):
                        tb["07_位置诊断_K分档"] = ss["kbk"]
                    if ss.get("nz") is not None:
                        tb["08_降噪检验"] = ss["nz"]
                    ss["zipb"] = export_all(tb)
                    ss["zipn"] = f"sector_{dt.datetime.now():%Y%m%d_%H%M}.zip"
                    gc.collect()
            if ss.get("zipb"):
                st.download_button(f"下载 {ss['zipn']}（{len(ss['zipb'])/1024:.0f} KB）",
                                   ss["zipb"], ss["zipn"], "application/zip",
                                   type="primary", use_container_width=True)

    # ---------------- ③ 位置诊断 ----------------
    with t3:
        st.markdown("### 想加任何买入条件，先在这里验")
        st.error("**不要用「带替补的过滤」去验证条件。** 实测：8 个买入位置过滤里"
                 "不过滤最好（+2.34%），两个相反方向的过滤都变差 0.6-0.7%；"
                 "而干净划分显示 **87% 的表观效果来自「替补往排名深处挖」**，"
                 "不是条件本身。排名是有信息的——第1名比第4、5名值钱。")
        st.markdown("**正确做法**：把同一批成交按条件劈成两半，不做替补。"
                    "两组笔数加起来等于总数，就没有污染。")
        if not ss.get("res"):
            st.info("先到「② 主回测」跑一次对照实验。")
        else:
            df, keep, pks, sig_, sr_ = ss["res"]
            base = list(keep)[0]
            pk0, tr0 = pks[base], keep[base]
            cond = st.selectbox("按什么条件划分",
                                ["板块已霸榜几天（执行时机）",
                                 "买入日在高位死叉后1-5天", "买入时 K≥75", "买入时 K≥60"])
            if st.button("运行干净划分", type="primary"):
                with st.spinner("计算中…"):
                    if cond.startswith("板块已霸榜"):
                        ss["ksplit"] = run_age_diagnosis(pk0, tr0, SF[sig_],
                                                         panel["cal"])
                        ss["kbk"] = None
                    elif cond == "买入日在高位死叉后1-5天":
                        mk = high_dead_cross_mask(KDF, DDF, 5); ly, ln = "死叉后1-5天", "其他"
                    elif cond == "买入时 K≥75":
                        mk = KDF >= 75; ly, ln = "K≥75", "K<75"
                    else:
                        mk = KDF >= 60; ly, ln = "K≥60", "K<60"
                        ss["ksplit"] = split_by_mask(pk0, tr0, mk, ly, ln)
                        ss["kbk"] = k_bucket_diagnosis(pk0, tr0)
            if ss.get("ksplit") is not None and len(ss["ksplit"]):
                st.dataframe(ss["ksplit"].style.format(
                    {"笔数": "{:.0f}", "占比": "{:.1%}", "平均收益": "{:+.2%}",
                     "中位收益": "{:+.2%}", "胜率": "{:.1%}", "聚类t": "{:.2f}"})
                    .background_gradient(subset=["中位收益"], cmap="RdYlGn"),
                    use_container_width=True)
                n_all = len(tr0.dropna(subset=["收益率"]))
                st.caption(f"各组笔数合计 {int(ss['ksplit']['笔数'].sum())}，"
                           f"总成交 {n_all} —— 相等说明是真划分，不是替补。")
                if cond.startswith("板块已霸榜"):
                    st.info("**这回答的是执行时机**：龙头刚换就买，和它已经领跑几天才买，"
                            "差别多大。\n\n各档如果差不多，说明**错过前几天不用懊恼**，"
                            "什么时候有钱什么时候买都行；如果「第1天」明显更好，"
                            "那就值得盯紧龙头切换。\n\n"
                            "判读时注意笔数：占比小的档标准误大，别被单个数字带走。")
                st.info("**判读**：胜率差的标准误约 2-3 个百分点，"
                        "差异小于这个量级就是噪音。\n\n"
                        "**你只拿 1-3 只，抓到右尾的概率低，实际体验更接近中位数**——"
                        "所以中位数和胜率对你比平均收益更有参考价值。")
            if ss.get("kbk") is not None and len(ss["kbk"]):
                st.markdown("**买入当天日线 SKDJ 的 K 值分档**")
                st.dataframe(ss["kbk"].style.format(
                    {"笔数": "{:.0f}", "平均收益": "{:+.2%}", "中位收益": "{:+.2%}",
                     "胜率": "{:.1%}", "占比": "{:.1%}"})
                    .background_gradient(subset=["平均收益"], cmap="RdYlGn"),
                    use_container_width=True)
                st.caption("此前实测：六档全部为正，K>75 那两档胜率最高。"
                           "板块内最强 = 涨得最多 = K 高，**强势股待在高位是特征不是缺陷**。")

    # ---------------- ④ 今日候选 ----------------
    with t4:
        st.success("**日常只需要这一页。** 左侧点「增量更新到最新」（几秒），"
                   "然后看下面的名单。前三页都是一次性验证，平时不用点。")
        sig2 = st.selectbox("板块信号", list(SF), key="s2", index=list(SF).index(DEF_SIG))
        if sig2 not in SF:
            sig2 = DEF_SIG
        sr2 = st.selectbox("板块内选股", STOCK_RULES, key="r2")
        if sr2 not in STOCK_RULES:
            sr2 = STOCK_RULES[0]
        d = panel["cal"][-1]
        _today = pd.Timestamp(dt.date.today())
        # 判断依据是"数据全不全"，不是"是不是今天"——收盘后当天数据就是完整可用的。
        _cov_px = float(panel["raw_close"].loc[d].notna().mean())
        _cov_mv = float(panel["circ_mv"].loc[d].notna().mean())
        _ref = float(panel["raw_close"].iloc[-6:-1].notna().mean().mean())
        _nel = int(elig.loc[d].sum())
        _nel_ref = float(elig.iloc[-6:-1].sum(axis=1).mean())
        _lag = int(np.busday_count(d.date(), _today.date()))
        if _lag >= 1:
            st.info(f"**选股依据：{d:%Y-%m-%d} 收盘数据**（今天是 {_today:%Y-%m-%d}）。"
                    + ("　今天的数据还没齐（通常是市值未发布），"
                       "点左侧「增量更新」可以试着补上。" if _lag == 1 else ""))
        if _nel < _nel_ref * 0.5:
            st.error(f"**今日合格股票只有 {_nel} 只，而前几日平均 {_nel_ref:.0f} 只。** "
                     "名单可能不可靠，建议稍后重新增量更新再看。")
        else:
            st.success(f"**选股依据：{d:%Y-%m-%d} 收盘数据**（今天是 {_today:%Y-%m-%d}），"
                       f"合格股票 {_nel} 只。"
                       "按回测口径，这份名单应在**下一个交易日开盘**买入。")
        f = SF[sig2].loc[d].dropna().sort_values(ascending=False)
        st.subheader(f"{d:%Y-%m-%d}　板块排名")
        st.dataframe(pd.DataFrame({"板块": f.index, "信号值": f.values,
                                   "合格成分股": [int(elig.loc[d, sectors[s]].sum())
                                                for s in f.index]}).head(10),
                     use_container_width=True, hide_index=True)
        pk = sector_then_stock(panel, elig, sectors, SF[sig2], [d],
                               top_sec, top_n, sr2, "最强", kdf=KDF, per_sec_cap=cap)
        if len(pk) < top_n:
            info = [f"{s3}: {int(elig.loc[d, sectors[s3]].sum())} 只合格"
                    for s3 in list(f.index[:top_sec])]
            st.warning(f"**只选出 {len(pk)} 只，少于设定的 {top_n} 只。** "
                       f"各板块合格数：{'；'.join(info)}。\n\n"
                       "候选按板块顺序取：先取最强板块里动量最高的，不够再取次强板块。"
                       "回测里有冷却期（同股 5 日内不重复）会自然分散，"
                       "单看某一天没有冷却历史，就会集中在最强板块。")
        if not len(pk):
            st.warning("今日无候选。")
        else:
            nm = basic.set_index("ts_code")["name"].to_dict()
            out = pd.DataFrame([{
                "序": int(r["rank"]), "代码": r["code"], "名称": nm.get(r["code"], ""),
                "板块": r["板块"],
                "收盘价": round(float(panel["raw_close"].loc[d, r["code"]]), 2),
                "流通市值(亿)": round(float(panel["circ_mv"].loc[d, r["code"]]) / 1e4),
                "20日涨幅": f"{r['score']:.1%}",
                "日线K": round(float(r["买入K"]), 1) if pd.notna(r.get("买入K")) else None
            } for _, r in pk.iterrows()])
            vc = out["板块"].value_counts()
            if len(vc) == 1 and len(out) > 1:
                st.warning(f"**{len(out)} 只全部来自「{vc.index[0]}」。** "
                           "候选按板块顺序取：最强板块的股票排在最前，"
                           "不够才轮到次强板块。回测里有 5 日冷却期会自然分散，"
                           "单看某一天没有冷却历史，就集中在最强板块。\n\n"
                           "**这是已验证口径的正常表现，但意味着没有分散。** "
                           "想强制分散，把侧边栏「每个板块最多取几只」设为 1——"
                           "**但那是没验证过的新口径，改了要重跑第②页的对照和滚动前推。**")
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("下载 CSV", out.to_csv(index=False).encode("utf-8-sig"),
                               f"picks_{d:%Y%m%d}.csv", "text/csv")
            st.info(f"**执行规则**：{d:%Y-%m-%d} 之后的下一个交易日开盘买入，"
                    f"**持有 {hold} 个交易日后开盘卖出**。"
                    "不设止盈止损——回测就是这个口径。"
                    "「日线K」仅供参考，实测按它过滤只会让结果变差。")

    if API_ERRORS:
        with st.expander(f"接口异常 {len(API_ERRORS)} 条"):
            st.write(API_ERRORS[-30:])


if __name__ == "__main__" and st is not None:
    main()
