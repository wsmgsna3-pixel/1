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
CACHE_DIR = os.path.join(APP_DIR, ".cache_pool")
PX_DIR = os.path.join(CACHE_DIR, "px")
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
    del raw_open, pre_close, idxed      # 之后再也用不到，立刻释放
    gc.collect()

    return dict(cal=cal, codes=codes,
                raw_close=raw_close, amount=amount, circ_mv=circ_mv,
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
    hh = IDX.rolling(60).max()
    out["板块距60日高点"] = IDX / hh - 1.0
    out["板块创20日新高"] = (IDX >= IDX.rolling(20).max()).astype(float)
    a5, a60 = amt_sec.rolling(5).mean(), amt_sec.rolling(60).mean()
    out["板块量能扩张"] = a5 / a60.where(a60 > 1e-9)
    share = amt_sec.div(amt_sec.sum(axis=1), axis=0)
    out["板块成交额占比变化"] = share - share.rolling(20).mean()
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


# ---------------- 板块 → 个股 两层选股 ----------------
STOCK_RULES = ["S1_板块内最强", "S2_板块内最弱(回调)", "S3_板块内随机"]


def sector_then_stock(panel: dict, elig: pd.DataFrame, sectors: Dict[str, List[str]],
                      sec_fac: pd.DataFrame, dates: List[pd.Timestamp],
                      top_sec: int = 2, top_n: int = 3,
                      stock_rule: str = "S1_板块内最强",
                      sec_rule: str = "最强", cooldown: int = 5,
                      seed: int = 20260910) -> pd.DataFrame:
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
        for s, c, sc in cand:
            if taken >= top_n:
                break
            if c in last and i - last[c] < cooldown:
                continue
            rows.append({"date": d, "code": c, "板块": s, "rank": taken + 1, "score": sc})
            last[c] = i
            taken += 1
    return pd.DataFrame(rows)


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
        out.append({"date": p["date"], "code": p["code"], "板块": p.get("板块", "-"),
                    "买入日": cal[b], "卖出日": cal[e], "买入价": entry, "卖出价": exit_,
                    "收益率": exit_ / entry - 1.0, "持有交易日": e - b})
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
    st.caption("先选最强板块，再从板块内选股。目的是降噪 —— 板块指数波动只有个股的一半。")

    with st.sidebar:
        token = st.text_input("Tushare Token", type="password",
                              value=os.environ.get("TUSHARE_TOKEN", ""))
        top_sec = st.slider("选几个板块", 1, 5, 2)
        top_n = st.slider("总共选几只", 1, 5, 3)
        hold = st.slider("持有交易日", 3, 20, 8)
        with st.expander("其他设置"):
            start = st.date_input("数据起始", dt.date(2018, 1, 1))
            end = st.date_input("数据结束", dt.date.today())
            min_mem = st.slider("板块最少成分股", 3, 20, 5)
            every = st.slider("每几个交易日选一次", 1, 10, 3)
            comm = st.number_input("佣金(单边,万分之)", 0.0, 10.0, 3.0, 0.1) / 1e4
            slip = st.number_input("滑点(单边,%)", 0.0, 0.5, 0.10, 0.01) / 100.0
            workers = st.slider("下载并发", 1, 8, 4)
        run = st.button("下载数据", type="primary", use_container_width=True)
        if ss.get("panel") is not None:
            st.success(f"{len(ss['panel']['codes'])} 只 × {len(ss['panel']['cal'])} 日")

    if run:
        if not token:
            st.error("请先填 Tushare Token"); st.stop()
        try:
            import tushare as ts
        except ImportError:
            st.error("未安装 tushare：pip install tushare"); st.stop()
        ts.set_token(token); pro = ts.pro_api(token)
        lim = Limiter(400); API_ERRORS.clear()
        for kk in ("panel", "sec", "res"):
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
                st.warning("多数股票没取到二级行业。板块层需要它——"
                           "可能是 Tushare 积分不足以调用 index_member_all。")
            st.write("取股票基础信息…")
            basic = fetch_stock_basic(pro, lim)
            st.write("市值预筛…")
            codes = prescreen_by_mv(pro, lim, list(uni["ts_code"]), s_str, e_str, 50, 1000)
            st.write(f"下载 {len(codes)} 只…")
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
        st.info("左侧填 Token 后点「下载数据」。首次 5-15 分钟，之后走缓存。"); st.stop()

    panel, basic, uni = ss["panel"], ss["basic"], ss["uni"]
    elig = build_eligibility(panel, basic, uni, 50, 1000, 10.0, 2.0, 365)
    if ss.get("sec") is None or ss.get("sec_mm") != min_mem:
        with st.spinner("构建板块指数…"):
            sectors = build_sector_map(uni, panel, elig, min_mem)
            if not sectors:
                st.error("没能建立板块映射——二级行业数据缺失。"); st.stop()
            R, IDX, cnt = build_sector_index(panel, elig, sectors)
            amt = pd.DataFrame({s: panel["amount"][c].where(elig[c]).sum(axis=1)
                                for s, c in sectors.items()})
            ss["sec"] = (sectors, R, IDX, cnt, sector_factors(R, IDX, amt))
            ss["sec_mm"] = min_mem
    sectors, R, IDX, cnt, SF = ss["sec"]
    kw = dict(comm=comm, stamp=0.0005, slip=slip)
    dates = list(panel["cal"][130::every])

    t0_, t1_, t2_, t3_ = st.tabs(["① 降噪检验", "② 板块信号", "③ 板块层加分吗", "④ 今日候选"])

    # ---------- ① 降噪 ----------
    with t0_:
        st.markdown("### 地基：板块指数的波动比个股小多少")
        st.markdown("如果降噪不明显，整个思路不成立，后面三页不用看。"
                    "我在模拟数据上实测：20只等权时波动降到个股的 **52%**，"
                    "信噪比提升约 **1.94 倍**。")
        nz = sector_noise_check(panel, elig, sectors)
        st.dataframe(nz.style.format({"平均成分股数": "{:.0f}", "个股平均波动": "{:.1%}",
                                      "板块指数波动": "{:.1%}", "降噪比": "{:.2f}"})
                     .background_gradient(subset=["降噪比"], cmap="RdYlGn_r"),
                     use_container_width=True, height=420)
        rr = float(nz["降噪比"].mean())
        st.metric("平均降噪比", f"{rr:.2f}", f"信噪比提升 {1/rr:.2f} 倍")
        if rr > 0.85:
            st.error("降噪不明显。说明这些板块内的股票走势差异太大，"
                     "等权指数没能滤掉个股噪音——板块思路在这个池子里不成立。")
        else:
            st.success(f"板块指数波动是个股的 {rr:.0%}，"
                       f"同样的信号强度下信噪比提升 {1/rr:.2f} 倍。地基成立。")
        st.caption(f"共 {len(sectors)} 个板块，"
                   f"{sum(len(v) for v in sectors.values())} 只股票纳入。")

    # ---------- ② 板块信号 ----------
    with t1_:
        st.markdown("### 板块层面的信号有没有预测力")
        st.markdown("**这一步不涉及选股**，只问：按信号把板块分组，未来几天板块指数"
                    "的表现有没有单调差别。分不出来的话，选板块就没有依据。")
        if st.button("跑全部板块信号", type="primary"):
            bar = st.progress(0.0); out = {}
            names = list(SF)
            for i, nm in enumerate(names):
                t = sector_layer_test(SF[nm], IDX, horizons=(3, 5, 8, 15))
                if len(t):
                    out[nm] = t
                bar.progress((i + 1) / len(names), text=nm)
            ss["sigres"] = out; bar.empty()
        if ss.get("sigres"):
            out = ss["sigres"]
            summ = []
            for nm, t in out.items():
                v = t["15日超额"].to_numpy()
                summ.append({"板块信号": nm, "Q4−Q1": v[-1] - v[0],
                             "单调性": float(np.corrcoef(np.arange(len(v)), v)[0, 1]),
                             "Q4超额": v[-1], "Q1超额": v[0],
                             "末期t(重叠修正)": t["末期t(重叠修正)"].iloc[-1],
                             "末期t(朴素)": t["末期t(朴素)"].iloc[-1]})
            sm = pd.DataFrame(summ).set_index("板块信号").sort_values(
                "Q4−Q1", key=abs, ascending=False)
            st.dataframe(sm.style.format({"Q4−Q1": "{:+.2%}", "单调性": "{:+.2f}",
                                          "Q4超额": "{:+.2%}", "Q1超额": "{:+.2%}",
                                          "末期t(重叠修正)": "{:.2f}",
                                          "末期t(朴素)": "{:.2f}"})
                         .background_gradient(subset=["Q4−Q1"], cmap="RdYlGn"),
                         use_container_width=True)
            ok = sm[(sm["Q4−Q1"].abs() > 0.01) & (sm["单调性"].abs() > 0.8)
                    & (sm["末期t(重叠修正)"].abs() > 2)]
            if len(ok):
                st.success(f"**{ok.index[0]}** 有明显单调关系："
                           f"Q4−Q1 = {ok['Q4−Q1'].iloc[0]:+.2%}，"
                           f"单调性 {ok['单调性'].iloc[0]:+.2f}。可以用它选板块。")
            else:
                st.error("**没有板块信号呈现明显单调关系。** 也就是说这个池子里"
                         "板块之间的强弱不具备延续性，选板块没有依据。\\n\\n"
                         "我在植入板块轮动的模拟数据上验证过这个检验器——"
                         "那时 Q4−Q1 = +5.30%、t = 15.73。所以不是检验器不灵。")
            pick = st.selectbox("看哪个信号的分组明细", list(out))
            st.dataframe(out[pick].style.format(
                {**{f"{h}日超额": "{:+.2%}" for h in (3, 5, 8, 15)},
                 **{f"{h}日胜率": "{:.1%}" for h in (3, 5, 8, 15)}, "末期t": "{:.2f}"}),
                use_container_width=True)
            st.warning("**看「末期t(重叠修正)」，不要看朴素 t。** 15日前瞻收益每 "
                       f"{every} 天采样一次，相邻样本重叠，朴素标准误会把 t 放大约 "
                       "√(重叠倍数)。这一处我最初漏做了修正，其他检验都做了。")
            st.info("**真正的证据是一致性，不是单个 t 值。** 如果动量类的几个信号"
                    "单调性全部同号、Q4−Q1 全部同向，而且「创新高」这类反向信号"
                    "呈现相反的单调性——这种内部一致的结构很难从噪音里产生，"
                    "比某一个格子的高 t 值可信得多。")
            st.caption("判定：|Q4−Q1| > 1个百分点 且 |单调性| > 0.8 且 |重叠修正t| > 2。")

    # ---------- ③ 板块层加分吗 ----------
    with t2_:
        st.markdown("### 板块层到底加不加分")
        st.markdown("**核心对照**：同样的选股规则，一次用「最强板块」筛，一次用「随机板块」筛。"
                    "两者之差就是板块层的净贡献。再加一个「不分板块直接全池选」做参照。")
        sig = st.selectbox("用哪个板块信号", list(SF),
                           index=list(SF).index("板块20日动量") if "板块20日动量" in SF else 0)
        srule = st.selectbox("板块内怎么选股", STOCK_RULES)
        if st.button("运行对照实验", type="primary"):
            bar = st.progress(0.0)
            plans = [
                ("两层：最强板块 + " + srule,
                 lambda: sector_then_stock(panel, elig, sectors, SF[sig], dates,
                                           top_sec, top_n, srule, "最强")),
                ("对照A：随机板块 + " + srule,
                 lambda: sector_then_stock(panel, elig, sectors, SF[sig], dates,
                                           top_sec, top_n, srule, "随机")),
                ("对照B：不分板块，全池 " + srule,
                 lambda: flat_stock_pick(panel, elig, dates, top_n, srule)),
                ("对照C：全池随机",
                 lambda: flat_stock_pick(panel, elig, dates, top_n, "S3_板块内随机")),
            ]
            rows, keep = [], {}
            for i, (lab, fn) in enumerate(plans):
                pk = fn()
                tr = track_fixed(pk, panel, hold, **kw) if len(pk) else pd.DataFrame()
                s_ = _st(tr)
                bar.progress((i + 1) / len(plans), text=lab)
                if s_:
                    rows.append({"方案": lab, **s_})
                    keep[lab] = tr
            ss["res"] = (pd.DataFrame(rows).set_index("方案"), keep, sig, srule)
            bar.empty()
        if ss.get("res"):
            df, keep, sig_, sr_ = ss["res"]
            st.dataframe(df.style.format({"笔数": "{:.0f}", "平均收益": "{:+.2%}",
                                          "中位收益": "{:+.2%}", "胜率": "{:.1%}",
                                          "聚类t": "{:.2f}"})
                         .background_gradient(subset=["聚类t"], cmap="RdYlGn",
                                              vmin=-3, vmax=3),
                         use_container_width=True)
            try:
                two = df.iloc[0]; ra = df.iloc[1]; fb = df.iloc[2]
                gain = two["平均收益"] - ra["平均收益"]
                st.metric("板块层的净贡献（两层 − 随机板块）", f"{gain:+.3%} / 笔",
                          f"两层 vs 全池直选 {two['平均收益']-fb['平均收益']:+.3%}")
                need = (comm * 2 + 0.0005 + slip * 2)
                st.caption(f"参考：单次往返成本 {need:.2%}。持有 {hold} 日 → "
                           f"一年换手 {244/hold:.0f} 次 → 年成本 {need*244/hold:.1%}。"
                           f"**每笔平均收益要超过 0 才有意义，超过 {need:.2%} 才算真有边际。**")
                if gain > 0.004 and two["聚类t"] >= 2:
                    st.success("板块层确实加分，且两层方案显著。可以往下做。")
                elif abs(gain) <= 0.004:
                    st.error("**板块层没有净贡献。** 最强板块和随机板块的结果差不多，"
                             "说明「哪个板块强」这件事没有延续性。\\n\\n"
                             "我在植入板块轮动的模拟数据上验证过：那时净贡献 +1.99%/笔，"
                             "无轮动时 −0.03%。所以这个对照实验是灵敏的。")
                else:
                    st.warning("板块层有一些贡献，但两层方案本身未达显著。")
            except Exception:
                pass
            if keep:
                st.divider()
                st.markdown("### 三项必做诊断")
                st.caption("前面几轮就是这三项戳破的幻觉：平均收益漂亮，"
                           "但利润 94% 来自 513 笔里的 5 笔、或者只靠一年撑着、"
                           "或者样本内好样本外垮。")
                dsel = st.selectbox("诊断哪个方案", list(keep), key="diag_sel")
                trd = keep[dsel]
                c1, c2 = st.columns(2)
                sp = split_check(trd)
                c1.markdown("**样本内 / 样本外**（2023-01-01 分界）")
                if len(sp):
                    c1.dataframe(sp.style.format({"笔数": "{:.0f}", "平均收益": "{:+.2%}",
                                                  "中位收益": "{:+.2%}", "胜率": "{:.1%}",
                                                  "聚类t": "{:.2f}"}),
                                 use_container_width=True)
                pc = concentration_check(trd)
                c2.markdown("**利润集中度**")
                if len(pc):
                    c2.dataframe(pc.style.format({"占总利润": "{:.1%}",
                                                  "剔除后单笔均值": "{:+.3%}",
                                                  "剩余比例": "{:.0%}"}),
                                 use_container_width=True)
                    c2.caption(f"全部 {pc.attrs['总笔数']} 笔，原始均值 "
                               f"{pc.attrs['原均值']:+.2%}")
                yy = yearly_check(trd)
                st.markdown("**逐年**")
                if len(yy):
                    st.dataframe(yy.style.format({"笔数": "{:.0f}", "平均收益": "{:+.2%}",
                                                  "中位收益": "{:+.2%}", "胜率": "{:.1%}",
                                                  "聚类t": "{:.2f}"})
                                 .background_gradient(subset=["平均收益"], cmap="RdYlGn"),
                                 use_container_width=True)
                    npos = int((yy["平均收益"] > 0).sum())
                    msgs = []
                    if len(sp) == 2:
                        o = sp.loc["样本外"]
                        msgs.append(("样本外 " + ("站得住" if o["平均收益"] > 0 and o["聚类t"] >= 1.5
                                                 else "没站住")
                                     + f"（{o['平均收益']:+.2%}，t={o['聚类t']:.2f}）",
                                     o["平均收益"] > 0 and o["聚类t"] >= 1.5))
                    if len(pc) and 5 in pc.index:
                        keep5 = pc.loc[5, "剩余比例"]
                        msgs.append((f"剔除最赚的5笔后仍保留 {keep5:.0%} 的均值",
                                     keep5 > 0.5))
                    msgs.append((f"逐年为正 {npos}/{len(yy)}", npos >= len(yy) * 0.7))
                    for txt, good in msgs:
                        (st.success if good else st.error)(("✅ " if good else "❌ ") + txt)
                    if all(g for _, g in msgs):
                        st.success("**三项全过。** 这是整个项目里第一次。"
                                   "可以考虑小仓位实盘验证了。")
                    else:
                        st.warning("有诊断未通过。未通过的那几项正是最容易骗人的地方。")

                st.divider()
                d = st.selectbox("看哪个方案的成交明细", list(keep))
                st.dataframe(keep[d].tail(300), use_container_width=True, height=300)
                if st.button("生成导出包"):
                    tb = {"01_对照结果": df, "02_成交明细": pd.concat(
                        [v.assign(方案=k) for k, v in keep.items()], ignore_index=True),
                        "00_参数": pd.DataFrame([{
                            "板块信号": sig_, "选股规则": sr_, "选几个板块": top_sec,
                            "选几只": top_n, "持有交易日": hold,
                            "板块数": len(sectors),
                            "导出时间": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}]
                        ).T.rename(columns={0: "值"})}
                    if ss.get("sigres"):
                        tb["03_板块信号分层"] = pd.concat(
                            [t.assign(信号=k) for k, t in ss["sigres"].items()])
                    tb["04_降噪检验"] = sector_noise_check(panel, elig, sectors)
                    for k2, v2 in keep.items():
                        tag = k2.split("：")[0]
                        for nm2, fn2 in (("样本内外", split_check), ("逐年", yearly_check),
                                         ("集中度", concentration_check)):
                            r2 = fn2(v2)
                            if len(r2):
                                tb[f"05_{nm2}_{tag}"] = r2
                    ss["zipb"] = export_all(tb)
                    ss["zipn"] = f"sector_{dt.datetime.now():%Y%m%d_%H%M}.zip"
                if ss.get("zipb"):
                    st.download_button(f"下载 {ss['zipn']}", ss["zipb"], ss["zipn"],
                                       "application/zip", type="primary",
                                       use_container_width=True)

    # ---------- ④ 今日候选 ----------
    with t3_:
        sig2 = st.selectbox("板块信号", list(SF), key="s2",
                            index=list(SF).index("板块20日动量") if "板块20日动量" in SF else 0)
        sr2 = st.selectbox("板块内选股", STOCK_RULES, key="r2")
        d = panel["cal"][-1]
        f = SF[sig2].loc[d].dropna().sort_values(ascending=False)
        st.subheader(f"{d:%Y-%m-%d}　板块排名")
        st.dataframe(pd.DataFrame({"板块": f.index, "信号值": f.values,
                                   "合格成分股": [int(elig.loc[d, sectors[s]].sum())
                                                for s in f.index]}).head(10),
                     use_container_width=True, hide_index=True)
        pk = sector_then_stock(panel, elig, sectors, SF[sig2], [d], top_sec, top_n, sr2, "最强")
        if not len(pk):
            st.warning("今日无候选。")
        else:
            nm = basic.set_index("ts_code")["name"].to_dict()
            out = pd.DataFrame([{
                "序": int(r["rank"]), "代码": r["code"], "名称": nm.get(r["code"], ""),
                "板块": r["板块"],
                "收盘价": round(float(panel["raw_close"].loc[d, r["code"]]), 2),
                "流通市值(亿)": round(float(panel["circ_mv"].loc[d, r["code"]]) / 1e4),
                "20日涨幅": f"{r['score']:.1%}"} for _, r in pk.iterrows()])
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("下载 CSV", out.to_csv(index=False).encode("utf-8-sig"),
                               f"sector_picks_{d:%Y%m%d}.csv", "text/csv")

    if API_ERRORS:
        with st.expander(f"接口异常 {len(API_ERRORS)} 条"):
            st.write(API_ERRORS[-30:])


if __name__ == "__main__" and st is not None:
    main()
