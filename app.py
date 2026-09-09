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
FACTOR_DEF = {
    "mom_ra":  ("风险调整动量(60日,跳过最近5日)", 1.0),
    "trend_q": ("趋势质量(斜率×R²)",             1.0),
    "rel_str": ("板块相对强度(60日超额)",         0.5),
    "vol_exp": ("量能扩张(5日额/60日额)",         0.5),
    "dist_hi": ("距60日高点(越近越高)",           0.5),
    "rev5":    ("5日反转(近5日涨幅)",            -0.5),
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
ALL_DEF = {**FACTOR_DEF, **DIAG_DEF}
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


PX_COLS = ["trade_date", "open", "close", "pre_close", "pct_chg", "amount", "circ_mv"]


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
    adj_open = (adj_close.shift(1) * (raw_open / pre_close).where(pre_close > 0)).astype(np.float32)

    # 涨跌停判定（创业板/科创板 20%，其余 10%）
    lim_pct = pd.Series([0.20 if (c.startswith("30") or c.startswith("688")) else 0.10
                         for c in codes], index=codes, dtype=np.float32)
    limit_up_open = (raw_open >= pre_close.mul(1.0 + lim_pct, axis=1) - 0.004) & tradable
    limit_dn_open = (raw_open <= pre_close.mul(1.0 - lim_pct, axis=1) + 0.004) & tradable
    del raw_open, pre_close, idxed      # 之后再也用不到，立刻释放
    gc.collect()

    return dict(cal=cal, codes=codes,
                raw_close=raw_close, amount=amount, circ_mv=circ_mv,
                adj_close=adj_close, adj_open=adj_open, tradable=tradable,
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


def size_neutralize(f: pd.DataFrame, logsize: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """
    截面上把因子对 log 市值做回归，取残差。
    用来回答：剥掉市值这层之后，这个因子还剩下什么。
    """
    x = logsize.where(mask).astype(np.float64)
    y = f.where(mask).astype(np.float64)
    xc = x.sub(x.mean(axis=1), axis=0)
    yc = y.sub(y.mean(axis=1), axis=0)
    sxx = (xc ** 2).sum(axis=1)
    beta = (xc * yc).sum(axis=1) / sxx.where(sxx > 1e-12)
    return yc.sub(xc.mul(beta, axis=0)).astype(np.float32)


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


def scan_all_factors(factors: Dict[str, pd.DataFrame], adj_close: pd.DataFrame,
                     elig: pd.DataFrame, rebal: List[pd.Timestamp], horizon_weeks: int,
                     keys: List[str], neutralize: bool = False,
                     progress=None, gap: int = 1) -> tuple:
    """一次跑完所有因子，返回 (汇总表, 分年度IC表)。"""
    rows, years = [], {}
    ls = factors.get("logsize")
    for n, k in enumerate(keys):
        if k not in factors:
            continue
        f = factors[k]
        if neutralize and ls is not None and k != "logsize":
            f = size_neutralize(f, ls, elig)
        r = layered_test(f, adj_close, elig, rebal, horizon_weeks, 10, gap=gap)
        if progress:
            progress((n + 1) / len(keys), ALL_DEF.get(k, (k,))[0])
        if not r.get("ok"):
            continue
        yr = r["ic_year"]
        same = int((np.sign(yr) == np.sign(r["ic_mean"])).sum()) if len(yr) else 0
        # 三条证据方向是否一致: IC、分组单调性、多空价差
        agree = bool(np.sign(r["ic_mean"]) == np.sign(r["monotonic"])
                     and np.sign(r["ic_mean"]) == np.sign(r["spread"]))
        rows.append({"因子": ALL_DEF.get(k, (k, 0))[0], "key": k,
                     "IC均值": r["ic_mean"], "t(重叠修正)": r["t_nw"],
                     "单调性": r["monotonic"], "多空价差": r["spread"],
                     "方向一致": "是" if agree else "否",
                     "同号年数": f"{same}/{len(yr)}", "_same": same, "_ny": len(yr),
                     "_agree": agree, "IC>0占比": r["ic_pos"], "期数": r["n_period"]})
        years[ALL_DEF.get(k, (k, 0))[0]] = r["ic_year"]
    summ = pd.DataFrame(rows)
    ydf = pd.DataFrame(years).T if years else pd.DataFrame()
    return summ, ydf


def gap_scan(factors: Dict[str, pd.DataFrame], adj_close: pd.DataFrame,
             elig: pd.DataFrame, rebal: List[pd.Timestamp], horizon_weeks: int,
             keys: List[str], gaps=(0, 1, 3, 5, 10), progress=None) -> pd.DataFrame:
    """
    入场延迟扫描 —— 区分真实反转与微观结构噪音的决定性检验。
    真实的定价效应能扛住推迟几天入场；买卖价差跳动造出来的假反转，
    在 gap 从 0 加到 1-3 天时就会大幅衰减，因为噪音只存在于那一天的价格里。
    """
    rows = []
    for n, k in enumerate(keys):
        if k not in factors:
            continue
        row = {"因子": ALL_DEF.get(k, (k, 0))[0], "key": k}
        for g in gaps:
            r = layered_test(factors[k], adj_close, elig, rebal, horizon_weeks, 10, gap=g)
            row[f"t@延迟{g}日"] = r.get("t_nw", np.nan) if r.get("ok") else np.nan
        base = row.get("t@延迟0日", np.nan)
        far = row.get(f"t@延迟{gaps[-1]}日", np.nan)
        row["残留比例"] = (abs(far) / abs(base)) if (np.isfinite(base) and abs(base) > 1e-9
                                                and np.isfinite(far)) else np.nan
        rows.append(row)
        if progress:
            progress((n + 1) / len(keys), row["因子"])
    return pd.DataFrame(rows)


ROUND_TRIP_COST = 0.0003 * 2 + 0.0005 + 0.001 * 2   # 佣金双边 + 印花税 + 滑点双边 = 0.31%


def horizon_scan(factors: Dict[str, pd.DataFrame], adj_close: pd.DataFrame,
                 elig: pd.DataFrame, rebal: List[pd.Timestamp], keys: List[str],
                 horizons=(1, 2, 3, 4, 6, 8), gap: int = 1,
                 cost: float = ROUND_TRIP_COST, progress=None) -> pd.DataFrame:
    """
    持有期扫描 —— 回答"该持有多久"。

    信号有半衰期。持有期短于半衰期，吃到的信号浓度高但换手成本高；
    长于半衰期，大部分持仓时间都在拿过期信号。这里把两边一起算：
      多头超额 = 做多那一组相对当期截面均值的超额（IC为负时是D1，为正时是D10）
      年化毛超额 = 多头超额 × (52/持有周数)
      换手成本   = 单次往返成本 × (52/持有周数)
      净超额     = 年化毛超额 - 换手成本

    这里刻意不用"多空价差的一半"。A股融券做空不现实，如果效应全部来自
    "涨得多的那组暴跌"，纯多头一分钱都吃不到，用价差折半会严重高估。
    """
    rows = []
    for n, k in enumerate(keys):
        if k not in factors:
            continue
        row = {"因子": ALL_DEF.get(k, (k, 0))[0], "key": k}
        best, best_h = -9e9, np.nan
        for h in horizons:
            r = layered_test(factors[k], adj_close, elig, rebal, h, 10, gap=gap)
            if not r.get("ok"):
                row[f"t@{h}周"] = np.nan
                row[f"净超额@{h}周"] = np.nan
                continue
            t_ = r["t_nw"]
            turns = 52.0 / h
            net = r["long_excess"] * turns - cost * turns
            row[f"t@{h}周"] = t_
            row[f"净超额@{h}周"] = net
            if np.isfinite(t_) and abs(t_) >= 2.0 and net > best:
                best, best_h = net, h
                row["_ls"] = r["long_side"]
                row["_lsh"] = r["long_share"]
                row["_sh"] = r["short_excess"] * turns
        # 净超额为负就不该叫"最优"。之前这里会把一个负数显示成推荐值。
        if best > 0:
            row["最优持有周"] = best_h
            row["最优净超额"] = best
            row["做多哪组"] = row.pop("_ls", "")
            row["多头贡献占比"] = row.pop("_lsh", np.nan)
            row["空头年化(拿不到)"] = row.pop("_sh", np.nan)
        else:
            row["最优持有周"] = np.nan
            row["最优净超额"] = np.nan
            row["做多哪组"] = "—"
            row["多头贡献占比"] = np.nan
            row["空头年化(拿不到)"] = np.nan
            for kk in ("_ls", "_lsh", "_sh"):
                row.pop(kk, None)
        rows.append(row)
        if progress:
            progress((n + 1) / len(keys), row["因子"])
    return pd.DataFrame(rows)


def export_bundle(tables: Dict[str, pd.DataFrame]) -> bytes:
    """把所有结果表打包成一个 zip（纯标准库，不依赖 openpyxl）。"""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            if df is None or not len(df):
                continue
            z.writestr(f"{name}.csv", df.to_csv().encode("utf-8-sig"))
    return buf.getvalue()


def factor_corr(factors: Dict[str, pd.DataFrame], elig: pd.DataFrame,
                rebal: List[pd.Timestamp], keys: List[str]) -> pd.DataFrame:
    """调仓日截面秩相关的平均值。用来看这些因子到底是几个独立的赌注。"""
    ks = [k for k in keys if k in factors]
    dates = rebal[::4]                      # 抽样即可，够稳定
    acc = np.zeros((len(ks), len(ks)))
    cnt = 0
    for d in dates:
        if d not in elig.index:
            continue
        m = elig.loc[d]
        cols = []
        for k in ks:
            v = factors[k].loc[d].where(m)
            cols.append(v.rank())
        M = pd.concat(cols, axis=1, keys=ks).dropna()
        if len(M) < 30:
            continue
        acc += M.corr().to_numpy()
        cnt += 1
    if cnt == 0:
        return pd.DataFrame()
    return pd.DataFrame(acc / cnt, index=[ALL_DEF[k][0] for k in ks],
                        columns=[ALL_DEF[k][0] for k in ks])


# ======================================================================
# 四、回测引擎
# ======================================================================
def run_backtest(panel: dict, score: pd.DataFrame, elig: pd.DataFrame,
                 rebal: List[pd.Timestamp], p: dict) -> dict:
    """
    次日开盘成交；涨停不买、跌停不卖、停牌顺延；调仓日按目标权重再平衡（含无交易带）。
    """
    cal = panel["cal"]
    codes = panel["codes"]
    cidx = {c: j for j, c in enumerate(codes)}

    AC = panel["adj_close"].to_numpy(dtype=np.float64)
    AO = panel["adj_open"].to_numpy(dtype=np.float32)
    TRD = panel["tradable"].to_numpy(dtype=bool)
    LU = panel["limit_up_open"].to_numpy(dtype=bool)
    LD = panel["limit_dn_open"].to_numpy(dtype=bool)
    SC = score.reindex(index=cal, columns=codes).to_numpy(dtype=np.float32)
    EL = elig.reindex(index=cal, columns=codes).fillna(False).to_numpy(dtype=bool)

    # 市场宽度 -> 目标仓位
    A = panel["adj_close"]
    above = ((A > A.rolling(60).mean()) & elig).sum(axis=1)
    base = elig.sum(axis=1)
    breadth = (above / base.where(base > 0)).reindex(cal).ffill().fillna(0.5).to_numpy()
    lo_b, hi_b = p["breadth_lo"], p["breadth_hi"]
    scale = np.clip((breadth - lo_b) / max(1e-9, hi_b - lo_b), 0.0, 1.0)
    exposure = p["min_expo"] + (1.0 - p["min_expo"]) * scale
    if not p["use_breadth"]:
        exposure = np.ones_like(exposure)

    rebal_set = set(pd.DatetimeIndex(rebal))
    i0 = max(70, int(np.argmax(cal >= pd.Timestamp(p["bt_start"]))) if (cal >= pd.Timestamp(p["bt_start"])).any() else 70)
    i1 = len(cal) - 1

    cash = float(p["capital"])
    pos: Dict[int, dict] = {}          # j -> {shares, entry_px, entry_i, hw}
    pending: List[tuple] = []          # (j, 'B'/'S', target_shares)
    eq_hist, dates_hist, npos_hist, expo_hist = [], [], [], []
    trades = []

    comm, stamp, slip = p["commission"], p["stamp"], p["slippage"]
    N, buf = int(p["top_n"]), int(p["buffer_rank"])
    min_hold, max_hold = int(p["min_hold_w"]) * 5, int(p["max_hold_w"]) * 5

    def equity(i: int) -> float:
        v = cash
        for j, ps in pos.items():
            px = AC[i, j]
            if not np.isfinite(px):
                px = ps["last_px"]
            v += ps["shares"] * px
        return v

    for i in range(i0, i1 + 1):
        # ---- 1. 用今日开盘价执行昨日挂单 ----
        still = []
        for (j, side, tgt) in pending:
            px = AO[i, j]
            if (not TRD[i, j]) or (not np.isfinite(px)) or px <= 0:
                still.append((j, side, tgt))                    # 停牌顺延
                continue
            if side == "B":
                if LU[i, j]:
                    still.append((j, side, tgt))                # 涨停买不到
                    continue
                if j not in pos and len(pos) >= N:              # 已满仓，不开新仓
                    continue
                fill = px * (1.0 + slip)
                shares = int(tgt // 100) * 100
                cost = shares * fill
                fee = max(5.0, cost * comm)
                if shares <= 0 or cost + fee > cash:
                    afford = int(max(0.0, (cash - 5.0)) / (fill * 1.001) // 100) * 100
                    shares = min(shares, afford)
                    cost = shares * fill
                    fee = max(5.0, cost * comm) if shares > 0 else 0.0
                if shares <= 0:
                    continue
                cash -= cost + fee
                if j in pos:
                    old = pos[j]
                    tot = old["shares"] + shares
                    old["entry_px"] = (old["entry_px"] * old["shares"] + fill * shares) / tot
                    old["shares"] = tot
                else:
                    pos[j] = {"shares": shares, "entry_px": fill, "entry_i": i,
                              "hw": fill, "last_px": fill}
                trades.append({"date": cal[i], "code": codes[j], "side": "买",
                               "price": fill, "shares": shares, "amount": cost})
            else:
                if LD[i, j]:
                    still.append((j, side, tgt))                # 跌停卖不掉
                    continue
                ps = pos.get(j)
                if ps is None:
                    continue
                shares = int(min(ps["shares"], max(0, tgt)))
                shares = int(shares // 100) * 100 if shares < ps["shares"] else ps["shares"]
                if shares <= 0:
                    continue
                fill = px * (1.0 - slip)
                gross = shares * fill
                fee = max(5.0, gross * comm) + gross * stamp
                cash += gross - fee
                pnl = (fill - ps["entry_px"]) / ps["entry_px"]
                trades.append({"date": cal[i], "code": codes[j], "side": "卖",
                               "price": fill, "shares": shares, "amount": gross,
                               "ret": pnl, "hold_days": i - ps["entry_i"]})
                ps["shares"] -= shares
                if ps["shares"] <= 0:
                    pos.pop(j, None)
        pending = still

        # ---- 2. 收盘估值 ----
        for j, ps in pos.items():
            if np.isfinite(AC[i, j]):
                ps["last_px"] = AC[i, j]
                ps["hw"] = max(ps["hw"], AC[i, j])
        eq = equity(i)
        eq_hist.append(eq)
        dates_hist.append(cal[i])
        npos_hist.append(len(pos))
        expo_hist.append(exposure[i])
        if i == i1:
            break

        # ---- 3. 收盘生成明日订单 ----
        forced = set()
        for j, ps in list(pos.items()):
            px = ps["last_px"]
            held = i - ps["entry_i"]
            hit_trail = px <= ps["hw"] * (1.0 - p["trail_stop"])
            hit_hard = px <= ps["entry_px"] * (1.0 - p["hard_stop"])
            if hit_trail or hit_hard or held >= max_hold - 1:
                pending.append((j, "S", ps["shares"]))
                forced.add(j)

        if cal[i] in rebal_set:
            sc = SC[i].copy()
            sc[~EL[i]] = np.nan
            valid = np.where(np.isfinite(sc))[0]
            if len(valid) >= 5:
                order = valid[np.argsort(-sc[valid])]
                rank = {j: r + 1 for r, j in enumerate(order)}
                target = list(order[:N])

                # 缓冲带：已持仓且排名仍在 buf 内则保留
                keep = [j for j in pos if j not in forced and rank.get(j, 10 ** 9) <= buf
                        and (i - pos[j]["entry_i"]) >= min_hold]
                keep += [j for j in pos if j not in forced and (i - pos[j]["entry_i"]) < min_hold]
                keep = list(dict.fromkeys(keep))
                final = list(dict.fromkeys(keep + [j for j in target if j not in keep]))[:N]

                w = exposure[i] / max(1, N)
                for j in list(pos):
                    if j in forced:
                        continue
                    if j not in final:
                        if (i - pos[j]["entry_i"]) >= min_hold:
                            pending.append((j, "S", pos[j]["shares"]))
                        continue
                    tgt_val = eq * w
                    cur_val = pos[j]["shares"] * pos[j]["last_px"]
                    if cur_val > tgt_val * (1.0 + p["no_trade_band"]):
                        cut = int((cur_val - tgt_val) / pos[j]["last_px"] // 100) * 100
                        if cut > 0:
                            pending.append((j, "S", cut))
                    elif cur_val < tgt_val * (1.0 - p["no_trade_band"]):
                        add = int((tgt_val - cur_val) / pos[j]["last_px"] // 100) * 100
                        if add > 0:
                            pending.append((j, "B", add))
                for j in final:
                    if j in pos:
                        continue
                    px = AC[i, j]
                    if np.isfinite(px) and px > 0:
                        pending.append((j, "B", int(eq * w / px // 100) * 100))

    eqs = pd.Series(eq_hist, index=pd.DatetimeIndex(dates_hist), name="策略")
    tdf = pd.DataFrame(trades)

    # 基准：股票池等权（吃掉板块 beta 后才谈 alpha）
    ret_all = panel["adj_close"].pct_change()
    bench_r = ret_all.where(elig.shift(1).fillna(False)).mean(axis=1)
    bench_r = bench_r.reindex(eqs.index).fillna(0.0)
    bench = float(p["capital"]) * (1.0 + bench_r).cumprod()
    bench.name = "股票池等权"

    return {"equity": eqs, "bench": bench, "trades": tdf,
            "npos": pd.Series(npos_hist, index=eqs.index),
            "exposure": pd.Series(expo_hist, index=eqs.index)}


def perf_stats(eq: pd.Series, trades: pd.DataFrame, npos: pd.Series,
               bench: Optional[pd.Series] = None) -> dict:
    r = eq.pct_change().dropna()
    n = len(eq)
    yrs = max(n / 252.0, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / yrs) - 1.0
    dd = eq / eq.cummax() - 1.0
    sharpe = r.mean() / r.std() * np.sqrt(252.0) if r.std() > 1e-12 else np.nan
    out = {"年化收益": cagr, "最大回撤": dd.min(), "夏普": sharpe,
           "卡玛": cagr / abs(dd.min()) if dd.min() < -1e-9 else np.nan,
           "总收益": eq.iloc[-1] / eq.iloc[0] - 1.0, "回测年数": yrs}
    if bench is not None and len(bench) > 5:
        b = bench.reindex(eq.index).ffill()
        bcagr = (b.iloc[-1] / b.iloc[0]) ** (1.0 / yrs) - 1.0
        out["基准年化"] = bcagr
        out["年化超额"] = cagr - bcagr
    if trades is not None and len(trades) and "ret" in trades.columns:
        s = trades.dropna(subset=["ret"])
        if len(s):
            win = s[s["ret"] > 0]["ret"]
            los = s[s["ret"] <= 0]["ret"]
            out["交易笔数"] = len(s)
            out["胜率"] = len(win) / len(s)
            out["盈亏比"] = (win.mean() / abs(los.mean())) if len(los) and abs(los.mean()) > 1e-9 else np.nan
            out["平均持仓周"] = s["hold_days"].mean() / 5.0
    # 空窗统计
    empty = (npos == 0)
    wk = empty.groupby([npos.index.isocalendar().year, npos.index.isocalendar().week]).all()
    by_year = wk.groupby(level=0).sum()
    out["_空窗周_按年"] = by_year
    out["最大单年空窗周"] = int(by_year.max()) if len(by_year) else 0
    return out


# ======================================================================
# 五、Streamlit 界面
# ======================================================================
def _mem_mb(panel: dict, factors: Optional[dict] = None) -> float:
    tot = 0
    for k, v in panel.items():
        if isinstance(v, pd.DataFrame):
            tot += v.memory_usage(deep=False).sum()
    if factors:
        for v in factors.values():
            tot += v.memory_usage(deep=False).sum()
    return tot / 1e6


# Streamlit 每动一次控件就重跑整个脚本。1500 只股票时，不加缓存的话
# build_eligibility + composite_score 会在每次拖动滑块时重算一遍全量矩阵，
# 几秒钟的卡顿加上反复分配大数组，是这个应用最现实的崩溃来源。
if st is not None:
    @st.cache_data(show_spinner=False, max_entries=2)
    def cached_elig(_panel, _basic, _uni, key: str, mv_lo, mv_hi, min_price, min_amt, min_days):
        return build_eligibility(_panel, _basic, _uni, mv_lo, mv_hi, min_price, min_amt, int(min_days))

    @st.cache_data(show_spinner=False, max_entries=2)
    def cached_score(_factors, _elig, key: str, wkey: tuple):
        return composite_score(_factors, _elig, dict(wkey))
else:                                              # 无 streamlit 时直通
    def cached_elig(_panel, _basic, _uni, key, mv_lo, mv_hi, min_price, min_amt, min_days):
        return build_eligibility(_panel, _basic, _uni, mv_lo, mv_hi, min_price, min_amt, int(min_days))

    def cached_score(_factors, _elig, key, wkey):
        return composite_score(_factors, _elig, dict(wkey))


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.2%}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def main():
    st.set_page_config(page_title="科技股排名轮动选股系统", layout="wide")
    st.title("科技 / 军工 / 新能源 / 机器人 —— 排名轮动选股系统")
    st.caption("排名轮动，非信号触发：每周持有分数最高的 N 只，结构上不存在空窗。")

    ss = st.session_state
    ss.setdefault("panel", None)

    # ---------------- 侧边栏 ----------------
    with st.sidebar:
        st.header("① 数据")
        token = st.text_input("Tushare Token", type="password",
                              value=os.environ.get("TUSHARE_TOKEN", ""))
        c1, c2 = st.columns(2)
        start = c1.date_input("起始", dt.date(2018, 1, 1))
        end = c2.date_input("结束", dt.date.today())
        l1 = st.multiselect("申万一级行业（整体纳入）", SW_L1_ALL, SW_L1_DEFAULT)
        l2 = st.multiselect("申万二级行业（子行业纳入）", SW_L2_ALL, SW_L2_DEFAULT)
        d1, d2 = st.columns(2)
        workers = d1.slider("并发线程数", 1, 8, 4)
        per_min = d2.slider("每分钟请求上限", 60, 800, 400, 20)
        st.caption("频次限制按账号算，线程共用一个限流器，不会因并发被封。"
                   "上限按你的 Tushare 积分设：2000 积分对应 500/分钟。")
        use_cache = st.checkbox("使用本地缓存（中断可续传）", True)
        prescreen = st.checkbox("下载前先做市值预筛", True,
                                help="用 16 次全市场快照，只保留历史上曾进入市值区间的股票。"
                                     "用的是各采样日的当期市值，无前视偏差，通常能砍掉 1/3 下载量。")
        max_stk = st.number_input("最多下载股票数（0=不限）", 0, 3000, 0, 50)
        dl = st.button("下载 / 更新数据", type="primary", use_container_width=True)

        st.header("② 股票池硬约束")
        mv_lo = st.number_input("流通市值下限（亿）", 1.0, 5000.0, 50.0, 10.0)
        mv_hi = st.number_input("流通市值上限（亿）", 10.0, 20000.0, 1000.0, 50.0)
        min_price = st.number_input("最低股价（元）", 0.0, 200.0, 10.0, 1.0)
        min_amt = st.number_input("20日均成交额下限（亿）", 0.0, 50.0, 2.0, 0.5)
        min_days = st.number_input("上市满（自然日）", 0, 1500, 365, 30)

    if dl:
        if not token:
            st.error("请先填 Tushare Token")
            st.stop()
        try:
            import tushare as ts
        except ImportError:
            st.error("未安装 tushare：pip install tushare")
            st.stop()
        ts.set_token(token)
        pro = ts.pro_api(token)
        lim = Limiter(per_min)
        API_ERRORS.clear()
        s_str, e_str = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

        # 重新下载前先释放上一份数据，否则新旧两份同时在内存里
        for k in ("panel", "factors", "last_bt"):
            ss.pop(k, None)
        gc.collect()

        with st.status("正在获取数据…", expanded=True) as status:
            st.write("1/4 取申万成分股…")
            uni = fetch_universe(pro, lim, l1, l2)
            if not len(uni):
                st.error("行业成分股为空。可能是 Tushare 积分不足，无法调用申万接口。")
                st.stop()
            codes = list(uni["ts_code"])
            st.write(f"   候选 {len(codes)} 只")

            st.write("2/4 取股票基础信息（含退市）…")
            basic = fetch_stock_basic(pro, lim)

            if prescreen:
                st.write("3/4 市值预筛…")
                n0 = len(codes)
                codes = prescreen_by_mv(pro, lim, codes, s_str, e_str, mv_lo, mv_hi)
                st.write(f"   {n0} → {len(codes)} 只（剔除 {n0-len(codes)} 只从未进入市值区间的）")
            else:
                st.write("3/4 跳过预筛")

            if max_stk:
                codes = codes[: int(max_stk)]
            est_mb = len(codes) * 1950 * 4 * 14 / 1e6
            st.write(f"4/4 并发下载 {len(codes)} 只（{workers} 线程，预计常驻内存 ~{est_mb:.0f} MB）…")

            bar = st.progress(0.0)
            t0 = time.time()

            def _cb(done, total, ok):
                el = time.time() - t0
                eta = el / max(done, 1) * (total - done)
                bar.progress(done / total,
                             text=f"{done}/{total}  成功 {ok}  已用 {el/60:.1f} 分  剩余约 {eta/60:.1f} 分")

            px = download_all(token, codes, s_str, e_str, lim, use_cache, workers, _cb)
            if not px:
                st.error("一只股票都没下到。检查 Token、积分权限和网络。")
                st.stop()

            st.write("构建面板与因子…")
            panel = build_panel(px)
            px.clear()
            del px
            gc.collect()
            factors = compute_factors(panel["adj_close"], panel["amount"], circ_mv=panel["circ_mv"])

            ss["panel"] = panel
            ss["basic"] = basic
            ss["uni"] = uni
            ss["factors"] = factors
            ss["data_key"] = f"{len(panel['codes'])}|{panel['cal'][0]:%Y%m%d}|{panel['cal'][-1]:%Y%m%d}"
            ss["api_errors"] = list(API_ERRORS)
            st.cache_data.clear()
            status.update(
                label=f"完成：{len(panel['codes'])} 只 × {len(panel['cal'])} 个交易日，"
                      f"实占内存 {_mem_mb(panel, factors):.0f} MB，耗时 {(time.time()-t0)/60:.1f} 分钟",
                state="complete")

    if ss.get("panel") is None:
        st.info("左侧填入 Tushare Token 后点「下载 / 更新数据」。首次全量下载约需 5-15 分钟，之后走本地缓存。")
        st.stop()

    panel, basic, uni, factors = ss["panel"], ss["basic"], ss["uni"], ss["factors"]

    dkey = ss.get("data_key", "na")
    elig = cached_elig(panel, basic, uni, dkey, mv_lo, mv_hi, min_price, min_amt, int(min_days))
    rebal_all = weekly_rebal_dates(panel["cal"])

    t1, t2, t3, t4 = st.tabs(["数据总览", "因子分层检验", "策略回测", "当前选股"])

    # ---------------- 数据总览 ----------------
    with t1:
        cnt = elig.sum(axis=1)
        a, b, c, d = st.columns(4)
        a.metric("下载股票数", len(panel["codes"]))
        b.metric("交易日数", len(panel["cal"]))
        c.metric("当前合格数", int(cnt.iloc[-1]))
        d.metric("数据占用内存", f"{_mem_mb(panel, factors):.0f} MB")
        st.caption(f"历史平均合格数 {cnt.mean():.0f} 只")
        st.subheader("每日合格股票数量")
        st.line_chart(cnt.rename("合格数"))
        st.caption("若某段时间合格数长期低于 30，说明市值/股价门槛在那个阶段过严，排名的区分度会下降。")
        if ss.get("api_errors"):
            with st.expander(f"接口异常 {len(ss['api_errors'])} 条"):
                st.write(ss["api_errors"][-40:])

    # ---------------- 因子分层检验 ----------------
    with t2:
        st.subheader("先回答一个问题：这些打分有没有排序能力？")
        st.markdown(
            "**t 的符号只说明方向，门槛是 |t| > 2。** t 为负不是失败，是因子要反过来用。\n\n"
            "但看结果之前先记住两件事：① 动量、趋势质量、相对强度、距高点、5日反转"
            "本质都是「近期价格强弱」的变体，彼此高度相关，**六个因子大约只是两个独立赌注**；"
            "② 池子有 50-1000 亿的市值上下限，动量高的股票平均更靠近上沿，"
            "所以「动量为负」可能只是「小市值跑赢」的伪装——下面的市值诊断就是查这个的。")

        c0 = st.columns(5)
        hz = c0[0].select_slider("持有期（周）", [1, 2, 4, 6, 8], 4)
        gap = c0[1].select_slider("入场延迟（交易日）", [0, 1, 3, 5, 10], 1,
                                  help="0 = 用排名当日收盘价入场。这会让因子的分子和"
                                       "未来收益的分母共用同一个价格，噪音会凭空造出负相关。"
                                       "回测是次日开盘成交，所以 1 才是与回测一致的口径。")
        ls_ = c0[2].date_input("样本起", dt.date(2018, 1, 1), key="ls")
        le_ = c0[3].date_input("样本止", dt.date.today(), key="le")
        neu = c0[4].checkbox("市值中性化", False,
                             help="截面上把因子对 log 流通市值回归取残差。"
                                  "勾选后再看一遍 IC：如果因子显著性大幅塌掉，"
                                  "说明它原本的效果主要来自市值暴露，不是因子本身。")
        rb = [d for d in rebal_all if pd.Timestamp(ls_) <= d <= pd.Timestamp(le_)]

        if st.button("扫描全部因子（含市值诊断）", type="primary"):
            bar = st.progress(0.0)
            summ, ydf = scan_all_factors(factors, panel["adj_close"], elig, rb,
                                         int(hz), TEST_KEYS, neu,
                                         lambda p, n: bar.progress(p, text=n), gap=int(gap))
            ss["scan"] = (summ, ydf, factor_corr(factors, elig, rb, TEST_KEYS),
                          int(hz), neu, int(gap))
            bar.empty()

        if ss.get("scan"):
            summ, ydf, corr, hz_done, neu_done, gap_done = ss["scan"]
            tag = "（已市值中性化）" if neu_done else ""
            st.markdown(f"**汇总{tag}　持有期 {hz_done} 周　入场延迟 {gap_done} 日**")
            show = summ.drop(columns=[c for c in summ.columns
                                      if c == "key" or c.startswith("_")]).copy()
            st.dataframe(
                show.style.format({"IC均值": "{:.4f}", "t(重叠修正)": "{:.2f}",
                                   "单调性": "{:.2f}", "多空价差": "{:.2%}",
                                   "IC>0占比": "{:.1%}"})
                    .background_gradient(subset=["t(重叠修正)"], cmap="RdYlGn", vmin=-4, vmax=4),
                use_container_width=True)
            st.caption(
                "**「方向一致」是这张表里最该先看的一列。** IC、分组单调性、多空价差"
                "是同一件事的三种量法，方向不一致说明因子的效果集中在少数高波动时段，"
                "总均值和分组结果各说各话——这种因子不能用。"
                "「同号年数」低于 8/9 的同样不能用：总均值只是两段相反行情的平均数。")

            diag = summ[summ["key"] == "logsize"]
            if len(diag):
                d0 = diag.iloc[0]
                st.markdown("**市值诊断**")
                if abs(d0["t(重叠修正)"]) >= 2:
                    direc = "小市值跑赢大市值" if d0["IC均值"] < 0 else "大市值跑赢小市值"
                    st.warning(
                        f"流通市值本身就是个显著因子（IC {d0['IC均值']:.4f}，"
                        f"修正 t {d0['t(重叠修正)']:.2f}），方向是**{direc}**。"
                        "请务必勾选「市值中性化」再扫一遍：如果价格类因子的显著性"
                        "在中性化后大幅塌掉，那它们原本测出来的效果主要是市值暴露，"
                        "照着这个结果去建仓等于在赌市值风格，不是在赌你想赌的东西。")
                else:
                    st.success(f"流通市值本身不显著（修正 t {d0['t(重叠修正)']:.2f}），"
                               "价格类因子的结果没有被市值污染。")

            st.markdown("**分年度 IC**")
            if len(ydf):
                st.dataframe(ydf.style.format("{:.4f}")
                             .background_gradient(cmap="RdYlGn", vmin=-0.08, vmax=0.08),
                             use_container_width=True)
                st.caption("这张表比总均值重要得多。如果某因子在 2019-2021 是一个符号、"
                           "2023 年之后翻成另一个符号，那它的总均值只是两段相反行情的平均数，"
                           "拿去做实盘等于赌行情会退回从前。逐年同号才叫稳定。")

            if len(corr):
                st.markdown("**因子截面相关性**")
                st.dataframe(corr.style.format("{:.2f}")
                             .background_gradient(cmap="coolwarm", vmin=-1, vmax=1),
                             use_container_width=True)
                st.caption("相关性 0.7 以上的因子之间几乎没有增量信息，"
                           "把它们一起加进打分只是把同一个赌注下三遍，并不会分散风险。")

            tradable = summ[summ["key"].isin(FACTOR_KEYS)].copy()
            # 三道门槛全过才给权重。只看 t 会把"效果集中在少数时段"和
            # "两段相反行情的平均数"这两类因子放进来，那是在拟合过去。
            def _pass(r):
                return (np.isfinite(r["t(重叠修正)"]) and abs(r["t(重叠修正)"]) >= 2.0
                        and r["_agree"] and r["_ny"] > 0 and r["_same"] / r["_ny"] >= 8 / 9)
            tradable["通过"] = tradable.apply(_pass, axis=1)
            passed = tradable[tradable["通过"]]
            st.markdown(f"**三道门槛全过的可交易因子：{len(passed)} / {len(tradable)}**")
            st.caption("门槛：|修正 t| ≥ 2　且　IC/单调性/价差方向一致　且　同号年数 ≥ 8/9")
            for _, r in tradable.iterrows():
                why = []
                if not (np.isfinite(r["t(重叠修正)"]) and abs(r["t(重叠修正)"]) >= 2.0):
                    why.append("不显著")
                if not r["_agree"]:
                    why.append("三条证据方向打架")
                if r["_ny"] and r["_same"] / r["_ny"] < 8 / 9:
                    why.append(f"逐年符号不稳({r['同号年数']})")
                st.write(("✅ " if r["通过"] else "❌ ") + r["因子"]
                         + ("" if r["通过"] else "　— " + "、".join(why)))

            if len(passed) == 0:
                st.error("一个都没过。不要去调仓位和止损，那救不回来——"
                         "问题在因子本身，需要换一批因子重来。")
            else:
                # 等权 + 符号，不按 t 的大小定权重：t 越大权重越大等于对
                # 样本内的显著性做二次拟合，样本外通常更差。
                sug = {k: 0.0 for k in FACTOR_KEYS}
                for _, r in passed.iterrows():
                    sug[r["key"]] = float(np.sign(r["t(重叠修正)"]))
                st.write("建议权重（通过的等权取符号，未通过置 0）：",
                         {ALL_DEF[k][0]: v for k, v in sug.items() if v != 0})
                st.caption("刻意不按 t 的大小分配权重——那等于对样本内显著性再拟合一次，"
                           "样本外通常更差。等权更稳。")
                if st.button("把建议权重写入回测页"):
                    for k, v in sug.items():
                        ss["w_" + k] = v
                    ss.pop("last_bt", None)
                    st.rerun()

        st.divider()
        st.markdown("### 入场延迟扫描 —— 区分真反转与噪音")
        st.markdown(
            "反转类因子的分子用的是排名当日的收盘价，而未来收益的分母也是它。"
            "这一天价格里的买卖价差跳动会同时抬高因子、压低未来收益，**凭空造出负相关**。"
            "真实的定价效应扛得住推迟几天入场，价差跳动扛不住。\n\n"
            "**看 0 日到 1 日那一步的落差**：崩塌就是噪音，稳住就是真效应。")
        if st.button("运行延迟扫描"):
            bar2 = st.progress(0.0)
            gs = gap_scan(factors, panel["adj_close"], elig, rb, int(hz), FACTOR_KEYS,
                          gaps=(0, 1, 3, 5, 10),
                          progress=lambda p, n: bar2.progress(p, text=n))
            ss["gapscan"] = gs
            bar2.empty()
        if ss.get("gapscan") is not None:
            gs = ss["gapscan"]
            gcols = [c for c in gs.columns if c.startswith("t@")]
            st.dataframe(
                gs.drop(columns=["key"]).style
                  .format({**{c: "{:.2f}" for c in gcols}, "残留比例": "{:.0%}"})
                  .background_gradient(subset=gcols, cmap="RdYlGn", vmin=-5, vmax=5),
                use_container_width=True)
            drops = []
            for _, r in gs.iterrows():
                t0, t1 = r.get("t@延迟0日"), r.get("t@延迟1日")
                if np.isfinite(t0) and abs(t0) >= 2 and np.isfinite(t1):
                    if abs(t1) < abs(t0) * 0.6:
                        drops.append(r["因子"])
            if drops:
                st.error("以下因子在延迟 1 天后显著性就崩掉了一半以上，"
                         "**它们测出来的效应主要是微观结构噪音，不可交易**：\n\n"
                         + "、".join(drops))
            else:
                st.success("没有因子在延迟 1 天时崩塌，反转效应扛得住延迟入场。")

        st.divider()
        st.markdown("### 持有期扫描 —— 该拿多久")
        st.markdown(
            "信号有半衰期。持有期短于半衰期，信号浓度高但换手成本高；"
            "长于半衰期，大半仓位时间都在拿过期信号。下表把两边一起算："
            f"单次往返成本按 {ROUND_TRIP_COST:.2%}（佣金双边+印花税+滑点双边）。\n\n"
            "**净超额只算多头那一组**（IC 为负时是 D1，为正时是 D10）相对当期截面均值的超额。"
            "刻意不用「多空价差的一半」——A 股融券做空不现实，如果效应全部来自"
            "「涨得多的那组暴跌」，纯多头一分钱都吃不到。"
            "「多头贡献占比」低于 40% 就说明钱主要在你拿不到的空头端。")
        if st.button("运行持有期扫描"):
            bar3 = st.progress(0.0)
            hs = horizon_scan(factors, panel["adj_close"], elig, rb, FACTOR_KEYS,
                              horizons=(1, 2, 3, 4, 6, 8), gap=int(gap),
                              progress=lambda p, n: bar3.progress(p, text=n))
            ss["hscan"] = hs
            bar3.empty()
        if ss.get("hscan") is not None:
            hs = ss["hscan"]
            tc = [c for c in hs.columns if c.startswith("t@")]
            nc = [c for c in hs.columns if c.startswith("净超额@")]
            st.dataframe(
                hs.drop(columns=["key"]).style
                  .format({**{c: "{:.2f}" for c in tc},
                           **{c: "{:.1%}" for c in nc},
                           "最优净超额": "{:.1%}", "最优持有周": "{:.0f}",
                           "多头贡献占比": "{:.0%}", "空头年化(拿不到)": "{:.1%}"},
                          na_rep="—")
                  .background_gradient(subset=nc, cmap="RdYlGn", vmin=-0.15, vmax=0.15),
                use_container_width=True)
            good = hs.dropna(subset=["最优净超额"])
            good = good[good["最优净超额"] > 0]
            if len(good):
                wk = good.sort_values("最优净超额", ascending=False).iloc[0]
                st.success(f"净超额最高的是「{wk['因子']}」：做多 {wk['做多哪组']} 组、"
                           f"持有 {wk['最优持有周']:.0f} 周，估计年化净超额 {wk['最优净超额']:.1%}，"
                           f"其中多头贡献占比 {wk['多头贡献占比']:.0%}。"
                           "请把回测页的最长持有期调到这个量级——"
                           "拿满 8 周意味着大半仓位时间都在持有已经过期的信号。")
                low = good[good["多头贡献占比"] < 0.4]
                if len(low):
                    st.warning("以下因子的钱主要在空头端（涨得多的那组暴跌），"
                               "纯多头拿不到，别被总价差骗了：\n\n"
                               + "、".join(low["因子"].tolist()))
            else:
                st.error("没有任何因子在任何持有期上做到净超额为正。"
                         "扣掉换手成本后这个方向不成立，不要往下做回测。")

        st.divider()
        st.markdown("### 一键导出全部结果")
        _tabs = {}
        if ss.get("scan"):
            _tabs["01_汇总"] = ss["scan"][0].drop(
                columns=[c for c in ss["scan"][0].columns if c.startswith("_")])
            _tabs["02_分年度IC"] = ss["scan"][1]
            _tabs["03_因子相关性"] = ss["scan"][2]
        if ss.get("gapscan") is not None:
            _tabs["04_入场延迟扫描"] = ss["gapscan"]
        if ss.get("hscan") is not None:
            _tabs["05_持有期扫描"] = ss["hscan"]
        if _tabs:
            meta = pd.DataFrame([{
                "导出时间": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "股票数": len(panel["codes"]), "交易日数": len(panel["cal"]),
                "样本区间": f"{ls_}~{le_}", "持有期(周)": hz, "入场延迟(日)": gap,
                "市值中性化": "是" if neu else "否",
                "市值区间(亿)": f"{mv_lo:.0f}-{mv_hi:.0f}", "最低股价": min_price,
                "调仓次数": len(rb)}]).T.rename(columns={0: "值"})
            _tabs["00_运行参数"] = meta
            st.download_button(
                f"下载全部结果（{len(_tabs)} 张表，zip）",
                export_bundle(dict(sorted(_tabs.items()))),
                f"factor_scan_{dt.date.today():%Y%m%d}.zip", "application/zip",
                type="primary", use_container_width=True)
            st.caption("包含运行参数、汇总、分年度 IC、相关性矩阵、延迟扫描、持有期扫描。"
                       "参数表一并导出，免得回头对不上是哪次跑的。")
        else:
            st.caption("先运行上面的扫描，这里才会出现下载按钮。")

        st.divider()
        st.markdown("**单因子细看**")
        fkey = st.selectbox("因子", TEST_KEYS, format_func=lambda k: ALL_DEF[k][0])
        if st.button("画分层曲线"):
            f = factors[fkey]
            if neu and "logsize" in factors and fkey != "logsize":
                f = size_neutralize(f, factors["logsize"], elig)
            res = layered_test(f, panel["adj_close"], elig, rb, int(hz), 10, gap=int(gap))
            if not res.get("ok"):
                st.error("样本不足，放宽日期或降低门槛。")
            else:
                m = st.columns(5)
                m[0].metric("多空价差", f"{res['spread']:.2%}")
                m[1].metric("IC 均值", f"{res['ic_mean']:.4f}")
                m[2].metric("t(重叠修正)", f"{res['t_nw']:.2f}")
                m[3].metric("单调性", f"{res['monotonic']:.2f}")
                m[4].metric("IC>0 占比", f"{res['ic_pos']:.1%}")
                st.bar_chart(res["group_excess"].rename(f"未来{hz}周超额（相对截面均值）"))
                st.caption(f"做多 **{res['long_side']}** 组，多头超额 {res['long_excess']:+.2%}／"
                           f"{hz}周，空头端 {res['short_excess']:+.2%}（A股拿不到）。"
                           f"多头贡献占比 {res['long_share']:.0%}。")
                st.line_chart(res["curve"])
                st.line_chart(res["ic"].rolling(12).mean().rename("IC(12期均线)"))

    # ---------------- 策略回测 ----------------
    with t3:
        st.subheader("完整策略回测")
        w1, w2 = st.columns([2, 3])
        with w1:
            st.markdown("**因子权重**")
            weights = {}
            for k in FACTOR_KEYS:
                weights[k] = st.slider(FACTOR_DEF[k][0], -2.0, 2.0, FACTOR_DEF[k][1], 0.1, key="w_" + k)
        with w2:
            g1, g2 = st.columns(2)
            top_n = g1.slider("持股数 N", 3, 15, 5)
            buf = g2.slider("缓冲带（跌出该名次才卖）", top_n, 60, min(15, max(top_n, 15)))
            min_hw = g1.slider("最短持有（周）", 1, 4, 1)
            max_hw = g2.slider("最长持有（周）", 2, 12, 8)
            trail = g1.slider("移动止损", 0.05, 0.30, 0.12, 0.01)
            hard = g2.slider("固定止损", 0.05, 0.30, 0.10, 0.01)
            use_bd = g1.checkbox("市场宽度调仓位", True)
            min_ex = g2.slider("最低仓位", 0.0, 1.0, 0.30, 0.05)
            bt_s = g1.date_input("回测起", dt.date(2018, 1, 1), key="bs")
            bt_e = g2.date_input("回测止", dt.date.today(), key="be")
            st.caption("建议：2018-2022 作为样本内调参，2023 年之后只跑一次，不回头改。")

        if st.button("运行回测", type="primary"):
            score = cached_score(factors, elig, dkey, tuple(sorted(weights.items())))
            rb = [d for d in rebal_all if pd.Timestamp(bt_s) <= d <= pd.Timestamp(bt_e)]
            prm = dict(capital=1_000_000.0, top_n=top_n, buffer_rank=buf,
                       min_hold_w=min_hw, max_hold_w=max_hw,
                       trail_stop=trail, hard_stop=hard,
                       use_breadth=use_bd, min_expo=min_ex, breadth_lo=0.20, breadth_hi=0.60,
                       commission=0.0003, stamp=0.0005, slippage=0.001,
                       no_trade_band=0.25, bt_start=bt_s)
            with st.spinner("回测中…"):
                r = run_backtest(panel, score, elig, rb, prm)
                eq = r["equity"].loc[pd.Timestamp(bt_s):pd.Timestamp(bt_e)]
                bh = r["bench"].reindex(eq.index)
                stats = perf_stats(eq, r["trades"], r["npos"].reindex(eq.index), bh)
            ss["last_bt"] = (eq, bh, r, stats)

        if ss.get("last_bt"):
            eq, bh, r, stats = ss["last_bt"]
            k = st.columns(4)
            k[0].metric("年化收益", _fmt(stats.get("年化收益")))
            k[1].metric("最大回撤", _fmt(stats.get("最大回撤")))
            k[2].metric("年化超额（对池内等权）", _fmt(stats.get("年化超额")))
            k[3].metric("夏普", f"{stats.get('夏普', float('nan')):.2f}")
            k2 = st.columns(4)
            k2[0].metric("胜率", _fmt(stats.get("胜率", float("nan"))))
            k2[1].metric("盈亏比", f"{stats.get('盈亏比', float('nan')):.2f}")
            k2[2].metric("平均持仓周", f"{stats.get('平均持仓周', float('nan')):.1f}")
            k2[3].metric("最大单年空窗周", stats.get("最大单年空窗周", 0))

            norm = pd.DataFrame({"策略": eq / eq.iloc[0], "股票池等权": bh / bh.iloc[0]})
            st.line_chart(norm)
            st.line_chart((eq / eq.cummax() - 1.0).rename("回撤"))
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**每年空窗周数**（约束：≤5）")
                st.dataframe(stats["_空窗周_按年"].rename("空窗周"), use_container_width=True)
            with cB:
                st.markdown("**仓位与持股数**")
                st.line_chart(pd.DataFrame({"仓位": r["exposure"].reindex(eq.index),
                                            "持股数/10": r["npos"].reindex(eq.index) / 10.0}))
            if len(r["trades"]):
                st.markdown("**成交明细**")
                st.dataframe(r["trades"].tail(300), use_container_width=True, height=320)
                st.download_button("下载全部成交记录 CSV",
                                   r["trades"].to_csv(index=False).encode("utf-8-sig"),
                                   "trades.csv", "text/csv")
            st.info("判断标准：样本外年化超过池内等权基准 8-10 个百分点、最大回撤 40% 以内，"
                    "这个系统就可用。达不到就承认，不要继续调参。")

    # ---------------- 当前选股 ----------------
    with t4:
        st.subheader("最新一期选股")
        weights_now = {k: st.session_state.get("w_" + k, FACTOR_DEF[k][1]) for k in FACTOR_KEYS}
        score = cached_score(factors, elig, dkey, tuple(sorted(weights_now.items())))
        d = panel["cal"][-1]
        s = score.loc[d].dropna().sort_values(ascending=False)
        if not len(s):
            st.warning("最新交易日没有合格股票，请放宽筛选条件。")
        else:
            nshow = st.slider("显示前几名", 5, 50, 20)
            top = s.head(nshow)
            nm = basic.set_index("ts_code")["name"].to_dict() if len(basic) else {}
            ind = uni.set_index("ts_code")["ind_name"].to_dict() if len(uni) else {}
            tb = pd.DataFrame({
                "代码": top.index,
                "名称": [nm.get(c, "") for c in top.index],
                "行业": [ind.get(c, "") for c in top.index],
                "收盘价": [panel["raw_close"].loc[d, c] for c in top.index],
                "流通市值(亿)": [panel["circ_mv"].loc[d, c] / 1e4 for c in top.index],
                "综合分": top.values,
            })
            for k in FACTOR_KEYS:
                tb[FACTOR_DEF[k][0]] = [factors[k].loc[d, c] for c in top.index]
            st.caption(f"数据日期：{d:%Y-%m-%d}　合格池：{int(elig.loc[d].sum())} 只")
            st.dataframe(tb.round(3), use_container_width=True, height=520)
            st.download_button("下载选股结果 CSV", tb.to_csv(index=False).encode("utf-8-sig"),
                               f"picks_{d:%Y%m%d}.csv", "text/csv")
        st.caption("提醒：ST 判定用的是当前名称，历史上曾被 ST 后来摘帽的股票无法完全还原，"
                   "这是本系统已知的一个小口径偏差。")


if __name__ == "__main__" and st is not None:
    main()
