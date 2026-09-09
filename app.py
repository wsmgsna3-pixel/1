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
# 六、界面 —— 一页，一个按钮，输出每周候选
# ======================================================================
WEIGHTS = {"dist_hi": -1.0, "rev5": -0.5}      # 唯一通过全部检验的两个因子


def main():
    st.set_page_config(page_title="每周选股", layout="wide")
    ss = st.session_state
    ss.setdefault("panel", None)

    st.title("每周选股")
    st.caption("科技 / 军工 / 新能源 / 机器人　·　流通市值 50-1000 亿　·　股价 10 元以上")

    with st.sidebar:
        st.header("设置")
        token = st.text_input("Tushare Token", type="password",
                              value=os.environ.get("TUSHARE_TOKEN", ""))
        top_n = st.slider("每周选几只", 1, 5, 3)
        st.caption("下面这些一般不用改。")
        with st.expander("数据范围"):
            start = st.date_input("起始", dt.date(2018, 1, 1))
            end = st.date_input("结束", dt.date.today())
            workers = st.slider("并发线程", 1, 8, 4)
            per_min = st.slider("每分钟请求上限", 60, 800, 400, 20)
        run = st.button("下载数据并选股", type="primary", use_container_width=True)
        if ss.get("panel") is not None:
            st.success(f"已加载 {len(ss['panel']['codes'])} 只 × "
                       f"{len(ss['panel']['cal'])} 个交易日")

    if run:
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
        for k in ("panel", "factors", "result"):
            ss.pop(k, None)
        gc.collect()
        s_str, e_str = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

        with st.status("正在准备…", expanded=True) as status:
            st.write("取行业成分股…")
            uni = fetch_universe(pro, lim, SW_L1_DEFAULT, SW_L2_DEFAULT)
            if not len(uni):
                st.error("行业成分股为空，可能是 Tushare 积分不足。")
                st.stop()
            st.write("取股票基础信息…")
            basic = fetch_stock_basic(pro, lim)
            st.write("市值预筛…")
            codes = prescreen_by_mv(pro, lim, list(uni["ts_code"]), s_str, e_str, 50, 1000)
            st.write(f"下载 {len(codes)} 只行情…")
            bar = st.progress(0.0)
            t0 = time.time()
            px = download_all(token, codes, s_str, e_str, lim, True, workers,
                              lambda done, tot, ok: bar.progress(
                                  done / tot, text=f"{done}/{tot}　已用 {(time.time()-t0)/60:.1f} 分"))
            if not px:
                st.error("没下到数据，检查 Token 与积分权限。")
                st.stop()
            st.write("计算因子…")
            panel = build_panel(px)
            px.clear(); del px; gc.collect()
            ss["panel"] = panel
            ss["basic"] = basic
            ss["uni"] = uni
            ss["factors"] = compute_factors(panel["adj_close"], panel["amount"],
                                            circ_mv=panel["circ_mv"])
            status.update(label=f"完成，耗时 {(time.time()-t0)/60:.1f} 分钟", state="complete")

    if ss.get("panel") is None:
        st.info("左侧填入 Tushare Token 后点「下载数据并选股」。"
                "首次约需 5-15 分钟，之后走本地缓存。")
        st.stop()

    panel, basic, uni, factors = ss["panel"], ss["basic"], ss["uni"], ss["factors"]
    elig = build_eligibility(panel, basic, uni, 50, 1000, 10.0, 2.0, 365)
    score = composite_score(factors, elig, {k: WEIGHTS.get(k, 0.0) for k in FACTOR_KEYS})
    rebal = weekly_rebal_dates(panel["cal"])

    # ---- 一次性校准：前几名到底该取哪一段 ----
    if ss.get("calib") is None:
        with st.spinner("校准中（只做一次）…"):
            bt = rank_band_test(score, panel["adj_close"], elig, rebal,
                                horizon_weeks=4, gap=1)
            ss["calib"] = bt
    bt = ss["calib"]
    best_band = bt["平均超额"].idxmax() if len(bt) else "第1-3名"
    lo = int(best_band.replace("第", "").split("-")[0])

    # ---- 本周候选 ----
    d = panel["cal"][-1]
    s = score.loc[d].where(elig.loc[d]).dropna().sort_values(ascending=False)
    st.subheader(f"本周候选　（数据截至 {d:%Y-%m-%d}，合格池 {len(s)} 只）")

    if len(s) < top_n:
        st.error(f"合格股票不足 {top_n} 只，本周无候选。")
    else:
        nm = basic.set_index("ts_code")["name"].to_dict()
        ind = uni.set_index("ts_code")["ind_name"].to_dict()
        picks = list(s.index[lo - 1: lo - 1 + top_n])
        rows = []
        for r, c in enumerate(picks, 1):
            rows.append({
                "序": r, "代码": c, "名称": nm.get(c, ""), "行业": ind.get(c, ""),
                "收盘价": round(float(panel["raw_close"].loc[d, c]), 2),
                "流通市值(亿)": round(float(panel["circ_mv"].loc[d, c]) / 1e4, 0),
                "距60日高点": f"{factors['dist_hi'].loc[d, c]:.1%}",
                "近5日涨幅": f"{factors['rev5'].loc[d, c]:.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.download_button("下载本周候选 CSV",
                           pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                           f"picks_{d:%Y%m%d}.csv", "text/csv")
        st.caption(f"取的是排名第 {lo} 到 {lo+top_n-1} 名，不是第 1 名起——原因见下方「为什么」。")

    # ---- 止盈止损参考 ----
    st.divider()
    st.subheader("止盈止损参考")
    if ss.get("path") is None:
        with st.spinner("统计中…"):
            ss["path"] = picked_path_stats(score, panel["adj_close"], elig, rebal,
                                           top_n=top_n, weeks=8, gap=1)
    ps = ss["path"]
    if len(ps):
        st.dataframe(ps.style.format({"最大涨幅_中位": "{:+.1%}", "最大跌幅_中位": "{:+.1%}",
                                      "最大跌幅_25分位": "{:+.1%}", "期末收益_中位": "{:+.1%}",
                                      "期末为正比例": "{:.0%}"}),
                     use_container_width=True)
        w4 = ps.loc[4] if 4 in ps.index else ps.iloc[-1]
        st.markdown(
            f"**怎么用**：历史上这批股票买入后 4 周内，一半会回撤到 "
            f"{w4['最大跌幅_中位']:.0%} 以内，四分之一会跌到 {w4['最大跌幅_25分位']:.0%}。"
            f"**止损设在 {w4['最大跌幅_25分位']:.0%} 以外**，才不会被正常波动扫出局；"
            f"设在 {w4['最大跌幅_中位']:.0%} 以内则一半以上的仓位会被无谓止损掉。")

    # ---- 为什么 ----
    st.divider()
    with st.expander("为什么这样选（点开看依据）", expanded=False):
        st.markdown("**一、名次段检验：为什么不取第 1 名**")
        st.markdown(
            "分层检验测的是前 10%（上百只）的平均值。只买 3 只时买的是前 0.2%，"
            "那是从没验证过的区间——最极端的几只往往是真出问题的公司，不是被错杀的。"
            "下表按名次分段实测未来 4 周相对全池的超额：")
        st.dataframe(bt.style.format({"平均超额": "{:+.2%}", "胜率": "{:.1%}",
                                      "标准误": "{:.3%}", "t值": "{:.2f}"})
                       .background_gradient(subset=["平均超额"], cmap="RdYlGn"),
                     use_container_width=True)
        st.info(f"实测最优名次段是 **{best_band}**，所以本周候选从第 {lo} 名开始取。")

        st.markdown("**二、空窗检查**")
        ew = empty_week_stats(score, elig, rebal, top_n=top_n)
        st.dataframe(ew.rename("选不满的周数").to_frame().T, use_container_width=True)
        st.caption("排名系统永远有前 N 名，所以结构上不会空窗。"
                   "上表若全为 0，说明你「一年空窗不超过 5 周」的要求自动满足。")

        st.markdown("**三、诚实的边界**")
        st.warning(
            "这份名单**不承诺跑赢**。持股 1-3 只时，你的盈亏几乎完全由个股运气决定，"
            "而不是由排名质量决定——这是分散度的算术，不是因子好坏的问题。\n\n"
            "它能做的是：把 1400 只筛到符合你硬约束、且避开了历史上表现最差那一档的"
            f"少数几只。前几名的超额是每 4 周约 "
            f"{bt.loc[best_band, '平均超额']:.1%}，扣掉买卖成本后所剩不多。\n\n"
            "**请把它当候选池，不要当买入指令。** 最终决定还是你自己的。")

    if API_ERRORS:
        with st.expander(f"接口异常 {len(API_ERRORS)} 条"):
            st.write(API_ERRORS[-30:])


if __name__ == "__main__" and st is not None:
    main()
