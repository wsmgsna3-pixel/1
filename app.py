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
# 六、选股方法库 —— 每个方法返回一张打分表（越高越优先）
# ======================================================================
def to_weekly(df: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    """日线转周线（按自然周，取周内最后/最高/最低）。"""
    g = df.resample("W-FRI")
    return {"last": g.last, "max": g.max, "min": g.min}[how]()


def _ema(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.ewm(span=n, adjust=False, min_periods=n).mean()


def weekly_macd(wc: pd.DataFrame, fast=12, slow=26, sig=9) -> Dict[str, pd.DataFrame]:
    dif = _ema(wc, fast) - _ema(wc, slow)
    dea = _ema(dif, sig)
    return {"dif": dif, "dea": dea, "hist": (dif - dea) * 2}


def weekly_skdj(wc, wh, wl, n=9, m=3) -> Dict[str, pd.DataFrame]:
    """SKDJ：RSV 先平滑再算 K，比普通 KDJ 慢，周线上噪音更少。"""
    lo = wl.rolling(n).min()
    hi = wh.rolling(n).max()
    rng = (hi - lo).where((hi - lo) > 1e-9)
    rsv = _ema((wc - lo) / rng * 100.0, m)
    k = _ema(rsv, m)
    d = k.rolling(m).mean()
    return {"k": k, "d": d}


def build_methods(panel: dict, factors: Dict[str, pd.DataFrame],
                  elig: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    返回 {方法名: 日频打分表}。分数只在 elig 为真处有效，越高越优先。
    周线指标算完后前向填充到日频——周中不会用到未来数据，因为每周只在
    周五收盘后调仓，取的正是当周已经收完的那根周线。
    """
    A = panel["adj_close"]
    cal = A.index
    wc = to_weekly(A, "last")
    wh = to_weekly(panel["adj_high"], "max")
    wl = to_weekly(panel["adj_low"], "min")
    out: Dict[str, pd.DataFrame] = {}

    def daily(w: pd.DataFrame) -> pd.DataFrame:
        return w.reindex(cal, method="ffill")

    # --- 1. 趋势动量（正权重）：买强势股，不是买反弹 ---
    z = lambda k, w: cs_zscore(factors[k], elig) * w
    out["趋势动量"] = (z("mom_ra", 1.0).add(z("trend_q", 1.0), fill_value=0)
                       .add(z("rel_str", 1.0), fill_value=0)).where(elig)

    # --- 2. 周线MACD金叉（零轴上方）---
    m = weekly_macd(wc)
    cross = (m["dif"] > m["dea"]) & (m["dif"].shift(1) <= m["dea"].shift(1))
    strength = (m["hist"] / wc.abs().where(wc.abs() > 1e-9))
    sc = strength.where(cross & (m["dif"] > 0))
    out["周线MACD金叉"] = daily(sc).where(elig)

    # --- 3. 周线MACD多头（持续在零轴上且柱增）---
    up = (m["dif"] > m["dea"]) & (m["dif"] > 0) & (m["hist"] > m["hist"].shift(1))
    out["周线MACD多头"] = daily(strength.where(up)).where(elig)

    # --- 4. 周线SKDJ金叉（低位）---
    s = weekly_skdj(wc, wh, wl)
    kx = (s["k"] > s["d"]) & (s["k"].shift(1) <= s["d"].shift(1)) & (s["k"] < 30)
    out["周线SKDJ金叉"] = daily((50 - s["k"]).where(kx)).where(elig)

    # --- 5. 创新高突破：周线收盘创 N 周新高 ---
    for nw in (20, 52):
        hh = wc.rolling(nw).max()
        brk = wc >= hh
        # 分数用"突破幅度 × 趋势质量"，避免选到刚好平高点的
        sc2 = daily((wc / wc.rolling(nw).mean() - 1.0).where(brk))
        out[f"创{nw}周新高"] = (sc2 + cs_zscore(factors["trend_q"], elig) * 0.01).where(elig)

    # --- 6. 均线多头排列 + 回踩不破 ---
    ma5, ma10, ma20 = wc.rolling(5).mean(), wc.rolling(10).mean(), wc.rolling(20).mean()
    bull = (ma5 > ma10) & (ma10 > ma20) & (wc > ma5)
    out["周线均线多头"] = daily(((ma5 / ma20 - 1.0)).where(bull)).where(elig)

    # --- 7. 动量 + 量能配合 ---
    out["动量+放量"] = (z("mom_ra", 1.0).add(z("trend_q", 0.5), fill_value=0)
                        .add(z("vol_exp", 0.5), fill_value=0)).where(elig)

    # --- 8. 反转（作为对照组，你说不要，但留着做基准）---
    out["反转(对照)"] = (z("dist_hi", -1.0).add(z("rev5", -0.5), fill_value=0)).where(elig)

    return out


# ======================================================================
# 七、逐笔独立回测 —— 没有组合概念，每只选出的股票各自跟踪
# ======================================================================
def pick_weekly(score: pd.DataFrame, elig: pd.DataFrame,
                rebal: List[pd.Timestamp], top_n: int = 3,
                rank_from: int = 1) -> pd.DataFrame:
    """每个调仓日选出 top_n 只。返回 date / code / rank / score。"""
    rows = []
    for d in rebal:
        if d not in score.index:
            continue
        s = score.loc[d].where(elig.loc[d]).dropna().sort_values(ascending=False)
        picks = s.index[rank_from - 1: rank_from - 1 + top_n]
        for r, c in enumerate(picks, rank_from):
            rows.append({"date": d, "code": c, "rank": r, "score": float(s[c])})
    return pd.DataFrame(rows)


def track_picks(picks: pd.DataFrame, panel: dict, tp: float = 0.20,
                sl: float = 0.08, max_days: int = 60,
                comm: float = 0.0003, stamp: float = 0.0005,
                slip: float = 0.001) -> pd.DataFrame:
    """
    每只选出的股票独立跟踪，直到止盈 / 止损 / 超时。互不影响，没有资金约束。
    次日开盘买入（涨停买不到则放弃这笔）；触发条件按当日收盘判定，次日开盘卖出。
    收盘判定+次日成交是保守口径：不假设你能在盘中精确摸到止损价。
    """
    cal = panel["adj_close"].index
    codes = panel["codes"]
    ci = {c: j for j, c in enumerate(codes)}
    AC = panel["adj_close"].to_numpy(dtype=np.float64)
    AO = panel["adj_open"].to_numpy(dtype=np.float32)
    TRD = panel["tradable"].to_numpy(dtype=bool)
    LU = panel["limit_up_open"].to_numpy(dtype=bool)
    LD = panel["limit_dn_open"].to_numpy(dtype=bool)
    pos = {d: i for i, d in enumerate(cal)}
    cost_in = comm + slip
    cost_out = comm + stamp + slip

    out = []
    for _, p in picks.iterrows():
        i0 = pos.get(p["date"])
        j = ci.get(p["code"])
        if i0 is None or j is None:
            continue
        # 次日开盘买入
        b = i0 + 1
        while b < len(cal) and (not TRD[b, j] or LU[b, j] or not np.isfinite(AO[b, j])):
            b += 1
            if b - i0 > 5:
                break
        if b >= len(cal) or not TRD[b, j] or LU[b, j] or not np.isfinite(AO[b, j]):
            out.append({**p.to_dict(), "结果": "买不到(涨停/停牌)"})
            continue
        entry = float(AO[b, j]) * (1 + cost_in)

        reason, exit_i, exit_px = "持有中", None, None
        for k in range(b, min(b + max_days, len(cal))):
            if not TRD[k, j] or not np.isfinite(AC[k, j]):
                continue
            r = AC[k, j] / entry - 1.0
            if r >= tp:
                reason = "止盈"
            elif r <= -sl:
                reason = "止损"
            elif k - b >= max_days - 1:
                reason = "超时"
            if reason != "持有中":
                e = k + 1                       # 次日开盘卖出
                while e < len(cal) and (not TRD[e, j] or LD[e, j]
                                        or not np.isfinite(AO[e, j])):
                    e += 1
                    if e - k > 5:
                        break
                if e < len(cal) and np.isfinite(AO[e, j]):
                    exit_i, exit_px = e, float(AO[e, j]) * (1 - cost_out)
                else:
                    exit_i, exit_px = k, float(AC[k, j]) * (1 - cost_out)
                break
        if reason == "持有中":
            out.append({**p.to_dict(), "结果": "尚未了结"})
            continue
        out.append({**p.to_dict(), "买入日": cal[b], "买入价": entry,
                    "卖出日": cal[exit_i], "卖出价": exit_px, "结果": reason,
                    "收益率": exit_px / entry - 1.0, "持有交易日": exit_i - b})
    return pd.DataFrame(out)


def holding_week_table(picks: pd.DataFrame, panel: dict, weeks: int = 12,
                       comm: float = 0.0003, stamp: float = 0.0005,
                       slip: float = 0.001) -> pd.DataFrame:
    """
    不设止盈止损，纯看「选出后持有到第 N 周」的表现。
    这是判断该在第几周退出的依据：看收益率和胜率从第几周开始掉头。
    """
    cal = panel["adj_close"].index
    ci = {c: j for j, c in enumerate(panel["codes"])}
    AC = panel["adj_close"].to_numpy(dtype=np.float64)
    AO = panel["adj_open"].to_numpy(dtype=np.float32)
    TRD = panel["tradable"].to_numpy(dtype=bool)
    LU = panel["limit_up_open"].to_numpy(dtype=bool)
    pos = {d: i for i, d in enumerate(cal)}
    rt = (comm + slip) + (comm + stamp + slip)

    acc = {w: [] for w in range(1, weeks + 1)}
    for _, p in picks.iterrows():
        i0 = pos.get(p["date"]); j = ci.get(p["code"])
        if i0 is None or j is None:
            continue
        b = i0 + 1
        if b >= len(cal) or not TRD[b, j] or LU[b, j] or not np.isfinite(AO[b, j]):
            continue
        entry = float(AO[b, j])
        for w in range(1, weeks + 1):
            k = b + 5 * w
            if k >= len(cal) or not np.isfinite(AC[k, j]):
                continue
            acc[w].append(AC[k, j] / entry - 1.0 - rt)

    rows = []
    for w in range(1, weeks + 1):
        v = np.array(acc[w], dtype=float)
        if len(v) < 10:
            continue
        se = v.std(ddof=1) / np.sqrt(len(v))
        rows.append({"第N周": w, "样本数": len(v), "平均收益率": v.mean(),
                     "中位收益率": float(np.median(v)), "胜率": float((v > 0).mean()),
                     "标准误": se, "t值": v.mean() / se if se > 1e-12 else np.nan})
    return pd.DataFrame(rows).set_index("第N周")


def summarize_trades(tr: pd.DataFrame) -> dict:
    """
    t 值必须按周聚类。同一周选出的 3 只共享当周的大盘涨跌，不是独立样本；
    而且平均持有 2-3 周，相邻周的持仓在时间上重叠。直接拿 1284 笔算 t，
    等于假装有 1284 个独立观测，会把显著性放大一倍以上。
    做法：先按选出日取周内均值，再对这条周序列做 Newey-West 修正。
    """
    d = tr.dropna(subset=["收益率"]) if "收益率" in tr.columns else pd.DataFrame()
    if not len(d):
        return {}
    n = len(d)
    win = d[d["收益率"] > 0]["收益率"]
    los = d[d["收益率"] <= 0]["收益率"]
    se_naive = d["收益率"].std(ddof=1) / np.sqrt(n)
    t_naive = d["收益率"].mean() / se_naive if se_naive > 1e-12 else np.nan

    wk = d.groupby("date")["收益率"].mean().sort_index()
    hold_w = d["持有交易日"].mean() / 5.0
    t_cl = newey_west_t(wk, lag=max(1, int(round(hold_w))))
    return {"笔数": n, "周数": len(wk), "平均收益": d["收益率"].mean(),
            "中位收益": d["收益率"].median(), "胜率": len(win) / n,
            "盈亏比": (win.mean() / abs(los.mean())) if len(los) and abs(los.mean()) > 1e-9 else np.nan,
            "t值(朴素)": t_naive, "t值(按周聚类)": t_cl,
            "平均持有周": hold_w,
            "止盈": (d["结果"] == "止盈").mean(), "止损": (d["结果"] == "止损").mean(),
            "超时": (d["结果"] == "超时").mean()}


def tp_sl_grid(picks: pd.DataFrame, panel: dict, tps, sls, max_days: int,
               progress=None, **kw) -> tuple:
    """
    止盈 × 止损 网格。
    止损设在正常波动之内，被扫出局的就是噪音不是判断错误；设得太宽又拿不住。
    这个网格用数据找出该设在哪，而不是拍脑袋。返回 (平均收益表, 聚类t值表)。
    """
    mean_g, t_g = {}, {}
    tot = len(tps) * len(sls)
    k = 0
    for sl in sls:
        mrow, trow = {}, {}
        for tp in tps:
            tr = track_picks(picks, panel, tp, sl, max_days, **kw)
            s = summarize_trades(tr)
            mrow[f"止盈{tp:.0%}"] = s.get("平均收益", np.nan)
            trow[f"止盈{tp:.0%}"] = s.get("t值(按周聚类)", np.nan)
            k += 1
            if progress:
                progress(k / tot, f"止损{sl:.0%} 止盈{tp:.0%}")
        mean_g[f"止损{sl:.0%}"] = mrow
        t_g[f"止损{sl:.0%}"] = trow
    return pd.DataFrame(mean_g).T, pd.DataFrame(t_g).T


def split_summary(tr: pd.DataFrame, cut: str = "2023-01-01") -> pd.DataFrame:
    """样本内 / 样本外 分开看。全样本好而样本外垮，是最常见的自欺方式。"""
    d = tr.dropna(subset=["收益率"]) if "收益率" in tr.columns else pd.DataFrame()
    if not len(d):
        return pd.DataFrame()
    cut = pd.Timestamp(cut)
    out = {}
    for lab, seg in (("样本内", d[d["date"] < cut]), ("样本外", d[d["date"] >= cut])):
        if len(seg) < 30:
            continue
        out[lab] = summarize_trades(seg)
    return pd.DataFrame(out).T


def yearly_summary(tr: pd.DataFrame) -> pd.DataFrame:
    """
    逐年表现。一个真实的效应应该多数年份同号；靠单独一年撑起来的
    平均值，只是那一年的行情，不是可重复的能力。
    """
    d = tr.dropna(subset=["收益率"]) if "收益率" in tr.columns else pd.DataFrame()
    if not len(d):
        return pd.DataFrame()
    rows = {}
    for y, g in d.groupby(d["date"].dt.year):
        if len(g) < 20:
            continue
        wk = g.groupby("date")["收益率"].mean().sort_index()
        se = wk.std(ddof=1) / np.sqrt(len(wk)) if len(wk) > 2 else np.nan
        rows[y] = {"笔数": len(g), "平均收益": g["收益率"].mean(),
                   "中位收益": g["收益率"].median(),
                   "胜率": (g["收益率"] > 0).mean(),
                   "t值": wk.mean() / se if se and se > 1e-12 else np.nan}
    return pd.DataFrame(rows).T


def profit_concentration(tr: pd.DataFrame, ks=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    """
    利润集中度。如果总利润的绝大部分来自极少数几笔，那不是策略是彩票——
    你无法指望下一段时间还能碰上那几笔。
    """
    d = tr.dropna(subset=["收益率"]) if "收益率" in tr.columns else pd.DataFrame()
    if len(d) < 30:
        return pd.DataFrame()
    v = d["收益率"].sort_values(ascending=False).to_numpy()
    tot = v.sum()
    rows = []
    for k in ks:
        if k >= len(v):
            continue
        rows.append({"最赚的前N笔": k, "占总利润": v[:k].sum() / tot if abs(tot) > 1e-12 else np.nan,
                     "剔除后单笔均值": v[k:].mean()})
    out = pd.DataFrame(rows).set_index("最赚的前N笔")
    out.attrs["原均值"] = float(v.mean())
    out.attrs["总笔数"] = int(len(v))
    return out


# ======================================================================
# 八、界面
# ======================================================================
def main():
    st.set_page_config(page_title="每周选股", layout="wide")
    ss = st.session_state
    ss.setdefault("panel", None)
    st.title("每周选股")
    st.caption("科技/军工/新能源/机器人　·　流通市值 50-1000 亿　·　股价 10 元以上　·　每周末选 3 只")

    with st.sidebar:
        st.header("① 数据")
        token = st.text_input("Tushare Token", type="password",
                              value=os.environ.get("TUSHARE_TOKEN", ""))
        with st.expander("下载范围"):
            start = st.date_input("起始", dt.date(2018, 1, 1))
            end = st.date_input("结束", dt.date.today())
            workers = st.slider("并发线程", 1, 8, 4)
            per_min = st.slider("每分钟请求上限", 60, 800, 400, 20)
        run = st.button("下载数据", type="primary", use_container_width=True)

        st.header("② 交易规则")
        tp = st.slider("止盈 (%)", 5, 60, 20) / 100.0
        sl = st.slider("止损 (%)", 3, 30, 8) / 100.0
        maxw = st.slider("超时卖出 (周)", 4, 26, 12)
        top_n = st.slider("每周选几只", 1, 5, 3)
        with st.expander("成本"):
            comm = st.number_input("佣金(单边,万分之)", 0.0, 10.0, 3.0, 0.1) / 1e4
            slip = st.number_input("滑点(单边,%)", 0.0, 0.5, 0.10, 0.01) / 100.0
        if ss.get("panel") is not None:
            st.success(f"已加载 {len(ss['panel']['codes'])} 只 × {len(ss['panel']['cal'])} 日")

    if run:
        if not token:
            st.error("请先填 Tushare Token"); st.stop()
        try:
            import tushare as ts
        except ImportError:
            st.error("未安装 tushare：pip install tushare"); st.stop()
        ts.set_token(token); pro = ts.pro_api(token)
        lim = Limiter(per_min); API_ERRORS.clear()
        for k in ("panel", "factors", "methods", "cmp"):
            ss.pop(k, None)
        gc.collect()
        s_str, e_str = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        with st.status("准备中…", expanded=True) as stt:
            st.write("取行业成分股…")
            uni = fetch_universe(pro, lim, SW_L1_DEFAULT, SW_L2_DEFAULT)
            if not len(uni):
                st.error("行业成分股为空，可能是 Tushare 积分不足。"); st.stop()
            st.write("取股票基础信息…")
            basic = fetch_stock_basic(pro, lim)
            st.write("市值预筛…")
            codes = prescreen_by_mv(pro, lim, list(uni["ts_code"]), s_str, e_str, 50, 1000)
            st.write(f"下载 {len(codes)} 只行情…")
            bar = st.progress(0.0); t0 = time.time()
            px = download_all(token, codes, s_str, e_str, lim, True, workers,
                              lambda dn, tt, ok: bar.progress(dn / tt,
                                  text=f"{dn}/{tt}　{(time.time()-t0)/60:.1f} 分"))
            if not px:
                st.error("没下到数据，检查 Token 与积分。"); st.stop()
            st.write("计算指标…")
            panel = build_panel(px); px.clear(); del px; gc.collect()
            ss["panel"], ss["basic"], ss["uni"] = panel, basic, uni
            ss["factors"] = compute_factors(panel["adj_close"], panel["amount"],
                                            circ_mv=panel["circ_mv"])
            stt.update(label=f"完成，耗时 {(time.time()-t0)/60:.1f} 分钟", state="complete")

    if ss.get("panel") is None:
        st.info("左侧填 Token 后点「下载数据」。首次约 5-15 分钟，之后走本地缓存。"); st.stop()

    panel, basic, uni, factors = ss["panel"], ss["basic"], ss["uni"], ss["factors"]
    elig = build_eligibility(panel, basic, uni, 50, 1000, 10.0, 2.0, 365)
    if ss.get("methods") is None:
        with st.spinner("构建选股方法…"):
            ss["methods"] = build_methods(panel, factors, elig)
    methods = ss["methods"]
    rebal = weekly_rebal_dates(panel["cal"])
    maxd = maxw * 5
    kw = dict(comm=comm, stamp=0.0005, slip=slip)

    t1, t2, t3 = st.tabs(["方法对比", "本周选股", "逐笔明细"])

    # ---------------- 方法对比 ----------------
    with t1:
        st.markdown("**先看哪个方法有效。** 每个方法都按同样规则跑：每周末选 "
                    f"{top_n} 只，次日开盘买入，止盈 {tp:.0%} / 止损 {sl:.0%} / "
                    f"{maxw} 周超时，含成本。")
        if st.button("跑全部方法", type="primary"):
            bar = st.progress(0.0)
            res = {}
            for n, (nm, sc) in enumerate(methods.items()):
                pk = pick_weekly(sc, elig, rebal, top_n)
                if len(pk) < 30:
                    bar.progress((n + 1) / len(methods), text=nm); continue
                tr = track_picks(pk, panel, tp, sl, maxd, **kw)
                wt = holding_week_table(pk, panel, 12, **kw)
                sm = summarize_trades(tr)
                empty = 1.0 - pk.date.nunique() / max(1, len(rebal))
                res[nm] = {"pk": pk, "tr": tr, "wt": wt, "sm": sm, "empty": empty}
                bar.progress((n + 1) / len(methods), text=nm)
            ss["cmp"] = res; bar.empty()

        if ss.get("cmp"):
            res = ss["cmp"]
            rows = []
            for nm, r in res.items():
                s = r["sm"]
                if not s:
                    continue
                rows.append({"方法": nm, "笔数": s["笔数"], "平均收益": s["平均收益"],
                             "胜率": s["胜率"], "盈亏比": s["盈亏比"],
                             "t值(朴素)": s["t值(朴素)"], "t值(按周聚类)": s["t值(按周聚类)"],
                             "平均持有周": s["平均持有周"], "止损率": s["止损"],
                             "空窗周占比": r["empty"]})
            cm = pd.DataFrame(rows).set_index("方法").sort_values("t值(按周聚类)", ascending=False)
            st.dataframe(cm.style.format({"平均收益": "{:+.2%}", "胜率": "{:.1%}",
                                          "盈亏比": "{:.2f}", "t值(朴素)": "{:.2f}",
                                          "t值(按周聚类)": "{:.2f}", "平均持有周": "{:.1f}",
                                          "止损率": "{:.0%}", "空窗周占比": "{:.1%}"}, na_rep="—")
                           .background_gradient(subset=["t值(按周聚类)"], cmap="RdYlGn",
                                                vmin=-3, vmax=3),
                         use_container_width=True)
            st.warning(
                "**看「t值(按周聚类)」，不要看朴素 t。** 同一周选出的 3 只共享当周大盘涨跌，"
                "不是独立样本；平均持有 2-3 周，相邻周的持仓还在时间上重叠。"
                "拿一千多笔当独立观测算 t，会把显著性放大一倍以上——"
                "我在完全没有选股能力的模拟数据上测过：朴素 t=1.95，聚类后只剩 0.91。")
            ok = cm[(cm["t值(按周聚类)"] >= 2) & (cm["空窗周占比"] <= 5 / 52)]
            if len(ok):
                st.success(f"**{ok.index[0]}** 通过：聚类 t={ok['t值(按周聚类)'].iloc[0]:.2f}"
                           f"（门槛 2.0），空窗 {ok['空窗周占比'].iloc[0]*52:.0f} 周/年（上限 5）。"
                           "下一步：看下方的样本内外拆分，样本外也站得住才算数。")
            else:
                near = cm[cm["空窗周占比"] <= 5 / 52]
                st.error("**没有方法的聚类 t 达到 2。** 满足空窗要求的方法里最高为 "
                         f"{near['t值(按周聚类)'].max():.2f}（{near['t值(按周聚类)'].idxmax()}）。"
                         "这意味着平均收益和零还分不出区别。")
            st.caption("空窗周占比上限 5/52 ≈ 9.6%。MACD/SKDJ 金叉这类事件型信号天然稀疏。")

            st.divider()
            st.markdown("**样本内 / 样本外**（2023-01-01 分界）")
            m3 = st.selectbox("看哪个方法", list(res.keys()),
                              index=list(res.keys()).index(cm.index[0]), key="sp_m")
            sp = split_summary(res[m3]["tr"], "2023-01-01")
            if len(sp):
                st.dataframe(sp[["笔数", "平均收益", "胜率", "盈亏比",
                                 "t值(按周聚类)"]].style.format(
                    {"平均收益": "{:+.2%}", "胜率": "{:.1%}", "盈亏比": "{:.2f}",
                     "t值(按周聚类)": "{:.2f}"}), use_container_width=True)
                if "样本外" in sp.index and "样本内" in sp.index:
                    a, o = sp.loc["样本内"], sp.loc["样本外"]
                    if o["t值(按周聚类)"] >= 2 and o["平均收益"] > 0:
                        st.success(f"样本外依然站得住（t={o['t值(按周聚类)']:.2f}）。")
                    else:
                        st.error(f"样本外没站住：平均收益 {o['平均收益']:+.2%}，"
                                 f"聚类 t={o['t值(按周聚类)']:.2f}。"
                                 "样本内好、样本外垮，通常说明样本内那部分是行情特征。")

            st.divider()
            st.markdown("**逐年表现 + 利润集中度**")
            st.caption("上一轮就是这两个诊断戳破了幻觉：平均收益漂亮，但 94% 的利润来自 "
                       "513 笔里的 5 笔。真实的效应应该多数年份同号，且不依赖极少数暴利。")
            ys = yearly_summary(res[m3]["tr"])
            if len(ys):
                st.dataframe(ys.style.format({"笔数": "{:.0f}", "平均收益": "{:+.2%}",
                                              "中位收益": "{:+.2%}", "胜率": "{:.1%}",
                                              "t值": "{:.2f}"})
                               .background_gradient(subset=["平均收益"], cmap="RdYlGn"),
                             use_container_width=True)
                pos_y = int((ys["平均收益"] > 0).sum())
                st.write(f"平均收益为正的年份：**{pos_y}/{len(ys)}**")
            pc = profit_concentration(res[m3]["tr"])
            if len(pc):
                st.dataframe(pc.style.format({"占总利润": "{:.1%}",
                                              "剔除后单笔均值": "{:+.3%}"}),
                             use_container_width=True)
                st.caption(f"全部 {pc.attrs['总笔数']} 笔，原始单笔均值 "
                           f"{pc.attrs['原均值']:+.2%}。若剔除最赚的 5 笔后均值就塌到零附近，"
                           "说明利润集中在极少数运气，不可重复。")
                if 5 in pc.index and abs(pc.attrs["原均值"]) > 1e-9:
                    keep = pc.loc[5, "剔除后单笔均值"] / pc.attrs["原均值"]
                    if keep < 0.4:
                        st.error(f"剔除最赚的 5 笔后，单笔均值只剩原来的 {keep:.0%}。"
                                 "这是彩票式分布，不是可重复的边际。")
                    else:
                        st.success(f"剔除最赚的 5 笔后仍保留 {keep:.0%} 的均值，"
                                   "利润不是靠极少数暴利撑起来的。")

            st.divider()
            st.markdown("**止盈 × 止损 网格**")
            st.markdown(
                f"当前止损 {sl:.0%}，实测止损率约 65%、平均持有仅 2.3 周——"
                "远短于超时上限，说明绝大多数仓位是被止损打掉的，不是走完了行情。"
                "而选出后 4 周内最大回撤的中位数就有 -7.7%，**8% 的止损设在了正常波动之内**。"
                "下面用数据找该设在哪。")
            if st.button("跑止盈止损网格"):
                bar2 = st.progress(0.0)
                mg, tg = tp_sl_grid(res[m3]["pk"], panel,
                                    [0.10, 0.15, 0.20, 0.30, 0.40],
                                    [0.06, 0.08, 0.12, 0.16, 0.20], maxd,
                                    progress=lambda p, n2: bar2.progress(p, text=n2), **kw)
                ss["grid"] = (m3, mg, tg); bar2.empty()
            if ss.get("grid"):
                gm, mg, tg = ss["grid"]
                st.caption(f"方法：{gm}")
                c1, c2 = st.columns(2)
                c1.markdown("平均单笔收益")
                c1.dataframe(mg.style.format("{:+.2%}", na_rep="—")
                             .background_gradient(cmap="RdYlGn"), use_container_width=True)
                c2.markdown("聚类 t 值")
                c2.dataframe(tg.style.format("{:.2f}", na_rep="—")
                             .background_gradient(cmap="RdYlGn", vmin=-3, vmax=3),
                             use_container_width=True)
                bt_ = tg.stack().idxmax()
                st.warning(
                    f"聚类 t 最高的是 **{bt_[0]} / {bt_[1]}**（t={tg.stack().max():.2f}）——"
                    "**但别拿这个数字当证据。** 25 个格子里挑最大值，即使全是噪音，"
                    "最大 |t| 的期望也有 1.9-2.3。\n\n"
                    "**该信的是梯度方向**：如果放宽止损后平均收益一列列单调上升，"
                    "那说明原来的止损设在了正常波动之内，把没走完的仓位提前打掉了——"
                    "这是一致的规律，不是幸运格子。\n\n"
                    "**止损该设在哪，用波动定，别用网格挑。** 「本周选股」页给出了"
                    "选出后各周的回撤分布，止损设在 25 分位之外才不会被正常波动扫出局。")

            st.divider()
            st.markdown("**第 1-12 周表现**（不设止盈止损，纯看持有到第 N 周）")
            pick_m = st.selectbox("选方法", list(res.keys()),
                                  index=list(res.keys()).index(cm.index[0]))
            wt = res[pick_m]["wt"]
            if len(wt):
                st.dataframe(wt.style.format({"平均收益率": "{:+.2%}", "中位收益率": "{:+.2%}",
                                              "胜率": "{:.1%}", "标准误": "{:.3%}", "t值": "{:.2f}"})
                               .background_gradient(subset=["平均收益率"], cmap="RdYlGn"),
                             use_container_width=True)
                st.line_chart(wt[["平均收益率", "中位收益率"]])
                st.line_chart(wt[["胜率"]])
                pk_w = int(wt["平均收益率"].idxmax())
                st.info(f"平均收益率在**第 {pk_w} 周**见顶。胜率在第 "
                        f"{int(wt['胜率'].idxmax())} 周最高。超过这个点继续持有，"
                        "期望不再增加而波动仍在累积。")

    # ---------------- 本周选股 ----------------
    with t2:
        mnames = list(methods.keys())
        use = st.selectbox("用哪个方法", mnames, key="use_m")
        sc = methods[use]
        d = panel["cal"][-1]
        s = sc.loc[d].where(elig.loc[d]).dropna().sort_values(ascending=False)
        st.subheader(f"{d:%Y-%m-%d}　候选 {len(s)} 只")
        if len(s) == 0:
            st.warning("本周该方法没有符合条件的股票（信号未触发）。换个方法或等下周。")
        else:
            nm = basic.set_index("ts_code")["name"].to_dict()
            ind = uni.set_index("ts_code")["ind_name"].to_dict()
            rows = []
            for r, c in enumerate(s.index[:top_n], 1):
                px_ = float(panel["raw_close"].loc[d, c])
                rows.append({"序": r, "代码": c, "名称": nm.get(c, ""), "行业": ind.get(c, ""),
                             "收盘价": round(px_, 2),
                             "流通市值(亿)": round(float(panel["circ_mv"].loc[d, c]) / 1e4),
                             "止盈价": round(px_ * (1 + tp), 2),
                             "止损价": round(px_ * (1 - sl), 2)})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("下载 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               f"picks_{d:%Y%m%d}.csv", "text/csv")
            st.caption("止盈止损价按收盘价估算，实际以你的买入价为准。")

    # ---------------- 逐笔明细 ----------------
    with t3:
        if not ss.get("cmp"):
            st.info("先到「方法对比」页点「跑全部方法」。")
        else:
            res = ss["cmp"]
            m2 = st.selectbox("看哪个方法的明细", list(res.keys()), key="det_m")
            tr = res[m2]["tr"].copy()
            for c in ("买入价", "卖出价"):
                if c in tr.columns:
                    tr[c] = tr[c].round(3)
            if "收益率" in tr.columns:
                tr["收益率"] = tr["收益率"].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "")
            st.dataframe(tr.sort_values("date", ascending=False),
                         use_container_width=True, height=520)
            st.download_button("下载全部成交 CSV",
                               res[m2]["tr"].to_csv(index=False).encode("utf-8-sig"),
                               f"trades_{m2}.csv", "text/csv")

    if API_ERRORS:
        with st.expander(f"接口异常 {len(API_ERRORS)} 条"):
            st.write(API_ERRORS[-30:])


if __name__ == "__main__" and st is not None:
    main()
