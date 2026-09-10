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
# SKDJ 选股 —— 日线金叉 + 加速确认 + 周线形态过滤
# ======================================================================
def _ema_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.ewm(span=n, adjust=False, min_periods=n).mean()


def skdj(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
         n: int = 9, m: int = 3):
    """
    SKDJ（慢速KD）：
        RSV = EMA((C - LLV(L,n)) / (HHV(H,n) - LLV(L,n)) * 100, m)
        K   = EMA(RSV, m)
        D   = MA(K, m)
    """
    lo = low.rolling(n).min()
    hi = high.rolling(n).max()
    rng = (hi - lo).where((hi - lo) > 1e-9)
    rsv = _ema_df((close - lo) / rng * 100.0, m)
    k = _ema_df(rsv, m)
    d = k.rolling(m).mean()
    return k, d


def weekly_skdj_state(panel: dict, n: int = 9, m: int = 3):
    """
    周线 SKDJ 状态，前向填充到日频。返回：
      wk_above : 周线 K > D（已金叉状态）
      wk_since : 距最近一次周线死叉过了几周（从未死叉记为 999）
    周线只用当周已收完的那根，不会用到未来数据。
    """
    wc = panel["adj_close"].resample("W-FRI").last()
    wh = panel["adj_high"].resample("W-FRI").max()
    wl = panel["adj_low"].resample("W-FRI").min()
    k, d = skdj(wc, wh, wl, n, m)
    above = k > d
    dead = (k < d) & (k.shift(1) >= d.shift(1))          # 周线死叉

    idx = np.arange(len(dead.index), dtype=float)
    marker = pd.DataFrame(np.where(dead.to_numpy(), idx[:, None], np.nan),
                          index=dead.index, columns=dead.columns).ffill()
    since = pd.DataFrame(idx[:, None] - marker.to_numpy(),
                         index=dead.index, columns=dead.columns).fillna(999.0)

    cal = panel["adj_close"].index
    return (above.reindex(cal, method="ffill").fillna(False),
            since.reindex(cal, method="ffill").fillna(999.0))


BUY_MODES = {
    "A_金叉次日": 0,      # 不确认，D+1 开盘买
    "B_确认1天": 1,       # D+1 收盘确认扩大，D+2 开盘买
    "C_确认2天": 2,       # D+1、D+2 连续扩大，D+3 开盘买
}
WK_FILTERS = ["F0_不过滤", "F1_距死叉≥10周", "F2_周线已金叉"]


def skdj_picks(panel: dict, elig: pd.DataFrame, n: int, buy_mode: str,
               wk_filter: str, k_max: float = 25.0, top_n: int = 3,
               wk_n: int = 9, min_weeks: int = 10) -> pd.DataFrame:
    """
    生成候选：日线 K 上穿 D 且 K<k_max → 按 buy_mode 决定确认天数与决策日 →
    决策日按 round(K-D, 2) 排序，并列时流通市值大者优先 → 取前 top_n。

    K-D 取两位小数，与交易软件显示精度一致；这样"数值相同"才真的会发生，
    市值打破平局这条规则才有意义（不取整的话浮点数几乎不可能精确相等）。
    """
    A, H, L = panel["adj_close"], panel["adj_high"], panel["adj_low"]
    k, d = skdj(A, H, L, n, 3)
    kd = k - d
    gold = (k > d) & (k.shift(1) <= d.shift(1)) & (k < k_max)

    conf = int(BUY_MODES[buy_mode])
    ok = gold.copy()
    for j in range(1, conf + 1):                        # 逐日确认 K-D 持续扩大
        ok &= (kd.shift(-j) > kd.shift(-(j - 1)))
    decide = ok.shift(conf).fillna(False)               # 决策日 = 金叉日 + conf

    if wk_filter != "F0_不过滤":
        wk_above, wk_since = weekly_skdj_state(panel, wk_n, 3)
        decide &= (wk_above if wk_filter == "F2_周线已金叉"
                   else (wk_since >= min_weeks))

    decide &= elig.reindex_like(decide).fillna(False)
    kd_r = kd.round(2)                                   # 交易软件显示精度
    cmv = panel["circ_mv"]
    cal = list(A.index)
    rows = []
    for i, dt_ in enumerate(cal):
        sel = decide.iloc[i]
        if not sel.any():
            continue
        codes = sel[sel].index
        sub = pd.DataFrame({"kd": kd_r.iloc[i].reindex(codes),
                            "mv": cmv.iloc[i].reindex(codes)}).dropna()
        if not len(sub):
            continue
        sub = sub.sort_values(["kd", "mv"], ascending=[False, False])
        for r, c in enumerate(sub.index[:top_n], 1):
            rows.append({"date": dt_, "code": c, "rank": r,
                         "kd": float(sub.loc[c, "kd"]),
                         "mv": float(sub.loc[c, "mv"])})
    return pd.DataFrame(rows)


def track_skdj(picks: pd.DataFrame, panel: dict, n: int, max_days: int = 30,
               comm: float = 0.0003, stamp: float = 0.0005,
               slip: float = 0.001) -> pd.DataFrame:
    """
    次日开盘买入（涨停买不到则放弃）；日线死叉次日开盘卖出，或满 max_days 超时。
    同一只股票在前一笔未了结前不重复建仓。
    """
    A, H, L = panel["adj_close"], panel["adj_high"], panel["adj_low"]
    k, d = skdj(A, H, L, n, 3)
    DEAD = ((k < d) & (k.shift(1) >= d.shift(1))).to_numpy(dtype=bool)

    cal = A.index
    ci = {c: j for j, c in enumerate(panel["codes"])}
    AO = panel["adj_open"].to_numpy(dtype=np.float32)
    TRD = panel["tradable"].to_numpy(dtype=bool)
    LU = panel["limit_up_open"].to_numpy(dtype=bool)
    LD = panel["limit_dn_open"].to_numpy(dtype=bool)
    pos = {dt_: i for i, dt_ in enumerate(cal)}
    cin, cout = comm + slip, comm + stamp + slip

    busy_until: Dict[str, int] = {}
    out = []
    for _, p in picks.sort_values("date").iterrows():
        i0 = pos.get(p["date"]); j = ci.get(p["code"])
        if i0 is None or j is None:
            continue
        if busy_until.get(p["code"], -1) > i0:          # 前一笔还没了结
            continue
        b = i0 + 1
        if b >= len(cal) or not TRD[b, j] or LU[b, j] or not np.isfinite(AO[b, j]):
            continue
        entry = float(AO[b, j]) * (1 + cin)

        reason, ex = "尚未了结", None
        for t in range(b, min(b + max_days, len(cal))):
            if DEAD[t, j] and t > b:
                reason = "死叉"
            elif t - b >= max_days - 1:
                reason = "超时"
            if reason != "尚未了结":
                e = t + 1
                while e < len(cal) and (not TRD[e, j] or LD[e, j]
                                        or not np.isfinite(AO[e, j])):
                    e += 1
                    if e - t > 5:
                        break
                ex = e if e < len(cal) and np.isfinite(AO[e, j]) else None
                break
        if ex is None:
            continue
        exit_px = float(AO[ex, j]) * (1 - cout)
        busy_until[p["code"]] = ex
        out.append({"date": p["date"], "code": p["code"], "rank": p["rank"],
                    "kd": p["kd"], "买入日": cal[b], "买入价": entry,
                    "卖出日": cal[ex], "卖出价": exit_px, "结果": reason,
                    "收益率": exit_px / entry - 1.0, "持有交易日": ex - b})
    return pd.DataFrame(out)


def random_control(panel: dict, elig: pd.DataFrame, dates, top_n: int = 3,
                   seed: int = 20260909) -> pd.DataFrame:
    """随机对照：同样每天选 top_n 只，纯随机。任何方法必须显著优于它。"""
    rng = np.random.default_rng(seed)
    rows = []
    for dt_ in dates:
        if dt_ not in elig.index:
            continue
        c = elig.loc[dt_]
        c = c[c].index
        if len(c) < top_n:
            continue
        for r, code in enumerate(rng.choice(c, top_n, replace=False), 1):
            rows.append({"date": dt_, "code": code, "rank": r, "kd": np.nan, "mv": np.nan})
    return pd.DataFrame(rows)


def empty_weeks_per_year(picks: pd.DataFrame, cal: pd.DatetimeIndex,
                         warmup: int = 130) -> pd.Series:
    """整周一只都没选出来的周数，按年汇总。"""
    c = cal[warmup:]
    allw = pd.Series(1, index=c).resample("W-FRI").size()
    if not len(picks):
        return pd.Series(len(allw), index=[c[0].year])
    hit = picks.groupby(pd.to_datetime(picks["date"])).size().resample("W-FRI").size()
    hit = hit.reindex(allw.index, fill_value=0)
    empty = (hit == 0)
    return empty.groupby(empty.index.year).sum()


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
# 界面 —— 一页，一个按钮
# ======================================================================
def _stats(tr: pd.DataFrame) -> dict:
    """按日聚类的 t 值：同一天选出的几只共享当天大盘涨跌，不是独立样本。"""
    d = tr.dropna(subset=["收益率"]) if len(tr) and "收益率" in tr.columns else pd.DataFrame()
    if len(d) < 30:
        return {}
    w = d[d["收益率"] > 0]["收益率"]
    l = d[d["收益率"] <= 0]["收益率"]
    day = d.groupby("date")["收益率"].mean().sort_index()
    hold = d["持有交易日"].mean()
    return {"笔数": len(d), "平均收益": d["收益率"].mean(), "中位收益": d["收益率"].median(),
            "胜率": len(w) / len(d),
            "盈亏比": (w.mean() / abs(l.mean())) if len(l) and abs(l.mean()) > 1e-9 else np.nan,
            "聚类t": newey_west_t(day, lag=max(1, int(round(hold)))),
            "平均持有日": hold, "死叉退出": (d["结果"] == "死叉").mean()}


def main():
    st.set_page_config(page_title="SKDJ 选股验证", layout="wide")
    ss = st.session_state
    ss.setdefault("panel", None)
    st.title("SKDJ 选股")
    st.caption("日线金叉(K<25) → 加速确认 → 周线形态过滤 → 死叉卖出")

    with st.sidebar:
        token = st.text_input("Tushare Token", type="password",
                              value=os.environ.get("TUSHARE_TOKEN", ""))
        top_n = st.slider("每天选几只", 1, 5, 3)
        maxd = st.slider("超时卖出(交易日)", 10, 60, 30)
        with st.expander("其他设置"):
            start = st.date_input("数据起始", dt.date(2018, 1, 1))
            end = st.date_input("数据结束", dt.date.today())
            k_max = st.slider("金叉时 K 的上限", 10, 50, 25)
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
        for kk in ("panel", "res"):
            ss.pop(kk, None)
        gc.collect()
        s_str, e_str = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        with st.status("下载中…", expanded=True) as stt:
            st.write("取行业成分股…")
            uni = fetch_universe(pro, lim, SW_L1_DEFAULT, SW_L2_DEFAULT)
            if not len(uni):
                st.error("行业成分股为空，可能是 Tushare 积分不足。"); st.stop()
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
    kw = dict(comm=comm, stamp=0.0005, slip=slip)

    tab1, tab2 = st.tabs(["回测结果", "今日候选"])

    with tab1:
        st.markdown(f"一次跑完 **2 种 N × 3 个买入时点 × 3 种周线过滤 = 18 个组合**，"
                    f"外加随机对照组。全部用同一套规则：每天选 {top_n} 只，"
                    f"次日开盘买，死叉次日开盘卖，{maxd} 日超时。")
        if st.button("运行全部组合", type="primary", use_container_width=True):
            combos = [(n, bm, wf) for n in (6, 9) for bm in BUY_MODES for wf in WK_FILTERS]
            bar = st.progress(0.0); rows = []; keep = {}
            for i, (n, bm, wf) in enumerate(combos):
                pk = skdj_picks(panel, elig, n, bm, wf, k_max, top_n)
                tr = track_skdj(pk, panel, n, maxd, **kw) if len(pk) else pd.DataFrame()
                s = _stats(tr)
                bar.progress((i + 1) / (len(combos) + 1), text=f"N={n} {bm} {wf}")
                if not s:
                    continue
                ew = empty_weeks_per_year(pk, panel["cal"])
                rows.append({"N": n, "买入时点": bm, "周线过滤": wf, **s,
                             "空窗周/年": float(ew.mean())})
                keep[f"N{n}|{bm}|{wf}"] = tr
            rc = random_control(panel, elig, list(panel["cal"][130:]), top_n)
            trc = track_skdj(rc.assign(kd=np.nan), panel, 9, maxd, **kw)
            sc = _stats(trc)
            bar.empty()
            ss["res"] = (pd.DataFrame(rows), sc, keep)

        if ss.get("res"):
            df, sc, keep = ss["res"]
            if sc:
                st.error(f"**随机对照组：平均收益 {sc['平均收益']:+.2%}，"
                         f"胜率 {sc['胜率']:.1%}，聚类 t = {sc['聚类t']:.2f}。**　"
                         "下面任何组合都必须明显超过它才算有东西。")
            show = df.sort_values("聚类t", ascending=False)
            st.dataframe(show.style.format(
                {"平均收益": "{:+.2%}", "中位收益": "{:+.2%}", "胜率": "{:.1%}",
                 "盈亏比": "{:.2f}", "聚类t": "{:.2f}", "平均持有日": "{:.1f}",
                 "死叉退出": "{:.0%}", "空窗周/年": "{:.1f}"})
                .background_gradient(subset=["聚类t"], cmap="RdYlGn", vmin=-3, vmax=3),
                use_container_width=True, height=560, hide_index=True)

            best = show.iloc[0]
            T_BAR = 3.0
            ok = show[(show["聚类t"] >= T_BAR) & (show["空窗周/年"] <= 5)]
            if len(ok):
                b = ok.iloc[0]
                st.success(f"**通过：N={b['N']}　{b['买入时点']}　{b['周线过滤']}**　"
                           f"聚类 t={b['聚类t']:.2f}，平均单笔 {b['平均收益']:+.2%}，"
                           f"胜率 {b['胜率']:.1%}，空窗 {b['空窗周/年']:.1f} 周/年。")
            else:
                near = show[show["空窗周/年"] <= 5]
                hi = near["聚类t"].max() if len(near) else np.nan
                st.error(f"**没有组合达到门槛。** 满足空窗要求(≤5周/年)的组合里，"
                         f"最高聚类 t = {hi:.2f}，门槛 {T_BAR:.1f}。")
            st.caption(f"门槛用 {T_BAR:.1f} 而不是 2.0：18 个格子里挑最大值，"
                       "即使全是噪音，最大 t 的期望也有 2.7-3.0。")

            st.markdown("### 梯度分析 —— 比单个格子可靠")
            st.caption("单个格子的高 t 可能是运气。如果「确认」或「周线过滤」真的有用，"
                       "应该在所有组合上都体现出来，而不是只在某一格。")
            c1, c2 = st.columns(2)
            g1 = df.groupby("买入时点")[["聚类t", "平均收益", "胜率", "笔数"]].mean()
            c1.markdown("**加速确认有没有用**")
            c1.dataframe(g1.style.format({"聚类t": "{:.2f}", "平均收益": "{:+.2%}",
                                          "胜率": "{:.1%}", "笔数": "{:.0f}"}),
                         use_container_width=True)
            g2 = df.groupby("周线过滤")[["聚类t", "平均收益", "胜率", "空窗周/年"]].mean()
            c2.markdown("**周线过滤有没有用**")
            c2.dataframe(g2.style.format({"聚类t": "{:.2f}", "平均收益": "{:+.2%}",
                                          "胜率": "{:.1%}", "空窗周/年": "{:.1f}"}),
                         use_container_width=True)
            g3 = df.groupby("N")[["聚类t", "平均收益", "胜率"]].mean()
            st.markdown("**N=6 还是 N=9**")
            st.dataframe(g3.style.format({"聚类t": "{:.2f}", "平均收益": "{:+.2%}",
                                          "胜率": "{:.1%}"}), use_container_width=True)

            with st.expander("导出 / 成交明细"):
                sel = st.selectbox("看哪个组合的明细", list(keep))
                tr = keep[sel]
                st.dataframe(tr.tail(300), use_container_width=True, height=320)
                if st.button("生成导出包"):
                    tb = {"01_全部组合": df, "02_随机对照": pd.DataFrame([sc]),
                          "03_按买入时点": g1, "04_按周线过滤": g2, "05_按N": g3,
                          f"06_明细_{sel}": tr}
                    ss["zipb"] = export_all(tb)
                    ss["zipn"] = f"skdj_{dt.datetime.now():%Y%m%d_%H%M}.zip"
                if ss.get("zipb"):
                    st.download_button(f"下载 {ss['zipn']}", ss["zipb"], ss["zipn"],
                                       "application/zip", type="primary")

    with tab2:
        n2 = st.selectbox("N", [9, 6])
        bm2 = st.selectbox("买入时点", list(BUY_MODES))
        wf2 = st.selectbox("周线过滤", WK_FILTERS)
        pk = skdj_picks(panel, elig, n2, bm2, wf2, k_max, top_n)
        if not len(pk):
            st.warning("这个组合下历史上没有候选。")
        else:
            last = pk["date"].max()
            cur = pk[pk["date"] == last]
            nm = basic.set_index("ts_code")["name"].to_dict()
            ind = uni.set_index("ts_code")["ind_name"].to_dict()
            st.subheader(f"最近一次触发：{last:%Y-%m-%d}（{len(cur)} 只）")
            if last < panel["cal"][-1]:
                st.caption(f"数据最新日期是 {panel['cal'][-1]:%Y-%m-%d}，"
                           f"最近 {(panel['cal'][-1]-last).days} 天没有新信号。")
            out = pd.DataFrame([{
                "序": int(r["rank"]), "代码": r["code"], "名称": nm.get(r["code"], ""),
                "行业": ind.get(r["code"], ""),
                "收盘价": round(float(panel["raw_close"].loc[last, r["code"]]), 2),
                "流通市值(亿)": round(float(r["mv"]) / 1e4),
                "K−D": round(float(r["kd"]), 2)} for _, r in cur.iterrows()])
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("下载 CSV", out.to_csv(index=False).encode("utf-8-sig"),
                               f"skdj_{last:%Y%m%d}.csv", "text/csv")
            st.caption("卖出按日线 SKDJ 死叉执行，不设固定止盈止损。")

    if API_ERRORS:
        with st.expander(f"接口异常 {len(API_ERRORS)} 条"):
            st.write(API_ERRORS[-30:])


if __name__ == "__main__" and st is not None:
    main()
