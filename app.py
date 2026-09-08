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
import time
import pickle
import datetime as dt
from typing import Dict, List, Optional

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


# ======================================================================
# 一、Tushare 数据层
# ======================================================================
class Limiter:
    """简单的每分钟请求数限流器。"""

    def __init__(self, per_min: int = 400):
        self.per_min = max(1, int(per_min))
        self.calls: List[float] = []

    def wait(self):
        now = time.time()
        self.calls = [c for c in self.calls if now - c < 60.0]
        if len(self.calls) >= self.per_min:
            sleep_s = 60.0 - (now - self.calls[0]) + 0.3
            if sleep_s > 0:
                time.sleep(sleep_s)
            now = time.time()
            self.calls = [c for c in self.calls if now - c < 60.0]
        self.calls.append(time.time())


def api_call(fn, lim: Limiter, retries: int = 3, **kwargs):
    """带限流与重试的 Tushare 调用。失败返回 None。"""
    last = None
    for k in range(retries):
        try:
            lim.wait()
            return fn(**kwargs)
        except Exception as e:  # 网络抖动 / 限频
            last = e
            time.sleep(2.0 * (k + 1))
    if st is not None:
        st.session_state.setdefault("api_errors", []).append(f"{getattr(fn,'__name__','api')}: {last}")
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


def fetch_one_stock(pro, lim: Limiter, ts_code: str, start: str, end: str,
                    use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    单只股票的日线 + 每日指标，落地磁盘缓存。
    返回列: trade_date, open, high, low, close, pre_close, pct_chg, amount, circ_mv
    注意 Tushare 的 pre_close 已做除权处理，pct_chg 因此是正确的复权收益率。
    """
    path = _px_path(ts_code)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
            if cached is not None and len(cached):
                lo, hi = cached["trade_date"].min(), cached["trade_date"].max()
                if lo <= pd.Timestamp(start) and hi >= pd.Timestamp(end) - pd.Timedelta(days=12):
                    return cached
        except Exception:
            pass

    d = api_call(pro.daily, lim, ts_code=ts_code, start_date=start, end_date=end)
    if d is None or len(d) == 0:
        return None
    b = api_call(pro.daily_basic, lim, ts_code=ts_code, start_date=start, end_date=end,
                 fields="ts_code,trade_date,circ_mv,total_mv,turnover_rate")
    keep = ["trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "amount"]
    d = d[[c for c in keep if c in d.columns]].copy()
    if b is not None and len(b):
        d = d.merge(b[["trade_date", "circ_mv"]], on="trade_date", how="left")
    else:
        d["circ_mv"] = np.nan
    d["trade_date"] = pd.to_datetime(d["trade_date"], format="%Y%m%d")
    d = d.sort_values("trade_date").reset_index(drop=True)
    try:
        with open(path, "wb") as f:
            pickle.dump(d, f)
    except Exception:
        pass
    return d


def build_panel(px: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    把逐股票的长表拼成宽表面板，并构造复权价。
    adj_close[t] = adj_close[t-1] * (1 + pct_chg/100)
    adj_open[t]  = adj_close[t-1] * open[t] / pre_close[t]
    """
    codes = [c for c, d in px.items() if d is not None and len(d) > 30]
    if not codes:
        raise ValueError("没有可用的价格数据")

    def wide(col: str) -> pd.DataFrame:
        s = {c: px[c].set_index("trade_date")[col] for c in codes if col in px[c].columns}
        return pd.DataFrame(s).sort_index()

    raw_close = wide("close")
    raw_open = wide("open")
    pre_close = wide("pre_close")
    pct = wide("pct_chg")
    amount = wide("amount")            # 单位: 千元
    circ_mv = wide("circ_mv")          # 单位: 万元

    cal = raw_close.index
    tradable = raw_close.notna()       # 有行情=可交易，NaN 视为停牌

    adj_close = (1.0 + pct.fillna(0.0) / 100.0).cumprod()
    adj_close = adj_close.where(tradable).ffill()
    prev_adj = adj_close.shift(1)
    ratio_o = (raw_open / pre_close).where(pre_close > 0)
    adj_open = prev_adj * ratio_o

    # 涨跌停判定（创业板/科创板 20%，其余 10%）
    lim_pct = pd.Series(
        [0.20 if (c.startswith("30") or c.startswith("688")) else 0.10 for c in raw_close.columns],
        index=raw_close.columns, dtype=float)
    up_gate = pre_close.mul(1.0 + lim_pct, axis=1) - 0.004
    dn_gate = pre_close.mul(1.0 - lim_pct, axis=1) + 0.004
    limit_up_open = (raw_open >= up_gate) & tradable
    limit_dn_open = (raw_open <= dn_gate) & tradable

    return dict(cal=cal, codes=list(raw_close.columns),
                raw_close=raw_close, raw_open=raw_open, amount=amount, circ_mv=circ_mv,
                adj_close=adj_close, adj_open=adj_open, tradable=tradable,
                limit_up_open=limit_up_open, limit_dn_open=limit_dn_open)


# ======================================================================
# 二、因子计算
# ======================================================================
def compute_factors(adj_close: pd.DataFrame, amount: pd.DataFrame, win: int = 60) -> Dict[str, pd.DataFrame]:
    """全部因子一次性向量化算完，返回 {因子名: 宽表}。"""
    A = adj_close
    logp = np.log(A.where(A > 0))
    ret = A.pct_change()

    mom = A.shift(5) / A.shift(win + 5) - 1.0
    vol = ret.rolling(win).std() * np.sqrt(252.0)
    mom_ra = mom / vol.where(vol > 1e-9)

    # 滚动线性回归: 斜率与 R²。窗口内自变量取全局序号，相关性对平移不变。
    t = pd.Series(np.arange(len(A.index), dtype=float), index=A.index)
    my = logp.rolling(win).mean()
    mt = t.rolling(win).mean()
    mty = logp.mul(t, axis=0).rolling(win).mean()
    cov_ty = mty.sub(my.mul(mt, axis=0))
    var_t = (win * win - 1.0) / 12.0
    var_y = (logp ** 2).rolling(win).mean() - my ** 2
    slope = cov_ty / var_t
    r2 = (cov_ty ** 2) / (var_t * var_y.where(var_y > 1e-12))
    trend_q = slope * 252.0 * r2.clip(0.0, 1.0)

    r60 = A / A.shift(win) - 1.0
    rel_str = r60.sub(r60.mean(axis=1), axis=0)

    a5 = amount.rolling(5).mean()
    a60 = amount.rolling(win).mean()
    vol_exp = a5 / a60.where(a60 > 1e-9)

    dist_hi = A / A.rolling(win).max() - 1.0
    rev5 = A / A.shift(5) - 1.0

    return {"mom_ra": mom_ra, "trend_q": trend_q, "rel_str": rel_str,
            "vol_exp": vol_exp, "dist_hi": dist_hi, "rev5": rev5}


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

    # 上市满 N 天 + 未退市 + 非 ST
    bmap = basic.set_index("ts_code") if len(basic) else pd.DataFrame()
    listed = pd.DataFrame(True, index=cal, columns=codes)
    for c in codes:
        if c in bmap.index:
            row = bmap.loc[c]
            row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            ld = pd.to_datetime(str(row.get("list_date")), format="%Y%m%d", errors="coerce")
            if pd.notna(ld):
                listed[c] &= (cal >= ld + pd.Timedelta(days=min_list_days))
            dd = pd.to_datetime(str(row.get("delist_date")), format="%Y%m%d", errors="coerce")
            if pd.notna(dd):
                listed[c] &= (cal < dd - pd.Timedelta(days=5))
            nm = str(row.get("name", ""))
            if "ST" in nm.upper() or "退" in nm:
                listed[c] = False
    ok &= listed

    # 申万成分进出日期（有则用，做时点行业归属）
    if len(uni) and "in_date" in uni.columns:
        um = uni.set_index("ts_code")
        for c in codes:
            if c not in um.index:
                continue
            r = um.loc[c]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            idt = pd.to_datetime(str(r.get("in_date")), format="%Y%m%d", errors="coerce")
            odt = pd.to_datetime(str(r.get("out_date")), format="%Y%m%d", errors="coerce")
            if pd.notna(idt):
                ok[c] &= (cal >= idt)
            if pd.notna(odt):
                ok[c] &= (cal < odt)
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


def layered_test(fac: pd.DataFrame, adj_close: pd.DataFrame, elig: pd.DataFrame,
                 rebal: List[pd.Timestamp], horizon_weeks: int = 4,
                 n_group: int = 10) -> dict:
    """
    分层检验：每个调仓日按因子值分 n_group 组，看未来 horizon_weeks 周收益是否单调。
    同时输出 IC 序列。累计曲线用不重叠窗口，避免重叠样本夸大统计量。
    """
    cal = adj_close.index
    pos = {d: i for i, d in enumerate(cal)}
    step = horizon_weeks * 5
    rows, ic_rows = [], []

    for d in rebal:
        i = pos.get(d)
        if i is None or i + step >= len(cal):
            continue
        f = fac.iloc[i].where(elig.iloc[i]).to_numpy(dtype=float)
        fwd = (adj_close.iloc[i + step] / adj_close.iloc[i] - 1.0).to_numpy(dtype=float)
        m = np.isfinite(f) & np.isfinite(fwd)
        if m.sum() < n_group * 3:
            continue
        fv, rv = f[m], fwd[m]
        grp = pd.qcut(pd.Series(fv).rank(method="first"), n_group,
                      labels=False, duplicates="drop").to_numpy()
        means = [np.nanmean(rv[grp == g]) if (grp == g).sum() else np.nan for g in range(n_group)]
        rows.append(pd.Series(means, index=[f"D{g+1}" for g in range(n_group)], name=d))
        ic_rows.append(pd.Series({"date": d, "ic": _spearman(fv, rv)}))

    if not rows:
        return {"ok": False}

    grp_df = pd.DataFrame(rows)
    ic = pd.DataFrame(ic_rows).set_index("date")["ic"].dropna()

    # 不重叠累计曲线
    sub = grp_df.iloc[::max(1, horizon_weeks)]
    curve = (1.0 + sub).cumprod()

    ic_mean = float(ic.mean()) if len(ic) else np.nan
    ic_std = float(ic.std()) if len(ic) else np.nan
    icir = ic_mean / ic_std if ic_std and ic_std > 1e-12 else np.nan
    tstat = icir * np.sqrt(len(ic)) if np.isfinite(icir) else np.nan
    top, bot = grp_df.columns[-1], grp_df.columns[0]
    # 单调性: 各组均值与组序号的秩相关
    ordered = grp_df.mean(axis=0).to_numpy()
    mono = _spearman(np.arange(len(ordered), dtype=float), ordered, min_n=4)

    return {"ok": True, "group_mean": grp_df.mean(axis=0), "curve": curve, "ic": ic,
            "ic_mean": ic_mean, "icir": icir, "tstat": tstat,
            "spread": float(grp_df[top].mean() - grp_df[bot].mean()),
            "monotonic": mono, "n_period": len(grp_df)}


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

    AC = panel["adj_close"].to_numpy(dtype=float)
    AO = panel["adj_open"].to_numpy(dtype=float)
    TRD = panel["tradable"].to_numpy(dtype=bool)
    LU = panel["limit_up_open"].to_numpy(dtype=bool)
    LD = panel["limit_dn_open"].to_numpy(dtype=bool)
    SC = score.reindex(index=cal, columns=codes).to_numpy(dtype=float)
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
        per_min = st.slider("每分钟请求上限", 60, 800, 400, 20)
        use_cache = st.checkbox("使用本地缓存", True)
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
        pro = ts.pro_api()
        lim = Limiter(per_min)
        s_str, e_str = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

        with st.status("正在获取数据…", expanded=True) as status:
            st.write("1/3 取申万成分股…")
            uni = fetch_universe(pro, lim, l1, l2)
            if not len(uni):
                st.error("行业成分股为空。可能是 Tushare 积分不足，无法调用申万接口。")
                st.stop()
            st.write(f"   候选 {len(uni)} 只")

            st.write("2/3 取股票基础信息（含退市）…")
            basic = fetch_stock_basic(pro, lim)

            st.write("3/3 下载日线与每日指标…")
            codes = list(uni["ts_code"])
            if max_stk:
                codes = codes[: int(max_stk)]
            bar = st.progress(0.0)
            px, fail = {}, 0
            for k, c in enumerate(codes):
                d = fetch_one_stock(pro, lim, c, s_str, e_str, use_cache)
                if d is not None:
                    px[c] = d
                else:
                    fail += 1
                if k % 5 == 0 or k == len(codes) - 1:
                    bar.progress((k + 1) / len(codes), text=f"{k+1}/{len(codes)}  失败 {fail}")

            panel = build_panel(px)
            ss["panel"] = panel
            ss["basic"] = basic
            ss["uni"] = uni
            ss["factors"] = compute_factors(panel["adj_close"], panel["amount"])
            status.update(label=f"完成：{len(panel['codes'])} 只，{len(panel['cal'])} 个交易日", state="complete")

    if ss.get("panel") is None:
        st.info("左侧填入 Tushare Token 后点「下载 / 更新数据」。首次全量下载约需 5-15 分钟，之后走本地缓存。")
        st.stop()

    panel, basic, uni, factors = ss["panel"], ss["basic"], ss["uni"], ss["factors"]

    elig = build_eligibility(panel, basic, uni, mv_lo, mv_hi, min_price, min_amt, int(min_days))
    ss["elig"] = elig
    rebal_all = weekly_rebal_dates(panel["cal"])

    t1, t2, t3, t4 = st.tabs(["数据总览", "因子分层检验", "策略回测", "当前选股"])

    # ---------------- 数据总览 ----------------
    with t1:
        cnt = elig.sum(axis=1)
        a, b, c, d = st.columns(4)
        a.metric("下载股票数", len(panel["codes"]))
        b.metric("交易日数", len(panel["cal"]))
        c.metric("当前合格数", int(cnt.iloc[-1]))
        d.metric("历史平均合格数", f"{cnt.mean():.0f}")
        st.subheader("每日合格股票数量")
        st.line_chart(cnt.rename("合格数"))
        st.caption("若某段时间合格数长期低于 30，说明市值/股价门槛在那个阶段过严，排名的区分度会下降。")
        if ss.get("api_errors"):
            with st.expander(f"接口异常 {len(ss['api_errors'])} 条"):
                st.write(ss["api_errors"][-40:])

    # ---------------- 因子分层检验 ----------------
    with t2:
        st.subheader("先回答一个问题：这个打分有没有排序能力？")
        st.markdown("**如果第1组打不过第10组、或者没有单调性，这个因子就是废的**——"
                    "后面加止损、加仓位管理都救不回来。这一步不涉及任何买卖逻辑。")
        cc = st.columns(3)
        fkey = cc[0].selectbox("因子", FACTOR_KEYS, format_func=lambda k: FACTOR_DEF[k][0])
        hz = cc[1].select_slider("持有期（周）", [1, 2, 4, 6, 8], 4)
        ng = cc[2].slider("分组数", 5, 10, 10)
        sp = st.columns(2)
        ls = sp[0].date_input("样本起", dt.date(2018, 1, 1), key="ls")
        le = sp[1].date_input("样本止", dt.date.today(), key="le")

        if st.button("运行分层检验", type="primary"):
            rb = [d for d in rebal_all if pd.Timestamp(ls) <= d <= pd.Timestamp(le)]
            res = layered_test(factors[fkey], panel["adj_close"], elig, rb, int(hz), int(ng))
            if not res.get("ok"):
                st.error("样本不足，放宽日期或降低门槛。")
            else:
                m = st.columns(5)
                m[0].metric("多空价差", f"{res['spread']:.2%}")
                m[1].metric("IC 均值", f"{res['ic_mean']:.4f}")
                m[2].metric("ICIR", f"{res['icir']:.3f}")
                m[3].metric("t 统计量", f"{res['tstat']:.2f}")
                m[4].metric("单调性", f"{res['monotonic']:.2f}")
                st.bar_chart(res["group_mean"].rename(f"未来{hz}周平均收益"))
                st.line_chart(res["curve"])
                st.line_chart(res["ic"].rolling(12).mean().rename("IC(12期均线)"))
                st.markdown(
                    "**怎么看**：单调性接近 +1 或 -1、|IC均值| > 0.03、|t| > 2，才算这个因子有话说。"
                    "单调性接近 0 意味着分组收益是噪音。IC 均线长期在 0 附近来回穿，说明因子只在个别时段有效。")

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
            score = composite_score(factors, elig, weights)
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
        score = composite_score(factors, elig, weights_now)
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
