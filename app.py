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

本版改动
--------
- 页面顶部和侧边栏显示程序版本，用来确认网页运行的是最新代码。
- 「乖离规则检验」改为第③页的独立段落，不再需要先运行乖离划分。
- 第④页候选表和「候选股事后表现」加「高出30日线」一列，30%以上标 ⚠️。

更早的改动
----------
- 第③页乖离划分下新增「乖离规则检验」：跳过高出30日线30%以上的票，
  比较基准 / 剔除不补位 / 顺延补位的复利倍数与账户回撤。

更早的改动
----------
- 侧边栏默认改为每次5只、冷却5日。
- 第③页新增「乖离」干净划分：按选中时股价高出30日线的幅度（10%以内/10%-30%/30%以上）
  分组，看收益、大亏比例和持有中浮亏；以及「逐只查看」：输入日期和代码，列出当天的
  乖离、SKDJ、MACD、20日涨幅和次日买入持有到期的结果。

更早的改动
----------
- 第③页新增「延迟入场」检验：同一份名单，比较次日直接买 / 等SKDJ上行(K>D且K上升)再买 /
  等MACD柱连续2天回升再买（最多等20日）。除收益外，给出持有期内最大浮亏、
  浮亏曾超20%的比例、账户复利与回撤。

更早的改动
----------
- 第②页新增「账户净值模拟（真实复利）」：分1份/4份/每天一份入场，给出复利倍数、
  最大回撤（中位与最差起点）、回撤起止日期和逐年复利收益。
- 第③页新增「日线SKDJ下跌趋势」检验：从75上方跌下来、死叉后一直没金叉、K和D都在75下方。
  ① 干净划分；② 基准 / 剔除不补位（实盘做法）/ 剔除顺延补位 三种做法，按复利和回撤比较。
- 第④页候选表和「候选股事后表现」加「日线SKDJ」一列，标出处在下跌趋势的票，用来核对定义。

更早的改动
----------
- 第④页新增「候选股事后表现」：过去 N 个选股日的全部候选，按回测口径算收益
  （已满20日的与回测逐笔一致，未满的按最新收盘算浮动）；可粘贴自己的实盘记录，
  自动比较「你选中的」和「同一天没选中的」。
- 第③页改名「专项检验」，只保留：板块内排名分档、执行时机（龙头领跑天数），
  以及一份「已经验证过、结论是不用的想法」记录。删除了板块结构、绝对动量、
  热点过气、K值/高位死叉划分等已验证无效的检验及其代码。

更早的改动
----------
- 侧边栏默认值 = 滚动前推六年选中的配置：板块60日动量、3个板块、每次3只、
  每板块最多2只、持有20日、每1日选一次、冷却2日，持有到期不止损。
- 「今日候选」按回测完全相同的日程（同起点、每 N 日一次、同样冷却）重放到最新一天，
  名单 = 回测在这一天会选的票；「冷却交易日」实现名单轮换。
- 「资金年化(近似)」= 总收益 ÷ 总持有交易日 × 244：每笔平均收益不能乘笔数。
- 修复：下载时误报缓存不可写。
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
CACHE_TTL_HOURS = 24                       # 缓存 24 小时后自动失效，重新下载
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
            # 超过 CACHE_TTL_HOURS 小时的缓存视为过期。行情每天都会新增，
            # 隔夜的缓存即使覆盖了请求区间，也缺最近的交易日。
            if time.time() - os.path.getmtime(path) > CACHE_TTL_HOURS * 3600:
                raise TimeoutError("cache expired")
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
def build_sector_map(uni: pd.DataFrame, panel: dict) -> Dict[str, List[str]]:
    """
    code → 申万二级行业。**不做任何筛选**。

    旧版本按"全期平均合格数 ≥ 5"决定一个行业算不算板块，那有两个毛病：
      1. 前视偏差 —— 回测 2019 年时，板块名单是用截至 2026 年的数据定的；
         一个 2024 年才活跃起来的板块，在 2019 年的回测里就已经存在。
      2. 窗口依赖 —— 换个数据起始日期，板块名单就变，同一天能选出不同的股票。
    现在改成逐日判定（见 build_sector_index 的 avail），两个毛病同时消失。
    """
    if "l2_name" not in uni.columns:
        return {}
    m = uni.dropna(subset=["l2_name"]).drop_duplicates("ts_code")
    m = m[m["ts_code"].isin(panel["codes"])]
    grp: Dict[str, List[str]] = {}
    for sec, g in m.groupby("l2_name"):
        codes = [c for c in g["ts_code"] if c in panel["codes"]]
        if codes:
            grp[str(sec)] = codes
    return grp


def build_sector_index(panel: dict, elig: pd.DataFrame,
                       sectors: Dict[str, List[str]],
                       min_members: int = 5) -> tuple:
    """
    每个板块的等权指数，外加**逐日的可用性**。
    返回 (板块日收益表, 板块指数, 每日成分股数, 每日是否可用)。

    可用 = 当天该板块的合格成分股 ≥ min_members。只用当天的信息，
    所以不含前视；也因此换数据窗口不会改变任何一天的判定。
    指数本身用 ≥3 只就算（够画出走势），但不足 min_members 的日子不可选。
    """
    ret = panel["adj_close"].pct_change()
    rows, cnts, avs = {}, {}, {}
    for sec, codes in sectors.items():
        m = elig[codes]
        n = m.sum(axis=1)
        rows[sec] = ret[codes].where(m).mean(axis=1).where(n >= 3)
        cnts[sec] = n
        avs[sec] = (n >= min_members)
    R = pd.DataFrame(rows)
    IDX = (1.0 + R.fillna(0.0)).cumprod().where(R.notna()).ffill()
    return R, IDX, pd.DataFrame(cnts), pd.DataFrame(avs)


def sector_noise_check(panel: dict, elig: pd.DataFrame,
                       sectors: Dict[str, List[str]],
                       min_members: int = 5) -> pd.DataFrame:
    """
    最关键的前置检验：板块指数的波动到底比个股小多少？
    如果降噪幅度不明显，"换单位"这个思路就不成立，后面不用做了。
    """
    ret = panel["adj_close"].pct_change()
    R, _, cnt, av = build_sector_index(panel, elig, sectors, min_members)
    rows = []
    for sec, codes in sectors.items():
        m = elig[codes]
        iv = ret[codes].where(m).std().mean() * np.sqrt(252)     # 成分股平均年化波动
        sv = R[sec].std() * np.sqrt(252)                          # 板块指数年化波动
        rows.append({"板块": sec, "平均成分股数": float(m.sum(axis=1).mean()),
                     "可用天数占比": float(av[sec].mean()),
                     "个股平均波动": float(iv), "板块指数波动": float(sv),
                     "降噪比": float(sv / iv) if iv > 0 else np.nan})
    d = pd.DataFrame(rows).set_index("板块").sort_values("平均成分股数", ascending=False)
    return d


def sector_factors(R: pd.DataFrame, IDX: pd.DataFrame,
                   amt_sec: pd.DataFrame,
                   avail: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
    """
    板块层面的候选信号。全部只用过去数据。
    avail 给出后，不可用的日子会被置空 —— 那天这个板块就不会被选中。
    """
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
    if avail is not None:
        av = avail.reindex(index=IDX.index, columns=IDX.columns).fillna(False)
        out = {k: v.where(av) for k, v in out.items()}
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


def skdj_downtrend_mask(K: pd.DataFrame, D: pd.DataFrame, level: float = 75.0,
                        look: int = 10) -> pd.DataFrame:
    """
    「日线 SKDJ 从 75 上方跌下来、一直在跌」的状态，逐日逐股 True/False，只用当天及以前的数据：
      ① 当前处在死叉状态（K < D），而且这段死叉开始前的 look 个交易日内 K 曾经 ≥ level
         （= 从高位跌下来的那次死叉）；
      ② 死叉以来一直没有金叉（中间只要 K 重新上穿 D，这段就结束）；
      ③ 当天 K 和 D 都已经跌到 level 以下。
    刚死叉、K 还在 75 上方的头几天不算（那一段之前验证过，反而更好）。
    """
    below = (K < D)
    below_prev = below.shift(1).fillna(False).astype(bool)
    cross = below & ~below_prev                                   # 死叉那天
    was_high = K.rolling(look, min_periods=1).max().shift(1) >= level
    start_high = (cross & was_high).astype(float).where(cross)   # 只在死叉那天有值
    state = start_high.ffill().where(below, 0.0).fillna(0.0) > 0.5
    return state & (K < level) & (D < level)


STOCK_RULES = ["S1_板块内最强", "S2_板块内最弱(回调)", "S3_板块内随机"]


# 曾经验证过、已移除的买入/卖出条件（K值过滤、高位死叉、止损、利润保护、
# 成分股数、绝对动量、热点过气……）结论汇总在第③页「已经验证过、结论是不用的想法」。


def sector_then_stock(panel: dict, elig: pd.DataFrame, sectors: Dict[str, List[str]],
                      sec_fac: pd.DataFrame, dates: List[pd.Timestamp],
                      top_sec: int = 2, top_n: int = 3,
                      stock_rule: str = "S1_板块内最强",
                      sec_rule: str = "最强", cooldown: int = 5,
                      seed: int = 20260910, kdf: pd.DataFrame = None,
                      per_sec_cap: int = 0, min_members: int = 5,
                      skip_mask: pd.DataFrame = None) -> pd.DataFrame:
    """
    两层选股：先按 sec_fac 选出 top_sec 个板块，再在板块内按 stock_rule 选股。
    sec_rule="随机" 时板块层用随机选择 —— 这是判断"板块层有没有加分"的对照组。

    cooldown（冷却）就是"轮换"本身：同一只股票入选后 cooldown 个交易日内不再入选，
    名额顺延给板块内下一名。例：每 3 日选一次 + 冷却 5 日 → 板块和排名不变时，
    名单在「第1-2名」和「第3-4名」之间交替；每 1 日选一次 + 冷却 2 日 → 每天交替。
    冷却 ≤ 选股间隔时冷却不起作用，连续选股日名单会一模一样。

    skip_mask（可选）：当天为 True 的股票直接跳过，名额顺延给后面的名次；
    跳过的票不补回来，前几个板块凑不满就少选。默认 None，结果与已验证口径完全一致。
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
            _sc = sectors[s]
            _ok = elig.loc[d, _sc].fillna(False).to_numpy(dtype=bool)   # 一次取整行，比逐只 loc 快几十倍
            codes = [c for c, o in zip(_sc, _ok) if o]
            # 当天合格成分股不足的板块不可选（信号那边已置空，这里再挡一道）
            if len(codes) < min_members:
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
            for rk, c in enumerate(order, 1):
                cand.append((s, c, float(v[c]), rk))
        taken = 0
        used: Dict[str, int] = {}
        krow = kdf.loc[d] if (kdf is not None and d in kdf.index) else None
        srow = skip_mask.loc[d] if (skip_mask is not None and d in skip_mask.index) else None
        for s, c, sc, rk in cand:
            if taken >= top_n:
                break
            if c in last and i - last[c] < cooldown:
                continue
            # 候选是按板块顺序排的（最强板块的股票全部在前）。
            # per_sec_cap>0 则每个板块最多取这么多只，剩下的名额轮到下一个板块。
            if per_sec_cap > 0 and used.get(s, 0) >= per_sec_cap:
                continue
            if srow is not None and c in srow.index and bool(srow[c]):
                continue
            used[s] = used.get(s, 0) + 1
            kv = float(krow[c]) if (krow is not None and c in krow.index) else np.nan
            rows.append({"date": d, "code": c, "板块": s, "rank": taken + 1,
                         "板块内名次": rk, "score": sc, "买入K": kv})
            last[c] = i
            taken += 1
    return pd.DataFrame(rows)


RANK_BANDS_SEC = [(1, 2), (3, 4), (5, 6), (7, 10)]


def in_sector_rank_test(panel: dict, elig: pd.DataFrame, sectors: Dict[str, List[str]],
                        sec_fac: pd.DataFrame, dates: List[pd.Timestamp],
                        top_sec: int = 2, hold: int = 20, step_days: int = 3,
                        bands=RANK_BANDS_SEC, min_members: int = 5,
                        comm: float = 0.0003, stamp: float = 0.0005,
                        slip: float = 0.001) -> dict:
    """
    板块内排名分档 —— 回答「轮换到第3、4名有没有代价」。

    每个选股日取最强的 top_sec 个板块，板块内按20日涨幅排名，
    第1-2、3-4、5-6、7-10名**分别**按固定持有期成交。
    不设冷却、不做替补：各档来自同一天、同一个板块，唯一的差别就是名次。

    配对差：同一天、同一板块里「该档均值 − 第1-2名均值」，先按日平均，
    再做 Newey-West 重叠修正（持有期远长于选股间隔，相邻样本高度重叠）。
    """
    A = panel["adj_close"]
    m20 = (A / A.shift(20) - 1.0)
    lab = {r: f"第{lo}-{hi}名" for lo, hi in bands for r in range(lo, hi + 1)}
    maxr = max(hi for _, hi in bands)
    rows = []
    for d in dates:
        if d not in sec_fac.index:
            continue
        f = sec_fac.loc[d].dropna()
        if len(f) < top_sec + 1:
            continue
        for s in f.sort_values(ascending=False).index[:top_sec]:
            _sc = sectors[s]
            _ok = elig.loc[d, _sc].fillna(False).to_numpy(dtype=bool)
            codes = [c for c, o in zip(_sc, _ok) if o]
            if len(codes) < min_members:
                continue
            v = m20.loc[d, codes].dropna().sort_values(ascending=False)
            for rk, c in enumerate(v.index[:maxr], 1):
                rows.append({"date": d, "code": c, "板块": s, "板块内名次": rk,
                             "档": lab[rk], "score": float(v[c])})
    if not rows:
        return {}
    pk = pd.DataFrame(rows)
    tr = track_fixed(pk, panel, hold, comm=comm, stamp=stamp, slip=slip)
    if not len(tr):
        return {}
    tr = tr.dropna(subset=["收益率"]).merge(
        pk[["date", "code", "板块内名次", "档"]], on=["date", "code"], how="left")
    order = [f"第{lo}-{hi}名" for lo, hi in bands]
    tr["年"] = pd.to_datetime(tr["date"]).dt.year
    lag = max(1, int(np.ceil(hold / max(step_days, 1))))

    summ = []
    for g in order:
        sub = tr[tr["档"] == g]
        if len(sub) < 30:
            continue
        yr = sub.groupby("年")["收益率"].mean()
        summ.append({"档": g, "笔数": len(sub), "平均收益": sub["收益率"].mean(),
                     "中位收益": sub["收益率"].median(),
                     "胜率": float((sub["收益率"] > 0).mean()),
                     "聚类t(重叠修正)": newey_west_t(
                         sub.groupby("date")["收益率"].mean().sort_index(), lag),
                     "逐年为正": f"{int((yr > 0).sum())}/{len(yr)}"})
    summ = pd.DataFrame(summ).set_index("档") if summ else pd.DataFrame()

    # 配对：同一天同一板块
    cell = tr.groupby(["date", "板块", "档"])["收益率"].mean().unstack("档")
    base = order[0]
    pair = []
    if base in cell.columns:
        for g in order[1:]:
            if g not in cell.columns:
                continue
            dd = (cell[g] - cell[base]).dropna()
            if len(dd) < 30:
                continue
            day = dd.groupby(level="date").mean().sort_index()
            yr = day.groupby(pd.DatetimeIndex(day.index).year).mean()
            pair.append({"对比": f"{g} − {base}", "配对样本": len(dd),
                         "平均差": dd.mean(), "中位差": dd.median(),
                         "t(重叠修正)": newey_west_t(day, lag),
                         "差为正的年份": f"{int((yr > 0).sum())}/{len(yr)}"})
    pair = pd.DataFrame(pair).set_index("对比") if pair else pd.DataFrame()

    yearly = tr.pivot_table(index="年", columns="档", values="收益率", aggfunc="mean")
    yearly = yearly[[c for c in order if c in yearly.columns]]
    return {"summary": summ, "pair": pair, "yearly": yearly}


def candidate_outcomes(picks: pd.DataFrame, panel: dict, hold: int = 20,
                       comm: float = 0.0003, stamp: float = 0.0005,
                       slip: float = 0.001) -> pd.DataFrame:
    """
    候选股事后表现：每只候选按回测口径（选股日次日开盘买，第 hold 个交易日开盘卖，
    含成本）算收益；还没满 hold 日的，按最新收盘价算浮动收益（同样扣买卖成本）。
    已满 hold 日的结果与 track_fixed 逐笔一致。
    """
    if picks is None or not len(picks):
        return pd.DataFrame()
    cal = panel["adj_close"].index
    n = len(cal)
    ci = {c: j for j, c in enumerate(panel["codes"])}
    pos = {d: i for i, d in enumerate(cal)}
    AO = panel["adj_open"].to_numpy(dtype=np.float64)
    AC = panel["adj_close"].to_numpy(dtype=np.float64)
    AH = panel["adj_high"].to_numpy(dtype=np.float64)
    AL = panel["adj_low"].to_numpy(dtype=np.float64)
    RO = panel["raw_open"].to_numpy(dtype=np.float64)
    TRD = panel["tradable"].to_numpy(dtype=bool)
    LU = panel["limit_up_open"].to_numpy(dtype=bool)
    LD = panel["limit_dn_open"].to_numpy(dtype=bool)
    cin, cout = comm + slip, comm + stamp + slip
    rows = []
    for r in picks.itertuples(index=False):
        d, c = r.date, r.code
        i0, j = pos.get(d), ci.get(c)
        row = {"选股日": d, "序": int(getattr(r, "rank", 0)), "code": c,
               "板块": getattr(r, "板块", ""),
               "板块内名次": getattr(r, "板块内名次", np.nan),
               "买入日": pd.NaT, "买入价(实际)": np.nan, "状态": "", "收益率": np.nan,
               "已满期": False, "持有交易日": np.nan, "期间最高": np.nan, "期间最低": np.nan}
        if i0 is None or j is None:
            continue
        b = i0 + 1
        if b >= n:
            row["状态"] = "下一交易日开盘买"
            rows.append(row)
            continue
        row["买入日"] = cal[b]
        if not TRD[b, j] or LU[b, j] or not np.isfinite(AO[b, j]):
            row["状态"] = "开盘涨停或停牌，买不到"
            rows.append(row)
            continue
        p0 = AO[b, j]
        row["买入价(实际)"] = round(float(RO[b, j]), 2) if np.isfinite(RO[b, j]) else np.nan
        entry = p0 * (1 + cin)
        e = b + hold
        done = False
        if e < n:
            while e < n and (not TRD[e, j] or LD[e, j] or not np.isfinite(AO[e, j])):
                e += 1
                if e - b > hold + 5:
                    break
            if e < n and np.isfinite(AO[e, j]):
                row["收益率"] = AO[e, j] * (1 - cout) / entry - 1.0
                row["状态"] = f"已满{hold}日（{cal[e]:%m-%d}卖）"
                row["已满期"] = True
                row["持有交易日"] = e - b
                seg_h, seg_l = AH[b:e, j], AL[b:e, j]
                done = True
        if not done:
            last = n - 1
            while last > b and not np.isfinite(AC[last, j]):
                last -= 1
            if np.isfinite(AC[last, j]):
                row["收益率"] = AC[last, j] * (1 - cout) / entry - 1.0
            row["状态"] = f"持有中·第{last - b + 1}天"
            row["持有交易日"] = last - b
            seg_h, seg_l = AH[b:last + 1, j], AL[b:last + 1, j]
        if np.isfinite(seg_h).any():
            row["期间最高"] = float(np.nanmax(seg_h) / p0 - 1.0)
        if np.isfinite(seg_l).any():
            row["期间最低"] = float(np.nanmin(seg_l) / p0 - 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_my_trades(text: str) -> pd.DataFrame:
    """
    解析用户粘贴的实盘记录：每行一条「日期 代码」，日期可写 2026-09-15 / 2026/9/15 / 20260915，
    代码写6位数字即可。日期既可以是选股日也可以是买入日。
    """
    import re
    out = []
    for line in (text or "").splitlines():
        m_d = re.search(r"(20\d{2})[-/.年]?(\d{1,2})[-/.月]?(\d{1,2})", line)
        m_c = re.search(r"(?<!\d)(\d{6})(?!\d)", re.sub(r"20\d{2}[-/.年]?\d{1,2}[-/.月]?\d{1,2}日?", " ", line))
        if not (m_d and m_c):
            continue
        try:
            dd = pd.Timestamp(int(m_d.group(1)), int(m_d.group(2)), int(m_d.group(3)))
        except Exception:
            continue
        out.append({"日期": dd, "代码6位": m_c.group(1)})
    return pd.DataFrame(out).drop_duplicates() if out else pd.DataFrame(columns=["日期", "代码6位"])


def match_my_trades(hist: pd.DataFrame, mine: pd.DataFrame) -> pd.Series:
    """给 hist 每一行标记「你买了」：先按选股日匹配，匹配不上再按买入日。"""
    flag = pd.Series(False, index=hist.index)
    if mine is None or not len(mine) or not len(hist):
        return flag
    c6 = hist["code"].str[:6]
    for _, m in mine.iterrows():
        hit = (c6 == m["代码6位"]) & (pd.to_datetime(hist["选股日"]) == m["日期"])
        if not hit.any():
            hit = (c6 == m["代码6位"]) & (pd.to_datetime(hist["买入日"]) == m["日期"])
        flag |= hit
    return flag


def account_sim(tr: pd.DataFrame, dates: List[pd.Timestamp], hold: int = 20,
                every: int = 1, tranches=(1, 4, 0)) -> dict:
    """
    账户净值模拟（真实复利）：资金分成 k 份，每份隔 gap 个选股日入场，
    买入当天名单（名单里有几只就平分几只），持有 hold 日后卖出，收回的钱立刻买当天的新名单。
    k=0 表示「每个选股日都投一份」（≈ 回测里的满仓轮动）。
    某天名单为空（例如被过滤掉），那一份就拿现金等到下一轮。
    k 份不同的起始日会给出不同的结果，全部算出来看中位和最差——
    这就是「什么时候开始」的运气成分。
    """
    if tr is None or not len(tr):
        return {}
    H = max(1, int(round(hold / max(every, 1))))
    dts = list(dates)
    basket = tr.dropna(subset=["收益率"]).groupby("date")["收益率"].mean()
    r = pd.Series(dts, index=dts).map(basket).fillna(0.0).to_numpy()
    n = len(r)
    yrs = max((dts[-1] - dts[0]).days / 365.25, 0.5)
    out, curve = [], None
    for k in tranches:
        kk = H if k == 0 else k
        gap = max(1, H // kk)
        finals, dds, paths = [], [], []
        for s0 in range(gap):
            vals = np.full(kk, 1.0 / kk)
            eq_v = []
            # 净值在每份到期卖出时结算；持仓中的浮亏不计入，真实回撤会更深一些
            nxt = {s0 + j * gap: j for j in range(kk)}          # 入场日 -> 份号
            pend = {}
            for i in range(n):
                if i in pend:                                   # 到期结算
                    j, ri = pend.pop(i)
                    vals[j] *= 1.0 + ri
                    nxt[i] = j
                if i in nxt:                                    # 入场
                    j = nxt.pop(i)
                    if i + H < n:
                        pend[i + H] = (j, r[i])
                eq_v.append(vals.sum())
            ev = np.array(eq_v)
            finals.append(ev[-1])
            dds.append(float((ev / np.maximum.accumulate(ev) - 1.0).min()))
            paths.append(ev)
        f = np.array(finals)
        lab = "每个选股日投一份（≈回测）" if k == 0 else f"分{kk}份，每{gap}个选股日投一份"
        if kk == 1:
            lab = f"一次全仓，每{H}个选股日换一次"
        out.append({"入场方式": lab, "起始日数": len(f),
                    "最终倍数(中位)": float(np.median(f)), "最终倍数(最差)": float(f.min()),
                    "最终倍数(最好)": float(f.max()),
                    "复利年化(中位)": float(np.median(f) ** (1 / yrs) - 1),
                    "最大回撤(中位)": float(np.median(dds)), "最大回撤(最差)": float(min(dds))})
        if k == 0:
            curve = pd.Series(paths[0], index=pd.DatetimeIndex(dts))
    res = {"summary": pd.DataFrame(out).set_index("入场方式")}
    if curve is not None:
        dd = curve / curve.cummax() - 1.0
        tr_ = dd.idxmin(); pk = curve[:tr_].idxmax()
        rec = curve[tr_:][curve[tr_:] >= curve[pk]]
        res.update({"curve": curve, "dd": dd, "peak": pk, "trough": tr_,
                    "recover": rec.index[0] if len(rec) else None,
                    "yearly": curve.groupby(curve.index.year).last().pct_change()
                    .fillna(curve.groupby(curve.index.year).last().iloc[0] - 1.0)})
    return res


def skdj_filter_test(panel: dict, elig: pd.DataFrame, sectors: Dict[str, List[str]],
                     sec_fac: pd.DataFrame, dates: List[pd.Timestamp], mask: pd.DataFrame,
                     base_pk: pd.DataFrame, base_tr: pd.DataFrame, pool_tr: pd.DataFrame,
                     top_sec: int, top_n: int, cooldown: int, per_sec_cap: int,
                     min_members: int, hold: int, every: int, kdf: pd.DataFrame = None,
                     cut: str = "2023-01-01", flag_label: str = "下跌趋势（你不会买的）",
                     **kw) -> dict:
    """
    买入过滤的通用检验（最早用于 SKDJ 下跌趋势，乖离规则也用它）：
      ① 干净划分：同一批成交按「买入决策当天是否处在下跌趋势」分两组，不替补；
      ② 三种做法对比，都按全部资金（空着的钱收益记 0）和真实复利的账户净值来比：
         基准            —— 名单全买；
         剔除，不补位    —— 名单不变，下跌趋势的票不买，钱空着（= 你实盘的做法）；
         剔除，顺延补位  —— 下跌趋势的票跳过，名额给板块内后面的名次。
    """
    c = pd.Timestamp(cut)
    lag = max(1, int(np.ceil(hold / max(every, 1))))

    def flag(df):
        out = []
        for dt_, cc in zip(df["date"], df["code"]):
            try:
                out.append(bool(mask.at[dt_, cc]))
            except Exception:
                out.append(False)
        return np.array(out, dtype=bool)

    bt = base_tr.dropna(subset=["收益率"]).copy()
    bt["下跌趋势"] = flag(bt)
    if pool_tr is not None and len(pool_tr):
        pm = pool_tr.dropna(subset=["收益率"]).groupby("date")["收益率"].mean()
        bt["同期超额"] = bt["收益率"] - bt["date"].map(pm)
    bt["年"] = pd.to_datetime(bt["date"]).dt.year
    val = "同期超额" if "同期超额" in bt.columns else "收益率"
    rows = []
    for lab, sub in ((flag_label, bt[bt["下跌趋势"]]), ("其他", bt[~bt["下跌趋势"]])):
        if not len(sub):
            continue
        day = sub.groupby("date")[val].mean().dropna().sort_index()
        yr = sub.groupby("年")[val].mean()
        rows.append({"分组": lab, "笔数": len(sub), "占比": len(sub) / len(bt),
                     "平均收益": sub["收益率"].mean(), "中位收益": sub["收益率"].median(),
                     "胜率": float((sub["收益率"] > 0).mean()),
                     "20日内最惨10%": sub["收益率"].quantile(0.10),
                     val: sub[val].mean(), f"t(重叠修正,{val})": newey_west_t(day, lag),
                     "为正年数": f"{int((yr > 0).sum())}/{len(yr)}"})
    split = pd.DataFrame(rows).set_index("分组")
    yearly_split = bt.pivot_table(index="年", columns="下跌趋势", values=val, aggfunc="mean")
    yearly_split = yearly_split.rename(columns={True: "下跌趋势", False: "其他"})

    # 三种做法
    v_drop = bt[~bt["下跌趋势"]].drop(columns=["下跌趋势"])
    pk_sub = sector_then_stock(panel, elig, sectors, sec_fac, dates, top_sec, top_n,
                               "S1_板块内最强", "最强", cooldown=cooldown, kdf=kdf,
                               per_sec_cap=per_sec_cap, min_members=min_members,
                               skip_mask=mask)
    v_sub = track_fixed(pk_sub, panel, hold, **kw).dropna(subset=["收益率"]) if len(pk_sub) else pd.DataFrame()
    plans = {"基准：名单全买": bt, "剔除，不补位（你的实盘做法）": v_drop, "剔除，顺延补位": v_sub}
    bdays = float(bt["持有交易日"].sum())
    b_in = float(bt.loc[pd.to_datetime(bt["date"]) < c, "持有交易日"].sum())
    b_y = bt.groupby("年")["持有交易日"].sum()

    def fa(t, den):
        if not den:
            return np.nan
        return float(t["收益率"].sum() / den * 244) if len(t) else 0.0

    out, curves, yr_tab = [], {}, {}
    for nm, t in plans.items():
        if len(t):
            t = t.copy(); t["年"] = pd.to_datetime(t["date"]).dt.year
        ty = pd.Series({y: fa(t[t["年"] == y] if len(t) else t, b_y[y]) for y in b_y.index})
        yr_tab[nm] = ty
        sim = account_sim(t, dates, hold, every, tranches=(4, 0))
        sm = sim.get("summary", pd.DataFrame())
        d0 = sm.iloc[-1] if len(sm) else None
        d4 = sm.iloc[0] if len(sm) else None
        cmp_ = (ty - yr_tab[list(plans)[0]]).dropna() if nm != list(plans)[0] else pd.Series(dtype=float)
        tin = t[pd.to_datetime(t["date"]) < c] if len(t) else t
        out.append({"做法": nm, "笔数": len(t), "相当于基准的仓位": len(t) / len(bt),
                    "单利年化(按全部资金)": fa(t, bdays),
                    "样本内": fa(tin, b_in), "样本外": fa(t.drop(tin.index) if len(t) else t, bdays - b_in),
                    "逐年胜过基准": "-" if nm == list(plans)[0] else f"{int((cmp_ > 0).sum())}/{len(cmp_)}",
                    "中位收益": t["收益率"].median() if len(t) else np.nan,
                    "胜率": float((t["收益率"] > 0).mean()) if len(t) else np.nan,
                    "复利倍数(每天一份)": d0["最终倍数(中位)"] if d0 is not None else np.nan,
                    "最大回撤(每天一份)": d0["最大回撤(中位)"] if d0 is not None else np.nan,
                    "复利倍数(分4份,最差起点)": d4["最终倍数(最差)"] if d4 is not None else np.nan,
                    "最大回撤(分4份,最差起点)": d4["最大回撤(最差)"] if d4 is not None else np.nan})
        if "curve" in sim:
            curves[nm] = sim["curve"]
    return {"split": split, "yearly_split": yearly_split,
            "plans": pd.DataFrame(out).set_index("做法"),
            "yearly": pd.DataFrame(yr_tab), "curves": pd.DataFrame(curves),
            "share": float(bt["下跌趋势"].mean())}


def macd_hist(panel: dict) -> pd.DataFrame:
    """日线 MACD(12,26,9) 柱子 = 2×(DIF−DEA)，与常见行情软件同口径（基于复权收盘价）。"""
    A = panel["adj_close"]
    ema = lambda x, p: x.ewm(span=p, adjust=False, min_periods=p).mean()
    dif = ema(A, 12) - ema(A, 26)
    return (2.0 * (dif - ema(dif, 9))).astype(np.float32)


def track_delayed(picks: pd.DataFrame, panel: dict, hold: int = 20,
                  trig: pd.DataFrame = None, max_wait: int = 20,
                  comm: float = 0.0003, stamp: float = 0.0005,
                  slip: float = 0.001) -> tuple:
    """
    延迟入场：股票上名单后，从选股日当天起等待触发条件（trig 为 True，用当天收盘数据判断），
    触发后下一个交易日开盘买入，持有 hold 个交易日后开盘卖出；max_wait 个交易日内一直没触发就放弃。
    trig=None 表示不等待（= 回测基准，结果与 track_fixed 逐笔一致）。
    同一只股票多次上名单、等到的是同一个触发日时，只买一次。
    另外记录持有期内的最大浮亏（期间最低）和最大浮盈（期间最高），用来衡量「拿着的过程」有多难受。
    返回 (成交表, 上名单总数, 放弃数)。
    """
    cal = panel["adj_close"].index
    n = len(cal)
    ci = {c: j for j, c in enumerate(panel["codes"])}
    pos = {d: i for i, d in enumerate(cal)}
    AO = panel["adj_open"].to_numpy(dtype=np.float32)
    AC = panel["adj_close"].to_numpy(dtype=np.float64)
    AH = panel["adj_high"].to_numpy(dtype=np.float64)
    AL = panel["adj_low"].to_numpy(dtype=np.float64)
    RO = panel["raw_open"].to_numpy(dtype=np.float32)
    TRD = panel["tradable"].to_numpy(dtype=bool)
    LU = panel["limit_up_open"].to_numpy(dtype=bool)
    LD = panel["limit_dn_open"].to_numpy(dtype=bool)
    T = (trig.reindex(index=cal, columns=panel["codes"]).fillna(False).to_numpy(dtype=bool)
         if trig is not None else None)
    cin, cout = comm + slip, comm + stamp + slip
    out, seen, total, gave_up = [], set(), 0, 0
    cols = ["date", "code"] + (["板块"] if "板块" in picks.columns else [])
    for p in picks[cols].itertuples(index=False):
        d0, code = p[0], p[1]
        sec = p[2] if len(p) > 2 else "-"
        i0, j = pos.get(d0), ci.get(code)
        if i0 is None or j is None:
            continue
        total += 1
        if T is None:
            t = i0
        else:
            t = None
            for q in range(i0, min(i0 + max_wait, n - 2) + 1):
                if T[q, j]:
                    t = q
                    break
            if t is None:
                if i0 + max_wait < n - 1:          # 等满了还没触发才算放弃；还在等的不算
                    gave_up += 1
                continue
        if (code, t) in seen:
            continue
        seen.add((code, t))
        b = t + 1
        if b >= n or not TRD[b, j] or LU[b, j] or not np.isfinite(AO[b, j]):
            continue
        e = b + hold
        while e < n and (not TRD[e, j] or LD[e, j] or not np.isfinite(AO[e, j])):
            e += 1
            if e - b > hold + 5:
                break
        if e >= n or not np.isfinite(AO[e, j]):
            continue
        p0 = float(AO[b, j])
        lo, hi = AL[b:e, j], AH[b:e, j]
        out.append({"date": cal[t], "选股日": d0, "code": code, "板块": sec,
                    "买入日": cal[b], "卖出日": cal[e], "等待天数": t - i0,
                    "买入价较选股日": p0 / AC[i0, j] - 1.0 if np.isfinite(AC[i0, j]) else np.nan,
                    "买入价(实际)": round(float(RO[b, j]), 2) if np.isfinite(RO[b, j]) else np.nan,
                    "收益率": float(AO[e, j]) * (1 - cout) / (p0 * (1 + cin)) - 1.0,
                    "持有交易日": e - b,
                    "期间最低": float(np.nanmin(lo) / p0 - 1.0) if np.isfinite(lo).any() else np.nan,
                    "期间最高": float(np.nanmax(hi) / p0 - 1.0) if np.isfinite(hi).any() else np.nan})
    return pd.DataFrame(out), total, gave_up


ENTRY_RULES = {
    "次日直接买（基准）": None,
    "等SKDJ上行才买：K>D且K在上升（已经上行就次日买）": "skdj",
    "等MACD柱连续2天回升才买（已经在回升就次日买）": "macd",
}


def entry_delay_test(pk: pd.DataFrame, panel: dict, dates: List[pd.Timestamp],
                     KDF: pd.DataFrame, DDF: pd.DataFrame, MH: pd.DataFrame,
                     hold: int = 20, every: int = 1, max_wait: int = 20,
                     cut: str = "2023-01-01", **kw) -> dict:
    """同一份名单，只改「什么时候买」，比较收益、持有过程中的浮亏和账户回撤。"""
    trig = {"skdj": (KDF > DDF) & (KDF > KDF.shift(1)),
            "macd": (MH > MH.shift(1)) & (MH.shift(1) > MH.shift(2))}
    c = pd.Timestamp(cut)
    res, curves, yrs = [], {}, {}
    base_nm, base_days, b_in, b_y = None, None, None, None

    def fa(t, den):
        return float(t["收益率"].sum() / den * 244) if den else np.nan

    for nm, key in ENTRY_RULES.items():
        tr, total, gave = track_delayed(pk, panel, hold, trig.get(key) if key else None,
                                        max_wait, **kw)
        if not len(tr):
            continue
        tr["年"] = pd.to_datetime(tr["date"]).dt.year
        if base_nm is None:
            base_nm = nm
            base_days = float(tr["持有交易日"].sum())
            b_in = float(tr.loc[tr["date"] < c, "持有交易日"].sum())
            b_y = tr.groupby("年")["持有交易日"].sum()
        yy = pd.Series({y: fa(tr[tr["年"] == y], b_y[y]) for y in b_y.index})
        yrs[nm] = yy
        cmp_ = (yy - yrs[base_nm]).dropna() if nm != base_nm else pd.Series(dtype=float)
        sim = account_sim(tr, dates, hold, every, tranches=(4, 0))
        sm = sim.get("summary", pd.DataFrame())
        d0 = sm.iloc[-1] if len(sm) else None
        d4 = sm.iloc[0] if len(sm) else None
        ins = tr[tr["date"] < c]
        res.append({"入场方式": nm, "实际买入笔数": len(tr),
                    "放弃比例": gave / total if total else np.nan,
                    "平均等待天数": tr["等待天数"].mean(),
                    "买入价较选股日(中位)": tr["买入价较选股日"].median(),
                    "平均收益": tr["收益率"].mean(), "中位收益": tr["收益率"].median(),
                    "胜率": float((tr["收益率"] > 0).mean()),
                    "持有中最大浮亏(中位)": tr["期间最低"].median(),
                    "浮亏曾超20%的比例": float((tr["期间最低"] <= -0.20).mean()),
                    "单利年化(按全部资金)": fa(tr, base_days),
                    "样本内": fa(ins, b_in), "样本外": fa(tr.drop(ins.index), base_days - b_in),
                    "逐年胜过基准": "-" if nm == base_nm else f"{int((cmp_ > 0).sum())}/{len(cmp_)}",
                    "复利倍数(每天一份)": d0["最终倍数(中位)"] if d0 is not None else np.nan,
                    "最大回撤(每天一份)": d0["最大回撤(中位)"] if d0 is not None else np.nan,
                    "最大回撤(分4份,最差起点)": d4["最大回撤(最差)"] if d4 is not None else np.nan})
        if "curve" in sim:
            curves[nm] = sim["curve"]
    if not res:
        return {}
    return {"summary": pd.DataFrame(res).set_index("入场方式"),
            "yearly": pd.DataFrame(yrs), "curves": pd.DataFrame(curves)}


BIAS_GROUPS = ["高出30日线10%以内", "高出10%-30%", "高出30%以上"]


def ma30_bias(panel: dict) -> pd.DataFrame:
    """股价高出（复权）30日均线的幅度：收盘价 ÷ 30日均线 − 1。只用当天及以前的数据。"""
    A = panel["adj_close"]
    # 停牌日收盘价为空：30天窗口里至少有20个有效收盘价就计算，避免偶尔停牌让整段变成空值
    return (A / A.rolling(30, min_periods=20).mean() - 1.0).astype(np.float32)


def bias_split(pk: pd.DataFrame, panel: dict, BIAS: pd.DataFrame, pool_tr: pd.DataFrame = None,
               hold: int = 20, every: int = 1, **kw) -> dict:
    """
    乖离的干净划分：同一批成交，按选股日当天股价高出30日线的幅度分三组，不做替补。
    除了20日收益，还看持有过程：持有中最大浮亏、浮亏曾超20%的比例、最惨10%的结果。
    """
    tr, _, _ = track_delayed(pk, panel, hold, None, 20, **kw)   # 与回测逐笔一致，另带期间最低
    if not len(tr):
        return {}
    b = []
    for d_, c_ in zip(tr["date"], tr["code"]):
        try:
            b.append(float(BIAS.at[d_, c_]))
        except Exception:
            b.append(np.nan)
    tr["乖离"] = np.array(b, dtype=float)
    tr = tr[np.isfinite(tr["乖离"])].copy()
    tr["分组"] = np.select([tr["乖离"] < 0.10, tr["乖离"] < 0.30, tr["乖离"] >= 0.30],
                          BIAS_GROUPS, default=None)
    if pool_tr is not None and len(pool_tr):
        pm = pool_tr.dropna(subset=["收益率"]).groupby("date")["收益率"].mean()
        tr["同期超额"] = tr["收益率"] - tr["date"].map(pm)
    val = "同期超额" if "同期超额" in tr.columns else "收益率"
    tr["年"] = pd.to_datetime(tr["date"]).dt.year
    lag = max(1, int(np.ceil(hold / max(every, 1))))
    rows = []
    for g in BIAS_GROUPS:
        sub = tr[tr["分组"] == g]
        if not len(sub):
            continue
        day = sub.groupby("date")[val].mean().dropna().sort_index()
        yr = sub.groupby("年")[val].mean()
        rows.append({"分组": g, "笔数": len(sub), "占比": len(sub) / len(tr),
                     "平均收益": sub["收益率"].mean(), "中位收益": sub["收益率"].median(),
                     "胜率": float((sub["收益率"] > 0).mean()),
                     "最惨10%": sub["收益率"].quantile(0.10),
                     "亏损超20%的比例": float((sub["收益率"] <= -0.20).mean()),
                     "持有中最大浮亏(中位)": sub["期间最低"].median(),
                     "浮亏曾超20%的比例": float((sub["期间最低"] <= -0.20).mean()),
                     val: sub[val].mean(),
                     f"t(重叠修正,{val})": newey_west_t(day, lag) if len(day) >= 12 else np.nan,
                     "为正年数": f"{int((yr > 0).sum())}/{len(yr)}"})
    summ = pd.DataFrame(rows).set_index("分组")
    yearly = tr.pivot_table(index="年", columns="分组", values=val, aggfunc="mean")
    yearly = yearly[[c for c in BIAS_GROUPS if c in yearly.columns]]
    big_y = tr.assign(大亏=(tr["收益率"] <= -0.20)).pivot_table(
        index="年", columns="分组", values="大亏", aggfunc="mean")
    big_y = big_y[[c for c in BIAS_GROUPS if c in big_y.columns]]
    n_y = tr.pivot_table(index="年", columns="分组", values="收益率", aggfunc="size")
    n_y = n_y.reindex(columns=yearly.columns)
    # 两头对比：同一天两组都有成交的日子，差值的重叠修正 t
    verdict = ""
    hi, lo = tr[tr["分组"] == BIAS_GROUPS[2]], tr[tr["分组"] == BIAS_GROUPS[0]]
    if len(hi) >= 30 and len(lo) >= 30:
        dd = (hi.groupby("date")[val].mean() - lo.groupby("date")[val].mean()).dropna().sort_index()
        t_d = newey_west_t(dd, lag) if len(dd) >= 30 else np.nan
        yh, yl = hi.groupby("年")[val].agg(["mean", "size"]), lo.groupby("年")[val].agg(["mean", "size"])
        j = yh.join(yl, lsuffix="_h", rsuffix="_l", how="inner")
        j = j[(j["size_h"] >= 20) & (j["size_l"] >= 20)]
        diff = j["mean_h"] - j["mean_l"]
        verdict = (f"「高出30%以上」减「10%以内」：平均 {hi[val].mean() - lo[val].mean():+.2%}；"
                   f"两组都有足够笔数的 {len(diff)} 年里，{int((diff > 0).sum())} 年为正；"
                   + (f"同一天两组都有成交的 {len(dd)} 天，差值 t={t_d:.2f}。" if pd.notna(t_d)
                      else "同一天两组都有成交的日子太少，算不了差值 t。"))
    dist = tr["乖离"].describe(percentiles=[.1, .25, .5, .75, .9]).to_frame("选中时高出30日线").T
    return {"summary": summ, "yearly": yearly, "big_y": big_y, "n_y": n_y,
            "verdict": verdict, "dist": dist, "val": val}


def stock_snapshot(mine: pd.DataFrame, panel: dict, BIAS: pd.DataFrame, KDF: pd.DataFrame,
                   DDF: pd.DataFrame, MH: pd.DataFrame, names: dict, live_pk: pd.DataFrame = None,
                   hold: int = 20, **kw) -> pd.DataFrame:
    """逐只查看：给定「日期 代码」，列出当天收盘时的乖离、SKDJ、MACD、20日涨幅，以及次日买入持有到期的结果。"""
    if mine is None or not len(mine):
        return pd.DataFrame()
    cal = panel["adj_close"].index
    codes = list(panel["codes"])
    A = panel["adj_close"]
    rows = []
    for _, m in mine.iterrows():
        cc = next((c for c in codes if c[:6] == m["代码6位"]), None)
        if cc is None:
            rows.append({"代码": m["代码6位"], "日期": m["日期"].strftime("%Y-%m-%d"), "说明": "不在股票池里"})
            continue
        k = cal.searchsorted(m["日期"], side="right") - 1
        if k < 0:
            continue
        d = cal[k]
        one = pd.DataFrame({"date": [d], "code": [cc]})
        tr, _, _ = track_delayed(one, panel, hold, None, 20, **kw)
        kv, dv = KDF.at[d, cc], DDF.at[d, cc]
        mh0 = MH.at[d, cc]
        mh1 = MH[cc].iloc[k - 1] if k >= 1 else np.nan
        bv = BIAS.at[d, cc]
        r20 = A[cc].iloc[k] / A[cc].iloc[k - 20] - 1.0 if k >= 20 else np.nan
        on_list = ""
        if live_pk is not None and len(live_pk):
            on_list = "是" if ((live_pk["date"] == d) & (live_pk["code"] == cc)).any() else "否"
        g = ("" if not np.isfinite(bv) else BIAS_GROUPS[0] if bv < 0.10
             else BIAS_GROUPS[1] if bv < 0.30 else BIAS_GROUPS[2])
        rows.append({"代码": cc, "名称": names.get(cc, ""), "日期": d.strftime("%Y-%m-%d"),
                     "当天在名单上": on_list,
                     "收盘价": round(float(panel["raw_close"].at[d, cc]), 2),
                     "高出30日线": float(bv) if np.isfinite(bv) else np.nan, "乖离分组": g,
                     "20日涨幅": float(r20) if np.isfinite(r20) else np.nan,
                     "SKDJ K": round(float(kv), 1) if np.isfinite(kv) else np.nan,
                     "SKDJ D": round(float(dv), 1) if np.isfinite(dv) else np.nan,
                     "MACD柱": round(float(mh0), 3) if np.isfinite(mh0) else np.nan,
                     "MACD柱较前日": ("变长/回升" if (np.isfinite(mh0) and np.isfinite(mh1) and mh0 > mh1)
                                  else "缩短/下降" if (np.isfinite(mh0) and np.isfinite(mh1)) else ""),
                     "次日买持有到期收益": float(tr["收益率"].iloc[0]) if len(tr) else np.nan,
                     "持有中最大浮亏": float(tr["期间最低"].iloc[0]) if len(tr) else np.nan})
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


def cap_annual(tr: pd.DataFrame) -> float:
    """
    按资金占用天数折算的年化（单利）：总收益 ÷ 总持有交易日 × 244。
    假设资金一直满仓轮动——卖出后第二天就买入新名单。
    每笔平均收益不能乘以笔数：每天买几只、每只拿20天，同时在手里的有几十只。
    """
    if tr is None or not len(tr) or "持有交易日" not in tr.columns:
        return np.nan
    hd = tr["持有交易日"].clip(lower=1).sum()
    return float(tr["收益率"].sum() / hd * 244) if hd > 0 else np.nan


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
                 min_members: int = 5, cooldown: int = 5,
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
                               "S1_板块内最强", "最强", cooldown=cooldown,
                               per_sec_cap=per_sec_cap, min_members=min_members)
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
        _cap = ((f"|每板块≤{per_sec_cap}" if per_sec_cap > 0 else "|板块不限")
                + f"|冷却{cooldown}日")
        picked.append({"年": y,
                       "选中配置": f"{bcfg[0]}|{bcfg[1]}板块|{bcfg[2]}只|{bcfg[3]}日{_cap}",
                       "历史t": best, "当年笔数": len(cur),
                       "当年平均收益": cur["收益率"].mean(),
                       "当年资金年化": cap_annual(cur),
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
            "年数": len(yr), "逐年为正": int((yr > 0).sum()),
            "资金年化": cap_annual(wf),
            "平均持有日": float(wf["持有交易日"].mean()) if "持有交易日" in wf.columns else np.nan}


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
            "资金年化(近似)": cap_annual(d),
            "聚类t(朴素)": day.mean() / se if se > 1e-12 else np.nan}


APP_VERSION = "2026-09-25 · 乖离版"


def main():
    st.set_page_config(page_title="板块轮动选股", layout="wide")
    ss = st.session_state
    ss.setdefault("panel", None)
    st.title("板块轮动选股")
    st.caption(f"程序版本：**{APP_VERSION}**（如果这里显示的不是最新版本，说明网页还在运行旧代码，"
               "需要在 GitHub 覆盖 app.py 后，于 Manage app 里 Reboot）")
    st.caption("先选最强板块，再从板块内选动量最高的股票。"
               "科技/军工/新能源/机器人　·　流通市值 50-1000 亿　·　股价 10 元以上")

    with st.sidebar:
        st.caption(f"程序版本：{APP_VERSION}")
        token = st.text_input("Tushare Token", type="password",
                              value=os.environ.get("TUSHARE_TOKEN", ""))
        st.caption("默认值：板块60日动量 · 3个板块 · 每次5只 · 每板块最多2只 · 持有20日 · "
                   "每1日选一次 · 冷却5日 · 持有到期不止损（滚动前推六年都选中这个配置，"
                   "每次5只+冷却5日是试过的组合里按年t最高、回撤最小的）。")
        top_sec = st.slider("选几个板块", 1, 5, 3,
                            help="每板块最多2只时：每次5只 = 第1、2板块各2只 + 第3板块1只；"
                                 "每次3只 = 第1板块2只 + 第2板块1只。")
        top_n = st.slider("每次选几只", 1, 5, 5)
        hold = st.slider("持有交易日", 3, 30, 20, help="持有到期、次日开盘买、到期日开盘卖，不设止盈止损。")
        cap = st.slider("每个板块最多取几只（0=不限）", 0, 5, 2,
                        help="候选按板块顺序取。设为 2：最强板块最多取2只，剩下的名额给下一个板块。"
                             "改了这个，前面的回测结论都要重跑。")
        st.markdown("**名单轮换**")
        every = st.slider("每几个交易日选一次", 1, 10, 1,
                          help="回测和「今日候选」用同一个日程。1 = 每天都是选股日。")
        cool = st.slider("同一只股票冷却几个交易日", 1, 20, 5,
                         help="入选后这么多个交易日内不再入选，名额顺延给板块内下一名——这就是轮换。"
                              "1 日选一次 + 冷却 5 日 = 连续5天的名单没有重复的股票。")
        if cool <= every:
            st.caption(f"⚠️ 冷却 {cool} ≤ 选股间隔 {every}，**冷却不起作用**："
                       "板块和排名不变时，连续选股日名单一模一样。")
        else:
            _k = int(np.ceil(cool / every))
            st.caption(f"同一只股票每 **{_k}** 个选股日最多入选一次。"
                       + ("板块和排名不变时，名单在「前几名」和「后几名」之间**交替**。"
                          if _k == 2 else
                          f"板块和排名不变时，名单要轮 {_k} 组才回到第1名，挖得较深。"))
        with st.expander("其他设置"):
            start = st.date_input("数据起始（选「验证」模式时生效）", dt.date(2018, 1, 1))
            end = st.date_input("数据结束", dt.date.today())
            min_mem = st.slider("板块最少成分股", 3, 20, 5)
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
            st.caption(f"行情缓存 {_cn} 只股票 · {_cmb:.0f}MB · "
                       f"{CACHE_TTL_HOURS} 小时后自动失效")
        else:
            st.caption("行情缓存为空")
        if st.button("清除缓存并重新下载", use_container_width=True):
            n = clear_day_cache()
            for kk in ("panel", "sec", "res", "nz", "sigres", "wf", "kres",
                       "ksplit", "kbk", "elig", "elig_key", "kdf", "ddf",
                       "rankres", "live", "live_key", "hist_key", "hist", "dtm", "dtm_key", "skdj", "mh", "mh_key", "delay",
                       "biasdf", "bias_key", "bias", "biasrule"):
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
                   "elig", "elig_key", "kmask_key", "sec_mm", "rankres", "live", "live_key",
                   "hist_key", "hist", "dtm", "dtm_key", "skdj", "mh", "mh_key", "delay",
                       "biasdf", "bias_key", "bias", "biasrule"):
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
                os.makedirs(PX_DIR, exist_ok=True)
                _t = os.path.join(PX_DIR, ".wtest")
                with open(_t, "wb") as _f:
                    _f.write(b"1")
                os.remove(_t)
                _cached = len([x for x in os.listdir(PX_DIR) if x.endswith(".pkl")])
                st.write(f"   磁盘缓存可用，已有 {_cached} 只股票的缓存")
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
            sectors = build_sector_map(uni, panel)
            if not sectors:
                st.error("没能建立板块映射——二级行业数据缺失。"); st.stop()
            R, IDX, cnt, AV = build_sector_index(panel, ss["elig"], sectors, min_mem)
            amt = pd.DataFrame({s: panel["amount"][c].where(ss["elig"][c]).sum(axis=1)
                                for s, c in sectors.items()})
            kdf, ddf = daily_kd(panel)
            ss["sec"] = (sectors, R, IDX, cnt,
                         sector_factors(R, IDX, amt, AV), AV)
            ss["kdf"], ss["ddf"] = kdf, ddf
            ss["elig_key"] = dkey
            ss.pop("nz", None); ss.pop("sigres", None)
            ss.pop("res", None); ss.pop("wf", None)
            gc.collect()
    elig = ss["elig"]
    sectors, R, IDX, cnt, SF, AV = ss["sec"]
    KDF, DDF = ss["kdf"], ss["ddf"]
    if ss.get("dtm_key") != dkey or ss.get("dtm") is None:
        ss["dtm"] = skdj_downtrend_mask(KDF, DDF)
        ss["dtm_key"] = dkey
    DTM = ss["dtm"]
    if ss.get("mh_key") != dkey or ss.get("mh") is None:
        ss["mh"] = macd_hist(panel)
        ss["mh_key"] = dkey
    MH = ss["mh"]
    if ss.get("bias_key") != dkey or ss.get("biasdf") is None:
        ss["biasdf"] = ma30_bias(panel)
        ss["bias_key"] = dkey
    BIAS = ss["biasdf"]
    kw = dict(comm=comm, stamp=0.0005, slip=slip)
    dates = list(panel["cal"][130::every])
    st.caption(f"数据截至 **{panel['cal'][-1]:%Y-%m-%d}**　·　"
               f"{len(panel['codes'])} 只 × {len(panel['cal'])} 个交易日　·　"
               f"{len(sectors)} 个板块")
    # 默认信号依据滚动前推：最近一次滚动前推六年都选中「板块60日动量」。
    DEF_SIG = "板块60日动量" if "板块60日动量" in SF else list(SF)[0]

    t1, t2, t3, t4 = st.tabs(["① 板块信号", "② 主回测", "③ 专项检验", "④ 今日候选"])

    # ---------------- ① 板块信号 ----------------
    with t1:
        st.markdown("### 板块层面的信号有没有预测力")
        st.caption("**换新数据后跑一次就够，平时不用动。** 按信号把板块分四组，"
                   "看未来 3/5/8/15 天板块指数的表现——不涉及选股。")
        with st.expander("地基：板块指数比个股降噪多少（也是一次性的）"):
            if st.button("运行降噪检验"):
                with st.spinner("计算中…"):
                    ss["nz"] = sector_noise_check(panel, elig, sectors, min_mem)
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
                       "早期回测里 20日动量曾好于 60日动量，"
                       "而最近一次滚动前推六年都选中 60日动量。\n\n"
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
        st.caption(f"默认「{DEF_SIG}」不是挑的——是最近一次滚动前推六年都选中的配置。"
                   "**不要再逐个试信号挑最好的**，那会让后面所有检验失效。")
        st.markdown("四个方案同时跑：两层 / 随机板块 / 不分板块 / 全池随机。"
                    "**两层减随机板块 = 板块层的净贡献**，这个对照能干净分离"
                    "「选板块」和「板块内选股」两层的功劳。")

        if st.button("运行对照实验", type="primary", use_container_width=True):
            bar = st.progress(0.0)
            plans = [
                ("两层：最强板块 + " + srule,
                 lambda: sector_then_stock(panel, elig, sectors, SF[sig], dates,
                                           top_sec, top_n, srule, "最强", cooldown=cool,
                                           kdf=KDF, per_sec_cap=cap, min_members=min_mem)),
                ("对照A：随机板块 + " + srule,
                 lambda: sector_then_stock(panel, elig, sectors, SF[sig], dates,
                                           top_sec, top_n, srule, "随机", cooldown=cool,
                                           kdf=KDF, per_sec_cap=cap, min_members=min_mem)),
                ("对照B：不分板块，全池 " + srule,
                 lambda: flat_stock_pick(panel, elig, dates, top_n, srule, cooldown=cool)),
                ("对照C：全池随机",
                 lambda: flat_stock_pick(panel, elig, dates, top_n, "S3_板块内随机",
                                         cooldown=cool)),
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
            ss["res_lab"] = (f"{top_sec}板块｜{top_n}只｜每板块≤{cap or '不限'}｜持有{hold}日｜"
                             f"每{every}日选｜冷却{cool}日")
            ss.pop("wf", None); ss.pop("ksplit", None); ss.pop("skdj", None); ss.pop("delay", None)
            ss.pop("bias", None); ss.pop("biasrule", None)
            bar.empty(); gc.collect()

        if ss.get("res"):
            df, keep, pks, sig_, sr_ = ss["res"]
            st.caption(f"这张表的口径：{ss.get('res_lab', '')}")
            st.dataframe(df.style.format({"笔数": "{:.0f}", "平均收益": "{:+.2%}",
                                          "中位收益": "{:+.2%}", "胜率": "{:.1%}",
                                          "资金年化(近似)": "{:+.1%}",
                                          "聚类t(朴素)": "{:.2f}"})
                         .background_gradient(subset=["资金年化(近似)"], cmap="RdYlGn"),
                         use_container_width=True)
            st.caption("**资金年化(近似)** = 总收益 ÷ 总持有交易日 × 244，单利，假设卖出后马上买入新名单、"
                       "资金一直满仓。它才是能直接理解的年收益——**每笔平均收益不能乘笔数**："
                       "每天买3只、每只拿20天，同时在手里的有约60只，资金被分成60份。"
                       "**聚类t(朴素)** 没做重叠修正，持有20日、每日选股时大约要除以3。")
            try:
                two, ra, fb = df.iloc[0], df.iloc[1], df.iloc[2]
                st.metric("板块层净贡献（两层 − 随机板块）",
                          f"{two['平均收益']-ra['平均收益']:+.3%} / 笔",
                          f"资金年化差 {two['资金年化(近似)']-ra['资金年化(近似)']:+.1%}")
                rt = (comm * 2 + 0.0005 + slip * 2)
                st.caption(f"单次往返成本 {rt:.2%}；持有 {hold} 日 → 年换手 "
                           f"{244/hold:.0f} 次 → 年成本 {rt*244/hold:.1%}。"
                           f"**每笔平均收益要超过 {rt:.2%} 才算真有边际。**")
            except Exception:
                pass

            st.divider()
            st.markdown("### 账户净值模拟（真实复利）")
            st.caption("上面的「资金年化」是单利，会高估。这里按实盘方式模拟：资金分成几份、分批入场，"
                       f"每份买当天名单、持有 {hold} 日卖出后，收回的钱马上买当天新名单，一路复利。"
                       "不同的开始日期会得到不同结果，都算出来给中位和最差。")
            _acct = account_sim(keep[list(keep)[0]], dates, hold, every, tranches=(1, 4, 0))
            if _acct:
                st.dataframe(_acct["summary"].style.format(
                    {"起始日数": "{:.0f}", "最终倍数(中位)": "×{:.2f}", "最终倍数(最差)": "×{:.2f}",
                     "最终倍数(最好)": "×{:.2f}", "复利年化(中位)": "{:+.1%}",
                     "最大回撤(中位)": "{:.0%}", "最大回撤(最差)": "{:.0%}"}),
                    use_container_width=True)
                if "curve" in _acct:
                    st.line_chart(_acct["curve"].rename("账户净值（每个选股日投一份）"), height=220)
                    _rc = _acct["recover"]
                    st.caption(f"最大回撤 {_acct['dd'].min():.0%}：从 {_acct['peak']:%Y-%m-%d} 的高点，"
                               f"跌到 {_acct['trough']:%Y-%m-%d} 的低点，"
                               + (f"{_rc:%Y-%m-%d} 才回到原高点。" if _rc is not None else "至今没有回到原高点。")
                               + "　逐年（复利）：" + "　".join(
                                   f"{y} {v:+.0%}" for y, v in _acct["yearly"].items()))
                st.info("**分批入场去掉的是「开始时点」的运气**（看「最差」两列），"
                        "去不掉策略本身的回撤（看每天一份那一行）。"
                        "**投入这个策略的钱，要按能承受「最大回撤(最差)」那一列来定。**"
                        "净值按每份到期卖出时结算，持仓中的浮亏没算进去，真实回撤会更深一些。")

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
                f"**{cap if cap else '不限'}** 只 · 每 **{every}** 日选一次 · "
                f"冷却 **{cool}** 日 · 持有到期不止损 · 成本设置。\n\n"
                "所以改「选几个板块」和「持有交易日」对这里没影响（它自己会搜）；"
                "改「每次选几只」「每板块最多几只」「每几日选一次」「冷却」会改变结果。")
            st.error("**每年年初只用截至上一年底的数据挑配置，再用它跑这一年。** "
                     "全样本上挑一个最优配置再看它的「样本外」，等于用样本外做了选择，"
                     "那个数字不算数。")
            if st.button("运行滚动前推", type="primary"):
                bar3 = st.progress(0.0)
                picked, wf = walk_forward(
                    panel, elig, sectors, SF, dates,
                    top_secs=(2, 3), top_ns=(top_n,), holds=(15, 20),
                    start_year=2021, per_sec_cap=cap, min_members=min_mem, cooldown=cool,
                    progress=lambda p, n2: bar3.progress(p, text=n2), **kw)
                ss["wf"] = (picked, wf); bar3.empty(); gc.collect()
            if ss.get("wf"):
                picked, wf = ss["wf"]
                if not len(wf):
                    st.warning("样本不足。")
                else:
                    s5 = wf_summary(wf, hold_days=hold, step_days=every)
                    m5 = st.columns(5)
                    m5[0].metric("资金年化(近似)", f"{s5['资金年化']:+.1%}",
                                 f"平均持有 {s5['平均持有日']:.1f} 日")
                    m5[1].metric("每笔平均收益", f"{s5['平均收益']:+.2%}")
                    m5[2].metric("胜率", f"{s5['胜率']:.1%}")
                    m5[3].metric("t(按年，最严格)", f"{s5['t(按年)']:.2f}",
                                 f"逐年为正 {s5['逐年为正']}/{s5['年数']}")
                    m5[4].metric("笔数", f"{s5['笔数']}")
                    st.caption("**每笔平均收益不能乘以笔数。** 每天买几只、每只拿几十天，"
                               "同时在手的有几十只，资金被分成几十份。资金年化 = 总收益 ÷ 总持有交易日 × 244，"
                               "单利、假设资金一直满仓轮动。你实际只拿 1-3 只，结果会比它分散得多。")
                    st.caption(f"对照：按日朴素 t={s5['t(按日,朴素)']:.2f}，"
                               f"按日重叠修正 t={s5['t(按日,重叠修正)']:.2f}。"
                               "**该看按年**——每年重新挑一次配置，年与年之间才真正独立。")
                    st.dataframe(picked.style.format(
                        {"历史t": "{:.2f}", "当年笔数": "{:.0f}", "当年资金年化": "{:+.1%}",
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
                                  "持有交易日", "买入K",
                                  "买入价(复权)", "卖出价(复权)"]
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
                              "每几日选一次": every, "冷却交易日": cool,
                              "每板块最多几只": cap, "板块数": len(sectors),
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
                        tb["06_执行时机_龙头领跑天数"] = ss["ksplit"]
                    if ss.get("nz") is not None:
                        tb["08_降噪检验"] = ss["nz"]
                    if ss.get("bias"):
                        tb["20_乖离划分_汇总"] = ss["bias"]["summary"]
                        tb["20_乖离划分_逐年超额"] = ss["bias"]["yearly"]
                        tb["20_乖离划分_逐年大亏比例"] = ss["bias"]["big_y"]
                    if ss.get("biasrule"):
                        tb["21_乖离规则_三种做法"] = ss["biasrule"]["plans"]
                        tb["21_乖离规则_逐年"] = ss["biasrule"]["yearly"].T
                    if ss.get("delay"):
                        tb["19_延迟入场_对比"] = ss["delay"]["summary"]
                        tb["19_延迟入场_逐年"] = ss["delay"]["yearly"].T
                    if ss.get("skdj"):
                        tb["17_SKDJ下跌趋势_划分"] = ss["skdj"]["split"]
                        tb["17_SKDJ下跌趋势_三种做法"] = ss["skdj"]["plans"]
                        tb["17_SKDJ下跌趋势_逐年"] = ss["skdj"]["yearly"].T
                    try:
                        _ac = account_sim(keep[list(keep)[0]], dates, hold, every, tranches=(1, 4, 0))
                        if _ac:
                            tb["18_账户净值模拟"] = _ac["summary"]
                    except Exception:
                        pass
                    if ss.get("rankres"):
                        _rr = ss["rankres"]
                        for _k2, _nm2 in (("summary", "09_排名分档_汇总"),
                                          ("pair", "10_排名分档_配对差"),
                                          ("yearly", "11_排名分档_逐年")):
                            if _k2 in _rr and len(_rr[_k2]):
                                tb[_nm2] = _rr[_k2]
                    ss["zipb"] = export_all(tb)
                    ss["zipn"] = f"sector_{dt.datetime.now():%Y%m%d_%H%M}.zip"
                    gc.collect()
            if ss.get("zipb"):
                st.download_button(f"下载 {ss['zipn']}（{len(ss['zipb'])/1024:.0f} KB）",
                                   ss["zipb"], ss["zipn"], "application/zip",
                                   type="primary", use_container_width=True)

    # ---------------- ③ 位置诊断 ----------------
    with t3:
        st.markdown("### 板块内排名分档：轮换到第3、4名有没有代价")
        st.caption(f"每个选股日取最强的 **{top_sec}** 个板块，把板块内按20日涨幅排出的"
                   "第1-2、3-4、5-6、7-10名**分开**，各自持有 "
                   f"**{hold}** 个交易日。不设冷却、不做替补——同一天、同一板块比较，"
                   "唯一的差别就是名次。用侧边栏的「选几个板块」「持有交易日」「每几个交易日选一次」。")
        sig3 = st.selectbox("板块信号", list(SF), key="s3", index=list(SF).index(DEF_SIG))
        if sig3 not in SF:
            sig3 = DEF_SIG
        if st.button("运行排名分档检验", type="primary"):
            with st.spinner("逐档成交中（每只股票都要算一遍，1 日选一次时稍慢）…"):
                ss["rankres"] = in_sector_rank_test(
                    panel, elig, sectors, SF[sig3], dates, top_sec=top_sec, hold=hold,
                    step_days=every, min_members=min_mem, **kw)
                ss["rankres_lab"] = f"{sig3}｜{top_sec}板块｜持有{hold}日｜每{every}日"
                gc.collect()
        _rr = ss.get("rankres")
        if _rr is not None and not _rr:
            st.warning("样本不足，没有结果。")
        elif _rr:
            st.caption(f"参数：{ss.get('rankres_lab', '')}")
            if len(_rr["summary"]):
                st.dataframe(_rr["summary"].style.format(
                    {"笔数": "{:.0f}", "平均收益": "{:+.2%}", "中位收益": "{:+.2%}",
                     "胜率": "{:.1%}", "聚类t(重叠修正)": "{:.2f}"})
                    .background_gradient(subset=["中位收益"], cmap="RdYlGn"),
                    use_container_width=True)
            if len(_rr["pair"]):
                st.markdown("**配对差（同一天、同一板块，减去第1-2名）——主要看这张**")
                st.dataframe(_rr["pair"].style.format(
                    {"配对样本": "{:.0f}", "平均差": "{:+.2%}", "中位差": "{:+.2%}",
                     "t(重叠修正)": "{:.2f}"})
                    .background_gradient(subset=["t(重叠修正)"], cmap="RdYlGn",
                                         vmin=-3, vmax=3),
                    use_container_width=True)
                _i34 = [i for i in _rr["pair"].index if str(i).startswith("第3-4名")]
                _p34 = _rr["pair"].loc[_i34[0]] if _i34 else _rr["pair"].iloc[0]
                _t34 = _p34["t(重叠修正)"]
                if pd.notna(_t34) and _t34 <= -2:
                    st.error(f"**{_p34.name}：t={_t34:.2f}，显著更差。** "
                             "往后轮换是有代价的——每天换名单等于每隔一天买一次更差的票。"
                             "那就接受名单重复（冷却 ≤ 选股间隔），不要为了每天不同往后挖。")
                elif pd.notna(_t34) and _t34 >= 2:
                    st.success(f"**{_p34.name}：t={_t34:.2f}，第3-4名反而更好。** "
                               "轮换没有代价，但这个方向和「动量」本身相反，先看逐年是否稳定再信。")
                elif pd.notna(_t34) and _t34 <= -1.5:
                    st.warning(f"**{_p34.name}：t={_t34:.2f}，接近显著地更差**"
                               f"（平均差 {_p34['平均差']:+.2%}，中位差 {_p34['中位差']:+.2%}，"
                               f"差为正的年份 {_p34['差为正的年份']}）。"
                               "不能说轮换没代价。更稳妥的是接受名单重复，或者把持有期、板块数换一组再看方向是否一致。")
                elif pd.notna(_t34):
                    st.info(f"**{_p34.name}：t={_t34:.2f}，看不出差别。** "
                            "轮换到第3-4名在统计上不花代价。**但「看不出差别」不等于「证明没差别」**"
                            f"——平均差 {_p34['平均差']:+.2%}，差为正的年份 {_p34['差为正的年份']}，"
                            "两个一起看。确认后，再到第②页用新的「选股间隔/冷却」跑对照和滚动前推。")
            if len(_rr["yearly"]):
                st.markdown("**逐年平均收益**")
                st.dataframe(_rr["yearly"].style.format("{:+.2%}")
                             .background_gradient(cmap="RdYlGn", axis=None),
                             use_container_width=True)
            st.caption("你只拿 1-3 只，抓到右尾的概率低，**中位收益和胜率比平均收益更贴近真实体验**。")
        st.divider()

        st.markdown("### 乖离：选中时股价高出30日线多少")
        st.caption("第②页「两层」方案的**同一批成交**，按选股日收盘时股价高出30日均线的幅度分三组"
                   "（10%以内 / 10%-30% / 30%以上），不做替补。除了20日收益，还看持有过程："
                   "持有中最大浮亏、浮亏曾超20%的比例、最惨10%的结果。分组界线是事先定的。")
        if not ss.get("res"):
            st.info("先到「② 主回测」跑一次对照实验。")
        else:
            if st.button("运行乖离划分", type="primary"):
                _df8, _keep8, _pks8, _, _ = ss["res"]
                _b8 = list(_keep8)[0]
                _pool8 = next((v for k, v in _keep8.items() if str(k).startswith("对照C")), None)
                with st.spinner("划分同一批成交…"):
                    ss["bias"] = bias_split(_pks8[_b8], panel, BIAS, _pool8, hold=hold, every=every, **kw)
                    ss["bias_lab"] = ss.get("res_lab", "")
                    gc.collect()
            _bs = ss.get("bias")
            if _bs:
                st.caption(f"口径：{ss.get('bias_lab', '')}")
                _sb = _bs["summary"]
                st.dataframe(_sb.style.format(
                    {"笔数": "{:.0f}", "占比": "{:.1%}", "平均收益": "{:+.2%}", "中位收益": "{:+.2%}",
                     "胜率": "{:.1%}", "最惨10%": "{:+.1%}", "亏损超20%的比例": "{:.1%}",
                     "持有中最大浮亏(中位)": "{:+.1%}", "浮亏曾超20%的比例": "{:.1%}",
                     "同期超额": "{:+.2%}", "收益率": "{:+.2%}",
                     **{c: "{:.2f}" for c in _sb.columns if c.startswith("t(")}})
                    .background_gradient(subset=["亏损超20%的比例"], cmap="RdYlGn_r"),
                    use_container_width=True)
                if _bs["verdict"]:
                    st.caption(_bs["verdict"])
                st.dataframe(_bs["dist"].style.format("{:.1%}", subset=[c for c in _bs["dist"].columns if c != "count"]),
                             use_container_width=True)
                with st.expander("逐年：同期超额、亏损超20%的比例、笔数"):
                    st.dataframe(_bs["yearly"].style.format("{:+.2%}")
                                 .background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
                    st.dataframe(_bs["big_y"].style.format("{:.0%}")
                                 .background_gradient(cmap="RdYlGn_r", axis=None), use_container_width=True)
                    st.dataframe(_bs["n_y"].fillna(0).astype(int), use_container_width=True)
                st.info("**怎么判断**：如果「高出30%以上」这组的「亏损超20%的比例」「浮亏曾超20%的比例」"
                        "明显更高，**而且**平均收益、同期超额并没有更高，逐年多数年份如此——"
                        "那乖离大就是真实的危险信号，值得写成规则再检验。\n\n"
                        "如果这组大亏多、但大赚也多（平均收益不低），那它只是波动更大，"
                        "避开它会同时避开大亏和大赚。")

        st.markdown("### 乖离规则检验：跳过高出30日线30%以上的票")
        st.caption("不用先跑乖离划分，可以直接运行。把「高出30%以上不买」写成规则，与基准同一配置比较："
                   "剔除后钱空着（不补位），或名额顺延给板块内后面的名次（顺延补位）。"
                   "按全部资金、真实复利和账户回撤比较。")
        if not ss.get("res"):
            st.info("先到「② 主回测」跑一次对照实验。")
        elif st.button("运行乖离规则检验", type="primary"):
            _df9, _keep9, _pks9, _sig9, _ = ss["res"]
            _b9 = list(_keep9)[0]
            _pool9 = next((v for k, v in _keep9.items() if str(k).startswith("对照C")), None)
            with st.spinner("三种做法各算一遍、模拟账户净值…"):
                ss["biasrule"] = skdj_filter_test(
                    panel, elig, sectors, SF[_sig9], dates, (BIAS >= 0.30).fillna(False),
                    _pks9[_b9], _keep9[_b9], _pool9, top_sec, top_n, cool, cap, min_mem,
                    hold, every, kdf=KDF, flag_label="高出30日线30%以上", **kw)
                ss["biasrule_lab"] = ss.get("res_lab", "")
                gc.collect()
        _br = ss.get("biasrule")
        if _br:
            st.caption(f"口径：{ss.get('biasrule_lab', '')}")
            st.markdown("**三种做法对比**")
            st.dataframe(_br["plans"].style.format(
                {"笔数": "{:.0f}", "相当于基准的仓位": "{:.0%}", "单利年化(按全部资金)": "{:+.1%}",
                 "样本内": "{:+.1%}", "样本外": "{:+.1%}", "中位收益": "{:+.2%}", "胜率": "{:.1%}",
                 "复利倍数(每天一份)": "×{:.2f}", "最大回撤(每天一份)": "{:.0%}",
                 "复利倍数(分4份,最差起点)": "×{:.2f}", "最大回撤(分4份,最差起点)": "{:.0%}"}),
                use_container_width=True)
            st.caption("表格较宽，手机上请向左滑动查看后面的列。")
            if len(_br["curves"]):
                st.line_chart(_br["curves"], height=240)
            with st.expander("逐年单利年化（按全部资金）"):
                st.dataframe(_br["yearly"].T.style.format("{:+.1%}")
                             .background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
            st.info("**要算有用**：最大回撤（尤其「分4份,最差起点」）明显更小，同时复利倍数、"
                    "样本内外、逐年都不明显差于基准。")
        st.divider()
        st.markdown("**逐只查看**（每行：日期 代码；默认填的是你截图里的9只）")
        _snap_default = ("2026-07-03 688359\n2026-07-03 002643\n2026-07-03 300489\n"
                         "2026-07-07 301045\n2026-07-08 300671\n"
                         "2026-08-25 603002\n2026-08-26 300909\n2026-08-27 688432\n2026-09-01 688209")
        _snap_txt = st.text_area("要查看的股票", value=_snap_default, height=190, key="snap_txt",
                                 label_visibility="collapsed")
        _mine9 = parse_my_trades(_snap_txt)
        if len(_mine9):
            _nm9 = basic.set_index("ts_code")["name"].to_dict() if "ts_code" in basic.columns else {}
            _snap = stock_snapshot(_mine9, panel, BIAS, KDF, DDF, MH, _nm9, ss.get("live"), hold, **kw)
            if len(_snap):
                st.dataframe(_snap.style.format(
                    {"高出30日线": "{:+.1%}", "20日涨幅": "{:+.1%}", "次日买持有到期收益": "{:+.1%}",
                     "持有中最大浮亏": "{:+.1%}"}, na_rep=""),
                    use_container_width=True, hide_index=True)
                st.caption("所有指标都是当天收盘时就能看到的；「次日买持有到期收益」按回测口径"
                           "（次日开盘买、第20个交易日开盘卖、含成本）。「当天在名单上」按侧边栏当前配置判断。"
                           "日期不是交易日时取之前最近的交易日。")
        st.divider()

        st.markdown("### 延迟入场：等日线转好再买")
        st.caption("同一份名单，只改「什么时候买」：① 次日直接买（基准）；② 等日线SKDJ上行再买"
                   "（K>D，而且K比前一天高——K还在75上方但已经拐头向下的，也要等），选股当天已经上行就照常次日买；③ 等MACD柱连续2天回升再买（绿柱缩短或红柱变长），"
                   "已经在回升就照常次日买。触发后次日开盘买、持有到期；**最多等20个交易日，等不到就放弃**。"
                   "同一只股票多次上名单、等到同一个触发日，只买一次。")
        if not ss.get("res"):
            st.info("先到「② 主回测」跑一次对照实验。")
        else:
            if st.button("运行延迟入场检验", type="primary"):
                _df7, _keep7, _pks7, _, _ = ss["res"]
                _b7 = list(_keep7)[0]
                with st.spinner("三种买法各算一遍、模拟账户净值…"):
                    ss["delay"] = entry_delay_test(_pks7[_b7], panel, dates, KDF, DDF, MH,
                                                   hold=hold, every=every, max_wait=20, **kw)
                    ss["delay_lab"] = ss.get("res_lab", "")
                    gc.collect()
            _dl = ss.get("delay")
            if _dl:
                st.caption(f"口径：{ss.get('delay_lab', '')}")
                st.markdown("**汇总表**（较宽，手机上请向左滑动查看后面的列）")
                st.dataframe(_dl["summary"].style.format(
                    {"实际买入笔数": "{:.0f}", "放弃比例": "{:.0%}", "平均等待天数": "{:.1f}",
                     "买入价较选股日(中位)": "{:+.1%}", "平均收益": "{:+.2%}", "中位收益": "{:+.2%}",
                     "胜率": "{:.1%}", "持有中最大浮亏(中位)": "{:+.1%}", "浮亏曾超20%的比例": "{:.0%}",
                     "单利年化(按全部资金)": "{:+.1%}", "样本内": "{:+.1%}", "样本外": "{:+.1%}",
                     "复利倍数(每天一份)": "×{:.2f}", "最大回撤(每天一份)": "{:.0%}",
                     "最大回撤(分4份,最差起点)": "{:.0%}"}),
                    use_container_width=True)
                if len(_dl["curves"]):
                    st.line_chart(_dl["curves"], height=240)
                with st.expander("逐年单利年化（按全部资金）"):
                    st.dataframe(_dl["yearly"].T.style.format("{:+.1%}")
                                 .background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
                st.info("**这张表同时看收益和「拿着的过程」**：「持有中最大浮亏」「浮亏曾超20%的比例」"
                        "是每笔交易在持有期间最深跌到过多少——这正是只看20日平均收益看不到的部分。\n\n"
                        "延迟入场要算有用，需要：浮亏和账户回撤明显更小，同时复利倍数、样本内外、"
                        "逐年都不明显差于基准。「买入价较选股日」为正，说明等待经常是在更高的价格买入。")
        st.divider()

        st.markdown("### 日线SKDJ下跌趋势：不买从75上方跌下来、一直在跌的票")
        st.caption("定义（只用买入决策当天及以前的数据）：① 这段死叉开始前10个交易日内K曾经≥75，"
                   "即从高位跌下来；② 死叉以来一直没有金叉；③ 当天K和D都已在75以下。"
                   "刚死叉、K还在75上方的头几天**不算**。先到第④页「候选股事后表现」看「日线SKDJ」一列，"
                   "确认标出来的票和你自己的判断一致，再看这里的结果。")
        if not ss.get("res"):
            st.info("先到「② 主回测」跑一次对照实验。")
        else:
            if st.button("运行SKDJ下跌趋势检验", type="primary"):
                _df6, _keep6, _pks6, _sig6, _ = ss["res"]
                _b6 = list(_keep6)[0]
                _pool6 = next((v for k, v in _keep6.items() if str(k).startswith("对照C")), None)
                with st.spinner("划分、三种做法各算一遍、模拟账户净值…"):
                    ss["skdj"] = skdj_filter_test(
                        panel, elig, sectors, SF[_sig6], dates, DTM, _pks6[_b6], _keep6[_b6], _pool6,
                        top_sec, top_n, cool, cap, min_mem, hold, every, kdf=KDF, **kw)
                    ss["skdj_lab"] = ss.get("res_lab", "")
                    gc.collect()
            _sk = ss.get("skdj")
            if _sk:
                st.caption(f"口径：{ss.get('skdj_lab', '')}　·　基准成交里处在下跌趋势的占 {_sk['share']:.0%}")
                st.markdown("**① 干净划分：同一批成交，下跌趋势的票 vs 其他**")
                _sp = _sk["split"]
                st.dataframe(_sp.style.format(
                    {"笔数": "{:.0f}", "占比": "{:.1%}", "平均收益": "{:+.2%}", "中位收益": "{:+.2%}",
                     "胜率": "{:.1%}", "20日内最惨10%": "{:+.1%}", "同期超额": "{:+.2%}", "收益率": "{:+.2%}",
                     **{c: "{:.2f}" for c in _sp.columns if c.startswith("t(")}}),
                    use_container_width=True)
                with st.expander("逐年（同期超额）"):
                    st.dataframe(_sk["yearly_split"].style.format("{:+.2%}")
                                 .background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
                st.markdown("**② 三种做法对比**")
                st.dataframe(_sk["plans"].style.format(
                    {"笔数": "{:.0f}", "相当于基准的仓位": "{:.0%}", "单利年化(按全部资金)": "{:+.1%}",
                     "样本内": "{:+.1%}", "样本外": "{:+.1%}", "中位收益": "{:+.2%}", "胜率": "{:.1%}",
                     "复利倍数(每天一份)": "×{:.2f}", "最大回撤(每天一份)": "{:.0%}",
                     "复利倍数(分4份,最差起点)": "×{:.2f}", "最大回撤(分4份,最差起点)": "{:.0%}"}),
                    use_container_width=True)
                if len(_sk["curves"]):
                    st.line_chart(_sk["curves"], height=240)
                with st.expander("逐年单利年化（按全部资金）"):
                    st.dataframe(_sk["yearly"].T.style.format("{:+.1%}")
                                 .background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
                st.info("**怎么判断**：你的实盘做法是「剔除，不补位」。它要算有用，需要"
                        "① 复利倍数不低于基准、最大回撤明显更小（这是你最关心的）；"
                        "② 样本内、样本外两段都不差于基准；③ 逐年多数年份胜过基准。\n\n"
                        "**只看7月那几天不算数**：挑出最差的一段回头看，任何过滤都显得有用。"
                        "要看它在全部8年里，是不是也在上涨行情中把后来大涨的票过滤掉了。")
        st.divider()

        st.markdown("### 执行时机：龙头板块已经领跑几天")
        st.caption("第②页「两层」方案的同一批成交，按买入当天排第一的板块已经连续领跑几天分组，不做替补。"
                   "回答的是：龙头刚换就买，和它已经领跑一段时间才买，差别有多大。")
        if not ss.get("res"):
            st.info("先到「② 主回测」跑一次对照实验。")
        else:
            df, keep, pks, sig_, sr_ = ss["res"]
            base = list(keep)[0]
            pk0, tr0 = pks[base], keep[base]
            if st.button("运行执行时机划分", type="primary"):
                with st.spinner("计算中…"):
                    ss["ksplit"] = run_age_diagnosis(pk0, tr0, SF[sig_], panel["cal"])
            if ss.get("ksplit") is not None and len(ss["ksplit"]):
                st.dataframe(ss["ksplit"].style.format(
                    {"笔数": "{:.0f}", "占比": "{:.1%}", "平均收益": "{:+.2%}",
                     "中位收益": "{:+.2%}", "胜率": "{:.1%}", "聚类t": "{:.2f}"})
                    .background_gradient(subset=["中位收益"], cmap="RdYlGn"),
                    use_container_width=True)
                n_all = len(tr0.dropna(subset=["收益率"]))
                st.caption(f"各组笔数合计 {int(ss['ksplit']['笔数'].sum())}，总成交 {n_all} —— "
                           "相等说明是真划分，不是替补。「聚类t」没有做重叠修正，大约要除以3再看。")
                st.info("各档如果差不多，说明**错过前几天不用懊恼**，什么时候有钱什么时候买都行；"
                        "如果「第1天」明显更好，才值得盯紧龙头切换。占比小的档标准误大，别被单个数字带走。")
            elif ss.get("ksplit") is not None:
                st.warning("样本不足，没有结果。")
        st.divider()

        st.markdown("### 已经验证过、结论是不用的想法")
        st.caption("留作记录，免得以后重复验证。每一项都是用全部回测交易、和同配置基准比较得出的。")
        st.markdown(
            "- **止损 8% / 10%**：复利结果和持有到期几乎一样（6年都约 ×2.0），胜率从 43% 降到 33%。选了持有到期。\n"
            "- **利润保护**（涨过10%回吐一半、涨过20%回落到+10%）：资金年化从 24% 降到 5%-21%，砍掉的正是后面继续大涨的票。\n"
            "- **跳过日线K低的票**（K<30/40/50，带补位）：三个阈值不平滑，2023年以后全部不如不跳过。\n"
            "- **避开高位死叉**：死叉后1-5天买入的一组反而更好（平均 +3.0% 对 +1.4%）。\n"
            "- **只买成分股多的板块**：方向相反——5-7只的小板块同期超额 +3.5%，16只以上 -0.7%，但逐年不稳，也不改规则。\n"
            "- **板块内上涨比例（集群效应）、合格数变化**：扣掉大盘后各组差别很小，t 都不到 2。\n"
            "- **绝对动量**（只买60日动量>0的板块）：年化 26.3% → 22.9%，样本外更差，2022年要空仓9周。\n"
            "- **热点未过气**（只买20日动量前8的板块）：年化 26.3% → 22.9%，9年里只有4年好过基准。\n"
            "- **每天5只代替3只**：结果基本一样（25.4% 对 26.3%），可以当候选池用，但不是改进。")

    # ---------------- ④ 今日候选 ----------------
    with t4:
        st.success("**日常只需要这一页。** 左侧点「下载数据」更新到最新，"
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
                       "稍后点左侧「下载数据」可以试着补上。" if _lag == 1 else ""))
        if _nel < _nel_ref * 0.5:
            st.error(f"**今日合格股票只有 {_nel} 只，而前几日平均 {_nel_ref:.0f} 只。** "
                     "名单可能不可靠，建议稍后重新下载数据再看。")
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

        # 旧版这里只传 [今天] 一天，没有冷却历史 —— 回测里有轮换，今日候选页却没有，
        # 于是板块和排名不变时天天给出同一份名单。现在按回测**完全相同**的日程
        # （同一个起点、每 every 日一次、同样的冷却）从头推到今天，名单 = 回测在这一天会选的票。
        if not dates:
            st.warning("数据太短，没有选股日。"); st.stop()
        lkey = (dkey, sig2, sr2, top_sec, top_n, cap, every, cool)
        if ss.get("live_key") != lkey:
            with st.spinner("按回测日程重放选股（含冷却轮换）…"):
                ss["live"] = sector_then_stock(panel, elig, sectors, SF[sig2], dates,
                                               top_sec, top_n, sr2, "最强", cooldown=cool,
                                               kdf=KDF, per_sec_cap=cap,
                                               min_members=min_mem)
                ss["live_key"] = lkey
        pk_all = ss["live"]
        ld = dates[-1]
        cal_list = list(panel["cal"])
        gap = cal_list.index(d) - cal_list.index(ld)
        if gap > 0:
            st.warning(f"**{d:%Y-%m-%d} 不是选股日。** 按「每 {every} 个交易日选一次」，"
                       f"最近的选股日是 **{ld:%Y-%m-%d}**，下一个选股日在 "
                       f"**{every - gap} 个交易日**之后。下面显示 {ld:%Y-%m-%d} 的名单。\n\n"
                       "想每天都拿到新名单：侧边栏设「每几个交易日选一次 = 1」「冷却 = 2」，"
                       "**并先在第②页跑对照和滚动前推确认这个口径。**")
        pk = pk_all[pk_all["date"] == ld] if len(pk_all) else pd.DataFrame()
        if cool <= every:
            st.warning(f"冷却 {cool} ≤ 选股间隔 {every}，**冷却不起作用**：板块和排名不变时名单不会变。")

        if len(pk) < top_n:
            info = [f"{s3}: {int(elig.loc[ld, sectors[s3]].sum())} 只合格"
                    for s3 in list(SF[sig2].loc[ld].dropna()
                                   .sort_values(ascending=False).index[:top_sec])]
            st.warning(f"**只选出 {len(pk)} 只，少于设定的 {top_n} 只。** "
                       f"各板块合格数：{'；'.join(info)}。\n\n"
                       "排名靠前的股票还在冷却期、或受「每个板块最多取几只」限制时，"
                       "前几个板块的剩余股票可能不够填满名额。")
        nm = basic.set_index("ts_code")["name"].to_dict()

        if not len(pk):
            st.warning("今日无候选。")
        else:
            rows_ = []
            for _, r in pk.iterrows():
                c = r["code"]
                rc = panel["raw_close"].loc[d, c]
                mv = panel["circ_mv"].loc[:d, c].dropna()
                kk = KDF.loc[d, c] if c in KDF.columns else np.nan
                _dt = bool(DTM.at[d, c]) if (c in DTM.columns and d in DTM.index) else False
                rows_.append({
                    "序": int(r["rank"]), "代码": c, "名称": nm.get(c, ""), "板块": r["板块"],
                    "板块内名次": int(r["板块内名次"]) if pd.notna(r.get("板块内名次")) else None,
                    "20日涨幅(选股日)": f"{r['score']:.1%}",
                    f"收盘价({d:%m-%d})": round(float(rc), 2) if pd.notna(rc) else None,
                    "流通市值(亿)": round(float(mv.iloc[-1]) / 1e4) if len(mv) else None,
                    "日线K": round(float(kk), 1) if pd.notna(kk) else None,
                    "日线SKDJ": "下跌趋势" if _dt else "",
                    "高出30日线": (f"{float(BIAS.at[d, c]):+.0%}" + (" ⚠️" if float(BIAS.at[d, c]) >= 0.30 else ""))
                    if (c in BIAS.columns and pd.notna(BIAS.at[d, c])) else ""})
            out = pd.DataFrame(rows_)
            st.subheader(f"{ld:%Y-%m-%d}　候选名单")
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("下载 CSV", out.to_csv(index=False).encode("utf-8-sig"),
                               f"picks_{ld:%Y%m%d}.csv", "text/csv")
            st.info(f"**执行规则（回测口径）**：{ld:%Y-%m-%d} 之后的下一个交易日开盘买入，"
                    f"**持有 {hold} 个交易日后开盘卖出**，不设止盈止损。\n\n"
                    "「板块内名次」是该股在本板块里按20日涨幅的名次；名次靠后说明前面的票在冷却中，"
                    "名额顺延到了它（实测板块内前10名收益看不出差别）。\n\n"
                    "「高出30日线」标 ⚠️ 的是高出30%以上：8年里这类票亏损超20%的概率约为其他票的三倍，"
                    "但平均收益并不更低——要不要跳过，看第③页「乖离规则检验」。\n\n"
                    "「日线K」仅供参考：实测跳过K低的票、避开高位死叉的票，都不能提高收益。"
                    "**看到日线在跌就不买，等于偏离回测口径。**")

        # 最近几个选股日的名单 —— 用来确认轮换确实在发生
        if len(pk_all):
            recent = [x for x in dates if x in set(pk_all["date"])][-10:]
            hist = pk_all[pk_all["date"].isin(recent)].copy()
            hist["标签"] = [f"{nm.get(c, c)}（{sec}#{int(rk)}）"
                          for c, sec, rk in zip(hist["code"], hist["板块"], hist["板块内名次"])]
            tab = (hist.sort_values(["date", "rank"])
                   .groupby("date")["标签"].apply(lambda x: "　".join(x))
                   .sort_index(ascending=False).rename("名单（板块#板块内名次）"))
            with st.expander(f"最近 {len(tab)} 个选股日的名单（看轮换是否在发生）", expanded=gap > 0):
                st.dataframe(tab.reset_index().rename(columns={"date": "选股日"}).assign(
                    选股日=lambda x: pd.to_datetime(x["选股日"]).dt.strftime("%Y-%m-%d")),
                    use_container_width=True, hide_index=True)
                st.caption("和第②页回测是同一套日程、同一个冷却规则算出来的，"
                           "所以这里每一行都是回测里真实发生过的选股。")

        # ---------------- 候选股事后表现 ----------------
        st.divider()
        st.markdown("### 候选股事后表现：对照你的实盘选择")
        st.caption(f"过去每个选股日的全部候选，按回测口径算收益：选股日次日开盘买、第 {hold} 个交易日开盘卖，"
                   f"含手续费和滑点。还没满 {hold} 天的按最新收盘价算浮动收益。")
        if len(pk_all):
            n_back = st.slider("回看多少个选股日", 20, 120, 60, 10, key="hist_n")
            _days = [x for x in dates if x in set(pk_all["date"])][-n_back:]
            hkey = (lkey, n_back, hold)
            if ss.get("hist_key") != hkey:
                ss["hist"] = candidate_outcomes(pk_all[pk_all["date"].isin(_days)], panel, hold, **kw)
                ss["hist_key"] = hkey
            H = ss["hist"].copy()
        else:
            H = pd.DataFrame()
        if not len(H):
            st.info("还没有候选记录。")
        else:
            H["名称"] = H["code"].map(nm).fillna("")
            H["乖离"] = [float(BIAS.at[dd_, cc]) if (cc in BIAS.columns and dd_ in BIAS.index) else np.nan
                        for dd_, cc in zip(pd.to_datetime(H["选股日"]), H["code"])]
            H["日线SKDJ"] = ["下跌趋势" if (cc in DTM.columns and dd_ in DTM.index and bool(DTM.at[dd_, cc]))
                           else "" for dd_, cc in zip(pd.to_datetime(H["选股日"]), H["code"])]
            my_txt = st.text_area(
                "你实际买入的记录（每行一条：日期 代码。日期写选股日或买入日都可以）",
                placeholder="2026-09-15 300413\n2026-09-17 603533",
                key="my_trades", height=110)
            mine = parse_my_trades(my_txt)
            H["你买了"] = match_my_trades(H, mine)
            if len(mine):
                st.caption(f"识别到 {len(mine)} 条记录，匹配上 {int(H['你买了'].sum())} 只候选。"
                           "关掉网页后需要重新粘贴，建议把记录保存在手机备忘录里。")
            done = H[H["已满期"]]
            live_ = H[~H["已满期"] & H["收益率"].notna()]
            if len(done) and (done["乖离"] >= 0.30).any():
                _hb = done[done["乖离"] >= 0.30]; _lb = done[~(done["乖离"] >= 0.30)]
                st.caption(f"回看期内已满期的候选里，高出30日线30%以上的 {len(_hb)} 只：平均 {_hb['收益率'].mean():+.2%}，"
                           f"亏损超20%的占 {(_hb['收益率'] <= -0.2).mean():.0%}；其他 {len(_lb)} 只：平均 "
                           f"{_lb['收益率'].mean():+.2%}，亏损超20%的占 {(_lb['收益率'] <= -0.2).mean():.0%}。"
                           "只是最近几十天，结论以第③页8年数据为准。")
            if len(done) and (done["日线SKDJ"] == "下跌趋势").any():
                _dn = done[done["日线SKDJ"] == "下跌趋势"]; _ot = done[done["日线SKDJ"] != "下跌趋势"]
                st.caption(f"回看期内已满期的候选里，标为「下跌趋势」的 {len(_dn)} 只平均 {_dn['收益率'].mean():+.2%}"
                           f"（胜率 {(_dn['收益率'] > 0).mean():.0%}），其他 {len(_ot)} 只平均 "
                           f"{_ot['收益率'].mean():+.2%}（胜率 {(_ot['收益率'] > 0).mean():.0%}）。"
                           "这只是最近几十天，结论要看第③页用8年数据跑的检验。")
            c1, c2, c3 = st.columns(3)
            if len(done):
                c1.metric(f"已满{hold}日的候选", f"{len(done)} 只", f"平均 {done['收益率'].mean():+.2%}")
                c2.metric("中位收益", f"{done['收益率'].median():+.2%}",
                          f"胜率 {(done['收益率'] > 0).mean():.0%}")
            if len(live_):
                c3.metric("持有中的候选", f"{len(live_)} 只", f"浮动平均 {live_['收益率'].mean():+.2%}")

            # 你的选择 vs 同一天没选的
            if H["你买了"].any():
                st.markdown("**你选中的 vs 同一天没选中的**")
                cmp_rows = []
                for dday, g in H[H["收益率"].notna()].groupby("选股日"):
                    if not g["你买了"].any() or g["你买了"].all():
                        continue
                    a_ = g.loc[g["你买了"], "收益率"].mean()
                    b_ = g.loc[~g["你买了"], "收益率"].mean()
                    cmp_rows.append({"选股日": dday, "已满期": bool(g["已满期"].all()),
                                     "你选中的": a_, "同日没选中的": b_, "差": a_ - b_})
                if cmp_rows:
                    C = pd.DataFrame(cmp_rows).sort_values("选股日", ascending=False)
                    Cd = C[C["已满期"]]
                    if len(Cd):
                        st.metric(f"已满{hold}日的 {len(Cd)} 次选择：平均每次比没选中的",
                                  f"{Cd['差'].mean():+.2%}",
                                  f"{int((Cd['差'] > 0).sum())}/{len(Cd)} 次选得更好")
                    st.dataframe(C.assign(选股日=lambda x: pd.to_datetime(x["选股日"]).dt.strftime("%Y-%m-%d"))
                                 .style.format({"你选中的": "{:+.2%}", "同日没选中的": "{:+.2%}",
                                                "差": "{:+.2%}"})
                                 .background_gradient(subset=["差"], cmap="RdYlGn", vmin=-0.1, vmax=0.1),
                                 use_container_width=True, hide_index=True)
                    st.info("**怎么判断你的挑选有没有用**：只看「已满期」的行。攒够 **30 次以上**再下结论——"
                            "单只股票20天的涨跌幅度常常在 ±15% 以上，十几次的平均差几个百分点完全可能是运气。\n\n"
                            "如果30次以后「选得更好」的次数明显超过一半、平均差持续为正，说明你的判断有价值；"
                            "如果接近一半或更差，就改回机械执行（按序号买），省心也不吃亏。")
                else:
                    st.caption("匹配上的选股日里，没有「既有你买的、也有你没买的」可以比较。")

            # 按序号汇总：确认候选之间事前没有差别
            if len(done) >= 20:
                by_rank = done.groupby("序")["收益率"].agg(["size", "mean", "median",
                                                         lambda x: (x > 0).mean()])
                by_rank.columns = ["只数", "平均收益", "中位收益", "胜率"]
                with st.expander("按名单序号汇总（已满期的候选）"):
                    st.dataframe(by_rank.style.format({"平均收益": "{:+.2%}", "中位收益": "{:+.2%}",
                                                       "胜率": "{:.0%}"}),
                                 use_container_width=True)
                    st.caption("回看期很短，各序号之间的差别基本是噪音；长期回测里板块内前10名看不出差别。")

            show = H.sort_values(["选股日", "序"], ascending=[False, True]).copy()
            show["选股日"] = pd.to_datetime(show["选股日"]).dt.strftime("%Y-%m-%d")
            show["买入日"] = pd.to_datetime(show["买入日"]).dt.strftime("%m-%d").fillna("")
            show["你买了"] = show["你买了"].map({True: "✓", False: ""})
            show["高出30日线"] = [(f"{v:+.0%}" + (" ⚠️" if v >= 0.30 else "")) if pd.notna(v) else ""
                               for v in show["乖离"]]
            show = show[["选股日", "序", "名称", "code", "板块", "板块内名次", "高出30日线", "日线SKDJ", "买入日",
                         "买入价(实际)", "状态", "收益率", "期间最高", "期间最低", "你买了"]].rename(columns={"code": "代码"})
            st.dataframe(show.style.format({"收益率": "{:+.2%}", "期间最高": "{:+.1%}",
                                            "期间最低": "{:+.1%}", "买入价(实际)": "{:.2f}",
                                            "板块内名次": "{:.0f}"}, na_rep="")
                         .background_gradient(subset=["收益率"], cmap="RdYlGn", vmin=-0.2, vmax=0.2),
                         use_container_width=True, hide_index=True, height=460)
            st.caption("「期间最高/最低」是持有期内相对买入价的最大涨幅和最大跌幅——"
                       "能看出多少票是先跌后涨、多少是涨了又回落。")
            st.download_button("下载候选股事后表现 CSV",
                               show.to_csv(index=False).encode("utf-8-sig"),
                               f"candidates_history_{d:%Y%m%d}.csv", "text/csv")

    if API_ERRORS:
        with st.expander(f"接口异常 {len(API_ERRORS)} 条"):
            st.write(API_ERRORS[-30:])


if __name__ == "__main__" and st is not None:
    main()
