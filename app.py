# -*- coding: utf-8 -*-
"""
智选股 V2.0 - 鱼身策略
收盘后运行，次日高开+冲高1.5%触发买入，止损5%
新增：资金流向+筹码分析+换手率+板块热度+评分优化
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import os
import pickle
import concurrent.futures

warnings.filterwarnings("ignore")

pro = None
CHECKPOINT_FILE = "bt_checkpoint.csv"
CACHE_FILE = "market_cache_v2.pkl"

CACHE_DAILY = pd.DataFrame()
CACHE_ADJ = pd.DataFrame()
CACHE_BASIC = pd.DataFrame()
CACHE_MONEYFLOW = pd.DataFrame()
CACHE_CHIP = pd.DataFrame()

st.set_page_config(page_title="智选股 V2.0", layout="wide")
st.title("智选股 V2.0 - 鱼身策略")
st.caption("收盘后运行，次日高开+冲高1.5%触发买入，止损5%")

@st.cache_data(ttl=3600*12)
def safe_api(func_name, **kwargs):
    global pro
    if pro is None:
        return pd.DataFrame()
    func = getattr(pro, func_name)
    for _ in range(3):
        try:
            df = func(**kwargs)
            if df is not None and not df.empty:
                return df
            time.sleep(0.5)
        except:
            time.sleep(1)
    return pd.DataFrame()

def safe_api_nocache(func_name, **kwargs):
    global pro
    if pro is None:
        return pd.DataFrame()
    func = getattr(pro, func_name)
    for _ in range(3):
        try:
            df = func(**kwargs)
            if df is not None and not df.empty:
                return df
            time.sleep(0.5)
        except:
            time.sleep(1)
    return pd.DataFrame()

@st.cache_data(ttl=3600*24*7)
def load_stock_basic():
    df = safe_api("stock_basic", list_status="L", fields="ts_code,name,list_date")
    if df.empty:
        return pd.DataFrame()
    df = df[~df["name"].str.contains("ST", na=False)]
    df = df[~df["ts_code"].str.startswith("43")]
    df = df[~df["ts_code"].str.startswith("83")]
    df = df[~df["ts_code"].str.startswith("87")]
    df = df[~df["ts_code"].str.startswith("92")]
    return df

@st.cache_data(ttl=3600*24*7)
def load_industry_map():
    global pro
    if pro is None:
        return {}
    try:
        sw = pro.index_classify(level="L1", src="SW2021")
        if sw.empty:
            return {}
        result = {}
        for code in sw["index_code"].tolist():
            members = pro.index_member(index_code=code, is_new="Y")
            if not members.empty:
                for c in members["con_code"]:
                    result[c] = code
            time.sleep(0.05)
        return result
    except:
        return {}

def load_market_cache(trade_days_list):
    global CACHE_DAILY, CACHE_ADJ, CACHE_BASIC, CACHE_MONEYFLOW, CACHE_CHIP

    if os.path.exists(CACHE_FILE):
        st.success("发现本地缓存，极速加载中...")
        try:
            with open(CACHE_FILE, "rb") as f:
                cached = pickle.load(f)
            CACHE_DAILY = cached["daily"]
            CACHE_ADJ = cached["adj"]
            CACHE_BASIC = cached.get("basic", pd.DataFrame())
            CACHE_MONEYFLOW = cached.get("moneyflow", pd.DataFrame())
            CACHE_CHIP = cached.get("chip", pd.DataFrame())
            st.info("缓存加载成功！包含行情+资金流+筹码数据")
            return True
        except:
            os.remove(CACHE_FILE)

    earliest = min(trade_days_list)
    latest = max(trade_days_list)
    start = (datetime.strptime(earliest, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
    end = (datetime.strptime(latest, "%Y%m%d") + timedelta(days=30)).strftime("%Y%m%d")

    cal = safe_api_nocache("trade_cal", start_date=start, end_date=end, is_open="1")
    if cal.empty:
        return False
    all_dates = cal["cal_date"].tolist()

    st.info(f"首次运行，下载 {start} 至 {end} 全量数据（含资金流+筹码），请耐心等待...")

    daily_list = []
    adj_list = []
    basic_list = []
    mf_list = []
    chip_list = []

    bar = st.progress(0, text="下载行情数据...")
    total = len(all_dates)

    def fetch_one(date):
        d = safe_api_nocache("daily", trade_date=date)
        a = safe_api_nocache("adj_factor", trade_date=date)
        b = safe_api_nocache("daily_basic", trade_date=date,
                             fields="ts_code,turnover_rate,circ_mv,pe,pb,volume_ratio")
        mf = safe_api_nocache("moneyflow", trade_date=date)
        chip = safe_api_nocache("cyq_perf", trade_date=date)
        return d, a, b, mf, chip

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_one, date): date for date in all_dates}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                d, a, b, mf, chip = future.result()
                if not d.empty:
                    daily_list.append(d)
                if not a.empty:
                    adj_list.append(a)
                if not b.empty:
                    basic_list.append(b)
                if not mf.empty:
                    mf_list.append(mf)
                if not chip.empty:
                    chip_list.append(chip)
            except:
                pass
            if i % 5 == 0 or i == total - 1:
                bar.progress((i+1)/total, text=f"下载中: {i+1}/{total} 天")
    bar.empty()

    if not daily_list:
        st.error("下载失败，请检查网络或Token")
        return False

    with st.spinner("整理数据并保存缓存..."):
        CACHE_DAILY = pd.concat(daily_list).drop_duplicates(subset=["ts_code","trade_date"])
        CACHE_DAILY = CACHE_DAILY.set_index(["ts_code","trade_date"]).sort_index()

        CACHE_ADJ = pd.concat(adj_list).drop_duplicates(subset=["ts_code","trade_date"])
        CACHE_ADJ = CACHE_ADJ.set_index(["ts_code","trade_date"]).sort_index()

        if basic_list:
            CACHE_BASIC = pd.concat(basic_list).drop_duplicates(subset=["ts_code","trade_date"])
            CACHE_BASIC = CACHE_BASIC.set_index(["ts_code","trade_date"]).sort_index()

        if mf_list:
            CACHE_MONEYFLOW = pd.concat(mf_list).drop_duplicates(subset=["ts_code","trade_date"])
            CACHE_MONEYFLOW = CACHE_MONEYFLOW.set_index(["ts_code","trade_date"]).sort_index()

        if chip_list:
            CACHE_CHIP = pd.concat(chip_list).drop_duplicates(subset=["ts_code","trade_date"])
            CACHE_CHIP = CACHE_CHIP.set_index(["ts_code","trade_date"]).sort_index()

        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump({
                    "daily": CACHE_DAILY,
                    "adj": CACHE_ADJ,
                    "basic": CACHE_BASIC,
                    "moneyflow": CACHE_MONEYFLOW,
                    "chip": CACHE_CHIP,
                }, f)
            st.success("全量数据已缓存，下次启动将秒开！")
        except Exception as e:
            st.warning(f"缓存写入失败: {e}")
    return True

def get_day_data(trade_date):
    global CACHE_DAILY, CACHE_ADJ, CACHE_BASIC, CACHE_MONEYFLOW, CACHE_CHIP
    try:
        daily = CACHE_DAILY.xs(trade_date, level="trade_date").reset_index()
        daily.rename(columns={"index": "ts_code"}, inplace=True)
        if "ts_code" not in daily.columns:
            daily = daily.reset_index()
    except:
        daily = pd.DataFrame()

    try:
        basic = CACHE_BASIC.xs(trade_date, level="trade_date").reset_index()
    except:
        basic = pd.DataFrame()

    try:
        mf = CACHE_MONEYFLOW.xs(trade_date, level="trade_date").reset_index()
    except:
        mf = pd.DataFrame()

    try:
        chip = CACHE_CHIP.xs(trade_date, level="trade_date").reset_index()
    except:
        chip = pd.DataFrame()

    return daily, basic, mf, chip

def get_stock_history_fast(ts_code, end_date, lookback=90):
    global CACHE_DAILY, CACHE_ADJ
    start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=lookback*2)).strftime("%Y%m%d")
    try:
        daily = CACHE_DAILY.loc[ts_code].copy()
        daily = daily[(daily.index >= start) & (daily.index <= end_date)]
        daily = daily.reset_index()
        daily.columns = ["trade_date"] + list(daily.columns[1:])
        daily = daily.sort_values("trade_date").reset_index(drop=True)
    except:
        return pd.DataFrame()
    if daily.empty or len(daily) < 30:
        return pd.DataFrame()
    try:
        adj = CACHE_ADJ.loc[ts_code]["adj_factor"].copy()
        adj = adj[(adj.index >= start) & (adj.index <= end_date)]
        adj_df = adj.reset_index()
        adj_df.columns = ["trade_date", "adj_factor"]
        daily = daily.merge(adj_df, on="trade_date", how="left")
        daily["adj_factor"] = daily["adj_factor"].ffill().fillna(1.0)
        latest_adj = daily["adj_factor"].iloc[-1]
        for col in ["open", "high", "low", "close"]:
            if col in daily.columns:
                daily[col] = daily[col] * daily["adj_factor"] / latest_adj
    except:
        pass
    daily["ts_code"] = ts_code
    return daily

@st.cache_data(ttl=3600*12)
def get_stock_history_live(ts_code, end_date, lookback=90):
    start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=lookback*2)).strftime("%Y%m%d")
    daily = safe_api("daily", ts_code=ts_code, start_date=start, end_date=end_date)
    if daily is None or daily.empty or len(daily) < 30:
        return pd.DataFrame()
    adj = safe_api("adj_factor", ts_code=ts_code, start_date=start, end_date=end_date)
    daily = daily.sort_values("trade_date").reset_index(drop=True)
    if not adj.empty:
        daily = daily.merge(adj[["trade_date", "adj_factor"]], on="trade_date", how="left")
        daily["adj_factor"] = daily["adj_factor"].ffill().fillna(1.0)
        latest_adj = daily["adj_factor"].iloc[-1]
        for col in ["open", "high", "low", "close"]:
            daily[col] = daily[col] * daily["adj_factor"] / latest_adj
    daily["ts_code"] = ts_code
    return daily

def calc_indicators(hist, trade_date, extra=None):
    if hist.empty or len(hist) < 30:
        return None
    close = hist["close"]
    today = hist[hist["trade_date"] == trade_date]
    if today.empty:
        return None
    idx = today.index[0]
    if idx < 19:
        return None

    ma20 = close.iloc[idx-19:idx+1].mean()
    ma20_prev = close.iloc[idx-20:idx].mean() if idx >= 20 else ma20
    ma60 = close.iloc[max(0,idx-59):idx+1].mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_bar = ((dif - dea) * 2).iloc[idx]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/12, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/12, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / (loss + 1e-9))).iloc[idx]

    row = hist.iloc[idx]
    c = float(row["close"])
    h = float(row["high"])
    l = float(row["low"])
    upper_shadow = (h - c) / c * 100
    body_range = h - l
    body_pos = (c - l) / body_range if body_range > 0 else 0

    pct_1d = float(row["pct_chg"]) if "pct_chg" in row else 0
    pct_5d = (c / float(hist.iloc[idx-5]["close"]) - 1) * 100 if idx >= 5 else 0
    pct_20d = (c / float(hist.iloc[idx-20]["close"]) - 1) * 100 if idx >= 20 else 0

    low_60 = hist["low"].iloc[max(0,idx-59):idx+1].min()
    from_bottom = (c - low_60) / low_60 * 100 if low_60 > 0 else 0

    vol_5 = hist["vol"].iloc[max(0,idx-4):idx+1].mean()
    vol_20 = hist["vol"].iloc[max(0,idx-19):idx+1].mean()
    vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1

    consec_limit = False
    if idx >= 1 and "pct_chg" in hist.columns:
        prev_pct = float(hist.iloc[idx-1]["pct_chg"])
        if pct_1d >= 9.5 and prev_pct >= 9.5:
            consec_limit = True

    winner_rate = 60
    net_mf = 0
    turnover_rate = 5
    volume_ratio = 1

    if extra is not None:
        winner_rate = float(extra.get("winner_rate", 60))
        net_mf_amount = extra.get("net_mf_amount", 0)
        net_mf = float(net_mf_amount) if pd.notna(net_mf_amount) else 0
        tr = extra.get("turnover_rate", 5)
        turnover_rate = float(tr) if pd.notna(tr) else 5
        vr = extra.get("volume_ratio", 1)
        volume_ratio = float(vr) if pd.notna(vr) else 1

    return {
        "close": c, "high": h, "low": l,
        "ma20": ma20, "ma60": ma60, "ma20_prev": ma20_prev,
        "macd_bar": macd_bar, "rsi": rsi,
        "upper_shadow": upper_shadow, "body_pos": body_pos,
        "pct_1d": pct_1d, "pct_5d": pct_5d, "pct_20d": pct_20d,
        "from_bottom": from_bottom,
        "vol_ratio": vol_ratio,
        "consec_limit": consec_limit,
        "winner_rate": winner_rate,
        "net_mf": net_mf,
        "turnover_rate": turnover_rate,
        "volume_ratio": volume_ratio,
    }

def calc_score(ind, sector_score, market_strong):
    detail = {}

    # 1. 技术面 0-25分
    tech = 0
    if ind["close"] > ind["ma20"] and ind["ma20"] > ind["ma20_prev"]:
        tech += 8
    if ind["close"] > ind["ma60"]:
        tech += 5
    if ind["macd_bar"] > 0:
        tech += 7
    rsi = ind["rsi"]
    if 50 <= rsi <= 65:
        tech += 5
    elif 65 < rsi <= 75:
        tech += 3
    elif rsi > 75:
        tech += 1
    detail["tech"] = min(tech, 25)

    # 2. 买入时机 0-20分（偏离MA20越小越好）
    timing = 0
    dev = (ind["close"] - ind["ma20"]) / ind["ma20"] * 100
    if dev <= 2:
        timing += 12
    elif dev <= 5:
        timing += 8
    elif dev <= 10:
        timing += 4
    else:
        timing += 0
    p5 = ind["pct_5d"]
    if 2 <= p5 <= 8:
        timing += 8
    elif p5 < 2:
        timing += 5
    elif p5 <= 15:
        timing += 2
    else:
        timing += 0
    detail["timing"] = min(timing, 20)

    # 3. 量能健康 0-15分
    vr = ind["vol_ratio"]
    tr = ind["turnover_rate"]
    vol = 0
    if 1.2 <= vr <= 2.5:
        vol += 10
    elif 1.0 <= vr < 1.2:
        vol += 6
    elif 2.5 < vr <= 3.5:
        vol += 5
    elif vr > 3.5:
        vol += 0
    else:
        vol += 2
    if 3 <= tr <= 10:
        vol += 5
    elif tr < 3:
        vol += 2
    elif tr <= 15:
        vol += 2
    else:
        vol += 0
    detail["vol"] = min(vol, 15)

    # 4. 鱼身判断 0-15分（从底部涨幅越小越好）
    fb = ind["from_bottom"]
    if fb <= 20:
        fish = 15
    elif fb <= 40:
        fish = 10
    elif fb <= 70:
        fish = 5
    else:
        fish = 0
    detail["fish"] = fish

    # 5. 资金+筹码 0-15分（新增）
    chip_mf = 0
    wr = ind["winner_rate"]
    if 55 <= wr <= 75:
        chip_mf += 8
    elif 75 < wr <= 85:
        chip_mf += 5
    elif wr < 55:
        chip_mf += 2
    else:
        chip_mf += 0
    net_mf = ind["net_mf"]
    if net_mf > 10000:
        chip_mf += 7
    elif net_mf > 0:
        chip_mf += 4
    elif net_mf > -5000:
        chip_mf += 1
    else:
        chip_mf += 0
    detail["chip_mf"] = min(chip_mf, 15)

    # 6. 板块热度 0-10分
    detail["sector"] = min(sector_score, 10)

    # 7. 大盘环境 0-10分
    detail["market"] = 10 if market_strong else 3

    total = sum(detail.values())
    return total, detail

def risk_tag(ind):
    score = 0
    if ind["from_bottom"] > 80:
        score += 2
    if ind["from_bottom"] > 120:
        score += 2
    if ind["consec_limit"]:
        score += 3
    if ind["pct_5d"] > 15:
        score += 1
    if ind["rsi"] > 78:
        score += 1
    if ind["winner_rate"] > 85:
        score += 1
    if ind["turnover_rate"] > 15:
        score += 1
    if score >= 4:
        return "高风险"
    if score >= 2:
        return "谨慎"
    return "安全"

@st.cache_data(ttl=3600*12)
def get_market_strong(trade_date):
    start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
    df = safe_api("index_daily", ts_code="000300.SH", start_date=start, end_date=trade_date)
    if df.empty or len(df) < 20:
        return False
    df = df.sort_values("trade_date")
    return float(df.iloc[-1]["close"]) > df["close"].tail(20).mean()

@st.cache_data(ttl=3600*12)
def get_sector_scores(trade_date):
    global pro
    start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d")
    scores = {}
    try:
        sw = pro.index_classify(level="L1", src="SW2021")
        for code in sw["index_code"].tolist():
            idf = safe_api("sw_daily", index_code=code, start_date=start, end_date=trade_date)
            if idf.empty or len(idf) < 3:
                continue
            idf = idf.sort_values("trade_date")
            ret_1d = float(idf.iloc[-1]["pct_chg"]) if "pct_chg" in idf.columns else 0
            ret_5d = (float(idf.iloc[-1]["close"]) / float(idf.iloc[0]["close"]) - 1) * 100
            s = 0
            if ret_1d >= 1.5:
                s += 5
            elif ret_1d >= 0.5:
                s += 3
            if ret_5d >= 3:
                s += 5
            elif ret_5d >= 1:
                s += 3
            scores[code] = min(s, 10)
    except:
        pass
    return scores
def run_screen(trade_date, top_n, min_price, min_mv, max_mv,
               max_turnover, for_backtest=False):
    global pro
    market_strong = get_market_strong(trade_date)
    sector_scores = get_sector_scores(trade_date)
    industry_map = load_industry_map()
    basics = load_stock_basic()

    if for_backtest:
        daily, basic, mf, chip = get_day_data(trade_date)
    else:
        daily = safe_api_nocache("daily", trade_date=trade_date)
        basic = safe_api_nocache("daily_basic", trade_date=trade_date,
                                 fields="ts_code,turnover_rate,circ_mv,pe,pb,volume_ratio")
        mf = safe_api_nocache("moneyflow", trade_date=trade_date)
        chip = safe_api_nocache("cyq_perf", trade_date=trade_date)

    if daily.empty:
        return pd.DataFrame()

    df = daily.copy()
    if not basic.empty:
        df = df.merge(basic, on="ts_code", how="left")
    else:
        df["circ_mv"] = 0
        df["turnover_rate"] = 5
        df["volume_ratio"] = 1

    if not mf.empty:
        mf_cols = ["ts_code", "net_mf_amount"] if "net_mf_amount" in mf.columns else ["ts_code"]
        df = df.merge(mf[mf_cols], on="ts_code", how="left")
    else:
        df["net_mf_amount"] = 0

    chip_dict = {}
    if not chip.empty and "winner_rate" in chip.columns:
        chip_dict = dict(zip(chip["ts_code"], chip["winner_rate"]))

    df = df.merge(basics[["ts_code", "name"]], on="ts_code", how="inner")
    df["circ_mv_b"] = df["circ_mv"] / 10000
    df["net_mf_amount"] = df["net_mf_amount"].fillna(0)
    df["turnover_rate"] = df["turnover_rate"].fillna(5)
    df["volume_ratio"] = df["volume_ratio"].fillna(1)

    df = df[df["close"] >= min_price]
    df = df[df["circ_mv_b"] >= min_mv]
    df = df[df["circ_mv_b"] <= max_mv]
    df = df[df["turnover_rate"] <= max_turnover]

    df["winner_rate"] = df["ts_code"].map(chip_dict).fillna(60)
    df = df[(df["winner_rate"] >= 45) & (df["winner_rate"] <= 88)]

    df = df.sort_values("pct_chg", ascending=False).head(200)

    records = []
    for row in df.itertuples():
        if for_backtest:
            hist = get_stock_history_fast(row.ts_code, trade_date, lookback=90)
        else:
            hist = get_stock_history_live(row.ts_code, trade_date, lookback=90)
        if hist.empty:
            continue

        extra = {
            "winner_rate": chip_dict.get(row.ts_code, 60),
            "net_mf_amount": getattr(row, "net_mf_amount", 0),
            "turnover_rate": getattr(row, "turnover_rate", 5),
            "volume_ratio": getattr(row, "volume_ratio", 1),
        }

        ind = calc_indicators(hist, trade_date, extra=extra)
        if ind is None:
            continue
        if ind["close"] <= ind["ma20"]:
            continue
        if ind["ma20"] <= ind["ma20_prev"]:
            continue
        if ind["upper_shadow"] > 5:
            continue
        if ind["body_pos"] < 0.6:
            continue
        if ind["consec_limit"]:
            continue
        if ind["pct_20d"] > 30:
            continue
        if ind["pct_5d"] > 20:
            continue

        ind_code = industry_map.get(row.ts_code, "")
        s_score = sector_scores.get(ind_code, 3)
        score, detail = calc_score(ind, s_score, market_strong)
        tag = risk_tag(ind)

        rec = {
            "ts_code": row.ts_code,
            "name": row.name,
            "close": ind["close"],
            "pct_1d": ind["pct_1d"],
            "pct_5d": ind["pct_5d"],
            "pct_20d": ind["pct_20d"],
            "from_bottom": ind["from_bottom"],
            "rsi": round(ind["rsi"], 1),
            "winner_rate": ind["winner_rate"],
            "net_mf": round(ind["net_mf"] / 10000, 1),
            "turnover_rate": round(ind["turnover_rate"], 1),
            "vol_ratio": round(ind["vol_ratio"], 2),
            "score": score,
            "tag": tag,
            "market": "强势" if market_strong else "弱势",
            "tech": detail["tech"],
            "timing": detail["timing"],
            "vol_score": detail["vol"],
            "fish": detail["fish"],
            "chip_mf": detail["chip_mf"],
            "sector": detail["sector"],
            "mkt_score": detail["market"],
            "buy_low": round(ind["close"], 2),
            "buy_high": round(ind["close"] * 1.02, 2),
            "stop_loss": round(ind["close"] * 0.95, 2),
            "target": round(ind["close"] * 1.08, 2),
        }

        if for_backtest:
            d0_close = ind["close"]
            d0 = datetime.strptime(trade_date, "%Y%m%d")
            start_f = (d0 + timedelta(days=1)).strftime("%Y%m%d")
            end_f = (d0 + timedelta(days=20)).strftime("%Y%m%d")
            fut = safe_api_nocache("daily", ts_code=row.ts_code,
                                   start_date=start_f, end_date=end_f)
            rec["R_D1"] = np.nan
            rec["R_D3"] = np.nan
            rec["R_D5"] = np.nan
            rec["triggered"] = False
            if not fut.empty:
                fut = fut.sort_values("trade_date").reset_index(drop=True)
                if len(fut) >= 1:
                    next_open = float(fut.iloc[0]["open"])
                    next_high = float(fut.iloc[0]["high"])
                    if next_open > d0_close:
                        trigger = next_open * 1.015
                        if next_high >= trigger:
                            rec["triggered"] = True
                            for n, key in [(1,"R_D1"),(3,"R_D3"),(5,"R_D5")]:
                                if len(fut) >= n:
                                    sell = float(fut.iloc[n-1]["close"])
                                    rec[key] = round((sell - trigger) / trigger * 100, 2)
        records.append(rec)

    if not records:
        return pd.DataFrame()
    result = pd.DataFrame(records)
    result = result.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result)+1))
    return result

# ---------------------------
# 侧边栏
# ---------------------------
with st.sidebar:
    st.header("参数设置")
    token = st.text_input("Tushare Token", type="password")
    st.subheader("股票池过滤")
    min_price = st.number_input("最低股价(元)", value=10.0, min_value=1.0, step=1.0)
    min_mv = st.number_input("最小流通市值(亿)", value=50.0, min_value=10.0, step=10.0)
    max_mv = st.number_input("最大流通市值(亿)", value=1000.0, min_value=100.0, step=100.0)
    max_turnover = st.number_input("最大换手率(%)", value=15.0, min_value=1.0, step=1.0)
    st.subheader("实盘选股")
    top_n = st.slider("输出Top N候选股", 3, 10, 5)
    st.subheader("回测设置")
    bt_end = st.date_input("回测截止日期", value=datetime.now().date())
    bt_days = st.number_input("回测天数", value=30, min_value=5, max_value=90, step=5)
    bt_top_n = st.number_input("每日推荐数", value=4, min_value=1, max_value=10)
    resume = st.checkbox("开启断点续传", value=True)
    if st.button("清除缓存"):
        for f in [CHECKPOINT_FILE, CACHE_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.success("缓存已清除")

# ---------------------------
# Token初始化
# ---------------------------
if not token:
    st.info("请在左侧输入 Tushare Token 后开始使用")
    st.stop()
ts.set_token(token)
pro = ts.pro_api()

# ---------------------------
# 主界面
# ---------------------------
tab1, tab2 = st.tabs(["实盘选股", "历史回测"])

with tab1:
    st.subheader("实盘选股 - 今日候选")
    st.caption("收盘后运行，第二天9:25判断高开，9:30后冲高1.5%触发买入")
    screen_date = st.date_input("选股日期", value=datetime.now().date())
    if st.button("开始选股"):
        date_str = screen_date.strftime("%Y%m%d")
        with st.spinner("正在分析全市场数据，请稍候..."):
            result = run_screen(date_str, top_n, min_price, min_mv, max_mv,
                                max_turnover, for_backtest=False)
            market_state = get_market_strong(date_str)
        if result.empty:
            st.warning("今日未找到符合条件的股票，可适当放宽参数")
        else:
            ms = "强势（沪深300站上MA20）" if market_state else "弱势（沪深300跌破MA20）"
            st.success(f"筛选完成，共推荐 {len(result)} 只候选股")
            st.info(f"当前大盘状态：{ms}")
            for _, row in result.iterrows():
                with st.expander(f"No.{row['rank']}  {row['name']}（{row['ts_code']}）  {row['tag']}  评分:{row['score']}"):
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("今日涨幅", f"{row['pct_1d']:+.2f}%")
                    c2.metric("5日涨幅", f"{row['pct_5d']:+.2f}%")
                    c3.metric("20日涨幅", f"{row['pct_20d']:+.2f}%")
                    c4.metric("本轮涨幅", f"{row['from_bottom']:+.1f}%")
                    c5,c6,c7,c8 = st.columns(4)
                    c5.metric("RSI", row["rsi"])
                    c6.metric("筹码获利%", f"{row['winner_rate']:.1f}%")
                    c7.metric("资金净流入(万)", f"{row['net_mf']:.0f}")
                    c8.metric("换手率", f"{row['turnover_rate']:.1f}%")
                    cc1,cc2,cc3,cc4 = st.columns(4)
                    cc1.metric("建议买入区间", f"{row['buy_low']}~{row['buy_high']}")
                    cc2.metric("止损价(-5%)", row["stop_loss"])
                    cc3.metric("目标价(+8%)", row["target"])
                    cc4.metric("大盘环境", row["market"])
                    dims   = ["tech","timing","vol_score","fish","chip_mf","sector","mkt_score"]
                    labels = ["技术/25","时机/20","量能/15","鱼身/15","资金筹码/15","板块/10","大盘/10"]
                    maxes  = [25,20,15,15,15,10,10]
                    dcols  = st.columns(7)
                    for i,(d,lb,mx) in enumerate(zip(dims,labels,maxes)):
                        dcols[i].metric(lb, f"{row[d]}/{mx}")
            show_df = result[["rank","name","ts_code","close","pct_1d","pct_5d",
                               "from_bottom","rsi","winner_rate","net_mf",
                               "turnover_rate","score","tag","buy_low","stop_loss","target"]].copy()
            show_df.columns = ["排名","名称","代码","现价","今日%","5日%",
                                "本轮%","RSI","筹码%","净流入万","换手%",
                                "评分","风险","买入","止损","目标"]
            st.dataframe(show_df, use_container_width=True)
            csv = show_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("导出CSV", csv, f"result_{date_str}.csv", "text/csv")
    st.caption("本工具仅供学习研究，不构成投资建议。股市有风险，投资需谨慎。")

with tab2:
    st.subheader("历史回测 - 策略验证")
    st.caption("模拟：次日高开+盘中冲高1.5%触发买入，持有N天后收盘价卖出")
    if st.button("启动回测"):
        end_str = bt_end.strftime("%Y%m%d")
        cal = safe_api_nocache("trade_cal",
                               start_date=(bt_end - timedelta(days=int(bt_days)*3)).strftime("%Y%m%d"),
                               end_date=end_str)
        if cal.empty:
            st.error("无法获取交易日历，请检查Token")
            st.stop()
        dates = cal[cal["is_open"]==1].sort_values("cal_date")["cal_date"].tail(int(bt_days)).tolist()
        if not load_market_cache(dates):
            st.stop()
        processed = set()
        results = []
        if resume and os.path.exists(CHECKPOINT_FILE):
            try:
                ex = pd.read_csv(CHECKPOINT_FILE)
                processed = set(ex["trade_date"].astype(str).unique())
                results.append(ex)
                st.success(f"读取断点存档，已跳过 {len(processed)} 个交易日")
            except:
                pass
        dates_to_run = [d for d in dates if d not in processed]
        if not dates_to_run:
            st.success("所有日期已计算完毕！")
        else:
            bar = st.progress(0, text="回测中...")
            err_ph = st.empty()
            for i, date in enumerate(dates_to_run):
                try:
                    res = run_screen(date, int(bt_top_n), min_price, min_mv, max_mv,
                                     max_turnover, for_backtest=True)
                    if not res.empty:
                        res["trade_date"] = date
                        first = not os.path.exists(CHECKPOINT_FILE)
                        res.to_csv(CHECKPOINT_FILE, mode="a", index=False,
                                   header=first, encoding="utf-8-sig")
                        results.append(res)
                except Exception as e:
                    err_ph.warning(f"{date} 处理异常: {e}")
                bar.progress((i+1)/len(dates_to_run),
                             text=f"回测中: {date} ({i+1}/{len(dates_to_run)})")
            bar.empty()
        if results:
            final = pd.concat(results).reset_index(drop=True)
            final = final.sort_values(["trade_date","rank"], ascending=[False,True])
            st.header("回测统计报告 V2.0")
            col1, col2 = st.columns(2)
            col1.metric("总选股记录", f"{len(final)} 条")
            col1.metric("涉及交易日", f"{final['trade_date'].nunique()} 天")
            triggered = final[final["triggered"]==True] if "triggered" in final.columns else pd.DataFrame()
            if not triggered.empty:
                col2.metric("触发买入比例", f"{len(triggered)/len(final)*100:.1f}%")
                cols3 = st.columns(3)
                for i, n in enumerate([1,3,5]):
                    key = f"R_D{n}"
                    if key in triggered.columns:
                        valid = triggered.dropna(subset=[key])
                        if not valid.empty:
                            avg = valid[key].mean()
                            win = (valid[key] > 0).mean() * 100
                            loss = (valid[key] < -5).mean() * 100
                            cols3[i].metric(f"D+{n} 均收益/胜率",
                                            f"{avg:.2f}% / {win:.1f}%",
                                            delta=f"亏损>5%: {loss:.1f}%")
                st.subheader("按排名分层分析")
                for n in [1,3,5]:
                    key = f"R_D{n}"
                    if key not in triggered.columns:
                        continue
                    valid = triggered.dropna(subset=[key])
                    if valid.empty:
                        continue
                    grp = valid.groupby("rank")[key].agg(
                        avg="mean",
                        win_rate=lambda x: (x > 0).mean() * 100,
                        n="count"
                    ).round(2)
                    grp.columns = ["均值%", "胜率%", "样本数"]
                    st.write(f"D+{n} 分层表现：")
                    st.dataframe(grp, use_container_width=True)

                st.subheader("按风险标签分析")
                for n in [1,3,5]:
                    key = f"R_D{n}"
                    if key not in triggered.columns:
                        continue
                    valid = triggered.dropna(subset=[key])
                    if valid.empty or "tag" not in valid.columns:
                        continue
                    grp = valid.groupby("tag")[key].agg(
                        avg="mean",
                        win_rate=lambda x: (x > 0).mean() * 100,
                        n="count"
                    ).round(2)
                    grp.columns = ["均值%", "胜率%", "样本数"]
                    st.write(f"D+{n} 风险标签表现：")
                    st.dataframe(grp, use_container_width=True)
            else:
                col2.info("无触发买入记录")
            st.subheader("回测明细")
            show_cols = ["trade_date","rank","name","ts_code","close","score",
                         "tag","winner_rate","net_mf","turnover_rate",
                         "triggered","R_D1","R_D3","R_D5"]
            show_cols = [c for c in show_cols if c in final.columns]
            st.dataframe(final[show_cols], use_container_width=True)
            csv = final.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下载回测结果CSV", csv, f"backtest_{end_str}.csv", "text/csv")
        else:
            st.warning("回测未产生结果，请检查日期范围或Token权限")
