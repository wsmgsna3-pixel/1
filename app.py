# -*- coding: utf-8 -*-
"""
周线 SKDJ 底部脱离系统 (V14.7 持仓确认审计版)
------------------------------------------------
1. 【主框架冻结】：V14.5 的50亿股票池、SKDJ买点、原评分、T+1买入与退出规则保持不变。
2. 【排序回归原版】：市值和V40均不参与入选日排序，彻底取消100亿优先。
3. 【双口径研究】：实盘主结果固定Top 3，同时保留原版Top 5排名研究及1/2/5组合对照。
4. 【持仓确认】：记录买入后第3日和第5日确认分，0分单列为F级候选，1分不再误标失败。
5. 【可执行审计】：记录D3收盘/D4开盘、D5收盘/D6开盘及退出后潜在最高收益，但不自动卖出。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import os
import re
import pickle
import gzip
import tempfile
import shutil
import gc
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    fcntl = None

warnings.filterwarnings("ignore")

# ---------------------------
# 全局持久化缓存配置
# ---------------------------
CHECKPOINT_FILE = "skdj_v14_7_holding_confirm_checkpoint.csv"
PROCESSED_DATES_FILE = "skdj_v14_7_holding_confirm_processed_dates.csv"
MARKET_CACHE_FILE = "skdj_market_data_master.pkl"
MARKET_CACHE_DIR = "skdj_market_data_daily_cache"

# ---------------------------
# 页面基础配置
# ---------------------------
st.set_page_config(page_title="SKDJ V14.7 持仓确认审计版", layout="wide")
st.title("🔬 周线 SKDJ 底部脱离系统 (V14.7 持仓确认审计版)")
st.markdown("🔒 **V14.5原版排序 · Top3实盘/Top5研究 · V40只用于买入后D3/D5确认**")

# ---------------------------
# Token 清洗与安全请求模块
# ---------------------------
def clean_token_str(raw_token: str) -> str:
    if not raw_token: return ""
    return re.sub(r'[\s\u3000\ufeff\xa0\r\n]+', '', str(raw_token)).strip()

def verify_token_connection(token_str: str):
    if not token_str:
        return False, "Token 为空，请在侧边栏填入 Token。"
    try:
        ts.set_token(token_str)
        pro = ts.pro_api(token_str)
        test_df = pro.trade_cal(exchange='SSE', start_date='20260801', end_date='20260805')
        if test_df is not None and not test_df.empty:
            return True, "验证通过"
        return False, "Token 校验未返回数据，请检查网络连接。"
    except Exception as e:
        err_msg = str(e)
        if "token不对" in err_msg or "-40001" in err_msg:
            return False, "您的 Token 不正确，请检查复制内容。"
        return False, f"接口校验失败: {err_msg}"

def safe_tushare_call(func, max_retries=3, sleep_time=0.8, **kwargs):
    for attempt in range(max_retries):
        try:
            df = func(**kwargs)
            if df is not None and not df.empty:
                return df
            time.sleep(sleep_time)
        except Exception:
            time.sleep(sleep_time * (attempt + 1))
    return pd.DataFrame()


def _atomic_replace_bytes(write_callback, target_path):
    target_dir = os.path.dirname(os.path.abspath(target_path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(target_path) + ".", suffix=".tmp", dir=target_dir
    )
    os.close(fd)
    try:
        write_callback(temp_path)
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _atomic_write_csv(dataframe, target_path):
    def writer(temp_path):
        dataframe.to_csv(temp_path, index=False, encoding="utf-8-sig")
        with open(temp_path, "rb") as file_obj:
            os.fsync(file_obj.fileno())

    if os.path.exists(target_path):
        try:
            shutil.copy2(target_path, target_path + ".bak")
        except OSError:
            pass
    _atomic_replace_bytes(writer, target_path)


def _read_csv_safely(target_path):
    if not os.path.exists(target_path):
        return pd.DataFrame()
    for candidate in (target_path, target_path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            return pd.read_csv(candidate, encoding="utf-8-sig", low_memory=False)
        except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
    return pd.DataFrame()


@contextmanager
def _checkpoint_lock():
    """只锁定数秒钟的结果提交，不锁下载、计算或页面会话。"""
    lock_path = CHECKPOINT_FILE + ".lock"
    handle = open(lock_path, "a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _commit_scan_date(new_rows, trade_date, candidate_count, config_id):
    """先提交结果，再登记完成日；即使无信号也不会在下次重复扫描。"""
    with _checkpoint_lock():
        if new_rows is not None and not new_rows.empty:
            existing = _read_csv_safely(CHECKPOINT_FILE)
            combined = (
                pd.concat([existing, new_rows], ignore_index=True, sort=False)
                if not existing.empty else new_rows.copy()
            )
            if "Trade_Date" in combined.columns:
                combined["Trade_Date"] = (
                    combined["Trade_Date"].astype(str).str.replace(r"\.0$", "", regex=True)
                )
            dedup_cols = [c for c in ("Trade_Date", "Config_ID", "Strategy_ID", "ts_code") if c in combined.columns]
            if {"Trade_Date", "ts_code"}.issubset(dedup_cols):
                combined = combined.drop_duplicates(dedup_cols, keep="last")
            sort_columns = [col for col in ("Trade_Date", "Strategy_ID", "Rank") if col in combined.columns]
            if sort_columns:
                combined = combined.sort_values(sort_columns, kind="mergesort")
            _atomic_write_csv(combined.reset_index(drop=True), CHECKPOINT_FILE)

        processed = _read_csv_safely(PROCESSED_DATES_FILE)
        completed_row = pd.DataFrame([{
            "Trade_Date": str(trade_date),
            "Config_ID": str(config_id),
            "Candidate_Count": int(candidate_count),
            "Completed_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }])
        processed = (
            pd.concat([processed, completed_row], ignore_index=True, sort=False)
            if not processed.empty else completed_row
        )
        processed["Trade_Date"] = processed["Trade_Date"].astype(str).str.replace(r"\.0$", "", regex=True)
        processed = processed.drop_duplicates(
            [c for c in ("Trade_Date", "Config_ID") if c in processed.columns], keep="last"
        ).sort_values([c for c in ("Config_ID", "Trade_Date") if c in processed.columns])
        _atomic_write_csv(processed.reset_index(drop=True), PROCESSED_DATES_FILE)

# ---------------------------
# 科技白名单池构建
# ---------------------------
@st.cache_data(ttl=3600*24*7, show_spinner=False)
def load_custom_tech_whitelist(token):
    token_c = clean_token_str(token)
    if not token_c: return set(), {}
    
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    
    stock_basic = safe_tushare_call(pro.stock_basic, list_status='L', fields='ts_code,symbol,name,industry,market,list_date')
    if stock_basic.empty: return set(), {}
        
    BOARDS = ("主板", "创业板", "科创板")
    valid_stocks = stock_basic[stock_basic['market'].isin(BOARDS)].copy()
    valid_stocks = valid_stocks[~valid_stocks['name'].str.contains('ST|退', na=False)]
    valid_stocks = valid_stocks[~valid_stocks['ts_code'].str.startswith('92')]
    
    CORE_TECH_L1 = {"电子", "计算机", "通信", "国防军工"}
    EXTENDED_TECH_L1 = {"机械设备", "电力设备", "医药生物", "汽车", "基础化工", "有色金属", "建筑材料"}
    TECH_INDUSTRY_KEYWORDS = {
        "半导体", "电子元件", "元件", "光学光电子", "消费电子", "电子化学品",
        "计算机设备", "软件开发", "IT服务", "通信设备", "军工电子", "航空装备",
        "航天装备", "自动化设备", "机器人", "激光设备", "工控设备", "仪器仪表",
        "电池", "光伏设备", "风电设备", "电网设备", "电机", "医疗器械",
        "生物制品", "汽车电子", "金属新材料", "非金属材料", "膜材料", "碳纤维",
    }
    
    sw_indices = safe_tushare_call(pro.index_classify, level='L1', src='SW2021')
    tech_l1_names = CORE_TECH_L1.union(EXTENDED_TECH_L1)
    target_sw = sw_indices[sw_indices['industry_name'].isin(tech_l1_names)] if not sw_indices.empty else pd.DataFrame()
    
    stock_sw_map = {}
    if not target_sw.empty:
        for _, s_row in target_sw.iterrows():
            idx_code = s_row['index_code']
            ind_name = s_row['industry_name']
            m_df = safe_tushare_call(pro.index_member, index_code=idx_code, is_new='Y')
            if not m_df.empty:
                for c_code in m_df['con_code']:
                    stock_sw_map[c_code] = ind_name
            time.sleep(0.03)
            
    whitelist_set = set()
    name_map = dict(zip(stock_basic['ts_code'], stock_basic['name']))
    
    for _, row in valid_stocks.iterrows():
        code = row['ts_code']
        ind_basic = str(row['industry']) if pd.notna(row['industry']) else ""
        sw_l1 = stock_sw_map.get(code, "")
        
        if sw_l1 in CORE_TECH_L1: whitelist_set.add(code); continue
        if sw_l1 in EXTENDED_TECH_L1:
            if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS) or ind_basic == "" or sw_l1 in {"机械设备", "电力设备", "医药生物"}:
                whitelist_set.add(code); continue
        if any(kw in ind_basic for kw in TECH_INDUSTRY_KEYWORDS): whitelist_set.add(code); continue

    return whitelist_set, name_map

# ---------------------------
# 增量下载引擎
# ---------------------------
def _market_partition_path(trade_date):
    os.makedirs(MARKET_CACHE_DIR, exist_ok=True)
    return os.path.join(MARKET_CACHE_DIR, f"{trade_date}.pkl.gz")


def _market_partition_exists(trade_date):
    try:
        return os.path.getsize(_market_partition_path(trade_date)) >= 100
    except OSError:
        return False


def _valid_market_partition(payload, trade_date):
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != 1 or str(payload.get("trade_date")) != str(trade_date):
        return False
    daily = payload.get("daily")
    adj = payload.get("adj")
    basic = payload.get("daily_basic")
    if not all(isinstance(frame, pd.DataFrame) for frame in (daily, adj, basic)):
        return False
    required_daily = {"ts_code", "trade_date", "open", "high", "low", "close", "vol"}
    required_adj = {"ts_code", "trade_date", "adj_factor"}
    if daily.empty or adj.empty:
        return False
    if not required_daily.issubset(daily.columns) or not required_adj.issubset(adj.columns):
        return False
    # 防止Tushare网络波动只返回一小部分股票，却被永久当成完整缓存。
    if int(payload.get("daily_count", 0)) < 1000:
        return False
    if int(payload.get("adj_count", 0)) < 1000:
        return False
    return True


def _read_market_partition(trade_date):
    partition_path = _market_partition_path(trade_date)
    if not os.path.exists(partition_path):
        return None
    try:
        with gzip.open(partition_path, "rb") as file_obj:
            payload = pickle.load(file_obj)
        if _valid_market_partition(payload, trade_date):
            return payload
        try:
            os.remove(partition_path)
        except OSError:
            pass
        return None
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        try:
            os.remove(partition_path)
        except OSError:
            pass
        return None


def _write_market_partition(payload, trade_date):
    partition_path = _market_partition_path(trade_date)

    def writer(temp_path):
        with gzip.open(temp_path, "wb", compresslevel=3) as file_obj:
            pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
        with open(temp_path, "rb") as file_obj:
            os.fsync(file_obj.fileno())

    _atomic_replace_bytes(writer, partition_path)


def sync_market_data_incrementally(start_date, end_date, token, whitelist_set):
    token_c = clean_token_str(token)
    ts.set_token(token_c)
    pro = ts.pro_api(token_c)
    
    cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_date, end_date=end_date)
    if cal_raw.empty: return []
        
    cal_open = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date', ascending=True)
    all_dates = cal_open['cal_date'].tolist()
    
    today_str = datetime.now().strftime("%Y%m%d")
    valid_dates = [d for d in all_dates if d <= today_str]
    
    missing_dates = [
        d for d in valid_dates
        if not _market_partition_exists(d)
    ]
    
    if missing_dates:
        my_bar = st.progress(0, text=f"📥 检测到 {len(missing_dates)} 天增量行情需要同步...")
        for i, d in enumerate(missing_dates):
            df_d = safe_tushare_call(pro.daily, max_retries=3, sleep_time=0.8, trade_date=d)
            df_a = safe_tushare_call(pro.adj_factor, max_retries=3, sleep_time=0.8, trade_date=d)
            df_b = safe_tushare_call(pro.daily_basic, max_retries=3, sleep_time=0.8, trade_date=d, fields='ts_code,trade_date,circ_mv')

            if not df_d.empty and not df_a.empty:
                payload = {
                    "version": 1,
                    "trade_date": d,
                    "daily_count": len(df_d),
                    "adj_count": len(df_a),
                    "daily": df_d,
                    "adj": df_a,
                    "daily_basic": df_b if not df_b.empty else pd.DataFrame(),
                }
                if _valid_market_partition(payload, d):
                    _write_market_partition(payload, d)

            if (i + 1) % 5 == 0 or i == len(missing_dates) - 1:
                my_bar.progress((i+1)/len(missing_dates), text=f"📥 行情同步中: {i+1}/{len(missing_dates)}")
            time.sleep(0.25)
        my_bar.empty()
    return valid_dates

# ---------------------------
# 极速轻量化内存索引引擎
# ---------------------------
@st.cache_resource(ttl=3600*12, show_spinner=False)
def _build_market_index(valid_dates, whitelist_keys, cache_stamp):
    del cache_stamp
    whitelist_set = set(whitelist_keys)
    with st.spinner("正在构建全样本前复权索引..."):
        daily_list, adj_list, basic_list = [], [], []
        for trade_date in valid_dates:
            payload = _read_market_partition(trade_date)
            if payload is None:
                continue
            df_d = payload['daily']
            df_a = payload['adj']
            df_b = payload['daily_basic']
            if whitelist_set:
                df_d = df_d[df_d['ts_code'].isin(whitelist_set)]
                df_a = df_a[df_a['ts_code'].isin(whitelist_set)]
                if not df_b.empty:
                    df_b = df_b[df_b['ts_code'].isin(whitelist_set)]
            if not df_d.empty and not df_a.empty:
                daily_list.append(df_d)
                adj_list.append(df_a)
                if not df_b.empty:
                    basic_list.append(df_b)

        daily_raw = pd.concat(daily_list, ignore_index=True) if daily_list else pd.DataFrame()
        adj_raw = pd.concat(adj_list, ignore_index=True) if adj_list else pd.DataFrame()
        basic_raw = pd.concat(basic_list, ignore_index=True) if basic_list else pd.DataFrame()

        if daily_raw.empty or adj_raw.empty: return {}, pd.DataFrame()

        merged_all = daily_raw.merge(adj_raw[['ts_code', 'trade_date', 'adj_factor']], on=['ts_code', 'trade_date'], how='inner')
        merged_all['trade_date_str'] = merged_all['trade_date'].astype(str)
        merged_all = merged_all.drop_duplicates(['ts_code', 'trade_date_str'], keep='last')
        merged_all = merged_all.sort_values(['ts_code', 'trade_date_str'])
        del daily_raw, adj_raw, daily_list, adj_list
        gc.collect()

        stock_qfq_dict = {}
        for ts_code, group in merged_all.groupby('ts_code'):
            df_g = group.copy()
            latest_adj = df_g['adj_factor'].iloc[-1]
            if latest_adj > 0:
                for col in ['open', 'high', 'low', 'close', 'pre_close']:
                    if col in df_g.columns:
                        df_g[col] = df_g[col] * df_g['adj_factor'] / latest_adj
            df_g = df_g.set_index('trade_date_str')
            stock_qfq_dict[ts_code] = df_g
        del merged_all
        gc.collect()
            
        if not basic_raw.empty:
            basic_raw['trade_date'] = basic_raw['trade_date'].astype(str)
            basic_indexed = basic_raw.drop_duplicates(subset=['ts_code', 'trade_date']).set_index(['trade_date', 'ts_code'])
        else:
            basic_indexed = pd.DataFrame()

    return stock_qfq_dict, basic_indexed


def load_optimized_market_data(start_date, end_date, token, _whitelist_keys):
    token_c = clean_token_str(token)
    whitelist_set = set(_whitelist_keys)
    valid_dates = sync_market_data_incrementally(start_date, end_date, token_c, whitelist_set)
    if not valid_dates:
        return {}, pd.DataFrame()
    valid_paths = [_market_partition_path(date) for date in valid_dates]
    cache_stamp = (
        sum(os.path.exists(path) for path in valid_paths),
        max((os.path.getmtime(path) for path in valid_paths if os.path.exists(path)), default=0),
    )
    return _build_market_index(tuple(valid_dates), tuple(sorted(whitelist_set)), cache_stamp)

# ---------------------------
# 🚀 核心引擎：翻转打分模型（通用于选股与回测）
# ---------------------------
def compute_breakout_signal(ts_code, end_date, stock_qfq_dict):
    if ts_code not in stock_qfq_dict: return {}
    df_full = stock_qfq_dict[ts_code]
    
    df_daily = df_full[df_full.index <= end_date]
    res = {}
    if df_daily.empty or len(df_daily) < 100: return res

    row_friday = df_daily.iloc[-1]
    is_20cm = any(ts_code.startswith(prefix) for prefix in ['300', '301', '688', '689'])
    limit_rate = 0.195 if is_20cm else 0.095
    pre_close_val = row_friday.get('pre_close', np.nan)
    if pd.isna(pre_close_val) or pre_close_val <= 0:
        pre_close_val = df_daily.iloc[-2]['close'] if len(df_daily) >= 2 else row_friday['open']
            
    is_friday_yiziban = (row_friday['high'] == row_friday['low']) and ((row_friday['close'] - pre_close_val) / pre_close_val >= limit_rate)
    if is_friday_yiziban: return res

    df = df_daily.copy().reset_index()
    df['dt'] = pd.to_datetime(df['trade_date_str'])
    df['year_week'] = df['dt'].dt.strftime('%G_%V') 

    weekly_df = df.groupby('year_week', as_index=False).agg({
        'trade_date_str': 'last', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'
    }).sort_values('trade_date_str').reset_index(drop=True)

    n, m = 6, 3
    if len(weekly_df) < n + 15: return res

    weekly_df['lowv'] = weekly_df['low'].rolling(window=n).min()
    weekly_df['highv'] = weekly_df['high'].rolling(window=n).max()
    diff = (weekly_df['highv'] - weekly_df['lowv']).replace(0, 0.001)

    raw_rsv = (weekly_df['close'] - weekly_df['lowv']) / diff * 100
    weekly_df['rsv'] = raw_rsv.ewm(span=m, adjust=False).mean()
    weekly_df['k'] = weekly_df['rsv'].ewm(span=m, adjust=False).mean()
    weekly_df['d'] = weekly_df['k'].rolling(window=m).mean()
    weekly_df['ma5_vol'] = weekly_df['vol'].shift(1).rolling(window=5).mean()
    weekly_df['ma20'] = weekly_df['close'].rolling(window=20).mean()

    curr_w = weekly_df.iloc[-1]
    prev_w = weekly_df.iloc[-2]
    
    if pd.isna(curr_w['k']) or pd.isna(prev_w['k']) or pd.isna(curr_w['d']): return res

    is_breakout_25 = (curr_w['k'] > 25.0) and (prev_w['k'] <= 25.0)
    is_bullish = curr_w['k'] > curr_w['d']
    if not (is_breakout_25 and is_bullish): return res

    recent_15_weeks = weekly_df.tail(15)
    k_history_before_breakout = recent_15_weeks['k'].iloc[:-1] 
    recent_k_min = k_history_before_breakout.min()
    weeks_under_25 = (k_history_before_breakout < 25.0).sum()
    
    ma20_curr = curr_w['ma20'] if pd.notna(curr_w['ma20']) else curr_w['close']
    trend_type = "均线上方" if curr_w['close'] >= ma20_curr else "均线下方(超跌)"
    vol_ratio = curr_w['vol'] / curr_w['ma5_vol'] if (pd.notna(curr_w['ma5_vol']) and curr_w['ma5_vol'] > 0) else 1.0

    res['is_buy_signal'] = True
    res['k'] = round(curr_w['k'], 2)
    res['d'] = round(curr_w['d'], 2)
    res['recent_k_min'] = round(recent_k_min, 2)
    res['weeks_under_25'] = int(weeks_under_25)
    res['signal_close'] = curr_w['close'] 
    res['trend_type'] = trend_type
    res['vol_ratio'] = round(vol_ratio, 2)
    
    score = 0.0
    if curr_w['close'] >= ma20_curr: score += 20.0
    else: score -= 5.0
        
    if 22.0 <= recent_k_min <= 25.0: score += 30.0    
    elif 15.0 <= recent_k_min < 22.0: score += 15.0   
    elif 5.0 <= recent_k_min < 15.0: score -= 10.0    
    else: score -= 25.0                               
        
    if 1 <= weeks_under_25 <= 2: score += 30.0        
    elif 3 <= weeks_under_25 <= 5: score += 15.0      
    elif 6 <= weeks_under_25 <= 9: score -= 5.0       
    else: score -= 20.0                               
        
    k_val = curr_w['k']
    if 25.0 < k_val <= 32.0: score += 10.0
    elif k_val > 38.0: score -= 10.0
        
    if 1.0 <= vol_ratio <= 2.5: score += 10.0
    elif vol_ratio > 4.0: score -= 15.0

    res['Total_Score'] = round(score, 1)
    return res

# ---------------------------
# V40.6 日线发动确认层（只诊断，不改变V14.5买点）
# ---------------------------
def compute_v40_daily_diagnostics(ts_code, end_date, stock_qfq_dict):
    default_res = {
        'V40_Confirm_Score': np.nan,
        'V40_State': '数据不足',
        'Daily_Breakout10': False,
        'Daily_Trend_OK': False,
        'Daily_MACD_OK': False,
        'Daily_Volume_OK': False,
        'Daily_Candle_OK': False,
        'Daily_Overheat': False,
        'Daily_Vol_Ratio': np.nan,
        'Daily_MA20_Slope5 (%)': np.nan,
        'Daily_Bias20 (%)': np.nan,
        'Daily_Return5 (%)': np.nan,
        'Daily_ATR14 (%)': np.nan,
        'Daily_Breakout_Distance (%)': np.nan,
        'Daily_Body_Ratio (%)': np.nan,
        'Daily_Close_Position (%)': np.nan,
        'Daily_DIF': np.nan,
        'Daily_MACD_Bar': np.nan,
    }
    if ts_code not in stock_qfq_dict:
        return default_res

    df = stock_qfq_dict[ts_code]
    df = df[df.index <= str(end_date)].copy()
    if len(df) < 120:
        return default_res

    for col in ('open', 'high', 'low', 'close', 'vol'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'vol'])
    if len(df) < 120:
        return default_res

    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    vol = df['vol']

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_bar = (dif - dea) * 2.0

    prev10_high = high.shift(1).rolling(10).max()
    prior5_vol = vol.shift(1).rolling(5).mean()
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = true_range.rolling(14).mean()

    curr_close = float(close.iloc[-1])
    curr_open = float(open_.iloc[-1])
    curr_high = float(high.iloc[-1])
    curr_low = float(low.iloc[-1])
    curr_range = max(curr_high - curr_low, 1e-9)

    vol_ratio = float(vol.iloc[-1] / prior5_vol.iloc[-1]) if pd.notna(prior5_vol.iloc[-1]) and prior5_vol.iloc[-1] > 0 else np.nan
    ma20_slope5 = float((ma20.iloc[-1] / ma20.iloc[-6] - 1.0) * 100.0) if pd.notna(ma20.iloc[-6]) and ma20.iloc[-6] > 0 else np.nan
    bias20 = float((curr_close / ma20.iloc[-1] - 1.0) * 100.0) if pd.notna(ma20.iloc[-1]) and ma20.iloc[-1] > 0 else np.nan
    return5 = float((curr_close / close.iloc[-6] - 1.0) * 100.0) if close.iloc[-6] > 0 else np.nan
    atr14_pct = float(atr14.iloc[-1] / curr_close * 100.0) if pd.notna(atr14.iloc[-1]) and curr_close > 0 else np.nan
    breakout_distance = float((curr_close / prev10_high.iloc[-1] - 1.0) * 100.0) if pd.notna(prev10_high.iloc[-1]) and prev10_high.iloc[-1] > 0 else np.nan
    body_ratio = float(max(curr_close - curr_open, 0.0) / curr_range * 100.0)
    close_position = float((curr_close - curr_low) / curr_range * 100.0)

    breakout10 = bool(pd.notna(prev10_high.iloc[-1]) and curr_close > prev10_high.iloc[-1])
    trend_ok = bool(
        pd.notna(ma20.iloc[-1]) and pd.notna(ma60.iloc[-1]) and pd.notna(ma120.iloc[-1])
        and curr_close > ma20.iloc[-1]
        and ma20.iloc[-1] > ma20.iloc[-2]
        and ma60.iloc[-1] > ma120.iloc[-1]
    )
    macd_ok = bool(
        pd.notna(dif.iloc[-1]) and pd.notna(macd_bar.iloc[-1]) and pd.notna(macd_bar.iloc[-2])
        and dif.iloc[-1] > 0 and macd_bar.iloc[-1] > macd_bar.iloc[-2]
    )
    volume_ok = bool(pd.notna(vol_ratio) and 1.3 <= vol_ratio <= 3.0)
    candle_ok = bool(curr_close > curr_open and body_ratio >= 60.0 and close_position >= 70.0)
    overheat = bool(
        (pd.notna(vol_ratio) and vol_ratio > 4.0)
        or (pd.notna(return5) and return5 > 15.0)
        or (pd.notna(bias20) and bias20 > 12.0)
    )

    # 五项均为等权诊断；过热单独标记，不用主观权重覆盖V14.5评分。
    confirm_score = int(sum((breakout10, trend_ok, macd_ok, volume_ok, candle_ok)))
    if confirm_score == 0:
        state = '0分-F级候选'
    elif confirm_score == 1:
        state = '1分-弱观察'
    elif confirm_score == 2:
        state = '2分-中性'
    elif overheat:
        state = '3~5分-确认但过热'
    else:
        state = '3~5分-已确认'

    default_res.update({
        'V40_Confirm_Score': confirm_score,
        'V40_State': state,
        'Daily_Breakout10': breakout10,
        'Daily_Trend_OK': trend_ok,
        'Daily_MACD_OK': macd_ok,
        'Daily_Volume_OK': volume_ok,
        'Daily_Candle_OK': candle_ok,
        'Daily_Overheat': overheat,
        'Daily_Vol_Ratio': round(vol_ratio, 3) if pd.notna(vol_ratio) else np.nan,
        'Daily_MA20_Slope5 (%)': round(ma20_slope5, 3) if pd.notna(ma20_slope5) else np.nan,
        'Daily_Bias20 (%)': round(bias20, 3) if pd.notna(bias20) else np.nan,
        'Daily_Return5 (%)': round(return5, 3) if pd.notna(return5) else np.nan,
        'Daily_ATR14 (%)': round(atr14_pct, 3) if pd.notna(atr14_pct) else np.nan,
        'Daily_Breakout_Distance (%)': round(breakout_distance, 3) if pd.notna(breakout_distance) else np.nan,
        'Daily_Body_Ratio (%)': round(body_ratio, 2),
        'Daily_Close_Position (%)': round(close_position, 2),
        'Daily_DIF': round(float(dif.iloc[-1]), 4),
        'Daily_MACD_Bar': round(float(macd_bar.iloc[-1]), 4),
    })
    return default_res


def build_research_groups(candidate_df, portfolio_slots=3, research_top_n=5):
    """原版排序不动，同时生成Top3、1/2/5组合、Top5研究和全候选审计。"""
    if candidate_df.empty:
        return pd.DataFrame()

    portfolio_slots = max(int(portfolio_slots), 1)
    research_top_n = max(int(research_top_n), 5, portfolio_slots)
    # whitelist_keys本来就是按代码排序；显式加入ts_code保证同分结果稳定且可复现。
    base_sorted = candidate_df.sort_values(
        ['Total_Score', 'ts_code'], ascending=[False, True], kind='mergesort', na_position='last'
    ).copy()
    base_sorted['Original_Rank'] = np.arange(1, len(base_sorted) + 1)
    top123 = base_sorted.head(portfolio_slots)
    research_top = base_sorted.head(research_top_n)

    desired_ranks = [1, 2, 5]
    combo = research_top[research_top['Original_Rank'].isin(desired_ranks)].copy()
    if len(combo) < portfolio_slots:
        fallback = research_top[~research_top.index.isin(combo.index)].sort_values('Original_Rank')
        combo = pd.concat([combo, fallback.head(portfolio_slots - len(combo))], sort=False)
    combo = combo.head(portfolio_slots)

    group_defs = [
        ('TOP123', '原版Top1/2/3主结果', top123),
        ('TOP125', 'Top1/2/5研究组合', combo),
        ('RESEARCH5', f'原版Top{research_top_n}排名研究', research_top),
        ('AUDIT_ALL', '全候选审计', base_sorted),
    ]
    frames = []
    for strategy_id, strategy_name, frame in group_defs:
        if frame.empty:
            continue
        out = frame.copy().reset_index(drop=True)
        out['Rank'] = np.arange(1, len(out) + 1)
        out['Strategy_ID'] = strategy_id
        out['Strategy_Name'] = strategy_name
        frames.append(out)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


# ---------------------------
# 🚀 独立出局系统 (仅限回测模式使用)
# ---------------------------
def track_future_performance(ts_code, selection_date, signal_close, stock_qfq_dict, hold_weeks=12):
    default_res = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    default_res.update({
        'Exit_Reason': '持仓中', 'Buy_Price': np.nan, 'Gap_pct (%)': np.nan, 
        'Exit_Date': None, 'Final_Return (%)': np.nan, 'Hold_Days': 0,
        'Confirm_D3_Score': np.nan, 'Confirm_D3_State': '未到第3日',
        'Confirm_D5_Score': np.nan, 'Confirm_D5_State': '未到第5日',
        'Return_D3_Close (%)': np.nan, 'Return_D4_Open (%)': np.nan,
        'Return_D5_Close (%)': np.nan, 'Return_D6_Open (%)': np.nan,
        'D4_Open_Date': None, 'D6_Open_Date': None,
        'Max_After_D4_Open (%)': np.nan, 'Max_After_D6_Open (%)': np.nan,
        'Final_If_D3_Zero_Exit_D4_Open (%)': np.nan,
        'Final_If_D5_Zero_Exit_D6_Open (%)': np.nan,
        'D3_Zero_Missed_Max (%)': np.nan, 'D5_Zero_Missed_Max (%)': np.nan,
    })
    
    if ts_code not in stock_qfq_dict: return default_res
    df_full = stock_qfq_dict[ts_code]
    hist_future = df_full[df_full.index > selection_date]
    results = default_res.copy()
    
    if hist_future.empty: return results

    next_row = hist_future.iloc[0]
    buy_price = next_row['open']
    if pd.isna(buy_price) or buy_price <= 0: return results

    is_20cm = any(ts_code.startswith(prefix) for prefix in ['300', '301', '688', '689'])
    limit_rate_pct = 19.0 if is_20cm else 9.5
    gap_pct = (buy_price - signal_close) / signal_close * 100.0
    
    is_monday_yiziban = (next_row['open'] == next_row['high'] == next_row['low']) and (gap_pct >= limit_rate_pct)
    if is_monday_yiziban:
        results['Exit_Reason'] = f"一字板无法买入(剔除: {round(gap_pct, 1)}%)"
        results['Buy_Price'] = round(buy_price, 2)  
        return results

    if is_20cm and gap_pct > 8.0:
        results['Exit_Reason'] = f"双创高开过大(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results
    elif not is_20cm and gap_pct > 5.0:
        results['Exit_Reason'] = f"主板高开过大(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results
    if gap_pct < -4.0:
        results['Exit_Reason'] = f"恶劣低开(剔除: {round(gap_pct, 2)}%)"
        results['Buy_Price'] = round(buy_price, 2)
        return results

    results['Buy_Price'] = round(buy_price, 2)
    exit_triggered = False
    tier = 0  
    peak_price = buy_price
    pending_exit_reason = None  
    hard_stop_limit = -0.10 
    
    max_days = hold_weeks * 5
    for i in range(len(hist_future)):
        if i >= max_days: break 
            
        row = hist_future.iloc[i]
        day_count = i + 1
        current_week = ((day_count - 1) // 5) + 1 
        curr_open, curr_close, curr_high, curr_low = row['open'], row['close'], row['high'], row['low']
        curr_date = hist_future.index[i]

        if day_count == 4:
            results['Return_D4_Open (%)'] = round((curr_open - buy_price) / buy_price * 100.0, 2)
            results['D4_Open_Date'] = curr_date
        elif day_count == 6:
            results['Return_D6_Open (%)'] = round((curr_open - buy_price) / buy_price * 100.0, 2)
            results['D6_Open_Date'] = curr_date
        
        if pending_exit_reason is not None and day_count >= 2:
            if "保本" in pending_exit_reason: final_return = 2.0  
            else: final_return = (curr_open - buy_price) / buy_price * 100.0
            exit_triggered = True
            results['Exit_Reason'] = pending_exit_reason
            results['Final_Return (%)'] = round(final_return, 2)
            results['Exit_Date'] = curr_date
            results['Hold_Days'] = day_count
            results[f'Return_W{current_week} (%)'] = round(final_return, 2)
            break
        
        peak_price = max(peak_price, curr_high)
        peak_profit_pct = (peak_price - buy_price) / buy_price
        
        if day_count >= 2:
            if (curr_low - buy_price) / buy_price <= hard_stop_limit:
                final_return = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
                exit_triggered = True
                results['Exit_Reason'] = "认栽出局(破-10%)"
                results['Final_Return (%)'] = round(final_return, 2)
                results['Exit_Date'] = curr_date
                results['Hold_Days'] = day_count
                results[f'Return_W{current_week} (%)'] = round(final_return, 2)
                break

        # 只记录确认状态，不据此改变原V14.5退出规则。
        if day_count in (3, 5):
            confirm_diag = compute_v40_daily_diagnostics(ts_code, curr_date, stock_qfq_dict)
            results[f'Confirm_D{day_count}_Score'] = confirm_diag.get('V40_Confirm_Score', np.nan)
            results[f'Confirm_D{day_count}_State'] = confirm_diag.get('V40_State', '数据不足')
            results[f'Return_D{day_count}_Close (%)'] = round(
                (curr_close - buy_price) / buy_price * 100.0, 2
            )
        
        if tier == 0 and peak_profit_pct >= 0.10: tier = 1  
        if tier == 1:
            if curr_close <= buy_price * 1.02: pending_exit_reason = "保本离场(+2%)"
            elif peak_profit_pct >= 0.20: tier = 2  
        if tier == 2:
            giveback = (peak_price - curr_close) / peak_price
            if giveback >= 0.15: pending_exit_reason = "移动止盈(回撤15%)"
        
        if day_count == 5 and not exit_triggered and pending_exit_reason is None:
            w1_ret = (curr_close - buy_price) / buy_price * 100.0
            if w1_ret <= -3.0:
                exit_triggered = True
                results['Exit_Reason'] = f"首周不及预期截断({round(w1_ret, 1)}%)"
                results['Final_Return (%)'] = round(w1_ret, 2)
                results['Exit_Date'] = curr_date
                results['Hold_Days'] = 5
                results['Return_W1 (%)'] = round(w1_ret, 2)
                break
            
        if day_count % 5 == 0:
            results[f'Return_W{current_week} (%)'] = round((curr_close - buy_price) / buy_price * 100.0, 2)
            
    if not exit_triggered and len(hist_future) >= max_days:
        last_price = hist_future.iloc[max_days - 1]['close']
        final_return = (last_price - buy_price) / buy_price * 100.0
        results[f'Return_W{hold_weeks} (%)'] = round(final_return, 2)
        results['Exit_Reason'] = "12周期满平仓"
        results['Final_Return (%)'] = round(final_return, 2)
        results['Exit_Date'] = hist_future.index[max_days - 1]
        results['Hold_Days'] = max_days

    # 反事实审计只观察，不改变原始退出。最高收益统一限制在原12周期持仓窗口内。
    horizon = hist_future.iloc[:min(max_days, len(hist_future))]
    if len(horizon) >= 4:
        if pd.isna(results.get('Return_D4_Open (%)')):
            d4_row = horizon.iloc[3]
            results['Return_D4_Open (%)'] = round((d4_row['open'] - buy_price) / buy_price * 100.0, 2)
            results['D4_Open_Date'] = horizon.index[3]
        max_after_d4 = (horizon.iloc[3:]['high'].max() - buy_price) / buy_price * 100.0
        results['Max_After_D4_Open (%)'] = round(max_after_d4, 2)
    if len(horizon) >= 6:
        if pd.isna(results.get('Return_D6_Open (%)')):
            d6_row = horizon.iloc[5]
            results['Return_D6_Open (%)'] = round((d6_row['open'] - buy_price) / buy_price * 100.0, 2)
            results['D6_Open_Date'] = horizon.index[5]
        max_after_d6 = (horizon.iloc[5:]['high'].max() - buy_price) / buy_price * 100.0
        results['Max_After_D6_Open (%)'] = round(max_after_d6, 2)

    original_final = results.get('Final_Return (%)', np.nan)
    d3_zero = results.get('Confirm_D3_Score') == 0
    d5_zero = results.get('Confirm_D5_Score') == 0
    d4_open_ret = results.get('Return_D4_Open (%)', np.nan)
    d6_open_ret = results.get('Return_D6_Open (%)', np.nan)
    results['Final_If_D3_Zero_Exit_D4_Open (%)'] = (
        d4_open_ret if d3_zero and pd.notna(d4_open_ret) else original_final
    )
    results['Final_If_D5_Zero_Exit_D6_Open (%)'] = (
        d6_open_ret if d5_zero and pd.notna(d6_open_ret) else original_final
    )
    if d3_zero and pd.notna(d4_open_ret) and pd.notna(results.get('Max_After_D4_Open (%)')):
        results['D3_Zero_Missed_Max (%)'] = round(
            results['Max_After_D4_Open (%)'] - d4_open_ret, 2
        )
    if d5_zero and pd.notna(d6_open_ret) and pd.notna(results.get('Max_After_D6_Open (%)')):
        results['D5_Zero_Missed_Max (%)'] = round(
            results['Max_After_D6_Open (%)'] - d6_open_ret, 2
        )
        
    return results

def repair_checkpoint_df(df_in):
    df_out = df_in.copy()
    w_cols = [c for c in df_out.columns if c.startswith('Return_W') and c.endswith('(%)')]
    if w_cols: w_cols = sorted(w_cols, key=lambda x: int(x.replace('Return_W', '').replace(' (%)', '')))
    
    if 'Final_Return (%)' not in df_out.columns:
        def get_final_ret(r):
            if not w_cols: return 0.0
            rets = r[w_cols].dropna()
            return rets.iloc[-1] if not rets.empty else 0.0
        df_out['Final_Return (%)'] = df_out.apply(get_final_ret, axis=1)
    if 'Exit_Date' not in df_out.columns: df_out['Exit_Date'] = None
    if 'Hold_Days' not in df_out.columns:
        def get_hold_days(r):
            if not w_cols: return 0
            rets = r[w_cols].dropna()
            return len(rets) * 5 if not rets.empty else 0
        df_out['Hold_Days'] = df_out.apply(get_hold_days, axis=1)
    return df_out

# ---------------------------
# UI 控制流与输入侧边栏
# ---------------------------
with st.sidebar:
    st.header("⚙️ 模式与分析配置")
    
    st.info("💡 **双模引擎说明**：\n将追溯天数设为 **1**，触发【盘中极速选股】(不保存)；设为 **>1**，触发【历史回测】。系统已锁定回测模式仅在周末生效，防止污染。")
    
    BACKTEST_DAYS = st.number_input("追溯交易天数 (设为1为极速选股)", value=1, step=30, min_value=1)
    PORTFOLIO_SLOTS = st.number_input("实盘仓位数", value=3, min_value=1, max_value=5, step=1)
    RESEARCH_TOP_N = st.number_input("排名研究数量 (至少5名)", value=5, min_value=5, max_value=20, step=1)
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    
    st.markdown("---")
    if st.button("🗑️ 清空行情缓存"):
        if os.path.isdir(MARKET_CACHE_DIR): shutil.rmtree(MARKET_CACHE_DIR)
        for cache_path in (MARKET_CACHE_FILE, MARKET_CACHE_FILE + ".tmp"):
            if os.path.exists(cache_path): os.remove(cache_path)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("底层行情缓存已清理！")
            
    if st.button("🗑️ 清除历史回测记录"):
        for result_path in (
            CHECKPOINT_FILE, CHECKPOINT_FILE + ".bak",
            PROCESSED_DATES_FILE, PROCESSED_DATES_FILE + ".bak",
        ):
            if os.path.exists(result_path): os.remove(result_path)
        st.success("历史记录已清理！")
            
    st.markdown("---")
    st.subheader("💰 护城河底座")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("扫描最低流通市值(亿)", value=50.0)
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)
    st.caption("流通市值只保留为审计字段，不参与排序。")
    
    st.markdown("---")
    secret_token = st.secrets.get("TUSHARE_TOKEN", "") if hasattr(st, "secrets") else ""
    TS_TOKEN_INPUT = st.text_input("🔑 Tushare Token", value=secret_token, type="password")

token_clean = clean_token_str(TS_TOKEN_INPUT)
is_picking_mode = (int(BACKTEST_DAYS) == 1)
CURRENT_CONFIG_ID = (
    f"price{float(MIN_PRICE):.2f}_mv{float(MIN_MV):.2f}"
    f"_max{float(MAX_MV):.2f}_slots{int(PORTFOLIO_SLOTS)}_research{int(RESEARCH_TOP_N)}"
)

# ---------------------------
# 主流程：启动引擎
# ---------------------------
btn_label = "🚀 启动盘中极速选股 (天数=1)" if is_picking_mode else "🚀 启动历史回测分析 (天数>1)"

if st.button(btn_label):
    is_valid, msg = verify_token_connection(token_clean)
    if not is_valid:
        st.error(f"❌ **Token 预检拦截**：{msg}")
    else:
        try:
            # 每次新任务先释放上一轮大型内存索引，避免连续回测叠加占用内存。
            st.cache_resource.clear()
            ts.set_token(token_clean)
            pro = ts.pro_api(token_clean)
            
            with st.spinner("正在精准筛选科技池白名单标的..."):
                whitelist_set, basic_name_map = load_custom_tech_whitelist(token_clean)
                whitelist_keys = tuple(sorted(whitelist_set))
                
            if not whitelist_keys:
                st.error("❌ 未能获取到科技白名单股票，请检查 Token 积分或网络。")
            else:
                st.info(f"💡 成功锁定科技白名单股票池：共 **{len(whitelist_keys)}** 只标的。")
                
                # 🌟 修复关键点：强制系统多看 15 天的日历，确保准确识别未来的周末
                lookback_days = max(int(BACKTEST_DAYS) * 3, 900) 
                start_cal = (datetime.strptime(backtest_date_end.strftime("%Y%m%d"), "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
                end_cal_extended = (datetime.strptime(backtest_date_end.strftime("%Y%m%d"), "%Y%m%d") + timedelta(days=15)).strftime("%Y%m%d")
                
                cal_raw = safe_tushare_call(pro.trade_cal, exchange='SSE', start_date=start_cal, end_date=end_cal_extended)
                if cal_raw.empty:
                    st.error("❌ 无法获取交易日历。")
                else:
                    cal_open = cal_raw[cal_raw['is_open'] == 1].sort_values('cal_date', ascending=True)
                    all_trade_days = cal_open['cal_date'].tolist()
                    
                    # 过滤出小于等于我们所选截止日期的真实可交易日列表
                    end_str = backtest_date_end.strftime("%Y%m%d")
                    trade_days_list = [d for d in all_trade_days if d <= end_str]
                    
                    if not trade_days_list:
                        st.error("❌ 未获取到有效交易日。")
                    else:
                        td_df = pd.DataFrame({'cal_date': all_trade_days})
                        td_df['dt'] = pd.to_datetime(td_df['cal_date'])
                        td_df['year_week'] = td_df['dt'].dt.strftime('%G_%V')
                        
                        # 🌟 核心模式分流与周末防线
                        if is_picking_mode:
                            dates_to_run = [trade_days_list[-1]] 
                        else:
                            # 找出所有包含在完整日历里的“真正的一周最后一天”
                            valid_scan_dates = set(td_df.groupby('year_week')['cal_date'].max().tolist())
                            
                            processed_dates = set()
                            if os.path.exists(PROCESSED_DATES_FILE):
                                try:
                                    existing_df = _read_csv_safely(PROCESSED_DATES_FILE)
                                    if 'Config_ID' in existing_df.columns:
                                        existing_df = existing_df[existing_df['Config_ID'].astype(str) == CURRENT_CONFIG_ID]
                                    existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
                                    processed_dates = set(existing_df['Trade_Date'].unique())
                                except Exception: pass
                            elif os.path.exists(CHECKPOINT_FILE):
                                # 兼容首次升级时已有的结果文件。
                                try:
                                    existing_df = _read_csv_safely(CHECKPOINT_FILE)
                                    if 'Config_ID' in existing_df.columns:
                                        existing_df = existing_df[existing_df['Config_ID'].astype(str) == CURRENT_CONFIG_ID]
                                    existing_df['Trade_Date'] = existing_df['Trade_Date'].astype(str)
                                    processed_dates = set(existing_df['Trade_Date'].unique())
                                except Exception: pass
                            recent_trade_days = trade_days_list[-int(BACKTEST_DAYS):]
                            
                            # 只有这一天既在最近回测范围内，又是真正的周末，且没被处理过，才允许进入回测
                            dates_to_run = [d for d in recent_trade_days if d not in processed_dates and d in valid_scan_dates]
                            dates_to_run.sort()
                        
                        if not dates_to_run and not is_picking_mode:
                            st.success("🎉 指定区间回测数据已全部跑完！(如果您选择了周中日期，系统已自动跳过以保护数据纯洁性)")
                        elif dates_to_run:
                            fetch_start = (datetime.strptime(min(dates_to_run), "%Y%m%d") - timedelta(days=300)).strftime("%Y%m%d")
                            fetch_end = (datetime.strptime(max(dates_to_run), "%Y%m%d") + timedelta(days=200)).strftime("%Y%m%d")
                            
                            stock_qfq_dict, basic_indexed = load_optimized_market_data(fetch_start, fetch_end, token_clean, whitelist_keys)
                            
                            if not stock_qfq_dict:
                                st.warning("⚠️ 未能加载到行情数据，请重试。")
                            else:
                                bar = st.progress(0, text="执行 V14.5 数据扫描...")
                                
                                for i, date in enumerate(dates_to_run):
                                    records = []
                                    for ts_code in whitelist_keys:
                                        if ts_code not in stock_qfq_dict: continue
                                        df_stock = stock_qfq_dict[ts_code]
                                        if date not in df_stock.index: continue
                                            
                                        row_latest = df_stock.loc[date]
                                        if isinstance(row_latest, pd.DataFrame): row_latest = row_latest.iloc[-1]
                                            
                                        curr_close = row_latest['close']
                                        if curr_close < MIN_PRICE: continue
                                            
                                        circ_mv_billion = np.nan
                                        if not basic_indexed.empty and (date, ts_code) in basic_indexed.index:
                                            circ_mv_billion = basic_indexed.loc[(date, ts_code)]['circ_mv'] / 10000.0
                                        
                                        if pd.notna(circ_mv_billion):
                                            if circ_mv_billion < MIN_MV or circ_mv_billion > MAX_MV: continue
                                        
                                        ind = compute_breakout_signal(ts_code, date, stock_qfq_dict)
                                        if not ind or not ind.get('is_buy_signal'): continue

                                        v40_diag = compute_v40_daily_diagnostics(ts_code, date, stock_qfq_dict)
                                        if pd.isna(circ_mv_billion):
                                            cap_tier = "市值未知"
                                        elif circ_mv_billion < 100.0:
                                            cap_tier = "50~100亿"
                                        elif circ_mv_billion < 200.0:
                                            cap_tier = "100~200亿"
                                        elif circ_mv_billion < 500.0:
                                            cap_tier = "200~500亿"
                                        else:
                                            cap_tier = "500亿以上"
                                            
                                        stock_name = basic_name_map.get(ts_code, ts_code)
                                        record_dict = {
                                            'ts_code': ts_code, 'name': stock_name, 'Signal_Close': ind['signal_close'], 
                                            'SKDJ_K': ind['k'], 'SKDJ_D': ind['d'], 
                                            'D_Min(10W)': ind['recent_k_min'], 'Weeks_Under': ind['weeks_under_25'],
                                            'Trend_Type': ind['trend_type'], 'vol_ratio': ind['vol_ratio'],
                                            'circ_mv': round(circ_mv_billion, 2) if pd.notna(circ_mv_billion) else np.nan, 
                                            'Cap_Tier': cap_tier,
                                            'Total_Score': ind['Total_Score'],
                                        }
                                        record_dict.update(v40_diag)
                                        
                                        if not is_picking_mode:
                                            future_returns = track_future_performance(ts_code, date, ind['signal_close'], stock_qfq_dict, hold_weeks=12)
                                            record_dict.update(future_returns)
                                            
                                        records.append(record_dict)
                                            
                                    if records:
                                        candidate_df = pd.DataFrame(records)
                                        selected_groups = build_research_groups(
                                            candidate_df,
                                            portfolio_slots=int(PORTFOLIO_SLOTS),
                                            research_top_n=int(RESEARCH_TOP_N),
                                        )
                                        selected_groups['Trade_Date'] = date
                                        selected_groups['Config_ID'] = CURRENT_CONFIG_ID
                                        
                                        if is_picking_mode:
                                            fdf = selected_groups[selected_groups['Strategy_ID'] == 'TOP123'].copy()
                                            st.subheader(
                                                f"🎯 V14.5原版主结果 [{date}] - Top {PORTFOLIO_SLOTS}"
                                            )
                                            main_cols = [
                                                'Original_Rank', 'name', 'ts_code', 'Total_Score', 'circ_mv', 'Cap_Tier',
                                                'V40_Confirm_Score', 'V40_State', 'Daily_Breakout10',
                                                'Daily_Trend_OK', 'Daily_MACD_OK', 'Daily_Volume_OK',
                                                'Daily_Overheat', 'SKDJ_K', 'Weeks_Under',
                                            ]
                                            main_cols = [c for c in main_cols if c in fdf.columns]
                                            try:
                                                st.dataframe(fdf[main_cols].style.background_gradient(subset=['Total_Score'], cmap='YlOrRd'), width='stretch')
                                            except Exception:
                                                st.dataframe(fdf[main_cols], width='stretch')

                                            with st.expander("查看Top1/2/5组合、Top5排名及全部诊断候选"):
                                                audit_live = selected_groups[selected_groups['Strategy_ID'] != 'TOP123'].copy()
                                                audit_cols = [
                                                    'Strategy_Name', 'Original_Rank', 'name', 'ts_code', 'Total_Score',
                                                    'circ_mv', 'V40_Confirm_Score', 'V40_State',
                                                    'Daily_Vol_Ratio', 'Daily_Bias20 (%)', 'Daily_Return5 (%)',
                                                ]
                                                audit_cols = [c for c in audit_cols if c in audit_live.columns]
                                                st.dataframe(audit_live[audit_cols], width='stretch')
                                        else:
                                            _commit_scan_date(selected_groups, date, len(candidate_df), CURRENT_CONFIG_ID)
                                    elif not is_picking_mode:
                                        _commit_scan_date(pd.DataFrame(), date, 0, CURRENT_CONFIG_ID)
                                        
                                    bar.progress((i+1)/len(dates_to_run), text=f"扫描中: {date} (捕获 {len(records)} 只目标)")
                                    
                                bar.empty()
                                if is_picking_mode:
                                    st.success("🎉 今日极速选股完成！(当前为选股模式，数据未写入历史回测库)")
                                else:
                                    st.success("🎉 回测数据更新完毕！请查看下方报告。")
                                    
        except Exception as e:
            st.error(f"❌ **运行异常拦截**：{str(e)}")

# ---------------------------
# 全景分析展示区
# ---------------------------
if not is_picking_mode and not os.path.exists(CHECKPOINT_FILE) and os.path.exists(PROCESSED_DATES_FILE):
    # 所有扫描周均无候选时也给出明确结果，避免误以为回测没有运行。
    processed_only = _read_csv_safely(PROCESSED_DATES_FILE)
    if not processed_only.empty and 'Config_ID' in processed_only.columns:
        processed_only = processed_only[
            processed_only['Config_ID'].astype(str) == CURRENT_CONFIG_ID
        ].copy()
    if not processed_only.empty:
        completed_count = processed_only['Trade_Date'].astype(str).nunique()
        st.info(f"当前参数已完成 {completed_count} 个扫描周，但没有出现符合V14.5条件的标的。")

if os.path.exists(CHECKPOINT_FILE) and not is_picking_mode:
    st.markdown("---")
    try:
        raw_res = _read_csv_safely(CHECKPOINT_FILE)
        if raw_res.empty:
            raise pd.errors.EmptyDataError
        if 'Config_ID' in raw_res.columns:
            raw_res = raw_res[raw_res['Config_ID'].astype(str) == CURRENT_CONFIG_ID].copy()
        else:
            raw_res['Config_ID'] = CURRENT_CONFIG_ID
        if raw_res.empty:
            raise pd.errors.EmptyDataError
        raw_res['Trade_Date'] = raw_res['Trade_Date'].astype(str)

        repaired_res = repair_checkpoint_df(raw_res)
        if 'Strategy_ID' not in repaired_res.columns:
            repaired_res['Strategy_ID'] = 'TOP123'
            repaired_res['Strategy_Name'] = '原版Top1/2/3主结果'

        processed_df = _read_csv_safely(PROCESSED_DATES_FILE)
        if not processed_df.empty and 'Config_ID' in processed_df.columns:
            processed_df = processed_df[processed_df['Config_ID'].astype(str) == CURRENT_CONFIG_ID].copy()
        completed_dates = (
            sorted(processed_df['Trade_Date'].astype(str).unique().tolist())
            if not processed_df.empty and 'Trade_Date' in processed_df.columns
            else sorted(repaired_res['Trade_Date'].astype(str).unique().tolist())
        )
        total_completed_weeks = len(completed_dates)

        is_rejected = repaired_res['Exit_Reason'].astype(str).str.contains('剔除', na=False)
        valid_all = repaired_res[~is_rejected].copy()
        selection_ids = ['TOP123', 'TOP125', 'RESEARCH5']
        selected_raw = repaired_res[repaired_res['Strategy_ID'].isin(selection_ids)].copy()
        selected_valid = valid_all[valid_all['Strategy_ID'].isin(selection_ids)].copy()
        audit_valid = valid_all[valid_all['Strategy_ID'] == 'AUDIT_ALL'].copy()
        main_signals = selected_valid[selected_valid['Strategy_ID'] == 'TOP123'].copy()
        combo_signals = selected_valid[selected_valid['Strategy_ID'] == 'TOP125'].copy()
        research_signals = selected_valid[selected_valid['Strategy_ID'] == 'RESEARCH5'].copy()

        st.header("📈 V14.7 原版排名与持仓确认审计报告")
        st.caption("V14.5入场和原评分完全不变；V40与市值不参与买入排序，D3/D5退出仅作反事实审计。")

        st.subheader("⚖️ 原版Top1/2/3、Top1/2/5与Top5研究口径")
        strategy_rows = []
        strategy_names = {
            'TOP123': '原版Top1/2/3主结果',
            'TOP125': 'Top1/2/5研究组合',
            'RESEARCH5': f'原版Top{int(RESEARCH_TOP_N)}排名研究',
        }
        for strategy_id in selection_ids:
            raw_group = selected_raw[selected_raw['Strategy_ID'] == strategy_id]
            valid_group = selected_valid[selected_valid['Strategy_ID'] == strategy_id]
            closed_group = valid_group[valid_group['Exit_Reason'].astype(str) != '持仓中'].copy()
            closed_group['Final_Return (%)'] = pd.to_numeric(
                closed_group['Final_Return (%)'], errors='coerce'
            )
            closed_group = closed_group.dropna(subset=['Final_Return (%)'])

            date_counts = raw_group.groupby('Trade_Date').size() if not raw_group.empty else pd.Series(dtype=float)
            target_count = int(RESEARCH_TOP_N) if strategy_id == 'RESEARCH5' else int(PORTFOLIO_SLOTS)
            underfilled = sum(int(date_counts.get(d, 0)) < target_count for d in completed_dates)
            no_pick = sum(int(date_counts.get(d, 0)) == 0 for d in completed_dates)

            strategy_rows.append({
                '方案': strategy_names[strategy_id],
                '已完成扫描周': total_completed_weeks,
                '有入选周': int(raw_group['Trade_Date'].nunique()) if not raw_group.empty else 0,
                '空窗周': no_pick,
                '选不满目标周': underfilled,
                '入选笔数': len(raw_group),
                '完成交易': len(closed_group),
                '胜率': (closed_group['Final_Return (%)'] > 0).mean() * 100 if not closed_group.empty else np.nan,
                '平均收益': closed_group['Final_Return (%)'].mean() if not closed_group.empty else np.nan,
                '中位数收益': closed_group['Final_Return (%)'].median() if not closed_group.empty else np.nan,
                '破-10%率': closed_group['Exit_Reason'].astype(str).str.contains('破-10%').mean() * 100 if not closed_group.empty else np.nan,
            })
        strategy_stats = pd.DataFrame(strategy_rows)
        for col in ('胜率', '平均收益', '中位数收益', '破-10%率'):
            strategy_stats[col] = strategy_stats[col].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
        st.dataframe(strategy_stats, width='stretch', hide_index=True)

        if not main_signals.empty:
            main_closed = main_signals[main_signals['Exit_Reason'].astype(str) != '持仓中'].copy()
            main_closed['Final_Return (%)'] = pd.to_numeric(main_closed['Final_Return (%)'], errors='coerce')
            main_closed = main_closed.dropna(subset=['Final_Return (%)'])
            if not main_closed.empty:
                win_count = int((main_closed['Final_Return (%)'] > 0).sum())
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("原版Top3完成交易", f"{len(main_closed)} 笔")
                col_m2.metric("原版Top3胜率", f"{win_count / len(main_closed) * 100:.1f}%", f"{win_count}胜")
                col_m3.metric("原版Top3平均收益", f"{main_closed['Final_Return (%)'].mean():.2f}%")
                col_m4.metric("原版Top3中位数", f"{main_closed['Final_Return (%)'].median():.2f}%")

                st.subheader("🗓️ 原版Top3周度表现（仍为存活样本口径）")
                cols_row1 = st.columns(4)
                cols_row2 = st.columns(4)
                cols_row3 = st.columns(4)
                for w in range(1, 13):
                    col_name = f'Return_W{w} (%)'
                    if col_name not in main_signals.columns:
                        continue
                    valid_week = main_signals.dropna(subset=[col_name]).copy()
                    if w <= 4:
                        target_col = cols_row1[w - 1]
                    elif w <= 8:
                        target_col = cols_row2[w - 5]
                    else:
                        target_col = cols_row3[w - 9]
                    with target_col:
                        if not valid_week.empty:
                            week_values = pd.to_numeric(valid_week[col_name], errors='coerce').dropna()
                            if not week_values.empty:
                                st.metric(
                                    f"W{w} 均益/胜率 (存活{len(week_values)}只)",
                                    f"{week_values.mean():.2f}% / {(week_values > 0).mean() * 100:.1f}%",
                                )

        research_closed = research_signals[
            research_signals['Exit_Reason'].astype(str) != '持仓中'
        ].copy()
        research_closed['Final_Return (%)'] = pd.to_numeric(
            research_closed['Final_Return (%)'], errors='coerce'
        )
        research_closed = research_closed.dropna(subset=['Final_Return (%)'])
        if not research_closed.empty:
            st.subheader(f"🏆 原版前{int(RESEARCH_TOP_N)}名逐名表现")
            rank_stats = research_closed.groupby('Original_Rank', observed=False).agg(
                样本数=('Final_Return (%)', 'count'),
                平均分=('Total_Score', 'mean'),
                胜率=('Final_Return (%)', lambda x: (x > 0).mean() * 100),
                均益=('Final_Return (%)', 'mean'),
                中位数=('Final_Return (%)', 'median'),
                止损率=('Exit_Reason', lambda x: x.astype(str).str.contains('破-10%').mean() * 100),
            ).reset_index().head(int(RESEARCH_TOP_N))
            for col in ('胜率', '均益', '中位数', '止损率'):
                rank_stats[col] = rank_stats[col].map('{:.2f}%'.format)
            st.dataframe(rank_stats, width='stretch', hide_index=True)

        audit_closed = audit_valid[audit_valid['Exit_Reason'].astype(str) != '持仓中'].copy()
        audit_closed['Final_Return (%)'] = pd.to_numeric(audit_closed['Final_Return (%)'], errors='coerce')
        audit_closed = audit_closed.dropna(subset=['Final_Return (%)'])
        if not audit_closed.empty:
            st.subheader("🔎 全候选旁证：市值层与入选日V40状态（均不参与排序）")
            diag_col1, diag_col2 = st.columns(2)
            cap_stats = audit_closed.groupby('Cap_Tier', observed=False).agg(
                样本数=('Final_Return (%)', 'count'),
                胜率=('Final_Return (%)', lambda x: (x > 0).mean() * 100),
                均益=('Final_Return (%)', 'mean'),
                中位数=('Final_Return (%)', 'median'),
            ).reset_index()
            state_stats = audit_closed.groupby('V40_State', observed=False).agg(
                样本数=('Final_Return (%)', 'count'),
                胜率=('Final_Return (%)', lambda x: (x > 0).mean() * 100),
                均益=('Final_Return (%)', 'mean'),
                中位数=('Final_Return (%)', 'median'),
            ).reset_index()
            for frame in (cap_stats, state_stats):
                for col in ('胜率', '均益', '中位数'):
                    frame[col] = frame[col].map('{:.2f}%'.format)
            with diag_col1:
                st.markdown("**流通市值层**")
                st.dataframe(cap_stats, width='stretch', hide_index=True)
            with diag_col2:
                st.markdown("**入选日V40确认状态**")
                st.dataframe(state_stats, width='stretch', hide_index=True)

        if not main_signals.empty and not main_closed.empty:
            st.subheader("🧭 原版Top3买入后第3日 / 第5日确认分")
            hold_cols = st.columns(2)
            for idx, day in enumerate((3, 5)):
                score_col = f'Confirm_D{day}_Score'
                if score_col not in main_closed.columns:
                    continue
                reached = main_closed[main_closed[score_col].notna()].copy()
                if reached.empty:
                    continue
                hold_stats = reached.groupby(score_col, observed=False).agg(
                    样本数=('Final_Return (%)', 'count'),
                    胜率=('Final_Return (%)', lambda x: (x > 0).mean() * 100),
                    均益=('Final_Return (%)', 'mean'),
                    中位数=('Final_Return (%)', 'median'),
                ).reset_index().rename(columns={score_col: '确认分'})
                for col in ('胜率', '均益', '中位数'):
                    hold_stats[col] = hold_stats[col].map('{:.2f}%'.format)
                with hold_cols[idx]:
                    st.markdown(f"**第{day}个交易日（到达样本{len(reached)}只）**")
                    st.dataframe(hold_stats, width='stretch', hide_index=True)

            st.subheader("🧪 确认分0的真实次日开盘退出反事实")
            counterfactual_rows = []
            counterfactual_defs = [
                ('保持原规则', 'Final_Return (%)', None, None),
                ('D3为0→D4开盘退出', 'Final_If_D3_Zero_Exit_D4_Open (%)', 'Confirm_D3_Score', 'Max_After_D4_Open (%)'),
                ('D5为0→D6开盘退出', 'Final_If_D5_Zero_Exit_D6_Open (%)', 'Confirm_D5_Score', 'Max_After_D6_Open (%)'),
            ]
            for label, return_col, score_col, max_col in counterfactual_defs:
                values = pd.to_numeric(main_closed.get(return_col), errors='coerce')
                valid_values = values.dropna()
                affected = int((main_closed[score_col] == 0).sum()) if score_col else 0
                missed_20 = 0
                avg_missed_max = np.nan
                if score_col and max_col in main_closed.columns:
                    affected_rows = main_closed[main_closed[score_col] == 0].copy()
                    max_values = pd.to_numeric(affected_rows[max_col], errors='coerce')
                    missed_20 = int((max_values >= 20.0).sum())
                    avg_missed_max = max_values.mean()
                counterfactual_rows.append({
                    '方案': label,
                    '有效样本': len(valid_values),
                    '触发退出': affected,
                    '平均收益': valid_values.mean() if not valid_values.empty else np.nan,
                    '中位数': valid_values.median() if not valid_values.empty else np.nan,
                    '胜率': (valid_values > 0).mean() * 100 if not valid_values.empty else np.nan,
                    '触发后仍曾达到20%': missed_20,
                    '触发样本后续最高收益均值': avg_missed_max,
                })
            counterfactual_stats = pd.DataFrame(counterfactual_rows)
            for col in ('平均收益', '中位数', '胜率', '触发样本后续最高收益均值'):
                counterfactual_stats[col] = counterfactual_stats[col].map(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "-"
                )
            st.dataframe(counterfactual_stats, width='stretch', hide_index=True)
            st.caption("反事实采用D3收盘后D4开盘、D5收盘后D6开盘，尚未计入额外交易成本；主交易结果未被改变。")

        if not main_signals.empty:
            st.subheader("📋 原版Top3主结果交割流水单")
            disp_cols = [
                'Trade_Date', 'name', 'ts_code', 'Original_Rank', 'Total_Score', 'circ_mv', 'Cap_Tier',
                'V40_Confirm_Score', 'V40_State', 'Confirm_D3_State', 'Confirm_D5_State',
                'Return_D3_Close (%)', 'Return_D4_Open (%)',
                'Return_D5_Close (%)', 'Return_D6_Open (%)',
                'SKDJ_K', 'D_Min(10W)', 'Weeks_Under', 'Signal_Close', 'Buy_Price',
                'Exit_Date', 'Hold_Days', 'Exit_Reason', 'Final_Return (%)',
            ]
            final_disp = [c for c in disp_cols if c in main_signals.columns]

            def color_exit_reason(val):
                if isinstance(val, str):
                    if '截断' in val: return 'color: white; background-color: #8B4513'
                    elif '认栽' in val: return 'color: white; background-color: darkred'
                    elif '保本' in val: return 'color: white; background-color: darkgoldenrod'
                    elif '移动止盈' in val: return 'color: white; background-color: darkgreen'
                    elif '期满' in val: return 'color: blue'
                return ''

            display_main = main_signals[final_disp].sort_values(
                ['Trade_Date', 'Original_Rank'], ascending=[False, True], kind='mergesort'
            )
            try:
                styled_port = display_main.style
                if 'Exit_Reason' in display_main.columns:
                    styled_port = styled_port.map(color_exit_reason, subset=['Exit_Reason'])
                st.dataframe(styled_port, width='stretch')
            except Exception:
                st.dataframe(display_main, width='stretch')

        csv_data = repaired_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 导出Top3/Top5与D3/D5确认审计流水单 (CSV)",
            data=csv_data,
            file_name="skdj_v14_7_holding_confirmation_audit_export.csv",
            mime="text/csv",
            on_click="ignore",
            key="download_v14_7_holding_confirm_audit",
        )
    except pd.errors.EmptyDataError:
        st.info("🕒 当前暂无满足条件的回测记录。")
    except Exception as report_error:
        st.warning(f"回测数据已保留，但报告暂时无法显示：{report_error}")
