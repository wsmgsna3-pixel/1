# -*- coding: utf-8 -*-
"""
选股王 · V40.6 可靠缓存三仓回测版
------------------------------------------------
核心改进 (基于数据复盘后的终极定型):
1. [硬门槛 1：盘子基座] 侧边栏默认流通市值下限为 200 亿，隔绝微盘股的画线诱多陷阱。
2. [硬门槛 2：温和爆破] 突破量比上限严格锁定在 3.0倍 (1.3 <= vol <= 3.0)，绞杀“天量见天价”的分歧坑。
3. [硬门槛 3：开盘定生死] 在 T+1 买入引擎中加入集合竞价拦截器。若高开>5%或低开<-3%，直接放弃买入，剔除该标的！
4. [废除主观加分] 尊重客观数据，剔除原有的“洗盘2-3次加分”逻辑，所有分数纯靠量价真实动能。
5. [回测口径统一] 股票池改为V40.4历史申万科技池，股价>=20元、流通市值200~1000亿元。
6. [真实组合] 30万元初始资金、最多3只持仓、单只目标约10万元，并计入滑点、佣金和卖出税费。
7. [可靠续传] 行情/复权/市值按交易日立即落盘；有标的与无标的日均记录完成状态。
------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
import os
import pickle
import json
import hashlib
import tempfile
import shutil

warnings.filterwarnings("ignore")

CACHE_FILE_NAME = "market_data_cache_v40_6.pkl" 
CACHE_DIR_NAME = "market_data_cache_v40_6_days"
CACHE_VERSION = 3

INITIAL_CAPITAL = 300_000.0
MAX_PORTFOLIO_POSITIONS = 3
POSITION_BUDGET = INITIAL_CAPITAL / MAX_PORTFOLIO_POSITIONS
LOT_SIZE = 100

# 与V40.4一致的历史申万科技行业口径。
CORE_TECH_L1 = {'电子', '计算机', '通信', '国防军工'}
EXTENDED_TECH_L1 = {
    '机械设备', '电力设备', '医药生物', '汽车',
    '基础化工', '有色金属', '建筑材料',
}
TECH_INDUSTRY_KEYWORDS = {
    '半导体', '电子元件', '元件', '光学光电子', '消费电子', '电子化学品',
    '计算机设备', '软件开发', 'IT服务', '通信设备', '军工电子', '航空装备',
    '航天装备', '自动化设备', '机器人', '激光设备', '工控设备', '仪器仪表',
    '电池', '光伏设备', '风电设备', '电网设备', '电机', '医疗器械',
    '生物制品', '汽车电子', '金属新材料', '非金属材料', '膜材料', '碳纤维',
}

# ---------------------------
# 全局变量与探针
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_DAILY_BASIC = pd.DataFrame()
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_BASIC = pd.DataFrame()
GLOBAL_TECH_PERIODS = {}
SINA_STATUS = {'success': 0, 'fail': 0} 

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V40.6 可靠缓存版", layout="wide")
st.title("选股王 V40.6：箱体首发 + 可靠续传三仓")

# ---------------------------
# 新浪实时行情引擎
# ---------------------------
def get_sina_realtime_kline(ts_code):
    global SINA_STATUS
    code_split = ts_code.split('.')
    if len(code_split) != 2: return None
    sina_code = code_split[1].lower() + code_split[0]
    
    url = f"http://hq.sinajs.cn/list={sina_code}"
    headers = {'Referer': 'https://finance.sina.com.cn'}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = 'gbk'
        data_str = response.text.split('="')[1].split('";')[0]
        if not data_str: 
            SINA_STATUS['fail'] += 1
            return None
        data_list = data_str.split(',')
        
        SINA_STATUS['success'] += 1
        return {
            'trade_date_str': datetime.now().strftime('%Y%m%d'),
            'open': float(data_list[1]),
            'pre_close': float(data_list[2]),
            'close': float(data_list[3]),
            'high': float(data_list[4]),
            'low': float(data_list[5]),
            'vol': (float(data_list[8]) / 100) * (240 / 225) 
        }
    except Exception:
        SINA_STATUS['fail'] += 1
        return None

# ---------------------------
# 基础 API 函数
# ---------------------------
@st.cache_data(ttl=3600*12) 
def safe_get(func_name, **kwargs):
    global pro
    if pro is None: return pd.DataFrame(columns=['ts_code']) 
    func = getattr(pro, func_name) 
    try:
        for _ in range(3):
            try:
                if func_name == 'index_daily': 
                    df = pro.index_daily(**kwargs)
                else: 
                    df = func(**kwargs)
                if df is not None and not df.empty: return df
                time.sleep(0.5)
            except: time.sleep(1); continue
        return pd.DataFrame(columns=['ts_code']) 
    except Exception as e: return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    lookback_days = max(num_days * 3, 365) 
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)
    if cal.empty: return []
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()

def direct_get_with_retry(func_name, attempts=4, **kwargs):
    """下载阶段不缓存失败结果，网络恢复后可立即续传。"""
    global pro
    if pro is None:
        return pd.DataFrame(columns=['ts_code'])
    func = getattr(pro, func_name)
    for attempt in range(attempts):
        try:
            df = func(**kwargs)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        time.sleep(min(1.0 + attempt, 3.0))
    return pd.DataFrame(columns=['ts_code'])


def fetch_daily_components(date, existing=None, need_daily_basic=True):
    """只下载该交易日仍缺失的组件。"""
    data = dict(existing or {})
    requests_map = {
        'adj': ('adj_factor', {'trade_date': date}),
        'daily': ('daily', {'trade_date': date}),
    }
    if need_daily_basic:
        requests_map['daily_basic'] = ('daily_basic', {'trade_date': date})
    for key, (func_name, kwargs) in requests_map.items():
        current = data.get(key)
        if isinstance(current, pd.DataFrame) and not current.empty:
            continue
        data[key] = direct_get_with_retry(func_name, **kwargs)
    return data


def atomic_pickle_dump(value, target_path):
    target_path = os.path.abspath(target_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix='.tmp_v40_6_', dir=os.path.dirname(target_path))
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target_path)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def atomic_csv_save(df, target_path):
    target_path = os.path.abspath(target_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix='.tmp_v40_6_', suffix='.csv', dir=os.path.dirname(target_path))
    os.close(fd)
    try:
        df.to_csv(temp_path, index=False, encoding='utf-8-sig')
        os.replace(temp_path, target_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def cache_day_path(date):
    return os.path.join(CACHE_DIR_NAME, f'{date}.pkl')


def load_day_cache(date):
    path = cache_day_path(date)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def day_components_complete(data, need_daily_basic=True):
    required = ['adj', 'daily'] + (['daily_basic'] if need_daily_basic else [])
    return all(isinstance(data.get(key), pd.DataFrame) and not data[key].empty for key in required)

def normalize_date(value, default=''):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip().replace('-', '').replace('/', '')
    if text.endswith('.0'):
        text = text[:-2]
    return text if len(text) == 8 and text.isdigit() else default


@st.cache_data(ttl=3600*24)
def load_stock_basic_history():
    frames = []
    fields = 'ts_code,symbol,name,market,exchange,list_status,list_date,delist_date'
    for status in ['L', 'P', 'D']:
        df = safe_get('stock_basic', list_status=status, fields=fields)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True).drop_duplicates('ts_code', keep='first')
    result = result[
        result['market'].isin(['主板', '创业板', '科创板'])
        & result['exchange'].ne('BSE')
        & ~result['ts_code'].str.endswith('.BJ', na=False)
    ].copy()
    return result


def industry_row_is_tech(row):
    l1 = str(row.get('l1_name', ''))
    l2 = str(row.get('l2_name', ''))
    l3 = str(row.get('l3_name', ''))
    if l1 in CORE_TECH_L1:
        return True
    if l1 not in EXTENDED_TECH_L1:
        return False
    combined = f'{l2}|{l3}'
    return any(keyword in combined for keyword in TECH_INDUSTRY_KEYWORDS)


@st.cache_data(ttl=3600*24*7)
def load_sw_tech_memberships_history():
    l1_df = safe_get('index_classify', level='L1', src='SW2021')
    if l1_df.empty:
        return pd.DataFrame()
    target_names = CORE_TECH_L1 | EXTENDED_TECH_L1
    target_l1 = l1_df[l1_df['industry_name'].isin(target_names)]
    if target_l1.empty:
        return pd.DataFrame()

    frames = []
    targets = target_l1[['index_code', 'industry_name']].to_dict('records')
    load_bar = st.progress(0, text='正在构建V40.4同口径申万历史科技池...')
    total_calls = max(len(targets) * 2, 1)
    call_no = 0
    for item in targets:
        for flag in ['Y', 'N']:
            df = safe_get('index_member_all', l1_code=item['index_code'], is_new=flag)
            call_no += 1
            load_bar.progress(call_no / total_calls)
            if not df.empty:
                frames.append(df)
            time.sleep(0.05)
    load_bar.empty()
    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    for col in ['l1_name', 'l2_name', 'l3_name', 'in_date', 'out_date', 'is_new']:
        if col not in result.columns:
            result[col] = ''
    result = result[result.apply(industry_row_is_tech, axis=1)].copy()
    result['in_date'] = result['in_date'].apply(lambda x: normalize_date(x, '19000101'))
    result['out_date'] = result['out_date'].apply(lambda x: normalize_date(x, '99991231'))
    result = result.drop_duplicates(
        ['ts_code', 'l1_name', 'l2_name', 'l3_name', 'in_date', 'out_date']
    )
    return result


def build_tech_period_index(memberships):
    index = {}
    for row in memberships.itertuples(index=False):
        index.setdefault(str(row.ts_code), []).append({
            'in_date': str(row.in_date),
            'out_date': str(row.out_date),
            'l1': str(row.l1_name),
            'l2': str(row.l2_name),
            'l3': str(row.l3_name),
        })
    return index


def get_tech_membership(ts_code, trade_date):
    for period in GLOBAL_TECH_PERIODS.get(str(ts_code), []):
        if period['in_date'] <= trade_date < period['out_date']:
            return period
    return None


def stock_active_on_date(row, trade_date):
    list_date = normalize_date(getattr(row, 'list_date', ''), '19000101')
    delist_date = normalize_date(getattr(row, 'delist_date', ''), '99991231')
    return list_date <= trade_date < delist_date

# ---------------------------
# 数据获取与复权引擎
# ---------------------------
def split_cached_frame_by_date(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    work = frame.reset_index() if 'trade_date' in list(frame.index.names) else frame.copy()
    if 'trade_date' not in work.columns:
        return {}
    work['trade_date'] = work['trade_date'].astype(str)
    return {str(date): group.copy() for date, group in work.groupby('trade_date')}


def set_global_market_frames(adj_frames, daily_frames, basic_frames):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_DAILY_BASIC, GLOBAL_QFQ_BASE_FACTORS

    adj = pd.concat(adj_frames, ignore_index=True)
    adj['trade_date'] = adj['trade_date'].astype(str)
    adj['adj_factor'] = pd.to_numeric(adj['adj_factor'], errors='coerce').fillna(0)
    GLOBAL_ADJ_FACTOR = (
        adj.drop_duplicates(['ts_code', 'trade_date'])
        .set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    )

    daily = pd.concat(daily_frames, ignore_index=True)
    daily['trade_date'] = daily['trade_date'].astype(str)
    GLOBAL_DAILY_RAW = (
        daily.drop_duplicates(['ts_code', 'trade_date'])
        .set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    )

    basic = pd.concat(basic_frames, ignore_index=True)
    basic['trade_date'] = basic['trade_date'].astype(str)
    GLOBAL_DAILY_BASIC = (
        basic.drop_duplicates(['ts_code', 'trade_date'])
        .set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
    )

    latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
    try:
        latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
        GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
    except Exception:
        GLOBAL_QFQ_BASE_FACTORS = {}


def get_all_historical_data(trade_days_list, use_cache=True):
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_DAILY_BASIC, GLOBAL_QFQ_BASE_FACTORS
    global GLOBAL_STOCK_BASIC, GLOBAL_TECH_PERIODS
    if not trade_days_list: return False
    
    with st.spinner('正在同步V40.4同口径历史股票池与行业成分...'):
        GLOBAL_STOCK_BASIC = load_stock_basic_history()
        memberships = load_sw_tech_memberships_history()
        GLOBAL_TECH_PERIODS = build_tech_period_index(memberships)
    if GLOBAL_STOCK_BASIC.empty or not GLOBAL_TECH_PERIODS:
        st.error('历史股票池或申万历史行业成分加载失败，无法保证与V40.4样本一致。')
        return False

    latest_trade_date = max(trade_days_list) 
    earliest_trade_date = min(trade_days_list)
    
    start_date = (datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    requested_end = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=150)
    end_date = min(requested_end, datetime.now()).strftime("%Y%m%d")
    
    all_trade_dates_df = safe_get('trade_cal', start_date=start_date, end_date=end_date, is_open='1')
    if all_trade_dates_df.empty: return False
    all_dates = sorted(all_trade_dates_df['cal_date'].astype(str).unique().tolist())
    required_dates = set(all_dates)
    analysis_dates = set(map(str, trade_days_list))
    os.makedirs(CACHE_DIR_NAME, exist_ok=True)

    legacy_seed = {'adj': {}, 'daily': {}, 'daily_basic': {}}
    if use_cache and os.path.exists(CACHE_FILE_NAME):
        try:
            with open(CACHE_FILE_NAME, 'rb') as f:
                cached_data = pickle.load(f)
            cached_market_coverage = set(map(str, cached_data.get('covered_market_dates', [])))
            cached_basic_coverage = set(map(str, cached_data.get('covered_basic_dates', [])))
            if (
                cached_data.get('cache_version') == CACHE_VERSION
                and required_dates.issubset(cached_market_coverage)
                and analysis_dates.issubset(cached_basic_coverage)
                and all(isinstance(cached_data.get(k), pd.DataFrame) and not cached_data[k].empty
                        for k in ['adj', 'daily', 'daily_basic'])
            ):
                GLOBAL_ADJ_FACTOR = cached_data['adj']
                GLOBAL_DAILY_RAW = cached_data['daily']
                GLOBAL_DAILY_BASIC = cached_data['daily_basic']
                latest_global_date = GLOBAL_ADJ_FACTOR.index.get_level_values('trade_date').max()
                try:
                    latest_adj_df = GLOBAL_ADJ_FACTOR.loc[(slice(None), latest_global_date), 'adj_factor']
                    GLOBAL_QFQ_BASE_FACTORS = latest_adj_df.droplevel(1).to_dict()
                except Exception:
                    GLOBAL_QFQ_BASE_FACTORS = {}
                st.success(f"⚡ 行情、复权和市值缓存已完整覆盖 {len(all_dates)} 个交易日。")
                return True
            for key in legacy_seed:
                legacy_seed[key] = split_cached_frame_by_date(cached_data.get(key, pd.DataFrame()))
        except Exception:
            legacy_seed = {'adj': {}, 'daily': {}, 'daily_basic': {}}

    adj_frames, daily_frames, basic_frames = [], [], []
    failed_dates = []
    consecutive_failures = 0
    downloaded_now = 0
    reused_days = 0
    my_bar = st.progress(0, text="检查逐日持久化缓存...")

    for i, date in enumerate(all_dates):
        need_daily_basic = date in analysis_dates
        data = load_day_cache(date) if use_cache else {}
        for key in ['adj', 'daily', 'daily_basic']:
            current = data.get(key)
            if not isinstance(current, pd.DataFrame) or current.empty:
                seeded = legacy_seed[key].get(date)
                if isinstance(seeded, pd.DataFrame) and not seeded.empty:
                    data[key] = seeded

        was_complete = day_components_complete(data, need_daily_basic=need_daily_basic)
        if not was_complete:
            data = fetch_daily_components(date, data, need_daily_basic=need_daily_basic)
            if any(isinstance(data.get(k), pd.DataFrame) and not data[k].empty
                   for k in ['adj', 'daily', 'daily_basic']):
                atomic_pickle_dump(data, cache_day_path(date))

        if day_components_complete(data, need_daily_basic=need_daily_basic):
            adj_frames.append(data['adj'])
            daily_frames.append(data['daily'])
            if need_daily_basic:
                basic_frames.append(data['daily_basic'])
            consecutive_failures = 0
            if was_complete:
                reused_days += 1
            else:
                downloaded_now += 1
        else:
            failed_dates.append(date)
            consecutive_failures += 1

        my_bar.progress(
            (i + 1) / len(all_dates),
            text=f"缓存核验 {i+1}/{len(all_dates)}｜本次新增 {downloaded_now}｜已复用 {reused_days}",
        )
        if consecutive_failures >= 3:
            break
    my_bar.empty()

    if (
        failed_dates
        or len(adj_frames) != len(all_dates)
        or len(basic_frames) != len(analysis_dates)
    ):
        st.warning(
            f"网络中断或数据缺失：已成功持久化 {len(adj_frames)}/{len(all_dates)} 个交易日。"
            "请在网络恢复后重新启动，程序只会补下载缺失项。"
        )
        return False

    with st.spinner("正在构建内存索引和快速总缓存..."):
        set_global_market_frames(adj_frames, daily_frames, basic_frames)
        atomic_pickle_dump({
            'cache_version': CACHE_VERSION,
            'covered_market_dates': all_dates,
            'covered_basic_dates': sorted(analysis_dates),
            'start_date': all_dates[0],
            'end_date': all_dates[-1],
            'adj': GLOBAL_ADJ_FACTOR,
            'daily': GLOBAL_DAILY_RAW,
            'daily_basic': GLOBAL_DAILY_BASIC,
        }, CACHE_FILE_NAME)
    st.success(f"✅ {len(all_dates)} 个交易日已全部持久化，后续逐日分析不再依赖行情网络请求。")
    return True

def get_qfq_data_v4_optimized_final(ts_code, start_date, end_date, use_sina=False):
    global GLOBAL_DAILY_RAW, GLOBAL_ADJ_FACTOR, GLOBAL_QFQ_BASE_FACTORS
    if GLOBAL_DAILY_RAW.empty: return pd.DataFrame()
    
    latest_adj_factor = GLOBAL_QFQ_BASE_FACTORS.get(ts_code, np.nan)
    if pd.isna(latest_adj_factor): return pd.DataFrame() 

    try:
        daily_df = GLOBAL_DAILY_RAW.loc[ts_code]
        daily_df = daily_df.loc[(daily_df.index >= start_date) & (daily_df.index <= end_date)].copy()
        adj_series = GLOBAL_ADJ_FACTOR.loc[ts_code]['adj_factor']
        adj_series = adj_series.loc[(adj_series.index >= start_date) & (adj_series.index <= end_date)]
    except KeyError: return pd.DataFrame()
    
    if daily_df.empty or adj_series.empty: return pd.DataFrame()
    
    df = daily_df.merge(adj_series.rename('adj_factor'), left_index=True, right_index=True, how='left')
    df = df.dropna(subset=['adj_factor'])
    
    for col in ['open', 'high', 'low', 'close', 'pre_close']:
        if col in df.columns:
            df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor
    
    df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})
    df = df.sort_values('trade_date_str').set_index('trade_date_str')
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col + '_qfq']
        
    final_df = df[['open', 'high', 'low', 'close', 'pre_close', 'vol']].copy() 

    if use_sina:
        today_str = datetime.now().strftime('%Y%m%d')
        if end_date == today_str:
            sina_data = get_sina_realtime_kline(ts_code)
            if sina_data and sina_data['close'] > 0:
                sina_row = pd.DataFrame([sina_data]).set_index('trade_date_str')
                if today_str in final_df.index:
                    final_df.loc[today_str] = sina_row.iloc[0]
                else:
                    final_df = pd.concat([final_df, sina_row])
                    
    return final_df

# ---------------------------
# 周线 MACD 波浪洗盘次数统计函数
# ---------------------------
def count_macd_wave_pullbacks(df_calc):
    if len(df_calc) < 60: return -1 
    
    df = df_calc.copy()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    df['dt'] = pd.to_datetime(df['trade_date_str'])
    iso_cal = df['dt'].dt.isocalendar()
    df['year_week'] = iso_cal.year.astype(str) + "_" + iso_cal.week.astype(str).str.zfill(2)
    
    weekly_df = df.groupby('year_week', as_index=False).agg({
        'trade_date_str': 'last',
        'low': 'min',
        'high': 'max',
        'macd': 'last'
    }).sort_values('trade_date_str').reset_index(drop=True)
    
    if len(weekly_df) < 10: return -1
    
    min_idx = weekly_df['low'].idxmin()
    sub_df = weekly_df.loc[min_idx:].reset_index(drop=True)
    if len(sub_df) < 5: return -1
    
    running_max = sub_df['high'].iloc[0]
    in_pullback = False
    pullback_count = 0
    
    for i in range(1, len(sub_df)):
        curr_high = sub_df.loc[i, 'high']
        curr_low = sub_df.loc[i, 'low']
        curr_macd = sub_df.loc[i, 'macd']
        
        if curr_high > running_max:
            running_max = curr_high
            if in_pullback:
                in_pullback = False
        else:
            drawdown = (running_max - curr_low) / running_max
            if curr_macd < 0 and drawdown >= 0.05:
                if not in_pullback:
                    in_pullback = True
                    pullback_count += 1
                    
    return pullback_count

# ---------------------------
# 核心指标计算 (加入四大神盾)
# ---------------------------
@st.cache_data(ttl=3600*12) 
def compute_trend_indicators(ts_code, end_date, use_sina=False, _run_id=None):
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    df = get_qfq_data_v4_optimized_final(ts_code, start_date, end_date, use_sina=use_sina)
    res = {}
    if df.empty or len(df) < 120: return res 
    
    # 1. 日线基础指标
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['ma120'] = df['close'].rolling(120).mean()
    df['ma5_vol'] = df['vol'].shift(1).rolling(5).mean()  
    
    # 10 日箱体
    df['box_high_10'] = df['high'].rolling(window=10).max().shift(1)
    
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    
    df_calc = df.dropna().copy().reset_index()
    if len(df_calc) < 20: return res

    # 2. 周线波浪过滤
    wave_count = count_macd_wave_pullbacks(df_calc)
    if wave_count < 2 or wave_count > 5:
        return res  

    # 3. 周线风控
    df_calc['dt'] = pd.to_datetime(df_calc['trade_date_str'])
    iso_cal = df_calc['dt'].dt.isocalendar()
    df_calc['year_week'] = iso_cal.year.astype(str) + "_" + iso_cal.week.astype(str).str.zfill(2)
    
    weekly_df = df_calc.groupby('year_week', as_index=False).agg({
        'trade_date_str': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }).sort_values('trade_date_str').reset_index(drop=True)
    
    if len(weekly_df) < 10: return res
    weekly_df['w_ma20'] = weekly_df['close'].rolling(20).mean()
    
    w_curr = weekly_df.iloc[-1]
    w_prev = weekly_df.iloc[-2] if len(weekly_df) >= 2 else w_curr
    
    w_bias_safe = True
    if not pd.isna(w_curr['w_ma20']) and w_curr['w_ma20'] > 0:
        w_bias = (w_curr['close'] - w_curr['w_ma20']) / w_curr['w_ma20']
        if w_bias > 0.45: w_bias_safe = False
            
    w_shadow_safe = True
    w_prev_range = w_prev['high'] - w_prev['low']
    w_prev_upper_shadow = w_prev['high'] - max(w_prev['open'], w_prev['close'])
    if w_prev_range > 0 and (w_prev_upper_shadow / w_prev_range) >= 0.60:
        w_shadow_safe = False

    is_weekly_safe = w_bias_safe and w_shadow_safe

    # 4. 日线突破点火信号 
    row = df_calc.iloc[-1]
    prev_row = df_calc.iloc[-2]
    
    is_daily_trend_up = row['ma60'] > row['ma120']
    
    is_box_breakout = (row['close'] > row['box_high_10']) and (prev_row['close'] <= prev_row['box_high_10'])
    is_daily_breakout = row['close'] > row['ma20'] * 1.02
    is_daily_ma20_healthy = row['ma20'] >= prev_row['ma20']
    
    # 【改动2：突破量比 ≤ 3.0倍】
    vol_ratio = row['vol'] / row['ma5_vol'] if row['ma5_vol'] > 0 else 0
    is_daily_vol_strong = (1.3 <= vol_ratio <= 3.0)
    
    candle_range = row['high'] - row['low']
    candle_body = row['close'] - row['open']
    is_solid_yang = (row['close'] > row['open']) and (candle_body >= candle_range * 0.6 if candle_range > 0 else True)
    is_macd_healthy = (row['dif'] > 0) and (row['macd'] > prev_row['macd'])
    
    res['is_v38_buy_signal'] = (is_weekly_safe and 
                                is_daily_trend_up and 
                                is_box_breakout and 
                                is_daily_breakout and 
                                is_daily_ma20_healthy and 
                                is_daily_vol_strong and 
                                is_solid_yang and 
                                is_macd_healthy)
    
    if res['is_v38_buy_signal']:
        res['vol_ratio'] = vol_ratio
        res['pre_close'] = prev_row['close']            
        res['wave_count'] = wave_count  
        
    res['last_close'] = row['close']
    res['bottom_line'] = row['low'] 
    res['ma20'] = row['ma20']
    
    return res

# ---------------------------
# 三层简化止盈止损系统 (加入集合竞价拦截器)
# ---------------------------
def get_medium_term_future(
    ts_code, selection_date, signal_close, bottom_line, hold_weeks=8,
    buy_slippage_pct=0.20, sell_slippage_pct=0.20,
    commission_pct=0.03, sell_tax_pct=0.05, use_sina=False,
):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_fetch = (d0 - timedelta(days=60)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=150)).strftime("%Y%m%d") 
    
    hist_full = get_qfq_data_v4_optimized_final(ts_code, start_date=start_fetch, end_date=end_future, use_sina=use_sina)
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    results['Exit_Reason'] = "持仓中"
    results['Entry_Date'] = ''
    results['Exit_Date'] = ''
    results['Buy_Price'] = np.nan
    results['Raw_Exit_Price'] = np.nan
    results['Net_Exit_Price'] = np.nan
    results['Realized_Return (%)'] = np.nan
    results['Holding_Days'] = 0
    results['Gap_pct (%)'] = np.nan
    results['Tradable'] = False
    
    if hist_full.empty or len(hist_full) < 30: return results
    
    hist_full['open'] = pd.to_numeric(hist_full['open'], errors='coerce')
    hist_full['high'] = pd.to_numeric(hist_full['high'], errors='coerce')
    hist_full['low'] = pd.to_numeric(hist_full['low'], errors='coerce')
    hist_full['close'] = pd.to_numeric(hist_full['close'], errors='coerce')
    
    hist_future = hist_full[hist_full.index > selection_date]
    if hist_future.empty: return results

    next_row = hist_future.iloc[0]

    is_main_board = not (ts_code.startswith('300') or ts_code.startswith('301') 
                          or ts_code.startswith('688') or ts_code.startswith('689'))
    is_one_word_limit = (is_main_board and pd.notna(next_row['open']) and pd.notna(next_row['high']) 
                          and pd.notna(next_row['low']) and next_row['open'] == next_row['high'] == next_row['low'])
    if is_one_word_limit:
        results['Exit_Reason'] = "一字板无法买入(剔除)"
        results['Buy_Price'] = round(next_row['open'], 2)  
        return results

    raw_open = next_row['open']
    if pd.isna(raw_open) or raw_open <= 0:
        return results

    # 【改动3：T+1 集合竞价拦截器】防核按钮与高开诱多
    if signal_close and signal_close > 0:
        gap_pct = (raw_open - signal_close) / signal_close * 100
        results['Gap_pct (%)'] = round(gap_pct, 2)
        if gap_pct < -3.0 or gap_pct > 5.0:
            results['Exit_Reason'] = f"开盘幅度不符(剔除: {round(gap_pct, 2)}%)"
            results['Buy_Price'] = round(raw_open, 3)
            # 直接返回，不再执行后续持仓运算
            return results

    buy_price = raw_open * (1.0 + buy_slippage_pct / 100.0)
    buy_price *= 1.0 + commission_pct / 100.0
    results['Entry_Date'] = normalize_date(next_row.name)
    results['Buy_Price'] = round(buy_price, 3)
    results['Tradable'] = True

    def net_exit_price(raw_sell_price):
        net_sell = raw_sell_price * (1.0 - sell_slippage_pct / 100.0)
        net_sell *= 1.0 - (commission_pct + sell_tax_pct) / 100.0
        return net_sell

    def net_return(raw_sell_price):
        return (net_exit_price(raw_sell_price) / buy_price - 1.0) * 100.0

    def finalize_exit(raw_sell_price, current_week, day_count, reason, exit_date):
        final_return = net_return(raw_sell_price)
        results['Exit_Reason'] = reason
        results['Exit_Date'] = normalize_date(exit_date)
        results['Raw_Exit_Price'] = round(raw_sell_price, 3)
        results['Net_Exit_Price'] = round(net_exit_price(raw_sell_price), 3)
        results['Realized_Return (%)'] = round(final_return, 4)
        results['Holding_Days'] = int(day_count)
        results[f'Return_W{current_week} (%)'] = final_return
        return final_return

    exit_triggered = False
    tier = 0  
    peak_close = buy_price
    pending_exit_reason = None  
    
    is_20cm = ts_code.startswith('300') or ts_code.startswith('301') or ts_code.startswith('688')
    hard_stop_limit = -0.12 if is_20cm else -0.08
    
    for i in range(len(hist_future)):
        if i >= hold_weeks * 5: break 
            
        row = hist_future.iloc[i]
        day_count = i + 1
        current_week = ((day_count - 1) // 5) + 1 
        
        curr_open = row['open']
        curr_close = row['close']
        curr_high = row['high']
        curr_low = row['low']
        
        if pending_exit_reason is not None:
            finalize_exit(curr_open, current_week, day_count, pending_exit_reason, row.name)
            exit_triggered = True
            break
        
        peak_close = max(peak_close, curr_high)
        peak_profit_pct = (peak_close - buy_price) / buy_price
        
        if (curr_low - buy_price) / buy_price <= hard_stop_limit:
            raw_stop_price = buy_price * (1.0 + hard_stop_limit)
            raw_exit_price = curr_open if curr_open < raw_stop_price else raw_stop_price
            finalize_exit(
                raw_exit_price, current_week, day_count,
                f"固定止损(破{int(hard_stop_limit*100)}%)", row.name,
            )
            exit_triggered = True
            break
        
        if tier == 0 and peak_profit_pct >= 0.10:
            tier = 1
                
        if tier == 1:
            if curr_close < buy_price * 0.995:
                pending_exit_reason = "保本止盈"
            elif peak_profit_pct >= 0.20:
                tier = 2
                
        if tier == 2:
            giveback = (peak_close - curr_close) / peak_close
            if giveback >= 0.15:
                pending_exit_reason = "移动止盈(回撤15%)"
            
        if day_count % 5 == 0:
            results[f'Return_W{current_week} (%)'] = net_return(curr_close)
            
    if not exit_triggered and len(hist_future) >= hold_weeks * 5:
        last_row = hist_future.iloc[hold_weeks * 5 - 1]
        finalize_exit(
            last_row['close'], hold_weeks, hold_weeks * 5,
            '周期结束平仓', last_row.name,
        )
    elif not exit_triggered:
        results['Holding_Days'] = min(len(hist_future), hold_weeks * 5)
        
    return results

# ---------------------------
# 核心回测循环
# ---------------------------
def cached_frame_for_date(frame, trade_date):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    try:
        result = frame.xs(str(trade_date), level='trade_date').reset_index()
        return result
    except (KeyError, ValueError):
        return pd.DataFrame()


def run_backtest_for_a_day(
    last_trade, TOP_BACKTEST, MIN_MV, MAX_MV, MIN_PRICE,
    buy_slippage_pct, sell_slippage_pct, commission_pct, sell_tax_pct,
    use_sina=False, run_timestamp=None,
):
    global GLOBAL_STOCK_BASIC, GLOBAL_TECH_PERIODS, GLOBAL_DAILY_RAW, GLOBAL_DAILY_BASIC

    query_date = last_trade
    daily_all = (
        safe_get('daily', trade_date=query_date)
        if use_sina else cached_frame_for_date(GLOBAL_DAILY_RAW, query_date)
    )
    
    if use_sina and daily_all.empty:
        for i in range(1, 10):
            temp_date = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=i)).strftime("%Y%m%d")
            daily_all = safe_get('daily', trade_date=temp_date)
            if not daily_all.empty:
                query_date = temp_date
                break
                
    if daily_all.empty: return pd.DataFrame(), "数据缺失"

    if GLOBAL_STOCK_BASIC.empty:
        return pd.DataFrame(), "历史股票基础信息缺失"
    df = daily_all.merge(GLOBAL_STOCK_BASIC, on='ts_code', how='inner')
    
    daily_basic = (
        safe_get('daily_basic', trade_date=query_date)
        if use_sina else cached_frame_for_date(GLOBAL_DAILY_BASIC, query_date)
    )
    if not daily_basic.empty:
        df = df.merge(daily_basic[['ts_code','circ_mv']], on='ts_code', how='left')
    else: 
        return pd.DataFrame(), "市值数据缺失"
    
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    
    df = df[df.apply(lambda r: stock_active_on_date(r, last_trade), axis=1)]
    df = df[~df['name'].str.contains('ST|退', na=False)]
    
    df = df[(df['close'] >= MIN_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    
    records = []
    for row in df.itertuples():
        membership = get_tech_membership(row.ts_code, last_trade)
        if membership is None:
            continue
            
        ind = compute_trend_indicators(row.ts_code, last_trade, use_sina=use_sina, _run_id=run_timestamp)
        if not ind or not ind.get('is_v38_buy_signal'): 
            continue
            
        if use_sina and ind['last_close'] < MIN_PRICE:
            continue
            
        pct_chg = (ind['last_close'] - ind['pre_close']) / ind['pre_close'] * 100
        score_breakout = pct_chg * 10 
        score_vol = ind['vol_ratio'] * 10
        total_score = score_breakout + score_vol
        
        # 【改动4：废除主观加分】去掉了 wave_cnt 的 30 分加成，让排序回归真实的量价爆发力度。
        wave_cnt = ind.get('wave_count', 3)
            
        future_returns = get_medium_term_future(
            row.ts_code, last_trade, ind['last_close'], ind['bottom_line'],
            hold_weeks=8,
            buy_slippage_pct=buy_slippage_pct,
            sell_slippage_pct=sell_slippage_pct,
            commission_pct=commission_pct,
            sell_tax_pct=sell_tax_pct,
            use_sina=use_sina,
        )
        
        record_dict = {
            'ts_code': row.ts_code, 'name': row.name, 'Signal_Close': ind['last_close'], 
            'market': row.market,
            'SW_L1': membership['l1'], 'SW_L2': membership['l2'], 'SW_L3': membership['l3'],
            'Wave_Count': wave_cnt,
            'circ_mv': row.circ_mv_billion,
            'Total_Score': round(total_score, 1),
            'Breakout_S': round(score_breakout, 1),
            'Volume_S': round(score_vol, 1)
        }
        record_dict.update(future_returns)
        records.append(record_dict)
            
    if not records: return pd.DataFrame(), "无标的"
    
    fdf = pd.DataFrame(records)
    final_df = fdf.sort_values('Total_Score', ascending=False).head(TOP_BACKTEST).copy()
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    return final_df, None


# ---------------------------
# 30万元、最多3只真实组合
# ---------------------------
def bool_series(series):
    if series is None:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])


def build_portfolio_backtest(
    signals, trade_days,
    initial_capital=INITIAL_CAPITAL,
    max_positions=MAX_PORTFOLIO_POSITIONS,
    position_budget=POSITION_BUDGET,
    lot_size=LOT_SIZE,
):
    days = sorted({normalize_date(day) for day in trade_days if normalize_date(day)})
    empty_columns = [
        'Trade_Date', 'Cash', 'Market_Value', 'Equity', 'Positions',
        'Exposure_pct', 'Is_Empty', 'Daily_Return_pct', 'Drawdown_pct',
    ]
    if signals is None or signals.empty or not days:
        return pd.DataFrame(columns=empty_columns), pd.DataFrame(), pd.DataFrame(), {
            'Initial_Capital': initial_capital, 'Final_Equity': initial_capital,
            'Total_Return_pct': 0.0, 'Max_Drawdown_pct': 0.0,
            'Empty_Ratio_pct': 100.0, 'Invested_Days': 0,
            'Average_Positions': 0.0, 'Average_Exposure_pct': 0.0,
            'Executed_Entries': 0, 'Closed_Trades': 0, 'Open_Positions': 0,
            'Closed_Win_Rate_pct': np.nan, 'Fixed_Stop_Rate_pct': np.nan,
            'Rejected_Full': 0, 'Rejected_Duplicate': 0,
        }

    work = signals.copy()
    for col in ['Trade_Date', 'Entry_Date', 'Exit_Date']:
        if col not in work.columns:
            work[col] = ''
        work[col] = work[col].map(normalize_date)
    for col in ['Rank', 'Total_Score', 'Buy_Price', 'Net_Exit_Price', 'Realized_Return (%)']:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors='coerce')
    if 'Tradable' in work.columns:
        work = work[bool_series(work['Tradable'])].copy()
    else:
        work = work[work['Buy_Price'].notna()].copy()
    work = work.drop_duplicates(['Trade_Date', 'ts_code'], keep='last')
    work = work.sort_values(['Entry_Date', 'Rank', 'Total_Score'], ascending=[True, True, False])

    first_day, last_day = days[0], days[-1]
    price_start = (datetime.strptime(first_day, '%Y%m%d') - timedelta(days=15)).strftime('%Y%m%d')
    price_series = {}
    for ts_code in sorted(work['ts_code'].dropna().astype(str).unique()):
        hist = get_qfq_data_v4_optimized_final(ts_code, price_start, last_day, use_sina=False)
        if hist.empty:
            price_series[ts_code] = pd.Series(dtype=float)
        else:
            closes = pd.to_numeric(hist['close'], errors='coerce').dropna()
            closes.index = closes.index.astype(str)
            price_series[ts_code] = closes.sort_index()

    def close_on_or_before(ts_code, trade_date, fallback):
        series = price_series.get(ts_code, pd.Series(dtype=float))
        subset = series[series.index <= trade_date]
        return float(subset.iloc[-1]) if not subset.empty else float(fallback)

    entry_groups = {
        date: group.sort_values(['Rank', 'Total_Score'], ascending=[True, False])
        for date, group in work.groupby('Entry_Date', dropna=False) if date
    }
    cash = float(initial_capital)
    active = {}
    executed = []
    orders = []
    curve_rows = []

    def audit(row, action, reason, positions_before):
        orders.append({
            'Signal_Date': row.get('Trade_Date', ''),
            'Entry_Date': row.get('Entry_Date', ''),
            'Rank': row.get('Rank', np.nan),
            'ts_code': row.get('ts_code', ''),
            'name': row.get('name', ''),
            'Total_Score': row.get('Total_Score', np.nan),
            'Wave_Count': row.get('Wave_Count', np.nan),
            'Portfolio_Action': action,
            'Portfolio_Reason': reason,
            'Positions_Before': positions_before,
        })

    day_set = set(days)
    for _, row in work[~work['Entry_Date'].isin(day_set)].iterrows():
        audit(row, '未计入组合', 'T+1买入日不在组合窗口', 0)

    for trade_date in days:
        # 与V40.4一致：先处理当日开盘买入；当日稍后退出的旧仓不提前腾位置。
        for _, row in entry_groups.get(trade_date, pd.DataFrame()).iterrows():
            ts_code = str(row['ts_code'])
            positions_before = len(active)
            if ts_code in active:
                audit(row, '未买入', '同一股票已在持仓', positions_before)
                continue
            if len(active) >= int(max_positions):
                audit(row, '未买入', '3个仓位已满', positions_before)
                continue
            buy_price = float(row['Buy_Price']) if pd.notna(row['Buy_Price']) else np.nan
            if not np.isfinite(buy_price) or buy_price <= 0:
                audit(row, '未买入', '买入价无效', positions_before)
                continue
            budget = min(float(position_budget), cash)
            shares = int(np.floor(budget / buy_price / int(lot_size)) * int(lot_size))
            if shares < int(lot_size):
                audit(row, '未买入', '可用现金不足一手', positions_before)
                continue
            cost = shares * buy_price
            cash -= cost
            trade = {
                'Signal_Date': row.get('Trade_Date', ''),
                'Entry_Date': trade_date,
                'Rank': row.get('Rank', np.nan),
                'ts_code': ts_code,
                'name': row.get('name', ''),
                'market': row.get('market', ''),
                'SW_L1': row.get('SW_L1', ''),
                'SW_L2': row.get('SW_L2', ''),
                'SW_L3': row.get('SW_L3', ''),
                'Wave_Count': row.get('Wave_Count', np.nan),
                'Total_Score': row.get('Total_Score', np.nan),
                'Buy_Price': round(buy_price, 3),
                'Shares': shares,
                'Entry_Cost': round(cost, 2),
                'Planned_Exit_Date': row.get('Exit_Date', ''),
                'Actual_Exit_Date': '',
                'Net_Exit_Price': np.nan,
                'Exit_Proceeds': np.nan,
                'PnL': np.nan,
                'Portfolio_Return (%)': np.nan,
                'Exit_Reason': row.get('Exit_Reason', ''),
                'Portfolio_Status': '持仓中',
                '_fallback_price': buy_price,
            }
            active[ts_code] = trade
            executed.append(trade)
            audit(row, '已买入', f'买入{shares}股', positions_before)

        exiting_codes = []
        for ts_code, trade in active.items():
            if normalize_date(trade.get('Planned_Exit_Date', '')) != trade_date:
                continue
            match = work[
                (work['Trade_Date'] == trade['Signal_Date']) & (work['ts_code'] == ts_code)
            ]
            net_exit_price = pd.to_numeric(match['Net_Exit_Price'], errors='coerce').iloc[-1]
            if not np.isfinite(net_exit_price) or net_exit_price <= 0:
                net_exit_price = close_on_or_before(ts_code, trade_date, trade['Buy_Price'])
            proceeds = trade['Shares'] * float(net_exit_price)
            cash += proceeds
            pnl = proceeds - trade['Entry_Cost']
            trade['Actual_Exit_Date'] = trade_date
            trade['Net_Exit_Price'] = round(float(net_exit_price), 3)
            trade['Exit_Proceeds'] = round(proceeds, 2)
            trade['PnL'] = round(pnl, 2)
            trade['Portfolio_Return (%)'] = round(pnl / trade['Entry_Cost'] * 100.0, 4)
            trade['Portfolio_Status'] = '已平仓'
            exiting_codes.append(ts_code)
        for ts_code in exiting_codes:
            active.pop(ts_code, None)

        market_value = 0.0
        for ts_code, trade in active.items():
            mark = close_on_or_before(ts_code, trade_date, trade['_fallback_price'])
            market_value += trade['Shares'] * mark
            trade['_last_mark'] = mark
        equity = cash + market_value
        curve_rows.append({
            'Trade_Date': trade_date,
            'Cash': round(cash, 2),
            'Market_Value': round(market_value, 2),
            'Equity': round(equity, 2),
            'Positions': len(active),
            'Exposure_pct': round(market_value / equity * 100.0 if equity > 0 else 0.0, 2),
            'Is_Empty': len(active) == 0,
        })

    curve = pd.DataFrame(curve_rows)
    curve['Daily_Return_pct'] = curve['Equity'].pct_change().fillna(
        curve['Equity'].iloc[0] / float(initial_capital) - 1.0
    ) * 100.0
    curve['Drawdown_pct'] = (curve['Equity'] / curve['Equity'].cummax() - 1.0) * 100.0

    for trade in executed:
        if trade['Portfolio_Status'] == '持仓中':
            mark = float(trade.get('_last_mark', trade['Buy_Price']))
            trade['Mark_Date'] = last_day
            trade['Mark_Price'] = round(mark, 3)
            trade['Market_Value'] = round(trade['Shares'] * mark, 2)
    ledger = pd.DataFrame(executed)
    if not ledger.empty:
        ledger = ledger.drop(columns=[c for c in ledger.columns if c.startswith('_')], errors='ignore')
    orders_df = pd.DataFrame(orders)
    closed = ledger[ledger['Portfolio_Status'] == '已平仓'].copy() if not ledger.empty else pd.DataFrame()

    summary = {
        'Initial_Capital': float(initial_capital),
        'Final_Equity': float(curve.iloc[-1]['Equity']),
        'Total_Return_pct': (float(curve.iloc[-1]['Equity']) / float(initial_capital) - 1.0) * 100.0,
        'Max_Drawdown_pct': float(curve['Drawdown_pct'].min()),
        'Empty_Ratio_pct': float(curve['Is_Empty'].mean() * 100.0),
        'Invested_Days': int((~curve['Is_Empty']).sum()),
        'Average_Positions': float(curve['Positions'].mean()),
        'Average_Exposure_pct': float(curve['Exposure_pct'].mean()),
        'Executed_Entries': int(len(ledger)),
        'Closed_Trades': int(len(closed)),
        'Open_Positions': int((ledger['Portfolio_Status'] == '持仓中').sum()) if not ledger.empty else 0,
        'Closed_Win_Rate_pct': float((closed['PnL'] > 0).mean() * 100.0) if not closed.empty else np.nan,
        'Fixed_Stop_Rate_pct': float(closed['Exit_Reason'].astype(str).str.contains('固定止损').mean() * 100.0) if not closed.empty else np.nan,
        'Rejected_Full': int(orders_df['Portfolio_Reason'].eq('3个仓位已满').sum()) if not orders_df.empty else 0,
        'Rejected_Duplicate': int(orders_df['Portfolio_Reason'].eq('同一股票已在持仓').sum()) if not orders_df.empty else 0,
    }
    return curve, ledger, orders_df, summary


def build_run_fingerprint(config):
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]

# ---------------------------
# UI 及 主程序
# ---------------------------
with st.sidebar:
    st.header("V40.6 可靠缓存三仓回测版")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数 (设为 1 即启动实盘雷达)", value=100, step=1)
    
    TOP_BACKTEST = st.number_input("每日优选 TopK", value=3)
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 清除行情缓存"):
        if os.path.exists(CACHE_FILE_NAME):
            os.remove(CACHE_FILE_NAME)
        if os.path.isdir(CACHE_DIR_NAME):
            shutil.rmtree(CACHE_DIR_NAME)
        st.success("行情、复权和市值逐日缓存已清除。")
    CHECKPOINT_DIR = "backtest_checkpoint_v40_6_reliable"
    if st.button("🗑️ 清除断点记录 (重新回测)"):
        if os.path.isdir(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
        st.success("所有V40.6可靠续传进度已清理！")
            
    st.markdown("---")
    st.subheader("💰 核心护城河门槛")
    MIN_PRICE = st.number_input("最低股价 (元)", value=20.0) 
    col1, col2 = st.columns(2)
    MIN_MV = col1.number_input("最小市值(亿)", value=200.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)

    st.markdown("---")
    st.subheader("💸 与V40.4一致的交易成本")
    BUY_SLIPPAGE = st.number_input("买入滑点(%)", min_value=0.0, value=0.20, step=0.05)
    SELL_SLIPPAGE = st.number_input("卖出滑点(%)", min_value=0.0, value=0.20, step=0.05)
    COMMISSION = st.number_input("单边佣金(%)", min_value=0.0, value=0.03, step=0.01)
    SELL_TAX = st.number_input("卖出税费(%)", min_value=0.0, value=0.05, step=0.01)
    st.caption("组合固定为30万元、最多3只、单只目标约10万元。")

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button(f"🚀 启动 V40.6 四大神盾追踪"):
    SINA_STATUS = {'success': 0, 'fail': 0}
    trade_days_list = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_list: st.stop()

    run_config = {
        'version': 'v40.6_reliable_cache_1',
        'end_date': backtest_date_end.strftime('%Y%m%d'),
        'backtest_days': int(BACKTEST_DAYS),
        'top_k': int(TOP_BACKTEST),
        'min_price': float(MIN_PRICE),
        'min_mv': float(MIN_MV),
        'max_mv': float(MAX_MV),
        'buy_slippage': float(BUY_SLIPPAGE),
        'sell_slippage': float(SELL_SLIPPAGE),
        'commission': float(COMMISSION),
        'sell_tax': float(SELL_TAX),
    }
    run_key = build_run_fingerprint(run_config)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, f'signals_{run_key}.csv')
    STATUS_FILE = os.path.join(CHECKPOINT_DIR, f'status_{run_key}.csv')

    checkpoint_signals = pd.DataFrame()
    status_df = pd.DataFrame(columns=['Trade_Date', 'Status', 'Detail'])
    if not RESUME_CHECKPOINT:
        for path in [CHECKPOINT_FILE, STATUS_FILE]:
            if os.path.exists(path):
                os.remove(path)
    else:
        if os.path.exists(CHECKPOINT_FILE):
            try:
                checkpoint_signals = pd.read_csv(CHECKPOINT_FILE)
                checkpoint_signals['Trade_Date'] = checkpoint_signals['Trade_Date'].map(normalize_date)
            except Exception:
                st.error('信号断点文件损坏，请点击“清除断点记录”后重新回测。')
                st.stop()
        if os.path.exists(STATUS_FILE):
            try:
                status_df = pd.read_csv(STATUS_FILE)
                status_df['Trade_Date'] = status_df['Trade_Date'].map(normalize_date)
            except Exception:
                st.error('完成日期断点文件损坏，请点击“清除断点记录”后重新回测。')
                st.stop()

    processed_dates = set(
        status_df.loc[status_df['Status'].eq('completed'), 'Trade_Date'].astype(str)
    )
    
    if not get_all_historical_data(trade_days_list, use_cache=True): st.stop()
            
    dates_to_run = [d for d in trade_days_list if d not in processed_dates]
    if not dates_to_run:
        st.success(f"🎉 扫描已全部完毕！可靠断点已覆盖 {len(processed_dates)}/{len(trade_days_list)} 天。")
    else:
        bar = st.progress(0, text="箱体首发与四大神盾过滤中...")
        for i, date in enumerate(dates_to_run):
            
            is_realtime_radar = (int(BACKTEST_DAYS) == 1 and date == datetime.now().strftime("%Y%m%d"))
            run_timestamp = time.time() if is_realtime_radar else None
            
            res, err = run_backtest_for_a_day(
                date, int(TOP_BACKTEST), MIN_MV, MAX_MV, MIN_PRICE,
                BUY_SLIPPAGE, SELL_SLIPPAGE, COMMISSION, SELL_TAX,
                use_sina=is_realtime_radar, run_timestamp=run_timestamp
            )
            
            if not res.empty:
                res['Trade_Date'] = date
                checkpoint_signals = pd.concat([checkpoint_signals, res], ignore_index=True)
                checkpoint_signals['Trade_Date'] = checkpoint_signals['Trade_Date'].map(normalize_date)
                checkpoint_signals = checkpoint_signals.drop_duplicates(
                    ['Trade_Date', 'ts_code'], keep='last'
                )
                atomic_csv_save(checkpoint_signals, CHECKPOINT_FILE)

            analysis_completed = (not res.empty) or (err == '无标的')
            if analysis_completed:
                status_row = pd.DataFrame([{
                    'Trade_Date': date,
                    'Status': 'completed',
                    'Detail': f'信号{len(res)}只' if not res.empty else '无标的',
                }])
                status_df = pd.concat([status_df, status_row], ignore_index=True)
                status_df = status_df.drop_duplicates('Trade_Date', keep='last')
                atomic_csv_save(status_df, STATUS_FILE)
                processed_dates.add(date)
            else:
                st.warning(f'{date} 未完成：{err or "未知数据错误"}，未写入完成断点，下次会自动重试。')

            bar.progress(
                (i+1)/len(dates_to_run),
                text=f"分析中: {date}｜已持久化 {len(processed_dates)}/{len(trade_days_list)} 天",
            )
        bar.empty()

    results = [checkpoint_signals] if not checkpoint_signals.empty else []
    
    if int(BACKTEST_DAYS) == 1:
        st.markdown("---")
        if SINA_STATUS['success'] > 0:
            st.success(f"✅ **盘中实时探针响应正常**：成功接入新浪底层数据 {SINA_STATUS['success']} 次，行情已接管。")
        elif SINA_STATUS['fail'] > 0:
            st.error(f"❌ **盘中实时探针警告**：新浪数据抓取失败 {SINA_STATUS['fail']} 次。请确认当前是否在交易时间。")
        else:
            st.info("ℹ️ 实时探针未触发（可能由于基础选股条件未通过）。")
        st.markdown("---")
    
    if results:
        all_res = pd.concat(results)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
        
        st.header(f"📊 V40.6 可靠缓存三仓回测版")

        portfolio_curve, portfolio_ledger, portfolio_orders, portfolio_summary = build_portfolio_backtest(
            all_res,
            trade_days_list,
            initial_capital=INITIAL_CAPITAL,
            max_positions=MAX_PORTFOLIO_POSITIONS,
            position_budget=POSITION_BUDGET,
            lot_size=LOT_SIZE,
        )

        st.subheader("💼 30万元·最多3只真实组合")
        metrics_row1 = st.columns(4)
        metrics_row1[0].metric("组合期末权益", f"¥{portfolio_summary['Final_Equity']:,.0f}")
        metrics_row1[1].metric("组合总收益", f"{portfolio_summary['Total_Return_pct']:.2f}%")
        metrics_row1[2].metric("最大回撤", f"{portfolio_summary['Max_Drawdown_pct']:.2f}%")
        metrics_row1[3].metric("真实空仓率", f"{portfolio_summary['Empty_Ratio_pct']:.1f}%")

        metrics_row2 = st.columns(4)
        metrics_row2[0].metric(
            "有持仓日",
            f"{portfolio_summary['Invested_Days']}/{len(portfolio_curve)}"
            if not portfolio_curve.empty else "0/0",
        )
        metrics_row2[1].metric("平均持仓数", f"{portfolio_summary['Average_Positions']:.2f}/3")
        metrics_row2[2].metric("平均资金暴露", f"{portfolio_summary['Average_Exposure_pct']:.1f}%")
        metrics_row2[3].metric(
            "已执行/已平仓/在持",
            f"{portfolio_summary['Executed_Entries']}/{portfolio_summary['Closed_Trades']}/{portfolio_summary['Open_Positions']}",
        )

        metrics_row3 = st.columns(4)
        closed_win = portfolio_summary['Closed_Win_Rate_pct']
        fixed_stop = portfolio_summary['Fixed_Stop_Rate_pct']
        metrics_row3[0].metric("已平仓胜率", "N/A" if pd.isna(closed_win) else f"{closed_win:.1f}%")
        metrics_row3[1].metric("固定止损率", "N/A" if pd.isna(fixed_stop) else f"{fixed_stop:.1f}%")
        metrics_row3[2].metric("因仓位已满未买", f"{portfolio_summary['Rejected_Full']}笔")
        metrics_row3[3].metric("因重复持仓未买", f"{portfolio_summary['Rejected_Duplicate']}笔")

        if not portfolio_curve.empty:
            equity_chart = portfolio_curve[['Trade_Date', 'Equity']].copy()
            equity_chart['Trade_Date'] = pd.to_datetime(equity_chart['Trade_Date'], format='%Y%m%d')
            st.line_chart(equity_chart.set_index('Trade_Date')['Equity'])

        st.subheader("🗓️ 周度生存与收益切片 (剔除不符合开盘要求的无效标的)")
        cols_row1 = st.columns(4)
        cols_row2 = st.columns(4)
        
        # 排除掉那些开盘直接被剔除的标的进行胜率统计
        valid_trades_only = all_res[~all_res['Exit_Reason'].str.contains('剔除', na=False)]
        
        for w in range(1, 9):
            col_name = f'Return_W{w} (%)'
            if col_name in valid_trades_only.columns:
                valid = valid_trades_only.dropna(subset=[col_name]) 
                target_col = cols_row1[w-1] if w <= 4 else cols_row2[w-5]
                with target_col:
                    if not valid.empty:
                        avg = valid[col_name].mean()
                        win = (valid[col_name] > 0).mean() * 100
                        st.metric(f"W{w} 均益/胜率 (存活{len(valid)}只)", f"{avg:.2f}% / {win:.1f}%")
                    else:
                        st.metric(f"W{w} 无持仓", "N/A")
                        
        st.subheader("📋 实战定型榜单")
        display_cols = [
            'Rank', 'Trade_Date', 'Entry_Date', 'name', 'ts_code', 'market', 'SW_L1', 'SW_L2', 'SW_L3',
            'Wave_Count', 'Signal_Close', 'Buy_Price', 'Gap_pct (%)', 'Total_Score', 'Breakout_S',
            'Volume_S', 'circ_mv', 'Exit_Date', 'Net_Exit_Price', 'Realized_Return (%)', 'Exit_Reason'
        ] + [f'Return_W{w} (%)' for w in range(1, 9)]
        final_cols = [c for c in display_cols if c in all_res.columns]
    
        display_df = all_res[final_cols].sort_values(['Trade_Date', 'Rank'], ascending=[False, True]).reset_index(drop=True)
        
        def color_exit(val):
            if isinstance(val, str):
                if '剔除' in val: return 'color: white; background-color: darkgray'
                elif '固定止损' in val: return 'color: white; background-color: darkred'
                elif '保本止盈' in val: return 'color: orange'
                elif '移动止盈' in val: return 'color: green'
                elif '周期结束平仓' in val: return 'color: blue'
            return ''
        
        if 'Exit_Reason' in display_df.columns:
            try:
                st.dataframe(display_df.style.map(color_exit, subset=['Exit_Reason']), use_container_width=True)
            except AttributeError:
                st.dataframe(display_df.style.applymap(color_exit, subset=['Exit_Reason']), use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
        
        csv = all_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载完整信号轨迹 (CSV)", csv, "signals_v40_6_same_pool.csv", "text/csv")

        download_cols = st.columns(3)
        download_cols[0].download_button(
            "📥 下载组合曲线",
            portfolio_curve.to_csv(index=False).encode('utf-8-sig'),
            "portfolio_curve_v40_6_same_pool.csv",
            "text/csv",
        )
        download_cols[1].download_button(
            "📥 下载组合成交账本",
            portfolio_ledger.to_csv(index=False).encode('utf-8-sig'),
            "portfolio_ledger_v40_6_same_pool.csv",
            "text/csv",
        )
        download_cols[2].download_button(
            "📥 下载买入/未买审计",
            portfolio_orders.to_csv(index=False).encode('utf-8-sig'),
            "portfolio_orders_v40_6_same_pool.csv",
            "text/csv",
        )

        if not portfolio_ledger.empty:
            st.subheader("🧾 真实组合成交账本")
            st.dataframe(portfolio_ledger, use_container_width=True)
    else:
        st.warning("⚠️ 暂无符合条件的标的。")
