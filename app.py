# -*- coding: utf-8 -*-
"""
选股王 · V40.6 辅助版1.0
------------------------------------------------
定位：V40.4 的辅助突破诊断工具，不替代或修改 V40.4。

辅助版1.0核心改进：
1. [快速开发] 默认回测100个交易日；候选扫描与未来收益评估分离，只评估两种排序的候选并集。
2. [双模式对照] 同一批历史数据同时输出“原版量价排名”和“辅助质量排名”，避免重复下载和重复逐日扫描。
3. [辅助诊断] 增加箱体宽度、MA20乖离、突破距离、5日涨幅、ATR、收盘位置、换手率、日宽度与板块宽度。
4. [软分级] 量比>2.5等过热特征不直接删除，降为B级观察；A级候选才进入辅助组合。
5. [真实成本保护] 盈利达到10%后，从下一交易日起启用买入成本+0.3%的保护止损。
6. [幸存者偏差修复] 提前退出后的实现收益延续到后续周，周度报表同时显示持仓生存数量。
7. [可靠续传] 继续复用V40.6可靠逐日行情缓存，但使用独立断点与导出文件。

保留原V40.6口径：
1. [硬门槛 1：盘子基座] 侧边栏默认流通市值下限为 200 亿，隔绝微盘股的画线诱多陷阱。
2. [硬门槛 2：温和爆破] 突破量比上限严格锁定在 3.0倍 (1.3 <= vol <= 3.0)，绞杀“天量见天价”的分歧坑。
3. [硬门槛 3：开盘定生死] 在 T+1 买入引擎中加入集合竞价拦截器。若高开>5%或低开<-3%，直接放弃买入，剔除该标的！
4. [废除主观加分] 尊重客观数据，剔除原有的“洗盘2-3次加分”逻辑，所有分数纯靠量价真实动能。
5. [回测口径统一] 股票池改为V40.4历史申万科技池，股价>=20元、流通市值200~1000亿元。
6. [真实组合] 30万元初始资金、最多3只持仓、单只目标约10万元，并计入滑点、佣金和卖出税费。
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
AUXILIARY_MAX_POSITIONS = 1
AUXILIARY_POSITION_BUDGET = INITIAL_CAPITAL / MAX_PORTFOLIO_POSITIONS
AUXILIARY_SCORE_THRESHOLD = 55.0
PROTECTION_TRIGGER_PCT = 10.0
PROTECTION_BUFFER_PCT = 0.3

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
st.set_page_config(page_title="选股王 V40.6 辅助版1.0", layout="wide")
st.title("选股王 V40.6 辅助版1.0：突破雷达 + 双模式诊断")

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

    if basic_frames:
        basic = pd.concat(basic_frames, ignore_index=True)
        basic['trade_date'] = basic['trade_date'].astype(str)
        GLOBAL_DAILY_BASIC = (
            basic.drop_duplicates(['ts_code', 'trade_date'])
            .set_index(['ts_code', 'trade_date']).sort_index(level=[0, 1])
        )
    else:
        GLOBAL_DAILY_BASIC = pd.DataFrame()

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
    successful_market_dates, successful_basic_dates = [], []
    failed_dates = []
    failed_details = {}
    scanned_dates = []
    consecutive_failures = 0
    downloaded_now = 0
    reused_days = 0
    my_bar = st.progress(0, text="检查逐日持久化缓存...")

    for i, date in enumerate(all_dates):
        scanned_dates.append(date)
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

        market_complete = day_components_complete(data, need_daily_basic=False)
        basic_complete = (
            isinstance(data.get('daily_basic'), pd.DataFrame)
            and not data['daily_basic'].empty
        )
        if market_complete:
            adj_frames.append(data['adj'])
            daily_frames.append(data['daily'])
            successful_market_dates.append(date)
        if need_daily_basic and basic_complete:
            basic_frames.append(data['daily_basic'])
            successful_basic_dates.append(date)

        if market_complete and (not need_daily_basic or basic_complete):
            consecutive_failures = 0
            if was_complete:
                reused_days += 1
            else:
                downloaded_now += 1
        else:
            failed_dates.append(date)
            missing_parts = []
            if not market_complete:
                missing_parts.extend(['daily/adj_factor'])
            if need_daily_basic and not basic_complete:
                missing_parts.append('daily_basic')
            failed_details[date] = '+'.join(missing_parts) or '未知组件'
            consecutive_failures += 1

        my_bar.progress(
            (i + 1) / len(all_dates),
            text=f"缓存核验 {i+1}/{len(all_dates)}｜本次新增 {downloaded_now}｜已复用 {reused_days}",
        )
        if consecutive_failures >= 3:
            break
    my_bar.empty()

    failed_set = set(failed_dates)
    trailing_failed_dates = []
    for date in reversed(all_dates):
        if date in failed_set:
            trailing_failed_dates.append(date)
        else:
            break
    trailing_failed_set = set(trailing_failed_dates)
    allow_unpublished_tail = bool(failed_set) and (
        len(scanned_dates) == len(all_dates)
        and failed_set == trailing_failed_set
        and all(date >= latest_trade_date for date in trailing_failed_set)
    )
    hard_failure = (
        len(scanned_dates) != len(all_dates)
        or bool(failed_set - trailing_failed_set)
        or (bool(failed_set) and not allow_unpublished_tail)
    )

    if hard_failure:
        examples = [f'{date}({failed_details.get(date, "未扫描")})' for date in failed_dates[:5]]
        st.warning(
            f"网络中断或历史区间内部数据缺失：已扫描 {len(scanned_dates)}/{len(all_dates)} 日，"
            f"完整行情 {len(successful_market_dates)} 日。缺失示例：{examples}。"
            "请在网络恢复后重启，程序只补缺失项。"
        )
        return False

    if allow_unpublished_tail:
        details = [f'{date}({failed_details[date]})' for date in sorted(trailing_failed_set)]
        st.warning(
            f"与V40.4口径一致：末端计划交易日尚未产生完整数据 {details}，"
            "保留在250日计划样本中，但该日不产生选股信号。"
        )

    with st.spinner("正在构建内存索引和快速总缓存..."):
        set_global_market_frames(adj_frames, daily_frames, basic_frames)
        atomic_pickle_dump({
            'cache_version': CACHE_VERSION,
            'covered_market_dates': sorted(set(successful_market_dates)),
            'covered_basic_dates': sorted(set(successful_basic_dates)),
            'start_date': all_dates[0],
            'end_date': all_dates[-1],
            'adj': GLOBAL_ADJ_FACTOR,
            'daily': GLOBAL_DAILY_RAW,
            'daily_basic': GLOBAL_DAILY_BASIC,
        }, CACHE_FILE_NAME)
    st.success(
        f"✅ 计划数据日 {len(all_dates)}，已持久化完整行情 {len(successful_market_dates)} 日；"
        "后续逐日分析直接读取本地数据。"
    )
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
    
    # 10日箱体与波动诊断。全部使用D0及此前数据，不读取未来。
    df['box_high_10'] = df['high'].rolling(window=10).max().shift(1)
    df['box_low_10'] = df['low'].rolling(window=10).min().shift(1)
    previous_close = df['close'].shift(1)
    true_range = pd.concat([
        df['high'] - df['low'],
        (df['high'] - previous_close).abs(),
        (df['low'] - previous_close).abs(),
    ], axis=1).max(axis=1)
    df['atr14'] = true_range.rolling(14).mean()
    # 突破前5个交易日涨幅，刻意排除信号当日，避免把当日长阳重复计分。
    df['prior_5d_return_pct'] = (df['close'].shift(1) / df['close'].shift(6) - 1.0) * 100.0
    
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
        ma20_bias_pct = (row['close'] / row['ma20'] - 1.0) * 100.0 if row['ma20'] > 0 else np.nan
        breakout_distance_pct = (
            (row['close'] / row['box_high_10'] - 1.0) * 100.0
            if row['box_high_10'] > 0 else np.nan
        )
        box_width_10_pct = (
            (row['box_high_10'] / row['box_low_10'] - 1.0) * 100.0
            if row['box_low_10'] > 0 else np.nan
        )
        atr_pct = row['atr14'] / row['close'] * 100.0 if row['close'] > 0 else np.nan
        close_location = (
            (row['close'] - row['low']) / candle_range * 100.0
            if candle_range > 0 else 100.0
        )
        res['vol_ratio'] = vol_ratio
        res['pre_close'] = prev_row['close']            
        res['wave_count'] = wave_count
        res['day_gain_pct'] = (row['close'] / prev_row['close'] - 1.0) * 100.0
        res['ma20_bias_pct'] = ma20_bias_pct
        res['breakout_distance_pct'] = breakout_distance_pct
        res['box_width_10_pct'] = box_width_10_pct
        res['prior_5d_return_pct'] = row['prior_5d_return_pct']
        res['atr_pct'] = atr_pct
        res['close_location_pct'] = close_location
        res['box_high_10'] = row['box_high_10']
        res['box_low_10'] = row['box_low_10']
        
    res['last_close'] = row['close']
    res['bottom_line'] = row['low'] 
    res['ma20'] = row['ma20']
    
    return res

# ---------------------------
# 三层简化止盈止损系统 (加入集合竞价拦截器)
# ---------------------------
def get_medium_term_future(
    ts_code, selection_date, signal_close, bottom_line, breakout_line=None, hold_weeks=8,
    buy_slippage_pct=0.20, sell_slippage_pct=0.20,
    commission_pct=0.03, sell_tax_pct=0.05, use_sina=False,
):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_fetch = (d0 - timedelta(days=60)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=150)).strftime("%Y%m%d") 
    
    hist_full = get_qfq_data_v4_optimized_final(ts_code, start_date=start_fetch, end_date=end_future, use_sina=use_sina)
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    for w in range(1, hold_weeks + 1):
        results[f'Eligible_W{w}'] = False
        results[f'Held_W{w}'] = False
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
    results['MFE_pct (%)'] = np.nan
    results['MAE_pct (%)'] = np.nan
    results['Days_To_MFE10'] = np.nan
    results['False_Breakout_5D'] = False
    results['Protection_Trigger_Day'] = np.nan
    results['Protection_Stop_Price'] = np.nan
    
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
        # 退出后的实现收益延续到后续周，避免只统计幸存强者。
        for week in range(current_week, hold_weeks + 1):
            results[f'Return_W{week} (%)'] = final_return
            results[f'Eligible_W{week}'] = True
            results[f'Held_W{week}'] = False
        return final_return

    exit_triggered = False
    tier = 0  
    peak_close = buy_price
    trough_low = buy_price
    pending_exit_reason = None  
    protection_active_from_day = None
    protection_stop_price = buy_price * (1.0 + PROTECTION_BUFFER_PCT / 100.0)
    
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
        trough_low = min(trough_low, curr_low)
        peak_profit_pct = (peak_close - buy_price) / buy_price

        if (
            day_count <= 5 and breakout_line is not None
            and np.isfinite(breakout_line) and breakout_line > 0
            and curr_close <= breakout_line
        ):
            results['False_Breakout_5D'] = True

        if pd.isna(results['Days_To_MFE10']) and peak_profit_pct >= PROTECTION_TRIGGER_PCT / 100.0:
            results['Days_To_MFE10'] = int(day_count)
        
        hard_stop_price = buy_price * (1.0 + hard_stop_limit)
        protection_active = (
            protection_active_from_day is not None
            and day_count >= protection_active_from_day
        )
        active_stop_price = max(hard_stop_price, protection_stop_price) if protection_active else hard_stop_price
        if curr_low <= active_stop_price:
            raw_exit_price = curr_open if curr_open < active_stop_price else active_stop_price
            if protection_active and protection_stop_price >= hard_stop_price:
                stop_reason = f"成本保护止损(+{PROTECTION_BUFFER_PCT:.1f}%原始价)"
            else:
                stop_reason = f"固定止损(破{int(hard_stop_limit*100)}%)"
            finalize_exit(
                raw_exit_price, current_week, day_count,
                stop_reason, row.name,
            )
            exit_triggered = True
            break
        
        if tier == 0 and peak_profit_pct >= PROTECTION_TRIGGER_PCT / 100.0:
            tier = 1
            protection_active_from_day = day_count + 1
            results['Protection_Trigger_Day'] = int(day_count)
            results['Protection_Stop_Price'] = round(protection_stop_price, 3)
                
        if tier == 1:
            if peak_profit_pct >= 0.20:
                tier = 2
                
        if tier == 2:
            giveback = (peak_close - curr_close) / peak_close
            if giveback >= 0.15:
                pending_exit_reason = "移动止盈(回撤15%)"
            
        if day_count % 5 == 0:
            results[f'Return_W{current_week} (%)'] = net_return(curr_close)
            results[f'Eligible_W{current_week}'] = True
            results[f'Held_W{current_week}'] = True
            
    if not exit_triggered and len(hist_future) >= hold_weeks * 5:
        last_row = hist_future.iloc[hold_weeks * 5 - 1]
        finalize_exit(
            last_row['close'], hold_weeks, hold_weeks * 5,
            '周期结束平仓', last_row.name,
        )
    elif not exit_triggered:
        results['Holding_Days'] = min(len(hist_future), hold_weeks * 5)

    results['MFE_pct (%)'] = round((peak_close / buy_price - 1.0) * 100.0, 2)
    results['MAE_pct (%)'] = round((trough_low / buy_price - 1.0) * 100.0, 2)
        
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


def _finite_number(value, default=np.nan):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def add_auxiliary_diagnostics(candidates):
    """在D0数据上计算可解释的辅助质量分；不使用任何未来收益。"""
    if candidates.empty:
        return candidates
    work = candidates.copy()
    work['Day_Signal_Count'] = len(work)
    work['Sector_Signal_Count'] = work.groupby('SW_L1')['ts_code'].transform('size')

    auxiliary_scores = []
    grades = []
    grade_reasons = []
    for _, row in work.iterrows():
        vol_ratio = _finite_number(row.get('Vol_Ratio'))
        box_width = _finite_number(row.get('Box_Width_10_pct'))
        ma20_bias = _finite_number(row.get('MA20_Bias_pct'))
        breakout_distance = _finite_number(row.get('Breakout_Distance_pct'))
        prior_5d = _finite_number(row.get('Prior_5D_Return_pct'))
        close_location = _finite_number(row.get('Close_Location_pct'))
        day_gain = _finite_number(row.get('Day_Gain_pct'))
        day_count = int(_finite_number(row.get('Day_Signal_Count'), 0))
        sector_count = int(_finite_number(row.get('Sector_Signal_Count'), 0))

        # 质量分采用封顶/居中评分，避免当日涨幅和量比越大就无限加分。
        box_score = np.clip((20.0 - box_width) / 15.0, 0.0, 1.0) * 25.0
        close_score = np.clip((close_location - 60.0) / 40.0, 0.0, 1.0) * 20.0
        bias_score = np.clip((18.0 - ma20_bias) / 16.0, 0.0, 1.0) * 20.0
        volume_score = np.clip(1.0 - abs(vol_ratio - 1.9) / 1.0, 0.0, 1.0) * 15.0
        breakout_score = np.clip((10.0 - breakout_distance) / 10.0, 0.0, 1.0) * 10.0
        breadth_score = 5.0 if day_count >= 2 else 0.0
        sector_score = 5.0 if sector_count == 2 else (2.0 if sector_count == 1 else 0.0)
        chase_penalty = max(0.0, prior_5d - 15.0) * 0.5 + max(0.0, day_gain - 12.0) * 0.5
        auxiliary_score = float(np.clip(
            box_score + close_score + bias_score + volume_score
            + breakout_score + breadth_score + sector_score - chase_penalty,
            0.0, 100.0,
        ))

        reasons = []
        if vol_ratio > 2.5:
            reasons.append('量比>2.5')
        if ma20_bias > 18.0:
            reasons.append('偏离MA20>18%')
        if prior_5d > 25.0:
            reasons.append('突破前5日涨幅>25%')
        if breakout_distance > 10.0:
            reasons.append('突破箱顶>10%')
        if close_location < 70.0:
            reasons.append('收盘位置<70%')
        if auxiliary_score < AUXILIARY_SCORE_THRESHOLD:
            reasons.append(f'辅助分<{AUXILIARY_SCORE_THRESHOLD:.0f}')

        auxiliary_scores.append(round(auxiliary_score, 2))
        grades.append('A_可执行' if not reasons else 'B_观察')
        grade_reasons.append('通过辅助质量门槛' if not reasons else '；'.join(reasons))

    work['Auxiliary_Score'] = auxiliary_scores
    work['Auxiliary_Grade'] = grades
    work['Auxiliary_Grade_Reason'] = grade_reasons
    work = work.sort_values(['Total_Score', 'ts_code'], ascending=[False, True]).reset_index(drop=True)
    work['Original_Rank'] = np.arange(1, len(work) + 1)
    auxiliary_order = work.sort_values(
        ['Auxiliary_Score', 'Total_Score', 'ts_code'], ascending=[False, False, True]
    ).index
    auxiliary_rank_map = {index: rank for rank, index in enumerate(auxiliary_order, start=1)}
    work['Auxiliary_Candidate_Rank'] = work.index.map(auxiliary_rank_map)
    return work


def run_backtest_for_a_day(
    last_trade, TOP_BACKTEST, EVALUATION_POOL_SIZE, MIN_MV, MAX_MV, MIN_PRICE,
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
        basic_columns = [
            column for column in ['ts_code', 'circ_mv', 'turnover_rate']
            if column in daily_basic.columns
        ]
        df = df.merge(daily_basic[basic_columns], on='ts_code', how='left')
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
            
        pct_chg = ind['day_gain_pct']
        score_breakout = pct_chg * 10 
        score_vol = ind['vol_ratio'] * 10
        total_score = score_breakout + score_vol
        
        # 【改动4：废除主观加分】去掉了 wave_cnt 的 30 分加成，让排序回归真实的量价爆发力度。
        wave_cnt = ind.get('wave_count', 3)
            
        record_dict = {
            'ts_code': row.ts_code, 'name': row.name, 'Signal_Close': ind['last_close'], 
            'market': row.market,
            'SW_L1': membership['l1'], 'SW_L2': membership['l2'], 'SW_L3': membership['l3'],
            'Wave_Count': wave_cnt,
            'circ_mv': row.circ_mv_billion,
            'Total_Score': round(total_score, 1),
            'Breakout_S': round(score_breakout, 1),
            'Volume_S': round(score_vol, 1),
            'Day_Gain_pct': round(ind['day_gain_pct'], 4),
            'Vol_Ratio': round(ind['vol_ratio'], 4),
            'MA20_Bias_pct': round(ind['ma20_bias_pct'], 4),
            'Breakout_Distance_pct': round(ind['breakout_distance_pct'], 4),
            'Box_Width_10_pct': round(ind['box_width_10_pct'], 4),
            'Prior_5D_Return_pct': round(ind['prior_5d_return_pct'], 4),
            'ATR_pct': round(ind['atr_pct'], 4),
            'Close_Location_pct': round(ind['close_location_pct'], 2),
            'Turnover_Rate': round(_finite_number(getattr(row, 'turnover_rate', np.nan)), 4),
            '_Bottom_Line': ind['bottom_line'],
            '_Breakout_Line': ind['box_high_10'],
        }
        records.append(record_dict)
            
    if not records: return pd.DataFrame(), "无标的"
    
    candidates = add_auxiliary_diagnostics(pd.DataFrame(records))
    pool_size = max(int(EVALUATION_POOL_SIZE), int(TOP_BACKTEST))
    original_pool = candidates.nsmallest(pool_size, 'Original_Rank')
    grade_a_candidates = candidates[candidates['Auxiliary_Grade'].eq('A_可执行')]
    auxiliary_pool_source = grade_a_candidates if not grade_a_candidates.empty else candidates
    auxiliary_pool = auxiliary_pool_source.nsmallest(pool_size, 'Auxiliary_Candidate_Rank')
    evaluation_indexes = sorted(set(original_pool.index) | set(auxiliary_pool.index))
    evaluated = candidates.loc[evaluation_indexes].copy()

    future_rows = []
    for _, candidate in evaluated.iterrows():
        future_rows.append(get_medium_term_future(
            candidate['ts_code'], last_trade, candidate['Signal_Close'],
            candidate['_Bottom_Line'], breakout_line=candidate['_Breakout_Line'],
            hold_weeks=8,
            buy_slippage_pct=buy_slippage_pct,
            sell_slippage_pct=sell_slippage_pct,
            commission_pct=commission_pct,
            sell_tax_pct=sell_tax_pct,
            use_sina=use_sina,
        ))
    future_df = pd.DataFrame(future_rows, index=evaluated.index)
    for column in future_df.columns:
        evaluated[column] = future_df[column]

    evaluated['Original_Selected'] = evaluated['Original_Rank'].le(int(TOP_BACKTEST))
    evaluated['Auxiliary_Selected'] = False
    evaluated['Auxiliary_Rank'] = np.nan
    auxiliary_executable = evaluated[
        evaluated['Auxiliary_Grade'].eq('A_可执行')
        & bool_series(evaluated['Tradable'])
    ].sort_values(['Auxiliary_Score', 'Total_Score'], ascending=[False, False]).head(int(TOP_BACKTEST))
    if not auxiliary_executable.empty:
        evaluated.loc[auxiliary_executable.index, 'Auxiliary_Selected'] = True
        evaluated.loc[auxiliary_executable.index, 'Auxiliary_Rank'] = np.arange(
            1, len(auxiliary_executable) + 1
        )

    evaluated['Rank'] = evaluated['Original_Rank']
    evaluated['Evaluation_Pool_Size'] = pool_size
    evaluated['Qualified_Candidate_Count'] = len(candidates)
    evaluated = evaluated.drop(columns=['_Bottom_Line', '_Breakout_Line'], errors='ignore')
    evaluated = evaluated.sort_values(['Original_Rank', 'Auxiliary_Candidate_Rank'])
    return evaluated.reset_index(drop=True), None


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
    mode_label='V40.6原版量价排名',
):
    days = sorted({normalize_date(day) for day in trade_days if normalize_date(day)})
    empty_columns = [
        'Trade_Date', 'Cash', 'Market_Value', 'Equity', 'Positions',
        'Exposure_pct', 'Is_Empty', 'Daily_Return_pct', 'Drawdown_pct',
    ]
    if signals is None or signals.empty or not days:
        return pd.DataFrame(columns=empty_columns), pd.DataFrame(), pd.DataFrame(), {
            'Mode': mode_label,
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
    full_position_reason = f'{int(max_positions)}个仓位已满'

    def audit(row, action, reason, positions_before):
        orders.append({
            'Diagnostic_Mode': mode_label,
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
                audit(row, '未买入', full_position_reason, positions_before)
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
                'Diagnostic_Mode': mode_label,
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
            'Diagnostic_Mode': mode_label,
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
    curve['Drawdown_pct'] = (
        curve['Equity'] / curve['Equity'].cummax().clip(lower=float(initial_capital)) - 1.0
    ) * 100.0

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
        'Mode': mode_label,
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
        'Cost_Protection_Rate_pct': float(closed['Exit_Reason'].astype(str).str.contains('成本保护').mean() * 100.0) if not closed.empty else np.nan,
        'Rejected_Full': int(orders_df['Portfolio_Reason'].eq(full_position_reason).sum()) if not orders_df.empty else 0,
        'Rejected_Duplicate': int(orders_df['Portfolio_Reason'].eq('同一股票已在持仓').sum()) if not orders_df.empty else 0,
    }

    closed_pnl = pd.to_numeric(closed.get('PnL', pd.Series(dtype=float)), errors='coerce').dropna()
    gross_profit = float(closed_pnl[closed_pnl > 0].sum())
    gross_loss = float(-closed_pnl[closed_pnl < 0].sum())
    total_pnl = float(closed_pnl.sum())
    top1_pnl = float(closed_pnl.nlargest(1).sum()) if len(closed_pnl) else 0.0
    top3_pnl = float(closed_pnl.nlargest(min(3, len(closed_pnl))).sum()) if len(closed_pnl) else 0.0
    summary.update({
        'Closed_PnL': total_pnl,
        'Profit_Factor': gross_profit / gross_loss if gross_loss > 0 else np.nan,
        'PnL_Ex_Top1': total_pnl - top1_pnl,
        'PnL_Ex_Top3': total_pnl - top3_pnl,
        'Top1_Profit_Share_pct': top1_pnl / total_pnl * 100.0 if total_pnl > 0 else np.nan,
        'Top3_Profit_Share_pct': top3_pnl / total_pnl * 100.0 if total_pnl > 0 else np.nan,
    })
    return curve, ledger, orders_df, summary


def prepare_mode_signals(all_results, mode):
    """把同一候选并集映射为原版或辅助版的可回测信号。"""
    if all_results is None or all_results.empty:
        return pd.DataFrame()
    work = all_results.copy()
    if mode == 'original':
        if 'Original_Selected' not in work.columns:
            return pd.DataFrame()
        work = work[bool_series(work['Original_Selected'])].copy()
        work['Rank'] = pd.to_numeric(work['Original_Rank'], errors='coerce')
        work['Selection_Mode'] = '原版量价排名'
    elif mode == 'auxiliary':
        if 'Auxiliary_Selected' not in work.columns:
            return pd.DataFrame()
        work = work[bool_series(work['Auxiliary_Selected'])].copy()
        work['Rank'] = pd.to_numeric(work['Auxiliary_Rank'], errors='coerce')
        work['Original_Total_Score'] = pd.to_numeric(work['Total_Score'], errors='coerce')
        work['Total_Score'] = pd.to_numeric(work['Auxiliary_Score'], errors='coerce')
        work['Selection_Mode'] = '辅助质量排名'
    else:
        raise ValueError(f'未知模式: {mode}')
    return work.sort_values(['Trade_Date', 'Rank']).reset_index(drop=True)


def portfolio_comparison_frame(summaries):
    rows = []
    for summary in summaries:
        rows.append({
            '模式': summary.get('Mode', ''),
            '期末权益': summary.get('Final_Equity', np.nan),
            '总收益(%)': summary.get('Total_Return_pct', np.nan),
            '最大回撤(%)': summary.get('Max_Drawdown_pct', np.nan),
            '空仓率(%)': summary.get('Empty_Ratio_pct', np.nan),
            '平均持仓': summary.get('Average_Positions', np.nan),
            '资金暴露(%)': summary.get('Average_Exposure_pct', np.nan),
            '成交数': summary.get('Executed_Entries', 0),
            '胜率(%)': summary.get('Closed_Win_Rate_pct', np.nan),
            '固定止损率(%)': summary.get('Fixed_Stop_Rate_pct', np.nan),
            'Profit Factor': summary.get('Profit_Factor', np.nan),
            '扣除最大1笔后利润': summary.get('PnL_Ex_Top1', np.nan),
            '扣除最大3笔后利润': summary.get('PnL_Ex_Top3', np.nan),
            '最大1笔利润占比(%)': summary.get('Top1_Profit_Share_pct', np.nan),
            '前三笔利润占比(%)': summary.get('Top3_Profit_Share_pct', np.nan),
        })
    return pd.DataFrame(rows)


def weekly_mode_summary(signals, mode_label):
    rows = []
    if signals is None or signals.empty:
        return pd.DataFrame()
    for week in range(1, 9):
        return_col = f'Return_W{week} (%)'
        eligible_col = f'Eligible_W{week}'
        held_col = f'Held_W{week}'
        if return_col not in signals.columns:
            continue
        if eligible_col in signals.columns:
            eligible_mask = bool_series(signals[eligible_col])
        else:
            eligible_mask = pd.to_numeric(signals[return_col], errors='coerce').notna()
        subset = signals[eligible_mask].copy()
        returns = pd.to_numeric(subset.get(return_col), errors='coerce').dropna()
        held = int(bool_series(subset[held_col]).sum()) if held_col in subset.columns else len(returns)
        rows.append({
            '模式': mode_label,
            '周期': f'W{week}',
            '完整样本': int(len(returns)),
            '仍在持仓': held,
            '生存率(%)': held / len(returns) * 100.0 if len(returns) else np.nan,
            '全样本均益(%)': float(returns.mean()) if len(returns) else np.nan,
            '全样本胜率(%)': float((returns > 0).mean() * 100.0) if len(returns) else np.nan,
        })
    return pd.DataFrame(rows)


def build_run_fingerprint(config):
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]

# ---------------------------
# UI 及 主程序
# ---------------------------
with st.sidebar:
    st.header("V40.6 辅助版1.0")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数 (开发建议100天)", min_value=1, value=100, step=1)
    
    TOP_BACKTEST = st.number_input("每日优选 TopK", min_value=1, value=3, step=1)
    EVALUATION_POOL_SIZE = st.number_input(
        "每种排序最多评估候选数", min_value=3, value=10, step=1,
        help="先完成D0双排序，再只计算两种排序前N名的候选并集；数值越小越快。",
    )
    AUXILIARY_SCORE_THRESHOLD = st.number_input(
        "A级辅助分门槛", min_value=0.0, max_value=100.0,
        value=float(AUXILIARY_SCORE_THRESHOLD), step=1.0,
    )
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    if st.button("🗑️ 清除行情缓存"):
        if os.path.exists(CACHE_FILE_NAME):
            os.remove(CACHE_FILE_NAME)
        if os.path.isdir(CACHE_DIR_NAME):
            shutil.rmtree(CACHE_DIR_NAME)
        st.success("行情、复权和市值逐日缓存已清除。")
    CHECKPOINT_DIR = "backtest_checkpoint_v40_6_auxiliary_1_0"
    if st.button("🗑️ 清除断点记录 (重新回测)"):
        if os.path.isdir(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
        st.success("所有辅助版1.0断点进度已清理！")
            
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
    st.caption("同时输出原版三仓、辅助评分三仓和辅助角色单仓，均从30万元起算。")

TS_TOKEN = st.text_input("Tushare Token", type="password")
if not TS_TOKEN: st.stop()
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

if st.button("🚀 启动 V40.6 辅助版1.0诊断"):
    SINA_STATUS = {'success': 0, 'fail': 0}
    trade_days_list = get_trade_days(backtest_date_end.strftime("%Y%m%d"), int(BACKTEST_DAYS))
    if not trade_days_list: st.stop()

    run_config = {
        'version': 'v40.6_auxiliary_1_0',
        'end_date': backtest_date_end.strftime('%Y%m%d'),
        'backtest_days': int(BACKTEST_DAYS),
        'top_k': int(TOP_BACKTEST),
        'evaluation_pool_size': int(EVALUATION_POOL_SIZE),
        'auxiliary_score_threshold': float(AUXILIARY_SCORE_THRESHOLD),
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
                date, int(TOP_BACKTEST), int(EVALUATION_POOL_SIZE), MIN_MV, MAX_MV, MIN_PRICE,
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

            # 与V40.4一致：交易日历中已列入、但尚未发布daily/daily_basic的
            # 末端日期仍记为已处理，该日不产生信号，不改变250日计划样本起点。
            completed_without_signal = {'无标的', '数据缺失', '市值数据缺失'}
            analysis_completed = (not res.empty) or (err in completed_without_signal)
            if analysis_completed:
                if not res.empty:
                    detail = f'评估候选{len(res)}只'
                elif err == '无标的':
                    detail = '无标的'
                else:
                    detail = f'数据缺失({err})'
                status_row = pd.DataFrame([{
                    'Trade_Date': date,
                    'Status': 'completed',
                    'Detail': detail,
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
    window_status = status_df[status_df['Trade_Date'].isin(set(trade_days_list))].copy()
    missing_signal_days = int(
        window_status['Detail'].astype(str).str.startswith('数据缺失').sum()
    ) if not window_status.empty else 0
    processed_window_days = len(set(window_status['Trade_Date']))
    st.info(
        f"回测起跑线：计划交易日 {len(trade_days_list)} 天｜"
        f"已处理 {processed_window_days} 天｜"
        f"有完整选股数据 {processed_window_days - missing_signal_days} 天｜"
        f"末端数据未发布 {missing_signal_days} 天。"
    )
    
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
        all_res = pd.concat(results, ignore_index=True)
        all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)

        original_signals = prepare_mode_signals(all_res, 'original')
        auxiliary_signals = prepare_mode_signals(all_res, 'auxiliary')

        original_curve, original_ledger, original_orders, original_summary = build_portfolio_backtest(
            original_signals,
            trade_days_list,
            initial_capital=INITIAL_CAPITAL,
            max_positions=MAX_PORTFOLIO_POSITIONS,
            position_budget=POSITION_BUDGET,
            lot_size=LOT_SIZE,
            mode_label='旧量价排序·三仓（同新版退出）',
        )
        auxiliary_three_curve, auxiliary_three_ledger, auxiliary_three_orders, auxiliary_three_summary = build_portfolio_backtest(
            auxiliary_signals,
            trade_days_list,
            initial_capital=INITIAL_CAPITAL,
            max_positions=MAX_PORTFOLIO_POSITIONS,
            position_budget=POSITION_BUDGET,
            lot_size=LOT_SIZE,
            mode_label='辅助质量排序·三仓诊断',
        )
        auxiliary_one_curve, auxiliary_one_ledger, auxiliary_one_orders, auxiliary_one_summary = build_portfolio_backtest(
            auxiliary_signals,
            trade_days_list,
            initial_capital=INITIAL_CAPITAL,
            max_positions=AUXILIARY_MAX_POSITIONS,
            position_budget=AUXILIARY_POSITION_BUDGET,
            lot_size=LOT_SIZE,
            mode_label='辅助质量排序·单仓角色',
        )

        summaries = [original_summary, auxiliary_three_summary, auxiliary_one_summary]
        comparison = portfolio_comparison_frame(summaries)
        st.header("📊 V40.6 辅助版1.0诊断结果")
        st.caption(
            "旧量价排序仅用于同批数据对照；它与辅助排序均使用辅助版1.0的真实成本保护。"
            "辅助单仓用于观察未来与V40.4共用三仓时的角色上限，不代表完整联合组合。"
        )
        st.subheader("💼 三种组合口径对照")
        st.dataframe(comparison.round(2), use_container_width=True, hide_index=True)

        all_curves = pd.concat(
            [frame for frame in [original_curve, auxiliary_three_curve, auxiliary_one_curve] if not frame.empty],
            ignore_index=True,
        ) if any(not frame.empty for frame in [original_curve, auxiliary_three_curve, auxiliary_one_curve]) else pd.DataFrame()
        all_ledgers = pd.concat(
            [frame for frame in [original_ledger, auxiliary_three_ledger, auxiliary_one_ledger] if not frame.empty],
            ignore_index=True,
        ) if any(not frame.empty for frame in [original_ledger, auxiliary_three_ledger, auxiliary_one_ledger]) else pd.DataFrame()
        all_orders = pd.concat(
            [frame for frame in [original_orders, auxiliary_three_orders, auxiliary_one_orders] if not frame.empty],
            ignore_index=True,
        ) if any(not frame.empty for frame in [original_orders, auxiliary_three_orders, auxiliary_one_orders]) else pd.DataFrame()

        if not all_curves.empty:
            equity_chart = all_curves[['Diagnostic_Mode', 'Trade_Date', 'Equity']].copy()
            equity_chart['Trade_Date'] = pd.to_datetime(equity_chart['Trade_Date'], format='%Y%m%d')
            equity_chart = equity_chart.pivot(index='Trade_Date', columns='Diagnostic_Mode', values='Equity')
            st.line_chart(equity_chart)

        st.subheader("🧪 A/B级候选质量")
        diagnostic_trades = all_res[bool_series(all_res.get('Tradable', pd.Series(False, index=all_res.index)))].copy()
        diagnostic_trades['_Return'] = pd.to_numeric(
            diagnostic_trades.get('Realized_Return (%)'), errors='coerce'
        )
        diagnostic_trades['_Fixed_Stop'] = diagnostic_trades.get(
            'Exit_Reason', pd.Series('', index=diagnostic_trades.index)
        ).astype(str).str.contains('固定止损')
        grade_rows = []
        for grade, group in diagnostic_trades.groupby('Auxiliary_Grade', dropna=False):
            returns = group['_Return'].dropna()
            grade_rows.append({
                '辅助等级': grade,
                '样本': len(group),
                '平均收益(%)': returns.mean() if len(returns) else np.nan,
                '中位收益(%)': returns.median() if len(returns) else np.nan,
                '胜率(%)': (returns > 0).mean() * 100.0 if len(returns) else np.nan,
                '固定止损率(%)': group['_Fixed_Stop'].mean() * 100.0 if len(group) else np.nan,
                '平均MFE(%)': pd.to_numeric(group.get('MFE_pct (%)'), errors='coerce').mean(),
                '平均MAE(%)': pd.to_numeric(group.get('MAE_pct (%)'), errors='coerce').mean(),
            })
        st.dataframe(pd.DataFrame(grade_rows).round(2), use_container_width=True, hide_index=True)

        st.subheader("🗓️ 全样本周度收益与真实持仓生存")
        weekly_comparison = pd.concat([
            weekly_mode_summary(original_signals, '旧量价排序'),
            weekly_mode_summary(auxiliary_signals, '辅助质量排序'),
        ], ignore_index=True)
        st.dataframe(weekly_comparison.round(2), use_container_width=True, hide_index=True)
        st.caption("提前退出的实现收益会延续到后续周；“仍在持仓”单独显示，因此不会再把退出者从胜率分母中删除。")

        st.subheader("📋 辅助候选诊断榜单")
        display_cols = [
            'Trade_Date', 'name', 'ts_code', 'market', 'SW_L1', 'SW_L2', 'SW_L3',
            'Original_Rank', 'Original_Selected', 'Total_Score',
            'Auxiliary_Candidate_Rank', 'Auxiliary_Rank', 'Auxiliary_Selected',
            'Auxiliary_Score', 'Auxiliary_Grade', 'Auxiliary_Grade_Reason',
            'Wave_Count', 'Day_Signal_Count', 'Sector_Signal_Count',
            'Day_Gain_pct', 'Vol_Ratio', 'MA20_Bias_pct', 'Breakout_Distance_pct',
            'Box_Width_10_pct', 'Prior_5D_Return_pct', 'ATR_pct', 'Close_Location_pct',
            'Turnover_Rate', 'Signal_Close', 'Entry_Date', 'Buy_Price', 'Gap_pct (%)',
            'MFE_pct (%)', 'MAE_pct (%)', 'Days_To_MFE10', 'False_Breakout_5D',
            'Protection_Trigger_Day', 'Protection_Stop_Price', 'Exit_Date',
            'Net_Exit_Price', 'Realized_Return (%)', 'Exit_Reason'
        ] + [f'Return_W{w} (%)' for w in range(1, 9)]
        final_cols = [c for c in display_cols if c in all_res.columns]

        display_df = all_res[final_cols].sort_values(
            ['Trade_Date', 'Auxiliary_Score'], ascending=[False, False]
        ).reset_index(drop=True)
        
        def color_exit(val):
            if isinstance(val, str):
                if '剔除' in val: return 'color: white; background-color: darkgray'
                elif '固定止损' in val: return 'color: white; background-color: darkred'
                elif '成本保护' in val: return 'color: teal'
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
        
        st.download_button(
            "📥 下载完整辅助诊断轨迹",
            all_res.to_csv(index=False).encode('utf-8-sig'),
            "signals_v40_6_auxiliary_1_0.csv",
            "text/csv",
        )

        download_cols = st.columns(4)
        download_cols[0].download_button(
            "📥 下载三模式组合曲线",
            all_curves.to_csv(index=False).encode('utf-8-sig'),
            "portfolio_curve_v40_6_auxiliary_1_0.csv",
            "text/csv",
        )
        download_cols[1].download_button(
            "📥 下载三模式成交账本",
            all_ledgers.to_csv(index=False).encode('utf-8-sig'),
            "portfolio_ledger_v40_6_auxiliary_1_0.csv",
            "text/csv",
        )
        download_cols[2].download_button(
            "📥 下载三模式买入审计",
            all_orders.to_csv(index=False).encode('utf-8-sig'),
            "portfolio_orders_v40_6_auxiliary_1_0.csv",
            "text/csv",
        )
        download_cols[3].download_button(
            "📥 下载模式汇总",
            comparison.to_csv(index=False).encode('utf-8-sig'),
            "portfolio_summary_v40_6_auxiliary_1_0.csv",
            "text/csv",
        )

        if not all_ledgers.empty:
            st.subheader("🧾 三模式真实组合成交账本")
            st.dataframe(all_ledgers, use_container_width=True)
    else:
        st.warning("⚠️ 暂无符合条件的标的。")
