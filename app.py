# -*- coding: utf-8 -*-
"""
选股王 · V40.6 实战定型版 (四大神盾)
------------------------------------------------
核心改进 (基于数据复盘后的终极定型):
1. [硬门槛 1：盘子基座] 侧边栏默认流通市值提高至 250 亿，彻底隔绝微盘股的画线诱多陷阱。
2. [硬门槛 2：温和爆破] 突破量比上限严格锁定在 3.0倍 (1.3 <= vol <= 3.0)，绞杀“天量见天价”的分歧坑。
3. [硬门槛 3：开盘定生死] 在 T+1 买入引擎中加入集合竞价拦截器。若高开>5%或低开<-3%，直接放弃买入，剔除该标的！
4. [废除主观加分] 尊重客观数据，剔除原有的“洗盘2-3次加分”逻辑，所有分数纯靠量价真实动能。
------------------------------------------------
稳定性修复（不改变任何选股、排序、买入、卖出和收益计算规则）:
1. 行情、复权和市值数据按交易日原子分片保存；中断后只补缺失日期。
2. 结果、扫描账本和任务状态原子保存；页面刷新或网络重连后从断点续跑。
3. 无标的日期同样记入扫描账本；重跑某日时覆盖旧快照，不重复追加。
4. 加入任务锁、Token预检、临时异常重试和磁盘报告恢复。
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
import shutil
import tempfile
import hashlib
import json
import re

try:
    import fcntl
except ImportError:  # Windows 本地运行时使用后备锁
    fcntl = None

warnings.filterwarnings("ignore")

APP_VERSION = "V40.6-S1"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_CACHE_BASENAME = "market_data_cache_v40_6.pkl"
MARKET_CACHE_ROOT = os.path.join(APP_DIR, "market_data_cache_v40_6_stable")
CHECKPOINT_FILE = os.path.join(APP_DIR, "backtest_v40_6_stable_history.csv")
SCAN_LEDGER_FILE = os.path.join(APP_DIR, "backtest_v40_6_stable_scanned_dates.csv")
RUN_TASK_FILE = os.path.join(APP_DIR, "backtest_v40_6_stable_running_task.json")
RUN_LOCK_FILE = os.path.join(APP_DIR, "backtest_v40_6_stable_running.lock")
TUSHARE_REQUEST_TIMEOUT_SECONDS = 20
DAYS_PER_BATCH = 5

# ---------------------------
# 全局变量与探针
# ---------------------------
pro = None 
GLOBAL_ADJ_FACTOR = pd.DataFrame() 
GLOBAL_DAILY_RAW = pd.DataFrame() 
GLOBAL_DAILY_BASIC = pd.DataFrame()
GLOBAL_QFQ_BASE_FACTORS = {} 
GLOBAL_STOCK_INDUSTRY = {} 
GLOBAL_STOCK_BASIC = pd.DataFrame()
SINA_STATUS = {'success': 0, 'fail': 0} 
API_ERRORS = []
_RUN_LOCK_HANDLE = None

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王 V40.6-S1 稳定修复版", layout="wide")
st.title("选股王 V40.6：箱体首发 + 四大神盾")
st.caption("V40.6-S1 仅修复缓存、断点续跑和页面崩溃问题；选股与买卖规则保持不变。")

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
def clean_token_str(raw_token):
    if not raw_token:
        return ""
    return re.sub(r'[\s\u3000\ufeff\xa0\r\n]+', '', str(raw_token)).strip()


def record_api_error(message):
    if len(API_ERRORS) < 300:
        API_ERRORS.append(str(message))


def safe_get(func_name, required=False, max_retries=3, sleep_time=0.8, **kwargs):
    """失败结果不进入Streamlit缓存，避免一次空返回把当天锁死12小时。"""
    global pro
    if pro is None:
        message = f"{func_name}: Tushare尚未初始化"
        record_api_error(message)
        if required:
            raise RuntimeError(message)
        return pd.DataFrame()
    try:
        func = getattr(pro, func_name)
    except Exception as exc:
        message = f"当前Tushare SDK不支持接口 {func_name}: {exc}"
        record_api_error(message)
        if required:
            raise RuntimeError(message) from exc
        return pd.DataFrame()

    last_error = None
    for attempt in range(max_retries):
        try:
            df = func(**kwargs)
            if df is not None and not df.empty:
                return df
            last_error = RuntimeError("接口返回空数据")
        except Exception as exc:
            last_error = exc
        time.sleep(sleep_time * (attempt + 1))

    message = f"{func_name}({kwargs})失败: {last_error}"
    record_api_error(message)
    if required:
        raise RuntimeError(message) from last_error
    return pd.DataFrame()


def verify_token_connection(token_str):
    if not token_str:
        return False, "Token为空，请填写Tushare Token。"
    try:
        ts.set_token(token_str)
        test_pro = ts.pro_api(token_str)
        setattr(test_pro, "_DataApi__timeout", TUSHARE_REQUEST_TIMEOUT_SECONDS)
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
        test_df = test_pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
        if test_df is not None and not test_df.empty:
            return True, "验证通过"
        return False, "Token校验未返回数据，请检查网络连接。"
    except Exception as exc:
        message = str(exc)
        if "token不对" in message or "-40001" in message:
            return False, "Token不正确，请检查复制内容。"
        return False, f"接口校验失败: {message}"


def parse_yyyymmdd(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\.0$", "", str(value)).replace("-", "")
    return text if re.fullmatch(r"\d{8}", text) else None


def atomic_write_csv(df, path):
    """先写临时文件再原子替换，并保留上一份可恢复备份。"""
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        with open(tmp_path, "rb") as file_obj:
            os.fsync(file_obj.fileno())
        if os.path.exists(path):
            try:
                shutil.copy2(path, path + ".bak")
            except OSError:
                pass
        elif os.path.exists(path + ".bak"):
            os.remove(path + ".bak")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_csv_safe(path):
    for candidate in (path, path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            return pd.read_csv(candidate, encoding="utf-8-sig", low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError):
            continue
    return pd.DataFrame()


def atomic_write_json(value, path):
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        if os.path.exists(path):
            try:
                shutil.copy2(path, path + ".bak")
            except OSError:
                pass
        elif os.path.exists(path + ".bak"):
            os.remove(path + ".bak")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_json_safe(path):
    for candidate in (path, path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as file_obj:
                value = json.load(file_obj)
            if isinstance(value, dict):
                return value
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return {}


def remove_with_backup(path):
    for candidate in (path, path + ".bak"):
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except OSError:
            pass


def atomic_write_pickle(value, path):
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as file_obj:
            pickle.dump(value, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def acquire_run_lock(stale_seconds=600):
    global _RUN_LOCK_HANDLE
    if fcntl is not None:
        try:
            lock_handle = open(RUN_LOCK_FILE, "a+", encoding="utf-8")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_handle.seek(0)
            lock_handle.truncate()
            lock_handle.write(str(time.time()))
            lock_handle.flush()
            _RUN_LOCK_HANDLE = lock_handle
            return True
        except (OSError, BlockingIOError):
            try:
                lock_handle.close()
            except Exception:
                pass
            return False

    if os.path.exists(RUN_LOCK_FILE):
        try:
            if time.time() - os.path.getmtime(RUN_LOCK_FILE) > stale_seconds:
                os.remove(RUN_LOCK_FILE)
        except OSError:
            pass
    try:
        fd = os.open(RUN_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(time.time()).encode("utf-8"))
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_run_lock():
    global _RUN_LOCK_HANDLE
    try:
        if _RUN_LOCK_HANDLE is not None:
            if fcntl is not None:
                fcntl.flock(_RUN_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
            _RUN_LOCK_HANDLE.close()
            _RUN_LOCK_HANDLE = None
            return
        if fcntl is None and os.path.exists(RUN_LOCK_FILE):
            os.remove(RUN_LOCK_FILE)
    except OSError:
        pass

def get_trade_days(end_date_str, num_days):
    lookback_days = max(num_days * 3, 365) 
    start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cal = safe_get('trade_cal', required=True, exchange='SSE', start_date=start_date, end_date=end_date_str)
    trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str]
    return trade_days_df['cal_date'].head(num_days).tolist()

@st.cache_data(ttl=3600*24*7, show_spinner=False)
def load_industry_mapping(token_hash):
    del token_hash
    global pro
    if pro is None:
        raise RuntimeError("Tushare尚未初始化，无法构建科技白名单")
    try:
        sw_indices = safe_get('index_classify', required=True, level='L1', src='SW2021')
        white_list_names = ['电子', '计算机', '通信', '医药生物', '国防军工', '机械设备']
        target_indices = sw_indices[sw_indices['industry_name'].isin(white_list_names)]
        index_codes = target_indices['index_code'].tolist()
        if not index_codes:
            raise RuntimeError("未取得V40.6原科技行业目录")
        
        all_members = []
        load_bar = st.progress(0, text="正在加载硬科技白名单赛道数据...")
        for i, idx_code in enumerate(index_codes):
            df = safe_get('index_member', required=True, index_code=idx_code, is_new='Y')
            if not df.empty: 
                df['industry_code'] = idx_code
                all_members.append(df)
            time.sleep(0.05) 
            load_bar.progress((i + 1) / len(index_codes))
        load_bar.empty()
        
        if not all_members:
            raise RuntimeError("未取得V40.6原科技白名单成分")
        full_df = pd.concat(all_members).drop_duplicates(subset=['con_code'])
        return dict(zip(full_df['con_code'], full_df['industry_code']))
    except Exception:
        try:
            load_bar.empty()
        except Exception:
            pass
        raise


@st.cache_data(ttl=3600*24, show_spinner=False)
def load_stock_basic(token_hash):
    del token_hash
    stock_basic = safe_get('stock_basic', required=True, list_status='L', fields='ts_code,name')
    if stock_basic.empty:
        raise RuntimeError("stock_basic加载失败")
    return stock_basic.drop_duplicates('ts_code', keep='last').reset_index(drop=True)

# ---------------------------
# 数据获取与复权引擎
# ---------------------------
def pool_cache_dir(whitelist_set):
    pool_hash = hashlib.sha1(
        "|".join(sorted(whitelist_set)).encode('utf-8')
    ).hexdigest()[:12]
    cache_dir = os.path.join(MARKET_CACHE_ROOT, pool_hash)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir, pool_hash


def market_partition_path(trade_date, cache_dir):
    return os.path.join(cache_dir, f"market_{trade_date}.pkl")


def valid_market_partition(payload, trade_date, pool_hash, require_basic=False):
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != 2 or str(payload.get("trade_date")) != str(trade_date):
        return False
    if str(payload.get('pool_hash')) != str(pool_hash):
        return False
    daily = payload.get("daily")
    adj = payload.get("adj")
    basic = payload.get("daily_basic")
    if not isinstance(daily, pd.DataFrame) or not isinstance(adj, pd.DataFrame):
        return False
    if int(payload.get("raw_daily_count", 0)) < 1000 or int(payload.get("raw_adj_count", 0)) < 1000:
        return False
    required_daily = {"ts_code", "trade_date", "open", "high", "low", "close", "vol"}
    if daily.empty or adj.empty:
        return False
    if not required_daily.issubset(daily.columns):
        return False
    if not {"ts_code", "trade_date", "adj_factor"}.issubset(adj.columns):
        return False
    if not daily["trade_date"].astype(str).eq(str(trade_date)).any():
        return False
    if not adj["trade_date"].astype(str).eq(str(trade_date)).any():
        return False
    if require_basic:
        if not isinstance(basic, pd.DataFrame) or basic.empty:
            return False
        if int(payload.get("raw_basic_count", 0)) < 1000:
            return False
        if not {"ts_code", "trade_date", "circ_mv"}.issubset(basic.columns):
            return False
    return True


def read_market_partition(trade_date, cache_dir, pool_hash, require_basic=False):
    path = market_partition_path(trade_date, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as file_obj:
            payload = pickle.load(file_obj)
        return payload if valid_market_partition(payload, trade_date, pool_hash, require_basic) else None
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        return None


def legacy_cache_paths():
    paths = [
        os.path.join(APP_DIR, LEGACY_CACHE_BASENAME),
        os.path.join(os.getcwd(), LEGACY_CACHE_BASENAME),
    ]
    return list(dict.fromkeys(os.path.abspath(path) for path in paths))


def load_legacy_cache_index():
    """复用V40.6旧整包缓存；只补市值数据后迁移为可靠分片。"""
    for path in legacy_cache_paths():
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as file_obj:
                cached = pickle.load(file_obj)
            daily = cached.get("daily", pd.DataFrame())
            adj = cached.get("adj", pd.DataFrame())
            if daily.empty or adj.empty:
                continue
            if not isinstance(daily.index, pd.MultiIndex):
                daily = daily.drop_duplicates(['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date'])
            if not isinstance(adj.index, pd.MultiIndex):
                adj = adj.drop_duplicates(['ts_code', 'trade_date']).set_index(['ts_code', 'trade_date'])
            daily = daily.reorder_levels(['trade_date', 'ts_code']).sort_index()
            adj = adj.reorder_levels(['trade_date', 'ts_code']).sort_index()
            legacy_dates = sorted(
                set(daily.index.get_level_values('trade_date').astype(str))
                & set(adj.index.get_level_values('trade_date').astype(str))
            )
            return daily, adj, legacy_dates
        except Exception as exc:
            record_api_error(f"旧行情缓存读取失败，改用分片下载: {exc}")
    return pd.DataFrame(), pd.DataFrame(), []


def extract_legacy_day(frame, trade_date):
    if frame is None or frame.empty:
        return pd.DataFrame()
    try:
        part = frame.loc[str(trade_date)].reset_index()
        part['trade_date'] = str(trade_date)
        return part
    except (KeyError, TypeError, ValueError):
        return pd.DataFrame()


def sync_market_data_incrementally(start_date, end_date, basic_required_dates, whitelist_set):
    """每成功一个交易日立即原子保存；失败日期不写完成分片。"""
    calendar = safe_get(
        'trade_cal', required=True, exchange='SSE', start_date=start_date, end_date=end_date
    )
    today_str = datetime.now().strftime("%Y%m%d")
    valid_dates = (
        calendar[(calendar['is_open'] == 1) & (calendar['cal_date'].astype(str) <= today_str)]
        .sort_values('cal_date')['cal_date'].astype(str).tolist()
    )
    required_basic = {str(date) for date in basic_required_dates}
    if len(required_basic) == 1:
        anchor = next(iter(required_basic))
        prior_dates = [date for date in valid_dates if date <= anchor][-10:]
        required_basic.update(prior_dates)

    cache_dir, pool_hash = pool_cache_dir(whitelist_set)
    missing_dates = [
        date for date in valid_dates
        if read_market_partition(
            date, cache_dir, pool_hash, require_basic=date in required_basic
        ) is None
    ]
    complete_hits = len(valid_dates) - len(missing_dates)
    st.caption(
        f"行情缓存：完整命中 {complete_hits}/{len(valid_dates)} 个交易日；"
        f"待补 {len(missing_dates)} 个。"
    )

    legacy_daily = pd.DataFrame()
    legacy_adj = pd.DataFrame()
    base_missing_dates = [
        date for date in missing_dates
        if read_market_partition(date, cache_dir, pool_hash, require_basic=False) is None
    ]
    legacy_marker_path = os.path.join(cache_dir, 'legacy_cache_dates.json')
    legacy_marker = read_json_safe(legacy_marker_path)
    known_legacy_dates = set(legacy_marker.get('dates', []))
    should_open_legacy = bool(base_missing_dates) and (
        not legacy_marker or bool(set(base_missing_dates) & known_legacy_dates)
    )
    if should_open_legacy:
        legacy_daily, legacy_adj, legacy_dates = load_legacy_cache_index()
        known_legacy_dates = set(legacy_dates)
        atomic_write_json({'dates': sorted(known_legacy_dates)}, legacy_marker_path)

    failed_dates = []
    migrated_dates = 0
    if missing_dates:
        bar = st.progress(0, text=f"从断点补充 {len(missing_dates)} 个交易日行情...")
        for idx, trade_date in enumerate(missing_dates):
            existing = read_market_partition(
                trade_date, cache_dir, pool_hash, require_basic=False
            )
            daily = existing.get('daily', pd.DataFrame()) if existing else pd.DataFrame()
            adj = existing.get('adj', pd.DataFrame()) if existing else pd.DataFrame()
            basic = existing.get('daily_basic', pd.DataFrame()) if existing else pd.DataFrame()
            raw_daily_count = int(existing.get('raw_daily_count', 0)) if existing else 0
            raw_adj_count = int(existing.get('raw_adj_count', 0)) if existing else 0
            raw_basic_count = int(existing.get('raw_basic_count', 0)) if existing else 0

            used_legacy = False
            if daily.empty:
                daily_all = extract_legacy_day(legacy_daily, trade_date)
                raw_daily_count = len(daily_all)
                daily = daily_all[daily_all['ts_code'].isin(whitelist_set)].copy() if not daily_all.empty else pd.DataFrame()
                used_legacy = not daily.empty
            if adj.empty:
                adj_all = extract_legacy_day(legacy_adj, trade_date)
                raw_adj_count = len(adj_all)
                adj = adj_all[adj_all['ts_code'].isin(whitelist_set)].copy() if not adj_all.empty else pd.DataFrame()
                used_legacy = used_legacy or not adj.empty
            if daily.empty:
                daily_all = safe_get('daily', trade_date=trade_date)
                raw_daily_count = len(daily_all)
                daily = daily_all[daily_all['ts_code'].isin(whitelist_set)].copy() if not daily_all.empty else pd.DataFrame()
            if adj.empty:
                adj_all = safe_get('adj_factor', trade_date=trade_date)
                raw_adj_count = len(adj_all)
                adj = adj_all[adj_all['ts_code'].isin(whitelist_set)].copy() if not adj_all.empty else pd.DataFrame()
            if trade_date in required_basic and basic.empty:
                basic_all = safe_get(
                    'daily_basic', trade_date=trade_date,
                    fields='ts_code,trade_date,circ_mv'
                )
                raw_basic_count = len(basic_all)
                basic = basic_all[basic_all['ts_code'].isin(whitelist_set)].copy() if not basic_all.empty else pd.DataFrame()

            payload = {
                'version': 2,
                'trade_date': trade_date,
                'pool_hash': pool_hash,
                'saved_at': datetime.now().isoformat(timespec='seconds'),
                'raw_daily_count': int(raw_daily_count),
                'raw_adj_count': int(raw_adj_count),
                'raw_basic_count': int(raw_basic_count),
                'daily': daily,
                'adj': adj,
                'daily_basic': basic,
            }
            base_complete = valid_market_partition(
                payload, trade_date, pool_hash, require_basic=False
            )
            fully_complete = valid_market_partition(
                payload, trade_date, pool_hash, require_basic=trade_date in required_basic
            )
            if base_complete:
                atomic_write_pickle(payload, market_partition_path(trade_date, cache_dir))
                migrated_dates += int(used_legacy)
            if not fully_complete:
                failed_dates.append(trade_date)

            if (idx + 1) % 5 == 0 or idx == len(missing_dates) - 1:
                bar.progress(
                    (idx + 1) / len(missing_dates),
                    text=(
                        f"行情同步 {idx+1}/{len(missing_dates)}；"
                        f"成功 {idx+1-len(failed_dates)}，失败 {len(failed_dates)}"
                    ),
                )
            time.sleep(0.12)
        bar.empty()

    if failed_dates:
        st.warning(
            f"有 {len(failed_dates)} 个交易日行情暂未完整返回；示例: {failed_dates[:8]}。"
            "成功日期已保存，本次继续回测完整日期，下次只补失败日期。"
        )
    if migrated_dates:
        st.caption(f"已从旧版整包缓存迁移 {migrated_dates} 个交易日，避免重复下载行情和复权。")
    return valid_dates, failed_dates, cache_dir, pool_hash


@st.cache_resource(ttl=3600*12, show_spinner=False)
def build_market_index_from_partitions(valid_dates_key, cache_dir, pool_hash, cache_stamp):
    del cache_stamp
    daily_parts, adj_parts, basic_parts = [], [], []
    complete_dates = []
    for trade_date in valid_dates_key:
        payload = read_market_partition(
            trade_date, cache_dir, pool_hash, require_basic=False
        )
        if payload is None:
            continue
        daily_parts.append(payload['daily'])
        adj_parts.append(payload['adj'])
        if isinstance(payload.get('daily_basic'), pd.DataFrame) and not payload['daily_basic'].empty:
            basic_parts.append(payload['daily_basic'])
        complete_dates.append(trade_date)

    if not daily_parts or not adj_parts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, tuple()

    daily_raw = pd.concat(daily_parts, ignore_index=True)
    adj_raw = pd.concat(adj_parts, ignore_index=True)
    basic_raw = pd.concat(basic_parts, ignore_index=True) if basic_parts else pd.DataFrame()

    daily_raw['trade_date'] = daily_raw['trade_date'].astype(str)
    adj_raw['trade_date'] = adj_raw['trade_date'].astype(str)
    adj_raw['adj_factor'] = pd.to_numeric(adj_raw['adj_factor'], errors='coerce')
    adj_raw = adj_raw.dropna(subset=['adj_factor'])
    daily_index = (
        daily_raw.drop_duplicates(['ts_code', 'trade_date'], keep='last')
        .set_index(['ts_code', 'trade_date']).sort_index()
    )
    adj_index = (
        adj_raw.drop_duplicates(['ts_code', 'trade_date'], keep='last')
        .set_index(['ts_code', 'trade_date']).sort_index()
    )
    if not basic_raw.empty:
        basic_raw['trade_date'] = basic_raw['trade_date'].astype(str)
        basic_index = (
            basic_raw.drop_duplicates(['ts_code', 'trade_date'], keep='last')
            .set_index(['ts_code', 'trade_date']).sort_index()
        )
    else:
        basic_index = pd.DataFrame()

    adj_latest = adj_index.reset_index().sort_values(['ts_code', 'trade_date'])
    base_factors = (
        adj_latest.groupby('ts_code', sort=False).tail(1)
        .set_index('ts_code')['adj_factor'].to_dict()
    )
    return daily_index, adj_index, basic_index, base_factors, tuple(complete_dates)


def get_all_historical_data(trade_days_list, use_cache=True):
    del use_cache  # 稳定版始终使用可恢复分片；清除按钮可显式全量重建。
    global GLOBAL_ADJ_FACTOR, GLOBAL_DAILY_RAW, GLOBAL_DAILY_BASIC, GLOBAL_QFQ_BASE_FACTORS
    if not trade_days_list:
        return {'ok': False, 'failed_dates': [], 'complete_dates': []}

    latest_trade_date = max(trade_days_list)
    earliest_trade_date = min(trade_days_list)
    start_date = (
        datetime.strptime(earliest_trade_date, "%Y%m%d") - timedelta(days=365)
    ).strftime("%Y%m%d")
    theoretical_end = datetime.strptime(latest_trade_date, "%Y%m%d") + timedelta(days=150)
    end_date = min(theoretical_end, datetime.now()).strftime("%Y%m%d")

    whitelist_set = set(GLOBAL_STOCK_INDUSTRY)
    if not whitelist_set:
        return {'ok': False, 'failed_dates': [], 'complete_dates': []}
    valid_dates, failed_dates, cache_dir, pool_hash = sync_market_data_incrementally(
        start_date, end_date, set(trade_days_list), whitelist_set
    )
    shard_paths = [
        market_partition_path(date, cache_dir) for date in valid_dates
        if os.path.exists(market_partition_path(date, cache_dir))
    ]
    cache_stamp = (
        len(shard_paths),
        max((os.path.getmtime(path) for path in shard_paths), default=0.0),
    )
    (
        GLOBAL_DAILY_RAW,
        GLOBAL_ADJ_FACTOR,
        GLOBAL_DAILY_BASIC,
        GLOBAL_QFQ_BASE_FACTORS,
        complete_dates,
    ) = build_market_index_from_partitions(
        tuple(valid_dates), cache_dir, pool_hash, cache_stamp
    )
    ok = not GLOBAL_DAILY_RAW.empty and not GLOBAL_ADJ_FACTOR.empty
    return {'ok': ok, 'failed_dates': failed_dates, 'complete_dates': list(complete_dates)}


def get_market_slice(frame, trade_date):
    if frame is None or frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return pd.DataFrame()
    try:
        return frame.xs(str(trade_date), level='trade_date').reset_index()
    except (KeyError, TypeError, ValueError):
        return pd.DataFrame()

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
def get_medium_term_future(ts_code, selection_date, signal_close, bottom_line, hold_weeks=8, use_sina=False):
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    start_fetch = (d0 - timedelta(days=60)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=150)).strftime("%Y%m%d") 
    
    hist_full = get_qfq_data_v4_optimized_final(ts_code, start_date=start_fetch, end_date=end_future, use_sina=use_sina)
    results = {f'Return_W{w} (%)': np.nan for w in range(1, hold_weeks + 1)}
    results['Exit_Reason'] = "持仓中"
    results['Buy_Price'] = np.nan
    results['Gap_pct (%)'] = np.nan
    
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

    buy_price = next_row['open']
    if pd.isna(buy_price) or buy_price <= 0:
        return results

    # 【改动3：T+1 集合竞价拦截器】防核按钮与高开诱多
    if signal_close and signal_close > 0:
        gap_pct = (buy_price - signal_close) / signal_close * 100
        results['Gap_pct (%)'] = round(gap_pct, 2)
        if gap_pct < -3.0 or gap_pct > 5.0:
            results['Exit_Reason'] = f"开盘幅度不符(剔除: {round(gap_pct, 2)}%)"
            results['Buy_Price'] = round(buy_price, 2)
            # 直接返回，不再执行后续持仓运算
            return results

    results['Buy_Price'] = round(buy_price, 2)

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
            final_return = (curr_open - buy_price) / buy_price * 100.0
            exit_triggered = True
            results['Exit_Reason'] = pending_exit_reason
            results[f'Return_W{current_week} (%)'] = final_return
            break
        
        peak_close = max(peak_close, curr_high)
        peak_profit_pct = (peak_close - buy_price) / buy_price
        
        if (curr_low - buy_price) / buy_price <= hard_stop_limit:
            final_return = min(hard_stop_limit * 100, (curr_open - buy_price) / buy_price * 100)
            exit_triggered = True
            results['Exit_Reason'] = f"固定止损(破{int(hard_stop_limit*100)}%)"
            results[f'Return_W{current_week} (%)'] = final_return
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
            results[f'Return_W{current_week} (%)'] = (curr_close - buy_price) / buy_price * 100.0
            
    if not exit_triggered and len(hist_future) >= hold_weeks * 5:
        last_price = hist_future.iloc[hold_weeks * 5 - 1]['close']
        results[f'Return_W{hold_weeks} (%)'] = (last_price - buy_price) / buy_price * 100.0
        results['Exit_Reason'] = "周期结束平仓"
        
    return results

# ---------------------------
# 核心回测循环
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, MIN_MV, MAX_MV, MIN_PRICE, use_sina=False, run_timestamp=None):
    global GLOBAL_STOCK_INDUSTRY, GLOBAL_STOCK_BASIC

    query_date = last_trade
    daily_all = get_market_slice(GLOBAL_DAILY_RAW, query_date)
    
    if use_sina and daily_all.empty:
        for i in range(1, 10):
            temp_date = (datetime.strptime(last_trade, "%Y%m%d") - timedelta(days=i)).strftime("%Y%m%d")
            daily_all = get_market_slice(GLOBAL_DAILY_RAW, temp_date)
            daily_basic_test = get_market_slice(GLOBAL_DAILY_BASIC, temp_date)
            if not daily_all.empty and not daily_basic_test.empty:
                query_date = temp_date
                break
                
    if daily_all.empty:
        return pd.DataFrame(), "数据缺失", 0

    if GLOBAL_STOCK_BASIC.empty:
        return pd.DataFrame(), "股票基础数据缺失", 0
    df = daily_all.merge(GLOBAL_STOCK_BASIC, on='ts_code', how='left')
    
    daily_basic = get_market_slice(GLOBAL_DAILY_BASIC, query_date)
    if not daily_basic.empty:
        df = df.merge(daily_basic[['ts_code','circ_mv']], on='ts_code', how='left')
    else: 
        return pd.DataFrame(), "市值数据缺失", 0
    
    df['circ_mv_billion'] = df['circ_mv'] / 10000 
    
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[~df['ts_code'].str.startswith('92')] 
    
    df = df[(df['close'] >= MIN_PRICE)]
    df = df[(df['circ_mv_billion'] >= MIN_MV) & (df['circ_mv_billion'] <= MAX_MV)]
    
    records = []
    for row in df.itertuples():
        if GLOBAL_STOCK_INDUSTRY and row.ts_code not in GLOBAL_STOCK_INDUSTRY: 
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
            
        future_returns = get_medium_term_future(row.ts_code, last_trade, ind['last_close'], ind['bottom_line'], hold_weeks=8, use_sina=use_sina)
        
        record_dict = {
            'ts_code': row.ts_code, 'name': row.name, 'Signal_Close': ind['last_close'], 
            'Wave_Count': wave_cnt,
            'circ_mv': row.circ_mv_billion,
            'Total_Score': round(total_score, 1),
            'Breakout_S': round(score_breakout, 1),
            'Volume_S': round(score_vol, 1)
        }
        record_dict.update(future_returns)
        records.append(record_dict)
            
    if not records:
        return pd.DataFrame(), "无标的", 0
    
    fdf = pd.DataFrame(records)
    final_df = fdf.sort_values('Total_Score', ascending=False).head(TOP_BACKTEST).copy()
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    return final_df, None, len(records)


# ---------------------------
# 稳定断点与任务状态（不参与策略计算）
# ---------------------------
def make_config_id(top_k, min_mv, max_mv, min_price):
    payload = {
        'strategy': 'V40.6-original',
        'top_k': int(top_k),
        'min_mv': float(min_mv),
        'max_mv': float(max_mv),
        'min_price': float(min_price),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]


def canonicalize_history(df):
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    clean = df.copy()
    clean = clean.drop(columns=[c for c in clean.columns if str(c).startswith('Unnamed:')], errors='ignore')
    if 'Trade_Date' in clean.columns:
        clean['Trade_Date'] = clean['Trade_Date'].map(parse_yyyymmdd)
        clean = clean.dropna(subset=['Trade_Date'])
    if 'Config_ID' in clean.columns:
        clean['Config_ID'] = clean['Config_ID'].astype(str)
    if 'ts_code' in clean.columns:
        clean['ts_code'] = clean['ts_code'].astype(str).str.strip()
    keys = [key for key in ('Config_ID', 'Trade_Date', 'ts_code') if key in clean.columns]
    if keys:
        clean = clean.drop_duplicates(keys, keep='last')
    return clean.reset_index(drop=True)


def replace_checkpoint_date(new_rows, trade_date, config_id):
    existing = canonicalize_history(read_csv_safe(CHECKPOINT_FILE))
    if not existing.empty and {'Trade_Date', 'Config_ID'}.issubset(existing.columns):
        keep = ~(
            existing['Trade_Date'].astype(str).eq(str(trade_date))
            & existing['Config_ID'].astype(str).eq(str(config_id))
        )
        existing = existing.loc[keep].copy()
    if new_rows is not None and not new_rows.empty:
        rows = new_rows.copy()
        rows['Trade_Date'] = str(trade_date)
        rows['Config_ID'] = str(config_id)
        combined = pd.concat([existing, rows], ignore_index=True, sort=False) if not existing.empty else rows
    else:
        combined = existing
    combined = canonicalize_history(combined)
    sort_cols = [column for column in ('Trade_Date', 'Rank') if column in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols, kind='mergesort')
    atomic_write_csv(combined.reset_index(drop=True), CHECKPOINT_FILE)


def mark_scan_status(trade_date, raw_count, selected_count, config_id, status, error=''):
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    row = pd.DataFrame([{
        'Trade_Date': str(trade_date),
        'Raw_Signal_Count': int(raw_count),
        'Selected_Count': int(selected_count),
        'Scan_Status': str(status),
        'Error': str(error),
        'Config_ID': str(config_id),
        'Updated_At': datetime.now().isoformat(timespec='seconds'),
    }])
    ledger = pd.concat([ledger, row], ignore_index=True, sort=False) if not ledger.empty else row
    ledger['Trade_Date'] = ledger['Trade_Date'].map(parse_yyyymmdd)
    ledger = ledger.dropna(subset=['Trade_Date'])
    ledger['Config_ID'] = ledger['Config_ID'].astype(str)
    ledger = ledger.drop_duplicates(['Trade_Date', 'Config_ID'], keep='last')
    atomic_write_csv(ledger.sort_values('Trade_Date').reset_index(drop=True), SCAN_LEDGER_FILE)


def completed_scan_dates(config_id):
    completed = set()
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    if not ledger.empty and {'Trade_Date', 'Config_ID', 'Scan_Status'}.issubset(ledger.columns):
        match = ledger[
            ledger['Config_ID'].astype(str).eq(str(config_id))
            & ledger['Scan_Status'].astype(str).eq('COMPLETED')
        ]
        completed.update(filter(None, (parse_yyyymmdd(value) for value in match['Trade_Date'])))

    # 若恰好在结果写入与账本写入之间退出，有结果的日期仍可恢复为完成状态。
    history = canonicalize_history(read_csv_safe(CHECKPOINT_FILE))
    if not history.empty and 'Trade_Date' in history.columns:
        if 'Config_ID' in history.columns:
            history = history[history['Config_ID'].astype(str).eq(str(config_id))]
        completed.update(filter(None, (parse_yyyymmdd(value) for value in history['Trade_Date'])))
    return completed


def save_task(task):
    value = dict(task)
    value['Updated_At'] = datetime.now().isoformat(timespec='seconds')
    atomic_write_json(value, RUN_TASK_FILE)


def clear_all_market_caches():
    removed = []
    if os.path.isdir(MARKET_CACHE_ROOT):
        shutil.rmtree(MARKET_CACHE_ROOT)
        removed.append(MARKET_CACHE_ROOT)
    for path in legacy_cache_paths():
        if os.path.isfile(path):
            os.remove(path)
            removed.append(path)
    st.cache_resource.clear()
    return removed


def clear_config_records(config_id):
    history = canonicalize_history(read_csv_safe(CHECKPOINT_FILE))
    if not history.empty and 'Config_ID' in history.columns:
        history = history[~history['Config_ID'].astype(str).eq(str(config_id))]
        atomic_write_csv(history.reset_index(drop=True), CHECKPOINT_FILE)
    ledger = read_csv_safe(SCAN_LEDGER_FILE)
    if not ledger.empty and 'Config_ID' in ledger.columns:
        ledger = ledger[~ledger['Config_ID'].astype(str).eq(str(config_id))]
        atomic_write_csv(ledger.reset_index(drop=True), SCAN_LEDGER_FILE)

# ---------------------------
# UI 及 主程序
# ---------------------------
with st.sidebar:
    st.header("V40.6-S1 稳定修复版")
    backtest_date_end = st.date_input("分析截止日期", value=datetime.now().date())
    BACKTEST_DAYS = st.number_input("分析天数 (设为 1 即启动实盘雷达)", min_value=1, value=100, step=1)
    
    TOP_BACKTEST = st.number_input("每日优选 TopK", min_value=1, value=3)
    
    st.markdown("---")
    RESUME_CHECKPOINT = st.checkbox("🔥 开启断点续传", value=True)
    st.caption("每完成一个交易日就安全保存；中断、刷新或网络重连后可继续。")
    clear_market_clicked = st.button("🗑️ 清除全部行情缓存")
    clear_history_clicked = st.button("🗑️ 清除断点记录 (重新回测)")
            
    st.markdown("---")
    st.subheader("💰 核心护城河门槛")
    MIN_PRICE = st.number_input("最低股价 (元)", value=10.0) 
    col1, col2 = st.columns(2)
    # 【改动1：市值基准提升】默认过滤掉 250亿以下的微盘股
    MIN_MV = col1.number_input("最小市值(亿)", value=250.0) 
    MAX_MV = col2.number_input("最大市值(亿)", value=1000.0)

TS_TOKEN_INPUT = st.text_input("Tushare Token", type="password")

if clear_market_clicked:
    if acquire_run_lock():
        try:
            removed = clear_all_market_caches()
            if removed:
                st.success(f"已清除 {len(removed)} 个行情缓存位置。")
            else:
                st.info("没有找到可清除的行情缓存。")
        finally:
            release_run_lock()
    else:
        st.warning("回测正在使用行情缓存，请等待当前批次结束。")

if clear_history_clicked:
    if acquire_run_lock():
        try:
            for file_path in (CHECKPOINT_FILE, SCAN_LEDGER_FILE, RUN_TASK_FILE):
                remove_with_backup(file_path)
            st.success("历史结果、扫描账本和断点任务已清理。")
        finally:
            release_run_lock()
    else:
        st.warning("回测正在写入结果，请等待当前批次结束。")

token_clean = clean_token_str(TS_TOKEN_INPUT)
is_realtime_mode = int(BACKTEST_DAYS) == 1
current_config_id = make_config_id(TOP_BACKTEST, MIN_MV, MAX_MV, MIN_PRICE)
task_before = read_json_safe(RUN_TASK_FILE)

if task_before.get('State') in {'RUNNING', 'PAUSED_ERROR', 'WAITING_DATA'}:
    done_count = int(task_before.get('Completed_Days', 0))
    total_count = int(task_before.get('Total_Days', 0))
    state_map = {'RUNNING': '运行中', 'PAUSED_ERROR': '异常暂停', 'WAITING_DATA': '等待缺失数据'}
    st.info(f"🔄 检测到断点回测：{state_map.get(task_before.get('State'))}，已完成 {done_count}/{total_count} 日。")

resume_clicked = False
stop_clicked = False
if task_before.get('State') in {'PAUSED_ERROR', 'WAITING_DATA'}:
    resume_clicked = st.button("▶️ 从断点继续")
if task_before.get('State') in {'RUNNING', 'PAUSED_ERROR', 'WAITING_DATA'}:
    stop_clicked = st.button("⏹️ 停止断点回测")

if stop_clicked:
    stopped_task = read_json_safe(RUN_TASK_FILE)
    stopped_task['State'] = 'STOPPED'
    save_task(stopped_task)
    st.warning("断点任务已停止；已完成的行情和结果仍保留。")

if resume_clicked:
    task_before['State'] = 'RUNNING'
    task_before['Error_Count'] = 0
    task_before.pop('Last_Error', None)
    save_task(task_before)

start_clicked = st.button("🚀 启动 V40.6 四大神盾追踪")
if start_clicked:
    token_ok, token_message = verify_token_connection(token_clean)
    if not token_ok:
        st.error(f"❌ Token预检拦截：{token_message}")
    else:
        # 新任务只保留一个大型行情索引，避免不同区间叠加耗尽内存。
        st.cache_resource.clear()
    if token_ok and not is_realtime_mode:
        new_task = {
            'State': 'RUNNING',
            'Config_ID': current_config_id,
            'Params': {
                'Backtest_Days': int(BACKTEST_DAYS),
                'Top_K': int(TOP_BACKTEST),
                'End_Date': backtest_date_end.strftime('%Y%m%d'),
                'Min_Price': float(MIN_PRICE),
                'Min_MV': float(MIN_MV),
                'Max_MV': float(MAX_MV),
            },
            'Reset_Config': not bool(RESUME_CHECKPOINT),
            'Completed_Days': 0,
            'Total_Days': 0,
            'Error_Count': 0,
        }
        save_task(new_task)

active_task = read_json_safe(RUN_TASK_FILE)
run_history = active_task.get('State') == 'RUNNING' and not stop_clicked
run_realtime = start_clicked and is_realtime_mode and not run_history
rerun_needed = False
realtime_result = pd.DataFrame()
realtime_error = None

if run_history or run_realtime:
    if not token_clean:
        if run_history:
            active_task['State'] = 'PAUSED_ERROR'
            active_task['Last_Error'] = 'Token为空'
            save_task(active_task)
        st.error("❌ Token为空，历史回测断点已保留。")
    elif not acquire_run_lock():
        st.info("另一个页面会话正在执行同一回测，本页面不会重复启动。")
    else:
        try:
            API_ERRORS.clear()
            SINA_STATUS = {'success': 0, 'fail': 0}
            if run_history:
                params = active_task['Params']
                run_days = int(params['Backtest_Days'])
                run_top_k = int(params['Top_K'])
                run_end_date = str(params['End_Date'])
                run_min_price = float(params['Min_Price'])
                run_min_mv = float(params['Min_MV'])
                run_max_mv = float(params['Max_MV'])
                run_config_id = str(active_task['Config_ID'])
                if active_task.pop('Reset_Config', False):
                    clear_config_records(run_config_id)
                    save_task(active_task)
            else:
                run_days = 1
                run_top_k = int(TOP_BACKTEST)
                run_end_date = backtest_date_end.strftime('%Y%m%d')
                run_min_price = float(MIN_PRICE)
                run_min_mv = float(MIN_MV)
                run_max_mv = float(MAX_MV)
                run_config_id = current_config_id

            ts.set_token(token_clean)
            pro = ts.pro_api(token_clean)
            try:
                setattr(pro, "_DataApi__timeout", TUSHARE_REQUEST_TIMEOUT_SECONDS)
            except Exception as exc:
                record_api_error(f"无法设置Tushare请求超时，将使用SDK默认值: {exc}")
            token_hash = hashlib.sha1(token_clean.encode('utf-8')).hexdigest()[:12]
            with st.spinner("正在加载V40.6原科技白名单和股票名称..."):
                GLOBAL_STOCK_INDUSTRY = load_industry_mapping(token_hash)
                GLOBAL_STOCK_BASIC = load_stock_basic(token_hash)

            trade_days_list = get_trade_days(run_end_date, run_days)
            if not trade_days_list:
                raise RuntimeError("未取得回测交易日")
            processed = completed_scan_dates(run_config_id) if run_history else set()
            pending_dates = [date for date in trade_days_list if date not in processed]
            if run_history:
                active_task['Total_Days'] = len(trade_days_list)
                active_task['Completed_Days'] = len(trade_days_list) - len(pending_dates)
                save_task(active_task)

            if not pending_dates:
                if run_history:
                    remove_with_backup(RUN_TASK_FILE)
                    st.success("🎉 指定区间回测已全部完成！")
                else:
                    st.warning("未取得可扫描日期。")
            else:
                sync_info = get_all_historical_data(trade_days_list, use_cache=True)
                if not sync_info['ok']:
                    raise RuntimeError("未能加载到任何完整行情；已下载部分仍保留在分片缓存中")

                batch_dates = pending_dates if run_realtime else pending_dates[:DAYS_PER_BATCH]
                bar = st.progress(0, text="箱体首发与四大神盾过滤中...")
                stopped_during_batch = False
                incomplete_dates = []
                completed_in_batch = 0
                for index, date in enumerate(batch_dates):
                    if run_history and read_json_safe(RUN_TASK_FILE).get('State') == 'STOPPED':
                        stopped_during_batch = True
                        break

                    is_realtime_radar = (
                        run_realtime and date == datetime.now().strftime('%Y%m%d')
                    )
                    run_timestamp = time.time() if is_realtime_radar else None
                    result, error, raw_count = run_backtest_for_a_day(
                        date, run_top_k, run_min_mv, run_max_mv, run_min_price,
                        use_sina=is_realtime_radar, run_timestamp=run_timestamp,
                    )

                    data_incomplete = error in {'数据缺失', '市值数据缺失', '股票基础数据缺失'}
                    if run_realtime:
                        realtime_result = result.copy()
                        realtime_error = error
                        if error and error != '无标的':
                            st.warning(f"[{date}] {error}；本次不把该日记为完成。")
                    elif data_incomplete:
                        incomplete_dates.append(date)
                        mark_scan_status(date, raw_count, 0, run_config_id, 'INCOMPLETE', error)
                    else:
                        replace_checkpoint_date(result, date, run_config_id)
                        mark_scan_status(
                            date, raw_count, len(result), run_config_id, 'COMPLETED', error or ''
                        )
                        completed_in_batch += 1
                        active_task['Completed_Days'] = len(completed_scan_dates(run_config_id))
                        active_task['Error_Count'] = 0
                        active_task['Last_Date'] = date
                        save_task(active_task)

                    bar.progress(
                        (index + 1) / max(len(batch_dates), 1),
                        text=f"分析中: {date} (候选 {raw_count} 只)",
                    )
                bar.empty()

                if run_realtime:
                    if not realtime_result.empty:
                        st.subheader(f"🎯 实盘雷达结果 [{batch_dates[0]}]")
                        st.dataframe(realtime_result, use_container_width=True)
                    elif realtime_error == '无标的':
                        st.info(f"[{batch_dates[0]}] 暂无符合V40.6原条件的股票。")
                elif stopped_during_batch:
                    st.warning("断点任务已停止；本批已完成的结果已经安全保存。")
                else:
                    completed_after = completed_scan_dates(run_config_id)
                    remaining_dates = [date for date in trade_days_list if date not in completed_after]
                    unattempted_remain = len(pending_dates) > len(batch_dates)
                    if unattempted_remain:
                        st.success(
                            f"✅ 本批完成 {completed_in_batch} 日；其余日期将从断点自动续跑。"
                        )
                        rerun_needed = True
                    elif remaining_dates:
                        active_task['State'] = 'WAITING_DATA'
                        active_task['Completed_Days'] = len(trade_days_list) - len(remaining_dates)
                        active_task['Missing_Dates'] = remaining_dates
                        save_task(active_task)
                        st.warning(
                            f"完整日期已全部处理；仍有 {len(remaining_dates)} 日数据不完整，"
                            "断点已保留。数据发布或网络恢复后点击“从断点继续”即可只补这些日期。"
                        )
                    else:
                        remove_with_backup(RUN_TASK_FILE)
                        st.success("🎉 回测数据更新完毕！")

            if run_realtime:
                st.markdown("---")
                if SINA_STATUS['success'] > 0:
                    st.success(f"✅ 盘中实时探针响应正常：成功接入新浪数据 {SINA_STATUS['success']} 次。")
                elif SINA_STATUS['fail'] > 0:
                    st.error(f"❌ 新浪实时数据抓取失败 {SINA_STATUS['fail']} 次，请确认当前是否在交易时间。")
                else:
                    st.info("ℹ️ 实时探针未触发（可能由于基础选股条件未通过）。")
        except Exception as error:
            if run_history:
                latest_task = read_json_safe(RUN_TASK_FILE) or active_task
                error_count = int(latest_task.get('Error_Count', 0)) + 1
                latest_task['Error_Count'] = error_count
                latest_task['Last_Error'] = str(error)
                if error_count < 3:
                    latest_task['State'] = 'RUNNING'
                    rerun_needed = True
                    st.warning(f"⚠️ 临时异常，断点已保留，将自动重试 ({error_count}/3)：{error}")
                else:
                    latest_task['State'] = 'PAUSED_ERROR'
                    st.error(f"❌ 连续3次失败，任务已安全暂停：{error}")
                save_task(latest_task)
            else:
                st.error(f"❌ 运行异常：{error}")
        finally:
            release_run_lock()

if rerun_needed:
    time.sleep(0.8)
    st.rerun()


# ---------------------------
# 报告始终从磁盘恢复，页面刷新后仍可查看和下载
# ---------------------------
latest_task = read_json_safe(RUN_TASK_FILE)
report_config_id = str(latest_task.get('Config_ID', current_config_id))
all_history = canonicalize_history(read_csv_safe(CHECKPOINT_FILE))
if not all_history.empty and 'Config_ID' in all_history.columns:
    all_res = all_history[all_history['Config_ID'].astype(str).eq(report_config_id)].copy()
else:
    all_res = pd.DataFrame()

ledger = read_csv_safe(SCAN_LEDGER_FILE)
if not ledger.empty and {'Config_ID', 'Scan_Status'}.issubset(ledger.columns):
    ledger_current = ledger[ledger['Config_ID'].astype(str).eq(report_config_id)]
    completed_count = int(ledger_current['Scan_Status'].astype(str).eq('COMPLETED').sum())
    incomplete_count = int(ledger_current['Scan_Status'].astype(str).eq('INCOMPLETE').sum())
    if completed_count or incomplete_count:
        st.caption(f"断点账本：已完成 {completed_count} 日；待补数据 {incomplete_count} 日。")

if not all_res.empty:
    all_res['Trade_Date'] = all_res['Trade_Date'].astype(str)
    st.header("📊 V40.6 实战定型版")
    st.subheader("🗓️ 周度生存与收益切片 (剔除不符合开盘要求的无效标的)")
    cols_row1 = st.columns(4)
    cols_row2 = st.columns(4)

    # 以下统计、展示和下载字段保持V40.6原样。
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
        'Rank', 'Trade_Date', 'name', 'ts_code', 'Wave_Count', 'Signal_Close',
        'Buy_Price', 'Gap_pct (%)', 'Total_Score', 'Breakout_S', 'Volume_S',
        'circ_mv', 'Exit_Reason'
    ] + [f'Return_W{w} (%)' for w in range(1, 9)]
    final_cols = [column for column in display_cols if column in all_res.columns]
    display_df = all_res[final_cols].sort_values(
        ['Trade_Date', 'Rank'], ascending=[False, True]
    ).reset_index(drop=True)

    def color_exit(value):
        if isinstance(value, str):
            if '剔除' in value: return 'color: white; background-color: darkgray'
            if '固定止损' in value: return 'color: white; background-color: darkred'
            if '保本止盈' in value: return 'color: orange'
            if '移动止盈' in value: return 'color: green'
            if '周期结束平仓' in value: return 'color: blue'
        return ''

    if 'Exit_Reason' in display_df.columns:
        try:
            st.dataframe(display_df.style.map(color_exit, subset=['Exit_Reason']), use_container_width=True)
        except AttributeError:
            st.dataframe(display_df.style.applymap(color_exit, subset=['Exit_Reason']), use_container_width=True)
    else:
        st.dataframe(display_df, use_container_width=True)

    csv = all_res.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 下载完整轨迹 (CSV)", csv, "export_v40_6_final.csv", "text/csv")
elif not realtime_result.empty:
    pass
elif ledger.empty:
    st.info("尚未运行历史回测。")
else:
    st.warning("当前参数已有已完成扫描日，但暂无符合条件的标的。")

if API_ERRORS:
    with st.expander("运行诊断（最近接口异常）"):
        for message in API_ERRORS[-20:]:
            st.text(message)
