# -*- coding: utf-8 -*-
"""
选股王 · V11.0 最终决战策略：V9.0 框架 + 强化 MACD 趋势共振版
更新说明：
1. 【**策略精调 V11.0**】：核心变动：
   - 目标：以 V9.0 为基础，精准修复 D+3 周期低胜率问题。
   - **MACD (w_macd)** 从 0.10 大幅提升至 **0.20** (强化中期趋势共振，筛选能持续走高 3-5 天的股票)。
   - **当日涨幅 (w_pct)** 和 **换手率 (w_turn)** 从 0.15 降至 **0.10** (为 MACD 腾出权重)。
   - **资金流 (w_mf)**、**60日位置 (w_position)** 和 **波动率 (w_volatility)** 权重维持 V9.0 水平，保持核心动力和防御性。

   新权重结构：资金流(0.35) + 趋势(0.20) + 防御(0.25) + 动能(0.20) = 1.00
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings("ignore")

# ---------------------------
# 页面设置
# ---------------------------
[span_0](start_span)st.set_page_config(page_title="选股王 · V11.0 最终决战策略", layout="wide")[span_0](end_span)
[span_1](start_span)st.title("选股王 · V11.0 最终决战策略（V9.0 框架 + 强化 MACD 趋势共振版）")[span_1](end_span)
[span_2](start_span)st.markdown("🎯 **V11.0 策略：在 $\mathbf{V9.0}$ 的基础上，将 $\mathbf{MACD}$ 权重提升到 $\mathbf{0.20}$，目标是巩固 $\mathbf{D+1}$ 胜率，并突破 $\mathbf{D+3}$ 胜率到 $\mathbf{50\%}$。**")[span_2](end_span)

# ---------------------------
# 全局变量初始化
# ---------------------------
pro = None

# ---------------------------
# 辅助函数 (关键优化点 1：移除 0.5 秒强制等待)
# ---------------------------
@st.cache_data(ttl=3600*12)
def safe_get(func_name, **kwargs):
    """安全调用 Tushare API，移除 0.5s 强制等待，改为 0.06s 以符合 1000次/分频次"""
    global pro
    if pro is None:
        [span_3](start_span)return pd.DataFrame(columns=['ts_code'])[span_3](end_span)
    func = getattr(pro, func_name)
    try:
        df = func(**kwargs)
        [span_4](start_span)if df is None or (isinstance(df, pd.DataFrame) and df.empty):[span_4](end_span)
            time.sleep(0.06) # 1000次/分钟 相当于 0.06秒/次
            [span_5](start_span)return pd.DataFrame(columns=['ts_code'])[span_5](end_span)
        time.sleep(0.06) # 1000次/分钟 相当于 0.06秒/次
        return df
    except Exception as e:
        time.sleep(0.06) # 1000次/分钟 相当于 0.06秒/次
        return pd.DataFrame(columns=['ts_code'])

def get_trade_days(end_date_str, num_days):
    """获取 num_days 个交易日作为选股日"""
    [span_6](start_span)start_date = (datetime.strptime(end_date_str, "%Y%m%d") - timedelta(days=num_days * 2)).strftime("%Y%m%d")[span_6](end_span)
    [span_7](start_span)cal = safe_get('trade_cal', start_date=start_date, end_date=end_date_str)[span_7](end_span)
    [span_8](start_span)if cal.empty or 'is_open' not in cal.columns:[span_8](end_span)
        [span_9](start_span)st.error("无法获取交易日历，请检查 Token 或 Tushare 权限。")[span_9](end_span)
        return []
    [span_10](start_span)trade_days_df = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)[span_10](end_span)
    [span_11](start_span)trade_days_df = trade_days_df[trade_days_df['cal_date'] <= end_date_str][span_11](end_span)
    [span_12](start_span)return trade_days_df['cal_date'].head(num_days).tolist()[span_12](end_span)

@st.cache_data(ttl=3600*24)
def get_adj_factor(ts_code, start_date, end_date):
    [span_13](start_span)df = safe_get('adj_factor', ts_code=ts_code, start_date=start_date, end_date=end_date)[span_13](end_span)
    [span_14](start_span)if df.empty or 'adj_factor' not in df.columns: return pd.DataFrame()[span_14](end_span)
    [span_15](start_span)df['adj_factor'] = pd.to_numeric(df['adj_factor'], errors='coerce').fillna(0)[span_15](end_span)
    [span_16](start_span)df = df.set_index('trade_date').sort_index()[span_16](end_span)
    return df['adj_factor']

@st.cache_data(ttl=3600*12)
def get_qfq_data_v4(ts_code, start_date, end_date, adj_factor_series=None):
    """
    获取单个股票的前复权数据。在批量模式下，adj_factor_series 会预先传入。
    """
    [span_17](start_span)daily_df = safe_get('daily', ts_code=ts_code, start_date=start_date, end_date=end_date)[span_17](end_span)
    [span_18](start_span)if daily_df.empty: return pd.DataFrame()[span_18](end_span)
    [span_19](start_span)daily_df = daily_df.set_index('trade_date').sort_index()[span_19](end_span)
    
    if adj_factor_series is None:
        [span_20](start_span)adj_factor_series = get_adj_factor(ts_code, start_date, end_date)[span_20](end_span)

    [span_21](start_span)if adj_factor_series.empty: return pd.DataFrame()[span_21](end_span)
    
    [span_22](start_span)df = daily_df.merge(adj_factor_series.rename('adj_factor'), left_index=True, right_index=True, how='left')[span_22](end_span)
    [span_23](start_span)df = df.dropna(subset=['adj_factor'])[span_23](end_span)
    [span_24](start_span)if df.empty: return pd.DataFrame()[span_24](end_span)
    
    # 确保 adj_factor 在合并后存在且是 Series
    if 'adj_factor' not in df.columns: return pd.DataFrame()

    [span_25](start_span)latest_adj_factor = df['adj_factor'].iloc[-1][span_25](end_span)
    [span_26](start_span)for col in ['open', 'high', 'low', 'close', 'pre_close']:[span_26](end_span)
        if col in df.columns:
            [span_27](start_span)if latest_adj_factor > 1e-9:[span_27](end_span)
                [span_28](start_span)df[col + '_qfq'] = df[col] * df['adj_factor'] / latest_adj_factor[span_28](end_span)
            else:
                [span_29](start_span)df[col + '_qfq'] = df[col][span_29](end_span)
    [span_30](start_span)df = df.reset_index().rename(columns={'trade_date': 'trade_date_str'})[span_30](end_span)
    [span_31](start_span)df['trade_date'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')[span_31](end_span)
    [span_32](start_span)df = df.sort_values('trade_date').set_index('trade_date_str')[span_32](end_span)
    [span_33](start_span)for col in ['open', 'high', 'low', 'close']:[span_33](end_span)
        [span_34](start_span)df[col] = df[col + '_qfq'][span_34](end_span)
    [span_35](start_span)return df[['open', 'high', 'low', 'close', 'vol']].copy()[span_35](end_span)

# ----------------------------------------------------
# 关键优化点 2.1：批量获取所有历史数据
# ----------------------------------------------------
def get_bulk_history_and_adj(ts_codes, selection_date):
    """
    批量获取所有候选股的历史 (120天) 和未来 (15天) 数据，
    并获取复权因子。

    返回: {ts_code: {'hist': pd.DataFrame, 'adj_factor': pd.Series}}
    """
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    # 历史数据 (120 天)
    start_hist = (d0 - timedelta(days=120 * 2)).strftime("%Y%m%d") # 预留时间
    end_hist = selection_date

    # 未来数据 (15 天)
    start_future = (d0 + timedelta(days=1)).strftime("%Y%m%d")
    end_future = (d0 + timedelta(days=15)).strftime("%Y%m%d")

    # 1. 批量获取复权因子 (效率更高)
    # Tushare adj_factor 接口不支持批量，仍需循环调用
    adj_map = {
        ts_code: get_adj_factor(ts_code, start_hist, end_future)
        for ts_code in ts_codes
    }

    # 2. 批量获取历史和未来行情数据
    data_map = {}
    for ts_code in ts_codes:
        adj_factor_series = adj_map.get(ts_code)
        
        # 获取包含选股日及以前的历史数据（用于指标计算）
        hist_df = get_qfq_data_v4(ts_code, start_hist, end_hist, adj_factor_series=adj_factor_series)
        
        # 获取选股日以后的未来价格数据（用于回测收益计算）
        future_df = get_qfq_data_v4(ts_code, start_future, end_future, adj_factor_series=adj_factor_series)
        
        data_map[ts_code] = {
            'hist_data': hist_df, # 包含选股日当日数据
            'future_data': future_df # 选股日后第一天开始
        }
        
    return data_map

# ----------------------------------------------------
# 关键优化点 2.2：使用预加载的数据计算指标
# ----------------------------------------------------
def get_future_prices_optimized(ts_code, selection_date, preloaded_data, days_ahead=[1, 3, 5]):
    """使用预加载的未来数据计算收益率"""
    d0 = datetime.strptime(selection_date, "%Y%m%d")
    results = {}
    
    # 获取选股当日的收盘价（作为计算收益的基准价）
    # 在 hist_data 中获取选股日的收盘价
    hist = preloaded_data.get('hist_data', pd.DataFrame())
    future = preloaded_data.get('future_data', pd.DataFrame())

    if hist.empty or 'close' not in hist.columns:
        for n in days_ahead: results[f'Return_D{n}'] = np.nan
        return results
    
    selection_price_adj = hist['close'].iloc[-1]
    
    if future.empty or 'close' not in future.columns:
        for n in days_ahead: results[f'Return_D{n}'] = np.nan
        return results

    [span_36](start_span)future['close'] = pd.to_numeric(future['close'], errors='coerce')[span_36](end_span)
    [span_37](start_span)future = future.dropna(subset=['close'])[span_37](end_span)
    [span_38](start_span)future = future.reset_index(drop=True)[span_38](end_span)

    for n in days_ahead:
        col_name = f'Return_D{n}'
        if len(future) >= n:
            future_price = future.iloc[n-1]['close']
            if pd.notna(selection_price_adj) and selection_price_adj > 1e-9:
                [span_39](start_span)results[col_name] = (future_price / selection_price_adj - 1) * 100[span_39](end_span)
            else:
                [span_40](start_span)results[col_name] = np.nan[span_40](end_span)
        else:
            [span_41](start_span)results[col_name] = np.nan[span_41](end_span)
    return results


def compute_indicators_optimized(ts_code, preloaded_data):
    """使用预加载的历史数据计算 MACD, 60日位置等指标"""
    df = preloaded_data.get('hist_data', pd.DataFrame())
    res = {}
    [span_42](start_span)if df.empty or len(df) < 3 or 'close' not in df.columns: return res[span_42](end_span)
    
    # 确保只使用 120 天的数据进行指标计算
    df = df.tail(120)

    [span_43](start_span)df['close'] = pd.to_numeric(df['close'], errors='coerce').astype(float)[span_43](end_span)
    [span_44](start_span)df['low'] = pd.to_numeric(df['low'], errors='coerce').astype(float)[span_44](end_span)
    [span_45](start_span)df['high'] = pd.to_numeric(df['high'], errors='coerce').astype(float)[span_45](end_span)
    [span_46](start_span)df['vol'] = pd.to_numeric(df['vol'], errors='coerce').fillna(0)[span_46](end_span)
    [span_47](start_span)df['pct_chg'] = df['close'].pct_change().fillna(0) * 100[span_47](end_span)
    close = df['close']
    [span_48](start_span)res['last_close'] = close.iloc[-1][span_48](end_span)
    
    # MACD 计算
    if len(close) >= 26:
        [span_49](start_span)ema12 = close.ewm(span=12, adjust=False).mean()[span_49](end_span)
        [span_50](start_span)ema26 = close.ewm(span=26, adjust=False).mean()[span_50](end_span)
        [span_51](start_span)diff = ema12 - ema26[span_51](end_span)
        [span_52](start_span)dea = diff.ewm(span=9, adjust=False).mean()[span_52](end_span)
        [span_53](start_span)res['macd_val'] = ((diff - dea) * 2).iloc[-1][span_53](end_span)
    [span_54](start_span)else: res['macd_val'] = np.nan[span_54](end_span)
        
    # 量比计算
    vols = df['vol'].tolist()
    if len(vols) >= 6 and vols[-6:-1] and np.mean(vols[-6:-1]) > 1e-9:
        [span_55](start_span)res['vol_ratio'] = vols[-1] / np.mean(vols[-6:-1])[span_55](end_span)
    [span_56](start_span)else: res['vol_ratio'] = np.nan[span_56](end_span)
        
    # 10日回报、波动率计算
    [span_57](start_span)res['10d_return'] = close.iloc[-1]/close.iloc[-10] - 1 if len(close)>=10 and close.iloc[-10]!=0 else 0[span_57](end_span)
    [span_58](start_span)res['volatility'] = df['pct_chg'].tail(10).std() if len(df)>=10 else 0[span_58](end_span)
    
    # 60日位置计算
    [span_59](start_span)if len(df) >= 60:[span_59](end_span)
        [span_60](start_span)hist_60 = df.tail(60)[span_60](end_span)
        [span_61](start_span)min_low = hist_60['low'].min()[span_61](end_span)
        [span_62](start_span)max_high = hist_60['high'].max()[span_62](end_span)
        [span_63](start_span)current_close = hist_60['close'].iloc[-1][span_63](end_span)
        
        [span_64](start_span)if max_high == min_low: res['position_60d'] = 50.0[span_64](end_span)
        [span_65](start_span)else: res['position_60d'] = (current_close - min_low) / (max_high - min_low) * 100[span_65](end_span)
    [span_66](start_span)else: res['position_60d'] = np.nan[span_66](end_span)
    
    return res

# ----------------------------------------------------


# ----------------------------------------------------
# 侧边栏参数 (定义 BACKTEST_DAYS 等变量)
# ----------------------------------------------------
with st.sidebar:
    st.header("模式与日期选择")
    [span_67](start_span)backtest_date_end = st.date_input("选择**回测结束日期**", value=datetime.now().date(), max_value=datetime.now().date())[span_67](end_span)
    [span_68](start_span)BACKTEST_DAYS = int(st.number_input("**自动回测天数 (N)**", value=20, step=1, min_value=1, max_value=50, help="程序将自动回测最近 N 个交易日。建议设置为 20 天以获得更可靠的统计数据。"))[span_68](end_span)
    
    st.markdown("---")
    st.header("核心参数")
    [span_69](start_span)FINAL_POOL = int(st.number_input("最终入围评分数量 (M)", value=10, step=1, min_value=1))[span_69](end_span)
    [span_70](start_span)TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=10, step=1))[span_70](end_span)
    [span_71](start_span)TOP_BACKTEST = int(st.number_input("回测分析 Top K", value=3, step=1, min_value=1))[span_71](end_span)
    
    st.markdown("---")
    st.header("🛒 灵活过滤条件")
    [span_72](start_span)MIN_PRICE = st.number_input("最低股价 (元)", value=10.0, step=0.5, min_value=0.1)[span_72](end_span)
    [span_73](start_span)MAX_PRICE = st.number_input("最高股价 (元)", value=300.0, step=5.0, min_value=1.0)[span_73](end_span)
    [span_74](start_span)MIN_TURNOVER = st.number_input("最低换手率 (%)", value=2.0, step=0.5, min_value=0.1)[span_74](end_span)
    [span_75](start_span)MIN_CIRC_MV_BILLIONS = st.number_input("最低流通市值 (亿元)", value=20.0, step=1.0, min_value=1.0, help="例如：输入 20 代表流通市值必须大于等于 20 亿元。")[span_75](end_span)
    [span_76](start_span)MIN_AMOUNT_MILLIONS = st.number_input("最低成交额 (亿元)", value=0.6, step=0.1, min_value=0.1)[span_76](end_span)
    [span_77](start_span)MIN_AMOUNT = MIN_AMOUNT_MILLIONS * 100000000[span_77](end_span)

# ---------------------------
# Token 输入与初始化
# ---------------------------
[span_78](start_span)TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")[span_78](end_span)
if not TS_TOKEN:
    [span_79](start_span)st.warning("请输入 Tushare Token 才能运行脚本。")[span_79](end_span)
    st.stop()
[span_80](start_span)ts.set_token(TS_TOKEN)[span_80](end_span)
[span_81](start_span)pro = ts.pro_api()[span_81](end_span)

# ---------------------------
# 核心回测逻辑函数
# ---------------------------
def run_backtest_for_a_day(last_trade, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS):
    """为单个交易日运行选股和回测逻辑"""
    
    # 1. 拉取全市场 Daily 数据 (省略合并和过滤的重复代码)
    [span_82](start_span)daily_all = safe_get('daily', trade_date=last_trade)[span_82](end_span)
    [span_83](start_span)if daily_all.empty or 'ts_code' not in daily_all.columns: return pd.DataFrame(), f"数据缺失或拉取失败：{last_trade}"[span_83](end_span)

    [span_84](start_span)pool_raw = daily_all.reset_index(drop=True)[span_84](end_span)
    [span_85](start_span)stock_basic = safe_get('stock_basic', list_status='L', fields='ts_code,name,list_date')[span_85](end_span)
    [span_86](start_span)REQUIRED_BASIC_COLS = ['ts_code','turnover_rate','amount','total_mv','circ_mv'][span_86](end_span)
    [span_87](start_span)daily_basic = safe_get('daily_basic', trade_date=last_trade, fields=','.join(REQUIRED_BASIC_COLS))[span_87](end_span)
    [span_88](start_span)mf_raw = safe_get('moneyflow', trade_date=last_trade)[span_88](end_span)
    pool_merged = pool_raw.copy()

    [span_89](start_span)if not stock_basic.empty and 'name' in stock_basic.columns:[span_89](end_span)
        [span_90](start_span)pool_merged = pool_merged.merge(stock_basic[['ts_code','name','list_date']], on='ts_code', how='left')[span_90](end_span)
    else:
        [span_91](start_span)pool_merged['name'] = pool_merged['ts_code'][span_91](end_span)
        [span_92](start_span)pool_merged['list_date'] = '20000101'[span_92](end_span)
        
    if not daily_basic.empty:
        [span_93](start_span)cols_to_merge = [c for c in REQUIRED_BASIC_COLS if c in daily_basic.columns][span_93](end_span)
        [span_94](start_span)if 'amount' in pool_merged.columns and 'amount' in cols_to_merge:[span_94](end_span)
            [span_95](start_span)pool_merged = pool_merged.drop(columns=['amount'])[span_95](end_span)
        [span_96](start_span)pool_merged = pool_merged.merge(daily_basic[cols_to_merge], on='ts_code', how='left')[span_96](end_span)
    
    [span_97](start_span)moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])[span_97](end_span)
    if not mf_raw.empty:
        [span_98](start_span)possible = ['net_mf','net_mf_amount','net_mf_in'][span_98](end_span)
        for c in possible:
            if c in mf_raw.columns:
                [span_99](start_span)moneyflow = mf_raw[['ts_code', c]].rename(columns={c:'net_mf'}).fillna(0)[span_99](end_span)
                break            
    if not moneyflow.empty:
        [span_100](start_span)pool_merged = pool_merged.merge(moneyflow, on='ts_code', how='left')[span_100](end_span)
        
    [span_101](start_span)pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0)[span_101](end_span)
    [span_102](start_span)pool_merged['turnover_rate'] = pool_merged['turnover_rate'].fillna(0)[span_102](end_span)
   
    # 3. 执行硬性条件过滤
    [span_103](start_span)df = pool_merged.copy()[span_103](end_span)
    [span_104](start_span)df['close'] = pd.to_numeric(df['close'], errors='coerce')[span_104](end_span)
    [span_105](start_span)df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').fillna(0)[span_105](end_span)
    [span_106](start_span)df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000 # 转换为万元[span_106](end_span)
    [span_107](start_span)df['circ_mv_billion'] = pd.to_numeric(df['circ_mv'], errors='coerce').fillna(0) / 10000[span_107](end_span)
    [span_108](start_span)df['name'] = df['name'].astype(str)[span_108](end_span)
    
    # 过滤 ST 股/退市股/北交所/次新股
    [span_109](start_span)mask_st = df['name'].str.contains('ST|退', case=False, na=False)[span_109](end_span)
    [span_110](start_span)df = df[~mask_st][span_110](end_span)
    [span_111](start_span)mask_bj = df['ts_code'].str.startswith('92')[span_111](end_span)
    [span_112](start_span)df = df[~mask_bj][span_112](end_span)
    [span_113](start_span)TODAY = datetime.strptime(last_trade, "%Y%m%d")[span_113](end_span)
    [span_114](start_span)MIN_LIST_DAYS = 120[span_114](end_span)
    [span_115](start_span)df['list_date_dt'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')[span_115](end_span)
    [span_116](start_span)df['days_listed'] = (TODAY - df['list_date_dt']).dt.days[span_116](end_span)
    [span_117](start_span)mask_cyb_kcb = df['ts_code'].str.startswith(('30','68'))[span_117](end_span)
    [span_118](start_span)mask_new = df['days_listed'] < MIN_LIST_DAYS[span_118](end_span)
    [span_119](start_span)df = df[~((mask_cyb_kcb) & (mask_new))][span_119](end_span)

    # 过滤价格
    [span_120](start_span)mask_price = (df['close'] >= MIN_PRICE) & (df['close'] <= MAX_PRICE)[span_120](end_span)
    [span_121](start_span)df = df[mask_price][span_121](end_span)
    # 过滤流通市值
    [span_122](start_span)mask_circ_mv = df['circ_mv_billion'] >= MIN_CIRC_MV_BILLIONS[span_122](end_span)
    [span_123](start_span)df = df[mask_circ_mv][span_123](end_span)
    # 过滤换手率
    [span_124](start_span)mask_turn = df['turnover_rate'] >= MIN_TURNOVER[span_124](end_span)
    [span_125](start_span)df = df[mask_turn][span_125](end_span)
    # 过滤成交额
    [span_126](start_span)mask_amt = df['amount'] * 1000 >= MIN_AMOUNT[span_126](end_span)
    [span_127](start_span)df = df[mask_amt][span_127](end_span)
    
    [span_128](start_span)df = df.reset_index(drop=True)[span_128](end_span)

    if len(df) == 0: return pd.DataFrame(), f"过滤后无股票：{last_trade}"

    # 4. 遴选决赛名单 (基于当日涨幅和换手率的混合初筛)
    limit_pct = int(FINAL_POOL * 0.7)
    df_pct = df.sort_values('pct_chg', ascending=False).head(limit_pct).copy()
    limit_turn = FINAL_POOL - len(df_pct)
    existing_codes = set(df_pct['ts_code'])
    df_turn = df[~df['ts_code'].isin(existing_codes)].sort_values('turnover_rate', ascending=False).head(limit_turn).copy()
    [span_129](start_span)final_candidates = pd.concat([df_pct, df_turn]).reset_index(drop=True)[span_129](end_span)
    
    # =================================================================================
    # 🚨 关键优化点 2.3：批量获取历史数据和未来收益数据，代替循环内的 API 调用
    # =================================================================================
    final_ts_codes = final_candidates['ts_code'].tolist()
    preloaded_data_map = get_bulk_history_and_adj(final_ts_codes, last_trade)
 
    # 5. 深度评分 (使用预加载的数据)
    records = []
    for row in final_candidates.itertuples():
        ts_code = row.ts_code
        preloaded_data = preloaded_data_map.get(ts_code, {})
        
        rec = {
            'ts_code': ts_code, 'name': getattr(row, 'name', ts_code),
            'Close': getattr(row, 'close', np.nan),
            'Circ_MV (亿)': getattr(row, 'circ_mv_billion', np.nan),
            [span_130](start_span)'Pct_Chg (%)': getattr(row, 'pct_chg', 0),[span_130](end_span)
            'turnover': getattr(row, 'turnover_rate', 0),
            'net_mf': getattr(row, 'net_mf', 0)
        }
        
        # 使用优化后的函数，不再发起 API 调用
        ind = compute_indicators_optimized(ts_code, preloaded_data)
        rec.update({
            'vol_ratio': ind.get('vol_ratio', 0), 'macd': ind.get('macd_val', 0),
            [span_131](start_span)'10d_return': ind.get('10d_return', 0),[span_131](end_span)
            'volatility': ind.get('volatility', 0), 'position_60d': ind.get('position_60d', np.nan)
        })
        
        # 使用优化后的函数，不再发起 API 调用
        future_returns = get_future_prices_optimized(ts_code, last_trade, preloaded_data)
        rec.update({
            'Return_D1 (%)': future_returns.get('Return_D1', np.nan),
            [span_132](start_span)'Return_D3 (%)': future_returns.get('Return_D3', np.nan),[span_132](end_span)
            'Return_D5 (%)': future_returns.get('Return_D5', np.nan),
        })

        records.append(rec)
    
    [span_133](start_span)fdf = pd.DataFrame(records)[span_133](end_span)
    [span_134](start_span)if fdf.empty: return pd.DataFrame(), f"评分列表为空：{last_trade}"[span_134](end_span)

    # 6. 归一化与 V11.0 策略精调评分
    def normalize(series):
        [span_135](start_span)series_nn = series.dropna()[span_135](end_span)
        [span_136](start_span)if series_nn.max() == series_nn.min(): return pd.Series([0.5] * len(series), index=series.index)[span_136](end_span)
        [span_137](start_span)return (series - series_nn.min()) / (series_nn.max() - series_nn.min() + 1e-9)[span_137](end_span)

    [span_138](start_span)fdf['s_pct'] = normalize(fdf['Pct_Chg (%)'])[span_138](end_span)
    [span_139](start_span)fdf['s_turn'] = normalize(fdf['turnover'])[span_139](end_span)
    [span_140](start_span)fdf['s_vol'] = normalize(fdf['vol_ratio'])[span_140](end_span)
    [span_141](start_span)fdf['s_mf'] = normalize(fdf['net_mf'])[span_141](end_span)
    [span_142](start_span)fdf['s_macd'] = normalize(fdf['macd'])[span_142](end_span)
    [span_143](start_span)fdf['s_trend'] = normalize(fdf['10d_return'])[span_143](end_span)
    [span_144](start_span)fdf['s_volatility'] = normalize(fdf['volatility'])[span_144](end_span)
    [span_145](start_span)fdf['s_position'] = fdf['position_60d'] / 100[span_145](end_span)
    
    # ----------------------------------------------------------------------------------
    # 🚨 V11.0 最终决战策略：V9.0 框架 + 强化 MACD 趋势共振版
    
    # 核心权重：资金流，占比 35%
    [span_146](start_span)w_mf = 0.35[span_146](end_span) # 35% - 资金流 (核心动力，保持 V9.0)

    # 动能权重：当日动能，占比 20%
    [span_147](start_span)w_pct = 0.10[span_147](end_span) # 10% - 当日涨幅 (削弱)
    [span_148](start_span)w_turn = 0.10[span_148](end_span) # 10% - 换手率 (削弱)
    
    # 防御权重：安全边际与波动控制，占比 25%
    [span_149](start_span)w_position = 0.15[span_149](end_span) # 15% - 60日位置 (保持 V9.0)
    [span_150](start_span)w_volatility = 0.10[span_150](end_span) # 10% - 波动率 (保持 V9.0)
 
    # 趋势权重：中期趋势，占比 20%
    [span_151](start_span)w_macd = 0.20[span_151](end_span) # 20% - MACD (**大幅强化，目标改善 D+3**)
    
    # 彻底归零项
    [span_152](start_span)w_vol = 0.00[span_152](end_span) # 0% - 量比
    [span_153](start_span)w_trend = 0.00[span_153](end_span) # 0% - 10日回报
    
    # Sum: 0.35+0.10+0.10+0.15+0.10+0.20 = 1.00
    
    score = (
        [span_154](start_span)fdf['s_pct'] * w_pct + fdf['s_turn'] * w_turn +[span_154](end_span)
        [span_155](start_span)fdf['s_mf'] * w_mf +[span_155](end_span)
        [span_156](start_span)fdf['s_macd'] * w_macd +[span_156](end_span)
        
        # 引入防御：60日位置越低越好 (1-s_position)，波动率越低越好 (1-s_volatility)
        (1 - fdf['s_position']) * [span_157](start_span)w_position +[span_157](end_span)
        (1 - fdf['s_volatility']) * [span_158](start_span)w_volatility +[span_158](end_span)
        
        # 归零项
        [span_159](start_span)fdf['s_vol'] * w_vol +[span_159](end_span)
        fdf['s_trend'] * w_trend     
    )
    [span_160](start_span)fdf['综合评分'] = score * 100[span_160](end_span)
    [span_161](start_span)fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)[span_161](end_span)
    [span_162](start_span)fdf.index += 1[span_162](end_span)
    # ----------------------------------------------------------------------------------


    return fdf.head(TOP_BACKTEST).copy(), None

# ---------------------------
# 主运行块 (保持不变)
# ---------------------------
if st.button(f"🚀 开始 {BACKTEST_DAYS} 日自动回测"):
    
    [span_163](start_span)st.warning("⚠️ **V11.0 版本已更换为 V9.0 框架 + 强化 MACD 趋势共振策略，目标是突破 D+3 胜率。**")[span_163](end_span)
   
    [span_164](start_span)trade_days_str = get_trade_days(backtest_date_end.strftime("%Y%m%d"), BACKTEST_DAYS)[span_164](end_span)
    if not trade_days_str:
        [span_165](start_span)st.error("无法获取交易日列表，请检查日期或 Token。")[span_165](end_span)
        st.stop()
    
    [span_166](start_span)st.header(f"📈 正在进行 {BACKTEST_DAYS} 个交易日的回测...")[span_166](end_span)
    
    results_list = []
    total_days = len(trade_days_str)
    
    [span_167](start_span)progress_text = st.empty()[span_167](end_span)
    [span_168](start_span)my_bar = st.progress(0)[span_168](end_span)
    
    for i, trade_date in enumerate(trade_days_str):
        [span_169](start_span)progress_text.text(f"🚀 正在处理第 {i+1}/{total_days} 个交易日：{trade_date}")[span_169](end_span)
      
        daily_result_df, error = run_backtest_for_a_day(
            trade_date, TOP_BACKTEST, FINAL_POOL, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, MIN_CIRC_MV_BILLIONS
        [span_170](start_span))
        
        if error:
            st.warning(f"跳过 {trade_date}：{error}")[span_170](end_span)
        elif not daily_result_df.empty:
            [span_171](start_span)daily_result_df['Trade_Date'] = trade_date[span_171](end_span)
            [span_172](start_span)results_list.append(daily_result_df)[span_172](end_span)
            
        [span_173](start_span)my_bar.progress((i + 1) / total_days)[span_173](end_span)

    [span_174](start_span)progress_text.text("✅ 回测完成，正在汇总结果...")[span_174](end_span)
    [span_175](start_span)my_bar.empty()[span_175](end_span)
    
    if not results_list:
        [span_176](start_span)st.error("所有交易日的回测均失败或无结果。")[span_176](end_span)
        st.stop()
        
    [span_177](start_span)all_results = pd.concat(results_list)[span_177](end_span)
    
    [span_178](start_span)st.header(f"📊 最终平均回测结果 (Top {TOP_BACKTEST}，共 {total_days} 个交易日)")[span_178](end_span)
    
    [span_179](start_span)for n in [1, 3, 5]:[span_179](end_span)
        [span_180](start_span)col = f'Return_D{n} (%)'[span_180](end_span)
        
        [span_181](start_span)filtered_returns = all_results.copy()[span_181](end_span)
        [span_182](start_span)valid_returns = filtered_returns.dropna(subset=[col])[span_182](end_span)

        if not valid_returns.empty:
            [span_183](start_span)avg_return = valid_returns[col].mean()[span_183](end_span)
            [span_184](start_span)hit_rate = (valid_returns[col] > 0).sum() / len(valid_returns) * 100 if len(valid_returns) > 0 else 0.0[span_184](end_span)
            [span_185](start_span)total_count = len(valid_returns)[span_185](end_span)
        else:
            [span_186](start_span)avg_return = np.nan[span_186](end_span)
            [span_187](start_span)hit_rate = 0.0[span_187](end_span)
            [span_188](start_span)total_count = 0[span_188](end_span)
            
        st.metric(f"Top {TOP_BACKTEST}：D+{n} 平均收益 / 准确率", 
            [span_189](start_span)f"{avg_return:.2f}% / {hit_rate:.1f}%",[span_189](end_span)
            [span_190](start_span)help=f"总有效样本数：{total_count}。**V11.0 已应用 V9.0 框架 + 强化 MACD 趋势共振策略。**")[span_190](end_span)

    [span_191](start_span)st.header("📋 每日回测详情 (Top K 明细)")[span_191](end_span)
    
    display_cols = ['Trade_Date', 'name', 'ts_code', '综合评分', 
                    'Close', 'Pct_Chg (%)', 'Circ_MV (亿)',
                    [span_192](start_span)'Return_D1 (%)', 'Return_D3 (%)', 'Return_D5 (%)'][span_192](end_span)
    
    st.dataframe(all_results[display_cols].sort_values('Trade_Date', ascending=False), use_container_width=True)
