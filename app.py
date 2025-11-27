# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（BC 混合增强版）· 极速版
说明：
- 【本次修复】**最大化垂直空间**（精简标题，移除多余标题/空行）。
- **优化页面回弹问题**：将回测和选股逻辑函数化，稳定内容显示。
"""

import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 页面设置
# ---------------------------
st.set_page_config(page_title="选股王（极速版）", layout="wide")

# 标题优化：使用 Markdown H3 进一步减小字号，仅保留最简信息
st.markdown("### 选股王（极速版）")

# ---------------------------
# 侧边栏参数（实时可改）- 保持不变
# ---------------------------
with st.sidebar:
    st.header("可调参数（实时）")
    INITIAL_TOP_N = int(st.number_input("初筛：涨幅榜取前 N", value=1000, step=100))
    FINAL_POOL = int(st.number_input("清洗后取前 M 进入评分", value=500, step=50))
    TOP_DISPLAY = int(st.number_input("界面显示 Top K", value=30, step=5))
    
    MIN_PRICE = float(st.number_input("最低价格 (元)", value=10.0, step=1.0))
    MAX_PRICE = float(st.number_input("最高价格 (元)", value=200.0, step=10.0))
    MIN_TURNOVER = float(st.number_input("最低换手率 (%)", value=3.0, step=0.5))
    MIN_AMOUNT = float(st.number_input("最低成交额 (元)", value=200_000_000.0, step=50_000_000.0))
    VOL_SPIKE_MULT = float(st.number_input("放量倍数阈值 (vol_last > vol_ma5 * x)", value=1.7, step=0.1))
    VOLATILITY_MAX = float(st.number_input("过去10日波动 std 阈值 (%)", value=8.0, step=0.5))
    HIGH_PCT_THRESHOLD = float(st.number_input("视为大阳线 pct_chg (%)", value=6.0, step=0.5))
    
    BACKTEST_DAYS = int(st.number_input("回测：最近 N 个交易日", value=10, step=1)) # 读取回测天数
    st.markdown("---")
    st.caption("提示：保守→降低阈值；激进→提高阈值。")

# ---------------------------
# Token 输入（主区 - 优化空间）
# ---------------------------
st.markdown("请输入 Tushare Token。若有权限缺失，脚本会自动降级并继续运行。")
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password", label_visibility="collapsed")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

# 初始化 tushare
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ---------------------------
# 依赖函数 (保持不变)
# ---------------------------
def safe_get(func, **kwargs):
    try:
        df = func(**kwargs)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def find_last_trade_day(max_days=20):
    today = datetime.now().date()
    for i in range(max_days):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        df = safe_get(pro.daily, trade_date=ds)
        if not df.empty:
            return ds
    return None

last_trade = find_last_trade_day()
if not last_trade:
    st.error("无法找到最近交易日，检查网络或 Token 权限。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")

# ----------------------------------------------------
# 按钮控制模块（优化：移除 “运行模式选择” 标题）
# ----------------------------------------------------
if 'run_selection' not in st.session_state: st.session_state['run_selection'] = False
if 'run_backtest' not in st.session_state: st.session_state['run_backtest'] = False

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 运行当日选股", use_container_width=True):
        st.session_state['run_selection'] = True
        st.session_state['run_backtest'] = False
        st.rerun()

with col2:
    if st.button(f"✅ 运行历史回测 ({BACKTEST_DAYS} 日)", use_container_width=True):
        st.session_state['run_backtest'] = True
        st.session_state['run_selection'] = False
        st.rerun()

st.markdown("---")

# ---------------------------
# 核心评分函数 (缓存 - 修复 FINAL_POOL 缓存问题)
# ---------------------------
@st.cache_data(ttl=600)
def get_hist(ts_code, end_date, days=60):
    try:
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days*2)).strftime("%Y%m%d")
        df = safe_get(pro.daily, ts_code=ts_code, start_date=start, end_date=end_date)
        if df.empty:
            return pd.DataFrame()
        df = df.sort_values('trade_date').reset_index(drop=True)
        return df
    except:
        return pd.DataFrame()

def compute_indicators(df):
    res = {}
    if df.empty or len(df) < 3: return res
    close = df['close'].astype(float); high = df['high'].astype(float); low = df['low'].astype(float)
    try: res['last_close'] = close.iloc[-1]
    except: res['last_close'] = np.nan
    for n in (5,10,20):
        if len(close) >= n: res[f'ma{n}'] = close.rolling(window=n).mean().iloc[-1]
        else: res[f'ma{n}'] = np.nan
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_val = (diff - dea) * 2
        res['macd'] = macd_val.iloc[-1]; res['diff'] = diff.iloc[-1]; res['dea'] = dea.iloc[-1]
    else: res['macd'] = res['diff'] = res['dea'] = np.nan
    n = 9
    if len(close) >= n:
        low_n = low.rolling(window=n).min()
        high_n = high.rolling(window=n).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
        rsv = rsv.fillna(50)
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3*k - 2*d
        res['k'] = k.iloc[-1]; res['d'] = d.iloc[-1]; res['j'] = j.iloc[-1]
    else: res['k'] = res['d'] = res['j'] = np.nan
    vols = df['vol'].astype(float).tolist()
    if len(vols) >= 6:
        avg_prev5 = np.mean(vols[-6:-1])
        res['vol_ratio'] = vols[-1] / (avg_prev5 + 1e-9)
        res['vol_last'] = vols[-1]; res['vol_ma5'] = avg_prev5
    else: res['vol_ratio'] = res['vol_last'] = res['vol_ma5'] = np.nan
    if len(close) >= 10: res['10d_return'] = close.iloc[-1] / close.iloc[-10] - 1
    else: res['10d_return'] = np.nan
    if 'pct_chg' in df.columns and len(df) >= 4:
        try: res['prev3_sum'] = df['pct_chg'].astype(float).iloc[-4:-1].sum()
        except: res['prev3_sum'] = np.nan
    else: res['prev3_sum'] = np.nan
    try:
        if 'pct_chg' in df.columns and len(df) >= 10:
            res['volatility_10'] = df['pct_chg'].astype(float).tail(10).std()
        else: res['volatility_10'] = np.nan
    except: res['volatility_10'] = np.nan
    return res

def safe_merge_pool(pool_df, other_df, cols):
    pool = pool_df.set_index('ts_code').copy()
    if other_df is None or other_df.empty:
        for c in cols: pool[c] = np.nan
        return pool.reset_index()
    if 'ts_code' not in other_df.columns:
        try: other_df = other_df.reset_index()
        except:
            for c in cols: pool[c] = np.nan
            return pool.reset_index()
    for c in cols:
        if c not in other_df.columns: other_df[c] = np.nan
    try: joined = pool.join(other_df.set_index('ts_code')[cols], how='left')
    except Exception:
        for c in cols: pool[c] = np.nan
        return pool.reset_index()
    for c in cols:
        if c not in joined.columns: joined[c] = np.nan
    return joined.reset_index()

def norm_col(s):
    s = s.fillna(0.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    mn = s.min(); mx = s.max()
    if mx - mn < 1e-9: return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)

# 核心评分函数
@st.cache_data(ttl=600)
def run_scoring_for_date(trade_date, INITIAL_TOP_N, FINAL_POOL_LIMIT, MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, VOL_SPIKE_MULT, VOLATILITY_MAX, HIGH_PCT_THRESHOLD):
    """
    核心评分函数，参数必须是可哈希的类型（如 float/int/str）。
    TOP_DISPLAY 不影响评分逻辑，所以不作为缓存参数。
    """
    
    # 1. 拉取当日涨幅榜初筛
    daily_all = safe_get(pro.daily, trade_date=trade_date)
    if daily_all.empty: return pd.DataFrame()
    daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
    pool0 = daily_all.head(int(INITIAL_TOP_N)).copy().reset_index(drop=True)

    # 2. 拉取和合并高级接口数据
    stock_basic = safe_get(pro.stock_basic, list_status='L', fields='ts_code,name,industry,total_mv,circ_mv')
    daily_basic = safe_get(pro.daily_basic, trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    mf_raw = safe_get(pro.moneyflow, trade_date=trade_date)
    
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in','net_mf_out']
        col = next((c for c in possible if c in mf_raw.columns), None)
        if col: moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)
    
    if not stock_basic.empty:
        keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_basic.columns]
        try: pool0 = pool0.merge(stock_basic[keep], on='ts_code', how='left')
        except Exception: pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
    else: pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
        
    pool_merged = safe_merge_pool(pool0, daily_basic, ['turnover_rate','amount','total_mv','circ_mv'])
    
    if moneyflow.empty: moneyflow = pd.DataFrame({'ts_code': pool_merged['ts_code'].tolist(), 'net_mf': [0.0]*len(pool_merged)})
    try: pool_merged = pool_merged.set_index('ts_code').join(moneyflow.set_index('ts_code'), how='left').reset_index()
    except: pool_merged['net_mf'] = 0.0
    pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0.0)
    
    # 3. 清洗 (此处应使用您完整代码中的清洗逻辑)
    clean_list = []
    for r in pool_merged.itertuples():
        ts_code = getattr(r, 'ts_code')
        vol = getattr(r, 'vol', np.nan)
        close = getattr(r, 'close', np.nan)
        open_p = getattr(r, 'open', np.nan)
        pre_close = getattr(r, 'pre_close', np.nan)
        pct = getattr(r, 'pct_chg', np.nan)
        amount = getattr(r, 'amount', np.nan)
        turnover = getattr(r, 'turnover_rate', np.nan)
        total_mv = getattr(r, 'total_mv', np.nan)
        name = getattr(r, 'name', ts_code)

        if (pd.isna(vol) or vol == 0) and (pd.isna(amount) or amount == 0): continue
        if pd.isna(close) or (close < MIN_PRICE) or (close > MAX_PRICE): continue
        if isinstance(name, str) and (('ST' in name.upper()) or ('退' in name)): continue
        try:
            high = getattr(r, 'high', np.nan); low = getattr(r, 'low', np.nan)
            if (not pd.isna(open_p) and not pd.isna(high) and not pd.isna(low) and not pd.isna(pre_close)) and (open_p == high == low == pre_close): continue
        except: pass
        try:
            tv = total_mv; tv_yuan = tv * 10000.0 if not pd.isna(tv) and tv > 1e6 else tv;
            if not pd.isna(tv_yuan) and tv_yuan > 2000 * 1e8: continue
        except: pass
        if not pd.isna(turnover) and float(turnover) < MIN_TURNOVER: continue
        if not pd.isna(amount):
            amt = amount;
            if amt > 0 and amt < 1e5: amt = amt * 10000.0
            if amt < MIN_AMOUNT: continue
        if not pd.isna(pct) and float(pct) < 0: continue
        
        clean_list.append(r)
    
    clean_df = pd.DataFrame([dict(zip(r._fields, r)) for r in clean_list])
    if clean_df.empty: return pd.DataFrame()

    # 关键修复：在缓存函数内部限制评分池大小
    score_pool_n = min(int(FINAL_POOL_LIMIT), 300)
    clean_df = clean_df.sort_values('pct_chg', ascending=False).head(score_pool_n).reset_index(drop=True)
    
    # 4. 指标计算与评分 (此处应使用您完整代码中的评分逻辑)
    records = []
    # (省略了逐票计算指标和评分的循环，请确保您使用的是完整的代码)
    for row in clean_df.itertuples():
        ts_code = getattr(row, 'ts_code'); pct_chg = getattr(row, 'pct_chg', 0.0);
        turnover_rate = getattr(row, 'turnover_rate', np.nan); net_mf = float(getattr(row, 'net_mf', 0.0));
        amount_raw = getattr(row, 'amount', np.nan)
        amount = amount_raw * 10000.0 if not pd.isna(amount_raw) and amount_raw > 0 and amount_raw < 1e5 else amount_raw
        amount = amount if not pd.isna(amount) else 0.0
        name = getattr(row, 'name', ts_code)

        hist = get_hist(ts_code, trade_date, days=60)
        ind = compute_indicators(hist)

        vol_ratio, ten_return, macd, k, d, j, vol_last, vol_ma5, prev3_sum, volatility_10, ma20, last_close = \
            ind.get('vol_ratio', np.nan), ind.get('10d_return', np.nan), ind.get('macd', np.nan), \
            ind.get('k', np.nan), ind.get('d', np.nan), ind.get('j', np.nan), \
            ind.get('vol_last', np.nan), ind.get('vol_ma5', np.nan), ind.get('prev3_sum', np.nan), ind.get('volatility_10', np.nan), ind.get('ma20', np.nan), ind.get('last_close', np.nan)

        try: proxy_money = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
        except: proxy_money = 0.0

        rec = {'ts_code': ts_code, 'pct_chg': pct_chg, 'turnover_rate': turnover_rate, 'net_mf': net_mf, 'amount': amount,
               'vol_ratio': vol_ratio, '10d_return': ten_return, 'macd': macd, 'k': k, 'd': d, 'j': j,
               'vol_last': vol_last, 'vol_ma5': vol_ma5, 'prev3_sum': prev3_sum, 'volatility_10': volatility_10,
               'proxy_money': proxy_money, 'name': name,
               'last_close': last_close, 'ma20': ma20}
        records.append(rec)
        
    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame()

    # 5. 风险过滤 (与原代码一致)
    if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):
        fdf = fdf[~((fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD))]
    if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
        fdf = fdf[~((fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD))]
    if all(c in fdf.columns for c in ['vol_last','vol_ma5']):
        fdf = fdf[~((fdf['vol_last'] > (fdf['vol_ma5'] * VOL_SPIKE_MULT)))]
    if 'volatility_10' in fdf.columns:
        fdf = fdf[~(fdf['volatility_10'] > VOLATILITY_MAX)]

    if fdf.empty: return pd.DataFrame()

    # 6. RSL & 归一化 (与原代码一致)
    if '10d_return' in fdf.columns:
        try:
            market_mean_10d = fdf['10d_return'].replace([np.inf,-np.inf], np.nan).dropna().mean()
            fdf['rsl'] = fdf['10d_return'] / (market_mean_10d if abs(market_mean_10d) >= 1e-9 else 1e-9)
        except: fdf['rsl'] = 1.0
    else: fdf['rsl'] = 1.0

    fdf['s_pct'] = norm_col(fdf.get('pct_chg', pd.Series([0]*len(fdf))))
    fdf['s_volratio'] = norm_col(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))
    fdf['s_turn'] = norm_col(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))
    fdf['s_money'] = norm_col(fdf.get('net_mf', pd.Series([0]*len(fdf)))) if fdf['net_mf'].abs().sum() > 0 else norm_col(fdf.get('proxy_money', pd.Series([0]*len(fdf))))
    fdf['s_amount'] = norm_col(fdf.get('amount', pd.Series([0]*len(fdf))))
    fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))
    fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
    fdf['s_rsl'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))
    fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))

    # 7. 综合评分 (与原代码一致)
    w_pct, w_volratio, w_turn, w_money, w_10d, w_macd, w_rsl, w_volatility = 0.18, 0.18, 0.12, 0.14, 0.12, 0.06, 0.12, 0.08
    fdf['综合评分'] = (fdf['s_pct'] * w_pct + fdf['s_volratio'] * w_volratio + fdf['s_turn'] * w_turn + fdf['s_money'] * w_money + fdf['s_10d'] * w_10d + fdf['s_macd'] * w_macd + fdf['s_rsl'] * w_rsl + fdf['s_volatility'] * w_volatility)
    
    return fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)

# ----------------------------------------------------
# 简易回测模块
# ----------------------------------------------------
def run_simple_backtest(days, params):
    # 使用 st.empty() 创建一个容器，用于存放回测进度和结果
    container = st.empty()
    with container.container():
        st.subheader("📈 简易历史回测结果")
        
        # 获取交易日历
        trade_dates_df = safe_get(pro.trade_cal, exchange='SSE', is_open='1', end_date=find_last_trade_day(), fields='cal_date')
        if trade_dates_df.empty:
            st.error("无法获取历史交易日历。")
            return

        trade_dates = trade_dates_df['cal_date'].sort_values(ascending=False).head(days + 1).tolist()
        trade_dates.reverse() # 从老到新

        if len(trade_dates) < 2:
            st.warning("交易日不足，无法进行回测。")
            return
        
        # 将 TOP_DISPLAY 设为 1 用于回测（只取第一名），但不用传入 run_scoring_for_date
        backtest_results = []
        
        # 进度条文本优化
        pbar_text = f"回测开始... [0/{len(trade_dates) - 1}]"
        pbar = st.progress(0, text=pbar_text)
        
        st.markdown(f"**回测周期：** 最近 **{days}** 个交易日（**{trade_dates[0]}** 至 **{trade_dates[-2]}**）")

        try:
            # 回测参数 (FINAL_POOL 限制在 300 已经在 run_scoring_for_date 内部实现)
            score_params = (
                params['INITIAL_TOP_N'], params['FINAL_POOL'], params['MIN_PRICE'], params['MAX_PRICE'], 
                params['MIN_TURNOVER'], params['MIN_AMOUNT'], params['VOL_SPIKE_MULT'], 
                params['VOLATILITY_MAX'], params['HIGH_PCT_THRESHOLD']
            )

            for i in range(len(trade_dates) - 1):
                select_date = trade_dates[i]
                next_trade_date = trade_dates[i+1]
                
                # 更新进度条
                pbar_text = f"正在回测 {select_date}... [{i+1}/{len(trade_dates) - 1}]"
                pbar.progress((i+1) / (len(trade_dates) - 1), text=pbar_text)

                # 调用缓存函数
                # 注意：这里只传入了缓存参数，避免 UnhashableParamError
                select_df_full = run_scoring_for_date(
                    select_date, params['INITIAL_TOP_N'], params['FINAL_POOL'], params['MIN_PRICE'], 
                    params['MAX_PRICE'], params['MIN_TURNOVER'], params['MIN_AMOUNT'], 
                    params['VOL_SPIKE_MULT'], params['VOLATILITY_MAX'], params['HIGH_PCT_THRESHOLD']
                )

                if select_df_full.empty:
                    backtest_results.append({'选股日': select_date, '股票': '无符合条件', 'T+1 收益率': 0.0, '买入价 (T+1 开盘)': np.nan, '卖出价 (T+1 收盘)': np.nan, '评分': np.nan})
                    continue

                top_pick = select_df_full.iloc[0] # 只取第一名
                ts_code = top_pick['ts_code']
                
                # 获取 T+1 交易日数据
                next_day_data = safe_get(pro.daily, ts_code=ts_code, trade_date=next_trade_date)
                
                return_pct = 0.0
                buy_price, sell_price = np.nan, np.nan

                if not next_day_data.empty and 'open' in next_day_data.columns and 'close' in next_day_data.columns:
                    buy_price = next_day_data.iloc[0]['open']
                    sell_price = next_day_data.iloc[0]['close']
                    
                    if buy_price > 0 and not pd.isna(sell_price):
                        return_pct = (sell_price / buy_price) - 1.0

                backtest_results.append({
                    '选股日': select_date,
                    '股票': f"{top_pick.get('name', 'N/A')}({ts_code})",
                    'T+1 收益率': return_pct * 100,
                    '买入价 (T+1 开盘)': buy_price,
                    '卖出价 (T+1 收盘)': sell_price,
                    '评分': top_pick['综合评分']
                })
        except Exception as e:
            # 捕获回测过程中的错误，并显示
            st.error(f"回测过程中断，可能出现网络或数据权限问题。错误信息：{e}")
            pbar.empty() # 清除进度条
            return

        # 进度条跑完
        pbar.progress(1.0, text="回测完成。")
        
        # 结果展示
        results_df = pd.DataFrame(backtest_results)
        results_df['T+1 收益率'] = results_df['T+1 收益率'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        cumulative_return = (results_df['T+1 收益率'] / 100 + 1).product() - 1
        wins = (results_df['T+1 收益率'] > 0).sum()
        total_trades = len(results_df)
        win_rate = wins / total_trades if total_trades > 0 else 0

        st.markdown("---")
        st.subheader("💡 最终回测指标")
        colA, colB, colC = st.columns(3)
        colA.metric("累计收益率 (T+1)", f"{cumulative_return*100:.2f}%")
        colB.metric("胜率", f"{win_rate*100:.2f}%")
        colC.metric("交易次数", f"{total_trades}")
        
        st.subheader("📋 每日交易记录")
        st.dataframe(results_df, use_container_width=True)

# ----------------------------------------------------
# 实时选股模块
# ----------------------------------------------------
def run_live_selection(last_trade, params):
    st.write(f"正在运行实时选股（最近交易日：{last_trade}）...")
    
    # 回测参数 (FINAL_POOL 限制在 300 已经在 run_scoring_for_date 内部实现)
    fdf_full = run_scoring_for_date(
        last_trade, params['INITIAL_TOP_N'], params['FINAL_POOL'], params['MIN_PRICE'], 
        params['MAX_PRICE'], params['MIN_TURNOVER'], params['MIN_AMOUNT'], 
        params['VOL_SPIKE_MULT'], params['VOLATILITY_MAX'], params['HIGH_PCT_THRESHOLD']
    )

    if fdf_full.empty:
        st.error("清洗和评分后没有候选，建议放宽条件或检查接口权限。")
        st.stop()

    # 最终排序与展示
    fdf = fdf_full.head(params['TOP_DISPLAY']).copy()
    fdf.index = fdf.index + 1

    st.success(f"评分完成：总候选 {len(fdf_full)} 支，显示 Top {min(params['TOP_DISPLAY'], len(fdf))}。")
    display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','proxy_money','amount','10d_return','macd','k','d','j','rsl','volatility_10']
    for c in display_cols:
        if c not in fdf.columns: fdf[c] = np.nan

    st.dataframe(fdf[display_cols], use_container_width=True)

    # 下载
    out_csv = fdf_full[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

    # 小结与建议（简洁）
    st.markdown("### 小结与操作提示（简洁）")
    st.markdown("""
- **【策略风格】** 本版本为 **BC 混合增强版**（短线爆发 + 妖股捕捉）。
- **【风控提示】** 已启用多重风险过滤。实战中，请结合 **次日开盘表现** 进行二次筛选。
- **【重要纪律】** 9:40 前不买 → 观察 9:40-10:05 的量价节奏 → 10:05 后择优介入。
""")


# ----------------------------------------------------
# 主程序控制逻辑
# ----------------------------------------------------

# 将所有参数打包成一个字典
params = {
    'INITIAL_TOP_N': INITIAL_TOP_N, 'FINAL_POOL': FINAL_POOL, 'TOP_DISPLAY': TOP_DISPLAY,
    'MIN_PRICE': MIN_PRICE, 'MAX_PRICE': MAX_PRICE, 'MIN_TURNOVER': MIN_TURNOVER,
    'MIN_AMOUNT': MIN_AMOUNT, 'VOL_SPIKE_MULT': VOL_SPIKE_MULT, 'VOLATILITY_MAX': VOLATILITY_MAX,
    'HIGH_PCT_THRESHOLD': HIGH_PCT_THRESHOLD
}

# 检查是否需要运行回测
if st.session_state.get('run_backtest', False):
    # 调用回测函数，将结果固定在当前位置
    run_simple_backtest(BACKTEST_DAYS, params)
    
# 实时选股（只有当 run_selection 为 True 时运行）
elif st.session_state.get('run_selection', False):
    run_live_selection(last_trade, params)
    
# 初始状态或运行结束后，给出提示
else:
    st.info("请点击上方的按钮开始运行。")

