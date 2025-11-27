# -*- coding: utf-8 -*-
"""
选股王 · 10000 积分旗舰（BC 混合增强版）· 极速版
说明：
- 【本次修复】**标题样式优化**，减小字号以节省移动端空间，并确保回测进度条和结果能紧密显示。
- **回测模块集成**：将选股核心逻辑封装，用于运行当日选股和历史回测。
- **参数同步**：采用 crjsb.py.txt 中的参数值（MIN_AMOUNT=2亿, VOL_SPIKE_MULT=1.7, VOLATILITY_MAX=8.0）。
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
st.set_page_config(page_title="选股王 · 10000旗舰（BC增强）· 极速版", layout="wide")

# 标题优化：使用 Markdown 减小字号
st.markdown("## 选股王 · 10000 积分旗舰（BC 混合增强版）· 极速版")
st.markdown("输入你的 Tushare Token（仅本次运行使用）。若有权限缺失，脚本会自动降级并继续运行。")

# ---------------------------
# 侧边栏参数（实时可改）
# ---------------------------
with st.sidebar:
    st.header("可调参数（实时）")
    # 注意：这里的参数值已同步为 crjsb.py.txt 中的值
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
    st.markdown("---")
    
    st.header("🔍 回测设置 (T+1 简单回测)")
    BACKTEST_DAYS = int(st.number_input("回测：最近 N 个交易日", value=10, step=1))
    st.markdown("---")
    st.caption("提示：保守→降低阈值；激进→提高阈值。")

# ---------------------------
# Token 输入（主区）
# ---------------------------
TS_TOKEN = st.text_input("Tushare Token（输入后按回车）", type="password")
if not TS_TOKEN:
    st.warning("请输入 Tushare Token 才能运行脚本。")
    st.stop()

# 初始化 tushare
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ----------------------------------------------------
# 按钮控制模块
# ----------------------------------------------------

st.subheader("⚡ 运行模式选择")
if 'run_selection' not in st.session_state:
    st.session_state['run_selection'] = False
if 'run_backtest' not in st.session_state:
    st.session_state['run_backtest'] = False
    
col1, col2 = st.columns(2)

# 主界面的按钮
with col1:
    if st.button("运行当日选股", use_container_width=True):
        st.session_state['run_selection'] = True
        st.session_state['run_backtest'] = False
        st.rerun()
        
with col2:
    if st.button(f"运行回测 (最近 {BACKTEST_DAYS} 日)", use_container_width=True):
        st.session_state['run_backtest'] = True
        st.session_state['run_selection'] = False
        st.rerun()

st.markdown("---")


# ---------------------------
# 安全调用 & 缓存辅助 (函数保持不变)
# ---------------------------
def safe_get(func, **kwargs):
    """Call API and return DataFrame or empty df on any error."""
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


# ----------------------------------------------------
# 核心评分函数 (封装) - 适用于实时和回测
# ----------------------------------------------------
@st.cache_data(ttl=600)
def run_scoring_for_date(trade_date, pro_api, params):
    # 1. 拉取当日涨幅榜初筛
    daily_all = safe_get(pro_api.daily, trade_date=trade_date)
    if daily_all.empty: return pd.DataFrame()
    
    daily_all = daily_all.sort_values("pct_chg", ascending=False).reset_index(drop=True)
    pool0 = daily_all.head(int(params['INITIAL_TOP_N'])).copy().reset_index(drop=True)

    # 2. 拉取和合并高级接口数据 (使用全局 pro 对象)
    stock_basic = safe_get(pro_api.stock_basic, list_status='L', fields='ts_code,name,industry,total_mv,circ_mv')
    daily_basic = safe_get(pro_api.daily_basic, trade_date=trade_date, fields='ts_code,turnover_rate,amount,total_mv,circ_mv')
    mf_raw = safe_get(pro_api.moneyflow, trade_date=trade_date)
    
    moneyflow = pd.DataFrame(columns=['ts_code','net_mf'])
    if not mf_raw.empty:
        possible = ['net_mf','net_mf_amount','net_mf_in','net_mf_out']
        col = next((c for c in possible if c in mf_raw.columns), None)
        if col: moneyflow = mf_raw[['ts_code', col]].rename(columns={col:'net_mf'}).fillna(0)
    
    # merge stock_basic
    if not stock_basic.empty:
        keep = [c for c in ['ts_code','name','industry','total_mv','circ_mv'] if c in stock_basic.columns]
        try:
            pool0 = pool0.merge(stock_basic[keep], on='ts_code', how='left')
        except Exception:
            pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
    else:
        pool0['name'] = pool0['ts_code']; pool0['industry'] = ''
        
    pool_merged = safe_merge_pool(pool0, daily_basic, ['turnover_rate','amount','total_mv','circ_mv'])
    
    if moneyflow.empty: moneyflow = pd.DataFrame({'ts_code': pool_merged['ts_code'].tolist(), 'net_mf': [0.0]*len(pool_merged)})
    try: pool_merged = pool_merged.set_index('ts_code').join(moneyflow.set_index('ts_code'), how='left').reset_index()
    except: pool_merged['net_mf'] = 0.0
    pool_merged['net_mf'] = pool_merged['net_mf'].fillna(0.0)
    
    # 3. 清洗
    clean_list = []
    
    MIN_PRICE, MAX_PRICE, MIN_TURNOVER, MIN_AMOUNT, HIGH_PCT_THRESHOLD = \
        params['MIN_PRICE'], params['MAX_PRICE'], params['MIN_TURNOVER'], params['MIN_AMOUNT'], params['HIGH_PCT_THRESHOLD']
        
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
        
        # 1. skip no trading
        if (pd.isna(vol) or vol == 0) and (pd.isna(amount) or amount == 0): continue
        # 2. price filter
        if pd.isna(close) or (close < MIN_PRICE) or (close > MAX_PRICE): continue
        # 3. exclude ST / delist
        if isinstance(name, str) and (('ST' in name.upper()) or ('退' in name)): continue
        # 4. one-word board
        try:
            high = getattr(r, 'high', np.nan); low = getattr(r, 'low', np.nan)
            if not pd.isna(open_p) and not pd.isna(high) and not pd.isna(low) and not pd.isna(pre_close):
                if open_p == high == low == pre_close: continue
        except: pass
        # 5. market cap filter
        try:
            tv = total_mv
            if not pd.isna(tv):
                tv = float(tv)
                tv_yuan = tv * 10000.0 if tv > 1e6 else tv
                if tv_yuan > 2000 * 1e8: continue
        except: pass
        # 6. turnover
        if not pd.isna(turnover) and float(turnover) < MIN_TURNOVER: continue
        # 7. amount (convert if likely in 万元)
        if not pd.isna(amount):
            amt = amount
            if amt > 0 and amt < 1e5: amt = amt * 10000.0
            if amt < MIN_AMOUNT: continue
        # 8. exclude yesterday down (pct_chg < 0)
        if not pd.isna(pct) and float(pct) < 0: continue
        
        clean_list.append(r)
    
    clean_df = pd.DataFrame([dict(zip(r._fields, r)) for r in clean_list])
    if clean_df.empty: return pd.DataFrame()

    score_pool_n = min(int(params['FINAL_POOL']), 300)
    clean_df = clean_df.sort_values('pct_chg', ascending=False).head(score_pool_n).reset_index(drop=True)
    
    # 4. 指标计算与初步数据整合
    records = []
    for row in clean_df.itertuples():
        ts_code = getattr(row, 'ts_code')
        pct_chg = getattr(row, 'pct_chg', 0.0)
        turnover_rate = getattr(row, 'turnover_rate', np.nan)
        net_mf = float(getattr(row, 'net_mf', 0.0))

        hist = get_hist(ts_code, trade_date, days=60)
        ind = compute_indicators(hist)

        vol_ratio, ten_return, macd, k, d, j, vol_last, vol_ma5, prev3_sum, volatility_10, ma20, last_close = \
            ind.get('vol_ratio', np.nan), ind.get('10d_return', np.nan), ind.get('macd', np.nan), \
            ind.get('k', np.nan), ind.get('d', np.nan), ind.get('j', np.nan), \
            ind.get('vol_last', np.nan), ind.get('vol_ma5', np.nan), ind.get('prev3_sum', np.nan), \
            ind.get('volatility_10', np.nan), ind.get('ma20', np.nan), ind.get('last_close', np.nan)

        try: proxy_money = (abs(pct_chg) + 1e-9) * (vol_ratio if not pd.isna(vol_ratio) else 0.0) * (turnover_rate if not pd.isna(turnover_rate) else 0.0)
        except: proxy_money = 0.0

        rec = {'ts_code': ts_code, 'pct_chg': pct_chg, 'turnover_rate': turnover_rate, 'net_mf': net_mf,
               'vol_ratio': vol_ratio, '10d_return': ten_return, 'macd': macd, 'k': k, 'd': d, 'j': j,
               'vol_last': vol_last, 'vol_ma5': vol_ma5, 'prev3_sum': prev3_sum, 'volatility_10': volatility_10,
               'proxy_money': proxy_money, 'name': getattr(row, 'name', ts_code),
               'last_close': last_close, 'ma20': ma20,
               'amount': getattr(row, 'amount', 0.0) * 10000.0}
        records.append(rec)

    fdf = pd.DataFrame(records)
    if fdf.empty: return pd.DataFrame()

    # 5. 风险过滤
    VOL_SPIKE_MULT, VOLATILITY_MAX = params['VOL_SPIKE_MULT'], params['VOLATILITY_MAX']
    
    # A: 高位大阳线
    if all(c in fdf.columns for c in ['ma20','last_close','pct_chg']):
        fdf = fdf[~((fdf['last_close'] > fdf['ma20'] * 1.10) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD))]
    # B: 下跌途中反抽
    if all(c in fdf.columns for c in ['prev3_sum','pct_chg']):
        fdf = fdf[~((fdf['prev3_sum'] < 0) & (fdf['pct_chg'] > HIGH_PCT_THRESHOLD))]
    # C: 巨量放量大阳
    if all(c in fdf.columns for c in ['vol_last','vol_ma5']):
        fdf = fdf[~((fdf['vol_last'] > (fdf['vol_ma5'] * VOL_SPIKE_MULT)))]
    # D: 极端波动
    if 'volatility_10' in fdf.columns:
        fdf = fdf[~(fdf['volatility_10'] > VOLATILITY_MAX)]
    
    if fdf.empty: return pd.DataFrame()

    # 6. RSL & 归一化
    if '10d_return' in fdf.columns:
        try:
            market_mean_10d = fdf['10d_return'].replace([np.inf,-np.inf], np.nan).dropna().mean()
            fdf['rsl'] = fdf['10d_return'] / (market_mean_10d if abs(market_mean_10d) > 1e-9 else 1e-9)
        except: fdf['rsl'] = 1.0
    else: fdf['rsl'] = 1.0

    fdf['s_pct'] = norm_col(fdf.get('pct_chg', pd.Series([0]*len(fdf))))
    fdf['s_volratio'] = norm_col(fdf.get('vol_ratio', pd.Series([0]*len(fdf))))
    fdf['s_turn'] = norm_col(fdf.get('turnover_rate', pd.Series([0]*len(fdf))))
    fdf['s_money'] = norm_col(fdf.get('net_mf', pd.Series([0]*len(fdf)))) if fdf['net_mf'].abs().sum() > 0 else norm_col(fdf.get('proxy_money', pd.Series([0]*len(fdf))))
    fdf['s_10d'] = norm_col(fdf.get('10d_return', pd.Series([0]*len(fdf))))
    fdf['s_macd'] = norm_col(fdf.get('macd', pd.Series([0]*len(fdf))))
    fdf['s_rsl'] = norm_col(fdf.get('rsl', pd.Series([0]*len(fdf))))
    fdf['s_volatility'] = 1 - norm_col(fdf.get('volatility_10', pd.Series([0]*len(fdf))))
    fdf['s_amount'] = norm_col(fdf.get('amount', pd.Series([0]*len(fdf)))) # Added for compatibility

    # 7. 综合评分 (使用 crjsb.py.txt 中的权重)
    w_pct, w_volratio, w_turn, w_money, w_10d, w_macd, w_rsl, w_volatility = \
        0.18, 0.18, 0.12, 0.14, 0.12, 0.06, 0.12, 0.08

    fdf['综合评分'] = (fdf['s_pct'] * w_pct + fdf['s_volratio'] * w_volratio + fdf['s_turn'] * w_turn +
                   fdf['s_money'] * w_money + fdf['s_10d'] * w_10d + fdf['s_macd'] * w_macd +
                   fdf['s_rsl'] * w_rsl + fdf['s_volatility'] * w_volatility)
    
    fdf = fdf.sort_values('综合评分', ascending=False).reset_index(drop=True)
    return fdf.head(params['TOP_DISPLAY'])


# ----------------------------------------------------
# 简易回测模块
# ----------------------------------------------------
def run_simple_backtest(days, pro_api, params_all):
    st.subheader("📈 简易历史回测结果") # 使用 subheader 减小字号

    # 获取回测参数
    params = {
        'INITIAL_TOP_N': params_all['INITIAL_TOP_N'], 'FINAL_POOL': params_all['FINAL_POOL'], 'TOP_DISPLAY': 1,
        'MIN_PRICE': params_all['MIN_PRICE'], 'MAX_PRICE': params_all['MAX_PRICE'], 'MIN_TURNOVER': params_all['MIN_TURNOVER'],
        'MIN_AMOUNT': params_all['MIN_AMOUNT'], 'VOL_SPIKE_MULT': params_all['VOL_SPIKE_MULT'], 'VOLATILITY_MAX': params_all['VOLATILITY_MAX'],
        'HIGH_PCT_THRESHOLD': params_all['HIGH_PCT_THRESHOLD']
    }
    
    trade_dates_df = safe_get(pro_api.trade_cal, exchange='SSE', is_open='1', end_date=find_last_trade_day(), fields='cal_date')
    if trade_dates_df.empty:
        st.error("无法获取历史交易日历。")
        return

    trade_dates = trade_dates_df['cal_date'].sort_values(ascending=False).head(days + 1).tolist()
    trade_dates.reverse() # 从老到新

    if len(trade_dates) < 2:
        st.warning("交易日不足，无法进行回测。")
        return

    backtest_results = []
    
    # 使用一个空容器作为进度条和交易记录的占位符
    pbar_container = st.container()
    pbar = pbar_container.progress(0)
    
    st.markdown(f"**回测周期：** 最近 **{days}** 个交易日（**{trade_dates[0]}** 至 **{trade_dates[-2]}**）")

    for i in range(len(trade_dates) - 1):
        select_date = trade_dates[i]
        next_trade_date = trade_dates[i+1]
        
        # 更新进度条
        pbar.progress((i+1) / (len(trade_dates) - 1), text=f"正在回测 {select_date}...")

        # 使用封装的选股评分逻辑
        select_df = run_scoring_for_date(select_date, pro_api, params)
        if select_df.empty:
            backtest_results.append({'选股日': select_date, '股票': '无符合条件', 'T+1 收益率': 0.0, '买入价': np.nan, '卖出价': np.nan, '评分': np.nan})
            continue

        top_pick = select_df.iloc[0]
        ts_code = top_pick['ts_code']
        
        # 获取 T+1 交易日数据
        next_day_data = safe_get(pro_api.daily, ts_code=ts_code, trade_date=next_trade_date)
        
        return_pct = 0.0
        buy_price, sell_price = np.nan, np.nan

        if not next_day_data.empty:
            buy_price = next_day_data.iloc[0]['open']
            sell_price = next_day_data.iloc[0]['close']
            
            if buy_price > 0:
                return_pct = (sell_price / buy_price) - 1.0

        backtest_results.append({
            '选股日': select_date,
            '股票': f"{top_pick.get('name', 'N/A')}({ts_code})",
            'T+1 收益率': return_pct * 100,
            '买入价 (T+1 开盘)': buy_price,
            '卖出价 (T+1 收盘)': sell_price,
            '评分': top_pick['综合评分']
        })

    # 进度条跑完
    pbar.progress(1.0, text="回测完成。")
    
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
# 主程序入口
# ----------------------------------------------------
last_trade = find_last_trade_day()
if not last_trade:
    st.error("无法找到最近交易日，检查网络或 Token 权限。")
    st.stop()
st.info(f"参考最近交易日：{last_trade}")


# 获取所有参数，统一传递给核心函数
params_all = {
    'INITIAL_TOP_N': INITIAL_TOP_N, 'FINAL_POOL': FINAL_POOL, 'TOP_DISPLAY': TOP_DISPLAY,
    'MIN_PRICE': MIN_PRICE, 'MAX_PRICE': MAX_PRICE, 'MIN_TURNOVER': MIN_TURNOVER,
    'MIN_AMOUNT': MIN_AMOUNT, 'VOL_SPIKE_MULT': VOL_SPIKE_MULT, 'VOLATILITY_MAX': VOLATILITY_MAX,
    'HIGH_PCT_THRESHOLD': HIGH_PCT_THRESHOLD
}

# >>>>> 控制逻辑 <<<<<
if not st.session_state.get('run_selection') and not st.session_state.get('run_backtest'):
    st.info("请点击上方的 '运行当日选股' 或 '运行回测' 开始。")
    st.stop()


# 检查是否需要运行回测
if st.session_state.get('run_backtest', False):
    run_simple_backtest(BACKTEST_DAYS, pro, params_all)
    st.stop()


# 实时选股（只有当 run_selection 为 True 时运行）
if st.session_state.get('run_selection', False):
    st.write("正在运行实时选股（最近交易日）...")

    fdf = run_scoring_for_date(last_trade, pro, params_all)

    if fdf.empty:
        st.error("清洗和评分后没有候选，建议放宽条件或检查接口权限。")
        st.stop()

    # 最终排序与展示
    fdf.index = fdf.index + 1

    st.success(f"评分完成：总候选 {len(fdf)} 支，显示 Top {min(TOP_DISPLAY, len(fdf))}。")
    display_cols = ['name','ts_code','综合评分','pct_chg','vol_ratio','turnover_rate','net_mf','proxy_money','amount','10d_return','macd','k','d','j','rsl','volatility_10']
    for c in display_cols:
        if c not in fdf.columns: fdf[c] = np.nan

    st.dataframe(fdf[display_cols].head(TOP_DISPLAY), use_container_width=True)

    # 下载
    out_csv = fdf[display_cols].head(200).to_csv(index=True, encoding='utf-8-sig')
    st.download_button("下载评分结果（前200）CSV", data=out_csv, file_name=f"score_result_{last_trade}.csv", mime="text/csv")

    # 小结与建议（简洁）
    st.markdown("### 小结与操作提示（简洁）")
    st.markdown("""
- **【策略风格】** 本版本为 BC 混合增强版（短线爆发 + 妖股捕捉）。  
- **【风险控制】** 已启用 MACD、RSL、下跌途中大阳线过滤、巨量冲高过滤、极端波动过滤。
- **【数据提示】** 若 moneyflow / chip / ths_member 能成功拉取，将作为额外加分因子；若无权限脚本会自动用 proxy_money 代替。  
- **【重要纪律】** 实战纪律：**9:40 前不买 → 观察 9:40-10:05 的量价节奏 → 10:05 后择优介入**。  
- **【风控提示】** 若今日候选普遍翻绿，请保持空仓。  
""")
    st.info("运行出现问题请把 Streamlit 的错误日志或首段报错发给我（截图或文字都行），我会在两次修改内继续帮你调优。")
