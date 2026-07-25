import tushare as ts
import pandas as pd
import requests
import datetime
import time
import streamlit as st # 必须引入网页 UI 库

# ==========================================
# 通用基础模块：指标计算与实时数据抓取
# ==========================================
def get_sina_realtime_kline(ts_code):
    code_split = ts_code.split('.')
    if len(code_split) != 2: return None
    sina_code = code_split[1].lower() + code_split[0]
    
    url = f"http://hq.sinajs.cn/list={sina_code}"
    headers = {'Referer': 'https://finance.sina.com.cn'}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = 'gbk'
        data_str = response.text.split('="')[1].split('";')[0]
        if not data_str: return None
        data_list = data_str.split(',')
        
        return {
            'ts_code': ts_code,
            'trade_date': datetime.datetime.now().strftime('%Y%m%d'),
            'open': float(data_list[1]),
            'high': float(data_list[4]),
            'low': float(data_list[5]),
            'close': float(data_list[3]),
            'vol': (float(data_list[8]) / 100) * (240 / 225) 
        }
    except Exception:
        return None

def calc_indicators(df):
    df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA5_Vol'] = df['vol'].rolling(window=5).mean()
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    return df

def get_core_stock_pool(pro):
    stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    target_industries = ['电子', '计算机', '通信', '医药生物', '国防军工', '机械设备']
    filtered_stocks = stock_basic[stock_basic['industry'].isin(target_industries)]

    trade_cal = pro.trade_cal(exchange='', is_open='1', start_date='20260701', end_date=datetime.datetime.now().strftime('%Y%m%d'))
    last_trade_date = trade_cal.iloc[-1]['cal_date']

    daily_basic = pro.daily_basic(ts_code='', trade_date=last_trade_date, fields='ts_code,circ_mv')
    daily_basic = daily_basic[(daily_basic['circ_mv'] >= 2000000) & (daily_basic['circ_mv'] <= 10000000)]
    
    return pd.merge(filtered_stocks, daily_basic, on='ts_code')

# ==========================================
# 核心功能一：实盘雷达引擎 (网页渲染版)
# ==========================================
def run_realtime_radar_web(pro):
    st.markdown("### 📡 正在测试实时数据源连通性...")
    test_data = get_sina_realtime_kline('000001.SZ')
    if test_data and test_data['close'] > 0:
        st.success(f"✅ 网络通信正常 (平安银行现价: {test_data['close']} 元)")
    else:
        st.error("❌ 失败，请检查网络！")
        return

    candidate_pool = get_core_stock_pool(pro)
    stock_list = candidate_pool['ts_code'].tolist()
    total_count = len(stock_list)
    final_signals = []

    st.info(f"🚀 锁定 {total_count} 只核心赛道资产，开始实时扫描...")
    
    # 网页进度条组件
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, ts_code in enumerate(stock_list):
        try:
            stock_name = candidate_pool[candidate_pool['ts_code'] == ts_code]['name'].values[0]
            # 更新进度条和文本
            progress_bar.progress((i + 1) / total_count)
            status_text.text(f"🔍 正在扫描 [{i+1}/{total_count}]: {stock_name} ({ts_code})")

            df_hist = pro.daily(ts_code=ts_code, start_date='20260501', end_date=datetime.datetime.now().strftime('%Y%m%d'))
            if df_hist.empty: continue
            
            today_data = get_sina_realtime_kline(ts_code)
            if today_data is None or today_data['close'] < 20: continue 

            df_today = pd.DataFrame([today_data])
            df = pd.concat([df_today, df_hist], ignore_index=True)
            df = calc_indicators(df)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            cond_ma20 = latest['close'] > latest['MA20']
            total_range = latest['high'] - latest['low']
            if total_range == 0: continue
            body_range = latest['close'] - latest['open']
            cond_body = (body_range > 0) and ((body_range / total_range) > 0.6)
            cond_macd = (latest['DIF'] > 0) and (latest['MACD'] > prev['MACD'])
            cond_vol = latest['vol'] > (latest['MA5_Vol'] * 1.2)
            
            if cond_ma20 and cond_body and cond_macd and cond_vol:
                final_signals.append({
                    '代码': ts_code, '名称': stock_name,
                    '现价': round(latest['close'], 2), 'MA20防守': round(latest['MA20'], 2)
                })
            
            time.sleep(0.06) 
        except Exception:
            continue

    status_text.text("✅ 扫描完成！")
    st.markdown("---")
    st.markdown("### 🎯 V38.4 尾盘终极狙击名单")
    
    if final_signals:
        st.warning("【免死金牌提示】：重点关注现价 > 40元 的标的！")
        for s in final_signals:
            st.success(f"🔥 **{s['名称']} ({s['代码']})** | 现价: **{s['现价']}** 元 | 防守线: {s['MA20防守']} 元")
    else:
        st.info("⚠️ 今日无标的满足饱满实体大阳线，严格空仓。")

# ==========================================
# 核心功能二：历史回测引擎 (网页渲染版)
# ==========================================
def run_backtest_engine_web(pro):
    start_date = '20250101'
    st.info(f"⏳ 正在拉取数据，回测区间起点：{start_date} ...")
    
    candidate_pool = get_core_stock_pool(pro)
    stock_list = candidate_pool['ts_code'].tolist()
    total_count = len(stock_list)
    backtest_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ts_code in enumerate(stock_list):
        try:
            stock_name = candidate_pool[candidate_pool['ts_code'] == ts_code]['name'].values[0]
            progress_bar.progress((i + 1) / total_count)
            status_text.text(f"🔄 回测进度 [{i+1}/{total_count}]: {stock_name} ({ts_code})")
            
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=datetime.datetime.now().strftime('%Y%m%d'))
            if len(df) < 60: continue
            df = calc_indicators(df)
            
            hard_sl_rate = 0.88 if ts_code.startswith('300') or ts_code.startswith('688') else 0.92
            
            for j in range(30, len(df) - 15):
                current = df.iloc[j]
                prev = df.iloc[j-1]
                
                if current['close'] < 20: continue
                total_range = current['high'] - current['low']
                if total_range == 0: continue
                body_range = current['close'] - current['open']
                
                cond_ma20 = current['close'] > current['MA20']
                cond_body = (body_range > 0) and ((body_range / total_range) > 0.6)
                cond_macd = (current['DIF'] > 0) and (current['MACD'] > prev['MACD'])
                cond_vol = current['vol'] > (current['MA5_Vol'] * 1.2)
                
                if cond_ma20 and cond_body and cond_macd and cond_vol:
                    buy_price = current['close']
                    buy_date = current['trade_date']
                    buy_low = current['low']
                    
                    sell_price = 0
                    sell_reason = ""
                    hold_days = 0
                    max_profit = 0
                    
                    for k in range(j+1, j+16):
                        if k >= len(df): break
                        future_k = df.iloc[k]
                        hold_days += 1
                        
                        curr_profit = (future_k['high'] - buy_price) / buy_price
                        if curr_profit > max_profit: max_profit = curr_profit
                        
                        if future_k['low'] <= buy_price * hard_sl_rate:
                            sell_price = buy_price * hard_sl_rate
                            sell_reason = "触及强制止损"
                            break
                        if future_k['close'] < buy_low:
                            sell_price = future_k['close']
                            sell_reason = "跌破大阳最低价"
                            break
                        if max_profit >= 0.15 and future_k['close'] < future_k['MA10']:
                            sell_price = future_k['close']
                            sell_reason = "15%二档止盈"
                            break
                        if max_profit >= 0.10 and max_profit < 0.15 and future_k['close'] < future_k['MA20']:
                            sell_price = future_k['close']
                            sell_reason = "10%保本"
                            break
                        if hold_days == 15:
                            sell_price = future_k['close']
                            sell_reason = "完成3周推演"
                            
                    if sell_price > 0:
                        profit_rate = round((sell_price - buy_price) / buy_price * 100, 2)
                        backtest_results.append({
                            '名称': stock_name, '买入日': buy_date, '盈亏(%)': profit_rate, '出局原因': sell_reason
                        })
                        
            time.sleep(0.06)
        except Exception:
            continue
            
    status_text.text("✅ 回测推演完成！")
    
    if backtest_results:
        res_df = pd.DataFrame(backtest_results)
        win_rate = round(len(res_df[res_df['盈亏(%)'] > 0]) / len(res_df) * 100, 2)
        
        st.markdown("---")
        st.markdown("### 📊 回测总结报表")
        st.metric(label="整体胜率 (>0收益)", value=f"{win_rate}%")
        st.metric(label="总计触发交易", value=f"{len(res_df)} 次")
        st.dataframe(res_df) # 在网页直接展示表格
    else:
        st.warning("⚠️ 选定周期内未触发任何交易信号。")

# ==========================================
# 网页界面搭建 (Streamlit UI)
# ==========================================
st.set_page_config(page_title="V38.4 量化系统", page_icon="🚀")
st.title("🚀 V38.4 终极双轨弹性量化系统")
st.markdown("---")

# 密码输入框：输入内容会自动变成星号，不会泄露，也不会保存在代码里
user_token = st.text_input("🔑 请粘贴您的 Tushare Token:", type="password")

if user_token:
    ts.set_token(user_token)
    try:
        pro = ts.pro_api()
        st.success("Token 验证通过，系统就绪。")
        
        # 网页单选按钮
        run_mode = st.radio("请选择运行模式：", ["⚡ 实盘雷达模式 (每日 14:45 运行)", "📊 历史回测模式 (验证策略胜率)"])
        
        # 网页触发按钮
        if st.button("▶️ 立即运行"):
            if "雷达" in run_mode:
                run_realtime_radar_web(pro)
            else:
                run_backtest_engine_web(pro)
                
    except Exception as e:
        st.error(f"Token 无效或网络异常，请核对。")
