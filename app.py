import tushare as ts
import pandas as pd
import requests
import datetime
import time
import streamlit as st 

# ==========================================
# 核心模块：实时数据与指标计算
# ==========================================
def get_sina_realtime_kline(ts_code):
    """获取新浪实时行情，用于盘中缝合"""
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
            'vol': (float(data_list[8]) / 100) * (240 / 225) # 尾盘预估全天量
        }
    except Exception:
        return None

def calc_indicators(df):
    """统一指标计算引擎"""
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
    """双重名单强制过滤：锁定核心赛道，封杀毒药板块"""
    stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    
    # 核心资产白名单关键字 (涵盖科技、医药、军工、机械设备)
    core_kw = '电子|半导体|元器件|通信|计算机|软件|电脑|IT|医药|医疗|制药|军工|航空|航天|机械|机床|仪器'
    # 绝对禁碰黑名单关键字 (坚决剔除汽车、证券、酒、钢铁等)
    toxic_kw = '汽车|证券|酒|钢铁|煤炭|银行|保险|房地产|金融|建筑|交运|食品'
    
    # 执行过滤：必须匹配白名单，且绝对不能包含黑名单
    filtered = stock_basic[
        stock_basic['industry'].str.contains(core_kw, na=False) & 
        ~stock_basic['industry'].str.contains(toxic_kw, na=False)
    ]

    trade_cal = pro.trade_cal(exchange='', is_open='1', start_date='20260701', end_date=datetime.datetime.now().strftime('%Y%m%d'))
    last_trade_date = trade_cal.iloc[-1]['cal_date']

    daily_basic = pro.daily_basic(ts_code='', trade_date=last_trade_date, fields='ts_code,circ_mv')
    # 严格锁定 200亿 到 1000亿 流通市值
    daily_basic = daily_basic[(daily_basic['circ_mv'] >= 2000000) & (daily_basic['circ_mv'] <= 10000000)]
    
    return pd.merge(filtered, daily_basic, on='ts_code')

# ==========================================
# 网页界面与侧边栏 (Streamlit UI)
# ==========================================
st.set_page_config(page_title="V38.4 量化雷达", page_icon="🚀", layout="wide")
st.title("🚀 V38.4 终极双轨弹性量化系统")

# 侧边栏控制中枢
st.sidebar.header("⚙️ 战术控制台")
user_token = st.sidebar.text_input("🔑 Tushare Token (万分权限):", type="password")
backtest_days = st.sidebar.number_input("📅 回测天数 (设为 1 即为实时雷达)", min_value=1, max_value=300, value=1)
max_stocks = st.sidebar.number_input("🎯 最大选出数量", min_value=1, max_value=20, value=3)
run_button = st.sidebar.button("▶️ 启动引擎")

st.markdown("---")

if run_button and user_token:
    ts.set_token(user_token)
    try:
        pro = ts.pro_api()
        mode_text = "【实盘雷达模式】" if backtest_days == 1 else f"【历史回测模式 - 过去 {backtest_days} 天】"
        st.info(f"✅ 系统就绪，正在启动 {mode_text}...")
        
        # 获取纯净股票池
        candidate_pool = get_core_stock_pool(pro)
        stock_list = candidate_pool['ts_code'].tolist()
        total_count = len(stock_list)
        
        st.write(f"🛡️ 经过严苛的市值与黑白名单过滤，成功锁定 **{total_count}** 只硬核资产，开始全盘扫描...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        backtest_results = []
        radar_signals = []
        
        start_date = (datetime.datetime.now() - datetime.timedelta(days=backtest_days * 2 + 100)).strftime('%Y%m%d')
        
        for i, ts_code in enumerate(stock_list):
            try:
                stock_name = candidate_pool[candidate_pool['ts_code'] == ts_code]['name'].values[0]
                progress_bar.progress((i + 1) / total_count)
                status_text.text(f"🔍 扫描中 [{i+1}/{total_count}]: {stock_name} ({ts_code})")
                
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=datetime.datetime.now().strftime('%Y%m%d'))
                if df.empty: continue
                
                # 如果是实盘雷达 (回测天数=1)，强行抓取并缝合新浪实时数据
                if backtest_days == 1:
                    today_data = get_sina_realtime_kline(ts_code)
                    if today_data:
                        df_today = pd.DataFrame([today_data])
                        df = pd.concat([df_today, df], ignore_index=True)
                
                df = calc_indicators(df)
                if len(df) < 30: continue
                
                hard_sl_rate = 0.88 if ts_code.startswith('300') or ts_code.startswith('688') else 0.92
                
                # 确定循环区间
                eval_start = len(df) - backtest_days
                eval_start = max(30, eval_start)
                
                for j in range(eval_start, len(df)):
                    current = df.iloc[j]
                    prev = df.iloc[j-1]
                    
                    # 价格铁闸门：>= 20元
                    if current['close'] < 20: continue
                    
                    total_range = current['high'] - current['low']
                    if total_range == 0: continue
                    body_range = current['close'] - current['open']
                    
                    # 四重入场铁律
                    cond_ma20 = current['close'] > current['MA20']
                    cond_body = (body_range > 0) and ((body_range / total_range) > 0.6)
                    cond_macd = (current['DIF'] > 0) and (current['MACD'] > prev['MACD'])
                    cond_vol = current['vol'] > (current['MA5_Vol'] * 1.2)
                    
                    if cond_ma20 and cond_body and cond_macd and cond_vol:
                        buy_price = current['close']
                        buy_date = current['trade_date']
                        buy_low = current['low']
                        
                        # 如果是历史回测，执行推演
                        if j < len(df) - 1:
                            w1_ret, w2_ret, w3_ret = None, None, None
                            sell_price = 0
                            sell_reason = "持仓中(未达3周)"
                            hold_days = 0
                            max_profit = 0
                            
                            for k in range(j+1, len(df)):
                                future_k = df.iloc[k]
                                hold_days += 1
                                
                                curr_profit = (future_k['high'] - buy_price) / buy_price
                                if curr_profit > max_profit: max_profit = curr_profit
                                
                                # 优先判断止损/卖出条件
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
                                
                                # 记录存活周期的收益 (若提前止损则字段保持空白)
                                if hold_days == 5: w1_ret = round((future_k['close'] - buy_price) / buy_price * 100, 2)
                                if hold_days == 10: w2_ret = round((future_k['close'] - buy_price) / buy_price * 100, 2)
                                if hold_days == 15:
                                    w3_ret = round((future_k['close'] - buy_price) / buy_price * 100, 2)
                                    sell_price = future_k['close']
                                    sell_reason = "完成3周推演"
                                    break
                                    
                            if sell_price > 0:
                                final_profit = round((sell_price - buy_price) / buy_price * 100, 2)
                            else:
                                final_profit = round((df.iloc[-1]['close'] - buy_price) / buy_price * 100, 2)
                                
                            backtest_results.append({
                                '代码': ts_code, '名称': stock_name, '买入日': buy_date, 
                                '买入价': round(buy_price, 2), '最终盈亏(%)': final_profit,
                                'W1收益(%)': w1_ret, 'W2收益(%)': w2_ret, 'W3收益(%)': w3_ret,
                                '出局原因': sell_reason
                            })
                            
                        # 如果是最后一天，记录为实盘信号
                        if j == len(df) - 1:
                            radar_signals.append({
                                '代码': ts_code, '名称': stock_name,
                                '现价': round(buy_price, 2), '防守线': round(current['MA20'], 2)
                            })
                time.sleep(0.06)
            except Exception:
                continue
                
        status_text.text("✅ 全盘扫描与推演完成！")
        st.markdown("---")
        
        # =================输出层=================
        if backtest_days == 1:
            st.subheader("🎯 今日实时雷达狙击名单")
            if radar_signals:
                # 按照用户设定的最大数量截取
                display_signals = radar_signals[:max_stocks]
                st.warning(f"【免死金牌提示】：重点关注现价 > 40元 的标的！(已根据侧边栏限制展示前 {len(display_signals)} 只)")
                for s in display_signals:
                    st.success(f"🔥 **{s['名称']} ({s['代码']})** | 现价: **{s['现价']}** 元 | 防守线(MA20): {s['防守线']} 元")
            else:
                st.info("⚠️ 经受住抛压的个股为零，今日无标的满足满格大阳线条件，严格执行空仓。")
                
        else:
            st.subheader(f"📊 历史回测报表 ({backtest_days}天)")
            if backtest_results:
                res_df = pd.DataFrame(backtest_results)
                # 填充空白以匹配断头行情
                res_df = res_df.fillna("") 
                win_rate = round(len(res_df[res_df['最终盈亏(%)'] > 0]) / len(res_df) * 100, 2)
                
                col1, col2 = st.columns(2)
                col1.metric(label="总体胜率 (>0收益)", value=f"{win_rate}%")
                col2.metric(label="触发总交易次数", value=f"{len(res_df)} 次")
                
                st.markdown("#### 📜 详细交易与周期推演清单 (空白表示中途触发执行条件夭折)")
                st.dataframe(res_df.head(max_stocks * backtest_days)) # 动态展示限制
            else:
                st.warning("⚠️ 选定周期内，未触发任何一笔符合全维度铁律的交易。")

    except Exception as e:
        st.error(f"❌ 运行报错，请检查网络或 Token。错误信息：{str(e)}")
elif run_button and not user_token:
    st.error("⚠️ 请在侧边栏输入 Tushare Token。")
