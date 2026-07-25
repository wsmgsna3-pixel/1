import tushare as ts
import pandas as pd
import requests
import datetime
import time
import sys
import os

# ==========================================
# 通用基础模块：指标计算与实时数据抓取
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
            'vol': (float(data_list[8]) / 100) * (240 / 225) # 14:45预估全天量
        }
    except Exception:
        return None

def calc_indicators(df):
    """统一的指标计算引擎"""
    df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA5_Vol'] = df['vol'].rolling(window=5).mean()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    return df

def get_core_stock_pool(pro):
    """获取严格过滤后的股票池 (板块 + 200亿-1000亿市值)"""
    stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    # 剔除汽车，加入机械设备
    target_industries = ['电子', '计算机', '通信', '医药生物', '国防军工', '机械设备']
    filtered_stocks = stock_basic[stock_basic['industry'].isin(target_industries)]

    # 获取最近一个交易日的日期来查市值
    trade_cal = pro.trade_cal(exchange='', is_open='1', start_date='20260701', end_date=datetime.datetime.now().strftime('%Y%m%d'))
    last_trade_date = trade_cal.iloc[-1]['cal_date']

    daily_basic = pro.daily_basic(ts_code='', trade_date=last_trade_date, fields='ts_code,circ_mv')
    # 200亿 到 1000亿 流通市值
    daily_basic = daily_basic[(daily_basic['circ_mv'] >= 2000000) & (daily_basic['circ_mv'] <= 10000000)]
    
    return pd.merge(filtered_stocks, daily_basic, on='ts_code')

# ==========================================
# 核心功能一：实盘雷达引擎 (14:45 运行)
# ==========================================
def run_realtime_radar(pro):
    print("\n" + "="*60)
    print("🛡️  启动 [实盘雷达模式] - 每日 14:45 运行")
    print("="*60)
    
    # 网络连通性测试
    sys.stdout.write("📡 正在测试实时数据源连通性... ")
    test_data = get_sina_realtime_kline('000001.SZ')
    if test_data and test_data['close'] > 0:
        print(f"✅ 成功 (平安银行现价: {test_data['close']} 元)\n")
    else:
        print("❌ 失败，请检查网络！\n")
        return

    candidate_pool = get_core_stock_pool(pro)
    stock_list = candidate_pool['ts_code'].tolist()
    total_count = len(stock_list)
    final_signals = []

    print(f"🚀 锁定 {total_count} 只核心赛道资产，开始实时扫描...")
    
    for i, ts_code in enumerate(stock_list):
        try:
            stock_name = candidate_pool[candidate_pool['ts_code'] == ts_code]['name'].values[0]
            sys.stdout.write(f"\r🔍 正在扫描 [{i+1}/{total_count}]: {stock_name} ({ts_code})    ")
            sys.stdout.flush()

            df_hist = pro.daily(ts_code=ts_code, start_date='20260501', end_date=datetime.datetime.now().strftime('%Y%m%d'))
            if df_hist.empty: continue
            
            today_data = get_sina_realtime_kline(ts_code)
            if today_data is None or today_data['close'] < 20: continue # 价格铁闸门 >= 20元

            df_today = pd.DataFrame([today_data])
            df = pd.concat([df_today, df_hist], ignore_index=True)
            df = calc_indicators(df)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 条件判定
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
                print(f"\n🔥 [捕获信号] {stock_name} ({ts_code}) | 现价: {round(latest['close'], 2)} 突破！")
            
            time.sleep(0.06) # Tushare 接口防封限制
        except Exception:
            continue

    print("\n\n" + "="*60)
    print("🎯 V38.4 尾盘终极狙击名单 (操作建议：14:50 确认)")
    print("="*60)
    if final_signals:
        print("【免死金牌提示】：重点关注现价 > 40元 的标的！\n")
        for s in final_signals:
            print(f"✅ {s['名称']} ({s['代码']}) - 现价: {s['现价']} 元 | 防守线: {s['MA20防守']} 元")
    else:
        print("⚠️ 今日无标的满足饱满实体大阳线，严格空仓。")
    print("="*60 + "\n")

# ==========================================
# 核心功能二：历史回测引擎
# ==========================================
def run_backtest_engine(pro):
    print("\n" + "="*60)
    print("📊 启动 [历史回测模式] - 验证 V38.4 市场适应性")
    print("="*60)
    
    start_date = '20250101' # 可在此处修改回测起点
    print(f"⏳ 正在拉取数据，回测区间起点：{start_date} ...\n")
    
    candidate_pool = get_core_stock_pool(pro)
    stock_list = candidate_pool['ts_code'].tolist()
    total_count = len(stock_list)
    
    backtest_results = []
    
    for i, ts_code in enumerate(stock_list):
        try:
            stock_name = candidate_pool[candidate_pool['ts_code'] == ts_code]['name'].values[0]
            sys.stdout.write(f"\r🔄 回测进度 [{i+1}/{total_count}]: {stock_name} ({ts_code})    ")
            sys.stdout.flush()
            
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=datetime.datetime.now().strftime('%Y%m%d'))
            if len(df) < 60: continue
            
            df = calc_indicators(df)
            
            # 判断标的属性设置强制止损比例
            hard_sl_rate = 0.88 if ts_code.startswith('300') or ts_code.startswith('688') else 0.92
            
            for j in range(30, len(df) - 15): # 预留15天用于走势推演
                current = df.iloc[j]
                prev = df.iloc[j-1]
                
                # 买入条件判断
                if current['close'] < 20: continue
                total_range = current['high'] - current['low']
                if total_range == 0: continue
                body_range = current['close'] - current['open']
                
                cond_ma20 = current['close'] > current['MA20']
                cond_body = (body_range > 0) and ((body_range / total_range) > 0.6)
                cond_macd = (current['DIF'] > 0) and (current['MACD'] > prev['MACD'])
                cond_vol = current['vol'] > (current['MA5_Vol'] * 1.2)
                
                if cond_ma20 and cond_body and cond_macd and cond_vol:
                    # 记录买入信息 (假设 14:50 买入，买入价即收盘价)
                    buy_price = current['close']
                    buy_date = current['trade_date']
                    buy_low = current['low'] # 大阳线最低价用于假突破防守
                    
                    # 模拟持仓推演 (未来 15 个交易日)
                    sell_price = 0
                    sell_reason = ""
                    hold_days = 0
                    max_profit = 0
                    
                    for k in range(j+1, j+16):
                        if k >= len(df): break
                        future_k = df.iloc[k]
                        hold_days += 1
                        
                        # 记录期间最高浮盈
                        curr_profit = (future_k['high'] - buy_price) / buy_price
                        if curr_profit > max_profit: max_profit = curr_profit
                        
                        # 1. 灾难防线：盘中强制止损 (-8% / -12%)
                        if future_k['low'] <= buy_price * hard_sl_rate:
                            sell_price = buy_price * hard_sl_rate
                            sell_reason = "触及强制止损"
                            break
                            
                        # 2. 逻辑防线：收盘假突破止损
                        if future_k['close'] < buy_low:
                            sell_price = future_k['close']
                            sell_reason = "跌破大阳最低价"
                            break
                            
                        # 3. 动态止盈：二档防守 (盈利>15%，破MA10卖出)
                        if max_profit >= 0.15 and future_k['close'] < future_k['MA10']:
                            sell_price = future_k['close']
                            sell_reason = "15%二档止盈(破MA10)"
                            break
                            
                        # 4. 动态防守：一档防守 (盈利>10%，破MA20卖出)
                        if max_profit >= 0.10 and max_profit < 0.15 and future_k['close'] < future_k['MA20']:
                            sell_price = future_k['close']
                            sell_reason = "10%一档保本(破MA20)"
                            break
                            
                        # 若熬过 15 天未触发任何卖出条件，按期末收盘价结算
                        if hold_days == 15:
                            sell_price = future_k['close']
                            sell_reason = "完成3周推演持仓"
                            
                    if sell_price > 0:
                        profit_rate = round((sell_price - buy_price) / buy_price * 100, 2)
                        backtest_results.append({
                            '代码': ts_code, '名称': stock_name,
                            '买入日期': buy_date, '买入价': round(buy_price, 2),
                            '卖出价': round(sell_price, 2), '盈亏幅度(%)': profit_rate,
                            '持仓天数': hold_days, '出局原因': sell_reason
                        })
                        
            time.sleep(0.06)
        except Exception:
            continue
            
    print("\n\n✅ 回测推演完成！正在生成分析报表...")
    
    if backtest_results:
        res_df = pd.DataFrame(backtest_results)
        win_rate = round(len(res_df[res_df['盈亏幅度(%)'] > 0]) / len(res_df) * 100, 2)
        total_trades = len(res_df)
        
        # 导出 Excel 报表
        file_name = f"V38_4_回测报告_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        res_df.to_excel(file_name, index=False)
        
        print("\n" + "="*45)
        print("📊 回测总结报表")
        print("="*45)
        print(f"总计触发买入信号：{total_trades} 次")
        print(f"整体胜率 (>0收益)：{win_rate}%")
        print(f"详细交易记录已保存至当前目录: {file_name}")
        print("="*45 + "\n")
    else:
        print("⚠️ 选定周期内未触发任何符合所有条件的交易信号。")

# ==========================================
# 终端控制台菜单
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("🚀 欢迎使用 V38.4 终极双轨弹性量化系统")
    print("="*60)
    
    user_token = input("🔑 请输入您的 Tushare Token (输入后按回车): ").strip()
    ts.set_token(user_token)
    pro = ts.pro_api()
    
    while True:
        print("\n请选择您要运行的模式：")
        print("  [1] ⚡ 实盘雷达模式 (每日 14:45 运行，输出今日买入信号)")
        print("  [2] 📊 历史回测模式 (盘后运行，验证策略在近期行情的胜率)")
        print("  [0] ❌ 退出系统")
        
        choice = input("\n👉 请输入对应数字 (1/2/0): ").strip()
        
        if choice == '1':
            run_realtime_radar(pro)
        elif choice == '2':
            run_backtest_engine(pro)
        elif choice == '0':
            print("\n系统已安全退出，祝实盘顺利。")
            sys.exit()
        else:
            print("⚠️ 输入无效，请重新选择。")
