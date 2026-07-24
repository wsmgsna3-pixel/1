import tushare as ts
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. 初始化设置与参数
# ==========================================
# 请替换为你自己的 Tushare Token
TUSHARE_TOKEN = 'YOUR_TUSHARE_TOKEN'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# 回测时间段（建议选取包含牛熊转换的完整周期，例如最近两年）
START_DATE = '20220101'
END_DATE = '20240101'

# 股票池过滤参数
MIN_PRICE = 10.0          # 股价大于 10 元
MIN_CIRC_MV = 300000      # 流通市值大于 30 亿 (tushare单位为万元, 30亿=300000万)
MAX_CIRC_MV = 5000000     # 流通市值小于 500 亿

def get_stock_pool():
    """获取符合基础条件的股票池"""
    print("正在获取并筛选股票池...")
    # 获取最近一个交易日的每日指标
    df_basic = pro.daily_basic(trade_date=END_DATE)
    if df_basic.empty:
        # 如果END_DATE不是交易日，获取最新交易日数据
        df_basic = pro.daily_basic(trade_date='') 
        
    # 筛选流通市值 30亿 - 500亿，且收盘价 > 10元
    df_filtered = df_basic[(df_basic['circ_mv'] >= MIN_CIRC_MV) & 
                           (df_basic['circ_mv'] <= MAX_CIRC_MV) & 
                           (df_basic['close'] >= MIN_PRICE)]
    
    stock_list = df_filtered['ts_code'].tolist()
    print(f"筛选完毕，共有 {len(stock_list)} 只股票符合基础白名单要求。")
    return stock_list

def get_hist_data(ts_code):
    """获取单只股票历史数据并计算均线"""
    df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
    if df.empty or len(df) < 120:
        return pd.DataFrame()
    
    # Tushare 获取的数据是按日期倒序的，必须按时间正序排列以便回测
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # 计算日线级别的均线和成交量均线
    df['ma5_vol'] = df['vol'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()    # 代表中期/周线趋势
    df['ma120'] = df['close'].rolling(window=120).mean()  # 代表长期趋势
    
    # 剔除均线计算初期的空值
    df = df.dropna().reset_index(drop=True)
    return df

def run_v38_strategy(df, ts_code):
    """V38.0 核心状态机与交易引擎"""
    status = 'EMPTY'
    buy_price = 0.0
    bottom_line = 0.0
    
    ma20_active = False
    ma10_active = False
    
    trade_records = []
    
    # 从第4天开始循环，以确保能读取前几天的状态（用于判断回踩）
    for i in range(4, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        current_close = row['close']
        current_low = row['low']
        current_high = row['high']
        current_vol = row['vol']
        
        current_ma10 = row['ma10']
        current_ma20 = row['ma20']
        
        # ==========================================
        # 🟢 【买入逻辑】: 右侧突围
        # ==========================================
        if status == 'EMPTY':
            # 1. 判定大背景：周线趋势向上 (日线MA60 > MA120)
            if row['ma60'] > row['ma120']:
                
                # 2. 判定回踩：前几天曾在20日线下方 (洗盘动作)
                pulled_back = df.iloc[i-3]['close'] < df.iloc[i-3]['ma20'] and \
                              df.iloc[i-2]['close'] < df.iloc[i-2]['ma20']
                              
                # 3. 判定买点：昨天还在MA20下，今天放量收盘站上MA20
                if pulled_back and prev_row['close'] < prev_row['ma20'] and current_close > current_ma20:
                    if current_vol > row['ma5_vol']: # 必须放量
                        status = 'HOLDING'
                        buy_price = current_close
                        bottom_line = current_low  # 锁定大阳线最低价
                        
                        ma20_active = False
                        ma10_active = False
                        buy_date = row['trade_date']
                        continue 
                        
        # ==========================================
        # 🛡️ 【持仓逻辑】: 三级动态防御体系
        # ==========================================
        elif status == 'HOLDING':
            # --- 第一步：解锁判定 ---
            if not ma10_active:
                current_bias = (current_high - current_ma20) / current_ma20
                if current_bias >= 0.15: # 乖离率达 15%，挂入二档
                    ma10_active = True
                    ma20_active = True
                    
            if not ma20_active and not ma10_active:
                profit_pct = (current_close - buy_price) / buy_price
                key_a = profit_pct >= 0.10             # 利润垫10%
                key_b = current_ma20 > buy_price       # 均线上移越过成本
                if key_a or key_b:
                    ma20_active = True                 # 挂入一档
                    
            # --- 第二步：防守判定 ---
            sell_triggered = False
            sell_reason = ""
            
            if ma10_active:
                if current_close < current_ma10:
                    sell_triggered = True
                    sell_reason = "二档止盈：高位跌破10日线"
            elif ma20_active:
                if current_close < current_ma20:
                    sell_triggered = True
                    sell_reason = "一档保本/止盈：有效跌破20日线"
            else:
                if current_close < bottom_line:
                    sell_triggered = True
                    sell_reason = "初始止损：假突破，跌破发车底线"

            # --- 第三步：卖出结算 ---
            if sell_triggered:
                sell_price = current_close
                trade_profit = (sell_price - buy_price) / buy_price
                
                trade_records.append({
                    'ts_code': ts_code,
                    'buy_date': buy_date,
                    'sell_date': row['trade_date'],
                    'buy_price': round(buy_price, 2),
                    'sell_price': round(sell_price, 2),
                    'profit_pct': round(trade_profit * 100, 2),
                    'reason': sell_reason
                })
                status = 'EMPTY'

    # 回测结束时如果还持仓，按最后一天价格强制平仓（方便统计）
    if status == 'HOLDING':
        final_price = df.iloc[-1]['close']
        trade_records.append({
            'ts_code': ts_code,
            'buy_date': buy_date,
            'sell_date': df.iloc[-1]['trade_date'],
            'buy_price': round(buy_price, 2),
            'sell_price': round(final_price, 2),
            'profit_pct': round(((final_price - buy_price) / buy_price) * 100, 2),
            'reason': "回测结束强制平仓"
        })

    return trade_records

# ==========================================
# 主程序：批量执行回测
# ==========================================
if __name__ == "__main__":
    stock_pool = get_stock_pool()
    
    # 为了测试速度，这里只取前 50 只股票进行回测。实盘回测时可去掉 [:50]
    test_pool = stock_pool[:50] 
    
    all_trades = []
    print(f"开始对 {len(test_pool)} 只股票进行 V38.0 策略回测...")
    
    for idx, code in enumerate(test_pool):
        try:
            df_hist = get_hist_data(code)
            if not df_hist.empty:
                trades = run_v38_strategy(df_hist, code)
                all_trades.extend(trades)
        except Exception as e:
            print(f"处理 {code} 时出错: {e}")
            
        # 防止触发 Tushare 接口频率限制
        time.sleep(0.2) 
        if (idx + 1) % 10 == 0:
            print(f"已完成 {idx + 1} 只股票回测...")

    # ==========================================
    # 输出回测绩效报告
    # ==========================================
    if all_trades:
        df_results = pd.DataFrame(all_trades)
        
        total_trades = len(df_results)
        win_trades = len(df_results[df_results['profit_pct'] > 0])
        win_rate = win_trades / total_trades * 100
        avg_profit = df_results['profit_pct'].mean()
        max_profit = df_results['profit_pct'].max()
        max_loss = df_results['profit_pct'].min()
        
        print("\n" + "="*40)
        print("🏆 V38.0 中线共振狙击 回测报告")
        print("="*40)
        print(f"总交易次数:   {total_trades} 次")
        print(f"系统胜率:     {win_rate:.2f}%")
        print(f"单次平均收益: {avg_profit:.2f}%")
        print(f"单次最大盈利: {max_profit:.2f}%")
        print(f"单次最大亏损: {max_loss:.2f}%")
        print("="*40)
        
        # 打印排名前 5 的交易看看系统是怎么吃到大肉的
        print("\n✨ 暴利案例 (Top 5 盈利交易):")
        print(df_results.sort_values('profit_pct', ascending=False).head(5).to_string(index=False))
    else:
        print("在所选时间和股票池内，没有触发任何交易信号。")
