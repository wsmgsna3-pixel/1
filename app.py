import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import getpass
from datetime import datetime, timedelta
import time

# ==========================================
# 1. 初始配置与安全认证
# ==========================================
print("【主力锁仓·筹码穿透系统】初始化...")
print("本策略利用 Tushare 10000积分权限，调用每日筹码分布(CYQ)与盈利预测数据。")

# 安全输入 Token，不硬编码
my_token = getpass.getpass("👉 请输入您的 Tushare Token (输入时不可见，回车确认): ")
ts.set_token(my_token)
pro = ts.pro_api()

class Config:
    # 回测设置
    START_DATE = '20241101'  # 建议回测最近2-3个月，因为筹码数据量巨大
    END_DATE = '20241220'    # 回测结束日期
    INITIAL_CASH = 1000000   # 初始资金 100万
    MAX_POSITIONS = 5        # 最大持仓只数
    STOP_LOSS = -0.05        # 止损 5%
    TAKE_PROFIT = 0.15       # 止盈 15% (超短线爆发)
    FEE_RATE = 0.0003        # 手续费

cfg = Config()

# ==========================================
# 2. 核心数据获取模块 (10000积分 专属能力)
# ==========================================
def get_trading_days(start, end):
    df = pro.trade_cal(exchange='', start_date=start, end_date=end, is_open='1')
    return df['cal_date'].tolist()

def fetch_data_for_date(date):
    """
    获取单日全市场数据，利用积分优势进行多维数据融合
    """
    try:
        # 1. 基础行情 (价格、成交量)
        df_daily = pro.daily(trade_date=date)
        
        # 2. 每日指标 (换手率、量比、流通市值)
        df_basic = pro.daily_basic(trade_date=date, fields='ts_code,turnover_rate,circ_mv,pe_ttm')
        
        # 3. 【核心VIP数据】每日筹码情况 (需高积分)
        # win_rate: 获利盘比例 (0-100)，越高代表上方抛压越小
        # cost_50: 市场平均成本
        df_cyq = pro.cyq_perf(trade_date=date) 
        
        # 合并数据
        df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
        df_merge = pd.merge(df_merge, df_cyq, on='ts_code', how='inner')
        
        return df_merge
    except Exception as e:
        print(f"数据获取失败 {date}: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 策略选股逻辑 (The Strategy)
# ==========================================
def select_stocks(df_data, date):
    """
    选股逻辑：
    1. 获利盘比例 > 85% (主力高度控盘，且大部分人都在赚钱，惜售)
    2. 换手率 < 10% (并未出现高位出货，锁仓状态)
    3. 市值 50亿 - 800亿 (剔除太小盘和巨无霸)
    4. 涨幅 > 2% 且 < 9.5% (当天有启动迹象，但未涨停)
    """
    if df_data.empty:
        return []

    # 过滤条件
    condition = (
        (df_data['win_rate'] >= 85) &          # 核心：85%筹码获利
        (df_data['turnover_rate'] < 10) &      # 核心：锁仓未出货
        (df_data['turnover_rate'] > 1) &       # 过滤僵尸股
        (df_data['circ_mv'] > 500000) &        # 市值大于50亿
        (df_data['circ_mv'] < 8000000) &       # 市值小于800亿
        (df_data['pct_chg'] > 2.0) &           # 当日启动
        (df_data['pct_chg'] < 9.5)             # 未涨停，给买入机会
    )
    
    selected = df_data[condition].copy()
    
    # 按照获利盘比例排序，取前3名 (强者恒强)
    selected = selected.sort_values(by='win_rate', ascending=False).head(3)
    
    return selected['ts_code'].tolist()

# ==========================================
# 4. 回测引擎 (Backtest Engine)
# ==========================================
class Backtest:
    def __init__(self, config):
        self.cfg = config
        self.cash = config.INITIAL_CASH
        self.positions = {} # {ts_code: {'cost': price, 'vol': volume, 'date': date}}
        self.history_value = [] # 记录每日总资产
        self.trade_log = [] # 交易记录

    def run(self):
        dates = get_trading_days(self.cfg.START_DATE, self.cfg.END_DATE)
        print(f"开始回测区间: {self.cfg.START_DATE} 至 {self.cfg.END_DATE}, 共 {len(dates)} 个交易日")
        
        for date in dates:
            print(f"\nProcessing {date} ... ", end="")
            
            # 1. 获取当日数据
            df_today = fetch_data_for_date(date)
            if df_today.empty:
                continue
            
            # 构建价格查找字典，加快速度
            price_map = df_today.set_index('ts_code')['close'].to_dict()
            high_map = df_today.set_index('ts_code')['high'].to_dict()
            low_map = df_today.set_index('ts_code')['low'].to_dict()

            # 2. 持仓管理 (止盈止损)
            codes_to_sell = []
            current_codes = list(self.positions.keys())
            
            for code in current_codes:
                if code not in price_map: continue # 停牌或数据缺失
                
                cost = self.positions[code]['cost']
                current_price = price_map[code]
                low_price = low_map.get(code, current_price)
                high_price = high_map.get(code, current_price)
                
                # 收益率计算
                pnl_pct = (current_price - cost) / cost
                
                # 止损逻辑 (按最低价触发)
                if (low_price - cost) / cost <= self.cfg.STOP_LOSS:
                    sell_price = cost * (1 + self.cfg.STOP_LOSS) # 模拟止损价成交
                    self.sell(code, sell_price, date, "止损触发")
                    
                # 止盈逻辑 (按最高价触发)
                elif (high_price - cost) / cost >= self.cfg.TAKE_PROFIT:
                    sell_price = cost * (1 + self.cfg.TAKE_PROFIT)
                    self.sell(code, sell_price, date, "止盈触发")
                
                # 持仓超过5天强制换股 (保持资金流动性)
                elif self.days_held(code, date) >= 5:
                    self.sell(code, current_price, date, "持仓超时平仓")

            # 3. 选股与买入
            if len(self.positions) < self.cfg.MAX_POSITIONS:
                targets = select_stocks(df_today, date)
                for code in targets:
                    if len(self.positions) >= self.cfg.MAX_POSITIONS: break
                    if code in self.positions: continue
                    
                    buy_price = price_map.get(code)
                    if buy_price:
                        self.buy(code, buy_price, date)
            
            # 4. 结算当日资产
            total_asset = self.cash
            for code, pos in self.positions.items():
                if code in price_map:
                    total_asset += pos['vol'] * price_map[code]
                else:
                    # 停牌用成本价计算
                    total_asset += pos['vol'] * pos['cost']
            
            self.history_value.append({'date': date, 'total_asset': total_asset})
            print(f"当日资产: {int(total_asset)}")
            
            # 避免过于频繁请求 (礼貌性延迟，虽然你有10000积分)
            time.sleep(0.1)

    def buy(self, code, price, date):
        # 资金分配：等权分配
        available_slot = self.cfg.MAX_POSITIONS - len(self.positions)
        if available_slot <= 0: return
        
        target_val = self.cash / available_slot
        vol = int(target_val / price / 100) * 100 # 向下取整到100股
        
        if vol > 0:
            cost = vol * price * (1 + self.cfg.FEE_RATE)
            if self.cash >= cost:
                self.cash -= cost
                self.positions[code] = {'cost': price, 'vol': vol, 'date': date}
                self.trade_log.append({'date': date, 'action': 'BUY', 'code': code, 'price': price})
                print(f" -> 买入 {code} @ {price}")

    def sell(self, code, price, date, reason):
        pos = self.positions.pop(code)
        revenue = pos['vol'] * price * (1 - self.cfg.FEE_RATE - 0.001) # 卖出多千分之一印花税
        self.cash += revenue
        profit = (revenue - (pos['vol'] * pos['cost']))
        self.trade_log.append({'date': date, 'action': 'SELL', 'code': code, 'price': price, 'reason': reason, 'profit': profit})
        print(f" -> 卖出 {code} @ {price} [{reason}] 盈利: {int(profit)}")

    def days_held(self, code, current_date):
        buy_date_str = self.positions[code]['date']
        d1 = datetime.strptime(buy_date_str, '%Y%m%d')
        d2 = datetime.strptime(current_date, '%Y%m%d')
        return (d2 - d1).days

    def analyze(self):
        df_res = pd.DataFrame(self.history_value)
        df_res['date'] = pd.to_datetime(df_res['date'])
        df_res.set_index('date', inplace=True)
        
        # 计算最大回撤
        df_res['peak'] = df_res['total_asset'].cummax()
        df_res['drawdown'] = (df_res['total_asset'] - df_res['peak']) / df_res['peak']
        max_dd = df_res['drawdown'].min()
        
        total_ret = (df_res['total_asset'].iloc[-1] - self.cfg.INITIAL_CASH) / self.cfg.INITIAL_CASH * 100
        
        print("\n" + "="*30)
        print("【回测结果摘要】")
        print(f"总收益率: {total_ret:.2f}%")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"交易次数: {len(self.trade_log)}")
        print("="*30)

        # 绘图
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(df_res.index, df_res['total_asset'], color='red', label='Strategy Asset')
        plt.title(f'Strategy Performance (Points: 10000+ Exclusive) Return: {total_ret:.2f}%')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.bar(df_res.index, df_res['drawdown'], color='green', label='Drawdown')
        plt.legend()
        plt.grid(True)
        plt.show()

# ==========================================
# 5. 执行
# ==========================================
if __name__ == "__main__":
    if not my_token:
        print("错误：未输入Token，程序退出。")
    else:
        engine = Backtest(cfg)
        engine.run()
        engine.analyze()
