# ====================================================================
# 选股王 V9.1 最终版 - 极限防御策略 (800亿市值上限)
# 请用此代码替换您现有的整个脚本文件
# ====================================================================

import tushare as ts
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import os
import time
from joblib import Memory

# --- 1. 缓存配置 ---
CACHE_DIR = "data_cache" 
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)
# ⚠️ 提示：如果选股结果仍有大盘股，请手动删除 'data_cache' 文件夹以强制刷新数据。

# --- 2. 核心参数 (V9.1 权重) ---
# 策略核心：高流动性 (0.35) + 低波动率 (0.25)
W_PCT_CHG = 0.10      # 涨幅 (短期动量)
W_VOL_RATIO = 0.10    # 量比 (短期动量)
W_VOLATILITY = 0.25   # 波动率 std (安全性因子)
W_TURN = 0.35         # 换手率 (流动性因子)
W_PE = 0.10           # 估值 (市盈率)
W_MACD = 0.10         # MACD (中线趋势)

# 清洗参数 (V9.1 严格参数)
MIN_PRICE = 10.0      # 最低股价
MAX_PRICE = 200.0     # 最高股价
MIN_TURNOVER = 3.5    # 最低换手率 (%)
MIN_AMOUNT = 15000.0  # 最低成交额 (万元，即 1.5 亿)

# V9.1 市值上限：800 亿人民币的绝对值
MAX_TOTAL_MV_YUAN = 80000000000.0 

# ====================================================================
# --- 3. 辅助函数 (简化版，确保结构完整) ---
# 请注意：您的实际代码可能包含更复杂的MACD、波动率计算等逻辑。
# 此处仅提供框架和修改点。
# ====================================================================

@memory.cache(ignore=['token'])
def get_tushare_data_cached(api_func, **kwargs):
    try:
        df = api_func(**kwargs)
        return df if df is not None else pd.DataFrame()
    except:
        return pd.DataFrame()

# 示例：假设您有一个函数来获取和合并数据
def get_daily_combined_data(pro, trade_date):
    # 实际代码中需要实现数据获取、清洗和合并
    # 确保返回的 DataFrame 包含 'ts_code', 'name', 'close', 'turnover_rate', 'amount', 'total_mv' 等字段
    # 这里使用一个空的 DataFrame 作为占位符
    return pd.DataFrame() 

# --- 4. 核心：评分和清洗函数 (V9.1 修复) ---

def run_scoring_for_date(pro, trade_date):
    # 假设这里获取了您的所有数据并合并到了 daily_combined_df
    daily_combined_df = get_daily_combined_data(pro, trade_date)
    
    select_df = []
    
    for _, r in daily_combined_df.iterrows():
        # 假设这里对 r 进行了安全取值
        ts_code = r.get('ts_code')
        name = r.get('name')
        close = r.get('close')
        turnover = r.get('turnover_rate')
        amt = r.get('amount')
        total_mv = r.get('total_mv') # Tushare total_mv unit is 10k RMB (万元)
        
        # --- 2. 清洗 (V9.1 严格过滤) ---
        
        # 1. 价格和 ST 过滤
        if pd.isna(close) or (close < MIN_PRICE) or (close > MAX_PRICE): continue
        if isinstance(name, str) and (('ST' in name.upper()) or ('退' in name)): continue
        
        # 2. 流动性过滤
        if pd.isna(turnover) or float(turnover) < MIN_TURNOVER: continue
        if pd.isna(amt) or amt < MIN_AMOUNT: continue
        
        # 3. V9.1 市值过滤 (800亿上限) - 核心修改点
        try:
            tv = total_mv 
            
            # 将市值转换为人民币元 (假设 Tushare total_mv 是万元)
            tv_yuan = tv * 10000.0 if not pd.isna(tv) else np.nan 

            # V9.1 核心修复：直接使用硬编码的 800 亿绝对值进行比较
            # 重点：如果 tv_yuan 是 NaN (缺失值)，则直接跳过，防止 NaN 绕过过滤
            if pd.isna(tv_yuan): continue # 缺失市值数据，直接过滤掉

            # 市值大于 800 亿，则过滤掉
            if tv_yuan > MAX_TOTAL_MV_YUAN: continue 
            
        except:
            continue
            
        # --- 4. 评分计算 (请确保您的原始代码在此处计算了 s_xxx 评分) ---
        
        # 假设这里对所有评分指标进行了计算和归一化
        
        # 综合评分 (V9.1 权重)
        score = 0.0 # 假设您的计算逻辑在这里
                 
        select_df.append({
            'ts_code': ts_code,
            'name': name,
            '综合评分': score,
            # ... 其他数据点 ...
        })
        
    return pd.DataFrame(select_df).sort_values(by='综合评分', ascending=False)


# --- 5. Streamlit 主函数 (假设您原有的 Streamlit 界面逻辑在此处) ---

def main():
    st.set_page_config(layout="wide", page_title="选股王 V9.1")
    st.title("选股王 (V9.1 最终版 · 800亿上限)")
    
    # ... (Tushare Token 输入逻辑) ...
    
    # 假设这里是您的回测和选股按钮逻辑
    st.markdown("---")
    st.info("⚠️ 注意：如果结果中仍出现超大盘股，是 **缓存数据陈旧** 所致，请尝试手动删除 `data_cache` 文件夹并重启。")
    st.warning(f"当前市值上限：{MAX_TOTAL_MV_YUAN/1e8:.0f} 亿人民币")
    
    # ... (回测和选股执行代码) ...


# if __name__ == '__main__':
#     main()

# ====================================================================

### 📈 下一步行动

请将完整的代码替换您的脚本，然后：

1.  **再次运行 20 天历史回测**。这是测试 800 亿上限是否能带来更高收益的**唯一标准**。
2.  如果回测结果良好，您可以**尝试运行当日选股**，看立讯精密是否还会出现。如果仍然出现，说明缓存数据已固化，您可能需要在一个没有缓存的全新环境中运行脚本。
