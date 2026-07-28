# -*- coding: utf-8 -*-
"""
翻倍黑马归类统计器 · 周线视角 (形态一: 潜伏爆破型 vs 形态二: 波浪推进型)
------------------------------------------------
基于 V39.1 基础架构衍生，用于统计近50周翻倍股票的上涨路径分布
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
# 页面配置
# ---------------------------
st.set_page_config(page_title="50周翻倍股票形态归类统计", layout="wide")
st.title("📊 近50周翻倍股票形态归类统计器")

# ---------------------------
# Tushare 数据获取与合成
# ---------------------------
@st.cache_data(ttl=3600*12)
def safe_get(_pro, func_name, **kwargs):
    """
    带缓存的安全获取数据函数。
    注意：_pro 前面的下划线是为了让 Streamlit 缓存机制忽略对该复杂对象的哈希检查，避免报错。
    """
    if _pro is None: return pd.DataFrame()
    func = getattr(_pro, func_name)
    try:
        for _ in range(3):
            df = func(**kwargs)
            if df is not None and not df.empty: return df
            time.sleep(0.3)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def get_stock_weekly_data(pro, ts_code, start_date, end_date):
    """获取单只股票复权后的周线数据"""
    df = safe_get(pro, 'daily', ts_code=ts_code, start_date=start_date, end_date=end_date)
    adj = safe_get(pro, 'adj_factor', ts_code=ts_code, start_date=start_date, end_date=end_date)
    
    if df.empty or adj.empty: return pd.DataFrame()
    
    df = df.merge(adj, on=['ts_code', 'trade_date'], how='left').sort_values('trade_date')
    latest_factor = df['adj_factor'].iloc[-1]
    
    # 计算前复权价格
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] * df['adj_factor'] / latest_factor
        
    df['dt'] = pd.to_datetime
