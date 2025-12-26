import streamlit as st
import tushare as ts
import pandas as pd

st.set_page_config(page_title="V18.3 侦探模式", layout="wide")
st.title("🕵️‍♂️ V18.3 侦探模式：Rank 1 到底怎么开盘？")

with st.sidebar:
    my_token = st.text_input("Tushare Token", type="password")
    run_btn = st.button("🔍 开始侦查", type="primary")

if run_btn and my_token:
    ts.set_token(my_token)
    pro = ts.pro_api()
    
    # 随机选几个日期进行抽查
    dates_to_check = ['20240604', '20240815', '20241010', '20250108', '20250320']
    
    report = []
    
    progress = st.progress(0)
    
    for i, date in enumerate(dates_to_check):
        progress.progress((i+1)/len(dates_to_check))
        
        # 1. 找当天的 Rank 1
        df_daily = pro.daily(trade_date=date)
        df_cyq = pro.cyq_perf(trade_date=date)
        
        if df_daily.empty or df_cyq.empty: continue
        
        # 合并算 Bias
        df = pd.merge(df_daily, df_cyq, on='ts_code')
        df['bias'] = (df['close'] - df['cost_50pct']) / df['cost_50pct']
        
        # 筛选 11-20元
        df = df[
            (df['bias'] > -0.03) & (df['bias'] < 0.15) & 
            (df['close'] >= 11.0) & (df['close'] <= 20.0)
        ].sort_values('bias')
        
        if df.empty: continue
        
        champion = df.iloc[0] # 选出当天的冠军
        code = champion['ts_code']
        buy_date_price = champion['close']
        
        # 2. 看它“第二天”怎么开盘
        # 获取下一交易日
        next_day_df = pro.daily(ts_code=code, start_date=date, end_date='20251231')
        next_day_df = next_day_df.sort_values('trade_date')
        
        if len(next_day_df) >= 2:
            # next_day_df.iloc[0] 是买入当天
            # next_day_df.iloc[1] 是第二天
            next_day_data = next_day_df.iloc[1]
            
            open_price = next_day_data['open']
            pre_close = next_day_data['pre_close'] # 也就是前一天的收盘价
            gap = (open_price - pre_close) / pre_close * 100
            
            report.append({
                '选股日期': date,
                '代码': code,
                '买入日收盘': buy_date_price,
                '次日开盘': open_price,
                '次日昨收': pre_close,
                '开盘表现': f"{gap:.2f}%",
                '状态': "🔥 高开" if gap > 0 else "🧊 低开"
            })
            
    st.table(pd.DataFrame(report))
    
    # 统计
    df_rep = pd.DataFrame(report)
    if not df_rep.empty:
        high_count = len(df_rep[df_rep['状态'].str.contains("高开")])
        st.metric("抽查样本中高开比例", f"{high_count}/{len(df_rep)}")
        if high_count == 0:
            st.warning("结论：Rank 1 股票几乎全是低开！您的‘高开过滤’策略可能没有操作空间。")
        else:
            st.success("结论：存在高开样本！之前的‘无数据’是代码缓存问题，可以修复！")
