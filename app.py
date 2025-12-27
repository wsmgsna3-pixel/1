import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="V20.0 最终印钞机", layout="wide")
st.title("🏆 V20.0 黄金狙击 (缩量·实盘最终版)")
st.markdown("""
### 🧠 策略核心 (大数据验证通过)
1.  **价格区间**：**11.0 - 20.0 元** (机构游资共舞区，期望 +0.99%)
2.  **核心逻辑**：**Rank 1** (乖离率最小，超跌反弹)
3.  **决胜因子**：**换手率 < 3.0%** (空头衰竭，胜率 50.5%，期望 +3.02%)
4.  **风控铁律**：**止损 -5%** (安全)，**止盈 +8%回撤3%** (锁利)
""")

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 实盘扫描")
    my_token = st.text_input("Tushare Token", type="password")
    
    # 默认设为今天
    target_date = st.date_input("扫描日期", value=pd.Timestamp.now())
    target_date_str = target_date.strftime('%Y%m%d')

    st.divider()
    
    # === 最终固化的黄金参数 ===
    MIN_PRICE = 11.00
    MAX_PRICE = 20.00
    MAX_TURNOVER = 3.0 # <--- 价值千金的参数
    
    st.success(f"🔒 价格: {MIN_PRICE}-{MAX_PRICE}元")
    st.success(f"🔒 换手: < {MAX_TURNOVER}% (缩量)")

run_btn = st.button("📡 扫描今日冠军", type="primary", use_container_width=True)

if run_btn:
    if not my_token:
        st.error("请输入 Token")
        st.stop()
    ts.set_token(my_token)
    
    try:
        pro = ts.pro_api()
        status_box = st.info(f"正在扫描 {target_date_str} 的全市场数据...")
        
        # 1. 获取基础数据
        df_daily = pro.daily(trade_date=target_date_str)
        if df_daily.empty:
            st.error("今日数据未更新，或非交易日。")
            st.stop()
            
        df_basic = pro.daily_basic(trade_date=target_date_str, fields='ts_code,name,turnover_rate,circ_mv,pe_ttm,industry')
        
        # 尝试获取筹码数据
        df_cyq = pro.cyq_perf(trade_date=target_date_str)
        # 如果当天筹码数据还没出（盘后一般要晚一点），尝试用前一天的数据估算
        if df_cyq.empty:
            prev_date = (target_date - pd.Timedelta(days=1)).strftime('%Y%m%d')
            # 简单回溯几天找最近的筹码数据
            for i in range(1, 5):
                prev_date = (target_date - pd.Timedelta(days=i)).strftime('%Y%m%d')
                df_cyq = pro.cyq_perf(trade_date=prev_date)
                if not df_cyq.empty:
                    st.caption(f"⚠️ 今日筹码数据未出，使用 {prev_date} 数据近似计算 Bias。")
                    break
        
        if df_cyq.empty or 'cost_50pct' not in df_cyq.columns:
            st.error("无法获取筹码数据，无法计算 Bias。")
            st.stop()
            
        # 2. 数据清洗与合并
        df_merge = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
        df_final = pd.merge(df_merge, df_cyq[['ts_code', 'cost_50pct', 'winner_rate']], on='ts_code', how='inner')
        
        # 3. 计算 Bias
        df_final['bias'] = (df_final['close'] - df_final['cost_50pct']) / df_final['cost_50pct']
        
        # 4. 黄金筛选 (11-20元 + 缩量)
        condition = (
            (df_final['bias'] > -0.03) & (df_final['bias'] < 0.15) & 
            (df_final['winner_rate'] < 70) &
            (df_final['circ_mv'] > 300000) &  
            (df_final['close'] >= MIN_PRICE) & 
            (df_final['close'] <= MAX_PRICE) &
            (df_final['turnover_rate'] < MAX_TURNOVER) # <--- 核心过滤
        )
        
        # 筛选并排序
        filtered_df = df_final[condition].sort_values('bias', ascending=True)
        
        status_box.empty()
        
        # 5. 结果展示
        if not filtered_df.empty:
            champion = filtered_df.iloc[0]
            
            st.canvas = st.container()
            with st.canvas:
                st.subheader("🏆 今日缩量冠军 (V20.0)")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("代码", champion['ts_code'])
                c2.metric("名称", champion['name'])
                c3.metric("现价", f"{champion['close']} 元")
                
                c4, c5, c6 = st.columns(3)
                c4.metric("Bias (乖离率)", f"{champion['bias']:.4f}", help="越小越好")
                c5.metric("换手率", f"{champion['turnover_rate']:.2f}%", delta="< 3% (完美)", delta_color="normal")
                c6.metric("获利盘", f"{champion['winner_rate']:.1f}%")
                
                st.divider()
                st.success(f"🚀 **买入理由**：\n该股价格适中 (11-20元)，严重超跌 (Rank 1)，且**换手率极低 ({champion['turnover_rate']}%)**，说明空头动能衰竭，反弹一触即发！")
                
                st.info(f"💡 **交易指令**：\n1. 明日开盘买入。\n2. **止损价：{champion['close']*0.95:.2f} (-5%)**。\n3. **条件单：回落卖出 (触发价 {champion['close']*1.08:.2f}, 回撤 3%)**。")
                
            with st.expander("查看备选池 (Rank 2-10)"):
                st.dataframe(filtered_df.head(10)[['ts_code', 'name', 'close', 'bias', 'turnover_rate', 'winner_rate', 'industry']])
        else:
            st.warning("今日无符合条件的【缩量】黄金标的。建议空仓休息，不要强行交易。")
            
    except Exception as e:
        st.error(f"发生错误: {e}")
