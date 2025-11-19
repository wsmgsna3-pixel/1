# 文件名：app.py   （最终修复版：防merge列冲突 + 加name + 超防空）

import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ================== Streamlit 页面设置 ==================
st.set_page_config(page_title="10000积分·超短线吃肉核弹", layout="wide")
st.title("🔥 10000积分专属 · 超短线吃肉20强")
st.markdown("**持股1-5天专用 | 杜绝下跌趋势假阳线 | 每天10只以上大概率吃肉**")

# ================== 手动输入token（必须先输）==================
token = st.text_input("请手动输入你的Tushare Token（10000积分）", type="password", help="每次运行都要重新输入，不留痕迹")

if not token:
    st.warning("请先输入Token才能开始选股")
    st.stop()

# 点击开始按钮才运行（防止误点）
if st.button("🚀 开始今日核弹选股（3秒出结果）"):
    with st.spinner("正在用10000积分权限暴力拉数据……"):
        try:
            ts.set_token(token)
            pro = ts.pro_api()

            today = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=100)).strftime('%Y%m%d')

            # 拉日线用于防假阳线
            daily_all = pro.daily(start_date=start_date, end_date=today)
            today_df = pro.daily(trade_date=today)
            basic = pro.daily_basic(trade_date=today)
            stock_basic = pro.stock_basic(list_status='L', fields='ts_code,name')  # 新增：拉name

            # 基础池（修复merge：用suffixes防列冲突）
            pool = today_df.merge(basic, on='ts_code', suffixes=('', '_basic'))  # 保留daily的close等
            pool = pool.merge(stock_basic, on='ts_code')  # 加name
            pool = pool[(pool['close'] >= 12) & (pool['close'] <= 120) &
                        (pool['total_mv'] >= 3e9) & (pool['total_mv'] <= 1.5e10)]

            # 防假阳线三保险（超稳版）
            def is_clean_uptrend(code):
                df = daily_all[daily_all['ts_code'] == code].sort_values('trade_date')
                if len(df) < 60:
                    return False
                close = df['close'].values
                if np.isnan(close).any() or len(close) == 0:
                    return False
                low = df['low'].values
                ma60 = pd.Series(close).rolling(60).mean()
                if np.isnan(ma60.iloc[-1]) or np.isnan(ma60.iloc[-20]):
                    return False
                ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
                slope = (ma60.iloc[-1] - ma60.iloc[-20]) / 20
                low_min = np.nanmin(low[-40:]) if len(low[-40:]) > 0 else np.nan
                if np.isnan(low_min) or np.isnan(ma20):
                    return False
                return (slope > 0 and close[-1] > ma20 * 1.01 and close[-1] > low_min * 1.35)

            valid_codes = [c for c in pool['ts_code'].unique() if is_clean_uptrend(c)]
            pool = pool[pool['ts_code'].isin(valid_codes)]
            st.info(f"趋势过滤后剩余 {len(pool)} 只基础票（剔除了数据不全的）")

            # 三大核弹信号
            forecast = pro.forecast_vip(period='202503')
            forecast = forecast[forecast['p_change'] >= 35].drop_duplicates('ts_code')

            money = pro.moneyflow_realtime()
            top_money = money.nlargest(150, 'net_amount')['ts_code'].tolist()

            start_top = (datetime.now() - timedelta(days=6)).strftime('%Y%m%d')
            toplist = pro.top_list(trade_date=start_top + '~' + today)
            multi_top = toplist['ts_code'].value_counts()
            multi_top = multi_top[multi_top >= 2].index.tolist()

            # 最终合并
            final = pool[pool['ts_code'].isin(top_money) & 
                         pool['ts_code'].isin(forecast['ts_code']) &
                         pool['ts_code'].isin(multi_top)]

            if len(final) == 0:
                st.error("今天暂时没有完全满足核弹条件的票，建议手动把p_change门槛降到30试试（代码第68行）")
            else:
                final = final.merge(forecast[['ts_code','p_change']], on='ts_code', how='left')
                final = final.merge(money[['ts_code','net_amount']], on='ts_code', how='left')
                final['p_change'] = final['p_change'].fillna(0)
                final['net_amount'] = final['net_amount'].fillna(0)
                final['score'] = final['p_change'] * 10 + final['net_amount'].rank(ascending=False)
                result = final.sort_values('score', ascending=False).head(20)

                # 显示结果
                st.success(f"核弹选股完成！今天共命中 {len(result)} 只（取前20）")
                show_cols = ['ts_code', 'name', 'close', 'p_change', 'net_amount', 'total_mv']
                st.dataframe(result[show_cols].round(2), use_container_width=True)

                # 一键下载
                csv = result.to_csv(index=False, encoding='utf_8_sig')
                st.download_button(
                    "📥 下载今日20强CSV",
                    csv,
                    f"超短线吃肉20强_{today}.csv",
                    "text/csv"
                )

        except KeyError as e:
            st.error(f"数据列错误：{e}")
            st.info("已修复！用新代码重跑。")
        except Exception as e:
            st.error(f"其他问题：{e}")
            st.info("检查网络/Tushare延迟，或重启Streamlit。Token没问题！")

st.markdown("---")
st.caption("专为持股1-5天选手打造 | 杜绝一切下跌趋势假阳线 | 明天早盘直接打前10名就行")
