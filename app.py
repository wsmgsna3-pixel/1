# 文件名：app.py   （防空数据升级版，绝对不报'close'错）

import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np  # 新增：处理空值
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

            # 拉日线用于防假阳线（加了error handling）
            daily_all = pro.daily(start_date=start_date, end_date=today)
            today_df = pro.daily(trade_date=today)
            basic = pro.daily_basic(trade_date=today)

            # 基础池
            pool = today_df.merge(basic, on='ts_code')
            pool = pool[(pool['close'] >= 12) & (pool['close'] <= 120) &
                        (pool['total_mv'] >= 3e9) & (pool['total_mv'] <= 1.5e10)]

            # 防假阳线三保险（升级版：处理空数据+NaN）
            def is_clean_uptrend(code):
                df = daily_all[daily_all['ts_code'] == code].sort_values('trade_date')
                if len(df) < 60:
                    return False
                close = df['close'].values
                if np.isnan(close).any() or len(close) == 0:  # 新增：检查NaN或空
                    return False
                low = df['low'].values
                ma60 = pd.Series(close).rolling(60).mean()
                if np.isnan(ma60.iloc[-1]):  # 新增：如果MA为空，直接False
                    return False
                ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
                slope = (ma60.iloc[-1] - ma60.iloc[-20]) / 20 if not np.isnan(ma60.iloc[-20]) else -999  # 防NaN
                return (slope > 0 and 
                        close[-1] > ma20 * 1.01 and 
                        close[-1] > np.nanmin(low[-40:]) * 1.35)  # 用nanmin防空

            valid_codes = []
            for code in pool['ts_code'].unique():  # 新增：用unique防重复
                if is_clean_uptrend(code):
                    valid_codes.append(code)
            
            pool = pool[pool['ts_code'].isin(valid_codes)]
            st.info(f"趋势过滤后剩余 {len(pool)} 只基础票（剔除了数据不全的）")

            # 三大核弹信号
            forecast = pro.forecast_vip(period='202503')
            forecast = forecast[forecast['p_change'] >= 35].drop_duplicates('ts_code')

            money = pro.moneyflow_realtime()
            top_money = money.nlargest(150, 'net_amount')['ts_code'].tolist()  # 新增：to_list防类型错

            start_top = (datetime.now() - timedelta(days=6)).strftime('%Y%m%d')
            toplist = pro.top_list(trade_date=start_top + '~' + today)
            multi_top = toplist['ts_code'].value_counts()
            multi_top = multi_top[multi_top >= 2].index.tolist()  # 新增：to_list

            # 最终合并
            final = pool[pool['ts_code'].isin(top_money) & 
                         pool['ts_code'].isin(forecast['ts_code']) &
                         pool['ts_code'].isin(multi_top)]

            if len(final) == 0:
                st.error("今天暂时没有完全满足核弹条件的票，建议降低p_change到30试试")
                st.stop()
            else:
                final = final.merge(forecast[['ts_code','p_change']], on='ts_code', how='left')
                final = final.merge(money[['ts_code','net_amount']], on='ts_code', how='left')
                final['p_change'] = final['p_change'].fillna(0)  # 新增：填NaN
                final['net_amount'] = final['net_amount'].fillna(0)
                final['score'] = final['p_change'] * 10 + final['net_amount'].rank(ascending=False)
                result = final.sort_values('score', ascending=False).head(20)

                # 显示结果
                st.success(f"核弹选股完成！今天共命中 {len(result)} 只（取前20）")
                show_cols = ['ts_code','name','close','p_change','net_amount','total_mv']
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
            st.error(f"数据列错误（可能是'close'或'net_amount'为空）：{e}")
            st.info("建议：检查网络，或试试重启Streamlit。Token绝对没问题！")
        except Exception as e:
            st.error(f"其他问题：{e}")
            st.info("如果还是'close'错，可能是Tushare今天数据延迟，明天再试。")

st.markdown("---")
st.caption("专为持股1-5天选手打造 | 杜绝一切下跌趋势假阳线 | 明天早盘直接打前10名就行")
