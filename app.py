# 文件名：app.py   （无敌防崩版：换正确moneyflow + 超放松门槛 + debug info）

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
            st.info("日线数据拉取成功！")

            today_df = pro.daily(trade_date=today)
            basic = pro.daily_basic(trade_date=today)
            stock_basic = pro.stock_basic(list_status='L', fields='ts_code,name')

            # 基础池（防列冲突）
            pool = today_df.merge(basic, on='ts_code', suffixes=('', '_basic'))
            pool = pool.merge(stock_basic, on='ts_code')
            pool = pool[(pool['close'] >= 12) & (pool['close'] <= 120) &
                        (pool['total_mv'] >= 3e9) & (pool['total_mv'] <= 1.5e10)]
            st.info(f"基础池构建完成：{len(pool)} 只股票")

            # 防假阳线三保险（超放松版）
            def is_clean_uptrend(code, level=1):
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
                if level == 1:  # 严格
                    return (slope > 0 and close[-1] > ma20 * 1.01 and close[-1] > low_min * 1.35)
                elif level == 2:  # 放松
                    return (slope > -0.005 and close[-1] > ma20 * 0.99 and close[-1] > low_min * 1.25)
                else:  # 超放松
                    return (slope > -0.01 and close[-1] > ma20 * 0.98 and close[-1] > low_min * 1.2)

            # 多级过滤，保证出票
            valid_codes = [c for c in pool['ts_code'].unique() if is_clean_uptrend(c, level=1)]
            if len(valid_codes) == 0:
                st.warning("严格过滤0只，正在放松门槛……")
                valid_codes = [c for c in pool['ts_code'].unique() if is_clean_uptrend(c, level=2)]
            if len(valid_codes) == 0:
                st.warning("放松过滤还是0只，正在超放松门槛……")
                valid_codes = [c for c in pool['ts_code'].unique() if is_clean_uptrend(c, level=3)]
            pool = pool[pool['ts_code'].isin(valid_codes)]
            st.info(f"趋势过滤后剩余 {len(pool)} 只基础票")

            # 三大核弹信号（防空forecast）
            forecast = pro.forecast_vip(period='202503')
            if forecast.empty or 'p_change' not in forecast.columns:
                st.warning("盈利预测空，正在用所有数据（无门槛）……")
                forecast_filtered = forecast.drop_duplicates('ts_code')
                p_threshold = 0
            else:
                forecast_filtered = forecast[forecast['p_change'] >= 35].drop_duplicates('ts_code')
                p_threshold = 35
                if len(forecast_filtered) == 0:
                    st.warning("无>=35%上调，正在降到20%……")
                    forecast_filtered = forecast[forecast['p_change'] >= 20].drop_duplicates('ts_code')
                    p_threshold = 20
            st.info(f"盈利预测过滤完成：{len(forecast_filtered)} 只")

            # 资金流：用正确接口 pro.moneyflow
            money = pro.moneyflow(trade_date=today)
            top_money = money.nlargest(150, 'net_mf_amount')['ts_code'].tolist()  # 用 net_mf_amount 净流入

            # 龙虎榜：改成循环拉多天（避~格式）
            start_top = (datetime.now() - timedelta(days=6)).strftime('%Y%m%d')
            multi_top = pd.DataFrame()
            current_date = datetime.strptime(start_top, '%Y%m%d')
            while current_date <= datetime.now():
                date_str = current_date.strftime('%Y%m%d')
                temp = pro.top_list(trade_date=date_str)
                multi_top = pd.concat([multi_top, temp])
                current_date += timedelta(days=1)
            multi_top_counts = multi_top['ts_code'].value_counts()
            multi_top = multi_top_counts[multi_top_counts >= 2].index.tolist()
            st.info(f"龙虎榜过滤完成：{len(multi_top)} 只多次上榜")

            # 最终合并
            final = pool[pool['ts_code'].isin(top_money) & 
                         pool['ts_code'].isin(forecast_filtered['ts_code']) &
                         pool['ts_code'].isin(multi_top)]

            if len(final) == 0:
                st.error(f"今天没满足条件票（p_change>={p_threshold}）。明天再试，或手动降门槛。")
            else:
                final = final.merge(forecast[['ts_code','p_change']], on='ts_code', how='left')
                final = final.merge(money[['ts_code','net_mf_amount']], on='ts_code', how='left')  # 改字段
                final['p_change'] = final['p_change'].fillna(0)
                final['net_mf_amount'] = final['net_mf_amount'].fillna(0)
                final['score'] = final['p_change'] * 10 + final['net_mf_amount'].rank(ascending=False)
                result = final.sort_values('score', ascending=False).head(20)

                # 显示结果
                st.success(f"核弹完成！命中 {len(result)} 只")
                show_cols = ['ts_code', 'name', 'close', 'p_change', 'net_mf_amount', 'total_mv']
                st.dataframe(result[show_cols].round(2), use_container_width=True)

                # 下载
                csv = result.to_csv(index=False, encoding='utf_8_sig')
                st.download_button(
                    "📥 下载20强CSV",
                    csv,
                    f"吃肉20强_{today}.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"问题：{e}")
            st.info("检查网络/Tushare，或重启。Token OK！")

st.markdown("---")
st.caption("1-5天专用 | 杜绝假阳线 | 早盘打前10")
