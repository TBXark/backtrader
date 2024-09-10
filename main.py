import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def backtest_strategy(df, strategy_function, initial_capital=10000):
    position = 0
    capital = initial_capital
    trades = []
    daily_returns = []
    equity_curve = [initial_capital]

    for i in range(1, len(df)):
        current_data = df.iloc[i]
        prev_data = df.iloc[i-1]
        
        action, amount = strategy_function(current_data, prev_data, position, capital)
        
        if action == "买入" and amount > 0:
            cost = amount * current_data['close']
            if cost <= capital:
                capital -= cost
                position += amount
                trades.append({
                    "时间": current_data['timestamp'],
                    "操作": "买入",
                    "数量": amount,
                    "价格": current_data['close'],
                    "资金": capital
                })
        elif action == "卖出" and position > 0:
            sell_amount = min(amount, position)
            capital += sell_amount * current_data['close']
            position -= sell_amount
            trades.append({
                "时间": current_data['timestamp'],
                "操作": "卖出",
                "数量": sell_amount,
                "价格": current_data['close'],
                "资金": capital
            })
        
        current_equity = capital + position * current_data['close']
        equity_curve.append(current_equity)
        daily_return = (current_equity - equity_curve[-2]) / equity_curve[-2]
        daily_returns.append(daily_return)

    final_equity = capital + position * df.iloc[-1]['close']
    return trades, final_equity, daily_returns, equity_curve

def rsi_strategy(current_data, prev_data, position, capital):
    current_rsi = current_data['RSI']
    current_close = current_data['close']
    prev_close = prev_data['close']
    
    if current_rsi > 80 and position > 0:
        return "卖出", position
    elif current_rsi < 20 and position == 0:
        shares_to_buy = capital // current_close
        return "买入", shares_to_buy
    elif position > 0 and (current_close - prev_close) > 10:
        return "卖出", position
    elif position > 0 and (prev_close - current_close) > 5:
        return "卖出", position
    
    return "持有", 0

def calculate_metrics(initial_capital, final_equity, daily_returns, equity_curve):
    total_return = (final_equity - initial_capital) / initial_capital * 100
    cagr = (final_equity / initial_capital) ** (252 / len(daily_returns)) - 1
    
    daily_returns_series = pd.Series(daily_returns)
    sharpe_ratio = np.sqrt(252) * daily_returns_series.mean() / daily_returns_series.std()
    sortino_ratio = np.sqrt(252) * daily_returns_series.mean() / daily_returns_series[daily_returns_series < 0].std()
    
    max_drawdown = np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1)
    calmar_ratio = cagr / abs(max_drawdown)
    
    best_day = daily_returns_series.max()
    worst_day = daily_returns_series.min()
    
    metrics = {
        "总回报率": f"{total_return:.2f}%",
        "年化收益率 (CAGR)": f"{cagr*100:.2f}%",
        "夏普比率": f"{sharpe_ratio:.2f}",
        "索提诺比率": f"{sortino_ratio:.2f}",
        "最大回撤": f"{max_drawdown*100:.2f}%",
        "卡玛比率": f"{calmar_ratio:.2f}",
        "最佳单日收益": f"{best_day*100:.2f}%",
        "最差单日收益": f"{worst_day*100:.2f}%",
        "交易次数": len(trades),
        "最终权益": final_equity
    }
    
    return metrics

uploaded_file = st.file_uploader("选择CSV文件", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=',')
    print(df.head())
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.dropna()
    df_resampled = df.set_index('timestamp').resample('15T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df_resampled = df_resampled.dropna()
    df_resampled['RSI'] = calculate_rsi(df_resampled['close'], window=14)    
    df_resampled = df_resampled.dropna()
    df_resampled = df_resampled.reset_index()

    
    st.write("数据预览:")
    st.write(df_resampled.head())
    
    columns = df_resampled.columns.tolist()

    default_x = 'timestamp' if 'timestamp' in columns else columns[0]
    default_y = 'close' if 'close' in columns else columns[0]
    
    x_axis = st.selectbox("选择X轴", options=columns, index=columns.index(default_x))
    y_axis = st.selectbox("选择Y轴", options=columns, index=columns.index(default_y))
    
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=df_resampled[x_axis], y=df_resampled[y_axis], name=y_axis), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_resampled[x_axis], y=df_resampled['RSI'], name='RSI'), row=2, col=1)
    
    fig.update_layout(height=600, title_text=f'{y_axis} and RSI vs {x_axis}')
    fig.update_yaxes(title_text=y_axis, row=1, col=1)
    fig.update_yaxes(title_text='RSI', row=2, col=1)
    
    st.plotly_chart(fig)

    # 回测逻辑
    if 'close' in df_resampled.columns and 'RSI' in df_resampled.columns:
        
        initial_capital = 100000
        trades, final_equity, daily_returns, equity_curve = backtest_strategy(df_resampled, rsi_strategy, initial_capital)
        
        st.subheader("回测结果")
        st.write("交易记录:")
        
        if trades:
            df_trades = pd.DataFrame(trades)
            df_trades['时间'] = pd.to_datetime(df_trades['时间'])
            df_trades['时间'] = df_trades['时间'].dt.strftime('%Y-%m-%d %H:%M')
            df_trades['价格'] = df_trades['价格'].round(2)
            df_trades['资金'] = df_trades['资金'].round(2)
            st.table(df_trades)
        else:
            st.write("没有执行任何交易。")

        metrics = calculate_metrics(initial_capital, final_equity, daily_returns, equity_curve)
        st.subheader("策略表现指标")
        for key, value in metrics.items():
            st.write(f"{key}: {value}")
        
        # 绘制权益曲线
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(y=equity_curve, mode='lines', name='权益曲线'))
        fig_equity.update_layout(title='策略权益曲线', xaxis_title='交易日', yaxis_title='权益')
        st.plotly_chart(fig_equity)

    else:
        st.error("数据中缺少 'close' 或 'RSI' 列，无法进行回测。")

else:
    st.write("请上传CSV文件以查看回测结果。")
