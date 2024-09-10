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

    final_assets = capital + position * df.iloc[-1]['close']
    return trades, final_assets

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
    
    short_signals = df_resampled[df_resampled['RSI'] > 80]
    fig.add_trace(go.Scatter(
        x=short_signals[x_axis],
        y=short_signals[y_axis],
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='red'),
        name='做空信号'
    ), row=1, col=1)

    long_signals = df_resampled[df_resampled['RSI'] < 20]
    fig.add_trace(go.Scatter(
        x=long_signals[x_axis],
        y=long_signals[y_axis],
        mode='markers',
        marker=dict(symbol='triangle-up', size=10, color='green'),
        name='做多信号'
    ), row=1, col=1)
    
    fig.update_layout(height=600, title_text=f'{y_axis} and RSI vs {x_axis}')
    fig.update_yaxes(title_text=y_axis, row=1, col=1)
    fig.update_yaxes(title_text='RSI', row=2, col=1)
    
    st.plotly_chart(fig)

    # 回测逻辑
    if 'close' in df_resampled.columns and 'RSI' in df_resampled.columns:
        trades, final_assets = backtest_strategy(df_resampled, rsi_strategy)

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

        st.write(f"最终资产: {final_assets:.2f}")
    else:
        st.error("数据中缺少 'close' 或 'RSI' 列，无法进行回测。")

else:
    st.write("请上传CSV文件以查看回测结果。")
