import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
from datetime import datetime
from io import StringIO
from tqdm import tqdm

st.set_page_config(page_title="Trading Strategy Simulator", layout="wide")

# Suppress matplotlib user warnings
import warnings
warnings.filterwarnings("ignore")

# Helper functions
def calculate_rsi(series, period=14):
    delta = series.diff().dropna()
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    gain = up.rolling(window=period, min_periods=1).mean()
    loss = down.rolling(window=period, min_periods=1).mean()
    RS = gain / loss
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def calculate_supertrend(df, period=10, multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    df['atr'] = (df['high'] - df['low']).rolling(window=period).mean()
    df['upper_band'] = hl2 + (multiplier * df['atr'])
    df['lower_band'] = hl2 - (multiplier * df['atr'])
    df['supertrend'] = np.nan
    in_uptrend = True

    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upper_band'].iloc[i - 1]:
            in_uptrend = True
        elif df['close'].iloc[i] < df['lower_band'].iloc[i - 1]:
            in_uptrend = False

        if in_uptrend:
            df.loc[i, 'supertrend'] = df['lower_band'].iloc[i]
        else:
            df.loc[i, 'supertrend'] = df['upper_band'].iloc[i]

    df['trend'] = np.where(df['close'] > df['supertrend'], 'up', 'down')
    df.drop(['atr'], axis=1, inplace=True)
    return df

def simulate_trades(df, trading_mode, y_capital, quantity, rsi_period, rsi_buy_threshold, rsi_sell_threshold, x_profit, stop_loss):
    """
    Simulate a simple RSI-based trading strategy on the provided DataFrame with OHLC data.
    """
    if df.empty:
        return [], 0.0

    # Sort and prepare data
    df = df.sort_values('date').reset_index(drop=True)

    # Calculate RSI
    df['rsi'] = calculate_rsi(df['close'], period=rsi_period)

    # Positions
    position_active = False
    buy_price = None
    total_realized_pnl = 0.0
    trades = []

    # Determine the trading quantity based on mode at the start
    if trading_mode == 'capital':
        if df['close'].iloc[0] > 0:
            sim_quantity = math.floor(y_capital / df['close'].iloc[0])
        else:
            sim_quantity = 0
    else:
        sim_quantity = quantity

    # Iterate over each row (candle)
    for i in range(rsi_period, len(df)):
        current_rsi = df['rsi'].iloc[i]
        ltp = df['close'].iloc[i]

        if sim_quantity <= 0:
            # If no shares can be bought, just continue
            continue

        if not position_active:
            # Check entry condition
            if current_rsi <= rsi_buy_threshold:
                # Enter position
                position_active = True
                buy_price = ltp
        else:
            # Check exit conditions
            profit_per_share = ltp - buy_price
            if (profit_per_share >= x_profit) or (current_rsi >= rsi_sell_threshold) or (profit_per_share <= -stop_loss):
                # Close position
                realized_pnl = profit_per_share * sim_quantity
                total_realized_pnl += realized_pnl
                trades.append({
                    'entry_price': buy_price,
                    'exit_price': ltp,
                    'profit_per_share': profit_per_share,
                    'quantity': sim_quantity,
                    'pnl': realized_pnl,
                    'time': df['date'].iloc[i].strftime("%Y-%m-%d %H:%M:%S")
                })
                position_active = False
                buy_price = None

    return trades, total_realized_pnl

def generate_trade_report(trades, total_realized_pnl, trading_mode, y_capital, quantity):
    if not trades:
        return "No trades were executed."

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0].shape[0]
    losses = trades_df[trades_df['pnl'] < 0].shape[0]
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0.0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losses > 0 else 0.0

    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    peak = trades_df['cumulative_pnl'].cummax()
    drawdowns = trades_df['cumulative_pnl'] - peak
    max_drawdown = drawdowns.min()

    if trading_mode == 'capital':
        return_on_capital = (total_realized_pnl / y_capital) * 100 if y_capital > 0 else 0.0
    else:
        avg_entry_price = trades_df['entry_price'].mean() if not trades_df.empty else 0.0
        total_invested = quantity * avg_entry_price if avg_entry_price > 0 else 0.0
        return_on_capital = (total_realized_pnl / total_invested) * 100 if total_invested > 0 else 0.0

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else np.inf

    # Sharpe Ratio (simple approximation)
    if trades_df['pnl'].std() != 0:
        sharpe_ratio = (trades_df['pnl'].mean() / trades_df['pnl'].std()) * np.sqrt(252)
    else:
        sharpe_ratio = np.nan

    report = (
        f"----- TRADE ANALYTICS REPORT -----\n"
        f"Total Trades: {total_trades}\n"
        f"Wins: {wins}, Losses: {losses}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Average Win: {avg_win:.2f}, Average Loss: {avg_loss:.2f}\n"
        f"Gross Profit: {gross_profit:.2f}\n"
        f"Gross Loss: {gross_loss:.2f}\n"
        f"Net Profit: {total_realized_pnl:.2f}\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Return on Capital Deployed: {return_on_capital:.2f}%\n"
        f"Max Drawdown: {max_drawdown:.2f}\n"
        f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
    )

    return report, trades_df

def plot_signals(df, rsi_period, supertrend_period, supertrend_multiplier):
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    df = calculate_supertrend(df, period=supertrend_period, multiplier=supertrend_multiplier)

    df['buy_signal'] = np.where((df['trend'] == 'up') & (df['rsi'] < 30), df['close'], np.nan)
    df['sell_signal'] = np.where((df['trend'] == 'down') & (df['rsi'] > 70), df['close'], np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Price and supertrend
    ax1.plot(df['date'], df['close'], label='Close Price', color='blue')
    ax1.plot(df['date'], df['supertrend'], label='Supertrend', color='green')
    ax1.scatter(df['date'], df['buy_signal'], label='Buy Signal', color='green', marker='^', s=100, zorder=5)
    ax1.scatter(df['date'], df['sell_signal'], label='Sell Signal', color='red', marker='v', s=100, zorder=5)
    ax1.set_title('Price with Supertrend and Buy/Sell Signals')
    ax1.legend()
    ax1.grid(True)

    # RSI
    ax2.plot(df['date'], df['rsi'], label='RSI', color='orange')
    ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax2.set_title('RSI')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig

# Streamlit UI
st.title("Trading Strategy Simulator")

uploaded_file = st.file_uploader("Upload your CSV file containing OHLC data:", type=["csv"])

with st.sidebar:
    st.header("Parameters")
    trading_mode = st.selectbox("Trading Mode", ["capital", "quantity"])
    if trading_mode == "capital":
        y_capital = st.number_input("Total capital (INR)", value=100000.0, min_value=1000.0)
        quantity = 0  # not used in capital mode directly
    else:
        y_capital = 0.0
        quantity = st.number_input("Quantity", value=1000, min_value=1, step=100)

    rsi_period = st.number_input("RSI Period", value=14, min_value=1)
    x_profit = st.number_input("Target Profit per share (INR)", value=2.0, min_value=0.1, step=0.1)
    stop_loss = st.number_input("Stop Loss per share (INR)", value=10.0, min_value=0.1, step=0.1)
    rsi_buy_threshold = st.number_input("RSI Buy Threshold", value=30, min_value=1, max_value=100)
    rsi_sell_threshold = st.number_input("RSI Sell Threshold", value=60, min_value=1, max_value=100)
    supertrend_period = st.number_input("Supertrend Period", value=10, min_value=1)
    supertrend_multiplier = st.number_input("Supertrend Multiplier", value=3.0, min_value=0.1, step=0.1)

run_simulation = st.button("Run Simulation")

if run_simulation and uploaded_file is not None:
    # Read CSV
    data = uploaded_file.read().decode('utf-8')
    df = pd.read_csv(StringIO(data))

    # Ensure required columns
    required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(set(df.columns.str.lower())):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        # Normalize columns
        df.columns = df.columns.str.lower().str.strip()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date').reset_index(drop=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)

        # Run simulation
        with st.spinner("Simulating trades..."):
            trades, total_realized_pnl = simulate_trades(
                df,
                trading_mode=trading_mode,
                y_capital=y_capital,
                quantity=quantity,
                rsi_period=rsi_period,
                rsi_buy_threshold=rsi_buy_threshold,
                rsi_sell_threshold=rsi_sell_threshold,
                x_profit=x_profit,
                stop_loss=stop_loss
            )

        report = generate_trade_report(trades, total_realized_pnl, trading_mode, y_capital, quantity)

        if isinstance(report, tuple):
            report_str, trades_df = report
            st.subheader("Trading Report")
            st.text(report_str)

            st.subheader("Trades Detail")
            st.dataframe(trades_df)

            st.subheader("Strategy Chart")
            fig = plot_signals(df, rsi_period, supertrend_period, supertrend_multiplier)
            st.pyplot(fig)
        else:
            # No trades executed
            st.text(report)

elif run_simulation and uploaded_file is None:
    st.warning("Please upload a CSV file before running the simulation.")
