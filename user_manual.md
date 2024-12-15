# **User Manual: RSI-Based Algorithmic Trading Script**

## **Overview**
This script provides an automated trading framework using the **Relative Strength Index (RSI)**, **Supertrend**, and other technical indicators for both live and simulated (paper) trading. It uses CSV data for backtesting and supports live trading setups via APIs like Zerodha's KiteConnect.

---

## **Setup and Configuration**
### 1. **Pre-Requisites**
- **Python 3.x** installed.
- Install dependencies:
  ```bash
  pip install pandas numpy matplotlib yfinance
  ```
- A valid **CSV file** with historical OHLC data for paper trading (e.g., `sbin_ns.csv`).

### 2. **Key Script Configurations**
Edit the following parameters in the script to match your requirements:

#### **Trading Mode**
- **TRADING_MODE**: `'quantity'` (default) for quantity-based trading or `'capital'` for capital-based trading.

#### **Common Parameters**
- `INSTRUMENT`: Stock ticker, e.g., `"NSE:SBIN"`.
- `RSI_PERIOD`: Number of periods for RSI calculation (default: 14).
- `RSI_BUY_THRESHOLD`: RSI level to trigger a buy signal (default: 30).
- `RSI_SELL_THRESHOLD`: RSI level to trigger a sell signal (default: 60).
- `STOP_LOSS`: Maximum allowable loss per trade in INR (default: 10.0).
- `X_PROFIT`: Minimum profit per share to exit a trade (default: 2.0).

#### **Paper Trading Parameters**
- `CSV_FILE`: Path to your historical OHLC data file.
- `REPORT_FILE_TXT`: Path to save trade analytics reports.
- `REPORT_FILE`: Path to save detailed trade reports in CSV format.

#### **Live Trading Parameters**
- Uncomment and set your API keys (`API_KEY`, `API_SECRET`, `ACCESS_TOKEN`) for Zerodha if using live trading.

---

## **How to Run**
### 1. **Paper Trading**
1. Set `TRADING_MODE` to `'quantity'` or `'capital'`.
2. Ensure the CSV file (`CSV_FILE`) is present and properly formatted.
3. Execute the script:
   ```bash
   python algo_trading_rsi.py
   ```
4. Monitor the logs for trading activity and analytics (`app.log`).

### 2. **Live Trading**
1. Set your trading credentials (`API_KEY`, `API_SECRET`, `ACCESS_TOKEN`).
2. Enable the KiteConnect API integration in the script.
3. Run the script as above. Ensure real-time data feed is available.

---

## **Core Functionalities**
### **1. RSI and Supertrend Calculations**
- **RSI**: Used to determine overbought and oversold conditions.
- **Supertrend**: Identifies market trends and generates buy/sell signals.

### **2. Buy/Sell Signal Generation**
- A **Buy Signal** is triggered when:
  - RSI < `RSI_BUY_THRESHOLD`
  - Supertrend indicates an upward trend.
- A **Sell Signal** is triggered when:
  - RSI > `RSI_SELL_THRESHOLD`
  - Supertrend indicates a downward trend.
  - Profit/loss conditions are met.

### **3. Trade Execution**
- Simulates orders in paper trading mode.
- Places real orders via APIs in live trading mode.

### **4. Analytics and Reporting**
- Calculates key metrics such as:
  - Total Trades, Win/Loss Rate, Net Profit, Profit Factor.
  - Maximum Drawdown and Sharpe Ratio.
- Generates a detailed trade report in CSV and TXT formats.

---

## **Data Requirements**
The CSV file (`CSV_FILE`) should contain the following columns:
- `date`: Timestamp of the candle.
- `open`, `high`, `low`, `close`: OHLC prices.
- `volume`: Trading volume.

---

## **Visualizations**
### **1. RSI Chart**
- Plots RSI with buy/sell signals marked.
- Highlights overbought (70) and oversold (30) levels.

### **2. Supertrend Chart**
- Displays close prices with supertrend and buy/sell signals.

To view the plots, ensure you have **matplotlib** installed.

---

## **Error Handling**
- Missing or improperly formatted CSV data is logged and handled gracefully.
- Live trading API issues are logged with detailed error messages.

---

## **Advanced Features**
### Fetch Historical Data
The script includes a utility (`fetch_stock_data_in_chunks`) to fetch historical data from Yahoo Finance for specified intervals.

---

## **Disclaimer**
This script is for educational purposes only. Always backtest thoroughly before live trading. Use it at your own risk. Ensure compliance with trading regulations and risk management principles.

---

Feel free to ask if you need additional details or enhancements!
