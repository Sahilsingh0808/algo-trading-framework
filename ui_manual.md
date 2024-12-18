### UI Manual: **How to Use the Algo Trading App**

Welcome to the **Algo Trading Framework**! 🚀 This guide will walk you through how to use the app step by step. It’s designed to be simple and user-friendly.

---

### 🌐 **Access the App**
Open your browser and visit the app using this link:
👉 [**https://algo-trading-framework.streamlit.app/**](https://algo-trading-framework.streamlit.app/)

---

### 🏁 **Getting Started**

1. **Choose Data Source**:
   - You will see a **radio button** to choose how you want to provide stock data:
     - **Upload CSV**: If you have historical stock data in a CSV file.
     - **Fetch from Yahoo Finance**: Automatically download stock data based on your inputs.

---

### 📁 **Option 1: Upload CSV**
1. Select **Upload CSV**.
2. Click the **Browse files** button to upload your CSV file.
   - Ensure your file has the following columns: `date`, `open`, `high`, `low`, `close`, `volume`.
3. Proceed to the **Sidebar Parameters** section to configure your trading strategy.

---

### 📡 **Option 2: Fetch from Yahoo Finance**
1. Select **Fetch from Yahoo Finance**.
2. Enter the following details in the form:
   - **Stock Symbol**: The ticker symbol of the stock (e.g., `AAPL` for Apple, `TSLA` for Tesla).
   - **Start Date**: The start date for historical data.
   - **End Date**: The end date for historical data.
   - **Interval**: Choose the data interval (e.g., `1m`, `5m`, `1d`).
3. Click **Fetch Data** to download the stock data.
4. Once the data is fetched, you’ll see a preview table. You’re now ready to configure your trading strategy!

---

### ⚙️ **Configure Your Trading Strategy**
On the left **Sidebar**, configure the following parameters:

1. **Trading Mode**:
   - `capital`: Simulate trades with a fixed capital amount.
   - `quantity`: Simulate trades with a fixed number of shares.

2. **Parameters**:
   - **Capital** (if in `capital` mode): The amount of money you want to allocate.
   - **Quantity** (if in `quantity` mode): The number of shares to trade.
   - **RSI Period**: The lookback period for RSI calculation.
   - **Target Profit**: The profit per share (in INR) to take a position.
   - **Stop Loss**: The maximum loss per share (in INR) before exiting a position.
   - **RSI Buy Threshold**: RSI value below which a buy signal is triggered.
   - **RSI Sell Threshold**: RSI value above which a sell signal is triggered.
   - **Supertrend Period**: The lookback period for the Supertrend indicator.
   - **Supertrend Multiplier**: The multiplier for the Supertrend calculation.

---

### ▶️ **Run Simulation**
1. Once you’ve uploaded or fetched data and configured the parameters:
   - Click the **Run Simulation** button.
2. The app will process the data and apply your trading strategy.

---

### 📊 **View Results**
1. **Strategy Chart**:
   - Visualize price, buy/sell signals, and RSI trends on the chart.
2. **Trade Summary**:
   - Get a detailed report of trades, including entry/exit prices and profit/loss.
3. **Download Data**:
   - Download the processed data with trading signals as a CSV file for further analysis.

---

### 🎉 **That’s It!**
You’re now ready to explore and optimize your trading strategy. If you have any questions or encounter issues, feel free to contact support. Happy trading! 🚀📈

---

We hope you enjoy using the **Algo Trading Framework**! 😊