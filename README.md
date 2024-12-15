# **Algo Trading Framework**

An advanced algorithmic trading framework that integrates **Relative Strength Index (RSI)** and **Supertrend** indicators for both backtesting and live trading. This versatile script supports paper trading with historical data and can be seamlessly configured for live trading via APIs like Zerodha's KiteConnect.

### **Features**
- **Flexible Trading Modes**:  
  - **Quantity-Based Trading**: Trade with a fixed number of shares.  
  - **Capital-Based Trading**: Allocate a specific capital for trades.  

- **Indicators and Strategies**:  
  - **RSI**: Detects overbought and oversold conditions.  
  - **Supertrend**: Identifies market trends and provides actionable buy/sell signals.  

- **Paper Trading**: Simulates trading with historical data from a CSV file for risk-free strategy testing.  

- **Live Trading Integration**: Connect to real-time APIs (e.g., KiteConnect) for executing live trades.  

- **Trade Analytics and Reporting**:  
  - Generates detailed trade reports (CSV and TXT).  
  - Key metrics: Win Rate, Profit Factor, Drawdown, Sharpe Ratio.  

- **Visualization**:  
  - Dynamic plots for RSI and Supertrend with buy/sell signals.  

- **Error Handling**: Robust logging and graceful handling of missing data or API issues.  

### **Getting Started**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/algo-trading-framework.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure trading parameters in the script (e.g., CSV file, RSI thresholds).  
4. Run the script for paper trading or connect live trading APIs.

### **User Manual**
For detailed instructions on configuring and using the framework, refer to the [User Manual](user_manual.md).  

### **Disclaimer**  
This framework is for educational and experimental purposes only. Use it at your own risk. Always backtest strategies thoroughly before live trading and adhere to trading regulations.

---
