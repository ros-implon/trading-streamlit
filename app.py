import streamlit as st
from datetime import date, timedelta
from strategy import ma_crossover_signals, backtest_ma, optimize_ma
from utils import fetch_data
import matplotlib.pyplot as plt

st.set_page_config(page_title="MA Crossover AlgoTrading", layout="centered")

st.title("📈 Moving Average Crossover Algo Trading App")

# Sidebar
st.sidebar.header("User Inputs")
symbol = st.sidebar.text_input("Stock Symbol (e.g. INFY.NS)", value="INFY.NS")
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", date.today())
interval = st.sidebar.selectbox(
    "Data Interval",
    options=["Daily", "Hourly", "Every 3 hours", "Every 4 hours"],
    index=0
)
capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, step=1000)
ma_type = st.sidebar.selectbox("MA Type", options=['SMA', 'EMA'])
stoploss = st.sidebar.slider("Stop Loss (%)", 1, 20, 5)
takeprofit = st.sidebar.slider("Take Profit (%)", 1, 50, 10)
optimize = st.sidebar.checkbox("Optimize Parameters?", value=True)

# Data load
if st.button("Download Data"):
    if interval == "Daily":
        yf_interval = "1d"
        resample = None
    elif interval == "Hourly":
        yf_interval = "1h"
        resample = None
    elif interval == "Every 3 hours":
        yf_interval = "1h"
        resample = "3H"
    elif interval == "Every 4 hours":
        yf_interval = "1h"
        resample = "4H"
    df = fetch_data(symbol, str(start_date), str(end_date), yf_interval, resample)
    st.session_state['df'] = df
    st.success(f"Loaded data for {symbol} ({len(df)} rows)")

# Check if data loaded
df = st.session_state.get('df', None)
if df is not None:
    st.write(df.tail())
    if optimize:
        st.info("Running optimization, please wait...")
        progress_bar = st.progress(0)
        def update_progress(progress):
            progress_bar.progress(progress)
        params, result = optimize_ma(df, capital, stoploss, takeprofit, ma_type, progress_callback=update_progress)
        progress_bar.empty()  # Remove the bar after optimization
        short_win, long_win = params
        st.success(f"Best Params: Short MA: {short_win}, Long MA: {long_win}")
    else:
        short_win = st.sidebar.number_input("Short Window", 5, 50, 10)
        long_win = st.sidebar.number_input("Long Window", 10, 200, 50)
        result = backtest_ma(df, short_win, long_win, capital, stoploss, takeprofit, ma_type)

    st.subheader("Backtest Results")
    st.write(f"Final Equity: ₹{result['final_equity']:.2f}")
    st.write(f"Total Return: {result['total_return_pct']:.2f}%")
    st.write(f"Sharpe Ratio: {result['sharpe']:.2f}")
    st.write(f"Total Trades: {len(result['trades'])}")
    st.dataframe(result['trades'])

    # Download results
    csv = result['trades'].to_csv(index=False)
    st.download_button("Download Trades CSV", csv, "trades.csv", "text/csv")

    # Plot chart
    df_signals = ma_crossover_signals(df, short_win, long_win, ma_type)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df_signals.index, df_signals['Close'], label='Close')
    ax.plot(df_signals.index, df_signals['ma_short'], label=f'Short MA ({short_win})')
    ax.plot(df_signals.index, df_signals['ma_long'], label=f'Long MA ({long_win})')
    buy_signals = df_signals[df_signals['trade_signal'] == 2]
    sell_signals = df_signals[df_signals['trade_signal'] == -2]
    ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy', s=100)
    ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell', s=100)
    ax.legend()
    st.pyplot(fig)

    st.info("To use this for a different symbol or parameter, change the sidebar options and click 'Download Data'.")

else:
    st.warning("Please download data to begin.")

st.caption("Built for beginners. Not financial advice. Use at your own risk!")
