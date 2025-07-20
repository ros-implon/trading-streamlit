import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def ma_crossover_signals(df, short_win, long_win, ma_type='SMA'):
    """Add moving averages and signals to DataFrame."""
    df = df.copy()
    if ma_type == 'SMA':
        df['ma_short'] = df['Close'].rolling(window=short_win).mean()
        df['ma_long'] = df['Close'].rolling(window=long_win).mean()
    elif ma_type == 'EMA':
        df['ma_short'] = df['Close'].ewm(span=short_win, adjust=False).mean()
        df['ma_long'] = df['Close'].ewm(span=long_win, adjust=False).mean()
    else:
        raise ValueError("ma_type must be 'SMA' or 'EMA'")
    df['signal'] = 0
    df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
    df.loc[df['ma_short'] < df['ma_long'], 'signal'] = -1
    df['trade_signal'] = df['signal'].diff()
    return df

def backtest_ma(df, short_win, long_win, capital, stoploss_pct, takeprofit_pct, ma_type='SMA'):
    """
    Backtests a moving average crossover strategy with stoploss/takeprofit.
    Returns a dict with trades DataFrame, final equity, total return, and Sharpe.
    """
    df = ma_crossover_signals(df, short_win, long_win, ma_type)
    df = df.dropna().copy()
    position = 0
    entry_price = None
    entry_idx = None
    equity = capital
    trades = []

    for idx, row in df.iterrows():
        current_price = row['Close']
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]
        signal_val = row['trade_signal']
        if isinstance(signal_val, pd.Series):
            signal_val = signal_val.iloc[0]

        # Entry signal (buy)
        if position == 0 and signal_val == 2:  # Cross up
            position = equity // current_price  # Number of shares to buy (whole shares)
            entry_price = current_price
            entry_idx = idx
            equity -= position * entry_price

        # Exit conditions (sell)
        elif position > 0:
            current_return = (current_price - entry_price) / entry_price
            stoploss_hit = current_return <= -stoploss_pct / 100
            takeprofit_hit = current_return >= takeprofit_pct / 100
            cross_exit = signal_val == -2
            if stoploss_hit or takeprofit_hit or cross_exit:
                sell_price = current_price
                equity += position * sell_price
                trades.append({
                    "entry_date": entry_idx,
                    "exit_date": idx,
                    "entry_price": float(entry_price),
                    "exit_price": float(sell_price),
                    "shares": int(position),
                    "pnl": float((sell_price - entry_price) * position),
                    "return_pct": float((sell_price - entry_price) / entry_price * 100)
                })
                position = 0
                entry_price = None
                entry_idx = None

    # Close any open position at the end
    if position > 0:
        sell_price = float(df.iloc[-1]['Close'])
        equity += position * sell_price
        trades.append({
            "entry_date": entry_idx,
            "exit_date": df.index[-1],
            "entry_price": float(entry_price),
            "exit_price": float(sell_price),
            "shares": int(position),
            "pnl": float((sell_price - entry_price) * position),
            "return_pct": float((sell_price - entry_price) / entry_price * 100)
        })

    # Ensure DataFrame has correct columns, even if empty
    trades_df = pd.DataFrame(trades)
    expected_cols = [
        "entry_date", "exit_date", "entry_price", "exit_price",
        "shares", "pnl", "return_pct"
    ]
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=expected_cols)
    else:
        # Remove non-numeric return_pct (should never happen, but just in case)
        if 'return_pct' in trades_df.columns:
            trades_df = trades_df[pd.to_numeric(trades_df['return_pct'], errors='coerce').notnull()]

    # Compute return and sharpe safely
    total_return = ((equity - capital) / capital) * 100 if capital > 0 else 0

    if (
        not trades_df.empty and 
        'return_pct' in trades_df.columns and 
        trades_df['return_pct'].std() not in [None, 0]
    ):
        sharpe = trades_df['return_pct'].mean() / (trades_df['return_pct'].std() + 1e-8) * np.sqrt(252)
    else:
        sharpe = 0

    return {
        "trades": trades_df,
        "final_equity": equity,
        "total_return_pct": total_return,
        "sharpe": sharpe
    }

def optimize_ma(df, capital, stoploss, takeprofit, ma_type, progress_callback=None):
    best_result = None
    best_params = None
    total = 0
    short_range = list(range(5, 30, 1))
    long_range = list(range(6, 100, 2))
    total_iterations = len(short_range) * len(long_range)
    count = 0

    for short_win in short_range:
        for long_win in long_range:
            if long_win <= short_win:
                continue  # long_win should be > short_win
            res = backtest_ma(df, short_win, long_win, capital, stoploss, takeprofit, ma_type)
            if best_result is None or res["total_return_pct"] > best_result["total_return_pct"]:
                best_result = res
                best_params = (short_win, long_win)
            count += 1
            if progress_callback:
                progress_callback(count / total_iterations)
    return best_params, best_result
