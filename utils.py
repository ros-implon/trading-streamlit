import yfinance as yf
import pandas as pd

def fetch_data(symbol, start, end, yf_interval="1d", resample=None):
    df = yf.download(symbol, start=start, end=end, interval=yf_interval, progress=False)
    if df.empty:
        raise ValueError(f"No data fetched for {symbol} with interval {yf_interval}")
    # Flatten multiindex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in downloaded data for {symbol}. Columns available: {df.columns}")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.dropna()
    if resample:
        df = df.resample(resample).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    return df

