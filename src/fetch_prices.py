# fetch_prices.py
import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def download_prices(ticker="EURJPY=X", start=None, end=None, interval="1h", out_csv="data/raw/prices.csv"):
    """
    Descarga OHLCV y guarda CSV. Ticker ejemplo: 'EURJPY=X', 'EURUSD=X', 'XAUUSD=X', '^DJI'
    interval: "1m","5m","15m","1h","1d" (yfinance limitations)
    """
    print(f"Descargando {ticker} [{interval}]...")
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError("No se descargaron precios. Revisa ticker/rango.")
    df = df.reset_index().rename(columns={'Datetime':'timestamp'})
    # Normalizar nombres
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume","Date":"timestamp"})
    # Asegurar timestamp tipo datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Guardado {len(df)} filas en {out_csv}")
    return df

if __name__ == "__main__":
    # ejemplo: últimos 30 días
    download_prices(ticker="EURJPY=X", start=None, end=None, interval="1h", out_csv="data/raw/prices.csv")
