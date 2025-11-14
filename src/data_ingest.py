# data_ingest.py
import yfinance as yf
import pandas as pd
from datetime import datetime
from src import config

def download_symbol(ticker, start=None, end=None, interval="1h", out_csv=None):
    """
    Descarga OHLCV desde yfinance.
    ticker: ejemplo 'EURUSD=X' 'XAUUSD=X' '^DJI' etc.
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError("No se descargaron datos. Revisa ticker o rango.")
    df = df.reset_index().rename(columns={'Datetime':'timestamp'})
    df = df[['Datetime','Open','High','Low','Close','Volume']].copy()
    df.columns = ['timestamp','open','high','low','close','volume']
    if out_csv is None:
        out_csv = config.PRICES_CSV
    df.to_csv(out_csv, index=False)
    print(f"Guardado {len(df)} filas en {out_csv}")
    return df

def load_news(csv_path=None):
    if csv_path is None:
        csv_path = config.NEWS_CSV
    df = pd.read_csv(csv_path, parse_dates=[config.TIME_COL])
    return df

def load_prices(csv_path=None):
    if csv_path is None:
        csv_path = config.PRICES_CSV
    df = pd.read_csv(csv_path, parse_dates=[config.TIME_COL])
    return df

if __name__ == "__main__":
    # demo: descargar ejemplo (ajusta ticker y rango)
    try:
        download_symbol("EURUSD=X", start="2020-01-01", end=None, interval="1h", out_csv=config.PRICES_CSV)
    except Exception as e:
        print("Error descarga (usa CSV local):", e)
