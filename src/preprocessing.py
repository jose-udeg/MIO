# preprocessing.py
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta

def clean_text(text):
    """
    Limpieza básica: bajar a minúsculas, quitar URLs, signos extras.
    No le hagas demasiada limpieza si usarás modelos tipo BERT.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^a-z0-9áéíóúüñ.,;:¡!¿?() ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_technical_indicators(prices_df):
    df = prices_df.copy().reset_index(drop=True)
    df['close'] = df['close'].astype(float)
    df['ret'] = df['close'].pct_change().fillna(0)
    # SMA & EMA
    df['sma10'] = df['close'].rolling(10, min_periods=1).mean()
    df['ema12'] = ta.trend.ema_indicator(df['close'], window=12).fillna(method='bfill')
    # RSI
    df['rsi14'] = ta.momentum.rsi(df['close'], window=14).fillna(0)
    # ATR (volatilidad)
    df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).fillna(0)
    # Llenar NaNs si quedan
    df = df.fillna(method='bfill').fillna(0)
    return df

def scale_numeric_sequences(X_num):
    """
    X_num: ndarray (N, T, F)
    Escala por feature globalmente con StandardScaler
    """
    N, T, F = X_num.shape
    resh = X_num.reshape(-1, F)
    scaler = StandardScaler()
    resh_s = scaler.fit_transform(resh)
    return resh_s.reshape(N, T, F), scaler

if __name__ == "__main__":
    # prueba rápida
    df = pd.DataFrame({
        'timestamp': pd.date_range("2023-01-01", periods=30, freq='H'),
        'open': np.linspace(1.1, 1.2, 30),
        'high': np.linspace(1.12, 1.22, 30),
        'low': np.linspace(1.09, 1.19, 30),
        'close': np.linspace(1.11, 1.21, 30),
        'volume': np.random.randint(100,500,30)
    })
    df2 = add_technical_indicators(df)
    print(df2.head())
