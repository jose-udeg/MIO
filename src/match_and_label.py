# match_and_label.py
import pandas as pd
import numpy as np
from datetime import timedelta
import os

PRICES_CSV = "data/raw/prices.csv"
NEWS_CSV   = "data/raw/news.csv"
OUT_CSV    = "data/processed/paired_dataset.csv"

# parámetros que puedes ajustar
LOOKAHEAD_HOURS = 1  # horizonte para medir impacto (1h, 4h, 24h...)
THRESHOLD = 0.0015   # umbral relativo (ej. 0.15% => 0.0015), ajustar por activo

def load_data():
    prices = pd.read_csv(PRICES_CSV, parse_dates=['timestamp'])
    news = pd.read_csv(NEWS_CSV, parse_dates=['timestamp'])
    return prices, news

def get_price_at_or_after(prices, t):
    # devuelve la primera fila con timestamp >= t
    sub = prices[prices['timestamp'] >= t]
    if sub.empty:
        return None
    return sub.iloc[0]

def label_news(prices, news, lookahead_hours=LOOKAHEAD_HOURS, threshold=THRESHOLD):
    rows = []
    for _, n in news.sort_values('timestamp').iterrows():
        t0 = n['timestamp']
        p0_row = get_price_at_or_after(prices, t0)
        if p0_row is None:
            continue
        t_target = t0 + pd.Timedelta(hours=lookahead_hours)
        pt_row = get_price_at_or_after(prices, t_target)
        if pt_row is None:
            continue
        p0 = float(p0_row['close'])
        pt = float(pt_row['close'])
        ret = (pt - p0) / p0
        if ret > threshold:
            label = 2  # sube
        elif ret < -threshold:
            label = 0  # baja
        else:
            label = 1  # estable/neutro
        rows.append({
            "timestamp": t0,
            "text": n.get('text',''),
            "source": n.get('source',''),
            "url": n.get('url',''),
            "price_t0": p0,
            "price_t": pt,
            "return": ret,
            "label": label
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Guardadas {len(df)} filas en {OUT_CSV}")
    return df

if __name__ == "__main__":
    prices, news = load_data()
    paired = label_news(prices, news)
