#!/usr/bin/env python3
"""
src/auto_pipeline.py

Pipeline automático:
- Obtener noticias relacionadas con una divisa (NewsAPI o scraping RSS).
- Descargar precios históricos del par (yfinance).
- Emparejar noticia -> precio(t0) -> precio(t0 + window_hours).
- Calcular retorno y etiqueta (sube/baja/estable).
- Guardar CSV procesado listo para entrenamiento.

Uso:
python src/auto_pipeline.py --symbol EURJPY --ticker "EURJPY=X" --interval 60m --window 4 --newsapi_key TU_KEY
ó (sin API key, usa RSS):
python src/auto_pipeline.py --symbol EURJPY --ticker "EURJPY=X" --interval 60m --window 4 --use_rss

Notas:
- Requiere: pandas, requests, beautifulsoup4, yfinance, python-dateutil
- Ajusta keywords en KEYWORDS_BY_SYMBOL si quieres búsquedas más específicas.
"""

import os
import argparse
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from dateutil import parser as dparser
from bs4 import BeautifulSoup
import yfinance as yf

# -----------------------------
# Config / keywords por divisa
# -----------------------------
KEYWORDS_BY_SYMBOL = {
    # clave: lista de keywords para buscar noticias relacionadas
    "EURJPY": ["EURJPY", "EUR/JPY", "euro yen", "euro yenes", "euro jpy", "bank of japan", "european central bank", "banco central europeo", "boj", "japan"],
    "EURUSD": ["EURUSD", "EUR/USD", "euro dollar", "euro usd", "euro", "dollar", "ecb", "european central bank"],
    "USDJPY": ["USDJPY", "USD/JPY", "dollar yen", "usd jpy", "fed", "boj"],
    "XAUUSD": ["gold", "xauusd", "gold price", "oro", "gold ounce", "gold usd", "inflation", "fed"],
    # agrega más si quieres
}

# Default output
OUT_DIR = "data/raw/auto_pipeline"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# NewsAPI fetcher
# -----------------------------
def fetch_news_newsapi(keywords, api_key, from_dt, to_dt, page_size=100):
    """
    Busca noticias en NewsAPI.org que contengan cualquiera de las keywords.
    Devuelve lista de dicts: [{'publishedAt':..., 'title':..., 'description':..., 'url':...}, ...]
    Requiere API key de NewsAPI.
    """
    all_articles = []
    url = "https://newsapi.org/v2/everything"
    q = " OR ".join([f'"{k}"' for k in keywords])  # consulta OR de palabras/frases
    params = {
        "q": q,
        "from": from_dt.isoformat(),
        "to": to_dt.isoformat(),
        "language": "en",
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": api_key
    }
    resp = requests.get(url, params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"NewsAPI error {resp.status_code}: {resp.text}")
    data = resp.json()
    for a in data.get("articles", []):
        published = a.get("publishedAt")
        # some articles have None publishedAt - skip
        if not published:
            continue
        text = (a.get("title") or "") + " - " + (a.get("description") or "")
        all_articles.append({
            "timestamp": published,
            "text": text.strip(),
            "source": a.get("source", {}).get("name"),
            "url": a.get("url")
        })
    return all_articles

# -----------------------------
# Simple RSS scrapper (fallback)
# -----------------------------
DEFAULT_RSS_SOURCES = [
    "https://www.reuters.com/markets/wealth/rss",  # Reuters markets
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",  # WSJ markets
    "https://www.bloomberg.com/feed/podcast/etf-report.xml",  # bloomberg (podcast feed)
]

def fetch_news_rss(keywords, from_dt, to_dt, rss_list=None):
    """
    Obtiene titulares de varias fuentes RSS y filtra por keywords.
    Es más ruidoso que NewsAPI pero no requiere API key.
    """
    rss_list = rss_list or DEFAULT_RSS_SOURCES
    articles = []
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    for rss in rss_list:
        try:
            r = requests.get(rss, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "xml")
            items = soup.find_all("item")
            for it in items:
                title = it.title.string if it.title else ""
                desc = it.description.string if it.description else ""
                pub = None
                if it.pubDate:
                    try:
                        pub = dparser.parse(it.pubDate.string)
                    except Exception:
                        pub = None
                # filter by date range
                if pub is None:
                    continue
                if not (from_dt <= pub <= to_dt):
                    continue
                text = f"{title} - {desc}"
                # filter by keywords
                low = text.lower()
                if any(k.lower() in low for k in keywords):
                    articles.append({
                        "timestamp": pub.isoformat(),
                        "text": text.strip(),
                        "source": rss,
                        "url": None
                    })
        except Exception as e:
            print(f"⚠️ RSS fetch error {rss}: {e}")
        time.sleep(0.5)
    return articles

# -----------------------------
# Prices fetcher (yfinance)
# -----------------------------
def download_prices_yf(ticker, start_dt, end_dt, interval="60m"):
    """
    Descarga OHLCV con yfinance. ticker p.ej. 'EURJPY=X' o 'XAUUSD=X'
    interval: '1h' -> '60m' en yfinance, '1d', '1m' etc.
    """
    # yfinance uses '60m' or '1d', map common
    interval_map = {"1h":"60m", "60m":"60m", "1d":"1d", "1m":"1m", "5m":"5m", "15m":"15m"}
    intv = interval_map.get(interval, interval)
    print(f"Descargando {ticker} {intv} desde {start_dt} hasta {end_dt} ...")
    df = yf.download(ticker, start=start_dt, end=end_dt + timedelta(days=1), interval=intv, progress=False)
    if df.empty:
        print("⚠️ yfinance devolvió dataframe vacío.")
        return df
    df = df.reset_index().rename(columns={"Datetime":"timestamp"})
    # Normalizar columnas
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume","timestamp":"timestamp"})
    # Asegurar timestamp tipo datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df[['timestamp','open','high','low','close','volume']]

# -----------------------------
# Emparejar noticia -> precio
# -----------------------------
def get_price_at(df_prices, t, method="nearest_forward"):
    """
    df_prices: DataFrame con 'timestamp' ascendente
    t: datetime
    method:
      - 'nearest' -> index de precio más cercano (antes o después)
      - 'nearest_forward' -> primer precio con timestamp >= t
      - 'previous' -> último precio con timestamp <= t
    """
    if df_prices.empty:
        return None
    if method == "previous":
        prev = df_prices[df_prices['timestamp'] <= t]
        return prev.tail(1).squeeze() if not prev.empty else None
    if method == "nearest_forward":
        ge = df_prices[df_prices['timestamp'] >= t]
        return ge.head(1).squeeze() if not ge.empty else None
    # nearest fallback
    diffs = (df_prices['timestamp'] - t).abs()
    idx = diffs.idxmin()
    return df_prices.loc[idx]

# -----------------------------
# Build dataset function
# -----------------------------
def build_for_symbol(symbol, ticker, interval="1h", window_hours=4, lookback_days=7,
                     newsapi_key=None, use_rss=False, out_csv=None):
    """
    symbol: p.ej. 'EURJPY'
    ticker: p.ej. 'EURJPY=X' (yfinance)
    interval: '1h','1d','5m'...
    window_hours: horizonte lookahead (ej. 4 -> price at t+4h)
    lookback_days: cuántos días de noticias/price buscar hacia atrás desde ahora
    newsapi_key: tu API key de NewsAPI (opcional)
    use_rss: si True usa RSS en lugar de NewsAPI
    """
    now = datetime.utcnow()
    from_dt = now - timedelta(days=lookback_days)
    to_dt = now

    keywords = KEYWORDS_BY_SYMBOL.get(symbol.upper(), [symbol])
    print(f"Keywords usadas para {symbol}: {keywords}")

    # 1) noticias
    if (newsapi_key is not None) and (not use_rss):
        try:
            print("Buscando noticias en NewsAPI...")
            articles = fetch_news_newsapi(keywords, newsapi_key, from_dt, to_dt)
        except Exception as e:
            print("NewsAPI falló:", e)
            print("Cayendo a RSS...")
            articles = fetch_news_rss(keywords, from_dt, to_dt)
    else:
        print("Usando RSS / scraping para noticias...")
        articles = fetch_news_rss(keywords, from_dt, to_dt)

    if not articles:
        print("⚠️ No se encontraron noticias. Intenta ampliar lookback_days o usar NewsAPI.")
        return None

    # normalizar artículos a DataFrame
    df_news = pd.DataFrame(articles)
    # parse timestamp
    df_news['timestamp'] = pd.to_datetime(df_news['timestamp'], errors='coerce')
    df_news = df_news.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    print(f"Noticias encontradas: {len(df_news)}")

    # 2) descargar precios
    prices = download_prices_yf(ticker, from_dt - timedelta(days=1), to_dt + timedelta(days=1), interval=interval)
    if prices is None or prices.empty:
        print("⚠️ No hay datos de precios. Deteniendo pipeline.")
        return None
    prices = prices.sort_values('timestamp').reset_index(drop=True)

    # 3) Emparejar y crear dataset
    rows = []
    for i, r in df_news.iterrows():
        t0 = r['timestamp'].to_pydatetime()
        price_at_t0 = get_price_at(prices, t0, method="nearest_forward")
        if price_at_t0 is None:
            continue
        t_exit = t0 + timedelta(hours=window_hours)
        price_exit = get_price_at(prices, t_exit, method="nearest_forward")
        if price_exit is None:
            continue
        p0 = float(price_at_t0['close'])
        pt = float(price_exit['close'])
        ret = (pt - p0) / p0
        # etiqueta ternaria opcional, aquí 3 clases: baja(0), estable(1), sube(2)
        thr = 0.0005 if interval in ("1m","5m","15m","60m","1h") else 0.002  # ajustar
        if ret > thr:
            label = 2
        elif ret < -thr:
            label = 0
        else:
            label = 1

        rows.append({
            "symbol": symbol,
            "ticker": ticker,
            "news_ts": r['timestamp'],
            "text": r['text'],
            "source": r.get('source'),
            "url": r.get('url'),
            "price_t0": p0,
            "price_texit": pt,
            "ret": ret,
            "label": label
        })

    if not rows:
        print("⚠️ No se emparejaron noticias con precios (quizá resolution o rango).")
        return None

    df_out = pd.DataFrame(rows)
    out_csv = out_csv or os.path.join(OUT_DIR, f"{symbol}_{ticker}_{interval}_w{window_hours}.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Dataset guardado en: {out_csv}")
    print(df_out[['news_ts','label','ret']].head(10))
    return df_out

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, required=True, help="Clave simbólica (p.ej. EURJPY)")
    p.add_argument("--ticker", type=str, required=True, help="Ticker para yfinance (p.ej. 'EURJPY=X')")
    p.add_argument("--interval", type=str, default="1h", help="Intervalo de precios: 1h, 1d, 5m, 60m ...")
    p.add_argument("--window", type=int, default=4, help="Horas hacia adelante para medir impacto (ej. 4)")
    p.add_argument("--lookback", type=int, default=14, help="Días hacia atrás para buscar noticias/precios")
    p.add_argument("--newsapi_key", type=str, default=None, help="Clave NewsAPI (opcional)")
    p.add_argument("--use_rss", action="store_true", help="Forzar uso de RSS en lugar de NewsAPI")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_for_symbol(
        symbol=args.symbol,
        ticker=args.ticker,
        interval=args.interval,
        window_hours=args.window,
        lookback_days=args.lookback,
        newsapi_key=args.newsapi_key,
        use_rss=args.use_rss
    )
