# src/generate_raw_data.py
"""
Genera dos archivos:
- data/raw/news.csv  -> con noticias de NewsAPI
- data/raw/prices.csv -> con datos de precios de Yahoo Finance
Estos archivos luego se usan con build_dataset.py
"""

import os
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from datetime import datetime, timedelta

# CONFIGURA AQUÍ
API_KEY = "2fdb14780bc5468ca967fff469478238"
SYMBOL = "EURJPY"           # puedes cambiarlo por USDJPY, XAUUSD, etc.
TICKER = "EURJPY=X"         # símbolo de Yahoo Finance
NEWS_QUERY = ["EURJPY", "euro yen", "Bank of Japan", "ECB"]
NEWS_DAYS = 28               # días hacia atrás para obtener noticias
PRICE_PERIOD = "60d"        # periodo para descargar precios
PRICE_INTERVAL = "1h"       # intervalo de datos
RAW_DIR = "data/raw"

os.makedirs(RAW_DIR, exist_ok=True)

# 🔹 1. Obtener noticias desde NewsAPI
print(f"📰 Descargando noticias de los últimos {NEWS_DAYS} días...")
newsapi = NewsApiClient(api_key=API_KEY)
from_date = (datetime.utcnow() - timedelta(days=NEWS_DAYS)).strftime("%Y-%m-%d")
all_articles = []

for term in NEWS_QUERY:
    res = newsapi.get_everything(
        q=term,
        from_param=from_date,
        language='en',
        sort_by='relevancy',
        page_size=100,
    )
    for art in res.get('articles', []):
        all_articles.append({
            "symbol": SYMBOL,
            "publishedAt": art['publishedAt'],
            "title": art['title'],
            "description": art['description'],
            "content": art['content'],
            "source": art['source']['name'],
            "url": art['url']
        })

df_news = pd.DataFrame(all_articles).drop_duplicates(subset="title").dropna(subset=["title"])
news_path = os.path.join(RAW_DIR, "news.csv")
df_news.to_csv(news_path, index=False)
print(f"✅ Archivo de noticias guardado en {news_path} ({len(df_news)} noticias)")

# 🔹 2. Descargar precios históricos desde Yahoo Finance
print(f"📈 Descargando precios de {TICKER} ({PRICE_PERIOD}, {PRICE_INTERVAL}) ...")
df_prices = yf.download(TICKER, period=PRICE_PERIOD, interval=PRICE_INTERVAL, progress=False)
df_prices = df_prices.reset_index().rename(columns={"Date": "timestamp"})
price_path = os.path.join(RAW_DIR, "prices.csv")
df_prices.to_csv(price_path, index=False)
print(f"✅ Archivo de precios guardado en {price_path} ({len(df_prices)} registros)")
