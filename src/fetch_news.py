# fetch_news.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# Pon tu API key aquí o como variable de entorno
NEWSAPI_KEY = "2fdb14780bc5468ca967fff469478238"

def fetch_news(keywords=["eur","jpy","euro","yen","bank of japan","boj","forex"], days_back=7,
               language="en", page_size=100, out_csv="data/raw/news.csv"):
    """
    Descarga titulares relacionados con keywords en los últimos days_back días.
    Devuelve DataFrame con columns: timestamp (datetime), title, description, text (title+desc)
    """
    print("Descargando noticias con NewsAPI...")
    url = "https://newsapi.org/v2/everything"
    to_date = datetime.utcnow()
    from_date = to_date - timedelta(days=days_back)
    all_articles = []

    for q in keywords:
        params = {
            "q": q,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": NEWSAPI_KEY
        }
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            print("NewsAPI error:", r.status_code, r.text[:200])
            continue
        data = r.json()
        for a in data.get("articles", []):
            ts = a.get("publishedAt")
            # some articles might have None
            if ts:
                all_articles.append({
                    "timestamp": ts,
                    "title": a.get("title") or "",
                    "description": a.get("description") or "",
                    "source": a.get("source", {}).get("name",""),
                    "url": a.get("url","")
                })
    if not all_articles:
        print("No se obtuvieron noticias (revisa tu API key o keywords).")
        return pd.DataFrame()
    df = pd.DataFrame(all_articles)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['text'] = (df['title'] + " - " + df['description']).str.strip()
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Guardadas {len(df)} noticias en {out_csv}")
    return df

if __name__ == "__main__":
    fetch_news()
