import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from textblob import TextBlob
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import yfinance as yf

# ======================
# CONFIGURACIÓN
# ======================
MODEL_PATH = "models/forex_sentiment_model.h5"
NEWS_API_KEY = "2fdb14780bc5468ca967fff469478238"  # ← usa tu propia API key
SYMBOL = "EURJPY"
TICKER = "EURJPY=X"
INTERVAL = "1h"
LOOKBACK_HOURS = 24  # cuántas horas atrás buscar noticias

# ======================
# FUNCIONES AUXILIARES
# ======================
def fetch_recent_news(symbol, hours=LOOKBACK_HOURS):
    """Descarga noticias recientes relacionadas con el símbolo."""
    print(f"📰 Buscando noticias recientes de {symbol}...")
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    # Usar solo la fecha sin hora para evitar problemas de formato
    from_date = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%d")
    all_articles = newsapi.get_everything(
        q=symbol,
        from_param=from_date,  # Usar el formato YYYY-MM-DD
        language="en",
        sort_by="publishedAt",
        page_size=20,
    )
    articles = all_articles.get("articles", [])
    news_df = pd.DataFrame(
        [(a["title"], a["description"], a["publishedAt"]) for a in articles],
        columns=["title", "description", "timestamp"],
    )
    return news_df

def analyze_sentiment(text):
    """Calcula el sentimiento textual con TextBlob."""
    blob = TextBlob(str(text))
    return blob.sentiment.polarity  # -1 (negativo) a +1 (positivo)

# ======================
# PREDICCIÓN PRINCIPAL
# ======================
def predict_sentiment():
    print("🔹 Cargando modelo entrenado...")
    model = load_model(MODEL_PATH)

    print("📊 Descargando precios recientes...")
    end = datetime.utcnow()
    start = end - timedelta(days=2)
    data = yf.download(TICKER, start=start, end=end, interval=INTERVAL)
    last_price = data["Close"].iloc[-1]

    print("🧠 Procesando noticias...")
    news = fetch_recent_news(SYMBOL)
    if news.empty:
        print("⚠️ No se encontraron noticias recientes.")
        return

    # Combinar título y descripción
    news["text"] = news["title"].fillna('') + " " + news["description"].fillna('')
    news["sentiment"] = news["text"].apply(analyze_sentiment)

    # Tokenización simple (igual que en entrenamiento)
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(news["text"])
    sequences = tokenizer.texts_to_sequences(news["text"])
    padded = pad_sequences(sequences, maxlen=100)

    print("🤖 Realizando predicciones con el modelo...")
    preds = model.predict(padded)
    news["pred_label"] = np.argmax(preds, axis=1)

    # Etiquetas
    label_map = {0: "Baja 📉", 1: "Sube 📈", 2: "Neutra ➖"}
    news["pred_label_text"] = news["pred_label"].map(label_map)

    print("\n✅ Resultados de predicción:")
    print(news[["timestamp", "title", "sentiment", "pred_label_text"]])

    # ======================
    # ANÁLISIS GLOBAL
    # ======================
    avg_sentiment = news["sentiment"].mean()
    bullish = (news["pred_label"] == 1).sum()
    bearish = (news["pred_label"] == 0).sum()
    neutral = (news["pred_label"] == 2).sum()

    print("\n📊 Sentimiento promedio del mercado: {:.3f}".format(avg_sentiment))
    print(f"📈 Alcistas: {bullish} | 📉 Bajistas: {bearish} | ➖ Neutras: {neutral}")
    print(f"💱 Precio actual de {SYMBOL}: {last_price:.3f}")

    # ======================
    # RECOMENDACIÓN AUTOMÁTICA
    # ======================
    if bullish > bearish and avg_sentiment > 0.1:
        recommendation = "✅ RECOMENDACIÓN: COMPRAR (Buy)"
    elif bearish > bullish and avg_sentiment < -0.1:
        recommendation = "⚠️ RECOMENDACIÓN: VENDER (Sell)"
    else:
        recommendation = "➖ RECOMENDACIÓN: MANTENER (Hold)"

    print("\n💡 " + recommendation)

    # ======================
    # GUARDAR RESULTADOS
    # ======================
    news.to_csv(f"predictions_{SYMBOL}.csv", index=False)
    print(f"\n📁 Resultados guardados en: predictions_{SYMBOL}.csv")

# ======================
# EJECUCIÓN
# ======================
if __name__ == "__main__":
    predict_sentiment()
