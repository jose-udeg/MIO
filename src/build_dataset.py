# src/build_dataset.py
import pandas as pd
import numpy as np
import os

# ========================
# CONFIGURACIÓN
# ========================
NEWS_CSV = "data/raw/news.csv"
PRICES_CSV = "data/raw/prices.csv"
OUTPUT_CSV = "data/processed/dataset.csv"

# Cuántas horas después de la noticia medimos el cambio
TIME_WINDOW_HOURS = 1  

# ========================
# FUNCIÓN PRINCIPAL
# ========================
def build_dataset():
    print("📥 Cargando datos...")
    news_df = pd.read_csv(NEWS_CSV)
    prices_df = pd.read_csv(PRICES_CSV)

    # Renombrar columnas para que coincidan
    news_df = news_df.rename(columns={'publishedAt': 'timestamp', 'title': 'text'})
    prices_df = prices_df.rename(columns={'Datetime': 'timestamp'})

    # Asegurar formato de tiempo
    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
    prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])

    labels = []
    matched_prices = []

    print("🔗 Emparejando noticias con precios...")
    for i, row in news_df.iterrows():
        t_news = row['timestamp']

        # Precio justo antes de la noticia
        price_before = prices_df[prices_df['timestamp'] <= t_news].tail(1)
        # Precio después del evento (por ejemplo, una hora después)
        price_after = prices_df[prices_df['timestamp'] > t_news].head(1)

        if len(price_before) == 0 or len(price_after) == 0:
            continue  # Si no hay datos suficientes, saltar

        close_before = price_before['Close'].values[0]
        close_after = price_after['Close'].values[0]

        # Etiqueta: 1 si sube, 0 si baja
        label = 1 if close_after > close_before else 0
        labels.append(label)

        matched_prices.append(price_before[['Open', 'High', 'Low', 'Close']].values[0])

    # Construir DataFrame final
    processed_df = pd.DataFrame({
        'text': news_df['text'][:len(labels)],
        'label': labels,
        'open': [p[0] for p in matched_prices],
        'high': [p[1] for p in matched_prices],
        'low': [p[2] for p in matched_prices],
        'close': [p[3] for p in matched_prices]
    })

    # Guardar el dataset procesado
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    processed_df.to_csv(OUTPUT_CSV, index=False)
    
    # Guardar arrays numpy para train_bert.py
    np.save(os.path.join('data/processed', 'X_text.npy'), processed_df['text'].values)
    np.save(os.path.join('data/processed', 'X_num.npy'), 
            processed_df[['open', 'high', 'low', 'close']].values)
    np.save(os.path.join('data/processed', 'y.npy'), processed_df['label'].values)
    np.save(os.path.join('data/processed', 'timestamps.npy'), 
            pd.to_datetime(news_df['timestamp'][:len(labels)]).values)
    
    print(f"✅ Dataset procesado guardado en {OUTPUT_CSV}")
    print("✅ Archivos .npy generados en data/processed/")


if __name__ == "__main__":
    build_dataset()
