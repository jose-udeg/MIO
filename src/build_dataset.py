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

# Umbral (en porcentaje) para decidir si un movimiento es significativo
# 0.0005 = 0.05% de cambio.
PRICE_CHANGE_THRESHOLD = 0.0005

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

    # --- CORRECCIÓN 1: Asegurar que los precios sean numéricos ---
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        prices_df[col] = pd.to_numeric(prices_df[col], errors='coerce')
    prices_df = prices_df.dropna(subset=price_cols)
    # -----------------------------------------------------------

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

        # Evitar división por cero
        if close_before == 0:
            continue

        # Calcular cambio porcentual
        pct_change = (close_after - close_before) / close_before

        # --- CORRECCIÓN 2: Etiquetado con umbral ---
        if pct_change > PRICE_CHANGE_THRESHOLD:
            label = 1  # Sube significativamente
        elif pct_change < -PRICE_CHANGE_THRESHOLD:
            label = 0  # Baja significativamente
        else:
            label = 2  # Neutral (ruido)
        
        labels.append(label)
        matched_prices.append(price_before[['Open', 'High', 'Low', 'Close']].values[0])

    # --- CORRECCIÓN 3: Construir DataFrame INCLUYENDO timestamp ---
    processed_df = pd.DataFrame({
        'timestamp': news_df['timestamp'][:len(labels)].values, # IMPORTANTE: Incluir fecha aquí
        'text': news_df['text'][:len(labels)].values,
        'label': labels,
        'open': [p[0] for p in matched_prices],
        'high': [p[1] for p in matched_prices],
        'low': [p[2] for p in matched_prices],
        'close': [p[3] for p in matched_prices]
    })

    # --- CORRECCIÓN 4: Filtrar el ruido (eliminar etiqueta 2) ---
    print(f"Total de muestras antes de filtrar: {len(processed_df)}")
    processed_df = processed_df[processed_df['label'] != 2].reset_index(drop=True)
    print(f"Total de muestras después de filtrar (solo 0 y 1): {len(processed_df)}")

    # Guardar el dataset procesado en CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    processed_df.to_csv(OUTPUT_CSV, index=False)
    
    # --- CORRECCIÓN 5: Guardar archivos .npy usando el DataFrame FILTRADO ---
    # Así aseguramos que todos tengan el mismo tamaño (ej. 117)
    np.save(os.path.join('data/processed', 'X_text.npy'), processed_df['text'].values)
    np.save(os.path.join('data/processed', 'X_num.npy'), processed_df[['open', 'high', 'low', 'close']].values)
    np.save(os.path.join('data/processed', 'y.npy'), processed_df['label'].values)
    np.save(os.path.join('data/processed', 'timestamps.npy'), processed_df['timestamp'].values)
    
    print(f"✅ Dataset procesado guardado en {OUTPUT_CSV}")
    print("✅ Archivos .npy generados en data/processed/")

if __name__ == "__main__":
    build_dataset()