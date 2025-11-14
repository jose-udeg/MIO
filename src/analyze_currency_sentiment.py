import os
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from tensorflow.keras.models import load_model
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Ruta al modelo entrenado
MODEL_PATH = "models/forex_sentiment_model.keras"

# === 1. Obtener precios desde Yahoo Finance ===
def get_currency_prices(currency_pair="EURJPY=X", period="5d", interval="1h"):
    """
    Obtiene precios recientes de divisas usando Yahoo Finance.
    Ejemplo de pares válidos:
        - EURUSD=X
        - USDJPY=X
        - GBPUSD=X
        - EURJPY=X
    """
    try:
        df = yf.download(tickers=currency_pair, period=period, interval=interval)
        if df.empty:
            print("⚠️ No se obtuvieron datos desde Yahoo Finance.")
            return None
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close"
        })
        return df
    except Exception as e:
        print(f"⚠️ Error al obtener datos: {e}")
        return None


# === 2. Analizar sentimiento ===
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 1
    elif sentiment < 0:
        return -1
    else:
        return 0


# === 3. Graficar precios y predicciones ===
def plot_price_prediction(actual_prices, predicted_prices, currency_pair="EURJPY=X"):
    os.makedirs("outputs/plots", exist_ok=True)
    output_path = f"outputs/plots/{currency_pair.replace('=X','')}_prediction.png"

    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices, label="Precio Real", marker='o', color='blue')
    plt.plot(predicted_prices, label="Predicción del Modelo", linestyle='--', color='red')

    plt.title(f"Predicción de tendencia para {currency_pair.replace('=X','')}")
    plt.xlabel("Tiempo (últimos registros)")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Gráfica generada en: {output_path}")


# === 4. Flujo principal ===
if __name__ == "__main__":
    currency_pair = "EURJPY=X"

    print(f"📊 Obteniendo datos de {currency_pair} desde Yahoo Finance...")
    df = get_currency_prices(currency_pair)

    if df is None or df.empty:
        print("⚠️ No se pudieron obtener datos del mercado.")
        exit()

    print("✅ Datos de mercado obtenidos.")

    # Cargar el modelo
    print("📦 Cargando modelo...")
    model = load_model(MODEL_PATH)
    print("✅ Modelo cargado correctamente.")

    # Precios reales (cierres)
    actual_prices = df["close"].values

    # Preparar los datos para el modelo
    if len(actual_prices) < 10:
        print("⚠️ No hay suficientes datos para predecir.")
        exit()

    # Reshape los datos para que coincidan con la forma esperada por el modelo
    X_input = actual_prices[-10:].reshape(1, -1)  # Reshape a (1, 10)
    # Asegurar que tenga la longitud correcta (pad con ceros si es necesario)
    if X_input.shape[1] < 100:
        X_input = np.pad(X_input, ((0, 0), (0, 100 - X_input.shape[1])), 'constant')
    
    prediction = model.predict(X_input)
    predicted_prices = np.append(actual_prices[:-1], prediction.flatten())

    # Generar gráfica
    plot_price_prediction(actual_prices, predicted_prices, currency_pair)

    print("🎯 Análisis completado.")
