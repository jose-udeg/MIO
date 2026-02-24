# src/train.py
import os
import tensorflow as tf
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models.bert_multimodal import bert_multimodal_model

# ========================
# CONFIGURACIÓN
# ========================
MODEL_PATH = "models_checkpoints/bert_multimodal_model"
TOKENIZER_NAME = "bert-base-uncased"
NEWS_CSV = "data/raw/news.csv"
PRICES_CSV = "data/raw/prices.csv"

# ========================
# CARGA DE DATOS
# ========================
def load_data():
    print("📥 Cargando datos...")
    news_df = pd.read_csv(NEWS_CSV)
    prices_df = pd.read_csv(PRICES_CSV)

    # Supongamos que 'text' es la columna de noticias
    texts = news_df['text'].astype(str)

    # Datos de precios simulados (deberían venir del CSV real)
    price_features = prices_df[['open', 'high', 'low', 'close']].values

    # Etiquetas simuladas (1 = sube, 0 = baja)
    labels = np.random.randint(0, 2, size=len(texts))

    return texts, price_features, labels

# ========================
# PREPROCESAMIENTO
# ========================
def preprocess_data(texts, prices, labels, tokenizer, max_length=128):
    print("🧠 Tokenizando textos...")
    encoding = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

    X_text = encoding['input_ids']
    X_prices = np.array(prices, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X_text, X_prices, y

# ========================
# ENTRENAMIENTO
# ========================
def train_model():
    # Tokenizador y datos
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    texts, prices, labels = load_data()

    X_text, X_prices, y = preprocess_data(texts, prices, labels, tokenizer)

    # Dividir en entrenamiento y validación
    X_text_train, X_text_val, X_prices_train, X_prices_val, y_train, y_val = train_test_split(
        X_text, X_prices, y, test_size=0.2, random_state=42
    )

    # Crear modelo
    model = bert_multimodal_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n🚀 Entrenando modelo...")
    history = model.fit(
        [X_text_train, X_prices_train],
        y_train,
        validation_data=([X_text_val, X_prices_val], y_val),
        epochs=3,
        batch_size=8
    )

    # Crear carpeta de modelos
    os.makedirs("models_checkpoints", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Modelo guardado en {MODEL_PATH}")

    return model, tokenizer

# ========================
# EJECUCIÓN
# ========================
if __name__ == "__main__":
    train_model()
