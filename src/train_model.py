import pandas as pd
import numpy as np
import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# ⚙️ Argumentos por línea de comando
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="Entrenar modelo LSTM multimodal (texto + mercado).")
parser.add_argument('--data', type=str, default='data/raw/news.csv',
                    help='Ruta del archivo CSV con las noticias y etiquetas.')
parser.add_argument('--epochs', type=int, default=15,
                    help='Número de épocas de entrenamiento.')
args = parser.parse_args()

# -------------------------------------------------------------
# 📂 Cargar dataset
# -------------------------------------------------------------
print("📂 Cargando dataset desde:", args.data)
df = pd.read_csv(args.data)

# Verificamos columnas esperadas
expected_cols = {'text', 'label'}
if not expected_cols.issubset(df.columns):
    print("⚠️ El archivo no contiene columna 'text'. Se generará una columna vacía (texto simulado).")
    df['text'] = "No news content"
if 'label' not in df.columns:
    raise ValueError("❌ El dataset no tiene una columna llamada 'label'.")

# Asegurar que las etiquetas sean enteros
df['label'] = df['label'].astype(int)

# -------------------------------------------------------------
# 🧠 Tokenización de texto
# -------------------------------------------------------------
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# -------------------------------------------------------------
# ✂️ Separar datos
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    padded, df['label'], test_size=0.2, random_state=42
)

# -------------------------------------------------------------
# 🧩 Construcción del modelo LSTM
# -------------------------------------------------------------
print("🧠 Construyendo modelo...")

model = Sequential([
    Embedding(5000, 64, input_length=100),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# -------------------------------------------------------------
# 🚀 Entrenamiento
# -------------------------------------------------------------
print("🚀 Entrenando modelo...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=args.epochs,
    batch_size=8,
    verbose=1
)

# -------------------------------------------------------------
# 💾 Guardar modelo y tokenizer
# -------------------------------------------------------------
os.makedirs("models", exist_ok=True)
model_path = f"models/forex_sentiment_model.keras"
tokenizer_path = f"models/vectorizer.pkl"

# Guardar el modelo
model.save(model_path)

# Guardar el tokenizer
import joblib
joblib.dump(tokenizer, tokenizer_path)

print(f"✅ Modelo entrenado y guardado en: {model_path}")
print(f"✅ Tokenizer guardado en: {tokenizer_path}")
