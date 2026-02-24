# src/evaluate.py
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.models.bert_multimodal import bert_multimodal_model

# ========================
# CONFIGURACIÓN
# ========================
MODEL_PATH = "models_checkpoints/bert_multimodal_model"
TOKENIZER_NAME = "bert-base-uncased"
NEWS_CSV = "data/raw/news.csv"
PRICES_CSV = "data/raw/prices.csv"

# ========================
# CARGA DE MODELO Y TOKENIZER
# ========================
print("Cargando modelo y tokenizer...")
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

# Si ya está guardado el modelo entrenado
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'TFBertModel': tf.keras.Model})
    print("Modelo cargado desde disco.")
except Exception as e:
    print("⚠️ No se encontró modelo entrenado, creando uno nuevo.")
    model = bert_multimodal_model()
    model.build([(None, 128), (None, 4)])

# ========================
# FUNCIONES AUXILIARES
# ========================
def preprocess_texts(texts, tokenizer, max_length=128):
    """Tokeniza textos con el tokenizer de BERT."""
    encoding = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    return encoding['input_ids']

def generate_fake_price_data(n):
    """Genera datos de precios simulados (para pruebas)."""
    return np.random.rand(n, 4)

# ========================
# EVALUACIÓN
# ========================
def evaluate_model():
    # Cargar dataset (aquí se usa una muestra simple)
    news_df = pd.read_csv(NEWS_CSV)
    texts = news_df['text'].astype(str)

    # Preprocesar texto y precios
    text_inputs = preprocess_texts(texts, tokenizer)
    price_data = generate_fake_price_data(len(texts))

    # Obtener predicciones
    predictions = model.predict([text_inputs, price_data])
    labels = (predictions > 0.5).astype(int).flatten()

    # Mostrar resultados
    print("\n=== Ejemplo de predicciones ===")
    for i in range(min(5, len(texts))):
        print(f"\n📰 Noticia: {texts.iloc[i]}")
        print(f"📊 Predicción (1=Sube, 0=Baja): {labels[i]} | Prob: {predictions[i][0]:.3f}")

if __name__ == "__main__":
    evaluate_model()
