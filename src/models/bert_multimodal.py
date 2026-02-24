# src/models/bert_multimodal.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from transformers import TFDistilBertModel

def build_bert_multimodal_model(
    bert_model_name,
    max_len,
    numeric_shape,
    bert_trainable=True,
    dropout_rate=0.3,
    lr=2e-5
):
    # ==========================================
    # 1. Rama de TEXTO (BERT)
    # ==========================================
    # Definir inputs explícitamente con nombres y tipos
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    
    # Cargar modelo base
    # IMPORTANTE: use_safetensors=False para evitar errores de carga en Colab
    bert = TFDistilBertModel.from_pretrained(bert_model_name, use_safetensors=False)
    bert.trainable = bert_trainable
    
    # Pasar inputs al modelo BERT
    bert_output = bert(input_ids=input_ids, attention_mask=attention_mask)
    
    # Obtener el token [CLS] (representación de la frase completa)
    # En DistilBert es el primer token del last_hidden_state (índice 0)
    cls_token = bert_output.last_hidden_state[:, 0, :]
    
    # Capa densa para procesar el texto
    x_text = layers.Dense(64, activation='relu')(cls_token)
    x_text = layers.Dropout(dropout_rate)(x_text)

    # ==========================================
    # 2. Rama NUMÉRICA (Precios)
    # ==========================================
    num_input = layers.Input(shape=numeric_shape, name='num_input')
    
    # === MEJORA CLAVE: Normalización ===
    # Normaliza los datos numéricos (precios) para que estén en una escala
    # que la red neuronal pueda aprender más rápido y mejor.
    norm = layers.BatchNormalization()(num_input) 
    
    x_num = layers.Flatten()(norm)
    x_num = layers.Dense(32, activation='relu')(x_num)
    x_num = layers.Dropout(dropout_rate)(x_num)

    # ==========================================
    # 3. Concatenación (Multimodal)
    # ==========================================
    concatenated = layers.Concatenate()([x_text, x_num])
    
    # Capas finales de clasificación
    x = layers.Dense(32, activation='relu')(concatenated)
    
    # Salida binaria (0 o 1) con sigmoide
    output = layers.Dense(1, activation='sigmoid')(x)

    # ==========================================
    # 4. Construir y compilar modelo
    # ==========================================
    model = models.Model(inputs=[input_ids, attention_mask, num_input], outputs=output)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model