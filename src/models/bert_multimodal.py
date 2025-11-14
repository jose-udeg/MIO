# src/models/bert_multimodal.py
"""
Modelo BERT (TF) + rama numérica (LSTM) multimodal.
Requiere: transformers (TFBertModel, BertTokenizerFast)
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Concatenate
from tensorflow.keras.models import Model
from transformers import TFBertModel

def build_bert_multimodal_model(bert_model_name,
                                max_len,
                                numeric_shape,
                                bert_trainable=False,
                                dropout_rate=0.3,
                                lr=2e-5):
    """
    - bert_model_name: cadena (ej. 'bert-base-uncased' o 'distilbert-base-uncased')
    - max_len: tamaño máximo de tokens (p.ej. 128)
    - numeric_shape: tuple (T, F) para la rama numérica
    - bert_trainable: si True ajusta pesos de BERT (fine-tune). Si False, BERT frozen.
    """
    # Cargar BERT (TF)
    bert = TFBertModel.from_pretrained(bert_model_name)

    # Inputs de texto (token ids + attention mask)
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

    # Pasar por BERT
    bert_outputs = bert(input_ids, attention_mask=attention_mask)
    # pooled_output: representación [CLS]
    pooled_output = bert_outputs.pooler_output  # shape (batch, hidden_size)

    # Opcional: permitir fine-tuning
    bert.trainable = bool(bert_trainable)

    # Capa densa sobre el embedding de texto
    x_text = Dense(128, activation='relu')(pooled_output)
    x_text = Dropout(dropout_rate)(x_text)
    x_text = Dense(64, activation='relu')(x_text)

    # Rama numérica: LSTM sobre secuencia de indicadores
    num_input = Input(shape=numeric_shape, name='num_input')  # (T, F)
    x_num = LSTM(64, name='lstm_num')(num_input)
    x_num = Dropout(0.2)(x_num)
    x_num = Dense(64, activation='relu')(x_num)

    # Fusionar ambas ramas
    merged = Concatenate(name='concat')([x_text, x_num])
    z = Dense(64, activation='relu')(merged)
    z = Dropout(dropout_rate)(z)
    z = Dense(32, activation='relu')(z)

    # Salida: 3 clases (baja, estable, sube)
    out = Dense(3, activation='softmax', name='out')(z)

    model = Model(inputs=[input_ids, attention_mask, num_input], outputs=out)

    # Compilar con LR pequeño (recomendado para transformers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
