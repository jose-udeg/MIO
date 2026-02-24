# src/train_bert.py
"""
Entrenador para el modelo BERT + numeric.
Asume que ya ejecutaste src/build_dataset.py y existen:
- data/processed/X_text.npy   (array de strings)
- data/processed/X_num.npy    (np.ndarray shape (N, T, F))
- data/processed/y.npy        (labels 0/1/2)
- data/processed/timestamps.npy (opcional)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from src import config
from src.models.bert_multimodal import build_bert_multimodal_model

# Parámetros (puedes moverlos a config.py si prefieres)
BERT_NAME = "distilbert-base-uncased"  # Cambiamos a DistilBERT
MAX_LEN = config.MAX_LEN if hasattr(config, 'MAX_LEN') else 128
BATCH_SIZE = 8                         # transformers con TF suelen usar batches pequeños
EPOCHS = 4
LR = 2e-5
FINE_TUNE_BERT = True               # True para ajustar pesos de BERT (requiere GPU)

# Ruta processed
PROC_DIR = config.PROCESSED_DIR if hasattr(config, 'PROCESSED_DIR') else os.path.join("data","processed")

def load_processed():
    X_text = np.load(os.path.join(PROC_DIR, "X_text.npy"), allow_pickle=True)
    X_num = np.load(os.path.join(PROC_DIR, "X_num.npy"))
    y = np.load(os.path.join(PROC_DIR, "y.npy"))
    try:
        ts = np.load(os.path.join(PROC_DIR, "timestamps.npy"), allow_pickle=True)
    except:
        ts = None
    return X_text, X_num, y, ts

def temporal_train_test_split(X_text, X_num, y, ts, test_size=0.2):
    """
    Hace split temporal: usa las muestras más antiguas para train y las más recientes para test.
    Si no hay timestamps, hace stratified split aleatorio.
    """
    if ts is None:
        return train_test_split(X_text, X_num, y, test_size=test_size, random_state=config.RANDOM_SEED, stratify=y)
    order = np.argsort(ts)
    N = len(order)
    cutoff = int(N * (1 - test_size))
    train_idx = order[:cutoff]
    test_idx = order[cutoff:]
    return X_text[train_idx], X_text[test_idx], X_num[train_idx], X_num[test_idx], y[train_idx], y[test_idx]

def main():
    print("Cargando datos procesados...")
    X_text, X_num, y, ts = load_processed()
    print("Muestras totales:", len(y))
    # tokenizador BERT
    print("Cargando tokenizer:", BERT_NAME)
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_NAME)

    # tokenizar textos (retorna tf tensors)
    enc = tokenizer(list(X_text.astype(str)), truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='np')
    input_ids_all = enc['input_ids']
    attention_mask_all = enc['attention_mask']

    # split temporal (preferible)
    X_text_tr, X_text_te, X_num_tr, X_num_te, y_tr, y_te = temporal_train_test_split(input_ids_all, X_num, y, ts, test_size=0.2)
    att_tr = attention_mask_all[np.argsort(ts)[:len(X_text_tr)]] if ts is not None else None
    # The above att_tr logic is fragile; simpler approach: split using same indices
    # Let's create indices properly:
    if ts is not None:
        order = np.argsort(ts)
        cutoff = int(len(order)*(1-0.2))
        train_idx = order[:cutoff]
        test_idx = order[cutoff:]
        input_ids_tr = input_ids_all[train_idx]
        input_ids_te = input_ids_all[test_idx]
        att_tr = attention_mask_all[train_idx]
        att_te = attention_mask_all[test_idx]
    else:
        # fallback: random stratified split based on labels (but temporal is better)
        input_ids_tr, input_ids_te, att_tr, att_te, _tmp1, _tmp2 = train_test_split(
            input_ids_all, attention_mask_all, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
        )
        # align numeric and labels with same split (use same indices)
        # Recompute X_num_tr, X_num_te, y_tr, y_te using stratified split indices
        # Simpler: use sklearn to split all jointly
        input_ids_tr, input_ids_te, att_tr, att_te, X_num_tr, X_num_te, y_tr, y_te = train_test_split(
            input_ids_all, attention_mask_all, X_num, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
        )

    # If ts provided we already have input_ids_tr/te above; else it's set
    if ts is not None:
        # already have input_ids_tr/input_ids_te, att_tr, att_te and X_num_tr etc were set earlier
        X_num_tr = X_num[train_idx]
        X_num_te = X_num[test_idx]
        # labels
        y_tr = y[train_idx]
        y_te = y[test_idx]
    else:
        # previously set by fallback splitting
        pass

    print("Tamaños train/test:", input_ids_tr.shape, X_num_tr.shape, y_tr.shape, input_ids_te.shape, X_num_te.shape, y_te.shape)

    # balanceo de clases (class weights)
    cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_tr), y=y_tr)
    cw_dict = {i: cw[i] for i in range(len(cw))}
    print("Class weights:", cw_dict)

    # construir modelo
    numeric_shape = X_num_tr.shape[1:]  # (T, F)
    model = build_bert_multimodal_model(
        bert_model_name=BERT_NAME,
        max_len=MAX_LEN,
        numeric_shape=numeric_shape,
        bert_trainable=FINE_TUNE_BERT,
        dropout_rate=0.3,
        lr=LR
    )
    model.summary()

    # Callback: guardar mejor modelo por val_loss
    out_dir = config.MODEL_DIR
    os.makedirs(out_dir, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, "bert_fx_best.h5"), monitor='val_loss', save_best_only=True, save_weights_only=False)
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Entrenar
    history = model.fit(
        x={'input_ids': input_ids_tr, 'attention_mask': att_tr, 'num_input': X_num_tr},
        y=y_tr,
        validation_data=({'input_ids': input_ids_te, 'attention_mask': att_te, 'num_input': X_num_te}, y_te),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw_dict,
        callbacks=[ckpt, early]
    )

    # Evaluación final
    preds_proba = model.predict({'input_ids': input_ids_te, 'attention_mask': att_te, 'num_input': X_num_te}, batch_size=BATCH_SIZE)
    preds = np.argmax(preds_proba, axis=1)
    print("=== Classification report ===")
    print(classification_report(y_te, preds, digits=4))
    print("=== Confusion matrix ===")
    print(confusion_matrix(y_te, preds))

    # Guardar tokenizer + configuración de BERT usada
    tokenizer.save_pretrained(os.path.join(out_dir, "bert_tokenizer"))
    # Guardar modelo final (weights ya guardadas por checkpoint)
    model.save(os.path.join(out_dir, "bert_multimodal_model"), include_optimizer=False)
    print("Modelo y tokenizer guardados en:", out_dir)

if __name__ == "__main__":
    main()
