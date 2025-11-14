# embedding_lstm_multimodal.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

def build_model(vocab_size, max_len, numeric_shape, embedding_dim=128, dropout_rate=0.3):
    # text branch
    text_in = Input(shape=(max_len,), dtype='int32', name='text_input')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, mask_zero=True)(text_in)
    x = LSTM(128)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)

    # numeric branch
    num_in = Input(shape=numeric_shape, name='num_input')
    y = LSTM(64)(num_in)
    y = Dropout(0.2)(y)
    y = Dense(64, activation='relu')(y)

    # merge
    merged = Concatenate()([x, y])
    z = Dense(64, activation='relu')(merged)
    z = Dropout(dropout_rate)(z)
    z = Dense(32, activation='relu')(z)
    out = Dense(3, activation='softmax')(z)

    model = Model(inputs=[text_in, num_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
