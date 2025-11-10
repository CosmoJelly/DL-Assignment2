import tensorflow as tf
from tensorflow.keras import layers, models

def create_bilstm_model(vocab_size, embedding_dim=128, max_len=120):

    input_left = layers.Input(shape=(max_len,), name="left_input")
    input_right = layers.Input(shape=(max_len,), name="right_input")

    embedding = layers.Embedding(vocab_size, embedding_dim, input_length=max_len, mask_zero=True)
    encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=False, dropout=0.3))

    left_encoded = encoder(embedding(input_left))
    right_encoded = encoder(embedding(input_right))

    diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([left_encoded, right_encoded])
    dense = layers.Dense(128, activation="relu")(diff)
    dense = layers.Dropout(0.3)(dense)
    output = layers.Dense(1, activation="sigmoid")(dense)

    model = models.Model(inputs=[input_left, input_right], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model