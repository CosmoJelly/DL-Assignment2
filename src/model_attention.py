import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Positional Encoding
def positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.constant(pos_encoding, dtype=tf.float32)

# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn = tf.keras.Sequential([
        layers.Dense(ff_dim, activation="relu"),
        layers.Dense(inputs.shape[-1])
    ])
    ffn_output = ffn(out1)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# Pure Attention Encoder
def create_attention_encoder_model(vocab_size, embedding_dim=64, max_len=80, num_heads=2, ff_dim=128, num_layers=1):
    
    input_left = layers.Input(shape=(max_len,), name="left_input")
    input_right = layers.Input(shape=(max_len,), name="right_input")

    embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
    pos_encoding = positional_encoding(max_len, embedding_dim)

    def encode(x):
        x = embedding(x)
        x = layers.Add()([x, pos_encoding[:, :max_len, :]])
        for _ in range(num_layers):
            x = transformer_encoder(x, head_size=embedding_dim, num_heads=num_heads, ff_dim=ff_dim)
        return layers.GlobalAveragePooling1D()(x)

    left_encoded = encode(input_left)
    right_encoded = encode(input_right)

    diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([left_encoded, right_encoded])
    dense = layers.Dense(64, activation="relu")(diff)
    dense = layers.Dropout(0.3)(dense)
    output = layers.Dense(1, activation="sigmoid")(dense)

    model = models.Model(inputs=[input_left, input_right], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model