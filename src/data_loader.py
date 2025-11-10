import os
import glob
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Config
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MAX_VOCAB = 20000
MAX_LEN = 120
TEST_SIZE = 0.15
VAL_SIZE = 0.10
BATCH_SIZE = 32
OOV_TOKEN = "<OOV>"

def clean_text(s: str) -> str:

    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\n", " ")
    return s

def load_all_clauses(data_dir: str) -> pd.DataFrame:

    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "clause_text" not in df.columns or "clause_type" not in df.columns:
            raise ValueError(f"{p} must contain 'clause_text' and 'clause_type' columns.")
        df["clause_text"] = df["clause_text"].astype(str)
        dfs.append(df[["clause_text", "clause_type"]])
    combined = pd.concat(dfs, ignore_index=True)
    combined["text_clean"] = combined["clause_text"].apply(clean_text)
    return combined

def create_pairs(df: pd.DataFrame):

    type_to_indices = defaultdict(list)
    for idx, t in enumerate(df["clause_type"]):
        type_to_indices[t].append(idx)

    # Positive pairs
    pos_pairs = []
    for t, idxs in type_to_indices.items():
        if len(idxs) < 2:
            continue
        n_samples = min(len(idxs) * 2, (len(idxs)*(len(idxs)-1))//2)
        chosen = set()
        while len(chosen) < n_samples:
            a, b = random.sample(idxs, 2)
            key = tuple(sorted((a, b)))
            if key not in chosen:
                chosen.add(key)
                pos_pairs.append((a, b, 1))

    # Negative pairs
    all_idx = list(range(len(df)))
    neg_pairs = []
    while len(neg_pairs) < len(pos_pairs):
        a, b = random.sample(all_idx, 2)
        if df.loc[a, "clause_type"] != df.loc[b, "clause_type"]:
            neg_pairs.append((a, b, 0))

    pairs = pos_pairs + neg_pairs
    random.shuffle(pairs)
    return pairs

def build_datasets(df: pd.DataFrame):

    pairs = create_pairs(df)
    left_texts = [df.loc[a, "text_clean"] for a, b, _ in pairs]
    right_texts = [df.loc[b, "text_clean"] for a, b, _ in pairs]
    labels = np.array([lbl for _, _, lbl in pairs], dtype=np.int32)

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(df["text_clean"].tolist())

    seq_left = tokenizer.texts_to_sequences(left_texts)
    seq_right = tokenizer.texts_to_sequences(right_texts)

    X_left = pad_sequences(seq_left, maxlen=MAX_LEN, padding="post", truncating="post")
    X_right = pad_sequences(seq_right, maxlen=MAX_LEN, padding="post", truncating="post")

    X = np.stack([X_left, X_right], axis=1)
    y = labels

    # Split train/val/test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, random_state=SEED, stratify=y_tmp
    )

    def make_tf_dataset(Xp, yp, shuffle=True):
        left = Xp[:, 0, :]
        right = Xp[:, 1, :]
        ds = tf.data.Dataset.from_tensor_slices(((left, right), yp))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(yp), seed=SEED)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = make_tf_dataset(X_train, y_train)
    val_ds = make_tf_dataset(X_val, y_val, shuffle=False)
    test_ds = make_tf_dataset(X_test, y_test, shuffle=False)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    print(f"[INFO] Tokenizer saved to artifacts/tokenizer.json")
    print(f"[INFO] Train/Val/Test sizes: {len(X_train)}, {len(X_val)}, {len(X_test)}")
    return train_ds, val_ds, test_ds, tokenizer