import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.data_loader import load_all_clauses, build_datasets
from src.model_bilstm import create_bilstm_model
from src.model_attention import create_attention_encoder_model


def evaluate_model(model, test_ds, model_name="model"):

    y_true, y_pred = [], []
    for (l, r), y in test_ds:
        preds = model.predict([l, r])
        y_true.extend(y.numpy())
        y_pred.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_bin = (y_pred > 0.5).astype(int)

    precision = precision_score(y_true, y_bin)
    recall = recall_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f"\n[{model_name.upper()} RESULTS]")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Save metrics
    with open(f"artifacts/{model_name}_results.txt", "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")


def train_model(model_type="bilstm", data_dir="data", epochs=10):

    df = load_all_clauses(data_dir)
    train_ds, val_ds, test_ds, tokenizer = build_datasets(df)
    vocab_size = min(len(tokenizer.word_index) + 1, 20000)

    if model_type.lower() == "bilstm":
        model = create_bilstm_model(vocab_size=vocab_size, max_len=120)
    elif model_type.lower() == "attention":
        model = create_attention_encoder_model(vocab_size=vocab_size, max_len=120)
    else:
        raise ValueError("Invalid model_type. Choose 'bilstm' or 'attention'.")

    os.makedirs("artifacts", exist_ok=True)
    ckpt_path = f"artifacts/{model_type}_best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    print(f"[INFO] Training {model_type.upper()} model...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title(f"{model_type.upper()} Training Accuracy")
    plt.legend()
    plt.savefig(f"artifacts/{model_type}_accuracy.png")

    evaluate_model(model, test_ds, model_name=model_type)
    print(f"[INFO] Best model saved at: {ckpt_path}")


if __name__ == "__main__":
    train_model(model_type="attention", epochs=50)