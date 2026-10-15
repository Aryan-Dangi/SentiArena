import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, SimpleRNN, LSTM, GRU, Dense, Dropout

SEED = 42
VOCAB_SIZE = 10000
MAX_LENGTH = 250
EPOCHS = 3
BATCH_SIZE = 128
MODEL_DIR = "models"

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)


def build_model(model_type):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, 64, input_length=MAX_LENGTH))

    if model_type == "CNN":
        model.add(Conv1D(64, 5, activation="relu"))
        model.add(GlobalMaxPooling1D())
    elif model_type == "RNN":
        model.add(SimpleRNN(48))
    elif model_type == "LSTM":
        model.add(LSTM(48))
    elif model_type == "GRU":
        model.add(GRU(48))

    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("Loading the IMDb movie-review dataset...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, padding="post", truncating="post")

    results = []
    for model_type in ["CNN", "RNN", "LSTM", "GRU"]:
        print(f"\nTraining {model_type}...")
        model = build_model(model_type)
        start = time.time()
        history = model.fit(
            x_train, np.asarray(y_train),
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        loss, accuracy = model.evaluate(x_test, np.asarray(y_test), verbose=0)
        elapsed = round(time.time() - start, 1)
        model.save(os.path.join(MODEL_DIR, f"{model_type.lower()}.keras"))

        results.append({
            "model": model_type,
            "test_accuracy": round(float(accuracy), 4),
            "test_loss": round(float(loss), 4),
            "training_seconds": elapsed,
            "last_val_accuracy": round(float(history.history["val_accuracy"][-1]), 4)
        })
        print(f"{model_type}: test accuracy={accuracy:.4f}, time={elapsed}s")

    with open(os.path.join(MODEL_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    word_index = imdb.get_word_index()
    with open(os.path.join(MODEL_DIR, "word_index.json"), "w", encoding="utf-8") as f:
        json.dump(word_index, f)

    print("\nDone. Start the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
