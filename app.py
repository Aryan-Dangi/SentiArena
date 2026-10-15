import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="SentimentLab", page_icon="💬", layout="wide")

MODEL_DIR = "models"
VOCAB_SIZE = 10000
MAX_LENGTH = 250
MODEL_NAMES = ["CNN", "RNN", "LSTM", "GRU"]


@st.cache_resource
def load_assets():
    with open(os.path.join(MODEL_DIR, "word_index.json"), encoding="utf-8") as f:
        word_index = json.load(f)
    models = {
        name: tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{name.lower()}.keras"))
        for name in MODEL_NAMES
    }
    return word_index, models


def encode_review(text, word_index):
    tokens = []
    for word in text.lower().split():
        index = word_index.get(word, 2)
        if index >= VOCAB_SIZE:
            index = 2
        tokens.append(index + 3)
    return pad_sequences([tokens], maxlen=MAX_LENGTH, padding="post", truncating="post")


st.title("💬 SentimentLab")
st.write("Compare four deep-learning models on the same movie review.")
st.caption("CNN · Simple RNN · LSTM · GRU")

if not os.path.exists(os.path.join(MODEL_DIR, "word_index.json")):
    st.warning("Models are not trained yet. In the project folder, run `python train.py` first.")
    st.stop()

word_index, models = load_assets()

with st.sidebar:
    st.header("About the project")
    st.write("Dataset: IMDb movie reviews")
    st.write("Task: Positive vs. negative sentiment")
    st.write("Models: CNN, RNN, LSTM and GRU")
    st.write("Input limit: 250 tokens")

review = st.text_area(
    "Enter a movie review",
    value="The story was engaging, the acting was excellent, and I really enjoyed the film.",
    height=150,
    placeholder="Type or paste a review here..."
)
selected_models = st.multiselect("Models to compare", MODEL_NAMES, default=MODEL_NAMES)

if st.button("Analyze review", type="primary", disabled=not review.strip() or not selected_models):
    encoded = encode_review(review, word_index)
    rows = []
    cols = st.columns(min(len(selected_models), 4))

    for i, name in enumerate(selected_models):
        probability = float(models[name].predict(encoded, verbose=0)[0][0])
        sentiment = "Positive" if probability >= 0.5 else "Negative"
        confidence = probability if probability >= 0.5 else 1 - probability
        rows.append({
            "Model": name,
            "Prediction": sentiment,
            "Positive probability": round(probability, 3),
            "Confidence": round(confidence * 100, 1)
        })
        with cols[i % len(cols)]:
            st.metric(name, sentiment, f"{confidence * 100:.1f}% confidence")

    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.bar_chart(pd.DataFrame(rows).set_index("Model")[["Positive probability"]], y="Positive probability")
    st.caption("A probability closer to 1 indicates positive sentiment; closer to 0 indicates negative sentiment.")

results_path = os.path.join(MODEL_DIR, "results.json")
if os.path.exists(results_path):
    with open(results_path, encoding="utf-8") as f:
        results = pd.DataFrame(json.load(f))
    st.divider()
    st.subheader("Test-set results")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.bar_chart(results.set_index("model")[["test_accuracy"]], y="test_accuracy")
