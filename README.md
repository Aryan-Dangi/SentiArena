# SentimentLab — Movie Review Sentiment Analysis

SentimentLab is a deep learning project that classifies movie reviews as **positive or negative** using four neural network architectures: CNN, RNN, LSTM, and GRU.

The project includes model training, performance evaluation, and a simple Streamlit frontend for testing reviews and comparing model predictions.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy and Pandas
- Streamlit
- IMDb Movie Reviews Dataset

## Models

- **CNN:** Extracts local patterns from text.
- **RNN:** Processes text sequentially.
- **LSTM:** Learns long-term dependencies in sequences.
- **GRU:** Uses gated recurrent units to learn sequence patterns.

## Features

- Sentiment prediction for movie reviews.
- Comparison of four deep learning models.
- Confidence scores for predictions.
- Model performance evaluation on the test dataset.
- Interactive web interface.

## Installation and Setup

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd SentimentLab
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Train the models**

```bash
python train.py
```

The IMDb dataset will be downloaded automatically on the first run. Trained models and evaluation results will be saved in the `models/` directory.

**5. Launch the application**

```bash
streamlit run app.py
```

## Project Structure

```text
SentimentLab/
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── models/
```

The `models/` directory is generated during training and contains the trained models, vocabulary mapping, and evaluation results.

## Objective

The objective of this project is to understand and compare CNN, RNN, LSTM, and GRU architectures for text classification using a common dataset and evaluation process.

## Future Improvements

- Add model accuracy and training-time visualizations.
- Improve text preprocessing and vocabulary handling.
- Deploy the application online.

---

**Project:** SentimentLab  
**Domain:** Deep Learning · Natural Language Processing
