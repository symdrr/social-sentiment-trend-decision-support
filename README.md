# Social Sentiment Trend Analysis (Turkish NLP)

This project analyzes **social sentiment trends over time** using Turkish tweets.
It includes a complete pipeline from text preprocessing to model training and
monthly sentiment trend analysis.

The goal is to observe how collective emotions change over time based on social media data
and provide a simple **decision support perspective**.

---

## Project Structure

├── 01_preprocess_text.py
├── 02_train_lstm_fasttext.py
├── 03_predict_and_monthly_trends.py
├── tweet_noemoji_cleaned.csv
├── tweets_training.csv
├── trained_lstm_model.h5
└── README.md

---

## Pipeline Overview

### 1. Text Preprocessing (`01_preprocess_text.py`)
- Converts emojis into textual representations
- Translates common emoji meanings into Turkish
- Cleans text by removing unnecessary characters
- Outputs a cleaned dataset

**Input**
- `tweetsTrainingData2.csv`

**Output**
- `tweet_noemoji_cleaned.csv`

---

### 2. Model Training (`02_train_lstm_fasttext.py`)
- Filters valid emotion labels
- Encodes emotions numerically
- Uses **pre-trained FastText Turkish embeddings**
- Trains an **LSTM-based neural network**
- (Optional) Hyperparameter tuning with Optuna
- Saves the trained model

**Input**
- `tweet_noemoji_cleaned.csv`
- `cc.tr.300.vec` (FastText Turkish embeddings)

**Output**
- `tweets_training.csv`
- `trained_lstm_model.h5`

---

### 3. Prediction & Monthly Trend Analysis (`03_predict_and_monthly_trends.py`)
- Loads the trained LSTM model
- Predicts emotions for each tweet
- Aggregates results on a monthly basis
- Visualizes sentiment distribution over time
- Outputs both raw predictions and percentage distributions

**Input**
- `tweet_noemoji_cleaned.csv`
- `trained_lstm_model.h5`
- `cc.tr.300.vec`

**Output**
- `tweet_with_predictions.csv`
- Monthly sentiment distribution (counts & percentages)

---

## Emotion Classes

The model predicts the following emotion categories:

- Mutluluk (Happiness)
- Üzüntü (Sadness)
- Öfke (Anger)
- Korku (Fear)
- Tiksinme (Disgust)
- Nötr (Neutral)

---

## Notes

- The dataset used in this project is anonymized.
- FastText embeddings are external pre-trained resources.
- The project focuses on **end-to-end NLP workflow** rather than model optimization alone.
- Code is intentionally kept simple and readable for clarity.

---

## Technologies Used

- Python
- Pandas / NumPy
- TensorFlow (Keras)
- Scikit-learn
- FastText (Turkish embeddings)
- Matplotlib
- Optuna (optional)

---

## Purpose

This project was developed as part of my work on:
- Natural Language Processing
- Sentiment Analysis
- Time-based trend analysis
- Decision support systems

It also serves as a personal learning and portfolio project.

