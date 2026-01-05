import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
import optuna

# --------------------------------------------------
# Bu dosya 2 iş yapıyor:
# 1) Temizlenmiş tweetleri label'layıp train datası oluşturuyor
# 2) FastText (cc.tr.300.vec) ile embedding çıkarıp LSTM eğitiyor
# --------------------------------------------------

INPUT_CSV = "tweet_noemoji_cleaned.csv"     # 1. dosyanın çıktısı
TRAINING_CSV = "tweets_training.csv"        # label'lı eğitim datası
FASTTEXT_VEC = "cc.tr.300.vec"              # FastText embedding dosyası
MODEL_OUT = "trained_lstm_model.h5"


# Label map (duygular sabit)
label_map = {
    "mutluluk": 0,
    "üzüntü": 1,
    "öfke": 2,
    "korku": 3,
    "tiksinme": 4,
    "nötr": 5
}


def load_fasttext_vectors(path: str) -> dict:
    # FastText dosyası büyük, ilk açılış biraz uzun sürebilir
    embedding_dict = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        next(f)  # header
        for line in f:
            values = line.rstrip().split(" ")
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embedding_dict[word] = vector
    return embedding_dict


def get_tweet_vector(tokens, embedding_dict, dim=300):
    vectors = [embedding_dict[w] for w in tokens if w in embedding_dict]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


def build_model(num_classes: int, lstm_units_1=128, lstm_units_2=64, dropout_1=0.3, dropout_2=0.3, dense_units=64, lr=0.0005):
    model = Sequential()
    # 300 boyutlu embedding'i LSTM'e verebilmek için (1, 300) şekline sokuyoruz
    model.add(Reshape((1, 300), input_shape=(300,)))

    model.add(LSTM(lstm_units_1, activation="tanh", return_sequences=True))
    model.add(Dropout(dropout_1))

    model.add(LSTM(lstm_units_2, activation="tanh", return_sequences=False))
    model.add(Dropout(dropout_2))

    model.add(Dense(dense_units, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    optimizer = Adam(learning_rate=lr)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def run_optuna(X_train, y_train, X_val, y_val, num_classes):
    # Optuna: hızlı deneme için epoch düşük tuttum (10)
    def objective(trial):
        lstm_units = trial.suggest_int("lstm_units", 64, 256)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        dense_units = trial.suggest_int("dense_units", 32, 128)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = Sequential()
        model.add(Reshape((1, 300), input_shape=(300,)))
        model.add(LSTM(lstm_units, activation="tanh"))
        model.add(Dropout(dropout))
        model.add(Dense(dense_units, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=["accuracy"]
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=batch_size,
            verbose=0
        )

        return history.history["val_accuracy"][-1]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params


if __name__ == "__main__":
    # 1) Preprocess çıktı dosyasını al
    df = pd.read_csv(INPUT_CSV)

    # 2) Sadece beklenen duyguları tut, label map uygula
    df = df[df["emotion"].isin(label_map.keys())].copy()
    df["label_encoded"] = df["emotion"].map(label_map).astype(int)

    df.to_csv(TRAINING_CSV, index=False)
    print("Label mapping uygulandı:", label_map)
    print("Kalan satır sayısı:", len(df))

    # 3) Tokenize (çok basic split)
    df["tokens"] = df["tweet_cleaned_tr"].astype(str).apply(lambda x: x.split())

    # 4) FastText vector dosyasını yükle
    print("FastText embedding dosyası yükleniyor...")
    embedding_dict = load_fasttext_vectors(FASTTEXT_VEC)
    print(f"{len(embedding_dict)} kelime yüklendi.")

    # 5) Tweet embedding matrisini çıkar
    tweet_vectors = df["tokens"].apply(lambda t: get_tweet_vector(t, embedding_dict))
    X = np.stack(tweet_vectors.values)
    y = df["label_encoded"].values
    print("Tweet embedding matrisi boyutu:", X.shape)

    # 6) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7) Basit model eğitimi
    model = build_model(num_classes=6)
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test doğruluğu (baseline): {acc:.4f}")

    # 8) (Opsiyonel) Optuna ile parametre denemesi
    # Not: İstersen bunu açarsın, yoksa baseline yeter.
    RUN_OPTUNA = False
    if RUN_OPTUNA:
        best = run_optuna(X_train, y_train, X_test, y_test, num_classes=6)
        print("Optuna best params:", best)

        # best paramlarla tekrar train
        tuned_model = Sequential()
        tuned_model.add(Reshape((1, 300), input_shape=(300,)))
        tuned_model.add(LSTM(best["lstm_units"], activation="tanh"))
        tuned_model.add(Dropout(best["dropout"]))
        tuned_model.add(Dense(best["dense_units"], activation="relu"))
        tuned_model.add(Dense(6, activation="softmax"))

        tuned_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=best["lr"]),
            metrics=["accuracy"]
        )

        tuned_model.fit(
            X_train, y_train,
            epochs=40,
            batch_size=best["batch_size"],
            validation_data=(X_test, y_test)
        )

        loss, acc = tuned_model.evaluate(X_test, y_test, verbose=0)
        print(f"Test doğruluğu (tuned): {acc:.4f}")

        tuned_model.save(MODEL_OUT)
        print("Model saved:", MODEL_OUT)
    else:
        # Baseline modeli kaydediyoruz
        model.save(MODEL_OUT)
        print("Model saved:", MODEL_OUT)

