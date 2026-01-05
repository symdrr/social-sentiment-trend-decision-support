import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --------------------------------------------------
# Bu dosya ne yapıyor?
# - Eğittiğim modeli (trained_lstm_model.h5) açıyorum
# - Yeni tweet datasında tahmin yapıyorum
# - Sonra aylık duygu dağılımını hem sayı hem yüzde olarak çıkarıyorum
# --------------------------------------------------

INPUT_NEW_CSV = "tweet_noemoji_cleaned.csv"
PREDICTED_CSV = "tweet_noemoji_cleaned_new_with_predictions.csv"

MODEL_PATH = "trained_lstm_model.h5"
FASTTEXT_VEC = "cc.tr.300.vec"   # 2. dosyada kullandığımız aynı embedding

label_map = {
    0: "mutluluk",
    1: "üzüntü",
    2: "öfke",
    3: "korku",
    4: "tiksinme",
    5: "nötr"
}

def load_fasttext_vectors(path: str) -> dict:
    # FastText dosyası büyük, açılması biraz sürebilir
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

if __name__ == "__main__":
    # 1) Yeni tweet datasını al
    df_new = pd.read_csv(INPUT_NEW_CSV)

    # Not: Bu dosyada tweet_cleaned_tr ve date kolonları olmalı
    df_new["tokens"] = df_new["tweet_cleaned_tr"].astype(str).apply(lambda x: x.split())

    # 2) Embedding sözlüğünü yükle
    print("FastText yükleniyor...")
    embedding_dict = load_fasttext_vectors(FASTTEXT_VEC)
    print(f"{len(embedding_dict)} kelime yüklendi.")

    # 3) Yeni tweetler için embedding çıkar
    tweet_vectors = df_new["tokens"].apply(lambda t: get_tweet_vector(t, embedding_dict))
    X_new = np.stack(tweet_vectors.values)

    # 4) Modeli yükle ve tahmin yap
    model = load_model(MODEL_PATH)

    y_pred_probs = model.predict(X_new)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    df_new["predicted_label_num"] = y_pred_labels
    df_new["predicted_emotion"] = df_new["predicted_label_num"].map(label_map)

    # 5) Tahminli dosyayı kaydet
    df_new.to_csv(PREDICTED_CSV, index=False)
    print("Tahminler kaydedildi:", PREDICTED_CSV)

    # --------------------------------------------------
    # 6) Aylık duygu dağılımı (sayı)
    # --------------------------------------------------
    df_new["date"] = pd.to_datetime(df_new["date"])
    df_new["year_month"] = df_new["date"].dt.to_period("M")

    monthly_counts = (
        df_new
        .groupby(["year_month", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )

    pivot_counts = monthly_counts.pivot(
        index="year_month",
        columns="predicted_emotion",
        values="count"
    ).fillna(0)

    # Grafik (stacked bar)
    pivot_counts.plot(kind="bar", stacked=True, figsize=(14, 7), width=0.8)
    plt.title("Aylık Bazda Toplumsal Duygu Dağılımı")
    plt.xlabel("Yıl-Ay")
    plt.ylabel("Tweet Sayısı")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 7) Aylık duygu dağılımı (yüzde)
    # --------------------------------------------------
    pivot_percent = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100

    # Ekranda daha okunur dursun diye format
    pd.set_option("display.float_format", lambda x: f"{x:5.1f}%")
    print("\nAylık duygu dağılımı (%)")
    print(pivot_percent)