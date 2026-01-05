import pandas as pd
import emoji
import re
import numpy as np

# CSV dosyasını yüklüyoruz (tweet ve emotion kolonları var)
# Dosya proje klasöründe olmalı
df = pd.read_csv("tweetsTrainingData.csv", sep=";")

# --------------------------------------------------
# EMOJI -> TEXT DÖNÜŞÜMÜ
# --------------------------------------------------
# Amaç: 😂 gibi emojileri metne çevirip modelde kullanabilmek

def emoji_to_text(text):
    text = str(text)
    # Emoji'leri metin karşılığına çevirir (😂 -> face_with_tears_of_joy)
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Tweet içindeki emojileri metne çeviriyoruz
df["tweet_cleaned"] = df["tweet"].apply(emoji_to_text)

# --------------------------------------------------
# EMOJI METİNLERİNİ TÜRKÇEYE ÇEVİRME
# --------------------------------------------------
# En sık geçen emoji anlamlarını Türkçeleştiriyoruz
# (Modelin Türkçe kelimelerle öğrenmesi için)

emoji_meanings_tr = {
    "white_heart": "beyaz kalp",
    "fallen_leaf": "düşmüş yaprak",
    "grinning_face_with_sweat": "terleyen gülen yüz",
    "crescent_moon": "hilal ay",
    "smiling_face_with_heart-eyes": "kalp gözlü gülümseme",
    "thumbs_up": "başparmak yukarı",
    "balloon": "balon",
    "broken_heart": "kırık kalp",
    "woman_facepalming": "kadın yüzüne avuç içi koyuyor",
    "cherry_blossom": "kiraz çiçeği",
    "reminder_ribbon": "hatırlatma kurdelesi",
    "blossom": "çiçek",
    "heart_suit": "kırmızı kalp",
    "wine_glass": "şarap kadehi",
    "snowflake": "kar tanesi",
    "seedling": "filiz",
    "winking_face": "göz kırpan yüz",
    "nerd_face": "inek surat",
    "partying_face": "parti yapan yüz",
    "check_mark": "tik işareti",
    "purple_heart": "mor kalp",
    "black_heart": "siyah kalp",
    "woman_fairy": "perili kadın",
    "bouquet": "çiçek buketi",
    "face_with_raised_eyebrow": "kaş kaldıran yüz",
    "cloud": "bulut",
    "new_moon_face": "yeni ay yüzü",
    "first_quarter_moon_face": "ilk dördün ay yüzü",
    "face_savoring_food": "yemeğin tadını çıkaran yüz",
    "rocket": "roket",
    "butterfly": "kelebek",
    "man_frowning_medium-light_skin_tone": "orta açık tenli kaşlarını çatan adam",
    "anatomical_heart": "anatomik kalp",
    "pleading_face": "yalvaran yüz",
    "paw_prints": "patı izi",
    "blue_heart": "mavi kalp",
    "waning_gibbous_moon": "azalan ay",
    "tulip": "lale",
    "herb": "ot",
    "black_small_square": "küçük siyah kare",
    "smiling_face_with_tear": "gözyaşlı gülümseme",
    "four_leaf_clover": "dört yapraklı yonca",
    "backhand_index_pointing_right": "sağa işaret eden el",
    "face_with_tears_of_joy": "gözyaşlarıyla gülen yüz",
    "cigarette": "sigara",
    "smiling_face_with_hearts": "kalplerle gülen yüz",
    "wilted_flower": "solmuş çiçek",
    "dove": "güvercin",
    "raised_hand": "kalkmış el",
    "neutral_face": "nötr yüz",
    "smiling_face_with_sunglasses": "gözlüklü gülen yüz",
    "see-no-evil_monkey": "gözlerini kapatan maymun",
    "upside-down_face": "ters çevrilmiş yüz",
    "folded_hands_light_skin_tone": "katlanmış eller (açık ten)",
    "musical_notes": "müzik notaları",
    "hand_with_fingers_splayed": "açık el",
    "thinking_face": "düşünen yüz",
    "winking_face_with_tongue": "dil çıkaran göz kırpan yüz",
    "woman_facepalming_medium-light_skin_tone": "orta açık tenli kadın yüzüne avuç içi koyuyor",
    "growing_heart": "büyüyen kalp",
    "yellow_heart": "sarı kalp",
    "zany_face": "şaşkın yüz",
    "waning_crescent_moon": "azalan hilal ay",
    "rose": "gül",
    "red_heart": "kırmızı kalp",
    "woman_running_light_skin_tone": "açık tenli koşan kadın",
    "face_with_hand_over_mouth": "ağzını kapatan yüz",
    "teddy_bear": "peluş ayı",
    "green_heart": "yeşil kalp",
    "loudly_crying_face": "ağlayan yüz",
    "heart_with_arrow": "okla delinmiş kalp",
    "revolving_hearts": "dönen kalpler",
    "beating_heart": "atan kalp",
    "glowing_star": "parlayan yıldız",
    "star-struck": "yıldızlarla dolu gözler",
    "smiling_face_with_halo": "hale ile gülen yüz",
    "bow_and_arrow": "yay ve ok",
    "coffin": "tabut",
    "maple_leaf": "akçaağaç yaprağı",
    "unamused_face": "memnuniyetsiz yüz",
    "woman_shrugging_medium-light_skin_tone": "orta açık tenli omuz silken kadın",
    "check_mark_button": "tik işaretli buton",
    "leaf_fluttering_in_wind": "rüzgarda sallanan yaprak",
    "heart_on_fire": "yanan kalp",
    "pensive_face": "düşünceli yüz",
    "thumbs_up_light_skin_tone": "açık tenli başparmak yukarı",
    "flushed_face": "kızarmış yüz",
    "beaming_face_with_smiling_eyes": "gözleri gülen yüz",
    "person_facepalming": "yüzüne avuç içi koyan kişi",
    "kiss_mark": "öpücük izi",
    "ring": "yüzük",
    "woman_fairy_light_skin_tone": "açık tenli peri kadın",
    "sign_of_the_horns_light_skin_tone": "açık tenli el işareti",
    "face_with_steam_from_nose": "burundan buhar çıkaran yüz",
    "clown_face": "palyaço yüzü",
    "last_quarter_moon_face": "son dördün ay yüzü",
    "sleeping_face": "uyuyan yüz",
    "raising_hands": "kollar yukarı kaldırılmış",
    "peach": "şeftali",
    "rolling_on_the_floor_laughing": "yerlerde yuvarlanarak gülen",
    "hot_beverage": "sıcak içecek",
    "zipper-mouth_face": "ağzı fermuarlı yüz",
    "writing_hand_light_skin_tone": "açık tenli yazan el",
    "double_exclamation_mark": "çift ünlem işareti",
    "sparkler": "kıvılcım",
}

def translate_emoji_text(text):
    text = str(text)
    # Emoji isimlerini Türkçe karşılıklarıyla değiştir
    for emoji_en, emoji_tr in emoji_meanings_tr.items():
        text = re.sub(rf"\b{re.escape(emoji_en)}\b", emoji_tr, text)
    return text

# Türkçeleştirilmiş emoji metinlerini uyguluyoruz
df["tweet_cleaned_tr"] = df["tweet_cleaned"].apply(translate_emoji_text)

# --------------------------------------------------
# METİN TEMİZLEME
# --------------------------------------------------
# Noktalama, özel karakter vs. kaldırılıyor
# Sadece harf, sayı ve Türkçe karakterler kalıyor

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["tweet_cleaned_tr"] = df["tweet_cleaned_tr"].apply(clean_text)

# Temizlenmiş veriyi yeni bir CSV olarak kaydediyoruz
df.to_csv("tweet_noemoji_cleaned.csv", index=False)

# Script bittiğinde konsola küçük bir bilgi basalım
if __name__ == "__main__":
    print("Preprocessing tamamlandı. Çıktı: tweet_noemoji_cleaned.csv")