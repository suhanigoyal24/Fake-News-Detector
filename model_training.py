# model_training.py

import pandas as pd
import string
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

MODEL_FILE = "model_state.pkl"

# --------------------------------------------------------------------------
# Preprocess Text
# --------------------------------------------------------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return " ".join(tokens)

# --------------------------------------------------------------------------
# Map domain to human-readable source name
# --------------------------------------------------------------------------
SOURCE_MAPPING = {
    "www.bhaskar.com": "Dainik Bhaskar",
    "bhaskar.com": "Dainik Bhaskar",
    "www.ndtv.com": "NDTV",
    "ndtv.com": "NDTV",
    "www.bbc.com": "BBC",
    "bbc.com": "BBC",
    "www.hindustantimes.com": "Hindustan Times",
    "hindustantimes.com": "Hindustan Times",
    "www.indiatoday.in": "India Today",
    "indiatoday.in": "India Today",
    "www.thequint.com": "The Quint",
    "thequint.com": "The Quint",
    "www.altnews.in": "Alt News",
    "altnews.in": "Alt News",
    "www.snopes.com": "Snopes",
    "snopes.com": "Snopes",
    # Add more sources as needed
}

def get_source_name(url):
    if isinstance(url, str) and url.strip():
        try:
            domain = url.split("/")[2]
            return SOURCE_MAPPING.get(domain, domain)
        except IndexError:
            return "Unknown Source"
    return "Unknown Source"

# --------------------------------------------------------------------------
# Load Dataset
# --------------------------------------------------------------------------
def load_dataset():
    df = pd.read_csv("news_dataset.csv", encoding="utf-8")

    # Remove duplicate columns and reset index
    df = df.loc[:, ~df.columns.duplicated()]
    df.reset_index(drop=True, inplace=True)

    # Drop rows with missing essential info
    df = df.dropna(subset=["text", "label", "article_url"])

    # Ensure cleaned text exists
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    # Extract human-readable source name
    df["source"] = df["article_url"].apply(get_source_name)

    df.reset_index(drop=True, inplace=True)
    return df

# --------------------------------------------------------------------------
# Train Model
# --------------------------------------------------------------------------
def train_model():
    df = load_dataset()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["cleaned_text"])
    y = df["label"]

    model = SVC(probability=True)  # SVM with probability estimates
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump((df, vectorizer, model), f)

    print("Model trained and saved successfully.")
    return df, vectorizer, model

# --------------------------------------------------------------------------
# Load or Retrain Automatically
# --------------------------------------------------------------------------
def load_or_train():
    if not os.path.exists(MODEL_FILE):
        return train_model()

    if os.path.exists("news_dataset.csv"):
        csv_time = os.path.getmtime("news_dataset.csv")
        model_time = os.path.getmtime(MODEL_FILE)

        if csv_time > model_time:
            print("Dataset updated — retraining model...")
            return train_model()

    with open(MODEL_FILE, "rb") as f:
        df, vectorizer, model = pickle.load(f)

    return df, vectorizer, model

# --------------------------------------------------------------------------
# Predict Function
# --------------------------------------------------------------------------
def predict_news(text, confidence_threshold=80):
    cleaned = preprocess_text(text)
    vector_input = vectorizer.transform([cleaned])

    pred_label = model.predict(vector_input)[0]
    confidence = float(model.predict_proba(vector_input).max()) * 100

    # Filter dataset for matching label
    filtered_df = df[df["label"] == pred_label].copy()
    filtered_df = filtered_df[
        filtered_df["article_url"].notna() & (filtered_df["article_url"].str.strip() != "")
    ]

    if filtered_df.empty:
        matched_headline = "No similar article found"
        article_url = "No URL available"
        source_name = "Unknown Source"
    else:
        similarities = cosine_similarity(
            vector_input, vectorizer.transform(filtered_df["cleaned_text"])
        )
        idx = similarities.argmax()
        closest_row = filtered_df.iloc[idx]

        raw_url = closest_row.get("article_url", "")
        article_url = raw_url.strip() if isinstance(raw_url, str) and raw_url.strip() else "No URL available"
        source_name = get_source_name(article_url)

        matched_headline = closest_row["text"]

    prediction_text = (
        "Possibly Fake / Needs Verification" if confidence < confidence_threshold else pred_label.upper()
    )

    return {
        "prediction": prediction_text,
        "confidence": round(confidence, 2),
        "source": source_name,
        "confidence_source": f"{round(confidence, 2)}% credibility estimate",
        "matched_headline": matched_headline,
        "article_url": article_url
    }

# --------------------------------------------------------------------------
# Load model at import
# --------------------------------------------------------------------------
df, vectorizer, model = load_or_train()

if __name__ == "__main__":
    print("Model ready and dataset loaded.")
