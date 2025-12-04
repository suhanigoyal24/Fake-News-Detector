# app.py

import streamlit as st
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from model_training import preprocess_text, get_source_name
from constant_fakes import IMPOSSIBLE_STATEMENTS


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
API_KEY = "AIzaSyCMClQLBsetyl9pr8E5JOibbbgr0orGHyY"
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


# ---------------------------------------------------
# Load TF-IDF + SVM Model
# ---------------------------------------------------
with open("model_state.pkl", "rb") as f:
    df, vectorizer, model = pickle.load(f)

if "cleaned_text" not in df.columns:
    df["cleaned_text"] = df["text"].apply(preprocess_text)


# ---------------------------------------------------
# Fact Checking API Function
# ---------------------------------------------------
def fact_check_api(query):
    params = {
        "query": query,
        "key": API_KEY,
    }

    try:
        response = requests.get(FACTCHECK_URL, params=params)
        data = response.json()

        if "claims" not in data:
            return None

        claim = data["claims"][0]
        text = claim.get("text", "")
        rating = claim["claimReview"][0].get("textualRating", "").lower()
        url = claim["claimReview"][0].get("url", "")

        return {
            "text": text,
            "rating": rating,
            "url": url
        }

    except Exception as e:
        return None


# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown("""
<style>
.title { font-size: 40px; font-weight: 800; color: #2A6FDB; text-align: center; }
.subtitle { text-align:center; color:#444; margin-bottom:20px; }
textarea { border-radius:12px !important; font-size:17px !important; }
.result-box { padding:18px; font-size:22px; border-radius:10px; text-align:center; font-weight:700; }
.real { background:#065F46; color:#D1FAE5; border-left:6px solid #34D399; }
.fake { background:#7F1D1D; color:#FECACA; border-left:6px solid #F87171; }
.info { font-size:17px; margin-top:10px; }
.url-link { color:#2563EB; }
.stButton button { border-radius:10px; border:2px solid #ff9999; color:#ff4d4d; background:white; font-size:17px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Verify whether the news snippet is real or fake</div>", unsafe_allow_html=True)


# ---------------------------------------------------
# Input
# ---------------------------------------------------
news_text = st.text_area(
    "",
    height=140,
    placeholder="Enter news snippet..."
)


# ---------------------------------------------------
# Prediction Logic
# ---------------------------------------------------
if st.button("Check News"):
    if not news_text.strip():
        st.warning("Please enter a news snippet.")
        st.stop()

    text_lower = news_text.lower()

    # Impossible Statement Check
    if any(imp in text_lower for imp in IMPOSSIBLE_STATEMENTS):
        st.markdown("<div class='result-box fake'>✗ FAKE News</div>", unsafe_allow_html=True)
        st.markdown("<div class='info'><b>Reason:</b> Contains impossible claim.</div>", unsafe_allow_html=True)
        st.stop()

    # Fact Check API Check
    fact_data = fact_check_api(news_text)

    if fact_data:
        rating = fact_data["rating"]
        fact_url = fact_data["url"]

        if any(x in rating for x in ["false", "fake", "incorrect", "pants on fire", "misleading"]):
            st.markdown("<div class='result-box fake'>✗ FAKE News</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box real'>✓ REAL News</div>", unsafe_allow_html=True)

        if fact_url:
            st.markdown(f"<div class='info'><b>Reference:</b> <a href='{fact_url}' class='url-link' target='_blank'>Click Here</a></div>", unsafe_allow_html=True)

        st.stop()

    # TF-IDF + SVM Fallback
    cleaned = preprocess_text(news_text)
    vector_input = vectorizer.transform([cleaned])

    # Prediction
    pred = model.predict(vector_input)[0]
    probs = model.predict_proba(vector_input)[0]
    confidence = max(probs) * 100

    # Similar source
    similarities = cosine_similarity(vector_input, vectorizer.transform(df["cleaned_text"])).flatten()
    idx = similarities.argmax()
    closest = df.iloc[idx]

    article_url = closest["article_url"] if "article_url" in closest else "No URL"
    source_name = get_source_name(article_url)

    # Display Result
    if str(pred).lower() == "real":
        st.markdown("<div class='result-box real'>REAL News</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box fake'>FAKE News</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='info'><b>Confidence:</b> {confidence:.2f}%</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='info'><b>Source:</b> {source_name}</div>", unsafe_allow_html=True)

    if article_url and article_url != "No URL":
        st.markdown(f"<div class='info'><b>Article URL:</b> <a class='url-link' href='{article_url}' target='_blank'>Click Here</a></div>", unsafe_allow_html=True)
