import streamlit as st
import pickle
import requests
import numpy as np
from model_training import preprocess_text
from constant_fakes import IMPOSSIBLE_STATEMENTS
from urllib.parse import urlparse
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

# --- SQLite Review System ---
conn = sqlite3.connect("reviews.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    review TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def add_review(name, review):
    cursor.execute("INSERT INTO reviews (name, review) VALUES (?, ?)", (name, review))
    conn.commit()

def get_reviews():
    cursor.execute("SELECT name, review, created_at FROM reviews ORDER BY created_at DESC")
    return cursor.fetchall()


# ---------------- CONFIG ----------------
FACTCHECK_API_KEY = "AIzaSyCMClQLBsetyl9pr8E5JOibbbgr0orGHyY"
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

st.set_page_config(page_title="Fake News Detector", layout="centered")

# ---------------- LOAD MODEL ----------------
with open("model_state.pkl", "rb") as f:
    df, vectorizer, model = pickle.load(f)

# Ensure cleaned text column exists
if "cleaned_text" not in df.columns:
    df["cleaned_text"] = df["text"].apply(preprocess_text)

# ---------------- SESSION STATE ----------------
if "fact_index" not in st.session_state:
    st.session_state.fact_index = 0

# ---------------- STYLING ----------------
st.markdown("""
<style>
.title { font-size: 40px; font-weight: 800; color: #2A6FDB; text-align: center; }
.subtitle { text-align:center; color:#444; margin-bottom:20px; }
textarea { border-radius:12px !important; font-size:17px !important; }
.result-box { padding:18px; font-size:22px; border-radius:10px; text-align:center; font-weight:700; }
.real { background:#065F46; color:#D1FAE5; border-left:6px solid #34D399; }
.fake { background:#7F1D1D; color:#FECACA; border-left:6px solid #F87171; }
.card { background:white; padding:15px; border-radius:12px; box-shadow:0 4px 6px rgba(0,0,0,0.1); margin-bottom:15px; }
.card-title { font-weight:700; font-size:16px; margin-bottom:8px; color:#2A6FDB; }
.card-text { font-size:14px; color:#555; }
.info { font-size:15px; margin-top:8px; }
.url-link { color:#2A6FDB; text-decoration:none; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Just paste the snippet and click 'Check News' to check whether the news is Real or Fake</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR: ABOUT ----------------
st.sidebar.markdown(
    "<div class='card'><div class='card-title'>About This App</div>"
    "<div class='card-text'>This application fetches real-time news from multiple sources and helps users check if it is real or fake.</div></div>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR: ROTATING FACTS ----------------
facts = [
    "Over 50% of adults read news online.",
    "Fake news spreads faster than real news.",
    "Headlines often exaggerate to get clicks.",
    "Always verify news from multiple sources."
]
current_fact = facts[st.session_state.fact_index]
st.sidebar.markdown(
    f"<div class='card'><div class='card-title'>Did You Know?</div>"
    f"<div class='card-text'>{current_fact}</div></div>",
    unsafe_allow_html=True
)
st.session_state.fact_index = (st.session_state.fact_index + 1) % len(facts)

# ---------------- SIDEBAR: REVIEW ----------------
st.sidebar.markdown("<div class='card'><div class='card-title'>Leave a Review</div>", unsafe_allow_html=True)
review_name = st.sidebar.text_input("Your Name", value="Anonymous")
review_text = st.sidebar.text_area("Write your review here...", height=80)
if st.sidebar.button("Submit Review"):
    if review_text.strip():
        add_review(review_name.strip() or "Anonymous", review_text.strip())
        st.sidebar.success("Review submitted successfully!")
    else:
        st.sidebar.warning("Please write something before submitting.")


# ---------------- MAIN INPUT ----------------
news_text = st.text_area("", height=140, placeholder="Enter news snippet...")

# ---------------- FACT-CHECK API ----------------
def fact_check_api(query):
    params = {"query": query, "key": FACTCHECK_API_KEY}
    try:
        resp = requests.get(FACTCHECK_URL, params=params)
        data = resp.json()
        if "claims" not in data or len(data["claims"]) == 0:
            return None
        claim = data["claims"][0]
        review = claim["claimReview"][0]
        return {
            "text": claim.get("text", ""),
            "rating": review.get("textualRating", "").lower(),
            "url": review.get("url", "")
        }
    except Exception as e:
        st.error(f"Fact-check API error: {e}")
        return None

# ---------------- URL → Source Name Mapping ----------------
def get_source_name(article_url):
    if not article_url or article_url.strip().lower() in ["none", ""]:
        return "Unknown Source"

    try:
        parsed = urlparse(article_url)
        domain = parsed.netloc.lower()

        # Remove common subdomains
        for sub in ["www.", "m.", "news.", "en."]:
            domain = domain.replace(sub, "")

        # Take main part before first dot
        name_part = domain.split(".")[0]

        # Replace dashes or underscores with spaces and capitalize words
        site_name = " ".join(word.capitalize() for word in name_part.replace("-", " ").replace("_", " ").split())
        return site_name if site_name else "Unknown Source"
    except:
        return "Unknown Source"

# ---------------- IMPOSSIBLE CHECK ----------------
def is_impossible(text):
    for stmt in IMPOSSIBLE_STATEMENTS:
        if stmt.lower() in text.lower():
            return True, stmt
    return False, ""

# ---------------- DB MATCH ----------------
def find_closest_match(cleaned_text, df, vectorizer):
    vectors = vectorizer.transform(df["cleaned_text"])
    input_vec = vectorizer.transform([cleaned_text])
    similarity = cosine_similarity(input_vec, vectors)[0]
    idx = similarity.argmax()

    return {
        "text": df.iloc[idx]["text"],
        "label": df.iloc[idx]["label"],
        "source_site": df.iloc[idx]["source_site"],
        "article_url": df.iloc[idx]["article_url"],
        "similarity": float(similarity[idx] * 100)
    }

# ---------------- MAIN CHECK BUTTON ----------------
if st.button("Check News", key="check_news_btn"):

    if not news_text.strip():
        st.warning("Please enter a news snippet.")
        st.stop()

    # 1️⃣ Impossible check
    impossible, matched_phrase = is_impossible(news_text)
    if impossible:
        st.markdown("<div class='result-box fake'>FAKE News</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info'><b>Reason:</b> Contains impossible claim → '{matched_phrase}'</div>", unsafe_allow_html=True)
        st.stop()

    # 2️⃣ Preprocess text
    cleaned = preprocess_text(news_text)

    # 3️⃣ Fact-check API
    fact_data = fact_check_api(news_text)
    if fact_data:
        rating = fact_data["rating"]
        article_url = fact_data["url"]
        source_name = get_source_name(article_url)

        fake_terms = ["false", "fake", "incorrect", "misleading", "pants on fire"]

        if any(w in rating.lower() for w in fake_terms):
            st.markdown("<div class='result-box fake'>FAKE News</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box real'>REAL News</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='info'><b>Source:</b> {source_name}</div>", unsafe_allow_html=True)
        if article_url and article_url.strip():
            st.markdown(f"<div class='info'><b>URL:</b> <a class='url-link' href='{article_url}' target='_blank'>Click Here</a></div>", unsafe_allow_html=True)
        st.stop()

    # 4️⃣ TF-IDF + ML fallback
    input_vec = vectorizer.transform([cleaned])
    pred = model.predict(input_vec)[0]
    prob = model.predict_proba(input_vec)[0]
    confidence = max(prob) * 100

    # 5️⃣ Closest DB match
    closest_match = find_closest_match(cleaned, df, vectorizer)
    closest_article_url = closest_match.get("article_url", "")
    # Use get_source_name on the URL if source_site is empty or NaN
    closest_source_name = closest_match.get("source_site")
    if not closest_source_name or str(closest_source_name).strip().lower() in ["nan", ""]:
        closest_source_name = get_source_name(closest_article_url)


    # Display results
    if str(pred).lower() == "real":
        st.markdown("<div class='result-box real'>REAL News</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box fake'>FAKE News</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='info'><b>Confidence:</b> {confidence:.2f}%</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='info'><b>Closest Source:</b> {closest_source_name}</div>", unsafe_allow_html=True)
    if closest_article_url and isinstance(closest_article_url, str) and closest_article_url.strip():
        st.markdown(f"<div class='info'><b>URL:</b> <a class='url-link' href='{closest_article_url}' target='_blank'>Click Here</a></div>", unsafe_allow_html=True)
