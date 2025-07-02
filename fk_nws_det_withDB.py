# pip install spacy textblob scikit-learn pandas sqlite3
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import string
import pandas as pd
import sqlite3  # Database module

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ---------------------- Sample Data ----------------------
data = {
    'text': [
        "Donald Trump wins the US election again in 2024.",
        "Aliens have landed in India.",
        "Scientists discovered a cure for COVID-19.",
        "WHO declares monkeypox a global emergency in July 2023.",
        "Government launches new education policy for 2023.",
        "Supreme court orders restoration of 100-acre forest in Telengana.",
        "Japan gifts 2 Bullet trains worth Rs.600cr to India for free.",
        "Elon Musk announces mission to Mars by end of next year 2026.",
        "Bihar sports minister gives blanket in 40 degrees heat.",
        "Man builds time machine using microwave and magnets.",
        "Scientists confirm chocolate can cure all diseases.",
        "Elon Musk buys Taj Mahal.",
        "Aliens spotted on the earth."
    ],
    'label': ['real', 'fake', 'real', 'real', 'real', 'real', 'real', 'real', 'real', 'fake', 'fake', 'fake', 'fake']
}
df = pd.DataFrame(data)

# ---------------------- Preprocessing ----------------------
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# ---------------------- Model Training ----------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'].apply(preprocess_text))
y = df['label']
model = MultinomialNB()
model.fit(X, y)

# ---------------------- SQLite DB Setup ----------------------
def setup_database():
    conn = sqlite3.connect('news_predictions_db.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT,
            prediction TEXT,
            score REAL
        )
    ''')
    conn.commit()
    conn.close()

# Function to store predictions in the database
def store_prediction(input_text, prediction, score):
    conn = sqlite3.connect('news_predictions.db')  # Ensure consistent database name
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (input_text, prediction, score)
        VALUES (?, ?, ?)
    ''', (input_text, prediction, score))
    conn.commit()
    conn.close()

# ---------------------- Prediction Logic ----------------------
def analyze_news(text):
    cleaned_input = preprocess_text(text)
    vector_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vector_input)[0]
    probability = model.predict_proba(vector_input).max() * 100
    result = {
        "prediction": "REAL" if prediction == "real" else "FAKE",
        "score": round(probability, 2)
    }
    store_prediction(text, result["prediction"], result["score"])
    return result

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    setup_database()  # Setup database when running the script
    print("\n📰 Welcome to the Fake News Detector!")
    user_input = input("Enter a news headline or short article: ")
    result = analyze_news(user_input)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence Score: {result['score']}%")
