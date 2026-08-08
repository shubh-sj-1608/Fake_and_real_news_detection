import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load saved model and vectorizer
model = joblib.load("final_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w not in stop_words])
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
    return text

# --- UI ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Paste a news article or headline below to check if it's likely **Real** or **Fake**.")

user_input = st.text_area("Enter news text:", height=200, placeholder="Paste article text here...")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        # decision_function gives confidence-like score for LinearSVC
        confidence = model.decision_function(vectorized)[0]

        if prediction == 1:
            st.success(f"✅ This looks like **REAL** news")
        else:
            st.error(f"🚨 This looks like **FAKE** news")

        st.caption(f"Model confidence score: {confidence:.3f} (further from 0 = more confident)")

st.markdown("---")
st.caption("Minor Project — Fake News Detection using NLP & Machine Learning (Linear SVM, 98.6% test accuracy)")