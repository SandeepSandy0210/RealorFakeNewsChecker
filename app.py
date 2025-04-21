import sklearn
import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Classifier")
st.markdown("Enter a news article below to detect whether it's real or fake.")

# Input field
news_input = st.text_area("News Article Text", height=200)

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vec = vectorizer.transform([news_input])
        result = model.predict(input_vec)[0]
        label = "Real News ✅" if result == 1 else "Fake News ❌"
        st.success(f"Prediction: **{label}**")
