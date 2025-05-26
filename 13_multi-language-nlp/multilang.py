import streamlit as st
import joblib
import pandas as pd
from langdetect import detect
from textblob import TextBlob

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('models/sentiment_optimized_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_models()

# App layout
st.title("Multilingual Sentiment Analysis")
st.write("Analyze sentiment in multiple languages")

# Input text
text_input = st.text_area("Enter your text here:", height=150)

if st.button("Analyze Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Detect language
        try:
            lang = detect(text_input)
            st.write(f"Detected language: {lang}")
        except:
            lang = 'en'
            st.write("Could not detect language, defaulting to English")
        
        # Preprocess text (simplified version)
        processed_text = preprocess_text(text_input, lang)
        
        # Vectorize
        features = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        # TextBlob sentiment for comparison
        tb_sentiment = TextBlob(text_input).sentiment.polarity
        
        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Sentiment", prediction)
            st.write(f"Confidence: {max(proba)*100:.1f}%")
        
        with col2:
            st.metric("TextBlob Sentiment", 
                      "Positive" if tb_sentiment > 0 else "Negative")
            st.write(f"Polarity: {tb_sentiment:.2f}")
        
        # Show explanation
        st.subheader("Analysis")
        st.write("""
        - **Predicted Sentiment**: From our trained machine learning model
        - **TextBlob Sentiment**: From rule-based sentiment analysis for comparison
        """)

# Sidebar with info
st.sidebar.title("About")
st.sidebar.info("""
This app analyzes sentiment in multiple languages using:
- Machine Learning (Logistic Regression)
- Text preprocessing for each language
- Language auto-detection
""")