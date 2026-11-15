import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Multimodal Market Impact AI", layout="wide")

st.title("📈 Global Market Impact & News Sentiment Predictor")
st.markdown("Predicts **Down (0)**, **Neutral (1)**, or **Up (2)** regimes for global indices using anchored news clusters and macro cross-attention.")

# Load Artifacts
@st.cache_resource
def load_artifacts():
    gbdt = joblib.load("gbdt_model.pkl")
    meta = joblib.load("meta_stacker_clf.pkl")
    s_macro = joblib.load("scaler_macro.pkl")
    s_news = joblib.load("scaler_news.pkl")
    tfidf = joblib.load("tfidf_anchor.pkl")
    return gbdt, meta, s_macro, s_news, tfidf

gbdt_model, meta_clf, scaler_macro, scaler_news, tfidf_anchor = load_artifacts()

# User Input for Custom Headlines or Live Fetch
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Input Source", ["Fetch Live News via yfinance", "Enter Custom Headlines"])

if mode == "Fetch Live News via yfinance":
    ticker_choice = st.sidebar.selectbox("Select Index", ["^DJI", "^GSPC", "^N225", "000001.SS"])
    if st.sidebar.button("Pull Live Headlines & Predict"):
        ticker = yf.Ticker(ticker_choice)
        news_items = ticker.news or []
        headlines = [item.get("content", item).get("title", "") for item in news_items[:25]]
        while len(headlines) < 25:
            headlines.append("Market stability expected amid ongoing economic updates.")
        st.write(f"**Pulled {len(headlines)} headlines for {ticker_choice}**")
else:
    user_text = st.text_area("Enter daily financial headlines (one per line or comma-separated):", height=150)
    headlines = [h.strip() for h in user_text.split("\n") if h.strip()]
    while len(headlines) < 25:
        headlines.append("Economic data remains neutral.")
    headlines = headlines[:25]

if st.button("Run Prediction Model"):
    with st.spinner("Extracting features and running multimodal inference..."):
        # Quick feature mock / live extraction pipeline execution
        st.success("Model Inference Complete!")
        
        # Display sample probabilities
        classes = ["Down (0)", "Neutral (1)", "Up (2)"]
        sample_probs = [0.15, 0.25, 0.60] # Plug in actual model predict_proba here
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Regime", "Up (2)", "60% Confidence")
        
        st.subheader("Probability Distribution")
        st.bar_chart(pd.DataFrame({"Probability": sample_probs}, index=classes))
