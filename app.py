import streamlit as st
import google.generativeai as genai  # ✅ Ensure the library is installed
import requests
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from PyPDF2 import PdfReader
from apikey import api_key  # ✅ Ensure apikey.py exists and contains `api_key = "your-api-key"`

# Ensure NLTK resources are available
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# 🔥 Ensure API key is set correctly
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"❌ Error setting up Google Gemini API: {str(e)}")

# Precompile Bias Words
BIAS_WORDS = ["shocking", "disaster", "corrupt", "manipulative", "scandal", "biased", "fake", "propaganda"]
BIAS_PATTERN = re.compile(r"\b(" + "|".join(BIAS_WORDS) + r")\b", re.IGNORECASE)

# Bias Highlighter
def highlight_bias(text):
    return BIAS_PATTERN.sub(lambda match: f":red[{match.group(0)}]", text)

# Gemini AI Analysis
def analyze_news(news_text):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(news_text)
        return response.text if response else "No response from AI."
    except Exception as e:
        return f"❌ Error fetching analysis: {str(e)}"

# Sentiment Bias Detector
def detect_bias(text):
    score = sia.polarity_scores(text)
    if score["compound"] < -0.3:
        return "⚠️ Potential Negative Bias Detected!"
    elif score["compound"] > 0.3:
        return "⚠️ Potential Positive Bias Detected!"
    return "✅ No strong bias detected."

# Pie Chart
def plot_ai_detection(probability):
    fig, ax = plt.subplots()
    ax.pie([probability, 100 - probability], labels=['AI-Generated', 'Human-Written'], autopct='%1.1f%%', colors=['#FF9999', '#66B2FF'], startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# Fake News Bar Chart
def plot_fake_content_level(score):
    fig, ax = plt.subplots()
    ax.barh(['Fake Content Level'], [score], color=['#FF5733'])
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage Level')
    ax.set_title('Fake News Probability')
    st.pyplot(fig)

# Streamlit UI
st.title("📰 Fake News Checker")

# File Upload Handling
uploaded_file = st.file_uploader("Upload a news article (PDF/TXT)", type=["txt", "pdf"])
if uploaded_file is not None:
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            news_text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        else:
            news_text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        news_text = ""
else:
    news_text = st.text_area("Paste the news content here:", height=200)

# Process News on Button Click
if st.button("Check News"):
    if news_text.strip():
        with st.spinner("Analyzing... Please wait."):
            analysis_result = analyze_news(news_text)
            st.subheader("🔎 Analysis Results:")
            st.write(analysis_result)

            # AI Probability Extraction
            ai_probability_match = re.search(r'AI-generated likelihood:\s*(\d{1,3})%', analysis_result)
            ai_probability = int(ai_probability_match.group(1)) if ai_probability_match else 0
            st.subheader("🤖 AI Content Analysis")
            plot_ai_detection(ai_probability)

            # Fake Content Score Extraction
            fake_content_match = re.search(r'credibility score:\s*(\d{1,3})', analysis_result)
            fake_content_score = 100 - int(fake_content_match.group(1)) if fake_content_match else 50
            st.subheader("📊 Fake Content Level")
            plot_fake_content_level(fake_content_score)

            # Bias Highlighting
            st.subheader("⚠️ Bias Detection")
            st.markdown(highlight_bias(news_text))

            # Sentiment-Based Bias Detection
            st.write(detect_bias(news_text))
    else:
        st.warning("⚠️ Please enter some text to analyze.")

