import streamlit as st
import google.generativeai as genai
from apikey import api_key
import requests
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK dependencies lazily
def setup_nltk():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_nltk()

# Configure Gemini API
genai.configure(api_key=api_key)

# Precompile bias words for faster regex matching
BIAS_WORDS = ["shocking", "disaster", "corrupt", "manipulative", "scandal", "biased", "fake", "propaganda"]
BIAS_PATTERN = re.compile(r"\b(" + "|".join(BIAS_WORDS) + r")\b", re.IGNORECASE)

def highlight_bias(text):
    return BIAS_PATTERN.sub(lambda match: f":red[{match.group(0)}]", text)

def analyze_news(news_text):
    prompt = f"""
    Analyze the following news content:
    {news_text}
    
    Provide:
    - A credibility score (0-100)
    - Fact-checking summary
    - AI-generated likelihood (0-100%)
    - Any bias indicators
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching analysis: {str(e)}"

def detect_bias(text):
    score = sia.polarity_scores(text)
    if score["compound"] < -0.3:
        return "⚠️ Potential Negative Bias Detected!"
    elif score["compound"] > 0.3:
        return "⚠️ Potential Positive Bias Detected!"
    return "✅ No strong bias detected."

def plot_ai_detection(probability):
    labels = ['AI-Generated', 'Human-Written']
    sizes = [probability, 100 - probability]
    colors = ['#FF9999', '#66B2FF']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

def plot_fake_content_level(score):
    fig, ax = plt.subplots()
    ax.barh(['Fake Content Level'], [score], color=['#FF5733'])
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage Level')
    ax.set_title('Fake News Probability')
    st.pyplot(fig)

# Streamlit UI
st.title("📰 Fake News Checker")

# File Upload Option
uploaded_file = st.file_uploader("Upload a news article (PDF/TXT)", type=["txt", "pdf"])
if uploaded_file is not None:
    news_text = uploaded_file.read().decode("utf-8")
else:
    news_text = st.text_area("Paste the news content here:", height=200)

if st.button("Check News"):
    if news_text.strip():
        with st.spinner("Analyzing... Please wait."):
            analysis_result = analyze_news(news_text)
            st.subheader("🔎 Analysis Results:")
            st.write(analysis_result)

            # Extract AI probability if available
            ai_probability_match = re.search(r'AI-generated likelihood:\s*(\d{1,3})%', analysis_result)
            if ai_probability_match:
                ai_probability = int(ai_probability_match.group(1))
                st.subheader("🤖 AI Content Analysis")
                plot_ai_detection(ai_probability)

            # Extract Fake Content Score if available
            fake_content_match = re.search(r'credibility score:\s*(\d{1,3})', analysis_result)
            if fake_content_match:
                fake_content_score = 100 - int(fake_content_match.group(1))
                st.subheader("📊 Fake Content Level")
                plot_fake_content_level(fake_content_score)

            # Bias Highlighting
            st.subheader("⚠️ Bias Detection")
            st.markdown(highlight_bias(news_text))

            # Sentiment-Based Bias Detection
            bias_result = detect_bias(news_text)
            st.write(bias_result)
    else:
        st.warning("⚠️ Please enter some text to analyze.")
