import streamlit as st
from fk_nws_det_withDB import analyze_news

# Set up the app
st.set_page_config(
    page_title="FAKE NEWS DETECTOR",
    page_icon="🕵️",
    layout="centered"
)

# Custom CSS for better styling (simplified)
st.markdown("""
<style>
    .header {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 24px;
        text-align: center;
    }
    .fake {
        background-color: #ffcccc;
        color: #ff0000;
    }
    .real {
        background-color: #ccffcc;
        color: #009900;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="header">🕵️ Fake News Detector</div>', unsafe_allow_html=True)

# Text input area
user_input = st.text_area(
    "Enter a news headline or short article:",
    "Aliens spotted on earth",
    height=150
)

# Analysis button
if st.button("Analyze News", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner('Analyzing...'):
            result = analyze_news(user_input)
        
        # Display results
        if result["prediction"] == "FAKE":
            st.markdown(f'<div class="result-box fake">✗ FAKE News</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box real">✓ REAL News</div>', unsafe_allow_html=True)
        
        st.markdown(f'**Credibility Score:** {result["score"]}%')
    else:
        st.warning("Please enter some text to analyze")
