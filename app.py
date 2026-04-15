import streamlit as st
import pandas as pd
import time
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------------------------------------------------
# 1. CUSTOM ATTENTION LAYER (Required for Deep Learning Model)
# -------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros')
        super(SimpleAttention, self).build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# -------------------------------------------------------------------------
# 2. APP CONFIGURATION & APPLE-STYLE CSS
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligence: Big Data Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup Session State for the Navigation Button
if 'nav_menu' not in st.session_state:
    st.session_state.nav_menu = "Home"

def go_to_tool():
    st.session_state.nav_menu = "Intelligence Tool"

# Apple macOS Premium Styling
st.markdown("""
    <style>
    /* Global Apple Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .stApp {
        background-color: #F5F5F7; /* Apple Light Gray Background */
        color: #1D1D1F;
    }

    /* Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Apple Pill Button */
    div.stButton > button {
        background-color: #0071E3; /* Apple Blue */
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 980px !important; /* Perfect Pill Shape */
        border: none !important;
        padding: 14px 32px !important;
        box-shadow: 0 4px 14px rgba(0, 113, 227, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    div.stButton > button:hover {
        background-color: #0077ED !important;
        transform: scale(1.03) !important; /* Smooth pop-out effect */
        box-shadow: 0 8px 24px rgba(0, 113, 227, 0.45) !important;
    }
    div.stButton > button:active {
        transform: scale(0.98) !important; /* Satisfying click effect */
    }
    
    /* Input Text Area */
    .stTextArea textarea {
        background-color: #FFFFFF;
        border-radius: 14px;
        border: 1px solid #D2D2D7;
        padding: 16px;
        font-size: 16px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.02);
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #0071E3;
        box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.1);
    }

    /* Apple Glass Cards */
    .mac-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.04);
        border: 1px solid rgba(255,255,255,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .mac-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.08);
    }
    
    .engine-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #86868B;
        margin-bottom: 8px;
    }
    
    .result-text {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    
    .pos { color: #34C759; } /* Apple Green */
    .neg { color: #FF3B30; } /* Apple Red */
    
    .conf-text {
        font-size: 14px;
        color: #86868B;
        font-weight: 400;
        margin-top: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 3. BACKEND: LOAD BOTH MODELS
# -------------------------------------------------------------------------
@st.cache_resource
def load_all_assets():
    # 1. Load Naive Bayes (Fast Engine)
    df = pd.read_csv("yelp_web.csv").dropna(subset=['text'])
    df = df[df['stars'] != 3]
    df['sentiment'] = df['stars'].apply(lambda x: 'positive' if x > 3 else 'negative')
    
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    X_vec = tfidf.fit_transform(df['text'])
    nb_model = MultinomialNB().fit(X_vec, df['sentiment'])
    
    # 2. Load Attention LSTM (Deep Engine)
    try:
        dl_model = load_model('sentiment_attention_model.keras', custom_objects={'SimpleAttention': SimpleAttention})
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        deep_engine_status = True
    except Exception as e:
        dl_model, tokenizer = None, None
        deep_engine_status = False

    return df, tfidf, nb_model, dl_model, tokenizer, deep_engine_status

df, tfidf, nb_model, dl_model, tokenizer, deep_engine_status = load_all_assets()

# -------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION (Upgraded to Apple Style)
# -------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
st.sidebar.title("Navigation")

# Connect the sidebar radio to the session state!
menu = st.sidebar.radio("Go to:", ["Home", "Intelligence Tool", "Project Details"], key="nav_menu")

st.sidebar.markdown("---")

# UPGRADED STUDENT PROFILE CARD
st.sidebar.markdown("""
    <div style="background-color: #FFFFFF; padding: 20px; border-radius: 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.03); border: 1px solid #E5E5EA;">
        <p style="color: #86868B; font-size: 11px; margin-bottom: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Developed by</p>
        <p style="color: #1D1D1F; font-size: 17px; font-weight: 600; margin: 0; letter-spacing: -0.3px;">Mohammad Hasnain</p>
        <p style="color: #0071E3; font-size: 13px; margin: 4px 0 0 0; font-weight: 500;">BS Artificial Intelligence</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# DUAL ENGINE STATUS
if deep_engine_status:
    st.sidebar.success(f"✅ Dual-Engine Online\n\n📊 {len(df):,} Reviews")
else:
    st.sidebar.warning(f"⚠️ Fast Engine Only\n\n📊 {len(df):,} Reviews")

# -------------------------------------------------------------------------
# 5. PAGE: HOME
# -------------------------------------------------------------------------
if menu == "Home":
    st.title("Intelligence")
    st.markdown("<h2 style='color: #86868B; font-weight: 300; margin-top: -15px;'>Pro-level sentiment analysis.</h2>", unsafe_allow_html=True)
    
    st.image("https://images.unsplash.com/photo-1551434678-e076c223a692?q=80&w=2850&auto=format&fit=crop", use_column_width=True)
    
    st.markdown("""
    ### The next generation of Big Data Feedback.
    This project combines traditional **Machine Learning** with modern **Deep Learning** to analyze customer sentiment with incredible precision.

    - ⚡ **Dual-Engine Architecture:** Compares statistical ML with Deep Learning.
    - ⚖️ **Balanced Dataset:** Undersampled Yelp Big Data.
    - 🧠 **Context Awareness:** LSTM understands the sequence and context of words.
    """)
    
    st.write("")
    st.button("Launch Intelligence Tool", on_click=go_to_tool)

# -------------------------------------------------------------------------
# 6. PAGE: INTELLIGENCE TOOL
# -------------------------------------------------------------------------
elif menu == "Intelligence Tool":
    st.title("Feedback Analyzer")
    st.markdown("<p style='color: #86868B; font-size: 18px;'>Type a review and let the Dual-Engine AI analyze its sentiment.</p>", unsafe_allow_html=True)
    
    st.write("")

    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        user_input = st.text_area("Review Text", height=200, placeholder="Example: The service was slow but the food was absolutely delicious!", label_visibility="hidden")
        analyze_btn = st.button("Analyze Sentiment")

    with col2:
        st.markdown("<p style='font-size: 14px; font-weight: 600; color: #86868B; margin-bottom: 10px;'>PREDICTION RESULTS</p>", unsafe_allow_html=True)
        
        results_container = st.container()
        quote_placeholder = st.empty()
        
        if analyze_btn:
            if user_input.strip():
                quotes = [
                    "**“Design is not just what it looks like and feels like. Design is how it works.”**\n— Steve Jobs",
                    "**“Innovation distinguishes between a leader and a follower.”**\n— Steve Jobs",
                    "**“Simplicity is the ultimate sophistication.”**\n— Leonardo da Vinci"
                ]
                selected_quote = random.choice(quotes)
                
                quote_placeholder.info(f"✨ **Analyzing...**\n\n{selected_quote}")
                time.sleep(2.0)
                
                # --- PREDICTIONS ---
                input_vec = tfidf.transform([user_input])
                nb_pred = nb_model.predict(input_vec)[0]
                nb_conf = np.max(nb_model.predict_proba(input_vec)[0])
                
                nb_class = "pos" if nb_pred == 'positive' else "neg"
                nb_emoji = "😊" if nb_pred == 'positive' else "😡"
                
                if deep_engine_status:
                    seq = tokenizer.texts_to_sequences([user_input])
                    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
                    dl_prob = dl_model.predict(padded)[0][0]
                    dl_sent = "positive" if dl_prob > 0.5 else "negative"
                    dl_conf = dl_prob if dl_prob > 0.5 else (1 - dl_prob)
                    
                    dl_class = "pos" if dl_sent == 'positive' else "neg"
                    dl_emoji = "😊" if dl_sent == 'positive' else "😡"

                # --- RENDER HTML ---
                html_output = f"""
<div class="mac-card">
<div class="engine-title">⚡ Fast Engine (Naive Bayes)</div>
<p class="result-text {nb_class}">{nb_emoji} {nb_pred.capitalize()}</p>
<p class="conf-text">Confidence: {nb_conf*100:.1f}%</p>
</div>
"""
                if deep_engine_status:
                    html_output += f"""
<div class="mac-card">
<div class="engine-title">🧠 Deep Engine (Attention LSTM)</div>
<p class="result-text {dl_class}">{dl_emoji} {dl_sent.capitalize()}</p>
<p class="conf-text">Confidence: {dl_conf*100:.1f}%</p>
</div>
"""
                with results_container:
                    st.markdown(html_output, unsafe_allow_html=True)
                
                quote_placeholder.empty()

            else:
                st.warning("⚠️ Please enter text first.")

# -------------------------------------------------------------------------
# 7. PAGE: PROJECT DETAILS
# -------------------------------------------------------------------------
elif menu == "Project Details":
    st.title("Project Architecture")
    st.markdown("<p style='color: #86868B; font-size: 18px;'>Deep dive into the models powering this application.</p>", unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("### 🛠️ System Architecture (The Big Data Pipeline)")
    st.write("This system was built to handle the **Volume** and **Variety** of the Yelp Open Dataset.")
    
    st.markdown("""
    1.  **Data Ingestion (Chunking):** * The raw file was **8.6 GB** (JSON).
        * Used Python Generators to stream data line-by-line to avoid Memory Overflow (RAM Crash).
    
    2.  **ETL & Preprocessing:**
        * **Extraction:** Parsed JSON to CSV.
        * **Transformation:** Removed 3-star (neutral) reviews to sharpen accuracy.
        * **Balancing:** Detected Class Imbalance (80% Positive) and applied **Undersampling** to create a perfect 50/50 split.

    3.  **Machine Learning:**
        * **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency).
        * **Model:** Multinomial Naive Bayes (Probabilistic Classifier).
    
    ---
    **Dataset Source:** [Yelp Open Dataset](https://www.yelp.com/dataset)
    """)

    st.markdown("---")
    
    st.markdown("### 🧠 How the Models Work (Old vs. New)")
    st.write("This application features a **Dual-Engine** approach, allowing us to compare traditional Machine Learning with advanced Deep Learning in real-time.")

    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("""
<div class="mac-card">
<h4 style="color:#0071E3;">⚡ The Old Model (Fast Engine)</h4>
<b>Algorithm:</b> Multinomial Naive Bayes<br>
<b>How it works:</b> This is a statistical model based on probability. It breaks the user's review down into individual words and counts them (TF-IDF vectorization). 
<br><br>
<b>Pros:</b> Incredibly fast and lightweight. Excellent for analyzing millions of rows of Big Data instantly.<br>
<b>Cons:</b> It suffers from "Bag of Words" syndrome. It does not understand word order. If a review says <i>"The food was not good,"</i> it might see the word "good" and accidentally mark it positive.
</div>
""", unsafe_allow_html=True)

    with colB:
        st.markdown("""
<div class="mac-card">
<h4 style="color:#0071E3;">🧠 The New Model (Deep Engine)</h4>
<b>Algorithm:</b> Custom Attention LSTM (Deep Learning)<br>
<b>How it works:</b> This is an artificial neural network with a "memory" (LSTM) and an "Attention Mechanism". Unlike Naive Bayes, it reads the sentence in order. 
<br><br>
<b>Pros:</b> It understands complex context. The Attention Layer acts like a human eye, assigning mathematical "weight" to the most important words in the sentence. It can easily detect sarcasm and complex sentence structures.<br>
<b>Cons:</b> Requires significantly more computing power and time to train.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Dataset Statistics")
    st.write("To ensure the AI is not biased, the dataset was strictly balanced using an Undersampling Algorithm.")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Positive Samples", f"{len(df[df['sentiment']=='positive']):,}")
    m3.metric("Negative Samples", f"{len(df[df['sentiment']=='negative']):,}")
