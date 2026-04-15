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
# 2. APP CONFIGURATION & PREMIUM APPLE CSS
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligence: Big Data Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded" # Forces sidebar to be open on load
)

# Manage navigation via Session State for the "Launch" button
if 'nav_menu' not in st.session_state:
    st.session_state.nav_menu = "Home"

def go_to_tool():
    st.session_state.nav_menu = "Intelligence Tool"

st.markdown("""
    <style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    
    .stApp { background-color: #F5F5F7; color: #1D1D1F; }

    /* Fix Sidebar Toggle Button */
    button[kind="headerNoContext"] {
        background-color: #0071E3 !important;
        color: white !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 10px rgba(0,113,227,0.3) !important;
    }

    /* Hide Default Headers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Apple Pill Button */
    div.stButton > button {
        background-color: #0071E3 !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 980px !important; 
        border: none !important;
        padding: 14px 32px !important;
        box-shadow: 0 4px 14px rgba(0, 113, 227, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    div.stButton > button:hover {
        background-color: #0077ED !important;
        transform: scale(1.03) !important;
        box-shadow: 0 8px 24px rgba(0, 113, 227, 0.45) !important;
    }

    /* Apple Glass Cards (Fixed indention bug) */
    .mac-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: saturate(180%) blur(25px);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.6);
    }
    .engine-title { font-size: 12px; font-weight: 600; color: #86868B; text-transform: uppercase; margin-bottom: 8px; }
    .result-text { font-size: 30px; font-weight: 700; margin: 0; }
    .pos { color: #34C759; } 
    .neg { color: #FF3B30; } 
    .conf-text { font-size: 14px; color: #86868B; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 3. BACKEND: LOAD MODELS
# -------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    df = pd.read_csv("yelp_web.csv").dropna(subset=['text'])
    df = df[df['stars'] != 3]
    df['sentiment'] = df['stars'].apply(lambda x: 'positive' if x > 3 else 'negative')
    
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    X_vec = tfidf.fit_transform(df['text'])
    nb_model = MultinomialNB().fit(X_vec, df['sentiment'])
    
    try:
        dl_model = load_model('sentiment_attention_model.keras', custom_objects={'SimpleAttention': SimpleAttention})
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        deep_status = True
    except:
        dl_model, tokenizer, deep_status = None, None, False
        
    return df, tfidf, nb_model, dl_model, tokenizer, deep_status

df, tfidf, nb_model, dl_model, tokenizer, deep_engine_status = load_assets()

# -------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=70)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Intelligence Tool", "Project Details"], key="nav_menu")

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="background-color: #FFFFFF; padding: 20px; border-radius: 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.03); border: 1px solid #E5E5EA;">
        <p style="color: #86868B; font-size: 11px; margin-bottom: 4px; font-weight: 600; text-transform: uppercase;">Developed by</p>
        <p style="color: #1D1D1F; font-size: 17px; font-weight: 600; margin: 0;">Mohammad Hasnain</p>
        <p style="color: #0071E3; font-size: 13px; margin: 4px 0 0 0;">BS Artificial Intelligence</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
if deep_engine_status:
    st.sidebar.success(f"✅ Dual-Engine Online\n\n📊 {len(df):,} Reviews")
else:
    st.sidebar.warning("⚠️ Deep Engine Offline")

# -------------------------------------------------------------------------
# 5. PAGE: HOME
# -------------------------------------------------------------------------
if menu == "Home":
    st.title("Intelligence")
    st.markdown("<h2 style='color: #86868B; font-weight: 300; margin-top: -15px;'>Pro-level sentiment analysis.</h2>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1551434678-e076c223a692?q=80&w=2850&auto=format&fit=crop", use_column_width=True)
    
    st.markdown("""
    ### The next generation of Big Data Feedback.
    - ⚡ **Dual-Engine Architecture:** Statistical ML vs. Deep Learning.
    - ⚖️ **Balanced Dataset:** Undersampled Yelp Big Data.
    - 🧠 **Context Awareness:** Attention LSTM understands word order.
    """)
    st.write("")
    st.button("Launch Intelligence Tool", on_click=go_to_tool)

# -------------------------------------------------------------------------
# 6. PAGE: INTELLIGENCE TOOL
# -------------------------------------------------------------------------
elif menu == "Intelligence Tool":
    st.title("Feedback Analyzer")
    user_input = st.text_area("Review", height=200, placeholder="Example: The service was slow but the food was delicious!", label_visibility="hidden")
    
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            col_res1, col_res2 = st.columns([1.5, 1])
            with col_res2:
                # 1. Naive Bayes
                nb_p = nb_model.predict(tfidf.transform([user_input]))[0]
                nb_c = np.max(nb_model.predict_proba(tfidf.transform([user_input]))[0])
                nb_clr = "pos" if nb_p == "positive" else "neg"
                nb_emoji = "😊" if nb_p == "positive" else "😡"
                
                # 2. Attention Model
                if deep_engine_status:
                    seq = tokenizer.texts_to_sequences([user_input])
                    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
                    dl_p = dl_model.predict(padded)[0][0]
                    dl_s = "positive" if dl_p > 0.5 else "negative"
                    dl_c = dl_p if dl_p > 0.5 else (1 - dl_p)
                    dl_clr = "pos" if dl_s == "positive" else "neg"
                    dl_emoji = "😊" if dl_s == "positive" else "😡"

                # RENDER HTML
                html = f"""
<div class="mac-card">
<div class="engine-title">⚡ Fast Engine (Naive Bayes)</div>
<p class="result-text {nb_clr}">{nb_emoji} {nb_p.capitalize()}</p>
<p class="conf-text">Confidence: {nb_c*100:.1f}%</p>
</div>
"""
                if deep_engine_status:
                    html += f"""
<div class="mac-card">
<div class="engine-title">🧠 Deep Engine (Attention LSTM)</div>
<p class="result-text {dl_clr}">{dl_emoji} {dl_s.capitalize()}</p>
<p class="conf-text">Confidence: {dl_c*100:.1f}%</p>
</div>
"""
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter text first.")

# -------------------------------------------------------------------------
# 7. PAGE: PROJECT DETAILS
# -------------------------------------------------------------------------
elif menu == "Project Details":
    st.title("Project Architecture")
    st.markdown("### 🛠️ System Architecture (The Big Data Pipeline)")
    st.markdown("""
    1.  **Data Ingestion (Chunking):** * The raw file was **8.6 GB** (JSON).
        * Used Python Generators to stream data line-by-line.
    2.  **ETL & Preprocessing:**
        * **Extraction:** Parsed JSON to CSV.
        * **Transformation:** Removed 3-star (neutral) reviews.
        * **Balancing:** Applied **Undersampling** to create a 50/50 split.
    3.  **Machine Learning:**
        * **Vectorization:** TF-IDF.
        * **Model:** Multinomial Naive Bayes & Attention LSTM.
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Statistics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Positive Samples", f"{len(df[df['sentiment']=='positive']):,}")
    m3.metric("Negative Samples", f"{len(df[df['sentiment']=='negative']):,}")
    st.bar_chart(df['sentiment'].value_counts(), color="#0071E3")
