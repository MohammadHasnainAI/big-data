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
# 1. CUSTOM ATTENTION LAYER
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
# 2. APP CONFIG & APPLE PREMIUM CSS
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligence: Big Data Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'nav_menu' not in st.session_state:
    st.session_state.nav_menu = "Home"

def go_to_tool():
    st.session_state.nav_menu = "Intelligence Tool"

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    .stApp { background-color: #F5F5F7; }

    /* Fix Sidebar Toggle Arrow */
    button[kind="headerNoContext"] {
        background-color: #0071E3 !important;
        color: white !important;
        border-radius: 12px !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Apple Pill Button */
    div.stButton > button {
        background-color: #0071E3 !important;
        color: white !important;
        border-radius: 980px !important;
        padding: 12px 30px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 20px rgba(0,113,227,0.3);
    }

    /* Apple macOS Cards */
    .mac-card {
        background: white;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid #E5E5EA;
        position: relative;
    }
    
    /* Highlight the Winner */
    .winner-card {
        border: 2px solid #0071E3 !important;
        box-shadow: 0 10px 30px rgba(0,113,227,0.1) !important;
    }
    
    .winner-badge {
        position: absolute;
        top: -12px;
        right: 20px;
        background: #0071E3;
        color: white;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: bold;
    }

    .pos { color: #34C759; font-weight: bold; font-size: 28px; }
    .neg { color: #FF3B30; font-weight: bold; font-size: 28px; }
    .engine-tag { font-size: 11px; color: #86868B; text-transform: uppercase; letter-spacing: 0.5px; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 3. BACKEND LOAD
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
        status = True
    except:
        dl_model, tokenizer, status = None, None, False
    return df, tfidf, nb_model, dl_model, tokenizer, status

df, tfidf, nb_model, dl_model, tokenizer, deep_engine_status = load_assets()

# -------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=70)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Intelligence Tool", "Project Details"], key="nav_menu")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background: white; padding: 20px; border-radius: 16px; border: 1px solid #E5E5EA;">
    <p style="margin:0; font-size: 10px; color: #86868B; font-weight: 600;">STUDENT AI PROJECT</p>
    <p style="margin:0; font-weight: bold; font-size: 18px; color: #1D1D1F;">Mohammad Hasnain</p>
    <p style="margin:0; font-size: 13px; color: #0071E3;">BS Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"✅ System Live | {len(df):,} reviews")

# -------------------------------------------------------------------------
# 5. PAGES
# -------------------------------------------------------------------------
if menu == "Home":
    st.title("Intelligence")
    st.markdown("<p style='color: #86868B; font-size: 20px;'>Pro-level sentiment analysis.</p>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1551434678-e076c223a692?q=80&w=2850&auto=format&fit=crop", use_column_width=True)
    st.markdown("### The next generation of Big Data Feedback.")
    st.button("Launch Intelligence Tool", on_click=go_to_tool)

elif menu == "Intelligence Tool":
    st.title("Feedback Analyzer")
    user_input = st.text_area("Review", height=150, placeholder="Paste customer feedback here...", label_visibility="hidden")
    
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            # Engine 1: Naive Bayes
            nb_p = nb_model.predict(tfidf.transform([user_input]))[0]
            nb_c = np.max(nb_model.predict_proba(tfidf.transform([user_input]))[0])
            
            # Engine 2: Deep Learning
            dl_c = 0 # Default if model fails
            if deep_engine_status:
                seq = tokenizer.texts_to_sequences([user_input])
                padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
                dl_prob = dl_model.predict(padded, verbose=0)[0][0]
                dl_sent = "POSITIVE" if dl_prob > 0.5 else "NEGATIVE"
                dl_c = dl_prob if dl_prob > 0.5 else (1 - dl_prob)

            # Determine Winner
            nb_is_winner = nb_c >= dl_c
            
            # --- RENDER HTML (Fixed indention bug) ---
            html = f"""
<div class="mac-card {'winner-card' if nb_is_winner else ''}">
    {'<div class="winner-badge">MOST TRUSTED</div>' if nb_is_winner else ''}
    <p class="engine-tag">⚡ FAST ENGINE (NAIVE BAYES)</p>
    <p class="{'pos' if nb_p=='positive' else 'neg'}">{nb_p.upper()}</p>
    <p style="font-size:14px; color:#86868B;">Confidence: {nb_c*100:.1f}%</p>
</div>
"""
            if deep_engine_status:
                html += f"""
<div class="mac-card {'winner-card' if not nb_is_winner else ''}">
    {'<div class="winner-badge">MOST TRUSTED</div>' if not nb_is_winner else ''}
    <p class="engine-tag">🧠 DEEP ENGINE (ATTENTION)</p>
    <p class="{'pos' if dl_sent=='POSITIVE' else 'neg'}">{dl_sent}</p>
    <p style="font-size:14px; color:#86868B;">Confidence: {dl_c*100:.1f}%</p>
</div>
"""
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.warning("Please enter text first.")

elif menu == "Project Details":
    st.title("Architecture")
    st.markdown("### 🛠️ System Architecture (The Big Data Pipeline)")
    st.markdown("""
    1. **Data Ingestion:** Used Python Generators to stream **8.6GB JSON** data.
    2. **ETL:** Extracted values to CSV, cleaned neutral reviews.
    3. **Balancing:** Undersampled for a perfect 50/50 split.
    4. **Dual-Model Inference:** Compares statistical probability with contextual neural attention.
    """)
    st.bar_chart(df['sentiment'].value_counts(), color="#0071E3")
