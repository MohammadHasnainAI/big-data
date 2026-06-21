import streamlit as st
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests
import base64
import json

# -------------------------------------------------------------------------
# GROQ API KEY — paste your key here
# -------------------------------------------------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

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
# 2. APP CONFIG & CSS
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligence AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'nav_menu' not in st.session_state:
    st.session_state.nav_menu = "Home"
if 'voice_text' not in st.session_state:
    st.session_state.voice_text = ""

def go_to_tool():
    st.session_state.nav_menu = "Intelligence Tool"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"], [data-testid="stSidebar"] * {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    .stApp { background-color: #F5F5F7; }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E5EA !important;
    }

    div.stButton > button {
        background-color: #0071E3 !important;
        color: white !important;
        border-radius: 980px !important;
        padding: 12px 30px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    .mac-card {
        background: white;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #E5E5EA;
        margin-bottom: 20px;
        position: relative;
    }

    .winner-card { border: 2px solid #0071E3 !important; }
    .winner-badge {
        position: absolute; top: -12px; right: 20px;
        background: #0071E3; color: white; padding: 2px 12px;
        border-radius: 20px; font-size: 10px; font-weight: bold;
    }

    .engine-tag { font-size: 11px; color: #86868B; text-transform: uppercase; font-weight: 600; margin-bottom: 8px; }
    .pos { color: #34C759 !important; font-weight: 700; font-size: 26px; margin: 0; }
    .neg { color: #FF3B30 !important; font-weight: 700; font-size: 26px; margin: 0; }
    .conf-val { font-size: 14px; color: #86868B; margin-top: 4px; }

    .voice-box {
        background: white;
        border-radius: 18px;
        padding: 24px;
        border: 1px solid #E5E5EA;
        margin-bottom: 20px;
        text-align: center;
    }

    .voice-title {
        font-size: 14px;
        font-weight: 600;
        color: #86868B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }

    .transcribed-box {
        background: #F5F5F7;
        border-radius: 12px;
        padding: 14px;
        margin-top: 12px;
        font-size: 15px;
        color: #1D1D1F;
        text-align: left;
        border: 1px solid #E5E5EA;
    }
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
# 4. GROQ WHISPER TRANSCRIPTION
# -------------------------------------------------------------------------
def transcribe_audio_groq(audio_bytes):
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model": "whisper-large-v3"}
        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            return None
    except Exception as e:
        return None

# -------------------------------------------------------------------------
# 5. SIDEBAR
# -------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Intelligence Tool", "Project Details"], key="nav_menu")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background: #F5F5F7; padding: 16px; border-radius: 12px; border: 1px solid #E5E5EA;">
    <p style="margin:0; font-size: 10px; color: #86868B; font-weight: 600;">STUDENT AI PROJECT</p>
    <p style="margin:0; font-weight: 700; font-size: 16px; color: #1D1D1F;">Mohammad Hasnain</p>
    <p style="margin:0; font-size: 12px; color: #0071E3; font-weight: 500;">BS Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"✅ System Live | {len(df):,} Reviews")

# -------------------------------------------------------------------------
# 6. PAGES
# -------------------------------------------------------------------------
if menu == "Home":
    st.title("Intelligence")
    st.markdown("<p style='color: #86868B; font-size: 20px;'>Pro-level sentiment analysis.</p>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1551434678-e076c223a692?q=80&w=2850&auto=format&fit=crop", use_column_width=True)
    st.button("Launch Intelligence Tool", on_click=go_to_tool)

elif menu == "Intelligence Tool":
    st.title("Feedback Analyzer")

    # ── VOICE INPUT SECTION ──
    st.markdown('<div class="voice-box">', unsafe_allow_html=True)
    st.markdown('<p class="voice-title">🎙️ Voice Input — Speak your review</p>', unsafe_allow_html=True)
    st.markdown("Record your voice and it will be automatically converted to text", unsafe_allow_html=True)

    audio_input = st.audio_input("Click to record your review")

    if audio_input is not None:
        with st.spinner("Transcribing your voice..."):
            audio_bytes = audio_input.read()
            transcribed = transcribe_audio_groq(audio_bytes)
            if transcribed:
                st.session_state.voice_text = transcribed
                st.success("✅ Voice transcribed successfully!")
                st.markdown(f'<div class="transcribed-box">📝 <b>You said:</b> {transcribed}</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Could not transcribe. Please try again.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── TEXT INPUT — pre-filled with voice text if available ──
    st.markdown("**Or type your review below:**")
    user_input = st.text_area(
        "Review",
        value=st.session_state.voice_text,
        height=150,
        placeholder="Paste or type feedback here...",
        label_visibility="hidden"
    )

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            # Run Engines
            nb_p = nb_model.predict(tfidf.transform([user_input]))[0]
            nb_c = np.max(nb_model.predict_proba(tfidf.transform([user_input]))[0])

            dl_c = 0
            dl_sent = "N/A"
            if deep_engine_status:
                seq = tokenizer.texts_to_sequences([user_input])
                padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
                dl_prob = dl_model.predict(padded, verbose=0)[0][0]
                dl_sent = "POSITIVE" if dl_prob > 0.5 else "NEGATIVE"
                dl_c = dl_prob if dl_prob > 0.5 else (1 - dl_prob)

            nb_win = nb_c >= dl_c

            html_nb = f"""
<div class="mac-card {'winner-card' if nb_win else ''}">
{f'<div class="winner-badge">MOST TRUSTED</div>' if nb_win else ''}
<p class="engine-tag">⚡ FAST ENGINE (NAIVE BAYES)</p>
<p class="{'pos' if nb_p=='positive' else 'neg'}">{nb_p.upper()}</p>
<p class="conf-val">Confidence: {nb_c*100:.1f}%</p>
</div>"""
            st.markdown(html_nb, unsafe_allow_html=True)

            if deep_engine_status:
                html_dl = f"""
<div class="mac-card {'winner-card' if not nb_win else ''}">
{f'<div class="winner-badge">MOST TRUSTED</div>' if not nb_win else ''}
<p class="engine-tag">🧠 DEEP ENGINE (ATTENTION)</p>
<p class="{'pos' if dl_sent=='POSITIVE' else 'neg'}">{dl_sent}</p>
<p class="conf-val">Confidence: {dl_c*100:.1f}%</p>
</div>"""
                st.markdown(html_dl, unsafe_allow_html=True)
        else:
            st.warning("Enter text or record your voice first.")

elif menu == "Project Details":
    st.title("Architecture")

    st.markdown("### 🛠️ System Architecture (The Big Data Pipeline)")
    st.markdown("""
    This system was built to handle the **Volume** and **Variety** of the Yelp Open Dataset.

    1.  **Data Ingestion (Chunking):** The raw file was **8.6 GB** (JSON). Used Python Generators to stream data line-by-line.
    2.  **ETL & Preprocessing:** Parsed JSON to CSV, removed neutral (3-star) reviews.
    3.  **Balancing:** Applied **Undersampling** to achieve a perfect 50/50 split.
    4.  **Vectorization:** Used **TF-IDF** for the Fast Engine and **Embedding/Padding** for the Deep Engine.
    """)

    st.markdown("---")

    st.markdown("### 🧠 Dual-Engine Intelligence")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("""
        **⚡ Fast Engine (Naive Bayes)**
        * **Type:** Statistical Probability
        * **Strength:** Extremely fast, works with massive datasets.
        * **Weakness:** Doesn't understand word order (Bag of Words).
        """)

    with colB:
        st.markdown("""
        **🧠 Deep Engine (Attention LSTM)**
        * **Type:** Deep Learning Neural Network
        * **Strength:** Understands context and word sequences using an Attention Mechanism.
        * **Weakness:** Computationally expensive.
        """)

    st.markdown("---")
    st.markdown("### 📊 Dataset Statistics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Positive Samples", f"{len(df[df['sentiment']=='positive']):,}")
    m3.metric("Negative Samples", f"{len(df[df['sentiment']=='negative']):,}")
    st.bar_chart(df['sentiment'].value_counts(), color="#0071E3")
