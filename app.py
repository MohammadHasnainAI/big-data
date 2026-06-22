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

# -------------------------------------------------------------------------
# GROQ API KEY — stored in Streamlit secrets
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
# 2. APP CONFIG & PROFESSIONAL CSS
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], [data-testid="stSidebar"] * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stApp { background-color: #F5F5F7; }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E5EA !important;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Blue buttons */
    div.stButton > button {
        background-color: #0071E3 !important;
        color: white !important;
        border-radius: 980px !important;
        padding: 12px 30px !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #0077ED !important;
        transform: scale(1.02) !important;
    }

    /* Result cards */
    .mac-card {
        background: white;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #E5E5EA;
        margin-bottom: 16px;
        position: relative;
        transition: all 0.2s ease;
    }
    .mac-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.10); }
    .winner-card { border: 2px solid #0071E3 !important; }
    .winner-badge {
        position: absolute; top: -13px; right: 20px;
        background: #0071E3; color: white; padding: 3px 14px;
        border-radius: 20px; font-size: 10px; font-weight: 700;
        letter-spacing: 0.5px;
    }
    .engine-tag {
        font-size: 11px; color: #86868B;
        text-transform: uppercase; font-weight: 600;
        margin-bottom: 8px; letter-spacing: 0.5px;
    }
    .pos { color: #34C759 !important; font-weight: 700; font-size: 28px; margin: 0; }
    .neg { color: #FF3B30 !important; font-weight: 700; font-size: 28px; margin: 0; }
    .conf-val { font-size: 14px; color: #86868B; margin-top: 6px; }

    /* ── VOICE SECTION ── */
    .voice-container {
        background: white;
        border-radius: 20px;
        padding: 32px 28px;
        border: 1px solid #E5E5EA;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 24px;
        text-align: center;
    }

    .voice-header {
        font-size: 18px;
        font-weight: 600;
        color: #1D1D1F;
        margin-bottom: 6px;
    }

    .voice-subtext {
        font-size: 14px;
        color: #86868B;
        margin-bottom: 24px;
    }

    .transcribed-result {
        background: #F5F5F7;
        border-radius: 14px;
        padding: 16px 20px;
        margin-top: 16px;
        font-size: 15px;
        color: #1D1D1F;
        text-align: left;
        border: 1px solid #E5E5EA;
        line-height: 1.6;
    }

    .transcribed-label {
        font-size: 11px;
        font-weight: 600;
        color: #86868B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .success-pill {
        display: inline-block;
        background: #E8F8EE;
        color: #34C759;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 20px;
        margin-bottom: 12px;
    }

    .divider-text {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 20px 0;
        color: #86868B;
        font-size: 13px;
        font-weight: 500;
    }
    .divider-text::before,
    .divider-text::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #E5E5EA;
    }

    /* Style the Streamlit audio recorder */
    [data-testid="stAudioInput"] {
        display: flex !important;
        justify-content: center !important;
        margin: 0 auto !important;
    }

    [data-testid="stAudioInput"] > div {
        border-radius: 50% !important;
        width: 72px !important;
        height: 72px !important;
        background: #0071E3 !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(0, 113, 227, 0.35) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stAudioInput"] > div:hover {
        background: #0077ED !important;
        transform: scale(1.06) !important;
        box-shadow: 0 6px 24px rgba(0, 113, 227, 0.45) !important;
    }

    [data-testid="stAudioInput"] svg {
        fill: white !important;
        width: 28px !important;
        height: 28px !important;
    }

    /* Text area */
    [data-testid="stTextArea"] textarea {
        border-radius: 14px !important;
        border: 1.5px solid #E5E5EA !important;
        font-size: 15px !important;
        padding: 14px !important;
        background: white !important;
        transition: border 0.2s ease !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #0071E3 !important;
        box-shadow: 0 0 0 3px rgba(0,113,227,0.12) !important;
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
        return None
    except:
        return None

# -------------------------------------------------------------------------
# 5. SIDEBAR
# -------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Intelligence Tool", "Project Details"], key="nav_menu")
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background:#F5F5F7;padding:16px;border-radius:12px;border:1px solid #E5E5EA;">
    <p style="margin:0;font-size:10px;color:#86868B;font-weight:600;">STUDENT AI PROJECT</p>
    <p style="margin:0;font-weight:700;font-size:16px;color:#1D1D1F;">Mohammad Hasnain</p>
    <p style="margin:0;font-size:12px;color:#0071E3;font-weight:500;">BS Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption(f"✅ System Live | {len(df):,} Reviews")

# -------------------------------------------------------------------------
# 6. PAGES
# -------------------------------------------------------------------------
if menu == "Home":
    st.title("Intelligence")
    st.markdown("<p style='color:#86868B;font-size:20px;'>Pro-level sentiment analysis.</p>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1551434678-e076c223a692?q=80&w=2850&auto=format&fit=crop", use_column_width=True)
    st.button("Launch Intelligence Tool", on_click=go_to_tool)

elif menu == "Intelligence Tool":
    st.title("Feedback Analyzer")
    st.markdown("<p style='color:#86868B;font-size:16px;margin-bottom:28px;'>Type your review or speak it — we'll analyze the sentiment instantly.</p>", unsafe_allow_html=True)

    # ── VOICE SECTION ──
    st.markdown('<div class="voice-container">', unsafe_allow_html=True)
    st.markdown('<p class="voice-header">🎙️ Speak Your Review</p>', unsafe_allow_html=True)
    st.markdown('<p class="voice-subtext">Tap the microphone, say your review, and we\'ll convert it to text automatically</p>', unsafe_allow_html=True)

    audio_input = st.audio_input(" ", label_visibility="collapsed")

    if audio_input is not None:
        with st.spinner("Converting your voice to text..."):
            audio_bytes = audio_input.read()
            transcribed = transcribe_audio_groq(audio_bytes)
            if transcribed:
                st.session_state.voice_text = transcribed
                st.markdown('<div class="success-pill">✓ Voice converted successfully</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="transcribed-result">
                    <div class="transcribed-label">What you said</div>
                    {transcribed}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Could not convert voice. Please try again.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── DIVIDER ──
    st.markdown('<div class="divider-text">or type your review below</div>', unsafe_allow_html=True)

    # ── TEXT INPUT ──
    user_input = st.text_area(
        "Review",
        value=st.session_state.voice_text,
        height=140,
        placeholder="Write your review here...",
        label_visibility="hidden"
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_btn = st.button("Analyze Sentiment", use_container_width=True)

    if analyze_btn:
        if user_input.strip():
            with st.spinner("Analyzing..."):
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

                st.markdown("<br>", unsafe_allow_html=True)
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"""
<div class="mac-card {'winner-card' if nb_win else ''}">
{f'<div class="winner-badge">MOST TRUSTED</div>' if nb_win else ''}
<p class="engine-tag">⚡ Fast Engine — Naive Bayes</p>
<p class="{'pos' if nb_p=='positive' else 'neg'}">{nb_p.upper()}</p>
<p class="conf-val">Confidence: {nb_c*100:.1f}%</p>
</div>""", unsafe_allow_html=True)

                with col_b:
                    if deep_engine_status:
                        st.markdown(f"""
<div class="mac-card {'winner-card' if not nb_win else ''}">
{f'<div class="winner-badge">MOST TRUSTED</div>' if not nb_win else ''}
<p class="engine-tag">🧠 Deep Engine — Attention LSTM</p>
<p class="{'pos' if dl_sent=='POSITIVE' else 'neg'}">{dl_sent}</p>
<p class="conf-val">Confidence: {dl_c*100:.1f}%</p>
</div>""", unsafe_allow_html=True)
        else:
            st.warning("Please enter a review or record your voice first.")

elif menu == "Project Details":
    st.title("Architecture")

    st.markdown("### 🛠️ System Architecture (The Big Data Pipeline)")
    st.markdown("""
    This system was built to handle the **Volume** and **Variety** of the Yelp Open Dataset.

    1. **Data Ingestion (Chunking):** The raw file was **8.6 GB** (JSON). Used Python Generators to stream data line-by-line.
    2. **ETL & Preprocessing:** Parsed JSON to CSV, removed neutral (3-star) reviews.
    3. **Balancing:** Applied **Undersampling** to achieve a perfect 50/50 split.
    4. **Vectorization:** Used **TF-IDF** for the Fast Engine and **Embedding/Padding** for the Deep Engine.
    """)

    st.markdown("---")
    st.markdown("### 🧠 Dual-Engine Intelligence")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("""
        **⚡ Fast Engine (Naive Bayes)**
        * **Type:** Statistical Probability
        * **Strength:** Extremely fast, works with massive datasets.
        * **Weakness:** Doesn't understand word order.
        """)

    with colB:
        st.markdown("""
        **🧠 Deep Engine (Attention LSTM)**
        * **Type:** Deep Learning Neural Network
        * **Strength:** Understands context using Attention Mechanism.
        * **Weakness:** Computationally expensive.
        """)

    st.markdown("---")
    st.markdown("### 📊 Dataset Statistics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Positive Samples", f"{len(df[df['sentiment']=='positive']):,}")
    m3.metric("Negative Samples", f"{len(df[df['sentiment']=='negative']):,}")
    st.bar_chart(df['sentiment'].value_counts(), color="#0071E3")
