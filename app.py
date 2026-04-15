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
# 2. APP CONFIGURATION & SMOOTH CLASSIC CSS
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligence: Big Data Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup Session State for the new Home Page Button
if 'nav_menu' not in st.session_state:
    st.session_state.nav_menu = "Home"

def go_to_tool():
    st.session_state.nav_menu = "Intelligence Tool"

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Smooth & Professional Red Button */
    div.stButton > button {
        background-color: #D32323; /* Classic Yelp Red */
        color: white !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #b31e1e !important;
        color: white !important;
        box-shadow: 0 8px 15px rgba(211, 35, 35, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    div.stButton > button:active {
        transform: translateY(1px) !important;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #D32323;
        font-weight: bold;
    }
    
    /* Input Area Focus Effect */
    .stTextArea textarea:focus {
        border-color: #D32323 !important;
        box-shadow: 0 0 0 2px rgba(211, 35, 35, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 3. BACKEND: LOAD BOTH MODELS
# -------------------------------------------------------------------------
@st.cache_resource
def load_all_assets():
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
        deep_engine_status = True
    except Exception as e:
        dl_model, tokenizer = None, None
        deep_engine_status = False

    return df, tfidf, nb_model, dl_model, tokenizer, deep_engine_status

df, tfidf, nb_model, dl_model, tokenizer, deep_engine_status = load_all_assets()

# -------------------------------------------------------------------------
# 4. CLASSIC SIDEBAR NAVIGATION
# -------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
st.sidebar.title("Navigation")

# Radio button linked to session state!
menu = st.sidebar.radio("Go to:", ["Home", "Intelligence Tool", "Project Details"], key="nav_menu")

st.sidebar.markdown("---")

st.sidebar.markdown("""
    <div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <small style="color: #555;">Developed by:</small><br>
        <strong style="font-size: 1.1em; color: #333;">Mohammad Hasnain</strong><br>
        <span style="color: #0066cc;">BS Artificial Intelligence</span>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
if deep_engine_status:
    st.sidebar.success(f"✅ Dual-Engine Online\n\n📊 {len(df):,} Reviews")
else:
    st.sidebar.warning(f"⚠️ Fast Engine Only\n\n📊 {len(df):,} Reviews")

# -------------------------------------------------------------------------
# 5. PAGE: HOME
# -------------------------------------------------------------------------
if menu == "Home":
    st.title("🧠 Intelligence: Big Data Analyzer")
    st.image("https://cdn.dribbble.com/users/2064121/screenshots/15865261/media/58102a06145892552601724682057636.jpg?compress=1&resize=1200x900", use_column_width=True)
    
    st.markdown("""
    ### Welcome to the Big Data Feedback System
    This project uses **Machine Learning (Naive Bayes)** and **Deep Learning (Attention LSTM)** to analyze customer sentiment from the **Yelp Open Dataset**.

    **Key Features:**
    - ⚡ **Dual-Engine Analysis:** Compares probability ML with Neural Networks.
    - ⚖️ **Balanced Dataset:** Undersampled to prevent bias.
    - 📂 **Big Data Pipeline:** Handled 8GB+ of raw JSON data.
    """)
    
    st.write("")
    # Added the missing button to the home page!
    st.button("Launch Intelligence Tool 🚀", on_click=go_to_tool)

# -------------------------------------------------------------------------
# 6. PAGE: INTELLIGENCE TOOL
# -------------------------------------------------------------------------
elif menu == "Intelligence Tool":
    st.title("🚀 Customer Feedback Analyzer")
    st.write("Enter unstructured review text below to detect sentiment using AI.")
    
    st.divider()

    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        user_input = st.text_area("✍️ Input Feedback:", height=200, placeholder="Example: The service was slow but the food was absolutely delicious!")
        analyze_btn = st.button("Analyze Sentiment")

    with col2:
        st.write("#### 🔍 Prediction Results")
        
        results_container = st.container()
        quote_placeholder = st.empty()
        
        if analyze_btn:
            if user_input.strip():
                # Original Quotes List
                quotes = [
                    "**“It always seems impossible until it’s done.”**\n— Steve Jobs",
                    "**“The future depends on what you do today.”**\n— Albert Einstein",
                    "**“Opportunities don't happen, you create them.”**\n— William James",
                    "**“A journey of a thousand miles begins with a single step.”**\n— Lao Tzu"
                ]
                selected_quote = random.choice(quotes)
                
                quote_placeholder.info(f"💡 **Processing Big Data...**\n\n{selected_quote}")
                
                with st.spinner("Analyzing vectors..."):
                    time.sleep(2.0)
                
                # --- PREDICTIONS ---
                input_vec = tfidf.transform([user_input])
                nb_pred = nb_model.predict(input_vec)[0]
                nb_conf = np.max(nb_model.predict_proba(input_vec)[0])
                
                if deep_engine_status:
                    seq = tokenizer.texts_to_sequences([user_input])
                    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
                    dl_prob = dl_model.predict(padded)[0][0]
                    dl_sent = "positive" if dl_prob > 0.5 else "negative"
                    dl_conf = dl_prob if dl_prob > 0.5 else (1 - dl_prob)

                # --- RENDER HTML (BUG FIXED: Written on one line to prevent markdown errors) ---
                nb_bg = "#d4edda" if nb_pred == 'positive' else "#f8d7da"
                nb_border = "#28a745" if nb_pred == 'positive' else "#dc3545"
                nb_text = "#155724" if nb_pred == 'positive' else "#721c24"
                nb_emoji = "😊" if nb_pred == 'positive' else "😡"
                
                html_output = f'<div style="background-color: {nb_bg}; padding: 15px; border-radius: 8px; border-left: 5px solid {nb_border}; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);"><h4 style="color: {nb_text}; margin:0; font-size: 14px;">⚡ FAST ENGINE (NAIVE BAYES)</h4><p style="color: {nb_text}; font-size: 22px; font-weight: bold; margin: 5px 0 0 0;">{nb_emoji} {nb_pred.upper()}</p><p style="color: #444; margin: 0; font-size: 14px;">Confidence: <b>{nb_conf*100:.1f}%</b></p></div>'

                if deep_engine_status:
                    dl_bg = "#d4edda" if dl_sent == 'positive' else "#f8d7da"
                    dl_border = "#28a745" if dl_sent == 'positive' else "#dc3545"
                    dl_text = "#155724" if dl_sent == 'positive' else "#721c24"
                    dl_emoji = "😊" if dl_sent == 'positive' else "😡"
                    
                    html_output += f'<div style="background-color: {dl_bg}; padding: 15px; border-radius: 8px; border-left: 5px solid {dl_border}; box-shadow: 0 4px 6px rgba(0,0,0,0.05);"><h4 style="color: {dl_text}; margin:0; font-size: 14px;">🧠 DEEP ENGINE (ATTENTION LSTM)</h4><p style="color: {dl_text}; font-size: 22px; font-weight: bold; margin: 5px 0 0 0;">{dl_emoji} {dl_sent.upper()}</p><p style="color: #444; margin: 0; font-size: 14px;">Confidence: <b>{dl_conf*100:.1f}%</b></p></div>'

                with results_container:
                    st.markdown(html_output, unsafe_allow_html=True)
                
                quote_placeholder.info(f"✨ **Inspiration for you:**\n\n{selected_quote}")
                time.sleep(15)
                quote_placeholder.empty()

            else:
                st.warning("⚠️ Please enter text first.")

# -------------------------------------------------------------------------
# 7. PAGE: PROJECT DETAILS
# -------------------------------------------------------------------------
elif menu == "Project Details":
    st.title("ℹ️ Project Documentation")
    
    st.markdown("""
    ### Big Data Management & Processing
    **Student:** Mohammad Hasnain  
    **Program:** BS Artificial Intelligence (5th Semester)

    ---
    #### 🎓 Academic Supervision
    **Supervisor:** Engr. Aneela Habib  
    *Big Data Management and Processing*
    ---
    """)

    # PIPELINE SECTION ADDED HERE
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
    
    colA, colB = st.columns(2)
    with colA:
        st.info("**⚡ The Old Model (Fast Engine)**\n\n**Algorithm:** Multinomial Naive Bayes\n\n**How it works:** A statistical model based on probability. It breaks the review down into individual words and counts them.\n\n**Pros:** Incredibly fast and lightweight.\n\n**Cons:** It suffers from \"Bag of Words\" syndrome. It does not understand word order.")
    with colB:
        st.success("**🧠 The New Model (Deep Engine)**\n\n**Algorithm:** Custom Attention LSTM\n\n**How it works:** An artificial neural network with an \"Attention Mechanism\". It reads the sentence in order and assigns weight to important words.\n\n**Pros:** Understands complex context and sarcasm.\n\n**Cons:** Requires significantly more computing power to train.")

    st.markdown("---")
    st.markdown("### 📊 Dataset Statistics")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Positive Samples", f"{len(df[df['sentiment']=='positive']):,}")
    m3.metric("Negative Samples", f"{len(df[df['sentiment']=='negative']):,}")
    
    st.write("")
    st.markdown("**Visualizing Class Balance:**")
    chart_data = df['sentiment'].value_counts()
    st.bar_chart(chart_data, color="#D32323")
    st.caption("Figure 1: Perfect 50/50 Class Balance achieved via Undersampling Algorithm.")
