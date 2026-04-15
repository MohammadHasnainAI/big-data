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
# 2. APP CONFIGURATION & PROFESSIONAL CSS
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligence: Big Data Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Navigation Button Styling */
    div.stButton > button {
        background-color: #D32323; /* Yelp Red */
        color: white;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #b31e1e;
        color: white;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #D32323;
    }
    
    /* Premium Result Cards */
    .result-card {
        padding: 20px; 
        border-radius: 12px; 
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border-left: 6px solid;
    }
    .positive-card { background-color: #f0fdf4; border-color: #22c55e; }
    .negative-card { background-color: #fef2f2; border-color: #ef4444; }
    .card-title { margin: 0; font-size: 1.1rem; color: #374151; font-weight: 600; text-transform: uppercase;}
    .card-conf { margin: 5px 0 0 0; font-size: 1.5rem; font-weight: bold; }
    .pos-text { color: #15803d; }
    .neg-text { color: #b91c1c; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 3. BACKEND: LOAD BOTH MODELS (CACHED FOR SPEED)
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
# 4. SIDEBAR NAVIGATION
# -------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Intelligence Tool", "Project Details"])

st.sidebar.markdown("---")

st.sidebar.markdown("""
    <div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3;">
        <small>Developed by:</small><br>
        <strong>Mohammad Hasnain</strong><br>
        BS Artificial Intelligence
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
    ### Welcome to the Advanced Feedback System
    This project combines **Machine Learning (Naive Bayes)** and **Deep Learning (Attention LSTM)** to analyze customer sentiment.

    **Key Features:**
    - ⚡ **Dual-Engine NLP Analysis:** Compares statistical ML with Deep Learning.
    - ⚖️ **Balanced Dataset:** Undersampled Yelp Big Data.
    - 🧠 **Context Awareness:** LSTM understands the sequence and context of words.

    👈 Select **Intelligence Tool** from the sidebar to test the AI.
    """)

# -------------------------------------------------------------------------
# 6. PAGE: INTELLIGENCE TOOL
# -------------------------------------------------------------------------
elif menu == "Intelligence Tool":
    st.title("🚀 Customer Feedback Analyzer")
    st.write("Enter unstructured review text below to detect sentiment using our Dual-Engine AI.")
    st.divider()

    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        user_input = st.text_area("✍️ Input Feedback:", height=200, placeholder="Example: The service was slow but the food was absolutely delicious!")
        analyze_btn = st.button("Run Dual-Engine Analysis", type="primary")

    with col2:
        st.write("#### 🔍 Prediction Results")
        quote_placeholder = st.empty()
        results_placeholder = st.empty()
        
        if analyze_btn:
            if user_input.strip():
                # --- QUOTES LIST ---
                quotes = [
                    "**“It always seems impossible until it’s done.”**\n— Steve Jobs",
                    "**“The future depends on what you do today.”**\n— Albert Einstein",
                    "**“Opportunities don't happen, you create them.”**\n— William James",
                    "**“The secret of getting ahead is getting started.”**\n— Winston Churchill",
                    "**“A journey of a thousand miles begins with a single step.”**\n— Lao Tzu"
                ]
                selected_quote = random.choice(quotes)
                
                # 1. SHOW QUOTE & SPINNER
                quote_placeholder.info(f"💡 **Processing via Neural Networks...**\n\n{selected_quote}")
                
                with st.spinner("Analyzing text vectors & attention weights..."):
                    time.sleep(2.5) 
                
                # 2. RUN ENGINE 1: NAIVE BAYES
                input_vec = tfidf.transform([user_input])
                nb_pred = nb_model.predict(input_vec)[0]
                nb_conf = np.max(nb_model.predict_proba(input_vec)[0])
                
                # Format Naive Bayes UI
                nb_css = "positive-card" if nb_pred == 'positive' else "negative-card"
                nb_text_css = "pos-text" if nb_pred == 'positive' else "neg-text"
                nb_emoji = "😊" if nb_pred == 'positive' else "😡"
                
                html_output = f"""
                <div class="result-card {nb_css}">
                    <p class="card-title">⚡ Fast Engine (Naive Bayes)</p>
                    <p class="card-conf {nb_text_css}">{nb_emoji} {nb_pred.upper()} <span style="font-size:1rem; font-weight:normal;">({nb_conf*100:.1f}%)</span></p>
                </div>
                """
                
                # 3. RUN ENGINE 2: ATTENTION LSTM (If available)
                if deep_engine_status:
                    seq = tokenizer.texts_to_sequences([user_input])
                    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
                    dl_prob = dl_model.predict(padded)[0][0]
                    dl_sent = "positive" if dl_prob > 0.5 else "negative"
                    dl_conf = dl_prob if dl_prob > 0.5 else (1 - dl_prob)
                    
                    dl_css = "positive-card" if dl_sent == 'positive' else "negative-card"
                    dl_text_css = "pos-text" if dl_sent == 'positive' else "neg-text"
                    dl_emoji = "😊" if dl_sent == 'positive' else "😡"
                    
                    html_output += f"""
                    <div class="result-card {dl_css}">
                        <p class="card-title">🧠 Deep Engine (Attention LSTM)</p>
                        <p class="card-conf {dl_text_css}">{dl_emoji} {dl_sent.upper()} <span style="font-size:1rem; font-weight:normal;">({dl_conf*100:.1f}%)</span></p>
                    </div>
                    """
                
                # 4. SHOW RESULTS & INSPIRATION
                results_placeholder.markdown(html_output, unsafe_allow_html=True)
                quote_placeholder.info(f"✨ **Inspiration for you:**\n\n{selected_quote}")
                
                # Wait 15 seconds then clear quote
                time.sleep(15)
                quote_placeholder.empty()

            else:
                st.warning("⚠️ Please enter text first.")
        else:
            st.info("Waiting for input...")

# -------------------------------------------------------------------------
# 7. PAGE: PROJECT DETAILS
# -------------------------------------------------------------------------
elif menu == "Project Details":
    st.title("ℹ️ Project Documentation")
    
    st.markdown("""
    ### Big Data Management & Processing
    **Student:** Mohammad Hasnain  
    **Program:** BS Artificial Intelligence

    ---
    #### 🎓 Academic Supervision
    **Supervisor:** Engr. Aneela Habib  
    *Big Data Management and Processing*
    ---
    """)

    st.markdown("### 📊 Dataset Statistics")
    st.write("To ensure the AI is not biased, the dataset was strictly balanced.")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Positive Samples", f"{len(df[df['sentiment']=='positive']):,}")
    m3.metric("Negative Samples", f"{len(df[df['sentiment']=='negative']):,}")
    
    st.write("")
    st.markdown("**Visualizing Class Balance:**")
    chart_data = df['sentiment'].value_counts()
    st.bar_chart(chart_data, color="#D32323")
    st.caption("Figure 1: Perfect 50/50 Class Balance achieved via Undersampling Algorithm.")

    st.markdown("""
    ---
    ### 🛠️ Dual-Engine Architecture
    This system was upgraded to handle both speed and context understanding.

    1. **Data Ingestion & Balancing:** Streamed 8.6 GB of raw Yelp JSON data and applied undersampling for a 50/50 split.
    2. **Engine 1 (Fast): Multinomial Naive Bayes.** Uses TF-IDF vectorization to probabilistically map words to sentiments. Highly efficient for Big Data.
    3. **Engine 2 (Deep): Custom Attention LSTM.** A neural network that uses an Attention Mechanism to "look back" at previous words in a sentence, allowing it to understand complex context and sarcasm.
    """)
