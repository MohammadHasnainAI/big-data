
import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------------------------------------------------
# 1. APP CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligence: Big Data Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# 2. PROFESSIONAL STYLING (CSS)
# -------------------------------------------------------------------------
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
    }
    div.stButton > button:hover {
        background-color: #b31e1e;
        color: white;
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
    }
    
    /* Card Styling for Home Page */
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 5px solid #D32323;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 3. BACKEND: LOAD DATA & TRAIN MODEL
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    csv_file = "yelp_web.csv"
    try:
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['text'])
        df = df[df['stars'] != 3]
        df['sentiment'] = df['stars'].apply(lambda x: 'positive' if x > 3 else 'negative')
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'yelp_web.csv' not found. Please upload it to GitHub.")
        st.stop()

df = load_data()

# Train Model on the fly (Cached)
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
X_vec = tfidf.fit_transform(df['text'])
y = df['sentiment']
model = MultinomialNB()
model.fit(X_vec, y)

# -------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Intelligence Tool", "Project Details"])

st.sidebar.markdown("---")

# STUDENT PROFILE CARD
st.sidebar.markdown("""
    <div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3;">
        <small>Developed by:</small><br>
        <strong>Mohammad Hasnain</strong><br>
        BS Artificial Intelligence
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"✅ System Online | {len(df):,} Reviews")

# -------------------------------------------------------------------------
# 5. PAGE: HOME
# -------------------------------------------------------------------------
if menu == "Home":
    st.title("🧠 Intelligence: Big Data Analyzer")
    st.image("https://cdn.dribbble.com/users/2064121/screenshots/15865261/media/58102a06145892552601724682057636.jpg?compress=1&resize=1200x900", use_column_width=True)
    
    st.markdown("""
    ### Welcome to the Big Data Feedback System
    This project uses **Machine Learning (Naive Bayes)** and **Big Data Processing** to analyze customer sentiment from the **Yelp Open Dataset**.

    **Key Features:**
    - ⚡ **Real-time NLP Analysis** of unstructured text.
    - ⚖️ **Balanced Dataset** (Undersampling) to prevent bias.
    - 📂 **Big Data Pipeline** handling 8GB+ of raw JSON data.

    👈 Select **Intelligence Tool** from the sidebar to start the analysis.
    """)

# -------------------------------------------------------------------------
# 6. PAGE: INTELLIGENCE TOOL (The App)
# -------------------------------------------------------------------------
elif menu == "Intelligence Tool":
    st.title("🚀 Customer Feedback Analyzer")
    st.write("Enter unstructured review text below to detect sentiment using AI.")
    
    st.divider()

    col1, col2 = st.columns([2, 1], gap="medium")

    with col1:
        user_input = st.text_area("✍️ Input Feedback:", height=200, placeholder="Example: The service was slow but the food was absolutely delicious!")
        analyze_btn = st.button("Analyze Sentiment", type="primary")

    with col2:
        st.write("#### 🔍 Prediction Result")
        
        if analyze_btn:
            if user_input.strip():
                with st.spinner("Processing Big Data vectors..."):
                    time.sleep(0.8) # UI Effect
                    
                    # Prediction Logic
                    input_vec = tfidf.transform([user_input])
                    prediction = model.predict(input_vec)[0]
                    probs = model.predict_proba(input_vec)[0]
                    
                    conf_neg = probs[0]
                    conf_pos = probs[1]
                    
                    if prediction == 'positive':
                        st.success(f"😊 POSITIVE RESPONSE")
                        st.metric("Confidence", f"{conf_pos*100:.1f}%")
                        st.progress(conf_pos)
                    else:
                        st.error(f"😡 NEGATIVE RESPONSE")
                        st.metric("Confidence", f"{conf_neg*100:.1f}%")
                        st.progress(conf_neg)
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
    **Program:** BS Artificial Intelligence (5th Semester)

    ---

    #### 🎓 Academic Supervision
    **Supervisor:** Engr. Aneela Habib  
    *Big Data Management and Processing*

    ---

    ### 🛠️ System Architecture (The Big Data Pipeline)
    This system was built to handle the **Volume** and **Variety** of the Yelp Open Dataset.

    1.  **Data Ingestion (Chunking):** - The raw file was **8.6 GB** (JSON).
        - Used Python Generators to stream data line-by-line to avoid Memory Overflow (RAM Crash).
    
    2.  **ETL & Preprocessing:**
        - **Extraction:** Parsed JSON to CSV.
        - **Transformation:** Removed 3-star (neutral) reviews to sharpen accuracy.
        - **Balancing:** Detected Class Imbalance (80% Positive) and applied **Undersampling** to create a perfect 50/50 split.

    3.  **Machine Learning:**
        - **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency).
        - **Model:** Multinomial Naive Bayes (Probabilistic Classifier).
    
    ---
    **Dataset Source:** [Yelp Open Dataset](https://www.yelp.com/dataset)
    """)
