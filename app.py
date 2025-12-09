
import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Yelp Sentiment AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (To Hide Streamlit Branding) ---
st.markdown("""
    <style>
    /* Hide the Streamlit Hamburger Menu & Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Button Style */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 24px;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    csv_file = "yelp_web.csv"
    try:
        df = pd.read_csv(csv_file)
        # Ensure data is clean
        df = df.dropna(subset=['text'])
        df = df[df['stars'] != 3]
        df['sentiment'] = df['stars'].apply(lambda x: 'positive' if x > 3 else 'negative')
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: '{csv_file}' not found. Please upload it to GitHub.")
        st.stop()

df = load_data()

# --- 4. MODEL TRAINING ---
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
X_vec = tfidf.fit_transform(df['text'])
y = df['sentiment']
model = MultinomialNB()
model.fit(X_vec, y)

# --- 5. SIDEBAR: PROJECT METADATA ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/ad/Yelp_Logo.svg", width=150)
    st.title("Project Controls")
    
    st.markdown("### 👤 Student Details")
    st.info("**Name:** [Your Name Here]\n**ID:** [Your ID Here]\n**Semester:** 5th (BS AI)")
    
    st.markdown("---")
    st.markdown("### 📊 Live Dataset Stats")
    st.write(f"**Total Records:** {len(df):,}")
    
    # Simple Chart
    st.caption("Class Balance (Pos vs Neg)")
    st.bar_chart(df['sentiment'].value_counts(), color=["#FF4B4B"])  # Red color theme
    
    st.markdown("---")
    st.success("✅ System Status: **Online**")

# --- 6. MAIN INTERFACE ---
# Header Section
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.write("") # Spacer
with col_title:
    st.title("Sentiment Analysis Engine")
    st.caption("Big Data Processing • Natural Language Processing • Machine Learning")

st.markdown("---")

# Input Section
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("📝 Input Review")
    user_input = st.text_area("Enter customer feedback below:", height=200, placeholder="Type here... e.g. 'The delivery was incredibly fast and the food was hot!'")
    
    # Action Button
    analyze_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)

with col2:
    st.subheader("🔍 Prediction Results")
    
    if analyze_btn:
        if user_input.strip():
            # Add a Loading Spinner to look professional
            with st.spinner("Processing text vectors..."):
                time.sleep(1) # Fake delay to let user see the spinner
                
                # Predict
                input_vec = tfidf.transform([user_input])
                prediction = model.predict(input_vec)[0]
                probs = model.predict_proba(input_vec)[0]
                
                conf_neg = probs[0]
                conf_pos = probs[1]
                
                # Display Card
                if prediction == 'positive':
                    st.markdown("""
                        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                            <h2 style="color: #155724; margin:0;">😊 POSITIVE</h2>
                            <p>The model is <b>{:.1f}%</b> confident.</p>
                        </div>
                    """.format(conf_pos*100), unsafe_allow_html=True)
                    st.progress(conf_pos)
                    
                else:
                    st.markdown("""
                        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;">
                            <h2 style="color: #721c24; margin:0;">😡 NEGATIVE</h2>
                            <p>The model is <b>{:.1f}%</b> confident.</p>
                        </div>
                    """.format(conf_neg*100), unsafe_allow_html=True)
                    st.progress(conf_neg)

        else:
            st.warning("⚠️ Please enter some text to analyze.")
    
    else:
        st.info("👈 Enter text and click Analyze to see the AI in action.")

# --- 7. TECHNICAL FOOTER ---
with st.expander("🛠️ View System Architecture (Big Data Pipeline)"):
    st.markdown("""
    1. **Data Ingestion:** Chunk-based streaming of 8GB JSON Data.
    2. **ETL Process:** Cleaning, Filtering (3-star removal), and Dynamic Balancing.
    3. **Undersampling:** Random majority undersampling to achieve 50/50 Class Balance.
    4. **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) with Bigrams.
    5. **Inference:** Multinomial Naive Bayes Probabilistic Classifier.
    """)
