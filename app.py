
import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Intelligence: Big Data Analyzer", # Browser Tab Title
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL UI STYLING ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Button Styling */
    div.stButton > button {
        background-color: #D32323; /* Brand Red */
        color: white;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
    }
    div.stButton > button:hover {
        background-color: #b31e1e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD DATA ---
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
        st.error("❌ Error: 'yelp_web.csv' not found.")
        st.stop()

df = load_data()

# --- 4. TRAIN MODEL ---
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
X_vec = tfidf.fit_transform(df['text'])
y = df['sentiment']
model = MultinomialNB()
model.fit(X_vec, y)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/ad/Yelp_Logo.svg", width=150)
    st.markdown("## ⚙️ Control Panel")
    
    # STUDENT INFO
    st.info("**System:** Intelligence Analyzer\n**Student:** [Your Name Here]\n**Semester:** 5th (BS AI)")
    
    st.markdown("---")
    st.metric("Reviews Processed", f"{len(df):,}")
    st.caption("Balanced Data Distribution")
    st.bar_chart(df['sentiment'].value_counts())
    st.success("✅ System Online")

# --- 6. MAIN INTERFACE ---
col1, col2 = st.columns([5, 1])
with col1:
    # YOUR SELECTED TITLE
    st.title("Intelligence") 
    st.subheader("Big Data Customer Feedback Analyzer")
with col2:
    st.write("") 

st.markdown("---")

col_left, col_right = st.columns([2, 1], gap="medium")

with col_left:
    st.subheader("📝 Input Data")
    user_input = st.text_area("Enter customer feedback:", height=150, placeholder="e.g. The food was fantastic but the waiting time was horrible.")
    analyze_btn = st.button("🚀 Analyze Feedback", use_container_width=True)

with col_right:
    st.subheader("🔍 Intelligence Report")
    
    if analyze_btn:
        if user_input.strip():
            with st.spinner("Processing Big Data vectors..."):
                time.sleep(0.8) # UX effect
                
                # Predict
                input_vec = tfidf.transform([user_input])
                prediction = model.predict(input_vec)[0]
                probs = model.predict_proba(input_vec)[0]
                
                conf_neg = probs[0]
                conf_pos = probs[1]
                
                if prediction == 'positive':
                    st.markdown(f"""
                        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                            <h3 style="color: #155724; margin:0;">😊 POSITIVE</h3>
                            <p>Confidence: <b>{conf_pos*100:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(conf_pos)
                else:
                    st.markdown(f"""
                        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;">
                            <h3 style="color: #721c24; margin:0;">😡 NEGATIVE</h3>
                            <p>Confidence: <b>{conf_neg*100:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(conf_neg)
        else:
            st.warning("⚠️ Please enter text first.")
    else:
        st.info("👈 Ready for analysis.")

# --- 7. BIG DATA FOOTER ---
with st.expander("🛠️ System Architecture (Big Data Pipeline)"):
    st.markdown("""
    * **Data Source:** Yelp Open Dataset (8GB JSON)
    * **ETL Pipeline:** Streaming Chunking -> Filtering -> Undersampling
    * **Processing:** 40,000 Balanced Reviews
    * **Model:** Naive Bayes Classifier with TF-IDF
    """)
