
import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Intelligence: Big Data Analyzer",
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
    
    /* Custom Red Buttons */
    div.stButton > button {
        background-color: #D32323; 
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
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #ddd;
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

# --- 5. PROFESSIONAL SIDEBAR ---
with st.sidebar:
    # BIG DATA IMAGE (Cloud/Data Icon)
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    
    st.markdown("## ⚙️ Project Dashboard")
    
    # --- STUDENT PROFILE (Blue Card) ---
    st.markdown("""
    <div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <small style="color: #555; font-weight: bold;">DEVELOPER PROFILE</small><br>
        <strong style="font-size: 18px; color: #0d47a1;">Mohammad Hasnain</strong><br>
        <span style="color: #666; font-size: 14px;">BS Artificial Intelligence (5th Sem)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # --- METRICS SECTION ---
    st.markdown("### 📊 Dataset Metrics")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Total Rows", f"{len(df)//1000}k", delta="Balanced")
    with col_s2:
        st.metric("Features", "5,000", delta="TF-IDF")
    
    st.caption("Class Distribution (50/50 Split)")
    st.bar_chart(df['sentiment'].value_counts())
    
    # --- SUPERVISOR SECTION ---
    st.markdown("---")
    st.markdown("#### 🎓 Supervision")
    st.write("**Engr. Aneela Habib**")
    st.caption("Big Data Mgmt & Processing")

# --- 6. MAIN INTERFACE ---
col1, col2 = st.columns([5, 1])
with col1:
    st.title("Intelligence") 
    st.subheader("Big Data Customer Feedback Analyzer")
with col2:
    st.write("") 

st.markdown("---")

col_left, col_right = st.columns([2, 1], gap="medium")

with col_left:
    st.subheader("📝 Input Data Stream")
    user_input = st.text_area("Enter unstructured review text:", height=150, placeholder="e.g. The analytics dashboard was fast but the data export failed.")
    analyze_btn = st.button("🚀 Process & Analyze", use_container_width=True)

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
                        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <h3 style="color: #155724; margin:0;">😊 POSITIVE</h3>
                            <p style="margin:0;">Confidence Score: <b>{conf_pos*100:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(conf_pos)
                else:
                    st.markdown(f"""
                        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <h3 style="color: #721c24; margin:0;">😡 NEGATIVE</h3>
                            <p style="margin:0;">Confidence Score: <b>{conf_neg*100:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(conf_neg)
        else:
            st.warning("⚠️ Data Stream Empty. Please input text.")
    else:
        st.info("👈 System Ready. Awaiting Input.")

# --- 7. BIG DATA ARCHITECTURE FOOTER ---
with st.expander("🛠️ View System Architecture (Big Data Pipeline)"):
    st.markdown("""
    ### ⚙️ How this System Handles Big Data
    1.  **Data Ingestion (Streaming):** Utilized chunk-based processing to handle the 8GB Yelp JSON file.
    2.  **ETL Transformation:** * *Extraction:* Parsed unstructured JSON.
        * *Transformation:* Removed neutral reviews & applied Undersampling for class balance.
        * *Loading:* Optimized CSV loaded into memory.
    3.  **Machine Learning:** TF-IDF Vectorization (Bigrams) + Multinomial Naive Bayes.
    """)
