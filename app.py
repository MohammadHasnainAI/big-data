
import streamlit as st
import pandas as pd
import time
import random
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
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #D32323;
    }
    
    /* Quote Box Styling */
    .stAlert {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
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

# Train Model
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
        
        # Create a placeholder for the quote
        quote_placeholder = st.empty()
        
        if analyze_btn:
            if user_input.strip():
                # --- QUOTES LIST (15 Quotes) ---
                quotes = [
                    "**“It always seems impossible until it’s done.”**\n— Steve Jobs",
                    "**“In the middle of every difficulty lies opportunity.”**\n— Nelson Mandela",
                    "**“The future depends on what you do today.”**\n— Albert Einstein",
                    "**“Don’t let yesterday take up too much of today.”**\n— Mahatma Gandhi",
                    "**“Act as if what you do makes a difference. It does.”**\n— Will Rogers",
                    "**“Opportunities don't happen, you create them.”**\n— William James",
                    "**“Success is walking from failure to failure with no loss of enthusiasm.”**\n— Chris Grosser",
                    "**“The secret of getting ahead is getting started.”**\n— Winston Churchill",
                    "**“What you get by achieving your goals is not as important as what you become by achieving your goals.”**\n— Mark Twain",
                    "**“Hardships often prepare ordinary people for an extraordinary destiny.”**\n— Zig Ziglar",
                    "**“Quality is not an act, it is a habit.”**\n— C.S. Lewis",
                    "**“Everything you’ve ever wanted is sitting on the other side of fear.”**\n— Aristotle",
                    "**“Do what you can, with what you have, where you are.”**\n— George Addair",
                    "**“A journey of a thousand miles begins with a single step.”**\n— Theodore Roosevelt",
                    "**“The journey of a thousand miles begins with one step.”**\n— Lao Tzu"
                ]
                
                selected_quote = random.choice(quotes)
                
                # 1. SHOW QUOTE DURING PROCESSING
                quote_placeholder.info(f"💡 **Processing Big Data...**\n\n{selected_quote}")
                
                with st.spinner("Analyzing vectors..."):
                    time.sleep(2.5) 
                
                # 2. PERFORM PREDICTION
                input_vec = tfidf.transform([user_input])
                prediction = model.predict(input_vec)[0]
                probs = model.predict_proba(input_vec)[0]
                
                conf_neg = probs[0]
                conf_pos = probs[1]
                
                # 3. SHOW RESULT
                if prediction == 'positive':
                    st.markdown(f"""
                        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                            <h3 style="color: #155724; margin:0;">😊 POSITIVE RESPONSE</h3>
                            <p>Confidence: <b>{conf_pos*100:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(conf_pos)
                else:
                    st.markdown(f"""
                        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;">
                            <h3 style="color: #721c24; margin:0;">😡 NEGATIVE RESPONSE</h3>
                            <p>Confidence: <b>{conf_neg*100:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(conf_neg)
                
                # 4. SHOW QUOTE AFTER PREDICTION (For 15 Seconds)
                # We update the text to indicate it's an 'Inspiration' now
                quote_placeholder.info(f"✨ **Inspiration for you:**\n\n{selected_quote}")
                
                # Wait 15 seconds so user can read it
                time.sleep(15)
                
                # Finally clear the quote
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
    **Program:** BS Artificial Intelligence (5th Semester)

    ---

    #### 🎓 Academic Supervision
    **Supervisor:** Engr. Aneela Habib  
    *Big Data Management and Processing*

    ---
    """)

    st.markdown("### 📊 Dataset Statistics")
    st.write("To ensure the AI is not biased, the dataset was strictly balanced.")
    
    # METRICS
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Positive Samples", f"{len(df[df['sentiment']=='positive']):,}")
    m3.metric("Negative Samples", f"{len(df[df['sentiment']=='negative']):,}")
    
    st.write("")
    st.markdown("**Visualizing Class Balance:**")
    
    # CHART - Uses the Red Brand Color
    chart_data = df['sentiment'].value_counts()
    st.bar_chart(chart_data, color="#D32323")
    
    st.caption("Figure 1: Perfect 50/50 Class Balance achieved via Undersampling Algorithm.")

    st.markdown("""
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
