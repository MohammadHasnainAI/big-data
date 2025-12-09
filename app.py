
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Yelp Sentiment AI",
    page_icon="⭐",
    layout="wide"
)

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    # Since app.py and csv are in the same folder, we just use the filename
    csv_file = "yelp_reviews_100k.csv"
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        st.error(f"❌ Error: Could not find '{csv_file}'. Make sure it is in the same folder as app.py")
        st.stop()
            
    # Cleaning: Drop missing text
    df = df.dropna(subset=['text'])
    
    # Filter: Remove Neutral (3-star) reviews for better accuracy
    df = df[df['stars'] != 3]
    
    # Labeling: 4-5 stars = Positive, 1-2 stars = Negative
    df['sentiment'] = df['stars'].apply(lambda x: 'positive' if x > 3 else 'negative')
    return df

# Load Data
df = load_data()

# --- 3. TRAIN MODEL ---
# Using Bigrams (1,2) to catch phrases like "not good"
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
X_vec = tfidf.fit_transform(df['text'])
y = df['sentiment']

model = MultinomialNB()
model.fit(X_vec, y)

# --- 4. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.header("📊 Project Dashboard")
    st.info(f"**Dataset:** Yelp Open Dataset")
    st.write(f"**Processed Reviews:** {len(df):,}")
    
    # Sentiment Distribution
    st.write("### Data Distribution")
    st.bar_chart(df['sentiment'].value_counts())
    
    st.caption("Big Data Management & Processing | BS AI Project")

# --- 5. MAIN INTERFACE ---
st.title("⭐ Yelp Review Sentiment AI")
st.markdown("### Intelligent Customer Feedback Analysis")
st.write("This system uses **Naive Bayes** trained on **100,000 real-world reviews** to detect sentiment.")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("✍️ Paste a review here:", height=150, placeholder="e.g. The service was slow but the food was absolutely delicious!")
    analyze_btn = st.button("Analyze Sentiment", type="primary")

with col2:
    st.markdown("#### 🔍 AI Prediction")
    if analyze_btn and user_input.strip():
        # Transform & Predict
        input_vec = tfidf.transform([user_input])
        prediction = model.predict(input_vec)[0]
        probs = model.predict_proba(input_vec)[0] # [Prob_Neg, Prob_Pos]
        
        conf_neg = probs[0]
        conf_pos = probs[1]
        
        if prediction == 'positive':
            st.success("😊 **POSITIVE**")
            st.metric("Confidence", f"{conf_pos*100:.1f}%")
            st.progress(conf_pos)
        else:
            st.error("😡 **NEGATIVE**")
            st.metric("Confidence", f"{conf_neg*100:.1f}%")
            st.progress(conf_neg)
            
    elif analyze_btn:
        st.warning("Please enter text first.")

# --- 6. INSIGHTS ---
with st.expander("📈 See how the Big Data was processed"):
    st.write("1. **Data Ingestion:** Streamed 8GB JSON file line-by-line (Chunking).")
    st.write("2. **Preprocessing:** Removed neutral reviews and vectorized text using TF-IDF.")
    st.write("3. **Modeling:** Trained a Probabilistic Naive Bayes classifier.")
