
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- 1. PAGE CONFIGURATION (Professional Look) ---
st.set_page_config(
    page_title="Yelp Sentiment AI",
    page_icon="⭐",
    layout="wide"
)

# --- 2. LOAD DATA FUNCTION ---
@st.cache_data
def load_data():
    csv_file = "yelp_reviews_100k.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        try:
            # Fallback for local testing
            df = pd.read_csv(r"C:\Users\hasna\Downloads\yelp_reviews_100k.csv")
        except:
            st.error(f"❌ Error: Could not find '{csv_file}'. Please run the Data Processor script first.")
            st.stop()
            
    # Data Cleaning
    df = df.dropna(subset=['text'])
    
    # FILTER: Remove 3-star reviews (Neutral reviews confuse the model)
    df = df[df['stars'] != 3]
    
    # Labeling: 4-5 stars = Positive, 1-2 stars = Negative
    df['sentiment'] = df['stars'].apply(lambda x: 'positive' if x > 3 else 'negative')
    return df

# Load the data
df = load_data()

# --- 3. TRAIN MODEL BEHIND THE SCENES ---
# We use ngram_range=(1,2) to capture phrases like "not good"
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
X_vec = tfidf.fit_transform(df['text'])
y = df['sentiment']

model = MultinomialNB()
model.fit(X_vec, y)

# --- 4. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.header("📊 Data Dashboard")
    st.info(f"**Dataset:** Yelp Open Data")
    st.write(f"**Total Reviews:** {len(df):,}")
    
    # Calculate counts
    pos_count = len(df[df['sentiment']=='positive'])
    neg_count = len(df[df['sentiment']=='negative'])
    
    # Simple Bar Chart for Context
    st.write("### Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())
    
    st.markdown("---")
    st.caption("Big Data Management Project | BS AI")

# --- 5. MAIN APP INTERFACE ---
st.title("⭐ Yelp Review Sentiment AI")
st.markdown("### Intelligent Customer Feedback Analysis")
st.write("This AI processes natural language to determine if a customer review is **Positive** or **Negative**.")

# Layout: Input on left, Results on right
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("✍️ Paste a review here:", height=150, placeholder="e.g. The food was cold and the service was terrible.")
    analyze_btn = st.button("Analyze Sentiment", type="primary")

with col2:
    st.markdown("#### 🔍 AI Prediction")
    if analyze_btn and user_input.strip():
        # Predict
        input_vec = tfidf.transform([user_input])
        prediction = model.predict(input_vec)[0]
        probs = model.predict_proba(input_vec)[0] # [Neg, Pos]
        
        conf_neg = probs[0]
        conf_pos = probs[1]
        
        if prediction == 'positive':
            st.success("😊 **POSITIVE**")
            st.metric("Confidence Score", f"{conf_pos*100:.1f}%")
            st.progress(conf_pos)
        else:
            st.error("😡 **NEGATIVE**")
            st.metric("Confidence Score", f"{conf_neg*100:.1f}%")
            st.progress(conf_neg)
    elif analyze_btn:
        st.warning("Please enter text to analyze.")

# --- 6. MODEL INSIGHTS (The 'Pro' Feature) ---
with st.expander("📈 How does this model work?"):
    st.write("This model uses **Naive Bayes** and **TF-IDF Vectorization**.")
    st.write("It was trained on **100,000 real-world reviews** from the Yelp Dataset.")
    st.info("Top Positive keywords usually include: *great, amazing, delicious, friendly*")
    st.info("Top Negative keywords usually include: *terrible, rude, bad, dry, wait*")
