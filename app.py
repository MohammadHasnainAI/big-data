
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- PAGE CONFIG ---
st.set_page_config(page_title="Review Analyzer Pro", page_icon="🤖")

# --- TITLE & SIDEBAR ---
st.title("Amazon Review Sentiment Analysis 🤖")
st.markdown("Enter a review to see if the AI thinks it is **Positive** or **Negative**.")

st.sidebar.header("Model Metrics")
st.sidebar.info("This model uses **TF-IDF (1-2 ngrams)** and **Naive Bayes**.")

# --- LOAD DATA FUNCTION ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("amazon_reviews.csv")
    except FileNotFoundError:
        try:
            # Fallback for local Windows path
            df = pd.read_csv(r"C:\Users\hasna\Downloads\amazon_reviews.csv")
        except:
            st.error("❌ File not found! Please upload 'amazon_reviews.csv' to the same folder.")
            st.stop()
            
    df = df.dropna(subset=['reviewText'])
    # Logic: 4-5 stars = Positive, 1-3 stars = Negative
    df['sentiment'] = df['overall'].apply(lambda x: 'positive' if x > 3 else 'negative')
    return df

df = load_data()

# --- TRAIN MODEL (Improved) ---
# We use ngram_range=(1,2) so it understands "not good" vs "good"
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
X_vec = tfidf.fit_transform(df['reviewText'])
y = df['sentiment']

model = MultinomialNB()
model.fit(X_vec, y)

# Show data stats in sidebar
st.sidebar.write(f"Training on {len(df)} reviews")
st.sidebar.write(f"Positive: {len(df[df['sentiment']=='positive'])}")
st.sidebar.write(f"Negative: {len(df[df['sentiment']=='negative'])}")

# --- USER INTERFACE ---
# No default text, just a placeholder
user_input = st.text_area("Type a review here:", placeholder="Example: The battery life is terrible...")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        # Transform input
        input_vec = tfidf.transform([user_input])
        
        # Get prediction and PROBABILITY (Confidence)
        prediction = model.predict(input_vec)[0]
        probs = model.predict_proba(input_vec)[0] # [Prob_Negative, Prob_Positive]
        
        # 'classes_' usually sorts alphabetically: ['negative', 'positive']
        # So probs[0] is negative, probs[1] is positive
        conf_neg = probs[0]
        conf_pos = probs[1]

        # Display Result
        if prediction == 'positive':
            st.success(f"😊 **Positive** (Confidence: {conf_pos*100:.1f}%)")
            st.progress(conf_pos)
        else:
            st.error(f"😡 **Negative** (Confidence: {conf_neg*100:.1f}%)")
            st.progress(conf_neg)
            
    else:
        st.warning("⚠️ Please type a review first!")
