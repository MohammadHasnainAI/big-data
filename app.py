
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- PAGE CONFIG ---
st.set_page_config(page_title="Review Analyzer", page_icon="🛒")

# --- TITLE ---
st.title("Amazon Review Sentiment Analysis 🛒")
st.write("Enter a product review to see if it's Positive or Negative!")

# --- LOAD DATA FUNCTION ---
@st.cache_data
def load_data():
    # 1. Try loading from the current folder (Works on Streamlit Cloud)
    try:
        df = pd.read_csv("amazon_reviews.csv")
    except FileNotFoundError:
        # 2. If not found, try the local path (Works on your Laptop)
        # We use raw string r"..." for Windows paths
        try:
            df = pd.read_csv(r"C:\Users\hasna\Downloads\amazon_reviews.csv")
        except FileNotFoundError:
            # 3. If still not found, stop the app and show a clear error
            st.error("❌ File not found! Please make sure 'amazon_reviews.csv' is in the same folder as this app.")
            st.stop()
            
    # --- DATA CLEANING ---
    # Drop rows where the review text is missing
    df = df.dropna(subset=['reviewText'])
    
    # Create 'sentiment' column: 4-5 stars = positive, 1-3 = negative
    df['sentiment'] = df['overall'].apply(lambda x: 'positive' if x > 3 else 'negative')
    
    return df

# Load the data
df = load_data()

# --- TRAIN MODEL ---
# We train the model fresh every time the app loads
tfidf = TfidfVectorizer(stop_words='english')
X_vec = tfidf.fit_transform(df['reviewText'])
y = df['sentiment']

model = MultinomialNB()
model.fit(X_vec, y)

# --- USER INTERFACE ---
user_input = st.text_area("Type a review here:", placeholder="E.g., I loved this product, it works great!")

if st.button("Analyze Sentiment"):
    if user_input:
        # Convert user text to numbers
        input_vec = tfidf.transform([user_input])
        # Predict
        prediction = model.predict(input_vec)[0]
        
        # Show result with color
        if prediction == 'positive':
            st.success("😊 This review is **Positive**!")
            st.balloons()
        else:
            st.error("😡 This review is **Negative**!")
    else:
        st.warning("Please type something first.")

# --- SIDEBAR INFO ---
st.sidebar.title("About")
st.sidebar.info("This project uses Naive Bayes and TF-IDF to classify reviews as Positive or Negative.")
