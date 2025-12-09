
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- TITLE ---
st.title("Amazon Review Sentiment Analysis 🛒")
st.write("Enter a product review to see if it's Positive or Negative!")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Try loading from the same folder (for Web/Streamlit Cloud)
        df = pd.read_csv("amazon_reviews.csv")
    except:
        # Fallback to Downloads folder (for Local Laptop)
        # NOTE: If you are running this locally and the file is in Downloads, 
        # ensure this path matches your actual username path.
        df = pd.read_csv(r"C:\Users\hasna\Downloads\amazon_reviews.csv")
        
    df = df.dropna(subset=['reviewText'])
    df['sentiment'] = df['overall'].apply(lambda x: 'positive' if x > 3 else 'negative')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- 2. TRAIN MODEL ---
tfidf = TfidfVectorizer(stop_words='english')
X_vec = tfidf.fit_transform(df['reviewText'])
y = df['sentiment']

model = MultinomialNB()
model.fit(X_vec, y)

# --- 3. USER INTERFACE ---
user_input = st.text_area("Type a review here:", "The product is amazing!")

if st.button("Analyze Sentiment"):
    if user_input:
        input_vec = tfidf.transform([user_input])
        prediction = model.predict(input_vec)[0]
        
        if prediction == 'positive':
            st.success("😊 This review is **Positive**!")
            st.balloons()
        else:
            st.error("😡 This review is **Negative**!")
    else:
        st.warning("Please type something first.")
