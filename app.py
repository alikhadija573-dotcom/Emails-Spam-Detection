import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ====== CHECK IF MODEL FILES EXIST ======
if not os.path.exists("vectorizer.pkl") or not os.path.exists("model.pkl"):
    st.error("❌ Model files not found!")
    st.info("👉 Please run 'train_model.py' first to generate vectorizer.pkl and model.pkl")
    st.stop()  # stop execution if files missing

# Load saved model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ====== STREAMLIT UI ======
st.title("📧 Email Classifier")

html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit Email Spam Detector ML App</h2>
</div>
<br>
"""
st.markdown(html_temp, unsafe_allow_html=True)

input_sms = st.text_area("✉️ Enter the message")

if st.button('🔍 Analyze'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])

        prob = model.predict_proba(vector_input)[0]
        spam_prob = prob[1] * 100
        ham_prob = prob[0] * 100

        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam Message Detected")
        else:
            st.success("✅ Safe Message (Not Spam)")

        st.write(f"📊 Spam Probability: {spam_prob:.2f}%")
        st.write(f"📊 Not Spam Probability: {ham_prob:.2f}%")

# Footer
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
st.write("© 2026 Khadija Ali | Made With ❤️ in Pakistan 🇵🇰")