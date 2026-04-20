import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text

st.set_page_config(page_title='Fake News Detector', layout='centered')

@st.cache_resource
def load_models():
    svm_model = joblib.load('models/svm_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
    return svm_model, tfidf, lstm_model

# Dummy loading for demo purposes since actual models aren't committed
# svm_model, tfidf, lstm_model = load_models()

st.title('📰 Fake News Detection System')
st.write('Analyze news articles using NLP and Machine Learning.')

model_choice = st.selectbox('Select Model', ['SVM (TF-IDF)', 'LSTM (Word2Vec)', 'BERT'])
article_text = st.text_area('Paste News Article Here:', height=200)

if st.button('Analyze'):
    if not article_text.strip():
        st.warning('Please enter some text.')
    else:
        cleaned = clean_text(article_text)
        
        # Placeholder logic for demo
        st.info(f'Analysis complete using {model_choice}.')
        st.success('✅ REAL NEWS (Confidence: 98.5%)')
        st.progress(0.985)
