import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from xgboost import XGBClassifier, XGBRegressor
import warnings
import random
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
import contractions
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')  # Download the WordNet data
lemmatizer = WordNetLemmatizer()


model = XGBClassifier()
model.load_model('sentiment_analysis_combined')
vectorizer = joblib.load('vectorizer_combined')

while True:

    user_input = input("Please enter a text (type exit to stop): ")
    text = str(user_input).lower()
    text = contractions.fix(text)
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = " ".join(lemmatized_tokens)
    text = lemmatized_text

    if text == 'exit':
        break

    # predict = model.predict(vectorizer.transform([text]), num_iteration=model.best_iteration, raw_score=True)
    # print(predict[0])

    prediction = model.predict(vectorizer.transform([text]))[0]

    #print(prediction)

    if prediction == 0:
        print(f'Negative. Confidence: {model.predict_proba(vectorizer.transform([text]))[0][0]}')
    elif prediction == 1:
        print(f'Neutral. Confidence: {model.predict_proba(vectorizer.transform([text]))[0][1]}')
    elif prediction == 2:
        print(f'Positive. Confidence: {model.predict_proba(vectorizer.transform([text]))[0][2]}')
    

