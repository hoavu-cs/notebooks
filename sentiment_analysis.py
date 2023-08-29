import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from xgboost import XGBClassifier, XGBRegressor
import warnings
import random
import matplotlib.pyplot as plt
import joblib


model = XGBClassifier()
model.load_model('sentiment_analysis.model')
vectorizer = joblib.load('vectorizer.joblib')

while True:

    user_input = input("Please enter a text (type exit to stop): ")
    text = str(user_input)

    if text == 'exit':
        break

    prediction = model.predict(vectorizer.transform([text]))

    if prediction == 0:
        print(f'Negative. Confidence: {model.predict_proba(vectorizer.transform([text]))[0][0]}')
    elif prediction == 1:
        print(f'Positive. Confidence: {model.predict_proba(vectorizer.transform([text]))[0][1]}')
    

