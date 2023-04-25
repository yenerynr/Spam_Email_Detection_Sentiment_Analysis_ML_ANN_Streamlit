import pandas as pd
import streamlit as st
import keras
#from tensorflow import keras
from PIL import Image
from keras.models import Sequential
#from keras_preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences

##loading the ann model
model = keras.models.load_model("ann_model/ann_model")

## load the copy of the dataset
df = pd.read_csv("emails.csv")

## set page configuration
st.set_page_config(page_title = 'Email Classifier', layout = 'wide')

## add page title and content
st.title('Email Classifier using Artificial Nerual Network')
st.write('Please Enter an email to be classified:')

## add image
image = Image.open("spam.jpg")
st.image(image, use_column_width=True)

## get user input
email_text = st.text_input('Email Text:')

## convert text to numerical values
word_index = {word: index for index, word in enumerate(df.columns[:-1])}
numerical_email = [word_index[word] for word in email_text.lower().split() if word in word_index]

## pad the numerical emails so that it can have a unique shape as the training data 
padded_email = pad_sequences([numerical_email],maxlen=3000)

## make the prediction
if st.button('Predict'):
    prediction = model.predict(padded_email)

    if prediction > 0.5:
        st.write('This email is spam')
    else: 
        st.write('This email is not spam')

    ##    python -m streamlit run app_ann.py
    ##    cd C:\Users\yener\OneDrive\Desktop\neuralnetworks
