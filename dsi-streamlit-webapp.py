#DSI Streamlit Web App

#import libraires
import streamlit as st
import pandas as pd
import joblib

#load model pipeline object
model = joblib.load('model.joblib')

#lots of functionality available within streamlit, check the documentation
#for a full run down!

#add title and instructions

st.title("Purchasing Prediction Model")
st.subheader("Input customer information and click for a likelihood of purchase")

#age input form
age = st.number_input(
    label = "Enter Customer Age.",
    min_value = 18,
    max_value = 120,
    value = 35)

#gender input form
gender = st.radio(
    label = "Enter Customer Gender.",
    options = ["M","F"])

#credit score input form
credit_score = st.number_input(
    label = "Enter Customer Credit Score",
    min_value = 0,
    max_value = 1000,
    value = 500)

#submit inputs to model
if st.button("Submit for Prediction"):
    
    #collate inputs into a dataframe for predictions
    new_data = pd.DataFrame({'age':[age],
                             'gender':[gender],
                             'credit_score':[credit_score]})
    
    #apply model pipeline to input, extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    #output prediction
    st.subheader(f'Based on attributes entered, our model predicts a {pred_proba:.0%} probability of a purchase!')
