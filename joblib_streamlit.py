# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 14:20:05 2023

@author: lidya
"""

import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("trained_model_gold_fixed.sav")

# Define the function to make predictions
def predict_output(spx, uso, slv, usd):
    input_data = pd.DataFrame({
        'SPX': [spx],
        'USO': [uso],
        'SLV': [slv],
        'USD': [usd]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title("Regressor Model Prediction App")

    # User input
    spx = st.number_input("Enter SPX value", min_value=0.0)
    uso = st.number_input("Enter USO value", min_value=0.0)
    slv = st.number_input("Enter SLV value", min_value=0.0)
    usd = st.number_input("Enter USD value", min_value=0.0)

    # Predict button
    if st.button("Predict"):
        prediction = predict_output(spx, uso, slv, usd)
        st.success(f"The predicted output is: {prediction:.2f}")

if __name__ == "__main__":
    main()
