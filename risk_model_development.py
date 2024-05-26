# risk_model_development.py

import streamlit as st

def risk_model_development_page():
    st.title("Risk Model Development")
    st.write("This is the Risk Model Development page.")
    model1_button = st.sidebar.button('Model 1')
    model2_button = st.sidebar.button('Model 2')
    model3_button = st.sidebar.button('Model 3')

    if model1_button:
        st.image("static/model1.png")
    if model2_button:
        st.image("static/model2.png")
    if model3_button:
        st.image("static/model3.png")

