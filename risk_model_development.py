import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from utils import set_bg, logo3, CSS_styling

def risk_model_development_page():
    
    # Custom CSS for styling
    set_bg('static/Light blue background.jpg')
    logo3('static/ICURISK_Logo.png')
    CSS_styling()
    
    model1_button = st.sidebar.button('Model 1', key='model1_button')
    model2_button = st.sidebar.button('Model 2', key='model2_button')
    model3_button = st.sidebar.button('Model 3', key='model3_button')
    model4_button = st.sidebar.button('Model 4', key='model4_button')

    if model1_button:
        st.image("static/ICU ROC Curve.png")
    if model2_button:
        st.image("static/ICU Confusion Matrix.png")
    if model3_button:
        st.image("static/Mortality ROC Curve.png")
    if model3_button:
        st.image("static/Mortality Confusion Matrix.png")
