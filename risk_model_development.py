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

    if model1_button:
        st.image("static/model1.png")
    if model2_button:
        st.image("static/model2.png")
    if model3_button:
        st.image("static/model3.png")
