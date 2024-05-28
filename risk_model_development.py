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

    st.markdown("""
        <style>
        .stButton button {
            background-color: #6eb52f;
            color: white;
            margin-bottom: 10px;
            border-radius: 5px;
            border: none;
            padding: 10px;
            font-size: 16px;
            text-align: center;
        }
        .stButton button:hover {
            background-color: #5ca024;
        }
        .stButton button:active {
            background-color: #4b8520;
        }
        </style>
        """, unsafe_allow_html=True)
    
    roc_button = st.sidebar.button('ROC Curve Comparisons', key='roc_button')
    cm_button = st.sidebar.button('Confusion Matrix Comparisons', key='cm_button')

    if roc_button:
        st.markdown("### ROC Curve Comparison: ICU vs. Mortality")
        st.image("static/ICU ROC Curve.png", caption="ICU ROC Curve")
        st.image("static/Mortality ROC Curve.png", caption="Mortality ROC Curve")

    if cm_button:
        st.image("static/ICU Confusion Matrix.png", caption="ICU Confusion Matrix")
        st.image("static/Mortality Confusion Matrix.png", caption="Mortality Confusion Matrix")
        st.markdown("### Confusion Matrix Comparison: ICU vs. Mortality")
