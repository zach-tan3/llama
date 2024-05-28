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
            margin-bottom: 20px;
            border-radius: 10px;
            border: none;
            padding: 15px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            width: 100%;
        }
        .stButton button:hover {
            background-color: #5ca024;
        }
        .stButton button:active {
            background-color: #4b8520;
        }
        .image-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .image-container img {
            width: 48%;
            height: auto;
        }
        </style>
        """, unsafe_allow_html=True)
    
    roc_button = st.sidebar.button('ROC Curve Comparisons', key='roc_button')
    cm_button = st.sidebar.button('Confusion Matrix Comparisons', key='cm_button')

    if roc_button:
        st.markdown("### ROC Curve Comparison: ICU vs. Mortality")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("static/ICU ROC Curve.png", caption="ICU ROC Curve")
        st.image("static/Mortality ROC Curve.png", caption="Mortality ROC Curve")
        st.markdown('</div>', unsafe_allow_html=True)

    if cm_button:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("static/ICU Confusion Matrix.png", caption="ICU Confusion Matrix")
        st.image("static/Mortality Confusion Matrix.png", caption="Mortality Confusion Matrix")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("### Confusion Matrix Comparison: ICU vs. Mortality")
