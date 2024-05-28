# risk_model_development.py
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
        body {
            font-family: "sans serif";
            background-color: #f0f0f5;
        }
        .stButton button {
            background-color: #6eb52f;
            color: white;
            margin-bottom: 10px;
            border-radius: 5px;
            border: none;
            padding: 10px;
            text-align: left;
            display: flex;
            align-items: center;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #5ca024;
        }
        .stButton button:active {
            background-color: #4b8520;
        }
        .stSidebar {
            background-color: #e0e0ef;
        }
        .stSidebar .stButton button {
            background-color: transparent;
            color: black;
            text-align: left;
            padding: 10px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
        }
        .stSidebar .stButton button:hover {
            background-color: #dcdcdc;
        }
        .stSidebar .stButton button:active {
            background-color: #bcbcbc;
        }
        .stSidebar .stSelectbox, .stSidebar .stSlider {
            margin-bottom: 20px;
        }
        .stChatMessage {
            margin-bottom: 20px;
        }
        .header-container {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        .header-container img {
            width: 200px;
            margin-right: 20px;
        }
        .vertical-line {
            border-left: 2px solid #6eb52f;
            height: 80px;
            margin-right: 20px;
        }
        .logo-text {
            font-weight: 700;
            font-size: 40px;
            color: #000000;
            padding-top: 18px;
        }
        </style>
        """, unsafe_allow_html=True)



    model1_button = st.sidebar.button('Model 1', key='model1_button')
    model2_button = st.sidebar.button('Model 2', key='model2_button')
    model3_button = st.sidebar.button('Model 3', key='model3_button')

    if model1_button:
        st.image("static/model1.png")
    if model2_button:
        st.image("static/model2.png")
    if model3_button:
        st.image("static/model3.png")
