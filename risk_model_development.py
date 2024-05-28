# risk_model_development.py
import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from utils import set_bg

def risk_model_development_page():
    
    # Custom CSS for styling
    set_bg('static/Light blue background.jpg')
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
        }
        .stButton button:hover {
            background-color: #5ca024;
        }
        .stButton button:active {
            background-color: #4b8520;
        }
        .stButton button img {
            margin-right: 10px;
        }
        .stSidebar {
            background-color: #e0e0ef;
        }
        .stSidebar .stButton button {
            background-color: transparent;
            color: #ff4d4d;
            text-align: left;
            padding: 10px;
            border: none;
            border-radius: 5px;
        }
        .stSidebar .stButton button:hover {
            background-color: #ffe6e6;
        }
        .stSidebar .stButton button:active {
            background-color: #ffcccc;
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
        .icon-button {
            display: flex;
            align-items: center;
        }
        .icon-button img {
            margin-right: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Title and description with logo
    LOGO_IMAGE = "static/ICURISK_Logo.png"
    st.markdown(
        f"""
        <div class="header-container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            <div class='vertical-line'></div>
            <p class="logo-text">Risk Model Development 📊</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    model1_button = st.sidebar.markdown(
        """
        <button class="icon-button">
            <img src="data:image/png;base64,{}" width="20"/>
            <span>Model 1</span>
        </button>
        """.format(base64.b64encode(open("static/model1_icon.png", "rb").read()).decode()),
        unsafe_allow_html=True
    )
    model2_button = st.sidebar.markdown(
        """
        <button class="icon-button">
            <img src="data:image/png;base64,{}" width="20"/>
            <span>Model 2</span>
        </button>
        """.format(base64.b64encode(open("static/model2_icon.png", "rb").read()).decode()),
        unsafe_allow_html=True
    )
    model3_button = st.sidebar.markdown(
        """
        <button class="icon-button">
            <img src="data:image/png;base64,{}" width="20"/>
            <span>Model 3</span>
        </button>
        """.format(base64.b64encode(open("static/model3_icon.png", "rb").read()).decode()),
        unsafe_allow_html=True
    )

    if st.sidebar.button('Model 1'):
        st.image("static/model1.png")
    if st.sidebar.button('Model 2'):
        st.image("static/model2.png")
    if st.sidebar.button('Model 3'):
        st.image("static/model3.png")
