# main website launch

import streamlit as st
import replicate
import os
import pandas as pd
import torch
import numpy as np
from io import BytesIO
import torch.nn as nn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import openai
import base64
from dotenv import load_dotenv
from risk_calculator.py import risk_calculator_page
from saved_patient_data.py import saved_patient_data_page
from risk_model_development.py import risk_model_development_page
#from utils import load_saved_patient_data, save_patient_data, append_to_csv

# Set Streamlit configuration
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: "sans serif";
        background-color: #f0f0f5;
    }
    .stButton button {
        background-color: #6eb52f;
        color: white;
    }
    .stSidebar {
        background-color: #e0e0ef;
    }
    .stSidebar .stButton button {
        background-color: #6eb52f;
        color: white;
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

# Sidebar navigation dropdown
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to", ["Risk Calculator w/ ChatGPT", "Saved Patient Data", "Risk Model Development"])

if page == "Risk Calculator w/ ChatGPT":
    risk_calculator_page()
elif page == "Saved Patient Data":
    saved_patient_data_page()
elif page == "Risk Model Development":
    risk_model_development_page()
