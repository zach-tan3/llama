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
from risk_calculator import risk_calculator_page
from saved_patient_data import saved_patient_data_page
from risk_model_development import risk_model_development_page
from utils import CSS_styling

# Set Streamlit configuration
st.set_page_config(layout="wide")

# Custom CSS for styling
CSS_styling()

# Sidebar navigation dropdown
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to", ["Risk Calculator w/ ChatGPT", "Saved Patient Data", "Risk Model Development"])

if page == "Risk Calculator w/ ChatGPT":
    risk_calculator_page()
elif page == "Saved Patient Data":
    saved_patient_data_page()
elif page == "Risk Model Development":
    risk_model_development_page()
