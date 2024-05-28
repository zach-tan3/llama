import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from utils import load_saved_patient_data, update_patient_data

def saved_patient_data_page():
    
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

    # Title and description with logo
    LOGO_IMAGE = "static/ICURISK_Logo.png"
    st.markdown(
        f"""
        <div class="header-container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            <div class='vertical-line'></div>
            <p class="logo-text">Saved Patient Data 🗂️</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load saved data
    data = load_saved_patient_data()

    if data.empty:
        st.write("No saved patient data found. The table will appear here when you save patient data.")
    else:
        st.dataframe(data)
    
    # Allow updating ICU and Mortality status
    st.sidebar.header("Update Patient Data")
    patient_id = st.sidebar.text_input("Patient ID to Update")
    icu_status = st.sidebar.selectbox("ICU Admission >24 hours", ["Unknown", "Yes", "No"])
    mortality_status = st.sidebar.selectbox("Mortality", ["Unknown", "Yes", "No"])
    
    if st.sidebar.button("Update Status"):
        data = update_patient_data(patient_id, icu_status, mortality_status)
        if str(patient_id) in data["Patient ID"].astype(str).values:
            st.sidebar.write("Patient data updated successfully.")
        else:
            st.sidebar.write("Patient ID not found in saved data.")

