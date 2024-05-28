import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from utils import load_saved_patient_data, save_patient_data, append_to_csv

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
        if not data.empty and patient_id in data["Patient ID"].values:
            data.loc[data['Patient ID'] == patient_id, 'ICU Admission >24 hours'] = icu_status
            data.loc[data['Patient ID'] == patient_id, 'Mortality'] = mortality_status
            data.to_csv("saved_data.csv", index=False)
            st.write("Patient data updated successfully.")
        else:
            st.write("Patient ID not found in saved data.")
