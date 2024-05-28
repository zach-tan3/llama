import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from utils import load_saved_patient_data, update_patient_data, delete_patient_data, set_bg, logo2

def saved_patient_data_page():
    
    # Custom CSS for styling
    set_bg('static/Light blue background.jpg')
    logo2('static/ICURISK_Logo.png')
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
        .stButton.red-button button {
            background-color: #ff4d4d;
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
    
    # Load saved data
    data = load_saved_patient_data()
    
    table_placeholder = st.empty()

    if data.empty:
        table_placeholder.write("No saved patient data found. The table will appear here when you save patient data.")
    else:
        table_placeholder.dataframe(data)
    
    # Allow updating ICU and Mortality status
    st.sidebar.header("Update Patient Data")
    patient_id = st.sidebar.text_input("Patient ID to Update")
    icu_status = st.sidebar.selectbox("ICU Admission >24 hours", ["Unknown", "Yes", "No"])
    mortality_status = st.sidebar.selectbox("Mortality", ["Unknown", "Yes", "No"])
    
    if st.sidebar.button("Update Status"):
        data = update_patient_data(patient_id, icu_status, mortality_status)
        if str(patient_id) in data["Patient ID"].astype(str).values:
            st.sidebar.write("Patient data updated successfully.")
            table_placeholder.dataframe(data)  # Update the table in the same placeholder
        else:
            st.sidebar.write("Patient ID not found in saved data.")
    
    # Allow deleting a row based on Patient ID
    st.sidebar.header("Delete Patient Data")
    delete_patient_id = st.sidebar.text_input("Patient ID to Delete")
    
    if st.sidebar.button("Delete Row", key="delete_button"):
        data, success = delete_patient_data(delete_patient_id)
        if success:
            st.sidebar.write("Patient data deleted successfully.")
        else:
            st.sidebar.write("Patient ID not found in saved data.")
        table_placeholder.dataframe(data)  # Update the table in the same placeholder
