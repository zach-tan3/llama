import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from utils import load_saved_patient_data, update_patient_data, delete_patient_data, set_bg, logo2, CSS_styling

def saved_patient_data_page():
    
    # Custom CSS for styling
    set_bg('static/Light blue background.jpg')
    logo2('static/ICURISK_Logo.png')
    CSS_styling()
    
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
