# saved_patient_data.py

import streamlit as st
import pandas as pd
import numpy as np
import base64
def saved_patient_data_page():
    
    # Load saved data
    data = pd.read_csv("saved_patient_data.csv")
    
    st.dataframe(data)
    
    # Allow updating ICU and Mortality status
    st.sidebar.header("Update Patient Data")
    patient_id = st.sidebar.text_input("Patient ID to Update")
    icu_status = st.sidebar.selectbox("ICU Admission >24 hours", ["Yes", "No"])
    mortality_status = st.sidebar.selectbox("Mortality", ["Yes", "No"])
    
    if st.sidebar.button("Update Status"):
        data.loc[data['Patient ID'] == patient_id, 'ICU Admission >24 hours'] = icu_status
        data.loc[data['Patient ID'] == patient_id, 'Mortality'] = mortality_status
        data.to_csv("saved_patient_data.csv", index=False)
        st.write("Patient data updated successfully.")
