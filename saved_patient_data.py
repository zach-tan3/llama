# saved_patient_data.py

import streamlit as st
import pandas as pd

# Function to load saved patient data
def load_saved_patient_data():
    if os.path.exists("saved_patient_data.csv"):
        return pd.read_csv("saved_patient_data.csv")
    else:
        return pd.DataFrame(columns=["Patient ID", "Age", "PreopEGFRMDRD", "Intraop", "ASACategoryBinned", "AnemiaCategoryBinned", "RDW15.7", "SurgicalRiskCategory", "AnesthesiaTypeCategory", "GradeofKidneyDisease", "PriorityCategory", "ICU > 24h", "Mortality"])

# Function to save patient data
def save_patient_data(patient_data):
    df = load_saved_patient_data()
    df = df.append(patient_data, ignore_index=True)
    df.to_csv("saved_patient_data.csv", index=False)

def saved_patient_data_page():
    st.title("Saved Patient Data")
    df = load_saved_patient_data()
    
    st.write(df)
    
    st.sidebar.header("Update Patient Data")
    patient_id = st.sidebar.selectbox("Select Patient ID", df["Patient ID"].unique())
    
    icu_gt_24h = st.sidebar.selectbox("ICU Admission > 24 hours", ["Yes", "No"])
    mortality = st.sidebar.selectbox("Mortality", ["Yes", "No"])
    
    if st.sidebar.button("Save"):
        df.loc[df["Patient ID"] == patient_id, ["ICU > 24h", "Mortality"]] = [icu_gt_24h, mortality]
        df.to_csv("saved_patient_data.csv", index=False)
        st.sidebar.success("Patient data updated successfully")

# Utility function to append data to CSV files
def append_to_csv(filename, data):
    df = pd.read_csv(filename)
    df = df.append(data, ignore_index=True)
    df.to_csv(filename, index=False)

