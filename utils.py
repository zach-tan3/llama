import pandas as pd
import gspread
import streamlit as st
import base64
from oauth2client.service_account import ServiceAccountCredentials

# Define the scope
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

# Add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_name('dscp-saved-patient-data-424710-94efe1a6f49f.json', scope)

# Authorize the clientsheet 
client = gspread.authorize(creds)

def load_saved_patient_data():
    # Get the instance of the Spreadsheet
    sheet = client.open('saved_patient_data')
    # Get the first sheet of the Spreadsheet
    sheet_instance = sheet.get_worksheet(0)
    # Get all the records of the data
    records_data = sheet_instance.get_all_records()
    # Convert the json to dataframe
    records_df = pd.DataFrame.from_dict(records_data)
    return records_df

def save_patient_data(data):
    # Get the instance of the Spreadsheet
    sheet = client.open('saved_patient_data')
    # Get the first sheet of the Spreadsheet
    sheet_instance = sheet.get_worksheet(0)
    # Append the new row to Google Sheets
    row = [data.get(col) for col in load_saved_patient_data().columns]
    sheet_instance.append_row(row)

def update_patient_data(patient_id, icu_status, mortality_status):
    # Get the instance of the Spreadsheet
    sheet = client.open('saved_patient_data')
    # Get the first sheet of the Spreadsheet
    sheet_instance = sheet.get_worksheet(0)
    # Find the row with the given patient_id
    cell = sheet_instance.find(str(patient_id))
    if cell:
        row = cell.row
        # Update the ICU Admission and Mortality status
        icu_col = sheet_instance.find("ICU Admission >24 hours").col
        mortality_col = sheet_instance.find("Mortality").col
        sheet_instance.update_cell(row, icu_col, icu_status)
        sheet_instance.update_cell(row, mortality_col, mortality_status)
    return load_saved_patient_data()

def delete_patient_data(patient_id):
    # Get the instance of the Spreadsheet
    sheet = client.open('saved_patient_data')
    # Get the first sheet of the Spreadsheet
    sheet_instance = sheet.get_worksheet(0)
    # Get all the records of the data
    records_data = sheet_instance.get_all_records()
    # Convert the json to dataframe
    records_df = pd.DataFrame.from_dict(records_data)
    # Check if the patient ID exists in the data
    if str(patient_id) in records_df["Patient ID"].astype(str).values:
        # Find the row with the given patient_id and delete it
        records_df = records_df[records_df['Patient ID'].astype(str) != str(patient_id)]
        # Clear the existing sheet
        sheet_instance.clear()
        # Set the headers
        sheet_instance.update([records_df.columns.values.tolist()] + records_df.values.tolist())
        return records_df, True
    else:
        return records_df, False

def set_bg(main_bg):
    # set bg name
    main_bg_ext = "jpg"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def logo1(image):
    st.markdown(
        f"""
        <div class="header-container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(image, "rb").read()).decode()}">
            <div class='vertical-line'></div>
            <p class="logo-text">Risk Calculator w/ ChatGPT! 🤖</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def logo2(image):
    st.markdown(
        f"""
        <div class="header-container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(image, "rb").read()).decode()}">
            <div class='vertical-line'></div>
            <p class="logo-text">Saved Patient Data 🗂️</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def logo3(image):
    st.markdown(
        f"""
        <div class="header-container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(image, "rb").read()).decode()}">
            <div class='vertical-line'></div>
            <p class="logo-text">Risk Model Development 📊</p>
        </div>
        """,
        unsafe_allow_html=True
    )
