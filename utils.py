import pandas as pd
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Define the scope
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

# Add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_name('dscp-saved-patient-data-424710-94efe1a6f49f.json', scope)

# Authorize the clientsheet 
client = gspread.authorize(creds)

def save_patient_data(data):
    st.sidebar.write("Saving patient data...")
    df = load_saved_patient_data()
    df = df.append(data, ignore_index=True)
    df.to_csv("saved_data.csv", index=False)
    print("Data saved successfully.")

def append_to_csv(data, csv_file):
    st.sidebar.write("Appending data...")
    df = pd.DataFrame([data])
    df.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))
    print("Data appended successfully.")

def load_saved_patient_data():
    # get the instance of the Spreadsheet
    sheet = client.open('saved_patient_data')
    # get the first sheet of the Spreadsheet
    sheet_instance = sheet.get_worksheet(0)
    # get all the records of the data
    records_data = sheet_instance.get_all_records()
    # convert the json to dataframe
    records_df = pd.DataFrame.from_dict(records_data)
    return records_df
