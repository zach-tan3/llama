import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build

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

def write_cells(spreadsheet_id, update_data):
    # Function to execute batchUpdate request
    updating = sheet.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id, body=update_data)
    updating.execute()

def delete_patient_data(patient_id):
    # Get the instance of the Spreadsheet
    sheet = client.open('saved_patient_data')
    # Get the first sheet of the Spreadsheet
    sheet_instance = sheet.get_worksheet(0)
    # Find the row with the given patient_id
    cell = sheet_instance.find(str(patient_id))
    if cell:
        row = cell.row
        # Prepare the batch update request body
        body = {
            "requests": [
                {
                    "deleteDimension": {
                        "range": {
                            "sheetId": sheet_instance.id,
                            "dimension": "ROWS",
                            "startIndex": row - 1,
                            "endIndex": row
                        }
                    }
                }
            ]
        }
        # Execute the batch update request
        write_cells(sheet_instance, body)
    return load_saved_patient_data()
