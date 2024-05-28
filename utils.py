import pandas as pd
import os

def save_patient_data(data):
    print("Saving patient data:", data)
    st.sidebar.write("Saving patient data")
    df = load_saved_patient_data()
    df = df.append(data, ignore_index=True)
    df.to_csv("saved_data.csv", index=False)
    print("Data saved successfully.")

def append_to_csv(data, csv_file):
    print("Appending data to CSV:", data)
    df = pd.DataFrame([data])
    df.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))
    print("Data appended successfully.")

def load_saved_patient_data():
    print("Loading saved patient data...")
    if os.path.exists("saved_data.csv"):
        try:
            data = pd.read_csv("saved_data.csv")
            print("Data loaded successfully.")
            return data
        except pd.errors.EmptyDataError:
            print("Empty data error encountered. Returning empty dataframe.")
            return pd.DataFrame(columns=["Patient ID", "Age", "PreopEGFRMDRD", "Intraop", "ASACategoryBinned", "AnemiaCategoryBinned", "RDW15.7", "SurgicalRiskCategory", "AnesthesiaTypeCategory", "GradeofKidneyDisease", "PriorityCategory", "ICU Probability", "Mortality Probability", "ICU Admission >24 hours", "Mortality"])
    else:
        print("File not found. Returning empty dataframe.")
        return pd.DataFrame(columns=["Patient ID", "Age", "PreopEGFRMDRD", "Intraop", "ASACategoryBinned", "AnemiaCategoryBinned", "RDW15.7", "SurgicalRiskCategory", "AnesthesiaTypeCategory", "GradeofKidneyDisease", "PriorityCategory", "ICU Probability", "Mortality Probability", "ICU Admission >24 hours", "Mortality"])
