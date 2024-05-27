# utils.py

import pandas as pd

def load_saved_patient_data():
    if os.path.exists("saved_data.csv"):
        return pd.read_csv("saved_data.csv")
    else:
        return pd.DataFrame(columns=["Patient ID", "Age", "PreopEGFRMDRD", "Intraop", "ASACategoryBinned", "AnemiaCategoryBinned", "RDW15.7", "SurgicalRiskCategory", "AnesthesiaTypeCategory", "GradeofKidneyDisease", "PriorityCategory", "ICU Probability", "Mortality Probability", "ICU Admission >24 hours", "Mortality"])

def save_patient_data(data):
    df = load_saved_patient_data()
    df = df.append(data, ignore_index=True)
    df.to_csv("saved_data.csv", index=False)

def append_to_csv(data, csv_file):
    df = pd.DataFrame([data])
    df.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))
