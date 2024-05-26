# utils.py

import pandas as pd

def load_saved_patient_data():
    if os.path.exists("saved_patient_data.csv"):
        return pd.read_csv("saved_patient_data.csv")
    else:
        return pd.DataFrame(columns=["Patient ID", "Age", "PreopEGFRMDRD", "Intraop", "ASACategoryBinned", "AnemiaCategoryBinned", "RDW15.7", "SurgicalRiskCategory", "AnesthesiaTypeCategory", "GradeofKidneyDisease", "PriorityCategory", "ICU > 24h", "Mortality"])

def save_patient_data(patient_data):
    df = load_saved_patient_data()
    df = df.append(patient_data, ignore_index=True)
    df.to_csv("saved_patient_data.csv", index=False)

def append_to_csv(filename, data):
    df = pd.read_csv(filename)
    df = df.append(data, ignore_index=True)
    df.to_csv(filename, index=False)

