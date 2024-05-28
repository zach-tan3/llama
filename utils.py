import pandas as pd
import os

def save_patient_data(data):
    print("Saving patient data:", data)
    df = load_saved_patient_data()
    df = df.append(data, ignore_index=True)
    df.to_csv("saved_data.csv", index=False)
    print("Data saved successfully.")

def append_to_csv(data, csv_file):
    print("Appending data to CSV:", data)
    df = pd.DataFrame([data])
    df.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))
    print("Data appended successfully.")
