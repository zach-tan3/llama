import streamlit as st
import replicate
import os
import pandas as pd
import torch
import numpy as np
from io import BytesIO
import torch.nn as nn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import openai
import base64
from dotenv import load_dotenv

from utils import save_patient_data, append_to_csv

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "This is a risk calculator for need for admission into an Intensive Care Unit (ICU) of a patient post-surgery and for Mortality. Ask me anything."}]

# Function to handle saving patient data
def handle_save_patient_data():
    with st.sidebar.form(key='patient_id_form'):
        patient_id = st.text_input("Enter Patient ID (type 'exit' to cancel):")
        submit_button = st.form_submit_button("Submit ID")

        if submit_button:
            if patient_id.lower() == 'exit':
                st.sidebar.write("Patient data not saved.")
            else:
                # Collect data from session state
                prediction_data = st.session_state.get('prediction_data', {})
                prediction_data["Patient ID"] = patient_id
                save_patient_data(prediction_data)
                st.sidebar.write("Patient data saved successfully.")
                # Reset saving state
                st.session_state.saving = False
'''def handle_save_patient_data():
    patient_id = st.sidebar.text_input("Enter Patient ID (type 'exit' to cancel):")
    if st.sidebar.button("Submit ID"):
        if patient_id.lower() == 'exit':
            st.write("Patient data not saved.")
        else:
            # Collect data from session state
            prediction_data = st.session_state.get('prediction_data', {})
            prediction_data["Patient ID"] = patient_id
            save_patient_data(prediction_data)
            st.write("Patient data saved successfully.")
            # Reset saving state
            st.session_state.saving = False'''
            
# Function for main risk calculator
def risk_calculator_page():
    # Title and description with logo
    LOGO_IMAGE = "static/ICURISK_Logo.png"
    st.markdown(
        f"""
        <div class="header-container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            <div class='vertical-line'></div>
            <p class="logo-text">Risk Calculator w/ ChatGPT! 🤖</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    openai.api_key = ''
    
    # Initialize session state for messages if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "This is a risk calculator for need for admission into an Intensive Care Unit (ICU) of a patient post-surgery and for Mortality. Ask me anything."}]
    
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # initialize model
    if "model" not in st.session_state:
        st.session_state.model = "gpt-3.5-turbo"
    
    # User input
    if user_prompt := st.chat_input("Your prompt"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
    
        # Generate responses
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
    
            try:
                response = openai.chat.completions.create(
                    model=st.session_state.model,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                )
    
                # Extract the content from the response
                full_response = response.choices[0].message.content
                message_placeholder.markdown(full_response)
    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()
    
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    
    st.session_state.last_prediction_probability = " "
    
    
    # Create an instance of the model
    if os.path.exists('icu_classifier.pkl') and os.path.exists('mortality_classifier.pkl'):
        icu_classifier = joblib.load('icu_classifier.pkl')
        mortality_classifier = joblib.load('mortality_classifier.pkl')
    else:
        st.error('Model files not found. Please ensure the files are uploaded.')
    
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "This is a risk calculator for need for of admission into an Intensive Care Unit (ICU) of a paitent post-surgery. Ask me anything"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    
    Age = st.sidebar.slider('Age', 18, 99, 40)
    PreopEGFRMDRD = st.sidebar.slider('PreopEGFRMDRD', 0, 160, 80)
    ASACategoryBinned = st.sidebar.selectbox('ASA Category Binned', ['i', 'ii', 'iii', 'iv-vi'])
    GradeofKidneyDisease = st.sidebar.selectbox('Grade of Kidney Disease', ['blank', 'g1', 'g2', 'g3a', 'g3b', 'g4', 'g5'])
    AnemiaCategoryBinned = st.sidebar.selectbox('Anemia Category Binned', ['none', 'mild', 'moderate/severe'])
    RDW157 = st.sidebar.selectbox('RDW15.7', ['<= 15.7', '>15.7'])
    SurgicalRiskCategory = st.sidebar.selectbox('SurgRisk', ['low', 'moderate', 'high'])
    Intraop = st.sidebar.slider('Intraop', 0, 1, 0)
    AnesthesiaTypeCategory = st.sidebar.selectbox('Anaestype', ['ga', 'ra'])
    PriorityCategory = st.sidebar.selectbox('Priority', ['elective', 'emergency'])
    
    prediction_prompt = {'Age': Age,
                         'PreopEGFRMDRD': PreopEGFRMDRD, 
                         'ASACategoryBinned': ASACategoryBinned,
                         'GradeofKidneyDisease': GradeofKidneyDisease,
                         'AnemiaCategoryBinned': AnemiaCategoryBinned, 
                         'RDW15.7': RDW157, 
                         'SurgicalRiskCategory': SurgicalRiskCategory, 
                         'Intraop': Intraop,
                         'AnaesthesiaTypeCategory': AnesthesiaTypeCategory, 
                         'PriorityCategory': PriorityCategory}
    
    if st.sidebar.button('Predict'):
        with st.chat_message("user"):
            st.write(prediction_prompt)
    
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Preprocess your input data
                input_data = pd.DataFrame({ 'Age': [Age],
                                            'PreopEGFRMDRD': [PreopEGFRMDRD],
                                            'ASACategoryBinned': [ASACategoryBinned],
                                            'GradeofKidneyDisease': [GradeofKidneyDisease],
                                            'AnemiaCategoryBinned': [AnemiaCategoryBinned],
                                            'RDW15.7': [RDW157],
                                            'SurgicalRiskCategory': [SurgicalRiskCategory],
                                            'Intraop': [Intraop],
                                            'AnaesthesiaTypeCategory': [AnesthesiaTypeCategory],
                                            'PriorityCategory': [PriorityCategory]})    
    
                # Mappings of categorical values
                #Age
                #PreopEGFRMDRD
                ASAcategorybinned_mapper = {"i":0, "ii":1, 'iii':2, 'iv-vi':3}
                GradeofKidneydisease_mapper = {"blank":0, "g1":1, "g2":2, "g3a":3,"g3b":4, "g4":5, "g5":6}
                Anemiacategorybinned_mapper = {"none": 0, "mild":1, "moderate/severe":2}
                RDW157_mapper = {"<= 15.7":0, ">15.7":1}
                SurgRiskCategory_mapper = {"low":0, "moderate":1, "high":2}
                anaestype_mapper = {"ga": 0, "ra": 1}
                priority_mapper = {"elective": 0, "emergency": 1}
                
                # Map categorical values
                input_data['ASACategoryBinned'] = input_data['ASACategoryBinned'].map(ASAcategorybinned_mapper)
                input_data['GradeofKidneyDisease'] = input_data['GradeofKidneyDisease'].map(GradeofKidneydisease_mapper)
                input_data['AnemiaCategoryBinned'] = input_data['AnemiaCategoryBinned'].map(Anemiacategorybinned_mapper)
                input_data['RDW15.7'] = input_data['RDW15.7'].map(RDW157_mapper)
                input_data['SurgicalRiskCategory'] = input_data['SurgicalRiskCategory'].map(SurgRiskCategory_mapper)
                input_data['AnaesthesiaTypeCategory'] = input_data['AnaesthesiaTypeCategory'].map(anaestype_mapper)
                input_data['PriorityCategory'] = input_data['PriorityCategory'].map(priority_mapper)
    
                # Convert to PyTorch tensor
                input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
                
                # Generate prediction probabilities
                icu_probability = icu_classifier.predict_proba(input_tensor)[:, 1].item() * 100
                mortality_probability = mortality_classifier.predict_proba(input_tensor)[:, 1].item() * 100
                
                # Display prediction probabilities
                #st.write(f"ICU Predicted probability: {icu_probability:.2f}%")
                #st.write(f"Mortality Predicted probability: {mortality_probability:.2f}%")
                
                # Save prediction probability
                st.session_state.last_icu_prediction_probability = f"ICU Predicted probability: {icu_probability:.2f}%"
                st.session_state.last_mortality_prediction_probability = f"Mortality Predicted probability: {mortality_probability:.2f}%"
                
                # Display prediction
                st.write(st.session_state.last_icu_prediction_probability)
                st.write(st.session_state.last_mortality_prediction_probability)
    
                message = {"role": "assistant", "content": st.session_state.last_icu_prediction_probability}
                st.session_state.messages.append(message)
                message = {"role": "assistant", "content": st.session_state.last_mortality_prediction_probability}
                st.session_state.messages.append(message)

                # Save the prediction data to session state
                st.session_state.prediction_data = {
                    "Age": Age,
                    "PreopEGFRMDRD": PreopEGFRMDRD,
                    "Intraop": Intraop,
                    "ASACategoryBinned": ASACategoryBinned,
                    "AnemiaCategoryBinned": AnemiaCategoryBinned,
                    "RDW15.7": RDW157,
                    "SurgicalRiskCategory": SurgicalRiskCategory,
                    "AnesthesiaTypeCategory": AnesthesiaTypeCategory,
                    "GradeofKidneyDisease": GradeofKidneyDisease,
                    "PriorityCategory": PriorityCategory,
                    "ICU Admission>24 hours": '',
                    "Mortality": ''
                }
            st.sidebar.button('Save Patient Data', on_click=handle_save_patient_data)
