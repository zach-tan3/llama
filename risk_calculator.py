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
from utils import save_patient_data, update_patient_data, load_saved_patient_data, set_bg, logo1, clear_chat_history, handle_save_patient_data
           
# Function for main risk calculator
def risk_calculator_page():
    # Title and description with logo
    set_bg('static/Light blue background.jpg')
    logo1('static/ICURISK_Logo.png')
    
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    '''
    # Verify if API key is set
    if not openai.api_key:
        st.error("OpenAI API key is missing! Please set the API key in the .env file.")
    else:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "This is a risk calculator for need for admission into an Intensive Care Unit (ICU) of a patient post-surgery and for Mortality. Ask me anything."}]
            )
            st.write(response)
        except openai.error.AuthenticationError:
            st.error("Invalid OpenAI API key! Please check your API key and try again.")
    '''
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initialize model
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
    
            for response in openai.ChatCompletion.create(
                model=st.session_state.model,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
    
            message_placeholder.markdown(full_response)
    
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    st.session_state.last_prediction_probability = " "
    
    # Create an instance of the model
    if os.path.exists('icu_classifier.pkl') and os.path.exists('mortality_classifier.pkl'):
        icu_classifier = joblib.load('icu_classifier.pkl')
        mortality_classifier = joblib.load('mortality_classifier.pkl')
    else:
        st.error('Model files not found. Please ensure the files are uploaded.')
    
    # Sidebar input elements
    st.sidebar.header("Input Parameters")

    Age = st.sidebar.slider('Age', 18, 99, 40)
    PreopEGFRMDRD = st.sidebar.slider('PreopEGFRMDRD', 0, 160, 80)
    Intraop = st.sidebar.slider('Intraop', 0, 1, 0)
    ASACategoryBinned = st.sidebar.selectbox('ASA Category Binned', ['i', 'ii', 'iii', 'iv-vi'])
    AnemiaCategoryBinned = st.sidebar.selectbox('Anemia Category Binned', ['None', 'Mild', 'Moderate/Severe'])
    RDW157 = st.sidebar.selectbox('RDW15.7', ['<= 15.7', '>15.7'])
    SurgicalRiskCategory = st.sidebar.selectbox('Surgical Risk', ['Low', 'Moderate', 'High'])
    AnesthesiaTypeCategory = st.sidebar.selectbox('Anesthesia Type', ['Ga', 'Ra'])
    GradeofKidneyDisease = st.sidebar.selectbox('Grade of Kidney Disease', ['Blank', 'G1', 'G2', 'G3a', 'G3b', 'G4', 'G5'])
    PriorityCategory = st.sidebar.selectbox('Priority', ['Elective', 'Emergency'])

    prediction_prompt = {'Age': Age,
                         'PreopEGFRMDRD': PreopEGFRMDRD, 
                         'Intraop': Intraop,
                         'ASACategoryBinned': ASACategoryBinned,
                         'AnemiaCategoryBinned': AnemiaCategoryBinned, 
                         'RDW15.7': RDW157, 
                         'SurgicalRiskCategory': SurgicalRiskCategory, 
                         'AnesthesiaTypeCategory': AnesthesiaTypeCategory, 
                         'GradeofKidneyDisease': GradeofKidneyDisease,
                         'PriorityCategory': PriorityCategory}

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    
    if st.sidebar.button('Predict'):
        with st.chat_message("user"):
            st.write(prediction_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Preprocess your input data
                input_data = pd.DataFrame({ 'Age': [Age],
                                            'PreopEGFRMDRD': [PreopEGFRMDRD],
                                            'Intraop': [Intraop],
                                            'ASACategoryBinned': [ASACategoryBinned],
                                            'AnemiaCategoryBinned': [AnemiaCategoryBinned],
                                            'RDW15.7': [RDW157],
                                            'SurgicalRiskCategory': [SurgicalRiskCategory],
                                            'AnesthesiaTypeCategory': [AnesthesiaTypeCategory],
                                            'GradeofKidneyDisease': [GradeofKidneyDisease],
                                            'PriorityCategory': [PriorityCategory]})    

                # Mappings of categorical values
                ASAcategorybinned_mapper = {"i": 0, "ii": 1, 'iii': 2, 'iv-vi': 3}
                GradeofKidneydisease_mapper = {"Blank": 0, "G1": 1, "G2": 2, "G3a": 3, "G3b": 4, "G4": 5, "G5": 6}
                Anemiacategorybinned_mapper = {"None": 0, "Mild": 1, "Moderate/Severe": 2}
                RDW157_mapper = {"<= 15.7": 0, ">15.7": 1}
                SurgRiskCategory_mapper = {"Low": 0, "Moderate": 1, "High": 2}
                anaestype_mapper = {"Ga": 0, "Ra": 1}
                priority_mapper = {"Elective": 0, "Emergency": 1}
                
                # Map categorical values
                input_data['ASACategoryBinned'] = input_data['ASACategoryBinned'].map(ASAcategorybinned_mapper)
                input_data['GradeofKidneyDisease'] = input_data['GradeofKidneyDisease'].map(GradeofKidneydisease_mapper)
                input_data['AnemiaCategoryBinned'] = input_data['AnemiaCategoryBinned'].map(Anemiacategorybinned_mapper)
                input_data['RDW15.7'] = input_data['RDW15.7'].map(RDW157_mapper)
                input_data['SurgicalRiskCategory'] = input_data['SurgicalRiskCategory'].map(SurgRiskCategory_mapper)
                input_data['AnesthesiaTypeCategory'] = input_data['AnesthesiaTypeCategory'].map(anaestype_mapper)
                input_data['PriorityCategory'] = input_data['PriorityCategory'].map(priority_mapper)

                # Convert to PyTorch tensor
                input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
                
                # Generate prediction probabilities
                icu_probability = icu_classifier.predict_proba(input_tensor)[:, 1].item() * 100
                mortality_probability = mortality_classifier.predict_proba(input_tensor)[:, 1].item() * 100
                
                # Save prediction probability
                st.session_state.last_icu_prediction_probability = f"ICU Predicted probability: {icu_probability:.2f}%"
                st.session_state.last_mortality_prediction_probability = f"Mortality Predicted probability: {mortality_probability:.2f}%"
                
                # Display prediction
                st.write(st.session_state.last_icu_prediction_probability)
                st.write(st.session_state.last_mortality_prediction_probability)

                message = {"role": "assistant", "content": "Mortality prediction: " + st.session_state.last_icu_prediction_probability}
                st.session_state.messages.append(message)
                message = {"role": "assistant", "content": "Mortality prediction: " + st.session_state.last_mortality_prediction_probability}
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
                    "ICU Admission >24 hours": 'Unknown',
                    "Mortality": 'Unknown'
                }

                # Show the "Save Patient Data" button
                st.session_state.show_save_button = True

    # Conditionally display the "Save Patient Data" button
    if st.session_state.get('show_save_button', False):
        if st.sidebar.button('Save Patient Data'):
            st.session_state.show_patient_form = True
    
    # Conditionally display the patient ID form
    if st.session_state.get('show_patient_form', False):
        with st.sidebar.form(key='patient_id_form'):
            st.session_state.patient_id = st.text_input("Enter Patient ID (type 'exit' to cancel):")
            submit_button = st.form_submit_button("Submit ID")

        if submit_button:
            handle_save_patient_data()
