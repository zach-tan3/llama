# risk_calculator.py

import streamlit as st
import pandas as pd
import torch
import joblib
import openai
from io import BytesIO
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

def risk_calculator_page():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "This is a risk calculator for need for admission into an Intensive Care Unit (ICU) of a patient post-surgery and for Mortality. Ask me anything."}]

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "model" not in st.session_state:
        st.session_state.model = "gpt-3.5-turbo"

    if user_prompt := st.chat_input("Your prompt"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

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

    if os.path.exists('icu_classifier.pkl') and os.path.exists('mortality_classifier.pkl'):
        icu_classifier = joblib.load('icu_classifier.pkl')
        mortality_classifier = joblib.load('mortality_classifier.pkl')
    else:
        st.error('Model files not found. Please ensure the files are uploaded.')

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "This is a risk calculator for need for admission into an Intensive Care Unit (ICU) of a patient post-surgery and for Mortality. Ask me anything."}]

    st.sidebar.header("Input Parameters")

    Age = st.sidebar.slider('Age', 18, 99, 40)
    PreopEGFRMDRD = st.sidebar.slider('PreopEGFRMDRD', 0, 160, 80)
    Intraop = st.sidebar.slider('Intraop', 0, 1, 0)
    ASACategoryBinned = st.sidebar.selectbox('ASA Category Binned', ['I', 'Ii', 'Iii', 'Iv-Vi'])
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

    col1, col2 = st.sidebar.columns([1, 1])
    
    with col1:
        if st.button('Predict'):
            with st.chat_message("user"):
                st.write(prediction_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
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
        
                    ASAcategorybinned_mapper = {"I": 0, "Ii": 1, 'Iii': 2, 'Iv-Vi': 3}
                    GradeofKidneydisease_mapper = {"Blank": 0, "G1": 1, "G2": 2, "G3a": 3, "G3b": 4, "G4": 5, "G5": 6}
                    Anemiacategorybinned_mapper = {"None": 0, "Mild": 1, "Moderate/Severe": 2}
                    RDW157_mapper = {"<= 15.7": 0, ">15.7": 1}
                    SurgRiskCategory_mapper = {"Low": 0, "Moderate": 1, "High": 2}
                    anaestype_mapper = {"Ga": 0, "Ra": 1}
                    priority_mapper = {"Elective": 0, "Emergency": 1}
                    
                    input_data['ASACategoryBinned'] = input_data['ASACategoryBinned'].map(ASAcategorybinned_mapper)
                    input_data['GradeofKidneyDisease'] = input_data['GradeofKidneyDisease'].map(GradeofKidneydisease_mapper)
                    input_data['AnemiaCategoryBinned'] = input_data['AnemiaCategoryBinned'].map(Anemiacategorybinned_mapper)
                    input_data['RDW15.7'] = input_data['RDW15.7'].map(RDW157_mapper)
                    input_data['SurgicalRiskCategory'] = input_data['SurgicalRiskCategory'].map(SurgRiskCategory_mapper)
                    input_data['AnesthesiaTypeCategory'] = input_data['AnesthesiaTypeCategory'].map(anaestype_mapper)
                    input_data['PriorityCategory'] = input_data['PriorityCategory'].map(priority_mapper)
        
                    input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
                    
                    icu_probability = icu_classifier.predict_proba(input_tensor)[:, 1].item() * 100
                    mortality_probability = mortality_classifier.predict_proba(input_tensor)[:, 1].item() * 100
                    
                    st.session_state.last_icu_prediction_probability = f"ICU Predicted probability: {icu_probability:.2f}%"
                    st.session_state.last_mortality_prediction_probability = f"Mortality Predicted probability: {mortality_probability:.2f}%"
                    
                    st.write(st.session_state.last_icu_prediction_probability)
                    st.write(st.session_state.last_mortality_prediction_probability)
        
                    message = {"role": "assistant", "content": "Mortality prediction: " + st.session_state.last_icu_prediction_probability}
                    st.session_state.messages.append(message)
                    message = {"role": "assistant", "content": "Mortality prediction: " + st.session_state.last_mortality_prediction_probability}
                    st.session_state.messages.append(message)

    with col2:
        st.button('Clear Chat History', on_click=clear_chat_history)
