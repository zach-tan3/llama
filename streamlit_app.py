import streamlit as st
import replicate
import os
import pandas as pd
import torch
import numpy as np
import requests
from io import BytesIO

# Load the model
model_path = "discriminator"  # Path to your model file in the GitHub repository
model = torch.load(model_path, map_location=torch.device('cpu'))

# App title
st.set_page_config(page_title="Model Prediction")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input, llm):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
    return output

if st.sidebar.button('Predict'):
    gender = st.sidebar.selectbox('Gender', ['MALE', 'FEMALE'])
    anaestype = st.sidebar.selectbox('Anaestype', ['GA', 'EA'])
    priority = st.sidebar.selectbox('Priority', ['Elective', 'Emergency'])
    age = st.sidebar.slider('Age', 18, 99, 40)
    surgrisk = st.sidebar.selectbox('SurgRisk', ['Low', 'Moderate', 'High'])
    race = st.sidebar.selectbox('Race', ['Chinese', 'Others'])
    
    prompt = {'gender': gender, 'anaestype': anaestype, 'priority': priority, 'age': age, 'surgrisk': surgrisk, 'race': race}
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Preprocess your input data
            input_data = pd.DataFrame({'GENDER': [gender],
                                       'AnaestypeCategory': [anaestype],
                                       'PriorityCategory': [priority],
                                       'AGEcategory': [age],
                                       'SurgRiskCategory': [surgrisk],
                                       'RaceCategory': [race]})

            # Map categorical values
            input_data['GENDER'] = input_data['GENDER'].map({'MALE': 0, 'FEMALE': 1})
            input_data['AnaestypeCategory'] = input_data['AnaestypeCategory'].map({'GA': 0, 'EA': 1})
            input_data['PriorityCategory'] = input_data['PriorityCategory'].map({'Elective': 0, 'Emergency': 1})
            input_data['AGEcategory'] = input_data['AGEcategory'].map({'18-29': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70-79': 5, '80-89': 6, '90-99': 7})
            input_data['SurgRiskCategory'] = input_data['SurgRiskCategory'].map({'Low': 0, 'Moderate': 1, 'High': 2})
            input_data['RaceCategory'] = input_data['RaceCategory'].map({'Chinese': 0, 'Others': 1})

            # Convert to PyTorch tensor
            input_tensor = torch.tensor(input_data.values, dtype=torch.float32)

            # Generate prediction
            output = model(input_tensor)

            # Convert output to probability
            probability = torch.sigmoid(output).item()
            
            # Generate LLM response
            response = generate_llama2_response(prompt, llm)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            
            # Display prediction
            st.write(f"Predicted probability: {probability:.2f}")

    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
