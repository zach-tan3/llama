import streamlit as st
import replicate
import os
import pandas as pd
import torch
import numpy as np
import requests
from io import BytesIO
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create an instance of the model
model_path = "discriminator"  # Path to your model file in the GitHub repository
model = Discriminator(input_size=6)  # Assuming the input size is 6, you need to update it accordingly
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# App title
st.set_page_config(page_title="Model Prediction")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Models and parameters')
    selected_model = st.selectbox('Choose a Llama2 model', ['Llama2-70B', 'Llama2-13B', 'Llama2-7B'], key='selected_model')
    if selected_model == 'Llama2-70B':
        llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    elif selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    temperature = st.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('max_length', min_value=32, max_value=9999, value=120, step=8)
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

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

if st.sidebar.button('Predict'):
    gender = st.sidebar.selectbox('Gender', ['MALE', 'FEMALE'])
    anaestype = st.sidebar.selectbox('Anaestype', ['GA', 'EA'])
    priority = st.sidebar.selectbox('Priority', ['Elective', 'Emergency'])
    age = st.sidebar.slider('Age', 18, 99, 40)
    surgrisk = st.sidebar.selectbox('SurgRisk', ['Low', 'Moderate', 'High'])
    race = st.sidebar.selectbox('Race', ['Chinese', 'Others'])

    age_category = None
    if age < 30:
        age_category = '18-29'
    elif age < 40:
        age_category = '30-39'
    elif age < 50:
        age_category = '40-49'
    elif age < 60:
        age_category = '50-59'
    elif age < 70:
        age_category = '60-69'
    elif age < 80:
        age_category = '70-79'
    elif age < 90:
        age_category = '80-89'
    else:
        age_category = '90-99'

    prompt = {'gender': gender, 'anaestype': anaestype, 'priority': priority, 'age': age_category, 'surgrisk': surgrisk, 'race': race}
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Preprocess your input data
            input_data = pd.DataFrame({'GENDER': [gender],
                                       'AnaestypeCategory': [anaestype],
                                       'PriorityCategory': [priority],
                                       'AGEcategory': [age_category],
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
            with torch.no_grad():
                probability = model(input_tensor)
                predicted = (probability >= 0.5).float()  # Here, you are using a threshold of 0.5 to determine the class.
            
            # Display prediction
            st.write(f"Predicted probability: {probability.item():.2f}")

    message = {"role": "assistant", "content": f"Predicted probability: {predicted.item():.2f}"}
    st.session_state.messages.append(message)
