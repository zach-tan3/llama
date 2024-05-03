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
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-70B', 'Llama2-13B', 'Llama2-7B'], key='selected_model')
    if selected_model == 'Llama2-70B':
        llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    elif selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=9999, value=120, step=8)
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
            with torch.no_grad():
                probability = model(input_tensor)
                predicted = (probability >= 0.5).float()  # Here, you are using a threshold of 0.5 to determine the class.
            
            with st.chat_message("assistant"):
                st.write(f"Predicted probability: {probability.item():.2f}")
            
            # Display prediction
            st.write(f"Predicted probability: {probability.item():.2f}")

    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

