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
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

st.title("My Own ChatGPT!🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# initialize model
if "model" not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"

# user input
if user_prompt := st.chat_input("Your prompt"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # generate responses
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


# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")
st.session_state.last_prediction_probability = " "


# Create an instance of the model
if os.path.exists('icu_classifier.pkl') and os.path.exists('mortality_classifier.pkl'):
    icu_classifier = joblib.load('icu_classifier.pkl')
    mortality_classifier = joblib.load('mortality_classifier.pkl')
else:
    st.error('Model files not found. Please ensure the files are uploaded.')
        
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "This is a risk calculator for need for of admission into an Intensive Care Unit (ICU) of a paitent post-surgery. Ask me anything"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
'''
# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input, llm):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. "
    string_dialogue = "You are a helpful healthcare assistant designed to aid users with healthcare-related questions. You are not a substitute for professional medical advice. Always consult a healthcare provider for medical concerns. You only respond once as 'Assistant'.\n\n"
    string_dialogue += "You are part of a project that aims to revolutionize healthcare by leveraging data science and Generative AI technologies to improve patient care and optimize clinical workflows. By integrating Generative AI, the goal is to create a cutting-edge framework capable of autonomously generating a wide range of rich and diverse content, including text, images, and other media types. Our primary focus is on creating a risk calculator to predict mortality and the need for intensive care unit (ICU) stay using data analytics and Meta AI Technologies."
    string_dialogue += "You are to give the last predicted probability from your chat history of need for ICU stay if asked and explain that the predicted probability is the probability of need for ICU stay after a surgery."
    string_dialogue += "The following is a data dictionary of an explanation of each variable which you are to explain to the user if asked: \n"
    string_dialogue += "AGE: Age\n"
    string_dialogue += "GENDER: Gender\n"
    string_dialogue += "RCRIScore: Revised Cardiac Risk Index, see [Wikipedia](https://en.wikipedia.org/wiki/Revised_Cardiac_Risk_Index)\n"
    string_dialogue += "AnemiaCategory: Based on concentration of haemoglobin as per WHO guidelines. May be None, Mild, Moderate, Severe\n"
    string_dialogue += "PreopEGFRMDRD: EGFR = estimated glomerular filtration rate. MDRD = Modification of Diet in Renal Disease equation. Measure of pre-exisiting kidney disease.\n"
    string_dialogue += "GradeofKidneyDisease: Classification of kidney disease statsus based on GFR (see above): see [Kidney.org](https://www.kidney.org/professionals/explore-your-knowledge/how-to-classify-ckd)\n"
    string_dialogue += "AnaesthesiaTypeCategory: General or Regional anaesthesia\n"
    string_dialogue += "PriorityCategory: Elective or Emergency surgery (Emregency = must be done within 24 hours)\n"
    string_dialogue += "AGEcategory: Categorisation of age\n"
    string_dialogue += "SurgicalRiskCategory: Surgical Risk Category (may be low, High, Moderate). Based on based on the 2014 European Society of Cardiology (ESC) and the European Society of Anaesthesiology (ESA) guidelines\n"
    string_dialogue += "RaceCategory: Race\n"
    string_dialogue += "AnemiaCategoryBinned: See #5; Moderate and Severe combined\n"
    string_dialogue += "RDW157: Red Cell Distribution Width > 15.7%\n"
    string_dialogue += "ASACategoryBinned: Surgical risk category, based on ASA-PS. [ASA-PS](https://www.asahq.org/standards-and-practice-parameters/statement-on-asa-physical-status-classification-system)\n"
    
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt, llm)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
'''
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

            message = {"role": "assistant", "content": "Mortality prediction: " + st.session_state.last_icu_prediction_probability}
            st.session_state.messages.append(message)
            message = {"role": "assistant", "content": "Mortality prediction: " + st.session_state.last_mortality_prediction_probability}
            st.session_state.messages.append(message)
