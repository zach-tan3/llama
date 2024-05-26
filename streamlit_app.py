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

# Set Streamlit configuration
st.set_page_config(page_title="ICURISK with ChatGPT", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: "sans serif";
        background-color: #f0f0f5;
    }
    .stButton button {
        background-color: #6eb52f;
        color: white;
    }
    .stSidebar {
        background-color: #e0e0ef;
    }
    .stSidebar .stButton button {
        background-color: #6eb52f;
        color: white;
    }
    .stSidebar .stSelectbox, .stSidebar .stSlider {
        margin-bottom: 20px;
    }
    .stChatMessage {
        margin-bottom: 20px;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .header-container img {
        width: 200px;
        margin-right: 20px;
    }
    .vertical-line {
        border-left: 2px solid #6eb52f;
        height: 80px;
        margin-right: 20px;
    }
    .logo-text {
        font-weight: 700;
        font-size: 40px;
        color: #000000;
        padding-top: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

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

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load models
if os.path.exists('icu_classifier.pkl') and os.path.exists('mortality_classifier.pkl'):
    icu_classifier = joblib.load('icu_classifier.pkl')
    mortality_classifier = joblib.load('mortality_classifier.pkl')
else:
    st.error('Model files not found. Please ensure the files are uploaded.')

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "This is a risk calculator for need for admission into an Intensive Care Unit (ICU) of a patient post-surgery and for Mortality. Ask me anything."}]

# Function for Risk Calculator w/ ChatGPT page
def risk_calculator_page():
    # Initialize session state for messages if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "This is a risk calculator for need for admission into an Intensive Care Unit (ICU) of a patient post-surgery and for Mortality. Ask me anything."}]

    # Sidebar input elements
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
            ASAcategorybinned_mapper = {"I": 0, "Ii": 1, 'Iii': 2, 'Iv-Vi': 3}
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
            
            # Display prediction probabilities
            st.session_state.last_icu_prediction_probability = f"ICU Predicted probability: {icu_probability:.2f}%"
            st.session_state.last_mortality_prediction_probability = f"Mortality Predicted probability: {mortality_probability:.2f}%"
    
    with col2:
        st.button('Clear Chat History', on_click=clear_chat_history)

# Display prediction results
if 'last_icu_prediction_probability' in st.session_state and 'last_mortality_prediction_probability' in st.session_state:
    st.subheader("Prediction Results")
    st.write(st.session_state.last_icu_prediction_probability)
    st.write(st.session_state.last_mortality_prediction_probability)
    st.markdown("</div>", unsafe_allow_html=True)

    # Chatbot interaction section
    st.markdown("<h2 class='section-title'>Chatbot Interaction</h2>", unsafe_allow_html=True)
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
                response = openai.ChatCompletion.create(
                    model=st.session_state.model,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                )

                # Extract the content from the response
                full_response = response.choices[0].message["content"]
                message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Function for Risk Model Development page
def risk_model_development_page():
    st.title("Risk Model Development")
    st.write("This is the Risk Model Development page.")
    model1_button = st.sidebar.button('Model 1')
    model2_button = st.sidebar.button('Model 2')
    model3_button = st.sidebar.button('Model 3')

    if model1_button:
        st.image("static/model1.png")
    if model2_button:
        st.image("static/model2.png")
    if model3_button:
        st.image("static/model3.png")

# Sidebar navigation dropdown
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to", ["Risk Calculator w/ ChatGPT", "Risk Model Development"])

if page == "Risk Calculator w/ ChatGPT":
    risk_calculator_page()
elif page == "Risk Model Development":
    risk_model_development_page()
