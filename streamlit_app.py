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
from dotenv import load_dotenv
import base64

# Set Streamlit configuration with a new theme
st.set_page_config(
    page_title="ICURISK with ChatGPT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: "sans serif";
        background-color: #e0f7fa;
    }
    .stButton button {
        background-color: #6eb52f;
        color: white;
    }
    .stSidebar {
        background-color: #e0f0ef;
    }
    .stSidebar .stButton button {
        background-color: #6eb52f;
        color: white;
    }
    .stSidebar .stSelectbox, .stSidebar .stSlider {
        margin-bottom: 20px;
    }
    .stChatMessage {
        margin-bottom: 10px;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #262730;
    }
    .sub-title {
        font-size: 1.25rem;
        color: #262730;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 20px;
    }
    .input-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .predict-button {
        background-color: #6eb52f;
        color: white;
        width: 100%;
        padding: 10px;
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
        margin-top: 20px;
    }
    .result-container {
        margin-top: 20px;
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .header-container img {
        width: 80px;
        margin-right: 20px;
    }
    .vertical-line {
        border-left: 2px solid #6eb52f;
        height: 80px;
        margin-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("""
<div class='header-container'>
    st.image(r'static_images/ICURISK Logo.png')
    <div class='vertical-line'></div>
    <h1 class='main-title'>ICURISK with ChatGPT! 🤖</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>This is a risk calculator for need for admission into an Intensive Care Unit (ICU) of a patient post-surgery and for Mortality.</p>", unsafe_allow_html=True)

# Load environment variables
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
    st.session_state.messages = [{"role": "assistant", "content": "Hi! The name is Vision, here to answer any mind-boggling enquiries. Ask me anything."}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Main container for input parameters
st.header("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.slider('Age', 18, 99, 40)
    PreopEGFRMDRD = st.slider('PreopEGFRMDRD', 0, 160, 80)
    Intraop = st.slider('Intraop', 0, 1, 0)

with col2:
    AnemiaCategoryBinned = st.selectbox('Anemia Category Binned', ['none', 'mild', 'moderate/severe'])
    ASACategoryBinned = st.selectbox('ASA Category Binned', ['i', 'ii', 'iii', 'iv-vi'])
    RDW157 = st.selectbox('RDW15.7', ['<= 15.7', '>15.7'])
    GradeofKidneyDisease = st.selectbox('Grade of Kidney Disease', ['blank', 'g1', 'g2', 'g3a', 'g3b', 'g4', 'g5'])

with col3:
    SurgicalRiskCategory = st.selectbox('Surgical Risk Category', ['low', 'moderate', 'high'])
    AnesthesiaTypeCategory = st.selectbox('Anesthesia Type Category', ['ga', 'ra'])
    PriorityCategory = st.selectbox('Priority Category', ['elective', 'emergency'])

prediction_prompt = {
    'Age': Age,
    'PreopEGFRMDRD': PreopEGFRMDRD,
    'Intraop': Intraop,
    'GradeofKidneyDisease': GradeofKidneyDisease,
    'AnemiaCategoryBinned': AnemiaCategoryBinned,
    'ASACategoryBinned': ASACategoryBinned,
    'RDW15.7': RDW157,
    'SurgicalRiskCategory': SurgicalRiskCategory,
    'AnesthesiaTypeCategory': AnesthesiaTypeCategory,
    'PriorityCategory': PriorityCategory
}

# Prediction button and processing
if st.button('Predict', key='predict', help='Click to predict ICU admission and mortality'):
    with st.spinner("Thinking..."):
        # Preprocess your input data
        input_data = pd.DataFrame([prediction_prompt])

        # Mappings of categorical values
        ASAcategorybinned_mapper = {"i": 0, "ii": 1, 'iii': 2, 'iv-vi': 3}
        GradeofKidneyDisease_mapper = {"blank": 0, "g1": 1, "g2": 2, "g3a": 3, "g3b": 4, "g4": 5, "g5": 6}
        AnemiaCategoryBinned_mapper = {"none": 0, "mild": 1, "moderate/severe": 2}
        RDW157_mapper = {"<= 15.7": 0, ">15.7": 1}
        SurgicalRiskCategory_mapper = {"low": 0, "moderate": 1, "high": 2}
        AnesthesiaTypeCategory_mapper = {"ga": 0, "ra": 1}
        PriorityCategory_mapper = {"elective": 0, "emergency": 1}

        # Map categorical values
        input_data['ASACategoryBinned'] = input_data['ASACategoryBinned'].map(ASAcategorybinned_mapper)
        input_data['GradeofKidneyDisease'] = input_data['GradeofKidneyDisease'].map(GradeofKidneyDisease_mapper)
        input_data['AnemiaCategoryBinned'] = input_data['AnemiaCategoryBinned'].map(AnemiaCategoryBinned_mapper)
        input_data['RDW15.7'] = input_data['RDW15.7'].map(RDW157_mapper)
        input_data['SurgicalRiskCategory'] = input_data['SurgicalRiskCategory'].map(SurgicalRiskCategory_mapper)
        input_data['AnesthesiaTypeCategory'] = input_data['AnesthesiaTypeCategory'].map(AnesthesiaTypeCategory_mapper)
        input_data['PriorityCategory'] = input_data['PriorityCategory'].map(PriorityCategory_mapper)

        # Convert to PyTorch tensor
        input_tensor = torch.tensor(input_data.values, dtype=torch.float32)

        # Generate prediction probabilities
        icu_probability = icu_classifier.predict_proba(input_tensor)[:, 1].item() * 100
        mortality_probability = mortality_classifier.predict_proba(input_tensor)[:, 1].item() * 100

        # Display prediction probabilities
        st.session_state.last_icu_prediction_probability = f"ICU Predicted probability: {icu_probability:.2f}%"
        st.session_state.last_mortality_prediction_probability = f"Mortality Predicted probability: {mortality_probability:.2f}%"

# Display prediction results
if 'last_icu_prediction_probability' in st.session_state and 'last_mortality_prediction_probability' in st.session_state:
    st.subheader("Prediction Results")
    st.write(st.session_state.last_icu_prediction_probability)
    st.write(st.session_state.last_mortality_prediction_probability)
    st.markdown("</div>", unsafe_allow_html=True)

# Chatbot interaction section moved to the bottom
st.markdown("<h2 class='section-title'>Chatbot Interaction</h2>", unsafe_allow_html=True)

# Initialize session state for messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! The name is Vision, here to answer any mind-boggling enquiries. Ask me anything."}]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize model
if "model" not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"

# User input for the chatbot
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

st.markdown("</div>", unsafe_allow_html=True)
