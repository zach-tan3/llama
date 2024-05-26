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

# Set Streamlit configuration with a new theme
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
    </style>
    """, unsafe_allow_html=True)

st.title("ICURISK with ChatGPT! 🤖")

# Load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state for messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "This is a risk calculator for need for admission into an Intensive Care Unit (ICU) of a patient post-surgery and for Mortality. Ask me anything."}]

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

        try:
            response = openai.ChatCompletion.create(
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

# Load models
if os.path.exists('icu_classifier.pkl') and os.path.exists('mortality_classifier.pkl'):
    icu_classifier = joblib.load('icu_classifier.pkl')
    mortality_classifier = joblib.load('mortality_classifier.pkl')
else:
    st.error('Model files not found. Please ensure the files are uploaded.')

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "This is a risk calculator for need for admission into an Intensive Care Unit (ICU) of a patient post-surgery. Ask me anything."}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Sidebar input elements
st.sidebar.header("Input Parameters")

AGE = st.sidebar.slider('Age', 18, 99, 40)
PREOPEGFRMDRD = st.sidebar.slider('PreopEGFRMDRD', 0, 160, 80)
ASACATEGORYBINNED = st.sidebar.selectbox('ASA Category Binned', ['I', 'II', 'III', 'IV-VI'])
GRADEOFKIDNEYDISEASE = st.sidebar.selectbox('Grade of Kidney Disease', ['BLANK', 'G1', 'G2', 'G3A', 'G3B', 'G4', 'G5'])
ANEMIACATEGORYBINNED = st.sidebar.selectbox('Anemia Category Binned', ['NONE', 'MILD', 'MODERATE/SEVERE'])
RDW157 = st.sidebar.selectbox('RDW15.7', ['<= 15.7', '>15.7'])
SURGICALRISKCATEGORY = st.sidebar.selectbox('Surgical Risk Category', ['LOW', 'MODERATE', 'HIGH'])
INTRAOP = st.sidebar.slider('Intraop', 0, 1, 0)
ANESTHESIATYPECATEGORY = st.sidebar.selectbox('Anesthesia Type Category', ['GA', 'RA'])
PRIORITYCATEGORY = st.sidebar.selectbox('Priority Category', ['ELECTIVE', 'EMERGENCY'])

prediction_prompt = {
    'AGE': AGE,
    'PREOPEGFRMDRD': PREOPEGFRMDRD,
    'ASACATEGORYBINNED': ASACATEGORYBINNED,
    'GRADEOFKIDNEYDISEASE': GRADEOFKIDNEYDISEASE,
    'ANEMIACATEGORYBINNED': ANEMIACATEGORYBINNED,
    'RDW15.7': RDW157,
    'SURGICALRISKCATEGORY': SURGICALRISKCATEGORY,
    'INTRAOP': INTRAOP,
    'ANESTHESIATYPECATEGORY': ANESTHESIATYPECATEGORY,
    'PRIORITYCATEGORY': PRIORITYCATEGORY
}

# Prediction button and processing
if st.sidebar.button('Predict'):
    with st.chat_message("user"):
        st.write(prediction_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Preprocess your input data
            input_data = pd.DataFrame([prediction_prompt])

            # Mappings of categorical values
            ASACATEGORYBINNED_mapper = {"I": 0, "II": 1, 'III': 2, 'IV-VI': 3}
            GRADEOFKIDNEYDISEASE_mapper = {"BLANK": 0, "G1": 1, "G2": 2, "G3A": 3, "G3B": 4, "G4": 5, "G5": 6}
            ANEMIACATEGORYBINNED_mapper = {"NONE": 0, "MILD": 1, "MODERATE/SEVERE": 2}
            RDW157_mapper = {"<= 15.7": 0, ">15.7": 1}
            SURGICALRISKCATEGORY_mapper = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
            ANESTHESIATYPECATEGORY_mapper = {"GA": 0, "RA": 1}
            PRIORITYCATEGORY_mapper = {"ELECTIVE": 0, "EMERGENCY": 1}

            # Map categorical values
            input_data['ASACATEGORYBINNED'] = input_data['ASACATEGORYBINNED'].map(ASACATEGORYBINNED_mapper)
            input_data['GRADEOFKIDNEYDISEASE'] = input_data['GRADEOFKIDNEYDISEASE'].map(GRADEOFKIDNEYDISEASE_mapper)
            input_data['ANEMIACATEGORYBINNED'] = input_data['ANEMIACATEGORYBINNED'].map(ANEMIACATEGORYBINNED_mapper)
            input_data['RDW15.7'] = input_data['RDW15.7'].map(RDW157_mapper)
            input_data['SURGICALRISKCATEGORY'] = input_data['SURGICALRISKCATEGORY'].map(SURGICALRISKCATEGORY_mapper)
            input_data['ANESTHESIATYPECATEGORY'] = input_data['ANESTHESIATYPECATEGORY'].map(ANESTHESIATYPECATEGORY_mapper)
            input_data['PRIORITYCATEGORY'] = input_data['PRIORITYCATEGORY'].map(PRIORITYCATEGORY_mapper)

            # Convert to PyTorch tensor
            input_tensor = torch.tensor(input_data.values, dtype=torch.float32)

            # Generate prediction probabilities
            icu_probability = icu_classifier.predict_proba(input_tensor)[:, 1].item() * 100
            mortality_probability = mortality_classifier.predict_proba(input_tensor)[:, 1].item() * 100

            # Display prediction probabilities
            st.session_state.last_icu_prediction_probability = f"ICU Predicted Probability: {icu_probability:.2f}%"
            st.session_state.last_mortality_prediction_probability = f"Mortality Predicted Probability: {mortality_probability:.2f}%"

            st.write(st.session_state.last_icu_prediction_probability)
            st.write(st.session_state.last_mortality_prediction_probability)

            st.session_state.messages.append({"role": "assistant", "content": st.session_state.last_icu_prediction_probability})
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.last_mortality_prediction_probability})

# Display prediction results
if 'last_icu_prediction_probability' in st.session_state and 'last_mortality_prediction_probability' in st.session_state:
    st.subheader("Prediction Results")
    st.write(st.session_state.last_icu_prediction_probability)
    st.write(st.session_state.last_mortality_prediction_probability)

# Chatbot interaction section moved to the bottom
st.markdown("<h2 class='section-title'>Chatbot Interaction</h2>", unsafe_allow_html=True)

# Initialize session state for messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! The name is Vision, here to answer any mind-boggling enquiries. Ask me anything."}]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
            full_response = response.choices[0].message.content
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.markdown("</div>", unsafe_allow_html=True)
