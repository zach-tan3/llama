# app.py

import streamlit as st
from risk_calculator import risk_calculator_page
from saved_patient_data import saved_patient_data_page
from risk_model_development import risk_model_development_page
from dotenv import load_dotenv
import openai
import base64

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

# Sidebar navigation dropdown
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to", ["Risk Calculator w/ ChatGPT", "Saved Patient Data", "Risk Model Development"])

if page == "Risk Calculator w/ ChatGPT":
    risk_calculator_page()
elif page == "Saved Patient Data":
    saved_patient_data_page()
elif page == "Risk Model Development":
    risk_model_development_page()

