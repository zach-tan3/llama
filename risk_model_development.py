# risk_model_development.py
import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from utils import set_bg, logo3, CSS_styling

def risk_model_development_page():
    
    # Custom CSS for styling
    set_bg('static/Light blue background.jpg')
    logo3('static/ICURISK_Logo.png')
    CSS_styling()

    model1_button = st.sidebar.markdown(
        """
        <button class="icon-button">
            <img src="data:image/png;base64,{}" width="20"/>
            <span>Model 1</span>
        </button>
        """.format(base64.b64encode(open("static/model1_icon.png", "rb").read()).decode()),
        unsafe_allow_html=True
    )
    model2_button = st.sidebar.markdown(
        """
        <button class="icon-button">
            <img src="data:image/png;base64,{}" width="20"/>
            <span>Model 2</span>
        </button>
        """.format(base64.b64encode(open("static/model2_icon.png", "rb").read()).decode()),
        unsafe_allow_html=True
    )
    model3_button = st.sidebar.markdown(
        """
        <button class="icon-button">
            <img src="data:image/png;base64,{}" width="20"/>
            <span>Model 3</span>
        </button>
        """.format(base64.b64encode(open("static/model3_icon.png", "rb").read()).decode()),
        unsafe_allow_html=True
    )

    if st.sidebar.button('Model 1'):
        st.image("static/model1.png")
    if st.sidebar.button('Model 2'):
        st.image("static/model2.png")
    if st.sidebar.button('Model 3'):
        st.image("static/model3.png")
