import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from streamlit_option_menu import option_menu
from utils import set_bg, logo3, CSS_styling

def risk_model_development_page():
    
    # Custom CSS for styling
    set_bg('static/Light blue background.jpg')
    logo3('static/ICURISK_Logo.png')
    CSS_styling()

    with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Settings'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    selected

    main_page_sidebar = st.sidebar.empty()
    with main_page_sidebar:
        selected_option = option_menu(
            menu_title = 'Navigation',
            menu_icon = 'list-columns-reverse',
            icons = ['box-arrow-in-right', 'person-plus', 'x-circle','arrow-counterclockwise'],
            options = ['Login', 'Create Account', 'Forgot Password?', 'Reset Password'],
            styles = {
                "container": {"padding": "5px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px"}} )
    
    roc_button = st.sidebar.button('ROC Curve Comparisons', key='roc_button')
    cm_button = st.sidebar.button('Confusion Matrix Comparisons', key='cm_button')

    if roc_button:
        st.markdown("### ROC Curve Comparison: ICU vs. Mortality")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("static/ICU ROC Curve.png", caption="ICU ROC Curve")
        st.image("static/Mortality ROC Curve.png", caption="Mortality ROC Curve")
        st.markdown('</div>', unsafe_allow_html=True)

    if cm_button:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("static/ICU Confusion Matrix.png", caption="ICU Confusion Matrix")
        st.image("static/Mortality Confusion Matrix.png", caption="Mortality Confusion Matrix")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("### Confusion Matrix Comparison: ICU vs. Mortality")
