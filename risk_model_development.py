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

    selected_option = option_menu(
        menu_title='Model Comparisons',
        menu_icon='list-columns-reverse',
        icons=['bar-chart-line', 'diagram-3'],
        options=['ROC Curve Comparisons', 'Confusion Matrix Comparisons'],
        styles={
            "container": {"padding": "5px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "padding": "10px", "border-radius": "5px"},
            "nav-link-selected": {"background-color": "#6eb52f", "color": "white"}
        }
    )

    if selected_option == 'ROC Curve Comparisons':
        st.markdown("### ROC Curve Comparison: ICU vs. Mortality")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("static/ICU ROC Curve.png", caption="ICU ROC Curve", use_column_width=True)
        st.image("static/Mortality ROC Curve.png", caption="Mortality ROC Curve", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if selected_option == 'Confusion Matrix Comparisons':
        st.markdown("### Confusion Matrix Comparison: ICU vs. Mortality")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("static/ICU Confusion Matrix.png", caption="ICU Confusion Matrix", use_column_width=True)
        st.image("static/Mortality Confusion Matrix.png", caption="Mortality Confusion Matrix", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
