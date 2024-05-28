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

    # Sidebar menu with smaller font sizes
    with st.sidebar:
        selected_option = option_menu(
            menu_title='Model Comparisons',
            menu_icon='ui-radios',
            icons=['graph-up', 'diagram-3', 'bar-chart', 'sliders', 'cursor'],
            options=['ROC Curve', 'Confusion Matrix', 'Model Performance', 'Feature Selection', 'Model Selection'],
            styles={
                "container": {"padding": "5px"},
                "nav-title": {"font-size": "12px", "font-weight": "bold"},
                "nav-link": {"font-size": "12px", "text-align": "left", "margin": "0px", "padding": "10px", "border-radius": "5px"},
                "nav-link-selected": {"background-color": "#6eb52f", "color": "white"}
            }
        )

    if selected_option == 'ROC Curve':
        st.markdown("### ROC Curve Comparison: ICU vs. Mortality")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ICU ROC Curve")
            st.image("static/ICU ROC Curve.png", use_column_width=True)
        with col2:
            st.markdown("#### Mortality ROC Curve")
            st.image("static/Mortality ROC Curve.png", use_column_width=True)
        st.markdown("""
            **ROC Curve Comparison:**
            The ROC (Receiver Operating Characteristic) curve is a graphical representation of the diagnostic ability of a binary classifier system. 
            In the context of ICU admission and mortality, it helped to compare the true positive rate (sensitivity) against the false positive rate (1-specificity) at various threshold settings. 
            By comparing the ROC curves of different models, we were able to assess which model performed better in distinguishing between positive and negative outcomes.
        """)

    if selected_option == 'Confusion Matrix':
        st.markdown("### Confusion Matrix Comparison: ICU vs. Mortality")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ICU Confusion Matrix")
            st.image("static/ICU Confusion Matrix.png", use_column_width=True)
        with col2:
            st.markdown("#### Mortality Confusion Matrix")
            st.image("static/Mortality Confusion Matrix.png", use_column_width=True)
        st.markdown("""
            **Confusion Matrix Comparison:**
            The confusion matrix is a specific table layout that allows visualization of the performance of an algorithm. 
            Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa). 
            This comparison was crucial in understanding the number of true positives, true negatives, false positives, and false negatives, thereby providing insights into the model's accuracy and types of errors it made.
        """)

    if selected_option == 'Model Performance':
        st.markdown("### Model Performance Comparison: Train and Test Accuracy")
        col1, col2, col3 = st.columns([1, 5, 1])
        with col1:
            st.empty()
        with col2:
            st.markdown("#### ICU Train and Test Accuracy")
            st.image("static/ICU Train Test Accuracy of Different Models.png", use_column_width=True)
            st.markdown("#### Mortality Train and Test Accuracy")
            st.image("static/Mortality Train Test Accuracy of Different Models.png", use_column_width=True)
        with col3:
            st.empty()
        st.markdown("""
            **Model Performance Comparison:**
            This comparison involved analyzing the train and test accuracy of different models. 
            Train accuracy refers to the performance of the model on the training dataset, whereas test accuracy indicates how well the model generalizes to an unseen dataset. 
            Evaluating both metrics helped us in identifying models that not only fit the training data well but also performed robustly on new data, thus avoiding overfitting or underfitting.
        """)

    if selected_option == 'Feature Selection':
        st.markdown("### Feature Selection: Random Forest and Logistic Regression")
        col1, col2, col3 = st.columns([1, 7, 1])
        with col1:
            st.empty()
        with col2:
            st.markdown("#### ICU Feature Importance")
            st.image("static/ICU Feature Importance.png", use_column_width=True)
            st.markdown("#### Mortality Feature Importance")
            st.image("static/Mortality Feature Importance.png", use_column_width=True)
        with col3:
            st.empty()
        st.markdown("""
            **Feature Selection:**
            Feature selection is the process of selecting a subset of relevant features for model construction. 
            The Random Forest feature importance and Logistic Regression coefficients were used to identify the most significant predictors for ICU admission and mortality. 
            This step was essential in simplifying the model, improving its performance, and providing more interpretable results by focusing on the most impactful features.
        """)

    if selected_option == 'Model Selection':
        st.markdown("### Model Selection: ICU vs. Mortality")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ICU Model Selection")
            st.image("static/ICU Model Selection.png", use_column_width=True)
        with col2:
            st.markdown("#### Mortality Model Selection")
            st.image("static/Mortality Model Selection.png", use_column_width=True)
        st.markdown("""
            **Model Selection:**
            This comparison involved evaluating the performance of different models focused specifically on ICU admission and mortality. 
            By zooming in on the top-performing models, we were able to make an informed decision about which model to select based on various performance metrics and their ability to generalize to new data. 
            This process ensured that the best possible model was chosen for each task, balancing accuracy, robustness, and interpretability.
        """)

# Run the risk model development page
risk_model_development_page()
