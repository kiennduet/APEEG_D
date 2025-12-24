import streamlit as st
from utils_store.M04_Classification import UI_train_ml

def show_train_model():
    st.header("🎓 Train Machine Learning Model")
    st.info("Upload your feature set and labels to train a classification or regression model.")
    with st.expander("Model Training Interface", expanded=True):
        UI_train_ml()