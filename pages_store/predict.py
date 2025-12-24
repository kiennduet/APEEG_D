import streamlit as st
from utils_store.M04_Classification import UI_predict_ml

def show_predict():
    st.header("🔮 Make Predictions")
    st.info("Load a pre-trained model and new EEG data (or features) to generate predictions.")
    with st.expander("Prediction Interface", expanded=True):
        UI_predict_ml()