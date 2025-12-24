import streamlit as st
from utils_store.M03_FeatureExtraction import UI_feature_extraction, ui_plot_topo

def show_feature_extraction():
    st.header("🛠️ EEG Feature Extraction")
    if 'raw_dataset_single' in st.session_state and st.session_state.raw_dataset_single:
        st.subheader("Extract Features")
        
        if 'features_subjects' not in st.session_state:
            st.session_state.features_subjects = None

        with st.expander("Results", expanded=True):
            feature_results = UI_feature_extraction(raw_dataset=st.session_state.raw_dataset_single)
            
            if feature_results is not None:
                st.session_state.features_subjects = feature_results

        if st.session_state.features_subjects is not None:
            ui_plot_topo(raw_dataset=st.session_state.raw_dataset_single, 
                         features_subjects=st.session_state.features_subjects)
            
    else:
        st.warning("Please load EEG data in the 'Load & View EEG Data' section before extracting features.")
    