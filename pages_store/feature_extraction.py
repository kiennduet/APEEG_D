import streamlit as st
from utils_store.M03_FeatureExtraction import UI_feature_extraction, ui_plot_topo, ui_select_feature, ui_plot_feature_line

def show_feature_extraction():
    st.header("🛠️ EEG Feature Extraction")
    if 'raw_dataset_single' in st.session_state and st.session_state.raw_dataset_single:
        st.subheader("Extract Features")
        selected_features = ui_select_feature()

        if 'df_features_subjects' not in st.session_state:
            st.session_state.df_features_subjects = None

        with st.expander("Results", expanded=True):
            feature_results = UI_feature_extraction(raw_dataset=st.session_state.raw_dataset_single,
                                                    selected_features=selected_features)
            
            if feature_results is not None:
                st.session_state.df_features_subjects = feature_results

        if st.session_state.df_features_subjects is not None:
            ui_plot_topo(df_features_subjects=st.session_state.df_features_subjects, 
                         selected_features=selected_features)
            
            ui_plot_feature_line(df_features_subjects=st.session_state.df_features_subjects,
                                 selected_features=selected_features)
            
    else:
        st.warning("Please load EEG data in the 'Load & View EEG Data' section before extracting features.")
    