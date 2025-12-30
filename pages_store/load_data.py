import streamlit as st
from utils_store.M01_DataLoader import ui_eeg_subjects_uploader, ui_select_subject
from utils_store.M02_PSDTransform import UI_plot_psd
from utils_store.M03_FeatureExtraction import ui_adjust_param_fooof, plot_fooof

def show_load_data():
    st.header("📤 Load and Explore EEG Data")
    input_path = "input/temp_rawData"

    st.subheader("1. Load Your Data")
    
    if 'raw_dataset_single' not in st.session_state:
        st.session_state.raw_dataset_single = None
    if 'raw_data_selected' not in st.session_state:
        st.session_state.raw_data_selected = None

    st.session_state.raw_dataset_single = ui_eeg_subjects_uploader(input_path=input_path)

    if st.session_state.raw_dataset_single:
        st.session_state.raw_data_selected = ui_select_subject(raw_dataset=st.session_state.raw_dataset_single)

        if st.session_state.raw_data_selected:
            st.markdown("### Power Spectral Density (PSD) & FOOOF Analysis")
            with st.expander("Show PSD and FOOOF Plots", expanded=True):
                freqs, psd, selected_channels = UI_plot_psd(st.session_state.raw_data_selected)

                if freqs is not None and psd is not None:
                    channel_names = st.session_state.raw_data_selected.ch_names
                    pe_settings, ape_settings = ui_adjust_param_fooof()

                    selected_channel_fooof = st.selectbox("Select Channel for FOOOF Fitting:",channel_names,
                                                          key="fooof_channel_selector_single")
                    _, ffitting_fig = plot_fooof(
                        freqs=freqs,psds=psd,
                        raw_data=st.session_state.raw_data_selected,
                        pe_settings=pe_settings,ape_settings=ape_settings,
                        channel_names=[selected_channel_fooof],show_fplot=True
                    )
                    st.pyplot(ffitting_fig)
    st.markdown("---")