import streamlit as st
import os

# Import your utility functions
from utils_store.M01_DataLoader import ui_eeg_subjects_uploader, delete_folder, ui_select_subject, ui_eeg_groups_uploader
from utils_store.M02_PSDTransform import UI_plot_psd
from utils_store.M03_FeatureExtraction import UI_feature_extraction, ui_plot_topo, UI_feature_extraction_groups, ui_adjust_param_fooof, plot_fooof
from utils_store.M04_Classification import UI_predict_ml, UI_train_ml

# --- Page Configuration ---
st.set_page_config(
    page_title="EEG Analysis Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        /* Sidebar styling */
        .css-1d391kg .stButton>button {
            color: #ffffff;
            background-color: #3498db;
            border-radius: 8px;
            padding: 10px 15px;
            width: 100%;
            text-align: left;
            margin-bottom: 8px;
            border: none;
            font-weight: 500;
            transition: background-color 0.3s ease;
        }
        .css-1d391kg .stButton>button:hover {
            background-color: #2980b9;
        }
        .css-1d391kg .stButton>button:focus {
            outline: none;
            box-shadow: 0 0 0 2px #ffffff, 0 0 0 4px #3498db;
        }
        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
            color: #ecf0f1;
        }
    </style>
    """, unsafe_allow_html=True)


# --- App Title ---
st.title("🧠 EEG Analysis Pro")
st.caption("Your all-in-one platform for EEG data processing and machine learning.")
st.markdown("---")


# --- Initialize Session State for Current Page ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# --- Function to Change Page ---
def set_page(page_name):
    st.session_state.current_page = page_name

# --- Navigation Sidebar ---
with st.sidebar:
    st.header("Navigation Panel")
    pages = {
        "Home": "🏠 Home",
        "Load Data": "📤 Load & View EEG Data",
        "Feature Extraction": "🛠️ Feature Extraction",
        "Train Model": "🎓 Train ML Model",
        "Predict": "🔮 Make Predictions"
    }

    for page_key, page_display_name in pages.items():
        if st.button(page_display_name, on_click=set_page, args=(page_key,), key=f"btn_{page_key}", use_container_width=True):
            pass

    st.markdown("---")
    st.info("💡 Tip: Upload your EEG data in the 'Load Data' section to begin.")


# --- Display Content Based on Current Page ---
if st.session_state.current_page == "Home":
    st.header("Welcome to the EEG Analysis Dashboard!")
    st.markdown("""
        <div style="background-color:#eaf5ff; padding: 20px; border-radius: 10px; border: 1px solid #cce0ff;">
        <p style="font-size: 1.1em;">
        This dashboard provides a comprehensive suite of tools for analyzing EEG data.
        Navigate through the different sections using the sidebar to:
        </p>
        <ul>
            <li>📤 <strong>Load and Visualize</strong> your EEG recordings.</li>
            <li>🛠️ <strong>Extract Meaningful Features</strong> from the signals.</li>
            <li>🎓 <strong>Train Machine Learning Models</strong> for classification or regression tasks.</li>
            <li>🔮 <strong>Make Predictions</strong> on new, unseen data.</li>
        </ul>
        <p style="font-size: 1.1em;">
        Select an option from the navigation panel on the left to get started.
        </p>
        </div>
    """, unsafe_allow_html=True)
    st.subheader("Getting Started")
    col1, col2 = st.columns(2)
    with col1:
        st.info("New to EEG analysis? Start by uploading your data in the 'Load & View EEG Data' section.")
    with col2:
        st.success("Have features and want to train a model? Head to 'Train ML Model'.")


elif st.session_state.current_page == "Load Data":
    st.header("📤 Load and Explore EEG Data")
    input_path = "input/temp_rawData"

    st.subheader("1. Load Single Subject Data")
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

                    selected_channel_fooof = st.selectbox(
                        "Select Channel for FOOOF Fitting:",
                        channel_names,
                        key="fooof_channel_selector_single"
                    )
                    _, ffitting_fig = plot_fooof(
                        freqs=freqs,
                        psds=psd,
                        raw_data=st.session_state.raw_data_selected,
                        pe_settings=pe_settings,
                        ape_settings=ape_settings,
                        channel_names=[selected_channel_fooof],
                        show_fplot=True
                    )
                    st.pyplot(ffitting_fig)
                else:
                    st.info("PSD data not available or not plotted yet.")
    st.markdown("---")


elif st.session_state.current_page == "Feature Extraction":
    st.header("🛠️ EEG Feature Extraction")

    if 'raw_dataset_single' in st.session_state and st.session_state.raw_dataset_single:
        st.subheader("Extract Features for Single Subjects")
        
        # --- LOGIC QUẢN LÝ CACHE ---
        if 'features_subjects_single' not in st.session_state:
            st.session_state.features_subjects_single = None

        with st.expander("Feature Extraction Controls & Results", expanded=True):
            # CASE 1: Chưa có features -> Hiển thị UI trích xuất và tính toán
            if st.session_state.features_subjects_single is None:
                st.info("Configure parameters below and extract features.")
                st.session_state.features_subjects_single = UI_feature_extraction(raw_dataset=st.session_state.raw_dataset_single)
            
            # CASE 2: Đã có features -> Hiển thị kết quả từ Cache + Nút Reset
            else:
                col_status, col_reset = st.columns([3, 1])
                with col_status:
                    st.success(f"✅ Features extracted! ({len(st.session_state.features_subjects_single)} rows)")
                with col_reset:
                    # Nút Reset để tính lại từ đầu
                    if st.button("🔄 Reset / Recalculate"):
                        st.session_state.features_subjects_single = None
                        st.rerun()
                
                # --- ĐÂY LÀ PHẦN HIỂN THỊ KẾT QUẢ KHI ĐANG DÙNG CACHE ---
                st.markdown("#### Extracted Features Table (Cached)")
                st.dataframe(st.session_state.features_subjects_single, use_container_width=True)
                
                # Nút tải xuống (Optional - thường rất hữu ích)
                # csv = st.session_state.features_subjects_single.to_csv(index=False).encode('utf-8')
                # st.download_button("Download Features CSV", csv, "features.csv", "text/csv")

        # --- PHẦN VẼ BIỂU ĐỒ TOPOMAP ---
        # Chỉ hiển thị plot nếu đã có dữ liệu features
        if st.session_state.features_subjects_single is not None:
            st.markdown("---")
            st.markdown("### Topographic Maps")
            # Khi bạn chọn biến trong hàm này, features_subjects_single đã có dữ liệu,
            # nên code sẽ chạy vào 'CASE 2' ở trên (không tính toán lại),
            # và hiển thị bảng dữ liệu nhanh chóng, sau đó vẽ biểu đồ ở đây.
            ui_plot_topo(raw_dataset=st.session_state.raw_dataset_single, features_subjects=st.session_state.features_subjects_single)
            
    else:
        st.warning("Please load EEG data in the 'Load & View EEG Data' section before extracting features.")

    st.markdown("---")


elif st.session_state.current_page == "Train Model":
    st.header("🎓 Train Machine Learning Model")
    st.info("Upload your feature set and labels to train a classification or regression model.")
    with st.expander("Model Training Interface", expanded=True):
        UI_train_ml()

elif st.session_state.current_page == "Predict":
    st.header("🔮 Make Predictions")
    st.info("Load a pre-trained model and new EEG data (or features) to generate predictions.")
    with st.expander("Prediction Interface", expanded=True):
        UI_predict_ml()

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 EEG Analysis Pro. All rights reserved.")
st.sidebar.caption("Developed by: Nguyen Duc Kien")
