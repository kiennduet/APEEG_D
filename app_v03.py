import streamlit as st
from pages_store.home import show_home
from pages_store.load_data import show_load_data
from pages_store.feature_extraction import show_feature_extraction
from pages_store.train_model import show_train_model
from pages_store.predict import show_predict

# --- Page Configuration ---
st.set_page_config(
    page_title="EEG Analysis Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        .css-1d391kg .stButton>button {
            color: #ffffff; background-color: #3498db; border-radius: 8px;
            padding: 10px 15px; width: 100%; text-align: left; margin-bottom: 8px;
            border: none; font-weight: 500; transition: background-color 0.3s ease;
        }
        .css-1d391kg .stButton>button:hover { background-color: #2980b9; }
    </style>
    """, unsafe_allow_html=True)

# --- App Title ---
st.title("🧠 EEG Analysis Pro")
st.caption("Your all-in-one platform for EEG data processing and machine learning.")
st.markdown("---")

# --- Navigation Logic ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

def set_page(page_name):
    st.session_state.current_page = page_name

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
        st.button(
            page_display_name, 
            on_click=set_page, 
            args=(page_key,), 
            key=f"btn_{page_key}", 
            use_container_width=True
        )

    if st.session_state.current_page == "Home":
        st.markdown("---")
        st.info("💡 Tip: Upload your EEG data in the 'Load Data' section to begin.")
        st.caption("© 2024 EEG Analysis Pro.")
        st.caption("Developed by: Nguyen Duc Kien & Tran Quang Duy")

# --- Page Routing ---
if st.session_state.current_page == "Home":
    show_home()
elif st.session_state.current_page == "Load Data":
    show_load_data()
elif st.session_state.current_page == "Feature Extraction":
    show_feature_extraction()
elif st.session_state.current_page == "Train Model":
    show_train_model()
elif st.session_state.current_page == "Predict":
    show_predict()