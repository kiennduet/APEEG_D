import streamlit as st

def show_home():
    st.header("Welcome to the EEG Analysis Dashboard!")
    st.markdown("""
        <div style="background-color:#eaf5ff; padding: 20px; border-radius: 10px; border: 1px solid #cce0ff;">
        <p style="font-size: 1.1em;">
        This dashboard provides a comprehensive suite of tools for analyzing EEG data.
        </p>
        <ul>
            <li>📤 <strong>Load and Visualize</strong> your EEG recordings.</li>
            <li>🛠️ <strong>Extract Meaningful Features</strong> from the signals.</li>
            <li>🎓 <strong>Train Machine Learning Models</strong> for classification or regression tasks.</li>
            <li>🔮 <strong>Make Predictions</strong> on new, unseen data.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Getting Started")
    col1, col2 = st.columns(2)
    with col1:
        st.info("New to EEG analysis? Start by uploading your data in the 'Load & View EEG Data' section.")
    with col2:
        st.success("Have features and want to train a model? Head to 'Train ML Model'.")