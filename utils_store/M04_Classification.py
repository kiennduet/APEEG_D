import pickle
import os
import streamlit as st
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from utils_store.M01_DataLoader import ui_select_channels, ui_eeg_subjects_uploader
from utils_store.M03_FeatureExtraction import plot_two_topomaps_features, ui_select_feature, select_features_from_df, select_channels_from_df, plot_p_value_topomap

def add_label(features_subjects, label):
    features_subjects['Label'] = label
    return features_subjects

def load_features_subjects(file):
    features_subjects = pd.read_csv(file, header=0, index_col=0)
    return features_subjects


def train_ml(X, y, num_folds = 5, num_loops = 10, save_path=None):
    model_results = pd.DataFrame(columns=["Model Name", "Mean", "Std"])
    models = [
        LogisticRegression(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        SVC(),
    ]
    mean_accuracies = np.empty((num_loops,1))

    for model in models:
        model_name = model.__class__.__name__
        mean_accuracies = np.empty((num_loops,1))

        for state in range(1,num_loops+1):
            # Perform K-fold cross-validation and calculate accuracy scores
            stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=state)
            cv_scores = cross_val_score(model, X, y, cv=stratified_kfold)

            # Calculate and print mean accuracy across all folds
            mean_accuracy = cv_scores.mean()
            mean_accuracies[state-1] = mean_accuracy
        
        mean_accuracies_looped = np.mean(mean_accuracies)*100
        std_accuracies_looped  = np.std(mean_accuracies)*100
        model_result = pd.DataFrame([{
            "Model Name": model_name,
            "Mean": mean_accuracies_looped,
            "Std": std_accuracies_looped
        }])
        model_results = pd.concat([model_results, model_result], ignore_index=True)

        if save_path is not None:
            model.fit(X,y)
            model_file = os.path.join(save_path, f"model_{model_name}.pkl")
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

    return model_results

def predict_ml(features_subjects, models):

    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(features_subjects)
        predictions[name] = y_pred

    pred_results = pd.DataFrame(predictions, index=features_subjects.index)

    return pred_results

def ui_load_features_train_groups(raw_dataset, feature_names):
    if raw_dataset:
        selected_channels = ui_select_channels(raw_dataset)

    uploaded_g1 = st.file_uploader("Upload Group 1:", type=["csv"])
    name_g1  = st.text_input("Name for Group 1:", value="")
    uploaded_g2 = st.file_uploader("Upload Group 2:", type=["csv"])
    name_g2  = st.text_input("Name for Group 2:", value="")

    label_g1 = 0
    label_g2 = 1

    if uploaded_g1 is not None and uploaded_g2 is not None:
        df_g1 = load_features_subjects(uploaded_g1)
        df_g1 = select_channels_from_df(select_features_from_df(df_g1, feature_names),selected_channels)
        
        df_g2 = load_features_subjects(uploaded_g2)
        df_g2 = select_channels_from_df(select_features_from_df(df_g2, feature_names),selected_channels)

        return df_g1, df_g2, label_g1, label_g2, name_g1, name_g2
    return None

def creat_data_4train(df_g1, df_g2, label_g1, label_g2):
    features_g1_labeled = add_label(features_subjects=df_g1, label=label_g1)
    features_g2_labeled = add_label(features_subjects=df_g2, label=label_g2)
    df = pd.concat([features_g1_labeled, features_g2_labeled])

    X = df.drop(columns=['Label'])
    y = df['Label']
    return df, X, y

def ui_load_features_predict_subjects():

    uploaded_subjects = st.file_uploader("Upload Features for Prediction:", type=["csv"])

    if uploaded_subjects:
        features_subjects = load_features_subjects(uploaded_subjects)

        return features_subjects 

def ui_load_models():
    uploaded_models = st.file_uploader("Upload models (.pkl)", type=["pkl"], accept_multiple_files=True)
    models = {}

    if uploaded_models:
        for file in uploaded_models:
            model_name = file.name.replace(".pkl", "")
            models[model_name] = pickle.load(file)

    return models

def UI_train_ml():

    st.sidebar.header("", divider="orange")
    st.sidebar.header(":orange[Classification]")
    st.sidebar.subheader("Classification Adjustments")
    selected_features = ui_select_feature()
    num_folds = st.sidebar.slider("Number of folds:", value=5, min_value=5, max_value=20, step=5)
    num_loops = st.sidebar.slider("Number of loops:", value=5, min_value=5, max_value=100, step=5)
    save_path = st.sidebar.text_input("Type path if you want to save models:", value=None)
    raw_dataset = ui_eeg_subjects_uploader(input_path="input/temp_rawData")

    result = ui_load_features_train_groups(raw_dataset, selected_features)
    if result:
        df_g1, df_g2, label_g1, label_g2, name_g1, name_g2 = result
        df, X, y = creat_data_4train(df_g1, df_g2, label_g1, label_g2)
        st.dataframe(df)

        st.header("", divider="rainbow")
        st.header(":orange[Topographic plot]")
        
        topo_fig = plot_two_topomaps_features(df_g1, df_g2, selected_features,
                                              name_g1, name_g2)
        st.pyplot(topo_fig)
        
        topo_p_fig = plot_p_value_topomap(df_g1, df_g2, selected_features)
        st.pyplot(topo_p_fig)

        st.header(":orange[Classification]")
        model_results = train_ml(X, y, num_folds, num_loops, save_path)
        st.dataframe(model_results)

def UI_predict_ml(features_subjects = None):

    st.header("", divider="rainbow")
    st.header(":orange[Prediction]")
    
    if features_subjects is None:
        features_subjects = ui_load_features_predict_subjects()
    models = ui_load_models()

    if features_subjects is not None and not features_subjects.empty and models:
        pred_results = predict_ml(features_subjects, models)
        st.dataframe(pred_results) 

    return


