import pickle
import os
import streamlit as st
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils_store.M01_DataLoader import ui_select_channels, ui_eeg_subjects_uploader
from utils_store.M03_FeatureExtraction import ui_select_feature, select_features_from_df, select_channels_from_df, ui_plot_topo_2group, ui_plot_feature_line

def add_label(features_subjects, label):
    features_subjects['Label'] = label
    return features_subjects

def load_features_subjects(file):
    features_subjects = pd.read_csv(file, header=0, index_col=0)
    return features_subjects

def train_ml(X, y, num_folds=5, num_loops=10, save_path=None):
    # 1. Định nghĩa không gian tham số (Grid) cho từng loại model
    param_grids = {
        "LR": (LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'),
            {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__penalty": ['l1', 'l2']}),
        "RF": (RandomForestClassifier(class_weight='balanced'), 
            {"clf__n_estimators": [10, 50, 100, 200], "clf__max_depth": [None, 10, 20, 30],
             "clf__min_samples_split": [5, 10], "clf__min_samples_leaf": [4, 5, 10]}),
        "GB": (GradientBoostingClassifier(), 
            {"clf__learning_rate": [0.01, 0.1], "clf__n_estimators": [100, 200]}),
        "SVC": (SVC(class_weight='balanced', probability=True), 
            {"clf__C": [0.1, 1, 10], "clf__gamma": ['scale', 'auto']}),
    }
    
    rkf = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=num_loops, random_state=42)
    results = []

    for name, (model, grid) in param_grids.items():
        # 2. Tạo pipeline có scaler để tránh rò rỉ dữ liệu khi Grid Search
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        
        # 3. Tìm bộ tham số tốt nhất (Grid Search)
        grid_search = GridSearchCV(pipe, grid, cv=num_folds, n_jobs=-1).fit(X, y)
        best_model = grid_search.best_estimator_
        
        # 4. Đánh giá bộ tham số tốt nhất bằng Repeated CV (đúng logic loops/folds của bạn)
        scores = cross_val_score(best_model, X, y, cv=rkf, n_jobs=-1)
        loop_means = scores.reshape(num_loops, num_folds).mean(axis=1)
        
        results.append({
            "Model Name": best_model.named_steps['clf'].__class__.__name__, 
            "Mean": loop_means.mean() * 100, 
            "Std": loop_means.std() * 100
        })

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f"model_{name}.pkl"), "wb") as f:
                pickle.dump(best_model, f)
        
    return pd.DataFrame(results)



def train_ml_withUI(X, y, num_folds=5, num_loops=10, save_path=None):
    param_grids = {
        "LR": (LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'),
            {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__penalty": ['l1', 'l2']}),
        "RF": (RandomForestClassifier(class_weight='balanced'), 
            {"clf__n_estimators": [50, 200], "clf__max_depth": [None, 20], "clf__min_samples_leaf": [5, 10]}),
        # "GB": (GradientBoostingClassifier(), {"clf__learning_rate": [0.01, 0.1], "clf__n_estimators": [100, 200]}),
        # "SVC": (SVC(class_weight='balanced', probability=True), {"clf__C": [0.1, 1, 10], "clf__gamma": ['scale', 'auto']}),
    }

    results, total_steps = [], len(param_grids) * num_loops
    bar = st.progress(0)
    status = st.empty()

    for i, (name, (model, grid)) in enumerate(param_grids.items()):
        # 1. Tìm tham số tốt nhất (Grid Search) - chạy 1 lần cho mỗi model
        status.text(f"🔎 Optimizing for {name} classifier...")
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        best_model = GridSearchCV(pipe, grid, cv=num_folds, n_jobs=-1).fit(X, y).best_estimator_
        
        # 2. Chạy từng loop để cập nhật thanh tiến trình UI
        loop_means = []
        for lp in range(num_loops):
            current_step = i * num_loops + lp
            bar.progress(current_step / total_steps)
            status.text(f"⏳ {name}: Running loop {lp+1}/{num_loops}...")
            
            # Mỗi loop dùng một random_state khác nhau (giống logic cũ của bạn)
            cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=lp)
            score = cross_val_score(best_model, X, y, cv=cv, n_jobs=-1).mean()
            loop_means.append(score)
            
        results.append({
            "Model Name": name, 
            "Mean": np.mean(loop_means) * 100, 
            "Std": np.std(loop_means) * 100
        })

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            pickle.dump(best_model, open(os.path.join(save_path, f"model_{name}.pkl"), "wb"))

    bar.progress(1.0)
    return pd.DataFrame(results)

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

    st.markdown("**📥 Upload feature tables file**")
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
    num_loops = st.sidebar.slider("Number of loops:", value=1, min_value=1, max_value=100, step=1)
    save_path = st.sidebar.text_input("Type path if you want to save models:", value=None)

    st.markdown("**📥 Upload example EEG file**") 
    raw_dataset = ui_eeg_subjects_uploader(input_path="input/temp_rawData")

    result = ui_load_features_train_groups(raw_dataset, selected_features)
    
    if result:
        df_g1, df_g2, label_g1, label_g2, name_g1, name_g2 = result
        df, X, y = creat_data_4train(df_g1, df_g2, label_g1, label_g2)
        
        st.subheader("Data Preview")
        st.dataframe(df)
        st.header(":orange[Classification]")
        
        col1, col2 = st.columns(2)
        
        if col1.button("🚀 Start Training Models", use_container_width=True):
            # Chạy hàm huấn luyện và lưu vào session_state
            with st.spinner("Training in progress..."):
                st.session_state.model_results = train_ml_withUI(X, y, num_folds, num_loops, save_path)
            st.success("Training Complete!")

        if "model_results" in st.session_state:
            st.dataframe(st.session_state.model_results)

        # Nút 2: Phân tích đặc trưng (Topo & Line plot)
        if col2.button("🔥 Analysis features", use_container_width=True):
            ui_plot_topo_2group(df_g1, df_g2, selected_features, name_g1, name_g2)
            ui_plot_feature_line(df_g1, selected_features, df_g2, name_g1, name_g2)

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


