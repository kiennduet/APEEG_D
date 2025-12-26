import os
import streamlit as st
import shutil
import mne
from collections import defaultdict
import re

def delete_folder(path):
    if os.path.exists(path) and os.path.isdir(path):
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

def create_directory(directory):
    """Ensure the input directory exists."""
    os.makedirs(directory, exist_ok=True)

def get_sorted_eeg_channels(channel_list):
    """
    Hàm nhận vào list channel và trả về list đã được sắp xếp theo chuẩn EEG 10-10.
    """
    # 1. Danh sách chuẩn hiện đại (Hệ thống 10-10)
    STANDARD_ORDER = [
        'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
        'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
        'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
        'A1', 'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', 'A2',
        'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
        'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
        'PO9', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'PO10',
        'O1', 'OZ', 'O2', 'I1', 'IZ', 'I2'
    ]
    
    # 2. Xử lý Mapping và Alias
    order_map = {name: i for i, name in enumerate(STANDARD_ORDER)}
    aliases = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8', 'M1': 'A1', 'M2': 'A2'}
    
    # Gộp alias vào order_map
    for old, new in aliases.items():
        if new in order_map:
            order_map[old] = order_map[new]

    # 3. Định nghĩa logic sắp xếp (Internal Function)
    def internal_key(ch_name):
        clean = str(ch_name).upper().strip()
        if clean in order_map:
            return (0, order_map[clean]) # Nhóm 0: Kênh chuẩn, sắp xếp theo index
        return (1, clean)              # Nhóm 1: Kênh lạ, sắp xếp theo tên (A-Z)

    return sorted(list(channel_list), key=internal_key)

@st.cache_resource
def read_eegf(file_path):
    """Read EEG file based on its extension and replace patient name with file name."""
    readers = {
        'set': mne.io.read_raw_eeglab,
        'fif': mne.io.read_raw_fif
    }
    
    ext = file_path.split('.')[-1].lower()
    if ext in readers:
        raw_data = readers[ext](file_path, preload=True)
        
        # Lấy tên file từ đường dẫn
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        
        # Thay thế tên bệnh nhân bằng tên file
        raw_data.info['subject_info'] = {'his_id': file_name_without_ext}
        
        return raw_data
    raise ValueError(f"File format not supported: {ext}")

def get_id_subject(raw_data):
    """Trích xuất tên file của subject từ raw_data."""
    return raw_data.info.get('subject_info', {}).get('his_id', "Unknown")

def get_id_subjects(raw_dataset):
    """Trích xuất danh sách subjects từ raw_dataset."""
    return [raw_data.info['subject_info']['his_id'] for raw_data in raw_dataset]

def filter_subject_data(raw_dataset, selected_subject):
    """Lọc raw_data theo subject được chọn."""
    return next(
        (rd for rd in raw_dataset if rd.info['subject_info']['his_id'] == selected_subject),
        None
    )

@st.cache_data
def process_eeg_files(uploaded_files, input_path, file_type):
    """Process uploaded EEG files and return loaded datasets."""
    create_directory(input_path)
    
    file_types = {".fif": ["fif"], ".set": ["set", "fdt"]}
    file_groups = defaultdict(lambda: {"set": None, "fdt": None, "fif": None})
    raw_dataset = []

    # Nhóm các file theo base name
    for file in uploaded_files:
        base_name, ext = os.path.splitext(file.name)
        ext = ext.lstrip('.').lower()
        if ext in file_types[file_type]:
            file_groups[base_name][ext] = file

    # Xử lý file
    for base_name, files in file_groups.items():
        if file_type == ".set" and files["set"] and files["fdt"]:
            temp_set_path = os.path.join(input_path, f"{base_name}.set")
            temp_fdt_path = os.path.join(input_path, f"{base_name}.fdt")

            with open(temp_set_path, "wb") as temp_set_file:
                temp_set_file.write(files["set"].getbuffer())
            with open(temp_fdt_path, "wb") as temp_fdt_file:
                temp_fdt_file.write(files["fdt"].getbuffer())

            temp_file_path = temp_set_path  # Dùng .set để đọc dữ liệu

        elif file_type == ".fif" and files["fif"]:
            temp_file_path = os.path.join(input_path, f"{base_name}.fif")
            with open(temp_file_path, "wb") as temp_fif_file:
                temp_fif_file.write(files["fif"].getbuffer())
        else:
            continue  # Bỏ qua nếu thiếu file cần thiết

        raw_data = read_eegf(temp_file_path)
        raw_dataset.append(raw_data)

    return raw_dataset

def ui_eeg_subjects_uploader(input_path):
    """UI for EEG file uploading and loading."""
    
    file_type = st.selectbox("Select file type", [".set", ".fif"])
    file_types = {".fif": ["fif"], ".set": ["set", "fdt"]}
    
    uploaded_files = st.file_uploader(
        f"Upload {file_type} file" if file_type == ".fif" else "Upload .set and .fdt files",
        type=file_types[file_type], accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing files..."):
            raw_dataset = process_eeg_files(uploaded_files, input_path, file_type)
        
        st.success(f"{len(raw_dataset)} subjects available")
        return raw_dataset

def ui_eeg_groups_uploader(input_path):
    """UI for uploading EEG files into multiple groups."""
    
    file_type = st.selectbox("Select file type", [".set", ".fif"])
    file_types = {".fif": ["fif"], ".set": ["set", "fdt"]}

    num_groups = st.text_input("Enter number of groups", "0")
    if not num_groups.isdigit():
        st.error("Please enter a valid number of groups.")
        return []
    
    num_groups = int(num_groups)
    eeg_groups = []

    for idx_group in range(num_groups):
        st.subheader(f"Group {idx_group + 1}")
        uploaded_files = st.file_uploader(
            f"Upload files for Group {idx_group + 1}",
            type=file_types[file_type], accept_multiple_files=True, key=f"uploader_{idx_group}"
        )
        
        if uploaded_files:
            with st.spinner(f"Processing files for Group {idx_group + 1}..."):
                raw_dataset = process_eeg_files(uploaded_files, input_path, file_type)
            eeg_groups.append(raw_dataset)
        else:
            eeg_groups.append([])

    if any(len(group) > 0 for group in eeg_groups):
        st.success(f"{sum(len(group) for group in eeg_groups)} subjects uploaded across {num_groups} groups.")
        return eeg_groups, num_groups
    else:
        st.error("No EEG data uploaded. Please upload valid EEG files.")
        return None, None


def ui_select_subject(raw_dataset):
    """Giao diện chọn subject từ danh sách."""
    list_subject_names = get_id_subjects(raw_dataset)

    selected_subject = st.sidebar.selectbox(
        'Choose a subject:',
        options=list_subject_names,
        index=None,
    )

    if not selected_subject:
        st.warning("Please select a subject to continue.")
        st.stop()

    raw_data = filter_subject_data(raw_dataset, selected_subject)
    return raw_data

def ui_select_channels(raw_dataset, purpose=None):
    """UI để chọn kênh EEG từ danh sách subjects"""

    if not isinstance(raw_dataset, list) or not all(hasattr(rd, "ch_names") for rd in raw_dataset):
        raw_data = raw_dataset
        selected_channels = st.sidebar.multiselect(
            'Choose EEG channels:',
            options=raw_data.ch_names,
            default=raw_data.ch_names
        )

    else:
        list_channels = set()
        for raw_data in raw_dataset:
            list_channels.update(raw_data.ch_names)
        all_channels_sorted = get_sorted_eeg_channels(list_channels)

        selected_channels = st.sidebar.multiselect(
            'Choose EEG channels:',
            options=all_channels_sorted,
            default=all_channels_sorted,
            key = purpose
        )
    
    return get_sorted_eeg_channels(selected_channels)

