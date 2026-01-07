import numpy as np
import mne
import pandas as pd
import seaborn as sns
from fooof import FOOOF
from dataclasses import dataclass
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import ttest_ind
from utils_store.M01_DataLoader import ui_select_channels, get_id_subject
from utils_store.M02_PSDTransform import ui_adjust_param_psd, psd_trans

class PSDSettings:
    def __init__(self, f_range=(0.1, 45), n_per_seg=512, n_fft=512):
        self.f_range = f_range
        self.n_per_seg = n_per_seg
        self.n_fft = n_fft

@dataclass
class FOOOFSettings:
    f_range: List[int]
    peak_width_limits: List[int]
    max_n_peaks: int
    min_peak_height: float
    peak_threshold: float
    aperiodic_mode: str

def select_features_from_df(df, feature_names):
    if isinstance(feature_names, list):
        pattern = '|'.join([f'^{f}_' for f in feature_names])
        return df.filter(regex=pattern)
    else:
        return df.filter(like=feature_names)
    
def select_channels_from_df(df, selected_channels):
    """
    Tạo DataFrame mới chỉ chứa các cột thuộc danh sách selected_channels.
    selected_channels: list các tên kênh (ví dụ ['FP1', 'F3'])
    """
    # Chuyển selected_channels về chữ hoa để so sánh chính xác
    selected_channels = [ch.upper() for ch in selected_channels]
    # Lọc các cột có phần Channel (sau dấu _) nằm trong danh sách đã chọn
    new_columns = [
        col for col in df.columns 
        if col.split('_', 1)[-1].upper() in selected_channels
    ]
    return df[new_columns]    

def get_fooof_settings(setting_type):
    st.sidebar.header(f"{setting_type} Setting:", divider="gray")
    if setting_type:
        st.sidebar.header(f"{setting_type} Setting: ", divider="gray")

    settings_params = {
        'f_range': st.sidebar.select_slider(f'{setting_type}: Frequency Range (Hz):', options=list(range(0, 125)), value=(0, 40)),
        'peak_width': st.sidebar.select_slider(f'{setting_type}: Peak Width Range (Hz):', options=list(range(0, 50)), value=(1, 20)),
        'max_n_peaks': st.sidebar.slider(f'{setting_type}: Max Number of Peaks', value=10, min_value=1, max_value=100),
        'min_peak_height': st.sidebar.slider(f'{setting_type}: Min Peak Height', 0.0, 1.0, 0.1),
        'peak_threshold': st.sidebar.slider(f'{setting_type}: Peak Threshold', -20.0, 0.0, -10.0),
        'aperiodic_mode': st.sidebar.selectbox(f'{setting_type}: Aperiodic Mode', ['fixed', 'knee'])
    }

    return FOOOFSettings(
        f_range=[settings_params['f_range'][0] + 0.01, settings_params['f_range'][1]],  # Truy cập tuple f_range
        peak_width_limits=[settings_params['peak_width'][0], settings_params['peak_width'][1]],  # Truy cập tuple peak_width
        max_n_peaks=settings_params['max_n_peaks'],
        min_peak_height=settings_params['min_peak_height'],
        peak_threshold=settings_params['peak_threshold'],
        aperiodic_mode=settings_params['aperiodic_mode']
    )    

def ext_features_subject(
                raw_data: mne.io.Raw, 
                freqs: np.ndarray, psds: np.ndarray, 
                channel_names: List[str],
                pe_settings: Optional[FOOOFSettings] = None,
                ape_settings: Optional[FOOOFSettings] = None):
    """
    Extract features from the PSD of one patient using FOOOF for both periodic and aperiodic components.

    Parameters:
    - raw_data: mne.io.Raw
        The raw EEG data containing the channel information.
    - freqs: np.ndarray
        Array of frequency values.
    - psds: np.ndarray
        Array of PSD values corresponding to each frequency.

    - channel_names: List[str]
        List of channel names to extract features from. If None, all channels will be processed.

    Returns:
    - features: Extracted features for all channels as a flattened array.

    """
    # Default settings
    pe_settings = pe_settings or FOOOFSettings(f_range=[4, 16], peak_width_limits=[1, 20], max_n_peaks=1,
                                               min_peak_height=0.01, peak_threshold=-5, aperiodic_mode='fixed')

    ape_settings = ape_settings or FOOOFSettings(f_range=[0.5, 40], peak_width_limits=[1, 20], max_n_peaks=int(1e10),
                                                 min_peak_height=0.1, peak_threshold=-10, aperiodic_mode='fixed')

    fm_periodic = FOOOF(peak_width_limits=pe_settings.peak_width_limits, max_n_peaks=pe_settings.max_n_peaks,
                        min_peak_height=pe_settings.min_peak_height, peak_threshold=pe_settings.peak_threshold,
                        aperiodic_mode=pe_settings.aperiodic_mode)

    fm_aperiodic = FOOOF(peak_width_limits=ape_settings.peak_width_limits, max_n_peaks=ape_settings.max_n_peaks,
                         min_peak_height=ape_settings.min_peak_height, peak_threshold=ape_settings.peak_threshold,
                         aperiodic_mode=ape_settings.aperiodic_mode)


    features = [np.full(5, np.nan) for _ in range(len(channel_names))]  # Pre-fill with NaN for all channels
    for idx, ch_name in enumerate(channel_names):
        if ch_name in raw_data.ch_names:
            ch_idx = raw_data.ch_names.index(ch_name)

            fm_periodic.fit(freqs, psds[ch_idx], pe_settings.f_range)
            fm_aperiodic.fit(freqs, psds[ch_idx], ape_settings.f_range)

            aperiodic_params = fm_aperiodic.get_params('aperiodic_params').flatten()
            peak_params = fm_periodic.get_params('peak_params').flatten()
            peak_params[np.isnan(peak_params)] = 0

            features[idx]  = np.concatenate((peak_params, aperiodic_params))

    features_array = np.array(features).flatten()
    return features_array #xem xet xuat settings

def ext_features_subjects(raw_dataset: List[mne.io.Raw], 
                          psd_settings:PSDSettings,
                          channel_names, selected_features,
                          pe_settings: FOOOFSettings, 
                          ape_settings: FOOOFSettings):
    
    features_subjects = []
    id_subjects = []

    for raw_data in raw_dataset:
        id_subject = get_id_subject(raw_data=raw_data)
        id_subjects.append(id_subject)

        freqs, psds = psd_trans(raw_data=raw_data, psd_settings=psd_settings)
        features_subject = ext_features_subject(raw_data=raw_data, freqs=freqs, psds=psds, 
                                                channel_names=channel_names, 
                                                pe_settings=pe_settings,ape_settings=ape_settings)
        
        features_subjects.append(features_subject.flatten())
        
    feature_names = ['CF', 'BW', 'PW', 'OS', 'EXP']
    column_labels = [f"{feature}_{ch}" for ch in channel_names for feature in feature_names]

    df_features_subjects = pd.DataFrame(features_subjects, index=id_subjects, columns=column_labels)
    
    return select_features_from_df(df_features_subjects, feature_names=selected_features)

def plot_fooof(freqs: np.ndarray, psds: np.ndarray, raw_data=None, 
                channel_names=None, single_channel_mode=False, show_fplot=True,
                pe_settings: Optional[FOOOFSettings] = None,
                ape_settings: Optional[FOOOFSettings] = None) -> Tuple[np.ndarray, Optional[plt.Figure]]:

    pe_settings = pe_settings or FOOOFSettings(f_range=[4, 16], peak_width_limits=[1, 20], max_n_peaks=1,
                                               min_peak_height=0.01, peak_threshold=-5, aperiodic_mode='fixed')

    ape_settings = ape_settings or FOOOFSettings(f_range=[0.5, 40], peak_width_limits=[1, 20], max_n_peaks=int(1e10),
                                                 min_peak_height=0.1, peak_threshold=-10, aperiodic_mode='fixed')

    fm_periodic = FOOOF(peak_width_limits=pe_settings.peak_width_limits, max_n_peaks=pe_settings.max_n_peaks,
                        min_peak_height=pe_settings.min_peak_height, peak_threshold=pe_settings.peak_threshold,
                        aperiodic_mode=pe_settings.aperiodic_mode)

    fm_aperiodic = FOOOF(peak_width_limits=ape_settings.peak_width_limits, max_n_peaks=ape_settings.max_n_peaks,
                         min_peak_height=ape_settings.min_peak_height, peak_threshold=ape_settings.peak_threshold,
                         aperiodic_mode=ape_settings.aperiodic_mode)

    # Determine which channels to process
    if channel_names and raw_data:
        ch_indices = [raw_data.ch_names.index(ch_name) for ch_name in channel_names if ch_name in raw_data.ch_names]
        if len(channel_names) == 1:
            single_channel_mode = True
            print("\nSingle Channel Mode\n")
    elif channel_names is None and raw_data is None:
        ch_indices = range(len(psds))

    features = []
    for ch_idx in ch_indices:
        fm_periodic.fit(freqs, psds[ch_idx], pe_settings.f_range)
        fm_aperiodic.fit(freqs, psds[ch_idx], ape_settings.f_range)

        rsquare_of_fit_periodic = fm_periodic.get_params('r_squared') if fm_periodic.has_model else None
        rsquare_of_fit_aperiodic = fm_aperiodic.get_params('r_squared') if fm_aperiodic.has_model else None

        aperiodic_params = fm_aperiodic.get_params('aperiodic_params')
        peak_params = fm_periodic.get_params('peak_params').flatten()
        peak_params[np.isnan(peak_params)] = 0 # Replace NaN values in peak parameters with 0

        features.extend(np.concatenate((peak_params, aperiodic_params)))
    features = np.array([float(f) for f in features])
    print(features)

    fig = None
    if single_channel_mode and show_fplot:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        fm_periodic.plot(ax=axs[0])
        fm_aperiodic.plot(ax=axs[1])

        axs[0].set_title(f'Periodic Fitting Result in Channel {channel_names[0]}')
        axs[1].set_title(f'Aperiodic Fitting Result in Channel {channel_names[0]}')
        axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs[0].text(0.65, 0.55, f"CF = {features[0]:.3f} \nPW = {features[1]:.3f} \nBW = {features[2]:.3f} \nGoodness: {rsquare_of_fit_periodic:.3f}", 
                    transform=axs[0].transAxes, fontsize=15, color='black')
        axs[1].text(0.65, 0.55, f"OS = {features[3]:.3f} \nEXP = {features[4]:.3f} \nGoodness: {rsquare_of_fit_aperiodic:.3f}", 
                    transform=axs[1].transAxes, fontsize=15, color='black')    
    return features, fig

def plot_topomap_features(df_features_subjects, feature_names):

    sample_feat = df_features_subjects.columns[0].split('_')[0]
    selected_channels = [c.replace(f"{sample_feat}_", "") for c in df_features_subjects.filter(like=f"{sample_feat}_").columns]

    info = mne.create_info(ch_names=selected_channels, ch_types='eeg', sfreq=250)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    fig, axes = plt.subplots(nrows=1, ncols=len(feature_names), figsize=( 4*len(feature_names), 5))
    if len(feature_names) == 1:
        axes = [axes]
    
    for ax, feature_name in zip(axes, feature_names):
        df_selected_feature = select_features_from_df(df_features_subjects, feature_name)
                
        all_values = df_selected_feature.to_numpy().flatten()
        global_mean = np.nanmean(all_values)
        global_std = np.nanstd(all_values)

        mean_feature_channels = df_selected_feature.mean(axis=0)
        mean_feature_values = mean_feature_channels.to_numpy()
        vmin, vmax = mean_feature_values.min(), mean_feature_values.max()
        im, _ = mne.viz.plot_topomap(mean_feature_values, info, axes=ax, show=False, cmap='RdBu_r', vlim=(vmin, vmax))
        ax.set_title(f"{feature_name} [{global_mean:.2f}±{global_std:.2f}]")
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
    
    return fig

def plot_two_topomaps_features(df_features_group1, df_features_group2, feature_names, name_g1, name_g2):
    sample_feat = df_features_group1.columns[0].split('_')[0]
    selected_channels = [c.replace(f"{sample_feat}_", "") for c in df_features_group1.filter(like=f"{sample_feat}_").columns]

    info = mne.create_info(ch_names=selected_channels, ch_types='eeg', sfreq=250)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    num_features = len(feature_names)
    fig, axes = plt.subplots(nrows=2, ncols=num_features, figsize=(4 * num_features, 6))

    if num_features == 1:
        axes = np.array([axes]).reshape(2, 1)

    all_values_for_scaling = []
    mean_features_group1_dict = {}
    mean_features_group2_dict = {}
    p_value_features = {}

    for feature_name in feature_names:
        
        feature_group1 = select_features_from_df(df_features_group1, feature_name)
        feature_group2 = select_features_from_df(df_features_group2, feature_name)

        # Mean of all subjects (NxChannels => 1xChannels)
        mean_feature_group1_channels = feature_group1.mean(axis=0)
        mean_feature_group1_values = mean_feature_group1_channels.to_numpy()
        mean_features_group1_dict[feature_name] = mean_feature_group1_values

        mean_feature_group2_channels = feature_group2.mean(axis=0)
        mean_feature_group2_values = mean_feature_group2_channels.to_numpy()
        mean_features_group2_dict[feature_name] = mean_feature_group2_values

        all_values_for_scaling.extend(mean_feature_group1_values)
        all_values_for_scaling.extend(mean_feature_group2_values)

        # Subject-wise average (NxChannels => Nx1)
        avg_subjects_g1 = feature_group1.mean(axis=1)
        avg_subjects_g2 = feature_group2.mean(axis=1)
        t_stat, p_val = ttest_ind(avg_subjects_g1, avg_subjects_g2, equal_var=False)
        p_value_features[feature_name] = p_val

    for i, feature_name in enumerate(feature_names):
        current_feature_values = np.concatenate([
            mean_features_group1_dict[feature_name],
            mean_features_group2_dict[feature_name]
        ])
        vmin, vmax = current_feature_values.min(), current_feature_values.max()

        # p-value
        p_value = p_value_features[feature_name]
        p_str = f"p={p_value:.3f}" if p_value >= 0.001 else "p < .001"
        significance = "*" if p_value < 0.05 else ""

        axes[0, i].set_title(f'{feature_name} ({p_str}{significance})')
        
        im1, _ = mne.viz.plot_topomap(mean_features_group1_dict[feature_name], info, axes=axes[0, i], show=False, cmap='RdBu_r', vlim=(vmin, vmax))
        if i == 0: axes[0, i].set_ylabel(name_g1, rotation=90, size='large', labelpad=10)

        im2, _ = mne.viz.plot_topomap(mean_features_group2_dict[feature_name], info, axes=axes[1, i], show=False, cmap='RdBu_r', vlim=(vmin, vmax))
        if i == 0: axes[1, i].set_ylabel(name_g2, rotation=90, size='large', labelpad=10)

        cbar_ax = fig.add_axes([axes[1, i].get_position().x0, 0.01, axes[1, i].get_position().width, 0.02])
        plt.colorbar(im2, cax=cbar_ax, orientation='horizontal')

    return fig

def plot_p_value_topomap(df1, df2, feature_names):
    sample_feat = df1.columns[0].split('_')[0]
    selected_channels = [c.replace(f"{sample_feat}_", "") for c in df1.filter(like=f"{sample_feat}_").columns]

    info = mne.create_info(ch_names=selected_channels, ch_types='eeg', sfreq=250)
    info.set_montage(mne.channels.make_standard_montage('standard_1020'))

    fig, axes = plt.subplots(nrows=1, ncols=len(feature_names), figsize=(4 * len(feature_names), 6))
    if len(feature_names) == 1: axes = [axes]

    for i, feat in enumerate(feature_names):
        p_values_feat = []
        for ch in selected_channels:
            col_name = f"{feat}_{ch}"
            
            group1 = df1[col_name]
            group2 = df2[col_name]
            
            # Welch's t-test (equal_var=False)
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
            # Tránh p-value bằng 0 dẫn đến log = vô cực
            p_val = max(p_val, 1e-10) 
            # -log10(p)
            neg_log_p = -np.log10(p_val)
            p_values_feat.append(neg_log_p)

        p_values_feat = np.array(p_values_feat)

        im, _ = mne.viz.plot_topomap(
            p_values_feat, info, axes=axes[i], show=False, 
            cmap='Reds',
            # vlim=(p_values_feat.min(), p_values_feat.max())
            vlim=(0, 2))
        
        if i == 0: axes[i].set_ylabel("-log10(p-Value)", rotation=90, size='large', labelpad=10)              
        cbar = plt.colorbar(im, ax=axes[i], orientation='horizontal', pad=0.1, shrink=0.8)
        cbar.ax.axvline(1.301, color='black', linestyle='--')

    plt.tight_layout()
    return fig

def plot_feature_line(df1, df2=None, feature="CF", name_g1="G1", name_g2="G2"):
    all_data, stats_list = [], []
    
    # 1. Loop xử lý dữ liệu và tính toán thống kê
    # Kiểm tra 'is not None' ở đây là đúng
    for df, name in [(df1, name_g1), (df2, name_g2)]:
        if df is not None: 
            # Lọc feature và chuyển dạng Long
            tmp = select_features_from_df(df, feature).melt(var_name='Ch', value_name='Val')
            tmp['Group'] = name
            tmp['Ch'] = tmp['Ch'].str.replace(f'{feature}_', '', regex=False)
            
            # Tính Mean ± Std
            stats_list.append(f"{name}: {tmp['Val'].mean():.2f} ± {tmp['Val'].std():.2f}")
            all_data.append(tmp)
    
    if not all_data: return None # Tránh lỗi nếu không có dữ liệu nào

    df_plot = pd.concat(all_data)
    
    # 2. Sắp xếp Channel
    order = [c.replace(f'{feature}_', '') for c in df1.filter(like=f'{feature}_').columns]
    df_plot['Ch'] = pd.Categorical(df_plot['Ch'], categories=order, ordered=True)
    
    # 3. Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # --- SỬA LỖI TẠI DÒNG NÀY ---
    # Thay 'if df2' thành 'if df2 is not None'
    use_hue = 'Group' if df2 is not None else None
    
    sns.lineplot(
        data=df_plot, x='Ch', y='Val', 
        hue=use_hue, 
        marker='o', errorbar='sd', palette='tab10', ax=ax
    )
    
    # 4. Cấu hình hiển thị
    ax.set_title(f"{feature} ({ '  |  '.join(stats_list) })", fontweight='bold')
    ax.set(xlabel='EEG Channels', ylabel='Value')
    ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    
    return fig

def ui_adjust_param_fooof():
    st.sidebar.header("", divider="gray")
    st.sidebar.subheader("FOOOF Adjustments")
    if st.sidebar.selectbox("Parameters:", ["Default", "Periodic and Aperiodic"]) == "Periodic and Aperiodic":
        pe_settings = get_fooof_settings("Periodic")
        ape_settings = get_fooof_settings("Aperiodic")
        return pe_settings, ape_settings
    return None, None

def ui_select_feature():
    selected_features = st.multiselect(
        "Choose features:", 
        options = ['CF', 'BW', 'PW', 'OS', 'EXP'], 
        default = ['CF', 'BW', 'PW', 'OS', 'EXP'])
    return selected_features

def ui_plot_topo(df_features_subjects, selected_features):
    st.subheader("", divider="gray")
    st.subheader("Topographic Plot")
    topo_fig = plot_topomap_features(df_features_subjects=df_features_subjects, feature_names=selected_features)
    st.pyplot(topo_fig)
    
def ui_plot_topo_2group(df1, df2, feature_names, name_g1, name_g2):
    st.header("", divider="rainbow")
    st.header(":orange[Topographic plot]")
    
    topo_fig = plot_two_topomaps_features(df1, df2, feature_names,
                                          name_g1, name_g2)
    st.pyplot(topo_fig)
    
    topo_p_fig = plot_p_value_topomap(df1, df2, feature_names)
    st.pyplot(topo_p_fig)

def ui_plot_feature_line(df_g1, selected_features, df_g2=None, name_g1="Group 1", name_g2="Group 2"):
    st.subheader("Line Plot Analysis", divider="gray")
    for feature in selected_features:
        fig = plot_feature_line(df1=df_g1, df2=df_g2, feature=feature, 
                                name_g1=name_g1, name_g2=name_g2)
        st.pyplot(fig)

def UI_feature_extraction(raw_dataset, selected_features):
    st.sidebar.header("", divider="orange")
    st.sidebar.header(":orange[Feature Extraction]")

    selected_channels = ui_select_channels(raw_dataset=raw_dataset)
    psd_settings = ui_adjust_param_psd()
    pe_settings, ape_settings = ui_adjust_param_fooof()

    if st.sidebar.button("🚀 Run Feature Extraction", use_container_width=True):
        with st.spinner("Extracting features... please wait."):
            features_subjects = ext_features_subjects(
                raw_dataset=raw_dataset, psd_settings=psd_settings,
                channel_names=selected_channels, selected_features = selected_features,
                pe_settings=pe_settings, ape_settings=ape_settings
                )
            st.subheader("Fitting Results")
            st.dataframe(features_subjects)
            return features_subjects
    st.sidebar.header("", divider="orange")
    return None

def UI_feature_extraction_groups(eeg_groups, num_groups):
    #Have not use yet
    st.sidebar.header("", divider="orange")
    st.sidebar.header(":orange[Feature Extraction]")
    
    selected_channels = ui_select_channels(raw_dataset=eeg_groups[0])
    psd_settings = ui_adjust_param_psd()
    pe_settings, ape_settings = ui_adjust_param_fooof()

    st.subheader("", divider="rainbow")
    st.subheader("Fitting Results")

    features_subjects_groups = []
    for i in range(num_groups):
        raw_dataset = eeg_groups[i]
        features_subjects = ext_features_subjects(raw_dataset=raw_dataset,
                                            psd_settings=psd_settings,
                                            channel_names=selected_channels,
                                            pe_settings=pe_settings, 
                                            ape_settings=ape_settings)
        
        st.markdown(f"Group {i+1}")
        st.dataframe(features_subjects)
        features_subjects_groups.append(features_subjects)
    
    return features_subjects_groups



