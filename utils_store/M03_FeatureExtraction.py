import numpy as np
import mne
import pandas as pd
from fooof import FOOOF
from dataclasses import dataclass
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import streamlit as st
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


def get_features(df, feature_names):
    """
    Lấy các cột liên quan đến feature_name. 
    Chấp nhận feature_name là string hoặc list.
    """
    if isinstance(feature_names, list):
        # Nếu là list, tạo regex: ^(CF_|BW_|...)
        pattern = '|'.join([f'^{f}_' for f in feature_names])
        # pattern = f'^{feature_names}_'
        return df.filter(regex=pattern)
    else:
        # Nếu là string, dùng like cho chắc chắn (giống cách bạn test thành công)
        return df.filter(like=feature_names)



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
    
    return get_features(df_features_subjects, feature_names=selected_features)


def plot_topomap_features(raw_dataset, df_features_subjects, feature_names):

    sample_feat = df_features_subjects.columns[0].split('_')[0]
    selected_channels = [c.replace(f"{sample_feat}_", "") for c in df_features_subjects.filter(like=f"{sample_feat}_").columns]

    info = mne.create_info(ch_names=selected_channels, ch_types='eeg', sfreq=250)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    fig, axes = plt.subplots(nrows=1, ncols=len(feature_names), figsize=( 4*len(feature_names), 5))
    
    if len(feature_names) == 1:
        axes = [axes]
    
    for ax, feature_name in zip(axes, feature_names):
        df_selected_feature = get_features(df_features_subjects, feature_name)
        mean_feature_channels = df_selected_feature.mean(axis=0)
        mean_feature_values = mean_feature_channels.to_numpy()
        vmin, vmax = mean_feature_values.min(), mean_feature_values.max()
        im, _ = mne.viz.plot_topomap(mean_feature_values, info, axes=ax, show=False, cmap='RdBu_r', vlim=(vmin, vmax))
        ax.set_title(feature_name)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

    mean_feature_filtered = df_features_subjects.filter(regex='|'.join(feature_names)).mean(axis=0)
    
    return fig, mean_feature_filtered
    
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

def ui_plot_topo(raw_dataset, df_features_subjects, selected_features):
    topo_fig, mean_features_channels = plot_topomap_features(raw_dataset=raw_dataset, 
                                                    df_features_subjects=df_features_subjects, 
                                                    feature_names=selected_features)
    st.subheader("", divider="gray")
    st.subheader("Topographic Plot")
    st.pyplot(topo_fig)

    return topo_fig
    

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


