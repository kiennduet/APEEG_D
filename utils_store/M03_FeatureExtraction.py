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
from utils_store.M01_DataLoader import ui_select_channels, get_id_subject, get_sorted_eeg_channels
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

def plot_combined_topomaps(df1, df2, feature_names, name_g1, name_g2, orientation='h'):
    # 1. Khởi tạo Info
    ch_names = [c.split('_')[1] for c in df1.filter(like=f"{feature_names[0]}_").columns]
    info = mne.create_info(ch_names, 250, 'eeg').set_montage('standard_1020')
    n_feat = len(feature_names)
    
    # 2. Thiết lập Grid dựa trên orientation ('h' = ngang, 'v' = dọc)
    is_h = orientation.lower() == 'h'
    nrows, ncols = (3, n_feat) if is_h else (n_feat, 3)
    figsize = (3 * n_feat, 9) if is_h else (10, 3 * n_feat)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained")
    axes = np.atleast_2d(axes)

    for i, feat in enumerate(feature_names):
        # Tính toán dữ liệu
        g1, g2 = df1.filter(like=f"{feat}_"), df2.filter(like=f"{feat}_")
        m1, m2 = g1.mean(0).values, g2.mean(0).values
        p_vals = -np.log10(np.clip(ttest_ind(g1, g2, axis=0, equal_var=False)[1], 1e-10, 1))
        _, p_glob = ttest_ind(g1.mean(1), g2.mean(1), equal_var=False)
        v = (min(m1.min(), m2.min()), max(m1.max(), m2.max()))

        # Xác định vị trí axes dựa trên hướng
        ax_list = [axes[0, i], axes[1, i], axes[2, i]] if is_h else [axes[i, 0], axes[i, 1], axes[i, 2]]
        cbar_dir = 'horizontal' if is_h else 'vertical'

        # Vẽ 3 bản đồ (G1, G2, P-map)
        im1, _ = mne.viz.plot_topomap(m1, info, axes=ax_list[0], vlim=v, show=False, cmap='RdBu_r')
        im2, _ = mne.viz.plot_topomap(m2, info, axes=ax_list[1], vlim=v, show=False, cmap='RdBu_r')
        im3, _ = mne.viz.plot_topomap(p_vals, info, axes=ax_list[2], vlim=(0, 2), show=False, cmap='Reds')

        # Thêm Colorbars
        plt.colorbar(im2, ax=ax_list[1], orientation=cbar_dir, shrink=0.7)
        cbar = plt.colorbar(im3, ax=ax_list[2], orientation=cbar_dir, shrink=0.7)
        if is_h: cbar.ax.axvline(1.301, color='black', ls='--')
        else: cbar.ax.axhline(1.301, color='black', ls='--')

        # Gán nhãn (Labels)
        if is_h:
            ax_list[0].set_title(f"{feat} (p={p_glob:.3f}{'*' if p_glob < .05 else ''})", fontweight='bold')
            if i == 0:
                for r, txt in enumerate([name_g1, name_g2, "-log10(p-Value)"]): axes[r, 0].set_ylabel(txt, fontweight='bold')
        else:
            ax_list[0].set_ylabel(f"{feat} (p={p_glob:.3f}{'*' if p_glob < .05 else ''})", fontweight='bold', size='large')
            if i == 0:
                for c, txt in enumerate([name_g1, name_g2, "-log10(p-Value)"]): axes[0, c].set_title(txt, fontweight='bold')

    return fig

def plot_feature_line(df1, df2=None, feature="CF", name_g1="G1", name_g2="G2", custom_order=None):
    all_data, stats_list = [], []
    
    # 1. Lấy và chuẩn bị dữ liệu
    for df, name in [(df1, name_g1), (df2, name_g2)]:
        if df is not None:
            tmp = select_features_from_df(df, feature).melt(var_name='Ch', value_name='Val')
            tmp['Group'], tmp['Ch'] = name, tmp['Ch'].str.replace(f'{feature}_', '', regex=False)
            stats_list.append(f"{name}: {tmp['Val'].mean():.2f} ± {tmp['Val'].std():.2f}")
            all_data.append(tmp)
    
    if not all_data: return None
    df_plot = pd.concat(all_data)
    
    # 2. SẮP XẾP CHANNEL (Sử dụng hàm get_sorted_eeg_channels)
    raw_channels = df_plot['Ch'].unique()
    sorted_order = get_sorted_eeg_channels(raw_channels, custom_order=custom_order)
    
    # Ép kiểu Categorical để seaborn vẽ theo thứ tự đã sắp xếp
    df_plot['Ch'] = pd.Categorical(df_plot['Ch'], categories=sorted_order, ordered=True)
    df_plot = df_plot.sort_values('Ch') # Đảm bảo lineplot nối các điểm theo đúng thứ tự

    # 3. Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df_plot, x='Ch', y='Val', hue='Group' if df2 is not None else None, 
                 marker='o', errorbar='sd', palette='tab10', ax=ax)
    
    # 4. Cấu hình hiển thị
    ax.set_title(f"{feature} ({ '  |  '.join(stats_list) })", fontweight='bold')
    ax.set(xlabel='EEG Channels (Sorted)', ylabel='Value')
    ax.grid(True, ls='--', alpha=0.5)
    plt.xticks(rotation=45)
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

    topo_2g = plot_combined_topomaps(df1, df2, feature_names,name_g1, name_g2,orientation='h')
    st.pyplot(topo_2g)

def ui_plot_feature_line(df_g1, selected_features, df_g2=None, name_g1="Group 1", name_g2="Group 2"):
    st.subheader("Line Plot Analysis", divider="gray")

    custom_order = ['F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5',
                    'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    for feature in selected_features:
        fig = plot_feature_line(df1=df_g1, df2=df_g2, feature=feature, 
                                name_g1=name_g1, name_g2=name_g2,custom_order=custom_order)
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



