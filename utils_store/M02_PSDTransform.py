import mne
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from fooof import FOOOF
from typing import List
from utils_store.M01_DataLoader import ui_select_channels, get_id_subject, get_sorted_eeg_channels


class PSDSettings:
    def __init__(self, f_range=(0.1, 45), n_per_seg=512, n_fft=512, window='hamming'):
        self.f_range = f_range
        self.n_per_seg = n_per_seg
        self.n_fft = n_fft
        self.window = window  # Thêm tham số cửa sổ

def psd_trans(raw_data, psd_settings: PSDSettings, unitV=True):
    """
    Computes the Power Spectral Density (PSD) of raw EEG data using Welch's method.
    """
    if unitV:
        raw_data = raw_data.copy().apply_function(lambda x: x * 1e6)
    
    psd_result = raw_data.compute_psd(
        method='welch', fmin=psd_settings.f_range[0], fmax=psd_settings.f_range[1],
        n_fft=psd_settings.n_fft, n_per_seg=psd_settings.n_per_seg, window=psd_settings.window)
    
    return psd_result.freqs, psd_result.get_data()


def plot_psd(raw_data, freqs, psds, channel_names):
    """
    Plot the Power Spectral Density (PSD) of raw EEG data using Welch's method.
    """
    sub_id = get_id_subject(raw_data=raw_data)
    ch_indices = [raw_data.ch_names.index(ch_name) for ch_name in channel_names if ch_name in raw_data.ch_names]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ch_indices)))
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for idx, ch_idx in enumerate(ch_indices):
        psd_channel = psds[ch_idx]
        ax.plot(freqs, np.log10(psd_channel), color=colors[idx], label=channel_names[idx])
        
    ax.set_title(f'Power Spectral Density (Welch): {sub_id}')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Log Power [dB]')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    ax.grid(True, which='both', color='#c6c6c6', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    
    return fig

def ui_adjust_param_psd(purpose=None):
    st.sidebar.header("", divider="gray")
    st.sidebar.subheader("PSD Transform Adjustments")

    psd_choice = st.sidebar.selectbox("Parameters:", ["Default", "Custom"], key=purpose)
    
    if psd_choice == "Custom":
        f_range = st.sidebar.select_slider('Frequency Range (Hz):', options=list(range(0, 126)), value=(0, 45), key=purpose)
        n_per_seg = st.sidebar.slider('Number per segments:', value=512, min_value=64, max_value=1024, step=64, key=purpose)
        n_fft = st.sidebar.slider('Number of FFT points:', value=512, min_value=64, max_value=1024, step=64, key=purpose)
        window = st.sidebar.selectbox("Window Function:", ["hann", "hamming", "blackman", "bartlett", "flattop"], key=purpose)
    else:
        f_range, n_per_seg, n_fft, window = (0.1, 45), 512, 512, "hamming"

    return PSDSettings(f_range=f_range, n_per_seg=n_per_seg, n_fft=n_fft, window=window)


def UI_plot_psd(raw_data):
    """UI xử lý biến đổi PSD"""
    st.sidebar.header("", divider="orange")
    st.sidebar.header(":orange[Transform to Frequency Domain]")
    
    selected_channels = ui_select_channels(raw_data, purpose="PSD")
    psd_settings = ui_adjust_param_psd()

    st.sidebar.header("", divider="orange")

    freqs, psds = psd_trans(raw_data=raw_data, psd_settings=psd_settings)
    psd_fig = plot_psd(raw_data=raw_data, freqs=freqs, psds=psds, channel_names=selected_channels)
    
    st.subheader("", divider="rainbow")
    st.subheader("Power Spectrum Plot")
    st.pyplot(psd_fig)

    return freqs, psds, selected_channels



    