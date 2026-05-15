import mne
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from fooof import FOOOF
from typing import List
import plotly.graph_objects as go
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


def plot_raw_eeg_plotly(raw_data, channel_names, tmin=0, tmax=None, amplitude_scale=1e6):
    """
    Plot raw EEG in time domain using Plotly (interactive).

    Parameters
    ----------
    raw_data        : MNE raw object
    channel_names   : list of channel names to plot
    tmin            : start time (seconds)
    tmax            : end time (seconds); if None, uses duration from tmin
    amplitude_scale : µV (default 1e6)
    """
    if tmax is None:
        tmax = tmin + 10  # Default 10-second window

    # Lấy data
    data, times = raw_data.get_data(
        picks=channel_names, tmin=tmin, tmax=tmax, return_times=True)

    # Scale to µV
    data = data * amplitude_scale

    # Tạo Plotly figure
    fig = go.Figure()

    # Thêm một trace per channel với offset
    for i, ch_name in enumerate(channel_names):
        offset = i * (data.max() - data.min()) * 0.5
        fig.add_trace(go.Scatter(
            x=times, y=data[i] + offset,
            mode='lines', name=ch_name,
            line=dict(width=1),
            hovertemplate=f'{ch_name}<br>Time: %{{x:.3f}}s<br>Amp: %{{y:.2f}}µV<extra></extra>'
        ))

    fig.update_layout(
        title=f'Raw EEG Time Domain ({get_id_subject(raw_data)})',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (µV) — offset per channel',
        height=500,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')

    return fig


def UI_plot_raw_eeg(raw_data):
    """UI for time-domain EEG visualization."""
    st.subheader("Raw EEG Time Domain", divider="gray")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_sec = st.number_input("Start time (s):", value=0.0, min_value=0.0,
                                     max_value=float(raw_data.times[-1] - 1))
    with col2:
        duration = st.number_input("Duration (s):", value=10.0, min_value=0.1,
                                   max_value=float(raw_data.times[-1] - start_sec))
    with col3:
        channels = st.multiselect("Channels:", raw_data.ch_names,
                                  default=raw_data.ch_names[:8], key="raw_eeg_ch")

    if channels and st.button("▶ Plot Time Domain", use_container_width=True, key="btn_raw"):
        fig = plot_raw_eeg_plotly(raw_data, channels,
                                   tmin=start_sec, tmax=start_sec + duration)
        st.plotly_chart(fig, use_container_width=True)


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



    