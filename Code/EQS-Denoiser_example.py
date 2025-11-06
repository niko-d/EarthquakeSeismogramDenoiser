"""
EQS-Denoiser

Author: Niko Dahmen
Email: nikolaj.dahmen@eaps.ethz.ch
Affiliation: ETH Zurich
Date: 2025-08-11
Version: 1.0
"""

import sys
sys.path.append("/home/niko/Schreibtisch/EQ_denoising/Submission/Material/Code")

import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import tensorflow as tf
from obspy import UTCDateTime
import scipy

import numpy as np
from DenoisingFunctions_public import client, normalize_percentile
from obspy.clients.fdsn import Client

# LOAD MODEL
model = tf.keras.models.load_model("/home/niko/Schreibtisch/EQ_denoising/Submission/Material/Models/model_1000k_onlyweights.keras", compile=False)

def get_data(network, station, channel, client, utc_start):
    """
    Retrieve and preprocess seismic waveform data.
    Parameters
    ----------
    network : str
        Seismic network code (e.g., "XB").
    station : str
        Station code.
    channel : str
        Channel prefix (e.g., "BH").
    client : obspy.clients.fdsn.Client
        ObsPy FDSN client for waveform retrieval.
    utc_start : obspy.UTCDateTime
        Reference start time for the waveform.
    Returns
    -------
    stream : obspy.Stream
        Preprocessed three-component velocity stream at 100 Hz.
    """

    duration = 61.2

    pre_filt = [1 / 100, 1 / 20, 45, 50]
    # pre_filt = [1 / 100, 1 / 20, 45, 47]
    time_window = [duration/2, duration/2]
    t_pick = utc_start + time_window[0]
    buffer = 30
    # GET DATA
    stream = client.get_waveforms(network=network, station=station, channel=channel + "?",location='*',
                                  starttime=t_pick - time_window[0] - buffer, endtime=t_pick + time_window[1] + buffer,
                                  attach_response=True).merge()

    # SELECT PREFERRED SENSOR, REMOVE RESPONSE, DOWNSAMPLE, TRIM TO MATCHING WINDOW
    stream = stream.select(location=stream[0].stats.location)

    stream.remove_response(output="VEL", pre_filt=pre_filt, water_level=None)# 60

    if stream[0].stats.sampling_rate > 100:
        stream.resample(100)  # pre filter

    stream.trim(stream[0].stats.starttime+buffer, stream[0].stats.endtime-buffer)


    return stream

def wv2stft(stream):
    """
      Convert waveform data to normalized short-time Fourier transform (STFT) coefficient tensors.
      Parameters
      ----------
      stream : obspy.Stream
          Three-component velocity waveform stream (Z, N, E).
      Returns
      -------
      t : np.ndarray
          Time vector of STFT.
      f : np.ndarray
          Frequency vector of STFT.
      stft_tmp : np.ndarray
          Raw STFT array of shape (1, 64, 256, 6) containing real/imag components.
      stft_tmp_norm : np.ndarray
          Normalized STFT array of shape (1, 64, 256, 6).
      """
    stft_parameters = {"nperseg": 48, "nfft": 126, "fs": 100, "noverlap": 24}

    components = []
    for trace in stream:
        components.append(trace.stats.channel[2])
    components.sort(reverse=True)
    stream.sort(reverse=True)

    # get waveform data
    data_z = stream.select(component=components[0])[0].data[:6120]
    data_n = stream.select(component=components[1])[0].data[:6120]
    data_e = stream.select(component=components[2])[0].data[:6120]

    # GET STFT MODEL INPUT, NORM
    stft_tmp = np.zeros((64, 256, 6))
    stft_tmp_norm = np.zeros((64, 256, 6))

    for c, data in enumerate([data_z, data_n, data_e]):
        f, t, _stft = scipy.signal.stft(data, **stft_parameters)
        stft_tmp_1c = np.stack((_stft.real, _stft.imag), axis=2)
        stft_tmp_1c_norm = normalize_percentile(stft_tmp_1c)

        # RAW DATA
        stft_tmp[:, :, c * 2] = stft_tmp_1c[:, :, 0]
        stft_tmp[:, :, c * 2 + 1] = stft_tmp_1c[:, :, 1]
        # NORM DATA
        stft_tmp_norm[:, :, c * 2] = stft_tmp_1c_norm[:, :, 0]
        stft_tmp_norm[:, :, c * 2 + 1] = stft_tmp_1c_norm[:, :, 1]

    stft_tmp = np.expand_dims(stft_tmp, axis=0)
    stft_tmp_norm = np.expand_dims(stft_tmp_norm, axis=0)

    return t, f, stft_tmp, stft_tmp_norm


def plot_stft(t_stft, f_stft, stft_tmp, label="Preprocessd and normalised data ", components = ['Z', '1', '2']):
    """
    Plot STFT spectrograms for three components.
    Parameters
    ----------
    t_stft : np.ndarray
        STFT time vector.
    f_stft : np.ndarray
        STFT frequency vector.
    stft_tmp : np.ndarray
        STFT tensor of shape (1, 64, 256, 6).
    label : str, optional
        Plot title prefix.
    components : list of str, optional
        Component labels.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)


    channel_indices = [0, 2, 4]  # real parts only

    for ax, comp, ch in zip(axes, components, channel_indices):
        ampl = np.log10(np.abs(stft_tmp[0, :, :, ch]+1j*stft_tmp[0, :, :, ch+1]))
        im = ax.pcolormesh(t_stft, f_stft, ampl, shading='auto')
        ax.set_ylabel(f"{comp} - Freq (Hz)")
        fig.colorbar(im, ax=ax, orientation="vertical", label="log10 Amplitude")

    plt.suptitle(label + str(wv[0].id[:-1]) + "".join(components))

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_masks(y_predict, t_stft, f_stft):
    """
    Plot predicted event masks for three components.
    Parameters
    ----------
    y_predict : np.ndarray
        Predicted mask tensor of shape (1, 64, 256, 3).
    t_stft : np.ndarray
        STFT time vector.
    f_stft : np.ndarray
        STFT frequency vector.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    components = ['Z', '1', '2']
    channel_indices = [0, 1, 2]  # assuming y_predict has these as components

    for ax, comp, ch in zip(axes, components, channel_indices):
        im = ax.pcolormesh(t_stft, f_stft, y_predict[0, :, :, ch],
                           vmin=0, vmax=1, cmap="cubehelix_r", shading='auto')
        ax.set_ylabel(f"{comp} - Freq (Hz)")
        fig.colorbar(im, ax=ax, orientation="vertical", label="Event mask")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.suptitle("Predicted event mask " + str(wv[0].id[:-1]) + "".join(components), y=1.0)

    plt.show()


def plot_denoiseddata(t_stft, f_stft, stft_tmp, y_predict, components = ['Z', '1', '2'], plot_timeseries=True):
    """
    Plot denoised seismic data in either time or time-frequency domain.

    Parameters
    ----------
    t_stft : np.ndarray
        STFT time vector.
    f_stft : np.ndarray
        STFT frequency vector.
    stft_tmp : np.ndarray
        Input STFT tensor of shape (1, 64, 256, 6), containing real and imaginary parts
        for three components (Z, N/1, E/2).
    y_predict : np.ndarray
        Predicted event mask tensor of shape (1, 64, 256, 3).
    components : list of str, optional
        Component labels to plot. Default is ['Z', '1', '2'].
    plot_timeseries : bool, optional
        If True, plots the denoised time-domain signals (inverse STFT).
        If False, plots the log10 amplitude spectrograms of the denoised STFT.
    Returns
    -------
    None
    """
    stft_parameters = {"nperseg": 48, "nfft": 126, "fs": 100, "noverlap": 24}

    i = 0  # time window example

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)


    print(np.shape(stft_tmp))
    for j in range(3):
        # Denoised

        stft_denoised = (stft_tmp[i, :, :, j * 2] + 1j * stft_tmp[i, :, :, j * 2 + 1]) * y_predict[i, :, :, j]
        time_denoised, tr_denoised = istft(stft_denoised, **stft_parameters)

        if plot_timeseries:
            axes[j].plot(time_denoised, tr_denoised, color="k")

            axes[j].set_ylabel("Amplitude")
            axes[j].set_xlim(0, 61.2)
        else:
            ampl = np.log10(np.abs(stft_denoised))
            im = axes[j].pcolormesh(t_stft, f_stft, ampl, shading='auto',vmin=-10,vmax=-6)
            fig.colorbar(im, ax=axes[j], orientation="vertical", label="log10 Amplitude")
            axes[j].set_ylabel(f"{components[j]} - Freq (Hz)")

    axes[0].set_title("Denoised data " + str(wv[0].id[:-1]) + "".join(components))

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# %%
# Station and time
network, station, channel = "CH", "MFERR", "HH"
utc_start = UTCDateTime("2025-02-07T20:44:50")
# client = Client('ETH')



# Data download + preprocessing
wv  = get_data(network, station, channel, client, utc_start)
wv.plot()

# Data in STFT shape
t_stft, f_stft, stft_tmp, stft_tmp_norm = wv2stft(wv)

# Plotting data
plot_stft(t_stft, f_stft, stft_tmp, label="Preprocessd data ")
plot_stft(t_stft, f_stft, stft_tmp_norm, label="Preprocessd and normalised data ")

# Mask prediction
y_predict = model.predict(stft_tmp_norm,verbose=0)

# Plotting event masks
plot_masks(y_predict,t_stft, f_stft)

# Plotting denoise spectrogram
plot_denoiseddata(t_stft, f_stft,stft_tmp,y_predict, plot_timeseries=False)
# Plotting denoise timeseries
plot_denoiseddata(t_stft, f_stft,stft_tmp,y_predict, plot_timeseries=True)


