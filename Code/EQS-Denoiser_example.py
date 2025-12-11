"""
EQS-Denoiser

Author: Niko Dahmen
Email: nikolaj.dahmen@eaps.ethz.ch
Affiliation: ETH Zurich
Date: 2025-08-11
Version: 1.0
"""

import sys
sys.path.append("/Material/Code")

import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import tensorflow as tf
from obspy import UTCDateTime
import scipy

import numpy as np
from DenoisingFunctions_public import client, normalize_percentile
from DenoisingFunctions_public import get_data, wv2stft, plot_stft, plot_masks, plot_denoiseddata

# LOAD MODEL, ADJUST PATHS
model = tf.keras.models.load_model("/Material/Models/model_1000k_onlyweights.keras", compile=False)

# ADJUST PATHS IF DIFFERENT
module_dir = "/home/niko/Schreibtisch/EQ_denoising/Submission/Material/Code/"
sys.path.append(module_dir)


# %%
# Station and time
network, station, channel = "CH", "MFERR", "HH"
utc_start = UTCDateTime("2025-02-07T20:44:50")
# client = Client('ETH')

# Data download + preprocessing
wv  = get_data(network, station, channel, client, utc_start)
# plot data
wv.plot()

# Data in STFT shape
t_stft, f_stft, stft_tmp, stft_tmp_norm = wv2stft(wv)

# Plotting data
plot_stft(t_stft, f_stft, stft_tmp, label="Preprocessd data ", stream=wv)
plot_stft(t_stft, f_stft, stft_tmp_norm, label="Preprocessd and normalised data ", stream=wv)

# Mask prediction
y_predict = model.predict(stft_tmp_norm,verbose=0)

# Plotting event masks
plot_masks(y_predict,t_stft, f_stft)

# Plotting denoise spectrogram
_ = plot_denoiseddata(t_stft, f_stft,stft_tmp,y_predict, plot_timeseries=False, stream=wv)
# Plotting denoise timeseries
denoised_timeseries = plot_denoiseddata(t_stft, f_stft,stft_tmp,y_predict, plot_timeseries=True, stream=wv)
