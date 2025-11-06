"""
EQS-Denoiser

Author: Niko Dahmen
Email: nikolaj.dahmen@eaps.ethz.ch
Affiliation: ETH Zurich
Date: 2025-08-11
Version: 1.0
"""

from obspy.clients.fdsn import Client
import numpy as np
import os
from sklearn.preprocessing import RobustScaler


client = Client('ETH')

def check_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def select_channel(_stream):
    _chan = []
    for tr in _stream:  # remove low sampling rate channels
        if tr.stats.sampling_rate < 100:
            _stream.remove(tr)
            continue
        _chan.append(tr.id[-3:-1])
    unique_channels = np.unique(_chan)

    if "HH" in unique_channels:
        return _stream.select(channel="HH*")
    elif "HG" in unique_channels:
        return _stream.select(channel="HG*")
    elif "EH" in unique_channels:
        return _stream.select(channel="EH*")
    else:
        return _stream[:3]


def normalize_percentile(data,quantile_range=(25,75),unit_variance=False,limit=1000):
    """
    Robust normalisation of samples  1-component data
    For data with outliers (w hard/soft clipping), seperately for real and imag.
    Input: data and optional arguments
    Output: normalised data
    Requires: sklearn Robustscaler
    """
    # get data
    data_real = data[:,:,0]  # real
    data_imag = data[:,:,1]  # imag
    # define scaler
    scaler =  RobustScaler(unit_variance=unit_variance)
    # apply to data: first flatten and then reshape
    data_real_norm = scaler.fit_transform(np.expand_dims(data_real.flatten(),1)).reshape((64,256))
    data_imag_norm = scaler.fit_transform(np.expand_dims(data_imag.flatten(),1)).reshape((64,256))


    # clip high values: hard clippping

    data_real_norm[data_real_norm>limit]=limit
    data_real_norm[data_real_norm<-limit]=-limit
    data_imag_norm[data_imag_norm>limit]=limit
    data_imag_norm[data_imag_norm<-limit]=-limit

    data_return = np.zeros(data.shape)
    data_return[:,:,0] = data_real_norm
    data_return[:,:,1] = data_imag_norm

    return data_return
