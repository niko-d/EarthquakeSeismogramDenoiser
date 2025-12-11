"""
EQS-Denoiser

Author: Niko Dahmen
Email: nikolaj.dahmen@eaps.ethz.ch
Affiliation: ETH Zurich
Date: 2025-08-11
Version: 1.0
"""
from obspy import Stream
from scipy.ndimage import uniform_filter1d
from concurrent.futures import ThreadPoolExecutor, as_completed
from obspy.clients.fdsn import Client
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
import scipy.signal
from scipy.signal import stft, istft
import matplotlib.pyplot as plt
from obspy import UTCDateTime
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



def running_std_causal(arr, window_size):
    """
    Compute the causal running standard deviation of a 1D array.
    Parameters:
    - arr: 1D numpy array
        Input signal.
    - window_size: int
        Size of the running window.
    Returns:
    - 1D numpy array of same length as `arr` containing the running standard deviation.
    """
    arr = np.nan_to_num(arr, nan=0.0)  # check
    mean = uniform_filter1d(arr, size=window_size, mode='mirror', origin=0)  # running mean
    mean_sq = uniform_filter1d(arr**2, size=window_size, mode='mirror', origin=0)  # running mean of squared variance
    variance = mean_sq - mean**2  # variance
    return np.sqrt(variance)  # return standard deviation

def get_designaled_noise(_denoised_snippets, _original):
    """
    Compute the noise component ("designaled noise") by subtracting denoised snippets from the original signal.
    Parameters:
    - _denoised_snippets: obspy.Stream
        Stream of denoised waveform snippets.
    - _original: obspy.Stream
        Original waveform stream corresponding to the denoised snippets.
    Returns:
    - obspy.Stream containing the noise traces (original - denoised).
      Returns None if there is a length mismatch between original and denoised traces.
    Notes:
    - Assumes each trace in `_denoised_snippets` has a corresponding trace in `_original`.
    - Trims and merges `_denoised_snippets` if it contains more than 3 traces.
    """
    _denoised = _denoised_snippets.copy()
    if len(_denoised)>3:
        _denoised.merge(method=1, fill_value=0)
        _denoised.trim(_original[0].stats.starttime, _original[0].stats.endtime, fill_value=0, pad=True)

    components = [trace.stats.channel[2] for trace in _denoised]

    orig_data = [_original.select(component=comp)[0].data for comp in components]
    denoised_data = [_denoised.select(component=comp)[0].data for comp in components]

    if any(len(o) != len(d) for o, d in zip(orig_data, denoised_data)):
        print("Length mismatch!")
        return None

    # Process noise calculation more efficiently
    _noise = _original.copy()
    for i, comp in enumerate(components):
        _noise.select(component=comp)[0].data = orig_data[i] - denoised_data[i]

    del _denoised
    return _noise


def stream_tta(_event_stream, _noise_stream,id=0,white_noise_factor=0.01,constant_noise=False):
    """
    Applies Test-Time Augmentation (TTA) by injecting noise into an event waveform.
    Parameters:
    - _event_stream: obspy.Stream
        Stream containing event waveforms to be augmented.
    - _noise_stream: obspy.Stream
        Stream containing noise waveforms to be added to the event stream.
    - id: int, optional (default=0)
        Identifier to be stored in the waveform metadata.
    - white_noise_factor: float, optional (default=0.01)
        Scaling factor for the injected noise.
    - constant_noise: bool, optional (default=False)
        Currently unused parameter (potential for future modifications).
    Returns:
    - _event_noiseinjected: obspy.Stream
        Stream with noise-injected event waveforms.
    """
    # Copy the event stream to avoid modifying the original data
    _event_noiseinjected = _event_stream.copy()
    # Iterate over the event stream and corresponding noise traces
    for tr_denoised, tr_noise in zip(_event_noiseinjected, _noise_stream):
        # Inject noise: multiply noise trace's running standard deviation by random Gaussian noise
        if constant_noise:
            tr_denoised.data += white_noise_factor * np.std(tr_noise.data) * np.random.randn(
                len(tr_denoised.data))
        else:
            tr_denoised.data += white_noise_factor * running_std_causal(tr_noise.data, 10000) * np.random.randn(len(tr_denoised.data))

        # Update metadata: Assign a formatted ID to the 'location' field
        tr_denoised.stats.location = str(id).zfill(2)
    return _event_noiseinjected

def cluster_picks(pick_array, peak_vals, delta=1,print_results=False):
    """
    Cluster picks that are close in time and compute the weighted median for each cluster.
    Parameters:
    - pick_array: 1D numpy array
        Array of pick times (e.g., in seconds or sample indices).
    - peak_vals: 1D numpy array
        Corresponding confidence or amplitude values for each pick.
    - delta: float, optional
        Maximum separation between picks to consider them in the same cluster.
    - print_results: bool, optional
        If True, prints cluster details and median values.
    Returns:
    - medians: list of weighted median values for each cluster.
    - clusters: list of lists, each containing indices or times of picks in that cluster.
    Notes:
    - Picks are first sorted in ascending order.
    - Weighted median is computed using `peak_vals` as weights.
    """
    # check for multiple picks
    # Convert to timestamps once and sort
    sorted_indices = np.argsort(pick_array)
    sorted_picks = pick_array[sorted_indices]
    # sorted_peak_vals = peak_vals[sorted_indices]  # model confidence vals

    # Find cluster breaks using numpy's diff
    diffs = np.diff(sorted_picks)
    breaks = np.where(diffs > delta)[0] + 1
    cluster_indices = np.split(sorted_indices, breaks)
    medians = [weighted_median(pick_array[indices], peak_vals[indices]) for indices in cluster_indices]
    clusters = [[pick_array[i] for i in indices] for indices in cluster_indices]

    # Print results
    if print_results:
        for i, (cluster, median) in enumerate(zip(clusters, medians)):
            print(f"Cluster {i}: {cluster}")
            print(f" → Median: {median}\n")
    return medians, clusters

def weighted_std(values, weights):
    """
     Compute the weighted standard deviation of an array.
     Parameters:
     - values: 1D array-like
         Data values.
     - weights: 1D array-like
         Corresponding weights for each value.
     Returns:
     - float: weighted standard deviation.
     Notes:
     - Adds a small epsilon (1e-30) to weights to avoid division by zero.
     """
    average = np.average(values, weights=weights+1e-30)
    variance = np.average((values-average)**2, weights=weights+1e-30)
    return np.sqrt(variance)

def weighted_median(argmax_values, max_values):
    """
    Compute the weighted median of an array.
    Parameters:
    - values: 1D array-like
        Data values.
    - weights: 1D array-like
        Corresponding weights for each value.
    Returns:
    - float: weighted median value.
    Notes:
    - Weighted median is the value where the cumulative sum of weights reaches 50% of the total weight.
    """
    argmax_values = np.asarray(argmax_values)
    max_values = np.asarray(max_values)

    # Sort values and weights by the values
    sorted_indices = np.argsort(argmax_values)
    sorted_vals = argmax_values[sorted_indices]
    sorted_weights = max_values[sorted_indices]

    # Compute cumulative sum of weights
    cum_weights = np.cumsum(sorted_weights)
    total_weight = np.sum(sorted_weights)

    # Find the index where cumulative weight exceeds or equals 50%
    median_idx = np.searchsorted(cum_weights, total_weight / 2.0)
    return sorted_vals[median_idx]

def tta_uncertainty(confidence_timeseries,pick_utc,pick_tolerance=1.0,confidence=0.5):
    """
    Estimate uncertainty of a pick using Test-Time Augmentation (TTA).

    Parameters:
    - confidence_timeseries: list of obspy.Trace
        Traces containing model confidence values over time.
    - pick_utc: float or obspy.UTCDateTime
        Pick time around which to evaluate uncertainty.
    - pick_tolerance: float, optional
        Time window (+/-) around the pick to consider.
    - confidence: float, optional
        Confidence threshold for determining if a pick is "reached".

    Returns:
    - tuple:
        - float: uncertainty estimate (1 + weighted std of argmax positions).
        - float: fraction of TTA traces exceeding the confidence threshold.

    Notes:
    - Handles multiple traces by slicing around the pick time and computing the argmax of each slice.
    - If no traces overlap the window, consider adding a guard to avoid errors.
    """

    # Precompute time window bounds
    t_start, t_end = pick_utc - pick_tolerance, pick_utc + pick_tolerance

    # Pre-slice all traces and filter those overlapping the window
    sliced_traces = [
        trace.slice(t_start, t_end)
        for trace in confidence_timeseries
        if trace.stats.starttime <= t_end and trace.stats.endtime >= t_start
    ]

    # Compute argmax for each sliced trace
    _argmax = [np.argmax(trace.data) for trace in sliced_traces]
    _max = np.array([np.max(trace.data) for trace in sliced_traces])
    reached_threshold = np.mean(_max>confidence)

    return 1  + weighted_std(_argmax,_max), reached_threshold


def process_peak_times(peak_times, peak_vals, annotations, channel_pattern, start_time, pick_tolerance=1,confidence=0.5):
    """
    Cluster peak times, convert to UTC, and compute TTA-based uncertainty for each median pick.

    Parameters:
    - peak_times: 1D array-like
        Times of detected peaks (relative or absolute).
    - peak_vals: 1D array-like
        Confidence or amplitude values associated with each peak.
    - annotations: obspy.Stream or PickList
        Model output containing confidence traces for each channel.
    - channel_pattern: str
        Pattern to select specific channel(s) from annotations (e.g., "*_P").
    - start_time: float or obspy.UTCDateTime
        Reference start time to convert relative picks to UTC.
    - pick_tolerance: float, optional
        Maximum allowed separation (seconds) for clustering picks.
    - confidence: float, optional
        Threshold for TTA confidence evaluation.

    Returns:
    - picks_median_utc: list of float
        UTC times of median picks after clustering.
    - results: list of tuples
        Each tuple contains (uncertainty, fraction_above_confidence) for the corresponding pick.

    Notes:
    - Returns empty lists if no peaks are provided.
    - Uses `tta_uncertainty` to quantify timing uncertainty from multiple confidence traces.
    """
    if len(peak_times) == 0:
        return [], []  # Return empty lists for picks and results if no peaks

    # Cluster picks and convert to UTC
    picks_median, picks = cluster_picks(peak_times, peak_vals, delta=pick_tolerance)  # original
    # picks_median, picks = cluster_picks(peak_times, annotations, delta=pick_tolerance)

    picks_median_utc = [start_time + t for t in picks_median]

    # Pre-select traces for the channel to avoid repeated selections
    selected_traces = annotations.select(channel=channel_pattern)

    # Compute uncertainty for each median pick
    results = [
        tta_uncertainty(selected_traces, pick, pick_tolerance=1, confidence=confidence)
        for pick in picks_median_utc
    ]

    return picks_median_utc, results

def process_snippet(event_streams, st_designaled, repeat, pick_tolerance, p_confidence, s_confidence, model_picking):
    """
    Process a set of event streams with Test-Time Augmentation (TTA) to compute phase picks and uncertainties.

    Parameters:
    - event_streams: tuple of obspy.Stream
        Streams containing Z, N, and E components of the event.
    - st_designaled: obspy.Stream
        Noise stream extracted from the original data for TTA.
    - repeat: int
        Number of TTA repetitions (different noise injections).
    - pick_tolerance: float
        Temporal tolerance for clustering picks.
    - p_confidence: float
        Minimum confidence threshold for P-wave picks.
    - s_confidence: float
        Minimum confidence threshold for S-wave picks.
    - model_picking: SeisBench model instance
        Phase picking model with `annotate` and `classify_aggregate` methods.

    Returns:
    - dict with keys 'p_picks' and 's_picks', each a list of tuples:
        (median pick time, uncertainty, fraction above confidence, event_id)

    Notes:
    - Handles trimming of traces, injection of designaled noise, TTA annotation, clustering, and uncertainty estimation.
    """
    _st_z, _st_n, _st_e = event_streams

    # Adjust the streams
    add = 5 if _st_z.stats.npts >= 6120 else 5 + (6120 - _st_z.stats.npts) / 200
    for st in (_st_z, _st_n, _st_e):
        st.trim(st.stats.starttime - add, st.stats.endtime + add, pad=True, fill_value=0)

    # Prepare noise data for the event
    _noise = st_designaled.copy()
    _start, _end = _st_z.stats.starttime, _st_z.stats.endtime
    for tr in _noise:
        tr.trim(_start, _end, pad=True, fill_value=0)

    # Build TTA collection with injected noise
    event_tta_collection = Stream()
    for i in range(repeat):
        event_tta_collection += stream_tta(Stream([_st_z, _st_n, _st_e]), _noise, id=i, white_noise_factor=0.01,
                                           constant_noise=True)

    annotations = model_picking.annotate(event_tta_collection, batch_size=repeat)
    annotations.trim(_start, _end, pad=True, fill_value=0)

    # Get phase picks above confidence thresholds
    picks_current_tta = model_picking.classify_aggregate(annotations, argdict={}).picks
    p_picks_current_tta = picks_current_tta.select(min_confidence=p_confidence, phase="P")
    s_picks_current_tta = picks_current_tta.select(min_confidence=s_confidence, phase="S")
    p_peak_times = np.array([p.peak_time - _start for p in p_picks_current_tta])
    s_peak_times = np.array([s.peak_time - _start for s in s_picks_current_tta])

    p_peak_times_maxval = np.array([p.peak_value for p in p_picks_current_tta])
    s_peak_time_maxval = np.array([s.peak_value for s in s_picks_current_tta])


    p_picks_median, p_results = process_peak_times(
        peak_times=p_peak_times,
        peak_vals=p_peak_times_maxval,
        annotations=annotations,
        channel_pattern="*_P",
        start_time=_start,
        pick_tolerance=pick_tolerance,
        confidence=p_confidence
    )
    s_picks_median, s_results = process_peak_times(
        peak_times=s_peak_times,
        peak_vals=s_peak_time_maxval,
        annotations=annotations,
        channel_pattern="*_S",
        start_time=_start,
        pick_tolerance=pick_tolerance,
        confidence=s_confidence
    )

    event_id = _st_z.id
    # Unpack the two values (uncertainty and extra_metric) from each result tuple
    p_picks = [(median, res[0], res[1], event_id) for median, res in zip(p_picks_median, p_results)]
    s_picks = [(median, res[0], res[1], event_id) for median, res in zip(s_picks_median, s_results)]

    return {'p_picks': p_picks, 's_picks': s_picks}


# --- Main parallel loop using threads ---
def process_stream(st_denoised, st_designaled, model_picking, repeat=20, pick_tolerance=1, p_confidence=0.5,
                       s_confidence=0.5,min_share_models=0.25):
    """
    Parallel processing of denoised event streams using Test-Time Augmentation (TTA) to generate
    P- and S-wave picks with associated uncertainties.

    Parameters:
    - st_denoised: obspy.Stream
        Denoised seismic event streams (Z, N, E components).
    - st_designaled: obspy.Stream
        Noise streams extracted from original recordings for TTA.
    - model_picking: SeisBench model instance
        Phase picking model with `annotate` and `classify_aggregate` methods.
    - repeat: int, optional
        Number of TTA repetitions (default: 20).
    - pick_tolerance: float, optional
        Temporal tolerance for clustering picks (default: 1s).
    - p_confidence: float, optional
        Confidence threshold for P-wave picks (default: 0.5).
    - s_confidence: float, optional
        Confidence threshold for S-wave picks (default: 0.5).
    - min_share_models: float, optional
        Minimum fraction of TTA models above threshold to keep a pick (default: 0.25).

    Returns:
    - dict with keys 'p_picks' and 's_picks', each a list of tuples:
        (median pick time, uncertainty, fraction above confidence, event_id)

    Notes:
    - Uses ThreadPoolExecutor to parallelize TTA processing across all events.
    - Filters picks based on min_share_models to retain only robust detections.
    """

    all_results = {'p_picks': [], 's_picks': []}

    st_denoised.sort(keys=['starttime'])
    st_designaled.sort(keys=['starttime'])

    st_z_list = st_denoised.select(component="Z")
    st_n_list = st_denoised.select(component="N")
    st_e_list = st_denoised.select(component="E")
    event_streams_list = list(zip(st_z_list, st_n_list, st_e_list))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_snippet,
                event_streams,
                st_designaled,
                repeat,
                pick_tolerance,
                p_confidence,
                s_confidence,
                model_picking
            )
            for event_streams in event_streams_list
        ]
        for future in as_completed(futures):
            result = future.result()
            all_results['p_picks'].extend(result['p_picks'])
            all_results['s_picks'].extend(result['s_picks'])


    all_results['p_picks'] = [entry for entry in all_results["p_picks"] if entry[2] > min_share_models]
    all_results['s_picks'] = [entry for entry in all_results["s_picks"] if entry[2] > min_share_models]
    return all_results


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


def plot_stft(t_stft, f_stft, stft_tmp, label="Preprocessd and normalised data ", components = ['Z', '1', '2'],stream=None):
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

    if stream is not None:
        plt.suptitle(label + str(stream[0].id[:-1]) + "".join(components))

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
    # plt.suptitle("Predicted event mask " + str(wv[0].id[:-1]) + "".join(components), y=1.0)

    plt.show()


def plot_denoiseddata(t_stft, f_stft, stft_tmp, y_predict, components = ['Z', '1', '2'], plot_timeseries=True, stream=None):
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


    _waveforms = np.zeros([3,6120])

    for j in range(3):
        # Denoised

        stft_denoised = (stft_tmp[i, :, :, j * 2] + 1j * stft_tmp[i, :, :, j * 2 + 1]) * y_predict[i, :, :, j]
        time_denoised, tr_denoised = istft(stft_denoised, **stft_parameters)
        _waveforms[j] = tr_denoised

        if plot_timeseries:
            axes[j].plot(time_denoised, tr_denoised, color="k")

            axes[j].set_ylabel("Amplitude")
            axes[j].set_xlim(0, 61.2)
        else:
            ampl = np.log10(np.abs(stft_denoised))
            im = axes[j].pcolormesh(t_stft, f_stft, ampl, shading='auto',vmin=-10,vmax=-6)
            fig.colorbar(im, ax=axes[j], orientation="vertical", label="log10 Amplitude")
            axes[j].set_ylabel(f"{components[j]} - Freq (Hz)")

    if stream is not None:
        axes[0].set_title("Denoised data " + str(stream[0].id[:-1]) + "".join(components))

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    return _waveforms