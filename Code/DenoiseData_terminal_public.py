"""
EQS-Denoiser

Author: Niko Dahmen
Email: nikolaj.dahmen@eaps.ethz.ch
Affiliation: ETH Zurich
Date: 2025-08-11
Version: 1.0
"""
import argparse
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import time
import tensorflow as tf
from obspy import UTCDateTime
import obspy
import scipy
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor
from DenoisingFunctions_public import select_channel, check_dir, normalize_percentile # client
from obspy.clients.fdsn.client import Client

def compare_arrays_time_overlap(array1, array2,overlap=0.75):
    """
    Compare two arrays of time intervals and select overlapping intervals with higher scores.

    Each row in the input arrays should contain:
    [peak, start_time, end_time, score, maxval].

    For each interval in array1, the function finds intervals in array2 that overlap
    by at least a fraction `overlap` of the smaller interval's duration. It keeps
    the interval with the higher score for overlapping pairs.

    Intervals in array2 not overlapping any interval in array1 are also included.

    Parameters:
    -----------
    array1 : array-like (N1 x 5)
        First array of intervals.

    array2 : array-like (N2 x 5)
        Second array of intervals.

    overlap : float, optional (default=0.75)
        Minimum required overlap fraction relative to the smaller interval.

    Returns:
    --------
    final_rows : np.ndarray
        Combined array of intervals after comparison.

    origins : list of int
        Indicator list where 0 means interval from array1, 1 from array2.
    """
    result = []  # To store resulting rows and origin
    used_indices_array2 = set()  # Track rows in array2 that have been processed

    for i, row1 in enumerate(array1):
        peak1, start1, end1, score1, maxval1 = row1
        best_match = (row1, 0)  # Default to row1

        for j, row2 in enumerate(array2):
            peak2, start2, end2, score2, maxval2 = row2

            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            overlap_duration = max(0, overlap_end - overlap_start)

            # Calculate the smaller window duration
            window1_duration = end1 - start1
            window2_duration = end2 - start2
            smaller_window = min(window1_duration, window2_duration)

            # Check if overlap is >90% of the smaller window
            if overlap_duration > overlap * smaller_window:
                # Compare scores
                if score2 > score1:
                    best_match = (row2, 1)
                used_indices_array2.add(j)  # Mark row2 as used
                # break  # Exit the loop once a match is found

        # Add the best match for this row
        result.append(best_match)

    # Handle rows in array2 that were not matched
    for j, row2 in enumerate(array2):
        if j not in used_indices_array2:
            result.append((row2, 1))  # Add unmatched rows from array2

    # Extract rows and origin indicators
    final_rows = np.array([r[0] for r in result])
    origins = [r[1] for r in result]

    return final_rows, origins

def get_mask_timeseries(mask_array):
    """
    Extract two time series from a 4D mask array by computing a weighted mean of maximum mask values
    across selected channels at each time step, then splitting the result into even and odd time steps.

    The weighted mean gives double weight to the first channel and equal weight to the next two.

    Parameters:
    -----------
    mask_array : np.ndarray
        4D array where the last dimension indexes channels.

    Returns:
    --------
    array_even : np.ndarray
        Concatenated values from even-indexed time steps.

    array_odd : np.ndarray
        Concatenated values from odd-indexed time steps.
    """
    # extract time series of mask as mean value of max. / mean mask values at each time step, equal weight for vertical and horizontal

    timeseries_3comp = (2* np.max(mask_array[:, :, :, 0], axis=1) + np.max(mask_array[:, :, :, 1], axis=1) + np.max(mask_array[:, :, :, 2], axis=1)) / 4

    steps = timeseries_3comp.shape[0]
    # step through overlapping array
    array_even, array_odd = np.array([]), np.array([])
    for i in range(steps):
        _tmp = timeseries_3comp[i]
        # add data to even or odd array
        if i % 2 == 0:
            array_even = np.concatenate((array_even, _tmp), axis=0)
        else:
            array_odd = np.concatenate((array_odd, _tmp), axis=0)
    return array_even, array_odd

def get_peaks(timeseries, threshold=0.1,shift_correction=0):
    """
    Detect peaks in a timeseries exceeding a given threshold and find their onset and end points.

    Peaks are detected using a minimum distance between peaks. For each peak, the function finds:
    - The left boundary where the signal falls below 0.01 before the peak.
    - The right boundary where the signal falls below 0.05 after the peak.
    - The sum of values between the left and right boundaries.
    - The peak value itself.

    Parameters:
    -----------
    timeseries : np.ndarray
        1D array of signal values.

    threshold : float, optional (default=0.1)
        Minimum height of peaks to be detected.

    shift_correction : int, optional (default=0)
        Value subtracted from detected indices to adjust for any offset.

    Returns:
    --------
    np.ndarray
        Array of detected peaks with columns:
        [peak_index, left_boundary_index, right_boundary_index, sum_between_boundaries, peak_value].
    """
    peaks, _ = find_peaks(timeseries, height=threshold,distance=128)

    peaks_info = []
    # find onset/end of each peak
    for peak in peaks:
        left_index = None
        for i in range(peak - 1, -1, -1):
            if timeseries[i] < 0.01:
                left_index = i
                break
        if left_index is None:
            left_index = 0

        right_index = None
        max_len = len(timeseries) - peak
        for i in range(peak, peak + np.min([max_len, 9999])):
            # if timeseries[i] < 0.01:
            if timeseries[i] < 0.05:
                right_index = i
                break

        if right_index is None:
            right_index = i

        _mask_vals = np.sum(timeseries[left_index:right_index])

        peaks_info.append([peak-shift_correction,left_index-shift_correction,right_index-shift_correction,_mask_vals,timeseries[peak]])

    return np.array(peaks_info)

# Function to process each segment
def process_segment(i):
    """
    Compute and normalize the Short-Time Fourier Transform (STFT) for a data segment across three components.

    Extracts a time segment from each component (Z, N, E), computes the STFT, stores real and imaginary parts,
    and applies percentile normalization. If data length is insufficient for the segment, returns None.

    Parameters:
    -----------
    i : int
        Index of the segment to process.

    Returns:
    --------
    tuple of np.ndarray or None
        Tuple containing:
        - Original STFT array of shape (1, 64, 256, 6) with real and imaginary parts for each component.
        - Normalized STFT array of the same shape.
        Returns None if the segment is skipped due to insufficient data length.
    """
    # len_sample = 6120
    # shift_samples = int(len_sample / 2)
    # stft_parameters = {"nperseg": 48, "nfft": 126, "fs": 100, "noverlap": 24}

    if set_verbose:
        print(f"Computing STFT segment: {i}")
    stft_tmp, stft_tmp_norm = np.zeros((64, 256, 6)), np.zeros((64, 256, 6))

    for j, data in enumerate([data_z, data_n, data_e]):
        snippet_tmp = data[i * shift_samples:i * shift_samples + len_sample]
        if len(snippet_tmp) != len_sample:
            if set_verbose:
                print(f"Skipped segment {i} due to insufficient data length")
            return None  # Returning None if skipped

        f, t, _stft = scipy.signal.stft(snippet_tmp, **stft_parameters)

        # Store the original STFT
        stft_tmp[:, :, j * 2] = _stft.real
        stft_tmp[:, :, j * 2 + 1] = _stft.imag

        # Normalize
        stft_tmp_1c = np.stack((_stft.real, _stft.imag), axis=2)
        stft_tmp_1c = normalize_percentile(stft_tmp_1c)
        stft_tmp_norm[:, :, j * 2] = stft_tmp_1c[:, :, 0]
        stft_tmp_norm[:, :, j * 2 + 1] = stft_tmp_1c[:, :, 1]

    return np.expand_dims(stft_tmp, axis=0), np.expand_dims(stft_tmp_norm, axis=0)


def process_seismic_data(network, station, channel, start_time, duration, verbose, threshold,saveraw,model_name,min_peak_height=0.1,client_str="ETH"):
    """
    Download, preprocess, denoise, and detect seismic signals from waveform data using a deep learning model.

    Parameters
    ----------
    network : str
        Network code of the seismic station (e.g., 'IU').
    station : str
        Station code of the seismic station (e.g., 'ANMO').
    channel : str
        Channel code prefix (e.g., 'BH'), the function appends '?' to select components.
    start_time : UTCDateTime or str
        Start time of the data segment to process (can be UTCDateTime object or ISO format string).
    duration : float
        Duration in seconds of the seismic data window to process around the center time.
    verbose : bool
        If True, print detailed processing information.
    threshold : float
        Detection threshold for event signal acceptance.
    saveraw : str
        If set to 'True' or 'true', saves the original raw waveform data to a MiniSEED file before processing.
    model_name : str
        Name of the deep learning model file (without extension) located in the models directory.
    min_peak_height : float, optional
        Minimum peak height threshold for event detection in the predicted mask time series (default is 0.1).

    Returns
    -------
    denoised_stream : obspy.Stream
        Obspy Stream object containing denoised seismic waveforms for detected event windows.
    detection_start_times : list of UTCDateTime
        List of start times corresponding to detected events.
    detection_scores : list of float
        Confidence scores of the detected events.
    filtered_results : list
        Detailed info on detected peaks including start, end indices and detection scores.

    Notes
    -----
    - The function downloads waveform data with a large buffer to ensure coverage.
    - Waveform responses are removed and data resampled to 100 Hz if needed.
    - If `saveraw` is 'True' or 'true', the original waveform is saved as a MiniSEED file before denoising.
    - Data is processed in overlapping segments using STFT and fed to a trained deep learning model.
    - Detection is performed on predicted masks by identifying peaks in event probability.
    - The function refines detected event windows and produces denoised waveforms using the model's mask output.
    - Outputs are sorted in time and trimmed to avoid overlaps between adjacent detected events.
    - Global variables `data_z`, `data_n`, `data_e` hold component data during processing for efficiency.
    - Parallel processing is used for STFT computation of segments to speed up processing.

    """

    global data_z, data_n, data_e  # much faster than having as function input (?)
    global set_verbose, len_sample, stft_parameters, shift_samples
    global client

    client = Client(client_str)
    print(client_str)

    t_pick = UTCDateTime(start_time)+duration/2
    time_window = [duration/2, duration/2]

    detection_threshold = threshold

    set_verbose = verbose

    plot_stuff = False  #
    # do not change
    buffer = 600
    len_sample = 6120
    shift_samples = int(len_sample / 2)
    bins_overlap = 128

    pre_filt = [1 / 100, 1 / 20, 45, 50]
    stft_parameters = {"nperseg": 48, "nfft": 126, "fs": 100, "noverlap": 24}
    model_path = model_name   # folder were model weights are strored

    # load model
    model = tf.keras.models.load_model(model_path, compile=False)

    if verbose:
        model_verbose = "auto"
    else:
        model_verbose = 0


    # START DATA PREPROCESSING
    start_time = time.time()
    wv = client.get_waveforms(network=network, station=station, channel=channel + "?",location='*',
                                  starttime=t_pick - time_window[0] - buffer, endtime=t_pick + time_window[1] + buffer,
                                  attach_response=True).merge()

    if verbose:
        print("Downloading waveform data --- %s seconds ---" % (time.time() - start_time))

    wv = select_channel(wv)
    if verbose:
        print(wv[0].stats.sampling_rate)

    start_time = time.time()
    wv = wv.select(location=wv[0].stats.location)
    wv.remove_response(output="VEL", pre_filt=pre_filt, water_level=None)# 60
    if verbose:
        print("Removing response --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    if wv[0].stats.sampling_rate > 100:
        wv.resample(100)  # pre filter
    wv.trim(wv[0].stats.starttime+buffer, wv[0].stats.endtime-buffer)
    if verbose:
        print("Downsampling --- %s seconds ---" %  (time.time() - start_time))

    if verbose:
        print(wv)


    sampling_rate = wv[0].stats.sampling_rate
    start_time = time.time()
    duration_seconds = wv[0].stats.endtime - wv[0].stats.starttime
    duration_samples = sampling_rate * duration_seconds

    num_segments = int((2 * duration_samples / len_sample))

    # collect components and sort Z as first
    components = []
    for trace in wv:
        # The last character of the channel code represents the component
        components.append(trace.stats.channel[2])
    components.sort(reverse=True)

    if verbose:
        print(components)

    # get waveform data
    data_z = wv.select(component=components[0])[0].data
    data_n = wv.select(component=components[1])[0].data
    data_e = wv.select(component=components[2])[0].data

    # get start times of time windows
    utc_start_list = []
    for n in range(num_segments):
        utc_start_list.append(wv[0].stats.starttime+n*shift_samples/sampling_rate)

    # Compute STFT for time windows
    # based on global data_z, ...
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_segment, range(num_segments)))

    # Filter out None results and concatenate
    stft_collection = np.concatenate([result[0] for result in results if result is not None], axis=0)
    stft_norm_collection = np.concatenate([result[1] for result in results if result is not None], axis=0)

    if verbose:
        print("Dimensions of collected STFT data (raw and normalised)")
        print("STFT Collection shape:", stft_collection.shape)
        print("Normalized STFT Collection shape:", stft_norm_collection.shape)
        print("parallel Computing model input (STFT) ---%s seconds ---" %  (time.time() - start_time))

    start_time = time.time()

    # MAKE DL PREDICTION
    y_predict = model.predict(stft_norm_collection,verbose=model_verbose)
    if verbose:
        print("Inference / Applying model---%s seconds ---" %  (time.time() - start_time))

    t = np.linspace(0,61.2,256)
    bin_spacing = (255/256) * (t[1]-t[0])

    # DETECTING EVENT SIGNALS AND SELECTING PREFERRED TIME WINDOW
    # compute mask time series and extract non-overlapping windows as separate even and odd time series
    mask_timeseries_even, mask_timeseries_odd = get_mask_timeseries(y_predict)
    # get peaks with start and end, with fixed min. threshold of 0.1 for max of time series (=at leats one bin with mask value>0.1)
    peak_info_even = get_peaks(mask_timeseries_even, threshold=min_peak_height,shift_correction=128) # account for 50% time shift
    peak_info_odd = get_peaks(mask_timeseries_odd, threshold=min_peak_height,shift_correction=0)

    # combine and compare both detection results, keep better detection (higher score)
    filtered_results, origin = compare_arrays_time_overlap(peak_info_even, peak_info_odd)

    if verbose:
        print("Number of initially detected signals EVEN / ODD / COMBINED / origin_check")
        print(np.shape(peak_info_even))
        print(np.shape(peak_info_odd))
        print(np.shape(filtered_results))
        print(np.shape(origin))


    # Select data and mask based on list
    selected_stft, selected_masks, selected_utc = [], [], []
    detection_score, detection_start = [], []

    for filtered_result, even_odd in zip(filtered_results,origin):
        # check if "better" solution in even or odd-numbered row.
        if even_odd==0: # even-number row

            index_window = int(2 * ((filtered_result[1]+128) // 256))
            bin_start = (filtered_result[1]+bins_overlap) % 256
            if bin_start >250:
                index_window += 2
                bin_start = 0

            selected_masks.append(y_predict[index_window])
            selected_stft.append(stft_collection[index_window])
            selected_utc.append(utc_start_list[index_window])
            detection_start.append(utc_start_list[index_window] + bin_start*bin_spacing)

            if plot_stuff:
                plt.pcolormesh(y_predict[index_window,:,:,0],cmap="cubehelix_r",vmin=0,vmax=0.1)
                plt.axvline((detection_start[-1]-selected_utc[-1])/bin_spacing)
        else:
            index_window = int(2 * (filtered_result[1] // 256) + 1)
            bin_start = filtered_result[1] % 256

            if bin_start >250:
                index_window += 2
                bin_start = 0

            selected_masks.append(y_predict[index_window])
            selected_stft.append(stft_collection[index_window])
            selected_utc.append(utc_start_list[index_window])
            detection_start.append(utc_start_list[index_window] + bin_start*bin_spacing)

            if plot_stuff:
                plt.pcolormesh(y_predict[index_window , :, :, 0], cmap="cubehelix_r", vmin=0, vmax=0.1)
                plt.axvline((detection_start[-1]-selected_utc[-1])/bin_spacing)

        detection_score.append(filtered_result[3])

        if plot_stuff:
            plt.title([even_odd,filtered_result[3],filtered_result[1]])
            plt.show()

    selected_masks = np.array(selected_masks)
    selected_stft = np.array(selected_stft)

    if verbose:
        print("Saved data and mask for detection windows: masks / stft / utc list")
        print(np.shape(selected_masks))
        print(np.shape(selected_stft))
        print(np.shape(selected_utc))

    # RECOMPUTE MASKS BASED ON DETECTION START
    shift_seconds = bin_spacing*42#10  # trying to align estimated signal start with binning
    first_stft = True
    shift_samples = int(len_sample/2)
    # wv_snippets = []#obspy.Stream()#[]
    stream_start_end = []
    new_window_start = []
    for _utc in detection_start:

        stft_tmp, stft_tmp_norm = np.zeros((64, 256, 6)), np.zeros((64, 256, 6))
        wv_snippet = wv.copy()

        wv_snippet.trim(_utc-shift_seconds,_utc+(65-shift_seconds),pad=True,fill_value=0)
        new_window_start.append(_utc-shift_seconds)

        data_z = wv_snippet.select(component=components[0])[0].data
        data_n = wv_snippet.select(component=components[1])[0].data
        data_e = wv_snippet.select(component=components[2])[0].data


        for j, data in enumerate([data_z,data_n,data_e]):
            snippet_tmp = data[:len_sample]

            if len(snippet_tmp)!=len_sample:
                if verbose:
                    print("skipped:", j)
                continue
            f, t, _stft = scipy.signal.stft(snippet_tmp, **stft_parameters)
            # original
            stft_tmp[:,:,j*2] = _stft.real
            stft_tmp[:,:,j*2+1] = _stft.imag

            # norm
            stft_tmp_1c = np.stack((_stft.real, _stft.imag),axis=2)
            stft_tmp_1c = normalize_percentile(stft_tmp_1c)
            stft_tmp_norm[:,:,j*2] = stft_tmp_1c[:,:,0]
            stft_tmp_norm[:,:,j*2+1] = stft_tmp_1c[:,:,1]

        stft_tmp = np.expand_dims(stft_tmp, axis=0)
        stft_tmp_norm = np.expand_dims(stft_tmp_norm, axis=0)
        if first_stft:
            stft_collection_subset = stft_tmp
            stft_norm_collection_subset = stft_tmp_norm

            first_stft = False
        else:
            stft_collection_subset = np.concatenate((stft_collection_subset, stft_tmp), axis=0)
            stft_norm_collection_subset = np.concatenate((stft_norm_collection_subset, stft_tmp_norm), axis=0)
        stream_start_end.append([wv_snippet[0].stats.starttime,wv_snippet[0].stats.endtime])


    if verbose:
        print("Recomputed STFT,  masks, utc for new time windows")
        print(np.shape(stft_collection_subset))
        print(np.shape(stft_norm_collection_subset))
        print(len(stream_start_end))

    # Make new prediction
    y_predict_event = model.predict(stft_norm_collection_subset,verbose=model_verbose)

    # use new prediction if better
    #  compute detection score of new time windows, keep if score improved (compared to detection windows)
    # keep if above detection threshold
    stft_final_subset, masks_subset, utc_start_subset = [], [], []
    # mask_timeseries_info = []
    ev_signal_startstop = []
    for i,y_event in enumerate(y_predict_event):
        # _timeseries = (2* np.max(y_event[:, :, 0], axis=0) + np.max(y_event[:, :, 2], axis=0) + np.max(y_event[:, :, 4], axis=0)) / 4
        _timeseries = (2* np.max(y_event[:, :, 0], axis=0) + np.max(y_event[:, :, 1], axis=0) + np.max(y_event[:, :, 2], axis=0)) / 4
        # print("3 model outputs")
        _peak = get_peaks(_timeseries, threshold=min_peak_height, shift_correction=0)

        keep_old = True
        _score = filtered_results[i][3]
        if plot_stuff:
            plt.plot(_timeseries)

            plt.ylim(0, 1)
            plt.title("new: " + str(int(_peak[0][3])) + " - old: " + str(int(filtered_results[i][3])))
            plt.show()

        if len(_peak)>0:  # if peak found, check if score of new time window is higher

            if plot_stuff:
                plt.axvline(_peak[0][0], color="k")
                plt.axvline(_peak[0][1])
                plt.axvline(_peak[0][2])


            if _peak[0][3] > 0.5*filtered_results[i][3]: # new score higher (not much lower) than old score, collect new windows
                _score = _peak[0][3]
                keep_old = False

        if _score>detection_threshold:
            if keep_old:
                stft_final_subset.append(selected_stft[i])
                masks_subset.append(selected_masks[i])
                utc_start_subset.append(selected_utc[i])
                _detect_duration = filtered_results[i][2] - filtered_results[i][1]
                ev_signal_startstop.append([detection_start[i], detection_start[i]+_detect_duration])

            else:
                stft_final_subset.append(stft_collection_subset[i])
                masks_subset.append(y_event)
                utc_start_subset.append(stream_start_end[i][0])

                ev_signal_startstop.append([stream_start_end[i][0] + _peak[0][1]*bin_spacing, stream_start_end[i][0] + _peak[0][2]*bin_spacing])


    masks_subset = np.array(masks_subset)
    stft_final_subset = np.array(stft_final_subset)

    if verbose:
        print("Final data and mask for detection windows: masks / stft / utc list / start end list")
        print(masks_subset.shape)
        print(stft_final_subset.shape)
        print(len(utc_start_subset))
        print(len(ev_signal_startstop))

    # DENOISE DATA AND PRODUCE WAVEFORMS BASED ON COLLECTED MASK AND STFT
    num = stft_final_subset.shape[0]
    st_denoised_collection = obspy.Stream()

    for i in range(num):
        st_denoised = wv.copy()
        st_denoised.trim(utc_start_subset[i],utc_start_subset[i]+61.19)
        for j in range(3):
            _stft = stft_final_subset[i, :, :, j * 2] + 1j * stft_final_subset[i, :, :, j * 2 + 1]
            # _stft_denoised = _stft * masks_subset[i, :, :, j * 2]
            _stft_denoised = _stft * masks_subset[i, :, :, j] # 3 output model

            t_td, td_signal = istft(_stft_denoised,**stft_parameters)

            st_denoised.select(component=components[j])[0].data = td_signal  # fix comps!!!!!!!!!!!!!!!!!!!



        st_denoised_collection += st_denoised

    # sorting and packaging
    # stream trimming in place , need to run together with st_denoised_collection generation
    segments = [st_denoised_collection[3 * i: 3 * (i + 1)] for i in range(num)]  # get list of streams

    # start times of the first trace in each segment
    start_times = [segment[0].stats.starttime for segment in segments]  # get start time for list of streams
    sorted_indices = sorted(range(len(segments)), key=lambda i: start_times[i])  # get indices to sort in time
    segments_sorted = [segments[i] for i in sorted_indices]  # sort streams in time
    ev_signal_startstop_sorted = [ev_signal_startstop[i] for i in sorted_indices] # sort signal start/end list in time


    trimmed_streams = obspy.Stream()
    for j in range(0, len(segments_sorted)-1):
        current_stream = segments_sorted[j]
        next_stream = segments_sorted[j + 1]

        # check if streams overlapping; streams 61.2s waveforms, with (shorter) detected signals within
        # modify endtime of current stream, and start time of next stream, if required
        if current_stream[0].stats.endtime>next_stream[0].stats.starttime: # end of current stream after start of next
            # j = int(i/3)
            # check if event signal windows overlap
            if ev_signal_startstop_sorted[j][1]<next_stream[0].stats.starttime:  # signal in current stream ends before next stream starts
                current_stream.trim(endtime=ev_signal_startstop_sorted[j][1]-0.01)
                if verbose:
                    print("no signal overlap with next stream")
            elif ev_signal_startstop_sorted[j][1]+3<ev_signal_startstop_sorted[j+1][0]: # signal in current stream ends before next signal starts, w buffer
                current_stream.trim(endtime=ev_signal_startstop_sorted[j][1]-0.01)  # cut current stream to end at start of next signal, w buffer
                next_stream.trim(starttime=ev_signal_startstop_sorted[j+1][0]-3) # cut next stream to start at start of next signal, w buffer
                if verbose:
                    print("no signal overlap")
            else:   # end of current signal after start of next signal
                current_stream.trim(endtime=ev_signal_startstop_sorted[j+1][0]-3-0.01) # if still overlapping, cut first stream at start of next signal, w buffer
                next_stream.trim(starttime=ev_signal_startstop_sorted[j+1][0]-3) # cut next stream to star of next signal, w buffer
                # break
                if verbose:
                    print("signal overlap")


        trimmed_streams += current_stream
    # add last stream not included in loop
    trimmed_streams += next_stream



    print("Time window: " + str(wv[0].stats.starttime) + " - " + str(wv[0].stats.endtime))
    print("(only time window of detected events saved)")


    dir_tmp = str(Path(model_name).parent / ("DOY" + str(t_pick.julday).zfill(3))) + "/"

    check_dir(dir_tmp)
    print(trimmed_streams)
    trimmed_streams.write(dir_tmp +  trimmed_streams[0].id[:-1] + "_denoised.mseed",format="MSEED")

    if saveraw=="True" or saveraw=="true":
        wv.write(dir_tmp + wv[0].id[:-1] + "_original.mseed",format="MSEED")

    trimmed_streams.merge().plot()
    wv.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise seismic data for specified network, station, and channel.")

    # Adding arguments
    # required
    parser.add_argument("network", type=str, help="Seismic network code (e.g., 'CH')")
    parser.add_argument("station", type=str, help="Seismic station code (e.g., 'DIX')")
    parser.add_argument("channel", type=str, help="Seismic channel code (e.g., 'BH')")
    parser.add_argument("start_time", type=str, help="UTC time, as str")
    parser.add_argument("duration", type=float, help="Duration in seconds")
    # optional
    parser.add_argument("--threshold",default=10.0, type=float, help="Collect detections with detection score above this threshold, e.g. 10")
    parser.add_argument("--saveraw",default="False", type=str, help="Save preprocessed raw data if set to True")
    parser.add_argument("--model_name", default="model_paper",type=str, help="Path to model name of model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--min_peak_height", type=float, help="Min. threshold of peak finder (0-1)")
    parser.add_argument("--client_str", type=str, help="e.g. ETH")

    # Parsing arguments
    args = parser.parse_args()

    # Calling the main function
    process_seismic_data(args.network,
                         args.station,
                         args.channel,
                         args.start_time,
                         args.duration,
                         args.verbose,
                         args.threshold,
                         args.saveraw,
                         args.model_name,
                         args.min_peak_height,
                         args.client_str)


# %%
