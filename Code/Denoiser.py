import tensorflow as tf
import obspy.core
from functools import cache
import numpy as np
from scipy.signal import stft, istft
import scipy
from DenoisingFunctions_public import select_channel, check_dir, normalize_percentile # client
from scipy.signal import find_peaks
from pathlib import Path

class Denoiser(object):
    def __init__(self, data_client, metadata_client, 
                 model_path, threshold, min_peak_height, verbose = False):
        
        self.threshold = threshold
        self.min_peak_height = min_peak_height
        self.data_client = data_client
        self.metadata_client = metadata_client
        self.verbose = verbose
        self.buffer = 600
        self.len_sample = 6120
        self.shift_samples = int(self.len_sample / 2)
        self.bins_overlap = 128
        self.model_name = model_path

        self.pre_filt = [1 / 100, 1 / 20, 45, 50]
        self.stft_parameters = {"nperseg": 48, "nfft": 126, "fs": 100, "noverlap": 24}
        
        if verbose:
            print("Loading model")
        
        self.model = tf.keras.models.load_model(model_path, compile=False)
        
        if verbose:
            print("Model ready")
        
    @cache
    def _query_server(self, network, station):
        return self.metadata_client.get_stations(network = network, station = station, location = "*", channel = "*", level="response")
        
                      
    def _get_metadata(self, network, station, location, channel, starttime, endtime):
        inventory = self._query_server(network, station)
        return inventory.select(network, station, location, channel, starttime, endtime)
    
    def _process_segment(self, data_window):
        """
        data_window: numpy array of shape (len_sample, 3)
                     Columns are Z, N, E components.
        Returns:
                (raw_stft, norm_stft) each shaped (1, 64, 256, 6)
        """

        # Expecting shape (len_sample, 3)
        if data_window.shape[0] != self.len_sample or data_window.shape[1] != 3:
            print("Returning none")
            return None

        stft_tmp       = np.zeros((64, 256, 6), dtype=float)
        stft_tmp_norm  = np.zeros((64, 256, 6), dtype=float)

        # Loop over 3 components: 0=Z, 1=N, 2=E
        for j in range(3):
            snippet_tmp = data_window[:, j]

            # STFT for one component
            _, _, _stft = scipy.signal.stft(snippet_tmp, **self.stft_parameters)

            # real/imag into CNN layout
            stft_tmp[:, :, j*2]     = _stft.real
            stft_tmp[:, :, j*2 + 1] = _stft.imag

            # Normalize (2 channels)
            block = np.stack((_stft.real, _stft.imag), axis=2)       # (64,256,2)
            block_norm = normalize_percentile(block)                 # (64,256,2)

            stft_tmp_norm[:, :, j*2]     = block_norm[:, :, 0]
            stft_tmp_norm[:, :, j*2 + 1] = block_norm[:, :, 1]

        # match old return shape
        return (
            stft_tmp[np.newaxis, ...],        # (1,64,256,6)
            stft_tmp_norm[np.newaxis, ...]
            )
        
        
    def compare_arrays_time_overlap(self, array1, array2,overlap=0.75):
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
    
    
    def get_mask_timeseries(self, mask_array):
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
        array_even = timeseries_3comp[0::2].reshape(-1)
        array_odd = timeseries_3comp[1::2].reshape(-1)
        return array_even, array_odd
    
    def get_peaks(self, timeseries, threshold=0.1,shift_correction=0):
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
    
        
    def run_data(self, network, station, location, channel, starttime, endtime, verbose = 0, plot_stuff = False):
        
        # starttime and endtime should be multiples of 61.2s
        # get data
        buffer = self.buffer
        data = self.data_client.get_waveforms(network, station, location,f"{channel}?",
                                              starttime - buffer, endtime + buffer)
        
        # get a continuous stream for further processing, no necessity to deal with gaps
        data.merge(fill_value='interpolate', method = 1)
        
        if len(data) != 3:
            print(f"Couldn't receive all data for {network}.{station}.{location}.{channel} {starttime} {endtime}")
            return None
        
        # epoch handling missing
        metadata = self._get_metadata(network, station, location, f"{channel}?",starttime, starttime)
        data.remove_response(inventory = metadata, output="VEL", pre_filt=self.pre_filt, water_level=None) # 60
        data.resample(100)
        data.trim(data[0].stats.starttime + buffer, data[0].stats.endtime - buffer)
        
        components = []
        for trace in data:
            components.append(trace.stats.channel[-1])
        components.sort(reverse = True)
        data_z = data[0].data
        data_n = data[1].data
        data_e = data[2].data
        
        data_stack = np.column_stack([data_z, data_n, data_e])
        
        utc_start_list = []
        _t = starttime
        while _t + 61.2 <= endtime:
            utc_start_list.append(_t)
            _t = _t + 61.2 / 2
        
        results = []
               
        for i in range(len(utc_start_list)):
            start = i * self.shift_samples
            end   = start + self.len_sample
            data_window = data_stack[start:end, :]
            results.append(self._process_segment(data_window))
        
        valid = [r for r in results if r is not None]
        selected_starttimes = [t for t, r in zip(utc_start_list, results) if r is not None]
        stft_collection = np.concatenate([result[0] for result in valid], axis=0)
        stft_norm_collection = np.concatenate([result[1] for result in valid], axis=0)
        print("Currently selected starttimes: ")
        print(selected_starttimes)
        # MAKE DL PREDICTION
        model_verbose = 0
        y_predict = self.model.predict(stft_norm_collection,verbose=model_verbose)
        if verbose:
            print("Inference / Applying model---%s seconds ---" %  (time.time() - start_time))

        t = np.linspace(0,61.2,256)
        bin_spacing = (255/256) * (t[1]-t[0])

        # DETECTING EVENT SIGNALS AND SELECTING PREFERRED TIME WINDOW
        # compute mask time series and extract non-overlapping windows as separate even and odd time series
        mask_timeseries_even, mask_timeseries_odd = self.get_mask_timeseries(y_predict)
        # get peaks with start and end, with fixed min. threshold of 0.1 for max of time series (=at leats one bin with mask value>0.1)
        peak_info_even = self.get_peaks(mask_timeseries_even, threshold=self.min_peak_height,shift_correction=128) # account for 50% time shift
        peak_info_odd = self.get_peaks(mask_timeseries_odd, threshold=self.min_peak_height,shift_correction=0)

        # combine and compare both detection results, keep better detection (higher score)
        filtered_results, origin = self.compare_arrays_time_overlap(peak_info_even, peak_info_odd)

        
        # Select data and mask based on list
        selected_stft, selected_masks, selected_utc = [], [], []
        detection_score, detection_start = [], []
    
        for filtered_result, even_odd in zip(filtered_results,origin):
            # check if "better" solution in even or odd-numbered row.
            print("Filtered result: ", filtered_result)
            if even_odd==0: # even-number row
    
                index_window = int(2 * ((filtered_result[1]+128) // 256))
                bin_start = (filtered_result[1]+self.bins_overlap) % 256
                if bin_start >250:
                    index_window += 2
                    bin_start = 0
              
                if index_window >= len(y_predict):
                    print("index_window > len(y_predict)")
                    continue
    
    
                selected_masks.append(y_predict[index_window])
                selected_stft.append(stft_collection[index_window])
                selected_utc.append(selected_starttimes[index_window])
                detection_start.append(selected_starttimes[index_window] + bin_start*bin_spacing)
    
                if plot_stuff:
                    plt.pcolormesh(y_predict[index_window,:,:,0],cmap="cubehelix_r",vmin=0,vmax=0.1)
                    plt.axvline((detection_start[-1]-selected_utc[-1])/bin_spacing)
            else:
                index_window = int(2 * (filtered_result[1] // 256) + 1)
    
                bin_start = filtered_result[1] % 256
    
                if bin_start >250:
                    index_window += 2
                    bin_start = 0
                if index_window >= len(y_predict):
                    print("index_window > len(y_predict)")
                    continue
    
    
                selected_masks.append(y_predict[index_window])
                selected_stft.append(stft_collection[index_window])
                selected_utc.append(selected_starttimes[index_window])
                detection_start.append(selected_starttimes[index_window] + bin_start*bin_spacing)
    
                if plot_stuff:
                    plt.pcolormesh(y_predict[index_window , :, :, 0], cmap="cubehelix_r", vmin=0, vmax=0.1)
                    plt.axvline((detection_start[-1]-selected_utc[-1])/bin_spacing)
    
            detection_score.append(filtered_result[3])
    
            if plot_stuff:
                plt.title([even_odd,filtered_result[3],filtered_result[1]])
                plt.show()
    
        selected_masks = np.array(selected_masks)
        selected_stft = np.array(selected_stft)
        print("Selected UTC:")
        print(selected_utc)
        # RECOMPUTE MASKS BASED ON DETECTION START
        shift_seconds = bin_spacing*42#10  # trying to align estimated signal start with binning
        first_stft = True
        stream_start_end = []
        new_window_start = []
        stft_collection_subset = np.zeros((len(detection_start), 64, 256, 6), dtype=np.float32)
        stft_norm_collection_subset = np.zeros((len(detection_start), 64, 256, 6), dtype=np.float32)
        print("Selections after filtering: ")
        print(detection_start)
        for i, _utc in enumerate(detection_start):
    
            stft_tmp, stft_tmp_norm = np.zeros((64, 256, 6)), np.zeros((64, 256, 6))
            
            # find start and end index
            
            startidx = int((_utc - data[0].stats.starttime - shift_seconds) * 100)
            endidx = startidx + 6120
            new_window_start.append(_utc-shift_seconds)
    
            if len(data_window) < self.len_sample:
                print("Not enough data, skipping")
                continue
            data_window = data_stack[startidx:endidx, :]
            print(_utc)
            print(data_window)
            print(data_window.shape)
            stft_tmp, stft_tmp_norm = self._process_segment(data_window)
            stft_collection_subset[i] = np.expand_dims(stft_tmp, axis=0)
            stft_norm_collection_subset[i] = np.expand_dims(stft_tmp_norm, axis=0)
            stream_start_end.append((_utc - shift_seconds,
                                     _utc + 65 - shift_seconds))
        
        # Make new prediction
        y_predict_event = self.model.predict(stft_norm_collection_subset,verbose=model_verbose)
        
        stft_final_subset, masks_subset, utc_start_subset = [], [], []
    # mask_timeseries_info = []
        ev_signal_startstop = []
        for i,y_event in enumerate(y_predict_event):
            # _timeseries = (2* np.max(y_event[:, :, 0], axis=0) + np.max(y_event[:, :, 2], axis=0) + np.max(y_event[:, :, 4], axis=0)) / 4
            _timeseries = (2* np.max(y_event[:, :, 0], axis=0) + np.max(y_event[:, :, 1], axis=0) + np.max(y_event[:, :, 2], axis=0)) / 4
            # print("3 model outputs")
            _peak = self.get_peaks(_timeseries, threshold= self.min_peak_height, shift_correction=0)
    
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
    
            if _score > self.threshold:
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
        
        num = stft_final_subset.shape[0]
        st_denoised_collection = obspy.Stream()
    
        for i in range(num):
            
            for j in range(3):
                stats = data[j].stats
                stats.starttime = utc_start_subset[i]
                _stft = stft_final_subset[i, :, :, j * 2] + 1j * stft_final_subset[i, :, :, j * 2 + 1]
                # _stft_denoised = _stft * masks_subset[i, :, :, j * 2]
                _stft_denoised = _stft * masks_subset[i, :, :, j] # 3 output model
    
                t_td, td_signal = istft(_stft_denoised,**self.stft_parameters)
                trace_data = td_signal
                stats.npts = len(trace_data)
                st_denoised_collection += obspy.core.Trace(trace_data, header = stats)    
        # sorting and packaging
        # stream trimming in place , need to run together with st_denoised_collection generation
        segments = [st_denoised_collection[3 * i: 3 * (i + 1)] for i in range(num)]  # get list of streams
    
        # start times of the first trace in each segment
        start_times = [segment[0].stats.starttime for segment in segments]  # get start time for list of streams
        sorted_indices = sorted(range(len(segments)), key=lambda i: start_times[i])  # get indices to sort in time
        segments_sorted = [segments[i] for i in sorted_indices]  # sort streams in time
        ev_signal_startstop_sorted = [ev_signal_startstop[i] for i in sorted_indices] # sort signal start/end list in time
        print("Startstop: ", len(ev_signal_startstop_sorted))
        for element in ev_signal_startstop_sorted:
            print(element)

        # trimming needs to be reimplemented to fit current data strucutre
        
        # dummy
        trimmed_streams = obspy.core.Stream()
        for trace in segments_sorted:
            print("Trace: ", trace)
            trimmed_streams += trace
        #trimmed_streams = obspy.core.Stream(segments_sorted)
    
    
        dir_tmp = str(Path(self.model_name).parent / ("DOY" + str(data[0].stats.starttime.julday).zfill(3))) + "/"
    
        check_dir(dir_tmp)
        print(trimmed_streams)
        trimmed_streams.write(dir_tmp +  trimmed_streams[0].id[:-1] + "_denoised.mseed",format="MSEED")
        trimmed_streams.plot()
        #if saveraw=="True" or saveraw=="true":
        #    data.write(dir_tmp + wv[0].id[:-1] + "_original.mseed",format="MSEED")

        
        pass