import obspy.core
import cupy as cp
import numpy as np
import scipy
import time
import logging
import threading
import queue
import tensorflow as tf

from obspy.signal.invsim import cosine_taper, cosine_sac_taper
from obspy.signal.util import _npts2nfft
from functools import cache
from DenoisingFunctions_public import check_dir, normalize_percentile
from scipy.signal import find_peaks
from pathlib import Path
from scipy.signal import istft

tf.config.set_visible_devices([], 'GPU')
_PROGRAM_START = time.perf_counter()

logger = logging.getLogger(__name__)

SENTINEL = object()


class RelativeTimeFormatter(logging.Formatter):
    """
    Class to make useful debugging output
    """

    def format(self, record):
        elapsed = time.perf_counter() - _PROGRAM_START
        record.relative_time = f"{elapsed:8.2f}s"
        return super().format(record)


def setup_logging(debug=False):
    """
    Makes the desired logging output.

    debug: boolean, if True, print verbose output
    """

    level = logging.DEBUG if debug else logging.INFO

    handler = logging.StreamHandler()
    formatter = RelativeTimeFormatter(
        fmt="[%(relative_time)s] %(name)s.%(funcName)s: %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


class Denoiser(object):
    """
    Module implementing denoising according to
    Nikolaj Dahmen; John Clinton; Men‐Andrin Meier; Luca Scarabello 'Toward
    Operational Earthquake Seismogram Denoising'
    https://doi.org/10.1785/0120250198

    I have made some tests with Cuda, the results not being satisfactory,
    i.e. the performance gain is minimal.

    The software, as it is, is first I/O bound and then memory bound and
    doesn't profit from GPUs, at least not with the current pricing scheme
    of ETH. This decision might have to be revisited at another stage when
    the environment changes.

    The GPU optimised code is left, but currently unused.

    The code uses a pipeline concept when used in multiday mode
    (run_timerange()). One thread reads the files using the client
    (typically fully I/O bound), one thread does the processing. This way,
    the next file can be loaded while the proceeding file is being
    processed.

    Authors: Niko Dahmen, Roman Racine
    """

    def __init__(self, data_client, metadata_client,
                 model_path, threshold, min_peak_height, debug=False):
        """
        data_client: obspy client to get data, SDS might be fastest if
        you use obspy >= 1.5.0

        metadata_client: obspy client to get metadata, e.g. fdsn web
                         service
        model_path: path to trained model
        threshold:
        min_peak_height:
        """

        self.threshold = threshold
        self.min_peak_height = min_peak_height
        self.data_client = data_client
        self.metadata_client = metadata_client
        self.buffer = 600
        self.len_sample = 6120
        self.shift_samples = int(self.len_sample / 2)
        self.bins_overlap = 128
        self.model_name = model_path

        t = np.linspace(0, 61.2, 256)
        self.bin_spacing = (255/256) * (t[1]-t[0])
        self.pre_filt = [1 / 100, 1 / 20, 45, 50]
        self.stft_parameters = {"nperseg": 48, "nfft": 126, "fs": 100,
                                "noverlap": 24}
        self.response_cache = {}
        self.model = tf.keras.models.load_model(model_path, compile=False)
        setup_logging(debug=debug)
        logger.debug("")

    def _loader_thread(self, startday, endday, network, station, location,
                       channel, output_queue):
        """
        This thread reads files using the provided client. This can run while
        the consumer thread is processing the proceeding file.

        startday: obspy.core.UTCDateTime, first day which should be processed
        endday: obspy.core.UTCDateTime, last day which should be processed
        output_queue: queue to which read files are put
        """

        logger.debug("")
        currentday = startday
        while currentday <= endday:
            data = self.data_client.get_waveforms(network, station,
                                                  location, f"{channel}?",
                                                  currentday - self.buffer,
                                                  currentday + 86400 +
                                                  self.buffer)
            output_queue.put((data, currentday, currentday + 86400,
                              network, station, location, channel))
            currentday += 86400
        output_queue.put(SENTINEL)

    def _consumer_thread(self, input_queue):
        """
        This threads consumes obspy.core.Stream objects from the queue and
        runs run_data() on it.

        input_queue: queue to consume from
        """

        logger.debug("")
        while True:
            content = input_queue.get()
            if content is SENTINEL:
                return
            (data, startday, endday, network, station, location, channel) \
                = content
            logger.debug(content)
            self.run_data(network, station, location, channel, startday,
                          endday, data)

    @cache
    def _query_server(self, network, station):
        """
        Queries an fdsn web server to get the full inventory for a given
        network and station.

        Uses caching to avoid multiple requests for the same data.

        network: network code
        station: station code

        returns: obspy inventory
        """

        logger.debug("")
        return self.metadata_client.get_stations(network=network,
                                                 station=station, location="*",
                                                 channel="*", level="response")

    def _get_metadata(self, network, station, location, channel,
                      starttime, endtime):
        """
        strips location, channel, starttime and endtime from request
        and then queries _query_server which then loads the full inventory
        for a station if needed and otherwise returns this information
        from cache. This should reduce queries to fdsnws.

        network: network code
        station: station code
        location: location code
        channel: channel code
        starttime: start time (obspy.core.UTCDateTime)
        endtime: end time (obspy.core.UTCDateTime)
        """

        logger.debug("")
        inventory = self._query_server(network, station)
        return inventory.select(network, station, location, channel,
                                starttime, endtime)

    def _get_response_parameters(self, data, inventory):
        """
        Computes the response parameters for a specific input length for
        a specific station configuration. These can then used be directly
        instead of being recomputed every time when remove_response is called.

        data: obspy.core.Stream: Input data for which response parameters
              should be computed
        inventory: Matching inventory (created by inventory.select()).

        Returns response parameters
        """

        network = inventory[0].code
        station = inventory[0][0].code
        key = inventory[0][0][0]
        npts = len(data[0].data)

        # we leave out channel code as the key, assuming that all channels have
        # the same response
        dictkey = (network, station, key.location_code, str(key.start_date),
                   npts)
        if dictkey in self.response_cache:
            logger.debug("Cache hit")
            return self.response_cache[dictkey]

        logger.debug("Cache miss")

        response = data[0]._get_response(inventory)
        taper_coeffs = cosine_taper(npts, 0.05,
                                    sactaper=True, halfcosine=False)
        nfft = _npts2nfft(npts)
        freq_response, freqs = \
            response.get_evalresp_response(data[0].stats.delta, nfft,
                                           output="VEL")
        freq_domain_taper = cosine_sac_taper(freqs, flimit=self.pre_filt)
        freq_response[0] = 0.0
        freq_response[1:] = 1.0 / freq_response[1:]
        self.response_cache[dictkey] = (taper_coeffs, freq_response, freqs,
                                        freq_domain_taper, nfft)
        return (taper_coeffs, freq_response, freqs, freq_domain_taper, nfft)

    def _fast_remove_response(self, data, inventory):
        """
        Remove instrument response using cached inverse frequency response.

        Used the relevant part of obspy.core.Trace and make use of possible
        caching for frequency_response, and tapers, as they are always
        the same for the same length and response

        Parameters
        ----------
        data : np.ndarray
            Time-domain signal, shape (n_samples,)

        inventory: obspy inventory

        Returns
        -------
        np.ndarray
            Corrected time-domain trace
        """

        npts = len(data[0].data)
        taper_coeffs, freq_response, freqs, freq_domain_taper, nfft \
            = self._get_response_parameters(data, inventory)

        for trace in data:
            channel = trace.data
            channel = channel.astype(np.float32)
            channel -= channel.mean()
            channel *= taper_coeffs
            spec = np.fft.rfft(channel, n=nfft)
            spec *= freq_domain_taper
            spec *= freq_response
            trace.data = np.fft.irfft(spec, n=npts)
        return

    def stft_gpu(self, signal_np, fs=100, nperseg=48, noverlap=24,
                 nfft=126, target_frames=256):
        """
        GPU-accelerated STFT using CuPy, written by microsoft copilot
        Matches SciPy STFT output shape: (freq, time) = (64, 256)

        Status: Works, but is not faster than original code,
        performance gains are lost when pre and postprocessing
        input and output

        Also: stft is not the bottle neck in this code
        """

        logger.debug("")
        # ---- CPU → GPU ----
        x = cp.asarray(signal_np, dtype=cp.float32)

        step = nperseg - noverlap
        needed_len = (target_frames - 1) * step + nperseg

        # ---- pad so we always get target_frames ----
        if x.shape[0] < needed_len:
            x = cp.pad(x, (0, needed_len - x.shape[0]))

        # ---- strided framing (NO copy!) ----
        shape = (target_frames, nperseg)
        strides = (x.strides[0] * step, x.strides[0])

        frames = cp.lib.stride_tricks.as_strided(
            x,
            shape=shape,
            strides=strides)

        # ---- windowing ----
        window = cp.hanning(nperseg)
        frames *= window

        # ---- FFT on GPU ----
        spec = cp.fft.rfft(frames, n=nfft, axis=1)

        # ---- return to CPU (freq x time) ----
        return cp.asnumpy(spec.T).astype(np.complex64)

    def _process_segment(self, data_window):
        """
        data_window: numpy array of shape (len_sample, 3)
                     Columns are Z, N, E components.
        Returns:
                (raw_stft, norm_stft) each shaped (1, 64, 256, 6)
        """

        logger.debug("")
        # Expecting shape (len_sample, 3)
        if data_window.shape[0] != self.len_sample or \
                data_window.shape[1] != 3:
            logger.debug("returning None")
            return None

        stft_tmp = np.zeros((64, 256, 6), dtype=float)
        stft_tmp_norm = np.zeros((64, 256, 6), dtype=float)

        # Loop over 3 components: 0=Z, 1=N, 2=E
        for j in range(3):
            snippet_tmp = data_window[:, j]

            # STFT for one component
            _, _, _stft = scipy.signal.stft(snippet_tmp,
                                            **self.stft_parameters)
    #        _stft = self.stft_gpu(snippet_tmp)

            # real/imag into CNN layout
            stft_tmp[:, :, j*2] = _stft.real
            stft_tmp[:, :, j*2 + 1] = _stft.imag

            # Normalize (2 channels)
            block = np.stack((_stft.real, _stft.imag), axis=2)
            block_norm = normalize_percentile(block)

            stft_tmp_norm[:, :, j*2] = block_norm[:, :, 0]
            stft_tmp_norm[:, :, j*2 + 1] = block_norm[:, :, 1]

        # match old return shape
        return (
            stft_tmp[np.newaxis, ...],        # (1,64,256,6)
            stft_tmp_norm[np.newaxis, ...]
            )

    def compare_arrays_time_overlap(self, array1, array2, overlap=0.75):
        """
        Compare two arrays of time intervals and select overlapping
        intervals with higher scores.

        Each row in the input arrays should contain:
        [peak, start_time, end_time, score, maxval].

        For each interval in array1, the function finds intervals in
        array2 that overlap by at least a fraction `overlap` of the smaller
        interval's duration. It keeps
        the interval with the higher score for overlapping pairs.

        Intervals in array2 not overlapping any interval in array1 are also
        included.

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

        logger.debug("")
        result = []  # To store resulting rows and origin
        # Track rows in array2 that have been processed
        used_indices_array2 = set()

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
        Extract two time series from a 4D mask array by computing a
        weighted mean of maximum mask values across selected channels
        at each time step, then splitting the result into even and odd
        time steps.

        The weighted mean gives double weight to the first channel and
        equal weight to the next two.

        Parameters:
        -----------
        mask_array : np.ndarray
            4D array where the last dimension indexes channels.

        array_odd : np.ndarray
            Concatenated values from odd-indexed time steps.
        """

        # extract time series of mask as mean value of max. / mean mask values
        # at each time step, equal weight for vertical and horizontal
        logger.debug("")
        timeseries_3comp = (2 * np.max(mask_array[:, :, :, 0], axis=1) +
                            np.max(mask_array[:, :, :, 1], axis=1) +
                            np.max(mask_array[:, :, :, 2], axis=1)) / 4

        timeseries_3comp.shape[0]
        # step through overlapping array
        array_even = timeseries_3comp[0::2].reshape(-1)
        array_odd = timeseries_3comp[1::2].reshape(-1)
        return array_even, array_odd

    def get_peaks(self, timeseries, threshold=0.1, shift_correction=0):
        """
        Detect peaks in a timeseries exceeding a given threshold
        and find their onset and end points.

        Peaks are detected using a minimum distance between peaks.
        For each peak, the function finds:
        - The left boundary where the signal falls below 0.01 before the peak.
        - The right boundary where the signal falls below 0.05 after the peak.
        - The sum of values between the left and right boundaries.
        - The peak value itself.

        threshold : float, optional (default=0.1)
            Minimum height of peaks to be detected.

        shift_correction : int, optional (default=0)
            Value subtracted from detected indices to adjust for any offset.

        Returns:
        --------
        np.ndarray
            Array of detected peaks with columns:
            [peak_index, left_boundary_index, right_boundary_index
            sum_between_boundaries, peak_value].
        """

        logger.debug("")
        peaks, _ = find_peaks(timeseries, height=threshold, distance=128)

        peaks_info = []
        for peak in peaks:
            left = np.where(timeseries[:peak] < 0.01)[0]
            left_index = left[-1] if len(left) else 0

            right = np.where(timeseries[peak:] < 0.05)[0]
            right_index = peak + right[0] \
                if len(right) else len(timeseries) - 1

            _mask_vals = np.sum(timeseries[left_index:right_index])
            peaks_info.append([peak-shift_correction,
                               left_index-shift_correction,
                               right_index-shift_correction,
                               _mask_vals, timeseries[peak]])

        return np.array(peaks_info)

    def _get_data(self, network, station, location, channel, starttime,
                  endtime, data=None):
        """
        computes restituted data ready for use.

        Returns restituted data, three components, gaps interpolated if
        necessary as a tuple: (data, data_stack)
        data: obspy.core.Stream containing the dat
        data_stack: numpy stack containing the numerical values of the three
                    channels
        """

        logger.debug("")
        buffer = self.buffer

        if not data:
            data = self.data_client.get_waveforms(network, station,
                                                  location, f"{channel}?",
                                                  starttime - buffer,
                                                  endtime + buffer)
        data.merge(fill_value='interpolate', method=1)
        if len(data) != 3:
            logger.debug("Couldn't receive all data for "
                         "{network}.{station}.{location}.{channel}"
                         "{starttime} {endtime}")
            return None

        metadata = self._get_metadata(network, station, location,
                                      f"{channel}?", starttime, starttime)

        if data[0].stats.sampling_rate % 100 == 0:
            data.filter("lowpass", freq=45.0, corners=4, zerophase=True)
            data.decimate(factor=int(data[0].stats.sampling_rate // 100),
                          no_filter=True)
        else:
            data.resample(100)
        self._fast_remove_response(data, metadata)
        data.trim(data[0].stats.starttime + buffer,
                  data[0].stats.endtime - buffer)

        components = []
        for trace in data:
            components.append(trace.stats.channel[-1])
        components.sort(reverse=True)

        data_stack = np.column_stack([data[0].data,
                                      data[1].data,
                                      data[2].data])

        return (data, data_stack)

    def _compute_stfts(self, data_stack, starttime, endtime):
        """
        Computes all stfts in the given window starttime, endtime,
        each for 61.2s, shifted by 61.2/2 s.

        returns selected_starttimes, stft_collection, stft_norm_collection
        selected_starttimes: Start time of all valid windows (windows
        which are shorter than 61.2s are not computed.
        stft_collection:
        stft_norm_collection:
        """

        logger.debug("")
        step = 61.2 / 2

        utc_start_list = [starttime + i * step
                          for i in range(int((endtime -
                                              starttime - 61.2) //
                                         step) + 1)]

        num_windows = (data_stack.shape[0] -
                       self.len_sample) // self.shift_samples + 1
        starts = np.arange(num_windows) * self.shift_samples
        windows = np.stack([data_stack[s:s+self.len_sample] for s in starts],
                           axis=0)
        results = [self._process_segment(w) for w in windows]

        valid = [r for r in results if r is not None]
        selected_starttimes = [t for t, r in zip(utc_start_list, results)
                               if r is not None]
        stft_collection = np.concatenate([result[0] for result in valid],
                                         axis=0)
        stft_norm_collection = np.concatenate([result[1] for result in valid],
                                              axis=0)

        return (selected_starttimes, stft_collection, stft_norm_collection)

    def _detect_event_signals(self, stft_norm_collection):
        """
        Detect event signals using detection algorithm
        """

        logger.debug("")
        model_verbose = 0

        y_predict = self.model.predict(stft_norm_collection,
                                       verbose=model_verbose)
        mask_timeseries_even, mask_timeseries_odd = \
            self.get_mask_timeseries(y_predict)
        # get peaks with start and end, with fixed min. threshold
        # of 0.1 for max of time series (=at leats one bin with mask value>0.1)

        # account for 50% time shift
        peak_info_even = self.get_peaks(mask_timeseries_even,
                                        threshold=self.min_peak_height,
                                        shift_correction=128)
        peak_info_odd = self.get_peaks(mask_timeseries_odd,
                                       threshold=self.min_peak_height,
                                       shift_correction=0)
        filtered_results, origin = \
            self.compare_arrays_time_overlap(peak_info_even, peak_info_odd)
        return (filtered_results, origin, y_predict)

    def _select_data_and_mask(self, filtered_results, origin, y_predict,
                              stft_collection, selected_starttimes):
        """
        if signal detected, make choice which stft window to use
        """

        logger.debug("")
        # Select data and mask based on list
        selected_stft, selected_masks, selected_utc = [], [], []
        detection_score, detection_start = [], []

        for filtered_result, even_odd in zip(filtered_results, origin):
            # check if "better" solution in even or odd-numbered row.
            if even_odd == 0:
                index_window = int(2 * ((filtered_result[1]+128) // 256))
                bin_start = (filtered_result[1]+self.bins_overlap) % 256
            else:
                index_window = int(2 * (filtered_result[1] // 256) + 1)
                bin_start = filtered_result[1] % 256

            if bin_start > 250:
                index_window += 2
                bin_start = 0

            if index_window >= len(y_predict):
                continue

            selected_masks.append(y_predict[index_window])
            selected_stft.append(stft_collection[index_window])
            selected_utc.append(selected_starttimes[index_window])
            detection_start.append(selected_starttimes[index_window] +
                                   bin_start*self.bin_spacing)
            detection_score.append(filtered_result[3])

        selected_masks = np.array(selected_masks)
        selected_stft = np.array(selected_stft)
        return (selected_masks, selected_stft, selected_utc,
                detection_start, detection_score)

    def _recompute_mask(self, detection_start, starttime, data_stack):
        """
        For detected signal, make optimised detection choosing optimal window
        """

        logger.debug("")
        # 10  # trying to align estimated signal start with binning
        shift_seconds = self.bin_spacing*42
        stream_start_end = []
        new_window_start = []
        stft_collection_subset = np.zeros((len(detection_start),
                                           64, 256, 6), dtype=np.float32)
        stft_norm_collection_subset = np.zeros((len(detection_start),
                                                64, 256, 6), dtype=np.float32)
        for i, _utc in enumerate(detection_start):
            # find start and end index
            startidx = int((_utc - starttime - shift_seconds) * 100)
            endidx = startidx + 6120
            new_window_start.append(_utc-shift_seconds)
            data_window = data_stack[startidx:endidx, :]
            if len(data_window) < self.len_sample:
                logger.info("Not enough data, skipping")
                continue
            stft_tmp, stft_tmp_norm = self._process_segment(data_window)
            stft_collection_subset[i] = np.expand_dims(stft_tmp, axis=0)
            stft_norm_collection_subset[i] = np.expand_dims(stft_tmp_norm,
                                                            axis=0)
            stream_start_end.append((_utc - shift_seconds,
                                     _utc + 65 - shift_seconds))

        return (stft_collection_subset, stft_norm_collection_subset,
                stream_start_end)

    def _make_final_selection(self, y_predict_event, filtered_results,
                              detection_start, selected_stft,
                              selected_masks, selected_utc,
                              stft_collection_subset, stream_start_end, data):
        """
        Choose the best out of all computed results for a given earthquake.
        """

        logger.debug("")
        stft_final_subset, masks_subset, utc_start_subset = [], [], []
        stream_start_end_final = []
        for i, y_event in enumerate(y_predict_event):
            _timeseries = (2 * np.max(y_event[:, :, 0], axis=0) +
                           np.max(y_event[:, :, 1], axis=0) +
                           np.max(y_event[:, :, 2], axis=0)) / 4
            _peak = self.get_peaks(_timeseries,
                                   threshold=self.min_peak_height,
                                   shift_correction=0)
            keep_old = True
            _score = filtered_results[i][3]

            # if peak found, check if score of new time window is higher
            if len(_peak) > 0:
                # new score higher (not much lower) than old score,
                # collect new window
                if _peak[0][3] > 0.5*filtered_results[i][3]:
                    _score = _peak[0][3]
                    keep_old = False

            if _score > self.threshold:
                if keep_old:
                    stft_final_subset.append(selected_stft[i])
                    masks_subset.append(selected_masks[i])
                    utc_start_subset.append(selected_utc[i])
                    detect_duration = filtered_results[i][2] -\
                        filtered_results[i][1]
                    stream_start_end_final.append((detection_start[i],
                                                   detection_start[i] +
                                                   detect_duration))
                else:
                    stft_final_subset.append(stft_collection_subset[i])
                    masks_subset.append(y_event)
                    utc_start_subset.append(stream_start_end[i][0])
                    stream_start_end_final.append([stream_start_end[i][0] +
                                                   _peak[0][1] *
                                                   self.bin_spacing,
                                                   stream_start_end[i][0] +
                                                   _peak[0][2] *
                                                   self.bin_spacing])

        masks_subset = np.array(masks_subset)
        stft_final_subset = np.array(stft_final_subset)
        num = stft_final_subset.shape[0]
        st_denoised_collection = obspy.Stream()

        for i in range(num):

            for j in range(3):
                stats = data[j].stats
                stats.starttime = utc_start_subset[i]
                _stft = stft_final_subset[i, :, :, j * 2] + 1j * \
                    stft_final_subset[i, :, :, j * 2 + 1]
                # _stft_denoised = _stft * masks_subset[i, :, :, j * 2]
                # 3 output model
                _stft_denoised = _stft * masks_subset[i, :, :, j]

                t_td, td_signal = istft(_stft_denoised, **self.stft_parameters)
                trace_data = td_signal
                stats.npts = len(trace_data)
                st_denoised_collection += obspy.core.Trace(trace_data,
                                                           header=stats)
        # sorting and packaging
        # stream trimming in place , need to run together with
        # st_denoised_collection generation
        segments = [st_denoised_collection[3 * i: 3 * (i + 1)]
                    for i in range(num)]

        # start times of the first trace in each segment
        start_times = [segment[0].stats.starttime for segment in segments]
        sorted_indices = sorted(range(len(segments)),
                                key=lambda i: start_times[i])
        trimmed_streams = obspy.core.Stream()
        _tmp_stream = []
        for i in sorted_indices:
            trimmed_streams += segments[i]
            _tmp_stream.append(stream_start_end_final[i])
        return (trimmed_streams, _tmp_stream)

    def _output(self, starttime, trimmed_streams):
        """
        Writes results to the disk, print output, might also plot etc.
        """

        logger.debug("")

        output_stream = obspy.core.Stream()
        for i in range(3):
            for j in range(int(len(trimmed_streams) // 3)):
                output_stream += trimmed_streams[3*j + i]
        output_stream._cleanup()
        if not len(trimmed_streams):
            logger.debug("No events found")
            return
        dir_tmp = str(Path(self.model_name).parent /
                      ("DOY" + str(starttime.julday).zfill(3))) + "/"
        check_dir(dir_tmp)
        output_stream.write(dir_tmp +
                            trimmed_streams[0].id[:-1] +
                            "_denoised.mseed",
                            format="MSEED")

    def _trim_streams(self, trimmed_streams, startstop):
        """
        Handle cases where traces of two subsequent picks overlap
        """

        logger.debug("")
        new_trimmed_stream = obspy.core.Stream()
        for i in range(0, len(startstop)-1):
            # no overlap
            if trimmed_streams[3*i].stats.endtime < \
                    trimmed_streams[3*(i+1)].stats.starttime:
                for j in range(0, 3):
                    new_trimmed_stream += trimmed_streams[3*i+j]
                logger.debug("No overlap")
                continue

            # overlap, but signal part of preceding set ends
            # before start of next data set
            if startstop[i][1] < trimmed_streams[3*(i+1)].stats.starttime:
                logger.debug("No signal overlap with next stream")
                for j in range(0, 3):
                    new_trimmed_stream += trimmed_streams[3*i+j].slice(
                        endtime=startstop[i][1]-0.01)
            # overlap, but signal part of preceding set ends before signal
            # part of next data set
            elif startstop[i][1] + 3 < startstop[i+1][0]:
                logger.debug("no signal overlap")

                for j in range(0, 3):
                    new_trimmed_stream += \
                        trimmed_streams[3*i+j].slice(
                            endtime=startstop[i][1] - 0.01
                            )
                    trimmed_streams[3*(i+1)+j] = \
                        trimmed_streams[3*(i+1)+j].slice(
                            starttime=startstop[i+1][0] - 3
                            )
            # overlap not resolvable
            else:
                logger.debug("signal overlap")
                for j in range(0, 3):
                    new_trimmed_stream += \
                        trimmed_streams[3*i+j].slice(
                            endtime=startstop[i+1][1] - 3 - 0.01
                            )
                    trimmed_streams[3*(i+1)+j] = \
                        trimmed_streams[3*(i+1)+j].slice(
                            starttime=startstop[i+1][0] - 3
                            )
        if len(startstop) > 1:
            for j in range(0, 3):
                new_trimmed_stream += trimmed_streams[3*i+j]
        return new_trimmed_stream

    def run_timerange(self, network, station, location, channel,
                      startday, endday):
        """
        Run multiple days in one go. This optimises the running time as
        repsonses only need to be computed once.
        network: network code
        station: station code
        location: location code
        channel: channel code
        startday: first day which should be processed (obspy.core.UTCDateTime)
        endday: last day which should be processed (obspy.core.UTCDateTime)
        """

        logger.debug("")
        day_queue = queue.Queue(maxsize=1)
        loader = threading.Thread(
            target=self._loader_thread,
            args=(startday, endday, network, station, location,
                  channel, day_queue),
            daemon=True,
        )
        loader.start()
        self._consumer_thread(day_queue)

    def run_data(self, network, station, location, channel, starttime,
                 endtime, data=None):
        """
        Principal method, driving the rest of the code.

        Parameters:
        network: network code
        station: station code
        location: location code
        channel: channel code
        starttime: start time (obspy.core.UTCDateTime)
        endtime: end time  (obspy.core.UTCDateTime)

        Returns: writes to disk, currently doesn't return anything.
        """

        logger.debug("")

        model_verbose = 0
        # starttime and endtime should be multiples of 61.2s

        # get data and bring it into shape
        data, data_stack = self._get_data(network, station, location, channel,
                                          starttime, endtime, data)

        if not data:
            return None

        # compute stfts
        selected_starttimes, stft_collection, stft_norm_collection =  \
            self._compute_stfts(data_stack, starttime, endtime)

        # make prediction
        filtered_results, origin, y_predict = \
            self._detect_event_signals(stft_norm_collection)

        # make selection
        selected_masks, selected_stft, selected_utc, detection_start, \
            detection_score = \
            self._select_data_and_mask(filtered_results, origin, y_predict,
                                       stft_collection, selected_starttimes)

        # recompute mask
        stft_collection_subset, stft_norm_collection_subset, \
            stream_start_end = \
            self._recompute_mask(detection_start,
                                 data[0].stats.starttime,
                                 data_stack)

        # Make new prediction
        y_predict_event = self.model.predict(stft_norm_collection_subset,
                                             verbose=model_verbose)

        trimmed_streams, stream_start_end = \
            self._make_final_selection(y_predict_event,
                                       filtered_results,
                                       detection_start,
                                       selected_stft,
                                       selected_masks,
                                       selected_utc,
                                       stft_collection_subset,
                                       stream_start_end, data)

        trimmed_streams = self._trim_streams(trimmed_streams, stream_start_end)

        # output
        self._output(data[0].stats.starttime, trimmed_streams)
