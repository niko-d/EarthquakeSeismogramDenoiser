import obspy.core
# import cupy as cp
import numpy as np
import scipy
import time
import logging
import threading
import queue
import json                                         # NEW: for _save_picks()
import tensorflow as tf
from tensorflow.keras.layers import Layer
from obspy.signal.invsim import cosine_taper, cosine_sac_taper
from obspy.signal.util import _npts2nfft
from functools import cache
from DenoisingFunctions_public import check_dir, normalize_percentile
from scipy.signal import find_peaks
from pathlib import Path
from scipy.signal import istft
from obspy.signal.invsim import cosine_taper, cosine_sac_taper
from obspy.signal.util import _npts2nfft
from obspy import Stream, Trace  # NEW: used in _stream_tta()
from concurrent.futures import ThreadPoolExecutor, as_completed  # NEW: used in _process_picks()
tf.config.set_visible_devices([], 'GPU')
_PROGRAM_START = time.perf_counter()

logger = logging.getLogger(__name__)

SENTINEL = object()


# run_data()
#  ├── _round_to_window()               ← snap endtime to exact multiple of 61.2 s
#  ├── _get_data()                      ← fetch, resample, remove response, gap detection
#  ├── _compute_stfts()                 ← sliding window STFT over full day
#  ├── _detect_event_signals()          ← EQS first pass, peak detection on mask timeseries
#  ├── _select_data_and_mask()          ← select best STFT window per detection
#  ├── _recompute_mask()                ← re-align window to estimated signal start
#  ├── model.predict()                  ← EQS second pass on re-aligned windows
#  ├── _make_final_selection()          ← window scoring/selection, returns arrays; A ≤ D accepted
#  ├── _apply_eqshyb()                  ← optional, only if eqs2_model loaded and A > 0
#  |    └── eqs2_model.predict()        ← hybrid time-domain refinement (inputs: noisy + EQS denoised + EQS mask)
#  ├── _build_streams()                 ← ISTFT + stream assembly, EQS or EQShyb path; sorts by starttime
#  ├── _filter_close_detections_streams() ← remove near-duplicate detections, keep higher-scoring
#  ├── _trim_streams()                  ← resolve overlapping detections, apply signal buffer
#  ├── _output()                        ← zero-fill gaps, write MiniSEED to disk
#  ├── _pick()                          ← optional, only if picker configured and snippets exist
#  |    ├── _get_designaled_noise()     ← per-snippet noise = original - denoised
#  |    └── _process_picks()            ← parallel TTA picking loop (ThreadPoolExecutor)
#  |         └── _process_snippet()    ← per-detection: TTA augmentation + phase picking + polarity
#  |              ├── _stream_tta()    ← inject std-scaled white noise, seeded by TTA id
#  |              ├── picker.annotate()           ← SeisBench batch annotation
#  |              ├── picker.classify_aggregate() ← aggregate TTA picks
#  |              ├── _process_peak_times()       ← cluster picks + uncertainty per phase
#  |              |    ├── _cluster_picks()       ← group nearby picks, one per event
#  |              |    |    └── _weighted_median() ← confidence-weighted pick time
#  |              |    └── _tta_uncertainty()     ← timing spread across TTA reps
#  |              |         └── _weighted_std()  ← confidence-weighted std
#  |              └── _predict_polarity_tta()     ← optional, per accepted P pick; reuses TTA Z collection
#  └── _save_picks()                    ← scale uncertainties, write picks JSON to disk (same DOY directory)


def apply_pre_filt(data, samp_rate, pre_filt,taper_seconds=300):
    """Apply ObsPy's remove_response pre_filt step (no response correction).

    Calls ObsPy functions directly. Reproduces the pre_filt block of
    obspy.core.trace.Trace.remove_response with defaults:
        zero_mean=True, taper=True, taper_fraction=0.05

    Parameters
    ----------
    data      : array-like        Raw time-domain signal.
    samp_rate : float             Sample rate in Hz.
    pre_filt  : (f1, f2, f3, f4) Bandpass corner frequencies in Hz.

    Returns
    -------
    ndarray float64  Pre-filtered signal in the time domain.
    """
    data = np.array(data, dtype=np.float64)
    npts = len(data)

    data -= data.mean()

    p_fraction = (taper_seconds * samp_rate) / npts
    data *= cosine_taper(npts, p=p_fraction, sactaper=True, halfcosine=False)

    nfft  = _npts2nfft(npts)
    spec  = np.fft.rfft(data, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / samp_rate)

    spec *= cosine_sac_taper(freqs, flimit=pre_filt)

    # return np.fft.irfft(spec)[0:npts]
    return np.fft.irfft(spec,n=nfft)[0:npts]


def apply_pre_filt_trace(trace, pre_filt,taper_seconds=300):
    """Apply pre_filt to a single ObsPy Trace, returns a new Trace.

    Parameters
    ----------
    trace    : obspy.Trace        Input trace (not modified).
    pre_filt : (f1, f2, f3, f4)  Bandpass corner frequencies in Hz.

    Returns
    -------
    obspy.Trace  Copy with pre-filtered data (float64).
    """
    out = trace.copy()
    out.data = apply_pre_filt(trace.data, trace.stats.sampling_rate, pre_filt,taper_seconds=taper_seconds)
    return out


def apply_pre_filt_stream(stream, pre_filt,taper_seconds=300):
    """Apply pre_filt to every trace in an ObsPy Stream, returns a new Stream.

    Parameters
    ----------
    stream   : obspy.Stream       Input stream (not modified).
    pre_filt : (f1, f2, f3, f4)  Bandpass corner frequencies in Hz.

    Returns
    -------
    obspy.Stream  New stream with pre-filtered traces (float64).
    """
    return Stream([apply_pre_filt_trace(tr, pre_filt,taper_seconds=taper_seconds) for tr in stream])

def _predict_polarity_tta(
    z_tta_collection,
    z_starttime,
    z_sampling_rate,
    p_pick,
    polarity_model,
    win=256,
    threshold=0.33,
):
    """
    Polarity prediction using the already-augmented TTA Z traces from
    _stream_tta — no re-augmentation or noise re-scaling needed.

    z_tta_collection  : obspy.Stream, full TTA collection from _stream_tta,
                        containing repeat Z/N/E augmented traces
    z_starttime       : obspy.UTCDateTime, starttime of the original Z snippet
                        (before the +add padding in _process_snippet)
    z_sampling_rate   : float, samples per second
    p_pick            : obspy.UTCDateTime, accepted P pick time
    polarity_model    : tf.keras.Model, input shape (batch, win) or (batch, win, 1)
    win               : int, sample window centred on P pick (default 256)
    threshold         : float, min winning class probability; below → undecidable

    Returns dict:
        label            : str, 'positive' | 'negative' | 'undecidable'
        probabilities    : np.ndarray shape (3,), mean softmax over TTA batch
        all_predictions  : np.ndarray shape (repeat, 3), per-repetition softmax
    """
    labels = np.array(["negative", "undecidable", "positive"])

    p_idx = int(round((p_pick - z_starttime) * z_sampling_rate))
    half  = win // 2

    batch = []
    for tr in z_tta_collection.select(component='Z'):
        z = np.asarray(tr.data, dtype=np.float32)
        z_win = np.zeros(win, dtype=np.float32)
        start = p_idx - half
        src0  = max(start, 0)
        src1  = min(start + win, z.shape[0])
        dst0  = src0 - start
        z_win[dst0: dst0 + (src1 - src0)] = z[src0:src1]
        batch.append(z_win)

    z_batch = np.stack(batch, axis=0)                                    # (repeat, win)
    abs_max = np.maximum(np.max(np.abs(z_batch), axis=1, keepdims=True), 1e-20)
    z_batch /= abs_max

    if polarity_model.input_shape[-1] == 1:
        z_batch = z_batch[:, :, np.newaxis]                              # (repeat, win, 1)

    pred      = polarity_model(z_batch, training=True).numpy()           # (repeat, 3)
    mean_pred = pred.mean(axis=0)                                        # (3,)

    label = labels[np.argmax(mean_pred)]
    if mean_pred.max() < threshold:
        label = "undecidable"

    return {
        "label":           label,
        "probabilities":   mean_pred,
        "all_predictions": pred,
    }


class MaxAbsNorm1D(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x):
        # x: (B, T, C)
        m = tf.reduce_max(tf.abs(x), axis=1, keepdims=True)
        m = tf.maximum(m, self.eps)
        return x / m

@tf.keras.utils.register_keras_serializable(package="custom")
class ReflectPad1D(Layer):
    def __init__(self, pad, **kwargs):
        super().__init__(**kwargs)
        self.pad = int(pad)

    def call(self, x):
        if self.pad <= 0:
            return x
        return tf.pad(x, [[0, 0], [self.pad, self.pad], [0, 0]], mode="REFLECT")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"pad": self.pad})
        return cfg

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
                 model_path, min_peak_height, eqs2_model_path=None,
                 picker=None, picking_kwargs=None,
                 polarity_model_path=None, polarity_kwargs=None,
                 debug=False):
        """
        data_client          : obspy client to get data
        metadata_client      : obspy client to get metadata
        model_path           : path to trained EQS model
        min_peak_height      : min peak height for detection
        eqs2_model_path      : optional path to EQShyb model
        picker               : optional SeisBench picker for phase picking
        picking_kwargs       : optional dict of kwargs passed to _process_picks
        polarity_model_path  : optional path to polarity model; if given the
                               model is loaded and applied to every accepted
                               P pick; expects input (batch, 256) or
                               (batch, 256, 1)
        polarity_kwargs      : optional dict, currently supports key
                               'threshold' (float, default 0.33) — minimum
                               winning class probability to accept a polarity
                               label, below which 'undecidable' is returned
        debug                : enable verbose logging
        """

        self.min_peak_height = min_peak_height
        self.data_client = data_client
        self.metadata_client = metadata_client
        self.threshold = 10
        self.buffer = 300
        self.len_sample = 6120
        self.shift_samples = int(self.len_sample / 2)
        self.bins_overlap = 128
        self.model_name = model_path
        self.pre_filt = [1 / 100, 1 / 20, 45, 50]
        self.stft_parameters = {"nperseg": 48, "nfft": 126, "fs": 100,"noverlap": 24}
        self.REALIGN_SCORE_TOLERANCE = 0.5#1 # 0.5 # NEW added
        self.signal_buffer_s = 3.0  # buffer to start save denoised stream with at least 3s before signal start (ideally)
        self.one_sample_s = 1.0 / self.stft_parameters["fs"]  # = 0.01s at 100 Hz

        # t = np.linspace(0, 61.2, 256)  # OLD
        # self.bin_spacing = (255/256) * (t[1]-t[0])  # OLD
        self.bin_spacing = (self.stft_parameters["nperseg"] - self.stft_parameters["noverlap"]) / self.stft_parameters["fs"]  # = 0.24

        self.response_cache = {}
        self.model = tf.keras.models.load_model(model_path, compile=False)

        # EQShyb / EQS2
        self.eqs2_model = tf.keras.models.load_model(
            eqs2_model_path,
            custom_objects={"ReflectPad1D": ReflectPad1D},
            compile=False
        ) if eqs2_model_path else None

        # PICKER
        self.picker = picker
        self.picking_kwargs = picking_kwargs or {}
        self.uncertainty_scaling = {
            'p_picks': {'scale_sample': 4*1.904, 'offset_sample': 9.249},
            's_picks': {'scale_sample': 4*2.211, 'offset_sample': 3.600},
        }
        # POLARITY
        self.polarity_model = tf.keras.models.load_model(
            polarity_model_path,
            custom_objects={"custom>MaxAbsNorm1D": MaxAbsNorm1D},
            compile=False
        ) if polarity_model_path else None
        self.polarity_threshold = (polarity_kwargs or {}).get('threshold', 0.33)


        self.components = None  # set in _get_data()


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
            day_end = self._round_to_window(currentday, currentday + 86400)  # NEW use multiple of 61.2s, used below
            data = self.data_client.get_waveforms(network, station,
                                                  location, f"{channel}?",
                                                  currentday - self.buffer,
                                                  day_end + self.buffer)
            output_queue.put((data, currentday, day_end,
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
        # taper length fraction
        p_fraction = (self.buffer * data[0].stats.sampling_rate) / npts

        taper_coeffs = cosine_taper(npts, p_fraction,
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
            channel = trace.data.astype(np.float32)
            channel -= channel.mean()
            channel *= taper_coeffs
            spec = np.fft.rfft(channel, n=nfft)
            spec *= freq_domain_taper
            spec *= freq_response
            # trace.data = np.fft.irfft(spec, n=npts)  # OLD
            trace.data = np.fft.irfft(spec, n=nfft)[:npts]  # NEW

        return

    # def stft_gpu(self, signal_np, fs=100, nperseg=48, noverlap=24,
    #              nfft=126, target_frames=256):
    #     """
    #     GPU-accelerated STFT using CuPy, written by microsoft copilot
    #     Matches SciPy STFT output shape: (freq, time) = (64, 256)
    #
    #     Status: Works, but is not faster than original code,
    #     performance gains are lost when pre and postprocessing
    #     input and output
    #
    #     Also: stft is not the bottle neck in this code
    #     """
    #
    #     logger.debug("")
    #     # ---- CPU → GPU ----
    #     x = cp.asarray(signal_np, dtype=cp.float32)
    #
    #     step = nperseg - noverlap
    #     needed_len = (target_frames - 1) * step + nperseg
    #
    #     # ---- pad so we always get target_frames ----
    #     if x.shape[0] < needed_len:
    #         x = cp.pad(x, (0, needed_len - x.shape[0]))
    #
    #     # ---- strided framing (NO copy!) ----
    #     shape = (target_frames, nperseg)
    #     strides = (x.strides[0] * step, x.strides[0])
    #
    #     frames = cp.lib.stride_tricks.as_strided(
    #         x,
    #         shape=shape,
    #         strides=strides)
    #
    #     # ---- windowing ----
    #     window = cp.hanning(nperseg)
    #     frames *= window
    #
    #     # ---- FFT on GPU ----
    #     spec = cp.fft.rfft(frames, n=nfft, axis=1)
    #
    #     # ---- return to CPU (freq x time) ----
    #     return cp.asnumpy(spec.T).astype(np.complex64)

    def _process_segment(self, data_window):
        """
        data_window: numpy array of shape (len_sample, 3)
                     Columns are Z, N, E components.
        Returns:
                (raw_stft, norm_stft) each shaped (64, 256, 6)
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

        return stft_tmp, stft_tmp_norm # (64,256,6)
        # return (
        #     stft_tmp[np.newaxis, ...],        # (1,64,256,6)
        #     stft_tmp_norm[np.newaxis, ...]
        #     )

    def _compare_arrays_time_overlap(self, array1, array2, overlap=0.75):
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
                # Check if overlap is >X% of the smaller window
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

    def _get_mask_timeseries(self, mask_array):
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

        # NEW collect gaps
        gap_list = data.get_gaps()
        gap_intervals = [(g[4], g[5]) for g in gap_list]
        # data.merge(fill_value='interpolate', method=1)
        data.merge(fill_value=0, method=1)

        if len(data) != 3:
            logger.debug("Couldn't receive all data for "
                         "{network}.{station}.{location}.{channel}"
                         "{starttime} {endtime}")
            return None

        metadata = self._get_metadata(network, station, location,
                                      f"{channel}?", starttime, starttime)

        # apply filter as in obspy remove_response prefilter & remove any other AA filter
        data = apply_pre_filt_stream(data, self.pre_filt,taper_seconds=self.buffer)

        if data[0].stats.sampling_rate % 100 == 0:
            data.decimate(factor=int(data[0].stats.sampling_rate // 100),no_filter=True)
        else:
            # data.filter("lowpass", freq=45.0, corners=8, zerophase=False)  # NEW - add filter ???
            data.resample(100,no_filter=True) # no filter default, additioonal AA off by frequency taper

        self._fast_remove_response(data, metadata)

        data.trim(data[0].stats.starttime + buffer,
                  data[0].stats.endtime - buffer)


        self.components = sorted([tr.stats.channel[-1] for tr in data], reverse=True)  # NEW get components and fix order in data


        # z comp first, other componets abitrarily
        data_stack = np.column_stack([
            data.select(component=self.components[0])[0].data,
            data.select(component=self.components[1])[0].data,
            data.select(component=self.components[2])[0].data
        ])

        logger.info(f"data_stack columns (components={self.components}): "
                    f"col0_std={data_stack[:, 0].std():.12f}, "
                    f"col1_std={data_stack[:, 1].std():.12f}, "
                    f"col2_std={data_stack[:, 2].std():.12f}, "
                    f"col0_shape={np.shape(data_stack[:, 0])}, "
                    f"col1_shape={np.shape(data_stack[:, 1])}, "
                    f"col2_shape={np.shape(data_stack[:, 2])}, "
                    f"col0_second_half_std={data_stack[data_stack.shape[0] // 2:, 0].std():.12f}, "
                    f"col0_second_half_max={np.abs(data_stack[data_stack.shape[0] // 2:, 0]).max():.12f}")

        # return (data, data_stack)  # OLD
        return (data, data_stack, gap_intervals)  # NEW
    def _compute_stfts(self, data_stack, starttime, endtime):
        """
        Computes all STFTs in the given window, each 61.2s, shifted by 30.6s (50% overlap).

        starttime: UTCDateTime, actual start of data_stack[0] (data[0].stats.starttime),
                   used for sample-accurate UTC timestamp generation.
        endtime:   UTCDateTime, end of processing window, used only for logging.

        returns:
        selected_starttimes: list of UTCDateTime, start time of each valid window
        stft_collection:     np.ndarray (W, 64, 256, 6), raw STFT all valid windows
        stft_norm_collection: np.ndarray (W, 64, 256, 6), normalised STFT all valid windows
        """

        logger.debug("")
        # step = 61.2 / 2

        num_windows = (data_stack.shape[0] - self.len_sample) // self.shift_samples + 1
        utc_start_list = [starttime + i * self.shift_samples / self.stft_parameters["fs"]
                          for i in range(num_windows)]

        starts = np.arange(num_windows) * self.shift_samples
        windows = np.stack([data_stack[s:s+self.len_sample] for s in starts],
                           axis=0)
        results = [self._process_segment(w) for w in windows]

        valid = [r for r in results if r is not None]
        selected_starttimes = [t for t, r in zip(utc_start_list, results)
                               if r is not None]
        # stft_collection = np.concatenate([result[0] for result in valid],
        #                                  axis=0)
        stft_collection = np.stack([result[0] for result in valid], axis=0)  # (N,64,256,6)
        # stft_norm_collection = np.concatenate([result[1] for result in valid],
        #                                       axis=0)
        stft_norm_collection = np.stack([result[1] for result in valid], axis=0)  # (N,64,256,6)
        return (selected_starttimes, stft_collection, stft_norm_collection)

    def _detect_event_signals(self, stft_norm_collection):
        """
        Detect event signals using detection algorithm
        """

        logger.debug("")
        model_verbose = 0

        y_predict = self.model.predict(stft_norm_collection.astype(np.float32),
                                       verbose=model_verbose)


        mask_timeseries_even, mask_timeseries_odd = \
            self._get_mask_timeseries(y_predict)

        # get peaks with start and end, with fixed min. threshold
        #  for max of time series (=at leats one bin with mask value>0.1)

        # account for 50% time shift
        peak_info_even = self.get_peaks(mask_timeseries_even,
                                        threshold=self.min_peak_height,
                                        shift_correction=128)
        peak_info_odd = self.get_peaks(mask_timeseries_odd,
                                       threshold=self.min_peak_height,
                                       shift_correction=0)



        filtered_results, origin = \
            self._compare_arrays_time_overlap(peak_info_even, peak_info_odd)

        logger.info(f"_detect_event_signals: {len(filtered_results)} detections found "
                    f"(even peaks: {len(peak_info_even)}, odd peaks: {len(peak_info_odd)})")
        if len(filtered_results):
            logger.info(f"  peak index range: {filtered_results[:, 0].min():.0f} — {filtered_results[:, 0].max():.0f} "
                        f"(of {len(mask_timeseries_even) + len(mask_timeseries_odd)} total bins; "
                        f"bin spacing {self.bin_spacing}s → "
                        f"last detection ~{filtered_results[:, 0].max() * self.bin_spacing / 3600:.1f}h into data)")

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

            if bin_start > 250:  # if detection in end of window
                index_window += 2
                bin_start = 0

            if index_window >= len(y_predict):
                logger.info(f"_select_data_and_mask: {len(detection_start)} selected from "
                            f"{len(filtered_results)} detections "
                            f"({len(filtered_results) - len(detection_start)} dropped: index out of range)")
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
        # new_window_start = []
        stft_collection_subset = np.zeros((len(detection_start),
                                           64, 256, 6), dtype=np.float32)
        stft_norm_collection_subset = np.zeros((len(detection_start),
                                                64, 256, 6), dtype=np.float32)
        for i, _utc in enumerate(detection_start):
            # find start and end index
            startidx = int((_utc - starttime - shift_seconds) * self.stft_parameters["fs"])
            endidx = startidx + 6120
            # new_window_start.append(_utc-shift_seconds)
            data_window = data_stack[startidx:endidx, :]
            if len(data_window) < self.len_sample:
                logger.info("Not enough data, skipping")
                stream_start_end.append(None)  # NEW maintain index alignment
                continue
            # NEW
            stft_result, stft_norm_result = self._process_segment(data_window)
            stft_collection_subset[i] = stft_result  # remove the batch dim
            stft_norm_collection_subset[i] = stft_norm_result

            stream_start_end.append((_utc - shift_seconds,
                                     _utc + 65 - shift_seconds))

        return (stft_collection_subset, stft_norm_collection_subset,
                stream_start_end)

    def _make_final_selection(self, y_predict_event, filtered_results,
                              detection_start, selected_stft,
                              selected_masks, selected_utc,
                              stft_collection_subset, stream_start_end):# , denoised_hyb=None):
        """
        Choose the best out of all computed results for a given earthquake.
        """

        logger.debug("")
        stft_final_subset, masks_subset, utc_start_subset = [], [], []
        stream_start_end_final = []
        scores_final = []  # NEW
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
                # if _peak[0][3] > 0.5*filtered_results[i][3]:  # ??? add 0.5 as variable
                if stream_start_end[i] is None: # NEW None guard
                    logger.warning(f"Detection {i}: skipped window, keeping original")
                elif _peak[0][3] > self.REALIGN_SCORE_TOLERANCE * filtered_results[i][3]:
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
                    scores_final.append(_score)
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
                    scores_final.append(_score)
        masks_subset = np.array(masks_subset)
        stft_final_subset = np.array(stft_final_subset)

        if stft_final_subset.shape[0] == 0:
            return (np.zeros((0, 64, 256, 6), dtype=np.float32),
                    np.zeros((0, 64, 256, 3), dtype=np.float32),
                    [], [], [])  # NEW []

        return stft_final_subset, masks_subset, utc_start_subset, stream_start_end_final, scores_final  # NEW


    def _build_streams(self, stft_final_subset, masks_subset, # NEW
                       utc_start_subset, stream_start_end_final,
                       data, denoised_hyb=None):
        """
        Build denoised ObsPy Stream from selected windows.
        Uses denoised_hyb if provided (EQShyb path), otherwise ISTFT of masked STFT (EQS path).
        """
        logger.debug("")
        num = stft_final_subset.shape[0]
        st_denoised_collection = obspy.core.Stream()
        use_eqshyb = (denoised_hyb is not None) and (num > 0)

        for i in range(num):
            for j, comp in enumerate(self.components):
                stats = data.select(component=comp)[0].stats.copy()
                stats.starttime = utc_start_subset[i]

                if use_eqshyb:
                    trace_data = denoised_hyb[i, :, j]
                else:
                    _stft = (stft_final_subset[i, :, :, j * 2] +
                             1j * stft_final_subset[i, :, :, j * 2 + 1])
                    _, trace_data = istft(_stft * masks_subset[i, :, :, j],
                                          **self.stft_parameters)
                trace_data = np.ascontiguousarray(trace_data, dtype=np.float32)
                stats.npts = len(trace_data)
                st_denoised_collection += obspy.core.Trace(trace_data, header=stats)

        segments = [st_denoised_collection[3 * i: 3 * (i + 1)] for i in range(num)]
        # start_times = [seg[0].stats.starttime for seg in segments]
        start_times = [seg.select(component=self.components[0])[0].stats.starttime for seg in segments]

        # sorted_indices = sorted(range(len(segments)), key=lambda i: start_times[i])
        sorted_indices = sorted(range(len(segments)), key=lambda i: stream_start_end_final[i][0])
        trimmed_streams = obspy.core.Stream()
        stream_start_end_sorted = []
        for i in sorted_indices:
            trimmed_streams += segments[i]
            stream_start_end_sorted.append(stream_start_end_final[i])

        return trimmed_streams, stream_start_end_sorted

    def _apply_eqshyb(self, stft_final_subset, masks_subset,
                      utc_start_subset, data_stack, starttime):  # NEW
        """
        Apply EQShyb hybrid denoiser to accepted EQS detections.

        Takes the noisy waveform directly from data_stack (avoids re-slicing),
        standardizes it, applies EQS mask to get stage-1 denoised waveform,
        then runs the hybrid model on both inputs alongside the EQS mask.

        Parameters
        ----------
        stft_final_subset : np.ndarray, shape (N, 64, 256, 6)
            Raw STFT for each accepted detection window.
        masks_subset : np.ndarray, shape (N, 64, 256, 3)
            EQS predicted masks for each accepted detection window.
        utc_start_subset : list of UTCDateTime
            Start time of each accepted detection window.
        data_stack : np.ndarray, shape (total_samples, 3)
            Full restituted waveform, columns ordered Z, N, E.
        starttime : UTCDateTime
            Start time of data_stack[0], used for sample index calculation.

        Returns
        -------
        denoised_hyb : np.ndarray, shape (N, 6120, 3)
            EQShyb denoised waveforms, amplitude in physical units.
        """

        logger.debug("")
        num = stft_final_subset.shape[0]

        # ── noisy waveform: slice directly from data_stack ───────────────────
        raw_td = np.zeros((num, self.len_sample, 3), dtype=np.float32)
        for i, utc in enumerate(utc_start_subset):
            startidx = int((utc - starttime) * self.stft_parameters["fs"])
            endidx = startidx + self.len_sample
            segment = data_stack[startidx:endidx, :]
            if len(segment) < self.len_sample:
                logger.warning(
                    f"EQShyb segment {i} at {utc}: only {len(segment)} samples "
                    f"available, zero-filling remainder"
                )
                raw_td[i, :len(segment), :] = segment
            else:
                raw_td[i] = segment

        # ── standardize — matches training convention exactly ─────────────────
        raw_td_t = raw_td.transpose(0, 2, 1)  # (N, 3, 6120)
        eqs2_mean = np.mean(raw_td_t, axis=2, keepdims=True)  # per-channel mean
        eqs2_std = np.std(raw_td_t, axis=(1, 2), keepdims=True)  # global std per item
        eqs2_std = np.maximum(eqs2_std, 1e-8)  # matches tf.maximum(std, eps)
        norm_in1 = ((raw_td_t - eqs2_mean) / eqs2_std).transpose(0, 2, 1)  # (N, 6120, 3)

        # ── EQS stage-1 denoised: broadcast mask multiply then ISTFT ─────────
        stft_c = np.stack([
            stft_final_subset[:, :, :, c * 2] + 1j * stft_final_subset[:, :, :, c * 2 + 1]
            for c in range(3)
        ], axis=-1)  # (N, 64, 256, 3)
        stft_masked = stft_c * masks_subset  # (N, 64, 256, 3)

        eqs1_td = np.zeros((num, self.len_sample, 3), dtype=np.float32)
        for c in range(3):
            for i in range(num):
                _, sig = istft(stft_masked[i, :, :, c], **self.stft_parameters)
                eqs1_td[i, :, c] = sig

        norm_in2 = (eqs1_td.transpose(0, 2, 1) / eqs2_std).transpose(0, 2, 1)  # (N, 6120, 3)

        # ── concatenate and predict ───────────────────────────────────────────
        eqs2_td_in = np.concatenate([norm_in1, norm_in2], axis=2)  # (N, 6120, 6)
        eqs2_out = self.eqs2_model.predict(
            {"time_domain": eqs2_td_in, "spectral_domain": masks_subset},
            verbose=0
        )

        # ── rescale: model output is (clean - mean_clean) / std ──────────────
        denoised_hyb = eqs2_out['output_wave'] * eqs2_std.transpose(0, 2, 1)  # (N, 6120, 3)

        logger.debug(f"EQShyb applied to {num} detections")
        return denoised_hyb

    def _output(self, starttime, trimmed_streams, gap_intervals):
        """
        Writes results to the disk, print output, might also plot etc.
        """

        logger.debug("")

        output_stream = obspy.core.Stream()
        # for i in range(3):
        #     for j in range(int(len(trimmed_streams) // 3)):
        #         output_stream += trimmed_streams[3*j + i]
        if not len(trimmed_streams):
            logger.debug("No events found")
            return

        for comp in self.components:
            for j in range(int(len(trimmed_streams) // 3)):
                triple = trimmed_streams[3 * j: 3 * (j + 1)]
                output_stream += triple.select(component=comp)[0]

        output_stream._cleanup()

        # mask gap regions — zero out samples that fall within recorded gap intervals
        if gap_intervals:
            for tr in output_stream:
                for gap_start, gap_end in gap_intervals:
                    # convert UTCDateTime to sample indices relative to this trace
                    i_start = (gap_start - tr.stats.starttime) * tr.stats.sampling_rate
                    i_end = (gap_end - tr.stats.starttime) * tr.stats.sampling_rate
                    i_start = max(0, int(i_start))
                    i_end = min(len(tr.data), int(np.ceil(i_end)))
                    if i_end > i_start:  # if gaps not overlapping - i_end is smaller (negative) than i_start (min 0)
                        tr.data[i_start:i_end] = 0#np.nan
                        logger.debug(f"Gap masked in {tr.id}: "
                                     f"{gap_start} — {gap_end} "
                                     f"(samples {i_start}:{i_end})")

        dir_tmp = str(Path(self.model_name).parent /
                      ("DOY" + str(starttime.julday).zfill(3))) + "/"
        check_dir(dir_tmp)

        # for tr in output_stream:
        #     tr.data = np.nan_to_num(tr.data, nan=0.0)

        output_stream.write(dir_tmp +
                            trimmed_streams[0].id[:-1] +
                            "_denoised.mseed",
                            format="MSEED", encoding="FLOAT32")

    def _trim_streams(self, trimmed_streams, startstop):
        logger.debug("")
        new_trimmed_stream = obspy.core.Stream()

        if len(startstop) == 0:
            return new_trimmed_stream

        if len(startstop) == 1:
            for comp in self.components:
                new_trimmed_stream += trimmed_streams.select(component=comp)[0]
            return new_trimmed_stream

        buf = self.signal_buffer_s
        one_sample = self.one_sample_s

        for i in range(0, len(startstop) - 1):
            triple_i = trimmed_streams[3 * i:     3 * (i + 1)]
            triple_next = trimmed_streams[3 * (i + 1): 3 * (i + 2)]

            tr_i = triple_i.select(component=self.components[0])[0]
            tr_next = triple_next.select(component=self.components[0])[0]

            if tr_i.stats.endtime < tr_next.stats.starttime:
                for comp in self.components:
                    new_trimmed_stream += triple_i.select(component=comp)[0]
                logger.debug("No overlap")
                continue

            if startstop[i][1] < tr_next.stats.starttime:
                logger.debug("No signal overlap with next stream")
                for comp in self.components:
                    new_trimmed_stream += triple_i.select(component=comp)[0].slice(
                        endtime=startstop[i][1] - one_sample)

            elif startstop[i][1] + buf < startstop[i + 1][0]:
                logger.debug("No signal overlap")
                for comp in self.components:
                    ################
                    slice_end = startstop[i][1] - one_sample
                    # if triple_i.select(component=comp)[0].slice(endtime=slice_end).stats.npts == 0:
                        # logger.warning(f"_trim_streams: zero-length slice at detection {i} "
                        #                f"comp={comp}, "
                        #                f"stream_start={triple_i.select(component=comp)[0].stats.starttime}, "
                        #                f"stream_end={triple_i.select(component=comp)[0].stats.endtime}, "
                        #                f"slice_end={slice_end}, "
                        #                f"signal_start={startstop[i][0]}, signal_end={startstop[i][1]}, "
                        #                f"next_signal_start={startstop[i+1][0]}, next_signal_end={startstop[i+1][1]}, ")

                    ################
                    new_trimmed_stream += triple_i.select(component=comp)[0].slice(
                        endtime=startstop[i][1] - one_sample)
                    # modify in place on the next triple
                    tr_next = triple_next.select(component=comp)[0]
                    tr_next = tr_next.slice(starttime=startstop[i + 1][0] - buf)
                    trimmed_streams[3 * (i + 1) + self.components.index(comp)] = tr_next
            else:
                logger.debug("Signal overlap")
                for comp in self.components:
                    ################
                    slice_end = startstop[i + 1][0] - buf - one_sample
                    # if triple_i.select(component=comp)[0].slice(endtime=slice_end).stats.npts == 0:
                    #     logger.warning(f"_trim_streams: zero-length slice at detection {i} "
                    #                    f"comp={comp}, "
                    #                    f"stream_start={triple_i.select(component=comp)[0].stats.starttime}, "
                    #                    f"stream_end={triple_i.select(component=comp)[0].stats.endtime}, "
                    #                    f"slice_end={slice_end}, "
                    #                    f"signal_start={startstop[i][0]}, signal_end={startstop[i][1]}, "
                    #                    f"next_signal_start={startstop[i+1][0]}, next_signal_end={startstop[i+1][1]}, ")


                    ######################
                    new_trimmed_stream += triple_i.select(component=comp)[0].slice(
                        endtime=startstop[i + 1][0] - buf - one_sample)
                    tr_next = triple_next.select(component=comp)[0]
                    tr_next = tr_next.slice(starttime=startstop[i + 1][0] - buf)
                    trimmed_streams[3 * (i + 1) + self.components.index(comp)] = tr_next

        last = len(startstop) - 1
        triple_last = trimmed_streams[3 * last: 3 * (last + 1)]
        for comp in self.components:
            new_trimmed_stream += triple_last.select(component=comp)[0]

        # for tr in new_trimmed_stream:
        #     if tr.stats.starttime < data_start:
        #         tr.trim(starttime=data_start, pad=True, fill_value=0)
        #     if tr.stats.endtime > data_end:
        #         tr.trim(endtime=data_end, pad=True, fill_value=0)

        return new_trimmed_stream


    def _filter_close_detections_streams(self, trimmed_streams, stream_start_end_final, scores_final):
        """
        Remove detections whose signal start is closer than signal_buffer_s
        to the previous detection, keeping the higher-scoring one.
        Operates on already-sorted trimmed_streams and stream_start_end_final
        from _build_streams().
        """
        if len(stream_start_end_final) <= 1:
            return trimmed_streams, stream_start_end_final

        keep = [0]
        for i in range(1, len(stream_start_end_final)):
            sep = stream_start_end_final[i][0] - stream_start_end_final[keep[-1]][0]
            if sep >= self.signal_buffer_s:
                keep.append(i)
            else:
                if scores_final[i] > scores_final[keep[-1]]:
                    logger.info(f"_filter_close_detections: dropping detection {keep[-1]} "
                                f"(score {scores_final[keep[-1]]:.2f} < {scores_final[i]:.2f}, "
                                f"separation {sep:.2f}s), keeping detection {i}")
                    keep[-1] = i
                else:
                    logger.info(f"_filter_close_detections: dropping detection {i} "
                                f"(score {scores_final[i]:.2f} <= {scores_final[keep[-1]]:.2f}, "
                                f"separation {sep:.2f}s), keeping detection {keep[-1]}")

        filtered_streams = obspy.core.Stream()
        for i in keep:
            filtered_streams += trimmed_streams[3 * i: 3 * (i + 1)]

        filtered_startstop = [stream_start_end_final[i] for i in keep]
        return filtered_streams, filtered_startstop

    # =========================================================================
    # NEW: Phase picking methods — integrated from DenoisingFunctions_public.py
    # All methods prefixed with _ (private). Only _pick() and _save_picks()
    # are called from run_data(); all others are internal helpers.
    # =========================================================================

    def _weighted_std(self, values, weights):
        """
        Compute the weighted standard deviation of an array.
        Used by _tta_uncertainty() to quantify pick timing spread across TTA reps.

        values  : 1D array-like
        weights : 1D array-like, corresponding weights
        Returns : float, weighted standard deviation
        """
        average = np.average(values, weights=weights + 1e-30)
        variance = np.average((values - average) ** 2, weights=weights + 1e-30)
        return np.sqrt(variance)


    def _weighted_median(self, argmax_values, max_values):
        """
        Compute the weighted median of an array.
        Used by _cluster_picks() to find the representative pick time per cluster.

        argmax_values : 1D array-like, pick times
        max_values    : 1D array-like, confidence weights
        Returns       : float, weighted median pick time
        """
        argmax_values = np.asarray(argmax_values)
        max_values = np.asarray(max_values)
        sorted_indices = np.argsort(argmax_values)
        sorted_vals = argmax_values[sorted_indices]
        sorted_weights = max_values[sorted_indices]
        cum_weights = np.cumsum(sorted_weights)
        total_weight = np.sum(sorted_weights)
        median_idx = np.searchsorted(cum_weights, total_weight / 2.0)
        return sorted_vals[median_idx]


    def _cluster_picks(self, pick_array, peak_vals, delta=1):
        """
        Cluster picks close in time and compute weighted median per cluster.
        Used by _process_peak_times() to consolidate TTA picks into one per event.

        pick_array : 1D numpy array, pick times in seconds relative to window start
        peak_vals  : 1D numpy array, confidence values per pick
        delta      : float, max separation (s) to consider picks the same cluster
        Returns    : (medians, clusters)
            medians  : list of weighted median times, one per cluster
            clusters : list of lists of raw pick times per cluster
        """
        sorted_indices = np.argsort(pick_array)
        sorted_picks = pick_array[sorted_indices]
        diffs = np.diff(sorted_picks)
        breaks = np.where(diffs > delta)[0] + 1
        cluster_indices = np.split(sorted_indices, breaks)
        medians = [self._weighted_median(pick_array[idx], peak_vals[idx])
                   for idx in cluster_indices]
        clusters = [[pick_array[i] for i in idx] for idx in cluster_indices]
        return medians, clusters


    def _tta_uncertainty(self, confidence_timeseries, pick_utc,
                         pick_tolerance=1.0, confidence=0.5):
        """
        Estimate pick uncertainty from TTA confidence traces.
        Used by _process_peak_times() per median pick.

        confidence_timeseries : list of obspy.Trace, model confidence over time,
                                one trace per TTA repetition
        pick_utc              : UTCDateTime, pick time to evaluate around
        pick_tolerance        : float, window half-width (s) around pick
        confidence            : float, threshold for counting a trace as "above"
        Returns               : (uncertainty, fraction_above_confidence)
            uncertainty            : float, 1 + weighted std of argmax positions
            fraction_above_confidence : float, fraction of TTA traces above threshold
        """
        t_start = pick_utc - pick_tolerance
        t_end = pick_utc + pick_tolerance
        sliced_traces = [
            trace.slice(t_start, t_end)
            for trace in confidence_timeseries
            if trace.stats.starttime <= t_end and trace.stats.endtime >= t_start
        ]
        _argmax = [np.argmax(trace.data) for trace in sliced_traces]
        _max = np.array([np.max(trace.data) for trace in sliced_traces])
        reached_threshold = np.mean(_max > confidence)
        return 1 + self._weighted_std(_argmax, _max), reached_threshold


    def _process_peak_times(self, peak_times, peak_vals, annotations,
                            channel_pattern, start_time,
                            pick_tolerance=1, confidence=0.5):
        """
        Cluster TTA peak times, convert to UTC, compute uncertainty per pick.
        Used by _process_snippet() separately for P and S phases.

        peak_times      : 1D array, peak times relative to window start (s)
        peak_vals       : 1D array, confidence values per peak
        annotations     : obspy.Stream, model confidence traces (all TTA reps)
        channel_pattern : str, e.g. "*_P" or "*_S" to select phase channel
        start_time      : UTCDateTime, window start for converting to absolute UTC
        pick_tolerance  : float, clustering tolerance (s)
        confidence      : float, threshold for fraction_above_confidence
        Returns         : (picks_median_utc, results)
            picks_median_utc : list of UTCDateTime, one per cluster
            results          : list of (uncertainty, fraction_above_confidence) tuples
        """
        if len(peak_times) == 0:
            return [], []
        picks_median, _ = self._cluster_picks(peak_times, peak_vals,
                                              delta=pick_tolerance)
        picks_median_utc = [start_time + t for t in picks_median]
        selected_traces = annotations.select(channel=channel_pattern)
        results = [
            self._tta_uncertainty(selected_traces, pick,
                                  pick_tolerance=1, confidence=confidence)
            for pick in picks_median_utc
        ]
        return picks_median_utc, results


    def _stream_tta(self, _event_stream, _noise_stream, id=0,
                    white_noise_factor=0.01):
        """
        Apply one TTA augmentation by injecting amplitude-scaled white noise.
        The noise amplitude envelope is derived from _noise_stream (designaled noise)
        via std — white Gaussian noise is then scaled by that envelope.
        Each id produces a different but fully reproducible noise realisation.

        _event_stream     : obspy.Stream, denoised event waveforms (Z/N/E)
        _noise_stream     : obspy.Stream, designaled noise for amplitude scaling
        id                : int, TTA index — seeds the RNG for reproducibility
        white_noise_factor: float, global scaling of injected noise amplitude
        Returns           : obspy.Stream, augmented event stream
        """
        _event_noiseinjected = _event_stream.copy()
        # seed by id — same id always produces same noise sequence
        rng = np.random.default_rng(seed=id)
        for comp in self.components:
            tr_denoised = _event_noiseinjected.select(component=comp)[0]
            tr_noise = _noise_stream.select(component=comp)[0]
            tr_denoised.data += (white_noise_factor
                                 * np.std(tr_noise.data)
                                 * rng.standard_normal(len(tr_denoised.data)))
            tr_denoised.stats.location = str(id).zfill(2)
        return _event_noiseinjected


    def _get_designaled_noise(self, _denoised_snippets, _original):
        """
        Compute per-snippet designaled noise = original - denoised.
        Both inputs must be a single Z/N/E triple (exactly 1 trace per component).
        Length alignment is the caller's responsibility (_pick() enforces this);
        a last-resort truncation guard is included here.

        _denoised_snippets : obspy.Stream, single denoised event snippet (3 traces)
        _original          : obspy.Stream, original stream sliced to same window (3 traces)
        Returns            : obspy.Stream, noise traces (3 components),
                             or empty Stream on component/trace-count mismatch
        """
        missing_orig = [c for c in self.components if len(_original.select(component=c)) == 0]
        missing_denoised = [c for c in self.components if len(_denoised_snippets.select(component=c)) == 0]
        missing = missing_orig + missing_denoised


        if missing:
            logger.warning(f"_get_designaled_noise: missing components {missing} "
                           f"(orig channels: {[tr.stats.channel for tr in _original]}, "
                           f"self.components: {self.components}) — returning empty stream")
            return obspy.core.Stream()

        _noise = obspy.core.Stream()
        for comp in self.components:
            orig_comp = _original.select(component=comp)
            denoised_comp = _denoised_snippets.select(component=comp)

            # guard: exactly 1 trace per component expected (snippet, not continuous)
            if len(orig_comp) != 1 or len(denoised_comp) != 1:
                logger.warning(f"_get_designaled_noise: expected 1 trace per component, "
                               f"got {len(orig_comp)} original and "
                               f"{len(denoised_comp)} denoised for component {comp} "
                               f"— merge inputs before calling")
                return obspy.core.Stream()

            tr_orig = orig_comp[0]
            tr_denoised = denoised_comp[0]
            n_orig = len(tr_orig.data)
            n_denoised = len(tr_denoised.data)

            if n_orig != n_denoised:
                # last-resort truncation — caller should have aligned lengths
                n = min(n_orig, n_denoised)
                logger.warning(f"_get_designaled_noise: length mismatch on "
                               f"component {comp} ({n_orig} vs {n_denoised}) "
                               f"— truncating to {n} samples")
                orig_data = tr_orig.data[:n]
                denoised_data = tr_denoised.data[:n]
            else:
                orig_data = tr_orig.data
                denoised_data = tr_denoised.data

            noise_tr = tr_orig.copy()
            noise_tr.data = (orig_data - denoised_data).astype(np.float32)
            _noise += noise_tr


        return _noise


    def _process_snippet(self, event_streams, st_designaled, repeat,
                         pick_tolerance, p_confidence, s_confidence):
        """
        Run TTA phase picking on a single event Z/N/E triple.
        Builds repeat augmented copies of the snippet with scaled white noise,
        annotates them all in one batch, clusters picks, and computes uncertainty.
        Optionally runs polarity prediction on each accepted P pick using the
        same TTA collection (no re-augmentation needed).

        event_streams  : tuple of (tr_Z, tr_N, tr_E) obspy.Trace objects
        st_designaled  : obspy.Stream, per-snippet noise for TTA amplitude scaling
        repeat         : int, number of TTA augmentations
        pick_tolerance : float, clustering tolerance in seconds
        p_confidence   : float, min confidence for P picks
        s_confidence   : float, min confidence for S picks
        Returns        : dict with keys 'p_picks' and 's_picks', each a list of
                         (median_utc, uncertainty, fraction_above_conf, event_id)
                         **P picks additionally carry a polarity dict as fifth**
                         **element when self.polarity_model is not None.**
        """
        _st_z, _st_1, _st_2 = event_streams

        add = 5 if _st_z.stats.npts >= 6120 else 5 + (6120 - _st_z.stats.npts) / 200
        for st in (_st_z, _st_1, _st_2):
            st.trim(st.stats.starttime - add, st.stats.endtime + add,
                    pad=True, fill_value=0)

        _noise = obspy.core.Stream([
            st_designaled.select(component=c)[0].copy()
            for c in self.components
        ])

        _start, _end = _st_z.stats.starttime, _st_z.stats.endtime
        for tr in _noise:
            tr.trim(_start, _end, pad=True, fill_value=0)

        event_tta_collection = Stream()
        for i in range(repeat):
            event_tta_collection += self._stream_tta(
                Stream([_st_z, _st_1, _st_2]), _noise,
                id=i, white_noise_factor=0.01)

        annotations = self.picker.annotate(event_tta_collection, batch_size=repeat)
        annotations.sort(keys=['location'])
        annotations.trim(_start, _end, pad=True, fill_value=0)

        picks_current_tta = self.picker.classify_aggregate(annotations, argdict={}).picks
        p_picks_tta = picks_current_tta.select(min_confidence=p_confidence, phase="P")
        s_picks_tta = picks_current_tta.select(min_confidence=s_confidence, phase="S")

        p_peak_times = np.array([p.peak_time - _start for p in p_picks_tta])
        s_peak_times = np.array([s.peak_time - _start for s in s_picks_tta])
        p_peak_vals = np.array([p.peak_value for p in p_picks_tta])
        s_peak_vals = np.array([s.peak_value for s in s_picks_tta])

        p_picks_median, p_results = self._process_peak_times(
            peak_times=p_peak_times, peak_vals=p_peak_vals,
            annotations=annotations, channel_pattern="*_P",
            start_time=_start, pick_tolerance=pick_tolerance,
            confidence=p_confidence
        )
        s_picks_median, s_results = self._process_peak_times(
            peak_times=s_peak_times, peak_vals=s_peak_vals,
            annotations=annotations, channel_pattern="*_S",
            start_time=_start, pick_tolerance=pick_tolerance,
            confidence=s_confidence
        )

        event_id = _st_z.id

        # **polarity — reuses event_tta_collection, no re-augmentation needed**
        p_picks = []
        for p_median, p_result in zip(p_picks_median, p_results):
            entry = (p_median, p_result[0], p_result[1], event_id)
            # **if polarity model configured on self, append polarity dict**
            if self.polarity_model is not None:
                polarity = _predict_polarity_tta(
                    z_tta_collection=event_tta_collection,
                    z_starttime=_st_z.stats.starttime,
                    z_sampling_rate=_st_z.stats.sampling_rate,
                    p_pick=p_median,
                    polarity_model=self.polarity_model,
                    threshold=self.polarity_threshold,
                )
                entry = entry + (polarity,)
            p_picks.append(entry)

        s_picks = [(m, r[0], r[1], event_id)
                   for m, r in zip(s_picks_median, s_results)]

        return {'p_picks': p_picks, 's_picks': s_picks}

    def _process_picks(self, st_denoised, st_designaled,
                       repeat=20, pick_tolerance=1,
                       p_confidence=0.5, s_confidence=0.5,
                       min_share_models=0.25):
        """
        Parallel TTA picking loop over all detected event snippets.
        Processes each Z/N/E triple independently via ThreadPoolExecutor.
        Results collected via as_completed(); pick order in output list is
        non-deterministic but irrelevant since picks carry UTC time and event_id.

        st_denoised      : obspy.Stream, denoised snippets (Z/N/E triples)
        st_designaled    : obspy.Stream, per-snippet noise for TTA scaling,
                           aligned to st_denoised (same number of triples)
        repeat           : int, number of TTA augmentations per snippet
        pick_tolerance   : float, clustering tolerance (s)
        p_confidence     : float, min confidence for P picks
        s_confidence     : float, min confidence for S picks
        min_share_models : float, min fraction of TTA reps above threshold
        Returns          : dict with keys 'p_picks' and 's_picks'
        """
        all_results = {'p_picks': [], 's_picks': []}

        # sort both streams by starttime so Z/N/E triples zip correctly
        st_denoised.sort(keys=['starttime'])
        st_designaled.sort(keys=['starttime'])

        # explicit component selection + sort ensures deterministic pairing
        st_z_list = sorted(st_denoised.select(component=self.components[0]),
                           key=lambda tr: tr.stats.starttime)
        st_1_list = sorted(st_denoised.select(component=self.components[1]),
                           key=lambda tr: tr.stats.starttime)
        st_2_list = sorted(st_denoised.select(component=self.components[2]),
                           key=lambda tr: tr.stats.starttime)

        event_streams_list = list(zip(st_z_list, st_1_list, st_2_list))
        designaled_streams_list = [st_designaled[3 * i: 3 * (i + 1)]
                                   for i in range(len(event_streams_list))]

        assert len(event_streams_list) == len(designaled_streams_list), (
            f"Mismatch: {len(event_streams_list)} event streams vs "
            f"{len(designaled_streams_list)} designaled streams"
        )

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._process_snippet,
                    event_streams,
                    designaled_stream,
                    repeat, pick_tolerance,
                    p_confidence, s_confidence
                )
                # for event_streams in event_streams_list
                for event_streams, designaled_stream in zip(event_streams_list, designaled_streams_list)
            ]
            # for future in futures:
            for future in as_completed(futures):  # should be fine here
                result = future.result()
                all_results['p_picks'].extend(result['p_picks'])
                all_results['s_picks'].extend(result['s_picks'])

        # filter by minimum share of TTA models above confidence threshold
        all_results['p_picks'] = [e for e in all_results['p_picks']
                                   if e[2] > min_share_models]
        all_results['s_picks'] = [e for e in all_results['s_picks']
                                   if e[2] > min_share_models]
        return all_results


    def _pick(self, trimmed_streams, data_original):
        """
        Run phase picking on denoised event snippets using TTA.
        Computes per-snippet designaled noise (original - denoised) to provide
        the amplitude info for TTA white noise scaling. Memory cost scales
        with number of detections x snippet length.

        trimmed_streams : obspy.Stream
            Denoised event snippets from _trim_streams(), Z/N/E per detection.
            Picker runs on these only — never on continuous data.
        data_original   : obspy.Stream
            Original restituted full-day stream from _get_data().
            Only in memory (not saved to disk). Used only to compute
            per-snippet noise via (original - denoised).

        Returns dict with keys 'p_picks' and 's_picks', each a list of tuples:
            (median pick time, uncertainty, fraction above confidence, event_id)
        """
        # logger.debug(f"_pick: data_original has {len(data_original)} traces: "
        #              f"{[tr.stats.channel for tr in data_original]}")

        if not len(trimmed_streams):
            logger.info("No denoised snippets — skipping picking")
            return {'p_picks': [], 's_picks': []}

        data_start = data_original[0].stats.starttime
        data_end = data_original[0].stats.endtime

        num_detections = len(trimmed_streams) // 3
        st_designaled_snippets = obspy.core.Stream()

        for i in range(num_detections):
            snippet = trimmed_streams[3 * i: 3 * (i + 1)]
            # logger.debug(f"Detection {i}: snippet components = "
            #              f"{[tr.stats.channel for tr in snippet]}, "
            #              f"npts = {[tr.stats.npts for tr in snippet]}")

            tr_ref = snippet.select(component=self.components[0])[0]
            if tr_ref.stats.npts == 0:  # TODO check why this happens
                logger.warning(f"Detection {i}: zero-length snippet — skipping")
                continue

            # start = tr_ref.stats.starttime
            # end = tr_ref.stats.endtime
            start = max(tr_ref.stats.starttime, data_start)  # clamp
            end = min(tr_ref.stats.endtime, data_end)  # clam
            npts = tr_ref.stats.npts

            # slice original to same window — tr.slice() avoids copying full day
            original_snippet = obspy.core.Stream()

            for tr in data_original:
                tr_sliced = tr.slice(start, end)
                # logger.debug(f"_pick slice: tr={tr.id} "
                #              f"data={tr.stats.starttime}—{tr.stats.endtime} "
                #              f"slice={start}—{end} "
                #              f"result_npts={tr_sliced.stats.npts} "
                #              f"expected_npts={npts}")

                if tr_sliced.stats.npts != npts:
                    tr_sliced = tr_sliced.trim(
                        start,
                        start + (npts - 1) * tr_sliced.stats.delta,
                        nearest_sample=True, pad=True, fill_value=0
                    )
                original_snippet += tr_sliced

            # logger.warning(f"snippet:{snippet[0].stats.starttime} — {snippet[0].stats.endtime}")
            # logger.warning(f"original_snippet: {original_snippet[0].stats.starttime} — {original_snippet[0].stats.endtime}")

            noise_snippet = self._get_designaled_noise(snippet, original_snippet)
            if len(noise_snippet) == 0:
                logger.warning(f"Detection {i}: _get_designaled_noise failed — "
                               f"using zero noise for this snippet")
                # zero-fill to preserve index alignment with trimmed_streams
                for tr in snippet:
                    zero_tr = tr.copy()
                    zero_tr.data = np.zeros_like(tr.data)
                    st_designaled_snippets += zero_tr
            else:
                st_designaled_snippets += noise_snippet

        picks = self._process_picks(
            st_denoised=trimmed_streams,
            st_designaled=st_designaled_snippets,
            **self.picking_kwargs
        )
        logger.info(f"Picks: {len(picks['p_picks'])} P, "
                    f"{len(picks['s_picks'])} S")
        return picks


    def _save_picks(self, picks, starttime):
        """
        Save picks to JSON alongside MiniSEED output.
        Format: {"p_picks": [{"time": ..., "uncertainty": ...,
                               "share": ..., "id": ...,
                               "polarity": ..., "polarity_probabilities": ...}, ...],
                 "s_picks": [...]}

        picks     : dict from _pick(), keys 'p_picks' and 's_picks'
                    P pick tuples are (time, uncertainty, share, id) or
                    (time, uncertainty, share, id, polarity_dict) when
                    polarity model is configured.
        starttime : UTCDateTime, used for output directory naming (same as _output)
        """
        dir_tmp = str(Path(self.model_name).parent /
                      ("DOY" + str(starttime.julday).zfill(3))) + "/"
        check_dir(dir_tmp)

        serialisable = {}
        for phase, pick_list in picks.items():
            entries = []
            for pick in pick_list:
                t, u, s, eid = pick[:4]

                # scale raw TTA std to physically meaningful seconds;
                # coefficients derived from empirical calibration against reference picks
                coeff = self.uncertainty_scaling[phase]
                uncertainty_scaled = (coeff['scale_sample'] * u + coeff['offset_sample']) / self.stft_parameters["fs"]

                entry = {"time": str(t), "uncertainty": uncertainty_scaled, "share": s, "id": eid}
                # polarity dict present as fifth element for P picks
                if len(pick) == 5:
                    polarity = pick[4]
                    entry["polarity"] = polarity["label"]
                    entry["polarity_probabilities"] = polarity["probabilities"].tolist()
                    # entry["polarity_all_predictions"] = polarity["all_predictions"].tolist()
                entries.append(entry)
            serialisable[phase] = entries

        out_path = dir_tmp + f"picks_DOY{str(starttime.julday).zfill(3)}.json"
        with open(out_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"Picks written to {out_path}")

    # =========================================================================
    # END picking methods
    # =========================================================================


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

    def _round_to_window(self, starttime, endtime):
        """
        Adjust endtime so that (endtime - starttime) is a multiple of 61.2s.
        Rounds up to avoid losing the last partial window.
        window_s = len_sample / fs = 6120 / 100 = 61.2s
        """
        window_s = self.len_sample / self.stft_parameters["fs"]  # 61.2s
        duration = endtime - starttime
        n_windows = int(np.ceil(duration / window_s))
        return starttime + n_windows * window_s

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

        # starttime and endtime should be multiples of 61.2s

        # IN:  starttime (UTCDateTime, start of processing window),
        #      endtime (UTCDateTime, requested end of processing window)
        # OUT: endtime (UTCDateTime, adjusted so that endtime - starttime
        #               is an exact multiple of 61.2s, rounded up)
        endtime = self._round_to_window(starttime, endtime)


        # IN:  network, station, location, channel, starttime, endtime, data (optional raw Stream)
        # OUT: data (Stream, 3 components, restituted, 100 Hz, buffer trimmed),
        #      data_stack (np.ndarray, shape (N, 3), columns Z/N/E,
        #                  restituted velocity waveforms in physical units),
        #      gap_intervals (list of (UTCDateTime, UTCDateTime), gap start/end pairs
        #                     recorded before merge — empty list if no gaps)
        result = self._get_data(network, station, location, channel,
                                          starttime, endtime, data)

        if result is None:
            return None
        data, data_stack, gap_intervals = result

        # compute stfts
        # IN:  data_stack (N, 3), starttime (UTCDateTime), endtime (UTCDateTime)
        # OUT: selected_starttimes (list of UTCDateTime, one per valid window),
        #      stft_collection (np.ndarray, shape (W, 64, 256, 6), raw STFT all windows),
        #      stft_norm_collection (np.ndarray, shape (W, 64, 256, 6), normalised STFT)
        selected_starttimes, stft_collection, stft_norm_collection =  \
            self._compute_stfts(data_stack, data[0].stats.starttime, endtime)

        # make prediction
        # IN:  stft_norm_collection (W, 64, 256, 6)
        # OUT: filtered_results (np.ndarray, (D, 5), peak/start/end/score/maxval per detection),
        #      origin (list of int, 0=even 1=odd stream for each detection),
        #      y_predict (np.ndarray, (W, 64, 256, 3), EQS mask predictions all windows)
        filtered_results, origin, y_predict = \
            self._detect_event_signals(stft_norm_collection)

        # make selection
        # IN:  filtered_results (D, 5), origin (D,), y_predict (W, 64, 256, 3),
        #      stft_collection (W, 64, 256, 6), selected_starttimes (list, W)
        # OUT: selected_masks (np.ndarray, (D, 64, 256, 3), EQS mask per detection),
        #      selected_stft (np.ndarray, (D, 64, 256, 6), raw STFT per detection),
        #      selected_utc (list of UTCDateTime, window start per detection),
        #      detection_start (list of UTCDateTime, estimated signal start per detection),
        #      detection_score (list of float, score per detection)
        selected_masks, selected_stft, selected_utc, detection_start, \
            detection_score = \
            self._select_data_and_mask(filtered_results, origin, y_predict,
                                       stft_collection, selected_starttimes)

        if len(detection_start) == 0:
            logger.info("No detections found — skipping refinement and output")
            return None

        # recompute mask
        # IN:  detection_start (list of UTCDateTime, D),
        #      data[0].stats.starttime (UTCDateTime, reference for sample indexing),
        #      data_stack (N, 3)
        # OUT: stft_collection_subset (np.ndarray, (D, 64, 256, 6), re-aligned raw STFT),
        #      stft_norm_collection_subset (np.ndarray, (D, 64, 256, 6), re-aligned norm STFT),
        #      stream_start_end (list of (UTCDateTime, UTCDateTime) or None, D,
        #                        None where window was skipped due to insufficient data)
        stft_collection_subset, stft_norm_collection_subset, \
            stream_start_end = \
            self._recompute_mask(detection_start,
                                 data[0].stats.starttime,
                                 data_stack)

        # Make new prediction
        # IN:  stft_norm_collection_subset (D, 64, 256, 6)
        # OUT: y_predict_event (np.ndarray, (D, 64, 256, 3), EQS mask re-aligned windows)
        y_predict_event = self.model.predict(stft_norm_collection_subset,
                                             verbose=0)

        # window selection — returns arrays needed for EQShyb
        # IN:  y_predict_event (D, 64, 256, 3), filtered_results (D, 5),
        #      detection_start (list, D), selected_stft (D, 64, 256, 6),
        #      selected_masks (D, 64, 256, 3), selected_utc (list, D),
        #      stft_collection_subset (D, 64, 256, 6),
        #      stream_start_end (list of (UTCDateTime, UTCDateTime) or None, D)
        # OUT: stft_final_subset (np.ndarray, (A, 64, 256, 6), raw STFT accepted detections),
        #      masks_subset (np.ndarray, (A, 64, 256, 3), EQS mask accepted detections),
        #      utc_start_subset (list of UTCDateTime, A, window start accepted detections),
        #      stream_start_end_final (list of (UTCDateTime, UTCDateTime), A,
        #                              signal start/end per accepted detection)
        #      scores_final (list of float, A, detection score per accepted detection)
        #      A = number of accepted detections (<= D)
        stft_final_subset, masks_subset, utc_start_subset, stream_start_end_final, scores_final = \
            self._make_final_selection(y_predict_event, filtered_results,
                                       detection_start, selected_stft,
                                       selected_masks, selected_utc,
                                       stft_collection_subset,
                                       stream_start_end)

        # EQShyb — optional, runs only if model loaded and detections exist
        # IN:  stft_final_subset (A, 64, 256, 6), masks_subset (A, 64, 256, 3),
        #      utc_start_subset (list, A), data_stack (N, 3),
        #      data[0].stats.starttime (UTCDateTime, reference for sample indexing)
        # OUT: denoised_hyb (np.ndarray, (A, 6120, 3), EQShyb denoised waveforms)
        #      skipped if eqs2_model is None or no accepted detections
        denoised_hyb = None
        if self.eqs2_model is not None and stft_final_subset.shape[0] > 0:
            denoised_hyb = self._apply_eqshyb(stft_final_subset, masks_subset,
                                              utc_start_subset, data_stack,
                                              data[0].stats.starttime)

        # stream construction — uses EQShyb output if available, else EQS ISTFT
        #      utc_start_subset (list, A), stream_start_end_final (list, A),
        #      data (Stream, for trace metadata),
        #      denoised_hyb (A, 6120, 3) or None — if None uses EQS ISTFT path
        # OUT: trimmed_streams (obspy.Stream, sorted denoised traces),
        #      stream_start_end_final (list of (UTCDateTime, UTCDateTime), A, sorted)
        trimmed_streams, stream_start_end_final = \
            self._build_streams(stft_final_subset, masks_subset,
                                utc_start_subset, stream_start_end_final,
                                data, denoised_hyb)


        # remove detections too close together to be resolved by _trim_streams
        # IN:  trimmed_streams (obspy.Stream, sorted denoised traces),
        #      stream_start_end_final (list of (UTCDateTime, UTCDateTime), A, sorted),
        #      scores_final (list of float, A, detection score per accepted detection)
        # OUT: trimmed_streams (obspy.Stream, duplicates removed, higher-scoring kept),
        #      stream_start_end_final (list of (UTCDateTime, UTCDateTime), filtered)
        trimmed_streams, stream_start_end_final  = \
            self._filter_close_detections_streams(trimmed_streams, stream_start_end_final, scores_final)

        if len(stream_start_end_final) == 0:
            logger.info("No detections remaining after proximity filter")
            return None

        # trimmed_streams = self._trim_streams(trimmed_streams, stream_start_end)  # OLD
        # IN:  trimmed_streams (obspy.Stream), stream_start_end_final (list, A)
        # OUT: trimmed_streams (obspy.Stream, overlap-trimmed)
        trimmed_streams = self._trim_streams(trimmed_streams, stream_start_end_final)  # NEW

        # output - save to MSEED
        # IN:  data[0].stats.starttime (UTCDateTime, for output directory naming),
        #      trimmed_streams (obspy.Stream, overlap-trimmed denoised traces),
        #      gap_intervals (list of (UTCDateTime, UTCDateTime), gap regions to zero-mask
        #                    in output — empty list if no gaps)
        # OUT: writes denoised MiniSEED to disk with gap regions zeroed, returns nothing
        self._output(data[0].stats.starttime, trimmed_streams, gap_intervals)

        # phase picking — optional, only if picker configured and snippets exist
        # IN:  trimmed_streams (obspy.Stream, denoised snippets only),
        #      data (obspy.Stream, original restituted, in memory only — not re-fetched)
        # OUT: picks written to JSON on disk; MiniSEED already saved above as checkpoint
        if self.picker is not None and len(trimmed_streams):
            picks = self._pick(trimmed_streams, data)
            self._save_picks(picks, data[0].stats.starttime)

# %%
if __name__ == "__main__":

    # %% With picker + polarity
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    import sys
    sys.path.append("/home/niko/Earthquake-Seismogram-Denoiser/Code")
    from Denoiser_EQShyb import Denoiser

    import seisbench.models as sbm

    picker = sbm.EQTransformer.from_pretrained("ethz")

    # clients
    data_client     = Client("ETH")      # or "IRIS", "GFZ", SDS client, etc.
    metadata_client = Client("ETH")      # fdsn client for response

    denoiser = Denoiser(
        data_client     = data_client,
        metadata_client = metadata_client,
        model_path      = "/home/niko/Earthquake-Seismogram-Denoiser/Models/model_1000k_onlyweights.keras",
        min_peak_height = 0.33,
        eqs2_model_path = "/home/niko/Earthquake-Seismogram-Denoiser/Models/EQS2.keras",  # optional, omit for EQS only
        picker=picker,  # optional, omit to skip picking
        picking_kwargs={
            "repeat": 20,
            "pick_tolerance": 1,
            "p_confidence": 0.5,
            "s_confidence": 0.5,
            "min_share_models": 0.25
        },
        polarity_model_path="/home/niko/Schreibtisch/Polarity/Model/polarity_cnn_mixeddata_globalmaxavg_dropout02.keras",

        polarity_kwargs={"threshold": 0.33},
        debug=True
    )

    # ── single window ─────────────────────────────────────────────────────────
    denoiser.run_data(
        network   = "CH",
        station   = "SEMOS",
        location  = "*",
        channel   = "HG",
        starttime = UTCDateTime("2025-02-07T17:00:00"),
        endtime   = UTCDateTime("2025-02-08T17:00:00")
    )
# %%
from obspy import read
st = read("/home/niko/Earthquake-Seismogram-Denoiser/Models/DOY038/CH.SEMOS..HG_denoised.mseed")
st.plot()
# st = st.select(component="Z").merge().trim(starttime,endtime, fill_value=0, pad=True)
# st.plot(type="dayplot",interval=1,vertical_scaling_range=10000)
# %%
#
import json
import matplotlib.pyplot as plt
from obspy import UTCDateTime

with open("/home/niko/Earthquake-Seismogram-Denoiser/Models/DOY038/picks_DOY038.json") as f:
    picks = json.load(f)

p_times = [UTCDateTime(p["time"]) for p in picks["p_picks"]]
s_times = [UTCDateTime(s["time"]) for s in picks["s_picks"]]

fig, ax = plt.subplots(figsize=(14, 3))
ax.vlines([t.matplotlib_date for t in p_times], 0, 1, color="red",   label=f"P ({len(p_times)})")
ax.vlines([t.matplotlib_date for t in s_times], 0, 1, color="blue",  label=f"S ({len(s_times)})")
ax.xaxis_date()
fig.autofmt_xdate()
ax.set_xlabel("Time")
ax.set_yticks([])
ax.legend()
ax.set_title("Picks DOY038")
plt.tight_layout()
plt.show()