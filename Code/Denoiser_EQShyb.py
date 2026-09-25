import obspy.core
# import cupy as cp
import numpy as np
import scipy
import time
import logging
import threading
import queue
import json
# import os# NEW: for _save_picks()
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""      # hides GPUs from torch *and* TF
import tensorflow as tf
from tensorflow.keras.layers import Layer
from obspy.signal.invsim import cosine_taper, cosine_sac_taper
from obspy.signal.util import _npts2nfft
from functools import cache
from DenoisingFunctions_public import check_dir#, normalize_percentile
from scipy.signal import find_peaks
from pathlib import Path
from scipy.signal import istft
from obspy.signal.util import _npts2nfft
from obspy import Stream, Trace  # NEW: used in _stream_tta()
from concurrent.futures import ThreadPoolExecutor, as_completed  # NEW: used in _process_picks()
#################NEW#######################################
from obspy.core.event import Catalog, Comment
from obspy.core.event.origin import Pick as ObsPyPick   # aliased: `Pick` is the local dataclass
from obspy.core.event.base import WaveformStreamID, QuantityError
from obspy.core.event.event import Event
from obspy.core.event.header import EvaluationMode
from scipy.fft import rfft as _rfft, irfft as _irfft, next_fast_len # NEW DIFF PREPROCESSING
#################NEW#######################################
# tf.config.set_visible_devices([], 'GPU')  # turns GPU off
_PROGRAM_START = time.perf_counter()

logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.WARNING)
SENTINEL = object()

# ─── Module-level definitions ─────────────────────────────────────────────────
#
#  Functions (module-level):
#    _normalize_stft_channels() ← robust per-channel (median/IQR) normalisation
#                                 of one (64, 256, 6) STFT window
#    _predict_polarity_tta()   ← TTA polarity prediction on the Z-component batch
#    setup_logging()           ← configure relative-time logging handler
#
#    LEGACY (defined, no longer called anywhere):
#    apply_pre_filt()          ← zero-mean, cosine taper, FFT, cosine_sac_taper, IFFT
#    apply_pre_filt_trace()    ← apply pre-filter to a single trace
#    apply_pre_filt_stream()   ← apply pre-filter to all traces in a stream
#
#  Classes (module-level):
#    Pick                      ← dataclass: accepted phase pick (time, uncertainty,
#                                           share, event_id, polarity)
#    ObsPyPick                 ← alias for obspy.core.event.origin.Pick, to avoid
#                                collision with the local Pick dataclass above
#    MaxAbsNorm1D              ← Tensorflow/Keras layer: per-channel max-abs normalisation
#                                (custom_objects for the polarity model)
#    ReflectPad1D              ← Tensorflow/Keras layer: reflect padding 1D
#                                (custom_objects for the EQShyb model)
#    RelativeTimeFormatter     ← logging.Formatter subclass: elapsed-time prefix
#    Denoiser                  ← main class, see call trees below
#
# ─── Window / STFT geometry ───────────────────────────────────────────────────
#
#    len_sample      = 6120 samples = 61.2 s   → one model window
#    stft hop        = nperseg - noverlap = 24 samples = 0.24 s = one STFT bin
#    bins per window = 256                     → model input (64, 256, 6)
#    shift_samples   = 3072 samples = 128 bins → stride between windows (49.8 %
#                      overlap).  Must be a multiple of the hop, so that windows
#                      can be sliced out of one global STFT and so that the
#                      128-bin even/odd correction (bins_overlap) is exact.
#
#    Units: _get_peaks() and everything derived from it (filtered_results columns
#    0–2, bins_overlap, shift_correction) are BIN INDICES.  Conversion to time is
#    always "× self.bin_spacing" (0.24 s).  Sample indices are always
#    "(utc - starttime) × fs".
#
# ─── Denoiser call trees ──────────────────────────────────────────────────────
#
# run_timerange()                        ← multi-day entry point; producer-consumer pipeline
#  ├── _loader_thread()                  ← producer: fetch one day per iteration, push to queue
#  |    └── _round_to_window()           ← snap day end to exact multiple of 61.2 s
#  └── _consumer_thread()                ← consumer: drain queue, call run_data() per day
#       └── run_data()                   ← see below
#
# run_data()                             ← single-window entry point
#  ├── _round_to_window()                ← snap endtime to exact multiple of 61.2 s
#  ├── _get_data()                       ← fetch, gap detection, restitution, 100 Hz
#  |    ├── _get_metadata()              ← inventory subset via _query_server (cached)
#  |    |    └── _query_server()         ← FDSN inventory fetch, cached per network/station
#  |    └── _preprocess_combined()       ← ONE FFT pass: cosine taper + pre-filt +
#  |                                        spectral decimation to 100 Hz +
#  |                                        response removal (evalresp interpolated)
#  ├── _compute_global_stfts()           ← 3 global STFTs, sliced into overlapping 256-bin windows;
#  |    |                                   edge bins recomputed to reproduce the
#  |    |                                   per-window zero padding exactly
#  |    └── _normalize_stft_channels()   ← robust normalisation per window
#  ├── _detect_event_signals()           ← EQS first pass, peak detection on mask timeseries
#  |    ├── model.predict()              ← EQS mask prediction on all windows
#  |    ├── _get_mask_timeseries()       ← collapse mask array to even/odd timeseries
#  |    ├── _get_peaks()                  ← peaks with onset/end boundaries (×2: even, odd)
#  |    └── _compare_arrays_time_overlap() ← merge even/odd detections, keep higher-scoring
#  ├── _select_data_and_mask()           ← STFT window per detection; returns the
#  |                                        surviving filtered_results rows ("kept")
#  ├── _recompute_mask()                 ← re-align window to estimated signal start
#  |    └── _process_segment()           ← STFT + normalisation for one re-aligned window
#  |         └── _normalize_stft_channels()
#  ├── model.predict()                   ← EQS second pass on re-aligned windows
#  ├── _make_final_selection()           ← window scoring/selection; A ≤ D accepted
#  |    └── _get_peaks()                  ← peaks on the re-aligned mask timeseries
#  ├── _apply_eqshyb()                   ← optional, only if eqs2_model loaded and A > 0
#  |    └── eqs2_model.predict()         ← hybrid model (noisy + EQS denoised + EQS mask)
#  ├── _build_streams()                  ← ISTFT + stream assembly, EQS or EQShyb path;
#  |                                        sorted by signal start
#  ├── _filter_close_detections_streams() ← drop near-duplicate detections, keep best
#  ├── _trim_streams()                   ← resolve overlapping snippets, apply signal buffer
#  ├── _pick()                           ← optional, only if picker configured
#  |    ├── _get_designaled_noise()      ← per-snippet noise = original - denoised
#  |    └── _process_picks()             ← parallel picking over (snippet, noise) jobs
#  |         └── _process_snippet()      ← per-detection: TTA + phase picking + polarity
#  |              ├── _stream_tta()               ← inject std-scaled white noise, seeded by id
#  |              ├── picker.annotate()           ← SeisBench batch annotation
#  |              ├── picker.classify_aggregate() ← aggregate TTA picks
#  |              ├── _process_peak_times()       ← cluster picks + uncertainty per phase
#  |              |    ├── _cluster_picks()       ← group nearby picks, one per cluster
#  |              |    |    └── _weighted_median() ← confidence-weighted pick time
#  |              |    └── _tta_uncertainty()     ← timing spread across TTA reps
#  |              |         └── _weighted_std()   ← confidence-weighted std
#  |              └── _predict_polarity_tta()     ← optional, per accepted P pick
#  ├── _save_picks()                     ← serialise Pick objects; JSON and/or SC3ML
#  |    ├── _scale_uncertainty()         ← raw TTA sample-domain std → seconds
#  |    └── _build_catalog()             ← optional to collect picks, only if pick_output includes "sc3ml"
#  ├── _filter_streams_by_picks()        ← optional, only if filter_by_pick enabled
#  └── _output()                         ← gap masking, optional zero-padding,
#                                          write denoised (and optionally raw) MiniSEED
#
# ─── LEGACY: defined but not called by the pipeline ───────────────────────────
#
#    apply_pre_filt / _trace / _stream   ← replaced by _preprocess_combined()
#    _get_response_parameters()          ← cached response for _fast_remove_response()
#    _fast_remove_response()             ← replaced by _preprocess_combined()
#    _compute_stfts()                    ← per-window STFT; replaced by
#                                          _compute_global_stfts() (identical output)
#    commented-out normalize_percentile / sklearn variants (lines 184–265)

###############################################################################################
# LEGACY
def apply_pre_filt(data, samp_rate, pre_filt,taper_seconds=300):
    """Apply ObsPy's remove_response pre_filt step (no response correction).

    LEGACY: not called by the pipeline. _preprocess_combined() now performs the
    pre-filter, decimation and response removal in a single FFT pass. Kept for
    reference and A/B comparison.

    Reproduces the pre_filt block of obspy.core.trace.Trace.remove_response with
    zero_mean=True, taper=True, and a cosine taper given in seconds rather than
    as a fraction.

    data          : array-like        Raw time-domain signal.
    samp_rate     : float             Sample rate in Hz.
    pre_filt      : (f1, f2, f3, f4)  Bandpass corner frequencies in Hz.
    taper_seconds : float             Taper length per side, in seconds.

    Returns : ndarray float64, pre-filtered signal, same length as `data`.
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

    LEGACY: not called by the pipeline (see apply_pre_filt).

    trace         : obspy.Trace        Input trace (not modified).
    pre_filt      : (f1, f2, f3, f4)   Bandpass corner frequencies in Hz.
    taper_seconds : float              Taper length per side, in seconds.

    Returns : obspy.Trace, copy with pre-filtered data (float64).
    """
    out = trace.copy()
    out.data = apply_pre_filt(trace.data, trace.stats.sampling_rate, pre_filt,taper_seconds=taper_seconds)
    return out


def apply_pre_filt_stream(stream, pre_filt,taper_seconds=300):
    """Apply pre_filt to every trace in an ObsPy Stream, returns a new Stream.

    LEGACY: not called by the pipeline (see apply_pre_filt).

    stream        : obspy.Stream       Input stream (not modified).
    pre_filt      : (f1, f2, f3, f4)   Bandpass corner frequencies in Hz.
    taper_seconds : float              Taper length per side, in seconds.

    Returns : obspy.Stream, new stream with pre-filtered traces (float64).
    """
    return Stream([apply_pre_filt_trace(tr, pre_filt,taper_seconds=taper_seconds) for tr in stream])

# import numpy as np
# from sklearn.preprocessing import RobustScaler
#
#
# def normalize_percentile(
#     data,
#     quantile_range=(25, 75),
#     unit_variance=False,
#     limit=1000,
# ):
#     """
#     Robust normalization of 1-component complex data.
#
#     Real and imaginary components are normalized separately using
#     sklearn's RobustScaler, then clipped to [-limit, limit].
#
#     -------
#     np.ndarray
#         Normalized and clipped data with the same shape as `data`.
#     """
#     scaler = RobustScaler(
#         quantile_range=quantile_range,
#         unit_variance=unit_variance,
#     )
#     data_real = data[..., 0]
#     data_imag = data[..., 1]
#
#     real_norm = scaler.fit_transform(data_real.reshape(-1, 1)).reshape(data_real.shape)
#     imag_norm = scaler.fit_transform(data_imag.reshape(-1, 1)).reshape(data_imag.shape)
#
#     real_norm = np.clip(real_norm, -limit, limit)
#     imag_norm = np.clip(imag_norm, -limit, limit)
#
#     result = np.empty_like(data)
#     result[..., 0] = real_norm
#     result[..., 1] = imag_norm
#
#     return result

# END LEGACY
###############################################################################################

def _normalize_stft_channels(  # ADDED; TESTING WITHOUT SKLEARN
    data,
    quantile_range=(25, 75),
    unit_variance=False,  # Legacy
    limit=1000,
):
    """
    Robust per-channel normalisation of one STFT window.

    Replaces the former normalize_percentile()/RobustScaler path. For
    data.shape == (64, 256, 6), each of the 6 channels (real/imag of Z, N, E) is
    centred on its own median and divided by its own inter-quantile range, then
    clipped to [-limit, limit]. A zero range is replaced by 1.

    The input is not modified; the returned array is float32, the dtype the EQS
    model consumes.

    data           : np.ndarray (..., C); the C channels are normalised separately
    quantile_range : (low, high) percentiles defining the scale
    unit_variance  : unused, kept for signature compatibility (legacy)
    limit          : float, clip bound applied after scaling

    Returns : np.ndarray float32, same shape as `data`
    """

    # normalized = data.copy()
    normalized = data.astype(np.float32, copy=True)

    q_min, q_max = quantile_range

    # Compute median, lower quantile and upper quantile for each channel.
    center, q_low, q_high = np.nanpercentile(normalized,  # CHANGED TO NANPERCENTILE
        (50, q_min, q_max),axis=tuple(range(normalized.ndim - 1)))

    scale = q_high - q_low
    scale[scale == 0] = 1

    normalized -= center
    normalized /= scale

    np.clip(normalized,-limit,limit,out=normalized)

    return normalized

def _predict_polarity_tta(
    z_tta_collection,
    z_starttime,
    z_sampling_rate,
    p_pick,
    polarity_model,
    win=256,
    threshold=0.33,
    training=True
):
    """
    Polarity prediction from the TTA Z traces already produced by _stream_tta()
    — no re-augmentation and no noise re-scaling.

    One `win`-sample window centred on the P pick is cut from every Z trace in
    the TTA collection, zero-padded where the window runs past the trace, and
    peak-normalised. The batch goes through the polarity model; with
    training=True any dropout stays active, so the spread across the batch
    reflects both TTA noise and model uncertainty. The softmax vectors are
    averaged, and the winning class is returned unless its mean probability is
    below `threshold`, in which case 'undecidable' is returned.

    z_tta_collection : obspy.Stream, full TTA collection from _stream_tta()
                       (Z traces are selected here)
    z_starttime      : obspy.UTCDateTime, starttime of the padded Z snippet used
                       for picking, i.e. the reference the pick times refer to
    z_sampling_rate  : float, samples per second
    p_pick           : obspy.UTCDateTime, accepted P pick time
    polarity_model   : tf.keras.Model, input (batch, win) or (batch, win, 1)
    win              : int, sample window centred on the P pick (default 256)
    threshold        : float, min winning-class probability; below → undecidable
    training         : bool, keep dropout active (MC dropout) during inference

    Returns dict:
        label           : 'positive' | 'negative' | 'undecidable'
        probabilities   : np.ndarray (3,), mean softmax over the batch,
                          ordered [negative, undecidable, positive]
        all_predictions : np.ndarray (repeat, 3), per-repetition softmax
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

    pred      = polarity_model(z_batch, training=training).numpy()           # (repeat, 3)
    mean_pred = pred.mean(axis=0)                                        # (3,)

    label = labels[np.argmax(mean_pred)]
    if mean_pred.max() < threshold:
        label = "undecidable"

    return {
        "label":           label,
        "probabilities":   mean_pred,
        "all_predictions": pred,
    }

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Pick:
    """
    Accepted phase pick produced by the picking pipeline.

    time        : obspy.UTCDateTime, pick time
    uncertainty : float, weighted std of TTA argmax positions (raw sample units;
                  scaled to seconds by _scale_uncertainty() via uncertainty_scaling)
    share       : float, fraction of TTA repetitions whose peak confidence
                  exceeded the phase confidence threshold
    event_id    : str, trace id of the Z component (e.g. "CH.SEMOS..HGZ")
    polarity    : dict | None, result of _predict_polarity_tta(), P picks only.
                  Keys: 'label' (str), 'probabilities' (np.ndarray shape (3,)),
                        'all_predictions' (np.ndarray shape (repeat, 3))
    """
    time:        object          # obspy.UTCDateTime — not imported at module level
    uncertainty: float
    share:       float
    event_id:    str
    polarity:    Optional[dict] = field(default=None)

class MaxAbsNorm1D(tf.keras.layers.Layer):
    """
    Keras layer: divide each (batch, channel) by its own maximum absolute value
    over time. Part of the polarity model, so it must be passed in
    custom_objects when that model is loaded (see __init__, line 595).
    `eps` floors the divisor so an all-zero trace cannot produce NaNs.

    Input/output shape: (batch, time, channels)
    """
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
    """
    Keras layer: reflect-pad the time axis by `pad` samples on each side. Part
    of the EQShyb (EQS2) model, so it must be passed in custom_objects when that
    model is loaded (see __init__, line 571). Registered as a serialisable Keras
    object.

    Input shape:  (batch, time, channels)
    Output shape: (batch, time + 2*pad, channels)
    """
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
    Configure the root logger with a single stream handler that prefixes each
    line with elapsed time, logger name and function name.

    Existing root handlers are cleared, so this replaces any logging
    configuration set earlier in the session. Called once from
    Denoiser.__init__().

    debug : bool, DEBUG level if True, otherwise INFO
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

######################################################################################################################
# Main Denoiser class
class Denoiser(object):
    """
    Operational earthquake seismogram denoising, detection and phase picking.

    Implements the method of
    Nikolaj Dahmen, John Clinton, Men-Andrin Meier, Luca Scarabello,
    'Toward Operational Earthquake Seismogram Denoising',
    https://doi.org/10.1785/0120250198

    Pipeline per processing window (see the call tree at the top of this file):
    fetch and restitute waveforms → STFT over sliding 61.2 s windows → EQS mask
    prediction and detection → window re-alignment and second EQS pass →
    optional EQShyb time-domain refinement → stream assembly and overlap
    resolution → optional TTA phase picking and polarity → MiniSEED and pick
    files on disk.

    Two entry points:
      run_data()      — one time window (typically one day)
      run_timerange() — several days; a loader thread fetches the next day while
                        the consumer thread processes the current one

    Outputs are written next to the EQS model file, in a DOY<julday> folder:
      <stream_id>_denoised.mseed          denoised detection snippets
      <stream_id>_raw.mseed               restituted input (only if save_raw)
      picks_<stream_id>_DOY<julday>.json  picks (if pick_output includes json)
      picks_<stream_id>_DOY<julday>.xml   SC3ML (if pick_output includes sc3ml)

    Note that the number of traces in the denoised file is not necessarily the
    number of detections: _output() runs Stream._cleanup(), which merges
    snippets that end up exactly contiguous.

    Authors: Niko Dahmen, Roman Racine
    """

    def __init__(self, data_client, metadata_client,
                 model_path, min_peak_height, eqs2_model_path=None,
                 picker=None, picking_kwargs=None,
                 polarity_model_path=None, polarity_kwargs=None,
                 filter_by_pick=False,pick_output="json", debug=False):
        """
        Load the models and set all processing constants.

        data_client          : obspy client used to fetch waveforms
        metadata_client      : obspy client used to fetch station inventory
        model_path           : path to the trained EQS model. Its parent
                               directory is also the output root: results go to
                               <parent>/DOY<julday>/
        min_peak_height      : float, minimum mask-timeseries peak height for a
                               detection (first-pass threshold)
        eqs2_model_path      : optional path to the EQShyb (EQS2) model. If
                               given, accepted detections are refined in the
                               time domain and the EQS2 uncertainty calibration
                               is used
        picker               : optional SeisBench picker. If None, picking and
                               pick output are skipped entirely
        picking_kwargs       : optional dict passed to _process_picks()
                               (repeat, pick_tolerance, p_confidence,
                               s_confidence, min_share_models). A 'max_workers'
                               entry is consumed here into self.pick_workers.
                               The dict is copied, so the caller's is untouched
        polarity_model_path  : optional path to the polarity model, applied to
                               every accepted P pick. Expects input
                               (batch, 256) or (batch, 256, 1)
        polarity_kwargs      : optional dict; 'threshold' (float, default 0.33)
                               minimum winning-class probability, and
                               'mc_dropout' (bool, default True) to keep dropout
                               active during polarity inference
        filter_by_pick       : optional bool (default False). If True, only
                               detections with at least one P or S pick are
                               written to MiniSEED. Requires `picker`
        pick_output          : "json" | "sc3ml" | "both" (default "json").
                               Ignored when picker is None
        debug                : bool, enables DEBUG-level logging

        Instance attributes set here that control the processing:
          threshold            10    minimum summed mask value over a detection
                                     window for it to be accepted (set very low, not main detection threshold)
          buffer               300   pre/post seconds added to each fetch, used
                                     by the taper and trimmed off afterwards.
                                     Reduce for short windows
          len_sample           6120  samples per model window (61.2 s at 100 Hz)
          shift_samples        3072  stride between windows = 128 STFT bins.
                                     Must stay a multiple of the STFT hop (24)
          bins_overlap         128   bins shared by adjacent windows; maps
                                     even-stream peaks onto the odd stream
          pre_filt                   cosine taper corners (Hz) for restitution
          stft_parameters            nperseg 48, noverlap 24, nfft 126, fs 100
                                     → one window gives (64 freq, 256 time) bins
          bin_spacing          0.24  seconds per STFT bin
          REALIGN_SCORE_TOLERANCE 1  the re-aligned window replaces the original
                                     only if its score exceeds this factor times
                                     the original score
          signal_buffer_s      3.0   seconds kept before the signal start when
                                     trimming, and minimum separation between
                                     two detections
          pad_seconds          0     if > 0, each output trace is zero-padded
                                     backwards by this many seconds, clamped so
                                     traces never overlap. Implemented for SeisComP scamp, scmag
          one_sample_s         0.01  one sample in seconds at 100 Hz
          save_raw             False if True, _output() also writes the
                                     restituted input as <stream_id>_raw.mseed
          uncertainty_scaling        empirical sample-domain → seconds
                                     calibration, EQS or EQS2 depending on
                                     eqs2_model_path
          response_cache       {}    keyed by (net, sta, loc, epoch, npts);
                                     only used by the legacy response path
          components           None  set in _get_data(), Z first

        Raises ValueError if pick_output is not one of the three allowed values.
        """

        self.min_peak_height = min_peak_height  # main detection threshold
        self.data_client = data_client  # data client
        self.metadata_client = metadata_client  # metadata client
        self.threshold = 10  # minimum summed mask value over a detection window to accept a peak
        # !dont change for long windows for safe response removal:
        self.buffer = 300 # pre/post buffer (s) added to data fetch window for response removal (reduce for shorter time windows)
        self.len_sample = 6120  # length of denoiser prediction window
        # self.shift_samples = int(self.len_sample / 2)  # hop size between even/odd STFT streams (half window) # ORIGINAL
        self.shift_samples = 3072  # MOD for GLOBAL STFT, allows using global STFT and modifying it

        self.bins_overlap = 128  # number of overlapping STFT time bins between adjacent windows
        self.model_name = model_path  # path to EQS model, reused for output directory naming
        self.pre_filt = [1 / 100, 1 / 20, 45, 50]   # bandpass corners (Hz) for cosine taper pre-filter
        # lower upper corner frequency would remove high freq noise for few foreign 100sps stations (restitution noise),
        # but model was also trained this noise.
        # self.pre_filt = [1 / 100, 1 / 20, 45, 47.5]   # bandpass corners (Hz) for cosine taper pre-filter

        print("CHECK PREFILT")
        self.stft_parameters = {"nperseg": 48, "nfft": 126, "fs": 100,"noverlap": 24}
        self.REALIGN_SCORE_TOLERANCE = 1#0.5#1 # 0.5 # NEW added
        self.signal_buffer_s = 3.0  # buffer to start save denoised stream with at least 3s before signal start (ideally)
        self.pad_seconds = 120.0  #  NEW ZEROPADDING FOR SEISCOMP
        self.one_sample_s = 1.0 / self.stft_parameters["fs"]  # = 0.01s at 100 Hz

        self.bin_spacing = (self.stft_parameters["nperseg"] - self.stft_parameters["noverlap"]) / self.stft_parameters["fs"]  # = 0.24

        self.response_cache = {}
        self.model = tf.keras.models.load_model(model_path, compile=False)  # EQS model

        # EQShyb / EQS2
        self.eqs2_model = tf.keras.models.load_model(
            eqs2_model_path,
            custom_objects={"ReflectPad1D": ReflectPad1D},
            compile=False
        ) if eqs2_model_path else None
        self.save_raw = False

        # PICKER
        self.picker = picker  # passed seisbench picker
        self.picking_kwargs = dict(picking_kwargs or {})   # copy: don't mutate caller's dict
        self.pick_workers = self.picking_kwargs.pop("max_workers", 1)  # LEAVE AT 1 FOR REPRODUCABILITY

        if eqs2_model_path is None: # uncertainty scaling; calibrated for EQS + EQTransformer-ethz, check floor in weighed_std
            self.uncertainty_scaling = {
                'p_picks': {'scale_sample': 4*1.904, 'offset_sample': 9.249},
                's_picks': {'scale_sample': 4*2.211, 'offset_sample': 3.600},
            }
            logger.warning("adjust floor in weighted_std for EQS uncertainty scaling")
        else: # calibrated for EQS2 + EQTransformer-ethz
            self.uncertainty_scaling = {
                'p_picks': {'scale_sample': 11.622, 'offset_sample': 0},
                's_picks': {'scale_sample': 12.348, 'offset_sample': 0},
            }
        # POLARITY
        self.polarity_model = tf.keras.models.load_model(
            polarity_model_path,
            custom_objects={"custom>MaxAbsNorm1D": MaxAbsNorm1D},
            compile=False
        ) if polarity_model_path else None
        self.polarity_threshold = (polarity_kwargs or {}).get('threshold', 0.33)  # polarity minimum threshold; 0.33 --> effectively no threshold
        self.polarity_mc_dropout = (polarity_kwargs or {}).get('mc_dropout', True)   # flag to turn of MC dropout

        self.filter_by_pick = filter_by_pick  # if True, only write MiniSEED for detections with a P or S pick
        self.components = None  # set in _get_data()

        # pick_output : "json" | "sc3ml" | "both"
        if pick_output not in ("json", "sc3ml", "both"):
            raise ValueError(f"pick_output must be 'json', 'sc3ml' or 'both', got {pick_output!r}")
        self.pick_output = pick_output

        setup_logging(debug=debug)
        logger.debug("")


    def _loader_thread(self, startday, endday, network, station, location,
                       channel, output_queue):
        """
        Producer thread for run_timerange(): fetches one day at a time and pushes
        it onto the queue, so the next day downloads while the current one is
        processed.

        Each day is fetched with self.buffer seconds of extra data on each side,
        and its end is snapped to a whole number of 61.2 s windows by
        _round_to_window(). A SENTINEL is pushed after the last day.

        startday     : obspy.UTCDateTime, first day to process
        endday       : obspy.UTCDateTime, last day to process (inclusive)
        network      : str, FDSN network code
        station      : str, FDSN station code
        location     : str, FDSN location code (wildcards accepted)
        channel      : str, 2-char channel prefix; "?" is appended here
        output_queue : queue.Queue receiving (data, day_start, day_end, network,
                       station, location, channel) tuples, then SENTINEL
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
        Consumer thread for run_timerange(): takes one day off the queue and runs
        run_data() on it. Returns when SENTINEL is received. Runs in the calling
        thread, so exceptions propagate to the caller.

        input_queue : queue.Queue filled by _loader_thread()
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
        Return the inventory subset matching one request.

        The full station inventory is fetched (and cached) by _query_server();
        this method only applies inventory.select(), which keeps FDSN traffic at
        one request per station even when location, channel or time window vary.

        network   : str, network code
        station   : str, station code
        location  : str, location code (wildcards accepted)
        channel   : str, channel code (wildcards accepted)
        starttime : obspy.UTCDateTime, start of the validity window
        endtime   : obspy.UTCDateTime, end of the validity window

        Returns : obspy.Inventory, the selected subset
        """
        logger.debug("")
        inventory = self._query_server(network, station)
        return inventory.select(network, station, location, channel,
                                starttime, endtime)

    def _get_response_parameters(self, data, inventory):
        """
        Build, and cache, everything needed to deconvolve one trace length.

        LEGACY: only used by _fast_remove_response(), which the pipeline no
        longer calls. _preprocess_combined() computes the same quantities inline.

        The cache key is (network, station, location, epoch start, npts) — the
        channel code is deliberately omitted, assuming all three components share
        one response.

        The response is evaluated at RESP_NFFT (32768) frequencies and linearly
        interpolated onto the full FFT grid, real and imaginary parts separately.
        The response is a smooth rational function of frequency, so the error is
        a few times 1e-8 relative while avoiding millions of evalresp calls.

        data      : obspy.Stream, used only for length, delta and sampling rate
        inventory : obspy.Inventory, already selected for this station

        Returns tuple:
            taper_coeffs      np.ndarray (npts,), time-domain cosine taper
            freq_response     np.ndarray, INVERSE response on the FFT grid
                              (element 0 zeroed)
            freqs             np.ndarray, FFT frequencies in Hz
            freq_domain_taper np.ndarray, cosine_sac_taper for self.pre_filt
            nfft              int, FFT length used
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

        # OLD-----------
        # nfft = _npts2nfft(npts)
        # freq_response, freqs = \
        #     response.get_evalresp_response(data[0].stats.delta, nfft,
        #                                    output="VEL")
        # freq_domain_taper = cosine_sac_taper(freqs, flimit=self.pre_filt)
        # freq_response[0] = 0.0
        # freq_response[1:] = 1.0 / freq_response[1:]
        # self.response_cache[dictkey] = (taper_coeffs, freq_response, freqs,
        #                                 freq_domain_taper, nfft)
        # NEW-----------
        nfft = _npts2nfft(npts)
        freqs = np.fft.rfftfreq(nfft, d=data[0].stats.delta)

        # evaluate response at few points, interpolate to full grid
        RESP_NFFT = 32768
        resp_raw, freqs_small = response.get_evalresp_response(
            data[0].stats.delta, RESP_NFFT, output="VEL")
        resp_raw[0] = 1.0
        inv_resp_small = 1.0 / resp_raw

        freq_response = (
            np.interp(freqs, freqs_small, inv_resp_small.real) +
            1j * np.interp(freqs, freqs_small, inv_resp_small.imag)
        )
        freq_response[0] = 0.0

        freq_domain_taper = cosine_sac_taper(freqs, flimit=self.pre_filt)
        self.response_cache[dictkey] = (taper_coeffs, freq_response, freqs,
                                        freq_domain_taper, nfft)


        return (taper_coeffs, freq_response, freqs, freq_domain_taper, nfft)

    def _fast_remove_response(self, data, inventory):
        """
        Deconvolve the instrument response in place using cached parameters.

        LEGACY: not called by the pipeline; _preprocess_combined() does this as
        part of a single FFT pass.

        Each trace is demeaned, tapered, transformed, multiplied by the
        frequency-domain taper and the inverse response, and transformed back.

        data      : obspy.Stream, 3 traces of equal length; modified in place
        inventory : obspy.Inventory, already selected for this station

        Returns : None
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



    def _process_segment(self, data_window):  # NEW COMBINED 3 COMP
        """
        data_window: numpy array of shape (len_sample, 3)
                     Columns are Z, N, E components.
        Returns:
            (raw_stft, norm_stft) each shaped (64, 256, 6)
        """

        logger.debug("")

        if data_window.shape[0] != self.len_sample or \
                data_window.shape[1] != 3:
            logger.debug("returning None")
            return None

        # stft_tmp = np.zeros((64, 256, 6), dtype=float)
        # stft_tmp_norm = np.zeros((64, 256, 6), dtype=float)
        stft_tmp = np.zeros((64, 256, 6), dtype=np.float32)
        stft_tmp_norm = np.zeros((64, 256, 6), dtype=np.float32)
        # =========================================================
        # STFT
        # =========================================================
        for j in range(3):
            snippet_tmp = data_window[:, j]

            t_stft = time.perf_counter()

            _, _, _stft = scipy.signal.stft(
                snippet_tmp,
                **self.stft_parameters)
            # _stft = self.stft_gpu(snippet_tmp)

            # Real / imaginary
            stft_tmp[:, :, j * 2] = _stft.real
            stft_tmp[:, :, j * 2 + 1] = _stft.imag

        # =========================================================
        # NORMALIZATION
        # =========================================================
        t_norm = time.perf_counter()

        stft_tmp_norm = _normalize_stft_channels(
            stft_tmp,
            quantile_range=(25, 75),
            unit_variance=False,
            limit=1000,
        )

        return stft_tmp, stft_tmp_norm


    def _compare_arrays_time_overlap(self, array1, array2, overlap=0.75):
        """
        Merge the even and odd detection lists, keeping the better of each pair.

        Two intervals count as the same detection when they overlap by more than
        `overlap` times the shorter of the two durations. For each row of array1,
        every overlapping row of array2 is considered and the higher-scoring one
        wins. Rows of array2 that never overlapped anything are appended.

        The output therefore starts with one row per array1 entry, in array1
        order, followed by the unmatched array2 rows: it is NOT sorted in time.
        Callers that index other lists alongside it must preserve this order.

        All positions are in BINS, as produced by get_peaks().

        array1  : array-like (N1, 5), rows [peak, start, end, score, maxval]
                  (even stream, already shift-corrected)
        array2  : array-like (N2, 5), same layout (odd stream)
        overlap : float, minimum overlap fraction of the shorter interval

        Returns tuple:
            final_rows np.ndarray (D, 5), merged detections
            origins    list of int, 0 if the row came from array1, 1 from array2
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
        Collapse the 4D mask array into two 1D detection timeseries.

        For each window and time bin, the maximum mask value over frequency is
        taken per component, then averaged with double weight on the first
        component (Z) and single weight on the two horizontals.

        Windows are then split by parity and concatenated. Because shift_samples
        is 128 bins and a window is 256 bins, the even windows tile the time axis
        exactly, and so do the odd ones, offset by 128 bins. Array index therefore
        equals global bin index for the even stream.

        mask_array : np.ndarray (W, 64, 256, 3), EQS masks

        Returns tuple:
            array_even np.ndarray, concatenated bins of windows 0, 2, 4, …
            array_odd  np.ndarray, concatenated bins of windows 1, 3, 5, …
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



    def _get_peaks(self, timeseries, threshold=0.1, shift_correction=0):  # ORIGINAL
        """
        Find detection peaks in a mask timeseries, with onset and end.

        Peaks must exceed `threshold` and be at least 128 bins (30.7 s) apart.
        For each peak the onset is the last bin before it that fell below 0.01,
        and the end the first bin after it that falls below 0.05; both thresholds
        are fixed. The score is the sum of the timeseries between those bounds,
        which favours long, strong signals.

        All returned indices are BIN INDICES (0.24 s per bin), not seconds.
        Convert with "× self.bin_spacing".

        timeseries       : 1D np.ndarray, even or odd mask timeseries
        threshold        : float, minimum peak height
        shift_correction : int, subtracted from all three indices, in bins. Used
                           with bins_overlap (128) for the even stream so both
                           streams end up in one coordinate system

        Returns : np.ndarray (P, 5), columns
                  [peak, onset, end, score, peak_value]; empty array if no peak
                  passes the threshold.
        """

        logger.debug("")
        peaks, _ = find_peaks(timeseries, height=threshold, distance=128)

        peaks_info = []
        for peak in peaks:  # typically only for ~100 peaks / 24h
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


    # ################### QUICK TEST ----------------------------------------------------------
    def _preprocess_combined(self, data, inventory):
        """
        Restitute and downsample to 100 Hz in one FFT round-trip per trace.

        Replaces the former three-step chain apply_pre_filt_stream() →
        decimate()/resample() → _fast_remove_response(). All three were
        frequency-domain operations, so their operators are combined and applied
        once:

          1. demean, cosine taper (self.buffer seconds per side)
          2. forward FFT at the original sample rate
          3. multiply by cosine_sac_taper(self.pre_filt) × inverse response
          4. keep only bins up to the new Nyquist (50 Hz) and inverse-FFT at the
             shorter length — this performs the decimation, with the taper acting
             as the anti-alias filter
          5. rescale by nfft_new / nfft_orig, since the inverse FFT divides by
             its own length

        Two things make this fast. The FFT length comes from next_fast_len(npts)
        rather than _npts2nfft(npts): the doubling in the latter guards against
        circular convolution, which cannot occur here because the spectrum is
        only multiplied point-wise. And the response is evaluated at 32768
        frequencies (freq_resolution <0.01 Hz for 250sps data)
        and interpolated onto the FFT grid rather than evaluated per bin.

        data      : obspy.Stream, 3 traces of equal length and sample rate.
                    Modified in place: data, sampling_rate and npts are all
                    replaced with the 100 Hz version
        inventory : obspy.Inventory, already selected for this station

        Returns : None

        Requires the original sample rate to be at least 100 Hz.
        """
        fs_orig = data[0].stats.sampling_rate
        fs_target = self.stft_parameters["fs"]          # 100
        npts_orig = len(data[0].data)
        npts_new = int(round(npts_orig * fs_target / fs_orig))

        # CHECK saver way to compute nfft_orig, nfft_new
        # _ratio = Fraction(fs_orig / fs_target).limit_denominator(1000)
        # _p, _q = _ratio.numerator, _ratio.denominator
        # _k = next_fast_len(int(np.ceil(npts_orig / _p)))
        # nfft_orig, nfft_new = _p * _k, _q * _k


        # next_fast_len >= npts is enough with long buffer + cosine taper
        nfft_orig = next_fast_len(npts_orig)
        freqs = np.fft.rfftfreq(nfft_orig, d=1.0 / fs_orig)

        # ── frequency-domain taper (anti-alias + bandpass) ────────────
        freq_taper = cosine_sac_taper(freqs, flimit=self.pre_filt)

        # ── inverse response — evaluate at few points, interpolate ────
        RESP_NFFT = 32768
        # RESP_NFFT = np.min([32768,nfft_orig])# CHANGE TO THIS FOR SHORT TRACES?

        response = data[0]._get_response(inventory)
        resp_raw, freqs_small = response.get_evalresp_response(
            1.0 / fs_orig, RESP_NFFT, output="VEL")
        resp_raw[0] = 1.0
        inv_resp_small = 1.0 / resp_raw

        # check if prob at long period (?)
        freq_response = (
            np.interp(freqs, freqs_small, inv_resp_small.real) +
            1j * np.interp(freqs, freqs_small, inv_resp_small.imag)
        )
        freq_response[0] = 0.0

        # ── combined operator: taper × inverse response ───────────────
        combined = freq_taper * freq_response

        # ── time-domain taper ─────────────────────────────────────────
        p_frac = (self.buffer * fs_orig) / npts_orig
        taper = cosine_taper(npts_orig, p=p_frac,
                             sactaper=True, halfcosine=False)

        # ── spectral decimation sizes ─────────────────────────────────
        nfft_new = next_fast_len(npts_new)

        # print("==============TEST===============")
        # print(nfft_new == nfft_orig * fs_target / fs_orig)
        # print("=============================")

        n_copy = min(
            int(nfft_orig * fs_target / (2 * fs_orig)) + 1,
            nfft_new // 2 + 1,
            nfft_orig // 2 + 1,
        )
        scale = nfft_new / nfft_orig


        for trace in data:
            x = trace.data.astype(np.float64)
            x -= x.mean()
            x *= taper

            spec = _rfft(x, n=nfft_orig, workers=-1)
            spec *= combined

            spec_new = np.zeros(nfft_new // 2 + 1, dtype=spec.dtype)
            spec_new[:n_copy] = spec[:n_copy] * scale

            trace.data = _irfft(spec_new, n=nfft_new, workers=-1)[:npts_new]
            trace.stats.sampling_rate = fs_target
            trace.stats.npts = npts_new
    # ################### END QUICK TEST ----------------------------------------------------------


    def _get_data(self, network, station, location, channel, starttime,
                  endtime, data=None):
        """
        Fetch one window of waveforms and return them restituted at 100 Hz.

        Steps: fetch (unless `data` is supplied) with self.buffer seconds extra on
        each side → record gap intervals before merging loses them → merge with
        zero fill → select the matching inventory → restitute and downsample via
        _preprocess_combined() → trim the buffer off again.

        Sets self.components, sorted descending so Z comes first and the two
        horizontals follow in whatever order the station uses (N/E or 1/2).
        data_stack columns follow the same order.

        network   : str, FDSN network code
        station   : str, FDSN station code
        location  : str, FDSN location code (wildcards accepted)
        channel   : str, 2-char channel prefix; "?" is appended here
        starttime : obspy.UTCDateTime, start of the processing window
        endtime   : obspy.UTCDateTime, end of the processing window
        data      : obspy.Stream or None; if given it is used instead of
                    fetching, and must already include the buffer

        Returns tuple, or None when fewer than 3 traces survive the merge:
            data          obspy.Stream, 3 restituted traces at 100 Hz, buffer
                          trimmed off
            data_stack    np.ndarray (N, 3) float64, columns in self.components
                          order (Z first), velocity in m/s
            gap_intervals list of (UTCDateTime, UTCDateTime), gaps found before
                          merging; zeroed again in _output()
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



        self._preprocess_combined(data, metadata)
        ################### END QUICK TEST ----------------------------------------------------------

        data.trim(data[0].stats.starttime + buffer,
                  data[0].stats.endtime - buffer)


        self.components = sorted([tr.stats.channel[-1] for tr in data], reverse=True)  # NEW get components and fix order in data


        # z comp first, other components can be abitrarily sorted
        data_stack = np.column_stack([
            data.select(component=self.components[0])[0].data,
            data.select(component=self.components[1])[0].data,
            data.select(component=self.components[2])[0].data
        ])


        # return (data, data_stack)  # OLD
        return (data, data_stack, gap_intervals)  # NEW


    def _compute_stfts(self, data_stack, starttime, endtime):  # ORIGINAL + MOD
        """
        LEGACY — not called. run_data() uses _compute_global_stfts(), whose
        output is bit-identical to this (verified over a full record, all bins).
        Kept for reference and A/B testing.

        Per-window implementation: one scipy.signal.stft call per window and
        component, over a zero-copy sliding view of data_stack.

        Unlike _compute_global_stfts(), this does not require shift_samples to be
        a multiple of the STFT hop.

        data_stack : np.ndarray (N, 3), columns in self.components order
        starttime  : obspy.UTCDateTime of data_stack[0]
        endtime    : obspy.UTCDateTime, unused, kept for signature parity

        Returns tuple:
            selected_starttimes  list of UTCDateTime, start of each valid window
            stft_collection      np.ndarray (W, 64, 256, 6) float32, raw STFT
            stft_norm_collection np.ndarray (W, 64, 256, 6) float32, normalised
        """

        logger.debug("")
        # step = 61.2 / 2

        # num_windows = (data_stack.shape[0] - self.len_sample) // self.shift_samples + 1
        num_windows = max(0, (data_stack.shape[0] - self.len_sample) // self.shift_samples + 1)
        utc_start_list = [starttime + i * self.shift_samples / self.stft_parameters["fs"]
                          for i in range(num_windows)]

        # # ORIGINAL
        # starts = np.arange(num_windows) * self.shift_samples
        # windows = np.stack([data_stack[s:s+self.len_sample] for s in starts],
        #                    axis=0) # can it be replaced with np.sliding_window_view
        # # end ORIGINAL
        # NEW
        windows = np.lib.stride_tricks.as_strided(
            data_stack,
            shape=(num_windows, self.len_sample, 3),
            strides=(self.shift_samples * data_stack.strides[0],
                     data_stack.strides[0], data_stack.strides[1]),
            writeable=False
        )
        #end NEW

        # # ORIIGNAL
        # results = [self._process_segment(w) for w in windows]
        #
        # valid = [r for r in results if r is not None]
        # selected_starttimes = [t for t, r in zip(utc_start_list, results)
        #                        if r is not None]
        # stft_collection = np.stack([result[0] for result in valid], axis=0)  # (N,64,256,6)
        # stft_norm_collection = np.stack([result[1] for result in valid], axis=0)  # (N,64,256,6)
        # # end ORIGINAL

        # NEW
        stft_collection = np.zeros((num_windows, 64, 256, 6), dtype=np.float32)
        stft_norm_collection = np.zeros((num_windows, 64, 256, 6), dtype=np.float32)
        valid_mask = np.ones(num_windows, dtype=bool)
        for i in range(num_windows):
            result = self._process_segment(windows[i])
            if result is None:
                valid_mask[i] = False
            else:
                stft_collection[i] = result[0]
                stft_norm_collection[i] = result[1]

        stft_collection = stft_collection[valid_mask]
        stft_norm_collection = stft_norm_collection[valid_mask]
        selected_starttimes = [t for t, v in zip(utc_start_list, valid_mask) if v]
        # end NEW

        return (selected_starttimes, stft_collection, stft_norm_collection)

    def _compute_global_stfts(self, data_stack, starttime, endtime):
        """
        STFT the whole record once per component, then slice out model windows.

        One scipy.signal.stft call per component covers the full time range.
        Model window i is the 256-bin slice starting at global bin
        i * shift_samples/hop, which is the same as the STFT of samples
        [i*shift_samples : i*shift_samples + len_sample]. Every interior bin is
        therefore computed once instead of twice, since adjacent windows overlap
        by ~50 %.

        The two outermost bins are the exception. A per-window STFT pads 24 zeros
        on each side, so its bin 0 sees 24 zeros plus the first 24 samples, and
        its bin 255 the last 24 samples plus 24 zeros; the global STFT has real
        neighbouring data there instead. Both bins are recomputed from the
        half-zeroed segments, using scipy's own periodic Hann window scaled by
        1/sum(win) to match its 'spectrum' scaling. The result is bit-identical
        to the per-window version, zero-padding artefacts included, which is what
        the model was trained on.

        Requires shift_samples to be a whole number of STFT hops (asserted);
        3072 samples = 128 bins.

        data_stack : np.ndarray (N, 3), columns in self.components order
        starttime  : obspy.UTCDateTime of data_stack[0]
        endtime    : obspy.UTCDateTime, unused, kept for signature parity

        Returns tuple:
            utc_start_list       list of UTCDateTime, start of each window
                                 (starttime + i * shift_samples / fs)
            stft_collection      np.ndarray (W, 64, 256, 6) float32, raw STFT
            stft_norm_collection np.ndarray (W, 64, 256, 6) float32, normalised

        W is 0 when data_stack is shorter than one window; callers must handle
        the empty case. At 24 h / 100 Hz this is ~12 s and ~2.1 GB for the two
        collections, of which ~7 s is the normalisation loop.
        """
        logger.debug("")
        hop = self.stft_parameters["nperseg"] - self.stft_parameters["noverlap"]  # 24

        nperseg = self.stft_parameters["nperseg"]  # 48
        nfft = self.stft_parameters["nfft"]  # 126
        half_seg = nperseg // 2  # 24

        assert self.shift_samples % hop == 0, \
            "shift_samples must be a multiple of the STFT hop"

        bins_per_shift = self.shift_samples // hop  # 128
        bins_per_window = 256

        win = scipy.signal.get_window('hann', nperseg)  # periodic — scipy's default
        win = win / win.sum()


        # num_windows = (data_stack.shape[0] - self.len_sample) // self.shift_samples + 1
        num_windows = max(0, (data_stack.shape[0] - self.len_sample) // self.shift_samples + 1)
        utc_start_list = [
            starttime + i * self.shift_samples / self.stft_parameters["fs"]
            for i in range(num_windows)
        ]

        # ── slice + fix edge bins ─────────────────────────────────────
        stft_collection = np.zeros((num_windows, 64, 256, 6), dtype=np.float32)

        for j in range(3):
            _, _, Zxx = scipy.signal.stft(data_stack[:, j], **self.stft_parameters)


            for i in range(num_windows):
                bin_start = i * bins_per_shift
                window_slice = Zxx[:, bin_start:bin_start + bins_per_window].copy()

                s = i * self.shift_samples  # sample offset in data_stack

                # recompute bin 0: [zeros(24) | signal[s:s+24]] × Hann → FFT
                seg0 = np.zeros(nperseg)
                seg0[half_seg:] = data_stack[s:s + half_seg, j]
                seg0 *= win
                window_slice[:, 0] = np.fft.rfft(seg0, n=nfft)

                # recompute bin 255: [signal[s+6096:s+6120] | zeros(24)] × Hann → FFT
                seg_end = np.zeros(nperseg)
                end_idx = s + self.len_sample - half_seg
                seg_end[:half_seg] = data_stack[end_idx:end_idx + half_seg, j]
                seg_end *= win
                window_slice[:, -1] = np.fft.rfft(seg_end, n=nfft)

                stft_collection[i, :, :, j * 2] = window_slice.real
                stft_collection[i, :, :, j * 2 + 1] = window_slice.imag

            del Zxx

        # ── normalize ─────────────────────────────────────────────────
        stft_norm_collection = np.zeros_like(stft_collection)
        for i in range(num_windows):
            stft_norm_collection[i] = _normalize_stft_channels(
                stft_collection[i],
                quantile_range=(25, 75),
                unit_variance=False,
                limit=1000,
            )

        return (utc_start_list, stft_collection, stft_norm_collection)


    def _detect_event_signals(self, stft_norm_collection):
        """
        First EQS pass: predict masks for every window and detect signals.

        The mask array is collapsed into two timeseries by _get_mask_timeseries(),
        one from the even-numbered windows and one from the odd ones, each a
        continuous concatenation of 256-bin blocks. Peaks are found in both.
        Even-stream positions are shifted back by bins_overlap (128) so both
        streams share one coordinate system; this is exact because shift_samples
        is 128 bins. The two peak lists are then merged by
        _compare_arrays_time_overlap(), which keeps the higher-scoring of any
        overlapping pair.

        stft_norm_collection : np.ndarray (W, 64, 256, 6) float32

        Returns tuple:
            filtered_results np.ndarray (D, 5), one row per detection:
                             [peak, start, end, score, maxval]. Columns 0–2 are
                             BIN INDICES, not seconds. Even-derived rows come
                             first, then unmatched odd rows, so rows are NOT in
                             time order
            origin           list of int, 0 if the row came from the even stream,
                             1 if from the odd one
            y_predict        np.ndarray (W, 64, 256, 3), EQS masks, all windows
        """
        #################
        logger.debug("")
        model_verbose = 0

        y_predict = self.model.predict(stft_norm_collection, batch_size=32, # ORGIGINAL
                                       verbose=model_verbose)

        mask_timeseries_even, mask_timeseries_odd = \
            self._get_mask_timeseries(y_predict)

        # get peaks with start and end, with fixed min. threshold
        #  for max of time series (=at leats one bin with mask value>0.1)

        # account for 50% time shift
        peak_info_even = self._get_peaks(mask_timeseries_even,
                                        threshold=self.min_peak_height,
                                        shift_correction=128)
        peak_info_odd = self._get_peaks(mask_timeseries_odd,
                                       threshold=self.min_peak_height,
                                       shift_correction=0)

        filtered_results, origin = \
            self._compare_arrays_time_overlap(peak_info_even, peak_info_odd)

        logger.info(f"_detect_event_signals: {len(filtered_results)} detections found "
                    f"(even peaks: {len(peak_info_even)}, odd peaks: {len(peak_info_odd)})")

        return (filtered_results, origin, y_predict)

    def _select_data_and_mask(self, filtered_results, origin, y_predict,
                              stft_collection, selected_starttimes):
        """
        Map each detection onto the STFT window that contains it.

        The peak position is an index into the concatenated even or odd
        timeseries, so it is split into a window index and a bin offset within
        that window; even rows have their 128-bin correction added back first.
        When a detection falls in the last few bins of a window, the next window
        of the same parity is used and the offset reset to 0, so the signal does
        not sit on the window edge.

        Detections whose window index runs past the end of the prediction array
        are dropped. The surviving rows of filtered_results are returned as
        `kept`, so every returned list stays index-aligned; callers must use
        `kept` from here on rather than the original filtered_results.

        filtered_results    : np.ndarray (D, 5) from _detect_event_signals()
        origin              : list of int (D,), 0 even / 1 odd
        y_predict           : np.ndarray (W, 64, 256, 3), EQS masks
        stft_collection     : np.ndarray (W, 64, 256, 6), raw STFT
        selected_starttimes : list of UTCDateTime (W,), window start times

        Returns tuple, all of length K ≤ D and index-aligned:
            selected_masks  np.ndarray (K, 64, 256, 3)
            selected_stft   np.ndarray (K, 64, 256, 6)
            selected_utc    list of UTCDateTime, window start per detection
            detection_start list of UTCDateTime, estimated signal start
                            (window start + bin offset × self.bin_spacing)
            kept            np.ndarray (K, 5), the surviving filtered_results
        """
        # Select data and mask based on list
        selected_stft, selected_masks, selected_utc = [], [], []
        detection_start = []
        kept = []

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
                continue

            selected_masks.append(y_predict[index_window])
            selected_stft.append(stft_collection[index_window])
            selected_utc.append(selected_starttimes[index_window])
            detection_start.append(selected_starttimes[index_window] +
                                   bin_start*self.bin_spacing)
            kept.append(filtered_result)             # CHANGED: keep the full row

        n_dropped = len(filtered_results) - len(kept)
        if n_dropped:
            logger.info(f"_select_data_and_mask: {n_dropped} of {len(filtered_results)} "
                        f"detections dropped (index out of range)")

        selected_masks = np.array(selected_masks)
        selected_stft = np.array(selected_stft)
        return (selected_masks, selected_stft, selected_utc,
                detection_start, np.array(kept))

        # return (selected_masks, selected_stft, selected_utc,
        #         detection_start, detection_score)

    def _recompute_mask(self, detection_start, starttime, data_stack):
        """
        Re-cut each detection window so the signal starts at a known bin.

        The window is moved so the estimated signal start falls 42 bins (10.08 s)
        in, which keeps the onset away from the window edge and near the position
        the model saw most often in training. The window is cut from data_stack,
        then re-STFTed by _process_segment().

        Windows that would run past either end of data_stack are skipped: their
        entry in stream_start_end is None and their STFT rows stay zero.
        _make_final_selection() checks for None and falls back to the original
        window, so the detection is not lost.

        detection_start : list of UTCDateTime (D,), estimated signal starts
        starttime       : UTCDateTime of data_stack[0]
        data_stack      : np.ndarray (N, 3)

        Returns tuple, all of length D:
            stft_collection_subset      np.ndarray (D, 64, 256, 6) float32
            stft_norm_collection_subset np.ndarray (D, 64, 256, 6) float32
            stream_start_end            list of (UTCDateTime, UTCDateTime) or
                                        None, the re-aligned window bounds
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
            endidx = startidx + self.len_sample
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
        Decide, per detection, whether to keep the original or the re-aligned
        window, and drop everything below the acceptance threshold.

        For each re-aligned window the mask timeseries is rebuilt and peaks are
        found. The re-aligned window replaces the original only when it was
        computed at all (stream_start_end[i] is not None) and its score exceeds
        REALIGN_SCORE_TOLERANCE times the original score. Whichever window wins
        must then score above self.threshold to be accepted.

        The signal window is derived differently in the two cases: the original
        path uses the first-pass onset and end bins, the re-aligned path the
        onset and end bins of the new peak. Both convert bins to seconds with
        "× self.bin_spacing" — filtered_results columns 1 and 2 are bin indices,
        so the multiplication is required (this was a unit bug before).

        All inputs must be index-aligned, which the `kept` return of
        _select_data_and_mask() guarantees.

        y_predict_event        : np.ndarray (D, 64, 256, 3), second-pass masks
        filtered_results       : np.ndarray (D, 5), the `kept` rows
        detection_start        : list of UTCDateTime (D,)
        selected_stft          : np.ndarray (D, 64, 256, 6), original windows
        selected_masks         : np.ndarray (D, 64, 256, 3), original masks
        selected_utc           : list of UTCDateTime (D,), original window starts
        stft_collection_subset : np.ndarray (D, 64, 256, 6), re-aligned windows
        stream_start_end       : list of (UTCDateTime, UTCDateTime) or None (D,)

        Returns tuple, all of length A ≤ D:
            stft_final_subset      np.ndarray (A, 64, 256, 6)
            masks_subset           np.ndarray (A, 64, 256, 3)
            utc_start_subset       list of UTCDateTime, window start per detection
            stream_start_end_final list of (UTCDateTime, UTCDateTime), signal
                                   start and end per detection
            scores_final           list of float, accepted score per detection

        When nothing is accepted, empty arrays of the right shape and three empty
        lists are returned.
        """

        logger.debug("")
        stft_final_subset, masks_subset, utc_start_subset = [], [], []
        stream_start_end_final = []
        scores_final = []  # NEW
        for i, y_event in enumerate(y_predict_event):
            _timeseries = (2 * np.max(y_event[:, :, 0], axis=0) +
                           np.max(y_event[:, :, 1], axis=0) +
                           np.max(y_event[:, :, 2], axis=0)) / 4
            _peak = self._get_peaks(_timeseries,
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
                # print("CHECK REALIGN_SCORE_TOLERANCE")

            if _score > self.threshold:
                if keep_old:
                    stft_final_subset.append(selected_stft[i])
                    masks_subset.append(selected_masks[i])
                    utc_start_subset.append(selected_utc[i])
                    # detect_duration = filtered_results[i][2] -\  # ORIGINAL
                    #     filtered_results[i][1]
                    detect_duration = (filtered_results[i][2] - filtered_results[i][1]) * self.bin_spacing # NEW / CORRECTION


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
        Turn accepted detections into an ObsPy Stream of denoised snippets.

        With denoised_hyb given, its waveforms are used directly (EQShyb path);
        otherwise each component is reconstructed by ISTFT of the masked STFT
        (EQS path). Both produce len_sample samples starting at the window start,
        in float32.

        Traces are grouped Z/N/E per detection in self.components order, and the
        groups are sorted by signal start (stream_start_end_final[i][0]), not by
        window start. Everything downstream relies on the triples staying
        contiguous and in this order.

        stft_final_subset      : np.ndarray (A, 64, 256, 6), raw STFT
        masks_subset           : np.ndarray (A, 64, 256, 3), EQS masks
        utc_start_subset       : list of UTCDateTime (A,), window starts
        stream_start_end_final : list of (UTCDateTime, UTCDateTime) (A,)
        data                   : obspy.Stream, source of the trace headers
        denoised_hyb           : np.ndarray (A, 6120, 3) or None

        Returns tuple:
            trimmed_streams        obspy.Stream, 3*A traces, sorted by signal
                                   start, Z/N/E contiguous per detection
            stream_start_end_final list of (UTCDateTime, UTCDateTime), sorted
                                   the same way
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

        The two model inputs are normalised the way EQS2 was trained: the noisy
        waveform is demeaned per channel and divided by one std per detection
        (over all three components), and the EQS stage-1 waveform is divided by
        that same std without demeaning. The output is rescaled by the same std,
        so the result is back in physical units.
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

    def _output(self, starttime, trimmed_streams, gap_intervals, data_raw=None):
        """
        Write the denoised snippets, and optionally the restituted input, to
        MiniSEED.

        Traces are regrouped by component so each component's snippets are
        contiguous, then Stream._cleanup() merges any that are exactly adjacent
        (or that overlap with identical samples). This means the number of traces
        in the file is not necessarily the number of detections. Samples inside
        recorded gaps are then zeroed, since the model produces output there from
        zero-filled input.

        When self.pad_seconds > 0, each trace is additionally extended backwards
        with zeros by that many seconds, clamped so it never reaches into the
        previous trace of the same channel (one sample of separation is kept).
        This gives downstream systems such as SeisComP some lead-in before each
        onset. The clamp state is local to this call, so padding does not carry
        across processing windows.

        Output goes to <model parent>/DOY<julday>/, FLOAT32 encoded:
            <stream_id>_denoised.mseed   always
            <stream_id>_raw.mseed        only if data_raw is given and
                                         self.save_raw is True

        starttime       : UTCDateTime, used for the DOY folder name — pass the
                          same value to _save_picks() so both land together
        trimmed_streams : obspy.Stream, 3*A traces, Z/N/E contiguous
        gap_intervals   : list of (UTCDateTime, UTCDateTime) from _get_data()
        data_raw        : obspy.Stream or None, the restituted input to archive

        Returns : None. Returns early without writing when the stream is empty.
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


        # ── START NEW PART prepend zero-padding to each trace ───────────────────────────
        if self.pad_seconds > 0:
            output_stream.sort(keys=['channel', 'starttime'])
            prev_end = {}
            for tr in output_stream:
                pad_start = tr.stats.starttime - self.pad_seconds
                if tr.id in prev_end:
                    pad_start = max(pad_start,
                                    prev_end[tr.id] + self.one_sample_s)
                prev_end[tr.id] = tr.stats.endtime
                tr.trim(starttime=pad_start, pad=True, fill_value=0)
        # ── END OF NEW PART ───────────────────────────

        dir_tmp = str(Path(self.model_name).parent /
                      ("DOY" + str(starttime.julday).zfill(3))) + "/"
        check_dir(dir_tmp)

        # for tr in output_stream:
        #     tr.data = np.nan_to_num(tr.data, nan=0.0)

        output_stream.write(dir_tmp +
                            trimmed_streams[0].id[:-1] +
                            "_denoised.mseed",
                            format="MSEED", encoding="FLOAT32")

        if data_raw is not None and getattr(self, "save_raw", False):
            data_raw.write(dir_tmp + data_raw[0].id[:-1] + "_raw.mseed",
                           format="MSEED", encoding="FLOAT32")

    def _trim_streams(self, trimmed_streams, startstop):
        """
        Resolve overlaps between consecutive detection snippets.

        Each snippet is 61.2 s long, so neighbouring detections often overlap even
        when their signals do not. Walking through the detections in order, four
        cases are handled:

          1. the traces do not overlap at all           → keep unchanged
          2. traces overlap but this signal ends before
             the next trace starts                      → cut at this signal end
          3. traces and signals overlap, but the gap to
             the next signal exceeds signal_buffer_s    → cut at this signal end,
                                                          and start the next
                                                          snippet signal_buffer_s
                                                          before its own signal
          4. the signals themselves overlap             → cut signal_buffer_s
                                                          before the next signal,
                                                          and start the next
                                                          snippet there

        Cases 3 and 4 modify the following triple in place inside trimmed_streams,
        so each detection is only ever cut once. The last detection is appended
        untouched.

        Requires trimmed_streams sorted by signal start with Z/N/E contiguous per
        detection, and startstop index-aligned with it — both guaranteed by
        _build_streams() and _filter_close_detections_streams(). startstop[i][1]
        (the signal end) is read here, so its units matter: see
        _make_final_selection().

        trimmed_streams : obspy.Stream, 3*A traces
        startstop       : list of (UTCDateTime, UTCDateTime) (A,), signal bounds

        Returns : obspy.Stream, the trimmed snippets in the same order
        """
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
                # logger.debug("No overlap")
                continue

            if startstop[i][1] < tr_next.stats.starttime:
                # logger.debug("No signal overlap with next stream")
                for comp in self.components:
                    new_trimmed_stream += triple_i.select(component=comp)[0].slice(
                        endtime=startstop[i][1] - one_sample)

            elif startstop[i][1] + buf < startstop[i + 1][0]:
                # logger.debug("No signal overlap")
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
        Detections closer than signal_buffer_s cannot be separated by
        _trim_streams(), which would produce zero-length slices, so one of each
        pair is dropped here. Comparison is always against the last kept
        detection, so a chain of close detections collapses to its best member.

        Returns (filtered_streams, filtered_startstop), index-aligned and still
        sorted by signal start.
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

    def _weighted_std(self, values, weights, floor=0):
        """
        Compute the weighted standard deviation of an array.
        Used by _tta_uncertainty() to quantify pick timing spread across TTA reps.

        values  : 1D array-like
        weights : 1D array-like, corresponding weights
        Returns : float, weighted standard deviation
        """
        average = np.average(values, weights=weights + 1e-30)
        variance = np.average((values - average) ** 2, weights=weights + 1e-30)
        return np.sqrt(variance + floor**2)


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
        return self._weighted_std(_argmax, _max, floor=1), reached_threshold


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


    def _stream_tta(self, _event_stream, _noise_std, id=0, # ORIGINAL
                    white_noise_factor=0.01):
        """
        Apply one TTA augmentation by injecting amplitude-scaled white noise.

        The per-component noise levels are computed once per detection by
        _process_snippet() from the designaled noise, trimmed to the unpadded
        window, and passed in here as plain numbers. Each id produces a different
        but fully reproducible noise realisation, and is also written into
        stats.location so the annotations can be grouped by repetition later.

        _event_stream     : obspy.Stream, denoised event waveforms (Z/N/E)
        _noise_std        : sequence of float, one std per component, in
                            self.components order
        id                : int, TTA index — seeds the RNG and becomes the
                            2-digit location code
        white_noise_factor: float, global scaling of the injected noise

        Returns : obspy.Stream, augmented copy of _event_stream
        """
        _event_noiseinjected = _event_stream.copy()
        # seed by id — same id always produces same noise sequence
        rng = np.random.default_rng(seed=id)
        for comp, noise_std_comp in zip(self.components, _noise_std):
            tr_denoised = _event_noiseinjected.select(component=comp)[0]
            tr_denoised.data += (white_noise_factor
                                 * noise_std_comp
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
        The three traces are copied before being padded by `add` seconds on each
        side. The padding gives the picker room to slide its window past arrivals
        near the snippet edges, and the copy keeps that padding out of
        trimmed_streams, which is written to disk later. The noise std is computed
        on the unpadded window, so the zero padding cannot dilute it.

        event_streams  : tuple of (tr_Z, tr_N, tr_E) obspy.Trace objects
        st_designaled  : obspy.Stream, per-snippet designaled noise for TTA
                         amplitude scaling
        repeat         : int, number of TTA augmentations
        pick_tolerance : float, clustering tolerance (s)
        p_confidence   : float, min picker confidence to accept a P pick
        s_confidence   : float, min picker confidence to accept an S pick
        Returns        : dict with keys 'p_picks' and 's_picks', each a list
                         of Pick objects. P picks carry a polarity dict in
                         Pick.polarity when self.polarity_model is not None;
                         S picks always have Pick.polarity = None.
        """
        # compute noise std per component
        _noise = obspy.core.Stream([
            st_designaled.select(component=c)[0].copy()
            for c in self.components
        ])
        _st_z, _st_1, _st_2 = [tr.copy() for tr in event_streams]  #
        n_start, n_end = _st_z.stats.starttime, _st_z.stats.endtime
        for tr in _noise: # select noise before paddign for std comp.
            tr.trim(n_start, n_end, pad=True, fill_value=0)

        add = 5 if _st_z.stats.npts >= 6120 else 5 + (6120 - _st_z.stats.npts) / 200
        # make longer for seisbench picker, white noise added to 0s part with TTA
        for st in (_st_z, _st_1, _st_2):
            st.trim(st.stats.starttime - add, st.stats.endtime + add,
                    pad=True, fill_value=0)
        _start, _end = _st_z.stats.starttime, _st_z.stats.endtime   # padded start/end for annotations

        event_tta_collection = Stream()
        # compute noise std once here for all TTA repeats
        noise_std_3comp = [
            np.std(_noise.select(component=comp)[0].data)
            for comp in self.components
        ]

        for i in range(repeat):  # per detection/waveform -> return 20 copies noise-augmented
            event_tta_collection += self._stream_tta(
                Stream([_st_z, _st_1, _st_2]), _noise_std=noise_std_3comp,
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
            polarity = None
            if self.polarity_model is not None:
                polarity = _predict_polarity_tta(
                    z_tta_collection=event_tta_collection,
                    z_starttime=_st_z.stats.starttime,
                    z_sampling_rate=_st_z.stats.sampling_rate,
                    p_pick=p_median,
                    polarity_model=self.polarity_model,
                    threshold=self.polarity_threshold,
                    training=self.polarity_mc_dropout
                )
            p_picks.append(Pick(
                time=p_median,
                uncertainty=p_result[0],
                share=p_result[1],
                event_id=event_id,
                polarity=polarity,
            ))

        s_picks = [
            Pick(time=m, uncertainty=r[0], share=r[1], event_id=event_id)
            for m, r in zip(s_picks_median, s_results)
        ]

        return {'p_picks': p_picks, 's_picks': s_picks}



    def _process_picks(self, snippet_jobs,  # NEW
                           repeat=20, pick_tolerance=1,
                           p_confidence=0.5, s_confidence=0.5,
                           min_share_models=0.25,
                           max_workers=None):
        """
        Run TTA picking over pre-paired (snippet, noise) jobs.

        _pick() builds the pairs, so no positional re-matching is needed here and
        the pairing cannot shift when a detection is missing its noise.

        snippet_jobs : list of (event_triple, noise_stream); event_triple is an
                       obspy.Stream of exactly 3 traces (Z/N/E) and noise_stream
                       the matching designaled noise of equal length and
                       starttime
        max_workers  : int or None; None uses self.pick_workers. Kept at 1 by
                       default, since thread scheduling changes the TTA ordering
                       and therefore the picks

        Returns dict with 'p_picks' and 's_picks', each a list of Pick objects
        accumulated over all jobs.
        """
        all_results = {'p_picks': [], 's_picks': []}
        workers = max_workers or self.pick_workers

        if workers == 1:
            results = [self._process_snippet(event_streams, noise_stream,
                                             repeat, pick_tolerance, p_confidence, s_confidence)
                       for event_streams, noise_stream in snippet_jobs]      # ◀ CHANGED: iterate tuples directly
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self._process_snippet, event_streams, noise_stream,
                                           repeat, pick_tolerance, p_confidence, s_confidence)
                           for event_streams, noise_stream in snippet_jobs]  # ◀ CHANGED: iterate tuples directly
                results = [f.result() for f in futures]

        for result in results:
            all_results['p_picks'].extend(result['p_picks'])
            all_results['s_picks'].extend(result['s_picks'])

        all_results['p_picks'] = [p for p in all_results['p_picks'] if p.share > min_share_models]
        all_results['s_picks'] = [p for p in all_results['s_picks'] if p.share > min_share_models]

        return all_results

    def _pick(self, trimmed_streams, data_original):  # NEW
        """
        Run phase picking on denoised event snippets using TTA.
        Computes per-snippet designaled noise (original - denoised) to provide
        the amplitude info for TTA white noise scaling. Memory cost scales
        with number of detections x snippet length.
        Every detection contributes exactly one (snippet, noise) pair, so the
        pairing in _process_picks() cannot shift. Zero-length snippets and failed
        noise computations both get a zero-filled placeholder rather than being
        skipped.

        trimmed_streams : obspy.Stream
            Denoised event snippets from _trim_streams(), Z/N/E per detection.
            Picker runs on these only — never on continuous data.
        data_original   : obspy.Stream
            Original restituted full-day stream from _get_data().
            Only in memory (not saved to disk). Used only to compute
            per-snippet noise via (original - denoised).

        Returns dict with keys 'p_picks' and 's_picks', each a list of Pick
                objects (time, uncertainty, share, event_id, polarity).
        """
        # logger.debug(f"_pick: data_original has {len(data_original)} traces: "
        #              f"{[tr.stats.channel for tr in data_original]}")


        num_detections = len(trimmed_streams) // 3
        snippet_jobs = []
        data_start = data_original[0].stats.starttime
        data_end = data_original[0].stats.endtime

        for i in range(num_detections):
            snippet = trimmed_streams[3 * i: 3 * (i + 1)]
            tr_ref = snippet.select(component=self.components[0])[0]

            event_triple = tuple(snippet.select(component=c)[0]
                                 for c in self.components)  # where

            if tr_ref.stats.npts == 0:
                logger.warning(f"Detection {i}: zero-length snippet — using zero noise")
                noise_snippet = obspy.core.Stream()
                for tr in snippet:
                    zero_tr = tr.copy()
                    zero_tr.data = np.zeros_like(tr.data)
                    noise_snippet += zero_tr
                snippet_jobs.append((event_triple, noise_snippet))
                continue

            start = max(tr_ref.stats.starttime, data_start)  # clamp
            end = min(tr_ref.stats.endtime, data_end)  # clam
            npts = tr_ref.stats.npts

            original_snippet = obspy.core.Stream()
            for tr in data_original:
                tr_sliced = tr.slice(start, end)
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
                noise_snippet = obspy.core.Stream()                # ◀ CHANGED: local stream
                for tr in snippet:
                    zero_tr = tr.copy()
                    zero_tr.data = np.zeros_like(tr.data)
                    noise_snippet += zero_tr

            snippet_jobs.append((event_triple, noise_snippet))


        picks = self._process_picks(snippet_jobs, **self.picking_kwargs)

        logger.info(f"Picks: {len(picks['p_picks'])} P, "
                    f"{len(picks['s_picks'])} S")
        return picks



    def _save_picks(self, picks, starttime, stream_id=None):
        """
        Save picks alongside the MiniSEED output. Format(s) controlled by
        self.pick_output ("json" | "sc3ml" | "both").

        "polarity" and "polarity_probabilities" are present only when a polarity
        model is configured (Pick.polarity is not None); S picks never carry polarity.

        SC3ML layout: one Catalog holding one Event with all P and S picks,
        built by _build_catalog().

        Uncertainty is scaled from raw TTA sample-domain std to seconds by
        _scale_uncertainty() — applied exactly once, in both output paths.

        picks     : dict with keys 'p_picks' and 's_picks', each a list of Pick
                    objects as produced by _pick().
        starttime : obspy.UTCDateTime, used for output directory naming (DOYxxx),
                    must match the starttime passed to _output().
        stream_id : str or None, stream identifier without component character,
                    e.g. "CH.SEMOS..HG". When None, derived from the first
                    available pick's event_id by stripping the trailing component
                    character (event_id[:-1]).
        """
        dir_tmp = str(Path(self.model_name).parent /
                      ("DOY" + str(starttime.julday).zfill(3))) + "/"
        check_dir(dir_tmp)

        if stream_id is None:
            first = (picks['p_picks'] or picks['s_picks'] or [None])[0]
            stream_id = first.event_id[:-1] if first else "unknown"

        stem = f"picks_{stream_id}_DOY{str(starttime.julday).zfill(3)}"

        if self.pick_output in ("json", "both"):
            serialisable = {}
            for phase, pick_list in picks.items():
                entries = []
                for pick in pick_list:
                    entry = {
                        "time": str(pick.time),
                        "uncertainty": self._scale_uncertainty(pick, phase),
                        "share": pick.share,
                        "id": pick.event_id,
                    }
                    if pick.polarity is not None:
                        entry["polarity"] = pick.polarity["label"]
                        entry["polarity_probabilities"] = pick.polarity["probabilities"].tolist()
                    entries.append(entry)
                serialisable[phase] = entries

            out_path = dir_tmp + stem + ".json"
            with open(out_path, "w") as f:
                json.dump(serialisable, f, indent=2)
            logger.info(f"Picks written to {out_path}")

        if self.pick_output in ("sc3ml", "both"):
            out_path = dir_tmp + stem + ".xml"
            self._build_catalog(picks).write(out_path, "SC3ML")
            logger.info(f"Pick catalog written to {out_path}")

    def _filter_streams_by_picks(self, trimmed_streams, picks):
        """
        Keep only detections that have at least one P or S pick falling
        within their time window.

        trimmed_streams : obspy.Stream, Z/N/E triples, one per detection
        picks           : dict with keys 'p_picks' and 's_picks', each a
                          list of Pick objects as returned by _pick()
        Returns         : obspy.Stream, filtered to detections with picks
        """
        all_pick_times = (
                [p.time for p in picks['p_picks']]
                + [p.time for p in picks['s_picks']]
        )
        n = len(trimmed_streams) // 3
        kept = obspy.core.Stream()
        for i in range(n):
            triple = trimmed_streams[3 * i: 3 * (i + 1)]
            tr_z = triple.select(component=self.components[0])[0]
            t0, t1 = tr_z.stats.starttime, tr_z.stats.endtime
            if any(t0 <= pt <= t1 for pt in all_pick_times):
                kept += triple
        logger.info(
            f"filter_by_pick: keeping {len(kept) // 3}/{n} detections with picks"
        )
        return kept
    # =========================================================================
    # END picking methods
    # =========================================================================
    ################NEW########################3
    def _scale_uncertainty(self, pick, phase):
        """
        Convert raw TTA sample-domain std to seconds using the empirical
        calibration in self.uncertainty_scaling. Called once per pick;
        both the JSON and SC3ML writers consume the result.

        pick  : Pick, as produced by _pick()
        phase : str, 'p_picks' or 's_picks'
        Returns : float, timing uncertainty in seconds
        """
        coeff = self.uncertainty_scaling[phase]
        return (coeff['scale_sample'] * pick.uncertainty
                + coeff['offset_sample']) / self.stft_parameters["fs"]

    def _build_catalog(self, picks):
        """
        Build an obspy Catalog holding a single Event with all P and S picks.

        Uncertainty is taken from _scale_uncertainty() — already calibrated,
        applied symmetrically as lower/upper. `share` is carried as
        confidence_level (in percent). Polarity, when a polarity model is
        configured, maps directly onto ObsPy's Pick.polarity vocabulary
        ('positive' | 'negative' | 'undecidable'); the softmax vector is
        attached as a Comment since it has no native field.

        picks   : dict with keys 'p_picks' / 's_picks', lists of Pick objects
        Returns : obspy.core.event.Catalog
        """
        event_picks = []
        for phase, hint in (('p_picks', 'P'), ('s_picks', 'S')):
            for p in picks[phase]:
                code = p.event_id.split(".")
                if len(code) == 4:
                    net, sta, loc, cha = code
                elif len(code) == 3:
                    net, sta, loc = code
                    cha = ""
                else:
                    logger.warning(f"_build_catalog: unparsable event_id {p.event_id!r} — skipping pick")
                    continue

                unc = self._scale_uncertainty(p, phase)
                obspy_pick = ObsPyPick(
                    time=p.time,
                    waveform_id=WaveformStreamID(network_code=net, station_code=sta,
                                                 location_code=loc, channel_code=cha),
                    phase_hint=hint,
                    evaluation_mode=EvaluationMode("manual"),
                    time_errors=QuantityError(lower_uncertainty=unc,
                                              upper_uncertainty=unc,
                                              confidence_level=p.share * 100),
                )
                if p.polarity is not None:
                    obspy_pick.polarity = p.polarity["label"]
                    obspy_pick.comments.append(Comment(
                        text="polarity_probabilities="
                             + json.dumps(p.polarity["probabilities"].tolist())))
                event_picks.append(obspy_pick)

        cat = Catalog()
        cat.append(Event(event_type="earthquake", picks=event_picks))
        return cat
    ################NEW########################3

    def run_timerange(self, network, station, location, channel,
                      startday, endday):
        """
        Process several days, overlapping download and computation.

        A loader thread fetches one day at a time into a queue of size 1 while
        this method runs the consumer loop in the calling thread, so the next day
        downloads while the current one is processed. The response cache and the
        loaded models are reused across days.

        Both threads share this instance, so a run is still single-station.

        network  : str, FDSN network code
        station  : str, FDSN station code
        location : str, FDSN location code (wildcards accepted)
        channel  : str, 2-char channel prefix, e.g. "HH"
        startday : obspy.UTCDateTime, first day to process
        endday   : obspy.UTCDateTime, last day to process (inclusive)

        Returns : None — results are written to disk per day by run_data()
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
        Rounding up means the last window may extend past the requested endtime;
        _get_data() fetches self.buffer seconds beyond it anyway.
        """
        window_s = self.len_sample / self.stft_parameters["fs"]  # 61.2s
        duration = endtime - starttime
        n_windows = int(np.ceil(duration / window_s))
        return starttime + n_windows * window_s

    def run_data(self, network, station, location, channel, starttime,
                 endtime, data=None):
        """
        Run the full pipeline for one time window. Main entry point.

        Fetches and restitutes the waveforms, computes STFTs over sliding
        windows, runs EQS twice (detection, then a re-aligned refinement),
        optionally refines accepted detections with EQShyb, assembles and trims
        the denoised snippets, optionally picks phases with TTA, and writes
        MiniSEED plus pick files to disk. Each step is annotated inline with its
        inputs and outputs.

        Returns early, having written nothing, when the data cannot be fetched,
        when no detection survives selection, or when none survives the proximity
        filter.

        network   : str, FDSN network code
        station   : str, FDSN station code
        location  : str, FDSN location code (wildcep detectionsards accepted)
        channel   : str, 2-char channel prefix, e.g. "HH" or "HG" (the component
                    wildcard is appended internally)
        starttime : obspy.UTCDateTime, start of the processing window
        endtime   : obspy.UTCDateTime, requested end; rounded up internally to
                    the next exact multiple of 61.2 s
        data      : obspy.Stream or None, pre-fetched raw waveforms including the
                    self.buffer margin; fetched via self.data_client if None

        Returns : None — everything is written to <model parent>/DOY<julday>/
        """

        logger.debug("")

        # snap the window to whole model windows
        # IN:  starttime (UTCDateTime, start of processing window),
        #      endtime (UTCDateTime, requested end of processing window)
        # OUT: endtime (UTCDateTime, adjusted so that endtime - starttime is an
        #               exact multiple of 61.2 s, rounded up — the last window
        #               may therefore extend past the requested end)
        endtime = self._round_to_window(starttime, endtime)


        # fetch + restitute
        # IN:  network, station, location, channel, starttime, endtime,
        #      data (optional raw Stream, must already include the buffer)
        # OUT: data (Stream, 3 components, restituted, 100 Hz, buffer trimmed),
        #      data_stack (np.ndarray, (N, 3), columns in self.components order
        #                  (Z first), restituted velocity in m/s),
        #      gap_intervals (list of (UTCDateTime, UTCDateTime), recorded before
        #                     the merge zero-filled them — empty list if no gaps)
        #      None when fewer than 3 traces survive the merge
        result = self._get_data(network, station, location, channel,
                                          starttime, endtime, data)
        if result is None:
            return None
        data, data_stack, gap_intervals = result

        # compute stfts
        # IN:  data_stack (N, 3), data[0].stats.starttime (UTCDateTime,
        #      time of data_stack[0]), endtime (UTCDateTime, unused)
        # OUT: selected_starttimes (list of UTCDateTime, W, window start times,
        #                           spaced self.shift_samples / fs apart),
        #      stft_collection (np.ndarray, (W, 64, 256, 6) float32, raw STFT),
        #      stft_norm_collection (np.ndarray, (W, 64, 256, 6) float32, normalised)
        #      W = 0 if the record is shorter than one 61.2 s window
        selected_starttimes, stft_collection, stft_norm_collection =  \
            self._compute_global_stfts(data_stack, data[0].stats.starttime, endtime)

        # make prediction — EQS first pass + peak detection
        # IN:  stft_norm_collection (W, 64, 256, 6)
        # OUT: filtered_results (np.ndarray, (D, 5), columns
        #                        [peak, start, end, score, maxval]; columns 0-2 are
        #                        BIN INDICES (0.24 s per bin), not seconds.
        #                        Even-derived rows first, then unmatched odd rows,
        #                        so the rows are NOT in time order),
        #      origin (list of int, D, 0 = even stream, 1 = odd stream),
        #      y_predict (np.ndarray, (W, 64, 256, 3), EQS masks, all windows)
        filtered_results, origin, y_predict = \
            self._detect_event_signals(stft_norm_collection)


        # make selection — map each detection onto its STFT window
        # IN:  filtered_results (D, 5), origin (D,), y_predict (W, 64, 256, 3),
        #      stft_collection (W, 64, 256, 6), selected_starttimes (list, W)
        # OUT: selected_masks (np.ndarray, (K, 64, 256, 3), EQS mask per detection),
        #      selected_stft (np.ndarray, (K, 64, 256, 6), raw STFT per detection),
        #      selected_utc (list of UTCDateTime, K, window start per detection),
        #      detection_start (list of UTCDateTime, K, estimated signal start),
        #      filtered_results (np.ndarray, (K, 5), ONLY the rows that survived;
        #                        detections whose window index ran past the end of
        #                        y_predict are dropped here, so the original array
        #                        must not be used from this point on. K <= D)
        selected_masks, selected_stft, selected_utc, detection_start, \
            filtered_results = \
            self._select_data_and_mask(filtered_results, origin, y_predict,
                                       stft_collection, selected_starttimes)

        if len(detection_start) == 0:
            logger.info("No detections found — skipping refinement and output")
            return None


        # free the full-record arrays: only the per-detection subsets are needed
        # from here on
        del stft_collection, stft_norm_collection, y_predict

        # recompute mask — re-cut each window so the onset sits 10.08 s in
        # IN:  detection_start (list of UTCDateTime, K),
        #      data[0].stats.starttime (UTCDateTime, reference for sample indexing),
        #      data_stack (N, 3)
        # OUT: stft_collection_subset (np.ndarray, (K, 64, 256, 6), re-aligned raw STFT),
        #      stft_norm_collection_subset (np.ndarray, (K, 64, 256, 6), re-aligned norm),
        #      stream_start_end (list of (UTCDateTime, UTCDateTime) or None, K;
        #                        None where the re-aligned window would fall outside
        #                        data_stack — those rows stay zero and the original
        #                        window is kept by _make_final_selection)
        stft_collection_subset, stft_norm_collection_subset, \
            stream_start_end = \
            self._recompute_mask(detection_start,
                                 data[0].stats.starttime,
                                 data_stack)

        # Make new prediction — EQS second pass on the re-aligned windows
        # IN:  stft_norm_collection_subset (K, 64, 256, 6)
        # OUT: y_predict_event (np.ndarray, (K, 64, 256, 3), EQS masks)
        y_predict_event = self.model.predict(stft_norm_collection_subset,
                                             verbose=0)

        # window selection — choose original vs re-aligned window, apply threshold
        # IN:  y_predict_event (K, 64, 256, 3), filtered_results (K, 5),
        #      detection_start (list, K), selected_stft (K, 64, 256, 6),
        #      selected_masks (K, 64, 256, 3), selected_utc (list, K),
        #      stft_collection_subset (K, 64, 256, 6),
        #      stream_start_end (list of (UTCDateTime, UTCDateTime) or None, K)
        #      all of these must stay index-aligned — guaranteed by the
        #      filtered_results returned above
        # OUT: stft_final_subset (np.ndarray, (A, 64, 256, 6), raw STFT accepted),
        #      masks_subset (np.ndarray, (A, 64, 256, 3), EQS mask accepted),
        #      utc_start_subset (list of UTCDateTime, A, window start accepted),
        #      stream_start_end_final (list of (UTCDateTime, UTCDateTime), A,
        #                              signal start/end per accepted detection;
        #                              bin counts converted with self.bin_spacing)
        #      scores_final (list of float, A, detection score per accepted detection)
        #      A = number of accepted detections (<= K)
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
        # OUT: denoised_hyb (np.ndarray, (A, 6120, 3), EQShyb denoised waveforms
        #      in physical units), or None if eqs2_model is None or A == 0
        denoised_hyb = None
        if self.eqs2_model is not None and stft_final_subset.shape[0] > 0:
            denoised_hyb = self._apply_eqshyb(stft_final_subset, masks_subset,
                                              utc_start_subset, data_stack,
                                              data[0].stats.starttime)

        # free the sample-domain array: _recompute_mask() and _apply_eqshyb()
        # were its only consumers (~200 MB for a day at 100 Hz)
        del data_stack

        # assemble streams — ISTFT (EQS) or EQShyb waveforms
        # IN:  stft_final_subset (A, 64, 256, 6), masks_subset (A, 64, 256, 3),
        #      utc_start_subset (list, A), stream_start_end_final (list, A),
        #      data (Stream, source of the trace headers),
        #      denoised_hyb (A, 6120, 3) or None — if None the EQS ISTFT path is used
        # OUT: trimmed_streams (obspy.Stream, 3*A traces, Z/N/E contiguous per
        #                       detection, groups sorted by SIGNAL start),
        #      stream_start_end_final (list, A, sorted the same way)
        trimmed_streams, stream_start_end_final = \
            self._build_streams(stft_final_subset, masks_subset,
                                utc_start_subset, stream_start_end_final,
                                data, denoised_hyb)


        # resolve overlaps between consecutive snippets, apply signal_buffer_s
        # IN:  trimmed_streams (obspy.Stream), stream_start_end_final (list, A)
        # OUT: trimmed_streams (obspy.Stream, snippets cut so that neighbouring
        #                       detections no longer overlap)
        trimmed_streams, stream_start_end_final  = \
            self._filter_close_detections_streams(trimmed_streams, stream_start_end_final, scores_final)

        if len(stream_start_end_final) == 0:
            logger.info("No detections remaining after proximity filter")
            return None

        # IN:  trimmed_streams (obspy.Stream), stream_start_end_final (list, A)
        # OUT: trimmed_streams (obspy.Stream, overlap-trimmed)
        trimmed_streams = self._trim_streams(trimmed_streams, stream_start_end_final)  # NEW

        # phase picking — optional, only if picker configured and snippets exist
        # IN:  trimmed_streams (obspy.Stream, denoised snippets),
        #      data (obspy.Stream, original restituted, in memory — not re-fetched;
        #            used to build the designaled noise for the TTA augmentation)
        # OUT: picks (dict, keys 'p_picks'/'s_picks', each a list of Pick objects),
        #      and pick files written to disk as JSON and/or SC3ML per
        #      self.pick_output, in the same DOY folder as the MiniSEED
        picks = None
        if self.picker is not None and len(trimmed_streams):
            picks = self._pick(trimmed_streams, data)
            self._save_picks(picks, data[0].stats.starttime, data[0].id[:-1])

        # filter by pick — optional, only if filter_by_pick enabled and picks exist
        # IN:  trimmed_streams (obspy.Stream, overlap-trimmed denoised traces),
        #      picks (dict, or None if the picker is not configured)
        # OUT: streams_to_save (obspy.Stream, subset of trimmed_streams whose time
        #      window contains at least one accepted P or S pick; equals
        #      trimmed_streams unchanged when filter_by_pick is False or picks is None)
        streams_to_save = (
            self._filter_streams_by_picks(trimmed_streams, picks)
            if self.filter_by_pick and picks is not None
            else trimmed_streams)

        # output - save to MSEED
        # IN:  data[0].stats.starttime (UTCDateTime, gives the DOY folder name —
        #                               the same value is passed to _save_picks so
        #                               both land together),
        #      streams_to_save (obspy.Stream, the traces to write),
        #      gap_intervals (list of (UTCDateTime, UTCDateTime), zeroed in the
        #                     output because the model produces signal there from
        #                     zero-filled input),
        #      data (obspy.Stream, restituted input; written as <id>_raw.mseed
        #            only when self.save_raw is True)
        # OUT: writes <id>_denoised.mseed (and optionally <id>_raw.mseed) to
        #      <model parent>/DOY<julday>/. Note that Stream._cleanup() merges
        #      snippets that end up exactly contiguous, so the number of traces
        #      in the file is not necessarily the number of detections.
        self._output(data[0].stats.starttime, streams_to_save, gap_intervals)
# %%
# %%
if __name__ == "__main__":

    # %% With picker + polarity
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    import sys
    sys.path.append("/home/niko/Earthquake-Seismogram-Denoiser/Code")
    from Denoiser_EQShyb import Denoiser
    import time
    import seisbench.models as sbm
    # from DenoisingFunctions import client_sed
    picker = sbm.EQTransformer.from_pretrained("ethz")


    # clients
    # data_client     = client_sed
    # metadata_client = client_sed
    data_client = Client("ETH")      # or "IRIS", "GFZ", SDS client, etc.
    metadata_client = Client("ETH")      # fdsn client for response

    denoiser = Denoiser(
        data_client     = data_client,
        metadata_client = metadata_client,
        model_path      = "/home/niko/Earthquake-Seismogram-Denoiser/Models/model_1000k_onlyweights.keras",
        min_peak_height = 0.33,
        eqs2_model_path = "/home/niko/Earthquake-Seismogram-Denoiser/Models/EQS2.keras",  # optional, omit for EQS only
        picker=picker,  # optional, omit to skip picking
        picking_kwargs={
            "repeat": 20,  # should be < 100; 2 digits max.
            "pick_tolerance": 1,
            "p_confidence": 0.5,
            "s_confidence": 0.5,
            "min_share_models": 0.25,
            "max_workers": 1 # leave at 1
        },
        # polarity_model_path="/home/niko/Schreibtisch/Polarity/Model/polarity_cnn_mixeddata_globalmaxavg_dropout02.keras",
        polarity_model_path="/home/niko/Schreibtisch/EQ_denoising/NextGen/polarity_paper.keras", # SAME
        polarity_kwargs={"threshold": 0.33, "mc_dropout": True},
        debug=True
    )

    start_full = time.perf_counter()
    denoiser.save_raw = True
    # ── single window ─────────────────────────────────────────────────────────
    denoiser.run_data(
        network   = "CH",
        station   = "MFERR",
        location  = "*",
        channel   = "HH",
        starttime = UTCDateTime("2025-02-07T19:00:00"),
        endtime   = UTCDateTime("2025-02-08T19:00:00")
    )

    elapsed_full = time.perf_counter() - start_full
    print(elapsed_full)
