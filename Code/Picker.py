import obspy
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from obspy import Stream, Trace
from pathlib import Path
import os
import json

def check_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)


class Picker(object):
    
    def __init__(self, logger, picker, model_name, stft_parameters, picking_kwargs=None,
                 polarity_model_path=None, polarity_kwargs=None, ):
        
        # PICKER
        self.picker = picker
        print("Picker set to: ", picker)
        self.picking_kwargs = picking_kwargs or {}
        self.uncertainty_scaling = {
            'p_picks': {'scale_sample': 4*1.904, 'offset_sample': 9.249},
            's_picks': {'scale_sample': 4*2.211, 'offset_sample': 3.600},
        }
        self.stft_parameters = stft_parameters
        
        # POLARITY
        self.polarity_model = tf.keras.models.load_model(
            polarity_model_path,
            custom_objects={"custom>MaxAbsNorm1D": MaxAbsNorm1D},
            compile=False
        ) if polarity_model_path else None
        self.polarity_threshold = (polarity_kwargs or {}).get('threshold', 0.33)
        self.components = None
        self.logger = logger
        self.model_name = model_name
        
    def _weighted_std(self, values, weights):
        """
        Compute the weighted standard deviation of an array.
        Used by _tta_uncertainty() to quantify pick timing spread across TTA reps.

        values  : 1D array-like
        weights : 1D array-like, corresponding weights
        Returns : float, weighted standard deviation
        """

        
        w = weights + 1e-30
        
        wsum = w.sum()
        mean = np.dot(values, w) / wsum
        variance = np.dot(values * values, w) / wsum - mean * mean
        
        return np.sqrt(np.maximum(variance, 0.0))


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
        _argmax = np.array([np.argmax(trace.data) for trace in sliced_traces])
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
            self.logger.warning(f"_get_designaled_noise: missing components {missing} "
                           f"(orig channels: {[tr.stats.channel for tr in _original]}, "
                           f"self.components: {self.components}) — returning empty stream")
            return obspy.core.Stream()

        _noise = obspy.core.Stream()
        for comp in self.components:
            orig_comp = _original.select(component=comp)
            denoised_comp = _denoised_snippets.select(component=comp)

            # guard: exactly 1 trace per component expected (snippet, not continuous)
            if len(orig_comp) != 1 or len(denoised_comp) != 1:
                self.logger.warning(f"_get_designaled_noise: expected 1 trace per component, "
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
                self.logger.warning(f"_get_designaled_noise: length mismatch on "
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


    def _pick(self, trimmed_streams, data_original, components):
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
        # self.logger.debug(f"_pick: data_original has {len(data_original)} traces: "
        #              f"{[tr.stats.channel for tr in data_original]}")

        self.components = components
        
        if not len(trimmed_streams):
            self.logger.info("No denoised snippets — skipping picking")
            return {'p_picks': [], 's_picks': []}

        data_start = data_original[0].stats.starttime
        data_end = data_original[0].stats.endtime

        num_detections = len(trimmed_streams) // 3
        st_designaled_snippets = obspy.core.Stream()

        for i in range(num_detections):
            snippet = trimmed_streams[3 * i: 3 * (i + 1)]
            # self.logger.debug(f"Detection {i}: snippet components = "
            #              f"{[tr.stats.channel for tr in snippet]}, "
            #              f"npts = {[tr.stats.npts for tr in snippet]}")

            tr_ref = snippet.select(component=self.components[0])[0]
            if tr_ref.stats.npts == 0:  # TODO check why this happens
                self.logger.warning(f"Detection {i}: zero-length snippet — skipping")
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
                # self.logger.debug(f"_pick slice: tr={tr.id} "
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

            # self.logger.warning(f"snippet:{snippet[0].stats.starttime} — {snippet[0].stats.endtime}")
            # self.logger.warning(f"original_snippet: {original_snippet[0].stats.starttime} — {original_snippet[0].stats.endtime}")

            noise_snippet = self._get_designaled_noise(snippet, original_snippet)
            if len(noise_snippet) == 0:
                self.logger.warning(f"Detection {i}: _get_designaled_noise failed — "
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
        self.logger.info(f"Picks: {len(picks['p_picks'])} P, "
                    f"{len(picks['s_picks'])} S")
        return picks


    def _save_picks(self, picks, starttime, plot = False, traces = False):
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
                # No polarity so far
                
                if traces and plot:
                    for trace in traces:
                        print(trace.id, eid, trace.stats.starttime, trace.stats.endtime, t)
                        if trace.id == eid and trace.stats.starttime <= t and trace.stats.endtime >= t:
                            print("Match")
                            fig = trace.plot(show=False)
                            ax = fig.axes[0]
                            color = 'g'
                            if 's_' in phase or 'S_' in phase:
                                color='b'
                            ax.axvline(t, color=color, linewidth=2)
                            fig.savefig(f"/tmp/{trace.id}-{str(t)}.png")
                            break

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
        self.logger.info(f"Picks written to {out_path}")
