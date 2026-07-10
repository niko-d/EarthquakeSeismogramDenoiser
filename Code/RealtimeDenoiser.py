import Denoiser
import threading
import queue
import numpy as np

class RealtimeDenoiser(Denoiser.Denoiser):
    def __init__(self, no_of_threads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.realtime_queue = queue.Queue(maxsize=256)
        self.realtime_mode = True
        self._workers = []
        self.buffer = 10
        
        for i in range(no_of_threads):
            t = threading.Thread(
                args=(self.realtime_queue,),
                target=self._consumer_thread,
                name=f"consumer-{i}",
                daemon=True
                )
            t.start()
            self._workers.append(t)
        pass
    
    def push(self, stream):
        self.realtime_queue.put(stream)
    
    def _realtime_consumer(self):
        while True:
            content = self.realtime_queue.get()
            if content is SENTINEL:
                return
            self.run_realtime(stream)
            
    def _detect_event_signals_realtime(self, stft_norm_collection):
        """
        Detect event signals using detection algorithm
        """

#        logger.debug("")
        model_verbose = 0

        y_predict = self.model.predict(stft_norm_collection.astype(np.float32),
                                       verbose=model_verbose)
        mask_timeseries_even, mask_timeseries_odd = \
            self.get_mask_timeseries(y_predict)
        # get peaks with start and end, with fixed min. threshold
        # of 0.1 for max of time series (=at leats one bin with mask value>0.1)

        # account for 50% time shift
        peak_info_even = self.get_peaks(mask_timeseries_even,
                                        threshold=self.min_peak_height,
                                        shift_correction=128)
#        peak_info_odd = self.get_peaks(mask_timeseries_odd,
#                                       threshold=self.min_peak_height,
#                                       shift_correction=0)
#        filtered_results, origin = \
#            self.compare_arrays_time_overlap(peak_info_even, peak_info_odd)
 
         # no peak_info_odd in realtime
            
#        logger.info(f"_detect_event_signals: {len(filtered_results)} detections found "
#                    f"(even peaks: {len(peak_info_even)}, odd peaks: {len(peak_info_odd)})")
#        if len(filtered_results):
#            logger.info(f"  peak index range: {filtered_results[:, 0].min():.0f} — {filtered_results[:, 0].max():.0f} "
#                        f"(of {len(mask_timeseries_even) + len(mask_timeseries_odd)} total bins; "
#                        f"bin spacing {self.bin_spacing}s → "
#                        f"last detection ~{filtered_results[:, 0].max() * self.bin_spacing / 3600:.1f}h into data)")

        return (peak_info_even, y_predict)

    def run_realtime(self, stream):
        # Realtime processing, similar to offline processing with a few exceptions:
        # - "buffer" at the start and end of the data is smaller (10s by default)
        # 
        # Parameters:
        # stream: obspy.core.Stream object, containing three components of the same station
        # the assumption is, that the first and the last 10s of the data are cut away after 
        # pre-filtering, the first 61.2s of the remainder are then fed to the algorithm
        #
        # Current implementation (work ongoing):
        # 1. download 81.2s of data (61.2 with 10s buffer on each side)
        # 2. shape it (taper (to be checked), pre-filtering, downsampling, deconvolution)
        # 3. Run detection
        # 4. Store results in a queue for further processing (not yet ready)
         
        # do not run on non-sane data
        data = self.check_sanity(stream)
        if not data:
            return False
        
        data, data_stack = self._shape_data(data)
        if not data:
            return False
        
        # compute stfts
        # IN:  data_stack (N, 3), starttime (UTCDateTime), endtime (UTCDateTime)
        # OUT: selected_starttimes (list of UTCDateTime, one per valid window),
        #      stft_collection (np.ndarray, shape (W, 64, 256, 6), raw STFT all windows),
        #      stft_norm_collection (np.ndarray, shape (W, 64, 256, 6), normalised STFT)
        
        # check whether this works with just one window
        selected_starttimes, stft_collection, stft_norm_collection =  \
            self._compute_stfts(data_stack, data[0].stats.starttime, data[0].stats.endtime)
            
        # make prediction
        # IN:  stft_norm_collection (W, 64, 256, 6)
        # OUT: filtered_results (np.ndarray, (D, 5), peak/start/end/score/maxval per detection),
        #      origin (list of int, 0=even 1=odd stream for each detection),
        #      y_predict (np.ndarray, (W, 64, 256, 3), EQS mask predictions all windows)
        peak_info, y_predict = \
            self._detect_event_signals_realtime(stft_norm_collection)
        print(stream[0].id, peak_info, y_predict)

    def check_sanity(self, data):   
        # check whether we have three components of data for the same time window
        # also check whether there are gaps, overlaps etc.
        
        # 1. check that we don't have any gaps
        if len(data.get_gaps()) > 0:
            print("Data contains gaps")
            return False
        
        # 2. check that we have exactly three different streams of the same station
        if len(data) != 3: 
            print("data doesn't contain three traces")
            return False
        nsl = set([])
        channel = set([])
        for trace in data:
            network, station, location, _channel = trace.id.split('.')
            nsl.add(f"network.station.location")
            channel.add(_channel)
        if len(nsl) != 1 or len(channel) != 3:
            print("data not three channels of one station")
            return False
        
        # 3. check whether stream contains at least 81.2s of data on all
        starttime = max(trace.stats.starttime for trace in data)
        endtime = min(trace.stats.endtime for trace in data)
        if endtime - starttime < 81.2:
            print("stream doesn't contain at least 81.2s of data")
            return False
        
        # if all tests pass, this is valid data
        endtime = starttime + 81.2
        data.trim(starttime, endtime, nearest_sample=True)
        return data
     
    def _shape_data(self, stream):
        stats = stream[0].stats
        starttime = stats.starttime
        endtime = stats.endtime
        network, station, location, channel = stats.network, stats.station, stats.location, stats.channel
        metadata = self._get_metadata(network, station, location, f"{channel[0:2]}*", starttime, starttime)
        if not metadata:
            print(f"No metadata found for {stream[0].id}") 
            return (False, False)
        data = self.apply_pre_filt_stream(stream, taper_seconds = 10)
        
        if data[0].stats.sampling_rate % 100 == 0:
            data.decimate(factor=int(data[0].stats.sampling_rate // 100),no_filter=True)
        else:
            # data.filter("lowpass", freq=45.0, corners=8, zerophase=False)  # NEW - add filter ???
            data.resample(100,no_filter=True) # no filter default, additioonal AA off by frequency taper
        
        self._fast_remove_response(data, metadata)
        data.trim(data[0].stats.starttime + 10,
                  data[0].stats.starttime + 10 + 61.2, nearest_sample=True)

        self.components = sorted([tr.stats.channel[-1] for tr in data], reverse=True)  # NEW get components and fix order in data


        # z comp first, other componets abitrarily
        data_stack = np.column_stack([
            data.select(component=self.components[0])[0].data,
            data.select(component=self.components[1])[0].data,
            data.select(component=self.components[2])[0].data
        ])
        fig = data.plot(show=False)
        fig.savefig(f"/tmp/{data[0].id}-{data[0].stats.starttime}.png")
        return (data, data_stack)