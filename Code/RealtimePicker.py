#
# This class runs the denoiser and the picker close to real time. 
# It will then feed the results into seiscomp for location and further downstream
# processing. 
# 

import Denoiser
import obspy.clients.seedlink.easyseedlink
import obspy.realtime.rttrace
import seisbench.models as sbm
import obspy.clients.fdsn
import numpy

class Client(obspy.clients.seedlink.easyseedlink.EasySeedLinkClient):
    
    def __init__(self, server_url, callback, autoconnect=True):
        super().__init__(server_url, autoconnect)
        self.callback = callback
        
    def on_data(self, trace):
        self.callback(trace)

class RealtimePicker(object):
    
    def __init__(self, seedlink_address, station_list):
        self.seedlink_client = Client(seedlink_address, self.add_data)
        self.traces = {}
        self.last_processed = {}
        self.data_length = 70
        self.shift_size = 61.2 / 2
        self._setup_picking()
        
        for chn in station_list:
            network, station, location, channel = chn.split('.')
            locchannel = f"{location}{channel}"
            self.seedlink_client.select_stream(network, station, locchannel)
        self.seedlink_client.run()
    
    def add_data(self, trace):
        if trace.id not in self.traces:
            self.traces[trace.id] = obspy.realtime.rttrace.RtTrace(max_length=90)
            
            def make_process(id):
                def closure(data):
                    return self.process_data(data, id)
                return closure
            
            self.traces[trace.id].register_rt_process(make_process(trace.id))
        self.traces[trace.id].append(trace)
                
    def process_data(self, data, id):
        # general algorithm
        # if less than data_length data is present: do nothing
        # elif data has gaps: empty the buffer
        # elif: if timestamp of last datapoint is <= next_data_point + data_length: do nothing
        # elif: ill formed data, should not be reached
        # else: Process data from next_data_point to next_data_point + data_length
        #       set next_data_point = next_data_point + shift_size
        
        trace = self.traces[id]
        
        if not id in self.last_processed:
            self.last_processed[id] = trace.stats.starttime
        
        # do nothing if data is too short
        if trace.stats.endtime - trace.stats.starttime < self.data_length:
            print("Not enough data")
            return data
        
        # discard data if data has gaps
        # Not necessary, the library already does this automatically
        # elif trace.data.is_masked:
        #    del self.traces[trace.id]
        #    
        #    return
        
        # not enough data since last processing, do nothing
        elif self.last_processed[id] -  self.shift_size + self.data_length > trace.stats.endtime:
            print("not enough new data")
            return data
        
        # something wrong
        # hier
#        elif self.last_processed[id] - self.shift_size < trace.stats.starttime:
#            print("something wrong")
#            return data
        
        # nothing applies, process data
        else: 
            print("Processing")
            if self.last_processed[id] - self.shift_size < trace.stats.starttime:
                self.last_processed[id] = trace.stats.starttime
            _data = trace.slice(self.last_processed[id]- self.shift_size,
                                  self.last_processed[id]- self.shift_size + self.data_length)
            self.last_processed[id] += self.shift_size
            self.denoise_picker.push(_data)
        return data
            
    def _setup_picking(self):
        picker = sbm.EQTransformer.from_pretrained("ethz")
        client = obspy.clients.fdsn.Client("ETH")
        self.denoise_picker = Denoiser.Denoiser(client, client, "../Models/model_1000k_onlyweights.keras", 
                                                0.33, debug = True, picker = picker)
        self.denoise_picker.setup_realtime_processing(no_of_threads=4)
