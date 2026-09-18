#
# This class runs the denoiser and the picker close to real time. 
# It will then feed the results into seiscomp for location and further downstream
# processing. 
# 

import RealtimeDenoiser
import obspy.clients.seedlink.easyseedlink
import obspy.realtime.rttrace
import seisbench.models as sbm
import obspy.clients.fdsn
import numpy
import multiprocessing
from Denoiser import SENTINEL


class Client(obspy.clients.seedlink.easyseedlink.EasySeedLinkClient):
    
    def __init__(self, server_url, callback, autoconnect=True):
        super().__init__(server_url, autoconnect)
        self.callback = callback
        
    def on_data(self, trace):
        self.callback(trace)

class DataStream(object):
    
    def __init__(self, trace):
        self.stream = []
        self.id = trace.id[0:-1]
        self.add(trace)
        self.next_process = trace.stats.starttime + 85
        pass
    
    def __iter__(self):
        return iter(self.stream)

    def add(self, trace):
        # add if trace.id[0:-1] matches, otherwise return False
        if trace.id[0:-1] != self.id:
            return False
        for tr in self.stream:
            if tr.id == trace.id:
                if abs(trace.stats.endtime - tr.stats.endtime) > 180 or abs(trace.stats.starttime - tr.stats.starttime) > 180 or trace.stats.starttime < tr.stats.endtime:
                    print(f"{tr.id}: bogus data detected, reinitialising")
                    self.stream = []
                    break
                tr.append(trace)
                return True
        newtrace = obspy.realtime.rttrace.RtTrace(max_length=130)
        newtrace.append(trace)
        self.stream.append(newtrace)
        return True
    
    def print(self):
        for trace in self.stream:
            print(trace)

    def length(self):
        # show maximum length if it holds three traces (all components)
        # i.e. the length that all three components have in common
        # if it holds less, return 0
        if len(self.stream) < 3:
            return 0
        return self.endtime() - self.starttime()
    
    def starttime(self):
        # shows maximum starttime of all three traces or none if there are not three traces
        if len(self.stream) < 3: 
            return None
        return max([trace.stats.starttime for trace in self.stream])
    
    def endtime(self):
         # shows minimum endtime of all three traces or none if there are not three traces
        if len(self.stream) < 3:
            return None
        return min([trace.stats.endtime for trace in self.stream])


def setup_result_processor():
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue(maxsize=256)
    p = ctx.Process(
                target=result_processor,
                args=(queue,),
            )
    p.start()
    print(f"started result processor")
    return queue


def result_processor(queue):
    print("Pick Result processor running")
    with open("/tmp/pick-output.txt", "w") as f:
        while True:
            content = queue.get()
            print("Dequeued: ", content)
            if content is SENTINEL:
                return
            f.write(str(content))
        




class RealtimePicker(object):
    
    def __init__(self, seedlink_address, station_list):
        self.shift_size = 61.2 / 2
        self._setup_picking()
        self.streams = {}
        self.seedlink_client = Client(seedlink_address, self.process_data)

        for chn in station_list:
            network, station, location, channel = chn.split('.')
            locchannel = f"{location}{channel}"
            self.seedlink_client.select_stream(network, station, locchannel)
        self.seedlink_client.run()
    
                
    def process_data(self, trace):
        # general algorithm
        # if less than data_length data is present: do nothing
        # elif data has gaps: empty the buffer
        # elif: if timestamp of last datapoint is <= next_data_point + data_length: do nothing
        # elif: ill formed data, should not be reached
        # else: Process data from next_data_point to next_data_point + data_length
        #       set next_data_point = next_data_point + shift_size
        
                
        streamid = trace.id[0:-1]
        
        if streamid in self.streams:
            self.streams[streamid].add(trace)
        else:
            self.streams[streamid] = DataStream(trace)
        
        datastream = self.streams[streamid]
        if datastream.next_process + 81.2 <= datastream.endtime():
            stream = obspy.core.Stream([trace.slice(datastream.next_process, 
                     datastream.next_process + 81.2, nearest_sample=True) 
                     for trace in datastream])
            self.queue.put(stream)
            print(f"Queue length: {self.queue.qsize()}")
            datastream.next_process += self.shift_size
        
#        else: 
#            print("Not enough new data")
        

    def _setup_picking(self):
        #picker = sbm.EQTransformer.from_pretrained("ethz")
        client = obspy.clients.fdsn.Client("ETH")
        inventory = client.get_stations(network="CH",starttime = obspy.core.UTCDateTime(), level = 'response')
        result_queue = setup_result_processor()
        self.queue = RealtimeDenoiser.setup_consumers(8, "ETH", "ETH", "../Models/model_1000k_onlyweights.keras", 
                                                0.33,  eqs2_model_path = "../Models/EQS2.keras", picker = 'ethz',  
                                                polarity_model_path="../Models/polarity_paper.keras", polarity_kwargs={"threshold": 0.33},
                                                debug = True, inventory = inventory, res_queue = result_queue, picker_kwargs={'pick_output':'sc3ml'})
