import Denoiser
import obspy.clients.fdsn
import obspy.core

client = obspy.clients.fdsn.Client("ETH")

d = Denoiser.Denoiser(client, client, "../Models/model_1000k_onlyweights.keras", 10.0, 0.1 )
d.run_data('CH','MFERR','','HH', obspy.core.UTCDateTime(2025,1,1), obspy.core.UTCDateTime(2025,1,2,0,0,0))