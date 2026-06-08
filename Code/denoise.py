import Denoiser
import obspy.clients.fdsn
import obspy.core
import seisbench.models as sbm
picker = sbm.EQTransformer.from_pretrained("ethz")


client = obspy.clients.fdsn.Client("ETH")
print(picker)
d = Denoiser.Denoiser(client, client, "../Models/model_1000k_onlyweights.keras", 0.33, debug = True, picker = picker)
d.run_data('CH','MFERR','','HH', obspy.core.UTCDateTime(2025,2,7,17), obspy.core.UTCDateTime(2025,2,7,20,0,0))