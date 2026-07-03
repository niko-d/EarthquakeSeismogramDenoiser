import Denoiser
import obspy.clients.fdsn
import obspy.core
import seisbench.models as sbm
picker = sbm.EQTransformer.from_pretrained("ethz")


client = obspy.clients.fdsn.Client("ETH")
print(picker)
d = Denoiser.Denoiser(client, client, "../Models/model_1000k_onlyweights.keras", 0.33, debug = True, picker = picker,
                      polarity_model_path="../Models/polarity_paper.keras", polarity_kwargs={"threshold": 0.33},)
d.run_timerange('CH','MFERR','','HH', obspy.core.UTCDateTime(2025,2,7,0), obspy.core.UTCDateTime(2025,2,8,0,0,0))