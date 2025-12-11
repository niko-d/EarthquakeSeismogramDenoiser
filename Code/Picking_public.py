"""
EQS-Denoiser

Author: Niko Dahmen
Email: nikolaj.dahmen@eaps.ethz.ch
Affiliation: ETH Zurich
Date: 2025-08-11
Version: 1.0
"""

import sys
import glob
import numpy as np
import obspy
from obspy import Stream
from scipy.ndimage import uniform_filter1d
from concurrent.futures import ThreadPoolExecutor, as_completed
import seisbench.models as sbm

# ADJUST PATH IF DIFFERENT
module_dir = "/Material/Code/"
sys.path.append(module_dir)
from DenoisingFunctions_public import process_stream, get_designaled_noise

# %% START HERE: ANALYSIS 3363
import obspy
from natsort import natsorted
from seisbench.util.annotations import PickList


start = "2025-02-07T17:00:00"

# Prepared data, denoised and unfitlered data
# ADJUST PATH IF DIFFERENT
files_denoised = glob.glob("../Material/ExampleData/*denoised.mseed")
files_original = glob.glob("../Material/ExampleData/*original.mseed")

current_picker = sbm.EQTransformer.from_pretrained("ethz")

min_conf_raw, min_conf_denoised = 0.5, 0.5 # model confidence


# Picks on denoised data are saved here
all_results = {'p_picks': [],
               's_picks': [],
               }

for file_denoised,file_original in zip(files_denoised,files_original):  # loop through stations
    # unfiltered data (only pre-processed)
    st_original = obspy.read(file_original)  # alternatively, get raw data in counts from FDSN server
    # merge data to single traces in case more exist, fix time
    st_original.merge(method=1, fill_value=0)
    st_original.trim(obspy.UTCDateTime(start), obspy.UTCDateTime(start) + 86400, fill_value=0, pad=True)

    picks_original_list = current_picker.classify(st_original).picks.select(min_confidence=min_conf_raw)


    # read denoised data
    st_denoised = obspy.read(file_denoised)
    # merge data to single traces in case more exist, fix time
    st_denoised.merge(method=1, fill_value=0)
    st_denoised.trim(obspy.UTCDateTime(start), obspy.UTCDateTime(start) + 86400, fill_value=0, pad=True)
    # get extracted noise, difference denoised and raw data, waveforms must match in start and end time
    st_designaled = get_designaled_noise(st_denoised, st_original)

    # make picks with TTA (test time augmentation) - note picks can vary due to random noise. More robust values are
    # obtained for larger numbers of repeat paramters, or by fixing the random noise that is added
    # faster processing in parallel
    current_results = process_stream(
        st_denoised=obspy.read(file_denoised),  # denoised stream snippets (or not preferred continuous stream)
        st_designaled=st_designaled,  # designaled stream
        model_picking=current_picker,  # phase picking instance
        repeat=20,  # repetition test time augmentation for uncertainty
        pick_tolerance=1,  # pick tolerance in seconds, consider pick +/- this values as variations of same pick
        p_confidence=min_conf_denoised,  # picking confidence P wave
        s_confidence=min_conf_denoised,  # picking confidence S wave
        min_share_models=0.25  # required share of models above threshold to consider pick
    )

    for key, value in current_results.items():  # combine results from multiple stations
        all_results.setdefault(key, []).extend(value if isinstance(value, list) else [value])


st_original.plot()
st_denoised.plot()

print("P&S picks raw data")
print(len(picks_original_list))
print("P&S picks denoised data")
print(len(all_results["p_picks"])+len(all_results["s_picks"]))


# %%

