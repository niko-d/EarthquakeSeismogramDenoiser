# Earthquake-Seismogram-Denoiser


## EQS Test Environment Setup

## Setup Instructions (Ubuntu)

Create and activate a virtual environment named `eqs` and install all dependencies listed in `requirements.txt`:

```bash
python3 -m venv eqs_test
source eqs/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Code examples:

Single time window denoising:
Notebook in EQS-Denoiser_example.py

Denoising and signal detection on continuous data

Data download from "ETH" Client, preprocessing, signal detection and denoising for 600s following 2025-02-07T17:00:00. both raw and denoised data are saved
```
python ./DenoiseData_terminal_public.py CH MFERR ?? 2025-02-07T17:00:00 600 --saveraw True --model_name ../Models/model_1000k_onlyweights.keras --client_str ETH
```
