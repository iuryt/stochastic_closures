import xarray as xr
import numpy as np
from statistics import compute_error_kbins, compute_correlation_kbins
import dataloader
import params

# %%
# Path to the data
path = lambda run: f"../../../dns_runs/run1/noise_test/{run}/snapshots/*.h5"

# %%
# Autocorrelation and cross-correlation

runa = "noise00"
da = dataloader.load_task_data(path(runa), params.task, params.kmax)

for runb in ["noise00", "noise04a", "noise08a", "noise12a"]:

    print(f"{runa}_{runb}")
    db = dataloader.load_task_data(path(runb), params.task, params.kmax)
    print("Done loading data")
    E = compute_error_kbins(da, db)
    print("Done computing error")
    C = compute_correlation_kbins(da, db)
    print("Done computing correlations")    
    xr.merge([C,E]).to_netcdf(f"../../data/{runa}_{runb}.nc")
    print("Done saving netCDF file")
    
# %%
# Cross-correlation between same noise amplitude

for run in ["noise04", "noise08", "noise12"]:
    
    runa = f"{run}a"
    runb = f"{run}b"
    
    print(f"{runa}_{runb}")
    da = dataloader.load_task_data(path(runa), params.task, params.kmax)
    db = dataloader.load_task_data(path(runb), params.task, params.kmax)
    print("Done loading data")
    E = compute_error_kbins(da, db)
    print("Done computing error")
    C = compute_correlation_kbins(da, db)
    print("Done computing correlations")
    xr.merge([C,E]).to_netcdf(f"../../data/{runa}_{runb}.nc")
    print("Done saving netCDF file")
