# %%

from statistics import compute_statistics
import params
import sys
import dataloader
import os
from misc import format_string
import xarray as xr

print_file = sys.stderr

# %%
# Autocorrelation and cross-correlation

print("Loading data", file = print_file)
runa = "no_noise_0"
da = xr.open_dataset(os.path.join(params.data_dir, runa + ".nc")).load()[format_string(params.task)]

for runb in [runa]+params.noise_runs:
# for runb in params.noise_runs:

    if runb!=runa:
        runb = f"{runb}_00"

    print(f"\n\n{runa}_{runb}", file = print_file)
    db = xr.open_dataset(os.path.join(params.data_dir, runb + ".nc")).load()[format_string(params.task)]
    print("Done loading data", file = print_file)

    ds = compute_statistics(da, db, print_file=print_file).sel(k=slice(0,params.kmax))
    ds.to_netcdf(os.path.join(params.data_dir,f"{runa}_{runb}.nc"))
    print("Done saving netCDF file", file = print_file)

# %%
# Cross-correlation between same noise amplitude

for run in params.noise_runs:
    
    runa = f"{run}_00"
    runb = f"{run}_01"

    print(f"\n\n{runa}_{runb}", file = print_file)
    da = xr.open_dataset(os.path.join(params.data_dir, runa + ".nc")).load()[format_string(params.task)]
    db = xr.open_dataset(os.path.join(params.data_dir, runb + ".nc")).load()[format_string(params.task)]
    print("Done loading data", file = print_file)

    ds = compute_statistics(da, db, print_file=print_file).sel(k=slice(0,params.kmax))
    ds.to_netcdf(os.path.join(params.data_dir,f"{runa}_{runb}.nc"))
    print("Done saving netCDF file", file = print_file)