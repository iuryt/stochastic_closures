import params
import sys
import dataloader


print_file = sys.stderr

# %%
# Path to the data
path = lambda run: f"../../../dns_runs/run1/noise_test/{run}/snapshots/*.h5"


run = "no_noise_0"
dataloader.load_save_netcdf(path(run), run, params, print_file)

for run in params.noise_runs:

    dataloader.load_save_netcdf(path(run+"_00"), run+"_00", params, print_file)
    dataloader.load_save_netcdf(path(run+"_01"), run+"_01", params, print_file)