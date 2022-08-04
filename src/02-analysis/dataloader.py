from glob import glob
import numpy as np
import h5py
import xarray as xr
from tqdm import tqdm
import os
from misc import format_string

def load_task_data(path_to_files, task, kmax=500, step_task=2):
    # List of files
    fnames = glob(path_to_files)
    # Sort the files
    ind = np.argsort([int(fname.split("s")[-1].split(".")[0]) for fname in fnames])
    fnames = np.array(fnames)[ind]

    data = []
    time = []
    for fname in tqdm(fnames):
        with h5py.File(fname,"r") as file:
            kx = file["scales"]["kx"][:]
            ky = file["scales"]["ky"][:]
            
            indx = np.abs(kx)<kmax
            indy = np.abs(ky)<kmax
            
            data.append(file["tasks"][task][:, indx, :][:, :, indy][::step_task])
            time.append(file["scales"]["sim_time"][::step_task])
            kx, ky = kx[indx], ky[indy]

    data = np.vstack(data)
    time = np.hstack(time)

    coords = dict(
        time = ("time", time),
        ky = ("ky", ky),
        kx = ("kx", kx)
    )
    da = xr.DataArray(data, dims=("time", "kx", "ky"), coords=coords)
    da = da.sortby("kx").sortby("ky")
    return da

def load_save_netcdf(fname, run, params, print_file, invalid_netcdf=True):

    print(f"\n\n{run}", file = print_file)
    da = load_task_data(fname, params.task, params.kmax, params.step_task)
    print("Done loading data", file = print_file)

    param_name = format_string(params.task)
    if invalid_netcdf:
        ds = xr.Dataset({f"{param_name}":da})
        ds.to_netcdf(os.path.join(params.data_dir, run + ".nc"), engine="h5netcdf", invalid_netcdf=True)
    else:
        ds = xr.Dataset({f"{param_name}_real":da.real, f"{param_name}_imag":da.imag})
        ds.to_netcdf(os.path.join(params.data_dir, run + ".nc"))
    print("Done saving netCDF file", file = print_file)
    

def load_complex_netcdf(fname, params):
    ds = xr.open_dataset(fname).load()
    param_name = format_string(params.task)
    return ds[f"{param_name}_real"]+1j*ds[f"{param_name}_imag"]