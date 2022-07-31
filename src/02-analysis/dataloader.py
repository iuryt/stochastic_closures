from glob import glob
import numpy as np
import h5py
import xarray as xr
from tqdm import tqdm

def load_task_data(path_to_files, task, kmax=500):
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
            
            data.append(file["tasks"][task][:, indx, :][:, :, indy])
            time.append(file["scales"]["sim_time"][:])
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
