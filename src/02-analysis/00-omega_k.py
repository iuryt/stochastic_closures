import h5py
import xarray as xr
import numpy as np
from tqdm import tqdm
from correlation import correlate
from scipy import signal

from dask_jobqueue import SLURMCluster
cluster = SLURMCluster(cores=1,memory="20GB")
cluster.scale(jobs=30)    # Deploy ten single-node jobs

import dask
import dask.array as da
from dask.distributed import Client,progress
client = Client(cluster)  # Connect this local process to remote workers

@dask.delayed
def rfft2(s,run_name="run1d2"):
    fname = f"../../../dns_runs/{run_name}/snapshots/snapshots_s{s}.h5"
    data = h5py.File(fname,"r")
    ωk = np.fft.rfft2(data["tasks"]["ω"])
    return ωk,data["scales"]["sim_time"][:]

for run_name in ["run1d2"]+[f"run{i}" for i in range(2,5+1)]:
    print(run_name)
    R = dask.compute(*[rfft2(s,run_name=run_name) for s in np.arange(1,30+1)])
    ωk = np.vstack([Ri[0] for Ri in R])
    time = np.hstack([Ri[1] for Ri in R])
    time = time-time[0]

    l,m,n = ωk.shape 
    kx,ky = np.arange(m),np.arange(n)

    ωk = xr.Dataset(
        dict(
            ωk = (("time", "kx", "ky"), ωk),
        ),
        coords=dict(
            time=("time", time),
            kx=("kx", kx),
            ky=("ky", ky),
        )
    )

    print("saving file ...")
    ωk.to_netcdf(f"../../data/omega_{run_name}.nc", engine="h5netcdf", invalid_netcdf=True)
