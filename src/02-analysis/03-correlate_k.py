import h5py
import xarray as xr
import numpy as np
from tqdm import tqdm
from correlation import correlate
from xhistogram.xarray import histogram
from scipy import signal
from cmcrameri import cm
import matplotlib.pyplot as plt
from glob import glob


njobs = 100

from dask_jobqueue import SLURMCluster
cluster = SLURMCluster(cores=1,memory="5GB")
cluster.scale(jobs=njobs)    # Deploy ten single-node jobs

import dask
import dask.array as da
from dask.distributed import Client,progress
from dask.diagnostics import ProgressBar
client = Client(cluster)  # Connect this local process to remote workers


fnames = glob("../../data/omega_run*")
fnames.sort()


ωk = (
    xr.open_mfdataset(
        fnames,combine="nested",
        concat_dim="noise",
        chunks=dict(time=50,kx=1000,ky=1000)
    ).assign_coords(noise=[0,1e-6,1e-8,1e-10,1e-12]).ωk
)


kbins = np.arange(np.ceil(np.abs(ωk.kx + 1j*ωk.ky).max().values)+2)[::2]

K = (np.abs(ωk.kx + 1j*ωk.ky)*xr.ones_like(ωk.time)).rename("k")
H = histogram(K,bins=kbins,dim=("kx","ky"))

weights = np.abs(ωk.sel(noise=1e-12)-ωk.sel(noise=0))
C = histogram(K,bins=kbins,weights=weights,dim=("kx","ky")).load()
C = C/H





fig, ax = plt.subplots()

np.log10(C).plot(ax=ax, vmin=3, vmax=8, cmap=cm.acton)

ax.grid(True, linestyle="--", alpha=0.7, color="0.1")
ax.set(
    xscale="log",
    ylim=[0,C.time.max()],
    xlim=[1,C.k_bin.max()],
    xlabel="k",
    ylabel="time",
    xticks=[1,10,100,1000]
)