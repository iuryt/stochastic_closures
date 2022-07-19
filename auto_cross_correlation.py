import h5py
import xarray as xr
import numpy as np
from tqdm import tqdm
from correlation import correlate
from xhistogram.xarray import histogram
from scipy import signal
from cmcrameri import cm
import matplotlib.pyplot as plt

njobs = 100

from dask_jobqueue import SLURMCluster
cluster = SLURMCluster(cores=1,memory="5GB")
cluster.scale(jobs=njobs)    # Deploy ten single-node jobs

import dask
import dask.array as da
from dask.distributed import Client,progress
from dask.diagnostics import ProgressBar
client = Client(cluster)  # Connect this local process to remote workers


from glob import glob

fnames = glob("data/omega_run*")
fnames.sort()


data = {noise:xr.open_dataset(fnames[i]).ωk for i,noise in enumerate([0,1e-6,1e-8,1e-10,1e-12])}


@dask.delayed
def compute_C(kx_slice,noise1,noise2,kbins):
    
    # allocate in memory the autocorrelation matrix C(k,lag)
    C = np.zeros((kbins.size-1,data[0].time.size))

    ωk1 = data[noise1].sel(kx=kx_slice).load()
    ωk2 = data[noise2].sel(kx=kx_slice).load()

    kxs = ωk1.kx.values
    kys = ωk1.ky.values

    ωk1 = ωk1.values
    ωk2 = ωk2.values
    for i,kx in enumerate(kxs):
        for j,ky in enumerate(kys):
            # %%timeit
            ωkij1 = ωk1[:,i,j]
            ωkij2 = ωk2[:,i,j]

            # calculate the autocorrelation using fft method
            lags,cij = correlate(ωkij1,ωkij2)

            # find the bin indexes
            k = np.abs(kx + 1j*ky)

            ind = np.argwhere((k>=kbins)[:-1]&(k<kbins)[1:])[0][0]

            C[ind] = C[ind]+cij

    return C


noise1 = 0
noise2 = 1e-12

kbins = np.arange(np.ceil(np.abs(data[0].kx + 1j*data[0].ky).max().values)+2)[::2]
kx_slices = np.array_split(data[0].kx.values,500)

C = dask.compute(*[compute_C(kx_slice,noise1,noise2,kbins) for kx_slice in kx_slices])
C = np.dstack(C).sum(-1)

K = np.abs(data[0].kx + 1j*data[0].ky).rename("k")
H = histogram(K,bins=kbins).rename(k_bin="k")

C = xr.DataArray(
    C,
    dims=("k", "lag"),
    coords=dict(
        lag=("lag", data[0].time.values),
        k=("k", H.k.values),
    )
).T/H

fname = f"{-np.log10(noise2):.0f}"
C.to_netcdf(f"data/C_{fname}.nc")




kw = dict(
    contourf = dict(
        cmap=cm.acton,
        vmin=-0.1, vmax=1,
        levels=np.arange(0, 1+0.1, 0.1),
        add_colorbar=False,
        extend="both",
    ),
    contour = dict(
        levels=[0.5],
        colors=["0.3"],
        linestyle="--"
    )
)
fig,ax = plt.subplots()

h = C.plot.contourf(ax=ax, **kw["contourf"])
fig.colorbar(h, ax=ax, label="correlation")
C.plot.contour(ax=ax, **kw["contour"])

ax.set(
    yscale="log",
    xscale="log",
    ylim=[1e-5,C.lag.max()],
    xlabel="k",
    ylabel="lag"
)

fig.savefig(f"img/{fname}.png",facecolor="w",dpi=200)
