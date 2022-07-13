import h5py
import xarray as xr
from xhistogram.xarray import histogram
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from cmcrameri import cm
from tqdm import tqdm
from scipy.signal import correlate,correlation_lags

ωh = []
for s in tqdm(np.arange(1,20+1)):
    fname = f"../dns_runs/run1d2/snapshots/snapshots_s{s}.h5"
    data = h5py.File(fname,"r")
    ωh.append(np.fft.rfft2(data["tasks"]["ω"]))
ωh = np.vstack(ωh)

l,m,n = ωh.shape 
ky,kx = np.arange(m),np.arange(n)
timestep = data["scales"]["timestep"][0]



lags = correlation_lags(l,l)
kbins = ky[::2]

# allocate in memory the vector of number of points in each bin
N = np.zeros(kbins.size-1)
# allocate in memory the autocorrelation matrix C(k,lag)
C = np.zeros((kbins.size-1,len(lags)))

# run loop over ky and kx
for i,kyi in enumerate(tqdm(ky)):
    for j,kxj in enumerate(kx):
        # get the time series for the fourier coeff
        ωhij = ωh[:,i,j]
        # calculate the autocorrelation using fft method
        cij = correlate(ωhij,ωhij,method="fft").real # select the real part
        cij = cij/cij[len(lags)//2] # scale by the zero-lagged correlation
        
        # find the bin indexes
        kij = np.abs(kxj + 1j*kyi)
        
        try:
            ind = np.argwhere((kij>=kbins)[:-1]&(kij<kbins)[1:])[0][0]

            C[ind] = C[ind]+cij
            N[ind] = N[ind]+1
        except:
            pass
C = C/N[:,np.newaxis]
 

C = xr.DataArray(
    C,
    dims=("k", "lag"),
    coords=dict(
        lag=("lag", lags*timestep),
        k=("k", 0.5*(kbins[:-1]+kbins[1:])),
    )
).sel(lag=slice(0,None)).T
C = C/C.sel(lag=0)





kw = dict(
    cmap=cm.acton,
    vmin=-0.1, vmax=1,
    levels=np.arange(0, 1+0.1, 0.1),
    add_colorbar=False,
)

fig, ax = plt.subplots(figsize=(7,4))

h = C.plot.contourf(ax=ax, **kw)
fig.colorbar(h, ax=ax, label="correlation")
C.plot.contour(levels=[0.5], ax=ax, colors=["0.3"])

# txt = ax.text(0.5e1, 2.5e-3, "0.5")
# txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='w'),
#                        path_effects.Normal()])

ax.set(
    xscale="log",
    yscale="log",
    ylim=[1e-5,5e-3],
)

ax.grid(True, linestyle="--")

fig.savefig("img/C.png", dpi=300, facecolor="w")