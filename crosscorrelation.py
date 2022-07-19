import h5py
import xarray as xr
import numpy as np
from tqdm import tqdm
from correlation import correlate
from scipy import signal
from cmcrameri import cm

run = "run1d2"
ωk = []
time = []
for s in tqdm(np.arange(1,30+1)):
    fname = f"../dns_runs/{run}/snapshots/snapshots_s{s}.h5"
    data = h5py.File(fname,"r")
    ωk.append(np.fft.rfft2(data["tasks"]["ω"]))
ωk = np.vstack(ωk)


run2 = "run2"
ωk2 = []
time = []
for s in tqdm(np.arange(1,30+1)):
    fname = f"../dns_runs/{run2}/snapshots/snapshots_s{s}.h5"
    data = h5py.File(fname,"r")
    ωk2.append(np.fft.rfft2(data["tasks"]["ω"]))
ωk2 = np.vstack(ωk2)


kx, ky = np.meshgrid(np.arange(ωk.shape[1]),np.arange(ωk.shape[2]))

lags = signal.correlation_lags(ωk.shape[0],ωk.shape[0])
lags = lags[len(lags)//2:]
kbins = np.arange(np.ceil(np.abs(kx + 1j*ky).max())+2)[::2]


# allocate in memory the vector of number of points in each bin
N = np.zeros(kbins.size-1)
# allocate in memory the autocorrelation matrix C(k,lag)
C = np.zeros((kbins.size-1,len(lags)))


for i,kxi in enumerate(tqdm(kx[0])):
    for j,kyj in enumerate(ky.T[0]):
  
        # calculate the autocorrelation using fft method
        lags,cij = correlate(ωk[:,i,j],ωk2[:,i,j],method="scipy")

        # find the bin indexes
        k = np.abs(kxi + 1j*kyj)

        ind = np.argwhere((k>=kbins)[:-1]&(k<kbins)[1:])[0][0]

        C[ind] = C[ind]+cij
        N[ind] = N[ind]+1
    
C = C/N[:,np.newaxis]

K = 0.5*(kbins[:-1]+kbins[1:])
lags = lags*data["scales"]["timestep"][0]


C = xr.DataArray(
    C,
    dims=("k", "lag"),
    coords=dict(
        lag=("lag", lags),
        k=("k", K),
    )
).T

C.to_netcdf(f"data/C_{run2}.nc")




kw = dict(
    contourf = dict(
        vmin=-1, vmax=1,
        levels=np.arange(-1,1+0.2,0.2), 
        cmap=cm.bam_r,
        extend="both",
    ),
    contour = dict(
        levels=[0.5],
        colors="0.3",
        linestyle="--"
    )
)
fig,ax = plt.subplots()

h = ax.contourf(K,lags,C.T,**kw["contourf"])
fig.colorbar(h, ax=ax,label="correlation")
ax.contour(K,lags,C.T,**kw["contour"])

ax.set(
    yscale="log",
    xscale="log",
    ylim=[1e-5,1e-2],
    xlabel="k",
    ylabel="lag"
)

fig.savefig(f"img/{run2}.png",facecolor="w",dpi=200)
