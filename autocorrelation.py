import h5py
import xarray as xr
from xhistogram.xarray import histogram
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
from tqdm import tqdm

ωh = []
for s in np.arange(1,15+1):
    fname = f"../dns_runs/run1d2/snapshots/snapshots_s{s}.h5"
    print(fname)
    data = h5py.File(fname,"r")
    for ω in tqdm(data["tasks"]["ω"]):
        l,m = ω.shape
        ωhi = np.fft.fft2(ω)[np.newaxis,:l//2,:m//2]
        ωh.append(ωhi)
ωh = np.vstack(ωh)
kx,ky = np.meshgrid(data["scales"]["kx"],data["scales"]["ky"][:l//2])
timestep = data["scales"]["timestep"][0]

def autocorrelation(A,n_lags=20,norm=True):
    nt,ny,nx = A.shape
    
    Ci = np.zeros((n_lags,ny,nx))+0*1j
    for lag in tqdm(range(n_lags)):
        Ci[lag,:] = (1 / (nt-lag+1)) * (A[:nt-lag] * np.conj(A[lag:nt])).sum(0)
        
    if norm:
        Ci = Ci/Ci[0,:,:]
    return Ci

n_lags = 60
C = autocorrelation(ωh,n_lags=n_lags)

C = xr.DataArray(
    C,
    dims=("lag","kx","ky"),
    coords=dict(
        lag=("lag", np.arange(n_lags)*timestep),
        kx=("kx", kx[0]),
        ky=("ky", ky.T[0]),
    )
)


nk = 1000+1
bins = kx[0,:][:nk]

K = np.abs(kx+1j*ky)
K = xr.ones_like(C)*K
K.name = "k"

H = histogram(K, bins=bins, dim=["kx", "ky"])
Ck = (
    histogram(K, bins=bins, weights=C.real, dim=["kx", "ky"])/H+
 1j*histogram(K, bins=bins, weights=C.imag, dim=["kx", "ky"])/H
)
Ck.name = "Correlation"
Ck = Ck.assign_coords(lag=Ck.lag*timestep)





k = Ck.k_bin.values[120:-1:150]
fs = lambda ki,a,b: a + (ki-k.min())*(b-a)/(k.max()-k.min()) 
fig, ax = plt.subplots(1,2,figsize=(10,5))

np.abs(Ck).T.plot.contourf(cmap=cm.acton,ax=ax[0])
for ki in k:
    (
        np.abs(Ck).sel(k_bin=ki).plot
        .line(
            x="lag",ax=ax[1],
            label=f"{ki:.1f}",
            color=f"{fs(ki,0.1,0.7):1f}"
        )
    )
ax[1].legend(title="k")

ax[0].set(
    title="a)"+50*" ",
    yticks=k,
    ylabel="k"
)

ax[1].set(
    title="b)"+60*" ",
    ylabel="",
    xlim=[0,Ck.lag.max()],
    ylim=[0,1]
)
_ = [a.grid(True, linestyle="--") for a in ax]

xticks = np.arange(0,Ck.lag.max(),4e-4)
for a in ax:
    a.set(
        xlabel="time [10$^{-4}$]",
        xticks=xticks,
        xticklabels=[f"{xi*1e4:.0f}" for xi in xticks],
    )

fig.savefig("img/Cxt.png",dpi=300)