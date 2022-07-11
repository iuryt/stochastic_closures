import h5py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ωh = []
for s in np.arange(30,37+1):
    fname = f"snapshots/snapshots_s{s}.h5"
    print(fname)
    data = h5py.File(fname,"r")
    for ω in tqdm(data["tasks"]["ω"]):
        l,m = ω.shape
        ωhi = np.fft.fft2(ω)[np.newaxis,:l//2,:m//2]
        ωh.append(ωhi)
ωh = np.vstack(ωh)
kx,ky = np.meshgrid(*(2*[np.fft.fftfreq(l)[:l//2]]))
K = np.abs(kx+1j*ky)

def autocorrelation(A,n_lags=20):
    nt,ny,nx = A.shape
    Ci = np.zeros((n_lags,ny,nx))
    for lag in tqdm(range(n_lags)):
        Ci[lag,:] = (1 / (nt-lag+1)) * (A[:nt-lag] * np.conj(A[lag:nt])).sum(0)
    return Ci

n_lags = 30
C = autocorrelation(ωh,n_lags=n_lags)
C = C/C[0,:,:]



c = []
k = []

nk = 10+1
ks = kx[0,:][:nk]
for kl,kh in tqdm(zip(ks[:-1],ks[1:]), total=len(ks[:-1])):
    N = ((K>=kl)&(K<kh))[np.newaxis,:,:]*np.ones(C.shape)
    N[N==0] = np.nan
    
    c.append(np.nanmean(np.nanmean(C*N,1),1))
    k.append(0.5*(kl+kh))
c = np.vstack(c)
k = np.array(k)
lag = np.arange(n_lags)

c = xr.Dataset(
    data_vars=dict(
        C_real=(("k","lag"),c.real),
        C_imag=(("k","lag"),c.imag)
    ),
    coords=dict(
        k=("k",k),
        lag=("lag",lag)
    )
)
c.to_netcdf("autocorrelation.nc")


# fig,ax = plt.subplots()
# C = ax.contourf(lag,k,np.abs(c))
# fig.colorbar(C)