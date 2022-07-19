import h5py
import xarray as xr
from xhistogram.xarray import histogram
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from tqdm import tqdm
from scipy import signal


def correlate(x1,x2=None,method="numpy",normalize=True):
    
    if x2 is None:
        x2 = x1
    
    n = len(x1)
    
    lags = signal.correlation_lags(len(x1),len(x2))

    if method=="scipy":
        c = (1/(n-lags+1))*signal.correlate(x1,x2,method="auto")
        c = c[len(lags)//2:]
    elif method=="numpy":
        c = []
        for lag in np.arange(lags.max()+1):
            c.append((x1[:n-lag]*np.conj(x2[lag:n])).mean(0))
        c = np.hstack(c)
    else:
        raise ValueError("Acceptable method flags are 'scipy' or 'numpy'")
        

    lags = lags[len(lags)//2:]

    if normalize:
        var1 = np.sqrt((x1*np.conj(x1)).mean())
        var2 = np.sqrt((x2*np.conj(x2)).mean())
        c = c/var1/var2
    
    return lags,c.real

if __name__ == "__main__":
    
    n = 1000
    a = 0#5e-1
    t = np.linspace(0,10*np.pi,n)
    x = np.sin(t) + a*np.random.randn(n) + 10 + 1j*np.cos(t)

    lags,c = correlate(x,method="scipy")
    lags,ci = correlate(x,method="numpy")


    fig,ax = plt.subplots()
    ax.plot(lags*np.diff(t)[0],c)
    ax.plot(lags*np.diff(t)[0],ci)