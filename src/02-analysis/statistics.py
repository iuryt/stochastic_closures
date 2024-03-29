import xarray as xr
from xhistogram.xarray import histogram
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal
import sys

def correlate(x1,x2=None,method="scipy",normalize=True):
    
    if x2 is None:
        x2 = x1
    
    n = len(x1)
    
    lags = signal.correlation_lags(len(x1),len(x2))

    if method=="scipy":
        c = (1/(n-lags))*signal.correlate(x1,x2,method="fft")
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


def compute_correlation_kbins(da, db, step_bin = 2):
    """
    da,db == (xarray.DataArray,xarray.DataArray)
    """
    kbins = np.arange(0,np.ceil((np.abs(da.kx + 1j*da.ky)).values.max())+step_bin, step_bin)

    # allocate in memory the autocorrelation matrix C(k,lag)
    C = np.zeros((kbins.size-1,da.time.size))

    kxs = da.kx.values
    kys = da.ky.values

    a = da.values
    b = db.values

    for i,kx in enumerate(tqdm(kxs)):
        for j,ky in enumerate(kys):
            # %%timeit
            aij = a[:,i,j]
            bij = b[:,i,j]

            # calculate the autocorrelation using fft method
            _, cij = correlate(aij,bij)

            # find the bin indexes
            k = np.abs(kx + 1j*ky)

            ind = np.argwhere((k>=kbins)[:-1]&(k<kbins)[1:])[0][0]

            C[ind] = C[ind]+cij

    K = np.abs(da.kx + 1j*da.ky).rename("k")
    H = histogram(K, bins = kbins).rename(k_bin = "k")

    C = xr.DataArray(
        C,
        dims = ("k", "time"),
        coords = dict(
            time = ("time", da.time.values-da.time.values.min()),
            k = ("k", H.k.values),
        )
    ).T/H
    
    C = C.rename("correlation")

    return C


def apply_func_kbins(da, db, step_bin = 2, normalize=True, reset_time=True, name="error",
                    func = lambda a,b: np.abs(a-b),
                    norm_func = lambda E,a: E/np.abs(a),
                    average=True):
    """
    da,db == (xarray.DataArray,xarray.DataArray)
    """

    E = func(da,db)
    
    if normalize:
        E = norm_func(E,da)
        
    if reset_time:
        E = E.assign_coords(time=E.time.values-E.time.values.min())

    kbins = np.arange(0,np.ceil((np.abs(da.kx + 1j*da.ky)).values.max())+step_bin, step_bin)

    K = np.abs(da.kx + 1j*da.ky).rename("k")
    H = histogram(K, bins = kbins).rename(k_bin = "k")

    E = (
        histogram((K*xr.ones_like(E)).rename("k"), bins = kbins, weights=E, dim=["kx","ky"]).rename(k_bin = "k")
    )
    
    if average:
        E = E/H
 
    E = E.rename(name)

    return E


def compute_statistics(da, db, print_file=sys.stderr):   
    # Create list of statistics to be computed
    S = []

    print("Computing errors", file = print_file)

    name = f"normalized_error"
    func = lambda a,b: np.abs(a-b)
    norm_func = lambda E,a: E/np.abs(a)
    S.append(apply_func_kbins(da, db, func = func, norm_func=norm_func, name = name))

    name = f"spectrum_of_difference"
    func = lambda a,b: np.abs(a-b)**2
    S.append(apply_func_kbins(da, db, func = func, normalize = False, name = name, average = False))

    name = f"difference_of_spectrum"
    func = lambda a,b: np.abs(np.abs(a)**2-np.abs(b)**2)
    S.append(apply_func_kbins(da, db, func = func, normalize = False, name = name, average = False))
    print("Done computing errors", file = print_file)

    print("Computing spectra", file = print_file)
    
    func = lambda a,b: (a*np.conj(b)).real

    name = f"cross_spectrum"
    S.append(apply_func_kbins(da, db, func = func, normalize = False, name = name, average = False))
    name = f"spectrum_1"
    S.append(apply_func_kbins(da, da, func = func, normalize = False, name = name, average = False))
    name = f"spectrum_2"
    S.append(apply_func_kbins(db, db, func = func, normalize = False, name = name, average = False))
    print("Done computing spectra", file = print_file)


    print("Computing correlations", file = print_file)
    S.append(compute_correlation_kbins(da, db))
    print("Done computing correlations", file = print_file)

    return xr.merge(S)
    

if __name__ == "__main__":
    
    n = 1000
    a = 0#5e-1
    t = np.linspace(0,10*np.pi,n)
    x = np.sin(t) + a*np.random.randn(n)

    lags,c = correlate(x,method="scipy")
    lags,ci = correlate(x,method="numpy")

    print(np.allclose(c,ci))

    fig,ax = plt.subplots()
    ax.plot(lags*np.diff(t)[0],c)
    ax.plot(lags*np.diff(t)[0],ci)
