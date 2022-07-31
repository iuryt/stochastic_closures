import xrft
import xarray as xr
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


n=150
t = np.linspace(0,5*2*np.pi,n)
y = np.sin(t)
da = xr.DataArray(y, dims=('t',), coords={'t':t})

var = da.var()

# Nearest size with power of 2
size = int(2**np.ceil(np.log2(da.size)))

da_padded = xrft.padding.pad(da, t=(0, size-da.size))
fc = xrft.fft(da_padded)

corr = xrft.ifft(fc * np.conj(fc)).real / var

corr = corr.sel(t=da.t,method="nearest") / (np.arange(1,da.size+1)[::-1])

# corr = xrft.ifft(xrft.fft(da)*xrft.fft(np.conj(da.assign_coords(t=da.t.values[::-1]).sortby("t")))).real

# corr.plot()


# lags = signal.correlation_lags(da.size,da.size)
# corr_scipy = signal.correlate(da.values,da.values)#*(1/(n-lags))
# corr_scipy = corr_scipy[corr_scipy.size//2:]
# corr_scipy = xr.ones_like(corr)*corr_scipy


# fig, ax = plt.subplots()
# corr.plot()
# corr_scipy.plot()










import numpy as np

n = 256
t = np.linspace(0,4*2*np.pi,n)
data = np.sin(t)+10

# Nearest size with power of 2
size = 2 ** np.ceil(np.log2(2*len(data) - 1)).astype('int')

# Variance
var = data.var()

# Normalized data
ndata = data - np.mean(data)

# Compute the FFT
fft = np.fft.fft(ndata, size)

# Get the power spectrum
pwr = fft * np.conj(fft)

# Calculate the autocorrelation from inverse FFT of the power spectrum
acorr = np.fft.ifft(pwr).real / var
acorr = acorr[:len(data)]/(np.arange(1,data.size+1)[::-1])

plt.plot(t, acorr)


