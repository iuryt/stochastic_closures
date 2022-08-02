import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# Number of points
n = 50

# Create some dummy data
x,y = np.meshgrid(*(2*[np.linspace(0,2*np.pi,n)]))
z = (np.sin(x+y)+np.cos(y))


ks = 3  # kernel size
pad = (ks-1)//2

kw_conv = dict(
    in_channels=1,
    out_channels=1,
    kernel_size=ks,
    stride=1,
    padding=pad,
    padding_mode="circular",
    dtype=torch.float64,
    bias=False
)

# Convolution with PyTorch
m = nn.Conv2d(**kw_conv)
with torch.no_grad():
    m.weight[0,0,:] = 0
    m.weight[0,0,:][0,0] = 1
    
zt = torch.DoubleTensor(z[np.newaxis, np.newaxis, :, :])
zct = m(zt).detach().numpy()[0, 0, :, :]

# Convolution with SciPy
weights = m.weight.detach().numpy()[0, 0, ::-1, ::-1] # need to flip weights
zcs = signal.convolve2d(z, weights, boundary="wrap", mode="same")

# Error
error = (zct-zcs)

# Print the comparison
print(np.allclose(zcs,zct))



# Plot the results
fig, ax = plt.subplots(2, 2, figsize=(7,7))
ax = np.ravel(ax)
fig.subplots_adjust(hspace=0.2)

titles = ["Data", "Convolution (Pytorch)", "Convolution (SciPy)", "Error"]

for a,D,title in zip(ax, [z, zct, zcs, error], titles):
    C = a.pcolor(D)
    fig.colorbar(C, ax=a)
    a.set(
        title=f"{title}\n",
        xticks=[],
        yticks=[]
    )
