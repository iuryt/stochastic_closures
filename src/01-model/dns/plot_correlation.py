import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from cmcrameri import cm

noises = [0,1e-6,1e-8,1e-10,1e-12]

ds = []
for noise in noises:
    fname = f"data/C_{-np.log10(noise):.0f}.nc" if noise!=0 else f"data/C_0.nc"
    ds.append(xr.open_dataset(fname))
ds = xr.concat(ds,dim="noise").assign_coords(noise=noises)
C = ds.__xarray_dataarray_variable__



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
        linestyles="--"
    )
)

fig, ax = plt.subplots(1,5,figsize=(12,4))

for i,a in enumerate(ax):
    
    h = C.isel(noise=i).plot.contourf(ax=a, **kw["contourf"])
    C.isel(noise=i).plot.contour(ax=a, **kw["contour"])
    
    a.grid(True, linestyle="--", alpha=0.7, color="0.1")
    a.set(
        yscale="log",
        xscale="log",
        ylim=[1e-5,C.lag.max()],
        xlabel="k",
        ylabel="lag",
        xticks=[1,10,100,1000]
    )
    
ax[0].set(title="autocorrelation")
_ = [a.set(ylabel="", yticklabels=[]) for a in ax[1:]]
fig.colorbar(h, ax=ax, orientation="horizontal", label="correlation", shrink=0.3, pad=0.2)

fig.savefig("img/C_noise.png", facecolor="w", dpi=300, bbox_inches="tight")
