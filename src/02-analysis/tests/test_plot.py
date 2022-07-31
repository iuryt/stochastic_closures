import xarray as xr
import matplotlib.pyplot as plt
from cmcrameri import cm
import numpy as np

C = (
    xr.open_mfdataset("../../data/C*", concat_dim = "noise", combine = "nested")
    .__xarray_dataarray_variable__.load().assign_coords(noise = [0, 1e-2, 1e-4, 1e-6])
)

C = C.sortby("noise")

kw = dict(
    contourf = dict(
        cmap = cm.acton,
        vmin = -0.1, vmax = 1,
        levels = np.arange(0, 1+0.1, 0.1),
        add_colorbar = False,
        extend = "both",
    ),
    contour = dict(
        levels = [0.5],
        colors = ["0.3"],
        linestyle = "--"
    )
)

fig,ax = plt.subplots(2, C.noise.size//2, figsize = (9, 7))
ax = np.ravel(ax)

fig.subplots_adjust(hspace=0.4, wspace=0.4)

for a,noise in zip(ax, C.noise):
    h = C.sel(noise = noise).plot.contourf(ax = a, **kw["contourf"])
    C.sel(noise = noise).plot.contour(ax = a, **kw["contour"])

    a.set(
        yscale = "log",
        xscale = "log",
        ylim = [1e-5,C.time.max()],
        xlabel = "k",
        ylabel = "time lag"
    )

fig.colorbar(h, ax = ax, label = "correlation", shrink = 0.6)

ax[0].set_title("Autocorrelation")