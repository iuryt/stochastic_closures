
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
import params
import os

import dataloader

import xrft
# %%

def plot(ds):
    fig, ax = plt.subplots(2,len(noise),figsize=(12,4))
    
    level = kw["error"]["contour"]["levels"][0]
    s = f"e={level:.1f}"
    ax[0,-1].text(1e2, 1e-1, s, zorder=10, ha="left", color="r")
    
    level = kw["correlation"]["contour"]["levels"][0]
    s = f"c={level:.1f}"
    ax[1,0].text(1e2, 4e-2, s, zorder=10, ha="left", color="r")
    
    for a in np.ravel(ax):
        a.plot(32, ds.time.max(), marker="v", markersize=10, color="0.5", zorder=10)
        (1/(ds.k*np.sqrt(2*1.62))).isel(k=slice(1,None)).rename("cfl").plot(ax=a, color="0.4")
    ax[0,0].text(1e1,3e-2,"(k$\sqrt{2E}$)$^{-1}$", fontsize=8, color="0.3", rotation=-33, va="top")
        
    for i,axi in enumerate(np.array(ax).T):
        hc = (
            ds.isel(noise=i, k=slice(1,None)).correlation
            .plot.contourf(ax=axi[0], **kw["correlation"]["contourf"])
        )
        
        hc2 = (
            ds.isel(noise=i, k=slice(1,None)).correlation
            .plot.contour(ax=axi[1], **kw["correlation"]["contour"])
        )

        he = (
            ds.isel(noise=i, k=slice(1,None)).normalized_error
            .plot.contourf(ax=axi[1], **kw["error"]["contourf"])
        )
        
        he2 = (
            ds.isel(noise=i, k=slice(1,None)).normalized_error
            .plot.contour(ax=axi[0], **kw["error"]["contour"])
        )

        axi[0].set(
            yscale="log",
            ylim=[1e-4,ds.time.max()],
        )

        axi[1].set(
            ylim=[1e-4,ds.time.max()],
        )

    fig.colorbar(hc, ax=ax[0,:], label="Correlation")
    fig.colorbar(he, ax=ax[1,:], label="Normalized Error")



    for a in np.ravel(ax):
        a.set(
            xscale="log",
            xlabel="k",
            ylabel="time",
        )

    _ = [a.set(yticklabels=[], ylabel="") for a in np.ravel(ax[:,1:])]
    _ = [a.set(xticklabels=[], xlabel="") for a in np.ravel(ax[0,:])]

    _ = [a.grid(True, linestyle="--", alpha=0.7) for a in np.ravel(ax)]

    _ = [a.set(title="") for a in ax[1]]
    
    return fig, ax


# %%

kw = dict(
    correlation = dict(
            contourf = dict(
                cmap=cm.acton,
                vmin=-0.1, vmax=1,
                levels=np.arange(0, 1+0.1, 0.1),
                add_colorbar=False,
                extend="both",
            ),
            contour = dict(
                levels=[0.6],
                colors=["r"],
                linestyles="--"
            )
        ),
    error = dict(
            contourf = dict(
                cmap=cm.devon,
                vmin=0, vmax=1,
                levels=np.arange(0, 1+0.1, 0.1),
                add_colorbar=False,
                extend="both",
            ),
            contour = dict(
                levels=[1],
                linestyles="--",
                colors=["r"],
            )
    )
)



# %%

print("Auto/cross-correlation noise-free and noisy simulations")
fnames = [os.path.join(params.data_dir,"no_noise_0_no_noise_0.nc")]
fnames = fnames+[os.path.join(params.data_dir,f"no_noise_0_{run}_00.nc") for run in params.noise_runs]

noise = [0]
for fname in fnames[1:]:
    noise.append(10**-int(fname.split("_")[-2]))

ds = xr.open_mfdataset(fnames, concat_dim="noise", combine="nested")
ds = ds.assign_coords(noise=noise)

ds = ds.sortby("noise")

print("Done loading data")

fig, ax = plot(ds)
ax[0,0].set_title("Autocorrelation")
fig.savefig("../../img/CE_noise.png", **params.kw["savefig"])
print("Done saving figure")

# %%

print("Compare runs with the same noise amplitude")
fnames = [os.path.join(params.data_dir,f"{run}_00_{run}_01.nc") for run in params.noise_runs]

noise = []
for fname in fnames:
    noise.append(10**-int(fname.split("_")[-2]))

ds = xr.open_mfdataset(fnames, concat_dim="noise", combine="nested")
ds = ds.assign_coords(noise=noise)

ds = ds.sortby("noise")

print("Done loading data")

fig, ax = plot(ds)
fig.savefig("../../img/CE_noise_ab.png", **params.kw["savefig"])
print("Done saving figure")


# # %%

# from cycler import cycler
# import matplotlib as mpl

# tslice = slice(None,None,20)

# colors = mpl.cm.Greys(np.linspace(1,0.4,ds.isel(time=tslice).time.size))
# custom_cycler = cycler(color=colors)

# spectrum_of_difference = ds.spectrum_of_difference*ds.k**2
# difference_of_spectrum = ds.difference_of_spectrum*ds.k**2

# fig, ax = plt.subplots(2, spectrum_of_difference.noise.size-1, figsize=(13,5))

# _ = [a.set_prop_cycle(custom_cycler) for a in np.ravel(ax)]

# kw = dict(hue = "time", add_legend = False)
# for axi, noise in zip(ax.T, ds.noise.values[1:]):
#     (ds.spectrum_1.sel(noise=noise).isel(time=0)*(ds.k**2)).plot.line(ax=axi[0], color="r", zorder=10)
#     (ds.spectrum_1.sel(noise=noise).isel(time=0)*(ds.k**2)).plot.line(ax=axi[1], color="r", zorder=10)
#     spectrum_of_difference.sel(noise=noise).isel(time=tslice).plot(**kw, ax=axi[0], label="spectrum of difference")
#     difference_of_spectrum.sel(noise=noise).isel(time=tslice).plot(**kw, ax=axi[1], label="difference of spectrum")
#     axi[0].set(ylabel="Spectrum of difference")
#     axi[1].set(title="", ylabel="Difference of spectrum")

#     for a in axi:
#         a.set(
#             xscale="log",
#             yscale="log",
#             xlabel="k",
#             ylim=[-8,0],
#             xlim=[1e0,1.5e3]
#         )
#         a.grid(True, linestyle="--", alpha=0.7)

# for a in np.ravel(ax[:,1:]):
#     a.set(ylabel="")
# # %%

# # %%

# # %%
