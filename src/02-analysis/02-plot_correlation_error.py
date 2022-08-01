
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from cmcrameri import cm
import params

# %%

def plot(ds):
    fig, ax = plt.subplots(2,len(noise),figsize=(12,4))

    for i,axi in enumerate(np.array(ax).T):
        hc = ds.isel(noise=i, k=slice(1,None)).correlation.plot.contourf(ax=axi[0], **kw["correlation"]["contourf"])
        he = ds.isel(noise=i, k=slice(1,None)).error.plot.contourf(ax=axi[1], **kw["error"]["contourf"])

        axi[0].set(
            yscale="log",
            ylim=[1e-4,ds.time.max()],
        )

        axi[1].set(
            yscale="log",
            ylim=[1e-4,ds.time.max()],
        )

    fig.colorbar(hc, ax=ax[0,:], label="Correlation")
    fig.colorbar(he, ax=ax[1,:], label="Error")


    for a in np.ravel(ax):
        a.set(
            xscale="log",
            xlabel="k",
            ylabel="time",
        )

    _ = [a.set(yticklabels=[], ylabel="") for a in np.ravel(ax[:,1:])]
    _ = [a.set(xticklabels=[], xlabel="") for a in np.ravel(ax[0,:])]

    _ = [a.grid(True, linestyle="--", alpha=0.7) for a in np.ravel(ax)]

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
                levels=[0.5],
                colors=["0.3"],
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
    )
)


# %%

print("Auto/cross-correlation noise-free and noisy simulations")
fnames = glob("../../data/noise00*")
fnames.sort()

noise = [0]
for fname in fnames:
    exp = -int(fname.split("noise")[-1].split("a")[0].split("b")[0].split(".")[0])
    if exp!=0:
        noise.append(10**exp)

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
fnames = list(set(glob("../../data/noise*"))-set(glob("../../data/noise00*")))
fnames.sort()

noise = []
for fname in fnames:
    exp = -int(fname.split("noise")[-1].split("b")[0].split(".")[0])
    noise.append(10**exp)

ds = xr.open_mfdataset(fnames, concat_dim="noise", combine="nested")
ds = ds.assign_coords(noise=noise)

ds = ds.sortby("noise")

print("Done loading data")

fig, ax = plot(ds)
fig.savefig("../../img/CE_noise_ab.png", **params.kw["savefig"])
print("Done saving figure")

