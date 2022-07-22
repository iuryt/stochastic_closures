import h5py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from cmcrameri import cm

fname = "/work/isimoesdesousa_umassd_edu/projects/GFD/dns_runs/run1d2/snapshots/snapshots_s1.h5"
data = h5py.File(fname,"r")

noises = [0,1e-6,1e-8,1e-10,1e-12]

def get_scalars(run_name,noise):
    
    fnames = glob(f"../../../dns_runs/{run_name}/scalars/scalars_s*.h5")
    fnames.sort()
    
    E = []
    Z = []
    time = []
    for fname in fnames:
        data = h5py.File(fname,"r")
        E.append(np.ravel(data["tasks"]["E"][:]))
        Z.append(np.ravel(data["tasks"]["Z"][:]))
        time.append(np.ravel(data["scales"]["sim_time"][:]))
    
    time = np.hstack(time)
    ind = np.argsort(time)
    
    E = np.hstack(E)[ind]
    Z = np.hstack(Z)[ind]
    time = time[ind]
    
    # time = time-time[0]
    
    
    E = (
            xr.DataArray(E,dims=("time"),coords=dict(time=("time",time)))
            .expand_dims("noise").rename("E")
        ).assign_coords(noise=("noise",[noise]))

    
    Z = (
            xr.DataArray(Z,dims=("time"),coords=dict(time=("time",time)))
            .expand_dims("noise").rename("Z")
        ).assign_coords(noise=("noise",[noise]))
    
    return xr.merge([E,Z])


ds = []
for i,run_name in enumerate(["run1c"]):
    dsi = get_scalars(run_name,0)
    ds.append(dsi)
scalars = xr.merge(ds,compat="override") 

ds = []
for i,run_name in enumerate(["run1d2"]+[f"run{i}" for i in range(2,5+1)]):
    dsi = get_scalars(run_name,noises[i])
    ds.append(dsi)
    
scalars = xr.concat([scalars,xr.merge(ds)],"time")

tmin = scalars.time.min().values
tmid = scalars.sel(noise=1e-6).dropna("time").time.min().values
tmax = scalars.sel(noise=1e-6).dropna("time").time.max().values

fig,ax = plt.subplots(1,2)
fig.subplots_adjust(wspace=0.01)
for a in ax:
    scalars.E.plot.line(ax=a,hue="noise",add_legend=False)
    a.grid(True, linestyle="--", alpha=0.5)
    
ax[0].legend(noises,title="Noise")
ax[0].set(
    xlim=[tmin,tmid]
)
ax[1].set(
    ylabel="",
    yticklabels=[],
    xlim=[tmid,tmax]
)

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
