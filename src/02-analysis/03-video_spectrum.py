
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
import params
import os
from tqdm import tqdm


# %%

print("Auto/cross-correlation noise-free and noisy simulations")
fnames = [os.path.join(params.data_dir,"no_noise_0_no_noise_0.nc")]
fnames = fnames+[os.path.join(params.data_dir,f"no_noise_0_{run}_00.nc") for run in params.noise_runs]
fnames = fnames[:5]

noise = [0]
for fname in fnames[1:]:
    noise.append(10**-int(fname.split("_")[-2]))

ds = xr.open_mfdataset(fnames, concat_dim="noise", combine="nested")
ds = ds.assign_coords(noise=noise)

ds = ds.sortby("noise")

# %%

from cycler import cycler
import matplotlib as mpl

tslice = slice(None,20,5)

Δt = np.diff(ds.isel(time=tslice).time.values)[0]

colors = mpl.cm.Greys(np.linspace(1,0.4,ds.isel(time=tslice).time.size))
custom_cycler = cycler(color=colors)

spectrum_of_difference = ds.spectrum_of_difference*ds.k**2
difference_of_spectrum = ds.difference_of_spectrum*ds.k**2

fig, ax = plt.subplots(3, spectrum_of_difference.noise.size-1, figsize=(13,7))
_ = [a.set_prop_cycle(custom_cycler) for a in np.ravel(ax)]


ax[0,0].text(1e2,1e-2,"E$_0$(k)", color="r")
ax[0,0].text(1,2e-8,f"$\\uparrow\,\,\Delta$t={Δt:.2f}", color="0.2")
ax[0,0].text(32+10,2e-8,"k$_{fr}$ = 32", color="0.2")

kw = dict(hue = "time", add_legend = False)
for axi, noise in zip(ax.T, ds.noise.values[1:]):
    (ds.spectrum_1.sel(noise=noise).isel(time=0)*(ds.k**2)).plot.line(ax=axi[0], color="r", zorder=10)
    (ds.spectrum_1.sel(noise=noise).isel(time=0)*(ds.k**2)).plot.line(ax=axi[1], color="r", zorder=10)
    spectrum_of_difference.sel(noise=noise).isel(time=tslice).plot(**kw, ax=axi[0], label="spectrum of difference")
    difference_of_spectrum.sel(noise=noise).isel(time=tslice).plot(**kw, ax=axi[1], label="difference of spectrum")
    
    (ds.spectrum_2.sel(noise=noise).isel(time=tslice)*(ds.k**2)).plot(**kw, ax=axi[2], label="spectrum")
    axi[0].set(ylabel="Spectrum of difference")
    axi[1].set(title="", ylabel="Difference of spectrum")

    for a in axi:
        a.axvline(32, color="k", linestyle="--", zorder=0)
        a.set(
            xscale="log",
            yscale="log",
            xlabel="k",
            ylim=[1e-8,1e0],
            xlim=[1e0,1.5e3]
        )
        a.grid(True, linestyle="--", alpha=0.7)

for a in np.ravel(ax[:,1:]):
    a.set(ylabel="")

fig.savefig("../../img/spectrum_difference.png", **params.kw["savefig"])

# %%

# Create a folder for the video frames
if not os.path.exists("../../img/spectrum_video"):
    os.makedirs("../../img/spectrum_video")

# Delete all existing frames
for f in os.listdir("../../img/spectrum_video"):
    os.remove(os.path.join("../../img/spectrum_video", f))
    

nt = 4
colors = mpl.cm.Greys(np.linspace(0.4,1,nt))
custom_cycler = cycler(color=colors)

for i,ti in enumerate(tqdm(np.arange(1,ds.time.size,1))):
    fig,ax = plt.subplots()
    fig.suptitle(f"time = {ds.time.values[ti]:04.2f}")
    ax.set_prop_cycle(custom_cycler)
    (
        (ds.spectrum_2.sel(noise=1e-4)*(ds.k**2)).isel(time=slice(max(ti-nt,0),ti,1))
        .rolling(k=9, center=True).mean()
        .plot.line(hue="time", xscale="log", add_legend=False, yscale="log")
    )
    ax.set(
        xlim=[50,1e3],
        ylim=[1e-4, 1e-3],
        ylabel="E(k)"
    )
    
    # Save the figure frame
    fig.savefig(f"../../img/spectrum_video/frame_{i:04d}.png", **params.kw["savefig"])

    # Close the figure
    plt.close(fig)


# Create a folder for the video frames
if not os.path.exists("../../img/spectrum_difference_video"):
    os.makedirs("../../img/spectrum_difference_video")

# Delete all existing frames
for f in os.listdir("../../img/spectrum_difference_video"):
    os.remove(os.path.join("../../img/spectrum_difference_video", f))


for i,ti in enumerate(np.arange(0,ds.time.size,2)):
    print(f"Frame {i}")
    fig, ax = plt.subplots(2, spectrum_of_difference.noise.size-1, figsize=(13,5))
    fig.suptitle(f"time = {ds.time.values[ti]:04.2f}")
    _ = [a.set_prop_cycle(custom_cycler) for a in np.ravel(ax)]


    kw = dict(color="0.2", add_legend = False)
    for axi, noise in zip(ax.T, ds.noise.values[1:]):
        (ds.spectrum_1.sel(noise=noise).isel(time=ti)*(ds.k**2)).drop("time").plot.line(ax=axi[0], color="r", zorder=10)
        (ds.spectrum_1.sel(noise=noise).isel(time=ti)*(ds.k**2)).drop("time").plot.line(ax=axi[1], color="r", zorder=10)
        spectrum_of_difference.sel(noise=noise).isel(time=ti).drop("time").plot(**kw, ax=axi[0], label="spectrum of difference")
        difference_of_spectrum.sel(noise=noise).isel(time=ti).drop("time").plot(**kw, ax=axi[1], label="difference of spectrum")
        axi[0].set(ylabel="Spectrum of difference")
        axi[1].set(title="", ylabel="Difference of spectrum")

        for a in axi:
            a.axvline(32, color="k", linestyle="--", zorder=0)
            a.set(
                xscale="log",
                yscale="log",
                xlabel="k",
                ylim=[1e-8,1e0],
                xlim=[1e0,1.5e3]
            )
            a.grid(True, linestyle="--", alpha=0.7)

    for a in np.ravel(ax[:,1:]):
        a.set(ylabel="")

    # Save the figure frame
    fig.savefig(f"../../img/spectrum_difference_video/frame_{i:04d}.png", **params.kw["savefig"])

    # Close the figure
    plt.close(fig)

# %%

# # Convert the frames to a video
# os.system('ffmpeg -y -framerate 10 -pattern_type glob -i "../../img/spectrum_difference_video/*" ../../img/spectrum_difference.mp4')

# # %%
# # Delete all the frames
# for f in os.listdir("../../img/spectrum_difference_video"):
#     os.remove(os.path.join("../../img/spectrum_difference_video", f))


# %%
