import h5py
import xarray as xr
from xhistogram.xarray import histogram
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
from tqdm import tqdm
from glob import glob

letters = "a b c".split()

ds = {"E":[],"Z":[]}

time,time_scalar = [],[]
fnames = []
for letter in letters:
    
    fnamesi = glob(f"../dns_runs/run1{letter}/snapshots/*.h5")
    for fname in tqdm(fnamesi):
        data = h5py.File(fname,"r")
        timei = data["scales"]["sim_time"][:]
        time.append(timei)
        fnames.append([fname]*timei.size)
        
    fnamesi = glob(f"../dns_runs/run1{letter}/scalars/*.h5")
    for fname in fnamesi:
        data_scalar = h5py.File(fname,"r")
        time_scalar.append(data_scalar["scales"]["sim_time"][:])
        for key in ds:
            ds[key].append(data_scalar["tasks"][key][:,0,0])
            
x,y = data["scales"]["x"]["1.0"][:],data["scales"]["y"]["1.0"][:]    
time = np.hstack(time)
fnames = np.hstack(fnames)

fnames = fnames[np.argsort(time)]
time = time[np.argsort(time)]

ds = xr.Dataset({key:("time",np.hstack(ds[key])) for key in ds}, coords={"time":("time",np.hstack(time_scalar))})
ds = ds.sortby(ds.time)
ds = ds.isel(time=np.unique(ds.time,return_index=True)[1])





vm = 100
kw = dict(vmin=-vm,vmax=vm,cmap=cm.vik)

tis = [0.5,7]
fig, ax = plt.subplots(2,2,figsize=(6.5,7))

[a.grid(True,linestyle="--",alpha=0.5) for a in np.ravel(ax)]

for ti in tis:
    [a.axvline(ti,linestyle="--",color="0.2") for a in ax[-1]]
    
for i,key in enumerate(["E","Z"]):
    ds[key].plot(ax=ax[-1][i],color="0.2")
    ds[key].sel(time=tis,method="nearest").plot(ax=ax[-1][i],linewidth=0,marker=7,color="tab:red")
    

for i,ti in enumerate(tqdm(tis)):
    fname = fnames[np.argmin(np.abs(time-ti))]
    a = ax[0][i]
    data = h5py.File(fname,"r")
    ω = data["tasks"]["ω"][1]
    a.pcolormesh(x,y,ω,**kw)
    a.axis("scaled")
    
    # inset axes....
    axins = a.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.pcolormesh(x,y,ω,**kw)
    # sub region of the original image
    axins.set_xlim(1.5, 2.5)
    axins.set_ylim(1.5, 2.5)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    a.indicate_inset_zoom(axins, edgecolor="black")
    
fig.savefig("img/EZ.png",dpi=300,facecolor="w")

# e = data["tasks"]["e"][0]
# l,m = e.shape
# eh = np.fft.fft2(e)[:l//2,:m//2]

# kx,ky = np.meshgrid(data["scales"]["kx"],data["scales"]["ky"][:l//2])
# K = np.abs(kx + 1j*ky)

