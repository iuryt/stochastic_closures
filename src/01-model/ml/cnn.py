# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import xarray as xr
from glob import glob
import os
import numpy as np

import matplotlib.pyplot as plt

# %%
# Path to the data
path = "../../../../dns_runs/run0/run1c/filtered/"
# List of files
fnames = glob(os.path.join(path,"*.nc"))
# Sort the files
ind = np.argsort([int(fname.split("s")[-1].split(".")[0]) for fname in fnames])
fnames = np.array(fnames)[ind]

#%%

# Load the data
ds = xr.open_mfdataset(fnames, concat_dim="time", combine="nested").load()

# Define input and output variables

channels = ["ux", "uy"]
input_data = (
    xr.concat([ds[channel] for channel in channels],"channel")
        .assign_coords(channel=channels)
)

channels = ["im_txx", "im_txy", "im_tyy"]
output_data = (
    xr.concat([ds[channel] for channel in channels],"channel")
        .assign_coords(channel=channels)
)


# %%


class Net(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, num_filters=5, kernel_size=5, stride=1, β=0):
        super().__init__()
        if kernel_size%2==0:
            ValueError(f"kernel_size should be an odd number, given {kernel_size}")
        padding = (kernel_size-1)//2
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size, stride, padding, padding_mode="circular")
        self.conv2 = nn.Conv2d(num_filters, num_filters+β, kernel_size, stride, padding, padding_mode="circular")
        self.conv3 = nn.Conv2d(num_filters+β, num_filters+2*β, kernel_size, stride, padding, padding_mode="circular")
        self.conv4 = nn.Conv2d(num_filters+2*β, num_filters+3*β, kernel_size, stride, padding, padding_mode="circular")
        self.linear1 = nn.Linear(num_filters+3*β, out_channels)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear1(x)
        return x

    
net = Net()
loss_function = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

