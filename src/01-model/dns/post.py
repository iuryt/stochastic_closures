"""Post-processing tools."""

import numpy as np
import dedalus.public as de
from dedalus.core.future import FutureField
import h5py
import xarray


def build_domain(N, L, comm=None):
    """Build domain object."""
    x_basis = de.Fourier('x', N, interval=(0, L), dealias=3/2)
    y_basis = de.Fourier('y', N, interval=(0, L), dealias=3/2)
    domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64, comm=comm)
    return domain


def load_field_hdf5(filename, domain, task, index=-1, layout='g'):
    """Load a field from HDF5 file."""
    field = domain.new_field(name=task)
    with h5py.File(filename, 'r') as file:
        field[layout] = file['tasks'][task][index]
    return field


def load_field_netcdf(filename, domain, task, layout='g'):
    """Load a field from netcdf file."""
    with xarray.open_dataset(filename) as file:
        field = xarray_to_field(file[task], layout)
    return field


def field_to_xarray(field, layout='g'):
    """Convert Dedalus field to xarray dataset."""
    # Evaluate operators
    if isinstance(field, FutureField):
        field = field.evaluate()
    data = field[layout]
    domain = field.domain
    layout = domain.dist.get_layout_object(layout)
    coords = []
    for axis in range(domain.dim):
        basis = domain.bases[axis]
        if layout.grid_space[axis]:
            label = basis.name
            scale = basis.grid(field.scales[axis])
        else:
            label = basis.element_name
            scale = basis.elements
        coords.append((label, scale))
    xr_data = xarray.DataArray(data, coords=coords)
    return xr_data


def xarray_to_field(data, domain, layout='g'):
    """Convert xarray dataset to Dedalus field."""
    field = domain.new_field(name=data.name)
    field[layout] = data.data
    return field
