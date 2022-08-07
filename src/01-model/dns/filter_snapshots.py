"""
Filter and compute subgrid stresses from saved output.

Methods:
    sigma: f = exp(-0.5*(k/parameter)**2)
    epsilon: f = exp(log(parameter) * (k / kcut)**2)
    mu: f = exp(-0.5 * (k / (parameter * kcut))**2))
    sharp: f = 1 if k < kcut else 0

kcut: cutoff wavenumber for epsilon  (N_filt - 1) // 2

Usage:
    filter_snapshots.py sigma <parameter> <N_filt> <files>... [--output=<dir>]
    filter_snapshots.py epsilon <parameter> <N_filt> <files>... [--output=<dir>]
    filter_snapshots.py mu <parameter> <N_filt> <files>... [--output=<dir>]
    filter_snapshots.py sharp <N_filt> <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./filtered]
    -h --help     Show this screen.

"""

import h5py
import post
import filter
import xarray as xr
import numpy as np
import param_dns


def save_subgrid_fields(filename, method, parameter, N_filt, output_path, comm):
    """Compute and save subgrid velocity, stress, and strain components."""
    if comm.rank == 0:
        print(filename)
    out = {}
    # Get writes
    with h5py.File(filename, 'r') as file:
        write_numbers = file['scales']['write_number'][:]
    domain = post.build_domain(param_dns.N, param_dns.L, comm=comm)
    for index, write in enumerate(write_numbers):
        if comm.rank == 0: 
            print(index)
        ψ = post.load_field_hdf5(filename, domain, 'ψ', index=index, layout='c')
        if comm.rank == 0:
            print('Done loading fields')
        # Filter velocities
        dx = domain.bases[0].Differentiate
        dy = domain.bases[1].Differentiate
        ux = dy(ψ).evaluate()
        uy = (-dx(ψ)).evaluate()
        F = filter.build_filter(domain, method, parameter, N=N_filt)
        out['ux'] = F_ux = F(ux).evaluate()
        out['uy'] = F_uy = F(uy).evaluate()
        if comm.rank == 0: 
            print('Done filtering fields')
        # Compute implicitly filtered subgrid stress components
        out['txx_im'] = txx_im = (F(ux*ux) - F_ux*F_ux).evaluate()
        out['tyy_im'] = tyy_im = (F(uy*uy) - F_uy*F_uy).evaluate()
        out['txy_im'] = txy_im = (F(ux*uy) - F_ux*F_uy).evaluate()
        # Compute explicitly filtered subgrid stress components
        out['txx_ex'] = txx_ex = (F(ux*ux) - F(F_ux*F_ux)).evaluate()
        out['tyy_ex'] = tyy_ex = (F(uy*uy) - F(F_uy*F_uy)).evaluate()
        out['txy_ex'] = txy_ex = (F(ux*uy) - F(F_ux*F_uy)).evaluate()
        # Compute vorticity tendency
        fx_im = (- dx(txx_im) - dy(txy_im)).evaluate()
        fy_im = (- dx(txy_im) - dy(tyy_im)).evaluate()
        fx_ex = (- dx(txx_ex) - dy(txy_ex)).evaluate()
        fy_ex = (- dx(txy_ex) - dy(tyy_ex)).evaluate()
        out['pi_im'] = (dx(fy_im) - dy(fx_im)).evaluate()
        out['pi_ex'] = (dx(fy_ex) - dy(fx_ex)).evaluate()
        if comm.rank == 0: 
            print('Done computing stresses')
        # Truncate and convert to xarray
        for key in out:
            field = out[key]
            field.require_coeff_space()
            field.set_scales(N_filt / param_dns.N)
            out[key] = post.field_to_xarray(field, layout='g')
        if comm.rank == 0: 
            print('Done converting to xarray')
        # Save to netcdf
        if comm.rank == 0:
            output_filename = output_path.joinpath(f"write_{write}.nc")
            ds = xr.Dataset(out)
            ds.attrs['method'] = method
            ds.attrs['parameter'] = parameter
            ds.attrs['N_filt'] = N_filt
            ds.to_netcdf(output_filename)
        comm.barrier()
    if comm.rank == 0: 
        print('Done saving files')


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools.parallel import Sync
    from mpi4py import MPI

    args = docopt(__doc__)

    # Get the filenames
    files = args['<files>']

    # Get method
    for key in ["sigma", "epsilon", "mu"]:
        if args[key]:
            method = key
            break

    # Get N_filt and parameter
    N_filt = int(args['<N_filt>'])
    parameter = args['<parameter>']

    # Get output directory
    output_path = pathlib.Path(args['--output']).absolute()

    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    
    # Loop over files
    for filename in files:
        save_subgrid_fields(filename, method, N_filt, parameter, output_path, MPI.COMM_WORLD)

 
