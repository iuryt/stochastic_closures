
"""
Plot planes from joint analysis files.

Usage:
    filter_snapshots.py <N_filt> <mlog10_ep> <files>... [--output=<dir>] [--parallel]

Options:
    --output=<dir>  Output directory [default: ./filtered]
    --parallel      Distribute analysis over COMM_WORLD

"""

import h5py
import post
import filter
import xarray
import numpy as np
import param_dns


def save_subgrid_fields(filename, N_filt, epsilon, comm, output_path):
    """Compute and save subgrid velocity, stress, and strain components."""
    print(filename)
    out = {}
    # Load streamfunction
    domain = post.build_domain(param_dns.N, param_dns.L, comm=comm)
    ψ = post.load_field_hdf5(filename, domain, 'ψ', 0, layout='c')
    print('Done loading fields')
    # Filter velocities
    dx = domain.bases[0].Differentiate
    dy = domain.bases[1].Differentiate
    ux = dy(ψ).evaluate()
    uy = (-dx(ψ)).evaluate()
    F = filter.build_gaussian_filter(domain, N_filt, epsilon)
    out['ux'] = F_ux = F(ux).evaluate()
    out['uy'] = F_uy = F(uy).evaluate()
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
    fx_im = (- dx(txx_im) - dy(tyx_im)).evaluate()
    fy_im = (- dx(txy_im) - dy(tyy_im)).evaluate()
    fx_ex = (- dx(txx_ex) - dy(tyx_ex)).evaluate()
    fy_ex = (- dx(txy_ex) - dy(tyy_ex)).evaluate()
    out['pi_im'] = (dx(fy_im) - dy(fx_im)).evaluate()
    out['pi_ex'] = (dx(fy_ex) - dy(fx_ex)).evaluate()
    print('Done computing stresses')
    # Truncate and convert to xarray
    for key in out:
        field = out[key]
        field.require_coeff_space()
        field.set_scales(N_filt / param_dns.N)
        out[key] = post.field_to_xarray(field, layout='g')
    print('Done converting to xarray')
    # Save to netcdf
    ds = xarray.Dataset(out)
    input_path = pathlib.Path(filename)
    output_filename = output_path.joinpath(input_path.stem).with_suffix('.nc')
    ds.to_netcdf(output_filename)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools.parallel import Sync
    from mpi4py import MPI

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    # Distribute
    if args['--parallel']:
        comm = MPI.COMM_WORLD
        files = args['<files>']
        raise NotImplementedError()
    else:
        rank = MPI.COMM_WORLD.rank
        size = MPI.COMM_WORLD.size
        comm = MPI.COMM_SELF
        files = args['<files>'][rank::size]
    # Run
    N_filt = int(args['<N_filt>'])
    epsilon = 10**(-float(args['<mlog10_ep>']))
    for file in files:
        save_subgrid_fields(file, N_filt, epsilon, comm, output_path)

