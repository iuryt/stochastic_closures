"""
Plot planes from joint analysis files.

Usage:
    plot_spectrum.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./spectrum]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from dedalus.extras import plot_tools
from mpi4py import MPI


# Bases and domain
import dedalus.public as de
import param_dns as param

x_basis = de.Fourier('x', param.N, interval=(0, param.L), dealias=3/2)
y_basis = de.Fourier('y', param.N, interval=(0, param.L), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64, comm=MPI.COMM_SELF)

kx = domain.elements(0)
ky = domain.elements(1)
dkx = dky = 2 * np.pi / param.L
k = (kx**2 + ky**2)**0.5

kmax = int(np.ceil(np.max(k)))
bins = np.arange(1, kmax+1, 2)
kcen = bins[:-1] + np.diff(bins)/2
hist_samples, _ = np.histogram(k, bins=bins)


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    figsize = (8, 6)
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    # Plot writes
    fig = plt.figure(figsize=figsize)
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            # Load data
            ψc = file['tasks']['ψ'][index]
            # Compute 1D spectrum samples
            E_k2 = 0.5 * k**2 * np.abs(ψc)**2 / dkx / dky
            E_k1 = E_k2 * 2 * np.pi * k
            E_k1 *= (2 - (kx == 0))  # Double-count positive kx because of R2C transform
            # Plot histogram
            pow_samples, _ = np.histogram(k, bins=bins, weights=E_k1)
            spectrum = pow_samples / hist_samples / np.diff(bins)
            plt.loglog(kcen, param.ε**(2/3)*kcen**(-5/3), '--k')
            plt.loglog(kcen, param.η**(2/3)*kcen**(-3), '--k')
            plt.loglog(kcen, spectrum, '.-')
            plt.ylim([1e-12, 1e0])
            plt.xlabel("k")
            plt.ylabel("E(k)")
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            fig.suptitle(title, x=0.48, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

