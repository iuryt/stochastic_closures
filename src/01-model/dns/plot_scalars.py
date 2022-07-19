"""
Plot scalars from single analysis file.

Usage:
    plot_scalars.py <file> [--output=<dir>]

Options:
    --output=<output>  Output file [default: ./img_scalars.pdf]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


def main(filename, output):
    """Plot scalar time-series."""

    # Data selection
    tasks = ['E','εα','εν','Z','ηα','ην']
    slices = (slice(None), 0, 0)

    # Plot tasks
    fig, axes = plt.subplots(len(tasks), 2, figsize=(10,10))
    if len(tasks) == 1:
        axes = [axes]

    with h5py.File(filename, mode='r') as file:
        sim_time = file['scales']['sim_time'][:]
        for i, task in enumerate(tasks):
            if task in file['tasks']:
                dset = file['tasks'][task]
                ax = axes[i][0]
                ax.loglog(sim_time[slices[0]], np.abs(dset[slices]), '.-')
                ax.set_ylabel(task)
                ax.grid()
                ax = axes[i][1]
                ax.plot(sim_time[slices[0]], dset[slices], '.-')
                ax.set_ylabel(task)
                ax.grid()
        ax.set_xlabel('sim time')

    # Finalize figure
    # plt.xlabel('sim time')
    # plt.legend(loc='lower right')
    plt.savefig(output)

if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    main(args['<file>'], args['--output'])

