source setup.sh
mpiexec python3 sim_dns.py
mpiexec python3 -m dedalus merge_procs scalars --cleanup
python3 -m dedalus merge_sets scalars.h5 scalars/*.h5
python3 plot_scalars.py scalars.h5
mpiexec python3 -m dedalus merge_procs snapshots --cleanup
mpiexec python3 plot_slices.py snapshots/*.h5
mpiexec python3 plot_spectrum.py snapshots/*.h5