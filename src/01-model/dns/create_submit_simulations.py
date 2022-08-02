"""
Creates and submits simulations for the DNS model.

Example usage:
    python3 create_submit_simulations.py --noise_amp=7 --n_ensembles=100 --output_dir=../../../../dns_runs/run1/ensembles/ --no-submit
"""

def create_submit_simulation(args,i):
    """
    Creates and submits a simulation for the DNS model.
    """
    import os
    import shutil
    import numpy as np
    
    # Create the simulation name
    if args.noise:
        sim_name = f'{{}}noise_{{:03.0f}}_{{:0{np.ceil(np.log10(args.n_ensembles)).astype(int)+1}.0f}}'.format(args.prefix, -args.noise_amp, i)
    else:
        sim_name = f'{{}}no_noise_{{:0{np.ceil(np.log10(args.n_ensembles)).astype(int)+1}.0f}}'.format(args.prefix, i)

    # Create the simulation directory
    sim_dir = os.path.join(args.output_dir, sim_name)
    os.makedirs(sim_dir, exist_ok=True)
    print("Created simulation directory: {}".format(sim_dir))

    # Create simbolic link to the setup file
    os.symlink(args.setup_file, os.path.join(sim_dir, 'setup.sh'))

    # Create symbolic link to the restart file
    os.symlink(args.restart_file, os.path.join(sim_dir, 'restart.h5'))

    print("Created symbolic links to the setup and restart files.")

    # Copy Dedalus config file to the simulation directory
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dedalus.cfg'), sim_dir)

    # Copy run_commands.sh to the simulation directory
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_commands.sh'), sim_dir)

    print("Copied Dedalus config file and run_commands.sh to the simulation directory.")

    # Copy all Python files to the simulation directory
    for file in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        if file.endswith('.py'):
            shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), file), sim_dir)

    print("Copied all Python files to the simulation directory.")

    # Add the noise amplitude to param_dns.py
    with open(os.path.join(sim_dir, 'param_dns.py'), 'a') as f:
        if args.noise:
            f.write('\n')
            f.write('noise_amp = {}\n'.format(10**(args.noise_amp/2)))
        else:
            f.write('\n')
            f.write('noise_amp = 0\n')

    print("Added the noise amplitude to param_dns.py.")

    # Create and possibly submit the batch file
    create_batch_file(args.partition, args.nodes, args.ntasks_per_node, args.time, sim_name, sim_dir, submit=args.submit)

    print("Created the batch file.")

    if args.submit:
        print("Submitted the batch file.")
    else:
        print("Not submitted the batch file.")


def create_batch_file(partition, nodes, ntasks_per_node, time, job_name, output_dir, submit=True):
    """
    Creates a batch file for a simulation.
    """
    
    import os
    import subprocess
    import shutil
    import tempfile

    # Create a temporary directory for the job
    tmp_dir = tempfile.mkdtemp()

    # Create a temporary file for the batch script
    batch_file = os.path.join(tmp_dir, 'batch_script.sh')
    with open(batch_file, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --partition={}\n'.format(partition))
        f.write('#SBATCH --nodes={}\n'.format(nodes))
        f.write('#SBATCH --ntasks-per-node={}\n'.format(ntasks_per_node))
        f.write('#SBATCH --time={}\n'.format(time))
        f.write('#SBATCH --job-name={}\n'.format(job_name))
        f.write('#SBATCH --output={}/job.out\n'.format(output_dir))
        f.write('#SBATCH --error={}/job.err\n'.format(output_dir))
        f.write('\n')
        f.write('bash run_commands.sh\n')

    # Move the batch script to the output directory
    shutil.move(batch_file, output_dir)

    # Change directory to the output directory
    os.chdir(output_dir)

    if submit:
        # Submit the job
        subprocess.call(['sbatch', 'batch_script.sh'])

    # Clean up the temporary directory
    shutil.rmtree(tmp_dir)

    # Change directory back to the original directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    # Get the arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=str, default='xeon-p8', help='Partition to run the job on')
    parser.add_argument('--nodes', type=int, default=8, help='Number of nodes to run the job on')
    parser.add_argument('--ntasks_per_node', type=int, default=48, help='Number of tasks to run on each node')
    parser.add_argument('--time', type=str, default='15:00:00', help='Time to run the job for')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to output the job to')
    parser.add_argument('--noise_amp', type=float, default=0, help='Noise amplitude (10**(-amp)) added to the kinetic energy spectrum (ignored if --no-noise is set)')
    parser.add_argument('--noise', default=False, action=argparse.BooleanOptionalAction, help='Add noise to the kinetic energy spectrum')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for the simulation')
    parser.add_argument('--n_ensembles', type=int, default=1, help='Number of ensembles to run')
    parser.add_argument('--submit', default=True, action=argparse.BooleanOptionalAction, help='Whether to submit the ensembles')
    parser.add_argument('--setup_file', type=str, default='../../../../setup.sh', help='Path to the setup file')
    parser.add_argument('--restart_file', type=str, default='../../../../restart.h5', help='Path to the restart file')
    args = parser.parse_args()

    print("Creating {} ensembles with noise amplitude 1e{}".format(args.n_ensembles, args.noise_amp))

    # Loop over the number of ensembles
    for i in range(args.n_ensembles):
        print("\n\nCreating ensemble {}".format(i))
        create_submit_simulation(args, i)
