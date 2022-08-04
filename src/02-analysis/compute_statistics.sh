#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=10:00:00
# Streams
#SBATCH --output=job.out
#SBATCH --error=job.err
# Content
source setup.sh
python3 01-compute_statistics.py

