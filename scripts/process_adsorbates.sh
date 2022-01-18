#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB

python3 process_adsorbates.py
