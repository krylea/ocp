#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --partition=t4v1,t4v2,p100,rtx6000
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=25GB

python -u -m torch.distributed.launch --nproc_per_node=8 main.py --distributed --num-gpus 8 --mode train --config-yml ads-dpp.yml