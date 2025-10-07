#!/bin/bash
########## Reserve computing resources ##########
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --partition=gpu-v100

########## Set environment variables ##########
module load cuda/12.1.1

########## Run your script ##########
source .venv/bin/activate
python rlhf_activation_patching.py
