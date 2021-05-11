#!/bin/bash

#SBATCH --job-name=1M_eval_VAE
#SBATCH --output=1M_eval_gpu_job_output.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=128G
#SBATCH --gres=gpu:v100:2
#SBATCH --partition=gpu
#SBATCH --time=2-
#SBATCH --mail-user=josh.beasley@yale.edu
#SBATCH --mail-type=ALL

module load miniconda
conda activate thesis
python3 1M_new_vae.py > 1M_output.txt