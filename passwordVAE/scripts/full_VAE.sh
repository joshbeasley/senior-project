#!/bin/bash

#SBATCH --job-name=full_VAE
#SBATCH --output=full_gpu_job_output.txt
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
python3 test_vae.py --train ../datasets/full_rockyou_train.txt --save full_VAE_model --epochs 5 --batch 64 --gpus 2