#!/bin/bash

#SBATCH --job-name=100K_VAE
#SBATCH --output=100K_gpu_job_output.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:2
#SBATCH --partition=gpu
#SBATCH --time=2-
#SBATCH --mail-user=josh.beasley@yale.edu
#SBATCH --mail-type=ALL

module load miniconda
conda activate thesis
python3 test_vae.py --train ../datasets/100K_rockyou_train.txt --save 100K_VAE_model --epochs 50 --batch 128 --gpus 2