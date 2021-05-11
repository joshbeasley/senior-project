#!/bin/bash

#SBATCH --job-name=1M_VAE
#SBATCH --output=1M_gpu_job_output.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:v100:2
#SBATCH --partition=gpu
#SBATCH --time=2-
#SBATCH --mail-user=josh.beasley@yale.edu
#SBATCH --mail-type=ALL

module load miniconda
conda activate thesis
python3 test_vae.py --train ../datasets/1M_rockyou_train.txt --save 1M_VAE_model --epochs 50 --batch 128 --gpus 2