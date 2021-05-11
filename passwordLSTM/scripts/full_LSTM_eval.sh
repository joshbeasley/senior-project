#!/bin/bash

#SBATCH --job-name=full_LSTM_eval
#SBATCH --output=full_eval_gpu_job_output.txt
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
python3 test_lstm.py --test --chars 2000000 --temp 0.7 --load full_LSTM_model >> output_full.txt