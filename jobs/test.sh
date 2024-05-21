#!/bin/bash
#SBATCH --job-name=mpqepp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=log_test_%A_%a.out

source activate hqeqs
hqe train --dataset fb15k_237_betae -tr "/*/*:*" -va "/*/*:*" -te "/*/*:*" -e 3 -b 64 -w
