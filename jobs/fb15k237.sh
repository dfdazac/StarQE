#!/bin/bash
#SBATCH --job-name=mpqepp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=log_test_%A_%a.out

PROJ_FOLDER=mpqepp

# Copy data to scratch
cp -r $HOME/$PROJ_FOLDER $TMPDIR
cd $TMPDIR/$PROJ_FOLDER

source activate hqeqs
hqe train --dataset fb15k237 -tr "/*/*:100" -va "/*/*:100" -te "/*/*:100" -e 3 -b 64 -w