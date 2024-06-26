#!/bin/bash
#SBATCH --job-name=nell995-sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=log_test_%A_%a.out

PROJ_FOLDER=mpqepp

# Copy data to scratch
cp -r $HOME/$PROJ_FOLDER $TMPDIR
cd $TMPDIR/$PROJ_FOLDER

source activate hqeqs
wandb agent --count 1 dfdazac/mpqepp/9v7taup3