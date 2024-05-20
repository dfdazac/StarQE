#!/bin/bash
#SBATCH --job-name=mpqepp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=log_test_%A_%a.out

PROJ_FOLDER=mpqepp

# Copy data to scratch
cp -r $HOME/$PROJ_FOLDER $TMPDIR
cd $TMPDIR/$PROJ_FOLDER

source activate hqeqs
wandb agent --count 1 dfdazac/mpqepp/z875awfz