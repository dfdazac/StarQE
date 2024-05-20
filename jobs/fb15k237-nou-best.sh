#!/bin/bash
#SBATCH --job-name=mpqepp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=10:00:00
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=log_test_%A_%a.out

PROJ_FOLDER=mpqepp

# Copy data to scratch
cp -r $HOME/$PROJ_FOLDER $TMPDIR
cd $TMPDIR/$PROJ_FOLDER

source activate hqeqs
hqe train \
    --train-data=/*/0qual:* \
    --validation-data=/*/0qual:* \
    --test-data=/*/0qual:* \
    --use-wandb \
    --activation=relu \
    --batch-size=64 \
    --dataset=fb15k237 \
    --dropout=0.1 \
    --embedding-dim=512 \
    --epochs=30 \
    --learning-rate=0.0004269404596549132 \
    --message-weighting=attention \
    --num-layers=3 \
    --use-bias=True
