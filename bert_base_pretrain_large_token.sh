#!/bin/bash
#SBATCH --job-name=p_token_xl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpc3_gpu

source ~/.bashrc
conda activate TokenMIR

srun python train.py pretrain_from_tokens --ds_size=large --ds_path=~/data/fma_large_tokens --batch_size=64 --token_sequence_length=1344 --epochs=150 --d_model=768 --n_head=12 --dim_feed=3072 --dropout=0.1 --layers=4 --gpus=1 --precision=16 --masking_percentage=0.30 --name=?
