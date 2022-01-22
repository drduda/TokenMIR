#!/bin/bash
#SBATCH --job-name=bert_base_token
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpc3_gpu

read varname

srun python train.py pretrain_from_tokens --ds_path=~/data/fma_medium_tokens --batch_size=64 --token_sequence_length=1600 --epochs=300 --d_model=768 --n_head=12 --dim_feed=1024 --dropout=0.1 --layers=4 --gpus=4 --precision=16 --masking_percentage=0.30 --name=$varname
