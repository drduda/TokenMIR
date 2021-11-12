#!/bin/bash
#SBATCH --job-name=bert_base
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpc3_gpu

srun python train.py classify --ds_path=/home/student/m/mduda/data/fma_medium_tokens --batch_size=64 --token_sequence_length=1024 --epochs=50 --d_model=512 --n_head=8 --dim_feed=1024 --dropout=0.2 --layers=4 --gpus=1 --precision=16
