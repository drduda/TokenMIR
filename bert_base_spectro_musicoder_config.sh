#!/bin/bash
#SBATCH --job-name=bert_base_spectro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpc3_gpu

source ~/.bashrc
conda activate TokenMIR

srun python train.py classify_from_spectrograms --fma_dir=~/data --batch_size=64 --epochs=300 --d_model=768 --n_head=12 --dim_feed=1024 --dropout=0.1 --layers=4 --gpus=1 --precision=16 --snippet_length=1024 --n_mels=128 --n_fft=2048 --hop_length=1024 --fma_subset="medium"
