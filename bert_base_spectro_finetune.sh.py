#!/bin/bash
#SBATCH --job-name=bert_base_spectro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpc3_gpu

source ~/.bashrc
conda activate TokenMIR

srun python train.py pretrain_from_spectrograms --fma_dir=~/data --backbone_path=? --learning_rate=? --batch_size=64 --epochs=? --gpus=1 --precision=16 --snippet_length=1024 --n_mels=128 --n_fft=480 --hop_length=128 --fma_subset="medium"