#!/bin/bash
#SBATCH --job-name=f_spectro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpc3_gpu

source ~/.bashrc
conda activate TokenMIR

srun python train.py finetune_from_spectrograms --fma_dir=~/data --backbone_path=? --learning_rate=? --batch_size=? --epochs=40 --gpus=1 --precision=16 --snippet_length=1344 --n_mels=86 --n_fft=512 --hop_length=128 --fma_subset="medium" --name=?
