#!/bin/bash
#SBATCH --job-name=bert_base_token
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpc3_gpu

source ~/.bashrc
conda activate TokenMIR

srun python train.py finetune_from_tokens --ds_path=~/data/fma_medium_tokens --batch_size=? --token_sequence_length=1344 --epochs=40 --backbone_path=? --learning_rate=? --gpus=1 --precision=16 --name=?
