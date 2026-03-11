#!/bin/bash
# 激活 conda
source /gaoxieping/miniconda3/etc/profile.d/conda.sh
conda activate fcrmamba
cd /gaoxieping/dsh/FCRMamba/videomamba/downstream/CVC-12kSegmentation

python train.py \
    --exp FCRMamba_MIX7_teacher \
    --gpu 0 \
    --batch_size 1 \
    --model endomambaseg_small \
    --root_path /gaoxieping/dsh/data/downstream/CVC-ClinicVideoDB/ \
    --seed 11 \
    --n_skip 3\
    --base_lr 1e-4 \
    --out_dir /gaoxieping/dsh/FCRMamba/videomamba/out/Segmentation/
    # --wandb True
