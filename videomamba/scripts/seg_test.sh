#!/bin/bash

# 激活 conda
source /gaoxieping/miniconda3/etc/profile.d/conda.sh
conda activate endomamba
cd /gaoxieping/dsh/FCRMamba/videomamba/downstream/CVC-12kSegmentation

python train.py \
    --gpu 0 \
    --batch_size 1 \
    --model endomambaseg_small \
    --seed 11 \
    --n_skip 3\
    --test \
    --model endomambaseg_small \
    --pretrained_model_weights /gaoxieping/dsh/FCRMamba/videomamba/out/Segmentation/FCRMamba_MIX7_teacher_s11_skip3/best_model.pth \
    --root_path /gaoxieping/dsh/data/downstream/CVC-ClinicVideoDB/