#!/bin/bash

cd /gaoxieping/dsh/FPRL/videomamba/downstream/CVC-12kSegmentation

python train.py \
    --gpu 0 \
    --batch_size 1 \
    --model endomambaseg_small \
    --seed 11 \
    --n_skip 3\
    --test \
    --model endomambaseg_small \
    --pretrained_model_weights ../../out/Segmentation/FPRL_MIX7_teacher_s11_skip3/best_model.pth \
    --root_path /gaoxieping/dsh/data/downstream/CVC-ClinicVideoDB/
