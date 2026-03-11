#!/bin/bash

# 激活 conda
source /gaoxieping/miniconda3/etc/profile.d/conda.sh
conda activate fcrmamba
cd /gaoxieping/dsh/FCRMamba/videomamba

torchrun --nproc_per_node=2 video_sm/run_endomamba_pretraining.py \
  --batch_size 64 \
  --epochs 400 \
  --model "pretrain_endomamba_small_patch16_224" \
  --mix_datasets "MIX7" \
  --num_frames 2 \
  --pcf_total_frames 50 \
  --num_workers 4 \
  --teacher_model videomamba_small
