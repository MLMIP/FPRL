#!/bin/bash

cd /gaoxieping/dsh/FPRL/videomamba/downstream/STFT
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

torchrun \
    --nproc_per_node=1 \
    tools/train_net.py \
    --master_port=$((RANDOM + 10000)) \
    --config-file configs/STFT/kumc_R_50_STFT.yaml \
    OUTPUT_DIR /gaoxieping/dsh/FPRL/videomamba/out/Detection/fcrmamba_small_b64_seqlen2withWeaker__withteacher_MIX7
