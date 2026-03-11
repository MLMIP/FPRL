#!/bin/bash

# 激活 conda
source /gaoxieping/miniconda3/etc/profile.d/conda.sh
conda activate fcrmamba
cd /gaoxieping/dsh/FCRMamba/videomamba/downstream/SV-RCNet

python train_singlenet_phase_1fc.py