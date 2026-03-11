#!/bin/bash

# 激活 conda
source /gaoxieping/miniconda3/etc/profile.d/conda.sh
conda activate fcrmamba
cd /gaoxieping/dsh/FCRMamba/videomamba/downstream/PolypDiagClassification

python eval_finetune1.py \
    --data_path /gaoxieping/dsh/data/downstream/PolypDiag/ \
    --output_dir /gaoxieping/dsh/FCRMamba/videomamba/out/Classification/