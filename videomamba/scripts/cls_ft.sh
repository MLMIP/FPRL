#!/bin/bash
cd /gaoxieping/dsh/FPRL/videomamba/downstream/PolypDiagClassification
python eval_finetune.py \
    --data_path /gaoxieping/dsh/data/downstream/PolypDiag/ \
    --output_dir ../../out/Classification/
