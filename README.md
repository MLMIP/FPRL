# FPRL

![image](FPRL.png)

This repository provides the official PyTorch implementation of the paper **Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis**, which has been accepted by **CVPR 2026**.  


---

## Installation
We can install packages using provided `environment.yml`. For details, please refer to [EndoMamba](https://github.com/TianCuteQY/EndoMamba), and we thank the authors for their great work.

```shell
cd FPRL
conda env create -f environment.yml
conda activate FPRL
```

## Data Preparation
We gratefully acknowledge the use of datasets from [Endo-FM](https://github.com/med-air/Endo-FM) and [SV-RCNet](https://github.com/YuemingJin/SV-RCNet), and thank the authors for their valuable work.


## Weights
pretrain weight:

[Pretrain](https://pan.baidu.com/s/1rW4A8d7gig8dNCeIc3SafQ?pwd=k7dv)

downstream weight:

[Classification](https://pan.baidu.com/s/1VEMYXF3H6C2NoxD1B9B2Bw?pwd=ba2h)

[Segmentation](https://pan.baidu.com/s/1nXzaXkziD55zVSzuvTw9xQ?pwd=82qk)

[Detection](https://pan.baidu.com/s/1uKe3WOyYL72hrLg6kLLa6A?pwd=7tyv)

[Recognition](https://pan.baidu.com/s/16Cg2DN10W9fqChdyvlEfOg?pwd=9pr4)

## Pre-training
```shell
cd FPRL/videomamba
bash scripts/pretrain.sh
```

## Fine-tuning
```shell
# PolypDiag (Classification)
cd FPRL/videomamba
bash scripts/cls_ft.sh

# CVC-12k (Segmentation)
cd FPRL/videomamba
bash scripts/seg_ft.sh

# KUMC (Detection)
cd FPRL/videomamba
bash scripts/det_ft.sh

# Cholec80 (Recognition)
cd FPRL/videomamba
bash scripts/rec_ft.sh
```

## Acknowledgement
Our code is based on [Endo-FM](https://github.com/med-air/Endo-FM), [VideoMamba](https://github.com/OpenGVLab/VideoMamba), [EndoMamba](https://github.com/TianCuteQY/EndoMamba), and [MMCRL](https://github.com/MLMIP/MMCRL). Thanks them for releasing their codes.


## Citation
```
@inproceedings{zhang2026fprl,
  title={Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis},
  author={Zhang, Yuan and Dou, Sihao and Hu, Kai and Deng, Shuhua and Cao, Chunhong and Xiao, Fen and Gao, Xieping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
