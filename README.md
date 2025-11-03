# VLSDA: Vision-Language Model Supervised Domain Adaptation for Cross-Domain Object Detection in Remote Sensing

By Junhong Lu, Hao Chen.

This repository contains the implementation accompanying our paper VLSDA: Vision-Language Model Supervised Domain Adaptation for Cross-Domain Object Detection in Remote Sensing.
If you find it helpful for your research, please consider citing:

```
@ARTICLE{11222736,
  author={Lu, Junhong and Chen, Hao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={VLSDA: Vision-Language Model Supervised Domain Adaptation for Cross-Domain Object Detection in Remote Sensing}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Semantics;Feature extraction;Remote sensing;Prototypes;Object detection;Detectors;Adaptation models;Contrastive learning;Noise measurement;Sensors;Remote sensing;domain adaptive object detection;vision-language model;cross-domain alignment},
  doi={10.1109/TGRS.2025.3627226}}
```


![](/figs/fig1.jpg)

## Acknowledgment
This implementation is bulit upon [RemoteSensingTeacher](https://github.com/h751410234/RemoteSensingTeacher), [DATR](https://github.com/h751410234/DATR), [RS5M](https://github.com/om-ai-lab/RS5M) and [DGTRS](https://github.com/MitsuiChen14/DGTRS).

## Installation
Please refer to the instructions [here](dino.yml). We leave our system information for reference.

* OS: Ubuntu 20.04
* Python: 3.10.14
* CUDA: 12.4
* PyTorch: 2.4.0 (Other versions may work but are not guaranteed.)
* torchvision: 0.19.0

```
conda env create -f dino.yml
conda activate dino
cd models/dino/ops
bash make.sh
cd ../../..
```

Clone the repo and download pretrained VLM checkpoints from [RS5M](https://github.com/om-ai-lab/RS5M) or [DGTRS](https://github.com/MitsuiChen14/DGTRS):
```
git clone https://github.com/om-ai-lab/RS5M
git clone https://github.com/MitsuiChen14/DGTRS
```
After downloading, place them into the following folders:
```
VLSDA/
├── DGTRS/
│ ├── model/
│ ├── ckpt/
│ │ ├── LRSCLIP_ViT-B-16.pt
│ │ ├── LRSCLIP_ViT-L-14.pt
│ └── ...
├── GeoRSCLIP/
│ ├── ckpt/
│ │ ├── RS5M_ViT-B-32.pt
│ │ ├── RS5M_ViT-L-14.pt
│ └── ...
```

## Dataset Preparation
Please construct the datasets following these steps:

- Download the datasets following [RemoteSensingTeacher](https://github.com/h751410234/RemoteSensingTeacher).

- Convert the annotation files into COCO-format annotations.

- Modify the dataset path setting within the script [DAcoco.py](./datasets/DAcoco.py)

```
'dateset's name': {
    'train_img'  : '',  #train image dir
    'train_anno' : '',  #train coco format json file
    'val_img'    : '',  #val image dir
    'val_anno'   : '',  #val coco format json file
},
```
- Add domain adaptation direction within the script [__init__.py](./datasets/__init__.py). During training, the domain adaptation direction will be automatically parsed and corresponding data will be loaded. In our paper, we provide four adaptation directions for remote sensing scenarios.
```
DAOD_dataset = [
    'xView_to_DOTA',      #dateset's name1_to_dateset's name2
    'UCASAOD_to_CARPK',
    'CARPK_to_UCASAOD',
    'HRRSD_to_SSDD',
]
```

## Training / Evaluation
We provide training script as follows.
We divide the training process into two stages with single GPU. The settings for each stage can be found in the config folder.

(1) For the Burn-In stage:
```
python main.py --output_dir work_dir/xView_to_DOTA -c config/DA/RS/DINO_4scale_xView_to_DOTA.py --amp --options backend='open_clip' clip_model_name='ViT-B-32' pretrained='GeoRSCLIP/ckpt/RS5M_ViT-B-32.pt'
```
(2) For the Teacher-Student Mutual Learning stage, it is necessary to use the optimal model obtained from the first stage of training.
```
python main_teacher.py --output_dir work_dir/xView_to_DOTA -c config/DA/RS/DINO_4scale_xView_to_DOTA.py --amp --options backend='open_clip' clip_model_name='ViT-B-32' pretrained='GeoRSCLIP/ckpt/RS5M_ViT-B-32.pt'
```

We provide evaluation script to evaluate pre-trained model.
```
python main_teacher.py --output_dir work_dir/xView_to_DOTA -c config/DA/RS/DINO_4scale_xView_to_DOTA.py --amp --options backend='open_clip' clip_model_name='ViT-B-32' pretrained='GeoRSCLIP/ckpt/RS5M_ViT-B-32.pt' --eval --resume path/to/checkpoint
```

## Pre-trained models
We provide specific experimental configurations and pre-trained models to facilitate the reproduction of our results. 
You can learn the details of VLSDA through the paper, and please cite our papers if the code is useful for your papers. Thank you!

Task | mAP50 | Config | Model 
------------|-------| -------------| -------------
**xView_to_DOTA**  | 68.3% | [cfg](config/DA/RS/DINO_4scale_xView_to_DOTA.py) | [model](https://drive.google.com/file/d/1_rwvNWuJRGc5F5tdLPxuVCMBhYWvbcjt/view?usp=sharing)
**UCASAOD_to_CARPK** | 79.8% | [cfg](config/DA/RS/DINO_4scale_UCASAOD_to_CARPK.py) | [model](https://drive.google.com/file/d/1O9YPOZDIDCTRCyGhTCfet_Xu433eOlFs/view?usp=sharing)
**CARPK_to_UCASAOD** | 79.9% | [cfg](config/DA/RS/DINO_4scale_CARPK_to_UCASAOD.py) | [model](https://drive.google.com/file/d/1dCl64J3tml-QNv7vIEJN_C4A609ZsqAC/view?usp=sharing)
**HRRSD_to_SSDD** | 70.5% | [cfg](config/DA/RS/DINO_4scale_HRRSD_to_SSDD.py) | [model](https://drive.google.com/file/d/1XB6pZNcS6SZZ5oQVp-N3TzMCEnzqF-yz/view?usp=sharing)

## Reference
https://github.com/IDEA-Research/DINO

https://github.com/h751410234/RemoteSensingTeacher

https://github.com/h751410234/DATR

https://github.com/om-ai-lab/RS5M

https://github.com/MitsuiChen14/DGTRS
