# VLSDA: Vision-Language Model Supervised Domain Adaptation for Cross-Domain Object Detection in Remote Sensing

By .

This repository contains the implementation accompanying our paper VLSDA: Vision-Language Model Supervised Domain Adaptation for Cross-Domain Object Detection in Remote Sensing.
This work is currently under review by IEEE Transactions on Geoscience and Remote Sensing. Once the paper is formally published, we will update this repository with the final citation.

![](/figs/fig1.jpg)

## Acknowledgment
This implementation is bulit upon [RemoteSensingTeacher](https://github.com/h751410234/RemoteSensingTeacher), [DATR](https://github.com/h751410234/DATR), [RS5M](https://github.com/om-ai-lab/RS5M) and [DGTRS](https://github.com/MitsuiChen14/DGTRS).

## Installation
Please refer to the instructions [here](dino.yml). We leave our system information for reference.

* OS: Ubuntu 20.04
* Python: 3.10.14
* CUDA: 12.4
* PyTorch: 2.4.0 (For building MultiScaleDeformableAttention)
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
python main.py --output_dir work_dir/xView_to_DOTA -c config/DA/RS_default/DINO_4scale_xView_to_DOTA.py --amp --options backend='open_clip' clip_model_name='ViT-B-32' pretrained='GeoRSCLIP/ckpt/RS5M_ViT-B-32.pt'
```
(2) For the Teacher-Student Mutual Learning stage, it is necessary to use the optimal model obtained from the first stage of training.
```
python main_teacher.py --output_dir work_dir/xView_to_DOTA -c config/DA/RS_default/DINO_4scale_xView_to_DOTA.py --amp --options backend='open_clip' clip_model_name='ViT-B-32' pretrained='GeoRSCLIP/ckpt/RS5M_ViT-B-32.pt'
```

We provide evaluation script to evaluate pre-trained model.
```
python main_teacher.py --output_dir work_dir/xView_to_DOTA -c config/DA/RS_default/DINO_4scale_xView_to_DOTA.py --amp --options backend='open_clip' clip_model_name='ViT-B-32' pretrained='GeoRSCLIP/ckpt/RS5M_ViT-B-32.pt' --eval
```

## Pre-trained models
Pre-trained models and configuration files will be released soon for public access.

## Reference
https://github.com/IDEA-Research/DINO
https://github.com/h751410234/RemoteSensingTeacher
https://github.com/h751410234/DATR
https://github.com/om-ai-lab/RS5M
https://github.com/MitsuiChen14/DGTRS