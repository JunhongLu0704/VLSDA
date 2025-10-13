# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
#---遥感
from .DAcoco import build_city_DA,build_sim2city_DA,build_city2BDD_DA, build_RS_DA



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args,strong_aug = False):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    #-----增添自然
    if args.dataset_file == 'city':
        return build_city_DA(image_set, args,strong_aug)

    if args.dataset_file == 'sim2city':
        return build_sim2city_DA(image_set, args,strong_aug)

    if args.dataset_file == 'city2bdd100k':
        return build_city2BDD_DA(image_set, args,strong_aug)

    DAOD_dataset = [
        #----------遥感场景------------------
        'xView_to_DOTA',
        'UCASAOD_to_CARPK',
        'CARPK_to_UCASAOD',
        'HRRSD_to_SSDD',
    ]
    if args.dataset_file in DAOD_dataset:
        return build_RS_DA(image_set, args, strong_aug)

    raise ValueError(f'dataset {args.dataset_file} not supported')
