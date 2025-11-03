# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils
from clip_model import Clip_model
import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test, train_one_epoch_with_self_training
from models.dino import EMA
import warnings


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    # parser.add_argument('--dataset_file', default='xView_to_DOTA')
    parser.add_argument('--coco_path', type=str, default='')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument('--ignore_VLPA', action='store_true', )
    parser.add_argument('--ignore_GCCA', action='store_true', )
    parser.add_argument("--VLPA_loss", nargs="+", default=['loss_da', 'loss_clip_s', 'loss_clip_t'])
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args):
    # torch.cuda.set_per_process_memory_fraction(0.875, device=0)
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            warnings.warn("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'debug', None):
        args.debug = False
    args.start_epoch = args.burn_epochs

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False,
                          name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: " + ' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:' + str(n_parameters))
    logger.info(
        "params:\n" + json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    dataset_train = build_dataset(image_set='train', args=args)

    dataset_val = build_dataset(image_set='val', args=args)

    if args.strong_aug:  # 是否创建半监督强增广dataset
        dataset_train_strong_aug = build_dataset(image_set='train', args=args, strong_aug=True)
    else:
        dataset_train_strong_aug = None

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if dataset_train_strong_aug is not None:  # 半监督强增广使用
            sampler_train_strong_aug = DistributedSampler(dataset_train_strong_aug)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if dataset_train_strong_aug is not None:  # 半监督强增广使用
            sampler_train_strong_aug = torch.utils.data.RandomSampler(dataset_train_strong_aug)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_da, num_workers=args.num_workers,
                                   pin_memory=True, persistent_workers=args.num_workers > 0)
    data_loader_val = DataLoader(dataset_val, 4, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True, persistent_workers=args.num_workers > 0)

    if dataset_train_strong_aug is not None:  # 半监督强增广使用
        batch_sampler_train_strong_aug = torch.utils.data.BatchSampler(
            sampler_train_strong_aug, args.batch_size, drop_last=True)

        data_loader_train_strong_aug = DataLoader(dataset_train_strong_aug,
                                                  batch_sampler=batch_sampler_train_strong_aug,
                                                  collate_fn=utils.collate_fn_da, num_workers=args.num_workers,
                                                  pin_memory=True, persistent_workers=args.num_workers > 0)
    else:
        data_loader_train_strong_aug = None

    # build clip model
    cats = data_loader_train.dataset.source.coco.cats
    class_names = [cats[i]['name'] for i in sorted(cats.keys())]
    clip_model = Clip_model(device, class_names, amp=args.amp,
                            backend=args.backend,
                            clip_model_name=args.clip_model_name,
                            pretrained=args.pretrained)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                           steps_per_epoch=len(data_loader_train), epochs=args.epochs,
                                                           pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)  # Safe loading
        model_without_ddp.load_state_dict(checkpoint['model'])

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict(
            {k: v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        model_name = 'best_model.pth' if 'self_train' in str(output_dir) else 'best_ema_teacher.pth'
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        else:
            checkpoint = torch.load(os.path.join(output_dir, model_name), map_location='cpu', weights_only=True)
        if not args.distributed:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        else:  # DDP
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
            model_without_ddp.load_state_dict(state_dict, strict=True)
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir,
                                              wo_class_error=wo_class_error, args=args)

        best_model_fitness = test_stats['coco_eval_bbox'][1]
        best_model_bbox_ap_per_category = test_stats['bbox_ap_per_category']

        with open(output_dir / "log_eval.txt", 'w') as f:
            f.write(f'mAP50 -->  {best_model_fitness:.3f}\n')
            f.write(f'best_semi_ema -->  {best_model_bbox_ap_per_category}\n')

        if args.output_dir and utils.is_main_process():
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / 'latest.pth')

        json_path = os.path.join(output_dir, "eval_result_summary.json")
        eval_results = {
            'best_all_map50': test_stats['coco_eval_bbox'][1],
            'best_all_bbox_ap_per_category': test_stats['bbox_ap_per_category'],
        }
        with open(json_path, "w") as f_json:
            json.dump(eval_results, f_json, indent=4)

        return

    # ==========self-training准备工作，创建EMA Teacher模型=====================
    # ----EMA
    ema_teacher = EMA.ModelEMA(model, decay=args.ema_decay_teacher)  # teacher model
    # ---记录评估指标---
    ema_teacher_eval = []
    ema_teacher2_eval = []
    ema_student_eval = []
    eval_results = {
        'best_all_map50': 0,
        'best_all_epoch': 0,
        'best_all_bbox_ap_per_category': 0,
        'best_teacher_map50': 0,
        'best_teacher_epoch': 0,
        'best_teacher_bbox_ap_per_category': 0,
        'best_teacher2_map50': 0,
        'best_teacher2_epoch': 0,
        'best_teacher2_bbox_ap_per_category': 0,
        'best_student_map50': 0,
        'best_student_epoch': 0,
        'best_student_bbox_ap_per_category': 0,
    }
    # -----------
    print("Start training")
    # args.start_epoch = 36    #----修改初始位置用于DEBUG
    start_time = time.time()
    scaler = None
    for epoch in range(args.start_epoch, args.epochs):

        epoch_start_time = time.time()

        # ----1.不采用self-training训练
        if args.distributed:
            sampler_train.set_epoch(epoch)
            sampler_train_strong_aug.set_epoch(epoch)

        # ----准备工作
        if epoch == args.burn_epochs:
            # 考虑重新设置学习率，学习率*10（待实现）
            print('学习率:')
            for p in optimizer.param_groups:
                print(p['lr'])

            # （0）----------加载最优检测模型----------------
            checkpoint_path = os.path.join(os.path.dirname(output_dir), 'pre_train', 'best_ema_teacher.pth')
            if args.debug:
                checkpoint_path = 'checkpoint/CARPK_to_UCASAOD/LRSCLIP_ViT-B-16_full/pre_train/best_ema_teacher.pth'

            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint['ema_model']['clip_Amount'] *= 0.0
            if not args.distributed:
                model_without_ddp.load_state_dict(checkpoint['ema_model'], strict=True)
                ema_teacher.ema.load_state_dict(checkpoint['ema_model'], strict=True)
            else:  # DDP
                state_dict = {k.replace("module.", ""): v for k, v in checkpoint['ema_model'].items()}
                model_without_ddp.load_state_dict(state_dict, strict=True)
                ema_teacher.ema.load_state_dict(state_dict, strict=True)

            # # 评估，用于debug
            # _ = evaluate(
            #     ema_teacher.ema, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            #     wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            # )
            ema_teacher2 = EMA.CosineEMA(ema_teacher.ema, decay_start=args.ema_decay_best_model,
                                         total_epoch=args.epochs - args.burn_epochs)  # 使用较大的ema_decay加快模型学习效率

        # （2）----------self-training训练 with Domain Adaptation---------------)
        train_stats, scaler = train_one_epoch_with_self_training(
            model, ema_teacher, clip_model, criterion, data_loader_train, data_loader_train_strong_aug, optimizer,
            device,
            epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args,
            logger=(logger if args.save_log else None), scaler=scaler)

        # ----3.训练后，使用EMA更新teacher model和 best_ema_model
        # （1）---------- 更新teacher model -----------------
        # ema_teacher.update(model)
        # （2）---------- 更新best_ema_model model -----------------
        ema_teacher2.update_decay(epoch - args.burn_epochs)  # 更新semi_ema的decay
        ema_teacher2.update(ema_teacher.ema)

        # ----4.评估并保存模型
        # （1）---------- 官方保存每一个epoch student model-----------------
        if not args.onecyclelr:
            lr_scheduler.step()
        # if args.output_dir:
        #     if (epoch + 1) % args.save_checkpoint_interval == 0:
        #         weights = {
        #             'model': model_without_ddp.state_dict(),
        #             # 'optimizer': optimizer.state_dict(),
        #             # 'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             'args': args,
        #         }
        #         utils.save_on_master(weights, os.path.join(output_dir, 'checkpoint.pth'))

        # （2）---------- 评估结果-----------------
        # ----官方原版student eval
        test_stats_student, coco_evaluator_student = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        if args.output_dir and utils.is_main_process():
            ema_student_eval.append(test_stats_student['coco_eval_bbox'][1])
            # 记录结果
            with open(output_dir / "ema_student_eval.txt", 'a') as f:
                f.write(f"{test_stats_student['coco_eval_bbox'][1]:.4f}\n")
            if test_stats_student['coco_eval_bbox'][1] > eval_results['best_all_map50']:
                eval_results['best_all_map50'] = test_stats_student['coco_eval_bbox'][1]
                eval_results['best_all_epoch'] = epoch
                eval_results['best_all_bbox_ap_per_category'] = test_stats_student['bbox_ap_per_category']
                checkpoint_path = os.path.join(output_dir, 'best_model.pth')
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

            if test_stats_student['coco_eval_bbox'][1] > eval_results['best_student_map50']:
                eval_results['best_student_map50'] = test_stats_student['coco_eval_bbox'][1]
                eval_results['best_student_epoch'] = epoch
                eval_results['best_student_bbox_ap_per_category'] = test_stats_student['bbox_ap_per_category']

        test_stats_teacher, coco_evaluator_teacher = evaluate(
            ema_teacher.ema, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        if args.output_dir and utils.is_main_process():
            ema_teacher_eval.append(test_stats_teacher['coco_eval_bbox'][1])
            # 记录结果
            with open(output_dir / "ema_teacher_eval.txt", 'a') as f:
                f.write(f"{test_stats_teacher['coco_eval_bbox'][1]:.4f}\n")
            if test_stats_teacher['coco_eval_bbox'][1] > eval_results['best_all_map50']:
                eval_results['best_all_map50'] = test_stats_teacher['coco_eval_bbox'][1]
                eval_results['best_all_epoch'] = epoch
                eval_results['best_all_bbox_ap_per_category'] = test_stats_teacher['bbox_ap_per_category']
                checkpoint_path = os.path.join(output_dir, 'best_model.pth')
                utils.save_on_master({
                    'model': ema_teacher.ema.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

            if test_stats_teacher['coco_eval_bbox'][1] > eval_results['best_teacher_map50']:
                eval_results['best_teacher_map50'] = test_stats_teacher['coco_eval_bbox'][1]
                eval_results['best_teacher_epoch'] = epoch
                eval_results['best_teacher_bbox_ap_per_category'] = test_stats_teacher['bbox_ap_per_category']

        test_stats_teacher2, coco_evaluator_teacher2 = evaluate(
            ema_teacher2.ema, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        if args.output_dir and utils.is_main_process():
            ema_teacher2_eval.append(test_stats_teacher2['coco_eval_bbox'][1])
            # 记录结果
            with open(output_dir / "ema_teacher2_eval.txt", 'a') as f:
                f.write(f"{test_stats_teacher2['coco_eval_bbox'][1]:.4f}\n")
            if test_stats_teacher2['coco_eval_bbox'][1] > eval_results['best_all_map50']:
                eval_results['best_all_map50'] = test_stats_teacher2['coco_eval_bbox'][1]
                eval_results['best_all_epoch'] = epoch
                eval_results['best_all_bbox_ap_per_category'] = test_stats_teacher2['bbox_ap_per_category']
                checkpoint_path = os.path.join(output_dir, 'best_model.pth')
                utils.save_on_master({
                    'model': ema_teacher2.ema.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

            if test_stats_teacher2['coco_eval_bbox'][1] > eval_results['best_teacher2_map50']:
                eval_results['best_teacher2_map50'] = test_stats_teacher2['coco_eval_bbox'][1]
                eval_results['best_teacher2_epoch'] = epoch
                eval_results['best_teacher2_bbox_ap_per_category'] = test_stats_teacher2['bbox_ap_per_category']

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_student_{k}': v for k, v in test_stats_student.items()},
            **{f'test_teacher_{k}': v for k, v in test_stats_teacher.items()},
        }
        json_path = os.path.join(output_dir, "result_summary.json")
        with open(json_path, "w") as f_json:
            json.dump(eval_results, f_json, indent=4)
            # （3）记录日志

        with open(output_dir / "log_best.txt", 'w') as f:
            f.write('best_checkpoint -->  map50:%s , epoch:%s\n' % (
                eval_results['best_student_map50'], eval_results['best_student_epoch']))
            f.write(
                'best_semi_ema -->  map50:%s , epoch:%s\n' %
                (eval_results['best_teacher2_map50'], eval_results['best_teacher2_epoch']))

            res = eval_results['best_teacher2_bbox_ap_per_category']
            f.write(f'best_semi_ema -->  {res}\n')
            f.write('best_teacher -->  map50:%s , epoch:%s\n' % (
                eval_results['best_teacher_map50'], eval_results['best_teacher_epoch']))
            res = eval_results['best_teacher_bbox_ap_per_category']
            f.write(f'best_teacher -->  {res}\n')

        ep_paras = {
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
