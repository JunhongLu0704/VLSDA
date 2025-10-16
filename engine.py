# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
import time
from typing import Iterable
from pycocotools.coco import COCO
from models.dino.EMA import ModelEMA
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from models.dino.dino import PostProcess
from models.dino.self_training_utils import *
from util.utils import to_device

buffer_names = ['global_proto', 'Amount', 'clip_global_proto', 'clip_Amount']


def train_one_epoch(model: torch.nn.Module, teacher_model: ModelEMA, clip_model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None, scaler=None):
    torch.cuda.empty_cache()
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    _cnt = 0
    #
    # global_proto = torch.zeros(9, 256).cuda(non_blocking=True)
    # global_amount = torch.zeros(9).cuda(non_blocking=True)
    for samples, targets, _, _ in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            label_samples_img = get_unlabel_img(samples)[1]
            targets = get_clip_feats(args, targets, clip_model, label_samples_img)
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            sync_buffers(model, buffer_names)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            # warmup_epochs = 8  # 前 5 个 epoch 平滑升权
            # if epoch < warmup_epochs:
            #     scale = (epoch + 1) / warmup_epochs  # 0.2 → 0.4 → ... → 1.0
            #     loss_dict['loss_VLPA'] *= scale
            #     loss_dict['loss_GCCA'] *= scale
            #     loss_dict['loss_backbone_DA'] *= scale

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()

            # # debug使用
            #  print('1111111111111')
            #  for name, param in model.named_parameters():
            #      if param.grad is None:
            #          print(name)

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)
        teacher_model.update(model)

        metric_logger.update(loss=loss_value, )
        # **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 300 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat, scaler


# =============加入self-training版本======================

def train_one_epoch_with_self_training(model: torch.nn.Module,
                                       teacher_model: ModelEMA,
                                       clip_model: torch.nn.Module,
                                       criterion: torch.nn.Module,
                                       data_loader: Iterable,
                                       data_loader_strong_aug: Iterable,
                                       optimizer: torch.optim.Optimizer,
                                       device: torch.device,
                                       epoch: int,
                                       max_norm: float = 0,
                                       wo_class_error=False,
                                       lr_scheduler=None,
                                       args=None,
                                       logger=None,
                                       ema_m=None,
                                       scaler=None):
    torch.cuda.empty_cache()
    # 初始化梯度缩放器
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    postprocessors = {'bbox': PostProcess()}

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f"Epoch: [{epoch}]"
    print_freq = 50
    _cnt = 0

    # ----用于记录损失
    cache_loss_array = []
    cache_self_training_loss_array = []

    # --------选择使用强增广的数据加载器
    use_data_loader = data_loader_strong_aug if data_loader_strong_aug is not None else data_loader

    for samples, source_labels, target_labels, samples_strong_aug in metric_logger.log_every(
            use_data_loader, print_freq, header, logger=logger):

        # 将数据移动到 device
        samples = samples.to(device, non_blocking=True)
        source_labels = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in source_labels]
        target_labels = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in target_labels]
        samples_strong_aug = samples_strong_aug.to(device, non_blocking=True)

        # -------------0. 获取强增广样本 -------------
        unlabel_samples_img_strong_aug = get_unlabel_img(samples_strong_aug)[0]
        # -------------1. 使用 teacher_model 对目标域图像（弱增广）推理 -------------
        unlabel_samples_img, label_samples_img = get_unlabel_img(samples)
        with torch.no_grad():
            target_predict_results = teacher_model.ema(unlabel_samples_img)
            source_labels = get_clip_feats(args, source_labels, clip_model, label_samples_img)

        orig_unlabel_target_sizes = torch.stack(
            [torch.tensor([1, 1], device=device) for i in range(len(target_labels))],
            dim=0)  # 保证坐标归一化
        # 转换预测结果为 bbox 格式（此时boxes格式为 [x1,y1,x2,y2]）
        target_predict_results = postprocessors['bbox'](target_predict_results, orig_unlabel_target_sizes,
                                                        not_to_xyxy=True, self_training=True)

        # -------------2. 固定阈值获取伪标签（Clip 前） -------------
        target_predict_results = deal_pesudo_label(target_labels, target_predict_results)
        target_pseudo_labels = rescale_pseudo_targets(unlabel_samples_img, target_predict_results)
        target_pseudo_labels, idx_list = update_threshold(target_pseudo_labels, model, clip_model,
                                                          unlabel_samples_img, unlabel_samples_img_strong_aug,
                                                          args)

        # -------------4. 学生模型对强增广图像进行推理 -------------
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples_strong_aug, source_labels, target_labels=target_pseudo_labels,
                                self_training_flag=True)
            else:
                outputs = model(samples_strong_aug, target_labels=target_pseudo_labels,
                                self_training_flag=True)

            sync_buffers(model, buffer_names)

            # 拆分输出为源域和目标域
            source_outputs, target_outputs = spilt_output(outputs)
            # 根据伪标签提取有效预测，并处理格式（可能会更新 Clip 后伪标签）
            valid_target_outputs, target_pseudo_labels = get_valid_output(
                target_outputs, target_pseudo_labels, idx_list
            )

            # -------------5. 计算损失 -------------
            weight_dict = criterion.weight_dict
            # source_outputs['da_output']['global_proto_DA']['clip_text_feats'] = clip_model.text_features
            loss_dict_source = criterion(source_outputs, source_labels, target_domain_flag=False)
            loss_dict_target = criterion(valid_target_outputs, target_pseudo_labels, target_domain_flag=True)

            # for source:  dino监督损失 + 域适应损失
            losses_source = sum(
                loss_dict_source[k] * weight_dict[k] for k in loss_dict_source.keys() if k in weight_dict)

            # for target:  dino监督损失
            losses_target = sum(
                loss_dict_target[k] * weight_dict[k] for k in loss_dict_target.keys() if k in weight_dict)

            # 若没有损失，则置零
            if losses_target == 0:
                losses_target = torch.tensor(0)

            losses = losses_source + losses_target * weight_dict['loss_self_training']

        # -------------6. 反向传播 -------------
        loss_dict_reduced = utils.reduce_dict(loss_dict_source)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)
        teacher_model.update(model)

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(**{name + '_thr': thr for thr, name in zip(cls_thr, clip_model.class_names)})

        _cnt += 1
        # if args.debug and _cnt % 15 == 0:
        #     print("BREAK!" * 5)
        #     break
    # 若存在loss权重衰减或匹配微调，也执行
    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # -----记录损失日志-----
    if utils.is_main_process():
        cache_loss_array.append(losses_source.detach().cpu().numpy())
        cache_self_training_loss_array.append(losses_target.detach().cpu().numpy())
        cache_loss_mean = np.asarray(cache_loss_array).mean()
        cache_ssod_loss_mean = np.asarray(cache_self_training_loss_array).mean()
        with open(os.path.join(args.output_dir, 'loss_txt'), 'a') as f:
            f.write(f"sup_loss: {cache_loss_mean} , ssod_loss: {cache_ssod_loss_mean}\n")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat, scaler


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False,
             args=None, logger=None):
    torch.cuda.empty_cache()
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_list = []
    for samples, _, targets in metric_logger.log_every(data_loader, 50, header, logger=logger):
        samples = samples.to(device, non_blocking=True)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                # outputs,global_proto,global_amount  = model(samples, targets)
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

        #     loss_dict = criterion(outputs, targets, return_indices=False)
        # weight_dict = criterion.weight_dict
        #
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))
        # **loss_dict_reduced_scaled,
        # **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)
        # res_cpu = {}
        # for tgt, out in zip(targets, results):
        #     img_id = tgt['image_id'].item()
        #     res_cpu[img_id] = {
        #         'boxes': out['boxes'].detach().cpu(),
        #         'scores': out['scores'].detach().cpu(),
        #         'labels': out['labels'].detach().cpu()
        #     }
        # # 把这一 batch 的结果合并到全局字典
        # pred_buffer.update(res_cpu)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            for tgt, out in zip(targets, results):
                img_id = int(tgt['image_id'])
                # results['boxes'] 为 xyxy；转 COCO xywh
                boxes = out['boxes'].detach().cpu().tolist()
                scores = out['scores'].detach().cpu().tolist()
                labels = out['labels'].detach().cpu().tolist()
                for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                    output_state_list.append({
                        "image_id": img_id,
                        "category_id": int(l),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s)
                    })

        _cnt += 1
        # if args.debug:
        #     if _cnt % 15 == 0:
        #         print("BREAK!" * 5)
        #         break

    # if coco_evaluator is not None:
    #     t1 = time.time()
    #     coco_evaluator.update(pred_buffer)
    #     print(f"evaluate time {time.time() - t1:.2f}s")
    if args.save_results:
        # gather detections from all ranks
        all_dets = utils.all_gather(output_state_list)  # list[list[dict]]
        if utils.is_main_process():
            flat = []
            for sub in all_dets:
                flat.extend(sub)
            # 推断 split 名称（若 dataset 有 image_set 属性）
            split_name = getattr(data_loader.dataset, 'image_set', 'val')
            out_dir = os.path.join(output_dir, "inference", split_name)
            os.makedirs(out_dir, exist_ok=True)
            out_json = os.path.join(out_dir, "bbox.json")
            import json
            with open(out_json, "w") as f:
                json.dump(flat, f)
            print(f"[save_results] wrote {len(flat)} detections → {out_json}")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # 示例使用
    category_names = [c['name'] for _, c in coco_evaluator.coco_gt.cats.items()]
    stats['bbox_ap_per_category'] = get_per_category_ap(coco_evaluator.coco_eval['bbox'], category_names)
    return stats, coco_evaluator


def get_per_category_ap(coco_eval, category_names, iou_threshold=0.50):
    # 获取精度数组
    precision = coco_eval.eval['precision']  # [IoU, Recall, Category, Area, MaxDets]
    iou_index = int((iou_threshold - 0.50) / 0.05)  # 计算对应的索引
    # 计算每个类别的AP
    # 通常，AP是在所有IoU阈值和Recall值上的平均
    # 我们通常只考虑 Area=all 和 MaxDets=100
    # 这里假设 Area index = 0 和 MaxDets index = 2
    ap_per_category = {}
    for idx, cat_id in enumerate(coco_eval.params.catIds):
        # 提取该类别的精度
        precision_cat = precision[iou_index, :, idx, 0, 2]  # [IoU, Recall]

        # 忽略 -1 的值（表示没有检测）
        precision_cat = precision_cat[precision_cat > -1]

        if precision_cat.size:
            ap = np.mean(precision_cat)
        else:
            ap = float('nan')

        category_name = category_names[idx]
        ap_per_category[category_name] = ap

    segm_ap_str = ', '.join([f"{cat}: {ap:.3f}" for cat, ap in ap_per_category.items()])
    return segm_ap_str


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None,
         logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device, non_blocking=True)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                    "image_id": int(image_id),
                    "category_id": l,
                    "bbox": b,
                    "score": s,
                }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)

    return final_res
