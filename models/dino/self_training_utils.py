import numpy as np
import os
import cv2
import torch
import torchvision.transforms as transforms
from util import box_ops
import time
from torchvision.ops.boxes import batched_nms, box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


def _make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_unlabel_img(nestedtensor):
    images, masks = nestedtensor.decompose()
    b, c, h, w = images.shape
    unlabel_b = b // 2
    unlabel_img = images[unlabel_b:, :, :, :]
    label_img = images[:unlabel_b, :, :, :]
    return unlabel_img, label_img


def visualize_bbox(img_target, boxes, labels, scores, classname):
    """
    Visualize the bounding boxes on the image.

    Args:
        img_target (Tensor): The image to draw boxes on.
        boxes (Tensor): Bounding box coordinates of shape (N, 4).
        labels (Tensor): Labels for each detected object.
        scores (Tensor): Confidence scores for each detection.

    """
    to_pil = transforms.ToPILImage()
    img_target = to_pil(img_target[[2, 1, 0]].to(torch.uint8))

    fig, ax = plt.subplots(1)
    for i in range(len(boxes)):
        # Extract coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = boxes[i]

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        # Annotate the box with the label and score
        ax.text(x1, y1, f'{classname[labels[i]]}: {scores[i]:.2f}', fontsize=8,
                bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.imshow(img_target)
    plt.axis('off')  # Hide the axes
    plt.show(block=True)
    # plt.close(fig)


def deal_pesudo_label(unlabel_target_list, target_predict_results):
    unlabel_target_format_list = []
    for i in range(len(unlabel_target_list)):
        cache_unlabel_target_format = {}
        unlabel_target = unlabel_target_list[i]
        cache_unlabel_target_format.update(**target_predict_results[i])
        cache_unlabel_target_format['image_id'] = unlabel_target['image_id']
        cache_unlabel_target_format['area'] = unlabel_target['area']
        cache_unlabel_target_format['iscrowd'] = unlabel_target['iscrowd']
        cache_unlabel_target_format['orig_size'] = unlabel_target['orig_size']
        cache_unlabel_target_format['size'] = unlabel_target['size']
        unlabel_target_format_list.append(cache_unlabel_target_format)
    return unlabel_target_format_list


def rescale_pseudo_targets(unlabel_samples_img, results, nms_th=0.7):
    _b, _c, _h, _w = unlabel_samples_img.shape

    for k in range(len(results)):
        _h_real, _w_real = results[k]['size']
        results[k]['boxes'] = box_ops.box_cxcywh_to_xyxy(results[k]['boxes'])

        # （1）恢复为原图坐标
        results[k]['boxes'][:, [0, 2]] = results[k]['boxes'][:, [0, 2]] * _w
        results[k]['boxes'][:, [1, 3]] = results[k]['boxes'][:, [1, 3]] * _h

        # （2）NMS
        keep_inds = batched_nms(results[k]['boxes'],
                                results[k]['scores'],
                                results[k]['labels'], nms_th)[:100]

        for key in ['labels', 'boxes', 'scores']:
            results[k][key] = results[k][key][keep_inds]
        results[k]['boxes_decode'] = results[k]['boxes'].clone()
        # # （3）比例缩放
        results[k]['boxes'][:, [0, 2]] = results[k]['boxes'][:, [0, 2]] / _w_real
        results[k]['boxes'][:, [1, 3]] = results[k]['boxes'][:, [1, 3]] / _h_real
        results[k]['boxes'] = box_ops.box_xyxy_to_cxcywh(results[k]['boxes'])
    return results


def expand_boxes(boxes: torch.Tensor, factor: float = 1.2,
                 min_size: int = 64) -> torch.Tensor:
    """
    Expand boxes by a scale factor around center, with optional min size clamp and boundary clipping.

    Args:
        boxes (Tensor): [..., 4] in (x1, y1, x2, y2) format
        factor (float): expansion factor (e.g., 1.2 means 20% larger)
        min_size (int): minimum side length after expansion

    Returns:
        Tensor: expanded boxes [..., 4] (x1, y1, x2, y2)
    """
    x0, y0, x1, y1 = boxes.clone().unbind(-1)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = (x1 - x0) * factor
    h = (y1 - y0) * factor

    # Clamp minimum size
    w = torch.clamp(w, min=min_size)
    h = torch.clamp(h, min=min_size)

    new_x0 = cx - 0.5 * w
    new_y0 = cy - 0.5 * h
    new_x1 = cx + 0.5 * w
    new_y1 = cy + 0.5 * h

    expanded = torch.stack([new_x0, new_y0, new_x1, new_y1], dim=-1)

    return expanded

@torch.no_grad()
def update_threshold(results, model, clip_model,
                     unlabel_imgs_weak,  # 弱增强输入, Tensor[B, C, H, W]
                     unlabel_imgs_strong,  # 强增强输入, Tensor[B, C, H, W]（bbox 坐标一致）
                     args):
    unlabel_imgs_weak = Denormalize(unlabel_imgs_weak)
    unlabel_imgs_strong = Denormalize(unlabel_imgs_strong)

    device = unlabel_imgs_weak.device
    idx_list = []

    for k, result in enumerate(results):
        valid_idx = result['scores'] > 0.3
        for key in ['labels', 'boxes', 'scores', 'boxes_decode']:
            result[key] = result[key][valid_idx]
        if valid_idx.any():
            idx_list.append(k)

    for k, result in enumerate(results):
        boxes = result['boxes_decode'].clone()
        result['clip_img_feat_bg'] = clip_model.build_background_embedding(unlabel_imgs_weak[k], expand_boxes(boxes, args.shift_range))
        result['clip_img_feat_bg_strong_aug'] = clip_model.build_background_embedding(unlabel_imgs_strong[k], expand_boxes(boxes, args.shift_range))
        x0, y0, x1, y1 = boxes.unbind(-1)
        widths = x1 - x0
        heights = y1 - y0
        large_mask = widths * heights > args.small_size ** 2  # 小目标标记
        boxes = expand_boxes(boxes[large_mask], args.shift_range)
        if boxes.numel() == 0:
            result.update({
                'clip_img_feat': torch.empty((0, args.clip_dim), device=device),
                'clip_img_feat_strong_aug': torch.empty((0, args.clip_dim), device=device),
                'clip_labels': torch.empty((0,), device=device, dtype=torch.int64),
            })
            continue

        feat_weak = clip_model.encode_image(unlabel_imgs_weak[k], boxes)
        feat_strong = clip_model.encode_image(unlabel_imgs_strong[k], boxes)
        clip_labels = result['labels'][large_mask]
        result.update({
            'clip_img_feat': feat_weak,
            'clip_img_feat_strong_aug': feat_strong,
            'clip_labels': clip_labels,
        })
    return results, idx_list

    # visualize_bbox(img_target=unlabel_imgs_weak[k].cpu() * 255,  # Use your original image
    #                boxes=result['boxes_decode'].cpu(),  # Bounding boxes from result
    #                labels=result['labels'].cpu(),  # Labels from result
    #                scores=result['scores'].cpu(),
    #                classname=clip_model.class_names)  # Scores from result


@torch.no_grad()
def get_clip_feats(args, results, clip_model, labeled_imgs):
    labeled_imgs = Denormalize(labeled_imgs)
    device = labeled_imgs.device
    for k in range(len(results)):
        boxes = results[k]['boxes_decode'].clone()
        results[k]['clip_img_feat_bg'] = clip_model.build_background_embedding(labeled_imgs[k], expand_boxes(boxes, args.shift_range))
        x0, y0, x1, y1 = boxes.unbind(-1)
        widths = x1 - x0
        heights = y1 - y0
        large_mask = widths * heights > args.small_size ** 2  # 小目标标记
        boxes = expand_boxes(boxes[large_mask], args.shift_range)
        if boxes.numel() == 0:
            results[k].update({
                'clip_img_feat': torch.empty((0, args.clip_dim), device=device),
                'clip_labels': torch.empty((0,), device=device, dtype=torch.int64),
            })
            continue
        feat_weak = clip_model.encode_image(labeled_imgs[k], boxes)
        clip_labels = results[k]['labels'][large_mask]
        results[k].update({
            'clip_img_feat': feat_weak,
            'clip_labels': clip_labels,
        })
        # visualize_bbox(img_target=labeled_imgs[k] *255,  # Use your original image
        #                boxes=results[k]['boxes_decode'].cpu(),  # Bounding boxes from result
        #                labels=results[k]['labels'].cpu(),  # Labels from result
        #                scores=results[k]['labels'].cpu(),
        #                classname=clip_model.class_names)  # Scores from result
    return results


def spilt_output(output_dict):
    source_dict = {}
    pesudo_dict = {}
    for k, v in output_dict.items():
        if 'target' in k:
            pesudo_dict[k] = v
        else:
            source_dict[k] = v
    return source_dict, pesudo_dict


def get_valid_output(target_outputs, target_pseudo_labels, idx):
    valid_target_outputs = {}
    for k, v in target_outputs.items():
        if 'pred' in k:
            valid_target_outputs[k] = v[idx, :, :]
        elif 'aux_outputs_target' in k:
            cache_list = []
            for sub_v_dict in v:
                cache_dict = {}
                cache_dict['pred_logits'] = sub_v_dict['pred_logits'][idx, :, :]
                cache_dict['pred_boxes'] = sub_v_dict['pred_boxes'][idx, :, :]
                cache_list.append(cache_dict)
            valid_target_outputs[k] = cache_list

        elif 'interm_outputs_target' in k:
            cache_dict = {}
            cache_dict['pred_logits'] = v['pred_logits'][idx, :, :]
            cache_dict['pred_boxes'] = v['pred_boxes'][idx, :, :]
            valid_target_outputs[k] = cache_dict

        elif 'interm_outputs_for_matching_pre_target' in k:
            cache_dict = {}
            cache_dict['pred_logits'] = v['pred_logits'][idx, :, :]
            cache_dict['pred_boxes'] = v['pred_boxes'][idx, :, :]
            valid_target_outputs[k] = cache_dict

        else:
            assert '不存在的输出结果'
    target_pseudo_labels2 = [target_pseudo_labels[i] for i in idx]

    return valid_target_outputs, target_pseudo_labels2


# ======================可视化DEBUG使用=============================

def Denormalize(img):
    # 这是归一化的 mean 和std
    channel_mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
    channel_std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
    # 这是反归一化的 mean 和std
    MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
    STD = [1 / std for std in channel_std]
    # 归一化和反归一化生成器
    denormalizer = transforms.Normalize(mean=MEAN, std=STD)
    de_img = denormalizer(img)
    return de_img


def draw_img(img, unlabel_samples_img_strong_aug, data_dict, unlabel_target, save_dir):
    _h, _w = data_dict['size'].cpu().numpy()
    boxes = data_dict['boxes'].cpu().numpy()  # 中心点+宽高
    boxes[:, [0, 2]] *= _w
    boxes[:, [1, 3]] *= _h
    img = img.copy()
    img2 = img.copy()
    # ----for real labels
    boxes_label = unlabel_target['boxes'].cpu().numpy()  # 中心点+宽高
    boxes_label[:, [0, 2]] *= _w
    boxes_label[:, [1, 3]] *= _h

    for i, box in enumerate(boxes_label):
        cls = unlabel_target['labels'][i].cpu().numpy()
        x_c, y_c, w, h = [int(i) for i in box]
        x1, y1, x2, y2 = x_c - w // 2, y_c - h // 2, x_c + w // 2, y_c + h // 2
        img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, 'label.jpg'), img2)
    # -------------

    if unlabel_samples_img_strong_aug is not None:
        unlabel_samples_img_strong_aug = unlabel_samples_img_strong_aug.copy()
        for i, box in enumerate(boxes):
            cls = data_dict['labels'][i].cpu().numpy()
            x_c, y_c, w, h = [int(i) for i in box]
            x1, y1, x2, y2 = x_c - w // 2, y_c - h // 2, x_c + w // 2, y_c + h // 2
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            unlabel_samples_img_strong_aug = cv2.rectangle(unlabel_samples_img_strong_aug, (x1, y1), (x2, y2),
                                                           (0, 0, 255), 2)
        # cv2.imshow('a',img)
        # cv2.imshow('b',unlabel_samples_img_strong_aug)
        cv2.imwrite(os.path.join(save_dir, 'a.jpg'), img)
        cv2.imwrite(os.path.join(save_dir, 'b.jpg'), unlabel_samples_img_strong_aug)

        print('停止')
        time.sleep(5000000)


def show_pesudo_label_with_gt(unlabel_img_array, unlabel_pseudo_targets, unlabel_targets, idx_list,
                              unlabel_samples_img_strong_aug_array, save_dir='./show_pseudo'):
    _make_dir(save_dir)

    for n, idx in enumerate(idx_list):
        unlabel_img = unlabel_img_array[idx].detach().cpu()  # 根据索引找
        unlabel_samples_img_strong_aug = unlabel_samples_img_strong_aug_array[idx].detach().cpu()  # 根据索引找
        unlabel_pseudo_target = unlabel_pseudo_targets[idx]  # 根据索引找
        unlabel_target = unlabel_targets[idx]  # 根据索引找

        # 对图像进行反归一化
        unlabel_img = Denormalize(unlabel_img).numpy()
        unlabel_img *= 255.0
        unlabel_img = unlabel_img.transpose(1, 2, 0).astype(np.uint8)
        if unlabel_samples_img_strong_aug is not None:
            unlabel_samples_img_strong_aug = Denormalize(unlabel_samples_img_strong_aug).numpy()
            unlabel_samples_img_strong_aug *= 255.0
            unlabel_samples_img_strong_aug = unlabel_samples_img_strong_aug.transpose(1, 2, 0).astype(np.uint8)

        draw_img(unlabel_img, unlabel_samples_img_strong_aug, unlabel_pseudo_target, unlabel_target, save_dir)
