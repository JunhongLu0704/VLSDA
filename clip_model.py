import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torchvision import transforms
import open_clip
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from torchvision.ops import box_iou


def visualize_cropped_tensors(cropped_tensors, save_path="imgs/cropped_grid.png", nrow=8):
    """
    将 cropped_tensors 保存为图像网格图
    cropped_tensors: Tensor(N, C, H, W)，必须为 [0, 1] 范围内
    save_path: 保存路径
    """
    # 反归一化，如果你有预处理的 Normalize 操作
    mean = torch.tensor([0.485, 0.456, 0.406], device=cropped_tensors.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=cropped_tensors.device).view(1, 3, 1, 1)
    unnorm = cropped_tensors * std + mean  # 反归一化

    grid = make_grid(unnorm.clamp(0, 1).cpu(), nrow=nrow)
    ndarr = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")

    plt.figure(figsize=(nrow * 2, (len(cropped_tensors) // nrow + 1) * 2))
    plt.imshow(ndarr)
    plt.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved cropped image grid to {save_path}")


def _convert_to_rgb(image):
    return image.convert('RGB')


def jitter_boxes(
        fg_boxes: torch.Tensor,  # (N, 4) 前景框，xyxy
        img_size,  # (H, W)
        times: int = 4,
        frac: float = 0.3,  # 抖动比例
        iou_thr: float = 0.2,  # IoU 阈值
) -> torch.Tensor:
    """
    从前景框中采样 `num_bg` 个负框，确保其与任何前景 IoU < τ。
    若采样不足，则补足整图框。
    返回: [num_bg, 4] Tensor
    """
    H, W = img_size
    device = fg_boxes.device
    N = fg_boxes.size(0)
    if N == 0:
        return torch.empty((0, 4), device=device)
    if N * times < 128:
        times = 128 // N + 1

    x1, y1, x2, y2 = fg_boxes.unbind(-1)
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box_scale = torch.stack([H * torch.ones_like(w), W * torch.ones_like(h), w, h], dim=-1)
    aug_scale = box_scale * frac  # [n,4]
    offset = torch.randn(times, fg_boxes.shape[0], 4, device=fg_boxes.device) * aug_scale[None, ...]
    new_box = torch.stack([cx, cy, w, h], dim=-1)[None, ...].expand(times, N, 4) + offset
    new_box = new_box.view(-1, 4)  # [N*K, 4]
    x_c, y_c, w, h = new_box.unbind(-1)
    new_box = torch.stack([(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=-1)
    new_box[..., [0, 2]] = new_box[..., [0, 2]].clamp(0, W - 1)
    new_box[..., [1, 3]] = new_box[..., [1, 3]].clamp(0, H - 1)
    # 过滤 IoU
    ious = box_iou(new_box, fg_boxes)  # [N*K, N]
    valid_mask = (ious.max(dim=1).values < iou_thr) & (w * h > 32 * 32)
    valid_cands = new_box[valid_mask]
    return valid_cands


class Clip_model(nn.Module):

    def __init__(
            self,
            device: torch.device,
            class_names: list,
            backend: str = 'open_clip',
            clip_model_name: str = "ViT-B-32",
            pretrained: str = "GeoRSCLIP/ckpt/RS5M_ViT-B-32.pt",
            amp: bool = False
    ):
        assert backend in ['open_clip', 'long_clip']
        super().__init__()
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.amp = amp

        # 1) 创建并载入open_clip
        #    若需要自定义预处理流程，可独立使用 transforms
        if backend == "open_clip":
            self.model = open_clip.create_model_and_transforms(clip_model_name, pretrained=None)[0]
            checkpoint = torch.load(pretrained, map_location="cpu", weights_only=True)
            msg = self.model.load_state_dict(checkpoint, strict=True)
            self.model = self.model.to(device)
            print(msg)

        elif backend == "long_clip":
            from DGTRS.model import longclip
            self.model = longclip.load(pretrained, device=device)[0]
        else:
            raise ValueError(f"Unknown backend {backend}")

        self.preprocess = transforms.Compose([
            # transforms.Resize(
            #     size=224,
            #     interpolation=transforms.InterpolationMode.BICUBIC,
            # ),
            # transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        self.preprocess_bg = transforms.Compose([
            transforms.Resize(64),
            transforms.GaussianBlur(kernel_size=11),
            transforms.Resize(
                size=224,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def build_background_embedding(
            self,
            image_tensor: torch.Tensor,  # (C,H,W) 0–1
            fg_boxes: torch.Tensor,  # (N,4) xyxy
    ):
        """
        根据前景框生成 jitter 背景框并提取 CLIP 特征.
        返回:
            bg_feats: Tensor[M, D]  -- M = fg_boxes.shape[0]
        """

        bg_boxes = torch.empty((0, 4), device=fg_boxes.device)
        if fg_boxes.numel():
            H, W = image_tensor.shape[-2:]
            bg_boxes = jitter_boxes(fg_boxes, (H, W))[:16]
        if bg_boxes.numel():
            bg_feats = self.encode_image(image_tensor, bg_boxes)
        else:
            with torch.cuda.amp.autocast(enabled=self.amp):
                image_tensor = self.preprocess_bg(image_tensor.unsqueeze(0))
                bg_feats = self.model.encode_image(image_tensor)
                bg_feats = F.normalize(bg_feats, dim=-1)  # [N, D]
        return bg_feats  # [N,D]，可后续均值或直接用作负样本

    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor, bboxes: torch.Tensor, batch_size: int = 16):
        """
        image_tensor: Tensor(C, H, W) [0,1]
        bboxes: Tensor(n,4)  (xyxy格式)
        return: Tensor(1, C)
        """
        if bboxes.shape[0] < 1:
            return torch.empty(0, self.text_features.size(0), dtype=bboxes.dtype, device=bboxes.device)


        batch_indices = torch.zeros((bboxes.shape[0], 1), dtype=bboxes.dtype, device=bboxes.device)
        bboxes = torch.cat([batch_indices, bboxes], dim=1)  # 形状 [n, 5]
        cropped_tensors = ops.roi_align(
            image_tensor.unsqueeze(0),  # 添加 batch 维度
            bboxes,
            output_size=(224, 224),
            spatial_scale=1.0,  # 保持原尺度
            sampling_ratio=2  # 设为 1，减少插值误差
        )
        cropped_tensors = self.preprocess(cropped_tensors)
        # visualize_cropped_tensors(cropped_tensors)
        all_feats = []
        with torch.cuda.amp.autocast(enabled=self.amp):
            for i in range(0, cropped_tensors.size(0), batch_size):
                batch = cropped_tensors[i:i + batch_size]  # [B, C, 224, 224]
                feats = self.model.encode_image(batch)  # [B, C]
                feats = F.normalize(feats, dim=-1)
                all_feats.append(feats)

        return torch.cat(all_feats, dim=0)  # [n, C]
