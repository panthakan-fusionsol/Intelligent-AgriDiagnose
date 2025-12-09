#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage example:
  /bin/python3 simple_gradcam.py \
    --weight /path/to/corn_disease_resnet18_model_xxx.pth \
    --img_size 448 \
    --images_dir ./suwan_cropped \
    --output_path ./gradcam_results/resnet18 \
    --model_type resnet18 \
    --alpha 0.35

Notes:
- รองรับทั้ง --weight และ --weight_path
- ถ้ารูปอยู่ใน root ตรง ๆ ก็ทำงานได้
- ถ้ามี subfolders จะรักษาโครงสร้างเดิมไว้ใน output
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


# -----------------------------
# Grad-CAM core
# -----------------------------
def compute_gradcam(model: nn.Module, image_tensor: torch.Tensor, layer_name: str):
    """
    image_tensor: [B,C,H,W] on device
    return: numpy heatmaps list shape [B, H', W'] normalized to [0,1]
    """
    layer = None
    for named, module in model.named_modules():
        if named == layer_name:
            layer = module
            break
    assert layer is not None, f"{layer_name} not found"

    feats = None
    grads = None

    def forward_hook(_module, _input, output):
        nonlocal feats
        feats = output  # [B,C,H',W']

    def backward_hook(_module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]  # [B,C,H',W']

    h1 = layer.register_forward_hook(forward_hook)
    # for PyTorch >= 1.8, register_full_backward_hook is correct
    h2 = layer.register_full_backward_hook(backward_hook)

    model.eval()
    preds: torch.Tensor = model(image_tensor)  # [B, num_classes]
    class_id = preds.softmax(dim=-1).argmax(dim=-1)  # [B]
    score = preds.gather(1, class_id.view(-1, 1)).sum()

    model.zero_grad(set_to_none=True)
    score.backward()

    h1.remove()
    h2.remove()

    # grads, feats: [B, C, H', W']
    weights = grads.mean(dim=(-2, -1))  # [B, C]
    cam = (weights[:, :, None, None] * feats).sum(dim=1)  # [B, H', W']
    cam = torch.relu(cam)

    B = cam.shape[0]
    cam_flat = cam.view(B, -1)
    maxv = cam_flat.max(dim=-1).values.clamp_min(1e-8)
    cam = (cam / maxv.view(B, 1, 1)).detach().cpu().numpy()
    return cam


# -----------------------------
# IO helpers
# -----------------------------
def list_image_groups(root: Path, exts=None):
    """
    เดินโฟลเดอร์ (recursive) แล้วคืนรายการเป็น groups ของรูปในแต่ละโฟลเดอร์
    Return: List of tuples: (dir_path, [image_paths_sorted])
    """
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    groups = []

    root = Path(root)
    if root.is_file():
        # Single image mode
        if root.suffix.lower() in exts:
            groups.append((root.parent, [root]))
        return groups

    for dirpath, _, filenames in os.walk(root):
        pics = []
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() in exts:
                pics.append(p)
        if pics:
            pics.sort()
            groups.append((Path(dirpath), pics))
    return groups


def load_batch_image_tensors(paths, tf):
    """
    โหลดรูปเป็น batch (PIL -> tensor -> normalize)
    """
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGBA");
        background = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img = Image.alpha_composite(background.convert("RGBA"), img)
        img = rgb_img.convert("RGB");
        imgs.append(tf(img))
    if len(imgs) == 0:
        return torch.empty(0)
    return torch.stack(imgs)  # [B,C,H,W]


def overlay_heatmap(img_paths, heatmaps, alpha=0.35, colormap=cv2.COLORMAP_JET):
    """
    img_paths: list[Path]
    heatmaps:  list/np.ndarray [B,H,W] (float 0..1)
    Return: list of BGR numpy images with overlay
    """
    # อ่านภาพต้นฉบับ
    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in img_paths]
    for k, im in enumerate(imgs):
        if im is None:
            raise RuntimeError(f"cv2.imread failed: {img_paths[k]}")

    outs = []
    for i in range(len(imgs)):
        hmap = cv2.resize(heatmaps[i], (imgs[i].shape[1], imgs[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        hmap_u8 = np.uint8(255 * np.clip(hmap, 0, 1))
        hmap_color = cv2.applyColorMap(hmap_u8, colormap)
        blended = cv2.addWeighted(imgs[i], 1 - alpha, hmap_color, alpha, 0)
        outs.append(blended)
    return outs


# -----------------------------
# Model helpers
# -----------------------------
def resolve_class_names(state_dict_like):
    """
    คืน class_names (list ตามลำดับ index) และ class_to_idx (dict)
    รองรับหลาย format ที่อาจพบใน checkpoint ต่าง ๆ
    """
    class_names = None
    class_to_idx = None

    if "class_names" in state_dict_like and isinstance(state_dict_like["class_names"], (list, tuple)):
        class_names = list(state_dict_like["class_names"])
        # Try to build class_to_idx if absent
        class_to_idx = state_dict_like.get("class_to_idx", None)
        if class_to_idx is None:
            class_to_idx = {name: i for i, name in enumerate(class_names)}

    elif "id2class" in state_dict_like and isinstance(state_dict_like["id2class"], dict):
        id2class = state_dict_like["id2class"]
        # keys may be "0","1",... or ints
        pairs = []
        for k, v in id2class.items():
            try:
                idx = int(k)
            except Exception:
                idx = k
            pairs.append((idx, v))
        pairs.sort(key=lambda x: x[0])
        class_names = [v for _, v in pairs]
        class_to_idx = {v: i for i, v in enumerate(class_names)}

    elif "class2id" in state_dict_like and isinstance(state_dict_like["class2id"], dict):
        c2i = state_dict_like["class2id"]
        # sort by index value
        items = sorted(c2i.items(), key=lambda kv: kv[1])
        class_names = [k for k, _ in items]
        class_to_idx = dict(c2i)

    elif "class_to_idx" in state_dict_like and isinstance(state_dict_like["class_to_idx"], dict):
        c2i = state_dict_like["class_to_idx"]
        items = sorted(c2i.items(), key=lambda kv: kv[1])
        class_names = [k for k, _ in items]
        class_to_idx = dict(c2i)

    else:
        raise ValueError("Cannot resolve classes from checkpoint: expected one of class_names/id2class/class2id/class_to_idx")

    # Normalize to lower ifคุณต้องการ; ที่นี่คงชื่อเดิมเพื่อความตรงตาม training
    return class_names, class_to_idx


def build_model(model_type: str, num_classes: int, state):
    """
    สร้างโมเดล + โหลดน้ำหนัก
    """
    if model_type.lower() == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type.lower() == "resnet50":
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    msd = state.get("model_state_dict", None)
    if msd is None:
        raise ValueError("Checkpoint missing key 'model_state_dict'")
    model.load_state_dict(msd, strict=True)
    return model


def default_layer_name(model_type: str) -> str:
    """
    คืนชื่อเลเยอร์ conv สุดท้ายสำหรับ GradCAM ตามสถาปัตยกรรม
    """
    mt = model_type.lower()
    if mt == "resnet18":
        return "layer4.1.conv2"   # BasicBlock
    elif mt == "resnet50":
        return "layer4.2.conv3"   # Bottleneck
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Grad-CAM for corn disease models")
    parser.add_argument("--weight_path", "--weight", dest="weight_path", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./gradcam_results")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_type", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--layer", type=str, default=None, help="ชื่อเลเยอร์ conv ที่จะใช้ทำ GradCAM; ถ้าไม่ระบุจะเลือกอัตโนมัติ")
    parser.add_argument("--alpha", type=float, default=0.35, help="ระดับความทึบของ Heatmap (0..1)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = Path(args.weight_path)
    images_root = Path(args.images_dir)
    out_root = Path(args.output_path)

    assert weight_path.exists(), f"weight not found: {weight_path}"
    assert images_root.exists(), f"images_dir not found: {images_root}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    state = torch.load(weight_path, map_location="cpu")
    class_names, class_to_idx = resolve_class_names(state)
    num_classes = len(class_names)

    # Build model
    model = build_model(args.model_type, num_classes, state).to(device).eval()

    # Decide layer
    layer_name = args.layer or default_layer_name(args.model_type)

    # Transform
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # Find image groups
    groups = list_image_groups(images_root)
    if not groups:
        raise RuntimeError(f"No images found under: {images_root}")

    print(f"Found {len(groups)} folder(s) with images under {images_root}")
    print(f"Using layer: {layer_name}")
    print(f"Saving to: {out_root}")

    # Iterate groups (folder by folder)
    for dir_path, img_paths in tqdm(groups, desc="Folders"):
        # relative subdir path to mirror into output
        rel_dir = dir_path.relative_to(images_root) if dir_path.is_relative_to(images_root) else Path(dir_path.name)
        out_dir = out_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if len(img_paths) == 0:
            continue

        # process by batches
        for i in tqdm(range(0, len(img_paths), args.batch_size), leave=False, desc=rel_dir.as_posix()):
            batch_paths = img_paths[i:i + args.batch_size]
            batch_tensor = load_batch_image_tensors(batch_paths, tf)
            if batch_tensor.numel() == 0:
                continue
            batch_tensor = batch_tensor.to(device)

            # compute heatmaps
            heatmaps = compute_gradcam(model, batch_tensor, layer_name=layer_name)

            # overlay and save
            overlays = overlay_heatmap(batch_paths, heatmaps, alpha=args.alpha, colormap=cv2.COLORMAP_JET)
            for j, p in enumerate(batch_paths):
                out_file = out_dir / f"{p.stem}_gradcam.png"
                ok = cv2.imwrite(str(out_file), overlays[j])
                if not ok:
                    raise RuntimeError(f"Failed to write file: {out_file}")

            # free a bit
            del batch_tensor
            torch.cuda.empty_cache()

    print("Done. Grad-CAM images are saved under:", out_root)


if __name__ == "__main__":
    main()
