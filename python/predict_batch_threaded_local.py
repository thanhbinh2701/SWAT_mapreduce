#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch prediction script for Hadoop mapper.
Usage:
  python predict_batch_threaded_local.py <image_or_list.txt> <ckpt.pth> <class_folder_or_labels.txt> [model_cfg] [device]
"""

import os
import sys
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import argparse

# --- Constants ---
DEFAULT_BATCH = 32

def set_thread_limits():
    """Giới hạn số luồng tính toán để tránh quá tải node Hadoop."""
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

def safe_path(p: str):
    """Chuẩn hóa đường dẫn Hadoop local."""
    p = Path(p)
    if not p.exists():
        alt = Path(".") / p.name
        if alt.exists():
            return alt
    return p

def load_class_names(class_path: Path):
    """Load tên class từ thư mục hoặc file labels."""
    if not class_path.exists():
        raise FileNotFoundError(f"class path not found: {class_path}")
    if class_path.is_file():
        lines = [ln.strip() for ln in class_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return lines
    subdirs = sorted([d.name for d in class_path.iterdir() if d.is_dir()])
    if subdirs:
        return subdirs
    files = sorted([f.stem for f in class_path.iterdir() if f.is_file()])
    return files

def load_checkpoint(ckpt_path: Path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def build_model(ckpt_state, model_cfg, device, num_classes):
    """Khởi tạo model và load checkpoint."""
    if "vitb32" in model_cfg.lower():
        model = models.vit_b_32(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    model.to(device).eval()
    try:
        model.load_state_dict(ckpt_state, strict=False)
        print("✔ Model state loaded successfully", file=sys.stderr)
    except Exception as e:
        print(f"⚠ Warning: model load failed: {e}", file=sys.stderr)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return model, preprocess

def read_image_list(path: Path):
    """Đọc danh sách ảnh từ file hoặc 1 ảnh đơn."""
    if path.is_file() and path.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tif",".tiff"):
        return [str(path)]
    if path.is_file():
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return lines
    return [str(path)]

def batch_predict(model, preprocess, paths, device):
    """Dự đoán batch."""
    results = []
    tensors = []
    valid_paths = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            x = preprocess(img)
            tensors.append(x)
            valid_paths.append(p)
        except Exception as e:
            print(f"❌ Error loading image {p}: {e}", file=sys.stderr)
            results.append((p, 0, 0.0))  # Lỗi load -> trả về index 0
    if not tensors:
        return results
    batch = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_idx = torch.max(probs, dim=1)
        for i, path in enumerate(valid_paths):
            results.append((path, int(top_idx[i].item()), float(top_probs[i].item())))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("ckpt")
    parser.add_argument("class_folder")
    parser.add_argument("model_cfg", nargs="?", default="vitb32_openclip_laion400m")
    parser.add_argument("device", nargs="?", default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()

    set_thread_limits()
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    input_path = safe_path(args.input)
    ckpt_path = safe_path(args.ckpt)
    class_path = safe_path(args.class_folder)

    imgs = read_image_list(input_path)
    if not imgs:
        return

    try:
        class_names = load_class_names(class_path)
    except Exception as e:
        print(f"❌ Error loading class names: {e}", file=sys.stderr)
        sys.exit(1)

    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes", file=sys.stderr)

    ckpt_state = load_checkpoint(ckpt_path, device)
    model, preprocess = build_model(ckpt_state, args.model_cfg, device, num_classes)

    bs = max(1, args.batch_size)
    for i in range(0, len(imgs), bs):
        batch_paths = imgs[i:i+bs]
        preds = batch_predict(model, preprocess, batch_paths, device)
        for path, idx, prob in preds:
            # Luôn lấy tên class từ class_folder, không dùng 'unknown'
            cls = class_names[idx] if idx < len(class_names) else class_names[0]
            print(f"{path},{cls},{prob:.4f}")  # chỉ in stdout kết quả

if __name__ == "__main__":
    main()
