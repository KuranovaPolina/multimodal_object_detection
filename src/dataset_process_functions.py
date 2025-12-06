import cv2
import numpy as np
import random

from torch.utils.data import DataLoader

from Faster_RCNN_model_with_cross_model_attention import RGBDDetectionDataset

def compute_depth_min_max(samples):
    d_min = float("inf")
    d_max = float("-inf")

    for s in samples:
        depth = cv2.imread(s["depth"], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        if depth.size == 0:
            continue
        m = depth.min()
        M = depth.max()
        if M <= 0:
            continue
        d_min = min(d_min, m)
        d_max = max(d_max, M)

    if not np.isfinite(d_min):
        d_min = 0.0
    if not np.isfinite(d_max) or d_max <= d_min:
        d_max = d_min + 1.0
    return d_min, d_max

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def train_val_split(samples, depth_min, depth_max, img_size, device, batch_size = 4, collate_fn = collate_fn):
    random.shuffle(samples)
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_dataset = RGBDDetectionDataset(train_samples, img_size, depth_min, depth_max)
    val_dataset = RGBDDetectionDataset(val_samples, img_size, depth_min, depth_max)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    print("Train size:", len(train_dataset), "Val size:", len(val_dataset))

    return train_loader, val_loader
