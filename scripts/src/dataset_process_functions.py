import cv2
import numpy as np
import random
import torch

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

class RGBDDetectionDataset(Dataset):
    def __init__(self, samples, img_size, depth_min, depth_max):
        self.samples = samples
        self.img_size = img_size
        self.depth_min = depth_min
        self.depth_max = depth_max

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        rgb = cv2.imread(item["rgb"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        rgb = rgb.astype(np.float32)  # [0,255]

        depth = cv2.imread(item["depth"], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = cv2.resize(depth, (self.img_size, self.img_size))

        d = np.clip(depth, self.depth_min, self.depth_max)
        d_norm = (d - self.depth_min) / (self.depth_max - self.depth_min + 1e-6)  # [0,1]

        depth_ch = d_norm[..., None]  # (H,W,1)

        rgbd = np.concatenate([rgb / 255.0, depth_ch], axis=2)  # (H,W,4)
        rgbd = torch.from_numpy(rgbd).permute(2, 0, 1)  # (4,H,W)

        boxes = []
        labels = []
        for obj in item["objects"]:
            cx = obj["x_center"]
            cy = obj["y_center"]
            w = obj["width"]
            h = obj["height"]

            x1 = (cx - w / 2.0) * self.img_size
            y1 = (cy - h / 2.0) * self.img_size
            x2 = (cx + w / 2.0) * self.img_size
            y2 = (cy + h / 2.0) * self.img_size

            x1 = np.clip(x1, 0, self.img_size - 1)
            y1 = np.clip(y1, 0, self.img_size - 1)
            x2 = np.clip(x2, 0, self.img_size - 1)
            y2 = np.clip(y2, 0, self.img_size - 1)

            boxes.append([x1, y1, x2, y2])
            # +1, т.к. 0 — background
            labels.append(obj["cls_id"] + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        return rgbd, target

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

def train_val_split(samples, depth_min, depth_max, img_size, device, batch_size = 4):
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
