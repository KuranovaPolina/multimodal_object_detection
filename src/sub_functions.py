import os
import cv2
import numpy as np

from glob import glob

def load_classes(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]
    
def read_yolo_label(classes, label_path):
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            objects.append(
                {
                    "cls_id": cls_id,
                    "class_name": classes[cls_id],
                    "x_center": x,
                    "y_center": y,
                    "width": w,
                    "height": h,
                }
            )
    return objects

def collect_samples(classes, base_path):
    samples = []
    subfolders = sorted([f for f in os.listdir(base_path) if f.isdigit()])

    for folder in subfolders:
        img_dir = os.path.join(base_path, folder, "images")
        depth_dir = os.path.join(base_path, folder, "depth")
        label_dir = os.path.join(base_path, folder, "labels")

        if not (
            os.path.isdir(img_dir)
            and os.path.isdir(depth_dir)
            and os.path.isdir(label_dir)
        ):
            continue

        img_files = sorted(glob(os.path.join(img_dir, "*.png")))

        for img_path in img_files:
            fname = os.path.basename(img_path)
            depth_path = os.path.join(depth_dir, fname)
            label_path = os.path.join(label_dir, fname.replace(".png", ".txt"))

            if not os.path.exists(depth_path):
                continue
            if not os.path.exists(label_path):
                continue

            objs = read_yolo_label(label_path)
            if len(objs) == 0:
                continue

            samples.append(
                {
                    "rgb": img_path,
                    "depth": depth_path,
                    "label": label_path,
                    "objects": objs,
                }
            )
    return samples

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
