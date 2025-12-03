import kagglehub

import os

import cv2
import torch
from torch.utils.data import DataLoader

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.ops import MultiScaleRoIAlign

import numpy as np
import matplotlib.pyplot as plt

import random

from sub_functions import load_classes, collect_samples, compute_depth_min_max, collate_fn
from RGBD_functions import RGBDDetectionDataset, RGBD_Backbone

IMG_SIZE = 256

BASE_PATH = "/kaggle/input/dataset/new_ds"
CLASSES_PATH = os.path.join(BASE_PATH, "merged_classes.txt")

def train_val_split(DEPTH_MIN, DEPTH_MAX, batch_size = 4, collate_fn = collate_fn):
    random.shuffle(samples)
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_dataset = RGBDDetectionDataset(train_samples, IMG_SIZE, DEPTH_MIN, DEPTH_MAX)
    val_dataset = RGBDDetectionDataset(val_samples, IMG_SIZE, DEPTH_MIN, DEPTH_MAX)

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

def Faster_RCNN_model_with_cross_model_attention(device):
    backbone = RGBD_Backbone().to(device)

    # FPN produces 5 maps: P2, P3, P4, P5, P6
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3", "pool"],  # 5 levels
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes_with_bg,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # 4 канала → 4 mean/std
        image_mean=[0.5, 0.5, 0.5, 0.5],
        image_std=[0.25, 0.25, 0.25, 0.25],
    ).to(device)

    print("Модель Faster R-CNN + Cross-Modal Attention (8x8) создана.")

    return model

def train(model, epochs = 10):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            n_batches += 1

        print(f"Epoch {epoch}/{epochs} | train_loss = {total_loss / n_batches:.4f}")

    torch.save(model.state_dict(), "rgbd_fasterrcnn_crossmodal_8x8.pth")
    print("Модель сохранена: rgbd_fasterrcnn_crossmodal_8x8.pth")

def show_n_predictions(model, samples, classes, DEPTH_MIN, DEPTH_MAX, n = 5, score_thresh=0.5):
    model.eval()
    chosen = random.sample(samples, min(n, len(samples)))

    for idx, sample in enumerate(chosen):
        print(f"\n============= SAMPLE {idx+1} =============")

        # читаем RGB/DEPTH так же, как в датасете
        rgb = cv2.imread(sample["rgb"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        rgb_vis = rgb.copy()

        depth = cv2.imread(sample["depth"], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))
        d = np.clip(depth, DEPTH_MIN, DEPTH_MAX)
        d_norm = (d - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN + 1e-6)

        depth_ch = d_norm[..., None]  # [0,1]

        rgbd = np.concatenate([rgb.astype(np.float32) / 255.0, depth_ch], axis=2)
        rgbd_t = torch.from_numpy(rgbd).permute(2, 0, 1).float().to(device)

        with torch.no_grad():
            outputs = model([rgbd_t])[0]

        boxes = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
    
        for box, label, score in zip(boxes, labels, scores):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box.astype(int)
            cls_id = int(label) - 1  # back to 0..C-1
            if cls_id < 0 or cls_id >= len(classes):
                continue
            cls_name = classes[cls_id]

            cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                rgb_vis,
                f"{cls_name} {score:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        plt.figure(figsize=(8, 8))
        plt.title("Predictions")
        plt.imshow(rgb_vis)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    classes = load_classes(CLASSES_PATH)
    print("Классы:", classes)

    num_classes = len(classes)
    num_classes_with_bg = num_classes + 1

    samples = collect_samples(classes, BASE_PATH)
    print("Всего валидных RGB+Depth+Label пар:", len(samples))
    print("Пример sample:", samples[0])

    DEPTH_MIN, DEPTH_MAX = compute_depth_min_max(samples)
    print("DEPTH_MIN, DEPTH_MAX:", DEPTH_MIN, DEPTH_MAX)

    train_loader, val_loader = train_val_split(DEPTH_MIN, DEPTH_MAX, batch_size = 4, collate_fn = collate_fn)

    model = Faster_RCNN_model_with_cross_model_attention(device)

    train(model, epochs = 10)

    show_n_predictions(model, samples, classes, DEPTH_MIN, DEPTH_MAX, n = 5, score_thresh=0.6)


