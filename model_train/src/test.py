import cv2

import torch

import numpy as np
import matplotlib.pyplot as plt

import random

def show_n_predictions(model, samples, classes, depth_min, depth_max, img_size, device, n = 5, score_thresh=0.5):
    model.eval()
    chosen = random.sample(samples, min(n, len(samples)))

    for idx, sample in enumerate(chosen):
        print(f"\n============= SAMPLE {idx+1} =============")

        rgb = cv2.imread(sample["rgb"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (img_size, img_size))
        rgb_vis = rgb.copy()

        depth = cv2.imread(sample["depth"], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = cv2.resize(depth, (img_size, img_size))
        d = np.clip(depth, depth_min, depth_max)
        d_norm = (d - depth_min) / (depth_max - depth_min + 1e-6)

        depth_ch = d_norm[..., None] 

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
