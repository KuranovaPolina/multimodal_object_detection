# Это скрипт для сравнения глубины RealSense и глубины, предсказанной MiDaS, но:

# в отличие от предыдущих,
# он:

# ✅ берёт случайные изображения

# — выбирает num_samples случайных RGB–Depth пар из датасета.

# ✅ считает метрики качества предсказанной глубины:

# MAE — средняя абсолютная ошибка

# RMSE — корень среднеквадратичной ошибки

# Corr — корреляция

# SSIM — структурное сходство

# ✅ строит визуализацию

# — показывает 3 картинки:
# RGB → RealSense Depth → MiDaS Depth.

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from multimodal_object_detection.src.depth_estimator import MiDaSDepthEstimator
from glob import glob
from skimage.metrics import structural_similarity as ssim


def normalize(depth):
    depth = depth.astype(np.float32)
    if depth.max() - depth.min() < 1e-6:
        return np.zeros_like(depth)
    return (depth - depth.min()) / (depth.max() - depth.min())


def compute_metrics(real, pred):
    real = normalize(real)
    pred = normalize(pred)

    mae = np.mean(np.abs(real - pred))
    rmse = np.sqrt(np.mean((real - pred)**2))
    corr = np.corrcoef(real.flatten(), pred.flatten())[0, 1]
    ssim_val = ssim(real, pred, data_range=1.0)

    return mae, rmse, corr, ssim_val


def evaluate_random_images(base_path, num_samples=5, device="cpu"):
    folders = sorted([f for f in os.listdir(base_path) if f.isdigit()])

    pairs = []
    for folder in folders:
        images = sorted(glob(os.path.join(base_path, folder, "images", "*.png")))
        depths = sorted(glob(os.path.join(base_path, folder, "depth", "*.png")))

        for img, d in zip(images, depths):
            pairs.append((img, d))

    random.shuffle(pairs)
    selected = pairs[:num_samples]

    estimator = MiDaSDepthEstimator(device=device)

    for rgb_path, depth_path in selected:

        print(f"\n[INFO] === New sample ===")
        print(f"RGB:   {rgb_path}")
        print(f"Depth: {depth_path}")

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_real = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth_pred = estimator.predict_from_array(rgb)

        depth_real_n = normalize(depth_real)
        depth_pred_n = normalize(depth_pred)

        mae, rmse, corr, ssim_val = compute_metrics(depth_real_n, depth_pred_n)

        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Corr: {corr:.4f}")
        print(f"SSIM: {ssim_val:.4f}")

        # --- Visualization ---
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(rgb)
        ax[0].set_title("RGB")
        ax[0].axis("off")

        ax[1].imshow(depth_real_n, cmap="gray")
        ax[1].set_title("RealSense Depth")
        ax[1].axis("off")

        ax[2].imshow(depth_pred_n, cmap="gray")
        ax[2].set_title("MiDaS Depth")
        ax[2].axis("off")

        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--n", default=5, type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    evaluate_random_images(args.base, args.n, args.device)
