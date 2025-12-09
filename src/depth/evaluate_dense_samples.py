# =========================================================
#  Dense depth evaluation on Kaggle (top-N most crowded images)
# =========================================================


# Этот файл — автоматический анализатор глубины на всём датасете, который:

# 🎯 Выбирает N самых сложных изображений (где больше всего объектов)
# 🎯 Сравнивает RealSense глубину с MiDaS глубиной
# 🎯 Считает метрики качества
# 🎯 Показывает визуализацию сравнения

# Короче — это оценка качества глубины на датасете, а не на одной картинке (как compare_depths.py).

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from skimage.metrics import structural_similarity as ssim

# MiDaS estimator (читается из твоего файла depth_estimator.py)
from depth_estimator import MiDaSDepthEstimator


BASE_PATH = "/kaggle/input/dataset/new_ds"


def normalize(depth):
    depth = depth.astype(np.float32)
    if depth.max() - depth.min() < 1e-6:
        return np.zeros_like(depth)
    return (depth - depth.min()) / (depth.max() - depth.min())


def compute_metrics(real, pred):
    real = normalize(real)
    pred = normalize(pred)

    mae = np.mean(np.abs(real - pred))
    rmse = np.sqrt(np.mean((real - pred) ** 2))

    # correlation requires flatten
    corr = np.corrcoef(real.flatten(), pred.flatten())[0, 1]

    ssim_val = ssim(real, pred, data_range=1.0)

    return mae, rmse, corr, ssim_val


def count_objects(label_path):
    if not os.path.exists(label_path):
        return 0
    try:
        with open(label_path, "r") as f:
            return len([l for l in f.readlines() if len(l.strip()) > 0])
    except:
        return 0


def evaluate_dense_dataset(base_path=BASE_PATH, top_n=5, device="cpu"):

    folders = sorted([f for f in os.listdir(base_path) if f.isdigit()])

    samples = []

    for folder in folders:
        img_paths = sorted(glob(os.path.join(base_path, folder, "images", "*.png")))

        for img_path in img_paths:
            file = os.path.basename(img_path)
            depth_path = os.path.join(base_path, folder, "depth", file)
            label_path = os.path.join(base_path, folder, "labels", file.replace(".png", ".txt"))

            if not os.path.exists(depth_path):
                continue

            obj_cnt = count_objects(label_path)
            samples.append((obj_cnt, img_path, depth_path))

    # выбираем top-N с наибольшим числом объектов
    samples.sort(key=lambda x: -x[0])
    selected = samples[:top_n]

    estimator = MiDaSDepthEstimator(device=device)

    for obj_cnt, rgb_path, depth_path in selected:
        print(f"\n============================")
        print(f"Image: {rgb_path}")
        print(f"Objects: {obj_cnt}")

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_real = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

        # MiDaS depth
        depth_midas = estimator.predict_from_array(rgb)

        # Normalize both
        real_n = normalize(depth_real)
        pred_n = normalize(depth_midas)

        # metrics
        mae, rmse, corr, ssim_val = compute_metrics(real_n, pred_n)

        print(f"MAE :  {mae:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"Corr:  {corr:.4f}")
        print(f"SSIM:  {ssim_val:.4f}")

        # Visualization inside kaggle output
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(rgb)
        ax[0].set_title(f"RGB ({obj_cnt} objects)")
        ax[0].axis("off")

        ax[1].imshow(real_n, cmap="gray")
        ax[1].set_title("RealSense Depth (normalized)")
        ax[1].axis("off")

        ax[2].imshow(pred_n, cmap="gray")
        ax[2].set_title("MiDaS Depth")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()


# ========= RUN =========

evaluate_dense_dataset(BASE_PATH, top_n=5, device="cpu")
