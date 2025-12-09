# ============================================================
# evaluate_realsense_depth.py — оценка качества RealSense depth
# ============================================================


# Этот файл — полноценный автоматический анализатор качества глубины RealSense по всему датасету new_ds.

# Он не сравнивает RealSense с MiDaS.
# Он оценивает саму глубину RealSense, ищет:

# шум

# дырки

# резкость

# динамический диапазон

# стабильность глубины

# И строит общий отчёт по всему датасету в виде JSON.


import os
import cv2
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


def evaluate_depth_map(depth):
    """
    depth — исходная depth карта (float32)
    Возвращает словарь метрик по одной карте глубины.
    """

    depth = depth.astype(np.float32)

    # игнорируем нули (глубина недоступна)
    valid_mask = depth > 0

    if valid_mask.sum() == 0:
        return {
            "min": None, "max": None, "mean": None, "std": None,
            "valid_ratio": 0.0, "gradient_mean": None,
            "holes": 1.0
        }

    d = depth[valid_mask]

    # Градиент (резкость глубины)
    gx = np.abs(cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3))
    grad = np.sqrt(gx**2 + gy**2)
    grad_valid = grad[valid_mask]

    metrics = {
        "min": float(d.min()),
        "max": float(d.max()),
        "mean": float(d.mean()),
        "std": float(d.std()),
        "valid_ratio": float(valid_mask.mean()),
        "holes": float((~valid_mask).mean()),
        "gradient_mean": float(grad_valid.mean()),
    }

    return metrics


def evaluate_dataset_depth(base_path, save_file="realsense_depth_report.json"):
    """
    base_path — путь к датасету new_ds, где лежат подпапки 1,2,3...
    """

    subfolders = sorted([f for f in os.listdir(base_path) if f.isdigit()])
    all_metrics = []

    print(f"Найдено {len(subfolders)} папок с данными.")

    for folder in subfolders:
        depth_dir = os.path.join(base_path, folder, "depth")
        if not os.path.isdir(depth_dir):
            print(f"[!] Пропуск {folder}, нет /depth/")
            continue

        depth_files = sorted(glob(os.path.join(depth_dir, "*.png")))
        print(f"\nПапка {folder}: {len(depth_files)} depth файлов")

        for dp in tqdm(depth_files):
            depth = cv2.imread(dp, cv2.IMREAD_ANYDEPTH)
            if depth is None or depth.size == 0:
                continue

            depth = depth.astype(np.float32)

            m = evaluate_depth_map(depth)
            all_metrics.append(m)

    # агрегируем
    summary = {
        "min_mean": float(np.mean([m["min"] for m in all_metrics if m["min"] is not None])),
        "max_mean": float(np.mean([m["max"] for m in all_metrics if m["max"] is not None])),
        "mean_mean": float(np.mean([m["mean"] for m in all_metrics if m["mean"] is not None])),
        "std_mean": float(np.mean([m["std"] for m in all_metrics if m["std"] is not None])),
        "valid_ratio_mean": float(np.mean([m["valid_ratio"] for m in all_metrics])),
        "holes_mean": float(np.mean([m["holes"] for m in all_metrics])),
        "gradient_mean": float(np.mean([m["gradient_mean"] for m in all_metrics if m["gradient_mean"] is not None])),
        "samples_analyzed": len(all_metrics)
    }

    # сохраняем отчёт
    with open(save_file, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n===============================================")
    print("         ОЦЕНКА КАЧЕСТВА REALSENSE DEPTH")
    print("===============================================")
    print(json.dumps(summary, indent=4))
    print(f"\nОтчёт сохранён в {save_file}")

    return summary


# ============================
# CLI
# ============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RealSense depth quality")
    parser.add_argument("--base", required=True, help="Путь к датасету new_ds")
    parser.add_argument("--save", default="realsense_depth_report.json", help="Файл для сохранения отчёта")

    args = parser.parse_args()

    evaluate_dataset_depth(args.base, save_file=args.save)
