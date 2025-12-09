# ============================================================
# compare_depths.py — визуальная оценка глубины (RGB + RealSense + MiDaS)
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

from multimodal_object_detection.src.depth_estimator import MiDaSDepthEstimator


def normalize_depth(depth):
    depth = depth.astype(np.float32)
    dmin, dmax = depth.min(), depth.max()
    if dmax - dmin < 1e-6:
        return np.zeros_like(depth)
    return (depth - dmin) / (dmax - dmin)


def compare_depths(rgb_path, depth_path, save_path="depth_visualization.png", device="cuda"):
    # ===== 1. Load RGB =====
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(f"RGB not found: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # ===== 2. Load RealSense depth =====
    depth_real = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_real is None:
        raise FileNotFoundError(f"Depth not found: {depth_path}")
    depth_real_norm = normalize_depth(depth_real)

    # ===== 3. MiDaS depth estimation =====
    estimator = MiDaSDepthEstimator(device=device)
    depth_midas = estimator.predict_from_array(rgb)
    depth_midas_norm = normalize_depth(depth_midas)

    # ===== 4. Visualization =====
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(rgb)
    ax[0].set_title("RGB")
    ax[0].axis("off")

    ax[1].imshow(depth_real_norm, cmap="gray")
    ax[1].set_title("RealSense Depth")
    ax[1].axis("off")

    ax[2].imshow(depth_midas_norm, cmap="gray")
    ax[2].set_title("MiDaS Estimated Depth")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[OK] Saved visualization: {save_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--save", default="depth_visualization.png")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    compare_depths(args.rgb, args.depth, args.save, args.device)
