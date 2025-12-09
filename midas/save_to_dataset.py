import sys
import os
import cv2
import numpy as np
from glob import glob

# ============================
# ✅ ПРАВИЛЬНО ДОБАВЛЯЕМ src В PATH
# ============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

# ✅ ИМПОРТ ТВОЕГО MiDaS
from depth.depth_estimator import MiDaSDepthEstimator


# ============================
# CONFIG (LOCAL WINDOWS)
# ============================
BASE_PATH = r"D:\labels\new_ds"          # где лежит исходный датасет
OUT_BASE  = r"D:\labels\new_ds_midas"    # куда сохраняем с MiDaS


# ============================
# MAIN PROCESS (LOCAL)
# ============================
def process_dataset(base_path, out_base):
    estimator = MiDaSDepthEstimator()

    folders = sorted([f for f in os.listdir(base_path) if f.isdigit()])

    for folder in folders:
        img_dir   = os.path.join(base_path, folder, "images")
        depth_dir = os.path.join(base_path, folder, "depth")
        label_dir = os.path.join(base_path, folder, "labels")

        out_folder  = os.path.join(out_base, folder)
        out_img_dir = os.path.join(out_folder, "images")
        out_dep_dir = os.path.join(out_folder, "depth")
        out_mds_dir = os.path.join(out_folder, "depth_midas")
        out_lbl_dir = os.path.join(out_folder, "labels")

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_dep_dir, exist_ok=True)
        os.makedirs(out_mds_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        images = sorted(glob(os.path.join(img_dir, "*.png")))
        depths = sorted(glob(os.path.join(depth_dir, "*.png")))
        labels = sorted(glob(os.path.join(label_dir, "*.txt")))

        print(f"[INFO] Folder {folder}: {len(images)} images")

        for img_path, d_path, l_path in zip(images, depths, labels):
            fname = os.path.basename(img_path)

            # ---------- COPY RGB, RealDepth, Labels ----------
            os.system(f'copy "{img_path}" "{out_img_dir}\\{fname}"')
            os.system(f'copy "{d_path}" "{out_dep_dir}\\{fname}"')
            os.system(f'copy "{l_path}" "{out_lbl_dir}\\{fname.replace(".png",".txt")}"')

            # ---------- MiDaS PREDICTION ----------
            rgb = cv2.imread(img_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            depth_pred = estimator.predict_from_array(rgb).astype(np.float32)

            # ✅ НОРМАЛИЗАЦИЯ
            depth_pred = depth_pred - depth_pred.min()
            depth_pred = depth_pred / (depth_pred.max() + 1e-8)

            # ✅ СОХРАНЕНИЕ В uint16 БЕЗ ПОТЕРЬ
            depth_pred_u16 = (depth_pred * 65535).astype(np.uint16)

            cv2.imwrite(os.path.join(out_mds_dir, fname), depth_pred_u16)

    print("\n✅ MiDaS depth generation FINISHED")
    print(f"✅ New dataset is in: {out_base}")


# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    process_dataset(BASE_PATH, OUT_BASE)
