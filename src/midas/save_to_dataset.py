import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from depth.depth_estimator import MiDaSDepthEstimator


# ============================
# CONFIG (LOCAL WINDOWS)
# ============================
BASE_PATH = r"D:\labels\new_ds"        # ✅ где лежит исходный датасет
OUT_BASE  = r"D:\labels\new_ds_midas"  # ✅ куда сохранить датасет с MiDaS


# ============================
# MiDaS WRAPPER (LOCAL)
# ============================
class MiDaSDepthEstimator:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[INFO] Using device: {self.device}")

        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

    def predict_from_array(self, rgb):
        input_tensor = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(rgb.shape[0], rgb.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return prediction.cpu().numpy()


# ============================
# MAIN PROCESS (LOCAL)
# ============================
def process_dataset(base_path, out_base):
    estimator = MiDaSDepthEstimator()

    folders = sorted([f for f in os.listdir(base_path) if f.isdigit()])

    for folder in folders:
        img_dir = os.path.join(base_path, folder, "images")
        real_depth_dir = os.path.join(base_path, folder, "depth")
        label_dir = os.path.join(base_path, folder, "labels")

        out_folder = os.path.join(out_base, folder)
        os.makedirs(out_folder, exist_ok=True)

        out_img_dir   = os.path.join(out_folder, "images")
        out_depth_dir = os.path.join(out_folder, "depth")
        out_midas_dir = os.path.join(out_folder, "depth_midas")
        out_lbl_dir   = os.path.join(out_folder, "labels")

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        os.makedirs(out_midas_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        images = sorted(glob(os.path.join(img_dir, "*.png")))
        depths = sorted(glob(os.path.join(real_depth_dir, "*.png")))
        labels = sorted(glob(os.path.join(label_dir, "*.txt")))

        print(f"[INFO] Folder {folder}: {len(images)} images")

        for img_path, d_path, l_path in zip(images, depths, labels):
            fname = os.path.basename(img_path)

            # ---------- COPY RGB, RealDepth, Labels ----------
            os.system(f'copy "{img_path}" "{out_img_dir}\\{fname}"')
            os.system(f'copy "{d_path}" "{out_depth_dir}\\{fname}"')
            os.system(f'copy "{l_path}" "{out_lbl_dir}\\{fname.replace(".png",".txt")}"')

            # ---------- MiDaS PREDICTION ----------
            rgb = cv2.imread(img_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            depth_pred = estimator.predict_from_array(rgb).astype(np.float32)

            # ✅ ПРАВИЛЬНАЯ НОРМАЛИЗАЦИЯ
            depth_pred = depth_pred - depth_pred.min()
            depth_pred = depth_pred / (depth_pred.max() + 1e-8)

            # ✅ ПРАВИЛЬНОЕ СОХРАНЕНИЕ В uint16 (БЕЗ ПОТЕРИ ТОЧНОСТИ)
            depth_pred_u16 = (depth_pred * 65535).astype(np.uint16)

            cv2.imwrite(os.path.join(out_midas_dir, fname), depth_pred_u16)

    print("\n✅ MiDaS depth generation FINISHED")
    print(f"✅ New dataset is in: {out_base}")


# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    process_dataset(BASE_PATH, OUT_BASE)
