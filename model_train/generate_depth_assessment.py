import os
import torch
import cv2
import matplotlib.pyplot as plt

from glob import glob

from cfg.config import BASE_PATH, PATH_FOR_ASS_DEPTH

model_type = "MiDaS_small"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    midas_model = torch.hub.load("intel-isl/MiDaS", model_type)
    midas_model.to(device)
    midas_model.eval()

    print("midas_model created")

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")   

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform 

    subfolders = sorted([f for f in os.listdir(BASE_PATH) if f.isdigit()])
    for folder in subfolders:
        img_dir = os.path.join(BASE_PATH, folder, "images")

        if not os.path.isdir(img_dir):
            continue        

        img_files = sorted(glob(os.path.join(img_dir, "*.png")))

        for img_path in img_files:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         

            input_batch = transform(img).to(device)

            with torch.no_grad():
                prediction = midas_model(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()

            new_path = os.path.join(PATH_FOR_ASS_DEPTH + os.path.basename(img_path))

            plt.imsave(new_path, output, cmap='gray')
