import cv2
import numpy as np
import torch
import time
# import pyrealsense2 as rs

import os

from glob import glob

from cfg.config import CLASSES_PATH, MODEL_PATH, IMG_SIZE, MODEL, DEPTH_MIN, DEPTH_MAX
from src.dataset_download_functions import load_classes
from src.rgbd_base_model import rgbd_base_model
from src.rgbd_attention_model import rgbd_attention_model
from src.rgbdd_attention_model import rgbdd_attention_model
from src.rgbdd_gating_model import rgbdd_gating_model

from src.predict import predict_rgbd_sample, predict_rgbdd_sample

img_size = IMG_SIZE
depth_min = DEPTH_MIN
depth_max = DEPTH_MAX

midas_model_type = "MiDaS_small"

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device:", device)

    classes = load_classes(CLASSES_PATH)
    print("Classes:", classes)

    num_classes_with_bg = len(classes) + 1

    if MODEL == "rgbd_base_model":
        model = rgbd_base_model(device, num_classes_with_bg)
    elif MODEL == "rgbd_attention_model":
        model = rgbd_attention_model(device, num_classes_with_bg)
    elif MODEL == "rgbdd_attention_model":
        model = rgbdd_attention_model(device, num_classes_with_bg)
    elif MODEL == "rgbdd_gating_model":
        model = rgbdd_gating_model(device, num_classes_with_bg)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    if MODEL == "rgbdd_attention_model" or MODEL == "rgbdd_gating_model":
        midas_model = torch.hub.load("intel-isl/MiDaS", midas_model_type)
        midas_model.to(device)
        midas_model.eval()

        print("midas_model created")

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")     

        if midas_model_type == "DPT_Large" or midas_model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

    print("ДОБАВИТЬ СЮДА РАБОТУ С REALSENCE")
    img_dir = "/Users/polinakuranova/.cache/kagglehub/datasets/radmilasegen/dataset/versions/1/new_ds/1/images/"
    depth_dir = "/Users/polinakuranova/.cache/kagglehub/datasets/radmilasegen/dataset/versions/1/new_ds/1/depth/"

    img_files = sorted(glob(os.path.join(img_dir, "*.png")))

    for img_path in img_files:
        rgb = cv2.imread(img_path)
        depth = cv2.imread(os.path.join(depth_dir, os.path.basename(img_path)), cv2.IMREAD_ANYDEPTH).astype(np.float32)
        
        start = time.time()

        if MODEL == "rgbd_base_model" or MODEL == "rgbd_attention_model":
            rgb_vis, depth = predict_rgbd_sample(classes, rgb, depth, model, img_size, depth_min, depth_max, device, score_thresh=0.75)
        elif MODEL == "rgbdd_attention_model" or MODEL == "rgbdd_gating_model":
            m_depth = midas_model(transform(rgb).to(device)).squeeze().detach().cpu().numpy() # ???
            rgb_vis, depth, m_depth = predict_rgbdd_sample(classes, rgb, depth, m_depth, model, img_size, depth_min, depth_max, device, score_thresh=0.75)

        end = time.time()

        print(f"Time: {end - start:.4f} seconds")

        if rgb_vis.shape[2] == 3:
            rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)

        cv2.imshow("Predictions", rgb_vis)
        cv2.imshow("Depth", depth)
        if MODEL == "rgbdd_attention_model" or MODEL == "rgbdd_gating_model":
            cv2.imshow("Midas depth", m_depth)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # if FROM_REALSENCE:
    #     pipeline = rs.pipeline()
    #     config = rs.config()
    #     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    #     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    #     pipeline.start(config)



    
