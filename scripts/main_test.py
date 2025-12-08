import cv2
import numpy as np
import torch
# import pyrealsense2 as rs

import os

from glob import glob

from cfg.config import CLASSES_PATH, MODEL_PATH, IMG_SIZE
from src.dataset_download_functions import load_classes
from src.models import Faster_RCNN_model_with_cross_model_attention

from src.test import predict_1_sample

FROM_REALSENCE = False

img_size = IMG_SIZE

depth_min = 0
depth_max = 65535.0

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    classes = load_classes(CLASSES_PATH)
    print("Classes:", classes)

    num_classes = len(classes) + 1

    model = Faster_RCNN_model_with_cross_model_attention(device, classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    if FROM_REALSENCE == False:
        img_dir = "/Users/polinakuranova/.cache/kagglehub/datasets/radmilasegen/dataset/versions/1/new_ds/1/images/"
        depth_dir = "/Users/polinakuranova/.cache/kagglehub/datasets/radmilasegen/dataset/versions/1/new_ds/1/depth/"

        img_files = sorted(glob(os.path.join(img_dir, "*.png")))

    else:
        print("ДОБАВИТЬ СЮДА РАБОТУ С REALSENCE")

    # добавить обработку глубины в midas (?)

    for img_path in img_files:
        rgb = cv2.imread(img_path)
        depth = cv2.imread(os.path.join(depth_dir, os.path.basename(img_path)), cv2.IMREAD_ANYDEPTH).astype(np.float32)

        rgb_vis, depth = predict_1_sample(classes, rgb, depth, model, img_size, depth_min, depth_max, device, score_thresh=0.75)

        if rgb_vis.shape[2] == 3:
            rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)

        cv2.imshow("Predictions", rgb_vis)
        cv2.imshow("Depth", depth)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # if FROM_REALSENCE:
    #     pipeline = rs.pipeline()
    #     config = rs.config()
    #     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    #     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    #     pipeline.start(config)

    # else:
    #     pass


    
