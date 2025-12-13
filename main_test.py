import cv2
import numpy as np
import torch
import time
import pyrealsense2 as rs

from config.config import CLASSES_PATH, MODEL_PATH, IMG_SIZE, MODEL, DEPTH_MIN, DEPTH_MAX
from src.dataset_download_functions import load_classes
from src.rgbd_base_model import rgbd_base_model
from src.rgbd_attention_model import rgbd_attention_model
from src.rgbdd_attention_model import rgbdd_attention_model
from src.rgbdd_gating_model import rgbdd_gating_model
from src.midas import MiDaSDepthEstimator

from src.predict import predict_rgbd_sample, predict_rgbdd_sample

img_size = IMG_SIZE
depth_min = DEPTH_MIN
depth_max = DEPTH_MAX

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

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), weights_only=False)
    model.eval()

    if MODEL == "rgbdd_attention_model" or MODEL == "rgbdd_gating_model":
        estimator = MiDaSDepthEstimator(device)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    print("[INFO] RealSense camera started")

    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            
            start = time.time()

            if MODEL == "rgbd_base_model" or MODEL == "rgbd_attention_model":
                rgb_vis, depth = predict_rgbd_sample(classes, rgb, depth, model, img_size, depth_min, depth_max, device, score_thresh=0.75)
            elif MODEL == "rgbdd_attention_model" or MODEL == "rgbdd_gating_model":
                depth_pred = estimator.predict_from_array(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).astype(np.float32)
                depth_pred = depth_pred - depth_pred.min()
                depth_pred = depth_pred / (depth_pred.max() + 1e-8)

                depth_pred_u16 = (depth_pred * depth_max).astype(np.uint16)

                rgb_vis, depth, m_depth = predict_rgbdd_sample(classes, rgb, depth, depth_pred_u16, model, img_size, depth_min, depth_max, device, score_thresh=0.75)

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

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
