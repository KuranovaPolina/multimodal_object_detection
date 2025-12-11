import cv2
import torch

import numpy as np


def predict_rgbd_sample(classes, rgb, depth, model, img_size, depth_min, depth_max, device, score_thresh=0.5):
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size))
    rgb_vis = rgb.copy()

    depth = cv2.resize(depth, (img_size, img_size))
    d = np.clip(depth, depth_min, depth_max)
    d_norm = (d - depth_min) / (depth_max - depth_min + 1e-6)

    depth_ch = d_norm[..., None] 

    rgbd = np.concatenate([rgb.astype(np.float32) / 255.0, depth_ch], axis=2)
    rgbd_t = torch.from_numpy(rgbd).permute(2, 0, 1).float().to(device)

    with torch.no_grad():
        outputs = model([rgbd_t])[0]

    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cls_id = int(label) - 1  # back to 0..C-1
        if cls_id < 0 or cls_id >= len(classes):
            continue
        cls_name = classes[cls_id]

        cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            rgb_vis,
            f"{cls_name} {score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    return rgb_vis, depth

def predict_rgbdd_sample(classes, rgb, depth, m_depth, model, img_size, depth_min, depth_max, device, score_thresh=0.5):
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.uint8)
    dR = depth.astype(np.uint8)
    dM = m_depth.astype(np.float32)

    h0, w0 = rgb.shape[:2]

    scale = min(img_size / h0, img_size / w0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)

    rgb_rs = cv2.resize(rgb, (new_w, new_h))
    dR_rs  = cv2.resize(dR,  (new_w, new_h))
    dM_rs  = cv2.resize(dM,  (new_w, new_h))

    pad_w = img_size - new_w
    pad_h = img_size - new_h
    pad_left = pad_w // 2
    pad_top  = pad_h // 2

    rgb_pad = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    dR_pad  = np.zeros((img_size, img_size), dtype=np.float32)
    dM_pad  = np.zeros((img_size, img_size), dtype=np.float32)

    rgb_pad[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = rgb_rs
    dR_pad [pad_top:pad_top+new_h, pad_left:pad_left+new_w] = dR_rs
    dM_pad [pad_top:pad_top+new_h, pad_left:pad_left+new_w] = dM_rs

    rgb_norm = rgb_pad.astype(np.float32) / 255.0
    dR_norm  = (dR_pad - depth_min) / (depth_max - depth_min + 1e-6)
    dM_norm  = (dM_pad - depth_min) / (depth_max - depth_min + 1e-6)

    x = np.concatenate(
        [rgb_norm, dR_norm[..., None], dM_norm[..., None]],
        axis=2
    )
    x_t = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model([x_t[0]])[0]

    boxes  = out["boxes"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()

    vis = rgb_pad.copy()

    for box, lbl, sc in zip(boxes, labels, scores):
        if sc < score_thresh:
            continue
        x1,y1,x2,y2 = box.astype(int)
        cls_id = int(lbl) - 1
        if cls_id < 0 or cls_id >= len(classes):
            continue
        name = classes[cls_id]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            vis,
            f"{name} {sc:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2)
        
    return vis, dR_rs, dM_rs
