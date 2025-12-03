import torch.nn as nn
import torch

import torch.nn.functional as F

from torch.utils.data import Dataset

from torchvision.models.resnet import resnet50

from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

import cv2
import numpy as np

class RGBDDetectionDataset(Dataset):
    def __init__(self, samples, img_size, DEPTH_MIN, DEPTH_MAX):
        self.samples = samples
        self.img_size = img_size
        self.DEPTH_MIN = DEPTH_MIN
        self.DEPTH_MAX = DEPTH_MAX

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        rgb = cv2.imread(item["rgb"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        rgb = rgb.astype(np.float32)  # [0,255]

        depth = cv2.imread(item["depth"], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = cv2.resize(depth, (self.img_size, self.img_size))

        d = np.clip(depth, self.DEPTH_MIN, self.DEPTH_MAX)
        d_norm = (d - self.DEPTH_MIN) / (self.DEPTH_MAX - self.DEPTH_MIN + 1e-6)  # [0,1]

        depth_ch = d_norm[..., None]  # (H,W,1)

        rgbd = np.concatenate([rgb / 255.0, depth_ch], axis=2)  # (H,W,4)
        rgbd = torch.from_numpy(rgbd).permute(2, 0, 1)  # (4,H,W)

        boxes = []
        labels = []
        for obj in item["objects"]:
            cx = obj["x_center"]
            cy = obj["y_center"]
            w = obj["width"]
            h = obj["height"]

            x1 = (cx - w / 2.0) * self.img_size
            y1 = (cy - h / 2.0) * self.img_size
            x2 = (cx + w / 2.0) * self.img_size
            y2 = (cy + h / 2.0) * self.img_size

            x1 = np.clip(x1, 0, self.img_size - 1)
            y1 = np.clip(y1, 0, self.img_size - 1)
            x2 = np.clip(x2, 0, self.img_size - 1)
            y2 = np.clip(y2, 0, self.img_size - 1)

            boxes.append([x1, y1, x2, y2])
            # +1, т.к. 0 — background
            labels.append(obj["cls_id"] + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        return rgbd, target
    

# Model

class ConvStem(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.stem(x) 
    
class CrossModalAttention(nn.Module):
    def __init__(self, channels=64, attn_size=8):
        super().__init__()
        self.channels = channels
        self.attn_size = attn_size

        self.pool_rgb = nn.AdaptiveAvgPool2d((attn_size, attn_size))
        self.pool_d = nn.AdaptiveAvgPool2d((attn_size, attn_size))

        self.q_rgb = nn.Conv2d(channels, channels, 1)
        self.k_d = nn.Conv2d(channels, channels, 1)
        self.v_d = nn.Conv2d(channels, channels, 1)

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, F_rgb, F_d):
        B, C, H, W = F_rgb.shape

        R = self.pool_rgb(F_rgb)
        D = self.pool_d(F_d) 

        Q = self.q_rgb(R)
        K = self.k_d(D)
        V = self.v_d(D)

        B, C, h, w = Q.shape
        T = h * w 

        Qf = Q.view(B, C, T)
        Kf = K.view(B, C, T)
        Vf = V.view(B, C, T)

        A = torch.bmm(Qf.transpose(1, 2), Kf)
        A = torch.softmax(A, dim=-1)

        F_att_flat = torch.bmm(Vf, A.transpose(1, 2))
        F_att_small = F_att_flat.view(B, C, h, w)   

        F_att = F.interpolate(F_att_small, size=(H, W), mode="bilinear", align_corners=False)

        fused = self.alpha * F_att + (1.0 - self.alpha) * F_rgb
        return fused
    
class RGBD_Backbone(nn.Module):
    def __init__(self, trainable_layers=5):
        super().__init__()

        self.rgb_stem = ConvStem(in_channels=3)
        self.depth_stem = ConvStem(in_channels=1)

        self.fusion = CrossModalAttention(channels=64, attn_size=8)

        self.body = resnet50(weights=None)

        self.body.conv1 = nn.Conv2d(
            64,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        layers_to_train = ["layer2", "layer3", "layer4"]
        for name, param in self.body.named_parameters():
            if not any([layer in name for layer in layers_to_train]):
                param.requires_grad = False

        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[
                256,   
                512,   
                1024,  
                2048, 
            ],
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :]

        F_rgb = self.rgb_stem(rgb)
        F_d = self.depth_stem(depth)

        F_fused = self.fusion(F_rgb, F_d)

        x = self.body.conv1(F_fused)
        x = self.body.bn1(x)
        x = self.body.relu(x)
        x = self.body.maxpool(x)

        C2 = self.body.layer1(x)
        C3 = self.body.layer2(C2)
        C4 = self.body.layer3(C3)
        C5 = self.body.layer4(C4)

        feats = self.fpn({"0": C2, "1": C3, "2": C4, "3": C5})
        return feats

