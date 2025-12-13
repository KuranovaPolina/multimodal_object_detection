import torch.nn as nn
import torch

import torch.nn.functional as F

from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.ops import MultiScaleRoIAlign

from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

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
        self.body = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

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
            in_channels_list=[ 256, 512, 1024, 2048,],
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool(),)
        
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

def rgbd_attention_model(device, num_classes_with_bg):
    backbone = RGBD_Backbone().to(device)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3", "pool"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes_with_bg,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        image_mean=[0.5, 0.5, 0.5, 0.5],
        image_std=[0.25, 0.25, 0.25, 0.25],
    ).to(device)

    print("Model Faster R-CNN RGBD with attention created")

    return model
