import torch.nn as nn
import torch

from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.ops import MultiScaleRoIAlign

from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

class ConvStem(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x) 
    
class CrossModalSpatialAttention(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.att_conv = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, 3, 1)
        )
    def forward(self, f_rgb, f_r, f_m):
        cat = torch.cat([f_rgb, f_r, f_m], dim=1)   
        logits = self.att_conv(cat)                
        weights = torch.softmax(logits, dim=1)      
        w_rgb = weights[:,0:1]
        w_r   = weights[:,1:2]
        w_m   = weights[:,2:3]
        fused = w_rgb * f_rgb + w_r * f_r + w_m * f_m
        return fused

class FiveChannelStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_stem = ConvStem(3)
        self.r_stem   = ConvStem(1)
        self.m_stem   = ConvStem(1)
        self.att      = CrossModalSpatialAttention(channels=64)

    def forward(self, x):
        f_rgb = self.rgb_stem(x[:, :3])
        f_r   = self.r_stem  (x[:, 3:4])
        f_m   = self.m_stem  (x[:, 4:5])
        f_fused = self.att(f_rgb, f_r, f_m)
        return f_fused 

class RGBRM_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = FiveChannelStem()
        self.body = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.body.conv1 = nn.Conv2d(64, 64, 7, 2, 3, bias=False)

        for name, p in self.body.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                p.requires_grad = False

        self.fpn = FeaturePyramidNetwork(
            [256, 512, 1024, 2048],
            256,
            extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = 256

    def forward(self, x):
        x = self.stem(x)
        x = self.body.conv1(x)
        x = self.body.bn1(x)
        x = self.body.relu(x)
        x = self.body.maxpool(x)

        c2 = self.body.layer1(x)
        c3 = self.body.layer2(c2)
        c4 = self.body.layer3(c3)
        c5 = self.body.layer4(c4)

        feats = self.fpn({"0": c2, "1": c3, "2": c4, "3": c5})
        return feats

def rgbdd_attention_model(device, num_classes_with_bg):
    backbone = RGBRM_Backbone().to(device)

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
        image_mean=[0.5] * 5,
        image_std=[0.25] * 5
    ).to(device)

    print("Model Faster R-CNN RGBDD with attention created")

    return model