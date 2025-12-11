import torch.nn as nn
import torch

from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.ops import MultiScaleRoIAlign

from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

class ConvStem(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)
    
class CrossModalGating(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, A, B):
        return self.alpha * A + (1.0 - self.alpha) * B

class FiveChannelStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = ConvStem(3)
        self.r   = ConvStem(1)
        self.m   = ConvStem(1)
        self.g1  = CrossModalGating()
        self.g2  = CrossModalGating()

    def forward(self, x):
        f_rgb = self.rgb(x[:, :3])
        f_r   = self.r  (x[:, 3:4])
        f_m   = self.m  (x[:, 4:5])

        f_rgb_r = self.g1(f_rgb, f_r)
        f_all   = self.g2(f_rgb_r, f_m)
        return f_all

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

def rgbdd_gating_model(device, num_classes_with_bg):
    backbone = RGBRM_Backbone().to(device)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3", "pool"],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes_with_bg,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        image_mean=[0.5] * 5,
        image_std=[0.25] * 5
    ).to(device)

    print("Model Faster R-CNN RGBDD with gating created")

    return model
