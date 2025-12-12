import torch.nn as nn

from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN

def rgbd_base_model(device, num_classes_with_bg):
    backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.IMAGENET1K_V1, trainable_layers=5)
    old_conv = backbone.body.conv1

    backbone.body.conv1 = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes_with_bg,
        image_mean=[0.5, 0.5, 0.5, 0.5],
        image_std=[0.25, 0.25, 0.25, 0.25],
    ).to(device)

    print("Model Faster R-CNN RGBD created")

    return model
