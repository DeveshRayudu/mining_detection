"""
U-Net Deep Learning Architecture with Pretrained ResNet-34 Encoder.
Adapted for 8-channel Sentinel-2 multispectral inputs.
"""

from typing import Optional
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_unet_model(
    in_channels: int = 8,
    classes: int = 1,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
) -> nn.Module:
    """
    Constructs a U-Net model with a ResNet-34 backbone.
    The first convolutional layer is automatically adapted from 3 channels to 8 channels,
    preserving pretrained ImageNet weights for RGB channels while initializing new channels.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # Output raw logits for numerical stability with BCE/Focal
    )
    return model
