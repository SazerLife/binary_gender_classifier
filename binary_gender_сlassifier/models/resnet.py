import torch
from typing import Dict
from torch import nn


class ResNet(nn.Module):
    def __init__(self, model: str, in_channels: int, classes_count: int):
        super(ResNet, self).__init__()
        self.model = torch.hub.load("pytorch/vision", model)
        self.model.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.model.fc = nn.Linear(
            in_features=512, out_features=classes_count, bias=True
        )

    def __call__(self, x: Dict[str, torch.Tensor]):
        y = {"labels": self.model(x["audio"])}
        return y
