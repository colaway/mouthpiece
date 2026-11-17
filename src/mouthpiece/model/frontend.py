"""Visual front-end: 3D conv stem + 2D ResNet-18 trunk, applied per frame.

Ported from mpc001/Visual_Speech_Recognition_for_Multiple_Languages
(espnet/nets/pytorch_backend/backbones/), Copyright 2021 Imperial College London
(Pingchuan Ma), Apache License 2.0. Parameter names match the released checkpoints.
"""

from __future__ import annotations

from torch import Tensor, nn

from mouthpiece.model.layers import Swish


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = Swish()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = Swish()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = x if self.downsample is None else self.downsample(x)
        result: Tensor = self.relu2(out + residual)
        return result


class ResNet(nn.Module):
    """ResNet-18 trunk without the stem (the 3D conv plays that role)."""

    def __init__(self, layers: tuple[int, int, int, int] = (2, 2, 2, 2)) -> None:
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers: list[nn.Module] = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        out: Tensor = self.avgpool(x)
        return out.flatten(1)


class Conv3dResNet(nn.Module):
    """(B, 1, T, 88, 88) grayscale clips -> (B, T, 512) per-frame features."""

    def __init__(self) -> None:
        super().__init__()
        self.frontend_nout = 64
        self.trunk = ResNet()
        self.frontend3D = nn.Sequential(  # noqa: N815 - checkpoint key
            nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            Swish(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        b = x.size(0)
        x = self.frontend3D(x)  # (B, C, T, H, W)
        t = x.size(2)
        x = x.transpose(1, 2).reshape(b * t, x.size(1), x.size(3), x.size(4))
        return self.trunk(x).view(b, t, -1)  # type: ignore[no-any-return]
