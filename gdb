import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from Relu_Linear_Attention import LiteMLA

class DWConv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        )

    def forward(self, x):
        return self.dwconv(x)


class QKVGatedLiteMLA(nn.Module):
    def __init__(
        self,
        channels,
        scales=(5,),
        dim=8,
        heads=None,
    ):
        super().__init__()

        # Q / K / V 深度卷积分支
        self.dw_q = DWConv(channels, kernel_size=3)
        self.dw_k = DWConv(channels, kernel_size=5)
        self.dw_v = DWConv(channels, kernel_size=3)

        # Q / K / V LiteMLA（通道保持不变）
        self.mla_q = LiteMLA(
            in_channels=channels,
            out_channels=channels,
            scales=scales,
            dim=dim,
            heads=heads,
        )
        self.mla_k = LiteMLA(
            in_channels=channels,
            out_channels=channels,
            scales=scales,
            dim=dim,
            heads=heads,
        )
        self.mla_v = LiteMLA(
            in_channels=channels,
            out_channels=channels,
            scales=scales,
            dim=dim,
            heads=heads,
        )

        # 门控
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)

        fq = self.mla_q(self.dw_q(x))  # (B, C, H, W)
        fk = self.mla_k(self.dw_k(x))
        fv = self.mla_v(self.dw_v(x))

        # 门控
        gate = self.sigmoid(fq + fk + fv)  # (B, C, H, W)

        #  关键：先门控
        fq = fq * gate
        fk = fk * gate
        fv = fv * gate

        # 再拼接
        out = torch.cat([fq, fk, fv], dim=1)  # (B, 3C, H, W)

        return out


if __name__ == '__main__':
    x = torch.randn(1, 64, 7, 7).cuda()

    model = QKVGatedLiteMLA(
        channels=64,
        scales=(5,),
    ).cuda()

    out = model(x)
    print(out.shape)  # torch.Size([1, 192, 7, 7])
