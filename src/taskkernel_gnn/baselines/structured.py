"""Structured-grid FNO and U-Net baselines for PDEBench."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.modes = int(modes)
        scale = 1.0 / max(in_channels * out_channels, 1)
        self.weight_positive = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )
        self.weight_negative = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    @staticmethod
    def _multiply(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", values, weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = torch.fft.rfft2(values)
        output = torch.zeros(
            values.shape[0], self.weight_positive.shape[1], values.shape[-2],
            transformed.shape[-1], dtype=torch.cfloat, device=values.device,
        )
        mx = min(self.modes, transformed.shape[-2])
        my = min(self.modes, transformed.shape[-1])
        output[:, :, :mx, :my] = self._multiply(
            transformed[:, :, :mx, :my], self.weight_positive[:, :, :mx, :my]
        )
        output[:, :, -mx:, :my] = self._multiply(
            transformed[:, :, -mx:, :my], self.weight_negative[:, :, :mx, :my]
        )
        return torch.fft.irfft2(output, s=values.shape[-2:])


class FourierNeuralOperator2d(nn.Module):
    def __init__(self, channels: int = 2, width: int = 32, modes: int = 12, layers: int = 4) -> None:
        super().__init__()
        self.lift = nn.Conv2d(channels + 2, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Conv2d(2 * width, channels, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        squeeze = state.ndim == 3
        if squeeze:
            state = state.unsqueeze(0)
        batch, height, width, _ = state.shape
        y = torch.linspace(0, 1, height, device=state.device, dtype=state.dtype)
        x = torch.linspace(0, 1, width, device=state.device, dtype=state.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx, yy), -1).expand(batch, -1, -1, -1)
        values = self.lift(torch.cat((state, grid), -1).permute(0, 3, 1, 2))
        for spectral, local in zip(self.spectral, self.local):
            values = F.gelu(spectral(values) + local(values))
        delta = self.project(values).permute(0, 2, 3, 1)
        result = state + delta
        return result.squeeze(0) if squeeze else result

    step = forward


class _DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1), nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class UNet2d(nn.Module):
    def __init__(self, channels: int = 2, width: int = 32) -> None:
        super().__init__()
        self.down1 = _DoubleConv(channels, width)
        self.down2 = _DoubleConv(width, 2 * width)
        self.middle = _DoubleConv(2 * width, 4 * width)
        self.up2 = _DoubleConv(6 * width, 2 * width)
        self.up1 = _DoubleConv(3 * width, width)
        self.output = nn.Conv2d(width, channels, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        squeeze = state.ndim == 3
        if squeeze:
            state = state.unsqueeze(0)
        values = state.permute(0, 3, 1, 2)
        down1 = self.down1(values)
        down2 = self.down2(F.avg_pool2d(down1, 2))
        middle = self.middle(F.avg_pool2d(down2, 2))
        up2 = F.interpolate(middle, size=down2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.up2(torch.cat((up2, down2), 1))
        up1 = F.interpolate(up2, size=down1.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.output(self.up1(torch.cat((up1, down1), 1)))
        result = values + delta
        result = result.permute(0, 2, 3, 1)
        return result.squeeze(0) if squeeze else result

    step = forward
