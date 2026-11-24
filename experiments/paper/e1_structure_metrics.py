"""Resolution-independent structural metrics for E1 two-dimensional fields."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _channelwise_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean local SSIM over channels using an 11x11 uniform window."""
    x = prediction.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)
    channels = x.shape[1]
    kernel = min(11, x.shape[-2], x.shape[-1])
    padding = kernel // 2
    ranges = (y.amax(dim=(-2, -1), keepdim=True) - y.amin(dim=(-2, -1), keepdim=True)).clamp_min(1e-6)
    c1 = (0.01 * ranges).square()
    c2 = (0.03 * ranges).square()
    weight = torch.ones(channels, 1, kernel, kernel, device=x.device, dtype=x.dtype) / (kernel * kernel)
    mu_x = F.conv2d(x, weight, padding=padding, groups=channels)
    mu_y = F.conv2d(y, weight, padding=padding, groups=channels)
    var_x = F.conv2d(x * x, weight, padding=padding, groups=channels) - mu_x.square()
    var_y = F.conv2d(y * y, weight, padding=padding, groups=channels) - mu_y.square()
    cov = F.conv2d(x * y, weight, padding=padding, groups=channels) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    ).clamp_min(1e-12)
    return score.mean()


def _amplitude_ratio(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_std = prediction.flatten(0, 1).std(dim=0)
    target_std = target.flatten(0, 1).std(dim=0).clamp_min(1e-12)
    return (pred_std / target_std).mean()


def _radial_spectral_error(prediction: torch.Tensor, target: torch.Tensor, bins: int = 32) -> torch.Tensor:
    """Relative L1 error between radial power spectra, averaged over channels."""
    height, width, channels = target.shape
    fy = torch.fft.fftfreq(height, device=target.device, dtype=target.dtype)
    fx = torch.fft.fftfreq(width, device=target.device, dtype=target.dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    edges = torch.linspace(0, radius.max() + 1e-7, bins + 1, device=target.device, dtype=target.dtype)
    indices = torch.bucketize(radius.reshape(-1), edges[1:-1])
    values = []
    for channel in range(channels):
        pred_power = torch.fft.fft2(prediction[..., channel]).abs().square().reshape(-1)
        target_power = torch.fft.fft2(target[..., channel]).abs().square().reshape(-1)
        pred_radial = torch.zeros(bins, device=target.device, dtype=target.dtype)
        target_radial = torch.zeros_like(pred_radial)
        pred_radial.scatter_add_(0, indices, pred_power)
        target_radial.scatter_add_(0, indices, target_power)
        values.append((pred_radial - target_radial).abs().sum() / target_radial.sum().clamp_min(1e-12))
    return torch.stack(values).mean()


def _gradient_energy_ratio(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def energy(values: torch.Tensor) -> torch.Tensor:
        dx = values[:, 1:, :] - values[:, :-1, :]
        dy = values[1:, :, :] - values[:-1, :, :]
        return torch.sqrt(dx.square().sum(dim=(0, 1)) + dy.square().sum(dim=(0, 1)))
    return (energy(prediction) / energy(target).clamp_min(1e-12)).mean()


@torch.no_grad()
def structure_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return SSIM, amplitude ratio, spectral error and gradient-energy ratio."""
    return {
        "ssim": float(_channelwise_ssim(prediction, target)),
        "amplitude_ratio": float(_amplitude_ratio(prediction, target)),
        "spectral_error": float(_radial_spectral_error(prediction, target)),
        "gradient_ratio": float(_gradient_energy_ratio(prediction, target)),
    }
