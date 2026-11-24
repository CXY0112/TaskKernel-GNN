"""Translation-tolerant morphology metrics for E1 reaction-diffusion fields.

The metrics deliberately complement rather than replace pointwise trajectory
errors.  All channels are compared independently and then averaged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance


MINKOWSKI_THRESHOLDS = np.linspace(-2.0, 2.0, 9, dtype=np.float64)
PERSISTENCE_SIZE = 32
SPECTRAL_SIZE = 64
MIN_PERSISTENCE = 0.05
MAX_PERSISTENCE_POINTS = 64


def _resize(values: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.as_tensor(values, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    if tensor.shape[-2:] != (size, size):
        tensor = F.interpolate(tensor, size=(size, size), mode="area")
    return tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64, copy=False)


def _standardized_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_mean = target.mean()
    target_std = max(float(target.std()), 1e-12)
    target_z = (target - target_mean) / target_std
    # Use the reference scale for both fields.  Independent standardization would
    # amplify an almost-flat prediction and falsely reward residual numerical texture.
    prediction_z = (prediction - target_mean) / target_std
    return np.clip(prediction_z, -4.0, 4.0), np.clip(target_z, -4.0, 4.0)


def _minkowski_curve(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    descriptors = []
    foreground_structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    background_structure = np.ones((3, 3), dtype=np.uint8)
    for threshold in MINKOWSKI_THRESHOLDS:
        mask = values > threshold
        area = float(mask.mean())
        padded = np.pad(mask.astype(np.int8), 1)
        perimeter = (
            np.abs(np.diff(padded, axis=0)).sum()
            + np.abs(np.diff(padded, axis=1)).sum()
        ) / (height * width)
        _, components = label(mask, structure=foreground_structure)
        background_labels, background_count = label(~mask, structure=background_structure)
        border = np.unique(np.concatenate((
            background_labels[0], background_labels[-1],
            background_labels[:, 0], background_labels[:, -1],
        )))
        outside = int(np.count_nonzero(border))
        holes = max(int(background_count) - outside, 0)
        euler_density = (int(components) - holes) / (height * width)
        descriptors.append((area, float(perimeter), float(euler_density)))
    return np.asarray(descriptors, dtype=np.float64)


def _relative_curve_distance(prediction: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    pred_curve = _minkowski_curve(prediction)
    target_curve = _minkowski_curve(target)
    numerator = np.abs(pred_curve - target_curve).sum(axis=0)
    denominator = np.abs(target_curve).sum(axis=0).clip(min=1e-8)
    components = numerator / denominator
    return float(components.mean()), components


def _persistence_h0(values: np.ndarray) -> np.ndarray:
    """Finite lower-star H0 intervals on a four-neighbour pixel complex."""
    height, width = values.shape
    flat = values.reshape(-1)
    order = np.argsort(flat, kind="stable")
    parent = np.arange(flat.size, dtype=np.int32)
    birth = flat.copy()
    active = np.zeros(flat.size, dtype=bool)
    intervals: list[tuple[float, float]] = []

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            following = int(parent[item])
            parent[item] = root
            item = following
        return root

    for item_value in order:
        item = int(item_value)
        active[item] = True
        row, column = divmod(item, width)
        neighbours = []
        if row: neighbours.append(item - width)
        if row + 1 < height: neighbours.append(item + width)
        if column: neighbours.append(item - 1)
        if column + 1 < width: neighbours.append(item + 1)
        death = float(flat[item])
        for neighbour in neighbours:
            if not active[neighbour]:
                continue
            first, second = find(item), find(neighbour)
            if first == second:
                continue
            if birth[first] <= birth[second]:
                survivor, dying = first, second
            else:
                survivor, dying = second, first
            persistence = death - float(birth[dying])
            if persistence >= MIN_PERSISTENCE:
                intervals.append((float(birth[dying]), death))
            parent[dying] = survivor
    if not intervals:
        return np.empty((0, 2), dtype=np.float64)
    diagram = np.asarray(intervals, dtype=np.float64)
    persistence = diagram[:, 1] - diagram[:, 0]
    if len(diagram) > MAX_PERSISTENCE_POINTS:
        diagram = diagram[np.argpartition(persistence, -MAX_PERSISTENCE_POINTS)[-MAX_PERSISTENCE_POINTS:]]
    return diagram


def _diagram_wasserstein(first: np.ndarray, second: np.ndarray) -> float:
    """RMS-normalized 2-Wasserstein distance with L-infinity ground metric."""
    n, m = len(first), len(second)
    if n == 0 and m == 0:
        return 0.0
    size = n + m
    large = 1e8
    cost = np.full((size, size), large, dtype=np.float64)
    if n and m:
        difference = np.abs(first[:, None, :] - second[None, :, :]).max(axis=2)
        cost[:n, :m] = difference ** 2
    if n:
        diagonal = ((first[:, 1] - first[:, 0]) * 0.5) ** 2
        cost[np.arange(n), m + np.arange(n)] = diagonal
    if m:
        diagonal = ((second[:, 1] - second[:, 0]) * 0.5) ** 2
        cost[n + np.arange(m), np.arange(m)] = diagonal
    if n and m:
        cost[n:, m:] = 0.0
    rows, columns = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, columns].sum() / max(n, m, 1)))


def _bidirectional_persistence_distance(prediction: np.ndarray, target: np.ndarray) -> float:
    lower = _diagram_wasserstein(_persistence_h0(prediction), _persistence_h0(target))
    upper = _diagram_wasserstein(_persistence_h0(-prediction), _persistence_h0(-target))
    return 0.5 * (lower + upper)


def _normalized_radial_spectrum(values: np.ndarray, bins: int = 32) -> tuple[np.ndarray, np.ndarray] | None:
    values = values - values.mean()
    power = np.abs(np.fft.fft2(values)) ** 2
    total = float(power.sum())
    if total <= 1e-12:
        return None
    fy = np.fft.fftfreq(values.shape[0])
    fx = np.fft.fftfreq(values.shape[1])
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    edges = np.linspace(0.0, radius.max() + 1e-12, bins + 1)
    indices = np.clip(np.digitize(radius.reshape(-1), edges) - 1, 0, bins - 1)
    radial = np.bincount(indices, weights=power.reshape(-1), minlength=bins).astype(np.float64)
    radial /= radial.sum()
    centers = 0.5 * (edges[:-1] + edges[1:]) / radius.max()
    return centers, radial


def _spectral_wasserstein(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = _normalized_radial_spectrum(prediction)
    truth = _normalized_radial_spectrum(target)
    if truth is None:
        return 0.0 if pred is None else 1.0
    if pred is None:
        return 1.0
    return float(wasserstein_distance(pred[0], truth[0], u_weights=pred[1], v_weights=truth[1]))


@torch.no_grad()
def morphology_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction_np = prediction.detach().cpu().numpy().astype(np.float64, copy=False)
    target_np = target.detach().cpu().numpy().astype(np.float64, copy=False)
    persistence_prediction = _resize(prediction_np, PERSISTENCE_SIZE)
    persistence_target = _resize(target_np, PERSISTENCE_SIZE)
    spectral_prediction = _resize(prediction_np, SPECTRAL_SIZE)
    spectral_target = _resize(target_np, SPECTRAL_SIZE)
    minkowski, areas, perimeters, eulers, persistence, spectral = [], [], [], [], [], []
    for channel in range(target_np.shape[-1]):
        pred_z, target_z = _standardized_pair(
            spectral_prediction[..., channel], spectral_target[..., channel]
        )
        distance, components = _relative_curve_distance(pred_z, target_z)
        minkowski.append(distance); areas.append(components[0]); perimeters.append(components[1]); eulers.append(components[2])
        pred_p, target_p = _standardized_pair(
            persistence_prediction[..., channel], persistence_target[..., channel]
        )
        persistence.append(_bidirectional_persistence_distance(pred_p, target_p))
        spectral.append(_spectral_wasserstein(
            spectral_prediction[..., channel], spectral_target[..., channel]
        ))
    return {
        "minkowski_distance": float(np.mean(minkowski)),
        "minkowski_area_distance": float(np.mean(areas)),
        "minkowski_perimeter_distance": float(np.mean(perimeters)),
        "minkowski_euler_distance": float(np.mean(eulers)),
        "persistence_wasserstein": float(np.mean(persistence)),
        "spectral_wasserstein": float(np.mean(spectral)),
    }
