from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Epoch:
    start_s: float
    end_s: float
    state: str
    overlap_s: float


def interval_overlap(a: float, b: float, intervals: list[tuple[float, float]]) -> float:
    """Longitud de la unión de intersecciones entre una época y crisis oficiales."""
    return float(sum(max(0.0, min(b, v) - max(a, u)) for u, v in intervals))


def label_epoch(
    start_s: float,
    end_s: float,
    intervals: list[tuple[float, float]],
    tolerance_s: float = 1e-9,
) -> tuple[str, float]:
    overlap = interval_overlap(start_s, end_s, intervals)
    duration = end_s - start_s
    if overlap <= tolerance_s:
        state = "sin_crisis_anotada"
    elif abs(overlap - duration) <= tolerance_s:
        state = "ictal"
    else:
        state = "transicion"
    return state, overlap


def make_epochs(
    n_samples: int,
    fs: float,
    duration_s: float,
    intervals: list[tuple[float, float]],
    step_fraction: float = 0.5,
) -> list[Epoch]:
    """Crea épocas de una duración concreta y avance igual a la mitad."""
    if duration_s <= 0:
        raise ValueError("duration_s debe ser positiva.")
    n = round(duration_s * fs)
    step = max(1, round(n * step_fraction))
    if n > n_samples:
        return []

    out: list[Epoch] = []
    for start_idx in range(0, n_samples - n + 1, step):
        a = start_idx / fs
        b = (start_idx + n) / fs
        state, overlap = label_epoch(a, b, intervals)
        out.append(Epoch(a, b, state, overlap))
    return out


def epoch_array(x: np.ndarray, epoch: Epoch, fs: float) -> np.ndarray:
    i0 = round(epoch.start_s * fs)
    i1 = round(epoch.end_s * fs)
    return np.asarray(x[i0:i1], dtype=float)
