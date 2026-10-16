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
    if not np.isfinite(a) or not np.isfinite(b) or b < a:
        raise ValueError("La época debe tener límites finitos y end_s >= start_s.")
    clipped = []
    for u, v in intervals:
        if not np.isfinite(u) or not np.isfinite(v) or v < u:
            raise ValueError("Los intervalos anotados deben tener límites finitos y end_s >= start_s.")
        left, right = max(a, u), min(b, v)
        if right > left:
            clipped.append((left, right))
    total = 0.0
    current_start = current_end = None
    for left, right in sorted(clipped):
        if current_start is None:
            current_start, current_end = left, right
        elif left <= current_end:
            current_end = max(current_end, right)
        else:
            total += current_end - current_start
            current_start, current_end = left, right
    if current_start is not None:
        total += current_end - current_start
    return float(total)


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
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs debe ser positiva y finita.")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s debe ser positiva.")
    if not np.isfinite(step_fraction) or step_fraction <= 0 or step_fraction > 1:
        raise ValueError("step_fraction debe estar en el intervalo (0, 1].")
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
