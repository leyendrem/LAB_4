from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def to_microvolts(x_volts: np.ndarray) -> np.ndarray:
    """Convierte V → µV. Debe llamarse una sola vez durante la carga."""
    return np.asarray(x_volts, dtype=float) * 1e6


def bandpass_eeg(
    x_uv: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Pasabanda Butterworth SOS; procesamiento offline sin fase."""
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError("Debe cumplirse 0 < low_hz < high_hz < fs/2.")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x_uv, dtype=float))


def filter_with_context(
    x_uv: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Filtra el contexto amplio antes de recortar épocas."""
    return bandpass_eeg(x_uv, fs, low_hz, high_hz, order)


def edge_review(
    raw_uv: np.ndarray,
    filtered_uv: np.ndarray,
    n_samples: int | None = None,
) -> dict:
    """Devuelve una revisión simple de amplitud en bordes; no borra muestras."""
    n = min(n_samples or 2, len(raw_uv), len(filtered_uv))
    if n == 0:
        return {"n_edge_samples": 0, "raw_edge_abs_max_uv": np.nan, "filtered_edge_abs_max_uv": np.nan}
    return {
        "n_edge_samples": int(n),
        "raw_edge_abs_max_uv": float(np.max(np.abs(raw_uv[:n]))),
        "filtered_edge_abs_max_uv": float(np.max(np.abs(filtered_uv[:n]))),
        "raw_end_abs_max_uv": float(np.max(np.abs(raw_uv[-n:]))),
        "filtered_end_abs_max_uv": float(np.max(np.abs(filtered_uv[-n:]))),
    }


def preprocess_channel(
    x_uv: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> dict[str, np.ndarray]:
    raw = np.asarray(x_uv, dtype=float)
    filtered = filter_with_context(raw, fs, low_hz, high_hz, order)
    return {"raw_uv": raw, "filtered_uv": filtered}
