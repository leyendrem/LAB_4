"""Indicadores por época (Persona 3 · métricas).

Contrato de unidades: la señal de entrada está SIEMPRE en µV (la conversión V → µV se hace
una sola vez en ``io.read_edf``). Salidas:

    rms_uv            µV
    ll_uv_per_sample  µV/muestra   (depende de fs: comparar solo con la misma fs)
    power_uv2         µV²          (potencia total 0.5–40 Hz)
    <banda>_uv2       µV²          (potencia absoluta en la banda)
    <banda>_rel       adimensional (potencia de banda / potencia total 0.5–40 Hz)
    entropy           adimensional, en [0, 1] (entropía espectral normalizada 0.5–40 Hz)

Una señal plana no tiene potencias relativas ni entropía definidas: se devuelve NaN, no 0.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

# Convenciones operativas. 30–40 Hz completa el denominador; NO es "toda la banda gamma".
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "high_30_40": (30.0, 40.0),
}

TOTAL_BAND_HZ: tuple[float, float] = (0.5, 40.0)
BASIC_METRICS = ("rms_uv", "ll_uv_per_sample", "power_uv2", "entropy")


# --------------------------------------------------------------------------- #
# Métricas temporales
# --------------------------------------------------------------------------- #
def centered_rms(x_uv: np.ndarray) -> float:
    """RMS centrado, en µV: sqrt(mean((x - mean(x))²))."""
    x = np.asarray(x_uv, dtype=float)
    y = x - np.mean(x)
    return float(np.sqrt(np.mean(y * y)))


def line_length(x_uv: np.ndarray) -> float:
    """Longitud de línea media por paso de muestra, en µV/muestra: mean(|x[n] - x[n-1]|)."""
    x = np.asarray(x_uv, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(x))))


# --------------------------------------------------------------------------- #
# Métricas espectrales
# --------------------------------------------------------------------------- #
def bandpower(f: np.ndarray, p: np.ndarray, a: float, b: float) -> float:
    """Integra la PSD (µV²/Hz) en [a, b] interpolando los extremos de la banda.

    Los puntos compartidos entre bandas contiguas no suman área extra.
    """
    inside = (f > a) & (f < b)
    grid = np.r_[a, f[inside], b]
    values = np.interp(grid, f, p)
    return float(np.trapezoid(values, grid))


def spectral_entropy(
    f: np.ndarray,
    p: np.ndarray,
    low: float = TOTAL_BAND_HZ[0],
    high: float = TOTAL_BAND_HZ[1],
) -> float:
    """Entropía espectral normalizada H = -Σ p_k ln p_k / ln K, con K bins en [low, high].

    Devuelve NaN si no hay bins o si la masa espectral es nula / no finita.
    """
    mask = (f >= low) & (f <= high)
    q = np.asarray(p[mask], dtype=float)
    if q.size < 2:
        return float("nan")
    mass = float(q.sum())
    if not np.isfinite(mass) or mass <= 0:
        return float("nan")
    q = q / mass
    positive = q > 0
    return float(-np.sum(q[positive] * np.log(q[positive])) / np.log(len(q)))


def psd_features(
    x_uv: np.ndarray,
    fs: float,
    segment_s: float = 2.0,
    overlap: float = 0.5,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """PSD de Welch (Hann, segmentos de ``segment_s``, solape ``overlap``) y métricas derivadas.

    Con segment_s = 2 s la resolución de bins es fs/nperseg = 0.5 Hz. Una época de 2 s contiene
    un único segmento (sin promediado); épocas más largas promedian varios.
    """
    bands = BANDS if bands is None else bands
    x = np.asarray(x_uv, dtype=float)
    nperseg = round(segment_s * fs)
    noverlap = round(nperseg * overlap)
    if nperseg < 16 or len(x) < nperseg:
        raise ValueError("La época es demasiado corta para el segmento Welch configurado.")

    y = x - np.mean(x)
    f, p = welch(
        y,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    total = bandpower(f, p, *TOTAL_BAND_HZ)
    out = {
        "power_uv2": total,
        "entropy": spectral_entropy(f, p, *TOTAL_BAND_HZ),
        "psd_freq_hz": f,
        "psd_uv2_per_hz": p,
    }
    for name, (a, b) in bands.items():
        val = bandpower(f, p, a, b)
        out[f"{name}_uv2"] = val
        # total ≤ 0 (señal plana) o no finito → relativa indefinida (NaN, nunca 0).
        out[f"{name}_rel"] = val / total if np.isfinite(total) and total > 0 else np.nan
    return out


def nan_features(bands: dict[str, tuple[float, float]] | None = None) -> dict:
    """Fila de métricas vacía (NaN) con exactamente las mismas claves que ``epoch_features``."""
    bands = BANDS if bands is None else bands
    out = {k: np.nan for k in BASIC_METRICS}
    for name in bands:
        out[f"{name}_uv2"] = np.nan
        out[f"{name}_rel"] = np.nan
    return out


def empty_spectrum() -> dict:
    return {"freq_hz": np.array([]), "psd_uv2_per_hz": np.array([])}


def epoch_features(
    x_uv: np.ndarray,
    fs: float,
    segment_s: float = 2.0,
    overlap: float = 0.5,
    bands: dict[str, tuple[float, float]] | None = None,
) -> tuple[dict, dict]:
    """Punto único de entrada: métricas escalares + espectro completo de la época.

    Lanza ``ValueError`` si la época es más corta que el segmento Welch.
    """
    basic = {
        "rms_uv": centered_rms(x_uv),
        "ll_uv_per_sample": line_length(x_uv),
    }
    spec = psd_features(x_uv, fs, segment_s=segment_s, overlap=overlap, bands=bands)
    basic.update({k: v for k, v in spec.items() if not k.startswith("psd_")})
    return basic, {
        "freq_hz": spec["psd_freq_hz"],
        "psd_uv2_per_hz": spec["psd_uv2_per_hz"],
    }
