"""Banderas de calidad por época, calculadas sobre la señal CRUDA en µV.

Las banderas marcan épocas para revisión visual; no equivalen a «artefacto confirmado» y
nunca borran datos en silencio.
"""
from __future__ import annotations

import numpy as np


def epoch_quality(
    raw_uv: np.ndarray,
    flat_threshold_uv: float = 0.1,
    extreme_threshold_uv: float = 600.0,
    rail_limits_uv: tuple[float, float] | None = None,
    rail_tolerance_uv: float = 0.0,
) -> dict:
    """Banderas de una época.

    - quality_flat:       desviación estándar < ``flat_threshold_uv`` (didáctico, configurable).
    - quality_extreme:    |x| máximo > ``extreme_threshold_uv``. Amplitud grande NO prueba saturación.
    - quality_saturation: alguna muestra en los límites físicos del digitalizador declarados en la
                          cabecera EDF (``rail_limits_uv``) ± ``rail_tolerance_uv``. Sin límites
                          conocidos no se evalúa (False, y ``quality_rail_fraction`` = NaN).
    - quality_nan_fraction / quality_valid_fraction: datos no finitos.
    """
    x = np.asarray(raw_uv, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return {
            "quality_valid_fraction": 0.0,
            "quality_nan_fraction": 1.0,
            "quality_flat": True,
            "quality_extreme": False,
            "quality_saturation": False,
            "quality_rail_fraction": np.nan,
        }

    xf = x[finite]
    if rail_limits_uv is None:
        rail_fraction = np.nan
        saturation = False
    else:
        lo, hi = rail_limits_uv
        at_rail = (xf <= lo + rail_tolerance_uv) | (xf >= hi - rail_tolerance_uv)
        rail_fraction = float(np.mean(at_rail))
        saturation = bool(at_rail.any())

    return {
        "quality_valid_fraction": float(np.mean(finite)),
        "quality_nan_fraction": float(np.mean(~finite)),
        "quality_flat": bool(np.std(xf) < flat_threshold_uv),
        "quality_extreme": bool(np.max(np.abs(xf)) > extreme_threshold_uv),
        "quality_saturation": saturation,
        "quality_rail_fraction": rail_fraction,
    }
