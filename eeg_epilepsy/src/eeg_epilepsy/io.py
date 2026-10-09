"""Lectura de EDF (interfaz mínima que consume analysis.py).

Convención del proyecto: MNE entrega voltios; ``get_channel_uv`` convierte a µV UNA sola vez, al
extraer el arreglo que se va a analizar. No multiplicar de nuevo por 1e6 aguas abajo.
El EDF se lee de forma perezosa (preload=False): no se carga completo en memoria.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import mne
import numpy as np

V_TO_UV = 1e6


def read_edf(path: str | Path) -> tuple[mne.io.BaseRaw, dict]:
    """Abre el EDF sin cargarlo y devuelve (raw, metadatos). ``raw.close()`` al terminar."""
    path = Path(path)
    # MNE avisa y añade sufijos (-0/-1) cuando el EDF repite un nombre de canal (T8-P8). Es esperado:
    # la duplicación se resuelve con data/derived/channel_selection.json.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Channel names are not unique")
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    fs = float(raw.info["sfreq"])
    meta = {
        "source": path.name,
        "format": "EDF",
        "fs": fs,
        "duration_s": float(raw.n_times / fs),
        "n_channels": int(len(raw.ch_names)),
        "units": "µV",
        "original_units": "V (MNE)",
        "conversion": "V_to_uV_once",
        "channels": list(raw.ch_names),
    }
    return raw, meta


def get_channel_uv(raw: mne.io.BaseRaw, channel: str) -> np.ndarray:
    """Canal completo en µV (conversión V → µV única)."""
    return raw.get_data(picks=[channel])[0] * V_TO_UV
