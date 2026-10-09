"""Aplica exclusiones documentadas solo a la selección de análisis."""
from __future__ import annotations
import json
from pathlib import Path

def load_channel_selection(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el contrato de canales: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("excluded_channels", []), list):
        raise ValueError("excluded_channels debe ser una lista")
    return data

def select_analysis_channels(channel_names: list[str], selection_path: str | Path) -> list[str]:
    """Devuelve canales disponibles menos las exclusiones explícitas, en orden original."""
    config = load_channel_selection(selection_path)
    excluded = {item["name"] for item in config.get("excluded_channels", [])}
    missing = excluded.difference(channel_names)
    if missing:
        raise ValueError("Canales excluidos no encontrados en el EDF: " + ", ".join(sorted(missing)))
    selected = [name for name in channel_names if name not in excluded]
    if not selected:
        raise ValueError("La selección de canales quedó vacía")
    return selected
