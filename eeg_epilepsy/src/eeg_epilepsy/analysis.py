"""Integración del pipeline.

    analyze_file()         EDF → DataFrame de métricas por época (+ metadatos + espectros)
    descriptive_summary()  mediana, IQR, n y % marcado por calidad, por canal/ventana/estado
    export_results()       CSV + JSON de trazabilidad en data/derived/
    run_analysis()         lo anterior, leyendo rutas y parámetros de config/analysis_config.json

Todos los parámetros (filtro, épocas, Welch, bandas, calidad, ventana de análisis) salen de
``config/analysis_config.json``. El dashboard debe usar este mismo módulo y esa misma config.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .data.audit import read_events, read_original_header
from .data.channel_selection import load_channel_selection, select_analysis_channels
from .io import get_channel_uv, read_edf
from .metrics import BASIC_METRICS, empty_spectrum, epoch_features, nan_features
from .preprocessing import edge_review, preprocess_channel
from .quality import epoch_quality
from .segmentation import epoch_array, make_epochs

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "analysis_config.json"

# Respaldo si no existe el JSON; debe coincidir con config/analysis_config.json.
DEFAULT_CONFIG: dict = {
    "data": {
        "edf": "data/raw/chb01_03.edf",
        "summary": "data/raw/chb01-summary.txt",
        "manifest": "data/raw/download_manifest.json",
        "derived_dir": "data/derived",
    },
    "channels": {
        "minimum": 4,
        "avoid_duplicates": True,
        "selection_file": "data/derived/channel_selection.json",
    },
    "analysis": {"window_s": [2876, 3156]},
    "filter": {"type": "Butterworth", "order": 4, "low_hz": 0.5, "high_hz": 40.0,
               "method": "sosfiltfilt", "offline": True},
    "epochs": {"durations_s": [2, 4, 8], "step_fraction": 0.5},
    "welch": {"window": "hann", "segment_s": 2.0, "overlap": 0.5,
              "resolution_hz": 0.5, "scaling": "density"},
    "bands_hz": {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 13],
                 "beta": [13, 30], "high_30_40": [30, 40]},
    "quality": {"flat_threshold_uv": 0.1, "extreme_threshold_uv": 600.0, "rail_tolerance_lsb": 1.0},
}

DATASET_INFO = {
    "name": "CHB-MIT Scalp EEG Database",
    "version": "1.0.0",
    "url": "https://physionet.org/content/chbmit/1.0.0/",
    "doi": "10.13026/C2K01R",
    "license": "Open Data Commons Attribution License v1.0",
}

SPECTRUM_RANGE_HZ = (0.5, 40.0)  # rango PSD que se conserva por época (el de las métricas)
STATES_BINARY = ("ictal", "sin_crisis_anotada")
_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Configuración y trazabilidad
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict:
    """Lee la configuración JSON; completa con valores por defecto lo que falte."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if p.exists():
        user = json.loads(p.read_text(encoding="utf-8-sig"))
        for section, value in user.items():
            if isinstance(value, dict) and isinstance(cfg.get(section), dict):
                cfg[section].update(value)
            else:
                cfg[section] = value
    return cfg


def resolve(path: str | Path, root: Path = ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "pandas", "mne"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _bands(cfg: dict) -> dict[str, tuple[float, float]]:
    return {k: (float(v[0]), float(v[1])) for k, v in cfg["bands_hz"].items()}


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Longitud de la unión de intervalos (no suma solapes dos veces)."""
    total, end = 0.0, -np.inf
    for a, b in sorted(intervals):
        a = max(a, end)
        if b > a:
            total += b - a
            end = b
    return float(total)


def _quality_flag(q: dict) -> bool:
    """True si la época tiene cualquier bandera de calidad (revisable, no 'artefacto confirmado')."""
    return bool(
        q["quality_flat"]
        or q["quality_extreme"]
        or q["quality_saturation"]
        or q["quality_nan_fraction"] > 0
    )


def _manifest_entry(manifest_path: Path | None, filename: str, sha256: str) -> dict:
    """Datos de descarga registrados por scripts/reproduce.py, verificando el SHA-256."""
    if manifest_path is None or not Path(manifest_path).exists():
        return {"found": False}
    entry = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig")).get(filename)
    if not entry:
        return {"found": False}
    expected = entry.get("sha256")
    return {
        "found": True,
        "url": entry.get("url"),
        "downloaded_at_utc": entry.get("downloaded_at_utc"),
        "size_bytes": entry.get("size_bytes"),
        "sha256_manifest": expected,
        "sha256_matches": (expected == sha256) if expected else None,
    }


def acquisition_limits(edf_path: str | Path, channels_in_raw: list[str]) -> dict[str, dict]:
    """Límites físicos/digitales de la cabecera EDF por canal (nombres de MNE).

    Se alinean por posición; si MNE y la cabecera difieren en número de canales no se asume nada.
    """
    _, _, acquisition = read_original_header(Path(edf_path))
    if len(acquisition) != len(channels_in_raw):
        return {}
    return {
        name: {
            "physical_min_uv": a["physical_min"],
            "physical_max_uv": a["physical_max"],
            "digital_min": a["digital_min"],
            "digital_max": a["digital_max"],
            "lsb_uv": a["physical_units_per_code"],
            "unit_in_header": a["original_unit"],
        }
        for name, a in zip(channels_in_raw, acquisition)
    }


def analysis_channels(raw, edf_name: str, selection_path: str | Path | None) -> tuple[list[str], dict]:
    """Canales EEG a analizar: aplica ``channel_selection.json`` (módulo data.channel_selection).

    - Parte de los canales EEG según MNE (no ficticios ni no EEG).
    - Aplica las exclusiones documentadas; un nombre excluido inexistente es error.
    - Si tras excluir quedan nombres repetidos (T8-P8-0/-1) se aborta: duplicarían las métricas.
    - Sin archivo de selección solo se elimina el sufijo duplicado, conservando el primero.
    Devuelve (canales, selección) con selección = {file, excluded{nombre: motivo}, review_channels,
    quality_review_complete, policy}.
    """
    names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    selection = {"file": None, "excluded": {}, "review_channels": [], "quality_review_complete": False,
                 "policy": None}

    def base(n):
        return re.sub(r"-\d+$", "", n.strip()).casefold()

    if selection_path is not None:
        cfg = load_channel_selection(selection_path)
        if cfg.get("file") and cfg["file"] != edf_name:
            raise ValueError(f"channel_selection.json corresponde a {cfg['file']}, no a {edf_name}.")
        chosen = select_analysis_channels(names, selection_path)
        repeated = sorted({n for n in chosen if [base(m) for m in chosen].count(base(n)) > 1})
        if repeated:
            raise ValueError(f"Tras aplicar channel_selection.json siguen duplicados: {repeated}. "
                             "Añádelos a excluded_channels para no duplicar métricas.")
        selection.update({
            "file": cfg.get("file"),
            "excluded": {i["name"]: i.get("reason", "") for i in cfg.get("excluded_channels", [])},
            "review_channels": list(cfg.get("provisional_review_channels", [])),
            "quality_review_complete": bool(cfg.get("quality_review_complete", False)),
            "policy": cfg.get("policy"),
        })
    else:
        chosen, seen = [], set()
        for n in names:
            if base(n) not in seen:
                seen.add(base(n))
                chosen.append(n)
    return chosen, selection


# --------------------------------------------------------------------------- #
# Pipeline por archivo
# --------------------------------------------------------------------------- #
def analyze_file(
    edf_path: str | Path,
    summary_path: str | Path,
    durations_s: tuple[int, ...] | None = None,
    minimum_channels: int | None = None,
    config: dict | None = None,
    channel_selection: str | Path | None = None,
    window_s: tuple[float, float] | None | str = "config",
    manifest_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Pipeline completo para un EDF (sin concatenarlo con otros EDF).

    Flujo: lectura → canales (aplica ``channel_selection``) → anotaciones oficiales → por canal:
    µV, filtro, segmentación por duración, calidad y ``epoch_features`` → DataFrame.

    ``channel_selection``: ruta a ``data/derived/channel_selection.json``; sus exclusiones se
    aplican solo a los canales analizados (el EDF no se modifica). ``window_s`` limita las
    MÉTRICAS a épocas completas dentro de [a, b] s desde el inicio del EDF; el filtrado se hace
    sobre el registro completo antes de recortar, y ``None`` analiza todo el registro.

    Devuelve ``(metrics, meta, spectra)``:
      metrics  DataFrame: una fila por (canal, ventana, época).
      meta     metadatos + configuración + trazabilidad (serializable a JSON).
      spectra  {"canal|ventana|índice": {"freq_hz": [...], "psd_uv2_per_hz": [...]}} en 0.5–40 Hz.
    """
    cfg = config or load_config()
    durations_s = tuple(durations_s or cfg["epochs"]["durations_s"])
    minimum_channels = int(minimum_channels or cfg["channels"]["minimum"])
    if isinstance(window_s, str):  # "config"
        window_s = cfg.get("analysis", {}).get("window_s")

    edf_path, summary_path = Path(edf_path), Path(summary_path)
    raw, meta = read_edf(edf_path)
    try:
        return _analyze(raw, meta, edf_path, summary_path, cfg, durations_s, minimum_channels,
                        channel_selection, window_s, manifest_path)
    finally:
        raw.close()


def _analyze(raw, meta, edf_path, summary_path, cfg, durations_s, minimum_channels,
             channel_selection, window_s, manifest_path):
    flt, welch_cfg, qcfg = cfg["filter"], cfg["welch"], cfg["quality"]
    bands = _bands(cfg)
    step_fraction = float(cfg["epochs"]["step_fraction"])
    fs, duration = meta["fs"], meta["duration_s"]

    # --- canales (exclusiones documentadas) ---------------------------------------------
    channels, selection = analysis_channels(raw, edf_path.name, channel_selection)
    if len(channels) < minimum_channels:
        raise ValueError(f"Se requieren al menos {minimum_channels} derivaciones; quedaron {len(channels)}.")

    # --- anotaciones oficiales (audit.read_events valida límites y solapes) ---------------
    intervals = [(e["start_s"], e["end_s"]) for e in read_events(summary_path, edf_path.name, duration)]
    if window_s is not None:
        w0, w1 = float(window_s[0]), float(window_s[1])
        if not (0 <= w0 < w1 <= duration):
            raise ValueError(f"La ventana de análisis {window_s} no cabe en el registro (0–{duration} s).")
    else:
        w0, w1 = 0.0, duration

    limits = acquisition_limits(edf_path, list(raw.ch_names))
    tol_lsb = float(qcfg.get("rail_tolerance_lsb", 1.0))
    resolution_hz = fs / round(welch_cfg["segment_s"] * fs)

    rows: list[dict] = []
    spectra: dict[str, dict] = {}
    freq_cache: dict[int, list] = {}  # misma rejilla de frecuencias para todas las épocas

    for channel in channels:
        x = get_channel_uv(raw, channel)  # V → µV, una sola vez
        stages = preprocess_channel(x, fs, flt["low_hz"], flt["high_hz"], int(flt["order"]))
        meta.setdefault("edge_review", {})[channel] = edge_review(stages["raw_uv"], stages["filtered_uv"])

        lim = limits.get(channel)
        rail = (lim["physical_min_uv"], lim["physical_max_uv"]) if lim else None
        tol_uv = tol_lsb * lim["lsb_uv"] if lim else 0.0

        for duration_s in durations_s:
            epochs = make_epochs(len(x), fs, duration_s, intervals, step_fraction)
            for idx, epoch in enumerate(epochs):  # idx = posición en la rejilla completa (estable)
                if epoch.start_s < w0 - _EPS or epoch.end_s > w1 + _EPS:
                    continue
                y = epoch_array(stages["filtered_uv"], epoch, fs)
                raw_epoch = epoch_array(stages["raw_uv"], epoch, fs)
                quality = epoch_quality(
                    raw_epoch,
                    flat_threshold_uv=float(qcfg["flat_threshold_uv"]),
                    extreme_threshold_uv=float(qcfg["extreme_threshold_uv"]),
                    rail_limits_uv=rail,
                    rail_tolerance_uv=tol_uv,
                )
                try:
                    feats, spec = epoch_features(
                        y, fs, welch_cfg["segment_s"], welch_cfg["overlap"], bands
                    )
                except ValueError:  # época más corta que el segmento Welch
                    feats, spec = nan_features(bands), empty_spectrum()

                length = epoch.end_s - epoch.start_s
                rows.append({
                    "session": edf_path.stem,
                    "channel": channel,
                    "window_s": duration_s,
                    "epoch_index": idx,
                    "start_s": epoch.start_s,
                    "end_s": epoch.end_s,
                    "state": epoch.state,
                    "overlap_s": epoch.overlap_s,
                    "ictal_fraction": epoch.overlap_s / length if length > 0 else np.nan,
                    **quality,
                    "quality_flag": _quality_flag(quality),
                    **feats,
                })

                f = np.asarray(spec["freq_hz"])
                key = f"{channel}|{duration_s}|{idx}"
                if f.size:
                    sel = (f >= SPECTRUM_RANGE_HZ[0]) & (f <= SPECTRUM_RANGE_HZ[1])
                    if len(f) not in freq_cache:
                        freq_cache[len(f)] = f[sel].tolist()
                    spectra[key] = {
                        "freq_hz": freq_cache[len(f)],
                        "psd_uv2_per_hz": np.asarray(spec["psd_uv2_per_hz"])[sel].tolist(),
                    }
                else:
                    spectra[key] = {"freq_hz": [], "psd_uv2_per_hz": []}

    metrics = pd.DataFrame(rows)
    annotated_s = _union_length(intervals)
    inside_window = _union_length([(max(a, w0), min(b, w1)) for a, b in intervals if b > w0 and a < w1])
    edf_sha, summ_sha = sha256_file(edf_path), sha256_file(summary_path)

    meta.update({
        "selected_channels": channels,
        "review_channels": [c for c in selection["review_channels"] if c in channels],
        "channel_selection": {
            "file": selection.get("file"),
            "excluded": selection["excluded"],
            "excluded_applied": sorted(selection["excluded"]),
            "quality_review_complete": selection["quality_review_complete"],
            "policy": selection.get("policy"),
            "note": "Exclusión de datos seleccionados para analizar; no implica mala calidad. "
                    "El EDF original no se modifica.",
        },
        "annotated_intervals_s": intervals,
        # Tiempo/fracción ictal según el resumen OFICIAL del experto, no una detección automática.
        "annotated_ictal_time_s": annotated_s,
        "annotated_ictal_fraction_of_record": annotated_s / duration,
        "analysis_window_s": [w0, w1],
        "annotated_ictal_fraction_of_window": inside_window / (w1 - w0),
        "analysis_durations_s": list(durations_s),
        "welch_resolution_hz": resolution_hz,
        "acquisition_limits": {c: limits[c] for c in channels if c in limits},
        "filter_context": "registro completo filtrado antes de recortar épocas",
        "config": cfg,
        "traceability": {
            "dataset": DATASET_INFO,
            "edf_file": edf_path.name,
            "edf_sha256": edf_sha,
            "edf_download": _manifest_entry(manifest_path, edf_path.name, edf_sha),
            "summary_file": summary_path.name,
            "summary_sha256": summ_sha,
            "summary_download": _manifest_entry(manifest_path, summary_path.name, summ_sha),
            "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "software_versions": _versions(),
        },
    })
    return metrics, meta, spectra


def run_analysis(
    config_path: str | Path | None = None,
    root: str | Path = ROOT,
    full_record: bool = False,
    export: bool = True,
) -> tuple[pd.DataFrame, dict, dict]:
    """Corre el pipeline con las rutas de la config. ``full_record`` ignora la ventana de análisis."""
    root = Path(root)
    cfg = load_config(config_path)
    d = cfg["data"]
    sel_path = resolve(cfg["channels"]["selection_file"], root)
    metrics, meta, spectra = analyze_file(
        resolve(d["edf"], root),
        resolve(d["summary"], root),
        config=cfg,
        channel_selection=sel_path if sel_path.exists() else None,
        window_s=None if full_record else "config",
        manifest_path=resolve(d["manifest"], root),
    )
    if export:
        export_results(metrics, meta, spectra, resolve(d["derived_dir"], root))
    return metrics, meta, spectra


# --------------------------------------------------------------------------- #
# Resúmenes descriptivos
# --------------------------------------------------------------------------- #
def flag_series(metrics: pd.DataFrame) -> pd.Series:
    """Bandera de calidad por época (usa ``quality_flag`` o la reconstruye desde quality_*)."""
    if "quality_flag" in metrics:
        return metrics["quality_flag"].fillna(False).astype(bool)

    def col(name, default):
        return metrics[name] if name in metrics else pd.Series(default, index=metrics.index)

    return (
        col("quality_flat", False).fillna(False).astype(bool)
        | col("quality_extreme", False).fillna(False).astype(bool)
        | col("quality_saturation", False).fillna(False).astype(bool)
        | (col("quality_nan_fraction", 0.0).fillna(0) > 0)
    )


def summary_metric_columns(metrics: pd.DataFrame) -> list[str]:
    """Métricas principales + potencias absolutas y relativas de cada banda presentes."""
    band_abs = [c for c in metrics.columns if c.endswith("_uv2") and c != "power_uv2"]
    band_rel = [c for c in metrics.columns if c.endswith("_rel")]
    return [c for c in BASIC_METRICS if c in metrics.columns] + band_abs + band_rel


def descriptive_summary(metrics: pd.DataFrame, exclude_flagged: bool = False) -> pd.DataFrame:
    """Mediana e IQR por canal × ventana × estado, más n de épocas y % marcado por calidad.

    exclude_flagged=True calcula mediana/IQR solo con épocas sin banderas, para ver qué
    resultados dependen de esa exclusión. ``quality_flag_percent`` siempre se calcula sobre
    todas las épocas del grupo. Las épocas solapadas no son observaciones independientes:
    esto es descriptivo, no inferencial.
    """
    group_cols = ["channel", "window_s", "state"]
    cols = summary_metric_columns(metrics)
    flags = flag_series(metrics)
    rows = []
    for keys, d in metrics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        used = d[~flags.loc[d.index]] if exclude_flagged else d
        for metric in cols:
            values = used[metric].dropna()
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_iqr"] = (
                float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan
            )
        n_flagged = int(flags.loc[d.index].sum())
        row["n_epochs"] = int(len(used))
        row["n_epochs_total"] = int(len(d))
        row["n_flagged"] = n_flagged
        row["quality_flag_percent"] = float(100 * n_flagged / len(d)) if len(d) else np.nan
        row["excluded_flagged"] = bool(exclude_flagged)
        rows.append(row)
    return pd.DataFrame(rows)


def binary_subset(metrics: pd.DataFrame, exclude_flagged: bool = False) -> pd.DataFrame:
    """Solo ``ictal`` vs ``sin_crisis_anotada``: las épocas de transición quedan fuera."""
    d = metrics[metrics["state"].isin(STATES_BINARY)]
    if exclude_flagged:
        d = d[~flag_series(d)]
    return d.copy()


def psd_by_state(metrics: pd.DataFrame, spectra: dict, exclude_flagged: bool = False) -> pd.DataFrame:
    """PSD mediana (µV²/Hz) por canal × ventana × estado, para la figura «PSD por estado»."""
    flags = flag_series(metrics)
    rows = []
    for (channel, window, state), d in metrics.groupby(["channel", "window_s", "state"]):
        if exclude_flagged:
            d = d[~flags.loc[d.index]]
        keys = [f"{channel}|{window}|{i}" for i in d["epoch_index"]]
        keys = [k for k in keys if spectra.get(k, {}).get("freq_hz")]
        if not keys:
            continue
        med = np.median(np.asarray([spectra[k]["psd_uv2_per_hz"] for k in keys], dtype=float), axis=0)
        rows.append(pd.DataFrame({
            "channel": channel, "window_s": window, "state": state,
            "freq_hz": spectra[keys[0]]["freq_hz"], "psd_median_uv2_per_hz": med,
            "n_epochs": len(keys),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Exportación
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"No serializable: {type(o)}")


OUTPUT_NAMES = {
    "metricas": "metricas.csv",
    "resumen": "resumen_descriptivo.csv",
    "resumen_sin_marcadas": "resumen_descriptivo_sin_marcadas.csv",
    "psd_por_estado": "psd_mediana_por_estado.csv",
    "config": "configuracion_y_trazabilidad.json",
}


def export_results(
    metrics: pd.DataFrame,
    meta: dict,
    spectra: dict,
    out_dir: str | Path = "data/derived",
) -> dict[str, Path]:
    """Guarda CSV/JSON en ``out_dir``. Devuelve {nombre: ruta} de lo escrito."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / v for k, v in OUTPUT_NAMES.items()}

    metrics.to_csv(paths["metricas"], index=False)
    descriptive_summary(metrics).to_csv(paths["resumen"], index=False)
    descriptive_summary(metrics, exclude_flagged=True).to_csv(paths["resumen_sin_marcadas"], index=False)

    psd = psd_by_state(metrics, spectra)
    if psd.empty:
        paths.pop("psd_por_estado")
    else:
        psd.to_csv(paths["psd_por_estado"], index=False)

    paths["config"].write_text(
        json.dumps(
            {"metadata": meta, "n_spectra": len(spectra),
             "spectra_key_format": "canal|ventana_s|indice_epoca"},
            indent=2, ensure_ascii=False, default=_json_default,
        ),
        encoding="utf-8",
    )
    return paths
