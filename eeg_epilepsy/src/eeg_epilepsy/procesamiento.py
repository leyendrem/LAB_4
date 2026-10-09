"""Ejemplo de procesamiento EEG con exclusiones documentadas y unidades explícitas."""
from pathlib import Path
import re
import hashlib
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt
from eeg_epilepsy.data.channel_selection import select_analysis_channels

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDF = PROJECT_ROOT / "data/raw/chb01_03.edf"
SUMMARY = PROJECT_ROOT / "data/raw/chb01-summary.txt"
SELECTION = PROJECT_ROOT / "data/derived/channel_selection.json"
CONTEXT = (2800.0, 3232.0)
ANALYSIS = (2876.0, 3156.0)

def read_official_intervals(summary_path: Path, edf_name: str, duration_s: float):
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"File Name:\s*", text)[1:]
    block = next((b for b in blocks if b.splitlines()[0].strip() == edf_name), None)
    if block is None:
        raise ValueError(f"No hay bloque de anotaciones para {edf_name}")
    count_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)
    if count_match is None:
        raise ValueError("No se encontró el número de crisis anotadas")
    starts = [int(v) for v in re.findall(r"Seizure(?:\s+\d+)? Start Time:\s*(\d+) seconds", block)]
    ends = [int(v) for v in re.findall(r"Seizure(?:\s+\d+)? End Time:\s*(\d+) seconds", block)]
    if len(starts) != len(ends) or len(starts) != int(count_match.group(1)):
        raise ValueError("La cantidad de intervalos no coincide con el resumen oficial")
    intervals = list(zip(starts, ends))
    if any(not (0 <= a < b <= duration_s) for a, b in intervals):
        raise ValueError("Hay un intervalo anotado fuera de los límites del EDF")
    return intervals

def prepare_reference_channel(edf_path: Path = EDF, summary_path: Path = SUMMARY, channel: str = "FP1-F7"):
    """Carga metadatos, aplica exclusiones, convierte V→µV una vez y filtra contexto."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    try:
        fs = float(raw.info["sfreq"])
        duration = raw.n_times / fs
        intervals = read_official_intervals(summary_path, edf_path.name, duration)
        selected = select_analysis_channels(list(raw.ch_names), SELECTION)
        if channel not in selected:
            raise ValueError(f"Canal {channel!r} no disponible para análisis tras exclusiones. Opciones: {selected}")
        if not (0 <= CONTEXT[0] < CONTEXT[1] <= duration):
            raise ValueError("La ventana de contexto excede la duración del EDF")
        i0, i1 = round(CONTEXT[0]*fs), round(CONTEXT[1]*fs)
        # MNE conserva internamente el EDF en voltios; conversión única al arreglo analítico en µV.
        x_uv = raw.get_data(picks=[channel], start=i0, stop=i1)[0] * 1e6
        sos = butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
        filtered_uv = sosfiltfilt(sos, x_uv)
        time_s = CONTEXT[0] + np.arange(x_uv.size) / fs
        inside = (time_s >= ANALYSIS[0]) & (time_s < ANALYSIS[1])
        result = {
            "channel": channel, "fs_hz": fs, "duration_s": duration,
            "intervals_s": intervals, "selected_channels": selected,
            "time_context_s": time_s, "raw_uv": x_uv,
            "filtered_uv": filtered_uv, "time_analysis_s": time_s[inside],
            "raw_analysis_uv": x_uv[inside], "filtered_analysis_uv": filtered_uv[inside],
            "edf_sha256": hashlib.sha256(edf_path.read_bytes()).hexdigest(),
            "filter": {"type":"Butterworth bandpass", "order":4, "low_hz":0.5, "high_hz":40, "method":"SOS + sosfiltfilt", "causal":False}
        }
        return result
    finally:
        raw.close()

def main():
    result = prepare_reference_channel()
    print(f"Canal: {result['channel']} | fs={result['fs_hz']} Hz | duración={result['duration_s']:.1f} s")
    print(f"Crisis oficiales: {result['intervals_s']}")
    print(f"Canales seleccionados: {len(result['selected_channels'])}; SHA-256 EDF: {result['edf_sha256']}")
    print("Contexto filtrado en µV. El filtrado es offline/no causal; revisar bordes antes de interpretar.")

if __name__ == "__main__":
    main()
