"""Calcula las métricas por época y exporta los resultados a data/derived/.

    uv run python scripts/run_analysis.py            # ventana de análisis de config/analysis_config.json
    uv run python scripts/run_analysis.py --full     # registro completo (más lento)

Requiere data/raw/chb01_03.edf (scripts/download_data.py) y data/derived/channel_selection.json.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_epilepsy.analysis import run_analysis  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="ignora analysis.window_s")
    ap.add_argument("--config", default=None, help="ruta a un analysis_config.json alternativo")
    args = ap.parse_args()
    metrics, meta, _ = run_analysis(config_path=args.config, full_record=args.full)
    print(f"Filas de métricas: {len(metrics)}")
    print(f"Canales analizados ({len(meta['selected_channels'])}): {', '.join(meta['selected_channels'])}")
    print(f"Excluidos (channel_selection.json): {meta['channel_selection']['excluded_applied']}")
    print(f"Ventana de análisis: {meta['analysis_window_s']} s · crisis oficial: {meta['annotated_intervals_s']}")
    print(f"Marcadas por calidad: {int(metrics['quality_flag'].sum())} de {len(metrics)} épocas")
    print("Resultados en data/derived/: metricas.csv, resumen_descriptivo.csv, "
          "resumen_descriptivo_sin_marcadas.csv, psd_mediana_por_estado.csv, configuracion_y_trazabilidad.json")
