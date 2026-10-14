"""Primera pantalla de revisión EEG: tendencias de resultados existentes."""
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from eeg_epilepsy.visualization import METRICS, trend_figure

st.set_page_config(page_title="Revisión de EEG", page_icon="🧠", layout="wide")


@st.cache_data(show_spinner=False)
def load_results(root: str, versions: tuple):
    """La firma de archivos invalida el caché cuando cambian los resultados."""
    base = Path(root)
    config = json.loads((base / "config/analysis_config.json").read_text())
    derived = base / config["data"]["derived_dir"]
    metrics = pd.read_csv(derived / "metricas.csv")
    events = pd.read_csv(derived / "events.csv", encoding="utf-8-sig")
    provenance = json.loads((derived / "configuracion_y_trazabilidad.json").read_text())
    return metrics, events, config, provenance.get("metadata", {})


st.title("Revisión de EEG")
st.caption("Exploración retrospectiva · Tendencias por derivación")
try:
    config_path = ROOT / "config/analysis_config.json"
    config = json.loads(config_path.read_text())
    derived = ROOT / config["data"]["derived_dir"]
    files = [config_path, derived / "metricas.csv", derived / "events.csv",
             derived / "configuracion_y_trazabilidad.json"]
    versions = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    metrics, events, config, metadata = load_results(str(ROOT), versions)
    required = {"session", "channel", "window_s", "start_s", "end_s", "state", "quality_flag"}
    if not required.issubset(metrics.columns) or metrics.empty:
        raise ValueError("La tabla de métricas está vacía o incompleta.")
    if not {"file", "start_s", "end_s"}.issubset(events.columns):
        raise ValueError("La tabla de eventos está incompleta.")
except (OSError, ValueError, KeyError) as error:
    st.error(f"No se pudieron cargar los resultados: {error}")
    st.info("Genera los resultados con `uv run python scripts/run_analysis.py`.")
    st.stop()

with st.sidebar:
    st.header("Explorar registro")
    session = st.selectbox("Registro", sorted(metrics.session.unique()))
    selected = metrics.loc[metrics.session == session]
    channel = st.selectbox("Derivación", selected.channel.unique())
    available_metrics = [name for name in METRICS if name in selected.columns]
    metric = st.selectbox("Indicador", available_metrics,
                          format_func=lambda name: METRICS[name][0])
    durations = sorted(selected.loc[selected.channel == channel, "window_s"].unique())
    window = st.selectbox("Duración de la ventana", durations,
                          format_func=lambda value: f"{value:g} segundos")
    st.caption("Cada punto resume una ventana. Las ventanas solapadas no son observaciones independientes.")
    exclude = st.checkbox("Ocultar ventanas marcadas por calidad", value=False)

subset = selected.loc[(selected.channel == channel) & (selected.window_s == window)]
start, end = float(subset.start_s.min()), float(subset.end_s.max())
interval = st.slider("Intervalo a revisar (segundos desde el inicio del registro)",
                     min_value=start, max_value=end, value=(start, end), step=1.0)
if interval[0] >= interval[1]:
    st.info("Amplía el intervalo para visualizar los resultados.")
    st.stop()
session_events = events.loc[events.file.isin([str(session), str(session) + ".edf"])]
visible_events = session_events.loc[(session_events.end_s > interval[0]) &
                                  (session_events.start_s < interval[1])]
centers = (subset.start_s + subset.end_s) / 2
visible = subset.loc[centers.between(*interval)]
flagged = visible.quality_flag.astype(str).str.lower().isin(["true", "1"])

c1, c2, c3 = st.columns(3)
c1.metric("Registro", str(session))
c2.metric("Intervalo seleccionado", f"{interval[1] - interval[0]:g} s")
c3.metric("Ventanas para revisar", f"{int(flagged.sum())} / {len(visible)}")
if visible_events.empty:
    st.info("No hay crisis anotadas que coincidan con este intervalo. Esto no establece normalidad del EEG.")
else:
    periods = "; ".join(f"{r.start_s:g}–{r.end_s:g} s" for r in visible_events.itertuples())
    st.info(f"Crisis según anotaciones del registro: {periods}.")

fig = trend_figure(selected, session_events, channel, window, metric, interval, exclude)
st.plotly_chart(fig, width="stretch", key="trend")
EXPLANATIONS = {
    "rms_uv": "Resume la amplitud de la señal respecto a su media en cada ventana.",
    "ll_uv_per_sample": "Resume cuánto cambia la amplitud entre muestras consecutivas. Depende de la frecuencia de muestreo.",
    "entropy": "Describe cómo se distribuye la potencia entre frecuencias: valores altos indican una distribución más uniforme.",
    "power_uv2": "Resume la potencia espectral entre 0.5 y 40 Hz.",
}
description = EXPLANATIONS.get(metric, "Describe la potencia en la banda seleccionada.")
if metric.endswith("_rel"):
    description = "Porcentaje de la potencia de 0.5–40 Hz que corresponde a la banda seleccionada."
st.caption(description + " Los cambios requieren revisión del trazado EEG y del contexto clínico.")
if flagged.any():
    st.warning("Las banderas automáticas orientan la revisión; no confirman artefactos. Ocultarlas puede excluir actividad relevante.")

with st.expander("Detalles del análisis"):
    st.write("Intervalo con métricas disponibles:", f"{start:g}–{end:g} s")
    st.write("Frecuencia de muestreo:", f"{metadata.get('fs', 'No disponible')} Hz")
    st.write("Configuración utilizada para generar los resultados:")
    st.json(metadata.get("config", config), expanded=False)
    st.caption("La selección de ventana muestra resultados ya calculados; no vuelve a filtrar la señal.")

st.caption("Prototipo académico de revisión retrospectiva. Esta primera pantalla no incorpora aún trazado EEG, espectrograma ni diagnóstico automático.")
