"""Primera pantalla de revisión EEG: tendencias de resultados existentes."""
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
st.set_page_config(page_title="Revisión de EEG", page_icon="🧠", layout="wide")
try:
    from eeg_epilepsy.visualization import METRICS, trend_figure, psd_figure, histogram_figure, histogram_summary, signal_figure, spectrogram_figure, electrode_map_figure
except (ModuleNotFoundError, ImportError):
    st.error("No se encontró la versión necesaria de visualization.py.")
    st.code(str(ROOT / "src/eeg_epilepsy/visualization.py"))
    st.info("Copia el archivo visualization.py actualizado en esa ubicación y reinicia Streamlit.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_results(root: str, versions: tuple):
    """La firma de archivos invalida el caché cuando cambian los resultados."""
    base = Path(root)
    config = json.loads((base / "config/analysis_config.json").read_text())
    derived = base / config["data"]["derived_dir"]
    metrics = pd.read_csv(derived / "metricas.csv")
    events = pd.read_csv(derived / "events.csv", encoding="utf-8-sig")
    provenance = json.loads((derived / "configuracion_y_trazabilidad.json").read_text())
    psd_path = derived / "psd_mediana_por_estado.csv"
    psd = pd.read_csv(psd_path) if psd_path.exists() else pd.DataFrame()
    return metrics, events, config, provenance.get("metadata", {}), psd


st.title("Revisión de EEG")
st.caption("Versión 0.4 · Resumen por estado con mediana y rango intercuartílico")
try:
    config_path = ROOT / "config/analysis_config.json"
    config = json.loads(config_path.read_text())
    derived = ROOT / config["data"]["derived_dir"]
    files = [config_path, derived / "metricas.csv", derived / "events.csv",
             derived / "configuracion_y_trazabilidad.json"]
    psd_path = derived / "psd_mediana_por_estado.csv"
    if psd_path.exists():
        files.append(psd_path)
    versions = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    metrics, events, config, metadata, psd = load_results(str(ROOT), versions)
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

trend_tab, psd_tab, hist_tab, detail_tab, map_tab = st.tabs([
    "Evolución temporal",
    "Comparación espectral",
    "Distribución de indicadores",
    "Detalle EEG",
    "Ubicación de derivaciones",
])
with trend_tab:
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

with psd_tab:
    st.subheader("¿Cómo cambia la potencia entre frecuencias?")
    analysis_range = metadata.get("analysis_window_s", [start, end])
    st.info(f"Vista global del análisis: {analysis_range[0]:g}–{analysis_range[1]:g} s. "
            "La PSD guardada incluye ventanas marcadas y no cambia con el intervalo temporal ni con el filtro de calidad.")
    source_session = Path(str(metadata.get("source", ""))).stem
    if source_session != str(session):
        st.warning("La procedencia de la PSD no coincide con el registro seleccionado.")
    elif psd.empty:
        st.info("No hay un archivo de PSD exportado. Ejecuta el análisis para generarlo.")
    else:
        logarithmic = st.checkbox("Escala logarítmica de potencia", value=True)
        st.plotly_chart(psd_figure(psd, channel, window, logarithmic), width="stretch", key="psd")
        st.caption("Curvas medianas por estado; n indica el número de ventanas, no pacientes. Las ventanas de transición se excluyen de esta comparación.")
        st.caption("El proyecto documenta picos estrechos en 16 y 32 Hz que requieren revisión por posible interferencia. Esta vista no establece su origen.")

with hist_tab:
    st.subheader("¿Cómo se distribuye el indicador en cada estado?")
    st.caption("Se incluyen solo ventanas completas dentro del intervalo seleccionado. "
               "Cada estado se normaliza a 100% para comparar distribuciones de distinto tamaño. "
               "Se excluyen ventanas de transición y valores no definidos.")
    st.plotly_chart(histogram_figure(selected, channel, window, metric, interval, exclude),
                    width="stretch", key="histogram")
    summary = histogram_summary(selected, channel, window, metric, interval, exclude)
    st.markdown("**Resumen del indicador por estado**")
    if summary.empty:
        st.info("No hay ventanas válidas para calcular el resumen en esta selección.")
    else:
        st.dataframe(summary.style.format({col: "{:.3g}" for col in summary.columns[2:]}),
                     hide_index=True, width="stretch")
    st.caption("Q1 y Q3 son los percentiles 25 y 75: delimitan el 50% central de los valores. "
               "El rango intercuartílico (RIC) es Q3 − Q1. Se usan las mismas ventanas y filtros del histograma. "
               "n cuenta ventanas, no pacientes; las ventanas solapadas no son observaciones independientes.")
    st.caption("El solapamiento entre distribuciones y el número de ventanas deben considerarse al interpretar. No se realizan pruebas de significancia ni clasificación diagnóstica.")

with detail_tab:
    st.subheader("Inspección del segmento EEG")
    st.caption("Trazado y espectrograma de la derivación seleccionada. El color del espectrograma representa potencia, no probabilidad de crisis.")
    edf_path = ROOT / config["data"]["edf"]
    if not edf_path.exists():
        st.info("Para habilitar esta vista, descarga el registro EDF desde la carpeta eeg_epilepsy:")
        st.code("uv run python scripts/download_data.py", language="bash")
    elif edf_path.stem != str(session):
        st.warning("El EDF configurado no corresponde al registro seleccionado.")
    elif st.checkbox("Cargar trazado y espectrograma", key="load_eeg"):
        @st.cache_data(max_entries=4, show_spinner="Leyendo y filtrando la derivación…")
        def load_channel(path, version, channel_name, filter_json, expected_hash):
            import hashlib
            from eeg_epilepsy.io import read_edf, get_channel_uv
            from eeg_epilepsy.preprocessing import preprocess_channel
            if expected_hash:
                with open(path, "rb") as source:
                    actual = hashlib.file_digest(source, "sha256").hexdigest()
                if actual != expected_hash:
                    raise ValueError("El EDF no coincide con la huella de los resultados guardados.")
            raw, meta = read_edf(path)
            try:
                original = get_channel_uv(raw, channel_name)
                fc = json.loads(filter_json)
                stages = preprocess_channel(original, meta["fs"], fc["low_hz"], fc["high_hz"], fc["order"])
                return stages, meta["fs"]
            finally:
                raw.close()
        try:
            import numpy as np
            used_config = metadata.get("config", config)
            fc = used_config["filter"]
            stages, fs = load_channel(str(edf_path), (edf_path.stat().st_mtime_ns, edf_path.stat().st_size),
                                      channel, json.dumps(fc, sort_keys=True), metadata.get("traceability", {}).get("edf_sha256"))
            if interval[1] > len(stages["raw_uv"]) / fs:
                raise ValueError("El intervalo excede la duración del EDF.")
            max_duration = min(30.0, interval[1] - interval[0])
            duration = st.number_input("Duración del detalle (s)", min_value=1.0,
                                       max_value=max_duration, value=min(10.0, max_duration), step=1.0)
            detail_start = st.number_input("Inicio del detalle (s)", min_value=interval[0],
                                           max_value=interval[1] - duration, value=interval[0], step=1.0)
            show_raw = st.checkbox("Comparar con señal original", value=False)
            first, last = int(round(detail_start * fs)), int(round((detail_start + duration) * fs))
            original = stages["raw_uv"][first:last]
            filtered = stages["filtered_uv"][first:last]
            st.plotly_chart(signal_figure(original, filtered, fs, first/fs, channel, session_events, show_raw),
                            width="stretch", key="eeg_signal")
            st.caption(f"Filtro {fc['low_hz']:g}–{fc['high_hz']:g} Hz, aplicado al canal completo antes de recortar. "
                       "El trazado conserva todas las muestras; las ventanas marcadas por calidad no se borran de esta vista.")
            wc = used_config["welch"]
            if duration >= wc["segment_s"]:
                st.plotly_chart(spectrogram_figure(filtered, fs, first/fs, channel, session_events,
                                                   wc["segment_s"], wc["overlap"], fc["low_hz"], fc["high_hz"]),
                                width="stretch", key="spectrogram")
                st.caption(f"Segmentos Hann de {wc['segment_s']:g} s y solapamiento de {100*wc['overlap']:g}%. "
                           "La escala de color se ajusta al segmento; consulta los valores al comparar vistas.")
            else:
                st.info(f"Amplía el detalle a {wc['segment_s']:g} s para calcular el espectrograma.")
        except (OSError, ValueError, KeyError, ImportError) as error:
            st.error(f"No se pudo mostrar el EEG: {error}")

# Quinta pestaña: representación de la derivación seleccionada.
with map_tab:
    st.subheader("Ubicación de derivaciones EEG")
    st.write(f"**Derivación seleccionada:** {channel}")
    st.caption(
        "La línea roja conecta las posiciones de los dos electrodos "
        "de la derivación bipolar seleccionada."
    )

    image_path = ROOT / "assets" / "eeg_10_20.png"
    if not image_path.is_file():
        st.error("No se encontró assets/eeg_10_20.png. Revisa la ruta de la imagen.")
    else:
        try:
            image, found = electrode_map_figure(channel, image_path)
            left, center, right = st.columns([1, 2, 1])
            with center:
                st.image(image, caption=f"Derivación: {channel}", width="stretch")
            if not found:
                st.warning(
                    "No se pueden ubicar ambos electrodos en este esquema. "
                    "Por ejemplo, FT9 y FT10 no aparecen en la imagen original."
                )
            st.caption(
                "Referencia visual del sistema 10-20: T3 equivale a T7; "
                "T4 a T8; T5 a P7; T6 a P8. "
                "La ubicación es aproximada y no representa el origen de una crisis."
            )
        except (OSError, ValueError, ImportError) as error:
            st.error(f"No se pudo mostrar el mapa EEG: {error}")

with st.expander("Detalles del análisis"):
    st.write("Intervalo con métricas disponibles:", f"{start:g}–{end:g} s")
    st.write("Frecuencia de muestreo:", f"{metadata.get('fs', 'No disponible')} Hz")
    st.write("Configuración utilizada para generar los resultados:")
    st.json(metadata.get("config", config), expanded=False)
    st.caption("La selección de ventana muestra resultados ya calculados; no vuelve a filtrar la señal.")

st.caption("Prototipo académico de revisión retrospectiva. No constituye una herramienta de diagnóstico clínico.")
