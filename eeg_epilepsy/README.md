## Datos, adquisición y auditoría del EEG

### Objetivo

Descargar y auditar el registro chb01_03.edf de CHB-MIT,
verificar sus anotaciones, identificar canales redundantes
y generar figuras para inspeccionar el EEG crudo.

### Requisitos

- Python compatible con la versión indicada en pyproject.toml.
- uv instalado.
- Internet para la primera descarga.

### Ejecución

Desde la raíz del proyecto:

```powershell
uv sync
uv run python scripts/reproduce.py
```

El reproducible ejecuta:

1. Descarga de la señal y del resumen oficial.
2. Auditoría de cabecera, canales, unidades y eventos.
3. Comparación de canales redundantes.
4. Generación de figuras del EEG crudo y acercamientos.

Los archivos descargados se reutilizan en ejecuciones posteriores.
El manifiesto registra fuente, versión, tamaño y SHA-256.

### Organización del código

- scripts/reproduce.py: coordina la ejecución.
- src/eeg_epilepsy/data/download.py: descarga y trazabilidad.
- src/eeg_epilepsy/data/audit.py: auditoría de datos y eventos.
- src/eeg_epilepsy/data/check_channels.py: comparación de
  canales y figuras de inspección.

### Datos

- Dataset: CHB-MIT Scalp EEG Database, versión 1.0.0.
- Señal: chb01_03.edf.
- Anotaciones: chb01-summary.txt.
- Frecuencia verificada: 256 Hz.
- Duración verificada: 3600 s.
- Canales almacenados: 23.
- Unidades declaradas en EDF: microvoltios.
- Crisis anotada: 2996–3036 s.
- Intervalo de análisis: 2876–3156 s.

MNE entrega las señales EEG en voltios. El código convierte
una sola vez a microvoltios para comparar y visualizar.

### Resultados generados

En data/raw/:

- chb01_03.edf
- chb01-summary.txt
- download_manifest.json

En data/derived/:

- inventory.json
- channels_audit.csv
- events.csv
- channel_comparisons.json

En figures/:

- Figura general del EEG crudo.
- eeg_raw_inicio.png
- eeg_raw_durante.png
- eeg_raw_final.png
- eeg_raw_transitorio.png

### Decisiones documentadas

- T8-P8-0 y T8-P8-1 son duplicados exactos.
  Se conserva T8-P8-0.

- T7-P7 y P7-T7 presentan polaridad invertida y un
  desplazamiento constante de 0.390720391 microvoltios.
  Se conserva T7-P7.

Estas decisiones se documentan en
data/derived/channel_selection.json, creado manualmente.
El procesamiento posterior debe leerlo y aplicar las exclusiones.
El archivo EDF original no se modifica.

### Advertencias y límites

La advertencia de MNE sobre nombres T8-P8 duplicados es esperada:
MNE añade sufijos para distinguir las dos entradas.

La auditoría y la comparación de canales no equivalen a una
evaluación completa de calidad. La inspección visual inicial
se realizó en cuatro derivaciones.

El proyecto realiza revisión retrospectiva descriptiva.
Un segmento sin crisis anotada no representa un control sano.

### Referencia

Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0).
PhysioNet. https://doi.org/10.13026/C2K01R

Fuente oficial:
https://physionet.org/content/chbmit/1.0.0/

## Descarga reproducible y procesamiento con exclusiones

Comandos desde la raíz del proyecto:

```powershell
uv sync
uv run python scripts/download_data.py
uv run python scripts/reproduce.py
uv run python -m eeg_epilepsy.procesamiento
```

El script `scripts/download_data.py` es la entrada explícita para descargar el EDF y el resumen oficial. `scripts/reproduce.py` ejecuta descarga, auditoría y comparación de canales. El EDF no se modifica y no se convierte en una versión reducida.

`data/derived/channel_selection.json` es el contrato de selección para análisis. El módulo `src/eeg_epilepsy/data/channel_selection.py` lo lee y excluye `T8-P8-1` (duplicado exacto de `T8-P8-0`) y `P7-T7` (redundante con `T7-P7`) solo de los canales seleccionados para calcular métricas. Estas exclusiones no declaran mala calidad de los canales. El pipeline de ejemplo `src/eeg_epilepsy/procesamiento.py` aplica ese contrato antes de procesar.

**Fuente de descarga:** [PhysioNet — CHB-MIT Scalp EEG Database, versión 1.0.0](https://physionet.org/content/chbmit/1.0.0/). El script usa la ruta de archivos de PhysioNet para `chb01/chb01_03.edf` y `chb01/chb01-summary.txt`, y deja la URL y SHA-256 en `data/raw/download_manifest.json`.

**Unidades:** MNE entrega los datos en voltios. El EDF original se conserva y el pipeline convierte los arreglos destinados al análisis a microvoltios una sola vez. No multiplicar nuevamente por 1e6.

**Interpretación:** las anotaciones oficiales son la referencia. Un periodo sin crisis anotada no debe llamarse “normal”; las ventanas solapadas no son observaciones independientes. El filtrado `sosfiltfilt` es retrospectivo/no causal y requiere inspección de bordes.

## Cuadro rojo: métricas y análisis (`metrics.py`, `analysis.py`)

```powershell
uv run python scripts/run_analysis.py          # ventana 2876–3156 s (config/analysis_config.json)
uv run python scripts/run_analysis.py --full   # registro completo (más lento)
uv run pytest
```

**Código**

- `src/eeg_epilepsy/metrics.py`: definición de bandas, RMS centrado (µV), longitud de línea (µV/muestra),
  PSD de Welch, potencias absolutas/relativas, entropía espectral normalizada y `epoch_features()`,
  que llama a todas las anteriores. Señal plana → relativas y entropía en NaN (nunca 0).
- `src/eeg_epilepsy/analysis.py`:
  - `analyze_file()`: lectura → canales (aplica `channel_selection.json`) → anotaciones oficiales →
    por canal: µV, filtro, épocas de 2/4/8 s, calidad y `epoch_features()` → `DataFrame`.
  - `descriptive_summary()`: mediana, IQR, n y % marcado por calidad por canal × duración × estado
    (`exclude_flagged=True` para ver qué cambia al excluir épocas marcadas).
  - `export_results()`: escribe en `data/derived/`.
  - Auxiliares: `binary_subset()` (ictal vs sin crisis, sin transición), `psd_by_state()`, `run_analysis()`.
- `config/analysis_config.json`: filtro, épocas, Welch, bandas, calidad y ventana de análisis. El
  dashboard debe leer esta misma configuración.

**Reutiliza** de `data/`: `audit.read_events` (anotaciones con validación de límites y solapes),
`audit.read_original_header` (límites de adquisición) y `channel_selection` (exclusiones).

**Interfaz verde mínima** (`io.py`, `preprocessing.py`, `segmentation.py`, `quality.py`): versiones mínimas
para poder ejecutar y probar el cuadro rojo. Si existen las definitivas, basta con respetar estas firmas:
`read_edf(path) -> (raw, meta)`, `get_channel_uv(raw, canal)`, `preprocess_channel(x_uv, fs, low, high, order)`,
`edge_review(raw_uv, filt_uv)`, `make_epochs(n_muestras, fs, duracion_s, intervalos, step_fraction)`,
`epoch_array(x, epoca, fs)` y `epoch_quality(...)`.

**Salidas en `data/derived/`**

| Archivo | Contenido |
|---|---|
| `metricas.csv` | una fila por canal × duración × época (métricas, estado, `ictal_fraction`, banderas de calidad) |
| `resumen_descriptivo.csv` | mediana, IQR, n y % marcado, con todas las épocas |
| `resumen_descriptivo_sin_marcadas.csv` | lo mismo excluyendo épocas con bandera de calidad |
| `psd_mediana_por_estado.csv` | PSD mediana por canal × duración × estado |
| `configuracion_y_trazabilidad.json` | configuración usada, canales excluidos, límites de adquisición, SHA-256 (verificado contra `download_manifest.json`), versiones |

**Decisiones y límites**

- Las métricas se calculan solo en épocas completas dentro de 2876–3156 s; el filtro se aplica al
  registro completo antes de recortar. El tiempo se mantiene como segundos desde el inicio del EDF.
- Estado de época: `ictal` (totalmente cubierta por la crisis oficial), `transicion` (cubierta en parte;
  se excluye de la comparación binaria) y `sin_crisis_anotada` (nunca «normal»).
- Banderas de calidad (sobre la señal cruda; ayudan a la revisión visual, no son veredictos): señal
  plana (σ < 0.1 µV), amplitud extrema (|x| > 600 µV, umbral configurable) y saturación (muestra en el
  límite ±800 µV declarado en la cabecera). Amplitud grande por sí sola no es saturación. En chb01_03 no
  hay muestras en los límites; el máximo absoluto del registro es 721 µV.
- Las épocas solapadas no son observaciones independientes: los resúmenes son descriptivos, sin valores p.
- Observación (verificar con el trazado antes de interpretar beta y 30–40 Hz): la PSD sin crisis tiene
  picos estrechos y estacionarios en 16 y 32 Hz en casi todos los canales; no cambian con la crisis y son
  compatibles con interferencia no cerebral. No se aplicó filtro notch.
