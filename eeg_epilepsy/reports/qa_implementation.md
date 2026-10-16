# Implementación QA del laboratorio EEG

## Propósito

Esta rama convierte el plan de QA en controles ejecutables sobre el pipeline EEG. El objetivo no es demostrar diagnóstico clínico. El objetivo es impedir que una entrada inválida, una etiqueta temporal desplazada o una métrica indefinida se conviertan en una salida que parezca confiable.

## Copia de trabajo y alcance

La rama `qa/controlled-validation` parte del repositorio `LAB_4/eeg_epilepsy`. El EDF original y los resultados derivados se conservan como referencia. Los nuevos escenarios usan señales pequeñas y sintéticas para que cada decisión pueda reproducirse sin depender de una descarga o de una interpretación visual subjetiva.

## Control implementado

### Unión de intervalos en `segmentation.interval_overlap`

El código decía calcular la longitud de la unión de las intersecciones, pero sumaba cada intersección de forma independiente. Si dos anotaciones se superponían, la fracción ictal podía inflarse.

La implementación nueva recorta los intervalos a la época, los ordena y une los tramos contiguos antes de acumular su longitud. Además rechaza límites no finitos o invertidos.

Esto está alineado con el problema porque la etiqueta de una época debe representar tiempo anotado real, no el número de anotaciones que la cruzan. El control `test_overlapping_annotations_are_counted_once` reproduce el escenario adverso `(2,6)` y `(4,8)` dentro de `(0,10)` y exige una duración de 6 segundos, no 8.

### Validación de parámetros en `segmentation.make_epochs`

La generación de épocas ahora rechaza frecuencia de muestreo no positiva o no finita, duración inválida y `step_fraction` fuera de `(0,1]`.

La decisión es buena calidad porque una rejilla temporal mal configurada altera el número de observaciones, los límites de las épocas y la asignación de anotaciones. El test parametrizado `test_epoch_generation_rejects_invalid_parameters` comprueba que cada caso falle de forma controlada.

### Banderas de calidad

Los escenarios verifican que una señal plana no se borre y que una amplitud extrema no se confunda automáticamente con saturación. Esto mantiene la distinción del laboratorio entre:

- dato que requiere revisión;
- amplitud que supera un umbral operativo;
- muestra que toca los rieles de adquisición;
- artefacto confirmado, afirmación que QA no debe hacer solo con una bandera.

### Métricas de una señal constante

El escenario `test_constant_epoch_returns_missing_normalized_features` exige valores ausentes para entropía y potencia relativa cuando no existe masa espectral útil. Devolver cero sería peligroso porque aparentaría una medición válida de ausencia de potencia.

## Evidencia de ejecución

Comando ejecutado:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_segmentation.py tests\test_quality.py tests\test_metrics.py tests\test_qa_controlled_scenarios.py -q
```

Resultado:

```text
34 passed
```

La suite completa del repositorio debe ejecutarse después de completar la preparación del entorno y antes de integrar la rama. La batería parcial ejecutada cubre las funciones modificadas y los módulos de mayor riesgo temporal, de calidad y de métricas.

## Siguiente control de integración

1. Ejecutar toda la suite existente junto con `test_qa_controlled_scenarios.py`.
2. Ejecutar `run_analysis.py` contra `chb01_03.edf` y verificar que el conteo de épocas, estados y hash coincida con los resultados derivados.
3. Revisar manualmente en el dashboard una ventana sin crisis anotada, una ventana ictal, una transición y una época marcada por calidad.
4. Comparar `configuracion_y_trazabilidad.json` con `analysis_config.json` y con la huella del EDF.
5. Registrar cualquier fallo en la matriz QA con entrada, resultado observado, criterio y decisión.

## Límites

Estos controles verifican implementación y trazabilidad. No convierten las anotaciones en un detector automático, no validan una utilidad clínica y no permiten inferir causalidad, foco epiléptico ni comportamiento en tiempo real.
