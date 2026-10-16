\documentclass[aspectratio=169,10pt]{beamer}
\usepackage[spanish,es-nodecimaldot]{babel}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,graphicx,booktabs,tabularx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}
\usetheme{Madrid}
\usecolortheme{default}
\title[Revisión EEG]{EEG y revisión de crisis epilépticas}
\subtitle{Laboratorio de Señales Bioeléctricas y Visualización Interactiva}
\author{Equipo de laboratorio}
\institute{Prototipo retrospectivo de análisis y dashboard}
\date{\today}
\begin{document}
\begin{frame}
\titlepage
\end{frame}

\begin{frame}{1. Solicitud clínica y problemática}
\begin{block}{Necesidad del neurólogo}
Revisar registros EEG prolongados, navegar el trazado, examinar calidad y describir cambios de amplitud, actividad rítmica y distribución espectral alrededor de crisis anotadas.
\end{block}
\begin{itemize}
\item Conservar las anotaciones originales y comparar ventanas ictales con ventanas \emph{sin crisis anotada}.
\item Explicar cada indicador: fisiología, unidades, parámetros y límites.
\item Pregunta central: ¿qué cambios acompañan a las crisis anotadas y cuáles pueden confundirse con artefactos o variación fisiológica?
\end{itemize}
\alert{No diagnostica, no predice crisis ni localiza un foco causal.}
\end{frame}

\begin{frame}{2. Fisiología: del cerebro al EEG}
\begin{columns}[T]
\column{0.58\textwidth}
\begin{itemize}
\item El EEG de superficie refleja principalmente corrientes postsinápticas sumadas en poblaciones corticales.
\item No mide directamente la descarga de una neurona individual.
\item El montaje bipolar mide diferencias: \(V_{\mathrm{FP1-F7}}=V_{\mathrm{FP1}}-V_{\mathrm{F7}}\).
\item Movimiento/parpadeos pueden aumentar actividad lenta; músculo puede aumentar amplitud, longitud de línea y potencia rápida.
\end{itemize}
\column{0.38\textwidth}
\begin{block}{Consecuencia}
Una métrica o banda aislada no confirma una crisis. Se interpreta junto al trazado, anotaciones y calidad.
\end{block}
\end{columns}
\end{frame}

\begin{frame}{3. Datos y adquisición}
\begin{columns}[T]
\column{0.52\textwidth}
\begin{tabular}{ll}
\toprule
Dataset & CHB-MIT v1.0.0\\
Archivo & \texttt{chb01\_03.edf}\\
Frecuencia & 256 Hz\\
Duración & 3600 s\\
Canales & 23\\
Crisis anotada & 2996--3036 s\\
Ventana & 2876--3156 s\\
\bottomrule
\end{tabular}
\column{0.44\textwidth}
\begin{block}{Trazabilidad}
\begin{itemize}
\item Resumen oficial \texttt{chb01-summary.txt}.
\item SHA-256, tamaño, versión y metadatos.
\item Conservar segundos desde el inicio del EDF.
\item EDF original sin modificación.
\end{itemize}
\end{block}
\end{columns}
\footnotesize Fuente: PhysioNet, CHB-MIT Scalp EEG Database, DOI 10.13026/C2K01R.
\end{frame}

\begin{frame}{4. Indicadores y límites}
\small
\begin{tabularx}{\textwidth}{p{0.24\textwidth}p{0.35\textwidth}X}
\toprule
Indicador & Describe & Precaución\\
\midrule
RMS centrado ($\mu$V) & Amplitud respecto a la media & Montaje y artefactos\\
Longitud de línea ($\mu$V/muestra) & Cambio entre muestras consecutivas & Depende del muestreo y filtro\\
Potencia absoluta ($\mu$V$^2$) & Potencia integrada en banda & No es diagnóstico\\
Potencia relativa & Fracción de potencia 0.5--40 Hz & Puede subir si potencia absoluta baja\\
Entropía espectral & Distribución de potencia & Depende de bins y rango\\
Solapamiento ictal & Tiempo cubierto por anotación & No es detección automática\\
\bottomrule
\end{tabularx}
\vspace{2mm}
\footnotesize Bandas: delta 0.5--4 Hz, theta 4--8 Hz, alfa 8--13 Hz, beta 13--30 Hz. La banda 30--40 Hz no equivale a toda gamma.
\end{frame}

\begin{frame}{5. Pipeline y arquitectura}
\centering
\resizebox{0.95\textwidth}{!}{%
\begin{tikzpicture}[node distance=5mm and 7mm,
box/.style={draw,rounded corners,align=center,text width=2.55cm,minimum height=0.85cm,font=\scriptsize},
arr/.style={-{Latex[length=2mm]},thick}]
\node[box] (download) {Descarga\\EDF + resumen};
\node[box,right=of download] (audit) {Auditoría\\cabecera/eventos};
\node[box,right=of audit] (select) {Selección\\de canales};
\node[box,right=of select] (process) {Lectura, filtro,\\épocas y métricas};
\node[box,below=of process] (derived) {CSV/JSON derivados\\PSD y trazabilidad};
\node[box,left=of derived] (app) {Streamlit\\controles comunes};
\node[box,left=of app] (views) {Plotly\\vistas y detalle};
\draw[arr] (download)--(audit);
\draw[arr] (audit)--(select);
\draw[arr] (select)--(process);
\draw[arr] (process)--(derived);
\draw[arr] (derived)--(app);
\draw[arr] (app)--(views);
\end{tikzpicture}}
\vspace{2mm}
\begin{itemize}
\item Configuración central: \texttt{config/analysis\_config.json}.
\item Salidas: métricas por época, resúmenes descriptivos, PSD por estado y trazabilidad.
\item La interfaz consume resultados ya calculados; el detalle puede volver a leer el EDF.
\end{itemize}
\end{frame}

\begin{frame}{6. Diseño del dashboard}
\begin{columns}[T]
\column{0.48\textwidth}
\begin{enumerate}
\item Evolución temporal del indicador.
\item Comparación espectral (PSD).
\item Distribución por estado: histogramas, mediana y RIC.
\item Detalle EEG crudo/filtrado y espectrograma.
\item Mapa aproximado de derivaciones.
\end{enumerate}
\column{0.48\textwidth}
\begin{block}{Controles compartidos}
Registro, derivación bipolar, métrica, duración de ventana, intervalo temporal y opción de ocultar ventanas marcadas por calidad.
\end{block}
\begin{block}{Comunicación segura}
“Sin crisis anotada” no significa “normal”. Las banderas requieren revisión; no confirman artefactos.
\end{block}
\end{columns}
\end{frame}

\begin{frame}{7. Auditoría y hallazgos iniciales}
\begin{itemize}
\item Se identificaron dos redundancias: \texttt{T8-P8-0}/\texttt{T8-P8-1} idénticas y \texttt{T7-P7}/\texttt{P7-T7} con polaridad invertida y desplazamiento.
\item Se documentaron cambios de amplitud/morfología alrededor de la crisis, diferentes entre derivaciones.
\item El tramo 3070--3080 s se propone para inspección: no se etiqueta automáticamente como artefacto ni como otra crisis.
\item Se observaron picos estrechos en 16 y 32 Hz compatibles con interferencia; requieren contrastarse con el trazado.
\end{itemize}
\alert{La inspección inicial cubre segmentos y derivaciones seleccionados; no certifica la calidad de todo el registro.}
\end{frame}

\begin{frame}{8. Validación y calidad}
\begin{itemize}
\item Probar seno de 10 Hz, duplicación de amplitud, señal constante y anotación que cruza una época.
\item Comprobar conversión: amplitud en V a $\mu$V por $10^6$; potencia por $10^{12}$.
\item Comparar épocas de 2, 4 y 8 s; las ventanas solapadas no son independientes.
\item Informar mediana, RIC, número de ventanas y porcentaje marcado por calidad.
\item QA comprueba implementación y trazabilidad; no demuestra utilidad clínica.
\end{itemize}
\end{frame}

\begin{frame}{9. Conclusiones y límites}
\begin{block}{Qué aporta}
Pipeline reproducible de auditoría, selección de canales, filtrado, segmentación, métricas, banderas de calidad y visualización interactiva.
\end{block}
\begin{itemize}
\item Los resultados son descriptivos para el registro y las anotaciones examinadas.
\item \texttt{sosfiltfilt} es bidireccional/no causal: no sustenta latencia en tiempo real.
\item Un paciente y una crisis no permiten inferencia poblacional.
\item No afirmar diagnóstico, predicción, causalidad ni generalización clínica.
\end{itemize}
\centering \Large Preguntas
\end{frame}
\end{document}
