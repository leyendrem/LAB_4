"""Figuras interactivas para revisión retrospectiva del EEG.

No calcula diagnósticos ni modifica las métricas del pipeline.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


METRICS = {
    "rms_uv": ("Amplitud RMS centrada", "µV"),
    "ll_uv_per_sample": ("Longitud de línea media", "µV/muestra"),
    "power_uv2": ("Potencia total (0.5–40 Hz)", "µV²"),
    "entropy": ("Entropía espectral normalizada", "sin unidad"),
}
for _band, _label in {
    "delta": "Delta", "theta": "Theta", "alpha": "Alfa",
    "beta": "Beta", "high_30_40": "30–40 Hz",
}.items():
    METRICS[f"{_band}_uv2"] = (f"Potencia {_label}", "µV²")
    METRICS[f"{_band}_rel"] = (f"Potencia relativa {_label}", "%")

STATES = {
    "ictal": "Crisis anotada",
    "sin_crisis_anotada": "Sin crisis anotada",
    "transicion": "Ventana parcialmente cubierta por la crisis",
}


def trend_figure(
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    channel: str,
    window_s: float = 2,
    metric: str = "rms_uv",
    time_range: tuple[float, float] | None = None,
    exclude_flagged: bool = False,
) -> go.Figure:
    """Grafica una métrica de un canal y una duración de época.

    Cada punto se sitúa en el centro de su ventana, en segundos desde el
    inicio del EDF. Las relativas se muestran en %, sin modificar el CSV.
    Acepta una sola sesión para no mezclar relojes de registros distintos.
    Las ventanas marcadas se señalan o se ocultan dejando huecos.
    """
    if metric not in METRICS:
        raise ValueError(f"Métrica no admitida: {metric}")
    required = {"session", "channel", "window_s", "start_s", "end_s",
                "state", "quality_flag", metric}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    if metrics["session"].nunique() > 1:
        raise ValueError("Selecciona una sola sesión antes de graficar.")
    if time_range is not None and time_range[0] >= time_range[1]:
        raise ValueError("El inicio del intervalo debe ser menor que el final.")

    data = metrics.loc[(metrics.channel == channel) &
                       (metrics.window_s == window_s)].copy()
    data = data.sort_values("start_s")
    data["center_s"] = (data.start_s + data.end_s) / 2
    if time_range is not None:
        data = data.loc[data.center_s.between(*time_range)].copy()
    title, unit = METRICS[metric]
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white", title=f"{title} · {channel}",
        xaxis_title="Tiempo desde el inicio del registro (s)",
        yaxis_title=f"{title} ({unit})", hovermode="closest",
        margin=dict(l=65, r=25, t=65, b=85),
        legend=dict(orientation="h", y=-0.22),
    )
    if data.empty:
        fig.add_annotation(text="No hay datos para esta selección.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    flag = data.quality_flag.astype(str).str.lower().isin(["true", "1"])
    values = pd.to_numeric(data[metric], errors="coerce")
    if metric.endswith("_rel"):
        values = values * 100
    plotted = values.mask(flag) if exclude_flagged else values
    details = pd.DataFrame({
        "start": data.start_s, "end": data.end_s,
        "state": data.state.map(STATES).fillna(data.state),
        "quality": flag.map({True: "Requiere revisión", False: "Sin banderas automáticas"}),
    }, index=data.index)
    hover = ("Ventana: %{customdata[0]:.1f}–%{customdata[1]:.1f} s"
             "<br>Valor: %{y:.3g} " + unit +
             "<br>%{customdata[2]}<br>%{customdata[3]}<extra></extra>")
    fig.add_trace(go.Scatter(
        x=data.center_s, y=plotted, mode="lines", name=title,
        line=dict(color="#146C94", width=2), connectgaps=False,
        customdata=details.to_numpy(), hovertemplate=hover,
    ))
    if not exclude_flagged and flag.any():
        fig.add_trace(go.Scatter(
            x=data.loc[flag, "center_s"], y=values.loc[flag],
            mode="markers", name="Revisar calidad",
            marker=dict(color="#C87800", symbol="diamond", size=8),
            customdata=details.loc[flag].to_numpy(), hovertemplate=hover,
        ))

    if not {"start_s", "end_s"}.issubset(events.columns):
        raise ValueError("Los eventos deben incluir start_s y end_s.")
    selected_events = events
    if "file" in events.columns:
        session = str(data.session.iloc[0])
        selected_events = events.loc[events.file.isin([session, session + ".edf"])]
    lower, upper = time_range or (float(data.start_s.min()), float(data.end_s.max()))
    for event in selected_events.itertuples():
        if event.end_s <= lower or event.start_s >= upper:
            continue
        fig.add_vrect(
            x0=max(lower, event.start_s), x1=min(upper, event.end_s),
            fillcolor="#D64550", opacity=0.14, line_width=0,
            annotation_text="Crisis anotada", annotation_position="top left",
        )
    fig.update_xaxes(range=[lower, upper])
    if metric.endswith("_rel"):
        fig.update_yaxes(range=[0, 100])
    elif metric == "entropy":
        fig.update_yaxes(range=[0, 1])
    fig.add_annotation(
        text=f"Ventanas de {window_s:g} s · Indicador descriptivo, no diagnóstico automático",
        x=0, y=-0.18, xref="paper", yref="paper", showarrow=False, xanchor="left",
        font=dict(size=11, color="#586575"),
    )
    return fig
