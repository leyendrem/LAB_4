from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import mne
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EDF_PATH = PROJECT_ROOT / "data" / "raw" / "chb01_03.edf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived"
FIGURES_DIR = PROJECT_ROOT / "figures"


def compare_channels(raw, index_a, index_b, inverted=False):
    """Compara dos canales en todo el registro, por bloques de 60 s."""
    fs = float(raw.info["sfreq"])
    chunk_size = round(60 * fs)

    max_difference_uv = 0.0
    sample_count = 0

    # Estadísticas acumuladas de la diferencia.
    mean_difference_uv = 0.0
    difference_m2 = 0.0

    for start in range(0, raw.n_times, chunk_size):
        stop = min(start + chunk_size, raw.n_times)

        # MNE entrega EEG en voltios; convertir una vez a microvoltios.
        signals_uv = raw.get_data(
            picks=[index_a, index_b],
            start=start,
            stop=stop,
        ) * 1e6

        if not np.isfinite(signals_uv).all():
            raise ValueError(
                "Hay datos no finitos en los canales comparados."
            )

        # Si son inversos, su suma debería ser cero.
        if inverted:
            difference = signals_uv[0] + signals_uv[1]
        else:
            difference = signals_uv[0] - signals_uv[1]

        max_difference_uv = max(
            max_difference_uv,
            float(np.max(np.abs(difference))),
        )

        # Combinar medias y variancias de bloques de forma estable.
        block_count = int(difference.size)
        block_mean = float(np.mean(difference))
        block_m2 = float(
            np.sum((difference - block_mean) ** 2)
        )

        combined_count = sample_count + block_count
        delta = block_mean - mean_difference_uv

        difference_m2 += (
            block_m2
            + delta ** 2
            * sample_count
            * block_count
            / combined_count
        )

        mean_difference_uv += (
            delta * block_count / combined_count
        )

        sample_count = combined_count

    if sample_count == 0:
        raise ValueError("El registro no contiene muestras.")

    variance_uv2 = max(0.0, difference_m2 / sample_count)
    std_difference_uv = float(np.sqrt(variance_uv2))

    rmse_uv = float(
        np.sqrt(variance_uv2 + mean_difference_uv ** 2)
    )

    # Tolerancia numérica; no es un umbral de calidad clínica.
    tolerance_uv = 1e-6

    return {
        "channel_a": raw.ch_names[index_a],
        "channel_b": raw.ch_names[index_b],
        "comparison": (
            "a_equals_minus_b" if inverted else "a_equals_b"
        ),
        "samples_compared": int(sample_count),
        "max_difference_uv": float(max_difference_uv),
        "rmse_uv": rmse_uv,
        "mean_difference_uv": float(mean_difference_uv),
        "std_difference_uv": std_difference_uv,
        "tolerance_uv": tolerance_uv,
        "numerically_equivalent": bool(
            max_difference_uv <= tolerance_uv
        ),
    }


def main():
    audit_path = OUTPUT_DIR / "channels_audit.csv"
    events_path = OUTPUT_DIR / "events.csv"

    for path in (EDF_PATH, audit_path, events_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Falta {path.name}. "
                "Ejecuta primero la descarga y la auditoría."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with audit_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        audit = list(csv.DictReader(file))

    with events_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        events = [
            row
            for row in csv.DictReader(file)
            if row["file"] == EDF_PATH.name
        ]

    raw = mne.io.read_raw_edf(
        EDF_PATH,
        preload=False,
        verbose=False,
    )

    try:
        # Verificar que el CSV corresponde al EDF actual.
        if len(audit) != len(raw.ch_names):
            raise ValueError(
                "La auditoría no coincide con el número de canales."
            )

        for index, row in enumerate(audit):
            if (
                int(row["index"]) != index
                or row["mne_name"] != raw.ch_names[index]
            ):
                raise ValueError(
                    "La auditoría no coincide con el EDF. "
                    "Vuelve a ejecutar audit.py."
                )

        def find_index(original_name):
            matches = [
                int(row["index"])
                for row in audit
                if row["original_name"] == original_name
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Se esperaba un canal único: {original_name}"
                )

            return matches[0]

        # Encontrar ambas entradas originales T8-P8.
        duplicate_indices = [
            int(row["index"])
            for row in audit
            if row["original_name"] == "T8-P8"
        ]

        if len(duplicate_indices) != 2:
            raise ValueError(
                "Se esperaban dos entradas T8-P8 en este archivo."
            )

        results = [
            compare_channels(
                raw,
                duplicate_indices[0],
                duplicate_indices[1],
            ),
            compare_channels(
                raw,
                find_index("T7-P7"),
                find_index("P7-T7"),
                inverted=True,
            ),
        ]

        output_path = OUTPUT_DIR / "channel_comparisons.json"

        output_path.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\nCOMPARACIÓN EN TODO EL REGISTRO")

        for result in results:
            print(
                f"\n{result['channel_a']} / "
                f"{result['channel_b']}"
            )
            print(f"Comparación: {result['comparison']}")
            print(
                "Muestras comparadas: "
                f"{result['samples_compared']}"
            )
            print(
                "Diferencia máxima: "
                f"{result['max_difference_uv']:.9g} µV"
            )
            print(f"RMSE: {result['rmse_uv']:.9g} µV")
            print(
                "Media de diferencia: "
                f"{result['mean_difference_uv']:.9g} µV"
            )
            print(
                "Desviación de diferencia: "
                f"{result['std_difference_uv']:.9g} µV"
            )
            print(
                "Equivalentes exactos dentro de la tolerancia: "
                f"{result['numerically_equivalent']}"
            )

        # Selección provisional para inspección visual.
        channels = [
            "FP1-F7",
            "F7-T7",
            "FP2-F8",
            "F8-T8",
        ]

        fs = float(raw.info["sfreq"])
        start_s, end_s = 2876.0, 3156.0

        start_sample = round(start_s * fs)
        stop_sample = round(end_s * fs)

        if stop_sample > raw.n_times:
            raise ValueError(
                "El intervalo solicitado supera la duración del EDF."
            )

        signals_uv = raw.get_data(
            picks=channels,
            start=start_sample,
            stop=stop_sample,
        ) * 1e6

        if not np.isfinite(signals_uv).all():
            raise ValueError(
                "El segmento para visualizar tiene datos no finitos."
            )

        # Preservar segundos desde el inicio del EDF.
        times_s = (
            start_sample + np.arange(signals_uv.shape[1])
        ) / fs

        # Usar la misma escala para las cuatro derivaciones.
        amplitude_limit = max(
            1.0,
            float(np.max(np.abs(signals_uv))) * 1.05,
        )

        fig, axes = plt.subplots(
            len(channels),
            1,
            figsize=(14, 9),
            sharex=True,
            sharey=True,
        )

        try:
            for axis, name, signal in zip(
                axes, channels, signals_uv
            ):
                axis.plot(
                    times_s,
                    signal,
                    linewidth=0.45,
                    color="#17365D",
                )

                for event in events:
                    event_start = float(event["start_s"])
                    event_end = float(event["end_s"])

                    if (
                        event_end <= start_s
                        or event_start >= end_s
                    ):
                        continue

                    axis.axvspan(
                        event_start,
                        event_end,
                        color="#F4A261",
                        alpha=0.3,
                        label=(
                            "Crisis anotada: "
                            f"{event_start:g}–{event_end:g} s"
                        ),
                    )

                axis.set_ylabel(f"{name}\nµV")
                axis.set_ylim(
                    -amplitude_limit,
                    amplitude_limit,
                )
                axis.grid(alpha=0.2)

            if axes[0].get_legend_handles_labels()[0]:
                axes[0].legend(loc="upper right")

            axes[-1].set_xlabel(
                "Tiempo desde el inicio del EDF (s)"
            )
            axes[-1].set_xlim(start_s, end_s)

            fig.suptitle(
                f"{EDF_PATH.name} · EEG crudo · {fs:g} Hz\n"
                "Intervalo 2876–3156 s · "
                "sin filtro aplicado por el equipo"
            )

            fig.tight_layout(rect=[0, 0, 1, 0.93])

            figure_path = FIGURES_DIR / "eeg_raw_review.png"
            fig.savefig(figure_path, dpi=200)
            zoom_intervals = [
                ("inicio", 2988.0, 3004.0),
                ("durante", 3010.0, 3020.0),
                ("final", 3028.0, 3044.0),
                ("transitorio", 3070.0, 3080.0),
            ]

            for label, zoom_start, zoom_end in zoom_intervals:
                mask = (
                    (times_s >= zoom_start)
                    & (times_s < zoom_end)
                )

                zoom_fig, zoom_axes = plt.subplots(
                    len(channels),
                    1,
                    figsize=(14, 9),
                    sharex=True,
                    sharey=True,
                )

                try:
                    for axis, name, signal in zip(
                        zoom_axes, channels, signals_uv
                    ):
                        axis.plot(
                            times_s[mask],
                            signal[mask],
                            linewidth=0.7,
                            color="#17365D",
                        )

                        for event in events:
                            event_start = float(event["start_s"])
                            event_end = float(event["end_s"])

                            if (
                                event_end <= zoom_start
                                or event_start >= zoom_end
                            ):
                                continue

                            axis.axvspan(
                                max(event_start, zoom_start),
                                min(event_end, zoom_end),
                                color="#F4A261",
                                alpha=0.3,
                                label="Crisis anotada",
                            )

                        axis.set_ylabel(f"{name}\nµV")
                        axis.set_ylim(
                            -amplitude_limit,
                            amplitude_limit,
                        )
                        axis.grid(alpha=0.2)

                    if zoom_axes[0].get_legend_handles_labels()[0]:
                        zoom_axes[0].legend(loc="upper right")

                    zoom_axes[-1].set_xlim(
                        zoom_start, zoom_end
                    )
                    zoom_axes[-1].set_xlabel(
                        "Tiempo desde el inicio del EDF (s)"
                    )

                    zoom_fig.suptitle(
                        f"{EDF_PATH.name} · EEG crudo · {fs:g} Hz\n"
                        f"{zoom_start:g}–{zoom_end:g} s · "
                        "sin filtro aplicado por el equipo"
                    )

                    zoom_fig.tight_layout(
                        rect=[0, 0, 1, 0.93]
                    )

                    zoom_path = (
                        FIGURES_DIR / f"eeg_raw_{label}.png"
                    )

                    zoom_fig.savefig(zoom_path, dpi=200)

                    print(
                        "Acercamiento guardado: "
                        f"{zoom_path.relative_to(PROJECT_ROOT)}"
                    )

                finally:
                    plt.close(zoom_fig)
        finally:
            plt.close(fig)

        print(
            "\nComparaciones guardadas en: "
            f"{output_path.relative_to(PROJECT_ROOT)}"
        )
        print(
            "Figura guardada en: "
            f"{figure_path.relative_to(PROJECT_ROOT)}"
        )

    finally:
        raw.close()


if __name__ == "__main__":
    main()