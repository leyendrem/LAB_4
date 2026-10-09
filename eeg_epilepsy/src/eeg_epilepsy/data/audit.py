from pathlib import Path
from collections import Counter
import csv
import json
import re

import mne


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EDF_PATH = PROJECT_ROOT / "data" / "raw" / "chb01_03.edf"
SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "chb01-summary.txt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived"


def read_original_header(path):
    """Lee nombres y parámetros originales de adquisición del EDF."""
    with path.open("rb") as file:
        fixed_header = file.read(256)

        if len(fixed_header) != 256:
            raise ValueError("La cabecera EDF está incompleta.")

        number_channels = int(
            fixed_header[252:256].decode("ascii")
        )
        record_duration_s = float(
            fixed_header[244:252].decode("ascii")
        )

        signal_header = file.read(256 * number_channels)

    if number_channels <= 0:
        raise ValueError("El EDF no declara canales válidos.")

    if len(signal_header) != 256 * number_channels:
        raise ValueError("La cabecera de canales está incompleta.")

    if record_duration_s <= 0:
        raise ValueError("La duración del bloque EDF no es válida.")

    offset = 0

    def read_field(width):
        nonlocal offset

        values = [
            signal_header[
                offset + index * width:
                offset + (index + 1) * width
            ].decode("ascii").strip()
            for index in range(number_channels)
        ]

        offset += number_channels * width
        return values

    labels = read_field(16)
    transducers = read_field(80)
    units = read_field(8)
    physical_min = read_field(8)
    physical_max = read_field(8)
    digital_min = read_field(8)
    digital_max = read_field(8)
    prefilters = read_field(80)
    samples_per_record = read_field(8)
    read_field(32)  # Campo reservado.

    acquisition = []

    for index in range(number_channels):
        samples = int(samples_per_record[index])

        p_min = float(physical_min[index])
        p_max = float(physical_max[index])
        d_min = int(digital_min[index])
        d_max = int(digital_max[index])

        if samples <= 0:
            raise ValueError(
                f"Número de muestras inválido en {labels[index]}."
            )

        if p_min >= p_max or d_min >= d_max:
            raise ValueError(
                f"Límites de adquisición inválidos en {labels[index]}."
            )

        acquisition.append({
            "index": index,
            "original_name": labels[index],
            "transducer": transducers[index],
            "original_unit": units[index],
            "physical_min": p_min,
            "physical_max": p_max,
            "digital_min": d_min,
            "digital_max": d_max,
            "prefilter": prefilters[index],
            "samples_per_record": samples,
            "record_duration_s": record_duration_s,
            "fs_hz": samples / record_duration_s,
            # Paso físico declarado por código digital.
            # No es la ganancia del amplificador analógico.
            "physical_units_per_code": (
                (p_max - p_min) / (d_max - d_min)
            ),
        })

    return labels, units, acquisition


def read_summary_block(summary_path, filename):
    """Encuentra el bloque del EDF dentro del resumen oficial."""
    text = summary_path.read_text(encoding="utf-8")
    blocks = re.split(r"File Name:\s*", text)[1:]

    matches = [
        block
        for block in blocks
        if block.splitlines()[0].strip() == filename
    ]

    if len(matches) != 1:
        raise ValueError(
            "No se encontró un bloque único para este EDF."
        )

    return matches[0]


def read_events(summary_path, filename, duration):
    """Lee y valida las crisis anotadas en el resumen oficial."""
    block = read_summary_block(summary_path, filename)

    count_match = re.search(
        r"Number of Seizures in File:\s*(\d+)",
        block,
    )

    if count_match is None:
        raise ValueError(
            "Falta el número de crisis en el resumen."
        )

    expected_count = int(count_match.group(1))

    starts = [
        float(value)
        for value in re.findall(
            r"Seizure(?:\s+\d+)? Start Time:\s*(\d+)\s+seconds",
            block,
        )
    ]

    ends = [
        float(value)
        for value in re.findall(
            r"Seizure(?:\s+\d+)? End Time:\s*(\d+)\s+seconds",
            block,
        )
    ]

    if (
        len(starts) != len(ends)
        or len(starts) != expected_count
    ):
        raise ValueError(
            "El número de anotaciones no coincide con el resumen."
        )

    intervals = sorted(zip(starts, ends))

    for start, end in intervals:
        if not 0 <= start < end <= duration:
            raise ValueError(
                f"Intervalo fuera del registro: {start}–{end} s"
            )

    for previous, current in zip(
        intervals, intervals[1:]
    ):
        if current[0] < previous[1]:
            raise ValueError(
                "Hay anotaciones de crisis solapadas."
            )

    return [
        {
            "file": filename,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "source": summary_path.name,
        }
        for start, end in intervals
    ]


def write_csv(path, rows, columns):
    """Guarda una tabla con encabezados, incluso si no tiene filas."""
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    for path in (EDF_PATH, SUMMARY_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Falta el archivo: {path.name}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_names, original_units, acquisition = (
        read_original_header(EDF_PATH)
    )

    # Solo carga metadatos, no todas las muestras.
    raw = mne.io.read_raw_edf(
        EDF_PATH,
        preload=False,
        verbose=False,
    )

    try:
        fs = float(raw.info["sfreq"])
        duration = float(raw.n_times / fs)

        if len(original_names) != len(raw.ch_names):
            raise ValueError(
                "La cantidad de canales no coincide."
            )

        name_counts = Counter(original_names)
        seen = set()
        channels = []

        # Lista operativa para reconocer nombres de derivaciones.
        # No reemplaza la confirmación del montaje en el resumen.
        electrode_names = {
            "FP1", "FP2", "F3", "F4", "F7", "F8", "FZ",
            "C3", "C4", "CZ", "T3", "T4", "T5", "T6",
            "T7", "T8", "P3", "P4", "P7", "P8", "PZ",
            "O1", "O2", "A1", "A2", "FT9", "FT10",
        }

        for index, (name, unit) in enumerate(
            zip(original_names, original_units)
        ):
            normalized = name.upper().strip()
            parts = normalized.split("-")

            dummy = normalized in {"", "-"}

            eeg_candidate = (
                len(parts) == 2
                and all(
                    part in electrode_names
                    for part in parts
                )
            )

            repeated = name in seen
            seen.add(name)

            if dummy:
                decision = "excluir"
                reason = "Canal ficticio o sin nombre"

            elif not eeg_candidate:
                decision = "revisar"
                reason = (
                    "Nombre no reconocido como "
                    "derivación EEG bipolar"
                )

            elif repeated:
                decision = "revisar"
                reason = (
                    "Nombre repetido: comprobar "
                    "equivalencia de señales"
                )

            else:
                decision = "candidato"
                reason = (
                    "Confirmar montaje y trazado "
                    "antes de seleccionar"
                )

            channels.append({
                "index": index,
                "original_name": name,
                "mne_name": raw.ch_names[index],
                "original_unit": unit,
                "duplicate_name": name_counts[name] > 1,
                "dummy": dummy,
                "eeg_candidate": eeg_candidate,
                "decision": decision,
                "reason": reason,
            })

        events = read_events(
            SUMMARY_PATH,
            EDF_PATH.name,
            duration,
        )

        channel_frequencies = sorted({
            row["fs_hz"]
            for row in acquisition
        })

        inventory = {
            "file": EDF_PATH.name,
            "fs_hz": fs,
            "header_channel_frequencies_hz": channel_frequencies,
            "n_samples": int(raw.n_times),
            "duration_s": duration,
            "n_channels": len(channels),
            "original_units": sorted(set(original_units)),
            "nominal_256_hz_matches": bool(fs == 256),
            "all_header_channels_256_hz": bool(
                all(
                    row["fs_hz"] == 256
                    for row in acquisition
                )
            ),
            "n_annotated_seizures": len(events),
            "annotated_ictal_duration_s": float(
                sum(
                    event["duration_s"]
                    for event in events
                )
            ),
            "dummy_channel_count": sum(
                row["dummy"] for row in channels
            ),
            "unrecognized_channel_count": sum(
                not row["eeg_candidate"] and not row["dummy"]
                for row in channels
            ),
            "duplicate_original_names": sorted(
                name
                for name, count in name_counts.items()
                if count > 1
            ),
            "prefilter_values": sorted({
                row["prefilter"]
                for row in acquisition
            }),
            "analysis_start_s": 2876,
            "analysis_end_s": 3156,
            "analysis_range_inside_record": bool(
                duration >= 3156
            ),
            "channel_review_complete": False,
        }

        write_csv(
            OUTPUT_DIR / "channels_audit.csv",
            channels,
            list(channels[0]),
        )

        write_csv(
            OUTPUT_DIR / "acquisition_audit.csv",
            acquisition,
            list(acquisition[0]),
        )

        write_csv(
            OUTPUT_DIR / "events.csv",
            events,
            [
                "file",
                "start_s",
                "end_s",
                "duration_s",
                "source",
            ],
        )

        (OUTPUT_DIR / "inventory.json").write_text(
            json.dumps(
                inventory,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\nAUDITORÍA DEL EEG")
        print(f"Archivo: {EDF_PATH.name}")
        print(f"Frecuencia de MNE: {fs} Hz")
        print(
            "Frecuencias por canal en la cabecera: "
            f"{channel_frequencies} Hz"
        )
        print(f"Duración: {duration} segundos")
        print(f"Canales: {len(channels)}")
        print(
            "Unidades originales: "
            f"{inventory['original_units']}"
        )
        print(f"Crisis anotadas: {len(events)}")
        print(
            "Canales ficticios identificados: "
            f"{inventory['dummy_channel_count']}"
        )
        print(
            "Nombres no reconocidos como EEG: "
            f"{inventory['unrecognized_channel_count']}"
        )

        for event in events:
            print(
                f"Crisis: {event['start_s']}–"
                f"{event['end_s']} s "
                f"({event['duration_s']} s)"
            )

        print("\nCanales:")
        for channel in channels:
            print(
                f"{channel['original_name']} → "
                f"{channel['decision']}: "
                f"{channel['reason']}"
            )

        print("\nINFORMACIÓN DE ADQUISICIÓN")
        for row in acquisition:
            prefilter = (
                row["prefilter"]
                or "No informado en la cabecera"
            )
            transducer = (
                row["transducer"]
                or "No informado en la cabecera"
            )

            print(f"\nCanal: {row['original_name']}")
            print(f"Transductor: {transducer}")
            print(
                "Límites físicos: "
                f"{row['physical_min']} a "
                f"{row['physical_max']} "
                f"{row['original_unit']}"
            )
            print(
                "Límites digitales: "
                f"{row['digital_min']} a "
                f"{row['digital_max']}"
            )
            print(f"Prefiltrado declarado: {prefilter}")
            print(
                "Paso físico por código: "
                f"{row['physical_units_per_code']:.9g} "
                f"{row['original_unit']}/código"
            )

        print("\nResultados guardados en data/derived/:")
        print("- inventory.json")
        print("- channels_audit.csv")
        print("- events.csv")
        print("- acquisition_audit.csv")

    finally:
        raw.close()


if __name__ == "__main__":
    main()