import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from edf_writer import write_edf  # noqa: E402

FS = 256
SEIZURE = (60.0, 80.0)
WINDOW = (30.0, 100.0)


@pytest.fixture(scope="session")
def synthetic_record(tmp_path_factory):
    """EDF sintético de 120 s con la estructura de CHB-MIT: nombre T8-P8 repetido, P7-T7 = −T7-P7,
    una crisis anotada (60–80 s), un tramo plano, un pico extremo y un tramo en el riel del ADC."""
    d = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    t = np.arange(120 * FS) / FS
    seiz = (t >= SEIZURE[0]) & (t < SEIZURE[1])

    def base(i):
        x = 8 * rng.standard_normal(t.size) + 10 * np.sin(2 * np.pi * 10 * t + i)
        x[seiz] += 60 * np.sin(2 * np.pi * 5 * t[seiz])  # ritmo lento de gran amplitud
        return x

    fp1f7, f7t7, t7p7, p7o1, t8p8 = (base(i) for i in range(5))
    f7t7[int(40 * FS):int(44 * FS)] = 0.0                 # tramo plano
    fp1f7[int(50 * FS)] = 700.0                             # pico extremo (> 600 µV)
    p7o1[int(90 * FS):int(91 * FS)] = 800.0                 # en el riel del digitalizador
    signals = [("FP1-F7", fp1f7), ("F7-T7", f7t7), ("T7-P7", t7p7), ("P7-O1", p7o1),
               ("T8-P8", t8p8), ("T8-P8", t8p8.copy()), ("P7-T7", -t7p7)]
    edf = d / "chb99_03.edf"
    write_edf(edf, signals, fs=FS)
    summary = d / "chb99-summary.txt"
    summary.write_text(
        "File Name: chb99_03.edf\nNumber of Seizures in File: 1\n"
        f"Seizure Start Time: {int(SEIZURE[0])} seconds\nSeizure End Time: {int(SEIZURE[1])} seconds\n")
    selection = d / "channel_selection.json"
    selection.write_text(json.dumps({
        "file": "chb99_03.edf",
        "excluded_channels": [{"name": "T8-P8-1", "reason": "duplicado exacto"},
                              {"name": "P7-T7", "reason": "redundante con T7-P7"}],
        "provisional_review_channels": ["FP1-F7", "F7-T7"],
        "quality_review_complete": False}))
    return {"edf": edf, "summary": summary, "selection": selection, "dir": d}
