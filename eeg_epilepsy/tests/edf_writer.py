"""Escritor mínimo de EDF (16 bits) para pruebas sintéticas sin dependencias adicionales."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _f(text, width):
    return str(text).encode("ascii")[:width].ljust(width)


def write_edf(path, signals: dict[str, np.ndarray] | list[tuple[str, np.ndarray]], fs=256,
              pmin=-800.0, pmax=800.0, dmin=-2048, dmax=2047, unit="uV"):
    items = list(signals.items()) if isinstance(signals, dict) else list(signals)
    n = len(items)
    n_rec = len(items[0][1]) // fs
    head = (_f("0", 8) + _f("test", 80) + _f("test", 80) + _f("01.01.01", 8) + _f("00.00.00", 8)
            + _f(256 + 256 * n, 8) + _f("", 44) + _f(n_rec, 8) + _f("1", 8) + _f(n, 4))
    sig = b""
    for field, width in [(None, 16), ("", 80), (unit, 8), (pmin, 8), (pmax, 8), (dmin, 8), (dmax, 8),
                         ("", 80), (fs, 8), ("", 32)]:
        for name, _ in items:
            sig += _f(name if field is None else field, width)
    digital = []
    for _, x in items:
        d = np.round((np.asarray(x, float) - pmin) / (pmax - pmin) * (dmax - dmin) + dmin)
        digital.append(np.clip(d, dmin, dmax).astype("<i2"))
    data = bytearray()
    for r in range(n_rec):
        for d in digital:
            data += d[r * fs:(r + 1) * fs].tobytes()
    Path(path).write_bytes(head + sig + bytes(data))
