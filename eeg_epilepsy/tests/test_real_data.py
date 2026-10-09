"""Pruebas sobre el EDF real de CHB-MIT; se omiten si data/raw/chb01_03.edf no está descargado."""
from pathlib import Path

import pytest

from eeg_epilepsy.analysis import ROOT, analyze_file, load_config, sha256_file
import json

EDF = ROOT / "data" / "raw" / "chb01_03.edf"
pytestmark = pytest.mark.skipif(not EDF.exists(), reason="EDF real no descargado")


@pytest.fixture(scope="module")
def real():
    cfg = load_config()
    return analyze_file(EDF, ROOT / "data/raw/chb01-summary.txt", durations_s=(4,), config=cfg,
                        channel_selection=ROOT / "data/derived/channel_selection.json",
                        manifest_path=ROOT / "data/raw/download_manifest.json")


def test_real_exclusions_and_window(real):
    m, meta, _ = real
    assert len(meta["selected_channels"]) == 21
    assert not {"T8-P8-1", "P7-T7"} & set(m.channel)
    assert {"T8-P8-0", "T7-P7"} <= set(m.channel)
    assert meta["analysis_window_s"] == [2876.0, 3156.0] and meta["annotated_intervals_s"] == [(2996.0, 3036.0)]
    assert m.start_s.min() == 2876 and m.end_s.max() == 3156


def test_real_epoch_counts_and_units(real):
    m, meta, _ = real
    d = m[m.channel == "FP1-F7"]
    assert (d.state == "ictal").sum() == 19 and (d.state == "transicion").sum() == 2
    assert 5 < m.rms_uv.median() < 200  # µV plausibles de EEG de superficie (no voltios ni mV)
    assert meta["welch_resolution_hz"] == 0.5 and meta["fs"] == 256


def test_real_traceability_matches_manifest(real):
    _, meta, _ = real
    assert meta["traceability"]["edf_download"]["sha256_matches"] is True
    assert meta["traceability"]["edf_sha256"] == sha256_file(EDF)


def test_real_acquisition_limits_and_no_saturation(real):
    m, meta, _ = real
    lim = meta["acquisition_limits"]["FP1-F7"]
    assert (lim["physical_min_uv"], lim["physical_max_uv"]) == (-800.0, 800.0)
    assert not m.quality_saturation.any()
