import json

import numpy as np
import pandas as pd
import pytest

from eeg_epilepsy.analysis import (
    _manifest_entry, _union_length, analyze_file, binary_subset, descriptive_summary, export_results,
    load_config, psd_by_state, run_analysis, sha256_file,
)
from conftest import SEIZURE, WINDOW


# ------------------------------- resúmenes (sin EDF) -----------------------------------
def toy_metrics():
    rows = []
    for state, vals, flags in [
        ("sin_crisis_anotada", [1, 2, 3, 4, 100], [0, 0, 0, 0, 1]),
        ("ictal", [10, 20, 30], [0, 0, 0]),
        ("transicion", [5, 6], [0, 0]),
    ]:
        for i, (v, fl) in enumerate(zip(vals, flags)):
            rows.append({
                "channel": "A", "window_s": 4, "state": state, "epoch_index": i,
                "rms_uv": float(v), "ll_uv_per_sample": 1.0, "power_uv2": 1.0, "entropy": 0.5,
                "alpha_uv2": 1.0, "alpha_rel": 0.5, "quality_flag": bool(fl),
            })
    return pd.DataFrame(rows)


def row(df, state):
    return df[df.state == state].iloc[0]


def test_summary_median_iqr_n_and_flag_percent():
    s = descriptive_summary(toy_metrics())
    r = row(s, "sin_crisis_anotada")
    assert r["rms_uv_median"] == 3.0 and r["rms_uv_iqr"] == 2.0
    assert r["n_epochs"] == 5 and r["n_flagged"] == 1 and r["quality_flag_percent"] == pytest.approx(20.0)
    assert "alpha_rel_median" in s.columns


def test_summary_excluding_flagged_changes_result():
    r = row(descriptive_summary(toy_metrics(), exclude_flagged=True), "sin_crisis_anotada")
    assert r["rms_uv_median"] == 2.5 and r["n_epochs"] == 4 and r["n_epochs_total"] == 5
    assert r["quality_flag_percent"] == pytest.approx(20.0)


def test_summary_flag_rebuilt_from_quality_columns():
    df = toy_metrics().drop(columns="quality_flag")
    df["quality_flat"], df["quality_saturation"], df["quality_nan_fraction"] = False, False, 0.0
    df["quality_extreme"] = df["rms_uv"] > 50
    assert row(descriptive_summary(df), "sin_crisis_anotada")["n_flagged"] == 1


def test_binary_subset_excludes_transition():
    assert set(binary_subset(toy_metrics()).state) == {"ictal", "sin_crisis_anotada"}
    assert len(binary_subset(toy_metrics(), exclude_flagged=True)) == 7


def test_union_length_does_not_double_count():
    assert _union_length([(0, 10), (5, 12)]) == 12 and _union_length([(0, 2), (6, 8)]) == 4


def test_load_config_matches_json_and_overrides(tmp_path):
    cfg = load_config()
    assert cfg["welch"]["segment_s"] == 2.0 and cfg["analysis"]["window_s"] == [2876, 3156]
    assert cfg["quality"]["extreme_threshold_uv"] == 600.0
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"quality": {"flat_threshold_uv": 0.5}}))
    cfg2 = load_config(p)
    assert cfg2["quality"]["flat_threshold_uv"] == 0.5 and cfg2["filter"]["high_hz"] == 40.0


def test_manifest_entry_checks_sha(tmp_path):
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"a.edf": {"url": "u", "downloaded_at_utc": "d", "sha256": "abc"}}))
    assert _manifest_entry(m, "a.edf", "abc")["sha256_matches"] is True
    assert _manifest_entry(m, "a.edf", "zzz")["sha256_matches"] is False
    assert _manifest_entry(m, "otro.edf", "abc") == {"found": False}
    assert _manifest_entry(None, "a.edf", "abc") == {"found": False}


# ----------------------------- integración con EDF sintético ---------------------------
@pytest.fixture(scope="module")
def result(synthetic_record):
    cfg = load_config()
    cfg["analysis"]["window_s"] = list(WINDOW)
    return analyze_file(synthetic_record["edf"], synthetic_record["summary"], config=cfg,
                        channel_selection=synthetic_record["selection"])


def test_exclusions_are_applied(result):
    m, meta, _ = result
    assert set(m.channel) == {"FP1-F7", "F7-T7", "T7-P7", "P7-O1", "T8-P8-0"}
    assert "T8-P8-1" not in set(m.channel) and "P7-T7" not in set(m.channel)
    assert meta["channel_selection"]["excluded_applied"] == ["P7-T7", "T8-P8-1"]
    assert meta["review_channels"] == ["FP1-F7", "F7-T7"]
    assert meta["channel_selection"]["quality_review_complete"] is False


def test_metrics_limited_to_analysis_window_and_time_origin_kept(result):
    m, meta, _ = result
    assert m.start_s.min() >= WINDOW[0] and m.end_s.max() <= WINDOW[1]
    assert meta["analysis_window_s"] == list(WINDOW)
    assert meta["annotated_ictal_fraction_of_window"] == pytest.approx(20 / 70)
    assert meta["annotated_ictal_time_s"] == 20


def test_states_and_ictal_fraction(result):
    m, _, _ = result
    d4 = m[(m.channel == "FP1-F7") & (m.window_s == 4)]
    assert (d4.state == "ictal").sum() == 9  # inicios 60, 62, ..., 76
    assert (d4.state == "transicion").sum() == 2
    assert d4.loc[d4.state == "ictal", "ictal_fraction"].eq(1).all()
    assert d4.loc[d4.state == "sin_crisis_anotada", "ictal_fraction"].eq(0).all()
    t = d4[d4.state == "transicion"]
    assert ((t.ictal_fraction > 0) & (t.ictal_fraction < 1)).all()
    assert set(m.window_s) == {2, 4, 8}


def test_metrics_reflect_ictal_rhythm(result):
    m, _, _ = result
    d = m[(m.channel == "T7-P7") & (m.window_s == 4)]
    assert d[d.state == "ictal"].rms_uv.median() > 2 * d[d.state == "sin_crisis_anotada"].rms_uv.median()
    assert m[["rms_uv", "entropy", "power_uv2"]].notna().all().all()


def test_quality_flags(result):
    m, meta, _ = result
    f7 = m[m.channel == "F7-T7"]
    assert f7[f7.quality_flat].start_s.between(39, 44).all() and f7.quality_flat.any()
    fp1 = m[(m.channel == "FP1-F7") & (m.window_s == 4)]
    assert fp1.loc[fp1.quality_extreme, ["start_s", "end_s"]].apply(lambda r: r.start_s <= 50 < r.end_s, axis=1).all()
    assert fp1.quality_extreme.any() and not fp1.quality_saturation.any()  # 700 µV: extremo, pero no en el riel
    p7 = m[m.channel == "P7-O1"]
    assert p7.quality_saturation.any() and not m[m.channel == "T7-P7"].quality_saturation.any()
    assert (m.quality_flag == (m.quality_flat | m.quality_extreme | m.quality_saturation)).all()
    assert meta["acquisition_limits"]["FP1-F7"]["physical_max_uv"] == 800.0


def test_traceability_and_welch(result, synthetic_record):
    _, meta, _ = result
    assert meta["traceability"]["edf_sha256"] == sha256_file(synthetic_record["edf"])
    assert meta["welch_resolution_hz"] == pytest.approx(0.5)
    assert meta["config"]["filter"]["low_hz"] == 0.5


def test_spectra_range_and_psd_by_state(result):
    m, _, spectra = result
    k = f"FP1-F7|4|{int(m[(m.channel == 'FP1-F7') & (m.window_s == 4)].epoch_index.iloc[0])}"
    assert spectra[k]["freq_hz"][0] == 0.5 and spectra[k]["freq_hz"][-1] == 40.0
    assert not psd_by_state(m, spectra).empty


def test_export_roundtrip(result, tmp_path):
    m, meta, spectra = result
    paths = export_results(m, meta, spectra, tmp_path)
    assert all(p.exists() for p in paths.values())
    saved = json.loads(paths["config"].read_text(encoding="utf-8"))
    assert saved["metadata"]["config"]["quality"]["extreme_threshold_uv"] == 600.0
    assert pd.read_csv(paths["metricas"]).shape[0] == len(m)


def test_config_changes_flow_through(synthetic_record):
    cfg = load_config()
    cfg["quality"]["flat_threshold_uv"] = 1e6
    cfg["analysis"]["window_s"] = list(WINDOW)
    m, _, _ = analyze_file(synthetic_record["edf"], synthetic_record["summary"], durations_s=(4,), config=cfg,
                           channel_selection=synthetic_record["selection"])
    assert m.quality_flat.all() and set(m.window_s) == {4}


def test_full_record_when_window_is_none(synthetic_record):
    m, meta, _ = analyze_file(synthetic_record["edf"], synthetic_record["summary"], durations_s=(4,),
                              window_s=None, channel_selection=synthetic_record["selection"])
    assert m.start_s.min() == 0 and m.end_s.max() == 120 and meta["analysis_window_s"] == [0.0, 120.0]


def test_rejects_wrong_selection_file_and_bad_window(synthetic_record, tmp_path):
    bad = tmp_path / "sel.json"
    bad.write_text(json.dumps({"file": "chb01_03.edf", "excluded_channels": []}))
    with pytest.raises(ValueError, match="corresponde a"):
        analyze_file(synthetic_record["edf"], synthetic_record["summary"], channel_selection=bad)
    with pytest.raises(ValueError, match="no cabe"):
        analyze_file(synthetic_record["edf"], synthetic_record["summary"], window_s=(0, 500))


def test_rejects_annotation_beyond_record(synthetic_record, tmp_path):
    bad = tmp_path / "s.txt"
    bad.write_text("File Name: chb99_03.edf\nNumber of Seizures in File: 1\n"
                   "Seizure Start Time: 100 seconds\nSeizure End Time: 500 seconds\n")
    with pytest.raises(ValueError, match="fuera del registro"):
        analyze_file(synthetic_record["edf"], bad)


def test_run_analysis_from_config(synthetic_record, tmp_path):
    """run_analysis resuelve rutas y selección desde la configuración y exporta."""
    d = synthetic_record["dir"]
    cfg = {"data": {"edf": str(synthetic_record["edf"]), "summary": str(synthetic_record["summary"]),
                    "manifest": str(d / "no_existe.json"), "derived_dir": str(tmp_path / "out")},
           "channels": {"selection_file": str(synthetic_record["selection"])},
           "analysis": {"window_s": list(WINDOW)}}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    m, meta, _ = run_analysis(cfg_path, root=tmp_path)
    assert (tmp_path / "out" / "metricas.csv").exists() and len(meta["selected_channels"]) == 5


def test_duplicates_left_after_selection_abort(synthetic_record, tmp_path):
    """Si el contrato de canales no excluye el duplicado, el pipeline no debe duplicar métricas."""
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"file": "chb99_03.edf", "excluded_channels": [{"name": "P7-T7", "reason": "x"}]}))
    with pytest.raises(ValueError, match="siguen duplicados"):
        analyze_file(synthetic_record["edf"], synthetic_record["summary"], durations_s=(4,), channel_selection=sel)


def test_without_selection_file_only_suffix_duplicates_are_dropped(synthetic_record):
    cfg = load_config()
    cfg["analysis"]["window_s"] = list(WINDOW)
    m, meta, _ = analyze_file(synthetic_record["edf"], synthetic_record["summary"], durations_s=(4,), config=cfg)
    assert "T8-P8-1" not in set(m.channel) and "P7-T7" in set(m.channel)
    assert meta["channel_selection"]["excluded_applied"] == []


def test_excluded_channel_missing_in_edf_is_an_error(synthetic_record, tmp_path):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"file": "chb99_03.edf", "excluded_channels": [{"name": "NO-EXISTE", "reason": "x"}]}))
    with pytest.raises(ValueError, match="no encontrados"):
        analyze_file(synthetic_record["edf"], synthetic_record["summary"], channel_selection=sel)
