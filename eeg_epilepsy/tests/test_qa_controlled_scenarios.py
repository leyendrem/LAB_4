"""Escenarios QA controlados para riesgos de tiempo, estados y parámetros."""

import numpy as np
import pytest

from eeg_epilepsy.metrics import epoch_features
from eeg_epilepsy.quality import epoch_quality
from eeg_epilepsy.segmentation import interval_overlap, label_epoch, make_epochs


def test_overlapping_annotations_are_counted_once():
    """Evita inflar la fracción ictal si dos intervalos se superponen."""
    assert interval_overlap(0, 10, [(2, 6), (4, 8)]) == pytest.approx(6.0)
    state, overlap = label_epoch(0, 10, [(2, 6), (4, 8)])
    assert state == "transicion"
    assert overlap == pytest.approx(6.0)


@pytest.mark.parametrize("args", [
    (100, 0.0, 2.0, [], 0.5),
    (100, np.nan, 2.0, [], 0.5),
    (100, 100.0, 0.0, [], 0.5),
    (100, 100.0, 2.0, [], 0.0),
    (100, 100.0, 2.0, [], 1.1),
])
def test_epoch_generation_rejects_invalid_parameters(args):
    """No se permite que parámetros inválidos creen una rejilla engañosa."""
    with pytest.raises(ValueError):
        make_epochs(*args)


def test_quality_flags_do_not_delete_the_epoch():
    """Una señal plana queda marcada y conserva una salida auditable."""
    result = epoch_quality(np.zeros(512), flat_threshold_uv=0.1)
    assert result["quality_flat"] is True
    assert result["quality_valid_fraction"] == 1.0


def test_quality_distinguishes_extreme_amplitude_from_saturation():
    """Amplitud extrema no equivale a saturación si no toca los rieles."""
    result = epoch_quality(
        np.array([0.0, 650.0, -650.0]),
        extreme_threshold_uv=600.0,
        rail_limits_uv=(-800.0, 800.0),
    )
    assert result["quality_extreme"] is True
    assert result["quality_saturation"] is False


def test_constant_epoch_returns_missing_normalized_features():
    """Una señal sin masa espectral no se presenta como potencia relativa cero."""
    features, _ = epoch_features(np.zeros(512), fs=256.0, segment_s=2.0)
    assert np.isnan(features["entropy"])
    assert np.isnan(features["alpha_rel"])
