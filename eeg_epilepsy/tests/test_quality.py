import numpy as np

from eeg_epilepsy.quality import epoch_quality


def test_clean_signal_has_no_flags():
    x = 20 * np.random.default_rng(0).standard_normal(1024)
    q = epoch_quality(x, rail_limits_uv=(-800, 799.6), rail_tolerance_uv=0.39)
    assert not (q["quality_flat"] or q["quality_extreme"] or q["quality_saturation"])
    assert q["quality_nan_fraction"] == 0 and q["quality_rail_fraction"] == 0


def test_flat_and_constant():
    assert epoch_quality(np.zeros(512))["quality_flat"]


def test_extreme_threshold_is_configurable_and_not_saturation():
    x = np.zeros(512)
    x[10] = 700.0
    assert epoch_quality(x, extreme_threshold_uv=600)["quality_extreme"]
    q = epoch_quality(x, extreme_threshold_uv=600, rail_limits_uv=(-800, 799.6), rail_tolerance_uv=0.39)
    assert not q["quality_saturation"]  # amplitud grande ≠ saturación
    assert not epoch_quality(x, extreme_threshold_uv=1000)["quality_extreme"]


def test_saturation_needs_samples_at_the_rail():
    x = np.zeros(512)
    x[:3] = 799.6
    q = epoch_quality(x, rail_limits_uv=(-800, 799.6), rail_tolerance_uv=0.39)
    assert q["quality_saturation"] and q["quality_rail_fraction"] > 0
    assert epoch_quality(x)["quality_saturation"] is False  # sin límites no se evalúa
    assert np.isnan(epoch_quality(x)["quality_rail_fraction"])


def test_non_finite_values():
    x = np.ones(100)
    x[:10] = np.nan
    q = epoch_quality(x * 5)
    assert q["quality_nan_fraction"] == 0.1 and q["quality_valid_fraction"] == 0.9
    assert epoch_quality(np.full(10, np.nan))["quality_flat"]
