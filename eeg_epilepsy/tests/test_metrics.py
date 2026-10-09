import numpy as np
import pytest

from eeg_epilepsy.metrics import (
    BANDS, centered_rms, epoch_features, line_length, nan_features, psd_features, spectral_entropy,
)

FS = 256


def sine(freq=10.0, amp=10.0, seconds=4.0):
    t = np.arange(0, seconds, 1 / FS)
    return amp * np.sin(2 * np.pi * freq * t)


def test_centered_rms():
    assert centered_rms(np.array([1., 2., 3., 4.])) == pytest.approx(np.sqrt(1.25))


def test_line_length_constant_zero():
    assert line_length(np.ones(100)) == 0


def test_entropy_flat_spectrum_is_one():
    f = np.linspace(0.5, 40, 80)
    assert spectral_entropy(f, np.ones_like(f)) == pytest.approx(1.0)


# --- Guía §7.2, prueba 1: seno de 10 Hz -------------------------------------------------
def test_sine_10hz_peak_and_alpha_dominance():
    out = psd_features(sine(10), FS)
    f, p = out["psd_freq_hz"], out["psd_uv2_per_hz"]
    assert abs(f[np.argmax(p)] - 10.0) <= 0.5  # tolerancia = 1 bin (resolución 0.5 Hz)
    rel = {b: out[f"{b}_rel"] for b in BANDS}
    assert max(rel, key=rel.get) == "alpha"
    assert rel["alpha"] > 0.9  # fuga espectral permitida < 10 %


def test_welch_resolution_is_half_hz():
    f = psd_features(sine(10), FS)["psd_freq_hz"]
    assert np.diff(f)[0] == pytest.approx(0.5)


def test_relative_bands_sum_to_one():
    rng = np.random.default_rng(1)
    out = psd_features(rng.standard_normal(4 * FS), FS)
    assert sum(out[f"{b}_rel"] for b in BANDS) == pytest.approx(1.0, abs=1e-9)
    assert sum(out[f"{b}_uv2"] for b in BANDS) == pytest.approx(out["power_uv2"], rel=1e-9)


# --- Guía §7.2, prueba 2: duplicar amplitud ---------------------------------------------
def test_doubling_amplitude_scaling():
    x = sine(10) + sine(5, 4)
    a, _ = epoch_features(x, FS)
    b, _ = epoch_features(2 * x, FS)
    assert b["rms_uv"] == pytest.approx(2 * a["rms_uv"])
    assert b["ll_uv_per_sample"] == pytest.approx(2 * a["ll_uv_per_sample"])
    assert b["power_uv2"] == pytest.approx(4 * a["power_uv2"])
    assert b["alpha_uv2"] == pytest.approx(4 * a["alpha_uv2"])
    assert b["alpha_rel"] == pytest.approx(a["alpha_rel"])
    assert b["entropy"] == pytest.approx(a["entropy"])


# --- Guía §7.2, prueba 3: señal constante -----------------------------------------------
@pytest.mark.parametrize("level", [0.0, 5.3, 123.456])
def test_constant_signal_gives_missing_normalized_metrics(level):
    feats, _ = epoch_features(np.full(4 * FS, level), FS)
    assert feats["rms_uv"] == pytest.approx(0.0, abs=1e-9)
    assert feats["ll_uv_per_sample"] == 0.0
    assert np.isnan(feats["entropy"])  # NaN, no 0
    assert all(np.isnan(feats[f"{b}_rel"]) for b in BANDS)


# --- Guía §7.2, prueba 5: V → µV ---------------------------------------------------------
def test_unit_scaling_1e6_amplitude_1e12_power():
    x_uv = sine(10) + sine(20, 3)
    a, _ = epoch_features(x_uv, FS)
    v, _ = epoch_features(x_uv * 1e-6, FS)  # el mismo EEG expresado en voltios
    assert a["rms_uv"] == pytest.approx(v["rms_uv"] * 1e6)
    assert a["ll_uv_per_sample"] == pytest.approx(v["ll_uv_per_sample"] * 1e6)
    assert a["power_uv2"] == pytest.approx(v["power_uv2"] * 1e12)
    assert a["alpha_rel"] == pytest.approx(v["alpha_rel"])


def test_short_epoch_raises():
    with pytest.raises(ValueError):
        epoch_features(np.zeros(FS), FS)  # 1 s < segmento Welch de 2 s


def test_two_second_epoch_is_valid_single_segment():
    feats, spec = epoch_features(sine(10, seconds=2.0), FS)
    assert np.isfinite(feats["power_uv2"])
    assert len(spec["freq_hz"]) == FS + 1  # nperseg=512 → 257 bins


def test_nan_features_has_same_keys_as_epoch_features():
    feats, _ = epoch_features(sine(10), FS)
    assert set(nan_features()) == set(feats)


def test_custom_bands_are_respected():
    bands = {"low": (0.5, 8.0), "high": (8.0, 30.0)}
    out = psd_features(sine(10), FS, bands=bands)
    assert "low_uv2" in out and "high_rel" in out and "alpha_uv2" not in out
