from eeg_epilepsy.io import get_channel_uv, read_edf


def test_read_edf_is_lazy_and_converts_to_uv_once(synthetic_record):
    raw, meta = read_edf(synthetic_record["edf"])
    try:
        assert not raw.preload and meta["fs"] == 256 and meta["duration_s"] == 120
        assert "T8-P8-0" in raw.ch_names and "T8-P8-1" in raw.ch_names  # MNE añade sufijos
        x = get_channel_uv(raw, "FP1-F7")
        assert 10 < x.std() < 100 and abs(x).max() < 1000  # µV (no voltios ≈1e-5, ni mV)
    finally:
        raw.close()
