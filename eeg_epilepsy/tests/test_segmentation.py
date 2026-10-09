from eeg_epilepsy.segmentation import interval_overlap, label_epoch, make_epochs


def test_interval_overlap_does_not_double_count():
    assert interval_overlap(0, 10, [(2, 4), (6, 8)]) == 4


def test_partial_epoch_is_transition():
    state, overlap = label_epoch(9, 13, [(10, 12)])
    assert state == "transicion" and overlap == 2


def test_full_epoch_is_ictal():
    assert label_epoch(10, 12, [(10, 12)]) == ("ictal", 2)


def test_no_overlap_is_nonictal():
    assert label_epoch(0, 2, [(10, 12)]) == ("sin_crisis_anotada", 0)


def test_half_step_epochs():
    epochs = make_epochs(8 * 256, 256, 4, [])
    assert len(epochs) == 3 and epochs[0].start_s == 0 and epochs[1].start_s == 2
