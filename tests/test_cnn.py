"""Tests for the CNN baseline on small made-up data (CPU, a few seconds).
The real SWaT runs live in experiments/cnn/."""
import numpy as np
import pytest
import torch

from ics_tsfm.data.windows import window_batches
from ics_tsfm.models.cnn import CNNDetector, ConvNet


def waves(n=3000, seed=0):
    """Two columns of slow waves plus a little noise: easy to forecast."""
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.sin(t / 15), np.cos(t / 23)])
    return (x + 0.02 * rng.normal(size=x.shape)).astype(np.float32)


def small(seed=0, **kw):
    params = dict(history=20, layers=2, filters=8, batch_size=64, max_epochs=15,
                  patience=3, seed=seed, device="cpu", verbose=False)
    params.update(kw)
    return CNNDetector(**params)


# --- the network -------------------------------------------------------------------


def test_swat_config_has_the_same_trainable_parameters_as_fungs_keras_model():
    # 8 x Conv1D(32, k=3) + BatchNorm, Flatten((200 - 16) x 32), Dense(36).
    # conv: 36*32*3+32 = 3,488, then 7 x (32*32*3+32) = 21,728; BN: 8 x 64 = 512;
    # dense: 5,888*36+36 = 212,004. Keras' summary shows 238,244 "total" because
    # it also counts BatchNorm's 512 running statistics, which are not trained.
    net = ConvNet(n_columns=36, history=200, layers=8, filters=32, kernel=3)
    assert sum(p.numel() for p in net.parameters()) == 3_488 + 21_728 + 512 + 212_004


def test_output_is_one_predicted_row_per_window():
    net = ConvNet(n_columns=36, history=200)
    assert net(torch.zeros(5, 200, 36)).shape == (5, 36)


def test_history_too_short_for_the_layers_is_rejected():
    with pytest.raises(ValueError, match="too short"):
        ConvNet(n_columns=3, history=16, layers=8, kernel=3)


# --- windows: the model only ever sees the past ----------------------------------------


def test_gathered_windows_match_window_batches_and_exclude_the_target_row():
    x = waves(400)
    det = small()
    anchors = np.array([20, 21, 150, 399])
    windows, targets = det._gather(torch.as_tensor(x), torch.as_tensor(anchors))
    (ref_w, ref_t, _), = window_batches(x, anchors, history=20)
    assert np.array_equal(windows.numpy(), ref_w)
    assert np.array_equal(targets.numpy(), ref_t)
    # the window for anchor i stops at row i - 1: row i is only the target
    assert np.array_equal(windows[2, -1].numpy(), x[149])


def test_changing_the_target_row_or_later_does_not_change_the_prediction():
    x = waves(400)
    det = small().fit(x, np.arange(20, 300), np.arange(300, 400))
    i = 350
    changed = x.copy()
    changed[i:] += 5.0
    with torch.no_grad():
        det.model_.eval()
        p1 = det.model_(det._gather(torch.as_tensor(x), torch.tensor([i]))[0])
        p2 = det.model_(det._gather(torch.as_tensor(changed), torch.tensor([i]))[0])
    assert torch.allclose(p1, p2)


# --- training ------------------------------------------------------------------------


def test_training_lowers_the_loss_and_keeps_the_best_epoch():
    x = waves()
    det = small().fit(x, np.arange(20, 2400), np.arange(2420, 3000))
    losses = [h["val_loss"] for h in det.history_]
    assert losses[det.best_epoch_ - 1] == min(losses)
    assert det.history_[-1]["train_loss"] < det.history_[0]["train_loss"]
    # the kept weights are the best epoch's: re-scoring validation reproduces its loss
    val = det.score(x, np.arange(2420, 3000)).mean()
    assert val == pytest.approx(min(losses), rel=1e-5)


def test_early_stopping_stops_before_the_epoch_limit():
    x = waves()
    det = small(max_epochs=100, patience=2).fit(x, np.arange(20, 2400), np.arange(2420, 3000))
    assert len(det.history_) < 100
    assert len(det.history_) == det.best_epoch_ + 2   # stopped 2 epochs after the best


def test_same_seed_gives_the_same_model():
    x = waves()
    tr, va = np.arange(20, 2400), np.arange(2420, 3000)
    a = small(seed=1, max_epochs=3).fit(x, tr, va).score(x, va)
    b = small(seed=1, max_epochs=3).fit(x, tr, va).score(x, va)
    c = small(seed=2, max_epochs=3).fit(x, tr, va).score(x, va)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


# --- scoring -------------------------------------------------------------------------


def test_a_sudden_jump_scores_far_above_normal_rows():
    x = waves()
    det = small().fit(x, np.arange(20, 2400), np.arange(2420, 2700))
    attacked = x.copy()
    attacked[2800:2850, 0] += 1.5                      # column 0 jumps for 50 rows
    normal = det.score(attacked, np.arange(2720, 2790)).mean()
    attack = det.score(attacked, np.arange(2800, 2810)).mean()
    assert attack > 20 * normal


def test_score_series_is_nan_where_not_scored():
    x = waves(500)
    det = small(max_epochs=2).fit(x, np.arange(20, 400), np.arange(400, 500))
    series = det.score_series(x, np.arange(100, 500))
    assert np.isnan(series[:100]).all()
    assert not np.isnan(series[100:]).any()
