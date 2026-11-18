"""CNN anomaly detector (Phase A, Step 6): training, early stopping, scoring.

How it detects anomalies
------------------------
The CNN reads the `history` rows before row i and predicts row i. Trained only
on normal data, it learns how the plant usually moves on from its recent past.
The score at row i is the squared prediction error, averaged over the columns
(as in Fung et al.); an attack makes the plant do something the model did not
expect, so the error grows.

Training follows Fung et al. (Adam, learning rate 1e-3, batch 512, up to 100
epochs, mean squared error, stop when validation loss has not improved for 3
epochs and keep the best weights), with three deliberate changes:
  - samples come from our leak-free blocked split (SwatSplit.anchors), not a
    shuffle of overlapping windows;
  - the training samples are reshuffled every epoch (Fung's batch generator
    walks them in a fixed order and skips the first batch);
  - the target is the very next row, as their paper describes; their code
    skips one row and predicts two steps ahead.

Same interface as PCADetector (history, fit, score, score_series), so the
detection rule and metrics can treat every model the same way.

Speed: the whole scaled array sits on the GPU (Normal file: 481,500 x 36
floats, about 70 MB) and each batch of windows is gathered there by index,
the same windows window_batches would cut (a test checks this).
"""
import copy
import time

import numpy as np
import torch
from torch import nn

from ics_tsfm.models.cnn.network import ConvNet


class CNNDetector:
    """Usage:

        det = CNNDetector(seed=0)
        train, val = split.anchors(det.history)
        det.fit(split.normal, train, val)
        test_scores = det.score(split.test, split.test_rows([det.history]))
    """

    def __init__(self, history: int = 200, layers: int = 8, filters: int = 32, kernel: int = 3,
                 lr: float = 1e-3, batch_size: int = 512, max_epochs: int = 100,
                 patience: int = 3, seed: int = 0, device: str | None = None, verbose: bool = True):
        self.history = history
        self.layers, self.filters, self.kernel = layers, filters, kernel
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose

    # ---------------------------------------------------------------- helpers
    def _to_device(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _gather(self, x_dev: torch.Tensor, anchors: torch.Tensor):
        """Windows rows [i - history, i) and targets row i, for each anchor i."""
        offsets = torch.arange(-self.history, 0, device=self.device)
        return x_dev[anchors[:, None] + offsets], x_dev[anchors]

    @torch.no_grad()
    def _errors(self, x_dev: torch.Tensor, anchors: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """Squared prediction error per anchor, averaged over columns."""
        self.model_.eval()
        out = np.empty(len(anchors))
        a_dev = torch.as_tensor(np.asarray(anchors), device=self.device)
        for start in range(0, len(anchors), batch_size):
            windows, targets = self._gather(x_dev, a_dev[start:start + batch_size])
            err = ((self.model_(windows) - targets) ** 2).mean(dim=1)
            out[start:start + len(err)] = err.cpu().numpy()
        return out

    # ---------------------------------------------------------------- training
    def fit(self, x: np.ndarray, train_anchors: np.ndarray, val_anchors: np.ndarray) -> "CNNDetector":
        """Train on train_anchors; early-stop on the mean validation error.

        Records per-epoch losses in `history_` (list of dicts) and the epoch
        whose weights were kept in `best_epoch_`.
        """
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.model_ = ConvNet(x.shape[1], self.history, self.layers, self.filters, self.kernel).to(self.device)
        # eps=1e-7 is Keras' Adam default (PyTorch's is 1e-8)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, eps=1e-7)
        loss_fn = nn.MSELoss()
        x_dev = self._to_device(x)
        train_anchors = np.asarray(train_anchors)

        best_loss, best_state, waited = np.inf, None, 0
        self.history_ = []
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            self.model_.train()
            order = torch.as_tensor(rng.permutation(train_anchors), device=self.device)
            total, n = 0.0, 0
            for start in range(0, len(order), self.batch_size):
                windows, targets = self._gather(x_dev, order[start:start + self.batch_size])
                loss = loss_fn(self.model_(windows), targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(targets)
                n += len(targets)
            val_loss = float(self._errors(x_dev, val_anchors).mean())
            self.history_.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                                  "seconds": time.time() - t0})
            if self.verbose:
                print(f"  epoch {epoch:3d}  train {total / n:.5f}  val {val_loss:.5f}  "
                      f"({time.time() - t0:.1f} s)")

            if val_loss < best_loss:
                best_loss, best_state, waited = val_loss, copy.deepcopy(self.model_.state_dict()), 0
                self.best_epoch_ = epoch
            else:
                waited += 1
                if waited >= self.patience:
                    break

        self.model_.load_state_dict(best_state)
        return self

    # ---------------------------------------------------------------- scoring
    def score(self, x: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """Prediction error for each anchor, in the order given."""
        return self._errors(self._to_device(x), anchors)

    def score_series(self, x: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """One score per row of `x`: the score at each anchor, NaN elsewhere."""
        series = np.full(len(x), np.nan)
        series[anchors] = self.score(x, anchors)
        return series

    # ---------------------------------------------------------------- saving
    def save(self, path) -> None:
        torch.save({"state_dict": self.model_.state_dict(), "config": self.config(),
                    "history": self.history_, "best_epoch": self.best_epoch_}, path)

    @classmethod
    def load(cls, path, n_columns: int, device: str | None = None) -> "CNNDetector":
        """Rebuild a trained detector saved with `save`."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        det = cls(**ckpt["config"], device=device, verbose=False)
        det.model_ = ConvNet(n_columns, det.history, det.layers, det.filters, det.kernel).to(det.device)
        det.model_.load_state_dict(ckpt["state_dict"])
        det.history_, det.best_epoch_ = ckpt["history"], ckpt["best_epoch"]
        return det

    def config(self) -> dict:
        return {"history": self.history, "layers": self.layers, "filters": self.filters,
                "kernel": self.kernel, "lr": self.lr, "batch_size": self.batch_size,
                "max_epochs": self.max_epochs, "patience": self.patience, "seed": self.seed}
