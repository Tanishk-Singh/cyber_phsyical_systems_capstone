"""The CNN itself: a PyTorch port of Fung et al.'s Keras ConvNN.

Reference: github.com/pwwl/ics-anomaly-detection, detector/cnn.py (Apache 2.0).
Their model, layer by layer:

    input (history, n_columns)
    [Conv1D(filters, kernel 3, ReLU, no padding) -> BatchNormalization] x layers
    Flatten
    Dense(n_columns)            the predicted next row

With 8 layers, 32 filters and history 200 (Kravchik & Shabtai's SWaT setting,
which Fung uses for SWaT), each convolution trims kernel - 1 = 2 time steps,
so the flattened output is (200 - 16) x 32 = 5,888 numbers.

Keras defaults are copied where PyTorch's differ, so the port trains like the
original: Glorot-uniform weights with zero biases, and BatchNorm with
epsilon 1e-3 and Keras momentum 0.99 (= PyTorch momentum 0.01).
"""
import torch
from torch import nn


class ConvNet(nn.Module):
    """Maps a window (batch, history, n_columns) to a predicted row (batch, n_columns)."""

    def __init__(self, n_columns: int, history: int = 200, layers: int = 8,
                 filters: int = 32, kernel: int = 3):
        super().__init__()
        out_len = history - layers * (kernel - 1)
        if out_len < 1:
            raise ValueError(f"history {history} is too short for {layers} layers of kernel {kernel}")
        blocks = []
        in_ch = n_columns
        for _ in range(layers):
            blocks += [nn.Conv1d(in_ch, filters, kernel), nn.ReLU(),
                       nn.BatchNorm1d(filters, eps=1e-3, momentum=0.01)]
            in_ch = filters
        self.convs = nn.Sequential(*blocks)
        self.head = nn.Linear(out_len * filters, n_columns)
        self._keras_init()

    def _keras_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)    # Keras' glorot_uniform
                nn.init.zeros_(m.bias)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # PyTorch convolves over the last axis, so time goes last: (batch, columns, time)
        h = self.convs(windows.transpose(1, 2))
        return self.head(h.flatten(1))
