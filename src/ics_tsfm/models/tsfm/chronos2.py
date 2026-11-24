"""Chronos-2 as a Forecaster for ForecastDetector (Phase B, Step B3).

Which checkpoint
----------------
The Chronos-2 that Lowe et al. (ICML 2026 FMSD workshop) evaluate. The paper
names no size or checkpoint; "Chronos-2" is Amazon's 120M flagship (the only
other sizes are named chronos-2-small and chronos-2-synth). Its weights were
uploaded 2025-10-30 and never changed (later commits touch only the README),
so the paper used these same weights; autogluon/chronos-2 is a byte-identical
mirror. Loading at a fixed commit, and checking the weight hash in
experiments/check_env.py, keeps every machine on them.

What the model does with a context (read from chronos-forecasting 2.3.2)
------------------------------------------------------------------------
- Each series is first standardised with ITS OWN context's mean and spread
  (instance normalisation); a perfectly flat sensor gets spread 1e-5 instead
  of 0. So the model re-centres on wherever the recent window sits: slow
  drift is absorbed, and so is an attacker's held value once it fills the
  context. Our own StandardScaler only decides how the sensors are weighed
  when their errors are averaged into one score.
- The context is cut into patches of 16 rows; context 128 is 8 patches,
  context 4096 is 256.
- It always predicts whole patches of 16 rows and cuts them to the horizon
  asked for, so horizon 1 costs the same as horizon 16.
- Each row of the input array is its own group: group attention mixes the
  variates of one row and never mixes rows. So ForecastDetector's joint mode
  (one row = all sensors at one origin) lets the model read across sensors,
  and per-sensor mode (one row = one sensor) does not.
- The forecast is a set of quantiles; the point forecast used here is the
  median (quantile 0.5).
"""
import numpy as np

MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
WEIGHTS_SHA256 = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"


class Chronos2Forecaster:
    """Usage:

        det = ForecastDetector(Chronos2Forecaster(), context=128, horizon=1, joint=True)

    batch_size counts SERIES per forward pass, the way the Chronos-2 pipeline
    does: with 36 sensors in joint mode, 256 means about 8 origins per pass.
    Bigger is not always faster (measured on the laptop: 144 beat 256).

    dtype is the precision the network computes in. The input scaling and the
    returned forecasts stay float32 either way (the pipeline does this). On
    the laptop bfloat16 moved forecasts by 0.0005 on average; on the lab's
    RTX 5080 half precision runs about 3x faster than float32.
    """

    def __init__(self, device: str | None = None, batch_size: int = 256,
                 dtype: str = "float32", model_id: str = MODEL_ID, revision: str = REVISION):
        import torch
        from chronos import Chronos2Pipeline

        self.model_id, self.revision, self.batch_size, self.dtype = model_id, revision, batch_size, dtype
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = Chronos2Pipeline.from_pretrained(model_id, revision=revision, device_map=device)
        self.pipeline.model.to(getattr(torch, dtype))
        self._median = self.pipeline.quantiles.index(0.5)

    @property
    def max_context(self) -> int:
        """Longest context the model reads (8,192 rows for Chronos-2)."""
        return self.pipeline.model_context_length

    def predict(self, contexts: np.ndarray, horizon: int) -> np.ndarray:
        if contexts.ndim != 3:
            raise ValueError(f"contexts must be (n_series, n_variates, context), got {contexts.shape}")
        if contexts.shape[-1] > self.max_context:
            # the pipeline would silently keep only the last max_context rows
            raise ValueError(f"context {contexts.shape[-1]} is longer than Chronos-2 reads ({self.max_context})")
        forecasts = self.pipeline.predict(np.ascontiguousarray(contexts, np.float32),
                                          prediction_length=horizon, batch_size=self.batch_size,
                                          context_length=contexts.shape[-1])
        # one (n_variates, n_quantiles, horizon) tensor per series -> median
        return np.stack([f[:, self._median].float().cpu().numpy() for f in forecasts])

    def config(self) -> dict:
        return {"forecaster": "chronos-2", "model_id": self.model_id, "revision": self.revision,
                "dtype": self.dtype}
