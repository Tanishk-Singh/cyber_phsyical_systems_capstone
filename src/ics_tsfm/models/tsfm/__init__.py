"""Time-series foundation models as forecast-error detectors (Phase B).

forecast.py holds the model-independent part (origins, horizon, joint vs
per-sensor, chunked resumable runs); each model gets a small adapter that
implements the Forecaster protocol.
"""
from ics_tsfm.models.tsfm.forecast import ForecastDetector, Forecaster, LastValueForecaster, origins

__all__ = ["ForecastDetector", "Forecaster", "LastValueForecaster", "origins"]
