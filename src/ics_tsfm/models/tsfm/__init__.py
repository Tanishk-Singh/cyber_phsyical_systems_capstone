"""Time-series foundation models as forecast-error detectors (Phase B).

forecast.py holds the model-independent part (origins, horizon, joint vs
per-sensor, chunked resumable runs); fallback.py adds Forecast Fallback on
top of it; each model gets a small adapter that implements the Forecaster
protocol.
"""
from ics_tsfm.models.tsfm.fallback import (FF_PERCENTILE, FF_TRIGGER, ForecastFallbackDetector,
                                           fallback_thresholds, spread_blocks)
from ics_tsfm.models.tsfm.forecast import (ForecastDetector, Forecaster, LastValueForecaster,
                                           TimeLimitReached, origins)

__all__ = ["FF_PERCENTILE", "FF_TRIGGER", "ForecastDetector", "ForecastFallbackDetector", "Forecaster",
           "LastValueForecaster", "TimeLimitReached", "fallback_thresholds", "origins", "spread_blocks"]
