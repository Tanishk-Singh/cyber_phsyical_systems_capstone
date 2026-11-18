"""CNN baseline: a PyTorch port of Fung et al.'s ConvNN (see network.py and detector.py)."""
from ics_tsfm.models.cnn.detector import CNNDetector
from ics_tsfm.models.cnn.network import ConvNet

__all__ = ["CNNDetector", "ConvNet"]
