
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    channels: int
    duration: float
    subtype: str


@dataclass
class ProcessedAudio:
    samples: np.ndarray
    sample_rate: int
